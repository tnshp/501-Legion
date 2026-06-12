"""
Benchmark the JAX-vectorised SAC training inner loop.

Measures per-phase timing for the train_jax() hot path:

  select   — select_action_batch: network inference on (B, seq, 13)
  envstep  — JaxVecEnvAdapter.step: JAX game step + obs extraction + reward
  add      — replay_buffer.add_batch  [CPU with ReplayBuffer, GPU with JaxReplayBuffer]
  update   — SAC gradient updates (debt-scaled at num_envs/update_freq per vstep)

Optionally sweeps over multiple num_envs values to plot scaling behaviour.

Usage
-----
    python benchmark_jax_train_loop.py                              # default config
    python benchmark_jax_train_loop.py --config train.json --num-envs 128
    python benchmark_jax_train_loop.py --sweep 64,128,256,512      # scaling sweep
    python benchmark_jax_train_loop.py --jax-buffer                # GPU-resident buffer
    python benchmark_jax_train_loop.py --jax-buffer --sweep 256,512,1024  # both
"""
from __future__ import annotations

import argparse
import subprocess
import threading
import time
from collections import deque

import jax
import numpy as np
import torch

from model.SAC import P_network, Q_network
from sac_train import SACTrainer
from env.orbit_wars import OrbitWarsEnv
from jax_env import JaxVecEnvAdapter
from train_orbit_wars import load_config


# ─────────────────────────────────────────────────────────────────────────────
# GPU utilization sampler (nvidia-smi polled in a background thread)
# ─────────────────────────────────────────────────────────────────────────────

class GPUSampler(threading.Thread):
    """Polls nvidia-smi at ~10 Hz and records gpu/mem utilization + mem used.

    Self-disables silently if nvidia-smi is not available.
    """
    QUERY = ("--query-gpu=utilization.gpu,utilization.memory,memory.used,"
             "memory.total", "--format=csv,noheader,nounits")

    def __init__(self, period=0.1):
        super().__init__(daemon=True)
        self.period = period
        self._stop_evt = threading.Event()
        self.gpu_util, self.mem_util, self.mem_used = [], [], []
        self.mem_total = None
        self.available = self._probe()

    def _probe(self):
        try:
            subprocess.run(("nvidia-smi", *self.QUERY), capture_output=True,
                           text=True, timeout=5, check=True)
            return True
        except Exception:
            return False

    def _read(self):
        out = subprocess.run(("nvidia-smi", *self.QUERY), capture_output=True,
                             text=True, timeout=5, check=True).stdout
        g, m, used, total = (float(x) for x in out.strip().splitlines()[0].split(","))
        return g, m, used, total

    def run(self):
        if not self.available:
            return
        while not self._stop_evt.is_set():
            try:
                g, m, used, total = self._read()
                self.gpu_util.append(g); self.mem_util.append(m)
                self.mem_used.append(used); self.mem_total = total
            except Exception:
                pass
            self._stop_evt.wait(self.period)

    def stop(self):
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=2)

    def report(self):
        if not self.available or not self.gpu_util:
            print("  (nvidia-smi unavailable — GPU utilization not sampled)")
            return None
        gu  = np.array(self.gpu_util)
        mu  = np.array(self.mem_util)
        used = np.array(self.mem_used)
        print(f"  samples            : {len(gu)}")
        print(f"  GPU util  (compute): mean {gu.mean():5.1f}%   peak {gu.max():5.0f}%")
        print(f"  GPU util  (memory) : mean {mu.mean():5.1f}%   peak {mu.max():5.0f}%")
        print(f"  GPU memory  used   : mean {used.mean():6.0f} MiB  peak {used.max():6.0f}"
              f" MiB  / {self.mem_total:.0f} MiB")
        return float(gu.mean())


def verdict(gpu_util_mean: float | None, cpu_ms: float, gpu_ms: float,
            jax_buffer: bool = False):
    print("\n=== BOTTLENECK VERDICT ===")
    total = cpu_ms + gpu_ms
    if total <= 0:
        return
    cpu_label = "env rollout" if jax_buffer else "env rollout + buffer"
    gpu_label = "inference + buffer scatter + SAC updates" if jax_buffer else "network inference + SAC updates"
    if cpu_ms > gpu_ms:
        print(f"  CPU-bound: {cpu_label} = {cpu_ms:.3f} ms/vstep "
              f"({cpu_ms/total*100:.0f}%) vs GPU {gpu_ms:.3f} ms/vstep.")
        print("  JAX env.step() / obs extraction dominates.")
        print("  → Increase num_envs to amortise per-step GPU calls.")
    else:
        print(f"  GPU-bound: {gpu_label} = {gpu_ms:.3f} ms/vstep "
              f"({gpu_ms/total*100:.0f}%) vs CPU {cpu_ms:.3f} ms/vstep.")
        print("  → Increase batch_size or use a bigger model to improve GPU utilisation.")
    if gpu_util_mean is not None:
        if gpu_util_mean < 35:
            print(f"  GPU util mean {gpu_util_mean:.0f}% is LOW → GPU starved between kernels;")
            print("  feed bigger/fewer batches and cut per-step CPU work.")
        elif gpu_util_mean > 70:
            print(f"  GPU util mean {gpu_util_mean:.0f}% is HIGH → genuinely compute-bound;")
            print("  a faster GPU translates fairly directly into more transitions/s.")
        else:
            print(f"  GPU util mean {gpu_util_mean:.0f}% is MODERATE → partial starvation;")
            print("  a faster GPU helps the compute portion but CPU caps the gains.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_env(config: dict, num_envs: int, num_players: int = 2) -> JaxVecEnvAdapter:
    env_cfg    = config.get("environment", {})
    jax_cfg    = config.get("jax_env",     {})
    reward_cfg = config.get("reward",      [])
    return JaxVecEnvAdapter(
        num_envs        = num_envs,
        num_players     = num_players,
        episode_steps   = jax_cfg.get("episode_steps", 500),
        tanh_scale      = env_cfg.get("tanh_scale",    0.2),
        min_fleet_ships = env_cfg.get("min_fleet_ships", 3),
        reward_cfg      = reward_cfg,
        # fallback fields kept for compat when reward_cfg is empty
        reward_type     = jax_cfg.get("reward_type",  "ship_advantage"),
        reward_scale    = jax_cfg.get("reward_scale", 0.01),
        win_bonus       = jax_cfg.get("win_bonus",    100.0),
        opponent        = env_cfg.get("opponent",     "random"),
    )


def _build_trainer(config: dict, env: JaxVecEnvAdapter, device: str,
                   use_jax_buffer: bool = False) -> SACTrainer:
    t   = config.get("training",  {})
    m   = config.get("model",     {})
    tdl = config.get("td_lambda", {})
    net_kw = dict(
        state_dim   = OrbitWarsEnv.STATE_DIM,
        action_dim  = OrbitWarsEnv.ACTION_DIM,
        max_planets = 40,
        max_fleets  = 200,
        d_model     = m.get("d_model", 128),
    )
    return SACTrainer(
        env                = env,
        policy_net         = P_network(**net_kw),
        q1_net             = Q_network(**net_kw),
        q2_net             = Q_network(**net_kw),
        device             = device,
        learning_rate      = t.get("lr",            3e-4),
        gamma              = t.get("gamma",          0.99),
        tau                = t.get("tau",            5e-3),
        alpha              = t.get("alpha",          0.2),
        auto_alpha         = t.get("auto_alpha",     False),
        target_entropy     = t.get("target_entropy"),
        replay_buffer_size = t.get("buffer_size",    100_000),
        batch_size         = t.get("batch_size",     64),
        max_grad_norm      = t.get("grad_clip",      1.0),
        use_lambda_returns = tdl.get("enabled",      False),
        lambda_return      = tdl.get("lambda",       0.9),
        cache_size         = tdl.get("cache_size",   8000),
        block_size         = tdl.get("block_size",   50),
        refresh_freq       = tdl.get("refresh_freq", 1000),
        log_dir            = None,
        use_jax_buffer     = use_jax_buffer,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def _run(
    trainer:       SACTrainer,
    env:           JaxVecEnvAdapter,
    num_envs:      int,
    warmup_steps:  int,
    update_freq:   int,
    grad_steps:    int,
    batch_size:    int,
    device:        str,
    n_vsteps:      int,
    label:         str,
    jax_buffer:    bool = False,
) -> dict:
    """Run one benchmark pass and return timing statistics.

    warmup_steps=0 means gradient updates start from the first vector step
    (use this for the measured pass when the buffer is already pre-filled).

    When jax_buffer=True, a JAX sync is inserted after add_batch so that the
    async GPU scatter completes before the timer for the 'add' phase stops.
    Without it the scatter would bleed into the 'update' phase wall time,
    making 'add' look free and 'update' look artificially slow.
    """
    is_cuda  = (device == "cuda")
    phases   = {k: 0.0 for k in ("select", "envstep", "add", "update")}
    n_upds   = 0
    upd_debt = 0.0

    obs, _ = env.reset()
    if is_cuda:
        torch.cuda.synchronize()
    wall0 = time.perf_counter()

    for _ in range(n_vsteps):
        # ── select action ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if trainer.train_step < warmup_steps:
            actions = np.stack([env.action_space.sample() for _ in range(num_envs)])
        else:
            actions = trainer.select_action_batch(obs)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # ── env step ─────────────────────────────────────────────────────────
        next_obs, rewards, dones, _, _ = env.step(actions)
        t2 = time.perf_counter()

        # ── buffer ───────────────────────────────────────────────────────────
        trainer.replay_buffer.add_batch(
            obs, actions, rewards, next_obs, dones.astype(np.float32)
        )
        # With JaxReplayBuffer the scatter is dispatched asynchronously to the
        # GPU.  Block here so the 'add' timer captures the true scatter cost
        # rather than letting it bleed into the 'update' phase.
        if jax_buffer:
            trainer.replay_buffer.block_until_ready()
        trainer.train_step += num_envs
        obs = next_obs
        t3 = time.perf_counter()

        # ── gradient updates (debt mechanism) ────────────────────────────────
        if (trainer.train_step >= warmup_steps
                and len(trainer.replay_buffer) >= batch_size):
            upd_debt += num_envs / update_freq
            while upd_debt >= 1.0:
                for _ in range(grad_steps):
                    trainer.update()
                n_upds   += 1
                upd_debt -= 1.0
            if is_cuda:
                torch.cuda.synchronize()
        t4 = time.perf_counter()

        phases["select"]  += t1 - t0
        phases["envstep"] += t2 - t1
        phases["add"]     += t3 - t2
        phases["update"]  += t4 - t3

    wall   = time.perf_counter() - wall0
    n_tr   = n_vsteps * num_envs

    if label:
        _print_report(label, phases, wall, n_vsteps, n_tr, n_upds, device, jax_buffer)

    # With JaxReplayBuffer, 'add' is GPU work (JAX scatter); attribute it to
    # the GPU side so the bottleneck verdict is accurate.
    if jax_buffer:
        cpu_ms = phases["envstep"] / n_vsteps * 1000
        gpu_ms = (phases["select"] + phases["add"] + phases["update"]) / n_vsteps * 1000
    else:
        cpu_ms = (phases["envstep"] + phases["add"]) / n_vsteps * 1000
        gpu_ms = (phases["select"]  + phases["update"]) / n_vsteps * 1000

    return {"tps": n_tr / wall, "cpu_ms": cpu_ms, "gpu_ms": gpu_ms,
            "wall": wall, "n_updates": n_upds}


def _print_report(label, phases, wall, n_vsteps, n_tr, n_upds, device,
                  jax_buffer: bool = False):
    tps = n_tr / wall
    print(f"\n=== {label} ===")
    print(f"  {n_vsteps} vsteps × {n_tr//n_vsteps} envs = {n_tr} transitions"
          f" | {n_upds} gradient updates | device={device}"
          + ("  [JaxReplayBuffer]" if jax_buffer else ""))
    print(f"  {'TOTAL wall':20s}: {wall*1000:9.1f} ms  "
          f"({wall/n_tr*1000:.4f} ms/transition)  {tps:9.1f} trans/s")
    # 'add' moves from CPU to GPU when using JaxReplayBuffer.
    phase_device = {
        "select":  "GPU",
        "envstep": "CPU",
        "add":     "GPU*" if jax_buffer else "CPU",
        "update":  "GPU",
    }
    for k in ("select", "envstep", "add", "update"):
        share = phases[k] / wall * 100
        print(f"  {k:20s}: {phases[k]*1000:9.1f} ms  "
              f"({phases[k]/n_vsteps*1000:7.3f} ms/vstep)  "
              f"{share:5.1f}%  [{phase_device[k]}]")
    if jax_buffer:
        cpu_ms = phases["envstep"] / n_vsteps * 1000
        gpu_ms = (phases["select"] + phases["add"] + phases["update"]) / n_vsteps * 1000
        print(f"\n  CPU (envstep)           : {cpu_ms:.3f} ms/vstep")
        print(f"  GPU (select+add*+update): {gpu_ms:.3f} ms/vstep")
        print(f"  * add = JAX GPU scatter (async H2D + in-place scatter)")
    else:
        cpu_ms = (phases["envstep"] + phases["add"]) / n_vsteps * 1000
        gpu_ms = (phases["select"]  + phases["update"]) / n_vsteps * 1000
        print(f"\n  CPU (envstep + add): {cpu_ms:.3f} ms/vstep")
        print(f"  GPU (select + upd) : {gpu_ms:.3f} ms/vstep")
    bot = "CPU-bound" if cpu_ms > gpu_ms else "GPU-bound"
    print(f"  BOTTLENECK         : {bot}")
    if cpu_ms > gpu_ms:
        print("    → env.step() / obs extraction dominates. "
              "Increase num_envs to amortise per-step GPU calls, "
              "or reduce observation overhead.")
    else:
        print("    → network inference / SAC updates dominate. "
              "Increase batch_size or use a bigger model to improve GPU utilisation.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Benchmark the JAX-vectorised SAC training loop")
    ap.add_argument("--config",       default="train.json",
                    help="path to train config JSON (default: train.json)")
    ap.add_argument("--num-envs",     type=int, default=None,
                    help="parallel envs (default: jax_env.num_envs from config)")
    ap.add_argument("--num-players",  type=int, default=2,
                    help="players per game (default: 2)")
    ap.add_argument("--steps",        type=int, default=300,
                    help="measured vector steps (default: 300)")
    ap.add_argument("--sweep",        type=str, default=None,
                    help="comma-separated num_envs values to sweep, e.g. 64,128,256,512")
    ap.add_argument("--jax-buffer",   action="store_true", default=None,
                    help="use GPU-resident JaxReplayBuffer (overrides jax_env.jax_buffer "
                         "in config; requires JAX CUDA + enough VRAM)")
    args = ap.parse_args()

    config  = load_config(args.config)
    device  = ("cuda" if torch.cuda.is_available()
                and not config.get("execution", {}).get("cpu_force") else "cpu")
    t_cfg   = config.get("training", {})
    jax_cfg = config.get("jax_env",  {})

    # --jax-buffer flag takes precedence; fall back to config value.
    jax_buffer = args.jax_buffer if args.jax_buffer is not None else jax_cfg.get("jax_buffer", False)

    default_num_envs = args.num_envs or jax_cfg.get("num_envs", 256)
    update_freq_cfg  = t_cfg.get("update_freq", None)
    grad_steps       = t_cfg.get("gradient_steps", 1)
    batch_size       = t_cfg.get("batch_size",     64)
    warmup_steps     = t_cfg.get("warmup_steps",   500)

    sweep_sizes = (
        [int(x) for x in args.sweep.split(",")]
        if args.sweep else [default_num_envs]
    )

    print(f"Config : {args.config}   device: {device}")
    if device == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    reward_names = [c.get("scheme") for c in config.get("reward", [])]
    print(f"Reward : {reward_names}")
    if jax_buffer:
        print("Buffer : JaxReplayBuffer (GPU-resident JAX scatter/gather + DLPack)")
        print("         Set XLA_PYTHON_CLIENT_PREALLOCATE=false to share VRAM with PyTorch")
    else:
        print("Buffer : ReplayBuffer (CPU numpy)")
    print(f"grad_steps={grad_steps}  warmup_steps={warmup_steps}  batch_size={batch_size}")

    results = []

    for num_envs in sweep_sizes:
        update_freq = update_freq_cfg if update_freq_cfg is not None else num_envs
        print(f"\n{'='*70}")
        print(f"num_envs={num_envs}  num_players={args.num_players}  update_freq={update_freq}")
        print(f"{'='*70}")

        env     = _build_env(config, num_envs, args.num_players)
        trainer = _build_trainer(config, env, device, use_jax_buffer=jax_buffer)

        # ── Warmup: JIT compilation + fill buffer ─────────────────────────────
        # JAX traces and compiles step() on the first call. We run enough vsteps
        # to cross warmup_steps (so gradient updates also get compiled) plus a
        # small margin.  With JaxReplayBuffer the write and sample kernels are
        # also JIT-compiled on the first add_batch / sample call.
        warmup_vsteps = warmup_steps // num_envs + 2
        print(f"Warming up ({warmup_vsteps} vsteps) — JIT compile + buffer fill...")
        _run(trainer, env, num_envs, warmup_steps, update_freq, grad_steps,
             batch_size, device, n_vsteps=warmup_vsteps, label="",
             jax_buffer=jax_buffer)

        # ── Measured pass ─────────────────────────────────────────────────────
        # warmup_steps=0 so updates start immediately (buffer already filled).
        sampler = GPUSampler(); sampler.start()
        res = _run(trainer, env, num_envs, warmup_steps=0,
                   update_freq=update_freq, grad_steps=grad_steps,
                   batch_size=batch_size, device=device,
                   n_vsteps=args.steps,
                   label=f"JAX  num_envs={num_envs}  {args.num_players}p",
                   jax_buffer=jax_buffer)
        sampler.stop()

        print("\n=== GPU UTILIZATION ===")
        gpu_util = sampler.report()
        if device == "cuda":
            print(f"  torch peak allocated: {torch.cuda.max_memory_allocated()/1024**2:6.0f} MiB")
        verdict(gpu_util, res["cpu_ms"], res["gpu_ms"], jax_buffer=jax_buffer)

        results.append((num_envs, res))
        env.close()
        trainer.close()

    # ── Sweep summary ─────────────────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("SCALING SWEEP SUMMARY")
        print(f"{'='*70}")
        baseline_tps = results[0][1]["tps"]
        print(f"  {'num_envs':>10}  {'trans/s':>12}  {'speedup':>9}  "
              f"{'cpu ms/vs':>12}  {'gpu ms/vs':>12}")
        for n, r in results:
            speedup = r["tps"] / baseline_tps
            print(f"  {n:>10}  {r['tps']:>12.1f}  {speedup:>9.2f}x  "
                  f"{r['cpu_ms']:>12.3f}  {r['gpu_ms']:>12.3f}")
        print(f"  (baseline = num_envs={results[0][0]})")


if __name__ == "__main__":
    main()
