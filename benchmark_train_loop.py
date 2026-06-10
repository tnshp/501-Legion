"""
Benchmark the SAC training inner loop from a real train config and locate the
bottleneck (CPU env / host->device transfer / GPU compute). Supports both the
single-env rollout and a **parallel (vectorized) env** rollout so you can measure
the throughput gain from collecting many environments at once.

It mirrors train_orbit_wars.train()'s rollout: select_action, env.step,
buffer.add, per-step TensorBoard logging, and the periodic trainer.update().
The vectorized path mirrors SACTrainer's own VecEnv loop (select_action_batch,
add_batch, gymnasium auto-reset).

Outputs
-------
- steps/sec, transitions/sec, and ms/step for the loop,
- a per-phase breakdown (select / envstep / add / log / update) tagged CPU vs GPU,
- GPU utilization (mean/peak) and memory, sampled from nvidia-smi during the run,
- a transfer-vs-compute split of update(), and a one-line bottleneck verdict,
- with --num-envs N>1: a single-env baseline vs vectorized speedup.

Usage
-----
    python benchmark_train_loop.py --config train.json
    python benchmark_train_loop.py --config train.json --num-envs 8 --vec-mode async
    python benchmark_train_loop.py --config train.json --num-envs 4 --vec-mode sync --steps 400
"""
from __future__ import annotations

import argparse
import functools
import subprocess
import tempfile
import threading
import time
from collections import deque

import numpy as np
import torch
import gymnasium as gym

from model.SAC import P_network, Q_network
from sac_train import SACTrainer
from env.orbit_wars import (
    OrbitWarsEnv,
    RelativeShipAdvantage, RelativeProductionAdvantage,
    ShipGrowth, ProductionPlanetDelta, ProximityCaptureBonus,
    AbsoluteHoldings, FleetLaunchPenalty, LaunchDistancePenalty, StepPenalty,
    TerminalWinBonus, TimeDecayWinBonus,
    RewardScheme1, RewardScheme2, RewardScheme3, RewardScheme4,
)
from train_orbit_wars import make_env, load_config

REWARD_MAP = {
    "RelativeShipAdvantage":       RelativeShipAdvantage,
    "RelativeProductionAdvantage": RelativeProductionAdvantage,
    "ShipGrowth":                  ShipGrowth,
    "ProductionPlanetDelta":       ProductionPlanetDelta,
    "ProximityCaptureBonus":       ProximityCaptureBonus,
    "AbsoluteHoldings":            AbsoluteHoldings,
    "FleetLaunchPenalty":          FleetLaunchPenalty,
    "LaunchDistancePenalty":       LaunchDistancePenalty,
    "StepPenalty":                 StepPenalty,
    "TerminalWinBonus":            TerminalWinBonus,
    "TimeDecayWinBonus":           TimeDecayWinBonus,
    "RewardScheme1": RewardScheme1, "RewardScheme2": RewardScheme2,
    "RewardScheme3": RewardScheme3, "RewardScheme4": RewardScheme4,
}


def build_reward_scheme(config):
    schemes = []
    for cfg in config.get("reward", []):
        name = cfg.get("scheme", "RewardScheme1")
        params = {k: v for k, v in cfg.items() if k != "scheme"}
        schemes.append(REWARD_MAP[name](**params))
    return schemes or [RewardScheme1()]


# ─────────────────────────────────────────────────────────────────────────────
# Parallel-env support
# ─────────────────────────────────────────────────────────────────────────────
class OrbitGymAdapter(gym.Wrapper):
    """Make OrbitWarsEnv compatible with gymnasium's vector envs.

    OrbitWarsEnv.step returns `won` (a bool) as its 5th value instead of the
    gymnasium `info` dict. Vector envs (Sync/AsyncVectorEnv) require a dict, so
    this wraps step to emit `{"won", "fleets_sent", "n_fleets"}` and otherwise
    passes everything through unchanged.
    """
    def step(self, action):
        obs, reward, terminated, truncated, won = self.env.step(action)
        info = {"won": bool(won),
                "fleets_sent": int(getattr(self.env, "last_fleets_sent", 0)),
                "n_fleets": int(getattr(self.env, "last_n_fleets", 0))}
        return obs, reward, terminated, truncated, info


def _adapted_env_factory(config, n_players, max_planets, max_fleets):
    """Module-level factory (picklable → works with AsyncVectorEnv 'spawn').

    Each worker rebuilds its own reward scheme + env, so nothing CUDA/env-stateful
    is pickled across the process boundary.
    """
    env_cfg = config.get("environment", {})
    reward_scheme = build_reward_scheme(config)
    env, _ = make_env(
        n_players=n_players, opponent=env_cfg.get("opponent", "rule_based"),
        MAX_PLANETS=max_planets, MAX_FLEETS=max_fleets, reward_scheme=reward_scheme,
        mixed_random_ratio=config.get("curriculum", {}).get("mixed_random_ratio_start", 1.0),
        mixed_send_prob=env_cfg.get("mixed_send_prob", 0.3),
        tanh_scale=env_cfg.get("tanh_scale", 0.2),
        min_fleet_ships=env_cfg.get("min_fleet_ships", 3))
    return OrbitGymAdapter(env)


def build_vec_env(config, n_envs, mode, n_players, max_planets, max_fleets):
    """Build a gymnasium vector env of `n_envs` OrbitWars instances.

    mode="sync"  → SyncVectorEnv  (in-process, envs stepped sequentially; the
                   throughput win comes purely from batching the GPU calls).
    mode="async" → AsyncVectorEnv (one worker process per env → the CPU game sim
                   runs in true parallel; this is the path that scales the env on
                   a fast GPU where the Python env is the bottleneck). Uses the
                   'fork' start method: workers inherit the already-imported
                   modules (the kaggle env can't be cleanly re-imported under
                   spawn) and only ever run the CPU env, so they never touch the
                   parent's CUDA context.
    """
    factory = functools.partial(_adapted_env_factory, config, n_players,
                                max_planets, max_fleets)
    if mode == "async":
        return gym.vector.AsyncVectorEnv([factory] * n_envs, context="fork")
    return gym.vector.SyncVectorEnv([factory] * n_envs)


# ─────────────────────────────────────────────────────────────────────────────
# GPU utilization sampler (nvidia-smi polled in a background thread)
# ─────────────────────────────────────────────────────────────────────────────
class GPUSampler(threading.Thread):
    """Polls `nvidia-smi` at ~10 Hz and records gpu/mem utilization + mem used.

    nvidia-smi releases the GIL while the child process runs, so this barely
    perturbs the main-thread timing. If nvidia-smi is missing it self-disables.
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
        gu, mu, used = map(np.array, (self.gpu_util, self.mem_util, self.mem_used))
        print(f"  samples            : {len(gu)}")
        print(f"  GPU util  (compute): mean {gu.mean():5.1f}%   peak {gu.max():5.0f}%")
        print(f"  GPU util  (memory) : mean {mu.mean():5.1f}%   peak {mu.max():5.0f}%")
        print(f"  GPU memory  used   : mean {used.mean():6.0f} MiB  peak {used.max():6.0f}"
              f" MiB  / {self.mem_total:.0f} MiB")
        return float(gu.mean())


# ─────────────────────────────────────────────────────────────────────────────
def make_trainer(config, env, device):
    t = config.get("training", {})
    m = config.get("model", {})
    tdl = config.get("td_lambda", {})
    net_kw = dict(state_dim=OrbitWarsEnv.STATE_DIM, action_dim=OrbitWarsEnv.ACTION_DIM,
                  max_planets=OrbitWarsEnv.MAX_PLANETS, max_fleets=OrbitWarsEnv.MAX_FLEETS,
                  d_model=m.get("d_model", 128))
    policy, q1, q2 = P_network(**net_kw), Q_network(**net_kw), Q_network(**net_kw)
    return SACTrainer(
        env=env, policy_net=policy, q1_net=q1, q2_net=q2, device=device,
        learning_rate=t.get("lr", 3e-4), gamma=t.get("gamma", 0.99),
        tau=t.get("tau", 5e-3), alpha=t.get("alpha", 0.2),
        auto_alpha=t.get("auto_alpha", False), target_entropy=t.get("target_entropy"),
        replay_buffer_size=t.get("buffer_size", 100_000), batch_size=t.get("batch_size", 64),
        max_grad_norm=t.get("grad_clip", 1.0),
        use_lambda_returns=tdl.get("enabled", False), lambda_return=tdl.get("lambda", 0.9),
        cache_size=tdl.get("cache_size", 8000), block_size=tdl.get("block_size", 50),
        refresh_freq=tdl.get("refresh_freq", 1000), log_dir=tempfile.mkdtemp())


# ─────────────────────────────────────────────────────────────────────────────
# Single-env rollout (1 transition / step)
# ─────────────────────────────────────────────────────────────────────────────
def run_single(trainer, env, n_steps, warmup, update_freq, batch_size, device, label):
    fleets_window = deque(maxlen=50)
    reward_window = deque(maxlen=100)
    max_fleets_seen = 0
    writer = trainer.writer
    is_cuda = device == "cuda"
    t = {k: 0.0 for k in ("select", "envstep", "add", "log", "update")}
    state, _ = env.reset()
    n_updates = 0
    if is_cuda: torch.cuda.synchronize()
    wall0 = time.perf_counter()
    step = 0
    while step < n_steps:
        t0 = time.perf_counter()
        if trainer.train_step < warmup:
            action = env.action_space.sample()
        else:
            action = trainer.select_action(state)
        if is_cuda: torch.cuda.synchronize()
        t1 = time.perf_counter()

        next_state, reward, terminated, truncated, won = env.step(action)
        done = terminated or truncated
        t2 = time.perf_counter()

        trainer.replay_buffer.add(state, action, reward, next_state, float(done))
        trainer.train_step += 1
        state = next_state
        t3 = time.perf_counter()

        reward_window.append(float(reward))
        fleets_window.append(env.last_fleets_sent)
        max_fleets_seen = max(max_fleets_seen, env.last_n_fleets)
        if writer:
            writer.add_scalar("Reward/step", float(reward), trainer.train_step)
            writer.add_scalar("Reward/step_ma100", float(np.mean(reward_window)), trainer.train_step)
            writer.add_scalar("Policy/fleets_sent_ma50", float(np.mean(fleets_window)), trainer.train_step)
            writer.add_scalar("Env/fleets_present", env.last_n_fleets, trainer.train_step)
            writer.add_scalar("Env/fleets_present_max", max_fleets_seen, trainer.train_step)
        t4 = time.perf_counter()

        if (trainer.train_step >= warmup and trainer.train_step % update_freq == 0
                and len(trainer.replay_buffer) >= batch_size):
            trainer.update()
            if is_cuda: torch.cuda.synchronize()
            n_updates += 1
        t5 = time.perf_counter()

        t["select"] += t1 - t0; t["envstep"] += t2 - t1; t["add"] += t3 - t2
        t["log"] += t4 - t3; t["update"] += t5 - t4
        if done:
            state, _ = env.reset()
        step += 1
    wall = time.perf_counter() - wall0
    return _report(label, t, wall, n_steps, n_transitions=n_steps, n_updates=n_updates,
                   device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized rollout (n_envs transitions / step), mirrors SACTrainer's vec loop
# ─────────────────────────────────────────────────────────────────────────────
def run_vec(trainer, venv, n_steps, warmup, update_freq, batch_size, device, n_envs, label):
    reward_window = deque(maxlen=100)
    writer = trainer.writer
    is_cuda = device == "cuda"
    t = {k: 0.0 for k in ("select", "envstep", "add", "log", "update")}
    states, _ = venv.reset()
    n_updates = 0
    upd_debt = 0.0          # keep UTD identical to single-env: 1 update / update_freq transitions
    if is_cuda: torch.cuda.synchronize()
    wall0 = time.perf_counter()
    step = 0
    while step < n_steps:
        t0 = time.perf_counter()
        if trainer.train_step < warmup:
            actions = venv.action_space.sample()
        else:
            actions = trainer.select_action_batch(states)
        if is_cuda: torch.cuda.synchronize()
        t1 = time.perf_counter()

        next_states, rewards, terminated, truncated, infos = venv.step(actions)
        dones = np.logical_or(terminated, truncated)
        t2 = time.perf_counter()

        # Terminal-obs fix (only present under the old autoreset convention).
        real_next = next_states
        if "final_observation" in infos:
            real_next = next_states.copy()
            for i, (d, fo) in enumerate(zip(dones, infos["final_observation"])):
                if d and fo is not None:
                    real_next[i] = fo
        trainer.replay_buffer.add_batch(states, actions, rewards, real_next, dones)
        trainer.train_step += n_envs
        states = next_states
        t3 = time.perf_counter()

        reward_window.append(float(np.mean(rewards)))
        if writer:
            writer.add_scalar("Reward/step", float(np.mean(rewards)), trainer.train_step)
            writer.add_scalar("Reward/step_ma100", float(np.mean(reward_window)), trainer.train_step)
        t4 = time.perf_counter()

        if trainer.train_step >= warmup and len(trainer.replay_buffer) >= batch_size:
            upd_debt += n_envs / update_freq
            while upd_debt >= 1.0:
                trainer.update()
                n_updates += 1
                upd_debt -= 1.0
            if is_cuda: torch.cuda.synchronize()
        t5 = time.perf_counter()

        t["select"] += t1 - t0; t["envstep"] += t2 - t1; t["add"] += t3 - t2
        t["log"] += t4 - t3; t["update"] += t5 - t4
        step += 1
    wall = time.perf_counter() - wall0
    return _report(label, t, wall, n_steps, n_transitions=n_steps * n_envs,
                   n_updates=n_updates, device=device, n_envs=n_envs)


# ─────────────────────────────────────────────────────────────────────────────
def _report(label, t, wall, n_vsteps, n_transitions, n_updates, device, n_envs=1):
    kind = {"select": "GPU", "envstep": "CPU", "add": "CPU", "log": "CPU", "update": "GPU"}
    tps = n_transitions / wall
    env_tag = f", {n_envs} envs" if n_envs > 1 else ""
    print(f"\n=== {label}: {n_vsteps} steps{env_tag}, {n_transitions} transitions, "
          f"{n_updates} updates, device={device} ===")
    print(f"  {'TOTAL wall':18s}: {wall*1000:8.1f} ms   "
          f"({wall/n_transitions*1000:7.3f} ms/transition)   {tps:8.1f} transitions/s")
    for k in ("select", "envstep", "add", "log", "update"):
        share = t[k] / wall * 100
        print(f"  {k:18s}: {t[k]*1000:8.1f} ms   ({t[k]/n_vsteps*1000:7.3f} ms/vstep)"
              f"   {share:5.1f}%  [{kind[k]}]")
    cpu_ms = sum(t[k] for k in ("envstep", "add", "log")) / n_vsteps * 1000
    gpu_ms = sum(t[k] for k in ("select", "update")) / n_vsteps * 1000
    print(f"  {'-> CPU phases':18s}: {cpu_ms:7.3f} ms/vstep   "
          f"{'-> GPU phases':18s}: {gpu_ms:7.3f} ms/vstep")
    return {"tps": tps, "cpu_ms": cpu_ms, "gpu_ms": gpu_ms}


# ─────────────────────────────────────────────────────────────────────────────
def transfer_vs_compute(trainer, device, batch_size, iters=50):
    """Split update() into host->device transfer vs GPU compute."""
    if device != "cuda":
        print("  (CPU device — no host/device transfer to measure)")
        return
    if len(trainer.replay_buffer) < batch_size:
        print("  (buffer too small — skipped)")
        return
    s_shape = trainer.replay_buffer.states.shape[1:]
    a_shape = trainer.replay_buffer.actions.shape[1:]
    states = torch.from_numpy(np.zeros((batch_size, *s_shape), np.float32))
    actions = torch.from_numpy(np.zeros((batch_size, *a_shape), np.float32))
    rewards = torch.from_numpy(np.zeros((batch_size, 1), np.float32))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        states.to(device, non_blocking=True)
        actions.to(device, non_blocking=True)
        rewards.to(device, non_blocking=True)
        states.to(device, non_blocking=True)
        rewards.to(device, non_blocking=True)
        torch.cuda.synchronize()
    transfer_ms = (time.perf_counter() - t0) / iters * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        trainer.update()
        torch.cuda.synchronize()
    update_ms = (time.perf_counter() - t0) / iters * 1000

    compute_ms = max(0.0, update_ms - transfer_ms)
    bytes_per = sum(x.numel() * 4 for x in (states, actions, rewards, states, rewards))
    bw = bytes_per / (transfer_ms / 1000) / 1e9 if transfer_ms > 0 else 0.0
    print(f"  full update()      : {update_ms:7.3f} ms  (per update, not per step)")
    print(f"  H2D transfer       : {transfer_ms:7.3f} ms  ({transfer_ms/update_ms*100:4.1f}% of update)"
          f"   ~{bw:5.1f} GB/s  ({bytes_per/1024:.0f} KiB/batch)")
    print(f"  GPU compute (rest) : {compute_ms:7.3f} ms  ({compute_ms/update_ms*100:4.1f}% of update)")


def verdict(gpu_util_mean, cpu_ms, gpu_ms):
    print("\n=== BOTTLENECK VERDICT ===")
    total = cpu_ms + gpu_ms
    if total <= 0:
        return
    if cpu_ms > gpu_ms:
        print(f"  CPU-bound: env rollout + buffer + logging = {cpu_ms:.2f} ms/vstep "
              f"({cpu_ms/total*100:.0f}%) vs GPU {gpu_ms:.2f} ms/vstep.")
        print("  The env (numpy/python game sim) + per-step logging dominate. On a bigger")
        print("  GPU this gap WIDENS. Use --vec-mode async to step envs across CPU cores.")
    else:
        print(f"  GPU-bound: train update + action select = {gpu_ms:.2f} ms/vstep "
              f"({gpu_ms/total*100:.0f}%) vs CPU {cpu_ms:.2f} ms/vstep.")
        print("  Vectorizing batches the GPU calls (one fwd for N envs), so transitions/s")
        print("  rises even though per-vstep GPU time grows sub-linearly with N.")
    if gpu_util_mean is not None:
        if gpu_util_mean < 35:
            print(f"  GPU util mean {gpu_util_mean:.0f}% is LOW → GPU starved between kernels;")
            print("  feed bigger/fewer batches and cut per-step CPU work.")
        elif gpu_util_mean > 70:
            print(f"  GPU util mean {gpu_util_mean:.0f}% is HIGH → genuinely compute-bound; a faster")
            print("  GPU should translate fairly directly into more transitions/s.")
        else:
            print(f"  GPU util mean {gpu_util_mean:.0f}% is MODERATE → partial starvation; a faster")
            print("  GPU helps the compute portion but CPU/transfer caps the gains.")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Benchmark the SAC training inner loop from a train config.")
    ap.add_argument("--config", default="train.json", help="train config JSON (default: train.json)")
    ap.add_argument("--steps", type=int, default=400, help="measured vector steps (default: 400)")
    ap.add_argument("--warmup", type=int, default=None,
                    help="random-action warmup transitions before measuring (default: config warmup_steps)")
    ap.add_argument("--num-envs", type=int, default=1, help="parallel envs (default: 1 = single-env)")
    ap.add_argument("--vec-mode", choices=("sync", "async"), default="async",
                    help="sync=in-process (GPU batching only); async=multiprocess envs (default: async)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="with --num-envs>1, skip the single-env baseline comparison")
    args = ap.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and not config.get("execution", {}).get("cpu_force") else "cpu"
    train_cfg = config.get("training", {})
    env_cfg = config.get("environment", {})
    batch_size = train_cfg.get("batch_size", 64)
    update_freq = train_cfg.get("update_freq", 4)
    warmup = args.warmup if args.warmup is not None else train_cfg.get("warmup_steps", 500)
    n_players = env_cfg.get("n_players_default", 2)
    MAX_PLANETS, MAX_FLEETS = 40, 200

    print(f"Config: {args.config}   device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  (cc {torch.cuda.get_device_capability(0)})")
    print(f"batch_size={batch_size}  update_freq={update_freq}  warmup={warmup}  "
          f"td_lambda={config.get('td_lambda', {}).get('enabled', False)}  "
          f"num_envs={args.num_envs}  vec_mode={args.vec_mode}")
    reward_scheme = build_reward_scheme(config)
    print(f"Reward: {[r.__class__.__name__ for r in reward_scheme]}")

    baseline_tps = None

    # ── optional single-env baseline (build first, before any vec workers fork) ──
    if args.num_envs == 1 or not args.no_baseline:
        env, _ = make_env(n_players=n_players, opponent=env_cfg.get("opponent", "rule_based"),
                          MAX_PLANETS=MAX_PLANETS, MAX_FLEETS=MAX_FLEETS, reward_scheme=reward_scheme,
                          mixed_random_ratio=config.get("curriculum", {}).get("mixed_random_ratio_start", 1.0),
                          mixed_send_prob=env_cfg.get("mixed_send_prob", 0.3),
                          tanh_scale=env_cfg.get("tanh_scale", 0.2),
                          min_fleet_ships=env_cfg.get("min_fleet_ships", 3))
        trainer = make_trainer(config, env, device)
        run_single(trainer, env, warmup + 120, warmup, update_freq, batch_size, device, "warmup (ignored)")
        sampler = GPUSampler(); sampler.start()
        res = run_single(trainer, env, args.steps, 0, update_freq, batch_size, device, "SINGLE-ENV")
        sampler.stop()
        baseline_tps = res["tps"]
        print("\n=== GPU UTILIZATION (single-env) ===")
        gpu_util = sampler.report()
        if device == "cuda":
            print(f"  torch peak allocated: {torch.cuda.max_memory_allocated()/1024**2:6.0f} MiB")
        print("\n=== UPDATE: TRANSFER vs COMPUTE ===")
        transfer_vs_compute(trainer, device, batch_size)
        verdict(gpu_util, res["cpu_ms"], res["gpu_ms"])
        trainer.close(); env.close()

    # ── vectorized run ──────────────────────────────────────────────────────────
    if args.num_envs > 1:
        print(f"\n{'='*70}\nVECTORIZED  ({args.num_envs} envs, {args.vec_mode})\n{'='*70}")
        venv = build_vec_env(config, args.num_envs, args.vec_mode, n_players, MAX_PLANETS, MAX_FLEETS)
        trainer = make_trainer(config, venv, device)
        # warmup: fill buffer past `warmup` (each vstep adds num_envs transitions)
        warm_vsteps = warmup // args.num_envs + 30
        run_vec(trainer, venv, warm_vsteps, warmup, update_freq, batch_size, device,
                args.num_envs, "warmup (ignored)")
        sampler = GPUSampler(); sampler.start()
        res = run_vec(trainer, venv, args.steps, 0, update_freq, batch_size, device,
                      args.num_envs, "VECTORIZED")
        sampler.stop()
        print("\n=== GPU UTILIZATION (vectorized) ===")
        gpu_util = sampler.report()
        if device == "cuda":
            print(f"  torch peak allocated: {torch.cuda.max_memory_allocated()/1024**2:6.0f} MiB")
        verdict(gpu_util, res["cpu_ms"], res["gpu_ms"])
        trainer.close(); venv.close()

        if baseline_tps is not None:
            print(f"\n{'='*70}\n=== SPEEDUP ===")
            print(f"  single-env     : {baseline_tps:8.1f} transitions/s")
            print(f"  {args.num_envs:2d}-env {args.vec_mode:5s}  : {res['tps']:8.1f} transitions/s")
            print(f"  speedup        : {res['tps']/baseline_tps:6.2f}x  "
                  f"(ideal {args.num_envs}x)")
            print(f"{'='*70}")


if __name__ == "__main__":
    main()
