"""
Benchmark the per-step cost of the SAC training inner loop, driven by a real
train config (same JSON `train_orbit_wars.py` consumes), and locate the
bottleneck (CPU env / host->device transfer / GPU compute).

It mirrors train_orbit_wars.train()'s rollout exactly: select_action, env.step,
buffer.add, per-step TensorBoard logging, and the periodic trainer.update().

Outputs
-------
- steps/sec and ms/step for the whole inner loop,
- a per-phase breakdown (select / envstep / add / log / update) tagged CPU vs GPU,
- GPU utilization (mean/peak) and memory, sampled from nvidia-smi during the run,
- a transfer-vs-compute split of update(), and a one-line bottleneck verdict.

Usage
-----
    python benchmark_train_loop.py --config train.json
    python benchmark_train_loop.py --config train.json --steps 1000 --warmup 400
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque

import numpy as np
import torch

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
            return
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
                  max_planets=env.MAX_PLANETS, max_fleets=env.MAX_FLEETS,
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


def run(trainer, env, n_steps, warmup, update_freq, batch_size, device, label):
    fleets_window = deque(maxlen=50)
    reward_window = deque(maxlen=100)
    max_fleets_seen = 0
    writer = trainer.writer
    is_cuda = device == "cuda"
    t = {k: 0.0 for k in ("select", "envstep", "add", "log", "update")}
    state, _ = env.reset()
    n_updates = 0
    if is_cuda:
        torch.cuda.synchronize()
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

    # CPU vs GPU classification of each phase.
    kind = {"select": "GPU", "envstep": "CPU", "add": "CPU", "log": "CPU", "update": "GPU"}
    print(f"\n=== {label}: {n_steps} steps, {n_updates} updates, device={device} ===")
    print(f"  {'TOTAL wall':18s}: {wall*1000:8.1f} ms   ({wall/n_steps*1000:7.3f} ms/step)"
          f"   {n_steps/wall:8.1f} steps/s")
    for k in ("select", "envstep", "add", "log", "update"):
        share = t[k] / wall * 100
        print(f"  {k:18s}: {t[k]*1000:8.1f} ms   ({t[k]/n_steps*1000:7.3f} ms/step)"
              f"   {share:5.1f}%  [{kind[k]}]")
    cpu_ms = sum(t[k] for k in ("envstep", "add", "log")) / n_steps * 1000
    gpu_ms = sum(t[k] for k in ("select", "update")) / n_steps * 1000
    print(f"  {'-> CPU phases':18s}: {cpu_ms:7.3f} ms/step   "
          f"{'-> GPU phases':18s}: {gpu_ms:7.3f} ms/step")
    return wall / n_steps * 1000, n_steps / wall


# ─────────────────────────────────────────────────────────────────────────────
def transfer_vs_compute(trainer, device, batch_size, iters=50):
    """Split update() into host->device transfer vs GPU compute.

    Times a synthetic batch of the exact replay shapes for (a) H2D transfer only
    and (b) a full trainer.update(), then attributes the difference to compute.
    Requires a filled-enough buffer (caller pre-fills via warmup).
    """
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

    # (a) transfer-only: replicate update()'s H2D copies + sync
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        states.to(device, non_blocking=True)
        actions.to(device, non_blocking=True)
        rewards.to(device, non_blocking=True)
        states.to(device, non_blocking=True)   # next_states
        rewards.to(device, non_blocking=True)  # dones
        torch.cuda.synchronize()
    transfer_ms = (time.perf_counter() - t0) / iters * 1000

    # (b) full update (sample + transfer + fwd/bwd/opt)
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
    return transfer_ms, compute_ms


def verdict(gpu_util_mean, cpu_ms, gpu_ms):
    print("\n=== BOTTLENECK VERDICT ===")
    total = cpu_ms + gpu_ms
    if total <= 0:
        return
    if cpu_ms > gpu_ms:
        print(f"  CPU-bound: env rollout + buffer + logging = {cpu_ms:.2f} ms/step "
              f"({cpu_ms/total*100:.0f}%) vs GPU {gpu_ms:.2f} ms/step.")
        print("  The env (numpy/python game sim) and per-step TensorBoard logging dominate.")
        print("  On a bigger GPU this gap WIDENS — GPU phases shrink, env stays the same.")
        print("  Speedups: vectorize envs (collect N rollouts in parallel), batch the")
        print("            updates, throttle/aggregate TensorBoard logging.")
    else:
        print(f"  GPU-bound: train update + action select = {gpu_ms:.2f} ms/step "
              f"({gpu_ms/total*100:.0f}%) vs CPU {cpu_ms:.2f} ms/step.")
    if gpu_util_mean is not None:
        if gpu_util_mean < 35:
            print(f"  GPU util mean {gpu_util_mean:.0f}% is LOW → the GPU is starved: it waits on")
            print("  the CPU/transfer between kernels. A faster GPU alone won't help much;")
            print("  feed it bigger/fewer batches and cut per-step CPU work.")
        elif gpu_util_mean > 70:
            print(f"  GPU util mean {gpu_util_mean:.0f}% is HIGH → genuinely compute-bound;")
            print("  a faster GPU should translate fairly directly into more steps/s.")
        else:
            print(f"  GPU util mean {gpu_util_mean:.0f}% is MODERATE → partial starvation; a faster")
            print("  GPU helps the compute portion but CPU/transfer caps the gains.")


def main():
    ap = argparse.ArgumentParser(description="Benchmark the SAC training inner loop from a train config.")
    ap.add_argument("--config", default="train.json", help="train config JSON (default: train.json)")
    ap.add_argument("--steps", type=int, default=600, help="measured steps (default: 600)")
    ap.add_argument("--warmup", type=int, default=None,
                    help="random-action warmup before measuring (default: config warmup_steps)")
    args = ap.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and not config.get("execution", {}).get("cpu_force") else "cpu"
    train_cfg = config.get("training", {})
    env_cfg = config.get("environment", {})
    batch_size = train_cfg.get("batch_size", 64)
    update_freq = train_cfg.get("update_freq", 4)
    warmup = args.warmup if args.warmup is not None else train_cfg.get("warmup_steps", 500)

    print(f"Config: {args.config}   device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"(cc {torch.cuda.get_device_capability(0)})")
    print(f"batch_size={batch_size}  update_freq={update_freq}  warmup={warmup}  "
          f"td_lambda={config.get('td_lambda', {}).get('enabled', False)}")

    reward_scheme = build_reward_scheme(config)
    print(f"Reward: {[r.__class__.__name__ for r in reward_scheme]}")

    env, _ = make_env(
        n_players=env_cfg.get("n_players_default", 2),
        opponent=env_cfg.get("opponent", "rule_based"),
        MAX_PLANETS=40, MAX_FLEETS=200, reward_scheme=reward_scheme,
        mixed_random_ratio=config.get("curriculum", {}).get("mixed_random_ratio_start", 1.0),
        mixed_send_prob=env_cfg.get("mixed_send_prob", 0.3),
        tanh_scale=env_cfg.get("tanh_scale", 0.2),
        min_fleet_ships=env_cfg.get("min_fleet_ships", 3))
    trainer = make_trainer(config, env, device)

    # Warm up: fill the buffer past `warmup` and run a few updates (not measured).
    run(trainer, env, warmup + 120, warmup, update_freq, batch_size, device, "warmup (ignored)")

    # GPU-sampled measured run.
    sampler = GPUSampler(period=0.1)
    sampler.start()
    ms, sps = run(trainer, env, args.steps, 0, update_freq, batch_size, device, "BENCHMARK")
    sampler.stop()

    print("\n=== GPU UTILIZATION (during benchmark) ===")
    gpu_util_mean = sampler.report()
    if device == "cuda":
        print(f"  torch peak allocated: {torch.cuda.max_memory_allocated()/1024**2:6.0f} MiB")

    print("\n=== UPDATE: TRANSFER vs COMPUTE ===")
    transfer_vs_compute(trainer, device, batch_size)

    # Recompute CPU/GPU split for the verdict from the measured run phases.
    # (re-run cheaply is overkill; reuse the printed split via a short profile)
    cpu_ms, gpu_ms = _quick_split(trainer, env, batch_size, update_freq, device)
    verdict(gpu_util_mean, cpu_ms, gpu_ms)

    print(f"\nTIME PER STEP: {ms:.3f} ms   |   THROUGHPUT: {sps:.1f} steps/s")
    trainer.close()
    env.close()


def _quick_split(trainer, env, batch_size, update_freq, device, n=200):
    """A short, silent re-profile to get clean CPU-ms / GPU-ms for the verdict."""
    is_cuda = device == "cuda"
    cpu = gpu = 0.0
    state, _ = env.reset()
    for i in range(n):
        t0 = time.perf_counter()
        action = trainer.select_action(state)
        if is_cuda: torch.cuda.synchronize()
        t1 = time.perf_counter()
        next_state, reward, term, trunc, _ = env.step(action)
        trainer.replay_buffer.add(state, action, reward, next_state, float(term or trunc))
        trainer.train_step += 1
        state = next_state
        t2 = time.perf_counter()
        if trainer.train_step % update_freq == 0 and len(trainer.replay_buffer) >= batch_size:
            trainer.update()
            if is_cuda: torch.cuda.synchronize()
        t3 = time.perf_counter()
        gpu += (t1 - t0) + (t3 - t2)
        cpu += (t2 - t1)
        if term or trunc:
            state, _ = env.reset()
    return cpu / n * 1000, gpu / n * 1000


if __name__ == "__main__":
    main()
