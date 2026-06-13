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
import copy
import os
import subprocess
import threading
import time
from collections import deque

# Grow VRAM on demand instead of JAX's default 75 % grab, so the main process
# (torch + JaxReplayBuffer) and the --mp-env actor subprocess (its own JAX env)
# can coexist on one GPU.  Must be set BEFORE `import jax`.  Honoured if already
# set in the environment.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import torch

from model.SAC import P_network, Q_network
from sac_train import SACTrainer
from env.orbit_wars import OrbitWarsEnv
from jax_env import JaxVecEnvAdapter
# NB: load_config is imported lazily inside main() — importing train_orbit_wars
# at module scope pulls in agents.agent1, whose relative import breaks when the
# --mp-env spawn child re-imports this module to reconstruct __main__.


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
        state_dim       = OrbitWarsEnv.STATE_DIM,
        action_dim      = OrbitWarsEnv.ACTION_DIM,
        max_planets     = 40,
        max_fleets      = 200,
        d_model         = m.get("d_model", 128),
        dim_feedforward = m.get("ff_dim", 512),
        num_layers      = m.get("num_layers", 2),
        nhead           = m.get("num_heads", 4),
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
        tb_log_every       = config.get("io", {}).get("tb_log_every", 25),
        compile_mode       = m.get("compile_mode", "default"),
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
    profile_update: bool = False,
) -> dict:
    """Run one benchmark pass and return timing statistics.

    warmup_steps=0 means gradient updates start from the first vector step
    (use this for the measured pass when the buffer is already pre-filled).

    When jax_buffer=True, a JAX sync is inserted after add_batch so that the
    async GPU scatter completes before the timer for the 'add' phase stops.
    Without it the scatter would bleed into the 'update' phase wall time,
    making 'add' look free and 'update' look artificially slow.
    """
    is_cuda     = (device == "cuda")
    phases      = {k: 0.0 for k in ("select", "envstep", "add", "update")}
    _upd_phases = {k: 0.0 for k in ("sample", "q_target", "q1", "q2", "pi", "tail")}
    _n_upd_calls = 0
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
                    # Sub-phase profiling inserts ~7 cuda.synchronize() per update,
                    # which serialises the GPU and both inflates the update time and
                    # tanks measured util — so it is OFF by default; the coarse
                    # select/envstep/add/update timing below stays accurate.  Enable
                    # with --profile-update only when you want the sub-phase split.
                    _ures = trainer.update(_profile=(is_cuda and profile_update))
                    if _ures and "_phases" in _ures:
                        for _k, _v in _ures["_phases"].items():
                            _upd_phases[_k] += _v
                    _n_upd_calls += 1
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
        _print_report(label, phases, wall, n_vsteps, n_tr, n_upds, device, jax_buffer,
                      upd_phases=_upd_phases if is_cuda and _n_upd_calls > 0 else None,
                      n_upd_calls=_n_upd_calls)

    # With JaxReplayBuffer, 'add' is GPU work (JAX scatter); attribute it to
    # the GPU side so the bottleneck verdict is accurate.
    if jax_buffer:
        cpu_ms = phases["envstep"] / n_vsteps * 1000
        gpu_ms = (phases["select"] + phases["add"] + phases["update"]) / n_vsteps * 1000
    else:
        cpu_ms = (phases["envstep"] + phases["add"]) / n_vsteps * 1000
        gpu_ms = (phases["select"]  + phases["update"]) / n_vsteps * 1000

    return {"tps": n_tr / wall, "cpu_ms": cpu_ms, "gpu_ms": gpu_ms,
            "wall": wall, "n_updates": n_upds, "upd_phases": _upd_phases,
            "n_upd_calls": _n_upd_calls}


# ─────────────────────────────────────────────────────────────────────────────
# Threaded (actor-learner) throughput runner
#
# Mirrors train_orbit_wars.train_jax's threaded loop — env-rollout thread +
# learner thread sharing the trainer/buffer with a separate actor net synced
# every `actor_sync_every` UPDATES.  The serial _run() measures per-phase cost
# but cannot show CPU/GPU overlap; this measures the real end-to-end throughput
# the threaded training loop achieves, so the two trans/s numbers are directly
# comparable.
# ─────────────────────────────────────────────────────────────────────────────

def _run_threaded(
    trainer, env, num_envs, update_freq, grad_steps, batch_size, device,
    n_vsteps, actor_sync_every, rollout_lead, label,
) -> dict:
    is_cuda = (device == "cuda")

    def _orig(m):
        return getattr(m, "_orig_mod", m)

    # Separate actor net (env thread reads it; learner syncs into it).
    actor_net = copy.deepcopy(_orig(trainer.policy_net)).to(device).eval()
    if is_cuda and hasattr(torch, "compile"):
        actor_net = torch.compile(actor_net)
    actor_lock = threading.Lock()

    def sync_actor():
        with actor_lock:
            _orig(actor_net).load_state_dict(_orig(trainer.policy_net).state_dict())

    sync_actor()

    def actor_select(states):
        st = trainer._to(torch.FloatTensor(states))
        st = trainer.state_preprocessor(st)
        with actor_lock, torch.no_grad(), trainer._autocast():
            a, _ = trainer._sample(actor_net, st)
        return a.float().cpu().numpy()

    # Buffer is pre-filled by the warmup pass, so updates start immediately
    # (warmup budget = 0 here).
    lead_cap = max(rollout_lead * num_envs, batch_size)
    stop  = threading.Event()
    prod  = {"v": 0}
    upds  = {"n": 0}

    def env_worker():
        obs, _ = env.reset()
        while prod["v"] < n_vsteps and not stop.is_set():
            expected = int(trainer._update_count / grad_steps * update_freq)
            while (trainer.train_step - expected > lead_cap) and not stop.is_set():
                time.sleep(0.0005)
                expected = int(trainer._update_count / grad_steps * update_freq)
            actions = actor_select(obs)
            nobs, r, d, _, _ = env.step(actions)
            trainer.replay_buffer.add_batch(obs, actions, r, nobs, d.astype(np.float32))
            trainer.train_step += num_envs
            obs = nobs
            prod["v"] += 1
        stop.set()

    def learner_worker():
        upd = last_sync = 0
        while not stop.is_set():
            ts = trainer.train_step
            if len(trainer.replay_buffer) < batch_size:
                time.sleep(0.001)
                continue
            target = int(ts / update_freq * grad_steps)
            if upd >= target:
                time.sleep(0.0005)
                continue
            trainer.update()
            upd += 1
            if upd - last_sync >= actor_sync_every:
                sync_actor()
                last_sync = upd
        upds["n"] = upd

    # Reset both counters so the ratio math (env: _update_count→expected;
    # learner: train_step→target) is measured over just this pass and not
    # offset by the warmup/serial passes that ran before it.
    trainer.train_step    = 0
    trainer._update_count = 0
    if is_cuda:
        torch.cuda.synchronize()
    wall0 = time.perf_counter()
    learner_t = threading.Thread(target=learner_worker, daemon=True)
    env_t     = threading.Thread(target=env_worker,     daemon=True)
    learner_t.start(); env_t.start()
    env_t.join()
    stop.set()
    learner_t.join()
    if is_cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - wall0

    n_tr = n_vsteps * num_envs
    tps  = n_tr / wall
    print(f"\n=== {label} ===")
    print(f"  {n_vsteps} vsteps × {num_envs} envs = {n_tr} transitions"
          f" | {upds['n']} gradient updates | device={device}  [actor-learner threads]")
    print(f"  {'TOTAL wall':20s}: {wall*1000:9.1f} ms  "
          f"({wall/n_tr*1000:.4f} ms/transition)  {tps:9.1f} trans/s")
    return {"tps": tps, "wall": wall, "n_updates": upds["n"]}


# ─────────────────────────────────────────────────────────────────────────────
# Multiprocess actor (env in a subprocess) — REAL CPU/GPU overlap
#
# The threaded actor-learner above is ~5x SLOWER because the env's host-side
# obs-extraction (np.asarray off ~20 device arrays = the ~35 ms 'envstep') holds
# the GIL, so it cannot overlap the GPU update running in another thread.  A
# separate PROCESS has its own GIL, so the env step truly runs concurrently with
# the learner's gradient update.
#
# Pipeline (one transition lag): the learner dispatches update() for the data it
# already has WHILE the actor subprocess steps the env for the actions just sent.
# Critical path/vstep collapses from select+envstep+add+update (~92 ms) toward
# max(envstep, update)+select+add+IPC.
# ─────────────────────────────────────────────────────────────────────────────

def _mp_env_worker(conn, config, num_envs, num_players):
    """Subprocess entry point: own the JAX env, serve step requests over `conn`.

    Protocol: send initial reset obs once, then loop {recv actions → send
    (next_obs, rewards, dones)} until a None sentinel arrives.
    """
    env = _build_env(config, num_envs, num_players)
    obs, _ = env.reset()
    conn.send(obs)
    try:
        while True:
            actions = conn.recv()
            if actions is None:
                break
            nobs, r, d, _, _ = env.step(actions)
            conn.send((nobs, r, d.astype(np.float32)))
    finally:
        env.close()
        conn.close()


def _run_mp_env(trainer, config, num_envs, num_players, update_freq, grad_steps,
                batch_size, device, n_vsteps, label):
    import multiprocessing as mp

    is_cuda = (device == "cuda")
    ctx = mp.get_context("spawn")            # fork is unsafe once CUDA is init'd
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_mp_env_worker,
                       args=(child_conn, config, num_envs, num_players),
                       daemon=True)
    proc.start()
    child_conn.close()                        # parent keeps only its end

    obs = parent_conn.recv()                  # initial reset obs from the actor

    # ── Warmup: fill the buffer + JIT-compile env (worker) and update (here) ──
    warmup_vsteps = trainer.batch_size // num_envs + 3
    for _ in range(warmup_vsteps):
        actions = trainer.select_action_batch(obs)
        parent_conn.send(actions)
        nobs, r, d = parent_conn.recv()
        trainer.replay_buffer.add_batch(obs, actions, r, nobs, d)
        trainer.train_step += num_envs
        obs = nobs
    for _ in range(3):                        # compile the update graph
        if len(trainer.replay_buffer) >= batch_size:
            trainer.update()
    if is_cuda:
        torch.cuda.synchronize()

    # ── Measured pipelined pass ──────────────────────────────────────────────
    trainer.train_step = 0
    n_upds = 0
    # Prime: dispatch the first env step so the actor is busy during update #0.
    actions = trainer.select_action_batch(obs)
    parent_conn.send(actions)
    prev_obs, prev_actions = obs, actions

    if is_cuda:
        torch.cuda.synchronize()
    wall0 = time.perf_counter()
    for t in range(n_vsteps):
        # GPU update overlaps the actor stepping `prev_actions` (CPU, other proc)
        if len(trainer.replay_buffer) >= batch_size:
            trainer.update()
            n_upds += 1
        nobs, r, d = parent_conn.recv()       # result for prev_actions
        trainer.replay_buffer.add_batch(prev_obs, prev_actions, r, nobs, d)
        trainer.train_step += num_envs
        next_actions = trainer.select_action_batch(nobs)
        if t < n_vsteps - 1:
            parent_conn.send(next_actions)    # actor starts the next step now
        prev_obs, prev_actions = nobs, next_actions
    if is_cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - wall0

    parent_conn.send(None)                    # stop the actor
    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()

    n_tr = n_vsteps * num_envs
    tps  = n_tr / wall
    print(f"\n=== {label} ===")
    print(f"  {n_vsteps} vsteps × {num_envs} envs = {n_tr} transitions"
          f" | {n_upds} gradient updates | device={device}  [mp actor subprocess]")
    print(f"  {'TOTAL wall':20s}: {wall*1000:9.1f} ms  "
          f"({wall/n_tr*1000:.4f} ms/transition)  {tps:9.1f} trans/s")
    return {"tps": tps, "wall": wall, "n_updates": n_upds}


def _print_report(label, phases, wall, n_vsteps, n_tr, n_upds, device,
                  jax_buffer: bool = False,
                  upd_phases: dict | None = None, n_upd_calls: int = 0):
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

    if upd_phases is not None and n_upd_calls > 0:
        total = sum(upd_phases.values())
        print(f"\n  update sub-phases  ({n_upd_calls} gradient calls; "
              f"sync between phases adds ~5 µs overhead each):")
        labels = {
            "sample":   "sample+H2D",
            "q_target": "q_target (no_grad)",
            "q1":       "q1 fwd+bwd+optim",
            "q2":       "q2 fwd+bwd+optim",
            "pi":       "pi  fwd+bwd+optim",
            "tail":     "polyak + alpha",
        }
        for k, v in upd_phases.items():
            ms = v / n_upd_calls * 1000
            pct = v / max(total, 1e-9) * 100
            print(f"    {labels.get(k, k):22s}: {ms:7.3f} ms/call  ({pct:5.1f}%)")
        print(f"    {'── total':22s}: {total/n_upd_calls*1000:.3f} ms/call  "
              f"[actual update={phases['update']/n_vsteps*1000:.3f} ms/vstep]")


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
    ap.add_argument("--threaded",     action="store_true",
                    help="also run an actor-learner THREADED pass and report its "
                         "end-to-end trans/s next to the serial number (the serial "
                         "phase profile cannot show CPU/GPU overlap)")
    ap.add_argument("--mp-env",       action="store_true",
                    help="also run a MULTIPROCESS-actor pass (env in a subprocess) "
                         "that truly overlaps the CPU env.step with the GPU update, "
                         "and report its end-to-end trans/s vs serial")
    ap.add_argument("--profile-update", action="store_true",
                    help="break the update into sub-phases (sample/q_target/q1/q2/pi/"
                         "tail).  Adds ~7 cuda syncs/update that distort total time + "
                         "util, so leave OFF for representative throughput numbers.")
    args = ap.parse_args()

    from train_orbit_wars import load_config   # lazy: see module-top note
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
                   jax_buffer=jax_buffer, profile_update=args.profile_update)
        sampler.stop()

        print("\n=== GPU UTILIZATION (serial) ===")
        gpu_util = sampler.report()
        if device == "cuda":
            print(f"  torch peak allocated: {torch.cuda.max_memory_allocated()/1024**2:6.0f} MiB")
        verdict(gpu_util, res["cpu_ms"], res["gpu_ms"], jax_buffer=jax_buffer)

        # ── Threaded (actor-learner) pass ─────────────────────────────────────
        if args.threaded:
            sync_every   = jax_cfg.get("actor_sync_every", 2)
            rollout_lead = jax_cfg.get("rollout_lead", 8)
            t_sampler = GPUSampler(); t_sampler.start()
            tres = _run_threaded(
                trainer, env, num_envs, update_freq, grad_steps, batch_size,
                device, n_vsteps=args.steps, actor_sync_every=sync_every,
                rollout_lead=rollout_lead,
                label=f"JAX THREADED  num_envs={num_envs}  {args.num_players}p")
            t_sampler.stop()
            print("\n=== GPU UTILIZATION (threaded) ===")
            t_gpu = t_sampler.report()
            speedup = tres["tps"] / res["tps"] if res["tps"] else float("nan")
            s_util = f"  (GPU util {gpu_util:.0f}%)" if gpu_util is not None else ""
            t_util = f"  (GPU util {t_gpu:.0f}%)"    if t_gpu    is not None else ""
            print("\n=== SERIAL vs THREADED ===")
            print(f"  serial   : {res['tps']:9.1f} trans/s{s_util}")
            print(f"  threaded : {tres['tps']:9.1f} trans/s{t_util}")
            print(f"  speedup  : {speedup:.2f}x  (overlap of CPU env.step with GPU update)")

        # ── Multiprocess actor pass (env subprocess, real CPU/GPU overlap) ────
        if args.mp_env:
            m_sampler = GPUSampler(); m_sampler.start()
            mres = _run_mp_env(
                trainer, config, num_envs, args.num_players, update_freq,
                grad_steps, batch_size, device, n_vsteps=args.steps,
                label=f"JAX MP-ENV  num_envs={num_envs}  {args.num_players}p")
            m_sampler.stop()
            print("\n=== GPU UTILIZATION (mp-env) ===")
            m_gpu = m_sampler.report()
            speedup = mres["tps"] / res["tps"] if res["tps"] else float("nan")
            s_util = f"  (GPU util {gpu_util:.0f}%)" if gpu_util is not None else ""
            m_util = f"  (GPU util {m_gpu:.0f}%)"    if m_gpu    is not None else ""
            print("\n=== SERIAL vs MP-ENV ===")
            print(f"  serial : {res['tps']:9.1f} trans/s{s_util}")
            print(f"  mp-env : {mres['tps']:9.1f} trans/s{m_util}")
            print(f"  speedup: {speedup:.2f}x  (CPU env.step overlapped with GPU update)")

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
