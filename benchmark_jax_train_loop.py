"""
Light throughput check for the JAX-vectorised SAC training loop.

Builds the env + trainer straight from a train config (same `num_envs`, model,
opponent, reward, TD(λ), attn_impl, …) and runs **one** measured pass for a given
number of vector-steps, then reports transitions/sec. The pass uses the
`jax_env.actor` mode from the config (override with `--actor`):

  serial  — single-threaded select → step → add → update; also prints a per-phase
            breakdown (select / envstep / add / update) so you can see where the
            time goes.
  thread  — actor-learner threads (overlap CPU rollout with GPU update).
  mp_env  — env in a subprocess, pipelined with the learner's GPU update.

Usage
-----
    python benchmark_jax_train_loop.py                       # config's actor, 100 vsteps
    python benchmark_jax_train_loop.py --vsteps 200
    python benchmark_jax_train_loop.py --actor serial        # force a mode
    python benchmark_jax_train_loop.py --num-envs 256
"""
from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time

# Grow VRAM on demand instead of JAX's default 75 % grab, so the main process
# (torch) and the mp_env actor subprocess (its own JAX env) can coexist on one
# GPU.  Must be set BEFORE `import jax`.  Honoured if already set in the env.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import torch

from model.SAC import P_network, Q_network
from sac_train import SACTrainer
from jax_env import JaxVecEnvAdapter
# NB: load_config is imported lazily inside main() — importing train_orbit_wars
# at module scope pulls in agents.agent1, whose relative import breaks when the
# mp_env spawn child re-imports this module to reconstruct __main__.


# ─────────────────────────────────────────────────────────────────────────────
# GPU utilization sampler (nvidia-smi polled in a background thread)
# ─────────────────────────────────────────────────────────────────────────────

class GPUSampler(threading.Thread):
    """Polls nvidia-smi at ~10 Hz for gpu/mem utilization + mem used.

    Self-disables silently if nvidia-smi is unavailable.
    """
    QUERY = ("--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits")

    def __init__(self, period=0.1):
        super().__init__(daemon=True)
        self.period = period
        self._stop_evt = threading.Event()
        self.gpu_util, self.mem_used, self.mem_total = [], [], None
        self.available = self._probe()

    def _probe(self):
        try:
            subprocess.run(("nvidia-smi", *self.QUERY), capture_output=True,
                           text=True, timeout=5, check=True)
            return True
        except Exception:
            return False

    def run(self):
        if not self.available:
            return
        while not self._stop_evt.is_set():
            try:
                out = subprocess.run(("nvidia-smi", *self.QUERY), capture_output=True,
                                     text=True, timeout=5, check=True).stdout
                g, used, total = (float(x) for x in out.strip().splitlines()[0].split(","))
                self.gpu_util.append(g); self.mem_used.append(used); self.mem_total = total
            except Exception:
                pass
            self._stop_evt.wait(self.period)

    def stop(self):
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=2)

    def report(self):
        if not self.available or not self.gpu_util:
            print("  GPU util: (nvidia-smi unavailable)")
            return
        gu, used = np.array(self.gpu_util), np.array(self.mem_used)
        print(f"  GPU util : mean {gu.mean():4.0f}%  peak {gu.max():4.0f}%   "
              f"mem {used.max():.0f} / {self.mem_total:.0f} MiB")


# ─────────────────────────────────────────────────────────────────────────────
# Build env + trainer from a train config (mirrors train_orbit_wars.train_jax)
# ─────────────────────────────────────────────────────────────────────────────

def _build_env(config: dict, num_envs: int, num_players: int = 2) -> JaxVecEnvAdapter:
    env_cfg = config.get("environment", {})
    jax_cfg = config.get("jax_env",     {})
    return JaxVecEnvAdapter(
        num_envs        = num_envs,
        num_players     = num_players,
        episode_steps   = jax_cfg.get("episode_steps", 500),
        tanh_scale      = env_cfg.get("tanh_scale",    0.2),
        min_fleet_ships = env_cfg.get("min_fleet_ships", 3),
        reward_cfg      = config.get("reward", []),
        reward_type     = jax_cfg.get("reward_type",  "ship_advantage"),
        reward_scale    = jax_cfg.get("reward_scale", 0.01),
        win_bonus       = jax_cfg.get("win_bonus",    100.0),
        opponent        = env_cfg.get("opponent",     "random"),
    )


def _build_trainer(config: dict, env: JaxVecEnvAdapter, device: str) -> SACTrainer:
    t   = config.get("training",  {})
    m   = config.get("model",     {})
    tdl = config.get("td_lambda", {})
    net_kw = dict(
        state_dim       = JaxVecEnvAdapter.STATE_DIM,
        action_dim      = JaxVecEnvAdapter.ACTION_DIM,
        max_planets     = 40,
        max_fleets      = 200,
        d_model         = m.get("d_model", 128),
        dim_feedforward = m.get("ff_dim", 512),
        num_layers      = m.get("num_layers", 2),
        nhead           = m.get("num_heads", 4),
        attn_impl       = m.get("attn_impl", "torch"),
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
        env_stride         = env.num_envs if tdl.get("enabled", False) else 1,
        tb_log_every       = config.get("io", {}).get("tb_log_every", 25),
        compile_mode       = m.get("compile_mode", "default"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Serial pass (phase-profiled)
# ─────────────────────────────────────────────────────────────────────────────

def _run_serial(trainer, env, num_envs, warmup_steps, update_freq, grad_steps,
                batch_size, device, n_vsteps):
    """Single-threaded select→step→add→update loop. Returns (phases, wall, n_upds).

    warmup_steps=0 means updates start from the first vstep (buffer pre-filled).
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
        t0 = time.perf_counter()
        if trainer.train_step < warmup_steps:
            actions = np.stack([env.action_space.sample() for _ in range(num_envs)])
        else:
            actions = trainer.select_action_batch(obs)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        next_obs, rewards, dones, _, _ = env.step(actions)
        t2 = time.perf_counter()

        trainer.replay_buffer.add_batch(
            obs, actions, rewards, next_obs, dones.astype(np.float32))
        trainer.train_step += num_envs
        obs = next_obs
        t3 = time.perf_counter()

        if (trainer.train_step >= warmup_steps
                and len(trainer.replay_buffer) >= batch_size):
            upd_debt += num_envs / update_freq
            while upd_debt >= 1.0:
                for _ in range(grad_steps):
                    trainer.update()
                n_upds += 1
                upd_debt -= 1.0
            if is_cuda:
                torch.cuda.synchronize()
        t4 = time.perf_counter()

        phases["select"]  += t1 - t0
        phases["envstep"] += t2 - t1
        phases["add"]     += t3 - t2
        phases["update"]  += t4 - t3

    return phases, time.perf_counter() - wall0, n_upds


def _report_serial(phases, wall, n_vsteps, num_envs, n_upds, device, env_backend):
    n_tr = n_vsteps * num_envs
    tps  = n_tr / wall
    print(f"\n=== RESULT (serial) ===")
    print(f"  {n_vsteps} vsteps × {num_envs} envs = {n_tr} transitions | "
          f"{n_upds} updates | torch={device}")
    print(f"  {'TOTAL wall':9s}: {wall*1000:8.1f} ms   "
          f"({wall/n_tr*1000:.4f} ms/transition)   {tps:8.1f} trans/s")
    # envstep = JAX game step (on env_backend) + host-side obs extraction (CPU),
    # so it is tagged with the env's JAX backend rather than a flat "CPU".
    tag = {"select": "GPU", "envstep": f"{env_backend}+host", "add": "CPU", "update": "GPU"}
    for k in ("select", "envstep", "add", "update"):
        print(f"  {k:9s}: {phases[k]*1000:8.1f} ms   "
              f"({phases[k]/n_vsteps*1000:7.3f} ms/vstep)   "
              f"{phases[k]/wall*100:5.1f}%   [{tag[k]}]")
    dom = max(phases, key=phases.get)
    print(f"  dominant : {dom} ({phases[dom]/wall*100:.0f}% of wall)")
    return tps


# ─────────────────────────────────────────────────────────────────────────────
# Threaded (actor-learner) pass — overlaps CPU rollout with the GPU update
# ─────────────────────────────────────────────────────────────────────────────

def _run_threaded(trainer, env, num_envs, update_freq, grad_steps, batch_size,
                  device, n_vsteps, actor_sync_every, rollout_lead, env_backend):
    import copy
    is_cuda = (device == "cuda")

    def _orig(m):
        return getattr(m, "_orig_mod", m)

    actor_net = copy.deepcopy(_orig(trainer.policy_net)).to(device).eval()
    if is_cuda and hasattr(torch, "compile"):
        actor_net = torch.compile(actor_net)
    actor_lock = threading.Lock()

    def sync_actor():
        with actor_lock:
            _orig(actor_net).load_state_dict(_orig(trainer.policy_net).state_dict())

    sync_actor()

    def actor_select(states):
        st = trainer.state_preprocessor(trainer._to(torch.FloatTensor(states)))
        with actor_lock, torch.no_grad(), trainer._autocast():
            a, _ = trainer._sample(actor_net, st)
        return a.float().cpu().numpy()

    lead_cap = max(rollout_lead * num_envs, batch_size)
    stop = threading.Event()
    prod = {"v": 0}
    upds = {"n": 0}

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
            if len(trainer.replay_buffer) < batch_size:
                time.sleep(0.001)
                continue
            if upd >= int(trainer.train_step / update_freq * grad_steps):
                time.sleep(0.0005)
                continue
            trainer.update()
            upd += 1
            if upd - last_sync >= actor_sync_every:
                sync_actor()
                last_sync = upd
        upds["n"] = upd

    trainer.train_step = trainer._update_count = 0
    if is_cuda:
        torch.cuda.synchronize()
    wall0 = time.perf_counter()
    lt = threading.Thread(target=learner_worker, daemon=True)
    et = threading.Thread(target=env_worker, daemon=True)
    lt.start(); et.start(); et.join(); stop.set(); lt.join()
    if is_cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - wall0

    n_tr = n_vsteps * num_envs
    tps  = n_tr / wall
    print(f"\n=== RESULT (thread) ===")
    print(f"  {n_vsteps} vsteps × {num_envs} envs = {n_tr} transitions | "
          f"{upds['n']} updates | torch={device} env={env_backend}  [actor-learner threads]")
    print(f"  {'TOTAL wall':9s}: {wall*1000:8.1f} ms   "
          f"({wall/n_tr*1000:.4f} ms/transition)   {tps:8.1f} trans/s")
    return tps


# ─────────────────────────────────────────────────────────────────────────────
# Multiprocess actor (env in a subprocess) — real CPU/GPU overlap
# ─────────────────────────────────────────────────────────────────────────────

def _mp_env_worker(conn, config, num_envs, num_players):
    """Subprocess: own the JAX env, serve step requests over `conn`."""
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
                batch_size, device, n_vsteps, env_backend):
    import multiprocessing as mp
    is_cuda = (device == "cuda")
    ctx = mp.get_context("spawn")            # fork is unsafe once CUDA is init'd
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_mp_env_worker,
                       args=(child_conn, config, num_envs, num_players), daemon=True)
    proc.start()
    child_conn.close()
    obs = parent_conn.recv()

    # Warmup: fill buffer + JIT-compile env (worker) and update (here).
    for _ in range(trainer.batch_size // num_envs + 3):
        actions = trainer.select_action_batch(obs)
        parent_conn.send(actions)
        nobs, r, d = parent_conn.recv()
        trainer.replay_buffer.add_batch(obs, actions, r, nobs, d)
        trainer.train_step += num_envs
        obs = nobs
    for _ in range(3):
        if len(trainer.replay_buffer) >= batch_size:
            trainer.update()
    if is_cuda:
        torch.cuda.synchronize()

    # Measured pipelined pass: dispatch the next env step, then run the GPU update
    # for the data already in the buffer WHILE the actor subprocess steps.
    trainer.train_step = 0
    n_upds = 0
    actions = trainer.select_action_batch(obs)
    parent_conn.send(actions)
    prev_obs, prev_actions = obs, actions
    if is_cuda:
        torch.cuda.synchronize()
    wall0 = time.perf_counter()
    for t in range(n_vsteps):
        if len(trainer.replay_buffer) >= batch_size:
            trainer.update()
            n_upds += 1
        nobs, r, d = parent_conn.recv()
        trainer.replay_buffer.add_batch(prev_obs, prev_actions, r, nobs, d)
        trainer.train_step += num_envs
        next_actions = trainer.select_action_batch(nobs)
        if t < n_vsteps - 1:
            parent_conn.send(next_actions)
        prev_obs, prev_actions = nobs, next_actions
    if is_cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - wall0

    parent_conn.send(None)
    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()

    n_tr = n_vsteps * num_envs
    tps  = n_tr / wall
    print(f"\n=== RESULT (mp_env) ===")
    print(f"  {n_vsteps} vsteps × {num_envs} envs = {n_tr} transitions | "
          f"{n_upds} updates | torch={device} env={env_backend}  [mp actor subprocess]")
    print(f"  {'TOTAL wall':9s}: {wall*1000:8.1f} ms   "
          f"({wall/n_tr*1000:.4f} ms/transition)   {tps:8.1f} trans/s")
    return tps


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Light single-pass throughput check for the JAX SAC train loop")
    ap.add_argument("--config",      default="train.json",
                    help="train config JSON (default: train.json)")
    ap.add_argument("--vsteps",      type=int, default=100,
                    help="measured vector-steps (default: 100)")
    ap.add_argument("--num-envs",    type=int, default=None,
                    help="override jax_env.num_envs from config")
    ap.add_argument("--actor",       default=None, choices=["serial", "thread", "mp_env"],
                    help="override jax_env.actor from config")
    ap.add_argument("--num-players", type=int, default=2)
    ap.add_argument("--no-gpu-sample", action="store_true",
                    help="skip the nvidia-smi GPU-utilization sampler")
    args = ap.parse_args()

    from train_orbit_wars import load_config   # lazy: see module-top note
    config   = load_config(args.config)
    t_cfg    = config.get("training", {})
    jax_cfg  = config.get("jax_env",  {})

    device      = ("cuda" if torch.cuda.is_available()
                   and not config.get("execution", {}).get("cpu_force") else "cpu")
    env_backend = jax.default_backend().upper()           # GPU / CPU / TPU
    num_envs    = args.num_envs or jax_cfg.get("num_envs", 256)
    actor       = args.actor or jax_cfg.get("actor", "serial")
    update_freq = t_cfg.get("update_freq") or num_envs
    grad_steps  = t_cfg.get("gradient_steps", 1)
    batch_size  = t_cfg.get("batch_size",     64)
    warmup_steps = t_cfg.get("warmup_steps",  500)
    tdl_on      = config.get("td_lambda", {}).get("enabled", False)
    attn_impl   = config.get("model", {}).get("attn_impl", "torch")

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"Config        : {args.config}")
    print(f"torch device  : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"JAX env backend: {env_backend}")
    print(f"actor mode    : {actor}   opponent={config.get('environment', {}).get('opponent')}")
    print(f"num_envs={num_envs}  vsteps={args.vsteps}  batch_size={batch_size}  "
          f"update_freq={update_freq}  td_lambda={'on' if tdl_on else 'off'}  "
          f"attn_impl={attn_impl}")

    env     = _build_env(config, num_envs, args.num_players)
    trainer = _build_trainer(config, env, device)

    # ── Warmup (JIT compile + fill buffer) for the in-process modes ───────────
    # mp_env runs its own warmup against its subprocess env, so skip it here.
    if actor in ("serial", "thread"):
        warmup_vsteps = max(2, warmup_steps // num_envs + 2)
        print(f"Warming up ({warmup_vsteps} vsteps)...")
        _run_serial(trainer, env, num_envs, warmup_steps, update_freq, grad_steps,
                    batch_size, device, warmup_vsteps)

    # ── Measured pass (single mode) ───────────────────────────────────────────
    sampler = None if args.no_gpu_sample else GPUSampler()
    if sampler:
        sampler.start()

    if actor == "serial":
        phases, wall, n_upds = _run_serial(
            trainer, env, num_envs, 0, update_freq, grad_steps,
            batch_size, device, args.vsteps)
        if sampler:
            sampler.stop()
        _report_serial(phases, wall, args.vsteps, num_envs, n_upds, device, env_backend)
    elif actor == "thread":
        _run_threaded(trainer, env, num_envs, update_freq, grad_steps, batch_size,
                      device, args.vsteps, jax_cfg.get("actor_sync_every", 2),
                      jax_cfg.get("rollout_lead", 8), env_backend)
        if sampler:
            sampler.stop()
    else:  # mp_env
        _run_mp_env(trainer, config, num_envs, args.num_players, update_freq,
                    grad_steps, batch_size, device, args.vsteps, env_backend)
        if sampler:
            sampler.stop()

    if sampler:
        sampler.report()
    if device == "cuda":
        print(f"  torch mem: {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB peak")

    env.close()
    trainer.close()


if __name__ == "__main__":
    main()
