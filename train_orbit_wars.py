"""
SAC training for Kaggle Orbit Wars using the parallel JAX environment backend.

Runs N parallel JAX environments (VectorizedEnv) — all simulation on CPU/GPU
via XLA.  Opponents: random / greedy / mixed (curriculum-annealed).  Reward is
a per-step ship-advantage delta + win/loss bonus (configure via ``jax_env``).

Key config section::

    "jax_env": {
        "num_envs":       256,
        "num_players":    2,
        "episode_steps":  500,
        "reward_type":    "ship_advantage",
        "reward_scale":   0.01,
        "win_bonus":      100.0
    }

Usage
-----
    python train_orbit_wars.py                          # uses train.json
    python train_orbit_wars.py --config my_config.json  # custom config

All hyperparameters live in the JSON config file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import subprocess
import sys
import threading
import time
from collections import deque

# Grow VRAM on demand instead of JAX's default 75 % grab, so that the learner
# process (torch) and the actor subprocess (its own JAX env, under
# jax_env.actor="mp_env") can share one GPU.  Must precede `import jax`.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import torch
import jax

from model.SAC      import P_network, Q_network
from model          import action_decoder as _action_decoder
from sac_train      import SACTrainer

try:
    from jax_env import JaxVecEnvAdapter
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


def _linear_schedule(episode: int, start: float, end: float, decay_episodes: int) -> float:
    """Linearly interpolate from `start` → `end` over `decay_episodes` episodes."""
    frac = min(1.0, max(0.0, episode / max(1, decay_episodes - 1)))
    return start + (end - start) * frac


def _exponential_schedule(episode: int, start: float, end: float, decay_episodes: int) -> float:
    """Exponentially decay from `start` toward `end`.

    The decay constant is chosen so that at `episode = decay_episodes` the value
    is within 1% of `end` (i.e. exp(-k * decay_episodes) = 0.01 → k = log(100) / decay_episodes).
    """
    k = math.log(100.0) / max(1, decay_episodes)
    return end + (start - end) * math.exp(-k * episode)


def _make_schedule(schedule_type: str):
    """Return the schedule function for the given type ('linear' or 'exponential')."""
    if schedule_type == "exponential":
        return _exponential_schedule
    if schedule_type == "linear":
        return _linear_schedule
    raise ValueError(f"Unknown schedule type: {schedule_type!r}. Use 'linear' or 'exponential'.")


# ─────────────────────────────────────────────────────────────────────────────
# Multiprocess actor worker (env in its own process) — see train_jax(actor=mp_env)
#
# The threaded actor-learner overlaps nothing: the JAX env's host-side obs
# extraction holds the GIL, so it serialises against the GPU update on the other
# thread (measured ~5x SLOWER than serial).  A separate PROCESS has its own GIL,
# so env.step genuinely runs concurrently with the learner's gradient update.
# This worker owns the JAX env(s); the learner (main process) keeps ALL the
# bookkeeping (episodes, logging, checkpoints, rendering, update scheduling).
# ─────────────────────────────────────────────────────────────────────────────

def _mp_env_actor_worker(conn, env_kw, num_envs_2p, num_envs_4p, capture_render):
    """Subprocess: own env_2p (+env_4p), serve step requests over `conn`.

    Protocol (env-light): send the initial ``(obs, decode_state)`` once, then loop
    {recv (p0_engine, mix_ratio)  →  step_engine  →  send (next_obs, rewards,
    dones, wons, fleets_sent, n_fleets, state0, decode_state)} until a None
    sentinel arrives.  ``p0_engine`` is the learner-decoded player-0 engine action
    [B, NET_MP, 2]=(angle, ship-count); ``decode_state`` is the post-step planet
    state the learner needs to decode the NEXT action; ``mix_ratio`` is the current
    opponent="mixed" random ratio (or None); ``state0`` is the env_2p[0] pre-step
    GameState snapshot (for rendering) or None.
    """
    env_2p = JaxVecEnvAdapter(num_envs=num_envs_2p, num_players=2, **env_kw)
    env_4p = (JaxVecEnvAdapter(num_envs=num_envs_4p, num_players=4, **env_kw)
              if num_envs_4p > 0 else None)

    def _combined_dstate():
        """Concatenated planet decode-state across the 2p (+4p) envs."""
        d2 = env_2p.decode_state()
        if env_4p is None:
            return d2
        d4 = env_4p.decode_state()
        return {k: np.concatenate([d2[k], d4[k]], axis=0) for k in d2}

    obs_2p, _ = env_2p.reset()
    if env_4p is not None:
        obs_4p, _ = env_4p.reset()
        obs = np.concatenate([obs_2p, obs_4p], axis=0)
    else:
        obs = obs_2p
    # Initial obs + decode-state so the learner can decode the first action.
    conn.send((obs, _combined_dstate()))

    try:
        while True:
            msg = conn.recv()
            if msg is None:
                break
            p0_engine, mix_ratio = msg              # player-0 ENGINE action
            if mix_ratio is not None:
                env_2p.set_mixed_random_ratio(mix_ratio)
                if env_4p is not None:
                    env_4p.set_mixed_random_ratio(mix_ratio)
            state0 = None
            if capture_render:
                state0 = jax.tree_util.tree_map(
                    lambda x: np.asarray(x[0]), env_2p._state)
            if env_4p is not None:
                n2, r2, d2, _, w2 = env_2p.step_engine(p0_engine[:num_envs_2p])
                n4, r4, d4, _, w4 = env_4p.step_engine(p0_engine[num_envs_2p:])
                next_obs = np.concatenate([n2, n4], axis=0)
                rewards  = np.concatenate([r2, r4])
                dones    = np.concatenate([d2, d4])
                wons     = np.concatenate([w2, w4])
                fleets_sent = env_2p.last_fleets_sent + env_4p.last_fleets_sent
                n_fleets    = max(env_2p.last_n_fleets, env_4p.last_n_fleets)
            else:
                next_obs, rewards, dones, _, wons = env_2p.step_engine(p0_engine)
                fleets_sent = env_2p.last_fleets_sent
                n_fleets    = env_2p.last_n_fleets
            conn.send((next_obs, rewards, dones, wons,
                       fleets_sent, n_fleets, state0, _combined_dstate()))
    finally:
        env_2p.close()
        if env_4p is not None:
            env_4p.close()
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# JAX vectorised training loop
# ─────────────────────────────────────────────────────────────────────────────

def _log_vstep(vstep, num_episodes, episodes_done, episode_rewards,
               episode_wins, trainer, t0):
    """Console progress line for the JAX backend, keyed by vector-step.

    Reward / win-rate are averaged over the last 50 completed episodes — a fixed
    window, since the log cadence is now vector-steps rather than episodes (so a
    `[-log_interval:]` slice would no longer mean "the last log_interval episodes").
    """
    recent_r = episode_rewards[-50:]
    recent_w = episode_wins[-50:]
    avg_reward = float(np.mean(recent_r)) if recent_r else 0.0
    win_rate   = float(np.mean(recent_w)) * 100 if recent_w else 0.0
    elapsed    = time.perf_counter() - t0
    print(
        f"vs {vstep:>6} | Ep {episodes_done:>5}/{num_episodes} | "
        f"Avg(50ep): {avg_reward:+8.3f} | "
        f"Win%: {win_rate:5.1f} | "
        f"Buffer: {len(trainer.replay_buffer):>7} | "
        f"Steps: {trainer.train_step:>9} | "
        f"Upd: {trainer._update_count:>7} | "
        f"{elapsed:.0f}s"
    )


def train_jax(config: dict, MAX_PLANETS: int = 40, MAX_FLEETS: int = 100) -> list[float]:
    """Train SAC using the parallel JAX environment backend.

    Runs until ``num_episodes`` episodes have completed across all envs.
    Each vector-step adds ``num_envs`` transitions to the replay buffer.

    4-player support: ``ratio_4p`` fraction of envs run 4-player games; the
    remainder run 2-player games.  Both sets share the same replay buffer and
    policy network.

    Opponent modes (``environment.opponent`` in config):
      "random"     — uniform random actions for all opponent slots
      "greedy"     — vectorised greedy (score-based target + lead-intercept aim)
      "agent1"     — fuller vectorised port of agents/agent1.py (defence +
                     accurate-ship-count attack + sun avoidance)
      "mixed"      — per-step blend of "random" and "greedy" (curriculum-annealed)
      "self_play"  — same policy network for all players

    Parameters
    ----------
    config          : full JSON config dict
    MAX_PLANETS, MAX_FLEETS : network shape constants
    """
    if not _JAX_AVAILABLE:
        raise RuntimeError(
            "JAX backend requested but jax / jax_env could not be imported. "
            "Install JAX: https://github.com/google/jax#installation"
        )

    train_cfg  = config.get("training",    {})
    env_cfg    = config.get("environment", {})
    io_cfg     = config.get("io",          {})
    exec_cfg   = config.get("execution",   {})
    model_cfg  = config.get("model",       {})
    jax_cfg    = config.get("jax_env",     {})
    tdl_cfg    = config.get("td_lambda",   {})

    cpu_force = exec_cfg.get("cpu_force", False)
    device    = "cuda" if torch.cuda.is_available() and not cpu_force else "cpu"
    print(f"Device: {device}")

    ckpt_dir   = io_cfg.get("ckpt_dir",   "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    buffer_path = os.path.join(ckpt_dir, "replay_buffer.npz")

    num_episodes   = train_cfg.get("num_episodes",   250)
    max_steps      = train_cfg.get("max_steps",      500)
    warmup_steps   = train_cfg.get("warmup_steps",   500)
    update_freq    = train_cfg.get("update_freq",    4)

    gradient_steps = train_cfg.get("gradient_steps", 1)
    batch_size     = train_cfg.get("batch_size",     64)

    # Console-log / checkpoint / render cadence is measured in VECTOR-STEPS (one
    # env-step batch of num_envs transitions), not episodes. With num_envs envs
    # stepped together, episodes finish in bursts of ~num_envs at the same
    # vector-step, so an episode-count interval smaller than num_envs can never
    # fire at the configured rate. The vector-step counter ticks by exactly 1 per
    # iteration, so e.g. save_interval=100 reliably checkpoints every 100 vsteps.
    log_interval    = io_cfg.get("log_interval",    1)
    save_interval   = io_cfg.get("save_interval",   100)
    render_interval = io_cfg.get("render_interval", 0)   # 0 = disabled
    render_dir      = io_cfg.get("render_dir",      "replays")
    # TensorBoard: the JAX loop logs per-step (reward/fleets/env) and per-episode
    # (win-rate/length) scalars plus the trainer's gradient-update scalars. When
    # io.log_dir is left null we default it to a timestamped runs/ directory so
    # logging is on by default; gradient-update scalars (which each force a GPU
    # sync) are throttled by tb_log_every to keep them out of the hot path.
    log_dir         = io_cfg.get("log_dir")
    if log_dir is None:
        log_dir = os.path.join("runs", f"jax_{time.strftime('%Y%m%d_%H%M%S')}")
    tb_log_every    = io_cfg.get("tb_log_every", 25)

    # ── JAX env config ────────────────────────────────────────────────────────
    num_envs      = jax_cfg.get("num_envs",      256)
    episode_steps = jax_cfg.get("episode_steps", max_steps)
    ship_speed    = jax_cfg.get("ship_speed",    6.0)
    comet_speed   = jax_cfg.get("comet_speed",   4.0)
    reward_type   = jax_cfg.get("reward_type",   "ship_advantage")
    reward_scale  = jax_cfg.get("reward_scale",  0.01)
    win_bonus     = jax_cfg.get("win_bonus",     100.0)
    tanh_scale    = env_cfg.get("tanh_scale",    0.2)
    min_fleet_ships = env_cfg.get("min_fleet_ships", 3)
    # Rollout/learner overlap mode (jax_env.actor, default falls back to the
    # legacy jax_env.threaded bool):
    #   "mp_env"  — env in a SEPARATE PROCESS, learner in the main process; the
    #               only mode that truly overlaps the CPU env.step with the GPU
    #               update (~1.5x over serial).  Action selection stays on the
    #               learner so the policy is always current.  Not for self_play
    #               (opponent policy lives inside the env process).
    #   "thread"  — legacy actor-learner threads (GIL-bound; ~5x SLOWER, kept for
    #               reference / self_play).
    #   "serial"  — single-threaded loop.
    actor_sync_every = jax_cfg.get("actor_sync_every", 2)   # gradient UPDATES between actor weight syncs
    rollout_lead    = jax_cfg.get("rollout_lead", 8)        # max vsteps the env may lead the learner
    opponent = env_cfg.get("opponent", "random")

    _actor_default = "thread" if jax_cfg.get("threaded", True) else "serial"
    actor_mode = jax_cfg.get("actor", _actor_default)
    if actor_mode == "mp_env" and opponent == "self_play":
        print("actor='mp_env' is unsupported with opponent='self_play' "
              "(opponent policy lives in the env process) — falling back to 'thread'.")
        actor_mode = "thread"
    mp_env   = (actor_mode == "mp_env")
    threaded = (actor_mode == "thread")
    ratio_4p = env_cfg.get("ratio_4p", 0.0)

    # ── Mixed-opponent curriculum (opponent="mixed") ──────────────────────────
    # Each opponent independently acts randomly with probability mixed_random_ratio
    # (else greedy) per step. The ratio is annealed start→end over
    # mixed_random_ratio_decay EPISODES (default: num_episodes).
    mixed_send_prob = env_cfg.get("mixed_send_prob", 0.3)
    curr_cfg        = config.get("curriculum", {})
    mix_ratio_start = curr_cfg.get("mixed_random_ratio_start", 1.0)
    mix_ratio_end   = curr_cfg.get("mixed_random_ratio_end",   0.0)
    mix_ratio_decay = curr_cfg.get("mixed_random_ratio_decay")
    mix_ratio_sched = _make_schedule(
        curr_cfg.get("mixed_random_ratio_schedule", "linear"))
    mix_decay_eps   = mix_ratio_decay if mix_ratio_decay is not None else num_episodes
    is_mixed        = (opponent == "mixed")

    # ── Split envs between 2p and 4p ─────────────────────────────────────────
    num_envs_4p = int(num_envs * ratio_4p) if ratio_4p > 0 else 0
    num_envs_2p = num_envs - num_envs_4p

    if update_freq is None:
        update_freq = num_envs

    # reward_cfg passes the "reward" list to the adapter; reward_type/scale/win_bonus
    # are the fallback when no "reward" list is given.
    _reward_cfg = config.get("reward") or []
    _env_kw = dict(
        episode_steps=episode_steps,
        ship_speed=ship_speed,
        comet_speed=comet_speed,
        tanh_scale=tanh_scale,
        min_fleet_ships=min_fleet_ships,
        reward_type=reward_type,
        reward_scale=reward_scale,
        win_bonus=win_bonus,
        reward_cfg=_reward_cfg,
        opponent=opponent,
        mixed_random_ratio=mix_ratio_start,
        mixed_send_prob=mixed_send_prob,
    )
    env_2p = JaxVecEnvAdapter(num_envs=num_envs_2p, num_players=2, **_env_kw)
    env_4p = (JaxVecEnvAdapter(num_envs=num_envs_4p, num_players=4, **_env_kw)
              if num_envs_4p > 0 else None)

    tag = (f"{num_envs_2p}×2p"
           + (f" + {num_envs_4p}×4p" if env_4p else ""))
    _reward_names = [c.get("scheme") for c in _reward_cfg] if _reward_cfg else [reward_type]
    print(
        f"JAX backend: {tag} | opponent={opponent} | reward={_reward_names}"
    )
    if is_mixed:
        print(f"Mixed opponent: random_ratio {mix_ratio_start:.2f} → "
              f"{mix_ratio_end:.2f} over {mix_decay_eps} episodes "
              f"({curr_cfg.get('mixed_random_ratio_schedule', 'linear')}), "
              f"send_prob={mixed_send_prob:.2f}")
    print(f"TensorBoard: logging to {log_dir} (run: tensorboard --logdir runs)")

    def _current_mix_ratio(eps_done: int) -> float:
        """Scheduled opponent random-ratio at the given completed-episode count."""
        return mix_ratio_sched(eps_done, mix_ratio_start, mix_ratio_end, mix_decay_eps)

    # ── env-light decode: turn raw player-0 policy output into engine actions ──
    # The wedge + lead-intercept decode runs HERE in the learner (torch, on the
    # training device) rather than inside the env, so the env consumes engine-ready
    # (angle, ship-count) actions and the same module backs the standalone agent.
    def _p0_from_dstate(raw, dstate):
        """raw [n, NET_MP, 4] (numpy) + planet dstate → engine [n, NET_MP, 2] (numpy)."""
        a = torch.as_tensor(np.asarray(raw), dtype=torch.float32, device=device)
        out = _action_decoder.decode(
            a,
            torch.as_tensor(dstate["owner"],  device=device),
            torch.as_tensor(dstate["x"],      dtype=torch.float32, device=device),
            torch.as_tensor(dstate["y"],      dtype=torch.float32, device=device),
            torch.as_tensor(dstate["r"],      dtype=torch.float32, device=device),
            torch.as_tensor(dstate["ships"],  dtype=torch.float32, device=device),
            torch.as_tensor(dstate["active"], device=device),
            torch.as_tensor(dstate["omega"],  dtype=torch.float32, device=device),
            player=0, tanh_scale=tanh_scale,
        )
        p0 = torch.stack([out["angle"], out["num_ships"].float()], dim=-1)
        return p0.detach().cpu().numpy().astype(np.float32)

    def _p0_from_env(env, raw):
        """Decode player-0 engine action using `env`'s current planet state."""
        return _p0_from_dstate(raw, env.decode_state())

    # ── Networks ──────────────────────────────────────────────────────────────
    d_model = model_cfg.get("d_model", 128)
    net_kw  = dict(
        state_dim       = JaxVecEnvAdapter.STATE_DIM,
        action_dim      = JaxVecEnvAdapter.ACTION_DIM,
        max_planets     = MAX_PLANETS,
        max_fleets      = MAX_FLEETS,
        d_model         = d_model,
        dim_feedforward = model_cfg.get("ff_dim", 512),
        num_layers      = model_cfg.get("num_layers", 2),
        nhead           = model_cfg.get("num_heads", 4),
        # "flash" routes attention through SDPA (FlashAttention-2 on Ampere+,
        # safe fallback elsewhere); "torch" keeps the stock encoder.
        attn_impl       = model_cfg.get("attn_impl", "torch"),
    )
    policy_net = P_network(**net_kw)
    q1_net     = Q_network(**net_kw)
    q2_net     = Q_network(**net_kw)

    # ── SACTrainer ────────────────────────────────────────────────────────────
    lr          = train_cfg.get("lr",           3e-4)
    gamma       = train_cfg.get("gamma",        0.99)
    tau         = train_cfg.get("tau",          5e-3)
    alpha       = train_cfg.get("alpha",        0.2)
    auto_alpha  = train_cfg.get("auto_alpha",   False)
    target_ent  = train_cfg.get("target_entropy")
    buffer_size = train_cfg.get("buffer_size",  100_000)
    max_grad_norm = train_cfg.get("grad_clip",  1.0)

    use_lambda    = tdl_cfg.get("enabled",      False)
    lambda_return = tdl_cfg.get("lambda",       0.9)
    cache_size    = tdl_cfg.get("cache_size",   8000)
    block_size    = tdl_cfg.get("block_size",   50)
    refresh_freq  = tdl_cfg.get("refresh_freq", 1000)

    trainer = SACTrainer(
        env               = env_2p,   # reference env for obs/act shape
        policy_net        = policy_net,
        q1_net            = q1_net,
        q2_net            = q2_net,
        device            = device,
        learning_rate     = lr,
        gamma             = gamma,
        tau               = tau,
        alpha             = alpha,
        auto_alpha        = auto_alpha,
        target_entropy    = target_ent,
        replay_buffer_size= buffer_size,
        batch_size        = batch_size,
        max_grad_norm     = max_grad_norm,
        use_lambda_returns= use_lambda,
        lambda_return     = lambda_return,
        cache_size        = cache_size,
        block_size        = block_size,
        refresh_freq      = refresh_freq,
        # Each add_batch writes one row of `num_envs` env-interleaved transitions,
        # so env e's trajectory is at a stride of num_envs — TD(λ) needs this to
        # promote per-env (not cross-env) blocks into its λ-return cache.
        env_stride        = num_envs,
        log_dir           = log_dir,
        tb_log_every      = tb_log_every,
        compile_mode      = model_cfg.get("compile_mode", "default"),
    )

    resume = exec_cfg.get("resume")
    if resume:
        trainer.load_checkpoint(resume)
        trainer.load_replay_buffer(buffer_path)

    # Wire self-play policy after trainer is built so opponents use live weights
    if opponent == "self_play":
        env_2p.set_policy(trainer.select_action_batch)
        if env_4p is not None:
            env_4p.set_policy(trainer.select_action_batch)

    # ── Reset envs ────────────────────────────────────────────────────────────
    # In mp_env mode the actor SUBPROCESS owns + resets the envs and sends the
    # initial obs; resetting the main-process envs here would needlessly JIT and
    # allocate a second env copy, so skip it (obs_batch comes from the worker).
    if mp_env:
        obs_batch = None
    else:
        obs_2p, _ = env_2p.reset()
        if env_4p is not None:
            obs_4p, _ = env_4p.reset()
            obs_batch = np.concatenate([obs_2p, obs_4p], axis=0)
        else:
            obs_batch = obs_2p

    # ── Training state ────────────────────────────────────────────────────────
    episode_wins:    list[bool]  = []
    episode_rewards: list[float] = []
    reward_window: deque[float]  = deque(maxlen=100)
    fleets_window: deque[int]    = deque(maxlen=50)
    max_fleets_seen = 0

    ep_rewards    = np.zeros(num_envs, dtype=np.float64)
    ep_steps      = np.zeros(num_envs, dtype=np.int32)
    episodes_done = 0
    # Vector-step counter (ticks +1 per env-step batch). Drives the log /
    # checkpoint / render cadence so it is independent of bursty episode
    # completion (the mp_env and serial loops read it directly; the threaded
    # loop republishes it via shared_vstep for its learner thread).
    vstep         = 0

    # ── Render state ──────────────────────────────────────────────────────────
    # At most one render subprocess runs at a time; new requests are dropped
    # while one is in progress.  States are captured by recording env_2p[0]
    # before every step — no checkpoint needed, the actual training episode
    # is what gets rendered.
    _render_script  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "render_jax_episode.py")
    _render_states_file = os.path.join(ckpt_dir, "render_states.pkl")
    _render_proc: subprocess.Popen | None = None
    _render_seed    = 0
    # Per-episode state buffer for env_2p[0].  Each element is a squeezed
    # numpy GameState recorded at the START of each tick (before actions).
    _ep_states_0: list = []
    _ep_states_0_complete: list | None = None   # most recently sealed episode
    _ep_states_0_won: bool = False
    if render_interval > 0:
        os.makedirs(render_dir, exist_ok=True)
    # Fractional update debt: accumulate num_envs/update_freq per vector step so
    # gradient updates scale with data regardless of whether num_envs%update_freq==0.
    upd_debt      = 0.0

    t0 = time.perf_counter()

    print(f"Running until {num_episodes} episodes complete...")

    # ═════════════════════════════════════════════════════════════════════════
    # Multiprocess-actor loop (jax_env.actor="mp_env")
    #
    # The env lives in a subprocess; this (learner) process keeps all bookkeeping.
    # One-transition-lag pipeline: dispatch update() for the data already in the
    # buffer WHILE the actor steps the env for the actions just sent, so the
    # CPU env.step truly overlaps the GPU update (threads can't — shared GIL).
    # Action selection stays here, so the rollout policy is always the latest.
    # ═════════════════════════════════════════════════════════════════════════
    if mp_env:
        import multiprocessing as mp
        print("Actor in SUBPROCESS (mp_env) — env.step overlaps the GPU update")

        ctx = mp.get_context("spawn")        # fork is unsafe once CUDA is init'd
        parent_conn, child_conn = ctx.Pipe()
        actor_proc = ctx.Process(
            target=_mp_env_actor_worker,
            args=(child_conn, _env_kw, num_envs_2p, num_envs_4p,
                  render_interval > 0),
            daemon=True)
        actor_proc.start()
        child_conn.close()                    # parent keeps only its end

        # Initial obs + planet decode-state from the actor (the learner owns no
        # live env state in mp_env, so the worker ships the state needed to decode).
        obs_batch, dstate = parent_conn.recv()

        def _select(o):
            if trainer.train_step < warmup_steps:
                return np.stack([env_2p.action_space.sample()
                                 for _ in range(num_envs)])
            return trainer.select_action_batch(o)

        # Prime: dispatch the first env step so the actor is busy during update #0.
        # Player-0 is decoded HERE (torch, on the training device) from the raw
        # policy output; the mixed ratio rides alongside so the worker's opponent
        # anneals in lock-step. The replay buffer still stores the RAW action.
        prev_actions = _select(obs_batch)
        prev_obs     = obs_batch
        parent_conn.send((_p0_from_dstate(prev_actions, dstate),
                          _current_mix_ratio(0) if is_mixed else None))

        while episodes_done < num_episodes:
            vstep += 1
            # ── gradient updates — overlap the actor stepping prev_actions ────
            if (trainer.train_step >= warmup_steps
                    and len(trainer.replay_buffer) >= batch_size):
                upd_debt += num_envs / update_freq
                while upd_debt >= 1.0:
                    for _ in range(gradient_steps):
                        trainer.update()
                    upd_debt -= 1.0

            # ── env-step result for prev_actions ─────────────────────────────
            # `dstate` is the post-step planet state, used to decode the next action.
            (next_obs, rewards, dones, wons,
             fleets_sent, n_fleets, state0, dstate) = parent_conn.recv()

            if render_interval > 0 and state0 is not None:
                _ep_states_0.append(state0)

            trainer.replay_buffer.add_batch(
                prev_obs, prev_actions, rewards, next_obs, dones.astype(np.float32))
            trainer.train_step += num_envs

            ep_rewards[:] += rewards
            ep_steps[:]   += 1
            for r in rewards:
                reward_window.append(float(r))
            fleets_window.append(fleets_sent)
            max_fleets_seen = max(max_fleets_seen, n_fleets)

            if trainer.writer:
                trainer.writer.add_scalar("Reward/step_ma100",
                                          float(np.mean(reward_window)), trainer.train_step)
                trainer.writer.add_scalar("Policy/fleets_sent_ma50",
                                          float(np.mean(fleets_window)), trainer.train_step)
                trainer.writer.add_scalar("Env/fleets_present",
                                          n_fleets, trainer.train_step)
                trainer.writer.add_scalar("Env/fleets_present_max",
                                          max_fleets_seen, trainer.train_step)

            # ── finished episodes ────────────────────────────────────────────
            for i in range(num_envs):
                if dones[i]:
                    episode_wins.append(bool(wons[i]))
                    episode_rewards.append(float(ep_rewards[i]))
                    ep_idx = episodes_done
                    if trainer.writer:
                        trainer.writer.add_scalar("Misc/episode_length", int(ep_steps[i]), ep_idx)
                        trainer.writer.add_scalar("Misc/buffer_fill", len(trainer.replay_buffer), ep_idx)
                        trainer.writer.add_scalar("Misc/env_steps", trainer.train_step, ep_idx)
                        trainer.writer.add_scalar("Reward/win", 1.0 if wons[i] else 0.0, ep_idx)
                        trainer.writer.add_scalar("Reward/win_rate_10ep",
                                                  float(np.mean(episode_wins[-10:])) * 100, ep_idx)
                    episodes_done += 1
                    ep_rewards[i] = 0.0
                    ep_steps[i]   = 0
                    if render_interval > 0 and i == 0:
                        _ep_states_0_complete = _ep_states_0
                        _ep_states_0_won      = bool(wons[0])
                        _ep_states_0          = []

            # ── select next actions and dispatch NOW so the actor overlaps the
            #    next iteration's update ────────────────────────────────────────
            obs_batch = next_obs
            if episodes_done < num_episodes:
                prev_actions = _select(obs_batch)
                prev_obs     = obs_batch
                mr = _current_mix_ratio(episodes_done) if is_mixed else None
                if mr is not None and trainer.writer:
                    trainer.writer.add_scalar("Misc/mixed_random_ratio",
                                              mr, trainer.train_step)
                # decode player-0 with the post-step planet state from the worker
                parent_conn.send((_p0_from_dstate(prev_actions, dstate), mr))

            # ── console log (every log_interval vector-steps) ────────────────
            if vstep % log_interval == 0:
                _log_vstep(vstep, num_episodes, episodes_done, episode_rewards,
                           episode_wins, trainer, t0)

            # ── checkpoint (every save_interval vector-steps) ────────────────
            if vstep % save_interval == 0:
                trainer.save_checkpoint(
                    os.path.join(ckpt_dir, f"sac_vs{vstep:06d}.pt"))
                trainer.save_replay_buffer(buffer_path)

            # ── render (background subprocess; every render_interval vsteps) ──
            if (render_interval > 0
                    and vstep % render_interval == 0
                    and _ep_states_0_complete is not None):
                if _render_proc is None or _render_proc.poll() is not None:
                    result_tag = "WIN" if _ep_states_0_won else "LOSS"
                    html_path  = os.path.join(
                        render_dir, f"vs{vstep:06d}_jax_{result_tag}.html")
                    with open(_render_states_file, "wb") as _fh:
                        pickle.dump(_ep_states_0_complete, _fh)
                    _render_proc = subprocess.Popen(
                        [sys.executable, _render_script,
                         "--states-file", _render_states_file,
                         "--final-won",   "true" if _ep_states_0_won else "false",
                         "--output",      html_path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        parent_conn.send(None)                # stop the actor
        actor_proc.join(timeout=10)
        if actor_proc.is_alive():
            actor_proc.terminate()

    # ═════════════════════════════════════════════════════════════════════════
    # Actor-learner threaded loop
    #
    # Two threads share the trainer, replay buffer and (separate) actor network:
    #   env thread     — selects actions from the ACTOR net, steps the JAX envs
    #                    (CPU-heavy), writes transitions, owns episode bookkeeping.
    #   learner thread — samples the buffer and runs SAC gradient updates
    #                    (GPU-heavy), and every `actor_sync_every` updates copies
    #                    the fresh policy weights into the actor under a lock.
    #
    # Because CUDA / numpy / JAX release the GIL during their heavy C work, the
    # env thread's CPU rollout overlaps with the learner thread's GPU updates,
    # closing the GPU-idle gap that the serial loop leaves between env.step and
    # update.  Shutdown is via a stop Event + join(): the learner always finishes
    # its in-flight update before we save, and a final actor sync leaves the actor
    # matching the last trained weights.
    # ═════════════════════════════════════════════════════════════════════════
    if threaded:
        print(f"Actor-learner threading ON (actor synced every "
              f"{actor_sync_every} updates)")

        # ── actor network: env thread reads it, learner syncs into it ──────────
        actor_net = P_network(**net_kw).to(device)
        actor_net.eval()
        if device == "cuda" and hasattr(torch, "compile"):
            actor_net = torch.compile(actor_net)

        def _orig(m):
            # The compiled module wraps the real one as ._orig_mod; copying weights
            # orig→orig keeps the state_dict keys aligned (no _orig_mod. prefix).
            return getattr(m, "_orig_mod", m)

        actor_lock = threading.Lock()

        def sync_actor():
            # Runs only in the learner thread (the sole writer of policy_net), so
            # reading policy_net here is race-free; the lock guards the actor net
            # against the env thread's concurrent forward.
            with actor_lock:
                _orig(actor_net).load_state_dict(_orig(trainer.policy_net).state_dict())

        sync_actor()  # seed the actor with the initial policy weights

        def actor_select(states):
            st = trainer._to(torch.FloatTensor(states))
            st = trainer.state_preprocessor(st)
            with actor_lock, torch.no_grad(), trainer._autocast():
                a, _ = trainer._sample(actor_net, st)   # compiled forward path
            return a.float().cpu().numpy()

        # Self-play opponents read the (slightly stale) actor weights too.
        if opponent == "self_play":
            env_2p.set_policy(actor_select)
            if env_4p is not None:
                env_4p.set_policy(actor_select)

        stop_evt  = threading.Event()
        shared_ep = {"done": 0}
        # Vector-step counter published by the env thread; the learner thread polls
        # it for the checkpoint cadence (it can't read the env thread's local).
        shared_vstep = {"v": 0}
        # Backpressure cap: how many transitions the rollout may get ahead of the
        # learner's update budget.  Without this, a fast env thread (the update is
        # the heavier side here) would exhaust the episode budget before the
        # learner does its share of gradient steps, silently under-training.
        lead_cap  = max(rollout_lead * num_envs, batch_size)

        # ── learner thread ─────────────────────────────────────────────────────
        def learner_worker():
            upd = last_sync = last_ckpt = 0
            while not stop_evt.is_set():
                ts = trainer.train_step
                if ts < warmup_steps or len(trainer.replay_buffer) < batch_size:
                    time.sleep(0.001)
                    continue
                # Hold the serial update-to-transition ratio: one update per
                # update_freq transitions (×gradient_steps).  When behind, run flat
                # out; when caught up, sleep briefly instead of busy-spinning.
                target = int((ts - warmup_steps) / update_freq * gradient_steps)
                if upd >= target:
                    time.sleep(0.0005)
                    continue
                trainer.update()
                upd += 1
                if upd - last_sync >= actor_sync_every:
                    sync_actor()
                    last_sync = upd
                # Periodic checkpoint — done on the learner thread because it owns
                # the networks (the env thread never touches them).  Keyed off the
                # vector-step counter published by the env thread; bucket-crossing
                # (>) rather than == because the learner polls asynchronously and
                # may see vstep jump by more than one between iterations.
                v = shared_vstep["v"]
                if v // save_interval > last_ckpt:
                    last_ckpt = v // save_interval
                    trainer.save_checkpoint(os.path.join(ckpt_dir, f"sac_vs{v:06d}.pt"))
                    trainer.save_replay_buffer(buffer_path)
            sync_actor()   # final sync so the actor matches the last update

        # ── env-rollout thread ─────────────────────────────────────────────────
        def env_worker():
            nonlocal obs_batch, max_fleets_seen
            nonlocal _render_proc, _ep_states_0, _ep_states_0_complete, _ep_states_0_won
            vstep_local = 0
            while shared_ep["done"] < num_episodes and not stop_evt.is_set():
                vstep_local += 1
                shared_vstep["v"] = vstep_local   # publish for the learner thread
                # ── backpressure: keep the rollout within lead_cap transitions of
                #    the learner's update budget so the intended number of updates
                #    actually runs (no effect during warmup) ──────────────────────
                if trainer.train_step >= warmup_steps:
                    expected = warmup_steps + int(
                        trainer._update_count / gradient_steps * update_freq)
                    while (trainer.train_step - expected > lead_cap
                           and not stop_evt.is_set()):
                        time.sleep(0.0005)
                        expected = warmup_steps + int(
                            trainer._update_count / gradient_steps * update_freq)

                # ── anneal the mixed opponent (in-process envs) ─────────────────
                if is_mixed:
                    mr = _current_mix_ratio(shared_ep["done"])
                    env_2p.set_mixed_random_ratio(mr)
                    if env_4p is not None:
                        env_4p.set_mixed_random_ratio(mr)
                    if trainer.writer:
                        trainer.writer.add_scalar("Misc/mixed_random_ratio",
                                                  mr, trainer.train_step)

                # ── action selection (actor net; random during warmup) ──────────
                if trainer.train_step < warmup_steps:
                    actions = np.stack([
                        env_2p.action_space.sample() for _ in range(num_envs)
                    ])
                else:
                    actions = actor_select(obs_batch)

                if render_interval > 0:
                    _ep_states_0.append(jax.tree_util.tree_map(
                        lambda x: np.asarray(x[0]), env_2p._state
                    ))

                # ── env step(s): decode player-0 in-thread, env takes engine acts
                if env_4p is not None:
                    p0_2p = _p0_from_env(env_2p, actions[:num_envs_2p])
                    p0_4p = _p0_from_env(env_4p, actions[num_envs_2p:])
                    next_2p, rew_2p, done_2p, _, won_2p = env_2p.step_engine(p0_2p)
                    next_4p, rew_4p, done_4p, _, won_4p = env_4p.step_engine(p0_4p)
                    next_obs = np.concatenate([next_2p, next_4p], axis=0)
                    rewards  = np.concatenate([rew_2p,  rew_4p])
                    dones    = np.concatenate([done_2p, done_4p])
                    wons     = np.concatenate([won_2p,  won_4p])
                    fleets_sent = env_2p.last_fleets_sent + env_4p.last_fleets_sent
                    n_fleets    = max(env_2p.last_n_fleets, env_4p.last_n_fleets)
                else:
                    next_obs, rewards, dones, _, wons = env_2p.step_engine(
                        _p0_from_env(env_2p, actions))
                    fleets_sent = env_2p.last_fleets_sent
                    n_fleets    = env_2p.last_n_fleets

                trainer.replay_buffer.add_batch(
                    obs_batch, actions, rewards, next_obs, dones.astype(np.float32)
                )
                trainer.train_step += num_envs

                # In-place slice ops (not `+=` on the bare name) so Python does
                # not treat these enclosing arrays as env_worker locals.
                ep_rewards[:] += rewards
                ep_steps[:]   += 1

                for r in rewards:
                    reward_window.append(float(r))
                fleets_window.append(fleets_sent)
                max_fleets_seen = max(max_fleets_seen, n_fleets)

                if trainer.writer:
                    trainer.writer.add_scalar("Reward/step_ma100",
                                              float(np.mean(reward_window)), trainer.train_step)
                    trainer.writer.add_scalar("Policy/fleets_sent_ma50",
                                              float(np.mean(fleets_window)), trainer.train_step)
                    trainer.writer.add_scalar("Env/fleets_present",
                                              n_fleets, trainer.train_step)
                    trainer.writer.add_scalar("Env/fleets_present_max",
                                              max_fleets_seen, trainer.train_step)

                # ── track finished episodes ─────────────────────────────────────
                for i in range(num_envs):
                    if dones[i]:
                        episode_wins.append(bool(wons[i]))
                        episode_rewards.append(float(ep_rewards[i]))
                        ep_idx = shared_ep["done"]
                        if trainer.writer:
                            trainer.writer.add_scalar("Misc/episode_length", int(ep_steps[i]), ep_idx)
                            trainer.writer.add_scalar("Misc/buffer_fill", len(trainer.replay_buffer), ep_idx)
                            trainer.writer.add_scalar("Misc/env_steps", trainer.train_step, ep_idx)
                            trainer.writer.add_scalar("Reward/win", 1.0 if wons[i] else 0.0, ep_idx)
                            trainer.writer.add_scalar("Reward/win_rate_10ep",
                                                      float(np.mean(episode_wins[-10:])) * 100, ep_idx)
                        shared_ep["done"] += 1
                        ep_rewards[i] = 0.0
                        ep_steps[i]   = 0
                        if render_interval > 0 and i == 0:
                            _ep_states_0_complete = _ep_states_0
                            _ep_states_0_won      = bool(wons[0])
                            _ep_states_0          = []

                obs_batch = next_obs

                # ── console log (every log_interval vector-steps) ───────────────
                if vstep_local % log_interval == 0:
                    _log_vstep(vstep_local, num_episodes, shared_ep["done"],
                               episode_rewards, episode_wins, trainer, t0)

                # ── JAX render (background subprocess; every render_interval vsteps)
                if (render_interval > 0
                        and vstep_local % render_interval == 0
                        and _ep_states_0_complete is not None):
                    if _render_proc is None or _render_proc.poll() is not None:
                        result_tag = "WIN" if _ep_states_0_won else "LOSS"
                        html_path  = os.path.join(
                            render_dir, f"vs{vstep_local:06d}_jax_{result_tag}.html")
                        with open(_render_states_file, "wb") as _fh:
                            pickle.dump(_ep_states_0_complete, _fh)
                        _render_proc = subprocess.Popen(
                            [sys.executable, _render_script,
                             "--states-file", _render_states_file,
                             "--final-won",   "true" if _ep_states_0_won else "false",
                             "--output",      html_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )

        learner_t = threading.Thread(target=learner_worker, name="learner", daemon=True)
        env_t     = threading.Thread(target=env_worker,     name="env-rollout", daemon=True)
        learner_t.start()
        env_t.start()
        env_t.join()
        # The rollout is done; let the learner drain its remaining update budget so
        # the total gradient steps match the serial loop (which updates inline each
        # vstep), then stop it.  join() guarantees the last update finished before
        # we save.
        final_target = int((trainer.train_step - warmup_steps) / update_freq * gradient_steps)
        while trainer._update_count < final_target and learner_t.is_alive():
            time.sleep(0.01)
        stop_evt.set()
        learner_t.join()
        episodes_done = shared_ep["done"]

    # ── Serial main loop (fallback; jax_env.actor="serial") ───────────────────
    while (not threaded) and (not mp_env) and episodes_done < num_episodes:
        vstep += 1
        # ── anneal the mixed opponent ─────────────────────────────────────────
        if is_mixed:
            mr = _current_mix_ratio(episodes_done)
            env_2p.set_mixed_random_ratio(mr)
            if env_4p is not None:
                env_4p.set_mixed_random_ratio(mr)
            if trainer.writer:
                trainer.writer.add_scalar("Misc/mixed_random_ratio",
                                          mr, trainer.train_step)
        # ── action selection ─────────────────────────────────────────────────
        if trainer.train_step < warmup_steps:
            actions = np.stack([
                env_2p.action_space.sample() for _ in range(num_envs)
            ])
        else:
            actions = trainer.select_action_batch(obs_batch)

        # ── record env_2p[0] state for rendering (before the step) ──────────────
        if render_interval > 0:
            _ep_states_0.append(jax.tree_util.tree_map(
                lambda x: np.asarray(x[0]), env_2p._state
            ))

        # ── env step(s): decode player-0 in the learner, env takes engine actions
        if env_4p is not None:
            p0_2p = _p0_from_env(env_2p, actions[:num_envs_2p])
            p0_4p = _p0_from_env(env_4p, actions[num_envs_2p:])
            next_2p, rew_2p, done_2p, _, won_2p = env_2p.step_engine(p0_2p)
            next_4p, rew_4p, done_4p, _, won_4p = env_4p.step_engine(p0_4p)
            next_obs = np.concatenate([next_2p, next_4p], axis=0)
            rewards  = np.concatenate([rew_2p,  rew_4p])
            dones    = np.concatenate([done_2p, done_4p])
            wons     = np.concatenate([won_2p,  won_4p])
            fleets_sent = env_2p.last_fleets_sent + env_4p.last_fleets_sent
            n_fleets    = max(env_2p.last_n_fleets,  env_4p.last_n_fleets)
        else:
            next_obs, rewards, dones, _, wons = env_2p.step_engine(
                _p0_from_env(env_2p, actions))
            fleets_sent = env_2p.last_fleets_sent
            n_fleets    = env_2p.last_n_fleets

        trainer.replay_buffer.add_batch(
            obs_batch, actions, rewards, next_obs, dones.astype(np.float32)
        )
        trainer.train_step += num_envs

        ep_rewards += rewards
        ep_steps   += 1

        # ── per-step diagnostics ──────────────────────────────────────────────
        for r in rewards:
            reward_window.append(float(r))
        fleets_window.append(fleets_sent)
        max_fleets_seen = max(max_fleets_seen, n_fleets)

        if trainer.writer:
            trainer.writer.add_scalar("Reward/step_ma100",
                                      float(np.mean(reward_window)),
                                      trainer.train_step)
            trainer.writer.add_scalar("Policy/fleets_sent_ma50",
                                      float(np.mean(fleets_window)),
                                      trainer.train_step)
            trainer.writer.add_scalar("Env/fleets_present",
                                      n_fleets, trainer.train_step)
            trainer.writer.add_scalar("Env/fleets_present_max",
                                      max_fleets_seen, trainer.train_step)

        # ── track finished episodes ───────────────────────────────────────────
        for i in range(num_envs):
            if dones[i]:
                episode_wins.append(bool(wons[i]))
                episode_rewards.append(float(ep_rewards[i]))
                ep_idx = episodes_done
                if trainer.writer:
                    trainer.writer.add_scalar("Misc/episode_length",
                                              int(ep_steps[i]), ep_idx)
                    trainer.writer.add_scalar("Misc/buffer_fill",
                                              len(trainer.replay_buffer), ep_idx)
                    trainer.writer.add_scalar("Misc/env_steps",
                                              trainer.train_step, ep_idx)
                    trainer.writer.add_scalar("Reward/win",
                                              1.0 if wons[i] else 0.0, ep_idx)
                    trainer.writer.add_scalar("Reward/win_rate_10ep",
                                              float(np.mean(episode_wins[-10:])) * 100,
                                              ep_idx)
                episodes_done += 1
                ep_rewards[i] = 0.0
                ep_steps[i]   = 0

                # Seal the env_2p[0] episode buffer when that env finishes.
                # i==0 corresponds to env_2p[0] in both the 2p-only and 2p+4p
                # cases (env_2p occupies indices 0..num_envs_2p-1 in dones).
                if render_interval > 0 and i == 0:
                    _ep_states_0_complete = _ep_states_0
                    _ep_states_0_won      = bool(wons[0])
                    _ep_states_0          = []

        # ── gradient updates ──────────────────────────────────────────────────
        # Accumulate fractional updates so the ratio of gradient steps to env
        # transitions matches the single-env rate (1 update per update_freq
        # transitions), independent of num_envs.
        if (
            trainer.train_step >= warmup_steps
            and len(trainer.replay_buffer) >= batch_size
        ):
            upd_debt += num_envs / update_freq
            while upd_debt >= 1.0:
                for _ in range(gradient_steps):
                    trainer.update()
                upd_debt -= 1.0

        obs_batch = next_obs

        # ── console log (every log_interval vector-steps) ─────────────────────
        if vstep % log_interval == 0:
            _log_vstep(vstep, num_episodes, episodes_done, episode_rewards,
                       episode_wins, trainer, t0)

        # ── checkpoint (every save_interval vector-steps) ─────────────────────
        if vstep % save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"sac_vs{vstep:06d}.pt")
            trainer.save_checkpoint(ckpt_path)
            trainer.save_replay_buffer(buffer_path)

        # ── JAX render (background subprocess; every render_interval vsteps) ──
        # Renders the actual training episode of env_2p[0], not a new one.
        # Add your own condition here to render on specific events instead.
        if (render_interval > 0
                and vstep % render_interval == 0
                and _ep_states_0_complete is not None):
            # Drop if the previous render is still running.
            if _render_proc is None or _render_proc.poll() is not None:
                result_tag = "WIN" if _ep_states_0_won else "LOSS"
                html_path  = os.path.join(
                    render_dir, f"vs{vstep:06d}_jax_{result_tag}.html"
                )
                with open(_render_states_file, "wb") as _fh:
                    pickle.dump(_ep_states_0_complete, _fh)
                _render_proc = subprocess.Popen(
                    [
                        sys.executable, _render_script,
                        "--states-file", _render_states_file,
                        "--final-won",   "true" if _ep_states_0_won else "false",
                        "--output",      html_path,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

    # ── final save ────────────────────────────────────────────────────────────
    trainer.save_checkpoint(os.path.join(ckpt_dir, "sac_final.pt"))
    trainer.save_replay_buffer(buffer_path)
    trainer.close()
    env_2p.close()
    if env_4p is not None:
        env_4p.close()

    # Wait for any in-progress render to finish before exiting.
    if _render_proc is not None and _render_proc.poll() is None:
        print("Waiting for final render to complete...")
        _render_proc.wait()

    print(f"\nDone. {trainer.train_step} total env steps, {episodes_done} episodes.")
    return episode_rewards


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "train.json") -> dict:
    """Load training configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    """Parse only the config file path; all other settings come from JSON."""
    p = argparse.ArgumentParser(
        description="Train SAC on Orbit Wars",
        epilog="All configuration is read from the JSON file specified by --config"
    )
    p.add_argument("--config", type=str, default="train.json",
                   help="Path to training configuration JSON file (default: train.json)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config

    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # Set random seeds
    exec_cfg = config.get("execution", {})
    seed = exec_cfg.get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Random seed set to: {seed}")

    print()
    train_jax(config, MAX_PLANETS=40, MAX_FLEETS=200)
