"""
SAC training for Kaggle Orbit Wars.

Alternates between 2-player and 4-player episodes at a configurable ratio.
Opponents can be rule-based, random, or "mixed" — a per-step blend of the two
controlled by `mixed_random_ratio` (1.0 = fully random, 0.0 = fully rule-based).

Mixed-ratio curriculum
----------------------
When opponent is "mixed", the random_ratio can be linearly annealed over training
via `curriculum.mixed_random_ratio_start/_end/_decay`. Typically start high
(more random → easier opponent) and decay to a lower value (more rule-based →
harder). Set `_decay` to null to decay over the full `num_episodes`.

Usage
-----
    python train_orbit_wars.py                          # uses train.json
    python train_orbit_wars.py --config my_config.json  # custom config

All hyperparameters live in the JSON config file (model, training, environment,
curriculum, reward, io, execution sections).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from model.SAC      import P_network, Q_network
from sac_train      import SACTrainer
from env.orbit_wars import (
    OrbitWarsEnv,
    RewardScheme1, RewardScheme2, RewardScheme3, RewardScheme4,
)
from agents.agent1  import RuleBasedAgent

try:
    import kaggle_environments.envs.orbit_wars.orbit_wars as _ow
except ImportError:
    _ow = None


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
# Mixed opponent (per-step blend of random and rule-based)
# ─────────────────────────────────────────────────────────────────────────────

class MixedAgent:
    """
    Per-step blend of random and rule-based actions.

    At every call, independently samples whether to act randomly or rely on the
    wrapped RuleBasedAgent.

        random_ratio = 1.0  → every step is random
        random_ratio = 0.0  → every step is rule-based
        random_ratio = 0.5  → ~50/50 mix

    Random actions: for each owned planet, with probability `send_prob`, launch
    a fleet at a random angle with a random ship count (1..ships).
    """

    def __init__(self, random_ratio: float = 0.5, send_prob: float = 0.3, fleet_thrsh: int = 20):
        self.random_ratio = float(random_ratio)
        self.send_prob    = float(send_prob)
        self.fleet_thrsh =  fleet_thrsh
        self._base        = RuleBasedAgent()

    def reset(self):
        if hasattr(self._base, "reset"):
            self._base.reset()

    def __call__(self, obs, config=None):
        if random.random() < self.random_ratio:
            return self._random_action(obs)
        return self._base(obs, config)

    def _random_action(self, obs):
        if _ow is None:
            return []
        player = obs.get("player", -2)
        planets = [_ow.Planet(*p) for p in obs.get("planets", [])]
        moves = []
        for p in planets:
            if p.owner != player or p.ships < 1:
                continue
            if random.random() >= self.send_prob:
                continue
            if p.ships < self.fleet_thrsh:
                continue
            ships = random.randint(self.fleet_thrsh, int(p.ships))
            angle = random.uniform(0.0, 2.0 * math.pi)
            moves.append([p.id, angle, ships])
        return moves


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(n_players: int, opponent: str, MAX_PLANETS: int = 40, MAX_FLEETS: int = 100,
             reward_scheme=None, mixed_random_ratio: float = 0.5,
             mixed_send_prob: float = 0.3,
             tanh_scale: float = 0.2, min_fleet_ships: int = 3):
    """
    Build an OrbitWarsEnv.  Opponent choices: rule_based | random | mixed.

    "mixed" uses MixedAgent: each step independently chooses between random
    and rule-based actions based on `mixed_random_ratio` (1.0 → fully random,
    0.0 → fully rule-based).

    Returns
    -------
    env          : OrbitWarsEnv
    mixed_agents : list[MixedAgent]  — MixedAgent instances whose .random_ratio
                   can be updated each episode by the training loop to implement
                   curriculum scheduling. Empty for non-mixed opponents.
    """
    if opponent == "rule_based":
        opps = [RuleBasedAgent() for _ in range(n_players - 1)]
    elif opponent == "random":
        opps = ["random"] * (n_players - 1)
    elif opponent == "mixed":
        opps = [
            MixedAgent(random_ratio=mixed_random_ratio, send_prob=mixed_send_prob)
            for _ in range(n_players - 1)
        ]
    else:
        raise ValueError(f"Unknown opponent: {opponent!r}")

    mixed_agents = [a for a in opps if isinstance(a, MixedAgent)]

    env = OrbitWarsEnv(opponent=opps, player_id=0, n_players=n_players,
                       reward_scheme=reward_scheme,
                       tanh_scale=tanh_scale, min_fleet_ships=min_fleet_ships)
    env.MAX_FLEETS  = MAX_FLEETS
    env.MAX_PLANETS = MAX_PLANETS
    return env, mixed_agents


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict, reward_scheme=None, MAX_PLANETS: int = 40, MAX_FLEETS: int = 100) -> list[float]:
    """Train SAC agent using configuration from JSON.

    Parameters
    ----------
    config : dict
        Training configuration loaded from JSON file
    reward_scheme : list[RewardScheme]
        List of reward scheme instances (initialized separately)
    MAX_PLANETS, MAX_FLEETS : int
        Environment constants
    """
    # Extract config sections
    train_cfg = config.get("training", {})
    env_cfg = config.get("environment", {})
    curr_cfg = config.get("curriculum", {})
    io_cfg = config.get("io", {})
    exec_cfg = config.get("execution", {})
    model_cfg = config.get("model", {})
    if reward_scheme is None:
        reward_scheme = [RewardScheme1()]

    cpu_force = exec_cfg.get("cpu_force", False)
    device = "cuda" if torch.cuda.is_available() and not cpu_force else "cpu"
    print(f"Device: {device}")

    ckpt_dir = io_cfg.get("ckpt_dir", "checkpoints")
    render_dir = io_cfg.get("render_dir", "replays")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)
    # Replay buffer is persisted alongside checkpoints so a resumed run does not
    # have to rebuild it from scratch.
    buffer_path = os.path.join(ckpt_dir, "replay_buffer.npz")

    num_episodes = train_cfg.get("num_episodes", 250)

    opponent        = env_cfg.get("opponent",        "rule_based")
    ratio_4p        = env_cfg.get("ratio_4p",        0.3)
    mixed_send_prob = env_cfg.get("mixed_send_prob", 0.3)
    tanh_scale      = env_cfg.get("tanh_scale",      0.2)
    min_fleet_ships = env_cfg.get("min_fleet_ships",  3)

    # ── mixed_random_ratio scheduler (curriculum) ────────────────────────────
    ratio_start     = curr_cfg.get("mixed_random_ratio_start",    1.0)
    ratio_end       = curr_cfg.get("mixed_random_ratio_end",       0.0)
    ratio_decay     = curr_cfg.get("mixed_random_ratio_decay")
    ratio_schedule  = curr_cfg.get("mixed_random_ratio_schedule", "linear")
    ratio_decay_eps = ratio_decay if ratio_decay is not None else num_episodes
    schedule_fn     = _make_schedule(ratio_schedule)

    if opponent == "mixed":
        print(f"Mixed opponent: random_ratio {ratio_start:.2f} → {ratio_end:.2f} "
              f"over {ratio_decay_eps} episodes ({ratio_schedule}), "
              f"send_prob={mixed_send_prob:.2f}")

    # ── envs ──────────────────────────────────────────────────────────────────
    env_2p, mixed_2p = make_env(
        n_players=2, opponent=opponent,
        MAX_PLANETS=MAX_PLANETS, MAX_FLEETS=MAX_FLEETS, reward_scheme=reward_scheme,
        mixed_random_ratio=ratio_start, mixed_send_prob=mixed_send_prob,
        tanh_scale=tanh_scale, min_fleet_ships=min_fleet_ships,
    )
    if ratio_4p > 0.0:
        env_4p, mixed_4p = make_env(
            n_players=4, opponent=opponent,
            MAX_PLANETS=MAX_PLANETS, MAX_FLEETS=MAX_FLEETS, reward_scheme=reward_scheme,
            mixed_random_ratio=ratio_start, mixed_send_prob=mixed_send_prob,
            tanh_scale=tanh_scale, min_fleet_ships=min_fleet_ships,
        )
    else:
        env_4p, mixed_4p = None, []

    all_mixed_agents = mixed_2p + mixed_4p

    # ── networks — action_dim=4 to match OrbitWarsEnv.ACTION_DIM ─────────────
    d_model = model_cfg.get("d_model", 128)
    net_kw = dict(
        state_dim   = OrbitWarsEnv.STATE_DIM,    # 14
        action_dim  = OrbitWarsEnv.ACTION_DIM,   # 4
        max_planets = OrbitWarsEnv.MAX_PLANETS,  # 40
        max_fleets  = OrbitWarsEnv.MAX_FLEETS,   # 100
        d_model     = d_model,
        # Additional transformer parameters can be added here if needed
    )
    policy_net = P_network(**net_kw)
    q1_net     = Q_network(**net_kw)
    q2_net     = Q_network(**net_kw)

    # ── SACTrainer — env_2p provides obs/act shapes for the replay buffer ─────
    lr = train_cfg.get("lr", 3e-4)
    gamma = train_cfg.get("gamma", 0.99)
    tau = train_cfg.get("tau", 5e-3)
    alpha = train_cfg.get("alpha", 0.2)
    auto_alpha = train_cfg.get("auto_alpha", False)
    target_entropy = train_cfg.get("target_entropy")
    buffer_size = train_cfg.get("buffer_size", 100_000)
    batch_size = train_cfg.get("batch_size", 64)
    max_grad_norm = train_cfg.get("grad_clip", 1.0)
    log_dir = io_cfg.get("log_dir")

    # ── cache-based TD(λ) (optional; replaces target net with a λ-return cache)
    tdl_cfg            = config.get("td_lambda", {})
    use_lambda_returns = tdl_cfg.get("enabled", False)
    lambda_return      = tdl_cfg.get("lambda", 0.9)
    cache_size         = tdl_cfg.get("cache_size", 8000)
    block_size         = tdl_cfg.get("block_size", 50)
    refresh_freq       = tdl_cfg.get("refresh_freq", 1000)
    if use_lambda_returns:
        print(f"TD(λ) cache: λ={lambda_return}, S={cache_size}, B={block_size}, "
              f"refresh every {refresh_freq} steps (target network disabled)")

    trainer = SACTrainer(
        env               = env_2p,
        policy_net        = policy_net,
        q1_net            = q1_net,
        q2_net            = q2_net,
        device            = device,
        learning_rate     = lr,
        gamma             = gamma,
        tau               = tau,
        alpha             = alpha,
        auto_alpha        = auto_alpha,
        target_entropy    = target_entropy,
        replay_buffer_size= buffer_size,
        batch_size        = batch_size,
        max_grad_norm     = max_grad_norm,
        use_lambda_returns= use_lambda_returns,
        lambda_return     = lambda_return,
        cache_size        = cache_size,
        block_size        = block_size,
        refresh_freq      = refresh_freq,
        log_dir           = log_dir,
    )

    resume = exec_cfg.get("resume")
    if resume:
        trainer.load_checkpoint(resume)
        trainer.load_replay_buffer(buffer_path)

    episode_rewards: list[float] = []
    episode_wins: list[bool] = []
    t0 = time.perf_counter()

    max_steps = train_cfg.get("max_steps", 500)
    warmup_steps = train_cfg.get("warmup_steps", 500)
    update_freq = train_cfg.get("update_freq", 4)
    gradient_steps = train_cfg.get("gradient_steps", 1)
    log_interval = io_cfg.get("log_interval", 1)
    render_interval = io_cfg.get("render_interval", 10)
    save_interval = io_cfg.get("save_interval", 100)

    for episode in range(num_episodes):
        # ── curriculum: anneal mixed_random_ratio ─────────────────────────────
        current_ratio = schedule_fn(
            episode, ratio_start, ratio_end, ratio_decay_eps
        )
        for a in all_mixed_agents:
            a.random_ratio = current_ratio

        # ── pick 2p or 4p env ─────────────────────────────────────────────────
        use_4p  = env_4p is not None and random.random() < ratio_4p
        env     = env_4p if use_4p else env_2p
        tag     = "4p" if use_4p else "2p"

        state, _ = env.reset()
        ep_reward = 0.0
        ep_length = 0

        # ── episode rollout ───────────────────────────────────────────────────
        for _ in range(max_steps):
            if trainer.train_step < warmup_steps:
                action = env.action_space.sample()
            else:
                action = trainer.select_action(state)

            next_state, reward, terminated, truncated, won = env.step(action)
            done = terminated or truncated

            trainer.replay_buffer.add(
                state, action, reward, next_state, float(done)
            )
            trainer.train_step += 1
            ep_reward += reward
            ep_length += 1
            state = next_state

            if (
                trainer.train_step >= warmup_steps
                and trainer.train_step % update_freq == 0
                and len(trainer.replay_buffer) >= batch_size
            ):
                for _ in range(gradient_steps):
                    trainer.update()

            if done:
                break

        # Determine if player 0 won: has planets and all opponents have none.
        episode_wins.append(won)

        episode_rewards.append(ep_reward)
        trainer._log_episode(episode, ep_reward, ep_length)

        # ── per-episode tensorboard scalars ───────────────────────────────────
        win_float    = 1.0 if won else 0.0
        win_rate_10  = float(np.mean(episode_wins[-10:])) * 100
        if trainer.writer:
            trainer.writer.add_scalar("Misc/n_players",            4 if use_4p else 2, episode)
            trainer.writer.add_scalar("Misc/mixed_random_ratio",   current_ratio,      episode)
            trainer.writer.add_scalar("Reward/win",                win_float,          episode)
            trainer.writer.add_scalar("Reward/win_rate_10ep",      win_rate_10,        episode)

        # ── console log ───────────────────────────────────────────────────────
        if (episode + 1) % log_interval == 0:
            recent_rewards = episode_rewards[-log_interval:]
            recent_wins    = episode_wins[-log_interval:]
            avg_reward     = float(np.mean(recent_rewards))
            win_rate       = float(np.mean(recent_wins)) * 100 if recent_wins else 0.0
            elapsed        = time.perf_counter() - t0
            ratio_tag      = (f" | MixR: {current_ratio:.2f}"
                              if opponent == "mixed" else "")
            print(
                f"Ep {episode + 1:>5}/{num_episodes} [{tag}] | "
                f"Avg({log_interval}): {avg_reward:+8.3f} | "
                f"Last: {ep_reward:+8.3f} | "
                f"Win%: {win_rate:5.1f} | "
                f"Buffer: {len(trainer.replay_buffer):>7} | "
                f"Steps: {trainer.train_step:>7}{ratio_tag} | "
                f"{elapsed:.0f}s"
            )
            if trainer.writer:
                trainer.writer.add_scalar("Reward/moving_avg",    avg_reward, episode)

        # ── HTML replay ───────────────────────────────────────────────────────
        if (episode + 1) % render_interval == 0:
            result_tag = "WIN" if won else "LOSS"
            html_path = os.path.join(
                render_dir,
                f"ep{episode + 1:05d}_{tag}_{result_tag}.html",
            )
            env.render(html_path=html_path)

        # ── checkpoint ────────────────────────────────────────────────────────
        if (episode + 1) % save_interval == 0:
            ckpt_path = os.path.join(
                ckpt_dir, f"sac_ep{episode + 1:05d}.pt"
            )
            trainer.save_checkpoint(ckpt_path)
            trainer.save_replay_buffer(buffer_path)

    # ── final save ────────────────────────────────────────────────────────────
    trainer.save_checkpoint(os.path.join(ckpt_dir, "sac_final.pt"))
    trainer.save_replay_buffer(buffer_path)
    trainer.close()
    env_2p.close()
    if env_4p:
        env_4p.close()

    print(f"\nDone. {trainer.train_step} total env steps.")
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

    # Load and instantiate reward schemes
    reward_cfg_list = config.get("reward", [])
    reward_scheme_map = {
        "RewardScheme1": RewardScheme1,
        "RewardScheme2": RewardScheme2,
        "RewardScheme3": RewardScheme3,
        "RewardScheme4": RewardScheme4,
    }

    reward_scheme = []
    if reward_cfg_list:
        for reward_cfg in reward_cfg_list:
            scheme_name = reward_cfg.get("scheme", "RewardScheme1")
            RewardSchemeClass = reward_scheme_map.get(scheme_name, RewardScheme1)

            # Extract parameters, handling RewardScheme3 which has max_ticks instead
            if scheme_name == "RewardScheme3":
                params = {
                    "ship_scale": reward_cfg.get("ship_scale", 0.5),
                    "planet_scale": reward_cfg.get("planet_scale", 1.0),
                    "max_ticks": reward_cfg.get("max_ticks", 200),
                }
            else:
                params = {
                    "ship_scale": reward_cfg.get("ship_scale", 0.01),
                    "planet_scale": reward_cfg.get("planet_scale", 1.0),
                    "win_bonus": reward_cfg.get("win_bonus", 100.0),
                }

            reward_scheme.append(RewardSchemeClass(**params))
            print(f"Loaded {scheme_name} with params: {params}")
    else:
        # Fallback to RewardScheme1 if no reward config
        reward_scheme = [RewardScheme1()]
        print("No reward schemes in config, using default RewardScheme1")

    print(f"Training with {len(reward_scheme)} reward scheme(s)")
    print()

    # Start training
    train(config, MAX_PLANETS=40, MAX_FLEETS=100, reward_scheme=reward_scheme)
