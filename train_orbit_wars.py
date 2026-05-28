"""
SAC training for Kaggle Orbit Wars.

Alternates between 2-player and 4-player episodes at a configurable ratio.
Opponents are drawn from agents/agent1.py (RuleBasedAgent) or kaggle's built-in
random agent.  An HTML replay is saved every --render-interval episodes.

Usage
-----
    python train_orbit_wars.py                          # defaults
    python train_orbit_wars.py --num-episodes 5000 --ratio-4p 0.4 --auto-alpha
    python train_orbit_wars.py --resume checkpoints/sac_ep01000.pt
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch

from model.SAC      import P_network, Q_network
from sac_train      import SACTrainer
from env.orbit_wars import (
    OrbitWarsEnv,
    RewardScheme1, RewardScheme2, RewardScheme3,
)
from agents.agent1  import RuleBasedAgent


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(n_players: int, opponent: str, MAX_PLANETS: int = 40, MAX_FLEETS: int = 100,
             reward_scheme=None) -> OrbitWarsEnv:
    """Build an OrbitWarsEnv.  Opponent choices: rule_based | random | mixed."""
    if opponent == "rule_based":
        opp = [RuleBasedAgent() for _ in range(n_players - 1)]
    elif opponent == "random":
        opp = "random"
    elif opponent == "mixed":
        opp = [RuleBasedAgent() if i % 2 == 0 else "random"
               for i in range(n_players - 1)]
    else:
        raise ValueError(f"Unknown opponent: {opponent!r}")
    env = OrbitWarsEnv(opponent=opp, player_id=0, n_players=n_players, reward_scheme=reward_scheme)
    env.MAX_FLEETS = MAX_FLEETS  # ensure env attribute matches net_kw for SACTrainer
    env.MAX_PLANETS = MAX_PLANETS
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace, reward_scheme=None, MAX_PLANETS: int = 40, MAX_FLEETS: int = 100) -> list[float]:
    if reward_scheme is None:
        reward_scheme = [RewardScheme1()]

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.ckpt_dir,    exist_ok=True)
    os.makedirs(args.render_dir,  exist_ok=True)
    
    
    # ── envs ──────────────────────────────────────────────────────────────────
    env_2p = make_env(n_players=2, opponent=args.opponent, MAX_PLANETS=MAX_PLANETS, MAX_FLEETS=MAX_FLEETS, reward_scheme=reward_scheme)
    env_4p = make_env(n_players=4, opponent=args.opponent, MAX_PLANETS=MAX_PLANETS, MAX_FLEETS=MAX_FLEETS, reward_scheme=reward_scheme) \
             if args.ratio_4p > 0.0 else None

    # ── networks — action_dim=4 to match OrbitWarsEnv.ACTION_DIM ─────────────
    net_kw = dict(
        state_dim   = OrbitWarsEnv.STATE_DIM,    # 14
        action_dim  = OrbitWarsEnv.ACTION_DIM,   # 4
        max_planets = OrbitWarsEnv.MAX_PLANETS,  # 40
        max_fleets  = OrbitWarsEnv.MAX_FLEETS,   # 100
        d_model     = args.d_model,
    )
    policy_net = P_network(**net_kw)
    q1_net     = Q_network(**net_kw)
    q2_net     = Q_network(**net_kw)

    # ── SACTrainer — env_2p provides obs/act shapes for the replay buffer ─────
    trainer = SACTrainer(
        env               = env_2p,
        policy_net        = policy_net,
        q1_net            = q1_net,
        q2_net            = q2_net,
        device            = device,
        learning_rate     = args.lr,
        gamma             = args.gamma,
        tau               = args.tau,
        alpha             = args.alpha,
        auto_alpha        = args.auto_alpha,
        target_entropy    = args.target_entropy,
        replay_buffer_size= args.buffer_size,
        batch_size        = args.batch_size,
        log_dir           = args.log_dir,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    episode_rewards: list[float] = []
    t0 = time.perf_counter()

    for episode in range(args.num_episodes):
        # ── pick 2p or 4p env ─────────────────────────────────────────────────
        use_4p  = env_4p is not None and random.random() < args.ratio_4p
        env     = env_4p if use_4p else env_2p
        tag     = "4p" if use_4p else "2p"

        state, _ = env.reset()
        ep_reward = 0.0
        ep_length = 0

        # ── episode rollout ───────────────────────────────────────────────────
        for _ in range(args.max_steps):
            if trainer.train_step < args.warmup_steps:
                action = env.action_space.sample()
            else:
                action = trainer.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            trainer.replay_buffer.add(
                state, action, reward, next_state, float(done)
            )
            trainer.train_step += 1
            ep_reward += reward
            ep_length += 1
            state = next_state

            if (
                trainer.train_step >= args.warmup_steps
                and trainer.train_step % args.update_freq == 0
                and len(trainer.replay_buffer) >= args.batch_size
            ):
                for _ in range(args.gradient_steps):
                    trainer.update()

            if done:
                break

        episode_rewards.append(ep_reward)
        trainer._log_episode(episode, ep_reward, ep_length)

        if trainer.writer:
            trainer.writer.add_scalar("Misc/n_players", 4 if use_4p else 2, episode)

        # ── console log ───────────────────────────────────────────────────────
        if (episode + 1) % args.log_interval == 0:
            recent  = episode_rewards[-args.log_interval:]
            avg     = float(np.mean(recent))
            elapsed = time.perf_counter() - t0
            print(
                f"Ep {episode + 1:>5}/{args.num_episodes} [{tag}] | "
                f"Avg({args.log_interval}): {avg:+8.3f} | "
                f"Last: {ep_reward:+8.3f} | "
                f"Buffer: {len(trainer.replay_buffer):>7} | "
                f"Steps: {trainer.train_step:>7} | "
                f"{elapsed:.0f}s"
            )
            if trainer.writer:
                trainer.writer.add_scalar("Reward/moving_avg", avg, episode)

        # ── HTML replay ───────────────────────────────────────────────────────
        if (episode + 1) % args.render_interval == 0:
            html_path = os.path.join(
                args.render_dir,
                f"ep{episode + 1:05d}_{tag}.html",
            )
            env.render(html_path=html_path)

        # ── checkpoint ────────────────────────────────────────────────────────
        if (episode + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(
                args.ckpt_dir, f"sac_ep{episode + 1:05d}.pt"
            )
            trainer.save_checkpoint(ckpt_path)

    # ── final save ────────────────────────────────────────────────────────────
    trainer.save_checkpoint(os.path.join(args.ckpt_dir, "sac_final.pt"))
    trainer.close()
    env_2p.close()
    if env_4p:
        env_4p.close()

    print(f"\nDone. {trainer.train_step} total env steps.")
    return episode_rewards


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SAC on Orbit Wars")

    # environment
    p.add_argument("--opponent",   default="rule_based",
                   choices=["rule_based", "random", "mixed"],
                   help="Opponent type for non-learning seats")
    p.add_argument("--ratio-4p",  type=float, default=0.3,
                   help="Fraction of episodes played as 4-player (0 → 2p only)")
    p.add_argument("--max-steps", type=int,   default=500,
                   help="Episode truncation length")

    # training
    p.add_argument("--num-episodes",   type=int,   default=2000)
    p.add_argument("--warmup-steps",   type=int,   default=500,
                   help="Random exploration steps before SAC updates begin")
    p.add_argument("--batch-size",     type=int,   default=64)
    p.add_argument("--buffer-size",    type=int,   default=100_000)
    p.add_argument("--update-freq",    type=int,   default=1,
                   help="SAC update every N env steps")
    p.add_argument("--gradient-steps", type=int,   default=1,
                   help="Gradient updates per update cycle")
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--gamma",          type=float, default=0.99)
    p.add_argument("--tau",            type=float, default=5e-3)
    p.add_argument("--alpha",          type=float, default=0.2,
                   help="Fixed entropy temperature (ignored when --auto-alpha)")
    p.add_argument("--auto-alpha",     action="store_true",
                   help="Automatically tune entropy temperature")
    p.add_argument("--target-entropy", type=float, default=None,
                   help="Target entropy for auto-alpha (default: -act_n_dims)")

    # network
    p.add_argument("--d-model", type=int, default=128,
                   help="Transformer hidden dimension")

    # I/O
    p.add_argument("--log-dir",         default=None,
                   help="TensorBoard log directory (disabled if not set)")
    p.add_argument("--ckpt-dir",        default="checkpoints")
    p.add_argument("--render-dir",      default="replays")
    p.add_argument("--render-interval", type=int, default=10,
                   help="Save HTML replay every N episodes")
    p.add_argument("--save-interval",   type=int, default=100,
                   help="Save checkpoint every N episodes")
    p.add_argument("--log-interval",    type=int, default=10)
    p.add_argument("--resume",          default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--cpu",             action="store_true",
                   help="Force CPU even if CUDA is available")
    p.add_argument("--seed",            type=int, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
 
    train(args, MAX_PLANETS=40, MAX_FLEETS=200,
          reward_scheme=[RewardScheme1(win_bonus=20), RewardScheme3(ship_scale=10)])
