"""
SAC on Pendulum-v1 using plain MLP networks.

Purpose: verify the modular SACTrainer with a well-understood benchmark.
Pendulum is a good sanity check because:
  - Continuous action space  (action ∈ [-2, 2])
  - Solved SAC should reach ~-200 reward within ~50 episodes
  - Fast to run (no GPU needed)

Usage:
    python test_pendulum.py
"""

import copy
import numpy as np
import torch
import gymnasium as gym

from sac_train import SACTrainer
from model.mlp import MLP_Policy, MLP_Q, MLP_V


def make_mlp_sac_trainer(env, **trainer_kwargs) -> SACTrainer:
    """
    Build a SACTrainer wired up with MLP Q/V/Policy networks.
    No pre/post-processing needed — env observations and actions are already
    flat vectors compatible with the networks.
    """
    state_dim  = env.observation_space.shape[0]   # 3 for Pendulum
    action_dim = env.action_space.shape[0]         # 1 for Pendulum

    policy_net       = MLP_Policy(state_dim, action_dim)
    q1_net           = MLP_Q(state_dim, action_dim)
    q2_net           = MLP_Q(state_dim, action_dim)
    value_net        = MLP_V(state_dim)
    target_value_net = copy.deepcopy(value_net)

    return SACTrainer(
        env=env,
        policy_net=policy_net,
        q1_net=q1_net,
        q2_net=q2_net,
        value_net=value_net,
        target_value_net=target_value_net,
        **trainer_kwargs,
    )


def evaluate(trainer: SACTrainer, env, n_episodes: int = 5) -> float:
    """Run greedy evaluation (mean action, no exploration noise)."""
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total = 0.0
        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(trainer.device)
            with torch.no_grad():
                mu, _ = trainer.policy_net(state_t)
            action = mu.cpu().numpy()[0]
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    return float(np.mean(rewards))


if __name__ == "__main__":
    env      = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")

    trainer = make_mlp_sac_trainer(
        env,
        device="cuda",          # Pendulum is tiny — CPU is fine
        learning_rate=3e-4,
        gamma=0.99,
        tau=5e-3,
        alpha=0.2,
        batch_size=256,
        replay_buffer_size=100_000,
    )

    print(f"State dim : {env.observation_space.shape[0]}")
    print(f"Action dim: {env.action_space.shape[0]}")
    print(f"Device    : {trainer.device}")
    print()

    rewards = trainer.train(
        num_episodes=300,
        max_steps_per_episode=200,
        warmup_steps=1000,
        update_frequency=1,
        log_interval=10,
    )

    eval_reward = evaluate(trainer, eval_env, n_episodes=10)
    print(f"\nFinal greedy eval reward (10 episodes): {eval_reward:.2f}")
    print("Expected: ~-200 or better after 100 episodes on Pendulum-v1")

    trainer.save_checkpoint("sac_pendulum.pt")
