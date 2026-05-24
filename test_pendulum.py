"""
SAC v2 on Hopper-v5 — single env and vectorised env examples.

TensorBoard:
    tensorboard --logdir runs/
"""

import gymnasium as gym

from sac_train import SACTrainer
from model.mlp import MLP_Policy, MLP_Q

N_ENVS = 1   # set to 1 for single-env baseline


def make_mlp_sac_trainer(env, **trainer_kwargs) -> SACTrainer:
    """
    Build a SACTrainer (SAC v2) with MLP networks.
    Works with both a plain env and a VectorEnv — the trainer detects which.
    """
    # For VecEnv use single_observation_space; plain env uses observation_space
    obs_space = getattr(env, "single_observation_space", env.observation_space)
    act_space = getattr(env, "single_action_space",      env.action_space)
    state_dim  = obs_space.shape[0]
    action_dim = act_space.shape[0]

    return SACTrainer(
        env=env,
        policy_net=MLP_Policy(state_dim, action_dim, hidden_dim=256),
        q1_net=MLP_Q(state_dim, action_dim, hidden_dim=256),
        q2_net=MLP_Q(state_dim, action_dim, hidden_dim=256),
        **trainer_kwargs,
    )


if __name__ == "__main__":
    # ── environment setup ──────────────────────────────────────────────────────
    if N_ENVS > 1:
        # AsyncVectorEnv runs each env in its own process — use for heavy sims.
        # SyncVectorEnv is simpler and sufficient for lightweight envs.
        env = gym.make_vec("Hopper-v5", num_envs=N_ENVS,
                           vectorization_mode="async")
        log_dir = f"runs/hopper_v5_vec{N_ENVS}"
    else:
        env = gym.make("Hopper-v5")
        log_dir = "runs/hopper_v5_single"

    eval_env = gym.make("Hopper-v5")

    trainer = make_mlp_sac_trainer(
        env,
        device="cuda",
        learning_rate=3e-4,
        gamma=0.99,
        tau=5e-3,
        auto_alpha=True,
        batch_size=256,
        replay_buffer_size=1_000_000,
        log_dir=log_dir,
    )

    print(f"n_envs    : {trainer.n_envs}")
    print(f"Device    : {trainer.device}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    print()

    trainer.train(
        num_episodes=5000,
        max_steps_per_episode=1000,   # single-env only
        warmup_steps=5000,
        gradient_steps=N_ENVS,        # maintain same UTD ratio as single-env
        log_interval=10,
        eval_env=eval_env,
        profile=True,
        eval_interval=50,
        eval_episodes=5,
    )

    trainer.save_checkpoint(f"sac_hopper_v5_n{N_ENVS}.pt")
    trainer.close()
    env.close()
