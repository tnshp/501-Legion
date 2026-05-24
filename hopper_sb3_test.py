import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

def train_hopper_sac():
    # 1. Define paths for saving logs and models
    log_dir = "./tensorboard_logs/"
    model_dir = "./saved_models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 2. Create and wrap the Hopper-v5 environment
    def make_env():
        env = gym.make("Hopper-v5", render_mode=None)
        # Monitor captures episode rewards (ep_rew_mean) and lengths (ep_len_mean)
        env = Monitor(env) 
        return env

    # DummyVecEnv ensures SB3 standard vector logging functions optimally
    env = DummyVecEnv([make_env])

    # 3. Define the explicit 2x256 network architecture 
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],  # Actor (Policy) Network
            qf=[256, 256]   # Critic (Q-Function) Network
        )
    )

    # 4. Initialize the SAC Agent
    model = SAC(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",         
        target_entropy="auto",
        tensorboard_log=log_dir,
        verbose=1,
        device="auto"            
    )

    # 5. Setup a checkpoint callback to save progress
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,        
        save_path=model_dir,
        name_prefix="sac_hopper_v5"
    )

    # 6. Start training
    print("Starting SAC training on Hopper-v5...")
    total_timesteps = 1_000_000
    
    # log_interval=1 forces SB3 to push metrics to TensorBoard every single episode 
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="SAC_Hopper_V5_Run",
        log_interval=1  # <--- Ensures rapid logging of ep_rew_mean
    )

    # 7. Save the final trained model
    final_model_path = os.path.join(model_dir, "sac_hopper_v5_final")
    model.save(final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}")

    env.close()

if __name__ == "__main__":
    train_hopper_sac()
