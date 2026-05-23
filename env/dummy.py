import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env


class MatrixEnv(gym.Env):
    """Custom Environment with n x s continuous state and action spaces."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, state_dim: int = 3, action_dim: int = 2, max_state: int = 144, max_action: int = 44):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.max_state = max_state
        self.max_action = max_action

        # 1. Define the Action Space (Matrix of shape n x s)
        # It is highly recommended to normalize continuous actions to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.max_action, self.action_dim), dtype=np.float32
        )

        # 2. Define the Observation Space (Matrix of shape n x s)
        # We assume arbitrary bounds for the state, e.g., [-100, 100]
        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, shape=(self.max_state, self.state_dim), dtype=np.float32
        )

        #linear project action dim to state dim
        self.action_proj = np.random.rand(self.action_dim, self.state_dim).astype(np.float32)

        # Initialize state variables
        self.state = None
        self.current_step = 0
        self.max_steps = 50

    def reset(self, seed=None, options=None):
        # Handle the random seed for reproducibility
        super().reset(seed=seed)

        # Initialize a random starting state of shape (n, s)
        self.state = self.np_random.uniform(
            -1.0, 1.0, size=(self.max_state, self.state_dim)
        ).astype(np.float32)
        self.current_step = 0

        # Gymnasium reset must return (observation, info)
        info = {}
        return self.state, info

    def step(self, action):
        # SB3 automatically ensures the action matches the space shape (n, s)
        self.current_step += 1

        # Dummy environment logic: transition state based on action
        # For example: next_state = current_state + action
        action_ = np.dot(action, self.action_proj)  # Project action to state dimension
        #pad max action to max state
        if action_.shape[0] < self.max_state:
            padding = np.zeros((self.max_state - action_.shape[0], self.state_dim), dtype=np.float32)
            action_ = np.vstack((action_, padding))

        self.state = self.state + action_

        # Clip state to keep it strictly within observation_space bounds
        self.state = np.clip(
            self.state, self.observation_space.low, self.observation_space.high
        )

        # Dummy reward function: minimize the absolute values in the matrix
        reward = float(-np.sum(np.abs(self.state)))

        # Check termination and truncation conditions
        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {}

        # Gymnasium step must return (observation, reward, terminated, truncated, info)
        return self.state, reward, terminated, truncated, info

    def render(self):
        print(f"Step: {self.current_step} | State Matrix:\n{self.state}")


# ==========================================
# Verification & SB3 Compatibility Check
# ==========================================
if __name__ == "__main__":
    # Initialize env with custom matrix sizes (e.g., 5 x 2)
    env = MatrixEnv(state_dim=14, action_dim=8, max_state=144, max_action=44)

    print("Checking environment compatibility with SB3...")
    # check_env will raise an error or warning if something is wrong
    check_env(env, warn=True)
    print("Environment is 100% compatible with Stable-Baselines3!")

    # Quick test run
    obs, info = env.reset()
    print(f"\nInitial State Shape: {obs.shape}")

    random_action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(random_action)
    print(f"Action Taken Shape: {random_action.shape}")
    print(f"Next State Shape: {obs.shape}")
