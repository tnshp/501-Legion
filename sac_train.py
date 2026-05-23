import gymnasium as gym
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Dict, Optional, Tuple
from collections import deque
import random

from model.SAC import P_network, Q_network, V_network
from env.dummy import MatrixEnv


class ReplayBuffer:
    """Experience replay buffer for SAC"""

    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(-1),   # [B, 1]
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(-1),     # [B, 1]
        )

    def __len__(self):
        return len(self.buffer)


class SACTrainer:
    """
    Modular Soft Actor-Critic trainer.

    Accepts any policy / Q / V networks that follow the interface:
        policy_net.sample(state)       -> (action, log_prob [B, 1])
        q_net.forward(state, action)   -> [B]
        v_net.forward(state)           -> [B]

    Optional preprocessors let you adapt env observation / action shapes
    before they reach the networks (e.g. slicing sequence-length for the
    transformer variant).

    Args:
        policy_net, q1_net, q2_net, value_net, target_value_net:
            PyTorch modules satisfying the interface above.
        state_preprocessor:
            Called on the batched state tensor before every network forward.
            Use this to trim / pad sequence dimensions for transformer nets.
            Defaults to identity (no-op).
        action_preprocessor:
            Called on the batched action tensor sampled from the buffer.
            Use this to trim stored env actions to the network's expected
            sequence length.  Defaults to identity.
        action_postprocessor:
            Called on the raw numpy action array returned by select_action
            (shape: whatever the policy outputs) and must return an array
            compatible with the env action space.  Defaults to identity.
    """

    def __init__(
        self,
        env,
        policy_net: nn.Module,
        q1_net: nn.Module,
        q2_net: nn.Module,
        value_net: nn.Module,
        target_value_net: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        alpha: float = 0.2,
        replay_buffer_size: int = 100_000,
        batch_size: int = 32,
        state_preprocessor: Optional[Callable] = None,
        action_preprocessor: Optional[Callable] = None,
        action_postprocessor: Optional[Callable] = None,
    ):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        self.state_preprocessor  = state_preprocessor  or (lambda x: x)
        self.action_preprocessor = action_preprocessor or (lambda x: x)
        self.action_postprocessor = action_postprocessor or (lambda x: x)

        # ── networks ──────────────────────────────────────────────────────────
        self.policy_net       = policy_net.to(device)
        self.q1_net           = q1_net.to(device)
        self.q2_net           = q2_net.to(device)
        self.value_net        = value_net.to(device)
        self.target_value_net = target_value_net.to(device)

        self._hard_update(self.target_value_net, self.value_net)

        # ── optimisers ────────────────────────────────────────────────────────
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(),  lr=learning_rate)
        self.q1_optimizer     = optim.Adam(self.q1_net.parameters(),      lr=learning_rate)
        self.q2_optimizer     = optim.Adam(self.q2_net.parameters(),      lr=learning_rate)
        self.value_optimizer  = optim.Adam(self.value_net.parameters(),   lr=learning_rate)

        # ── replay buffer ─────────────────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        self.train_step = 0

    # =========================================================================
    # Helper utilities
    # =========================================================================

    def _hard_update(self, target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    # =========================================================================
    # Action selection (inference)
    # =========================================================================

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Map a raw env observation to an env-compatible action numpy array.
        Preprocessing / postprocessing is handled by the callables passed at
        construction time.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state_t = self.state_preprocessor(state_t)

        with torch.no_grad():
            action, _ = self.policy_net.sample(state_t)

        action_np = action.cpu().numpy()[0]
        return self.action_postprocessor(action_np)

    # =========================================================================
    # Training step
    # =========================================================================

    def update(self) -> Optional[Dict[str, float]]:
        """
        One SAC gradient step.

        Standard SAC objective (Haarnoja et al. 2018):
          Q-target : y = r + γ·(1-d)·V_target(s')
          V-target : v = min(Q1,Q2)(s,ã) − α·log π(ã|s),  ã ~ π
          Q-loss   : E[(Q(s,a) − y)²]
          V-loss   : E[(V(s) − v)²]
          π-loss   : E[α·log π(ã|s) − min(Q1,Q2)(s,ã)]
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states      = self.state_preprocessor(states.to(self.device))
        actions     = self.action_preprocessor(actions.to(self.device))
        rewards     = rewards.to(self.device)       # [B, 1]
        next_states = self.state_preprocessor(next_states.to(self.device))
        dones       = dones.to(self.device)         # [B, 1]

        # ── Q-targets via target value network ────────────────────────────────
        with torch.no_grad():
            v_next   = self.target_value_net(next_states)          # [B]
            q_target = rewards + (1.0 - dones) * self.gamma * v_next.unsqueeze(-1)

        # ── update Q1 ─────────────────────────────────────────────────────────
        q1_pred = self.q1_net(states, actions).unsqueeze(-1)       # [B, 1]
        q1_loss = nn.MSELoss()(q1_pred, q_target)
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        # ── update Q2 ─────────────────────────────────────────────────────────
        q2_pred = self.q2_net(states, actions).unsqueeze(-1)       # [B, 1]
        q2_loss = nn.MSELoss()(q2_pred, q_target)
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # ── update Value network ──────────────────────────────────────────────
        with torch.no_grad():
            a_tilde, lp = self.policy_net.sample(states)           # [B,...], [B,1]
            q1_pi = self.q1_net(states, a_tilde).unsqueeze(-1)
            q2_pi = self.q2_net(states, a_tilde).unsqueeze(-1)
            v_target = torch.min(q1_pi, q2_pi) - self.alpha * lp  # [B, 1]

        v_pred = self.value_net(states).unsqueeze(-1)              # [B, 1]
        v_loss = nn.MSELoss()(v_pred, v_target)
        self.value_optimizer.zero_grad()
        v_loss.backward()
        self.value_optimizer.step()

        # ── update Policy network ─────────────────────────────────────────────
        # Re-sample so gradients flow through the policy
        a_tilde, lp = self.policy_net.sample(states)
        q1_pi = self.q1_net(states, a_tilde).unsqueeze(-1)
        q2_pi = self.q2_net(states, a_tilde).unsqueeze(-1)
        policy_loss = (self.alpha * lp - torch.min(q1_pi, q2_pi)).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ── soft-update target value network ──────────────────────────────────
        self._soft_update(self.target_value_net, self.value_net)

        self.train_step += 1
        return {
            "q1_loss":    q1_loss.item(),
            "q2_loss":    q2_loss.item(),
            "v_loss":     v_loss.item(),
            "pi_loss":    policy_loss.item(),
        }

    # =========================================================================
    # Training loop
    # =========================================================================

    def train(
        self,
        num_episodes: int = 100,
        max_steps_per_episode: int = 500,
        warmup_steps: int = 5000,
        update_frequency: int = 1,
        log_interval: int = 10,
    ):
        """Main training loop."""
        episode_rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0

            for _ in range(max_steps_per_episode):
                if self.train_step < warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.select_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.replay_buffer.add(state, action, reward, next_state, float(done))

                if (
                    len(self.replay_buffer) >= self.batch_size
                    and self.train_step >= warmup_steps
                    and self.train_step % update_frequency == 0
                ):
                    self.update()

                self.train_step += 1
                episode_reward += reward
                state = next_state

                if done:
                    break

            episode_rewards.append(episode_reward)

            if (episode + 1) % log_interval == 0:
                avg = np.mean(episode_rewards[-log_interval:])
                print(
                    f"Ep {episode + 1:>4}/{num_episodes} | "
                    f"Avg reward: {avg:>10.3f} | "
                    f"Buffer: {len(self.replay_buffer):>6} | "
                    f"Steps: {self.train_step}"
                )

        return episode_rewards

    # =========================================================================
    # Checkpoint I/O
    # =========================================================================

    def save_checkpoint(self, path: str):
        torch.save({
            "policy_net":       self.policy_net.state_dict(),
            "q1_net":           self.q1_net.state_dict(),
            "q2_net":           self.q2_net.state_dict(),
            "value_net":        self.value_net.state_dict(),
            "target_value_net": self.target_value_net.state_dict(),
            "train_step":       self.train_step,
        }, path)
        print(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.q1_net.load_state_dict(ckpt["q1_net"])
        self.q2_net.load_state_dict(ckpt["q2_net"])
        self.value_net.load_state_dict(ckpt["value_net"])
        self.target_value_net.load_state_dict(ckpt["target_value_net"])
        self.train_step = ckpt["train_step"]
        print(f"Checkpoint loaded ← {path}")


# =============================================================================
# Factory: transformer variant (original behaviour)
# =============================================================================

def make_transformer_sac_trainer(env, **trainer_kwargs) -> SACTrainer:
    """
    Build a SACTrainer wired up with the transformer Q/V/Policy networks and
    the sequence-slicing preprocessors required for the galaxy-conquest env.

    Dimension conventions (see SACTrainer docstring in sac_train.py):
        state  env → network : [B, 144, 14] → [B, 140, 14]
        action env → network : [B,  44,  8] → [B,  40,  8]
    """
    max_planets = 40
    max_fleets  = 100
    net_seq_len = max_planets + max_fleets   # 140
    action_dim  = 8

    env_action_seq = env.action_space.shape[0]   # 44

    net_kw = dict(
        state_dim=14,
        action_dim=action_dim,
        max_planets=max_planets,
        max_fleets=max_fleets,
    )

    policy_net       = P_network(**net_kw)
    q1_net           = Q_network(**net_kw)
    q2_net           = Q_network(**net_kw)
    value_net        = V_network(state_dim=14, max_planets=max_planets, max_fleets=max_fleets)
    target_value_net = V_network(state_dim=14, max_planets=max_planets, max_fleets=max_fleets)

    def state_pre(s: torch.Tensor) -> torch.Tensor:
        B, S, F = s.shape
        if S > net_seq_len:
            return s[:, :net_seq_len, :]
        if S < net_seq_len:
            return torch.cat([s, torch.zeros(B, net_seq_len - S, F, device=s.device)], dim=1)
        return s

    def action_pre(a: torch.Tensor) -> torch.Tensor:
        B, A, D = a.shape
        if A > max_planets:
            return a[:, :max_planets, :]
        if A < max_planets:
            return torch.cat([a, torch.zeros(B, max_planets - A, D, device=a.device)], dim=1)
        return a

    def action_post(a: np.ndarray) -> np.ndarray:
        if a.shape[0] < env_action_seq:
            pad = np.zeros((env_action_seq - a.shape[0], action_dim), dtype=np.float32)
            a = np.vstack([a, pad])
        return a

    return SACTrainer(
        env=env,
        policy_net=policy_net,
        q1_net=q1_net,
        q2_net=q2_net,
        value_net=value_net,
        target_value_net=target_value_net,
        state_preprocessor=state_pre,
        action_preprocessor=action_pre,
        action_postprocessor=action_post,
        **trainer_kwargs,
    )


# =============================================================================
if __name__ == "__main__":
    env = MatrixEnv(state_dim=14, action_dim=8, max_state=144, max_action=44)

    trainer = make_transformer_sac_trainer(
        env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=1e-4,
        batch_size=32,
        replay_buffer_size=50_000,
    )

    rewards = trainer.train(
        num_episodes=100,
        max_steps_per_episode=50,
        warmup_steps=500,
        update_frequency=1,
        log_interval=10,
    )

    trainer.save_checkpoint("sac_model.pt")
