"""
SAC v2 training loop for Orbit Wars — dual-opponent mode.

Self-play design
────────────────
Both players share one policy architecture. Owner IDs are swapped before
encoding so the network always sees itself as player 0. Transitions from both
players go into the same replay buffer, doubling collected data.

Player 1 uses a *lagged frozen snapshot* (updated every opponent_update_interval
self-play episodes) to break the non-stationary feedback loop of pure self-play.

Rule-based opponent mode
────────────────────────
Episodes can alternate between self-play and play against rule-based agents
loaded from a directory (e.g. rule_based/). Controlled by rule_based_ratio in
train(). In rule-based episodes only player 0's transitions are stored (the
rule-based agent's moves aren't in the SAC action format).

Use load_rule_based_agents("rule_based/") to auto-import every agent() callable
from Python files in that directory — drop new files in to add more opponents.
"""

import copy
import importlib.util
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Dict, List, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

from kaggle_environments import make as make_kaggle_env
from model.SAC import P_network, Q_network, Encoder
from env.dummy import (
    encode_obs_as_player, decode_action, compute_reward_for_player,
    _obs_to_arrays, _swap_perspective,
    MAX_PLANETS, MAX_FLEETS, STATE_DIM, ACTION_DIM,
)


# =============================================================================
# Rule-based agent loader
# =============================================================================

def load_rule_based_agents(agents_dir: str) -> List[Callable]:
    """
    Import every Python file in agents_dir that exposes an agent(obs) callable.
    Returns a list of those callables. Safe to call with a non-existent dir
    (returns empty list with a warning).
    """
    if not os.path.isdir(agents_dir):
        print(f"  [warn] rule-based agents dir not found: {agents_dir}")
        return []

    agents = []
    for fname in sorted(os.listdir(agents_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        fpath = os.path.join(agents_dir, fname)
        spec = importlib.util.spec_from_file_location(fname[:-3], fpath)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:
            print(f"  [warn] failed to load {fname}: {exc}")
            continue
        if hasattr(mod, "agent"):
            agents.append(mod.agent)
            print(f"  Loaded rule-based agent: {fname}")
        else:
            print(f"  [warn] {fname} has no agent() — skipped")
    return agents


# =============================================================================
# Profiling helper
# =============================================================================

def _print_profile(step_times: list) -> None:
    arr    = np.array(step_times)
    avg_ms = arr.mean() * 1_000
    std_ms = arr.std()  * 1_000
    p95_ms = np.percentile(arr, 95) * 1_000
    print(
        f"  [profile] "
        f"step: {avg_ms:.2f} ± {std_ms:.2f} ms  "
        f"p95: {p95_ms:.2f} ms  |  "
        f"{1_000 / avg_ms:>7.1f} steps/s"
    )


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    """Circular replay buffer backed by pre-allocated numpy arrays."""

    def __init__(self, max_size: int, state_shape: tuple, action_shape: tuple):
        self._max  = int(max_size)
        self._ptr  = 0
        self._size = 0

        self.states      = np.zeros((self._max, *state_shape),  dtype=np.float32)
        self.actions     = np.zeros((self._max, *action_shape), dtype=np.float32)
        self.rewards     = np.zeros((self._max, 1),             dtype=np.float32)
        self.next_states = np.zeros((self._max, *state_shape),  dtype=np.float32)
        self.dones       = np.zeros((self._max, 1),             dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states     [self._ptr] = state
        self.actions    [self._ptr] = action
        self.rewards    [self._ptr] = reward
        self.next_states[self._ptr] = next_state
        self.dones      [self._ptr] = done
        self._ptr  = (self._ptr + 1) % self._max
        self._size = min(self._size + 1, self._max)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.from_numpy(self.states     [idxs]),
            torch.from_numpy(self.actions    [idxs]),
            torch.from_numpy(self.rewards    [idxs]),
            torch.from_numpy(self.next_states[idxs]),
            torch.from_numpy(self.dones      [idxs]),
        )

    def __len__(self):
        return self._size


# =============================================================================
# SAC Trainer
# =============================================================================

class SACTrainer:
    """
    SAC v2 trainer for Orbit Wars with self-play and optional rule-based opponents.

    Network interface:
        policy_net.sample(state)     -> (action [B, MAX_PLANETS, ACTION_DIM], log_prob [B, 1])
        q_net.forward(state, action) -> [B]
    """

    def __init__(
        self,
        obs_shape: tuple,
        act_shape: tuple,
        policy_net: nn.Module,
        q1_net: nn.Module,
        q2_net: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        alpha: float = 0.2,
        replay_buffer_size: int = 100_000,
        batch_size: int = 256,
        opponent_update_interval: int = 10,
        rule_based_agents: Optional[List[Callable]] = None,
        log_dir: Optional[str] = None,
    ):
        self.device     = device
        self.gamma      = gamma
        self.tau        = tau
        self.alpha      = alpha
        self.batch_size = batch_size
        self.opponent_update_interval = opponent_update_interval
        self.rule_based_agents = rule_based_agents or []

        # ── live networks ─────────────────────────────────────────────────────
        self.policy_net = policy_net.to(device)
        self.q1_net     = q1_net.to(device)
        self.q2_net     = q2_net.to(device)
        self.q1_target  = copy.deepcopy(q1_net).to(device)
        self.q2_target  = copy.deepcopy(q2_net).to(device)
        self._hard_update(self.q1_target, self.q1_net)
        self._hard_update(self.q2_target, self.q2_net)

        # ── frozen opponent snapshot (self-play only) ─────────────────────────
        self.opponent_net = copy.deepcopy(policy_net).to(device)
        self.opponent_net.eval()
        for p in self.opponent_net.parameters():
            p.requires_grad_(False)

        # ── optimisers ────────────────────────────────────────────────────────
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.q1_optimizer     = optim.Adam(self.q1_net.parameters(),     lr=learning_rate)
        self.q2_optimizer     = optim.Adam(self.q2_net.parameters(),     lr=learning_rate)

        # ── replay buffer ─────────────────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(replay_buffer_size, obs_shape, act_shape)
        self.train_step    = 0

        # ── observation encoder ───────────────────────────────────────────────
        self.encoder = Encoder(max_planets=MAX_PLANETS, max_fleets=MAX_FLEETS)

        # ── TensorBoard ───────────────────────────────────────────────────────
        self.writer        = SummaryWriter(log_dir=log_dir) if (log_dir and _TB_AVAILABLE) else None
        self._update_count = 0

    # =========================================================================
    # Utilities
    # =========================================================================

    def _hard_update(self, target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def update_opponent(self):
        """Snapshot the current policy into the frozen opponent."""
        self.opponent_net.load_state_dict(self.policy_net.state_dict())

    def _to(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(self.device, non_blocking=True)

    # =========================================================================
    # Action selection
    # =========================================================================

    def select_action(self, state_np: np.ndarray) -> np.ndarray:
        """state_np [seq, feat] → action_np [MAX_PLANETS, ACTION_DIM]."""
        s = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.policy_net.sample(s)
        return action.squeeze(0).cpu().numpy()

    def select_opponent_action(self, state_np: np.ndarray) -> np.ndarray:
        s = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.opponent_net.sample(s)
        return action.squeeze(0).cpu().numpy()

    # =========================================================================
    # Gradient update — SAC v2
    # =========================================================================

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states      = self._to(states)
        actions     = self._to(actions)
        rewards     = self._to(rewards)
        next_states = self._to(next_states)
        dones       = self._to(dones)

        # ── Q-targets ─────────────────────────────────────────────────────────
        with torch.no_grad():
            a_next, lp_next = self.policy_net.sample(next_states)
            q1_next = self.q1_target(next_states, a_next).unsqueeze(-1)
            q2_next = self.q2_target(next_states, a_next).unsqueeze(-1)
            q_target = rewards + (1.0 - dones) * self.gamma * (
                torch.min(q1_next, q2_next) - self.alpha * lp_next
            )

        # ── Q1 update ─────────────────────────────────────────────────────────
        q1_pred = self.q1_net(states, actions).unsqueeze(-1)
        q1_loss = nn.MSELoss()(q1_pred, q_target)
        self.q1_optimizer.zero_grad(); q1_loss.backward(); self.q1_optimizer.step()

        # ── Q2 update ─────────────────────────────────────────────────────────
        q2_pred = self.q2_net(states, actions).unsqueeze(-1)
        q2_loss = nn.MSELoss()(q2_pred, q_target)
        self.q2_optimizer.zero_grad(); q2_loss.backward(); self.q2_optimizer.step()

        # ── Policy update ─────────────────────────────────────────────────────
        a_tilde, lp = self.policy_net.sample(states)
        q1_pi = self.q1_net(states, a_tilde).unsqueeze(-1)
        q2_pi = self.q2_net(states, a_tilde).unsqueeze(-1)
        policy_loss = (self.alpha * lp - torch.min(q1_pi, q2_pi)).mean()
        self.policy_optimizer.zero_grad(); policy_loss.backward(); self.policy_optimizer.step()

        # ── Polyak-update target Q-networks ───────────────────────────────────
        self._soft_update(self.q1_target, self.q1_net)
        self._soft_update(self.q2_target, self.q2_net)

        # ── TensorBoard ───────────────────────────────────────────────────────
        if self.writer is not None:
            s = self._update_count
            self.writer.add_scalar("Loss/q1",              q1_loss.item(),     s)
            self.writer.add_scalar("Loss/q2",              q2_loss.item(),     s)
            self.writer.add_scalar("Loss/policy",          policy_loss.item(), s)
            self.writer.add_scalar("Policy/mean_log_prob", lp.mean().item(),   s)

        self._update_count += 1
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "pi_loss": policy_loss.item(),
        }

    # =========================================================================
    # Episode runner — self-play
    # =========================================================================

    def run_episode(self, warmup_steps: int):
        """
        Run one full self-play episode in the kaggle environment.

        Player 0 uses the live policy; player 1 uses the frozen opponent snapshot.
        Both players' transitions are stored in the same buffer (owner IDs are
        swapped before encoding so each player always appears as player 0).

        Returns:
            transitions : list of (state, action, reward, next_state, done) — both players
            ep_reward_p0: float
        """
        env = make_kaggle_env("orbit_wars", debug=False)
        env.reset()

        obs = env.steps[0][0].observation
        planets_np, _, _, _, _ = _obs_to_arrays(obs)
        initial_planets = planets_np.copy()

        state_p0 = encode_obs_as_player(self.encoder, obs, initial_planets, player_id=0)
        state_p1 = encode_obs_as_player(self.encoder, obs, initial_planets, player_id=1)

        transitions  = []
        ep_reward_p0 = 0.0

        while True:
            _, _, _, omega, _ = _obs_to_arrays(obs)
            planets_now = np.array(obs.planets, dtype=np.float32)
            in_warmup   = self.train_step < warmup_steps

            # ── Player 0: live policy ─────────────────────────────────────────
            if in_warmup:
                action_p0 = np.random.randn(MAX_PLANETS, ACTION_DIM).astype(np.float32)
            else:
                action_p0 = self.select_action(state_p0)

            swapped_p0, _ = _swap_perspective(planets_now, np.empty((0, 7), np.float32), player_id=0)
            moves_p0 = decode_action(action_p0, swapped_p0, omega)

            # ── Player 1: frozen opponent snapshot ────────────────────────────
            if in_warmup:
                action_p1 = np.random.randn(MAX_PLANETS, ACTION_DIM).astype(np.float32)
            else:
                action_p1 = self.select_opponent_action(state_p1)

            swapped_p1, _ = _swap_perspective(planets_now, np.empty((0, 7), np.float32), player_id=1)
            moves_p1 = decode_action(action_p1, swapped_p1, omega)

            # ── Step environment ──────────────────────────────────────────────
            step_results = env.step(actions=[moves_p0, moves_p1])
            new_obs  = step_results[0].observation
            done     = step_results[0].status != "ACTIVE"

            # ── Rewards ───────────────────────────────────────────────────────
            r_p0 = compute_reward_for_player(obs, new_obs, player_id=0)
            r_p1 = compute_reward_for_player(obs, new_obs, player_id=1)

            env_r0 = step_results[0].reward
            env_r1 = step_results[1].reward if len(step_results) > 1 else None

            if abs(env_r0) == 1:
                env_r0 *= 500
            if env_r1 is not None and abs(env_r1) == 1:
                env_r1 *= 500

            if env_r0 is not None:
                r_p0 += float(env_r0)
            if env_r1 is not None:
                r_p1 += float(env_r1)

            # ── Next states ───────────────────────────────────────────────────
            new_state_p0 = encode_obs_as_player(self.encoder, new_obs, initial_planets, player_id=0)
            new_state_p1 = encode_obs_as_player(self.encoder, new_obs, initial_planets, player_id=1)

            transitions.append((state_p0, action_p0, r_p0, new_state_p0, float(done)))
            transitions.append((state_p1, action_p1, r_p1, new_state_p1, float(done)))

            ep_reward_p0 += r_p0
            obs      = new_obs
            state_p0 = new_state_p0
            state_p1 = new_state_p1

            if done:
                break

        return transitions, ep_reward_p0, env

    # =========================================================================
    # Episode runner — rule-based opponent
    # =========================================================================

    def run_episode_vs_rulebased(self, agent_fn: Callable, warmup_steps: int):
        """
        Run one episode where player 1 is driven by a rule-based agent callable.

        agent_fn receives the raw kaggle observation with obs.player == 1 and
        returns a list of moves [[planet_id, angle, ships], ...].

        Only player 0's transitions are stored — the rule-based agent's raw moves
        can't be represented in the SAC action format.

        Returns:
            transitions : list of (state, action, reward, next_state, done) — player 0 only
            ep_reward_p0: float
        """
        env = make_kaggle_env("orbit_wars", debug=False)
        env.reset()

        obs_p0 = env.steps[0][0].observation
        obs_p1 = env.steps[0][1].observation  # player 1's view for the rule-based agent
        planets_np, _, _, _, _ = _obs_to_arrays(obs_p0)
        initial_planets = planets_np.copy()

        state_p0 = encode_obs_as_player(self.encoder, obs_p0, initial_planets, player_id=0)

        transitions  = []
        ep_reward_p0 = 0.0

        while True:
            _, _, _, omega, _ = _obs_to_arrays(obs_p0)
            planets_now = np.array(obs_p0.planets, dtype=np.float32)
            in_warmup   = self.train_step < warmup_steps

            # ── Player 0: live policy ─────────────────────────────────────────
            if in_warmup:
                action_p0 = np.random.randn(MAX_PLANETS, ACTION_DIM).astype(np.float32)
            else:
                action_p0 = self.select_action(state_p0)

            swapped_p0, _ = _swap_perspective(planets_now, np.empty((0, 7), np.float32), player_id=0)
            moves_p0 = decode_action(action_p0, swapped_p0, omega)

            # ── Player 1: rule-based agent ────────────────────────────────────
            moves_p1 = agent_fn(obs_p1) or []

            # ── Step environment ──────────────────────────────────────────────
            step_results = env.step(actions=[moves_p0, moves_p1])
            new_obs_p0   = step_results[0].observation
            done         = step_results[0].status != "ACTIVE"

            # ── Reward for player 0 only ──────────────────────────────────────
            r_p0   = compute_reward_for_player(obs_p0, new_obs_p0, player_id=0)
            env_r0 = step_results[0].reward
            if env_r0 is not None and abs(env_r0) == 1:
                env_r0 *= 500
            if env_r0 is not None:
                r_p0 += float(env_r0)

            # ── Next state ────────────────────────────────────────────────────
            new_state_p0 = encode_obs_as_player(self.encoder, new_obs_p0, initial_planets, player_id=0)

            transitions.append((state_p0, action_p0, r_p0, new_state_p0, float(done)))

            ep_reward_p0 += r_p0
            obs_p0   = new_obs_p0
            obs_p1   = step_results[1].observation  # advance rule-based agent's view
            state_p0 = new_state_p0

            if done:
                break

        return transitions, ep_reward_p0, env

    # =========================================================================
    # Training loop
    # =========================================================================

    def train(
        self,
        num_episodes: int        = 200,
        warmup_steps: int        = 500,
        update_frequency: int    = 4,
        gradient_steps: int      = 1,
        log_interval: int        = 10,
        checkpoint_interval: int = 50,
        checkpoint_path: str     = "sac_model.pt",
        render_interval: int     = 0,
        render_dir: str          = "renders",
        profile: bool            = False,
        rule_based_ratio: float  = 0.5,
    ):
        """
        rule_based_ratio : fraction of episodes that use a rule-based opponent.
            0.0 → pure self-play (same as sac_train.py)
            0.5 → alternating (default when rule_based_agents are loaded)
            1.0 → always rule-based
        Has no effect if no rule_based_agents were provided at construction.
        """
        if render_interval > 0:
            os.makedirs(render_dir, exist_ok=True)

        rb_agents   = self.rule_based_agents
        rb_cursor   = 0   # round-robin index through available rule-based agents
        sp_ep_count = 0   # self-play episode counter for opponent snapshot updates

        ep_rewards: list[float] = []
        _ep_times:  list[float] = []

        for ep in range(num_episodes):
            use_rulebased = bool(rb_agents) and (np.random.random() < rule_based_ratio)

            if use_rulebased:
                agent_fn  = rb_agents[rb_cursor % len(rb_agents)]
                rb_cursor += 1
                transitions, ep_reward, ep_env = self.run_episode_vs_rulebased(agent_fn, warmup_steps)
            else:
                if sp_ep_count % self.opponent_update_interval == 0:
                    self.update_opponent()
                sp_ep_count += 1
                transitions, ep_reward, ep_env = self.run_episode(warmup_steps)

            _t0 = time.perf_counter()

            for s, a, r, ns, d in transitions:
                self.replay_buffer.add(s, a, r, ns, d)
                self.train_step += 1

                if (
                    self.train_step >= warmup_steps
                    and self.train_step % update_frequency == 0
                ):
                    for _ in range(gradient_steps):
                        self.update()

            ep_rewards.append(ep_reward)

            if self.writer is not None:
                mode_tag = "rulebased" if use_rulebased else "selfplay"
                self.writer.add_scalar(f"Reward/episode_{mode_tag}", ep_reward,               ep)
                self.writer.add_scalar("Misc/buffer_fill",           len(self.replay_buffer), ep)
                self.writer.add_scalar("Misc/env_steps",             self.train_step,         ep)

            if profile:
                _ep_times.append(time.perf_counter() - _t0)

            if (ep + 1) % log_interval == 0:
                avg = float(np.mean(ep_rewards[-log_interval:]))
                if self.writer:
                    self.writer.add_scalar("Reward/moving_avg", avg, ep)
                mode_tag = "rb" if use_rulebased else "sp"
                print(
                    f"Ep {ep+1:>4}/{num_episodes} [{mode_tag}] | "
                    f"avg_reward_p0={avg:>9.2f} | "
                    f"buf={len(self.replay_buffer):>6} | "
                    f"steps={self.train_step}"
                )
                if profile and _ep_times:
                    _print_profile(_ep_times)
                    _ep_times.clear()

            if (ep + 1) % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_path)

            if render_interval > 0 and (ep + 1) % render_interval == 0:
                render_path = os.path.join(render_dir, f"ep{ep+1:04d}.html")
                html = ep_env.render(mode="html", width=800, height=600)
                with open(render_path, "w") as f:
                    f.write(html)
                print(f"  Render saved → {render_path}")

        self.save_checkpoint(checkpoint_path)
        return ep_rewards

    # =========================================================================
    # Checkpoint I/O
    # =========================================================================

    def save_checkpoint(self, path: str):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "q1_net":     self.q1_net.state_dict(),
            "q2_net":     self.q2_net.state_dict(),
            "q1_target":  self.q1_target.state_dict(),
            "q2_target":  self.q2_target.state_dict(),
            "train_step": self.train_step,
        }, path)
        print(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.q1_net.load_state_dict(ckpt["q1_net"])
        self.q2_net.load_state_dict(ckpt["q2_net"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.train_step = ckpt["train_step"]
        print(f"Checkpoint loaded ← {path}")

    def close(self):
        if self.writer is not None:
            self.writer.close()


# =============================================================================
# Factory
# =============================================================================

def make_transformer_sac_trainer(**trainer_kwargs) -> SACTrainer:
    """
    SACTrainer wired with transformer Q/Policy networks for the galaxy env.

    State  : [B, MAX_PLANETS + MAX_FLEETS, STATE_DIM]
    Action : [B, MAX_PLANETS, ACTION_DIM]  (raw policy output, no padding needed)
    """
    net_kw = dict(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        max_planets=MAX_PLANETS, max_fleets=MAX_FLEETS,
    )
    obs_shape = (MAX_PLANETS + MAX_FLEETS, STATE_DIM)
    act_shape = (MAX_PLANETS, ACTION_DIM)

    return SACTrainer(
        obs_shape=obs_shape,
        act_shape=act_shape,
        policy_net=P_network(**net_kw),
        q1_net=Q_network(**net_kw),
        q2_net=Q_network(**net_kw),
        **trainer_kwargs,
    )


# =============================================================================
if __name__ == "__main__":
    rb_agents = load_rule_based_agents("rule_based")

    trainer = make_transformer_sac_trainer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=1e-4,
        batch_size=32,
        replay_buffer_size=50_000,
        rule_based_agents=rb_agents,
        log_dir="runs/transformer_sac_dual",
    )

    print("TensorBoard: tensorboard --logdir runs/transformer_sac_dual")

    trainer.train(
        num_episodes=200,
        warmup_steps=500,
        update_frequency=4,
        gradient_steps=1,
        log_interval=10,
        render_interval=10,
        checkpoint_path="sac_model_dual.pt",
        rule_based_ratio=0.5,  # every other episode uses a rule-based opponent
    )

    trainer.close()
