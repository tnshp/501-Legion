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

import json
import copy
import importlib.util
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Callable, Dict, List, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

from kaggle_environments import make as make_kaggle_env
from model.SAC import P_network, Q_network, Encoder
from env.env_utils import (
    encode_obs_as_player, decode_action, compute_reward_for_player,
    _obs_to_arrays, _swap_perspective,
    MAX_PLANETS, MAX_FLEETS, STATE_DIM, ACTION_DIM,
)

_REWARD_NORM = 2000.0

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
        policy_net.sample(state)     -> (action [B, MAX_PLANETS, ACTION_DIM], log_prob [B, 1], n_valid_planets [B, 1])
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
        learning_rate_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        alpha: float = 0.2,
        replay_buffer_size: int = 100_000,
        batch_size: int = 256,
        opponent_update_interval: int = 10,
        checkpoint_load_path: str= None,
        training_data_load_path: str= None,
        replay_buffer_load_path: str = None,
        rule_based_agents: Optional[List[Callable]] = None,
        log_dir: Optional[str] = None,
        # td lambda params
        use_lambda_returns: bool = False,
        lambda_return: float = 0.9,
        cache_size: int = 8000,
        block_size: int = 50,
        refresh_freq: int = 1000,
        # gradient clipping / auto-alpha
        max_grad_norm: Optional[float] = None,
        auto_alpha: bool = False,
        target_entropy: Optional[float] = None,
        alpha_min: float = 0.05,
        alpha_max: float = 1.0,
    ):
        self.device     = device
        self._amp       = str(device).startswith("cuda")
        self.gamma      = gamma
        self.tau        = tau
        self.alpha      = alpha
        self.batch_size = batch_size
        self.opponent_update_interval = opponent_update_interval
        self.rule_based_agents = rule_based_agents or []


        # ── cache-based TD(λ) (Daley & Amato 2019) ────────────────────────────
        # When enabled, Q-targets are precomputed λ-returns stored in a cache that
        # is refreshed every `refresh_freq` env-steps — this REPLACES the target
        # network (no Polyak averaging, no min-over-target-nets). See _lambda_*.
        self.use_lambda_returns = use_lambda_returns
        self.lambda_return      = float(lambda_return)
        self.cache_size         = int(cache_size)
        self.block_size         = int(block_size)
        self.refresh_freq       = int(refresh_freq)
        self._cache_states      = None
        self._cache_actions     = None
        self._cache_targets     = None
        self._cache_n           = 0
        self._last_cache_step   = -(10 ** 9)
        self._cache_refreshes   = 0

        # ── gradient clipping / auto-alpha ────────────────────────────────────
        self.max_grad_norm = max_grad_norm
        self._nan_skips    = 0

        self.training_data = None
        if training_data_load_path is not None:
            with open(training_data_load_path, "r") as file:
                self.training_data = json.load(file)
        
        self.auto_alpha = auto_alpha
        # Per-sample action dimensionality used to scale a *dynamic* target
        # entropy by how many planets actually exist this step (see below) —
        # the static -prod(act_shape) heuristic overcounts padded planet slots.
        self._action_dim = int(act_shape[-1])
        if auto_alpha:
            # If the user pins target_entropy explicitly, honour it as a fixed
            # scalar (back-compat / experimentation). Otherwise leave it None:
            # update()/_lambda_update() then compute a per-sample target of
            # -n_valid_planets * action_dim, since log_prob is now masked to
            # only sum over planets that exist this step (see P_network.sample).
            self.target_entropy  = float(target_entropy) if target_entropy is not None else None
            self.log_alpha       = nn.Parameter(torch.zeros(1, device=device))
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate_alpha)
            # Bounds on alpha — keeps the dual-gradient auto-tuning from running
            # away to ~0 (policy collapse) or blowing up, without disabling it.
            self._log_alpha_min  = float(np.log(alpha_min))
            self._log_alpha_max  = float(np.log(alpha_max))
        else:
            self.target_entropy  = None
            self.log_alpha       = None
            self.alpha_optimizer = None

        # identity preprocessors (hook point for future normalization)
        self.state_preprocessor  = lambda x: x
        self.action_preprocessor = lambda x: x

        # ── live networks ─────────────────────────────────────────────────────
        self.policy_net = policy_net.to(device)
        self.q1_net     = q1_net.to(device)
        self.q2_net     = q2_net.to(device)
        # self.q1_target  = copy.deepcopy(q1_net).to(device)
        # self.q2_target  = copy.deepcopy(q2_net).to(device)
        # self._hard_update(self.q1_target, self.q1_net)
        # self._hard_update(self.q2_target, self.q2_net)

        if not self.use_lambda_returns:
            self.q1_target = copy.deepcopy(q1_net).to(device)
            self.q2_target = copy.deepcopy(q2_net).to(device)
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
        self._tb_log_every = 50  # gradient steps between loss scalar writes

        if checkpoint_load_path is not None:
            self.load_checkpoint(checkpoint_load_path)

        self.saved_buffer = None
        if replay_buffer_load_path is not None:
            self.saved_buffer = replay_buffer_load_path

        self.pitime = 0
        self.envtime = 0
        self.updatetime = 0

        # ── torch.compile (CUDA only) ─────────────────────────────────────────
        # Must come after load_checkpoint() so the checkpoint keys (no _orig_mod.
        # prefix) match the uncompiled state dict, and after all deepcopy() calls
        # so opponent/target snapshots remain separate uncompiled modules.
        if str(device).startswith("cuda"):
            self.policy_net   = torch.compile(self.policy_net)
            self.q1_net       = torch.compile(self.q1_net)
            self.q2_net       = torch.compile(self.q2_net)
            self.opponent_net = torch.compile(self.opponent_net)
            if not self.use_lambda_returns:
                self.q1_target = torch.compile(self.q1_target)
                self.q2_target = torch.compile(self.q2_target)

    # =========================================================================
    # Utilities
    # =========================================================================

    def _hard_update(self, target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())

    def _soft_update(self, target: nn.Module, source: nn.Module):
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.lerp_(sp, self.tau)

    def _clamp_log_alpha(self):
        """Keep alpha within [alpha_min, alpha_max] after each dual-gradient step."""
        with torch.no_grad():
            self.log_alpha.data.clamp_(self._log_alpha_min, self._log_alpha_max)

    def _target_entropy_for(self, n_valid: torch.Tensor):
        """Per-sample target entropy for the auto-alpha dual gradient.

        Uses a fixed override if one was supplied; otherwise scales with how
        many planets actually exist this step (-n_valid * action_dim), since
        log_prob (from P_network.sample) is masked to sum only over those
        planets. A static -prod(act_shape) target assumes every padded slot
        carries real entropy, which is unreachable and drives alpha to 0.
        """
        if self.target_entropy is not None:
            return self.target_entropy
        return -(n_valid.float() * self._action_dim)

    def _grad_norm(self, net: nn.Module) -> float:
        total = 0.0
        for p in net.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

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
        s = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy_net.sample(s)
        return action.squeeze(0).cpu().numpy()

    def select_opponent_action(self, state_np: np.ndarray) -> np.ndarray:
        s = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.opponent_net.sample(s)
        return action.squeeze(0).cpu().numpy()

    # =========================================================================
    # Gradient update — SAC v2
    # =========================================================================

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        if self.use_lambda_returns :
            return self._lambda_update()

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states      = self._to(states)
        actions     = self._to(actions)
        rewards     = self._to(rewards) / _REWARD_NORM
        next_states = self._to(next_states)
        dones       = self._to(dones)

        _amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._amp)

        # ── Q-targets ─────────────────────────────────────────────────────────
        with _amp, torch.no_grad():
            a_next, lp_next, _ = self.policy_net.sample(next_states)
            q1_next = self.q1_target(next_states, a_next).unsqueeze(-1)
            q2_next = self.q2_target(next_states, a_next).unsqueeze(-1)
            q_target = rewards + (1.0 - dones) * self.gamma * (
                torch.min(q1_next, q2_next) - self.alpha * lp_next
            )

        # ── Q1 + Q2 update (single backward over disjoint parameter sets) ────
        with _amp:
            q1_pred = self.q1_net(states, actions).unsqueeze(-1)
            q2_pred = self.q2_net(states, actions).unsqueeze(-1)
            q1_loss = F.mse_loss(q1_pred, q_target)
            q2_loss = F.mse_loss(q2_pred, q_target)
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # ── Policy update ─────────────────────────────────────────────────────
        with _amp:
            a_tilde, lp, n_valid = self.policy_net.sample(states)
            q1_pi = self.q1_net(states, a_tilde).unsqueeze(-1)
            q2_pi = self.q2_net(states, a_tilde).unsqueeze(-1)
            policy_loss = (self.alpha * lp - torch.min(q1_pi, q2_pi)).mean()
        self.policy_optimizer.zero_grad(); policy_loss.backward(); self.policy_optimizer.step()

        # ── Polyak-update target Q-networks ───────────────────────────────────
        self._soft_update(self.q1_target, self.q1_net)
        self._soft_update(self.q2_target, self.q2_net)

        # ── Auto-alpha ────────────────────────────────────────────────────────
        if self.auto_alpha:
            with _amp:
                target_entropy = self._target_entropy_for(n_valid)
                alpha_loss = -(self.log_alpha.exp() * (lp.detach() + target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self._clamp_log_alpha()
            self.alpha = self.log_alpha.exp().item()

        self._update_count += 1

        if self.writer is not None and self._update_count % self._tb_log_every == 0:
            s    = self._update_count
            q1_v = q1_loss.item()
            q2_v = q2_loss.item()
            pi_v = policy_loss.item()
            lp_v = lp.mean().item()
            q1_mag = q1_pred.float().mean().item()
            self.writer.add_scalar("Loss/q1",              q1_v,   s)
            self.writer.add_scalar("Loss/q2",              q2_v,   s)
            self.writer.add_scalar("Loss/policy",          pi_v,   s)
            self.writer.add_scalar("Policy/mean_log_prob", lp_v,   s)
            self.writer.add_scalar("Loss/mean_q1_pred",    q1_mag, s)
            self.writer.add_scalar("Loss/mean_q_target",   q_target.mean(), s)
            return {"q1_loss": q1_v, "q2_loss": q2_v, "pi_loss": pi_v}

        return None

    # =========================================================================
    # Episode runner — self-play
    # =========================================================================

    def run_episode(self, warmup_steps: int, num_opps: int):
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
        env.reset(num_opps+1)

        obs = env.steps[0][0].observation
        planets_np, _, _, _, _ = _obs_to_arrays(obs)
        initial_planets = planets_np.copy()

        state_p0 = encode_obs_as_player(self.encoder, obs, initial_planets, player_id=0)
        state_p1 = encode_obs_as_player(self.encoder, obs, initial_planets, player_id=1)
        if num_opps>2:
            state_p2 = encode_obs_as_player(self.encoder, obs, initial_planets, player_id=2)
            state_p3 = encode_obs_as_player(self.encoder, obs, initial_planets, player_id=3)


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

            opp_moves = []
            opp_actions = []
            for num in range(num_opps):
                # ── Player 1: frozen opponent snapshot ────────────────────────────
                if in_warmup:
                    action = np.random.randn(MAX_PLANETS, ACTION_DIM).astype(np.float32)
                else:
                    action = self.select_opponent_action(state_p1)

                swapped_p1, _ = _swap_perspective(planets_now, np.empty((0, 7), np.float32), player_id=num+1)
                moves_p1 = decode_action(action, swapped_p1, omega)
                opp_moves.append(moves_p1)
                opp_actions.append(action)

            # ── Step environment ──────────────────────────────────────────────
            step_results = env.step(actions= [moves_p0] + opp_moves)
            new_obs  = step_results[0].observation
            done     = step_results[0].status != "ACTIVE"

            # ── Rewards ───────────────────────────────────────────────────────
            r_p0 = compute_reward_for_player(obs, new_obs, player_id=0, num_opps = num_opps)
            r_p1 = compute_reward_for_player(obs, new_obs, player_id=1, num_opps = num_opps)

            env_r0 = step_results[0].reward
            env_r1 = step_results[1].reward if len(step_results) > 1 else None

            
            won = env_r0>0
            if abs(env_r0) >= 1:
                env_r0 *= 2000
            if env_r1 is not None and abs(env_r1) >= 1:
                env_r1 *= 2000

            if env_r0 is not None:
                r_p0 += float(env_r0)
            if env_r1 is not None:
                r_p1 += float(env_r1)

            # ── Next states ───────────────────────────────────────────────────
            new_state_p0 = encode_obs_as_player(self.encoder, new_obs, initial_planets, player_id=0)
            new_state_p1 = encode_obs_as_player(self.encoder, new_obs, initial_planets, player_id=1)


            transitions.append((state_p0, action_p0, r_p0, new_state_p0, float(done)))
            transitions.append((state_p1, opp_actions[0], r_p1, new_state_p1, float(done)))

            ep_reward_p0 += r_p0
            obs      = new_obs
            state_p0 = new_state_p0
            state_p1 = new_state_p1
            
            if num_opps>2:
                r_p2 = compute_reward_for_player(obs, new_obs, player_id=2, num_opps= num_opps)
                r_p3 = compute_reward_for_player(obs, new_obs, player_id=3, num_opps= num_opps)
                env_r2 = step_results[2].reward if len(step_results) > 2 else None
                env_r3 = step_results[3].reward if len(step_results) > 3 else None
                if env_r2 is not None and abs(env_r2) >= 1:
                    env_r2 *= 2000
                if env_r2 is not None:
                    r_p2 += float(env_r2)
                if env_r3 is not None and abs(env_r3) >= 1:
                    env_r3 *= 2000
                if env_r3 is not None:
                    r_p3 += float(env_r3)

                new_state_p2 = encode_obs_as_player(self.encoder, new_obs, initial_planets, player_id=2)
                new_state_p3 = encode_obs_as_player(self.encoder, new_obs, initial_planets, player_id=3)

                #transitions
                transitions.append((state_p2, opp_actions[1], r_p2, new_state_p2, float(done)))
                transitions.append((state_p3, opp_actions[2], r_p3, new_state_p3, float(done)))

                state_p2 = new_state_p2
                state_p3 = new_state_p3

            if done:
                break

        return transitions, ep_reward_p0, env, won

    # =========================================================================
    # Episode runner — rule-based opponent
    # =========================================================================

    def run_episode_vs_rulebased(self, agent_fns: Callable, warmup_steps: int, opponent_noise: float = 0.0):
        """
        Run one episode where player 1 is driven by a rule-based agent callable.

        agent_fn receives the raw kaggle observation with obs.player == 1 and
        returns a list of moves [[planet_id, angle, ships], ...].

        opponent_noise: probability [0, 1] that the rule-based agent's move is
        replaced by a random decoded action, making it a weaker/noisier opponent.

        Only player 0's transitions are stored — the rule-based agent's raw moves
        can't be represented in the SAC action format.

        Returns:
            transitions : list of (state, action, reward, next_state, done) — player 0 only
            ep_reward_p0: float
        """
        env = make_kaggle_env("orbit_wars", debug=False)
        is4p = len(agent_fns)>1
        num_opps = len(agent_fns)
        env.reset(4 if is4p else 2)
        obs_p0 = env.steps[0][0].observation
        opp_obs = []
        obs_p1 = env.steps[0][1].observation  # player 1's view for the rule-based agent
        opp_obs.append(obs_p1)
        obs_p2 = None
        obs_p3 = None
        if is4p:
            obs_p2 = env.steps[0][2].observation  # player 2's view for the rule-based agent
            obs_p3 = env.steps[0][3].observation  # player 3's view for the rule-based agent
            opp_obs.append(obs_p2)
            opp_obs.append(obs_p3)

        planets_np, _, _, _, _ = _obs_to_arrays(obs_p0)
        initial_planets = planets_np.copy()

        state_p0 = encode_obs_as_player(self.encoder, obs_p0, initial_planets, player_id=0)

        transitions  = []
        ep_reward_p0 = 0.0

        while True:
            _, _, _, omega, _ = _obs_to_arrays(obs_p0)
            planets_now = np.array(obs_p0.planets, dtype=np.float32)
            in_warmup   = self.train_step < warmup_steps
            
            t0 = time.time()

            # ── Player 0: live policy ─────────────────────────────────────────
            if in_warmup:
                action_p0 = np.random.randn(MAX_PLANETS, ACTION_DIM).astype(np.float32)
            else:
                action_p0 = self.select_action(state_p0)

            self.pitime += time.time() - t0

            t0 = time.time()

            swapped_p0, _ = _swap_perspective(planets_now, np.empty((0, 7), np.float32), player_id=0)
            moves_p0 = decode_action(action_p0, swapped_p0, omega)
            moves_opp = []

            for i in range(3 if is4p else 1):
                if opponent_noise > 0.0 and np.random.random() < opponent_noise:
                    rand_action_p1 = np.random.randn(MAX_PLANETS, ACTION_DIM).astype(np.float32)
                    swapped_p1, _ = _swap_perspective(planets_now, np.empty((0, 7), np.float32), player_id=i + 1)
                    moves_p1 = decode_action(rand_action_p1, swapped_p1, omega)
                else:
                    moves_p1 = agent_fns[0](opp_obs[i]) or []
                moves_opp.append(moves_p1)

            # ── Step environment ──────────────────────────────────────────────
            step_results = env.step(actions= [moves_p0] + moves_opp)
            new_obs_p0   = step_results[0].observation
            done         = step_results[0].status != "ACTIVE"

            # ── Reward for player 0 only ──────────────────────────────────────
            r_p0   = compute_reward_for_player(obs_p0, new_obs_p0, player_id=0, num_opps=num_opps)
            env_r0 = step_results[0].reward
            won = env_r0>0
            if env_r0 is not None and abs(env_r0) >= 1:
                env_r0 *= 2000
            if env_r0 is not None:
                r_p0 += float(env_r0)

            # ── Next state ────────────────────────────────────────────────────
            new_state_p0 = encode_obs_as_player(self.encoder, new_obs_p0, initial_planets, player_id=0)

            transitions.append((state_p0, action_p0, r_p0, new_state_p0, float(done)))

            ep_reward_p0 += r_p0
            obs_p0   = new_obs_p0
            opp_obs[0]   = step_results[1].observation  # advance rule-based agent's view
            if is4p:
                opp_obs[1] = step_results[2].observation
                opp_obs[2] = step_results[3].observation
            state_p0 = new_state_p0

            self.envtime+= time.time() - t0

            if done:
                break

        return transitions, ep_reward_p0, env, won
    
        # =========================================================================
    # Cache-based TD(λ)  (Daley & Amato, "Reconciling λ-Returns with
    # Experience Replay", NeurIPS 2019)
    #
    # Why: in Orbit Wars the reward for launching a fleet only lands ~10-20
    # steps later (flight time), so a 1-step (TD(0)) target assigns credit very
    # slowly. λ-returns interpolate between TD(0) and Monte-Carlo, propagating
    # that delayed reward back over many steps in a single update.
    #
    # How (faithful to the paper, fixed λ):
    #   • Promote S/B contiguous "blocks" of B transitions from the replay buffer
    #     into a cache C of size S (`_build_lambda_cache`). Because the buffer is
    #     filled in temporal order, a contiguous block IS a trajectory; the stored
    #     `done` flags cut returns at episode boundaries.
    #   • Compute each block's λ-returns backwards by recursion (Eq. 8 / footnote
    #     4), reusing one bootstrap value per transition — O(B) Q-evaluations per
    #     block instead of O(B²) (paper §3.1).
    #   • The stored returns are stable TD targets, so the target network is
    #     eliminated. The cache is rebuilt every `refresh_freq` env-steps using
    #     the current networks (paper §3.2).
    #   • SAC adaptation: the bootstrap is the SOFT value
    #         V(s') = min(Q1,Q2)(s',a') − α·logπ(a'|s'),   a' ~ π(·|s')
    #     so at λ=0 this reduces exactly to the standard SAC 1-step target.
    # =========================================================================

    def _soft_value(self, next_states_np: np.ndarray) -> np.ndarray:
        """Batched soft state-value V(s') for the cache bootstrap, computed in
        chunks under no_grad. Returns a 1-D numpy array (NaN/Inf → 0)."""
        out   = []
        chunk = max(self.batch_size, 256)
        _amp  = torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._amp)
        with torch.no_grad(), _amp:
            for i in range(0, len(next_states_np), chunk):
                ns = self.state_preprocessor(self._to(torch.from_numpy(next_states_np[i:i + chunk])))
                a, lp, _ = self.policy_net.sample(ns)
                q1 = self.q1_net(ns, a).unsqueeze(-1)
                q2 = self.q2_net(ns, a).unsqueeze(-1)
                v  = torch.min(q1, q2) - self.alpha * lp
                out.append(v.squeeze(-1).float().cpu().numpy())
        v_all = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
        return np.nan_to_num(v_all, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _build_lambda_cache(self) -> bool:
        """Promote S/B temporally-contiguous blocks into the cache and fill it
        with precomputed λ-return targets. Returns True on success."""
        buf = self.replay_buffer
        B   = self.block_size
        if len(buf) < B:
            return False
        n_blocks = max(1, self.cache_size // B)

        # ── sample valid block start indices (no seam-straddling) ─────────────
        if buf._size < buf._max:
            # Buffer not yet full: valid transitions live in [0, _size).
            starts = np.random.randint(0, buf._size - B + 1, size=n_blocks)
            wrap   = False
        else:
            # Full buffer: the only physical discontinuity is the write pointer
            # (newest|oldest seam). Reject blocks whose interior crosses it.
            ptr    = buf._ptr
            starts = np.empty(n_blocks, dtype=np.int64)
            filled = 0
            while filled < n_blocks:
                cand = np.random.randint(0, buf._max - B + 1, size=n_blocks - filled)
                ok   = ~((cand < ptr) & (ptr < cand + B))
                good = cand[ok]
                starts[filled:filled + len(good)] = good
                filled += len(good)
            wrap = True

        block_idx = starts[:, None] + np.arange(B)[None, :]        # [n_blocks, B]
        if wrap:
            block_idx %= buf._max
        flat_idx = block_idx.reshape(-1)

        states_np      = buf.states     [flat_idx]
        actions_np     = buf.actions    [flat_idx]
        rewards_np     = buf.rewards    [flat_idx].reshape(n_blocks, B) / _REWARD_NORM
        next_states_np = buf.next_states[flat_idx]
        dones_np       = buf.dones      [flat_idx].reshape(n_blocks, B)

        # ── soft-value bootstrap for every transition's next-state ────────────
        v_next = self._soft_value(next_states_np).reshape(n_blocks, B)

        # ── backward λ-return recursion (vectorised across blocks) ────────────
        #   Rλ_t = r_t + γ(1−d_t)[λ·Rλ_{t+1} + (1−λ)·V(s_{t+1})]
        # The last transition in a block has no in-block successor, so its
        # bootstrap falls back to V(s') (a plain 1-step target there).
        targets = np.empty((n_blocks, B), dtype=np.float32)
        lam, gam = self.lambda_return, self.gamma
        g = v_next[:, B - 1].copy()
        for t in range(B - 1, -1, -1):
            d    = dones_np[:, t]
            vt   = v_next[:, t]
            boot = vt if t == B - 1 else g
            g = rewards_np[:, t] + gam * (1.0 - d) * (lam * boot + (1.0 - lam) * vt)
            targets[:, t] = g

        self._cache_states  = states_np
        self._cache_actions = actions_np
        self._cache_targets = targets.reshape(-1, 1).astype(np.float32)
        self._cache_n       = self._cache_targets.shape[0]
        return True

    def _lambda_update(self) -> Optional[Dict[str, float]]:
        """One gradient step against the λ-return cache (no target network)."""
        # ── refresh the cache on schedule (every refresh_freq env-steps) ──────
        if (self._cache_targets is None
                or self.train_step - self._last_cache_step >= self.refresh_freq):
            if self._build_lambda_cache():
                self._last_cache_step = self.train_step
                self._cache_refreshes += 1
                if self.writer is not None:
                    self.writer.add_scalar("Cache/refreshes",
                                           self._cache_refreshes, self._update_count)
                    self.writer.add_scalar("Cache/target_mean",
                                           float(self._cache_targets.mean()),
                                           self._update_count)
        if self._cache_targets is None:
            return None

        idx     = np.random.randint(0, self._cache_n, size=self.batch_size)
        states  = self.state_preprocessor (self._to(torch.from_numpy(self._cache_states[idx])))
        actions = self.action_preprocessor(self._to(torch.from_numpy(self._cache_actions[idx])))
        targets = self._to(torch.from_numpy(self._cache_targets[idx]))

        if not torch.isfinite(targets).all():
            self._nan_skips += 1
            if self.writer is not None:
                self.writer.add_scalar("Misc/nan_skips", self._nan_skips, self._update_count)
            return None

        _amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._amp)

        # ── Q updates toward the precomputed λ-return (single backward) ──────
        with _amp:
            q1_pred = self.q1_net(states, actions).unsqueeze(-1)
            q2_pred = self.q2_net(states, actions).unsqueeze(-1)
            q1_loss = F.mse_loss(q1_pred, targets)
            q2_loss = F.mse_loss(q2_pred, targets)
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q1_net.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.q2_net.parameters(), self.max_grad_norm)
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # ── Policy update — identical SAC actor objective ─────────────────────
        with _amp:
            a_tilde, lp, n_valid = self.policy_net.sample(states)
            q1_pi = self.q1_net(states, a_tilde).unsqueeze(-1)
            q2_pi = self.q2_net(states, a_tilde).unsqueeze(-1)
            policy_loss = (self.alpha * lp - torch.min(q1_pi, q2_pi)).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        # NB: no target-network Polyak update — the cache is the stable target.

        # ── Auto-alpha ────────────────────────────────────────────────────────
        if self.auto_alpha:
            with _amp:
                target_entropy = self._target_entropy_for(n_valid)
                alpha_loss = -(self.log_alpha.exp() * (lp.detach() + target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self._clamp_log_alpha()
            self.alpha = self.log_alpha.exp().item()

        # ── TensorBoard ───────────────────────────────────────────────────────
        if self.writer is not None and self._update_count % self._tb_log_every == 0:
            s = self._update_count
            self.writer.add_scalar("Loss/q1",              q1_loss.item(),     s)
            self.writer.add_scalar("Loss/q2",              q2_loss.item(),     s)
            self.writer.add_scalar("Loss/policy",          policy_loss.item(), s)
            self.writer.add_scalar("Policy/mean_log_prob", lp.mean().item(),   s)
            self.writer.add_scalar("Policy/mean_n_valid_planets", n_valid.float().mean().item(), s)
            self.writer.add_scalar("Q/target_mean",        targets.mean().item(), s)
            self.writer.add_scalar("Alpha/value",          self.alpha,         s)
            self.writer.add_scalar("GradNorm/policy", self._grad_norm(self.policy_net), s)
            self.writer.add_scalar("GradNorm/q1",     self._grad_norm(self.q1_net),    s)
            self.writer.add_scalar("GradNorm/q2",     self._grad_norm(self.q2_net),    s)
            if self.auto_alpha:
                self.writer.add_scalar("Loss/alpha", alpha_loss.item(), s)
                self.writer.add_scalar("Alpha/target_entropy_mean",
                                       target_entropy.float().mean().item()
                                       if torch.is_tensor(target_entropy) else target_entropy,
                                       s)

        self._update_count += 1
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "pi_loss": policy_loss.item(),
        }


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
        checkpoint_path: str     = "checkpoints",
        render_interval: int     = 10,
        replay_storage_interval  = 50,
        render_dir: str          = "renders",
        profile: bool            = False,
        rule_based_ratio: float  = 1.0,
        opponent_noise: float    = 0.3,
        prob_4p : float          = 0.5,
    ):
        """
        rule_based_ratio : fraction of episodes that use a rule-based opponent.
            0.0 → pure self-play
            1.0 → always rule-based (default)
        Has no effect if no rule_based_agents were provided at construction.

        opponent_noise : probability that the rule-based opponent takes a random
        action on any given step, making it a weaker/noisier training target.
        0.0 → fully deterministic rule-based, 1.0 → fully random.
        """
        if render_interval > 0:
            os.makedirs(render_dir, exist_ok=True)


        rb_agents   = self.rule_based_agents
        rb_cursor   = 0   # round-robin index through available rule-based agents
        sp_ep_count = 0   # self-play episode counter for opponent snapshot updates

        ep_rewards: list[float] = []
        ep_wins:    list[bool]  = []
        _ep_times:  list[float] = []
        
        start = 0
        win_count_consec = 0
        opponent_noise_checkpoint = opponent_noise
        if self.training_data is not None:
            start = self.training_data["ep"]
            num_episodes = self.training_data["num_episodes"]
            opponent_noise_checkpoint = self.training_data["noise"]

        if self.saved_buffer is not None:
            self.load_replay_buffer(self.saved_buffer)
        
        opp_noise_inc = 0.05
        opponent_noise = opponent_noise_checkpoint    

        for ep in range(start, num_episodes):


            num_opps = 3 if np.random.random()<prob_4p else 1
            use_rulebased = bool(rb_agents) and (np.random.random() < rule_based_ratio)

            if use_rulebased:
                agent_fns = []
                for num in range(num_opps):
                    agent_fns.append(rb_agents[rb_cursor % len(rb_agents)])
                    rb_cursor += 1
                transitions, ep_reward, ep_env, won = self.run_episode_vs_rulebased(agent_fns, warmup_steps, opponent_noise)
            else:
                if sp_ep_count % self.opponent_update_interval == 0:
                    self.update_opponent()
                sp_ep_count += 1
                transitions, ep_reward, ep_env, won = self.run_episode(warmup_steps, num_opps)

            _t0 = time.perf_counter()

            t0 = time.time()

            for s, a, r, ns, d in transitions:
                self.replay_buffer.add(s, a, r, ns, d)
                self.train_step += 1

                if (
                    self.train_step >= warmup_steps
                    and self.train_step % update_frequency == 0
                ):
                    for _ in range(gradient_steps):
                        self.update()

            self.updatetime+= time.time() - t0

            ep_rewards.append(ep_reward)
            ep_wins.append(won)

            if self.writer is not None:
                mode_tag = "rulebased" if use_rulebased else "selfplay"
                ep_steps = len(transitions) // (1 if use_rulebased else 2)
                self.writer.add_scalar(f"Reward/episode_{mode_tag}", ep_reward,                     ep)
                self.writer.add_scalar(f"Reward/{mode_tag}_normed", ep_reward/_REWARD_NORM, ep)
                self.writer.add_scalar("Misc/buffer_fill",           len(self.replay_buffer),       ep)
                self.writer.add_scalar("Misc/env_steps",             self.train_step,               ep)
                self.writer.add_scalar("Misc/episode_length",        ep_steps,                      ep)
                self.writer.add_scalar("Misc/opponent_noise",        opponent_noise,                ep)
                self.writer.add_scalar("Reward/won",                 float(won),                    ep)

            if profile:
                _ep_times.append(time.perf_counter() - _t0)

            if (ep + 1) % log_interval == 0:
                avg      = float(np.mean(ep_rewards[-log_interval:]))
                win_rate = float(np.mean(ep_wins[-log_interval:]))
                if self.writer:
                    self.writer.add_scalar("Reward/moving_avg", avg,      ep)
                    self.writer.add_scalar("Reward/win_rate",   win_rate, ep)
                mode_tag = "rb" if use_rulebased else "sp"
                print(
                    f"Ep {ep+1:>4}/{num_episodes} [{mode_tag}] | "
                    f"avg_reward={avg:>9.2f} | "
                    f"win_rate={win_rate:.0%} | "
                    f"buf={len(self.replay_buffer):>6} | "
                    f"steps={self.train_step}"
                )
                print(f"pitime = {self.pitime}")
                print(f"envtime = {self.envtime}")
                print(f"updatetime = {self.updatetime}")
                if profile and _ep_times:
                    _print_profile(_ep_times)
                    _ep_times.clear()
            
            if (ep+1) % replay_storage_interval == 0:
                self.save_replay_buffer(checkpoint_path + f"/EP{ep+1}_replay_buffer.pt")

            if (ep + 1) % checkpoint_interval == 0:
                train_dict = {"ep": ep+1, "noise": opponent_noise, "num_episodes": num_episodes}
                self.save_checkpoint(checkpoint_path + f"/EP{ep+1}_.pt", train_dict)

            if render_interval > 0 and (ep + 1) % render_interval == 0:
                render_path = os.path.join(render_dir, f"ep_{ep+1}_{won}.html")
                html = ep_env.render(mode="html", width=800, height=600)
                with open(render_path, "w") as f:
                    f.write(html)
                print(f"  Render saved → {render_path}")
            if won and use_rulebased:
                win_count_consec+=1
            elif use_rulebased and not won:
                win_count_consec = 0
            if win_count_consec>=3:
                win_count_consec = 0
                opponent_noise-=opp_noise_inc

        self.save_checkpoint(checkpoint_path + f"/final/EP{ep+1}_.pt", train_dict)
        return ep_rewards

    # =========================================================================
    # Checkpoint I/O
    # =========================================================================
    def save_checkpoint(self, path: str, train_dict = None, replay_buffer = None):
        ckpt = {
            "policy_net":       self.policy_net.state_dict(),
            "q1_net":           self.q1_net.state_dict(),
            "q2_net":           self.q2_net.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "q1_optimizer":     self.q1_optimizer.state_dict(),
            "q2_optimizer":     self.q2_optimizer.state_dict(),
            "train_step":       self.train_step,
        }
        # Target nets only exist on the 1-step path.
        if not self.use_lambda_returns:
            ckpt["q1_target"] = self.q1_target.state_dict()
            ckpt["q2_target"] = self.q2_target.state_dict()
        if self.auto_alpha:
            ckpt["log_alpha"]            = self.log_alpha.item()
            ckpt["alpha_optimizer"]      = self.alpha_optimizer.state_dict()
        torch.save(ckpt, path)
        print(f"Checkpoint saved → {path}")
        if train_dict is not None:
            with open(path[:-2] + "json", "w", encoding="utf-8") as file:
                json.dump(train_dict, file, indent=4)
    
    def save_replay_buffer(self, path : str):
        replay_buffer_ckpt = {
            "states": self.replay_buffer.states,
            "actions": self.replay_buffer.actions,
            "rewards": self.replay_buffer.rewards,
            "next_states": self.replay_buffer.next_states,
            "dones": self.replay_buffer.dones,
            "_ptr": self.replay_buffer._ptr,
            "_size": self.replay_buffer._size,
        }
        torch.save(replay_buffer_ckpt, path)
    
    def load_replay_buffer(self, path: str):
        replay_buffer_ckpt = torch.load(path, weights_only=False)
        self.replay_buffer.states = replay_buffer_ckpt["states"]
        self.replay_buffer.actions = replay_buffer_ckpt["actions"]
        self.replay_buffer.rewards = replay_buffer_ckpt["rewards"]
        self.replay_buffer.next_states = replay_buffer_ckpt["next_states"]
        self.replay_buffer.dones = replay_buffer_ckpt["dones"]
        self.replay_buffer._ptr = replay_buffer_ckpt["_ptr"]
        self.replay_buffer._size = replay_buffer_ckpt["_size"]
        



    # def save_checkpoint(self, path: str, train_dict = None):
    #     torch.save({
    #         "policy_net": self.policy_net.state_dict(),
    #         "q1_net":     self.q1_net.state_dict(),
    #         "q2_net":     self.q2_net.state_dict(),
    #         "q1_target":  self.q1_target.state_dict(),
    #         "q2_target":  self.q2_target.state_dict(),
    #         "train_step": self.train_step,
    #     }, path)
    #     print(f"Checkpoint saved → {path}")
    #     if train_dict is not None:
    #         with open(path[:-2] + "json", "w", encoding="utf-8") as file:
    #             json.dump(train_dict, file, indent=4)
            
    @staticmethod
    def _strip_orig_mod(sd: dict) -> dict:
        return {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()
        }

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(self._strip_orig_mod(ckpt["policy_net"]))
        self.q1_net.load_state_dict(self._strip_orig_mod(ckpt["q1_net"]))
        self.q2_net.load_state_dict(self._strip_orig_mod(ckpt["q2_net"]))
        if not self.use_lambda_returns and "q1_target" in ckpt:
            self.q1_target.load_state_dict(self._strip_orig_mod(ckpt["q1_target"]))
            self.q2_target.load_state_dict(self._strip_orig_mod(ckpt["q2_target"]))
        if self.auto_alpha and "log_alpha" in ckpt:
            self.log_alpha.data.fill_(ckpt["log_alpha"])
            self._clamp_log_alpha()
            self.alpha = self.log_alpha.exp().item()
            if "alpha_optimizer" in ckpt:
                self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
        self.train_step = ckpt["train_step"]
        print(f"Checkpoint loaded ← {path}")
    
    # def load_checkpoint(self, path: str):
    #     ckpt = torch.load(path, map_location=self.device)
    #     self.policy_net.load_state_dict(ckpt["policy_net"])
    #     self.q1_net.load_state_dict(ckpt["q1_net"])
    #     self.q2_net.load_state_dict(ckpt["q2_net"])
    #     self.q1_target.load_state_dict(ckpt["q1_target"])
    #     self.q2_target.load_state_dict(ckpt["q2_target"])
    #     self.train_step = ckpt["train_step"]
    #     print(f"Checkpoint loaded ← {path}")

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

    with open("./config/config.json", "r") as file:
        config = json.load(file)

    trainer = make_transformer_sac_trainer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=config["learning_rate"],
        learning_rate_alpha=config["learning_rate_alpha"],
        batch_size=config["batch_size"],
        replay_buffer_size=config["replay_buffer_size"],
        rule_based_agents=rb_agents,
        log_dir="runs/latest",
        checkpoint_load_path = config["checkpoint_load_path"],
        training_data_load_path = config["training_data_load_path"],
        replay_buffer_load_path = config["replay_buffer_load_path"],
        use_lambda_returns = config["use_lambda_returns"],
        lambda_return = config["lambda_return"],
        cache_size = config["cache_size"],
        block_size = config["block_size"],
        refresh_freq = config["refresh_freq"],
        max_grad_norm = config.get("max_grad_norm"),
        auto_alpha = config.get("auto_alpha", False),
        alpha_min = config.get("alpha_min", 0.05),
        alpha_max = config.get("alpha_max", 1.0),
        target_entropy= config.get("target_entropy")
    )

    print(f"TensorBoard: tensorboard --logdir ./runs/latest")

    trainer.train(
        num_episodes = config["num_episodes"],
        warmup_steps = config["warmup_steps"],
        update_frequency = config["update_frequency"],
        gradient_steps = config["gradient_steps"],
        log_interval = config["log_interval"],
        render_interval = config["render_interval"],
        checkpoint_interval = config["checkpoint_interval"],
        replay_storage_interval = config["replay_storage_interval"],
        checkpoint_path = config["checkpoint_path"],
        rule_based_ratio = config["rule_based_ratio"],
        opponent_noise = config["opponent_noise"],
        prob_4p = config["prob_4p"]
    )
    # trainer.train(
    #     num_episodes=200,
    #     warmup_steps=500,
    #     update_frequency=8,
    #     gradient_steps=2,
    #     log_interval=10,
    #     render_interval=10,
    #     checkpoint_interval= 20,
    #     checkpoint_path="sac_model_dual.pt",
    #     rule_based_ratio=1.0,   # exclusively rule-based opponents
    #     opponent_noise=0.1,
    #     prob_4p = 0.5,
    # )

    trainer.close()
