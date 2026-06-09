import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Dict, Optional
import gymnasium as gym

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

from model.SAC import P_network, Q_network
from env.dummy import MatrixEnv


# =============================================================================
# Profiling helper
# =============================================================================

def _print_profile(step_times: list, n_envs: int = 1) -> None:
    """Print step-timing stats collected over the last log interval."""
    arr     = np.array(step_times)
    avg_ms  = arr.mean()  * 1_000
    std_ms  = arr.std()   * 1_000
    p95_ms  = np.percentile(arr, 95) * 1_000
    steps_per_sec       = 1.0 / arr.mean()
    transitions_per_sec = n_envs * steps_per_sec
    print(
        f"  [profile] "
        f"step: {avg_ms:.2f} ± {std_ms:.2f} ms  "
        f"p95: {p95_ms:.2f} ms  |  "
        f"{steps_per_sec:>7.1f} steps/s  "
        f"{transitions_per_sec:>8.1f} transitions/s"
        + (f"  (×{n_envs} envs)" if n_envs > 1 else "")
    )


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    """
    Circular replay buffer backed by pre-allocated numpy arrays.

    Faster than a deque-based buffer because:
    - Memory allocated once at construction — no per-step Python object creation.
    - np.random.randint sampling is ~100× faster than random.sample on large buffers.
    - torch.from_numpy() returns a zero-copy tensor that shares the numpy memory.
    - add_batch() writes N transitions in a single numpy slice (used by VecEnv).
    """

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

    def add_batch(self, states, actions, rewards, next_states, dones):
        """Write n transitions in one pass — for vectorised environment collection."""
        n    = len(states)
        idxs = np.arange(self._ptr, self._ptr + n) % self._max
        self.states     [idxs] = states
        self.actions    [idxs] = actions
        self.rewards    [idxs] = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        self.next_states[idxs] = next_states
        self.dones      [idxs] = np.asarray(dones,   dtype=np.float32).reshape(-1, 1)
        self._ptr  = int((self._ptr + n) % self._max)
        self._size = min(self._size + n, self._max)

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

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str):
        """Atomically write the filled portion of the buffer (+ ptr/size/max so
        the circular ordering can be restored) to `path`. Writes to a temp file
        first and os.replace()s it, so an interrupted save can't corrupt an
        existing buffer file."""
        n   = self._size
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            np.savez(
                f,
                states      = self.states     [:n],
                actions     = self.actions    [:n],
                rewards     = self.rewards     [:n],
                next_states = self.next_states[:n],
                dones       = self.dones      [:n],
                ptr  = np.int64(self._ptr),
                size = np.int64(self._size),
                max  = np.int64(self._max),
            )
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        """Restore buffer contents from `path`. Returns False if the file is
        missing. If the saved capacity differs from this buffer's, the saved
        transitions are restored as a linear prefill (clamped to capacity)."""
        if not os.path.exists(path):
            return False
        data       = np.load(path)
        saved_max  = int(data["max"])
        saved_size = int(data["size"])
        saved_ptr  = int(data["ptr"])
        n = min(saved_size, self._max)

        self.states     [:n] = data["states"]     [:n]
        self.actions    [:n] = data["actions"]    [:n]
        self.rewards    [:n] = data["rewards"]    [:n]
        self.next_states[:n] = data["next_states"][:n]
        self.dones      [:n] = data["dones"]      [:n]

        if saved_max == self._max:
            # Same capacity → restore exact circular state (ptr marks the seam).
            self._size = saved_size
            self._ptr  = saved_ptr
        else:
            # Different capacity → treat restored rows as a fresh linear prefill.
            self._size = n
            self._ptr  = n % self._max
        return True


# =============================================================================
# SAC Trainer
# =============================================================================

class SACTrainer:
    """
    Modular SAC v2 trainer with optional vectorised-environment support.

    SAC v2 objective (Haarnoja et al. 2018, revised — no V-network):
        Q-target : y = r + γ(1-d)·[min(Q1_tgt,Q2_tgt)(s',ã') − α·log π(ã'|s')]
        Q-loss   : E[(Q(s,a) − y)²]
        π-loss   : E[α·log π(ã|s) − min(Q1,Q2)(s,ã)]

    Vectorised environments
    -----------------------
    Pass a gymnasium VectorEnv (SyncVectorEnv or AsyncVectorEnv) as `env`.
    The trainer detects it via env.num_envs and switches to the vectorised
    training loop automatically.  Each env step then adds n_envs transitions
    to the buffer, and `gradient_steps` updates are run per step.

    To maintain the same update-to-data (UTD) ratio as a single-env run,
    set gradient_steps = n_envs.  A lower value trades sample efficiency for
    wall-clock speed.

    Example — 8 parallel Hopper environments:
        env = gym.make_vec("Hopper-v5", num_envs=8, vectorization_mode="async")
        trainer = SACTrainer(env, policy_net, q1_net, q2_net, ...)
        trainer.train(num_episodes=5000, gradient_steps=8)

    Network interface:
        policy_net.sample(state)     -> (action, log_prob [B, 1])
        q_net.forward(state, action) -> [B]
    """

    def __init__(
        self,
        env,
        policy_net: nn.Module,
        q1_net: nn.Module,
        q2_net: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        alpha: float = 0.2,
        auto_alpha: bool = False,
        target_entropy: Optional[float] = None,
        replay_buffer_size: int = 100_000,
        batch_size: int = 256,
        max_grad_norm: Optional[float] = 1.0,
        use_lambda_returns: bool = False,
        lambda_return: float = 0.9,
        cache_size: int = 8000,
        block_size: int = 50,
        refresh_freq: int = 1000,
        state_preprocessor: Optional[Callable] = None,
        action_preprocessor: Optional[Callable] = None,
        action_postprocessor: Optional[Callable] = None,
        log_dir: Optional[str] = None,
    ):
        self.env           = env
        self.device        = device
        # Enable TF32 matmul/conv on Ampere+ GPUs: the transformer is matmul-bound
        # and TF32 runs those ~1.5-2x faster with precision loss that is immaterial
        # for RL. Harmless no-op on CPU / older GPUs.
        self._dev_type     = torch.device(device).type
        if self._dev_type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # bf16 autocast for all network forwards (~2x on the transformer here).
        # bf16 keeps fp32's exponent range (no overflow / no GradScaler needed) and
        # autocast auto-runs log/exp/softmax/loss in fp32, so the log-prob math
        # stays precise. Master weights remain fp32 (backward runs outside).
        self._amp_enabled  = self._dev_type == "cuda"
        self.gamma         = gamma
        self.tau           = tau
        self.batch_size    = batch_size
        self.max_grad_norm = max_grad_norm

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

        # ── VecEnv detection ──────────────────────────────────────────────────
        self.is_vec_env = hasattr(env, "num_envs")
        self.n_envs     = env.num_envs if self.is_vec_env else 1
        if self.is_vec_env:
            obs_shape  = env.single_observation_space.shape
            act_shape  = env.single_action_space.shape
            act_n_dims = env.single_action_space.shape[0]
        else:
            obs_shape  = env.observation_space.shape
            act_shape  = env.action_space.shape
            act_n_dims = env.action_space.shape[0]

        # ── entropy temperature ────────────────────────────────────────────────
        self.auto_alpha = auto_alpha
        if auto_alpha:
            if target_entropy is None:
                target_entropy = -float(act_n_dims)
            self.target_entropy = target_entropy
            self.log_alpha      = torch.zeros(1, requires_grad=True, device=device)
            self.alpha          = self.log_alpha.exp().item()
            _alpha_kw = {"fused": True} if torch.device(device).type == "cuda" else {}
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate, **_alpha_kw)
        else:
            self.alpha = alpha

        self.state_preprocessor   = state_preprocessor   or (lambda x: x)
        self.action_preprocessor  = action_preprocessor  or (lambda x: x)
        self.action_postprocessor = action_postprocessor or (lambda x: x)

        # ── TensorBoard writer ────────────────────────────────────────────────
        if log_dir is not None and not _TB_AVAILABLE:
            print("Warning: tensorboard not installed — logging disabled. "
                  "Run: pip install tensorboard")
        self.writer        = SummaryWriter(log_dir=log_dir) if (log_dir and _TB_AVAILABLE) else None
        self._update_count = 0
        self._nan_skips    = 0

        # ── networks ──────────────────────────────────────────────────────────
        self.policy_net = policy_net.to(device)
        self.q1_net     = q1_net.to(device)
        self.q2_net     = q2_net.to(device)
        # Target networks exist only for the 1-step path. The cache-based TD(λ)
        # path eliminates them — its stable targets come from the refreshed cache.
        if not self.use_lambda_returns:
            self.q1_target = copy.deepcopy(q1_net).to(device)
            self.q2_target = copy.deepcopy(q2_net).to(device)
            self._hard_update(self.q1_target, self.q1_net)
            self._hard_update(self.q2_target, self.q2_net)

        # ── optimisers ────────────────────────────────────────────────────────
        # On CUDA use the *fused* Adam kernel: it updates all parameters in one
        # on-device launch and, crucially, keeps the step counter on the GPU. The
        # default single-tensor path calls `_get_value(step).item()` once PER
        # parameter, forcing ~one CPU↔GPU sync per param per update (~390 syncs /
        # update here) — a large, pure-overhead cost. fused=True removes it.
        adam_kw = {}
        if torch.device(self.device).type == "cuda":
            adam_kw["fused"] = True
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, **adam_kw)
        self.q1_optimizer     = optim.Adam(self.q1_net.parameters(),     lr=learning_rate, **adam_kw)
        self.q2_optimizer     = optim.Adam(self.q2_net.parameters(),     lr=learning_rate, **adam_kw)

        # ── replay buffer (pre-allocated, shape-aware) ────────────────────────
        self.replay_buffer = ReplayBuffer(replay_buffer_size, obs_shape, act_shape)
        self.train_step    = 0   # counts env interactions only (not gradient steps)

    # =========================================================================
    # Utilities
    # =========================================================================

    def _hard_update(self, target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    @staticmethod
    def _grad_norm(net: nn.Module) -> float:
        # Compute the global grad norm with a SINGLE host sync (one .item()) by
        # stacking per-param norms on-device, instead of .item() per parameter
        # (which was ~65 CPU↔GPU syncs per call, ×3 nets, on every logged update).
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        return torch.norm(torch.stack([g.norm(2) for g in grads]), 2).item()

    def _to(self, t: torch.Tensor) -> torch.Tensor:
        """Move tensor to device with non-blocking transfer (overlaps with GPU compute)."""
        return t.to(self.device, non_blocking=True)

    def _autocast(self):
        """bf16 mixed-precision context for network forwards (no-op off CUDA)."""
        return torch.autocast(device_type=self._dev_type, dtype=torch.bfloat16,
                              enabled=self._amp_enabled)

    # =========================================================================
    # Action selection
    # =========================================================================

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Single-env action — stochastic during training."""
        state_t = self._to(torch.FloatTensor(state).unsqueeze(0))
        state_t = self.state_preprocessor(state_t)
        with torch.no_grad(), self._autocast():
            action, _ = self.policy_net.sample(state_t)
        return self.action_postprocessor(action.float().cpu().numpy()[0])

    def select_action_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Vectorised action — takes (n_envs, *obs_shape) and returns
        (n_envs, *act_shape).  Used by the VecEnv training loop.
        """
        states_t = self._to(torch.FloatTensor(states))
        states_t = self.state_preprocessor(states_t)
        with torch.no_grad(), self._autocast():
            actions, _ = self.policy_net.sample(states_t)
        actions_np = actions.float().cpu().numpy()
        # apply postprocessor per-env (handles transformer padding etc.)
        return np.stack([self.action_postprocessor(a) for a in actions_np])

    # =========================================================================
    # Gradient update — SAC v2
    # =========================================================================

    def update(self) -> Optional[Dict[str, float]]:
        """
        One SAC v2 gradient step.  train_step is NOT incremented here —
        that is the responsibility of the training loop.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        if self.use_lambda_returns:
            return self._lambda_update()

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        # non_blocking=True overlaps H2D copy with GPU compute
        states      = self.state_preprocessor (self._to(states))
        actions     = self.action_preprocessor(self._to(actions))
        rewards     = self._to(rewards)
        next_states = self.state_preprocessor (self._to(next_states))
        dones       = self._to(dones)

        # ── Q-targets ─────────────────────────────────────────────────────────
        with torch.no_grad(), self._autocast():
            a_next, lp_next = self.policy_net.sample(next_states)
            q1_next = self.q1_target(next_states, a_next).unsqueeze(-1)
            q2_next = self.q2_target(next_states, a_next).unsqueeze(-1)
            q_target = (rewards + (1.0 - dones) * self.gamma * (
                torch.min(q1_next, q2_next) - self.alpha * lp_next
            )).float()

        # NaN/Inf guard: if the target is non-finite, skip this batch entirely so
        # one bad sample cannot poison the weights (clip_grad_norm_ would just
        # propagate the NaN through the total norm).
        if not torch.isfinite(q_target).all():
            self._nan_skips += 1
            if self.writer is not None:
                self.writer.add_scalar("Misc/nan_skips", self._nan_skips, self._update_count)
            return None

        # ── Q1 update ─────────────────────────────────────────────────────────
        with self._autocast():
            q1_pred = self.q1_net(states, actions).unsqueeze(-1)
            q1_loss = nn.MSELoss()(q1_pred, q_target)
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q1_net.parameters(), self.max_grad_norm)
        self.q1_optimizer.step()

        # ── Q2 update ─────────────────────────────────────────────────────────
        with self._autocast():
            q2_pred = self.q2_net(states, actions).unsqueeze(-1)
            q2_loss = nn.MSELoss()(q2_pred, q_target)
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q2_net.parameters(), self.max_grad_norm)
        self.q2_optimizer.step()

        # ── Policy update ──────────────────────────────────────────────────────
        with self._autocast():
            a_tilde, lp = self.policy_net.sample(states)
            q1_pi = self.q1_net(states, a_tilde).unsqueeze(-1)
            q2_pi = self.q2_net(states, a_tilde).unsqueeze(-1)
            policy_loss = (self.alpha * lp - torch.min(q1_pi, q2_pi)).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        # ── Polyak-update target Q-networks ───────────────────────────────────
        self._soft_update(self.q1_target, self.q1_net)
        self._soft_update(self.q2_target, self.q2_net)

        # ── Auto-alpha ────────────────────────────────────────────────────────
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha.exp() * (lp.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # ── TensorBoard ───────────────────────────────────────────────────────
        if self.writer is not None:
            s = self._update_count
            with torch.no_grad():
                _, sigma = self.policy_net.forward(states)
            self.writer.add_scalar("Loss/q1",              q1_loss.item(),     s)
            self.writer.add_scalar("Loss/q2",              q2_loss.item(),     s)
            self.writer.add_scalar("Loss/policy",          policy_loss.item(), s)
            self.writer.add_scalar("Policy/mean_log_prob", lp.mean().item(),   s)
            self.writer.add_scalar("Policy/sigma_mean",    sigma.mean().item(), s)
            self.writer.add_scalar("Q/target_mean",        q_target.mean().item(), s)
            self.writer.add_scalar("Alpha/value",          self.alpha,         s)
            self.writer.add_scalar("GradNorm/policy", self._grad_norm(self.policy_net), s)
            self.writer.add_scalar("GradNorm/q1",     self._grad_norm(self.q1_net),    s)
            self.writer.add_scalar("GradNorm/q2",     self._grad_norm(self.q2_net),    s)
            if self.auto_alpha:
                self.writer.add_scalar("Loss/alpha", alpha_loss.item(), s)

        self._update_count += 1
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "pi_loss": policy_loss.item(),
        }

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
        with torch.no_grad(), self._autocast():
            for i in range(0, len(next_states_np), chunk):
                ns = torch.from_numpy(next_states_np[i:i + chunk])
                ns = self.state_preprocessor(self._to(ns))
                a, lp = self.policy_net.sample(ns)                 # a:[c,...], lp:[c,1]
                q1 = self.q1_net(ns, a).unsqueeze(-1)              # [c,1]
                q2 = self.q2_net(ns, a).unsqueeze(-1)              # [c,1]
                v  = torch.min(q1, q2) - self.alpha * lp           # [c,1]
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
        rewards_np     = buf.rewards    [flat_idx].reshape(n_blocks, B)
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

        # ── Q updates toward the precomputed λ-return (shared by Q1 & Q2) ─────
        with self._autocast():
            q1_pred = self.q1_net(states, actions).unsqueeze(-1)
            q1_loss = nn.MSELoss()(q1_pred, targets)
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q1_net.parameters(), self.max_grad_norm)
        self.q1_optimizer.step()

        with self._autocast():
            q2_pred = self.q2_net(states, actions).unsqueeze(-1)
            q2_loss = nn.MSELoss()(q2_pred, targets)
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q2_net.parameters(), self.max_grad_norm)
        self.q2_optimizer.step()

        # ── Policy update — identical SAC actor objective ─────────────────────
        with self._autocast():
            a_tilde, lp = self.policy_net.sample(states)
            q1_pi = self.q1_net(states, a_tilde).unsqueeze(-1)
            q2_pi = self.q2_net(states, a_tilde).unsqueeze(-1)
            policy_loss = (self.alpha * lp - torch.min(q1_pi, q2_pi)).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        # NB: no target-network Polyak update — the cache is the stable target.

        # ── Auto-alpha (unchanged) ────────────────────────────────────────────
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha.exp() * (lp.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # ── TensorBoard ───────────────────────────────────────────────────────
        if self.writer is not None:
            s = self._update_count
            with torch.no_grad():
                _, sigma = self.policy_net.forward(states)
            self.writer.add_scalar("Loss/q1",              q1_loss.item(),     s)
            self.writer.add_scalar("Loss/q2",              q2_loss.item(),     s)
            self.writer.add_scalar("Loss/policy",          policy_loss.item(), s)
            self.writer.add_scalar("Policy/mean_log_prob", lp.mean().item(),   s)
            self.writer.add_scalar("Policy/sigma_mean",    sigma.mean().item(), s)
            self.writer.add_scalar("Q/target_mean",        targets.mean().item(), s)
            self.writer.add_scalar("Alpha/value",          self.alpha,         s)
            self.writer.add_scalar("GradNorm/policy", self._grad_norm(self.policy_net), s)
            self.writer.add_scalar("GradNorm/q1",     self._grad_norm(self.q1_net),    s)
            self.writer.add_scalar("GradNorm/q2",     self._grad_norm(self.q2_net),    s)
            if self.auto_alpha:
                self.writer.add_scalar("Loss/alpha", alpha_loss.item(), s)

        self._update_count += 1
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "pi_loss": policy_loss.item(),
        }

    # =========================================================================
    # Training loop dispatcher
    # =========================================================================

    def train(
        self,
        num_episodes: int = 100,
        max_steps_per_episode: int = 1000,
        warmup_steps: int = 5000,
        update_frequency: int = 1,
        gradient_steps: int = 1,
        log_interval: int = 10,
        eval_env=None,
        eval_interval: int = 50,
        eval_episodes: int = 5,
        profile: bool = False,
    ):
        """
        Main training entry point.

        Args:
            num_episodes:           Total episodes to collect (all envs combined).
            max_steps_per_episode:  Per-episode step cap (single-env only; VecEnv
                                    episodes run until terminated/truncated).
            warmup_steps:           Env steps of random exploration before learning.
            update_frequency:       Run an update every N env steps (single-env).
                                    With VecEnv, updates run every step regardless
                                    (controlled by gradient_steps instead).
            gradient_steps:         Gradient updates per env-step cycle.
                                    Set to n_envs to keep the same UTD ratio as a
                                    single-env run; set lower for wall-clock speed.
            eval_env:               Optional single env for greedy evaluation.
            eval_interval:          Evaluate every N completed episodes.
            eval_episodes:          Episodes per evaluation.
            profile:                When True, prints avg time per step and
                                    throughput (transitions/sec) at every log
                                    interval.  Useful for comparing single vs
                                    vectorised env speed.
        """
        if self.is_vec_env:
            return self._train_vec(
                num_episodes, warmup_steps, gradient_steps,
                log_interval, eval_env, eval_interval, eval_episodes, profile,
            )
        return self._train_single(
            num_episodes, max_steps_per_episode, warmup_steps,
            update_frequency, gradient_steps, log_interval,
            eval_env, eval_interval, eval_episodes, profile,
        )

    # =========================================================================
    # Single-env training loop
    # =========================================================================

    def _train_single(
        self, num_episodes, max_steps_per_episode, warmup_steps,
        update_frequency, gradient_steps, log_interval,
        eval_env, eval_interval, eval_episodes, profile,
    ):
        episode_rewards = []
        _step_times: list[float] = []

        for episode in range(num_episodes):
            state, _       = self.env.reset()
            episode_reward = 0.0
            episode_length = 0

            for _ in range(max_steps_per_episode):
                _t0 = time.perf_counter() if profile else 0.0

                if self.train_step < warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.select_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.replay_buffer.add(state, action, reward, next_state, float(done))
                self.train_step += 1

                if (
                    self.train_step >= warmup_steps
                    and self.train_step % update_frequency == 0
                    and len(self.replay_buffer) >= self.batch_size
                ):
                    for _ in range(gradient_steps):
                        self.update()

                if profile:
                    _step_times.append(time.perf_counter() - _t0)

                episode_reward += reward
                episode_length += 1
                state = next_state
                if done:
                    break

            episode_rewards.append(episode_reward)
            self._log_episode(episode, episode_reward, episode_length)

            if (episode + 1) % log_interval == 0:
                avg = float(np.mean(episode_rewards[-log_interval:]))
                if self.writer:
                    self.writer.add_scalar("Reward/moving_avg", avg, episode)
                print(
                    f"Ep {episode + 1:>4}/{num_episodes} | "
                    f"Avg({log_interval}): {avg:>10.3f} | "
                    f"Buffer: {len(self.replay_buffer):>7} | "
                    f"Steps: {self.train_step}"
                )
                if profile and _step_times:
                    _print_profile(_step_times, n_envs=1)
                    _step_times.clear()

            if eval_env is not None and (episode + 1) % eval_interval == 0:
                self._run_eval(eval_env, eval_episodes, episode)

        return episode_rewards

    # =========================================================================
    # Vectorised-env training loop
    # =========================================================================

    def _train_vec(
        self, num_episodes, warmup_steps, gradient_steps,
        log_interval, eval_env, eval_interval, eval_episodes, profile,
    ):
        """
        Step-based loop for VecEnv.  Collects n_envs transitions per step.

        Terminal-observation fix: gymnasium VecEnvs auto-reset on done, so
        next_obs at a done step is the *reset* observation of the new episode,
        not the terminal observation.  The true terminal obs is in
        infos['final_observation'][i].  We substitute it before storing so the
        Q-target sees the correct next-state.
        """
        states, _    = self.env.reset()
        ep_rewards   = np.zeros(self.n_envs, dtype=np.float64)
        ep_lengths   = np.zeros(self.n_envs, dtype=np.int32)
        completed    = []          # list of (reward,) for each finished episode
        last_log_ep  = 0
        _step_times: list[float] = []

        while len(completed) < num_episodes:
            _t0 = time.perf_counter() if profile else 0.0

            # ── collect ───────────────────────────────────────────────────────
            if self.train_step < warmup_steps:
                actions = self.env.action_space.sample()
            else:
                actions = self.select_action_batch(states)

            next_states, rewards, terminated, truncated, infos = self.env.step(actions)
            dones = terminated | truncated

            # Fix terminal observations before storing
            real_next = next_states.copy()
            if "final_observation" in infos:
                for i, (d, fo) in enumerate(zip(dones, infos["final_observation"])):
                    if d and fo is not None:
                        real_next[i] = fo

            self.replay_buffer.add_batch(states, actions, rewards, real_next, dones)
            self.train_step += self.n_envs

            ep_rewards += rewards
            ep_lengths += 1

            # ── track finished episodes ───────────────────────────────────────
            for i in range(self.n_envs):
                if dones[i]:
                    ep_idx = len(completed)
                    completed.append(ep_rewards[i])
                    self._log_episode(ep_idx, ep_rewards[i], int(ep_lengths[i]))
                    ep_rewards[i] = 0.0
                    ep_lengths[i] = 0

            states = next_states

            # ── gradient updates ──────────────────────────────────────────────
            if (
                self.train_step >= warmup_steps
                and len(self.replay_buffer) >= self.batch_size
            ):
                for _ in range(gradient_steps):
                    self.update()

            if profile:
                _step_times.append(time.perf_counter() - _t0)

            # ── console log ───────────────────────────────────────────────────
            n_done = len(completed)
            if n_done >= log_interval and n_done // log_interval > last_log_ep // log_interval:
                avg = float(np.mean(completed[-log_interval:]))
                last_log_ep = n_done
                if self.writer:
                    self.writer.add_scalar("Reward/moving_avg", avg, n_done)
                    self.writer.add_scalar("Misc/env_steps",    self.train_step, n_done)
                print(
                    f"Ep {n_done:>4}/{num_episodes} | "
                    f"Avg({log_interval}): {avg:>10.3f} | "
                    f"Buffer: {len(self.replay_buffer):>7} | "
                    f"Steps: {self.train_step} | "
                    f"n_envs: {self.n_envs}"
                )
                if profile and _step_times:
                    _print_profile(_step_times, n_envs=self.n_envs)
                    _step_times.clear()

            # ── eval ──────────────────────────────────────────────────────────
            n_done = len(completed)
            if (
                eval_env is not None
                and n_done > 0
                and n_done % eval_interval == 0
                and n_done != getattr(self, "_last_eval_ep", -1)
            ):
                self._last_eval_ep = n_done
                self._run_eval(eval_env, eval_episodes, n_done)

        return completed

    # =========================================================================
    # Evaluation & logging helpers
    # =========================================================================

    def _evaluate(self, eval_env, n_episodes: int = 5) -> float:
        """Greedy rollouts on a single env. Returns mean episode reward."""
        use_det = hasattr(self.policy_net, "deterministic_action")
        rewards = []
        for _ in range(n_episodes):
            state, _ = eval_env.reset()
            total    = 0.0
            while True:
                state_t = self._to(torch.FloatTensor(state).unsqueeze(0))
                state_t = self.state_preprocessor(state_t)
                with torch.no_grad():
                    action_t = (
                        self.policy_net.deterministic_action(state_t) if use_det
                        else self.policy_net.sample(state_t)[0]
                    )
                state, reward, terminated, truncated, _ = eval_env.step(
                    self.action_postprocessor(action_t.cpu().numpy()[0])
                )
                total += reward
                if terminated or truncated:
                    break
            rewards.append(total)
        return float(np.mean(rewards))

    def _log_episode(self, episode_idx: int, reward: float, length: int):
        if self.writer is not None:
            self.writer.add_scalar("Reward/episode",      reward,            episode_idx)
            self.writer.add_scalar("Misc/episode_length", length,            episode_idx)
            self.writer.add_scalar("Misc/buffer_fill",    len(self.replay_buffer), episode_idx)
            self.writer.add_scalar("Misc/env_steps",      self.train_step,   episode_idx)

    def _run_eval(self, eval_env, eval_episodes: int, episode_idx: int):
        eval_reward = self._evaluate(eval_env, n_episodes=eval_episodes)
        if self.writer:
            self.writer.add_scalar("Reward/eval", eval_reward, episode_idx)
        print(f"  → Eval ({eval_episodes} eps): {eval_reward:.3f}")

    def close(self):
        if self.writer is not None:
            self.writer.close()

    # =========================================================================
    # Checkpoint I/O
    # =========================================================================

    def save_checkpoint(self, path: str):
        ckpt = {
            "policy_net": self.policy_net.state_dict(),
            "q1_net":     self.q1_net.state_dict(),
            "q2_net":     self.q2_net.state_dict(),
            "train_step": self.train_step,
        }
        # Target nets only exist on the 1-step path.
        if not self.use_lambda_returns:
            ckpt["q1_target"] = self.q1_target.state_dict()
            ckpt["q2_target"] = self.q2_target.state_dict()
        torch.save(ckpt, path)
        print(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.q1_net.load_state_dict(ckpt["q1_net"])
        self.q2_net.load_state_dict(ckpt["q2_net"])
        if not self.use_lambda_returns and "q1_target" in ckpt:
            self.q1_target.load_state_dict(ckpt["q1_target"])
            self.q2_target.load_state_dict(ckpt["q2_target"])
        self.train_step = ckpt["train_step"]
        print(f"Checkpoint loaded ← {path}")

    def save_replay_buffer(self, path: str):
        self.replay_buffer.save(path)
        size_mb = os.path.getsize(path) / 1e6
        print(f"Replay buffer saved → {path} "
              f"({len(self.replay_buffer)} transitions, {size_mb:.0f} MB)")

    def load_replay_buffer(self, path: str) -> bool:
        if self.replay_buffer.load(path):
            print(f"Replay buffer loaded ← {path} "
                  f"({len(self.replay_buffer)} transitions)")
            return True
        print(f"No replay buffer found at {path}; starting with an empty buffer")
        return False


# =============================================================================
# Factory: transformer variant
# =============================================================================

def make_transformer_sac_trainer(env, **trainer_kwargs) -> SACTrainer:
    """
    SACTrainer wired with transformer Q/Policy networks for the galaxy env.

    State  env → network : [B, 144, 14] → [B, 140, 14]
    Action env → network : [B,  44,  4] → [B,  40,  4]
    """
    max_planets = 40
    max_fleets  = 100
    net_seq_len = max_planets + max_fleets
    # Derive the per-token action width from the env (OrbitWarsEnv.ACTION_DIM=4)
    # rather than hardcoding, so the policy head always matches what the
    # decoder consumes.
    action_dim  = env.action_space.shape[-1]
    env_action_seq = env.action_space.shape[0]

    net_kw = dict(state_dim=14, action_dim=action_dim,
                  max_planets=max_planets, max_fleets=max_fleets)

    policy_net = P_network(**net_kw)
    q1_net     = Q_network(**net_kw)
    q2_net     = Q_network(**net_kw)

    def state_pre(s):
        B, S, F = s.shape
        if S > net_seq_len:
            return s[:, :net_seq_len, :]
        if S < net_seq_len:
            return torch.cat([s, torch.zeros(B, net_seq_len - S, F, device=s.device)], 1)
        return s

    def action_pre(a):
        B, A, D = a.shape
        if A > max_planets:
            return a[:, :max_planets, :]
        if A < max_planets:
            return torch.cat([a, torch.zeros(B, max_planets - A, D, device=a.device)], 1)
        return a

    def action_post(a: np.ndarray) -> np.ndarray:
        if a.shape[0] < env_action_seq:
            pad = np.zeros((env_action_seq - a.shape[0], action_dim), dtype=np.float32)
            a   = np.vstack([a, pad])
        return a

    return SACTrainer(
        env=env,
        policy_net=policy_net,
        q1_net=q1_net,
        q2_net=q2_net,
        state_preprocessor=state_pre,
        action_preprocessor=action_pre,
        action_postprocessor=action_post,
        **trainer_kwargs,
    )


# =============================================================================
if __name__ == "__main__":
    env = MatrixEnv(state_dim=14, action_dim=4, max_state=144, max_action=44)

    trainer = make_transformer_sac_trainer(
        env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=1e-4,
        batch_size=32,
        replay_buffer_size=50_000,
        log_dir="runs/transformer_sac",
    )

    print("TensorBoard: tensorboard --logdir runs/transformer_sac")

    trainer.train(
        num_episodes=100,
        max_steps_per_episode=50,
        warmup_steps=500,
        gradient_steps=1,
        profile=True,
        log_interval=5,
    )

    trainer.save_checkpoint("sac_model.pt")
    trainer.close()
