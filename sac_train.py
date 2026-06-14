import copy
import math
import os
import threading
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Dict, Optional
import gymnasium as gym

try:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    _JAX_BUFFER_AVAILABLE = True
except ImportError:
    _JAX_BUFFER_AVAILABLE = False

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
        # Guards add/sample so a writer (env thread) and reader (learner thread)
        # in the threaded training loop can't tear a transition.  Uncontended
        # acquire is ~50 ns, so it is effectively free for the single-thread path.
        self._lock = threading.Lock()

        self.states      = np.zeros((self._max, *state_shape),  dtype=np.float32)
        self.actions     = np.zeros((self._max, *action_shape), dtype=np.float32)
        self.rewards     = np.zeros((self._max, 1),             dtype=np.float32)
        self.next_states = np.zeros((self._max, *state_shape),  dtype=np.float32)
        self.dones       = np.zeros((self._max, 1),             dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        with self._lock:
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
        with self._lock:
            idxs = np.arange(self._ptr, self._ptr + n) % self._max
            self.states     [idxs] = states
            self.actions    [idxs] = actions
            self.rewards    [idxs] = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
            self.next_states[idxs] = next_states
            self.dones      [idxs] = np.asarray(dones,   dtype=np.float32).reshape(-1, 1)
            self._ptr  = int((self._ptr + n) % self._max)
            self._size = min(self._size + n, self._max)

    def sample(self, batch_size: int):
        with self._lock:
            idxs = np.random.randint(0, self._size, size=batch_size)
            # numpy advanced indexing already returns fresh copies, so the
            # returned tensors are safe once the lock is released.
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
# GPU-resident Replay Buffer (JAX backend)
# =============================================================================

class JaxReplayBuffer:
    """Replay buffer backed by JAX arrays living on the GPU.

    Designed for the JAX parallel environment backend, where the CPU numpy
    random-scatter bottleneck in ReplayBuffer.sample() becomes the dominant
    cost (85 %+ of wall time at high num_envs).

    Write path  (add_batch): numpy → JAX GPU arrays via a JIT-compiled scatter
      with buffer donation so XLA can update in-place without copying the full
      buffer.  The H2D transfer happens once per vector step for num_envs
      sequential rows — cache-friendly and unavoidable.

    Read path   (sample): JIT-compiled GPU gather + JAX PRNG → DLPack →
      PyTorch CUDA tensor.  No CPU round-trip; no PCIe on the critical path.
      The A100's ~2 TB/s HBM bandwidth makes this ~100× faster than numpy
      random fancy-indexing on CPU.

    SACTrainer.update() works unchanged: _to() is a no-op for already-CUDA
    tensors, and the preprocessors operate on CUDA tensors via torch.cat etc.

    Requirements: JAX with CUDA backend; enough VRAM for the buffer
      (≈ capacity × 2 × obs_bytes + action/reward bytes).  On an A100 40 GB
      a 100 k buffer of shape (140, 13) costs ≈ 1.4 GB.

    Set XLA_PYTHON_CLIENT_PREALLOCATE=false before importing JAX so it does
    not claim all VRAM at startup, leaving headroom for PyTorch.
    """

    def __init__(self, capacity: int, obs_shape: tuple, action_shape: tuple):
        if not _JAX_BUFFER_AVAILABLE:
            raise RuntimeError(
                "JaxReplayBuffer requires JAX with CUDA support. "
                "Install with: pip install 'jax[cuda12]'"
            )
        self._cap  = int(capacity)
        self._ptr  = 0
        self._size = 0
        # Serialises add_batch (donates/reassigns the buffer arrays) against
        # sample (reads them) so the threaded loop's env and learner threads
        # can't race on the donated GPU arrays.
        self._lock = threading.Lock()

        # Pre-allocate all arrays on the JAX default device (GPU).
        self._obs  = jnp.zeros((self._cap, *obs_shape),    dtype=jnp.float32)
        self._acts = jnp.zeros((self._cap, *action_shape), dtype=jnp.float32)
        self._rews = jnp.zeros((self._cap, 1),             dtype=jnp.float32)
        self._nxts = jnp.zeros((self._cap, *obs_shape),    dtype=jnp.float32)
        self._dons = jnp.zeros((self._cap, 1),             dtype=jnp.float32)

        self._rng = jr.PRNGKey(0)

        # donate_argnums=(0..4): XLA may reuse the buffer arrays in-place for
        # the scatter output, avoiding a full-buffer copy on every write step.
        # After the call the original references are invalid; we always
        # reassign self._obs etc. from the returned tuple.
        self._jit_write = jax.jit(
            JaxReplayBuffer._write_fn, donate_argnums=(0, 1, 2, 3, 4)
        )
        # static_argnums=(7,): batch_size is always the same value, so JAX
        # compiles once and treats it as a compile-time shape constant, which
        # lets (n,) in jr.randint be a static shape rather than a traced value.
        self._jit_sample = jax.jit(
            JaxReplayBuffer._sample_fn, static_argnums=(7,)
        )

    # ── JIT-compiled kernels (static methods so jax.jit can trace them) ──────

    @staticmethod
    def _write_fn(obs, acts, rews, nxts, dons,
                  new_obs, new_acts, new_rews, new_nxts, new_dons, idxs):
        return (
            obs .at[idxs].set(new_obs),
            acts.at[idxs].set(new_acts),
            rews.at[idxs].set(new_rews),
            nxts.at[idxs].set(new_nxts),
            dons.at[idxs].set(new_dons),
        )

    @staticmethod
    def _sample_fn(key, obs, acts, rews, nxts, dons, size, n):
        # size is dynamic (grows during warmup); n is static (always batch_size).
        idxs = jr.randint(key, (n,), minval=0, maxval=size)
        return obs[idxs], acts[idxs], rews[idxs], nxts[idxs], dons[idxs]

    # ── Public interface (matches ReplayBuffer) ───────────────────────────────

    def add_batch(self, obs, acts, rews, nxts, dons):
        n    = len(obs)
        with self._lock:
            idxs = (jnp.arange(n, dtype=jnp.int32) + self._ptr) % self._cap
            (self._obs, self._acts, self._rews, self._nxts, self._dons) = (
                self._jit_write(
                    self._obs, self._acts, self._rews, self._nxts, self._dons,
                    jnp.asarray(obs,  dtype=jnp.float32),
                    jnp.asarray(acts, dtype=jnp.float32),
                    jnp.asarray(rews, dtype=jnp.float32).reshape(-1, 1),
                    jnp.asarray(nxts, dtype=jnp.float32),
                    jnp.asarray(dons, dtype=jnp.float32).reshape(-1, 1),
                    idxs,
                )
            )
            self._ptr  = int((self._ptr + n) % self._cap)
            self._size = min(self._size + n, self._cap)

    def add(self, obs, act, rew, nxt, don):
        """Single-transition add (delegates to add_batch for a batch of 1)."""
        self.add_batch(
            obs[np.newaxis], act[np.newaxis],
            np.array([rew],        dtype=np.float32),
            nxt[np.newaxis],
            np.array([float(don)], dtype=np.float32),
        )

    def sample(self, batch_size: int):
        """GPU gather → DLPack → PyTorch CUDA tensors (zero PCIe transfer)."""
        with self._lock:
            self._rng, subkey = jr.split(self._rng)
            arrays = self._jit_sample(
                subkey,
                self._obs, self._acts, self._rews, self._nxts, self._dons,
                jnp.int32(self._size),  # dynamic: changes as buffer fills
                batch_size,             # static: compile-time shape constant
            )
        # DLPack: zero-copy JAX GPU array → PyTorch CUDA tensor.
        # JAX handles stream synchronisation inside to_dlpack so the gather
        # is guaranteed complete before PyTorch reads the tensor.
        return tuple(torch.from_dlpack(a) for a in arrays)

    def __len__(self) -> int:
        return self._size

    # ── Persistence (GPU → numpy for I/O, only at checkpoint time) ───────────

    def save(self, path: str):
        n   = self._size
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            np.savez(
                f,
                states      = np.asarray(self._obs [:n]),
                actions     = np.asarray(self._acts[:n]),
                rewards     = np.asarray(self._rews[:n]),
                next_states = np.asarray(self._nxts[:n]),
                dones       = np.asarray(self._dons[:n]),
                ptr  = np.int64(self._ptr),
                size = np.int64(self._size),
                max  = np.int64(self._cap),
            )
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        data       = np.load(path)
        saved_max  = int(data["max"])
        saved_size = int(data["size"])
        saved_ptr  = int(data["ptr"])
        n = min(saved_size, self._cap)
        idxs = jnp.arange(n, dtype=jnp.int32)
        (self._obs, self._acts, self._rews, self._nxts, self._dons) = (
            self._jit_write(
                self._obs, self._acts, self._rews, self._nxts, self._dons,
                jnp.array(data["states"]     [:n], dtype=jnp.float32),
                jnp.array(data["actions"]    [:n], dtype=jnp.float32),
                jnp.array(data["rewards"]    [:n], dtype=jnp.float32),
                jnp.array(data["next_states"][:n], dtype=jnp.float32),
                jnp.array(data["dones"]      [:n], dtype=jnp.float32),
                idxs,
            )
        )
        if saved_max == self._cap:
            self._size = saved_size
            self._ptr  = saved_ptr
        else:
            self._size = n
            self._ptr  = n % self._cap
        return True

    def block_until_ready(self):
        """Block until all pending JAX GPU operations (scatter/gather) complete.

        Used by the benchmark to get accurate per-phase wall-clock timings.
        Not needed in normal training — JAX dispatches are async by design.
        """
        jax.block_until_ready(self._obs)


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
        env_stride: int = 1,
        state_preprocessor: Optional[Callable] = None,
        action_preprocessor: Optional[Callable] = None,
        action_postprocessor: Optional[Callable] = None,
        log_dir: Optional[str] = None,
        use_jax_buffer: bool = False,
        tb_log_every: int = 1,
        compile_mode: str = "default",
    ):
        self.env           = env
        self._compile_mode = (compile_mode or "default")
        # "cudagraph": capture the WHOLE update() (fwd+bwd+optim+polyak for all
        # three nets) into one replayable CUDA graph — see _build_cudagraph.  This
        # is the cure for the launch-bound update (low GPU util, small model) that
        # torch.compile's auto-cudagraphs (reduce-overhead) can't deliver here
        # because SAC's combined policy backward aliases their shared memory pool.
        self._use_cudagraph = (self._compile_mode == "cudagraph")
        self._cudagraph     = None     # lazily captured on the first update()
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
        # Number of envs whose transitions are interleaved per add_batch row. The
        # replay buffer is filled in row-major (env-interleaved) order by the
        # vectorised loops, so env e's trajectory is at a STRIDE of env_stride.
        # _build_lambda_cache uses this to promote per-ENV contiguous blocks (a
        # contiguous flat block would chain unrelated envs and corrupt λ-returns).
        self._lambda_env_stride = max(1, int(env_stride))
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
        # When no postprocessor is supplied (e.g. the JAX adapter handles its own
        # action decoding) skip the per-env Python loop in select_action_batch.
        self._has_action_post     = action_postprocessor is not None

        # ── TensorBoard writer ────────────────────────────────────────────────
        if log_dir is not None and not _TB_AVAILABLE:
            print("Warning: tensorboard not installed — logging disabled. "
                  "Run: pip install tensorboard")
        self.writer        = SummaryWriter(log_dir=log_dir) if (log_dir and _TB_AVAILABLE) else None
        self._update_count = 0
        # Log gradient-update scalars (which each force a GPU sync) once every
        # this many updates, keeping the hot path sync-free between logs.
        self._tb_log_every = max(1, int(tb_log_every))

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

        # With d_model=128 each transformer kernel finishes in <1 µs but Python
        # dispatch takes ~5 µs — the GPU idles between launches.  compile() traces
        # the forward graph and submits it as one fused kernel sequence.
        #
        # compile_mode (config: model.compile_mode):
        #   "none"            — disable torch.compile (eager)
        #   "default"         — fuse kernels (reduces, but does not eliminate, dispatch)
        #   "reduce-overhead" — CUDA graphs: capture the whole forward as one
        #                       replayable unit, eliminating per-kernel dispatch.
        #                       This is the right setting when the profile shows the
        #                       GPU "starved between kernels" (low mean util, small
        #                       model).  NOTE: nn.TransformerEncoder graph-breaks under
        #                       compile, which limits the gain — see the README note.
        #   "max-autotune"    — autotune GEMMs (long compile; best steady-state)
        if (self._compile_mode != "none" and self._dev_type == "cuda"
                and hasattr(torch, "compile")):
            # cudagraph mode still compiles with DEFAULT (inductor fusion → fewer,
            # bigger kernels); the manual CUDA-graph capture then eliminates the
            # per-kernel launch overhead on top.  Fusion alone (default) leaves the
            # launches; capture of unfused eager kernels is slower than fused — we
            # need both.
            if self._compile_mode in ("default", "cudagraph"):
                _ckw = {}
            else:
                _ckw = {"mode": self._compile_mode}
            self.policy_net = torch.compile(self.policy_net, **_ckw)
            self.q1_net     = torch.compile(self.q1_net,     **_ckw)
            self.q2_net     = torch.compile(self.q2_net,     **_ckw)
            if not self.use_lambda_returns:
                self.q1_target = torch.compile(self.q1_target, **_ckw)
                self.q2_target = torch.compile(self.q2_target, **_ckw)

        # ── optimisers ────────────────────────────────────────────────────────
        # On CUDA use the *fused* Adam kernel: it updates all parameters in one
        # on-device launch and, crucially, keeps the step counter on the GPU. The
        # default single-tensor path calls `_get_value(step).item()` once PER
        # parameter, forcing ~one CPU↔GPU sync per param per update (~390 syncs /
        # update here) — a large, pure-overhead cost. fused=True removes it.
        adam_kw = {}
        if torch.device(self.device).type == "cuda":
            adam_kw["fused"] = True
            # capturable=True keeps Adam's step counter on the GPU (no host sync),
            # which is mandatory for the optimiser step to run inside a CUDA graph.
            if self._use_cudagraph:
                adam_kw["capturable"] = True
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, **adam_kw)
        self.q1_optimizer     = optim.Adam(self.q1_net.parameters(),     lr=learning_rate, **adam_kw)
        self.q2_optimizer     = optim.Adam(self.q2_net.parameters(),     lr=learning_rate, **adam_kw)

        # ── replay buffer ─────────────────────────────────────────────────────
        # JaxReplayBuffer keeps all data on the GPU and samples via a JIT-compiled
        # gather + DLPack, eliminating the CPU random-indexing bottleneck.
        # Only use with the JAX parallel backend on a GPU with sufficient VRAM.
        # TD(λ) samples its gradient minibatches from the numpy λ-return CACHE,
        # not from the replay buffer (the buffer is only read during the periodic
        # cache refresh).  So the GPU-resident JaxReplayBuffer brings no speed-up
        # here and its arrays live on-device where the numpy cache build can't read
        # them — fall back to the CPU ReplayBuffer.  Also round capacity down to a
        # whole number of env-interleaved rows so the per-env stride stays aligned
        # across the circular wrap.
        if use_lambda_returns:
            if self._lambda_env_stride > 1:
                replay_buffer_size = ((replay_buffer_size // self._lambda_env_stride)
                                      * self._lambda_env_stride)
            if use_jax_buffer:
                print("TD(λ) enabled → using CPU ReplayBuffer (the λ-return cache, "
                      "not buffer.sample(), is the hot path; JaxReplayBuffer gives "
                      "no benefit and can't be read by the numpy cache build).")
            self.replay_buffer = ReplayBuffer(replay_buffer_size, obs_shape, act_shape)
            print(f"Using CPU ReplayBuffer ({replay_buffer_size} capacity, "
                  f"env_stride={self._lambda_env_stride})")
        elif use_jax_buffer:
            if not _JAX_BUFFER_AVAILABLE:
                raise RuntimeError(
                    "use_jax_buffer=True requires JAX with CUDA. "
                    "Install: pip install 'jax[cuda12]'"
                )
            self.replay_buffer = JaxReplayBuffer(replay_buffer_size, obs_shape, act_shape)
            print(f"Using JaxReplayBuffer (GPU-resident, {replay_buffer_size} capacity)")
        else:
            self.replay_buffer = ReplayBuffer(replay_buffer_size, obs_shape, act_shape)

        self.train_step = 0   # counts env interactions only (not gradient steps)

    # =========================================================================
    # Utilities
    # =========================================================================

    def _hard_update(self, target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())

    def _soft_update(self, target: nn.Module, source: nn.Module):
        # Polyak: tp ← (1-τ)·tp + τ·sp.  lerp_(b, w): a ← a + w·(b−a), which is
        # exactly this.  Done as ONE fused _foreach_ kernel instead of ~4 kernel
        # launches per parameter (×~65 params ×2 target nets, every update) — pure
        # launch overhead that dominates when the model is small and the loop is
        # launch-bound rather than compute-bound.
        with torch.no_grad():
            torch._foreach_lerp_(
                list(target.parameters()), list(source.parameters()), self.tau
            )

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

    def _sample(self, net, state):
        """Tanh-squashed action + log π(a|s), routing the forward through the
        module's ``__call__`` so torch.compile actually applies.

        ``net.sample()`` BYPASSES the compiled graph — torch.compile only wraps
        ``forward``/``__call__``, so a compiled module's ``.sample()`` falls back
        to the eager forward (and eager backward).  In SAC the policy is only ever
        used via sampling, so the whole transformer forward+backward ran eager —
        ~50 % of the update time and the main cause of low GPU utilisation.  Here
        the heavy ``net(state)`` is the compiled forward; only the cheap
        elementwise sampling math runs eager.  Numerically identical to
        P_network.sample.
        """
        mu, sigma  = net(state)
        eps        = torch.randn_like(sigma)
        action     = torch.tanh(mu + eps * sigma)
        log_prob   = -0.5 * eps ** 2 - sigma.log() - 0.5 * math.log(2.0 * math.pi)
        log_prob   = log_prob - torch.log(1.0 - action ** 2 + 1e-6)
        log_prob   = log_prob.sum(dim=(-2, -1), keepdim=True).squeeze(-1)
        return action, log_prob

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Single-env action — stochastic during training."""
        state_t = self._to(torch.FloatTensor(state).unsqueeze(0))
        state_t = self.state_preprocessor(state_t)
        with torch.no_grad(), self._autocast():
            action, _ = self._sample(self.policy_net, state_t)
        return self.action_postprocessor(action.float().cpu().numpy()[0])

    def select_action_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Vectorised action — takes (n_envs, *obs_shape) and returns
        (n_envs, *act_shape).  Used by the VecEnv training loop.
        """
        states_t = self._to(torch.FloatTensor(states))
        states_t = self.state_preprocessor(states_t)
        with torch.no_grad(), self._autocast():
            actions, _ = self._sample(self.policy_net, states_t)
        actions_np = actions.float().cpu().numpy()
        if not self._has_action_post:
            return actions_np
        # apply postprocessor per-env (handles transformer padding etc.)
        return np.stack([self.action_postprocessor(a) for a in actions_np])

    # =========================================================================
    # Gradient update — SAC v2
    # =========================================================================

    # ── CUDA-graph update (compile_mode="cudagraph") ─────────────────────────

    def _clip_grads_(self, params) -> None:
        """In-place global grad-norm clip with NO host sync (capturable).

        nn.utils.clip_grad_norm_ reads the total norm to the host to decide
        whether to scale, which breaks CUDA-graph capture.  This computes the
        scale entirely on-device: coef = clamp(max_norm / ‖g‖, ≤1), g *= coef.
        """
        if self.max_grad_norm is None:
            return
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return
        total = torch.norm(torch.stack(torch._foreach_norm(grads)))
        coef  = torch.clamp(self.max_grad_norm / (total + 1e-6), max=1.0)
        torch._foreach_mul_(grads, coef)

    def _graph_update_body(self) -> None:
        """The full SAC update on the STATIC graph buffers (self._g_*).

        Identical math to the eager update() below; written against fixed-address
        tensors so it can be captured once and replayed.  Grad clipping uses the
        sync-free _clip_grads_; zero_grad(set_to_none=False) keeps grad addresses
        stable across replays.  alpha/gamma/tau are baked as constants (cudagraph
        requires auto_alpha=False).
        """
        s, a, r        = self._g_states, self._g_actions, self._g_rewards
        ns, d          = self._g_next_states, self._g_dones
        with self._autocast():
            with torch.no_grad():
                a_next, lp_next = self._sample(self.policy_net, ns)
                q1_next = self.q1_target(ns, a_next).unsqueeze(-1)
                q2_next = self.q2_target(ns, a_next).unsqueeze(-1)
                q_target = (r + (1.0 - d) * self.gamma * (
                    torch.min(q1_next, q2_next) - self.alpha * lp_next)).float()
                q_target = torch.nan_to_num(q_target, nan=0.0, posinf=0.0, neginf=0.0)
            q1_pred = self.q1_net(s, a).unsqueeze(-1)
            q1_loss = nn.functional.mse_loss(q1_pred, q_target)
        self.q1_optimizer.zero_grad(set_to_none=False)
        q1_loss.backward()
        self._clip_grads_(self.q1_net.parameters())
        self.q1_optimizer.step()

        with self._autocast():
            q2_pred = self.q2_net(s, a).unsqueeze(-1)
            q2_loss = nn.functional.mse_loss(q2_pred, q_target)
        self.q2_optimizer.zero_grad(set_to_none=False)
        q2_loss.backward()
        self._clip_grads_(self.q2_net.parameters())
        self.q2_optimizer.step()

        with self._autocast():
            a_tilde, lp = self._sample(self.policy_net, s)
            q1_pi = self.q1_net(s, a_tilde).unsqueeze(-1)
            q2_pi = self.q2_net(s, a_tilde).unsqueeze(-1)
            policy_loss = (self.alpha * lp - torch.min(q1_pi, q2_pi)).mean()
        self.policy_optimizer.zero_grad(set_to_none=False)
        policy_loss.backward()
        self._clip_grads_(self.policy_net.parameters())
        self.policy_optimizer.step()

        self._soft_update(self.q1_target, self.q1_net)
        self._soft_update(self.q2_target, self.q2_net)

        # Stash losses in static buffers so logging can read them after replay.
        self._g_q1_loss.copy_(q1_loss.detach())
        self._g_q2_loss.copy_(q2_loss.detach())
        self._g_policy_loss.copy_(policy_loss.detach())

    def _build_cudagraph(self, states, actions, rewards, next_states, dones) -> None:
        """Allocate static buffers, warm up, and capture the update graph."""
        self._g_states      = states.clone()
        self._g_actions     = actions.clone()
        self._g_rewards     = rewards.clone()
        self._g_next_states = next_states.clone()
        self._g_dones       = dones.clone()
        self._g_q1_loss     = torch.zeros((), device=self.device)
        self._g_q2_loss     = torch.zeros((), device=self.device)
        self._g_policy_loss = torch.zeros((), device=self.device)

        # Warm up on a side stream so optimiser state + autograd grad tensors are
        # allocated and cuBLAS/cuDNN pick algorithms BEFORE capture (required).
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                self._graph_update_body()
        torch.cuda.current_stream().wait_stream(warm)

        self._cudagraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cudagraph):
            self._graph_update_body()

    def _update_cudagraph(self) -> None:
        """Sample → copy into static buffers → replay the captured update."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size)
        states      = self.state_preprocessor (self._to(states))
        actions     = self.action_preprocessor(self._to(actions))
        rewards     = self._to(rewards)
        next_states = self.state_preprocessor (self._to(next_states))
        dones       = self._to(dones)

        if self._cudagraph is None:
            self._build_cudagraph(states, actions, rewards, next_states, dones)
        else:
            self._g_states.copy_(states)
            self._g_actions.copy_(actions)
            self._g_rewards.copy_(rewards)
            self._g_next_states.copy_(next_states)
            self._g_dones.copy_(dones)
            self._cudagraph.replay()

        self._update_count += 1
        if self.writer is not None and self._update_count % self._tb_log_every == 0:
            s = self._update_count
            self.writer.add_scalar("Loss/q1",     self._g_q1_loss.item(),     s)
            self.writer.add_scalar("Loss/q2",     self._g_q2_loss.item(),     s)
            self.writer.add_scalar("Loss/policy", self._g_policy_loss.item(), s)
        return None

    def update(self, _profile: bool = False) -> Optional[Dict[str, float]]:
        """
        One SAC v2 gradient step.  train_step is NOT incremented here —
        that is the responsibility of the training loop.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        if self.use_lambda_returns:
            return self._lambda_update()

        if self._use_cudagraph and not self.auto_alpha:
            return self._update_cudagraph()

        def _ck() -> float:
            if _profile and self._dev_type == "cuda":
                torch.cuda.synchronize()
            return time.perf_counter()
        t0 = _ck()

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        # non_blocking=True overlaps H2D copy with GPU compute
        states      = self.state_preprocessor (self._to(states))
        actions     = self.action_preprocessor(self._to(actions))
        rewards     = self._to(rewards)
        next_states = self.state_preprocessor (self._to(next_states))
        dones       = self._to(dones)
        t1 = _ck()

        # ── Q-targets ─────────────────────────────────────────────────────────
        with torch.no_grad(), self._autocast():
            a_next, lp_next = self._sample(self.policy_net, next_states)
            # clone(): under compile_mode="reduce-overhead" each compiled forward
            # returns a view into a shared CUDA-graph static buffer that the NEXT
            # compiled call overwrites.  q1_next must survive the q2_target call
            # below (both feed torch.min), so copy it out of the graph pool.
            q1_next = self.q1_target(next_states, a_next).unsqueeze(-1).clone()
            q2_next = self.q2_target(next_states, a_next).unsqueeze(-1).clone()
            q_target = (rewards + (1.0 - dones) * self.gamma * (
                torch.min(q1_next, q2_next) - self.alpha * lp_next
            )).float()

        q_target = torch.nan_to_num(q_target, nan=0.0, posinf=0.0, neginf=0.0)
        t2 = _ck()

        # ── Q1 update ─────────────────────────────────────────────────────────
        with self._autocast():
            q1_pred = self.q1_net(states, actions).unsqueeze(-1)
            q1_loss = nn.MSELoss()(q1_pred, q_target)
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q1_net.parameters(), self.max_grad_norm)
        self.q1_optimizer.step()
        t3 = _ck()

        # ── Q2 update ─────────────────────────────────────────────────────────
        with self._autocast():
            q2_pred = self.q2_net(states, actions).unsqueeze(-1)
            q2_loss = nn.MSELoss()(q2_pred, q_target)
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q2_net.parameters(), self.max_grad_norm)
        self.q2_optimizer.step()
        t4 = _ck()

        # ── Policy update ──────────────────────────────────────────────────────
        with self._autocast():
            a_tilde, lp = self._sample(self.policy_net, states)
            # clone(): q1_pi must survive the q2_net call before torch.min — see
            # the q1_next clone() note above (CUDA-graph buffer reuse).
            q1_pi = self.q1_net(states, a_tilde).unsqueeze(-1).clone()
            q2_pi = self.q2_net(states, a_tilde).unsqueeze(-1).clone()
            policy_loss = (self.alpha * lp - torch.min(q1_pi, q2_pi)).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        t5 = _ck()

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
        t6 = _ck()

        # ── TensorBoard (throttled) ────────────────────────────────────────────
        # Every logged scalar forces a CPU↔GPU sync (.item() / grad-norm), and the
        # sigma read costs an extra policy forward.  On the JAX backend updates run
        # ~once per vector step, so logging every update would stall the GPU on
        # every step.  Throttle to once per `tb_log_every` updates: the curves are
        # still smooth but the syncs are amortised out of the hot path.
        s = self._update_count
        if self.writer is not None and (s % self._tb_log_every == 0):
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

        # Only the profiler consumes the return value; building it unconditionally
        # cost 3 × .item() (3 host syncs) on every update for nothing.
        if _profile:
            return {
                "_phases": {
                    "sample":   t1 - t0,
                    "q_target": t2 - t1,
                    "q1":       t3 - t2,
                    "q2":       t4 - t3,
                    "pi":       t5 - t4,
                    "tail":     t6 - t5,
                },
            }
        return None

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
    #   • Promote S/B "blocks" of B transitions from the replay buffer into a
    #     cache C of size S (`_build_lambda_cache`). A block is one env's
    #     temporally-ordered trajectory: with single-env collection that is a
    #     contiguous slice; with vectorised collection (env_stride S>1 the buffer
    #     is filled env-interleaved) it is a STRIDED slice, so the recursion never
    #     chains unrelated envs.  The stored `done` flags cut returns at episode
    #     boundaries.
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
                a, lp = self._sample(self.policy_net, ns)          # a:[c,...], lp:[c,1]
                # clone() q1 so the q2_net call can't overwrite it in the shared
                # CUDA-graph buffer before torch.min (reduce-overhead mode).
                q1 = self.q1_net(ns, a).unsqueeze(-1).clone()      # [c,1]
                q2 = self.q2_net(ns, a).unsqueeze(-1)              # [c,1]
                v  = torch.min(q1, q2) - self.alpha * lp           # [c,1]
                out.append(v.squeeze(-1).float().cpu().numpy())
        v_all = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
        return np.nan_to_num(v_all, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _build_lambda_cache(self) -> bool:
        """Promote per-env temporally-contiguous blocks into the cache and fill it
        with precomputed λ-return targets. Returns True on success.

        With env_stride S the buffer is filled env-interleaved (row r, env e at
        flat index r·S + e), so one env's trajectory is the STRIDED sequence
        r·S+e, (r+1)·S+e, …  We sample (start_row, env) pairs and gather strided
        blocks, so every block is a single env's contiguous trajectory and the
        λ-return recursion never chains unrelated envs; the stored `done` flags
        still cut returns at episode boundaries.  S = 1 recovers the original
        single-env contiguous behaviour exactly.
        """
        buf = self.replay_buffer
        B   = self.block_size
        S   = self._lambda_env_stride
        cap = buf._max
        n_rows_filled = buf._size // S            # complete env-interleaved rows
        if n_rows_filled < B:
            return False
        n_blocks = max(1, self.cache_size // B)
        env_idx  = np.random.randint(0, S, size=n_blocks)   # which env per block

        # ── sample valid block start ROWS (no seam-straddling) ────────────────
        if buf._size < cap:
            # Buffer not yet full: valid rows live in [0, n_rows_filled).
            start_row = np.random.randint(0, n_rows_filled - B + 1, size=n_blocks)
        else:
            # Full buffer: reject blocks whose row interval crosses the write
            # pointer's row (the newest|oldest seam).
            n_rows  = cap // S
            ptr_row = buf._ptr // S
            start_row = np.empty(n_blocks, dtype=np.int64)
            filled = 0
            while filled < n_blocks:
                cand = np.random.randint(0, n_rows - B + 1, size=n_blocks - filled)
                ok   = ~((cand < ptr_row) & (ptr_row < cand + B))
                good = cand[ok]
                start_row[filled:filled + len(good)] = good
                filled += len(good)

        # flat index of (start_row + t)·S + env  for t in [0, B)
        starts    = start_row * S + env_idx                        # [n_blocks]
        block_idx = starts[:, None] + (np.arange(B) * S)[None, :]   # [n_blocks, B]
        flat_idx  = block_idx.reshape(-1)

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

        targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

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
            a_tilde, lp = self._sample(self.policy_net, states)
            # clone(): q1_pi must survive the q2_net call before torch.min, since
            # reduce-overhead returns views into a reused CUDA-graph buffer.
            q1_pi = self.q1_net(states, a_tilde).unsqueeze(-1).clone()
            q2_pi = self.q2_net(states, a_tilde).unsqueeze(-1).clone()
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

        # ── TensorBoard (throttled — see update() for rationale) ───────────────
        s = self._update_count
        if self.writer is not None and (s % self._tb_log_every == 0):
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
        return None

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
                        else self._sample(self.policy_net, state_t)[0]
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
