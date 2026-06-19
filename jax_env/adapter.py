"""
JaxVecEnvAdapter — vectorised JAX environment adapter for SAC training.

Observations : float32[num_envs, NET_MP + 1, TOKEN_DIM=35]  (hybrid fleet-into-
               planet decoder: NET_MP planet tokens + 1 meta row; see jax_obs.py)
Actions      : float32[num_envs, NET_MAX_PLANETS=40, ACTION_DIM=4]

Action decoding is fully vectorised across all envs via batched einsum
pairwise wedge + direct atan2 aim (no per-tick lead simulation).

Opponent modes (set via ``opponent`` constructor arg):
  "random"     — random angle / random fraction for all owned planets
  "greedy"     — vectorised greedy: score-based target selection + direct aim
                 (mirrors agent1.get_custom_score heuristic without state).
                 NOT the genuine RuleBasedAgent — that lives in the Python
                 backend; "rule_based" is accepted here as a deprecated alias.
  "mixed"      — per-step blend of "random" and "greedy" (see mixed_random_ratio)
  "self_play"  — same policy network for all players; call set_policy() after
                 the trainer is created
"""
from __future__ import annotations

import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

from .vec_env import VectorizedEnv
from .constants import MAX_PLANETS as _JAX_MP, MAX_FLEETS as _JAX_MF
from .reset import batch_reset as _batch_reset
# On-device (JAX) opponent / reward / obs — the "fast path" that fuses these with
# the engine step into one jitted, vmapped function so they run on the GPU instead
# of in host NumPy.  The NumPy versions below stay as the fallback (self_play /
# LaunchDistancePenalty) and the parity oracle.
from .jax_opponents import make_opponent_fn as _make_opponent_fn
from .jax_reward import build_reward_fn as _build_reward_fn, SUPPORTED as _JAX_REWARD_OK
from .jax_obs import (
    make_extract_obs as _make_extract_obs,
    TOKEN_DIM as _TOKEN_DIM,
    SEQ_LEN as _SEQ_LEN,
)

_FAST_OPPONENTS = ("random", "greedy", "rule_based", "agent1", "mixed")

# The NumPy fallback/parity opponents live in agents/vec_opponents.py so the
# strategy logic is easy to find and edit.  Load them by explicit FILE PATH, not
# `import agents.vec_opponents`: kaggle_environments puts a different top-level
# `agents` module on sys.path (envs/lux_ai_s3/agents.py) that would shadow the
# project package and crash on its own relative import (see the note in
# train_orbit_wars.py).  Path loading sidesteps the name collision.
import importlib.util as _ilu
import os as _os
_vo_path = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "agents", "vec_opponents.py")
_vo_spec = _ilu.spec_from_file_location("orbit_vec_opponents", _vo_path)
_vec_opponents = _ilu.module_from_spec(_vo_spec)
_vo_spec.loader.exec_module(_vec_opponents)
greedy_opponent = _vec_opponents.greedy_opponent
agent1_opponent = _vec_opponents.agent1_opponent
random_opponent = _vec_opponents.random_opponent

# Network / observation shape constants (must match OrbitWarsEnv)
_NET_MP    = 40
_NET_MF    = 100        # kept for the legacy fast-path opponent fleet sweep; obs no
                        # longer emits per-fleet tokens (hybrid fleet-into-planet decoder)
_STATE_DIM = _TOKEN_DIM     # 35 — width of the hybrid planet token (see jax_obs.py)
_ACT_DIM   = 4

_CENTER     = 50.0
_ROT_LIMIT  = 50.0
_MAX_SPEED  = 6.0
_SUN_RADIUS = 10.0

# Precomputed constants reused every step
_PIDX, _QIDX = np.triu_indices(4, k=1)          # 6 upper-triangle pairs for d=4
_DIAG_MASK   = ~np.eye(_NET_MP, dtype=bool)      # [NMP, NMP] — exclude self-to-self


def _fleet_speed_np(ships: np.ndarray) -> np.ndarray:
    s = np.maximum(1.0, ships)
    return np.minimum(
        _MAX_SPEED,
        1.0 + (_MAX_SPEED - 1.0) * (np.log(s) / np.log(1000.0)) ** 1.5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reward-scheme support
#
# The Python backend composes a reward from a list of single-responsibility
# components (env/orbit_wars.py).  The JAX adapter mirrors each one with a
# vectorised numpy implementation that reads the pre-step (self._state) and
# post-step (new_state) GameState arrays directly — no per-env Python loop.
#
# Legacy numbered schemes (RewardScheme1-4) are thin compositions; they are
# expanded into their atomic components at construction time so the per-step
# reward path only ever handles atomic components.
# ─────────────────────────────────────────────────────────────────────────────

# Atomic components handled directly by _compute_config_reward.
_ATOMIC_SCHEMES = {
    "RelativeShipAdvantage", "RelativeProductionAdvantage", "ShipGrowth",
    "ProductionPlanetDelta", "ProximityCaptureBonus", "AbsoluteHoldings",
    "FleetLaunchPenalty", "LaunchDistancePenalty", "StepPenalty",
    "TerminalWinBonus", "TimeDecayWinBonus",
}


def _expand_scheme(name: str, params: dict) -> list:
    """Expand a (possibly legacy/composite) scheme into a list of
    (atomic_name, atomic_params) tuples, mirroring env/orbit_wars.py exactly."""
    if name in _ATOMIC_SCHEMES:
        return [(name, dict(params))]

    ship   = params.get("ship_scale",   0.01)
    planet = params.get("planet_scale", 1.0)
    bonus  = params.get("win_bonus",    100.0)
    if name == "RewardScheme1":
        return [("RelativeShipAdvantage",       {"ship_scale": ship}),
                ("RelativeProductionAdvantage", {"planet_scale": planet}),
                ("TerminalWinBonus",            {"win_bonus": bonus})]
    if name == "RewardScheme2":
        return [("ShipGrowth",           {"ship_scale": ship}),
                ("ProductionPlanetDelta", {"planet_scale": planet}),
                ("TerminalWinBonus",      {"win_bonus": bonus})]
    if name == "RewardScheme3":
        return [("FleetLaunchPenalty", {"ship_scale": params.get("ship_scale", 0.5)})]
    if name == "RewardScheme4":
        return [("AbsoluteHoldings",  {"ship_scale": ship, "planet_scale": planet}),
                ("TerminalWinBonus",  {"win_bonus": bonus})]
    raise ValueError(
        f"JAX adapter does not support reward scheme {name!r}. "
        f"Supported: {sorted(_ATOMIC_SCHEMES | {'RewardScheme1', 'RewardScheme2', 'RewardScheme3', 'RewardScheme4'})}"
    )


def _swept_pair_hit_batch(fx0, fy0, fx1, fy1, px0, py0, px1, py1, r):
    """Vectorised continuous swept-pair collision test (interpreter model).

    True where a point moving (fx0,fy0)->(fx1,fy1) and a circle of radius r
    moving (px0,py0)->(px1,py1) come within r for some t in [0, 1].
    All inputs broadcast against each other; returns a bool array.
    """
    d0x = fx0 - px0
    d0y = fy0 - py0
    dvx = (fx1 - fx0) - (px1 - px0)
    dvy = (fy1 - fy0) - (py1 - py0)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    disc = b * b - 4.0 * a * c
    sq = np.sqrt(np.maximum(disc, 0.0))
    a_safe = np.where(a < 1e-12, 1.0, a)
    t1 = (-b - sq) / (2.0 * a_safe)
    t2 = (-b + sq) / (2.0 * a_safe)
    moving_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    static_hit = c <= 0.0          # a≈0: relative motion negligible, test start
    return np.where(a < 1e-12, static_hit, moving_hit)


def _seg_dist_to_center(ax, ay, bx, by):
    """Vectorised min distance from the board centre (50,50) to segment a→b."""
    vx, vy = bx - ax, by - ay
    l2 = vx * vx + vy * vy
    l2_safe = np.where(l2 > 0.0, l2, 1.0)
    t = np.clip(((_CENTER - ax) * vx + (_CENTER - ay) * vy) / l2_safe, 0.0, 1.0)
    px = ax + t * vx
    py = ay + t * vy
    return np.sqrt((_CENTER - px) ** 2 + (_CENTER - py) ** 2)


class JaxVecEnvAdapter:
    """
    Thin wrapper around VectorizedEnv that speaks the same observation/action
    language as OrbitWarsEnv, so the same SAC training loop works with both.

    Parameters
    ----------
    num_envs        : number of parallel JAX environments
    num_players     : 2 or 4
    episode_steps   : max ticks per episode (time-limit done)
    ship_speed      : physics constant forwarded to VectorizedEnv
    comet_speed     : physics constant forwarded to VectorizedEnv
    tanh_scale      : pairwise-wedge tanh saturation scale
    min_fleet_ships : unused (kept for API parity with OrbitWarsEnv)
    reward_type     : "ship_advantage" (shaped) | "native" (terminal ±1 only)
    reward_scale    : per-step ship-advantage multiplier
    win_bonus       : terminal win/loss bonus magnitude
    opponent        : "random" | "greedy" | "agent1" | "mixed" | "self_play"
                      ("rule_based" is a deprecated alias for "greedy".)
                      "agent1" is a fuller vectorised port of the agents/agent1.py
                      rule-based bot (defence + accurate-ship-count attack).
                      For "self_play", call set_policy() after trainer creation.
                      For "mixed", each opponent independently acts randomly with
                      probability ``mixed_random_ratio`` (else greedy) on every
                      step; anneal the ratio over training with
                      set_mixed_random_ratio() to mirror the Python MixedAgent
                      curriculum (1.0 → fully random, 0.0 → fully greedy).
    mixed_random_ratio : initial random-vs-greedy blend for opponent="mixed".
    mixed_send_prob    : per-owned-planet launch probability for the random half.
    """

    MAX_PLANETS = _NET_MP
    MAX_FLEETS  = _NET_MF
    STATE_DIM   = _STATE_DIM
    ACTION_DIM  = _ACT_DIM

    def __init__(
        self,
        num_envs:        int,
        num_players:     int   = 2,
        episode_steps:   int   = 500,
        ship_speed:      float = 6.0,
        comet_speed:     float = 4.0,
        tanh_scale:      float = 0.2,
        min_fleet_ships: int   = 3,
        reward_type:     str   = "ship_advantage",
        reward_scale:    float = 0.01,
        win_bonus:       float = 100.0,
        opponent:        str   = "random",
        reward_cfg:      Optional[list] = None,
        mixed_random_ratio: float = 0.5,
        mixed_send_prob:    float = 0.3,
        reset_pool_size: int   = 2048,
    ):
        self.num_envs        = num_envs
        self.num_players     = num_players
        self.tanh_scale      = tanh_scale
        self.min_fleet_ships = min_fleet_ships
        self.reward_type     = reward_type
        self.reward_scale    = reward_scale
        self.win_bonus       = win_bonus
        self.opponent        = opponent
        # opponent="mixed": per-step blend of random and greedy actions. The
        # ratio is mutable so the training curriculum can anneal it (see
        # set_mixed_random_ratio); send_prob is the random half's launch chance.
        self.mixed_random_ratio = float(mixed_random_ratio)
        self.mixed_send_prob    = float(mixed_send_prob)
        self._policy_fn: Optional[Callable] = None
        self._episode_steps  = episode_steps

        # Parse config-based reward schemes.  When provided, these take precedence
        # over reward_type/reward_scale/win_bonus and mirror the Python-backend
        # reward exactly for apples-to-apples comparisons.  Legacy composite
        # schemes are expanded into their atomic components here, so the per-step
        # reward path only handles atomic components.  Stored as a list of
        # (atomic_name, params) tuples (a scheme may appear more than once).
        self._reward_cfg: Optional[list] = None
        self._reward_names: set = set()
        if reward_cfg:
            parsed: list = []
            for cfg in reward_cfg:
                scheme = cfg.get("scheme", "")
                params = {k: v for k, v in cfg.items() if k != "scheme"}
                parsed.extend(_expand_scheme(scheme, params))
            if parsed:
                self._reward_cfg = parsed
                self._reward_names = {n for n, _ in parsed}

        self._jax = VectorizedEnv(
            num_envs=num_envs,
            num_players=num_players,
            episode_steps=episode_steps,
            ship_speed=ship_speed,
            comet_speed=comet_speed,
        )
        self._state        = None
        self._seed_counter = 0
        self._prev_scores  = None

        seq_len = _SEQ_LEN                      # NET_MP planet tokens + 1 meta row
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(seq_len, _STATE_DIM), dtype=np.float32,
        )
        self.single_observation_space = self.observation_space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(_NET_MP, _ACT_DIM), dtype=np.float32,
        )
        self.single_action_space = self.action_space

        self.last_fleets_sent = 0
        self.last_n_fleets    = 0

        # ── On-device fast path ───────────────────────────────────────────────
        # When the opponent is a vectorised heuristic and every reward component is
        # JAX-portable, the opponent + engine step + reward run as ONE jitted,
        # vmapped function and obs extraction is jitted too — so per step the host
        # only does the done-env auto-reset and one obs copy, not the agent1 sweep
        # / reward / obs in NumPy.  self_play (torch opponent), LaunchDistancePenalty
        # (per-tick sim) and the reward_type fallback stay on the NumPy path.
        self._jax_fast = (
            opponent in _FAST_OPPONENTS
            and self._reward_cfg is not None
            and self._reward_names <= _JAX_REWARD_OK
        )
        # Observation extraction is on-device (JAX) on EVERY path — the hybrid
        # fleet-into-planet decoder is the single obs implementation.  One jitted
        # extractor per perspective (pid); pid 0 is the acting player, pid≥1 are
        # built lazily for self_play opponent observations.
        self._obs_extractors  = {0: _make_extract_obs(0)}
        self._extract_obs_jax = self._obs_extractors[0]
        if self._jax_fast:
            self._key          = jax.random.PRNGKey(0)
            self._extract_obs_jax = _make_extract_obs(0)
            self._reset_pool_size = reset_pool_size
            self._reset_pool = _batch_reset(
                np.arange(reset_pool_size) + 1_000_000,
                num_players=num_players,
                episode_steps=episode_steps,
                ship_speed=ship_speed,
                comet_speed=comet_speed,
            )
            self._pool_cursor = 0
            self._fused        = self._build_fused()

    def _build_fused(self):
        """Build the jitted, vmapped (opponent + engine step + reward + auto-reset)
        function.  When a done env is detected after the engine step, the post-step
        state is replaced on-device with a pre-generated pool state — no host
        roundtrip.

        Signature (vmapped over batch dim 0):
        ``(state, p0_engine, key, mix_ratio, pool_state) -> (new_state, reward, done, won)``
        """
        NP         = self.num_players
        NET        = _NET_MP
        step_bound = self._jax._step_bound
        opp_fn     = _make_opponent_fn(self.opponent, NP, self.mixed_send_prob)
        reward_fn  = _build_reward_fn(self._reward_cfg, self._episode_steps, NP)

        def _single(state, p0_engine, key, mix_ratio, pool_state):
            opp  = opp_fn(state, key, mix_ratio)
            full = opp.at[0, :NET, :].set(p0_engine)
            new_state = step_bound(state, full)
            reward = reward_fn(state, new_state)
            done   = new_state.done
            won    = (new_state.rewards[0] > 0) & done
            new_state = jax.lax.cond(done,
                                     lambda: pool_state,
                                     lambda: new_state)
            return new_state, reward, done, won

        return jax.jit(jax.vmap(_single, in_axes=(0, 0, 0, None, 0)))

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_policy(self, policy_fn: Callable):
        """Register the policy callable for self-play opponents.

        policy_fn(obs: ndarray[N, seq_len, STATE_DIM]) → ndarray[N, NET_MP, ACTION_DIM]
        Called once per opponent player per vector-step.
        """
        self._policy_fn = policy_fn

    def set_mixed_random_ratio(self, ratio: float):
        """Update the random-vs-greedy blend for opponent="mixed".

        Mirrors annealing MixedAgent.random_ratio in the Python backend: pass a
        value in [0, 1] (1.0 → fully random, 0.0 → fully greedy). No-op for
        other opponent modes.
        """
        self.mixed_random_ratio = float(np.clip(ratio, 0.0, 1.0))

    def reset(self, seed=None, options=None):
        seeds = np.arange(self._seed_counter, self._seed_counter + self.num_envs)
        self._seed_counter += self.num_envs
        self._state = self._jax.reset(seeds)
        self._prev_scores = self._compute_scores(self._state)
        return self._extract_obs(self._state), {}

    def close(self):
        pass

    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------

    def decode_state(self):
        """Planet arrays (player-0 perspective) + omega the torch decoder needs.

        Returned for the CURRENT state (the one the last-returned obs describes),
        so the learner can decode player-0's engine action before the next step.
        All arrays are numpy [num_envs, NET_MP]; omega is [num_envs].  Used by the
        env-light training path (decode in the learner) and mirrors exactly the
        comet-free, first-NET_MP-planets view env/orbit_wars.py decodes from.
        """
        s = self._state
        P = _NET_MP
        active = (np.asarray(s.planets.active, dtype=bool)[:, :P]
                  & ~np.asarray(s.planets.is_comet, dtype=bool)[:, :P])
        return {
            "owner":  np.asarray(s.planets.owner,  dtype=np.int32)[:, :P],
            "x":      np.asarray(s.planets.x,       dtype=np.float32)[:, :P],
            "y":      np.asarray(s.planets.y,       dtype=np.float32)[:, :P],
            "r":      np.asarray(s.planets.radius,  dtype=np.float32)[:, :P],
            "ships":  np.asarray(s.planets.ships,   dtype=np.float32)[:, :P],
            "active": active,
            "omega":  np.asarray(s.angular_velocity, dtype=np.float32).reshape(-1),
        }

    def step(self, actions_np: np.ndarray):
        """Decode raw player-0 policy output internally, then step.

        Convenience path (benchmarks, tests, single-process use). The env-light
        training loops instead decode player-0 in the learner and call
        ``step_engine``; this keeps a self-contained entry point.

        actions_np : float32[num_envs, NET_MP, ACTION_DIM] — player-0 policy output.
        Returns (obs_next, rewards, dones, truncateds, wons).
        """
        if self._jax_fast:
            return self._post_step_jax(self._decode_p0_numpy(actions_np))
        jax_acts = self._decode_actions_batch(actions_np)
        return self._post_step(jax_acts)

    def step_engine(self, p0_engine: np.ndarray):
        """Step with a pre-decoded player-0 ENGINE action (env-light path).

        p0_engine : float32[num_envs, NET_MP, 2] = (angle, absolute ship count),
        as produced by model.action_decoder.decode in the learner. Opponents are
        still generated here (they are env-side heuristics that need live state).
        Returns (obs_next, rewards, dones, truncateds, wons).
        """
        if self._jax_fast:
            return self._post_step_jax(p0_engine)
        jax_acts = self._build_engine_actions(p0_engine)
        return self._post_step(jax_acts)

    def _post_step(self, jax_acts):
        new_state, rewards_jax, dones_jax = self._jax.step(self._state, jax_acts)

        dones_np = np.array(dones_jax, dtype=bool)
        r_jax_np = np.array(rewards_jax, dtype=np.float32)   # [N, num_players]

        self.last_n_fleets = int(np.array(new_state.fleets.active).sum(axis=-1).mean())

        # ── reward ─────────────────────────────────────────────────────────────
        if self._reward_cfg is not None:
            player0_rewards = self._compute_config_reward(
                new_state, r_jax_np, dones_np
            )
        elif self.reward_type == "native":
            player0_rewards = r_jax_np[:, 0]
        else:
            new_scores = self._compute_scores(new_state)
            my_delta   = new_scores[:, 0] - self._prev_scores[:, 0]
            opp_delta  = (new_scores[:, 1:].sum(axis=1)
                          - self._prev_scores[:, 1:].sum(axis=1))
            player0_rewards = self.reward_scale * (my_delta - opp_delta)
            win_mask  = (r_jax_np[:, 0] > 0) & dones_np
            loss_mask = (r_jax_np[:, 0] < 0) & dones_np
            player0_rewards += self.win_bonus * win_mask.astype(np.float32)
            player0_rewards -= self.win_bonus * loss_mask.astype(np.float32)
            self._prev_scores = new_scores

        player0_wons = (r_jax_np[:, 0] > 0) & dones_np

        new_state  = self._auto_reset(new_state, dones_np)
        self._state = new_state
        obs_next   = self._extract_obs(new_state)
        truncateds = np.zeros(self.num_envs, dtype=bool)
        return obs_next, player0_rewards, dones_np, truncateds, player0_wons

    def _auto_reset(self, new_state, dones_np):
        """Reset only the finished envs and scatter them back into the batched
        state with a single device-side ``.at[idx].set()`` per leaf.  reset_single
        is still Python (rejection sampling can't be JIT-compiled), but runs only
        for the done envs; the merge is a vectorised scatter."""
        done_idxs = np.where(dones_np)[0]
        if len(done_idxs) > 0:
            n_done = len(done_idxs)
            fresh_seeds = np.arange(self._seed_counter, self._seed_counter + n_done)
            self._seed_counter += n_done
            fresh_list = [self._jax.reset_single(int(s)) for s in fresh_seeds]
            fresh_batch = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs), *fresh_list
            )
            idx_j = jnp.asarray(done_idxs)
            new_state = jax.tree_util.tree_map(
                lambda full, fr: full.at[idx_j].set(fr), new_state, fresh_batch
            )
            if self.reward_type != "native" and self._reward_cfg is None:
                self._prev_scores = self._compute_scores(new_state)
        return new_state

    @functools.cached_property
    def _gather_pool(self):
        """JIT-compiled pool gather: (pool, idxs) -> batched GameState."""
        @jax.jit
        def _gather(pool, idxs):
            return jax.tree_util.tree_map(lambda x: x[idxs], pool)
        return _gather

    def _next_pool_batch(self):
        """Pick B pre-generated reset states from the circular pool (device-side gather)."""
        B = self.num_envs
        idxs = (self._pool_cursor + jnp.arange(B)) % self._reset_pool_size
        self._pool_cursor = int((self._pool_cursor + B) % self._reset_pool_size)
        return self._gather_pool(self._reset_pool, idxs)

    def _post_step_jax(self, p0_engine: np.ndarray):
        """Fast path: opponent + engine step + reward + on-device auto-reset, all
        in one jitted vmapped call.  Done envs are replaced with pre-generated pool
        states entirely on-device — no host-side Python reset loop."""
        B  = self.num_envs
        p0 = np.asarray(p0_engine, dtype=np.float32)
        self.last_fleets_sent = int((p0[:, :, 1] > 0).sum(axis=1).mean())

        self._key, k = jax.random.split(self._key)
        keys = jax.random.split(k, B)
        pool_batch = self._next_pool_batch()
        new_state, reward_j, dones_j, wons_j = self._fused(
            self._state, jnp.asarray(p0), keys,
            jnp.float32(self.mixed_random_ratio), pool_batch)

        dones_np   = np.asarray(dones_j,   dtype=bool)
        rewards_np = np.asarray(reward_j,  dtype=np.float32)
        wons_np    = np.asarray(wons_j,    dtype=bool)
        self.last_n_fleets = int(np.asarray(new_state.fleets.active).sum(axis=-1).mean())

        self._state = new_state
        obs_next    = np.asarray(self._extract_obs_jax(new_state))
        truncateds  = np.zeros(B, dtype=bool)
        return obs_next, rewards_np, dones_np, truncateds, wons_np

    def _decode_p0_numpy(self, actions_np: np.ndarray) -> np.ndarray:
        """Player-0 wedge → engine action [B, NET_MP, 2], for the convenience
        ``step(raw)`` entry on the fast path (mirrors ``_decode_actions_batch``)."""
        state   = self._state
        p_owner = np.asarray(state.planets.owner,  dtype=np.int32)
        p_x     = np.asarray(state.planets.x,      dtype=np.float32)
        p_y     = np.asarray(state.planets.y,      dtype=np.float32)
        p_ships = np.asarray(state.planets.ships,  dtype=np.float32)
        p_act   = np.asarray(state.planets.active, dtype=bool)
        p_comet = np.asarray(state.planets.is_comet, dtype=bool)
        valid    = p_act[:, :_NET_MP] & ~p_comet[:, :_NET_MP]
        owned_p0 = (p_owner[:, :_NET_MP] == 0) & valid
        tmp = np.zeros((self.num_envs, self.num_players, _JAX_MP, 2), dtype=np.float32)
        self._decode_for_player(actions_np, tmp, 0, owned_p0, valid, p_x, p_y, p_ships)
        return tmp[:, 0, :_NET_MP, :]

    # -------------------------------------------------------------------------
    # Observation extraction — GameState → (num_envs, NET_MP+1, TOKEN_DIM)
    #
    # Hybrid fleet-into-planet decoder (jax_env/jax_obs.py): fleets are folded
    # into the planet tokens they relate to (incoming-hostile / outgoing-friendly
    # slots + pooled summary), and a trailing meta row carries the per-player
    # aggregate.  The on-device JAX builder is the single implementation; its
    # live-game NumPy equivalent for submission is model.SAC.Encoder.encode.
    #
    # When pid != 0, owner IDs are swapped (0 ↔ pid) so the acting player always
    # appears as owner 0 (self_play opponent observations).
    # -------------------------------------------------------------------------

    def _extract_obs(self, state, pid: int = 0) -> np.ndarray:
        fn = self._obs_extractors.get(pid)
        if fn is None:
            fn = _make_extract_obs(pid)
            self._obs_extractors[pid] = fn
        return np.asarray(fn(state))

    # -------------------------------------------------------------------------
    # Action decoding helpers
    # -------------------------------------------------------------------------

    def _pairwise_scores(self, actions_np: np.ndarray) -> np.ndarray:
        """Batch pairwise bivector attention — memory-efficient einsum form.

        Equivalent to calling pairwise_wedge(A[b], A[b]).sum(-1) for each b,
        but vectorised across all B envs with no per-env Python loop.

        actions_np : [B, NMP, 4]
        returns    : [B, NMP, NMP]  tanh-scaled wedge sum
        """
        A  = actions_np
        Ap = A[:, :, _PIDX]   # [B, NMP, 6]  upper-triangle p components
        Aq = A[:, :, _QIDX]   # [B, NMP, 6]  upper-triangle q components
        return np.tanh(self.tanh_scale * (
            np.einsum('bic,bjc->bij', Ap, Aq)
            - np.einsum('bic,bjc->bij', Aq, Ap)
        ))  # [B, NMP, NMP]

    def _decode_for_player(
        self,
        actions_np: np.ndarray,   # [B, NMP, 4]   policy output
        jax_acts:   np.ndarray,   # [B, NP, JAMP, 2]  modified in-place
        pid:        int,
        owned:      np.ndarray,   # [B, NMP] bool — absolute ownership mask for pid
        valid:      np.ndarray,   # [B, NMP] bool — active non-comet planets
        p_x:        np.ndarray,   # [B, JAMP]
        p_y:        np.ndarray,
        p_ships:    np.ndarray,
    ) -> int:
        """Vectorised wedge → direct-aim decoder for one player across all envs.

        Fills jax_acts[b, pid, slot] in-place. Returns total launch count.
        Direct atan2 replaces compute_launch_angle — the JAX engine handles
        flight physics, so lead prediction is not needed here.
        """
        B   = self.num_envs
        NMP = _NET_MP

        scores = self._pairwise_scores(actions_np)   # [B, NMP, NMP]

        src_m  = (owned & (p_ships[:, :NMP] > 1.0))[:, :, np.newaxis]  # [B, NMP, 1]
        tgt_m  = valid[:, np.newaxis, :]                                 # [B, 1, NMP]
        masked = np.where(
            src_m & tgt_m & _DIAG_MASK[np.newaxis] & (scores > 0),
            scores, 0.0
        )  # [B, NMP, NMP]

        bi     = np.arange(B)[:, np.newaxis]   # [B, 1]
        si     = np.arange(NMP)[np.newaxis, :] # [1, NMP]
        best_j = np.argmax(masked, axis=2)     # [B, NMP]
        best_s = masked[bi, si, best_j]        # [B, NMP]

        # Direct-aim angle: atan2(tgt_y - src_y, tgt_x - src_x)
        angles = np.arctan2(
            p_y[bi, best_j] - p_y[:, :NMP],
            p_x[bi, best_j] - p_x[:, :NMP],
        )   # [B, NMP]

        frac      = np.clip(best_s, 0.0, 1.0)
        num_ships = (frac * p_ships[:, :NMP]).astype(np.int32)  # [B, NMP]
        tgt_ships = p_ships[bi, best_j].astype(np.int32)        # [B, NMP]

        launch = (owned
                  & (best_s > 0.0)
                  & (num_ships > tgt_ships)
                  & (num_ships > 0))  # [B, NMP]

        b_idx, slot_idx = np.where(launch)
        jax_acts[b_idx, pid, slot_idx, 0] = angles[b_idx, slot_idx]
        # Engine now consumes ABSOLUTE ship counts (see step._launch_fleets).
        jax_acts[b_idx, pid, slot_idx, 1] = num_ships[b_idx, slot_idx]
        return int(len(b_idx))

    # -------------------------------------------------------------------------
    # Master action decode — dispatches to the chosen opponent strategy
    # -------------------------------------------------------------------------

    def _decode_actions_batch(self, actions_np: np.ndarray) -> jnp.ndarray:
        """
        Decode raw player-0 network output + generate opponent actions.

        actions_np : float32[B, NMP, 4]
        returns    : jnp.float32[B, num_players, JAMP, 2]
        """
        B    = self.num_envs
        NP   = self.num_players
        JAMP = _JAX_MP

        state   = self._state
        p_owner = np.asarray(state.planets.owner,  dtype=np.int32)
        p_x     = np.asarray(state.planets.x,      dtype=np.float32)
        p_y     = np.asarray(state.planets.y,      dtype=np.float32)
        p_ships = np.asarray(state.planets.ships,  dtype=np.float32)
        p_act   = np.asarray(state.planets.active, dtype=bool)
        p_comet = np.asarray(state.planets.is_comet, dtype=bool)

        jax_acts = np.zeros((B, NP, JAMP, 2), dtype=np.float32)
        valid    = p_act[:, :_NET_MP] & ~p_comet[:, :_NET_MP]   # [B, NMP]

        owned_p0 = (p_owner[:, :_NET_MP] == 0) & valid
        self.last_fleets_sent = self._decode_for_player(
            actions_np, jax_acts, 0, owned_p0, valid, p_x, p_y, p_ships
        )
        self._add_opponent_actions(jax_acts)
        return jnp.array(jax_acts, dtype=jnp.float32)

    def _build_engine_actions(self, p0_engine: np.ndarray) -> jnp.ndarray:
        """Assemble the full engine action from a pre-decoded player-0 action.

        p0_engine : float32[B, NMP, 2] = (angle, absolute ships). Player-0 slots
        are copied verbatim; opponents are generated from live state. Returns the
        batched engine action jnp.float32[B, num_players, JAMP, 2].
        """
        B, NP, JAMP = self.num_envs, self.num_players, _JAX_MP
        jax_acts = np.zeros((B, NP, JAMP, 2), dtype=np.float32)
        jax_acts[:, 0, :_NET_MP, :] = p0_engine
        self.last_fleets_sent = int((p0_engine[:, :, 1] > 0).sum(axis=1).mean())
        self._add_opponent_actions(jax_acts)
        return jnp.array(jax_acts, dtype=jnp.float32)

    def _add_opponent_actions(self, jax_acts: np.ndarray):
        """Fill jax_acts[:, 1:, ...] with the chosen opponent strategy in place.

        Opponents are env-side heuristics (greedy / random / mixed / self_play)
        that read the live planet state, so they stay in the env rather than the
        learner.  No-op for single-player games.
        """
        NP = self.num_players
        if NP <= 1:
            return
        B = self.num_envs
        state   = self._state
        p_owner = np.asarray(state.planets.owner,      dtype=np.int32)
        p_x     = np.asarray(state.planets.x,          dtype=np.float32)
        p_y     = np.asarray(state.planets.y,          dtype=np.float32)
        p_r     = np.asarray(state.planets.radius,     dtype=np.float32)
        p_ships = np.asarray(state.planets.ships,      dtype=np.float32)
        p_prod  = np.asarray(state.planets.production, dtype=np.float32)
        p_act   = np.asarray(state.planets.active,     dtype=bool)
        p_comet = np.asarray(state.planets.is_comet,   dtype=bool)
        omega   = np.asarray(state.angular_velocity,   dtype=np.float32).reshape(-1)
        valid   = p_act[:, :_NET_MP] & ~p_comet[:, :_NET_MP]

        if self.opponent == "self_play" and self._policy_fn is not None:
            for pid in range(1, NP):
                opp_obs  = self._extract_obs(state, pid=pid)   # [B, seq, 13]
                opp_acts = self._policy_fn(opp_obs)            # [B, NMP, 4]
                owned_opp = (p_owner[:, :_NET_MP] == pid) & valid
                self._decode_for_player(
                    opp_acts, jax_acts, pid, owned_opp, valid, p_x, p_y, p_ships
                )
        elif self.opponent in ("greedy", "rule_based"):  # rule_based: alias
            greedy_opponent(jax_acts, valid, p_owner, p_x, p_y, p_r,
                            p_ships, p_prod, omega, NP)
        elif self.opponent == "agent1":
            # Fuller rule-based port: needs the live fleet arrays for its defence
            # (incoming enemy fleets) and en-route accounting.
            f_owner  = np.asarray(state.fleets.owner,  dtype=np.int32)
            f_x      = np.asarray(state.fleets.x,      dtype=np.float32)
            f_y      = np.asarray(state.fleets.y,      dtype=np.float32)
            f_angle  = np.asarray(state.fleets.angle,  dtype=np.float32)
            f_ships  = np.asarray(state.fleets.ships,  dtype=np.float32)
            f_active = np.asarray(state.fleets.active, dtype=bool)
            agent1_opponent(jax_acts, valid, p_owner, p_x, p_y, p_r,
                            p_ships, p_prod, omega, NP,
                            f_owner, f_x, f_y, f_angle, f_ships, f_active)
        elif self.opponent == "mixed":
            # Per (env, opponent-player) coin flip: with prob mixed_random_ratio
            # that slot acts randomly this step, else greedy — the vectorised
            # analogue of the Python MixedAgent's per-call random/rule choice. The
            # two strategies are gated by complementary masks so each (env, player)
            # is filled by exactly one strategy (no overwrite).
            use_random = (np.random.random((B, NP)) < self.mixed_random_ratio)
            greedy_opponent(jax_acts, valid, p_owner, p_x, p_y, p_r,
                            p_ships, p_prod, omega, NP, env_gate=~use_random)
            random_opponent(jax_acts, valid, p_owner, p_ships, NP,
                            send_prob=self.mixed_send_prob, env_gate=use_random)
        else:
            random_opponent(jax_acts, valid, p_owner, p_ships, NP)

    # -------------------------------------------------------------------------
    # Ship-score helper (shaped rewards)
    # -------------------------------------------------------------------------

    def _compute_scores(self, state) -> np.ndarray:
        """Return [num_envs, num_players] total (planet + fleet) ships."""
        p_ships = np.asarray(state.planets.ships, dtype=np.float32)  # [B, NP]
        p_owner = np.asarray(state.planets.owner, dtype=np.int32)
        p_act   = np.asarray(state.planets.active, dtype=bool)
        f_ships = np.asarray(state.fleets.ships,  dtype=np.float32)  # [B, NF]
        f_owner = np.asarray(state.fleets.owner,  dtype=np.int32)
        f_act   = np.asarray(state.fleets.active, dtype=bool)

        pids   = np.arange(self.num_players)                          # [num_players]
        pm     = (p_owner[:, :, None] == pids) & p_act[:, :, None]   # [B, NP, P]
        fm     = (f_owner[:, :, None] == pids) & f_act[:, :, None]   # [B, NF, P]
        return (
            (p_ships[:, :, None] * pm).sum(axis=1)
            + (f_ships[:, :, None] * fm).sum(axis=1)
        ).astype(np.float32)  # [B, num_players]

    def _compute_config_reward(
        self,
        new_state,
        r_jax_np: np.ndarray,   # [B, num_players]  native JAX rewards (unused now)
        dones_np:  np.ndarray,   # [B] bool
    ) -> np.ndarray:
        """Compute per-env player-0 reward from config-based schemes.

        Vectorised numpy mirror of the composable components in
        env/orbit_wars.py.  Reads the pre-step (self._state) and post-step
        (new_state) GameState arrays directly; no per-env Python loop.  Legacy
        composite schemes were already expanded into atomic components in
        __init__, so only atomic components are handled here.
        """
        B       = self.num_envs
        NP      = self.num_players
        names   = self._reward_names
        rewards = np.zeros(B, dtype=np.float32)

        old, new = self._state, new_state

        # ── planet / fleet arrays (post-step) ─────────────────────────────────
        p_owner_n = np.asarray(new.planets.owner,      dtype=np.int32)
        p_ships_n = np.asarray(new.planets.ships,      dtype=np.float32)
        p_prod_n  = np.asarray(new.planets.production,  dtype=np.float32)
        p_act_n   = np.asarray(new.planets.active,     dtype=bool)
        p_comet_n = np.asarray(new.planets.is_comet,   dtype=bool)
        valid_n   = p_act_n & ~p_comet_n

        f_owner_n = np.asarray(new.fleets.owner,  dtype=np.int32)
        f_ships_n = np.asarray(new.fleets.ships,  dtype=np.float32)
        f_act_n   = np.asarray(new.fleets.active, dtype=bool)

        pids = np.arange(NP)[None, None, :]

        def _holdings(p_owner, p_ships, p_prod, valid, f_owner, f_ships, f_act):
            """Return (ships[B,NP], prod[B,NP]) — planet+fleet ships and planet
            production per player, mirroring _owned_ships / _owned_production."""
            pm    = (p_owner[:, :, None] == pids) & valid[:, :, None]
            ships = (p_ships[:, :, None] * pm).sum(axis=1)
            prod  = (p_prod[:, :, None]  * pm).sum(axis=1)
            fm    = (f_owner[:, :, None] == pids) & f_act[:, :, None]
            ships = ships + (f_ships[:, :, None] * fm).sum(axis=1)
            return ships.astype(np.float32), prod.astype(np.float32)

        ships_post, prod_post = _holdings(
            p_owner_n, p_ships_n, p_prod_n, valid_n,
            f_owner_n, f_ships_n, f_act_n,
        )

        # ── pre-step holdings (only for delta-based components) ────────────────
        _delta = {"RelativeShipAdvantage", "RelativeProductionAdvantage", "ShipGrowth"}
        if names & _delta:
            p_owner_o = np.asarray(old.planets.owner,    dtype=np.int32)
            p_ships_o = np.asarray(old.planets.ships,    dtype=np.float32)
            p_prod_o  = np.asarray(old.planets.production, dtype=np.float32)
            valid_o   = (np.asarray(old.planets.active, dtype=bool)
                         & ~np.asarray(old.planets.is_comet, dtype=bool))
            f_owner_o = np.asarray(old.fleets.owner,  dtype=np.int32)
            f_ships_o = np.asarray(old.fleets.ships,  dtype=np.float32)
            f_act_o   = np.asarray(old.fleets.active, dtype=bool)
            ships_pre, prod_pre = _holdings(
                p_owner_o, p_ships_o, p_prod_o, valid_o,
                f_owner_o, f_ships_o, f_act_o,
            )

        # ── terminal win/loss result (ship-count based, +1/-1/0 like Python) ──
        if names & {"TerminalWinBonus", "TimeDecayWinBonus"}:
            my       = ships_post[:, 0]
            best_opp = ships_post[:, 1:].max(axis=1) if NP > 1 else np.zeros(B, np.float32)
            result   = np.sign(my - best_opp).astype(np.float32)   # +1 / -1 / 0

        # ── owned-planet slot masks (capture/loss components) ─────────────────
        if names & {"ProductionPlanetDelta", "ProximityCaptureBonus"}:
            p_owner_o2 = np.asarray(old.planets.owner,   dtype=np.int32)
            valid_o2   = (np.asarray(old.planets.active, dtype=bool)
                          & ~np.asarray(old.planets.is_comet, dtype=bool))
            owned_old = (p_owner_o2 == 0) & valid_o2     # [B, P]
            owned_new = (p_owner_n == 0)  & valid_n      # [B, P]
            captured  = owned_new & ~owned_old
            lost      = owned_old & ~owned_new

        # ── exact per-step launched-fleet mask (slot range in circular buffer) ─
        if names & {"FleetLaunchPenalty", "LaunchDistancePenalty"}:
            MF       = f_owner_n.shape[1]
            pre_ptr  = np.asarray(old.next_fleet_slot, dtype=np.int64)   # [B]
            post_ptr = np.asarray(new.next_fleet_slot, dtype=np.int64)   # [B]
            n_total  = post_ptr - pre_ptr                                 # [B] launches (all players)
            slot_idx = np.arange(MF)[None, :]
            rel      = (slot_idx - pre_ptr[:, None]) % MF
            in_range = rel < n_total[:, None]                             # [B, MF]
            # A fleet counts as "launched" only if it survived the step (Python
            # sees it in new_obs); one that hit/oob/sun immediately is dropped.
            p0_launched = in_range & (f_owner_n == 0) & f_act_n           # [B, MF]

        # ── per-scheme contributions ──────────────────────────────────────────
        for name, params in self._reward_cfg:
            if name == "RelativeShipAdvantage":
                s     = params.get("ship_scale", 0.01)
                my_d  = ships_post[:, 0] - ships_pre[:, 0]
                opp_d = (ships_post[:, 1:].sum(1) - ships_pre[:, 1:].sum(1))
                rewards += (s * (my_d - opp_d)).astype(np.float32)

            elif name == "RelativeProductionAdvantage":
                s     = params.get("planet_scale", 1.0)
                my_d  = prod_post[:, 0] - prod_pre[:, 0]
                opp_d = (prod_post[:, 1:].sum(1) - prod_pre[:, 1:].sum(1))
                rewards += (s * (my_d - opp_d)).astype(np.float32)

            elif name == "ShipGrowth":
                s    = params.get("ship_scale", 0.01)
                ls   = params.get("loss_scale", 1.0)
                my_d = ships_post[:, 0] - ships_pre[:, 0]
                scale = np.where(my_d < 0, s * ls, s)
                rewards += (scale * my_d).astype(np.float32)

            elif name == "ProductionPlanetDelta":
                ps   = params.get("planet_scale", 1.0)
                ls   = params.get("loss_scale", 1.0)
                ss   = params.get("ship_scale", 0.0)
                p_ships_o = np.asarray(old.planets.ships, dtype=np.float32)
                p_prod_o  = np.asarray(old.planets.production, dtype=np.float32)
                cap_prod  = (p_prod_n * captured).sum(axis=1)
                lost_prod = (p_prod_o * lost).sum(axis=1)
                ship_cost = (np.log1p(np.maximum(0.0, p_ships_o)) * captured).sum(axis=1)
                rewards += (ps * (cap_prod - ls * lost_prod)
                            - ss * ship_cost).astype(np.float32)

            elif name == "ProximityCaptureBonus":
                scale    = params.get("scale", 1.0)
                ref_dist = max(1e-6, params.get("ref_dist", 25.0))
                px_n = np.asarray(new.planets.x, dtype=np.float32)
                py_n = np.asarray(new.planets.y, dtype=np.float32)
                px_o = np.asarray(old.planets.x, dtype=np.float32)
                py_o = np.asarray(old.planets.y, dtype=np.float32)
                # closeness of each captured planet (post pos) to each
                # already-owned planet (pre pos), summed — [B, P_cap, P_owned]
                dx = px_n[:, :, None] - px_o[:, None, :]
                dy = py_n[:, :, None] - py_o[:, None, :]
                close = np.exp(-np.sqrt(dx * dx + dy * dy) / ref_dist)
                mask  = captured[:, :, None] & owned_old[:, None, :]
                rewards += (scale * (close * mask).sum(axis=(1, 2))).astype(np.float32)

            elif name == "AbsoluteHoldings":
                ss = params.get("ship_scale", 0.01)
                ps = params.get("planet_scale", 1.0)
                rewards += (ss * ships_post[:, 0] + ps * prod_post[:, 0]).astype(np.float32)

            elif name == "FleetLaunchPenalty":
                s = params.get("ship_scale", 0.5)
                rewards -= (s * p0_launched.sum(axis=1)).astype(np.float32)

            elif name == "LaunchDistancePenalty":
                rewards += self._launch_distance_penalty(new, p0_launched, params)

            elif name == "StepPenalty":
                rewards -= np.float32(params.get("weight", 0.1))

            elif name == "TerminalWinBonus":
                wb = params.get("win_bonus", 100.0)
                rewards += (wb * result * dones_np).astype(np.float32)

            elif name == "TimeDecayWinBonus":
                wb    = params.get("win_bonus", 100.0)
                max_s = float(self._episode_steps)
                step_arr = np.asarray(new.step, dtype=np.float32)         # [B]
                decay = np.clip((max_s - step_arr) / max(1.0, max_s - 1.0), 0.0, 1.0)
                rewards += (wb * decay * result * dones_np).astype(np.float32)

        return rewards

    # -------------------------------------------------------------------------
    # LaunchDistancePenalty — vectorised flight-time forward simulation
    # -------------------------------------------------------------------------

    def _launch_distance_penalty(self, new_state, launched_mask, params) -> np.ndarray:
        """Penalise each fleet launched this step by how long it must fly to
        reach a planet (or the full horizon if it reaches none).

        Mirrors env/orbit_wars.py LaunchDistancePenalty / _fleet_flight_ticks:
        the straight-line flight is replayed against the engine's swept-collision
        model (with per-tick planet orbital motion), the source planet excluded.
        All launched fleets across all envs are simulated together; the per-tick
        loop is the only Python loop and short-circuits once every fleet resolves.
        """
        B         = self.num_envs
        scale     = params.get("scale", 1.0)
        time_norm = max(1e-6, params.get("time_norm", 20.0))
        max_ticks = int(params.get("max_ticks", 120))

        b_idx, slot_idx = np.where(launched_mask)
        rewards = np.zeros(B, dtype=np.float32)
        if len(b_idx) == 0:
            return rewards

        # Launched-fleet kinematics (post-step positions = start of the replay).
        fx     = np.asarray(new_state.fleets.x,     dtype=np.float32)[b_idx, slot_idx]
        fy     = np.asarray(new_state.fleets.y,     dtype=np.float32)[b_idx, slot_idx]
        fang   = np.asarray(new_state.fleets.angle, dtype=np.float32)[b_idx, slot_idx]
        fships = np.asarray(new_state.fleets.ships, dtype=np.float32)[b_idx, slot_idx]
        fsrc   = np.asarray(new_state.fleets.from_planet, dtype=np.int32)[b_idx, slot_idx]
        speed  = _fleet_speed_np(fships)                           # [F]

        # Per-fleet planet arrays (gathered from each fleet's env).
        px0   = np.asarray(new_state.planets.x,      dtype=np.float32)[b_idx]   # [F, P]
        py0   = np.asarray(new_state.planets.y,      dtype=np.float32)[b_idx]
        pr    = np.asarray(new_state.planets.radius, dtype=np.float32)[b_idx]
        pact  = np.asarray(new_state.planets.active, dtype=bool)[b_idx]
        pcom  = np.asarray(new_state.planets.is_comet, dtype=bool)[b_idx]
        omega = np.asarray(new_state.angular_velocity, dtype=np.float32)[b_idx]  # [F]

        P       = px0.shape[1]
        p_index = np.arange(P)[None, :]
        # Testable planets: active, non-comet, not the source planet.
        testable = pact & ~pcom & (p_index != fsrc[:, None])      # [F, P]

        orb     = np.sqrt((px0 - _CENTER) ** 2 + (py0 - _CENTER) ** 2)
        ang0    = np.arctan2(py0 - _CENTER, px0 - _CENTER)
        moving  = (omega[:, None] != 0.0) & ((orb + pr) < _ROT_LIMIT) & testable

        cos_a, sin_a = np.cos(fang), np.sin(fang)
        hit_tick = np.full(len(b_idx), float(max_ticks), dtype=np.float32)
        done     = np.zeros(len(b_idx), dtype=bool)
        fxp, fyp = fx.copy(), fy.copy()
        pcxp, pcyp = px0.copy(), py0.copy()

        for k in range(1, max_ticks + 1):
            fxc = fx + cos_a * speed * k
            fyc = fy + sin_a * speed * k
            ang = ang0 + omega[:, None] * k
            pcxc = np.where(moving, _CENTER + orb * np.cos(ang), px0)
            pcyc = np.where(moving, _CENTER + orb * np.sin(ang), py0)

            hit = _swept_pair_hit_batch(
                fxp[:, None], fyp[:, None], fxc[:, None], fyc[:, None],
                pcxp, pcyp, pcxc, pcyc, pr,
            ) & testable                                           # [F, P]
            any_hit = hit.any(axis=1) & ~done
            hit_tick = np.where(any_hit, float(k), hit_tick)
            done = done | any_hit

            # Planet hit takes priority; only otherwise can a fleet miss (off the
            # board or into the sun → charged the full horizon, hit_tick=max).
            oob = (fxc < 0.0) | (fxc > 100.0) | (fyc < 0.0) | (fyc > 100.0)
            sun = _seg_dist_to_center(fxp, fyp, fxc, fyc) < _SUN_RADIUS
            done = done | ((oob | sun) & ~done)

            if done.all():
                break
            fxp, fyp = fxc, fyc
            pcxp, pcyp = pcxc, pcyc

        total = np.zeros(B, dtype=np.float64)
        np.add.at(total, b_idx, hit_tick)
        rewards -= (scale * total / time_norm).astype(np.float32)
        return rewards
