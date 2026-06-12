"""
JaxVecEnvAdapter — vectorised JAX environment adapter for SAC training.

Observations : float32[num_envs, MAX_PLANETS + MAX_FLEETS, STATE_DIM=13]
Actions      : float32[num_envs, NET_MAX_PLANETS=40, ACTION_DIM=4]

Action decoding is fully vectorised across all envs via batched einsum
pairwise wedge + direct atan2 aim (no per-tick lead simulation).

Opponent modes (set via ``opponent`` constructor arg):
  "random"     — random angle / random fraction for all owned planets
  "rule_based" — vectorised greedy: score-based target selection + direct aim
                 (mirrors agent1.get_custom_score heuristic without state)
  "self_play"  — same policy network for all players; call set_policy() after
                 the trainer is created
"""
from __future__ import annotations

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

# Network / observation shape constants (must match OrbitWarsEnv)
_NET_MP    = 40
_NET_MF    = 100
_STATE_DIM = 13
_ACT_DIM   = 4

_CENTER    = 50.0
_ROT_LIMIT = 50.0
_MAX_SPEED = 6.0

# Precomputed constants reused every step
_PIDX, _QIDX = np.triu_indices(4, k=1)          # 6 upper-triangle pairs for d=4
_DIAG_MASK   = ~np.eye(_NET_MP, dtype=bool)      # [NMP, NMP] — exclude self-to-self


def _fleet_speed_np(ships: np.ndarray) -> np.ndarray:
    s = np.maximum(1.0, ships)
    return 1.0 + (_MAX_SPEED - 1.0) * (np.log(s) / np.log(1000.0)) ** 1.5


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
    opponent        : "random" | "rule_based" | "self_play"
                      For "self_play", call set_policy() after trainer creation.
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
    ):
        self.num_envs        = num_envs
        self.num_players     = num_players
        self.tanh_scale      = tanh_scale
        self.min_fleet_ships = min_fleet_ships
        self.reward_type     = reward_type
        self.reward_scale    = reward_scale
        self.win_bonus       = win_bonus
        self.opponent        = opponent
        self._policy_fn: Optional[Callable] = None
        self._episode_steps  = episode_steps

        # Parse config-based reward schemes.  When provided, these take precedence
        # over reward_type/reward_scale/win_bonus and mirror the Python-backend
        # reward exactly for apples-to-apples comparisons.
        _SUPPORTED = {"AbsoluteHoldings", "FleetLaunchPenalty", "TimeDecayWinBonus"}
        self._reward_cfg: Optional[dict] = None
        if reward_cfg:
            parsed: dict = {}
            for cfg in reward_cfg:
                scheme = cfg.get("scheme", "")
                if scheme not in _SUPPORTED:
                    raise ValueError(
                        f"JAX adapter does not yet support reward scheme {scheme!r}. "
                        f"Supported: {sorted(_SUPPORTED)}"
                    )
                parsed[scheme] = {k: v for k, v in cfg.items() if k != "scheme"}
            if parsed:
                self._reward_cfg = parsed

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

        seq_len = _NET_MP + _NET_MF
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

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_policy(self, policy_fn: Callable):
        """Register the policy callable for self-play opponents.

        policy_fn(obs: ndarray[N, seq_len, STATE_DIM]) → ndarray[N, NET_MP, ACTION_DIM]
        Called once per opponent player per vector-step.
        """
        self._policy_fn = policy_fn

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

    def step(self, actions_np: np.ndarray):
        """
        Parameters
        ----------
        actions_np : float32[num_envs, NET_MP, ACTION_DIM]  — player-0 policy output

        Returns
        -------
        obs_next   : float32[num_envs, seq_len, STATE_DIM]
        rewards    : float32[num_envs]
        dones      : bool[num_envs]
        truncateds : bool[num_envs]  (always False; JAX env handles truncation internally)
        wons       : bool[num_envs]
        """
        jax_acts = self._decode_actions_batch(actions_np)
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

        # ── auto-reset done environments ───────────────────────────────────────
        done_idxs = np.where(dones_np)[0]
        if len(done_idxs) > 0:
            fresh_seeds = np.arange(self._seed_counter,
                                    self._seed_counter + len(done_idxs))
            self._seed_counter += len(done_idxs)
            states_list = [
                jax.tree_util.tree_map(lambda x, i=i: x[i], new_state)
                for i in range(self.num_envs)
            ]
            for idx, seed in zip(done_idxs, fresh_seeds):
                states_list[idx] = self._jax.reset_single(int(seed))
            new_state = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs), *states_list
            )
            if self.reward_type != "native" and self._reward_cfg is None:
                self._prev_scores = self._compute_scores(new_state)

        self._state = new_state
        obs_next   = self._extract_obs(new_state)
        truncateds = np.zeros(self.num_envs, dtype=bool)
        return obs_next, player0_rewards, dones_np, truncateds, player0_wons

    # -------------------------------------------------------------------------
    # Observation extraction — GameState → (num_envs, seq_len, STATE_DIM)
    #
    # Mirrors model.SAC.Encoder.encode layout exactly:
    #   Planet: [owner_oh×4, radius, ships, production, moving, ang_vel, 0, x, y, ts]
    #   Fleet:  [owner_oh×4, angle,  ships, speed,      0,      0,       1, x, y, ts]
    #
    # When pid != 0, owner IDs are swapped (0 ↔ pid) so the acting player
    # always appears as owner 0 — matching the Python env's _swap_perspective.
    # -------------------------------------------------------------------------

    def _extract_obs(self, state, pid: int = 0) -> np.ndarray:
        B   = self.num_envs
        NMP = _NET_MP
        NMF = _NET_MF

        p_owner_raw = np.asarray(state.planets.owner,      dtype=np.int32)
        p_x         = np.asarray(state.planets.x,          dtype=np.float32)
        p_y         = np.asarray(state.planets.y,          dtype=np.float32)
        p_ix        = np.asarray(state.planets.init_x,     dtype=np.float32)
        p_iy        = np.asarray(state.planets.init_y,     dtype=np.float32)
        p_rad       = np.asarray(state.planets.radius,     dtype=np.float32)
        p_ships     = np.asarray(state.planets.ships,      dtype=np.float32)
        p_prod      = np.asarray(state.planets.production, dtype=np.float32)
        p_act       = np.asarray(state.planets.active,     dtype=bool)
        p_comet     = np.asarray(state.planets.is_comet,   dtype=bool)

        f_owner_raw = np.asarray(state.fleets.owner,  dtype=np.int32)
        f_x         = np.asarray(state.fleets.x,      dtype=np.float32)
        f_y         = np.asarray(state.fleets.y,      dtype=np.float32)
        f_angle     = np.asarray(state.fleets.angle,  dtype=np.float32)
        f_ships     = np.asarray(state.fleets.ships,  dtype=np.float32)
        f_act       = np.asarray(state.fleets.active, dtype=bool)

        ang_vel = np.asarray(state.angular_velocity, dtype=np.float32)  # [B]
        t_step  = np.asarray(state.step,             dtype=np.float32)  # [B]

        # Perspective swap: remap owners so pid's planets appear as owner 0
        if pid != 0:
            p_owner = p_owner_raw.copy()
            p_owner[p_owner_raw == 0]   = pid
            p_owner[p_owner_raw == pid] = 0

            f_owner = f_owner_raw.copy()
            f_owner[f_owner_raw == 0]   = pid
            f_owner[f_owner_raw == pid] = 0
        else:
            p_owner = p_owner_raw
            f_owner = f_owner_raw

        # ── Planet tokens ─────────────────────────────────────────────────────
        po   = p_owner[:, :NMP]
        ocl  = np.clip(po, 0, 3)
        p_oh = np.eye(4, dtype=np.float32)[ocl]     # [B, NMP, 4]
        p_oh[po == -1] = 0.0

        dx    = p_ix[:, :NMP] - _CENTER
        dy    = p_iy[:, :NMP] - _CENTER
        p_mov = ((np.sqrt(dx**2 + dy**2) + p_rad[:, :NMP]) < _ROT_LIMIT
                 ).astype(np.float32)

        p_ang = np.repeat(ang_vel[:, None], NMP, axis=1)
        p_ts  = np.repeat(t_step[:, None],  NMP, axis=1)

        p_tokens = np.stack([
            p_oh[:, :, 0], p_oh[:, :, 1], p_oh[:, :, 2], p_oh[:, :, 3],
            p_rad[:, :NMP], p_ships[:, :NMP], p_prod[:, :NMP],
            p_mov, p_ang,
            np.zeros((B, NMP), dtype=np.float32),   # is_fleet = 0
            p_x[:, :NMP], p_y[:, :NMP], p_ts,
        ], axis=-1)  # [B, NMP, 13]

        valid_p = (p_act[:, :NMP] & ~p_comet[:, :NMP]).astype(np.float32)
        p_tokens *= valid_p[:, :, None]

        # ── Fleet tokens ──────────────────────────────────────────────────────
        sort_ships = np.where(f_act, f_ships, -1.0)
        top_idx    = np.argsort(-sort_ships, axis=1)[:, :NMF]
        bi = np.arange(B)[:, None]

        fo   = f_owner[bi, top_idx]
        fx_s = f_x    [bi, top_idx]
        fy_s = f_y    [bi, top_idx]
        fa_s = f_angle[bi, top_idx]
        fs   = f_ships[bi, top_idx]
        fv   = f_act  [bi, top_idx].astype(np.float32)

        fo_cl = np.clip(fo, 0, 3)
        f_oh  = np.eye(4, dtype=np.float32)[fo_cl]   # [B, NMF, 4]
        f_oh[fo == -1] = 0.0

        f_spd = _fleet_speed_np(fs)
        f_ts  = np.repeat(t_step[:, None], NMF, axis=1)

        f_tokens = np.stack([
            f_oh[:, :, 0], f_oh[:, :, 1], f_oh[:, :, 2], f_oh[:, :, 3],
            fa_s, fs, f_spd,
            np.zeros((B, NMF), dtype=np.float32),
            np.zeros((B, NMF), dtype=np.float32),
            np.ones ((B, NMF), dtype=np.float32),   # is_fleet = 1
            fx_s, fy_s, f_ts,
        ], axis=-1)  # [B, NMF, 13]

        f_tokens *= fv[:, :, None]

        return np.concatenate([p_tokens, f_tokens], axis=1).astype(np.float32)

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
        jax_acts[b_idx, pid, slot_idx, 1] = frac[b_idx, slot_idx]
        return int(len(b_idx))

    # -------------------------------------------------------------------------
    # Master action decode — dispatches to the chosen opponent strategy
    # -------------------------------------------------------------------------

    def _decode_actions_batch(self, actions_np: np.ndarray) -> jnp.ndarray:
        """
        Decode player-0 network output + generate opponent actions.

        actions_np : float32[B, NMP, 4]
        returns    : jnp.float32[B, num_players, JAMP, 2]
        """
        B    = self.num_envs
        NP   = self.num_players
        JAMP = _JAX_MP

        state   = self._state
        p_owner = np.asarray(state.planets.owner,      dtype=np.int32)
        p_x     = np.asarray(state.planets.x,          dtype=np.float32)
        p_y     = np.asarray(state.planets.y,          dtype=np.float32)
        p_ships = np.asarray(state.planets.ships,      dtype=np.float32)
        p_prod  = np.asarray(state.planets.production, dtype=np.float32)
        p_act   = np.asarray(state.planets.active,     dtype=bool)
        p_comet = np.asarray(state.planets.is_comet,   dtype=bool)

        jax_acts = np.zeros((B, NP, JAMP, 2), dtype=np.float32)
        valid    = p_act[:, :_NET_MP] & ~p_comet[:, :_NET_MP]   # [B, NMP]

        # ── Player 0 ───────────────────────────────────────────────────────────
        owned_p0 = (p_owner[:, :_NET_MP] == 0) & valid
        n_launched = self._decode_for_player(
            actions_np, jax_acts, 0, owned_p0, valid, p_x, p_y, p_ships
        )
        self.last_fleets_sent = n_launched

        # ── Opponents ─────────────────────────────────────────────────────────
        if NP > 1:
            if self.opponent == "self_play" and self._policy_fn is not None:
                for pid in range(1, NP):
                    opp_obs  = self._extract_obs(state, pid=pid)   # [B, seq, 13]
                    opp_acts = self._policy_fn(opp_obs)            # [B, NMP, 4]
                    owned_opp = (p_owner[:, :_NET_MP] == pid) & valid
                    self._decode_for_player(
                        opp_acts, jax_acts, pid, owned_opp, valid, p_x, p_y, p_ships
                    )
            elif self.opponent == "rule_based":
                self._decode_opponent_greedy(
                    jax_acts, valid, p_owner, p_x, p_y, p_ships, p_prod
                )
            else:
                self._decode_opponent_random_vec(
                    jax_acts, valid, p_owner, p_ships
                )

        return jnp.array(jax_acts, dtype=jnp.float32)

    # -------------------------------------------------------------------------
    # Opponent strategy implementations
    # -------------------------------------------------------------------------

    def _decode_opponent_greedy(
        self,
        jax_acts: np.ndarray,
        valid:    np.ndarray,   # [B, NMP] active non-comet
        p_owner:  np.ndarray,   # [B, JAMP]
        p_x:      np.ndarray,
        p_y:      np.ndarray,
        p_ships:  np.ndarray,
        p_prod:   np.ndarray,
    ):
        """Vectorised greedy opponent — mirrors agent1.get_custom_score heuristic.

        For each owned planet with enough ships, aims directly at the target
        that maximises (100 - dist) + 15×production + 10×production×is_enemy,
        subject to the capture-feasibility mask (ships_sent > target_ships).
        Fully vectorised over all B envs with no Python loop.
        """
        B   = self.num_envs
        NMP = _NET_MP
        bi  = np.arange(B)[:, np.newaxis]
        si  = np.arange(NMP)[np.newaxis, :]

        px = p_x[:, :NMP]       # [B, NMP]
        py = p_y[:, :NMP]
        ps = p_ships[:, :NMP]
        pp = p_prod[:, :NMP]
        po = p_owner[:, :NMP]

        # Pairwise distances [B, src, tgt]
        # dx[b, s, t] = px[b, t] - px[b, s]  (tgt minus src)
        dx   = px[:, np.newaxis, :] - px[:, :, np.newaxis]   # [B, NMP, NMP]
        dy   = py[:, np.newaxis, :] - py[:, :, np.newaxis]
        dist = np.sqrt(dx**2 + dy**2)   # [B, NMP, NMP]

        # Target heuristic value (same formula across all pids)
        is_owned = (po != -1)   # [B, NMP]  — not neutral
        score_tgt = (100.0 - dist
                     + 15.0 * pp[:, np.newaxis, :]
                     + 10.0 * pp[:, np.newaxis, :] * is_owned[:, np.newaxis, :])

        frac_val = 0.7

        for pid in range(1, self.num_players):
            owned_pid  = (po == pid) & valid   # [B, NMP]
            not_owned  = (po != pid) & valid   # [B, NMP]  — enemy or neutral

            n_ships    = frac_val * ps         # [B, NMP]  ships to send

            src_ok  = (owned_pid & (ps > 10.0))[:, :, np.newaxis]  # [B, NMP, 1]
            tgt_ok  = not_owned[:, np.newaxis, :]                   # [B, 1, NMP]
            cap_ok  = n_ships[:, :, np.newaxis] > ps[:, np.newaxis, :]  # [B, NMP, NMP]

            final = np.where(
                src_ok & tgt_ok & cap_ok & _DIAG_MASK[np.newaxis],
                score_tgt, -np.inf
            )   # [B, NMP, NMP]

            best_tgt = np.argmax(final, axis=2)         # [B, NMP]
            best_val = final[bi, si, best_tgt]           # [B, NMP]

            launch = owned_pid & (ps > 10.0) & (best_val > -np.inf)
            b_idx, slot_idx = np.where(launch)
            if len(b_idx) == 0:
                continue

            tgt_idx = best_tgt[b_idx, slot_idx]
            angles  = np.arctan2(
                py[b_idx, tgt_idx] - py[b_idx, slot_idx],
                px[b_idx, tgt_idx] - px[b_idx, slot_idx],
            )
            jax_acts[b_idx, pid, slot_idx, 0] = angles
            jax_acts[b_idx, pid, slot_idx, 1] = frac_val

    def _decode_opponent_random_vec(
        self,
        jax_acts: np.ndarray,
        valid:    np.ndarray,   # [B, NMP]
        p_owner:  np.ndarray,   # [B, JAMP]
        p_ships:  np.ndarray,
    ):
        """Vectorised random opponent — ~30% of owned planets launch a random fleet."""
        B   = self.num_envs
        NMP = _NET_MP

        for pid in range(1, self.num_players):
            owned = ((p_owner[:, :NMP] == pid)
                     & valid
                     & (p_ships[:, :NMP] > 5.0))   # [B, NMP]
            launch = owned & (np.random.random((B, NMP)) < 0.3)

            b_idx, slot_idx = np.where(launch)
            if len(b_idx) == 0:
                continue
            n = len(b_idx)
            jax_acts[b_idx, pid, slot_idx, 0] = np.random.uniform(0.0, 2 * np.pi, n)
            jax_acts[b_idx, pid, slot_idx, 1] = np.random.uniform(0.3, 0.7, n)

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
        r_jax_np: np.ndarray,   # [B, num_players]  native JAX rewards
        dones_np:  np.ndarray,   # [B] bool
    ) -> np.ndarray:
        """Compute per-env reward from config-based schemes.

        Mirrors the Python backend exactly:
          AbsoluteHoldings   — per step, uses new_state (post-step)
          FleetLaunchPenalty — uses delta(p0 active fleets) across the step
          TimeDecayWinBonus  — terminal only, linearly decays with episode length
        """
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        cfg     = self._reward_cfg

        # ── shared arrays ─────────────────────────────────────────────────────
        p_owner_new = np.asarray(new_state.planets.owner,  dtype=np.int32)
        p_act_new   = np.asarray(new_state.planets.active, dtype=bool)

        # ── AbsoluteHoldings ──────────────────────────────────────────────────
        if "AbsoluteHoldings" in cfg:
            params       = cfg["AbsoluteHoldings"]
            ship_scale   = params.get("ship_scale",   0.01)
            planet_scale = params.get("planet_scale", 1.0)

            p_ships = np.asarray(new_state.planets.ships,      dtype=np.float32)
            p_prod  = np.asarray(new_state.planets.production, dtype=np.float32)
            f_ships = np.asarray(new_state.fleets.ships,  dtype=np.float32)
            f_owner = np.asarray(new_state.fleets.owner,  dtype=np.int32)
            f_act   = np.asarray(new_state.fleets.active, dtype=bool)

            my_p      = (p_owner_new == 0) & p_act_new
            my_f      = (f_owner == 0) & f_act
            my_ships  = (p_ships * my_p).sum(axis=1) + (f_ships * my_f).sum(axis=1)
            my_prod   = (p_prod  * my_p).sum(axis=1)
            rewards  += (ship_scale * my_ships + planet_scale * my_prod).astype(np.float32)

        # ── FleetLaunchPenalty ────────────────────────────────────────────────
        if "FleetLaunchPenalty" in cfg:
            ship_scale = cfg["FleetLaunchPenalty"].get("ship_scale", 0.5)

            # Fleet counts before this step (self._state, not yet overwritten)
            pf_owner = np.asarray(self._state.fleets.owner,  dtype=np.int32)
            pf_act   = np.asarray(self._state.fleets.active, dtype=bool)
            prev_f0  = ((pf_owner == 0) & pf_act).sum(axis=1)   # [B]

            # Fleet counts after this step (new_state, before auto-reset)
            nf_owner = np.asarray(new_state.fleets.owner,  dtype=np.int32)
            nf_act   = np.asarray(new_state.fleets.active, dtype=bool)
            new_f0   = ((nf_owner == 0) & nf_act).sum(axis=1)   # [B]

            launched  = np.maximum(0, new_f0 - prev_f0)
            rewards  -= (ship_scale * launched).astype(np.float32)

        # ── TimeDecayWinBonus ─────────────────────────────────────────────────
        if "TimeDecayWinBonus" in cfg:
            win_bonus = cfg["TimeDecayWinBonus"].get("win_bonus", 100.0)
            max_s     = float(self._episode_steps)

            step_arr  = np.asarray(new_state.step, dtype=np.float32)  # [B]
            decay     = np.maximum(0.0, (max_s - step_arr) / max(1.0, max_s - 1.0))

            win_mask  = (r_jax_np[:, 0] > 0) & dones_np
            loss_mask = (r_jax_np[:, 0] < 0) & dones_np
            rewards  += (win_bonus * decay * win_mask).astype(np.float32)
            rewards  -= (win_bonus * decay * loss_mask).astype(np.float32)

        return rewards
