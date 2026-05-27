"""
Orbit Wars kaggle environment utilities.

Shared by sac_train.py, sac_train_kaggle.py, and any other training scripts.
Provides observation encoding, action decoding, and reward shaping built on:
  - model/SAC.py  Encoder
  - env/training_loop.py  safe_angle / solve_intercept
"""

import math
import numpy as np

from model.SAC import Encoder
from env.training_loop import safe_angle, solve_intercept, solve_intercept_at_all_costs

MAX_PLANETS  = 44
MAX_FLEETS   = 1000
STATE_DIM    = 14
ACTION_DIM   = 8
SUN_X, SUN_Y = 50.0, 50.0


# =============================================================================
# Observation helpers
# =============================================================================

def _obs_to_arrays(obs):
    """Return (planets_np [n,7], fleets_np [m,7], comet_ids [k], omega, step)."""
    planets_np = np.array(obs.planets, dtype=np.float32)

    raw_fleets = obs.fleets if obs.fleets else []
    fleets_np  = np.array(raw_fleets, dtype=np.float32) if raw_fleets else np.empty((0, 7), dtype=np.float32)

    raw_comets = getattr(obs, "comet_planet_ids", None) or []
    comet_ids  = np.array(raw_comets, dtype=np.float32)

    omega = float(getattr(obs, "angular_velocity", 0.03))
    step  = int(getattr(obs, "step", 0))
    return planets_np, fleets_np, comet_ids, omega, step


def _swap_perspective(planets_np: np.ndarray, fleets_np: np.ndarray, player_id: int):
    """
    Relabel owner IDs so player_id appears as player 0 to the network.
    Planet/fleet IDs and positions are unchanged, so decoded moves remain valid.
    """
    if player_id == 0:
        return planets_np, fleets_np

    p = planets_np.copy()
    mask_zero = p[:, 1] == 0
    mask_pid  = p[:, 1] == player_id
    p[mask_zero, 1] = player_id
    p[mask_pid,  1] = 0

    if fleets_np.shape[0] > 0:
        f = fleets_np.copy()
        mask_fz = f[:, 1] == 0
        mask_fp = f[:, 1] == player_id
        f[mask_fz, 1] = player_id
        f[mask_fp, 1] = 0
    else:
        f = fleets_np

    return p, f


def encode_obs_as_player(
    encoder: Encoder, obs, initial_planets: np.ndarray, player_id: int = 0
) -> np.ndarray:
    """
    Encode observation from player_id's perspective.
    Owner IDs are swapped so the network always sees itself as player 0.
    Returns float32 array of shape [MAX_PLANETS + MAX_FLEETS, STATE_DIM].
    """
    planets_np, fleets_np, comet_ids, omega, step = _obs_to_arrays(obs)
    planets_np, fleets_np = _swap_perspective(planets_np, fleets_np, player_id)
    state, _ = encoder.encode(
        planets_np, fleets_np, initial_planets,
        omega, comet_ids, step, apply_padding=True,
    )
    return state.astype(np.float32)


def compute_reward_for_player(obs_prev, obs_next, player_id: int) -> float:
    """Shaped reward: change in (ships + production*10) for player_id."""
    def score(o):
        return sum(
            float(p[5]) + float(p[6]) * 10.0
            for p in o.planets if int(p[1]) == player_id
        )
    return score(obs_next) - score(obs_prev)


# =============================================================================
# Action decoder
# =============================================================================

def decode_action(
    action_np: np.ndarray,
    planets_np: np.ndarray,
    omega: float,
    player_id: int = 0,
    min_ships: int = 1,
) -> list:
    """
    Decode raw policy output into a list of kaggle moves.

      action[i, 0]  – send logit    (> 0 triggers dispatch from planet i)
      action[i, 1]  – temperature   (softplus → controls dispatch concentration;
                                     high = diffuse across many targets,
                                     low = concentrate on best target)
      action[i, 2:] – 6-dim key     (softmax of key dot-products → distribution over targets)

    Self-planet is included in the softmax without masking: the self-weight is the
    fraction of ships that stay (not dispatched). This makes the temperature the sole
    lever for dispatch volume — no separate sigmoid needed.

    Args:
        action_np  : [MAX_PLANETS, ACTION_DIM]  raw policy output
        planets_np : [n, 7]  current observation, perspective-swapped so
                     player_id's planets appear as owner 0
    Returns:
        [[planet_id, angle, ships_count], ...]
    """
    n = len(planets_np)
    if n == 0:
        return []

    act = action_np[:n]

    # Per-planet softmax temperature: softplus keeps it positive, floor at 0.1
    _x = act[:, 1]
    temp = np.maximum(np.where(_x > 20.0, _x, np.log1p(np.exp(_x))), 0.1).reshape(-1, 1)

    # Target distribution over all n planets including self.
    # Self-weight = fraction of ships retained (not dispatched).
    keys = act[:, 2:]
    scaled  = (keys @ keys.T) / temp          # [n, n], row i scaled by temp[i]
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    target_weights = np.exp(shifted)
    target_weights /= target_weights.sum(axis=1, keepdims=True)   # [n, n], rows sum to 1

    moves = []
    for i in range(n):
        pid, owner, px, py, _radius, ships, _prod = planets_np[i, :7]
        if int(owner) != 0:
            continue
        if float(ships) < min_ships:
            continue
        if float(act[i, 0]) <= 0.0:
            continue

        for j in range(n):
            if j == i:
                continue  # self-weight → ships stay on planet, nothing to emit
            num_ships = int(float(ships) * float(target_weights[i, j]))
            if num_ships < min_ships:
                continue

            _tid, _t_owner, tx, ty, t_radius, _t_ships, _t_prod = planets_np[j, :7]
            r = math.hypot(float(tx) - SUN_X, float(ty) - SUN_Y)
            is_orbiting = (r + float(t_radius)) < 50.0
            ix, iy, _, valid_path = solve_intercept(
                float(px), float(py), float(tx), float(ty),
                is_orbiting, omega, int(num_ships),
            )
            if not valid_path:
                ix, iy, _ = solve_intercept_at_all_costs(
                    float(px), float(py), float(tx), float(ty),
                    is_orbiting, omega, int(num_ships), t_radius,
                )
            angle = safe_angle(float(px), float(py), ix, iy, t_radius)
            if angle is None:
                continue
            moves.append([int(pid), angle, num_ships])

    return moves
