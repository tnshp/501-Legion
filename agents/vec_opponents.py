"""
Vectorised heuristic opponents for the JAX Orbit Wars environment.

These are the built-in *training* opponents the env plays the learning agent
against — fully vectorised over all B parallel environments with numpy (no JAX,
no torch, no kaggle import), so they stay cheap even in the mp_env worker.

  greedy_opponent  — each owned planet (with > MIN_SRC_SHIPS ships) sends a
                     fraction of its garrison at the single best enemy/neutral
                     planet it can capture this step, chosen by a closeness +
                     production + enemy-denial score.  Aims with a LEAD-INTERCEPT
                     angle (same fixed-point solver as model/action_decoder.py),
                     so it actually hits orbiting targets instead of trailing
                     them like the old direct-atan2 aim.
  random_opponent  — each owned planet launches a random-angle fleet with a
                     fixed probability.

Both write directly into the engine action buffer
``jax_acts[b, player, planet_slot, :] = (angle, absolute_ship_count)`` in place,
filling only the opponent seats (player ≥ 1).  ``env_gate`` (used by the "mixed"
opponent) optionally restricts which (env, player) slots a strategy may fill.
"""
from __future__ import annotations

import numpy as np

# ── Engine constants (mirror jax_env.constants / env physics) ──────────────────
NET_MP                 = 40      # planets the policy / opponents act on
CENTER                 = 50.0
ROTATION_RADIUS_LIMIT  = 50.0
MAX_SHIP_SPEED         = 6.0

# Greedy heuristic weights — a simplified, vectorised form of
# agents/agent1.py get_custom_score (keeps the three positive terms, drops the
# ship-cost and travel-time penalties).
FORMULA_DIST           = 100.0   # closeness term: (FORMULA_DIST − distance)
FORMULA_PROD_MULT      = 15.0    # reward per unit of target production
FORMULA_ENEMY_BONUS    = 10.0    # extra reward per unit of ENEMY production (denial)
GREEDY_SEND_FRAC       = 0.7     # fraction of a planet's garrison sent per launch
MIN_SRC_SHIPS          = 10.0    # planets at/below this many ships sit tight

_DIAG_MASK = ~np.eye(NET_MP, dtype=bool)   # [NMP, NMP] — exclude self-to-self


# ──────────────────────────────────────────────────────────────────────────────
# Shared geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def fleet_speed(ships: np.ndarray) -> np.ndarray:
    """Board-units/tick for a fleet of `ships` (matches the engine), elementwise."""
    s = np.maximum(ships, 1.0)
    speed = 1.0 + (MAX_SHIP_SPEED - 1.0) * (np.log(s) / np.log(1000.0)) ** 1.5
    return np.minimum(speed, MAX_SHIP_SPEED)


def lead_intercept_angle(mx, my, tx, ty, tr, speed, omega, iters: int = 8):
    """Interception angle aiming at the target's *future* position (numpy).

    Identical fixed-point solver to model.action_decoder._lead_angle: a static
    target reduces to a direct atan2; an orbiting target is intercepted by
    iterating the arrival time  t ← |T(t) − source| / speed  and aiming at T(t).
    All inputs are [N] arrays for the N launching fleets.
    """
    dx = tx - CENTER
    dy = ty - CENTER
    orb = np.sqrt(dx * dx + dy * dy)
    ang0 = np.arctan2(dy, dx)
    moving = (omega != 0.0) & (orb + tr < ROTATION_RADIUS_LIMIT)
    speed_safe = np.maximum(speed, 1e-6)

    # straight-line arrival time as the initial guess (exact for static targets)
    t = np.sqrt((tx - mx) ** 2 + (ty - my) ** 2) / speed_safe
    for _ in range(iters):
        a = ang0 + omega * t
        px = np.where(moving, CENTER + orb * np.cos(a), tx)
        py = np.where(moving, CENTER + orb * np.sin(a), ty)
        t = np.sqrt((px - mx) ** 2 + (py - my) ** 2) / speed_safe

    a = ang0 + omega * t
    px = np.where(moving, CENTER + orb * np.cos(a), tx)
    py = np.where(moving, CENTER + orb * np.sin(a), ty)
    return np.arctan2(py - my, px - mx)


# ──────────────────────────────────────────────────────────────────────────────
# Greedy opponent
# ──────────────────────────────────────────────────────────────────────────────

def greedy_opponent(jax_acts, valid, p_owner, p_x, p_y, p_r, p_ships, p_prod,
                    omega, num_players, env_gate=None):
    """Fill opponent seats with greedy launches (lead-intercept aim).

    Parameters
    ----------
    jax_acts : float32[B, num_players, JAMP, 2]  — modified in place; opponent
               slots get (angle, absolute ship count).
    valid    : bool[B, NMP]   — active, non-comet planets.
    p_owner  : int  [B, JAMP] — planet owners (−1 neutral).
    p_x,p_y  : float[B, JAMP] — planet positions.
    p_r      : float[B, JAMP] — planet radii (for the orbit-motion test).
    p_ships  : float[B, JAMP] — garrisons.
    p_prod   : float[B, JAMP] — productions.
    omega    : float[B]       — per-env angular velocity (rad/tick).
    num_players : int
    env_gate : bool[B, num_players] or None — if given, player `pid` may only
               launch from env `b` when env_gate[b, pid] (used by "mixed").
    """
    B   = jax_acts.shape[0]
    NMP = NET_MP
    bi  = np.arange(B)[:, np.newaxis]
    si  = np.arange(NMP)[np.newaxis, :]

    px = p_x[:, :NMP]
    py = p_y[:, :NMP]
    pr = p_r[:, :NMP]
    ps = p_ships[:, :NMP]
    pp = p_prod[:, :NMP]
    po = p_owner[:, :NMP]

    # Pairwise distances [B, src, tgt]:  dx[b, s, t] = px[b, t] − px[b, s]
    dx   = px[:, np.newaxis, :] - px[:, :, np.newaxis]
    dy   = py[:, np.newaxis, :] - py[:, :, np.newaxis]
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # Target value: closeness + production + extra for enemy-owned production.
    is_owned  = (po != -1)
    score_tgt = (FORMULA_DIST - dist
                 + FORMULA_PROD_MULT * pp[:, np.newaxis, :]
                 + FORMULA_ENEMY_BONUS * pp[:, np.newaxis, :] * is_owned[:, np.newaxis, :])

    for pid in range(1, num_players):
        owned_pid = (po == pid) & valid                         # [B, NMP]
        not_owned = (po != pid) & valid                         # enemy or neutral
        n_ships   = GREEDY_SEND_FRAC * ps                        # ships we would send

        src_ok = (owned_pid & (ps > MIN_SRC_SHIPS))[:, :, np.newaxis]   # [B, NMP, 1]
        tgt_ok = not_owned[:, np.newaxis, :]                            # [B, 1, NMP]
        cap_ok = n_ships[:, :, np.newaxis] > ps[:, np.newaxis, :]       # capturable

        final = np.where(
            src_ok & tgt_ok & cap_ok & _DIAG_MASK[np.newaxis],
            score_tgt, -np.inf,
        )   # [B, NMP, NMP]

        best_tgt = np.argmax(final, axis=2)     # [B, NMP]
        best_val = final[bi, si, best_tgt]      # [B, NMP]

        launch = owned_pid & (ps > MIN_SRC_SHIPS) & (best_val > -np.inf)
        if env_gate is not None:
            launch = launch & env_gate[:, pid][:, np.newaxis]
        b_idx, slot_idx = np.where(launch)
        if len(b_idx) == 0:
            continue

        tgt_idx    = best_tgt[b_idx, slot_idx]
        ships_sent = np.floor(GREEDY_SEND_FRAC * ps[b_idx, slot_idx])
        speed      = fleet_speed(ships_sent)
        angles     = lead_intercept_angle(
            px[b_idx, slot_idx], py[b_idx, slot_idx],          # source
            px[b_idx, tgt_idx],  py[b_idx, tgt_idx], pr[b_idx, tgt_idx],  # target
            speed, omega[b_idx],
        )

        jax_acts[b_idx, pid, slot_idx, 0] = angles
        jax_acts[b_idx, pid, slot_idx, 1] = ships_sent          # absolute count


# ──────────────────────────────────────────────────────────────────────────────
# Random opponent
# ──────────────────────────────────────────────────────────────────────────────

def random_opponent(jax_acts, valid, p_owner, p_ships, num_players,
                    send_prob: float = 0.3, env_gate=None):
    """Each owned planet (> 5 ships) launches a random-angle fleet w.p. send_prob."""
    B   = jax_acts.shape[0]
    NMP = NET_MP

    for pid in range(1, num_players):
        owned = ((p_owner[:, :NMP] == pid) & valid & (p_ships[:, :NMP] > 5.0))
        launch = owned & (np.random.random((B, NMP)) < send_prob)
        if env_gate is not None:
            launch = launch & env_gate[:, pid][:, np.newaxis]

        b_idx, slot_idx = np.where(launch)
        if len(b_idx) == 0:
            continue
        n = len(b_idx)
        jax_acts[b_idx, pid, slot_idx, 0] = np.random.uniform(0.0, 2 * np.pi, n)
        # absolute ship count = floor(frac · garrison)
        frac = np.random.uniform(0.3, 0.7, n)
        jax_acts[b_idx, pid, slot_idx, 1] = np.floor(frac * p_ships[b_idx, slot_idx])
