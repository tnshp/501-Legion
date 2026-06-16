"""
Vectorised heuristic opponents for the JAX Orbit Wars environment.

These are the NumPy *fallback / parity-oracle* opponents.  The training fast path
runs JAX ports of these on-device (see jax_env/jax_opponents.py); these NumPy
versions are used when the fast path is off (self_play, or a reward using
LaunchDistancePenalty) and as the parity reference in test_jax_port.py.  Fully
vectorised over all B parallel environments with numpy (no JAX/torch/kaggle).

  greedy_opponent  — each owned planet (with > MIN_SRC_SHIPS ships) sends a
                     fraction of its garrison at the single best enemy/neutral
                     planet it can capture this step, chosen by a closeness +
                     production + enemy-denial score.  Aims with a LEAD-INTERCEPT
                     angle (same fixed-point solver as model/action_decoder.py),
                     so it actually hits orbiting targets instead of trailing
                     them like the old direct-atan2 aim.
  agent1_opponent  — a fuller vectorised port of agents/agent1.py (the genuine
                     rule-based bot): the COMPLETE get_custom_score formula
                     (closeness + production + enemy-denial − ship-cost − travel
                     time), accurate ship counts (production headroom + en-route
                     accounting + an iterated predict_total_ships fixed point),
                     sun-collision avoidance, and a DEFENSIVE reinforcement pass
                     that reads live enemy fleets to find own planets that will
                     fall and ships reinforcements from the nearest spare planet.
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
SUN_RADIUS             = 10.0    # central sun (at CENTER,CENTER) — fleets that
                                 # cross it are destroyed
SCAN_HORIZON           = 60.0    # ticks of look-ahead (matches agent1's range(1,61))

# Greedy heuristic weights — a simplified, vectorised form of
# agents/agent1.py get_custom_score (keeps the three positive terms, drops the
# ship-cost and travel-time penalties).
FORMULA_DIST           = 100.0   # closeness term: (FORMULA_DIST − distance)
FORMULA_PROD_MULT      = 15.0    # reward per unit of target production
FORMULA_ENEMY_BONUS    = 10.0    # extra reward per unit of ENEMY production (denial)
GREEDY_SEND_FRAC       = 0.7     # fraction of a planet's garrison sent per launch
MIN_SRC_SHIPS          = 10.0    # planets at/below this many ships sit tight

# agent1 weights — the FULL get_custom_score (adds the two penalty terms the
# greedy form drops) plus its ship-count / threshold constants.
A1_ENEMY_BONUS_MULT    = 10.0    # FORMULA_ENEMY_BONUS_MULT (per unit enemy prod)
A1_TOTAL_SHIPS_PCT     = 0.7     # FORMULA_TOTAL_SHIPS_PERCENT (ship-cost penalty)
A1_ETA_PENALTY         = 2.0     # travel-time penalty weight (−2·eta)
A1_MIN_SHIPS_ATTACK    = 5.0     # MIN_SHIPS_MINE_ATTACK
A1_PROD_HEADROOM       = 3.0     # needed_now = t.ships + 1 + 3·t.production (enemy)
A1_OWNED_FRACTION      = 0.75    # skip re-attack only while we own < 75% of board
A1_PREDICT_ITERS       = 5       # predict_total_ships fixed-point iterations

_DIAG_MASK = ~np.eye(NET_MP, dtype=bool)   # [NMP, NMP] — exclude self-to-self


# ──────────────────────────────────────────────────────────────────────────────
# Shared geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def fleet_speed(ships: np.ndarray) -> np.ndarray:
    """Board-units/tick for a fleet of `ships` (matches the engine), elementwise."""
    s = np.maximum(ships, 1.0)
    speed = 1.0 + (MAX_SHIP_SPEED - 1.0) * (np.log(s) / np.log(1000.0)) ** 1.5
    return np.minimum(speed, MAX_SHIP_SPEED)


def lead_intercept(mx, my, tx, ty, tr, speed, omega, iters: int = 8):
    """Interception angle + arrival time aiming at the target's *future* position.

    Identical fixed-point solver to model.action_decoder._lead_angle: a static
    target reduces to a direct atan2; an orbiting target is intercepted by
    iterating the arrival time  t ← |T(t) − source| / speed  and aiming at T(t).
    All inputs are [N] arrays for the N launching fleets.

    Returns ``(angle, t_arrive)`` — the launch angle and the (continuous) number
    of ticks to reach the intercept point.
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
    return np.arctan2(py - my, px - mx), t


def lead_intercept_angle(mx, my, tx, ty, tr, speed, omega, iters: int = 8):
    """Interception angle only (back-compat wrapper around lead_intercept)."""
    ang, _ = lead_intercept(mx, my, tx, ty, tr, speed, omega, iters)
    return ang


def _seg_point_dist(cx, cy, ax, ay, bx, by):
    """Distance from point (cx,cy) to segment a→b, and the closest-point
    parameter t∈[0,1] along the segment.  Fully elementwise/broadcasting."""
    vx = bx - ax
    vy = by - ay
    wx = cx - ax
    wy = cy - ay
    vv = vx * vx + vy * vy
    t  = np.where(vv > 1e-9, (wx * vx + wy * vy) / np.where(vv > 1e-9, vv, 1.0), 0.0)
    t  = np.clip(t, 0.0, 1.0)
    qx = ax + t * vx
    qy = ay + t * vy
    return np.sqrt((cx - qx) ** 2 + (cy - qy) ** 2), t


def _sun_hit(mx, my, angle, speed, horizon: float = SCAN_HORIZON):
    """True where a fleet from (mx,my) at `angle`/`speed` crosses the sun within
    `horizon` ticks (mirrors agent1.sun_collision, as one swept segment)."""
    ex = mx + np.cos(angle) * speed * horizon
    ey = my + np.sin(angle) * speed * horizon
    d, _ = _seg_point_dist(CENTER, CENTER, mx, my, ex, ey)
    return d <= SUN_RADIUS


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


# ──────────────────────────────────────────────────────────────────────────────
# agent1 opponent — a fuller vectorised port of agents/agent1.py
# ──────────────────────────────────────────────────────────────────────────────

def _fleet_planet_sweep(px, py, pr, f_x, f_y, f_angle, f_ships, f_active):
    """Swept-segment collision of every fleet against every planet.

    Returns ``(hit, arrive)`` both [B, NF, NMP]:
      hit[b,f,m]    — fleet f's straight 60-tick path passes within planet m's
                      radius (and the fleet is active).
      arrive[b,f,m] — ticks until the fleet's closest approach to m.
    Mirrors agent1's per-tick path scan (range(1,61)) as one segment, which is
    exact for the engine's straight-line fleets.
    """
    fsp = fleet_speed(f_ships)                                   # [B, NF]
    fex = f_x + np.cos(f_angle) * fsp * SCAN_HORIZON
    fey = f_y + np.sin(f_angle) * fsp * SCAN_HORIZON
    d, t = _seg_point_dist(
        px[:, np.newaxis, :], py[:, np.newaxis, :],             # planet centres [B,1,NMP]
        f_x[:, :, np.newaxis], f_y[:, :, np.newaxis],           # segment start  [B,NF,1]
        fex[:, :, np.newaxis], fey[:, :, np.newaxis],           # segment end
    )
    hit    = (d <= pr[:, np.newaxis, :]) & f_active[:, :, np.newaxis]
    arrive = t * SCAN_HORIZON
    return hit, arrive


def agent1_opponent(jax_acts, valid, p_owner, p_x, p_y, p_r, p_ships, p_prod,
                    omega, num_players,
                    f_owner, f_x, f_y, f_angle, f_ships, f_active,
                    env_gate=None):
    """Fill opponent seats with a vectorised port of the agent1 rule-based bot.

    Two phases per opponent player, mirroring agents/agent1._step:
      1. DEFENCE  — detect own planets that incoming enemy fleets will capture
                    (own ships + production can't cover the inbound ships in
                    time) and ship a reinforcement from the nearest spare planet.
      2. ATTACK   — each remaining owned planet launches at the best-scoring
                    capturable target (full get_custom_score), sending exactly
                    the predicted ships needed (production headroom + en-route
                    accounting), with a lead-intercept aim and sun avoidance.

    Parameters mirror greedy_opponent, plus the live fleet arrays
    ``f_owner/f_x/f_y/f_angle/f_ships/f_active`` ([B, NF]) used for the defence
    and en-route computations.  ``jax_acts`` is modified in place.
    """
    B   = jax_acts.shape[0]
    NMP = NET_MP

    px = p_x[:, :NMP]; py = p_y[:, :NMP]; pr = p_r[:, :NMP]
    ps = p_ships[:, :NMP]; pp = p_prod[:, :NMP]; po = p_owner[:, :NMP]

    # Pairwise planet→planet distance [B, src, tgt] (position-only, pid-agnostic).
    dx   = px[:, np.newaxis, :] - px[:, :, np.newaxis]
    dy   = py[:, np.newaxis, :] - py[:, :, np.newaxis]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    diag = _DIAG_MASK[np.newaxis]
    is_enemy_owned = (po != -1)                       # [B, NMP] non-neutral planet

    # ── Fleet sweep (compute once; owner-independent) ─────────────────────────
    # Restrict to fleet columns that are active in *some* env — slots inactive
    # everywhere contribute nothing and dominate the [B,NF,NMP] memory.
    cols = np.where(f_active.any(axis=0))[0] if f_active.size else np.empty(0, int)
    if cols.size:
        fc_owner  = f_owner [:, cols]
        fc_ships  = f_ships [:, cols]
        fc_active = f_active[:, cols]
        hit, arrive = _fleet_planet_sweep(
            px, py, pr,
            f_x[:, cols], f_y[:, cols], f_angle[:, cols], fc_ships, fc_active)

    bi = np.arange(B)[:, np.newaxis]
    si = np.arange(NMP)[np.newaxis, :]

    for pid in range(1, num_players):
        owned     = (po == pid) & valid
        opp_tgt   = (po != pid) & valid                # enemy or neutral targets

        # ── Incoming enemy pressure / friendly en-route (per planet) ──────────
        if cols.size:
            enemy_f  = fc_active & (fc_owner != pid)    # [B, NFc]
            friend_f = fc_active & (fc_owner == pid)
            inc      = (fc_ships[:, :, np.newaxis]
                        * (hit & enemy_f[:, :, np.newaxis])).sum(axis=1)   # [B,NMP]
            arr_e    = np.where(hit & enemy_f[:, :, np.newaxis], arrive, np.inf)
            earliest = arr_e.min(axis=1)                                   # [B,NMP]
            en_route = (fc_ships[:, :, np.newaxis]
                        * (hit & friend_f[:, :, np.newaxis])).sum(axis=1)  # [B,NMP]
        else:
            inc      = np.zeros((B, NMP), np.float32)
            earliest = np.full((B, NMP), np.inf, np.float32)
            en_route = np.zeros((B, NMP), np.float32)

        # Ships a planet can safely commit = garrison minus inbound attackers.
        m_avail    = np.where(owned, np.maximum(0.0, ps - inc), 0.0)
        # needed_now = t.ships + 1 + 3·production (only for enemy-held targets).
        needed_now = ps + 1.0 + A1_PROD_HEADROOM * pp * is_enemy_owned

        # ── Phase 1: defensive reinforcement ──────────────────────────────────
        # A planet falls if inbound ships exceed garrison + production accrued by
        # the earliest arrival; reinforce the deficit from the nearest spare planet.
        earliest_f   = np.where(np.isfinite(earliest), earliest, 0.0)
        deficit      = inc - ps - pp * earliest_f
        under_attack = owned & (inc > 0.0) & (deficit > 0.0)
        need         = np.maximum(A1_MIN_SHIPS_ATTACK, np.ceil(deficit))   # [B,NMP]

        exhausted = np.zeros((B, NMP), bool)            # sources already committed
        qual = (owned[:, :, np.newaxis] & diag
                & (m_avail[:, :, np.newaxis] >= need[:, np.newaxis, :])
                & under_attack[:, np.newaxis, :])       # [B, src, tgt]
        d_reinf  = np.where(qual, dist, np.inf)
        best_src = np.argmin(d_reinf, axis=1)           # [B, tgt] nearest source
        reinf    = under_attack & np.isfinite(d_reinf.min(axis=1))

        b_idx, tgt_idx = np.where(reinf)
        if len(b_idx):
            src_idx = best_src[b_idx, tgt_idx]
            sent    = np.floor(np.minimum(m_avail[b_idx, src_idx],
                                          need[b_idx, tgt_idx]))
            speed   = fleet_speed(np.maximum(1.0, sent))
            ang, _  = lead_intercept(
                px[b_idx, src_idx], py[b_idx, src_idx],
                px[b_idx, tgt_idx], py[b_idx, tgt_idx], pr[b_idx, tgt_idx],
                speed, omega[b_idx])
            ok = (sent > 0.0) & ~_sun_hit(px[b_idx, src_idx], py[b_idx, src_idx],
                                          ang, speed)
            if env_gate is not None:
                ok = ok & env_gate[b_idx, pid]
            bb, ss = b_idx[ok], src_idx[ok]
            jax_acts[bb, pid, ss, 0] = ang[ok]
            jax_acts[bb, pid, ss, 1] = sent[ok]
            exhausted[bb, ss] = True                     # don't also attack from here

        # ── Phase 2: attack (full get_custom_score) ───────────────────────────
        min_ships      = ps[:, np.newaxis, :] + 1.0
        fspeed         = fleet_speed(np.maximum(1.0, min_ships))
        eta            = dist / fspeed
        en_tgt         = is_enemy_owned[:, np.newaxis, :]
        enemy_produced = eta * pp[:, np.newaxis, :] * en_tgt
        enemy_bonus    = pp[:, np.newaxis, :] * en_tgt
        total_ships    = min_ships + enemy_produced
        score = ((FORMULA_DIST - dist)
                 + FORMULA_PROD_MULT * pp[:, np.newaxis, :]
                 + A1_ENEMY_BONUS_MULT * enemy_bonus
                 - A1_TOTAL_SHIPS_PCT * total_ships
                 - A1_ETA_PENALTY * eta)

        base       = np.maximum(A1_MIN_SHIPS_ATTACK, needed_now - en_route)   # [B,tgt]
        owned_cnt  = owned.sum(axis=1, keepdims=True)
        board_cnt  = valid.sum(axis=1, keepdims=True)
        # Skip a target already covered by enough en-route ships — but only while
        # we still own < 75% of the board (mirrors agent1's late-game override).
        overcommit = (en_route >= needed_now) & (owned_cnt < A1_OWNED_FRACTION * board_cnt)

        src_ok = owned & (m_avail > A1_MIN_SHIPS_ATTACK) & ~exhausted   # [B,src]
        tgt_ok = opp_tgt & ~overcommit                                  # [B,tgt]
        cap_ok = base[:, np.newaxis, :] <= m_avail[:, :, np.newaxis]    # [B,src,tgt]

        final = np.where(
            src_ok[:, :, np.newaxis] & tgt_ok[:, np.newaxis, :] & cap_ok & diag,
            score, -np.inf,
        )
        best_tgt = np.argmax(final, axis=2)             # [B, src]
        best_val = final[bi, si, best_tgt]
        launch   = src_ok & (best_val > -np.inf)

        b_idx, src_idx = np.where(launch)
        if len(b_idx) == 0:
            continue
        tgt_idx = best_tgt[b_idx, src_idx]

        sx, sy = px[b_idx, src_idx], py[b_idx, src_idx]
        tx, ty = px[b_idx, tgt_idx], py[b_idx, tgt_idx]
        tr, om = pr[b_idx, tgt_idx], omega[b_idx]
        prod_t = pp[b_idx, tgt_idx]
        en_t   = is_enemy_owned[b_idx, tgt_idx].astype(np.float32)
        base0  = np.maximum(A1_MIN_SHIPS_ATTACK,
                            needed_now[b_idx, tgt_idx] - en_route[b_idx, tgt_idx])

        # predict_total_ships: ships grow with the production accrued over the
        # (ship-count-dependent) travel time — converge it as a fixed point.
        total = base0.copy()
        for _ in range(A1_PREDICT_ITERS):
            speed   = fleet_speed(np.maximum(1.0, total))
            _, t_arr = lead_intercept(sx, sy, tx, ty, tr, speed, om)
            total   = base0 + t_arr * prod_t * en_t
        total = np.floor(total)

        speed  = fleet_speed(np.maximum(1.0, total))
        ang, _ = lead_intercept(sx, sy, tx, ty, tr, speed, om)
        ok = (total > 0.0) & (total <= m_avail[b_idx, src_idx]) \
             & ~_sun_hit(sx, sy, ang, speed)
        if env_gate is not None:
            ok = ok & env_gate[b_idx, pid]

        bb, ss = b_idx[ok], src_idx[ok]
        jax_acts[bb, pid, ss, 0] = ang[ok]
        jax_acts[bb, pid, ss, 1] = total[ok]
