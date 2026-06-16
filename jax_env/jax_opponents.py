"""On-device (JAX) heuristic opponents for the vectorised Orbit Wars env.

Single-environment ports of the NumPy opponents in ``agents/vec_opponents.py``
(``random`` / ``greedy`` / ``agent1``).  Each returns the opponent action slots
``[num_players, MAX_PLANETS, 2] = (angle, absolute_ship_count)`` with only the
opponent seats (player ≥ 1, planets 0..NET_MP-1) filled; the caller adds the
player-0 engine action and runs the engine step.  ``make_opponent_fn`` returns a
``fn(state, key, mix_ratio) -> action`` used inside the fused, vmapped+jitted
rollout step, so the opponent runs on the GPU instead of in host NumPy.

The logic mirrors the NumPy versions for the deterministic opponents (greedy /
agent1).  Where both decide to launch, the angle / ship count agree to float
precision; but because ``jnp`` and ``np`` evaluate the lead-intercept
transcendentals (arctan2 / sin / cos / sqrt) to slightly different ULPs, a
borderline launch threshold (e.g. ``total <= m_avail``) flips on a *tiny* fraction
of (env, planet) slots — a different-but-equivalent opponent, which RL training is
robust to (the NumPy version isn't bit-reproducible across BLAS versions either).
test_jax_port.py asserts this disagreement rate stays well under 1%.

Dynamic ``np.where`` scatters become fixed-shape masked writes; the per-env
``cols`` fleet restriction is dropped (illegal under jit) — the full
``[MAX_FLEETS, NET_MP]`` sweep runs on the GPU, which is the whole point.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .constants import MAX_PLANETS as _JAMP

NET_MP    = 40
CENTER    = 50.0
ROT_LIMIT = 50.0
MAX_SPEED = 6.0
SUN_RADIUS = 10.0
HORIZON   = 60.0

# greedy weights
G_DIST = 100.0
G_PROD = 15.0
G_ENEMY = 10.0
G_SEND_FRAC = 0.7
G_MIN_SRC = 10.0

# agent1 weights
A1_ENEMY_BONUS = 10.0
A1_TOTAL_SHIPS_PCT = 0.7
A1_ETA = 2.0
A1_MIN_ATTACK = 5.0
A1_PROD_HEADROOM = 3.0
A1_OWNED_FRAC = 0.75
A1_PREDICT_ITERS = 5

_NMP_RANGE = jnp.arange(NET_MP)
_DIAG = ~jnp.eye(NET_MP, dtype=bool)        # [NMP, NMP] exclude self-to-self


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers (single-env; mirror vec_opponents.py in jnp)
# ──────────────────────────────────────────────────────────────────────────────

def _fleet_speed(ships):
    s = jnp.maximum(ships, 1.0)
    return jnp.minimum(MAX_SPEED,
                       1.0 + (MAX_SPEED - 1.0) * (jnp.log(s) / jnp.log(1000.0)) ** 1.5)


def _lead_intercept(mx, my, tx, ty, tr, speed, omega, iters: int = 8):
    """(angle, t_arrive) interception aim — fixed-point solver, elementwise."""
    dx = tx - CENTER
    dy = ty - CENTER
    orb = jnp.sqrt(dx * dx + dy * dy)
    ang0 = jnp.arctan2(dy, dx)
    moving = (omega != 0.0) & (orb + tr < ROT_LIMIT)
    speed_safe = jnp.maximum(speed, 1e-6)
    t = jnp.sqrt((tx - mx) ** 2 + (ty - my) ** 2) / speed_safe
    for _ in range(iters):
        a = ang0 + omega * t
        px = jnp.where(moving, CENTER + orb * jnp.cos(a), tx)
        py = jnp.where(moving, CENTER + orb * jnp.sin(a), ty)
        t = jnp.sqrt((px - mx) ** 2 + (py - my) ** 2) / speed_safe
    a = ang0 + omega * t
    px = jnp.where(moving, CENTER + orb * jnp.cos(a), tx)
    py = jnp.where(moving, CENTER + orb * jnp.sin(a), ty)
    return jnp.arctan2(py - my, px - mx), t


def _seg_point_dist(cx, cy, ax, ay, bx, by):
    vx = bx - ax
    vy = by - ay
    wx = cx - ax
    wy = cy - ay
    vv = vx * vx + vy * vy
    t  = jnp.where(vv > 1e-9, (wx * vx + wy * vy) / jnp.where(vv > 1e-9, vv, 1.0), 0.0)
    t  = jnp.clip(t, 0.0, 1.0)
    qx = ax + t * vx
    qy = ay + t * vy
    return jnp.sqrt((cx - qx) ** 2 + (cy - qy) ** 2), t


def _sun_hit(mx, my, angle, speed, horizon: float = HORIZON):
    ex = mx + jnp.cos(angle) * speed * horizon
    ey = my + jnp.sin(angle) * speed * horizon
    d, _ = _seg_point_dist(CENTER, CENTER, mx, my, ex, ey)
    return d <= SUN_RADIUS


def _planet_arrays(state):
    pl = state.planets
    px = pl.x[:NET_MP]
    py = pl.y[:NET_MP]
    pr = pl.radius[:NET_MP]
    ps = pl.ships[:NET_MP]
    pp = pl.production[:NET_MP].astype(jnp.float32)
    po = pl.owner[:NET_MP]
    valid = pl.active[:NET_MP] & ~pl.is_comet[:NET_MP]
    return px, py, pr, ps, pp, po, valid


# ──────────────────────────────────────────────────────────────────────────────
# Greedy opponent (single env) → [NP, JAMP, 2]
# ──────────────────────────────────────────────────────────────────────────────

def _greedy(state, num_players):
    px, py, pr, ps, pp, po, valid = _planet_arrays(state)
    omega = state.angular_velocity

    dx   = px[None, :] - px[:, None]
    dy   = py[None, :] - py[:, None]
    dist = jnp.sqrt(dx * dx + dy * dy)                      # [src, tgt]
    is_owned = (po != -1)
    score_tgt = (G_DIST - dist
                 + G_PROD * pp[None, :]
                 + G_ENEMY * pp[None, :] * is_owned[None, :])

    acts = jnp.zeros((num_players, _JAMP, 2), jnp.float32)
    for pid in range(1, num_players):
        owned_pid = (po == pid) & valid
        not_owned = (po != pid) & valid
        n_ships   = G_SEND_FRAC * ps

        src_ok = (owned_pid & (ps > G_MIN_SRC))[:, None]
        tgt_ok = not_owned[None, :]
        cap_ok = n_ships[:, None] > ps[None, :]
        final  = jnp.where(src_ok & tgt_ok & cap_ok & _DIAG, score_tgt, -jnp.inf)

        best_tgt = jnp.argmax(final, axis=1)               # [src]
        best_val = final[_NMP_RANGE, best_tgt]
        launch   = owned_pid & (ps > G_MIN_SRC) & (best_val > -jnp.inf)

        ships_sent = jnp.floor(G_SEND_FRAC * ps)           # [src]
        speed      = _fleet_speed(ships_sent)
        ang, _     = _lead_intercept(px, py, px[best_tgt], py[best_tgt],
                                     pr[best_tgt], speed, omega)
        acts = acts.at[pid, :NET_MP, 0].set(jnp.where(launch, ang, 0.0))
        acts = acts.at[pid, :NET_MP, 1].set(jnp.where(launch, ships_sent, 0.0))
    return acts


# ──────────────────────────────────────────────────────────────────────────────
# Random opponent (single env) → [NP, JAMP, 2]
# ──────────────────────────────────────────────────────────────────────────────

def _random(state, key, num_players, send_prob):
    pl = state.planets
    po = pl.owner[:NET_MP]
    ps = pl.ships[:NET_MP]
    valid = pl.active[:NET_MP] & ~pl.is_comet[:NET_MP]

    acts = jnp.zeros((num_players, _JAMP, 2), jnp.float32)
    for pid in range(1, num_players):
        key, k_launch, k_ang, k_frac = jax.random.split(key, 4)
        owned  = (po == pid) & valid & (ps > 5.0)
        launch = owned & (jax.random.uniform(k_launch, (NET_MP,)) < send_prob)
        ang    = jax.random.uniform(k_ang, (NET_MP,), minval=0.0, maxval=2 * jnp.pi)
        frac   = jax.random.uniform(k_frac, (NET_MP,), minval=0.3, maxval=0.7)
        ships  = jnp.floor(frac * ps)
        acts = acts.at[pid, :NET_MP, 0].set(jnp.where(launch, ang, 0.0))
        acts = acts.at[pid, :NET_MP, 1].set(jnp.where(launch, ships, 0.0))
    return acts


# ──────────────────────────────────────────────────────────────────────────────
# agent1 opponent (single env) → [NP, JAMP, 2]
# ──────────────────────────────────────────────────────────────────────────────

def _fleet_planet_sweep(px, py, pr, f_x, f_y, f_angle, f_ships, f_active):
    """(hit, arrive) both [NF, NMP] — swept-segment fleet-vs-planet collision."""
    fsp = _fleet_speed(f_ships)
    fex = f_x + jnp.cos(f_angle) * fsp * HORIZON
    fey = f_y + jnp.sin(f_angle) * fsp * HORIZON
    d, t = _seg_point_dist(px[None, :], py[None, :],
                           f_x[:, None], f_y[:, None], fex[:, None], fey[:, None])
    hit    = (d <= pr[None, :]) & f_active[:, None]
    arrive = t * HORIZON
    return hit, arrive


def _agent1(state, num_players):
    px, py, pr, ps, pp, po, valid = _planet_arrays(state)
    omega = state.angular_velocity
    fl = state.fleets

    dx   = px[None, :] - px[:, None]
    dy   = py[None, :] - py[:, None]
    dist = jnp.sqrt(dx * dx + dy * dy)                      # [src, tgt]
    is_enemy = (po != -1)

    hit, arrive = _fleet_planet_sweep(px, py, pr, fl.x, fl.y, fl.angle,
                                      fl.ships, fl.active)  # [NF, NMP]

    acts = jnp.zeros((num_players, _JAMP, 2), jnp.float32)
    for pid in range(1, num_players):
        owned   = (po == pid) & valid
        opp_tgt = (po != pid) & valid

        enemy_f  = fl.active & (fl.owner != pid)
        friend_f = fl.active & (fl.owner == pid)
        inc      = (fl.ships[:, None] * (hit & enemy_f[:, None])).sum(0)    # [NMP]
        arr_e    = jnp.where(hit & enemy_f[:, None], arrive, jnp.inf)
        earliest = arr_e.min(0)                                             # [NMP]
        en_route = (fl.ships[:, None] * (hit & friend_f[:, None])).sum(0)   # [NMP]

        m_avail    = jnp.where(owned, jnp.maximum(0.0, ps - inc), 0.0)
        needed_now = ps + 1.0 + A1_PROD_HEADROOM * pp * is_enemy

        # ── Phase 1: defensive reinforcement ──────────────────────────────────
        earliest_f   = jnp.where(jnp.isfinite(earliest), earliest, 0.0)
        deficit      = inc - ps - pp * earliest_f
        under_attack = owned & (inc > 0.0) & (deficit > 0.0)
        need         = jnp.maximum(A1_MIN_ATTACK, jnp.ceil(deficit))        # [NMP]

        qual = (owned[:, None] & _DIAG
                & (m_avail[:, None] >= need[None, :])
                & under_attack[None, :])                                    # [src, tgt]
        d_reinf  = jnp.where(qual, dist, jnp.inf)
        best_src = jnp.argmin(d_reinf, axis=0)                              # [tgt]
        reinf    = under_attack & jnp.isfinite(d_reinf.min(axis=0))         # [tgt]

        # For each (src, tgt): does tgt pick src for reinforcement, and is the
        # resulting launch valid (ships>0, no sun)?  Among the valid ones for a
        # given src, the highest tgt index wins (matches the NumPy last-write).
        sel = reinf[None, :] & (best_src[None, :] == _NMP_RANGE[:, None])   # [src, tgt]
        sent_st = jnp.floor(jnp.minimum(m_avail[:, None], need[None, :]))   # [src, tgt]
        spd_st  = _fleet_speed(jnp.maximum(1.0, sent_st))
        ang_st, _ = _lead_intercept(px[:, None], py[:, None],
                                    px[None, :], py[None, :], pr[None, :],
                                    spd_st, omega)
        sun_st  = _sun_hit(px[:, None], py[:, None], ang_st, spd_st)
        ok_st   = sel & (sent_st > 0.0) & ~sun_st
        any_ok  = ok_st.any(axis=1)                                         # [src]
        tstar   = jnp.where(ok_st, _NMP_RANGE[None, :], -1).max(axis=1)     # [src]
        tstar_c = jnp.maximum(tstar, 0)
        reinf_ang  = ang_st[_NMP_RANGE, tstar_c]
        reinf_sent = sent_st[_NMP_RANGE, tstar_c]
        exhausted  = any_ok                                                 # [src]

        acts = acts.at[pid, :NET_MP, 0].set(jnp.where(any_ok, reinf_ang, 0.0))
        acts = acts.at[pid, :NET_MP, 1].set(jnp.where(any_ok, reinf_sent, 0.0))

        # ── Phase 2: attack (full get_custom_score) ───────────────────────────
        min_ships      = ps[None, :] + 1.0                                  # per target
        fspeed         = _fleet_speed(jnp.maximum(1.0, min_ships))
        eta            = dist / fspeed
        enemy_produced = eta * pp[None, :] * is_enemy[None, :]
        enemy_bonus    = pp[None, :] * is_enemy[None, :]
        total_ships    = min_ships + enemy_produced
        score = ((G_DIST - dist)
                 + G_PROD * pp[None, :]
                 + A1_ENEMY_BONUS * enemy_bonus
                 - A1_TOTAL_SHIPS_PCT * total_ships
                 - A1_ETA * eta)

        base       = jnp.maximum(A1_MIN_ATTACK, needed_now - en_route)      # [tgt]
        owned_cnt  = owned.sum()
        board_cnt  = valid.sum()
        overcommit = (en_route >= needed_now) & (owned_cnt < A1_OWNED_FRAC * board_cnt)

        src_ok = owned & (m_avail > A1_MIN_ATTACK) & ~exhausted
        tgt_ok = opp_tgt & ~overcommit
        cap_ok = base[None, :] <= m_avail[:, None]
        final  = jnp.where(src_ok[:, None] & tgt_ok[None, :] & cap_ok & _DIAG,
                           score, -jnp.inf)
        best_tgt = jnp.argmax(final, axis=1)                               # [src]
        best_val = final[_NMP_RANGE, best_tgt]
        launch   = src_ok & (best_val > -jnp.inf)

        tx = px[best_tgt]
        ty = py[best_tgt]
        tr = pr[best_tgt]
        prod_t = pp[best_tgt]
        en_t   = is_enemy[best_tgt].astype(jnp.float32)
        base0  = jnp.maximum(A1_MIN_ATTACK, needed_now[best_tgt] - en_route[best_tgt])

        total = base0
        for _ in range(A1_PREDICT_ITERS):
            speed = _fleet_speed(jnp.maximum(1.0, total))
            _, t_arr = _lead_intercept(px, py, tx, ty, tr, speed, omega)
            total = base0 + t_arr * prod_t * en_t
        total = jnp.floor(total)

        speed  = _fleet_speed(jnp.maximum(1.0, total))
        ang, _ = _lead_intercept(px, py, tx, ty, tr, speed, omega)
        atk_ok = launch & (total > 0.0) & (total <= m_avail) & ~_sun_hit(px, py, ang, speed)

        # Attack writes only non-exhausted sources, so it never clobbers a
        # reinforcement (src_ok already excludes exhausted).
        cur_ang = acts[pid, :NET_MP, 0]
        cur_shp = acts[pid, :NET_MP, 1]
        acts = acts.at[pid, :NET_MP, 0].set(jnp.where(atk_ok, ang, cur_ang))
        acts = acts.at[pid, :NET_MP, 1].set(jnp.where(atk_ok, total, cur_shp))
    return acts


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def make_opponent_fn(opponent: str, num_players: int, send_prob: float = 0.3):
    """Return ``fn(state, key, mix_ratio) -> [num_players, MAX_PLANETS, 2]`` (single env).

    ``key`` is a JAX PRNG key (used by random / mixed); ``mix_ratio`` is the
    per-step random-vs-greedy blend for opponent="mixed" (ignored otherwise).
    """
    NP = int(num_players)
    if opponent in ("greedy", "rule_based"):
        return lambda state, key, mix_ratio: _greedy(state, NP)
    if opponent == "agent1":
        return lambda state, key, mix_ratio: _agent1(state, NP)
    if opponent == "random":
        return lambda state, key, mix_ratio: _random(state, key, NP, send_prob)
    if opponent == "mixed":
        def _mixed(state, key, mix_ratio):
            k_coin, k_rand = jax.random.split(key)
            g = _greedy(state, NP)
            r = _random(state, k_rand, NP, send_prob)
            # per-player coin flip: with prob mix_ratio act randomly, else greedy
            use_rand = jax.random.uniform(k_coin, (NP,)) < mix_ratio   # [NP]
            return jnp.where(use_rand[:, None, None], r, g)
        return _mixed
    raise ValueError(f"make_opponent_fn: unsupported opponent {opponent!r}")
