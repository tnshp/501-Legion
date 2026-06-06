"""
Gymnasium wrapper for the Kaggle Orbit Wars environment.

Observation space : Box(shape=(MAX_PLANETS + MAX_FLEETS, STATE_DIM), float32)
Action space      : Box(shape=(MAX_PLANETS, ACTION_DIM), float32)  in [-1, 1]

Per-planet action encoding  (ACTION_DIM = 4)
    Each row is a 4-dim vector in [-1, 1].  Pairwise wedge products between
    rows form an (n, n) attention score matrix that selects a target planet
    and fleet fraction for each owned source planet (see decode_action /
    pairwise_wedge).

Helper functions (imported by the self-play trainer)
    encode_obs_as_player
    decode_action
    pairwise_wedge
    compute_reward_for_player
    _obs_to_arrays
    _swap_perspective
"""

from __future__ import annotations

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

_EMPTY_FLEETS = np.empty((0, 7), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Observation parsing
# ─────────────────────────────────────────────────────────────────────────────

def _obs_to_arrays(obs):
    """
    Parse a kaggle orbit_wars observation into numpy arrays.

    Handles attribute-access (Observation object), dict-access, and kaggle's
    native list-of-lists format.

    Comet planets are stripped from `planets_np` (see below) — they never reach
    the encoder, action decoding, reward schemes, or flight-path raycasts.

    Returns
    -------
    planets_np         : [n, 7]  [id, owner, x, y, radius, ships, production]
                         (comets removed)
    fleets_np          : [m, 7]  [id, owner, x, y, angle, from_planet_id, ships]
    angular_velocity   : float
    omega              : float  (alias for angular_velocity)
    comet_ids          : np.ndarray of int  (the ids that were stripped)
    """
    def _get(obj, attr):
        return getattr(obj, attr, None) if not isinstance(obj, dict) else obj.get(attr)

    def _safe_owner(v):
        return -1 if v is None else int(v)

    def _parse_planet_row(p) -> list:
        if isinstance(p, (list, tuple)):
            return [p[0], _safe_owner(p[1]), p[2], p[3], p[4], p[5], p[6]]
        if isinstance(p, dict):
            return [p["id"], _safe_owner(p.get("owner")),
                    p["x"], p["y"], p["radius"], p["ships"], p["production"]]
        return [p.id, _safe_owner(getattr(p, "owner", -1)),
                p.x, p.y, p.radius, p.ships, p.production]

    def _parse_fleet_row(f) -> list:
        if isinstance(f, (list, tuple)):
            return [f[0], _safe_owner(f[1]), f[2], f[3], f[4], f[5], f[6]]
        if isinstance(f, dict):
            return [f["id"], _safe_owner(f.get("owner")),
                    f["x"], f["y"], f["angle"], f["from_planet_id"], f["ships"]]
        return [f.id, _safe_owner(getattr(f, "owner", -1)),
                f.x, f.y, f.angle, f.from_planet_id, f.ships]

    raw_planets = _get(obs, "planets")
    raw_fleets  = _get(obs, "fleets")
    angular_velocity = float(_get(obs, "angular_velocity") or 0.0)
    raw_comets  = _get(obs, "comet_planet_ids")

    if raw_planets is None:
        raw_planets = []
    if raw_fleets is None:
        raw_fleets = []
    if raw_comets is None:
        raw_comets = []

    # ── planets ──────────────────────────────────────────────────────────────
    if isinstance(raw_planets, np.ndarray) and raw_planets.ndim == 2:
        planets_np = raw_planets.astype(np.float32)
    else:
        rows = [_parse_planet_row(p) for p in raw_planets]
        planets_np = np.array(rows, dtype=np.float32) if rows else np.empty((0, 7), dtype=np.float32)

    # ── fleets ────────────────────────────────────────────────────────────────
    if isinstance(raw_fleets, np.ndarray) and raw_fleets.ndim == 2:
        fleets_np = raw_fleets.astype(np.float32)
    elif not raw_fleets:
        fleets_np = _EMPTY_FLEETS.copy()
    else:
        rows = [_parse_fleet_row(f) for f in raw_fleets]
        fleets_np = np.array(rows, dtype=np.float32) if rows else _EMPTY_FLEETS.copy()

    # ── comets ────────────────────────────────────────────────────────────────
    if isinstance(raw_comets, np.ndarray):
        comet_ids = raw_comets.astype(np.int32)
    else:
        comet_ids = np.array(list(raw_comets), dtype=np.int32)

    # Strip comets from the observation entirely. Comets are transient planets the
    # engine spawns at fixed steps; they add no strategic value but pollute the
    # launch-angle search (extra moving obstacles/targets) and the state. Removing
    # their rows here means they never reach the encoder, decode_action, the
    # reward schemes, or the flight-path raycasts. comet_ids is still returned for
    # signature stability but should now always describe an empty set of remaining
    # planets.
    if comet_ids.size > 0 and planets_np.shape[0] > 0:
        keep = ~np.isin(planets_np[:, 0].astype(np.int32), comet_ids)
        planets_np = planets_np[keep]

    omega = angular_velocity
    return planets_np, fleets_np, angular_velocity, omega, comet_ids


# ─────────────────────────────────────────────────────────────────────────────
# Perspective normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _swap_perspective(planets: np.ndarray, fleets: np.ndarray, player_id: int):
    """
    Re-label owners so the acting player always appears as player 0.

    player_id == 0 → identity (no-op copy).
    player_id == k → swap owner 0 ↔ k; all other owners are untouched.

    Works for any number of players (2–4).

    Returns copies of both arrays.
    """
    if player_id == 0:
        return planets.copy(), fleets.copy()

    pid = float(player_id)
    swapped_p = planets.copy()
    owner_p = swapped_p[:, 1]
    swapped_p[:, 1] = np.where(owner_p == 0.0, pid,
                      np.where(owner_p == pid,  0.0, owner_p))

    swapped_f = fleets.copy()
    if swapped_f.shape[0] > 0:
        owner_f = swapped_f[:, 1]
        swapped_f[:, 1] = np.where(owner_f == 0.0, pid,
                          np.where(owner_f == pid,  0.0, owner_f))

    return swapped_p, swapped_f


# ─────────────────────────────────────────────────────────────────────────────
# State encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_obs_as_player(encoder, obs, initial_planets: np.ndarray,
                         player_id: int = 0, time_step: int = 0) -> np.ndarray:
    """
    Encode a kaggle orbit_wars observation as [MAX_PLANETS+MAX_FLEETS, STATE_DIM].

    Parameters
    ----------
    encoder        : Encoder from model/SAC.py
    obs            : raw kaggle observation
    initial_planets: [n, 7] planets from the very first step (used for orbit detection)
    player_id      : 0 or 1 — whose perspective to use
    time_step      : current episode step

    Returns
    -------
    state : np.ndarray [MAX_PLANETS + MAX_FLEETS, STATE_DIM], dtype float32
    """
    planets_np, fleets_np, angular_velocity, _, _ = _obs_to_arrays(obs)
    s_planets, s_fleets = _swap_perspective(planets_np, fleets_np, player_id)

    state, _ = encoder.encode(
        s_planets, s_fleets, initial_planets,
        angular_velocity, time_step,
        apply_padding=True,
    )
    return state.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Action decoding
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_wedge(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Pairwise wedge (exterior) products between rows of U and V.

    For every pair (i, j) this computes the antisymmetric bivector
        wedge(U[i], V[j])[p,q] = U[i,p]*V[j,q] - U[i,q]*V[j,p]
    and returns the upper-triangle independent components.

    Parameters
    ----------
    U, V : (n, d) — action matrices (d=ACTION_DIM=4)

    Returns
    -------
    (n, n, d*(d-1)//2) — for d=4 the last axis has 6 independent components
    """
    n, d = U.shape
    # outer_UV[i,j,p,q] = U[i,p] * V[j,q]
    outer_UV = U[:, np.newaxis, :, np.newaxis] * V[np.newaxis, :, np.newaxis, :]
    # outer_VU[i,j,p,q] = V[j,p] * U[i,q]  — note j indexes V, i indexes U
    outer_VU = V[np.newaxis, :, :, np.newaxis] * U[:, np.newaxis, np.newaxis, :]
    wedge_matrices = outer_UV - outer_VU          # antisymmetric (n, n, d, d)
    row_idx, col_idx = np.triu_indices(d, k=1)   # upper-triangle pairs
    return wedge_matrices[:, :, row_idx, col_idx] # (n, n, C(d,2))


# ─────────────────────────────────────────────────────────────────────────────
# Fleet-physics helpers — exact copies of the kaggle interpreter's model so the
# launch-angle search is verified against the same collision maths the engine
# uses (kaggle_environments/envs/orbit_wars/orbit_wars.py: steps 0–4).
# ─────────────────────────────────────────────────────────────────────────────

_BOARD_SIZE            = 100.0
_CENTER                = 50.0
_SUN_RADIUS            = 10.0
_ROTATION_RADIUS_LIMIT = 50.0
_MAX_SHIP_SPEED        = 6.0
_INTERCEPT_TICKS       = 60     # lead-prediction horizon (matches agent1.py)


def _fleet_speed(ships: int) -> float:
    """Board-units travelled per tick by a fleet of `ships` (interpreter step 3)."""
    s = max(1, int(ships))
    speed = 1.0 + (_MAX_SHIP_SPEED - 1.0) * (math.log(s) / math.log(1000)) ** 1.5
    return min(speed, _MAX_SHIP_SPEED)


def _point_to_segment_distance(p, v, w) -> float:
    """Minimum distance from point p to segment v–w (interpreter copy)."""
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return math.hypot(p[0] - v[0], p[1] - v[1])
    t = max(0.0, min(1.0,
            ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    proj_x = v[0] + t * (w[0] - v[0])
    proj_y = v[1] + t * (w[1] - v[1])
    return math.hypot(p[0] - proj_x, p[1] - proj_y)


def _swept_pair_hit(A, B, P0, P1, r) -> bool:
    """True iff a fleet moving A→B and a planet moving P0→P1 come within r for
    some t in [0, 1] (interpreter copy — continuous swept-pair collision)."""
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


def compute_launch_angle(planet_by_id: dict, omega: float,
                         from_planet_id: int, to_planet_id: int,
                         num_ships: int) -> float | None:
    """
    Lead-intercept launch angle that mirrors the interpreter's fleet model.

    The engine launches a fleet from the SOURCE planet's edge (radius + 0.1
    outward), advances it `speed` units/tick along a FIXED angle, and scores a
    capture via a continuous swept collision against the TARGET planet's
    (possibly orbiting) position.  We therefore (1) predict the target's
    position at each future tick, (2) aim straight at that lead point, and
    (3) verify the straight-line flight actually connects under the exact
    swept-collision model — also rejecting paths that hit the sun or leave the
    board.  Returns None when no tick yields a verified hit, so the caller
    skips firing instead of launching a fleet that is sure to miss.

    Parameters
    ----------
    planet_by_id : {int planet_id: row[id, owner, x, y, radius, ships, prod]}
    omega        : float — system angular velocity (rad/tick)
    from_planet_id, to_planet_id : int
    num_ships    : int — fleet size (sets the speed)
    """
    if from_planet_id not in planet_by_id or to_planet_id not in planet_by_id:
        return None
    m_row = planet_by_id[from_planet_id]
    t_row = planet_by_id[to_planet_id]
    mx, my, mr = float(m_row[2]), float(m_row[3]), float(m_row[4])
    tx, ty, tr = float(t_row[2]), float(t_row[3]), float(t_row[4])
    speed = _fleet_speed(num_ships)

    # A planet orbits only when orbital_radius + radius < ROTATION_RADIUS_LIMIT;
    # otherwise it is static even though the system's omega is non-zero.
    def _kin(row):
        px, py, pr = float(row[2]), float(row[3]), float(row[4])
        orb = math.hypot(px - _CENTER, py - _CENTER)
        return (px, py, pr, orb, math.atan2(py - _CENTER, px - _CENTER),
                (omega != 0.0) and (orb + pr < _ROTATION_RADIUS_LIMIT))

    # Kinematics for every planet, in observation order — the engine awards a
    # collision to the FIRST planet a fleet's path crosses, so an intervening
    # planet must veto the shot even if the lead angle is otherwise perfect.
    kin = [(pid, _kin(row)) for pid, row in planet_by_id.items()]
    _, _, _, t_orb, t_ang, moving = _kin(t_row)

    def _pos_at(k, x, y, pr, orb, ang, mv):
        if not mv:
            return x, y
        a = ang + omega * k
        return _CENTER + orb * math.cos(a), _CENTER + orb * math.sin(a)

    def target_at(tick: int):
        if not moving:
            return tx, ty
        a = t_ang + omega * tick
        return _CENTER + t_orb * math.cos(a), _CENTER + t_orb * math.sin(a)

    def connects(angle: float, horizon: int) -> bool:
        # Launch just outside the source planet; fixed-angle straight flight.
        lx = mx + math.cos(angle) * (mr + 0.1)
        ly = my + math.sin(angle) * (mr + 0.1)
        fx_prev, fy_prev = lx, ly
        prev = [(k[1][0], k[1][1]) for k in kin]   # every planet at tick 0 (now)
        for step in range(1, horizon + 1):
            fx = lx + math.cos(angle) * speed * step
            fy = ly + math.sin(angle) * speed * step
            # Planets first, in engine order — the first one hit wins the fleet.
            for i, (pid, params) in enumerate(kin):
                px, py = _pos_at(step, *params)
                ppx, ppy = prev[i]
                if _swept_pair_hit((fx_prev, fy_prev), (fx, fy),
                                   (ppx, ppy), (px, py), params[2]):
                    return pid == to_planet_id   # hit target → good; else blocked
                prev[i] = (px, py)
            if not (0.0 <= fx <= _BOARD_SIZE and 0.0 <= fy <= _BOARD_SIZE):
                return False                    # flew off the board
            if _point_to_segment_distance((_CENTER, _CENTER),
                                          (fx_prev, fy_prev), (fx, fy)) < _SUN_RADIUS:
                return False                    # consumed by the sun
            fx_prev, fy_prev = fx, fy
        return False

    if moving:
        # Lead the target: aim at its predicted position for each arrival tick.
        # The radial bracket keeps the verified-candidate set small; connects()
        # is the real arbiter of whether that angle actually lands a hit.
        for k in range(1, _INTERCEPT_TICKS + 1):
            px, py = target_at(k)
            reach  = (mr + 0.1) + speed * k     # fleet's radial distance at tick k
            if abs(reach - math.hypot(px - mx, py - my)) > tr + speed:
                continue
            angle = math.atan2(py - my, px - mx)
            if connects(angle, _INTERCEPT_TICKS):
                return angle
        return None

    # Static target: a direct shot always reaches it — only the sun can block.
    angle   = math.atan2(ty - my, tx - mx)
    dist    = math.hypot(tx - mx, ty - my)
    horizon = min(600, int(dist / max(speed, 1e-6)) + 2)
    return angle if connects(angle, horizon) else None


def decode_action(action_np: np.ndarray, planets: np.ndarray, omega: float,
                  tanh_scale: float = 0.2, min_fleet_ships: int = 3) -> list:
    """
    Convert policy output to kaggle orbit_wars moves via pairwise bivector attention.

    Each planet's action row interacts with every other planet's row through a
    wedge product.  The resulting (n, n) score matrix selects a target planet
    and fleet fraction for each owned source planet.  `compute_launch_angle`
    then aims the fleet accounting for the target planet's orbital motion.

    Parameters
    ----------
    action_np       : (MAX_PLANETS, 4) — policy output in [-1, 1]
    planets         : (n, 7) — player-0-perspective array
                      [id, owner, x, y, radius, ships, production]
    omega           : float — angular velocity of the planet system
    tanh_scale      : float — scales the tanh input; smaller = flatter saturation
                      (e.g. 0.2 saturates at ~±10 instead of ~±3)
    min_fleet_ships : int — fleets with fewer ships than this are suppressed

    Returns
    -------
    moves : list of [planet_id (int), angle_radians (float), num_ships (int)]
    """
    planet_by_id = {int(row[0]): row for row in planets}

    # ── pairwise bivector attention ───────────────────────────────────────────
    pairwise_bivectors = pairwise_wedge(action_np, action_np)
    # tanh_scale < 1 flattens the curve so the model can express precise fractions.
    out = np.tanh(tanh_scale * pairwise_bivectors.sum(axis=-1))  # (MAX_PLANETS, MAX_PLANETS)

    owner_mask = planets[:, 1] == 0
    n_planets  = min(planets.shape[0], action_np.shape[0])
    moves      = []

    for i in range(n_planets):
        if not owner_mask[i]:
            continue
        ships = int(planets[i, 5])
        if ships <= 1:
            continue

        scores = out[i, :n_planets]           # restrict to real planets
        if scores.max() <= 0.0:
            continue

        idx            = int(np.argmax(scores))
        from_planet_id = int(planets[i, 0])
        to_planet_id   = int(planets[idx, 0])

        if from_planet_id == to_planet_id:
            continue

        frac      = float(scores[idx])
        num_ships = min(int(frac * ships), ships - 1)
        if num_ships < min_fleet_ships:
            continue

        angle_rad = compute_launch_angle(
            planet_by_id, omega, from_planet_id, to_planet_id, num_ships)
        if angle_rad is None:
            continue

        moves.append([from_planet_id, angle_rad, num_ships])

    return moves


# ─────────────────────────────────────────────────────────────────────────────
# Reward shaping — composable, single-responsibility components
# ─────────────────────────────────────────────────────────────────────────────
#
# Each class below scores ONE thing (a ship delta, a planet delta, a win bonus,
# …) and returns a float.  The training loop sums a *list* of these instances
# each step (OrbitWarsEnv.reward_scheme), so a full reward is assembled by
# composition rather than by one monolithic class.  Every component shares the
# call signature
#
#     reward = component(obs, new_obs, player_id, done, n_players, step, max_steps)
#
# where `obs`/`new_obs` are the pre/post-step observations, `done` is True on the
# terminal step, and `step`/`max_steps` give the current tick and episode limit
# (used only by the time-decaying win bonus).  `step`/`max_steps` are keyword
# defaults so the older `component(obs, new_obs, player_id=…, done=…)` call form
# still works.
#
# The legacy numbered schemes (RewardScheme1–4) are retained at the bottom as
# thin compositions of these components, so existing configs keep working.

# ── shared low-level helpers ──────────────────────────────────────────────────

def _owned_ships(planets: np.ndarray, fleets: np.ndarray, owner: int) -> float:
    """Total ships an owner controls — on planets AND in in-transit fleets, so
    launching a fleet does not create a spurious drop in the count."""
    p_mask  = planets[:, 1] == owner
    p_ships = float(planets[p_mask, 5].sum()) if p_mask.any() else 0.0
    f_ships = 0.0
    if fleets.shape[0] > 0:
        f_mask  = fleets[:, 1] == owner
        f_ships = float(fleets[f_mask, 6].sum()) if f_mask.any() else 0.0
    return p_ships + f_ships


def _planet_count(planets: np.ndarray, owner: int) -> int:
    """Number of planets owned by `owner`."""
    return int((planets[:, 1] == owner).sum())


def _owned_production(planets: np.ndarray, owner: int) -> float:
    """Total production of planets owned by `owner`."""
    mask = planets[:, 1] == owner
    return float(planets[mask, 6].sum()) if mask.any() else 0.0


def _terminal_result(planets_new: np.ndarray, fleets_new: np.ndarray,
                     player_id: int, n_players: int) -> int:
    """Win/loss/tie at game end, decided by total ships held.

    Returns +1 (player_id holds strictly more ships than every opponent),
    -1 (some opponent holds more), or 0 (tie).
    """
    opponent_ids = [p for p in range(n_players) if p != player_id]
    my       = _owned_ships(planets_new, fleets_new, player_id)
    best_opp = max((_owned_ships(planets_new, fleets_new, o) for o in opponent_ids),
                   default=0.0)
    if best_opp < my:
        return 1
    if best_opp > my:
        return -1
    return 0


def _time_decay_fraction(step: int, max_steps: int) -> float:
    """Linear decay weight in [0, 1]: 1.0 at step 1, 0.0 at step `max_steps`."""
    if max_steps <= 1:
        return 1.0
    frac = (max_steps - step) / (max_steps - 1)
    return min(1.0, max(0.0, frac))


# ── per-step shaping components ────────────────────────────────────────────────

class RelativeShipAdvantage:
    """Reward the change in ship advantage *relative to the opponents*.

        ship_scale × [ (my_ships_Δ) − Σ_opp(opp_ships_Δ) ]

    Positive when you gain ships faster than your opponents (or they lose ships
    faster than you).  Ships in flight are counted, so launching a fleet is
    reward-neutral until it actually fights.  (Ship half of the old RewardScheme1.)
    """

    def __init__(self, ship_scale: float = 0.01):
        self.ship_scale = ship_scale

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2, step: int = 0, max_steps: int = 500) -> float:
        planets_old, fleets_old, _, _, _ = _obs_to_arrays(obs)
        planets_new, fleets_new, _, _, _ = _obs_to_arrays(new_obs)
        opponent_ids = [p for p in range(n_players) if p != player_id]

        my_δ  = (_owned_ships(planets_new, fleets_new, player_id)
                 - _owned_ships(planets_old, fleets_old, player_id))
        opp_δ = sum(_owned_ships(planets_new, fleets_new, o)
                    - _owned_ships(planets_old, fleets_old, o) for o in opponent_ids)
        return float(self.ship_scale * (my_δ - opp_δ))


class RelativePlanetAdvantage:
    """Reward the change in planet-count advantage *relative to the opponents*.

        planet_scale × [ (my_planet_cnt_Δ) − Σ_opp(opp_planet_cnt_Δ) ]

    Each planet counts equally (use ProductionPlanetDelta to weight by output).
    (Planet half of the old RewardScheme1.)
    """

    def __init__(self, planet_scale: float = 1.0):
        self.planet_scale = planet_scale

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2, step: int = 0, max_steps: int = 500) -> float:
        planets_old, _, _, _, _ = _obs_to_arrays(obs)
        planets_new, _, _, _, _ = _obs_to_arrays(new_obs)
        opponent_ids = [p for p in range(n_players) if p != player_id]

        my_δ  = _planet_count(planets_new, player_id) - _planet_count(planets_old, player_id)
        opp_δ = sum(_planet_count(planets_new, o) - _planet_count(planets_old, o)
                    for o in opponent_ids)
        return float(self.planet_scale * (my_δ - opp_δ))


class ShipGrowth:
    """Reward the change in your OWN ship total, ignoring opponents.

        ship_scale × my_ships_Δ

    A pure self-improvement signal (ship half of the old RewardScheme2).
    """

    def __init__(self, ship_scale: float = 0.01):
        self.ship_scale = ship_scale

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2, step: int = 0, max_steps: int = 500) -> float:
        planets_old, fleets_old, _, _, _ = _obs_to_arrays(obs)
        planets_new, fleets_new, _, _, _ = _obs_to_arrays(new_obs)
        my_δ = (_owned_ships(planets_new, fleets_new, player_id)
                - _owned_ships(planets_old, fleets_old, player_id))
        return float(self.ship_scale * my_δ)


class ProductionPlanetDelta:
    """Reward captured planets and penalise lost ones, weighted by production.

        planet_scale × ( Σ production of planets gained this step
                         − Σ production of planets lost this step )

    Taking a high-production planet is worth more than a low-production one.
    (Planet half of the old RewardScheme2.)
    """

    def __init__(self, planet_scale: float = 1.0):
        self.planet_scale = planet_scale

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2, step: int = 0, max_steps: int = 500) -> float:
        planets_old, _, _, _, _ = _obs_to_arrays(obs)
        planets_new, _, _, _, _ = _obs_to_arrays(new_obs)

        old_owned = {int(r[0]): float(r[6]) for r in planets_old if int(r[1]) == player_id}
        new_owned = {int(r[0]): float(r[6]) for r in planets_new if int(r[1]) == player_id}
        captured = sum(prod for k, prod in new_owned.items() if k not in old_owned)
        lost     = sum(prod for k, prod in old_owned.items() if k not in new_owned)
        return float(self.planet_scale * (captured - lost))


class ProximityCaptureBonus:
    """Extra reward for capturing planets CLOSE to your existing territory.

        for each planet captured this step:
            + scale × exp(−d / ref_dist)

    where d is the distance (board units) from the captured planet to your
    NEAREST other owned planet, measured in the pre-step state — a faithful proxy
    for how far the capturing fleet had to travel.

    A capture right next to your base (small d) earns ≈ +scale; a capture flung
    across the map (large d) earns ≈ 0. This biases the agent toward expanding
    into nearby planets first instead of sending fleets far away: long-range
    sends both arrive late (the opponent grabs the easy planets meanwhile) AND
    now pay a smaller bonus, so the opening stops bleeding tempo.

    The board is 100×100; `ref_dist` sets the falloff — at d = ref_dist the bonus
    is e⁻¹ ≈ 0.37 × scale. This rewards the capture EVENT only (like
    ProductionPlanetDelta); pair it with that scheme if you also want value- /
    production-weighting, and with a win bonus for the terminal objective.

    Parameters
    ----------
    scale    : float, default 1.0  — bonus for an adjacent capture (d → 0)
    ref_dist : float, default 25.0 — distance decay constant, in board units
    """

    def __init__(self, scale: float = 1.0, ref_dist: float = 25.0):
        self.scale    = scale
        self.ref_dist = max(1e-6, ref_dist)

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2, step: int = 0, max_steps: int = 500) -> float:
        planets_old, _, _, _, _ = _obs_to_arrays(obs)
        planets_new, _, _, _, _ = _obs_to_arrays(new_obs)

        # Our planets BEFORE this step (positions) and the owner of every planet
        # before — a planet is "captured" if it is ours now but was not ours then.
        was_mine = {int(r[0]) for r in planets_old if int(r[1]) == player_id}
        my_old_xy = [(float(r[2]), float(r[3])) for r in planets_old
                     if int(r[1]) == player_id]

        total = 0.0
        for r in planets_new:
            if int(r[1]) != player_id or int(r[0]) in was_mine:
                continue                      # not a fresh capture for us
            px, py = float(r[2]), float(r[3])
            if not my_old_xy:
                closeness = 1.0               # no prior territory → treat as adjacent
            else:
                d = min(math.hypot(px - ox, py - oy) for ox, oy in my_old_xy)
                closeness = math.exp(-d / self.ref_dist)
            total += self.scale * closeness
        return float(total)


class AbsoluteHoldings:
    """Score CURRENT holdings each step (absolute, not a delta).

        ship_scale × my_ships_now + planet_scale × my_production_now

    Unlike the delta components this does NOT telescope over an episode, so the
    episode return reflects *how much was held and for how long* — holding
    high-production territory pays every step.  Keep ship_scale small relative to
    planet_scale (ship counts grow into the thousands).  (Old RewardScheme4 body.)
    """

    def __init__(self, ship_scale: float = 0.01, planet_scale: float = 1.0):
        self.ship_scale   = ship_scale
        self.planet_scale = planet_scale

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2, step: int = 0, max_steps: int = 500) -> float:
        planets_new, fleets_new, _, _, _ = _obs_to_arrays(new_obs)
        my_ships      = _owned_ships(planets_new, fleets_new, player_id)
        my_production = _owned_production(planets_new, player_id)
        return float(self.ship_scale * my_ships + self.planet_scale * my_production)


class FleetLaunchPenalty:
    """Flat penalty per fleet launched this step, discouraging fleet spam.

        −ship_scale × (number of new fleets owned by player_id)

    A fleet is "new" if its id appears in new_obs but not in obs; the penalty is
    independent of fleet size or destination.  (Old RewardScheme3.)
    """

    def __init__(self, ship_scale: float = 0.5):
        self.ship_scale = ship_scale

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2, step: int = 0, max_steps: int = 500) -> float:
        _, fleets_old, _, _, _ = _obs_to_arrays(obs)
        _, fleets_new, _, _, _ = _obs_to_arrays(new_obs)
        if fleets_new.shape[0] == 0:
            return 0.0
        old_ids = {int(r[0]) for r in fleets_old} if fleets_old.shape[0] > 0 else set()
        num_new = sum(1 for f in fleets_new
                      if int(f[0]) not in old_ids and int(f[1]) == player_id)
        return float(-self.ship_scale * num_new)


def _fleet_flight_ticks(fx0: float, fy0: float, angle: float, ships: float,
                        planets: np.ndarray, omega: float, max_ticks: int,
                        skip_id: int | None = None) -> int | None:
    """Replay a just-launched fleet's straight-line flight and return the number
    of ticks until it first crosses a planet (its target), or None if it reaches
    no planet within `max_ticks` (flies off the board, into the sun, or misses).

    Mirrors the interpreter's swept-collision model exactly — same helpers and
    constants as compute_launch_angle, including per-tick planet orbital motion.
    `skip_id` (the fleet's source planet) is excluded so a fleet launched from a
    planet's edge is never scored as targeting its own source.
    """
    speed = _fleet_speed(int(ships))

    # Per-planet kinematics at the fleet's launch instant (tick 0).
    kin = []
    for r in planets:
        if skip_id is not None and int(r[0]) == skip_id:
            continue
        px, py, pr = float(r[2]), float(r[3]), float(r[4])
        orb = math.hypot(px - _CENTER, py - _CENTER)
        ang = math.atan2(py - _CENTER, px - _CENTER)
        moving = (omega != 0.0) and (orb + pr < _ROTATION_RADIUS_LIMIT)
        kin.append((px, py, pr, orb, ang, moving))

    prev = [(k[0], k[1]) for k in kin]
    fx_prev, fy_prev = fx0, fy0
    for step in range(1, max_ticks + 1):
        fx = fx0 + math.cos(angle) * speed * step
        fy = fy0 + math.sin(angle) * speed * step
        for i, (px, py, pr, ang_orb, ang0, mv) in enumerate(kin):
            if mv:
                a = ang0 + omega * step
                cx, cy = _CENTER + ang_orb * math.cos(a), _CENTER + ang_orb * math.sin(a)
            else:
                cx, cy = px, py
            ppx, ppy = prev[i]
            if _swept_pair_hit((fx_prev, fy_prev), (fx, fy), (ppx, ppy), (cx, cy), pr):
                return step
            prev[i] = (cx, cy)
        if not (0.0 <= fx <= _BOARD_SIZE and 0.0 <= fy <= _BOARD_SIZE):
            return None                       # flew off the board
        if _point_to_segment_distance((_CENTER, _CENTER),
                                      (fx_prev, fy_prev), (fx, fy)) < _SUN_RADIUS:
            return None                       # consumed by the sun
        fx_prev, fy_prev = fx, fy
    return None


class LaunchDistancePenalty:
    """Penalise launching a fleet by how LONG it must fly to reach its target.

    For every fleet the agent launches this step (id present in new_obs, absent
    in obs, owned by player_id), the fleet's straight-line flight is replayed
    against the engine's swept-collision model to find the tick at which it first
    reaches a planet.  The penalty is

        −scale × (ticks_to_target / time_norm)   per fleet that reaches a planet
        −scale × (max_ticks       / time_norm)   per fleet that reaches none

    A fleet aimed at an adjacent planet (a handful of ticks) costs almost nothing;
    one flung across the map costs a lot; a fleet that misses everything is
    charged the full horizon.  Unlike ProximityCaptureBonus (which only pays once
    the far capture finally lands), this fires at the *moment of launch*, giving
    denser, faster credit against long-range opening sends.

    Charged by arrival TIME, not raw distance: larger fleets fly faster
    (_fleet_speed), so a big fast fleet to a mid-range planet costs less than a
    small slow one to the same planet — matching how long the agent is committed.

    NOTE: the most expensive reward component — it forward-simulates each launched
    fleet up to `max_ticks`.  Launches per step are few, but lower `max_ticks` if
    you need the speed.

    Parameters
    ----------
    scale     : float, default 1.0  — penalty magnitude per fleet at full cost
    time_norm : float, default 20.0 — flight ticks that map to one unit of cost
    max_ticks : int,   default 120  — flight-sim horizon; also the miss cost (ticks)
    """

    def __init__(self, scale: float = 1.0, time_norm: float = 20.0,
                 max_ticks: int = 120):
        self.scale     = scale
        self.time_norm = max(1e-6, time_norm)
        self.max_ticks = int(max_ticks)

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2, step: int = 0, max_steps: int = 500) -> float:
        _, fleets_old, omega_old, _, _ = _obs_to_arrays(obs)
        planets_new, fleets_new, omega_new, _, _ = _obs_to_arrays(new_obs)
        if fleets_new.shape[0] == 0:
            return 0.0

        old_ids = ({int(r[0]) for r in fleets_old}
                   if fleets_old.shape[0] > 0 else set())
        omega = omega_new if omega_new else omega_old

        total_ticks = 0.0
        for f in fleets_new:
            if int(f[0]) in old_ids or int(f[1]) != player_id:
                continue                      # not a fleet we launched this step
            fx, fy, angle, ships = float(f[2]), float(f[3]), float(f[4]), float(f[6])
            src = int(f[5])
            ticks = _fleet_flight_ticks(fx, fy, angle, ships, planets_new, omega,
                                        self.max_ticks, skip_id=src)
            total_ticks += self.max_ticks if ticks is None else ticks
        return float(-self.scale * total_ticks / self.time_norm)


class StepPenalty:
    """Constant negative reward on every timestep — a 'living cost'.

        −weight   on every step (terminal step included)

    Adds the value of TIME to the objective: since each step costs a fixed
    amount, any given capture is worth more the sooner it happens. This pushes
    the agent to grab nearby planets first (their fleets arrive sooner) and to
    finish games quickly rather than dithering. The reward is independent of the
    observations — every step is the same flat −weight.

    Pair it with a win bonus so the pressure is to win *fast*, not to end the
    game fast by any means: on its own a step penalty makes a quick loss look as
    good as a quick win. TimeDecayWinBonus complements it well (both reward
    speed, and the bonus keeps the sign of the outcome dominant).

    Parameters
    ----------
    weight : float, default 0.1 — magnitude w of the per-step penalty
    """

    def __init__(self, weight: float = 0.1):
        self.weight = weight

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2, step: int = 0, max_steps: int = 500) -> float:
        return float(-self.weight)


class TerminalWinBonus:
    """Flat ±win_bonus awarded on the terminal step (0 otherwise).

        +win_bonus on a win, −win_bonus on a loss, 0 on a tie.

    Win/loss is decided by total ships held at game end (see _terminal_result).
    This is the constant-magnitude bonus embedded in the old RewardScheme1/2/4.
    For a bonus that rewards *winning quickly*, use TimeDecayWinBonus instead.
    """

    def __init__(self, win_bonus: float = 100.0):
        self.win_bonus = win_bonus

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2, step: int = 0, max_steps: int = 500) -> float:
        if not done:
            return 0.0
        planets_new, fleets_new, _, _, _ = _obs_to_arrays(new_obs)
        return float(self.win_bonus * _terminal_result(planets_new, fleets_new,
                                                        player_id, n_players))


class TimeDecayWinBonus:
    """±win_bonus on the terminal step, scaled DOWN the later the game ends.

        bonus = result × win_bonus × (max_steps − step) / (max_steps − 1)

    where result ∈ {+1 win, −1 loss, 0 tie}.  The decay weight is 1.0 at step 1
    and falls linearly to 0.0 at step `max_steps`:

        win/lose at step 1     → full ±win_bonus
        win/lose at step 500   → 0   (with max_steps = 500)

    This rewards *winning fast* and softens *losing slowly* — a late loss is
    barely penalised, an early loss is penalised in full.  `step`/`max_steps` are
    supplied by OrbitWarsEnv.step (the current tick and the episode limit).
    """

    def __init__(self, win_bonus: float = 100.0):
        self.win_bonus = win_bonus

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2, step: int = 0, max_steps: int = 500) -> float:
        if not done:
            return 0.0
        planets_new, fleets_new, _, _, _ = _obs_to_arrays(new_obs)
        result = _terminal_result(planets_new, fleets_new, player_id, n_players)
        if result == 0:
            return 0.0
        return float(result * self.win_bonus * _time_decay_fraction(step, max_steps))


# ─────────────────────────────────────────────────────────────────────────────
# Legacy numbered schemes — kept for backward compatibility.
#
# Each is now a thin composition of the components above and reproduces the old
# behaviour exactly.  Prefer composing the named components directly in new
# configs; these remain so existing configs / checkpoints keep working.
# ─────────────────────────────────────────────────────────────────────────────

class RewardScheme1:
    """Legacy: RelativeShipAdvantage + RelativePlanetAdvantage + TerminalWinBonus."""

    def __init__(self, ship_scale: float = 0.01, planet_scale: float = 1.0,
                 win_bonus: float = 100.0):
        self.ship_scale   = ship_scale
        self.planet_scale = planet_scale
        self.win_bonus    = win_bonus
        self._parts = (RelativeShipAdvantage(ship_scale),
                       RelativePlanetAdvantage(planet_scale),
                       TerminalWinBonus(win_bonus))

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2, step: int = 0, max_steps: int = 500) -> float:
        return float(sum(p(obs, new_obs, player_id, done, n_players, step, max_steps)
                         for p in self._parts))


class RewardScheme2:
    """Legacy: ShipGrowth + ProductionPlanetDelta + TerminalWinBonus."""

    def __init__(self, ship_scale: float = 0.01, planet_scale: float = 1.0,
                 win_bonus: float = 100.0):
        self.ship_scale   = ship_scale
        self.planet_scale = planet_scale
        self.win_bonus    = win_bonus
        self._parts = (ShipGrowth(ship_scale),
                       ProductionPlanetDelta(planet_scale),
                       TerminalWinBonus(win_bonus))

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2, step: int = 0, max_steps: int = 500) -> float:
        return float(sum(p(obs, new_obs, player_id, done, n_players, step, max_steps)
                         for p in self._parts))


class RewardScheme3:
    """Legacy alias for FleetLaunchPenalty (planet_scale / max_ticks unused)."""

    def __init__(self, ship_scale: float = 0.5, planet_scale: float = 1.0,
                 max_ticks: int = 200):
        self.ship_scale   = ship_scale
        self.planet_scale = planet_scale
        self.max_ticks    = max_ticks
        self._part = FleetLaunchPenalty(ship_scale)

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2, step: int = 0, max_steps: int = 500) -> float:
        return self._part(obs, new_obs, player_id, done, n_players, step, max_steps)


class RewardScheme4:
    """Legacy: AbsoluteHoldings + TerminalWinBonus."""

    def __init__(self, ship_scale: float = 0.01, planet_scale: float = 1.0,
                 win_bonus: float = 100.0):
        self.ship_scale   = ship_scale
        self.planet_scale = planet_scale
        self.win_bonus    = win_bonus
        self._parts = (AbsoluteHoldings(ship_scale, planet_scale),
                       TerminalWinBonus(win_bonus))

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2, step: int = 0, max_steps: int = 500) -> float:
        return float(sum(p(obs, new_obs, player_id, done, n_players, step, max_steps)
                         for p in self._parts))


# ── Module-level default instances (backward compatibility) ──────────────────
reward_scheme_1 = RewardScheme1()
reward_scheme_2 = RewardScheme2()
reward_scheme_3 = RewardScheme3()
reward_scheme_4 = RewardScheme4()

# Alias used by test_orbit_wars_env.py
compute_reward_for_player = reward_scheme_1

# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium wrapper
# ─────────────────────────────────────────────────────────────────────────────

class OrbitWarsEnv(gym.Env):
    """
    Single-player gymnasium wrapper for Kaggle Orbit Wars.

    The acting agent always plays as player 0 (perspective-normalised by
    _swap_perspective). An opponent agent or string tag can be supplied.

    Class attributes (game constants)
    ----------------------------------
    MAX_PLANETS, MAX_FLEETS, STATE_DIM, ACTION_DIM

    Parameters
    ----------
    opponent      : str | callable | list
                    Opponent spec for all non-learning seats, or a list of
                    n_players-1 specs (one per seat).  Built-in strings:
                      "random"     — kaggle's built-in random agent
                      "rule_based" — agents/agent1.py RuleBasedAgent
                                     (one independent instance per seat)
                    Any callable with signature (obs, config=None) also works.
    player_id     : int (0–3)              — which seat the learning agent occupies
    n_players     : int (2–4)              — total players in the game
    encoder       : Encoder | None         — model.SAC.Encoder; created if None
    max_steps     : int                    — episode truncation limit
    reward_scheme : list[RewardSchemeN]    — ordered list of reward-scheme instances;
                    their returns are summed each step.  Hyperparameters (scales,
                    horizons, etc.) are configured on the instances themselves.
    """

    MAX_PLANETS: int = 40
    MAX_FLEETS:  int = 100
    STATE_DIM:   int = 13
    ACTION_DIM:  int = 4

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self,
                 opponent="random",
                 player_id: int = 0,
                 n_players: int = 2,
                 encoder=None,
                 max_steps: int = 500,
                 reward_scheme=None,
                 tanh_scale: float = 0.2,
                 min_fleet_ships: int = 3):
        super().__init__()

        if reward_scheme is None:
            reward_scheme = [RewardScheme1()]

        def _resolve(spec):
            if spec == "rule_based":
                from agents.agent1 import RuleBasedAgent
                return RuleBasedAgent()
            return spec

        if isinstance(opponent, list):
            opponents = [_resolve(o) for o in opponent]
        else:
            opponents = [_resolve(opponent) for _ in range(n_players - 1)]

        self.opponents        = opponents
        self.player_id        = player_id
        self.n_players        = n_players
        self.max_steps        = max_steps
        self.reward_scheme    = reward_scheme
        self.tanh_scale       = tanh_scale
        self.min_fleet_ships  = min_fleet_ships
        # Number of fleets our agent actually launched on the most recent step
        # (after decode_action's thresholds). Read by the training loop for the
        # moving-average "fleets sent" metric.
        self.last_fleets_sent = 0
        # Total fleets present in play (all players) after the most recent step —
        # this is what the encoder truncates to MAX_FLEETS, so the training loop
        # logs it to size MAX_FLEETS.
        self.last_n_fleets = 0

        if encoder is None:
            from model.SAC import Encoder
            encoder = Encoder(max_planets=self.MAX_PLANETS, max_fleets=self.MAX_FLEETS)
        self.encoder = encoder

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.MAX_PLANETS + self.MAX_FLEETS, self.STATE_DIM),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.MAX_PLANETS, self.ACTION_DIM),
            dtype=np.float32,
        )

        self._kaggle_env:   object | None       = None
        self._trainer:      object | None       = None
        self._current_obs:  object | None       = None
        self._initial_planets: np.ndarray | None = None
        self._time_step:    int                 = 0

    # ── internal ──────────────────────────────────────────────────────────────

    def _build_trainer(self):
        from kaggle_environments import make as kaggle_make
        self._kaggle_env = kaggle_make("orbit_wars", debug=False)
        agents = []
        opp_idx = 0
        for i in range(self.n_players):
            if i == self.player_id:
                agents.append(None)
            else:
                agents.append(self.opponents[opp_idx])
                opp_idx += 1
        self._trainer = self._kaggle_env.train(agents)

    def _encode(self, obs, time_step: int) -> np.ndarray:
        return encode_obs_as_player(
            self.encoder, obs, self._initial_planets,
            self.player_id, time_step,
        )

    def get_planet_counts(self) -> dict[int, int]:
        """Return planet ownership counts: {player_id: count_owned_by_player}."""
        if self._current_obs is None:
            return {}
        planets_np, _, _, _, _ = _obs_to_arrays(self._current_obs)
        counts = {}
        for pid in range(self.n_players):
            counts[pid] = int((planets_np[:, 1] == pid).sum())
        return counts

    # ── gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset stateful opponents (e.g. RuleBasedAgent) before each episode
        for opp in self.opponents:
            if hasattr(opp, "reset"):
                opp.reset()

        if self._trainer is None:
            self._build_trainer()

        raw_obs = self._trainer.reset()
        self._current_obs = raw_obs
        self._time_step   = 0

        planets_np, _, _, _, _ = _obs_to_arrays(raw_obs)
        s_planets, _ = _swap_perspective(
            planets_np, _EMPTY_FLEETS.copy(), self.player_id
        )
        self._initial_planets = s_planets

        state = self._encode(raw_obs, self._time_step)
        return state, {}

    def step(self, action: np.ndarray):
        assert self._trainer is not None, "Call reset() before step()."

        planets_np, fleets_np, omega, _, _ = _obs_to_arrays(self._current_obs)
        s_planets, _ = _swap_perspective(
            planets_np, _EMPTY_FLEETS.copy(), self.player_id
        )
        moves = decode_action(action, s_planets, omega,
                              tanh_scale=self.tanh_scale,
                              min_fleet_ships=self.min_fleet_ships)
        self.last_fleets_sent = len(moves)

        # Snapshot pre-step state as plain numpy arrays.  The kaggle environment
        # mutates obs0.planets / obs0.fleets in-place, so self._current_obs would
        # otherwise silently reflect post-step values by the time reward is computed.
        # planets_np is already comet-stripped by _obs_to_arrays.
        obs_pre = {
            "planets":          planets_np,
            "fleets":           fleets_np,
            "angular_velocity": float(omega),
        }

        raw_obs, _kaggle_reward, done, info = self._trainer.step(moves)
        self._time_step += 1

        # Elimination: in >2-player games the kaggle env keeps running while the
        # other players fight on, but once our agent owns no planets AND no fleets
        # it is out of the game and can take no meaningful action. Treat that as a
        # terminal (lost) state so the episode ends for us instead of idling.
        planets_new, fleets_new, _, _, _ = _obs_to_arrays(raw_obs)
        n_my_planets = int((planets_new[:, 1] == self.player_id).sum())
        n_my_fleets  = (int((fleets_new[:, 1] == self.player_id).sum())
                        if fleets_new.shape[0] > 0 else 0)
        eliminated   = (n_my_planets == 0) and (n_my_fleets == 0)
        # Total fleets in play (all players) — the quantity truncated to MAX_FLEETS.
        self.last_n_fleets = int(fleets_new.shape[0])

        truncated  = self._time_step >= self.max_steps
        terminated = (bool(done) or eliminated) and not truncated
        # Kaggle reward is 1 (win), -1 (loss), 0/None (mid-game).
        # bool(-1) == True in Python, so must check > 0 explicitly.
        # Elimination is always a loss, so it can never be a win.
        won = (not eliminated) and bool(done) and float(_kaggle_reward or 0) > 0

        reward = 0
        for r in self.reward_scheme:
            reward += r(
                obs_pre, raw_obs, self.player_id,
                done=terminated or truncated,
                n_players=self.n_players,
                step=self._time_step,
                max_steps=self.max_steps,
            )

        self._current_obs = raw_obs
        state = self._encode(raw_obs, self._time_step)

        return state, reward, terminated, truncated, won
    def render(self, mode: str = "human", html_path: str | None = None,
               width: int = 800, height: int = 600):
        if self._kaggle_env is None:
            return
        if html_path is not None:
            html = self._kaggle_env.render(mode="html", width=width, height=height, playing=True)
            with open(html_path, "w") as f:
                f.write(html)
            print(f"Replay saved : {html_path}")
        else:
            print(self._kaggle_env.render(mode="ansi"))

    def close(self):
        self._trainer    = None
        self._kaggle_env = None
