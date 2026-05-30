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

    Returns
    -------
    planets_np         : [n, 7]  [id, owner, x, y, radius, ships, production]
    fleets_np          : [m, 7]  [id, owner, x, y, angle, from_planet_id, ships]
    angular_velocity   : float
    omega              : float  (alias for angular_velocity)
    comet_ids          : np.ndarray of int
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
    planets_np, fleets_np, angular_velocity, _, comet_ids = _obs_to_arrays(obs)
    s_planets, s_fleets = _swap_perspective(planets_np, fleets_np, player_id)

    state, _ = encoder.encode(
        s_planets, s_fleets, initial_planets,
        angular_velocity, comet_ids, time_step,
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


def decode_action(action_np: np.ndarray, planets: np.ndarray, omega: float) -> list:
    """
    Convert policy output to kaggle orbit_wars moves via pairwise bivector attention.

    Each planet's action row interacts with every other planet's row through a
    wedge product.  The resulting (n, n) score matrix selects a target planet
    and fleet fraction for each owned source planet.  `compute_launch_angle`
    then aims the fleet accounting for the target planet's orbital motion.

    Parameters
    ----------
    action_np : (MAX_PLANETS, 4) — policy output in [-1, 1]
    planets   : (n, 7) — player-0-perspective array
                [id, owner, x, y, radius, ships, production]
    omega     : float — angular velocity of the planet system

    Returns
    -------
    moves : list of [planet_id (int), angle_radians (float), num_ships (int)]
    """
    planet_by_id = {int(row[0]): row for row in planets}

    # ── pairwise bivector attention ───────────────────────────────────────────
    pairwise_bivectors = pairwise_wedge(action_np, action_np)
    out = np.tanh(pairwise_bivectors.sum(axis=-1))   # (MAX_PLANETS, MAX_PLANETS)

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
        if num_ships <= 0:
            continue

        angle_rad = compute_launch_angle(
            planet_by_id, omega, from_planet_id, to_planet_id, num_ships)
        if angle_rad is None:
            continue

        moves.append([from_planet_id, angle_rad, num_ships])

    return moves


# ─────────────────────────────────────────────────────────────────────────────
# Reward shaping — class-based API
# ─────────────────────────────────────────────────────────────────────────────

class RewardScheme1:
    """
    Shaped reward: relative ship advantage + planet advantage + terminal bonus.

    Components
    ----------
    ship_scale   × (our_ships_Δ − sum_opp_ships_Δ)
    planet_scale × (our_planet_cnt_Δ − sum_opp_planet_cnt_Δ)
    ±100 terminal bonus on win / loss

    Parameters
    ----------
    ship_scale   : float, default 0.01
    planet_scale : float, default 1.0
    """

    def __init__(self, ship_scale: float = 0.01, planet_scale: float = 1.0, win_bonus: float = 100.0):
        self.ship_scale   = ship_scale
        self.planet_scale = planet_scale
        self.win_bonus     = win_bonus

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2) -> float:
        planets_old, fleets_old, _, _, _ = _obs_to_arrays(obs)
        planets_new, fleets_new, _, _, _ = _obs_to_arrays(new_obs)

        pid          = player_id
        opponent_ids = [p for p in range(n_players) if p != pid]

        def _ships(p, f, owner):
            # Count ships on planets AND in-transit fleets so fleet sends
            # don't create spurious negative signals.
            p_mask  = p[:, 1] == owner
            p_ships = float(p[p_mask, 5].sum()) if p_mask.any() else 0.0
            f_ships = 0.0
            if f.shape[0] > 0:
                f_mask  = f[:, 1] == owner
                f_ships = float(f[f_mask, 6].sum()) if f_mask.any() else 0.0
            return p_ships + f_ships

        def _count(p, owner):
            return int((p[:, 1] == owner).sum())

        my_ships_δ   = _ships(planets_new, fleets_new, pid) - _ships(planets_old, fleets_old, pid)
        opp_ships_δ  = sum(_ships(planets_new, fleets_new, o) - _ships(planets_old, fleets_old, o) for o in opponent_ids)
        ship_delta   = my_ships_δ - opp_ships_δ

        my_cnt_δ     = _count(planets_new, pid) - _count(planets_old, pid)
        opp_cnt_δ    = sum(_count(planets_new, o) - _count(planets_old, o) for o in opponent_ids)
        planet_delta = my_cnt_δ - opp_cnt_δ

        terminal_bonus = 0.0
        if done:
            my_ships_final = _ships(planets_new, fleets_new, pid)
            best_opp       = max((_ships(planets_new, fleets_new, o) for o in opponent_ids), default=0.0)
            if best_opp < my_ships_final:
                terminal_bonus =  self.win_bonus
            elif best_opp > my_ships_final:
                terminal_bonus = -self.win_bonus

        return float(self.ship_scale * ship_delta + self.planet_scale * planet_delta + terminal_bonus)


class RewardScheme2:
    """
    Shaped reward: own ship delta + own planet delta + terminal bonus.
    Ignores opponent metrics — pure self-improvement signal.

    Parameters
    ----------
    ship_scale   : float, default 0.01
    planet_scale : float, default 1.0
    """

    def __init__(self, ship_scale: float = 0.01, planet_scale: float = 1.0, win_bonus: float = 100.0):
        self.ship_scale   = ship_scale
        self.planet_scale = planet_scale
        self.win_bonus     = win_bonus

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2) -> float:
        planets_old, fleets_old, _, _, _ = _obs_to_arrays(obs)
        planets_new, fleets_new, _, _, _ = _obs_to_arrays(new_obs)

        pid          = player_id
        opponent_ids = [p for p in range(n_players) if p != pid]

        def _ships(p, f, owner):
            # Count ships on planets AND in-transit fleets so fleet sends
            # don't create spurious negative signals.
            p_mask  = p[:, 1] == owner
            p_ships = float(p[p_mask, 5].sum()) if p_mask.any() else 0.0
            f_ships = 0.0
            if f.shape[0] > 0:
                f_mask  = f[:, 1] == owner
                f_ships = float(f[f_mask, 6].sum()) if f_mask.any() else 0.0
            return p_ships + f_ships

        def _count(p, owner):
            return int((p[:, 1] == owner).sum())

        my_ships_δ = _ships(planets_new, fleets_new, pid) - _ships(planets_old, fleets_old, pid)
        my_cnt_δ   = _count(planets_new, pid) - _count(planets_old, pid)

        terminal_bonus = 0.0
        if done:
            my_ships_final = _ships(planets_new, fleets_new, pid)
            best_opp       = max((_ships(planets_new, fleets_new, o) for o in opponent_ids), default=0.0)
            if best_opp < my_ships_final:
                terminal_bonus =  self.win_bonus
            elif best_opp > my_ships_final:
                terminal_bonus = -self.win_bonus

        return float(self.ship_scale * my_ships_δ + self.planet_scale * my_cnt_δ + terminal_bonus)


class RewardScheme3:
    """
    Penalises the agent for sending fleets, discouraging spam.

    Each new fleet launched this step (present in new_obs but not in obs) that
    is owned by player_id incurs a flat penalty of -ship_scale, regardless of
    where the fleet is headed or how many ships it carries.

    Total penalty = -ship_scale * num_new_fleets_sent

    Parameters
    ----------
    ship_scale   : float, default 0.5 — penalty per fleet launched
    planet_scale : float, default 1.0 — unused, kept for API symmetry
    max_ticks    : int,   default 200 — unused, kept for API symmetry
    """

    def __init__(self, ship_scale: float = 0.5, planet_scale: float = 1.0,
                 max_ticks: int = 200):
        self.ship_scale   = ship_scale
        self.planet_scale = planet_scale
        self.max_ticks    = max_ticks

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2) -> float:
        _, fleets_old, _, _, _ = _obs_to_arrays(obs)
        _, fleets_new, _, _, _ = _obs_to_arrays(new_obs)

        if fleets_new.shape[0] == 0:
            return 0.0

        old_ids = {int(r[0]) for r in fleets_old} if fleets_old.shape[0] > 0 else set()

        num_new = sum(
            1 for f in fleets_new
            if int(f[0]) not in old_ids and int(f[1]) == player_id
        )

        return -self.ship_scale * num_new


class RewardScheme4:
    """
    State-based (absolute) reward — scores the player's CURRENT holdings each
    step rather than the change since the previous step.

    Unlike the delta schemes (1/2), this does NOT telescope over an episode:
    Σ_t reward_t = Σ_t (ship_scale·ships_t + planet_scale·planets_t) + bonus,
    so the episode total reflects *how much was held and for how long*.
    Capturing and *holding* territory yields a sustained positive signal;
    losing planets immediately lowers every subsequent step's reward.

        ship_scale   × my_ships_now
      + planet_scale × my_planet_count_now
      ± win_bonus    (terminal, on win/loss)

    Scaling note
    ------------
    my_ships grows into the hundreds/thousands via production, so keep
    ship_scale small relative to planet_scale or the ship term dominates and
    the agent is rewarded for hoarding ships rather than taking planets.

    Parameters
    ----------
    ship_scale   : float, default 0.01
    planet_scale : float, default 1.0
    win_bonus    : float, default 100.0
    """

    def __init__(self, ship_scale: float = 0.01, planet_scale: float = 1.0, win_bonus: float = 100.0):
        self.ship_scale   = ship_scale
        self.planet_scale = planet_scale
        self.win_bonus     = win_bonus

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2) -> float:
        # Only the post-step state matters for an absolute reward; obs unused.
        planets_new, fleets_new, _, _, _ = _obs_to_arrays(new_obs)

        pid          = player_id
        opponent_ids = [p for p in range(n_players) if p != pid]

        def _ships(p, f, owner):
            p_mask  = p[:, 1] == owner
            p_ships = float(p[p_mask, 5].sum()) if p_mask.any() else 0.0
            f_ships = 0.0
            if f.shape[0] > 0:
                f_mask  = f[:, 1] == owner
                f_ships = float(f[f_mask, 6].sum()) if f_mask.any() else 0.0
            return p_ships + f_ships

        def _count(p, owner):
            return int((p[:, 1] == owner).sum())

        my_ships = _ships(planets_new, fleets_new, pid)
        my_cnt   = _count(planets_new, pid)

        terminal_bonus = 0.0
        if done:
            best_opp = max((_ships(planets_new, fleets_new, o) for o in opponent_ids), default=0.0)
            if best_opp < my_ships:
                terminal_bonus =  self.win_bonus
            elif best_opp > my_ships:
                terminal_bonus = -self.win_bonus

        return float(self.ship_scale * my_ships + self.planet_scale * my_cnt + terminal_bonus)


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
    STATE_DIM:   int = 14
    ACTION_DIM:  int = 4

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self,
                 opponent="random",
                 player_id: int = 0,
                 n_players: int = 2,
                 encoder=None,
                 max_steps: int = 500,
                 reward_scheme=None):
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

        self.opponents     = opponents
        self.player_id     = player_id
        self.n_players     = n_players
        self.max_steps     = max_steps
        self.reward_scheme = reward_scheme

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

        planets_np, fleets_np, omega, _, comet_ids_np = _obs_to_arrays(self._current_obs)
        s_planets, _ = _swap_perspective(
            planets_np, _EMPTY_FLEETS.copy(), self.player_id
        )
        moves = decode_action(action, s_planets, omega)

        # Snapshot pre-step state as plain numpy arrays.  The kaggle environment
        # mutates obs0.planets / obs0.fleets in-place, so self._current_obs would
        # otherwise silently reflect post-step values by the time reward is computed.
        obs_pre = {
            "planets":          planets_np,
            "fleets":           fleets_np,
            "angular_velocity": float(omega),
            "comet_planet_ids": comet_ids_np,
        }

        raw_obs, _kaggle_reward, done, info = self._trainer.step(moves)
        self._time_step += 1

        truncated  = self._time_step >= self.max_steps
        terminated = bool(done) and not truncated
        won = _kaggle_reward

        reward = 0
        for r in self.reward_scheme:
            reward += r(
                obs_pre, raw_obs, self.player_id,
                done=terminated or truncated,
                n_players=self.n_players,
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
