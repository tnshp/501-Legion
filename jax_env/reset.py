"""Reset logic: generates one GameState from a seed.

Planet/comet generation uses Python/NumPy (rejection sampling can't be JIT-compiled).
The returned GameState is a valid JAX pytree ready for vmap and jit.
"""
import math
import random

import jax.numpy as jnp
import numpy as np

from .constants import (
    BOARD_SIZE, CENTER, SUN_RADIUS, ROTATION_RADIUS_LIMIT,
    COMET_RADIUS, COMET_PRODUCTION, PLANET_CLEARANCE,
    COMET_SPAWN_STEPS, MAX_PLANETS, MAX_FLEETS,
    MAX_COMET_GROUPS, MAX_COMET_PATH_LEN,
)
from .env_types import CometData, FleetState, GameState, PlanetState

MIN_PLANET_GROUPS = 5
MAX_PLANET_GROUPS = 10
MIN_STATIC_GROUPS = 3


# ---------------------------------------------------------------------------
# Internal helpers (pure Python / NumPy)
# ---------------------------------------------------------------------------

def _dist(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _generate_planets(rng):
    """Returns list of [id, owner, x, y, radius, ships, production]."""
    planets = []
    num_q1 = rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS)
    pid = 0

    # Phase 1: guaranteed static groups (outside orbit limit)
    static_groups = 0
    for _ in range(5000):
        if static_groups >= MIN_STATIC_GROUPS:
            break
        prod = rng.randint(1, 5)
        r = 1 + math.log(prod)
        angle = rng.uniform(0, math.pi / 2)
        min_orbital = ROTATION_RADIUS_LIMIT - r
        max_orbital = (BOARD_SIZE - CENTER - r) / max(math.cos(angle), math.sin(angle))
        if min_orbital > max_orbital:
            continue
        orbital_r = rng.uniform(min_orbital, max_orbital)
        x = CENTER + orbital_r * math.cos(angle)
        y = CENTER + orbital_r * math.sin(angle)
        if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
            continue
        if (BOARD_SIZE - x) - r < 0 or (BOARD_SIZE - y) - r < 0:
            continue
        if (x - CENTER) < r + 5 or (y - CENTER) < r + 5:
            continue
        ships = min(rng.randint(5, 99), rng.randint(5, 99))
        group = [
            [pid,     -1, y,            x,            r, ships, prod],
            [pid + 1, -1, BOARD_SIZE - x, y,           r, ships, prod],
            [pid + 2, -1, x,            BOARD_SIZE - y, r, ships, prod],
            [pid + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]
        valid = all(
            _dist(tp[2], tp[3], p[2], p[3]) >= p[4] + tp[4] + PLANET_CLEARANCE
            for tp in group for p in planets
        )
        if valid:
            planets.extend(group)
            pid += 4
            static_groups += 1

    # Phase 2: orbiting / mixed groups
    attempts = 0
    has_orbiting = False
    while len(planets) < num_q1 * 4 or (not has_orbiting and attempts < 5000):
        attempts += 1
        if attempts >= 5000:
            break
        prod = rng.randint(1, 5)
        r = 1 + math.log(prod)
        x = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)
        y = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)
        orbital_r = _dist(x, y, CENTER, CENTER)
        if orbital_r < SUN_RADIUS + r + 10:
            continue
        if orbital_r + r >= ROTATION_RADIUS_LIMIT:
            if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
                continue
        ships = rng.randint(5, 30)
        group = [
            [pid,     -1, y,            x,            r, ships, prod],
            [pid + 1, -1, BOARD_SIZE - x, y,           r, ships, prod],
            [pid + 2, -1, x,            BOARD_SIZE - y, r, ships, prod],
            [pid + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]
        valid = True
        for tp in group:
            tp_orb = _dist(tp[2], tp[3], CENTER, CENTER)
            tp_rot = tp_orb + tp[4] < ROTATION_RADIUS_LIMIT
            for p in planets:
                if _dist(p[2], p[3], tp[2], tp[3]) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break
                p_orb = _dist(p[2], p[3], CENTER, CENTER)
                p_rot = p_orb + p[4] < ROTATION_RADIUS_LIMIT
                if tp_rot != p_rot:
                    if abs(tp_orb - p_orb) < tp[4] + p[4] + PLANET_CLEARANCE:
                        valid = False
                        break
            if not valid:
                break
        if valid:
            if orbital_r + r < ROTATION_RADIUS_LIMIT:
                has_orbiting = True
            planets.extend(group)
            pid += 4

    return planets


def _generate_comet_paths(initial_planets, angular_velocity, spawn_step,
                           existing_comet_ids, comet_speed, rng):
    """Returns list of 4 symmetric (x,y) paths or None on failure.

    Mirrors orbit_wars.generate_comet_paths exactly.
    """
    comet_id_set = set(existing_comet_ids)
    for _ in range(300):
        e = rng.uniform(0.75, 0.93)
        a = rng.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue
        b = a * math.sqrt(1 - e ** 2)
        c_val = a * e
        phi = rng.uniform(math.pi / 6, math.pi / 3)

        dense = []
        num = 5000
        for i in range(num):
            t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
            ex = c_val + a * math.cos(t)
            ey = b * math.sin(t)
            x = CENTER + ex * math.cos(phi) - ey * math.sin(phi)
            y = CENTER + ex * math.sin(phi) + ey * math.cos(phi)
            dense.append((x, y))

        path = [dense[0]]
        cum = 0.0
        target = comet_speed
        for i in range(1, len(dense)):
            cum += _dist(dense[i][0], dense[i][1], dense[i - 1][0], dense[i - 1][1])
            if cum >= target:
                path.append(dense[i])
                target += comet_speed

        board_start = board_end = None
        for i, (x, y) in enumerate(path):
            if 0 <= x <= BOARD_SIZE and 0 <= y <= BOARD_SIZE:
                if board_start is None:
                    board_start = i
                board_end = i
        if board_start is None:
            continue
        visible = path[board_start : board_end + 1]
        if not (5 <= len(visible) <= MAX_COMET_PATH_LEN):
            continue

        paths = [
            [[y, x] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in visible],
        ]

        static_planets, orbiting_planets = [], []
        for p in initial_planets:
            if p[0] in comet_id_set:
                continue
            orb_r = _dist(p[2], p[3], CENTER, CENTER)
            if orb_r + p[4] < ROTATION_RADIUS_LIMIT:
                orbiting_planets.append(p)
            else:
                static_planets.append(p)

        valid = True
        buf = COMET_RADIUS + 0.5
        for k, (cx, cy) in enumerate(visible):
            if _dist(cx, cy, CENTER, CENTER) < SUN_RADIUS + COMET_RADIUS:
                valid = False
                break
            sym_pts = [
                (cy, cx),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cy, BOARD_SIZE - cx),
            ]
            for p in static_planets:
                for sp in sym_pts:
                    if _dist(sp[0], sp[1], p[2], p[3]) < p[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
            game_step = spawn_step - 1 + k
            for p in orbiting_planets:
                dx = p[2] - CENTER
                dy = p[3] - CENTER
                orb_r = math.sqrt(dx ** 2 + dy ** 2)
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                px = CENTER + orb_r * math.cos(cur_angle)
                py = CENTER + orb_r * math.sin(cur_angle)
                for sp in sym_pts:
                    if _dist(sp[0], sp[1], px, py) < p[4] + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            return paths, len(visible)
    return None, 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reset(seed: int, num_players: int = 2,
          episode_steps: int = 500, ship_speed: float = 6.0,
          comet_speed: float = 4.0) -> GameState:
    """Generate one fully-initialised GameState from an integer seed.

    This runs in Python/NumPy.  Call jax.vmap-incompatible but cheap to batch
    via a Python list comprehension + jax.tree_util.tree_map(jnp.stack, *states).
    """
    rng = random.Random(seed)
    angular_velocity = rng.uniform(0.025, 0.05)

    # ---- Planets -----------------------------------------------------------
    raw_planets = _generate_planets(rng)
    n_planets = len(raw_planets)
    assert n_planets + 4 <= MAX_PLANETS, (
        f"Too many planets ({n_planets}) for MAX_PLANETS={MAX_PLANETS}"
    )

    # Home planet assignment
    num_groups = n_planets // 4
    home_group = rng.randint(0, num_groups - 1)
    base = home_group * 4
    if num_players == 2:
        raw_planets[base][1] = 0
        raw_planets[base][5] = 10
        raw_planets[base + 3][1] = 1
        raw_planets[base + 3][5] = 10
    elif num_players == 4:
        for j in range(4):
            raw_planets[base + j][1] = j
            raw_planets[base + j][5] = 10

    # Pack into arrays, padding to MAX_PLANETS
    def _pad(arr, length, fill):
        out = np.full(MAX_PLANETS, fill, dtype=arr.dtype)
        out[:length] = arr
        return out

    rp = np.array(raw_planets, dtype=np.float32)  # [n_planets, 7]
    p_x      = _pad(rp[:, 2], n_planets, 0.0)
    p_y      = _pad(rp[:, 3], n_planets, 0.0)
    p_radius = _pad(rp[:, 4], n_planets, 1.0)
    p_ships  = _pad(rp[:, 5], n_planets, 0.0)
    p_prod   = _pad(rp[:, 6].astype(np.int32), n_planets, 0)
    p_owner  = _pad(rp[:, 1].astype(np.int32), n_planets, -1)
    p_active = np.zeros(MAX_PLANETS, dtype=bool)
    p_active[:n_planets] = True
    p_is_comet = np.zeros(MAX_PLANETS, dtype=bool)

    # Comet slots: reserve n_planets .. n_planets + 20 - 1
    comet_slot_base = n_planets  # first comet slot

    # ---- Comet paths -------------------------------------------------------
    comet_paths_arr  = np.zeros((MAX_COMET_GROUPS, 4, MAX_COMET_PATH_LEN, 2), dtype=np.float32)
    comet_lengths    = np.zeros(MAX_COMET_GROUPS, dtype=np.int32)
    comet_ships_arr  = np.zeros(MAX_COMET_GROUPS, dtype=np.int32)
    comet_planet_slots = np.zeros((MAX_COMET_GROUPS, 4), dtype=np.int32)

    initial_planets = [p[:] for p in raw_planets]
    existing_comet_ids: list = []

    for g, spawn_step in enumerate(COMET_SPAWN_STEPS):
        comet_rng = random.Random(f"orbit_wars-comet-{seed}-{spawn_step}")
        paths, path_len = _generate_comet_paths(
            initial_planets, angular_velocity, spawn_step,
            existing_comet_ids, comet_speed, comet_rng
        )

        # Assign planet slots for this group
        for k in range(4):
            slot = comet_slot_base + g * 4 + k
            comet_planet_slots[g, k] = slot
            p_is_comet[slot] = True

        if paths is None:
            # Failed to generate — leave this group as all-zeros, never activated
            for k in range(4):
                slot = comet_planet_slots[g, k]
                comet_lengths[g] = 0
            continue

        comet_ships = min(
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
        )
        comet_ships_arr[g] = comet_ships
        comet_lengths[g] = path_len

        for k in range(4):
            for t, (x, y) in enumerate(paths[k]):
                if t < MAX_COMET_PATH_LEN:
                    comet_paths_arr[g, k, t, 0] = x
                    comet_paths_arr[g, k, t, 1] = y

        # Track used comet IDs for collision checks in later groups
        next_id = n_planets + g * 4  # synthetic id used only inside reset
        for k in range(4):
            existing_comet_ids.append(next_id + k)
            # Add placeholder planet to initial_planets for later comet checks
            initial_planets.append([-1, -1, -99.0, -99.0, COMET_RADIUS, comet_ships, COMET_PRODUCTION])

    # ---- Fleets (empty at reset) -------------------------------------------
    fleets = FleetState(
        x=jnp.zeros(MAX_FLEETS, dtype=jnp.float32),
        y=jnp.zeros(MAX_FLEETS, dtype=jnp.float32),
        angle=jnp.zeros(MAX_FLEETS, dtype=jnp.float32),
        ships=jnp.zeros(MAX_FLEETS, dtype=jnp.float32),
        owner=jnp.full(MAX_FLEETS, -1, dtype=jnp.int32),
        from_planet=jnp.full(MAX_FLEETS, -1, dtype=jnp.int32),
        active=jnp.zeros(MAX_FLEETS, dtype=bool),
    )

    planets = PlanetState(
        x=jnp.array(p_x),
        y=jnp.array(p_y),
        init_x=jnp.array(p_x),
        init_y=jnp.array(p_y),
        radius=jnp.array(p_radius),
        ships=jnp.array(p_ships),
        production=jnp.array(p_prod),
        owner=jnp.array(p_owner),
        active=jnp.array(p_active),
        is_comet=jnp.array(p_is_comet),
    )

    comets = CometData(
        paths=jnp.array(comet_paths_arr),
        path_lengths=jnp.array(comet_lengths),
        spawn_ships=jnp.array(comet_ships_arr),
        path_indices=jnp.full(MAX_COMET_GROUPS, -1, dtype=jnp.int32),
        planet_slots=jnp.array(comet_planet_slots),
    )

    return GameState(
        planets=planets,
        fleets=fleets,
        comets=comets,
        step=jnp.array(0, dtype=jnp.int32),
        done=jnp.array(False, dtype=bool),
        rewards=jnp.zeros(num_players, dtype=jnp.float32),
        angular_velocity=jnp.array(angular_velocity, dtype=jnp.float32),
        next_fleet_slot=jnp.array(0, dtype=jnp.int32),
        num_players=jnp.array(num_players, dtype=jnp.int32),
    )


def batch_reset(seeds, num_players: int = 2, **kwargs) -> GameState:
    """Reset N environments and stack them into a single batched GameState.

    Returns a GameState where every array has an extra leading batch dimension.
    """
    import jax
    states = [reset(int(s), num_players=num_players, **kwargs) for s in seeds]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
