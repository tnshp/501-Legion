"""Pure-JAX game step.  JIT-compilable and vmappable.

Usage:
    from .step import step
    import functools, jax

    # Compile once (num_players is a static arg so JAX can unroll player loops)
    step_jit = jax.jit(functools.partial(step, num_players=2,
                                          episode_steps=500, ship_speed=6.0,
                                          comet_speed=4.0))
    # Vectorise over a batch of environments
    step_vmap = jax.vmap(step_jit)

    new_state = step_jit(state, actions)          # single env
    new_states = step_vmap(batched_state, batched_actions)  # N envs

Actions format:  float32[num_players, MAX_PLANETS, 2]
    actions[player, planet_slot, 0] = angle (radians)
    actions[player, planet_slot, 1] = ship fraction (0 = no launch)
"""
import functools

import jax
import jax.numpy as jnp

from .constants import (
    BOARD_SIZE, CENTER, SUN_RADIUS, ROTATION_RADIUS_LIMIT,
    MAX_PLANETS, MAX_FLEETS, MAX_COMET_GROUPS, MAX_COMET_PATH_LEN,
    COMET_RADIUS, COMET_PRODUCTION, COMET_SPAWN_STEPS,
)
from .math_utils import (
    dist, dist2,
    swept_pair_hit_time,
    point_to_segment_dist_sq,
    fleet_speed,
)
from .env_types import CometData, FleetState, GameState, PlanetState

_SPAWN_STEPS = jnp.array(COMET_SPAWN_STEPS, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Sub-steps
# ---------------------------------------------------------------------------

def _update_planet_positions(planets: PlanetState, angular_velocity, step):
    """Return new (x, y) for all planets after one tick.

    Orbiting planets: orbital_r + radius < ROTATION_RADIUS_LIMIT
    Static planets: unchanged.
    Comet planets: handled separately; their positions come from precomputed paths.
    """
    dx = planets.init_x - CENTER
    dy = planets.init_y - CENTER
    orbital_r = jnp.sqrt(dx ** 2 + dy ** 2)
    is_orbiting = (
        (orbital_r + planets.radius < ROTATION_RADIUS_LIMIT)
        & planets.active
        & ~planets.is_comet
    )
    init_angle = jnp.arctan2(dy, dx)
    new_angle = init_angle + angular_velocity * step.astype(jnp.float32)
    new_x = jnp.where(is_orbiting, CENTER + orbital_r * jnp.cos(new_angle), planets.x)
    new_y = jnp.where(is_orbiting, CENTER + orbital_r * jnp.sin(new_angle), planets.y)
    return new_x, new_y


def _advance_comets(planets: PlanetState, comets: CometData, game_step):
    """Advance comet positions by one tick.  Handles spawn, movement, expiry.

    Returns (new_planets, new_comets) where comet planet positions are updated
    and expired comets are deactivated.
    """
    # Which groups spawn this tick?  (step + 1) ∈ COMET_SPAWN_STEPS
    spawning = (game_step + 1) == _SPAWN_STEPS  # [5] bool

    # Which groups are already active?
    active = comets.path_indices >= 0           # [5] bool

    # New path indices
    # Spawning → 0; active → +1; inactive and not spawning → unchanged
    new_path_indices = jnp.where(
        spawning, 0,
        jnp.where(active, comets.path_indices + 1, comets.path_indices)
    )  # [5]

    # Compute new comet positions (for groups that are active or spawning)
    # paths: [5, 4, MAX_COMET_PATH_LEN, 2]
    path_idx_clipped = jnp.clip(new_path_indices, 0, MAX_COMET_PATH_LEN - 1)  # [5]

    # Gather positions: for each group g, each copy k → paths[g, k, idx[g], :]
    # Shape after vmap: [5, 4, 2]
    def _group_positions(g_paths, idx):
        return g_paths[:, idx, :]  # [4, 2]

    new_comet_pos = jax.vmap(_group_positions)(comets.paths, path_idx_clipped)  # [5,4,2]

    # Determine which groups to update
    is_active_or_spawning = (active | spawning)  # [5]

    # Expiry: path index has gone past the end of the valid path
    is_expired = active & (new_path_indices >= comets.path_lengths)  # [5]

    # --- Scatter comet positions into planet arrays ---
    # planet_slots: [5, 4], flat → [20]
    flat_slots = comets.planet_slots.reshape(-1)           # [20]
    flat_new_x = new_comet_pos[:, :, 0].reshape(-1)        # [20]
    flat_new_y = new_comet_pos[:, :, 1].reshape(-1)        # [20]

    # Activate newly spawning comets, deactivate expired ones
    flat_act_or_spawn = jnp.repeat(is_active_or_spawning, 4)  # [20]
    flat_expired      = jnp.repeat(is_expired, 4)              # [20]

    # Conditional scatter: only overwrite if group is active_or_spawning
    conditional_x = jnp.where(flat_act_or_spawn, flat_new_x, planets.x[flat_slots])
    conditional_y = jnp.where(flat_act_or_spawn, flat_new_y, planets.y[flat_slots])
    new_px = planets.x.at[flat_slots].set(conditional_x)
    new_py = planets.y.at[flat_slots].set(conditional_y)

    # Activate slots that are spawning (set active=True, owner=-1, ships from spawn_ships)
    flat_spawning = jnp.repeat(spawning, 4)  # [20]
    flat_spawn_ships = jnp.repeat(comets.spawn_ships, 4).astype(jnp.float32)  # [20]
    conditional_ships = jnp.where(flat_spawning, flat_spawn_ships, planets.ships[flat_slots])
    new_active = planets.active.at[flat_slots].set(
        jnp.where(flat_spawning, True, planets.active[flat_slots])
    )
    new_ships_p = planets.ships.at[flat_slots].set(conditional_ships)

    # Deactivate expired comet slots
    conditional_alive = jnp.where(flat_expired, False, new_active[flat_slots])
    new_active = new_active.at[flat_slots].set(conditional_alive)

    new_planets = planets._replace(
        x=new_px, y=new_py,
        ships=new_ships_p,
        active=new_active,
    )
    new_comets = comets._replace(path_indices=new_path_indices)
    return new_planets, new_comets


def _launch_fleets(
    planets: PlanetState,
    fleets: FleetState,
    next_slot: jnp.ndarray,
    actions: jnp.ndarray,
    num_players: int,
    ship_speed: float,
):
    """Process all player launch actions.  Returns updated planets, fleets, next_slot.

    actions: float32[num_players, MAX_PLANETS, 2]
      [:, :, 0] = angle (radians)
      [:, :, 1] = ship fraction; ≤0 = no launch

    Ships are deducted from the planet immediately.  Multiple actions for the
    same planet are processed left-to-right (first player first, then by slot).
    """
    angles  = actions[:, :, 0]           # [num_players, MAX_PLANETS]
    fracs   = actions[:, :, 1]           # [num_players, MAX_PLANETS]

    # Flatten to [num_players * MAX_PLANETS] to scan over
    n_actions = num_players * MAX_PLANETS
    flat_angles = angles.reshape(n_actions)
    flat_fracs  = fracs.reshape(n_actions)
    flat_players = jnp.repeat(jnp.arange(num_players, dtype=jnp.int32), MAX_PLANETS)
    flat_pslots  = jnp.tile(jnp.arange(MAX_PLANETS, dtype=jnp.int32), num_players)

    def _one_action(carry, i):
        fleets_, planets_, next_slot_ = carry

        player    = flat_players[i]
        p_slot    = flat_pslots[i]
        angle     = flat_angles[i]
        frac      = flat_fracs[i]

        avail_ships = planets_.ships[p_slot]
        raw_ships   = frac * avail_ships
        ships_i     = jnp.floor(jnp.maximum(raw_ships, 0.0)).astype(jnp.int32)

        is_valid = (
            (planets_.owner[p_slot] == player)
            & planets_.active[p_slot]
            & (ships_i > 0)
            & (avail_ships >= ships_i.astype(jnp.float32))
        )

        # Fleet start position: just outside the planet surface
        r = planets_.radius[p_slot]
        start_x = planets_.x[p_slot] + jnp.cos(angle) * (r + 0.1)
        start_y = planets_.y[p_slot] + jnp.sin(angle) * (r + 0.1)

        slot = (next_slot_ % MAX_FLEETS).astype(jnp.int32)

        # Overwrite fleet slot (circular buffer — evicts oldest if full)
        new_fx   = fleets_.x.at[slot].set(jnp.where(is_valid, start_x, fleets_.x[slot]))
        new_fy   = fleets_.y.at[slot].set(jnp.where(is_valid, start_y, fleets_.y[slot]))
        new_fa   = fleets_.angle.at[slot].set(jnp.where(is_valid, angle, fleets_.angle[slot]))
        new_fs   = fleets_.ships.at[slot].set(jnp.where(is_valid, ships_i.astype(jnp.float32), fleets_.ships[slot]))
        new_fo   = fleets_.owner.at[slot].set(jnp.where(is_valid, player, fleets_.owner[slot]))
        new_ffp  = fleets_.from_planet.at[slot].set(jnp.where(is_valid, p_slot, fleets_.from_planet[slot]))
        new_fact = fleets_.active.at[slot].set(jnp.where(is_valid, True, fleets_.active[slot]))

        new_fleets = fleets_._replace(
            x=new_fx, y=new_fy, angle=new_fa,
            ships=new_fs, owner=new_fo,
            from_planet=new_ffp, active=new_fact,
        )

        # Deduct ships from planet
        new_p_ships = planets_.ships.at[p_slot].add(
            jnp.where(is_valid, -ships_i.astype(jnp.float32), 0.0)
        )
        new_planets = planets_._replace(ships=new_p_ships)

        new_next = next_slot_ + jnp.where(is_valid, 1, 0)
        return (new_fleets, new_planets, new_next), None

    (new_fleets, new_planets, new_next), _ = jax.lax.scan(
        _one_action,
        (fleets, planets, next_slot),
        jnp.arange(n_actions, dtype=jnp.int32),
    )
    return new_planets, new_fleets, new_next


def _move_fleets_and_detect_collisions(
    planets: PlanetState,
    planet_x_new: jnp.ndarray,
    planet_y_new: jnp.ndarray,
    fleets: FleetState,
    ship_speed: float,
):
    """Move all active fleets one tick, using continuous swept-pair collision detection.

    Returns (new_fleets, hit_planet_idx):
      hit_planet_idx: int32[MAX_FLEETS]  — planet slot the fleet hit, or -1
    """
    speeds = jax.vmap(fleet_speed)(fleets.ships, jnp.full(MAX_FLEETS, ship_speed))

    new_fx = fleets.x + jnp.cos(fleets.angle) * speeds
    new_fy = fleets.y + jnp.sin(fleets.angle) * speeds

    # --- Swept collision detection: [MAX_FLEETS, MAX_PLANETS] ---
    # For each (fleet, planet) pair: earliest collision time t ∈ [0,1] or inf
    def _fleet_planet_hit(f_old_x, f_old_y, f_new_x, f_new_y, f_active):
        def _vs_one_planet(p_old_x, p_old_y, p_new_x, p_new_y, p_radius, p_active, p_checkable):
            check = p_active & p_checkable & f_active
            t = swept_pair_hit_time(
                f_old_x, f_old_y, f_new_x, f_new_y,
                p_old_x, p_old_y, p_new_x, p_new_y,
                p_radius,
            )
            return jnp.where(check, t, jnp.inf)

        return jax.vmap(_vs_one_planet)(
            planets.x, planets.y,
            planet_x_new, planet_y_new,
            planets.radius,
            planets.active,
            # Comets on their very first tick (active just set this tick) have
            # old_pos = off-board (-99,-99); skip collision check for them.
            # We approximate: comets that are active but currently off-board.
            (planets.x >= 0.0) | ~planets.is_comet,
        )  # [MAX_PLANETS]

    # Vectorise over fleets → [MAX_FLEETS, MAX_PLANETS]
    t_matrix = jax.vmap(_fleet_planet_hit)(
        fleets.x, fleets.y, new_fx, new_fy, fleets.active
    )  # [MAX_FLEETS, MAX_PLANETS]

    # Earliest collision per fleet
    first_planet = jnp.argmin(t_matrix, axis=1)        # [MAX_FLEETS]
    first_t      = t_matrix[jnp.arange(MAX_FLEETS), first_planet]  # [MAX_FLEETS]
    did_hit      = first_t < jnp.inf                   # [MAX_FLEETS]

    # Out-of-bounds check
    oob = (
        (new_fx < 0.0) | (new_fx > BOARD_SIZE) |
        (new_fy < 0.0) | (new_fy > BOARD_SIZE)
    ) & fleets.active

    # Sun intersection check (segment from old to new crosses sun circle)
    sun_dist_sq = jax.vmap(lambda ax, ay, bx, by:
        point_to_segment_dist_sq(CENTER, CENTER, ax, ay, bx, by)
    )(fleets.x, fleets.y, new_fx, new_fy)
    hits_sun = (sun_dist_sq < SUN_RADIUS ** 2) & fleets.active

    # Deactivate fleets that hit a planet, went OOB, or hit the sun
    remove = did_hit | oob | hits_sun
    hit_planet_idx = jnp.where(did_hit, first_planet, -1).astype(jnp.int32)

    new_fleets = fleets._replace(
        x=jnp.where(~remove, new_fx, fleets.x),
        y=jnp.where(~remove, new_fy, fleets.y),
        active=fleets.active & ~remove,
    )
    return new_fleets, hit_planet_idx


def _resolve_combat(planets: PlanetState, fleets: FleetState,
                    hit_planet_idx: jnp.ndarray, num_players: int):
    """Accumulate attacking fleets per planet and resolve combat.

    Returns updated PlanetState.
    """
    # Attacking ships per (planet, player): shape [MAX_PLANETS, num_players]
    # fleet hit planet p and is owned by player q → attacking[p, q] += ships
    fleet_hit = hit_planet_idx >= 0  # [MAX_FLEETS] bool — was this fleet involved in combat?

    # Build owner one-hot: [MAX_FLEETS, num_players]
    player_ids = jnp.arange(num_players, dtype=jnp.int32)
    owner_hot = (fleets.owner[:, None] == player_ids[None, :]).astype(jnp.float32)  # [MAX_FLEETS, num_players]

    # attacking[p, q] = sum_{f : hit_planet[f]==p, owner[f]==q} ships[f]
    # Use segment_sum trick: shape [MAX_PLANETS * num_players]
    # index = hit_planet_idx[f] * num_players + owner[f]
    safe_idx = jnp.maximum(hit_planet_idx, 0)  # avoid -1 becoming large index
    flat_idx = safe_idx * num_players + fleets.owner  # [MAX_FLEETS]
    # Use `fleet_hit` (not fleets.active): the colliding fleets were already
    # deactivated by _move_fleets_and_detect_collisions, so keying off active
    # would zero out every attacker and combat would never resolve.  A fleet
    # with hit_planet_idx >= 0 was active when it struck and must contribute.
    flat_idx = jnp.where(fleet_hit, flat_idx, MAX_PLANETS * num_players)
    # Use a pad slot for "no hit" contributions (index = MAX_PLANETS * num_players)
    attacking_flat = jax.ops.segment_sum(
        fleets.ships * fleet_hit.astype(jnp.float32),
        flat_idx,
        MAX_PLANETS * num_players + 1,
    )  # [MAX_PLANETS * num_players + 1]
    attacking = attacking_flat[: MAX_PLANETS * num_players].reshape(MAX_PLANETS, num_players)
    # attacking[p, q] = total attacking ships from player q at planet p

    def _planet_combat(p_idx):
        att = attacking[p_idx]           # [num_players]
        planet_owner = planets.owner[p_idx]
        planet_ships = planets.ships[p_idx]
        p_active = planets.active[p_idx]

        total_att = att.sum()
        has_combat = (total_att > 0) & p_active

        # Top two attacking players
        sorted_players = jnp.argsort(-att)  # descending
        top_player = sorted_players[0]
        top_ships  = att[sorted_players[0]]
        sec_ships  = att[sorted_players[1]] if num_players > 1 else jnp.array(0.0)

        survivor_ships = top_ships - sec_ships
        tie = top_ships == sec_ships
        survivor_ships = jnp.where(tie, 0.0, survivor_ships)
        survivor_owner = jnp.where(survivor_ships > 0, top_player, jnp.array(-1, dtype=jnp.int32))

        # Apply combat result to planet
        same_owner = survivor_owner == planet_owner
        no_survivor = survivor_ships <= 0

        # Case 1: surviving owner == planet owner → add survivors to garrison
        garrison_add = jnp.where(has_combat & same_owner & ~no_survivor, survivor_ships, 0.0)
        # Case 2: surviving owner != planet owner → fight garrison
        garrison_diff = planet_ships - survivor_ships
        captured = garrison_diff < 0
        new_garrison = jnp.where(has_combat & ~same_owner & ~no_survivor,
                                  jnp.where(captured, -garrison_diff, garrison_diff),
                                  planet_ships + garrison_add)
        new_garrison = jnp.where(p_active, new_garrison, planet_ships)
        new_owner = jnp.where(
            has_combat & ~same_owner & ~no_survivor & captured,
            survivor_owner,
            planet_owner,
        )
        return new_garrison, new_owner

    # vmap over all planet indices
    new_ships_planets, new_owners = jax.vmap(_planet_combat)(
        jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    )
    return planets._replace(ships=new_ships_planets, owner=new_owners)


def _compute_rewards_and_done(planets: PlanetState, fleets: FleetState,
                               step: jnp.ndarray, episode_steps: int,
                               num_players: int) -> tuple:
    """Returns (rewards float32[num_players], done bool)."""
    # Count planets + fleet ships per player
    player_ids = jnp.arange(num_players, dtype=jnp.int32)
    p_mask = planets.owner[:, None] == player_ids[None, :]  # [MAX_PLANETS, num_players]
    f_mask = fleets.owner[:, None]  == player_ids[None, :]  # [MAX_FLEETS, num_players]

    planet_ships = (planets.ships[:, None] * planets.active[:, None] * p_mask).sum(axis=0)
    fleet_ships  = (fleets.ships[:, None]  * fleets.active[:, None]  * f_mask).sum(axis=0)
    scores = planet_ships + fleet_ships  # [num_players]

    # Termination: step limit or only one player alive
    alive = scores > 0
    num_alive = alive.sum()
    time_limit = step >= episode_steps - 2

    done = time_limit | (num_alive <= 1)

    max_score = scores.max()
    rewards = jnp.where(
        (scores == max_score) & (max_score > 0),
        jnp.ones(num_players, dtype=jnp.float32),
        -jnp.ones(num_players, dtype=jnp.float32),
    )
    return rewards, done


# ---------------------------------------------------------------------------
# Main step function
# ---------------------------------------------------------------------------

def step(
    state: GameState,
    actions: jnp.ndarray,
    *,
    num_players: int = 2,
    episode_steps: int = 500,
    ship_speed: float = 6.0,
    comet_speed: float = 4.0,
) -> GameState:
    """Advance the game by one tick.

    Parameters
    ----------
    state:
        Current GameState (single env, not batched).
    actions:
        float32[num_players, MAX_PLANETS, 2]
        actions[p, i] = [angle, ship_fraction] for planet slot i, player p.
    num_players, episode_steps, ship_speed, comet_speed:
        Static configuration — must be the same across all calls within a
        jit/vmap scope (use functools.partial to bind them).

    Returns
    -------
    GameState with step incremented and all game state updated.
    """
    # No-op if already done
    def _do_step(s):
        planets = s.planets
        fleets  = s.fleets
        comets  = s.comets
        game_step = s.step

        # 1. Comet advancement (advance paths, spawn new groups, expire old ones)
        planets, comets = _advance_comets(planets, comets, game_step)

        # 2. Compute new planet positions (orbit rotation) — keep old for sweep check
        planet_x_old = planets.x
        planet_y_old = planets.y
        planet_x_new, planet_y_new = _update_planet_positions(
            planets, s.angular_velocity, game_step + 1
        )
        # Comet positions are already updated by _advance_comets; keep them
        planet_x_new = jnp.where(planets.is_comet, planets.x, planet_x_new)
        planet_y_new = jnp.where(planets.is_comet, planets.y, planet_y_new)

        # 3. Fleet launch (original order: launch before production)
        planets, fleets, next_slot = _launch_fleets(
            planets, fleets, s.next_fleet_slot, actions,
            num_players, ship_speed
        )

        # 4. Production
        owned_active = (planets.owner >= 0) & planets.active
        planets = planets._replace(
            ships=planets.ships + planets.production.astype(jnp.float32) * owned_active
        )

        # 5. Fleet movement + collision detection (uses old/new planet positions)
        # Temporarily expose old positions for the swept check
        planets_for_sweep = planets._replace(x=planet_x_old, y=planet_y_old)
        fleets, hit_planet_idx = _move_fleets_and_detect_collisions(
            planets_for_sweep, planet_x_new, planet_y_new, fleets, ship_speed
        )

        # 6. Apply new planet positions (orbiting; comet positions already set)
        planets = planets._replace(x=planet_x_new, y=planet_y_new)

        # 7. Combat resolution
        planets = _resolve_combat(planets, fleets, hit_planet_idx, num_players)

        # 8. Deactivate fleets that participated in combat
        fleets = fleets._replace(active=fleets.active & (hit_planet_idx < 0))

        # 9. Termination + rewards
        rewards, done = _compute_rewards_and_done(
            planets, fleets, game_step, episode_steps, num_players
        )

        return GameState(
            planets=planets,
            fleets=fleets,
            comets=comets,
            step=game_step + 1,
            done=done,
            rewards=jnp.where(done, rewards, s.rewards),
            angular_velocity=s.angular_velocity,
            next_fleet_slot=next_slot,
            num_players=s.num_players,
        )

    return jax.lax.cond(state.done, lambda s: s, _do_step, state)
