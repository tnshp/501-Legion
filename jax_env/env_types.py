"""JAX-compatible state types as NamedTuples (automatic pytree registration)."""
from typing import NamedTuple
import jax.numpy as jnp


class PlanetState(NamedTuple):
    """Per-planet arrays, shape [MAX_PLANETS] each.

    Planets 0..N_planets-1 hold actual planets; the rest are padding.
    Comet planets live at fixed slots determined at reset time.
    """
    x: jnp.ndarray          # float32 — current x (changes each tick for orbiting/comets)
    y: jnp.ndarray          # float32 — current y
    init_x: jnp.ndarray     # float32 — x at step 0 (used to compute orbit angle)
    init_y: jnp.ndarray     # float32 — y at step 0
    radius: jnp.ndarray     # float32
    ships: jnp.ndarray      # float32
    production: jnp.ndarray # int32
    owner: jnp.ndarray      # int32 — -1 = neutral, 0..num_players-1 = player
    active: jnp.ndarray     # bool  — False = padding / expired comet
    is_comet: jnp.ndarray   # bool  — True = extra-solar comet planet


class FleetState(NamedTuple):
    """Per-fleet arrays, shape [MAX_FLEETS] each.  Circular buffer.

    next_slot in GameState points to the oldest slot (next to be overwritten).
    """
    x: jnp.ndarray          # float32
    y: jnp.ndarray          # float32
    angle: jnp.ndarray      # float32 — heading in radians
    ships: jnp.ndarray      # float32
    owner: jnp.ndarray      # int32
    from_planet: jnp.ndarray # int32 — planet slot the fleet launched from
    active: jnp.ndarray     # bool


class CometData(NamedTuple):
    """Precomputed comet paths (generated once at reset, used throughout episode).

    paths[g, k, t, :] = (x, y) of comet group g, symmetry copy k, at path step t.
    Shapes:
      paths:        float32[MAX_COMET_GROUPS, 4, MAX_COMET_PATH_LEN, 2]
      path_lengths: int32[MAX_COMET_GROUPS]   — valid entries in path dim
      spawn_ships:  int32[MAX_COMET_GROUPS]   — ship count when comet spawns
      path_indices: int32[MAX_COMET_GROUPS]   — current path step, -1 = not yet spawned
      planet_slots: int32[MAX_COMET_GROUPS, 4] — indices into PlanetState arrays
    """
    paths: jnp.ndarray
    path_lengths: jnp.ndarray
    spawn_ships: jnp.ndarray
    path_indices: jnp.ndarray
    planet_slots: jnp.ndarray


class GameState(NamedTuple):
    """Complete state of one environment instance.

    All fields are JAX arrays — no Python scalars — so the struct is vmappable.

    Actions fed to step() have shape [num_players, MAX_PLANETS, 2]:
      actions[player, planet_slot, 0] = launch angle (radians)
      actions[player, planet_slot, 1] = ship fraction in (0, 1]; 0 = no launch
    """
    planets: PlanetState
    fleets: FleetState
    comets: CometData
    step: jnp.ndarray          # int32 scalar — incremented at end of each step()
    done: jnp.ndarray          # bool scalar
    rewards: jnp.ndarray       # float32[num_players]
    angular_velocity: jnp.ndarray  # float32 scalar
    next_fleet_slot: jnp.ndarray   # int32 scalar — circular buffer write pointer
    num_players: jnp.ndarray   # int32 scalar — 2 or 4
