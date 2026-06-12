import jax.numpy as jnp

# Array size caps — all environments share these static shapes
MAX_PLANETS = 60          # 20-40 regular + up to 20 comets, padded to power of 2
MAX_FLEETS = 1024         # circular buffer; oldest evicted if exceeded
MAX_COMET_GROUPS = 5      # exactly 5 spawn events per episode
MAX_COMET_PATH_LEN = 40   # max on-board path length (generate_comet_paths: 5-40)

# Game geometry
BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
PLANET_CLEARANCE = 7.0

# Comet groups spawn when (step + 1) equals one of these values
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]

# Action encoding: actions[player, planet_slot] = [angle_radians, ship_fraction]
# ship_fraction <= 0.0  →  no launch from this planet slot
ACTION_NO_LAUNCH = 0.0
