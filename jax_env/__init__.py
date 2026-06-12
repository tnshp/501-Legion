"""JAX-accelerated vectorised Orbit Wars environment."""
from .constants import MAX_PLANETS, MAX_FLEETS, MAX_COMET_GROUPS
from .env_types import GameState, PlanetState, FleetState, CometData
from .reset import reset, batch_reset
from .step import step
from .vec_env import VectorizedEnv
from .adapter import JaxVecEnvAdapter

__all__ = [
    "GameState", "PlanetState", "FleetState", "CometData",
    "reset", "batch_reset", "step",
    "VectorizedEnv",
    "JaxVecEnvAdapter",
    "MAX_PLANETS", "MAX_FLEETS", "MAX_COMET_GROUPS",
]
