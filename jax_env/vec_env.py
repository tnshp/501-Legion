"""High-level vectorised environment wrapping the JAX step and reset functions.

Example
-------
    env = VectorizedEnv(num_envs=256, num_players=2, episode_steps=500)
    state = env.reset(seeds=np.arange(256))

    # Dummy no-op actions
    actions = np.zeros((256, 2, MAX_PLANETS, 2), dtype=np.float32)

    state, rewards, dones = env.step(state, actions)
"""
import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .constants import MAX_PLANETS
from .reset import batch_reset, reset
from .step import step as _step_fn
from .env_types import GameState


class VectorizedEnv:
    """Wraps `reset` and `step` for N parallel environments.

    All heavy computation is JIT-compiled once on the first call.

    Parameters
    ----------
    num_envs:
        Number of environments to run in parallel.
    num_players:
        2 or 4.
    episode_steps:
        Maximum steps per episode (default 500).
    ship_speed:
        Maximum fleet speed (default 6.0).
    comet_speed:
        Comet movement speed per tick (default 4.0).
    """

    def __init__(
        self,
        num_envs: int,
        num_players: int = 2,
        episode_steps: int = 500,
        ship_speed: float = 6.0,
        comet_speed: float = 4.0,
    ):
        self.num_envs     = num_envs
        self.num_players  = num_players
        self.episode_steps = episode_steps
        self.ship_speed   = ship_speed
        self.comet_speed  = comet_speed

        # Bind static config to the step function, then jit + vmap
        _step_bound = functools.partial(
            _step_fn,
            num_players=num_players,
            episode_steps=episode_steps,
            ship_speed=ship_speed,
            comet_speed=comet_speed,
        )
        self._step_jit  = jax.jit(_step_bound)
        self._step_vmap = jax.jit(jax.vmap(_step_bound))
        # Un-jitted single-env step (statics already bound) so callers can fuse it
        # with on-device opponent/reward into one jitted function — see
        # jax_env.adapter's fast path.
        self._step_bound = _step_bound

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seeds: Sequence[int] | None = None) -> GameState:
        """Initialise all environments.

        Parameters
        ----------
        seeds:
            Integer seeds, one per environment.  If None, uses 0..num_envs-1.

        Returns
        -------
        Batched GameState with leading dimension = num_envs.
        """
        if seeds is None:
            seeds = np.arange(self.num_envs)
        assert len(seeds) == self.num_envs, (
            f"Expected {self.num_envs} seeds, got {len(seeds)}"
        )
        return batch_reset(
            seeds,
            num_players=self.num_players,
            episode_steps=self.episode_steps,
            ship_speed=self.ship_speed,
            comet_speed=self.comet_speed,
        )

    def reset_single(self, seed: int) -> GameState:
        """Reset a single environment (returns un-batched GameState)."""
        return reset(
            seed,
            num_players=self.num_players,
            episode_steps=self.episode_steps,
            ship_speed=self.ship_speed,
            comet_speed=self.comet_speed,
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        state: GameState,
        actions: jnp.ndarray,
    ) -> tuple[GameState, jnp.ndarray, jnp.ndarray]:
        """Step all N environments.

        Parameters
        ----------
        state:
            Batched GameState (leading dim = num_envs).
        actions:
            float32[num_envs, num_players, MAX_PLANETS, 2]
            actions[env, player, planet_slot] = [angle_rad, ship_fraction]
            ship_fraction ≤ 0  →  no launch.

        Returns
        -------
        new_state : GameState  — updated batched state
        rewards   : float32[num_envs, num_players]
        dones     : bool[num_envs]
        """
        new_state = self._step_vmap(state, actions)
        return new_state, new_state.rewards, new_state.done

    def step_single(
        self,
        state: GameState,
        actions: jnp.ndarray,
    ) -> tuple[GameState, jnp.ndarray, bool]:
        """Step a single un-batched environment."""
        new_state = self._step_jit(state, actions)
        return new_state, new_state.rewards, bool(new_state.done)

    # ------------------------------------------------------------------
    # Convenience: auto-reset done environments
    # ------------------------------------------------------------------

    def step_auto_reset(
        self,
        state: GameState,
        actions: jnp.ndarray,
        seeds: Sequence[int] | None = None,
    ) -> tuple[GameState, jnp.ndarray, jnp.ndarray]:
        """Step + reset any finished environments in-place.

        Finished environments are reset using the provided seeds (defaults to
        incrementing from 0).  Returns the same shapes as `step`.
        """
        new_state, rewards, dones = self.step(state, actions)

        done_indices = np.where(np.array(dones))[0]
        if len(done_indices) == 0:
            return new_state, rewards, dones

        if seeds is None:
            # Generate fresh seeds for done envs (use current step as entropy)
            base = int(new_state.step[done_indices[0]])
            fresh_seeds = [base + i for i in done_indices]
        else:
            fresh_seeds = [seeds[i] for i in done_indices]

        # Reset each done env and write back into batched state
        # (Python loop is fine — usually a small fraction of envs are done)
        states_list = [jax.tree_util.tree_map(lambda x: x[i], new_state)
                       for i in range(self.num_envs)]
        for idx, seed in zip(done_indices, fresh_seeds):
            states_list[idx] = self.reset_single(int(seed))

        merged = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states_list)
        return merged, rewards, dones

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    @property
    def action_shape(self):
        """Shape of the actions array for all envs: (num_envs, num_players, MAX_PLANETS, 2)."""
        return (self.num_envs, self.num_players, MAX_PLANETS, 2)

    @property
    def single_action_shape(self):
        """Shape of actions for one environment: (num_players, MAX_PLANETS, 2)."""
        return (self.num_players, MAX_PLANETS, 2)

    def noop_actions(self, batched: bool = True) -> jnp.ndarray:
        """Return a zero-action array (no launches)."""
        shape = self.action_shape if batched else self.single_action_shape
        return jnp.zeros(shape, dtype=jnp.float32)
