"""Example: run N parallel Orbit Wars environments with random agents.

Run with:
    conda run -n orbit_wars python example.py
"""
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import VectorizedEnv, MAX_PLANETS


# ---------------------------------------------------------------------------
# Random policy
# ---------------------------------------------------------------------------

def random_actions(rng_key, state, num_envs, num_players):
    """For each owned planet, launch ~50% of ships in a random direction."""
    rng_key, subkey = jax.random.split(rng_key)
    angles = jax.random.uniform(subkey, (num_envs, num_players, MAX_PLANETS), minval=0.0, maxval=2 * jnp.pi)

    rng_key, subkey = jax.random.split(rng_key)
    # Random fraction between 0.3 and 0.7 so ships are always split
    fracs = jax.random.uniform(subkey, (num_envs, num_players, MAX_PLANETS), minval=0.3, maxval=0.7)

    # Only launch from planets owned by the right player
    # state.planets.owner: [num_envs, MAX_PLANETS]
    player_ids = jnp.arange(num_players)[None, :, None]          # [1, num_players, 1]
    owned = state.planets.owner[:, None, :] == player_ids         # [num_envs, num_players, MAX_PLANETS]
    active = state.planets.active[:, None, :]                     # [num_envs, 1,           MAX_PLANETS]

    # Zero out launches from unowned or inactive planets
    fracs = fracs * owned * active

    actions = jnp.stack([angles, fracs], axis=-1)   # [num_envs, num_players, MAX_PLANETS, 2]
    return rng_key, actions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    NUM_ENVS     = 128
    NUM_PLAYERS  = 2
    EPISODE_STEPS = 500

    print(f"Orbit Wars JAX — {NUM_ENVS} parallel environments, {NUM_PLAYERS} players")
    print(f"Devices: {jax.devices()}\n")

    env = VectorizedEnv(
        num_envs=NUM_ENVS,
        num_players=NUM_PLAYERS,
        episode_steps=EPISODE_STEPS,
    )

    seeds = np.arange(NUM_ENVS)
    state = env.reset(seeds)
    print(f"Reset done — active planets per env (sample): {np.array(state.planets.active.sum(axis=-1)[:4])}")

    rng = jax.random.PRNGKey(0)

    # --- Warm up JIT ---
    rng, acts = random_actions(rng, state, NUM_ENVS, NUM_PLAYERS)
    state, rewards, dones = env.step(state, acts)
    state.step.block_until_ready()
    print("JIT warm-up done\n")

    # --- Run full episodes ---
    state = env.reset(seeds)
    rng = jax.random.PRNGKey(0)

    episode_rewards = np.zeros((NUM_ENVS, NUM_PLAYERS), dtype=np.float32)
    episode_lengths = np.zeros(NUM_ENVS, dtype=np.int32)
    finished        = np.zeros(NUM_ENVS, dtype=bool)
    total_steps     = 0

    t0 = time.perf_counter()

    for tick in range(EPISODE_STEPS):
        rng, acts = random_actions(rng, state, NUM_ENVS, NUM_PLAYERS)
        state, rewards, dones = env.step(state, acts)
        total_steps += NUM_ENVS

        # Record first-completion rewards/lengths
        newly_done = np.array(dones) & ~finished
        if newly_done.any():
            r = np.array(rewards)
            l = np.array(state.step)
            episode_rewards[newly_done] = r[newly_done]
            episode_lengths[newly_done] = l[newly_done]
            finished |= newly_done

        if finished.all():
            break

    state.step.block_until_ready()
    elapsed = time.perf_counter() - t0

    # --- Results ---
    print(f"All {NUM_ENVS} episodes finished in {int(state.step[0])} steps")
    print(f"Wall time : {elapsed:.3f}s")
    print(f"Throughput: {total_steps / elapsed:,.0f} env-steps/s\n")

    print("Episode length  — mean: {:.1f}  min: {}  max: {}".format(
        episode_lengths.mean(), episode_lengths.min(), episode_lengths.max()))
    print("Player 0 reward — mean: {:.3f}  win rate: {:.1%}".format(
        episode_rewards[:, 0].mean(),
        (episode_rewards[:, 0] > 0).mean(),
    ))
    print("Player 1 reward — mean: {:.3f}  win rate: {:.1%}".format(
        episode_rewards[:, 1].mean(),
        (episode_rewards[:, 1] > 0).mean(),
    ))

    # --- Comet activity at final step ---
    final_comet_indices = np.array(state.comets.path_indices)
    spawned = (final_comet_indices >= 0).sum(axis=-1)  # per env
    print(f"\nComet groups spawned per env — mean: {spawned.mean():.1f}  "
          f"min: {spawned.min()}  max: {spawned.max()}")


if __name__ == "__main__":
    main()
