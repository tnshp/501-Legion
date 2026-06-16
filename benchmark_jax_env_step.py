"""
Microbenchmark for the JAX Orbit Wars environment STEP — CPU vs GPU.

Isolates the environment cost from the learner so you can answer "does running
the JAX env on GPU actually speed things up vs CPU?".  Run the SAME script on a
CPU-only jaxlib and on a CUDA jaxlib (e.g. your A30 box) and compare the
transitions/s — the script prints the active JAX backend up top so the two runs
are unambiguous.

Two phases are timed per num_envs:

  engine  — VectorizedEnv.step only (pure JAX/XLA game step, vmapped + jitted).
            This is the part a GPU actually accelerates.  Timed with
            jax.block_until_ready so async dispatch can't hide the real compute.
  full    — JaxVecEnvAdapter.step: the engine step PLUS host-side obs extraction
            (a device→host copy every step on GPU), the numpy opponent, and the
            reward computation.  This is what the training loop pays per vstep.

Why both: a GPU can make `engine` much faster while `full` barely moves (or
regresses) because the per-step device→host obs copy and the numpy opponent run
on the CPU regardless.  Seeing the split tells you whether a GPU jaxlib is worth
it for THIS workload and at what num_envs the crossover happens.

Usage
-----
    python benchmark_jax_env_step.py                       # default sweep
    python benchmark_jax_env_step.py --sweep 64,256,1024,4096
    python benchmark_jax_env_step.py --mode engine         # pure step only
    python benchmark_jax_env_step.py --opponent greedy     # heavier opponent in 'full'

To force CPU even when a GPU jaxlib is installed (for the CPU baseline):
    JAX_PLATFORMS=cpu python benchmark_jax_env_step.py
"""
from __future__ import annotations

import argparse
import os
import time

# Grow VRAM on demand rather than JAX's default 75 % grab (matches the training
# scripts so the numbers are comparable).  Must precede `import jax`.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from jax_env import VectorizedEnv, JaxVecEnvAdapter
from jax_env.constants import MAX_PLANETS  # engine action planet dim (= 60)


# ─────────────────────────────────────────────────────────────────────────────
# Timed phases
# ─────────────────────────────────────────────────────────────────────────────

def _bench_engine(num_envs, num_players, episode_steps, steps, warmup):
    """Pure VectorizedEnv.step throughput (device compute only)."""
    env   = VectorizedEnv(num_envs=num_envs, num_players=num_players,
                          episode_steps=episode_steps)
    state = env.reset()
    # Zero actions = no launches.  The per-tick physics (rotation, production,
    # fleet movement, combat, comets) runs over fixed-size arrays every tick
    # regardless of launches, so XLA does representative work either way.
    actions = jnp.zeros((num_envs, num_players, MAX_PLANETS, 2), dtype=jnp.float32)

    # Warmup: triggers JIT trace/compile (first call is very slow).
    for _ in range(warmup):
        state, _, _ = env.step(state, actions)
    jax.block_until_ready(state)

    t0 = time.perf_counter()
    for _ in range(steps):
        state, _, _ = env.step(state, actions)
    jax.block_until_ready(state)          # force async dispatch to complete
    wall = time.perf_counter() - t0

    return steps * num_envs / wall, wall / steps * 1e3


def _bench_full(num_envs, num_players, episode_steps, steps, warmup, opponent):
    """Full JaxVecEnvAdapter.step throughput (engine + host obs + opponent + reward)."""
    env = JaxVecEnvAdapter(num_envs=num_envs, num_players=num_players,
                           episode_steps=episode_steps, opponent=opponent)
    env.reset()
    # Raw player-0 policy output [B, NET_MP, 4]; zeros decode to no launches.
    raw = np.zeros((num_envs, 40, 4), dtype=np.float32)

    for _ in range(warmup):
        env.step(raw)                     # adapter pulls to host → synchronous

    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(raw)
    wall = time.perf_counter() - t0
    env.close()

    return steps * num_envs / wall, wall / steps * 1e3


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="JAX env step microbenchmark (CPU vs GPU)")
    ap.add_argument("--sweep",         default="64,256,1024",
                    help="comma-separated num_envs values (default: 64,256,1024)")
    ap.add_argument("--steps",         type=int, default=100,
                    help="measured steps per num_envs (default: 100)")
    ap.add_argument("--warmup",        type=int, default=5,
                    help="warmup steps for JIT compile (default: 5)")
    ap.add_argument("--num-players",   type=int, default=2)
    ap.add_argument("--episode-steps", type=int, default=500)
    ap.add_argument("--opponent",      default="random",
                    choices=["random", "greedy", "agent1"],
                    help="opponent for the 'full' phase (default: random)")
    ap.add_argument("--mode",          default="both",
                    choices=["engine", "full", "both"],
                    help="which phase(s) to time (default: both)")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sweep.split(",")]

    backend = jax.default_backend()
    print("=" * 72)
    print("JAX ENV STEP MICROBENCHMARK")
    print("=" * 72)
    print(f"  jax {jax.__version__}  |  backend: {backend.upper()}  |  devices: {jax.devices()}")
    if backend == "cpu":
        print("  NOTE: running on CPU — install a CUDA jaxlib (pip install -U "
              "'jax[cuda12]') and re-run to get the GPU numbers.")
    print(f"  num_players={args.num_players}  episode_steps={args.episode_steps}  "
          f"steps={args.steps}  warmup={args.warmup}  opponent={args.opponent}")
    print(f"  phases: {args.mode}  "
          f"(engine = pure VectorizedEnv.step; full = adapter step incl. host obs)")
    print()

    do_eng  = args.mode in ("engine", "both")
    do_full = args.mode in ("full", "both")

    hdr = f"  {'num_envs':>9} |"
    if do_eng:
        hdr += f" {'engine trans/s':>15} {'ms/step':>9} |"
    if do_full:
        hdr += f" {'full trans/s':>14} {'ms/step':>9} |"
    if do_eng and do_full:
        hdr += f" {'engine/full':>11}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for n in sizes:
        row = f"  {n:>9} |"
        eng_tps = full_tps = None
        if do_eng:
            eng_tps, eng_ms = _bench_engine(
                n, args.num_players, args.episode_steps, args.steps, args.warmup)
            row += f" {eng_tps:>15,.0f} {eng_ms:>9.3f} |"
        if do_full:
            full_tps, full_ms = _bench_full(
                n, args.num_players, args.episode_steps, args.steps, args.warmup,
                args.opponent)
            row += f" {full_tps:>14,.0f} {full_ms:>9.3f} |"
        if do_eng and do_full and full_tps:
            row += f" {eng_tps / full_tps:>10.1f}x"
        print(row)

    print()
    print("  Read: 'engine' is the pure on-device game step; 'full' here uses the")
    print("  adapter's NumPy fallback (no reward_cfg), so it still pays the host")
    print("  opponent + obs copy.  In training the adapter's on-device FAST PATH")
    print("  (heuristic opponent + JAX-portable reward) runs the opponent + reward")
    print("  + obs on the GPU fused with the step — closing most of this gap.")


if __name__ == "__main__":
    main()
