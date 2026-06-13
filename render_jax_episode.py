"""
render_jax_episode.py
Run one JAX episode, convert each GameState to kaggle's observation format,
and export an HTML replay renderable in the browser.

Usage
-----
    conda run -n orbit_wars_sb3 python render_jax_episode.py \
        --checkpoint checkpoints/best.pt --output replay.html

    # Random policy (no checkpoint needed — good for quick testing)
    conda run -n orbit_wars_sb3 python render_jax_episode.py --random

Arguments
---------
    --checkpoint    Path to a .pt checkpoint produced by train_orbit_wars.py
    --output        Output HTML path (default: jax_replay.html)
    --opponent      "random" | "rule_based" (default: rule_based)
    --seed          Integer seed for planet generation (default: 42)
    --episode-steps Max ticks to record (default: 500)
    --ship-speed    Physics constant (default: 6.0)
    --comet-speed   Physics constant (default: 4.0)
    --random        Use a random policy regardless of --checkpoint
    --device        "cuda" | "cpu" (auto-detected if omitted)
    --open          Open the HTML in the default browser after saving (Windows)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch

from model.SAC import P_network
from jax_env import JaxVecEnvAdapter
from jax_env.vec_env import VectorizedEnv
from jax_env.env_types import GameState
from jax_env.constants import MAX_PLANETS, MAX_FLEETS, MAX_COMET_GROUPS
from train_orbit_wars import load_config


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_policy(checkpoint_path: str, device: str) -> P_network:
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Infer network dims from the saved state dict
    policy_sd = ck.get("policy", ck)
    try:
        state_dim  = ck.get("state_dim",  13)
        action_dim = ck.get("action_dim", 4)
        max_planets = ck.get("max_planets", 40)
        max_fleets  = ck.get("max_fleets", 100)
    except Exception:
        state_dim, action_dim, max_planets, max_fleets = 13, 4, 40, 100

    policy = P_network(
        state_dim=state_dim, action_dim=action_dim,
        max_planets=max_planets, max_fleets=max_fleets,
    ).to(device)
    policy.load_state_dict(policy_sd)
    policy.eval()
    return policy


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner  (uses VectorizedEnv directly — no auto-reset on done)
# ─────────────────────────────────────────────────────────────────────────────

def run_jax_episode(
    policy,                  # callable obs[1,seq,13]→action[1,NMP,4], or None for random
    opponent:    str = "rule_based",
    seed:        int = 42,
    episode_steps: int = 500,
    ship_speed:  float = 6.0,
    comet_speed: float = 4.0,
    device:      str = "cpu",
) -> list[GameState]:
    """
    Run a single JAX environment to completion and return the sequence of
    GameState snapshots (one per game tick, including the terminal state).

    All states are squeezed to remove the num_envs=1 batch dimension.
    """
    # We use the raw VectorizedEnv (no auto-reset) so we can capture the
    # terminal state.  A JaxVecEnvAdapter with num_envs=1 is kept alongside
    # purely for its obs-extraction and action-decoding helpers.
    jax_env = VectorizedEnv(
        num_envs=1,
        num_players=2,
        episode_steps=episode_steps,
        ship_speed=ship_speed,
        comet_speed=comet_speed,
    )
    helper = JaxVecEnvAdapter(
        num_envs=1,
        num_players=2,
        episode_steps=episode_steps,
        ship_speed=ship_speed,
        comet_speed=comet_speed,
        opponent=opponent,
    )

    # Reset — some seeds generate too many planets for MAX_PLANETS; retry a few
    # times with offset seeds before giving up.
    for _seed_offset in range(20):
        try:
            batched_state = jax_env.reset([seed + _seed_offset])
            break
        except AssertionError:
            if _seed_offset == 19:
                raise
    helper._state = batched_state

    def squeeze(s: GameState) -> GameState:
        return jax.tree_util.tree_map(lambda x: x[0], s)

    states: list[GameState] = [squeeze(batched_state)]

    for _ in range(episode_steps):
        # ── Player-0 action ───────────────────────────────────────────────────
        if policy is None:
            action_np = np.random.uniform(-1, 1, (1, 40, 4)).astype(np.float32)
        else:
            obs_np = helper._extract_obs(batched_state)        # [1, seq, 13]
            obs_t  = torch.FloatTensor(obs_np).to(device)
            with torch.no_grad():
                act_t, _, _ = policy.sample(obs_t)
            action_np = act_t.cpu().numpy()                    # [1, NMP, 4]

        # ── Decode + generate opponent actions ────────────────────────────────
        helper._state = batched_state
        jax_acts = helper._decode_actions_batch(action_np)     # [1, NP, JAMP, 2]

        # ── Step (no auto-reset) ──────────────────────────────────────────────
        batched_state, _, dones_jax = jax_env.step(batched_state, jax_acts)
        states.append(squeeze(batched_state))

        if bool(jnp.any(dones_jax)):
            break

    return states


# ─────────────────────────────────────────────────────────────────────────────
# Fleet ID tracker  (maps circular-buffer slots → stable kaggle fleet IDs)
# ─────────────────────────────────────────────────────────────────────────────

class _FleetIdTracker:
    def __init__(self):
        self._slot_id: dict[int, int] = {}   # slot → current fleet ID
        self._prev:    dict[int, tuple] = {} # slot → (owner, from_planet) last step
        self.next_id: int = 0

    def update(self, active, owner, from_planet):
        """Return {slot: fleet_id} for currently active fleet slots."""
        new_slot_id: dict[int, int] = {}
        new_prev:    dict[int, tuple] = {}

        for i, (is_active, ow, fp) in enumerate(zip(active, owner, from_planet)):
            if not is_active:
                continue
            key = (int(ow), int(fp))
            if i in self._slot_id and self._prev.get(i) == key:
                new_slot_id[i] = self._slot_id[i]   # same fleet, keep ID
            else:
                new_slot_id[i] = self.next_id        # new fleet in this slot
                self.next_id += 1
            new_prev[i] = key

        self._slot_id = new_slot_id
        self._prev    = new_prev
        return new_slot_id


# ─────────────────────────────────────────────────────────────────────────────
# GameState  →  kaggle observation dict
# ─────────────────────────────────────────────────────────────────────────────

def _state_to_obs(
    state:           GameState,
    player:          int,
    initial_planets: list,
    fleet_tracker:   _FleetIdTracker,
) -> dict:
    p = state.planets
    f = state.fleets
    c = state.comets

    # ── Planets ───────────────────────────────────────────────────────────────
    p_active = np.asarray(p.active, dtype=bool)
    planets = [
        [int(i), int(p.owner[i]),
         float(p.x[i]), float(p.y[i]),
         float(p.radius[i]), float(p.ships[i]),
         int(p.production[i])]
        for i in range(len(p_active)) if p_active[i]
    ]

    # ── Fleets ────────────────────────────────────────────────────────────────
    f_active     = np.asarray(f.active,      dtype=bool)
    f_owner      = np.asarray(f.owner,       dtype=np.int32)
    f_from_planet = np.asarray(f.from_planet, dtype=np.int32)

    slot_id_map = fleet_tracker.update(f_active, f_owner, f_from_planet)
    fleets = [
        [slot_id_map[i], int(f_owner[i]),
         float(f.x[i]), float(f.y[i]),
         float(f.angle[i]), int(f_from_planet[i]),
         float(f.ships[i])]
        for i in slot_id_map
    ]

    # ── Comets ────────────────────────────────────────────────────────────────
    path_indices = np.asarray(c.path_indices, dtype=np.int32)
    path_lengths = np.asarray(c.path_lengths, dtype=np.int32)
    planet_slots = np.asarray(c.planet_slots, dtype=np.int32)
    paths_arr    = np.asarray(c.paths,        dtype=np.float32)

    comets = []
    comet_planet_ids = []
    for g in range(MAX_COMET_GROUPS):
        if path_indices[g] < 0 or path_indices[g] >= path_lengths[g]:
            continue
        path_len = int(path_lengths[g])
        pids = [int(planet_slots[g, k]) for k in range(4)]
        comet_paths = [
            [[float(paths_arr[g, k, t, 0]), float(paths_arr[g, k, t, 1])]
             for t in range(path_len)]
            for k in range(4)
        ]
        comets.append({
            "planet_ids": pids,
            "paths": comet_paths,
            "path_index": int(path_indices[g]),
        })
        comet_planet_ids.extend(pids)

    return {
        "remainingOverageTime": 60,
        "step":              int(state.step),
        "player":            player,
        "angular_velocity":  float(state.angular_velocity),
        "planets":           planets,
        "fleets":            fleets,
        "initial_planets":   initial_planets,
        "next_fleet_id":     fleet_tracker.next_id,
        "comets":            comets,
        "comet_planet_ids":  comet_planet_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build kaggle steps list
# ─────────────────────────────────────────────────────────────────────────────

def _build_kaggle_steps(states: list[GameState], final_won: bool | None = None) -> list:
    """
    Convert a sequence of squeezed GameStates into the list-of-lists-of-dicts
    format expected by kaggle_environments.make(steps=...).

    Each entry is [agent_0_dict, agent_1_dict] for a 2-player game.

    final_won
        If provided, the last step's rewards are synthesised from this outcome
        instead of reading state.rewards.  Use this when the states were
        recorded before the terminal step (e.g. from training) so the last
        state still has rewards=[0,0].
    """
    num_players = 2

    fleet_tracker = _FleetIdTracker()

    # initial_planets comes from the very first state
    p0 = states[0].planets
    act0 = np.asarray(p0.active, dtype=bool)
    initial_planets = [
        [int(i), int(p0.owner[i]),
         float(p0.x[i]), float(p0.y[i]),
         float(p0.radius[i]), float(p0.ships[i]),
         int(p0.production[i])]
        for i in range(len(act0)) if act0[i]
    ]

    kaggle_steps = []
    for t, state in enumerate(states):
        is_last = (t == len(states) - 1)
        done    = bool(state.done) or is_last

        if done and final_won is not None:
            rewards = np.array([1.0, -1.0] if final_won else [-1.0, 1.0],
                               dtype=np.float32)
        else:
            rewards = np.asarray(state.rewards, dtype=np.float32)

        # Observation is the same for both players (shared world state); only
        # the 'player' field differs.  Build once, copy for player 1.
        obs_p0 = _state_to_obs(state, 0, initial_planets, fleet_tracker)

        step_agents = []
        for pid in range(num_players):
            obs = dict(obs_p0)
            obs["player"] = pid

            agent_state = {
                "action": [],
                "reward": float(rewards[pid]) if done else 0.0,
                "info":   {},
                "observation": obs,
                "status": "DONE" if done else "ACTIVE",
            }
            step_agents.append(agent_state)

        kaggle_steps.append(step_agents)

    return kaggle_steps


# ─────────────────────────────────────────────────────────────────────────────
# Public render entry-point (used by train_orbit_wars subprocess)
# ─────────────────────────────────────────────────────────────────────────────

def render_states_to_html(
    states:     list[GameState],
    output_path: str,
    final_won:  bool | None = None,
    width:      int = 800,
    height:     int = 800,
):
    """Convert a pre-recorded GameState sequence directly to an HTML replay.

    Parameters
    ----------
    states
        Sequence of squeezed (no batch dim) GameState objects as produced by
        the training loop recorder.  numpy arrays are fine; JAX arrays also work.
    output_path
        Where to write the HTML file.
    final_won
        Pass True/False when the states were recorded before the terminal step
        so that the last frame shows the correct win/loss reward.  Leave None
        when states already contain the terminal done state.
    """
    kaggle_steps = _build_kaggle_steps(states, final_won=final_won)
    save_html(kaggle_steps, output_path, width=width, height=height)


# ─────────────────────────────────────────────────────────────────────────────
# HTML export
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_kaggle_environments():
    """Import kaggle_environments, searching conda envs if not on sys.path."""
    try:
        import kaggle_environments  # noqa: F401
        return
    except ImportError:
        pass
    import glob
    for _p in sorted(glob.glob(
        os.path.expanduser("~/miniforge*/envs/*/lib/python*/site-packages")
    )):
        if os.path.isdir(os.path.join(_p, "kaggle_environments")):
            sys.path.insert(0, _p)
            return
    raise ImportError(
        "kaggle_environments not found. Install it or activate an environment that has it."
    )


def save_html(kaggle_steps: list, output_path: str, width: int = 800, height: int = 800):
    _ensure_kaggle_environments()
    from kaggle_environments import make

    env = make("orbit_wars", steps=kaggle_steps, debug=False)
    html = env.render(mode="html", width=width, height=height)
    with open(output_path, "w") as fh:
        fh.write(html)
    print(f"Replay saved → {os.path.abspath(output_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Render a JAX episode as HTML")

    # ── States-file mode (called by train_orbit_wars) ─────────────────────────
    parser.add_argument("--states-file",   default=None,
                        help="Pickle file of recorded GameState sequence "
                             "(written by train_orbit_wars). When given, all "
                             "episode-running args are ignored.")
    parser.add_argument("--final-won",     default=None, choices=["true", "false"],
                        help="Win/loss flag for the states-file terminal frame.")

    # ── Standalone / checkpoint mode ──────────────────────────────────────────
    parser.add_argument("--checkpoint",    default=None,             help="Path to .pt checkpoint")
    parser.add_argument("--output",        default="jax_replay.html",help="Output HTML path")
    parser.add_argument("--opponent",      default="rule_based",     choices=["random", "rule_based"])
    parser.add_argument("--seed",          type=int, default=42,     help="Episode seed")
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--ship-speed",    type=float, default=6.0)
    parser.add_argument("--comet-speed",   type=float, default=4.0)
    parser.add_argument("--random",        action="store_true",      help="Use random policy (no checkpoint needed)")
    parser.add_argument("--device",        default=None)
    parser.add_argument("--open",          action="store_true",      help="Open HTML in browser (Windows)")
    args = parser.parse_args()

    # ── States-file path: load recorded states, skip episode running ──────────
    if args.states_file is not None:
        import pickle
        with open(args.states_file, "rb") as fh:
            states = pickle.load(fh)
        final_won = {"true": True, "false": False}.get(args.final_won)
        print(f"Loaded {len(states)} states from {args.states_file}  "
              f"won={final_won}")
        print("Building HTML render ...")
        render_states_to_html(states, args.output, final_won=final_won)
        if args.open:
            subprocess.run(["cmd.exe", "/c", "start",
                            os.path.abspath(args.output)])
        return

    # ── Standalone path: run a fresh episode ─────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.random or args.checkpoint is None:
        policy = None
        print("Using random policy (no checkpoint).")
    else:
        policy = _load_policy(args.checkpoint, device)
        print(f"Loaded policy from {args.checkpoint}")

    print(f"Running episode  seed={args.seed}  opponent={args.opponent}  "
          f"max_steps={args.episode_steps} ...")
    states = run_jax_episode(
        policy=policy,
        opponent=args.opponent,
        seed=args.seed,
        episode_steps=args.episode_steps,
        ship_speed=args.ship_speed,
        comet_speed=args.comet_speed,
        device=device,
    )
    print(f"Episode finished after {len(states) - 1} steps.")

    final = states[-1]
    rewards = np.asarray(final.rewards)
    if rewards[0] > 0:
        print("Result: Player 0 (your agent) WON")
    elif rewards[0] < 0:
        print("Result: Player 0 (your agent) LOST")
    else:
        print("Result: DRAW (or game still in progress at step limit)")

    print("Converting to kaggle format ...")
    render_states_to_html(states, args.output)

    if args.open:
        subprocess.run(["cmd.exe", "/c", "start", os.path.abspath(args.output)])


if __name__ == "__main__":
    main()
