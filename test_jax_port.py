"""Parity tests: JAX (on-device) opponents / reward / obs vs the NumPy versions.

Confirms the JAX ports in jax_env/jax_opponents.py, jax_env/jax_reward.py and
jax_env/jax_obs.py reproduce the host-NumPy results in agents/vec_opponents.py
and JaxVecEnvAdapter, so moving them on-device does not change training dynamics.
"""
import importlib.util as _ilu
import os
import numpy as np
import jax
import jax.numpy as jnp

from jax_env import VectorizedEnv, JaxVecEnvAdapter
from jax_env import jax_opponents as JO
from jax_env import jax_obs as JOBS
from jax_env.jax_reward import build_reward_fn

# numpy opponents (path-loaded, as the adapter does)
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents", "vec_opponents.py")
_s = _ilu.spec_from_file_location("orbit_vec_opponents", _p)
VO = _ilu.module_from_spec(_s); _s.loader.exec_module(VO)

NMP = 40
REWARD = [
    {"scheme": "ShipGrowth", "ship_scale": 0.02, "loss_scale": 3.0},
    {"scheme": "ProductionPlanetDelta", "planet_scale": 1.0, "loss_scale": 5.0},
    {"scheme": "TimeDecayWinBonus", "win_bonus": 5.0},
]
# A config that exercises every ported component (for a broader reward check).
REWARD_ALL = [
    {"scheme": "RelativeShipAdvantage", "ship_scale": 0.01},
    {"scheme": "RelativeProductionAdvantage", "planet_scale": 0.5},
    {"scheme": "ShipGrowth", "ship_scale": 0.02, "loss_scale": 3.0},
    {"scheme": "ProductionPlanetDelta", "planet_scale": 1.0, "loss_scale": 5.0, "ship_scale": 0.1},
    {"scheme": "ProximityCaptureBonus", "scale": 2.0, "ref_dist": 25.0},
    {"scheme": "AbsoluteHoldings", "ship_scale": 0.01, "planet_scale": 0.5},
    {"scheme": "FleetLaunchPenalty", "ship_scale": 0.5},
    {"scheme": "StepPenalty", "weight": 0.1},
    {"scheme": "TerminalWinBonus", "win_bonus": 10.0},
    {"scheme": "TimeDecayWinBonus", "win_bonus": 5.0},
]


def _busy_state(B=8, num_players=2, n_warm=40, seed=0):
    """A mid-game batched state: reset + random launches so fleets are in flight."""
    env = VectorizedEnv(num_envs=B, num_players=num_players, episode_steps=500)
    state = env.reset(np.arange(seed, seed + B))
    rng = np.random.default_rng(seed)
    JAMP = state.planets.x.shape[1]
    for _ in range(n_warm):
        acts = np.zeros((B, num_players, JAMP, 2), np.float32)
        # random launches from all players to populate the board
        acts[:, :, :NMP, 0] = rng.uniform(0, 2 * np.pi, (B, num_players, NMP))
        acts[:, :, :NMP, 1] = rng.integers(0, 30, (B, num_players, NMP))
        state, _, _ = env.step(state, jnp.asarray(acts))
    return state, num_players


def _np_opponent(name, state, num_players):
    """Run the NumPy opponent → engine action buffer [B, NP, JAMP, 2]."""
    B = state.planets.x.shape[0]
    JAMP = state.planets.x.shape[1]
    p_owner = np.asarray(state.planets.owner, np.int32)
    p_x = np.asarray(state.planets.x, np.float32)
    p_y = np.asarray(state.planets.y, np.float32)
    p_r = np.asarray(state.planets.radius, np.float32)
    p_ships = np.asarray(state.planets.ships, np.float32)
    p_prod = np.asarray(state.planets.production, np.float32)
    p_act = np.asarray(state.planets.active, bool)
    p_comet = np.asarray(state.planets.is_comet, bool)
    omega = np.asarray(state.angular_velocity, np.float32).reshape(-1)
    valid = p_act[:, :NMP] & ~p_comet[:, :NMP]
    acts = np.zeros((B, num_players, JAMP, 2), np.float32)
    if name == "greedy":
        VO.greedy_opponent(acts, valid, p_owner, p_x, p_y, p_r, p_ships, p_prod,
                           omega, num_players)
    elif name == "agent1":
        f = state.fleets
        VO.agent1_opponent(acts, valid, p_owner, p_x, p_y, p_r, p_ships, p_prod,
                           omega, num_players,
                           np.asarray(f.owner, np.int32), np.asarray(f.x, np.float32),
                           np.asarray(f.y, np.float32), np.asarray(f.angle, np.float32),
                           np.asarray(f.ships, np.float32), np.asarray(f.active, bool))
    return acts


def _check_opponent(name, num_players=2, n_steps=12):
    """JAX vs NumPy opponent over a rollout.

    The two are equivalent to float precision, but ``np`` and ``jnp`` evaluate the
    lead-intercept transcendentals to slightly different ULPs, so a borderline
    launch threshold (e.g. ``total <= m_avail``) flips on a small fraction of
    (env, planet) slots.  We therefore assert the DISAGREEMENT RATE is tiny and
    that where both launch the angle / ship count agree closely — not bit-equality.
    """
    env = VectorizedEnv(num_envs=8, num_players=num_players, episode_steps=500)
    state = env.reset(np.arange(8))
    JAMP = state.planets.x.shape[1]
    fn = JO.make_opponent_fn(name, num_players)
    rng = np.random.default_rng(0)

    n_slots = n_diff = n_both = ship_off = 0
    ang_max = 0.0
    for _ in range(n_steps):
        np_a = _np_opponent(name, state, num_players)
        jx_a = np.asarray(jax.vmap(lambda s: fn(s, jax.random.PRNGKey(0), 0.0))(state))
        for pid in range(1, num_players):
            lp = np_a[:, pid, :NMP, 1] > 0
            lj = jx_a[:, pid, :NMP, 1] > 0
            n_slots += lp.size
            n_diff  += int((lp != lj).sum())
            both     = lp & lj
            n_both  += int(both.sum())
            if both.any():
                ship_off += int((np_a[:, pid, :NMP, 1][both] != jx_a[:, pid, :NMP, 1][both]).sum())
                ang_max   = max(ang_max,
                                np.abs(np_a[:, pid, :NMP, 0] - jx_a[:, pid, :NMP, 0])[both].max())
        # advance with random launches so the board evolves
        acts = np.zeros((8, num_players, JAMP, 2), np.float32)
        acts[:, :, :NMP, 0] = rng.uniform(0, 2 * np.pi, (8, num_players, NMP))
        acts[:, :, :NMP, 1] = rng.integers(0, 30, (8, num_players, NMP))
        state, _, _ = env.step(state, jnp.asarray(acts))

    diff_rate = n_diff / max(1, n_slots)
    print(f"  {name} {num_players}p: launch-decision disagree {n_diff}/{n_slots} "
          f"({diff_rate*100:.3f}%) | ship mismatches(both-launch) {ship_off}/{n_both} | "
          f"max|angle|={ang_max:.3g}")
    # tiny borderline-flip rate, near-zero ship mismatches, matching angles
    return diff_rate < 0.01 and ship_off <= max(1, n_both // 100) and ang_max < 1e-3


def _check_reward(reward_cfg, label):
    B = 8
    adapter = JaxVecEnvAdapter(num_envs=B, num_players=2, opponent="agent1",
                               reward_cfg=reward_cfg)
    adapter.reset()
    old = adapter._state
    JAMP = old.planets.x.shape[1]
    acts = adapter._build_engine_actions(np.zeros((B, NMP, 2), np.float32))
    new, rew, dones = adapter._jax.step(old, acts)
    r_np = adapter._compute_config_reward(new, np.asarray(rew, np.float32),
                                          np.asarray(dones, bool))
    rfn = build_reward_fn(adapter._reward_cfg, adapter._episode_steps, 2)
    r_jx = np.asarray(jax.vmap(rfn)(old, new))
    diff = np.abs(r_np - r_jx).max()
    print(f"  reward[{label}]: max|Δ|={diff:.3g}   "
          f"(np mean {r_np.mean():.4f}, jx mean {r_jx.mean():.4f})")
    return diff < 1e-3


def _check_obs():
    state, _ = _busy_state()
    adapter = JaxVecEnvAdapter(num_envs=state.planets.x.shape[0], num_players=2,
                               opponent="agent1")
    obs_np = adapter._extract_obs(state)                          # [B, seq, 13]
    obs_jx = np.asarray(JOBS.make_extract_obs(0)(state))
    # planet tokens (first NMP) are positional → compare exactly
    p_diff = np.abs(obs_np[:, :NMP] - obs_jx[:, :NMP]).max()
    # fleet tokens carry no positional meaning → compare as sorted-by-ships rows
    f_np = np.sort(obs_np[:, NMP:], axis=1)
    f_jx = np.sort(obs_jx[:, NMP:], axis=1)
    f_diff = np.abs(f_np - f_jx).max()
    print(f"  obs: max|Δ planets|={p_diff:.3g}   max|Δ fleets(sorted)|={f_diff:.3g}")
    return p_diff < 1e-3 and f_diff < 1e-3


if __name__ == "__main__":
    print(f"JAX backend: {jax.default_backend()}")
    ok = True
    print("Opponent parity (JAX vs NumPy):")
    ok &= _check_opponent("greedy")
    ok &= _check_opponent("agent1")
    ok &= _check_opponent("agent1", num_players=4)
    print("Reward parity:")
    ok &= _check_reward(REWARD, "train_st2")
    ok &= _check_reward(REWARD_ALL, "all-components")
    print("Obs parity:")
    ok &= _check_obs()
    print("\n" + ("ALL PARITY CHECKS PASSED" if ok else "*** PARITY FAILURES ***"))
    raise SystemExit(0 if ok else 1)
