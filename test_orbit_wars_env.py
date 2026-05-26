"""
Tests for env/orbit_wars.py — runs without the kaggle_environments package.

Each test section covers one exported symbol:
    _obs_to_arrays, _swap_perspective, encode_obs_as_player,
    decode_action, compute_reward_for_player, OrbitWarsEnv (mocked)
"""

import sys
import math
import types
import unittest
import numpy as np
from types import SimpleNamespace

# ── make model/ importable from project root ──────────────────────────────────
sys.path.insert(0, "/mnt/d/ML/Kaggle/StarWars/501-Legion")

from env.orbit_wars import (
    MAX_PLANETS, MAX_FLEETS, STATE_DIM, ACTION_DIM,
    _obs_to_arrays, _swap_perspective,
    encode_obs_as_player, decode_action, compute_reward_for_player,
    OrbitWarsEnv,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_planet(id_, owner, x, y, radius=5.0, ships=10, production=1):
    return SimpleNamespace(id=id_, owner=owner, x=x, y=y,
                           radius=radius, ships=ships, production=production)


def _make_fleet(id_, owner, x, y, angle=0.0, from_planet_id=0, ships=5):
    return SimpleNamespace(id=id_, owner=owner, x=x, y=y, angle=angle,
                           from_planet_id=from_planet_id, ships=ships)


def _make_obs(n_planets=4, n_fleets=2, angular_velocity=0.01,
              comet_planet_ids=None):
    """Create a minimal SimpleNamespace observation with realistic values."""
    planets = [
        _make_planet(0, 0,  60.0, 50.0, radius=5.0,  ships=20, production=2),
        _make_planet(1, 1,  40.0, 50.0, radius=5.0,  ships=15, production=2),
        _make_planet(2, -1, 50.0, 70.0, radius=3.0,  ships=0,  production=1),
        _make_planet(3, -1, 50.0, 30.0, radius=3.0,  ships=0,  production=1),
    ][:n_planets]
    fleets = [
        _make_fleet(0, 0, 62.0, 50.0, angle=0.1, from_planet_id=0, ships=5),
        _make_fleet(1, 1, 38.0, 50.0, angle=3.2, from_planet_id=1, ships=3),
    ][:n_fleets]
    return SimpleNamespace(
        planets=planets,
        fleets=fleets,
        angular_velocity=angular_velocity,
        comet_planet_ids=comet_planet_ids or [],
    )


def _make_obs_dict(n_planets=3, angular_velocity=0.02):
    """Dict-access observation (alternate kaggle format)."""
    return {
        "planets": [
            {"id": 0, "owner": 0, "x": 60.0, "y": 50.0, "radius": 5.0, "ships": 20, "production": 2},
            {"id": 1, "owner": 1, "x": 40.0, "y": 50.0, "radius": 5.0, "ships": 10, "production": 2},
            {"id": 2, "owner": None, "x": 50.0, "y": 70.0, "radius": 3.0, "ships": 0, "production": 1},
        ][:n_planets],
        "fleets": [],
        "angular_velocity": angular_velocity,
        "comet_planet_ids": [2],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. _obs_to_arrays
# ─────────────────────────────────────────────────────────────────────────────

class TestObsToArrays(unittest.TestCase):

    def test_attribute_access_shapes(self):
        obs = _make_obs(n_planets=4, n_fleets=2)
        planets, fleets, av, omega, comets = _obs_to_arrays(obs)
        self.assertEqual(planets.shape, (4, 7))
        self.assertEqual(fleets.shape,  (2, 7))
        self.assertAlmostEqual(av, 0.01)
        self.assertAlmostEqual(omega, 0.01)   # alias
        self.assertEqual(comets.dtype, np.int32)

    def test_dict_access_shapes(self):
        obs = _make_obs_dict(n_planets=3)
        planets, fleets, av, omega, comets = _obs_to_arrays(obs)
        self.assertEqual(planets.shape, (3, 7))
        self.assertEqual(fleets.shape,  (0, 7))
        self.assertAlmostEqual(av, 0.02)

    def test_none_owner_maps_to_minus_one(self):
        obs = _make_obs_dict(n_planets=3)
        planets, _, _, _, _ = _obs_to_arrays(obs)
        self.assertEqual(int(planets[2, 1]), -1)  # owner=None → -1

    def test_planet_column_order(self):
        obs = _make_obs(n_planets=1, n_fleets=0)
        planets, _, _, _, _ = _obs_to_arrays(obs)
        # [id, owner, x, y, radius, ships, production]
        self.assertEqual(int(planets[0, 0]), 0)    # id
        self.assertEqual(int(planets[0, 1]), 0)    # owner = player 0
        self.assertAlmostEqual(float(planets[0, 2]), 60.0)  # x
        self.assertAlmostEqual(float(planets[0, 5]), 20.0)  # ships

    def test_fleet_column_order(self):
        obs = _make_obs(n_planets=2, n_fleets=1)
        _, fleets, _, _, _ = _obs_to_arrays(obs)
        # [id, owner, x, y, angle, from_planet_id, ships]
        self.assertEqual(int(fleets[0, 0]), 0)           # id
        self.assertEqual(int(fleets[0, 1]), 0)           # owner
        self.assertAlmostEqual(float(fleets[0, 4]), 0.1, places=5)  # angle

    def test_empty_fleets(self):
        obs = _make_obs(n_planets=2, n_fleets=0)
        _, fleets, _, _, _ = _obs_to_arrays(obs)
        self.assertEqual(fleets.shape, (0, 7))

    def test_comet_ids_parsed(self):
        obs = _make_obs_dict(n_planets=3)
        _, _, _, _, comets = _obs_to_arrays(obs)
        self.assertIn(2, comets)

    def test_numpy_array_planets_passthrough(self):
        raw = np.random.rand(5, 7).astype(np.float32)
        obs = SimpleNamespace(planets=raw, fleets=np.empty((0, 7), dtype=np.float32),
                              angular_velocity=0.0, comet_planet_ids=np.array([], dtype=np.int32))
        planets, _, _, _, _ = _obs_to_arrays(obs)
        np.testing.assert_array_almost_equal(planets, raw)


# ─────────────────────────────────────────────────────────────────────────────
# 2. _swap_perspective
# ─────────────────────────────────────────────────────────────────────────────

class TestSwapPerspective(unittest.TestCase):

    def _planet_array(self):
        # owners: 0, 1, -1, 2
        p = np.array([
            [0, 0,  60, 50, 5, 20, 2],
            [1, 1,  40, 50, 5, 15, 2],
            [2, -1, 50, 70, 3,  0, 1],
            [3, 2,  50, 30, 3,  0, 1],
        ], dtype=np.float32)
        f = np.array([
            [0, 0, 62, 50, 0.1, 0, 5],
            [1, 1, 38, 50, 3.2, 1, 3],
        ], dtype=np.float32)
        return p, f

    def test_player0_identity(self):
        p, f = self._planet_array()
        sp, sf = _swap_perspective(p, f, player_id=0)
        np.testing.assert_array_equal(sp[:, 1], p[:, 1])
        np.testing.assert_array_equal(sf[:, 1], f[:, 1])

    def test_player1_swap_owners(self):
        p, f = self._planet_array()
        sp, sf = _swap_perspective(p, f, player_id=1)
        # owner 0 → 1
        self.assertEqual(int(sp[0, 1]), 1)
        # owner 1 → 0
        self.assertEqual(int(sp[1, 1]), 0)
        # owner -1 unchanged
        self.assertEqual(int(sp[2, 1]), -1)
        # owner 2 unchanged
        self.assertEqual(int(sp[3, 1]), 2)

    def test_player1_fleet_swap(self):
        p, f = self._planet_array()
        _, sf = _swap_perspective(p, f, player_id=1)
        self.assertEqual(int(sf[0, 1]), 1)   # was 0
        self.assertEqual(int(sf[1, 1]), 0)   # was 1

    def test_returns_copy(self):
        p, f = self._planet_array()
        sp, sf = _swap_perspective(p, f, player_id=0)
        sp[0, 1] = 99
        self.assertNotEqual(int(p[0, 1]), 99)   # original untouched

    def test_double_swap_is_identity(self):
        p, f = self._planet_array()
        sp1, sf1 = _swap_perspective(p, f, player_id=1)
        sp2, sf2 = _swap_perspective(sp1, sf1, player_id=1)
        np.testing.assert_array_equal(sp2[:, 1], p[:, 1])
        np.testing.assert_array_equal(sf2[:, 1], f[:, 1])


# ─────────────────────────────────────────────────────────────────────────────
# 3. encode_obs_as_player
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeObsAsPlayer(unittest.TestCase):

    def setUp(self):
        from model.SAC import Encoder
        self.encoder = Encoder(max_planets=MAX_PLANETS, max_fleets=MAX_FLEETS)

    def _initial_planets(self, obs):
        planets_np, _, _, _, _ = _obs_to_arrays(obs)
        sp, _ = _swap_perspective(planets_np,
                                  np.empty((0, 7), dtype=np.float32), 0)
        return sp

    def test_output_shape_player0(self):
        obs = _make_obs()
        init_p = self._initial_planets(obs)
        state = encode_obs_as_player(self.encoder, obs, init_p, player_id=0)
        self.assertEqual(state.shape, (MAX_PLANETS + MAX_FLEETS, STATE_DIM))
        self.assertEqual(state.dtype, np.float32)

    def test_output_shape_player1(self):
        obs = _make_obs()
        init_p = self._initial_planets(obs)
        state = encode_obs_as_player(self.encoder, obs, init_p, player_id=1)
        self.assertEqual(state.shape, (MAX_PLANETS + MAX_FLEETS, STATE_DIM))

    def test_padded_rows_are_zero(self):
        obs = _make_obs(n_planets=2, n_fleets=0)
        init_p = self._initial_planets(obs)
        state = encode_obs_as_player(self.encoder, obs, init_p, player_id=0)
        # Rows beyond n_planets should be zero-padded
        np.testing.assert_array_equal(state[2:MAX_PLANETS], 0.0)

    def test_different_players_give_different_states(self):
        obs = _make_obs()
        init_p = self._initial_planets(obs)
        s0 = encode_obs_as_player(self.encoder, obs, init_p, player_id=0)
        s1 = encode_obs_as_player(self.encoder, obs, init_p, player_id=1)
        # Owner fields differ after perspective swap
        self.assertFalse(np.allclose(s0, s1))

    def test_no_nan_in_output(self):
        obs = _make_obs(n_planets=4, n_fleets=2)
        init_p = self._initial_planets(obs)
        state = encode_obs_as_player(self.encoder, obs, init_p, player_id=0)
        self.assertFalse(np.any(np.isnan(state)))

    def test_time_step_changes_state(self):
        obs = _make_obs()
        init_p = self._initial_planets(obs)
        s0 = encode_obs_as_player(self.encoder, obs, init_p, player_id=0, time_step=0)
        s100 = encode_obs_as_player(self.encoder, obs, init_p, player_id=0, time_step=100)
        self.assertFalse(np.allclose(s0, s100))


# ─────────────────────────────────────────────────────────────────────────────
# 4. decode_action
# ─────────────────────────────────────────────────────────────────────────────

class TestDecodeAction(unittest.TestCase):

    def _planets(self):
        # [id, owner, x, y, radius, ships, production]
        return np.array([
            [0, 0, 60.0, 50.0, 5.0, 20.0, 2.0],  # ours, 20 ships
            [1, 1, 40.0, 50.0, 5.0, 15.0, 2.0],  # opponent
            [2, 0, 50.0, 70.0, 3.0,  1.0, 1.0],  # ours but only 1 ship → skip
            [3, -1, 50.0, 30.0, 3.0, 0.0, 1.0],  # neutral
        ], dtype=np.float32)

    def _action(self, launch_idx=(0,)):
        """Random action with launch gate open for specified planet indices."""
        act = np.full((MAX_PLANETS, ACTION_DIM), -0.5, dtype=np.float32)
        for i in launch_idx:
            act[i, 0] =  0.8   # gate open
            act[i, 1] =  1.0   # sin
            act[i, 2] =  0.0   # cos  → angle = π/2 ≈ 1.5708
            act[i, 3] =  0.6   # ship fraction → (0.6+1)/2=0.8 → 0.8*19≈15 ships
        return act

    def test_returns_list(self):
        moves = decode_action(self._action(), self._planets(), 0.01)
        self.assertIsInstance(moves, list)

    def test_only_launches_from_own_planets(self):
        act = self._action(launch_idx=(0, 1, 3))   # open gate for opponent & neutral too
        moves = decode_action(act, self._planets(), 0.01)
        planet_ids = [m[0] for m in moves]
        self.assertIn(0, planet_ids)
        self.assertNotIn(1, planet_ids)   # opponent planet
        self.assertNotIn(3, planet_ids)   # neutral planet

    def test_planet_with_one_ship_skipped(self):
        act = self._action(launch_idx=(2,))
        moves = decode_action(act, self._planets(), 0.01)
        planet_ids = [m[0] for m in moves]
        self.assertNotIn(2, planet_ids)

    def test_closed_gate_skips(self):
        act = self._action(launch_idx=())   # all gates closed
        moves = decode_action(act, self._planets(), 0.01)
        self.assertEqual(len(moves), 0)

    def test_angle_decoding(self):
        act = self._action(launch_idx=(0,))
        act[0, 1] = 1.0   # sin(90°)
        act[0, 2] = 0.0   # cos(90°)
        moves = decode_action(act, self._planets(), 0.01)
        self.assertEqual(len(moves), 1)
        pid, angle, ships = moves[0]
        self.assertAlmostEqual(angle, math.pi / 2, places=4)

    def test_ship_count_at_least_one(self):
        act = self._action(launch_idx=(0,))
        act[0, 3] = -1.0   # fraction → 0 → still sends 1
        moves = decode_action(act, self._planets(), 0.01)
        self.assertGreaterEqual(moves[0][2], 1)

    def test_ship_count_leaves_one_on_planet(self):
        act = self._action(launch_idx=(0,))
        act[0, 3] = 1.0   # max fraction
        planets = self._planets()
        ships_on_planet = int(planets[0, 5])   # 20
        moves = decode_action(act, planets, 0.01)
        self.assertLessEqual(moves[0][2], ships_on_planet - 1)

    def test_move_format(self):
        moves = decode_action(self._action(), self._planets(), 0.01)
        self.assertGreater(len(moves), 0)
        pid, angle, ships = moves[0]
        self.assertIsInstance(pid, int)
        self.assertIsInstance(angle, float)
        self.assertIsInstance(ships, int)

    def test_handles_all_zero_action(self):
        act = np.zeros((MAX_PLANETS, ACTION_DIM), dtype=np.float32)
        moves = decode_action(act, self._planets(), 0.01)
        # gate at 0.0 → not > 0 → no launches
        self.assertEqual(len(moves), 0)

    def test_handles_empty_planets(self):
        empty = np.empty((0, 7), dtype=np.float32)
        act = np.zeros((MAX_PLANETS, ACTION_DIM), dtype=np.float32)
        moves = decode_action(act, empty, 0.0)
        self.assertEqual(moves, [])


# ─────────────────────────────────────────────────────────────────────────────
# 5. compute_reward_for_player
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeReward(unittest.TestCase):

    def _obs_from_planets(self, planet_rows):
        """Build a minimal obs from a list of [id,owner,x,y,r,ships,prod] rows."""
        planets = [
            SimpleNamespace(id=int(r[0]), owner=int(r[1]) if r[1] != -1 else -1,
                            x=r[2], y=r[3], radius=r[4], ships=r[5], production=r[6])
            for r in planet_rows
        ]
        return SimpleNamespace(planets=planets, fleets=[],
                               angular_velocity=0.0, comet_planet_ids=[])

    def test_no_change_zero_reward(self):
        planets = [[0, 0, 60, 50, 5, 10, 1], [1, 1, 40, 50, 5, 10, 1]]
        obs = self._obs_from_planets(planets)
        r = compute_reward_for_player(obs, obs, player_id=0)
        self.assertAlmostEqual(r, 0.0)

    def test_gaining_planet_positive_reward(self):
        before = [[0, 0, 60, 50, 5, 10, 1], [1, 1, 40, 50, 5, 10, 1]]
        after  = [[0, 0, 60, 50, 5, 10, 1], [1, 0, 40, 50, 5,  5, 1]]  # planet 1 captured
        obs     = self._obs_from_planets(before)
        new_obs = self._obs_from_planets(after)
        r = compute_reward_for_player(obs, new_obs, player_id=0)
        self.assertGreater(r, 0.0)

    def test_losing_planet_negative_reward(self):
        before = [[0, 0, 60, 50, 5, 10, 1], [1, 1, 40, 50, 5, 10, 1]]
        after  = [[0, 1, 60, 50, 5,  5, 1], [1, 1, 40, 50, 5, 10, 1]]  # planet 0 lost
        obs     = self._obs_from_planets(before)
        new_obs = self._obs_from_planets(after)
        r = compute_reward_for_player(obs, new_obs, player_id=0)
        self.assertLess(r, 0.0)

    def test_win_terminal_bonus(self):
        before = [[0, 0, 60, 50, 5, 10, 1], [1, 1, 40, 50, 5,  1, 1]]
        after  = [[0, 0, 60, 50, 5, 12, 1], [1, 0, 40, 50, 5,  0, 1]]  # opponent wiped
        obs     = self._obs_from_planets(before)
        new_obs = self._obs_from_planets(after)
        r = compute_reward_for_player(obs, new_obs, player_id=0)
        self.assertGreater(r, 50.0)   # terminal bonus dominates

    def test_loss_terminal_bonus(self):
        before = [[0, 0, 60, 50, 5,  1, 1], [1, 1, 40, 50, 5, 10, 1]]
        after  = [[0, 1, 60, 50, 5,  0, 1], [1, 1, 40, 50, 5, 12, 1]]  # we are wiped
        obs     = self._obs_from_planets(before)
        new_obs = self._obs_from_planets(after)
        r = compute_reward_for_player(obs, new_obs, player_id=0)
        self.assertLess(r, -50.0)

    def test_returns_float(self):
        planets = [[0, 0, 60, 50, 5, 10, 1]]
        obs = self._obs_from_planets(planets)
        r = compute_reward_for_player(obs, obs, player_id=0)
        self.assertIsInstance(r, float)

    def test_symmetric_perspectives(self):
        before = [[0, 0, 60, 50, 5, 10, 1], [1, 1, 40, 50, 5, 10, 1]]
        after  = [[0, 0, 60, 50, 5, 12, 1], [1, 1, 40, 50, 5,  8, 1]]
        obs     = self._obs_from_planets(before)
        new_obs = self._obs_from_planets(after)
        r0 = compute_reward_for_player(obs, new_obs, player_id=0)
        r1 = compute_reward_for_player(obs, new_obs, player_id=1)
        self.assertAlmostEqual(r0, -r1, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# 6. OrbitWarsEnv (mocked kaggle)
# ─────────────────────────────────────────────────────────────────────────────

def _build_mock_kaggle(step_returns=None):
    """
    Build a fake kaggle_environments module and inject it into sys.modules.
    Returns the mock trainer so tests can inspect calls.
    """
    obs = _make_obs(n_planets=4, n_fleets=2)
    if step_returns is None:
        step_returns = [(obs, 0.0, False, {})] * 10 + [(obs, 1.0, True, {})]

    call_log = {"step": [], "reset": 0}

    class MockTrainer:
        def __init__(self):
            self._step_idx = 0

        def reset(self):
            call_log["reset"] += 1
            self._step_idx = 0
            return obs

        def step(self, action):
            call_log["step"].append(action)
            result = step_returns[min(self._step_idx, len(step_returns) - 1)]
            self._step_idx += 1
            return result

    class MockKaggleEnv:
        def train(self, agents):
            return MockTrainer()

        def render(self, mode="ansi"):
            return "<mock render>"

    mock_trainer = MockTrainer()

    def mock_make(name, debug=False):
        return MockKaggleEnv()

    kaggle_mod = types.ModuleType("kaggle_environments")
    kaggle_mod.make = mock_make
    sys.modules["kaggle_environments"] = kaggle_mod

    return call_log


class TestOrbitWarsEnv(unittest.TestCase):

    def setUp(self):
        from model.SAC import Encoder
        self.encoder = Encoder(max_planets=MAX_PLANETS, max_fleets=MAX_FLEETS)
        _build_mock_kaggle()

    def _make_env(self, **kwargs):
        return OrbitWarsEnv(opponent="random", encoder=self.encoder, **kwargs)

    def test_spaces(self):
        env = self._make_env()
        self.assertEqual(env.observation_space.shape, (MAX_PLANETS + MAX_FLEETS, STATE_DIM))
        self.assertEqual(env.action_space.shape, (MAX_PLANETS, ACTION_DIM))

    def test_reset_returns_correct_shape(self):
        env = self._make_env()
        obs, info = env.reset()
        self.assertEqual(obs.shape, (MAX_PLANETS + MAX_FLEETS, STATE_DIM))
        self.assertIsInstance(info, dict)

    def test_reset_dtype(self):
        env = self._make_env()
        obs, _ = env.reset()
        self.assertEqual(obs.dtype, np.float32)

    def test_step_returns_correct_types(self):
        env = self._make_env()
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(obs.shape, (MAX_PLANETS + MAX_FLEETS, STATE_DIM))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_truncation_at_max_steps(self):
        env = self._make_env(max_steps=3)
        env.reset()
        action = env.action_space.sample()
        for _ in range(2):
            _, _, terminated, truncated, _ = env.step(action)
            self.assertFalse(truncated)
        _, _, terminated, truncated, _ = env.step(action)
        self.assertTrue(truncated)

    def test_step_before_reset_raises(self):
        env = self._make_env()
        with self.assertRaises(AssertionError):
            env.step(env.action_space.sample())

    def test_multiple_resets(self):
        env = self._make_env()
        for _ in range(3):
            obs, _ = env.reset()
            self.assertEqual(obs.shape, (MAX_PLANETS + MAX_FLEETS, STATE_DIM))

    def test_action_space_sample_valid(self):
        env = self._make_env()
        env.reset()
        for _ in range(5):
            action = env.action_space.sample()
            self.assertTrue(env.action_space.contains(action))

    def test_obs_no_nan(self):
        env = self._make_env()
        obs, _ = env.reset()
        self.assertFalse(np.any(np.isnan(obs)))
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        self.assertFalse(np.any(np.isnan(obs)))

    def test_close(self):
        env = self._make_env()
        env.reset()
        env.close()
        self.assertIsNone(env._trainer)

    def test_player1_perspective(self):
        env = OrbitWarsEnv(opponent="random", player_id=1, encoder=self.encoder)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (MAX_PLANETS + MAX_FLEETS, STATE_DIM))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
