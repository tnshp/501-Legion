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
from agents.agent1 import RuleBasedAgent
# ── make model/ importable from project root ──────────────────────────────────
sys.path.insert(0, "/mnt/d/ML/Kaggle/StarWars/501-Legion")

from env.orbit_wars import (
    _obs_to_arrays, _swap_perspective,
    encode_obs_as_player, decode_action, pairwise_wedge,
    compute_reward_for_player,
    OrbitWarsEnv,
    RewardScheme1, RewardScheme2, RewardScheme3,
    reward_scheme_1, reward_scheme_2, reward_scheme_3,
)

MAX_PLANETS = OrbitWarsEnv.MAX_PLANETS
MAX_FLEETS  = OrbitWarsEnv.MAX_FLEETS
STATE_DIM   = OrbitWarsEnv.STATE_DIM
ACTION_DIM  = OrbitWarsEnv.ACTION_DIM


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
# 4a. pairwise_wedge
# ─────────────────────────────────────────────────────────────────────────────

class TestPairwiseWedge(unittest.TestCase):

    def test_output_shape_d4(self):
        """(n, 4) inputs → (n, n, 6) output  (C(4,2) = 6 independent planes)."""
        a = np.random.rand(5, 4).astype(np.float32)
        out = pairwise_wedge(a, a)
        self.assertEqual(out.shape, (5, 5, 6))

    def test_self_wedge_diagonal_zero(self):
        """wedge(v, v) = 0 for all v  (antisymmetry)."""
        a = np.random.rand(4, 4).astype(np.float32)
        out = pairwise_wedge(a, a)
        for i in range(4):
            np.testing.assert_array_almost_equal(out[i, i], 0.0)

    def test_antisymmetry(self):
        """pairwise_wedge(A, B)[i, j] = -pairwise_wedge(B, A)[j, i]."""
        a = np.random.rand(3, 4).astype(np.float32)
        b = np.random.rand(3, 4).astype(np.float32)
        ab = pairwise_wedge(a, b)
        ba = pairwise_wedge(b, a)
        np.testing.assert_array_almost_equal(ab, -ba.transpose(1, 0, 2))

    def test_known_value(self):
        """wedge([1,0,0,0], [0,1,0,0]) → first component = 1, rest = 0."""
        a = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        out = pairwise_wedge(a, b)   # (1, 1, 6)
        # pair (0,1): 1*1 - 0*0 = 1; all other pairs = 0
        self.assertAlmostEqual(float(out[0, 0, 0]), 1.0)
        for k in range(1, 6):
            self.assertAlmostEqual(float(out[0, 0, k]), 0.0)

    def test_all_zero_inputs(self):
        """All-zero action → all-zero wedge → tanh(0) = 0 → no launches."""
        a = np.zeros((4, 4), dtype=np.float32)
        out = pairwise_wedge(a, a)
        np.testing.assert_array_almost_equal(out, 0.0)

    def test_different_array_sizes(self):
        """Works for any n."""
        for n in [1, 2, 10]:
            a = np.random.rand(n, 4).astype(np.float32)
            out = pairwise_wedge(a, a)
            self.assertEqual(out.shape, (n, n, 6))


# ─────────────────────────────────────────────────────────────────────────────
# 4b. decode_action  (pairwise-bivector decoder)
# ─────────────────────────────────────────────────────────────────────────────

class TestDecodeAction(unittest.TestCase):
    """
    Geometry reference:
      Planet 0 (ours)  at (70, 80)  — top-right, 20 ships
      Planet 1 (opp)   at (30, 80)  — top-left,  15 ships
      Both far from sun (50,50), so direct paths are unblocked.

    To make planet 0 strongly target planet 1:
      action[0] = [1, 0, 0, 0],  action[1] = [0, 1, 0, 0]
      wedge sum = 1*1 - 0*0 = 1  → tanh(1) ≈ 0.76 > 0

    Expected direct angle from (70,80) → (30,80) = atan2(0, -40) = π.
    """

    def _planets_clear(self):
        """Two planets well clear of the sun."""
        return np.array([
            [0, 0, 70.0, 80.0, 5.0, 20.0, 2.0],   # ours, 20 ships
            [1, 1, 30.0, 80.0, 5.0, 15.0, 2.0],   # opponent
        ], dtype=np.float32)

    def _planets_full(self):
        """Four-planet fixture, planet 2 has only 1 ship."""
        return np.array([
            [0, 0, 70.0, 80.0, 5.0, 20.0, 2.0],   # ours, 20 ships
            [1, 1, 30.0, 80.0, 5.0, 15.0, 2.0],   # opponent
            [2, 0, 70.0, 20.0, 3.0,  1.0, 1.0],   # ours, 1 ship → skip
            [3, -1, 30.0, 20.0, 3.0, 5.0, 1.0],   # neutral
        ], dtype=np.float32)

    def _targeting_action(self, n=MAX_PLANETS):
        """action[0]=[1,0,0,0], action[1]=[0,1,0,0], rest zero → planet 0 targets planet 1."""
        act = np.zeros((n, 4), dtype=np.float32)
        if n > 0:
            act[0] = [1.0, 0.0, 0.0, 0.0]
        if n > 1:
            act[1] = [0.0, 1.0, 0.0, 0.0]
        return act

    # ── basic invariants ──────────────────────────────────────────────────────

    def test_returns_list(self):
        moves = decode_action(self._targeting_action(), self._planets_clear(), 0.0)
        self.assertIsInstance(moves, list)

    def test_empty_planets_returns_empty(self):
        act = self._targeting_action()
        moves = decode_action(act, np.empty((0, 7), dtype=np.float32), 0.0)
        self.assertEqual(moves, [])

    def test_all_zero_action_no_launch(self):
        """All-zero action → wedge = 0 → tanh(0) = 0 → scores.max() not > 0."""
        act = np.zeros((MAX_PLANETS, 4), dtype=np.float32)
        moves = decode_action(act, self._planets_full(), 0.0)
        self.assertEqual(len(moves), 0)

    # ── ownership / eligibility guards ────────────────────────────────────────

    def test_only_own_planets_launch(self):
        """Opponent and neutral planets never appear as move source."""
        act = self._targeting_action()
        moves = decode_action(act, self._planets_full(), 0.0)
        sources = {m[0] for m in moves}
        self.assertNotIn(1, sources)   # opponent
        self.assertNotIn(3, sources)   # neutral

    def test_one_ship_planet_skipped(self):
        """Planet with exactly 1 ship never launches."""
        act = self._targeting_action()
        moves = decode_action(act, self._planets_full(), 0.0)
        self.assertNotIn(2, {m[0] for m in moves})

    def test_no_self_target(self):
        """Source planet cannot target itself (checked via argmax restriction)."""
        act = self._targeting_action()
        planets = self._planets_clear()
        moves = decode_action(act, planets, 0.0)
        for from_id, _, _ in moves:
            # The argmax score for self is tanh(0)=0, which fails the >0 gate,
            # so with only one possible target (planet 1) we just verify it's used.
            self.assertEqual(len(moves), 1)
            break

    def test_at_most_one_move_per_source(self):
        """argmax picks exactly one target per source planet."""
        act = np.random.uniform(-1, 1, (MAX_PLANETS, 4)).astype(np.float32)
        moves = decode_action(act, self._planets_full(), 0.0)
        sources = [m[0] for m in moves]
        self.assertEqual(len(sources), len(set(sources)))

    # ── ship count ────────────────────────────────────────────────────────────

    def test_ships_leave_at_least_one(self):
        """Launched fleet never takes every ship from the source planet."""
        act = self._targeting_action()
        planets = self._planets_clear()
        moves = decode_action(act, planets, 0.0)
        for from_id, _, num_ships in moves:
            total = int(planets[planets[:, 0] == from_id, 5][0])
            self.assertLessEqual(num_ships, total - 1)

    # ── move format ───────────────────────────────────────────────────────────

    def test_move_format(self):
        """Each move is [int, float, int]."""
        moves = decode_action(self._targeting_action(), self._planets_clear(), 0.0)
        self.assertGreater(len(moves), 0)
        for from_id, angle, ships in moves:
            self.assertIsInstance(from_id, int)
            self.assertIsInstance(angle, float)
            self.assertIsInstance(ships, int)

    # ── angle correctness ─────────────────────────────────────────────────────

    def test_angle_toward_target_static(self):
        """omega=0: angle from (70,80) to (30,80) should equal atan2(0,-40) = π."""
        act   = self._targeting_action()
        moves = decode_action(act, self._planets_clear(), 0.0)
        self.assertEqual(len(moves), 1)
        _, angle, _ = moves[0]
        expected = math.atan2(80.0 - 80.0, 30.0 - 70.0)  # = π
        self.assertAlmostEqual(angle, expected, places=4)

    def test_angle_toward_target_orbiting(self):
        """omega != 0: function runs without error and returns a float angle."""
        act   = self._targeting_action()
        moves = decode_action(act, self._planets_clear(), omega=0.05)
        # With small omega the direct angle is usually still valid
        if moves:
            _, angle, _ = moves[0]
            self.assertIsInstance(angle, float)
            self.assertTrue(-math.pi <= angle <= math.pi)

    def test_sun_blocked_path_no_move(self):
        """Direct path through the sun → calculate_angle returns None → move skipped."""
        # Planet 0 (ours) directly below sun; planet 1 directly above sun.
        # The straight line from (50,35) to (50,65) passes through (50,50) = sun.
        planets = np.array([
            [0, 0, 50.0, 35.0, 3.0, 20.0, 2.0],
            [1, 1, 50.0, 65.0, 3.0, 15.0, 2.0],
        ], dtype=np.float32)
        act = self._targeting_action(n=2)
        moves = decode_action(act, planets, 0.0)
        self.assertEqual(len(moves), 0)


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
        r = compute_reward_for_player(obs, obs, player_id=0, done=False)
        self.assertAlmostEqual(r, 0.0)

    def test_gaining_planet_positive_reward(self):
        before = [[0, 0, 60, 50, 5, 10, 1], [1, 1, 40, 50, 5, 10, 1]]
        after  = [[0, 0, 60, 50, 5, 10, 1], [1, 0, 40, 50, 5,  5, 1]]  # planet 1 captured
        obs     = self._obs_from_planets(before)
        new_obs = self._obs_from_planets(after)
        r = compute_reward_for_player(obs, new_obs, player_id=0, done=False)
        self.assertGreater(r, 0.0)

    def test_losing_planet_negative_reward(self):
        before = [[0, 0, 60, 50, 5, 10, 1], [1, 1, 40, 50, 5, 10, 1]]
        after  = [[0, 1, 60, 50, 5,  5, 1], [1, 1, 40, 50, 5, 10, 1]]  # planet 0 lost
        obs     = self._obs_from_planets(before)
        new_obs = self._obs_from_planets(after)
        r = compute_reward_for_player(obs, new_obs, player_id=0, done=False)
        self.assertLess(r, 0.0)

    def test_win_terminal_bonus(self):
        before = [[0, 0, 60, 50, 5, 10, 1], [1, 1, 40, 50, 5,  1, 1]]
        after  = [[0, 0, 60, 50, 5, 12, 1], [1, 0, 40, 50, 5,  0, 1]]  # opponent wiped
        obs     = self._obs_from_planets(before)
        new_obs = self._obs_from_planets(after)
        r = compute_reward_for_player(obs, new_obs, player_id=0, done=True)
        self.assertGreater(r, 50.0)   # terminal bonus dominates

    def test_loss_terminal_bonus(self):
        before = [[0, 0, 60, 50, 5,  1, 1], [1, 1, 40, 50, 5, 10, 1]]
        after  = [[0, 1, 60, 50, 5,  0, 1], [1, 1, 40, 50, 5, 12, 1]]  # we are wiped
        obs     = self._obs_from_planets(before)
        new_obs = self._obs_from_planets(after)
        r = compute_reward_for_player(obs, new_obs, player_id=0, done=True)
        self.assertLess(r, -50.0)

    def test_returns_float(self):
        planets = [[0, 0, 60, 50, 5, 10, 1]]
        obs = self._obs_from_planets(planets)
        r = compute_reward_for_player(obs, obs, player_id=0, done=False)
        self.assertIsInstance(r, float)

    def test_symmetric_perspectives(self):
        before = [[0, 0, 60, 50, 5, 10, 1], [1, 1, 40, 50, 5, 10, 1]]
        after  = [[0, 0, 60, 50, 5, 12, 1], [1, 1, 40, 50, 5,  8, 1]]
        obs     = self._obs_from_planets(before)
        new_obs = self._obs_from_planets(after)
        r0 = compute_reward_for_player(obs, new_obs, player_id=0, done=False)
        r1 = compute_reward_for_player(obs, new_obs, player_id=1, done=False)
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
# 4c. Fleet trajectory accuracy
#     For each move produced by decode_action, simulate the fleet's per-tick
#     path (segment-circle collision) and verify it reaches the target planet.
# ─────────────────────────────────────────────────────────────────────────────

class TestFleetTrajectoryAccuracy(unittest.TestCase):
    """
    Verify that the angle returned by decode_action actually aims the fleet
    onto a path that intersects the target planet.

    Physics matches the kaggle game:
      fleet position at tick t : (fx + cos(a)*speed*t, fy + sin(a)*speed*t)
      planet position at tick t : (50 + r*cos(ang+omega*t), 50 + r*sin(ang+omega*t))
      collision via segment-circle test each tick (same as RewardScheme3).

    Planet configurations used here keep the direct path clear of the sun so
    that calculate_angle finds an orbit intercept rather than returning None.
      Source at (70, 80),  target at (30, 80) — both 30 units above the sun,
      straight line passes through (50, 80) which is 30 units from sun centre.
    """

    MAX_SPEED = 6.0
    SUN_X = SUN_Y = 50.0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _fleet_speed(self, ships: int) -> float:
        # Mirrors interpreter step 3: speed grows with fleet size, capped at MAX_SPEED.
        s = max(1, int(ships))
        spd = 1.0 + (self.MAX_SPEED - 1.0) * (math.log(s) / math.log(1000)) ** 1.5
        return min(spd, self.MAX_SPEED)

    @staticmethod
    def _swept(ax, ay, bx, by, p0x, p0y, p1x, p1y, r) -> bool:
        """Continuous swept-pair collision (interpreter copy): fleet A→B vs
        planet P0→P1 come within r for some t in [0, 1]."""
        d0x, d0y = ax - p0x, ay - p0y
        dvx = (bx - ax) - (p1x - p0x)
        dvy = (by - ay) - (p1y - p0y)
        a = dvx * dvx + dvy * dvy
        b = 2.0 * (d0x * dvx + d0y * dvy)
        c = d0x * d0x + d0y * d0y - r * r
        if a < 1e-12:
            return c <= 0.0
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return False
        sq = math.sqrt(disc)
        return (-b + sq) / (2.0 * a) >= 0.0 and (-b - sq) / (2.0 * a) <= 1.0

    def _simulate_hit(self, sx: float, sy: float, src_radius: float,
                      angle: float, speed: float, target: np.ndarray,
                      omega: float = 0.0, max_ticks: int = 200) -> tuple[bool, int]:
        """
        Faithful replica of the interpreter's fleet model: launch from the
        SOURCE planet's edge (radius + 0.1), advance `speed`/tick along a fixed
        angle, and test a continuous swept-pair collision against the (possibly
        orbiting) target each tick. Returns (hit, first_tick_hit).
        """
        tx, ty, tr = float(target[2]), float(target[3]), float(target[4])
        t_ang = math.atan2(ty - self.SUN_Y, tx - self.SUN_X)
        t_orb = math.hypot(tx - self.SUN_X, ty - self.SUN_Y)
        lx = sx + math.cos(angle) * (src_radius + 0.1)
        ly = sy + math.sin(angle) * (src_radius + 0.1)
        fpx, fpy = lx, ly
        ppx, ppy = tx, ty                       # target at tick 0 (current)
        for tick in range(1, max_ticks + 1):
            fxx = lx + math.cos(angle) * speed * tick
            fyy = ly + math.sin(angle) * speed * tick
            if omega:
                a = t_ang + omega * tick
                px = self.SUN_X + t_orb * math.cos(a)
                py = self.SUN_Y + t_orb * math.sin(a)
            else:
                px, py = tx, ty
            if self._swept(fpx, fpy, fxx, fyy, ppx, ppy, px, py, tr):
                return True, tick
            fpx, fpy = fxx, fyy
            ppx, ppy = px, py
        return False, -1

    def _find_target_idx(self, action_np: np.ndarray, planets: np.ndarray,
                         src_idx: int) -> int | None:
        """Reproduce decode_action's argmax to recover which planet was targeted."""
        out = np.tanh(pairwise_wedge(action_np, action_np).sum(axis=-1))
        n = min(planets.shape[0], action_np.shape[0])
        scores = out[src_idx, :n].copy()
        scores[src_idx] = -1.0   # mask self
        return int(np.argmax(scores)) if scores.max() > 0.0 else None

    def _sun_clear_planets(self):
        """
        Two planets whose straight-line path and orbit-intercept angles all avoid
        the sun.  (65,75)→(30,75): midpoint (47.5,75) is 26 units from sun; the
        orbit-intercept angles computed for omega 0.05 and 0.10 are also unblocked
        (verified: closest-approach distance to sun > 10 for those ticks).
        """
        return np.array([
            [0, 0, 65.0, 75.0, 5.0, 20.0, 2.0],   # ours, 20 ships
            [1, 1, 30.0, 75.0, 5.0, 15.0, 2.0],   # opponent
        ], dtype=np.float32)

    def _targeting_action(self, n: int = MAX_PLANETS) -> np.ndarray:
        """act[0]=[1,0,0,0], act[1]=[0,1,0,0]  → planet 0 strongly targets planet 1."""
        act = np.zeros((n, 4), dtype=np.float32)
        act[0] = [1.0, 0.0, 0.0, 0.0]
        act[1] = [0.0, 1.0, 0.0, 0.0]
        return act

    def _is_static_fallback(self, angle: float, src: np.ndarray,
                            tgt: np.ndarray) -> bool:
        """True if angle == atan2(ty-sy, tx-sx) — i.e. the static direct angle."""
        expected = math.atan2(float(tgt[3]) - float(src[3]),
                              float(tgt[2]) - float(src[2]))
        return abs(angle - expected) < 1e-9

    # ── static-planet tests (omega=0) ─────────────────────────────────────────

    def test_static_fleet_always_hits(self):
        """omega=0: direct angle always reaches the target (segment-circle cannot miss)."""
        planets = self._sun_clear_planets()
        moves = decode_action(self._targeting_action(), planets, omega=0.0)
        self.assertEqual(len(moves), 1)
        from_id, angle, num_ships = moves[0]
        src = planets[planets[:, 0] == from_id][0]
        tgt = planets[planets[:, 0] != from_id][0]
        speed = self._fleet_speed(num_ships)
        hit, _ = self._simulate_hit(src[2], src[3], src[4], angle, speed, tgt, omega=0.0)
        self.assertTrue(hit, f"Fleet missed static target (a={angle:.4f} spd={speed:.3f})")

    def test_static_random_actions_100pct_hit(self):
        """100 random actions on a 5-planet static map — every generated move hits."""
        planets = np.array([
            [0, 0,  70.0, 80.0, 5.0, 30.0, 2.0],
            [1, 1,  30.0, 80.0, 5.0, 20.0, 2.0],
            [2, 0,  80.0, 25.0, 4.0, 25.0, 2.0],
            [3, 1,  20.0, 25.0, 4.0, 15.0, 2.0],
            [4, -1, 50.0, 85.0, 3.0,  5.0, 1.0],
        ], dtype=np.float32)
        planet_by_id = {int(r[0]): r for r in planets}
        rng = np.random.default_rng(0)
        hits = misses = 0
        for _ in range(100):
            act = rng.uniform(-1, 1, (MAX_PLANETS, 4)).astype(np.float32)
            for from_id, angle, num_ships in decode_action(act, planets, omega=0.0):
                src_idx = int(np.where(planets[:, 0] == from_id)[0][0])
                tgt_idx = self._find_target_idx(act, planets, src_idx)
                if tgt_idx is None:
                    continue
                src, tgt = planet_by_id[from_id], planets[tgt_idx]
                speed = self._fleet_speed(num_ships)
                hit, _ = self._simulate_hit(src[2], src[3], src[4], angle, speed, tgt, omega=0.0)
                hits += hit; misses += (not hit)
        total = hits + misses
        if total == 0:
            self.skipTest("No moves generated")
        self.assertEqual(misses, 0,
                         f"Static hit rate {hits}/{total} — expected 100%")

    # ── orbiting-planet tests ─────────────────────────────────────────────────
    #
    # Planets at (65,75)→(30,75) are verified to have unblocked orbit-intercept
    # angles at both omega=0.05 (tick≈24) and omega=0.10 (tick≈26).
    # When decode_action uses the orbit intercept (not the static fallback), the
    # fleet must arrive within the target planet's radius.
    # Moves that used the static-fallback angle are skipped (the planet has moved).

    def test_orbiting_fleet_hits_moderate_omega(self):
        """omega=0.05: orbit intercept angle correctly leads to a hit."""
        planets = self._sun_clear_planets()
        moves = decode_action(self._targeting_action(), planets, omega=0.05)
        self.assertGreater(len(moves), 0, "No moves produced with omega=0.05")
        from_id, angle, num_ships = moves[0]
        src = planets[planets[:, 0] == from_id][0]
        tgt = planets[planets[:, 0] != from_id][0]
        if self._is_static_fallback(angle, src, tgt):
            self.skipTest("No orbit intercept found (sun-blocked fallback path)")
        speed = self._fleet_speed(num_ships)
        hit, _ = self._simulate_hit(src[2], src[3], src[4], angle, speed, tgt, omega=0.05)
        self.assertTrue(hit, f"Orbit intercept missed target omega=0.05 (a={angle:.4f})")

    def test_orbiting_fleet_hits_fast_omega(self):
        """omega=0.10: stronger orbit still intercepted correctly."""
        planets = self._sun_clear_planets()
        moves = decode_action(self._targeting_action(), planets, omega=0.10)
        self.assertGreater(len(moves), 0, "No moves produced with omega=0.10")
        from_id, angle, num_ships = moves[0]
        src = planets[planets[:, 0] == from_id][0]
        tgt = planets[planets[:, 0] != from_id][0]
        if self._is_static_fallback(angle, src, tgt):
            self.skipTest("No orbit intercept found (sun-blocked fallback path)")
        speed = self._fleet_speed(num_ships)
        hit, _ = self._simulate_hit(src[2], src[3], src[4], angle, speed, tgt, omega=0.10)
        self.assertTrue(hit, f"Orbit intercept missed target omega=0.10 (a={angle:.4f})")

    def test_orbiting_random_actions_intercept_moves_always_hit(self):
        """
        100 random actions on a 4-planet map with omega=0.05.
        Moves that used an orbit intercept angle (not the static fallback) must hit.
        Static-fallback moves are excluded — those aim at the planet's current
        position and can miss when the planet has moved significantly.
        """
        planets = np.array([
            [0, 0,  65.0, 75.0, 5.0, 30.0, 2.0],
            [1, 1,  30.0, 75.0, 5.0, 20.0, 2.0],
            [2, 0,  70.0, 25.0, 4.0, 25.0, 2.0],
            [3, 1,  25.0, 25.0, 4.0, 15.0, 2.0],
        ], dtype=np.float32)
        planet_by_id = {int(r[0]): r for r in planets}
        rng = np.random.default_rng(7)
        hits = misses = skipped = 0
        for _ in range(100):
            act = rng.uniform(-1, 1, (MAX_PLANETS, 4)).astype(np.float32)
            for from_id, angle, num_ships in decode_action(act, planets, omega=0.05):
                src_idx = int(np.where(planets[:, 0] == from_id)[0][0])
                tgt_idx = self._find_target_idx(act, planets, src_idx)
                if tgt_idx is None:
                    continue
                src, tgt = planet_by_id[from_id], planets[tgt_idx]
                if self._is_static_fallback(angle, src, tgt):
                    skipped += 1
                    continue   # fallback angle: planet moved, skip
                speed = self._fleet_speed(num_ships)
                hit, _ = self._simulate_hit(src[2], src[3], src[4], angle, speed, tgt, omega=0.05)
                hits += hit; misses += (not hit)
        total = hits + misses
        if total == 0:
            self.skipTest(f"No orbit-intercept moves generated (all {skipped} were static fallbacks)")
        self.assertEqual(misses, 0,
                         f"Orbit intercept hit rate {hits}/{total} — expected 100%")


# ─────────────────────────────────────────────────────────────────────────────
# 7. reward_scheme_3 — penalise fleets that miss all planets
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardScheme3(unittest.TestCase):
    """
    Geometry used throughout:
      Sun at (50, 50), radius 10.
      Planet at (80, 50), radius 5  — on the right half.
      Fleet launch point: (65, 50) — between sun and planet.

    angle=0     (right)   → fleet heads straight into planet         → HIT
    angle=π/2   (up)      → fleet departs perpendicular to planet    → MISS
    """

    def _planet(self, x=80.0, y=50.0, r=5.0):
        return SimpleNamespace(id=0, owner=-1, x=x, y=y,
                               radius=r, ships=0, production=1)

    def _fleet(self, fid, owner, x=65.0, y=50.0, angle=0.0, ships=10):
        return SimpleNamespace(id=fid, owner=owner, x=x, y=y,
                               angle=angle, from_planet_id=0, ships=ships)

    def _obs(self, planets=None, fleets=None, omega=0.0):
        return SimpleNamespace(
            planets=planets or [],
            fleets=fleets or [],
            angular_velocity=omega,
            comet_planet_ids=[],
        )

    # ── basic cases ───────────────────────────────────────────────────────────

    def test_no_new_fleets_zero(self):
        """No fleets at all → reward is 0."""
        obs = self._obs([self._planet()])
        r = reward_scheme_3(obs, obs, player_id=0, done=False)
        self.assertAlmostEqual(r, 0.0)

    def test_returns_float(self):
        obs = self._obs([self._planet()])
        r = reward_scheme_3(obs, obs, player_id=0, done=False)
        self.assertIsInstance(r, float)

    # ── fleet aimed at planet ─────────────────────────────────────────────────

    def test_fleet_aimed_at_planet_no_penalty(self):
        """Fleet heading directly toward a planet receives no penalty."""
        planet = self._planet(x=80.0, y=50.0, r=5.0)
        fleet  = self._fleet(0, 0, x=65.0, y=50.0, angle=0.0, ships=10)  # angle 0 → right
        obs     = self._obs([planet])
        new_obs = self._obs([planet], [fleet])
        r = reward_scheme_3(obs, new_obs, player_id=0, done=False)
        self.assertAlmostEqual(r, 0.0)

    # ── fleet that misses ─────────────────────────────────────────────────────

    def test_fleet_missing_all_planets_penalty(self):
        """Fleet going perpendicular to the planet → negative penalty."""
        planet = self._planet(x=80.0, y=50.0, r=5.0)
        fleet  = self._fleet(0, 0, x=65.0, y=50.0, angle=math.pi / 2, ships=10)
        obs     = self._obs([planet])
        new_obs = self._obs([planet], [fleet])
        r = reward_scheme_3(obs, new_obs, player_id=0, done=False)
        self.assertLess(r, 0.0)

    def test_fleet_aimed_away_from_all_planets(self):
        """Fleet going left while planet is on the right → miss → penalty."""
        planet = self._planet(x=80.0, y=50.0, r=5.0)
        fleet  = self._fleet(0, 0, x=65.0, y=50.0, angle=math.pi, ships=20)  # angle π → left
        obs     = self._obs([planet])
        new_obs = self._obs([planet], [fleet])
        r = reward_scheme_3(obs, new_obs, player_id=0, done=False)
        self.assertLess(r, 0.0)

    # ── attribution ──────────────────────────────────────────────────────────

    def test_existing_fleet_not_penalized(self):
        """Fleet present in both obs (same id) → not 'new' → no penalty."""
        planet = self._planet()
        fleet  = self._fleet(0, 0, angle=math.pi / 2, ships=10)  # would miss
        obs     = self._obs([planet], [fleet])   # fleet already existed
        new_obs = self._obs([planet], [fleet])
        r = reward_scheme_3(obs, new_obs, player_id=0, done=False)
        self.assertAlmostEqual(r, 0.0)

    def test_opponent_fleet_not_penalized(self):
        """Opponent's wasted fleet does not affect player_id=0's reward."""
        planet = self._planet()
        fleet  = self._fleet(0, owner=1, angle=math.pi / 2, ships=10)  # owner=1, miss
        obs     = self._obs([planet])
        new_obs = self._obs([planet], [fleet])
        r = reward_scheme_3(obs, new_obs, player_id=0, done=False)
        self.assertAlmostEqual(r, 0.0)

    # ── scaling ───────────────────────────────────────────────────────────────

    def test_penalty_scales_with_fleet_size(self):
        """Larger wasted fleet → more negative penalty."""
        planet = self._planet()
        obs = self._obs([planet])
        r_small = reward_scheme_3(
            obs, self._obs([planet], [self._fleet(0, 0, angle=math.pi / 2, ships=5)]),
            player_id=0, done=False,
        )
        r_large = reward_scheme_3(
            obs, self._obs([planet], [self._fleet(0, 0, angle=math.pi / 2, ships=100)]),
            player_id=0, done=False,
        )
        self.assertLess(r_small, 0.0)
        self.assertLess(r_large, r_small)

    def test_penalty_scales_with_ship_scale_param(self):
        """Higher ship_scale on the instance → more negative penalty for the same wasted fleet."""
        planet  = self._planet()
        fleet   = self._fleet(0, 0, angle=math.pi / 2, ships=20)
        obs     = self._obs([planet])
        new_obs = self._obs([planet], [fleet])
        r_low  = RewardScheme3(ship_scale=0.1)(obs, new_obs, player_id=0, done=False)
        r_high = RewardScheme3(ship_scale=1.0)(obs, new_obs, player_id=0, done=False)
        self.assertLess(r_low,  0.0)
        self.assertLess(r_high, r_low)

    # ── mixed fleets ─────────────────────────────────────────────────────────

    def test_mixed_hit_and_miss(self):
        """One hit fleet + one miss fleet → penalty equals single-miss penalty."""
        planet    = self._planet(x=80.0, y=50.0, r=5.0)
        fleet_hit = self._fleet(0, 0, angle=0.0,          ships=10)  # hits
        fleet_miss= self._fleet(1, 0, angle=math.pi / 2,  ships=10)  # misses
        obs       = self._obs([planet])

        r_both = reward_scheme_3(obs, self._obs([planet], [fleet_hit, fleet_miss]),
                                 player_id=0, done=False)
        r_miss_only = reward_scheme_3(obs, self._obs([planet], [fleet_miss]),
                                      player_id=0, done=False)
        self.assertAlmostEqual(r_both, r_miss_only, places=6)

    def test_two_miss_fleets_additive(self):
        """Two wasted fleets accumulate penalties."""
        planet  = self._planet()
        fleet_a = self._fleet(0, 0, angle=math.pi / 2, ships=10)
        fleet_b = self._fleet(1, 0, angle=math.pi / 2, ships=10)
        obs     = self._obs([planet])
        r_one = reward_scheme_3(obs, self._obs([planet], [fleet_a]),
                                player_id=0, done=False)
        r_two = reward_scheme_3(obs, self._obs([planet], [fleet_a, fleet_b]),
                                player_id=0, done=False)
        self.assertAlmostEqual(r_two, 2 * r_one, places=6)


# ─────────────────────────────────────────────────────────────────────────────

_PLAYER_COLORS  = {0: '#4488ff', 1: '#ff4444', -1: '#888888', 2: '#44cc44', 3: '#ffcc00'}
_PLAYER_LABELS  = {0: 'P0 (random)', 1: 'P1 (rule-based)',
                   2: 'P2 (rule-based)', 3: 'P3 (rule-based)'}


def _draw_frame(ax, obs, step_idx: int, total_steps: int, n_players: int = 2):
    """Render one observation onto ax (cleared each call)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from env.orbit_wars import _obs_to_arrays

    ax.clear()
    ax.set_facecolor('#0a0a1a')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.tick_params(colors='#888888')
    ax.set_title(f'Step {step_idx} / {total_steps}', color='white', pad=6)

    ax.add_patch(plt.Circle((50, 50), 10, color='#ffdd00', alpha=0.9, zorder=2))

    planets_np, fleets_np, _, _, _ = _obs_to_arrays(obs)

    for row in planets_np:
        _, owner, x, y, radius, ships, _ = row
        color = _PLAYER_COLORS.get(int(owner), '#888888')
        ax.add_patch(plt.Circle((x, y), max(radius, 2.0),
                                color=color, alpha=0.85, zorder=3))
        ax.text(x, y, str(int(ships)), color='white', ha='center', va='center',
                fontsize=7, fontweight='bold', zorder=4)

    for row in fleets_np:
        _, owner, x, y, *_, ships = row
        color = _PLAYER_COLORS.get(int(owner), '#888888')
        size  = max(20.0, min(150.0, float(ships) * 2.5))
        ax.scatter(x, y, s=size, color=color, alpha=0.75, zorder=5, marker='D')

    legend = [mpatches.Patch(color=_PLAYER_COLORS[i],
                             label=_PLAYER_LABELS.get(i, f'P{i}'))
              for i in range(n_players)]
    legend.append(mpatches.Patch(color=_PLAYER_COLORS[-1], label='Neutral'))
    ax.legend(handles=legend, loc='lower left',
              facecolor='#1a1a2e', labelcolor='white', fontsize=8)


def save_animation(all_obs: list, path: str = "simulation.gif", fps: int = 8,
                   n_players: int = 2):
    """
    Render a collected list of observations as an animated GIF or MP4.

    path ending in .mp4 uses FFMpeg; anything else uses Pillow (GIF).
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0a0a1a')
    total   = len(all_obs) - 1

    def _update(i):
        _draw_frame(ax, all_obs[i], i, total, n_players=n_players)

    anim = FuncAnimation(fig, _update, frames=len(all_obs),
                         interval=1000 // fps, repeat=False)

    if path.endswith('.mp4'):
        anim.save(path, writer=FFMpegWriter(fps=fps))
    else:
        anim.save(path, writer=PillowWriter(fps=fps))

    plt.close(fig)
    print(f"Animation saved : {path}  ({len(all_obs)} frames @ {fps} fps)")


def run_simulation(max_steps: int = 100, video_path: str = "simulation.gif",
                   fps: int = 8):
    """
    Simulate one episode and save an animated video.

    Player 0 : random actions
    Player 1 : RuleBasedAgent

    video_path ending in .mp4 uses FFMpeg; .gif uses Pillow.
    """
    from kaggle_environments import make as make_kaggle_env
    from env.orbit_wars import _obs_to_arrays, _swap_perspective, decode_action, compute_reward_for_player

    rule_based = RuleBasedAgent()

    env = make_kaggle_env("orbit_wars", debug=False)
    env.reset()

    obs_p0   = env.steps[0][0].observation
    obs_p1   = env.steps[0][1].observation
    all_obs  = [obs_p0]

    print("=" * 64)
    print("  Orbit Wars — random policy (p0) vs rule-based agent (p1)")
    print(f"  Max steps : {max_steps}")
    print("=" * 64)

    total_reward = 0.0
    final_step   = 0

    for step in range(max_steps):
        planets_np, _, _, omega, _ = _obs_to_arrays(obs_p0)

        action_p0 = np.random.uniform(-1.0, 1.0,
                                      (OrbitWarsEnv.MAX_PLANETS, OrbitWarsEnv.ACTION_DIM)
                                      ).astype(np.float32)
        swapped_p0, _ = _swap_perspective(planets_np,
                                          np.empty((0, 7), np.float32), player_id=0)
        moves_p0 = decode_action(action_p0, swapped_p0, omega)
        moves_p1 = rule_based(obs_p1)

        step_results = env.step(actions=[moves_p0, moves_p1])
        new_obs_p0   = step_results[0].observation
        new_obs_p1   = step_results[1].observation
        done         = step_results[0].status != "ACTIVE"
        final_step   = step + 1

        reward_scheme = RewardScheme2(ship_scale=0.5, planet_scale=1.0, win_bonus=20.0)
        reward        = reward_scheme.compute_reward(obs_p0, new_obs_p0, player_id=0, done=done)
        total_reward += reward
        all_obs.append(new_obs_p0)

        tag = " [DONE]" if done else ""
        print(f"step {final_step:3d} | reward {reward:+7.3f} | "
              f"cumulative {total_reward:+8.3f}{tag}")

        obs_p0, obs_p1 = new_obs_p0, new_obs_p1
        if done:
            break

    print(f"\nRendering {len(all_obs)} frames …")
    save_animation(all_obs, path=video_path, fps=fps)

    print()
    print("=" * 64)
    print(f"Episode ended after {final_step} steps  |  total reward : {total_reward:.3f}")
    print("=" * 64)


def run_simulation_4p(max_steps: int = 100, video_path: str = "simulation_4p.gif",
                      fps: int = 8):
    """
    4-player simulation:
      P0 — random actions
      P1, P2, P3 — independent RuleBasedAgent instances

    Uses env.reset(num_agents=4) to activate orbit_wars 4-player mode.
    """
    from kaggle_environments import make as make_kaggle_env
    from agents.agent1 import RuleBasedAgent
    from env.orbit_wars import _obs_to_arrays, _swap_perspective, decode_action, compute_reward_for_player

    N = 4
    rule_based = [RuleBasedAgent() for _ in range(N - 1)]

    env = make_kaggle_env("orbit_wars", debug=False)
    env.reset(num_agents=N)  # key: tells orbit_wars to use 4-player mode

    obs_list = [env.steps[0][i].observation for i in range(N)]
    all_obs  = [obs_list[0]]

    print("=" * 64)
    print("  4-Player Orbit Wars")
    print("  P0: random  |  P1 P2 P3: rule-based")
    print(f"  Max steps : {max_steps}")
    print("=" * 64)

    total_reward = 0.0
    final_step   = 0

    for step in range(max_steps):
        planets_np, _, _, omega, _ = _obs_to_arrays(obs_list[0])
        action_p0 = np.random.uniform(-1.0, 1.0,
                                      (OrbitWarsEnv.MAX_PLANETS, OrbitWarsEnv.ACTION_DIM)
                                      ).astype(np.float32)
        swapped_p0, _ = _swap_perspective(planets_np,
                                          np.empty((0, 7), np.float32), player_id=0)
        moves = [decode_action(action_p0, swapped_p0, omega)]
        for j, agent in enumerate(rule_based):
            moves.append(agent(obs_list[j + 1]))

        step_results = env.step(actions=moves)
        new_obs_list = [step_results[i].observation for i in range(N)]
        done         = step_results[0].status != "ACTIVE"
        final_step   = step + 1

        reward_scheme = RewardScheme2(ship_scale=0.5, planet_scale=1.0, win_bonus=20.0)
        reward        = reward_scheme.compute_reward(obs_list[0], new_obs_list[0], player_id=0, done=done)
        total_reward += reward
        all_obs.append(new_obs_list[0])

        tag = " [DONE]" if done else ""
        print(f"step {final_step:3d} | reward {reward:+7.3f} | "
              f"cumulative {total_reward:+8.3f}{tag}")

        obs_list = new_obs_list
        if done:
            break

    print(f"\nRendering {len(all_obs)} frames …")
    save_animation(all_obs, path=video_path, fps=fps, n_players=N)

    print()
    print("=" * 64)
    print(f"Episode ended after {final_step} steps  |  total reward : {total_reward:.3f}")
    print("=" * 64)


if __name__ == "__main__":
    env = OrbitWarsEnv(opponent="random", player_id=0)

    state, _ = env.reset()
    print("Initial state shape:", state.shape)
    print(state)

    # unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--4p", dest="four_player", action="store_true",
    #                     help="run 4-player simulation instead of 2-player")
    # parser.add_argument("--steps", type=int, default=500)
    # parser.add_argument("--fps",   type=int, default=4)
    # args = parser.parse_args()

    # if args.four_player:
    #     run_simulation_4p(max_steps=args.steps,
    #                       video_path="simulation_4p.gif", fps=args.fps)
    # else:
    #     run_simulation(max_steps=args.steps,
    #                    video_path="simulation.gif", fps=args.fps)
