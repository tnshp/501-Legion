"""
Gymnasium wrapper for the Kaggle Orbit Wars environment.

Observation space : Box(shape=(MAX_PLANETS + MAX_FLEETS, STATE_DIM), float32)
Action space      : Box(shape=(MAX_PLANETS, ACTION_DIM), float32)  in [-1, 1]

Per-planet action encoding
    col 0   : launch gate  (> 0 → launch from this planet)
    col 1-2 : angle vector (sin θ, cos θ)  →  θ = atan2(col1, col2)  [radians]
    col 3   : ship fraction in [-1, 1] → mapped to (0, 1) of available ships
    col 4-7 : reserved / unused

Helper functions (imported by the self-play trainer)
    encode_obs_as_player
    decode_action
    compute_reward_for_player
    _obs_to_arrays
    _swap_perspective
"""

from __future__ import annotations

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

MAX_PLANETS: int = 40
MAX_FLEETS: int = 100
STATE_DIM: int = 14
ACTION_DIM: int = 8

_EMPTY_FLEETS = np.empty((0, 7), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Observation parsing
# ─────────────────────────────────────────────────────────────────────────────

def _obs_to_arrays(obs):
    """
    Parse a kaggle orbit_wars observation into numpy arrays.

    Handles both attribute-access (Observation object) and dict-access.

    Returns
    -------
    planets_np         : [n, 7]  [id, owner, x, y, radius, ships, production]
    fleets_np          : [m, 7]  [id, owner, x, y, angle, from_planet_id, ships]
    angular_velocity   : float
    omega              : float  (alias for angular_velocity)
    comet_ids          : np.ndarray of int
    """
    def _get(obj, attr):
        return getattr(obj, attr, None) if not isinstance(obj, dict) else obj.get(attr)

    raw_planets = _get(obs, "planets")
    raw_fleets  = _get(obs, "fleets")
    angular_velocity = float(_get(obs, "angular_velocity") or 0.0)
    raw_comets  = _get(obs, "comet_planet_ids")

    # None-safe defaults (can't use `or []` — numpy arrays are truthy-ambiguous)
    if raw_planets is None:
        raw_planets = []
    if raw_fleets is None:
        raw_fleets = []
    if raw_comets is None:
        raw_comets = []

    # ── planets ──────────────────────────────────────────────────────────────
    if isinstance(raw_planets, np.ndarray) and raw_planets.ndim == 2:
        planets_np = raw_planets.astype(np.float32)
    else:
        rows = []
        for p in raw_planets:
            if isinstance(p, dict):
                rows.append([p["id"], p.get("owner", -1) if p.get("owner") is not None else -1,
                             p["x"], p["y"], p["radius"], p["ships"], p["production"]])
            else:
                owner = getattr(p, "owner", -1)
                rows.append([p.id, owner if owner is not None else -1,
                             p.x, p.y, p.radius, p.ships, p.production])
        planets_np = np.array(rows, dtype=np.float32) if rows else np.empty((0, 7), dtype=np.float32)

    # ── fleets ────────────────────────────────────────────────────────────────
    if isinstance(raw_fleets, np.ndarray) and raw_fleets.ndim == 2:
        fleets_np = raw_fleets.astype(np.float32)
    elif not raw_fleets:
        fleets_np = _EMPTY_FLEETS.copy()
    else:
        rows = []
        for f in raw_fleets:
            if isinstance(f, dict):
                rows.append([f["id"], f.get("owner", -1) if f.get("owner") is not None else -1,
                             f["x"], f["y"], f["angle"], f["from_planet_id"], f["ships"]])
            else:
                owner = getattr(f, "owner", -1)
                rows.append([f.id, owner if owner is not None else -1,
                             f.x, f.y, f.angle, f.from_planet_id, f.ships])
        fleets_np = np.array(rows, dtype=np.float32) if rows else _EMPTY_FLEETS.copy()

    # ── comets ────────────────────────────────────────────────────────────────
    if isinstance(raw_comets, np.ndarray):
        comet_ids = raw_comets.astype(np.int32)
    else:
        comet_ids = np.array(list(raw_comets), dtype=np.int32)

    omega = angular_velocity
    return planets_np, fleets_np, angular_velocity, omega, comet_ids


# ─────────────────────────────────────────────────────────────────────────────
# Perspective normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _swap_perspective(planets: np.ndarray, fleets: np.ndarray, player_id: int):
    """
    Re-label owners so the acting player always appears as player 0.

    player_id == 0 → identity (no-op copy).
    player_id == 1 → swap owner 0 ↔ 1; owners 2, 3, -1 are untouched.

    Returns copies of both arrays.
    """
    if player_id == 0:
        return planets.copy(), fleets.copy()

    swapped_p = planets.copy()
    owner_p = swapped_p[:, 1]
    swapped_p[:, 1] = np.where(owner_p == 0, 1.0,
                      np.where(owner_p == 1, 0.0, owner_p))

    swapped_f = fleets.copy()
    if swapped_f.shape[0] > 0:
        owner_f = swapped_f[:, 1]
        swapped_f[:, 1] = np.where(owner_f == 0, 1.0,
                          np.where(owner_f == 1, 0.0, owner_f))

    return swapped_p, swapped_f


# ─────────────────────────────────────────────────────────────────────────────
# State encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_obs_as_player(encoder, obs, initial_planets: np.ndarray,
                         player_id: int = 0, time_step: int = 0) -> np.ndarray:
    """
    Encode a kaggle orbit_wars observation as [MAX_PLANETS+MAX_FLEETS, STATE_DIM].

    Parameters
    ----------
    encoder        : Encoder from model/SAC.py
    obs            : raw kaggle observation
    initial_planets: [n, 7] planets from the very first step (used for orbit detection)
    player_id      : 0 or 1 — whose perspective to use
    time_step      : current episode step

    Returns
    -------
    state : np.ndarray [MAX_PLANETS + MAX_FLEETS, STATE_DIM], dtype float32
    """
    planets_np, fleets_np, angular_velocity, _, comet_ids = _obs_to_arrays(obs)
    s_planets, s_fleets = _swap_perspective(planets_np, fleets_np, player_id)

    state, _ = encoder.encode(
        s_planets, s_fleets, initial_planets,
        angular_velocity, comet_ids, time_step,
        apply_padding=True,
    )
    return state.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Action decoding
# ─────────────────────────────────────────────────────────────────────────────

def decode_action(action_np: np.ndarray, planets: np.ndarray, omega: float) -> list:
    """
    Convert policy output to a list of kaggle orbit_wars moves.

    Parameters
    ----------
    action_np : [MAX_PLANETS, ACTION_DIM]  tanh-squashed, values in [-1, 1]
                Encoding per planet row:
                  [0]   launch gate  (> 0 → launch)
                  [1:3] (sin θ, cos θ) → target heading θ = atan2([1],[2]) radians
                  [3]   ship fraction ([-1,1] → (0,1])
                  [4:8] reserved
    planets   : [n, 7]  swapped planet array [id, owner, x, y, radius, ships, production]
                Already in player-0 perspective (owner==0 means our planet).
    omega     : float — angular velocity (unused in decoding, kept for API symmetry)

    Returns
    -------
    moves : list of [planet_id (int), angle_radians (float), num_ships (int)]
    """
    moves = []
    n_planets = min(planets.shape[0], MAX_PLANETS)

    for i in range(n_planets):
        planet_id = int(planets[i, 0])
        owner     = int(planets[i, 1])
        ships     = int(planets[i, 5])

        # Only send from our own planets that have ships to spare
        if owner != 0 or ships <= 1:
            continue

        act = action_np[i]  # [ACTION_DIM]

        # Launch gate: positive → launch
        if act[0] <= 0.0:
            continue

        # Angle from (sin, cos) pair — atan2 is robust to scale
        angle_rad = math.atan2(float(act[1]), float(act[2]))

        # Ship fraction: [-1, 1] → [0, 1], then fraction of (ships - 1)
        frac = (float(act[3]) + 1.0) * 0.5          # [0, 1]
        num_ships = max(1, int(frac * (ships - 1)))   # keep ≥ 1 on planet

        moves.append([planet_id, angle_rad, num_ships])

    return moves


# ─────────────────────────────────────────────────────────────────────────────
# Reward shaping
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward_for_player(obs, new_obs, player_id: int) -> float:
    """
    Shaped reward for a single environment step.

    Components
    ----------
    ship_delta    : change in (our ships − opponent ships) on planets
    planet_delta  : change in (our planet count − opponent planet count)
    terminal      : ±100 for win / loss

    Returns
    -------
    reward : float
    """
    planets_old, _, _, _, _ = _obs_to_arrays(obs)
    planets_new, _, _, _, _ = _obs_to_arrays(new_obs)

    pid = player_id
    opp = 1 - pid

    def _ships(p, owner):
        mask = p[:, 1] == owner
        return float(p[mask, 5].sum()) if mask.any() else 0.0

    def _count(p, owner):
        return int((p[:, 1] == owner).sum())

    # Ship delta (encourages building production advantage)
    my_ships_δ  = _ships(planets_new, pid) - _ships(planets_old, pid)
    opp_ships_δ = _ships(planets_new, opp) - _ships(planets_old, opp)
    ship_delta  = my_ships_δ - opp_ships_δ

    # Planet count delta (encourages conquest)
    my_cnt_δ   = _count(planets_new, pid) - _count(planets_old, pid)
    opp_cnt_δ  = _count(planets_new, opp) - _count(planets_old, opp)
    planet_delta = my_cnt_δ - opp_cnt_δ

    # Terminal bonus
    my_cnt_new  = _count(planets_new, pid)
    opp_cnt_new = _count(planets_new, opp)
    terminal_bonus = 0.0
    if opp_cnt_new == 0 and my_cnt_new > 0:
        terminal_bonus =  100.0
    elif my_cnt_new == 0 and opp_cnt_new > 0:
        terminal_bonus = -100.0

    reward = 0.01 * ship_delta + 1.0 * planet_delta + terminal_bonus
    return float(reward)


# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium wrapper
# ─────────────────────────────────────────────────────────────────────────────

class OrbitWarsEnv(gym.Env):
    """
    Single-player gymnasium wrapper for Kaggle Orbit Wars.

    The acting agent always plays as player 0 (perspective-normalised by
    _swap_perspective). An opponent agent or string tag can be supplied.

    Parameters
    ----------
    opponent  : str | callable
                Kaggle agent spec for the opposing player (e.g. "random",
                a path to a submission, or an agent function).
    player_id : int  (0 or 1)
                Which seat the learning agent occupies in the kaggle env.
    encoder   : Encoder | None
                Instance of model.SAC.Encoder. Created automatically if None.
    max_steps : int
                Episode truncation limit (kaggle default is 500).
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self,
                 opponent="random",
                 player_id: int = 0,
                 encoder=None,
                 max_steps: int = 500):
        super().__init__()

        self.opponent  = opponent
        self.player_id = player_id
        self.max_steps = max_steps

        if encoder is None:
            from model.SAC import Encoder
            encoder = Encoder(max_planets=MAX_PLANETS, max_fleets=MAX_FLEETS)
        self.encoder = encoder

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(MAX_PLANETS + MAX_FLEETS, STATE_DIM),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(MAX_PLANETS, ACTION_DIM),
            dtype=np.float32,
        )

        self._kaggle_env    = None
        self._trainer       = None
        self._current_obs   = None
        self._initial_planets: np.ndarray | None = None
        self._time_step     = 0

    # ── internal ──────────────────────────────────────────────────────────────

    def _build_trainer(self):
        from kaggle_environments import make as kaggle_make
        self._kaggle_env = kaggle_make("orbit_wars", debug=False)
        agents = ([None, self.opponent] if self.player_id == 0
                  else [self.opponent, None])
        self._trainer = self._kaggle_env.train(agents)

    def _encode(self, obs, time_step: int) -> np.ndarray:
        return encode_obs_as_player(
            self.encoder, obs, self._initial_planets,
            self.player_id, time_step,
        )

    # ── gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Build trainer on first call; re-use for subsequent resets
        if self._trainer is None:
            self._build_trainer()

        raw_obs = self._trainer.reset()
        self._current_obs = raw_obs
        self._time_step   = 0

        # Snapshot initial planets for orbit-detection in encoder
        planets_np, _, _, _, _ = _obs_to_arrays(raw_obs)
        s_planets, _ = _swap_perspective(
            planets_np, _EMPTY_FLEETS.copy(), self.player_id
        )
        self._initial_planets = s_planets

        state = self._encode(raw_obs, self._time_step)
        return state, {}

    def step(self, action: np.ndarray):
        assert self._trainer is not None, "Call reset() before step()."

        # Decode action using the swapped planet list (player-0 perspective)
        planets_np, _, _, omega, _ = _obs_to_arrays(self._current_obs)
        s_planets, _ = _swap_perspective(
            planets_np, _EMPTY_FLEETS.copy(), self.player_id
        )
        moves = decode_action(action, s_planets, omega)

        # Step kaggle env
        raw_obs, _kaggle_reward, done, info = self._trainer.step(moves)
        self._time_step += 1

        reward     = compute_reward_for_player(self._current_obs, raw_obs, self.player_id)
        truncated  = self._time_step >= self.max_steps
        terminated = bool(done) and not truncated

        self._current_obs = raw_obs
        state = self._encode(raw_obs, self._time_step)

        return state, reward, terminated, truncated, info or {}

    def render(self, mode: str = "human"):
        if self._kaggle_env is not None:
            print(self._kaggle_env.render(mode="ansi"))

    def close(self):
        self._trainer    = None
        self._kaggle_env = None
