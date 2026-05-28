"""
Gymnasium wrapper for the Kaggle Orbit Wars environment.

Observation space : Box(shape=(MAX_PLANETS + MAX_FLEETS, STATE_DIM), float32)
Action space      : Box(shape=(MAX_PLANETS, ACTION_DIM), float32)  in [-1, 1]

Per-planet action encoding  (ACTION_DIM = 4)
    col 0 : launch gate     (> 0 → launch from this planet)
    col 1 : sin θ           ┐ angle vector → θ = atan2(col1, col2) [radians]
    col 2 : cos θ           ┘
    col 3 : fleet fraction  [-1, 1] → mapped to (0, 1) of available ships

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

_EMPTY_FLEETS = np.empty((0, 7), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Observation parsing
# ─────────────────────────────────────────────────────────────────────────────

def _obs_to_arrays(obs):
    """
    Parse a kaggle orbit_wars observation into numpy arrays.

    Handles attribute-access (Observation object), dict-access, and kaggle's
    native list-of-lists format.

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

    def _safe_owner(v):
        return -1 if v is None else int(v)

    def _parse_planet_row(p) -> list:
        if isinstance(p, (list, tuple)):
            return [p[0], _safe_owner(p[1]), p[2], p[3], p[4], p[5], p[6]]
        if isinstance(p, dict):
            return [p["id"], _safe_owner(p.get("owner")),
                    p["x"], p["y"], p["radius"], p["ships"], p["production"]]
        return [p.id, _safe_owner(getattr(p, "owner", -1)),
                p.x, p.y, p.radius, p.ships, p.production]

    def _parse_fleet_row(f) -> list:
        if isinstance(f, (list, tuple)):
            return [f[0], _safe_owner(f[1]), f[2], f[3], f[4], f[5], f[6]]
        if isinstance(f, dict):
            return [f["id"], _safe_owner(f.get("owner")),
                    f["x"], f["y"], f["angle"], f["from_planet_id"], f["ships"]]
        return [f.id, _safe_owner(getattr(f, "owner", -1)),
                f.x, f.y, f.angle, f.from_planet_id, f.ships]

    raw_planets = _get(obs, "planets")
    raw_fleets  = _get(obs, "fleets")
    angular_velocity = float(_get(obs, "angular_velocity") or 0.0)
    raw_comets  = _get(obs, "comet_planet_ids")

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
        rows = [_parse_planet_row(p) for p in raw_planets]
        planets_np = np.array(rows, dtype=np.float32) if rows else np.empty((0, 7), dtype=np.float32)

    # ── fleets ────────────────────────────────────────────────────────────────
    if isinstance(raw_fleets, np.ndarray) and raw_fleets.ndim == 2:
        fleets_np = raw_fleets.astype(np.float32)
    elif not raw_fleets:
        fleets_np = _EMPTY_FLEETS.copy()
    else:
        rows = [_parse_fleet_row(f) for f in raw_fleets]
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
    player_id == k → swap owner 0 ↔ k; all other owners are untouched.

    Works for any number of players (2–4).

    Returns copies of both arrays.
    """
    if player_id == 0:
        return planets.copy(), fleets.copy()

    pid = float(player_id)
    swapped_p = planets.copy()
    owner_p = swapped_p[:, 1]
    swapped_p[:, 1] = np.where(owner_p == 0.0, pid,
                      np.where(owner_p == pid,  0.0, owner_p))

    swapped_f = fleets.copy()
    if swapped_f.shape[0] > 0:
        owner_f = swapped_f[:, 1]
        swapped_f[:, 1] = np.where(owner_f == 0.0, pid,
                          np.where(owner_f == pid,  0.0, owner_f))

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
    action_np : [n_planets, 4]  tanh-squashed, values in [-1, 1]
                  [0] launch gate  (> 0 → launch)
                  [1] sin θ  ┐ heading θ = atan2([1],[2]) radians
                  [2] cos θ  ┘
                  [3] fleet fraction ([-1,1] → [0,1]) → num_ships = int(frac * ships)
    planets   : [n, 7]  swapped planet array [id, owner, x, y, radius, ships, production]
                Already in player-0 perspective (owner==0 means our planet).
    omega     : float — angular velocity (unused in decoding, kept for API symmetry)

    Returns
    -------
    moves : list of [planet_id (int), angle_radians (float), num_ships (int)]
    """
    moves = []
    n_planets = min(planets.shape[0], action_np.shape[0])

    for i in range(n_planets):
        planet_id = int(planets[i, 0])
        owner     = int(planets[i, 1])
        ships     = int(planets[i, 5])

        if owner != 0 or ships <= 1:
            continue

        act = action_np[i]

        if act[0] <= 0.0:
            continue

        angle_rad = math.atan2(float(act[1]), float(act[2]))
        frac      = (float(act[3]) + 1.0) * 0.5
        num_ships = min(int(frac * ships), ships - 1)

        if num_ships <= 0:
            continue

        moves.append([planet_id, angle_rad, num_ships])

    return moves


# ─────────────────────────────────────────────────────────────────────────────
# Reward shaping — class-based API
# ─────────────────────────────────────────────────────────────────────────────

class RewardScheme1:
    """
    Shaped reward: relative ship advantage + planet advantage + terminal bonus.

    Components
    ----------
    ship_scale   × (our_ships_Δ − sum_opp_ships_Δ)
    planet_scale × (our_planet_cnt_Δ − sum_opp_planet_cnt_Δ)
    ±100 terminal bonus on win / loss

    Parameters
    ----------
    ship_scale   : float, default 0.01
    planet_scale : float, default 1.0
    """

    def __init__(self, ship_scale: float = 0.01, planet_scale: float = 1.0, win_bonus: float = 100.0):
        self.ship_scale   = ship_scale
        self.planet_scale = planet_scale
        self.win_bonus     = win_bonus

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2) -> float:
        planets_old, _, _, _, _ = _obs_to_arrays(obs)
        planets_new, _, _, _, _ = _obs_to_arrays(new_obs)

        pid          = player_id
        opponent_ids = [p for p in range(n_players) if p != pid]

        def _ships(p, owner):
            mask = p[:, 1] == owner
            return float(p[mask, 5].sum()) if mask.any() else 0.0

        def _count(p, owner):
            return int((p[:, 1] == owner).sum())

        my_ships_δ   = _ships(planets_new, pid) - _ships(planets_old, pid)
        opp_ships_δ  = sum(_ships(planets_new, o) - _ships(planets_old, o) for o in opponent_ids)
        ship_delta   = my_ships_δ - opp_ships_δ

        my_cnt_δ     = _count(planets_new, pid) - _count(planets_old, pid)
        opp_cnt_δ    = sum(_count(planets_new, o) - _count(planets_old, o) for o in opponent_ids)
        planet_delta = my_cnt_δ - opp_cnt_δ

        terminal_bonus = 0.0
        if done:
            my_ships_final = _ships(planets_new, pid)
            best_opp       = max((_ships(planets_new, o) for o in opponent_ids), default=0.0)
            if best_opp < my_ships_final:
                terminal_bonus =  self.win_bonus
            elif best_opp > my_ships_final:
                terminal_bonus = -self.win_bonus

        return float(self.ship_scale * ship_delta + self.planet_scale * planet_delta + terminal_bonus)


class RewardScheme2:
    """
    Shaped reward: own ship delta + own planet delta + terminal bonus.
    Ignores opponent metrics — pure self-improvement signal.

    Parameters
    ----------
    ship_scale   : float, default 0.01
    planet_scale : float, default 1.0
    """

    def __init__(self, ship_scale: float = 0.01, planet_scale: float = 1.0, win_bonus: float = 100.0):
        self.ship_scale   = ship_scale
        self.planet_scale = planet_scale
        self.win_bonus     = win_bonus

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2) -> float:
        planets_old, _, _, _, _ = _obs_to_arrays(obs)
        planets_new, _, _, _, _ = _obs_to_arrays(new_obs)

        pid          = player_id
        opponent_ids = [p for p in range(n_players) if p != pid]

        def _ships(p, owner):
            mask = p[:, 1] == owner
            return float(p[mask, 5].sum()) if mask.any() else 0.0

        def _count(p, owner):
            return int((p[:, 1] == owner).sum())

        my_ships_δ = _ships(planets_new, pid) - _ships(planets_old, pid)
        my_cnt_δ   = _count(planets_new, pid) - _count(planets_old, pid)

        terminal_bonus = 0.0
        if done:
            my_ships_final = _ships(planets_new, pid)
            best_opp       = max((_ships(planets_new, o) for o in opponent_ids), default=0.0)
            if best_opp < my_ships_final:
                terminal_bonus =  self.win_bonus
            elif best_opp > my_ships_final:
                terminal_bonus = -self.win_bonus

        return float(self.ship_scale * my_ships_δ + self.planet_scale * my_cnt_δ + terminal_bonus)


class RewardScheme3:
    """
    Penalises fleets sent into open space (trajectory misses all planets).

    For each new fleet (present in new_obs but not obs) owned by player_id,
    the straight-line path is simulated tick-by-tick. Planet positions are
    projected forward with orbital angular velocity. If no planet is hit within
    max_ticks steps, -ship_scale * log(1 + ships_sent) is applied.

    Parameters
    ----------
    ship_scale   : float, default 0.5 — penalty per log-unit of wasted ships
    planet_scale : float, default 1.0 — unused, kept for API symmetry
    max_ticks    : int,   default 200 — trajectory simulation horizon
    """

    def __init__(self, ship_scale: float = 0.5, planet_scale: float = 1.0,
                 max_ticks: int = 200):
        self.ship_scale   = ship_scale
        self.planet_scale = planet_scale
        self.max_ticks    = max_ticks

    def __call__(self, obs, new_obs, player_id: int, done: bool,
                 n_players: int = 2) -> float:
        SUN_X = SUN_Y = 50.0

        _, fleets_old, _, _, _               = _obs_to_arrays(obs)
        planets_new, fleets_new, _, omega, _ = _obs_to_arrays(new_obs)

        if fleets_new.shape[0] == 0:
            return 0.0

        old_ids = {int(r[0]) for r in fleets_old} if fleets_old.shape[0] > 0 else set()

        # Pre-compute per-planet orbital parameters once per call
        p_base = []
        for row in planets_new:
            px, py, pr = float(row[2]), float(row[3]), float(row[4])
            p_ang = math.atan2(py - SUN_Y, px - SUN_X)
            p_orb = math.sqrt((px - SUN_X) ** 2 + (py - SUN_Y) ** 2)
            p_base.append((px, py, p_ang, p_orb, pr))

        penalty = 0.0

        for f_row in fleets_new:
            if int(f_row[0]) in old_ids or int(f_row[1]) != player_id:
                continue

            fx    = float(f_row[2])
            fy    = float(f_row[3])
            fang  = float(f_row[4])
            fship = max(1, int(f_row[6]))

            # Fleet speed (mirrors agents/agent1.py: get_fleet_speed)
            speed = 1.0 + 5.0 * (math.log(fship) / math.log(1000)) ** 1.5

            hits           = False
            prev_x, prev_y = fx, fy

            for tick in range(1, self.max_ticks + 1):
                nx = fx + math.cos(fang) * speed * tick
                ny = fy + math.sin(fang) * speed * tick

                for px0, py0, p_ang, p_orb, pr in p_base:
                    if omega != 0.0:
                        pcx = SUN_X + p_orb * math.cos(p_ang + omega * tick)
                        pcy = SUN_Y + p_orb * math.sin(p_ang + omega * tick)
                    else:
                        pcx, pcy = px0, py0

                    # Segment-circle collision (mirrors agents/agent1.py: collides)
                    dx, dy = nx - prev_x, ny - prev_y
                    ex, ey = pcx - prev_x, pcy - prev_y
                    ssq = dx * dx + dy * dy
                    if ssq == 0.0:
                        if ex * ex + ey * ey <= pr * pr:
                            hits = True
                            break
                    else:
                        t = max(0.0, min(1.0, (ex * dx + ey * dy) / ssq))
                        cx = prev_x + t * dx - pcx
                        cy = prev_y + t * dy - pcy
                        if cx * cx + cy * cy <= pr * pr:
                            hits = True
                            break

                if hits:
                    break
                prev_x, prev_y = nx, ny

            if not hits:
                penalty -= self.ship_scale * math.log1p(fship)

        return penalty


# ── Module-level default instances (backward compatibility) ──────────────────
reward_scheme_1 = RewardScheme1()
reward_scheme_2 = RewardScheme2()
reward_scheme_3 = RewardScheme3()

# Alias used by test_orbit_wars_env.py
compute_reward_for_player = reward_scheme_1

# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium wrapper
# ─────────────────────────────────────────────────────────────────────────────

class OrbitWarsEnv(gym.Env):
    """
    Single-player gymnasium wrapper for Kaggle Orbit Wars.

    The acting agent always plays as player 0 (perspective-normalised by
    _swap_perspective). An opponent agent or string tag can be supplied.

    Class attributes (game constants)
    ----------------------------------
    MAX_PLANETS, MAX_FLEETS, STATE_DIM, ACTION_DIM

    Parameters
    ----------
    opponent      : str | callable | list
                    Opponent spec for all non-learning seats, or a list of
                    n_players-1 specs (one per seat).  Built-in strings:
                      "random"     — kaggle's built-in random agent
                      "rule_based" — agents/agent1.py RuleBasedAgent
                                     (one independent instance per seat)
                    Any callable with signature (obs, config=None) also works.
    player_id     : int (0–3)              — which seat the learning agent occupies
    n_players     : int (2–4)              — total players in the game
    encoder       : Encoder | None         — model.SAC.Encoder; created if None
    max_steps     : int                    — episode truncation limit
    reward_scheme : list[RewardSchemeN]    — ordered list of reward-scheme instances;
                    their returns are summed each step.  Hyperparameters (scales,
                    horizons, etc.) are configured on the instances themselves.
    """

    MAX_PLANETS: int = 40
    MAX_FLEETS:  int = 100
    STATE_DIM:   int = 14
    ACTION_DIM:  int = 4

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self,
                 opponent="random",
                 player_id: int = 0,
                 n_players: int = 2,
                 encoder=None,
                 max_steps: int = 500,
                 reward_scheme=None):
        super().__init__()

        if reward_scheme is None:
            reward_scheme = [RewardScheme1()]

        def _resolve(spec):
            if spec == "rule_based":
                from agents.agent1 import RuleBasedAgent
                return RuleBasedAgent()
            return spec

        if isinstance(opponent, list):
            opponents = [_resolve(o) for o in opponent]
        else:
            opponents = [_resolve(opponent) for _ in range(n_players - 1)]

        self.opponents     = opponents
        self.player_id     = player_id
        self.n_players     = n_players
        self.max_steps     = max_steps
        self.reward_scheme = reward_scheme

        if encoder is None:
            from model.SAC import Encoder
            encoder = Encoder(max_planets=self.MAX_PLANETS, max_fleets=self.MAX_FLEETS)
        self.encoder = encoder

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.MAX_PLANETS + self.MAX_FLEETS, self.STATE_DIM),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.MAX_PLANETS, self.ACTION_DIM),
            dtype=np.float32,
        )

        self._kaggle_env:   object | None       = None
        self._trainer:      object | None       = None
        self._current_obs:  object | None       = None
        self._initial_planets: np.ndarray | None = None
        self._time_step:    int                 = 0

    # ── internal ──────────────────────────────────────────────────────────────

    def _build_trainer(self):
        from kaggle_environments import make as kaggle_make
        self._kaggle_env = kaggle_make("orbit_wars", debug=False)
        agents = []
        opp_idx = 0
        for i in range(self.n_players):
            if i == self.player_id:
                agents.append(None)
            else:
                agents.append(self.opponents[opp_idx])
                opp_idx += 1
        self._trainer = self._kaggle_env.train(agents)

    def _encode(self, obs, time_step: int) -> np.ndarray:
        return encode_obs_as_player(
            self.encoder, obs, self._initial_planets,
            self.player_id, time_step,
        )

    # ── gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset stateful opponents (e.g. RuleBasedAgent) before each episode
        for opp in self.opponents:
            if hasattr(opp, "reset"):
                opp.reset()

        if self._trainer is None:
            self._build_trainer()

        raw_obs = self._trainer.reset()
        self._current_obs = raw_obs
        self._time_step   = 0

        planets_np, _, _, _, _ = _obs_to_arrays(raw_obs)
        s_planets, _ = _swap_perspective(
            planets_np, _EMPTY_FLEETS.copy(), self.player_id
        )
        self._initial_planets = s_planets

        state = self._encode(raw_obs, self._time_step)
        return state, {}

    def step(self, action: np.ndarray):
        assert self._trainer is not None, "Call reset() before step()."

        planets_np, _, _, omega, _ = _obs_to_arrays(self._current_obs)
        s_planets, _ = _swap_perspective(
            planets_np, _EMPTY_FLEETS.copy(), self.player_id
        )
        moves = decode_action(action, s_planets, omega)

        raw_obs, _kaggle_reward, done, info = self._trainer.step(moves)
        self._time_step += 1

        truncated  = self._time_step >= self.max_steps
        terminated = bool(done) and not truncated

        reward = 0
        for r in self.reward_scheme:
            reward += r(
                self._current_obs, raw_obs, self.player_id,
                done=terminated or truncated,
                n_players=self.n_players,
            )

        self._current_obs = raw_obs
        state = self._encode(raw_obs, self._time_step)

        return state, reward, terminated, truncated, info or {}

    def render(self, mode: str = "human", html_path: str | None = None,
               width: int = 800, height: int = 600):
        if self._kaggle_env is None:
            return
        if html_path is not None:
            html = self._kaggle_env.render(mode="html", width=width, height=height, playing=True)
            with open(html_path, "w") as f:
                f.write(html)
            print(f"Replay saved : {html_path}")
        else:
            print(self._kaggle_env.render(mode="ansi"))

    def close(self):
        self._trainer    = None
        self._kaggle_env = None
