"""
Kaggle Orbit Wars competition submission.

Entry point: ``agent(obs)`` -> list of ``[planet_id, angle_rad, num_ships]`` moves,
exactly the move format the kaggle orbit_wars interpreter consumes.

Pipeline (single source of truth shared with training):

    obs  ──_obs_to_arrays──►  planets/fleets arrays  (comets stripped)
         ──_swap_perspective► acting player relabelled to owner 0
    Encoder.encode  ────────► [MAX_PLANETS+MAX_FLEETS, STATE_DIM] state
    P_network.sample ───────► per-planet [-1,1] action rows (stochastic)
    action_decoder.decode_to_moves ► kaggle moves

The policy is sampled rather than taken at its mean: this SAC policy's mean
(``deterministic_action``) collapsed toward zero during training while its
behaviour lives in the stochastic head (sigma ≈ 0.9), so the agent only emits
launches when sampled — exactly as it acted during the training rollouts.

The model/encoder/decoder modules are copied verbatim from the training repo
(``model/SAC.py``, ``model/action_decoder.py``, ``utils/pos_encoding.py``); only
the two tiny pure-numpy obs helpers (``_obs_to_arrays`` / ``_swap_perspective``)
are inlined here so the submission has no gymnasium dependency.

Rename this file to ``main.py`` for submission.
"""
import os
import sys

import numpy as np
import torch

# Locate this file's directory. Kaggle loads a submission by exec'ing the source
# with NO ``__file__`` defined (and appends the submission dir to sys.path so the
# sibling ``model``/``utils`` packages import), so guard for that case.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
except NameError:                       # exec'd without __file__ (kaggle runtime)
    _HERE = None

from model.SAC import P_network, Encoder
from model import action_decoder as AD

# ── Config — MUST match the trained checkpoint (see model/SAC.py defaults) ──────
STATE_DIM   = 13
ACTION_DIM  = 4
MAX_PLANETS = 40
MAX_FLEETS  = 200
TANH_SCALE  = 0.2          # wedge-score saturation, matches env/decoder default
NUM_LAYERS  = 4
FFD         = 512

_DEVICE     = torch.device("cpu")
_MODEL_NAME = "final_model.pt"
_EMPTY_FLEETS = np.empty((0, 7), dtype=np.float32)


def _find_model():
    """Resolve final_model.pt without relying on __file__ (absent under kaggle).

    Kaggle appends the submission dir to sys.path, so the weights sit in one of
    those dirs (or the cwd / this file's dir when imported normally).
    """
    seen = []
    for d in ([_HERE] if _HERE else []) + list(sys.path) + [os.getcwd()] + ["/kaggle_simulations/agent"]:
        if not d or d in seen:
            continue
        seen.append(d)
        # print(d)
        p = os.path.join(d, _MODEL_NAME)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{_MODEL_NAME} not found on sys.path / cwd {d}")

# Lazily-built singletons (constructed on first agent() call so importing this
# module is cheap and never touches the filesystem before it is needed).
_encoder = None
_policy  = None


# ──────────────────────────────────────────────────────────────────────────────
# Observation parsing  (pure-numpy copies of env/orbit_wars.py helpers)
# ──────────────────────────────────────────────────────────────────────────────

def _get(obj, attr):
    """Attribute- or dict-access on a kaggle Observation."""
    return getattr(obj, attr, None) if not isinstance(obj, dict) else obj.get(attr)


def _safe_owner(v):
    return -1 if v is None else int(v)


def _parse_planet_row(p):
    if isinstance(p, (list, tuple)):
        return [p[0], _safe_owner(p[1]), p[2], p[3], p[4], p[5], p[6]]
    if isinstance(p, dict):
        return [p["id"], _safe_owner(p.get("owner")),
                p["x"], p["y"], p["radius"], p["ships"], p["production"]]
    return [p.id, _safe_owner(getattr(p, "owner", -1)),
            p.x, p.y, p.radius, p.ships, p.production]


def _parse_fleet_row(f):
    if isinstance(f, (list, tuple)):
        return [f[0], _safe_owner(f[1]), f[2], f[3], f[4], f[5], f[6]]
    if isinstance(f, dict):
        return [f["id"], _safe_owner(f.get("owner")),
                f["x"], f["y"], f["angle"], f["from_planet_id"], f["ships"]]
    return [f.id, _safe_owner(getattr(f, "owner", -1)),
            f.x, f.y, f.angle, f.from_planet_id, f.ships]


def _rows_to_array(raw, parse_fn, empty):
    if raw is None:
        return empty.copy()
    if isinstance(raw, np.ndarray) and raw.ndim == 2:
        return raw.astype(np.float32)
    rows = [parse_fn(r) for r in raw]
    return np.array(rows, dtype=np.float32) if rows else empty.copy()


def _obs_to_arrays(obs):
    """Parse a kaggle obs into (planets, fleets, omega) with comets stripped.

    planets : [n, 7]  [id, owner, x, y, radius, ships, production]
    fleets  : [m, 7]  [id, owner, x, y, angle, from_planet_id, ships]
    """
    planets_np = _rows_to_array(_get(obs, "planets"), _parse_planet_row,
                                np.empty((0, 7), dtype=np.float32))
    fleets_np  = _rows_to_array(_get(obs, "fleets"), _parse_fleet_row, _EMPTY_FLEETS)
    omega = float(_get(obs, "angular_velocity") or 0.0)

    raw_comets = _get(obs, "comet_planet_ids") or []
    comet_ids = (raw_comets.astype(np.int32) if isinstance(raw_comets, np.ndarray)
                 else np.array(list(raw_comets), dtype=np.int32))
    if comet_ids.size > 0 and planets_np.shape[0] > 0:
        keep = ~np.isin(planets_np[:, 0].astype(np.int32), comet_ids)
        planets_np = planets_np[keep]

    return planets_np, fleets_np, omega


def _swap_perspective(planets, fleets, player_id):
    """Relabel owners so the acting player appears as owner 0 (swap 0 ↔ player_id).

    Positions / ids are untouched, so decoded moves carry the real planet ids.
    """
    if player_id == 0:
        return planets.copy(), fleets.copy()

    pid = float(player_id)
    sp = planets.copy()
    op = sp[:, 1]
    sp[:, 1] = np.where(op == 0.0, pid, np.where(op == pid, 0.0, op))

    sf = fleets.copy()
    if sf.shape[0] > 0:
        of = sf[:, 1]
        sf[:, 1] = np.where(of == 0.0, pid, np.where(of == pid, 0.0, of))
    return sp, sf


def _initial_planets(obs):
    """First-step planet array [n, 7] used by the encoder for orbit detection.

    The kaggle obs carries ``initial_planets`` for every active planet; only
    ids / positions / radii matter (get_planet_mapping), so comet rows present
    here are harmless.
    """
    raw = _get(obs, "initial_planets")
    return _rows_to_array(raw, _parse_planet_row, np.empty((0, 7), dtype=np.float32))


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_loaded():
    global _encoder, _policy
    if _policy is None:
        _encoder = Encoder(max_planets=MAX_PLANETS, max_fleets=MAX_FLEETS)
        net = P_network(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                        max_planets=MAX_PLANETS, max_fleets=MAX_FLEETS, num_layers= NUM_LAYERS, dim_feedforward= FFD)
        

        
        ckpt = torch.load(_find_model(), map_location=_DEVICE, weights_only=False)
       
        state_dict = {k.replace("_orig_mod.", "", 1): v
                      for k, v in ckpt["policy_net"].items()}
        net.load_state_dict(state_dict)
        net.eval()
        net.to(_DEVICE)
        _policy = net
        # print("load success")
    return _encoder, _policy


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

# @torch.no_grad()
def agent(obs):
    with torch.no_grad():
        encoder, policy = _ensure_loaded()
        # print(policy)

        player = _safe_owner(_get(obs, "player"))
        if player < 0:
            player = 0
        step = int(_get(obs, "step") or 0)

        planets_np, fleets_np, omega = _obs_to_arrays(obs)
        if planets_np.shape[0] == 0:
            return []

        # Re-label so we are owner 0, exactly as the policy was trained.
        s_planets, s_fleets = _swap_perspective(planets_np, fleets_np, player)
        init_planets = _initial_planets(obs)
        if init_planets.shape[0] == 0:
            init_planets = s_planets

        # Encode → policy → per-planet action rows in [-1, 1].
        state, _ = encoder.encode(s_planets, s_fleets, init_planets, omega, step,
                                apply_padding=True)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
        action, _ = policy.sample(state_t)                     # [1, MAX_PLANETS, 4]
        action_np = action[0].cpu().numpy()

        # Decode against the player-0-perspective planets (owner 0 == us); returned
        # planet ids are the real ones since the swap leaves ids untouched.
        moves = AD.decode_to_moves(action_np, s_planets, omega,
                                player=0, tanh_scale=TANH_SCALE)
        return moves
