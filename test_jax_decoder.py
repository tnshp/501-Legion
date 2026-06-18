"""Parity / correctness tests for the hybrid fleet-into-planet decoder.

  * binding   — a hand-built state with known fleets exercises the incoming/
                outgoing explicit slots, the pooled summary, the meta row, and a
                fleet that is BOTH outgoing-from one planet and incoming-to another.
  * parity    — the on-device JAX builder (jax_env/jax_obs.py, used in training)
                vs the NumPy live-game Encoder (model/SAC.py, used at submission)
                agree on the same scenario.
  * shape/jit — the jitted, vmapped extractor compiles and returns [B, 41, 35].

Run:  JAX_PLATFORMS=cpu python test_jax_decoder.py
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import jax.numpy as jnp

from jax_env.env_types import GameState, PlanetState, FleetState, CometData
from jax_env.constants import MAX_PLANETS, MAX_FLEETS
from jax_env.jax_obs import make_extract_obs, TOKEN_DIM, _NET_MP, K_IN, K_OUT
from model.SAC import Encoder

CENTER = 50.0

# Scenario (owners already in "us = 0" frame):
#   planet 0: us (owner 0)   at (30,50)  r2  10 ships  prod2
#   planet 1: enemy (owner1) at (70,50)  r2   8 ships  prod1
#   planet 2: neutral (-1)   at (50,82)  r2   5 ships  prod1
# fleets:
#   A: enemy (1), at (10,50) heading +x  → passes through planet0  : incoming→p0
#   B: us    (0), from planet0, at (32,50) heading +x → through p1 : out←p0, in→p1
#   C: enemy (1), from planet1, at (70,48) heading -y, 7 ships     : out←p1
_PLANETS = np.array([   # id, owner, x, y, r, ships, prod
    [0,  0, 30.0, 50.0, 2.0, 10.0, 2.0],
    [1,  1, 70.0, 50.0, 2.0,  8.0, 1.0],
    [2, -1, 50.0, 82.0, 2.0,  5.0, 1.0],
], dtype=np.float32)
_FLEETS = np.array([    # id, owner, x, y, angle, from_planet_id, ships
    [0, 1, 10.0, 50.0, 0.0,          99, 20.0],   # A
    [1, 0, 32.0, 50.0, 0.0,           0, 15.0],   # B
    [2, 1, 70.0, 48.0, -np.pi / 2.0,  1,  7.0],   # C
], dtype=np.float32)
_OMEGA = 0.0          # static planets → no orbital lead drift, clean parity
_STEP  = 10


def _make_state():
    """Build a single-env batched GameState from the scenario above."""
    n = _PLANETS.shape[0]
    px = np.zeros(MAX_PLANETS, np.float32); py = np.zeros(MAX_PLANETS, np.float32)
    pr = np.zeros(MAX_PLANETS, np.float32); psh = np.zeros(MAX_PLANETS, np.float32)
    ppr = np.zeros(MAX_PLANETS, np.int32);  pow_ = np.full(MAX_PLANETS, -1, np.int32)
    pact = np.zeros(MAX_PLANETS, bool)
    px[:n] = _PLANETS[:, 2]; py[:n] = _PLANETS[:, 3]; pr[:n] = _PLANETS[:, 4]
    psh[:n] = _PLANETS[:, 5]; ppr[:n] = _PLANETS[:, 6].astype(np.int32)
    pow_[:n] = _PLANETS[:, 1].astype(np.int32); pact[:n] = True

    planets = PlanetState(
        x=jnp.array(px), y=jnp.array(py), init_x=jnp.array(px), init_y=jnp.array(py),
        radius=jnp.array(pr), ships=jnp.array(psh), production=jnp.array(ppr),
        owner=jnp.array(pow_), active=jnp.array(pact),
        is_comet=jnp.zeros(MAX_PLANETS, bool),
    )

    m = _FLEETS.shape[0]
    fx = np.zeros(MAX_FLEETS, np.float32); fy = np.zeros(MAX_FLEETS, np.float32)
    fa = np.zeros(MAX_FLEETS, np.float32); fsh = np.zeros(MAX_FLEETS, np.float32)
    fow = np.full(MAX_FLEETS, -1, np.int32); ffr = np.full(MAX_FLEETS, -1, np.int32)
    fact = np.zeros(MAX_FLEETS, bool)
    fx[:m] = _FLEETS[:, 2]; fy[:m] = _FLEETS[:, 3]; fa[:m] = _FLEETS[:, 4]
    fsh[:m] = _FLEETS[:, 6]; fow[:m] = _FLEETS[:, 1].astype(np.int32)
    ffr[:m] = _FLEETS[:, 5].astype(np.int32); fact[:m] = True

    fleets = FleetState(
        x=jnp.array(fx), y=jnp.array(fy), angle=jnp.array(fa), ships=jnp.array(fsh),
        owner=jnp.array(fow), from_planet=jnp.array(ffr), active=jnp.array(fact),
    )

    comets = CometData(
        paths=jnp.zeros((5, 4, 40, 2), jnp.float32),
        path_lengths=jnp.zeros(5, jnp.int32), spawn_ships=jnp.zeros(5, jnp.int32),
        path_indices=jnp.full(5, -1, jnp.int32), planet_slots=jnp.zeros((5, 4), jnp.int32),
    )

    state = GameState(
        planets=planets, fleets=fleets, comets=comets,
        step=jnp.int32(_STEP), done=jnp.bool_(False),
        rewards=jnp.zeros(2, jnp.float32), angular_velocity=jnp.float32(_OMEGA),
        next_fleet_slot=jnp.int32(m), num_players=jnp.int32(2),
    )
    # add batch dim
    return jax.tree_util.tree_map(lambda x: x[None], state)


# Column offsets inside a token (mirror jax_obs / SAC layout)
_IN0 = 12               # incoming slot 0 = [ships, eta, owner]
_OUT0 = 12 + K_IN * 3   # outgoing slot 0 = [ships, sinθ, cosθ, speed]
_INP = _OUT0 + K_OUT * 4
_OUTP = _INP + 5


def test_binding():
    obs = np.asarray(make_extract_obs(0)(_make_state()))[0]   # [41, 35]
    assert obs.shape == (_NET_MP + 1, TOKEN_DIM), obs.shape

    p0, p1, p2 = obs[0], obs[1], obs[2]

    # planet0 (us): incoming = enemy fleet A (20 ships, owner 1)
    assert abs(p0[_IN0 + 0] - 20.0) < 1e-3, p0[_IN0:_IN0 + 3]
    assert abs(p0[_IN0 + 2] - 1.0)  < 1e-3, "incoming owner should be 1 (enemy)"
    assert 0.0 < p0[_IN0 + 1] < 1.0, "incoming eta in (0,1)"
    # planet0 outgoing = our fleet B (15 ships)
    assert abs(p0[_OUT0 + 0] - 15.0) < 1e-3, p0[_OUT0:_OUT0 + 4]
    assert abs(p0[_OUTP + 1] - np.log1p(15.0)) < 1e-3, "out pool Σships log1p(15)"

    # planet1 (enemy): incoming = our fleet B (15 ships, owner 0) — B is BOTH
    # outgoing-from-p0 and incoming-to-p1.
    assert abs(p1[_IN0 + 0] - 15.0) < 1e-3, p1[_IN0:_IN0 + 3]
    assert abs(p1[_IN0 + 2] - 0.0)  < 1e-3, "incoming owner should be 0 (us)"
    # planet1 outgoing = enemy fleet C (7 ships)
    assert abs(p1[_OUT0 + 0] - 7.0) < 1e-3, p1[_OUT0:_OUT0 + 4]

    # planet2 (neutral): no fleet relates to it
    assert p2[_IN0:_IN0 + 3].sum() == 0.0 and p2[_OUT0:_OUT0 + 4].sum() == 0.0

    # incoming pooled count: p0 and p1 each have exactly 1 hostile inbound
    assert abs(p0[_INP] - np.log1p(1.0)) < 1e-3
    assert abs(p1[_INP] - np.log1p(1.0)) < 1e-3

    # meta row: player0 has 1 planet, player1 has 1 planet, players 2/3 none
    meta = obs[_NET_MP, :12]
    assert abs(meta[0] - 1.0) < 1e-3 and abs(meta[1] - 1.0) < 1e-3
    assert meta[2] == 0.0 and meta[3] == 0.0
    print("  binding ........ OK")


def test_parity_with_encoder():
    obs = np.asarray(make_extract_obs(0)(_make_state()))[0]          # [41, 35]

    enc = Encoder(max_planets=_NET_MP, max_fleets=100)
    enc_state, _ = enc.encode(_PLANETS.copy(), _FLEETS.copy(),
                              _PLANETS.copy(), _OMEGA, _STEP)         # [41, 35]

    diff = np.abs(obs - enc_state)
    i, j = np.unravel_index(np.argmax(diff), diff.shape)
    assert diff.max() < 1e-3, f"max |jax-encoder|={diff.max():.2e} at token {i} col {j}"
    print(f"  jax↔Encoder .... OK (max diff {diff.max():.2e})")


def test_shape_jit_batch():
    state = _make_state()
    state4 = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (4,) + x.shape[1:]), state)
    obs = np.asarray(make_extract_obs(0)(state4))
    assert obs.shape == (4, _NET_MP + 1, TOKEN_DIM), obs.shape
    assert np.isfinite(obs).all()
    print("  shape/jit/batch  OK", obs.shape)


if __name__ == "__main__":
    print("hybrid decoder tests:")
    test_binding()
    test_parity_with_encoder()
    test_shape_jit_batch()
    print("ALL DECODER TESTS PASSED")
