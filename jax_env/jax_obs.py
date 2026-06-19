"""On-device (JAX) observation extraction for the vectorised Orbit Wars env.

Single-environment port of ``JaxVecEnvAdapter._extract_obs`` (which mirrors
``model.SAC.Encoder.encode``).  ``make_extract_obs`` returns a jitted, vmapped
``extract(state) -> [B, NET_MP+NET_MF, 13]`` so the obs tensor is built on the GPU
with the engine step instead of in host NumPy; only the final array is copied to
host for torch.

Token layout (matches the NumPy version exactly — verified in test_jax_port.py):
  Planet: [owner_oh×4, radius, ships, production, moving, ang_vel, 0, x, y, ts]
  Fleet:  [owner_oh×4, angle,  ships, speed,      0,      0,       1, x, y, ts]
Fleet tokens are the top NET_MF fleets by ship count.  ``pid`` swaps owners
(0 ↔ pid) so the acting player always appears as owner 0.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_NET_MP    = 40
_NET_MF    = 100
_CENTER    = 50.0
_ROT_LIMIT = 50.0
_MAX_SPEED = 6.0


def _fleet_speed(ships):
    s = jnp.maximum(1.0, ships)
    return jnp.minimum(
        _MAX_SPEED,
        1.0 + (_MAX_SPEED - 1.0) * (jnp.log(s) / jnp.log(1000.0)) ** 1.5,
    )


def _obs_single(state, pid: int = 0):
    NMP, NMF = _NET_MP, _NET_MF
    pl, fl = state.planets, state.fleets

    # Perspective swap: pid's units appear as owner 0.
    if pid != 0:
        p_owner = jnp.where(pl.owner == 0, pid,
                            jnp.where(pl.owner == pid, 0, pl.owner))
        f_owner = jnp.where(fl.owner == 0, pid,
                            jnp.where(fl.owner == pid, 0, fl.owner))
    else:
        p_owner, f_owner = pl.owner, fl.owner

    ang_vel = state.angular_velocity.astype(jnp.float32)   # scalar
    t_step  = state.step.astype(jnp.float32)               # scalar

    # ── Planet tokens ────────────────────────────────────────────────────────
    po   = p_owner[:NMP]
    p_oh = jnp.eye(4, dtype=jnp.float32)[jnp.clip(po, 0, 3)]            # [NMP,4]
    p_oh = jnp.where((po == -1)[:, None], 0.0, p_oh)

    dx    = pl.init_x[:NMP] - _CENTER
    dy    = pl.init_y[:NMP] - _CENTER
    p_mov = ((jnp.sqrt(dx * dx + dy * dy) + pl.radius[:NMP]) < _ROT_LIMIT
             ).astype(jnp.float32)

    p_tokens = jnp.stack([
        p_oh[:, 0], p_oh[:, 1], p_oh[:, 2], p_oh[:, 3],
        pl.radius[:NMP], pl.ships[:NMP], pl.production[:NMP].astype(jnp.float32),
        p_mov, jnp.full((NMP,), ang_vel),
        jnp.zeros((NMP,), jnp.float32),                    # is_fleet = 0
        pl.x[:NMP], pl.y[:NMP], jnp.full((NMP,), t_step),
    ], axis=-1)                                            # [NMP, 13]
    valid_p  = (pl.active[:NMP] & ~pl.is_comet[:NMP]).astype(jnp.float32)
    p_tokens = p_tokens * valid_p[:, None]

    # ── Fleet tokens (top NMF by ship count) ─────────────────────────────────
    sort_ships = jnp.where(fl.active, fl.ships, -1.0)
    top_idx    = jnp.argsort(-sort_ships)[:NMF]
    fo   = f_owner[top_idx]
    fs   = fl.ships[top_idx]
    fv   = fl.active[top_idx].astype(jnp.float32)
    f_oh = jnp.eye(4, dtype=jnp.float32)[jnp.clip(fo, 0, 3)]
    f_oh = jnp.where((fo == -1)[:, None], 0.0, f_oh)

    f_tokens = jnp.stack([
        f_oh[:, 0], f_oh[:, 1], f_oh[:, 2], f_oh[:, 3],
        fl.angle[top_idx], fs, _fleet_speed(fs),
        jnp.zeros((NMF,), jnp.float32), jnp.zeros((NMF,), jnp.float32),
        jnp.ones((NMF,), jnp.float32),                     # is_fleet = 1
        fl.x[top_idx], fl.y[top_idx], jnp.full((NMF,), t_step),
    ], axis=-1)                                            # [NMF, 13]
    f_tokens = f_tokens * fv[:, None]

    return jnp.concatenate([p_tokens, f_tokens], axis=0).astype(jnp.float32)


def make_extract_obs(pid: int = 0):
    """Return a jitted, batched obs extractor: ``state(batched) -> [B, seq, 13]``."""
    return jax.jit(jax.vmap(lambda s: _obs_single(s, pid)))
