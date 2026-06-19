"""On-device (JAX) observation extraction — hybrid fleet-into-planet decoder.

Approach E: instead of emitting separate fleet tokens, each fleet's information is
folded into the PLANET token it relates to, collapsing the transformer sequence
from ``NET_MP + NET_MF`` (≈140) down to ``NET_MP + 1`` (41) tokens.  Per planet:

  * ``K_IN``  explicit "most imminent incoming HOSTILE fleet" slots  [ships, eta, owner]
  * ``K_OUT`` explicit "largest OUTGOING friendly fleet" slots        [ships, sinθ, cosθ, speed]
  * a pooled summary of ALL incoming / outgoing fleets (so the long tail past the
    explicit slots is never silently dropped)

plus the base planet fields.  Token width = 12 + K_IN·3 + K_OUT·4 + 5 + 4 = 35.

Binding:
  * outgoing is exact — a fleet is outgoing from planet p iff ``from_planet == p``
    (and the fleet is ours, i.e. owner == planet owner after the perspective swap).
  * incoming is geometric — the swept-segment look-ahead ``_fleet_planet_sweep``
    (reused from jax_opponents) gives which planet each fleet will hit and the
    arrival tick; a fleet is incoming-hostile to p iff it hits p and its owner
    differs from p's owner.

The acting player's units always appear as owner 0 (perspective swap, ``pid``).

Because the fleet tokens are gone, the model can no longer reconstruct each
player's TOTAL ship strength (incl. in-flight) by summing token rows, so the
12-dim per-player aggregate (planet count / production / ship strength) is computed
here and packed into a trailing META ROW (index NET_MP); the model reads it back
and projects it into its single "metadata" token.  Output: ``[B, NET_MP+1, 35]``.

``make_extract_obs(pid)`` returns a jitted, vmapped extractor.  The NumPy live-game
equivalent for submission is ``model.SAC.Encoder.encode`` — they must agree
(test_jax_decoder.py).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .jax_opponents import (
    _fleet_planet_sweep,
    _fleet_speed,
    CENTER as _CENTER,
    ROT_LIMIT as _ROT_LIMIT,
)

_NET_MP = 40            # planet tokens (also the action planets)
K_IN    = 2             # explicit incoming-hostile fleet slots per planet
K_OUT   = 2             # explicit outgoing-friendly fleet slots per planet
HORIZON = 60.0          # look-ahead horizon used by _fleet_planet_sweep (ticks)
META_DIM = 12           # 4 players × [planet_count, log1p production, log1p ships]

# Token width: base(12) + incoming slots(K_IN·3) + outgoing slots(K_OUT·4)
#            + incoming pooled(5) + outgoing pooled(4)
TOKEN_DIM = 12 + K_IN * 3 + K_OUT * 4 + 5 + 4    # = 35
SEQ_LEN   = _NET_MP + 1                           # planet tokens + meta row


def _swap(owner, pid):
    """Perspective swap: pid's units → owner 0, owner-0 units → pid."""
    if pid == 0:
        return owner
    return jnp.where(owner == 0, pid, jnp.where(owner == pid, 0, owner))


def _topk_per_col(score, k: int, take_min: bool):
    """Indices of the top-k rows per column of ``score`` ([NF, NMP]).

    Returns a length-k list of [NMP] int index arrays.  ``take_min`` selects the
    smallest values (incoming, by ETA); otherwise the largest (outgoing, by
    ships).  Picks are removed between rounds via a scatter so each round yields
    a distinct fleet.  k is small (2) so the loop is fully unrolled under jit.
    """
    NMP = score.shape[1]
    cols = jnp.arange(NMP)
    fill = jnp.inf if take_min else -jnp.inf
    out = []
    s = score
    for _ in range(k):
        idx = jnp.argmin(s, axis=0) if take_min else jnp.argmax(s, axis=0)  # [NMP]
        out.append(idx)
        s = s.at[idx, cols].set(fill)
    return out


def _obs_single(state, pid: int = 0):
    NMP = _NET_MP
    pl, fl = state.planets, state.fleets

    p_owner = _swap(pl.owner, pid)
    f_owner = _swap(fl.owner, pid)

    ang_vel = state.angular_velocity.astype(jnp.float32)
    t_step  = state.step.astype(jnp.float32)

    # ── Planet arrays (first NMP) ────────────────────────────────────────────
    px = pl.x[:NMP]; py = pl.y[:NMP]; pr = pl.radius[:NMP]
    ps = pl.ships[:NMP]; pp = pl.production[:NMP].astype(jnp.float32)
    po = p_owner[:NMP]
    valid_p = (pl.active[:NMP] & ~pl.is_comet[:NMP]).astype(jnp.float32)

    p_oh = jnp.eye(4, dtype=jnp.float32)[jnp.clip(po, 0, 3)]
    p_oh = jnp.where((po == -1)[:, None], 0.0, p_oh)

    dx = pl.init_x[:NMP] - _CENTER
    dy = pl.init_y[:NMP] - _CENTER
    p_mov = ((jnp.sqrt(dx * dx + dy * dy) + pr) < _ROT_LIMIT).astype(jnp.float32)

    base = jnp.stack([
        p_oh[:, 0], p_oh[:, 1], p_oh[:, 2], p_oh[:, 3],
        pr, ps, pp, p_mov, jnp.full((NMP,), ang_vel),
        px, py, jnp.full((NMP,), t_step),
    ], axis=-1)                                          # [NMP, 12]

    # ── Fleet arrays (ALL fleets) ────────────────────────────────────────────
    fx = fl.x; fy = fl.y; fa = fl.angle; fs = fl.ships
    fo = f_owner; fact = fl.active; ffrom = fl.from_planet
    NF = fs.shape[0]

    # Geometric look-ahead: which planet each fleet hits and when.
    hit, arrive = _fleet_planet_sweep(px, py, pr, fx, fy, fa, fs, fact)  # [NF, NMP]
    eta = arrive / HORIZON                                # ∈ [0, 1]

    # Incoming = fleet hits planet p AND fleet owner ≠ p's owner (hostile).
    hostile = (hit & fact[:, None]
               & (fo != -1)[:, None]
               & (fo[:, None] != po[None, :]))           # [NF, NMP]
    fs_col  = fs[:, None]                                 # [NF, 1]

    # Incoming explicit slots: K_IN soonest-arriving hostile fleets per planet.
    in_score = jnp.where(hostile, eta, jnp.inf)          # [NF, NMP]
    in_idxs  = _topk_per_col(in_score, K_IN, take_min=True)
    cols = jnp.arange(NMP)
    in_slots = []
    for idx in in_idxs:
        ok = hostile[idx, cols]                          # [NMP] this pick valid?
        in_slots.append(jnp.stack([
            jnp.where(ok, fs[idx], 0.0),                  # ships
            jnp.where(ok, eta[idx, cols], 0.0),          # eta (normalised)
            jnp.where(ok, fo[idx].astype(jnp.float32), 0.0),  # opponent owner id
        ], axis=-1))                                     # [NMP, 3]
    in_slots = jnp.concatenate(in_slots, axis=-1)        # [NMP, K_IN*3]

    # Incoming pooled summary over ALL hostile fleets.
    in_cnt   = hostile.sum(0).astype(jnp.float32)        # [NMP]
    in_shsum = (fs_col * hostile).sum(0)
    eta_h    = jnp.where(hostile, eta, jnp.inf)
    in_soon  = eta_h.min(0)
    in_soon  = jnp.where(jnp.isfinite(in_soon), in_soon, 0.0)
    in_wnum  = (eta * fs_col * hostile).sum(0)
    in_wmean = jnp.where(in_shsum > 0, in_wnum / jnp.where(in_shsum > 0, in_shsum, 1.0), 0.0)
    in_maxsh = jnp.where(hostile, fs_col, 0.0).max(0)
    in_pool  = jnp.stack([
        jnp.log1p(in_cnt), jnp.log1p(in_shsum), in_soon, in_wmean, jnp.log1p(in_maxsh),
    ], axis=-1)                                          # [NMP, 5]

    # Outgoing = fleet launched from planet p AND still ours (owner == p owner).
    out_mask = (fact[:, None]
                & (ffrom[:, None] == cols[None, :])
                & (fo[:, None] == po[None, :]))          # [NF, NMP]

    # Outgoing explicit slots: K_OUT largest-by-ships friendly fleets per planet.
    out_score = jnp.where(out_mask, fs_col, -1.0)        # [NF, NMP]
    out_idxs  = _topk_per_col(out_score, K_OUT, take_min=False)
    out_slots = []
    for idx in out_idxs:
        ok = out_mask[idx, cols] & (fs[idx] > 0.0)
        ships = jnp.where(ok, fs[idx], 0.0)
        ang   = fa[idx]
        out_slots.append(jnp.stack([
            ships,
            jnp.where(ok, jnp.sin(ang), 0.0),
            jnp.where(ok, jnp.cos(ang), 0.0),
            jnp.where(ok, _fleet_speed(jnp.maximum(ships, 1.0)), 0.0),
        ], axis=-1))                                     # [NMP, 4]
    out_slots = jnp.concatenate(out_slots, axis=-1)      # [NMP, K_OUT*4]

    # Outgoing pooled summary over ALL friendly outgoing fleets.
    out_cnt   = out_mask.sum(0).astype(jnp.float32)
    out_shsum = (fs_col * out_mask).sum(0)
    safe_den  = jnp.where(out_shsum > 0, out_shsum, 1.0)
    out_wsin  = jnp.where(out_shsum > 0, (jnp.sin(fa)[:, None] * fs_col * out_mask).sum(0) / safe_den, 0.0)
    out_wcos  = jnp.where(out_shsum > 0, (jnp.cos(fa)[:, None] * fs_col * out_mask).sum(0) / safe_den, 0.0)
    out_pool  = jnp.stack([
        jnp.log1p(out_cnt), jnp.log1p(out_shsum), out_wsin, out_wcos,
    ], axis=-1)                                          # [NMP, 4]

    token = jnp.concatenate(
        [base, in_slots, out_slots, in_pool, out_pool], axis=-1)   # [NMP, TOKEN_DIM]
    token = token * valid_p[:, None]

    # ── Meta row: per-player [planet_count, log1p production, log1p ships] ────
    p_own_n = po                                          # swapped, first NMP
    p_act_n = (pl.active[:NMP] & ~pl.is_comet[:NMP])
    cnt, prod, ships = [], [], []
    for player in range(4):
        pm = (p_own_n == player) & p_act_n               # [NMP]
        fm = (fo == player) & fact                       # [NF]
        cnt.append(pm.sum().astype(jnp.float32))
        prod.append(jnp.log1p((pp * pm).sum()))
        ships.append(jnp.log1p((ps * pm).sum() + (fs * fm).sum()))
    meta = jnp.stack(cnt + prod + ships)                 # [12]
    meta_row = jnp.zeros((TOKEN_DIM,), jnp.float32).at[:META_DIM].set(meta)

    return jnp.concatenate([token, meta_row[None, :]], axis=0).astype(jnp.float32)


def make_extract_obs(pid: int = 0):
    """Return a jitted, batched obs extractor: ``state(batched) -> [B, NET_MP+1, TOKEN_DIM]``."""
    return jax.jit(jax.vmap(lambda s: _obs_single(s, pid)))
