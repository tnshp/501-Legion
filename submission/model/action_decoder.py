"""
Standalone PyTorch action decoder for Orbit Wars.

Converts a policy network's raw output ``[B, P, 4]`` (per-planet 4-vectors in
[-1, 1]) into engine-ready launches — a target planet, a ship count, and a
launch angle for every owned source planet — using the SAME pairwise-wedge
attention as ``env/orbit_wars.py``.  It depends only on ``torch`` (no JAX, no
``env`` / ``kaggle_environments`` import), so a competition submission is just

    Encoder → P_network → ActionDecoder

and the JAX training adapter can route its player-0 decode through the very same
module (single source of truth).

Launch-angle model
------------------
The kaggle engine flies a fleet in a straight line at a FIXED angle and awards a
capture via continuous swept-pair collision against the target's (possibly
orbiting) position.  We therefore aim with a lead and then verify the shot:

* **Static target** (``orbital_radius + radius ≥ ROTATION_RADIUS_LIMIT`` or
  ω = 0): the aim is pure geometry, ``θ = atan2(ty − my, tx − mx)``.

* **Orbiting target**: intercepting a point on a circle with a constant-speed
  straight pursuer is transcendental — ``|T(t) − L| = s·t`` has ``t`` both inside
  the orbit's trig and squared on the right, so there is no closed form.  We
  solve the arrival time ``t*`` by a few fixed-point iterations
  ``t ← |T(t) − L| / s`` (the standard iterated-lead method; converges fast while
  the fleet outruns the target's tangential speed) and aim at ``T(t*)``.

Either way the chosen angle is then checked ONCE against the engine's exact
discrete model (``_verify_connects``): the fleet is re-flown tick by tick and the
launch is kept only if the first planet its path crosses is the intended target,
and it is not first consumed by the sun or flung off the board.  This single
verify replaces the reference's up-to-60 per-candidate re-simulations.

All public functions are batched over ``B`` environments and vectorised over
planets; the only python loop is the bounded per-tick flight in the verifier.
"""
from __future__ import annotations

import math
import torch

# ── Engine constants (mirror env/orbit_wars.py / jax_env.constants) ────────────
_BOARD_SIZE            = 100.0
_CENTER                = 50.0
_SUN_RADIUS            = 10.0
_ROTATION_RADIUS_LIMIT = 50.0
_MAX_SHIP_SPEED        = 6.0
_INTERCEPT_ITERS       = 8        # fixed-point lead iterations for orbiting targets
_VERIFY_TICKS          = 600      # hard cap on the flight-verification horizon

# Upper-triangle index pairs for the d=4 wedge (6 independent components).
_PIDX, _QIDX = torch.triu_indices(4, 4, offset=1).unbind(0)


# ──────────────────────────────────────────────────────────────────────────────
# Low-level geometry (vectorised; mirror the interpreter's exact maths)
# ──────────────────────────────────────────────────────────────────────────────

def _fleet_speed(ships: torch.Tensor) -> torch.Tensor:
    """Board-units/tick for a fleet of `ships` (interpreter step 3), elementwise."""
    s = ships.clamp(min=1.0)
    speed = 1.0 + (_MAX_SHIP_SPEED - 1.0) * (torch.log(s) / math.log(1000.0)) ** 1.5
    return speed.clamp(max=_MAX_SHIP_SPEED)


def _swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
    """Vectorised continuous swept-pair collision (interpreter copy).

    True iff a fleet moving A→B and a planet moving P0→P1 come within `r` for some
    t in [0, 1].  All inputs are broadcastable tensors; returns a bool tensor.
    """
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r

    disc = b * b - 4.0 * a * c
    sq = torch.sqrt(disc.clamp(min=0.0))
    denom = (2.0 * a).clamp(min=1e-12)            # guard; |a|<1e-12 handled below
    t1 = (-b - sq) / denom
    t2 = (-b + sq) / denom
    moving_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    static_hit = c <= 0.0                          # near-zero relative motion
    return torch.where(a < 1e-12, static_hit, moving_hit)


def _point_to_segment_dist(px, py, ax, ay, bx, by):
    """Distance from point P to segment A–B (interpreter copy), vectorised."""
    abx = bx - ax
    aby = by - ay
    l2 = abx * abx + aby * aby
    t = ((px - ax) * abx + (py - ay) * aby) / l2.clamp(min=1e-12)
    t = t.clamp(0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    d = torch.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    # l2 == 0 (degenerate segment) → distance to the point A
    d0 = torch.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    return torch.where(l2 == 0.0, d0, d)


def _orbit_params(p_x, p_y, p_r, omega):
    """Per-planet orbital kinematics. Returns (orb, ang0, moving) elementwise.

    A planet orbits only when orbital_radius + radius < ROTATION_RADIUS_LIMIT and
    the system's angular velocity is non-zero; otherwise it is static.
    `omega` broadcasts against the planet dims (e.g. [B, 1] vs [B, P]).
    """
    dx = p_x - _CENTER
    dy = p_y - _CENTER
    orb = torch.sqrt(dx * dx + dy * dy)
    ang0 = torch.atan2(dy, dx)
    moving = (omega != 0.0) & (orb + p_r < _ROTATION_RADIUS_LIMIT)
    return orb, ang0, moving


# ──────────────────────────────────────────────────────────────────────────────
# Wedge attention + target / fraction selection
# ──────────────────────────────────────────────────────────────────────────────

def wedge_scores(action: torch.Tensor, tanh_scale: float = 0.2) -> torch.Tensor:
    """Pairwise bivector attention score matrix.

    action : [B, P, 4]  →  [B, P, P] = tanh(tanh_scale · Σ wedge(a_i, a_j)),
    identical to env/orbit_wars.py pairwise_wedge(a, a).sum(-1) then tanh.
    """
    ap = action[..., _PIDX]                         # [B, P, 6]
    aq = action[..., _QIDX]                         # [B, P, 6]
    wedge = (torch.einsum("bic,bjc->bij", ap, aq)
             - torch.einsum("bic,bjc->bij", aq, ap))
    return torch.tanh(tanh_scale * wedge)


def _select_targets(action, owned, valid, p_ships, tanh_scale):
    """Per source planet: pick the argmax-scored valid target and ship count.

    Returns (prelim, tgt_idx, num_ships, frac) where `prelim` [B, P] bool marks
    sources that pass every gate EXCEPT the launch-angle verify:
      source owned & ships > 1, best target valid & not self & score > 0,
      num_ships = min(⌊frac·ships⌋, ships−1) and num_ships > target garrison.
    """
    B, P, _ = action.shape
    scores = wedge_scores(action, tanh_scale)       # [B, P, P]

    eye = torch.eye(P, dtype=torch.bool, device=action.device)
    # Restrict targets to real planets and exclude self; argmax then needs > 0.
    col_ok = valid[:, None, :] & ~eye[None]         # [B, P, P]
    masked = torch.where(col_ok, scores, scores.new_full((), float("-inf")))

    best_score, tgt_idx = masked.max(dim=2)         # [B, P]
    frac = best_score.clamp(0.0, 1.0)

    # The reference int()s garrisons before the ship math; floor to match exactly.
    ships = torch.floor(p_ships)
    num_ships = torch.minimum(torch.floor(frac * ships), ships - 1.0)
    tgt_ships = torch.floor(torch.gather(p_ships, 1, tgt_idx))

    src_ok = owned & (ships > 1.0)
    prelim = src_ok & (best_score > 0.0) & (num_ships > tgt_ships)
    return prelim, tgt_idx, num_ships, frac


# ──────────────────────────────────────────────────────────────────────────────
# Launch angle: analytic lead + single engine-verify
# ──────────────────────────────────────────────────────────────────────────────

def _lead_angle(mx, my, tx, ty, tr, speed, omega):
    """Interception angle aiming at the target's *future* position.

    Static target → direct atan2.  Orbiting target → fixed-point solve of the
    arrival time t (t ← |T(t) − source| / speed) then aim at T(t).  All inputs are
    [N] tensors for the N launch candidates; `omega` is [N].
    """
    orb, ang0, moving = _orbit_params(tx, ty, tr, omega)

    # Fixed-point iterated lead (no-op for static targets: moving=False keeps t at
    # the straight-line value and the predicted point at the current position).
    t = torch.sqrt((tx - mx) ** 2 + (ty - my) ** 2) / speed.clamp(min=1e-6)
    for _ in range(_INTERCEPT_ITERS):
        ang = ang0 + omega * t
        px = torch.where(moving, _CENTER + orb * torch.cos(ang), tx)
        py = torch.where(moving, _CENTER + orb * torch.sin(ang), ty)
        t = torch.sqrt((px - mx) ** 2 + (py - my) ** 2) / speed.clamp(min=1e-6)

    ang = ang0 + omega * t
    px = torch.where(moving, _CENTER + orb * torch.cos(ang), tx)
    py = torch.where(moving, _CENTER + orb * torch.sin(ang), ty)
    return torch.atan2(py - my, px - mx)


def _verify_connects(angle, mx, my, mr, speed, tgt_col,
                     P_x, P_y, P_r, P_active, P_orb, P_ang0, P_moving, omega,
                     horizon):
    """Re-fly each candidate and confirm it first reaches its intended target.

    Mirrors env/orbit_wars.py connects(): straight fixed-angle flight from the
    source edge, per-tick swept collision against every active planet (the FIRST
    planet crossed wins the fleet), with off-board and sun termination.

    All per-launch scalars are [N]; planet arrays are [N, P]. Returns [N] bool.
    """
    N, P = P_x.shape
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    lx = mx + cos_a * (mr + 0.1)
    ly = my + sin_a * (mr + 0.1)

    fx_prev = lx.clone()
    fy_prev = ly.clone()
    prev_px = P_x.clone()                            # planet positions at tick 0
    prev_py = P_y.clone()

    arangeP = torch.arange(P, device=P_x.device)[None, :].expand(N, P)
    resolved = torch.zeros(N, dtype=torch.bool, device=P_x.device)
    result = torch.zeros(N, dtype=torch.bool, device=P_x.device)

    step = torch.zeros(N, device=P_x.device)
    for k in range(1, int(horizon) + 1):
        step += 1.0
        fx = lx + cos_a * speed * step
        fy = ly + sin_a * speed * step

        # Current planet positions (orbiting ones advance; static stay put).
        a = P_ang0 + omega[:, None] * step[:, None]
        cur_px = torch.where(P_moving, _CENTER + P_orb * torch.cos(a), P_x)
        cur_py = torch.where(P_moving, _CENTER + P_orb * torch.sin(a), P_y)

        hit = _swept_pair_hit(fx_prev[:, None], fy_prev[:, None],
                              fx[:, None], fy[:, None],
                              prev_px, prev_py, cur_px, cur_py, P_r)
        hit = hit & P_active                          # only real planets block/score

        # First planet hit this tick, in array order (lowest index wins).
        hit_idx = torch.where(hit, arangeP, arangeP.new_full((), P)).min(dim=1).values
        hit_any = hit_idx < P
        is_target = hit_any & (hit_idx == tgt_col)

        off = (fx < 0.0) | (fx > _BOARD_SIZE) | (fy < 0.0) | (fy > _BOARD_SIZE)
        sun = _point_to_segment_dist(_CENTER, _CENTER, fx_prev, fy_prev, fx, fy) < _SUN_RADIUS

        active = ~resolved
        # A planet hit resolves the shot (good iff it is the target). Otherwise an
        # off-board / sun crossing this tick resolves it as a miss.
        newly_hit = active & hit_any
        result = torch.where(newly_hit, is_target, result)
        resolved = resolved | newly_hit
        active = ~resolved
        newly_miss = active & (off | sun)
        resolved = resolved | newly_miss

        prev_px, prev_py = cur_px, cur_py
        fx_prev, fy_prev = fx, fy
        if bool(resolved.all()):
            break
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode(action, p_owner, p_x, p_y, p_r, p_ships, p_active, omega,
           *, player: int = 0, tanh_scale: float = 0.2):
    """Decode a batch of policy outputs into per-planet launches.

    Parameters
    ----------
    action   : [B, P, 4] float — policy output in [-1, 1] (acting player's
               perspective; ``player`` selects which owner id may launch).
    p_owner  : [B, P] int   — planet owners (−1 neutral).
    p_x,p_y  : [B, P] float — planet positions.
    p_r      : [B, P] float — planet radii.
    p_ships  : [B, P] float — planet garrisons.
    p_active : [B, P] bool  — real (active, comet-free) planet slots.
    omega    : [B] float    — system angular velocity (rad/tick).
    player   : int          — owner id allowed to launch from `action`.
    tanh_scale : float      — wedge score saturation (matches env decode).

    Returns a dict of [B, P] tensors:
      launch    : bool — a fleet is launched from this planet this step
      angle     : float — launch angle (radians); 0 where launch is False
      num_ships : long  — ships sent; 0 where launch is False
      target    : long  — chosen target planet index; −1 where launch is False
    """
    B, P, _ = action.shape
    dev = action.device
    owned = (p_owner == player) & p_active

    prelim, tgt_idx, num_ships, _frac = _select_targets(
        action, owned, p_active, p_ships, tanh_scale)

    launch = torch.zeros(B, P, dtype=torch.bool, device=dev)
    angle = torch.zeros(B, P, dtype=action.dtype, device=dev)
    target = torch.full((B, P), -1, dtype=torch.long, device=dev)

    b_idx, s_idx = torch.where(prelim)
    if b_idx.numel() > 0:
        t_idx = tgt_idx[b_idx, s_idx]                # [N] target planet indices
        mx = p_x[b_idx, s_idx]; my = p_y[b_idx, s_idx]; mr = p_r[b_idx, s_idx]
        tx = p_x[b_idx, t_idx]; ty = p_y[b_idx, t_idx]; tr = p_r[b_idx, t_idx]
        nships = num_ships[b_idx, s_idx]
        speed = _fleet_speed(nships)
        om = omega[b_idx]

        ang = _lead_angle(mx, my, tx, ty, tr, speed, om)

        # Per-launch planet arrays for the verifier (this env's planets).
        P_x = p_x[b_idx]; P_y = p_y[b_idx]; P_r = p_r[b_idx]
        P_active = p_active[b_idx]
        P_orb, P_ang0, P_moving = _orbit_params(P_x, P_y, P_r, om[:, None])

        connects = _verify_connects(
            ang, mx, my, mr, speed, t_idx,
            P_x, P_y, P_r, P_active, P_orb, P_ang0, P_moving, om, _VERIFY_TICKS)

        ok = connects
        launch[b_idx[ok], s_idx[ok]] = True
        angle[b_idx[ok], s_idx[ok]] = ang[ok].to(angle.dtype)
        target[b_idx[ok], s_idx[ok]] = t_idx[ok]

    num_ships = torch.where(launch, num_ships, torch.zeros_like(num_ships)).long()
    return {"launch": launch, "angle": angle, "num_ships": num_ships, "target": target}


@torch.no_grad()
def decode_to_moves(action, planets, omega, *, player: int = 0, tanh_scale: float = 0.2):
    """Single-environment convenience for competition submission.

    Parameters
    ----------
    action  : [P, 4] tensor or array — policy output for one observation.
    planets : [n, 7] tensor or array — [id, owner, x, y, radius, ships, production]
              (player-perspective, comet-free), exactly as env/orbit_wars.py
              decode_action expects.
    omega   : float — system angular velocity.

    Returns
    -------
    moves : list of [planet_id (int), angle (float), num_ships (int)] — the same
            format kaggle agents emit.
    """
    action = torch.as_tensor(action, dtype=torch.float64)
    planets = torch.as_tensor(planets, dtype=torch.float64)
    n = planets.shape[0]
    P = action.shape[0]

    p_id    = torch.full((P,), -1.0, dtype=torch.float64)
    p_owner = torch.full((P,), -2.0, dtype=torch.float64)
    p_x = torch.zeros(P); p_y = torch.zeros(P); p_r = torch.zeros(P)
    p_ships = torch.zeros(P); p_active = torch.zeros(P, dtype=torch.bool)
    p_id[:n]    = planets[:, 0]
    p_owner[:n] = planets[:, 1]
    p_x[:n] = planets[:, 2].double(); p_y[:n] = planets[:, 3].double()
    p_r[:n] = planets[:, 4].double(); p_ships[:n] = planets[:, 5].double()
    p_active[:n] = True

    out = decode(
        action[None].double(),
        p_owner[None].long(), p_x[None].double(), p_y[None].double(),
        p_r[None].double(), p_ships[None].double(), p_active[None],
        torch.tensor([float(omega)], dtype=torch.float64),
        player=player, tanh_scale=tanh_scale,
    )
    launch = out["launch"][0]
    moves = []
    for i in torch.where(launch)[0].tolist():
        moves.append([int(p_id[i].item()),
                      float(out["angle"][0, i].item()),
                      int(out["num_ships"][0, i].item())])
    return moves
