"""On-device (JAX) reward computation for the vectorised Orbit Wars env.

Single-environment ports of the composable reward components in
``env/orbit_wars.py`` (and their NumPy mirror in
``JaxVecEnvAdapter._compute_config_reward``).  ``build_reward_fn`` composes the
configured components into one pure function ``reward(old, new) -> scalar`` that
runs inside the fused, vmapped+jitted rollout step — so the reward is computed on
the GPU alongside the engine step instead of in host NumPy on a device→host copy.

Everything here mirrors the NumPy reward EXACTLY (verified by parity tests in
test_jax_port.py) so training dynamics are unchanged.  ``LaunchDistancePenalty``
(a per-tick flight simulation) is intentionally NOT ported; configs that use it
fall back to the NumPy reward path in the adapter.

Inputs ``old`` / ``new`` are single-env ``GameState`` (no leading batch dim); the
caller vmaps over the batch.
"""
from __future__ import annotations

import jax.numpy as jnp

# Components handled here; anything else (e.g. LaunchDistancePenalty) forces the
# adapter onto its NumPy reward fallback.
SUPPORTED = frozenset({
    "RelativeShipAdvantage", "RelativeProductionAdvantage", "ShipGrowth",
    "ProductionPlanetDelta", "ProximityCaptureBonus", "AbsoluteHoldings",
    "FleetLaunchPenalty", "StepPenalty", "TerminalWinBonus", "TimeDecayWinBonus",
})


def _holdings(planets, fleets, num_players):
    """(ships[NP], prod[NP]) — planet+fleet ships and planet production per player.

    Mirrors adapter._compute_config_reward._holdings for a single env.
    """
    pids  = jnp.arange(num_players)
    valid = planets.active & ~planets.is_comet
    pm    = (planets.owner[:, None] == pids) & valid[:, None]          # [P, NP]
    ships = (planets.ships[:, None] * pm).sum(0)                       # [NP]
    prod  = (planets.production[:, None].astype(jnp.float32) * pm).sum(0)
    fm    = (fleets.owner[:, None] == pids) & fleets.active[:, None]   # [F, NP]
    ships = ships + (fleets.ships[:, None] * fm).sum(0)
    return ships.astype(jnp.float32), prod.astype(jnp.float32)


def build_reward_fn(reward_cfg, episode_steps: int, num_players: int):
    """Compose a single-env ``reward(old, new) -> float32`` from the config.

    ``reward_cfg`` is the adapter's already-expanded list of ``(name, params)``
    atomic components (``self._reward_cfg``).  The Python loop below unrolls at
    trace time, so the returned function is jit-friendly.
    """
    names   = {n for n, _ in reward_cfg}
    unknown = names - SUPPORTED
    if unknown:
        raise ValueError(f"jax_reward cannot build {sorted(unknown)}; "
                         "use the NumPy reward path for these.")

    NP        = int(num_players)
    max_s     = float(episode_steps)
    need_delta   = bool(names & {"RelativeShipAdvantage",
                                 "RelativeProductionAdvantage", "ShipGrowth"})
    need_result  = bool(names & {"TerminalWinBonus", "TimeDecayWinBonus"})
    need_capture = bool(names & {"ProductionPlanetDelta", "ProximityCaptureBonus"})
    need_launch  = "FleetLaunchPenalty" in names

    def reward(old, new):
        done = new.done.astype(jnp.float32)
        total = jnp.float32(0.0)

        ships_post, prod_post = _holdings(new.planets, new.fleets, NP)
        if need_delta:
            ships_pre, prod_pre = _holdings(old.planets, old.fleets, NP)
        if need_result:
            my       = ships_post[0]
            best_opp = ships_post[1:].max() if NP > 1 else jnp.float32(0.0)
            result   = jnp.sign(my - best_opp)
        if need_capture:
            valid_o   = old.planets.active & ~old.planets.is_comet
            valid_n   = new.planets.active & ~new.planets.is_comet
            owned_old = (old.planets.owner == 0) & valid_o      # [P]
            owned_new = (new.planets.owner == 0) & valid_n
            captured  = owned_new & ~owned_old
            lost      = owned_old & ~owned_new
        if need_launch:
            MF       = new.fleets.owner.shape[0]
            n_total  = new.next_fleet_slot - old.next_fleet_slot
            rel      = (jnp.arange(MF) - old.next_fleet_slot) % MF
            in_range = rel < n_total
            p0_launched = in_range & (new.fleets.owner == 0) & new.fleets.active

        for name, params in reward_cfg:
            if name == "RelativeShipAdvantage":
                s     = params.get("ship_scale", 0.01)
                my_d  = ships_post[0] - ships_pre[0]
                opp_d = ships_post[1:].sum() - ships_pre[1:].sum()
                total = total + s * (my_d - opp_d)

            elif name == "RelativeProductionAdvantage":
                s     = params.get("planet_scale", 1.0)
                my_d  = prod_post[0] - prod_pre[0]
                opp_d = prod_post[1:].sum() - prod_pre[1:].sum()
                total = total + s * (my_d - opp_d)

            elif name == "ShipGrowth":
                s     = params.get("ship_scale", 0.01)
                ls    = params.get("loss_scale", 1.0)
                my_d  = ships_post[0] - ships_pre[0]
                scale = jnp.where(my_d < 0, s * ls, s)
                total = total + scale * my_d

            elif name == "ProductionPlanetDelta":
                ps        = params.get("planet_scale", 1.0)
                ls        = params.get("loss_scale", 1.0)
                ss        = params.get("ship_scale", 0.0)
                cap_prod  = (new.planets.production.astype(jnp.float32) * captured).sum()
                lost_prod = (old.planets.production.astype(jnp.float32) * lost).sum()
                ship_cost = (jnp.log1p(jnp.maximum(0.0, old.planets.ships)) * captured).sum()
                total = total + ps * (cap_prod - ls * lost_prod) - ss * ship_cost

            elif name == "ProximityCaptureBonus":
                scale    = params.get("scale", 1.0)
                ref_dist = max(1e-6, params.get("ref_dist", 25.0))
                dx    = new.planets.x[:, None] - old.planets.x[None, :]
                dy    = new.planets.y[:, None] - old.planets.y[None, :]
                close = jnp.exp(-jnp.sqrt(dx * dx + dy * dy) / ref_dist)
                mask  = captured[:, None] & owned_old[None, :]
                total = total + scale * (close * mask).sum()

            elif name == "AbsoluteHoldings":
                ss = params.get("ship_scale", 0.01)
                ps = params.get("planet_scale", 1.0)
                total = total + ss * ships_post[0] + ps * prod_post[0]

            elif name == "FleetLaunchPenalty":
                s = params.get("ship_scale", 0.5)
                total = total - s * p0_launched.sum()

            elif name == "StepPenalty":
                total = total - jnp.float32(params.get("weight", 0.1))

            elif name == "TerminalWinBonus":
                wb = params.get("win_bonus", 100.0)
                total = total + wb * result * done

            elif name == "TimeDecayWinBonus":
                wb    = params.get("win_bonus", 100.0)
                decay = jnp.clip((max_s - new.step.astype(jnp.float32))
                                 / max(1.0, max_s - 1.0), 0.0, 1.0)
                total = total + wb * decay * result * done

        return total.astype(jnp.float32)

    return reward
