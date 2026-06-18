"""
Detailed sub-step profiler for the envstep phase of the benchmark.

Monkey-patches JaxVecEnvAdapter to time every sub-call inside step(),
_decode_p0_numpy(), and _post_step_jax().  Also profiles the update()
path at the same granularity.

Runs a warmup (to JIT-compile everything), then a measured pass identical
to the benchmark, printing per-sub-step averages.
"""
from __future__ import annotations

import argparse
import os
import time
import collections

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import torch

from benchmark_jax_train_loop import _build_env, _build_trainer, _run_serial
from train_orbit_wars import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   default="train.json")
    ap.add_argument("--vsteps",   type=int, default=100)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--num-players", type=int, default=2)
    args = ap.parse_args()

    config   = load_config(args.config)
    t_cfgx   = config.get("training", {})
    jax_cfg  = config.get("jax_env", {})
    device   = ("cuda" if torch.cuda.is_available()
                and not config.get("execution", {}).get("cpu_force") else "cpu")
    num_envs = args.num_envs or jax_cfg.get("num_envs", 256)
    update_freq  = t_cfgx.get("update_freq") or num_envs
    grad_steps   = t_cfgx.get("gradient_steps", 1)
    batch_size   = t_cfgx.get("batch_size", 64)
    warmup_steps = t_cfgx.get("warmup_steps", 500)

    print(f"device={device}  num_envs={num_envs}  vsteps={args.vsteps}")
    print(f"batch_size={batch_size}  update_freq={update_freq}")

    env     = _build_env(config, num_envs, args.num_players)
    trainer = _build_trainer(config, env, device)

    # ── Warmup ────────────────────────────────────────────────────────────
    warmup_vsteps = max(2, warmup_steps // num_envs + 2)
    print(f"\nWarming up ({warmup_vsteps} vsteps)...")
    _run_serial(trainer, env, num_envs, warmup_steps, update_freq,
                grad_steps, batch_size, device, warmup_vsteps)
    print("Warmup done.\n")

    # ── Instrument env.step via monkey-patching ──────────────────────────
    # Accumulate per-sub-step times across all vsteps
    step_acc = collections.defaultdict(float)
    step_counts = collections.defaultdict(int)
    auto_reset_envs = []   # how many envs reset each step

    _orig_step = env.step.__func__  # unbound

    def _t():
        return time.perf_counter()

    def instrumented_step(self, actions_np):
        t0 = _t()

        # --- _decode_p0_numpy breakdown ---
        state = self._state

        t_d2h = _t()
        p_owner = np.asarray(state.planets.owner,  dtype=np.int32)
        p_x     = np.asarray(state.planets.x,      dtype=np.float32)
        p_y     = np.asarray(state.planets.y,      dtype=np.float32)
        p_ships = np.asarray(state.planets.ships,  dtype=np.float32)
        p_act   = np.asarray(state.planets.active, dtype=bool)
        p_comet = np.asarray(state.planets.is_comet, dtype=bool)
        t_d2h_end = _t()
        step_acc["decode.d2h_state"] += t_d2h_end - t_d2h

        t_masks = _t()
        from jax_env.adapter import _NET_MP, _JAX_MP
        valid    = p_act[:, :_NET_MP] & ~p_comet[:, :_NET_MP]
        owned_p0 = (p_owner[:, :_NET_MP] == 0) & valid
        tmp = np.zeros((self.num_envs, self.num_players, _JAX_MP, 2), dtype=np.float32)
        t_masks_end = _t()
        step_acc["decode.masks"] += t_masks_end - t_masks

        t_wedge = _t()
        self._decode_for_player(actions_np, tmp, 0, owned_p0, valid, p_x, p_y, p_ships)
        p0_engine = tmp[:, 0, :_NET_MP, :]
        t_wedge_end = _t()
        step_acc["decode.wedge_decode"] += t_wedge_end - t_wedge

        step_acc["decode.TOTAL"] += t_wedge_end - t0

        # --- _post_step_jax breakdown ---
        B  = self.num_envs
        t_prep = _t()
        p0 = np.asarray(p0_engine, dtype=np.float32)
        self.last_fleets_sent = int((p0[:, :, 1] > 0).sum(axis=1).mean())
        t_prep_end = _t()
        step_acc["post.prep"] += t_prep_end - t_prep

        t_rng = _t()
        self._key, k = jax.random.split(self._key)
        keys = jax.random.split(k, B)
        t_rng_end = _t()
        step_acc["post.rng_split"] += t_rng_end - t_rng

        t_h2d = _t()
        p0_jax = jnp.asarray(p0)
        mix_r  = jnp.float32(self.mixed_random_ratio)
        t_h2d_end = _t()
        step_acc["post.h2d_actions"] += t_h2d_end - t_h2d

        t_pool = _t()
        pool_batch = self._next_pool_batch()
        t_pool_end = _t()
        step_acc["post.pool_batch_gather"] += t_pool_end - t_pool

        t_fused = _t()
        new_state, reward_j, dones_j, wons_j = self._fused(
            self._state, p0_jax, keys, mix_r, pool_batch)
        jax.block_until_ready((new_state, reward_j, dones_j, wons_j))
        t_fused_end = _t()
        step_acc["post.fused_step (opp+engine+reward+reset)"] += t_fused_end - t_fused

        t_d2h2 = _t()
        dones_np   = np.asarray(dones_j,   dtype=bool)
        rewards_np = np.asarray(reward_j,  dtype=np.float32)
        wons_np    = np.asarray(wons_j,    dtype=bool)
        self.last_n_fleets = int(np.asarray(new_state.fleets.active).sum(axis=-1).mean())
        t_d2h2_end = _t()
        step_acc["post.d2h_results"] += t_d2h2_end - t_d2h2

        n_done = int(dones_np.sum())
        auto_reset_envs.append(n_done)
        self._state = new_state

        t_obs = _t()
        obs_next = np.asarray(self._extract_obs_jax(new_state))
        t_obs_end = _t()
        step_acc["post.extract_obs_jax + d2h"] += t_obs_end - t_obs

        step_acc["post.TOTAL"] += t_obs_end - t_prep

        for k2 in step_acc:
            step_counts[k2] = step_counts.get(k2, 0)
        step_acc["STEP TOTAL"] += _t() - t0

        truncateds = np.zeros(B, dtype=bool)
        return obs_next, rewards_np, dones_np, truncateds, wons_np

    # ── Instrument trainer.update breakdown ──────────────────────────────
    upd_acc = collections.defaultdict(float)
    upd_count = [0]

    _orig_update = trainer.update

    def instrumented_update(_profile=False):
        if len(trainer.replay_buffer) < trainer.batch_size:
            return None

        t0 = _t()

        # Sample
        t_samp = _t()
        states, actions, rewards, next_states, dones = trainer.replay_buffer.sample(
            trainer.batch_size)
        states      = trainer.state_preprocessor(trainer._to(states))
        actions     = trainer.action_preprocessor(trainer._to(actions))
        rewards     = trainer._to(rewards)
        next_states = trainer.state_preprocessor(trainer._to(next_states))
        dones       = trainer._to(dones)
        if device == "cuda":
            torch.cuda.synchronize()
        upd_acc["upd.sample+H2D"] += _t() - t_samp

        if trainer.use_lambda_returns:
            # For lambda path, just delegate after sampling cost is measured
            t_lam = _t()
            result = trainer._lambda_update()
            if device == "cuda":
                torch.cuda.synchronize()
            upd_acc["upd.lambda_update (rest)"] += _t() - t_lam
            upd_count[0] += 1
            upd_acc["upd.TOTAL"] += _t() - t0
            return result

        # Q-target
        t_qt = _t()
        with torch.no_grad(), trainer._autocast():
            a_next, lp_next = trainer._sample(trainer.policy_net, next_states)
            q1_next = trainer.q1_target(next_states, a_next).unsqueeze(-1).clone()
            q2_next = trainer.q2_target(next_states, a_next).unsqueeze(-1).clone()
            q_target = (rewards + (1.0 - dones) * trainer.gamma * (
                torch.min(q1_next, q2_next) - trainer.alpha * lp_next
            )).float()
        q_target = torch.nan_to_num(q_target, nan=0.0, posinf=0.0, neginf=0.0)
        if device == "cuda":
            torch.cuda.synchronize()
        upd_acc["upd.q_target"] += _t() - t_qt

        # Q1
        t_q1 = _t()
        import torch.nn as nn
        with trainer._autocast():
            q1_pred = trainer.q1_net(states, actions).unsqueeze(-1)
            q1_loss = nn.MSELoss()(q1_pred, q_target)
        trainer.q1_optimizer.zero_grad()
        q1_loss.backward()
        if trainer.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(trainer.q1_net.parameters(), trainer.max_grad_norm)
        trainer.q1_optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        upd_acc["upd.q1_update"] += _t() - t_q1

        # Q2
        t_q2 = _t()
        with trainer._autocast():
            q2_pred = trainer.q2_net(states, actions).unsqueeze(-1)
            q2_loss = nn.MSELoss()(q2_pred, q_target)
        trainer.q2_optimizer.zero_grad()
        q2_loss.backward()
        if trainer.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(trainer.q2_net.parameters(), trainer.max_grad_norm)
        trainer.q2_optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        upd_acc["upd.q2_update"] += _t() - t_q2

        # Policy
        t_pi = _t()
        with trainer._autocast():
            import math
            a_tilde, lp = trainer._sample(trainer.policy_net, states)
            q1_pi = trainer.q1_net(states, a_tilde).unsqueeze(-1).clone()
            q2_pi = trainer.q2_net(states, a_tilde).unsqueeze(-1).clone()
            policy_loss = (trainer.alpha * lp - torch.min(q1_pi, q2_pi)).mean()
        trainer.policy_optimizer.zero_grad()
        policy_loss.backward()
        if trainer.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(trainer.policy_net.parameters(), trainer.max_grad_norm)
        trainer.policy_optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        upd_acc["upd.policy_update"] += _t() - t_pi

        # Polyak
        t_poly = _t()
        trainer._soft_update(trainer.q1_target, trainer.q1_net)
        trainer._soft_update(trainer.q2_target, trainer.q2_net)
        if device == "cuda":
            torch.cuda.synchronize()
        upd_acc["upd.polyak"] += _t() - t_poly

        trainer._update_count += 1
        upd_count[0] += 1
        upd_acc["upd.TOTAL"] += _t() - t0
        return None

    # ── Patch and run ────────────────────────────────────────────────────
    import types
    env.step = types.MethodType(instrumented_step, env)
    trainer.update = instrumented_update

    is_cuda = (device == "cuda")
    n_vsteps = args.vsteps

    # We need to replicate _run_serial's logic here since we patched step/update
    obs, _ = env.reset()
    if is_cuda:
        torch.cuda.synchronize()

    phases = {k: 0.0 for k in ("select", "envstep", "add", "update")}
    n_upds = 0
    upd_debt = 0.0
    wall0 = time.perf_counter()

    for _ in range(n_vsteps):
        t0 = time.perf_counter()
        actions = trainer.select_action_batch(obs)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        next_obs, rewards, dones, _, _ = env.step(actions)
        t2 = time.perf_counter()

        trainer.replay_buffer.add_batch(
            obs, actions, rewards, next_obs, dones.astype(np.float32))
        trainer.train_step += num_envs
        obs = next_obs
        t3 = time.perf_counter()

        if (trainer.train_step >= 0
                and len(trainer.replay_buffer) >= batch_size):
            upd_debt += num_envs / update_freq
            while upd_debt >= 1.0:
                for _ in range(grad_steps):
                    trainer.update()
                n_upds += 1
                upd_debt -= 1.0
            if is_cuda:
                torch.cuda.synchronize()
        t4 = time.perf_counter()

        phases["select"]  += t1 - t0
        phases["envstep"] += t2 - t1
        phases["add"]     += t3 - t2
        phases["update"]  += t4 - t3

    wall = time.perf_counter() - wall0
    n_tr = n_vsteps * num_envs

    # ── Report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  MEASURED PASS: {n_vsteps} vsteps x {num_envs} envs = {n_tr} transitions")
    print(f"  Wall: {wall:.2f}s   {n_tr/wall:.0f} trans/s   {n_upds} updates")
    print("=" * 78)

    def _fmt(dt, N=n_vsteps):
        avg_ms = dt / N * 1000
        return f"{dt*1000:9.1f}ms total  {avg_ms:7.2f}ms/vstep"

    print(f"\n  TOP-LEVEL PHASES (per the serial loop):")
    for k in ("select", "envstep", "add", "update"):
        pct = phases[k] / wall * 100
        print(f"    {k:10s}: {_fmt(phases[k])}  {pct:5.1f}%")

    print(f"\n  ── ENVSTEP BREAKDOWN ({phases['envstep']*1000:.1f}ms, "
          f"{phases['envstep']/wall*100:.1f}% of wall) ──")
    decode_keys = sorted(k for k in step_acc if k.startswith("decode."))
    post_keys   = sorted(k for k in step_acc if k.startswith("post."))
    for k in decode_keys:
        pct = step_acc[k] / phases["envstep"] * 100 if phases["envstep"] > 0 else 0
        print(f"    {k:45s}: {_fmt(step_acc[k])}  {pct:5.1f}%")
    print()
    for k in post_keys:
        pct = step_acc[k] / phases["envstep"] * 100 if phases["envstep"] > 0 else 0
        print(f"    {k:45s}: {_fmt(step_acc[k])}  {pct:5.1f}%")

    if auto_reset_envs:
        arr = np.array(auto_reset_envs)
        print(f"\n    auto_reset: {arr.sum()} total resets across {n_vsteps} vsteps "
              f"(mean {arr.mean():.1f}/vstep, max {arr.max()}, "
              f"steps_with_resets={int((arr>0).sum())})")

    if upd_count[0] > 0:
        print(f"\n  ── UPDATE BREAKDOWN ({phases['update']*1000:.1f}ms, "
              f"{phases['update']/wall*100:.1f}% of wall, "
              f"{upd_count[0]} calls) ──")
        for k in sorted(upd_acc):
            pct = upd_acc[k] / phases["update"] * 100 if phases["update"] > 0 else 0
            avg = upd_acc[k] / upd_count[0] * 1000
            print(f"    {k:35s}: {upd_acc[k]*1000:9.1f}ms total  "
                  f"{avg:7.2f}ms/call  {pct:5.1f}%")

    if device == "cuda":
        print(f"\n  torch mem: {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB peak")

    print("=" * 78)

    env.close()
    trainer.close()


if __name__ == "__main__":
    main()
