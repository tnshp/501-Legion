"""
Benchmark the per-step cost of the training inner loop, broken down by phase.
Mirrors train_orbit_wars.train()'s rollout: select_action, env.step, buffer.add,
per-step TensorBoard logging, and the periodic trainer.update().

    python benchmark_train_loop.py [n_steps]
"""
import sys, time, tempfile
from collections import deque
import numpy as np
import torch

from model.SAC import P_network, Q_network
from sac_train import SACTrainer
from env.orbit_wars import OrbitWarsEnv, ShipGrowth, ProductionPlanetDelta, ProximityCaptureBonus
from torch.utils.tensorboard import SummaryWriter

N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 600
WARMUP  = 300
device  = "cuda" if torch.cuda.is_available() else "cpu"

reward_scheme = [ShipGrowth(0.05, loss_scale=2.0),
                 ProductionPlanetDelta(3.0, loss_scale=2.0, ship_scale=0.5),
                 ProximityCaptureBonus(1.0)]
env = OrbitWarsEnv(opponent="rule_based", n_players=2, max_steps=500,
                   reward_scheme=reward_scheme, tanh_scale=0.6, min_fleet_ships=8)
env.MAX_FLEETS, env.MAX_PLANETS = 100, 40

net_kw = dict(state_dim=OrbitWarsEnv.STATE_DIM, action_dim=4, max_planets=40,
              max_fleets=100, d_model=128, num_layers=4)
policy, q1, q2 = P_network(**net_kw), Q_network(**net_kw), Q_network(**net_kw)
logdir = tempfile.mkdtemp()
trainer = SACTrainer(env=env, policy_net=policy, q1_net=q1, q2_net=q2, device=device,
                     learning_rate=2e-4, gamma=0.99, batch_size=64, auto_alpha=True,
                     target_entropy=-20, replay_buffer_size=40000,
                     use_lambda_returns=True, lambda_return=0.9, cache_size=8000,
                     block_size=50, refresh_freq=1000, log_dir=logdir)
writer = trainer.writer

def run(n_steps, label):
    fleets_window = deque(maxlen=50)
    reward_window = deque(maxlen=100)
    max_fleets_seen = 0
    t = {k: 0.0 for k in ("select", "envstep", "add", "log", "update")}
    state, _ = env.reset()
    n_updates = 0
    torch.cuda.synchronize() if device == "cuda" else None
    wall0 = time.perf_counter()
    step = 0
    while step < n_steps:
        t0 = time.perf_counter()
        if trainer.train_step < WARMUP:
            action = env.action_space.sample()
        else:
            action = trainer.select_action(state)
        if device == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()

        next_state, reward, terminated, truncated, won = env.step(action)
        done = terminated or truncated
        t2 = time.perf_counter()

        trainer.replay_buffer.add(state, action, reward, next_state, float(done))
        trainer.train_step += 1
        state = next_state
        t3 = time.perf_counter()

        reward_window.append(float(reward))
        fleets_window.append(env.last_fleets_sent)
        max_fleets_seen = max(max_fleets_seen, env.last_n_fleets)
        if writer:
            writer.add_scalar("Reward/step", float(reward), trainer.train_step)
            writer.add_scalar("Reward/step_ma100", float(np.mean(reward_window)), trainer.train_step)
            writer.add_scalar("Policy/fleets_sent_ma50", float(np.mean(fleets_window)), trainer.train_step)
            writer.add_scalar("Env/fleets_present", env.last_n_fleets, trainer.train_step)
            writer.add_scalar("Env/fleets_present_max", max_fleets_seen, trainer.train_step)
        t4 = time.perf_counter()

        if (trainer.train_step >= WARMUP and trainer.train_step % 4 == 0
                and len(trainer.replay_buffer) >= 64):
            trainer.update()
            if device == "cuda": torch.cuda.synchronize()
            n_updates += 1
        t5 = time.perf_counter()

        t["select"]  += t1 - t0
        t["envstep"] += t2 - t1
        t["add"]     += t3 - t2
        t["log"]     += t4 - t3
        t["update"]  += t5 - t4
        if done:
            state, _ = env.reset()
        step += 1
    wall = time.perf_counter() - wall0

    print(f"\n=== {label}: {n_steps} steps, {n_updates} updates, device={device} ===")
    print(f"  total wall         : {wall*1000:8.1f} ms   ({wall/n_steps*1000:6.3f} ms/step)")
    for k in ("select", "envstep", "add", "log", "update"):
        print(f"  {k:18s}: {t[k]*1000:8.1f} ms   ({t[k]/n_steps*1000:6.3f} ms/step)")
    return wall / n_steps * 1000

# warm up the buffer + a few updates first (excluded from timing)
run(WARMUP + 80, "warmup (ignored)")
ms = run(N_STEPS, "BENCHMARK")
print(f"\nTIME PER STEP: {ms:.3f} ms")
