"""
SAC training loop for Orbit Wars (kaggle environment) with self-play.

State encoding : model/SAC.py Encoder  →  [max_planets+max_fleets, 14]
Action decoding: policy output [max_planets, 8] → [[planet_id, angle, ships], ...]
                 using safe_angle / solve_intercept from env/training_loop.py

Self-play design
────────────────
Both players share the same policy architecture. Before encoding any
observation, owner IDs are swapped (0↔1) so the policy always "sees itself
as player 0". This means both players' transitions are in the same format and
can be stored in the same replay buffer, doubling collected experience.

The opponent uses a *lagged frozen snapshot* of the policy (updated every
`opponent_update_interval` episodes) rather than the live weights. This breaks
the non-stationary feedback loop that causes cycling in pure self-play and
gives more stable Q-value targets.
"""

import copy
import os
import sys
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_environments import make
from model.SAC import P_network, Q_network, V_network, Encoder
from env.training_loop import safe_angle, solve_intercept

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MAX_PLANETS = 44
MAX_FLEETS  = 1000
STATE_DIM   = 14
ACTION_DIM  = 8
SUN_X, SUN_Y = 50.0, 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Observation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _obs_to_arrays(obs):
    """Return (planets_np [n,7], fleets_np [m,7], comet_ids [k], omega, step)."""
    planets_np = np.array(obs.planets, dtype=np.float32)

    raw_fleets = obs.fleets if obs.fleets else []
    fleets_np  = np.array(raw_fleets, dtype=np.float32) if raw_fleets else np.empty((0, 7), dtype=np.float32)

    raw_comets = getattr(obs, "comet_planet_ids", None) or []
    comet_ids  = np.array(raw_comets, dtype=np.float32)

    omega = float(getattr(obs, "angular_velocity", 0.03))
    step  = int(getattr(obs, "step", 0))
    return planets_np, fleets_np, comet_ids, omega, step


def _swap_perspective(planets_np: np.ndarray, fleets_np: np.ndarray, player_id: int):
    """
    Relabel owner IDs so `player_id` appears as player 0 to the network.
    Planet/fleet IDs and positions are unchanged, so decoded moves are still
    valid for the kaggle environment without any further translation.
    """
    if player_id == 0:
        return planets_np, fleets_np

    p = planets_np.copy()
    mask_zero = p[:, 1] == 0
    mask_pid  = p[:, 1] == player_id
    p[mask_zero, 1] = player_id
    p[mask_pid,  1] = 0

    if fleets_np.shape[0] > 0:
        f = fleets_np.copy()
        mask_fz = f[:, 1] == 0
        mask_fp = f[:, 1] == player_id
        f[mask_fz, 1] = player_id
        f[mask_fp, 1] = 0
    else:
        f = fleets_np

    return p, f


def encode_obs_as_player(
    encoder: Encoder, obs, initial_planets: np.ndarray, player_id: int = 0
) -> np.ndarray:
    """
    Encode observation from `player_id`'s perspective.
    Owner IDs are swapped so the network always sees itself as player 0.
    `initial_planets` is used only for orbit detection (position/radius),
    so it does not need to be swapped.
    """
    planets_np, fleets_np, comet_ids, omega, step = _obs_to_arrays(obs)
    planets_np, fleets_np = _swap_perspective(planets_np, fleets_np, player_id)
    state, _ = encoder.encode(
        planets_np, fleets_np, initial_planets,
        omega, comet_ids, step, apply_padding=True,
    )
    return state.astype(np.float32)


def compute_reward_for_player(obs_prev, obs_next, player_id: int) -> float:
    """Shaped reward: change in (ships + production*10) for `player_id`."""
    def score(o):
        return sum(
            float(p[5]) + float(p[6]) * 10.0
            for p in o.planets if int(p[1]) == player_id
        )
    return score(obs_next) - score(obs_prev)


# ─────────────────────────────────────────────────────────────────────────────
# Action decoder: [max_planets, ACTION_DIM] → [[planet_id, angle, ships], ...]
#
# action[i, 0]  – send logit  (> 0 triggers launch from planet i)
# action[i, 1]  – ship-fraction logit  (sigmoid → clamped to [0.1, 0.9])
# action[i, 2:] – 6-dim target key; dot-product attention picks target planet
# ─────────────────────────────────────────────────────────────────────────────
def decode_action(
    action_np: np.ndarray,
    planets_np: np.ndarray,
    omega: float,
    player_id: int = 0,
    min_ships: int = 5,
) -> list:
    """
    action_np  : [MAX_PLANETS, ACTION_DIM]  (raw policy output, numpy)
    planets_np : [n_planets, 7]              (current observation, already
                  perspective-swapped so `player_id` appears as owner 0)
    player_id  : which player to decode moves for (0 or 1)
    Returns    : [[planet_id, angle, ships_count], ...]
    """
    n = len(planets_np)
    if n == 0:
        return []

    act  = action_np[:n]          # [n, ACTION_DIM]
    keys = act[:, 2:]             # [n, 6]  target selection keys

    # Dot-product attention → target scores [n, n]; mask self-loops
    scores = keys @ keys.T        # [n, n]
    np.fill_diagonal(scores, -1e9)

    moves = []
    for i in range(n):
        pid, owner, px, py, radius, ships, _prod = planets_np[i, :7]
        # After perspective swap, our planets are always owner 0
        if int(owner) != 0:
            continue
        if float(ships) < min_ships:
            continue
        if float(act[i, 0]) <= 0.0:    # send decision
            continue

        # Ship fraction via numerically stable sigmoid
        x = float(act[i, 1])
        frac = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
        frac = max(0.1, min(0.9, frac))
        num_ships = int(float(ships) * frac)

        # Target: highest-scoring other planet
        j = int(np.argmax(scores[i]))
        _tid, _t_owner, tx, ty, t_radius, _t_ships, _t_prod = planets_np[j, :7]

        # Intercept for orbiting targets, then compute sun-safe angle
        r = math.hypot(float(tx) - SUN_X, float(ty) - SUN_Y)
        is_orbiting = (r + float(t_radius)) < 48.0
        ix, iy, _ = solve_intercept(
            float(px), float(py), float(tx), float(ty),
            is_orbiting, omega, int(num_ships),
        )
        angle = safe_angle(float(px), float(py), ix, iy)
        moves.append([int(pid), angle, num_ships])

    return moves


# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, max_size: int = 100_000):
        self.buf = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buf.append((state, action, float(reward), next_state, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.FloatTensor(np.stack(s)),                       # [B, seq, feat]
            torch.FloatTensor(np.stack(a)),                       # [B, max_planets, action_dim]
            torch.FloatTensor(np.array(r)).unsqueeze(-1),         # [B, 1]
            torch.FloatTensor(np.stack(ns)),                      # [B, seq, feat]
            torch.FloatTensor(np.array(d)).unsqueeze(-1),         # [B, 1]
        )

    def __len__(self):
        return len(self.buf)


# ─────────────────────────────────────────────────────────────────────────────
# SAC agent
# ─────────────────────────────────────────────────────────────────────────────
class SACKaggle:
    def __init__(
        self,
        device: str = "cpu",
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        alpha: float = 0.2,
        replay_size: int = 100_000,
        batch_size: int = 32,
        opponent_update_interval: int = 10,
    ):
        self.device     = device
        self.gamma      = gamma
        self.tau        = tau
        self.alpha      = alpha
        self.batch_size = batch_size
        self.opponent_update_interval = opponent_update_interval

        net_kw = dict(
            state_dim=STATE_DIM, action_dim=ACTION_DIM,
            max_planets=MAX_PLANETS, max_fleets=MAX_FLEETS,
        )
        self.policy   = P_network(**net_kw).to(device)
        self.q1       = Q_network(**net_kw).to(device)
        self.q2       = Q_network(**net_kw).to(device)
        self.v        = V_network(state_dim=STATE_DIM, max_planets=MAX_PLANETS, max_fleets=MAX_FLEETS).to(device)
        self.v_target = V_network(state_dim=STATE_DIM, max_planets=MAX_PLANETS, max_fleets=MAX_FLEETS).to(device)
        self._hard_update(self.v_target, self.v)

        # Frozen opponent snapshot — updated every `opponent_update_interval` episodes
        self.opponent_policy = copy.deepcopy(self.policy)
        self.opponent_policy.eval()
        for p in self.opponent_policy.parameters():
            p.requires_grad_(False)

        self.opt_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.opt_q1     = optim.Adam(self.q1.parameters(),     lr=lr)
        self.opt_q2     = optim.Adam(self.q2.parameters(),     lr=lr)
        self.opt_v      = optim.Adam(self.v.parameters(),      lr=lr)

        self.buf     = ReplayBuffer(max_size=replay_size)
        self.encoder = Encoder(max_planets=MAX_PLANETS, max_fleets=MAX_FLEETS)

    # ── parameter sync ────────────────────────────────────────────────────────
    def _hard_update(self, tgt: nn.Module, src: nn.Module):
        tgt.load_state_dict(src.state_dict())

    def _soft_update(self, tgt: nn.Module, src: nn.Module):
        for tp, sp in zip(tgt.parameters(), src.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def update_opponent(self):
        """Snapshot the current policy into the frozen opponent."""
        self.opponent_policy.load_state_dict(self.policy.state_dict())

    # ── inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def select_action(self, state_np: np.ndarray) -> np.ndarray:
        """state_np [seq, feat] → action_np [max_planets, action_dim]."""
        s = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)
        action, _ = self.policy.sample(s)
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def select_opponent_action(self, state_np: np.ndarray) -> np.ndarray:
        """Same as select_action but uses the frozen opponent snapshot."""
        s = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)
        action, _ = self.opponent_policy.sample(s)
        return action.squeeze(0).cpu().numpy()

    # ── gradient step ─────────────────────────────────────────────────────────
    def update(self) -> dict | None:
        if len(self.buf) < self.batch_size:
            return None

        s, a, r, ns, d = self.buf.sample(self.batch_size)
        s  = s.to(self.device)    # [B, seq, feat]
        a  = a.to(self.device)    # [B, max_planets, action_dim]
        r  = r.to(self.device)    # [B, 1]
        ns = ns.to(self.device)
        d  = d.to(self.device)    # [B, 1]

        # Q-targets via target-V
        with torch.no_grad():
            v_next   = self.v_target(ns)
            q_target = r + (1.0 - d) * self.gamma * v_next.unsqueeze(-1)

        # Update Q1
        q1_loss = nn.MSELoss()(self.q1(s, a).unsqueeze(-1), q_target)
        self.opt_q1.zero_grad(); q1_loss.backward(); self.opt_q1.step()

        # Update Q2
        q2_loss = nn.MSELoss()(self.q2(s, a).unsqueeze(-1), q_target)
        self.opt_q2.zero_grad(); q2_loss.backward(); self.opt_q2.step()

        # Update V  (soft target: min Q - α log π)
        with torch.no_grad():
            a_tilde, lp = self.policy.sample(s)
            v_soft = (
                torch.min(
                    self.q1(s, a_tilde).unsqueeze(-1),
                    self.q2(s, a_tilde).unsqueeze(-1),
                )
                - self.alpha * lp
            )
        v_loss = nn.MSELoss()(self.v(s).unsqueeze(-1), v_soft)
        self.opt_v.zero_grad(); v_loss.backward(); self.opt_v.step()

        # Update policy  (maximize min Q - α log π)
        a_tilde, lp = self.policy.sample(s)
        pi_loss = (
            self.alpha * lp
            - torch.min(
                self.q1(s, a_tilde).unsqueeze(-1),
                self.q2(s, a_tilde).unsqueeze(-1),
            )
        ).mean()
        self.opt_policy.zero_grad(); pi_loss.backward(); self.opt_policy.step()

        self._soft_update(self.v_target, self.v)

        return dict(q1=q1_loss.item(), q2=q2_loss.item(),
                    v=v_loss.item(), pi=pi_loss.item())

    # ── checkpoint I/O ────────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "v":  self.v.state_dict(), "v_target": self.v_target.state_dict(),
        }, path)
        print(f"Checkpoint saved → {path}")

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck["policy"])
        self.q1.load_state_dict(ck["q1"])
        self.q2.load_state_dict(ck["q2"])
        self.v.load_state_dict(ck["v"])
        self.v_target.load_state_dict(ck["v_target"])
        print(f"Checkpoint loaded ← {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner (self-play)
# ─────────────────────────────────────────────────────────────────────────────
def run_episode(agent: SACKaggle, warmup_steps: int, total_steps: int):
    """
    Run one full self-play episode.

    Both players use the shared policy (from their own perspective). Player 1
    uses the frozen opponent snapshot. Owner IDs are swapped before encoding
    so each player always sees itself as owner 0 — making both players'
    transitions interchangeable in the replay buffer.

    Returns:
        transitions : list[(state, action_np, reward, next_state, done)]
                      contains transitions for BOTH players (2× data per step)
        ep_reward_p0: float  (player 0's episode reward, for logging)
    """
    env = make("orbit_wars", debug=False)
    env.reset()

    obs = env.steps[0][0].observation
    planets_np, _, _, _, _ = _obs_to_arrays(obs)
    initial_planets = planets_np.copy()

    # Per-player states encoded from their respective perspectives
    state_p0 = encode_obs_as_player(agent.encoder, obs, initial_planets, player_id=0)
    state_p1 = encode_obs_as_player(agent.encoder, obs, initial_planets, player_id=1)

    transitions  = []
    ep_reward_p0 = 0.0
    in_warmup    = total_steps < warmup_steps

    while True:
        _, _, _, omega, _ = _obs_to_arrays(obs)

        # ── Player 0: current (learning) policy ──────────────────────────────
        if in_warmup:
            action_p0 = np.random.randn(MAX_PLANETS, ACTION_DIM).astype(np.float32)
        else:
            action_p0 = agent.select_action(state_p0)

        # Decode using perspective-swapped planets (player 0 already owns owner-0 planets)
        planets_now = np.array(obs.planets, dtype=np.float32)
        swapped_p0, _ = _swap_perspective(planets_now, np.empty((0, 7), np.float32), player_id=0)
        moves_p0 = decode_action(action_p0, swapped_p0, omega, player_id=0)

        # ── Player 1: frozen opponent snapshot ───────────────────────────────
        if in_warmup:
            action_p1 = np.random.randn(MAX_PLANETS, ACTION_DIM).astype(np.float32)
        else:
            action_p1 = agent.select_opponent_action(state_p1)

        # Decode from player 1's swapped perspective
        swapped_p1, _ = _swap_perspective(planets_now, np.empty((0, 7), np.float32), player_id=1)
        moves_p1 = decode_action(action_p1, swapped_p1, omega, player_id=1)

        # ── Step environment ──────────────────────────────────────────────────
        step_results = env.step(actions=[moves_p0, moves_p1])
        new_obs  = step_results[0].observation
        done     = step_results[0].status != "ACTIVE"

        # ── Rewards ───────────────────────────────────────────────────────────
        r_p0 = compute_reward_for_player(obs, new_obs, player_id=0)
        r_p1 = compute_reward_for_player(obs, new_obs, player_id=1)

        # Add any terminal signal from the env
        env_r0 = step_results[0].reward
        if env_r0 == 1:
            env_r0 = 500
        env_r1 = step_results[1].reward if len(step_results) > 1 else None
        if env_r0 is not None:
            r_p0 += float(env_r0)
        if env_r1 is not None:
            r_p1 += float(env_r1)

        # ── Next states ───────────────────────────────────────────────────────
        new_state_p0 = encode_obs_as_player(agent.encoder, new_obs, initial_planets, player_id=0)
        new_state_p1 = encode_obs_as_player(agent.encoder, new_obs, initial_planets, player_id=1)

        # Store both players' transitions — same format, double the data
        transitions.append((state_p0, action_p0, r_p0, new_state_p0, float(done)))
        transitions.append((state_p1, action_p1, r_p1, new_state_p1, float(done)))

        ep_reward_p0 += r_p0
        obs      = new_obs
        state_p0 = new_state_p0
        state_p1 = new_state_p1
        in_warmup = (total_steps + len(transitions) // 2) < warmup_steps

        if done:
            break

    return transitions, ep_reward_p0


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(
    num_episodes: int        = 200,
    warmup_steps: int        = 500,
    update_frequency: int    = 4,
    opponent_update_interval: int = 10,
    log_interval: int        = 10,
    checkpoint_interval: int = 50,
    save_path: str           = "sac_kaggle.pt",
    device: str | None       = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device} | opponent snapshot refreshed every {opponent_update_interval} episodes")

    agent       = SACKaggle(device=device, opponent_update_interval=opponent_update_interval)
    total_steps = 0
    ep_rewards  = []

    for ep in range(num_episodes):
        # Refresh the frozen opponent snapshot on schedule
        if ep % agent.opponent_update_interval == 0:
            agent.update_opponent()

        transitions, ep_reward = run_episode(agent, warmup_steps, total_steps)

        for s, a, r, ns, d in transitions:
            agent.buf.add(s, a, r, ns, d)
            total_steps += 1

            if total_steps >= warmup_steps and total_steps % update_frequency == 0:
                agent.update()

        ep_rewards.append(ep_reward)

        if (ep + 1) % log_interval == 0:
            avg = np.mean(ep_rewards[-log_interval:])
            print(
                f"Ep {ep+1:>4}/{num_episodes} | "
                f"avg_reward_p0={avg:>9.2f} | "
                f"buf={len(agent.buf):>6} | "
                f"steps={total_steps}"
            )

        if (ep + 1) % checkpoint_interval == 0:
            agent.save(save_path)

    agent.save(save_path)
    return ep_rewards


if __name__ == "__main__":
    train()
