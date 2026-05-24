"""
Evaluate a saved SAC checkpoint against the rule-based agent (misc/sample.py).

Usage
─────
python eval_vs_rulebased.py sac_kaggle.pt
python eval_vs_rulebased.py sac_kaggle.pt --games 5 --sac-player 1 --output replay.html --open

Arguments
─────────
checkpoint   : path to the .pt file saved by SACKaggle.save()
--output     : HTML render output path (default: eval_render.html)
--sac-player : which slot the SAC agent occupies, 0 or 1 (default: 0)
--games      : how many games to play; the last game is rendered (default: 1)
--device     : "cuda" or "cpu" (auto-detected if omitted)
--open       : open the HTML in the default browser after saving (Windows)
"""

import argparse
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from kaggle_environments import make

from model.SAC import P_network, Encoder
from sac_train_kaggle import (
    _obs_to_arrays,
    _swap_perspective,
    encode_obs_as_player,
    decode_action,
    MAX_PLANETS,
    MAX_FLEETS,
    STATE_DIM,
    ACTION_DIM,
)
from misc.sample import agent as rule_based_agent


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_policy(checkpoint_path: str, device: str) -> P_network:
    net_kw = dict(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        max_planets=MAX_PLANETS, max_fleets=MAX_FLEETS,
    )
    policy = P_network(**net_kw).to(device)
    ck = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return policy


# ─────────────────────────────────────────────────────────────────────────────
# SAC inference wrapper
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def sac_act(policy: P_network, encoder: Encoder, obs, initial_planets: np.ndarray,
            sac_player: int, device: str) -> list:
    """Encode obs from SAC's perspective, run policy, decode to kaggle moves."""
    state_np = encode_obs_as_player(encoder, obs, initial_planets, player_id=sac_player)
    s = torch.FloatTensor(state_np).unsqueeze(0).to(device)
    action, _ = policy.sample(s)
    action_np = action.squeeze(0).cpu().numpy()  # [max_planets, action_dim]

    planets_now = np.array(obs.planets, dtype=np.float32)
    _, _, _, omega, _ = _obs_to_arrays(obs)
    swapped, _ = _swap_perspective(planets_now, np.empty((0, 7), np.float32), player_id=sac_player)
    return decode_action(action_np, swapped, omega)


# ─────────────────────────────────────────────────────────────────────────────
# Single-game runner
# ─────────────────────────────────────────────────────────────────────────────
def play_game(
    policy: P_network,
    encoder: Encoder,
    device: str,
    sac_player: int = 0,
) -> tuple:
    """
    Run one full game step-by-step.

    Returns
    ───────
    env     : the finished kaggle env (used for rendering)
    result  : "WIN" | "LOSS" | "DRAW"  (from SAC's perspective)
    steps   : number of steps taken
    """
    rule_player = 1 - sac_player

    env = make("orbit_wars", debug=False)
    env.reset()

    # Capture initial planet positions for orbit detection in the encoder
    obs_p0 = env.steps[0][0].observation
    obs_p1 = env.steps[0][1].observation
    planets_np, _, _, _, _ = _obs_to_arrays(obs_p0)
    initial_planets = planets_np.copy()

    # Use the observation corresponding to each player's slot
    obs_sac  = obs_p0 if sac_player == 0 else obs_p1
    obs_rule = obs_p1 if sac_player == 0 else obs_p0

    steps = 0
    while True:
        moves_sac  = sac_act(policy, encoder, obs_sac, initial_planets, sac_player, device)
        moves_rule = rule_based_agent(obs_rule)

        if sac_player == 0:
            moves_p0, moves_p1 = moves_sac, moves_rule
        else:
            moves_p0, moves_p1 = moves_rule, moves_sac

        results  = env.step(actions=[moves_p0, moves_p1])
        obs_p0   = results[0].observation
        obs_p1   = results[1].observation
        obs_sac  = obs_p0 if sac_player == 0 else obs_p1
        obs_rule = obs_p1 if sac_player == 0 else obs_p0
        done     = results[0].status != "ACTIVE"
        steps   += 1

        if done:
            break

    # Determine outcome by counting total ships + planets owned at game end
    def power(pid):
        return sum(float(p[5]) + 1 for p in obs_p0.planets if int(p[1]) == pid)

    sac_power  = power(sac_player)
    rule_power = power(rule_player)

    if sac_power > rule_power:
        result = "WIN"
    elif sac_power < rule_power:
        result = "LOSS"
    else:
        result = "DRAW"

    return env, result, steps


# ─────────────────────────────────────────────────────────────────────────────
# HTML render
# ─────────────────────────────────────────────────────────────────────────────
def save_html(env, output_path: str, width: int = 800, height: int = 600):
    html = env.render(mode="html", width=width, height=height)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Render saved → {os.path.abspath(output_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate SAC vs rule-based agent")
    parser.add_argument("checkpoint",              help="Path to .pt checkpoint")
    parser.add_argument("--output",   default="eval_render.html", help="HTML output file")
    parser.add_argument("--sac-player", type=int, default=0, choices=[0, 1],
                        help="Player slot for the SAC agent (default: 0)")
    parser.add_argument("--games",    type=int, default=1, help="Games to play (default: 1)")
    parser.add_argument("--device",   default=None,        help="cuda | cpu (auto if omitted)")
    parser.add_argument("--open",     action="store_true", help="Open HTML in browser after saving")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sac_player = args.sac_player
    rule_player = 1 - sac_player

    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"SAC plays  : player {sac_player}  |  Rule-based plays: player {rule_player}")
    print()

    policy  = load_policy(args.checkpoint, device)
    encoder = Encoder(max_planets=MAX_PLANETS, max_fleets=MAX_FLEETS)

    wins = losses = draws = 0
    last_env = None

    for g in range(args.games):
        env, result, steps = play_game(policy, encoder, device, sac_player=sac_player)
        last_env = env

        wins   += result == "WIN"
        losses += result == "LOSS"
        draws  += result == "DRAW"

        tag = {"WIN": "✓", "LOSS": "✗", "DRAW": "="}.get(result, "?")
        print(f"  Game {g+1:>3}/{args.games}  {tag} {result:<5}  steps={steps}")

    print()
    print(f"Final  W {wins}  /  L {losses}  /  D {draws}  "
          f"(win rate {wins / args.games:.1%})")

    save_html(last_env, args.output)

    if args.open:
        filepath = os.path.abspath(args.output)
        subprocess.run(["cmd.exe", "/c", "start", filepath])


if __name__ == "__main__":
    main()
