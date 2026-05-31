"""
Print parameter counts for the SAC model using the current train.json config.
Usage: python model_info.py [--config train.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from model.SAC import P_network, Q_network


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main(config_path: str = "train.json"):
    with open(config_path) as f:
        cfg = json.load(f)

    m = cfg.get("model", {})
    d_model    = m.get("d_model",    128)
    num_layers = m.get("num_layers",   2)
    num_heads  = m.get("num_heads",    4)
    ff_dim     = m.get("ff_dim",     512)
    dropout    = m.get("dropout",    0.1)

    state_dim   = 14   # fixed by encoder output
    action_dim  = 4    # OrbitWarsEnv.ACTION_DIM
    max_planets = 40
    max_fleets  = 100

    policy = P_network(
        d_model=d_model, state_dim=state_dim, action_dim=action_dim,
        max_planets=max_planets, max_fleets=max_fleets,
        nhead=num_heads, num_layers=num_layers,
        dim_feedforward=ff_dim, dropout=dropout,
    )
    q1 = Q_network(
        state_dim=state_dim, action_dim=action_dim,
        max_planets=max_planets, max_fleets=max_fleets,
        d_model=d_model, nhead=num_heads, num_layers=num_layers,
        dim_feedforward=ff_dim, dropout=dropout,
    )

    p_total,  p_train  = count_params(policy)
    q_total,  q_train  = count_params(q1)
    all_total = p_total + 2 * q_total
    all_train = p_train + 2 * q_train

    col = 18
    print(f"\nConfig : {config_path}")
    print(f"  d_model={d_model}  layers={num_layers}  heads={num_heads}  ff_dim={ff_dim}\n")
    print(f"{'Network':<14} {'Total':>{col}} {'Trainable':>{col}}")
    print("-" * (14 + col * 2 + 2))
    print(f"{'P_network':<14} {p_total:>{col},} {p_train:>{col},}")
    print(f"{'Q_network (×2)':<14} {2*q_total:>{col},} {2*q_train:>{col},}")
    print("-" * (14 + col * 2 + 2))
    print(f"{'Total':<14} {all_total:>{col},} {all_train:>{col},}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train.json")
    args = parser.parse_args()
    main(args.config)
