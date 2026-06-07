"""
Migrate a pre-comet-removal / pre-metadata-token checkpoint to the current arch.

Two architecture changes since older checkpoints were saved:

  1. Comets removed  → per-planet state width 14 → 13, which dropped ONE input
     column (the "comet" feature) from every network's P/F input projection, so
     P.weight / F.weight went from [d_model, 58] → [d_model, 57].
  2. Metadata token added → a new `meta_proj = Linear(12, d_model)` layer that
     older checkpoints don't have.

This script removes the comet column from P.weight/F.weight and inserts a fresh
meta_proj, producing a checkpoint the current code can load. Every other learned
weight is preserved (the comet input weights and the brand-new meta_proj are the
only parts that can't carry over).

NOTE: the old replay_buffer.npz stores 14-wide states and is NOT compatible —
do not load it. Delete it (or point ckpt_dir somewhere fresh) and let the buffer
refill; resume only loads the buffer if the file exists.

Usage:
    python migrate_checkpoint.py OLD.pt NEW.pt
"""

import sys
import torch
import torch.nn as nn

# token_feats column index of the dropped "comet" feature. The OLD forward built
#   token_feats = cat([state[:,:,:5], state[:,:,6:10], ship_block, angle_block])
# so positions 5,6,7,8 came from state cols 6,7,8,9 = production, moving,
# angular_velocity, COMET. The comet is therefore token_feats column 8.
COMET_COL = 8
OLD_WIDTH = 58   # P/F input width with the comet feature present
NEW_WIDTH = 57   # P/F input width after comet removal
META_DIM  = 12   # meta_proj input: 4 players × [planet_count, production, ships]

NET_KEYS = ("policy_net", "q1_net", "q2_net", "q1_target", "q2_target")


def migrate_net(sd: dict) -> bool:
    """Migrate one network's state_dict in place. Returns True if anything changed."""
    changed = False
    for name in ("P.weight", "F.weight"):
        w = sd.get(name)
        if w is None:
            continue
        if w.shape[1] == OLD_WIDTH:
            sd[name] = torch.cat([w[:, :COMET_COL], w[:, COMET_COL + 1:]], dim=1)
            changed = True
        elif w.shape[1] != NEW_WIDTH:
            raise SystemExit(
                f"Unexpected {name} shape {tuple(w.shape)} — expected "
                f"{OLD_WIDTH} (old) or {NEW_WIDTH} (already migrated)."
            )
    if "meta_proj.weight" not in sd:
        d_model = sd["P.weight"].shape[0]
        lin = nn.Linear(META_DIM, d_model)          # fresh default init
        sd["meta_proj.weight"] = lin.weight.detach().clone()
        sd["meta_proj.bias"]   = lin.bias.detach().clone()
        changed = True
    return changed


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python migrate_checkpoint.py OLD.pt NEW.pt")
    old_path, new_path = sys.argv[1], sys.argv[2]

    ckpt = torch.load(old_path, map_location="cpu")
    any_changed = False
    for key in NET_KEYS:
        if key in ckpt and isinstance(ckpt[key], dict):
            if migrate_net(ckpt[key]):
                print(f"migrated {key}: dropped comet column + added fresh meta_proj")
                any_changed = True

    if not any_changed:
        print("Nothing to migrate — checkpoint already matches the current arch.")

    torch.save(ckpt, new_path)
    print(f"\nSaved migrated checkpoint → {new_path}")
    print("Resume from it, but DELETE / don't load the old replay_buffer.npz "
          "(14-wide states are incompatible).")


if __name__ == "__main__":
    main()
