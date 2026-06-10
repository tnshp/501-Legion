# 501-Legion


## Benchmarking the training loop

`benchmark_train_loop.py` profiles the per-step cost of the SAC inner loop from a
real train config and locates the bottleneck (CPU env / host→device transfer /
GPU compute). It reports steps/sec, **transitions/sec**, a per-phase breakdown
tagged CPU vs GPU, GPU utilization (sampled from `nvidia-smi`), and a
transfer-vs-compute split of `update()`.

```bash
python benchmark_train_loop.py --config train.json                       # single env
python benchmark_train_loop.py --config train.json --num-envs 8 --vec-mode async
python benchmark_train_loop.py --config train.json --num-envs 4 --vec-mode sync
```

**Parallel (vectorized) environments.** With `--num-envs N > 1` the loop collects
`N` environments per step via a gymnasium vector env, mirroring `SACTrainer`'s own
VecEnv path (`select_action_batch` + `add_batch`). The update-to-data ratio is
held identical to the single-env run (1 update per `update_freq` transitions), so
the printed **speedup** is an apples-to-apples transitions/sec comparison.

- `--vec-mode async` (default): one **worker process per env** (fork start
  method) → the pure-Python game sim runs in true parallel across CPU cores. Use
  this on a fast GPU where the env is the bottleneck.
- `--vec-mode sync`: envs stepped sequentially in-process; the only win is
  batching the GPU calls (one forward for `N` envs).

On a small, update-compute-bound GPU (e.g. RTX 3050 Ti) the speedup is modest
(~1.2×) because `update()` dominates and is UTD-matched; on a fast GPU (e.g. A30)
`update()` shrinks, the parallel-env collection and batched action-selection
dominate the savings, and async vectorization gives a much larger speedup. The
`OrbitWarsEnv.step` return (`won` instead of a gym `info` dict) is bridged by a
thin `OrbitGymAdapter` so the env plugs into gymnasium's `Sync/AsyncVectorEnv`.


## JSON train args

mixed_random_ratio_decay = None -> spawn all training episodes


## Action decoder launch mask

`decode_action` now gates fleet launches by **capture feasibility** instead of a
fixed minimum size. A move is emitted only if the fleet is **larger than the
target planet's current garrison**:

```
launch only if  num_ships > target_planet_ships
```

So against a 0-ship neutral any non-empty fleet qualifies, while a defended
planet requires a fleet that exceeds its defenders — the agent stops launching
fleets too small to take their target. (Caveat: it uses the target's ship count
at *launch* time and doesn't model production/reinforcement during the fleet's
flight.) The old `min_fleet_ships` parameter is now **deprecated/unused** but
kept in the signatures and config so existing callers don't break.


## Metadata token (global summary)

The transformer is fed an extra **metadata token** prepended to the planet/fleet
sequence. It encodes a per-player global summary so the model doesn't have to
reconstruct the overall picture by attending across individual tokens.

For each of the 4 player slots (player 0 = our agent after the perspective swap,
1–3 = opponents) it carries:

- **planet count**
- **total production** (`log1p`-compressed)
- **total ships** — planets **and** in-flight fleets (`log1p`-compressed)

= a 12-dim vector (`_aggregate_meta_features` in [model/SAC.py](model/SAC.py)),
computed inside `forward` from the encoded state, projected by a learned
`nn.Linear(12, d_model)` (`meta_proj`), and prepended as token 0. Because the
encoder is full self-attention, every planet/fleet token (and the `cls` value
token) can read it directly.

Implementation notes:
- Derived in-network from the state tensor, so the **state interface (140×13) and
  the replay buffer are unchanged** — nothing to migrate.
- `Q`/`V` read the `cls` token at position 0 (now followed by the metadata token);
  `P` reads per-planet outputs, which shift to `out[:, 1:1+max_planets]`.
- Neutral planets and padding rows have an all-zero owner one-hot, so they
  contribute to no player's totals.


## Comets removed from the pipeline

Comets (transient planets the kaggle engine spawns at fixed steps) are stripped
from every observation at the parsing boundary in `env.orbit_wars._obs_to_arrays`.
Their planet rows are dropped before anything downstream sees them, so they no
longer:

- enter the **state** (this removed the per-planet "comet" feature, taking the
  state width from **14 → 13**),
- appear as **targets** or as obstacles in the launch-angle search / flight-path
  raycasts (they were a source of mis-aimed fleets),
- contribute to **reward** schemes.

The engine still simulates comets internally; we simply never expose them to the
agent. The opponent `RuleBasedAgent` already ignored comets as targets, so it is
unchanged.

> **Breaking change — retrain from scratch.** The state width changed (14 → 13),
> so the networks' input projection shape changed. **Old checkpoints and saved
> replay buffers (`replay_buffer.npz`) are incompatible** and cannot be resumed.
> Set `execution.resume = null` and delete/relocate any old `replay_buffer.npz`
> in the checkpoint dir before training. (`STATE_DIM` lives on `OrbitWarsEnv`;
> the networks default to `state_dim=13`.)


## Reward schemes

Rewards are **composable**. The `"reward"` list in the config holds one or more
*components*; their per-step values are **summed** to form the reward the agent
sees. Each component scores exactly one thing and lives in
[env/orbit_wars.py](env/orbit_wars.py). The `"scheme"` key picks the class; every
other key is forwarded straight to its constructor, so a config only lists the
parameters that scheme actually declares.

Every component is called as
`component(obs, new_obs, player_id, done, n_players, step, max_steps)` and returns
a float. `obs`/`new_obs` are the pre/post-step observations, `done` is True on the
terminal step, and `step`/`max_steps` are the current tick and the episode limit
(used only by `TimeDecayWinBonus`).

### Per-step shaping components

| Component | Formula (per step) | What it rewards |
|---|---|---|
| `RelativeShipAdvantage` | `ship_scale × [ Δmy_ships − Σ Δopp_ships ]` | Gaining ships faster than opponents. Ships in flight are counted, so launching a fleet is neutral until it fights. |
| `RelativeProductionAdvantage` | `planet_scale × [ Δmy_production − Σ Δopp_production ]` | Growing your owned **production** faster than opponents (sums each planet's output, so high-output planets count for more). |
| `ShipGrowth` | `ship_scale × Δmy_ships` (×`loss_scale` when Δ<0) | Growing your own fleet, opponents ignored. `loss_scale>1` makes losing ships hurt more than gaining (loss-averse). |
| `ProductionPlanetDelta` | `planet_scale × (Σ prod gained − loss_scale × Σ prod lost) − ship_scale × Σ_captured log1p(defender_ships)` | Capturing planets by production, with two extras: `loss_scale>1` penalises lost production more than equal gains; the `ship_scale` term subtracts a per-capture cost based on the target's pre-capture garrison, so **cheaper (lightly defended) planets** are preferred. Both default off (`loss_scale=1`, `ship_scale=0`). |
| `ProximityCaptureBonus` | `Σ_captured scale × Σ_{owned} exp(−d / ref_dist)` | Capturing planets **well-connected** to your territory. For each capture it **sums** closeness to *every* owned planet, so a planet taken **between two** (or among several) owned planets stacks the bonus and scores high, while an isolated capture scores ≈0. Capture-event only (nothing on other steps); first capture with no territory = 0. Pair with `ProductionPlanetDelta` for value-weighting. |
| `AbsoluteHoldings` | `ship_scale × my_ships_now + planet_scale × my_prod_now` | *Holding* territory — scored every step, so it pays to keep high-production planets. Does **not** telescope: keep `ship_scale` small. |
| `FleetLaunchPenalty` | `−ship_scale × n_new_fleets` | (Penalty) discourages fleet spam — flat cost per fleet launched, regardless of size/destination. |
| `LaunchDistancePenalty` | `−scale × ticks_to_target / time_norm` per launched fleet (miss = `max_ticks`) | (Penalty) charges each launched fleet by how long it must **fly** to its target — replays the fleet's path against the engine's swept-collision model. Adjacent target ≈ free, cross-map send costs a lot, a fleet that hits nothing pays the full horizon. The *launch-time* counterpart to `ProximityCaptureBonus` (denser, fires immediately). Most expensive component — forward-simulates each launch up to `max_ticks`. |
| `StepPenalty` | `−weight` (every step) | (Penalty) a flat per-step "living cost" that adds the value of *time*: any capture is worth more the sooner it lands, so the agent prefers nearby planets and quick games. Pair with a win bonus (e.g. `TimeDecayWinBonus`) so it rewards winning *fast*, not just ending fast. |

### Terminal win-bonus components

Both award a bonus on the final step only, with sign from total ships held at game
end (`+` win, `−` loss, `0` tie).

| Component | Formula | Notes |
|---|---|---|
| `TerminalWinBonus` | `±win_bonus` | Constant magnitude regardless of when the game ends. |
| `TimeDecayWinBonus` | `±win_bonus × (max_steps − step) / (max_steps − 1)` | **Decays with game length:** full `±win_bonus` for a win/loss at step 1, linearly down to `0` at `step = max_steps`. Rewards *winning fast* and softens *losing slowly* (a late loss is barely penalised, an early loss in full). |

### Example config

```json
"reward": [
  { "scheme": "AbsoluteHoldings",   "ship_scale": 0.5, "planet_scale": 1.0 },
  { "scheme": "FleetLaunchPenalty", "ship_scale": 1.0 },
  { "scheme": "TimeDecayWinBonus",  "win_bonus": 1000 }
]
```

### Legacy numbered schemes (backward compatible)

The old monolithic `RewardScheme1–4` are retained as thin compositions of the
components above and reproduce their old behaviour exactly, so existing configs
keep working. Prefer composing the named components in new configs.

| Legacy | Equivalent composition |
|---|---|
| `RewardScheme1(ship_scale, planet_scale, win_bonus)` | `RelativeShipAdvantage(ship_scale)` + `RelativeProductionAdvantage(planet_scale)` + `TerminalWinBonus(win_bonus)` |
| `RewardScheme2(ship_scale, planet_scale, win_bonus)` | `ShipGrowth(ship_scale)` + `ProductionPlanetDelta(planet_scale)` + `TerminalWinBonus(win_bonus)` |
| `RewardScheme3(ship_scale)` | `FleetLaunchPenalty(ship_scale)` |
| `RewardScheme4(ship_scale, planet_scale, win_bonus)` | `AbsoluteHoldings(ship_scale, planet_scale)` + `TerminalWinBonus(win_bonus)` |

To adopt the time-decaying bonus, swap `TerminalWinBonus` → `TimeDecayWinBonus` in
the equivalent composition.


## TD -lambda
Cache-based TD(λ) for SAC
I implemented the Daley & Amato (2019) method — λ-returns stored in a periodically-refreshed cache that replaces the target network — as an opt-in path in sac_train.py. The existing 1-step code is untouched and remains the default.

Why this addresses your problem
In Orbit Wars the reward for launching a fleet only lands ~10–20 steps later (flight time). A TD(0) target has to relay that signal back one bootstrap at a time, which is exactly the slow credit-assignment you're seeing. λ-returns propagate the delayed reward across many steps in a single update.

How it works (faithful to the paper, fixed λ)
Block promotion (_build_lambda_cache): every refresh_freq env-steps, S/B contiguous blocks of B transitions are pulled from the replay buffer into a cache of size S. The buffer is filled in temporal order, so a contiguous block is a trajectory; stored done flags cut returns at episode boundaries. I reject blocks that straddle the circular buffer's write seam (the only physical discontinuity).
Backward recursion (Eq. 8 / footnote 4), vectorized across blocks, reusing one bootstrap value per transition — O(B) Q-evals per block, not O(B²).
No target network — the cached returns are the stable targets; I skip target-net creation, Polyak updates, and checkpoint them conditionally.
SAC adaptation
The only change to the recursion is the bootstrap. Instead of max_a' Q, I use the soft value:


V(s') = min(Q1,Q2)(s',a') − α·log π(a'|s'),   a' ~ π(·|s')
Rλ_t  = r_t + γ(1−d_t)·[λ·Rλ_{t+1} + (1−λ)·V(s_{t+1})]
So at λ=0 it reduces exactly to your current 1-step SAC target. The actor loss and auto-alpha are unchanged.

Verified
Compiles; full lambda path (build → soft-value → recursion → Q/π update → checkpoint round-trip without target nets) runs on the transformer nets.
Numerically exact: λ=0 reproduces the 1-step target r+γV; λ=1 reproduces the discounted Monte-Carlo return; λ=0.9 interpolates — all matched to 1e-4.
Config
Active config at /mnt/d/ML/Kaggle/StarWars/train.json now has TD(λ) enabled:


"td_lambda": { "enabled": true, "lambda": 0.9, "cache_size": 8000, "block_size": 50, "refresh_freq": 1000 }
The in-repo 501-Legion/train.json got the same block with enabled: false to keep the committed default's behavior.

Two knobs worth a thought
lambda: with γ=0.99, λ=0.9 gives γλ≈0.89 (~9-step half-life) — a reasonable match to fleet flight time. Push toward 0.95 if credit needs to reach further; lower it if Q-targets get noisy.
refresh_freq vs update_freq=4: 1000 env-steps = 250 gradient steps per cache rebuild. If targets feel stale (Q drifting far from Cache/target_mean in TensorBoard), lower it. I added Cache/refreshes and Cache/target_mean scalars to watch this.
One faithful simplification to flag: like the paper this is Peng's Q(λ) (the bootstrap ignores which action was actually taken), and the λ-return uses environment rewards with a soft-value bootstrap — it does not add per-step entropy bonuses to intermediate rewards. That keeps λ=0 identical to your current target; if you later want the fully-soft multi-step return we can augment rewards with α·H per step.