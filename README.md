# 501-Legion


## Benchmarking

Two scripts measure performance on the JAX backend. Both print the active JAX
backend (`CPU`/`GPU`) up top, so a CPU run and a GPU run are unambiguous — run the
same command on each machine (or force CPU with `JAX_PLATFORMS=cpu`) to compare.

### `benchmark_jax_env_step.py` — env step only (CPU vs GPU)

Isolates the **environment** cost from the learner to answer "does the JAX env run
faster on GPU?". Times two phases per `num_envs`:

- **engine** — `VectorizedEnv.step` only: the pure JAX/XLA game step (vmapped +
  jitted), timed with `jax.block_until_ready`. This is the part a GPU accelerates.
- **full** — `JaxVecEnvAdapter.step`: the engine step **plus** host-side obs
  extraction (a device→host copy each step on GPU), the numpy opponent, and the
  reward computation — i.e. what the training loop actually pays per vector-step.

The split is the point: a GPU can make `engine` much faster while `full` barely
moves, because the per-step obs copy and the opponent run on the CPU regardless —
so `full` only pulls ahead at large `num_envs`.

```bash
python benchmark_jax_env_step.py                          # default sweep (64,256,1024)
python benchmark_jax_env_step.py --sweep 256,1024,4096    # scaling
python benchmark_jax_env_step.py --opponent agent1        # heavier opponent in 'full'
JAX_PLATFORMS=cpu python benchmark_jax_env_step.py        # force the CPU baseline
```

Flags: `--sweep`, `--steps`, `--warmup`, `--num-players`, `--episode-steps`,
`--opponent {random,greedy,agent1}`, `--mode {engine,full,both}`.

### `benchmark_jax_train_loop.py` — full SAC inner loop

A **light, single-pass** throughput check: builds the env + trainer straight from
the train config (same `num_envs`, model, opponent, reward, TD(λ), `attn_impl`, …)
and runs **one** measured pass for `--vsteps` vector-steps using the config's
`jax_env.actor` mode, then reports **transitions/sec**. In `serial` mode it also
prints a per-phase breakdown (`select` / `envstep` / `add` / `update`) — `envstep`
is tagged with the actual **JAX backend** (`GPU+host` / `CPU+host`), so a GPU env
shows up as GPU.

```bash
python benchmark_jax_train_loop.py                       # config's actor, 100 vsteps
python benchmark_jax_train_loop.py --vsteps 200          # longer measured pass
python benchmark_jax_train_loop.py --actor serial        # force a mode (serial|thread)
python benchmark_jax_train_loop.py --num-envs 256        # override jax_env.num_envs
```

It runs **only** the configured (or `--actor`-overridden) mode — no sweep, no
stacking of serial + thread — so a check is quick. `--actor` mirrors the
`jax_env.actor` modes (see
[JAX environment config](#jax-environment-config-jax_env)); run it once per mode to
compare end-to-end transitions/sec and pick the fastest for a given machine. The
model honours `model.attn_impl`, so set `"flash"` in the config to benchmark the
FlashAttention path. Add `--no-gpu-sample` to skip the `nvidia-smi` poller.


## JSON train args

mixed_random_ratio_decay = None -> spawn all training episodes


## JAX environment config (`jax_env`)

The `jax_env` block configures the **parallel JAX environment backend** used by
`train_orbit_wars.py` (the only backend on this branch). `train_jax()` reads it via
`config["jax_env"]`. Each **vector-step** advances all `num_envs` environments one
game tick and writes `num_envs` transitions to the replay buffer.

The committed `train.json` keeps only the keys that change behaviour; every other
key in the table below is optional and falls back to its default:

```json
"jax_env": {
  "num_envs":         64,
  "actor":            "serial",
  "episode_steps":    500
}
```

### Keys

| key | default | what it controls |
|---|---|---|
| `num_envs` | `256` | Number of JAX environments stepped in parallel per vector-step; each vector-step adds `num_envs` transitions to the buffer. Larger = higher throughput (especially on a GPU `jaxlib`) and more decorrelated batches, at the cost of more VRAM/RAM and more env compute per step. Also sets the replay buffer's **per-env stride** for TD(λ), and — when `training.update_freq` is `null` — the update cadence defaults to one gradient update per vector-step (`update_freq = num_envs`). |
| `episode_steps` | `training.max_steps` | Game ticks before an episode is truncated and that env resets. |
| `actor` | `"thread"` if `threaded` else `"serial"` | Rollout/learner execution mode: `"serial"` \| `"thread"` — see **Actor modes** below. Takes precedence over `threaded`. (`"mp_env"` was removed once the env became GPU-resident; old configs map to `"serial"`.) |
| `threaded` | `true` | **Legacy bool**, only consulted to pick the default `actor` when `actor` is absent. Ignored once `actor` is set. |
| `actor_sync_every` | `2` | **`thread` mode only.** Gradient updates between copying the freshly-trained policy weights into the rollout actor net (larger = staler rollout policy, less locking). |
| `rollout_lead` | `8` | **`thread` mode only.** Backpressure cap — the env rollout may run at most `rollout_lead × num_envs` transitions ahead of the learner's update budget, so the intended number of gradient steps actually runs. |
| `ship_speed` | `6.0` | Engine physics: max fleet speed. |
| `comet_speed` | `4.0` | Engine physics: comet drift speed. |
| `reward_type` | `"ship_advantage"` | **Fallback reward only** (see precedence note). `"ship_advantage"` = `reward_scale × [(Δmy − Δopp) ships]`; `"native"` = terminal ±1 result only. |
| `reward_scale` | `0.01` | **Fallback only.** Multiplier for `reward_type="ship_advantage"`. |
| `win_bonus` | `100.0` | **Fallback only.** Terminal win/loss bonus magnitude. |
| `num_players` | — | **Ignored by the trainer.** 2p/4p is driven by `environment.ratio_4p`, not this key (see **2-player vs 4-player**). Kept only because the `JaxVecEnvAdapter`/benchmark constructors accept it. |

> The committed `train.json` drops two **redundant** keys: `threaded` (dead once
> `actor` is set explicitly) and `num_players` (never read by the trainer). Don't
> re-add them. `actor_sync_every` / `rollout_lead` are kept but only do anything in
> `thread` mode.

### Actor modes (`actor`)

Since the env runs the opponent + reward + obs **on-device**, fused with the
engine step (see [On-device fast path](#on-device-fast-path)), the rollout is
GPU-resident and `serial` is usually fastest — the old `mp_env` subprocess actor
(which existed to overlap a CPU-bound env with the GPU update) has been removed.

- **`serial`** (recommended, committed default) — single thread:
  `select → step → add → update` inline each vector-step. Simplest and fully
  deterministic. With a GPU `jaxlib` the env step is async JAX dispatch, so there
  is little host work left to hide.
- **`thread`** — an env-rollout thread and a learner thread share the trainer,
  buffer, and a separate **actor network** (synced every `actor_sync_every`
  updates, backpressured by `rollout_lead`). Kept for `self_play` (the opponent
  policy lives in-process) and experimentation; with a GPU-resident env it usually
  does not beat `serial` (both contend for the one GPU).

### Reward precedence (important)

If the top-level **`reward`** list is non-empty (the normal case — see
[Reward schemes](#reward-schemes)), the env computes reward from those composable
components and **`reward_type` / `reward_scale` / `win_bonus` are ignored**. Those
three only take effect as a fallback when `reward` is empty. So in a config that
already has a `reward` list, editing `reward_scale`/`win_bonus` here does nothing.

### 2-player vs 4-player

`jax_env.num_players` is **not** read by training. The trainer always builds a
2-player env set and, when `environment.ratio_4p > 0`, an additional 4-player set
holding `int(num_envs × ratio_4p)` of the envs; both sets share one replay buffer
and policy. Set the mix via `environment.ratio_4p` (`0.0` = all 2-player).

### CPU vs GPU placement

There is **no `jax_env.device` key** — the env runs on JAX's default backend:
**GPU when a CUDA-enabled `jaxlib` is installed, CPU otherwise** (nothing in the
code pins a device). `XLA_PYTHON_CLIENT_PREALLOCATE=false` is set before
`import jax` so the JAX env can share one GPU with PyTorch. Notes:

- `execution.cpu_force` only moves **PyTorch** to CPU; it does **not** affect JAX.
- To force the **env** onto CPU (e.g. to free the GPU for the learner), launch with
  `JAX_PLATFORMS=cpu python train_orbit_wars.py`.
- The on-device fast path runs the opponent + reward + obs on the same device as
  the engine step, so on a GPU `jaxlib` the per-step host cost is just the done-env
  auto-reset and one obs copy. Benchmark per machine with
  `benchmark_jax_env_step.py` (pure env step) and `benchmark_jax_train_loop.py`
  (full loop).

### On-device fast path

When `environment.opponent` is a built-in heuristic (`random` / `greedy` /
`agent1` / `mixed`) and every `reward` component is JAX-portable (i.e. **not**
`LaunchDistancePenalty`), the env fuses the **opponent + engine step + reward**
into one jitted, vmapped function and extracts the **observation** on-device too
([jax_env/jax_opponents.py](jax_env/jax_opponents.py),
[jax_env/jax_reward.py](jax_env/jax_reward.py),
[jax_env/jax_obs.py](jax_env/jax_obs.py)). So per step the host only does the
done-env auto-reset and a single obs copy. `self_play` (a torch-policy opponent) and
`LaunchDistancePenalty` automatically fall back to the NumPy path; the NumPy
implementations remain as that fallback and as the parity oracle
(`test_jax_port.py`).

The obs builder itself uses the **hybrid fleet-into-planet decoder** (see
[Hybrid decoder](#hybrid-fleet-into-planet-decoder-approach-e) below): the full
`[MAX_FLEETS=1024, NET_MP=40]` fleet sweep runs on-device fused into the obs step,
so the sequence arriving at the transformer is only 41 tokens instead of 140.


## Hybrid fleet-into-planet decoder (Approach E)

Instead of emitting a separate fleet token per in-flight fleet (the old layout: 40
planet tokens + 100 fleet tokens = **140** tokens), each fleet's information is
**folded into the planet token it relates to**. The transformer sequence becomes:

> **40 planet tokens + 1 meta row = 41 tokens**  (`NET_MP + 1`)

Attention complexity scales with sequence length squared, so 41 vs 140 tokens is
roughly **12× less attention work**. On the A30 this is expected to roughly halve
model forward/backward time.

### Token layout — width 35 (`TOKEN_DIM`)

Each of the 40 planet token rows:

| idx | field | note |
|-----|-------|------|
| 0:4 | owner one-hot (×4) | perspective-swapped: acting player = owner 0 |
| 4 | radius | |
| 5 | garrison ships | fed through `LearnedFourierScalarEncoding` |
| 6 | production | |
| 7 | moving flag | 1 if planet orbits (within rotation radius) |
| 8 | angular velocity | |
| 9–10 | x, y | fed to `LearnedFourierPosEncoding` |
| 11 | time step | |
| 12:15 | **incoming slot 0** = [ships, eta, opp_owner] | soonest hostile inbound fleet |
| 15:18 | **incoming slot 1** | 2nd soonest hostile inbound fleet |
| 18:22 | **outgoing slot 0** = [ships, sinθ, cosθ, speed] | largest friendly outgoing fleet |
| 22:26 | **outgoing slot 1** | 2nd largest friendly outgoing fleet |
| 26:31 | **incoming pooled** = [log1p cnt, log1p Σships, soonest_eta, wmean_eta, log1p maxships] | all hostile inbound |
| 31:35 | **outgoing pooled** = [log1p cnt, log1p Σships, wmean sinθ, wmean cosθ] | all friendly outgoing |

Row 40 (index `NET_MP`) is the **meta row** — carries `[4 × planet_count, 4 × log1p production, 4 × log1p total_ships]` (12 values, rest padded to 35). The model reads it off before the transformer and projects it into a single metadata token.

### Binding

- **Outgoing** — exact: a fleet is "outgoing from planet p" iff its `from_planet` slot equals p's index **and** its owner matches p's owner (after perspective swap).
- **Incoming** — geometric: the swept-segment look-ahead `_fleet_planet_sweep` over the full `[MAX_FLEETS=1024, NET_MP=40]` matrix; runs on-device, one call per env-step. A fleet is incoming-hostile to p iff its straight-line path (60-tick horizon) passes within p's radius **and** its owner differs from p's owner.
- A fleet aimed at planet p from planet q appears in **both** p's incoming slots (threat to p) and q's outgoing slots (asset from q) — this is correct: both planets are genuinely affected.
- The top-k explicit slots (k=2 per type) are complemented by pooled summaries so long-tail fleets beyond k are never silently dropped.

### Knobs

| constant | default | effect |
|----------|---------|--------|
| `K_IN` | 2 | explicit incoming slots per planet |
| `K_OUT` | 2 | explicit outgoing slots per planet |
| `TOKEN_DIM` | 35 | computed: `12 + K_IN×3 + K_OUT×4 + 5 + 4` |

`K_IN` and `K_OUT` are defined in both [jax_env/jax_obs.py](jax_env/jax_obs.py) and [model/SAC.py](model/SAC.py); the parity test `test_jax_decoder.py` will catch any drift between them.

### Breaking change — retrain required

`TOKEN_DIM` changed from 13 → 35 and `SEQ_LEN` from 140 → 41. Old checkpoints and
replay buffers are incompatible. Set `execution.resume = null` and clear any old
`replay_buffer.npz` before training.

### Files

| file | role |
|------|------|
| [jax_env/jax_obs.py](jax_env/jax_obs.py) | Training-time obs builder (JAX, on-device) |
| [model/SAC.py `Encoder`](model/SAC.py) | Submission-time obs builder (NumPy, live game) |
| [test_jax_decoder.py](test_jax_decoder.py) | Binding unit test + JAX↔Encoder parity (max diff ~1e-7) |


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

The transformer is fed an extra **metadata token** prepended to the planet sequence.
It encodes a per-player global summary so the model doesn't have to reconstruct the
overall picture by attending across individual tokens.

For each of the 4 player slots (player 0 = our agent after the perspective swap,
1–3 = opponents) it carries:

- **planet count**
- **total production** (`log1p`-compressed)
- **total ships** — planets **and** in-flight fleets (`log1p`-compressed)

= a 12-dim vector packed into the **trailing meta row** (index 40) of the
observation by the obs builder ([jax_env/jax_obs.py](jax_env/jax_obs.py) for
training, `Encoder.encode` for submission), projected by a learned
`nn.Linear(12, d_model)` (`meta_proj`), and prepended as token 0 inside the
transformer. Because the encoder is full self-attention, every planet token (and the
`cls` value token) can read it directly.

Implementation notes:
- Built in the obs layer rather than derived from the state tensor inside `forward`,
  so the meta row accounts for **in-flight fleet ships** (which are no longer
  represented as their own token rows in the hybrid decoder).
- `Q`/`V` read the `cls` token at position 0 (followed by the metadata token);
  `P` reads per-planet outputs at `out[:, 1:1+max_planets]`.
- Players absent in a 2-player game contribute zero to all three totals.


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

> **Breaking change — retrain from scratch.** The state width changed (14 → 13
> when comets were stripped, then 13 → 35 with the hybrid fleet-into-planet
> decoder). **Old checkpoints and replay buffers are incompatible.** Set
> `execution.resume = null` and clear any old `replay_buffer.npz` before training.
> `STATE_DIM` now equals `TOKEN_DIM = 35`; the networks default to `state_dim=TOKEN_DIM`.


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