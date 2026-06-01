# 501-Legion


## JSON train args

mixed_random_ratio_decay = None -> spawn all training episodes


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