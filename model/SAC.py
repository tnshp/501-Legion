import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np 

from utils.pos_encoding import (
    get_pos_encoding,
    LearnedFourierPosEncoding,
    LearnedFourierScalarEncoding,
    FourierAngleEncoding,
)

# ── Hybrid decoder geometry (NumPy, submission path) ─────────────────────────
# These constants MUST match jax_env/jax_obs.py / jax_opponents.py (the training
# obs builder); test_jax_decoder.py asserts the two implementations agree.
_ENC_CENTER   = 50.0
_ENC_HORIZON  = 60.0     # fleet look-ahead horizon (ticks)
_ENC_MAXSPEED = 6.0
_K_IN  = 2               # incoming-hostile slots per planet
_K_OUT = 2               # outgoing-friendly slots per planet


def _np_fleet_speed(ships):
    s = np.maximum(ships, 1.0)
    return np.minimum(_ENC_MAXSPEED,
                      1.0 + (_ENC_MAXSPEED - 1.0) * (np.log(s) / np.log(1000.0)) ** 1.5)


def _np_seg_point_dist(cx, cy, ax, ay, bx, by):
    """Distance from point (cx,cy) to segment (ax,ay)->(bx,by) and the clamped t."""
    vx = bx - ax; vy = by - ay
    wx = cx - ax; wy = cy - ay
    vv = vx * vx + vy * vy
    t  = np.where(vv > 1e-9, (wx * vx + wy * vy) / np.where(vv > 1e-9, vv, 1.0), 0.0)
    t  = np.clip(t, 0.0, 1.0)
    qx = ax + t * vx; qy = ay + t * vy
    return np.sqrt((cx - qx) ** 2 + (cy - qy) ** 2), t


class Encoder:
    """Live-game (submission) observation encoder — hybrid fleet-into-planet decoder.

    Single-env NumPy mirror of jax_env/jax_obs.py: each fleet is folded into the
    planet token it relates to (incoming-hostile / outgoing-friendly explicit
    slots + pooled summary) instead of emitting separate fleet tokens, and a
    trailing meta row carries the per-player aggregate.  Output:
    ``[max_planets + 1, TOKEN_DIM=35]`` (see the layout note above the networks).

      planets - [id, owner, x, y, radius, ships, production]
      fleets  - [id, owner, x, y, angle, from_planet_id, ships] or empty
    Owners are already perspective-swapped upstream (player 0 = acting player);
    comets are stripped upstream.  Outgoing binding maps fleet.from_planet_id to
    the planet ROW via the planet id, the live-game analogue of the engine's
    from_planet SLOT index used in training.
    """

    def __init__(self, max_planets=40, max_fleets=100, max_speed=6.0):
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        self.max_speed = max_speed
        self.sun = np.array([50, 50])
        #limits of the box in which planets can move
        self.box = [(0, 100),  (0, 100)]
        self.max_time_step = 500
    
    def get_speed(self, ships):
        return 1.0 + (self.max_speed - 1.0) * (np.log(ships) / np.log(1000)) ** 1.5
    
    def get_planet_mapping(self, initial_planets):
        sun = self.sun
        oribital_radius = ((initial_planets[:, 2:4] - sun)**2).sum(axis=1)**0.5
        planet_radius = initial_planets[:, 4]
        moving_planets = oribital_radius + planet_radius < 50

        planet_ids = initial_planets[:, 0]
        planets_mapping = {}
        for i, planet_id in enumerate(planet_ids):
            planets_mapping[int(planet_id)] = int(moving_planets[i])

        return planets_mapping
    
    def _pad_array(self, arr, max_size):
        current_size = min(arr.shape[0], max_size)
        feature_dim = arr.shape[1]

        padded_arr = np.zeros((max_size, feature_dim))
        mask = np.zeros(max_size)

        padded_arr[:current_size] = arr[:current_size]
        mask[:current_size] = 1

        return padded_arr, mask

    def encode(self, planets, fleets, initial_planets, angular_velocity, time_step,
               apply_padding=True):
        """Encode one live state → ``[max_planets + 1, TOKEN_DIM]`` (hybrid decoder).

        ``apply_padding`` is accepted for signature compatibility; the output is
        always fixed-size (planet rows are padded to ``max_planets`` and the meta
        row appended).  Returns ``(state, mask)`` where mask is 1 on real planet
        rows + the meta row.
        """
        assert planets.shape[-1] == 7, f"Expected planets to have 7 features, got {planets.shape[-1]}"
        NMP = self.max_planets
        n   = min(planets.shape[0], NMP)
        planets = planets[:n]

        if fleets is None or fleets.size == 0:
            fleets = np.empty((0, 7), dtype=np.float32)
        else:
            assert fleets.shape[-1] == 7, f"Expected fleets to have 7 features, got {fleets.shape[-1]}"

        # ── Planet base fields ───────────────────────────────────────────────
        po  = planets[:, 1].astype(int)                  # owner (already swapped)
        pid = planets[:, 0].astype(int)                  # planet id (for outgoing binding)
        px  = planets[:, 2].astype(np.float32)
        py  = planets[:, 3].astype(np.float32)
        pr  = planets[:, 4].astype(np.float32)
        psh = planets[:, 5].astype(np.float32)
        ppr = planets[:, 6].astype(np.float32)

        p_oh = np.zeros((n, 4), dtype=np.float32)
        ok   = po != -1
        p_oh[ok] = np.eye(4, dtype=np.float32)[np.clip(po[ok], 0, 3)]

        pmap   = self.get_planet_mapping(initial_planets)
        p_mov  = np.array([pmap.get(int(i), 0) for i in pid], dtype=np.float32)
        p_mov  = np.nan_to_num(p_mov)
        ang    = np.float32(angular_velocity)
        ts     = np.float32(time_step)

        base = np.column_stack([
            p_oh[:, 0], p_oh[:, 1], p_oh[:, 2], p_oh[:, 3],
            pr, psh, ppr, p_mov, np.full(n, ang, np.float32),
            px, py, np.full(n, ts, np.float32),
        ]).astype(np.float32)                            # [n, 12]

        # ── Fleet arrays ─────────────────────────────────────────────────────
        m = fleets.shape[0]
        if m > 0:
            fo    = fleets[:, 1].astype(int)
            fx    = fleets[:, 2].astype(np.float32)
            fy    = fleets[:, 3].astype(np.float32)
            fa    = fleets[:, 4].astype(np.float32)
            ffrom = fleets[:, 5].astype(int)
            fsh   = fleets[:, 6].astype(np.float32)

            fsp = _np_fleet_speed(fsh)
            fex = fx + np.cos(fa) * fsp * _ENC_HORIZON
            fey = fy + np.sin(fa) * fsp * _ENC_HORIZON
            d, t = _np_seg_point_dist(px[None, :], py[None, :],
                                      fx[:, None], fy[:, None], fex[:, None], fey[:, None])
            hit     = d <= pr[None, :]                   # [m, n]
            eta     = t                                  # arrive/HORIZON = t ∈ [0,1], [m,n]
            hostile = hit & (fo[:, None] != -1) & (fo[:, None] != po[None, :])
            out_mask = (ffrom[:, None] == pid[None, :]) & (fo[:, None] == po[None, :])
            fsh_c = fsh[:, None]
        else:
            hostile  = np.zeros((0, n), dtype=bool)
            out_mask = np.zeros((0, n), dtype=bool)
            eta = np.zeros((0, n), np.float32)
            fsh = fa = fo = np.zeros((0,), np.float32)
            fsh_c = np.zeros((0, 1), np.float32)

        cols = np.arange(n)

        # ── Incoming explicit slots (K_IN soonest-arriving hostile) ──────────
        cols2 = np.arange(n)
        in_score = np.where(hostile, eta, np.inf)        # [m, n]
        in_slots = np.zeros((n, _K_IN * 3), np.float32)
        s = in_score.copy()
        for k in range(_K_IN):
            if m == 0:
                break
            idx = np.argmin(s, axis=0)                   # [n]
            sel = hostile[idx, cols2]
            in_slots[:, 3 * k + 0] = np.where(sel, fsh[idx], 0.0)
            in_slots[:, 3 * k + 1] = np.where(sel, eta[idx, cols2], 0.0)
            in_slots[:, 3 * k + 2] = np.where(sel, fo[idx].astype(np.float32), 0.0)
            s[idx, cols2] = np.inf

        # ── Incoming pooled ──────────────────────────────────────────────────
        in_cnt   = hostile.sum(0).astype(np.float32)
        in_shsum = (fsh_c * hostile).sum(0)
        eta_h    = np.where(hostile, eta, np.inf)
        in_soon  = eta_h.min(0) if m else np.zeros(n, np.float32)
        in_soon  = np.where(np.isfinite(in_soon), in_soon, 0.0)
        den_i    = np.where(in_shsum > 0, in_shsum, 1.0)
        in_wmean = np.where(in_shsum > 0, (eta * fsh_c * hostile).sum(0) / den_i, 0.0)
        in_maxsh = (np.where(hostile, fsh_c, 0.0).max(0) if m else np.zeros(n, np.float32))
        in_pool  = np.column_stack([
            np.log1p(in_cnt), np.log1p(in_shsum), in_soon, in_wmean, np.log1p(in_maxsh),
        ]).astype(np.float32)                            # [n, 5]

        # ── Outgoing explicit slots (K_OUT largest-by-ships friendly) ────────
        out_score = np.where(out_mask, fsh_c, -1.0)
        out_slots = np.zeros((n, _K_OUT * 4), np.float32)
        s = out_score.copy()
        for k in range(_K_OUT):
            if m == 0:
                break
            idx   = np.argmax(s, axis=0)
            sel   = out_mask[idx, cols] & (fsh[idx] > 0.0)
            ships = np.where(sel, fsh[idx], 0.0)
            angk  = fa[idx]
            out_slots[:, 4 * k + 0] = ships
            out_slots[:, 4 * k + 1] = np.where(sel, np.sin(angk), 0.0)
            out_slots[:, 4 * k + 2] = np.where(sel, np.cos(angk), 0.0)
            out_slots[:, 4 * k + 3] = np.where(sel, _np_fleet_speed(np.maximum(ships, 1.0)), 0.0)
            s[idx, cols] = -1.0

        # ── Outgoing pooled ──────────────────────────────────────────────────
        out_cnt   = out_mask.sum(0).astype(np.float32)
        out_shsum = (fsh_c * out_mask).sum(0)
        den_o     = np.where(out_shsum > 0, out_shsum, 1.0)
        if m:
            out_wsin = np.where(out_shsum > 0, (np.sin(fa)[:, None] * fsh_c * out_mask).sum(0) / den_o, 0.0)
            out_wcos = np.where(out_shsum > 0, (np.cos(fa)[:, None] * fsh_c * out_mask).sum(0) / den_o, 0.0)
        else:
            out_wsin = np.zeros(n, np.float32); out_wcos = np.zeros(n, np.float32)
        out_pool = np.column_stack([
            np.log1p(out_cnt), np.log1p(out_shsum), out_wsin, out_wcos,
        ]).astype(np.float32)                            # [n, 4]

        tokens = np.concatenate([base, in_slots, out_slots, in_pool, out_pool], axis=1)  # [n, 35]

        # ── Meta row: per-player [planet_count, log1p production, log1p ships] ─
        meta = np.zeros(12, np.float32)
        for player in range(4):
            pm = (po == player)
            fm = (fo == player) if m else np.zeros(0, bool)
            meta[player]     = pm.sum()
            meta[4 + player] = np.log1p(ppr[pm].sum())
            meta[8 + player] = np.log1p(psh[pm].sum() + (fsh[fm].sum() if m else 0.0))

        state = np.zeros((NMP + 1, TOKEN_DIM), np.float32)
        state[:n] = tokens
        state[NMP, :12] = meta

        mask = np.zeros(NMP + 1, np.float32)
        mask[:n] = 1.0
        mask[NMP] = 1.0
        return state, (mask if apply_padding else None)
    
    def encode_batch(self, batched_planets, batched_fleets, batched_initial_planets, batched_angular_velocity,
                     batch_time_steps=None, apply_padding=True):
        """
        Encode a batch of states.

        Args:
            batched_planets: list of planet arrays, each [n_planets, 7]
            batched_fleets: list of fleet arrays, each [n_fleets, 7] or empty
            batched_initial_planets: list of initial planet arrays, each [n_initial, 7]
            batched_angular_velocity: array or list of angular velocities, shape [batch_size]
            batch_time_steps: array/list of time steps for each batch item, or single value for all
            apply_padding: whether to apply padding and return masks
            
        Returns:
            batch_states: [batch_size, max_planets + max_fleets, feature_dim]
            batch_masks: [batch_size, max_planets + max_fleets] if apply_padding=True, else None
        """
        batch_size = len(batched_planets)
        
        # Handle time steps
        if batch_time_steps is None:
            batch_time_steps = [100] * batch_size
        elif isinstance(batch_time_steps, (int, float)):
            batch_time_steps = [batch_time_steps] * batch_size
        
        batch_states = []
        batch_masks = []
        
        for i in range(batch_size):
            state, mask = self.encode(
                batched_planets[i],
                batched_fleets[i],
                batched_initial_planets[i],
                batched_angular_velocity[i],
                batch_time_steps[i],
                apply_padding=apply_padding
            )
            batch_states.append(state)
            if apply_padding:
                batch_masks.append(mask)
        
        # Stack into batch
        batch_states = np.stack(batch_states, axis=0)  # [batch_size, max_planets + max_fleets, feature_dim]
        
        if apply_padding:
            batch_masks = np.stack(batch_masks, axis=0)  # [batch_size, max_planets + max_fleets]
            return batch_states, batch_masks
        else:
            return batch_states, None


    
_META_DIM = 12  # 4 players × [planet_count, log1p production, log1p ships]

# ── Hybrid fleet-into-planet token layout (TOKEN_DIM = 35) ───────────────────
# Built by jax_env/jax_obs.py (training) and model.SAC.Encoder.encode (submission):
#   0:4 owner one-hot | 4 radius | 5 ships | 6 production | 7 moving | 8 ang_vel
#   9 x | 10 y | 11 time_step
#   12:18  incoming-hostile slots  (K_IN=2  × [ships, eta, owner])
#   18:26  outgoing-friendly slots (K_OUT=2 × [ships, sinθ, cosθ, speed])
#   26:31  incoming pooled [log1p cnt, log1p Σships, soonest_eta, wmean_eta, log1p maxships]
#   31:35  outgoing pooled [log1p cnt, log1p Σships, wmean sinθ, wmean cosθ]
# The obs has NET_MP planet tokens followed by ONE meta row (per-player aggregate);
# the model splits it off and projects it into a single "metadata" token.
TOKEN_DIM  = 35
_SHIP_COLS = [5, 12, 15, 18, 22]   # ship-magnitude columns → shared LearnedFourierScalarEncoding
_POS_COLS  = [9, 10]               # (x, y) → positional encoder
_TIME_COL  = 11                    # time_step → positional encoder
# Everything else fed raw to the token projection (owner, radius, prod, flags, slot
# angles/eta/owner, pooled summaries):
_RAW_COLS  = [0, 1, 2, 3, 4, 6, 7, 8,
              13, 14, 16, 17,
              19, 20, 21, 23, 24, 25,
              26, 27, 28, 29, 30, 31, 32, 33, 34]


def _embed_tokens(state, max_planets, ship_encoder, proj, pos_encoder):
    """Shared input embedding for the hybrid decoder.

    Splits the trailing meta row off, projects the NET_MP planet tokens (rich
    Fourier encoding of every ship-magnitude column, the rest raw) to d_model,
    and adds the learned positional encoding.

    Returns (token_emb [B, max_planets, d_model], meta_raw [B, _META_DIM]).
    """
    tokens = state[:, :max_planets, :]                 # [B, NMP, TOKEN_DIM]
    meta   = state[:, max_planets, :_META_DIM]         # [B, 12]

    ship_blocks = [ship_encoder(tokens[:, :, c:c + 1]) for c in _SHIP_COLS]
    raw         = tokens[:, :, _RAW_COLS]
    token_feats = torch.cat([raw, *ship_blocks], dim=-1)
    emb = proj(token_feats)                            # [B, NMP, d_model]

    pos  = tokens[:, :, _POS_COLS]                     # [B, NMP, 2]
    time = tokens[:, :, _TIME_COL]                     # [B, NMP]
    emb = emb + pos_encoder(pos, time)
    return emb, meta


def _proj_in_dim(ship_encoder):
    return len(_RAW_COLS) + len(_SHIP_COLS) * ship_encoder.out_dim


# ============================================================
# Flash-Attention encoder (SDPA)
# ============================================================
#
# The stock nn.TransformerEncoderLayer only reaches a fused/flash attention
# kernel on its inference "fast path"; in training (and under torch.compile) it
# graph-breaks and runs the slow math path.  These modules route attention
# through F.scaled_dot_product_attention, which dispatches to the
# FlashAttention-2 kernel on Ampere+ GPUs (A30, RTX 30xx) under fp16/bf16
# autocast and falls back to the memory-efficient / math kernel everywhere else
# (older GPUs, CPU, fp32).
#
# The parameter layout is deliberately IDENTICAL to nn.TransformerEncoderLayer
# (self_attn.in_proj_weight/in_proj_bias/out_proj, linear1, linear2, norm1,
# norm2, stacked under an .layers ModuleList), so a checkpoint trained with one
# implementation loads unchanged into the other — switch `attn_impl` freely
# without retraining.

class _SDPASelfAttention(nn.Module):
    """Multi-head self-attention via F.scaled_dot_product_attention.

    Parameters mirror nn.MultiheadAttention exactly (combined QKV projection in
    `in_proj_weight`/`in_proj_bias`, output projection in `out_proj`), so the
    state_dict keys match the stock encoder's `self_attn.*`.
    """

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead    = nhead
        self.head_dim = d_model // nhead
        self.dropout  = dropout
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias   = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj       = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        # Match nn.MultiheadAttention's init (xavier on the fused QKV, zero biases;
        # out_proj.weight keeps the nn.Linear default).
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x):                         # x: [B, L, D]
        B, L, _ = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)   # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        # [B, L, D] → [B, nhead, L, head_dim]
        q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        # No attention mask (all tokens attend to all — padding is zeroed
        # upstream), which keeps this on the flash-eligible path.
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0)
        attn = attn.transpose(1, 2).reshape(B, L, self.nhead * self.head_dim)
        return self.out_proj(attn)


class _SDPAEncoderLayer(nn.Module):
    """Post-norm transformer encoder layer (matches nn.TransformerEncoderLayer
    defaults: norm_first=False, ReLU FFN) but with SDPA self-attention."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = _SDPASelfAttention(d_model, nhead, dropout=dropout)
        self.linear1   = nn.Linear(d_model, dim_feedforward)
        self.dropout   = nn.Dropout(dropout)
        self.linear2   = nn.Linear(dim_feedforward, d_model)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout1  = nn.Dropout(dropout)
        self.dropout2  = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def _sa_block(self, x):
        return self.dropout1(self.self_attn(x))

    def _ff_block(self, x):
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, src):
        x = self.norm1(src + self._sa_block(src))
        x = self.norm2(x + self._ff_block(x))
        return x


class _SDPAEncoder(nn.Module):
    """Stack of _SDPAEncoderLayer (mirrors nn.TransformerEncoder's `.layers`)."""

    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, activation="relu"):
        super().__init__()
        self.layers = nn.ModuleList([
            _SDPAEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])

    def forward(self, src):
        x = src
        for layer in self.layers:
            x = layer(x)
        return x


def _build_encoder(d_model, nhead, num_layers, dim_feedforward, dropout,
                   attn_impl="torch"):
    """Construct the transformer encoder for the chosen attention implementation.

    attn_impl="flash"  → SDPA encoder (FlashAttention-2 on Ampere+; safe
                         fallback elsewhere), compile-friendly.
    attn_impl="torch"  → stock nn.TransformerEncoder (default; unchanged).

    Both produce identical state_dict keys, so checkpoints are interchangeable.
    """
    if attn_impl == "flash":
        return _SDPAEncoder(d_model, nhead, num_layers, dim_feedforward, dropout,
                            activation="relu")
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
        dropout=dropout, activation="relu", batch_first=True)
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


class Q_network(nn.Module):
    def __init__(self,
                 state_dim=TOKEN_DIM,
                 action_dim=4,   # matches OrbitWarsEnv.ACTION_DIM
                 max_planets=40,
                 max_fleets=100,
                 d_model=128,
                 nhead=4,
                 num_layers=3,
                 dim_feedforward=512,
                 dropout=0.1,
                 attn_impl="torch"):

        super(Q_network, self).__init__()
        self.max_planets = max_planets
        self.max_fleets = max_fleets

        # Each ship-magnitude column (planet garrison + incoming/outgoing slot
        # ships) is encoded with the SAME learned Fourier scalar encoder.
        self.ship_encoder  = LearnedFourierScalarEncoding()
        self.P = nn.Linear(_proj_in_dim(self.ship_encoder), d_model)
        self.A = nn.Linear(action_dim, d_model)

        self.pos_encoder = LearnedFourierPosEncoding(d_model)

        # Projects the per-player aggregate (the obs meta row, built in
        # jax_env/jax_obs.py / Encoder.encode) into a single "metadata" token.
        self.meta_proj = nn.Linear(_META_DIM, d_model)

        self.value_head = nn.Linear(d_model, 1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, d_model))

        #transformer - encoder only (attn_impl="flash" routes attention through SDPA)
        self.transformer = _build_encoder(
            d_model, nhead, num_layers, dim_feedforward, dropout, attn_impl)

        self.d_model = d_model


    def forward(self, state, action):
        """
        Args:
            state : [B, NET_MP + 1, TOKEN_DIM]  (planet tokens + trailing meta row)
            action: [B, max_planets, action_dim]

        Returns:
            value: [batch_size]
        """
        emb, meta = _embed_tokens(state, self.max_planets,
                                  self.ship_encoder, self.P, self.pos_encoder)
        meta_token = self.meta_proj(meta).unsqueeze(1)        # [B, 1, d_model]
        action = self.A(action)                               # [B, max_planets, d_model]

        batch_size = state.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        sep_token = self.sep_token.expand(batch_size, -1, -1)

        src = torch.cat((cls_token, meta_token, emb, sep_token, action), dim=1)

        out_ = self.transformer(src)
        out_cls = out_[:, 0, :]
        value = self.value_head(out_cls)
        return value.squeeze(-1)  # [batch_size]
    

class V_network(nn.Module):
    def __init__(self, 
                 state_dim=TOKEN_DIM,
                 max_planets=40,
                 max_fleets=100,
                 d_model=128,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=128,
                 dropout=0.1,
                 attn_impl="torch"):

        super(V_network, self).__init__()
        self.max_planets = max_planets
        self.max_fleets = max_fleets

        self.ship_encoder  = LearnedFourierScalarEncoding()
        self.P = nn.Linear(_proj_in_dim(self.ship_encoder), d_model)

        self.pos_encoder = LearnedFourierPosEncoding(d_model)

        # Projects the per-player aggregate (the obs meta row, built in
        # jax_env/jax_obs.py / Encoder.encode) into a single "metadata" token.
        self.meta_proj = nn.Linear(_META_DIM, d_model)

        self.value_head = nn.Linear(d_model, 1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        #transformer - encoder only (attn_impl="flash" routes attention through SDPA)
        self.transformer = _build_encoder(
            d_model, nhead, num_layers, dim_feedforward, dropout, attn_impl)
        self.d_model = d_model

    def forward(self, state):
        emb, meta = _embed_tokens(state, self.max_planets,
                                  self.ship_encoder, self.P, self.pos_encoder)
        meta_token = self.meta_proj(meta).unsqueeze(1)

        batch_size = state.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)

        src = torch.cat((cls_token, meta_token, emb), dim=1)
        out_ = self.transformer(src)
        out_cls = out_[:, 0, :]
        value = self.value_head(out_cls)
        return value.squeeze(-1)  # [batch_size]
    

class P_network(nn.Module):
    def __init__(self,
                 d_model=128,
                 state_dim=TOKEN_DIM,
                 action_dim=4,   # matches OrbitWarsEnv.ACTION_DIM
                 max_planets=40,
                 max_fleets=100, 
                 nhead=4,
                 num_layers=3,
                 dim_feedforward=128,
                 dropout=0.1,
                 attn_impl="torch"):

        super(P_network, self).__init__()

        self.max_planets = max_planets
        self.max_fleets = max_fleets

        self.ship_encoder  = LearnedFourierScalarEncoding()
        self.P = nn.Linear(_proj_in_dim(self.ship_encoder), d_model)

        self.pos_encoder = LearnedFourierPosEncoding(d_model)

        # Projects the per-player aggregate (the obs meta row, built in
        # jax_env/jax_obs.py / Encoder.encode) into a single "metadata" token.
        self.meta_proj = nn.Linear(_META_DIM, d_model)

        self.mu_head = nn.Linear(d_model, action_dim)
        self.sigma_head = nn.Linear(d_model, action_dim)

        #transformer - encoder only (attn_impl="flash" routes attention through SDPA)
        self.transformer = _build_encoder(
            d_model, nhead, num_layers, dim_feedforward, dropout, attn_impl)

        self.d_model = d_model
        self.action_dim = action_dim

    def forward(self, state):
        emb, meta = _embed_tokens(state, self.max_planets,
                                  self.ship_encoder, self.P, self.pos_encoder)
        meta_token = self.meta_proj(meta).unsqueeze(1)

        # Process through transformer. The metadata token sits at position 0, so
        # the planet tokens (which drive the per-planet action heads) start at 1.
        src = torch.cat((meta_token, emb), dim=1)  # [B, 1 + max_planets, d_model]
        out_ = self.transformer(src)

        #only pass num planets tokens through heads, as action is only for planets
        mu = self.mu_head(out_[:, 1:1 + self.max_planets, :])
        sigma = F.softplus(self.sigma_head(out_[:, 1:1 + self.max_planets, :])) + 1e-5
        # Clamp sigma to a bounded range so the entropy term cannot drive it to
        # extremes: too small spikes -log(sigma) in the log-prob, too large blows
        # up the (now tanh-squashed) action. exp(-5)≈6.7e-3, exp(2)≈7.39.
        sigma = sigma.clamp(min=math.exp(-5.0), max=math.exp(2.0))

        return mu, sigma

    def sample(self, state):
        """
        Sample a tanh-squashed action via reparameterisation and return log π(a|s).

        The action is squashed with tanh so it lies in [-1, 1] (matching the
        env's action_space and decode_action's assumption) and so the policy
        entropy is bounded — without this the entropy term drives sigma → ∞,
        producing exploding Q-targets and NaNs.

        Returns:
            action   : [B, max_planets, action_dim]  in [-1, 1]
            log_prob : [B, 1]
        """
        mu, sigma = self.forward(state)
        eps        = torch.randn_like(sigma)
        raw_action = mu + eps * sigma            # pre-squash Gaussian sample
        action     = torch.tanh(raw_action)      # bounded to [-1, 1]

        # Gaussian log-density of raw_action, then the tanh change-of-variables
        # correction: log π(a) = log N(raw) - Σ log(1 - tanh(raw)^2).
        log_prob = (
            -0.5 * eps ** 2
            - sigma.log()
            - 0.5 * math.log(2.0 * math.pi)
        )
        log_prob = log_prob - torch.log(1.0 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(dim=(-2, -1), keepdim=True).squeeze(-1)  # [B, 1]

        return action, log_prob

    def deterministic_action(self, state):
        """Greedy (mean) action, tanh-squashed to [-1, 1].  Used for evaluation."""
        mu, _ = self.forward(state)
        return torch.tanh(mu)
    
class ActionDecoder(nn.Module):
    def __init__(self, 
                 d_action=64):  
        super(ActionDecoder, self).__init__()
        self.d_action = d_action 

    def sample_action(self, mu, sigma):
        #gaussian sampling with reparameterization trick
        eps = torch.randn_like(sigma)
        action = mu + eps * sigma  # shape: [num_planets, action_dim]
        return action
    
    def forward(self, mu, sigma, mask=None):
        action = self.sample_action(mu, sigma)
        action = torch.mm(action, action.T)  # shape: [num_planets, num_planets]
        #softmax along rows to get probabilities of actions for each planet
        action = F.softmax(action, dim=1)

        #mask diagnols
        diag_mask = torch.eye(action.shape[0])
        action = action * (1 - diag_mask)
        
        if mask is not None:
            action = action * mask


        return action


# ============================================================
# Minimal sb3 Compatibility Adapters (added for sb3 integration)
# ============================================================

class FlatStateToMatrixAdapter:
    """
    Utility adapter to convert flat states (from sb3) to matrix format for networks.
    Used internally by the policy to reshape observations.
    """
    def __init__(self, max_planets: int, max_fleets: int, state_dim: int):
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        self.state_dim = state_dim
    
    @staticmethod
    def reshape_flat_to_matrix(flat_state: torch.Tensor, max_planets: int, 
                               max_fleets: int, state_dim: int) -> torch.Tensor:
        """Convert flat state [batch_size, total_features] to matrix [batch_size, max_planets+max_fleets, state_dim]."""
        batch_size = flat_state.shape[0] if flat_state.dim() > 1 else 1
        total_seq = max_planets + max_fleets
        matrix_state = flat_state.view(batch_size, total_seq, state_dim)
        return matrix_state
    
    @staticmethod
    def reshape_matrix_to_flat(matrix_state: torch.Tensor) -> torch.Tensor:
        """Convert matrix state back to flat format."""
        batch_size = matrix_state.shape[0]
        flat_state = matrix_state.view(batch_size, -1)
        return flat_state


class NetworkCompatibilityHelper:
    """
    Helper class to ensure custom networks are compatible with sb3's training loop.
    Provides utility methods for device placement, gradient management, etc.
    """
    
    @staticmethod
    def ensure_batch_dimension(tensor: torch.Tensor, expected_batch_size: int = None) -> torch.Tensor:
        """Ensure tensor has batch dimension."""
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor
    
    @staticmethod
    def prepare_state_for_network(state: torch.Tensor, max_planets: int, max_fleets: int, 
                                  state_dim: int, device: torch.device) -> torch.Tensor:
        """
        Prepare state tensor for network forward pass.
        Handles reshaping from flat sb3 format to matrix format expected by networks.
        """
        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Ensure on correct device
        state = state.to(device)
        
        # Reshape to matrix format if needed
        total_size = (max_planets + max_fleets) * state_dim
        if state.shape[-1] == total_size:
            state = state.view(state.shape[0], max_planets + max_fleets, state_dim)
        
        return state
    
    @staticmethod
    def prepare_action_for_network(action: torch.Tensor, max_planets: int, 
                                   action_dim: int, device: torch.device) -> torch.Tensor:
        """Prepare action tensor for network forward pass."""
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        action = action.to(device)
        
        # Reshape to matrix format if needed
        total_size = max_planets * action_dim
        if action.shape[-1] == total_size:
            action = action.view(action.shape[0], max_planets, action_dim)
        
        return action
