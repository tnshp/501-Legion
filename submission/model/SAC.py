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

class Encoder:
    # planets - [id, owner, x, y, radius, ships, production]
    # out - [owner[4], radius, ships, production, moving/static, angular_velocity, is_planet, x, y, time_step]
    # fleets - [id, owner, x, y, angle, from_planet_id, ships] or empty array
    # out - [owner[4], angle, ships, speed, dummy[2], is_planet, x, y, time_step] or empty array
    # (Comets are stripped from the observation upstream in env._obs_to_arrays, so
    #  there is no comet feature here — state width is 13.)

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

    def encode(self, planets, fleets, initial_planets, angular_velocity, time_step, apply_padding=True):


        #assert the shape of planets
        assert planets.shape[-1] == 7, f"Expected planets to have 7 features, but got {planets.shape[-1]}"

        # Handle empty fleets array
        if fleets.size == 0:
            fleets = np.empty((0, 7))
        else:
            assert fleets.shape[-1] == 7, f"Expected fleets to have 7 features, but got {fleets.shape[-1]}"
            # Sort by ship count descending so truncation to max_fleets keeps the largest fleets.
            fleets = fleets[np.argsort(fleets[:, 6])[::-1]]

        planets_owner = planets[:, 1].astype(int)
        planet_pos = planets[:, 2:4]

        angular_velocity = np.array([angular_velocity for _ in range(planets.shape[0])]).reshape(-1, 1)

        planets_mapping = self.get_planet_mapping(initial_planets)
        moving_planets = np.array([planets_mapping.get(planet_id, 0) for planet_id in planets[:, 0]]).reshape(-1, 1)
        #replace nan with 0
        moving_planets = np.nan_to_num(moving_planets)

        planets_owner_one_hot = np.zeros((planets_owner.shape[0], 4))
        for i in range(planets_owner.shape[0]):
            if planets_owner[i] != -1:
                planets_owner_one_hot[i] = np.eye(4)[planets_owner[i]]
        
        #adding time step as feature to planets and fleets
        times_step_array = np.array([time_step for _ in range(planets.shape[0])]).reshape(-1, 1)
        is_planet = np.zeros((planets.shape[0], 1))
        planets_encoded = np.hstack((planets_owner_one_hot, planets[:, 4:], moving_planets, angular_velocity,
                             is_planet, planet_pos,  times_step_array))

        # Handle empty fleets
        if fleets.shape[0] == 0:
            # Create empty fleets_encoded with correct feature dimension (13 features)
            fleets_encoded = np.empty((0, 13))
        else:
            #fleets
            fleets_owner = fleets[:, 1].astype(int)
            fleets_angle = fleets[:, 4].reshape(-1, 1)
            # fleets_from_planet_id = fleets[:, 5].reshape(-1, 1)
            fleets_ships = fleets[:, 6].reshape(-1, 1)
            fleets_speed = np.array([self.get_speed(ships) for ships in fleets_ships[:, 0]]).reshape(-1, 1)

            fleets_owner_one_hot = np.zeros((fleets_owner.shape[0], 4))
            for i in range(fleets_owner.shape[0]):
                if fleets_owner[i] != -1:
                    fleets_owner_one_hot[i] = np.eye(4)[fleets_owner[i]]

            dummy = np.zeros((fleets.shape[0], 2))
            is_fleet = np.ones((fleets.shape[0], 1))
            fleet_pos = fleets[:, 2:4]
            times_step_array_fleet = np.array([time_step for _ in range(fleets.shape[0])]).reshape(-1, 1)

            fleets_encoded = np.hstack((fleets_owner_one_hot, fleets_angle, fleets_ships, fleets_speed, dummy,
                                is_fleet,  fleet_pos, times_step_array_fleet))

        # Apply padding if requested
        if apply_padding:
            planets_encoded, planets_mask = self._pad_array(planets_encoded, self.max_planets)
            fleets_encoded, fleets_mask = self._pad_array(fleets_encoded, self.max_fleets)
            
            # Combine masks
            mask = np.concatenate([planets_mask, fleets_mask])
            
            state = np.vstack((planets_encoded, fleets_encoded))
            return state, mask
        else:
            state = np.vstack((planets_encoded, fleets_encoded))
            return state, None
    
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


    
_META_DIM = 12  # 4 players × [planet_count, production, ships]


def _aggregate_meta_features(state):
    """Per-player global summary derived from an encoded state batch.

    A learned projection of this vector is prepended to the transformer as a
    single "metadata" token, so every token can read the overall picture
    (how many planets / how much production / how many ships each player holds)
    directly, instead of having to reconstruct it by attending over the
    individual planet/fleet tokens.

    Parameters
    ----------
    state : [B, seq, 13] — encoder layout: owner one-hot at 0:4, ships at 5,
            production at 6 (planets only; fleets carry speed there), is_fleet
            flag at 9. Neutral planets and padding rows have an all-zero owner
            one-hot, so they contribute to no player's totals.

    Returns
    -------
    [B, 12] — concat over players 0..3 of [planet_count, log1p(production),
    log1p(ships)]. After the perspective swap player 0 is our agent and 1..3 are
    opponents; players absent in a 2-player game are simply zero. Ships include
    in-flight fleets, so it is the player's *total* ship strength. Ship and
    production totals are log1p-compressed (they reach into the thousands).
    """
    owner     = state[:, :, 0:4]                 # [B, seq, 4] one-hot (neutral/pad = 0)
    ships     = state[:, :, 5:6]                 # [B, seq, 1]
    prod      = state[:, :, 6:7]                 # [B, seq, 1] (planets only)
    is_fleet  = state[:, :, 9:10]                # [B, seq, 1] 1 = fleet, 0 = planet/pad
    is_planet = 1.0 - is_fleet

    planet_cnt = (owner * is_planet).sum(dim=1)              # [B, 4]
    ships_tot  = (owner * ships).sum(dim=1)                  # [B, 4] (planets + fleets)
    prod_tot   = (owner * prod * is_planet).sum(dim=1)       # [B, 4]

    ships_tot = torch.log1p(ships_tot.clamp(min=0.0))
    prod_tot  = torch.log1p(prod_tot.clamp(min=0.0))
    return torch.cat([planet_cnt, prod_tot, ships_tot], dim=-1)  # [B, 12]


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
                 state_dim=13,
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

        self.ship_encoder  = LearnedFourierScalarEncoding()
        self.angle_encoder = FourierAngleEncoding()
        # +ship & angle fourier dims, -1 for the raw ship slot ship_encoder replaces
        _proj_in = ((state_dim - 4) - 1
                    + self.ship_encoder.out_dim
                    + self.angle_encoder.out_dim)
        self.P = nn.Linear(_proj_in, d_model)
        self.F = nn.Linear(_proj_in, d_model)
        self.A = nn.Linear(action_dim, d_model)

        self.pos_encoder = LearnedFourierPosEncoding(d_model)

        # Projects the per-player aggregate summary into a single "metadata" token
        # (see _aggregate_meta_features) that is prepended to the transformer.
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
            state: [batch_size, state_seq_len, d_model] or [state_seq_len, d_model]
            action: [batch_size, action_seq_len, d_model] or [action_seq_len, d_model]
        
        Returns:
            value: [batch_size] or [1] (scalar if unbatched)
        """
        #true for seq length upto max_planets
        planets_mask = torch.zeros(state.shape[0], state.shape[1], device=state.device)
        planets_mask[:, :self.max_planets] = 1

        fleets_mask = torch.zeros(state.shape[0], state.shape[1], device=state.device)
        fleets_mask[:, self.max_planets:self.max_planets + self.max_fleets] = 1

        # print(planets_mask[0]) 
        # print(fleets_mask[0])

        # print(f"planets mask shape {planets_mask.shape}, fleets mask shape {fleets_mask.shape}")
        #Usable dimension upto 10 
        # Replace the raw ship count (index 5) with a log-normalized + learnable
        # Fourier encoding so the model can finely resolve ship magnitudes and the
        # ratios that decide a capture (see LearnedFourierScalarEncoding).
        ship_block  = self.ship_encoder(state[:, :, 5:6])
        # Fleet heading (index 4 = angle): periodic Fourier code so the model can
        # judge precisely whether a fleet's trajectory will strike a target planet.
        # (For planets index 4 is radius; self.P simply learns to downweight it.)
        angle_block = self.angle_encoder(state[:, :, 4:5])
        token_feats = torch.cat([state[:, :, :5], state[:, :, 6:9],
                                 ship_block, angle_block], dim=-1)
        planets = self.P(token_feats * planets_mask.unsqueeze(-1))
        fleets  = self.F(token_feats * fleets_mask.unsqueeze(-1))

        time_step = state[:, :, -1]  # Assuming time step is the last feature of the first token (planet)
        pos = state[:, :, 10:12]  # position is at indices 10 and 11 (13-wide state)
        pos_encoding = self.pos_encoder(pos, time_step)

        # Global summary token, built from the raw input state (before it is
        # overwritten below) and prepended to the transformer sequence.
        meta_token = self.meta_proj(_aggregate_meta_features(state)).unsqueeze(1)

        action = self.A(action)

        state = planets + fleets
        state = state + pos_encoding

        batch_size = state.shape[0]
        # state_n = state.shape[1]
        # action_n = action.shape[1]

        # Expand cls and sep tokens for batch
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # [1, batch_size, d_model]
        sep_token = self.sep_token.expand(batch_size, -1, -1)  # [1, batch_size, d_model]

        # Concatenate: [batch_size, seq_len, d_model]
        src = torch.cat((cls_token, meta_token, state,  sep_token, action), dim=1)

        out_ = self.transformer(src)
        out_cls = out_[:, 0, :]  # [batch_size, 1, d_model]
        value = self.value_head(out_cls)  # [batch_size, 1]
        
        return value.squeeze(-1)  # [batch_size]
    

class V_network(nn.Module):
    def __init__(self, 
                 state_dim=13,
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
        self.angle_encoder = FourierAngleEncoding()
        # +ship & angle fourier dims, -1 for the raw ship slot ship_encoder replaces
        _proj_in = ((state_dim - 4) - 1
                    + self.ship_encoder.out_dim
                    + self.angle_encoder.out_dim)
        self.P = nn.Linear(_proj_in, d_model)
        self.F = nn.Linear(_proj_in, d_model)

        self.pos_encoder = LearnedFourierPosEncoding(d_model)

        # Projects the per-player aggregate summary into a single "metadata" token
        # (see _aggregate_meta_features) that is prepended to the transformer.
        self.meta_proj = nn.Linear(_META_DIM, d_model)

        self.value_head = nn.Linear(d_model, 1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        #transformer - encoder only (attn_impl="flash" routes attention through SDPA)
        self.transformer = _build_encoder(
            d_model, nhead, num_layers, dim_feedforward, dropout, attn_impl)
        self.d_model = d_model

    def forward(self, state):
        
        planets_mask = torch.zeros(state.shape[0], state.shape[1], device=state.device)
        planets_mask[:, :self.max_planets] = 1

        fleets_mask = torch.zeros(state.shape[0], state.shape[1], device=state.device)
        fleets_mask[:, self.max_planets:self.max_planets + self.max_fleets] = 1

        # Replace the raw ship count (index 5) with a log-normalized + learnable
        # Fourier encoding so the model can finely resolve ship magnitudes and the
        # ratios that decide a capture (see LearnedFourierScalarEncoding).
        ship_block  = self.ship_encoder(state[:, :, 5:6])
        # Fleet heading (index 4 = angle): periodic Fourier code so the model can
        # judge precisely whether a fleet's trajectory will strike a target planet.
        # (For planets index 4 is radius; self.P simply learns to downweight it.)
        angle_block = self.angle_encoder(state[:, :, 4:5])
        token_feats = torch.cat([state[:, :, :5], state[:, :, 6:9],
                                 ship_block, angle_block], dim=-1)
        planets = self.P(token_feats * planets_mask.unsqueeze(-1))
        fleets  = self.F(token_feats * fleets_mask.unsqueeze(-1))

        time_step = state[:, :, -1]  # Assuming time step is the last feature of the first token (planet)
        pos = state[:, :, 10:12]  # position is at indices 10 and 11 (13-wide state)
        pos_encoding = self.pos_encoder(pos, time_step)

        # Global summary token, built from the raw input state (before it is
        # overwritten below) and prepended to the transformer sequence.
        meta_token = self.meta_proj(_aggregate_meta_features(state)).unsqueeze(1)

        state = planets + fleets
        state = state + pos_encoding
    
        # Expand cls token for batch
        batch_size = state.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1,  -1)  # [1, batch_size, d_model]

        src = torch.cat((cls_token, meta_token, state), dim=1)
        out_ = self.transformer(src)
        out_cls = out_[:, 0, :]  # [batch_size, 1, d_model]
        value = self.value_head(out_cls)  # [batch_size, 1]
        
        return value.squeeze(-1)  # [batch_size]
    

class P_network(nn.Module):
    def __init__(self,
                 d_model=128,
                 state_dim=13,
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
        self.angle_encoder = FourierAngleEncoding()
        # +ship & angle fourier dims, -1 for the raw ship slot ship_encoder replaces
        _proj_in = ((state_dim - 4) - 1
                    + self.ship_encoder.out_dim
                    + self.angle_encoder.out_dim)
        self.P = nn.Linear(_proj_in, d_model)
        self.F = nn.Linear(_proj_in, d_model)

        self.pos_encoder = LearnedFourierPosEncoding(d_model)

        # Projects the per-player aggregate summary into a single "metadata" token
        # (see _aggregate_meta_features) that is prepended to the transformer.
        self.meta_proj = nn.Linear(_META_DIM, d_model)

        self.mu_head = nn.Linear(d_model, action_dim)
        self.sigma_head = nn.Linear(d_model, action_dim)

        #transformer - encoder only (attn_impl="flash" routes attention through SDPA)
        self.transformer = _build_encoder(
            d_model, nhead, num_layers, dim_feedforward, dropout, attn_impl)

        self.d_model = d_model
        self.action_dim = action_dim

    def forward(self, state):
        planets_mask = torch.zeros(state.shape[0], state.shape[1], device=state.device)
        planets_mask[:, :self.max_planets] = 1

        fleets_mask = torch.zeros(state.shape[0], state.shape[1], device=state.device)
        fleets_mask[:, self.max_planets:self.max_planets + self.max_fleets] = 1

        # Replace the raw ship count (index 5) with a log-normalized + learnable
        # Fourier encoding so the model can finely resolve ship magnitudes and the
        # ratios that decide a capture (see LearnedFourierScalarEncoding).
        ship_block  = self.ship_encoder(state[:, :, 5:6])
        # Fleet heading (index 4 = angle): periodic Fourier code so the model can
        # judge precisely whether a fleet's trajectory will strike a target planet.
        # (For planets index 4 is radius; self.P simply learns to downweight it.)
        angle_block = self.angle_encoder(state[:, :, 4:5])
        token_feats = torch.cat([state[:, :, :5], state[:, :, 6:9],
                                 ship_block, angle_block], dim=-1)
        planets = self.P(token_feats * planets_mask.unsqueeze(-1))
        fleets  = self.F(token_feats * fleets_mask.unsqueeze(-1))

        time_step = state[:, :, -1]  # Assuming time step is the last feature of the first token (planet)
        pos = state[:, :, 10:12]  # position is at indices 10 and 11 (13-wide state)
        pos_encoding = self.pos_encoder(pos, time_step)

        # Global summary token, built from the raw input state (before it is
        # overwritten below) and prepended to the transformer sequence.
        meta_token = self.meta_proj(_aggregate_meta_features(state)).unsqueeze(1)

        state = planets + fleets
        state = state + pos_encoding

        # Process through transformer. The metadata token sits at position 0, so
        # the planet tokens (which drive the per-planet action heads) start at 1.
        src = torch.cat((meta_token, state), dim=1)  # [batch_size, 1 + seq_len, d_model]
        out_ = self.transformer(src)  # [batch_size, 1 + seq_len, d_model]

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
