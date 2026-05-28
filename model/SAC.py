import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np 

from utils.pos_encoding import get_pos_encoding

class Encoder:
    # planets - [id, owner, x, y, radius, ships, production]
    # out - [owner[4], radius, ships, production, moving/static, angular_velocity, comet, is_planet, x, y, time_step]
    # fleets - [id, owner, x, y, angle, from_planet_id, ships] or empty array
    # out - [owner[4], angle, ships, speed, dummy[3], is_planet, x, y, time_step] or empty array
    
    def __init__(self, max_planets=40, max_fleets=100, max_comet_planet_ids=10, max_speed=6.0):  
        self.max_planets = max_planets
        self.max_fleets = max_fleets 
        self.max_comet_planet_ids = max_comet_planet_ids
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

    def encode(self, planets, fleets, initial_planets, angular_velocity, comet_planet_ids, time_step, apply_padding=True):
       
        
        #assert the shape of planets
        assert planets.shape[-1] == 7, f"Expected planets to have 7 features, but got {planets.shape[-1]}"
        
        # Handle empty fleets array
        if fleets.size == 0:
            fleets = np.empty((0, 7))
        else:
            assert fleets.shape[-1] == 7, f"Expected fleets to have 7 features, but got {fleets.shape[-1]}"

        planets_owner = planets[:, 1].astype(int)
        planet_pos = planets[:, 2:4]
        
        angular_velocity = np.array([angular_velocity for _ in range(planets.shape[0])]).reshape(-1, 1)
        
        # Handle empty comet_planet_ids
        if comet_planet_ids.size == 0:
            comet_planet = np.zeros((planets.shape[0], 1))
        else:
            comet_planet = np.array([1 if planet_id in comet_planet_ids else 0 for planet_id in planets[:, 0]]).reshape(-1, 1)
        
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
                             comet_planet, is_planet, planet_pos,  times_step_array))
        
        # Handle empty fleets
        if fleets.shape[0] == 0:
            # Create empty fleets_encoded with correct feature dimension (14 features)
            fleets_encoded = np.empty((0, 14))
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

            dummy = np.zeros((fleets.shape[0], 3))
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
                     batched_comet_planet_ids, batch_time_steps=None, apply_padding=True):
        """
        Encode a batch of states.
        
        Args:
            batched_planets: list of planet arrays, each [n_planets, 7]
            batched_fleets: list of fleet arrays, each [n_fleets, 7] or empty
            batched_initial_planets: list of initial planet arrays, each [n_initial, 7]
            batched_angular_velocity: array or list of angular velocities, shape [batch_size]
            batched_comet_planet_ids: list of comet planet id arrays or empty
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
                batched_comet_planet_ids[i],
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


    
class Q_network(nn.Module):
    def __init__(self, 
                 state_dim=14, 
                 action_dim=8,
                 max_planets=40,
                 max_fleets=100, 
                 d_model=128, 
                 nhead=4, 
                 num_layers=3,  
                 dim_feedforward=2048, 
                 dropout=0.1):
        
        super(Q_network, self).__init__()
        self.max_planets = max_planets
        self.max_fleets = max_fleets

        self.P = nn.Linear(state_dim - 4, d_model)
        self.F = nn.Linear(state_dim - 4, d_model)
        self.A = nn.Linear(action_dim, d_model)

        self.value_head = nn.Linear(d_model, 1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, d_model))

        #transformer - encoder only
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Recommended for easier tensor handling
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, 
                                                 num_layers=num_layers, 
        )
        
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
        planets = self.P(state[:, :, :10] * planets_mask.unsqueeze(-1))     
        fleets = self.F(state[:, :, :10] * fleets_mask.unsqueeze(-1))

        time_step = state[:, :, -1]  # Assuming time step is the last feature of the first token (planet)
        pos = state[:, :, 11:13]  # Assuming position is at indices 11 and 12
        pos_encoding = get_pos_encoding(pos, time_step, self.d_model)

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
        src = torch.cat((cls_token, state,  sep_token, action), dim=1)

        out_ = self.transformer(src)
        out_cls = out_[:, 0, :]  # [batch_size, 1, d_model]
        value = self.value_head(out_cls)  # [batch_size, 1]
        
        return value.squeeze(-1)  # [batch_size]
    

class V_network(nn.Module):
    def __init__(self, 
                 state_dim=14,
                 max_planets=40,
                 max_fleets=100,
                 d_model=128, 
                 nhead=4, 
                 num_layers=2,  
                 dim_feedforward=128, 
                 dropout=0.1):
        
        super(V_network, self).__init__()
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        
        self.P = nn.Linear(state_dim - 4, d_model)
        self.F = nn.Linear(state_dim - 4, d_model)

        self.value_head = nn.Linear(d_model, 1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        #transformer - encoder only
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Recommended for easier tensor handling
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, 
                                                 num_layers=num_layers, 
        )
        self.d_model = d_model

    def forward(self, state):  
        
        planets_mask = torch.zeros(state.shape[0], state.shape[1], device=state.device)
        planets_mask[:, :self.max_planets] = 1

        fleets_mask = torch.zeros(state.shape[0], state.shape[1], device=state.device)
        fleets_mask[:, self.max_planets:self.max_planets + self.max_fleets] = 1

        planets = self.P(state[:, :, :10] * planets_mask.unsqueeze(-1))     
        fleets = self.F(state[:, :, :10] * fleets_mask.unsqueeze(-1))

        time_step = state[:, :, -1]  # Assuming time step is the last feature of the first token (planet)
        pos = state[:, :, 11:13]  # Assuming position is at indices 11 and 12
        pos_encoding = get_pos_encoding(pos, time_step, self.d_model)

        state = planets + fleets
        state = state + pos_encoding
    
        # Expand cls token for batch
        batch_size = state.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1,  -1)  # [1, batch_size, d_model]

        src = torch.cat((cls_token, state), dim=1)
        out_ = self.transformer(src)
        out_cls = out_[:, 0, :]  # [batch_size, 1, d_model]
        value = self.value_head(out_cls)  # [batch_size, 1]
        
        return value.squeeze(-1)  # [batch_size]
    

class P_network(nn.Module):
    def __init__(self, 
                 d_model=128, 
                 state_dim=14,
                 action_dim=8,
                 max_planets=40,
                 max_fleets=100, 
                 nhead=4, 
                 num_layers=3,  
                 dim_feedforward=128, 
                 dropout=0.1):
        
        super(P_network, self).__init__()

        self.max_planets = max_planets
        self.max_fleets = max_fleets

        self.P = nn.Linear(state_dim - 4, d_model)
        self.F = nn.Linear(state_dim - 4, d_model)

        self.mu_head = nn.Linear(d_model, action_dim)
        self.sigma_head = nn.Linear(d_model, action_dim)

        #transformer - encoder only
        encoder_layer = nn.TransformerEncoderLayer(
                            d_model=d_model, 
                            nhead=nhead, 
                            dim_feedforward=dim_feedforward,
                            dropout=dropout,
                            activation='relu',
                            batch_first=True  # Recommended for easier tensor handling
                        )
        self.transformer = nn.TransformerEncoder(encoder_layer, 
                                                 num_layers=num_layers, 
                        )
        
        self.d_model = d_model
        self.action_dim = action_dim

    def forward(self, state):
        planets_mask = torch.zeros(state.shape[0], state.shape[1], device=state.device)
        planets_mask[:, :self.max_planets] = 1

        fleets_mask = torch.zeros(state.shape[0], state.shape[1], device=state.device)
        fleets_mask[:, self.max_planets:self.max_planets + self.max_fleets] = 1

        planets = self.P(state[:, :, :10] * planets_mask.unsqueeze(-1))
        fleets = self.F(state[:, :, :10] * fleets_mask.unsqueeze(-1))

        time_step = state[:, :, -1]  # Assuming time step is the last feature of the first token (planet)
        pos = state[:, :, 11:13]  # Assuming position is at indices 11 and 12
        pos_encoding = get_pos_encoding(pos, time_step, self.d_model)

        state = planets + fleets
        state = state + pos_encoding

        # Process through transformer
        src = state  # [batch_size, seq_len, d_model]
        out_ = self.transformer(src)  # [batch_size, seq_len, d_model]

        #only pass num planets tokens through heads, as action is only for planets
        mu = self.mu_head(out_[:, :self.max_planets, :])
        sigma = F.softplus(self.sigma_head(out_[:, :self.max_planets, :])) + 1e-5

        return mu, sigma

    def sample(self, state):
        """
        Sample action via reparameterisation and return log π(a|s).

        Returns:
            action   : [B, max_planets, action_dim]
            log_prob : [B, 1]
        """
        mu, sigma = self.forward(state)
        eps = torch.randn_like(sigma)
        action = mu + eps * sigma
        log_prob = (
            -0.5 * ((action - mu) / sigma) ** 2
            - sigma.log()
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=(-2, -1), keepdim=True).squeeze(-1)  # [B, 1]
        return action, log_prob
    
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
