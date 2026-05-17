
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np 

from utils.pos_encoding import get_pos_encoding


class Encoder(nn.Module):
    def __init__(self, 
                 planet_dimension=10, 
                 fleet_dimension=7, 
                 d_model=128):  
        super(Encoder, self).__init__()
        self.P_ = nn.Linear(planet_dimension, d_model)
        self.F_ = nn.Linear(fleet_dimension, d_model)
        self.d_model = d_model  
    
    def convert(self, planets, fleets, initial_planets, angular_velocity, comet_planet_ids):
        # fleets - [id, owner, x, y, angle, from_planet_id, ships]
        # out - [owner, angle, from_planet_id, ships, speed]
        # planets - [id, owner, x, y, radius, ships, production]
        # out - [owner, radius, ships, production, moving/static, angular_velocity, comet]

        sun = np.array([50, 50])
        oribital_radius = ((initial_planets[:, 2:4] - sun)**2).sum(axis=1)**0.5
        planet_radius = initial_planets[:, 4]
        moving_planets = oribital_radius + planet_radius < 50
        
        #if moving planet then 1 else 0
        planet_ids = initial_planets[:, 0]
        moving_planets = moving_planets.astype(int)
        planets_mapping = {}
        for i, planet_id in enumerate(planet_ids):
            planets_mapping[int(planet_id)] = int(moving_planets[i])

        planets = np.hstack((planets, np.array([planets_mapping[int(planet_id)] for planet_id in planets[:, 0]]).reshape(-1, 1)))
        planets = np.hstack((planets, np.array([angular_velocity for _ in range(planets.shape[0])]).reshape(-1, 1)))
        planets = np.hstack((planets, np.array([1 if planet_id in comet_planet_ids else 0 for planet_id in planets[:, 0]]).reshape(-1, 1)))


        maxSpeed = 6.0

        #log base 10 
        fleets_speed = 1.0 + (maxSpeed - 1.0) * (np.log(fleets[:, -1]) / np.log(1000)) ** 1.5 

        #stack to fleets array
        fleets = np.hstack((fleets, fleets_speed.reshape(-1, 1)))
        
        return planets, fleets
    
    def forward(self,  planets, fleets, initial_planets, angular_velocity, comet_planet_ids, time_step):
        #shape: planets - [num_planets, 10], fleets - [num_fleets, 8]

        planets, fleets = self.convert(planets, fleets, initial_planets, angular_velocity, comet_planet_ids)
        
        planets = torch.tensor(planets, dtype=torch.float32)
        fleets = torch.tensor(fleets, dtype=torch.float32)
        
        #convert owner to one hot encoding
        planets_owner = planets[:, 1].long()
        planet_pos = planets[:, 2:4]

        #handle -1 case for unowned planets, make it [0,0,0,0] 
        planets_owner_one_hot = torch.zeros(planets_owner.shape[0], 4)
        for i in range(planets_owner.shape[0]):
            if planets_owner[i] != -1:
                planets_owner_one_hot[i] = F.one_hot(planets_owner[i], num_classes=4).float()

        planets = torch.cat((planets_owner_one_hot, planets[:, 4:]), dim=1)
        # planets = self.P(planets)
        
        fleets_owner = fleets[:, 1].long()
        fleet_pos = fleets[:, 2:4]
        fleets_owner_one_hot = torch.zeros(fleets_owner.shape[0], 4)
        for i in range(fleets_owner.shape[0]):
            if fleets_owner[i] != -1:
                fleets_owner_one_hot[i] = F.one_hot(fleets_owner[i], num_classes=4).float()

        #skipping from_planet_id, can be added later as separate features
        fleets = torch.cat((fleets_owner_one_hot, fleets[:, 4:5], fleets[:, 6:]), dim=1)

        planets = self.P_(planets)
        fleets = self.F_(fleets)

        planets = planets + get_pos_encoding(planet_pos, time_step, self.d_model)
        fleets = fleets + get_pos_encoding(fleet_pos, time_step, self.d_model)

        #combine cls_token, planets and fleets for transformer input
        state = torch.cat((planets, fleets), dim=0)
        return state
    

    
class Q_network(nn.Module):
    def __init__(self, 
                 d_model=128, 
                 
                 nhead=4, 
                 num_encoder_layers=2,  
                 dim_feedforward=128, 
                 dropout=0.1):
        
        super(Q_network, self).__init__()
        self.value_head = nn.Linear(d_model, 1)

        self.cls_token = nn.Parameter(torch.zeros(1, d_model))
        self.sep_token = nn.Parameter(torch.zeros(1, d_model))

        #transformer - encoder only
        self.transformer = nn.Transformer(d_model=d_model, 
                                          nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          dim_feedforward=dim_feedforward, 
                                          dropout=dropout, 
                                          activation='relu'
                                          )

        self.d_model = d_model
              
    def forward(self, state, action):  
        
        #tokens in state and action
        state_n = state.shape[0]
        action_n = action.shape[0]

        src = torch.cat((self.cls_token, state, self.sep_token, action), dim=0).unsqueeze(1)  # shape: [num_planets + num_fleets + 1, 1, d_model]
        out_ = self.transformer(src, src)
        out_cls = out_[0]  # shape: [d_model]
        value = self.value_head(out_cls)  # shape: [1]

        return value
    

class V_network(nn.Module):
    def __init__(self, 
                 d_model=128, 
                 nhead=4, 
                 num_encoder_layers=2,  
                 dim_feedforward=128, 
                 dropout=0.1):
        
        super(V_network, self).__init__()
        self.value_head = nn.Linear(d_model, 1)

        self.cls_token = nn.Parameter(torch.zeros(1, d_model))
        self.sep_token = nn.Parameter(torch.zeros(1, d_model))

        #transformer - encoder only
        self.transformer = nn.Transformer(d_model=d_model, 
                                          nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          dim_feedforward=dim_feedforward, 
                                          dropout=dropout, 
                                          activation='relu'
                                          )

        self.d_model = d_model

    def forward(self, state):  
        
        #tokens in state and action
        state_n = state.shape[0]
        
        src = torch.cat((self.cls_token, state), dim=0).unsqueeze(1)  # shape: [num_planets + num_fleets + 1, 1, d_model]
        out_ = self.transformer(src, src)
        out_cls = out_[0]  # shape: [d_model]
        value = self.value_head(out_cls)  # shape: [1]

        return value
    

class P_network(nn.Module):
    def __init__(self, 
                 d_model=128, 
                 action_dim=8, 
                 nhead=4, 
                 num_encoder_layers=2,  
                 dim_feedforward=128, 
                 dropout=0.1):
        
        super(P_network, self).__init__()

        self.mu_head = nn.Linear(d_model, action_dim)
        self.sigma_head = nn.Linear(d_model, action_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, d_model))
        self.sep_token = nn.Parameter(torch.zeros(1, d_model))

        #transformer - encoder only
        self.transformer = nn.Transformer(d_model=d_model, 
                                          nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          dim_feedforward=dim_feedforward, 
                                          dropout=dropout, 
                                          activation='relu'
                                          )

        self.d_model = d_model

    def forward(self, state):  
        
        #assuming state is only planets 
        state_n = state.shape[0]

        src = state  # shape: [num_planets + num_fleets + 1, 1, d_model]
        out_ = self.transformer(src, src)
        mu = self.mu_head(out_)  # shape: [num_planets, action_dim]
        sigma = F.softplus(self.sigma_head(out_)) + 1e-5  # shape: [num_planets, action_dim]
        
        return mu, sigma
    
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