import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, LayerNorm, global_add_pool


def make_graph(planets):
    planet_data = planets
    n = len(planet_data)

    # print(planet_data[0])
    # print()

    for x in planet_data:
        x.pop(4)

    # print(planet_data[0])

    # print(obs.planets[1])
    # print()
    # print(n)

    indices = torch.linspace(0,n-1,n, dtype = torch.int)
    pairs = torch.cartesian_prod(indices, indices)

    mask = pairs[:,0]!= pairs[:,1]
    edges = pairs[mask]
    x = torch.tensor(planet_data)


    data = Data(x=x, edge_index=edges.t().contiguous())
    return data

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs =  []
        self.vals = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size   

    def generate_batches(self):
        n = len(self.states)
        batch_start = np.arange(0, n, self.batch_size)
        indices = np.arange(n, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i: i+self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.probs),np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches
    

    
    def store_memory(self, s, p, v, r, done):
        self.states.append(s)      
        self.probs.append(p)     
        self.vals.append(v)    
        self.rewards.append(r)      
        self.dones.append(done)      
    
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class Agent(nn.Module):
    def __init__(self, num_features, device,  hidden_dims = 128, alpha = 0.0005, gamma=0.99,clip=0.2,critic_coef=0.5,entropy_coef=0.01, batch_size = 64, gae_lambda = 0.99, horizon = 2048, n_epochs = 10):
        super(Agent, self).__init__()
        self.num_features = num_features
        self.hidden_dims = hidden_dims

        self.conv_layer = GCNConv(self.num_features , self.hidden_dims)
        self.norm_layer = LayerNorm(self.hidden_dims)
        self.device = device

        self.conv_layer2 = GCNConv(self.hidden_dims , self.hidden_dims)
        self.norm_layer2 = LayerNorm(self.hidden_dims)

        self.actor_head = nn.Sequential(
            nn.Linear(self.hidden_dims * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(self.hidden_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.alpha = alpha
        self.gamma = gamma
        self.clip = clip
        self.critic_coef=critic_coef
        self.entropy_coef=entropy_coef
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.experience_memory = PPOMemory(batch_size=self.batch_size)
        self.horizon = horizon
        self.n_epochs = n_epochs

        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.alpha)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 100, gamma = self.gamma)


        self.to(self.device)

    def remember(self, s, p, v, r, done):
        self.experience_memory.store_memory(s, p, v, r, done)

    def forward(self, x, edge_index, non_graph_state_vars, ownership_mask):  #returns weights, log_probs, v
        x = self.conv_layer(x, edge_index)
        x = self.norm_layer(x)
        x = F.relu(x)

        x = self.conv_layer2(x, edge_index)
        x = self.norm_layer2(x)
        x = F.relu(x) # n x hidden_dims
        # global pooling

        batch = torch.zeros(x.size(0), dtype=torch.long, device = x.device)
        v = global_add_pool(x, batch)
        v = self.critic_head(v)
        v = v.squeeze()

        N = x.size(0)

        # ships = torch.tensor(planet_ship_data[:,0], dtype= torch.int16, device= self.device)
        # owned = torch.tensor(planet_ship_data[:,1], dtype= torch.bool, device= self.device)

        sources = x.unsqueeze(1).expand(N, N, -1)
        dests = x.unsqueeze(0).expand(N, N, -1)
        pairs = torch.cat([sources, dests], dim=-1) # nxnx(2*hidden_dims)
        
        scores = self.actor_head(pairs).squeeze(-1) # nxnx1 to nxn
        masked_scores = ownership_mask.unsqueeze(1),self.float() * scores
        

        alphas = F.softplus(masked_scores) + 1e-4
        dist = torch.distributions.Dirichlet(alphas)
        weights = dist.rsample()
        log_probs = dist.log_prob(weights)
        
        return weights, log_probs, v
    
        # pre_masked_actions = ships.unsqueeze(1) * F.softmax(scores, dim=1)
        
        # mask = owned.unsqueeze(1).float() # dim to nx1
        # actions = pre_masked_actions*mask # final dim = nxn
    
    def learn(self):
        states, old_probs_array, vals, rewards, dones, batches = self.experience_memory.generate_batches()
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)

        # v = v + α(δ) = v + α(v_p - v) = v + α(r + γ*v_f - v)
        advantages[n-1] = rewards[-1] - vals[-1]
        last_advantage = advantages[n-1]
        
        for i in reversed(range(n-1)): # 0 to n-2
            delta = rewards[i] + self.gamma * vals[i+1] * (1-int(dones[i+1])) - vals[i]
            advantages[i] = delta + self.gamma*self.gae_lambda*last_advantage
            last_advantage = advantages[i]
        
        returns = advantages - vals

        states_t = torch.tensor(states, dtype = torch.float32).to(device=self.device)
        probs_t = torch.tensor(old_probs_array, dtype = torch.float32).to(device=self.device)
        returns_t = torch.tensor(returns, dtype = torch.float32).to(device=self.device)

        for i in range(self.n_epochs):
           for batch in batches:
                state = states_t[batch]
                old_probs = probs_t[batch]
                batch_return = returns_t[batch]
                adv_batch = advantages[batch]
                adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)
                
                action_weights, log_probs, v = self.forward(state)
                
                ppo_ratio = torch.exp(log_probs/old_probs)
                clipped_ratio = torch.clamp(ppo_ratio, 1-self.clip, 1+self.clip)
                
                actor_loss = -torch.min(adv_batch*ppo_ratio, adv_batch*clipped_ratio)
                critic_loss = F.mse_loss(batch_return - v)
                total_loss = actor_loss + 0.5*critic_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        self.experience_memory.clear_memory()