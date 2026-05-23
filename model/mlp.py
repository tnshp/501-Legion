import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_Q(nn.Module):
    """Q(s, a) -> scalar for flat continuous state/action spaces."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # state: [B, state_dim]  action: [B, action_dim]
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)  # [B]


class MLP_V(nn.Module):
    """V(s) -> scalar for flat continuous state spaces."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)  # [B]


class MLP_Policy(nn.Module):
    """
    Diagonal Gaussian policy for flat continuous action spaces.

    Implements the same interface as the transformer P_network:
        forward(state) -> (mu, sigma)
        sample(state)  -> (action, log_prob)   ← required by SACTrainer
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor):
        h = self.trunk(state)
        mu = self.mu_head(h)
        sigma = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
        return mu, sigma  # [B, action_dim], [B, action_dim]

    def sample(self, state: torch.Tensor):
        """
        Reparameterised sample + log π(a|s).

        Returns:
            action   : [B, action_dim]
            log_prob : [B, 1]
        """
        mu, sigma = self.forward(state)
        eps = torch.randn_like(sigma)
        action = mu + eps * sigma
        log_prob = (
            -0.5 * ((action - mu) / sigma) ** 2
            - sigma.log()
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1, keepdim=True)  # [B, 1]
        return action, log_prob
