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
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor):
        """Returns pre-tanh Gaussian parameters (mu, sigma) in latent space."""
        h = self.trunk(state)
        mu = self.mu_head(h)
        sigma = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
        return mu, sigma  # [B, action_dim], [B, action_dim]

    def sample(self, state: torch.Tensor):
        """
        Reparameterised sample with tanh squashing and corrected log π(a|s).

        Without squashing, actions go outside the env's [-1,1] bounds and are
        silently clipped — creating a Q/reward mismatch that corrupts training.

        The log-prob correction for the tanh Jacobian is:
            log π(a) = log N(u; μ, σ) − Σ log(1 − tanh(u)² + ε)

        Returns:
            action   : [B, action_dim]  in (-1, 1) via tanh
            log_prob : [B, 1]           with tanh Jacobian correction
        """
        mu, sigma = self.forward(state)
        eps = torch.randn_like(sigma)
        u = mu + eps * sigma                   # pre-tanh sample
        action = torch.tanh(u)                 # squashed to (-1, 1)

        # Gaussian log-prob of the pre-tanh sample
        log_prob_gaussian = (
            -0.5 * ((u - mu) / sigma) ** 2
            - sigma.log()
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1, keepdim=True)            # [B, 1]

        # Subtract log-Jacobian of tanh: log(1 - tanh(u)^2) = log(1 - a^2)
        log_prob = log_prob_gaussian - torch.log(1 - action.pow(2) + 1e-6).sum(
            dim=-1, keepdim=True
        )                                       # [B, 1]
        return action, log_prob

    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """Greedy action for evaluation: tanh(μ), no noise."""
        mu, _ = self.forward(state)
        return torch.tanh(mu)
