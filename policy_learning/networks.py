"""Actor, critic and reward estimator extracted from the author's routing study."""
import torch
from torch import nn


class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Logits allow Categorical to calculate stable normalized log probabilities.
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(state)))))


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(state))))).squeeze(-1)


class RewardModel(nn.Module):
    """Optional supervised estimator for externally supplied ratings in [0, 1]."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        joined = torch.cat((state, action), dim=-1)
        return torch.sigmoid(self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(joined)))))).squeeze(-1)
