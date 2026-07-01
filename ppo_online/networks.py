import torch
import torch.nn as nn
from torch.distributions import Normal

class VectorActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def get_action_and_value(self, state_vector, action=None):
        mean = self.actor(state_vector)
        std = self.log_std.exp().expand_as(mean)
        distribution = Normal(mean, std)
        
        if action is None:
            action = distribution.sample()
            
        value = self.critic(state_vector)
        return action, distribution.log_prob(action).sum(axis=-1), distribution.entropy().sum(axis=-1), value

    def get_value(self, state_vector):
        return self.critic(state_vector)