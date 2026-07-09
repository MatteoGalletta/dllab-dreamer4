import torch

class PPOVectorBuffer:
    def __init__(self, buffer_size, num_envs, state_shape, action_dim, device):
        self.device = device
        self.max_size = buffer_size
        self.num_envs = num_envs
        self.ptr = 0
        
        state_shape = tuple(state_shape) if isinstance(state_shape, (tuple, list)) else (state_shape,)
        self.states = torch.zeros((buffer_size, num_envs, *state_shape), dtype=torch.float32).to(device)
        self.actions = torch.zeros((buffer_size, num_envs, action_dim), dtype=torch.float32).to(device)
        self.logprobs = torch.zeros((buffer_size, num_envs), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((buffer_size, num_envs), dtype=torch.float32).to(device)
        self.dones = torch.zeros((buffer_size, num_envs), dtype=torch.float32).to(device)
        self.values = torch.zeros((buffer_size, num_envs), dtype=torch.float32).to(device)

    def store(self, states, actions, logprobs, rewards, dones, values):
        # Store vectorized rollout slices directly on the target device.
        self.states[self.ptr] = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = actions.detach()
        self.logprobs[self.ptr] = logprobs.detach()
        self.rewards[self.ptr] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        self.values[self.ptr] = values.detach().squeeze(-1)
        self.ptr += 1


    def clear(self):
        self.ptr = 0

    def compute_returns_and_advantages(self, next_value, next_done, gamma=0.99, gae_lambda=0.95):
        advantages = torch.zeros_like(self.rewards)
        lastgaelam = 0
        
        for t in reversed(range(self.max_size)):
            if t == self.max_size - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
                
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            
        returns = advantages + self.values
        
        return advantages.to(self.device), returns.to(self.device)
