import torch

class PPOVectorBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, device):
        self.device = device
        self.max_size = buffer_size
        self.ptr = 0
        
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32).to(device)
        self.logprobs = torch.zeros(buffer_size, dtype=torch.float32).to(device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32).to(device)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32).to(device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32).to(device)

    def store(self, state, action, logprob, reward, done, value):
        self.states[self.ptr] = torch.tensor(state, dtype=torch.float32).to(self.device)
        self.actions[self.ptr] = action.detach()
        self.logprobs[self.ptr] = logprob.detach()
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value.detach()
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