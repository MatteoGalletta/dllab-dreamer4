import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .networks import VectorActorCritic

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, clip_coef=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = VectorActorCritic(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def update(self, buffer, advantages, returns, batch_size=64, ppo_epochs=10):
        b_actions = buffer.actions.to(self.device)
        b_logprobs = buffer.logprobs.to(self.device)
        
        inds = np.arange(buffer.max_size)
        
        for epoch in range(ppo_epochs):
            np.random.shuffle(inds)
            for start in range(0, buffer.max_size, batch_size):
                end = start + batch_size
                mb_inds = inds[start:end]
                
                b_states = buffer.states[mb_inds].to(self.device).float() 
                
                mb_actions = b_actions[mb_inds]
                mb_logprobs = b_logprobs[mb_inds]
                mb_advantages = advantages[mb_inds]
                mb_returns = returns[mb_inds]
                
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                _, new_logprob, entropy, new_value = self.network.get_action_and_value(b_states, mb_actions)
                
                logratio = new_logprob - mb_logprobs
                ratio = logratio.exp()
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                actor_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                critic_loss = 0.5 * ((new_value.view(-1) - mb_returns) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()