import torch
import numpy as np
import gymnasium as gym
import gym_pusht

from .agent import PPOAgent
from .buffer import PPOVectorBuffer



def extract_image(obs):
    """
    Push-T gibt oft ein Dictionary zurück. 
    Passe diesen Key ('image', 'pixels', etc.) an eure konkrete Gym-Umgebung an.
    Erwarteter Shape für PyTorch: (Channels, Height, Width)
    """
    if isinstance(obs, dict):
        img = obs['image'] 
        if img.shape[-1] == 3: 
            img = np.transpose(img, (2, 0, 1))
        return img
    else:
        if obs.shape[-1] == 3:
            obs = np.transpose(obs, (2, 0, 1))
        return obs

def train_pusht():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training startet auf: {device}")

    config = {
        "total_timesteps": 500_000,
        "rollout_steps": 2048,
        "batch_size": 128,
        "ppo_epochs": 10,
        "learning_rate": 3e-4,
        "clip_coef": 0.2,       
        "ent_coef": 0.01,       
        "gamma": 0.99,          
        "gae_lambda": 0.95      }
    env = gym.make("gym_pusht/PushT-v0", obs_type="state") 
    
    state, _ = env.reset()
    
    state_dim = state.shape[0]             
    action_dim = env.action_space.shape[0] 

    rollout_steps = config["rollout_steps"]
    num_updates = config["total_timesteps"] // rollout_steps

    agent = PPOAgent(
        state_dim=state_dim, 
        action_dim=action_dim, 
        lr=config["learning_rate"],
        clip_coef=config["clip_coef"],     
        ent_coef=config["ent_coef"]        
    )
    buffer = PPOVectorBuffer(buffer_size=rollout_steps, state_dim=state_dim, action_dim=action_dim, device=device)

    state, _ = env.reset()

    for update in range(num_updates):
        buffer.clear()
        
        #  Phase A: Rollouts
        for step in range(rollout_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action, logprob, _, value = agent.network.get_action_and_value(state_tensor)
            
            cpu_action = action.cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = env.step(cpu_action)
            done = terminated or truncated
            
            buffer.store(state, action, logprob, reward, done, value)
            
            state = next_state
            
            if done:
                state, _ = env.reset()
                
        #  Phase B: Bootstrapping 
        with torch.no_grad():
            next_state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            next_value = agent.network.get_value(next_state_tensor).item()
            
        advantages, returns = buffer.compute_returns_and_advantages(next_value, done, gamma=config["gamma"],
            gae_lambda=config["gae_lambda"]
        )
        
        #  Phase C: PPO Update 
        agent.update(
            buffer, 
            advantages, 
            returns, 
            batch_size=config["batch_size"], 
            ppo_epochs=config["ppo_epochs"]
        )
        
        #  Phase D: Logging 
        avg_reward = buffer.rewards.sum().item() / (buffer.dones.sum().item() + 1e-5) 
        print(f"Update {update+1}/{num_updates} | Mean Reward pro Ep: {avg_reward:.2f}")

    print("Training finished!")
    env.close()

if __name__ == "__main__":
    train_pusht()