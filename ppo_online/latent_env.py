import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path
import torch
import numpy as np

# 1. Dynamically find the root directory where your folders live.
# If latent_env.py is inside a subdirectory, parents[1] climbs up to the main project root.
project_root = Path(__file__).resolve().parents[1]
dreamer_src_path = os.path.join(project_root, "dreamer4-src")

# 2. Tell Python to look inside the dreamer4-src directory for modules
if dreamer_src_path not in sys.path:
    sys.path.append(dreamer_src_path)

# 3. Now Python can cleanly find the dreamer4 package and its model file!
from dreamer4.model import temporal_patchify, pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck

def pool_latents(z_unpacked: torch.Tensor) -> torch.Tensor:
    """Collapses spatial patches by taking their spatial token mean."""
    return z_unpacked.mean(dim=-2)


class LatentContextSampler:
    def __init__(self, npz_path: str, action_chunk_size: int, device: torch.device):
        self.device = device
        self.action_chunk_size = action_chunk_size
        
        data = np.load(npz_path, allow_pickle=True)
        self.images = data["images"]
        self.actions_raw = data["actions"]
        
        ends = data["episode_ends"]
        starts = np.zeros_like(ends)
        starts[1:] = ends[:-1]
        
        self.episodes = []
        for s, e in zip(starts, ends):
            raw_len = e - s
            seq_len = raw_len // action_chunk_size
            used = seq_len * action_chunk_size
            
            if seq_len >= 24:
                ep_imgs = self.images[s:s+used][::action_chunk_size]
                ep_acts = self.actions_raw[s:s+used].reshape(seq_len, -1)
                self.episodes.append((ep_imgs, ep_acts))
                
    def sample_context(self, batch_size: int, ctx_len: int = 24):
        batch_imgs, batch_acts = [], []
        indices = np.random.choice(len(self.episodes), size=batch_size, replace=True)
        
        for idx in indices:
            ep_imgs, ep_acts = self.episodes[idx]
            t_start = np.random.randint(0, len(ep_imgs) - ctx_len)
            
            batch_imgs.append(ep_imgs[t_start : t_start + ctx_len])
            batch_acts.append(ep_acts[t_start : t_start + ctx_len])
            
        imgs_tensor = torch.from_numpy(np.stack(batch_imgs)).permute(0, 1, 4, 2, 3).float() / 255.0
        acts_tensor = torch.from_numpy(np.stack(batch_acts)).float()
        
        return imgs_tensor.to(self.device), acts_tensor.to(self.device)


class LatentImaginationEnv:
    def __init__(self, dynamics, reward_head, encoder, frame_stack, packing_factor, device):
        self.dyn = dynamics
        self.reward_head = reward_head
        self.encoder = encoder
        
        self.frame_stack = frame_stack
        self.packing_factor = packing_factor
        self.device = device
        
        self.max_horizon = 10  # Hard capped dream length
        self.current_step = 0
        
        self.z_spatial_seq = None
        self.a_seq = None
        self.pooled_history = []

        self.dyn.eval()
        self.reward_head.eval()
        self.encoder.eval()

    @torch.no_grad()
    def reset(self, real_frames, real_actions):
        self.current_step = 0
        B, T = real_frames.shape[:2]
        
        patches = temporal_patchify(real_frames, patch_size=4)
        z_btLd, _ = self.encoder(patches)
        n_spatial = z_btLd.shape[2] // self.packing_factor
        self.z_spatial_seq = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=self.packing_factor)
        
        z_unpacked = unpack_spatial_to_bottleneck(self.z_spatial_seq, k=self.packing_factor)
        all_pooled = pool_latents(z_unpacked)
        
        self.pooled_history = [all_pooled[:, t] for t in range(T - self.frame_stack, T)]
        
        self.a_seq = torch.zeros((B, T, 16), device=self.device)
        self.a_seq[..., :real_actions.shape[-1]] = real_actions.clamp(-1, 1)
        
        return self._build_ppo_observation()

    @torch.no_grad()
    def step(self, ppo_action):
        B = self.z_spatial_seq.shape[0]
        self.current_step += 1
        
        new_a = torch.zeros((B, 1, 16), device=self.device)
        new_a[..., :ppo_action.shape[-1]] = ppo_action.clamp(-1, 1)
        self.a_seq = torch.cat([self.a_seq, new_a], dim=1)
        
        act_mask = torch.zeros(16, device=self.device)
        act_mask[:ppo_action.shape[-1]] = 1.0
        
        z_pred = self.dyn(self.z_spatial_seq, actions=self.a_seq, act_mask=act_mask)
        next_z_packed = z_pred[:, -1:]
        
        self.z_spatial_seq = torch.cat([self.z_spatial_seq, next_z_packed], dim=1)
        
        z_unpacked = unpack_spatial_to_bottleneck(next_z_packed, k=self.packing_factor)
        step_pooled = pool_latents(z_unpacked).squeeze(1)
        
        reward_logit = self.reward_head(step_pooled)
        rewards = torch.sigmoid(reward_logit).squeeze(-1)
        
        self.pooled_history.pop(0)
        self.pooled_history.append(step_pooled)
        next_obs = self._build_ppo_observation()
        
        dones = torch.zeros(B, device=self.device)
        if self.current_step >= self.max_horizon:
            dones = torch.ones(B, device=self.device)
            
        return next_obs, rewards, dones

    def _build_ppo_observation(self):
        return torch.stack(self.pooled_history, dim=1)