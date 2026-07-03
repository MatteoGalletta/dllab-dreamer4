import torch
import h5py
import hdf5plugin
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

class PushTSequenceDataset(Dataset):
    def __init__(self, h5_path, seq_len=50, action_chunk_size=5):
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.action_chunk_size = action_chunk_size
        self.raw_seq_len = seq_len * action_chunk_size
        
        # Öffne die Datei einmal kurz, um die Metadaten zu lesen
        with h5py.File(self.h5_path, 'r') as f:
            # Hinweis: Passe die Keys an, falls 'quentinll/lewm-pusht' leicht abweichende Namen nutzt
            # Derive episode ends from the actual dataset layout.
            self.num_frames = f['pixels'].shape[0]
            self.episode_ends = f['ep_offset'][:] + f['ep_len'][:]
            
        # Wir berechnen im Vorfeld alle gültigen Start-Indizes.
        # Eine Sequenz darf NICHT über das Ende einer Episode hinausragen.
        self.valid_start_indices = []
        episode_start = 0
        
        for end_idx in self.episode_ends:
            # end_idx ist der exklusive oder inklusive Endpunkt der Episode
            for i in range(episode_start, end_idx - self.raw_seq_len + 1):
                self.valid_start_indices.append(i)
            episode_start = end_idx

    def __len__(self):
        return len(self.valid_start_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_start_indices[idx]
        end_idx = start_idx + self.raw_seq_len
        
        with h5py.File(self.h5_path, 'r') as f:
            # Lade den zusammenhängenden Chunk aus der H5-Datei
            images = f['pixels'][start_idx:end_idx]    # Shape: (raw_seq_len, 224, 224, 3)
            actions = f['action'][start_idx:end_idx]  # Shape: (raw_seq_len, 2)
            states = f['state'][start_idx:end_idx]    # Shape: (raw_seq_len, 7) - falls vorhanden

        # One image/state per action chunk (first frame of each chunk).
        images = images[::self.action_chunk_size]
        states = states[::self.action_chunk_size]
        actions = actions.reshape(self.seq_len, -1)
        
        # Konvertierung in PyTorch-Tensoren
        # Dreamer erwartet Bilder meist im Format (Sequence, Channels, Height, Width)
        # Zudem skalieren wir die Pixelwerte von [0, 255] auf [0.0, 1.0]
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        actions = torch.from_numpy(actions).float()
        states = torch.from_numpy(states).float()
        
        return {
            "image": images,   # [seq_len, 3, 224, 224]
            "action": actions, # [seq_len, action_chunk_size*2]
            "state": states    # [seq_len, state_dim]
        }
        

def create_pusht_dataloader(
    h5_path=None,
    data_dirs=None,
    batch_size=1,
    seq_len=64,
    action_chunk_size=5,
    num_workers=2,
):

    dataset = PushTSequenceDataset(
        h5_path=h5_path,
        seq_len=seq_len,
        action_chunk_size=action_chunk_size,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,            
        num_workers=num_workers, 
        pin_memory=True,         
        drop_last=True           
        )
    
    return loader