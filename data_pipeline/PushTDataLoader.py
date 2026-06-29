import torch
import zarr
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

class PushTSequenceDataset(Dataset):
    def __init__(self, zarr_path=None, data_dirs=None, seq_len=64):
        """
        Initialisiert das Dataset, ohne die eigentlichen Sequenzen in den RAM zu laden.
        """
        self.seq_len = seq_len

        if data_dirs is None:
            if zarr_path is None:
                raise ValueError("Either zarr_path or data_dirs must be provided")
            data_dirs = [zarr_path]

        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]

        self.data_dirs = [str(Path(path)) for path in data_dirs]
        self.roots = [zarr.open(path, mode='r') for path in self.data_dirs]
        self.img_data = [root['data/img'] for root in self.roots]
        self.state_data = [root['data/state'] for root in self.roots]
        self.action_data = [root['data/action'] for root in self.roots]
        self.episode_ends = [root['meta/episode_ends'][:] for root in self.roots]

        self.valid_starts = []
        for source_idx, episode_ends in enumerate(self.episode_ends):
            start = 0
            for end in episode_ends:
                if (end - start) >= self.seq_len:
                    self.valid_starts.extend(
                        (source_idx, start_idx) for start_idx in range(start, end - self.seq_len + 1)
                    )
                start = end

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        """
        Wird für jeden Sample im Batch aufgerufen. Erst HIER wird der RAM belastet.
        """
        source_idx, start_idx = self.valid_starts[idx]
        end_idx = start_idx + self.seq_len
        
        images = self.img_data[source_idx][start_idx:end_idx]    # numpy float32, raw pixel values
        states = self.state_data[source_idx][start_idx:end_idx]  # numpy float32
        actions = self.action_data[source_idx][start_idx:end_idx]# numpy float32
        
        is_terminal = np.zeros((self.seq_len, 1), dtype=np.float32)
        if end_idx in self.episode_ends[source_idx]:
             is_terminal[-1] = 1.0


        images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
        
        return {
            'image': images_tensor,          # Shape: (T, C, H, W), Dtype: torch.float32
            'state': torch.from_numpy(states),
            'action': torch.from_numpy(actions),
            'is_terminal': torch.from_numpy(is_terminal)
        }
        

def create_pusht_dataloader(zarr_path=None, data_dirs=None, batch_size=1, seq_len=64, num_workers=2):
    """
    Fabrikfunktion, die den speichereffizienten DataLoader für DreamerV4 zurückgibt.
    """
    dataset = PushTSequenceDataset(zarr_path=zarr_path, data_dirs=data_dirs, seq_len=seq_len)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,            
        num_workers=num_workers, 
        pin_memory=True,         
        drop_last=True           
        )
    
    return loader