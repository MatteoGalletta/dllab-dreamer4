import torch
import zarr
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

class PushTSequenceDataset(Dataset):
    def __init__(self, zarr_path, seq_len=64):
        """
        Initialisiert das Dataset, ohne die eigentlichen Bilder in den RAM zu laden.
        """
        self.seq_len = seq_len
        
        self.root = zarr.open(zarr_path, mode='r')
        
        self.img_data = self.root['data/img']        # Oft Shape: (Total_Steps, 96, 96, 3)
        self.state_data = self.root['data/state']    # Oft Shape: (Total_Steps, 2)
        self.action_data = self.root['data/action']  # Oft Shape: (Total_Steps, 2)
        

        episode_ends = self.root['meta/episode_ends'][:]
        
  
        self.valid_starts = []
        start = 0
        for end in episode_ends:
            if (end - start) >= self.seq_len:
                self.valid_starts.extend(range(start, end - self.seq_len + 1))
            start = end

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        """
        Wird für jeden Sample im Batch aufgerufen. Erst HIER wird der RAM belastet.
        """
        start_idx = self.valid_starts[idx]
        end_idx = start_idx + self.seq_len
        
        images = self.img_data[start_idx:end_idx]    # numpy uint8!
        states = self.state_data[start_idx:end_idx]  # numpy float32
        actions = self.action_data[start_idx:end_idx]# numpy float32
        
        is_terminal = np.zeros((self.seq_len, 1), dtype=np.float32)
        if end_idx in self.root['meta/episode_ends'][:]:
             is_terminal[-1] = 1.0


        images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
        
        return {
            'image': images_tensor,          # Shape: (T, C, H, W), Dtype: torch.uint8
            'state': torch.from_numpy(states),
            'action': torch.from_numpy(actions),
            'is_terminal': torch.from_numpy(is_terminal)
        }
        

def create_pusht_dataloader(zarr_path, batch_size, seq_len, num_workers=2):
    """
    Fabrikfunktion, die den speichereffizienten DataLoader für DreamerV4 zurückgibt.
    """
    dataset = PushTSequenceDataset(zarr_path, seq_len=seq_len)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,            
        num_workers=num_workers, 
        pin_memory=True,         
        drop_last=True           
        )
    
    return loader