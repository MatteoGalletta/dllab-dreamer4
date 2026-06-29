import torch
import time
from data_pipeline.PushTDataLoader import create_pusht_dataloader

#change path
ZARR_PATH = "pusht_cchi_v7_replay.zarr" 

def run_sanity_check():
    print("Initialisiere DataLoader...")
    loader = create_pusht_dataloader(
        zarr_path=ZARR_PATH, 
        batch_size=16, 
        seq_len=64, 
        num_workers=2
    )
    
    print("DataLoader erstellt. Ziehe ersten Batch von der SSD...")
    start_time = time.time()
    
    batch = next(iter(loader))
    
    print(f"\n--- Batch geladen in {time.time() - start_time:.2f} Sekunden ---")
    for key, tensor in batch.items():
        print(f"{key.ljust(12)}: Shape {tensor.shape}, Dtype {tensor.dtype}")
        
    print("\n--- GPU-Transfer & Normalisierungs-Test ---")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU gefunden. Transferiere Daten auf: {device}")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"XPU gefunden. Transferiere Daten auf: {device}")
    else:
        print("Keine CUDA- oder XPU-GPU gefunden! Test läuft auf CPU.")
        device = torch.device("cpu")
        
    images_gpu = batch['image'].to(device)
    print(f"Bilder auf Device   : Dtype {images_gpu.dtype} (Sollte uint8 bleiben!)")
    
    images_norm = (images_gpu.float() / 255.0) - 0.5
    print(f"Nach Normalisierung : Dtype {images_norm.dtype}")
    print(f"Wertebereich        : Min {images_norm.min():.2f}, Max {images_norm.max():.2f}")
    
    print("\n--- Detaillierte Shape-Informationen ---")
    for key, tensor in batch.items():
        print(f"{key.ljust(12)}: Shape {str(tensor.shape).ljust(20)} | Dtype {str(tensor.dtype).ljust(12)} | Size {tensor.numel():,} elements")

if __name__ == "__main__":
    run_sanity_check()