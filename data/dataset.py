import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .transforms import get_train_transforms, get_val_transforms


class MIMICDataset(Dataset):
    """MIMIC multimodal medical dataset for chest X-ray + EHR classification.
    
    Loads chest X-ray images and corresponding EHR lab values.
    Supports modality dropout simulation for DFC training.
    
    Expected directory structure:
        data_dir/
        +-- images/          # Chest X-ray PNG images
        |   +-- img_001.png
        +-- ehr.npy          # EHR features array (N, 12)
        +-- labels.npy       # Labels array (N,) with values 0-4
        +-- split.json       # {"train": [idx_list], "val": [idx_list], "test": [idx_list]}
    """
    
    CLASS_NAMES = [
        "Normal", "Cardiac_Hypertrophy", "Pulmonary_Edema",
        "Pulmonary_Consolidation", "Pleural_Effusion"
    ]
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        modality_drop_prob: float = 0.0,
        num_ehr_features: int = 12,
    ):
        self.data_dir = data_dir
        self.split = split
        self.modality_drop_prob = modality_drop_prob
        self.num_ehr_features = num_ehr_features
        
        if transform is None:
            self.transform = get_train_transforms() if split == "train" else get_val_transforms()
        else:
            self.transform = transform
        
        split_file = os.path.join(data_dir, "split.json")
        with open(split_file, "r") as f:
            splits = json.load(f)
        self.indices = splits.get(split, [])
        
        ehr_path = os.path.join(data_dir, "ehr.npy")
        self.ehr_data = np.load(ehr_path).astype(np.float32)
        
        self.labels = np.load(os.path.join(data_dir, "labels.npy")).astype(np.int64)
        
        self.image_dir = os.path.join(data_dir, "images")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        sample_idx = self.indices[idx]
        
        img_path = os.path.join(self.image_dir, f"img_{sample_idx:05d}.png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        ehr = torch.from_numpy(self.ehr_data[sample_idx])
        
        label = int(self.labels[sample_idx])
        
        # Modality-level availability masks (scalar per sample, batched to (B,)).
        # Masking happens in feature space inside the DFC module (Eq.9), so the
        # raw inputs are kept intact and only the masks signal unavailability.
        ehr_mask = torch.tensor(1.0, dtype=torch.float32)
        img_mask = torch.tensor(1.0, dtype=torch.float32)
        
        if self.split == "train" and self.modality_drop_prob > 0:
            if np.random.random() < self.modality_drop_prob:
                drop_ehr = np.random.random() < 0.5
                if drop_ehr:
                    ehr_mask = torch.tensor(0.0, dtype=torch.float32)
                else:
                    img_mask = torch.tensor(0.0, dtype=torch.float32)
        
        return {
            "image": image,
            "ehr": ehr,
            "label": torch.tensor(label, dtype=torch.long),
            "ehr_mask": ehr_mask,
            "img_mask": img_mask,
            "sample_idx": sample_idx,
        }


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Build DataLoader with standardized settings.
    
    Args:
        dataset: PyTorch Dataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for GPU transfer.
        
    Returns:
        Configured DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(shuffle and len(dataset) > batch_size),
    )
