# data_loader.py
"""
Data loading utilities for Oxford-IIIT Pets dataset.
Wraps dataset_wrapper.py (lecturer-provided) into DataLoaders.
"""

import torch
from torch.utils.data import DataLoader
import dataset_wrapper

CLASS_NAMES = [
    'Long-haired cat',
    'Short-haired cat',
    'Long-haired dog',
    'Short-haired dog'
]


def get_dataloaders(img_size=128, batch_size=32, root_path='./data', num_workers=2):
    """
    Creates train/val/test DataLoaders from the 4-class pet dataset.

    Uses dataset_wrapper.get_pet_datasets() which:
      - Downloads Oxford-IIIT Pets
      - Regroups into 4 classes (long/short hair × cat/dog)
      - Applies Resize+ToTensor transforms
      - Splits 80/10/10 with fixed seed 42
    """
    train_dataset, val_dataset, test_dataset = dataset_wrapper.get_pet_datasets(
        img_width=img_size,
        img_height=img_size,
        root_path=root_path
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    print(f"Dataset loaded: {len(train_dataset)} train, "
          f"{len(val_dataset)} val, {len(test_dataset)} test samples")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Image size: {img_size}×{img_size}")

    return train_loader, val_loader, test_loader, CLASS_NAMES


def count_parameters(model):
    """Returns total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)