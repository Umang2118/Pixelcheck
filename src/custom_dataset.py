"""
custom_dataset.py — Image transforms and dataset loader.

ImageFolder maps classes alphabetically:
    0 → fake   (AI-generated / fake food)
    1 → real   (genuine / real food)
"""

import os
import shutil
import torch
from torchvision import transforms, datasets


def get_transform():
    """Standard ImageNet normalisation pipeline used for validation and inference."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def get_train_transform():
    """Data augmentation pipeline to prevent overfitting during training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

class TransformSubset(torch.utils.data.Dataset):
    """Wrapper to apply a specific transform to a split subset."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


def _is_valid_file(filename: str) -> bool:
    valid_ext = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm',
                 '.tif', '.tiff', '.webp')
    return filename.lower().endswith(valid_ext)


def _remove_ipynb_checkpoints(data_dir: str) -> None:
    checkpoint_path = os.path.join(data_dir, '.ipynb_checkpoints')
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)


def get_dataset(data_dir: str) -> datasets.ImageFolder:
    """
    Load an ImageFolder dataset from *data_dir* WITHOUT transforms initially.
    Transforms should be applied via TransformSubset after splitting.
    """
    _remove_ipynb_checkpoints(data_dir)
    # We load without transform, to allow applying separate train/val transforms later.
    return datasets.ImageFolder(
        root=data_dir,
        is_valid_file=_is_valid_file,
    )
