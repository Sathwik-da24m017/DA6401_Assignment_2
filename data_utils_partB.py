import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import defaultdict

def get_transforms(img_size=224, augment=True):
    """Return appropriate transforms for training/validation"""
    base_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    if augment:
        base_transforms.insert(1, transforms.RandomHorizontalFlip())
        base_transforms.insert(2, transforms.RandomRotation(15))
    
    return transforms.Compose(base_transforms)

def prepare_datasets(train_path, val_path, img_size=224):
    """Prepare train/val datasets with stratified split"""
    train_tf = get_transforms(img_size, augment=True)
    val_tf = get_transforms(img_size, augment=False)
    
    full_train = datasets.ImageFolder(train_path, transform=train_tf)
    
    # Stratified split
    idx_by_class = defaultdict(list)
    for idx, (_, lbl) in enumerate(full_train.samples):
        idx_by_class[lbl].append(idx)
    
    train_idx, val_idx = [], []
    for lbl, idxs in idx_by_class.items():
        np.random.shuffle(idxs)
        split = int(0.8 * len(idxs))
        train_idx += idxs[:split]
        val_idx += idxs[split:]
    
    # Create subsets
    train_ds = Subset(full_train, train_idx)
    
    # Reuse dataset object with val transforms
    full_train.transform = val_tf
    val_ds = Subset(full_train, val_idx)
    
    # Test set
    test_ds = datasets.ImageFolder(val_path, transform=val_tf)
    
    return train_ds, val_ds, test_ds

def create_dataloaders(train_ds, val_ds, test_ds, batch_size=32):
    """Create dataloaders from datasets"""
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, 
        shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, 
        shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, 
        shuffle=False, num_workers=4
    )
    return train_loader, val_loader, test_loader