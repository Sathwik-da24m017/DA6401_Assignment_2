import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Subset
import torchvision.transforms as T
import torchvision.datasets as datasets

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def get_transforms(img_size: int, augment: bool = True) -> T.Compose:
    transforms = [
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if augment:
        transforms.insert(1, T.RandomHorizontalFlip())
        transforms.insert(2, T.RandomRotation(10))
    return T.Compose(transforms)

def prepare_datasets(data_path: str, img_size: int, augment: bool = True):
    dataset = datasets.ImageFolder(
        data_path,
        transform=get_transforms(img_size, augment)
    )
    
    # Stratified split
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
    
    train_indices, val_indices = [], []
    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices)