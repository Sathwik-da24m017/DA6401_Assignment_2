import os
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb
from wandb.sdk.wandb_run import Run

wandb.login(key='your_api_key_here')

@dataclass
class ModelConfig:
    img_size: int = 128
    num_classes: int = 10
    batch_size: int = 32
    epochs: int = 10
    filter_organization: str = "double"
    activation: str = "silu"
    data_augmentation: bool = True
    batch_norm: bool = True
    dropout: float = 0.3
    dense_neurons: int = 256
    learning_rate: float = 1e-4

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "filter_organization": {"values": ["same", "double", "half"]},
        "activation": {"values": ["relu", "gelu", "silu", "mish"]},
        "data_augmentation": {"values": [True, False]},
        "batch_norm": {"values": [True, False]},
        "dropout": {"values": [0.1, 0.2, 0.3]},
        "dense_neurons": {"values": [128, 256, 512]},
        "learning_rate": {"values": [1e-3, 1e-4]},
    },
}

class CustomCNN(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self._build_model()

    def _build_model(self):
        # Convolutional blocks
        filters = self._get_filter_counts()
        conv_blocks = []
        in_channels = 3
        
        for out_channels in filters:
            conv_blocks.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels) if self.config.batch_norm else nn.Identity(),
                self._get_activation(self.config.activation),
                nn.MaxPool2d(kernel_size=2),
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_blocks)
        
        # Classifier
        flattened_size = (self.config.img_size // 32) ** 2 * filters[-1]
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, self.config.dense_neurons),
            self._get_activation(self.config.activation),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.dense_neurons, self.config.num_classes)
        )

    def _get_filter_counts(self):
        if self.config.filter_organization == "same":
            return [32] * 5
        elif self.config.filter_organization == "double":
            return [32, 64, 128, 256, 512]
        elif self.config.filter_organization == "half":
            return [512, 256, 128, 64, 32]
        raise ValueError("Invalid filter organization")

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "mish": nn.Mish(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

def get_transforms(config: ModelConfig) -> T.Compose:
    transforms_list = [
        T.Resize((config.img_size, config.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if config.data_augmentation:
        transforms_list.insert(1, T.RandomHorizontalFlip())
        transforms_list.insert(2, T.RandomRotation(10))
    return T.Compose(transforms_list)

def prepare_data(config: ModelConfig) -> tuple[DataLoader, DataLoader]:
    dataset = datasets.ImageFolder("inaturalist_12K/train", transform=get_transforms(config))
    
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
    
    train_loader = DataLoader(Subset(dataset, train_indices), 
                           batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices),
                         batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader

def train():
    run = wandb.init()
    config = ModelConfig(**{**run.config, "epochs": 10})
    
    run.name = (
        f"fo_{config.filter_organization}_act_{config.activation}_"
        f"aug_{config.data_augmentation}_bn_{config.batch_norm}_"
        f"do_{config.dropout}_dn_{config.dense_neurons}_lr_{config.learning_rate}"
    )
    
    train_loader, val_loader = prepare_data(config)
    model = CustomCNN(config)
    
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=WandbLogger(project="da6401_assignment2"),
        accelerator="auto",
    )
    
    trainer.fit(model, train_loader, val_loader)
    run.finish()

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="da6401_assignment2")
    wandb.agent(sweep_id, train, count=30)