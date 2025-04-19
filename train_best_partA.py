import os
from train_sweep import ModelConfig, CustomCNN, prepare_data
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

def train_best_model():
    best_config = ModelConfig(
        filter_organization="half",
        activation="silu",
        data_augmentation=True,
        batch_norm=True,
        dropout=0.2,
        dense_neurons=512,
        learning_rate=1e-4
    )

    wandb.init(
        project="da6401_assignment2",
        config=best_config.__dict__,
        name="best_model"
    )
    
    train_loader, val_loader = prepare_data(best_config)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="checkpoints",
        filename="best-model-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        mode="max"
    )
    
    model = CustomCNN(best_config)
    trainer = pl.Trainer(
        max_epochs=best_config.epochs,
        logger=WandbLogger(project="da6401_assignment2"),
        callbacks=[checkpoint_callback],
        accelerator="auto"
    )
    
    trainer.fit(model, train_loader, val_loader)
    wandb.finish()

if __name__ == "__main__":
    train_best_model()