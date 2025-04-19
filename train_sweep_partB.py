import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from config import COMMON_CONFIG, initialize_wandb
from data_utils import prepare_datasets, create_dataloaders
from model import FineTuneEfficientNetV2

def train():
    run = wandb.init()
    
    # Combine sweep and common config
    config = {**COMMON_CONFIG, **run.config}
    
    # Set run name
    run.name = (
        f"{config['freeze_before']}_freeze_"
        f"{config['dropout']}_dropout_"
        f"{config['learning_rate']}_lr"
    )
    
    # Prepare data
    train_path = "/kaggle/input/nature-12k/inaturalist_12K/train"
    test_path = "/kaggle/input/nature-12k/inaturalist_12K/val"
    train_ds, val_ds, test_ds = prepare_datasets(train_path, test_path, config["img_size"])
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, config["batch_size"]
    )
    
    # Setup logging and checkpointing
    wandb_logger = WandbLogger(project="da6401_assignment2", log_model="all")
    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath="finetune_ckpts",
        filename=f"ENETV2_{config['freeze_before']}f_{config['dropout']}d_{config['learning_rate']}lr"
    )
    
    # Initialize model
    model = FineTuneEfficientNetV2(config)
    
    # Train
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=wandb_logger,
        callbacks=[checkpoint_cb]
    )
    
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    
    # Log best model
    artifact = wandb.Artifact("efficientnetv2_best", type="model")
    artifact.add_file(checkpoint_cb.best_model_path)
    run.log_artifact(artifact)
    
    run.finish()

if __name__ == "__main__":
    sweep_id = initialize_wandb()
    wandb.agent(sweep_id, function=train, count=10)