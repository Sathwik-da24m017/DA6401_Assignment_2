import wandb

# Common configuration
COMMON_CONFIG = {
    "img_size": 224,
    "num_classes": 10,
    "batch_size": 32,
    "epochs": 10,
    "data_augmentation": True,
}

# Sweep configuration
SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "freeze_before": {"values": [3, 5, 7, 9]},
        "dropout": {"values": [0.2, 0.3, 0.4]},
        "learning_rate": {"values": [1e-2, 1e-3, 1e-4]},
    }
}

def initialize_wandb():
    wandb.login(key='f56388c51b488c425a228537fd2d35e5498a3a91')
    sweep_id = wandb.sweep(SWEEP_CONFIG, project="da6401_assignment2")
    return sweep_id