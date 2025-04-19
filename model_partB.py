import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl

class FineTuneEfficientNetV2(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Load pretrained EfficientNetV2-Small
        self.net = models.efficientnet_v2_s(pretrained=True)
        total_blocks = len(self.net.features)
        
        # Freeze blocks strategy
        freeze_upto = total_blocks - config["freeze_before"]
        for idx, block in enumerate(self.net.features):
            for param in block.parameters():
                param.requires_grad = (idx >= freeze_upto)
        
        # Replace classifier head
        in_features = self.net.classifier[1].in_features
        self.net.classifier = nn.Sequential(
            nn.Dropout(config["dropout"]),
            nn.Linear(in_features, config["num_classes"])
        )
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
    
    def configure_optimizers(self):
        # Only optimize parameters that require gradients
        params = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.Adam(params, lr=self.hparams["learning_rate"])