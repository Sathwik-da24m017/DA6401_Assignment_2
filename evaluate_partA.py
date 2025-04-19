import os
import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import wandb
from train_sweep import ModelConfig, CustomCNN, denormalize

def visualize_predictions(model, test_dataset, run):
    samples = {i: [] for i in range(model.config.num_classes)}
    for img, label in test_dataset:
        if len(samples[label]) < 3:
            samples[label].append(img)
        if all(len(v) == 3 for v in samples.values()):
            break
    
    predictions = {}
    for cls in range(model.config.num_classes):
        imgs = torch.stack(samples[cls]).to(device)
        with torch.no_grad():
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
        predictions[cls] = preds.cpu().tolist()
    
    fig, axes = plt.subplots(10, 3, figsize=(12, 30))
    for cls in range(model.config.num_classes):
        for j in range(3):
            img = denormalize(samples[cls][j])
            ax = axes[cls][j]
            ax.imshow(img.permute(1, 2, 0))
            ax.set_title(
                f"True: {test_dataset.classes[cls]}\n"
                f"Pred: {test_dataset.classes[predictions[cls][j]]}"
            )
            ax.axis("off")
    plt.tight_layout()
    wandb.log({"predictions": wandb.Image(fig)})
    plt.close()

def visualize_filters(model, run):
    first_conv = model.conv_layers[0]
    weights = first_conv.weight.data.cpu()
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    grid = torchvision.utils.make_grid(weights, nrow=8, padding=2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("First Layer Filters")
    plt.axis("off")
    wandb.log({"filters": wandb.Image(plt)})
    plt.close()

def evaluate():
    wandb.init(project="da6401_assignment2", job_type="eval")
    
    best_config = ModelConfig(
        filter_organization="half",
        activation="silu",
        data_augmentation=True,
        batch_norm=True,
        dropout=0.2,
        dense_neurons=512,
        learning_rate=1e-4
    )
    
    test_transform = T.Compose([
        T.Resize((best_config.img_size, best_config.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_path = "inaturalist_12K/test"
    if not os.path.exists(test_path):
        test_path = "inaturalist_12K/val"
    
    test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=best_config.batch_size, shuffle=False)
    
    model = CustomCNN.load_from_checkpoint(
        "checkpoints/best-model.ckpt",
        config=best_config
    )
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    wandb.log({"test_accuracy": accuracy})
    
    visualize_predictions(model, test_dataset, wandb.run)
    visualize_filters(model, wandb.run)
    wandb.finish()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate()