# DA6401_Assignment_2

DA24M017 DA6401 Assignment 2

You can run the required code for this assignment either using Jupyter notebooks or python files
# Deep Learning Assignment - Image Classification on iNaturalist Dataset

## Table of Contents
1. [Introduction](#introduction)
2. [Repository Structure](#repository-structure)
3. [Part A: Training CNN from Scratch](#part-a-training-cnn-from-scratch)
   - [Implementation Details](#implementation-details)
   - [How to Run](#how-to-run-part-a)
4. [Part B: Fine-tuning Pre-trained Model](#part-b-fine-tuning-pre-trained-model)
   - [Implementation Details](#implementation-details-1)
   - [How to Run](#how-to-run-part-b)
5. [Results](#results)
6. [Requirements](#requirements)
7. [References](#references)

## Introduction
This repository contains solutions for both parts of the DA6401 Deep Learning assignment:
- **Part A**: Building and training a Convolutional Neural Network (CNN) from scratch
- **Part B**: Fine-tuning a pre-trained EfficientNetV2 model

The project uses the iNaturalist 12K dataset with 10 classes, implementing proper data splits, hyperparameter optimization using WandB, and comprehensive model evaluation.

## Repository Structure
```
.
├── Part_A.ipynb                        # Jupyter Notebook for Part A
├── Part_B.ipynb                        # Jupyter Notebook for Part B
├── README.md                           # This documentation
├── best_model_checkpoints/            # Checkpoints from best sweep/final models
├── config_partB.py                     # Configuration for Part B
├── da6401-assignment2-parta.ipynb      # Additional/earlier notebook for Part A
├── da6401-assignment2-partb (4).ipynb  # Additional/earlier notebook for Part B
├── data_utils_partB.py                 # Data loading for Part B
├── evaluate_partA.py                   # Evaluation and visualization for Part A
├── inaturalist_12K/                    # Dataset folder
│   ├── train/
│   └── val/
├── model_checkpoints/                 # Saved model checkpoints (best & last)
├── model_partB.py                      # EfficientNetV2-based fine-tuning model
├── nature_12K.zip                      # Original dataset archive
├── train_best_partA.py                 # Train best CNN model Part A
├── train_sweep_partA.py               # Hyperparameter tuning for CNN Part A
├── train_sweep_partB.py               # Hyperparameter tuning for Part B
├── utils_partA.py                      # Utilities for CNN from scratch
└── wandb/                              # WandB logs and metadata
```

## Part A: Training CNN from Scratch

### Implementation Details
**Model Architecture:**
- 5 convolutional blocks (Conv2D → Activation → MaxPool2D)
- Configurable filter organization patterns:
  - Same filters (e.g., [32, 32, 32, 32, 32])
  - Doubling filters (e.g., [32, 64, 128, 256, 512])
  - Halving filters (e.g., [512, 256, 128, 64, 32])
- Multiple activation functions supported:
  - ReLU, GELU, SiLU, Mish
- Final classifier with dropout and dense layers

**Key Features:**
- WandB hyperparameter sweeping (30+ experiments)
- Stratified 80/20 train-validation split
- Comprehensive evaluation including:
  - Test accuracy metrics
  - 10×3 prediction visualization grid
  - First-layer filter visualizations

### How to Run Part A
```bash
# Run sweep
python train_sweep_partA.py

# Train with best config
python train_best_partA.py

# Evaluate
python evaluate_partA.py
```

Or use the notebook:
```bash
jupyter notebook Part_A.ipynb
```

## Part B: Fine-tuning Pre-trained Model

### Implementation Details
**Model Architecture:**
- EfficientNetV2-Small pre-trained on ImageNet
- Flexible fine-tuning strategies:
  - Freeze first N blocks (options: 3,5,7,9)
  - Replace final classifier head
  - Configurable dropout rates (0.2, 0.3, 0.4)
  - Learning rate options (1e-2, 1e-3, 1e-4)

**Key Features:**
- Partial model freezing for efficient training
- Bayesian hyperparameter optimization (10+ experiments)
- WandB integration for experiment tracking
- Complete fine-tuning pipeline with:
  - Layer freezing/unfreezing
  - Gradient updates only on unfrozen parameters

### How to Run Part B
```bash
# Run sweep
python train_sweep_partB.py
```

Or open and run:
```bash
jupyter notebook Part_B.ipynb
```

### Key Findings
- Fine-tuned model achieves ~10% higher accuracy
- EfficientNet trains faster despite larger size
- Best from-scratch configuration:
  - Filter organization: Half
  - Activation: SiLU
  - Dropout: 0.3
- Best fine-tuning configuration:
  - Freeze first 5 blocks
  - Learning rate: 1e-3
  - Dropout: 0.2

## Requirements
- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning
- Torchvision
- Weights & Biases (wandb)
- NumPy
- Matplotlib

Install all dependencies:
```bash
pip install -r requirements.txt
```

## References
1. PyTorch Lightning Documentation
2. EfficientNetV2 Paper: "EfficientNetV2: Smaller Models and Faster Training"
3. WandB Documentation for experiment tracking
4. iNaturalist Dataset Documentation


