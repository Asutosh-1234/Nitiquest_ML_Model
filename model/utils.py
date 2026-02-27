"""
utils.py
--------
Dataset loader for the YOLO-format Drone vs Bird dataset.
Supports both:
  - Classification mode: loads image + single class label
  - Detection mode: handled by Ultralytics YOLO directly
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


CLASS_NAMES = ['bird', 'drone']


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class DroneBirdDataset(Dataset):
    """
    Loads images and their YOLO-format labels for classification.
    Each label file contains lines: <class_id> <x> <y> <w> <h>
    The class_id of the first annotation is used as the image label.
    """
    def __init__(self, images_dir: str, labels_dir: str, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load label (use first annotation's class_id)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)
        label = 0  # default: bird
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    label = int(first_line.split()[0])

        return image, label


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────
def get_transforms(split: str = 'train'):
    """Return torchvision transforms for train/val/test splits."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────
def get_dataloader(images_dir: str, labels_dir: str, split: str = 'train',
                   batch_size: int = 32, num_workers: int = 4) -> DataLoader:
    dataset = DroneBirdDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transform=get_transforms(split)
    )
    shuffle = (split == 'train')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True)
    print(f"[{split.upper()}] {len(dataset)} images  |  {len(loader)} batches")
    return loader


# ─────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


def print_classification_report(y_true, y_pred):
    print("\n" + "=" * 50)
    print("Classification Report")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses,   label='Val Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs,   label='Val Acc')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
    plt.show()
