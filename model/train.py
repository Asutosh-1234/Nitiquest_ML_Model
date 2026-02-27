"""
train.py
--------
Train either:
  --mode cnn   : ResNet-50 fine-tuning (classification)
  --mode yolo  : YOLOv8 (detection + classification)

Examples:
  python train.py --mode cnn  --epochs 30 --batch_size 32
  python train.py --mode yolo --epochs 50 --model_size s
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── project imports ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from model import get_cnn_model, get_yolo_model
from utils import get_dataloader, plot_training_curves, CLASS_NAMES

# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Drone vs Bird Training Script")
    parser.add_argument('--mode',        type=str,   default='cnn',      choices=['cnn', 'yolo'])
    parser.add_argument('--data_root',   type=str,   default='./dataset', help='Root dataset folder')
    parser.add_argument('--output_dir',  type=str,   default='./runs',    help='Where to save models & logs')
    parser.add_argument('--epochs',      type=int,   default=30)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--model_size',  type=str,   default='n',         help='YOLOv8 size: n/s/m/l/x')
    parser.add_argument('--freeze',      action='store_true',             help='Freeze CNN backbone')
    parser.add_argument('--resume',      type=str,   default=None,        help='Path to checkpoint to resume from')
    return parser.parse_args()


# ─────────────────────────────────────────────
# CNN Training loop
# ─────────────────────────────────────────────
def train_cnn(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Dataloaders
    train_loader = get_dataloader(
        os.path.join(args.data_root, 'train', 'images'),
        os.path.join(args.data_root, 'train', 'labels'),
        split='train', batch_size=args.batch_size, num_workers=args.num_workers
    )
    val_loader = get_dataloader(
        os.path.join(args.data_root, 'valid', 'images'),
        os.path.join(args.data_root, 'valid', 'labels'),
        split='val', batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Model
    model = get_cnn_model(num_classes=2, pretrained=True, freeze_backbone=args.freeze)
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # Resume checkpoint
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}")

    print(f"\nStarting CNN training for {args.epochs} epochs...\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc  = 100.0 * correct / total

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_running_loss / val_total
        val_acc  = 100.0 * val_correct / val_total
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch+1:3d}/{args.epochs}]  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%  |  "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.output_dir, 'best_cnn.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'val_loss': val_loss,
            }, ckpt_path)
            print(f"  ✔ Saved best model  (val_acc={best_val_acc:.2f}%)")

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.2f}%")

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_cnn.pth'))

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         save_path=os.path.join(args.output_dir, 'training_curves.png'))


# ─────────────────────────────────────────────
# YOLO Training loop
# ─────────────────────────────────────────────
def train_yolo(args):
    model = get_yolo_model(model_size=args.model_size, num_classes=2)

    print(f"\nStarting YOLOv8-{args.model_size} training for {args.epochs} epochs...\n")

    results = model.train(
        data=os.path.abspath('data.yaml'),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch_size,
        lr0=args.lr,
        workers=args.num_workers,
        project=args.output_dir,
        name='yolo_run',
        pretrained=True,
        optimizer='AdamW',
        cos_lr=True,
        plots=True,
        verbose=True,
        device=0,
    )
    print("\nYOLO training complete.")
    print(f"Results saved to: {results.save_dir}")


if __name__ == '__main__':
    args = parse_args()
    print(f"\n{'='*50}")
    print(f"  Drone vs Bird Classifier — TRAIN ({args.mode.upper()})")
    print(f"{'='*50}")

    if args.mode == 'cnn':
        train_cnn(args)
    elif args.mode == 'yolo':
        train_yolo(args)
