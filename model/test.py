"""
test.py
--------
Evaluate a trained model on the test set.

Examples:
  python test.py --mode cnn  --weights ./runs/best_cnn.pth
  python test.py --mode yolo --weights ./runs/yolo_run/weights/best.pt
  python test.py --mode cnn  --weights ./runs/best_cnn.pth --single_image ./dataset/test/images/BT_001.jpg
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
sys.path.insert(0, os.path.dirname(__file__))
from model import get_cnn_model, get_yolo_model
from utils import (
    get_dataloader, plot_confusion_matrix,
    print_classification_report, get_transforms, CLASS_NAMES
)


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Drone vs Bird Testing Script")
    parser.add_argument('--mode',         type=str, default='cnn',  choices=['cnn', 'yolo'])
    parser.add_argument('--weights',      type=str, required=True,  help='Path to trained weights')
    parser.add_argument('--data_root',    type=str, default='./dataset')
    parser.add_argument('--output_dir',   type=str, default='./runs/test_results')
    parser.add_argument('--batch_size',   type=int, default=32)
    parser.add_argument('--num_workers',  type=int, default=4)
    parser.add_argument('--single_image', type=str, default=None,   help='Path to a single image for inference')
    parser.add_argument('--conf',         type=float, default=0.25, help='YOLO confidence threshold')
    return parser.parse_args()


# ─────────────────────────────────────────────
# CNN: evaluate on test set
# ─────────────────────────────────────────────
def test_cnn(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    model = get_cnn_model(num_classes=2, pretrained=False)
    checkpoint = torch.load(args.weights, map_location=device)
    # Support both raw state_dict and checkpoint dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded weights from: {args.weights}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.single_image:
        _cnn_single_inference(model, device, args.single_image, args.output_dir)
        return

    # Full test-set evaluation
    test_loader = get_dataloader(
        os.path.join(args.data_root, 'test', 'images'),
        os.path.join(args.data_root, 'test', 'labels'),
        split='test', batch_size=args.batch_size, num_workers=args.num_workers
    )

    all_preds, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%  ({correct}/{total})")

    print_classification_report(all_labels, all_preds)

    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_preds, save_path=cm_path)


def _cnn_single_inference(model, device, image_path: str, output_dir: str):
    """Run CNN on a single image and display result."""
    transform = get_transforms('test')
    raw_img = Image.open(image_path).convert('RGB')
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        pred_id = probs.argmax().item()
        confidence = probs[pred_id].item() * 100

    label = CLASS_NAMES[pred_id]
    print(f"\nPrediction : {label.upper()}  ({confidence:.1f}% confidence)")
    print(f"Probabilities — Bird: {probs[0]*100:.1f}%  |  Drone: {probs[1]*100:.1f}%")

    # Visualize
    plt.figure(figsize=(6, 6))
    plt.imshow(raw_img)
    color = '#e74c3c' if label == 'drone' else '#2ecc71'
    plt.title(f"Predicted: {label.upper()}  ({confidence:.1f}%)", fontsize=14, color=color, fontweight='bold')
    plt.axis('off')
    out_path = os.path.join(output_dir, 'single_inference.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved result to {out_path}")
    plt.show()


# ─────────────────────────────────────────────
# YOLO: evaluate on test set
# ─────────────────────────────────────────────
def test_yolo(args):
    from ultralytics import YOLO
    model = YOLO(args.weights)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.single_image:
        print(f"\nRunning YOLO inference on: {args.single_image}")
        results = model.predict(
            source=args.single_image,
            conf=args.conf,
            save=True,
            project=args.output_dir,
            name='single_pred'
        )
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                conf   = float(box.conf)
                print(f"  Detected: {CLASS_NAMES[cls_id]}  (conf={conf:.2f})")
        return

    # Full test-set evaluation
    print(f"\nRunning YOLO validation on test set...")
    metrics = model.val(
        data=os.path.abspath('data.yaml'),
        split='test',
        conf=args.conf,
        project=args.output_dir,
        name='yolo_test',
        plots=True,
        verbose=True,
    )
    print(f"\nTest mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"Test mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision        : {metrics.box.mp:.4f}")
    print(f"Recall           : {metrics.box.mr:.4f}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    args = parse_args()
    print(f"\n{'='*50}")
    print(f"  Drone vs Bird Classifier — TEST ({args.mode.upper()})")
    print(f"{'='*50}")

    if args.mode == 'cnn':
        test_cnn(args)
    elif args.mode == 'yolo':
        test_yolo(args)
