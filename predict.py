"""
predict.py
----------
Simple inference script. Just pass an image path and get the result.

Usage:
    python predict.py --image "path/to/image.jpg"
"""

import argparse
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

WEIGHTS = "runs/detect/runs/yolo_run6/weights/best.pt"
CLASS_NAMES = ['bird', 'drone']
COLORS = {'bird': '#2ecc71', 'drone': '#e74c3c'}

def predict(image_path: str):
    model = YOLO(WEIGHTS)
    results = model.predict(source=image_path, conf=0.25, verbose=False)

    img = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id    = int(box.cls)
            conf      = float(box.conf)
            label     = CLASS_NAMES[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append((label, conf, x1, y1, x2, y2))

            color = COLORS[label]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{label} {conf:.0%}",
                    color='white', fontsize=11, fontweight='bold',
                    bbox=dict(facecolor=color, alpha=0.8, pad=2))

    if not detections:
        ax.set_title("No detections found", fontsize=14, color='gray')
    else:
        title = " | ".join([f"{l} ({c:.0%})" for l, c, *_ in detections])
        ax.set_title(title, fontsize=13, fontweight='bold')

    ax.axis('off')
    plt.tight_layout()
    plt.savefig("prediction_result.jpg", bbox_inches='tight')
    plt.show()

    print("\n===== RESULTS =====")
    if not detections:
        print("No objects detected.")
    for label, conf, x1, y1, x2, y2 in detections:
        print(f"  {label.upper()}  —  confidence: {conf:.1%}  |  box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    print("===================")
    print("Result image saved to: prediction_result.jpg")

    return detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()
    predict(args.image)
