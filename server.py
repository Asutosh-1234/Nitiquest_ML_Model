"""
server.py
---------
Flask backend that connects the SkyWatch website to the YOLOv8 model.

Install:
    pip install flask flask-cors ultralytics pillow

Run:
    python server.py

Then open index.html in your browser.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)  # Allow requests from the HTML frontend

# ── Load model once at startup ───────────────────────────────
WEIGHTS = "runs/detect/runs/yolo_run6/weights/best.pt"
print(f"Loading model from: {WEIGHTS}")
model = YOLO(WEIGHTS)
CLASS_NAMES = ['bird', 'drone']
print("Model loaded! Server ready.")


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Read image
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Run inference
        results = model.predict(source=image, conf=0.25, verbose=False)

        # Parse detections
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                conf   = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    'label':      CLASS_NAMES[cls_id],
                    'confidence': round(conf, 4),
                    'box':        [round(x1), round(y1), round(x2), round(y2)]
                })

        return jsonify({'detections': detections})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': WEIGHTS})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
