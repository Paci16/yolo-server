from ultralytics import YOLO
import sys
import json

# Get image path
image_path = sys.argv[1] if len(sys.argv) > 1 else 'uploads/input.jpg'

# Load YOLO model
model = YOLO('yolov8n.pt')

# Run detection
results = model.predict(
    image_path,
    save=True,
    save_txt=True,
    project='runs/detect',
    name='predict',
    exist_ok=True
)

# Collect detections
detections = []
for r in results:
    boxes = r.boxes.xyxy.tolist()  # bounding boxes
    labels = [model.names[int(c)] for c in r.boxes.cls]  # class names
    confidences = r.boxes.conf.tolist()  # confidence scores
    for i in range(len(labels)):
        detections.append({
            "label": labels[i],
            "confidence": round(confidences[i], 3),
            "box": boxes[i]
        })

# Output JSON so app.js can read it
print(json.dumps({"detections": detections}))
