from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import tempfile
import uvicorn
import os
import requests

MODEL_PATH = "/tmp/yolov8n.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    print("Downloading YOLOv8n model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Model downloaded!")

# Load model YOLO
model = YOLO(MODEL_PATH)

# Create FastAPI app
app = FastAPI()

# Enable CORS so your Expo app can talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:19006"] untuk keamanan
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Save uploaded file to temp
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(await file.read())
    temp_file.close()

    # Run YOLO prediction
    results = model.predict(temp_file.name)

    detections = []
    for r in results:
        boxes = r.boxes.xyxy.tolist()
        labels = [model.names[int(c)] for c in r.boxes.cls]
        confidences = r.boxes.conf.tolist()
        for i in range(len(labels)):
            detections.append({
                "label": labels[i],
                "confidence": round(confidences[i], 3),
                "box": boxes[i]
            })

    return {"detections": detections}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
