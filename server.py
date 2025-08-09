from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import tempfile
import uvicorn

# Load YOLO model ONCE at startup
model = YOLO("yolov8n.pt")

# Create FastAPI app
app = FastAPI()

# Enable CORS so your Expo app can talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:19006"] for safety
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Save the uploaded image to a temp file
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
    # Run FastAPI on your local network
    uvicorn.run(app, host="0.0.0.0", port=5000)
