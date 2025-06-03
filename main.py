import os
import requests
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from io import BytesIO
from PIL import Image

# Constants
WEIGHTS_PATH = "best.pt"
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1sbtPi-0NJM2joeTakOY2kVaN0t60zmKk"

# Download model weights if not already present
def download_weights():
    if not os.path.exists(WEIGHTS_PATH):
        print("Downloading model weights...")
        response = requests.get(DRIVE_URL)
        with open(WEIGHTS_PATH, "wb") as f:
            f.write(response.content)
        print("Download complete.")

# Call download at startup
download_weights()

# Load YOLOv8 model
model = YOLO(WEIGHTS_PATH)

# Set up FastAPI app
app = FastAPI()

# Allow CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect/")
async def detect_liveness(file: UploadFile = File(...)):
    try:
        # Read image from request
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        # Run YOLOv8 inference
        results = model(image_np)[0]

        # Parse detections
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[cls]
            detections.append({
                "class": label,
                "confidence": round(conf, 2),
                "box": [x1, y1, x2, y2]
            })

        return {"status": "success", "detections": detections}

    except Exception as e:
        return {"status": "error", "message": str(e)}
