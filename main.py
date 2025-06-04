import os
import urllib.request
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO

app = FastAPI()

# Constants
WEIGHTS_PATH = "models/best.pt"
MODEL_URL = "https://huggingface.co/Ravurijeetendra12/YoloV8-face-liveness-model/resolve/main/best.pt"

# Download YOLOv8 weights if not already present
def download_weights():
    if not os.path.exists(WEIGHTS_PATH):
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        print("Downloading model weights from Hugging Face...")
        urllib.request.urlretrieve(MODEL_URL, WEIGHTS_PATH)
        print("Model weights downloaded.")

download_weights()

# Load YOLO model
model = YOLO(WEIGHTS_PATH)

@app.get("/")
def read_root():
    return {"message": "YOLOv8 FastAPI app is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded file as bytes and convert to numpy array
        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format"})

        # Run prediction
        results = model(img)

        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                detections.append({
                    "class": cls,
                    "confidence": round(conf, 3),
                    "bbox": [round(coord, 1) for coord in xyxy]
                })

        return {"detections": detections}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
