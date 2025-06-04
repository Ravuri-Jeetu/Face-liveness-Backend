import os
import urllib.request
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from pydantic import BaseModel
import base64
import re


app = FastAPI()

origins = [
    "http://localhost:3000",  # for local React dev
    "https://face-liveness-gray.vercel.app",  # your deployed frontend domain
    "*",  # or allow all for testing (not recommended for production)
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
    
class ImagePayload(BaseModel):
    image: str  
@app.post("/predict")
async def predict(payload: ImagePayload):
    try:
        # Extract base64 part after comma
        base64_str = re.sub('^data:image/.+;base64,', '', payload.image)
        img_bytes = base64.b64decode(base64_str)
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})

        results = model(img)

        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detections.append({
                    "class": cls,
                    "confidence": round(conf, 3),
                    "bbox": [round(coord, 1) for coord in xyxy]
                })

        return {"detections": detections}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

