from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('best.pt')  # Replace with your trained model path

# Define FastAPI app
app = FastAPI()

# Enable CORS (allow all origins for now; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class Frame(BaseModel):
    image_base64: str  # Expecting data:image/jpeg;base64,...

# Prediction endpoint
@app.post("/predict")
async def predict_liveness(frame: Frame):
    try:
        # Decode base64 image
        if "," in frame.image_base64:
            _, encoded = frame.image_base64.split(",", 1)
        else:
            encoded = frame.image_base64

        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Run YOLOv8 inference
        results = model(image, verbose=False)[0]

        # Extract detected classes
        detected_classes = results.boxes.cls.cpu().numpy().astype(int)

        # Check for 'real' class (assumed as class 1)
        if 1 in detected_classes:
            return {"status": "real"}
        else:
            return {"status": "fake"}

    except Exception as e:
        return {"status": "error", "detail": str(e)}
