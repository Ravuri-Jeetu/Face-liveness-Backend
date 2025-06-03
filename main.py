import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImagePayload(BaseModel):
    image: str

# Google Drive file ID and destination path
GDRIVE_FILE_ID = "1sbtPi-0NJM2joeTakOY2kVaN0t60zmKk"  # Replace with your actual file ID
WEIGHTS_PATH = "best.pt"

def download_file_from_google_drive(id: str, destination: str):
    """Download file from Google Drive handling large files."""
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# Download weights if not already present
if not os.path.exists(WEIGHTS_PATH):
    print("Downloading YOLOv8 weights from Google Drive...")
    download_file_from_google_drive(GDRIVE_FILE_ID, WEIGHTS_PATH)
    print("Download complete.")

# Load the YOLOv8 model
model = YOLO(WEIGHTS_PATH)

@app.post("/predict")
async def predict(payload: ImagePayload):
    try:
        header, encoded = payload.image.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model(img_np)
        prediction_text = "No face detected"

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            class_labels = {0: "Fake Face", 1: "Real Face"}
            prediction_text = f"{class_labels.get(cls_id, 'Unknown')} (Confidence: {conf:.2f})"
        else:
            prediction_text = "No face detected"

        return {"prediction": prediction_text}

    except Exception as e:
        print("Error during prediction:", e)
        return {"prediction": "Error during prediction"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
