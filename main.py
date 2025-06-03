import os
import requests
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

# Google Drive download link using your model's ID
MODEL_URL = "https://drive.google.com/uc?export=download&id=1sbtPi-0NJM2joeTakOY2kVaN0t60zmKk"
MODEL_PATH = "best.pt"

# Download the model if it doesn't exist
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        response = requests.get(MODEL_URL, allow_redirects=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Model downloaded.")
        else:
            raise Exception("Failed to download model from Google Drive.")

# Download and load model
download_model()
model = YOLO(MODEL_PATH)

# FastAPI app
app = FastAPI()

# Allow CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model(img)
    result_class = results[0].names[int(results[0].probs.top1)] if results[0].probs else "unknown"
    return {"result": result_class}
