import os
import subprocess
from fastapi import FastAPI
from ultralytics import YOLO

app = FastAPI()

# Set path to the model weights
WEIGHTS_PATH = "models/best.pt"

# Function to download model weights using gdown
def download_weights():
    if not os.path.exists(WEIGHTS_PATH):
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        print("Downloading model weights using gdown...")
        file_id = "1sbtPi-0NJM2joeTakOY2kVaN0t60zmKk"  # Replace with your actual file ID
        subprocess.run(["gdown", "--id", file_id, "-O", WEIGHTS_PATH], check=True)
        print("Download complete.")

# Download weights before loading the model
download_weights()

# Load YOLO model
model = YOLO(WEIGHTS_PATH)

@app.get("/")
def read_root():
    return {"message": "YOLOv8 FastAPI app is running!"}
