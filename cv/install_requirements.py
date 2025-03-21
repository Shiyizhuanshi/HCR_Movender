#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path

# Required packages
requirements = [
    "depthai",
    "mediapipe",
    "opencv-python<5.0",
    "numpy<3.0",
    "blobconverter",
    "keyboard",
    "tensorflow",
    "scikit-learn",
    "blobconverter"
]

def install_packages():
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", *requirements])

def download_model():
    print("Downloading YOLOv8n model using blobconverter...")
    import blobconverter

    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)
    
    blob_path = blobconverter.from_zoo(
        name="yolov8n_coco_640x352",
        shaves=6,
        zoo_type="depthai",
        output_dir=model_dir,
        compile_params=["-ip U8"],
    )

    print(f"Model downloaded to: {blob_path}")

def main():
    install_packages()
    download_model()
    print("\n All set! You're ready to run the program.")

if __name__ == "__main__":
    main()
