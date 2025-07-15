!pip install -q Ultralytics
from ultralytics import YOLO
# Load the YOLO model
model = YOLO("yolo12s.pt")
# Downloading dataset from kaggle
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="Znc5TT13G6UMrbR6fXOC")
project = rf.workspace("graduation-project-2023").project("plants-diseases-detection-and-classification")
version = project.version(12)
dataset = version.download("yolov12")
import os
# Path to YAML file
yaml_path = os.path.join(dataset.location, "data.yaml")

# Preview the file content
with open(yaml_path, 'r') as f:
    print(f.read())
 # Train the model
results=model.train(data="/kaggle/working/Plants-Diseases-Detection-and-Classification-12/data.yaml", epochs=100, imgsz=640,batch=10,patience=30)
# Testing the model
results = model("/kaggle/input/20k-multi-class-crop-disease-images/Validation/Common_Rust/Image_11.jpg",conf=0.45)  # list of Results objects
results[0].show()
  
  
