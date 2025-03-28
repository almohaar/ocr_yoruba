# custom_models/detection/yolov8_model.py
from ultralytics import YOLO
import os

class YOLOv8:
    def __init__(self, model_path=None):
        # Set the default model path to the weights folder
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'weights', 'yolov8n.pt')
        if os.path.exists(model_path):
            print(f"Loading local weights from: {model_path}")
            self.model = YOLO(model_path)
        else:
            print("Local weights not found, attempting download...")
            self.model = YOLO('yolov8n.pt')
    
    def train(self, **kwargs):
        # Forward the training call to the underlying YOLO model.
        return self.model.train(**kwargs)
    
    def predict(self, x):
        return self.model(x)
