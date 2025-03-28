import os
from custom_models.detection.yolov8_model import YOLOv8
from custom_utils.config import BATCH_SIZE, EPOCHS_STEPS

def main():
    # Path to your data configuration file (data.yaml)
    data_yaml = os.path.join(os.getcwd(), "data.yaml")
    
    # Initialize the YOLOv8 wrapper (this loads the model from local weights if available)
    model = YOLOv8()
    
    # Calculate total epochs (sum of your staged epochs)
    total_epochs = sum(EPOCHS_STEPS)
    
    # Use YOLOv8's built-in training API.
    model.train(data=data_yaml, epochs=total_epochs, batch=BATCH_SIZE, imgsz=640)

if __name__ == "__main__":
    main()
