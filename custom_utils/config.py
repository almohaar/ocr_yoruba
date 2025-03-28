import os

# Base directory for the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'datasets')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Training hyperparameters
BATCH_SIZE = 16
NUM_WORKERS = 4
DEVICE = 'cpu'  # or 'cuda'
LEARNING_RATE = 1e-4

# Training stages (epochs: e.g., 15, 15, 20)
EPOCHS_STEPS = [15, 15, 20]

# For detection: two classes (0: lower_case, 1: upper_case)
DETECTION_NUM_CLASSES = 2

# For recognition: adjust based on your character set (example: 60)
RECOGNITION_NUM_CLASSES = 60

# Target Character Error Rate (< 10%)
TARGET_CER = 0.10
