import os
from glob import glob
from torch.utils.data import Dataset
import torch
import cv2
import numpy as np

def load_image(image_path):
    """Loads an image and converts from BGR to RGB."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class RecursiveRecognitionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Traverses a folder structured as:
        root_dir/
            upper_case/   -> contains subfolders for each character (e.g., A, B, ...)
            lower_case/   -> contains subfolders for each character (e.g., a, b, ...)
        """
        self.samples = []
        # Iterate over the case folders
        for case in ['upper_case', 'lower_case']:
            case_folder = os.path.join(root_dir, case)
            if not os.path.isdir(case_folder):
                continue
            # Iterate over each character folder within the case folder
            for char_label in os.listdir(case_folder):
                label_folder = os.path.join(case_folder, char_label)
                if not os.path.isdir(label_folder):
                    continue
                # Get all images from this subfolder
                for file in os.listdir(label_folder):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Full path of image
                        image_path = os.path.join(label_folder, file)
                        # Append tuple: (image_path, character label, case label)
                        self.samples.append((image_path, char_label, case))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, char_label, case_label = self.samples[idx]
        img = load_image(image_path)
        if self.transform:
            img = self.transform(img)
        # Convert image to tensor (assumes RGB image)
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Determine ground truth text:
        # If lower_case, use the label directly; if upper_case, convert to uppercase.
        # (This assumes subfolder names are already correct; adjust if needed.)
        text = char_label if case_label == 'lower_case' else char_label.upper()
        # Optionally, you can combine both if needed (e.g., "upper_A")
        # text = f"{case_label}_{char_label}"
        
        return img_tensor, text

# Example usage:
# dataset = RecursiveRecognitionDataset("datasets/train")
