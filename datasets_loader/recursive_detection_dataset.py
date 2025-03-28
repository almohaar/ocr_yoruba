import os
from torch.utils.data import Dataset
import torch
import cv2

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def create_full_image_box(img):
    """Creates a dummy bounding box covering the entire image (normalized)."""
    # Normalized center is (0.5, 0.5) and full width/height is 1.0
    return (0, 0.5, 0.5, 1.0, 1.0)  # Here 0 is a default class index

class RecursiveDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        for case in ['upper_case', 'lower_case']:
            case_folder = os.path.join(root_dir, case)
            if not os.path.isdir(case_folder):
                continue
            for char_label in os.listdir(case_folder):
                label_folder = os.path.join(case_folder, char_label)
                if not os.path.isdir(label_folder):
                    continue
                for file in os.listdir(label_folder):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(label_folder, file)
                        # You might also assign a class index based on case
                        # For example, 0 for lower_case, 1 for upper_case
                        class_index = 0 if case == 'lower_case' else 1
                        self.samples.append((image_path, class_index))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, class_index = self.samples[idx]
        img = load_image(image_path)
        if self.transform:
            img = self.transform(img)
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        box = create_full_image_box(img)  # Use dummy box covering full image
        # You can replace 0 in the box tuple with class_index if needed.
        # For now, we'll assume a single dummy label.
        return img_tensor, [ (class_index, box[1], box[2], box[3], box[4]) ]

# Example usage:
# detection_dataset = RecursiveDetectionDataset("datasets/train")
