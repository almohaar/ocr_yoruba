import os
import cv2
import numpy as np

def load_image(image_path):
    """Load an image from disk and convert it from BGR to RGB."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def create_full_image_box(img):
    """
    Create a dummy bounding box that covers the full image.
    Returns a tuple: (class, x_center, y_center, width, height)
    Coordinates are normalized.
    """
    # For full image: center (0.5, 0.5) and size 1.0
    return (0, 0.5, 0.5, 1.0, 1.0)

def extract_text_from_filename(filename):
    """
    Extract ground truth text from the filename.
    Example: "a_yé.jpg" becomes "a yé"
    """
    name = os.path.splitext(filename)[0]
    return name.replace('_', ' ')

def augment_image(image):
    """
    Apply data augmentation.
    You can expand this function with rotations, scaling, etc.
    For now, we return the image as-is.
    """
    return image

def split_dataset(image_folder, split_ratio=(0.7, 0.15, 0.15)):
    """
    Split dataset into train, val, and test lists based on split_ratio.
    Returns three lists of image file paths.
    """
    from glob import glob
    image_files = sorted(glob(os.path.join(image_folder, '*.*')))
    total = len(image_files)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])
    return image_files[:train_end], image_files[train_end:val_end], image_files[val_end:]
