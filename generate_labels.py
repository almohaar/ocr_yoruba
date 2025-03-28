import os

def generate_labels(dataset_dir, output_ext=".txt"):
    """
    Traverse the dataset directory and generate YOLO-format label files.
    Each label file will contain a single line:
        <class_index> 0.5 0.5 1.0 1.0
    Assumptions:
      - If the path contains "upper_case", the class index is 0.
      - If the path contains "lower_case", the class index is 1.
    Processes files with .jpg, .jpeg, or .png extensions.
    """
    supported_exts = {".jpg", ".jpeg", ".png"}
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in supported_exts:
                continue
            
            image_path = os.path.join(root, file)
            label_path = os.path.splitext(image_path)[0] + output_ext
            
            # Skip if label file already exists.
            if os.path.exists(label_path):
                continue
            
            # Determine class index from folder names
            if "upper_case" in root:
                class_idx = 0
            elif "lower_case" in root:
                class_idx = 1
            else:
                print(f"Skipping image with unknown class: {image_path}")
                continue
            
            # Create dummy label: full-image bounding box
            label_line = f"{class_idx} 0.5 0.5 1.0 1.0\n"
            
            with open(label_path, "w") as f:
                f.write(label_line)
            
            print(f"Generated label for: {image_path}")

if __name__ == "__main__":
    dataset_dir = os.path.join(os.getcwd(), "datasets")
    print("Generating labels for the entire dataset...")
    generate_labels(dataset_dir)
    print("Label generation complete!")
