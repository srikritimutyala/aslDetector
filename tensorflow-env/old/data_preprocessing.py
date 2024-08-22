import os
import cv2
import numpy as np

# Define paths
dataset_path = '../images/asl_alphabet_test'
categories = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
img_size = 64

# Data storage
data = []
labels = []

# Define a mapping from filenames to labels if you have a specific mapping
label_map = {
    "A_test1.jpg": 0,
    "A_test2.jpg": 0,
    # Add other mappings here
}

# Load images and labels
for img in categories:
    img_path = os.path.join(dataset_path, img)
    print(f"Processing image: {img_path}")

    try:
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_array is not None:
            resized_array = cv2.resize(img_array, (img_size, img_size))
            data.append(resized_array)
            # Use the filename to get the label
            label = label_map.get(img, -1)  # Default to -1 if not found
            labels.append(label)
        else:
            print(f"Failed to load image: {img_path}")
    except Exception as e:
        print(f"Error processing image {img}: {e}")

# Convert to numpy arrays
data = np.array(data).reshape(-1, img_size, img_size, 1)
labels = np.array(labels)

print(f"Loaded {len(data)} images.")
