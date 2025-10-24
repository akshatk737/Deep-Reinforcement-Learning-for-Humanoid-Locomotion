import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read the image from {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_resized = cv2.resize(image_rgb, target_size)

    image_normalized = image_resized / 255.0

    image_array = np.asarray(image_normalized, dtype=np.float32)

    return image_array

image_path = "images/single1.jpg"
image_array = load_and_preprocess_image(image_path)

print("Image as NumPy array:\n", image_array)
print("\nShape:", image_array.shape)
