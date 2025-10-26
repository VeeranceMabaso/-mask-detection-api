import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def explore_dataset(data_path):
    """Explore the dataset structure and statistics"""
    
    classes = ['with_mask', 'without_mask']
    
    print("Dataset Exploration Report")
    print("=" * 40)
    
    # Count images per class
    for class_name in classes:
        class_path = os.path.join(data_path, class_name)
        if os.path.exists(class_path):
            num_images = len([f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            print(f"{class_name}: {num_images} images")
        else:
            print(f"{class_name} folder not found!")
    
    # Check image dimensions
    print("\n Checking image dimensions...")
    dimensions = []
    sample_class = classes[0]
    sample_path = os.path.join(data_path, sample_class)
    
    if os.path.exists(sample_path):
        sample_images = [f for f in os.listdir(sample_path) if f.endswith(('.png', '.jpg', '.jpeg'))][:5]
        
        for img_name in sample_images:
            img_path = os.path.join(sample_path, img_name)
            with Image.open(img_path) as img:
                dimensions.append(img.size)
                print(f"   {img_name}: {img.size} | Mode: {img.mode}")

if __name__ == "__main__":
    explore_dataset("../data/raw")