import os
import cv2
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def preprocess_image(image_path, output_path, target_size=(128, 128)):
    """Preprocess a single image by resizing and normalizing.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save processed image
        target_size (tuple): Target image dimensions (height, width)
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return False
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save processed image
    cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return True

def preprocess_dataset(input_dir, output_dir):
    """Preprocess entire PlantVillage dataset.
    
    Args:
        input_dir (str): Path to raw dataset directory
        output_dir (str): Path to save processed dataset
    """
    start_time = time.time()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get all image files
    image_files = list(input_path.glob('**/*.jpg')) + list(input_path.glob('**/*.jpeg')) + list(input_path.glob('**/*.png'))
    total_images = len(image_files)
    
    print(f"Found {total_images} images to process")
    
    # Process each image
    processed_count = 0
    for image_file in tqdm(image_files, desc="Processing images"):
        # Create relative path for output
        rel_path = image_file.relative_to(input_path)
        output_file = output_path / rel_path
        
        if preprocess_image(image_file, output_file):
            processed_count += 1
    
    processing_time = time.time() - start_time
    print(f"Successfully processed {processed_count} images in {processing_time:.2f} seconds")
    
    # Save metrics
    metrics = {
        "total_images": total_images,
        "processed_images": processed_count,
        "processing_time": processing_time,
        "timestamp": datetime.now().isoformat()
    }
    
    metrics_dir = os.path.join('metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, 'preprocessing.json')
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return processed_count

def main():
    # Define paths
    raw_data_dir = os.path.join('data', 'PlantVillage')
    processed_data_dir = os.path.join('data', 'processed_PlantVillage')
    
    # Process dataset
    if os.path.exists(raw_data_dir):
        processed_count = preprocess_dataset(raw_data_dir, processed_data_dir)
        print(f"Dataset preprocessing completed. {processed_count} images processed.")
    else:
        print(f"Error: Raw dataset directory not found at {raw_data_dir}")

if __name__ == "__main__":
    main()