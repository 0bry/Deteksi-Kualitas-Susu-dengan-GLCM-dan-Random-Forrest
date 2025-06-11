import os
import cv2
from pathlib import Path

def process_images():
    os.makedirs("processed_data", exist_ok=True)
    
    count = 0
    
    for root, dirs, files in os.walk("raw_data"):
        rel_path = os.path.relpath(root, "raw_data")
        output_dir = os.path.join("processed_data", rel_path)
        os.makedirs(output_dir, exist_ok=True)
        
for filename in files:
    if filename.lower().endswith(('.jpg', '.jpeg')):
        img = cv2.imread(os.path.join(root, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (1024, 1024))
        cv2.imwrite(os.path.join(output_dir, filename), resized)
        count += 1
    
    print(f"{count} images processed")

if __name__ == "__main__":
    process_images()