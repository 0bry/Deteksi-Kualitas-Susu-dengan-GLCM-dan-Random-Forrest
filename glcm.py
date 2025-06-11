import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

distance = 1
angle = 0

def extract_glcm_features(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = img_as_ubyte(gray)
        gray = (gray // 4) * 4
        
        glcm = graycomatrix(gray, [distance], [angle], levels=256, symmetric=True, normed=True)
        
        return {prop: graycoprops(glcm, prop)[0, 0] 
                for prop in ['correlation', 'contrast', 'homogeneity', 'energy', 'dissimilarity']}
    except:
        return None

def process_images_in_folder(folder_path, class_label):
    extensions = ('.jpg', '.jpeg')
    results = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(extensions):
            features = extract_glcm_features(os.path.join(folder_path, filename))
            if features:
                results.append({'class': class_label, **features})
    
    return results

def main():
    data_dir = r"G:\Project\Deteksi-Kualitas-Susu-dengan-GLCM-dan-Random-Forrest\data"
    
    all_results = []
    for class_folder in ['1', '2', '3']:
        folder_path = os.path.join(data_dir, class_folder)
        if os.path.exists(folder_path):
            all_results.extend(process_images_in_folder(folder_path, class_folder))
    
    if all_results:
        os.makedirs('features', exist_ok=True)
        pd.DataFrame(all_results).to_csv('features/features.csv', index=False)
        print("Features saved to features/features.csv")

if __name__ == "__main__":
    main()
