import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

def extract_glcm_features(image_path, distances, angles):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = img_as_ubyte(gray)
        gray = (gray // 4) * 4
        
        features = {}
        properties = ['correlation', 'contrast', 'homogeneity', 'energy', 'dissimilarity']
        
        for distance in distances:
            for angle in angles:
                glcm = graycomatrix(gray, [distance], [angle], levels=256, symmetric=True, normed=True)
                angle_deg = int(np.degrees(angle))
                
                for prop in properties:
                    key = f"d{distance}_a{angle_deg}_{prop}"
                    features[key] = graycoprops(glcm, prop)[0, 0]
                
        return features
    except:
        return None

def process_images_in_folder(folder_path, class_label, distances, angles):
    image_extensions = ('.jpg', '.jpeg')
    results = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(image_extensions):
            features = extract_glcm_features(os.path.join(folder_path, filename), distances, angles)
            if features:
                features['class'] = class_label
                results.append(features)
    
    return results

def create_csv_files(all_results, distances, angles, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    properties = ['correlation', 'contrast', 'homogeneity', 'energy', 'dissimilarity']
    
    for distance in distances:
        for angle in angles:
            angle_deg = int(np.degrees(angle))
            
            # Create DataFrame directly from filtered data
            data = []
            for result in all_results:
                row = {'class': result['class']}
                for prop in properties:
                    key = f"d{distance}_a{angle_deg}_{prop}"
                    if key in result:
                        row[prop] = result[key]
                if len(row) > 1:  # Has more than just 'class'
                    data.append(row)
            
            if data:
                df = pd.DataFrame(data)
                csv_path = os.path.join(output_dir, f"glcm_features_distance_{distance}_angle_{angle_deg}.csv")
                df.to_csv(csv_path, index=False)
                print(f"The features of range {distance} and angle {angle_deg} is saved")

def main():
    data_dir = r"G:\Project\Deteksi-Kualitas-Susu-dengan-GLCM-dan-Random-Forrest\data"
    output_dir = r"G:\Project\Deteksi-Kualitas-Susu-dengan-GLCM-dan-Random-Forrest\features"
    
    distances = [1, 2, 3, 4]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    all_results = []
    for class_folder in ['1', '2', '3']:
        folder_path = os.path.join(data_dir, class_folder)
        if os.path.exists(folder_path):
            all_results.extend(process_images_in_folder(folder_path, class_folder, distances, angles))
    
    create_csv_files(all_results, distances, angles, output_dir)

if __name__ == "__main__":
    main()
