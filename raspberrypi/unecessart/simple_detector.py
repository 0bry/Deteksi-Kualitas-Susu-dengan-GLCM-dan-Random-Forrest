#!/usr/bin/env python3
"""
Simplified Milk Quality Detector (Console Output Only)
For testing without LCD hardware
"""

import cv2
import numpy as np
import pandas as pd
import joblib
import time
import os
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import RPi.GPIO as GPIO

# Configuration
DISTANCE = 4  # GLCM distance parameter
ANGLE = 90    # GLCM angle parameter (in degrees)
BUTTON_PIN = 18
MODEL_PATH = "random_forest_model_d4_a90.pkl"
CSV_LOG = "milk_quality_log.csv"
QUALITY_MAP = {'1': 'Baik', '2': 'Rusak', '3': 'Rusak Berat'}

class SimpleMilkDetector:
    def __init__(self):
        self.setup_gpio()
        self.load_model()
        self.setup_camera()
        self.image_counter = 1
        
    def setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
    def setup_camera(self):
        self.camera = cv2.VideoCapture(0)
        
    def load_model(self):
        self.model = joblib.load(MODEL_PATH)
        print("Model loaded successfully")
        
    def capture_image(self):
        ret, frame = self.camera.read()
        if not ret:
            raise Exception("Failed to capture image")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_path = f"original_{self.image_counter}_{timestamp}.jpg"
        cv2.imwrite(original_path, frame)
        return frame, original_path
        
    def preprocess_image(self, image, original_path):
        h, w = image.shape[:2]
        crop_size = min(h, w) // 2
        center_x, center_y = w // 2, h // 2
        x1, y1 = center_x - crop_size // 2, center_y - crop_size // 2
        x2, y2 = center_x + crop_size // 2, center_y + crop_size // 2
        
        cropped = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (1024, 1024))
        
        processed_path = original_path.replace("original_", "processed_")
        cv2.imwrite(processed_path, resized)
        return resized, processed_path
        
    def extract_glcm_features(self, image):
        gray = img_as_ubyte(image)
        gray = (gray // 4) * 4
        angle_rad = np.radians(ANGLE)
        glcm = graycomatrix(gray, [DISTANCE], [angle_rad], levels=256, symmetric=True, normed=True)
        
        features = {}
        for prop in ['correlation', 'contrast', 'homogeneity', 'energy', 'dissimilarity']:
            features[prop] = graycoprops(glcm, prop)[0, 0]
        return features
        
    def classify_quality(self, features):
        feature_array = np.array([[features[f] for f in 
                                 ['correlation', 'contrast', 'homogeneity', 'energy', 'dissimilarity']]])
        return str(self.model.predict(feature_array)[0])
        
    def log_results(self, features, quality, comp_time):
        log_data = {
            'number': self.image_counter,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **features,
            'quality': QUALITY_MAP[quality],
            'computational_time': comp_time
        }
        
        df = pd.DataFrame([log_data])
        if os.path.exists(CSV_LOG):
            df.to_csv(CSV_LOG, mode='a', header=False, index=False)
        else:
            df.to_csv(CSV_LOG, index=False)
            
    def process_detection(self):
        try:
            print("Capturing and processing...")
            start_time = time.time()
            
            image, original_path = self.capture_image()
            processed_image, processed_path = self.preprocess_image(image, original_path)
            features = self.extract_glcm_features(processed_image)
            quality = self.classify_quality(features)
            comp_time = round(time.time() - start_time, 2)
            
            print(f"Kualitas susu: {QUALITY_MAP[quality]}")
            print(f"Computational Time: {comp_time}s")
            print(f"Images: {original_path}, {processed_path}")
            
            self.log_results(features, quality, comp_time)
            self.image_counter += 1
            
        except Exception as e:
            print(f"Error: {e}")
            
    def run(self):
        print("Simple Milk Quality Detector Started")
        print("Press button to capture and analyze (Ctrl+C to quit)")
        
        try:
            while True:
                if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                    self.process_detection()
                    time.sleep(1)
                    while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                        time.sleep(0.1)
                    print("Ready for next detection...")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.camera.release()
            GPIO.cleanup()

if __name__ == "__main__":
    detector = SimpleMilkDetector()
    detector.run()
