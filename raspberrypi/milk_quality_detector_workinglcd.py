#!/usr/bin/env python3
"""Simplified Milk Quality Detection System"""

import cv2
import RPi.GPIO as GPIO
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime
from RPLCD.i2c import CharLCD
from skimage.feature import graycomatrix, graycoprops
import joblib

# Configuration
BUTTON_PIN = 18
I2C_ADDR = 0x27
MODEL_PATH = "simple_rf_model_20250605_014338.pkl"
CSV_FILE = "milk_quality_results.csv"
PICTURES_DIR = "captured_images"
PROCESSED_DIR = "processed_images"

class MilkQualityDetector:
    def __init__(self):
        """Initialize hardware and load model."""
        self.camera = cv2.VideoCapture(0)
        self.lcd = CharLCD('PCF8574', I2C_ADDR, port=1, cols=16, rows=2)
        self.model = joblib.load(MODEL_PATH)
        self.quality_labels = {1: "Baik", 2: "Rusak", 3: "Rusak Berat"}
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        os.makedirs(PICTURES_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        # Initialize CSV
        if not os.path.exists(CSV_FILE):
            pd.DataFrame(columns=['timestamp', 'milk_quality', 'computation_time']).to_csv(CSV_FILE, index=False)
        
        self.show_lcd("Milk Quality", "Detector Ready")
    
    def show_lcd(self, line1, line2=""):
        """Display text on LCD."""
        self.lcd.clear()
        self.lcd.write_string(line1)
        if line2:
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string(line2)
            
            
    def process_image(self):
        """Capture and process image to grayscale 1024x1024."""
        ret, frame = self.camera.read()
        if not ret:
            return None, None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(PICTURES_DIR, f"milk_{timestamp}.jpg"), frame)
        
        # Crop to square and resize
        h, w = frame.shape[:2]
        size = min(h, w)
        start_x, start_y = (w - size) // 2, (h - size) // 2
        cropped = frame[start_y:start_y+size, start_x:start_x+size]
        resized = cv2.resize(cropped, (1024, 1024))
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Save processed image for inspection
        processed_path = os.path.join(PROCESSED_DIR, f"processed_{timestamp}.jpg")
        cv2.imwrite(processed_path, gray)
        
        return gray, timestamp
    
    def extract_features(self, image):
        """Extract GLCM features."""
        glcm = graycomatrix(image, [4], [np.pi/4], 256, symmetric=True, normed=True)
        return pd.DataFrame([{
            'Kontras': graycoprops(glcm, 'contrast')[0, 0],
            'Korelasi': graycoprops(glcm, 'correlation')[0, 0],
            'Homogenitas': graycoprops(glcm, 'homogeneity')[0, 0],
            'Disimilaritas': graycoprops(glcm, 'dissimilarity')[0, 0],
            'Energi': graycoprops(glcm, 'energy')[0, 0]
        }])    
    def analyze_milk(self):
        """Complete milk quality analysis process."""
        start_time = time.time()
        self.show_lcd("Processing...", "Please wait")
        
        # Process image
        image, timestamp = self.process_image()
        if image is None:
            self.show_lcd("Camera Error!")
            time.sleep(2)
            return
        
        # Extract features and predict
        features = self.extract_features(image)
        quality = self.model.predict(features)[0]
        quality_label = self.quality_labels[quality]
        computation_time = time.time() - start_time
        
        # Display and save results
        self.show_lcd("Kualitas susu:", quality_label)
        time.sleep(3)
        self.show_lcd("Waktu komputasi:", f"{computation_time:.2f}s")
        time.sleep(3)
        
        # Save to CSV
        pd.DataFrame([{
            'timestamp': timestamp,
            'milk_quality': quality,
            'computation_time': computation_time
        }]).to_csv(CSV_FILE, mode='a', header=False, index=False)
        
        self.show_lcd("Milk Quality", "Detector Ready")
    
    def run(self):
        """Main program loop."""
        try:
            print("Milk Quality Detector Started - Press button to analyze")
            while True:
                if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                    self.analyze_milk()
                    time.sleep(1)  # Debounce
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.camera.release()
            GPIO.cleanup()
            self.show_lcd("System", "Shutdown")

if __name__ == "__main__":
    detector = MilkQualityDetector()
    detector.run()
