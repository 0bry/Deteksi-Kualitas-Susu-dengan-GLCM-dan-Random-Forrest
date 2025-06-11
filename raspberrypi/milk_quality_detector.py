import cv2
import numpy as np
import pandas as pd
import joblib
import time
import os
import warnings
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import RPi.GPIO as GPIO

# LCD handling using RPLCD
try:
    from RPLCD.i2c import CharLCD
    LCD_AVAILABLE = True
except ImportError:
    LCD_AVAILABLE = False

# Configuration constants
DISTANCE, ANGLE, BUTTON_PIN = 4, 90, 18
I2C_ADDR = 0x27
MODEL_PATH, CSV_LOG = "model.pkl", "milk_quality_log.csv"
QUALITY_MAP = {'1': 'Baik', '2': 'Rusak', '3': 'Rusak Berat'}

class MilkQualityDetector:
    def __init__(self):
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
        os.makedirs("ori", exist_ok=True)
        os.makedirs("pro", exist_ok=True)
          # Setup hardware
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Setup LCD
        self.lcd = None
        if LCD_AVAILABLE:
            try:
                self.lcd = CharLCD('PCF8574', I2C_ADDR, port=1, cols=16, rows=2)
                self.lcd.clear()
                self.lcd.write_string("Milk Quality")
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string("Detector Ready")
            except:
                self.lcd = None
        
        # Setup camera and model
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.model = joblib.load(MODEL_PATH)
        self.image_counter = 1
        print("Milk Quality Detector Ready")
        
    def display_message(self, line1, line2=""):
        message = f"{line1}\n{line2}" if line2 else line1
        print(message.replace('\n', ' | '))
        if self.lcd:
            try:
                self.lcd.clear()
                self.lcd.write_string(line1)
                if line2:
                    self.lcd.cursor_pos = (1, 0)
                    self.lcd.write_string(line2)
            except:
                pass

    def process_detection(self):
        self.display_message("Capturing...", "Please wait")
        start_time = time.time()
        
        # Capture and save original image
        ret, frame = self.camera.read()
        if not ret:
            raise Exception("Failed to capture image")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_path = f"ori/original_{self.image_counter}_{timestamp}.jpg"
        cv2.imwrite(original_path, frame)
        
        # Preprocess image
        h, w = frame.shape[:2]
        crop_size = min(h, w) // 2
        center_x, center_y = w // 2, h // 2
        x1, y1 = center_x - crop_size // 2, center_y - crop_size // 2
        x2, y2 = center_x + crop_size // 2, center_y + crop_size // 2
        
        cropped = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (1024, 1024))
        
        processed_path = f"pro/processed_{self.image_counter}_{timestamp}.jpg"
        cv2.imwrite(processed_path, resized)
        
        # Extract GLCM features
        gray_ubyte = img_as_ubyte(resized)
        gray_ubyte = (gray_ubyte // 4) * 4
        glcm = graycomatrix(gray_ubyte, [DISTANCE], [np.radians(ANGLE)], 
                           levels=256, symmetric=True, normed=True)
        
        features = {prop: graycoprops(glcm, prop)[0, 0] 
                   for prop in ['correlation', 'contrast', 'homogeneity', 'energy', 'dissimilarity']}
        
        # Classify quality
        feature_names = ['correlation', 'contrast', 'homogeneity', 'energy', 'dissimilarity']
        feature_df = pd.DataFrame([[features[f] for f in feature_names]], columns=feature_names)
        quality = str(self.model.predict(feature_df)[0])
        quality_text = QUALITY_MAP[quality]
        
        comp_time = round(time.time() - start_time, 2)
        self.display_message(f"Kualitas: {quality_text}", f"Time: {comp_time}s")
        
        # Log results
        log_data = {
            'number': self.image_counter,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **features,
            'quality': quality_text,
            'computational_time': comp_time        }
        
        df = pd.DataFrame([log_data])
        df.to_csv(CSV_LOG, mode='a', header=not os.path.exists(CSV_LOG), index=False)
        print(f"Images saved: {original_path}, {processed_path}")
        self.image_counter += 1
        
    def run(self):
        print("Milk Quality Detector Started")
        print("Press button to capture and analyze")
        self.display_message("Press button", "to detect milk")
        
        try:
            while True:
                if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                    self.process_detection()
                    time.sleep(2)  # Display results for 2 seconds
                    
                    while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                        time.sleep(0.1)
                        
                    self.display_message("Press button", "to detect milk")
                    
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()
            
    def cleanup(self):
        if hasattr(self, 'camera'):
            self.camera.release()
        if self.lcd:
            self.lcd.clear()
        GPIO.cleanup()

if __name__ == "__main__":
    detector = MilkQualityDetector()
    detector.run()
