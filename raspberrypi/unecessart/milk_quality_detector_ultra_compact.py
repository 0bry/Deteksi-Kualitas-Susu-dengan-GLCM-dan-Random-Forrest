#!/usr/bin/env python3
"""Ultra-compact milk quality detector for Raspberry Pi"""

import cv2, numpy as np, pandas as pd, joblib, time, os, warnings
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import RPi.GPIO as GPIO

try:
    import board, busio
    from adafruit_character_lcd.character_lcd_rgb_i2c import Character_LCD_RGB_I2C
    LCD_OK = True
except: LCD_OK = False

warnings.filterwarnings('ignore')

class MilkDetector:
    def __init__(self):
        # Constants
        self.DIST, self.ANG, self.BTN = 4, 90, 18
        self.QUAL = {'1': 'Baik', '2': 'Rusak', '3': 'Rusak Berat'}
        self.FEATURES = ['correlation', 'contrast', 'homogeneity', 'energy', 'dissimilarity']
        
        # Setup
        for d in ['ori', 'pro']: os.makedirs(d, exist_ok=True)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.BTN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Hardware init
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 1920); self.cam.set(4, 1080)
        self.model = joblib.load("model.pkl")
        self.counter = 1
        
        # LCD setup
        self.lcd = None
        if LCD_OK:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self.lcd = Character_LCD_RGB_I2C(i2c, 16, 2)
                self.show("Milk Detector", "Ready")
            except: pass
        
        print("Ultra-compact detector ready")

    def show(self, l1, l2=""):
        msg = f"{l1} | {l2}" if l2 else l1
        print(msg)
        if self.lcd:
            try: self.lcd.clear(); self.lcd.message = f"{l1}\n{l2}"
            except: pass

    def detect(self):
        self.show("Capturing...")
        t = time.time()
        
        # Capture
        ret, img = self.cam.read()
        if not ret: raise Exception("Capture failed")
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"ori/img_{self.counter}_{ts}.jpg", img)
        
        # Process
        h, w = img.shape[:2]
        s = min(h, w) // 2
        cx, cy = w // 2, h // 2
        crop = img[cy-s//2:cy+s//2, cx-s//2:cx+s//2]
        gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (1024, 1024))
        cv2.imwrite(f"pro/proc_{self.counter}_{ts}.jpg", gray)
        
        # GLCM
        gu = (img_as_ubyte(gray) // 4) * 4
        glcm = graycomatrix(gu, [self.DIST], [np.radians(self.ANG)], 256, True, True)
        feats = {f: graycoprops(glcm, f)[0,0] for f in self.FEATURES}
        
        # Predict
        df = pd.DataFrame([[feats[f] for f in self.FEATURES]], columns=self.FEATURES)
        qual = self.QUAL[str(self.model.predict(df)[0])]
        
        dt = round(time.time() - t, 2)
        self.show(f"Quality: {qual}", f"Time: {dt}s")
        
        # Log
        log = {'number': self.counter, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               **feats, 'quality': qual, 'time': dt}
        pd.DataFrame([log]).to_csv("log.csv", mode='a', header=not os.path.exists("log.csv"), index=False)
        
        self.counter += 1

    def run(self):
        print("Press button to detect quality")
        try:
            while True:
                if GPIO.input(self.BTN) == GPIO.LOW:
                    try: self.detect()
                    except Exception as e: self.show("Error", str(e)[:10])
                    time.sleep(1)
                    while GPIO.input(self.BTN) == GPIO.LOW: time.sleep(0.1)
                    self.show("Ready", "Press button")
                time.sleep(0.1)
        except KeyboardInterrupt: print("\nExit")
        finally: self.cleanup()

    def cleanup(self):
        self.cam.release()
        if self.lcd: self.lcd.clear()
        GPIO.cleanup()

if __name__ == "__main__": MilkDetector().run()
