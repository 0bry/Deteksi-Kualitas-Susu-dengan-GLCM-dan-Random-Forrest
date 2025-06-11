#!/usr/bin/env python3
"""
Hardware Test Script for Raspberry Pi Components
"""

import time
import sys

def test_imports():
    """Test if all required libraries can be imported"""
    print("Testing library imports...")
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
        
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
        
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
        
    try:
        import joblib
        print("✓ Joblib imported successfully")
    except ImportError as e:
        print(f"✗ Joblib import failed: {e}")
        return False
        
    try:
        from skimage.feature import graycomatrix, graycoprops
        print("✓ Scikit-image imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-image import failed: {e}")
        return False
        
    try:
        import RPi.GPIO as GPIO
        print("✓ RPi.GPIO imported successfully")
    except ImportError as e:
        print(f"✗ RPi.GPIO import failed: {e}")
        return False
        
    return True

def test_camera():
    """Test camera functionality"""
    print("\nTesting camera...")
    try:
        import cv2
        camera = cv2.VideoCapture(0)
        ret, frame = camera.read()
        camera.release()
        
        if ret:
            print("✓ Camera working - captured frame successfully")
            print(f"  Frame shape: {frame.shape}")
            return True
        else:
            print("✗ Camera failed to capture frame")
            return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_gpio():
    """Test GPIO setup"""
    print("\nTesting GPIO...")
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        button_state = GPIO.input(18)
        print(f"✓ GPIO setup successful")
        print(f"  Button state (GPIO 18): {'Pressed' if button_state == 0 else 'Not pressed'}")
        
        GPIO.cleanup()
        return True
    except Exception as e:
        print(f"✗ GPIO test failed: {e}")
        return False

def test_model():
    """Test model loading"""
    print("\nTesting model loading...")
    try:
        import joblib
        model = joblib.load("random_forest_model_d4_a90.pkl")
        print("✓ Model loaded successfully")
        print(f"  Model type: {type(model)}")
        return True
    except FileNotFoundError:
        print("✗ Model file not found (random_forest_model_d4_a90.pkl)")
        print("  Please copy the model file to this directory")
        return False
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_lcd():
    """Test LCD (optional)"""
    print("\nTesting LCD (optional)...")
    try:
        import board
        import busio
        from adafruit_character_lcd.character_lcd_rgb_i2c import Character_LCD_RGB_I2C
        
        i2c = busio.I2C(board.SCL, board.SDA)
        lcd = Character_LCD_RGB_I2C(i2c, 16, 2)
        lcd.clear()
        lcd.message = "Test OK"
        print("✓ LCD working - test message displayed")
        time.sleep(2)
        lcd.clear()
        return True
    except ImportError:
        print("⚠ LCD libraries not installed (optional)")
        return True
    except Exception as e:
        print(f"⚠ LCD test failed: {e} (optional)")
        return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("Raspberry Pi Hardware Test")
    print("=" * 50)
    
    tests = [
        ("Library Imports", test_imports),
        ("Camera", test_camera),
        ("GPIO", test_gpio),
        ("Model Loading", test_model),
        ("LCD (Optional)", test_lcd)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} {status}")
        if not result and "Optional" not in test_name:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("✓ All essential tests passed! Ready to run detector.")
    else:
        print("✗ Some tests failed. Please fix issues before running detector.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
