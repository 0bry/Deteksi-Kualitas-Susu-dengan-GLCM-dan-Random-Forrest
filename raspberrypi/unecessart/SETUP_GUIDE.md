# Raspberry Pi 4 Milk Quality Detection Setup Guide

## Hardware Connections

### Components Required:
- Raspberry Pi 4
- USB Webcam or CSI Camera Module
- 16x2 LCD with I2C backpack
- Push button
- Jumper wires
- Breadboard (optional)

### Wiring Diagram:

**LCD (16x2 I2C):**
- VCC → 5V (Pin 2 or 4)
- GND → GND (Pin 6, 9, 14, 20, 25, 30, 34, 39)
- SDA → GPIO 2 (Pin 3)
- SCL → GPIO 3 (Pin 5)

**Push Button:**
- One terminal → GPIO 18 (Pin 12)
- Other terminal → GND (Pin 14)
- Note: Using internal pull-up resistor, no external resistor needed

**Camera:**
- USB Webcam → Any USB port
- OR CSI Camera → CSI camera connector

## Software Setup

### Step 1: Update Raspberry Pi OS
```bash
sudo apt update && sudo apt upgrade -y
```

### Step 2: Install System Dependencies
```bash
sudo apt install -y python3-pip python3-venv git
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt install -y libgtk-3-dev libpng-dev libjpeg-dev libopenexr-dev
sudo apt install -y libtiff-dev libwebp-dev
```

### Step 3: Enable I2C and Camera
```bash
sudo raspi-config
```
- Navigate to "Interfacing Options" or "Interface Options"
- Enable I2C
- Enable Camera (if using CSI camera)
- Reboot when prompted

### Step 4: Create Project Directory
```bash
mkdir ~/milk_quality_project
cd ~/milk_quality_project
```

### Step 5: Create Virtual Environment
```bash
python3 -m venv milk_env
source milk_env/bin/activate
```

### Step 6: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 7: Copy Required Files
Copy these files to your Raspberry Pi project directory:
- `milk_quality_detector.py`
- `requirements.txt`
- `random_forest_model_d4_a90.pkl` (from your training directory)

### Step 8: Test Hardware Connections

**Test Camera:**
```bash
python3 -c "import cv2; cap = cv2.VideoCapture(0); ret, frame = cap.read(); print('Camera OK' if ret else 'Camera Failed'); cap.release()"
```

**Test I2C LCD:**
```bash
sudo i2cdetect -y 1
```
You should see an address (usually 0x27 or 0x3f) showing your LCD.

**Test Button:**
```bash
python3 -c "import RPi.GPIO as GPIO; GPIO.setmode(GPIO.BCM); GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP); print(f'Button state: {GPIO.input(18)}'); GPIO.cleanup()"
```

## Running the Application

### Step 1: Activate Virtual Environment
```bash
cd ~/milk_quality_project
source milk_env/bin/activate
```

### Step 2: Run the Detector
```bash
python3 milk_quality_detector.py
```

### Step 3: Using the System
1. The LCD should display "Milk Quality Detector Ready"
2. Press the button to capture and analyze an image
3. Wait for processing (typically 2-5 seconds)
4. Results will be displayed on LCD:
   - Line 1: "Kualitas: [Baik/Rusak/Rusak Berat]"
   - Line 2: "Time: [X.X]s"
5. Images and results are automatically saved
6. Press Ctrl+C to stop the program

## Output Files

The system generates these files:
- `original_X_YYYYMMDD_HHMMSS.jpg` - Original captured images
- `processed_X_YYYYMMDD_HHMMSS.jpg` - Preprocessed images (cropped, grayscale, 1024x1024)
- `milk_quality_log.csv` - Log of all classifications with features and timing

## Troubleshooting

### Camera Issues:
- Check if camera is properly connected
- For USB camera: `lsusb` should show your camera
- For CSI camera: Check ribbon cable connection

### LCD Issues:
- Verify I2C is enabled: `sudo raspi-config`
- Check wiring connections
- Test I2C: `sudo i2cdetect -y 1`

### Button Issues:
- Check GPIO 18 connection
- Ensure button is connected to GND when pressed

### Permission Issues:
```bash
sudo usermod -a -G gpio,i2c,spi,camera $USER
```
Then logout and login again.

### Library Installation Issues:
If scikit-image fails to install:
```bash
pip install --no-build-isolation scikit-image
```

## Auto-Start on Boot (Optional)

To run automatically on startup:

1. Create service file:
```bash
sudo nano /etc/systemd/system/milk-detector.service
```

2. Add content:
```ini
[Unit]
Description=Milk Quality Detector
After=multi-user.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/milk_quality_project
Environment=PATH=/home/pi/milk_quality_project/milk_env/bin
ExecStart=/home/pi/milk_quality_project/milk_env/bin/python /home/pi/milk_quality_project/milk_quality_detector.py
Restart=always

[Install]
WantedBy=multi-user.target
```

3. Enable service:
```bash
sudo systemctl enable milk-detector.service
sudo systemctl start milk-detector.service
```

## Performance Notes

- Processing time: ~2-5 seconds per image
- Storage: ~1-2 MB per image pair
- RAM usage: ~200-300 MB
- Works best with good lighting conditions
- Crop coordinates can be manually adjusted in the code

## Model Information

- Using Random Forest classifier
- Features: GLCM correlation, contrast, homogeneity, energy, dissimilarity
- Distance: 4, Angle: 90 degrees
- Training accuracy: ~83.6%
- Classes: 1=Baik, 2=Rusak, 3=Rusak Berat
