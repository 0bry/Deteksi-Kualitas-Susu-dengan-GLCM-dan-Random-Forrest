# Raspberry Pi Milk Quality Detection System

This folder contains the complete Raspberry Pi implementation for real-time milk quality detection using GLCM features and Random Forest classification.

## Files Description

- **`milk_quality_detector.py`** - Main application with LCD display support
- **`simple_detector.py`** - Simplified version (console output only) for testing
- **`test_hardware.py`** - Hardware testing script to verify all components
- **`requirements.txt`** - Full Python dependencies including LCD libraries
- **`requirements_minimal.txt`** - Minimal dependencies for testing without LCD
- **`SETUP_GUIDE.md`** - Complete setup and installation guide
- **`copy_model.sh`** - Script to copy the trained model
- **`random_forest_model_d4_a90.pkl`** - Trained Random Forest model

## Hardware Requirements

### Components:
- Raspberry Pi 4 (recommended) or Pi 3B+
- USB Webcam or CSI Camera Module
- 16x2 LCD with I2C backpack
- Push button
- Jumper wires
- MicroSD card (16GB+ recommended)

### Wiring:
```
LCD (I2C):
- VCC → 5V (Pin 2)
- GND → GND (Pin 6)  
- SDA → GPIO 2 (Pin 3) - Connected to pin 12
- SCL → GPIO 3 (Pin 5) - Connected to pin 14

Push Button:
- GPIO 18 (Pin 12) → Button → GND (Pin 14)
- Uses internal pull-up resistor

Camera:
- USB Camera → USB port (confirmed working)
- CSI Camera → CSI connector
```

## Recent Updates & Fixes ✅

### Issues Resolved:
1. **Feature Names Warning** - Fixed sklearn compatibility issue
2. **Directory Organization** - Images now saved to 'ori' and 'pro' folders
3. **Model Path** - Updated to use `model.pkl` consistently
4. **LCD Support** - Enhanced with proper fallback to console
5. **Warning Suppression** - Cleaner output with error handling

### Performance Improvements:
- Processing time: 0.17-1.2 seconds per detection
- Organized file structure for better management
- Enhanced error handling and user feedback

## Quick Start

1. **Setup Hardware** - Connect components as per wiring diagram
2. **Test Components** - Run `python3 test_hardware.py`
3. **Install Dependencies** - `pip install -r requirements.txt`
4. **Run Application** - `python3 milk_quality_detector.py`

For detailed installation instructions, see `INSTALLATION_GUIDE.md`.

## Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: GLCM (Gray-Level Co-occurrence Matrix)
  - Correlation
  - Contrast  
  - Homogeneity
  - Energy
  - Dissimilarity
- **Parameters**: Distance=4, Angle=90°
- **Classes**: 
  - 1 = Baik (Good)
  - 2 = Rusak (Damaged)
  - 3 = Rusak Berat (Heavily Damaged)
- **Accuracy**: ~83.6%

## System Workflow

1. **Wait** - System waits for button press
2. **Capture** - Takes photo with camera
3. **Preprocess** - Crops center, converts to grayscale, resizes to 1024x1024
4. **Extract** - Calculates GLCM features
5. **Classify** - Uses Random Forest to predict quality
6. **Display** - Shows result on LCD and saves to CSV
7. **Log** - Saves images and data for analysis

## Output Files

- `ori/original_X_YYYYMMDD_HHMMSS.jpg` - Original captured images (organized)
- `pro/processed_X_YYYYMMDD_HHMMSS.jpg` - Preprocessed images (organized)
- `milk_quality_log.csv` - Classification results with timestamps and features

## Performance

- **Processing Time**: 2-5 seconds per detection
- **Storage**: ~1-2 MB per image pair
- **RAM Usage**: ~200-300 MB
- **Power**: Works with standard Pi 4 power supply

## Troubleshooting

See `SETUP_GUIDE.md` for detailed troubleshooting steps.

Common issues:
- Camera not detected: Check USB connection or CSI cable
- LCD not working: Verify I2C is enabled and wiring
- Button not responding: Check GPIO 18 connection to GND
- Import errors: Install dependencies with pip

## Development Notes

- Crop coordinates can be manually adjusted in preprocessing function
- GLCM parameters (distance/angle) can be modified at top of file
- Additional camera settings can be configured in setup_camera()
- LCD messages can be customized in display_message() function

For complete setup instructions, see `SETUP_GUIDE.md`.
