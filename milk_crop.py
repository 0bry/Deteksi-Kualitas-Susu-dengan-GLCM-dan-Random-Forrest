import cv2 # untuk pemrosesan gambar
import numpy as np # untuk manipulasi array
import pandas as pd # untuk manipulasi data dan penyimpanan ke CSV
import joblib # untuk menyimpan dan memuat model
import time # untuk menangani waktu
import os # untuk menangani direktori dan file
import warnings # untuk menangani peringatan
from datetime import datetime # untuk menangani tanggal dan waktu
from skimage.feature import graycomatrix, graycoprops # untuk menghitung GLCM dan ekstraksi fitur
from skimage import img_as_ubyte # untuk mengubah gambar menjadi tipe data uint8
import RPi.GPIO as GPIO # untuk menangani GPIO pada Raspberry Pi

#memastikan bahwa RPLCD.i2c tersedia
try:
    from RPLCD.i2c import CharLCD
    LCD_AVAILABLE = True
except ImportError:
    LCD_AVAILABLE = False

# konfigurasi
DISTANCE, ANGLE, = 4, 90 # jarak dan sudut untuk GLCM, pin untuk tombol
BUTTON_PIN = 18 # GPIO pin untuk tombol
I2C_ADDR = 0x27 # I2C address untuk LCD
MODEL_PATH, CSV_LOG = "model.pkl", "milk_quality_log.csv" # path untuk model dan log CSV
QUALITY_MAP = {'1': 'Baik', '2': 'Rusak', '3': 'Rusak Berat'} # mapping kualitas susu

class MilkQualityDetector: #membuat kelas untuk detektor kualitas susu
    def __init__(self):
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn') 
        os.makedirs("ori", exist_ok=True) #membuat direktori untuk menyimpan gambar asli
        os.makedirs("pro", exist_ok=True) #membuat direktori untuk menyimpan gambar yang telah diproses
        GPIO.setmode(GPIO.BCM) # menggunakan BCM pin numbering
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP) # mengatur pin tombol sebagai input dengan pull-up resistor
        
        self.lcd = None # inisialisasi LCD sebagai None
        if LCD_AVAILABLE: # cek apakah RPLCD.i2c tersedia
            try:
                self.lcd = CharLCD('PCF8574', I2C_ADDR, port=1, cols=16, rows=2) # inisialisasi LCD
                self.lcd.clear() # membersihkan LCD
                self.lcd.write_string("Milk Quality") # menulis pesan awal ke LCD
                self.lcd.cursor_pos = (1, 0) # mengatur posisi kursor ke baris kedua
                self.lcd.write_string("Detector Ready") # menulis pesan kedua ke LCD
            except:
                self.lcd = None
        
        #kamera
        self.camera = cv2.VideoCapture(0) # inisialisasi kamera
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # mengatur lebar frame kamera
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # mengatur tinggi frame kamera
        self.model = joblib.load(MODEL_PATH) # memuat model dari file
        self.image_counter = 1 # inisialisasi penghitung gambar
        print("Milk Quality Detector Ready") # menampilkan pesan bahwa detektor siap
        
    def display_message(self, line1, line2=""): # menampilkan pesan pada LCD dan terminal
        message = f"{line1}\n{line2}" if line2 else line1 # format pesan
        print(message.replace('\n', ' | ')) # menampilkan pesan di terminal
        if self.lcd: # jika LCD tersedia, tampilkan pesan di LCD
            try:
                self.lcd.clear() # membersihkan LCD
                self.lcd.write_string(line1) # menulis baris pertama ke LCD
                if line2: # jika ada baris kedua, tampilkan juga
                    self.lcd.cursor_pos = (1, 0) # mengatur posisi kursor ke baris kedua
                    self.lcd.write_string(line2) # menulis baris kedua ke LCD
            except:
                pass

    def process_detection(self): # proses deteksi kualitas susu
        self.display_message("Capturing...", "Please wait") 
        start_time = time.time() # mencatat waktu mulai proses
        
        ret, frame = self.camera.read() # menangkap frame dari kamera
        if not ret: # jika tidak berhasil menangkap frame, tampilkan pesan error
            raise Exception("Failed to capture image") 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # mendapatkan timestamp untuk penamaan file
        original_path = f"ori/original_{self.image_counter}_{timestamp}.jpg" # path untuk menyimpan gambar asli
        cv2.imwrite(original_path, frame) # menyimpan gambar asli ke file
        
        #cropping dengan koordinat tetap
        x1, y1 = 657, 370  # top-left corner
        x2, y2 = 1171, 884  # bottom-right corner
        
        cropped = frame[y1:y2, x1:x2] # melakukan cropping pada frame
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) # mengubah gambar ke grayscale
        resized = cv2.resize(gray, (1024, 1024)) # mengubah ukuran gambar ke 1024x1024
        
        processed_path = f"pro/processed_{self.image_counter}_{timestamp}.jpg" # path untuk menyimpan gambar yang telah diproses
        cv2.imwrite(processed_path, resized) # menyimpan gambar yang telah diproses ke file
        
        #ekstraksi fitur GLCM, (sesuai dengan model yang telah dilatih)
        gray_ubyte = img_as_ubyte(resized) # mengubah gambar grayscale menjadi tipe data uint8 (0-255)
        gray_ubyte = (gray_ubyte // 4) * 4 # mengurangi nilai piksel menjadi kelipatan 4 untuk mengurangi noise
        glcm = graycomatrix(gray_ubyte, [DISTANCE], [np.radians(ANGLE)], # menghitung matriks GLCM
                           levels=256, symmetric=True, normed=True) #menggunakan jarak dan sudut yang telah ditentukan
        
        features = {prop: graycoprops(glcm, prop)[0, 0] # mengembalikan fitur GLCM yang diinginkan
                   for prop in ['correlation', 'contrast', 'homogeneity', 'energy', 'dissimilarity']} 
        
        #klasifikasi kualitas susu menggunakan model yang telah dilatih
        feature_names = ['correlation', 'contrast', 'homogeneity', 'energy', 'dissimilarity'] # daftar nama fitur yang digunakan dalam model
        feature_df = pd.DataFrame([[features[f] for f in feature_names]], columns=feature_names) # membuat DataFrame dari fitur yang diekstrak
        quality = str(self.model.predict(feature_df)[0]) # melakukan prediksi kualitas susu menggunakan model
        quality_text = QUALITY_MAP[quality] # mendapatkan teks kualitas susu berdasarkan prediksi
        
        comp_time = round(time.time() - start_time, 2) # menghitung waktu komputasi
        self.display_message(f"Kualitas: {quality_text}", f"Time: {comp_time}s") # menampilkan hasil di LCD dan terminal
        
        # Log results
        log_data = { # menyimpan data log
            'number': self.image_counter, # nomor gambar
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # timestamp deteksi
            **features, # fitur GLCM yang diekstrak
            'quality': quality_text, # kualitas susu yang diprediksi
            'computational_time': comp_time # waktu komputasi
        }

        df = pd.DataFrame([log_data]) # membuat DataFrame dari data log
        df.to_csv(CSV_LOG, mode='a', header=not os.path.exists(CSV_LOG), index=False) # menyimpan
        print(f"Images saved: {original_path}, {processed_path}") # menyimpan data log ke file CSV
        self.image_counter += 1 # menambah penghitung gambar
        
    def run(self):
        print("Milk Quality Detector Started") # menampilkan pesan bahwa detektor telah dimulai
        print("Press button to capture and analyze") # menampilkan instruksi untuk menekan tombol
        self.display_message("Press button", "to detect milk") # menampilkan pesan awal di LCD
        
        try:
            while True:
                if GPIO.input(BUTTON_PIN) == GPIO.LOW: # jika tombol ditekan
                    self.process_detection() # memproses deteksi kualitas susu
                    time.sleep(2)  # Display results for 2 seconds
                    
                    while GPIO.input(BUTTON_PIN) == GPIO.LOW:# menunggu tombol dilepas
                        time.sleep(0.1)# untuk menghindari pembacaan berulang (bounce)
                        
                    self.display_message("Press button", "to detect milk") # menampilkan pesan untuk menekan tombol lagi
                    
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nShutting down...") 
        finally:
            self.cleanup()
            
    def cleanup(self):
        if hasattr(self, 'camera'): # jika kamera telah diinisialisasi
            self.camera.release() # melepaskan kamera
        if self.lcd:
            self.lcd.clear() # membersihkan tampilan LCD
        GPIO.cleanup()

if __name__ == "__main__":
    detector = MilkQualityDetector() # inisialisasi detektor kualitas susu
    detector.run()
