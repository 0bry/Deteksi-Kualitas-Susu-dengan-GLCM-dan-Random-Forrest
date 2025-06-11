import Augmentor
import os

def augment_data():
    count = 0
    
    for root, dirs, files in os.walk("processed_data"):
        if not files:
            continue
        
        p = Augmentor.Pipeline(root)
        p.flip_left_right(probability=0.5)
        p.rotate90(probability=0.5)
        p.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15)
        p.shear(probability=0.5, max_shear_left=10, max_shear_right=10)
        p.skew_tilt(probability=0.5, magnitude=0.3)
        
        original_count = len([f for f in files if f.lower().endswith(('.jpg', '.jpeg'))])
        p.sample(original_count * 2)
        count += original_count * 2
    
    print(f"{count} images augmented")

if __name__ == "__main__":
    augment_data()