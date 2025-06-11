import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_random_forest():
    data_path = "features/glcm_features_distance_4_angle_90.csv"
    df = pd.read_csv(data_path)
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("Classification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    
    os.makedirs('training', exist_ok=True)
    model_path = 'training/random_forest_model_d4_a90.pkl'
    joblib.dump(rf_model, model_path)
    
    return rf_model, accuracy, cm

if __name__ == "__main__":
    model, accuracy, confusion_matrix = train_random_forest()
    print(f"\nAccuracy: {accuracy:.4f}")
