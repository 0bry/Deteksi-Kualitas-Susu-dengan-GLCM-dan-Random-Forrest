import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import glob

def train_and_evaluate(csv_file):
    df = pd.read_csv(csv_file)
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    overall_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    parts = os.path.basename(csv_file).replace('.csv', '').split('_')
    distance = int(parts[-3])
    angle = int(parts[-1])
    
    result = {
        'Distance': distance,
        'Angle': angle,
        'Overall_Accuracy': overall_accuracy
    }
    
    for class_label in sorted(y.unique()):
        if str(class_label) in report:
            result[f'Class_{class_label}_Accuracy'] = report[str(class_label)]['precision']
    
    return result

def main():
    feature_files = glob.glob("g:/Project/Deteksi-Kualitas-Susu-dengan-GLCM-dan-Random-Forrest/features/glcm_features_*.csv")
    
    if not feature_files:
        print("No feature files found!")
        return
    
    results = []
    for csv_file in feature_files:
        try:
            result = train_and_evaluate(csv_file)
            results.append(result)
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
    
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values('Overall_Accuracy', ascending=False)
    
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv('g:/Project/Deteksi-Kualitas-Susu-dengan-GLCM-dan-Random-Forrest/random_forest_results_summary.csv', index=False)

if __name__ == "__main__":
    main()
