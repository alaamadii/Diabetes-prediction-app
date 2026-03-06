import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def prepare_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Assuming the target variable is 'Outcome' (common in diabetes datasets)
    # Adjust if your target column has a different name
    if 'Outcome' not in df.columns:
        print(f"Error: 'Outcome' column not found in data. Columns are: {df.columns}")
        # Look for a differently named target, perhaps 'Class', 'target'
        target_col = df.columns[-1] # Usually the last column
        print(f"Guessing {target_col} as target variable.")
    else:
        target_col = 'Outcome'

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    print(f"--- Training {model_name} ---")
    model.fit(X_train, y_train)
    
    print(f"--- Evaluating {model_name} ---")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

def main():
    data_file = 'data/diabetes.csv'
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")
        
    try:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(data_file)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file}")
        return

    # Save the scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Saved StandardScaler to {scaler_path}\n")

    # 1. Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model = train_and_evaluate(lr_model, X_train_scaled, y_train, X_test_scaled, y_test, "Logistic Regression")
    joblib.dump(lr_model, os.path.join(models_dir, 'logistic_regression.pkl'))
    
    # 2. Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model = train_and_evaluate(rf_model, X_train_scaled, y_train, X_test_scaled, y_test, "Random Forest")
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest.pkl'))

    # 3. SVM
    svm_model = SVC(random_state=42, probability=True)
    svm_model = train_and_evaluate(svm_model, X_train_scaled, y_train, X_test_scaled, y_test, "Support Vector Machine (SVM)")
    joblib.dump(svm_model, os.path.join(models_dir, 'svm.pkl'))
    
    print("All models trained and saved successfully.")

if __name__ == "__main__":
    main()
