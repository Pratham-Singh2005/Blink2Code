print("🚀 Preprocessing script started!")

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():
    try:
        print("🚀 Starting preprocessing...")

        # Load dataset
        dataset_path = "dataset/DataAnotations.xlsx"
        print(f"📂 Loading dataset from: {dataset_path}")

        df = pd.read_excel(dataset_path)
        print(f"✅ Dataset loaded successfully! Shape: {df.shape}")

        # Drop unnecessary columns
        print("🧹 Dropping unnecessary columns...")
        df = df.drop(columns=['Ground Truth', 'Eyeblink Type', 'eyeState'], errors='ignore')

        # Fill missing values
        print("🔄 Handling missing values...")
        df.fillna(df.mean(), inplace=True)

        # Check if target column exists
        print("🎯 Extracting features and target variable...")
        if 'iBlinkIsDetected' not in df.columns:
            print("❌ Error: 'iBlinkIsDetected' column is missing!")
            return None

        X = df.drop(columns=['iBlinkIsDetected'])
        y = df['iBlinkIsDetected']

        # Normalize features
        print("📊 Normalizing features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        print("✂️ Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Convert to PyTorch tensors
        print("🔄 Converting data to PyTorch tensors...")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

        print(f"✅ Preprocessing complete! Train Shape: {X_train_tensor.shape}, Test Shape: {X_test_tensor.shape}")
        return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, X_train.shape[1]

    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return None

if __name__ == "__main__":
    result = load_and_preprocess()
    if result is None:
        print("❌ Preprocessing failed!")
    else:
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, input_dim = result
        print(f"✅ Data processed successfully! Input dimension: {input_dim}")
