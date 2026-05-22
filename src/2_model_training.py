import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

class IntrusionDetectionModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        self.metadata = None

    def clean_byte_strings(self, df):
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
                )
                df[col] = df[col].apply(
                    lambda x: eval(x).decode('utf-8') if isinstance(x, str) and x.startswith("b'") else x
                )
        return df

    def encode_categoricals(self, df):
        encoders = {}
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        return df, encoders

    def load_data(self):
        try:
            train = pd.read_csv("data/BoT-IoT/processed/train.csv")
            val = pd.read_csv("data/BoT-IoT/processed/val.csv")
            test = pd.read_csv("data/BoT-IoT/processed/test.csv")

            print(f"Successfully loaded data:")
            print(f"- Train: {train.shape}")
            print(f"- Validation: {val.shape}")
            print(f"- Test: {test.shape}")

            with open("models/preprocessor_metadata.json") as f:
                self.metadata = json.load(f)
                print("Successfully loaded preprocessor metadata")

            train = self.clean_byte_strings(train)
            val = self.clean_byte_strings(val)
            test = self.clean_byte_strings(test)

            train, _ = self.encode_categoricals(train)
            val, _ = self.encode_categoricals(val)
            test, _ = self.encode_categoricals(test)

            return train, val, test
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            exit(1)

    def train(self, X_train, y_train):
        print("\nRunning cross-validation...")
        scores = cross_val_score(
            self.model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1
        )
        print(f"CV F1 scores: {scores}")
        print(f"Mean F1: {np.mean(scores):.3f} (±{np.std(scores):.3f})")

        print("\nTraining final model...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X, y, dataset_name):
        pred = self.model.predict(X)
        print(f"\n{dataset_name} Classification Report:")
        print(classification_report(y, pred, target_names=["Normal", "Anomaly"]))

        conf_matrix = confusion_matrix(y, pred)
        print(f"\n{dataset_name} Confusion Matrix:")
        print(conf_matrix)

        if dataset_name == "Validation Set":
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)

            print("\nTop 10 Important Features:")
            print(feature_importance.head(10))
            feature_importance.to_csv("models/feature_importance.csv", index=False)

    def save_model(self):
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, "models/intrusion_detection_rf.pkl")
        print("Model saved to models/intrusion_detection_rf.pkl")

if __name__ == "__main__":
    print("🚀 Starting intrusion detection model training...")

    detector = IntrusionDetectionModel()
    train, val, test = detector.load_data()

    # Ensure any of these columns are dropped if present in any of the datasets
    drop_cols = [col for col in ['class', 'label', 'difficulty_level']
                 if col in train.columns or col in val.columns or col in test.columns]

    # Prepare training set
    X_train = train.drop(columns=drop_cols, errors='ignore')
    y_train = train['label']
    print(f"Training features shape: {X_train.shape}")

    # Train and evaluate
    detector.train(X_train, y_train)

    X_val = val.drop(columns=drop_cols, errors='ignore')
    y_val = val['label']
    detector.evaluate(X_val, y_val, "Validation Set")

    X_test = test.drop(columns=drop_cols, errors='ignore')
    y_test = test['label']
    detector.evaluate(X_test, y_test, "Test Set")

    detector.save_model()

    print("\n✅ Model training and evaluation complete!")
