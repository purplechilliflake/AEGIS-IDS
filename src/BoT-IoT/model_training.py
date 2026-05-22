import pandas as pd
import numpy as np
import joblib
import os
import json
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings("ignore")

class IntrusionDetectionModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight=None,  # We'll set dynamically in training
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

    def load_data(self, max_rows=100_000):
        def smart_sample(path, max_rows):
            total_rows = sum(1 for _ in open(path)) - 1
            if total_rows <= max_rows:
                return pd.read_csv(path)
            frac = max_rows / total_rows
            return pd.read_csv(path).sample(frac=frac, random_state=42)

        try:
            print("📦 Loading BoT-IoT data with memory-safe sampling...")

            train_path = "data/BoT-IoT/processed/train.csv"
            val_path = "data/BoT-IoT/processed/val.csv"
            test_path = "data/BoT-IoT/processed/test.csv"

            train = smart_sample(train_path, max_rows)
            val = smart_sample(val_path, int(max_rows * 0.5))
            test = smart_sample(test_path, int(max_rows * 0.5))

            print(f"✔️ Loaded:")
            print(f"- Train: {train.shape}")
            print(f"- Validation: {val.shape}")
            print(f"- Test: {test.shape}")

            if os.path.exists("models/preprocessor_metadata.json"):
                with open("models/preprocessor_metadata.json") as f:
                    self.metadata = json.load(f)
                    print("📁 Preprocessor metadata loaded")

            train = self.clean_byte_strings(train)
            val = self.clean_byte_strings(val)
            test = self.clean_byte_strings(test)

            train, _ = self.encode_categoricals(train)
            val, _ = self.encode_categoricals(val)
            test, _ = self.encode_categoricals(test)

            return train, val, test

        except FileNotFoundError as e:
            print(f"❌ ERROR: {e}")
            exit(1)

    def train(self, X_train, y_train):
        print("\n🎯 Balancing data with SMOTE...")
        print("Before:", np.bincount(y_train))

        if len(np.unique(y_train)) < 2:
            print("⚠️ Only one class present in training data. Skipping training.")
            return

        smote = SMOTE(random_state=42)  # ← fixed line
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        print("After:", np.bincount(y_resampled))

        # Apply class weighting
        self.model.set_params(class_weight={0: 5, 1: 1})

        print("\n🔁 Running cross-validation...")
        scores = cross_val_score(
            self.model, X_resampled, y_resampled, cv=5, scoring='f1', n_jobs=-1
        )
        print(f"CV F1 scores: {scores}")
        print(f"Mean F1: {np.mean(scores):.3f} (±{np.std(scores):.3f})")

        print("\n🧠 Training final model...")
        self.model.fit(X_resampled, y_resampled)

    def evaluate(self, X, y, dataset_name):
        pred = self.model.predict(X)
        print(f"\n📋 {dataset_name} Classification Report:")
        print(classification_report(y, pred, target_names=["Normal", "Anomaly"]))

        cm = confusion_matrix(y, pred)
        print(f"\n📊 {dataset_name} Confusion Matrix:")
        print(cm)

        if dataset_name == "Validation Set":
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            print("\n⭐ Top 10 Features:")
            print(feature_importance.head(10))
            feature_importance.to_csv("models/feature_importance.csv", index=False)

    def save_model(self):
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, "models/intrusion_detection_rf.pkl")
        print("💾 Model saved to models/intrusion_detection_rf.pkl")

if __name__ == "__main__":
    print("🚀 Starting BoT-IoT model training with imbalance handling...")
    detector = IntrusionDetectionModel()
    train, val, test = detector.load_data(max_rows=100_000)

    drop_cols = [col for col in ['class', 'label', 'difficulty_level'] if col in train.columns]
    X_train = train.drop(columns=drop_cols, errors='ignore')
    y_train = train['label']

    print(f"🧮 Training set shape: {X_train.shape}")

    detector.train(X_train, y_train)

    X_val = val.drop(columns=drop_cols, errors='ignore')
    y_val = val['label']
    detector.evaluate(X_val, y_val, "Validation Set")

    X_test = test.drop(columns=drop_cols, errors='ignore')
    y_test = test['label']
    detector.evaluate(X_test, y_test, "Test Set")

    detector.save_model()
    print("\n✅ Training and evaluation complete.")
