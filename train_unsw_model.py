import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# === Load Preprocessed Dataset ===
df = pd.read_csv("data/UNSW-NB15/unsw_nb15_processed.csv")
print(f"Loaded processed UNSW-NB15: {df.shape[0]} rows")

# === Split Features and Target ===
X = df.drop(columns=["label"])
y = df["label"]

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# === Train Classifier ===
model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

# === Save Model and Feature Names ===
joblib.dump(model, "models/unsw_ids_rf.pkl")
joblib.dump(X.columns.tolist(), "models/unsw_ids_features.pkl")
print("✅ Model and feature list saved.")
