import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# === Load preprocessed CIC-IDS2017 data ===
df = pd.read_csv("data/CIC-IDS2017/cicids2017_processed.csv")

# === Step 1: Remove any rows with NaNs ===
df.dropna(inplace=True)

# === Step 2: Remove rows where 'label' is missing or not numeric ===
if 'label' not in df.columns:
    raise ValueError("❌ 'label' column not found in dataset.")
    
df = df[df['label'].notna()]
df = df[df['label'].astype(str).str.strip().ne("")]

# Optional: confirm label values
print("🧼 Cleaned labels - unique values:", df['label'].unique())

# === Step 3: Prepare features and labels ===
X = df.drop(columns=["label"])
y = df["label"]

# === Step 4: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# === Step 5: Train the model ===
model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# === Step 6: Evaluate the model ===
y_pred = model.predict(X_test)
print("\n📊 CIC-IDS2017 Classification Report:")
print(classification_report(y_test, y_pred))
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

# === Step 7: Save model and features ===
joblib.dump(model, "models/cicids_rf.pkl")
joblib.dump(X.columns.tolist(), "models/cicids_features.pkl")

print("✅ Model and feature list saved.")
