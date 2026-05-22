import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# === Define the selected features ===
top_features = top_features = [
    'sttl', 'dttl', 'ct_state_ttl', 'state', 'rate', 'swin', 'dload',
    'ct_srv_dst', 'dmean', 'ct_srv_src', 'ct_dst_src_ltm', 'smean',
    'ackdat', 'service', 'sload', 'is_sm_ips_ports', 'ct_dst_sport_ltm',
    'ct_dst_ltm', 'sbytes', 'stcpb', 'ct_src_dport_ltm', 'dwin', 'dtcpb',
    'ct_src_ltm', 'tcprtt', 'sinpkt', 'synack'
]
  # or list them manually

# === Load dataset and filter to top features
df = pd.read_csv("data/UNSW-NB15/unsw_nb15_processed.csv")

X = df[top_features]
y = df["label"]

# === Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# === Train the model
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# === Evaluate
y_pred = model.predict(X_test)
print("\n📊 Refined Classification Report:")
print(classification_report(y_test, y_pred))
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

# === Save model
joblib.dump(model, "models/unsw_rf_refined.pkl")
joblib.dump(top_features, "models/unsw_rf_refined_features.pkl")
print("✅ Refined model saved.")
