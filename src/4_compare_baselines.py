import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def clean_byte_strings(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
            df[col] = df[col].apply(lambda x: eval(x).decode('utf-8') if isinstance(x, str) and x.startswith("b'") else x)
    return df

def encode_categoricals(df):
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

# Load data
train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")

# Clean and encode
train = clean_byte_strings(train)
val = clean_byte_strings(val)
train, _ = encode_categoricals(train)
val, _ = encode_categoricals(val)

# Drop unused columns
drop_cols = [col for col in ['class', 'label', 'difficulty_level'] if col in train.columns]
X_train = train.drop(columns=drop_cols, errors='ignore')
y_train = train['label']

X_val = val.drop(columns=drop_cols, errors='ignore')
y_val = val['label']

# Compare models
models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "HIDT (Gradient Boosting)": GradientBoostingClassifier()
}

print("🔍 Comparing baseline models:")
for name, model in models.items():
    acc = evaluate_model(model, X_train, y_train, X_val, y_val)
    print(f"{name}: {acc:.4f}")
