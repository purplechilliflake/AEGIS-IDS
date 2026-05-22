import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# === Load the CSV (pick your target file)
input_path = "data/CIC-IDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = pd.read_csv(input_path)

print(f"Loaded dataset with shape: {df.shape}")

# === Clean column names (strip whitespaces)
df.columns = df.columns.str.strip()

# === Drop columns with all NaNs or constant values
df = df.dropna(axis=1, how='all')
df = df.loc[:, df.nunique() > 1]

# === Remove non-numeric and infinity entries
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

# === Encode Label column
# === Encode Label column
# === Clean and Encode Label column
if 'Label' in df.columns:
    df = df[df['Label'].notna()]  # drop nulls
    df = df[df['Label'].astype(str).str.strip().ne("")]  # drop empty strings
    df = df[~df['Label'].astype(str).str.contains('Infinity|NaN', na=False)]  # drop invalid labels

    print("✅ Unique raw labels:", df['Label'].unique())

    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'].astype(str))
else:
    raise ValueError("No 'Label' column found in dataset.")

print("✅ Label distribution:", df['Label'].value_counts().to_dict())

# === Separate Features and Labels
X = df.drop(columns=['Label'])
y = df['Label']

# === Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Combine and Save
processed = pd.DataFrame(X_scaled, columns=X.columns)
processed['label'] = y
processed.to_csv("data/CIC-IDS2017/cicids2017_processed.csv", index=False)

print("✅ Processed dataset saved to: data/CIC-IDS2017/cicids2017_processed.csv")
