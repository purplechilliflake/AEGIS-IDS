import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# === Load UNSW-NB15 Dataset ===
df = pd.read_csv("data/unsw-nb15/Training and Testing Sets/UNSW_NB15_training-set.csv")

print(f"Loaded UNSW-NB15 dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# === Drop Non-useful Columns ===
drop_cols = ['id', 'attack_cat'] if 'id' in df.columns else ['attack_cat']
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# === Encode Categorical Columns ===
cat_cols = df.select_dtypes(include='object').columns.tolist()
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# === Handle Missing Values ===
df.fillna(0, inplace=True)

# === Split Features and Label ===
X = df.drop(columns=['label'])
y = df['label']

# === Scale the Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Create Output DataFrame ===
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df['label'] = y

# === Save Processed Data ===
output_path = "data/UNSW-NB15/unsw_nb15_processed.csv"
processed_df.to_csv(output_path, index=False)

print(f"✅ Saved preprocessed dataset to: {output_path}")
