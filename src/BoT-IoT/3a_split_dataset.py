import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("📦 Loading BoT-IoT full dataset...")
input_path = "data/BoT-IoT/processed/dataset.csv"
df = pd.read_csv(input_path)

print(f"✅ Loaded: {df.shape}")

# Check label distribution
print("Label distribution:")
print(df['label'].value_counts())

# Ensure 'label' exists
if 'label' not in df.columns:
    raise ValueError("❌ No 'label' column found!")

# Stratified Split
try:
    print("✂️ Splitting into train/val/test...")
    train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    output_dir = "data/BoT-IoT/processed"
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("✅ Done.")
    print(f"Train: {train_df.shape}")
    print(f"Val:   {val_df.shape}")
    print(f"Test:  {test_df.shape}")

except Exception as e:
    print(f"❌ Error during splitting: {e}")
