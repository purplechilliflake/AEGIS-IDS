import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Paths
input_path = "data/BoT-IoT/cleaned_bot_iot.csv"
chunk_dir = "data/BoT-IoT/processed/chunks"
os.makedirs(chunk_dir, exist_ok=True)

# Set chunk size (adjustable)
CHUNK_SIZE = 100_000

# Label encoder to fit only once
label_encoder = LabelEncoder()

# Find the columns we'll normalize
sample = pd.read_csv(input_path, nrows=1)
feature_cols = sample.columns.drop(['attack', 'subcategory ']) if 'attack' in sample.columns else sample.columns[:-1]

# Step 1: Determine label encoding values from entire dataset
print("📥 Scanning file to fit label encoder...")
labels = pd.read_csv(input_path, usecols=['attack'], chunksize=CHUNK_SIZE)
all_labels = pd.concat(labels)
label_encoder.fit(all_labels['attack'])

# Step 2: Stream through chunks, encode, scale, and save
scaler = StandardScaler()
chunk_files = []

print("\n⚙️ Processing in chunks...")
reader = pd.read_csv(input_path, chunksize=CHUNK_SIZE)
for i, chunk in enumerate(reader):
    print(f"🔹 Chunk {i+1}")

    # Drop categorical columns
    if 'subcategory ' in chunk.columns:
        chunk.drop(columns=['subcategory '], inplace=True)
    if 'attack' not in chunk.columns:
        print("❌ Missing 'attack' column!")
        continue

    # Encode label
    chunk['label'] = label_encoder.transform(chunk['attack'])
    chunk.drop(columns=['attack'], inplace=True)

    # Drop any non-numeric leftover columns
    
    # Drop only columns with all non-numeric content
    for col in chunk.columns:
        if chunk[col].dtype == 'object' and chunk[col].nunique() < 10:
            chunk.drop(columns=[col], inplace=True)

# Clean bad values but keep rows with *some* missing data
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.dropna(subset=['label'], inplace=True)  # only drop rows missing label

# Fill other NaNs with 0 (safe fallback)
    chunk.fillna(0, inplace=True)

    X_chunk = chunk.drop(columns=['label'])

    if X_chunk.empty:
        print(f"⚠️ Skipping empty chunk {i}")
        continue

    X_scaled = scaler.fit_transform(X_chunk)


    if X_chunk.empty:
        print(f"⚠️ Skipping empty chunk {i}")
        continue

    X_scaled = scaler.fit_transform(X_chunk)
    chunk[X_chunk.columns] = X_scaled

    # Save this chunk
    out_path = os.path.join(chunk_dir, f"chunk_{i}.csv")
    chunk.to_csv(out_path, index=False)
    chunk_files.append(out_path)

print(f"\n✅ Processed {len(chunk_files)} chunks. Ready for splitting.")
