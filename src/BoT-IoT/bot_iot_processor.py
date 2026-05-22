import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import shutil

# Configuration
INPUT_DIR = "data/BoT-IoT"
OUTPUT_DIR = "data/BoT-IoT/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Minimal columns to keep (reduces size by ~60%)
KEEP_COLS = ['mean', 'stddev', 'sbytes', 'dbytes', 'attack'] 

# 1. Cleanup existing files
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR)

# 2. Initialize with first valid file
print("🚀 Initializing with first valid file...")
scaler = StandardScaler()
label_encoder = LabelEncoder()

# Find first valid data file (skip data_names.csv)
data_files = sorted([f for f in os.listdir(INPUT_DIR) 
                   if f.startswith("data_") and f.endswith(".csv") 
                   and f != "data_names.csv"])

if not data_files:
    raise ValueError("No valid data files found!")

first_file = os.path.join(INPUT_DIR, data_files[0])
init_df = pd.read_csv(first_file, usecols=KEEP_COLS)

# Fit scaler and encoder
scaler.fit(init_df.drop('attack', axis=1))  # Only fit on feature columns
label_encoder.fit(init_df['attack'])

# 3. Process files incrementally
output_path = os.path.join(OUTPUT_DIR, "dataset.csv")
processed_files = 0

for i, file in enumerate(data_files):
    file_path = os.path.join(INPUT_DIR, file)
    print(f"🔧 Processing {file} ({i+1}/{len(data_files)})...")
    
    try:
        # Process in tiny chunks (10K rows)
        for chunk_idx, chunk in enumerate(pd.read_csv(
            file_path,
            usecols=KEEP_COLS,
            chunksize=10_000
        )):
            # Skip empty chunks
            if len(chunk) == 0:
                continue
                
            # Verify required columns exist
            if not all(col in chunk.columns for col in KEEP_COLS):
                print(f"⚠️ Missing columns in {file}, chunk {chunk_idx}")
                continue
                
            # Clean data
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            chunk.dropna(inplace=True)
            
            # Skip if no valid rows left
            if len(chunk) == 0:
                continue
                
            # Transform features
            features = scaler.transform(chunk.drop('attack', axis=1))
            
            # Create output DataFrame
            temp_df = pd.DataFrame(features, columns=chunk.drop('attack', axis=1).columns)
            temp_df['label'] = label_encoder.transform(chunk['attack'])
            
            # Save immediately to conserve memory
            temp_df.to_csv(
                output_path,
                mode='a',
                header=not os.path.exists(output_path),
                index=False
            )
            
        processed_files += 1
        
    except Exception as e:
        print(f"⚠️ Error processing {file}: {str(e)}")
        continue

print("\n✅ Processing complete!")
print(f"Processed {processed_files}/{len(data_files)} files successfully")
print(f"Final dataset size: {os.path.getsize(output_path)/1e6:.1f} MB")

# Verify output
if os.path.exists(output_path):
    sample = pd.read_csv(output_path, nrows=5)
    print("\nSample of processed data:")
    print(sample)
else:
    print("\n❌ No output file was created")