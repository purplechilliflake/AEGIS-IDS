import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Paths - ADD FULL PATH VERIFICATION
chunk_dir = os.path.abspath("data/BoT-IoT/processed/chunks")
output_dir = os.path.abspath("data/BoT-IoT/processed")
os.makedirs(output_dir, exist_ok=True)

print(f"📂 Chunk directory: {chunk_dir}")
print(f"📂 Output directory: {output_dir}")

# Initialize empty DataFrames for collecting splits
train_all = pd.DataFrame()
val_all = pd.DataFrame()
test_all = pd.DataFrame()

# Get sorted list of chunk files
chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith(".csv")])
print(f"Found {len(chunk_files)} chunk files")

for idx, file in enumerate(chunk_files):
    path = os.path.join(chunk_dir, file)
    
    try:
        df = pd.read_csv(path)
        
        # Skip if empty
        if df.empty:
            print(f"⚠️ Empty chunk: {file}")
            continue
            
        # Ensure label column exists
        if 'label' not in df.columns:
            print(f"⚠️ Missing label in {file}")
            continue
            
        # Split with stratification if possible
        label_counts = df['label'].value_counts()
        if len(label_counts) >= 2 and label_counts.min() >= 2:
            train, temp = train_test_split(
                df, 
                test_size=0.4, 
                stratify=df['label'], 
                random_state=42
            )
            val, test = train_test_split(
                temp, 
                test_size=0.5, 
                stratify=temp['label'], 
                random_state=42
            )
        else:
            print(f"⚠️ Chunk {idx+1}: insufficient class variety")
            train, temp = train_test_split(df, test_size=0.4, random_state=42)
            val, test = train_test_split(temp, test_size=0.5, random_state=42)
            
        # Append to collections
        train_all = pd.concat([train_all, train])
        val_all = pd.concat([val_all, val])
        test_all = pd.concat([test_all, test])
        
        if (idx+1) % 50 == 0:
            print(f"✅ Processed {idx+1}/{len(chunk_files)} chunks")
            
    except Exception as e:
        print(f"❌ Failed on {file}: {str(e)}")
        continue

# Final save - USING PARQUET FOR BETTER PERFORMANCE
print("\n💾 Saving final splits...")
train_all.to_parquet(os.path.join(output_dir, "train.parquet"))
val_all.to_parquet(os.path.join(output_dir, "val.parquet"))
test_all.to_parquet(os.path.join(output_dir, "test.parquet"))

print("\n✅ Final split counts:")
print(f"Train: {len(train_all):,} rows")
print(f"Val: {len(val_all):,} rows")
print(f"Test: {len(test_all):,} rows")