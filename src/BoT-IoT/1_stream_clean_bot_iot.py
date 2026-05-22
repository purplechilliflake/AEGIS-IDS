import pandas as pd
import os

input_dir = "data/BoT-IoT"
output_path = os.path.join(input_dir, "cleaned_bot_iot.csv")

# Load column names from data_names.csv
column_names = pd.read_csv(os.path.join(input_dir, "data_names.csv")).columns.tolist()

# Filter only valid data files
file_list = sorted([
    f for f in os.listdir(input_dir)
    if f.startswith("data_") and f.endswith(".csv") and f != "data_names.csv"
])

if os.path.exists(output_path):
    os.remove(output_path)

total_rows = 0

for idx, file in enumerate(file_list):
    file_path = os.path.join(input_dir, file)
    print(f"📂 Processing {file} ({idx + 1}/{len(file_list)})")

    try:
        # Read file safely
        if idx == 0:
            df = pd.read_csv(file_path, header=0)
            df.columns = column_names
        else:
            df = pd.read_csv(file_path, header=None, skiprows=1)
            df.columns = column_names

        # Drop noisy columns
        drop_cols = ['saddr', 'daddr', 'sport', 'dport', 'category', 'proto', 'state', 'flgs', 'pkts', 'bytes', 'seq', 'dur']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

        # Drop rows that are obviously header rows (like strings in numeric columns)
        if df.shape[0] > 0 and df.iloc[0].astype(str).str.contains('pkSeqID').any():
            df = df[1:]

        # Only keep rows with at least one numeric column present
        df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        df.dropna(how='all', subset=[col for col in df.columns if col != 'attack'], inplace=True)

        total_rows += len(df)
        df.to_csv(output_path, mode='a', index=False, header=(idx == 0))

    except Exception as e:
        print(f"⚠️ Skipping {file} due to error: {e}")

print(f"\n✅ Final dataset saved to: {output_path}")
print(f"✅ Total rows collected: {total_rows}")
