# temp_check.py
import pandas as pd

df = pd.read_csv("data/NSL-KDD/KDDTrain+.csv")
print("Actual values in class column:")
print(df['class'].apply(type).value_counts())
print("\nSample values:")
print(df['class'].head())