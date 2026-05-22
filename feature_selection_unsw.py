import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

# === Load Preprocessed Dataset ===
df = pd.read_csv("data/UNSW-NB15/unsw_nb15_processed.csv")

X = df.drop(columns=["label"])
y = df["label"]

# === Train ExtraTrees for Feature Importance ===
model = ExtraTreesClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# === Plot Feature Importances
importances = model.feature_importances_
cols = X.columns
feature_importance_df = pd.DataFrame({"feature": cols, "importance": importances})
feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)

# Save to file
feature_importance_df.to_csv("data/UNSW-NB15/unsw_feature_importance.csv", index=False)
print("✅ Feature importance written to: unsw_feature_importance.csv")

# Optional: Visualize
feature_importance_df.head(20).plot.bar(x='feature', y='importance', legend=False, figsize=(12, 6))
plt.title("Top 20 Feature Importances (UNSW)")
plt.tight_layout()
plt.savefig("unsw_feature_importance_plot.png")
plt.show()
