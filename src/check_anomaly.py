import pandas as pd

df = pd.read_csv("data/processed/assetiq_predictions.csv")

latest = (
    df.sort_values(["unit", "cycle"])
      .groupby("unit")
      .tail(1)
)

print(latest["anomaly_label"].value_counts())

print("\nAnomaly percentage:")
print((latest["anomaly_label"] == -1).mean() * 100)