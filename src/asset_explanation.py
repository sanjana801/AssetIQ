import joblib
import shap
import pandas as pd


# ------------------------------------------------------------
# Asset-level decision explanation
# ------------------------------------------------------------

ASSET_ID = 1


# ------------------------------------------------------------
# Load AssetIQ prediction data
# ------------------------------------------------------------

df = pd.read_csv(
    "data/processed/assetiq_predictions.csv"
)

asset = df[df["unit"] == ASSET_ID].sort_values(
    "cycle"
)

latest = asset.tail(1)


# ------------------------------------------------------------
# Load trained RUL model
# ------------------------------------------------------------

model = joblib.load(
    "models/lgbm_rul.pkl"
)


# ------------------------------------------------------------
# Prepare SHAP explanation
# ------------------------------------------------------------

feature_cols = [
    col
    for col in df.columns
    if col.startswith("sensor_")
]

X_latest = latest[feature_cols]

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_latest
)


# ------------------------------------------------------------
# Identify major contributing features
# ------------------------------------------------------------

explanation = pd.DataFrame({
    "feature": feature_cols,
    "shap_value": shap_values[0]
})

explanation["abs_shap"] = (
    explanation["shap_value"].abs()
)

explanation = explanation.sort_values(
    "abs_shap",
    ascending=False
).head(5)


# ------------------------------------------------------------
# Display AssetIQ decision record
# ------------------------------------------------------------

print("\nAssetIQ Decision Record")
print("=" * 60)

print("Asset ID:", ASSET_ID)
print("Current cycle:", int(latest["cycle"].iloc[0]))

print(
    "Predicted RUL:",
    round(float(latest["predicted_RUL"].iloc[0]), 2)
)

print(
    "RUL lower bound:",
    round(float(latest["RUL_lower"].iloc[0]), 2)
)

print(
    "RUL upper bound:",
    round(float(latest["RUL_upper"].iloc[0]), 2)
)

print(
    "Uncertainty width:",
    round(float(latest["uncertainty_width"].iloc[0]), 2)
)

print(
    "Anomaly score:",
    round(float(latest["anomaly_score"].iloc[0]), 4)
)

print(
    "Risk score:",
    round(float(latest["risk_score"].iloc[0]), 4)
)

print(
    "Maintenance priority:",
    latest["maintenance_priority"].iloc[0]
)

print(
    "Recommendation:",
    latest["maintenance_recommendation"].iloc[0]
)

print("\nMain SHAP Contributors")
print("=" * 60)

print(
    explanation[
        ["feature", "shap_value"]
    ].to_string(index=False)
)