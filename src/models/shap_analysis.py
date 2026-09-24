import joblib
import shap
import pandas as pd
import os


model = joblib.load(
    "models/lgbm_rul.pkl"
)

df = pd.read_csv(
    "data/processed/assetiq_predictions.csv"
)

feature_cols = [
    col for col in df.columns
    if col.startswith("sensor_")
]


def get_asset_explanation(asset_id):

    asset = df[df["unit"] == asset_id].sort_values("cycle")
    latest = asset.tail(1)

    X_latest = latest[feature_cols]

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_latest)

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
    ).reset_index(drop=True)

    return explanation


if __name__ == "__main__":

    explanation = get_asset_explanation(1)

    print("\nAsset Explanation")
    print("=" * 50)

    print(
        explanation.head(10)[
            ["feature", "shap_value"]
        ].to_string(index=False)
    )