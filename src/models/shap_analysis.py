import joblib
import pandas as pd


model = joblib.load(
    "models/lgbm_rul.pkl"
)

df = pd.read_csv(
    "data/processed/assetiq_predictions.csv"
)

feature_cols = model.feature_name_


def get_asset_explanation(asset_id):

    asset = (
        df[df["unit"] == asset_id]
        .sort_values("cycle")
    )

    latest = asset.tail(1)

    X_latest = latest[feature_cols]

    contributions = model.predict(
        X_latest,
        pred_contrib=True
    )[0]

    explanation = pd.DataFrame({
        "feature": feature_cols,
        "shap_value": contributions[:-1]
    })

    explanation["abs_shap"] = (
        explanation["shap_value"].abs()
    )

    explanation = explanation.sort_values(
        "abs_shap",
        ascending=False
    ).reset_index(drop=True)

    return explanation