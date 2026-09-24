import joblib
import shap
import pandas as pd


def create_explainer(model):
    return shap.TreeExplainer(model)


def explain_predictions(explainer, X):
    return explainer.shap_values(X)


if __name__ == "__main__":

    # ------------------------------------------------------------
    # Load trained RUL model
    # ------------------------------------------------------------

    model = joblib.load(
        "models/lgbm_rul.pkl"
    )

    # ------------------------------------------------------------
    # Load processed dataset
    # ------------------------------------------------------------

    df = pd.read_csv(
        "data/processed/assetiq_predictions.csv"
    )

    feature_cols = [
        col
        for col in df.columns
        if col.startswith("sensor_")
    ]

    X = df[feature_cols]

    # Use a small sample for the first SHAP test
    X_sample = X.sample(
        n=100,
        random_state=42
    )

    # ------------------------------------------------------------
    # SHAP explanation
    # ------------------------------------------------------------

    explainer = create_explainer(model)

    shap_values = explain_predictions(
        explainer,
        X_sample
    )

    print("SHAP test successful")
    print("Samples explained:", len(X_sample))
    print("Features:", len(feature_cols))
    print("SHAP matrix shape:", shap_values.shape)