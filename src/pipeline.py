import joblib
import pandas as pd

from risk_engine import calculate_risk_score, assign_priority
from data_loader import load_cmapss_fd001
from preprocessing import preprocess_cmapss
from features import create_features
from anomaly import detect_anomalies


def add_risk_score(
    df,
    rul_min,
    rul_max,
    anomaly_min,
    anomaly_max,
    uncertainty_min,
    uncertainty_max,
):
    df = df.copy()

    # Training-derived normalization ranges
    df["rul_risk"] = 1 - (
        (df["predicted_RUL"] - rul_min)
        / (rul_max - rul_min + 1e-9)
    )

    df["anomaly_risk"] = (
        (df["anomaly_score"] - anomaly_min)
        / (anomaly_max - anomaly_min + 1e-9)
    )

    df["uncertainty_risk"] = (
        (df["uncertainty_width"] - uncertainty_min)
        / (uncertainty_max - uncertainty_min + 1e-9)
    )

    # Keep normalized values within [0, 1]
    df["rul_risk"] = df["rul_risk"].clip(0, 1)
    df["anomaly_risk"] = df["anomaly_risk"].clip(0, 1)
    df["uncertainty_risk"] = df["uncertainty_risk"].clip(0, 1)

    df["risk_score"] = calculate_risk_score(
    df["rul_risk"],
    df["anomaly_risk"],
    df["uncertainty_risk"]
)

    return df


def add_priority(df):
    df = df.copy()

    df["maintenance_priority"] = df["risk_score"].apply(
        assign_priority
    )

    return df


def add_recommendation(df):
    df = df.copy()

    def recommendation(priority):
        if priority == "Critical":
            return "Immediate inspection and maintenance"
        elif priority == "High":
            return "Schedule maintenance soon"
        elif priority == "Medium":
            return "Continue monitoring"
        else:
            return "Routine monitoring"

    df["maintenance_recommendation"] = (
        df["maintenance_priority"].apply(recommendation)
    )

    return df


def run_pipeline():
    df = load_cmapss_fd001()
    df = preprocess_cmapss(df)
    df = create_features(df)

    # ------------------------------------------------------------
    # Anomaly model
    # ------------------------------------------------------------

    df, anomaly_model = detect_anomalies(df)

    feature_cols = [
        col
        for col in df.columns
        if col.startswith("sensor_")
    ]

    # ------------------------------------------------------------
    # Load trained models
    # ------------------------------------------------------------

    rul_model = joblib.load(
        "models/lgbm_rul.pkl"
    )

    lower_model = joblib.load(
        "models/lgbm_rul_lower.pkl"
    )

    upper_model = joblib.load(
        "models/lgbm_rul_upper.pkl"
    )

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------

    df["predicted_RUL"] = (
        rul_model.predict(df[feature_cols])
        .clip(min=0)
    )

    df["RUL_lower"] = (
        lower_model.predict(df[feature_cols])
        .clip(min=0)
    )

    df["RUL_upper"] = (
        upper_model.predict(df[feature_cols])
        .clip(min=0)
    )

    # ------------------------------------------------------------
    # Ensure interval contains point prediction
    # ------------------------------------------------------------

    df["RUL_lower"] = pd.concat(
        [
            df["RUL_lower"],
            df["predicted_RUL"]
        ],
        axis=1
    ).min(axis=1)

    df["RUL_upper"] = pd.concat(
        [
            df["RUL_upper"],
            df["predicted_RUL"]
        ],
        axis=1
    ).max(axis=1)

    # ------------------------------------------------------------
    # Calculate uncertainty width
    # ------------------------------------------------------------

    df["uncertainty_width"] = (
        df["RUL_upper"] - df["RUL_lower"]
    )

    # ------------------------------------------------------------
    # Create training reference
    # Same 80/20 asset split used in evaluation
    # ------------------------------------------------------------

    train_units = (
        df["unit"]
        .drop_duplicates()
        .sample(
            frac=0.80,
            random_state=42
        )
    )

    train_df = df[
        df["unit"].isin(train_units)
    ]

    # ------------------------------------------------------------
    # Training-derived normalization ranges
    # ------------------------------------------------------------

    rul_min = train_df["predicted_RUL"].min()
    rul_max = train_df["predicted_RUL"].max()

    anomaly_min = train_df["anomaly_score"].min()
    anomaly_max = train_df["anomaly_score"].max()

    uncertainty_min = train_df["uncertainty_width"].min()
    uncertainty_max = train_df["uncertainty_width"].max()

    # ------------------------------------------------------------
    # Calculate AssetIQ risk score
    # ------------------------------------------------------------

    df = add_risk_score(
        df,
        rul_min,
        rul_max,
        anomaly_min,
        anomaly_max,
        uncertainty_min,
        uncertainty_max,
    )

    # ------------------------------------------------------------
    # Assign maintenance priority and recommendation
    # ------------------------------------------------------------

    df = add_priority(df)
    df = add_recommendation(df)

    return df, anomaly_model, rul_model

if __name__ == "__main__":
    df, anomaly_model, rul_model = run_pipeline()

    print(df[
        [
            "unit",
            "cycle",
            "predicted_RUL",
            "uncertainty_width",
            "risk_score",
            "maintenance_priority",
            "maintenance_recommendation",
        ]
    ].head())

    print("\nRisk Score Summary")
    print(df["risk_score"].describe())

    print("\nMaintenance Priority")
    print(df["maintenance_priority"].value_counts())

    print("\nPipeline completed successfully.")

    from pathlib import Path

    output_path = Path("data/processed/assetiq_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"\nSaved predictions to: {output_path}")