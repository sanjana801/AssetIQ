import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


MAINTENANCE_COST = 10000
FAILURE_COST = 100000
PERSISTENCE = 3

RUL_THRESHOLD = 22.2953
ANOMALY_THRESHOLD = -0.0347
UNCERTAINTY_THRESHOLD = 140.2637
RISK_THRESHOLD = 0.60


df = pd.read_csv(
    "data/processed/assetiq_predictions.csv"
)


# ------------------------------------------------------------
# Same 80/20 asset split
# ------------------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        df[["unit"]],
        df["RUL"],
        groups=df["unit"]
    )
)

train = df.iloc[train_idx].copy()
val = df.iloc[val_idx].copy()

print("Training assets:", train["unit"].nunique())
print("Validation assets:", val["unit"].nunique())
print()


# ------------------------------------------------------------
# Training-derived normalization
# ------------------------------------------------------------

def normalize(
    train_series,
    target_series
):
    minimum = train_series.min()
    maximum = train_series.max()

    return (
        (target_series - minimum)
        / (maximum - minimum + 1e-9)
    ).clip(0, 1)


# RUL risk
train_rul_risk = 1 - normalize(
    train["predicted_RUL"],
    train["predicted_RUL"]
)

val_rul_risk = 1 - normalize(
    train["predicted_RUL"],
    val["predicted_RUL"]
)

# Anomaly risk
train_anomaly_risk = normalize(
    train["anomaly_score"],
    train["anomaly_score"]
)

val_anomaly_risk = normalize(
    train["anomaly_score"],
    val["anomaly_score"]
)

# Uncertainty risk
train_uncertainty_risk = normalize(
    train["uncertainty_width"],
    train["uncertainty_width"]
)

val_uncertainty_risk = normalize(
    train["uncertainty_width"],
    val["uncertainty_width"]
)


# ------------------------------------------------------------
# Original AssetIQ weighted risk
# ------------------------------------------------------------

val["full_risk"] = (
    0.50 * val_rul_risk
    + 0.25 * val_anomaly_risk
    + 0.25 * val_uncertainty_risk
)


# ------------------------------------------------------------
# Component-removal versions
#
# Missing components are NOT given new weights.
# This lets us see the direct contribution of each component.
# ------------------------------------------------------------

val["rul_only"] = (
    0.50 * val_rul_risk
)

val["rul_anomaly"] = (
    0.50 * val_rul_risk
    + 0.25 * val_anomaly_risk
)

val["rul_uncertainty"] = (
    0.50 * val_rul_risk
    + 0.25 * val_uncertainty_risk
)


# ------------------------------------------------------------
# Convert scores to maintenance decisions
# ------------------------------------------------------------

def evaluate_policy(
    asset_df,
    score_column,
    threshold
):

    results = []

    for unit, asset in asset_df.groupby("unit"):

        asset = (
            asset
            .sort_values("cycle")
            .reset_index(drop=True)
        )

        signal = (
            asset[score_column] >= threshold
        )

        trigger_idx = None

        for i in range(
            PERSISTENCE - 1,
            len(signal)
        ):

            if signal.iloc[
                i - PERSISTENCE + 1:i + 1
            ].all():

                trigger_idx = i
                break

        failure_cycle = int(
            asset.iloc[-1]["cycle"]
        )

        if trigger_idx is not None:

            maintenance_cycle = int(
                asset.iloc[trigger_idx]["cycle"]
            )

            lead_time = (
                failure_cycle
                - maintenance_cycle
            )

            failure = 0
            cost = MAINTENANCE_COST

        else:

            maintenance_cycle = None
            lead_time = 0
            failure = 1
            cost = FAILURE_COST

        results.append({
            "unit": unit,
            "maintenance_cycle": maintenance_cycle,
            "lead_time": lead_time,
            "failure": failure,
            "cost": cost
        })

    result = pd.DataFrame(results)

    maintenance = result[
        "maintenance_cycle"
    ].notna().sum()

    failures = result["failure"].sum()

    triggered = result[
        result["maintenance_cycle"].notna()
    ]

    mean_lead = (
        triggered["lead_time"].mean()
        if len(triggered) > 0
        else 0
    )

    total_cost = result["cost"].sum()

    return (
        maintenance,
        failures,
        mean_lead,
        total_cost
    )


# ------------------------------------------------------------
# Use thresholds corresponding to the original decision rules
# ------------------------------------------------------------

policies = [
    (
        "RUL-only",
        "rul_only",
        0.50 * (
            1 - normalize(
                train["predicted_RUL"],
                pd.Series([RUL_THRESHOLD])
            ).iloc[0]
        )
    ),
    (
        "RUL + Anomaly",
        "rul_anomaly",
        0.50 * (
            1 - normalize(
                train["predicted_RUL"],
                pd.Series([RUL_THRESHOLD])
            ).iloc[0]
        )
        + 0.25 * normalize(
            train["anomaly_score"],
            pd.Series([ANOMALY_THRESHOLD])
        ).iloc[0]
    ),
    (
        "RUL + Uncertainty",
        "rul_uncertainty",
        0.50 * (
            1 - normalize(
                train["predicted_RUL"],
                pd.Series([RUL_THRESHOLD])
            ).iloc[0]
        )
        + 0.25 * normalize(
            train["uncertainty_width"],
            pd.Series([UNCERTAINTY_THRESHOLD])
        ).iloc[0]
    ),
    (
        "Full AssetIQ",
        "full_risk",
        RISK_THRESHOLD
    )
]


print(
    "Policy | Maintenance | Failures | "
    "Mean lead time | Total cost"
)

print("-" * 75)


for name, column, threshold in policies:

    result = evaluate_policy(
        val,
        column,
        threshold
    )

    maintenance, failures, lead, cost = result

    print(
        f"{name} | "
        f"{maintenance} | "
        f"{failures} | "
        f"{lead:.2f} | "
        f"₹{cost:,.0f}"
    )