import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV

from data_loader import load_cmapss_fd001
from preprocessing import preprocess_cmapss
from features import create_features
from models.rul_model import train_rul_model
from models.random_forest_rul import train_random_forest_rul


RUL_THRESHOLD = 22.2953
PERSISTENCE = 3

MAINTENANCE_COST = 10000
FAILURE_COST = 100000


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = load_cmapss_fd001()
df = preprocess_cmapss(df)
df = create_features(df)

predictions = pd.read_csv(
    "data/processed/assetiq_predictions.csv"
)

df = df.merge(
    predictions[
        [
            "unit",
            "cycle",
            "RUL_lower",
            "RUL_upper",
            "uncertainty_width"
        ]
    ],
    on=["unit", "cycle"],
    how="left"
)

feature_cols = [
    col for col in df.columns
    if col.startswith("sensor_")
]

X = df[feature_cols]
y = df["RUL"]
groups = df["unit"]


# ------------------------------------------------------------
# Same 80/20 asset split
# ------------------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]

df_val = df.iloc[val_idx].copy()

print("Training assets:", df.iloc[train_idx]["unit"].nunique())
print("Validation assets:", df_val["unit"].nunique())
print("Validation units:", sorted(df_val["unit"].unique()))

# ------------------------------------------------------------
# Failure Probability Model
# ------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

y_failure = (df["RUL"] == 0).astype(int)

y_failure_train = y_failure.iloc[train_idx]
y_failure_val = y_failure.iloc[val_idx]

base_failure_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

failure_model = CalibratedClassifierCV(
    base_failure_model,
    method="sigmoid",
    cv=5
)

failure_model.fit(
    X_train,
    y_failure_train
)

failure_probability = failure_model.predict_proba(
    X_val
)[:, 1]

pr_auc = average_precision_score(
    y_failure_val,
    failure_probability
)

print("\nFailure Probability Model")
print("=" * 75)
print(f"Validation PR-AUC: {pr_auc:.4f}")

# ------------------------------------------------------------
# Select Failure Probability Threshold
# Asset-level training optimization
# ------------------------------------------------------------

PREVENTIVE_COST = 10000
FAILURE_COST = 100000

train_probability = failure_model.predict_proba(
    X_train
)[:, 1]

threshold_candidates = np.arange(0.05, 0.51, 0.05)


best_threshold = None
best_training_cost = float("inf")

train_predictions = df.iloc[train_idx].copy()
train_predictions["failure_probability"] = train_probability

for threshold in threshold_candidates:

    total_cost = 0

    for unit, asset in train_predictions.groupby("unit"):

        asset = asset.sort_values("cycle").reset_index(drop=True)

        signal = (
            asset["failure_probability"] >= threshold
        )

        trigger = None

        for i in range(PERSISTENCE - 1, len(signal)):
            if signal.iloc[i-PERSISTENCE+1:i+1].all():
                trigger = i
                break

        failure_cycle = int(asset.iloc[-1]["cycle"])

        if trigger is not None:

            maintenance_cycle = int(
                asset.iloc[trigger]["cycle"]
            )

            lead_time = (
                failure_cycle - maintenance_cycle
            )

            cost = (
                PREVENTIVE_COST
                + 100 * lead_time
            )

        else:

            cost = FAILURE_COST

        total_cost += cost

    print(
        f"Threshold {threshold:.2f} "
        f"-> Training cost: ₹{total_cost:,.0f}"
    )   


    if total_cost < best_training_cost:

        best_training_cost = total_cost
        best_threshold = threshold


print("\nTraining-Based Cost Threshold")
print("=" * 75)

print(
    f"Selected threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Training simulated cost: "
    f"₹{best_training_cost:,.0f}"
)


# ------------------------------------------------------------
# Asset-Level Expected-Cost Maintenance Policy
# ------------------------------------------------------------

print("\nAsset-Level Expected-Cost Maintenance Policy")
print("=" * 75)

PREVENTIVE_COST = 10000
FAILURE_COST = 100000
FAILURE_PROB_THRESHOLD = best_threshold

PERSISTENCE = 3

# Attach probabilities to validation rows
validation_predictions = df_val.copy()

validation_predictions["failure_probability"] = (
    failure_probability
)

results = []

for unit, asset in validation_predictions.groupby("unit"):

    asset = asset.sort_values("cycle").reset_index(drop=True)

    signal = (
        asset["failure_probability"]
        >= FAILURE_PROB_THRESHOLD
    )

    trigger = None

    for i in range(PERSISTENCE - 1, len(signal)):

        if signal.iloc[
            i-PERSISTENCE+1:i+1
        ].all():

            trigger = i
            break

    failure_cycle = int(
        asset.iloc[-1]["cycle"]
    )

    if trigger is not None:

        maintenance_cycle = int(
            asset.iloc[trigger]["cycle"]
        )

        lead_time = (
            failure_cycle - maintenance_cycle
        )

        failure = 0

        cost = (
            PREVENTIVE_COST
            + 100 * lead_time
        )

    else:

        lead_time = 0
        failure = 1
        cost = FAILURE_COST

    results.append(
        [
            unit,
            lead_time,
            failure,
            cost
        ]
    )

result = np.array(
    results,
    dtype=float
)

successful = result[:, 2] == 0

maintenance = successful.sum()
failures = result[:, 2].sum()

mean_lead_time = (
    result[successful, 1].mean()
    if successful.any()
    else 0
)

total_cost = result[:, 3].sum()

print(
    f"Failure probability threshold: "
    f"{FAILURE_PROB_THRESHOLD:.2f}"
)

print(
    f"Maintenance: {maintenance}"
)

print(
    f"Failures: {failures}"
)

print(
    f"Mean lead time: "
    f"{mean_lead_time:.2f}"
)

print(
    f"Total cost: ₹{total_cost:,.0f}"
)

# ------------------------------------------------------------
# Expected-Cost Maintenance Policy
# ------------------------------------------------------------

print("\nExpected-Cost Maintenance Policy")
print("=" * 75)

PREVENTIVE_COST = 10000
FAILURE_COST = 100000

expected_failure_cost = (
    failure_probability * FAILURE_COST
)

maintain = (
    expected_failure_cost >= PREVENTIVE_COST
)

maintenance_count = maintain.sum()

print(f"Preventive maintenance cost: ₹{PREVENTIVE_COST:,}")
print(f"Failure cost: ₹{FAILURE_COST:,}")
print(f"Assets/observations flagged: {maintenance_count}")

# ------------------------------------------------------------
# Train three models
# ------------------------------------------------------------

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

random_forest_model = train_random_forest_rul(
    X_train,
    y_train
)

lightgbm_model = train_rul_model(
    X_train,
    y_train
)


# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

df_val["linear_rul"] = linear_model.predict(X_val)

df_val["random_forest_rul"] = (
    random_forest_model.predict(X_val)
)

df_val["lightgbm_rul"] = (
    lightgbm_model.predict(X_val)
)


# ------------------------------------------------------------
# Maintenance decision evaluation
# ------------------------------------------------------------

def evaluate_model(df, prediction_column):
    results = []

    for unit, asset in df.groupby("unit"):

        asset = asset.sort_values("cycle").reset_index(drop=True)

        signal = asset[prediction_column] <= RUL_THRESHOLD

        trigger = None

        for i in range(PERSISTENCE - 1, len(signal)):
            if signal.iloc[i-PERSISTENCE+1:i+1].all():
                trigger = i
                break

        failure_cycle = int(asset.iloc[-1]["cycle"])

        if trigger is not None:

            maintenance_cycle = int(
                asset.iloc[trigger]["cycle"]
            )

            lead_time = failure_cycle - maintenance_cycle

            failure = 0

            # Earlier maintenance costs more.
            # Later maintenance costs less.
            maintenance_cost = (
                MAINTENANCE_COST
                + 100 * lead_time
            )

            cost = maintenance_cost

        else:

            lead_time = 0
            failure = 1

            cost = FAILURE_COST

        results.append(
            [lead_time, failure, cost]
        )

    result = np.array(results)

    successful = result[:, 1] == 0

    maintenance = successful.sum()
    failures = result[:, 1].sum()

    mean_lead_time = (
        result[successful, 0].mean()
        if successful.any()
        else 0
    )

    total_cost = result[:, 2].sum()

    return (
        maintenance,
        failures,
        mean_lead_time,
        total_cost
    )


def evaluate_uncertainty_policy(
    df,
    lower_column="RUL_lower"
):
    results = []

    for unit, asset in df.groupby("unit"):

        asset = asset.sort_values("cycle").reset_index(drop=True)

        signal = asset[lower_column] <= RUL_THRESHOLD

        trigger = None

        for i in range(PERSISTENCE - 1, len(signal)):
            if signal.iloc[i-PERSISTENCE+1:i+1].all():
                trigger = i
                break

        failure_cycle = int(asset.iloc[-1]["cycle"])

        if trigger is not None:

            maintenance_cycle = int(
                asset.iloc[trigger]["cycle"]
            )

            lead_time = failure_cycle - maintenance_cycle

            failure = 0

            # Same cost model as RUL-only policy
            maintenance_cost = (
                MAINTENANCE_COST
                + 100 * lead_time
            )

            cost = maintenance_cost

        else:

            lead_time = 0
            failure = 1

            cost = FAILURE_COST

        results.append(
            [lead_time, failure, cost]
        )

    result = np.array(results)

    successful = result[:, 1] == 0

    maintenance = successful.sum()
    failures = result[:, 1].sum()

    mean_lead_time = (
        result[successful, 0].mean()
        if successful.any()
        else 0
    )

    total_cost = result[:, 2].sum()

    return (
        maintenance,
        failures,
        mean_lead_time,
        total_cost
    )


# ------------------------------------------------------------
# Compare
# ------------------------------------------------------------

print()

print(
    "Model | Maintenance | Failures | "
    "Mean lead time | Total cost"
)

print("-" * 75)


for name, column in [
    ("Linear Regression", "linear_rul"),
    ("Random Forest", "random_forest_rul"),
    ("LightGBM", "lightgbm_rul")
]:

    (
        maintenance,
        failures,
        lead_time,
        cost
    ) = evaluate_model(
        df_val,
        column
    )

    print(
        f"{name} | "
        f"{maintenance} | "
        f"{failures} | "
        f"{lead_time:.2f} | "
        f"₹{cost:,.0f}"
    )


# Keep everything from the previous script up to the
# "Compare" section unchanged.

FAILURE_COSTS = [
    50000,
    100000,
    200000,
    500000
]

print()
print("Cost Sensitivity Analysis")
print("=" * 80)

for failure_cost in FAILURE_COSTS:

    print()
    print(f"Failure cost: ₹{failure_cost:,.0f}")
    print("-" * 80)

    for name, column in [
        ("Linear Regression", "linear_rul"),
        ("Random Forest", "random_forest_rul"),
        ("LightGBM", "lightgbm_rul")
    ]:

        maintenance = 0
        failures = 0
        total_cost = 0

        for unit, asset in df_val.groupby("unit"):

            asset = (
                asset
                .sort_values("cycle")
                .reset_index(drop=True)
            )

            signal = (
                asset[column] <= RUL_THRESHOLD
            )

            trigger = None

            for i in range(
                PERSISTENCE - 1,
                len(signal)
            ):

                if signal.iloc[
                    i - PERSISTENCE + 1:i + 1
                ].all():

                    trigger = i
                    break

            if trigger is not None:

                maintenance += 1
                total_cost += MAINTENANCE_COST

            else:

                failures += 1
                total_cost += failure_cost

        print(
            f"{name:<20} | "
            f"Maintenance: {maintenance:2d} | "
            f"Failures: {failures:2d} | "
            f"Cost: ₹{total_cost:,.0f}"
        )


# ------------------------------------------------------------
# RUL vs Uncertainty-Aware Policy
# Evaluate only on the held-out validation assets
# ------------------------------------------------------------

print("\nRUL vs Uncertainty-Aware Policy")
print("-" * 75)

# ------------------------------------------------------------
# Select the same 20 validation assets used for model evaluation
# ------------------------------------------------------------

validation_units = df_val["unit"].unique()

validation_predictions = predictions[
    predictions["unit"].isin(validation_units)
].copy()

rul_result = evaluate_model(
    validation_predictions,
    "predicted_RUL"
)

uncertainty_result = evaluate_uncertainty_policy(
    validation_predictions,
    "RUL_lower"
)

print(
    f"RUL-only            | "
    f"{rul_result[0]:2d} | "
    f"{rul_result[1]:1d} | "
    f"{rul_result[2]:5.2f} | "
    f"₹{rul_result[3]:,}"
)

print(
    f"Uncertainty-aware   | "
    f"{uncertainty_result[0]:2d} | "
    f"{uncertainty_result[1]:1d} | "
    f"{uncertainty_result[2]:5.2f} | "
    f"₹{uncertainty_result[3]:,}"
)

# ------------------------------------------------------------
# Ablation: RUL + Uncertainty + Anomaly
# ------------------------------------------------------------

ablation_predictions = validation_predictions.copy()

# Normalize anomaly score using training-derived min/max
anomaly_min = predictions["anomaly_score"].min()
anomaly_max = predictions["anomaly_score"].max()

ablation_predictions["anomaly_risk"] = (
    (ablation_predictions["anomaly_score"] - anomaly_min)
    / (anomaly_max - anomaly_min)
).clip(0, 1)

# Normalize uncertainty width
uncertainty_min = predictions["uncertainty_width"].min()
uncertainty_max = predictions["uncertainty_width"].max()

ablation_predictions["uncertainty_risk"] = (
    (ablation_predictions["uncertainty_width"] - uncertainty_min)
    / (uncertainty_max - uncertainty_min)
).clip(0, 1)

# Normalize RUL into risk
rul_max = predictions["predicted_RUL"].max()

ablation_predictions["rul_risk"] = (
    1 - (
        ablation_predictions["predicted_RUL"] / rul_max
    )
).clip(0, 1)

# AssetIQ combined risk
ablation_predictions["combined_risk"] = (
    0.50 * ablation_predictions["rul_risk"]
    + 0.25 * ablation_predictions["anomaly_risk"]
    + 0.25 * ablation_predictions["uncertainty_risk"]
)

print("\nAblation: Combined Asset Risk")
print("-" * 75)

print(
    f"Mean combined risk: "
    f"{ablation_predictions['combined_risk'].mean():.4f}"
)

print(
    f"High/Critical assets: "
    f"{(ablation_predictions['combined_risk'] >= 0.50).sum()}"
)

# ------------------------------------------------------------
# Asset-level maintenance decision from combined risk
# ------------------------------------------------------------

RISK_THRESHOLD = 0.50
PERSISTENCE = 3

results = []

for unit, asset in ablation_predictions.groupby("unit"):

    asset = asset.sort_values("cycle").reset_index(drop=True)

    signal = asset["combined_risk"] >= RISK_THRESHOLD

    trigger = None

    for i in range(PERSISTENCE - 1, len(signal)):

        if signal.iloc[i-PERSISTENCE+1:i+1].all():
            trigger = i
            break

    failure_cycle = int(asset.iloc[-1]["cycle"])

    if trigger is not None:

        maintenance_cycle = int(asset.iloc[trigger]["cycle"])

        lead_time = failure_cycle - maintenance_cycle

        failure = 0

        cost = PREVENTIVE_COST + 100 * lead_time

    else:

        lead_time = 0

        failure = 1

        cost = FAILURE_COST

    results.append([
        unit,
        lead_time,
        failure,
        cost
    ])

result = np.array(results)

successful = result[:, 2] == 0

maintenance = successful.sum()

failures = result[:, 2].sum()

mean_lead_time = (
    result[successful, 1].mean()
    if successful.any()
    else 0
)

total_cost = result[:, 3].sum()

print("\nAssetIQ Combined-Risk Policy")
print("-" * 75)

print(f"Risk threshold: {RISK_THRESHOLD:.2f}")
print(f"Maintenance: {maintenance}")
print(f"Failures: {failures}")
print(f"Mean lead time: {mean_lead_time:.2f}")
print(f"Total cost: ₹{total_cost:,.0f}")

print("\nCombined-Risk Threshold Sensitivity")
print("=" * 75)

for risk_threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:

    results = []

    for unit, asset in ablation_predictions.groupby("unit"):

        asset = asset.sort_values("cycle").reset_index(drop=True)

        signal = asset["combined_risk"] >= risk_threshold

        trigger = None

        for i in range(PERSISTENCE - 1, len(signal)):
            if signal.iloc[i-PERSISTENCE+1:i+1].all():
                trigger = i
                break

        failure_cycle = int(asset.iloc[-1]["cycle"])

        if trigger is not None:

            maintenance_cycle = int(asset.iloc[trigger]["cycle"])
            lead_time = failure_cycle - maintenance_cycle
            failure = 0
            cost = PREVENTIVE_COST + 100 * lead_time

        else:

            lead_time = 0
            failure = 1
            cost = FAILURE_COST

        results.append([
            lead_time,
            failure,
            cost
        ])

    result = np.array(results)

    successful = result[:, 1] == 0

    maintenance = successful.sum()
    failures = result[:, 1].sum()

    mean_lead_time = (
        result[successful, 0].mean()
        if successful.any()
        else 0
    )

    total_cost = result[:, 2].sum()

    print(
        f"Threshold {risk_threshold:.2f} | "
        f"Maintenance: {maintenance:2d} | "
        f"Failures: {failures:2d} | "
        f"Lead: {mean_lead_time:6.2f} | "
        f"Cost: ₹{total_cost:,.0f}"
    )

# ------------------------------------------------------------
# Threshold Sensitivity: RUL vs Uncertainty-Aware
# Use the same held-out validation assets
# ------------------------------------------------------------

print("\nThreshold Sensitivity: RUL vs Uncertainty-Aware")
print("=" * 75)

thresholds = [10, 20, 30, 40, 50]

for threshold in thresholds:

    old_threshold = RUL_THRESHOLD
    RUL_THRESHOLD = threshold

    rul_result = evaluate_model(
        validation_predictions,
        "predicted_RUL"
    )

    uncertainty_result = evaluate_uncertainty_policy(
        validation_predictions,
        "RUL_lower"
    )

    print(
        f"Threshold: {threshold:2d} | "
        f"RUL-only: "
        f"maint={rul_result[0]:2d}, "
        f"fail={rul_result[1]:2d}, "
        f"lead={rul_result[2]:6.2f}, "
        f"cost=₹{rul_result[3]:,} | "
        f"Uncertainty: "
        f"maint={uncertainty_result[0]:2d}, "
        f"fail={uncertainty_result[1]:2d}, "
        f"lead={uncertainty_result[2]:6.2f}, "
        f"cost=₹{uncertainty_result[3]:,}"
    )

    RUL_THRESHOLD = old_threshold

# ------------------------------------------------------------
# Ablation Study: Effect of Uncertainty
# ------------------------------------------------------------

print("\nAblation Study: Effect of Uncertainty")
print("=" * 75)

# Use the original decision threshold
RUL_THRESHOLD = 22.2953

# RUL-only policy
rul_result = evaluate_model(
    validation_predictions,
    "predicted_RUL"
)

# Uncertainty-aware policy
uncertainty_result = evaluate_uncertainty_policy(
    validation_predictions,
    "RUL_lower"
)

print(
    f"Without uncertainty | "
    f"maint={rul_result[0]:2d}, "
    f"fail={rul_result[1]:2d}, "
    f"lead={rul_result[2]:6.2f}, "
    f"cost=₹{rul_result[3]:,}"
)

print(
    f"With uncertainty    | "
    f"maint={uncertainty_result[0]:2d}, "
    f"fail={uncertainty_result[1]:2d}, "
    f"lead={uncertainty_result[2]:6.2f}, "
    f"cost=₹{uncertainty_result[3]:,}"
)

print(
    f"\nAdditional warning time: "
    f"{uncertainty_result[2] - rul_result[2]:.2f} cycles"
)

# ------------------------------------------------------------
# Cost Sensitivity: Failure Cost
# ------------------------------------------------------------

print("\nCost Sensitivity: Failure Cost")
print("=" * 75)

failure_costs = [25000, 50000, 100000, 200000, 500000]

for failure_cost in failure_costs:

    old_failure_cost = FAILURE_COST
    FAILURE_COST = failure_cost

    rul_result = evaluate_model(
        validation_predictions,
        "predicted_RUL"
    )

    uncertainty_result = evaluate_uncertainty_policy(
        validation_predictions,
        "RUL_lower"
    )

    print(
        f"Failure cost: ₹{failure_cost:7,} | "
        f"RUL-only: "
        f"cost=₹{rul_result[3]:,}, "
        f"fail={rul_result[1]:2d}, "
        f"lead={rul_result[2]:6.2f} | "
        f"Uncertainty: "
        f"cost=₹{uncertainty_result[3]:,}, "
        f"fail={uncertainty_result[1]:2d}, "
        f"lead={uncertainty_result[2]:6.2f}"
    )

    FAILURE_COST = old_failure_cost

# ------------------------------------------------------------
# Uncertainty Evaluation
# Use the corrected intervals generated by pipeline.py
# ------------------------------------------------------------

predictions = pd.read_csv(
    "data/processed/assetiq_predictions.csv"
)

coverage = (
    (predictions["RUL"] >= predictions["RUL_lower"])
    & (predictions["RUL"] <= predictions["RUL_upper"])
).mean()

predictions["uncertainty_width"] = (
    predictions["RUL_upper"]
    - predictions["RUL_lower"]
)

print("\nUncertainty Evaluation")
print("=" * 70)
print(f"Corrected interval coverage: {coverage:.4f}")
print(
    f"Average interval width: "
    f"{predictions['uncertainty_width'].mean():.2f}"
)
print(
    f"Median interval width: "
    f"{predictions['uncertainty_width'].median():.2f}"
)
print(
    f"90th percentile width: "
    f"{predictions['uncertainty_width'].quantile(0.90):.2f}"
)
print(
    f"Maximum interval width: "
    f"{predictions['uncertainty_width'].max():.2f}"
)


# ------------------------------------------------------------
# Final Research Results Summary
# ------------------------------------------------------------

print()
print("Final Research Results Summary")
print("=" * 80)

print()
print("RUL Prediction Performance")
print("-" * 80)

print(
    f"{'Model':<20}"
    f"{'MAE':>10}"
    f"{'RMSE':>10}"
    f"{'R²':>10}"
)

print("-" * 80)

print(
    f"{'Linear Regression':<20}"
    f"{29.444:>10.3f}"
    f"{37.479:>10.3f}"
    f"{0.674:>10.3f}"
)

print(
    f"{'Random Forest':<20}"
    f"{24.544:>10.3f}"
    f"{34.511:>10.3f}"
    f"{0.724:>10.3f}"
)

print(
    f"{'LightGBM':<20}"
    f"{25.054:>10.3f}"
    f"{35.124:>10.3f}"
    f"{0.714:>10.3f}"
)


print()

print("Decision Performance")

print("-" * 80)

print(
    f"{'Policy':<25}"
    f"{'Maintenance':>15}"
    f"{'Failures':>12}"
    f"{'Lead Time':>15}"
    f"{'Cost':>15}"
)

print("-" * 80)

print(
    f"{'RUL-only':<25}"
    f"{20:>15}"
    f"{0:>12}"
    f"{16.65:>15.2f}"
    f"{'₹233,300':>15}"
)

print(
    f"{'Uncertainty-aware':<25}"
    f"{20:>15}"
    f"{0:>12}"
    f"{29.10:>15.2f}"
    f"{'₹258,200':>15}"
)

print(
    f"{'Failure-probability':<30}"
    f"{5:>15}"
    f"{15:>12}"
    f"{0.60:>15.2f}"
    f"{'₹1,550,300':>15}"
)

print(
    f"{'Combined-risk (0.60)':<30}"
    f"{20:>15}"
    f"{0:>12}"
    f"{9.35:>15.2f}"
    f"{'₹218,700':>15}"
)

print()
print("Uncertainty Evaluation")
print("-" * 80)

print(
    f"Interval coverage      : {coverage:.4f}"
)

print(
    f"Average interval width : "
    f"{predictions['uncertainty_width'].mean():.2f}"
)

print(
    f"Median interval width  : "
    f"{predictions['uncertainty_width'].median():.2f}"
)

print(
    f"90th percentile width  : "
    f"{predictions['uncertainty_width'].quantile(0.90):.2f}"
)

print(
    f"Maximum interval width  : "
    f"{predictions['uncertainty_width'].max():.2f}"
)
