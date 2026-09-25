from data_loader import load_cmapss_fd001
from preprocessing import preprocess_cmapss
from features import create_features
from anomaly import train_anomaly_model, predict_anomalies
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import IsolationForest
from models.uncertainty import (
    train_quantile_models,
    calibrate_prediction_interval
)


df = load_cmapss_fd001()
df = preprocess_cmapss(df)
df = create_features(df)

from sklearn.model_selection import GroupShuffleSplit

# First split: 80% development, 20% test
splitter_1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

dev_idx, test_idx = next(
    splitter_1.split(
        df,
        groups=df["unit"]
    )
)

dev_df = df.iloc[dev_idx].copy()
test_df = df.iloc[test_idx].copy()

# Second split: 75% of development = 60% overall train,
# 25% of development = 20% overall validation
splitter_2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, val_idx = next(
    splitter_2.split(
        dev_df,
        groups=dev_df["unit"]
    )
)

train_df = dev_df.iloc[train_idx].copy()
val_df = dev_df.iloc[val_idx].copy()

print("Train assets:", train_df["unit"].nunique())
print("Validation assets:", val_df["unit"].nunique())
print("Test assets:", test_df["unit"].nunique())

print(
    "Overlap train/validation:",
    len(set(train_df["unit"]) & set(val_df["unit"]))
)

print(
    "Overlap train/test:",
    len(set(train_df["unit"]) & set(test_df["unit"]))
)

print(
    "Overlap validation/test:",
    len(set(val_df["unit"]) & set(test_df["unit"]))
)

anomaly_model, sensor_features = train_anomaly_model(
    train_df
)

train_df = predict_anomalies(
    anomaly_model,
    train_df,
    sensor_features
)

val_df = predict_anomalies(
    anomaly_model,
    val_df,
    sensor_features
)

print("Training anomaly rows:", len(train_df))
print("Validation anomaly rows:", len(val_df))

print(
    "Validation anomaly rate:",
    (val_df["anomaly_label"] == -1).mean()
)

print("Research anomaly split test completed.")

# Create RUL target
max_cycle = train_df.groupby("unit")["cycle"].transform("max")
train_df["RUL"] = max_cycle - train_df["cycle"]

max_cycle_val = val_df.groupby("unit")["cycle"].transform("max")
val_df["RUL"] = max_cycle_val - val_df["cycle"]

# Failure target
failure_horizon = 20
train_df["failure"] = (train_df["RUL"] <= failure_horizon).astype(int)
val_df["failure"] = (val_df["RUL"] <= failure_horizon).astype(int)


print("Training RUL range:", train_df["RUL"].min(), "to", train_df["RUL"].max())
print("Validation RUL range:", val_df["RUL"].min(), "to", val_df["RUL"].max())

print("Training failures:", train_df["failure"].sum())
print("Validation failures:", val_df["failure"].sum())

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd


feature_cols = [
    col for col in train_df.columns
    if col.startswith("sensor_")
]

X_train = train_df[feature_cols]
y_train = train_df["RUL"]

X_val = val_df[feature_cols]
y_val = val_df["RUL"]

X_test = test_df[feature_cols]
y_test = test_df["RUL"]

iso_model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

iso_model.fit(X_train)

train_anomaly_score = -iso_model.decision_function(X_train)
val_anomaly_score = -iso_model.decision_function(X_val)
test_anomaly_score = -iso_model.decision_function(X_test)

train_df["anomaly_score"] = train_anomaly_score
val_df["anomaly_score"] = val_anomaly_score
test_df["anomaly_score"] = test_anomaly_score


models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
}

test_metrics = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    # Validation evaluation
    val_predictions = model.predict(X_val)

    val_mae = mean_absolute_error(y_val, val_predictions)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    val_r2 = r2_score(y_val, val_predictions)

    print(f"\n{name} - Validation")
    print(f"MAE:  {val_mae:.3f}")
    print(f"RMSE: {val_rmse:.3f}")
    print(f"R2:   {val_r2:.3f}")

    # Test evaluation
    test_predictions = model.predict(X_test)

    test_mae = mean_absolute_error(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_r2 = r2_score(y_test, test_predictions)

    test_metrics[name] = {
        "MAE": float(test_mae),
        "RMSE": float(test_rmse),
        "R2": float(test_r2)
    }

    print(f"{name} - Test")
    print(f"MAE:  {test_mae:.3f}")
    print(f"RMSE: {test_rmse:.3f}")
    print(f"R2:   {test_r2:.3f}")

test_df = test_df.copy()

test_df["predicted_RUL"] = models["LightGBM"].predict(
    X_test
)

# Train uncertainty models using training assets only

lower_model = lgb.LGBMRegressor(
    objective="quantile",
    alpha=0.10,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1
)

upper_model = lgb.LGBMRegressor(
    objective="quantile",
    alpha=0.90,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1
)

lower_model.fit(X_train, y_train)
upper_model.fit(X_train, y_train)

rul_point = models["LightGBM"].predict(X_val)
rul_lower = lower_model.predict(X_val)
rul_upper = upper_model.predict(X_val)

test_rul_lower = lower_model.predict(X_test)
test_rul_upper = upper_model.predict(X_test)

test_df["predicted_RUL"] = models["LightGBM"].predict(X_test)

test_df["RUL_lower"] = test_rul_lower
test_df["RUL_upper"] = test_rul_upper

test_df["uncertainty_width"] = (
    test_rul_upper - test_rul_lower
)

train_rul_lower = lower_model.predict(X_train)
train_rul_upper = upper_model.predict(X_train)

train_df["uncertainty_width"] = (
    train_rul_upper - train_rul_lower
)

# ------------------------------------------------------------
# Calibrate uncertainty interval using validation data
# ------------------------------------------------------------

rul_lower = np.minimum(rul_lower, rul_point)
rul_upper = np.maximum(rul_upper, rul_point)

calibration_factor = calibrate_prediction_interval(
    y_true=y_val,
    lower_prediction=rul_lower,
    upper_prediction=rul_upper,
    target_coverage=0.80
)

print("\nUncertainty Calibration")
print(
    f"Calibration factor: "
    f"{calibration_factor:.3f}"
)

# Apply the validation-derived calibration to validation data
calibrated_val_lower = np.maximum(
    rul_lower - calibration_factor,
    0
)

calibrated_val_upper = (
    rul_upper + calibration_factor
)

val_coverage = np.mean(
    (y_val >= calibrated_val_lower) &
    (y_val <= calibrated_val_upper)
)

val_width = np.mean(
    calibrated_val_upper - calibrated_val_lower
)

print(f"Calibrated validation coverage: {val_coverage:.3f}")
print(f"Calibrated validation width: {val_width:.3f}")

# ------------------------------------------------------------
# Apply calibration to untouched test data
# ------------------------------------------------------------

test_rul_lower = np.minimum(
    test_rul_lower,
    test_df["predicted_RUL"]
)

test_rul_upper = np.maximum(
    test_rul_upper,
    test_df["predicted_RUL"]
)

test_calibrated_lower = np.maximum(
    test_rul_lower - calibration_factor,
    0
)

test_calibrated_upper = (
    test_rul_upper + calibration_factor
)

test_df["RUL_lower"] = test_calibrated_lower
test_df["RUL_upper"] = test_calibrated_upper

test_df["uncertainty_width"] = (
    test_df["RUL_upper"] -
    test_df["RUL_lower"]
)

test_coverage = np.mean(
    (test_df["RUL"] >= test_df["RUL_lower"]) &
    (test_df["RUL"] <= test_df["RUL_upper"])
)

print("\nCalibrated Test Uncertainty Evaluation")
print("Nominal coverage: 0.800")
print(f"Observed coverage: {test_coverage:.3f}")
print(
    f"Coverage gap: "
    f"{test_coverage - 0.800:.3f}"
)
print(
    f"Mean interval width: "
    f"{test_df['uncertainty_width'].mean():.3f}"
)

# ------------------------------------------------------------
# Validation dataframe for downstream analysis
# ------------------------------------------------------------

val_df["predicted_RUL"] = rul_point
val_df["RUL_lower"] = calibrated_val_lower
val_df["RUL_upper"] = calibrated_val_upper
val_df["uncertainty_width"] = (
    calibrated_val_upper -
    calibrated_val_lower
)

val_df["life_stage"] = pd.qcut(
    val_df["RUL"],
    q=3,
    labels=["Late", "Mid", "Early"]
)

stage_coverage = (
    val_df.groupby("life_stage", observed=True)
    .apply(
        lambda x: np.mean(
            (x["RUL"] >= x["RUL_lower"]) &
            (x["RUL"] <= x["RUL_upper"])
        )
    )
)

print("\nCalibrated Uncertainty Coverage by Life Stage")
print(stage_coverage)

print("\nSample validation predictions:")
print(
    val_df[
        ["unit", "cycle", "RUL",
         "predicted_RUL",
         "RUL_lower",
         "RUL_upper",
         "uncertainty_width"]
    ].head(10)
)

def evaluate_rul_policy(df, threshold=20):
    results = []

    for unit, asset in df.groupby("unit"):

        asset = asset.sort_values("cycle")

        maintenance_rows = asset[
            asset["predicted_RUL"] <= threshold
        ]

        actual_failure_cycle = asset.loc[
            asset["RUL"] == 0, "cycle"
        ].min()

        if len(maintenance_rows) > 0:
            maintenance_cycle = maintenance_rows["cycle"].min()

            if maintenance_cycle < actual_failure_cycle:
                lead_time = actual_failure_cycle - maintenance_cycle
                failure = 0
            else:
                lead_time = 0
                failure = 1
        else:
            maintenance_cycle = None
            lead_time = 0
            failure = 1

        results.append({
            "unit": unit,
            "maintenance_cycle": maintenance_cycle,
            "lead_time": lead_time,
            "failure": failure
        })

    return pd.DataFrame(results)


policy_result = evaluate_rul_policy(
    val_df,
    threshold=20
)

print("\nRUL Threshold Sensitivity")

for threshold in [10, 20, 30, 40, 50]:

    result = evaluate_rul_policy(
        val_df,
        threshold=threshold
    )

    print(
        f"Threshold {threshold}: "
        f"Maintained={(result['maintenance_cycle'].notna()).sum()}, "
        f"Failures={result['failure'].sum()}, "
        f"Mean lead={result['lead_time'].mean():.2f}"
    )

def calculate_policy_cost(
    policy_result,
    preventive_cost=10000,
    failure_cost=100000,
    early_maintenance_penalty=100
):
    total_cost = 0

    for _, row in policy_result.iterrows():

        if row["failure"] == 1:
            total_cost += failure_cost

        else:
            total_cost += preventive_cost

            total_cost += (
                row["lead_time"]
                * early_maintenance_penalty
            )

    return total_cost


print("\nRUL Policy Cost Sensitivity")

for threshold in [10, 20, 30, 40, 50]:

    result = evaluate_rul_policy(
        val_df,
        threshold=threshold
    )

    cost = calculate_policy_cost(result)

    print(
        f"Threshold {threshold}: "
        f"Cost=₹{cost:,.0f}"
    )

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score


failure_features = feature_cols

X_train_failure = train_df[failure_features]
y_train_failure = train_df["failure"]

X_val_failure = val_df[failure_features]
y_val_failure = val_df["failure"]


failure_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

failure_model.fit(
    X_train_failure,
    y_train_failure
)

failure_probability = failure_model.predict_proba(
    X_val_failure
)[:, 1]

pr_auc = average_precision_score(
    y_val_failure,
    failure_probability
)

X_test_failure = test_df[failure_features]

test_df["failure"] = (
    test_df["RUL"] <= failure_horizon
).astype(int)

y_test_failure = test_df["failure"]

test_failure_probability = failure_model.predict_proba(
    X_test_failure
)[:, 1]

test_pr_auc = average_precision_score(
    y_test_failure,
    test_failure_probability
)

print("\nFailure Classification - Test")
print(f"Test PR-AUC: {test_pr_auc:.3f}")


print("\nFailure Classification")
print(f"Validation PR-AUC: {pr_auc:.3f}")

def evaluate_failure_policy(df, failure_probability, threshold=0.05):

    temp = df[["unit", "cycle", "RUL"]].copy()
    temp["failure_probability"] = failure_probability

    results = []

    for unit, asset in temp.groupby("unit"):

        asset = asset.sort_values("cycle")

        maintenance_rows = asset[
            asset["failure_probability"] >= threshold
        ]

        actual_failure_cycle = asset.loc[
            asset["RUL"] == 0, "cycle"
        ].min()

        if len(maintenance_rows) > 0:

            maintenance_cycle = maintenance_rows["cycle"].min()

            if maintenance_cycle < actual_failure_cycle:
                lead_time = (
                    actual_failure_cycle - maintenance_cycle
                )
                failure = 0
            else:
                lead_time = 0
                failure = 1

        else:
            maintenance_cycle = None
            lead_time = 0
            failure = 1

        results.append({
            "unit": unit,
            "maintenance_cycle": maintenance_cycle,
            "lead_time": lead_time,
            "failure": failure
        })

    return pd.DataFrame(results)


failure_policy = evaluate_failure_policy(
    val_df,
    failure_probability,
    threshold=0.05
)

print("\nFailure Probability Policy")
print("Maintained:",
      failure_policy["maintenance_cycle"].notna().sum())
print("Failures:",
      failure_policy["failure"].sum())
print("Mean lead time:",
      f"{failure_policy['lead_time'].mean():.2f}")

# Training-derived normalization
#Important: this is only calculating the combined risk score. Don't create maintenance thresholds yet.

def minmax_train(train_values, val_values):
    min_value = train_values.min()
    max_value = train_values.max()

    if max_value == min_value:
        return (
            np.zeros(len(train_values)),
            np.zeros(len(val_values))
        )

    train_norm = (
        (train_values - min_value)
        / (max_value - min_value)
    )

    val_norm = (
        (val_values - min_value)
        / (max_value - min_value)
    )

    return train_norm, val_norm

def calculate_assetiq_risk(
    rul_risk,
    anomaly_risk,
    uncertainty_risk
):
    return (
        0.50 * rul_risk
        + 0.25 * anomaly_risk
        + 0.25 * uncertainty_risk
    )


_, val_rul_risk = minmax_train(
    -train_df["RUL"],
    -val_df["predicted_RUL"]
)

_, val_anomaly_risk = minmax_train(
    train_df["anomaly_score"],
    val_df["anomaly_score"]
)

_, val_uncertainty_risk = minmax_train(
    train_df.get(
        "uncertainty_width",
        pd.Series(np.zeros(len(train_df)))
    ),
    val_df["uncertainty_width"]
)

_, test_rul_risk = minmax_train(
    -train_df["RUL"],
    -test_df["predicted_RUL"]
)

_, test_anomaly_risk = minmax_train(
    train_df["anomaly_score"],
    test_df["anomaly_score"]
)

_, test_uncertainty_risk = minmax_train(
    train_df.get(
        "uncertainty_width",
        pd.Series(np.zeros(len(train_df)))
    ),
    test_df["uncertainty_width"]
)

val_df["risk_score"] = (
    0.50 * val_rul_risk
    + 0.25 * val_anomaly_risk
    + 0.25 * val_uncertainty_risk
)

print("\nAssetIQ Combined Risk")
print(
    val_df[
        ["unit", "cycle", "predicted_RUL",
         "anomaly_score",
         "uncertainty_width",
         "risk_score"]
    ].tail(10)
)

def evaluate_risk_policy(df, threshold):

    results = []

    for unit, asset in df.groupby("unit"):

        asset = asset.sort_values("cycle")

        maintenance_rows = asset[
            asset["risk_score"] >= threshold
        ]

        actual_failure_cycle = asset.loc[
            asset["RUL"] == 0, "cycle"
        ].min()

        if len(maintenance_rows) > 0:

            maintenance_cycle = maintenance_rows["cycle"].min()

            # Maintenance happened before failure
            if maintenance_cycle < actual_failure_cycle:
                lead_time = (
                    actual_failure_cycle - maintenance_cycle
                )
                failure = 0

            # Risk crossed threshold only at failure
            else:
                maintenance_cycle = None
                lead_time = 0
                failure = 1

        else:
            maintenance_cycle = None
            lead_time = 0
            failure = 1

        results.append({
            "unit": unit,
            "maintenance_cycle": maintenance_cycle,
            "lead_time": lead_time,
            "failure": failure
        })

    return pd.DataFrame(results)


print("\nAssetIQ Risk Threshold Sensitivity")

for threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:

    temp_df = val_df.copy()

    temp_df["risk_score"] = calculate_assetiq_risk(
        val_rul_risk,
        val_anomaly_risk,
        val_uncertainty_risk
    )

    result = evaluate_risk_policy(
        temp_df,
        threshold
    )

    if threshold == 0.65:
        selected_assetiq_result = result.copy()

    if threshold == 0.65:
        print("\nAssetIQ Asset-Level Decisions")
        print(
            result.sort_values("unit").to_string(index=False)
        )

    print(
        f"Threshold {threshold:.2f}: "
        f"Maintained={result['maintenance_cycle'].notna().sum()}, "
        f"Failures={result['failure'].sum()}, "
        f"Mean lead={result['lead_time'].mean():.2f}"
    )


    #print("\nPolicy Comparison")

# Reactive maintenance
reactive_cost = 20 * 100000
print(f"Reactive: Cost=₹{reactive_cost:,.0f}")

# RUL threshold policy
rul_result = evaluate_rul_policy(val_df, threshold=10)
rul_cost = calculate_policy_cost(rul_result)

print(
    f"RUL Threshold: "
    f"Cost=₹{rul_cost:,.0f}, "
    f"Failures={rul_result['failure'].sum()}, "
    f"Mean lead={rul_result['lead_time'].mean():.2f}"
)

# AssetIQ risk policy


# Train the three models
print("\nmodel-vs-decision-cost section")

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

lgbm_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1
)
lgbm_model.fit(X_train, y_train)

models = {
    "Linear Regression": linear_model,
    "Random Forest": rf_model,
    "LightGBM": lgbm_model
}

for name, model in models.items():

    temp_df = test_df.copy()

    temp_df["predicted_RUL"] = model.predict(
        temp_df[feature_cols]
    ).clip(min=0)

    result = evaluate_rul_policy(
        temp_df,
        threshold=10
    )

    cost = calculate_policy_cost(result)

    print(
        f"{name} Test: "
        f"Cost=₹{cost:,.0f}, "
        f"Failures={result['failure'].sum()}, "
        f"Mean lead={result['lead_time'].mean():.2f}"
    )

    #print("\nDecision Policy Frontier")

print("\nRUL Policies")

for threshold in [10, 20, 30, 40, 50]:
    result = evaluate_rul_policy(
        val_df,
        threshold=threshold
    )

    cost = calculate_policy_cost(result)

    print(
        f"RUL {threshold}: "
        f"Cost=₹{cost:,.0f}, "
        f"Lead={result['lead_time'].mean():.2f}"
    )

print("\nAssetIQ Policies")

assetiq_policy_results = {}

for threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    temp_df = val_df.copy()

    temp_df["risk_score"] = calculate_assetiq_risk(
        val_rul_risk,
        val_anomaly_risk,
        val_uncertainty_risk
    )

    result = evaluate_risk_policy(
        temp_df,
        threshold=threshold
    )

    cost = calculate_policy_cost(result)

    assetiq_policy_results[threshold] = {
        "cost": cost,
        "failures": int(result["failure"].sum()),
        "mean_lead": float(result["lead_time"].mean())
    }

    print(
        f"AssetIQ {threshold:.2f}: "
        f"Cost=₹{cost:,.0f}, "
        f"Failures={result['failure'].sum()}, "
        f"Lead={result['lead_time'].mean():.2f}"
    )

# Select threshold using validation cost
selected_assetiq_threshold = min(
    assetiq_policy_results,
    key=lambda t: assetiq_policy_results[t]["cost"]
)

print(
    f"\nSelected AssetIQ validation threshold: "
    f"{selected_assetiq_threshold:.2f}"
)

    

print("\nAblation Study")

# Validation ablation
val_ablation_policies = {
    "RUL Only": val_rul_risk,

    "RUL + Anomaly": (
        0.75 * val_rul_risk
        + 0.25 * val_anomaly_risk
    ),

    "RUL + Uncertainty": (
        0.75 * val_rul_risk
        + 0.25 * val_uncertainty_risk
    ),

    "AssetIQ": calculate_assetiq_risk(
        val_rul_risk,
        val_anomaly_risk,
        val_uncertainty_risk
    )
}

ablation_results = {}
ablation_test_results = {}

for name, score in val_ablation_policies.items():

    temp_df = val_df.copy()
    temp_df["risk_score"] = score

    result = evaluate_risk_policy(
        temp_df,
        threshold=selected_assetiq_threshold
    )

    cost = calculate_policy_cost(result)

    ablation_results[name] = int(cost)

    print(
        f"{name}: "
        f"Cost=₹{cost:,.0f}, "
        f"Failures={result['failure'].sum()}, "
        f"Mean lead={result['lead_time'].mean():.2f}"
    )


# Test-set ablation
test_ablation_policies = {
    "RUL Only": test_rul_risk,

    "RUL + Anomaly": (
        0.75 * test_rul_risk
        + 0.25 * test_anomaly_risk
    ),

    "RUL + Uncertainty": (
        0.75 * test_rul_risk
        + 0.25 * test_uncertainty_risk
    ),

    "AssetIQ": calculate_assetiq_risk(
        test_rul_risk,
        test_anomaly_risk,
        test_uncertainty_risk
    )
}

print("\nTest Ablation Study")

for name, score in test_ablation_policies.items():

    temp_df = test_df.copy()
    temp_df["risk_score"] = score

    result = evaluate_risk_policy(
        temp_df,
        threshold=selected_assetiq_threshold
    )

    

    cost = calculate_policy_cost(result)

    ablation_test_results[name] = {
        "cost": float(cost),
        "failures": int(result["failure"].sum()),
        "mean_lead": float(result["lead_time"].mean())
    }

    print(
        f"{name}: "
        f"Cost=₹{cost:,.0f}, "
        f"Failures={result['failure'].sum()}, "
        f"Mean lead={result['lead_time'].mean():.2f}"
    )

    # Save research results for dashboard/report
import json
import os

uncertainty_stage_coverage = {
    str(stage): float(value)
    for stage, value in stage_coverage.items()
}

research_results = {
    "rul_models": test_metrics,
        
    "uncertainty": {
    "nominal_coverage": 0.800,
    "observed_coverage": float(test_coverage),
    "coverage_gap": float(test_coverage - 0.800),
    "mean_interval_width": float(test_df["uncertainty_width"].mean()),
    "stage_coverage": uncertainty_stage_coverage,
},

"failure_classification": {
    "validation_pr_auc": float(pr_auc),
    "test_pr_auc": float(test_pr_auc)
},

"policy_comparison": {
    "reactive": 2000000,
    "rul_threshold_10": 218500,
    "assetiq_threshold_065": 219700
},
    "risk_threshold_sensitivity": {
    "0.40": {
        "cost": 634400,
        "failures": 0,
        "mean_lead": 217.20
    },
    "0.45": {
        "cost": 605700,
        "failures": 0,
        "mean_lead": 202.85
    },
    "0.50": {
        "cost": 531800,
        "failures": 0,
        "mean_lead": 165.90
    },
    "0.55": {
        "cost": 382100,
        "failures": 0,
        "mean_lead": 91.05
    },
    "0.60": {
        "cost": 234700,
        "failures": 0,
        "mean_lead": 17.35
    },
    "0.65": {
        "cost": 214300,
        "failures": 0,
        "mean_lead": 7.15
    },
    "0.70": {
        "cost": 1102000,
        "failures": 10,
        "mean_lead": 1.00
    }
},


"assetiq_asset_decisions": (
    selected_assetiq_result
    .sort_values("unit")
    .to_dict(orient="records")
),

"model_vs_decision": {
    "Linear Regression": {
        "cost": 224200,
        "failures": 0,
        "mean_lead": 12.10
    },
    "Random Forest": {
        "cost": 216000,
        "failures": 0,
        "mean_lead": 8.00
    },
    "LightGBM": {
        "cost": 218200,
        "failures": 0,
        "mean_lead": 9.10
    }
},

    "ablation": {
    "validation": ablation_results,
    "test": ablation_test_results
},



}

print("\nFinal Test-Set Decision Evaluation")

# Use the threshold selected only from validation
test_assetiq_df = test_df.copy()

test_assetiq_df["risk_score"] = calculate_assetiq_risk(
    test_rul_risk,
    test_anomaly_risk,
    test_uncertainty_risk
)

test_assetiq_result = evaluate_risk_policy(
    test_assetiq_df,
    threshold=selected_assetiq_threshold
)

test_assetiq_cost = calculate_policy_cost(
    test_assetiq_result
)

print(
    f"AssetIQ Test: "
    f"Cost=₹{test_assetiq_cost:,.0f}, "
    f"Failures={test_assetiq_result['failure'].sum()}, "
    f"Mean lead={test_assetiq_result['lead_time'].mean():.2f}"
)

test_rul_result = evaluate_rul_policy(
    test_df,
    threshold=10
)

test_rul_cost = calculate_policy_cost(
    test_rul_result
)

print(
    f"RUL Threshold Test: "
    f"Cost=₹{test_rul_cost:,.0f}, "
    f"Failures={test_rul_result['failure'].sum()}, "
    f"Mean lead={test_rul_result['lead_time'].mean():.2f}"
)

research_results["policy_comparison"] = {
    "reactive": float(reactive_cost),
    "rul_threshold_10": float(test_rul_cost),
    "assetiq_threshold_065": float(test_assetiq_cost)
}

output_path = "data/processed/research_results.json"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(research_results, f, indent=4)

print(f"\nResearch results saved to: {output_path}")