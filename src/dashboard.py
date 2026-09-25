import json
import streamlit as st
import pandas as pd
from src.models.shap_analysis import get_asset_explanation

st.set_page_config(
    page_title="AssetIQ",
    page_icon="🏭",
    layout="wide"
)

with open("data/processed/research_results.json", "r") as f:
    results = json.load(f)

predictions = pd.read_csv(
    "data/processed/assetiq_predictions.csv"
)

st.title("AssetIQ")
st.subheader("Explainable & Uncertainty-Aware Maintenance Decision Intelligence")

st.success("Research results loaded successfully.")

st.header("RUL Model Performance")

model_results = results["rul_models"]

col1, col2, col3 = st.columns(3)

for col, (model_name, metrics) in zip(
    [col1, col2, col3],
    model_results.items()
):
    with col:
        st.subheader(model_name)
        st.metric("MAE", f"{metrics['MAE']:.3f}")
        st.metric("RMSE", f"{metrics['RMSE']:.3f}")
        st.metric("R²", f"{metrics['R2']:.3f}")

st.header("Prediction Accuracy vs Decision Cost")

model_decision_data = results["model_vs_decision"]

model_names = [
    "Linear Regression",
    "Random Forest",
    "LightGBM"
]

model_decision = pd.DataFrame([
    {
        "Model": name,
        "RUL MAE": model_results[name]["MAE"],
        "Simulated Cost": model_decision_data[name]["cost"],
        "Failures": model_decision_data[name]["failures"],
        "Mean Lead Time": model_decision_data[name]["mean_lead"]
    }
    for name in model_names
])

st.dataframe(
    model_decision.round(2),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "RUL metrics and decision costs are evaluated on the held-out test set. "
    "Costs use illustrative maintenance-cost assumptions."
)

st.subheader("Prediction Accuracy vs Decision Cost")

st.scatter_chart(
    model_decision,
    x="RUL MAE",
    y="Simulated Cost"
)



st.header("Uncertainty Evaluation")

uncertainty = results["uncertainty"]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Nominal Coverage", f"{uncertainty['nominal_coverage'] * 100:.1f}%")

with col2:
    st.metric("Observed Coverage", f"{uncertainty['observed_coverage'] * 100:.1f}%")

with col3:
    st.metric("Coverage Gap", f"{uncertainty['coverage_gap'] * 100:.1f} pp")

with col4:
    st.metric(
        "Mean Interval Width",
        f"{uncertainty['mean_interval_width']:.2f}"
    )

st.subheader("Uncertainty Coverage by Life Stage")

stage_coverage = results["uncertainty"]["stage_coverage"]

coverage_df = pd.DataFrame({
    "Life Stage": list(stage_coverage.keys()),
    "Observed Coverage": [
        value * 100 for value in stage_coverage.values()
    ]
})

st.dataframe(
    coverage_df.round(1),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Nominal prediction-interval coverage is 80%. "
    "Observed coverage varies across asset life stages."
)

st.bar_chart(
    coverage_df.set_index("Life Stage")
)

st.caption(
    "The nominal interval coverage is 80%. "
    "Observed coverage is lower in the early and late life stages, "
    "indicating that uncertainty coverage is not uniform throughout "
    "the asset lifecycle."
)

st.header("Failure Classification")

failure = results["failure_classification"]

col1, col2 = st.columns(2)
with col1:
    st.metric(
        "Validation PR-AUC",
        f"{failure['validation_pr_auc']:.3f}"
    )

with col2:
    st.metric(
        "Test PR-AUC",
        f"{failure['test_pr_auc']:.3f}"
    )

st.header("Maintenance Policy Comparison")

policy = results["policy_comparison"]


with col1:
    st.metric(
        "Reactive Maintenance",
        f"₹{policy['reactive']:,.0f}"
    )

with col2:
    st.metric(
        "RUL Threshold (10)",
        f"₹{policy['rul_threshold_10']:,.0f}"
    )

with col3:
    st.metric(
        "AssetIQ Risk (0.65)",
        f"₹{policy['assetiq_threshold_065']:,.0f}"
    )

st.header("Ablation Study")

ablation = results["ablation"]

st.subheader("Validation")

validation_ablation = ablation["validation"]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("RUL Only", f"₹{validation_ablation['RUL Only']:,.0f}")

with col2:
    st.metric("RUL + Anomaly", f"₹{validation_ablation['RUL + Anomaly']:,.0f}")

with col3:
    st.metric(
        "RUL + Uncertainty",
        f"₹{validation_ablation['RUL + Uncertainty']:,.0f}"
    )

with col4:
    st.metric("AssetIQ", f"₹{validation_ablation['AssetIQ']:,.0f}")


st.subheader("Test")

test_ablation = ablation["test"]

ablation_table = pd.DataFrame(test_ablation).T.reset_index()
ablation_table.columns = [
    "Policy",
    "Cost",
    "Failures",
    "Mean Lead Time"
]

st.dataframe(
    ablation_table,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Maintenance costs are simulated using illustrative preventive, failure, "
    "and early-maintenance penalty assumptions. The selected AssetIQ threshold "
    "is determined using validation data; test results are reported separately."
)

st.header("Fleet Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Assets",
        predictions["unit"].nunique()
    )

with col2:
    st.metric(
        "Total Records",
        f"{len(predictions):,}"
    )

with col3:
    anomaly_rate = (
        predictions["anomaly_label"] == -1
    ).mean() * 100

    st.metric(
        "Anomaly Rate",
        f"{anomaly_rate:.2f}%"
    )

with col4:
    st.metric(
        "Average Predicted RUL",
        f"{predictions['predicted_RUL'].mean():.1f}"
    )

st.header("Asset Detail")

asset_ids = sorted(predictions["unit"].unique())

selected_asset = st.selectbox(
    "Select Asset",
    asset_ids
)

asset_data = predictions[
    predictions["unit"] == selected_asset
]

latest = asset_data.iloc[-1]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current Cycle",
        int(latest["cycle"])
    )

with col2:
    st.metric(
        "Predicted RUL",
        f"{latest['predicted_RUL']:.1f}"
    )

with col3:
    st.metric(
        "RUL Interval",
        f"{latest['RUL_lower']:.1f} – {latest['RUL_upper']:.1f}"
    )

with col4:
    if latest["predicted_RUL"] <= 10:
        asset_status = "Critical"
    elif latest["anomaly_label"] == -1:
        asset_status = "Anomalous"
    else:
        asset_status = "Normal"

    st.metric(
    "Asset Status",
    asset_status
)

if "risk_score" in latest:
    st.metric(
        "AssetIQ Risk Score",
        f"{latest['risk_score']:.3f}"
    )

risk_score = latest["risk_score"]
predicted_rul = latest["predicted_RUL"]

if risk_score >= 0.70:
    priority = "Critical"
    recommendation = "Immediate maintenance inspection"

elif risk_score >= 0.65:
    priority = "High"
    recommendation = "Schedule maintenance soon"

elif risk_score >= 0.50:
    priority = "Medium"
    recommendation = "Monitor asset closely"

else:
    priority = "Low"
    recommendation = "Continue routine monitoring"

st.metric(
    "Maintenance Priority",
    priority
)

st.info(
    f"Recommended Action: {recommendation}"
)

st.header("RUL Trend")

asset_trend = (
    predictions[predictions["unit"] == selected_asset]
    .sort_values("cycle")
)

st.line_chart(
    asset_trend.set_index("cycle")[
        ["RUL", "predicted_RUL"]
    ]
)

st.header("Why is this asset at risk?")

explanation = get_asset_explanation(selected_asset).head(10).copy()

explanation["Direction"] = explanation["shap_value"].apply(
    lambda x: "Reduces predicted RUL"
    if x < 0
    else "Increases predicted RUL"
)

explanation["Contribution"] = explanation["shap_value"].abs().round(2)

st.dataframe(
    explanation[
        ["feature", "Direction", "Contribution"]
    ],
    use_container_width=True,
    hide_index=True
)

st.header("Top 10 Highest-Risk Assets")

latest_assets = (
    predictions
    .sort_values(["unit", "cycle"])
    .groupby("unit")
    .tail(1)
)

top_risk = (
    latest_assets
    .nlargest(10, "risk_score")
    [
        [
            "unit",
            "cycle",
            "predicted_RUL",
            "uncertainty_width",
            "risk_score"
        ]
    ]
    .copy()
)

top_risk.columns = [
    "Asset",
    "Current Cycle",
    "Predicted RUL",
    "Uncertainty",
    "Risk Score"
]

st.dataframe(
    top_risk.round(2),
    use_container_width=True,
    hide_index=True
)

st.header("Maintenance Decision Comparison")

policy = results["policy_comparison"]

comparison = pd.DataFrame({
    "Policy": [
        "Reactive Maintenance",
        "RUL Threshold",
        "AssetIQ Risk-Based"
    ],
    "Simulated Cost": [
        policy["reactive"],
        policy["rul_threshold_10"],
        policy["assetiq_threshold_065"]
    ]
})

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Costs are simulated using illustrative maintenance-cost assumptions "
    "and should not be interpreted as measured industrial savings."
)

st.header("Decision Simulator")

sensitivity = results["risk_threshold_sensitivity"]

thresholds = list(sensitivity.keys())

selected_threshold = st.select_slider(
    "Select AssetIQ Risk Threshold",
    options=thresholds,
    value="0.65"
)

selected = sensitivity[selected_threshold]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Simulated Cost", f"₹{selected['cost']:,}")

with col2:
    st.metric("Maintenance Count", 20 - selected["failures"])

with col3:
    st.metric("Failures", selected["failures"])

with col4:
    st.metric("Mean Lead Time", f"{selected['mean_lead']:.2f} cycles")

st.info(
    f"At a risk threshold of {selected_threshold}, "
    f"maintenance is triggered when the AssetIQ risk score reaches "
    f"this level. In the current validation simulation, "
    f"{20 - selected['failures']} of 20 assets are maintained before "
    f"recorded failure, with an average lead time of "
    f"{selected['mean_lead']:.2f} cycles."
)

st.subheader("Asset-Level Maintenance Decisions")

asset_decisions = pd.DataFrame(
    results["assetiq_asset_decisions"]
)

if not asset_decisions.empty:
    asset_decisions.columns = [
        "Asset",
        "Maintenance Cycle",
        "Lead Time",
        "Failure"
    ]

    st.dataframe(
        asset_decisions,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No asset-level maintenance decisions available.")

st.subheader("Maintenance Lead Time by Asset")

if not asset_decisions.empty:
    lead_time_chart = asset_decisions.set_index("Asset")[["Lead Time"]]
    st.bar_chart(lead_time_chart)
else:
    st.info("No maintenance lead-time data available.")


st.caption(
    "Simulation uses illustrative maintenance-cost assumptions "
    "and validation-set results. Threshold performance is dataset-specific."
)