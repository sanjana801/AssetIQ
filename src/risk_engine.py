def calculate_risk_score(
    rul_risk,
    anomaly_risk,
    uncertainty_risk
):
    return (
        0.50 * rul_risk
        + 0.25 * anomaly_risk
        + 0.25 * uncertainty_risk
    )


def assign_priority(risk_score):
    if risk_score >= 0.75:
        return "Critical"
    elif risk_score >= 0.50:
        return "High"
    elif risk_score >= 0.25:
        return "Medium"
    else:
        return "Low"