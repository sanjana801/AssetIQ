import pandas as pd


MAINTENANCE_COST = 10000
FAILURE_COST = 100000
PERSISTENCE = 3


def find_first_persistent_trigger(
    asset,
    condition,
    persistence=3,
):
    values = condition.astype(int)

    consecutive = (
        values.groupby(
            (values != values.shift()).cumsum()
        )
        .transform("size")
    )

    trigger = asset[
        (condition) &
        (consecutive >= persistence)
    ]

    if len(trigger) == 0:
        return None

    return int(trigger.iloc[0]["cycle"])


def evaluate_policy(
    df,
    signal_column,
):
    results = []

    for unit, asset in df.groupby("unit"):
        asset = asset.sort_values("cycle").copy()

        signal = asset[signal_column].astype(int)

        consecutive = (
            signal.groupby(
                (signal != signal.shift()).cumsum()
            )
            .transform("size")
        )

        trigger = asset[
            (signal == 1) &
            (consecutive >= PERSISTENCE)
        ]

        if len(trigger) > 0:
            maintenance_cycle = int(trigger.iloc[0]["cycle"])
            failure = 0
            lead_time = int(
                asset.iloc[-1]["cycle"] - maintenance_cycle
            )
        else:
            maintenance_cycle = None
            failure = 1
            lead_time = 0

        maintenance_cost = (
            MAINTENANCE_COST
            if maintenance_cycle is not None
            else 0
        )

        failure_cost = failure * FAILURE_COST

        results.append({
            "unit": unit,
            "maintenance_cycle": maintenance_cycle,
            "failure": failure,
            "lead_time": lead_time,
            "maintenance_cost": maintenance_cost,
            "failure_cost": failure_cost,
            "total_cost": maintenance_cost + failure_cost,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    df = pd.read_csv(
        "data/processed/assetiq_predictions.csv"
    )

    df["rul_signal"] = (
        df["predicted_RUL"] <= 10
    )

    df["assetiq_signal"] = (
        df["risk_score"] >= 0.65
    )

    rul = evaluate_policy(
        df,
        "rul_signal"
    )

    assetiq = evaluate_policy(
        df,
        "assetiq_signal"
    )

    for name, result in [
        ("RUL Threshold", rul),
        ("AssetIQ", assetiq),
    ]:
        print(f"\n{name}")
        print("Maintenance events:",
              result["maintenance_cycle"].notna().sum())
        print("Failures:",
              result["failure_cost"].gt(0).sum())
        print("Mean lead time:",
              round(result["lead_time"].mean(), 2))
        print("Total cost:",
              result["total_cost"].sum())