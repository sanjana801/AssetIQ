from sklearn.ensemble import IsolationForest


def train_anomaly_model(train_df):
    sensor_features = [
        col for col in train_df.columns
        if col.startswith("sensor_")
        and "rolling_mean" not in col
    ]

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )

    model.fit(train_df[sensor_features])

    return model, sensor_features


def predict_anomalies(model, df, sensor_features):
    df = df.copy()

    df["anomaly_label"] = model.predict(
        df[sensor_features]
    )

    df["anomaly_score"] = -model.decision_function(
        df[sensor_features]
    )

    return df