def create_features(df):
    df = df.copy()

    sensor_cols = [
        col for col in df.columns
        if col.startswith("sensor_")
    ]

    for sensor in sensor_cols:
        df[f"{sensor}_rolling_mean"] = (
            df.groupby("unit")[sensor]
            .transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )
        )

    return df