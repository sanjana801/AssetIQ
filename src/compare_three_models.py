from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data_loader import load_cmapss_fd001
from preprocessing import preprocess_cmapss
from features import create_features
from models.rul_model import train_rul_model
from models.random_forest_rul import train_random_forest_rul


# Load data
df = load_cmapss_fd001()
df = preprocess_cmapss(df)
df = create_features(df)

feature_cols = [
    col for col in df.columns
    if col.startswith("sensor_")
]

X = df[feature_cols]
y = df["RUL"]
groups = df["unit"]


# Same 80/20 asset split
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
y_val = y.iloc[val_idx]


# Train models
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

rf_model = train_random_forest_rul(
    X_train,
    y_train
)

lgbm_model = train_rul_model(
    X_train,
    y_train
)


# Predictions
predictions = {
    "Linear Regression": linear_model.predict(X_val),
    "Random Forest": rf_model.predict(X_val),
    "LightGBM": lgbm_model.predict(X_val)
}


# Results
print()
print("Model | MAE | RMSE | R2")
print("-" * 50)

for name, pred in predictions.items():

    mae = mean_absolute_error(
        y_val,
        pred
    )

    rmse = mean_squared_error(
        y_val,
        pred
    ) ** 0.5

    r2 = r2_score(
        y_val,
        pred
    )

    print(
        f"{name} | "
        f"{mae:.3f} | "
        f"{rmse:.3f} | "
        f"{r2:.3f}"
    )