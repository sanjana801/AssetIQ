from pathlib import Path
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_cmapss_fd001
from src.preprocessing import preprocess_cmapss
from src.features import create_features
from src.models.rul_model import train_rul_model


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

model = train_rul_model(X_train, y_train)

predictions = model.predict(X_val)

mae = mean_absolute_error(y_val, predictions)
rmse = mean_squared_error(y_val, predictions) ** 0.5
r2 = r2_score(y_val, predictions)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Validation assets:", df.iloc[val_idx]["unit"].nunique())
print("RUL MAE:", round(mae, 3))
print("RUL RMSE:", round(rmse, 3))
print("RUL R2:", round(r2, 3))

MODEL_PATH = Path("models/lgbm_rul.pkl")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("Model saved to:", MODEL_PATH)