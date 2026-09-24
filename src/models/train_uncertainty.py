import sys
from pathlib import Path
import joblib

# ------------------------------------------------------------
# Add project source directory to Python path
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------
from sklearn.model_selection import GroupShuffleSplit
from data_loader import load_cmapss_fd001
from preprocessing import preprocess_cmapss
from features import create_features
from models.uncertainty import train_quantile_models


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

lower_model, upper_model = train_quantile_models(
    X_train,
    y_train
)

lower = lower_model.predict(X_val)
upper = upper_model.predict(X_val)

lower = lower.clip(min=0)
upper = upper.clip(min=lower)

coverage = ((y_val >= lower) & (y_val <= upper)).mean()
width = (upper - lower).mean()

print("Validation assets:", df.iloc[val_idx]["unit"].nunique())
print("Interval coverage:", round(coverage, 4))
print("Average interval width:", round(width, 2))

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(lower_model, MODEL_DIR / "lgbm_rul_lower.pkl")
joblib.dump(upper_model, MODEL_DIR / "lgbm_rul_upper.pkl")

print("Uncertainty models saved.")