import lightgbm as lgb
import numpy as np


def train_quantile_models(X_train, y_train):

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

    return lower_model, upper_model


def calibrate_prediction_interval(
    y_true,
    lower_prediction,
    upper_prediction,
    target_coverage=0.80
):
    """
    Conformal-style symmetric calibration of an existing interval.

    The calibration factor is learned only from validation data.
    """

    lower_prediction = np.asarray(lower_prediction)
    upper_prediction = np.asarray(upper_prediction)
    y_true = np.asarray(y_true)

    lower_prediction = np.minimum(lower_prediction, upper_prediction)

    upper_prediction = np.maximum(
        upper_prediction,
        lower_prediction
    )

    interval_radius = (
        upper_prediction - lower_prediction
    ) / 2

    midpoint = (
        upper_prediction + lower_prediction
    ) / 2

    nonconformity = np.maximum(
        np.abs(y_true - midpoint) - interval_radius,
        0
    )

    quantile = np.quantile(
        nonconformity,
        target_coverage
    )

    return float(quantile)


def create_calibrated_prediction_interval(
    point_prediction,
    lower_prediction,
    upper_prediction,
    calibration_factor
):

    lower = np.minimum(
        lower_prediction - calibration_factor,
        point_prediction
    )

    upper = np.maximum(
        upper_prediction + calibration_factor,
        point_prediction
    )

    lower = max(0.0, float(lower))
    upper = float(upper)

    return lower, upper