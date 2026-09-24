import lightgbm as lgb


# ------------------------------------------------------------
# Train quantile models for RUL uncertainty
# ------------------------------------------------------------

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

    lower_model.fit(
        X_train,
        y_train
    )

    upper_model.fit(
        X_train,
        y_train
    )

    return lower_model, upper_model


# ------------------------------------------------------------
# Construct interval around the point prediction
# ------------------------------------------------------------

def create_prediction_interval(
    point_prediction,
    lower_prediction,
    upper_prediction
):
    lower = min(
        lower_prediction,
        point_prediction
    )

    upper = max(
        upper_prediction,
        point_prediction
    )

    return lower, upper