import lightgbm as lgb


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