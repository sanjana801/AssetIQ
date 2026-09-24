import lightgbm as lgb


def train_rul_model(X_train, y_train):
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    return model