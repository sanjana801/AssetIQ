from sklearn.ensemble import RandomForestRegressor


def train_random_forest_rul(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model