from sklearn.neural_network import MLPRegressor


def train_ann_model(X_train, y_train):
    model = MLPRegressor(
        hidden_layer_sizes=(20, 20),
        activation="tanh",
        solver="adam",
        max_iter=5000,
        learning_rate_init=0.001,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model