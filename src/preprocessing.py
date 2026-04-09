import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def scale_inputs(X):
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = x_scaler.fit_transform(X)
    return X_scaled, x_scaler


def split_data(X_scaled, y_E, y_Sy, y_UTS, test_size=0.3, random_state=42):
    return train_test_split(
        X_scaled, y_E, y_Sy, y_UTS,
        test_size=test_size,
        random_state=random_state
    )


def scale_targets(y_train, y_test):
    scaler = StandardScaler()

    y_train_scaled = scaler.fit_transform(np.array(y_train).reshape(-1, 1)).ravel()
    y_test_scaled = scaler.transform(np.array(y_test).reshape(-1, 1)).ravel()

    return y_train_scaled, y_test_scaled, scaler