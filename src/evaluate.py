import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test_scaled, scaler, name="Target"):
    y_pred_scaled = model.predict(X_test)

    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"{name} -> RMSE: {rmse:.4f}, R2: {r2:.4f}")

    return y_true, y_pred, rmse, r2


def show_values_all(
    yE_true, yE_pred,
    ySy_true, ySy_pred,
    yUTS_true, yUTS_pred,
    n=10,
    random_state=None
):
    results = pd.DataFrame()

    results["True E (GPa)"] = np.asarray(yE_true)
    results["Pred E (GPa)"] = np.asarray(yE_pred)
    results["True Sy (MPa)"] = np.asarray(ySy_true)
    results["Pred Sy (MPa)"] = np.asarray(ySy_pred)
    results["True UTS (MPa)"] = np.asarray(yUTS_true)
    results["Pred UTS (MPa)"] = np.asarray(yUTS_pred)

    results["Abs Error E"] = np.abs(results["True E (GPa)"] - results["Pred E (GPa)"])
    results["Abs Error Sy"] = np.abs(results["True Sy (MPa)"] - results["Pred Sy (MPa)"])
    results["Abs Error UTS"] = np.abs(results["True UTS (MPa)"] - results["Pred UTS (MPa)"])

    n = min(n, len(results))
    return results.sample(n=n, random_state=random_state).reset_index(drop=True)