import os
import pandas as pd

from src.data_loader import load_data, split_features_targets
from src.preprocessing import scale_inputs, split_data, scale_targets
from src.train import train_ann_model
from src.evaluate import evaluate_model, show_values_all
from src.visualize import (
    plot_correlation_heatmap,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_error_distribution,
)


os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)


def main():
    # Load data
    df = load_data("data/data_csv.csv")

    # Basic visualization
    plot_correlation_heatmap(df)

    # Split data
    X, y_E, y_Sy, y_UTS = split_features_targets(df)

    # Scale inputs
    X_scaled, _ = scale_inputs(X)

    # Train-test split
    X_train, X_test, yE_train, yE_test, ySy_train, ySy_test, yUTS_train, yUTS_test = split_data(
        X_scaled, y_E, y_Sy, y_UTS
    )

    # Scale targets
    yE_train_scaled, yE_test_scaled, scaler_E = scale_targets(yE_train, yE_test)
    ySy_train_scaled, ySy_test_scaled, scaler_Sy = scale_targets(ySy_train, ySy_test)
    yUTS_train_scaled, yUTS_test_scaled, scaler_UTS = scale_targets(yUTS_train, yUTS_test)

    # Train models
    model_E = train_ann_model(X_train, yE_train_scaled)
    model_Sy = train_ann_model(X_train, ySy_train_scaled)
    model_UTS = train_ann_model(X_train, yUTS_train_scaled)

    # Evaluate models
    true_E, pred_E, rmse_E, r2_E = evaluate_model(
        model_E, X_test, yE_test_scaled, scaler_E, "Young's Modulus"
    )
    true_Sy, pred_Sy, rmse_Sy, r2_Sy = evaluate_model(
        model_Sy, X_test, ySy_test_scaled, scaler_Sy, "Yield Strength"
    )
    true_UTS, pred_UTS, rmse_UTS, r2_UTS = evaluate_model(
        model_UTS, X_test, yUTS_test_scaled, scaler_UTS, "UTS"
    )

    # Plots
    plot_actual_vs_predicted(
        true_E, pred_E,
        "Young's Modulus: Actual vs Predicted",
        "youngs_modulus_actual_vs_predicted.png"
    )
    plot_actual_vs_predicted(
        true_Sy, pred_Sy,
        "Yield Strength: Actual vs Predicted",
        "yield_strength_actual_vs_predicted.png"
    )
    plot_actual_vs_predicted(
        true_UTS, pred_UTS,
        "UTS: Actual vs Predicted",
        "uts_actual_vs_predicted.png"
    )

    plot_residuals(
        true_E, pred_E,
        "Young's Modulus Residual Plot",
        "youngs_modulus_residuals.png"
    )
    plot_residuals(
        true_Sy, pred_Sy,
        "Yield Strength Residual Plot",
        "yield_strength_residuals.png"
    )
    plot_residuals(
        true_UTS, pred_UTS,
        "UTS Residual Plot",
        "uts_residuals.png"
    )

    plot_error_distribution(
        true_E, pred_E,
        "Young's Modulus Error Distribution",
        "youngs_modulus_error_distribution.png"
    )
    plot_error_distribution(
        true_Sy, pred_Sy,
        "Yield Strength Error Distribution",
        "yield_strength_error_distribution.png"
    )
    plot_error_distribution(
        true_UTS, pred_UTS,
        "UTS Error Distribution",
        "uts_error_distribution.png"
    )

    # Sample predictions CSV
    sample_rows = show_values_all(
        yE_true=true_E, yE_pred=pred_E,
        ySy_true=true_Sy, ySy_pred=pred_Sy,
        yUTS_true=true_UTS, yUTS_pred=pred_UTS,
        n=10,
        random_state=42
    )

    print("\nSample Predictions:")
    print(sample_rows)
    sample_rows.to_csv("outputs/sample_predictions.csv", index=False)

    # Metrics summary CSV
    metrics_df = pd.DataFrame({
        "Property": ["Young's Modulus", "Yield Strength", "UTS"],
        "RMSE": [rmse_E, rmse_Sy, rmse_UTS],
        "R2 Score": [r2_E, r2_Sy, r2_UTS]
    })

    print("\nModel Performance Summary:")
    print(metrics_df)
    metrics_df.to_csv("outputs/model_metrics.csv", index=False)


if __name__ == "__main__":
    main()