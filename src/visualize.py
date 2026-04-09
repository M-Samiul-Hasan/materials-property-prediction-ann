import os
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_output_dir():
    os.makedirs("outputs/figures", exist_ok=True)


def plot_correlation_heatmap(df):
    """Plot and save correlation heatmap."""
    ensure_output_dir()
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("outputs/figures/correlation_heatmap.png", dpi=300)
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred, title, filename):
    """Plot actual vs predicted values."""
    ensure_output_dir()
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{filename}", dpi=300)
    plt.show()


def plot_residuals(y_true, y_pred, title, filename):
    """Plot residuals against predicted values."""
    ensure_output_dir()
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{filename}", dpi=300)
    plt.show()


def plot_error_distribution(y_true, y_pred, title, filename):
    """Plot histogram of prediction errors."""
    ensure_output_dir()
    errors = y_true - y_pred
    plt.figure(figsize=(7, 5))
    plt.hist(errors, bins=20)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{filename}", dpi=300)
    plt.show()