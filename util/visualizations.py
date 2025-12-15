import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Sequence, Tuple, Dict, Any


def _finalize_figure(fig: plt.Figure, save_path: Optional[str], show: bool) -> None:
    """Common helper to save/close figures."""
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_training_history(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    bad: int,
    best_val_loss: float,
    save_path: str = "softmax_training_history.png",
    show: bool = True,
) -> Dict[str, float]:
    """Plot training/validation loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, label="Training Loss", linewidth=2, alpha=0.8)
    ax.plot(epochs_range, val_losses, label="Validation Loss", linewidth=2, alpha=0.8)
    best_epoch = len(train_losses) - bad
    ax.axvline(
        x=best_epoch,
        color="red",
        linestyle="--",
        label=f"Best Model (epoch {best_epoch})",
        alpha=0.6,
    )
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss (Cross Entropy)", fontsize=12, fontweight="bold")
    ax.set_title("Softmax Regression Training History", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    _finalize_figure(fig, save_path, show)
    return {
        "best_epoch": best_epoch,
        "final_train_loss": float(train_losses[-1]),
        "best_val_loss": float(best_val_loss),
    }


def plot_random_sample_distributions(
    P_test: np.ndarray,
    P_pred: np.ndarray,
    sample_indices: Optional[Iterable[int]] = None,
    sample_size: int = 10,
    labels: Sequence[str] = ("1", "2", "3", "4", "5", "6", "7+"),
    save_path: str = "softmax_distribution_comparison.png",
    show: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """Plot random sample true vs predicted distributions."""
    rng = np.random.default_rng(seed)
    if sample_indices is None:
        sample_size = min(sample_size, len(P_test))
        sample_indices = rng.choice(len(P_test), size=sample_size, replace=False)
    sample_indices = np.array(list(sample_indices))
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    x_pos = np.arange(len(labels))
    for i, idx in enumerate(sample_indices):
        true_dist = P_test[idx]
        pred_dist = P_pred[idx]
        width = 0.35
        axes[i].bar(
            x_pos - width / 2,
            true_dist,
            width,
            label="True",
            alpha=0.8,
            color="skyblue",
        )
        axes[i].bar(
            x_pos + width / 2,
            pred_dist,
            width,
            label="Predicted",
            alpha=0.8,
            color="salmon",
        )
        sample_mae = np.mean(np.abs(true_dist - pred_dist))
        axes[i].set_xlabel("Number of Tries", fontsize=9)
        axes[i].set_ylabel("Probability", fontsize=9)
        axes[i].set_title(
            f"Sample {idx} (MAE={sample_mae:.3f})", fontsize=10, fontweight="bold"
        )
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(labels)
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3, axis="y")
        axes[i].set_ylim(0, max(true_dist.max(), pred_dist.max()) * 1.1)
    fig.suptitle(
        "Predicted vs True Distribution (Random Samples)",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )
    _finalize_figure(fig, save_path, show)
    return sample_indices


def plot_error_analysis(
    P_test: np.ndarray,
    P_pred: np.ndarray,
    dim_names: Sequence[str] = (
        "1 try",
        "2 tries",
        "3 tries",
        "4 tries",
        "5 tries",
        "6 tries",
        "7+ tries",
    ),
    save_path: str = "softmax_error_analysis_per_dimension.png",
    show: bool = True,
) -> Dict[str, Any]:
    """Visualize per-dimension errors and basic stats."""
    errors = P_pred - P_test
    abs_errors = np.abs(errors)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    mae_per_dim = abs_errors.mean(axis=0)
    axes[0, 0].bar(
        range(7), mae_per_dim, color="steelblue", alpha=0.8, edgecolor="black"
    )
    axes[0, 0].set_xlabel("Number of Tries", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Mean Absolute Error", fontsize=12, fontweight="bold")
    axes[0, 0].set_title("MAE per Dimension", fontsize=13, fontweight="bold")
    axes[0, 0].set_xticks(range(7))
    axes[0, 0].set_xticklabels(dim_names, rotation=45, ha="right")
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    axes[0, 1].boxplot(
        [errors[:, i] for i in range(7)],
        tick_labels=dim_names,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
    )
    axes[0, 1].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axes[0, 1].set_xlabel("Number of Tries", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Prediction Error", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("Error Distribution per Dimension", fontsize=13, fontweight="bold")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    for i in range(7):
        axes[1, 0].hist(
            errors[:, i],
            bins=30,
            alpha=0.5,
            label=dim_names[i],
            edgecolor="black",
            linewidth=0.5,
        )
    axes[1, 0].axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Error")
    axes[1, 0].set_xlabel("Prediction Error", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Frequency", fontsize=12, fontweight="bold")
    axes[1, 0].set_title("Error Distribution Across All Dimensions", fontsize=13, fontweight="bold")
    axes[1, 0].legend(fontsize=8, loc="upper right")
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    for i in range(7):
        axes[1, 1].scatter(
            P_test[:, i],
            P_pred[:, i],
            alpha=0.5,
            s=20,
            color=colors[i],
            label=dim_names[i],
            edgecolors="none",
        )
    min_val = min(P_test.min(), P_pred.min())
    max_val = max(P_test.max(), P_pred.max())
    axes[1, 1].plot(
        [min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction"
    )
    axes[1, 1].set_xlabel("True Probability", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Predicted Probability", fontsize=12, fontweight="bold")
    axes[1, 1].set_title("True vs Predicted (All Dimensions)", fontsize=13, fontweight="bold")
    axes[1, 1].legend(fontsize=8, loc="upper left")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect("equal", adjustable="box")
    _finalize_figure(fig, save_path, show)
    return {"errors": errors, "abs_errors": abs_errors, "mae_per_dim": mae_per_dim}


def bootstrap_predictions(
    model: torch.nn.Module,
    X: np.ndarray,
    device: str,
    n_bootstrap: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap predictions with simple input noise."""
    predictions = []
    model.eval()
    X_tensor = torch.tensor(X, device=device)
    with torch.no_grad():
        base_pred = model(X_tensor).cpu().numpy()
    for _ in range(n_bootstrap):
        noise = np.random.normal(0, 0.05, X.shape).astype(np.float32)
        X_noisy_tensor = torch.tensor(X + noise, device=device)
        with torch.no_grad():
            pred = model(X_noisy_tensor).cpu().numpy()
        predictions.append(pred)
    predictions = np.array(predictions)
    pred_mean = predictions.mean(axis=0)
    pred_std = predictions.std(axis=0)
    pred_lower = np.percentile(predictions, 5, axis=0)
    pred_upper = np.percentile(predictions, 95, axis=0)
    return pred_mean, pred_std, pred_lower, pred_upper, base_pred


def plot_uncertainty(
    model: torch.nn.Module,
    X_samples: np.ndarray,
    P_samples_true: np.ndarray,
    device: str,
    n_bootstrap: int = 100,
    labels: Sequence[str] = ("1", "2", "3", "4", "5", "6", "7+"),
    save_path: str = "softmax_uncertainty_quantification.png",
    show: bool = True,
) -> Dict[str, Any]:
    """Plot prediction uncertainty via bootstrap."""
    pred_mean, pred_std, pred_lower, pred_upper, base_pred = bootstrap_predictions(
        model, X_samples, device, n_bootstrap=n_bootstrap
    )
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    x_pos = np.arange(len(labels))
    for i in range(len(axes)):
        axes[i].scatter(
            x_pos,
            P_samples_true[i],
            color="red",
            s=100,
            label="True",
            zorder=5,
            marker="*",
            edgecolors="black",
            linewidth=1.5,
        )
        axes[i].plot(
            x_pos,
            pred_mean[i],
            color="blue",
            marker="o",
            linewidth=2,
            markersize=8,
            label="Predicted (mean)",
            zorder=4,
        )
        axes[i].fill_between(
            x_pos, pred_lower[i], pred_upper[i], alpha=0.3, color="skyblue", label="90% CI"
        )
        axes[i].errorbar(
            x_pos,
            pred_mean[i],
            yerr=pred_std[i],
            fmt="none",
            ecolor="gray",
            alpha=0.5,
            capsize=3,
            zorder=3,
        )
        axes[i].set_xlabel("Number of Tries", fontsize=10, fontweight="bold")
        axes[i].set_ylabel("Probability", fontsize=10, fontweight="bold")
        axes[i].set_title(f"Sample {i}", fontsize=11, fontweight="bold")
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(labels)
        axes[i].legend(fontsize=7, loc="upper left")
        axes[i].grid(True, alpha=0.3, axis="y")
        axes[i].set_ylim(
            0, max(P_samples_true[i].max(), pred_upper[i].max()) * 1.1
        )
    fig.suptitle(
        "Prediction Uncertainty (Bootstrap with 90% Confidence Interval)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    _finalize_figure(fig, save_path, show)
    interval_width = (pred_upper - pred_lower).mean()
    return {
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "pred_lower": pred_lower,
        "pred_upper": pred_upper,
        "base_pred": base_pred,
        "avg_std": float(pred_std.mean()),
        "max_std": float(pred_std.max()),
        "avg_ci_width": float(interval_width),
    }


def plot_performance_metrics(
    P_test: np.ndarray,
    P_pred: np.ndarray,
    save_path: str = "softmax_performance_metrics.png",
    show: bool = True,
) -> Dict[str, Any]:
    """Plot aggregated performance metrics."""
    from scipy.spatial.distance import jensenshannon
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import r2_score

    mae = np.mean(np.abs(P_pred - P_test))
    rmse = np.sqrt(mean_squared_error(P_test, P_pred))
    eps = 1e-12
    kl = np.mean(np.sum(P_test * (np.log(P_test + eps) - np.log(P_pred + eps)), axis=1))
    js_distances = [jensenshannon(P_test[i], P_pred[i]) for i in range(len(P_test))]
    js_mean = np.mean(js_distances)
    cos_sim = np.mean(
        [cosine_similarity([P_test[i]], [P_pred[i]])[0, 0] for i in range(len(P_test))]
    )
    r2 = r2_score(P_test, P_pred)
    max_error = np.max(np.abs(P_pred - P_test))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics_names = ["MAE", "RMSE", "Max Error"]
    metrics_values = [mae, rmse, max_error]
    colors_bar = ["#3498db", "#e74c3c", "#f39c12"]
    axes[0, 0].bar(metrics_names, metrics_values, color=colors_bar, alpha=0.8, edgecolor="black")
    axes[0, 0].set_ylabel("Error Value", fontsize=12, fontweight="bold")
    axes[0, 0].set_title("Error Metrics", fontsize=13, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(metrics_values):
        axes[0, 0].text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontweight="bold")
    sim_names = ["KL Divergence", "JS Distance", "1 - Cos Sim"]
    sim_values = [kl, js_mean, 1 - cos_sim]
    colors_sim = ["#9b59b6", "#1abc9c", "#e67e22"]
    axes[0, 1].bar(sim_names, sim_values, color=colors_sim, alpha=0.8, edgecolor="black")
    axes[0, 1].set_ylabel("Distance/Divergence", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("Distribution Similarity Metrics (Lower is Better)", fontsize=13, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(sim_values):
        axes[0, 1].text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontweight="bold")
    axes[0, 2].bar(["R² Score"], [r2], color="#27ae60", alpha=0.8, edgecolor="black", width=0.5)
    axes[0, 2].axhline(1.0, color="red", linestyle="--", label="Perfect Score", alpha=0.6)
    axes[0, 2].set_ylabel("R² Score", fontsize=12, fontweight="bold")
    axes[0, 2].set_title("Coefficient of Determination", fontsize=13, fontweight="bold")
    axes[0, 2].set_ylim(0, 1.1)
    axes[0, 2].grid(True, alpha=0.3, axis="y")
    axes[0, 2].text(0, r2 + 0.02, f"{r2:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    axes[0, 2].legend()
    sample_mae = np.mean(np.abs(P_pred - P_test), axis=1)
    axes[1, 0].hist(sample_mae, bins=50, color="skyblue", alpha=0.8, edgecolor="black")
    axes[1, 0].axvline(mae, color="red", linestyle="--", linewidth=2, label=f"Mean MAE = {mae:.4f}")
    axes[1, 0].set_xlabel("Sample MAE", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Frequency", fontsize=12, fontweight="bold")
    axes[1, 0].set_title("Distribution of MAE Across Samples", fontsize=13, fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    sample_kl = np.sum(P_test * (np.log(P_test + eps) - np.log(P_pred + eps)), axis=1)
    axes[1, 1].hist(sample_kl, bins=50, color="lightcoral", alpha=0.8, edgecolor="black")
    axes[1, 1].axvline(kl, color="red", linestyle="--", linewidth=2, label=f"Mean KL = {kl:.4f}")
    axes[1, 1].set_xlabel("Sample KL Divergence", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Frequency", fontsize=12, fontweight="bold")
    axes[1, 1].set_title("Distribution of KL Divergence Across Samples", fontsize=13, fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    sample_cos_sim = [cosine_similarity([P_test[i]], [P_pred[i]])[0, 0] for i in range(len(P_test))]
    axes[1, 2].hist(sample_cos_sim, bins=50, color="lightgreen", alpha=0.8, edgecolor="black")
    axes[1, 2].axvline(cos_sim, color="red", linestyle="--", linewidth=2, label=f"Mean = {cos_sim:.4f}")
    axes[1, 2].set_xlabel("Cosine Similarity", fontsize=12, fontweight="bold")
    axes[1, 2].set_ylabel("Frequency", fontsize=12, fontweight="bold")
    axes[1, 2].set_title("Distribution of Cosine Similarity", fontsize=13, fontweight="bold")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis="y")
    _finalize_figure(fig, save_path, show)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "kl": float(kl),
        "js_mean": float(js_mean),
        "cos_sim": float(cos_sim),
        "r2": float(r2),
        "max_error": float(max_error),
        "sample_mae": sample_mae,
        "sample_kl": sample_kl,
        "sample_cos_sim": np.array(sample_cos_sim),
    }


def plot_comprehensive_summary(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    bad: int,
    best_val_loss: float,
    mae: float,
    rmse: float,
    kl: float,
    js_mean: float,
    cos_sim: float,
    r2: float,
    mae_per_dim: np.ndarray,
    P_test: np.ndarray,
    P_pred: np.ndarray,
    errors: np.ndarray,
    labels: Sequence[str] = ("1", "2", "3", "4", "5", "6", "7+"),
    save_path: str = "softmax_comprehensive_summary.png",
    show: bool = True,
) -> Dict[str, Any]:
    """Produce the combined multi-panel summary figure."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    epochs_range = range(1, len(train_losses) + 1)
    ax1.plot(epochs_range, train_losses, label="Train Loss", linewidth=2, alpha=0.8)
    ax1.plot(epochs_range, val_losses, label="Val Loss", linewidth=2, alpha=0.8)
    best_epoch = len(train_losses) - bad
    ax1.axvline(best_epoch, color="red", linestyle="--", label="Best Model", alpha=0.6)
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.set_title("Training History", fontweight="bold", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2 = fig.add_subplot(gs[0, 1])
    metrics_summary = {
        "MAE": mae,
        "RMSE": rmse,
        "KL Div": kl,
        "JS Dist": js_mean,
        "R²": r2,
        "Cos Sim": cos_sim,
    }
    bars = ax2.barh(
        list(metrics_summary.keys()),
        list(metrics_summary.values()),
        color=["#3498db", "#e74c3c", "#9b59b6", "#1abc9c", "#27ae60", "#f39c12"],
        alpha=0.8,
        edgecolor="black",
    )
    ax2.set_xlabel("Value", fontweight="bold")
    ax2.set_title("Performance Metrics Summary", fontweight="bold", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="x")
    for i, (k, v) in enumerate(metrics_summary.items()):
        ax2.text(v + 0.01, i, f"{v:.3f}", va="center", fontweight="bold")
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(range(7), mae_per_dim, color="steelblue", alpha=0.8, edgecolor="black")
    ax3.set_xlabel("Tries", fontweight="bold")
    ax3.set_ylabel("MAE", fontweight="bold")
    ax3.set_title("MAE per Dimension", fontweight="bold", fontsize=12)
    ax3.set_xticks(range(7))
    ax3.set_xticklabels(labels)
    ax3.grid(True, alpha=0.3, axis="y")
    ax4 = fig.add_subplot(gs[1, 0])
    colors_scatter = plt.cm.tab10(np.linspace(0, 1, 7))
    dim_names = [
        "1 try",
        "2 tries",
        "3 tries",
        "4 tries",
        "5 tries",
        "6 tries",
        "7+ tries",
    ]
    for i in range(7):
        ax4.scatter(
            P_test[:, i],
            P_pred[:, i],
            alpha=0.4,
            s=15,
            color=colors_scatter[i],
            label=dim_names[i],
        )
    min_val = min(P_test.min(), P_pred.min())
    max_val = max(P_test.max(), P_pred.max())
    ax4.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    ax4.set_xlabel("True Probability", fontweight="bold")
    ax4.set_ylabel("Predicted Probability", fontweight="bold")
    ax4.set_title("True vs Predicted", fontweight="bold", fontsize=12)
    ax4.legend(fontsize=7, loc="upper left")
    ax4.grid(True, alpha=0.3)
    ax5 = fig.add_subplot(gs[1, 1])
    all_errors = errors.flatten()
    ax5.hist(all_errors, bins=50, color="lightcoral", alpha=0.8, edgecolor="black")
    ax5.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Error")
    ax5.set_xlabel("Prediction Error", fontweight="bold")
    ax5.set_ylabel("Frequency", fontweight="bold")
    ax5.set_title("Error Distribution (All Dimensions)", fontweight="bold", fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis="y")
    ax6 = fig.add_subplot(gs[1, 2])
    sample_mae = np.mean(np.abs(P_pred - P_test), axis=1)
    ax6.hist(sample_mae, bins=50, color="skyblue", alpha=0.8, edgecolor="black")
    ax6.axvline(mae, color="red", linestyle="--", linewidth=2, label=f"Mean = {mae:.4f}")
    ax6.set_xlabel("Sample MAE", fontweight="bold")
    ax6.set_ylabel("Frequency", fontweight="bold")
    ax6.set_title("MAE Distribution Across Samples", fontweight="bold", fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis="y")
    sample_indices_summary = np.random.choice(len(P_test), size=min(3, len(P_test)), replace=False)
    for idx, sample_idx in enumerate(sample_indices_summary):
        ax = fig.add_subplot(gs[2, idx])
        true_dist = P_test[sample_idx]
        pred_dist = P_pred[sample_idx]
        width = 0.35
        x_pos_bar = np.arange(7)
        ax.bar(x_pos_bar - width / 2, true_dist, width, label="True", alpha=0.8, color="skyblue")
        ax.bar(x_pos_bar + width / 2, pred_dist, width, label="Pred", alpha=0.8, color="salmon")
        sample_mae_val = np.mean(np.abs(true_dist - pred_dist))
        ax.set_xlabel("Tries", fontweight="bold")
        ax.set_ylabel("Probability", fontweight="bold")
        ax.set_title(
            f"Sample {sample_idx} (MAE={sample_mae_val:.3f})", fontweight="bold", fontsize=11
        )
        ax.set_xticks(x_pos_bar)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(
        "Softmax Regression Model - Comprehensive Visualization Summary",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    _finalize_figure(fig, save_path, show)
    return {
        "best_epoch": best_epoch,
        "sample_indices_summary": sample_indices_summary,
        "sample_mae": sample_mae,
    }

