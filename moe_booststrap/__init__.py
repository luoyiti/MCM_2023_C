"""
MoE Bootstrap 模块

Sparse-gated Mixture of Experts 模型，包含 Bootstrap 不确定性估计。
用于 Wordle 尝试次数分布预测。
"""

from .config import (
    DEVICE,
    SEED,
    DATA_PATH,
    DIST_COLS,
    FEATURE_COLS,
    OUTPUT_DIR,
    # 超参数
    NUM_EXPERTS,
    TOP_K,
    HIDDEN_SIZE,
    DROPOUT,
    AUX_LOSS_WEIGHT,
    LR,
    WD,
    MAX_EPOCHS,
    PATIENCE,
    # Bootstrap 参数
    BOOTSTRAP_B,
    BOOTSTRAP_CI,
    BOOTSTRAP_EPOCH_SCALE,
    # 网格搜索参数
    GRID_NUM_EXPERTS,
    GRID_TOP_K,
    GRID_HIDDEN_SIZE,
    GRID_EPOCHS,
)

from .data import (
    load_and_split_data,
    make_weights_from_N,
)

from .moe import (
    SparseDispatcher,
    MLP,
    MoE,
)

from .losses import (
    soft_cross_entropy,
    weighted_soft_cross_entropy,
    expert_diversity_penalty,
)

from .metrics import (
    compute_metrics,
    compute_per_bin_metrics,
    compute_per_sample_metrics,
)

from .train import (
    set_seed,
    expert_output_separation_js,
    train_moe,
    evaluate,
)

from .search import (
    specialization_search,
    expert_topk_grid_search,
)

from .bootstrap import (
    bootstrap_predict,
    bootstrap_summary,
    compute_confidence_scores,
    bootstrap_evaluate,
)

from .plots import (
    plot_training_history,
    plot_random_sample_distributions,
    plot_error_analysis,
    analyze_expert_usage,
    compute_expert_outputs,
    plot_expert_gate_heatmap,
    plot_sample_expert_decomposition,
    plot_holdout_bar_with_ci,
    plot_holdout_violin,
    plot_uncertainty,
    plot_grid_search_results,
    plot_expert_specialization_analysis,
    plot_expert_parallel_coordinates,
)

from .utils import (
    save_predictions,
    save_bootstrap_predictions,
    save_holdout_predictions_with_ci,
    save_uncertainty_arrays,
    load_uncertainty_arrays,
    write_bootstrap_report,
    generate_summary_report,
    format_metrics_table,
)


__all__ = [
    # Config
    "DEVICE",
    "SEED",
    "DATA_PATH",
    "DIST_COLS",
    "FEATURE_COLS",
    "OUTPUT_DIR",
    "NUM_EXPERTS",
    "TOP_K",
    "HIDDEN_SIZE",
    "DROPOUT",
    "AUX_LOSS_WEIGHT",
    "LR",
    "WD",
    "MAX_EPOCHS",
    "PATIENCE",
    "BOOTSTRAP_B",
    "BOOTSTRAP_CI",
    "BOOTSTRAP_EPOCH_SCALE",
    "GRID_NUM_EXPERTS",
    "GRID_TOP_K",
    "GRID_HIDDEN_SIZE",
    "GRID_EPOCHS",
    # Data
    "load_and_split_data",
    "make_weights_from_N",
    # MoE
    "SparseDispatcher",
    "MLP",
    "MoE",
    # Losses
    "soft_cross_entropy",
    "weighted_soft_cross_entropy",
    "expert_diversity_penalty",
    # Metrics
    "compute_metrics",
    "compute_per_bin_metrics",
    "compute_per_sample_metrics",
    # Train
    "set_seed",
    "expert_output_separation_js",
    "train_moe",
    "evaluate",
    # Search
    "specialization_search",
    "expert_topk_grid_search",
    # Bootstrap
    "bootstrap_predict",
    "bootstrap_summary",
    "compute_confidence_scores",
    "bootstrap_evaluate",
    # Plots
    "plot_training_history",
    "plot_random_sample_distributions",
    "plot_error_analysis",
    "analyze_expert_usage",
    "compute_expert_outputs",
    "plot_expert_gate_heatmap",
    "plot_sample_expert_decomposition",
    "plot_holdout_bar_with_ci",
    "plot_holdout_violin",
    "plot_uncertainty",
    "plot_grid_search_results",
    "plot_expert_specialization_analysis",
    "plot_expert_parallel_coordinates",
    # Utils
    "save_predictions",
    "save_bootstrap_predictions",
    "save_holdout_predictions_with_ci",
    "save_uncertainty_arrays",
    "load_uncertainty_arrays",
    "write_bootstrap_report",
    "generate_summary_report",
    "format_metrics_table",
]


__version__ = "1.0.0"
