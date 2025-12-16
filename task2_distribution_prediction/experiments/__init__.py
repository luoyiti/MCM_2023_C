"""
Wordle 难度预测模型库
包含回归模型和分布预测模型
"""

from .forecasting_models import (
    load_data,
    train_lasso,
    train_ridge,
    train_elasticnet,
    train_mlp,
    train_randomforest,
    calculate_metrics,
    DEFAULT_FEATURE_COLS
)

from .distribution_models import (
    load_distribution_data,
    train_and_evaluate_distribution_model,
    compare_distribution_models,
    generate_distribution_report,
    LinearSoftmax,
    MLPSoftmax,
    DIST_FEATURE_COLS,
    DIST_TARGET_COLS
)

__all__ = [
    # 回归模型
    'load_data',
    'train_lasso',
    'train_ridge',
    'train_elasticnet',
    'train_mlp',
    'train_randomforest',
    'calculate_metrics',
    'DEFAULT_FEATURE_COLS',
    
    # 分布预测模型
    'load_distribution_data',
    'train_and_evaluate_distribution_model',
    'compare_distribution_models',
    'generate_distribution_report',
    'LinearSoftmax',
    'MLPSoftmax',
    'DIST_FEATURE_COLS',
    'DIST_TARGET_COLS',
]

__version__ = '1.0.0'
