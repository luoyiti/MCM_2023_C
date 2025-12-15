"""
配置文件 - 统一管理所有路径和参数
"""

import os

# ============================================================================
# 路径配置
# ============================================================================

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DATA = os.path.join(DATA_DIR, "reduced_features_train.csv")
TEST_DATA = os.path.join(DATA_DIR, "reduced_features_test.csv")
PROCESSED_DATA = os.path.join(DATA_DIR, "mcm_processed_data.csv")

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
RESULTS_BASE = PROJECT_ROOT

# 各模型结果目录
LASSO_RESULTS = os.path.join(RESULTS_BASE, "lasso_results")
RIDGE_RESULTS = os.path.join(RESULTS_BASE, "ridge_results")
ELASTICNET_RESULTS = os.path.join(RESULTS_BASE, "elasticNet_results")
MLP_RESULTS = os.path.join(RESULTS_BASE, "mlp_results")
RANDOMFOREST_RESULTS = os.path.join(RESULTS_BASE, "randomforest_results")
TABNET_RESULTS = os.path.join(RESULTS_BASE, "output")
LINEAR_SOFTMAX_RESULTS = os.path.join(RESULTS_BASE, "linear_softmax_results")
MLP_SOFTMAX_RESULTS = os.path.join(RESULTS_BASE, "mlp_softmax_results")
DISTRIBUTION_RESULTS = os.path.join(RESULTS_BASE, "distribution_results")

# ============================================================================
# 模型参数配置
# ============================================================================

# 通用参数
RANDOM_STATE = 42
CV_FOLDS = 5

# 特征列配置
DEFAULT_FEATURE_COLS = [
    "字母频率特征_weighted_reduced",
    "位置特征_PLS_reduced",
    "仿真模拟特征_weighted_reduced",
    "强化学习特征_weighted_reduced",
    "Zipf-value",
    "feedback_entropy",
    "letter_entropy",
    "max_consecutive_vowels",
    "semantic_distance",
]

# 分布预测特征列
DIST_FEATURE_COLS = [
    'Zipf-value',
    'letter_entropy',
    'feedback_entropy',
    'max_consecutive_vowels',
    'letter_freq_mean',
    'scrabble_score',
    'has_common_suffix',
    'num_rare_letters',
    'position_rarity',
    'positional_freq_min',
    'hamming_neighbors',
    'keyboard_distance',
    'semantic_distance',
    '1_try_simulate_random', '2_try_simulate_random', '3_try_simulate_random',
    '4_try_simulate_random', '5_try_simulate_random', '6_try_simulate_random', '7_try_simulate_random',
    '1_try_simulate_freq', '2_try_simulate_freq', '3_try_simulate_freq',
    '4_try_simulate_freq', '5_try_simulate_freq', '6_try_simulate_freq', '7_try_simulate_freq',
    '1_try_simulate_entropy', '2_try_simulate_entropy', '3_try_simulate_entropy',
    '4_try_simulate_entropy', '5_try_simulate_entropy', '6_try_simulate_entropy', '7_try_simulate_entropy',
    'rl_1_try_low_training', 'rl_2_try_low_training', 'rl_3_try_low_training',
    'rl_4_try_low_training', 'rl_5_try_low_training', 'rl_6_try_low_training', 'rl_7_try_low_training',
    'rl_1_try_high_training', 'rl_2_try_high_training', 'rl_3_try_high_training',
    'rl_4_try_high_training', 'rl_5_try_high_training', 'rl_6_try_high_training', 'rl_7_try_high_training',
    'rl_1_try_little_training', 'rl_2_try_little_training', 'rl_3_try_little_training',
    'rl_4_try_little_training', 'rl_5_try_little_training', 'rl_6_try_little_training', 'rl_7_try_little_training',
]

# 目标列配置
DEFAULT_TARGET_COL = "autoencoder_value"
DIST_TARGET_COLS = [
    "1_try", "2_tries", "3_tries", "4_tries",
    "5_tries", "6_tries", "7_or_more_tries_x"
]

# ============================================================================
# 绘图配置
# ============================================================================

PLOT_DPI = 300
PLOT_STYLE = "whitegrid"
PLOT_FONT = "Heiti TC"
PLOT_PALETTE = "husl"

# ============================================================================
# 辅助函数
# ============================================================================

def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def get_relative_path(from_file: str, to_path: str) -> str:
    """
    获取相对路径
    
    Parameters:
    -----------
    from_file : str
        源文件的绝对路径
    to_path : str
        目标路径（可以是绝对或相对）
        
    Returns:
    --------
    str : 相对路径
    """
    if os.path.isabs(to_path):
        return to_path
    
    from_dir = os.path.dirname(from_file)
    return os.path.relpath(to_path, from_dir)
