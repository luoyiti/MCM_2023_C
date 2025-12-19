"""
MoE 模型配置文件

包含所有超参数、路径配置和特征列定义。
"""

import os
import torch

# ========================== 路径配置 ==========================
# 获取当前模块所在目录
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)

# 数据路径
DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "mcm_processed_data.csv")

# 输出目录
OUTPUT_DIR = os.path.join(_MODULE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================== 设备配置 ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
RANDOM_SEED = SEED  # 别名

# ========================== Holdout 配置 ==========================
HOLDOUT_WORD = "eerie"
WORD_COL_CANDIDATES = ["word", "Word", "target_word", "answer", "Answer"]

# ========================== 特征列配置 ==========================
FEATURE_COLS = [
    "Zipf-value",
    "letter_entropy",
    "feedback_entropy",
    "max_consecutive_vowels",
    "letter_freq_mean",
    "scrabble_score",
    "has_common_suffix",
    "num_rare_letters",
    "position_rarity",
    "positional_freq_min",
    "hamming_neighbors",
    "keyboard_distance",
    "semantic_distance",
    "1_try_simulate_random",
    "2_try_simulate_random",
    "3_try_simulate_random",
    "4_try_simulate_random",
    "5_try_simulate_random",
    "6_try_simulate_random",
    "7_try_simulate_random",
    "1_try_simulate_freq",
    "2_try_simulate_freq",
    "3_try_simulate_freq",
    "4_try_simulate_freq",
    "5_try_simulate_freq",
    "6_try_simulate_freq",
    "7_try_simulate_freq",
    "1_try_simulate_entropy",
    "2_try_simulate_entropy",
    "3_try_simulate_entropy",
    "4_try_simulate_entropy",
    "5_try_simulate_entropy",
    "6_try_simulate_entropy",
    "7_try_simulate_entropy",
    "rl_1_try_low_training",
    "rl_2_try_low_training",
    "rl_3_try_low_training",
    "rl_4_try_low_training",
    "rl_5_try_low_training",
    "rl_6_try_low_training",
    "rl_7_try_low_training",
    "rl_1_try_high_training",
    "rl_2_try_high_training",
    "rl_3_try_high_training",
    "rl_4_try_high_training",
    "rl_5_try_high_training",
    "rl_6_try_high_training",
    "rl_7_try_high_training",
    "rl_1_try_little_training",
    "rl_2_try_little_training",
    "rl_3_try_little_training",
    "rl_4_try_little_training",
    "rl_5_try_little_training",
    "rl_6_try_little_training",
    "rl_7_try_little_training",
]

# 分布列（目标变量）
DIST_COLS = [
    "1_try",
    "2_tries",
    "3_tries",
    "4_tries",
    "5_tries",
    "6_tries",
    "7_or_more_tries_x",
]

# 参与人数列
N_COL = "number_of_reported_results"

# ========================== 训练超参数 ==========================
LR = 5e-3
WD = 1e-4  # Weight Decay
WEIGHT_DECAY = WD  # 别名
MAX_EPOCHS = 500
PATIENCE = 50
WEIGHT_MODE = "sqrt"

# ========================== MoE 超参数 ==========================
# 最佳配置（经 Moe_Softmax.py 调优验证）：
# - NUM_EXPERTS=2: 小样本上更稳定，更容易形成明确的专家分化
# - TOP_K=1: 形成"硬分群"，每个样本只路由到1个专家，最大化专家差异
# - AUX_COEF=1e-3: 平衡专家负载，避免专家塌陷
NUM_EXPERTS = 2
HIDDEN_SIZE = 64
TOP_K = 1
DROPOUT = 0.1
AUX_LOSS_WEIGHT = 1e-3
AUX_COEF = AUX_LOSS_WEIGHT  # 别名
# 【已禁用】参数差异化系数设为 0，观察无强制差异化时专家的自然分化
EXPERT_DIVERSITY_COEF = 0.0  # 参数差异化系数（已禁用）

# ========================== 专家输出差异化配置 ==========================
# 输出差异化损失：直接惩罚专家输出相似度，是真正实现专家分化的关键
# EXPERT_OUTPUT_DIVERSITY_COEF: 惩罚专家输出的余弦相似度
# EXPERT_JS_DIVERGENCE_COEF: 鼓励专家输出分布的JS散度（设为负值实现最大化）
# 【已禁用】设为 0 以观察无强制差异化时专家的自然分化行为
EXPERT_OUTPUT_DIVERSITY_COEF = 0.0  # 输出相似度惩罚权重（已禁用）
EXPERT_JS_DIVERGENCE_COEF = 0.0     # JS散度差异鼓励权重（已禁用）

# ========================== 网格搜索配置 ==========================
ENABLE_EXPERT_TOPK_SEARCH = True
# 聚焦于2专家配置，验证TOP_K=1的优越性
GRID_NUM_EXPERTS = [2, 3, 4]
GRID_TOP_K = [1, 2]
GRID_HIDDEN_SIZE = [64, 128]  # 减少搜索空间，聚焦最优区域
GRID_EPOCHS = int(MAX_EPOCHS * 0.5)
EXPERT_NUM_GRID = GRID_NUM_EXPERTS  # 别名
TOPK_GRID = GRID_TOP_K  # 别名
HIDDEN_SIZE_GRID = GRID_HIDDEN_SIZE  # 别名
GRID_SEARCH_EPOCH_SCALE = 0.5
GRID_SEARCH_WITH_BOOTSTRAP = False
GRID_SEARCH_BOOTSTRAP_B = 20

# ========================== 专家分化搜索配置 ==========================
ENABLE_SPECIALIZATION_SEARCH = True
SEARCH_MAX_TRIALS = 10
SEARCH_EPOCH_SCALE = 0.4
# 针对2专家+TOP_K=1配置，优化辅助损失系数
SEARCH_AUX_COEF_GRID = [5e-4, 1e-3, 2e-3]
# 探索更高的差异化系数以最大化专家差异
SEARCH_DIVERSITY_COEF_GRID = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
SEARCH_OBJECTIVE_BETA_JS = 0.5

# ========================== Bootstrap 配置 ==========================
BOOTSTRAP_B = 200
BOOTSTRAP_EPOCH_SCALE = 0.8
BOOTSTRAP_CI = 0.95
BOOTSTRAP_CI_LEVEL = BOOTSTRAP_CI  # 别名

# ========================== 环境变量覆盖 ==========================
def apply_env_overrides():
    """从环境变量中读取配置覆盖"""
    global ENABLE_SPECIALIZATION_SEARCH, BOOTSTRAP_B, MAX_EPOCHS
    global EXPERT_NUM_GRID, TOPK_GRID, HIDDEN_SIZE_GRID
    
    try:
        ENABLE_SPECIALIZATION_SEARCH = bool(int(os.getenv(
            "MOE_ENABLE_SEARCH", str(int(ENABLE_SPECIALIZATION_SEARCH)))))
    except Exception:
        pass
    
    try:
        BOOTSTRAP_B = int(os.getenv("MOE_BOOTSTRAP_B", str(BOOTSTRAP_B)))
    except Exception:
        pass
    
    try:
        _override_epochs = os.getenv("MOE_MAX_EPOCHS")
        if _override_epochs is not None:
            MAX_EPOCHS = int(_override_epochs)
    except Exception:
        pass
    
    try:
        _expert_grid_env = os.getenv("MOE_EXPERT_GRID")
        if _expert_grid_env:
            EXPERT_NUM_GRID = [int(x) for x in _expert_grid_env.split(",")]
    except Exception:
        pass
    
    try:
        _topk_grid_env = os.getenv("MOE_TOPK_GRID")
        if _topk_grid_env:
            TOPK_GRID = [int(x) for x in _topk_grid_env.split(",")]
    except Exception:
        pass
    
    try:
        _hidden_grid_env = os.getenv("MOE_HIDDEN_GRID")
        if _hidden_grid_env:
            HIDDEN_SIZE_GRID = [int(x) for x in _hidden_grid_env.split(",")]
    except Exception:
        pass

# 应用环境变量覆盖
apply_env_overrides()
