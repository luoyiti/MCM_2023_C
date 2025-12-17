"""
MoE (Mixture of Experts) + MLP + Softmax 分布预测脚本

================================================================================
概述：
    本脚本使用混合专家模型（Mixture of Experts, MoE）来预测 Wordle 游戏中
    玩家猜测次数的概率分布。MoE 模型通过门控网络将不同样本路由到不同的
    专家网络，每个专家是一个 MLP + Softmax 结构，输出 7 维概率分布
    （对应 1次、2次、...、6次、7次及以上猜对的概率）。

模型架构：
    输入 (55维特征) → 门控网络 → 选择 Top-K 专家 → 专家MLP → 加权组合 → 7维概率

主要功能：
    1) 数据加载：读取 mcm_processed_data.csv，提取特征和目标分布
    2) 数据预处理：归一化分布、划分 train/val/test、标准化特征
    3) 模型训练：MoE 模型 + 软标签交叉熵损失 + 早停机制
    4) 模型评估：MAE / KL / JS / CosSim / R² / MaxError 等多维度指标
    5) 可视化：训练曲线、分布对比、误差分析、专家使用率等图表
    6) 报告生成：输出综合总结报告

使用方式：
    python Moe_Softmax.py

输出目录：
    moe_output/ - 包含预测结果CSV、可视化图表、JSON报告、文字总结等

最佳超参数（经调优确定）：
    - num_experts=2: 专家数量设为2，在小样本上更稳定
    - top_k=1: 每个样本只路由到1个专家，形成"硬分群"效果
    - aux_coef=1e-3: 辅助损失系数，平衡专家负载
    - hidden_size=64: 每个专家MLP的隐藏层大小
    - lr=5e-3: 学习率
    - patience=50: 早停耐心值

作者：基于 mlp_forcasting.py 改编
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon

from moe import MoE

from util.visualizations import (
    plot_training_history,
    plot_random_sample_distributions,
    plot_error_analysis,
    plot_uncertainty,
    plot_performance_metrics,
    plot_comprehensive_summary,
)

# 设置中文字体
plt.rcParams['font.family'] = 'Heiti TC'

# ---------------- 全局配置 ----------------
DATA_PATH = "../data/mcm_processed_data.csv"
N_COL = "number_of_reported_results"
OUTPUT_DIR = "moe_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 与 mlp_forcasting.py 保持一致的特征列
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

DIST_COLS = [
    "1_try",
    "2_tries",
    "3_tries",
    "4_tries",
    "5_tries",
    "6_tries",
    "7_or_more_tries_x",
]

# ================================================================================
# 训练超参数配置
# ================================================================================
LR = 5e-3                # 学习率：控制每次参数更新的步长
WEIGHT_DECAY = 1e-4      # 权重衰减（L2正则化）：防止过拟合
MAX_EPOCHS = 500         # 最大训练轮次
PATIENCE = 50            # 早停耐心值：验证损失连续多少轮不下降则停止（调优后从30提高到50）
WEIGHT_MODE = "sqrt"     # 样本权重模式："sqrt"表示用人数的平方根作为权重

# ================================================================================
# MoE（混合专家）超参数配置 - 经过调优确定的最佳参数
# ================================================================================
NUM_EXPERTS = 2          # 专家数量：小样本场景下2个专家更稳定（原为4，调优后改为2）
HIDDEN_SIZE = 64         # 每个专家MLP的隐藏层神经元数量
TOP_K = 1                # Top-K路由：每个样本路由到几个专家（原为2，调优后改为1形成硬分群）
AUX_COEF = 1e-3          # 辅助损失系数：平衡专家负载的正则项（原为1e-2，调优后改为1e-3）


def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    设置随机种子，确保实验可复现
    
    参数:
        seed: 随机种子值，默认为42
    
    说明:
        同时设置numpy、torch CPU和torch GPU的随机种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================================================================================
# 损失函数与样本权重计算
# ================================================================================
def make_weights_from_N(N_array: np.ndarray, mode: str = "sqrt") -> np.ndarray:
    """
    根据每个样本的参与人数N计算样本权重
    
    参数:
        N_array: 每个样本的参与人数数组
        mode: 权重计算模式
            - "sqrt": 使用平方根，适度降低大样本权重
            - "log1p": 使用log(1+N)，更激进地压缩大样本权重
    
    返回:
        归一化后的样本权重数组
    
    原理:
        直接用N作为权重会让参与人数多的样本完全主导训练；
        通过sqrt或log变换，既保留了"人多的样本更重要"的信息，
        又避免了极端不平衡。
    """
    if mode == "sqrt":
        w = np.sqrt(N_array)
    elif mode == "log1p":
        w = np.log1p(N_array)
    else:
        raise ValueError("mode must be 'sqrt' or 'log1p'")
    # 归一化：使权重均值为1，避免影响学习率的有效大小
    w = w / (w.mean() + 1e-12)
    return w.astype(np.float32)


def soft_cross_entropy(p_hat: torch.Tensor, p_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    软标签交叉熵损失函数（无样本权重版本）
    
    参数:
        p_hat: 模型预测的概率分布，形状 (batch_size, 7)
        p_true: 真实的概率分布（软标签），形状 (batch_size, 7)
        eps: 数值稳定性的小常数，防止log(0)
    
    返回:
        批次平均的交叉熵损失
    
    公式:
        CE = -Σ p_true[i] * log(p_hat[i])  对每个样本
        返回所有样本的平均值
    
    与硬标签的区别:
        硬标签只有一个类别概率为1，其余为0；
        软标签是一个真实的概率分布，更适合回归分布的任务。
    """
    # clamp防止log(0)导致的数值问题
    p_hat = torch.clamp(p_hat, eps, 1.0)
    return -(p_true * torch.log(p_hat)).sum(dim=1).mean()


def weighted_soft_cross_entropy(
    p_hat: torch.Tensor, p_true: torch.Tensor, w: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    加权软标签交叉熵损失函数
    
    参数:
        p_hat: 模型预测的概率分布，形状 (batch_size, 7)
        p_true: 真实的概率分布（软标签），形状 (batch_size, 7)
        w: 样本权重，形状 (batch_size,)
        eps: 数值稳定性的小常数
    
    返回:
        加权后的批次平均交叉熵损失
    
    说明:
        参与人数多的样本会获得更高的权重，使模型更关注这些样本的预测准确性
    """
    p_hat = torch.clamp(p_hat, eps, 1.0)
    # 计算每个样本的交叉熵
    per_sample = -(p_true * torch.log(p_hat)).sum(dim=1)
    # 加权平均
    return (w * per_sample).mean()


# ================================================================================
# 数据加载与预处理
# ================================================================================
def load_and_split_data():
    """
    加载数据、预处理并划分数据集
    
    处理流程:
        1. 读取CSV数据文件
        2. 提取特征列，填充缺失值为中位数
        3. 提取目标分布列，归一化为概率分布
        4. 按 70/15/15 划分 train/val/test
        5. 对特征进行Z-score标准化
    
    返回:
        X_train, X_val, X_test: 标准化后的特征矩阵
        P_train, P_val, P_test: 归一化后的目标分布
        N_train, N_val, N_test: 样本人数（用于计算权重）
    """
    # 读取数据文件
    df = pd.read_csv(DATA_PATH)

    # 提取特征列，缺失值用中位数填充
    X = df[FEATURE_COLS].copy()
    X = X.fillna(X.median(numeric_only=True))

    # 提取目标分布列（玩家猜测次数的概率分布）
    P = df[DIST_COLS].copy().fillna(0.0)
    # 如果数据是百分比形式（如 23.5 表示 23.5%），转换为小数
    if P.to_numpy().max() > 1.5:
        P = P / 100.0
    # 确保概率非负
    P = P.clip(lower=0.0)
    # 归一化：确保每行概率和为1
    row_sum = P.sum(axis=1).replace(0, np.nan)
    P = P.div(row_sum, axis=0).fillna(1.0 / len(DIST_COLS))

    # 提取参与人数列（用于计算样本权重）
    if N_COL is not None and N_COL in df.columns:
        N = df[N_COL].fillna(df[N_COL].median()).clip(lower=1)
        N_np = N.to_numpy().astype(np.float32)
    else:
        N_np = None

    # 转换为numpy数组
    X_np = X.to_numpy().astype(np.float32)
    P_np = P.to_numpy().astype(np.float32)

    # 划分数据集：70% 训练集，15% 验证集，15% 测试集
    if N_np is None:
        # 无人数列的情况
        X_train, X_tmp, P_train, P_tmp = train_test_split(
            X_np, P_np, test_size=0.3, random_state=RANDOM_SEED
        )
        X_val, X_test, P_val, P_test = train_test_split(
            X_tmp, P_tmp, test_size=0.5, random_state=RANDOM_SEED
        )
        N_train = N_val = N_test = None
    else:
        # 有人数列的情况，同步划分
        X_train, X_tmp, P_train, P_tmp, N_train, N_tmp = train_test_split(
            X_np, P_np, N_np, test_size=0.3, random_state=RANDOM_SEED
        )
        X_val, X_test, P_val, P_test, N_val, N_test = train_test_split(
            X_tmp, P_tmp, N_tmp, test_size=0.5, random_state=RANDOM_SEED
        )

    # Z-score标准化：使特征均值为0，标准差为1
    # 注意：只在训练集上fit，验证集和测试集只做transform
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_val, X_test, P_train, P_val, P_test, N_train, N_val, N_test


# ================================================================================
# MoE 模型训练
# ================================================================================
def train_moe(
    X_train: np.ndarray,
    P_train: np.ndarray,
    X_val: np.ndarray,
    P_val: np.ndarray,
    Wtr: torch.Tensor | None,
    Wva: torch.Tensor | None,
) -> tuple:
    """
    训练混合专家（MoE）模型
    
    参数:
        X_train: 训练集特征，形状 (n_train, n_features)
        P_train: 训练集目标分布，形状 (n_train, 7)
        X_val: 验证集特征
        P_val: 验证集目标分布
        Wtr: 训练集样本权重，可为None
        Wva: 验证集样本权重，可为None
    
    返回:
        model: 训练好的MoE模型（已加载最佳参数）
        info: 训练信息字典，包含损失曲线等
    
    MoE模型结构:
        - 门控网络: 根据输入特征决定将样本路由到哪些专家
        - 专家网络: 每个专家是一个MLP+Softmax，输出7维概率
        - 输出: 加权组合被选中专家的输出
    
    损失函数:
        总损失 = 主损失(软标签交叉熵) + AUX_COEF * 辅助损失(负载平衡)
    """
    
    # 创建MoE模型
    # input_size: 输入特征维度
    # output_size: 输出维度（7个概率值）
    # num_experts: 专家数量
    # hidden_size: 每个专家MLP的隐藏层大小
    # noisy_gating: 是否在门控中加入噪声（有助于探索）
    # k: 每个样本路由到的专家数量
    model = MoE(
        input_size=X_train.shape[1],
        output_size=7,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        noisy_gating=True,
        k=TOP_K,
    ).to(DEVICE)

    # 使用Adam优化器，带权重衰减防止过拟合
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 将数据转换为PyTorch张量并移到设备上
    Xtr = torch.tensor(X_train, device=DEVICE)
    Ptr = torch.tensor(P_train, device=DEVICE)
    Xva = torch.tensor(X_val, device=DEVICE)
    Pva = torch.tensor(P_val, device=DEVICE)

    # 训练状态跟踪变量
    best_state = None              # 最佳模型参数
    best_val_loss = float("inf")   # 最佳验证损失
    bad = 0                        # 无改善的连续轮次计数
    train_losses: list[float] = [] # 训练损失历史
    val_losses: list[float] = []   # 验证损失历史
    aux_losses: list[float] = []   # 辅助损失历史（专家负载平衡）

    # 训练循环
    for epoch in range(1, MAX_EPOCHS + 1):
        # ========== 训练阶段 ==========
        model.train()
        # MoE前向传播返回: (p_hat, aux_loss)
        # p_hat: 预测的概率分布
        # aux_loss: 辅助损失（用于平衡专家负载，防止"专家塔陷"）
        p_hat, aux_loss = model(Xtr)
        
        # 计算主损失（软标签交叉熵）
        if Wtr is None:
            loss_main = soft_cross_entropy(p_hat, Ptr)
        else:
            loss_main = weighted_soft_cross_entropy(p_hat, Ptr, Wtr)
        
        # 总损失 = 主损失 + 辅助损失系数 * 辅助损失
        loss = loss_main + AUX_COEF * aux_loss

        # 反向传播和参数更新
        opt.zero_grad()
        loss.backward()
        opt.step()

        # ========== 验证阶段 ==========
        model.eval()
        with torch.no_grad():
            p_val, aux_val = model(Xva)
            if Wva is None:
                val_main = soft_cross_entropy(p_val, Pva)
            else:
                val_main = weighted_soft_cross_entropy(p_val, Pva, Wva)
            val_loss = val_main + AUX_COEF * aux_val

        # 记录损失历史
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        aux_losses.append(aux_loss.item())

        # 早停逻辑：如果验证损失有改善，保存最佳模型
        if val_loss.item() < best_val_loss - 1e-6:
            best_val_loss = val_loss.item()
            # 深拷贝模型参数到CPU
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        # 每50轮打印一次进度
        if epoch % 50 == 0:
            print(f"[MoE] epoch={epoch:3d} train_loss={loss.item():.4f} val_loss={val_loss.item():.4f} aux_loss={aux_loss.item():.6f}")

        # 早停：连续PATIENCE轮无改善则停止训练
        if bad >= PATIENCE:
            print(f"[MoE] Early stopping at epoch {epoch}.")
            break

    # 加载最佳模型参数
    if best_state:
        model.load_state_dict(best_state)

    info = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "aux_losses": aux_losses,
        "best_epoch": len(train_losses) - bad,
        "best_val_loss": best_val_loss,
        "bad": bad,
    }
    return model, info


# ================================================================================
# 模型评估
# ================================================================================
def evaluate(model, X_test: np.ndarray, P_test: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    在测试集上评估MoE模型
    
    参数:
        model: 训练好的MoE模型
        X_test: 测试集特征
        P_test: 测试集真实分布
    
    返回:
        P_pred: 预测的概率分布
        metrics: 评估指标字典
    
    评估指标说明:
        - MAE: 平均绝对误差，衡量每个概率值的平均偏差
        - KL: KL散度，衡量分布差异（越小越好）
        - JS: Jensen-Shannon散度，对称的分布差异度量
        - CosSim: 余弦相似度，衡量分布向量的方向相似性
        - R²: 决定系数，衡量模型解释方差的比例
        - MaxError: 最大误差，单个概率值的最大偏差
    """
    model.eval()
    Xte = torch.tensor(X_test, device=DEVICE)
    with torch.no_grad():
        P_pred, _ = model(Xte)  # 忽略辅助损失
        P_pred = P_pred.cpu().numpy()

    # 计算各项评估指标
    mae = np.mean(np.abs(P_pred - P_test))  # 平均绝对误差
    eps = 1e-12
    # KL散度: KL(p_true || p_pred)
    kl = np.mean(np.sum(P_test * (np.log(P_test + eps) - np.log(P_pred + eps)), axis=1))
    # JS散度: 对称版本的KL
    js_mean = np.mean([jensenshannon(P_test[i], P_pred[i]) for i in range(len(P_test))])
    # 余弦相似度
    cos_sim = np.mean([cosine_similarity([P_test[i]], [P_pred[i]])[0, 0] for i in range(len(P_test))])
    # R²决定系数
    r2 = r2_score(P_test, P_pred)
    # 最大误差
    max_error = np.max(np.abs(P_pred - P_test))

    metrics = {
        "mae": float(mae),
        "kl": float(kl),
        "js_mean": float(js_mean),
        "cos_sim": float(cos_sim),
        "r2": float(r2),
        "max_error": float(max_error),
    }
    
    # 打印评估结果
    print("\n[MoE] 测试集评估")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    return P_pred, metrics


def save_predictions(P_pred: np.ndarray, path: str) -> None:
    """
    保存预测结果到CSV文件
    
    参数:
        P_pred: 预测的概率分布，形状 (n_samples, 7)
        path: 保存路径
    
    输出文件格式:
        每行一个样本，7列对应各猜测次数的预测概率
    """
    df_pred = pd.DataFrame(P_pred, columns=[f"moe_pred_{c}" for c in DIST_COLS])
    df_pred.to_csv(path, index=False)
    print(f"[MoE] 预测结果已保存: {path}")


def write_report(metrics: dict, info: dict, path: str) -> None:
    lines = {
        "model": "MoE + MLP + Softmax",
        "num_experts": NUM_EXPERTS,
        "hidden_size": HIDDEN_SIZE,
        "top_k": TOP_K,
        "aux_coef": AUX_COEF,
        "best_epoch": info.get("best_epoch"),
        "best_val_loss": info.get("best_val_loss"),
        "metrics": metrics,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(lines, ensure_ascii=False, indent=2))
    print(f"[MoE] 报告已保存: {path}")


# ================================================================================
# MoE特有可视化：专家使用率分析
# ================================================================================
def analyze_expert_usage(model, X_data: np.ndarray, save_path: str):
    """
    分析并可视化每个专家的使用率
    
    参数:
        model: 训练好的MoE模型
        X_data: 用于分析的特征数据
        save_path: 图表保存路径
    
    返回:
        包含专家使用率和平均门控权重的字典
    
    说明:
        这个分析可以帮助理解:
        1. 门控网络是否有效地分配样本到不同专家
        2. 是否存在"专家塔陷"（所有样本都路由到同一个专家）
        3. 各专家的负载是否平衡
    """
    model.eval()
    Xte = torch.tensor(X_data, device=DEVICE)
    
    with torch.no_grad():
        gates, _ = model.noisy_top_k_gating(Xte, train=False)
        gates_np = gates.cpu().numpy()
    
    expert_usage = (gates_np > 0).mean(axis=0)
    expert_avg_weight = gates_np.mean(axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(NUM_EXPERTS)
    axes[0].bar(x, expert_usage, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Expert Index')
    axes[0].set_ylabel('Usage Rate')
    axes[0].set_title('Expert Usage Rate (fraction of samples using each expert)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Expert {i}' for i in range(NUM_EXPERTS)])
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(x, expert_avg_weight, color='coral', edgecolor='black')
    axes[1].set_xlabel('Expert Index')
    axes[1].set_ylabel('Average Gate Weight')
    axes[1].set_title('Average Gate Weight per Expert')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'Expert {i}' for i in range(NUM_EXPERTS)])
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[MoE] 专家使用率图已保存: {save_path}")
    
    return {
        "expert_usage": expert_usage.tolist(),
        "expert_avg_weight": expert_avg_weight.tolist(),
    }


def explain_expert_distributions(
    model,
    X_data: np.ndarray,
    P_true: np.ndarray,
    P_pred: np.ndarray,
    output_dir: str,
    prefix: str = "test",
):
    """
    对不同专家负责的样本分布进行解释（按门控路由分组）

    该函数回答两个核心问题：
        1) 每个专家主要负责哪一类样本？（样本占比、分布形状）
        2) 每个专家对这些样本的预测是否贴近真实分布？（专家内的真实/预测均值分布对比）

    输出：
        1) CSV：每个专家的样本数、占比、真实均值分布、预测均值分布、每桶MAE
        2) PNG：
            - 专家样本占比柱状图
            - 每个专家的真实vs预测均值分布对比图
    """

    model.eval()
    Xte = torch.tensor(X_data, device=DEVICE)
    with torch.no_grad():
        # gates: (n_samples, num_experts)，只有Top-K专家位置非0
        gates, _ = model.noisy_top_k_gating(Xte, train=False)
        gates_np = gates.cpu().numpy()

    # 对于 k=1，这里基本就是 one-hot；对于 k>1，这里表示“最主要专家”
    assigned_expert = gates_np.argmax(axis=1)

    # 统计每个专家分到的样本
    rows = []
    for e in range(NUM_EXPERTS):
        mask = assigned_expert == e
        cnt = int(mask.sum())
        if cnt == 0:
            # 避免空专家导致后续均值计算报错
            mean_true = np.zeros(len(DIST_COLS), dtype=np.float32)
            mean_pred = np.zeros(len(DIST_COLS), dtype=np.float32)
            mae_per_bin = np.zeros(len(DIST_COLS), dtype=np.float32)
        else:
            mean_true = P_true[mask].mean(axis=0)
            mean_pred = P_pred[mask].mean(axis=0)
            mae_per_bin = np.abs(P_pred[mask] - P_true[mask]).mean(axis=0)

        row = {
            "expert": e,
            "count": cnt,
            "ratio": float(cnt / len(X_data)) if len(X_data) > 0 else 0.0,
        }
        for i, c in enumerate(DIST_COLS):
            row[f"mean_true_{c}"] = float(mean_true[i])
            row[f"mean_pred_{c}"] = float(mean_pred[i])
            row[f"mae_{c}"] = float(mae_per_bin[i])
        rows.append(row)

    df_exp = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"moe_expert_distribution_summary_{prefix}.csv")
    df_exp.to_csv(csv_path, index=False)
    print(f"[MoE] 专家分布解释CSV已保存: {csv_path}")

    # ---------------- 可视化 1：样本占比 ----------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df_exp["expert"].astype(str), df_exp["ratio"], color="slateblue", edgecolor="black")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Sample Ratio")
    ax.set_title(f"Expert Sample Ratio ({prefix})")
    ax.grid(axis="y", alpha=0.3)
    ratio_path = os.path.join(output_dir, f"moe_expert_sample_ratio_{prefix}.png")
    plt.tight_layout()
    plt.savefig(ratio_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[MoE] 专家样本占比图已保存: {ratio_path}")

    # ---------------- 可视化 2：每个专家的均值分布对比 ----------------
    n_bins = len(DIST_COLS)
    x = np.arange(n_bins)
    fig, axes = plt.subplots(NUM_EXPERTS, 1, figsize=(12, 3.2 * NUM_EXPERTS), sharex=True)
    if NUM_EXPERTS == 1:
        axes = [axes]

    for e in range(NUM_EXPERTS):
        sub = df_exp[df_exp["expert"] == e].iloc[0]
        mean_true = np.array([sub[f"mean_true_{c}"] for c in DIST_COLS])
        mean_pred = np.array([sub[f"mean_pred_{c}"] for c in DIST_COLS])
        width = 0.35

        ax = axes[e]
        ax.bar(x - width / 2, mean_true, width, label="Mean True", color="steelblue", edgecolor="black")
        ax.bar(x + width / 2, mean_pred, width, label="Mean Pred", color="coral", edgecolor="black")
        ax.set_title(f"Expert {e} | count={int(sub['count'])} | ratio={sub['ratio']:.2f}")
        ax.set_ylabel("Probability")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(DIST_COLS, rotation=0)
    axes[-1].set_xlabel("Bins (tries)")

    dist_path = os.path.join(output_dir, f"moe_expert_mean_distribution_{prefix}.png")
    plt.tight_layout()
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[MoE] 专家均值分布对比图已保存: {dist_path}")

    # ---------------- 可视化 3：合并曲线图（更直观显示专家偏向） ----------------
    # 说明：
    #   - 将不同专家的“均值预测分布”画在同一张图中（折线/曲线）
    #   - 可选叠加该专家负责样本的“真实均值分布”（虚线），用于判断偏差来源
    fig, ax = plt.subplots(figsize=(12, 5))
    x_labels = DIST_COLS
    for e in range(NUM_EXPERTS):
        sub = df_exp[df_exp["expert"] == e].iloc[0]
        mean_true = np.array([sub[f"mean_true_{c}"] for c in DIST_COLS])
        mean_pred = np.array([sub[f"mean_pred_{c}"] for c in DIST_COLS])

        ax.plot(
            x,
            mean_pred,
            marker="o",
            linewidth=2.0,
            label=f"Expert {e} Mean Pred",
        )
        ax.plot(
            x,
            mean_true,
            marker="x",
            linestyle="--",
            linewidth=1.6,
            alpha=0.9,
            label=f"Expert {e} Mean True",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Bins (tries)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Expert Mean Distribution Curves ({prefix})")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)

    curve_path = os.path.join(output_dir, f"moe_expert_mean_distribution_curve_{prefix}.png")
    plt.tight_layout()
    plt.savefig(curve_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[MoE] 专家均值分布曲线图已保存: {curve_path}")

    return {
        "csv_path": csv_path,
        "ratio_path": ratio_path,
        "dist_path": dist_path,
        "curve_path": curve_path,
    }


def plot_aux_loss_curve(aux_losses: list, save_path: str):
    """
    绘制辅助损失曲线
    
    参数:
        aux_losses: 每个epoch的辅助损失列表
        save_path: 图表保存路径
    
    说明:
        辅助损失用于平衡专家负载，理想情况下应该:
        - 保持较低的值
        - 随训练进行逐渐稳定
        - 不应该出现剧烈波动
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(aux_losses, label='Auxiliary Loss (Load Balancing)', color='green', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Auxiliary Loss')
    ax.set_title('MoE Auxiliary Loss During Training')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[MoE] 辅助损失曲线已保存: {save_path}")


def generate_summary_report(metrics: dict, info: dict, expert_stats: dict, output_dir: str):
    """生成综合总结报告"""
    report = f"""
{'='*80}
MoE (Mixture of Experts) + MLP + Softmax 分布预测总结报告
{'='*80}

一、模型配置
-----------
专家数量 (num_experts): {NUM_EXPERTS}
每个专家的隐藏层大小 (hidden_size): {HIDDEN_SIZE}
Top-K 路由 (k): {TOP_K}
辅助损失系数 (aux_coef): {AUX_COEF}
学习率 (lr): {LR}
权重衰减 (weight_decay): {WEIGHT_DECAY}
早停耐心 (patience): {PATIENCE}

二、训练结果
-----------
最佳验证轮次: {info.get('best_epoch')}
最佳验证损失: {info.get('best_val_loss'):.6f}
总训练轮次: {len(info.get('train_losses', []))}

三、测试集评估指标
-----------------
MAE (平均绝对误差): {metrics['mae']:.6f}
  - 含义: 7个桶的概率平均误差约 {metrics['mae']*100:.2f}%

KL散度 (KL Divergence): {metrics['kl']:.6f}
  - 含义: 预测分布与真实分布的信息差异 (越小越好)

JS散度 (Jensen-Shannon): {metrics['js_mean']:.6f}
  - 含义: 对称的分布差异度量 (0~1, 越小越好)

余弦相似度 (Cosine Similarity): {metrics['cos_sim']:.6f}
  - 含义: 分布向量的相似程度 (越接近1越好)

R² (决定系数): {metrics['r2']:.6f}
  - 含义: 模型解释方差的比例 (越接近1越好)

最大误差 (Max Error): {metrics['max_error']:.6f}
  - 含义: 单个桶上的最大预测偏差

四、专家使用率分析
-----------------
"""
    for i in range(NUM_EXPERTS):
        usage = expert_stats['expert_usage'][i]
        weight = expert_stats['expert_avg_weight'][i]
        report += f"Expert {i}: 使用率={usage*100:.1f}%, 平均门控权重={weight:.4f}\n"

    report += f"""
五、模型解读
-----------
1. MoE 模型通过门控网络将不同样本路由到不同的专家网络
2. 每个样本由 top-{TOP_K} 个专家共同处理，加权组合输出
3. 辅助损失确保各专家被均衡使用，避免"专家塌陷"

六、输出文件
-----------
预测结果: {OUTPUT_DIR}/moe_softmax_pred_output.csv
训练曲线: {OUTPUT_DIR}/moe_training_history.png
分布对比: {OUTPUT_DIR}/moe_distribution_comparison.png
误差分析: {OUTPUT_DIR}/moe_error_analysis.png
专家使用率: {OUTPUT_DIR}/moe_expert_usage.png
辅助损失曲线: {OUTPUT_DIR}/moe_aux_loss.png
综合汇总: {OUTPUT_DIR}/moe_comprehensive_summary.png
JSON报告: {OUTPUT_DIR}/moe_report.json

{'='*80}
报告结束
{'='*80}
"""
    
    report_path = os.path.join(output_dir, "moe_summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(report)
    print(f"\n[MoE] 综合报告已保存: {report_path}")


# ---------------- 主流程 ----------------
def main():
    set_seed()
    print(f"设备: {DEVICE}")
    print(f"数据路径: {DATA_PATH}")
    print(f"MoE 配置: num_experts={NUM_EXPERTS}, hidden_size={HIDDEN_SIZE}, k={TOP_K}")

    (
        X_train,
        X_val,
        X_test,
        P_train,
        P_val,
        P_test,
        N_train,
        N_val,
        N_test,
    ) = load_and_split_data()

    print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")

    Wtr = torch.tensor(make_weights_from_N(N_train, WEIGHT_MODE), device=DEVICE) if N_train is not None else None
    Wva = torch.tensor(make_weights_from_N(N_val, WEIGHT_MODE), device=DEVICE) if N_val is not None else None

    model, info = train_moe(X_train, P_train, X_val, P_val, Wtr, Wva)
    P_pred, metrics = evaluate(model, X_test, P_test)

    pred_path = os.path.join(OUTPUT_DIR, "moe_softmax_pred_output.csv")
    save_predictions(P_pred, pred_path)

    # 可视化
    hist_info = plot_training_history(
        train_losses=info["train_losses"],
        val_losses=info["val_losses"],
        bad=info["bad"],
        best_val_loss=info["best_val_loss"],
        save_path=os.path.join(OUTPUT_DIR, "moe_training_history.png"),
    )

    sample_indices = plot_random_sample_distributions(
        P_test=P_test,
        P_pred=P_pred,
        sample_size=10,
        save_path=os.path.join(OUTPUT_DIR, "moe_distribution_comparison.png"),
    )

    error_stats = plot_error_analysis(
        P_test=P_test,
        P_pred=P_pred,
        save_path=os.path.join(OUTPUT_DIR, "moe_error_analysis.png"),
    )

    perf_metrics = plot_performance_metrics(
        P_test=P_test,
        P_pred=P_pred,
        save_path=os.path.join(OUTPUT_DIR, "moe_performance_metrics.png"),
    )

    plot_comprehensive_summary(
        train_losses=info["train_losses"],
        val_losses=info["val_losses"],
        bad=info["bad"],
        best_val_loss=info["best_val_loss"],
        mae=metrics["mae"],
        rmse=perf_metrics["rmse"],
        kl=metrics["kl"],
        js_mean=metrics["js_mean"],
        cos_sim=metrics["cos_sim"],
        r2=metrics["r2"],
        mae_per_dim=error_stats["mae_per_dim"],
        P_test=P_test,
        P_pred=P_pred,
        errors=error_stats["errors"],
        save_path=os.path.join(OUTPUT_DIR, "moe_comprehensive_summary.png"),
    )

    # MoE 特有的可视化
    expert_stats = analyze_expert_usage(model, X_test, os.path.join(OUTPUT_DIR, "moe_expert_usage.png"))
    plot_aux_loss_curve(info["aux_losses"], os.path.join(OUTPUT_DIR, "moe_aux_loss.png"))

    # 专家分布解释：按门控把样本分配给专家，然后统计每个专家的真实/预测均值分布
    explain_expert_distributions(
        model=model,
        X_data=X_test,
        P_true=P_test,
        P_pred=P_pred,
        output_dir=OUTPUT_DIR,
        prefix="test",
    )

    # 报告
    report_path = os.path.join(OUTPUT_DIR, "moe_report.json")
    write_report(metrics, info, report_path)

    # 综合总结
    generate_summary_report(metrics, info, expert_stats, OUTPUT_DIR)

    print("\n完成：MoE + MLP + Softmax 分布预测与可视化已输出到 moe_output/")


if __name__ == "__main__":
    main()
