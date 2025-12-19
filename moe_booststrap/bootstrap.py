"""
Bootstrap 不确定性估计模块

包含 Bootstrap 预测、置信区间计算和不确定性分析功能。
"""

import numpy as np
import torch

from .train import train_moe
from .data import make_weights_from_N
from .metrics import compute_metrics
from .config import (
    DEVICE,
    WEIGHT_MODE,
    NUM_EXPERTS,
    HIDDEN_SIZE,
    TOP_K,
    AUX_COEF,
    EXPERT_DIVERSITY_COEF,
    BOOTSTRAP_B,
    BOOTSTRAP_EPOCH_SCALE,
    BOOTSTRAP_CI_LEVEL,
    MAX_EPOCHS,
)


def bootstrap_predict(
    X_train: np.ndarray,
    P_train: np.ndarray,
    N_train: np.ndarray,
    X_val: np.ndarray,
    P_val: np.ndarray,
    N_val: np.ndarray,
    X_test: np.ndarray,
    X_holdout: np.ndarray = None,
    B: int = None,
    num_experts: int = None,
    hidden_size: int = None,
    top_k: int = None,
    max_epochs: int = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    使用 Bootstrap 对 MoE 模型进行不确定性估计。

    做法：
        重复 B 次：
            1) 对训练集进行 bootstrap 重采样（有放回，size=n_train）
            2) 训练一个 MoE 模型（不改结构与 loss）
            3) 在同一个测试集上预测，得到一份分布 P_pred^(b)

    参数:
        X_train, P_train, N_train: 训练数据
        X_val, P_val, N_val: 验证数据
        X_test: 测试特征
        X_holdout: Holdout 特征（可选）
        B: Bootstrap 次数
        num_experts, hidden_size, top_k: MoE 超参数
        max_epochs: 每次 bootstrap 的最大轮次
        verbose: 是否打印进度
    
    返回:
        - 若 X_holdout is None: P_test_all, shape (B, n_test, 7)
        - 否则: (P_test_all, P_holdout_all)
    """
    if B is None:
        B = BOOTSTRAP_B
    if num_experts is None:
        num_experts = NUM_EXPERTS
    if hidden_size is None:
        hidden_size = HIDDEN_SIZE
    if top_k is None:
        top_k = TOP_K
    if max_epochs is None:
        max_epochs = max(1, int(MAX_EPOCHS * BOOTSTRAP_EPOCH_SCALE))

    P_test_all = []
    P_holdout_all = []
    n_train = X_train.shape[0]

    # 预先计算验证集权重
    Wva = (
        torch.tensor(make_weights_from_N(N_val, WEIGHT_MODE), device=DEVICE)
        if N_val is not None
        else None
    )

    for b in range(B):
        if verbose:
            print(f"[Bootstrap] Run {b + 1}/{B}")

        # Bootstrap 重采样训练集
        idx = np.random.choice(n_train, size=n_train, replace=True)
        Xb = X_train[idx]
        Pb = P_train[idx]
        Nb = N_train[idx] if N_train is not None else None

        # 重新计算训练集权重
        Wb = (
            torch.tensor(make_weights_from_N(Nb, WEIGHT_MODE), device=DEVICE)
            if Nb is not None
            else None
        )

        # 训练模型
        model_b, _ = train_moe(
            Xb, Pb, X_val, P_val, Wb, Wva,
            num_experts=num_experts,
            hidden_size=hidden_size,
            top_k=top_k,
            aux_coef=AUX_COEF,
            expert_diversity_coef=EXPERT_DIVERSITY_COEF,
            max_epochs=max_epochs,
            verbose=False,
        )

        # 在 test 集预测
        model_b.eval()
        with torch.no_grad():
            Xte = torch.tensor(X_test, device=DEVICE)
            P_pred_b, _ = model_b(Xte)
            P_pred_b = P_pred_b.cpu().numpy()

            if X_holdout is not None:
                Xho = torch.tensor(X_holdout, device=DEVICE)
                P_hold_b, _ = model_b(Xho)
                P_hold_b = P_hold_b.cpu().numpy()

        P_test_all.append(P_pred_b)
        if X_holdout is not None:
            P_holdout_all.append(P_hold_b)

    P_test_boot = np.stack(P_test_all, axis=0)
    if X_holdout is None:
        return P_test_boot
    return P_test_boot, np.stack(P_holdout_all, axis=0)


def bootstrap_summary(
    P_boot: np.ndarray, 
    ci_level: float = None
) -> tuple:
    """
    对 bootstrap 预测做聚合，返回 (mean, std, low, high)。
    
    参数:
        P_boot: Bootstrap 预测结果，形状 (B, n_samples, 7)
        ci_level: 置信区间水平
    
    返回:
        (P_mean, P_std, P_low, P_high) 统计量
    """
    if ci_level is None:
        ci_level = BOOTSTRAP_CI_LEVEL
        
    alpha_low = (1 - ci_level) / 2
    alpha_high = 1 - alpha_low

    P_mean = P_boot.mean(axis=0)
    P_std = P_boot.std(axis=0)
    P_low = np.percentile(P_boot, alpha_low * 100, axis=0)
    P_high = np.percentile(P_boot, alpha_high * 100, axis=0)

    return P_mean, P_std, P_low, P_high


def compute_confidence_scores(P_std: np.ndarray) -> dict:
    """
    给报告用的"全局置信度/稳定性"统计。
    
    参数:
        P_std: 预测标准差，形状 (n_samples, 7)
    
    返回:
        包含置信度统计信息的字典
    """
    per_sample_std_mean = P_std.mean(axis=1)
    confidence_score = 1.0 / (per_sample_std_mean + 1e-6)
    return {
        "per_sample_std_mean": per_sample_std_mean,
        "confidence_score": confidence_score,
        "confidence_score_mean": float(confidence_score.mean()),
        "confidence_score_std": float(confidence_score.std()),
        "confidence_score_min": float(confidence_score.min()),
        "confidence_score_max": float(confidence_score.max()),
    }


def bootstrap_evaluate(
    P_boot: np.ndarray,
    P_test: np.ndarray,
    ci_level: float = None,
) -> dict:
    """
    评估 Bootstrap 预测结果。
    
    参数:
        P_boot: Bootstrap 预测结果
        P_test: 真实标签
        ci_level: 置信区间水平
    
    返回:
        包含 Bootstrap 评估指标的字典
    """
    P_mean, P_std, P_low, P_high = bootstrap_summary(P_boot, ci_level)
    
    # 使用均值预测计算指标
    metrics = compute_metrics(P_mean, P_test)
    
    # 置信度统计
    conf = compute_confidence_scores(P_std)
    
    # 置信区间覆盖率
    in_ci = ((P_test >= P_low) & (P_test <= P_high)).mean()
    
    return {
        "metrics": metrics,
        "confidence": conf,
        "ci_coverage": float(in_ci),
        "P_mean": P_mean,
        "P_std": P_std,
        "P_low": P_low,
        "P_high": P_high,
    }
