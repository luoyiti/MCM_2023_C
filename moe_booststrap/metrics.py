"""
评估指标模块

包含分布预测的各种性能指标计算。
"""

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity


def compute_metrics(P_pred: np.ndarray, P_test: np.ndarray) -> dict:
    """
    计算预测分布与真实分布之间的多种性能指标。
    
    参数:
        P_pred: 预测的概率分布，形状 (n_samples, 7)
        P_test: 真实的概率分布，形状 (n_samples, 7)
    
    返回:
        包含以下指标的字典：
        - mae: 平均绝对误差（越小越好）
        - rmse: 均方根误差（越小越好）
        - kl: KL散度，衡量分布差异（越小越好）
        - js_mean: Jensen-Shannon散度，对称的分布差异度量（越小越好）
        - cos_sim: 余弦相似度，衡量分布向量的方向相似性（越接近1越好）
        - r2: 决定系数（越接近1越好）
        - max_error: 单个概率值的最大偏差
        - mse: 均方误差（越小越好）
        - tv_distance: 总变差距离（越小越好）
    """
    # 基础误差指标
    mae = np.mean(np.abs(P_pred - P_test))
    mse = np.mean((P_pred - P_test) ** 2)
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(P_pred - P_test))
    
    # 分布差异指标
    eps = 1e-12
    # KL散度: KL(p_true || p_pred)
    kl = np.mean(
        np.sum(P_test * (np.log(P_test + eps) - np.log(P_pred + eps)), axis=1)
    )
    # Jensen-Shannon散度: 对称版本的KL
    js_mean = np.mean([
        jensenshannon(P_test[i], P_pred[i]) for i in range(len(P_test))
    ])
    # 总变差距离 (Total Variation Distance)
    tv_distance = np.mean(np.sum(np.abs(P_pred - P_test), axis=1) / 2.0)
    
    # 相似性指标
    cos_sim = np.mean([
        cosine_similarity([P_test[i]], [P_pred[i]])[0, 0] 
        for i in range(len(P_test))
    ])
    r2 = r2_score(P_test, P_pred)
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mse": float(mse),
        "kl": float(kl),
        "js_mean": float(js_mean),
        "tv_distance": float(tv_distance),
        "cos_sim": float(cos_sim),
        "r2": float(r2),
        "max_error": float(max_error),
    }


def compute_per_bin_metrics(P_pred: np.ndarray, P_test: np.ndarray) -> dict:
    """
    计算每个桶（bin）的性能指标。
    
    参数:
        P_pred: 预测分布
        P_test: 真实分布
    
    返回:
        包含每桶 MAE、RMSE 和偏差的字典
    """
    errors = P_pred - P_test
    mae_per_bin = np.mean(np.abs(errors), axis=0)
    rmse_per_bin = np.sqrt(np.mean(errors ** 2, axis=0))
    bias_per_bin = np.mean(errors, axis=0)
    
    return {
        "mae_per_bin": mae_per_bin.tolist(),
        "rmse_per_bin": rmse_per_bin.tolist(),
        "bias_per_bin": bias_per_bin.tolist(),
    }


def compute_per_sample_metrics(P_pred: np.ndarray, P_test: np.ndarray) -> dict:
    """
    计算每个样本的性能指标。
    
    参数:
        P_pred: 预测分布
        P_test: 真实分布
    
    返回:
        包含每样本 MAE、JS 散度等的字典
    """
    mae_per_sample = np.mean(np.abs(P_pred - P_test), axis=1)
    js_per_sample = np.array([
        jensenshannon(P_test[i], P_pred[i]) for i in range(len(P_test))
    ])
    
    return {
        "mae_per_sample": mae_per_sample,
        "js_per_sample": js_per_sample,
        "mae_mean": float(mae_per_sample.mean()),
        "mae_std": float(mae_per_sample.std()),
        "js_mean": float(js_per_sample.mean()),
        "js_std": float(js_per_sample.std()),
    }
