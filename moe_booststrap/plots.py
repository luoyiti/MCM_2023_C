"""
可视化模块

包含训练过程、模型性能和专家分析的各种可视化函数。
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from .moe import MoE
from .config import DEVICE, DIST_COLS

# 设置中文字体
plt.rcParams["font.family"] = "Heiti TC"
plt.rcParams["axes.unicode_minus"] = False


def plot_training_history(
    train_losses: list,
    val_losses: list,
    bad: int,
    best_val_loss: float,
    save_path: str = None,
    show: bool = True,
) -> dict:
    """
    绘制训练/验证损失曲线。
    
    参数:
        train_losses: 训练损失历史
        val_losses: 验证损失历史
        bad: 无改善的轮次数
        best_val_loss: 最佳验证损失
        save_path: 保存路径
        show: 是否显示图表
    
    返回:
        包含最佳轮次等信息的字典
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="Train", alpha=0.85)
    ax.plot(val_losses, label="Val", alpha=0.85)
    best_epoch = max(0, len(train_losses) - bad)
    ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.8, label=f"Best@{best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training History (best_val={best_val_loss:.4f})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"[Plot] 训练曲线已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return {"best_epoch": int(best_epoch), "best_val_loss": float(best_val_loss)}


def plot_random_sample_distributions(
    P_test: np.ndarray,
    P_pred: np.ndarray,
    sample_size: int = 10,
    save_path: str = None,
    show: bool = True,
) -> list:
    """
    随机抽样对比真实分布 vs 预测分布。
    
    参数:
        P_test: 真实分布
        P_pred: 预测分布
        sample_size: 抽样数量
        save_path: 保存路径
        show: 是否显示图表
    
    返回:
        抽样的索引列表
    """
    n = len(P_test)
    if n == 0:
        return []
    sample_size = int(min(sample_size, n))
    idx = np.random.choice(n, size=sample_size, replace=False).tolist()

    n_cols = min(5, sample_size)
    n_rows = (sample_size + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.3 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    x = np.arange(len(DIST_COLS))
    width = 0.35
    for k, i in enumerate(idx):
        ax = axes[k]
        ax.bar(x - width / 2, P_test[i], width, label="True", color="steelblue")
        ax.bar(x + width / 2, P_pred[i], width, label="Pred", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in DIST_COLS], fontsize=7)
        ax.set_ylim(0, max(0.35, float(max(P_test[i].max(), P_pred[i].max())) * 1.2))
        ax.set_title(f"Sample {i}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    for k in range(len(idx), len(axes)):
        axes[k].set_visible(False)

    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"[Plot] 随机样本分布对比图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return idx


def plot_error_analysis(
    P_test: np.ndarray, 
    P_pred: np.ndarray, 
    save_path: str = None,
    show: bool = True,
) -> dict:
    """
    误差分析：每桶 MAE + 样本级 MAE 分布。
    
    参数:
        P_test: 真实分布
        P_pred: 预测分布
        save_path: 保存路径
        show: 是否显示图表
    
    返回:
        包含误差统计的字典
    """
    errors = P_pred - P_test
    mae_per_dim = np.mean(np.abs(errors), axis=0)
    mae_per_sample = np.mean(np.abs(errors), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(DIST_COLS))

    axes[0].bar(x, mae_per_dim, color="teal", edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.replace("_", "\n") for c in DIST_COLS], fontsize=7)
    axes[0].set_ylabel("MAE")
    axes[0].set_title("各桶的平均绝对误差 (MAE per Bin)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].hist(mae_per_sample, bins=30, color="slateblue", alpha=0.75, edgecolor="black")
    axes[1].set_xlabel("每样本 MAE")
    axes[1].set_ylabel("样本数量")
    axes[1].set_title("样本级 MAE 分布")
    axes[1].axvline(np.mean(mae_per_sample), color="red", linestyle="--", 
                    label=f"均值={np.mean(mae_per_sample):.4f}")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"[Plot] 误差分析图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "mae_per_dim": mae_per_dim.tolist(), 
        "mae_per_sample": mae_per_sample, 
        "errors": errors
    }


def analyze_expert_usage(
    model: MoE, 
    X_data: np.ndarray, 
    save_path: str = None,
    show: bool = True,
    device: str = None,
) -> dict:
    """
    分析并可视化专家使用率。
    
    参数:
        model: MoE 模型
        X_data: 输入数据
        save_path: 保存路径
        show: 是否显示图表
        device: 计算设备
    
    返回:
        包含专家使用率统计的字典
    """
    if device is None:
        device = DEVICE
        
    model.eval()
    Xte = torch.tensor(X_data, device=device)
    with torch.no_grad():
        gates, _load = model.noisy_top_k_gating(Xte, train=False)
        gates_np = gates.cpu().numpy()

    num_experts = model.num_experts
    expert_usage = (gates_np > 0).mean(axis=0)
    expert_avg_weight = gates_np.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(num_experts)

    axes[0].bar(x, expert_usage, color="steelblue", edgecolor="black")
    axes[0].set_xlabel("Expert")
    axes[0].set_ylabel("Usage Rate")
    axes[0].set_title("专家使用率")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"E{i}" for i in range(num_experts)])
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, expert_avg_weight, color="coral", edgecolor="black")
    axes[1].set_xlabel("Expert")
    axes[1].set_ylabel("Avg Gate Weight")
    axes[1].set_title("平均门控权重")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"E{i}" for i in range(num_experts)])
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"[Plot] 专家使用率图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "expert_usage": expert_usage.tolist(),
        "expert_avg_weight": expert_avg_weight.tolist(),
    }


def compute_expert_outputs(
    model: MoE,
    X_data: np.ndarray,
    device: str = None,
) -> tuple:
    """
    计算各专家的输出和门控权重。
    
    参数:
        model: MoE 模型
        X_data: 输入数据
        device: 计算设备
    
    返回:
        (y_experts, gates, y_mixture) 元组
        - y_experts: shape (num_experts, n, 7)
        - gates: shape (n, num_experts)
        - y_mixture: shape (n, 7)
    """
    if device is None:
        device = DEVICE
        
    model.eval()
    Xte = torch.tensor(X_data, device=device)
    with torch.no_grad():
        # 逐专家前向
        y_exps = []
        for exp in model.experts:
            y_exp = exp(Xte)
            y_exps.append(y_exp)
        y_experts = torch.stack(y_exps, dim=0)

        # 门控权重
        gates, _ = model.noisy_top_k_gating(Xte, train=False)

        # 混合输出
        y_mix = (gates.unsqueeze(-1) * y_experts.permute(1, 0, 2)).sum(dim=1)

    return (
        y_experts.cpu().numpy(),
        gates.cpu().numpy(),
        y_mix.cpu().numpy(),
    )


def plot_expert_gate_heatmap(
    gates: np.ndarray,
    save_path: str = None,
    show: bool = True,
    max_samples: int = 500,
) -> None:
    """
    展示门控权重热力图。
    
    参数:
        gates: 门控权重矩阵
        save_path: 保存路径
        show: 是否显示图表
        max_samples: 最大显示样本数
    """
    n = gates.shape[0]
    idx = np.arange(n)
    if n > max_samples:
        idx = np.random.choice(n, size=max_samples, replace=False)
    G = gates[idx]
    # 按分配专家排序
    assigned = G.argmax(axis=1)
    order = np.argsort(assigned)
    G = G[order]
    
    fig, ax = plt.subplots(figsize=(1.2 * gates.shape[1] + 2, 0.12 * G.shape[0] + 2))
    im = ax.imshow(G, aspect="auto", cmap="viridis")
    ax.set_xlabel("Experts")
    ax.set_ylabel("Samples (sorted)")
    ax.set_xticks(np.arange(gates.shape[1]))
    ax.set_xticklabels([f"E{i}" for i in range(gates.shape[1])])
    ax.set_title("Gate Weights Heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Gate Weight")
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"[Plot] 门控权重热力图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_sample_expert_decomposition(
    P_true: np.ndarray,
    y_experts: np.ndarray,
    gates: np.ndarray,
    y_mix: np.ndarray,
    save_path: str = None,
    show: bool = True,
    sample_size: int = 6,
) -> None:
    """
    随机抽样展示专家分解：True vs Mixture 以及 Top-2 专家的输出。
    
    参数:
        P_true: 真实分布
        y_experts: 各专家输出
        gates: 门控权重
        y_mix: 混合输出
        save_path: 保存路径
        show: 是否显示图表
        sample_size: 抽样数量
    """
    n = P_true.shape[0]
    if n == 0:
        return
    idx = np.random.choice(n, size=min(sample_size, n), replace=False)
    x = np.arange(len(DIST_COLS))
    n_cols = 3
    n_rows = (len(idx) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.6 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    # 获取专家数量
    num_experts = y_experts.shape[0]
    
    for k, i in enumerate(idx):
        ax = axes[k]
        ax.plot(x, y_mix[i], marker="o", linewidth=2, label="Mixture")
        ax.plot(x, P_true[i], linestyle="--", marker="x", label="True")
        
        # 显示top-k个专家（最多显示所有专家，但不超过num_experts）
        g = gates[i]
        # 根据专家数量动态决定显示多少个专家：2个专家显示全部，3+个专家显示top-2
        top_k = num_experts if num_experts == 2 else min(2, num_experts)
        top_experts = np.argsort(-g)[:top_k]
        
        for t in top_experts:
            ax.plot(x, y_experts[t, i], linewidth=1.5, alpha=0.8, 
                   label=f"E{t} (w={g[t]:.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in DIST_COLS], fontsize=8)
        ax.set_ylim(0, max(0.35, float(max(y_mix[i].max(), P_true[i].max())) * 1.2))
        ax.set_title(f"Sample {i}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    
    for k in range(len(idx), len(axes)):
        axes[k].set_visible(False)
    
    # 动态调整标题
    top_k_display = num_experts if num_experts == 2 else min(2, num_experts)
    fig.suptitle(f"Mixture vs True with Top-{top_k_display} Expert Outputs ({num_experts} experts)", 
                fontsize=12)
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"[Plot] 样本分解图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_holdout_bar_with_ci(
    P_mean: np.ndarray,
    P_low: np.ndarray,
    P_high: np.ndarray,
    P_true: np.ndarray = None,
    word: str = "eerie",
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    绘制 Holdout 样本的预测分布（带置信区间）。
    
    参数:
        P_mean: Bootstrap 均值预测
        P_low, P_high: 置信区间下界和上界
        P_true: 真实分布（可选）
        word: 单词名称
        save_path: 保存路径
        show: 是否显示图表
    """
    mean = P_mean.mean(axis=0)
    low = P_low.mean(axis=0)
    high = P_high.mean(axis=0)
    yerr = np.vstack([np.maximum(0, mean - low), np.maximum(0, high - mean)])

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(DIST_COLS))
    ax.bar(x, mean, color="steelblue", alpha=0.8, label="Bootstrap Mean")
    ax.errorbar(x, mean, yerr=yerr, fmt="none", ecolor="black", capsize=4, linewidth=1)
    if P_true is not None:
        true_mean = P_true.mean(axis=0)
        ax.plot(x, true_mean, linestyle="--", marker="x", color="coral", label="True")
    ax.set_xticks(x)
    ax.set_xticklabels(DIST_COLS, rotation=30, ha="right")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, max(0.35, float(high.max()) * 1.25))
    ax.set_title(f"Holdout '{word}' Predicted Distribution (Mean + CI)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"[Plot] Holdout 柱状图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_holdout_violin(
    P_holdout_boot: np.ndarray,
    word: str = "eerie",
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    绘制 Holdout 样本的 Bootstrap 分布小提琴图。
    
    参数:
        P_holdout_boot: Bootstrap 预测结果，形状 (B, n_holdout, 7)
        word: 单词名称
        save_path: 保存路径
        show: 是否显示图表
    """
    B, n_h, d = P_holdout_boot.shape
    vals = []
    for j in range(d):
        vals.append(P_holdout_boot[:, :, j].reshape(B * n_h))

    fig, ax = plt.subplots(figsize=(12, 4.5))
    parts = ax.violinplot(vals, positions=np.arange(d), showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("slateblue")
        pc.set_edgecolor("black")
        pc.set_alpha(0.55)
    ax.set_xticks(np.arange(d))
    ax.set_xticklabels(DIST_COLS, rotation=30, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title(f"Holdout '{word}' Bootstrap Predictive Distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"[Plot] Holdout 小提琴图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_uncertainty(
    P_mean: np.ndarray,
    P_low: np.ndarray,
    P_high: np.ndarray,
    P_test: np.ndarray = None,
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    绘制整体预测不确定性图。
    
    参数:
        P_mean: Bootstrap 均值预测
        P_low, P_high: 置信区间
        P_test: 真实分布（可选）
        save_path: 保存路径
        show: 是否显示图表
    """
    mean_all = P_mean.mean(axis=0)
    low_all = P_low.mean(axis=0)
    high_all = P_high.mean(axis=0)
    yerr_all = np.vstack([
        np.maximum(0, mean_all - low_all), 
        np.maximum(0, high_all - mean_all)
    ])

    x = np.arange(len(DIST_COLS))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(x, mean_all, yerr=yerr_all, fmt="o", capsize=4, label="Bootstrap Mean + CI")
    if P_test is not None:
        true_all = P_test.mean(axis=0)
        ax.plot(x, true_all, linestyle="--", marker="x", color="coral", label="True Mean")
    ax.set_xticks(x)
    ax.set_xticklabels(DIST_COLS, rotation=30, ha="right")
    ax.set_ylim(0.0, max(0.35, float(high_all.max()) * 1.2))
    ax.set_title("Bootstrap Mean Distribution with CI")
    ax.set_ylabel("Probability")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"[Plot] 不确定性图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_grid_search_results(
    df: pd.DataFrame, 
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    可视化网格搜索结果。
    
    参数:
        df: 网格搜索结果 DataFrame
        save_path: 保存路径
        show: 是否显示图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 图1: 不同专家数量下的 MAE 分布
    ax1 = axes[0, 0]
    expert_groups = df.groupby("num_experts")["mae"].apply(list)
    expert_nums = sorted(expert_groups.index.tolist())
    ax1.boxplot([expert_groups.get(e, []) for e in expert_nums], 
                labels=[str(e) for e in expert_nums])
    ax1.set_xlabel("专家数量")
    ax1.set_ylabel("MAE")
    ax1.set_title("不同专家数量下的 MAE 分布")
    ax1.grid(axis="y", alpha=0.3)
    
    # 图2: 不同 TOP-K 下的 MAE 分布
    ax2 = axes[0, 1]
    topk_groups = df.groupby("top_k")["mae"].apply(list)
    topks = sorted(topk_groups.index.tolist())
    ax2.boxplot([topk_groups.get(k, []) for k in topks],
                labels=[str(k) for k in topks])
    ax2.set_xlabel("TOP-K")
    ax2.set_ylabel("MAE")
    ax2.set_title("不同 TOP-K 下的 MAE 分布")
    ax2.grid(axis="y", alpha=0.3)
    
    # 图3: 热力图
    ax3 = axes[1, 0]
    pivot_data = df.groupby(["num_experts", "top_k"])["mae"].min().unstack(fill_value=np.nan)
    im = ax3.imshow(pivot_data.values, aspect="auto", cmap="RdYlGn_r")
    ax3.set_xticks(range(len(pivot_data.columns)))
    ax3.set_xticklabels(pivot_data.columns)
    ax3.set_yticks(range(len(pivot_data.index)))
    ax3.set_yticklabels(pivot_data.index)
    ax3.set_xlabel("TOP-K")
    ax3.set_ylabel("专家数量")
    ax3.set_title("专家数量 vs TOP-K 的最小 MAE")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("MAE")
    
    # 图4: Top 10 配置的性能对比
    ax4 = axes[1, 1]
    top_10 = df.head(10).copy()
    top_10["config"] = top_10.apply(
        lambda r: f"E{int(r['num_experts'])}-K{int(r['top_k'])}-H{int(r['hidden_size'])}", 
        axis=1
    )
    x = np.arange(len(top_10))
    width = 0.35
    ax4.bar(x - width/2, top_10["mae"], width, label="MAE", color="steelblue")
    ax4.bar(x + width/2, top_10["js_mean"], width, label="JS", color="coral")
    ax4.set_xticks(x)
    ax4.set_xticklabels(top_10["config"], rotation=45, ha="right")
    ax4.set_ylabel("指标值")
    ax4.set_title("Top 10 配置的性能对比")
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)
    
    fig.suptitle("网格搜索结果分析", fontsize=14)
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"[Plot] 网格搜索分析图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_expert_specialization_analysis(
    y_experts: np.ndarray,
    y_mix: np.ndarray,
    P_test: np.ndarray,
    dist_cols: list = None,
    save_path: str = None,
    show: bool = True,
) -> dict:
    """
    分析并可视化专家专业化程度，放大不同专家之间的差异。
    
    使用多种可视化方法突出专家的差异性和专业化：
    1. 偏差热力图：显示每个专家与混合预测的差异
    2. 专家特征雷达图：展示每个专家在各个桶上的相对强度
    3. 预测范围分析：显示专家预测的分散程度
    
    参数:
        y_experts: 各专家的输出，形状 (num_experts, n_samples, n_bins)
        y_mix: 混合输出，形状 (n_samples, n_bins)
        P_test: 真实分布，形状 (n_samples, n_bins)
        dist_cols: 分布列名（默认使用配置中的 DIST_COLS）
        save_path: 保存路径
        show: 是否显示图表
    
    返回:
        包含分析统计的字典
    """
    if dist_cols is None:
        dist_cols = DIST_COLS
    
    num_experts, n_samples, n_bins = y_experts.shape
    
    # 计算每个专家的平均预测
    expert_means = y_experts.mean(axis=1)  # (num_experts, n_bins)
    mix_mean = y_mix.mean(axis=0)  # (n_bins,)
    true_mean = P_test.mean(axis=0)  # (n_bins,)
    
    # 计算偏差：专家预测 - 混合预测
    deviations = expert_means - mix_mean[np.newaxis, :]  # (num_experts, n_bins)
    
    # 计算每个专家与真实分布的差异
    expert_errors = expert_means - true_mean[np.newaxis, :]  # (num_experts, n_bins)
    
    # 计算预测范围（最大-最小）
    pred_max = y_experts.max(axis=0)  # (n_samples, n_bins)
    pred_min = y_experts.min(axis=0)  # (n_samples, n_bins)
    pred_range = pred_max - pred_min
    avg_range = pred_range.mean(axis=0)  # (n_bins,)
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # ===== 子图1: 专家偏差热力图 =====
    ax1 = fig.add_subplot(gs[0, :])
    im1 = ax1.imshow(deviations, aspect='auto', cmap='RdBu_r', 
                     vmin=-np.abs(deviations).max(), vmax=np.abs(deviations).max())
    ax1.set_xticks(range(n_bins))
    ax1.set_xticklabels(dist_cols, rotation=30, ha='right')
    ax1.set_yticks(range(num_experts))
    ax1.set_yticklabels([f'Expert {i}' for i in range(num_experts)])
    ax1.set_title('专家预测偏差热力图 (相对于混合预测)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('尝试次数分布', fontsize=11)
    ax1.set_ylabel('专家', fontsize=11)
    
    # 添加数值标注
    for i in range(num_experts):
        for j in range(n_bins):
            ax1.text(j, i, f'{deviations[i, j]:.3f}',
                    ha="center", va="center", color="black", fontsize=9)
    
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.02, pad=0.02)
    cbar1.set_label('偏差值 (专家 - 混合)', fontsize=10)
    
    # ===== 子图2: 专家与真实值的误差 =====
    ax2 = fig.add_subplot(gs[1, 0])
    x = np.arange(n_bins)
    width = 0.12
    colors = plt.cm.get_cmap('Set2')(np.linspace(0, 1, num_experts))
    
    for i in range(num_experts):
        offset = (i - num_experts/2 + 0.5) * width
        ax2.bar(x + offset, expert_errors[i], width, 
               label=f'Expert {i}', color=colors[i], alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dist_cols, rotation=30, ha='right')
    ax2.set_ylabel('预测误差 (专家 - 真实)', fontsize=11)
    ax2.set_xlabel('尝试次数', fontsize=11)
    ax2.set_title('各专家预测误差对比', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9, ncol=2)
    ax2.grid(axis='y', alpha=0.3)
    
    # ===== 子图3: 专家预测范围分析 =====
    ax3 = fig.add_subplot(gs[1, 1])
    
    for i in range(num_experts):
        ax3.plot(x, expert_means[i], marker='o', linewidth=2.5, 
                label=f'Expert {i}', color=colors[i], alpha=0.85)
    
    ax3.plot(x, mix_mean, marker='s', linewidth=3, 
            label='Mixture', color='black', linestyle='--', alpha=0.9)
    ax3.plot(x, true_mean, marker='^', linewidth=3, 
            label='True', color='red', linestyle=':', alpha=0.9)
    
    # 添加阴影区域显示预测范围
    ax3.fill_between(x, pred_min.mean(axis=0), pred_max.mean(axis=0), 
                     alpha=0.15, color='gray', label='专家预测范围')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(dist_cols, rotation=30, ha='right')
    ax3.set_ylabel('平均概率', fontsize=11)
    ax3.set_xlabel('尝试次数', fontsize=11)
    ax3.set_title('专家预测对比（增强对比度）', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    # ===== 子图4: 归一化相对差异 =====
    ax4 = fig.add_subplot(gs[2, 0])
    
    # 计算每个专家相对于平均值的百分比差异
    overall_mean = expert_means.mean(axis=0)  # 所有专家的平均
    relative_diff = ((expert_means - overall_mean[np.newaxis, :]) / 
                    (overall_mean[np.newaxis, :] + 1e-8) * 100)  # 百分比
    
    im4 = ax4.imshow(relative_diff, aspect='auto', cmap='PuOr', 
                     vmin=-np.abs(relative_diff).max(), vmax=np.abs(relative_diff).max())
    ax4.set_xticks(range(n_bins))
    ax4.set_xticklabels(dist_cols, rotation=30, ha='right')
    ax4.set_yticks(range(num_experts))
    ax4.set_yticklabels([f'Expert {i}' for i in range(num_experts)])
    ax4.set_title('归一化相对差异 (%)', fontsize=13, fontweight='bold')
    ax4.set_xlabel('尝试次数分布', fontsize=11)
    ax4.set_ylabel('专家', fontsize=11)
    
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label('相对差异 (%)', fontsize=10)
    
    # ===== 子图5: 专家分散度分析 =====
    ax5 = fig.add_subplot(gs[2, 1])
    
    # 计算每个桶上专家预测的标准差
    expert_std = y_experts.std(axis=0).mean(axis=0)  # (n_bins,)
    
    ax5.bar(x, avg_range, width=0.35, label='预测范围 (Max-Min)', 
           color='steelblue', alpha=0.7)
    ax5_twin = ax5.twinx()
    ax5_twin.plot(x, expert_std, marker='o', linewidth=2.5, 
                 label='预测标准差', color='coral', markersize=8)
    
    ax5.set_xticks(x)
    ax5.set_xticklabels(dist_cols, rotation=30, ha='right')
    ax5.set_ylabel('平均预测范围', fontsize=11, color='steelblue')
    ax5_twin.set_ylabel('预测标准差', fontsize=11, color='coral')
    ax5.set_xlabel('尝试次数', fontsize=11)
    ax5.set_title('专家预测分散度分析', fontsize=13, fontweight='bold')
    ax5.tick_params(axis='y', labelcolor='steelblue')
    ax5_twin.tick_params(axis='y', labelcolor='coral')
    ax5.grid(axis='y', alpha=0.3)
    
    # 合并图例
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    fig.suptitle('专家专业化与差异性分析', fontsize=15, fontweight='bold', y=0.995)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Plot] 专家专业化分析图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    # 返回统计信息
    return {
        "expert_means": expert_means,
        "deviations": deviations,
        "expert_errors": expert_errors,
        "avg_range": avg_range,
        "expert_std": expert_std,
        "relative_diff": relative_diff,
    }


def plot_expert_parallel_coordinates(
    y_experts: np.ndarray,
    y_mix: np.ndarray,
    P_test: np.ndarray,
    dist_cols: list = None,
    save_path: str = None,
    show: bool = True,
) -> dict:
    """
    使用真正的平行坐标图展示专家预测的差异性。
    
    在平行坐标图中，每个尝试次数是一个独立的垂直坐标轴，
    每条折线代表一个专家，线的高度表示该专家在该尝试次数上的相对排名（1-3）。
    
    参数:
        y_experts: 各专家的输出，形状 (num_experts, n_samples, n_bins)
        y_mix: 混合输出，形状 (n_samples, n_bins)
        P_test: 真实分布，形状 (n_samples, n_bins)
        dist_cols: 分布列名（默认使用配置中的 DIST_COLS）
        save_path: 保存路径
        show: 是否显示图表
    
    返回:
        包含专家统计信息的字典
    """
    if dist_cols is None:
        dist_cols = DIST_COLS
    
    num_experts, n_samples, n_bins = y_experts.shape
    
    # 计算每个专家的平均预测
    expert_means = y_experts.mean(axis=1)  # (num_experts, n_bins)
    mix_mean = y_mix.mean(axis=0)  # (n_bins,)
    true_mean = P_test.mean(axis=0)  # (n_bins,)
    
    # 计算每个尝试次数上的相对排名
    # expert_ranks[expert_idx, bin_idx] 表示该专家在该尝试次数上的排名（1=最高，num_experts=最低）
    expert_ranks = np.zeros_like(expert_means)
    for bin_idx in range(n_bins):
        # 获取该尝试次数上所有专家的平均值
        values = expert_means[:, bin_idx]
        # argsort 返回从小到大的索引，我们需要从大到小（排名1是最大值）
        sorted_indices = np.argsort(-values)  # 降序
        # 为每个专家分配排名（1, 2, 3, ...）
        for rank, expert_idx in enumerate(sorted_indices):
            expert_ranks[expert_idx, bin_idx] = rank + 1
    
    # 创建图形 - 使用更宽的尺寸以容纳平行坐标轴
    fig, axes = plt.subplots(2, 1, figsize=(16, 11))
    
    # 设置颜色 - 使用不同的colormap以支持更多专家
    if num_experts <= 8:
        colors = plt.cm.get_cmap('Set2')(np.linspace(0, 1, num_experts))
    else:
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, num_experts))
    
    # ===== 子图1: 排名平行坐标图 =====
    ax1 = axes[0]
    
    # 计算每个轴的位置
    axis_positions = np.arange(n_bins)
    
    # 排名范围动态调整：0.5 到 num_experts + 0.5
    y_min, y_max = 0.5, num_experts + 0.5
    
    # 绘制每个垂直坐标轴
    for i, pos in enumerate(axis_positions):
        ax1.plot([pos, pos], [y_min, y_max], 
                'k-', linewidth=2, alpha=0.5, zorder=1)
        # 添加轴标签
        ax1.text(pos, y_max + 0.3, dist_cols[i], 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 绘制每个专家的排名折线
    for expert_idx in range(num_experts):
        ranks = expert_ranks[expert_idx]
        ax1.plot(axis_positions, ranks, marker='o', linewidth=2.5, 
                label=f'Expert {expert_idx}', color=colors[expert_idx], 
                alpha=0.8, markersize=7, zorder=3)
    
    ax1.set_xlim(-0.5, n_bins - 0.5)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xticks([])  # 隐藏 x 轴刻度
    
    # 动态设置y轴刻度标签
    rank_ticks = list(range(1, num_experts + 1))
    rank_labels = [f'Rank {i}' + (' (最高)' if i == 1 else ' (最低)' if i == num_experts else '') 
                   for i in rank_ticks]
    ax1.set_yticks(rank_ticks)
    ax1.set_yticklabels(rank_labels)
    
    ax1.set_ylabel('相对排名', fontsize=13, fontweight='bold')
    ax1.set_title(f'专家预测的平行坐标图 - 相对排名（{num_experts}个专家，数值越小排名越高）', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # 动态调整图例列数
    legend_ncol = min(num_experts, 4)
    ax1.legend(loc='upper right', fontsize=10, ncol=legend_ncol, framealpha=0.95)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()  # 反转y轴，使排名1在顶部
    
    # ===== 子图2: 绝对值平行坐标图（保留原始信息作为对比）=====
    ax2 = axes[1]
    
    # 找出数据的全局最小值和最大值，用于统一所有轴的范围
    all_values = np.vstack([expert_means, mix_mean, true_mean])
    y_abs_min, y_abs_max = all_values.min(), all_values.max()
    y_abs_padding = (y_abs_max - y_abs_min) * 0.1
    
    # 绘制每个垂直坐标轴
    for i, pos in enumerate(axis_positions):
        ax2.plot([pos, pos], [y_abs_min - y_abs_padding, y_abs_max + y_abs_padding], 
                'k-', linewidth=2, alpha=0.5, zorder=1)
        # 添加轴标签
        ax2.text(pos, y_abs_max + y_abs_padding * 1.3, dist_cols[i], 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 绘制每个专家的平均值折线
    for expert_idx in range(num_experts):
        values = expert_means[expert_idx]
        ax2.plot(axis_positions, values, marker='o', linewidth=2.5, 
                label=f'Expert {expert_idx}', color=colors[expert_idx], 
                alpha=0.8, markersize=7, zorder=3)
    
    # 绘制混合预测和真实值
    ax2.plot(axis_positions, mix_mean, marker='s', linewidth=3, 
            label='Mixture', color='black', linestyle='--', 
            alpha=0.9, markersize=8, zorder=4)
    ax2.plot(axis_positions, true_mean, marker='^', linewidth=3, 
            label='True', color='red', linestyle=':', 
            alpha=0.9, markersize=8, zorder=4)
    
    ax2.set_xlim(-0.5, n_bins - 0.5)
    ax2.set_ylim(y_abs_min - y_abs_padding, y_abs_max + y_abs_padding * 1.5)
    ax2.set_xticks([])  # 隐藏 x 轴刻度
    ax2.set_ylabel('平均概率', fontsize=13, fontweight='bold')
    ax2.set_xlabel('尝试次数分布', fontsize=13, fontweight='bold')
    ax2.set_title('专家预测的平行坐标图 - 绝对值（参考）', fontsize=15, fontweight='bold', pad=20)
    
    # 动态调整图例列数（包括Mixture和True，所以+2）
    legend_ncol = min(num_experts + 2, 5)
    ax2.legend(loc='upper right', fontsize=10, ncol=legend_ncol, framealpha=0.95)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Plot] 专家平行坐标图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    # 计算专家差异度量（基于排名）
    # 1. 排名差异：计算每个专家与其他专家的排名欧氏距离
    rank_distances = np.zeros((num_experts, num_experts))
    for i in range(num_experts):
        for j in range(num_experts):
            rank_distances[i, j] = np.linalg.norm(expert_ranks[i] - expert_ranks[j])
    
    # 2. 峰值位置（基于原始平均值）
    peak_positions = expert_means.argmax(axis=1)
    
    # 3. 排名稳定性（排名的标准差，值越小表示该专家在各尝试次数上的排名越稳定）
    rank_stability = expert_ranks.std(axis=1)
    
    # 4. 平均排名（值越小表示该专家整体表现越好）
    avg_rank = expert_ranks.mean(axis=1)
    
    return {
        "expert_means": expert_means,
        "expert_ranks": expert_ranks,
        "rank_distances": rank_distances,
        "peak_positions": peak_positions,
        "rank_stability": rank_stability,
        "avg_rank": avg_rank,
        "mix_mean": mix_mean,
        "true_mean": true_mean,
    }
