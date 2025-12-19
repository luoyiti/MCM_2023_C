"""
工具模块

包含保存、加载、报告生成等辅助函数。
"""

import os
import numpy as np
import pandas as pd


def save_predictions(
    df: pd.DataFrame,
    P_pred: np.ndarray,
    dist_cols: list,
    save_path: str,
    extra_cols: dict = None,
) -> pd.DataFrame:
    """
    保存预测结果到 CSV。
    
    参数:
        df: 原始数据 DataFrame
        P_pred: 预测分布
        dist_cols: 分布列名
        save_path: 保存路径
        extra_cols: 额外添加的列
    
    返回:
        包含预测结果的 DataFrame
    """
    result = df.copy()
    for i, col in enumerate(dist_cols):
        result[f"pred_{col}"] = P_pred[:, i]
    
    if extra_cols:
        for k, v in extra_cols.items():
            result[k] = v
    
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    result.to_csv(save_path, index=False)
    print(f"[Save] 预测结果已保存: {save_path}")
    
    return result


def save_bootstrap_predictions(
    df: pd.DataFrame,
    P_mean: np.ndarray,
    P_low: np.ndarray,
    P_high: np.ndarray,
    dist_cols: list,
    save_path: str,
    ci: float = 0.95,
) -> pd.DataFrame:
    """
    保存 Bootstrap 预测结果（均值 + 置信区间）。
    
    参数:
        df: 原始数据 DataFrame
        P_mean: Bootstrap 均值预测
        P_low, P_high: 置信区间
        dist_cols: 分布列名
        save_path: 保存路径
        ci: 置信区间百分比
    
    返回:
        包含预测结果的 DataFrame
    """
    result = df.copy()
    for i, col in enumerate(dist_cols):
        result[f"mean_{col}"] = P_mean[:, i]
        result[f"low_{col}"] = P_low[:, i]
        result[f"high_{col}"] = P_high[:, i]
    
    # 添加汇总统计
    result["ci_width_mean"] = (P_high - P_low).mean(axis=1)
    result["pred_entropy"] = -np.sum(
        P_mean * np.log(np.clip(P_mean, 1e-12, 1.0)), axis=1
    )
    
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    result.to_csv(save_path, index=False)
    print(f"[Save] Bootstrap 预测结果已保存: {save_path} (CI={ci:.0%})")
    
    return result


def save_holdout_predictions_with_ci(
    df_holdout: pd.DataFrame,
    P_mean: np.ndarray,
    P_low: np.ndarray,
    P_high: np.ndarray,
    dist_cols: list,
    save_path: str,
) -> pd.DataFrame:
    """
    保存 Holdout 样本的预测结果（含置信区间）。
    
    参数:
        df_holdout: Holdout 数据 DataFrame
        P_mean: Bootstrap 均值预测
        P_low, P_high: 置信区间
        dist_cols: 分布列名
        save_path: 保存路径
    
    返回:
        包含预测结果的 DataFrame
    """
    result = df_holdout.copy()
    for i, col in enumerate(dist_cols):
        result[f"pred_mean_{col}"] = P_mean[:, i]
        result[f"pred_low_{col}"] = P_low[:, i]
        result[f"pred_high_{col}"] = P_high[:, i]
    
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    result.to_csv(save_path, index=False)
    print(f"[Save] Holdout 预测结果已保存: {save_path}")
    
    return result


def save_uncertainty_arrays(
    P_mean: np.ndarray,
    P_low: np.ndarray,
    P_high: np.ndarray,
    save_dir: str,
    prefix: str = "bootstrap",
) -> dict:
    """
    保存不确定性数组到 NPZ 文件。
    
    参数:
        P_mean: Bootstrap 均值预测
        P_low, P_high: 置信区间
        save_dir: 保存目录
        prefix: 文件名前缀
    
    返回:
        包含保存路径的字典
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{prefix}_uncertainty.npz")
    np.savez(
        save_path,
        P_mean=P_mean,
        P_low=P_low,
        P_high=P_high,
    )
    print(f"[Save] 不确定性数组已保存: {save_path}")
    
    return {"path": save_path}


def load_uncertainty_arrays(load_path: str) -> dict:
    """
    加载不确定性数组。
    
    参数:
        load_path: NPZ 文件路径
    
    返回:
        包含 P_mean, P_low, P_high 的字典
    """
    data = np.load(load_path)
    return {
        "P_mean": data["P_mean"],
        "P_low": data["P_low"],
        "P_high": data["P_high"],
    }


def write_bootstrap_report(
    metrics: dict,
    save_path: str,
    model_config: dict = None,
    data_info: dict = None,
) -> str:
    """
    生成 Bootstrap 评估报告。
    
    参数:
        metrics: 评估指标字典
        save_path: 保存路径
        model_config: 模型配置信息
        data_info: 数据信息
    
    返回:
        报告内容字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Bootstrap MoE 模型评估报告")
    lines.append("=" * 60)
    lines.append("")
    
    if model_config:
        lines.append("【模型配置】")
        for k, v in model_config.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    
    if data_info:
        lines.append("【数据信息】")
        for k, v in data_info.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    
    lines.append("【评估指标】")
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.6f}")
        else:
            lines.append(f"  {k}: {v}")
    lines.append("")
    
    lines.append("=" * 60)
    
    report = "\n".join(lines)
    
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Report] 评估报告已保存: {save_path}")
    
    return report


def generate_summary_report(
    train_info: dict,
    test_metrics: dict,
    bootstrap_metrics: dict = None,
    save_path: str = None,
) -> str:
    """
    生成综合训练总结报告。
    
    参数:
        train_info: 训练信息（best_epoch, best_val_loss 等）
        test_metrics: 测试集指标
        bootstrap_metrics: Bootstrap 指标（可选）
        save_path: 保存路径
    
    返回:
        报告内容字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("MoE 模型训练总结报告")
    lines.append("=" * 60)
    lines.append("")
    
    lines.append("【训练信息】")
    for k, v in train_info.items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.6f}")
        else:
            lines.append(f"  {k}: {v}")
    lines.append("")
    
    lines.append("【测试集指标】")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.6f}")
        elif isinstance(v, np.ndarray):
            lines.append(f"  {k}: {v.mean():.6f} (mean)")
        else:
            lines.append(f"  {k}: {v}")
    lines.append("")
    
    if bootstrap_metrics:
        lines.append("【Bootstrap 指标】")
        for k, v in bootstrap_metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.6f}")
            elif isinstance(v, np.ndarray):
                lines.append(f"  {k}: {v.mean():.6f} (mean)")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")
    
    lines.append("=" * 60)
    
    report = "\n".join(lines)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[Report] 训练总结已保存: {save_path}")
    
    return report


def format_metrics_table(metrics: dict, title: str = "指标汇总") -> str:
    """
    将指标字典格式化为表格字符串。
    
    参数:
        metrics: 指标字典
        title: 表格标题
    
    返回:
        格式化的表格字符串
    """
    max_key_len = max(len(str(k)) for k in metrics.keys())
    lines = []
    lines.append(f"\n{title}")
    lines.append("-" * (max_key_len + 20))
    
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"{k:<{max_key_len}} : {v:>12.6f}")
        elif isinstance(v, int):
            lines.append(f"{k:<{max_key_len}} : {v:>12d}")
        elif isinstance(v, np.ndarray):
            lines.append(f"{k:<{max_key_len}} : {v.mean():>12.6f} (mean)")
        else:
            lines.append(f"{k:<{max_key_len}} : {str(v):>12}")
    
    lines.append("-" * (max_key_len + 20))
    
    return "\n".join(lines)
