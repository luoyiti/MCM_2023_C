"""
使用 Dirichlet MoE 概率模型预测单词 EERIE 的成绩分布并作图。

依赖:
- 使用 `task2_distribution_prediction/models/Moe_Softmax_with_probability.py`
  中定义的模型与工具函数 (DirichletMoE, FEATURE_COLS, DIST_COLS 等)
- 训练数据: `data/mcm_processed_data.csv`
- EERIE 特征: `data/eerie.csv`

输出:
- 预测结果 CSV: `results/task2/eerie_distribution_moe_dirichlet.csv`
- 可视化 PNG:   `pictures/task2/eerie_distribution_moe_dirichlet.png`
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler

# 导入 Dirichlet MoE 模型及其工具函数 / 配置
from models.Moe_Softmax_with_probability import (  # type: ignore
    DirichletMoE,
    FEATURE_COLS,
    DIST_COLS,
    DEVICE,
    NUM_EXPERTS,
    HIDDEN_SIZE,
    TOP_K,
    MODEL_PATH,
    SCALER_PATH,
    predict_with_uncertainty,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "task2"
PICTURES_DIR = PROJECT_ROOT / "pictures" / "task2"

PROCESSED_DATA_PATH = DATA_DIR / "mcm_processed_data.csv"
EERIE_DATA_PATH = DATA_DIR / "eerie.csv"

RESULT_CSV = RESULTS_DIR / "eerie_distribution_moe_dirichlet.csv"
RESULT_FIG = PICTURES_DIR / "eerie_distribution_moe_dirichlet.png"


def _load_trained_model_and_scaler() -> tuple[DirichletMoE, StandardScaler]:
    """加载已保存的 Dirichlet MoE 模型和标准化器。"""
    model_path = Path(MODEL_PATH)
    scaler_path = Path(SCALER_PATH)

    if not model_path.exists():
        raise FileNotFoundError(f"找不到已保存的模型: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"找不到已保存的标准化器: {scaler_path}")

    with open(scaler_path, "rb") as f:
        scaler: StandardScaler = pickle.load(f)

    model = DirichletMoE(
        input_size=len(FEATURE_COLS),
        output_size=len(DIST_COLS),
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        k=TOP_K,
        noisy_gating=True,
    ).to(DEVICE)

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler


def _load_eerie_features(train_medians: pd.Series) -> np.ndarray:
    """从 eerie.csv 读取 EERIE 的特征，并用训练集中位数补齐缺失。"""
    df_eerie = pd.read_csv(EERIE_DATA_PATH)

    # 取第一行非空 word 的记录
    if "word" in df_eerie.columns:
        df_eerie = df_eerie[df_eerie["word"].notna()]
    df_eerie = df_eerie.head(1)

    # 确保包含所有 FEATURE_COLS
    for col in FEATURE_COLS:
        if col not in df_eerie.columns:
            df_eerie[col] = train_medians.get(col, 0.0)

    X_eerie = df_eerie[FEATURE_COLS].copy()
    X_eerie = X_eerie.fillna(train_medians)

    return X_eerie.to_numpy(dtype=np.float32)


def _plot_eerie_distribution(
    p: np.ndarray, 
    ci_lower: np.ndarray | None = None,
    ci_upper: np.ndarray | None = None,
    alpha0: float | None = None,
    save_path: Path | None = None
) -> None:
    """绘制 EERIE 的预测分布柱状图（带置信区间）。"""
    categories = [
        "1_try",
        "2_tries",
        "3_tries",
        "4_tries",
        "5_tries",
        "6_tries",
        "7_or_more_tries_x",
    ]

    probs = p.flatten()
    percents = probs * 100.0

    x = np.arange(len(categories))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, percents, color="steelblue", edgecolor="black", alpha=0.8, label="预测均值")
    
    # 如果有置信区间，添加误差线
    if ci_lower is not None and ci_upper is not None:
        ci_lower_flat = ci_lower.flatten() * 100.0
        ci_upper_flat = ci_upper.flatten() * 100.0
        
        # 计算误差（相对于均值的偏移）
        yerr_lower = percents - ci_lower_flat
        yerr_upper = ci_upper_flat - percents
        
        plt.errorbar(
            x, percents, 
            yerr=[yerr_lower, yerr_upper],
            fmt='none', 
            ecolor='darkred', 
            capsize=5, 
            capthick=2,
            linewidth=2,
            label=f'95% 置信区间'
        )
    
    plt.xticks(x, ["1", "2", "3", "4", "5", "6", "7+"], fontsize=11)
    plt.ylabel("概率 (%)", fontsize=12)
    plt.xlabel("尝试次数", fontsize=12)
    
    # 标题包含 alpha0 信息
    if alpha0 is not None:
        title = f"EERIE 预测成绩分布 (Dirichlet MoE)\n置信度参数 α₀ = {alpha0:.1f}"
    else:
        title = "EERIE 预测成绩分布 (Dirichlet MoE)"
    plt.title(title, fontsize=13, fontweight='bold')
    
    plt.grid(axis="y", alpha=0.3)
    plt.ylim(0, max(percents.max() * 1.2, 10))

    # 在柱状图上标注数值
    for xi, yi in zip(x, percents):
        plt.text(xi, yi + max(percents) * 0.02, f"{yi:.1f}%", 
                ha="center", va="bottom", fontsize=9, fontweight='bold')
    
    plt.legend(loc='upper right', fontsize=10)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        print(f"[eerie_moe] 分布图已保存: {save_path}")
    
    plt.close()


def main() -> None:
    print("=" * 70)
    print("使用 Dirichlet MoE 模型预测 EERIE 的分布 (含概率输出)")
    print("=" * 70)

    # 1. 加载已训练的模型与标准化器
    print("\n[1/3] 加载已训练的 Dirichlet MoE 模型...")
    model, scaler = _load_trained_model_and_scaler()

    # 2. 读取 EERIE 特征并进行预测
    print("\n[2/3] 读取 EERIE 特征并预测分布...")

    # 训练集中位数用于补齐缺失
    df_train = pd.read_csv(PROCESSED_DATA_PATH)
    train_medians = df_train[FEATURE_COLS].median(numeric_only=True)

    X_eerie_raw = _load_eerie_features(train_medians)
    X_eerie_s = scaler.transform(X_eerie_raw).astype(np.float32)

    results = predict_with_uncertainty(model, X_eerie_s, n_samples=1000)
    p_hat = results["p_hat"]  # (1, 7)
    alpha0 = results["alpha0"]  # (1,)
    ci_lower = results["ci_lower"]  # (1, 7)
    ci_upper = results["ci_upper"]  # (1, 7)

    # 3. 保存 CSV + 绘图
    print("\n[3/3] 保存结果并绘制分布图...")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    categories = list(DIST_COLS)
    probs = p_hat.flatten()
    percents = probs * 100.0
    ci_lower_pct = ci_lower.flatten() * 100.0
    ci_upper_pct = ci_upper.flatten() * 100.0

    df_out = pd.DataFrame(
        {
            "category": categories,
            "probability": probs,
            "percentage": percents,
            "ci_lower": ci_lower.flatten(),
            "ci_upper": ci_upper.flatten(),
            "ci_lower_pct": ci_lower_pct,
            "ci_upper_pct": ci_upper_pct,
        }
    )
    df_out.to_csv(RESULT_CSV, index=False)
    print(f"[eerie_moe] 预测结果已保存: {RESULT_CSV}")
    print(f"[eerie_moe] 置信度参数 alpha0 = {alpha0[0]:.1f}")

    _plot_eerie_distribution(
        p_hat, 
        ci_lower=ci_lower, 
        ci_upper=ci_upper, 
        alpha0=alpha0[0],
        save_path=RESULT_FIG
    )

    print("\n完成：EERIE 的 Dirichlet MoE 概率分布预测已生成。")


if __name__ == "__main__":
    main()

