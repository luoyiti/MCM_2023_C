"""
MLP + Softmax 分布预测脚本（来源：task2.ipynb 摘录整理）

功能：
1) 读取 data/mcm_processed_data.csv，抽取与 task2.ipynb 相同的特征列与目标分布列。
2) 按 70/15/15 划分训练/验证/测试，支持人数 N 加权的软标签交叉熵。
3) 训练 MLPSoftmaxRegression（单隐层 MLP + softmax 输出 7 维分布），含早停。
4) 在测试集评估（MAE / KL / JS / CosSim / R2 / MaxError），保存预测结果。
5) 借助 util/visualizations.py 生成训练曲线、随机样本对比、误差分析、
   不确定性、综合指标、综合汇总等图表。

使用方式：
    python forcasting/mlp_forcasting.py

输出：
    forcasting/output/ 下的若干 png + csv + txt 报告。
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon

from util.visualizations import (
    plot_training_history,
    plot_random_sample_distributions,
    plot_error_analysis,
    plot_uncertainty,
    plot_performance_metrics,
    plot_comprehensive_summary,
)

# ---------------- 全局配置 ----------------
DATA_PATH = "data/mcm_processed_data.csv"
N_COL = "number_of_reported_results"  # 总人数列，如无则设为 None
OUTPUT_DIR = "forcasting/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 与 task2.ipynb 保持一致的特征列
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

# 训练超参
LR = 1e-2
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 500
PATIENCE = 30
MLP_HIDDEN = 64
MLP_DROPOUT = 0.1
WEIGHT_MODE = "sqrt"  # "sqrt" 或 "log1p"


def set_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLPSoftmaxRegression(nn.Module):
    """单隐层 MLP + softmax 输出 7 维分布"""

    def __init__(self, d_in: int, hidden_dim: int = 32, n_out: int = 7, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        logits = self.fc2(h)
        return F.softmax(logits, dim=1)


# ---------------- 损失函数与权重 ----------------
def make_weights_from_N(N_array: np.ndarray, mode: str = "sqrt") -> np.ndarray:
    """
    将人数 N 映射为权重，避免大样本独占训练。
    mode: "sqrt" 或 "log1p"
    """
    if mode == "sqrt":
        w = np.sqrt(N_array)
    elif mode == "log1p":
        w = np.log1p(N_array)
    else:
        raise ValueError("mode must be 'sqrt' or 'log1p'")
    w = w / (w.mean() + 1e-12)
    return w.astype(np.float32)


def soft_cross_entropy(p_hat: torch.Tensor, p_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p_hat = torch.clamp(p_hat, eps, 1.0)
    return -(p_true * torch.log(p_hat)).sum(dim=1).mean()


def weighted_soft_cross_entropy(
    p_hat: torch.Tensor, p_true: torch.Tensor, w: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    p_hat = torch.clamp(p_hat, eps, 1.0)
    per_sample = -(p_true * torch.log(p_hat)).sum(dim=1)
    return (w * per_sample).mean()


# ---------------- 数据预处理 ----------------
def load_and_split_data():
    """加载数据、归一化分布、划分 train/val/test，并标准化特征。"""
    df = pd.read_csv(DATA_PATH)

    # 特征与目标分布
    X = df[FEATURE_COLS].copy()
    X = X.fillna(X.median(numeric_only=True))

    P = df[DIST_COLS].copy().fillna(0.0)
    if P.to_numpy().max() > 1.5:
        P = P / 100.0
    P = P.clip(lower=0.0)
    row_sum = P.sum(axis=1).replace(0, np.nan)
    P = P.div(row_sum, axis=0).fillna(1.0 / len(DIST_COLS))

    if N_COL is not None and N_COL in df.columns:
        N = df[N_COL].fillna(df[N_COL].median()).clip(lower=1)
        N_np = N.to_numpy().astype(np.float32)
    else:
        N_np = None

    X_np = X.to_numpy().astype(np.float32)
    P_np = P.to_numpy().astype(np.float32)

    # 先切分 train/tmp，再分 val/test
    if N_np is None:
        X_train, X_tmp, P_train, P_tmp = train_test_split(
            X_np, P_np, test_size=0.3, random_state=RANDOM_SEED
        )
        X_val, X_test, P_val, P_test = train_test_split(
            X_tmp, P_tmp, test_size=0.5, random_state=RANDOM_SEED
        )
        N_train = N_val = N_test = None
    else:
        X_train, X_tmp, P_train, P_tmp, N_train, N_tmp = train_test_split(
            X_np, P_np, N_np, test_size=0.3, random_state=RANDOM_SEED
        )
        X_val, X_test, P_val, P_test, N_val, N_test = train_test_split(
            X_tmp, P_tmp, N_tmp, test_size=0.5, random_state=RANDOM_SEED
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_val, X_test, P_train, P_val, P_test, N_train, N_val, N_test


# ---------------- 训练与评估 ----------------
def train_mlp(
    X_train: np.ndarray,
    P_train: np.ndarray,
    X_val: np.ndarray,
    P_val: np.ndarray,
    Wtr: torch.Tensor | None,
    Wva: torch.Tensor | None,
) -> tuple[MLPSoftmaxRegression, dict]:
    model = MLPSoftmaxRegression(
        d_in=X_train.shape[1], hidden_dim=MLP_HIDDEN, n_out=7, dropout=MLP_DROPOUT
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    Xtr = torch.tensor(X_train, device=DEVICE)
    Ptr = torch.tensor(P_train, device=DEVICE)
    Xva = torch.tensor(X_val, device=DEVICE)
    Pva = torch.tensor(P_val, device=DEVICE)

    best_state = None
    best_val_loss = float("inf")
    bad = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        p_hat = model(Xtr)
        if Wtr is None:
            loss = soft_cross_entropy(p_hat, Ptr)
        else:
            loss = weighted_soft_cross_entropy(p_hat, Ptr, Wtr)

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            p_val = model(Xva)
            if Wva is None:
                val_loss = soft_cross_entropy(p_val, Pva).item()
            else:
                val_loss = weighted_soft_cross_entropy(p_val, Pva, Wva).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = model.state_dict()
            bad = 0
        else:
            bad += 1

        if epoch % 50 == 0:
            print(f"[MLP] epoch={epoch:3d} train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

        if bad >= PATIENCE:
            print(f"[MLP] Early stopping at epoch {epoch}.")
            break

    if best_state:
        model.load_state_dict(best_state)

    info = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": len(train_losses) - bad,
        "best_val_loss": best_val_loss,
        "bad": bad,
    }
    return model, info


def evaluate(model: MLPSoftmaxRegression, X_test: np.ndarray, P_test: np.ndarray) -> tuple[np.ndarray, dict]:
    model.eval()
    Xte = torch.tensor(X_test, device=DEVICE)
    with torch.no_grad():
        P_pred = model(Xte).cpu().numpy()

    mae = np.mean(np.abs(P_pred - P_test))
    eps = 1e-12
    kl = np.mean(np.sum(P_test * (np.log(P_test + eps) - np.log(P_pred + eps)), axis=1))
    js_mean = np.mean([jensenshannon(P_test[i], P_pred[i]) for i in range(len(P_test))])
    cos_sim = np.mean([cosine_similarity([P_test[i]], [P_pred[i]])[0, 0] for i in range(len(P_test))])
    r2 = r2_score(P_test, P_pred)
    max_error = np.max(np.abs(P_pred - P_test))

    metrics = {
        "mae": float(mae),
        "kl": float(kl),
        "js_mean": float(js_mean),
        "cos_sim": float(cos_sim),
        "r2": float(r2),
        "max_error": float(max_error),
    }
    print("\n[MLP] 测试集评估")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    return P_pred, metrics


def save_predictions(P_pred: np.ndarray, path: str) -> None:
    df_pred = pd.DataFrame(P_pred, columns=[f"mlp_pred_{c}" for c in DIST_COLS])
    df_pred.to_csv(path, index=False)
    print(f"[MLP] 预测结果已保存: {path}")


def write_report(metrics: dict, info: dict, path: str) -> None:
    lines = {
        "best_epoch": info.get("best_epoch"),
        "best_val_loss": info.get("best_val_loss"),
        "metrics": metrics,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(lines, ensure_ascii=False, indent=2))
    print(f"[MLP] 报告已保存: {path}")


# ---------------- 主流程 ----------------
def main():
    set_seed()
    print(f"设备: {DEVICE}")
    print(f"数据路径: {DATA_PATH}")

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

    Wtr = torch.tensor(make_weights_from_N(N_train, WEIGHT_MODE), device=DEVICE) if N_train is not None else None
    Wva = torch.tensor(make_weights_from_N(N_val, WEIGHT_MODE), device=DEVICE) if N_val is not None else None

    model, info = train_mlp(X_train, P_train, X_val, P_val, Wtr, Wva)
    P_pred, metrics = evaluate(model, X_test, P_test)

    pred_path = os.path.join(OUTPUT_DIR, "mlp_softmax_pred_output.csv")
    save_predictions(P_pred, pred_path)

    # 可视化
    hist_info = plot_training_history(
        train_losses=info["train_losses"],
        val_losses=info["val_losses"],
        bad=info["bad"],
        best_val_loss=info["best_val_loss"],
        save_path=os.path.join(OUTPUT_DIR, "mlp_softmax_training_history.png"),
    )

    sample_indices = plot_random_sample_distributions(
        P_test=P_test,
        P_pred=P_pred,
        sample_size=10,
        save_path=os.path.join(OUTPUT_DIR, "mlp_softmax_distribution_comparison.png"),
    )
    error_stats = plot_error_analysis(
        P_test=P_test,
        P_pred=P_pred,
        save_path=os.path.join(OUTPUT_DIR, "mlp_softmax_error_analysis_per_dimension.png"),
    )
    uncertainty_stats = plot_uncertainty(
        model=model,
        X_samples=X_test[:5],
        P_samples_true=P_test[:5],
        device=DEVICE,
        n_bootstrap=100,
        save_path=os.path.join(OUTPUT_DIR, "mlp_softmax_uncertainty_quantification.png"),
    )
    perf_metrics = plot_performance_metrics(
        P_test=P_test,
        P_pred=P_pred,
        save_path=os.path.join(OUTPUT_DIR, "mlp_softmax_performance_metrics.png"),
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
        save_path=os.path.join(OUTPUT_DIR, "mlp_softmax_comprehensive_summary.png"),
    )

    # 报告
    report_path = os.path.join(OUTPUT_DIR, "mlp_softmax_report.txt")
    metrics_to_write = {**metrics, "best_epoch": hist_info["best_epoch"], "best_val_loss": hist_info["best_val_loss"]}
    write_report(metrics_to_write, info, report_path)

    print("\n完成：MLP + Softmax 分布预测与可视化已输出到 forcasting/output/")
    print(f"随机样本索引（分布对比）：{sample_indices}")
    print(f"不确定性统计：avg_std={uncertainty_stats['avg_std']:.4f}, avg_ci_width={uncertainty_stats['avg_ci_width']:.4f}")


if __name__ == "__main__":
    main()
"""
MLP回归模型：预测autoencoder_value
使用reduced_features_train.csv作为训练集，reduced_features_test.csv作为测试集
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 设置中文字体和图表样式
plt.rcParams['font.family'] = 'Heiti TC'  # 替换为你选择的字体

sns.set_style("whitegrid")
sns.set_palette("husl")

# ==================== 1. 数据加载 ====================
print("=" * 60)
print("MLP回归模型训练与评估")
print("=" * 60)

# 读取训练集和测试集
print("\n[1] 加载数据...")
df_train = pd.read_csv("data/reduced_features_train.csv")
df_test = pd.read_csv("data/reduced_features_test.csv")

print(f"训练集样本数: {len(df_train)}")
print(f"测试集样本数: {len(df_test)}")

# 定义特征列和目标列
feature_cols = [
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
target_col = "autoencoder_value"

# 提取特征和目标
X_train = df_train[feature_cols].copy()
y_train = df_train[target_col].copy()
X_test = df_test[feature_cols].copy()
y_test = df_test[target_col].copy()

# 缺失值处理
print("\n[2] 数据预处理...")
print(f"训练集缺失值: {X_train.isnull().sum().sum()}")
print(f"测试集缺失值: {X_test.isnull().sum().sum()}")

if X_train.isnull().sum().sum() > 0:
    X_train = X_train.fillna(X_train.median(numeric_only=True))
if X_test.isnull().sum().sum() > 0:
    X_test = X_test.fillna(X_train.median(numeric_only=True))  # 用训练集的中位数填充

# ==================== 2. 建立MLP回归流水线 ====================
print("\n[3] 构建MLP回归模型...")

pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        max_iter=5000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,  # 从训练集中划分10%作为验证集
        n_iter_no_change=20,
        tol=1e-4
    ))
])

# ==================== 3. 网格搜索调参 ====================
print("\n[4] 网格搜索最优超参数...")
print("（这可能需要几分钟时间，请耐心等待...）")

param_grid = {
    "mlp__hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)],
    "mlp__alpha": [1e-4, 1e-3, 1e-2],  # L2正则强度
    "mlp__learning_rate_init": [1e-4, 5e-4, 1e-3],
    "mlp__activation": ["relu", "tanh"],
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="r2",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

print(f"\n最优参数: {search.best_params_}")
print(f"最优交叉验证R²: {search.best_score_:.4f}")

best_model = search.best_estimator_

# ==================== 4. 模型评估 ====================
print("\n[5] 模型评估...")

# 在训练集上预测
y_train_pred = best_model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

# 在测试集上预测
y_test_pred = best_model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\n" + "=" * 60)
print("评估结果")
print("=" * 60)
print(f"\n训练集:")
print(f"  R²  = {train_r2:.4f}")
print(f"  RMSE = {train_rmse:.4f}")
print(f"  MAE  = {train_mae:.4f}")

print(f"\n测试集:")
print(f"  R²  = {test_r2:.4f}")
print(f"  RMSE = {test_rmse:.4f}")
print(f"  MAE  = {test_mae:.4f}")

# ==================== 5. 交叉验证评估 ====================
print("\n[6] 5折交叉验证评估...")
cv_scores = []
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kfold.split(X_train):
    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # 使用最优参数训练
    model_cv = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=search.best_params_["mlp__hidden_layer_sizes"],
            alpha=search.best_params_["mlp__alpha"],
            learning_rate_init=search.best_params_["mlp__learning_rate_init"],
            activation=search.best_params_["mlp__activation"],
            max_iter=5000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4
        ))
    ])
    
    model_cv.fit(X_cv_train, y_cv_train)
    y_cv_pred = model_cv.predict(X_cv_val)
    cv_r2 = r2_score(y_cv_val, y_cv_pred)
    cv_scores.append(cv_r2)

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)
print(f"交叉验证 R²: {cv_mean:.4f} ± {cv_std:.4f}")

# ==================== 6. 可视化 ====================
print("\n[7] 生成可视化图表...")

# 创建图表目录
import os
os.makedirs("mlp_results", exist_ok=True)

# 图1: 预测值 vs 真实值散点图（训练集和测试集）
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 训练集
axes[0].scatter(y_train, y_train_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
min_val = min(y_train.min(), y_train_pred.min())
max_val = max(y_train.max(), y_train_pred.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('True autoencoder_value', fontsize=12)
axes[0].set_ylabel('Predicted autoencoder_value', fontsize=12)
axes[0].set_title(f'Training Set\nR² = {train_r2:.4f}, RMSE = {train_rmse:.4f}', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 测试集
axes[1].scatter(y_test, y_test_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5, color='green')
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('True autoencoder_value', fontsize=12)
axes[1].set_ylabel('Predicted autoencoder_value', fontsize=12)
axes[1].set_title(f'Test Set\nR² = {test_r2:.4f}, RMSE = {test_rmse:.4f}', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mlp_results/1_prediction_scatter.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存: mlp_results/1_prediction_scatter.png")
plt.close()

# 图2: 残差图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 训练集残差
train_residuals = y_train - y_train_pred
axes[0].scatter(y_train_pred, train_residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted autoencoder_value', fontsize=12)
axes[0].set_ylabel('Residuals', fontsize=12)
axes[0].set_title('Training Set Residuals', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# 测试集残差
test_residuals = y_test - y_test_pred
axes[1].scatter(y_test_pred, test_residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5, color='green')
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted autoencoder_value', fontsize=12)
axes[1].set_ylabel('Residuals', fontsize=12)
axes[1].set_title('Test Set Residuals', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mlp_results/2_residuals.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存: mlp_results/2_residuals.png")
plt.close()

# 图3: 预测值分布对比
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 训练集
axes[0].hist(y_train, bins=30, alpha=0.6, label='True', color='blue', edgecolor='black')
axes[0].hist(y_train_pred, bins=30, alpha=0.6, label='Predicted', color='orange', edgecolor='black')
axes[0].set_xlabel('autoencoder_value', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Training Set: Distribution Comparison', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# 测试集
axes[1].hist(y_test, bins=20, alpha=0.6, label='True', color='blue', edgecolor='black')
axes[1].hist(y_test_pred, bins=20, alpha=0.6, label='Predicted', color='green', edgecolor='black')
axes[1].set_xlabel('autoencoder_value', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Test Set: Distribution Comparison', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('mlp_results/3_distribution_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存: mlp_results/3_distribution_comparison.png")
plt.close()

# 图4: 指标对比柱状图
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['R²', 'RMSE', 'MAE']
train_values = [train_r2, train_rmse, train_mae]
test_values = [test_r2, test_rmse, test_mae]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, train_values, width, label='Training Set', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, test_values, width, label='Test Set', alpha=0.8, edgecolor='black')

ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Model Performance Metrics Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('mlp_results/4_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存: mlp_results/4_metrics_comparison.png")
plt.close()

# 图5: 特征重要性（通过模型权重分析）
# 注意：MLP的特征重要性不如树模型直观，这里展示输入层到第一隐藏层的权重绝对值均值
scaler = best_model.named_steps['scaler']
mlp = best_model.named_steps['mlp']

# 获取第一层的权重（输入层到第一隐藏层）
if hasattr(mlp, 'coefs_') and len(mlp.coefs_) > 0:
    first_layer_weights = np.abs(mlp.coefs_[0])  # shape: (n_features, n_hidden_neurons)
    feature_importance = np.mean(first_layer_weights, axis=1)  # 对每个特征取平均
    
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = np.argsort(feature_importance)[::-1]
    
    ax.barh(range(len(feature_cols)), feature_importance[indices], edgecolor='black')
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels([feature_cols[i] for i in indices], fontsize=9)
    ax.set_xlabel('Average Absolute Weight', fontsize=12)
    ax.set_title('Feature Importance (First Layer Weights)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('mlp_results/5_feature_importance.png', dpi=300, bbox_inches='tight')
    print("  ✓ 保存: mlp_results/5_feature_importance.png")
    plt.close()

# 图6: 预测误差分布
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 训练集误差
axes[0].hist(train_residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
axes[0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Residuals', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title(f'Training Set Error Distribution\nMean={train_residuals.mean():.4f}, Std={train_residuals.std():.4f}', 
                  fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# 测试集误差
axes[1].hist(test_residuals, bins=20, alpha=0.7, color='green', edgecolor='black')
axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Residuals', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title(f'Test Set Error Distribution\nMean={test_residuals.mean():.4f}, Std={test_residuals.std():.4f}', 
                  fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('mlp_results/6_error_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存: mlp_results/6_error_distribution.png")
plt.close()

# ==================== 7. 生成预测结果文件 ====================
print("\n[8] 保存预测结果...")

# 为训练集添加预测值
df_train_result = df_train.copy()
df_train_result['autoencoder_value_pred'] = y_train_pred
df_train_result['residual'] = train_residuals
df_train_result['abs_error'] = np.abs(train_residuals)

# 为测试集添加预测值
df_test_result = df_test.copy()
df_test_result['autoencoder_value_pred'] = y_test_pred
df_test_result['residual'] = test_residuals
df_test_result['abs_error'] = np.abs(test_residuals)

# 保存结果
df_train_result.to_csv('mlp_results/train_predictions.csv', index=False)
df_test_result.to_csv('mlp_results/test_predictions.csv', index=False)
print("  ✓ 保存: mlp_results/train_predictions.csv")
print("  ✓ 保存: mlp_results/test_predictions.csv")

# ==================== 8. 生成详细报告 ====================
print("\n[9] 生成详细报告...")

report = f"""
{'='*80}
MLP回归模型预测报告
{'='*80}

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

一、数据概况
-----------
训练集样本数: {len(df_train)}
测试集样本数: {len(df_test)}
特征数量: {len(feature_cols)}
目标变量: {target_col}

特征列表:
{chr(10).join([f'  {i+1}. {col}' for i, col in enumerate(feature_cols)])}

二、模型配置
-----------
最优超参数:
{chr(10).join([f'  {k}: {v}' for k, v in search.best_params_.items()])}

网络结构: {search.best_params_['mlp__hidden_layer_sizes']}
激活函数: {search.best_params_['mlp__activation']}
L2正则化系数: {search.best_params_['mlp__alpha']}
学习率: {search.best_params_['mlp__learning_rate_init']}

三、模型性能评估
---------------
训练集性能:
  R²  (决定系数): {train_r2:.6f}
  RMSE (均方根误差): {train_rmse:.6f}
  MAE  (平均绝对误差): {train_mae:.6f}

测试集性能:
  R²  (决定系数): {test_r2:.6f}
  RMSE (均方根误差): {test_rmse:.6f}
  MAE  (平均绝对误差): {test_mae:.6f}

5折交叉验证:
  R²: {cv_mean:.6f} ± {cv_std:.6f}

四、误差分析
-----------
训练集残差统计:
  均值: {train_residuals.mean():.6f}
  标准差: {train_residuals.std():.6f}
  最小值: {train_residuals.min():.6f}
  最大值: {train_residuals.max():.6f}
  中位数: {train_residuals.median():.6f}

测试集残差统计:
  均值: {test_residuals.mean():.6f}
  标准差: {test_residuals.std():.6f}
  最小值: {test_residuals.min():.6f}
  最大值: {test_residuals.max():.6f}
  中位数: {test_residuals.median():.6f}

五、预测值统计
-------------
训练集:
  真实值范围: [{y_train.min():.4f}, {y_train.max():.4f}]
  预测值范围: [{y_train_pred.min():.4f}, {y_train_pred.max():.4f}]

测试集:
  真实值范围: [{y_test.min():.4f}, {y_test.max():.4f}]
  预测值范围: [{y_test_pred.min():.4f}, {y_test_pred.max():.4f}]

六、模型诊断
-----------
过拟合检查:
  训练集R² - 测试集R² = {train_r2 - test_r2:.6f}
  {'  ⚠️  警告: 可能存在过拟合' if (train_r2 - test_r2) > 0.1 else '  ✓ 模型泛化能力良好'}

残差正态性:
  训练集残差偏度: {train_residuals.skew():.4f}
  测试集残差偏度: {test_residuals.skew():.4f}
  {'  ⚠️  警告: 残差分布可能非正态' if abs(train_residuals.skew()) > 1 else '  ✓ 残差分布接近正态'}

七、输出文件
-----------
预测结果:
  - mlp_results/train_predictions.csv (训练集预测结果)
  - mlp_results/test_predictions.csv (测试集预测结果)

可视化图表:
  - mlp_results/1_prediction_scatter.png (预测值散点图)
  - mlp_results/2_residuals.png (残差图)
  - mlp_results/3_distribution_comparison.png (分布对比图)
  - mlp_results/4_metrics_comparison.png (指标对比图)
  - mlp_results/5_feature_importance.png (特征重要性图)
  - mlp_results/6_error_distribution.png (误差分布图)

八、结论与建议
-------------
1. 模型性能: 
   - 测试集R² = {test_r2:.4f}, 表明模型解释了约{test_r2*100:.1f}%的方差
   - RMSE = {test_rmse:.4f}, 平均预测误差为{test_rmse:.4f}

2. 模型质量:
   {'模型表现良好，预测精度较高' if test_r2 > 0.7 else '模型表现一般，可能需要进一步优化' if test_r2 > 0.5 else '模型表现较差，建议检查数据质量或尝试其他模型'}

3. 改进建议:
   - {'考虑增加更多特征或特征工程' if test_r2 < 0.7 else '模型表现良好，可考虑用于实际预测'}
   - {'注意过拟合风险，建议增加正则化强度' if (train_r2 - test_r2) > 0.1 else '模型泛化能力良好'}
   - 可以尝试集成学习方法进一步提升性能

{'='*80}
报告结束
{'='*80}
"""

with open('mlp_results/report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print("\n  ✓ 保存: mlp_results/report.txt")

print("\n" + "=" * 60)
print("所有任务完成！")
print("=" * 60)
print(f"\n所有结果已保存到 mlp_results/ 目录")
print(f"共生成 {len([f for f in os.listdir('mlp_results') if f.endswith('.png')])} 张图表和详细报告")

