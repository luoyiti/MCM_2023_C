"""
MoE (Mixture of Experts) + MLP + Dirichlet 概率分布预测脚本

================================================================================
概述：
    本脚本在 MoE + Softmax 模型基础上，引入 Dirichlet 分布来建模不确定性。
    模型额外预测一个集中度参数 alpha0，用 Dirichlet NLL 训练。
    
    推理时输出:
    1. p̂ (均值分布): Softmax 输出的 7 维概率分布
    2. alpha0 (置信度): 集中度参数，值越大表示模型越确定
    3. 置信区间: 通过 Dirichlet 采样得到的各分量的置信区间

模型架构：
    输入 (55维特征) → 门控网络 → 选择 Top-K 专家 → 专家MLP → 
    → [Softmax → p̂ (均值)] + [独立头 → alpha0 (集中度)]
    → Dirichlet 参数 α = alpha0 * p̂
    
损失函数:
    Dirichlet NLL: -log Dir(p_true | α)
    
输出目录：
    moe_dirichlet_output/ - 包含预测结果CSV、可视化图表、JSON报告等

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
from scipy.special import gammaln
import argparse

# 设置中文字体
plt.rcParams['font.family'] = 'Heiti TC'

# ---------------- 全局配置 ----------------
DATA_PATH = "./data/mcm_processed_data.csv"
N_COL = "number_of_reported_results"
OUTPUT_DIR = "moe_dirichlet_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 与 Moe_Softmax.py 保持一致的特征列
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
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 500
PATIENCE = 60
WEIGHT_MODE = "log1p"

# ================================================================================
# MoE 超参数配置
# ================================================================================
NUM_EXPERTS = 4
HIDDEN_SIZE = 256
TOP_K = 1
AUX_COEF = 5e-4

# ================================================================================
# Dirichlet 相关超参数
# ================================================================================
ALPHA0_MIN = 1.0      # alpha0 最小值
ALPHA0_MAX = 500.0    # alpha0 最大值，避免数值不稳定
ALPHA0_INIT = 10.0    # alpha0 初始偏置值
N_SAMPLES_CI = 1000   # 置信区间采样次数
CI_LEVEL = 0.95       # 置信水平


def set_seed(seed: int = RANDOM_SEED) -> None:
    """设置随机种子，确保实验可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================================================================================
# Dirichlet 专家网络
# ================================================================================
class DirichletExpert(nn.Module):
    """
    单个专家网络：MLP + Softmax (for p̂) + 独立头 (for alpha0)
    
    输出:
        p_hat: 均值分布 (softmax输出)
        alpha0: 集中度参数 (标量)
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # 均值分布头 (softmax)
        self.fc_p = nn.Linear(hidden_size, output_size)
        
        # 集中度头 (标量)
        self.fc_alpha0 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        
        # 均值分布
        p_hat = F.softmax(self.fc_p(h), dim=-1)
        
        # 集中度: 使用 softplus 确保正值，加偏置使初始值较大，并限制范围
        # softplus(x) ≈ x when x > 5, so bias helps start with reasonable alpha0
        alpha0_raw = F.softplus(self.fc_alpha0(h)) + ALPHA0_INIT
        alpha0 = torch.clamp(alpha0_raw, min=ALPHA0_MIN, max=ALPHA0_MAX)
        
        return p_hat, alpha0.squeeze(-1)


class DirichletMoE(nn.Module):
    """
    Mixture of Experts with Dirichlet output
    
    输出:
        p_hat: 加权组合后的均值分布
        alpha0: 加权组合后的集中度参数
        aux_loss: 负载平衡辅助损失
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_experts: int,
        hidden_size: int,
        k: int = 1,
        noisy_gating: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating
        self.output_size = output_size
        
        # 门控网络
        self.gate = nn.Linear(input_size, num_experts)
        self.noise_linear = nn.Linear(input_size, num_experts) if noisy_gating else None
        
        # 专家网络
        self.experts = nn.ModuleList([
            DirichletExpert(input_size, hidden_size, output_size)
            for _ in range(num_experts)
        ])
        
    def noisy_top_k_gating(self, x, train: bool = True):
        """带噪声的 Top-K 门控"""
        clean_logits = self.gate(x)
        
        if self.noisy_gating and train:
            noise = torch.randn_like(clean_logits) * F.softplus(self.noise_linear(x))
            noisy_logits = clean_logits + noise
        else:
            noisy_logits = clean_logits
            
        # Top-K 选择
        top_k_logits, top_k_indices = noisy_logits.topk(self.k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # 创建稀疏门控矩阵
        gates = torch.zeros_like(noisy_logits)
        gates.scatter_(1, top_k_indices, top_k_gates)
        
        # 计算负载平衡辅助损失
        importance = gates.sum(0)
        load = (gates > 0).float().sum(0)
        aux_loss = (importance.std() / (importance.mean() + 1e-8) +
                    load.std() / (load.mean() + 1e-8))
        
        return gates, aux_loss
    
    def forward(self, x, train: bool = True):
        gates, aux_loss = self.noisy_top_k_gating(x, train)
        
        # 收集专家输出
        expert_p_hats = []
        expert_alpha0s = []
        
        for expert in self.experts:
            p_hat, alpha0 = expert(x)
            expert_p_hats.append(p_hat)
            expert_alpha0s.append(alpha0)
        
        # Stack: (batch, num_experts, output_size) 和 (batch, num_experts)
        expert_p_hats = torch.stack(expert_p_hats, dim=1)
        expert_alpha0s = torch.stack(expert_alpha0s, dim=1)
        
        # 加权组合
        gates_expanded = gates.unsqueeze(-1)  # (batch, num_experts, 1)
        p_hat = (gates_expanded * expert_p_hats).sum(dim=1)
        alpha0 = (gates * expert_alpha0s).sum(dim=1)
        
        # 确保 p_hat 归一化
        p_hat = p_hat / (p_hat.sum(dim=-1, keepdim=True) + 1e-8)
        
        return p_hat, alpha0, aux_loss


# ================================================================================
# Dirichlet NLL 损失
# ================================================================================
def dirichlet_nll(p_true: torch.Tensor, p_hat: torch.Tensor, alpha0: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    计算 Dirichlet 负对数似然损失
    
    参数:
        p_true: 真实分布 (batch, K)
        p_hat: 预测的均值分布 (batch, K)
        alpha0: 集中度参数 (batch,)
        eps: 数值稳定性常数
    
    返回:
        平均 NLL 损失
    
    公式:
        α = alpha0 * p̂
        NLL = -log Dir(p_true | α)
            = -log Γ(alpha0) + Σ log Γ(αᵢ) - Σ (αᵢ - 1) log(p_true_i)
    """
    # 计算 Dirichlet 参数 α = alpha0 * p̂
    alpha = alpha0.unsqueeze(-1) * p_hat + eps  # (batch, K)
    alpha0_expanded = alpha0 + eps * p_hat.shape[-1]  # (batch,)
    
    # 避免 p_true 中有 0
    p_true_safe = torch.clamp(p_true, min=eps)
    
    # Dirichlet NLL
    # log Γ(alpha0) - Σ log Γ(αᵢ) + Σ (αᵢ - 1) log(p_true_i)
    log_gamma_sum = torch.lgamma(alpha0_expanded)
    log_gamma_parts = torch.lgamma(alpha).sum(dim=-1)
    log_prob = ((alpha - 1) * torch.log(p_true_safe)).sum(dim=-1)
    
    nll = -log_gamma_sum + log_gamma_parts - log_prob
    
    return nll.mean()


def weighted_dirichlet_nll(
    p_true: torch.Tensor,
    p_hat: torch.Tensor,
    alpha0: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """加权 Dirichlet NLL"""
    alpha = alpha0.unsqueeze(-1) * p_hat + eps
    alpha0_expanded = alpha0 + eps * p_hat.shape[-1]
    p_true_safe = torch.clamp(p_true, min=eps)
    
    log_gamma_sum = torch.lgamma(alpha0_expanded)
    log_gamma_parts = torch.lgamma(alpha).sum(dim=-1)
    log_prob = ((alpha - 1) * torch.log(p_true_safe)).sum(dim=-1)
    
    nll = -log_gamma_sum + log_gamma_parts - log_prob
    
    return (w * nll).mean()


# ================================================================================
# 数据加载与预处理
# ================================================================================
def make_weights_from_N(N_array: np.ndarray, mode: str = "sqrt") -> np.ndarray:
    """根据样本人数计算权重"""
    if mode == "sqrt":
        w = np.sqrt(N_array)
    elif mode == "log1p":
        w = np.log1p(N_array)
    else:
        raise ValueError("mode must be 'sqrt' or 'log1p'")
    w = w / (w.mean() + 1e-12)
    return w.astype(np.float32)


def load_and_split_data():
    """加载数据、预处理并划分数据集"""
    df = pd.read_csv(DATA_PATH)

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


# ================================================================================
# 模型训练
# ================================================================================
def train_dirichlet_moe(
    X_train: np.ndarray,
    P_train: np.ndarray,
    X_val: np.ndarray,
    P_val: np.ndarray,
    Wtr: torch.Tensor | None,
    Wva: torch.Tensor | None,
) -> tuple:
    """训练 Dirichlet MoE 模型"""
    
    model = DirichletMoE(
        input_size=X_train.shape[1],
        output_size=7,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        k=TOP_K,
        noisy_gating=True,
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
    aux_losses: list[float] = []
    alpha0_means: list[float] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        p_hat, alpha0, aux_loss = model(Xtr, train=True)
        
        if Wtr is None:
            loss_main = dirichlet_nll(Ptr, p_hat, alpha0)
        else:
            loss_main = weighted_dirichlet_nll(Ptr, p_hat, alpha0, Wtr)
        
        loss = loss_main + AUX_COEF * aux_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            p_val, alpha0_val, aux_val = model(Xva, train=False)
            if Wva is None:
                val_main = dirichlet_nll(Pva, p_val, alpha0_val)
            else:
                val_main = weighted_dirichlet_nll(Pva, p_val, alpha0_val, Wva)
            val_loss = val_main + AUX_COEF * aux_val

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        aux_losses.append(aux_loss.item())
        alpha0_means.append(alpha0.mean().item())

        if val_loss.item() < best_val_loss - 1e-6:
            best_val_loss = val_loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 50 == 0:
            print(f"[DirichletMoE] epoch={epoch:3d} train_loss={loss.item():.4f} "
                  f"val_loss={val_loss.item():.4f} alpha0_mean={alpha0.mean().item():.1f}")

        if bad >= PATIENCE:
            print(f"[DirichletMoE] Early stopping at epoch {epoch}.")
            break

    if best_state:
        model.load_state_dict(best_state)

    info = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "aux_losses": aux_losses,
        "alpha0_means": alpha0_means,
        "best_epoch": len(train_losses) - bad,
        "best_val_loss": best_val_loss,
        "bad": bad,
    }
    return model, info


# ================================================================================
# 置信区间采样
# ================================================================================
def sample_dirichlet_ci(
    p_hat: np.ndarray,
    alpha0: np.ndarray,
    n_samples: int = N_SAMPLES_CI,
    ci_level: float = CI_LEVEL,
) -> tuple[np.ndarray, np.ndarray]:
    """
    通过 Dirichlet 采样计算置信区间
    
    参数:
        p_hat: 均值分布 (n, K)
        alpha0: 集中度 (n,)
        n_samples: 采样次数
        ci_level: 置信水平
    
    返回:
        ci_lower: 下界 (n, K)
        ci_upper: 上界 (n, K)
    """
    n, K = p_hat.shape
    ci_lower = np.zeros_like(p_hat)
    ci_upper = np.zeros_like(p_hat)
    
    alpha_low = (1 - ci_level) / 2
    alpha_high = 1 - alpha_low
    
    for i in range(n):
        alpha = alpha0[i] * p_hat[i]
        # 确保 alpha > 0
        alpha = np.maximum(alpha, 1e-6)
        samples = np.random.dirichlet(alpha, size=n_samples)  # (n_samples, K)
        ci_lower[i] = np.percentile(samples, alpha_low * 100, axis=0)
        ci_upper[i] = np.percentile(samples, alpha_high * 100, axis=0)
    
    return ci_lower, ci_upper


# ================================================================================
# 模型评估与推理
# ================================================================================
def predict_with_uncertainty(
    model,
    X: np.ndarray,
    n_samples: int = N_SAMPLES_CI,
    ci_level: float = CI_LEVEL,
) -> dict:
    """
    推理并计算置信区间
    
    返回:
        p_hat: 均值分布
        alpha0: 集中度
        ci_lower: 置信区间下界
        ci_upper: 置信区间上界
    """
    model.eval()
    X_tensor = torch.tensor(X, device=DEVICE)
    
    with torch.no_grad():
        p_hat, alpha0, _ = model(X_tensor, train=False)
        p_hat_np = p_hat.cpu().numpy()
        alpha0_np = alpha0.cpu().numpy()
    
    ci_lower, ci_upper = sample_dirichlet_ci(p_hat_np, alpha0_np, n_samples, ci_level)
    
    return {
        "p_hat": p_hat_np,
        "alpha0": alpha0_np,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def evaluate(model, X_test: np.ndarray, P_test: np.ndarray) -> tuple[dict, dict]:
    """在测试集上评估模型"""
    results = predict_with_uncertainty(model, X_test)
    P_pred = results["p_hat"]
    alpha0 = results["alpha0"]

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
        "alpha0_mean": float(alpha0.mean()),
        "alpha0_std": float(alpha0.std()),
        "alpha0_min": float(alpha0.min()),
        "alpha0_max": float(alpha0.max()),
    }
    
    print("\n[DirichletMoE] 测试集评估")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    return results, metrics


# ================================================================================
# 快速验证指标（无采样，适用于超参搜索）
# ================================================================================
def quick_val_metrics(model, X_val: np.ndarray, P_val: np.ndarray) -> dict:
    """更快的验证评估：跳过CI采样，仅对均值分布做度量。"""
    model.eval()
    with torch.no_grad():
        Xv = torch.tensor(X_val, device=DEVICE)
        p_hat, _, _ = model(Xv, train=False)
        P_pred = p_hat.cpu().numpy()
    mae = float(np.mean(np.abs(P_pred - P_val)))
    js = float(np.mean([jensenshannon(P_val[i], P_pred[i]) for i in range(len(P_val))]))
    eps = 1e-12
    kl = float(np.mean(np.sum(P_val * (np.log(P_val + eps) - np.log(P_pred + eps)), axis=1)))
    return {"mae": mae, "js_mean": js, "kl": kl}


# ================================================================================
# 保存预测结果
# ================================================================================
def save_predictions(results: dict, path: str) -> None:
    """保存预测结果到 CSV"""
    p_hat = results["p_hat"]
    alpha0 = results["alpha0"]
    ci_lower = results["ci_lower"]
    ci_upper = results["ci_upper"]
    
    data = {}
    
    # 均值分布
    for i, col in enumerate(DIST_COLS):
        data[f"pred_{col}"] = p_hat[:, i]
    
    # 集中度
    data["alpha0"] = alpha0
    
    # 置信区间
    for i, col in enumerate(DIST_COLS):
        data[f"ci_lower_{col}"] = ci_lower[:, i]
        data[f"ci_upper_{col}"] = ci_upper[:, i]
    
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"[DirichletMoE] 预测结果已保存: {path}")


# ================================================================================
# 可视化
# ================================================================================
def plot_training_history(info: dict, save_path: str):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 损失曲线
    axes[0].plot(info["train_losses"], label="Train Loss", alpha=0.8)
    axes[0].plot(info["val_losses"], label="Val Loss", alpha=0.8)
    axes[0].axvline(info["best_epoch"], color="red", linestyle="--", label=f"Best Epoch: {info['best_epoch']}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 辅助损失
    axes[1].plot(info["aux_losses"], color="green", alpha=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Aux Loss")
    axes[1].set_title("Auxiliary Loss (Load Balancing)")
    axes[1].grid(alpha=0.3)
    
    # alpha0 均值变化
    axes[2].plot(info["alpha0_means"], color="purple", alpha=0.8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("alpha0 Mean")
    axes[2].set_title("Concentration Parameter alpha0 During Training")
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DirichletMoE] 训练历史已保存: {save_path}")


def plot_alpha0_distribution(alpha0: np.ndarray, save_path: str):
    """绘制 alpha0 分布"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(alpha0, bins=30, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("alpha0 (Concentration)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of alpha0 on Test Set")
    axes[0].axvline(alpha0.mean(), color="red", linestyle="--", label=f"Mean: {alpha0.mean():.1f}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # alpha0 vs 不确定性（CI宽度）
    # 不确定性与 1/alpha0 成正比
    uncertainty = 1.0 / alpha0
    axes[1].scatter(alpha0, uncertainty, alpha=0.5, s=10)
    axes[1].set_xlabel("alpha0 (Concentration)")
    axes[1].set_ylabel("Uncertainty (1/alpha0)")
    axes[1].set_title("alpha0 vs Uncertainty")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DirichletMoE] alpha0 分布已保存: {save_path}")


def plot_predictions_with_ci(
    P_test: np.ndarray,
    results: dict,
    sample_indices: list,
    save_path: str,
):
    """绘制带置信区间的预测分布对比"""
    p_hat = results["p_hat"]
    ci_lower = results["ci_lower"]
    ci_upper = results["ci_upper"]
    alpha0 = results["alpha0"]
    
    n_samples = len(sample_indices)
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    x = np.arange(len(DIST_COLS))
    width = 0.35
    
    for ax_idx, sample_idx in enumerate(sample_indices):
        ax = axes[ax_idx]
        
        true_dist = P_test[sample_idx]
        pred_dist = p_hat[sample_idx]
        lower = ci_lower[sample_idx]
        upper = ci_upper[sample_idx]
        a0 = alpha0[sample_idx]
        
        # 真实分布
        ax.bar(x - width/2, true_dist, width, label="True", color="steelblue", edgecolor="black")
        
        # 预测分布 + 置信区间
        ax.bar(x + width/2, pred_dist, width, label="Pred", color="coral", edgecolor="black")
        ax.errorbar(x + width/2, pred_dist, yerr=[pred_dist - lower, upper - pred_dist],
                    fmt="none", ecolor="black", capsize=3, capthick=1)
        
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in DIST_COLS], fontsize=7)
        ax.set_title(f"Sample {sample_idx}, alpha0={a0:.1f}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    
    # 隐藏多余的子图
    for ax_idx in range(len(sample_indices), len(axes)):
        axes[ax_idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DirichletMoE] 带置信区间的预测已保存: {save_path}")


def plot_ci_coverage(P_test: np.ndarray, results: dict, save_path: str):
    """分析置信区间覆盖率"""
    ci_lower = results["ci_lower"]
    ci_upper = results["ci_upper"]
    
    # 检查每个分量的覆盖率
    coverage = (P_test >= ci_lower) & (P_test <= ci_upper)
    coverage_per_dim = coverage.mean(axis=0)
    coverage_overall = coverage.mean()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(DIST_COLS))
    ax.bar(x, coverage_per_dim, color="teal", edgecolor="black")
    ax.axhline(CI_LEVEL, color="red", linestyle="--", label=f"Target: {CI_LEVEL:.0%}")
    ax.axhline(coverage_overall, color="orange", linestyle="-.", label=f"Overall: {coverage_overall:.1%}")
    
    ax.set_xticks(x)
    ax.set_xticklabels(DIST_COLS)
    ax.set_xlabel("Distribution Bins")
    ax.set_ylabel("Coverage Rate")
    ax.set_title(f"Confidence Interval Coverage (Target: {CI_LEVEL:.0%})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DirichletMoE] 置信区间覆盖率已保存: {save_path}")
    
    return {"coverage_per_dim": coverage_per_dim.tolist(), "coverage_overall": float(coverage_overall)}


def plot_error_vs_confidence(P_test: np.ndarray, results: dict, save_path: str):
    """分析误差与置信度的关系"""
    p_hat = results["p_hat"]
    alpha0 = results["alpha0"]
    
    # 计算每个样本的 MAE
    mae_per_sample = np.abs(p_hat - P_test).mean(axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # MAE vs alpha0
    axes[0].scatter(alpha0, mae_per_sample, alpha=0.5, s=15)
    axes[0].set_xlabel("alpha0 (Confidence)")
    axes[0].set_ylabel("MAE per Sample")
    axes[0].set_title("Prediction Error vs Confidence")
    axes[0].grid(alpha=0.3)
    
    # 分箱统计
    n_bins = 10
    alpha0_bins = np.percentile(alpha0, np.linspace(0, 100, n_bins + 1))
    bin_means = []
    bin_centers = []
    
    for i in range(n_bins):
        mask = (alpha0 >= alpha0_bins[i]) & (alpha0 < alpha0_bins[i + 1] + 1e-6)
        if mask.sum() > 0:
            bin_means.append(mae_per_sample[mask].mean())
            bin_centers.append((alpha0_bins[i] + alpha0_bins[i + 1]) / 2)
    
    axes[1].plot(bin_centers, bin_means, marker="o", linewidth=2)
    axes[1].set_xlabel("alpha0 (Confidence) - Binned")
    axes[1].set_ylabel("Mean MAE")
    axes[1].set_title("Binned: Error vs Confidence")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DirichletMoE] 误差-置信度关系已保存: {save_path}")


def plot_comprehensive_summary(
    P_test: np.ndarray,
    results: dict,
    metrics: dict,
    info: dict,
    save_path: str,
):
    """绘制综合总结图"""
    p_hat = results["p_hat"]
    alpha0 = results["alpha0"]
    ci_lower = results["ci_lower"]
    ci_upper = results["ci_upper"]
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 训练曲线
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(info["train_losses"], label="Train", alpha=0.8)
    ax1.plot(info["val_losses"], label="Val", alpha=0.8)
    ax1.axvline(info["best_epoch"], color="red", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Curve")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. alpha0 分布
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(alpha0, bins=25, edgecolor="black", alpha=0.7)
    ax2.axvline(alpha0.mean(), color="red", linestyle="--")
    ax2.set_xlabel("alpha0")
    ax2.set_ylabel("Count")
    ax2.set_title(f"alpha0 Distribution (mean={alpha0.mean():.1f})")
    ax2.grid(alpha=0.3)
    
    # 3. 指标汇总
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.axis("off")
    metric_text = "\n".join([
        f"MAE: {metrics['mae']:.4f}",
        f"KL Divergence: {metrics['kl']:.4f}",
        f"JS Divergence: {metrics['js_mean']:.4f}",
        f"Cosine Similarity: {metrics['cos_sim']:.4f}",
        f"R²: {metrics['r2']:.4f}",
        f"Max Error: {metrics['max_error']:.4f}",
        "",
        f"alpha0 Mean: {metrics['alpha0_mean']:.1f}",
        f"alpha0 Std: {metrics['alpha0_std']:.1f}",
        f"alpha0 Range: [{metrics['alpha0_min']:.1f}, {metrics['alpha0_max']:.1f}]",
    ])
    ax3.text(0.1, 0.5, metric_text, fontsize=12, verticalalignment="center", fontfamily="monospace")
    ax3.set_title("Evaluation Metrics")
    
    # 4. 随机样本分布对比
    ax4 = fig.add_subplot(2, 3, 4)
    sample_idx = np.random.randint(0, len(P_test))
    x = np.arange(len(DIST_COLS))
    width = 0.35
    ax4.bar(x - width/2, P_test[sample_idx], width, label="True", color="steelblue")
    ax4.bar(x + width/2, p_hat[sample_idx], width, label="Pred", color="coral")
    ax4.errorbar(x + width/2, p_hat[sample_idx],
                 yerr=[p_hat[sample_idx] - ci_lower[sample_idx], ci_upper[sample_idx] - p_hat[sample_idx]],
                 fmt="none", ecolor="black", capsize=3)
    ax4.set_xticks(x)
    ax4.set_xticklabels([c[:5] for c in DIST_COLS], fontsize=8)
    ax4.set_title(f"Sample {sample_idx}, alpha0={alpha0[sample_idx]:.1f}")
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)
    
    # 5. CI 覆盖率
    ax5 = fig.add_subplot(2, 3, 5)
    coverage = (P_test >= ci_lower) & (P_test <= ci_upper)
    coverage_per_dim = coverage.mean(axis=0)
    ax5.bar(x, coverage_per_dim, color="teal", edgecolor="black")
    ax5.axhline(CI_LEVEL, color="red", linestyle="--", label=f"Target: {CI_LEVEL:.0%}")
    ax5.set_xticks(x)
    ax5.set_xticklabels([c[:5] for c in DIST_COLS], fontsize=8)
    ax5.set_ylabel("Coverage")
    ax5.set_title("CI Coverage per Bin")
    ax5.legend()
    ax5.grid(axis="y", alpha=0.3)
    ax5.set_ylim(0, 1.05)
    
    # 6. 误差 vs 置信度
    ax6 = fig.add_subplot(2, 3, 6)
    mae_per_sample = np.abs(p_hat - P_test).mean(axis=1)
    ax6.scatter(alpha0, mae_per_sample, alpha=0.4, s=10)
    ax6.set_xlabel("alpha0 (Confidence)")
    ax6.set_ylabel("MAE")
    ax6.set_title("Error vs Confidence")
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DirichletMoE] 综合总结已保存: {save_path}")


def write_report(metrics: dict, info: dict, coverage_stats: dict, path: str) -> None:
    """保存 JSON 报告"""
    report = {
        "model": "DirichletMoE",
        "config": {
            "num_experts": NUM_EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "top_k": TOP_K,
            "aux_coef": AUX_COEF,
            "alpha0_min": ALPHA0_MIN,
            "alpha0_max": ALPHA0_MAX,
            "ci_level": CI_LEVEL,
        },
        "training": {
            "best_epoch": info.get("best_epoch"),
            "best_val_loss": info.get("best_val_loss"),
        },
        "metrics": metrics,
        "ci_coverage": coverage_stats,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, ensure_ascii=False, indent=2, fp=f)
    print(f"[DirichletMoE] 报告已保存: {path}")


def generate_summary_report(metrics: dict, info: dict, coverage_stats: dict, output_dir: str):
    """生成文字总结报告"""
    report = f"""
{'='*80}
Dirichlet MoE 分布预测总结报告
{'='*80}

一、模型配置
-----------
专家数量: {NUM_EXPERTS}
隐藏层大小: {HIDDEN_SIZE}
Top-K 路由: {TOP_K}
辅助损失系数: {AUX_COEF}
alpha0 范围: [{ALPHA0_MIN}, {ALPHA0_MAX}]
置信水平: {CI_LEVEL:.0%}

二、训练结果
-----------
最佳验证轮次: {info.get('best_epoch')}
最佳验证损失: {info.get('best_val_loss'):.6f}

三、测试集评估
-------------
MAE: {metrics['mae']:.6f} (约 {metrics['mae']*100:.2f}%)
KL散度: {metrics['kl']:.6f}
JS散度: {metrics['js_mean']:.6f}
余弦相似度: {metrics['cos_sim']:.6f}
R²: {metrics['r2']:.6f}
最大误差: {metrics['max_error']:.6f}

四、不确定性估计
---------------
alpha0 均值: {metrics['alpha0_mean']:.1f}
alpha0 标准差: {metrics['alpha0_std']:.1f}
alpha0 范围: [{metrics['alpha0_min']:.1f}, {metrics['alpha0_max']:.1f}]

置信区间覆盖率 (目标 {CI_LEVEL:.0%}):
  总体覆盖率: {coverage_stats['coverage_overall']:.1%}

五、模型解读
-----------
1. 均值分布 p̂: Softmax 输出的 7 维概率分布，表示各猜测次数的预测概率
2. 集中度 alpha0: 表示模型对预测的置信程度
   - alpha0 越大，置信区间越窄，模型越确定
   - alpha0 越小，置信区间越宽，模型越不确定
3. Dirichlet 参数: α = alpha0 × p̂
4. 置信区间: 通过从 Dir(α) 采样得到

六、输出文件
-----------
预测结果 (含CI): {OUTPUT_DIR}/dirichlet_moe_predictions.csv
训练曲线: {OUTPUT_DIR}/training_history.png
alpha0 分布: {OUTPUT_DIR}/alpha0_distribution.png
带CI的预测: {OUTPUT_DIR}/predictions_with_ci.png
CI覆盖率: {OUTPUT_DIR}/ci_coverage.png
误差-置信度: {OUTPUT_DIR}/error_vs_confidence.png
综合总结: {OUTPUT_DIR}/comprehensive_summary.png
JSON报告: {OUTPUT_DIR}/report.json

{'='*80}
"""
    
    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(report)
    print(f"\n[DirichletMoE] 文字报告已保存: {report_path}")


# ================================================================================
# 主函数
# ================================================================================
def main():
    parser = argparse.ArgumentParser(description="Dirichlet MoE with probability output")
    parser.add_argument("--search", action="store_true", help="运行超参数搜索并用最佳参数产出完整结果")
    parser.add_argument("--trials", type=int, default=0, help="随机试验次数（0使用预定义网格）")
    args = parser.parse_args()

    set_seed()
    print(f"设备: {DEVICE}")
    print(f"数据路径: {DATA_PATH}")

    # 加载数据
    (
        X_train, X_val, X_test,
        P_train, P_val, P_test,
        N_train, N_val, N_test,
    ) = load_and_split_data()

    print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")

    def run_once(config: dict):
        global LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, WEIGHT_MODE
        global NUM_EXPERTS, HIDDEN_SIZE, TOP_K, AUX_COEF

        # 备份
        _bk = (LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, WEIGHT_MODE,
               NUM_EXPERTS, HIDDEN_SIZE, TOP_K, AUX_COEF)

        try:
            LR = config.get("lr", LR)
            WEIGHT_DECAY = config.get("weight_decay", WEIGHT_DECAY)
            MAX_EPOCHS = config.get("max_epochs", MAX_EPOCHS)
            PATIENCE = config.get("patience", PATIENCE)
            WEIGHT_MODE = config.get("weight_mode", WEIGHT_MODE)
            NUM_EXPERTS = config.get("num_experts", NUM_EXPERTS)
            HIDDEN_SIZE = config.get("hidden_size", HIDDEN_SIZE)
            TOP_K = config.get("top_k", TOP_K)
            AUX_COEF = config.get("aux_coef", AUX_COEF)

            Wtr = torch.tensor(make_weights_from_N(N_train, WEIGHT_MODE), device=DEVICE) if N_train is not None else None
            Wva = torch.tensor(make_weights_from_N(N_val, WEIGHT_MODE), device=DEVICE) if N_val is not None else None

            model, info = train_dirichlet_moe(X_train, P_train, X_val, P_val, Wtr, Wva)
            quick = quick_val_metrics(model, X_val, P_val)
            return model, info, quick
        finally:
            (LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, WEIGHT_MODE,
             NUM_EXPERTS, HIDDEN_SIZE, TOP_K, AUX_COEF) = _bk

    if args.search:
        # 预定义小网格，控制时间成本
        grid = []
        for lr in [1e-3, 5e-4, 3e-4]:
            for hidden in [128, 256]:
                for experts in [4, 6]:
                    for weight_mode in ["sqrt", "log1p"]:
                        grid.append({
                            "lr": lr,
                            "hidden_size": hidden,
                            "num_experts": experts,
                            "top_k": 1,
                            "aux_coef": 5e-4,
                            "weight_decay": 1e-4,
                            "max_epochs": 250,
                            "patience": 40,
                            "weight_mode": weight_mode,
                        })

        best = None
        best_model = None
        print(f"开始超参搜索，共 {len(grid)} 组...")
        for i, cfg in enumerate(grid, 1):
            print(f"\n[Search] Trial {i}/{len(grid)}: {cfg}")
            model, info, quick = run_once(cfg)
            key = info.get("best_val_loss", float("inf"))
            print(f"[Search] val_loss={key:.6f}, quick_js={quick['js_mean']:.6f}, quick_mae={quick['mae']:.6f}")
            if (best is None) or (key < best[0] - 1e-8) or (abs(key - best[0]) < 1e-8 and quick["js_mean"] < best[2]["js_mean"]):
                best = (key, cfg, quick)
                best_model = model

        assert best is not None
        print(f"\n[Search] 最优配置: {best[1]} with val_loss={best[0]:.6f}, js={best[2]['js_mean']:.6f}")

        # 使用最优配置重新完整训练并在测试集产出结果
        cfg = best[1].copy()
        cfg.update({"max_epochs": MAX_EPOCHS, "patience": max(PATIENCE, 60)})
        model, info, _ = run_once(cfg)

        # 测试集评估与产出
        results, metrics = evaluate(model, X_test, P_test)
        save_predictions(results, os.path.join(OUTPUT_DIR, "dirichlet_moe_predictions.csv"))
        plot_training_history(info, os.path.join(OUTPUT_DIR, "training_history.png"))
        plot_alpha0_distribution(results["alpha0"], os.path.join(OUTPUT_DIR, "alpha0_distribution.png"))
        sample_indices = np.random.choice(len(P_test), size=min(10, len(P_test)), replace=False).tolist()
        plot_predictions_with_ci(P_test, results, sample_indices, os.path.join(OUTPUT_DIR, "predictions_with_ci.png"))
        coverage_stats = plot_ci_coverage(P_test, results, os.path.join(OUTPUT_DIR, "ci_coverage.png"))
        plot_error_vs_confidence(P_test, results, os.path.join(OUTPUT_DIR, "error_vs_confidence.png"))
        plot_comprehensive_summary(P_test, results, metrics, info, os.path.join(OUTPUT_DIR, "comprehensive_summary.png"))
        write_report(metrics, info, coverage_stats, os.path.join(OUTPUT_DIR, "report.json"))
        generate_summary_report(metrics, info, coverage_stats, OUTPUT_DIR)
        print(f"\n完成：使用最优参数训练并输出结果到 {OUTPUT_DIR}/")
        return

    # 默认：按当前全局配置直接训练并产出
    print(f"Dirichlet MoE 配置: num_experts={NUM_EXPERTS}, hidden_size={HIDDEN_SIZE}, k={TOP_K}")
    Wtr = torch.tensor(make_weights_from_N(N_train, WEIGHT_MODE), device=DEVICE) if N_train is not None else None
    Wva = torch.tensor(make_weights_from_N(N_val, WEIGHT_MODE), device=DEVICE) if N_val is not None else None
    model, info = train_dirichlet_moe(X_train, P_train, X_val, P_val, Wtr, Wva)
    results, metrics = evaluate(model, X_test, P_test)
    save_predictions(results, os.path.join(OUTPUT_DIR, "dirichlet_moe_predictions.csv"))
    plot_training_history(info, os.path.join(OUTPUT_DIR, "training_history.png"))
    plot_alpha0_distribution(results["alpha0"], os.path.join(OUTPUT_DIR, "alpha0_distribution.png"))
    sample_indices = np.random.choice(len(P_test), size=min(10, len(P_test)), replace=False).tolist()
    plot_predictions_with_ci(P_test, results, sample_indices, os.path.join(OUTPUT_DIR, "predictions_with_ci.png"))
    coverage_stats = plot_ci_coverage(P_test, results, os.path.join(OUTPUT_DIR, "ci_coverage.png"))
    plot_error_vs_confidence(P_test, results, os.path.join(OUTPUT_DIR, "error_vs_confidence.png"))
    plot_comprehensive_summary(P_test, results, metrics, info, os.path.join(OUTPUT_DIR, "comprehensive_summary.png"))
    write_report(metrics, info, coverage_stats, os.path.join(OUTPUT_DIR, "report.json"))
    generate_summary_report(metrics, info, coverage_stats, OUTPUT_DIR)
    print(f"\n完成：Dirichlet MoE 分布预测已输出到 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
