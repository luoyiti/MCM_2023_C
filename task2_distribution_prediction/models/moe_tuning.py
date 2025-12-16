"""
MoE 超参数调优脚本
测试多种配置，找到最佳的 MoE 模型
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon

from moe import MoE

# ---------------- 全局配置 ----------------
DATA_PATH = "data/mcm_processed_data.csv"
N_COL = "number_of_reported_results"
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_COLS = [
    "Zipf-value", "letter_entropy", "feedback_entropy", "max_consecutive_vowels",
    "letter_freq_mean", "scrabble_score", "has_common_suffix", "num_rare_letters",
    "position_rarity", "positional_freq_min", "hamming_neighbors", "keyboard_distance",
    "semantic_distance",
    "1_try_simulate_random", "2_try_simulate_random", "3_try_simulate_random",
    "4_try_simulate_random", "5_try_simulate_random", "6_try_simulate_random", "7_try_simulate_random",
    "1_try_simulate_freq", "2_try_simulate_freq", "3_try_simulate_freq",
    "4_try_simulate_freq", "5_try_simulate_freq", "6_try_simulate_freq", "7_try_simulate_freq",
    "1_try_simulate_entropy", "2_try_simulate_entropy", "3_try_simulate_entropy",
    "4_try_simulate_entropy", "5_try_simulate_entropy", "6_try_simulate_entropy", "7_try_simulate_entropy",
    "rl_1_try_low_training", "rl_2_try_low_training", "rl_3_try_low_training",
    "rl_4_try_low_training", "rl_5_try_low_training", "rl_6_try_low_training", "rl_7_try_low_training",
    "rl_1_try_high_training", "rl_2_try_high_training", "rl_3_try_high_training",
    "rl_4_try_high_training", "rl_5_try_high_training", "rl_6_try_high_training", "rl_7_try_high_training",
    "rl_1_try_little_training", "rl_2_try_little_training", "rl_3_try_little_training",
    "rl_4_try_little_training", "rl_5_try_little_training", "rl_6_try_little_training", "rl_7_try_little_training",
]

DIST_COLS = ["1_try", "2_tries", "3_tries", "4_tries", "5_tries", "6_tries", "7_or_more_tries_x"]


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_weights_from_N(N_array, mode="sqrt"):
    if mode == "sqrt":
        w = np.sqrt(N_array)
    else:
        w = np.log1p(N_array)
    w = w / (w.mean() + 1e-12)
    return w.astype(np.float32)


def soft_cross_entropy(p_hat, p_true, eps=1e-12):
    p_hat = torch.clamp(p_hat, eps, 1.0)
    return -(p_true * torch.log(p_hat)).sum(dim=1).mean()


def weighted_soft_cross_entropy(p_hat, p_true, w, eps=1e-12):
    p_hat = torch.clamp(p_hat, eps, 1.0)
    per_sample = -(p_true * torch.log(p_hat)).sum(dim=1)
    return (w * per_sample).mean()


def load_and_split_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].copy().fillna(df[FEATURE_COLS].median(numeric_only=True))
    P = df[DIST_COLS].copy().fillna(0.0)
    if P.to_numpy().max() > 1.5:
        P = P / 100.0
    P = P.clip(lower=0.0)
    row_sum = P.sum(axis=1).replace(0, np.nan)
    P = P.div(row_sum, axis=0).fillna(1.0 / len(DIST_COLS))

    N_np = None
    if N_COL in df.columns:
        N = df[N_COL].fillna(df[N_COL].median()).clip(lower=1)
        N_np = N.to_numpy().astype(np.float32)

    X_np = X.to_numpy().astype(np.float32)
    P_np = P.to_numpy().astype(np.float32)

    if N_np is None:
        X_train, X_tmp, P_train, P_tmp = train_test_split(X_np, P_np, test_size=0.3, random_state=RANDOM_SEED)
        X_val, X_test, P_val, P_test = train_test_split(X_tmp, P_tmp, test_size=0.5, random_state=RANDOM_SEED)
        N_train = N_val = N_test = None
    else:
        X_train, X_tmp, P_train, P_tmp, N_train, N_tmp = train_test_split(X_np, P_np, N_np, test_size=0.3, random_state=RANDOM_SEED)
        X_val, X_test, P_val, P_test, N_val, N_test = train_test_split(X_tmp, P_tmp, N_tmp, test_size=0.5, random_state=RANDOM_SEED)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_val, X_test, P_train, P_val, P_test, N_train, N_val, N_test


def train_and_evaluate(config, X_train, P_train, X_val, P_val, X_test, P_test, N_train, N_val):
    """使用指定配置训练和评估模型"""
    set_seed()
    
    model = MoE(
        input_size=X_train.shape[1],
        output_size=7,
        num_experts=config["num_experts"],
        hidden_size=config["hidden_size"],
        noisy_gating=True,
        k=config["top_k"],
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    Xtr = torch.tensor(X_train, device=DEVICE)
    Ptr = torch.tensor(P_train, device=DEVICE)
    Xva = torch.tensor(X_val, device=DEVICE)
    Pva = torch.tensor(P_val, device=DEVICE)

    Wtr = torch.tensor(make_weights_from_N(N_train), device=DEVICE) if N_train is not None else None
    Wva = torch.tensor(make_weights_from_N(N_val), device=DEVICE) if N_val is not None else None

    best_state = None
    best_val_loss = float("inf")
    bad = 0

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        p_hat, aux_loss = model(Xtr)
        if Wtr is None:
            loss_main = soft_cross_entropy(p_hat, Ptr)
        else:
            loss_main = weighted_soft_cross_entropy(p_hat, Ptr, Wtr)
        loss = loss_main + config["aux_coef"] * aux_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            p_val, aux_val = model(Xva)
            if Wva is None:
                val_main = soft_cross_entropy(p_val, Pva)
            else:
                val_main = weighted_soft_cross_entropy(p_val, Pva, Wva)
            val_loss = val_main + config["aux_coef"] * aux_val

        if val_loss.item() < best_val_loss - 1e-6:
            best_val_loss = val_loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= config["patience"]:
            break

    if best_state:
        model.load_state_dict(best_state)

    # 评估
    model.eval()
    Xte = torch.tensor(X_test, device=DEVICE)
    with torch.no_grad():
        P_pred, _ = model(Xte)
        P_pred = P_pred.cpu().numpy()

    eps = 1e-12
    mae = np.mean(np.abs(P_pred - P_test))
    kl = np.mean(np.sum(P_test * (np.log(P_test + eps) - np.log(P_pred + eps)), axis=1))
    js_mean = np.mean([jensenshannon(P_test[i], P_pred[i]) for i in range(len(P_test))])
    cos_sim = np.mean([cosine_similarity([P_test[i]], [P_pred[i]])[0, 0] for i in range(len(P_test))])

    return {
        "mae": mae,
        "kl": kl,
        "js_mean": js_mean,
        "cos_sim": cos_sim,
        "best_val_loss": best_val_loss,
        "epochs_trained": epoch,
    }


def main():
    print("=" * 60)
    print("MoE 超参数调优")
    print("=" * 60)

    X_train, X_val, X_test, P_train, P_val, P_test, N_train, N_val, N_test = load_and_split_data()
    print(f"训练集: {X_train.shape[0]} 样本, 验证集: {X_val.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")

    # 定义要测试的配置
    configs = [
        {"name": "配置1: 2专家, aux=1e-3, patience=50", "num_experts": 2, "top_k": 2, "hidden_size": 64, "lr": 5e-3, "weight_decay": 1e-4, "aux_coef": 1e-3, "patience": 50, "max_epochs": 500},
        {"name": "配置2: 2专家, aux=1e-4, patience=60", "num_experts": 2, "top_k": 2, "hidden_size": 64, "lr": 5e-3, "weight_decay": 1e-4, "aux_coef": 1e-4, "patience": 60, "max_epochs": 500},
        {"name": "配置3: 3专家, aux=1e-3, patience=50", "num_experts": 3, "top_k": 2, "hidden_size": 64, "lr": 5e-3, "weight_decay": 1e-4, "aux_coef": 1e-3, "patience": 50, "max_epochs": 500},
        {"name": "配置4: 2专家, hidden=128, lr=1e-3", "num_experts": 2, "top_k": 2, "hidden_size": 128, "lr": 1e-3, "weight_decay": 1e-4, "aux_coef": 1e-3, "patience": 50, "max_epochs": 500},
        {"name": "配置5: 2专家, hidden=32, lr=1e-2", "num_experts": 2, "top_k": 2, "hidden_size": 32, "lr": 1e-2, "weight_decay": 1e-4, "aux_coef": 1e-3, "patience": 50, "max_epochs": 500},
        {"name": "配置6: 2专家, k=1, aux=1e-3", "num_experts": 2, "top_k": 1, "hidden_size": 64, "lr": 5e-3, "weight_decay": 1e-4, "aux_coef": 1e-3, "patience": 50, "max_epochs": 500},
        {"name": "配置7: 3专家, k=1, hidden=128", "num_experts": 3, "top_k": 1, "hidden_size": 128, "lr": 1e-3, "weight_decay": 1e-4, "aux_coef": 1e-3, "patience": 60, "max_epochs": 500},
        {"name": "配置8: 2专家, 低正则化", "num_experts": 2, "top_k": 2, "hidden_size": 64, "lr": 5e-3, "weight_decay": 1e-5, "aux_coef": 1e-4, "patience": 60, "max_epochs": 500},
    ]

    results = []
    best_result = None
    best_config = None

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] 测试 {config['name']}")
        result = train_and_evaluate(config, X_train, P_train, X_val, P_val, X_test, P_test, N_train, N_val)
        result["config_name"] = config["name"]
        result["config"] = config
        results.append(result)

        print(f"  MAE={result['mae']:.4f}, KL={result['kl']:.4f}, JS={result['js_mean']:.4f}, CosSim={result['cos_sim']:.4f}")

        # 以 KL 作为主要评价指标
        if best_result is None or result["kl"] < best_result["kl"]:
            best_result = result
            best_config = config

    # 打印总结
    print("\n" + "=" * 60)
    print("调优结果总结")
    print("=" * 60)
    print(f"{'配置名称':<35} {'MAE':>8} {'KL':>8} {'JS':>8} {'CosSim':>8}")
    print("-" * 75)
    for r in results:
        print(f"{r['config_name']:<35} {r['mae']:>8.4f} {r['kl']:>8.4f} {r['js_mean']:>8.4f} {r['cos_sim']:>8.4f}")

    print("\n" + "=" * 60)
    print("最佳配置")
    print("=" * 60)
    print(f"配置名称: {best_config['name']}")
    print(f"  num_experts: {best_config['num_experts']}")
    print(f"  top_k: {best_config['top_k']}")
    print(f"  hidden_size: {best_config['hidden_size']}")
    print(f"  lr: {best_config['lr']}")
    print(f"  weight_decay: {best_config['weight_decay']}")
    print(f"  aux_coef: {best_config['aux_coef']}")
    print(f"  patience: {best_config['patience']}")
    print(f"\n最佳测试集指标:")
    print(f"  MAE: {best_result['mae']:.6f}")
    print(f"  KL: {best_result['kl']:.6f}")
    print(f"  JS: {best_result['js_mean']:.6f}")
    print(f"  CosSim: {best_result['cos_sim']:.6f}")

    # 保存结果
    df_results = pd.DataFrame([{
        "config_name": r["config_name"],
        "mae": r["mae"],
        "kl": r["kl"],
        "js_mean": r["js_mean"],
        "cos_sim": r["cos_sim"],
        "best_val_loss": r["best_val_loss"],
        "epochs_trained": r["epochs_trained"],
    } for r in results])
    df_results.to_csv("moe_output/tuning_results.csv", index=False)
    print("\n调优结果已保存到 moe_output/tuning_results.csv")

    return best_config, best_result


if __name__ == "__main__":
    best_config, best_result = main()
