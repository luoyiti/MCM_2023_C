"""
无强制差异化损失的 MoE 模型训练实验

本脚本取消所有专家差异化损失，观察专家在纯任务驱动下的自然分化行为。
"""

import sys
import os

# 确保可以导入 moe_booststrap 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 清除缓存的模块
for mod in list(sys.modules.keys()):
    if 'moe_booststrap' in mod:
        del sys.modules[mod]

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

# 导入 MoE Bootstrap 模块
from moe_booststrap import (
    DEVICE, SEED, DATA_PATH, DIST_COLS, OUTPUT_DIR,
    NUM_EXPERTS, TOP_K, HIDDEN_SIZE, AUX_LOSS_WEIGHT, LR, WD, MAX_EPOCHS, PATIENCE,
    BOOTSTRAP_B, BOOTSTRAP_CI, BOOTSTRAP_EPOCH_SCALE,
    load_and_split_data, make_weights_from_N,
    set_seed, train_moe, evaluate,
    bootstrap_predict, bootstrap_summary, bootstrap_evaluate,
    plot_training_history, plot_random_sample_distributions, plot_error_analysis,
    analyze_expert_usage, compute_expert_outputs,
    plot_sample_expert_decomposition, plot_holdout_bar_with_ci, plot_holdout_violin,
    plot_uncertainty, plot_expert_parallel_coordinates,
    save_uncertainty_arrays, generate_summary_report, format_metrics_table,
)

# 创建无差异化损失的输出目录
NO_DIV_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "no_diversity_loss")
os.makedirs(NO_DIV_OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("无强制差异化损失的 MoE 模型训练实验")
print("=" * 60)
print(f"\nDevice: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
print(f"Data path: {DATA_PATH}")
print(f"Output dir: {NO_DIV_OUTPUT_DIR}")

# ========================== 数据加载 ==========================
print("\n" + "=" * 60)
print("1. 数据加载与预处理")
print("=" * 60)

set_seed(SEED)

result = load_and_split_data(
    data_path=DATA_PATH,
    holdout_word="eerie",
    test_size=0.30,
    val_ratio=0.50,
)

X_train, X_val, X_test = result[0], result[1], result[2]
P_train, P_val, P_test = result[3], result[4], result[5]
N_train, N_val, N_test = result[6], result[7], result[8]
holdout_pack, scaler = result[9], result[10]

if holdout_pack is not None:
    X_holdout = holdout_pack["X"]
    P_holdout = holdout_pack["P_true"]
else:
    X_holdout, P_holdout = None, None

if N_train is not None:
    train_weights = make_weights_from_N(N_train, mode="sqrt")
else:
    train_weights = np.ones(len(X_train), dtype=np.float32)

print(f"\n数据集规模:")
print(f"  训练集: {len(X_train)} 样本")
print(f"  验证集: {len(X_val)} 样本")
print(f"  测试集: {len(X_test)} 样本")
if X_holdout is not None:
    print(f"  Holdout: {len(X_holdout)} 样本 (eerie)")
print(f"  特征数: {X_train.shape[1]}")
print(f"  输出维度: {P_train.shape[1]}")


# ========================== 模型配置与训练 ==========================
print("\n" + "=" * 60)
print("2. 模型配置与训练（无差异化损失）")
print("=" * 60)

# 关键配置：所有差异化损失系数设为 0
config = {
    "num_experts": NUM_EXPERTS,
    "top_k": TOP_K,
    "hidden_size": HIDDEN_SIZE,
    "aux_loss_weight": AUX_LOSS_WEIGHT,
    "expert_diversity_coef": 0.0,        # 【禁用】参数差异化
    "expert_output_diversity_coef": 0.0, # 【禁用】输出相似度惩罚
    "expert_js_divergence_coef": 0.0,    # 【禁用】JS散度鼓励
    "lr": LR,
    "wd": WD,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
}

print("\n模型配置（无差异化损失）:")
for k, v in config.items():
    print(f"  {k}: {v}")

Wtr = torch.tensor(train_weights, device=DEVICE) if train_weights is not None else None

# 训练模型 - 显式传入 0 值的差异化系数
model, train_info_dict = train_moe(
    X_train, P_train, X_val, P_val,
    Wtr=Wtr,
    num_experts=NUM_EXPERTS,
    top_k=TOP_K,
    hidden_size=HIDDEN_SIZE,
    aux_coef=AUX_LOSS_WEIGHT,
    expert_diversity_coef=0.0,           # 【禁用】参数差异化
    expert_output_diversity_coef=0.0,    # 【禁用】输出相似度惩罚
    expert_js_divergence_coef=0.0,       # 【禁用】JS散度鼓励
    lr=LR,
    weight_decay=WD,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
    device=DEVICE,
    verbose=True,
)

train_losses = train_info_dict["train_losses"]
val_losses = train_info_dict["val_losses"]
bad = train_info_dict["bad"]

# 是否在运行过程中弹出并更新专家差异图（热力图）。
# 将其保存在输出目录，同时在交互式会话中展示。
SHOW_EXPERT_DIFF_DURING_RUN = True

# ========================== 训练曲线可视化 ==========================
print("\n" + "=" * 60)
print("3. 训练曲线可视化")
print("=" * 60)

train_info = plot_training_history(
    train_losses, val_losses, bad, min(val_losses),
    save_path=os.path.join(NO_DIV_OUTPUT_DIR, "training_history_no_div.png"),
    show=False,
)
print(f"\n最佳轮次: {train_info['best_epoch']}")
print(f"最佳验证损失: {train_info['best_val_loss']:.6f}")

# ========================== 测试集评估 ==========================
print("\n" + "=" * 60)
print("4. 测试集评估")
print("=" * 60)

P_pred, test_metrics = evaluate(model, X_test, P_test, device=DEVICE)
print(format_metrics_table(test_metrics, "测试集评估指标（无差异化损失）"))

plot_random_sample_distributions(
    P_test, P_pred, sample_size=10,
    save_path=os.path.join(NO_DIV_OUTPUT_DIR, "sample_distributions_no_div.png"),
    show=False,
)

error_info = plot_error_analysis(
    P_test, P_pred,
    save_path=os.path.join(NO_DIV_OUTPUT_DIR, "error_analysis_no_div.png"),
    show=False,
)

print("\n各桶 MAE:")
for i, (col, mae) in enumerate(zip(DIST_COLS, error_info["mae_per_dim"])):
    print(f"  {col}: {mae:.6f}")


# ========================== 专家分析 ==========================
print("\n" + "=" * 60)
print("5. 专家分析（无差异化损失下的自然分化）")
print("=" * 60)

expert_stats = analyze_expert_usage(
    model, X_test,
    save_path=os.path.join(NO_DIV_OUTPUT_DIR, "expert_usage_no_div.png"),
    show=False,
    device=DEVICE,
)

print("\n专家使用率（无差异化损失）:")
for i, (usage, weight) in enumerate(zip(expert_stats["expert_usage"], expert_stats["expert_avg_weight"])):
    print(f"  Expert {i}: 使用率={usage:.2%}, 平均权重={weight:.4f}")

y_experts, gates, y_mix = compute_expert_outputs(model, X_test, device=DEVICE)

print(f"\n专家输出形状:")
print(f"  y_experts: {y_experts.shape}")
print(f"  gates: {gates.shape}")
print(f"  y_mix: {y_mix.shape}")


def _js_divergence(p, q, eps=1e-12):
    """计算两个离散分布的 JS 散度。p, q 为非负向量。"""
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


# 平行坐标图展示专家差异
parallel_stats = plot_expert_parallel_coordinates(
    y_experts, y_mix, P_test,
    save_path=os.path.join(NO_DIV_OUTPUT_DIR, "expert_parallel_coordinates_no_div.png"),
    show=False,
)

print(f"\n专家差异性分析（无差异化损失，基于相对排名）:")
print(f"  各专家峰值位置: {[DIST_COLS[p] for p in parallel_stats['peak_positions']]}")
print(f"  各专家平均排名: {parallel_stats['avg_rank']} (越小越好)")
print(f"  各专家排名稳定性: {parallel_stats['rank_stability']} (越小越稳定)")
print(f"\n专家间排名距离矩阵（基于排名的欧氏距离）:")
print(parallel_stats['rank_distances'])

# 专家分解可视化
plot_sample_expert_decomposition(
    P_test, y_experts, gates, y_mix,
    save_path=os.path.join(NO_DIV_OUTPUT_DIR, "expert_decomposition_no_div.png"),
    show=False,
    sample_size=6,
)

# ========================== Bootstrap 不确定性估计 ==========================
print("\n" + "=" * 60)
print("6. Bootstrap 不确定性估计")
print("=" * 60)

print(f"Bootstrap 配置:")
print(f"  B (迭代次数): {BOOTSTRAP_B}")
print(f"  CI (置信区间): {BOOTSTRAP_CI}")
print(f"  Epoch Scale: {BOOTSTRAP_EPOCH_SCALE}")

B_RUN = min(20, BOOTSTRAP_B)

# 注意：bootstrap_predict 使用配置文件中的差异化系数
# 由于 config.py 已被修改为 0，bootstrap 会自动使用无差异化损失
P_boot_test, P_boot_holdout = bootstrap_predict(
    X_train, P_train, N_train,
    X_val, P_val, N_val,
    X_test,
    X_holdout=X_holdout,
    B=B_RUN,
    num_experts=NUM_EXPERTS,
    top_k=TOP_K,
    hidden_size=HIDDEN_SIZE,
    max_epochs=int(MAX_EPOCHS * BOOTSTRAP_EPOCH_SCALE),
    verbose=True,
)

P_mean_test, P_std_test, P_low_test, P_high_test = bootstrap_summary(P_boot_test, ci_level=BOOTSTRAP_CI)
P_mean_holdout, P_std_holdout, P_low_holdout, P_high_holdout = bootstrap_summary(P_boot_holdout, ci_level=BOOTSTRAP_CI)

print(f"\n测试集 Bootstrap 结果:")
print(f"  均值形状: {P_mean_test.shape}")
print(f"  平均 CI 宽度: {(P_high_test - P_low_test).mean():.6f}")

print(f"\nHoldout Bootstrap 结果:")
print(f"  均值形状: {P_mean_holdout.shape}")
print(f"  平均 CI 宽度: {(P_high_holdout - P_low_holdout).mean():.6f}")

boot_result = bootstrap_evaluate(P_boot_test, P_test, ci_level=BOOTSTRAP_CI)
boot_metrics = boot_result["metrics"]

print(format_metrics_table(boot_metrics, "Bootstrap 评估指标（无差异化损失）"))
print(f"\nCI 覆盖率: {boot_result['ci_coverage']:.2%}")


# ========================== 不确定性可视化 ==========================
print("\n" + "=" * 60)
print("7. 不确定性可视化")
print("=" * 60)

plot_uncertainty(
    P_mean_test, P_low_test, P_high_test, P_test,
    save_path=os.path.join(NO_DIV_OUTPUT_DIR, "test_uncertainty_no_div.png"),
    show=False,
)

plot_holdout_bar_with_ci(
    P_mean_holdout, P_low_holdout, P_high_holdout, P_holdout,
    word="eerie",
    save_path=os.path.join(NO_DIV_OUTPUT_DIR, "holdout_bar_ci_no_div.png"),
    show=False,
)

plot_holdout_violin(
    P_boot_holdout,
    word="eerie",
    save_path=os.path.join(NO_DIV_OUTPUT_DIR, "holdout_violin_no_div.png"),
    show=False,
)

# ========================== 保存结果 ==========================
print("\n" + "=" * 60)
print("8. 保存结果")
print("=" * 60)

save_uncertainty_arrays(
    P_mean_test, P_low_test, P_high_test,
    save_dir=NO_DIV_OUTPUT_DIR,
    prefix="test_no_div",
)

report = generate_summary_report(
    train_info={
        "best_epoch": train_info["best_epoch"],
        "best_val_loss": train_info["best_val_loss"],
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "hidden_size": HIDDEN_SIZE,
        "expert_diversity_coef": 0.0,
        "expert_output_diversity_coef": 0.0,
        "expert_js_divergence_coef": 0.0,
    },
    test_metrics=test_metrics,
    bootstrap_metrics=boot_metrics,
    save_path=os.path.join(NO_DIV_OUTPUT_DIR, "summary_report_no_div.txt"),
)
print(report)

# ========================== 总结 ==========================
print("\n" + "=" * 60)
print("实验总结：无强制差异化损失下的专家分化")
print("=" * 60)

print("""
本实验取消了所有专家差异化损失：
  - expert_diversity_coef = 0.0（参数差异化）
  - expert_output_diversity_coef = 0.0（输出相似度惩罚）
  - expert_js_divergence_coef = 0.0（JS散度鼓励）

观察结果：
  1. 专家使用率分布
  2. 专家输出的自然差异（通过平行坐标图）
  3. 模型预测性能（MAE、JS散度等）
  4. Bootstrap 不确定性估计

输出文件保存在: {}
""".format(NO_DIV_OUTPUT_DIR))

print("\n关键输出文件:")
print(f"  - training_history_no_div.png: 训练曲线")
print(f"  - expert_usage_no_div.png: 专家使用率")
print(f"  - expert_parallel_coordinates_no_div.png: 专家差异性分析")
print(f"  - expert_decomposition_no_div.png: 专家分解可视化")
print(f"  - holdout_bar_ci_no_div.png: Holdout 预测（带 CI）")
print(f"  - summary_report_no_div.txt: 综合评估报告")
