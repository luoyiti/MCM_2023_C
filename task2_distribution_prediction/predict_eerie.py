"""
任务2：预测 EERIE 在 2023-03-01 的成绩分布

功能：
1. 计算 EERIE 的所有特征
2. 使用训练好的模型预测分布
3. 量化不确定性
4. 生成完整报告（CSV + TXT + 可视化）

输出：
- results/task2/eerie_distribution.csv     (7个百分比)
- results/task2/eerie_full_report.txt      (完整报告含不确定性)
- results/task2/eerie_visualization.png    (可视化图表)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入项目配置
from shared import PROCESSED_DATA, TASK2_RESULTS, TASK2_PICTURES, MODELS_DIR

print("\n" + "="*70)
print("任务2：预测 EERIE 的成绩分布")
print("="*70 + "\n")


# ============ 步骤1：计算 EERIE 特征 ============
print("[步骤 1/5] 计算 EERIE 的特征...")

from compute_eerie_features import compute_eerie_features

word = "EERIE"
date = "2023-03-01"

# 计算 EERIE 的完整特征
eerie_features_full = compute_eerie_features()
print(f"  ✓ 计算了 {len(eerie_features_full.columns)} 个特征")


# ============ 步骤2：加载训练数据和模型 ============
print("\n[步骤 2/5] 加载训练数据...")

df = pd.read_csv(PROCESSED_DATA)
print(f"  ✓ 加载了 {len(df)} 天的数据")
print(f"  ✓ 数据路径: {PROCESSED_DATA}")

# 目标变量
dist_cols = ['1_try', '2_tries', '3_tries', '4_tries', '5_tries', '6_tries', '7_or_more_tries_x']

# 特征列（排除非特征列）
exclude_cols = ['date', 'contest_number', 'word', 'number_of_reported_results',
                'number_in_hard_mode', 'sum'] + dist_cols
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].fillna(df[feature_cols].mean())
y = df[dist_cols].fillna(0)

print(f"  ✓ 特征数: {len(feature_cols)}, 样本数: {len(X)}")


# ============ 步骤3：训练模型（如果没有保存的模型）============
print("\n[步骤 3/5] 训练/加载预测模型...")

model_file = '../models/distribution_rf_model.pkl'

if os.path.exists(model_file):
    import pickle
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    print(f"  ✓ 加载已有模型: {model_file}")
else:
    from sklearn.ensemble import RandomForestRegressor
    print("  训练新模型...")
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # 保存模型
    os.makedirs('../models', exist_ok=True)
    import pickle
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ 模型已保存: {model_file}")


# ============ 步骤4：预测 EERIE ============
print("\n[步骤 4/5] 预测 EERIE 的分布...")

# 对齐 EERIE 特征与训练数据特征
print("  对齐特征列...")
for col in feature_cols:
    if col not in eerie_features_full.columns:
        # 缺失特征用训练数据均值填充
        eerie_features_full[col] = X[col].mean()

# 按照训练特征顺序重排
X_eerie = eerie_features_full[feature_cols]
print(f"  ✓ 特征对齐完成，共 {len(feature_cols)} 个特征")

# 使用随机森林的每棵树进行预测（不确定性估计）
tree_predictions = []
for tree in model.estimators_:
    pred = tree.predict(X_eerie)
    tree_predictions.append(pred[0])

tree_predictions = np.array(tree_predictions)
mean_pred = tree_predictions.mean(axis=0)
std_pred = tree_predictions.std(axis=0)

# 归一化到 100%
mean_pred = mean_pred / mean_pred.sum()

print(f"  ✓ 使用了 {len(model.estimators_)} 棵决策树")
print(f"  ✓ 预测完成")


# ============ 步骤5：生成输出 ============
print("\n[步骤 5/5] 生成结果...")
print(f"  ✓ 结果目录: {TASK2_RESULTS}")
print(f"  ✓ 图片目录: {TASK2_PICTURES}")

# 5.1 保存 CSV（简洁版）
categories = ['1_try', '2_tries', '3_tries', '4_tries', '5_tries', '6_tries', '7_or_more_tries']
results_df = pd.DataFrame({
    'category': categories,
    'percentage': mean_pred * 100,
    'std': std_pred * 100
})
csv_file = TASK2_RESULTS / 'eerie_distribution.csv'
results_df.to_csv(csv_file, index=False)
print(f"  ✓ CSV 已保存: {csv_file}")

# 5.2 生成完整报告
report_file = TASK2_RESULTS / 'eerie_full_report.txt'
with open(report_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write(f"Wordle 预测报告: {word} ({date})\n")
    f.write("="*70 + "\n\n")
    
    f.write("预测分布:\n")
    f.write("-"*70 + "\n")
    f.write(f"{'类别':15s} | {'百分比':10s} | {'95% 置信区间':20s}\n")
    f.write("-"*70 + "\n")
    
    for i, cat in enumerate(categories):
        pct = mean_pred[i] * 100
        std = std_pred[i] * 100
        lower = max(0, pct - 1.96 * std)
        upper = min(100, pct + 1.96 * std)
        f.write(f"{cat:15s} | {pct:9.2f}% | [{lower:6.2f}%, {upper:6.2f}%]\n")
    
    f.write("-"*70 + "\n")
    f.write(f"总计: {mean_pred.sum()*100:.1f}%\n")
    f.write("-"*70 + "\n\n")
    
    # 不确定性指标
    f.write("不确定性指标:\n")
    f.write(f"  平均标准差: {std_pred.mean()*100:.2f}%\n")
    max_std_idx = np.argmax(std_pred)
    f.write(f"  最不确定类别: {categories[max_std_idx]} (std={std_pred[max_std_idx]*100:.2f}%)\n")
    
    # 熵
    epsilon = 1e-10
    entropy = -np.sum(mean_pred * np.log(mean_pred + epsilon))
    f.write(f"  预测熵: {entropy:.3f}\n\n")
    
    # 期望尝试次数
    attempt_numbers = np.array([1, 2, 3, 4, 5, 6, 9])
    expected_attempts = np.sum(mean_pred * attempt_numbers)
    f.write(f"期望尝试次数: {expected_attempts:.2f}\n")
    f.write(f"成功率: {mean_pred[:6].sum()*100:.1f}%\n")
    f.write(f"失败率: {mean_pred[6]*100:.1f}%\n")
    
    f.write("\n" + "="*70 + "\n")

print(f"  ✓ 完整报告已保存: {report_file}")

# 5.3 可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：分布柱状图
x = np.arange(len(categories))
ax1.bar(x, mean_pred * 100, yerr=std_pred * 1.96 * 100, 
        alpha=0.7, capsize=5, color='steelblue')
ax1.set_xlabel('尝试次数')
ax1.set_ylabel('百分比 (%)')
ax1.set_title(f'{word} 预测分布 ({date})')
ax1.set_xticks(x)
ax1.set_xticklabels(['1', '2', '3', '4', '5', '6', '7+'])
ax1.grid(axis='y', alpha=0.3)

# 右图：累积分布
cumulative = np.cumsum(mean_pred) * 100
ax2.plot(x, cumulative, marker='o', linewidth=2, markersize=8, color='darkred')
ax2.fill_between(x, 0, cumulative, alpha=0.3, color='red')
ax2.set_xlabel('尝试次数')
ax2.set_ylabel('累积百分比 (%)')
ax2.set_title(f'{word} 累积分布')
ax2.set_xticks(x)
ax2.set_xticklabels(['1', '2', '3', '4', '5', '6', '7+'])
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 105])

plt.tight_layout()
viz_file = TASK2_PICTURES / 'eerie_visualization.png'
plt.savefig(viz_file, dpi=300, bbox_inches='tight')
print(f"  ✓ 可视化已保存: {viz_file}")

# 打印到屏幕
print("\n" + "="*70)
print("📊 预测结果预览")
print("="*70)
print(results_df.to_string(index=False))
print("="*70)

print("\n🎉 任务2 完成！")
print(f"   - CSV (结果): {csv_file}")
print(f"   - TXT (报告): {report_file}")
print(f"   - PNG (图片): {viz_file}")
print("="*70 + "\n")
