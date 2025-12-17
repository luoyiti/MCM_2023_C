"""
Elastic Net回归模型：预测autoencoder_value
使用reduced_features_train.csv作为训练集，reduced_features_test.csv作为测试集
Elastic Net结合了L1（Lasso）和L2（Ridge）正则化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 设置中文字体和图表样式
plt.rcParams['font.family'] = 'Heiti TC'  # 替换为你选择的字体

sns.set_style("whitegrid")
sns.set_palette("husl")

# ==================== 1. 数据加载 ====================
print("=" * 60)
print("Elastic Net回归模型训练与评估")
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

# ==================== 2. 建立Elastic Net回归流水线 ====================
print("\n[3] 构建Elastic Net回归模型...")

pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("elasticnet", ElasticNet(random_state=42, max_iter=2000))
])

# ==================== 3. 网格搜索调参 ====================
print("\n[4] 网格搜索最优超参数...")
print("（Elastic Net回归训练速度较快，请稍候...）")

# Elastic Net的主要超参数：
# - alpha: 正则化强度
# - l1_ratio: L1和L2正则化的混合比例（0=Ridge，1=Lasso，0.5=两者各半）
param_grid = {
    "elasticnet__alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    "elasticnet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],  # 从偏向Ridge到偏向Lasso
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
        ("elasticnet", ElasticNet(
            alpha=search.best_params_["elasticnet__alpha"],
            l1_ratio=search.best_params_["elasticnet__l1_ratio"],
            random_state=42,
            max_iter=2000
        ))
    ])
    
    model_cv.fit(X_cv_train, y_cv_train)
    y_cv_pred = model_cv.predict(X_cv_val)
    cv_r2 = r2_score(y_cv_val, y_cv_pred)
    cv_scores.append(cv_r2)

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)
print(f"交叉验证 R²: {cv_mean:.4f} ± {cv_std:.4f}")

# ==================== 6. 特征重要性分析（Elastic Net系数）====================
print("\n[7] 特征重要性分析...")
elasticnet_model = best_model.named_steps['elasticnet']
feature_importance = np.abs(elasticnet_model.coef_)
feature_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': elasticnet_model.coef_,
    'abs_coefficient': feature_importance
}).sort_values('abs_coefficient', ascending=False)

print("\n特征系数（按绝对值排序）:")
print(feature_importance_df.to_string(index=False))

# 统计被L1正则化压缩为零的特征数量
n_zero_coef = np.sum(elasticnet_model.coef_ == 0)
print(f"\n被L1正则化压缩为零的特征数量: {n_zero_coef}/{len(feature_cols)}")

# ==================== 7. 可视化 ====================
print("\n[8] 生成可视化图表...")

# 创建图表目录
import os
os.makedirs("elasticNet_results", exist_ok=True)

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
plt.savefig('elasticNet_results/1_prediction_scatter.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存: elasticNet_results/1_prediction_scatter.png")
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
plt.savefig('elasticNet_results/2_residuals.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存: elasticNet_results/2_residuals.png")
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
plt.savefig('elasticNet_results/3_distribution_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存: elasticNet_results/3_distribution_comparison.png")
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
plt.savefig('elasticNet_results/4_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存: elasticNet_results/4_metrics_comparison.png")
plt.close()

# 图5: 特征重要性（Elastic Net系数）
fig, ax = plt.subplots(figsize=(10, 6))
indices = np.argsort(feature_importance)[::-1]

colors = ['red' if coef < 0 else 'blue' for coef in elasticnet_model.coef_[indices]]
bars = ax.barh(range(len(feature_cols)), elasticnet_model.coef_[indices], color=colors, alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_yticks(range(len(feature_cols)))
ax.set_yticklabels([feature_cols[i] for i in indices], fontsize=9)
ax.set_xlabel('Coefficient Value', fontsize=12)
ax.set_title('Feature Importance (Elastic Net Coefficients)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# 添加数值标签
for i, (bar, coef) in enumerate(zip(bars, elasticnet_model.coef_[indices])):
    width = bar.get_width()
    if abs(width) > 0.001:  # 只显示非零系数
        ax.text(width if width > 0 else width - 0.02, bar.get_y() + bar.get_height()/2.,
                f'{coef:.3f}',
                ha='left' if width > 0 else 'right', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('elasticNet_results/5_feature_importance.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存: elasticNet_results/5_feature_importance.png")
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
plt.savefig('elasticNet_results/6_error_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存: elasticNet_results/6_error_distribution.png")
plt.close()

# 图7: L1_ratio参数影响分析
print("\n[9] 生成L1_ratio参数影响图...")
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
best_alpha = search.best_params_["elasticnet__alpha"]
train_scores_l1 = []
test_scores_l1 = []

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for l1_ratio in l1_ratios:
    elasticnet_temp = ElasticNet(alpha=best_alpha, l1_ratio=l1_ratio, random_state=42, max_iter=2000)
    elasticnet_temp.fit(X_train_scaled, y_train)
    
    train_pred = elasticnet_temp.predict(X_train_scaled)
    test_pred = elasticnet_temp.predict(X_test_scaled)
    
    train_scores_l1.append(r2_score(y_train, train_pred))
    test_scores_l1.append(r2_score(y_test, test_pred))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(l1_ratios, train_scores_l1, label='Training R²', linewidth=2, marker='o', markersize=8)
ax.plot(l1_ratios, test_scores_l1, label='Test R²', linewidth=2, marker='s', markersize=8)
ax.axvline(x=search.best_params_["elasticnet__l1_ratio"], color='r', linestyle='--', 
           label=f'Best l1_ratio = {search.best_params_["elasticnet__l1_ratio"]}', linewidth=2)
ax.set_xlabel('L1 Ratio (0=Ridge, 1=Lasso)', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title(f'Elastic Net L1 Ratio Effect (alpha={best_alpha})', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(l1_ratios)

plt.tight_layout()
plt.savefig('elasticNet_results/7_l1_ratio_effect.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存: elasticNet_results/7_l1_ratio_effect.png")
plt.close()

# 图8: Alpha参数路径（正则化路径）
print("\n[10] 生成正则化路径图...")
alphas = np.logspace(-2, 2, 50)  # 从0.01到100
best_l1_ratio = search.best_params_["elasticnet__l1_ratio"]
train_scores_alpha = []
test_scores_alpha = []

for alpha in alphas:
    elasticnet_temp = ElasticNet(alpha=alpha, l1_ratio=best_l1_ratio, random_state=42, max_iter=2000)
    elasticnet_temp.fit(X_train_scaled, y_train)
    
    train_pred = elasticnet_temp.predict(X_train_scaled)
    test_pred = elasticnet_temp.predict(X_test_scaled)
    
    train_scores_alpha.append(r2_score(y_train, train_pred))
    test_scores_alpha.append(r2_score(y_test, test_pred))

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(alphas, train_scores_alpha, label='Training R²', linewidth=2, marker='o', markersize=4)
ax.semilogx(alphas, test_scores_alpha, label='Test R²', linewidth=2, marker='s', markersize=4)
ax.axvline(x=best_alpha, color='r', linestyle='--', 
           label=f'Best α = {best_alpha}', linewidth=2)
ax.set_xlabel('Alpha (Regularization Strength)', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title(f'Elastic Net Regularization Path (l1_ratio={best_l1_ratio})', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('elasticNet_results/8_regularization_path.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存: elasticNet_results/8_regularization_path.png")
plt.close()

# ==================== 8. 生成预测结果文件 ====================
print("\n[11] 保存预测结果...")

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
df_train_result.to_csv('elasticNet_results/train_predictions.csv', index=False)
df_test_result.to_csv('elasticNet_results/test_predictions.csv', index=False)
print("  ✓ 保存: elasticNet_results/train_predictions.csv")
print("  ✓ 保存: elasticNet_results/test_predictions.csv")

# ==================== 9. 生成详细报告 ====================
print("\n[12] 生成详细报告...")

report = f"""
{'='*80}
Elastic Net回归模型预测报告
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

正则化系数 (alpha): {search.best_params_['elasticnet__alpha']}
L1比例 (l1_ratio): {search.best_params_['elasticnet__l1_ratio']}
  - l1_ratio = 0: 纯Ridge回归（L2正则化）
  - l1_ratio = 1: 纯Lasso回归（L1正则化）
  - l1_ratio = {search.best_params_['elasticnet__l1_ratio']}: L1和L2混合

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

四、特征重要性分析
---------------
Elastic Net回归系数（按绝对值排序）:
{chr(10).join([f'  {row["feature"]:30s}: {row["coefficient"]:8.4f} (|coef|={row["abs_coefficient"]:.4f})' 
               for _, row in feature_importance_df.iterrows()])}

被L1正则化压缩为零的特征数量: {n_zero_coef}/{len(feature_cols)}
  - L1正则化具有特征选择能力，可以将不重要的特征系数压缩为零
  - 当前模型保留了 {len(feature_cols) - n_zero_coef} 个有效特征

五、误差分析
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

六、预测值统计
-------------
训练集:
  真实值范围: [{y_train.min():.4f}, {y_train.max():.4f}]
  预测值范围: [{y_train_pred.min():.4f}, {y_train_pred.max():.4f}]

测试集:
  真实值范围: [{y_test.min():.4f}, {y_test.max():.4f}]
  预测值范围: [{y_test_pred.min():.4f}, {y_test_pred.max():.4f}]

七、模型诊断
-----------
过拟合检查:
  训练集R² - 测试集R² = {train_r2 - test_r2:.6f}
  {'  ⚠️  警告: 可能存在过拟合' if (train_r2 - test_r2) > 0.1 else '  ✓ 模型泛化能力良好'}

残差正态性:
  训练集残差偏度: {train_residuals.skew():.4f}
  测试集残差偏度: {test_residuals.skew():.4f}
  {'  ⚠️  警告: 残差分布可能非正态' if abs(train_residuals.skew()) > 1 else '  ✓ 残差分布接近正态'}

线性假设检查:
  Elastic Net回归假设特征与目标变量之间存在线性关系
  如果R²较低，可能需要考虑非线性模型（如多项式特征、MLP等）

八、输出文件
-----------
预测结果:
  - elasticNet_results/train_predictions.csv (训练集预测结果)
  - elasticNet_results/test_predictions.csv (测试集预测结果)

可视化图表:
  - elasticNet_results/1_prediction_scatter.png (预测值散点图)
  - elasticNet_results/2_residuals.png (残差图)
  - elasticNet_results/3_distribution_comparison.png (分布对比图)
  - elasticNet_results/4_metrics_comparison.png (指标对比图)
  - elasticNet_results/5_feature_importance.png (特征重要性图)
  - elasticNet_results/6_error_distribution.png (误差分布图)
  - elasticNet_results/7_l1_ratio_effect.png (L1比例影响图)
  - elasticNet_results/8_regularization_path.png (正则化路径图)

九、结论与建议
-------------
1. 模型性能: 
   - 测试集R² = {test_r2:.4f}, 表明模型解释了约{test_r2*100:.1f}%的方差
   - RMSE = {test_rmse:.4f}, 平均预测误差为{test_rmse:.4f}

2. 模型质量:
   {'模型表现良好，预测精度较高' if test_r2 > 0.7 else '模型表现一般，可能需要进一步优化' if test_r2 > 0.5 else '模型表现较差，建议检查数据质量或尝试其他模型'}

3. Elastic Net回归特点:
   - 结合了L1（Lasso）和L2（Ridge）正则化的优势
   - L1正则化具有特征选择能力，可以自动去除不重要的特征
   - L2正则化可以防止过拟合，提高模型稳定性
   - 线性模型，训练速度快，可解释性强
   - 系数可以直接解释特征对目标变量的影响

4. 改进建议:
   - {'考虑增加多项式特征或交互项' if test_r2 < 0.7 else '模型表现良好，可考虑用于实际预测'}
   - {'注意过拟合风险，建议增加正则化强度' if (train_r2 - test_r2) > 0.1 else '模型泛化能力良好'}
   - 可以尝试调整l1_ratio参数，在特征选择和模型稳定性之间找到平衡
   - 可以尝试非线性模型如MLP、SVR等
   - 考虑特征工程，如特征交互、多项式特征等

{'='*80}
报告结束
{'='*80}
"""

with open('elasticNet_results/report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print("\n  ✓ 保存: elasticNet_results/report.txt")

print("\n" + "=" * 60)
print("所有任务完成！")
print("=" * 60)
print(f"\n所有结果已保存到 elasticNet_results/ 目录")
print(f"共生成 {len([f for f in os.listdir('elasticNet_results') if f.endswith('.png')])} 张图表和详细报告")

