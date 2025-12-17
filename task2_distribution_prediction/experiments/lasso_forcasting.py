"""
LASSO回归模型：预测autoencoder_value
使用reduced_features_train.csv作为训练集，reduced_features_test.csv作为测试集
LASSO（L1正则化）具备特征选择能力
"""

import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# 字体与样式
plt.rcParams["font.family"] = "Heiti TC"
sns.set_style("whitegrid")
sns.set_palette("husl")

print("=" * 60)
print("LASSO回归模型训练与评估")
print("=" * 60)

# 1) 数据加载
print("\n[1] 加载数据...")
df_train = pd.read_csv("data/reduced_features_train.csv")
df_test = pd.read_csv("data/reduced_features_test.csv")
print(f"训练集样本数: {len(df_train)}")
print(f"测试集样本数: {len(df_test)}")

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

X_train = df_train[feature_cols].copy()
y_train = df_train[target_col].copy()
X_test = df_test[feature_cols].copy()
y_test = df_test[target_col].copy()

# 2) 缺失值处理
print("\n[2] 数据预处理...")
print(f"训练集缺失值: {X_train.isnull().sum().sum()}")
print(f"测试集缺失值: {X_test.isnull().sum().sum()}")
if X_train.isnull().sum().sum() > 0:
    X_train = X_train.fillna(X_train.median(numeric_only=True))
if X_test.isnull().sum().sum() > 0:
    X_test = X_test.fillna(X_train.median(numeric_only=True))

# 3) 建立流水线
print("\n[3] 构建LASSO回归模型...")
pipe = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("lasso", Lasso(random_state=42, max_iter=5000)),
    ]
)

# 4) 网格搜索
print("\n[4] 网格搜索最优超参数...")
print("（LASSO训练较快，请稍候...）")
param_grid = {
    "lasso__alpha": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="r2",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train)
print(f"\n最优参数: {search.best_params_}")
print(f"最优交叉验证R²: {search.best_score_:.4f}")
best_model = search.best_estimator_

# 5) 模型评估
print("\n[5] 模型评估...")
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\n" + "=" * 60)
print("评估结果")
print("=" * 60)
print("\n训练集:")
print(f"  R²  = {train_r2:.4f}")
print(f"  RMSE = {train_rmse:.4f}")
print(f"  MAE  = {train_mae:.4f}")
print("\n测试集:")
print(f"  R²  = {test_r2:.4f}")
print(f"  RMSE = {test_rmse:.4f}")
print(f"  MAE  = {test_mae:.4f}")

# 6) 交叉验证复算
print("\n[6] 5折交叉验证评估...")
cv_scores = []
for tr_idx, val_idx in cv.split(X_train):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
    model_cv = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha=search.best_params_["lasso__alpha"], random_state=42, max_iter=5000)),
        ]
    )
    model_cv.fit(X_tr, y_tr)
    y_val_pred = model_cv.predict(X_val)
    cv_scores.append(r2_score(y_val, y_val_pred))
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)
print(f"交叉验证 R²: {cv_mean:.4f} ± {cv_std:.4f}")

# 7) 特征重要性（LASSO系数，可稀疏）
print("\n[7] 特征重要性分析...")
lasso_model = best_model.named_steps["lasso"]
coef = lasso_model.coef_
feature_importance = np.abs(coef)
feature_importance_df = (
    pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": coef,
            "abs_coefficient": feature_importance,
        }
    )
    .sort_values("abs_coefficient", ascending=False)
)
print("\n特征系数（按绝对值排序）:")
print(feature_importance_df.to_string(index=False))
n_zero = np.sum(coef == 0)
print(f"\n被L1正则化压缩为零的特征数量: {n_zero}/{len(feature_cols)}")

# 8) 可视化
print("\n[8] 生成可视化图表...")
os.makedirs("lasso_results", exist_ok=True)

# 8.1 预测散点
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(y_train, y_train_pred, alpha=0.6, s=50, edgecolors="black", linewidth=0.5)
mn, mx = min(y_train.min(), y_train_pred.min()), max(y_train.max(), y_train_pred.max())
axes[0].plot([mn, mx], [mn, mx], "r--", lw=2)
axes[0].set_title(f"Training Set\nR²={train_r2:.4f}, RMSE={train_rmse:.4f}", fontsize=13, fontweight="bold")
axes[0].set_xlabel("True autoencoder_value")
axes[0].set_ylabel("Predicted autoencoder_value")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test, y_test_pred, alpha=0.6, s=50, edgecolors="black", linewidth=0.5, color="green")
mn, mx = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
axes[1].plot([mn, mx], [mn, mx], "r--", lw=2)
axes[1].set_title(f"Test Set\nR²={test_r2:.4f}, RMSE={test_rmse:.4f}", fontsize=13, fontweight="bold")
axes[1].set_xlabel("True autoencoder_value")
axes[1].set_ylabel("Predicted autoencoder_value")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lasso_results/1_prediction_scatter.png", dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ 保存: lasso_results/1_prediction_scatter.png")

# 8.2 残差
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(y_train_pred, train_residuals, alpha=0.6, s=50, edgecolors="black", linewidth=0.5)
axes[0].axhline(0, color="r", ls="--", lw=2)
axes[0].set_title("Training Set Residuals", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Residuals")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test_pred, test_residuals, alpha=0.6, s=50, edgecolors="black", linewidth=0.5, color="green")
axes[1].axhline(0, color="r", ls="--", lw=2)
axes[1].set_title("Test Set Residuals", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Residuals")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lasso_results/2_residuals.png", dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ 保存: lasso_results/2_residuals.png")

# 8.3 分布对比
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].hist(y_train, bins=30, alpha=0.6, label="True", color="blue", edgecolor="black")
axes[0].hist(y_train_pred, bins=30, alpha=0.6, label="Predicted", color="orange", edgecolor="black")
axes[0].set_title("Training Distribution", fontsize=13, fontweight="bold")
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].hist(y_test, bins=20, alpha=0.6, label="True", color="blue", edgecolor="black")
axes[1].hist(y_test_pred, bins=20, alpha=0.6, label="Predicted", color="green", edgecolor="black")
axes[1].set_title("Test Distribution", fontsize=13, fontweight="bold")
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("lasso_results/3_distribution_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ 保存: lasso_results/3_distribution_comparison.png")

# 8.4 指标对比
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ["R²", "RMSE", "MAE"]
train_vals = [train_r2, train_rmse, train_mae]
test_vals = [test_r2, test_rmse, test_mae]
x = np.arange(len(metrics))
w = 0.35
bars1 = ax.bar(x - w / 2, train_vals, w, label="Train", edgecolor="black", alpha=0.8)
bars2 = ax.bar(x + w / 2, test_vals, w, label="Test", edgecolor="black", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_title("Metrics Comparison", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
for bars in (bars1, bars2):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("lasso_results/4_metrics_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ 保存: lasso_results/4_metrics_comparison.png")

# 8.5 特征重要性
fig, ax = plt.subplots(figsize=(10, 6))
idx = np.argsort(feature_importance)[::-1]
colors = ["red" if c < 0 else "blue" for c in coef[idx]]
bars = ax.barh(range(len(feature_cols)), coef[idx], color=colors, alpha=0.7, edgecolor="black")
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(range(len(feature_cols)))
ax.set_yticklabels([feature_cols[i] for i in idx], fontsize=9)
ax.set_xlabel("Coefficient")
ax.set_title("Feature Importance (LASSO)", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")
for b, c in zip(bars, coef[idx]):
    if abs(c) > 0.001:
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f"{c:.3f}", ha="left" if c > 0 else "right", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("lasso_results/5_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ 保存: lasso_results/5_feature_importance.png")

# 8.6 误差分布
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].hist(train_residuals, bins=30, alpha=0.7, color="blue", edgecolor="black")
axes[0].axvline(0, color="r", ls="--", lw=2)
axes[0].set_title(f"Train Error Dist\nMean={train_residuals.mean():.4f}, Std={train_residuals.std():.4f}", fontsize=13, fontweight="bold")
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].hist(test_residuals, bins=20, alpha=0.7, color="green", edgecolor="black")
axes[1].axvline(0, color="r", ls="--", lw=2)
axes[1].set_title(f"Test Error Dist\nMean={test_residuals.mean():.4f}, Std={test_residuals.std():.4f}", fontsize=13, fontweight="bold")
axes[1].grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("lasso_results/6_error_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ 保存: lasso_results/6_error_distribution.png")

# 8.7 Alpha路径
print("\n[9] 生成正则化路径图...")
alphas = np.logspace(-4, 1, 40)  # 0.0001~10
scaler_path = StandardScaler()
X_train_scaled = scaler_path.fit_transform(X_train)
X_test_scaled = scaler_path.transform(X_test)
train_scores_alpha = []
test_scores_alpha = []
best_alpha = search.best_params_["lasso__alpha"]
for a in alphas:
    model_temp = Lasso(alpha=a, random_state=42, max_iter=5000)
    model_temp.fit(X_train_scaled, y_train)
    train_scores_alpha.append(r2_score(y_train, model_temp.predict(X_train_scaled)))
    test_scores_alpha.append(r2_score(y_test, model_temp.predict(X_test_scaled)))
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(alphas, train_scores_alpha, label="Training R²", marker="o", markersize=4)
ax.semilogx(alphas, test_scores_alpha, label="Test R²", marker="s", markersize=4)
ax.axvline(best_alpha, color="r", ls="--", label=f"Best α={best_alpha}")
ax.set_xlabel("Alpha (L1 Strength)")
ax.set_ylabel("R² Score")
ax.set_title("LASSO Regularization Path", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lasso_results/7_regularization_path.png", dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ 保存: lasso_results/7_regularization_path.png")

# 9) 结果文件
print("\n[10] 保存预测结果...")
df_train_res = df_train.copy()
df_train_res["autoencoder_value_pred"] = y_train_pred
df_train_res["residual"] = train_residuals
df_train_res["abs_error"] = np.abs(train_residuals)

df_test_res = df_test.copy()
df_test_res["autoencoder_value_pred"] = y_test_pred
df_test_res["residual"] = test_residuals
df_test_res["abs_error"] = np.abs(test_residuals)

df_train_res.to_csv("lasso_results/train_predictions.csv", index=False)
df_test_res.to_csv("lasso_results/test_predictions.csv", index=False)
print("  ✓ 保存: lasso_results/train_predictions.csv")
print("  ✓ 保存: lasso_results/test_predictions.csv")

# 10) 报告
print("\n[11] 生成详细报告...")
importance_lines = "\n".join(
    [
        f"  {row['feature']:30s}: {row['coefficient']:8.4f} (|coef|={row['abs_coefficient']:.4f})"
        for _, row in feature_importance_df.iterrows()
    ]
)
report = f"""
{'='*80}
LASSO回归模型预测报告
{'='*80}

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

一、数据概况
-----------
训练集样本数: {len(df_train)}
测试集样本数: {len(df_test)}
特征数量: {len(feature_cols)}
目标变量: {target_col}

特征列表:
{chr(10).join([f'  {i+1}. {c}' for i, c in enumerate(feature_cols)])}

二、模型配置
-----------
最优超参数:
{chr(10).join([f'  {k}: {v}' for k, v in search.best_params_.items()])}

正则化系数 (alpha): {best_alpha}
  - L1正则化可将不重要的特征系数压缩为零，具备特征选择能力

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
LASSO回归系数（按绝对值排序）:
{importance_lines}

被L1正则化压缩为零的特征数量: {n_zero}/{len(feature_cols)}
  - 当前模型保留了 {len(feature_cols) - n_zero} 个有效特征

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
  LASSO回归假设特征与目标变量之间存在线性关系
  如果R²较低，可能需要考虑非线性模型或特征工程

八、输出文件
-----------
预测结果:
  - lasso_results/train_predictions.csv (训练集预测结果)
  - lasso_results/test_predictions.csv (测试集预测结果)

可视化图表:
  - lasso_results/1_prediction_scatter.png (预测值散点图)
  - lasso_results/2_residuals.png (残差图)
  - lasso_results/3_distribution_comparison.png (分布对比图)
  - lasso_results/4_metrics_comparison.png (指标对比图)
  - lasso_results/5_feature_importance.png (特征重要性图)
  - lasso_results/6_error_distribution.png (误差分布图)
  - lasso_results/7_regularization_path.png (正则化路径图)

九、结论与建议
-------------
1. 模型性能:
   - 测试集R² = {test_r2:.4f}, 约{test_r2*100:.1f}%方差被解释
   - RMSE = {test_rmse:.4f}

2. 模型质量:
   {'模型表现良好，预测精度较高' if test_r2 > 0.7 else '模型表现一般，可能需要进一步优化' if test_r2 > 0.5 else '模型表现较差，建议检查数据或尝试其他模型'}

3. LASSO特点:
   - 具备特征选择能力，可稀疏化系数
   - 适合高维或存在冗余特征的场景
   - 如果存在多重共线，L1可能不稳定，可考虑Elastic Net

4. 改进建议:
   - 调整alpha以获得更稀疏/更平滑的模型
   - 尝试增加特征工程（多项式、交互项）
   - 若R²不足，可考虑Elastic Net或非线性模型（MLP、SVR）

{'='*80}
报告结束
{'='*80}
"""

with open("lasso_results/report.txt", "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print("\n  ✓ 保存: lasso_results/report.txt")

print("\n" + "=" * 60)
print("所有任务完成！")
print("=" * 60)
print(f"\n所有结果已保存到 lasso_results/ 目录")
print(f"共生成 {len([f for f in os.listdir('lasso_results') if f.endswith('.png')])} 张图表和详细报告")
