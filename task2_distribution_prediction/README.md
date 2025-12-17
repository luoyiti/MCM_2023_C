# Task 2: 分布预测 (Distribution Prediction)

## 任务目标

预测特定 Wordle 单词（EERIE）在 2023-03-01 的尝试次数分布。

## 目录结构

```
task2_distribution_prediction/
├── experiments/          # 探索性分析
│   ├── lasso_analysis.py
│   ├── random_forest_exploration.py
│   └── ...
├── models/              # 实际解决方案 ⭐
│   ├── train_rf_model.py
│   ├── ensemble_models.py
│   └── ...
├── compute_eerie_features.py  # 计算 EERIE 的特征
├── predict_eerie.py           # 主预测脚本
└── AutoEncoder.ipynb          # AutoEncoder 实验（已迁移到 feature_engineering/）
```

## 解题思路

### 1. 特征工程
使用 `feature_engineering/` 模块提取的单词属性：
- 字母特征：`letter_entropy`, `letter_freq_mean`, `num_rare_letters`
- 模拟特征：`mean_simulate_freq`, `mean_simulate_random`
- 强化学习特征：`rl_*_try_*`
- 降维特征：`autoencoder_value`（从 AutoEncoder 获得）

### 2. 模型训练（`models/`）
- **Random Forest Regressor**: 预测每个类别（1-7+ tries）的百分比
- **Bootstrap 不确定性估计**: 生成 95% 置信区间
- **特征重要性分析**: 识别关键预测因子

### 3. 实验探索（`experiments/`）
- **LASSO 特征选择**: 减少特征冗余
- **多模型对比**: XGBoost, LightGBM, Neural Networks
- **超参数调优**: Grid Search, Bayesian Optimization

## 运行方式

### 预测 EERIE 分布
```bash
cd task2_distribution_prediction
python predict_eerie.py
```

输出：
- `results/task2/eerie_distribution.csv`: 预测分布
- `results/task2/eerie_full_report.txt`: 详细报告
- `pictures/task2/eerie_visualization.png`: 可视化图表

### 训练新模型
```bash
cd task2_distribution_prediction/models
python train_rf_model.py
```

## 核心文件说明

- **`compute_eerie_features.py`**: 计算 EERIE 的特征向量
- **`predict_eerie.py`**: 加载模型，预测分布，生成报告
- **`models/train_rf_model.py`**: 训练 Random Forest 模型
- **`experiments/`**: 各种实验脚本（Lasso, XGBoost, etc.）

## 依赖项

```bash
# feature_engineering/ 模块（独立）
# shared/ 模块（路径配置）
pip install scikit-learn pandas numpy
```

## 结果示例

```
EERIE (2023-03-01) 预测分布:
  1 try:    0.38%
  2 tries:  3.78%
  3 tries: 17.98%
  4 tries: 31.62% ← 峰值
  5 tries: 27.51%
  6 tries: 15.34%
  7+ tries: 3.39%

期望尝试次数: 4.48
成功率: 96.6%
```

## 注意事项

⚠️ `AutoEncoder.ipynb` 保留在此目录用于参考，实际 AutoEncoder 代码在 `feature_engineering/`
⚠️ `experiments/` 中的脚本为探索性分析，不直接用于最终预测
⚠️ 实际解决方案在 `models/` 目录中
