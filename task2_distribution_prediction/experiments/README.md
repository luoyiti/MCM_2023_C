# Wordle 难度预测模型库

本目录包含用于 Wordle 单词难度预测的所有模型和工具。

## 📁 目录结构

```
forcasting/
├── __init__.py                    # 模块初始化文件
├── config.py                      # 配置文件（路径、参数）
├── README.md                      # 本文件
│
├── forecasting_models.py          # 回归模型库（统一接口）
├── distribution_models.py         # 分布预测模型库（Softmax）
│
├── lasso_forcasting.py           # Lasso 回归独立脚本
├── ridge_forcasting.py           # Ridge 回归独立脚本
├── elasticNet_forcasting.py      # ElasticNet 回归独立脚本
├── mlp_forcasting.py             # MLP 回归独立脚本
├── randomForest_forcasting.py    # RandomForest 回归独立脚本
├── tabNet_forcasting.py          # TabNet 回归独立脚本
├── softMax_forcasting.py         # Softmax 分布预测独立脚本
│
└── forcasting.ipynb              # Jupyter Notebook（统一实验）
```

## 🚀 快速开始

### 1. 作为模块使用

从项目根目录导入：

```python
from forcasting import (
    load_data,
    train_lasso,
    train_ridge,
    train_elasticnet,
    train_mlp,
    train_randomforest,
    train_tabnet,
    load_distribution_data,
    train_and_evaluate_distribution_model
)

# 训练回归模型
X_train, y_train, X_test, y_test = load_data()
lasso_results = train_lasso(X_train, y_train, X_test, y_test, cv_splits=5)

# 训练分布预测模型
data_dict = load_distribution_data()
linear_results = train_and_evaluate_distribution_model('linear', data_dict)
```

### 2. 运行独立脚本

从项目根目录运行：

```bash
# 运行 Lasso 回归
python -m forcasting.lasso_forcasting

# 运行分布预测
python -m forcasting.softMax_forcasting
```

### 3. 使用 Jupyter Notebook

打开 `forcasting.ipynb` 进行交互式实验和模型对比。

## 📊 支持的模型

### 回归模型（预测 autoencoder_value）

| 模型 | 文件 | 特点 |
|------|------|------|
| **Lasso** | `lasso_forcasting.py` | L1 正则化，特征选择 |
| **Ridge** | `ridge_forcasting.py` | L2 正则化，处理多重共线性 |
| **ElasticNet** | `elasticNet_forcasting.py` | L1+L2 正则化 |
| **MLP** | `mlp_forcasting.py` | 多层感知机，捕捉非线性 |
| **RandomForest** | `randomForest_forcasting.py` | 集成学习，特征重要性 |
| **TabNet** | `tabNet_forcasting.py` | 深度学习表格模型 |

### 分布预测模型（预测 7 维概率分布）

| 模型 | 文件 | 特点 |
|------|------|------|
| **Linear-Softmax** | `softMax_forcasting.py` | 线性分类器 |
| **MLP-Softmax** | `distribution_models.py` | 深度神经网络分类器 |

## 🔧 配置说明

所有路径和参数配置在 `config.py` 中：

- **数据路径**：`DATA_DIR`, `TRAIN_DATA`, `TEST_DATA`
- **输出路径**：各模型的 `*_RESULTS` 目录
- **模型参数**：`RANDOM_STATE`, `CV_FOLDS`
- **特征列**：`DEFAULT_FEATURE_COLS`, `DIST_FEATURE_COLS`

## 📈 输出结果

每个模型运行后会生成：

1. **预测结果 CSV**：`train_predictions.csv`, `test_predictions.csv`
2. **可视化图表**：
   - 预测散点图
   - 残差图
   - 特征重要性
   - 误差分布
3. **文本报告**：`report.txt` 包含详细评估指标

## 🛠️ 依赖库

```bash
# 核心依赖
numpy
pandas
scikit-learn
matplotlib
seaborn

# 深度学习模型
torch
pytorch-tabnet

# Jupyter
jupyter
ipykernel
```

## 📝 代码规范

- 所有路径使用 `config.py` 中的常量
- 函数包含完整的 docstring
- 使用类型提示（Type Hints）
- 遵循 PEP 8 代码风格

## 🔄 更新日志

**v1.0.0** (2025-12-15)
- ✅ 重构所有模型到统一模块
- ✅ 添加配置文件管理路径
- ✅ 优化代码结构和可维护性
- ✅ 添加完整文档

## 👥 作者

MCM 2023 Team C

## 📄 许可证

本项目仅用于学术研究和竞赛。
