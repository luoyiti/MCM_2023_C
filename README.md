# MCM 2023 Problem C - Wordle 数据分析与预测

> **2023 数学建模竞赛 (MCM) Problem C: Predicting Wordle Results**
> 
> 本项目提供完整的时间序列预测、单词属性分析和成绩分布预测解决方案。

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📁 项目结构（重构版 - 2025-12-16）

```
MCM_2023_C/
├── 📊 data/                      # 数据文件
│   ├── mcm_processed_data.csv   # ⭐ 主数据（358行 × 92列特征）
│   └── ...
│
├── 🔮 task1_reporting_volume/    # 任务1：报告人数预测 + Hard Mode分析
│   ├── q1_final_clean.py        # SARIMA 时间序列集成模型
│   ├── run_task1.py             # 🚀 任务1主入口
│   ├── analysis_hard_mode.py    # Hard Mode 影响因素分析
│   └── style_utils.py           # 可视化工具
│
├── 🎯 task2_distribution_prediction/  # 任务2：EERIE 成绩分布预测
│   ├── predict_eerie.py         # 🚀 任务2主入口（Random Forest）
│   ├── compute_eerie_features.py # 单词特征计算
│   ├── feature_engineering/      # 特征工程模块
│   ├── experiments/             # 实验性模型（ElasticNet, MLP等）
│   └── models/                  # 保存的模型文件
│
├── 🔧 shared/                    # 共享配置和工具
│   ├── config.py                # 统一路径配置
│   └── data_loader.py           # 数据加载工具
│
├── 📈 results/                   # 输出结果
│   ├── task1/                   # 任务1输出（报告、模型）
│   └── task2/                   # 任务2输出（预测结果）
│
├── 📸 pictures/                  # 可视化图表
│   ├── task1/                   # 任务1图表（诊断、因素重要性等）
│   └── task2/                   # 任务2图表（分布对比等）
│
├── 🗄️ backups/                   # 备份数据
│   └── 2023_MCM_Problem_C_Data.xlsx  # 原始Excel（归一化百分比）
│
├── 🏃 run_task1.sh              # 任务1快速运行脚本
├── 🏃 run_task2.sh              # 任务2快速运行脚本
└── 📋 requirements.txt          # Python依赖
```

---

## 🚀 快速开始

### 环境设置

```bash
# 创建 conda 环境（推荐 Python 3.11）
conda create -n mcm2023 python=3.11 -y
conda activate mcm2023

# 安装依赖
pip install -r requirements.txt
```

### 一键运行

```bash
# 任务1：预测 2023-03-01 报告人数 + Hard Mode 分析
./run_task1.sh

# 任务2：预测 EERIE 的成绩分布
./run_task2.sh
```

**输出**：
- 📄 CSV/TXT 结果 → `results/task1/` 或 `results/task2/`
- 📊 PNG 图表 → `pictures/task1/` 或 `pictures/task2/`

---

## 🎯 题目要求与解决方案

| 题目要求 | 解决方案 | 实现文件 |
|---------|---------|---------|
| **Q1a**: 预测 2023-03-01 报告人数（含置信区间） | SARIMA 时间序列集成 + 变点检测 | `task1_reporting_volume/q1_final_clean.py` |
| **Q1b**: 分析单词属性对 Hard Mode 的影响 | OLS + Lasso + 滞后特征分析 | `task1_reporting_volume/analysis_hard_mode.py` |
| **Q2**: 预测 EERIE 的 1-7 次猜中分布 | Random Forest（79特征） | `task2_distribution_prediction/predict_eerie.py` |

---

## 🔬 核心技术方案

### 📊 任务1：时间序列预测（报告人数）

**关键发现**：
- 🔴 **变点检测**：2022-03-18 出现结构性断裂，报告人数从 24.9万/天 → 5.2万/天（下降 78.9%）
- 📈 **预测结果**：2023-03-01 点预测 **19,807 人**，90% CI: [14,252, 27,782]

**技术栈**：
```python
✓ 变点检测 (PELT)          # 检测趋势突变
✓ SARIMA(1,1,2)x(1,0,1,7)  # 捕捉周周期性（7天）
✓ 滚动交叉验证              # 避免数据泄露
✓ 集成学习 (IVW)           # 逆方差加权
✓ Duan Smearing            # 对数回变换修正
```

**输出文件**：
- `results/task1/explanation_report.txt` - 解释性报告
- `results/task1/diagnostic_report.txt` - 模型诊断
- `pictures/task1/1_weekday_effects.png` - 工作日效应
- `pictures/task1/2_changepoint.png` - 变点可视化
- `pictures/task1/3_diagnostics.png` - 残差诊断

### 🎯 任务1b：Hard Mode 影响因素

**关键发现**：
- 📌 **滞后效应占主导**：前2-3天的 Hard Mode 比例贡献 **98%+ 重要性**
- 🔤 **单词属性影响微弱**：OLS R² = 0.23，Lasso 仅保留 20/79 特征

**Top 5 重要特征**：
1. `hard_mode_ratio_lag2` (39.5%) - 前2天比例
2. `hard_mode_ratio_lag3` (29.6%) - 前3天比例
3. `hard_mode_ratio_lag1` (28.5%) - 前1天比例
4. `position_self_entropy` (0.27%) - 位置自熵
5. `1_try_simulate_random` (0.25%) - 模拟成功率

**输出文件**：
- `pictures/task1/Feature_Importance_Hard_Mode_Ratio_Lag_vs_Attributes.png`

### 🎲 任务2：成绩分布预测（EERIE）

**数据驱动**：
- 📊 **训练数据**：358 个单词 × 79 个特征
- 🎯 **预测目标**：7 个类别（1-6 tries + 7+ tries）

**特征工程**（79 维）：
```
字母结构特征: num_rare_letters, has_double_letter, max_consecutive_vowels...
词频特征: Zipf-value, letter_freq_mean, positional_freq_mean...
熵特征: letter_entropy, feedback_entropy, position_self_entropy...
语义特征: semantic_distance, semantic_neighbors_count...
仿真特征: *_simulate_random, *_simulate_freq, *_simulate_entropy...
强化学习: rl_*_try_*, rl_expected_steps_*...
```

**模型选择**：Random Forest（基于实验对比选出）

**输出文件**：
- `results/task2/eerie_prediction.csv` - EERIE 预测结果
- `pictures/task2/eerie_distribution.png` - 分布对比图

---

## 📊 数据说明

### ⭐ 核心数据：`data/mcm_processed_data.csv`

| 类型 | 列数 | 说明 |
|-----|-----|------|
| 基础信息 | 3 | `date`, `word`, `contest_number` |
| 报告人数 | 2 | `number_of_reported_results`, `number_in_hard_mode` |
| 真实分布 | 7 | `1_try` ~ `7_or_more_tries_x` |
| 单词特征 | 79 | 字母结构、词频、熵、语义、仿真、RL... |

**⚠️ 重要**：
- ✅ **CSV 文件**包含真实报告人数（几万人规模）
- ❌ **Excel 文件**（`backups/2023_MCM_Problem_C_Data.xlsx`）是归一化的百分比数据（0-100）

---

## 🛠️ 技术栈

### 核心依赖

```python
pandas>=2.0.0         # 数据处理
numpy>=1.24.0         # 数值计算
matplotlib>=3.7.0     # 绘图
seaborn>=0.12.0       # 统计可视化
scikit-learn>=1.3.0   # 机器学习
statsmodels>=0.14.0   # 统计模型（SARIMA）
ruptures>=1.1.0       # 变点检测
holidays>=0.34        # 节假日数据
wordfreq>=3.0         # 词频统计
nltk>=3.8             # NLP 工具
```

### 可选依赖

```python
torch>=2.0.0          # 深度学习（用于 MoE 实验）
xgboost>=2.0.0        # 梯度提升（用于对比实验）
```

---

## 📖 详细使用说明

### 任务1：报告人数预测

```bash
# 方法1：使用 shell 脚本（推荐）
./run_task1.sh

# 方法2：直接运行 Python
cd task1_reporting_volume
conda run -n mcm2023 python run_task1.py
```

**输出详情**：
1. **文本报告**（`results/task1/`）：
   - `explanation_report.txt` - 变点、周末效应、节假日效应分析
   - `diagnostic_report.txt` - 残差诊断、模型性能指标
   
2. **可视化图表**（`pictures/task1/`）：
   - `1_weekday_effects.png` - 工作日 vs 周末报告人数对比
   - `2_changepoint.png` - 变点可视化（2022-03-18）
   - `3_diagnostics.png` - 残差分析、ACF、覆盖率验证
   - `4_factor_importance.png` - 因素重要性（波动性 97.7%）
   - `Feature_Importance_Hard_Mode_Ratio_Lag_vs_Attributes.png` - Hard Mode 分析

3. **模型文件**（`results/task1/`）：
   - `ensemble_result.pkl` - 集成模型（可用于后续预测）

### 任务2：EERIE 分布预测

```bash
# 方法1：使用 shell 脚本（推荐）
./run_task2.sh

# 方法2：直接运行 Python
cd task2_distribution_prediction
conda run -n mcm2023 python predict_eerie.py
```

**输出详情**：
1. **预测结果**（`results/task2/`）：
   - `eerie_prediction.csv` - EERIE 的 1-7 次分布概率
   
2. **可视化图表**（`pictures/task2/`）：
   - `eerie_distribution.png` - 预测分布 vs 平均分布对比

---

## 🔍 项目亮点

### ✨ 方法创新

1. **变点检测 + 分段建模**
   - 使用 PELT 算法自动检测时间序列的结构性变化
   - 避免数据泄露：滚动 CV 中每一折独立检测变点

2. **集成学习策略**
   - 多个 SARIMA 模型通过逆方差加权集成
   - 在 log 空间合并预测区间，提高覆盖率准确性

3. **惯性效应发现**
   - Hard Mode 使用具有强时间惯性（滞后效应占 98%+）
   - 单词属性对当天 Hard Mode 比例影响微弱

### 📊 数据工程

1. **特征工程完善**
   - 79 维单词特征涵盖字母、词频、熵、语义、仿真、RL
   - 自动化特征计算流程

2. **数据质量保障**
   - 识别并修复 Excel 归一化数据问题
   - 使用 CSV 真实数据进行建模

### 🎯 可解释性

1. **自动报告生成**
   - 变点位置、原因分析
   - 周末效应、节假日效应量化
   - 模型性能诊断（残差、覆盖率）

2. **可视化完整**
   - 每个分析步骤都有对应图表
   - 图表风格统一，信息清晰

---

## 📝 重要说明

### ⚠️ 数据格式警告

- **Excel 文件**（`backups/2023_MCM_Problem_C_Data.xlsx`）：
  - 包含的是**归一化的百分比数据**（0-100）
  - **不是**真实的报告人数
  - 主要用于特征列的获取

- **CSV 文件**（`data/mcm_processed_data.csv`）：
  - 包含**真实的报告人数**（几万人规模）
  - 包含完整的 79 维单词特征
  - **所有建模都基于此文件**

### 📂 文件组织

- **结果文件**统一存放在 `results/task1/` 和 `results/task2/`
- **图表文件**统一存放在 `pictures/task1/` 和 `pictures/task2/`
- **不再有**子文件夹下的重复 `results/` 目录

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 License

MIT License

---

## 📧 联系方式

如有问题，请通过 GitHub Issues 联系。

---

**最后更新**: 2025-12-16  
**项目状态**: ✅ 生产就绪
