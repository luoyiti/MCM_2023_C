# 特征工程模块 (Feature Engineering)

## 概述

本模块包含用于 Wordle 单词属性特征提取和转换的核心工具，为 Task 1 和 Task 2 提供统一的特征工程支持。

## 文件说明

### 核心脚本
- **`wordle_game_simulate.py`**: 模拟 Wordle 游戏，计算基于模拟的特征
  - `mean_simulate_freq`: 基于频率策略的平均尝试次数
  - `mean_simulate_random`: 基于随机策略的平均尝试次数
  - `*_try_simulate_*`: 各次数尝试的分布

- **`feedbackEntropy.py`**: 计算 Wordle 反馈熵
  - 衡量单词给出的反馈信息量
  - 用于评估单词的"信息价值"

- **`reinforcement_learning_wordle_game.py`**: 强化学习 Wordle 策略
  - 训练 RL 代理学习最优猜测策略
  - 生成 `rl_*_try_*` 特征

### Notebooks
- **`AutoEncoder.ipynb`**: 自动编码器降维
  - 将高维特征压缩为低维表示
  - 用于减少特征冗余，提升模型性能

- **`featureEngineering.ipynb`**: 特征工程探索
  - 特征分析和可视化
  - 新特征创建实验

## 使用方式

### Task 1: 时间序列预测
Task 1 使用以下滞后特征（前一天单词属性）：
```python
lag_features = [
    'mean_simulate_freq',
    'letter_entropy', 
    'mean_simulate_random',
    'has_common_suffix',
    'letter_freq_mean'
]
```

### Task 2: 分布预测
Task 2 使用完整的单词属性特征集，包括：
- 模拟特征 (`*_simulate_*`)
- 字母属性 (`num_rare_letters`, `letter_entropy`, etc.)
- 强化学习特征 (`rl_*_try_*`)
- AutoEncoder 降维特征 (`autoencoder_value`)

## 特征生成流程

```
原始单词列表
    ↓
wordle_game_simulate.py → 模拟特征
    ↓
feedbackEntropy.py → 反馈熵
    ↓
reinforcement_learning_wordle_game.py → RL 特征
    ↓
AutoEncoder.ipynb → 降维特征
    ↓
最终特征集 (mcm_processed_data.csv)
```

## 依赖项

```bash
pip install numpy pandas scikit-learn tensorflow
```

## 注意事项

⚠️ 本模块独立于 Task 1 和 Task 2，避免循环依赖
⚠️ AutoEncoder 模型保存在 `models/autoencoder_model.pkl`
⚠️ 特征计算可能需要较长时间（尤其是模拟和 RL）
