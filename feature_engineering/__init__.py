"""
特征工程模块（Feature Engineering）

本模块包含用于 Wordle 数据特征提取和转换的工具：
- wordle_game_simulate.py: 模拟 Wordle 游戏获取反馈熵特征
- feedbackEntropy.py: 计算反馈熵
- AutoEncoder.ipynb: 使用自动编码器进行降维
- reinforcement_learning_wordle_game.py: 强化学习模拟

这些特征被用于 Task 1（时间序列预测）和 Task 2（分布预测）。
"""

__all__ = [
    'wordle_game_simulate',
    'feedbackEntropy',
]
