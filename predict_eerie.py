import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def run_prediction(df, feature_cols, target_word="EERIE"):
    """
    专门用于预测特定单词（如 EERIE）的猜测分布
    """
    print("\n" + "="*50)
    print(f">>> Final Task: Predicting Distribution for '{target_word}'")
    print("="*50)
    
    # 1. 准备多输出回归数据 (Multi-Output Regression Data)
    # 我们需要同时预测 7 个目标值
    dist_cols = ['1_try', '2_tries', '3_tries', '4_tries', '5_tries', '6_tries', '7_or_more_tries_x']
    
    # 确保没有空值
    valid_idx = df[dist_cols].dropna().index
    X = df.loc[valid_idx, feature_cols]
    y = df.loc[valid_idx, dist_cols]
    
    # 2. 训练全量随机森林模型
    # 使用所有可用数据进行训练，以获得最佳的泛化能力
    print("Training Multi-Output Random Forest on full dataset...")
    rf_final = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_final.fit(X, y)
    
    # 3. 获取目标单词的特征向量
    # 自动处理列名大小写问题 (Word vs word)
    word_col = 'Word' if 'Word' in df.columns else 'word'
    
    if target_word in df[word_col].values:
        print(f"Target word '{target_word}' found in dataset. Using its actual features.")
        target_features = df.loc[df[word_col] == target_word, feature_cols]
    else:
        # 如果数据集中没有这个词，我们用第一行数据作为“替身”演示流程
        # 在实际比赛中，你应该手动构造一个包含 EERIE 特征的 DataFrame 传进来
        print(f"⚠️ Warning: '{target_word}' not found in data. Using the first row's features as a DEMO.")
        target_features = df.iloc[[0]][feature_cols]
        
    # 4. 执行预测
    pred_distribution = rf_final.predict(target_features)[0]
    
    # 5. 强制归一化 (Sum to 100%)
    # 随机森林输出的原始值之和可能在 0.99-1.01 之间，这里强制压缩到 1.0
    raw_sum = pred_distribution.sum()
    pred_distribution = pred_distribution / raw_sum
    
    # 验证归一化结果
    print(f"归一化前总和: {raw_sum:.4f} -> 归一化后总和: {pred_distribution.sum():.4f}")
    
    # 6. 格式化输出预测报告
    labels = ['1 Try', '2 Tries', '3 Tries', '4 Tries', '5 Tries', '6 Tries', 'Failed (X)']
    print(f"\n[Prediction Result] Estimated Distribution for '{target_word}':")
    print("-" * 40)
    for label, value in zip(labels, pred_distribution):
        # 输出格式：标签 : 百分比 (小数)
        print(f"{label:<10}: {value:.4%} ({value:.4f})")
    print("-" * 40)
    
    return pred_distribution