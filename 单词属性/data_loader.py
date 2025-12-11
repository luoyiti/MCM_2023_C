import pandas as pd
import numpy as np
import os

def load_and_clean_data(filepath):
    """加载并清洗数据 (支持 .csv 和 .xlsx)"""
    print(f"正在读取文件: {filepath} ...")
    
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext == '.xlsx' or file_ext == '.xls':
        df = pd.read_excel(filepath)
    else:
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 解码失败，尝试使用 GBK 编码读取...")
            df = pd.read_csv(filepath, encoding='gbk')
            
    df.columns = [c.strip() for c in df.columns]
    
    # 1. 构造目标变量：Hard Mode 比例
    hard_col = next((c for c in df.columns if 'hard' in c.lower() and 'number' in c.lower()), None)
    total_col = next((c for c in df.columns if 'reported' in c.lower() and 'number' in c.lower()), None)
    
    if hard_col and total_col:
        df['hard_mode_ratio'] = df[hard_col] / df[total_col]
    
    # 2. 清洗分布列
    dist_cols = ['1_try', '2_tries', '3_tries', '4_tries', '5_tries', '6_tries', '7_or_more_tries_x']
    mapped_dist_cols = []
    
    for target in dist_cols:
        match = next((c for c in df.columns if target.replace('_', ' ').lower() in c.replace('_', ' ').lower()), None)
        if match:
            mapped_dist_cols.append(match)
            if df[match].dtype == object:
                df[match] = df[match].astype(str).str.rstrip('%').astype('float')
                if df[match].mean() > 1: 
                    df[match] = df[match] / 100.0
    
    if len(mapped_dist_cols) == 7:
        rename_dict = dict(zip(mapped_dist_cols, dist_cols))
        df = df.rename(columns=rename_dict)
    
    # 填充缺失值
    df[dist_cols] = df[dist_cols].fillna(0)
    
    # --- 【新增】强制归一化 (Sum to 100% check) ---
    # 防止原始数据中有 99.9% 或 100.1% 的情况
    print("正在对分布数据进行归一化 (Sum=1)...")
    row_sums = df[dist_cols].sum(axis=1)
    # 避免除以0
    df[dist_cols] = df[dist_cols].div(row_sums.replace(0, 1), axis=0)
    
    # 3. 计算加权平均猜测次数
    df['avg_guesses'] = np.dot(df[dist_cols].values, np.array([1,2,3,4,5,6,7]))

    return df

def get_feature_cols():
    """返回高级特征列名列表"""
    return [
        'unique_letters', 'has_repeats', 'num_multiple_letters', 'num_rare_letters', 
        'num_vowels', 'starts_with_vowel', 'ends_with_vowel', 'num_consonants', 
        'contains_y', 'has_double_letter', 'max_consecutive_vowels', 
        'max_consecutive_consonants', 'scrabble_score', 'letter_entropy', 
        'position_rarity', 'keyboard_distance', 'hamming_neighbors', 
        'has_common_suffix', 'has_common_prefix', 'letter_freq_mean', 
        'letter_freq_min', 'positional_freq_mean', 'positional_freq_min',
        # 新增特征
        'word_freq', 'is_noun', 'is_verb', 'is_adj'
    ]

def create_lag_features(df, target_col='hard_mode_ratio', lags=3):
    """创建滞后特征"""
    df_lag = df.copy()
    lag_cols = []
    for i in range(1, lags + 1):
        col_name = f'lag_{i}'
        df_lag[col_name] = df_lag[target_col].shift(i)
        lag_cols.append(col_name)
    return df_lag.dropna(subset=lag_cols), lag_cols

def remove_collinear_features(df, feature_cols, threshold=0.95):
    """
    【新增】去除高度共线的特征
    :param threshold: 相关系数阈值，超过这个值会被删除
    """
    print(f"\n正在进行多重共线性检测 (阈值: {threshold})...")
    
    # 确保特征在df中存在
    valid_cols = [c for c in feature_cols if c in df.columns]
    
    # 计算相关矩阵 (绝对值)
    corr_matrix = df[valid_cols].corr().abs()
    
    # 只看上三角矩阵 (避免重复)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 找到相关系数大于阈值的列
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if len(to_drop) > 0:
        print(f"⚠️ 检测到高度共线特征，将移除: {to_drop}")
        # 比如 'num_vowels' 和 'num_consonants' 如果高度负相关，可能会删掉一个
    else:
        print("✅ 未检测到严重共线性特征。")
        
    # 返回保留下来的特征列表
    kept_features = [c for c in valid_cols if c not in to_drop]
    return kept_features