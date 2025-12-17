"""
数据加载工具

统一加载原始数据和处理后数据
"""

import pandas as pd
from .config import ORIGINAL_DATA, PROCESSED_DATA


def load_original_data():
    """
    加载原始 Excel 数据
    
    Returns:
        pd.DataFrame: 原始 Wordle 数据
    """
    df = pd.read_excel(ORIGINAL_DATA, header=1)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]
    return df.sort_values('Date').reset_index(drop=True)


def load_processed_data():
    """
    加载处理后的 CSV 数据（含特征工程）
    
    Returns:
        pd.DataFrame: 处理后的数据
    """
    return pd.read_csv(PROCESSED_DATA)


if __name__ == "__main__":
    print("测试数据加载...")
    
    print("\n1. 加载原始数据:")
    df_orig = load_original_data()
    print(f"   - 形状: {df_orig.shape}")
    print(f"   - 列: {list(df_orig.columns[:5])}...")
    print(f"   - 日期范围: {df_orig['Date'].min()} 到 {df_orig['Date'].max()}")
    
    print("\n2. 加载处理数据:")
    df_proc = load_processed_data()
    print(f"   - 形状: {df_proc.shape}")
    print(f"   - 特征数: {len(df_proc.columns) - 8}")  # 减去目标变量列
    
    print("\n✓ 数据加载测试完成")
