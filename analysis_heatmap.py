import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import style_utils
import data_loader

def plot_correlation_heatmap(df, feature_cols):
    print("\n" + "="*40)
    print(">>> 正在生成单色系极简热力图 (Monochromatic Style)")
    print("="*40)
    
    # 1. 准备数据 (逻辑不变)
    df_lag, lag_cols = data_loader.create_lag_features(df, 'hard_mode_ratio', 3)
    
    if 'avg_guesses' not in df_lag.columns:
         dist_cols = ['1_try', '2_tries', '3_tries', '4_tries', '5_tries', '6_tries', '7_or_more_tries_x']
         df_lag['avg_guesses'] = np.dot(df_lag[dist_cols].fillna(0).values, np.array([1,2,3,4,5,6,7]))

    # -----------------------------------------------------------
    # 🎨 配色方案：莫兰迪同色系 (Monochromatic Morandi)
    # -----------------------------------------------------------
    # 核心思路：用“白色”做中间点，两边延展出同色系的深浅变化
    # 这种配色会让图表看起来非常干净、统一
    
    # 颜色代码：
    # 负相关 (-1): #B0B5B9 (莫兰迪灰/浅灰蓝) - 这里的“浅”其实是有灰度的，保证看得见
    # 无相关 ( 0): #FFFFFF (纯白)
    # 正相关 (+1): #2C405A (深邃蓝/墨蓝)
    
    mono_colors = ['#B0B5B9', '#FFFFFF', '#2C405A']
    
    # 创建线性渐变色盘
    morandi_cmap = LinearSegmentedColormap.from_list("morandi_mono", mono_colors, N=256)
    
    style_utils.apply_morandi_style()
    
    # =======================================================
    # 图表 1: RQ1 (Hard Mode)
    # =======================================================
    print("绘制 RQ1 热力图...")
    target = 'hard_mode_ratio'
    all_features_rq1 = lag_cols + feature_cols
    
    corr_series = df_lag[all_features_rq1 + [target]].corr()[target].drop(target)
    corr_series_sorted = corr_series.abs().sort_values(ascending=False)
    
    # 只取前 20 个最重要的特征，避免图太长
    top_features = corr_series_sorted.head(20).index.tolist()
    cols_to_plot = [target] + top_features
    final_corr_matrix = df_lag[cols_to_plot].corr()

    plt.figure(figsize=(10, 12)) 
    sns.heatmap(final_corr_matrix, 
                annot=True,      
                fmt=".2f",       
                cmap=morandi_cmap,  # 应用新配色
                vmin=-1, vmax=1, 
                center=0,        
                square=True,
                linewidths=1,    # 加粗白色网格线，增强“极简”感
                linecolor='white',
                cbar_kws={"shrink": 0.7},
                annot_kws={"size": 9, "color": "#333333"}) # 数字颜色加深，防止在浅色背景看不清
    
    plt.title('RQ1: Hard Mode Ratio Drivers (Monochromatic)', fontsize=15, pad=20, fontweight='bold', color='#333333')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig('heatmap_rq1_mono.png', dpi=300)
    plt.show()

    # =======================================================
    # 图表 2: RQ2 (Difficulty)
    # =======================================================
    print("绘制 RQ2 热力图...")
    rq2_target = 'avg_guesses'
    
    corr_series_rq2 = df_lag[feature_cols + [rq2_target]].corr()[rq2_target].drop(rq2_target)
    sorted_features_rq2 = corr_series_rq2.abs().sort_values(ascending=False).index.tolist()
    
    top_n = 15
    cols_rq2 = [rq2_target] + sorted_features_rq2[:top_n]
    data_rq2 = df_lag[cols_rq2]
    corr_rq2 = data_rq2.corr()
    
    plt.figure(figsize=(10, 9))
    sns.heatmap(corr_rq2, 
                annot=True, 
                fmt=".2f", 
                cmap=morandi_cmap,  # 应用新配色
                vmin=-1, vmax=1, 
                center=0,
                square=True,
                linewidths=1,
                linecolor='white',
                cbar_kws={"shrink": 0.7},
                annot_kws={"size": 9, "color": "#333333"})
    
    plt.title(f'RQ2: Difficulty Factors (Top {top_n})', fontsize=15, pad=20, fontweight='bold', color='#333333')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('heatmap_rq2_mono.png', dpi=300)
    plt.show()
    
    print("✅ 单色系热力图已生成 (heatmap_rq1_mono.png, heatmap_rq2_mono.png)")

if __name__ == "__main__":
    df = data_loader.load_and_clean_data('data_final.csv') 
    features = data_loader.get_feature_cols()
    plot_correlation_heatmap(df, features)