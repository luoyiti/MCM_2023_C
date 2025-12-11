import matplotlib.pyplot as plt
import seaborn as sns

# 定义莫兰迪色系 (Morandi Palette)
MORANDI_COLORS = [
    '#A4B9B1', # 灰豆绿
    '#D7B7B2', # 干燥玫瑰粉
    '#97A0AD', # 雾霾蓝
    '#CDBBA7', # 奶茶驼
    '#8F9C93', # 橄榄灰
    '#BFA6A2'  # 烟熏紫
]

def apply_morandi_style():
    """应用莫兰迪风格全局设置"""
    sns.set_theme(style="whitegrid")
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=MORANDI_COLORS)
    plt.rcParams['font.sans-serif'] = ['Arial'] # 或 SimHei 显示中文
    plt.rcParams['axes.unicode_minus'] = False
    print("🎨 Morandi Style Applied!")

def plot_feature_importance(importances, title, top_n=10):
    """绘制特征重要性条形图"""
    plt.figure(figsize=(10, 6))
    # 使用莫兰迪色系中的某一种颜色，或者渐变
    top_features = importances.head(top_n)
    sns.barplot(x=top_features.values, y=top_features.index, palette=MORANDI_COLORS)
    plt.title(title, fontsize=14, fontweight='bold', color='#555555')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results_df, metric='R2'):
    """绘制模型对比图"""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=metric, y='Model', data=results_df, palette=MORANDI_COLORS)
    plt.title(f'Model Comparison ({metric})', fontsize=14, fontweight='bold', color='#555555')
    plt.axvline(x=0, color='#888888', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.show()