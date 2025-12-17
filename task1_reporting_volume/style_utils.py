"""
样式工具模块 - 为分析脚本提供绘图功能
"""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_feature_importance(importances: pd.Series, title: str = "Feature Importance", top_n: int = 10):
    """
    绘制特征重要性条形图
    
    Parameters:
    -----------
    importances : pd.Series
        特征重要性，索引为特征名，值为重要性分数
    title : str
        图表标题
    top_n : int
        显示前 N 个最重要的特征
    """
    # 取前 top_n 个
    top_features = importances.head(top_n)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features.sort_values().plot(kind='barh', ax=ax, color='steelblue')
    
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片到项目级别的 pictures/task1/ 目录
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from shared import TASK1_PICTURES
    
    output_dir = TASK1_PICTURES
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = title.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '') + '.png'
    output_path = output_dir / output_filename
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ 图表已保存: {output_path}")
    
    plt.close()
