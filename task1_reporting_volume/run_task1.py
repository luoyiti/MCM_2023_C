"""
任务1：预测报告人数 & Hard Mode 分析

运行此脚本完成：
1. 预测 2023-03-01 的报告人数（含置信区间）
2. 分析单词属性对 Hard Mode 比例的影响

输出：
- results/task1/ - CSV/TXT 结果文件
- pictures/task1/ - PNG 图片文件
"""

import sys
import os
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入项目配置
from shared import ORIGINAL_DATA, TASK1_RESULTS, TASK1_PICTURES

print("="*70)
print("任务1：报告人数预测 & Hard Mode 分析")
print("="*70)
print(f"数据来源: {ORIGINAL_DATA}")
print(f"结果目录: {TASK1_RESULTS}")
print(f"图片目录: {TASK1_PICTURES}")
print("="*70)

# 1. 运行时间序列预测
print("\n[1/2] 运行报告人数预测...")
print("  ⏳ 这可能需要几分钟，请耐心等待...\n")
try:
    # 使用配置的路径运行 q1（实时显示输出）
    # 注意：使用 CSV 文件，包含真实数据（Excel 是归一化的百分比）
    import subprocess
    from shared import PROCESSED_DATA
    
    cmd = [
        'python', 
        str(Path(__file__).parent / 'q1_final_clean.py'),
        '--input', str(PROCESSED_DATA),  # 使用 CSV（真实数据）
        '--output-dir', str(TASK1_RESULTS)  # CSV/TXT 结果
    ]
    # 使用 stdout=None 让输出实时显示
    result = subprocess.run(cmd, check=True)
    print("\n")
    
    # 移动图片文件到 pictures/task1/
    import shutil
    for png_file in TASK1_RESULTS.glob('*.png'):
        dest = TASK1_PICTURES / png_file.name
        shutil.move(str(png_file), str(dest))
        print(f"  ✓ 图片已移动: {dest}")
    
    print("✓ 报告人数预测完成")
except Exception as e:
    print(f"✗ 报告人数预测失败: {e}")

# 2. 运行 Hard Mode 分析
print("\n[2/2] 运行 Hard Mode 分析...")
try:
    from analysis_hard_mode import analyze_hard_mode
    analyze_hard_mode()
    print("✓ Hard Mode 分析完成")
except Exception as e:
    print(f"✗ Hard Mode 分析失败: {e}")
    print("   提示: 请确保 analysis_hard_mode.py 中有 analyze_hard_mode() 函数")

print("\n" + "="*70)
print("任务1 完成！")
print(f"CSV/TXT 结果: {TASK1_RESULTS}")
print(f"PNG 图片: {TASK1_PICTURES}")
print("="*70)
