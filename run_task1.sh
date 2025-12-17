#!/bin/bash

# MCM 2023 Problem C - 任务1: 报告人数预测 & Hard Mode 分析
# 
# 使用说明：
#   chmod +x run_task1.sh
#   ./run_task1.sh

echo "========================================="
echo "MCM 2023 任务1 - 报告人数预测"
echo "========================================="
echo ""

# 检查 conda 环境
if ! command -v conda &> /dev/null; then
    echo "❌ 错误：未找到 conda"
    echo "请先运行: ./setup_env.sh"
    exit 1
fi

# 检查 mcm2023 环境是否存在
if ! conda env list | grep -q "^mcm2023 "; then
    echo "❌ 错误：未找到 mcm2023 环境"
    echo "请先运行: ./setup_env.sh"
    exit 1
fi

echo "✓ 环境检查通过"

# 激活环境并运行
echo "✓ 使用环境: mcm2023 (Python 3.11)"

echo ""

# 运行预测（使用 conda run 确保在正确环境中）
echo "运行任务1..."
echo ""

cd task1_reporting_volume
conda run -n mcm2023 --no-capture-output python run_task1.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ 任务1 基础分析完成！"
    echo "========================================="
    echo ""
    
    # 运行模型对比
    echo "运行模型对比 (Ensemble vs Prophet vs Chronos)..."
    echo ""
    conda run -n mcm2023 --no-capture-output python model_comparison.py --input ../data/mcm_processed_data.csv
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================="
        echo "✓ 模型对比完成！"
        echo "========================================="
    else
        echo ""
        echo "⚠️  模型对比失败 (可能缺少 Prophet/Chronos)"
        echo "   主要模型 (Ensemble) 已成功运行"
    fi
    
    echo ""
    echo "========================================="
    echo "✓ 任务1 全部完成！"
    echo "========================================="
    echo "输出文件:"
    echo "  📄 文本报告 → results/task1/"
    echo "     - explanation_report.txt (含滞后特征分析)"
    echo "     - diagnostic_report.txt"
    echo "     - unified_comparison_report.txt"
    echo ""
    echo "  📊 可视化图表 → pictures/task1/"
    echo "     - 1_weekday_effects.png"
    echo "     - 3_diagnostics.png"
    echo "     - 4_factor_importance.png (含6个因素)"
    echo "     - 6_three_way_comparison_*.png"
    echo "========================================="
else
    echo ""
    echo "❌ 预测失败，请检查错误信息"
    exit 1
fi
