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
echo "⏳ 预计时间：5-15 分钟（包含 SARIMA 训练和交叉验证）"
echo ""

cd task1_reporting_volume
conda run -n mcm2023 --no-capture-output python run_task1.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ 任务1 完成！"
    echo "========================================="
    echo "输出文件:"
    echo "  - CSV/TXT: results/task1/"
    echo "  - PNG图片: pictures/task1/"
    echo "========================================="
else
    echo ""
    echo "❌ 预测失败，请检查错误信息"
    exit 1
fi
