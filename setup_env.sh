#!/bin/bash

# MCM 2023 Problem C - 环境配置脚本
# 创建 Python 3.11 环境并安装所有依赖

echo "========================================="
echo "MCM 2023 环境配置"
echo "========================================="
echo ""

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo "❌ 错误：未找到 conda"
    echo "请先安装 Miniconda 或 Anaconda"
    exit 1
fi

echo "✓ 找到 conda"

# 环境名称
ENV_NAME="mcm2023"

echo ""
echo "步骤 1/4: 检查环境是否存在..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  环境 '${ENV_NAME}' 已存在"
    read -p "是否删除并重新创建? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除旧环境..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "使用现有环境"
    fi
fi

echo ""
echo "步骤 2/4: 创建 Python 3.11 环境..."
conda create -n ${ENV_NAME} python=3.11 -y

if [ $? -ne 0 ]; then
    echo "❌ 环境创建失败"
    exit 1
fi

echo "✓ 环境创建成功"

echo ""
echo "步骤 3/4: 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

if [ $? -ne 0 ]; then
    echo "❌ 环境激活失败"
    exit 1
fi

echo "✓ 环境已激活: ${ENV_NAME}"

echo ""
echo "步骤 4/4: 安装依赖包..."
echo "   (这可能需要几分钟...)"
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  部分依赖安装失败，尝试单独安装核心包..."
    
    # 核心依赖
    pip install numpy pandas matplotlib scipy scikit-learn openpyxl
    pip install statsmodels ruptures holidays
    pip install wordfreq nltk
    
    # TensorFlow (可选)
    if [[ $(uname -m) == "arm64" ]] && [[ $(uname -s) == "Darwin" ]]; then
        echo "检测到 Apple Silicon Mac，安装 tensorflow-macos..."
        pip install tensorflow-macos tensorflow-metal
    else
        pip install tensorflow
    fi
fi

echo ""
echo "========================================="
echo "✓ 环境配置完成！"
echo "========================================="
echo ""
echo "使用方法："
echo "  1. 激活环境:"
echo "     conda activate ${ENV_NAME}"
echo ""
echo "  2. 运行任务:"
echo "     ./run_task1.sh  # 任务1"
echo "     ./run_task2.sh  # 任务2"
echo ""
echo "  3. 退出环境:"
echo "     conda deactivate"
echo ""
echo "========================================="
