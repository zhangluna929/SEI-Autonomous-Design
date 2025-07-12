#!/bin/bash

# 下载 ChemGPT 模型脚本
# 用于分子生成的预训练模型

echo "正在下载 ChemGPT 模型..."

# 创建模型目录
mkdir -p generator/chemGPT

# 使用 git lfs 下载模型（如果可用）
if command -v git-lfs &> /dev/null; then
    echo "使用 git-lfs 下载模型..."
    cd generator/chemGPT
    git lfs clone https://huggingface.co/ncfrey/ChemGPT-1.2B .
    cd ../..
else
    echo "git-lfs 不可用，请手动下载模型"
    echo "1. 安装 git-lfs: https://git-lfs.github.io/"
    echo "2. 运行: git lfs clone https://huggingface.co/ncfrey/ChemGPT-1.2B generator/chemGPT"
    echo "或者从 Hugging Face 手动下载模型文件到 generator/chemGPT/"
fi

echo "下载完成！" 