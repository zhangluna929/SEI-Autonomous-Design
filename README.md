# SEI Autonomous Design Platform / SEI 自主设计平台

## Overview / 概述

Multi-modal solid-state interface autonomous design platform for solid electrolyte interface (SEI) design. Integrates multi-scale coupling and reinforcement learning for molecular generation.

面向固体电解质界面（SEI）设计的多模态固态界面自主设计平台。集成多尺度耦合和强化学习进行分子生成。

## Architecture / 架构

```
sei-design-platform/
├── data/                    # Raw datasets / 原始数据集
│   ├── obelix/             # Crystal structures / 晶体结构
│   ├── polymer/            # Polymer data / 聚合物数据
│   └── sei_spectra/        # Spectroscopic data / 光谱数据
├── data_simulate/          # Synthetic data generators / 合成数据生成器
│   ├── xps_sim.py         # XPS spectra simulation / XPS光谱模拟
│   ├── nmr_sim.py         # NMR spectra simulation / NMR光谱模拟
│   ├── afm_sim.py         # AFM force curves simulation / AFM力曲线模拟
│   └── eis_sim.py         # EIS impedance simulation / EIS阻抗模拟
├── pretrain/               # Foundation model pretraining / 基础模型预训练
│   ├── models.py          # Multi-modal encoders / 多模态编码器
│   ├── datamodule.py      # Data preprocessing / 数据预处理
│   ├── train_lightning.py # Training pipeline / 训练流程
│   └── mobile_encoder.py  # Lightweight encoder / 轻量级编码器
├── finetune/              # Predictor fine-tuning / 预测器微调
│   ├── predictor.py       # Multi-task predictor / 多任务预测器
│   ├── train_predictor.py # Training script / 训练脚本
│   └── datamodule.py      # Data module / 数据模块
├── generator/             # Conditional generation / 条件生成
│   ├── cond_diffusion.py  # Diffusion model / 扩散模型
│   ├── sampler.py         # Sampling algorithms / 采样算法
│   └── run_generation.py  # Generation pipeline / 生成流程
├── active_learning/       # Active learning loop / 主动学习循环
│   └── select_uncertain.py # Uncertainty sampling / 不确定性采样
├── multiscale/            # Multi-scale coupling / 多尺度耦合
│   ├── md_lammps.py       # MD relaxation / MD松弛
│   ├── dft_queue.py       # DFT calculation queue / DFT计算队列
│   ├── dual_active_learning.py # Coarse-fine AL / 粗-精主动学习
│   └── phase_field.py     # Phase field simulation / 相场模拟
├── rl_generation/         # Reinforcement learning / 强化学习
│   ├── ppo_trainer.py     # PPO framework / PPO框架
│   ├── reward_functions.py # Multi-objective rewards / 多目标奖励
│   ├── synthesis_penalty.py # Synthesis feasibility / 合成可行性
│   └── chemgpt_rl.py      # ChemGPT RL fine-tuning / ChemGPT强化学习微调
├── build_master_dataset.py # Data homogenization / 数据同构化
├── requirements.txt       # Dependencies / 依赖项
└── demo_multiscale_rl.py  # Comprehensive demo / 综合演示
```

## Key Features / 核心功能

### Multi-Scale Coupling / 多尺度耦合
- MD-LAMMPS fast relaxation (100 structures/hour) / MD-LAMMPS快速松弛（100个结构/小时）
- DFT precise calculation (10 structures/hour) / DFT精确计算（10个结构/小时）
- Coarse-fine dual-layer active learning / 粗-精双层主动学习
- Phase field simulation for crack evolution / 裂纹演化相场模拟

### Reinforcement Learning Generation / 强化学习生成
- PPO fine-tuning of ChemGPT-1.2B / PPO微调ChemGPT-1.2B
- Multi-objective reward optimization (ΔE, σ, diversity) / 多目标奖励优化（ΔE、σ、多样性）
- Synthesis feasibility penalty mechanism / 合成可行性惩罚机制
- One-shot constraint satisfaction generation / 一次性约束满足生成

### Data Modalities / 数据模态
- Crystal structures (CIF → local environment graphs) / 晶体结构（CIF → 局部环境图）
- Polymer sequences (SMILES → SELFIES + 3D coordinates) / 聚合物序列（SMILES → SELFIES + 3D坐标）
- Spectroscopic data (FTIR/Raman/XRD resampling) / 光谱数据（FTIR/Raman/XRD重采样）
- Electrochemical data (XPS/NMR/AFM/EIS simulation) / 电化学数据（XPS/NMR/AFM/EIS模拟）

## Performance Targets / 性能指标

| Metric / 指标 | Target / 目标 | Status / 状态 |
|---------------|---------------|---------------|
| Interface Energy MAE / 界面能MAE | < 0.08 eV/nm² | Achieved / 已达成 |
| Conductivity R² / 电导率R² | > 0.7 | Achieved / 已达成 |
| Constraint Satisfaction / 约束满足率 | > 80% | 85% |
| Synthesis Feasibility / 合成可行性 | > 70% | 72% |

## Installation / 安装

```bash
pip install -r requirements.txt
```

## Usage / 使用方法

### Data Preparation / 数据准备
```bash
python build_master_dataset.py --out master.parquet
```

### Model Training / 模型训练
```bash
# Pretrain encoders / 预训练编码器
python pretrain/train_lightning.py

# Fine-tune predictor / 微调预测器
python finetune/train_predictor.py

# Train generator / 训练生成器
python generator/run_generation.py
```

### Multi-Scale Simulation / 多尺度模拟
```bash
# Run dual active learning / 运行双层主动学习
python multiscale/dual_active_learning.py

# Phase field simulation / 相场模拟
python multiscale/phase_field.py
```

### Reinforcement Learning / 强化学习
```bash
# RL fine-tuning / 强化学习微调
python rl_generation/chemgpt_rl.py
```

### Comprehensive Demo / 综合演示
```bash
python demo_multiscale_rl.py
```

## Makefile Commands / Makefile命令

```bash
make install     # Install dependencies / 安装依赖
make data        # Prepare datasets / 准备数据集
make pretrain    # Train foundation models / 训练基础模型
make finetune    # Fine-tune predictors / 微调预测器
make multiscale  # Run multi-scale simulations / 运行多尺度模拟
make rl          # Run reinforcement learning / 运行强化学习
make demo        # Run comprehensive demo / 运行综合演示
make clean       # Clean temporary files / 清理临时文件
make all         # Run full pipeline / 运行完整流程
```

## Dependencies / 依赖项

Core Libraries / 核心库:
- PyTorch >= 1.12.0
- Transformers >= 4.20.0
- RDKit >= 2022.3.0
- PyMatGen >= 2022.7.0

Multi-scale Computing / 多尺度计算:
- LAMMPS-Python >= 3.4.0
- FiPy >= 3.4.0
- Custodian >= 2022.5.0

Machine Learning / 机器学习:
- PyTorch Lightning >= 1.7.0
- Scikit-learn >= 1.0.0
- Wandb >= 0.13.0

## Technical Specifications / 技术规格

- Architecture: 256-dim embeddings with cross-attention fusion / 架构：256维嵌入与交叉注意力融合
- Hardware: Single A100 GPU for training, 16 VASP nodes for DFT / 硬件：单A100 GPU训练，16个VASP节点DFT计算
- Data: ~3k multimodal samples across 7 modalities / 数据：7种模态约3k多模态样本
- Performance: ΔE MAE < 0.08 eV/nm², σ R² > 0.7 / 性能：ΔE MAE < 0.08 eV/nm²，σ R² > 0.7
- Generation: 5k molecules in 20 minutes / 生成：20分钟生成5k分子
- Active learning: 48h cycles for 128 samples / 主动学习：48小时周期处理128个样本

## License / 许可证

MIT License 