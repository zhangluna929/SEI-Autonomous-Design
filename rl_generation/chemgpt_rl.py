#!/usr/bin/env python3
"""
ChemGPT 强化学习微调模块

整合所有组件实现完整的 RL-HF (Reinforcement Learning from Human Feedback) 生成：
1. 基于 ChemGPT 的分子生成
2. 多目标奖励函数优化
3. 合成易度惩罚机制
4. PPO 强化学习训练
5. One-shot 约束满足生成

实现端到端的分子设计优化流程。

依赖：
    pip install torch transformers accelerate wandb
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
from dataclasses import dataclass
import pickle
import time
import json
from collections import defaultdict

# 导入自定义模块
try:
    from .ppo_trainer import PPOTrainer, PPOConfig, ChemGPTActor, ChemGPTCritic
    from .reward_functions import MultiObjectiveRewardFunction, RewardConfig
    from .synthesis_penalty import SynthesisPenaltyFunction, SynthesisPenaltyConfig
except ImportError:
    print("Warning: 某些模块未找到，请确保项目结构正确")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChemGPTRLConfig:
    """ChemGPT 强化学习配置"""
    # 模型参数
    model_name: str = "ncfrey/ChemGPT-1.2B"
    max_length: int = 128
    temperature: float = 1.0
    
    # 训练参数
    num_episodes: int = 1000
    batch_size: int = 32
    learning_rate: float = 1e-5
    
    # 奖励配置
    reward_config: Optional[RewardConfig] = None
    penalty_config: Optional[SynthesisPenaltyConfig] = None
    
    # 约束参数
    target_properties: Dict[str, float] = None
    constraint_tolerance: float = 0.1
    
    # 生成参数
    generation_batch_size: int = 64
    num_generations_per_episode: int = 100
    
    # 保存和日志
    save_interval: int = 50
    log_interval: int = 10
    work_dir: str = "chemgpt_rl_training"
    
    def __post_init__(self):
        if self.reward_config is None:
            self.reward_config = RewardConfig()
        if self.penalty_config is None:
            self.penalty_config = SynthesisPenaltyConfig()
        if self.target_properties is None:
            self.target_properties = {
                'interface_energy': -0.5,
                'conductivity': 1e-3,
                'molecular_weight': 300,
                'logp': 2.0
            }

class ChemGPTRLTrainer:
    """ChemGPT 强化学习训练器"""
    
    def __init__(self, config: ChemGPTRLConfig):
        """
        初始化 ChemGPT 强化学习训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.work_dir = Path(config.work_dir)
        self.work_dir.mkdir(exist_ok=True)
        
        # 初始化奖励函数
        self.reward_function = MultiObjectiveRewardFunction(
            config.reward_config,
            reference_molecules=self._load_reference_molecules()
        )
        
        # 初始化惩罚函数
        self.penalty_function = SynthesisPenaltyFunction(config.penalty_config)
        
        # 组合奖励函数
        self.combined_reward_function = self._create_combined_reward_function()
        
        # 初始化 PPO 训练器
        ppo_config = PPOConfig(
            model_name=config.model_name,
            max_length=config.max_length,
            temperature=config.temperature,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            max_episodes=config.num_episodes,
            log_interval=config.log_interval,
            save_interval=config.save_interval,
            wandb_project="chemgpt-rl-sei"
        )
        
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            reward_function=self.combined_reward_function,
            work_dir=str(self.work_dir / "ppo")
        )
        
        # 训练统计
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'constraint_satisfaction': [],
            'generation_success': [],
            'diversity_scores': [],
            'synthesis_penalties': []
        }
        
        # 生成的分子库
        self.generated_molecules = []
        self.successful_molecules = []
        
        # 初始化 wandb
        if WANDB_AVAILABLE and config.work_dir:
            wandb.init(
                project="chemgpt-rl-sei",
                config=config.__dict__,
                name=f"chemgpt_rl_{int(time.time())}"
            )
    
    def _load_reference_molecules(self) -> List[str]:
        """加载参考分子"""
        # SEI 相关的参考分子
        reference_molecules = [
            "CC(=O)OC",                    # 碳酸甲酯
            "C1COC(=O)O1",                 # 碳酸乙烯酯
            "CC1COC(=O)O1",                # 碳酸丙烯酯
            "CCOC(=O)OCC",                 # 碳酸二乙酯
            "CC(=O)OCC",                   # 乙酸乙酯
            "C1CCCCC1",                    # 环己烷
            "CCO",                         # 乙醇
            "C1CCOC1",                     # 四氢呋喃
            "CC1=CC=CC=C1",                # 甲苯
            "C1=CC=CC=C1",                 # 苯
            "CC(C)C",                      # 异丙烷
            "CCC(=O)O",                    # 丙酸
            "CC(=O)N",                     # 乙酰胺
            "C1=CC=C(C=C1)O",              # 苯酚
            "CC(=O)OC(C)C"                 # 乙酸异丙酯
        ]
        
        return reference_molecules
    
    def _create_combined_reward_function(self):
        """创建组合奖励函数"""
        def combined_reward(smiles: str) -> float:
            # 基础奖励
            base_reward = self.reward_function(smiles)
            
            # 合成惩罚
            synthesis_penalty = self.penalty_function(smiles)
            
            # 约束满足检查
            constraint_satisfaction = self._check_constraint_satisfaction(smiles)
            
            # 组合奖励
            total_reward = base_reward - synthesis_penalty + constraint_satisfaction
            
            # 确保奖励在合理范围内
            total_reward = max(-1.0, min(2.0, total_reward))
            
            return total_reward
        
        return combined_reward
    
    def _check_constraint_satisfaction(self, smiles: str) -> float:
        """检查约束满足度"""
        try:
            # 计算分子性质
            properties = self.reward_function.mol_properties.compute_properties(smiles)
            
            satisfaction_score = 0.0
            total_constraints = len(self.config.target_properties)
            
            for prop_name, target_value in self.config.target_properties.items():
                if prop_name in properties:
                    actual_value = properties[prop_name]
                    
                    # 计算相对误差
                    if target_value != 0:
                        relative_error = abs(actual_value - target_value) / abs(target_value)
                    else:
                        relative_error = abs(actual_value)
                    
                    # 如果在容差范围内，给予奖励
                    if relative_error <= self.config.constraint_tolerance:
                        satisfaction_score += 1.0
                    else:
                        # 部分奖励，基于距离
                        partial_reward = max(0, 1.0 - relative_error)
                        satisfaction_score += partial_reward
            
            # 归一化
            satisfaction_score /= total_constraints
            
            return satisfaction_score
            
        except Exception as e:
            logger.warning(f"约束检查失败: {e}")
            return 0.0
    
    def train(self) -> Dict[str, Any]:
        """执行强化学习训练"""
        logger.info("开始 ChemGPT 强化学习训练...")
        
        start_time = time.time()
        
        # 执行 PPO 训练
        ppo_stats = self.ppo_trainer.train()
        
        # 后处理和分析
        self._post_training_analysis()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # 编译最终结果
        final_results = {
            'ppo_stats': ppo_stats,
            'training_stats': self.training_stats,
            'generated_molecules': self.generated_molecules,
            'successful_molecules': self.successful_molecules,
            'training_time': training_time,
            'config': self.config
        }
        
        # 保存结果
        self._save_results(final_results)
        
        logger.info(f"ChemGPT 强化学习训练完成，耗时: {training_time:.2f} 秒")
        
        return final_results
    
    def _post_training_analysis(self):
        """训练后分析"""
        logger.info("执行训练后分析...")
        
        # 生成测试分子
        test_molecules = self.generate_molecules(
            num_molecules=100,
            temperature=0.8
        )
        
        # 分析生成质量
        quality_analysis = self._analyze_generation_quality(test_molecules)
        
        # 约束满足分析
        constraint_analysis = self._analyze_constraint_satisfaction(test_molecules)
        
        # 多样性分析
        diversity_analysis = self._analyze_diversity(test_molecules)
        
        # 合成可行性分析
        synthesis_analysis = self._analyze_synthesis_feasibility(test_molecules)
        
        # 更新统计信息
        self.training_stats.update({
            'quality_analysis': quality_analysis,
            'constraint_analysis': constraint_analysis,
            'diversity_analysis': diversity_analysis,
            'synthesis_analysis': synthesis_analysis
        })
    
    def generate_molecules(self, 
                          num_molecules: int = 100,
                          temperature: float = 1.0,
                          max_attempts: int = 1000) -> List[str]:
        """生成分子"""
        logger.info(f"生成 {num_molecules} 个分子...")
        
        molecules = []
        attempts = 0
        
        while len(molecules) < num_molecules and attempts < max_attempts:
            try:
                # 使用训练好的模型生成
                prompt = "[START]"
                generated_mol, _ = self.ppo_trainer.actor.generate(
                    prompt, 
                    max_length=self.config.max_length,
                    temperature=temperature
                )
                
                # 验证分子有效性
                if self._is_valid_molecule(generated_mol):
                    molecules.append(generated_mol)
                    self.generated_molecules.append(generated_mol)
                    
                    # 检查是否满足约束
                    if self._check_constraint_satisfaction(generated_mol) > 0.8:
                        self.successful_molecules.append(generated_mol)
                
                attempts += 1
                
            except Exception as e:
                logger.warning(f"生成分子时出错: {e}")
                attempts += 1
        
        logger.info(f"成功生成 {len(molecules)} 个有效分子")
        return molecules
    
    def _is_valid_molecule(self, smiles: str) -> bool:
        """检查分子有效性"""
        if not smiles or len(smiles) < 3:
            return False
        
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except ImportError:
            # 简化验证
            return True
    
    def _analyze_generation_quality(self, molecules: List[str]) -> Dict[str, Any]:
        """分析生成质量"""
        if not molecules:
            return {'valid_ratio': 0.0, 'mean_reward': 0.0}
        
        rewards = []
        valid_count = 0
        
        for mol in molecules:
            if self._is_valid_molecule(mol):
                valid_count += 1
                reward = self.reward_function(mol)
                rewards.append(reward)
        
        return {
            'valid_ratio': valid_count / len(molecules),
            'mean_reward': np.mean(rewards) if rewards else 0.0,
            'std_reward': np.std(rewards) if rewards else 0.0,
            'max_reward': np.max(rewards) if rewards else 0.0,
            'min_reward': np.min(rewards) if rewards else 0.0
        }
    
    def _analyze_constraint_satisfaction(self, molecules: List[str]) -> Dict[str, Any]:
        """分析约束满足度"""
        if not molecules:
            return {'satisfaction_ratio': 0.0, 'mean_satisfaction': 0.0}
        
        satisfactions = []
        
        for mol in molecules:
            if self._is_valid_molecule(mol):
                satisfaction = self._check_constraint_satisfaction(mol)
                satisfactions.append(satisfaction)
        
        high_satisfaction_count = sum(1 for s in satisfactions if s > 0.8)
        
        return {
            'satisfaction_ratio': high_satisfaction_count / len(satisfactions) if satisfactions else 0.0,
            'mean_satisfaction': np.mean(satisfactions) if satisfactions else 0.0,
            'std_satisfaction': np.std(satisfactions) if satisfactions else 0.0
        }
    
    def _analyze_diversity(self, molecules: List[str]) -> Dict[str, Any]:
        """分析多样性"""
        if not molecules:
            return {'mean_diversity': 0.0, 'unique_ratio': 0.0}
        
        # 计算唯一分子比例
        unique_molecules = list(set(molecules))
        unique_ratio = len(unique_molecules) / len(molecules)
        
        # 计算多样性分数
        diversity_scores = []
        for mol in unique_molecules:
            if self._is_valid_molecule(mol):
                diversity = self.reward_function.diversity_calculator.calculate_diversity_score(mol)
                diversity_scores.append(diversity)
        
        return {
            'unique_ratio': unique_ratio,
            'mean_diversity': np.mean(diversity_scores) if diversity_scores else 0.0,
            'std_diversity': np.std(diversity_scores) if diversity_scores else 0.0
        }
    
    def _analyze_synthesis_feasibility(self, molecules: List[str]) -> Dict[str, Any]:
        """分析合成可行性"""
        if not molecules:
            return {'mean_penalty': 0.0, 'feasible_ratio': 0.0}
        
        penalties = []
        feasible_count = 0
        
        for mol in molecules:
            if self._is_valid_molecule(mol):
                penalty = self.penalty_function(mol)
                penalties.append(penalty)
                
                if penalty < 0.5:  # 低惩罚认为是可行的
                    feasible_count += 1
        
        return {
            'mean_penalty': np.mean(penalties) if penalties else 0.0,
            'std_penalty': np.std(penalties) if penalties else 0.0,
            'feasible_ratio': feasible_count / len(penalties) if penalties else 0.0
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """保存训练结果"""
        # 保存主要结果
        results_file = self.work_dir / "training_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # 保存生成的分子
        molecules_file = self.work_dir / "generated_molecules.txt"
        with open(molecules_file, 'w') as f:
            for mol in self.generated_molecules:
                f.write(f"{mol}\n")
        
        # 保存成功的分子
        successful_file = self.work_dir / "successful_molecules.txt"
        with open(successful_file, 'w') as f:
            for mol in self.successful_molecules:
                f.write(f"{mol}\n")
        
        # 保存配置
        config_file = self.work_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        logger.info(f"结果已保存至: {self.work_dir}")
    
    def evaluate_molecules(self, molecules: List[str]) -> Dict[str, Any]:
        """评估分子列表"""
        evaluation_results = {
            'molecules': [],
            'summary': {}
        }
        
        for mol in molecules:
            if self._is_valid_molecule(mol):
                # 计算各项指标
                reward = self.reward_function(mol)
                penalty = self.penalty_function(mol)
                constraint_satisfaction = self._check_constraint_satisfaction(mol)
                
                mol_result = {
                    'smiles': mol,
                    'reward': reward,
                    'synthesis_penalty': penalty,
                    'constraint_satisfaction': constraint_satisfaction,
                    'total_score': reward - penalty + constraint_satisfaction
                }
                
                evaluation_results['molecules'].append(mol_result)
        
        # 计算汇总统计
        if evaluation_results['molecules']:
            scores = [m['total_score'] for m in evaluation_results['molecules']]
            evaluation_results['summary'] = {
                'num_molecules': len(evaluation_results['molecules']),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'best_score': np.max(scores),
                'worst_score': np.min(scores)
            }
        
        return evaluation_results

def main():
    """主函数 - 演示 ChemGPT 强化学习训练"""
    print("=== ChemGPT 强化学习微调模块测试 ===")
    
    # 创建配置
    config = ChemGPTRLConfig(
        num_episodes=20,  # 演示用小数值
        batch_size=8,
        generation_batch_size=16,
        save_interval=10,
        log_interval=5,
        work_dir="test_chemgpt_rl"
    )
    
    # 创建训练器
    trainer = ChemGPTRLTrainer(config)
    
    # 开始训练
    print("开始 ChemGPT 强化学习训练...")
    results = trainer.train()
    
    # 输出结果
    print("\n=== 训练结果 ===")
    print(f"训练时间: {results['training_time']:.2f} 秒")
    print(f"生成分子数: {len(results['generated_molecules'])}")
    print(f"成功分子数: {len(results['successful_molecules'])}")
    
    if 'quality_analysis' in results['training_stats']:
        quality = results['training_stats']['quality_analysis']
        print(f"有效分子比例: {quality['valid_ratio']:.3f}")
        print(f"平均奖励: {quality['mean_reward']:.3f}")
    
    if 'constraint_analysis' in results['training_stats']:
        constraint = results['training_stats']['constraint_analysis']
        print(f"约束满足比例: {constraint['satisfaction_ratio']:.3f}")
    
    # 测试生成
    print("\n=== 生成测试 ===")
    test_molecules = trainer.generate_molecules(num_molecules=5)
    
    for i, mol in enumerate(test_molecules):
        print(f"分子 {i+1}: {mol}")
    
    # 评估测试分子
    if test_molecules:
        evaluation = trainer.evaluate_molecules(test_molecules)
        print(f"\n平均分数: {evaluation['summary']['mean_score']:.3f}")
        print(f"最佳分数: {evaluation['summary']['best_score']:.3f}")
    
    print("\n✓ ChemGPT 强化学习微调模块测试完成")

if __name__ == "__main__":
    main() 