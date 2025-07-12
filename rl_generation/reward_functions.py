#!/usr/bin/env python3
"""
强化学习奖励函数模块

定义多种奖励函数用于引导分子生成：
1. 基于性质的奖励（ΔE, σ, 合成难度等）
2. 多样性奖励
3. 有效性奖励
4. 组合奖励函数

依赖：
    pip install rdkit-pypi numpy
"""

import numpy as np
from typing import List, Dict, Any, Callable, Optional
import logging
from abc import ABC, abstractmethod

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Using simplified reward functions.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardFunction(ABC):
    """奖励函数基类"""
    
    @abstractmethod
    def __call__(self, smiles: str) -> float:
        """计算单个分子的奖励"""
        pass
    
    def batch_reward(self, smiles_list: List[str]) -> List[float]:
        """批量计算奖励"""
        return [self(smiles) for smiles in smiles_list]

class ValidityReward(RewardFunction):
    """分子有效性奖励"""
    
    def __init__(self, valid_reward: float = 1.0, invalid_penalty: float = -1.0):
        self.valid_reward = valid_reward
        self.invalid_penalty = invalid_penalty
    
    def __call__(self, smiles: str) -> float:
        if not RDKIT_AVAILABLE:
            return self.valid_reward if len(smiles) > 0 else self.invalid_penalty
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self.invalid_penalty
            
            # 检查分子是否有效
            if mol.GetNumAtoms() == 0:
                return self.invalid_penalty
            
            # 尝试生成canonical SMILES
            canonical = Chem.MolToSmiles(mol)
            if canonical:
                return self.valid_reward
            else:
                return self.invalid_penalty
                
        except Exception:
            return self.invalid_penalty

class PropertyReward(RewardFunction):
    """基于分子性质的奖励"""
    
    def __init__(self, 
                 target_mw: float = 300.0,
                 target_logp: float = 2.0,
                 mw_weight: float = 0.5,
                 logp_weight: float = 0.5):
        self.target_mw = target_mw
        self.target_logp = target_logp
        self.mw_weight = mw_weight
        self.logp_weight = logp_weight
    
    def __call__(self, smiles: str) -> float:
        if not RDKIT_AVAILABLE:
            # 简化实现：基于分子长度
            return 1.0 - abs(len(smiles) - 20) / 50.0
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return -1.0
            
            # 计算分子量
            mw = Descriptors.MolWt(mol)
            mw_score = 1.0 - abs(mw - self.target_mw) / self.target_mw
            
            # 计算LogP
            logp = Crippen.MolLogP(mol)
            logp_score = 1.0 - abs(logp - self.target_logp) / 5.0  # 假设LogP范围为±5
            
            # 组合得分
            total_score = (self.mw_weight * mw_score + 
                          self.logp_weight * logp_score)
            
            return np.clip(total_score, -1.0, 1.0)
            
        except Exception:
            return -1.0

class DiversityReward(RewardFunction):
    """多样性奖励"""
    
    def __init__(self, reference_smiles: List[str], similarity_threshold: float = 0.7):
        self.reference_smiles = reference_smiles
        self.similarity_threshold = similarity_threshold
        
        if RDKIT_AVAILABLE:
            self.reference_fps = []
            for smiles in reference_smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    self.reference_fps.append(fp)
    
    def __call__(self, smiles: str) -> float:
        if not RDKIT_AVAILABLE:
            # 简化实现：基于字符串相似度
            max_similarity = 0.0
            for ref_smiles in self.reference_smiles:
                similarity = self._string_similarity(smiles, ref_smiles)
                max_similarity = max(max_similarity, similarity)
            return 1.0 - max_similarity
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return -1.0
            
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            
            # 计算与参考分子的最大相似度
            max_similarity = 0.0
            for ref_fp in self.reference_fps:
                similarity = Chem.DataStructs.TanimotoSimilarity(fp, ref_fp)
                max_similarity = max(max_similarity, similarity)
            
            # 奖励不相似的分子
            if max_similarity < self.similarity_threshold:
                return 1.0 - max_similarity
            else:
                return -max_similarity
                
        except Exception:
            return -1.0
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """简单的字符串相似度计算"""
        if not s1 or not s2:
            return 0.0
        
        # 计算最长公共子序列
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n] / max(m, n)

class SynthesisReward(RewardFunction):
    """合成难度奖励"""
    
    def __init__(self, penalty_weight: float = 0.5):
        self.penalty_weight = penalty_weight
    
    def __call__(self, smiles: str) -> float:
        if not RDKIT_AVAILABLE:
            # 简化实现：基于分子复杂度
            complexity = len(set(smiles)) / len(smiles) if smiles else 0
            return 1.0 - complexity * self.penalty_weight
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return -1.0
            
            # 计算合成复杂度指标
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            num_stereo_centers = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
            
            # 复杂度惩罚
            complexity_penalty = (num_rings * 0.1 + 
                                num_aromatic_rings * 0.05 + 
                                num_stereo_centers * 0.2)
            
            return 1.0 - complexity_penalty * self.penalty_weight
            
        except Exception:
            return -1.0

class CompositeReward(RewardFunction):
    """组合奖励函数"""
    
    def __init__(self, reward_functions: List[RewardFunction], weights: List[float]):
        assert len(reward_functions) == len(weights), "奖励函数数量必须与权重数量匹配"
        self.reward_functions = reward_functions
        self.weights = weights
        self.weights = np.array(weights) / np.sum(weights)  # 归一化权重
    
    def __call__(self, smiles: str) -> float:
        total_reward = 0.0
        for reward_fn, weight in zip(self.reward_functions, self.weights):
            reward = reward_fn(smiles)
            total_reward += weight * reward
        return total_reward

class SEIPropertyReward(RewardFunction):
    """SEI相关性质奖励"""
    
    def __init__(self, 
                 target_conductivity: float = 1e-4,  # 目标离子电导率
                 target_stability: float = 4.0,     # 目标电化学稳定性
                 conductivity_weight: float = 0.6,
                 stability_weight: float = 0.4):
        self.target_conductivity = target_conductivity
        self.target_stability = target_stability
        self.conductivity_weight = conductivity_weight
        self.stability_weight = stability_weight
    
    def __call__(self, smiles: str) -> float:
        # 这里需要集成实际的性质预测模型
        # 目前使用简化的启发式方法
        
        if not RDKIT_AVAILABLE:
            return np.random.uniform(-0.5, 0.5)  # 随机奖励
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return -1.0
            
            # 基于分子描述符的简化预测
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            
            # 电导率评分（较小分子通常有更好的离子传导）
            conductivity_score = 1.0 - abs(mw - 200) / 500
            
            # 稳定性评分（基于LogP和其他因子）
            stability_score = 1.0 - abs(logp - 1.0) / 3.0
            
            # 组合得分
            total_score = (self.conductivity_weight * conductivity_score +
                          self.stability_weight * stability_score)
            
            return np.clip(total_score, -1.0, 1.0)
            
        except Exception:
            return -1.0

def create_sei_reward_function() -> CompositeReward:
    """创建SEI材料优化的组合奖励函数"""
    
    # 定义各个奖励组件
    validity_reward = ValidityReward(valid_reward=1.0, invalid_penalty=-2.0)
    property_reward = PropertyReward(target_mw=250.0, target_logp=1.5)
    sei_reward = SEIPropertyReward()
    synthesis_reward = SynthesisReward(penalty_weight=0.3)
    
    # 定义权重
    weights = [0.3, 0.25, 0.35, 0.1]  # 有效性、通用性质、SEI性质、合成难度
    
    return CompositeReward(
        reward_functions=[validity_reward, property_reward, sei_reward, synthesis_reward],
        weights=weights
    )

def test_reward_functions():
    """测试奖励函数"""
    test_molecules = [
        "CCO",  # 乙醇
        "C1=CC=CC=C1",  # 苯
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # 阿司匹林
        "invalid_smiles",  # 无效SMILES
        "C" * 100,  # 过长分子
    ]
    
    reward_fn = create_sei_reward_function()
    
    print("测试奖励函数:")
    for smiles in test_molecules:
        reward = reward_fn(smiles)
        print(f"SMILES: {smiles[:20]}{'...' if len(smiles) > 20 else ''}")
        print(f"奖励: {reward:.3f}")
        print("-" * 40)

if __name__ == "__main__":
    test_reward_functions() 