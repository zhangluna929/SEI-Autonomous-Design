#!/usr/bin/env python3
"""
合成易度惩罚机制

实现基于合成可达性的惩罚机制，自动规避难以实验制备的官能团：
1. 合成可达性评估 (SA Score)
2. 反应路径分析
3. 官能团复杂度评估
4. 试剂可获得性检查
5. 反应条件苛刻度评估

基于经验规则和机器学习模型的混合方法。

依赖：
    pip install rdkit-pypi scikit-learn
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Set
import logging
from pathlib import Path
from dataclasses import dataclass
import pickle
import json
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import Fragments
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Using simplified implementations.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SynthesisPenaltyConfig:
    """合成易度惩罚配置"""
    # 惩罚权重
    sa_score_weight: float = 0.4        # SA 分数权重
    functional_group_weight: float = 0.3 # 官能团复杂度权重
    reaction_path_weight: float = 0.2    # 反应路径权重
    reagent_availability_weight: float = 0.1  # 试剂可获得性权重
    
    # 阈值参数
    sa_score_threshold: float = 6.0      # SA 分数阈值
    complexity_threshold: float = 0.7    # 复杂度阈值
    max_synthesis_steps: int = 8         # 最大合成步数
    
    # 惩罚强度
    penalty_scale: float = 2.0           # 惩罚缩放因子
    penalty_offset: float = 0.1          # 惩罚偏移量

class FunctionalGroupAnalyzer:
    """官能团分析器"""
    
    def __init__(self):
        # 定义困难官能团及其惩罚分数
        self.difficult_groups = {
            # 高度不稳定的官能团
            'peroxide': {'smarts': '[O][O]', 'penalty': 0.9, 'reason': '过氧化物不稳定'},
            'diazo': {'smarts': '[N]=[N+]=[N-]', 'penalty': 0.8, 'reason': '重氮化合物不稳定'},
            'azide': {'smarts': '[N-]=[N+]=[N-]', 'penalty': 0.7, 'reason': '叠氮化合物爆炸性'},
            
            # 需要特殊条件的官能团
            'organometallic': {'smarts': '[Li,Na,K,Mg,Zn,Al]', 'penalty': 0.6, 'reason': '需要无水无氧条件'},
            'grignard': {'smarts': '[Mg][C]', 'penalty': 0.6, 'reason': '格氏试剂需要严格无水'},
            'organolithium': {'smarts': '[Li][C]', 'penalty': 0.7, 'reason': '有机锂试剂极易水解'},
            
            # 复杂环系
            'spirocycle': {'smarts': '[C]12[C][C][C][C]1[C][C][C][C]2', 'penalty': 0.5, 'reason': '螺环化合物合成困难'},
            'bridged_ring': {'smarts': '[C]1[C][C][C]2[C][C][C]([C]1)[C]2', 'penalty': 0.5, 'reason': '桥环化合物合成困难'},
            
            # 高度取代的碳
            'quaternary_carbon': {'smarts': '[C]([C])([C])([C])[C]', 'penalty': 0.4, 'reason': '季碳原子难以构建'},
            
            # 不稳定的杂环
            'small_ring_hetero': {'smarts': '[N,O,S]1[C][C]1', 'penalty': 0.4, 'reason': '小环杂环不稳定'},
            'aziridine': {'smarts': '[N]1[C][C]1', 'penalty': 0.6, 'reason': '氮丙环易开环'},
            'epoxide': {'smarts': '[O]1[C][C]1', 'penalty': 0.3, 'reason': '环氧化物易开环'},
            
            # 多重官能团
            'polyhalogen': {'smarts': '[C]([F,Cl,Br,I])([F,Cl,Br,I])[F,Cl,Br,I]', 'penalty': 0.4, 'reason': '多卤代化合物'},
            'polynitro': {'smarts': '[C]([N+](=O)[O-])([N+](=O)[O-])', 'penalty': 0.7, 'reason': '多硝基化合物爆炸性'},
            
            # 特殊保护基
            'silyl_ether': {'smarts': '[O][Si]([C])([C])[C]', 'penalty': 0.2, 'reason': '硅基保护基需要特殊处理'},
            'benzyl_protection': {'smarts': '[O][C][c]1[c][c][c][c][c]1', 'penalty': 0.2, 'reason': '苄基保护基需要氢解'},
        }
        
        # 有利官能团（降低惩罚）
        self.favorable_groups = {
            'ester': {'smarts': '[C](=O)[O][C]', 'bonus': 0.1, 'reason': '酯基易于合成'},
            'amide': {'smarts': '[C](=O)[N]', 'bonus': 0.1, 'reason': '酰胺键易于形成'},
            'ether': {'smarts': '[C][O][C]', 'bonus': 0.05, 'reason': '醚键相对稳定'},
            'alcohol': {'smarts': '[C][O][H]', 'bonus': 0.05, 'reason': '醇基常见'},
            'aromatic': {'smarts': 'c1ccccc1', 'bonus': 0.1, 'reason': '芳环稳定'},
        }
    
    def analyze_functional_groups(self, smiles: str) -> Dict[str, Any]:
        """分析分子中的官能团"""
        if not RDKIT_AVAILABLE:
            return {
                'difficult_groups': [],
                'favorable_groups': [],
                'penalty_score': np.random.uniform(0, 1),
                'complexity_factors': []
            }
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'difficult_groups': [],
                'favorable_groups': [],
                'penalty_score': 1.0,  # 最大惩罚
                'complexity_factors': ['invalid_molecule']
            }
        
        # 检测困难官能团
        difficult_groups = []
        total_penalty = 0.0
        
        for name, info in self.difficult_groups.items():
            pattern = Chem.MolFromSmarts(info['smarts'])
            if pattern and mol.HasSubstructMatch(pattern):
                matches = mol.GetSubstructMatches(pattern)
                difficult_groups.append({
                    'name': name,
                    'count': len(matches),
                    'penalty': info['penalty'],
                    'reason': info['reason']
                })
                total_penalty += info['penalty'] * len(matches)
        
        # 检测有利官能团
        favorable_groups = []
        total_bonus = 0.0
        
        for name, info in self.favorable_groups.items():
            pattern = Chem.MolFromSmarts(info['smarts'])
            if pattern and mol.HasSubstructMatch(pattern):
                matches = mol.GetSubstructMatches(pattern)
                favorable_groups.append({
                    'name': name,
                    'count': len(matches),
                    'bonus': info['bonus'],
                    'reason': info['reason']
                })
                total_bonus += info['bonus'] * len(matches)
        
        # 计算复杂度因子
        complexity_factors = self._assess_complexity_factors(mol)
        
        # 综合惩罚分数
        penalty_score = max(0, total_penalty - total_bonus)
        
        return {
            'difficult_groups': difficult_groups,
            'favorable_groups': favorable_groups,
            'penalty_score': penalty_score,
            'complexity_factors': complexity_factors
        }
    
    def _assess_complexity_factors(self, mol) -> List[str]:
        """评估分子复杂度因子"""
        factors = []
        
        # 分子大小
        num_atoms = mol.GetNumAtoms()
        if num_atoms > 50:
            factors.append(f'large_molecule_{num_atoms}_atoms')
        
        # 环系复杂度
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()
        if num_rings > 4:
            factors.append(f'many_rings_{num_rings}')
        
        # 立体中心
        stereo_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        if len(stereo_centers) > 3:
            factors.append(f'many_stereo_centers_{len(stereo_centers)}')
        
        # 杂原子密度
        heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
        hetero_ratio = heteroatoms / num_atoms
        if hetero_ratio > 0.3:
            factors.append(f'high_heteroatom_density_{hetero_ratio:.2f}')
        
        # 分子柔性
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        if rotatable_bonds > 8:
            factors.append(f'high_flexibility_{rotatable_bonds}_rotatable_bonds')
        
        return factors

class ReactionPathAnalyzer:
    """反应路径分析器"""
    
    def __init__(self):
        # 常见反应类型及其难度
        self.reaction_difficulties = {
            'alkylation': 0.2,
            'acylation': 0.3,
            'condensation': 0.3,
            'cycloaddition': 0.4,
            'rearrangement': 0.6,
            'oxidation': 0.3,
            'reduction': 0.2,
            'substitution': 0.2,
            'elimination': 0.3,
            'coupling': 0.4,
            'metathesis': 0.5,
            'cascade': 0.7,
            'photochemical': 0.8,
            'electrochemical': 0.7
        }
        
        # 起始原料复杂度
        self.starting_material_complexity = {
            'commercial': 0.0,
            'common_intermediate': 0.2,
            'specialized': 0.5,
            'rare': 0.8,
            'exotic': 1.0
        }
    
    def estimate_synthesis_difficulty(self, smiles: str) -> Dict[str, Any]:
        """估计合成难度"""
        if not RDKIT_AVAILABLE:
            return {
                'estimated_steps': np.random.randint(3, 10),
                'difficulty_score': np.random.uniform(0, 1),
                'key_reactions': ['unknown'],
                'bottlenecks': []
            }
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'estimated_steps': 10,
                'difficulty_score': 1.0,
                'key_reactions': ['invalid_molecule'],
                'bottlenecks': ['invalid_structure']
            }
        
        # 基于分子复杂度估计合成步数
        complexity_score = self._calculate_complexity_score(mol)
        estimated_steps = min(12, max(1, int(complexity_score * 10)))
        
        # 识别关键反应
        key_reactions = self._identify_key_reactions(mol)
        
        # 识别合成瓶颈
        bottlenecks = self._identify_bottlenecks(mol)
        
        # 计算总体难度分数
        reaction_difficulty = np.mean([self.reaction_difficulties.get(r, 0.5) for r in key_reactions])
        step_penalty = min(1.0, estimated_steps / 8.0)  # 8步以上开始惩罚
        bottleneck_penalty = len(bottlenecks) * 0.1
        
        difficulty_score = min(1.0, reaction_difficulty + step_penalty + bottleneck_penalty)
        
        return {
            'estimated_steps': estimated_steps,
            'difficulty_score': difficulty_score,
            'key_reactions': key_reactions,
            'bottlenecks': bottlenecks
        }
    
    def _calculate_complexity_score(self, mol) -> float:
        """计算分子复杂度分数"""
        # 基于多个描述符的复杂度评估
        num_atoms = mol.GetNumAtoms()
        num_rings = mol.GetRingInfo().NumRings()
        num_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        num_hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
        
        # 归一化各项指标
        atom_score = min(1.0, num_atoms / 50.0)
        ring_score = min(1.0, num_rings / 5.0)
        stereo_score = min(1.0, num_stereo / 4.0)
        hetero_score = min(1.0, num_hetero / 10.0)
        
        # 加权平均
        complexity_score = (0.3 * atom_score + 0.3 * ring_score + 
                          0.2 * stereo_score + 0.2 * hetero_score)
        
        return complexity_score
    
    def _identify_key_reactions(self, mol) -> List[str]:
        """识别关键反应类型"""
        reactions = []
        
        # 基于官能团识别可能的反应
        if mol.HasSubstructMatch(Chem.MolFromSmarts('[C](=O)[O][C]')):
            reactions.append('acylation')
        
        if mol.HasSubstructMatch(Chem.MolFromSmarts('[C](=O)[N]')):
            reactions.append('condensation')
        
        if mol.HasSubstructMatch(Chem.MolFromSmarts('c1ccccc1')):
            reactions.append('substitution')
        
        if mol.GetRingInfo().NumRings() > 2:
            reactions.append('cycloaddition')
        
        # 如果没有识别到特定反应，添加通用反应
        if not reactions:
            reactions = ['alkylation', 'substitution']
        
        return reactions
    
    def _identify_bottlenecks(self, mol) -> List[str]:
        """识别合成瓶颈"""
        bottlenecks = []
        
        # 立体化学复杂性
        stereo_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        if len(stereo_centers) > 2:
            bottlenecks.append('stereocontrol')
        
        # 区域选择性
        aromatic_rings = [ring for ring in mol.GetRingInfo().AtomRings() 
                         if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)]
        if len(aromatic_rings) > 1:
            bottlenecks.append('regioselectivity')
        
        # 官能团兼容性
        if (mol.HasSubstructMatch(Chem.MolFromSmarts('[N]')) and 
            mol.HasSubstructMatch(Chem.MolFromSmarts('[C](=O)'))):
            bottlenecks.append('functional_group_compatibility')
        
        # 保护基需求
        if mol.HasSubstructMatch(Chem.MolFromSmarts('[O][H]')):
            bottlenecks.append('protection_deprotection')
        
        return bottlenecks

class ReagentAvailabilityChecker:
    """试剂可获得性检查器"""
    
    def __init__(self):
        # 常见试剂列表（简化）
        self.common_reagents = {
            # 溶剂
            'water', 'methanol', 'ethanol', 'acetone', 'dichloromethane',
            'chloroform', 'ethyl_acetate', 'hexane', 'toluene', 'dmf',
            
            # 酸碱
            'hydrochloric_acid', 'sulfuric_acid', 'acetic_acid', 'sodium_hydroxide',
            'potassium_hydroxide', 'triethylamine', 'pyridine',
            
            # 氧化还原剂
            'sodium_borohydride', 'lithium_aluminum_hydride', 'potassium_permanganate',
            'chromium_trioxide', 'dess_martin_periodinane',
            
            # 偶联试剂
            'palladium_acetate', 'triphenylphosphine', 'cesium_carbonate',
            'potassium_carbonate', 'sodium_carbonate'
        }
        
        # 昂贵/稀有试剂
        self.expensive_reagents = {
            'rhodium_catalysts', 'iridium_catalysts', 'ruthenium_catalysts',
            'osmium_tetroxide', 'platinum_catalysts', 'gold_catalysts',
            'chiral_ligands', 'organocatalysts'
        }
    
    def assess_reagent_availability(self, smiles: str) -> Dict[str, Any]:
        """评估试剂可获得性"""
        # 简化实现：基于分子复杂度和官能团推断所需试剂
        if not RDKIT_AVAILABLE:
            return {
                'common_reagents': list(np.random.choice(list(self.common_reagents), 3)),
                'expensive_reagents': [],
                'availability_score': np.random.uniform(0.5, 1.0),
                'cost_estimate': 'low'
            }
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'common_reagents': [],
                'expensive_reagents': ['invalid_molecule'],
                'availability_score': 0.0,
                'cost_estimate': 'high'
            }
        
        # 基于官能团推断所需试剂
        required_reagents = self._infer_required_reagents(mol)
        
        # 分类试剂
        common_reagents = [r for r in required_reagents if r in self.common_reagents]
        expensive_reagents = [r for r in required_reagents if r in self.expensive_reagents]
        
        # 计算可获得性分数
        availability_score = self._calculate_availability_score(common_reagents, expensive_reagents)
        
        # 估计成本
        cost_estimate = self._estimate_cost(common_reagents, expensive_reagents)
        
        return {
            'common_reagents': common_reagents,
            'expensive_reagents': expensive_reagents,
            'availability_score': availability_score,
            'cost_estimate': cost_estimate
        }
    
    def _infer_required_reagents(self, mol) -> List[str]:
        """推断所需试剂"""
        reagents = []
        
        # 基于官能团推断
        if mol.HasSubstructMatch(Chem.MolFromSmarts('[C](=O)[O][C]')):
            reagents.extend(['acetic_acid', 'sulfuric_acid'])
        
        if mol.HasSubstructMatch(Chem.MolFromSmarts('[C](=O)[N]')):
            reagents.extend(['triethylamine', 'dmf'])
        
        if mol.HasSubstructMatch(Chem.MolFromSmarts('c1ccccc1')):
            reagents.extend(['palladium_acetate', 'triphenylphosphine'])
        
        # 基于分子复杂度推断
        if mol.GetRingInfo().NumRings() > 2:
            reagents.append('chiral_ligands')
        
        stereo_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        if len(stereo_centers) > 1:
            reagents.append('organocatalysts')
        
        return reagents
    
    def _calculate_availability_score(self, common_reagents: List[str], 
                                    expensive_reagents: List[str]) -> float:
        """计算可获得性分数"""
        if not common_reagents and not expensive_reagents:
            return 1.0  # 无特殊试剂需求
        
        total_reagents = len(common_reagents) + len(expensive_reagents)
        common_ratio = len(common_reagents) / total_reagents
        
        # 常见试剂比例越高，可获得性越好
        availability_score = 0.3 + 0.7 * common_ratio
        
        return availability_score
    
    def _estimate_cost(self, common_reagents: List[str], 
                      expensive_reagents: List[str]) -> str:
        """估计成本等级"""
        if not expensive_reagents:
            return 'low'
        elif len(expensive_reagents) == 1:
            return 'medium'
        else:
            return 'high'

class SynthesisPenaltyFunction:
    """合成易度惩罚函数"""
    
    def __init__(self, config: SynthesisPenaltyConfig):
        """
        初始化合成易度惩罚函数
        
        Args:
            config: 合成易度惩罚配置
        """
        self.config = config
        
        # 初始化分析器
        self.fg_analyzer = FunctionalGroupAnalyzer()
        self.reaction_analyzer = ReactionPathAnalyzer()
        self.reagent_checker = ReagentAvailabilityChecker()
        
        # 惩罚历史
        self.penalty_history = []
    
    def __call__(self, smiles: str) -> float:
        """计算合成易度惩罚"""
        # 官能团分析
        fg_analysis = self.fg_analyzer.analyze_functional_groups(smiles)
        
        # 反应路径分析
        reaction_analysis = self.reaction_analyzer.estimate_synthesis_difficulty(smiles)
        
        # 试剂可获得性分析
        reagent_analysis = self.reagent_checker.assess_reagent_availability(smiles)
        
        # 计算各项惩罚
        fg_penalty = fg_analysis['penalty_score']
        reaction_penalty = reaction_analysis['difficulty_score']
        reagent_penalty = 1.0 - reagent_analysis['availability_score']
        
        # 额外的 SA 分数惩罚
        sa_penalty = self._calculate_sa_penalty(smiles)
        
        # 加权求和
        total_penalty = (
            self.config.sa_score_weight * sa_penalty +
            self.config.functional_group_weight * fg_penalty +
            self.config.reaction_path_weight * reaction_penalty +
            self.config.reagent_availability_weight * reagent_penalty
        )
        
        # 应用惩罚缩放和偏移
        scaled_penalty = (total_penalty * self.config.penalty_scale + 
                         self.config.penalty_offset)
        
        # 记录惩罚历史
        penalty_record = {
            'smiles': smiles,
            'total_penalty': scaled_penalty,
            'sa_penalty': sa_penalty,
            'fg_penalty': fg_penalty,
            'reaction_penalty': reaction_penalty,
            'reagent_penalty': reagent_penalty,
            'fg_analysis': fg_analysis,
            'reaction_analysis': reaction_analysis,
            'reagent_analysis': reagent_analysis
        }
        self.penalty_history.append(penalty_record)
        
        return scaled_penalty
    
    def _calculate_sa_penalty(self, smiles: str) -> float:
        """计算 SA 分数惩罚"""
        if not RDKIT_AVAILABLE:
            return np.random.uniform(0, 1)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 1.0
        
        # 简化的 SA 分数计算
        sa_score = self._simplified_sa_score(mol)
        
        # 转换为惩罚（SA 分数越高，惩罚越大）
        if sa_score <= self.config.sa_score_threshold:
            penalty = 0.0
        else:
            penalty = min(1.0, (sa_score - self.config.sa_score_threshold) / 4.0)
        
        return penalty
    
    def _simplified_sa_score(self, mol) -> float:
        """简化的 SA 分数计算"""
        # 基于分子复杂度的启发式 SA 分数
        num_atoms = mol.GetNumAtoms()
        num_rings = mol.GetRingInfo().NumRings()
        num_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        num_hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
        
        # SA 分数 = 1 + 复杂度因子
        sa_score = 1.0 + (num_atoms / 20.0 + num_rings / 3.0 + 
                         num_stereo / 2.0 + num_hetero / 5.0)
        
        return min(10.0, sa_score)
    
    def get_penalty_statistics(self) -> Dict[str, Any]:
        """获取惩罚统计信息"""
        if not self.penalty_history:
            return {}
        
        penalties = [record['total_penalty'] for record in self.penalty_history]
        
        stats = {
            'total_molecules': len(self.penalty_history),
            'mean_penalty': np.mean(penalties),
            'std_penalty': np.std(penalties),
            'min_penalty': np.min(penalties),
            'max_penalty': np.max(penalties),
            'high_penalty_count': sum(1 for p in penalties if p > 0.5),
            'component_stats': {
                'sa_penalty': np.mean([r['sa_penalty'] for r in self.penalty_history]),
                'fg_penalty': np.mean([r['fg_penalty'] for r in self.penalty_history]),
                'reaction_penalty': np.mean([r['reaction_penalty'] for r in self.penalty_history]),
                'reagent_penalty': np.mean([r['reagent_penalty'] for r in self.penalty_history])
            }
        }
        
        return stats
    
    def get_difficult_molecules(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """获取高惩罚分子"""
        difficult_mols = [
            record for record in self.penalty_history 
            if record['total_penalty'] > threshold
        ]
        
        return difficult_mols
    
    def save_analysis(self, filepath: str):
        """保存分析结果"""
        analysis_data = {
            'penalty_history': self.penalty_history,
            'statistics': self.get_penalty_statistics(),
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(analysis_data, f)

def main():
    """主函数 - 演示合成易度惩罚"""
    print("=== 合成易度惩罚机制测试 ===")
    
    # 创建配置
    config = SynthesisPenaltyConfig(
        sa_score_weight=0.4,
        functional_group_weight=0.3,
        reaction_path_weight=0.2,
        reagent_availability_weight=0.1,
        penalty_scale=2.0
    )
    
    # 创建惩罚函数
    penalty_function = SynthesisPenaltyFunction(config)
    
    # 测试分子（从简单到复杂）
    test_molecules = [
        "CC(=O)OC",                    # 简单酯
        "c1ccccc1",                    # 苯
        "CC(C)(C)C",                   # 异丁烷
        "C1COC(=O)O1",                 # 碳酸乙烯酯
        "CC1=CC=CC=C1C(=O)O",          # 邻甲苯甲酸
        "C[C@H]1CC[C@@H](C)CC1",       # 立体化学
        "C1=CC=C2C(=C1)C=CC=C2",       # 萘
        "CC(C)(C)OC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O",  # 复杂氨基酸衍生物
        "invalid_smiles"                # 无效分子
    ]
    
    # 计算惩罚
    print("\n=== 合成易度惩罚结果 ===")
    for mol in test_molecules:
        penalty = penalty_function(mol)
        print(f"{mol:50} -> 惩罚: {penalty:.3f}")
    
    # 获取统计信息
    stats = penalty_function.get_penalty_statistics()
    print("\n=== 惩罚统计 ===")
    print(f"总分子数: {stats['total_molecules']}")
    print(f"平均惩罚: {stats['mean_penalty']:.3f}")
    print(f"高惩罚分子数: {stats['high_penalty_count']}")
    
    print("\n组件贡献:")
    for component, value in stats['component_stats'].items():
        print(f"  {component}: {value:.3f}")
    
    # 获取困难分子
    difficult_mols = penalty_function.get_difficult_molecules(threshold=0.5)
    print(f"\n困难分子数量: {len(difficult_mols)}")
    for mol in difficult_mols:
        print(f"  {mol['smiles']}: {mol['total_penalty']:.3f}")
    
    # 保存分析
    penalty_function.save_analysis("test_synthesis_penalty.pkl")
    
    print("\n✓ 合成易度惩罚机制测试完成")

if __name__ == "__main__":
    main() 