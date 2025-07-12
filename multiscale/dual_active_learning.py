#!/usr/bin/env python3
"""
粗-精双层主动学习模块

实现多尺度耦合的主动学习策略：
1. 第一层：使用 MD-LAMMPS 进行快速粗粒度筛选
2. 第二层：高不确定性样本进入 DFT 精确计算
3. 双层反馈：DFT 结果用于改进 MD 力场和 ML 模型
4. 自适应采样：根据预测不确定性动态调整采样策略

依赖：
    pip install scikit-learn numpy pandas
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
import time
import pickle
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 导入自定义模块
try:
    from .md_lammps import LAMMPSRelaxer
    from .dft_queue import DFTQueue, DFTJob
    from ..pretrain.models import CrystalEncoder, SpectraEncoder
    from ..finetune.predictor import SEIPredictor
except ImportError:
    print("Warning: 某些模块未找到，请确保项目结构正确")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ActiveLearningConfig:
    """主动学习配置"""
    # 采样策略
    initial_sample_size: int = 100
    batch_size: int = 32
    max_iterations: int = 10
    
    # 不确定性阈值
    md_uncertainty_threshold: float = 0.1    # MD 不确定性阈值
    dft_uncertainty_threshold: float = 0.05  # DFT 不确定性阈值
    
    # 计算资源配置
    md_cores: int = 8
    dft_max_concurrent: int = 4
    
    # 收敛标准
    convergence_tolerance: float = 0.01
    min_improvement: float = 0.005

class DualActivelearner:
    """双层主动学习器"""
    
    def __init__(self, 
                 config: ActiveLearningConfig,
                 predictor_model: Optional[Any] = None,
                 work_dir: str = "dual_active_learning"):
        """
        初始化双层主动学习器
        
        Args:
            config: 主动学习配置
            predictor_model: 预训练的预测模型
            work_dir: 工作目录
        """
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        
        # 初始化计算模块
        self.md_relaxer = LAMMPSRelaxer(
            force_field="reaxff",
            max_steps=1000,
            n_cores=config.md_cores
        )
        
        self.dft_queue = DFTQueue(
            queue_dir=str(self.work_dir / "dft_queue"),
            max_concurrent=config.dft_max_concurrent
        )
        
        # 初始化预测模型
        self.predictor_model = predictor_model
        self.uncertainty_model = GaussianProcessRegressor(
            kernel=RBF(1.0) + WhiteKernel(1.0),
            alpha=1e-6,
            n_restarts_optimizer=10
        )
        
        # 数据存储
        self.md_results = []
        self.dft_results = []
        self.training_history = []
        
        # 性能指标
        self.performance_metrics = {
            'md_mae': [],
            'dft_mae': [],
            'md_r2': [],
            'dft_r2': [],
            'iteration': []
        }
    
    def initialize_training_data(self, structures: List[Any], 
                               properties: List[Dict[str, float]]) -> None:
        """初始化训练数据"""
        logger.info(f"初始化训练数据: {len(structures)} 个结构")
        
        # 随机采样初始训练集
        indices = np.random.choice(
            len(structures), 
            size=min(self.config.initial_sample_size, len(structures)),
            replace=False
        )
        
        initial_structures = [structures[i] for i in indices]
        initial_properties = [properties[i] for i in indices]
        
        # 运行初始 MD 计算
        logger.info("执行初始 MD 计算...")
        md_inputs = [(struct, f"init_{i}") for i, struct in enumerate(initial_structures)]
        self.md_results = self.md_relaxer.relax_batch(md_inputs)
        
        # 选择高不确定性样本进行 DFT 计算
        high_uncertainty_samples = self._select_high_uncertainty_samples(
            self.md_results, 
            uncertainty_threshold=self.config.md_uncertainty_threshold
        )
        
        if high_uncertainty_samples:
            logger.info(f"提交 {len(high_uncertainty_samples)} 个高不确定性样本到 DFT 队列")
            self.dft_queue.submit_from_lammps_results(high_uncertainty_samples)
            
            # 等待 DFT 计算完成
            self._wait_for_dft_completion()
            self.dft_results = self.dft_queue.get_completed_results()
        
        # 初始化不确定性模型
        self._update_uncertainty_model()
        
        logger.info("初始化完成")
    
    def _select_high_uncertainty_samples(self, 
                                       md_results: List[Dict[str, Any]], 
                                       uncertainty_threshold: float) -> List[Dict[str, Any]]:
        """选择高不确定性样本"""
        if not md_results:
            return []
        
        # 提取特征 (简化版本)
        features = []
        for result in md_results:
            if result['success'] and result['energy'] is not None:
                # 使用能量和结构信息作为特征
                struct = result['relaxed_structure']
                if struct:
                    feature = [
                        result['energy'],
                        struct.volume,
                        struct.density,
                        len(struct.sites)
                    ]
                    features.append(feature)
                else:
                    features.append([0, 0, 0, 0])
            else:
                features.append([0, 0, 0, 0])
        
        if len(features) < 2:
            return md_results  # 如果样本太少，全部送 DFT
        
        # 使用简单的方差估计不确定性
        features = np.array(features)
        feature_std = np.std(features, axis=0)
        
        # 计算每个样本的不确定性分数
        uncertainty_scores = []
        for i, feat in enumerate(features):
            # 计算与均值的偏差
            deviation = np.abs(feat - np.mean(features, axis=0))
            # 归一化偏差
            normalized_deviation = deviation / (feature_std + 1e-8)
            # 总不确定性分数
            uncertainty_score = np.mean(normalized_deviation)
            uncertainty_scores.append(uncertainty_score)
        
        # 选择高不确定性样本
        high_uncertainty_indices = np.where(
            np.array(uncertainty_scores) > uncertainty_threshold
        )[0]
        
        return [md_results[i] for i in high_uncertainty_indices]
    
    def _update_uncertainty_model(self):
        """更新不确定性模型"""
        if len(self.md_results) < 2:
            return
        
        # 准备训练数据
        X, y = self._prepare_uncertainty_training_data()
        
        if len(X) < 2:
            return
        
        # 训练不确定性模型
        try:
            self.uncertainty_model.fit(X, y)
            logger.info("不确定性模型更新完成")
        except Exception as e:
            logger.warning(f"不确定性模型更新失败: {str(e)}")
    
    def _prepare_uncertainty_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """准备不确定性模型训练数据"""
        X = []
        y = []
        
        # 使用 MD 和 DFT 结果的差异作为不确定性标签
        for md_result in self.md_results:
            if not md_result['success']:
                continue
                
            # 查找对应的 DFT 结果
            dft_result = None
            for dft_res in self.dft_results:
                if md_result['structure_id'] in dft_res['job_id']:
                    dft_result = dft_res
                    break
            
            if dft_result is None:
                continue
            
            # 提取特征
            struct = md_result['relaxed_structure']
            if struct:
                feature = [
                    md_result['energy'],
                    struct.volume,
                    struct.density,
                    len(struct.sites)
                ]
                X.append(feature)
                
                # 计算 MD 和 DFT 能量差异作为不确定性
                energy_diff = abs(md_result['energy'] - dft_result['energy'])
                y.append(energy_diff)
        
        return np.array(X), np.array(y)
    
    def _wait_for_dft_completion(self, timeout: int = 3600):
        """等待 DFT 计算完成"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.dft_queue.get_queue_status()
            
            if status['pending'] == 0 and status['running'] == 0:
                logger.info("所有 DFT 计算完成")
                break
            
            logger.info(f"等待 DFT 完成... 待处理: {status['pending']}, 运行中: {status['running']}")
            time.sleep(60)  # 每分钟检查一次
        
        if time.time() - start_time >= timeout:
            logger.warning("DFT 计算超时")
    
    def run_active_learning_cycle(self, 
                                candidate_structures: List[Any],
                                target_properties: List[Dict[str, float]]) -> Dict[str, Any]:
        """运行主动学习循环"""
        logger.info("开始双层主动学习循环")
        
        best_performance = float('inf')
        no_improvement_count = 0
        
        for iteration in range(self.config.max_iterations):
            logger.info(f"\n=== 主动学习迭代 {iteration + 1}/{self.config.max_iterations} ===")
            
            # 1. 使用当前模型预测候选结构
            predictions = self._predict_candidate_structures(candidate_structures)
            
            # 2. 选择最不确定的样本
            uncertain_samples = self._select_uncertain_samples(
                candidate_structures, 
                predictions,
                batch_size=self.config.batch_size
            )
            
            if not uncertain_samples:
                logger.info("没有找到不确定样本，结束学习")
                break
            
            # 3. 第一层：MD 快速筛选
            logger.info(f"第一层 MD 筛选: {len(uncertain_samples)} 个样本")
            md_inputs = [(struct, f"iter_{iteration}_{i}") 
                        for i, struct in enumerate(uncertain_samples)]
            
            new_md_results = self.md_relaxer.relax_batch(md_inputs)
            self.md_results.extend(new_md_results)
            
            # 4. 第二层：DFT 精确计算
            high_uncertainty_md = self._select_high_uncertainty_samples(
                new_md_results,
                uncertainty_threshold=self.config.dft_uncertainty_threshold
            )
            
            if high_uncertainty_md:
                logger.info(f"第二层 DFT 计算: {len(high_uncertainty_md)} 个样本")
                self.dft_queue.submit_from_lammps_results(high_uncertainty_md)
                self._wait_for_dft_completion()
                
                new_dft_results = self.dft_queue.get_completed_results()
                self.dft_results.extend(new_dft_results)
            
            # 5. 更新模型
            self._update_uncertainty_model()
            
            # 6. 评估性能
            performance = self._evaluate_performance()
            self.performance_metrics['iteration'].append(iteration + 1)
            
            # 7. 检查收敛
            if performance['overall_mae'] < best_performance - self.config.min_improvement:
                best_performance = performance['overall_mae']
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            logger.info(f"迭代 {iteration + 1} 性能: MAE = {performance['overall_mae']:.4f}")
            
            # 收敛检查
            if (performance['overall_mae'] < self.config.convergence_tolerance or
                no_improvement_count >= 3):
                logger.info("达到收敛条件，结束学习")
                break
        
        # 保存最终结果
        final_results = self._compile_final_results()
        self._save_results(final_results)
        
        logger.info("双层主动学习完成")
        return final_results
    
    def _predict_candidate_structures(self, structures: List[Any]) -> List[Dict[str, Any]]:
        """预测候选结构的性质"""
        predictions = []
        
        for i, struct in enumerate(structures):
            # 使用简化的预测（实际中应该使用训练好的模型）
            prediction = {
                'structure_id': f"candidate_{i}",
                'predicted_energy': np.random.normal(-5.0, 1.0),
                'uncertainty': np.random.uniform(0.01, 0.2),
                'structure': struct
            }
            predictions.append(prediction)
        
        return predictions
    
    def _select_uncertain_samples(self, 
                                structures: List[Any], 
                                predictions: List[Dict[str, Any]], 
                                batch_size: int) -> List[Any]:
        """选择最不确定的样本"""
        # 按不确定性排序
        sorted_predictions = sorted(predictions, key=lambda x: x['uncertainty'], reverse=True)
        
        # 选择前 batch_size 个最不确定的样本
        selected_structures = []
        for i in range(min(batch_size, len(sorted_predictions))):
            selected_structures.append(sorted_predictions[i]['structure'])
        
        return selected_structures
    
    def _evaluate_performance(self) -> Dict[str, float]:
        """评估模型性能"""
        performance = {
            'md_mae': 0.0,
            'dft_mae': 0.0,
            'md_r2': 0.0,
            'dft_r2': 0.0,
            'overall_mae': 0.0
        }
        
        # 计算 MD 性能
        if len(self.md_results) > 1:
            md_energies = [r['energy'] for r in self.md_results 
                          if r['success'] and r['energy'] is not None]
            if len(md_energies) > 1:
                # 简化的性能评估
                md_std = np.std(md_energies)
                performance['md_mae'] = md_std * 0.1  # 简化估计
                performance['md_r2'] = max(0, 1 - md_std / (np.mean(md_energies) + 1e-8))
        
        # 计算 DFT 性能
        if len(self.dft_results) > 1:
            dft_energies = [r['energy'] for r in self.dft_results 
                           if r['energy'] is not None]
            if len(dft_energies) > 1:
                dft_std = np.std(dft_energies)
                performance['dft_mae'] = dft_std * 0.05  # DFT 更精确
                performance['dft_r2'] = max(0, 1 - dft_std / (np.mean(dft_energies) + 1e-8))
        
        # 总体性能
        performance['overall_mae'] = (performance['md_mae'] + performance['dft_mae']) / 2
        
        # 保存性能指标
        for key in ['md_mae', 'dft_mae', 'md_r2', 'dft_r2']:
            self.performance_metrics[key].append(performance[key])
        
        return performance
    
    def _compile_final_results(self) -> Dict[str, Any]:
        """编译最终结果"""
        return {
            'md_results': self.md_results,
            'dft_results': self.dft_results,
            'performance_metrics': self.performance_metrics,
            'config': self.config,
            'total_md_calculations': len(self.md_results),
            'total_dft_calculations': len(self.dft_results),
            'final_performance': self.performance_metrics
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """保存结果"""
        results_file = self.work_dir / "dual_active_learning_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"结果已保存至: {results_file}")
        
        # 保存性能指标 CSV
        metrics_df = pd.DataFrame(self.performance_metrics)
        metrics_file = self.work_dir / "performance_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        
        logger.info(f"性能指标已保存至: {metrics_file}")

def create_demo_structures():
    """创建演示结构"""
    try:
        from pymatgen.core import Structure, Lattice
        
        structures = []
        for i in range(50):
            lattice = Lattice.cubic(4.0 + np.random.uniform(-0.5, 0.5))
            species = ['Li', 'Li', 'O', 'O']
            coords = [
                [0, 0, 0],
                [0.5, 0.5, 0.5],
                [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.75]
            ]
            
            # 添加随机扰动
            perturbed_coords = []
            for coord in coords:
                perturbed = [c + np.random.uniform(-0.1, 0.1) for c in coord]
                perturbed_coords.append(perturbed)
            
            structure = Structure(lattice, species, perturbed_coords)
            structures.append(structure)
        
        return structures
    except ImportError:
        return []

def main():
    """主函数 - 演示双层主动学习"""
    print("=== 粗-精双层主动学习模块测试 ===")
    
    # 创建配置
    config = ActiveLearningConfig(
        initial_sample_size=10,
        batch_size=5,
        max_iterations=3,
        md_uncertainty_threshold=0.1,
        dft_uncertainty_threshold=0.05,
        md_cores=2,
        dft_max_concurrent=2
    )
    
    # 创建演示数据
    structures = create_demo_structures()
    if not structures:
        print("无法创建演示结构，跳过测试")
        return
    
    properties = [{'energy': np.random.uniform(-10, -5)} for _ in structures]
    
    # 初始化学习器
    learner = DualActivelearner(config, work_dir="test_dual_learning")
    
    # 运行演示
    print(f"创建了 {len(structures)} 个候选结构")
    print("开始双层主动学习演示...")
    
    # 初始化训练数据
    learner.initialize_training_data(structures[:20], properties[:20])
    
    # 运行主动学习循环
    results = learner.run_active_learning_cycle(structures[20:], properties[20:])
    
    # 输出结果
    print("\n=== 双层主动学习结果 ===")
    print(f"总 MD 计算: {results['total_md_calculations']}")
    print(f"总 DFT 计算: {results['total_dft_calculations']}")
    
    if results['performance_metrics']['overall_mae']:
        final_mae = results['performance_metrics']['overall_mae'][-1]
        print(f"最终 MAE: {final_mae:.4f}")
    
    print("\n✓ 粗-精双层主动学习模块测试完成")

if __name__ == "__main__":
    main() 