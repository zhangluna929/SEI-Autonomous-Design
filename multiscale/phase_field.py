#!/usr/bin/env python3
"""
相场模拟耦合模块

实现相场模拟与分子尺度计算的耦合，用于预测：
1. 微观裂纹的成核与扩展
2. SEI 界面的长时演化
3. 应力场与化学场的耦合
4. 多相界面的动力学行为

基于 Cahn-Hilliard 和 Allen-Cahn 方程的相场模型：
- 化学相分离
- 界面能最小化
- 应力-化学耦合
- 电化学反应动力学

依赖：
    pip install numpy scipy matplotlib fipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
from dataclasses import dataclass
import pickle
import time

try:
    import fipy as fp
    FIPY_AVAILABLE = True
except ImportError:
    FIPY_AVAILABLE = False
    print("Warning: FiPy not available. Using simplified phase field implementation.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PhaseFieldConfig:
    """相场模拟配置"""
    # 网格参数
    nx: int = 128
    ny: int = 128
    dx: float = 1.0  # 网格间距 (nm)
    dy: float = 1.0
    
    # 时间参数
    dt: float = 0.01  # 时间步长
    total_time: float = 100.0  # 总模拟时间
    
    # 物理参数
    interface_width: float = 2.0  # 界面宽度 (nm)
    interface_energy: float = 0.1  # 界面能 (J/m²)
    mobility: float = 1.0  # 迁移率
    
    # 化学参数
    diffusivity: float = 1e-9  # 扩散系数 (m²/s)
    reaction_rate: float = 1e-6  # 反应速率
    
    # 力学参数
    elastic_modulus: float = 10e9  # 弹性模量 (Pa)
    poisson_ratio: float = 0.3  # 泊松比
    
    # 电化学参数
    exchange_current: float = 1e-3  # 交换电流密度 (A/m²)
    overpotential: float = 0.1  # 过电位 (V)

class PhaseFieldSimulator:
    """相场模拟器"""
    
    def __init__(self, config: PhaseFieldConfig, work_dir: str = "phase_field_sim"):
        """
        初始化相场模拟器
        
        Args:
            config: 相场模拟配置
            work_dir: 工作目录
        """
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        
        # 创建网格
        self.x = np.linspace(0, config.nx * config.dx, config.nx)
        self.y = np.linspace(0, config.ny * config.dy, config.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # 初始化场变量
        self.phi = None      # 相场序参数 (0: 电解质, 1: SEI)
        self.c = None        # 浓度场
        self.stress = None   # 应力场
        self.potential = None # 电位场
        
        # 时间演化记录
        self.time_history = []
        self.phi_history = []
        self.crack_evolution = []
        
        # 使用 FiPy 或简化实现
        self.use_fipy = FIPY_AVAILABLE
        if self.use_fipy:
            self._setup_fipy_simulation()
        else:
            self._setup_simplified_simulation()
    
    def _setup_fipy_simulation(self):
        """设置 FiPy 相场模拟"""
        # 创建 FiPy 网格
        self.mesh = fp.Grid2D(
            nx=self.config.nx, 
            ny=self.config.ny,
            dx=self.config.dx,
            dy=self.config.dy
        )
        
        # 定义场变量
        self.phi_var = fp.CellVariable(mesh=self.mesh, name="phase field")
        self.c_var = fp.CellVariable(mesh=self.mesh, name="concentration")
        self.stress_var = fp.CellVariable(mesh=self.mesh, name="stress")
        
        # 设置初始条件
        self._set_initial_conditions_fipy()
        
        # 定义方程
        self._setup_equations_fipy()
    
    def _setup_simplified_simulation(self):
        """设置简化相场模拟"""
        # 初始化场变量
        self.phi = np.zeros((self.config.ny, self.config.nx))
        self.c = np.ones((self.config.ny, self.config.nx))
        self.stress = np.zeros((self.config.ny, self.config.nx))
        self.potential = np.zeros((self.config.ny, self.config.nx))
        
        # 设置初始条件
        self._set_initial_conditions_simplified()
    
    def _set_initial_conditions_fipy(self):
        """设置 FiPy 初始条件"""
        # 创建初始界面
        x0, y0 = self.config.nx * self.config.dx / 2, self.config.ny * self.config.dy / 2
        radius = min(self.config.nx, self.config.ny) * 0.2
        
        # 相场初始条件：中心为 SEI 相
        self.phi_var.setValue(0.0)
        distance = ((self.mesh.cellCenters[0] - x0)**2 + 
                   (self.mesh.cellCenters[1] - y0)**2)**0.5
        
        # 使用 tanh 函数创建平滑界面
        interface_profile = 0.5 * (1 + np.tanh((radius - distance) / self.config.interface_width))
        self.phi_var.setValue(interface_profile)
        
        # 浓度初始条件
        self.c_var.setValue(1.0)
        
        # 应力初始条件
        self.stress_var.setValue(0.0)
    
    def _set_initial_conditions_simplified(self):
        """设置简化模拟初始条件"""
        # 创建初始 SEI 层
        center_x, center_y = self.config.nx // 2, self.config.ny // 2
        radius = min(self.config.nx, self.config.ny) // 4
        
        # 创建圆形 SEI 区域
        for i in range(self.config.ny):
            for j in range(self.config.nx):
                distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                if distance < radius:
                    self.phi[i, j] = 1.0  # SEI 相
                else:
                    self.phi[i, j] = 0.0  # 电解质相
        
        # 添加界面平滑化
        self.phi = ndimage.gaussian_filter(self.phi, sigma=1.0)
        
        # 设置初始浓度梯度
        self.c = 1.0 - 0.5 * self.phi
        
        # 添加初始应力集中
        self._add_initial_stress_concentration()
    
    def _add_initial_stress_concentration(self):
        """添加初始应力集中"""
        # 在 SEI/电解质界面处添加应力
        grad_phi_x = np.gradient(self.phi, axis=1)
        grad_phi_y = np.gradient(self.phi, axis=0)
        grad_phi_magnitude = np.sqrt(grad_phi_x**2 + grad_phi_y**2)
        
        # 应力与界面梯度成正比
        self.stress = grad_phi_magnitude * self.config.elastic_modulus * 0.001
    
    def _setup_equations_fipy(self):
        """设置 FiPy 方程"""
        # Allen-Cahn 方程 (相场演化)
        # ∂φ/∂t = -M * δF/δφ
        
        # 自由能密度的变分
        free_energy_variation = (
            self.phi_var * (self.phi_var - 1) * (self.phi_var - 0.5) -
            self.config.interface_energy * self.phi_var.faceGrad.divergence
        )
        
        self.phi_equation = (
            fp.TransientTerm(var=self.phi_var) ==
            fp.DiffusionTerm(coeff=self.config.mobility, var=self.phi_var) -
            fp.ImplicitSourceTerm(coeff=self.config.mobility, var=self.phi_var) * free_energy_variation
        )
        
        # Cahn-Hilliard 方程 (浓度演化)
        # ∂c/∂t = ∇·(D∇μ), μ = δF/δc
        
        chemical_potential = (
            self.c_var * (1 - self.c_var) * (1 - 2 * self.c_var) -
            self.config.interface_energy * self.c_var.faceGrad.divergence
        )
        
        self.c_equation = (
            fp.TransientTerm(var=self.c_var) ==
            fp.DiffusionTerm(coeff=self.config.diffusivity, var=chemical_potential)
        )
        
        # 应力平衡方程 (简化)
        self.stress_equation = (
            fp.DiffusionTerm(coeff=1.0, var=self.stress_var) ==
            fp.ImplicitSourceTerm(coeff=1.0, var=self.stress_var) * 
            self.config.elastic_modulus * self.phi_var.faceGrad.divergence
        )
    
    def run_simulation(self) -> Dict[str, Any]:
        """运行相场模拟"""
        logger.info("开始相场模拟...")
        
        start_time = time.time()
        n_steps = int(self.config.total_time / self.config.dt)
        
        # 记录初始状态
        self._record_state(0)
        
        if self.use_fipy:
            results = self._run_fipy_simulation(n_steps)
        else:
            results = self._run_simplified_simulation(n_steps)
        
        end_time = time.time()
        
        logger.info(f"相场模拟完成，耗时: {end_time - start_time:.2f} 秒")
        
        # 分析结果
        analysis = self._analyze_results()
        
        # 保存结果
        self._save_results(results, analysis)
        
        return {
            'simulation_results': results,
            'analysis': analysis,
            'config': self.config,
            'runtime': end_time - start_time
        }
    
    def _run_fipy_simulation(self, n_steps: int) -> Dict[str, Any]:
        """运行 FiPy 相场模拟"""
        results = {
            'time_points': [],
            'phi_snapshots': [],
            'concentration_snapshots': [],
            'stress_snapshots': []
        }
        
        # 创建联合方程
        equations = self.phi_equation & self.c_equation & self.stress_equation
        
        for step in range(n_steps):
            # 求解方程
            equations.solve(dt=self.config.dt)
            
            # 记录状态
            if step % (n_steps // 10) == 0:
                current_time = step * self.config.dt
                results['time_points'].append(current_time)
                results['phi_snapshots'].append(self.phi_var.value.copy())
                results['concentration_snapshots'].append(self.c_var.value.copy())
                results['stress_snapshots'].append(self.stress_var.value.copy())
                
                logger.info(f"步骤 {step}/{n_steps}, 时间: {current_time:.2f}")
        
        return results
    
    def _run_simplified_simulation(self, n_steps: int) -> Dict[str, Any]:
        """运行简化相场模拟"""
        results = {
            'time_points': [],
            'phi_snapshots': [],
            'concentration_snapshots': [],
            'stress_snapshots': []
        }
        
        for step in range(n_steps):
            # 更新相场
            self._update_phase_field()
            
            # 更新浓度场
            self._update_concentration_field()
            
            # 更新应力场
            self._update_stress_field()
            
            # 记录状态
            if step % (n_steps // 10) == 0:
                current_time = step * self.config.dt
                results['time_points'].append(current_time)
                results['phi_snapshots'].append(self.phi.copy())
                results['concentration_snapshots'].append(self.c.copy())
                results['stress_snapshots'].append(self.stress.copy())
                
                self._record_state(current_time)
                logger.info(f"步骤 {step}/{n_steps}, 时间: {current_time:.2f}")
        
        return results
    
    def _update_phase_field(self):
        """更新相场 (Allen-Cahn 方程)"""
        # 计算拉普拉斯算子
        laplacian = self._compute_laplacian(self.phi)
        
        # 双势井项
        double_well = self.phi * (self.phi - 1) * (self.phi - 0.5)
        
        # 应力耦合项
        stress_coupling = self.stress * 0.001
        
        # 相场演化
        dphi_dt = (self.config.mobility * 
                  (self.config.interface_energy * laplacian - double_well - stress_coupling))
        
        # 更新相场
        self.phi += self.config.dt * dphi_dt
        
        # 保持边界条件
        self.phi = np.clip(self.phi, 0, 1)
    
    def _update_concentration_field(self):
        """更新浓度场 (Cahn-Hilliard 方程)"""
        # 计算化学势
        chemical_potential = (self.c * (1 - self.c) * (1 - 2 * self.c) - 
                            self.config.interface_energy * self._compute_laplacian(self.c))
        
        # 浓度演化
        dc_dt = self.config.diffusivity * self._compute_laplacian(chemical_potential)
        
        # 电化学反应项
        reaction_term = self._compute_reaction_term()
        
        # 更新浓度
        self.c += self.config.dt * (dc_dt + reaction_term)
        
        # 保持物理约束
        self.c = np.clip(self.c, 0, 1)
    
    def _update_stress_field(self):
        """更新应力场"""
        # 计算相场梯度
        grad_phi_x = np.gradient(self.phi, axis=1)
        grad_phi_y = np.gradient(self.phi, axis=0)
        
        # 界面应力
        interface_stress = (grad_phi_x**2 + grad_phi_y**2) * self.config.elastic_modulus * 0.01
        
        # 浓度梯度应力
        grad_c_x = np.gradient(self.c, axis=1)
        grad_c_y = np.gradient(self.c, axis=0)
        concentration_stress = (grad_c_x**2 + grad_c_y**2) * self.config.elastic_modulus * 0.005
        
        # 应力松弛
        stress_relaxation = -self.stress * 0.1
        
        # 更新应力
        self.stress += self.config.dt * (interface_stress + concentration_stress + stress_relaxation)
    
    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """计算拉普拉斯算子"""
        # 使用有限差分方法
        laplacian = np.zeros_like(field)
        
        # 内部点
        laplacian[1:-1, 1:-1] = (
            (field[2:, 1:-1] - 2 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / self.config.dy**2 +
            (field[1:-1, 2:] - 2 * field[1:-1, 1:-1] + field[1:-1, :-2]) / self.config.dx**2
        )
        
        # 边界条件 (零通量)
        laplacian[0, :] = laplacian[1, :]
        laplacian[-1, :] = laplacian[-2, :]
        laplacian[:, 0] = laplacian[:, 1]
        laplacian[:, -1] = laplacian[:, -2]
        
        return laplacian
    
    def _compute_reaction_term(self) -> np.ndarray:
        """计算电化学反应项"""
        # 简化的 Butler-Volmer 反应动力学
        # i = i₀ * (exp(αnFη/RT) - exp(-(1-α)nFη/RT))
        
        # 在 SEI/电解质界面处发生反应
        interface_mask = (self.phi > 0.1) & (self.phi < 0.9)
        
        reaction_term = np.zeros_like(self.c)
        
        # 简化的反应速率
        reaction_rate = (self.config.reaction_rate * 
                        np.exp(self.config.overpotential) * 
                        interface_mask)
        
        # 消耗电解质，生成 SEI
        reaction_term = -reaction_rate * self.c
        
        return reaction_term
    
    def _record_state(self, time: float):
        """记录当前状态"""
        self.time_history.append(time)
        if not self.use_fipy:
            self.phi_history.append(self.phi.copy())
            
            # 检测裂纹
            crack_info = self._detect_cracks()
            self.crack_evolution.append(crack_info)
    
    def _detect_cracks(self) -> Dict[str, Any]:
        """检测微观裂纹"""
        # 基于应力场检测裂纹
        stress_threshold = np.percentile(self.stress, 95)  # 95% 分位数
        crack_mask = self.stress > stress_threshold
        
        # 计算裂纹特征
        crack_area = np.sum(crack_mask) * self.config.dx * self.config.dy
        crack_perimeter = self._compute_perimeter(crack_mask)
        
        # 裂纹数量 (连通域)
        labeled_cracks, num_cracks = ndimage.label(crack_mask)
        
        return {
            'crack_area': crack_area,
            'crack_perimeter': crack_perimeter,
            'num_cracks': num_cracks,
            'max_stress': np.max(self.stress),
            'mean_stress': np.mean(self.stress)
        }
    
    def _compute_perimeter(self, mask: np.ndarray) -> float:
        """计算二值掩模的周长"""
        # 使用边缘检测
        edges = ndimage.binary_erosion(mask) ^ mask
        return np.sum(edges) * self.config.dx
    
    def _analyze_results(self) -> Dict[str, Any]:
        """分析模拟结果"""
        analysis = {
            'crack_growth_rate': 0.0,
            'interface_velocity': 0.0,
            'stress_concentration_factor': 0.0,
            'sei_thickness_evolution': [],
            'crack_evolution': self.crack_evolution
        }
        
        if len(self.crack_evolution) > 1:
            # 计算裂纹增长速率
            crack_areas = [info['crack_area'] for info in self.crack_evolution]
            time_points = self.time_history
            
            if len(crack_areas) > 1:
                growth_rates = np.diff(crack_areas) / np.diff(time_points)
                analysis['crack_growth_rate'] = np.mean(growth_rates)
            
            # 计算应力集中因子
            max_stresses = [info['max_stress'] for info in self.crack_evolution]
            mean_stresses = [info['mean_stress'] for info in self.crack_evolution]
            
            if mean_stresses[-1] > 0:
                analysis['stress_concentration_factor'] = max_stresses[-1] / mean_stresses[-1]
        
        # 计算 SEI 厚度演化
        if not self.use_fipy and len(self.phi_history) > 0:
            for phi_snapshot in self.phi_history:
                # 计算 SEI 层厚度 (简化)
                sei_mask = phi_snapshot > 0.5
                if np.any(sei_mask):
                    # 计算等效厚度
                    sei_area = np.sum(sei_mask) * self.config.dx * self.config.dy
                    sei_perimeter = self._compute_perimeter(sei_mask)
                    thickness = sei_area / sei_perimeter if sei_perimeter > 0 else 0
                    analysis['sei_thickness_evolution'].append(thickness)
                else:
                    analysis['sei_thickness_evolution'].append(0)
        
        return analysis
    
    def _save_results(self, results: Dict[str, Any], analysis: Dict[str, Any]):
        """保存模拟结果"""
        # 保存数据
        data_file = self.work_dir / "phase_field_results.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump({'results': results, 'analysis': analysis}, f)
        
        # 保存图像
        self._plot_results(results, analysis)
        
        logger.info(f"结果已保存至: {self.work_dir}")
    
    def _plot_results(self, results: Dict[str, Any], analysis: Dict[str, Any]):
        """绘制结果图像"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 绘制相场演化
        if results['phi_snapshots']:
            for i, (t, phi) in enumerate(zip(results['time_points'], results['phi_snapshots'])):
                if i < 3:
                    im = axes[0, i].imshow(phi, cmap='viridis', origin='lower')
                    axes[0, i].set_title(f'相场 t={t:.1f}')
                    axes[0, i].set_xlabel('x (nm)')
                    axes[0, i].set_ylabel('y (nm)')
                    plt.colorbar(im, ax=axes[0, i])
        
        # 绘制应力场演化
        if results['stress_snapshots']:
            for i, (t, stress) in enumerate(zip(results['time_points'], results['stress_snapshots'])):
                if i < 3:
                    im = axes[1, i].imshow(stress, cmap='hot', origin='lower')
                    axes[1, i].set_title(f'应力场 t={t:.1f}')
                    axes[1, i].set_xlabel('x (nm)')
                    axes[1, i].set_ylabel('y (nm)')
                    plt.colorbar(im, ax=axes[1, i])
        
        plt.tight_layout()
        plt.savefig(self.work_dir / "phase_field_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制分析结果
        if analysis['crack_evolution']:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            times = self.time_history
            crack_areas = [info['crack_area'] for info in analysis['crack_evolution']]
            max_stresses = [info['max_stress'] for info in analysis['crack_evolution']]
            num_cracks = [info['num_cracks'] for info in analysis['crack_evolution']]
            
            axes[0, 0].plot(times, crack_areas, 'b-o')
            axes[0, 0].set_xlabel('时间')
            axes[0, 0].set_ylabel('裂纹面积')
            axes[0, 0].set_title('裂纹面积演化')
            
            axes[0, 1].plot(times, max_stresses, 'r-o')
            axes[0, 1].set_xlabel('时间')
            axes[0, 1].set_ylabel('最大应力')
            axes[0, 1].set_title('最大应力演化')
            
            axes[1, 0].plot(times, num_cracks, 'g-o')
            axes[1, 0].set_xlabel('时间')
            axes[1, 0].set_ylabel('裂纹数量')
            axes[1, 0].set_title('裂纹数量演化')
            
            if analysis['sei_thickness_evolution']:
                axes[1, 1].plot(times, analysis['sei_thickness_evolution'], 'm-o')
                axes[1, 1].set_xlabel('时间')
                axes[1, 1].set_ylabel('SEI 厚度')
                axes[1, 1].set_title('SEI 厚度演化')
            
            plt.tight_layout()
            plt.savefig(self.work_dir / "crack_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """主函数 - 演示相场模拟"""
    print("=== 相场模拟耦合模块测试 ===")
    
    # 创建配置
    config = PhaseFieldConfig(
        nx=64,
        ny=64,
        dx=1.0,
        dy=1.0,
        dt=0.01,
        total_time=10.0,  # 短时间演示
        interface_width=2.0,
        interface_energy=0.1,
        mobility=1.0,
        elastic_modulus=10e9
    )
    
    # 创建模拟器
    simulator = PhaseFieldSimulator(config, work_dir="test_phase_field")
    
    # 运行模拟
    print("开始相场模拟...")
    results = simulator.run_simulation()
    
    # 输出结果
    print("\n=== 相场模拟结果 ===")
    print(f"模拟时间: {results['runtime']:.2f} 秒")
    
    analysis = results['analysis']
    print(f"裂纹增长速率: {analysis['crack_growth_rate']:.6f}")
    print(f"应力集中因子: {analysis['stress_concentration_factor']:.2f}")
    
    if analysis['sei_thickness_evolution']:
        initial_thickness = analysis['sei_thickness_evolution'][0]
        final_thickness = analysis['sei_thickness_evolution'][-1]
        print(f"SEI 厚度变化: {initial_thickness:.2f} → {final_thickness:.2f} nm")
    
    print("\n✓ 相场模拟耦合模块测试完成")

if __name__ == "__main__":
    main() 