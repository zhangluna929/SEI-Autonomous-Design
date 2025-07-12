#!/usr/bin/env python3
"""
MD-LAMMPS 快速松弛计算模块

实现基于 LAMMPS 的快速分子动力学松弛计算，作为 DFT 计算前的粗粒度预处理：
- 使用 ReaxFF 力场进行快速结构优化
- 自动生成 LAMMPS 输入文件
- 并行处理多个结构
- 输出松弛后的结构供 DFT 精确计算

依赖：
    pip install ase lammps-python pymatgen
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

try:
    from ase import Atoms
    from ase.io import read, write
    from ase.calculators.lammpslib import LAMMPSlib
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
except ImportError:
    print("Warning: ASE, LAMMPS, or pymatgen not installed. Some features may not work.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LAMMPSRelaxer:
    """LAMMPS 快速松弛计算器"""
    
    def __init__(self, 
                 force_field: str = "reaxff",
                 max_steps: int = 1000,
                 energy_tolerance: float = 1e-6,
                 force_tolerance: float = 1e-4,
                 n_cores: int = 1):
        """
        初始化 LAMMPS 松弛器
        
        Args:
            force_field: 力场类型 (reaxff, tersoff, etc.)
            max_steps: 最大优化步数
            energy_tolerance: 能量收敛标准 (eV)
            force_tolerance: 力收敛标准 (eV/Å)
            n_cores: 并行核心数
        """
        self.force_field = force_field
        self.max_steps = max_steps
        self.energy_tolerance = energy_tolerance
        self.force_tolerance = force_tolerance
        self.n_cores = n_cores
        
    def _generate_lammps_input(self, structure: Structure, work_dir: Path) -> str:
        """生成 LAMMPS 输入文件"""
        lammps_input = f"""
# LAMMPS 快速松弛计算
units real
atom_style charge
boundary p p p

# 读取结构
read_data structure.data

# 设置力场
pair_style reax/c NULL
pair_coeff * * ffield.reax C H O N Li F P S

# 设置电荷平衡
fix qeq all qeq/reax 1 0.0 10.0 1.0e-6 reax/c

# 能量最小化
minimize {self.energy_tolerance} {self.force_tolerance} {self.max_steps} {self.max_steps*10}

# 输出最终结构
write_data relaxed.data
write_dump all custom relaxed.xyz id type x y z

# 输出能量信息
variable pe equal pe
print "Final potential energy: ${{pe}} eV"
        """
        
        input_file = work_dir / "relax.in"
        with open(input_file, 'w') as f:
            f.write(lammps_input.strip())
        
        return str(input_file)
    
    def _structure_to_lammps_data(self, structure: Structure, work_dir: Path) -> str:
        """将 pymatgen Structure 转换为 LAMMPS data 文件"""
        # 转换为 ASE Atoms
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
        
        # 写入 LAMMPS data 格式
        data_file = work_dir / "structure.data"
        write(str(data_file), atoms, format='lammps-data')
        
        return str(data_file)
    
    def _setup_force_field(self, work_dir: Path) -> None:
        """设置力场文件"""
        if self.force_field == "reaxff":
            # 这里应该复制真实的 ReaxFF 参数文件
            # 为演示目的，创建一个占位文件
            ff_file = work_dir / "ffield.reax"
            with open(ff_file, 'w') as f:
                f.write("# ReaxFF parameter file placeholder\n")
                f.write("# In real implementation, use actual ReaxFF parameters\n")
    
    def relax_structure(self, structure: Structure, structure_id: str) -> Dict[str, Any]:
        """
        松弛单个结构
        
        Args:
            structure: 待松弛的结构
            structure_id: 结构标识符
            
        Returns:
            松弛结果字典
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            
            try:
                # 设置计算环境
                self._setup_force_field(work_dir)
                data_file = self._structure_to_lammps_data(structure, work_dir)
                input_file = self._generate_lammps_input(structure, work_dir)
                
                # 运行 LAMMPS
                cmd = f"lmp_serial -in {input_file}"
                result = subprocess.run(
                    cmd.split(),
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5分钟超时
                )
                
                if result.returncode != 0:
                    logger.error(f"LAMMPS failed for {structure_id}: {result.stderr}")
                    return {
                        'structure_id': structure_id,
                        'success': False,
                        'error': result.stderr,
                        'original_structure': structure,
                        'relaxed_structure': None,
                        'energy': None,
                        'steps': None
                    }
                
                # 读取松弛后的结构
                relaxed_file = work_dir / "relaxed.xyz"
                if relaxed_file.exists():
                    relaxed_atoms = read(str(relaxed_file))
                    adaptor = AseAtomsAdaptor()
                    relaxed_structure = adaptor.get_structure(relaxed_atoms)
                else:
                    relaxed_structure = None
                
                # 解析能量
                energy = None
                steps = None
                for line in result.stdout.split('\n'):
                    if "Final potential energy:" in line:
                        try:
                            energy = float(line.split(':')[1].strip().split()[0])
                        except (IndexError, ValueError):
                            pass
                    elif "minimization" in line.lower() and "steps" in line.lower():
                        try:
                            steps = int([x for x in line.split() if x.isdigit()][0])
                        except (IndexError, ValueError):
                            pass
                
                return {
                    'structure_id': structure_id,
                    'success': True,
                    'error': None,
                    'original_structure': structure,
                    'relaxed_structure': relaxed_structure,
                    'energy': energy,
                    'steps': steps,
                    'force_field': self.force_field
                }
                
            except subprocess.TimeoutExpired:
                logger.error(f"LAMMPS timeout for {structure_id}")
                return {
                    'structure_id': structure_id,
                    'success': False,
                    'error': "Timeout",
                    'original_structure': structure,
                    'relaxed_structure': None,
                    'energy': None,
                    'steps': None
                }
            except Exception as e:
                logger.error(f"Error in LAMMPS relaxation for {structure_id}: {str(e)}")
                return {
                    'structure_id': structure_id,
                    'success': False,
                    'error': str(e),
                    'original_structure': structure,
                    'relaxed_structure': None,
                    'energy': None,
                    'steps': None
                }
    
    def relax_batch(self, structures: List[Tuple[Structure, str]]) -> List[Dict[str, Any]]:
        """
        批量松弛结构
        
        Args:
            structures: [(structure, structure_id), ...] 列表
            
        Returns:
            松弛结果列表
        """
        logger.info(f"开始批量松弛 {len(structures)} 个结构...")
        
        results = []
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            # 提交任务
            future_to_id = {
                executor.submit(self.relax_structure, struct, struct_id): struct_id
                for struct, struct_id in structures
            }
            
            # 收集结果
            for future in as_completed(future_to_id):
                struct_id = future_to_id[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        logger.info(f"✓ {struct_id} 松弛完成 (能量: {result['energy']:.3f} eV)")
                    else:
                        logger.warning(f"✗ {struct_id} 松弛失败: {result['error']}")
                        
                except Exception as e:
                    logger.error(f"处理 {struct_id} 时出错: {str(e)}")
                    results.append({
                        'structure_id': struct_id,
                        'success': False,
                        'error': str(e),
                        'original_structure': None,
                        'relaxed_structure': None,
                        'energy': None,
                        'steps': None
                    })
        
        success_count = sum(1 for r in results if r['success'])
        logger.info(f"批量松弛完成: {success_count}/{len(structures)} 成功")
        
        return results

def create_test_structures() -> List[Tuple[Structure, str]]:
    """创建测试结构"""
    from pymatgen.core import Lattice
    
    # 创建简单的 Li2CO3 结构作为测试
    lattice = Lattice.cubic(5.0)
    species = ['Li', 'Li', 'C', 'O', 'O', 'O']
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.1, 0.1, 0.1],
        [0.4, 0.4, 0.1],
        [0.1, 0.4, 0.4]
    ]
    
    structures = []
    for i in range(3):
        # 添加小的随机扰动
        perturbed_coords = []
        for coord in coords:
            perturbed = [c + np.random.uniform(-0.05, 0.05) for c in coord]
            perturbed_coords.append(perturbed)
        
        structure = Structure(lattice, species, perturbed_coords)
        structures.append((structure, f"test_structure_{i}"))
    
    return structures

def main():
    """主函数 - 演示 MD 松弛功能"""
    print("=== MD-LAMMPS 快速松弛模块测试 ===")
    
    # 创建测试结构
    test_structures = create_test_structures()
    print(f"创建了 {len(test_structures)} 个测试结构")
    
    # 初始化松弛器
    relaxer = LAMMPSRelaxer(
        force_field="reaxff",
        max_steps=500,
        energy_tolerance=1e-6,
        force_tolerance=1e-4,
        n_cores=2
    )
    
    # 执行批量松弛
    results = relaxer.relax_batch(test_structures)
    
    # 输出结果统计
    print("\n=== 松弛结果统计 ===")
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"成功: {len(successful)}/{len(results)}")
    print(f"失败: {len(failed)}/{len(results)}")
    
    if successful:
        energies = [r['energy'] for r in successful if r['energy'] is not None]
        if energies:
            print(f"平均能量: {np.mean(energies):.3f} ± {np.std(energies):.3f} eV")
    
    if failed:
        print("\n失败原因:")
        for r in failed:
            print(f"  - {r['structure_id']}: {r['error']}")
    
    print("\n✓ MD-LAMMPS 快速松弛模块测试完成")

if __name__ == "__main__":
    main() 