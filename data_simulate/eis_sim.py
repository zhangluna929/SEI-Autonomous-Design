#!/usr/bin/env python3
"""
EIS 电化学阻抗谱模拟数据生成器

生成 500 个合成的 EIS 复阻抗谱，使用等效电路模型：
- Randles 电路：Rs + (Rct || CPE) + Warburg
- 频率范围：0.01 Hz - 100 kHz
- 模拟电池界面阻抗行为

输出：data/eis/eis_sim.arrow (PyArrow 格式)
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

def cpe_impedance(omega: np.ndarray, Q: float, n: float) -> np.ndarray:
    """常相位元件(CPE)阻抗: Z = 1/(Q*(jω)^n)"""
    return 1 / (Q * (1j * omega) ** n)

def warburg_impedance(omega: np.ndarray, sigma: float) -> np.ndarray:
    """Warburg 扩散阻抗: Z = σ/√(jω)"""
    return sigma / np.sqrt(1j * omega)

def randles_circuit(omega: np.ndarray, Rs: float, Rct: float, 
                   Q: float, n: float, sigma: float) -> np.ndarray:
    """
    Randles 等效电路模型
    Rs: 溶液电阻 (Ω)
    Rct: 电荷转移电阻 (Ω)  
    Q: CPE 参数 (S·s^n)
    n: CPE 指数 (0-1)
    sigma: Warburg 系数 (Ω·s^-0.5)
    """
    Z_cpe = cpe_impedance(omega, Q, n)
    Z_warburg = warburg_impedance(omega, sigma)
    
    # 并联：1/Z_parallel = 1/Rct + 1/Z_cpe
    Z_parallel = 1 / (1/Rct + 1/Z_cpe)
    
    # 串联：Z_total = Rs + Z_parallel + Z_warburg
    Z_total = Rs + Z_parallel + Z_warburg
    
    return Z_total

def generate_eis_spectrum(sample_id: str) -> Dict[str, Any]:
    """生成单个 EIS 谱"""
    # 随机参数范围（基于典型锂电池界面）
    Rs = np.random.uniform(1, 20)      # 溶液电阻 1-20 Ω
    Rct = np.random.uniform(10, 500)   # 电荷转移电阻 10-500 Ω
    Q = np.random.uniform(1e-6, 1e-3)  # CPE 参数
    n = np.random.uniform(0.7, 0.95)   # CPE 指数
    sigma = np.random.uniform(1, 50)   # Warburg 系数
    
    # 频率范围：0.01 Hz - 100 kHz，对数分布
    frequencies = np.logspace(-2, 5, 50)  # 50 个频率点
    omega = 2 * np.pi * frequencies
    
    # 计算复阻抗
    Z_complex = randles_circuit(omega, Rs, Rct, Q, n, sigma)
    
    # 添加噪声（1-3%）
    noise_level = np.random.uniform(0.01, 0.03)
    noise_real = np.random.normal(0, noise_level * np.abs(Z_complex.real))
    noise_imag = np.random.normal(0, noise_level * np.abs(Z_complex.imag))
    
    Z_complex += noise_real + 1j * noise_imag
    
    # 组织数据：[freq, Z_real, Z_imag] 交错存储
    spectrum_data = []
    for i, freq in enumerate(frequencies):
        spectrum_data.extend([
            freq,                    # 频率 (Hz)
            Z_complex[i].real,      # 实部 (Ω)
            -Z_complex[i].imag      # 虚部 (Ω, 取负号符合惯例)
        ])
    
    return {
        'sample_id': sample_id,
        'eis_spectrum': spectrum_data,
        'circuit_params': {
            'Rs': Rs,
            'Rct': Rct, 
            'Q': Q,
            'n': n,
            'sigma': sigma
        }
    }

def generate_eis_dataset(n_samples: int = 500) -> pd.DataFrame:
    """生成 EIS 数据集"""
    print(f"生成 {n_samples} 个 EIS 电化学阻抗谱...")
    
    records = []
    for i in range(n_samples):
        sample_id = f"eis_sample_{i:04d}"
        record = generate_eis_spectrum(sample_id)
        records.append({
            'sample_id': record['sample_id'],
            'eis_spectrum': record['eis_spectrum']
        })
        
        if (i + 1) % 100 == 0:
            print(f"已生成 {i + 1}/{n_samples} 个样本")
    
    return pd.DataFrame(records)

def main():
    # 创建输出目录
    output_dir = Path("data/eis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成数据集
    df = generate_eis_dataset(500)
    
    # 保存为 Arrow 格式
    table = pa.Table.from_pandas(df, preserve_index=False)
    output_path = output_dir / "eis_sim.arrow"
    pq.write_table(table, output_path)
    
    print(f"✓ EIS 数据集已保存至 {output_path}")
    print(f"  - 样本数量: {len(df)}")
    print(f"  - 每个谱包含: 50 个频率点 (0.01 Hz - 100 kHz)")
    print(f"  - 数据格式: [freq, Z_real, Z_imag] 交错存储")
    print(f"  - 文件大小: {output_path.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main() 