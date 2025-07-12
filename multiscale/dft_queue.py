#!/usr/bin/env python3
"""
DFT 计算队列管理模块

管理 VASP DFT 计算任务队列，实现：
- 自动 VASP 输入文件生成
- 任务队列管理和调度
- 计算状态监控
- 结果收集和错误处理
- 与 LAMMPS 松弛结果的无缝对接

依赖：
    pip install pymatgen custodian
"""

import os
import json
import shutil
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

try:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar, Incar, Potcar, Kpoints
    from pymatgen.io.vasp.outputs import Vasprun, Outcar
    from custodian import Custodian
    from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler
    from custodian.vasp.jobs import VaspJob
except ImportError:
    print("Warning: pymatgen or custodian not installed. Some features may not work.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class DFTJob:
    """DFT 计算任务"""
    job_id: str
    structure: Structure
    job_type: str = "relax"  # relax, static, band, etc.
    priority: int = 1  # 1=高优先级, 5=低优先级
    status: JobStatus = JobStatus.PENDING
    work_dir: Optional[str] = None
    submit_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    energy: Optional[float] = None
    forces: Optional[List[List[float]]] = None
    stress: Optional[List[List[float]]] = None
    relaxed_structure: Optional[Structure] = None
    error_message: Optional[str] = None
    vasp_settings: Optional[Dict[str, Any]] = None

class DFTQueue:
    """DFT 计算队列管理器"""
    
    def __init__(self, 
                 queue_dir: str = "dft_queue",
                 max_concurrent: int = 4,
                 vasp_cmd: str = "vasp_std",
                 potcar_dir: Optional[str] = None):
        """
        初始化 DFT 队列管理器
        
        Args:
            queue_dir: 队列工作目录
            max_concurrent: 最大并发任务数
            vasp_cmd: VASP 命令
            potcar_dir: POTCAR 文件目录
        """
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(exist_ok=True)
        
        self.max_concurrent = max_concurrent
        self.vasp_cmd = vasp_cmd
        self.potcar_dir = potcar_dir
        
        # 初始化数据库
        self.db_path = self.queue_dir / "jobs.db"
        self._init_database()
        
        # 默认 VASP 设置
        self.default_vasp_settings = {
            "relax": {
                "INCAR": {
                    "PREC": "Accurate",
                    "ENCUT": 520,
                    "EDIFF": 1e-6,
                    "EDIFFG": -0.01,
                    "NSW": 100,
                    "IBRION": 2,
                    "ISIF": 3,
                    "ISMEAR": 0,
                    "SIGMA": 0.05,
                    "LREAL": "Auto",
                    "LWAVE": False,
                    "LCHARG": False
                },
                "KPOINTS": {"density": 1000}  # k-point density
            },
            "static": {
                "INCAR": {
                    "PREC": "Accurate",
                    "ENCUT": 520,
                    "EDIFF": 1e-8,
                    "NSW": 0,
                    "ISMEAR": -5,
                    "LREAL": "Auto",
                    "LWAVE": True,
                    "LCHARG": True
                },
                "KPOINTS": {"density": 2000}
            }
        }
    
    def _init_database(self):
        """初始化任务数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    structure_json TEXT,
                    job_type TEXT,
                    priority INTEGER,
                    status TEXT,
                    work_dir TEXT,
                    submit_time REAL,
                    start_time REAL,
                    end_time REAL,
                    energy REAL,
                    forces_json TEXT,
                    stress_json TEXT,
                    relaxed_structure_json TEXT,
                    error_message TEXT,
                    vasp_settings_json TEXT
                )
            """)
            conn.commit()
    
    def submit_job(self, job: DFTJob) -> str:
        """提交 DFT 计算任务"""
        job.submit_time = time.time()
        job.work_dir = str(self.queue_dir / job.job_id)
        
        # 保存到数据库
        self._save_job_to_db(job)
        
        logger.info(f"提交 DFT 任务: {job.job_id} ({job.job_type})")
        return job.job_id
    
    def submit_from_lammps_results(self, lammps_results: List[Dict[str, Any]]) -> List[str]:
        """从 LAMMPS 松弛结果提交 DFT 任务"""
        job_ids = []
        
        for result in lammps_results:
            if not result['success'] or result['relaxed_structure'] is None:
                continue
                
            job_id = f"dft_{result['structure_id']}_{int(time.time())}"
            
            # 创建 DFT 任务
            job = DFTJob(
                job_id=job_id,
                structure=result['relaxed_structure'],
                job_type="relax",
                priority=2,  # 中等优先级
                vasp_settings=self.default_vasp_settings["relax"]
            )
            
            job_ids.append(self.submit_job(job))
        
        logger.info(f"从 LAMMPS 结果提交了 {len(job_ids)} 个 DFT 任务")
        return job_ids
    
    def _save_job_to_db(self, job: DFTJob):
        """保存任务到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO jobs VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                job.job_id,
                job.structure.to_json(),
                job.job_type,
                job.priority,
                job.status.value,
                job.work_dir,
                job.submit_time,
                job.start_time,
                job.end_time,
                job.energy,
                json.dumps(job.forces) if job.forces else None,
                json.dumps(job.stress) if job.stress else None,
                job.relaxed_structure.to_json() if job.relaxed_structure else None,
                job.error_message,
                json.dumps(job.vasp_settings) if job.vasp_settings else None
            ))
            conn.commit()
    
    def _load_job_from_db(self, job_id: str) -> Optional[DFTJob]:
        """从数据库加载任务"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # 解析数据
            structure = Structure.from_dict(json.loads(row[1]))
            relaxed_structure = Structure.from_dict(json.loads(row[12])) if row[12] else None
            forces = json.loads(row[10]) if row[10] else None
            stress = json.loads(row[11]) if row[11] else None
            vasp_settings = json.loads(row[14]) if row[14] else None
            
            return DFTJob(
                job_id=row[0],
                structure=structure,
                job_type=row[2],
                priority=row[3],
                status=JobStatus(row[4]),
                work_dir=row[5],
                submit_time=row[6],
                start_time=row[7],
                end_time=row[8],
                energy=row[9],
                forces=forces,
                stress=stress,
                relaxed_structure=relaxed_structure,
                error_message=row[13],
                vasp_settings=vasp_settings
            )
    
    def get_pending_jobs(self) -> List[DFTJob]:
        """获取待处理任务列表"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT job_id FROM jobs 
                WHERE status = ? 
                ORDER BY priority ASC, submit_time ASC
            """, (JobStatus.PENDING.value,))
            
            job_ids = [row[0] for row in cursor.fetchall()]
        
        return [self._load_job_from_db(job_id) for job_id in job_ids]
    
    def get_running_jobs(self) -> List[DFTJob]:
        """获取运行中任务列表"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT job_id FROM jobs WHERE status = ?
            """, (JobStatus.RUNNING.value,))
            
            job_ids = [row[0] for row in cursor.fetchall()]
        
        return [self._load_job_from_db(job_id) for job_id in job_ids]
    
    def _setup_vasp_input(self, job: DFTJob) -> bool:
        """设置 VASP 输入文件"""
        try:
            work_dir = Path(job.work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            
            # 写入 POSCAR
            poscar = Poscar(job.structure)
            poscar.write_file(work_dir / "POSCAR")
            
            # 写入 INCAR
            incar_dict = job.vasp_settings.get("INCAR", {})
            incar = Incar(incar_dict)
            incar.write_file(work_dir / "INCAR")
            
            # 写入 KPOINTS
            kpoints_settings = job.vasp_settings.get("KPOINTS", {"density": 1000})
            if "density" in kpoints_settings:
                kpoints = Kpoints.automatic_density(job.structure, kpoints_settings["density"])
            else:
                kpoints = Kpoints.from_dict(kpoints_settings)
            kpoints.write_file(work_dir / "KPOINTS")
            
            # 写入 POTCAR (如果有 POTCAR 目录)
            if self.potcar_dir:
                potcar = Potcar(job.structure.species, functional="PBE")
                potcar.write_file(work_dir / "POTCAR")
            
            return True
            
        except Exception as e:
            logger.error(f"设置 VASP 输入文件失败 {job.job_id}: {str(e)}")
            return False
    
    def _run_vasp_job(self, job: DFTJob) -> bool:
        """运行 VASP 计算"""
        try:
            work_dir = Path(job.work_dir)
            
            # 更新任务状态
            job.status = JobStatus.RUNNING
            job.start_time = time.time()
            self._save_job_to_db(job)
            
            # 运行 VASP
            if self.potcar_dir:
                # 使用 custodian 运行 (更稳定)
                handlers = [VaspErrorHandler(), UnconvergedErrorHandler()]
                jobs = [VaspJob(vasp_cmd=self.vasp_cmd)]
                
                c = Custodian(handlers, jobs, max_errors=5)
                c.run()
            else:
                # 直接运行 VASP (演示模式)
                cmd = [self.vasp_cmd]
                result = subprocess.run(
                    cmd,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1小时超时
                )
                
                if result.returncode != 0:
                    raise Exception(f"VASP failed: {result.stderr}")
            
            # 解析结果
            self._parse_vasp_output(job)
            
            job.status = JobStatus.COMPLETED
            job.end_time = time.time()
            self._save_job_to_db(job)
            
            logger.info(f"✓ DFT 任务完成: {job.job_id} (能量: {job.energy:.3f} eV)")
            return True
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.end_time = time.time()
            self._save_job_to_db(job)
            
            logger.error(f"✗ DFT 任务失败: {job.job_id} - {str(e)}")
            return False
    
    def _parse_vasp_output(self, job: DFTJob):
        """解析 VASP 输出"""
        work_dir = Path(job.work_dir)
        
        # 读取 vasprun.xml
        vasprun_file = work_dir / "vasprun.xml"
        if vasprun_file.exists():
            try:
                vasprun = Vasprun(str(vasprun_file))
                job.energy = vasprun.final_energy
                job.forces = vasprun.ionic_steps[-1]['forces'].tolist()
                job.relaxed_structure = vasprun.final_structure
                
                # 读取应力
                if hasattr(vasprun, 'ionic_steps') and vasprun.ionic_steps:
                    last_step = vasprun.ionic_steps[-1]
                    if 'stress' in last_step:
                        job.stress = last_step['stress'].tolist()
                        
            except Exception as e:
                logger.warning(f"解析 vasprun.xml 失败 {job.job_id}: {str(e)}")
        
        # 读取 OUTCAR (备用)
        outcar_file = work_dir / "OUTCAR"
        if outcar_file.exists() and job.energy is None:
            try:
                outcar = Outcar(str(outcar_file))
                if outcar.final_energy:
                    job.energy = outcar.final_energy
            except Exception as e:
                logger.warning(f"解析 OUTCAR 失败 {job.job_id}: {str(e)}")
    
    def run_queue(self, max_iterations: int = 100):
        """运行队列处理器"""
        logger.info(f"启动 DFT 队列处理器 (最大并发: {self.max_concurrent})")
        
        for iteration in range(max_iterations):
            # 获取当前状态
            running_jobs = self.get_running_jobs()
            pending_jobs = self.get_pending_jobs()
            
            if not pending_jobs and not running_jobs:
                logger.info("队列为空，退出")
                break
            
            # 启动新任务
            available_slots = self.max_concurrent - len(running_jobs)
            if available_slots > 0 and pending_jobs:
                jobs_to_start = pending_jobs[:available_slots]
                
                with ThreadPoolExecutor(max_workers=available_slots) as executor:
                    futures = []
                    for job in jobs_to_start:
                        if self._setup_vasp_input(job):
                            future = executor.submit(self._run_vasp_job, job)
                            futures.append(future)
                    
                    # 等待任务完成
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"任务执行异常: {str(e)}")
            
            # 等待一段时间再检查
            time.sleep(30)
        
        logger.info("DFT 队列处理器结束")
    
    def get_completed_results(self) -> List[Dict[str, Any]]:
        """获取完成的计算结果"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT job_id FROM jobs WHERE status = ?
            """, (JobStatus.COMPLETED.value,))
            
            job_ids = [row[0] for row in cursor.fetchall()]
        
        results = []
        for job_id in job_ids:
            job = self._load_job_from_db(job_id)
            if job:
                results.append({
                    'job_id': job.job_id,
                    'structure': job.structure,
                    'relaxed_structure': job.relaxed_structure,
                    'energy': job.energy,
                    'forces': job.forces,
                    'stress': job.stress,
                    'job_type': job.job_type,
                    'runtime': job.end_time - job.start_time if job.end_time and job.start_time else None
                })
        
        return results
    
    def get_queue_status(self) -> Dict[str, int]:
        """获取队列状态统计"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT status, COUNT(*) FROM jobs GROUP BY status
            """)
            
            status_counts = dict(cursor.fetchall())
        
        return {
            'pending': status_counts.get(JobStatus.PENDING.value, 0),
            'running': status_counts.get(JobStatus.RUNNING.value, 0),
            'completed': status_counts.get(JobStatus.COMPLETED.value, 0),
            'failed': status_counts.get(JobStatus.FAILED.value, 0),
            'cancelled': status_counts.get(JobStatus.CANCELLED.value, 0)
        }

def main():
    """主函数 - 演示 DFT 队列功能"""
    print("=== DFT 计算队列管理模块测试 ===")
    
    # 初始化队列管理器
    queue = DFTQueue(
        queue_dir="test_dft_queue",
        max_concurrent=2,
        vasp_cmd="echo 'VASP placeholder'",  # 演示用
        potcar_dir=None
    )
    
    # 创建测试任务
    from pymatgen.core import Structure, Lattice
    
    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ['Li', 'Li'], [[0, 0, 0], [0.5, 0.5, 0.5]])
    
    jobs = []
    for i in range(3):
        job = DFTJob(
            job_id=f"test_job_{i}",
            structure=structure,
            job_type="relax",
            priority=i + 1,
            vasp_settings=queue.default_vasp_settings["relax"]
        )
        jobs.append(job)
    
    # 提交任务
    for job in jobs:
        queue.submit_job(job)
    
    # 查看队列状态
    status = queue.get_queue_status()
    print(f"队列状态: {status}")
    
    # 模拟处理 (不实际运行 VASP)
    print("模拟队列处理...")
    pending_jobs = queue.get_pending_jobs()
    print(f"待处理任务: {len(pending_jobs)}")
    
    for job in pending_jobs:
        print(f"  - {job.job_id} (优先级: {job.priority})")
    
    print("\n✓ DFT 计算队列管理模块测试完成")

if __name__ == "__main__":
    main() 