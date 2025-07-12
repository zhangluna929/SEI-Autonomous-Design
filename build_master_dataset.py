# build_master_dataset.py

"""数据同构化脚本

该脚本读取以下三类原始数据并输出统一的 Arrow/Parquet 文件（master.parquet）：
1. 晶体数据库（obelix_all.csv）。使用 pymatgen 将 CIF 转换为局部环境图。
2. 聚合物数据库（若干 CSV）。使用 SELFIES 将 SMILES 转序列，并用 RDKit 生成 3D 构象。
3. 光谱数据库（FTIR/Raman/XRD，Excel）。对每条谱图按固定步长重采样并做 Min–Max 归一。

为了演示流程，脚本使用占位实现（missing 信息将写入 None）。如需完整图/坐标，请补充 CIF 文件及 SMILES 列名。

依赖(pip install)：
    pandas numpy pyarrow tqdm selfies rdkit-pypi pymatgen scipy

使用：
    python build_master_dataset.py --threads 24 --out master.parquet
"""

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import os

# ----- 可选科学库 -----
try:
    import selfies as sf
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from pymatgen.core import Structure
    from pymatgen.analysis.local_env import CrystalNN
except ImportError:
    sf = None  # type: ignore
    Chem = None  # type: ignore

MP_API_KEY = os.getenv("MP_API_KEY")
try:
    from pymatgen.ext.matproj import MPRester
    _mpr = MPRester(MP_API_KEY) if MP_API_KEY else None
except Exception:
    _mpr = None

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def resample_spectrum(x: np.ndarray, y: np.ndarray, step: float) -> np.ndarray:
    """重采样并返回定长向量（按最小 x 到最大 x）。"""
    from scipy.interpolate import interp1d

    xmin, xmax = x.min(), x.max()
    grid = np.arange(xmin, xmax + step, step)
    # 线性插值，外延使用 0
    interp = interp1d(x, y, kind="linear", bounds_error=False, fill_value=0.0)
    y_new = interp(grid)
    # 归一化
    if y_new.max() != y_new.min():
        y_new = (y_new - y_new.min()) / (y_new.max() - y_new.min())
    else:
        y_new = np.zeros_like(y_new)
    return y_new.astype(np.float32)

# -----------------------------------------------------------------------------
# Crystal processing
# -----------------------------------------------------------------------------

def crystal_row_to_graph(row: pd.Series) -> Optional[List[Dict[str, Any]]]:
    """将单行晶体信息转换为局部环境图（占位实现）。

    真实实现需要 CIF 内容或结构文件。此处若缺失，则返回 None。
    """
    cif_string = row.get("cif", None)
    if cif_string is None and _mpr is not None:
        formula = row.get("True Composition") or row.get("Reduced Composition")
        if formula:
            try:
                structs = _mpr.get_structures(formula)
                if structs:
                    structure = structs[0]
                    cif_string = structure.to(fmt="cif")
            except Exception as e:
                print(f"Warning: Failed to get structure for {formula}: {e}")
                cif_string = None
    
    if cif_string is None:
        return None
    
    try:
        from pymatgen.core import Structure
        from pymatgen.analysis.local_env import CrystalNN
        
        structure = Structure.from_str(cif_string, fmt="cif")
        cnn = CrystalNN()
        graph_data = []
        
        for idx, site in enumerate(structure):
            try:
                neigh = cnn.get_nn_info(structure, idx)
                for n in neigh:
                    graph_data.append({
                        "site": idx,
                        "neighbor": n["site_index"],
                        "weight": float(n["weight"]),
                    })
            except Exception as e:
                # 如果某个位点失败，跳过但继续处理其他位点
                continue
                
        return graph_data if graph_data else None
        
    except Exception as e:
        print(f"Warning: Failed to process crystal structure: {e}")
        return None

# -----------------------------------------------------------------------------
# Polymer processing
# -----------------------------------------------------------------------------

def smiles_to_selfies_and_coords(smiles: str) -> Dict[str, Any]:
    """SMILES -> SELFIES & 3D 坐标 (列表[float])。"""
    selfies_str = None
    coords = None
    
    if not smiles or smiles.strip() == "":
        return {"selfies": None, "coords": None}
    
    if sf is not None:
        try:
            selfies_str = sf.encoder(smiles)
        except Exception as e:
            print(f"Warning: SELFIES encoding failed for {smiles}: {e}")
            selfies_str = None
    
    if Chem is not None:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.AddHs(mol)
                # 尝试3D嵌入
                if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == 0:
                    # 优化几何结构
                    AllChem.UFFOptimizeMolecule(mol)
                    conf = mol.GetConformer()
                    coords = np.array(conf.GetPositions(), dtype=np.float32).tolist()
                else:
                    # 如果3D嵌入失败，尝试简单的2D坐标
                    coords = None
        except Exception as e:
            print(f"Warning: 3D coordinate generation failed for {smiles}: {e}")
            coords = None
    
    return {"selfies": selfies_str, "coords": coords}

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def build_dataset(
    crystal_csv: Path,
    polymer_csvs: List[Path],
    spectra_excels: List[Path],
    threads: int = 8,
    **kwargs
) -> pa.Table:
    # --- 1. 读取晶体数据 ---
    crystal_df = pd.read_csv(crystal_csv)

    # 可并行处理局部环境图
    graphs: List[Optional[Dict[str, Any]]] = [None] * len(crystal_df)
    with ProcessPoolExecutor(max_workers=threads) as ex:
        for idx, res in tqdm(
            enumerate(ex.map(crystal_row_to_graph, [r for _, r in crystal_df.iterrows()])),
            total=len(crystal_df),
            desc="Crystal graphs",
        ):
            graphs[idx] = res
    crystal_df["crystal_graph"] = graphs

    crystal_df.rename(columns={"ID": "sample_id"}, inplace=True)

    # --- 2. 读取聚合物数据 ---
    polymer_frames = []
    for csv_path in polymer_csvs:
        df = pd.read_csv(csv_path)
        polymer_frames.append(df)
    polymer_df = pd.concat(polymer_frames, ignore_index=True)

    # 处理 SMILES -> SELFIES + coords
    smiles_col = None
    for col in polymer_df.columns:
        if any(token in col.lower() for token in ["sml", "smiles", "smile"]):
            smiles_col = col
            break
    if smiles_col is None:
        smiles_list = [""] * len(polymer_df)
    else:
        smiles_list = polymer_df[smiles_col].fillna("").tolist()

    selfies_list: List[Optional[str]] = [None] * len(polymer_df)
    coords_list: List[Optional[List[Any]]] = [None] * len(polymer_df)
    with ProcessPoolExecutor(max_workers=threads) as ex:
        for idx, res in tqdm(
            enumerate(ex.map(smiles_to_selfies_and_coords, smiles_list)),
            total=len(smiles_list),
            desc="Polymer SELFIES/coords",
        ):
            selfies_list[idx] = res["selfies"]
            coords_list[idx] = res["coords"]
    polymer_df["selfies"] = selfies_list
    polymer_df["coords"] = coords_list

    polymer_df.rename(columns={"id": "sample_id", "ID": "sample_id"}, inplace=True)

    # --- 3. 处理谱图 ---
    spectra_records: List[Dict[str, Any]] = []
    for excel_path in spectra_excels:
        book = pd.read_excel(excel_path, sheet_name=None)
        is_xrd = "xrd" in str(excel_path).lower()
        step_val = 0.02 if is_xrd else 1.0
        for sheet, df in book.items():
            # 假设第一列是 x（波数/角度），后续列是样本
            x = df.iloc[:, 0].values.astype(float)
            for col in df.columns[1:]:
                y = df[col].values.astype(float)
                vec = resample_spectrum(x, y, step=step_val)
                spectra_records.append({
                    "sample_id": str(col),
                    "spectrum": vec.tolist(),
                })
    spectra_df = pd.DataFrame(spectra_records)

    # --- 3b. 读取模拟谱 Arrow (XPS/NMR/AFM/EIS) ---
    def arrow_to_df(path, col_name):
        if not Path(path).exists():
            return pd.DataFrame(columns=['sample_id', col_name])
        table = pq.read_table(path)
        return table.to_pandas()

    xps_df  = arrow_to_df(kwargs.get('xps_arrow', 'data/xps/xps_sim.arrow'), 'xps_spectrum')
    nmr_df  = arrow_to_df(kwargs.get('nmr_arrow', 'data/nmr/nmr_sim.arrow'), 'nmr_spectrum')
    afm_df  = arrow_to_df(kwargs.get('afm_arrow', 'data/afm/afm_sim.arrow'), 'afm_curve')
    eis_df  = arrow_to_df(kwargs.get('eis_arrow', 'data/eis/eis_sim.arrow'), 'eis_spectrum')

    # --- 4. 联合 & 对齐 ---
    master = (
        crystal_df[["sample_id", "crystal_graph"]]
        .merge(polymer_df[["sample_id", "selfies", "coords"]], on="sample_id", how="outer")
        .merge(spectra_df, on="sample_id", how="outer")
        .merge(xps_df, on="sample_id", how="outer")
        .merge(nmr_df, on="sample_id", how="outer")
        .merge(afm_df, on="sample_id", how="outer")
        .merge(eis_df, on="sample_id", how="outer")
    )

    # --- 5. 转 Arrow/Parquet ---
    table = pa.Table.from_pandas(master, preserve_index=False)
    return table

# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="多模态数据同构化")
    parser.add_argument("--crystal", default="data/obelix/obelix_all.csv")
    parser.add_argument(
        "--polymer_csvs",
        nargs="*",
        default=[
            "data/polymer/dataset_combine_ILthermo.csv",
            "data/polymer/dataset_comb.csv",
            "data/polymer/dataset_iolitech_final.csv",
        ],
    )
    parser.add_argument(
        "--spectra_excels",
        nargs="*",
        default=[
            "data/sei_spectra/ATR-FTIR.xlsx",
            "data/sei_spectra/Raman.xlsx",
            "data/sei_spectra/XRD.xlsx",
        ],
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default="master.parquet")
    parser.add_argument("--mp-api-key", default=os.getenv("MP_API_KEY"), help="Materials Project API key")
    parser.add_argument("--xps_arrow", default="data/xps/xps_sim.arrow")
    parser.add_argument("--nmr_arrow", default="data/nmr/nmr_sim.arrow")
    parser.add_argument("--afm_arrow", default="data/afm/afm_sim.arrow")
    parser.add_argument("--eis_arrow", default="data/eis/eis_sim.arrow")
    args = parser.parse_args()

    if args.mp_api_key:
        global _mpr
        from pymatgen.ext.matproj import MPRester
        _mpr = MPRester(args.mp_api_key)

    table = build_dataset(
        Path(args.crystal),
        [Path(p) for p in args.polymer_csvs],
        [Path(p) for p in args.spectra_excels],
        threads=args.threads,
        xps_arrow=args.xps_arrow,
        nmr_arrow=args.nmr_arrow,
        afm_arrow=args.afm_arrow,
        eis_arrow=args.eis_arrow,
    )

    pq.write_table(table, args.out)
    print(f"Wrote {args.out} with {table.num_rows} rows.")


if __name__ == "__main__":
    main() 