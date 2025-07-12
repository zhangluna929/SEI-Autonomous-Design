import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import trange
from pathlib import Path
import hashlib, argparse

def lorentz(x, A, x0, gamma):
    return A * (gamma ** 2 / ((x - x0) ** 2 + gamma ** 2))

def simulate_single(seed):
    rng = np.random.default_rng(seed)
    x = np.arange(-50, 400, 0.1)  # ppm grid (solid-state wide range)
    y = np.zeros_like(x)
    n_peaks = rng.integers(1, 4)
    for _ in range(n_peaks):
        x0 = rng.uniform(0, 300)
        A = rng.uniform(0.5, 1.0)
        gamma = rng.uniform(0.5, 2.0)
        y += lorentz(x, A, x0, gamma)
    # spinning sidebands every ±ν_r (assume 10 kHz -> 64 ppm)
    if rng.random() < 0.6:
        v_r = 64
        for shift in [v_r, -v_r]:
            y += 0.2 * lorentz(x, A * 0.3, x0 + shift, gamma)
    # add noise
    y += 0.02 * rng.normal(size=x.size)
    y -= y.min(); y /= y.ptp()
    return y.astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--out', default='data/nmr/nmr_sim.arrow')
    args = parser.parse_args()

    Path('data/nmr').mkdir(parents=True, exist_ok=True)
    spectra = []
    ids = []
    for i in trange(args.n, desc='Sim NMR'):
        s = simulate_single(i)
        spectra.append(s)
        ids.append(hashlib.md5(s.tobytes()).hexdigest())
    table = pa.table({'sample_id': ids, 'nmr_spectrum': pa.array(spectra, type=pa.list_(pa.float32()))})
    pq.write_table(table, args.out)
    print('Saved', args.out)

if __name__ == '__main__':
    main() 