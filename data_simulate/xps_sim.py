import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import trange
from pathlib import Path
import hashlib, os, argparse

def lorentz_gauss(x, A, x0, gamma, eta):
    # pseudo-voigt
    sigma = gamma / 2.355  # approximate conv.
    gauss = A * (1 - eta) * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    lorentz = A * eta * (gamma ** 2 / ((x - x0) ** 2 + gamma ** 2))
    return gauss + lorentz

def pink_noise(n):
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1
    spectrum = (1 / freqs) * np.exp(1j * 2 * np.pi * np.random.rand(len(freqs)))
    noise = np.fft.irfft(spectrum)
    noise = noise[:n]
    return noise / noise.ptp()

def simulate_single(seed):
    rng = np.random.default_rng(seed)
    x = np.arange(0, 1400, 0.1)
    y = np.zeros_like(x)
    n_peaks = rng.integers(2, 6)
    for _ in range(n_peaks):
        x0 = rng.uniform(20, 1300)
        A = rng.uniform(0.5, 1.0)
        gamma = rng.uniform(0.5, 3.0)
        eta = rng.uniform(0.3, 0.7)
        y += lorentz_gauss(x, A, x0, gamma, eta)
        # satellite
        if rng.random() < 0.4:
            y += lorentz_gauss(x, A * rng.uniform(0.05, 0.2), x0 + rng.uniform(5, 15), gamma * 1.5, eta)
    # add noise
    y += 0.02 * rng.normal(size=x.size) + 0.05 * pink_noise(x.size)
    # normalize
    y -= y.min()
    y /= y.ptp()
    return y.astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--out', default='data/xps/xps_sim.arrow')
    args = parser.parse_args()

    xps = []
    ids = []
    Path('data/xps').mkdir(parents=True, exist_ok=True)
    for i in trange(args.n, desc='Sim XPS'):
        spectrum = simulate_single(seed=i)
        sample_id = hashlib.md5(spectrum.tobytes()).hexdigest()
        xps.append(spectrum)
        ids.append(sample_id)
    arr_xps = pa.array(xps, type=pa.list_(pa.float32()))
    arr_id = pa.array(ids)
    table = pa.Table.from_arrays([arr_id, arr_xps], names=['sample_id', 'xps_spectrum'])
    pq.write_table(table, args.out)
    print('Saved', args.out)

if __name__ == '__main__':
    main() 