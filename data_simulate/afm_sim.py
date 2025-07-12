import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import trange
from pathlib import Path
import hashlib, argparse

# JKR model parameters ranges
R_RANGE = (10e-9, 50e-9)       # tip radius 10-50 nm
E_RANGE = (1e8, 5e9)           # composite modulus 0.1-5 GPa
GAMMA_RANGE = (0.05, 0.3)      # surface energy J/m2

def jkr_force(delta, R, E_star, gamma):
    # delta: indentation (m)
    a = ( (3*gamma*np.pi*R + 6*E_star*R*delta + np.sqrt( (3*np.pi*gamma*R)**2 + 12*np.pi*gamma*E_star*R*delta ) ) / ( 3*E_star ) )**(1/3)
    return (4./3)*E_star*np.sqrt(R)*delta**1.5 - np.sqrt(6*np.pi*gamma*E_star*R)*delta**0.75

def simulate_single(seed):
    rng = np.random.default_rng(seed)
    z = np.linspace(-5e-9, 50e-9, 550)  # approach from -5 nm to 50 nm indentation
    R = rng.uniform(*R_RANGE)
    E_star = rng.uniform(*E_RANGE)
    gamma = rng.uniform(*GAMMA_RANGE)
    delta = np.clip(z, 0, None)
    F = jkr_force(delta, R, E_star, gamma)
    noise = rng.normal(scale=0.05*F.max(), size=F.size)
    F = F + noise
    # normalize 0-1
    F -= F.min(); F /= F.ptp()
    return F.astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--out', default='data/afm/afm_sim.arrow')
    args = parser.parse_args()
    Path('data/afm').mkdir(parents=True, exist_ok=True)
    curves, ids = [], []
    for i in trange(args.n, desc='Sim AFM'):
        c = simulate_single(i)
        curves.append(c)
        ids.append(hashlib.md5(c.tobytes()).hexdigest())
    table = pa.table({'sample_id': ids, 'afm_curve': pa.array(curves, type=pa.list_(pa.float32()))})
    pq.write_table(table, args.out)
    print('Saved', args.out)

if __name__=='__main__':
    main() 