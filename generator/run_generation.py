import argparse, os, csv, torch
from pathlib import Path
import numpy as np
from sampler import GuidedSampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_emb', required=False, help='np.npy file of target embedding', default=None)
    parser.add_argument('--th_deltaE', type=float, default=0.1)
    parser.add_argument('--th_sigma', type=float, default=1e-4)
    parser.add_argument('--th_tg', type=float, default=-20)
    parser.add_argument('--out', default='generated_round0.csv')
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cond_vec = torch.zeros((args.batch, 256), device=device)  # placeholder condition
    sampler = GuidedSampler('predictor.ckpt', device=device)

    smiles_total = []
    while len(smiles_total) < args.n_samples:
        smiles = sampler.sample(cond_vec, batch=args.batch)
        smiles_total.extend(smiles)
        print(f"Generated {len(smiles_total)}/{args.n_samples}")

    # 简单写 CSV
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['smiles'])
        for s in smiles_total[: args.n_samples]:
            writer.writerow([s])
    print('Saved', args.out)

if __name__ == '__main__':
    main() 