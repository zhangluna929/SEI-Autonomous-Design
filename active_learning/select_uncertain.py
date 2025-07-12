"""Select Top-K most uncertain samples from generated SMILES using predictor.
生成 AL_round{n}_selected.csv
Usage:
    python active_learning/select_uncertain.py --generated generated_round0.csv \
        --predictor predictor.ckpt --topk 128 --out AL_round0_selected.csv
"""
import argparse, torch, pandas as pd
from ..finetune.predictor import Predictor
from ..pretrain.datamodule import selfies_tokenize, PAD_ID

def batch_iter(seq, batch):
    for i in range(0, len(seq), batch):
        yield seq[i:i+batch]

def uncertainty(smiles_list, model, device):
    ids = [torch.tensor(selfies_tokenize(s), device=device) for s in smiles_list]
    max_len = max(len(t) for t in ids)
    ids = [torch.nn.functional.pad(t, (0,max_len-len(t)), value=PAD_ID) for t in ids]
    seq_ids = torch.stack(ids)
    attn = seq_ids == PAD_ID
    dummy_spec = torch.zeros(seq_ids.size(0), 4000, device=device)
    batch_data = dict(seq_ids=seq_ids, attn_mask=attn, spectra=dummy_spec,
                      x_c=torch.zeros(1,1), edge_index=torch.zeros(2,1).long(),
                      edge_attr=torch.zeros(1,1), batch=torch.zeros(1).long())
    with torch.no_grad():
        _, _, _, log_vars = model(batch_data)  # Predictor.forward 返回 fused,(tuple), node_logits,patch_tokens but we only need log_vars stored inside model.log_vars (global)
    # 此处简化：取平均对数方差
    return log_vars.exp().mean(dim=1).cpu()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', default='generated_round0.csv')
    parser.add_argument('--predictor', default='predictor.ckpt')
    parser.add_argument('--topk', type=int, default=128)
    parser.add_argument('--out', default='AL_round0_selected.csv')
    parser.add_argument('--batch', type=int, default=256)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Predictor(vocab_size=30010).to(device)
    model.load_state_dict(torch.load(args.predictor, map_location=device))
    model.eval()
    df = pd.read_csv(args.generated)
    unc = []
    for chunk in batch_iter(df.smiles.tolist(), args.batch):
        unc.append(uncertainty(chunk, model, device))
    df['unc'] = torch.cat(unc).numpy()
    df.sort_values('unc', ascending=False).head(args.topk).to_csv(args.out, index=False)
    print('Selected', args.topk, 'most uncertain samples ->', args.out)

if __name__ == '__main__':
    main() 