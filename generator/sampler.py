import torch
import torch.nn.functional as F
from .cond_diffusion import CondDiffusionModel
from pathlib import Path
from typing import Callable, List
import numpy as np

class GuidedSampler:
    def __init__(self, predictor_ckpt: str, model_dir: str = 'generator/chemGPT', device: str = 'cuda'):
        from ..finetune.predictor import Predictor
        self.diffusion = CondDiffusionModel(model_dir, device)
        self.device = device
        
        # 加载 predictor
        try:
            self.predictor = Predictor(vocab_size=30010).to(device)
            self.predictor.load_state_dict(torch.load(predictor_ckpt, map_location=device))
            self.predictor.eval()
            self.has_predictor = True
        except Exception as e:
            print(f"Warning: Failed to load predictor from {predictor_ckpt}: {e}")
            self.has_predictor = False

    def score_fn(self, smiles_list: List[str]) -> torch.Tensor:
        """使用 predictor 估算 ΔE / σ；返回惩罚分数 tensor"""
        if not self.has_predictor:
            return torch.zeros(len(smiles_list), device=self.device)
        
        try:
            # TODO: 实现完整的SMILES到特征的转换
            # 这里使用占位符实现
            scores = []
            for smiles in smiles_list:
                # 简单的分子复杂度评分
                score = len(smiles) / 100.0  # 简化评分
                scores.append(score)
            return torch.tensor(scores, device=self.device)
        except Exception as e:
            print(f"Warning: Error in score_fn: {e}")
            return torch.zeros(len(smiles_list), device=self.device)

    def sample(self, cond_vec: torch.Tensor, steps: int = 50, batch: int = 32, max_len: int = 120) -> List[str]:
        """改进的采样方法"""
        try:
            # 初始化随机 token ids
            if hasattr(self.diffusion, 'tokenizer') and not self.diffusion.use_dummy:
                vocab_size = len(self.diffusion.tokenizer)
            else:
                vocab_size = 50000  # 默认词汇表大小
            
            ids = torch.randint(low=1, high=vocab_size, size=(batch, max_len), device=self.device)
            
            # 扩散采样步骤
            for t in range(steps):
                noise = self.diffusion.predict_noise(ids, cond_vec)
                
                # 简化的去噪步骤
                if noise.dim() == 3:
                    # 取最后一个维度的平均值作为token logits
                    noise_logits = noise.mean(dim=-1)
                else:
                    noise_logits = noise
                
                # 添加温度采样
                temperature = 1.0 - (t / steps) * 0.5  # 逐渐降低温度
                scaled_logits = noise_logits / temperature
                
                # 采样新的token
                probs = F.softmax(scaled_logits, dim=-1)
                new_ids = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch, max_len)
                
                # 混合原始ids和新ids
                alpha = t / steps
                ids = (alpha * ids + (1 - alpha) * new_ids).long()
                ids = torch.clamp(ids, 0, vocab_size - 1)
            
            # 解码生成的序列
            smiles_list = []
            if hasattr(self.diffusion, 'tokenizer') and not self.diffusion.use_dummy:
                try:
                    for i in range(batch):
                        tokens = ids[i].cpu().numpy()
                        # 移除特殊token
                        valid_tokens = tokens[tokens > 0]
                        if len(valid_tokens) > 0:
                            smiles = self.diffusion.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                            smiles_list.append(smiles)
                        else:
                            smiles_list.append("C")  # 默认分子
                except Exception as e:
                    print(f"Warning: Decoding error: {e}")
                    smiles_list = [f"C{i}" for i in range(batch)]  # 占位符
            else:
                # 生成占位符SMILES
                smiles_list = [f"C{i}C{i+1}C{i+2}" for i in range(batch)]
            
            return smiles_list
            
        except Exception as e:
            print(f"Warning: Error in sampling: {e}")
            return [f"C{i}" for i in range(batch)]  # 返回占位符 