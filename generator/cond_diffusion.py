import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import os

class CondDiffusionModel:
    """简单包装：在扩散反向步用 GPT 估计噪声 eps，支持条件向量拼接。"""
    def __init__(self, model_dir: str = 'generator/chemGPT', device: str = 'cuda'):
        self.device = device
        
        # 检查模型目录是否存在
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory {model_dir} not found. Using dummy implementation.")
            self.use_dummy = True
            self.emb_dim = 768
            return
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16).to(device)
            
            # 添加pad token如果不存在
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.emb_dim = self.model.config.n_embd if hasattr(self.model.config, 'n_embd') else self.model.config.hidden_size
            self.use_dummy = False
        except Exception as e:
            print(f"Warning: Failed to load model from {model_dir}: {e}. Using dummy implementation.")
            self.use_dummy = True
            self.emb_dim = 768

    @torch.inference_mode()
    def predict_noise(self, partial_ids: torch.Tensor, cond_vec: torch.Tensor):
        # partial_ids: [B, L] int64 ; cond_vec: [B, D]
        
        if self.use_dummy:
            # 返回随机噪声作为占位符
            B, L = partial_ids.shape
            return torch.randn(B, L, self.emb_dim, device=self.device)
        
        try:
            # 简化：将 cond_vec 投入模型 embedding 层 CLS 位
            inputs_embeds = self.model.transformer.wte(partial_ids)
            
            # 确保条件向量维度匹配
            if cond_vec.shape[1] > inputs_embeds.shape[2]:
                cond_vec = cond_vec[:, :inputs_embeds.shape[2]]
            elif cond_vec.shape[1] < inputs_embeds.shape[2]:
                # 填充零
                padding = torch.zeros(cond_vec.shape[0], inputs_embeds.shape[2] - cond_vec.shape[1], 
                                    device=cond_vec.device, dtype=cond_vec.dtype)
                cond_vec = torch.cat([cond_vec, padding], dim=1)
            
            # 替换第一 token embedding
            inputs_embeds[:, 0, :] = cond_vec
            outputs = self.model(inputs_embeds=inputs_embeds)
            
            # 使用最后隐藏层作为噪声估计
            last_hidden = outputs.last_hidden_state  # B,L,D
            return last_hidden
            
        except Exception as e:
            print(f"Warning: Error in predict_noise: {e}. Using random noise.")
            B, L = partial_ids.shape
            return torch.randn(B, L, self.emb_dim, device=self.device) 