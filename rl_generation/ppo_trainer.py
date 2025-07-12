#!/usr/bin/env python3
"""
PPO 强化学习训练框架

实现 Proximal Policy Optimization (PPO) 算法用于微调 ChemGPT：
1. 策略网络：基于 ChemGPT 的生成模型
2. 价值网络：估计状态价值
3. 经验回放：收集生成轨迹
4. 策略更新：最大化奖励信号

支持多目标优化：
- 分子性质奖励 (ΔE, σ, 多样性)
- 合成易度惩罚
- KL 散度正则化

依赖：
    pip install torch transformers accelerate wandb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
from dataclasses import dataclass
import pickle
import time
from collections import deque
import wandb

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer
    )
    from accelerate import Accelerator
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Using simplified implementation.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    """PPO 训练配置"""
    # 模型参数
    model_name: str = "ncfrey/ChemGPT-1.2B"
    max_length: int = 128
    temperature: float = 1.0
    
    # PPO 参数
    learning_rate: float = 1e-5
    batch_size: int = 32
    mini_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    
    # 训练参数
    max_episodes: int = 1000
    max_steps_per_episode: int = 50
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE 参数
    
    # KL 散度控制
    kl_coeff: float = 0.1
    target_kl: float = 0.01
    adaptive_kl: bool = True
    
    # 奖励参数
    reward_scale: float = 1.0
    reward_clip: float = 10.0
    
    # 日志和保存
    log_interval: int = 10
    save_interval: int = 100
    wandb_project: str = "sei-chemgpt-rl"

class PPOMemory:
    """PPO 经验回放缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def add(self, state, action, reward, log_prob, value, done):
        """添加经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_advantages(self, gamma: float, gae_lambda: float, last_value: float = 0):
        """计算 GAE 优势"""
        rewards = self.rewards + [last_value]
        values = self.values + [last_value]
        
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards) - 1)):
            delta = rewards[i] + gamma * values[i + 1] * (1 - self.dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
        
        self.advantages = advantages
        self.returns = [adv + val for adv, val in zip(advantages, self.values)]
    
    def get_batches(self, batch_size: int):
        """获取训练批次"""
        indices = np.random.permutation(len(self.states))
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            yield {
                'states': [self.states[idx] for idx in batch_indices],
                'actions': torch.tensor([self.actions[idx] for idx in batch_indices]),
                'old_log_probs': torch.tensor([self.log_probs[idx] for idx in batch_indices]),
                'advantages': torch.tensor([self.advantages[idx] for idx in batch_indices]),
                'returns': torch.tensor([self.returns[idx] for idx in batch_indices]),
                'values': torch.tensor([self.values[idx] for idx in batch_indices])
            }
    
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()

class ChemGPTActor(nn.Module):
    """基于 ChemGPT 的策略网络"""
    
    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config
        
        if TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
            
            # 添加特殊 token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            # 简化实现
            self.vocab_size = 50000
            self.embed_dim = 768
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
            self.transformer = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(self.embed_dim, 8, 2048),
                num_layers=6
            )
            self.lm_head = nn.Linear(self.embed_dim, self.vocab_size)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """前向传播"""
        if TRANSFORMERS_AVAILABLE:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            return outputs
        else:
            # 简化实现
            x = self.embedding(input_ids)
            x = self.transformer(x, x)
            logits = self.lm_head(x)
            return type('Outputs', (), {'logits': logits, 'loss': None})()
    
    def generate(self, prompt: str, max_length: int = None, temperature: float = None) -> Tuple[str, List[float]]:
        """生成分子 SELFIES 并返回对数概率"""
        if max_length is None:
            max_length = self.config.max_length
        if temperature is None:
            temperature = self.config.temperature
        
        if TRANSFORMERS_AVAILABLE:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # 生成序列
            generated_ids = []
            log_probs = []
            
            with torch.no_grad():
                for _ in range(max_length - input_ids.size(1)):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[:, -1, :] / temperature
                    
                    # 计算概率分布
                    probs = F.softmax(logits, dim=-1)
                    dist = Categorical(probs)
                    
                    # 采样下一个 token
                    next_token = dist.sample()
                    log_prob = dist.log_prob(next_token)
                    
                    generated_ids.append(next_token.item())
                    log_probs.append(log_prob.item())
                    
                    # 更新输入
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones(1, 1)], dim=1)
                    
                    # 检查结束条件
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
            
            # 解码生成的序列
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return generated_text, log_probs
        else:
            # 简化实现
            return "C[C@H]1CC[C@@H](C)CC1", [0.1] * 10  # 占位符

class ChemGPTCritic(nn.Module):
    """价值网络"""
    
    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config
        
        if TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
            
            # 添加价值头
            hidden_size = self.model.config.hidden_size
            self.value_head = nn.Linear(hidden_size, 1)
        else:
            # 简化实现
            self.embed_dim = 768
            self.embedding = nn.Embedding(50000, self.embed_dim)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(self.embed_dim, 8, 2048),
                num_layers=6
            )
            self.value_head = nn.Linear(self.embed_dim, 1)
    
    def forward(self, input_ids, attention_mask=None):
        """前向传播"""
        if TRANSFORMERS_AVAILABLE:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            
            # 使用最后一个 token 的隐藏状态
            last_hidden = hidden_states[:, -1, :]
            value = self.value_head(last_hidden)
            return value.squeeze(-1)
        else:
            # 简化实现
            x = self.embedding(input_ids)
            x = self.transformer(x)
            value = self.value_head(x.mean(dim=1))
            return value.squeeze(-1)

class PPOTrainer:
    """PPO 训练器"""
    
    def __init__(self, config: PPOConfig, reward_function, work_dir: str = "ppo_training"):
        """
        初始化 PPO 训练器
        
        Args:
            config: PPO 配置
            reward_function: 奖励函数
            work_dir: 工作目录
        """
        self.config = config
        self.reward_function = reward_function
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        
        # 初始化网络
        self.actor = ChemGPTActor(config)
        self.critic = ChemGPTCritic(config)
        
        # 保存原始策略用于 KL 散度计算
        self.ref_actor = ChemGPTActor(config)
        self.ref_actor.load_state_dict(self.actor.state_dict())
        
        # 优化器
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), 
            lr=config.learning_rate
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), 
            lr=config.learning_rate
        )
        
        # 经验回放
        self.memory = PPOMemory()
        
        # 训练统计
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.kl_divergences = deque(maxlen=100)
        
        # 初始化 wandb
        if config.wandb_project:
            wandb.init(
                project=config.wandb_project,
                config=config.__dict__,
                name=f"ppo_chemgpt_{int(time.time())}"
            )
    
    def collect_experience(self, num_episodes: int) -> Dict[str, float]:
        """收集经验"""
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            # 初始化回合
            state = "[START]"  # 起始提示
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.config.max_steps_per_episode):
                # 生成分子
                generated_mol, log_probs = self.actor.generate(state)
                
                # 计算奖励
                reward = self.reward_function(generated_mol)
                
                # 计算状态价值
                if TRANSFORMERS_AVAILABLE:
                    inputs = self.actor.tokenizer(state, return_tensors="pt", padding=True)
                    value = self.critic(inputs["input_ids"], inputs["attention_mask"])
                else:
                    # 简化实现
                    value = torch.tensor([0.0])
                
                # 存储经验
                self.memory.add(
                    state=state,
                    action=0,  # 简化，实际应该是 token 序列
                    reward=reward,
                    log_prob=sum(log_probs) if log_probs else 0.0,
                    value=value.item(),
                    done=True  # 每个分子生成都是一个完整的回合
                )
                
                episode_reward += reward
                episode_length += 1
                
                # 更新状态
                state = generated_mol
                
                # 检查结束条件
                if reward > 0.8:  # 高质量分子，提前结束
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # 计算优势
        self.memory.compute_advantages(self.config.gamma, self.config.gae_lambda)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'num_episodes': num_episodes
        }
    
    def update_policy(self) -> Dict[str, float]:
        """更新策略"""
        policy_losses = []
        value_losses = []
        kl_divergences = []
        
        for epoch in range(self.config.ppo_epochs):
            for batch in self.memory.get_batches(self.config.mini_batch_size):
                # 计算新的对数概率和价值
                if TRANSFORMERS_AVAILABLE:
                    # 这里需要重新计算，简化实现
                    new_log_probs = batch['old_log_probs']  # 占位符
                    values = batch['values']  # 占位符
                else:
                    new_log_probs = batch['old_log_probs']
                    values = batch['values']
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch['old_log_probs'])
                
                # 计算策略损失
                advantages = batch['advantages']
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                value_loss = F.mse_loss(values, batch['returns'])
                
                # 计算 KL 散度
                kl_div = (batch['old_log_probs'] - new_log_probs).mean()
                
                # 总损失
                total_loss = (policy_loss + 
                             self.config.value_loss_coeff * value_loss + 
                             self.config.kl_coeff * kl_div)
                
                # 反向传播
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # 记录损失
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                kl_divergences.append(kl_div.item())
        
        # 自适应 KL 系数
        mean_kl = np.mean(kl_divergences)
        if self.config.adaptive_kl:
            if mean_kl > self.config.target_kl * 2:
                self.config.kl_coeff *= 1.5
            elif mean_kl < self.config.target_kl / 2:
                self.config.kl_coeff *= 0.5
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'kl_divergence': mean_kl,
            'kl_coeff': self.config.kl_coeff
        }
    
    def train(self) -> Dict[str, Any]:
        """训练主循环"""
        logger.info("开始 PPO 训练...")
        
        training_stats = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'kl_divergences': []
        }
        
        for episode in range(self.config.max_episodes):
            # 收集经验
            experience_stats = self.collect_experience(self.config.batch_size)
            
            # 更新策略
            update_stats = self.update_policy()
            
            # 清空经验缓冲区
            self.memory.clear()
            
            # 记录统计信息
            training_stats['episode_rewards'].append(experience_stats['mean_reward'])
            training_stats['policy_losses'].append(update_stats['policy_loss'])
            training_stats['value_losses'].append(update_stats['value_loss'])
            training_stats['kl_divergences'].append(update_stats['kl_divergence'])
            
            # 日志记录
            if episode % self.config.log_interval == 0:
                logger.info(f"Episode {episode}: "
                           f"Reward={experience_stats['mean_reward']:.3f}, "
                           f"Policy Loss={update_stats['policy_loss']:.3f}, "
                           f"KL Div={update_stats['kl_divergence']:.3f}")
                
                # wandb 记录
                if self.config.wandb_project:
                    wandb.log({
                        'episode': episode,
                        'mean_reward': experience_stats['mean_reward'],
                        'policy_loss': update_stats['policy_loss'],
                        'value_loss': update_stats['value_loss'],
                        'kl_divergence': update_stats['kl_divergence'],
                        'kl_coeff': update_stats['kl_coeff']
                    })
            
            # 保存模型
            if episode % self.config.save_interval == 0:
                self.save_model(f"episode_{episode}")
        
        # 保存最终结果
        self.save_training_results(training_stats)
        
        logger.info("PPO 训练完成")
        return training_stats
    
    def save_model(self, checkpoint_name: str):
        """保存模型"""
        checkpoint_dir = self.work_dir / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # 保存模型状态
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config
        }, checkpoint_dir / "model.pt")
        
        # 保存 tokenizer (如果使用 transformers)
        if TRANSFORMERS_AVAILABLE:
            self.actor.tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
        
        logger.info(f"模型已保存至: {checkpoint_dir}")
    
    def save_training_results(self, stats: Dict[str, Any]):
        """保存训练结果"""
        results_file = self.work_dir / "training_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(stats, f)
        
        logger.info(f"训练结果已保存至: {results_file}")

def main():
    """主函数 - 演示 PPO 训练"""
    print("=== PPO 强化学习训练框架测试 ===")
    
    # 简化的奖励函数
    def simple_reward_function(molecule: str) -> float:
        """简化的奖励函数"""
        # 基于分子长度和复杂度的简单奖励
        if not molecule or len(molecule) < 5:
            return 0.0
        
        # 奖励适中长度的分子
        length_reward = 1.0 if 10 <= len(molecule) <= 50 else 0.5
        
        # 奖励包含特定官能团的分子
        functional_groups = ['C', 'O', 'N', 'S', 'P']
        diversity_reward = sum(1 for fg in functional_groups if fg in molecule) / len(functional_groups)
        
        return length_reward * diversity_reward
    
    # 创建配置
    config = PPOConfig(
        max_episodes=50,  # 演示用小数值
        batch_size=8,
        mini_batch_size=4,
        log_interval=5,
        save_interval=20,
        wandb_project=None  # 关闭 wandb
    )
    
    # 创建训练器
    trainer = PPOTrainer(
        config=config,
        reward_function=simple_reward_function,
        work_dir="test_ppo_training"
    )
    
    # 开始训练
    print("开始 PPO 训练演示...")
    stats = trainer.train()
    
    # 输出结果
    print("\n=== PPO 训练结果 ===")
    if stats['episode_rewards']:
        print(f"平均奖励: {np.mean(stats['episode_rewards']):.3f}")
        print(f"最高奖励: {np.max(stats['episode_rewards']):.3f}")
        print(f"最终奖励: {stats['episode_rewards'][-1]:.3f}")
    
    print("\n✓ PPO 强化学习训练框架测试完成")

if __name__ == "__main__":
    main() 