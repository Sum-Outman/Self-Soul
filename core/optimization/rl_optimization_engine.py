#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习优化引擎 - Reinforcement Learning Optimization Engine

使用强化学习算法（如PPO、DQN）优化演化策略本身：
1. PPO（Proximal Policy Optimization）：优化连续动作空间（参数调整）
2. DQN（Deep Q-Network）：优化离散动作空间（算法选择）
3. 混合方法：结合离散和连续动作空间

应用场景：
- 优化演化算法的参数选择（突变率、交叉率等）
- 动态选择最佳演化策略
- 自适应调整演化过程中的参数
- 学习不同任务类型的最优演化策略

设计原则：
- 模块化：RL算法可插拔
- 可扩展：支持多种RL算法
- 高效：优化计算资源使用
- 可解释：提供RL决策的解释
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import math
from collections import deque, defaultdict

# 检查PyTorch是否可用
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal, Categorical
    TORCH_AVAILABLE = True
except ImportError:
    logging.warning("PyTorch not available, RL functionality will be limited")
    TORCH_AVAILABLE = False

# 配置日志
logger = logging.getLogger(__name__)


class RLOptimizationAlgorithm(Enum):
    """强化学习优化算法类型"""
    PPO = "ppo"  # Proximal Policy Optimization
    DQN = "dqn"  # Deep Q-Network
    A2C = "a2c"  # Advantage Actor-Critic
    SAC = "sac"  # Soft Actor-Critic
    TD3 = "td3"  # Twin Delayed DDPG


class ActionSpaceType(Enum):
    """动作空间类型"""
    DISCRETE = "discrete"  # 离散动作（算法选择）
    CONTINUOUS = "continuous"  # 连续动作（参数调整）
    HYBRID = "hybrid"  # 混合动作空间


@dataclass
class RLOptimizerConfig:
    """强化学习优化器配置"""
    
    # 算法选择
    algorithm: RLOptimizationAlgorithm = RLOptimizationAlgorithm.PPO
    
    # 网络架构
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"
    
    # 学习参数
    learning_rate: float = 3e-4
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO裁剪参数
    entropy_coef: float = 0.01  # 熵正则化系数
    value_coef: float = 0.5  # 价值函数系数
    
    # 训练参数
    batch_size: int = 64
    buffer_size: int = 10000  # 经验回放缓冲区大小
    update_frequency: int = 10  # 更新频率（步数）
    num_epochs: int = 4  # 每次更新的训练轮数
    
    # 探索参数
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.01
    
    # 动作空间配置
    action_space_type: ActionSpaceType = ActionSpaceType.HYBRID
    
    # 离散动作配置
    discrete_actions: List[str] = field(default_factory=lambda: [
        "genetic_algorithm",
        "particle_swarm",
        "differential_evolution",
        "bayesian_optimization",
        "simulated_annealing"
    ])
    
    # 连续动作配置
    continuous_action_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "mutation_rate": (0.01, 0.5),
        "crossover_rate": (0.1, 0.9),
        "population_size": (10, 200),
        "learning_rate": (1e-5, 1e-2)
    })
    
    # 状态空间配置
    state_features: List[str] = field(default_factory=lambda: [
        "task_complexity",
        "task_type_encoding",
        "performance_history",
        "resource_constraints",
        "evolution_progress"
    ])


@dataclass
class RLTransition:
    """强化学习转移记录"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, transition: RLTransition):
        """添加转移记录"""
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> List[RLTransition]:
        """采样批量转移记录"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class ActorNetwork(nn.Module):
    """Actor网络（策略网络）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()
        
        layers = []
        prev_size = state_dim
        
        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # 输出层 - 连续动作的均值和标准差
        self.mean_layer = nn.Linear(prev_size, action_dim)
        self.log_std_layer = nn.Linear(prev_size, action_dim)
        
        # 初始化参数
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        """前向传播"""
        hidden = self.hidden_layers(state)
        
        # 计算动作均值和标准差
        mean = self.mean_layer(hidden)
        log_std = self.log_std_layer(hidden)
        
        # 使用tanh将均值限制在[-1, 1]范围内
        mean = torch.tanh(mean)
        
        # 限制标准差范围
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        return mean, std


class CriticNetwork(nn.Module):
    """Critic网络（价值网络）"""
    
    def __init__(self, state_dim: int, hidden_sizes: List[int]):
        super().__init__()
        
        layers = []
        prev_size = state_dim
        
        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        self.value_layer = nn.Linear(prev_size, 1)
        
        # 初始化参数
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        """前向传播"""
        hidden = self.hidden_layers(state)
        value = self.value_layer(hidden)
        return value


class PPOOptimizer:
    """PPO优化器实现"""
    
    def __init__(self, config: RLOptimizerConfig, state_dim: int, action_dim: int):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PPO optimizer")
        
        # 创建网络
        self.actor = ActorNetwork(state_dim, action_dim, config.hidden_sizes)
        self.critic = CriticNetwork(state_dim, config.hidden_sizes)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # 经验缓冲区
        self.buffer = ReplayBuffer(config.buffer_size)
        
        # 训练状态
        self.current_episode = 0
        self.total_steps = 0
        self.exploration_rate = config.exploration_rate
        
        logger.info(f"PPO优化器初始化完成，状态维度: {state_dim}, 动作维度: {action_dim}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            mean, std = self.actor(state_tensor)
            
            if deterministic:
                action = mean
            else:
                # 添加探索噪声
                normal = Normal(mean, std)
                action = normal.sample()
            
            # 计算动作的对数概率
            normal = Normal(mean, std)
            log_prob = normal.log_prob(action).sum(dim=-1)
            
            action = action.squeeze(0).numpy()
            log_prob = log_prob.item()
        
        return action, log_prob
    
    def update(self, transitions: List[RLTransition]):
        """更新策略"""
        if len(transitions) < self.config.batch_size:
            return
        
        # 准备批量数据
        states = torch.FloatTensor([t.state for t in transitions])
        actions = torch.FloatTensor([t.action for t in transitions])
        rewards = torch.FloatTensor([t.reward for t in transitions])
        next_states = torch.FloatTensor([t.next_state for t in transitions])
        dones = torch.FloatTensor([float(t.done) for t in transitions])
        
        # 计算优势函数
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            advantages = rewards + self.config.gamma * next_values * (1 - dones) - values
        
        # 多轮PPO更新
        for epoch in range(self.config.num_epochs):
            # 计算当前策略的对数概率
            mean, std = self.actor(states)
            normal = Normal(mean, std)
            current_log_probs = normal.log_prob(actions).sum(dim=-1)
            
            # 计算概率比率
            # 注意：这里简化了，实际PPO需要存储旧策略的对数概率
            ratios = torch.exp(current_log_probs - current_log_probs.detach())
            
            # 计算裁剪损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            value_pred = self.critic(states).squeeze()
            value_targets = advantages + values.detach()
            critic_loss = F.mse_loss(value_pred, value_targets)
            
            # 计算熵损失
            entropy = normal.entropy().mean()
            entropy_loss = -self.config.entropy_coef * entropy
            
            # 总损失
            total_loss = actor_loss + self.config.value_coef * critic_loss + entropy_loss
            
            # 反向传播
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        # 更新探索率
        self.exploration_rate = max(
            self.config.min_exploration_rate,
            self.exploration_rate * self.config.exploration_decay
        )
        
        logger.debug(f"PPO更新完成，损失: {total_loss.item():.4f}, 探索率: {self.exploration_rate:.4f}")
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'current_episode': self.current_episode,
            'total_steps': self.total_steps,
            'exploration_rate': self.exploration_rate,
        }
        torch.save(checkpoint, path)
        logger.info(f"检查点保存到: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.current_episode = checkpoint['current_episode']
        self.total_steps = checkpoint['total_steps']
        self.exploration_rate = checkpoint['exploration_rate']
        logger.info(f"检查点从 {path} 加载")


class ReinforcementLearningOptimizer:
    """强化学习优化器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 处理配置，转换字符串为枚举
        processed_config = config.copy() if config else {}
        
        # 转换算法字符串为枚举
        if "algorithm" in processed_config and isinstance(processed_config["algorithm"], str):
            algorithm_str = processed_config["algorithm"].upper()
            try:
                processed_config["algorithm"] = RLOptimizationAlgorithm[algorithm_str]
            except KeyError:
                logger.warning(f"未知算法: {algorithm_str}，使用默认PPO")
                processed_config["algorithm"] = RLOptimizationAlgorithm.PPO
        
        # 转换动作空间类型字符串为枚举
        if "action_space_type" in processed_config and isinstance(processed_config["action_space_type"], str):
            action_space_str = processed_config["action_space_type"].upper()
            try:
                processed_config["action_space_type"] = ActionSpaceType[action_space_str]
            except KeyError:
                logger.warning(f"未知动作空间类型: {action_space_str}，使用默认HYBRID")
                processed_config["action_space_type"] = ActionSpaceType.HYBRID
        
        self.config = RLOptimizerConfig(**processed_config)
        
        # 状态和动作维度
        # 注意：state_dim是编码后的状态维度，不是特征名称的数量
        # encode_state方法返回12维状态向量
        self.state_dim = 12  # 编码后的状态维度
        self.action_dim = len(self.config.continuous_action_bounds)
        
        # 创建RL算法实例
        self.algorithm = None
        if self.config.algorithm == RLOptimizationAlgorithm.PPO:
            if TORCH_AVAILABLE:
                self.algorithm = PPOOptimizer(self.config, self.state_dim, self.action_dim)
            else:
                logger.warning("PyTorch不可用，无法创建PPO优化器")
        
        # 训练历史
        self.training_history = {
            "episodes": [],
            "rewards": [],
            "episode_lengths": [],
            "exploration_rates": []
        }
        
        # 当前状态
        self.current_state = None
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        logger.info(f"强化学习优化器初始化完成，算法: {self.config.algorithm.value}")
    
    def encode_state(self, task_features: Dict[str, Any], 
                    performance_history: List[float],
                    evolution_progress: float) -> np.ndarray:
        """编码状态"""
        state = []
        
        # 任务复杂性
        state.append(task_features.get("task_complexity", 0.5))
        
        # 任务类型编码（简化）
        task_type = task_features.get("task_type", "generic")
        type_encoding = self._encode_task_type(task_type)
        state.extend(type_encoding)
        
        # 性能历史（最近几个时间步）
        recent_performance = performance_history[-5:] if performance_history else []
        if len(recent_performance) < 5:
            recent_performance.extend([0.0] * (5 - len(recent_performance)))
        state.extend(recent_performance[:5])  # 只取前5个
        
        # 资源约束（简化）
        resource_constraints = task_features.get("resource_constraints", {})
        state.append(resource_constraints.get("time_constrained", 0.0))
        state.append(resource_constraints.get("memory_constrained", 0.0))
        
        # 演化进度
        state.append(evolution_progress)
        
        return np.array(state, dtype=np.float32)
    
    def _encode_task_type(self, task_type: str) -> List[float]:
        """编码任务类型"""
        # 简化编码：根据任务类型的关键词
        encoding = [0.0, 0.0, 0.0]  # 分类、回归、强化学习
        
        task_type_lower = task_type.lower()
        if "classif" in task_type_lower:
            encoding[0] = 1.0
        elif "regress" in task_type_lower:
            encoding[1] = 1.0
        elif "reinforcement" in task_type_lower or "rl" in task_type_lower:
            encoding[2] = 1.0
        else:
            # 通用任务类型
            encoding = [0.33, 0.33, 0.33]
        
        return encoding
    
    def decode_action(self, action: np.ndarray) -> Dict[str, Any]:
        """解码动作为演化参数"""
        decoded = {}
        
        # 解码连续动作
        action_bounds = list(self.config.continuous_action_bounds.items())
        for i, (param_name, (low, high)) in enumerate(action_bounds):
            if i < len(action):
                # 裁剪动作值到[-1, 1]范围内
                clipped_action = np.clip(action[i], -1.0, 1.0)
                # 从[-1, 1]映射到实际范围
                scaled_action = (clipped_action + 1) / 2  # 映射到[0, 1]
                param_value = low + scaled_action * (high - low)
                decoded[param_name] = float(param_value)
        
        return decoded
    
    def start_episode(self, initial_state: np.ndarray):
        """开始新回合"""
        self.current_state = initial_state
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        logger.debug(f"开始新RL回合，初始状态: {initial_state}")
    
    def step(self, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool = False):
        """执行一步"""
        if self.current_state is None:
            raise ValueError("必须先调用start_episode()")
        
        # 记录转移
        transition = RLTransition(
            state=self.current_state.copy(),
            action=action.copy(),
            reward=reward,
            next_state=next_state.copy(),
            done=done
        )
        
        # 更新状态
        self.current_state = next_state
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # 添加到缓冲区
        if self.algorithm and hasattr(self.algorithm, 'buffer'):
            self.algorithm.buffer.push(transition)
        
        # 定期更新
        if self.algorithm and self.current_episode_length % self.config.update_frequency == 0:
            batch = self.algorithm.buffer.sample(self.config.batch_size)
            self.algorithm.update(batch)
        
        return transition
    
    def end_episode(self):
        """结束当前回合"""
        if self.current_state is None:
            return
        
        # 记录训练历史
        self.training_history["episodes"].append(self.algorithm.current_episode if self.algorithm else 0)
        self.training_history["rewards"].append(self.current_episode_reward)
        self.training_history["episode_lengths"].append(self.current_episode_length)
        self.training_history["exploration_rates"].append(
            self.algorithm.exploration_rate if self.algorithm else self.config.exploration_rate
        )
        
        # 更新算法状态
        if self.algorithm:
            self.algorithm.current_episode += 1
            self.algorithm.total_steps += self.current_episode_length
        
        logger.info(f"RL回合结束，奖励: {self.current_episode_reward:.2f}, 步数: {self.current_episode_length}")
        
        # 重置当前状态
        self.current_state = None
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def optimize_evolution_parameters(self, task_features: Dict[str, Any],
                                    performance_history: List[float],
                                    evolution_progress: float) -> Dict[str, Any]:
        """优化演化参数"""
        
        # 编码当前状态
        state = self.encode_state(task_features, performance_history, evolution_progress)
        
        # 选择动作
        if self.algorithm:
            action, _ = self.algorithm.select_action(state)
        else:
            # 随机动作作为回退
            action = np.random.uniform(-1, 1, size=self.action_dim)
        
        # 解码为演化参数
        evolution_params = self.decode_action(action)
        
        logger.debug(f"RL优化演化参数: {evolution_params}")
        return evolution_params
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        if not self.training_history["episodes"]:
            return {"status": "no_training_data"}
        
        rewards = self.training_history["rewards"]
        lengths = self.training_history["episode_lengths"]
        
        return {
            "total_episodes": len(rewards),
            "average_reward": np.mean(rewards) if rewards else 0.0,
            "best_reward": np.max(rewards) if rewards else 0.0,
            "average_episode_length": np.mean(lengths) if lengths else 0.0,
            "current_exploration_rate": self.algorithm.exploration_rate if self.algorithm else self.config.exploration_rate,
            "algorithm": self.config.algorithm.value
        }


# 工厂函数
def create_rl_optimizer(config: Optional[Dict[str, Any]] = None) -> ReinforcementLearningOptimizer:
    """创建强化学习优化器实例"""
    return ReinforcementLearningOptimizer(config)


if __name__ == "__main__":
    # 演示强化学习优化器
    print("=" * 80)
    print("强化学习优化器演示")
    print("=" * 80)
    
    try:
        # 创建RL优化器
        config = {
            "algorithm": "ppo",
            "learning_rate": 3e-4,
            "batch_size": 32,
            "hidden_sizes": [64, 64]
        }
        
        rl_optimizer = create_rl_optimizer(config)
        
        print(f"\n1. RL优化器初始化完成:")
        print(f"   算法: {rl_optimizer.config.algorithm.value}")
        print(f"   状态维度: {rl_optimizer.state_dim}")
        print(f"   动作维度: {rl_optimizer.action_dim}")
        
        # 模拟优化过程
        print("\n2. 模拟演化参数优化:")
        
        task_features = {
            "task_complexity": 0.7,
            "task_type": "neural_architecture_design",
            "resource_constraints": {
                "time_constrained": 0.3,
                "memory_constrained": 0.2
            }
        }
        
        performance_history = [0.6, 0.65, 0.7, 0.72, 0.75]
        evolution_progress = 0.5
        
        optimized_params = rl_optimizer.optimize_evolution_parameters(
            task_features, performance_history, evolution_progress
        )
        
        print(f"   优化后的参数:")
        for param, value in optimized_params.items():
            print(f"     {param}: {value:.4f}")
        
        # 获取训练统计
        stats = rl_optimizer.get_training_statistics()
        print(f"\n3. 训练统计:")
        for key, value in stats.items():
            print(f"    {key}: {value}")
        
        print("\n✓ 强化学习优化器演示完成")
        
    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()