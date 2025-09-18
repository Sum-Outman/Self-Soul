"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

"""
在线学习系统 - 实现AGI的自主学习和持续改进能力
Online Learning System - Implements AGI's autonomous learning and continuous improvement capabilities

提供持续学习、元学习、经验回放和自适应优化功能
Provides continuous learning, meta-learning, experience replay and adaptive optimization
"""
import time
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import pickle
from pathlib import Path

from .error_handling import error_handler
from .model_registry import model_registry

class ExperienceBuffer:
    """经验缓冲区 - 存储和管理学习经验"""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = []
        self.max_size = max_size
        self.priorities = []  # 经验优先级
        self.learning_signals = []  # 学习信号强度
        
    def add_experience(self, experience: Dict[str, Any], priority: float = 1.0, 
                      learning_signal: float = 1.0):
        """添加新经验到缓冲区"""
        if len(self.buffer) >= self.max_size:
            # 移除最低优先级的经验
            min_idx = np.argmin(self.priorities)
            self.buffer.pop(min_idx)
            self.priorities.pop(min_idx)
            self.learning_signals.pop(min_idx)
            
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.learning_signals.append(learning_signal)
        
    def sample_batch(self, batch_size: int = 32, prioritized: bool = True):
        """从缓冲区采样批次经验"""
        if len(self.buffer) == 0:
            return []
            
        if prioritized and len(self.buffer) > 0:
            # 基于优先级采样
            probabilities = np.array(self.priorities) / sum(self.priorities)
            indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), 
                                     p=probabilities, replace=False)
        else:
            # 均匀采样
            indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), 
                                     replace=False)
            
        return [self.buffer[i] for i in indices]
    
    def get_learning_signals(self, indices: List[int]):
        """获取指定经验的学信号强度"""
        return [self.learning_signals[i] for i in indices]
    
    def update_priorities(self, indices: List[int], new_priorities: List[float]):
        """更新经验的优先级"""
        for idx, priority in zip(indices, new_priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
                
    def size(self):
        """返回缓冲区当前大小"""
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.priorities.clear()
        self.learning_signals.clear()

class MetaLearner:
    """元学习器 - 学习如何更好地学习"""
    
    def __init__(self):
        self.learning_rates = {}  # 模型特定的学习率
        self.optimization_strategies = {}  # 优化策略
        self.transfer_knowledge = {}  # 迁移学习知识
        self.meta_learning_rate = 0.001  # 元学习率
        
    def optimize_learning_strategy(self, model_id: str, performance_metrics: Dict[str, float]):
        """基于性能指标优化学习策略"""
        current_strategy = self.optimization_strategies.get(model_id, {
            'learning_rate': 0.01,
            'batch_size': 32,
            'optimizer': 'adam',
            'momentum': 0.9
        })
        
        # 基于性能动态调整学习率
        if 'loss' in performance_metrics and 'accuracy' in performance_metrics:
            loss = performance_metrics['loss']
            accuracy = performance_metrics['accuracy']
            
            # 自适应学习率调整
            if loss < 0.1 and accuracy > 0.9:
                # 性能良好，降低学习率进行精细调优
                current_strategy['learning_rate'] *= 0.8
            elif loss > 0.5 or accuracy < 0.6:
                # 性能较差，增加学习率
                current_strategy['learning_rate'] *= 1.2
                
            # 限制学习率范围
            current_strategy['learning_rate'] = max(1e-6, min(0.1, current_strategy['learning_rate']))
            
        self.optimization_strategies[model_id] = current_strategy
        return current_strategy
    
    def transfer_learning(self, source_model: str, target_model: str, similarity: float = 0.7):
        """在模型间迁移学习知识"""
        if source_model in self.transfer_knowledge and similarity > 0.5:
            source_knowledge = self.transfer_knowledge[source_model]
            # 将相关知识迁移到目标模型
            if target_model not in self.transfer_knowledge:
                self.transfer_knowledge[target_model] = {}
                
            for key, value in source_knowledge.items():
                # 基于相似度调整迁移强度
                self.transfer_knowledge[target_model][key] = value * similarity
                
            return True
        return False
    
    def learn_to_learn(self, learning_episodes: List[Dict[str, Any]]):
        """元学习：从学习经历中学习如何更好地学习"""
        for episode in learning_episodes:
            model_id = episode.get('model_id')
            strategy = episode.get('strategy', {})
            outcome = episode.get('outcome', {})
            
            if model_id and 'improvement' in outcome:
                improvement = outcome['improvement']
                # 根据学习效果调整元学习参数
                self.meta_learning_rate *= (1 + 0.1 * improvement)

class OnlineLearningSystem:
    """在线学习系统 - AGI核心学习能力"""
    
    def __init__(self, learning_rate: float = 0.01, meta_learning_enabled: bool = True):
        self.learning_rate = learning_rate
        self.meta_learning_enabled = meta_learning_enabled
        self.experience_buffer = ExperienceBuffer(max_size=50000)
        self.meta_learner = MetaLearner()
        self.learning_history = []
        self.adaptive_parameters = {}
        self.last_update_time = time.time()
        
        # 学习统计
        self.learning_cycles = 0
        self.total_experiences = 0
        self.average_improvement = 0.0
        
    def continuous_learning(self, experience: Dict[str, Any]):
        """持续学习循环 - 核心学习方法"""
        try:
            # 存储经验
            priority = self._calculate_experience_priority(experience)
            learning_signal = self._calculate_learning_signal(experience)
            
            self.experience_buffer.add_experience(experience, priority, learning_signal)
            self.total_experiences += 1
            
            # 定期进行学习更新
            current_time = time.time()
            if current_time - self.last_update_time > 300:  # 每5分钟学习一次
                self._update_models()
                self.last_update_time = current_time
                
                if self.meta_learning_enabled:
                    self._meta_learn()
                    
            # 记录学习历史
            self._record_learning_cycle(experience)
            
            return True
            
        except Exception as e:
            error_handler.handle_error(e, "OnlineLearningSystem", "持续学习过程中出错")
            return False
    
    def _calculate_experience_priority(self, experience: Dict[str, Any]) -> float:
        """计算经验优先级"""
        priority = 1.0
        
        # 基于 novelty（新颖性）
        if 'novelty' in experience:
            priority *= (1 + experience['novelty'])
            
        # 基于 uncertainty（不确定性）
        if 'uncertainty' in experience:
            priority *= (1 + experience['uncertainty'])
            
        # 基于 learning_potential（学习潜力）
        if 'learning_potential' in experience:
            priority *= (1 + experience['learning_potential'])
            
        return min(10.0, max(0.1, priority))  # 限制优先级范围
    
    def _calculate_learning_signal(self, experience: Dict[str, Any]) -> float:
        """计算学习信号强度"""
        signal = 1.0
        
        # 基于 reward（奖励）
        if 'reward' in experience:
            signal *= (1 + abs(experience['reward']))
            
        # 基于 error（误差）
        if 'error' in experience:
            signal *= (1 + experience['error'])
            
        # 基于 surprise（惊讶度）
        if 'surprise' in experience:
            signal *= (1 + experience['surprise'])
            
        return min(5.0, max(0.1, signal))  # 限制信号强度范围
    
    def _update_models(self):
        """更新所有模型参数"""
        try:
            # 采样经验批次
            batch = self.experience_buffer.sample_batch(batch_size=64, prioritized=True)
            
            if not batch:
                return
                
            # 为每个相关模型更新参数
            model_updates = {}
            for experience in batch:
                model_id = experience.get('model_id')
                if model_id and model_id in model_registry.models:
                    if model_id not in model_updates:
                        model_updates[model_id] = []
                    model_updates[model_id].append(experience)
            
            # 应用更新到各个模型
            for model_id, experiences in model_updates.items():
                model = model_registry.get_model(model_id)
                if model and hasattr(model, 'online_update'):
                    # 获取元学习优化策略
                    strategy = self.meta_learner.optimize_learning_strategy(
                        model_id, 
                        model_registry.performance_metrics.get(model_id, {})
                    )
                    
                    # 应用在线更新
                    improvement = model.online_update(experiences, strategy, self.learning_rate)
                    
                    # 记录学习效果
                    if improvement is not None:
                        self.average_improvement = (
                            0.9 * self.average_improvement + 0.1 * improvement
                        )
            
            self.learning_cycles += 1
            
        except Exception as e:
            error_handler.handle_error(e, "OnlineLearningSystem", "模型更新过程中出错")
    
    def _meta_learn(self):
        """元学习优化"""
        try:
            # 从学习历史中提取元学习样本
            recent_history = self.learning_history[-100:] if len(self.learning_history) > 100 else self.learning_history
            
            if recent_history:
                self.meta_learner.learn_to_learn(recent_history)
                
                # 更新全局学习参数
                self._adapt_learning_parameters()
                
        except Exception as e:
            error_handler.handle_error(e, "OnlineLearningSystem", "元学习过程中出错")
    
    def _adapt_learning_parameters(self):
        """自适应调整学习参数"""
        # 基于学习效果动态调整学习率
        if self.average_improvement > 0.1:
            # 学习效果好，稍微增加学习率
            self.learning_rate *= 1.05
        elif self.average_improvement < 0.01:
            # 学习效果差，降低学习率
            self.learning_rate *= 0.9
            
        # 限制学习率范围
        self.learning_rate = max(1e-6, min(0.1, self.learning_rate))
        
        # 更新自适应参数
        self.adaptive_parameters = {
            'learning_rate': self.learning_rate,
            'meta_learning_rate': self.meta_learner.meta_learning_rate,
            'buffer_size': self.experience_buffer.size(),
            'average_improvement': self.average_improvement,
            'learning_cycles': self.learning_cycles,
            'total_experiences': self.total_experiences,
            'last_update': datetime.now().isoformat()
        }
    
    def _record_learning_cycle(self, experience: Dict[str, Any]):
        """记录学习周期"""
        learning_cycle = {
            'timestamp': datetime.now().isoformat(),
            'model_id': experience.get('model_id'),
            'experience_type': experience.get('type', 'unknown'),
            'learning_signal': self._calculate_learning_signal(experience),
            'improvement_estimate': self._estimate_learning_potential(experience),
            'buffer_size': self.experience_buffer.size()
        }
        
        self.learning_history.append(learning_cycle)
        
        # 保持历史记录大小合理
        if len(self.learning_history) > 1000:
            self.learning_history = self.learning_history[-1000:]
    
    def _estimate_learning_potential(self, experience: Dict[str, Any]) -> float:
        """估计学习潜力"""
        potential = 0.5  # 基础潜力
        
        # 基于经验特征调整潜力估计
        if 'novelty' in experience:
            potential += 0.2 * experience['novelty']
            
        if 'complexity' in experience:
            potential += 0.1 * min(1.0, experience['complexity'] / 10.0)
            
        if 'diversity' in experience:
            potential += 0.15 * experience['diversity']
            
        return min(1.0, max(0.0, potential))
    
    def get_learning_status(self) -> Dict[str, Any]:
        """获取学习系统状态"""
        return {
            'learning_cycles': self.learning_cycles,
            'total_experiences': self.total_experiences,
            'buffer_size': self.experience_buffer.size(),
            'learning_rate': self.learning_rate,
            'average_improvement': self.average_improvement,
            'meta_learning_enabled': self.meta_learning_enabled,
            'adaptive_parameters': self.adaptive_parameters,
            'last_update': self.last_update_time,
            'status': 'active'
        }
    
    def save_learning_state(self, filepath: str):
        """保存学习状态到文件"""
        try:
            state = {
                'experience_buffer': self.experience_buffer,
                'meta_learner': self.meta_learner,
                'learning_history': self.learning_history,
                'adaptive_parameters': self.adaptive_parameters,
                'learning_cycles': self.learning_cycles,
                'total_experiences': self.total_experiences,
                'average_improvement': self.average_improvement
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
                
            error_handler.log_info(f"学习状态已保存到: {filepath}", "OnlineLearningSystem")
            return True
            
        except Exception as e:
            error_handler.handle_error(e, "OnlineLearningSystem", "保存学习状态失败")
            return False
    
    def load_learning_state(self, filepath: str):
        """从文件加载学习状态"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
            self.experience_buffer = state.get('experience_buffer', ExperienceBuffer())
            self.meta_learner = state.get('meta_learner', MetaLearner())
            self.learning_history = state.get('learning_history', [])
            self.adaptive_parameters = state.get('adaptive_parameters', {})
            self.learning_cycles = state.get('learning_cycles', 0)
            self.total_experiences = state.get('total_experiences', 0)
            self.average_improvement = state.get('average_improvement', 0.0)
            
            error_handler.log_info(f"学习状态已从 {filepath} 加载", "OnlineLearningSystem")
            return True
            
        except Exception as e:
            error_handler.handle_error(e, "OnlineLearningSystem", "加载学习状态失败")
            return False
    
    def reset_learning(self):
        """重置学习系统"""
        self.experience_buffer.clear()
        self.meta_learner = MetaLearner()
        self.learning_history.clear()
        self.adaptive_parameters = {}
        self.learning_cycles = 0
        self.total_experiences = 0
        self.average_improvement = 0.0
        self.learning_rate = 0.01
        self.last_update_time = time.time()
        
        error_handler.log_info("学习系统已重置", "OnlineLearningSystem")

# 创建全局在线学习系统实例
online_learning_system = OnlineLearningSystem()

# 模块级函数：提供对全局实例方法的直接访问
def get_online_learning_system() -> OnlineLearningSystem:
    """获取全局在线学习系统实例"""
    return online_learning_system

def continuous_learning(experience: Dict[str, Any]) -> bool:
    """持续学习接口"""
    return online_learning_system.continuous_learning(experience)

def get_learning_status() -> Dict[str, Any]:
    """获取学习状态"""
    return online_learning_system.get_learning_status()

def save_learning_state(filepath: str) -> bool:
    """保存学习状态"""
    return online_learning_system.save_learning_state(filepath)

def load_learning_state(filepath: str) -> bool:
    """加载学习状态"""
    return online_learning_system.load_learning_state(filepath)

def reset_learning() -> bool:
    """重置学习系统"""
    online_learning_system.reset_learning()
    return True