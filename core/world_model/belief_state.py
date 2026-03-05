#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贝叶斯信念状态 - 实现智能体对世界状态的概率信念表示

核心功能:
1. 状态变量的概率分布表示
2. 贝叶斯更新（观测更新）
3. 预测更新（状态转移）
4. 信念融合与冲突解决
5. 不确定性度量和可视化

技术实现:
- 基于概率图模型的信念表示
- 离散和连续状态变量的统一处理
- 高效的贝叶斯推理算法
- 部分可观测环境下的信念维护

贝叶斯信念状态公式:
P(S_t | O_{1:t}) ∝ P(O_t | S_t) * Σ_{s_{t-1}} P(S_t | s_{t-1}) * P(s_{t-1} | O_{1:t-1})

其中:
- S_t: 时刻t的状态
- O_{1:t}: 到时刻t为止的所有观测
- P(S_t | s_{t-1}): 状态转移模型
- P(O_t | S_t): 观测模型

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import networkx as nx
from datetime import datetime

# 导入错误处理
from core.error_handling import ErrorHandler

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()


class BeliefRepresentation(Enum):
    """信念表示方法枚举"""
    DISCRETE_DISTRIBUTION = "discrete_distribution"      # 离散分布
    GAUSSIAN = "gaussian"                                # 高斯分布
    MIXTURE_MODEL = "mixture_model"                      # 混合模型
    PARTICLE_FILTER = "particle_filter"                  # 粒子滤波
    BAYESIAN_NETWORK = "bayesian_network"                # 贝叶斯网络


class UncertaintyMeasure(Enum):
    """不确定性度量方法枚举"""
    ENTROPY = "entropy"                    # 信息熵
    VARIANCE = "variance"                  # 方差
    CONFIDENCE_INTERVAL = "confidence_interval"  # 置信区间
    COVARIANCE = "covariance"              # 协方差矩阵
    KULLBACK_LEIBLER = "kullback_leibler"  # KL散度


class BeliefState:
    """
    贝叶斯信念状态 - 智能体对世界状态的概率信念
    
    核心特性:
    1. 多维状态空间表示
    2. 概率分布维护
    3. 不确定性量化
    4. 时间演化跟踪
    5. 信念更新和融合
    
    信念更新流程:
    1. 预测步: 基于状态转移模型预测下一时刻信念
    2. 更新步: 基于观测数据更新当前信念
    3. 归一化: 确保概率和为1
    4. 融合步: 融合多源信息
    
    技术实现:
    - 离散状态: 概率质量函数 (PMF)
    - 连续状态: 参数化分布 (高斯、混合)
    - 高维状态: 因子分解或粒子表示
    - 时间序列: 信念历史跟踪
    """
    
    def __init__(self, 
                 state_variables: Optional[List[str]] = None,
                 representation: Union[BeliefRepresentation, str] = BeliefRepresentation.DISCRETE_DISTRIBUTION,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化贝叶斯信念状态
        
        Args:
            state_variables: 状态变量列表
            representation: 信念表示方法
            config: 配置参数
        """
        # 配置参数
        self.config = config or self._get_default_config()
        
        # 状态变量
        self.state_variables = state_variables or []
        self.state_dimension = len(self.state_variables)
        
        # 信念表示方法
        if isinstance(representation, str):
            self.representation = BeliefRepresentation(representation)
        else:
            self.representation = representation
        
        # 信念存储
        self.belief_distribution = self._initialize_belief_distribution()
        self.belief_history: List[Dict[str, Any]] = []
        self.max_history_size = self.config.get('max_history_size', 1000)
        
        # 不确定性度量
        self.uncertainty_measures: Dict[str, float] = {}
        
        # 元数据
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "update_count": 0,
            "representation": self.representation.value,
            "state_variables": self.state_variables.copy()
        }
        
        # 性能统计
        self.performance_stats = {
            "updates_performed": 0,
            "predictions_made": 0,
            "fusions_performed": 0,
            "average_update_time": 0.0,
            "belief_entropy_history": [],
            "belief_confidence_history": []
        }
        
        logger.info(f"贝叶斯信念状态初始化完成，表示方法: {self.representation.value}")
        logger.info(f"状态变量: {self.state_variables}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'max_history_size': 1000,
            'uncertainty_threshold': 0.3,
            'belief_convergence_threshold': 0.01,
            'min_probability': 1e-10,
            'particle_count': 1000,
            'resampling_threshold': 0.5,
            'gaussian_approximation': True,
            'enable_history_tracking': True,
            'belief_fusion_method': 'bayesian_average'
        }
    
    def _initialize_belief_distribution(self) -> Dict[str, Any]:
        """
        初始化信念分布
        
        Returns:
            初始化后的信念分布
        """
        if self.representation == BeliefRepresentation.DISCRETE_DISTRIBUTION:
            # 离散分布：为每个状态变量创建均匀分布
            distribution = {}
            for var in self.state_variables:
                distribution[var] = {
                    "type": "discrete",
                    "states": [],      # 状态值列表
                    "probabilities": []  # 对应概率列表
                }
            return distribution
        
        elif self.representation == BeliefRepresentation.GAUSSIAN:
            # 高斯分布：为每个状态变量创建高斯分布
            distribution = {}
            for var in self.state_variables:
                distribution[var] = {
                    "type": "gaussian",
                    "mean": 0.0,
                    "variance": 1.0,
                    "confidence": 0.5
                }
            return distribution
        
        elif self.representation == BeliefRepresentation.PARTICLE_FILTER:
            # 粒子滤波：初始化粒子集
            particle_count = self.config.get('particle_count', 1000)
            distribution = {
                "type": "particle_filter",
                "particles": [],
                "weights": [],
                "effective_sample_size": particle_count
            }
            return distribution
        
        elif self.representation == BeliefRepresentation.BAYESIAN_NETWORK:
            # 贝叶斯网络：创建条件概率表
            distribution = {
                "type": "bayesian_network",
                "graph": nx.DiGraph(),
                "cpts": {},  # 条件概率表
                "marginals": {}
            }
            return distribution
        
        else:
            # 默认：混合模型
            distribution = {
                "type": "mixture_model",
                "components": [],
                "weights": []
            }
            return distribution
    
    def set_uniform_belief(self, state_variable: str, states: List[Any]) -> bool:
        """
        为状态变量设置均匀信念分布
        
        Args:
            state_variable: 状态变量名称
            states: 可能的状态值列表
            
        Returns:
            是否成功设置
        """
        try:
            if state_variable not in self.state_variables:
                self.state_variables.append(state_variable)
                self.state_dimension = len(self.state_variables)
            
            if self.representation == BeliefRepresentation.DISCRETE_DISTRIBUTION:
                n_states = len(states)
                uniform_prob = 1.0 / n_states if n_states > 0 else 0.0
                
                self.belief_distribution[state_variable] = {
                    "type": "discrete",
                    "states": states.copy(),
                    "probabilities": [uniform_prob] * n_states,
                    "last_updated": time.time()
                }
                
                logger.debug(f"为状态变量 {state_variable} 设置均匀分布，{n_states} 个状态")
                return True
            
            elif self.representation == BeliefRepresentation.GAUSSIAN:
                # 对于连续变量，均匀分布近似为大方差的高斯分布
                if all(isinstance(s, (int, float)) for s in states):
                    values = [float(s) for s in states]
                    mean_val = np.mean(values) if values else 0.0
                    var_val = np.var(values) if len(values) > 1 else 1.0
                    
                    self.belief_distribution[state_variable] = {
                        "type": "gaussian",
                        "mean": mean_val,
                        "variance": var_val,
                        "confidence": 0.5,
                        "last_updated": time.time()
                    }
                    
                    logger.debug(f"为状态变量 {state_variable} 设置高斯近似均匀分布")
                    return True
                else:
                    logger.error(f"状态变量 {state_variable} 的值不是数值类型，无法设置高斯分布")
                    return False
            
            else:
                logger.warning(f"表示方法 {self.representation.value} 不支持均匀分布设置")
                return False
                
        except Exception as e:
            logger.error(f"设置均匀信念分布失败: {e}")
            error_handler.handle_error(e, "BeliefState", "设置均匀信念分布失败")
            return False
    
    def update_belief(self, 
                     observation: Dict[str, Any],
                     observation_model: Dict[str, Any],
                     state_transition_model: Optional[Dict[str, Any]] = None) -> bool:
        """
        贝叶斯信念更新（预测+更新）
        
        Args:
            observation: 观测数据
            observation_model: 观测模型 P(O|S)
            state_transition_model: 状态转移模型 P(S_t|S_{t-1})（可选）
            
        Returns:
            是否成功更新
        """
        try:
            start_time = time.time()
            
            # 1. 预测步（如果有状态转移模型）
            if state_transition_model:
                self._predict_belief(state_transition_model)
            
            # 2. 更新步（贝叶斯更新）
            self._update_with_observation(observation, observation_model)
            
            # 3. 归一化
            self._normalize_belief()
            
            # 4. 更新元数据
            self.metadata["last_updated"] = datetime.now().isoformat()
            self.metadata["update_count"] += 1
            
            # 5. 记录历史
            if self.config.get('enable_history_tracking', True):
                self._record_belief_history()
            
            # 6. 更新性能统计
            update_time = time.time() - start_time
            self.performance_stats["updates_performed"] += 1
            self.performance_stats["average_update_time"] = (
                self.performance_stats["average_update_time"] * 
                (self.performance_stats["updates_performed"] - 1) + 
                update_time
            ) / self.performance_stats["updates_performed"]
            
            # 7. 计算不确定性度量
            self._compute_uncertainty_measures()
            
            logger.debug(f"信念更新完成，耗时: {update_time:.4f} 秒")
            return True
            
        except Exception as e:
            logger.error(f"信念更新失败: {e}")
            error_handler.handle_error(e, "BeliefState", "信念更新失败")
            return False
    
    def _predict_belief(self, state_transition_model: Dict[str, Any]) -> None:
        """
        预测步：基于状态转移模型预测下一时刻信念
        
        Args:
            state_transition_model: 状态转移模型
        """
        if self.representation == BeliefRepresentation.DISCRETE_DISTRIBUTION:
            self._predict_discrete(state_transition_model)
        elif self.representation == BeliefRepresentation.GAUSSIAN:
            self._predict_gaussian(state_transition_model)
        elif self.representation == BeliefRepresentation.PARTICLE_FILTER:
            self._predict_particle(state_transition_model)
        else:
            logger.warning(f"表示方法 {self.representation.value} 的预测步未实现")
        
        self.performance_stats["predictions_made"] += 1
    
    def _predict_discrete(self, state_transition_model: Dict[str, Any]) -> None:
        """离散分布的预测步"""
        # 简化实现：假设状态变量独立
        for var_name, var_belief in self.belief_distribution.items():
            if var_name in state_transition_model:
                transition_matrix = state_transition_model[var_name]
                
                if isinstance(transition_matrix, list) and len(transition_matrix) > 0:
                    # 转移矩阵：P(S_t | S_{t-1})
                    current_probs = var_belief.get("probabilities", [])
                    if len(current_probs) == len(transition_matrix):
                        # 矩阵乘法：新信念 = 转移矩阵^T * 当前信念
                        new_probs = np.dot(np.array(transition_matrix).T, np.array(current_probs))
                        var_belief["probabilities"] = new_probs.tolist()
                        var_belief["last_updated"] = time.time()
    
    def _predict_gaussian(self, state_transition_model: Dict[str, Any]) -> None:
        """高斯分布的预测步"""
        # 简化实现：线性高斯模型
        for var_name, var_belief in self.belief_distribution.items():
            if var_name in state_transition_model:
                transition_params = state_transition_model[var_name]
                
                if isinstance(transition_params, dict):
                    # 线性高斯：x_t = A * x_{t-1} + w, w ~ N(0, Q)
                    A = transition_params.get("A", 1.0)
                    Q = transition_params.get("Q", 0.1)
                    
                    mean = var_belief.get("mean", 0.0)
                    variance = var_belief.get("variance", 1.0)
                    
                    # 预测均值和方差
                    new_mean = A * mean
                    new_variance = A * A * variance + Q
                    
                    var_belief["mean"] = new_mean
                    var_belief["variance"] = new_variance
                    var_belief["last_updated"] = time.time()
    
    def _predict_particle(self, state_transition_model: Dict[str, Any]) -> None:
        """粒子滤波的预测步"""
        # 粒子滤波预测：根据动态模型传播粒子
        if "particles" in self.belief_distribution:
            particles = self.belief_distribution["particles"]
            weights = self.belief_distribution["weights"]
            
            # 简化：为每个粒子添加噪声
            for i in range(len(particles)):
                # 这里应该根据状态转移模型更新粒子状态
                # 简化实现：添加高斯噪声
                if isinstance(particles[i], dict):
                    for var_name in particles[i]:
                        if var_name in state_transition_model:
                            noise_std = state_transition_model[var_name].get("noise_std", 0.1)
                            particles[i][var_name] += np.random.normal(0, noise_std)
                elif isinstance(particles[i], (int, float)):
                    noise_std = state_transition_model.get("noise_std", 0.1)
                    particles[i] += np.random.normal(0, noise_std)
            
            self.belief_distribution["particles"] = particles
            self.belief_distribution["last_updated"] = time.time()
    
    def _update_with_observation(self, 
                                observation: Dict[str, Any], 
                                observation_model: Dict[str, Any]) -> None:
        """
        更新步：基于观测数据更新信念
        
        Args:
            observation: 观测数据
            observation_model: 观测模型
        """
        if self.representation == BeliefRepresentation.DISCRETE_DISTRIBUTION:
            self._update_discrete(observation, observation_model)
        elif self.representation == BeliefRepresentation.GAUSSIAN:
            self._update_gaussian(observation, observation_model)
        elif self.representation == BeliefRepresentation.PARTICLE_FILTER:
            self._update_particle(observation, observation_model)
        else:
            logger.warning(f"表示方法 {self.representation.value} 的更新步未实现")
    
    def _update_discrete(self, 
                        observation: Dict[str, Any], 
                        observation_model: Dict[str, Any]) -> None:
        """离散分布的更新步（贝叶斯更新）"""
        for var_name, var_belief in self.belief_distribution.items():
            if var_name in observation and var_name in observation_model:
                obs_value = observation[var_name]
                obs_model = observation_model[var_name]
                
                states = var_belief.get("states", [])
                probs = var_belief.get("probabilities", [])
                
                if isinstance(obs_model, dict) and "likelihood" in obs_model:
                    # 计算似然 P(O|S)
                    likelihood_func = obs_model["likelihood"]
                    
                    if callable(likelihood_func):
                        # 用户提供的似然函数
                        likelihoods = [likelihood_func(obs_value, state) for state in states]
                    else:
                        # 默认：高斯似然
                        likelihoods = []
                        for state in states:
                            if isinstance(state, (int, float)):
                                error = abs(obs_value - state)
                                likelihood = math.exp(-0.5 * error * error)
                                likelihoods.append(likelihood)
                            else:
                                likelihoods.append(1.0 if state == obs_value else 0.1)
                    
                    # 贝叶斯更新：后验 ∝ 似然 × 先验
                    min_prob = self.config.get('min_probability', 1e-10)
                    new_probs = []
                    for i in range(len(probs)):
                        posterior = max(likelihoods[i] * probs[i], min_prob)
                        new_probs.append(posterior)
                    
                    var_belief["probabilities"] = new_probs
                    var_belief["last_updated"] = time.time()
    
    def _update_gaussian(self, 
                        observation: Dict[str, Any], 
                        observation_model: Dict[str, Any]) -> None:
        """高斯分布的更新步（卡尔曼滤波更新）"""
        for var_name, var_belief in self.belief_distribution.items():
            if var_name in observation and var_name in observation_model:
                obs_value = observation[var_name]
                obs_model = observation_model[var_name]
                
                # 卡尔曼滤波更新
                prior_mean = var_belief.get("mean", 0.0)
                prior_var = var_belief.get("variance", 1.0)
                
                # 观测噪声方差
                obs_noise_var = obs_model.get("noise_variance", 0.1)
                
                # 卡尔曼增益
                kalman_gain = prior_var / (prior_var + obs_noise_var)
                
                # 更新均值和方差
                posterior_mean = prior_mean + kalman_gain * (obs_value - prior_mean)
                posterior_var = (1 - kalman_gain) * prior_var
                
                var_belief["mean"] = posterior_mean
                var_belief["variance"] = posterior_var
                var_belief["confidence"] = 1.0 / (1.0 + posterior_var)  # 简化置信度计算
                var_belief["last_updated"] = time.time()
    
    def _update_particle(self, 
                        observation: Dict[str, Any], 
                        observation_model: Dict[str, Any]) -> None:
        """粒子滤波的更新步（重要性重采样）"""
        if "particles" not in self.belief_distribution:
            return
        
        particles = self.belief_distribution["particles"]
        weights = self.belief_distribution.get("weights", [])
        
        if len(particles) == 0:
            return
        
        # 初始化权重（如果未设置）
        if len(weights) == 0:
            weights = [1.0 / len(particles)] * len(particles)
        
        # 计算重要性权重
        new_weights = []
        for i, particle in enumerate(particles):
            # 计算粒子权重：P(O|粒子)
            weight = self._compute_particle_weight(particle, observation, observation_model)
            new_weights.append(weight * weights[i])
        
        # 归一化权重
        weight_sum = sum(new_weights)
        if weight_sum > 0:
            new_weights = [w / weight_sum for w in new_weights]
        else:
            new_weights = [1.0 / len(particles)] * len(particles)
        
        # 检查有效样本大小
        effective_sample_size = 1.0 / sum(w * w for w in new_weights)
        
        # 如果需要，进行重采样
        resampling_threshold = self.config.get('resampling_threshold', 0.5)
        if effective_sample_size < resampling_threshold * len(particles):
            particles, new_weights = self._resample_particles(particles, new_weights)
        
        # 更新分布
        self.belief_distribution["particles"] = particles
        self.belief_distribution["weights"] = new_weights
        self.belief_distribution["effective_sample_size"] = effective_sample_size
        self.belief_distribution["last_updated"] = time.time()
    
    def _compute_particle_weight(self, 
                                particle: Any, 
                                observation: Dict[str, Any], 
                                observation_model: Dict[str, Any]) -> float:
        """计算粒子权重"""
        # 简化实现：假设观测噪声为高斯分布
        weight = 1.0
        
        if isinstance(particle, dict):
            for var_name, particle_value in particle.items():
                if var_name in observation and var_name in observation_model:
                    obs_value = observation[var_name]
                    obs_model = observation_model[var_name]
                    
                    if isinstance(obs_value, (int, float)) and isinstance(particle_value, (int, float)):
                        # 高斯似然
                        noise_std = obs_model.get("noise_std", 0.1)
                        error = obs_value - particle_value
                        likelihood = math.exp(-0.5 * (error / noise_std) ** 2) / (noise_std * math.sqrt(2 * math.pi))
                        weight *= likelihood
        elif isinstance(particle, (int, float)):
            # 单变量情况
            if len(observation) == 1:
                obs_value = list(observation.values())[0]
                if isinstance(obs_value, (int, float)):
                    noise_std = observation_model.get("noise_std", 0.1)
                    error = obs_value - particle
                    weight = math.exp(-0.5 * (error / noise_std) ** 2) / (noise_std * math.sqrt(2 * math.pi))
        
        return max(weight, self.config.get('min_probability', 1e-10))
    
    def _resample_particles(self, particles: List[Any], weights: List[float]) -> Tuple[List[Any], List[float]]:
        """重采样粒子"""
        n_particles = len(particles)
        
        # 系统重采样
        indices = []
        cumulative_weights = np.cumsum(weights)
        step = 1.0 / n_particles
        u = np.random.uniform(0, step)
        
        i = 0
        for j in range(n_particles):
            while u > cumulative_weights[i] and i < n_particles - 1:
                i += 1
            indices.append(i)
            u += step
        
        # 根据索引重采样粒子
        resampled_particles = [particles[i] for i in indices]
        resampled_weights = [1.0 / n_particles] * n_particles
        
        return resampled_particles, resampled_weights
    
    def _normalize_belief(self) -> None:
        """归一化信念分布"""
        if self.representation == BeliefRepresentation.DISCRETE_DISTRIBUTION:
            for var_name, var_belief in self.belief_distribution.items():
                if "probabilities" in var_belief:
                    probs = var_belief["probabilities"]
                    if probs:
                        prob_sum = sum(probs)
                        if prob_sum > 0:
                            var_belief["probabilities"] = [p / prob_sum for p in probs]
    
    def _record_belief_history(self) -> None:
        """记录信念历史"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "belief_snapshot": self._get_belief_snapshot(),
            "uncertainty_measures": self.uncertainty_measures.copy(),
            "metadata": self.metadata.copy()
        }
        
        self.belief_history.append(history_entry)
        
        # 限制历史大小
        if len(self.belief_history) > self.max_history_size:
            self.belief_history = self.belief_history[-self.max_history_size:]
    
    def _get_belief_snapshot(self) -> Dict[str, Any]:
        """获取信念快照"""
        snapshot = {}
        
        if self.representation == BeliefRepresentation.DISCRETE_DISTRIBUTION:
            for var_name, var_belief in self.belief_distribution.items():
                snapshot[var_name] = {
                    "type": "discrete",
                    "states": var_belief.get("states", []),
                    "probabilities": var_belief.get("probabilities", [])
                }
        
        elif self.representation == BeliefRepresentation.GAUSSIAN:
            for var_name, var_belief in self.belief_distribution.items():
                snapshot[var_name] = {
                    "type": "gaussian",
                    "mean": var_belief.get("mean", 0.0),
                    "variance": var_belief.get("variance", 1.0),
                    "confidence": var_belief.get("confidence", 0.5)
                }
        
        elif self.representation == BeliefRepresentation.PARTICLE_FILTER:
            snapshot = {
                "type": "particle_filter",
                "particle_count": len(self.belief_distribution.get("particles", [])),
                "effective_sample_size": self.belief_distribution.get("effective_sample_size", 0)
            }
        
        return snapshot
    
    def _compute_uncertainty_measures(self) -> None:
        """计算不确定性度量"""
        self.uncertainty_measures = {}
        
        if self.representation == BeliefRepresentation.DISCRETE_DISTRIBUTION:
            for var_name, var_belief in self.belief_distribution.items():
                if "probabilities" in var_belief:
                    probs = var_belief["probabilities"]
                    if probs:
                        # 计算熵：H = -Σ p_i log p_i
                        entropy = 0.0
                        for p in probs:
                            if p > 0:
                                entropy -= p * math.log(p)
                        
                        self.uncertainty_measures[f"{var_name}_entropy"] = entropy
                        
                        # 最大概率（置信度）
                        max_prob = max(probs) if probs else 0.0
                        self.uncertainty_measures[f"{var_name}_confidence"] = max_prob
        
        elif self.representation == BeliefRepresentation.GAUSSIAN:
            for var_name, var_belief in self.belief_distribution.items():
                variance = var_belief.get("variance", 1.0)
                self.uncertainty_measures[f"{var_name}_variance"] = variance
                self.uncertainty_measures[f"{var_name}_std"] = math.sqrt(variance)
        
        # 记录性能历史
        if "entropy" in str(self.uncertainty_measures):
            total_entropy = sum(v for k, v in self.uncertainty_measures.items() if "entropy" in k)
            self.performance_stats["belief_entropy_history"].append(total_entropy)
        
        if "confidence" in str(self.uncertainty_measures):
            avg_confidence = np.mean([v for k, v in self.uncertainty_measures.items() if "confidence" in k])
            self.performance_stats["belief_confidence_history"].append(avg_confidence)
    
    def get_most_likely_state(self, state_variable: Optional[str] = None) -> Dict[str, Any]:
        """
        获取最可能的状态
        
        Args:
            state_variable: 特定状态变量（可选）
            
        Returns:
            最可能的状态值
        """
        if state_variable:
            # 返回特定变量的最可能状态
            if state_variable in self.belief_distribution:
                var_belief = self.belief_distribution[state_variable]
                
                if self.representation == BeliefRepresentation.DISCRETE_DISTRIBUTION:
                    if "states" in var_belief and "probabilities" in var_belief:
                        states = var_belief["states"]
                        probs = var_belief["probabilities"]
                        
                        if states and probs and len(states) == len(probs):
                            max_idx = np.argmax(probs)
                            return {
                                state_variable: states[max_idx],
                                "probability": probs[max_idx],
                                "confidence": probs[max_idx]
                            }
                
                elif self.representation == BeliefRepresentation.GAUSSIAN:
                    mean = var_belief.get("mean", 0.0)
                    confidence = var_belief.get("confidence", 0.5)
                    return {
                        state_variable: mean,
                        "probability": confidence,  # 近似
                        "confidence": confidence
                    }
            
            return {state_variable: None, "probability": 0.0, "confidence": 0.0}
        
        else:
            # 返回所有变量的最可能状态
            result = {}
            for var_name in self.state_variables:
                var_result = self.get_most_likely_state(var_name)
                result.update(var_result)
            
            return result
    
    def get_belief_summary(self) -> Dict[str, Any]:
        """获取信念摘要"""
        summary = {
            "state_variables": self.state_variables.copy(),
            "representation": self.representation.value,
            "uncertainty_measures": self.uncertainty_measures.copy(),
            "most_likely_state": self.get_most_likely_state(),
            "performance_stats": {
                "updates_performed": self.performance_stats["updates_performed"],
                "average_update_time": self.performance_stats["average_update_time"],
                "belief_entropy_trend": self._get_entropy_trend(),
                "belief_confidence_trend": self._get_confidence_trend()
            },
            "metadata": self.metadata.copy()
        }
        
        return summary
    
    def _get_entropy_trend(self) -> str:
        """获取熵趋势"""
        history = self.performance_stats.get("belief_entropy_history", [])
        if len(history) < 2:
            return "stable"
        
        recent = history[-5:] if len(history) >= 5 else history
        if len(recent) < 2:
            return "stable"
        
        # 简单趋势分析
        first = np.mean(recent[:len(recent)//2])
        second = np.mean(recent[len(recent)//2:])
        
        if second < first * 0.9:
            return "decreasing"  # 不确定性降低
        elif second > first * 1.1:
            return "increasing"  # 不确定性增加
        else:
            return "stable"
    
    def _get_confidence_trend(self) -> str:
        """获取置信度趋势"""
        history = self.performance_stats.get("belief_confidence_history", [])
        if len(history) < 2:
            return "stable"
        
        recent = history[-5:] if len(history) >= 5 else history
        if len(recent) < 2:
            return "stable"
        
        # 简单趋势分析
        first = np.mean(recent[:len(recent)//2])
        second = np.mean(recent[len(recent)//2:])
        
        if second > first * 1.1:
            return "increasing"  # 置信度增加
        elif second < first * 0.9:
            return "decreasing"  # 置信度降低
        else:
            return "stable"
    
    def fuse_beliefs(self, other_belief_state: 'BeliefState', 
                    fusion_method: str = "bayesian_average") -> bool:
        """
        融合两个信念状态
        
        Args:
            other_belief_state: 另一个信念状态
            fusion_method: 融合方法
            
        Returns:
            是否成功融合
        """
        try:
            # 检查兼容性
            if self.representation != other_belief_state.representation:
                logger.error("信念表示方法不兼容，无法融合")
                return False
            
            if set(self.state_variables) != set(other_belief_state.state_variables):
                logger.warning("状态变量不完全相同，尝试部分融合")
            
            # 执行融合
            if fusion_method == "bayesian_average":
                self._fuse_bayesian_average(other_belief_state)
            elif fusion_method == "product_of_experts":
                self._fuse_product_of_experts(other_belief_state)
            elif fusion_method == "mixture_model":
                self._fuse_mixture_model(other_belief_state)
            else:
                logger.warning(f"融合方法 {fusion_method} 未实现，使用默认方法")
                self._fuse_bayesian_average(other_belief_state)
            
            # 归一化
            self._normalize_belief()
            
            # 更新元数据
            self.metadata["last_updated"] = datetime.now().isoformat()
            self.metadata["fusion_performed"] = True
            
            # 更新性能统计
            self.performance_stats["fusions_performed"] += 1
            
            logger.info(f"信念融合完成，方法: {fusion_method}")
            return True
            
        except Exception as e:
            logger.error(f"信念融合失败: {e}")
            error_handler.handle_error(e, "BeliefState", "信念融合失败")
            return False
    
    def _fuse_bayesian_average(self, other_belief_state: 'BeliefState') -> None:
        """贝叶斯平均融合"""
        # 简化实现：加权平均
        if self.representation == BeliefRepresentation.DISCRETE_DISTRIBUTION:
            for var_name in self.belief_distribution:
                if var_name in other_belief_state.belief_distribution:
                    self_probs = self.belief_distribution[var_name].get("probabilities", [])
                    other_probs = other_belief_state.belief_distribution[var_name].get("probabilities", [])
                    
                    if len(self_probs) == len(other_probs):
                        # 简单平均
                        fused_probs = [(p1 + p2) / 2 for p1, p2 in zip(self_probs, other_probs)]
                        self.belief_distribution[var_name]["probabilities"] = fused_probs
        
        elif self.representation == BeliefRepresentation.GAUSSIAN:
            for var_name in self.belief_distribution:
                if var_name in other_belief_state.belief_distribution:
                    self_mean = self.belief_distribution[var_name].get("mean", 0.0)
                    other_mean = other_belief_state.belief_distribution[var_name].get("mean", 0.0)
                    
                    self_var = self.belief_distribution[var_name].get("variance", 1.0)
                    other_var = other_belief_state.belief_distribution[var_name].get("variance", 1.0)
                    
                    # 逆方差加权平均
                    if self_var > 0 and other_var > 0:
                        weight_self = 1.0 / self_var
                        weight_other = 1.0 / other_var
                        total_weight = weight_self + weight_other
                        
                        fused_mean = (weight_self * self_mean + weight_other * other_mean) / total_weight
                        fused_var = 1.0 / total_weight
                        
                        self.belief_distribution[var_name]["mean"] = fused_mean
                        self.belief_distribution[var_name]["variance"] = fused_var
    
    def _fuse_product_of_experts(self, other_belief_state: 'BeliefState') -> None:
        """专家乘积融合"""
        # 简化实现：概率相乘后归一化
        if self.representation == BeliefRepresentation.DISCRETE_DISTRIBUTION:
            for var_name in self.belief_distribution:
                if var_name in other_belief_state.belief_distribution:
                    self_probs = self.belief_distribution[var_name].get("probabilities", [])
                    other_probs = other_belief_state.belief_distribution[var_name].get("probabilities", [])
                    
                    if len(self_probs) == len(other_probs):
                        # 乘积融合
                        fused_probs = [p1 * p2 for p1, p2 in zip(self_probs, other_probs)]
                        self.belief_distribution[var_name]["probabilities"] = fused_probs
    
    def _fuse_mixture_model(self, other_belief_state: 'BeliefState') -> None:
        """混合模型融合"""
        # 简化实现：创建混合分布
        logger.warning("混合模型融合方法未完全实现")


# 测试和示例函数
def create_example_belief_state() -> BeliefState:
    """创建示例信念状态"""
    # 定义状态变量
    state_variables = ["location", "temperature", "weather"]
    
    # 创建信念状态
    belief_state = BeliefState(
        state_variables=state_variables,
        representation=BeliefRepresentation.DISCRETE_DISTRIBUTION
    )
    
    # 设置初始信念（均匀分布）
    belief_state.set_uniform_belief("location", ["room_a", "room_b", "room_c"])
    belief_state.set_uniform_belief("temperature", ["cold", "warm", "hot"])
    belief_state.set_uniform_belief("weather", ["sunny", "cloudy", "rainy"])
    
    return belief_state


def test_belief_update() -> None:
    """测试信念更新"""
    print("测试贝叶斯信念更新...")
    
    # 创建信念状态
    belief_state = create_example_belief_state()
    
    # 创建观测模型
    observation_model = {
        "location": {
            "likelihood": lambda obs, state: 0.9 if obs == state else 0.1
        },
        "temperature": {
            "likelihood": lambda obs, state: 0.8 if obs == state else 0.2
        },
        "weather": {
            "likelihood": lambda obs, state: 0.7 if obs == state else 0.3
        }
    }
    
    # 创建观测数据
    observation = {
        "location": "room_a",
        "temperature": "warm",
        "weather": "sunny"
    }
    
    # 执行信念更新
    success = belief_state.update_belief(observation, observation_model)
    
    if success:
        print("信念更新成功!")
        
        # 获取最可能状态
        most_likely = belief_state.get_most_likely_state()
        print(f"最可能状态: {most_likely}")
        
        # 获取信念摘要
        summary = belief_state.get_belief_summary()
        print(f"信念摘要:")
        print(f"  状态变量: {summary['state_variables']}")
        print(f"  不确定性度量: {summary['uncertainty_measures']}")
        print(f"  更新次数: {summary['performance_stats']['updates_performed']}")
    else:
        print("信念更新失败!")


if __name__ == "__main__":
    test_belief_update()