#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
部分可观测性处理器 - 处理部分可观测环境下的状态估计问题

核心功能:
1. 观测模型管理与维护
2. 信念状态更新（贝叶斯滤波）
3. 信息增益计算
4. 主动感知策略
5. 多传感器融合

技术实现:
- 部分可观测马尔可夫决策过程 (POMDP) 框架
- 贝叶斯滤波算法（卡尔曼滤波、粒子滤波）
- 信息论度量（熵、互信息）
- 传感器模型与不确定性建模
- 多模态信息融合

POMDP 基本组件:
- 状态空间 S
- 动作空间 A
- 观测空间 O
- 状态转移模型 T(s'|s,a)
- 观测模型 Z(o|s',a)
- 奖励函数 R(s,a)

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import networkx as nx
from datetime import datetime

# 导入相关模块
from core.error_handling import ErrorHandler
from core.world_model.belief_state import BeliefState
from core.world_model.state_transition_model import StateTransitionModel

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()


class ObservationType(Enum):
    """观测类型枚举"""
    DIRECT = "direct"                # 直接观测
    NOISY = "noisy"                  # 带噪声观测
    PARTIAL = "partial"              # 部分状态观测
    CORRELATED = "correlated"        # 相关性观测
    MULTIMODAL = "multimodal"        # 多模态观测


class FilterType(Enum):
    """滤波器类型枚举"""
    KALMAN_FILTER = "kalman_filter"          # 卡尔曼滤波
    EXTENDED_KALMAN_FILTER = "extended_kalman_filter"  # 扩展卡尔曼滤波
    PARTICLE_FILTER = "particle_filter"      # 粒子滤波
    BAYESIAN_FILTER = "bayesian_filter"      # 贝叶斯滤波
    HISTOGRAM_FILTER = "histogram_filter"    # 直方图滤波


class InformationGainMetric(Enum):
    """信息增益度量枚举"""
    ENTROPY_REDUCTION = "entropy_reduction"      # 熵减少
    MUTUAL_INFORMATION = "mutual_information"    # 互信息
    KL_DIVERGENCE = "kl_divergence"              # KL散度
    FISHER_INFORMATION = "fisher_information"    # 费雪信息
    VARIANCE_REDUCTION = "variance_reduction"    # 方差减少


class PartialObservabilityHandler:
    """
    部分可观测性处理器 - 处理部分可观测环境下的状态估计
    
    核心特性:
    1. 观测模型管理
    2. 信念状态维护
    3. 贝叶斯滤波实现
    4. 信息增益计算
    5. 主动感知策略
    
    部分可观测问题:
    - 环境状态不完全可观测
    - 观测带有噪声
    - 观测可能只反映部分状态
    - 不同传感器的观测可能冲突
    
    解决方案:
    - 贝叶斯滤波：维护状态的概率信念
    - 信息论：量化观测的信息价值
    - 传感器融合：整合多源信息
    - 主动感知：选择信息丰富的动作
    
    贝叶斯滤波公式:
    bel(s_t) = η * P(o_t|s_t) * ∫ P(s_t|s_{t-1}, a_{t-1}) * bel(s_{t-1}) ds_{t-1}
    
    其中:
    - bel(s_t): 时刻t的信念状态
    - P(o_t|s_t): 观测模型
    - P(s_t|s_{t-1}, a_{t-1}): 状态转移模型
    - η: 归一化常数
    """
    
    def __init__(self, 
                 state_variables: Optional[List[str]] = None,
                 observation_models: Optional[Dict[str, Any]] = None,
                 filter_type: Union[FilterType, str] = FilterType.BAYESIAN_FILTER,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化部分可观测性处理器
        
        Args:
            state_variables: 状态变量列表
            observation_models: 观测模型字典
            filter_type: 滤波器类型
            config: 配置参数
        """
        # 配置参数
        self.config = config or self._get_default_config()
        
        # 状态变量
        self.state_variables = state_variables or []
        
        # 观测模型
        self.observation_models = observation_models or {}
        self._initialize_observation_models()
        
        # 滤波器类型
        if isinstance(filter_type, str):
            self.filter_type = FilterType(filter_type)
        else:
            self.filter_type = filter_type
        
        # 信念状态
        self.belief_state = BeliefState(
            state_variables=self.state_variables,
            representation="discrete_distribution",  # 默认离散表示
            config=self.config.get('belief_config', {})
        )
        
        # 状态转移模型（可选）
        self.transition_model: Optional[StateTransitionModel] = None
        
        # 滤波器实现
        self.filter_implementation = self._initialize_filter()
        
        # 信息增益计算器
        self.information_calculator = self._initialize_information_calculator()
        
        # 历史记录
        self.observation_history: List[Dict[str, Any]] = []
        self.belief_history: List[Dict[str, Any]] = []
        self.max_history_size = self.config.get('max_history_size', 1000)
        
        # 元数据
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "filter_type": self.filter_type.value,
            "state_variables": self.state_variables.copy(),
            "observation_model_count": len(self.observation_models)
        }
        
        # 性能统计
        self.performance_stats = {
            "observations_processed": 0,
            "belief_updates_performed": 0,
            "average_update_time": 0.0,
            "average_information_gain": 0.0,
            "belief_uncertainty_history": [],
            "observation_quality_history": []
        }
        
        logger.info(f"部分可观测性处理器初始化完成，滤波器类型: {self.filter_type.value}")
        logger.info(f"状态变量: {self.state_variables}")
        logger.info(f"观测模型数量: {len(self.observation_models)}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'max_history_size': 1000,
            'belief_update_threshold': 0.01,
            'min_observation_quality': 0.1,
            'information_gain_threshold': 0.05,
            'active_perception_enabled': True,
            'sensor_fusion_method': 'bayesian',
            'kalman_filter_process_noise': 0.1,
            'kalman_filter_measurement_noise': 0.1,
            'particle_filter_particle_count': 1000,
            'particle_filter_resampling_threshold': 0.5,
            'belief_config': {
                'max_history_size': 500,
                'uncertainty_threshold': 0.3
            }
        }
    
    def _initialize_observation_models(self) -> None:
        """初始化观测模型"""
        if not self.observation_models:
            # 如果没有提供观测模型，创建默认模型
            for var in self.state_variables:
                self.observation_models[var] = {
                    "type": "gaussian",
                    "noise_std": 0.1,
                    "observability": 1.0,  # 完全可观测
                    "reliability": 0.9,
                    "last_calibrated": time.time()
                }
    
    def _initialize_filter(self) -> Dict[str, Any]:
        """初始化滤波器"""
        filter_impl = {
            "type": self.filter_type.value,
            "initialized": False,
            "parameters": {},
            "state": {}
        }
        
        if self.filter_type == FilterType.KALMAN_FILTER:
            filter_impl["parameters"] = {
                "process_noise": self.config.get('kalman_filter_process_noise', 0.1),
                "measurement_noise": self.config.get('kalman_filter_measurement_noise', 0.1),
                "state_dimension": len(self.state_variables),
                "measurement_dimension": len(self.state_variables)  # 假设全观测
            }
            
            # 卡尔曼滤波器状态
            filter_impl["state"] = {
                "x": np.zeros(len(self.state_variables)),  # 状态估计
                "P": np.eye(len(self.state_variables)),    # 协方差矩阵
                "F": np.eye(len(self.state_variables)),    # 状态转移矩阵
                "H": np.eye(len(self.state_variables)),    # 观测矩阵
                "Q": self.config.get('kalman_filter_process_noise', 0.1) * np.eye(len(self.state_variables)),
                "R": self.config.get('kalman_filter_measurement_noise', 0.1) * np.eye(len(self.state_variables))
            }
        
        elif self.filter_type == FilterType.PARTICLE_FILTER:
            particle_count = self.config.get('particle_filter_particle_count', 1000)
            
            filter_impl["parameters"] = {
                "particle_count": particle_count,
                "resampling_threshold": self.config.get('particle_filter_resampling_threshold', 0.5),
                "state_dimension": len(self.state_variables)
            }
            
            # 粒子滤波器状态
            filter_impl["state"] = {
                "particles": np.random.randn(particle_count, len(self.state_variables)),
                "weights": np.ones(particle_count) / particle_count,
                "effective_sample_size": particle_count
            }
        
        elif self.filter_type == FilterType.HISTOGRAM_FILTER:
            # 直方图滤波器：为每个状态变量创建直方图
            histograms = {}
            for var in self.state_variables:
                # 默认：10个bin的直方图
                histograms[var] = {
                    "bins": 10,
                    "range": (0.0, 1.0),  # 默认范围
                    "probabilities": np.ones(10) / 10,
                    "bin_centers": np.linspace(0.0, 1.0, 10)
                }
            
            filter_impl["parameters"] = {
                "histograms": histograms,
                "smoothing": 0.01
            }
        
        return filter_impl
    
    def _initialize_information_calculator(self) -> Dict[str, Any]:
        """初始化信息增益计算器"""
        return {
            "metrics_enabled": {
                InformationGainMetric.ENTROPY_REDUCTION.value: True,
                InformationGainMetric.MUTUAL_INFORMATION.value: True,
                InformationGainMetric.KL_DIVERGENCE.value: False,
                InformationGainMetric.VARIANCE_REDUCTION.value: True
            },
            "last_calculations": {},
            "cumulative_gain": 0.0,
            "calculation_count": 0
        }
    
    def set_transition_model(self, transition_model: StateTransitionModel) -> bool:
        """
        设置状态转移模型
        
        Args:
            transition_model: 状态转移模型
            
        Returns:
            是否成功设置
        """
        try:
            self.transition_model = transition_model
            logger.info(f"设置状态转移模型: {transition_model.transition_type.value}")
            return True
        except Exception as e:
            logger.error(f"设置状态转移模型失败: {e}")
            return False
    
    def add_observation_model(self, 
                             state_variable: str,
                             model_type: Union[str, ObservationType],
                             model_parameters: Dict[str, Any]) -> bool:
        """
        添加观测模型
        
        Args:
            state_variable: 状态变量名称
            model_type: 模型类型
            model_parameters: 模型参数
            
        Returns:
            是否成功添加
        """
        try:
            if isinstance(model_type, str):
                model_type_enum = ObservationType(model_type)
            else:
                model_type_enum = model_type
            
            self.observation_models[state_variable] = {
                "type": model_type_enum.value,
                **model_parameters,
                "last_updated": time.time()
            }
            
            logger.info(f"为状态变量 {state_variable} 添加观测模型: {model_type_enum.value}")
            return True
            
        except Exception as e:
            logger.error(f"添加观测模型失败: {e}")
            error_handler.handle_error(e, "PartialObservabilityHandler", "添加观测模型失败")
            return False
    
    def process_observation(self, 
                           observation: Dict[str, Any],
                           action: Optional[str] = None,
                           update_belief: bool = True) -> Dict[str, Any]:
        """
        处理观测数据
        
        Args:
            observation: 观测数据
            action: 导致观测的动作（可选）
            update_belief: 是否更新信念状态
            
        Returns:
            处理结果
        """
        try:
            start_time = time.time()
            
            # 记录观测
            self._record_observation(observation, action)
            
            # 验证观测数据
            observation_quality = self._validate_observation(observation)
            
            # 预处理观测数据
            processed_observation = self._preprocess_observation(observation)
            
            # 如果需要，更新信念状态
            if update_belief:
                belief_update_result = self.update_belief(processed_observation, action)
            else:
                belief_update_result = {"success": True, "skipped": True}
            
            # 计算信息增益
            information_gain = self._calculate_information_gain(observation)
            
            # 准备结果
            result = {
                "success": True,
                "observation_quality": observation_quality,
                "processed_observation": processed_observation,
                "belief_updated": update_belief,
                "information_gain": information_gain,
                "processing_time": time.time() - start_time
            }
            
            # 合并信念更新结果
            if update_belief:
                result.update(belief_update_result)
            
            # 更新性能统计
            self.performance_stats["observations_processed"] += 1
            
            logger.debug(f"观测处理完成，质量: {observation_quality:.4f}, 信息增益: {information_gain:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"观测处理失败: {e}")
            error_handler.handle_error(e, "PartialObservabilityHandler", "观测处理失败")
            
            return {
                "success": False,
                "error": str(e),
                "observation_quality": 0.0,
                "information_gain": 0.0
            }
    
    def _record_observation(self, observation: Dict[str, Any], action: Optional[str]) -> None:
        """记录观测历史"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "observation": observation.copy(),
            "action": action,
            "metadata": {
                "observation_model_count": len(self.observation_models),
                "state_variables": self.state_variables.copy()
            }
        }
        
        self.observation_history.append(history_entry)
        
        # 限制历史大小
        if len(self.observation_history) > self.max_history_size:
            self.observation_history = self.observation_history[-self.max_history_size:]
    
    def _validate_observation(self, observation: Dict[str, Any]) -> float:
        """验证观测数据质量"""
        quality_score = 0.0
        valid_variables = 0
        
        for var_name, obs_value in observation.items():
            if var_name in self.observation_models:
                valid_variables += 1
                
                # 检查观测值是否在合理范围内
                model = self.observation_models[var_name]
                model_type = model.get("type", "gaussian")
                
                if model_type == "gaussian":
                    # 高斯模型：检查值是否在均值±3σ范围内
                    mean = model.get("mean", 0.0)
                    std = model.get("noise_std", 1.0)
                    
                    if isinstance(obs_value, (int, float)):
                        z_score = abs(obs_value - mean) / std if std > 0 else 0.0
                        if z_score <= 3.0:
                            quality_score += 1.0 - min(z_score / 3.0, 1.0)
                
                elif model_type == "discrete":
                    # 离散模型：检查值是否在允许的取值范围内
                    allowed_values = model.get("allowed_values", [])
                    if allowed_values:
                        if obs_value in allowed_values:
                            quality_score += 1.0
                    else:
                        quality_score += 1.0  # 没有限制，假设有效
                
                else:
                    # 其他模型：默认有效
                    quality_score += 1.0
        
        # 计算平均质量
        if valid_variables > 0:
            quality = quality_score / valid_variables
        else:
            quality = 0.0
        
        # 记录质量历史
        self.performance_stats["observation_quality_history"].append(quality)
        
        return quality
    
    def _preprocess_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """预处理观测数据"""
        processed = observation.copy()
        
        for var_name, obs_value in observation.items():
            if var_name in self.observation_models:
                model = self.observation_models[var_name]
                model_type = model.get("type", "gaussian")
                
                if model_type == "noisy":
                    # 降噪处理（简化）
                    noise_level = model.get("noise_level", 0.1)
                    if isinstance(obs_value, (int, float)):
                        # 简单移动平均（需要历史数据）
                        processed[var_name] = obs_value
                
                elif model_type == "partial":
                    # 部分观测：可能需要推断缺失部分
                    observability = model.get("observability", 1.0)
                    if observability < 1.0:
                        # 标记为部分观测
                        processed[f"{var_name}_partial"] = True
                        processed[f"{var_name}_completeness"] = observability
        
        return processed
    
    def update_belief(self, 
                     observation: Dict[str, Any],
                     action: Optional[str] = None) -> Dict[str, Any]:
        """
        更新信念状态（贝叶斯滤波）
        
        Args:
            observation: 观测数据
            action: 导致观测的动作（可选）
            
        Returns:
            更新结果
        """
        try:
            start_time = time.time()
            
            # 根据滤波器类型执行更新
            if self.filter_type == FilterType.KALMAN_FILTER:
                result = self._update_with_kalman_filter(observation, action)
            elif self.filter_type == FilterType.PARTICLE_FILTER:
                result = self._update_with_particle_filter(observation, action)
            elif self.filter_type == FilterType.HISTOGRAM_FILTER:
                result = self._update_with_histogram_filter(observation, action)
            elif self.filter_type == FilterType.BAYESIAN_FILTER:
                result = self._update_with_bayesian_filter(observation, action)
            else:
                # 默认：使用信念状态的贝叶斯更新
                result = self._update_with_belief_state(observation, action)
            
            # 记录信念历史
            self._record_belief_history()
            
            # 更新性能统计
            update_time = time.time() - start_time
            self.performance_stats["belief_updates_performed"] += 1
            self.performance_stats["average_update_time"] = (
                self.performance_stats["average_update_time"] * 
                (self.performance_stats["belief_updates_performed"] - 1) + 
                update_time
            ) / self.performance_stats["belief_updates_performed"]
            
            # 计算信念不确定性
            uncertainty = self._calculate_belief_uncertainty()
            self.performance_stats["belief_uncertainty_history"].append(uncertainty)
            
            # 更新元数据
            self.metadata["last_updated"] = datetime.now().isoformat()
            
            result.update({
                "update_time": update_time,
                "belief_uncertainty": uncertainty,
                "success": True
            })
            
            logger.debug(f"信念更新完成，不确定性: {uncertainty:.4f}, 耗时: {update_time:.4f} 秒")
            return result
            
        except Exception as e:
            logger.error(f"信念更新失败: {e}")
            error_handler.handle_error(e, "PartialObservabilityHandler", "信念更新失败")
            
            return {
                "success": False,
                "error": str(e),
                "belief_uncertainty": 1.0  # 最大不确定性
            }
    
    def _update_with_kalman_filter(self, 
                                  observation: Dict[str, Any],
                                  action: Optional[str] = None) -> Dict[str, Any]:
        """使用卡尔曼滤波器更新"""
        # 简化实现：更新信念状态
        if self.transition_model:
            # 预测步
            current_state = self.belief_state.get_most_likely_state()
            predicted_state = self.transition_model.predict_next_state(current_state, action)
            
            # 更新步（简化）
            for var_name, obs_value in observation.items():
                if var_name in self.observation_models:
                    model = self.observation_models[var_name]
                    
                    if model["type"] == "gaussian":
                        # 卡尔曼增益计算（简化）
                        prior_var = 1.0  # 简化
                        obs_noise_var = model.get("noise_std", 0.1) ** 2
                        kalman_gain = prior_var / (prior_var + obs_noise_var)
                        
                        # 更新信念（简化）
                        predicted_value = predicted_state.get(var_name)
                        if predicted_value is not None and isinstance(predicted_value, (int, float)):
                            updated_value = predicted_value + kalman_gain * (obs_value - predicted_value)
                            
                            # 更新信念状态（简化）
                            self.belief_state.set_uniform_belief(var_name, [updated_value])
        
        return {"filter_type": "kalman", "simplified": True}
    
    def _update_with_particle_filter(self, 
                                    observation: Dict[str, Any],
                                    action: Optional[str] = None) -> Dict[str, Any]:
        """使用粒子滤波器更新"""
        # 简化实现
        return {"filter_type": "particle", "simplified": True}
    
    def _update_with_histogram_filter(self, 
                                     observation: Dict[str, Any],
                                     action: Optional[str] = None) -> Dict[str, Any]:
        """使用直方图滤波器更新"""
        # 简化实现
        return {"filter_type": "histogram", "simplified": True}
    
    def _update_with_bayesian_filter(self, 
                                    observation: Dict[str, Any],
                                    action: Optional[str] = None) -> Dict[str, Any]:
        """使用贝叶斯滤波器更新"""
        # 这是我们的默认方法
        return self._update_with_belief_state(observation, action)
    
    def _update_with_belief_state(self, 
                                 observation: Dict[str, Any],
                                 action: Optional[str] = None) -> Dict[str, Any]:
        """使用信念状态更新（贝叶斯更新）"""
        # 创建观测模型
        observation_model = {}
        for var_name, obs_value in observation.items():
            if var_name in self.observation_models:
                model = self.observation_models[var_name]
                model_type = model.get("type", "gaussian")
                
                if model_type == "gaussian":
                    # 高斯似然函数
                    noise_std = model.get("noise_std", 0.1)
                    observation_model[var_name] = {
                        "likelihood": lambda obs, state, ns=noise_std: 
                            math.exp(-0.5 * ((obs - state) / ns) ** 2) / (ns * math.sqrt(2 * math.pi))
                    }
                else:
                    # 默认似然函数
                    observation_model[var_name] = {
                        "likelihood": lambda obs, state: 0.9 if obs == state else 0.1
                    }
        
        # 状态转移模型（如果有）
        state_transition_model = None
        if self.transition_model and action:
            # 简化：获取转移模型参数
            state_transition_model = {}
        
        # 更新信念状态
        success = self.belief_state.update_belief(
            observation=observation,
            observation_model=observation_model,
            state_transition_model=state_transition_model
        )
        
        return {
            "filter_type": "bayesian",
            "belief_update_success": success,
            "belief_summary": self.belief_state.get_belief_summary()
        }
    
    def _record_belief_history(self) -> None:
        """记录信念历史"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "belief_snapshot": self.belief_state._get_belief_snapshot(),
            "uncertainty": self._calculate_belief_uncertainty(),
            "metadata": self.metadata.copy()
        }
        
        self.belief_history.append(history_entry)
        
        # 限制历史大小
        if len(self.belief_history) > self.max_history_size:
            self.belief_history = self.belief_history[-self.max_history_size:]
    
    def _calculate_belief_uncertainty(self) -> float:
        """计算信念不确定性"""
        uncertainty = 0.0
        
        # 使用信念状态的不确定性度量
        uncertainty_measures = self.belief_state.uncertainty_measures
        
        if uncertainty_measures:
            # 计算平均不确定性（标准化）
            entropy_values = [v for k, v in uncertainty_measures.items() if "entropy" in k]
            if entropy_values:
                # 标准化熵值：最大熵 = log(n)，n为状态数
                max_entropy = math.log(10)  # 假设10个状态
                normalized_entropy = np.mean(entropy_values) / max_entropy if max_entropy > 0 else 0.0
                uncertainty = normalized_entropy
            else:
                # 使用其他不确定性度量
                var_values = [v for k, v in uncertainty_measures.items() if "variance" in k]
                if var_values:
                    # 标准化方差
                    max_var = 1.0  # 假设最大方差为1
                    normalized_var = np.mean(var_values) / max_var if max_var > 0 else 0.0
                    uncertainty = normalized_var
                else:
                    # 默认不确定性
                    uncertainty = 0.5
        
        return uncertainty
    
    def _calculate_information_gain(self, observation: Dict[str, Any]) -> float:
        """
        计算观测的信息增益
        
        Args:
            observation: 观测数据
            
        Returns:
            信息增益值
        """
        try:
            total_gain = 0.0
            valid_variables = 0
            
            for var_name, obs_value in observation.items():
                if var_name in self.observation_models:
                    # 计算单个变量的信息增益
                    var_gain = self._calculate_variable_information_gain(var_name, obs_value)
                    total_gain += var_gain
                    valid_variables += 1
            
            # 计算平均信息增益
            if valid_variables > 0:
                average_gain = total_gain / valid_variables
            else:
                average_gain = 0.0
            
            # 更新信息增益统计
            self.information_calculator["cumulative_gain"] += average_gain
            self.information_calculator["calculation_count"] += 1
            self.performance_stats["average_information_gain"] = (
                self.information_calculator["cumulative_gain"] / 
                self.information_calculator["calculation_count"]
            )
            
            return average_gain
            
        except Exception as e:
            logger.warning(f"计算信息增益失败: {e}")
            return 0.0
    
    def _calculate_variable_information_gain(self, var_name: str, obs_value: Any) -> float:
        """计算单个变量的信息增益"""
        gain = 0.0
        
        # 获取当前信念
        current_belief = self.belief_state.belief_distribution.get(var_name)
        if not current_belief:
            return 0.0
        
        # 根据信念表示类型计算信息增益
        if current_belief.get("type") == "discrete":
            # 离散分布：使用熵减少
            probs = current_belief.get("probabilities", [])
            if probs:
                # 当前熵
                current_entropy = 0.0
                for p in probs:
                    if p > 0:
                        current_entropy -= p * math.log(p)
                
                # 观测后的信念（简化：观测使对应状态概率增加）
                states = current_belief.get("states", [])
                if obs_value in states:
                    idx = states.index(obs_value)
                    
                    # 模拟观测后的概率分布
                    new_probs = probs.copy()
                    obs_strength = 0.7  # 观测强度
                    new_probs[idx] = probs[idx] * (1.0 + obs_strength)
                    
                    # 归一化
                    total = sum(new_probs)
                    if total > 0:
                        new_probs = [p / total for p in new_probs]
                        
                        # 计算新熵
                        new_entropy = 0.0
                        for p in new_probs:
                            if p > 0:
                                new_entropy -= p * math.log(p)
                        
                        # 信息增益 = 熵减少
                        gain = max(current_entropy - new_entropy, 0.0)
        
        elif current_belief.get("type") == "gaussian":
            # 高斯分布：使用方差减少
            variance = current_belief.get("variance", 1.0)
            
            # 观测后的方差（卡尔曼滤波更新）
            obs_noise_var = self.observation_models.get(var_name, {}).get("noise_std", 0.1) ** 2
            
            if variance > 0 and obs_noise_var > 0:
                # 卡尔曼增益
                K = variance / (variance + obs_noise_var)
                
                # 更新后的方差
                new_variance = (1 - K) * variance
                
                # 信息增益 = 方差减少百分比
                gain = (variance - new_variance) / variance if variance > 0 else 0.0
        
        return gain
    
    def get_expected_information_gain(self, 
                                     potential_observation: Dict[str, Any],
                                     current_belief: Optional[Dict[str, Any]] = None) -> float:
        """
        获取预期信息增益
        
        Args:
            potential_observation: 潜在观测
            current_belief: 当前信念（可选）
            
        Returns:
            预期信息增益
        """
        try:
            expected_gain = 0.0
            variable_count = 0
            
            for var_name, potential_value in potential_observation.items():
                if var_name in self.observation_models:
                    # 计算该变量的预期信息增益
                    var_gain = self._estimate_variable_information_gain(var_name, potential_value)
                    expected_gain += var_gain
                    variable_count += 1
            
            if variable_count > 0:
                return expected_gain / variable_count
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"计算预期信息增益失败: {e}")
            return 0.0
    
    def _estimate_variable_information_gain(self, var_name: str, potential_value: Any) -> float:
        """估计变量的信息增益"""
        # 简化估计：基于观测质量和当前不确定性
        model = self.observation_models.get(var_name, {})
        observability = model.get("observability", 1.0)
        reliability = model.get("reliability", 0.9)
        
        # 获取当前不确定性
        uncertainty = 0.0
        current_belief = self.belief_state.belief_distribution.get(var_name)
        if current_belief:
            if current_belief.get("type") == "discrete":
                probs = current_belief.get("probabilities", [])
                if probs:
                    entropy = 0.0
                    for p in probs:
                        if p > 0:
                            entropy -= p * math.log(p)
                    max_entropy = math.log(len(probs)) if probs else 1.0
                    uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # 预期信息增益 = 不确定性 * 观测质量
        observation_quality = observability * reliability
        expected_gain = uncertainty * observation_quality
        
        return expected_gain
    
    def recommend_observation_action(self, 
                                    possible_actions: List[str],
                                    current_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        推荐观测动作（主动感知）
        
        Args:
            possible_actions: 可能的动作列表
            current_state: 当前状态（可选）
            
        Returns:
            推荐动作和理由
        """
        try:
            if not self.config.get('active_perception_enabled', True):
                return {
                    "recommended_action": None,
                    "reason": "active_perception_disabled",
                    "information_gain": 0.0
                }
            
            # 如果没有动作，返回None
            if not possible_actions:
                return {
                    "recommended_action": None,
                    "reason": "no_possible_actions",
                    "information_gain": 0.0
                }
            
            # 计算每个动作的预期信息增益
            action_gains = {}
            for action in possible_actions:
                # 预测该动作可能产生的观测
                potential_observation = self._predict_observation_for_action(action, current_state)
                
                # 计算预期信息增益
                expected_gain = self.get_expected_information_gain(potential_observation)
                
                action_gains[action] = expected_gain
            
            # 选择信息增益最大的动作
            if action_gains:
                best_action = max(action_gains.items(), key=lambda x: x[1])
                best_action_name, best_gain = best_action
                
                return {
                    "recommended_action": best_action_name,
                    "information_gain": best_gain,
                    "all_action_gains": action_gains,
                    "reason": "max_information_gain"
                }
            else:
                return {
                    "recommended_action": possible_actions[0] if possible_actions else None,
                    "information_gain": 0.0,
                    "reason": "no_gain_calculated"
                }
                
        except Exception as e:
            logger.error(f"推荐观测动作失败: {e}")
            return {
                "recommended_action": possible_actions[0] if possible_actions else None,
                "information_gain": 0.0,
                "reason": f"error: {str(e)}"
            }
    
    def _predict_observation_for_action(self, 
                                       action: str,
                                       current_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """预测动作可能产生的观测"""
        # 简化实现：基于动作类型预测观测
        predicted_observation = {}
        
        # 根据动作类型预测可能观测的变量
        if "observe" in action.lower() or "sense" in action.lower():
            # 观测类动作：可能观测所有变量
            for var_name in self.state_variables:
                model = self.observation_models.get(var_name, {})
                observability = model.get("observability", 1.0)
                
                # 有一定概率观测到该变量
                if np.random.random() < observability:
                    # 使用当前信念的最可能值作为预测观测
                    most_likely = self.belief_state.get_most_likely_state(var_name)
                    if var_name in most_likely:
                        predicted_observation[var_name] = most_likely[var_name]
        
        elif "move" in action.lower() or "go" in action.lower():
            # 移动类动作：可能观测位置相关变量
            location_vars = [var for var in self.state_variables if "location" in var.lower() or "position" in var.lower()]
            for var_name in location_vars:
                predicted_observation[var_name] = "unknown_location"  # 简化
        
        else:
            # 其他动作：可能观测相关变量
            for var_name in self.state_variables:
                if np.random.random() < 0.3:  # 30%概率观测
                    predicted_observation[var_name] = "unknown"
        
        return predicted_observation
    
    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统摘要"""
        summary = {
            "filter_type": self.filter_type.value,
            "state_variables": self.state_variables.copy(),
            "observation_models": len(self.observation_models),
            "belief_uncertainty": self._calculate_belief_uncertainty(),
            "performance_stats": self.performance_stats.copy(),
            "information_gain_stats": {
                "cumulative_gain": self.information_calculator["cumulative_gain"],
                "average_gain": self.performance_stats["average_information_gain"],
                "calculation_count": self.information_calculator["calculation_count"]
            },
            "metadata": self.metadata.copy()
        }
        
        # 添加信念状态摘要
        if self.belief_state:
            summary["belief_summary"] = self.belief_state.get_belief_summary()
        
        # 添加转移模型信息
        if self.transition_model:
            summary["transition_model"] = {
                "type": self.transition_model.transition_type.value,
                "state_variables": self.transition_model.state_variables
            }
        
        return summary


# 测试和示例函数
def create_example_partial_observability_handler() -> PartialObservabilityHandler:
    """创建示例部分可观测性处理器"""
    # 定义状态变量
    state_variables = ["robot_position", "battery_level", "target_location"]
    
    # 定义观测模型
    observation_models = {
        "robot_position": {
            "type": "gaussian",
            "noise_std": 0.5,
            "observability": 0.9,
            "reliability": 0.8,
            "mean": 0.0  # 参考均值
        },
        "battery_level": {
            "type": "gaussian",
            "noise_std": 5.0,  # 百分比噪声
            "observability": 1.0,
            "reliability": 0.95
        },
        "target_location": {
            "type": "partial",
            "observability": 0.6,  # 部分可观测
            "reliability": 0.7,
            "allowed_values": ["room_a", "room_b", "room_c", "unknown"]
        }
    }
    
    # 创建处理器
    handler = PartialObservabilityHandler(
        state_variables=state_variables,
        observation_models=observation_models,
        filter_type=FilterType.BAYESIAN_FILTER
    )
    
    # 设置初始信念
    handler.belief_state.set_uniform_belief("robot_position", [0.0, 1.0, 2.0, 3.0, 4.0])
    handler.belief_state.set_uniform_belief("battery_level", [20.0, 40.0, 60.0, 80.0, 100.0])
    handler.belief_state.set_uniform_belief("target_location", ["room_a", "room_b", "room_c"])
    
    return handler


def test_partial_observability() -> None:
    """测试部分可观测性处理"""
    print("测试部分可观测性处理...")
    
    # 创建处理器
    handler = create_example_partial_observability_handler()
    
    # 创建观测数据
    observation = {
        "robot_position": 2.5,
        "battery_level": 75.0,
        "target_location": "room_b"
    }
    
    # 处理观测
    result = handler.process_observation(observation, action="observe_environment")
    
    print(f"观测处理结果:")
    print(f"  成功: {result['success']}")
    print(f"  观测质量: {result['observation_quality']:.4f}")
    print(f"  信息增益: {result['information_gain']:.4f}")
    print(f"  处理时间: {result['processing_time']:.4f} 秒")
    
    # 获取系统摘要
    summary = handler.get_system_summary()
    print(f"\n系统摘要:")
    print(f"  滤波器类型: {summary['filter_type']}")
    print(f"  状态变量: {summary['state_variables']}")
    print(f"  信念不确定性: {summary['belief_uncertainty']:.4f}")
    print(f"  平均信息增益: {summary['information_gain_stats']['average_gain']:.4f}")
    
    # 测试主动感知
    possible_actions = ["observe_position", "check_battery", "scan_for_target", "move_north"]
    recommendation = handler.recommend_observation_action(possible_actions)
    
    print(f"\n主动感知推荐:")
    print(f"  推荐动作: {recommendation['recommended_action']}")
    print(f"  预期信息增益: {recommendation['information_gain']:.4f}")
    print(f"  理由: {recommendation['reason']}")


if __name__ == "__main__":
    test_partial_observability()