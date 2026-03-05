#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
状态转移模型 - 实现世界状态演化的动态模型

核心功能:
1. 确定性状态转移模型
2. 概率状态转移模型
3. 学习的状态转移模型
4. 部分可观测环境下的状态转移
5. 多智能体交互模型

技术实现:
- 马尔可夫决策过程 (MDP) 框架
- 条件概率分布表示
- 神经网络动力学模型
- 贝叶斯非参数模型
- 可转移模型学习

状态转移公式:
P(S_{t+1} | S_t, A_t) = 转移模型

其中:
- S_t: 时刻t的状态
- A_t: 时刻t的动作
- P(S_{t+1} | S_t, A_t): 状态转移概率

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

# 导入错误处理
from core.error_handling import ErrorHandler

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()


class TransitionType(Enum):
    """状态转移类型枚举"""
    DETERMINISTIC = "deterministic"          # 确定性转移
    PROBABILISTIC = "probabilistic"          # 概率转移
    GAUSSIAN = "gaussian"                    # 高斯转移
    NEURAL_NETWORK = "neural_network"        # 神经网络转移
    BAYESIAN_NETWORK = "bayesian_network"    # 贝叶斯网络转移
    PARTIALLY_OBSERVABLE = "partially_observable"  # 部分可观测转移


class ModelLearningMethod(Enum):
    """模型学习方法枚举"""
    MAXIMUM_LIKELIHOOD = "maximum_likelihood"        # 最大似然估计
    BAYESIAN_INFERENCE = "bayesian_inference"        # 贝叶斯推断
    NEURAL_NETWORK = "neural_network"                # 神经网络学习
    GAUSSIAN_PROCESS = "gaussian_process"            # 高斯过程回归
    REINFORCEMENT_LEARNING = "reinforcement_learning"  # 强化学习


class StateTransitionModel:
    """
    状态转移模型 - 描述世界状态如何随时间演化
    
    核心特性:
    1. 支持多种转移类型
    2. 可学习的动态模型
    3. 不确定性量化
    4. 动作条件转移
    5. 时间相关性建模
    
    转移模型表示:
    - 离散状态: 转移矩阵
    - 连续状态: 参数化函数
    - 混合状态: 分层模型
    - 多智能体: 交互模型
    
    学习能力:
    - 从经验数据学习转移模型
    - 在线适应环境变化
    - 模型不确定性估计
    - 迁移学习到新领域
    """
    
    def __init__(self, 
                 state_variables: Optional[List[str]] = None,
                 action_space: Optional[List[str]] = None,
                 transition_type: Union[TransitionType, str] = TransitionType.PROBABILISTIC,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化状态转移模型
        
        Args:
            state_variables: 状态变量列表
            action_space: 动作空间列表
            transition_type: 转移类型
            config: 配置参数
        """
        # 配置参数
        self.config = config or self._get_default_config()
        
        # 状态变量
        self.state_variables = state_variables or []
        self.action_space = action_space or []
        
        # 转移类型
        if isinstance(transition_type, str):
            self.transition_type = TransitionType(transition_type)
        else:
            self.transition_type = transition_type
        
        # 转移模型存储
        self.transition_models = self._initialize_transition_models()
        self.learned_transitions: Dict[str, Any] = {}
        
        # 学习历史
        self.transition_history: List[Dict[str, Any]] = []
        self.max_history_size = self.config.get('max_history_size', 1000)
        
        # 元数据
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "transition_type": self.transition_type.value,
            "state_variables": self.state_variables.copy(),
            "action_space": self.action_space.copy(),
            "transition_count": 0
        }
        
        # 性能统计
        self.performance_stats = {
            "transitions_predicted": 0,
            "transitions_learned": 0,
            "prediction_accuracy": 0.0,
            "average_prediction_time": 0.0,
            "model_uncertainty_history": []
        }
        
        # 模型学习器（如果需要）
        self.model_learner = None
        if self.config.get('enable_learning', True):
            self._initialize_model_learner()
        
        logger.info(f"状态转移模型初始化完成，类型: {self.transition_type.value}")
        logger.info(f"状态变量: {self.state_variables}")
        logger.info(f"动作空间: {self.action_space}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'max_history_size': 1000,
            'enable_learning': True,
            'learning_rate': 0.01,
            'exploration_rate': 0.1,
            'model_uncertainty_threshold': 0.2,
            'min_transition_probability': 1e-6,
            'transition_smoothing': 0.01,
            'neural_network_hidden_layers': [64, 32],
            'gaussian_process_kernel': 'rbf',
            'bayesian_prior_strength': 0.5
        }
    
    def _initialize_transition_models(self) -> Dict[str, Any]:
        """
        初始化转移模型
        
        Returns:
            初始化后的转移模型
        """
        models = {}
        
        if self.transition_type == TransitionType.DETERMINISTIC:
            # 确定性转移：状态→状态的映射函数
            for var in self.state_variables:
                models[var] = {
                    "type": "deterministic",
                    "transition_function": None,  # 待设置
                    "action_conditioned": False
                }
        
        elif self.transition_type == TransitionType.PROBABILISTIC:
            # 概率转移：状态→状态的转移矩阵
            for var in self.state_variables:
                models[var] = {
                    "type": "probabilistic",
                    "transition_matrix": None,  # 待初始化
                    "action_conditioned": False,
                    "state_space": []  # 状态值列表
                }
        
        elif self.transition_type == TransitionType.GAUSSIAN:
            # 高斯转移：线性高斯模型
            for var in self.state_variables:
                models[var] = {
                    "type": "gaussian",
                    "A": 1.0,  # 状态转移系数
                    "B": 0.0,  # 动作转移系数
                    "Q": 0.1,  # 过程噪声方差
                    "action_conditioned": len(self.action_space) > 0
                }
        
        elif self.transition_type == TransitionType.NEURAL_NETWORK:
            # 神经网络转移：深度学习模型
            models = {
                "type": "neural_network",
                "model": None,  # 待初始化
                "state_dimension": len(self.state_variables),
                "action_dimension": len(self.action_space),
                "hidden_layers": self.config.get('neural_network_hidden_layers', [64, 32]),
                "trained": False
            }
        
        elif self.transition_type == TransitionType.BAYESIAN_NETWORK:
            # 贝叶斯网络转移：概率图模型
            models = {
                "type": "bayesian_network",
                "graph": nx.DiGraph(),
                "cpts": {},  # 条件概率表
                "learned": False
            }
        
        else:
            # 默认：部分可观测转移
            models = {
                "type": "partially_observable",
                "observation_model": None,
                "belief_transition": None
            }
        
        return models
    
    def _initialize_model_learner(self) -> None:
        """初始化模型学习器"""
        learning_method = self.config.get('learning_method', 'maximum_likelihood')
        
        if learning_method == ModelLearningMethod.NEURAL_NETWORK.value:
            self._initialize_neural_learner()
        elif learning_method == ModelLearningMethod.GAUSSIAN_PROCESS.value:
            self._initialize_gaussian_process_learner()
        elif learning_method == ModelLearningMethod.BAYESIAN_INFERENCE.value:
            self._initialize_bayesian_learner()
        else:
            # 默认：最大似然估计
            logger.info("使用最大似然估计作为默认学习器")
    
    def _initialize_neural_learner(self) -> None:
        """初始化神经网络学习器"""
        try:
            import torch
            import torch.nn as nn
            
            state_dim = len(self.state_variables)
            action_dim = len(self.action_space)
            
            # 简单的神经网络模型
            class TransitionNetwork(nn.Module):
                def __init__(self, state_dim, action_dim, hidden_layers):
                    super().__init__()
                    layers = []
                    
                    # 输入层：状态 + 动作
                    input_dim = state_dim + action_dim
                    
                    # 隐藏层
                    prev_dim = input_dim
                    for hidden_dim in hidden_layers:
                        layers.append(nn.Linear(prev_dim, hidden_dim))
                        layers.append(nn.ReLU())
                        prev_dim = hidden_dim
                    
                    # 输出层：下一状态预测
                    layers.append(nn.Linear(prev_dim, state_dim))
                    
                    self.network = nn.Sequential(*layers)
                
                def forward(self, state, action):
                    x = torch.cat([state, action], dim=-1)
                    return self.network(x)
            
            self.model_learner = {
                "type": "neural_network",
                "model": TransitionNetwork(state_dim, action_dim, 
                                         self.config.get('neural_network_hidden_layers', [64, 32])),
                "optimizer": None,  # 需要数据后初始化
                "criterion": nn.MSELoss(),
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
            }
            
            logger.info("神经网络学习器初始化完成")
            
        except ImportError:
            logger.warning("PyTorch 未安装，无法使用神经网络学习器")
            self.model_learner = None
    
    def _initialize_gaussian_process_learner(self) -> None:
        """初始化高斯过程学习器"""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel
            
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
            
            self.model_learner = {
                "type": "gaussian_process",
                "model": gp_model,
                "kernel": kernel,
                "trained": False
            }
            
            logger.info("高斯过程学习器初始化完成")
            
        except ImportError:
            logger.warning("scikit-learn 未安装，无法使用高斯过程学习器")
            self.model_learner = None
    
    def _initialize_bayesian_learner(self) -> None:
        """初始化贝叶斯学习器"""
        # 简化实现：狄利克雷先验
        self.model_learner = {
            "type": "bayesian_inference",
            "prior_strength": self.config.get('bayesian_prior_strength', 0.5),
            "transition_counts": defaultdict(lambda: defaultdict(float)),
            "learned": False
        }
        
        logger.info("贝叶斯学习器初始化完成")
    
    def set_deterministic_transition(self, 
                                    state_variable: str, 
                                    transition_func: Callable[[Any, Optional[str]], Any]) -> bool:
        """
        设置确定性转移函数
        
        Args:
            state_variable: 状态变量名称
            transition_func: 转移函数 f(state, action) → next_state
            
        Returns:
            是否成功设置
        """
        try:
            if state_variable not in self.state_variables:
                self.state_variables.append(state_variable)
            
            if self.transition_type == TransitionType.DETERMINISTIC:
                self.transition_models[state_variable] = {
                    "type": "deterministic",
                    "transition_function": transition_func,
                    "action_conditioned": True,  # 假设函数接受动作参数
                    "last_updated": time.time()
                }
                
                logger.info(f"为状态变量 {state_variable} 设置确定性转移函数")
                return True
            else:
                logger.error(f"当前转移类型 {self.transition_type.value} 不支持确定性转移函数")
                return False
                
        except Exception as e:
            logger.error(f"设置确定性转移函数失败: {e}")
            error_handler.handle_error(e, "StateTransitionModel", "设置确定性转移函数失败")
            return False
    
    def set_probabilistic_transition(self, 
                                    state_variable: str, 
                                    state_space: List[Any],
                                    transition_matrix: Optional[np.ndarray] = None) -> bool:
        """
        设置概率转移矩阵
        
        Args:
            state_variable: 状态变量名称
            state_space: 状态空间列表
            transition_matrix: 转移矩阵（可选，默认均匀分布）
            
        Returns:
            是否成功设置
        """
        try:
            if state_variable not in self.state_variables:
                self.state_variables.append(state_variable)
            
            if self.transition_type == TransitionType.PROBABILISTIC:
                n_states = len(state_space)
                
                # 默认转移矩阵：均匀分布
                if transition_matrix is None:
                    transition_matrix = np.ones((n_states, n_states)) / n_states
                
                # 验证转移矩阵
                if transition_matrix.shape != (n_states, n_states):
                    logger.error(f"转移矩阵形状 {transition_matrix.shape} 与状态空间大小 {n_states} 不匹配")
                    return False
                
                # 确保每行和为1
                for i in range(n_states):
                    row_sum = np.sum(transition_matrix[i])
                    if abs(row_sum - 1.0) > 1e-6:
                        transition_matrix[i] = transition_matrix[i] / row_sum
                
                self.transition_models[state_variable] = {
                    "type": "probabilistic",
                    "state_space": state_space.copy(),
                    "transition_matrix": transition_matrix.tolist(),
                    "action_conditioned": False,
                    "last_updated": time.time()
                }
                
                logger.info(f"为状态变量 {state_variable} 设置概率转移矩阵，状态空间大小: {n_states}")
                return True
            else:
                logger.error(f"当前转移类型 {self.transition_type.value} 不支持概率转移矩阵")
                return False
                
        except Exception as e:
            logger.error(f"设置概率转移矩阵失败: {e}")
            error_handler.handle_error(e, "StateTransitionModel", "设置概率转移矩阵失败")
            return False
    
    def set_gaussian_transition(self, 
                               state_variable: str,
                               A: float = 1.0,
                               B: float = 0.0,
                               Q: float = 0.1) -> bool:
        """
        设置高斯转移模型
        
        Args:
            state_variable: 状态变量名称
            A: 状态转移系数
            B: 动作转移系数
            Q: 过程噪声方差
            
        Returns:
            是否成功设置
        """
        try:
            if state_variable not in self.state_variables:
                self.state_variables.append(state_variable)
            
            if self.transition_type == TransitionType.GAUSSIAN:
                self.transition_models[state_variable] = {
                    "type": "gaussian",
                    "A": float(A),
                    "B": float(B),
                    "Q": float(Q),
                    "action_conditioned": B != 0.0,
                    "last_updated": time.time()
                }
                
                logger.info(f"为状态变量 {state_variable} 设置高斯转移模型: A={A}, B={B}, Q={Q}")
                return True
            else:
                logger.error(f"当前转移类型 {self.transition_type.value} 不支持高斯转移模型")
                return False
                
        except Exception as e:
            logger.error(f"设置高斯转移模型失败: {e}")
            error_handler.handle_error(e, "StateTransitionModel", "设置高斯转移模型失败")
            return False
    
    def predict_next_state(self, 
                          current_state: Dict[str, Any],
                          action: Optional[str] = None,
                          use_learned_model: bool = False) -> Dict[str, Any]:
        """
        预测下一状态
        
        Args:
            current_state: 当前状态
            action: 执行的动作（可选）
            use_learned_model: 是否使用学习到的模型
            
        Returns:
            预测的下一状态
        """
        try:
            start_time = time.time()
            
            # 选择模型：学习模型或预设模型
            if use_learned_model and self.learned_transitions:
                next_state = self._predict_with_learned_model(current_state, action)
            else:
                next_state = self._predict_with_preset_model(current_state, action)
            
            # 记录预测历史
            self._record_transition_prediction(current_state, action, next_state)
            
            # 更新性能统计
            prediction_time = time.time() - start_time
            self.performance_stats["transitions_predicted"] += 1
            self.performance_stats["average_prediction_time"] = (
                self.performance_stats["average_prediction_time"] * 
                (self.performance_stats["transitions_predicted"] - 1) + 
                prediction_time
            ) / self.performance_stats["transitions_predicted"]
            
            # 更新元数据
            self.metadata["last_updated"] = datetime.now().isoformat()
            self.metadata["transition_count"] += 1
            
            logger.debug(f"状态转移预测完成，耗时: {prediction_time:.4f} 秒")
            return next_state
            
        except Exception as e:
            logger.error(f"状态转移预测失败: {e}")
            error_handler.handle_error(e, "StateTransitionModel", "状态转移预测失败")
            
            # 返回当前状态作为保守预测
            return current_state.copy()
    
    def _predict_with_preset_model(self, 
                                  current_state: Dict[str, Any],
                                  action: Optional[str] = None) -> Dict[str, Any]:
        """使用预设模型预测下一状态"""
        next_state = {}
        
        for var_name, var_model in self.transition_models.items():
            current_value = current_state.get(var_name)
            
            if var_model["type"] == "deterministic":
                # 确定性转移
                transition_func = var_model.get("transition_function")
                if callable(transition_func):
                    try:
                        next_value = transition_func(current_value, action)
                        next_state[var_name] = next_value
                    except Exception as e:
                        logger.warning(f"确定性转移函数执行失败: {e}")
                        next_state[var_name] = current_value
                else:
                    next_state[var_name] = current_value
            
            elif var_model["type"] == "probabilistic":
                # 概率转移
                state_space = var_model.get("state_space", [])
                transition_matrix = var_model.get("transition_matrix", [])
                
                if state_space and transition_matrix:
                    # 找到当前状态的索引
                    if current_value in state_space:
                        idx = state_space.index(current_value)
                        
                        # 从转移矩阵中采样下一状态
                        probs = transition_matrix[idx]
                        next_idx = np.random.choice(len(state_space), p=probs)
                        next_state[var_name] = state_space[next_idx]
                    else:
                        # 当前状态不在已知状态空间中
                        next_state[var_name] = current_value
                else:
                    next_state[var_name] = current_value
            
            elif var_model["type"] == "gaussian":
                # 高斯转移
                A = var_model.get("A", 1.0)
                B = var_model.get("B", 0.0)
                Q = var_model.get("Q", 0.1)
                
                if isinstance(current_value, (int, float)):
                    # 线性高斯模型: x_{t+1} = A*x_t + B*a + w, w ~ N(0, Q)
                    action_effect = 0.0
                    if action is not None and B != 0.0:
                        # 简化：将动作映射为数值
                        action_effect = B * ((zlib.adler32(str(action).encode('utf-8')) & 0xffffffff) % 10)  # 简化映射
                    
                    noise = np.random.normal(0, math.sqrt(Q))
                    next_value = A * current_value + action_effect + noise
                    next_state[var_name] = next_value
                else:
                    next_state[var_name] = current_value
            
            else:
                # 其他模型类型：保持当前状态
                next_state[var_name] = current_value
        
        return next_state
    
    def _predict_with_learned_model(self, 
                                   current_state: Dict[str, Any],
                                   action: Optional[str] = None) -> Dict[str, Any]:
        """使用学习模型预测下一状态"""
        next_state = {}
        
        # 检查是否有学习到的模型
        if not self.learned_transitions:
            logger.warning("没有学习到的转移模型，使用预设模型")
            return self._predict_with_preset_model(current_state, action)
        
        # 根据学习器类型进行预测
        learner_type = self.model_learner.get("type") if self.model_learner else None
        
        if learner_type == "neural_network":
            next_state = self._predict_with_neural_network(current_state, action)
        elif learner_type == "gaussian_process":
            next_state = self._predict_with_gaussian_process(current_state, action)
        elif learner_type == "bayesian_inference":
            next_state = self._predict_with_bayesian_model(current_state, action)
        else:
            # 默认：最大似然估计
            next_state = self._predict_with_mle(current_state, action)
        
        return next_state
    
    def _predict_with_neural_network(self, 
                                    current_state: Dict[str, Any],
                                    action: Optional[str] = None) -> Dict[str, Any]:
        """使用神经网络预测下一状态"""
        # 简化实现：返回当前状态
        logger.warning("神经网络预测未完全实现，返回当前状态")
        return current_state.copy()
    
    def _predict_with_gaussian_process(self, 
                                      current_state: Dict[str, Any],
                                      action: Optional[str] = None) -> Dict[str, Any]:
        """使用高斯过程预测下一状态"""
        # 简化实现：返回当前状态
        logger.warning("高斯过程预测未完全实现，返回当前状态")
        return current_state.copy()
    
    def _predict_with_bayesian_model(self, 
                                    current_state: Dict[str, Any],
                                    action: Optional[str] = None) -> Dict[str, Any]:
        """使用贝叶斯模型预测下一状态"""
        # 简化实现：返回当前状态
        logger.warning("贝叶斯模型预测未完全实现，返回当前状态")
        return current_state.copy()
    
    def _predict_with_mle(self, 
                         current_state: Dict[str, Any],
                         action: Optional[str] = None) -> Dict[str, Any]:
        """使用最大似然估计预测下一状态"""
        next_state = {}
        
        # 检查学习到的转移
        for var_name in self.state_variables:
            current_value = current_state.get(var_name)
            
            if var_name in self.learned_transitions:
                learned_model = self.learned_transitions[var_name]
                
                if learned_model["type"] == "probabilistic":
                    # 从学习到的转移矩阵中采样
                    state_space = learned_model.get("state_space", [])
                    transition_matrix = learned_model.get("transition_matrix", [])
                    
                    if state_space and transition_matrix:
                        if current_value in state_space:
                            idx = state_space.index(current_value)
                            probs = transition_matrix[idx]
                            
                            # 添加平滑
                            smoothing = self.config.get('transition_smoothing', 0.01)
                            probs = [p + smoothing for p in probs]
                            probs_sum = sum(probs)
                            probs = [p / probs_sum for p in probs]
                            
                            next_idx = np.random.choice(len(state_space), p=probs)
                            next_state[var_name] = state_space[next_idx]
                        else:
                            # 当前状态不在学习到的状态空间中
                            next_state[var_name] = current_value
                    else:
                        next_state[var_name] = current_value
                else:
                    next_state[var_name] = current_value
            else:
                next_state[var_name] = current_value
        
        return next_state
    
    def _record_transition_prediction(self, 
                                     current_state: Dict[str, Any],
                                     action: Optional[str],
                                     predicted_state: Dict[str, Any]) -> None:
        """记录转移预测"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "current_state": current_state.copy(),
            "action": action,
            "predicted_state": predicted_state.copy(),
            "transition_type": self.transition_type.value
        }
        
        self.transition_history.append(history_entry)
        
        # 限制历史大小
        if len(self.transition_history) > self.max_history_size:
            self.transition_history = self.transition_history[-self.max_history_size:]
    
    def learn_from_transition(self, 
                             current_state: Dict[str, Any],
                             action: Optional[str],
                             next_state: Dict[str, Any]) -> bool:
        """
        从状态转移中学习
        
        Args:
            current_state: 当前状态
            action: 执行的动作
            next_state: 实际下一状态
            
        Returns:
            是否成功学习
        """
        try:
            start_time = time.time()
            
            # 记录转移经验
            transition_experience = {
                "current_state": current_state.copy(),
                "action": action,
                "next_state": next_state.copy(),
                "timestamp": time.time()
            }
            
            # 根据学习器类型进行学习
            learner_type = self.model_learner.get("type") if self.model_learner else None
            
            if learner_type == "maximum_likelihood":
                self._learn_with_mle(transition_experience)
            elif learner_type == "bayesian_inference":
                self._learn_with_bayesian(transition_experience)
            elif learner_type == "neural_network":
                self._learn_with_neural_network(transition_experience)
            elif learner_type == "gaussian_process":
                self._learn_with_gaussian_process(transition_experience)
            else:
                # 默认：最大似然估计
                self._learn_with_mle(transition_experience)
            
            # 更新学习到的转移模型
            self._update_learned_transitions()
            
            # 更新性能统计
            self.performance_stats["transitions_learned"] += 1
            
            # 记录学习历史
            self._record_learning_history(transition_experience)
            
            logger.debug(f"从转移中学习完成，耗时: {time.time() - start_time:.4f} 秒")
            return True
            
        except Exception as e:
            logger.error(f"从转移中学习失败: {e}")
            error_handler.handle_error(e, "StateTransitionModel", "从转移中学习失败")
            return False
    
    def _learn_with_mle(self, transition_experience: Dict[str, Any]) -> None:
        """使用最大似然估计学习"""
        current_state = transition_experience["current_state"]
        next_state = transition_experience["next_state"]
        
        for var_name in self.state_variables:
            current_value = current_state.get(var_name)
            next_value = next_state.get(var_name)
            
            if current_value is not None and next_value is not None:
                # 初始化学习到的转移
                if var_name not in self.learned_transitions:
                    self.learned_transitions[var_name] = {
                        "type": "probabilistic",
                        "state_space": [],
                        "transition_counts": defaultdict(lambda: defaultdict(float)),
                        "transition_matrix": []
                    }
                
                learned_model = self.learned_transitions[var_name]
                
                # 确保状态空间包含当前值和下一个值
                state_space = learned_model["state_space"]
                if current_value not in state_space:
                    state_space.append(current_value)
                if next_value not in state_space:
                    state_space.append(next_value)
                
                # 更新转移计数
                transition_counts = learned_model["transition_counts"]
                transition_counts[current_value][next_value] += 1.0
    
    def _learn_with_bayesian(self, transition_experience: Dict[str, Any]) -> None:
        """使用贝叶斯推断学习"""
        # 简化实现：与MLE类似，但添加先验
        self._learn_with_mle(transition_experience)
    
    def _learn_with_neural_network(self, transition_experience: Dict[str, Any]) -> None:
        """使用神经网络学习"""
        logger.warning("神经网络学习未完全实现")
    
    def _learn_with_gaussian_process(self, transition_experience: Dict[str, Any]) -> None:
        """使用高斯过程学习"""
        logger.warning("高斯过程学习未完全实现")
    
    def _update_learned_transitions(self) -> None:
        """更新学习到的转移模型"""
        for var_name, learned_model in self.learned_transitions.items():
            if learned_model["type"] == "probabilistic":
                state_space = learned_model["state_space"]
                transition_counts = learned_model["transition_counts"]
                
                n_states = len(state_space)
                if n_states == 0:
                    continue
                
                # 创建转移矩阵
                transition_matrix = np.zeros((n_states, n_states))
                
                for i, from_state in enumerate(state_space):
                    counts_from = transition_counts[from_state]
                    total_count = sum(counts_from.values())
                    
                    if total_count > 0:
                        # 归一化得到概率
                        for j, to_state in enumerate(state_space):
                            count = counts_from.get(to_state, 0.0)
                            transition_matrix[i][j] = count / total_count
                    else:
                        # 如果没有观察数据，使用均匀分布
                        transition_matrix[i] = np.ones(n_states) / n_states
                
                # 添加平滑
                smoothing = self.config.get('transition_smoothing', 0.01)
                transition_matrix = (transition_matrix + smoothing) / (1.0 + smoothing * n_states)
                
                learned_model["transition_matrix"] = transition_matrix.tolist()
                learned_model["last_updated"] = time.time()
    
    def _record_learning_history(self, transition_experience: Dict[str, Any]) -> None:
        """记录学习历史"""
        # 这里可以记录学习过程中的详细信息
        pass
    
    def get_transition_probability(self, 
                                  current_state: Dict[str, Any],
                                  next_state: Dict[str, Any],
                                  action: Optional[str] = None) -> float:
        """
        获取转移概率
        
        Args:
            current_state: 当前状态
            next_state: 下一状态
            action: 执行的动作（可选）
            
        Returns:
            转移概率
        """
        try:
            probability = 1.0
            
            for var_name in self.state_variables:
                current_value = current_state.get(var_name)
                next_value = next_state.get(var_name)
                
                if current_value is None or next_value is None:
                    continue
                
                var_prob = self._get_variable_transition_probability(
                    var_name, current_value, next_value, action
                )
                
                probability *= var_prob
            
            # 确保概率在合理范围内
            min_prob = self.config.get('min_transition_probability', 1e-6)
            probability = max(probability, min_prob)
            
            return probability
            
        except Exception as e:
            logger.error(f"计算转移概率失败: {e}")
            return 0.0
    
    def _get_variable_transition_probability(self, 
                                            var_name: str,
                                            current_value: Any,
                                            next_value: Any,
                                            action: Optional[str] = None) -> float:
        """获取单个变量的转移概率"""
        # 首先检查学习到的模型
        if var_name in self.learned_transitions:
            learned_model = self.learned_transitions[var_name]
            
            if learned_model["type"] == "probabilistic":
                state_space = learned_model.get("state_space", [])
                transition_matrix = learned_model.get("transition_matrix", [])
                
                if state_space and transition_matrix:
                    if current_value in state_space and next_value in state_space:
                        i = state_space.index(current_value)
                        j = state_space.index(next_value)
                        
                        if i < len(transition_matrix) and j < len(transition_matrix[i]):
                            return transition_matrix[i][j]
        
        # 如果没有学习到的模型，使用预设模型
        if var_name in self.transition_models:
            var_model = self.transition_models[var_name]
            
            if var_model["type"] == "probabilistic":
                state_space = var_model.get("state_space", [])
                transition_matrix = var_model.get("transition_matrix", [])
                
                if state_space and transition_matrix:
                    if current_value in state_space and next_value in state_space:
                        i = state_space.index(current_value)
                        j = state_space.index(next_value)
                        
                        if i < len(transition_matrix) and j < len(transition_matrix[i]):
                            return transition_matrix[i][j]
            
            elif var_model["type"] == "gaussian":
                # 高斯转移：计算概率密度
                A = var_model.get("A", 1.0)
                B = var_model.get("B", 0.0)
                Q = var_model.get("Q", 0.1)
                
                if isinstance(current_value, (int, float)) and isinstance(next_value, (int, float)):
                    # 预测值
                    predicted = A * current_value
                    
                    # 高斯概率密度
                    error = next_value - predicted
                    probability = math.exp(-0.5 * error * error / Q) / math.sqrt(2 * math.pi * Q)
                    return probability
        
        # 默认：均匀概率
        return 0.5
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        summary = {
            "transition_type": self.transition_type.value,
            "state_variables": self.state_variables.copy(),
            "action_space": self.action_space.copy(),
            "learned_transition_count": len(self.learned_transitions),
            "performance_stats": self.performance_stats.copy(),
            "metadata": self.metadata.copy()
        }
        
        # 添加学习器信息
        if self.model_learner:
            summary["model_learner"] = {
                "type": self.model_learner.get("type", "unknown"),
                "trained": self.model_learner.get("trained", False)
            }
        
        return summary


# 测试和示例函数
def create_example_transition_model() -> StateTransitionModel:
    """创建示例转移模型"""
    # 定义状态变量和动作空间
    state_variables = ["location", "battery_level"]
    action_space = ["move_north", "move_south", "charge"]
    
    # 创建转移模型
    transition_model = StateTransitionModel(
        state_variables=state_variables,
        action_space=action_space,
        transition_type=TransitionType.PROBABILISTIC
    )
    
    # 设置位置转移矩阵
    location_states = ["room_a", "room_b", "room_c", "room_d"]
    transition_matrix = np.array([
        [0.1, 0.6, 0.2, 0.1],  # room_a → [room_a, room_b, room_c, room_d]
        [0.6, 0.1, 0.2, 0.1],  # room_b → ...
        [0.2, 0.2, 0.1, 0.5],  # room_c → ...
        [0.1, 0.1, 0.5, 0.3]   # room_d → ...
    ])
    
    transition_model.set_probabilistic_transition(
        "location", location_states, transition_matrix
    )
    
    # 设置电池水平的高斯转移模型
    transition_model.set_gaussian_transition(
        "battery_level", A=0.95, B=0.1, Q=0.05
    )
    
    return transition_model


def test_transition_prediction() -> None:
    """测试转移预测"""
    print("测试状态转移预测...")
    
    # 创建转移模型
    transition_model = create_example_transition_model()
    
    # 创建当前状态
    current_state = {
        "location": "room_a",
        "battery_level": 80.0
    }
    
    # 选择动作
    action = "move_north"
    
    # 预测下一状态
    next_state = transition_model.predict_next_state(current_state, action)
    
    print(f"当前状态: {current_state}")
    print(f"执行动作: {action}")
    print(f"预测下一状态: {next_state}")
    
    # 计算转移概率
    probability = transition_model.get_transition_probability(current_state, next_state, action)
    print(f"转移概率: {probability:.4f}")
    
    # 获取模型摘要
    summary = transition_model.get_model_summary()
    print(f"模型摘要:")
    print(f"  转移类型: {summary['transition_type']}")
    print(f"  预测次数: {summary['performance_stats']['transitions_predicted']}")
    print(f"  学习次数: {summary['performance_stats']['transitions_learned']}")


if __name__ == "__main__":
    test_transition_prediction()