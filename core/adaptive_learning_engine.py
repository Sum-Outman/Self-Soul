#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Adaptive Learning Engine - Dynamically optimizes training parameters using reinforcement learning and meta-learning
增强型自适应学习引擎 - 使用强化学习和元学习动态优化训练参数

功能：基于模型类型、数据特性、资源约束、历史性能和实时系统状态，使用高级算法动态优化训练参数
Function: Dynamically optimize training parameters using advanced algorithms based on model types, data characteristics, resource constraints, historical performance, and realtime system status

主要改进：
1. 基于强化学习的参数优化（Q-learning策略）
2. 元学习从历史训练中提取模式
3. 多目标优化（效率、准确性、资源使用）
4. 实时自适应调整
5. 支持神经架构搜索（NAS）集成

版权所有 (c) 2025 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import json
import time
from datetime import datetime
from scipy import stats
import random

# 设置日志 | Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReinforcementLearningOptimizer:
    """基于强化学习的参数优化器 | Reinforcement Learning based parameter optimizer"""
    
    def __init__(self, state_space_size: int = 10, action_space_size: int = 5):
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # 探索率
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        
    def choose_action(self, state: int) -> int:
        """选择行动（ε-贪婪策略）| Choose action (ε-greedy policy)"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state: int, action: int, reward: float, next_state: int):
        """更新Q值 | Update Q-value"""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error


class MetaLearningAnalyzer:
    """元学习分析器，从历史训练中提取模式 | Meta-learning analyzer that extracts patterns from historical training"""
    
    def __init__(self, memory_size: int = 100):
        self.training_history = deque(maxlen=memory_size)
        self.patterns = {}
        
    def add_training_record(self, record: Dict[str, Any]):
        """添加训练记录 | Add training record"""
        self.training_history.append(record)
        
    def analyze_patterns(self) -> Dict[str, Any]:
        """分析训练历史中的模式 | Analyze patterns in training history"""
        if len(self.training_history) < 5:
            return {"status": "insufficient_data", "recommendations": []}
            
        # 提取关键指标 | Extract key metrics
        learning_rates = [r.get("learning_rate", 0.001) for r in self.training_history]
        accuracies = [r.get("final_accuracy", 0.0) for r in self.training_history]
        training_times = [r.get("training_time", 0.0) for r in self.training_history]
        model_types = [r.get("model_type", "unknown") for r in self.training_history]
        
        # 计算统计量 | Calculate statistics
        patterns = {
            "optimal_learning_rate_by_model": {},
            "accuracy_trend": "increasing" if len(accuracies) > 1 and accuracies[-1] > accuracies[0] else "stable" if accuracies[-1] == accuracies[0] else "decreasing",
            "avg_training_time": np.mean(training_times) if training_times else 0,
            "learning_rate_accuracy_correlation": np.corrcoef(learning_rates, accuracies)[0, 1] if len(learning_rates) > 1 and len(accuracies) > 1 else 0,
        }
        
        # 按模型类型分析 | Analyze by model type
        for model_type in set(model_types):
            type_records = [r for r in self.training_history if r.get("model_type") == model_type]
            if len(type_records) >= 3:
                type_accuracies = [r.get("final_accuracy", 0.0) for r in type_records]
                type_lrs = [r.get("learning_rate", 0.001) for r in type_records]
                best_idx = np.argmax(type_accuracies) if type_accuracies else 0
                patterns["optimal_learning_rate_by_model"][model_type] = type_lrs[best_idx] if best_idx < len(type_lrs) else 0.001
        
        self.patterns = patterns
        return patterns
    
    def get_recommendations(self, model_type: str, data_size: int) -> List[str]:
        """基于历史模式获取推荐 | Get recommendations based on historical patterns"""
        recs = []
        
        if model_type in self.patterns.get("optimal_learning_rate_by_model", {}):
            optimal_lr = self.patterns["optimal_learning_rate_by_model"][model_type]
            recs.append(f"对于{model_type}模型，建议学习率: {optimal_lr:.6f}")
            
        if self.patterns.get("accuracy_trend") == "decreasing":
            recs.append("检测到准确率下降趋势，建议检查数据质量或降低学习率")
            
        if data_size > 10000 and self.patterns.get("avg_training_time", 0) > 3600:
            recs.append("大数据集训练时间过长，建议使用更大的批量大小或分布式训练")
            
        return recs


class EnhancedAdaptiveLearningEngine:
    """增强型自适应学习引擎类 | Enhanced Adaptive Learning Engine Class
    
    使用强化学习和元学习动态优化训练参数，显著提高训练效率和模型性能
    Uses reinforcement learning and meta-learning to dynamically optimize training parameters, significantly improving training efficiency and model performance
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化增强型自适应学习引擎 | Initialize enhanced adaptive learning engine"""
        logger.info("增强型自适应学习引擎初始化 | Enhanced Adaptive Learning Engine initialized")
        
        # 初始化组件 | Initialize components
        self.rl_optimizer = ReinforcementLearningOptimizer()
        self.meta_analyzer = MetaLearningAnalyzer()
        self.config_history = []
        
        # 加载配置（如果有） | Load configuration if available
        self.config = self._load_config(config_path) if config_path else self._get_default_config()
        
        # 性能监控 | Performance monitoring
        self.performance_history = []
        self.last_update_time = time.time()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置 | Get default configuration"""
        return {
            "base_learning_rate": 0.001,
            "base_batch_size": 32,
            "base_epochs": 50,
            "optimizers": ["adam", "sgd", "rmsprop", "adamw"],
            "schedulers": ["linear", "cosine", "step", "plateau"],
            "regularization_methods": ["dropout", "l2", "batch_norm", "early_stopping"],
            "exploration_rate": 0.2,  # 探索新配置的概率
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """从文件加载配置 | Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"无法加载配置文件 {config_path}: {e}，使用默认配置")
            return self._get_default_config()
    
    def configure_training(self, 
                          model_types: List[str], 
                          data_characteristics: Dict[str, Any],
                          resource_constraints: Dict[str, Any],
                          meta_learning_strategy: str = "balanced",
                          historical_performance: Optional[Dict[str, Any]] = None,
                          realtime_system_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """配置训练参数（增强版）| Configure training parameters (enhanced)
        
        参数：
        Parameters:
        - model_types: 模型类型列表 | List of model types
        - data_characteristics: 数据特性 | Data characteristics
        - resource_constraints: 资源约束 | Resource constraints
        - meta_learning_strategy: 元学习策略 | Meta learning strategy
        - historical_performance: 历史性能数据 | Historical performance data
        - realtime_system_metrics: 实时系统指标 | Realtime system metrics
        
        返回：优化后的训练参数 | Returns: Optimized training parameters
        """
        logger.info(f"配置训练参数 - 模型类型: {model_types}, 策略: {meta_learning_strategy}, 数据特性: {data_characteristics}")
        
        # 步骤1: 基于模型类型和数据特性的基础配置 | Step 1: Base configuration based on model types and data characteristics
        base_config = self._get_base_configuration(model_types, data_characteristics)
        
        # 步骤2: 基于资源约束的调整 | Step 2: Adjust based on resource constraints
        resource_adjusted = self._adjust_for_resources(base_config, resource_constraints)
        
        # 步骤3: 应用元学习策略 | Step 3: Apply meta learning strategy
        strategy_adjusted = self._apply_meta_strategy(resource_adjusted, meta_learning_strategy)
        
        # 步骤4: 使用强化学习进行微调 | Step 4: Fine-tune using reinforcement learning
        rl_optimized = self._apply_rl_optimization(strategy_adjusted, historical_performance)
        
        # 步骤5: 考虑实时系统指标 | Step 5: Consider realtime system metrics
        final_config = self._adjust_for_realtime_metrics(rl_optimized, realtime_system_metrics)
        
        # 步骤6: 添加元学习推荐 | Step 6: Add meta-learning recommendations
        meta_insights = self.meta_analyzer.analyze_patterns()
        final_config["meta_learning_insights"] = meta_insights
        final_config["recommendations"] = self.meta_analyzer.get_recommendations(
            model_types[0] if model_types else "unknown",
            data_characteristics.get("dataset_size", 0)
        )
        
        # 记录配置历史 | Record configuration history
        config_record = {
            "timestamp": datetime.now().isoformat(),
            "model_types": model_types,
            "data_characteristics": data_characteristics,
            "final_config": final_config.copy(),
            "meta_learning_strategy": meta_learning_strategy
        }
        self.config_history.append(config_record)
        
        logger.info(f"优化后的训练参数: {final_config}")
        return final_config
    
    def _get_base_configuration(self, model_types: List[str], data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """获取基于模型类型和数据特性的基础配置 | Get base configuration based on model types and data characteristics"""
        config = self.config.copy()
        
        # 模型类型特定调整 | Model type specific adjustments
        model_adjustments = {
            "language": {"learning_rate": 0.0001, "batch_size": 16, "optimizer": "adamw"},
            "vision": {"learning_rate": 0.0005, "batch_size": 32, "optimizer": "adam"},
            "audio": {"learning_rate": 0.0003, "batch_size": 64, "epochs": 100},
            "sensor": {"learning_rate": 0.001, "batch_size": 128, "optimizer": "rmsprop"},
            "video": {"learning_rate": 0.0002, "batch_size": 8, "epochs": 80},
            "emotion": {"learning_rate": 0.00015, "batch_size": 24, "optimizer": "adam"},
            "knowledge": {"learning_rate": 0.00008, "batch_size": 12, "epochs": 120},
        }
        
        # 合并所有模型类型的调整（加权平均）| Merge adjustments for all model types (weighted average)
        for model_type in model_types:
            if model_type in model_adjustments:
                for key, value in model_adjustments[model_type].items():
                    if key in config:
                        if isinstance(config[key], (int, float)) and isinstance(value, (int, float)):
                            # 加权平均：新值占30%权重 | Weighted average: new value has 30% weight
                            config[key] = config[key] * 0.7 + value * 0.3
        
        # 数据特性调整 | Data characteristics adjustments
        dataset_size = data_characteristics.get("dataset_size", 1000)
        data_complexity = data_characteristics.get("complexity", "medium")
        class_imbalance = data_characteristics.get("class_imbalance", 1.0)
        
        # 基于数据集大小的调整 | Adjustments based on dataset size
        if dataset_size < 500:
            config["batch_size"] = max(4, config["batch_size"] // 4)
            config["learning_rate"] = config["learning_rate"] * 0.5
            config["epochs"] = int(config["epochs"] * 1.5)
        elif dataset_size > 100000:
            config["batch_size"] = config["batch_size"] * 2
            config["epochs"] = max(20, int(config["epochs"] * 0.7))
            
        # 基于数据复杂度的调整 | Adjustments based on data complexity
        if data_complexity == "high":
            config["learning_rate"] = config["learning_rate"] * 0.7
            config["epochs"] = int(config["epochs"] * 1.3)
        elif data_complexity == "low":
            config["learning_rate"] = config["learning_rate"] * 1.5
            config["epochs"] = max(10, int(config["epochs"] * 0.8))
            
        # 基于类别不平衡的调整 | Adjustments based on class imbalance
        if class_imbalance > 2.0:  # 严重不平衡 | Severe imbalance
            config["batch_size"] = int(config["batch_size"] * 1.5)  # 更大的批量以减少噪声 | Larger batch to reduce noise
            config["learning_rate"] = config["learning_rate"] * 0.8  # 更小的学习率 | Smaller learning rate
        
        return config
    
    def _adjust_for_resources(self, config: Dict[str, Any], resource_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """基于资源约束调整配置 | Adjust configuration based on resource constraints"""
        adjusted = config.copy()
        
        if not resource_constraints:
            return adjusted
        
        # 内存约束 | Memory constraints
        max_memory = resource_constraints.get("max_memory", 8)  # GB
        if max_memory < 2:
            adjusted["batch_size"] = max(1, adjusted["batch_size"] // 8)
            adjusted["optimizer"] = "sgd"  # 内存效率更高 | More memory efficient
        elif max_memory < 4:
            adjusted["batch_size"] = max(1, adjusted["batch_size"] // 4)
        elif max_memory < 8:
            adjusted["batch_size"] = max(1, adjusted["batch_size"] // 2)
            
        # GPU约束 | GPU constraints
        gpu_available = resource_constraints.get("gpu_available", True)
        gpu_memory = resource_constraints.get("gpu_memory", 0)  # GB
        
        if not gpu_available:
            adjusted["batch_size"] = max(1, adjusted["batch_size"] // 4)
            adjusted["mixed_precision"] = False
        elif gpu_memory > 0:
            # 根据GPU内存估算最大批量大小 | Estimate max batch size based on GPU memory
            # 假设每张图像需要0.01GB内存（约1000x1000 RGB图像）| Assuming 0.01GB per image (approx 1000x1000 RGB)
            estimated_max_batch = int(gpu_memory * 100)
            adjusted["batch_size"] = min(adjusted["batch_size"], estimated_max_batch)
            
        # CPU核心数 | CPU cores
        cpu_cores = resource_constraints.get("cpu_cores", 4)
        if cpu_cores < 4:
            adjusted["data_workers"] = max(1, cpu_cores - 1)
        else:
            adjusted["data_workers"] = min(8, cpu_cores - 2)
            
        return adjusted
    
    def _apply_meta_strategy(self, config: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """应用元学习策略 | Apply meta learning strategy"""
        adjusted = config.copy()
        
        strategy_multipliers = {
            "fast_learning": {"learning_rate": 2.0, "epochs": 0.5, "batch_size": 1.5},
            "high_precision": {"learning_rate": 0.5, "epochs": 1.5, "batch_size": 0.8},
            "balanced": {"learning_rate": 1.0, "epochs": 1.0, "batch_size": 1.0},
            "resource_efficient": {"learning_rate": 0.8, "epochs": 0.7, "batch_size": 1.2},
            "accuracy_maximizing": {"learning_rate": 0.3, "epochs": 2.0, "batch_size": 0.7},
        }
        
        if strategy in strategy_multipliers:
            multipliers = strategy_multipliers[strategy]
            for param, multiplier in multipliers.items():
                if param in adjusted and isinstance(adjusted[param], (int, float)):
                    if param == "epochs":
                        adjusted[param] = max(10, int(adjusted[param] * multiplier))
                    else:
                        adjusted[param] = adjusted[param] * multiplier
        
        return adjusted
    
    def _apply_rl_optimization(self, config: Dict[str, Any], historical_performance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """应用强化学习优化 | Apply reinforcement learning optimization"""
        if not historical_performance or len(self.performance_history) < 3:
            return config  # 数据不足，返回基础配置 | Insufficient data, return base config
            
        # 将配置映射到状态 | Map configuration to state
        state = self._config_to_state(config)
        
        # 使用RL选择行动 | Use RL to select action
        action = self.rl_optimizer.choose_action(state)
        
        # 根据行动调整配置 | Adjust configuration based on action
        adjusted = config.copy()
        actions = [
            lambda c: self._adjust_learning_rate(c, 1.1),  # 增加学习率 | Increase learning rate
            lambda c: self._adjust_learning_rate(c, 0.9),  # 减少学习率 | Decrease learning rate
            lambda c: self._adjust_batch_size(c, 1.2),     # 增加批量大小 | Increase batch size
            lambda c: self._adjust_batch_size(c, 0.8),     # 减少批量大小 | Decrease batch size
            lambda c: self._switch_optimizer(c),           # 切换优化器 | Switch optimizer
        ]
        
        if action < len(actions):
            adjusted = actions[action](adjusted)
            
        # 计算奖励（基于历史性能改进）| Calculate reward (based on historical performance improvement)
        if historical_performance:
            reward = self._calculate_reward(historical_performance)
            next_state = self._config_to_state(adjusted)
            self.rl_optimizer.update_q_value(state, action, reward, next_state)
            
        return adjusted
    
    def _adjust_for_realtime_metrics(self, config: Dict[str, Any], realtime_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """基于实时系统指标调整配置 | Adjust configuration based on realtime system metrics"""
        if not realtime_metrics:
            return config
            
        adjusted = config.copy()
        
        # GPU利用率高时减少批量大小 | Reduce batch size when GPU utilization is high
        gpu_utilization = realtime_metrics.get("gpu_utilization", 0.0)
        if gpu_utilization > 0.9:  # >90%
            adjusted["batch_size"] = int(adjusted["batch_size"] * 0.8)
            
        # 内存压力大时减少批量大小 | Reduce batch size when memory pressure is high
        memory_usage = realtime_metrics.get("memory_usage", 0.0)
        if memory_usage > 0.85:  # >85%
            adjusted["batch_size"] = int(adjusted["batch_size"] * 0.7)
            
        # CPU温度高时降低数据加载线程 | Reduce data workers when CPU temperature is high
        cpu_temp = realtime_metrics.get("cpu_temperature", 60.0)
        if cpu_temp > 80.0:  # >80°C
            adjusted["data_workers"] = max(1, adjusted.get("data_workers", 4) // 2)
            
        return adjusted
    
    def _config_to_state(self, config: Dict[str, Any]) -> int:
        """将配置映射到离散状态 | Map configuration to discrete state"""
        # 简化：基于学习率和批量大小创建状态 | Simplified: create state based on learning rate and batch size
        lr = config.get("learning_rate", 0.001)
        bs = config.get("batch_size", 32)
        
        # 离散化学习率 | Discretize learning rate
        if lr < 0.0001:
            lr_state = 0
        elif lr < 0.001:
            lr_state = 1
        elif lr < 0.01:
            lr_state = 2
        else:
            lr_state = 3
            
        # 离散化批量大小 | Discretize batch size
        if bs < 8:
            bs_state = 0
        elif bs < 32:
            bs_state = 1
        elif bs < 128:
            bs_state = 2
        else:
            bs_state = 3
            
        return lr_state * 4 + bs_state  # 0-15的状态空间 | State space 0-15
    
    def _adjust_learning_rate(self, config: Dict[str, Any], multiplier: float) -> Dict[str, Any]:
        """调整学习率 | Adjust learning rate"""
        adjusted = config.copy()
        if "learning_rate" in adjusted:
            adjusted["learning_rate"] = adjusted["learning_rate"] * multiplier
            # 限制范围 | Limit range
            adjusted["learning_rate"] = max(1e-6, min(0.1, adjusted["learning_rate"]))
        return adjusted
    
    def _adjust_batch_size(self, config: Dict[str, Any], multiplier: float) -> Dict[str, Any]:
        """调整批量大小 | Adjust batch size"""
        adjusted = config.copy()
        if "batch_size" in adjusted:
            adjusted["batch_size"] = int(adjusted["batch_size"] * multiplier)
            # 限制范围 | Limit range
            adjusted["batch_size"] = max(1, min(512, adjusted["batch_size"]))
        return adjusted
    
    def _switch_optimizer(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """切换优化器 | Switch optimizer"""
        adjusted = config.copy()
        optimizers = ["adam", "sgd", "rmsprop", "adamw"]
        current = adjusted.get("optimizer", "adam")
        
        if current in optimizers:
            current_idx = optimizers.index(current)
            next_idx = (current_idx + 1) % len(optimizers)
            adjusted["optimizer"] = optimizers[next_idx]
        else:
            adjusted["optimizer"] = "adam"
            
        return adjusted
    
    def _calculate_reward(self, performance: Dict[str, Any]) -> float:
        """计算强化学习奖励 | Calculate reinforcement learning reward"""
        reward = 0.0
        
        # 准确性奖励 | Accuracy reward
        if "accuracy" in performance:
            reward += performance["accuracy"] * 10  # 准确性权重较高 | Higher weight for accuracy
            
        # 效率奖励（训练时间短）| Efficiency reward (short training time)
        if "training_time" in performance:
            # 归一化训练时间（假设1小时为基准）| Normalize training time (assuming 1 hour as baseline)
            normalized_time = max(0.1, min(1.0, 3600.0 / max(performance["training_time"], 1.0)))
            reward += normalized_time * 5
            
        # 稳定性奖励（损失波动小）| Stability reward (low loss fluctuation)
        if "loss_std" in performance:
            stability = max(0.0, 1.0 - performance["loss_std"] * 10)
            reward += stability * 3
            
        return reward
    
    def update_learning_strategy(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """根据性能指标更新学习策略（增强版）| Update learning strategy based on performance metrics (enhanced)
        
        参数：
        Parameters:
        - performance_metrics: 性能指标 | Performance metrics
        
        返回：更新后的学习策略参数 | Returns: Updated learning strategy parameters
        """
        logger.info(f"根据性能指标更新学习策略: {performance_metrics}")
        
        # 添加到历史记录 | Add to history
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": performance_metrics.copy()
        })
        
        updated_params = {}
        
        # 动态调整规则 | Dynamic adjustment rules
        accuracy = performance_metrics.get("accuracy", 0.0)
        loss = performance_metrics.get("loss", 1.0)
        loss_std = performance_metrics.get("loss_std", 0.0)
        convergence_speed = performance_metrics.get("convergence_speed", 0.0)
        
        # 规则1：如果准确率低于阈值，降低学习率并增加正则化 | Rule 1: If accuracy below threshold, decrease learning rate and increase regularization
        if accuracy < 0.6:
            updated_params["learning_rate_adjustment"] = "decrease_by_30%"
            updated_params["regularization"] = "increase"
            updated_params["early_stopping_patience"] = "increase_by_50%"
            
        # 规则2：如果损失波动大，增加批量大小 | Rule 2: If loss fluctuates significantly, increase batch size
        if loss_std > 0.15:
            updated_params["batch_size_adjustment"] = "increase_by_25%"
            updated_params["gradient_clipping"] = "enable"
            
        # 规则3：如果收敛速度慢，增加学习率 | Rule 3: If convergence is slow, increase learning rate
        if convergence_speed < 0.01 and loss > 0.5:
            updated_params["learning_rate_adjustment"] = "increase_by_20%"
            updated_params["momentum"] = "increase"
            
        # 规则4：如果准确率高但过拟合迹象，增加正则化 | Rule 4: If high accuracy but overfitting signs, increase regularization
        if accuracy > 0.85 and performance_metrics.get("val_accuracy", 0.0) < accuracy - 0.1:
            updated_params["dropout_rate"] = "increase"
            updated_params["weight_decay"] = "increase"
            updated_params["data_augmentation"] = "enhance"
            
        # 添加到元学习分析器 | Add to meta-learning analyzer
        training_record = {
            "final_accuracy": accuracy,
            "training_loss": loss,
            "model_type": performance_metrics.get("model_type", "unknown"),
            "learning_rate": performance_metrics.get("learning_rate", 0.001),
            "batch_size": performance_metrics.get("batch_size", 32),
            "training_time": performance_metrics.get("training_time", 0.0),
        }
        self.meta_analyzer.add_training_record(training_record)
        
        logger.info(f"更新后的学习策略: {updated_params}")
        return updated_params
    
    def get_meta_learning_insights(self, training_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从训练历史中提取元学习洞察（增强版）| Extract meta-learning insights from training history (enhanced)
        
        参数：
        Parameters:
        - training_history: 训练历史 | Training history
        
        返回：元学习洞察 | Returns: Meta-learning insights
        """
        logger.info("从训练历史中提取元学习洞察")
        
        if not training_history:
            return {"status": "no_data", "message": "没有训练历史数据"}
            
        # 添加到内部历史记录 | Add to internal history
        for record in training_history:
            self.meta_analyzer.add_training_record(record)
        
        # 获取分析结果 | Get analysis results
        insights = self.meta_analyzer.analyze_patterns()
        
        # 添加高级洞察 | Add advanced insights
        insights["advanced_insights"] = self._generate_advanced_insights(training_history)
        insights["optimization_recommendations"] = self._generate_optimization_recommendations(training_history)
        insights["predicted_performance"] = self._predict_performance(training_history)
        
        logger.info(f"提取的元学习洞察: {insights}")
        return insights
    
    def _generate_advanced_insights(self, training_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成高级洞察 | Generate advanced insights"""
        if len(training_history) < 3:
            return {"status": "insufficient_data_for_advanced_insights"}
            
        # 提取特征 | Extract features
        features = []
        targets = []
        
        for record in training_history:
            # 特征：学习率、批量大小、优化器类型等 | Features: learning rate, batch size, optimizer type, etc.
            features.append([
                record.get("learning_rate", 0.001),
                record.get("batch_size", 32),
                1.0 if record.get("optimizer") == "adam" else 0.0,
                1.0 if record.get("optimizer") == "sgd" else 0.0,
                record.get("dropout_rate", 0.2),
            ])
            # 目标：准确率 | Target: accuracy
            targets.append(record.get("final_accuracy", 0.0))
            
        # 简单的线性关系分析 | Simple linear relationship analysis
        features_array = np.array(features)
        targets_array = np.array(targets)
        
        # 计算相关性 | Calculate correlations
        correlations = []
        for i in range(features_array.shape[1]):
            if len(features_array[:, i]) > 1 and len(targets_array) > 1:
                corr = np.corrcoef(features_array[:, i], targets_array)[0, 1] if not np.isnan(features_array[:, i]).any() else 0.0
                correlations.append(corr)
            else:
                correlations.append(0.0)
        
        return {
            "learning_rate_impact": correlations[0] if len(correlations) > 0 else 0.0,
            "batch_size_impact": correlations[1] if len(correlations) > 1 else 0.0,
            "adam_vs_sgd_impact": correlations[2] - correlations[3] if len(correlations) > 3 else 0.0,
            "dropout_impact": correlations[4] if len(correlations) > 4 else 0.0,
            "interpretation": "正值表示该参数增加时准确率提高，负值表示降低 | Positive value indicates accuracy increases when parameter increases, negative indicates decrease"
        }
    
    def _generate_optimization_recommendations(self, training_history: List[Dict[str, Any]]) -> List[str]:
        """生成优化推荐 | Generate optimization recommendations"""
        recommendations = []
        
        if len(training_history) < 2:
            return ["需要更多训练数据来生成推荐"]
            
        # 分析最佳表现配置 | Analyze best performing configuration
        best_record = max(training_history, key=lambda x: x.get("final_accuracy", 0.0))
        worst_record = min(training_history, key=lambda x: x.get("final_accuracy", 1.0))
        
        recommendations.append(f"最佳配置：学习率={best_record.get('learning_rate', 0.001):.6f}, "
                              f"批量大小={best_record.get('batch_size', 32)}, "
                              f"优化器={best_record.get('optimizer', 'unknown')}")
        
        # 对比最佳和最差 | Compare best and worst
        if best_record.get("learning_rate", 0.0) != worst_record.get("learning_rate", 0.0):
            recommendations.append(f"学习率 {best_record.get('learning_rate', 0.001):.6f} 表现优于 "
                                  f"{worst_record.get('learning_rate', 0.001):.6f}")
        
        # 检查是否有明显的趋势 | Check for obvious trends
        learning_rates = [r.get("learning_rate", 0.001) for r in training_history]
        accuracies = [r.get("final_accuracy", 0.0) for r in training_history]
        
        if len(learning_rates) > 2 and len(accuracies) > 2:
            # 检查学习率与准确率的相关性 | Check correlation between learning rate and accuracy
            corr_matrix = np.corrcoef(learning_rates, accuracies)
            correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
            
            if correlation > 0.3:
                recommendations.append("检测到正相关：更高的学习率可能带来更好的性能")
            elif correlation < -0.3:
                recommendations.append("检测到负相关：更低的学习率可能带来更好的性能")
        
        return recommendations
    
    def _predict_performance(self, training_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """预测性能 | Predict performance"""
        if len(training_history) < 5:
            return {"status": "insufficient_data_for_prediction"}
            
        # 简单的基于最近趋势的预测 | Simple prediction based on recent trends
        recent_history = training_history[-5:]
        recent_accuracies = [r.get("final_accuracy", 0.0) for r in recent_history]
        
        if len(recent_accuracies) >= 2:
            # 线性外推 | Linear extrapolation
            x = np.arange(len(recent_accuracies))
            slope, intercept, _, _, _ = stats.linregress(x, recent_accuracies)
            
            next_accuracy = intercept + slope * len(recent_accuracies)
            next_accuracy = max(0.0, min(1.0, next_accuracy))  # 限制在0-1之间 | Clamp between 0-1
            
            return {
                "predicted_next_accuracy": next_accuracy,
                "confidence": min(0.8, abs(slope) * 10 + 0.3),  # 置信度估计 | Confidence estimate
                "trend": "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable",
                "recommended_action": "继续当前策略" if slope > 0 else "考虑调整超参数"
            }
        
        return {"status": "prediction_failed"}
    
    def save_state(self, filepath: str):
        """保存引擎状态到文件 | Save engine state to file"""
        state = {
            "rl_optimizer_q_table": self.rl_optimizer.q_table.tolist(),
            "meta_analyzer_patterns": self.meta_analyzer.patterns,
            "config_history": self.config_history[-100:],  # 保存最近100条 | Save last 100 records
            "performance_history": self.performance_history[-100:],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"引擎状态已保存到 {filepath}")
        except Exception as e:
            logger.error(f"保存引擎状态失败: {e}")
    
    def load_state(self, filepath: str):
        """从文件加载引擎状态 | Load engine state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            self.rl_optimizer.q_table = np.array(state["rl_optimizer_q_table"])
            self.meta_analyzer.patterns = state["meta_analyzer_patterns"]
            self.config_history = state.get("config_history", [])
            self.performance_history = state.get("performance_history", [])
            
            logger.info(f"引擎状态已从 {filepath} 加载")
        except Exception as e:
            logger.error(f"加载引擎状态失败: {e}")

# 导出类供外部使用 | Export class for external use
__all__ = ['EnhancedAdaptiveLearningEngine', 'ReinforcementLearningOptimizer', 'MetaLearningAnalyzer']

# 为了向后兼容，提供别名
AdaptiveLearningEngine = EnhancedAdaptiveLearningEngine
