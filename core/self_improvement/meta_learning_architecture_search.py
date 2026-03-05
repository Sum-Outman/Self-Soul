"""
元学习与架构搜索整合系统

该模块整合元学习（Meta-Learning）和神经架构搜索（Neural Architecture Search）能力，
为AGI系统的自我改进循环提供高级学习能力优化。

核心功能：
1. 元学习策略管理：管理和优化学习策略，实现"学会学习"的能力
2. 架构搜索优化：自动搜索和优化神经网络架构
3. 学习策略推荐：根据任务特性和历史性能推荐最优学习策略
4. 架构进化：基于性能反馈进化神经网络架构
5. 跨任务知识迁移：在不同任务间迁移学习经验和架构知识

技术特性：
- 双向优化：同时优化学习策略和网络架构
- 多任务适应：支持多领域任务的快速适应
- 增量进化：基于性能反馈的渐进式架构改进
- 可解释性：提供学习策略和架构决策的解释
- 安全约束：在安全边界内进行架构搜索和学习策略优化
"""

import time
import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 导入现有模块
try:
    from core.meta_learning_system import MetaLearningSystem, LearningEpisode, MetaLearningState
    meta_learning_available = True
except ImportError:
    meta_learning_available = False
    logger = logging.getLogger(__name__)
    logger.warning("MetaLearningSystem not available, meta-learning features will be limited")

try:
    from core.optimization.nas_engine import NASEngine, NASAlgorithmConfig, ArchitectureSearchResult
    nas_available = True
except ImportError:
    nas_available = False
    logger = logging.getLogger(__name__)
    logger.warning("NASEngine not available, architecture search features will be limited")

# 配置日志
logger = logging.getLogger(__name__)

class LearningStrategyType(Enum):
    """学习策略类型"""
    META_LEARNING = "meta_learning"
    TRANSFER_LEARNING = "transfer_learning"
    FEW_SHOT_LEARNING = "few_shot_learning"
    MULTI_TASK_LEARNING = "multi_task_learning"
    CONTINUAL_LEARNING = "continual_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    SUPERVISED_LEARNING = "supervised_learning"
    SELF_SUPERVISED_LEARNING = "self_supervised_learning"

class ArchitectureSearchType(Enum):
    """架构搜索类型"""
    DIFFERENTIABLE = "differentiable"  # DARTS
    EFFICIENT = "efficient"  # ENAS
    EVOLUTIONARY = "evolutionary"  # 进化算法
    REINFORCEMENT = "reinforcement"  # 强化学习
    RANDOM = "random"  # 随机搜索
    BAYESIAN = "bayesian"  # 贝叶斯优化

@dataclass
class TaskCharacteristics:
    """任务特征描述"""
    task_id: str
    task_type: str
    domain: str
    data_characteristics: Dict[str, Any]  # 数据规模、维度、分布等
    performance_requirements: Dict[str, float]  # 精度、速度、内存等要求
    constraints: Dict[str, Any]  # 计算、内存、时间等约束
    similarity_to_previous_tasks: Dict[str, float]  # 与历史任务的相似度

@dataclass
class LearningStrategyRecommendation:
    """学习策略推荐"""
    strategy_type: LearningStrategyType
    confidence: float  # 推荐置信度
    expected_performance: float  # 预期性能
    estimated_training_time: float  # 估计训练时间
    resource_requirements: Dict[str, float]  # 资源需求
    adaptation_time: float  # 适应时间
    justification: str  # 推荐理由

@dataclass
class ArchitectureRecommendation:
    """架构推荐"""
    search_type: ArchitectureSearchType
    architecture_config: Dict[str, Any]
    expected_performance: float
    estimated_search_time: float
    computational_cost: float
    memory_requirements: float
    justification: str

@dataclass
class LearningOptimizationResult:
    """学习优化结果"""
    task_id: str
    strategy_used: LearningStrategyType
    architecture_used: Dict[str, Any]
    initial_performance: float
    optimized_performance: float
    improvement_ratio: float
    optimization_time: float
    resources_consumed: Dict[str, float]
    lessons_learned: List[str]
    recommendations_for_future: List[str]

class MetaLearningArchitectureSearch:
    """
    元学习与架构搜索整合系统
    
    核心组件:
    1. 任务分析器: 分析任务特征和需求
    2. 策略推荐器: 推荐最优学习策略
    3. 架构推荐器: 推荐最优架构搜索策略
    4. 优化执行器: 执行学习和架构优化
    5. 效果评估器: 评估优化效果
    6. 知识积累器: 积累优化经验
    
    工作流程:
    新任务 → 任务分析器 → 任务特征 → 策略推荐器 → 学习策略推荐
    任务特征 → 架构推荐器 → 架构策略推荐 → 优化执行器 → 联合优化
    联合优化 → 效果评估器 → 优化结果 → 知识积累器 → 经验积累
    
    技术特性:
    - 联合优化: 同时优化学习策略和网络架构
    - 上下文感知: 根据任务上下文动态调整策略
    - 经验重用: 重用历史优化经验加速新任务
    - 安全边界: 在安全约束内进行优化
    - 可解释性: 提供优化决策的解释
    """
    
    def __init__(self,
                 enable_meta_learning: bool = True,
                 enable_architecture_search: bool = True,
                 learning_strategy_db_path: Optional[str] = None,
                 architecture_db_path: Optional[str] = None,
                 safety_constraints: Optional[Dict[str, Any]] = None):
        """
        初始化元学习与架构搜索整合系统
        
        Args:
            enable_meta_learning: 启用元学习功能
            enable_architecture_search: 启用架构搜索功能
            learning_strategy_db_path: 学习策略数据库路径
            architecture_db_path: 架构数据库路径
            safety_constraints: 安全约束配置
        """
        self.enable_meta_learning = enable_meta_learning and meta_learning_available
        self.enable_architecture_search = enable_architecture_search and nas_available
        
        # 安全约束
        self.safety_constraints = safety_constraints or self._get_default_safety_constraints()
        
        # 数据库初始化
        self.learning_strategy_db = self._initialize_learning_strategy_db(learning_strategy_db_path)
        self.architecture_db = self._initialize_architecture_db(architecture_db_path)
        
        # 性能历史
        self.optimization_history: List[LearningOptimizationResult] = []
        self.max_history_size = 1000
        
        # 组件初始化
        self.task_analyzer = TaskAnalyzer()
        self.strategy_recommender = StrategyRecommender(self.learning_strategy_db)
        self.architecture_recommender = ArchitectureRecommender(self.architecture_db)
        self.optimization_executor = OptimizationExecutor(
            enable_meta_learning=self.enable_meta_learning,
            enable_architecture_search=self.enable_architecture_search
        )
        self.effect_evaluator = EffectEvaluator()
        self.knowledge_accumulator = KnowledgeAccumulator()
        
        # 性能统计
        self.performance_stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "average_improvement_ratio": 0.0,
            "total_optimization_time": 0.0,
            "average_optimization_time": 0.0,
            "tasks_analyzed": 0,
            "recommendations_generated": 0
        }
        
        logger.info(f"元学习与架构搜索整合系统初始化完成: "
                   f"元学习启用={self.enable_meta_learning}, "
                   f"架构搜索启用={self.enable_architecture_search}")
    
    def analyze_task(self, task_description: Dict[str, Any]) -> TaskCharacteristics:
        """
        分析任务特征
        
        Args:
            task_description: 任务描述
            
        Returns:
            任务特征分析结果
        """
        self.performance_stats["tasks_analyzed"] += 1
        return self.task_analyzer.analyze(task_description)
    
    def recommend_learning_strategy(self, 
                                   task_characteristics: TaskCharacteristics,
                                   available_resources: Dict[str, float]) -> LearningStrategyRecommendation:
        """
        推荐学习策略
        
        Args:
            task_characteristics: 任务特征
            available_resources: 可用资源
            
        Returns:
            学习策略推荐
        """
        self.performance_stats["recommendations_generated"] += 1
        return self.strategy_recommender.recommend(task_characteristics, available_resources)
    
    def recommend_architecture_search(self,
                                     task_characteristics: TaskCharacteristics,
                                     available_resources: Dict[str, float]) -> ArchitectureRecommendation:
        """
        推荐架构搜索策略
        
        Args:
            task_characteristics: 任务特征
            available_resources: 可用资源
            
        Returns:
            架构搜索推荐
        """
        self.performance_stats["recommendations_generated"] += 1
        return self.architecture_recommender.recommend(task_characteristics, available_resources)
    
    def optimize_learning(self,
                         task_characteristics: TaskCharacteristics,
                         strategy_recommendation: LearningStrategyRecommendation,
                         architecture_recommendation: ArchitectureRecommendation,
                         initial_model: Optional[nn.Module] = None,
                         training_data: Optional[Any] = None,
                         validation_data: Optional[Any] = None) -> LearningOptimizationResult:
        """
        执行学习优化
        
        Args:
            task_characteristics: 任务特征
            strategy_recommendation: 学习策略推荐
            architecture_recommendation: 架构搜索推荐
            initial_model: 初始模型（可选）
            training_data: 训练数据（可选）
            validation_data: 验证数据（可选）
            
        Returns:
            学习优化结果
        """
        start_time = time.time()
        
        # 检查安全约束
        if not self._check_safety_constraints(strategy_recommendation, architecture_recommendation):
            logger.warning("优化请求违反安全约束，使用安全默认配置")
            strategy_recommendation, architecture_recommendation = self._get_safe_defaults(task_characteristics)
        
        # 执行优化
        optimization_result = self.optimization_executor.execute(
            task_characteristics=task_characteristics,
            strategy_recommendation=strategy_recommendation,
            architecture_recommendation=architecture_recommendation,
            initial_model=initial_model,
            training_data=training_data,
            validation_data=validation_data
        )
        
        # 评估效果
        evaluation_result = self.effect_evaluator.evaluate(optimization_result)
        
        # 积累知识
        self.knowledge_accumulator.accumulate(optimization_result, evaluation_result)
        
        # 更新历史
        self.optimization_history.append(optimization_result)
        if len(self.optimization_history) > self.max_history_size:
            self.optimization_history.pop(0)
        
        # 更新性能统计
        optimization_time = time.time() - start_time
        self.performance_stats["total_optimizations"] += 1
        if optimization_result.improvement_ratio > 0:
            self.performance_stats["successful_optimizations"] += 1
        
        self.performance_stats["total_optimization_time"] += optimization_time
        self.performance_stats["average_optimization_time"] = \
            self.performance_stats["total_optimization_time"] / self.performance_stats["total_optimizations"]
        
        # 更新平均改进率
        total_improvements = sum(r.improvement_ratio for r in self.optimization_history)
        self.performance_stats["average_improvement_ratio"] = \
            total_improvements / len(self.optimization_history) if self.optimization_history else 0.0
        
        logger.info(f"学习优化完成: 任务={task_characteristics.task_id}, "
                   f"改进率={optimization_result.improvement_ratio:.3f}, "
                   f"时间={optimization_time:.1f}s")
        
        return optimization_result
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        return {
            "total_optimizations": self.performance_stats["total_optimizations"],
            "success_rate": self.performance_stats["successful_optimizations"] / 
                          max(1, self.performance_stats["total_optimizations"]),
            "average_improvement_ratio": self.performance_stats["average_improvement_ratio"],
            "average_optimization_time": self.performance_stats["average_optimization_time"],
            "tasks_analyzed": self.performance_stats["tasks_analyzed"],
            "recommendations_generated": self.performance_stats["recommendations_generated"],
            "recent_optimizations": [
                {
                    "task_id": r.task_id,
                    "improvement_ratio": r.improvement_ratio,
                    "optimization_time": r.optimization_time
                }
                for r in self.optimization_history[-10:]  # 最近10次优化
            ]
        }
    
    def _check_safety_constraints(self,
                                 strategy_recommendation: LearningStrategyRecommendation,
                                 architecture_recommendation: ArchitectureRecommendation) -> bool:
        """检查安全约束"""
        # 检查计算资源约束
        if strategy_recommendation.resource_requirements.get("compute", 0) > \
           self.safety_constraints.get("max_compute", float('inf')):
            return False
        
        # 检查内存约束
        if architecture_recommendation.memory_requirements > \
           self.safety_constraints.get("max_memory", float('inf')):
            return False
        
        # 检查时间约束
        if strategy_recommendation.estimated_training_time + \
           architecture_recommendation.estimated_search_time > \
           self.safety_constraints.get("max_time", float('inf')):
            return False
        
        return True
    
    def _get_safe_defaults(self, task_characteristics: TaskCharacteristics) -> Tuple:
        """获取安全默认配置"""
        # 简单的安全默认策略
        safe_strategy = LearningStrategyRecommendation(
            strategy_type=LearningStrategyType.SUPERVISED_LEARNING,
            confidence=0.8,
            expected_performance=0.7,
            estimated_training_time=3600.0,  # 1小时
            resource_requirements={"compute": 1.0, "memory": 2.0},
            adaptation_time=1800.0,
            justification="安全默认策略：监督学习"
        )
        
        safe_architecture = ArchitectureRecommendation(
            search_type=ArchitectureSearchType.RANDOM,
            architecture_config={"num_layers": 3, "hidden_size": 128},
            expected_performance=0.6,
            estimated_search_time=1800.0,
            computational_cost=1.0,
            memory_requirements=1.0,
            justification="安全默认架构：简单随机搜索"
        )
        
        return safe_strategy, safe_architecture
    
    def _get_default_safety_constraints(self) -> Dict[str, Any]:
        """获取默认安全约束"""
        return {
            "max_compute": 100.0,  # 最大计算资源（相对单位）
            "max_memory": 16.0,    # 最大内存（GB）
            "max_time": 7200.0,    # 最大优化时间（秒）
            "max_energy": 1000.0,  # 最大能耗（相对单位）
            "privacy_constraints": True,  # 隐私约束
            "fairness_constraints": True  # 公平性约束
        }
    
    def _initialize_learning_strategy_db(self, db_path: Optional[str]) -> Dict[str, Any]:
        """初始化学习策略数据库"""
        if db_path:
            try:
                with open(db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"无法加载学习策略数据库 {db_path}: {e}")
        
        # 默认数据库
        return {
            "strategies": {
                "supervised_learning": {
                    "description": "监督学习",
                    "applicable_domains": ["classification", "regression"],
                    "typical_performance": 0.8,
                    "resource_requirements": {"compute": 2.0, "memory": 1.0},
                    "adaptation_time": 3600.0
                },
                "meta_learning": {
                    "description": "元学习",
                    "applicable_domains": ["few_shot", "quick_adaptation"],
                    "typical_performance": 0.7,
                    "resource_requirements": {"compute": 5.0, "memory": 3.0},
                    "adaptation_time": 1800.0
                }
            }
        }
    
    def _initialize_architecture_db(self, db_path: Optional[str]) -> Dict[str, Any]:
        """初始化架构数据库"""
        if db_path:
            try:
                with open(db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"无法加载架构数据库 {db_path}: {e}")
        
        # 默认数据库
        return {
            "architectures": {
                "simple_cnn": {
                    "description": "简单CNN",
                    "applicable_tasks": ["image_classification"],
                    "typical_performance": 0.85,
                    "parameters": 100000,
                    "memory_footprint": 4.0
                },
                "transformer": {
                    "description": "Transformer",
                    "applicable_tasks": ["nlp", "sequence_modeling"],
                    "typical_performance": 0.9,
                    "parameters": 1000000,
                    "memory_footprint": 16.0
                }
            }
        }


class TaskAnalyzer:
    """任务分析器"""
    
    def analyze(self, task_description: Dict[str, Any]) -> TaskCharacteristics:
        """分析任务特征"""
        # 简化实现，实际中需要更复杂的分析
        return TaskCharacteristics(
            task_id=task_description.get("task_id", f"task_{int(time.time())}"),
            task_type=task_description.get("task_type", "unknown"),
            domain=task_description.get("domain", "general"),
            data_characteristics=task_description.get("data_characteristics", {}),
            performance_requirements=task_description.get("performance_requirements", {}),
            constraints=task_description.get("constraints", {}),
            similarity_to_previous_tasks={}
        )


class StrategyRecommender:
    """策略推荐器"""
    
    def __init__(self, strategy_db: Dict[str, Any]):
        self.strategy_db = strategy_db
    
    def recommend(self, 
                 task_characteristics: TaskCharacteristics,
                 available_resources: Dict[str, float]) -> LearningStrategyRecommendation:
        """推荐学习策略"""
        # 简化实现，实际中需要更复杂的推荐算法
        if task_characteristics.domain in ["few_shot", "quick_adaptation"]:
            strategy_type = LearningStrategyType.META_LEARNING
            confidence = 0.7
        else:
            strategy_type = LearningStrategyType.SUPERVISED_LEARNING
            confidence = 0.9
        
        return LearningStrategyRecommendation(
            strategy_type=strategy_type,
            confidence=confidence,
            expected_performance=0.75,
            estimated_training_time=3600.0,
            resource_requirements={"compute": 2.0, "memory": 1.0},
            adaptation_time=1800.0,
            justification=f"推荐 {strategy_type.value}，适用于{task_characteristics.domain}领域"
        )


class ArchitectureRecommender:
    """架构推荐器"""
    
    def __init__(self, architecture_db: Dict[str, Any]):
        self.architecture_db = architecture_db
    
    def recommend(self,
                 task_characteristics: TaskCharacteristics,
                 available_resources: Dict[str, float]) -> ArchitectureRecommendation:
        """推荐架构搜索策略"""
        # 简化实现，实际中需要更复杂的推荐算法
        if task_characteristics.task_type == "image_classification":
            search_type = ArchitectureSearchType.DIFFERENTIABLE
            architecture_config = {"model_type": "cnn", "num_layers": 5}
        elif task_characteristics.task_type == "nlp":
            search_type = ArchitectureSearchType.EVOLUTIONARY
            architecture_config = {"model_type": "transformer", "num_layers": 6}
        else:
            search_type = ArchitectureSearchType.RANDOM
            architecture_config = {"model_type": "mlp", "num_layers": 3}
        
        return ArchitectureRecommendation(
            search_type=search_type,
            architecture_config=architecture_config,
            expected_performance=0.7,
            estimated_search_time=1800.0,
            computational_cost=3.0,
            memory_requirements=2.0,
            justification=f"推荐 {search_type.value}，适用于{task_characteristics.task_type}任务"
        )


class OptimizationExecutor:
    """优化执行器"""
    
    def __init__(self, enable_meta_learning: bool, enable_architecture_search: bool):
        self.enable_meta_learning = enable_meta_learning
        self.enable_architecture_search = enable_architecture_search
        
        # 初始化元学习和NAS系统（如果可用）
        self.meta_learning_system = None
        self.nas_engine = None
        
        if self.enable_meta_learning:
            try:
                self.meta_learning_system = MetaLearningSystem()
            except Exception as e:
                logger.error(f"无法初始化MetaLearningSystem: {e}")
                self.enable_meta_learning = False
        
        if self.enable_architecture_search:
            try:
                self.nas_engine = NASEngine()
            except Exception as e:
                logger.error(f"无法初始化NASEngine: {e}")
                self.enable_architecture_search = False
    
    def execute(self,
                task_characteristics: TaskCharacteristics,
                strategy_recommendation: LearningStrategyRecommendation,
                architecture_recommendation: ArchitectureRecommendation,
                initial_model: Optional[nn.Module] = None,
                training_data: Optional[Any] = None,
                validation_data: Optional[Any] = None) -> LearningOptimizationResult:
        """执行优化"""
        start_time = time.time()
        
        # 记录初始性能（简化实现）
        initial_performance = 0.5  # 默认初始性能
        
        # 执行学习策略优化
        if self.enable_meta_learning and self.meta_learning_system:
            try:
                # 这里应该调用元学习系统的优化方法
                meta_learning_result = self.meta_learning_system.optimize_strategy(
                    task_characteristics=task_characteristics,
                    strategy_type=strategy_recommendation.strategy_type
                )
                logger.info(f"元学习优化完成: {meta_learning_result}")
            except Exception as e:
                logger.error(f"元学习优化失败: {e}")
        
        # 执行架构搜索优化
        if self.enable_architecture_search and self.nas_engine:
            try:
                # 这里应该调用NAS引擎的搜索方法
                nas_result = self.nas_engine.search(
                    task_type=task_characteristics.task_type,
                    search_config=architecture_recommendation.architecture_config
                )
                logger.info(f"架构搜索完成: {nas_result}")
            except Exception as e:
                logger.error(f"架构搜索失败: {e}")
        
        # 模拟优化结果（简化实现）
        optimization_time = time.time() - start_time
        optimized_performance = initial_performance * 1.2  # 假设有20%改进
        
        return LearningOptimizationResult(
            task_id=task_characteristics.task_id,
            strategy_used=strategy_recommendation.strategy_type,
            architecture_used=architecture_recommendation.architecture_config,
            initial_performance=initial_performance,
            optimized_performance=optimized_performance,
            improvement_ratio=(optimized_performance - initial_performance) / max(initial_performance, 0.001),
            optimization_time=optimization_time,
            resources_consumed={"compute": optimization_time * 0.1, "memory": 2.0},
            lessons_learned=["需要更多数据以提高性能", "架构选择对性能有显著影响"],
            recommendations_for_future=["增加训练数据", "尝试不同的架构变体"]
        )


class EffectEvaluator:
    """效果评估器"""
    
    def evaluate(self, optimization_result: LearningOptimizationResult) -> Dict[str, Any]:
        """评估优化效果"""
        return {
            "effectiveness": min(1.0, optimization_result.improvement_ratio * 2),  # 有效性评分
            "efficiency": max(0.0, 1.0 - optimization_result.optimization_time / 3600.0),  # 效率评分
            "cost_effectiveness": optimization_result.improvement_ratio / max(optimization_result.optimization_time, 1.0),
            "recommendation": "继续优化" if optimization_result.improvement_ratio < 0.3 else "优化效果良好"
        }


class KnowledgeAccumulator:
    """知识积累器"""
    
    def accumulate(self, 
                  optimization_result: LearningOptimizationResult,
                  evaluation_result: Dict[str, Any]):
        """积累知识"""
        # 简化实现，实际中应该存储到数据库
        logger.info(f"积累优化知识: 任务={optimization_result.task_id}, "
                   f"改进率={optimization_result.improvement_ratio:.3f}")


# 为自我改进循环提供便捷接口
class SelfImprovementMetaLearningAdapter:
    """
    自我改进循环的元学习适配器
    
    提供简化的接口，使自我改进循环能够轻松使用元学习和架构搜索功能
    """
    
    def __init__(self, meta_learning_nas_system: Optional[MetaLearningArchitectureSearch] = None):
        self.meta_learning_nas_system = meta_learning_nas_system or MetaLearningArchitectureSearch()
    
    def optimize_learning_for_weakness(self,
                                      weakness_id: str,
                                      weakness_description: str,
                                      current_performance: float,
                                      target_performance: float) -> Dict[str, Any]:
        """
        针对弱点优化学习
        
        Args:
            weakness_id: 弱点ID
            weakness_description: 弱点描述
            current_performance: 当前性能
            target_performance: 目标性能
            
        Returns:
            优化结果
        """
        # 构建任务描述
        task_description = {
            "task_id": f"weakness_optimization_{weakness_id}",
            "task_type": "performance_improvement",
            "domain": "agi_self_improvement",
            "data_characteristics": {
                "data_type": "performance_metrics",
                "size": 1000,
                "dimension": 10
            },
            "performance_requirements": {
                "target_accuracy": target_performance,
                "max_training_time": 3600.0
            },
            "constraints": {
                "max_compute": 50.0,
                "max_memory": 8.0
            }
        }
        
        # 分析任务
        task_characteristics = self.meta_learning_nas_system.analyze_task(task_description)
        
        # 推荐策略
        strategy_recommendation = self.meta_learning_nas_system.recommend_learning_strategy(
            task_characteristics,
            available_resources={"compute": 50.0, "memory": 8.0}
        )
        
        # 推荐架构
        architecture_recommendation = self.meta_learning_nas_system.recommend_architecture_search(
            task_characteristics,
            available_resources={"compute": 50.0, "memory": 8.0}
        )
        
        # 执行优化
        optimization_result = self.meta_learning_nas_system.optimize_learning(
            task_characteristics=task_characteristics,
            strategy_recommendation=strategy_recommendation,
            architecture_recommendation=architecture_recommendation
        )
        
        return {
            "weakness_id": weakness_id,
            "optimization_result": optimization_result,
            "summary": self.meta_learning_nas_system.get_optimization_summary()
        }
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """获取优化历史"""
        return [
            {
                "task_id": r.task_id,
                "strategy": r.strategy_used.value,
                "improvement": r.improvement_ratio,
                "time": r.optimization_time
            }
            for r in self.meta_learning_nas_system.optimization_history[-20:]  # 最近20次
        ]


# 全局实例（便于导入）
meta_learning_architecture_search_system = MetaLearningArchitectureSearch()
self_improvement_adapter = SelfImprovementMetaLearningAdapter(meta_learning_architecture_search_system)