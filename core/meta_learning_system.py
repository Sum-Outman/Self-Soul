"""
元学习系统模块
实现深度元学习机制，使系统能够学习如何学习，适应新任务并改进学习策略
"""

import numpy as np
import json
import time
from typing import Dict, List, Any, Optional, Callable
import logging
from dataclasses import dataclass
import pickle
from pathlib import Path
import torch

# 导入AGI核心系统
from .agi_core import AGI_SYSTEM

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LearningEpisode:
    """学习经验数据类"""
    task_type: str
    strategy_used: str
    success_metric: float
    learning_time: float
    resources_used: Dict[str, float]
    insights_gained: List[str]
    timestamp: float

@dataclass
class MetaLearningState:
    """元学习状态数据类"""
    current_strategy: str
    strategy_performance: Dict[str, float]
    adaptation_rate: float
    learning_curve: List[float]
    knowledge_transfer_efficiency: float

class MetaLearningSystem:
    """
    元学习系统 - 实现高级元学习能力
    使系统能够学习如何学习，优化学习策略，并适应新任务
    """
    
    def __init__(self, from_scratch: bool = False):
        self.learning_history: List[LearningEpisode] = []
        self.meta_state = MetaLearningState(
            current_strategy="default",
            strategy_performance={},
            adaptation_rate=0.1,
            learning_curve=[],
            knowledge_transfer_efficiency=0.5
        )
        
        # 学习策略库
        self.learning_strategies = {
            "reinforcement": self._reinforcement_learning_strategy,
            "supervised": self._supervised_learning_strategy,
            "unsupervised": self._unsupervised_learning_strategy,
            "transfer": self._transfer_learning_strategy,
            "meta": self._meta_learning_strategy,
            "active": self._active_learning_strategy
        }
        
        # 性能指标
        self.performance_metrics = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "average_learning_time": 0.0,
            "strategy_effectiveness": {},
            "recent_improvement": 0.0
        }
        
        # 记录是否从零开始训练
        self.from_scratch = from_scratch
        
        # 根据是否从零开始训练决定是否加载历史数据
        if not from_scratch:
            self._load_learning_history()
        else:
            logger.info("从零开始训练模式 - 不加载元学习历史数据")
    
    def _load_learning_history(self):
        """加载学习历史数据"""
        history_file = Path("data/meta_learning_history.pkl")
        if history_file.exists():
            try:
                with open(history_file, 'rb') as f:
                    data = pickle.load(f)
                    self.learning_history = data.get('history', [])
                    self.meta_state = data.get('state', self.meta_state)
                    self.performance_metrics = data.get('metrics', self.performance_metrics)
                logger.info(f"加载了 {len(self.learning_history)} 条学习历史记录")
            except Exception as e:
                logger.warning(f"加载学习历史失败: {e}")
    
    def _save_learning_history(self):
        """保存学习历史数据"""
        try:
            data = {
                'history': self.learning_history,
                'state': self.meta_state,
                'metrics': self.performance_metrics
            }
            with open("data/meta_learning_history.pkl", 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"保存学习历史失败: {e}")
    
    def _reinforcement_learning_strategy(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """强化学习策略 - 使用神经网络优化"""
        # 使用神经网络生成最优强化学习参数
        neural_config = agi_core.optimize_learning_strategy(
            strategy_type="reinforcement",
            task_data=task_data,
            historical_performance=self.meta_state.strategy_performance
        )
        
        return {
            "strategy": "reinforcement",
            "learning_rate": neural_config.get("learning_rate", 0.01),
            "exploration_rate": neural_config.get("exploration_rate", 0.3),
            "reward_shaping": neural_config.get("reward_shaping", True),
            "value_iteration_steps": neural_config.get("value_iteration_steps", 100),
            "neural_optimized": True
        }
    
    def _supervised_learning_strategy(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """监督学习策略 - 使用神经网络优化"""
        # 使用神经网络生成最优监督学习参数
        neural_config = agi_core.optimize_learning_strategy(
            strategy_type="supervised",
            task_data=task_data,
            historical_performance=self.meta_state.strategy_performance
        )
        
        return {
            "strategy": "supervised",
            "learning_rate": neural_config.get("learning_rate", 0.001),
            "batch_size": neural_config.get("batch_size", 32),
            "epochs": neural_config.get("epochs", 10),
            "validation_split": neural_config.get("validation_split", 0.2),
            "neural_optimized": True
        }
    
    def _unsupervised_learning_strategy(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """无监督学习策略 - 使用神经网络优化"""
        # 使用神经网络生成最优无监督学习参数
        neural_config = agi_core.optimize_learning_strategy(
            strategy_type="unsupervised",
            task_data=task_data,
            historical_performance=self.meta_state.strategy_performance
        )
        
        return {
            "strategy": "unsupervised",
            "clustering_method": neural_config.get("clustering_method", "kmeans"),
            "dimensionality_reduction": neural_config.get("dimensionality_reduction", "pca"),
            "anomaly_detection": neural_config.get("anomaly_detection", True),
            "neural_optimized": True
        }
    
    def _transfer_learning_strategy(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """迁移学习策略 - 使用神经网络优化"""
        # 使用神经网络生成最优迁移学习参数
        neural_config = agi_core.optimize_learning_strategy(
            strategy_type="transfer",
            task_data=task_data,
            historical_performance=self.meta_state.strategy_performance
        )
        
        return {
            "strategy": "transfer",
            "source_domain": neural_config.get("source_domain", task_data.get('similar_tasks', ['general'])[0]),
            "fine_tuning": neural_config.get("fine_tuning", True),
            "feature_extraction": neural_config.get("feature_extraction", True),
            "adaptation_layers": neural_config.get("adaptation_layers", 2),
            "neural_optimized": True
        }
    
    def _meta_learning_strategy(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """元学习策略 - 使用神经网络优化"""
        # 使用神经网络生成最优元学习参数
        neural_config = agi_core.optimize_learning_strategy(
            strategy_type="meta",
            task_data=task_data,
            historical_performance=self.meta_state.strategy_performance
        )
        
        return {
            "strategy": "meta",
            "meta_learning_rate": neural_config.get("meta_learning_rate", 0.001),
            "inner_loop_steps": neural_config.get("inner_loop_steps", 5),
            "outer_loop_steps": neural_config.get("outer_loop_steps", 3),
            "gradient_clipping": neural_config.get("gradient_clipping", True),
            "neural_optimized": True
        }
    
    def _active_learning_strategy(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """主动学习策略 - 使用神经网络优化"""
        # 使用神经网络生成最优主动学习参数
        neural_config = agi_core.optimize_learning_strategy(
            strategy_type="active",
            task_data=task_data,
            historical_performance=self.meta_state.strategy_performance
        )
        
        return {
            "strategy": "active",
            "query_strategy": neural_config.get("query_strategy", "uncertainty_sampling"),
            "batch_size": neural_config.get("batch_size", 16),
            "max_queries": neural_config.get("max_queries", 100),
            "diversity_measure": neural_config.get("diversity_measure", "cosine"),
            "neural_optimized": True
        }
    
    def select_learning_strategy(self, task_description: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        选择最佳学习策略基于任务描述和历史经验
        """
        # 分析任务类型
        task_type = self._analyze_task_type(task_description, task_data)
        
        # 基于历史性能选择策略
        best_strategy = self._get_best_strategy_for_task_type(task_type)
        
        # 获取策略配置
        strategy_config = self.learning_strategies[best_strategy](task_data)
        
        return {
            "selected_strategy": best_strategy,
            "strategy_config": strategy_config,
            "confidence": self._calculate_strategy_confidence(best_strategy, task_type),
            "task_type": task_type
        }
    
    def _analyze_task_type(self, task_description: str, task_data: Dict[str, Any]) -> str:
        """分析任务类型"""
        description = task_description.lower()
        
        if any(word in description for word in ["分类", "识别", "预测", "回归"]):
            return "supervised"
        elif any(word in description for word in ["聚类", "分组", "模式发现", "异常检测"]):
            return "unsupervised"
        elif any(word in description for word in ["决策", "控制", "优化", "奖励"]):
            return "reinforcement"
        elif any(word in description for word in ["迁移", "适应", "类似任务"]):
            return "transfer"
        elif any(word in description for word in ["元学习", "学习如何学习", "快速适应"]):
            return "meta"
        elif any(word in description for word in ["主动", "查询", "交互"]):
            return "active"
        else:
            return "general"
    
    def _get_best_strategy_for_task_type(self, task_type: str) -> str:
        """基于历史性能获取最佳策略"""
        # 初始化策略性能
        if not self.meta_state.strategy_performance:
            for strategy in self.learning_strategies.keys():
                self.meta_state.strategy_performance[strategy] = 0.7  # 默认置信度
        
        # 选择性能最高的策略
        best_strategy = max(
            self.meta_state.strategy_performance.items(),
            key=lambda x: x[1]
        )[0]
        
        # 偶尔探索新策略（10%的概率）
        if np.random.random() < 0.1:
            exploration_strategy = np.random.choice(list(self.learning_strategies.keys()))
            if self.meta_state.strategy_performance[exploration_strategy] > 0.5:
                best_strategy = exploration_strategy
        
        return best_strategy
    
    def _calculate_strategy_confidence(self, strategy: str, task_type: str) -> float:
        """计算策略置信度"""
        base_confidence = self.meta_state.strategy_performance.get(strategy, 0.7)
        
        # 根据任务类型调整置信度
        type_match_bonus = 0.2 if strategy == task_type else 0.0
        
        return min(1.0, base_confidence + type_match_bonus)
    
    def record_learning_episode(self, episode: LearningEpisode):
        """记录学习经验"""
        self.learning_history.append(episode)
        self.performance_metrics["total_episodes"] += 1
        
        # 更新策略性能
        if episode.success_metric > 0.7:  # 成功阈值
            self.performance_metrics["successful_episodes"] += 1
            improvement = (episode.success_metric - 0.7) * 0.1
            self.meta_state.strategy_performance[episode.strategy_used] = min(
                1.0, self.meta_state.strategy_performance.get(episode.strategy_used, 0.7) + improvement
            )
        else:
            penalty = (0.7 - episode.success_metric) * 0.05
            self.meta_state.strategy_performance[episode.strategy_used] = max(
                0.3, self.meta_state.strategy_performance.get(episode.strategy_used, 0.7) - penalty
            )
        
        # 更新学习曲线
        self.meta_state.learning_curve.append(episode.success_metric)
        if len(self.meta_state.learning_curve) > 100:
            self.meta_state.learning_curve = self.meta_state.learning_curve[-100:]
        
        # 更新适应率（基于最近表现）
        recent_performance = np.mean(self.meta_state.learning_curve[-10:]) if len(self.meta_state.learning_curve) >= 10 else 0.7
        self.meta_state.adaptation_rate = 0.1 + (recent_performance - 0.7) * 0.5
        
        # 更新知识迁移效率
        similar_episodes = [e for e in self.learning_history if e.task_type == episode.task_type]
        if len(similar_episodes) > 1:
            improvements = [e.success_metric for e in similar_episodes]
            self.meta_state.knowledge_transfer_efficiency = np.mean(improvements) / max(improvements)
        
        # 保存历史
        self._save_learning_history()
        
        logger.info(f"记录学习经验: {episode.task_type}, 策略: {episode.strategy_used}, 成功率: {episode.success_metric:.2f}")
    
    def optimize_learning_parameters(self, current_params: Dict[str, Any], performance_feedback: float) -> Dict[str, Any]:
        """优化学习参数"""
        optimized_params = current_params.copy()
        
        # 基于性能反馈调整参数
        if performance_feedback > 0.8:
            # 表现良好，小幅优化
            if 'learning_rate' in optimized_params:
                optimized_params['learning_rate'] *= 1.1
            if 'exploration_rate' in optimized_params:
                optimized_params['exploration_rate'] *= 0.9
        elif performance_feedback < 0.6:
            # 表现不佳，较大调整
            if 'learning_rate' in optimized_params:
                optimized_params['learning_rate'] *= 0.8
            if 'exploration_rate' in optimized_params:
                optimized_params['exploration_rate'] *= 1.2
        
        return optimized_params
    
    def generate_learning_insights(self) -> List[str]:
        """生成学习洞察"""
        insights = []
        
        # 分析最近的学习模式
        recent_episodes = self.learning_history[-10:] if len(self.learning_history) >= 10 else self.learning_history
        
        if recent_episodes:
            success_rate = np.mean([e.success_metric for e in recent_episodes])
            avg_time = np.mean([e.learning_time for e in recent_episodes])
            
            insights.append(f"最近学习成功率: {success_rate:.2f}")
            insights.append(f"平均学习时间: {avg_time:.2f}秒")
            
            # 策略效果分析
            strategy_perf = {}
            for episode in recent_episodes:
                if episode.strategy_used not in strategy_perf:
                    strategy_perf[episode.strategy_used] = []
                strategy_perf[episode.strategy_used].append(episode.success_metric)
            
            for strategy, perfs in strategy_perf.items():
                avg_perf = np.mean(perfs)
                insights.append(f"策略 '{strategy}' 平均效果: {avg_perf:.2f}")
        
        return insights
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "total_episodes": self.performance_metrics["total_episodes"],
            "success_rate": self.performance_metrics["successful_episodes"] / max(1, self.performance_metrics["total_episodes"]),
            "strategy_performance": self.meta_state.strategy_performance,
            "adaptation_rate": self.meta_state.adaptation_rate,
            "knowledge_transfer_efficiency": self.meta_state.knowledge_transfer_efficiency,
            "learning_curve_length": len(self.meta_state.learning_curve)
        }

# 单例实例
meta_learning_system = MetaLearningSystem()

if __name__ == "__main__":
    # 测试代码
    mls = MetaLearningSystem()
    
    print("=== 测试元学习系统 ===")
    
    # 测试策略选择
    task_desc = "图像分类任务"
    task_data = {"dataset_size": 1000, "similar_tasks": ["物体识别"]}
    
    strategy = mls.select_learning_strategy(task_desc, task_data)
    print(f"选择的学习策略: {strategy}")
    
    # 记录学习经验
    episode = LearningEpisode(
        task_type="supervised",
        strategy_used="supervised",
        success_metric=0.85,
        learning_time=120.5,
        resources_used={"cpu": 0.7, "memory": 0.6},
        insights_gained=["使用数据增强提高泛化能力"],
        timestamp=time.time()
    )
    mls.record_learning_episode(episode)
    
    # 显示系统统计
    stats = mls.get_system_stats()
    print("\n=== 系统统计 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 生成学习洞察
    insights = mls.generate_learning_insights()
    print("\n=== 学习洞察 ===")
    for insight in insights:
        print(f"- {insight}")
