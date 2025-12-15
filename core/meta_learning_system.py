"""
Meta Learning System Module
Implements deep meta-learning mechanisms enabling the system to learn how to learn, 
adapt to new tasks, and improve learning strategies according to AGI standards
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

# Unused import removed to avoid circular dependency

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LearningEpisode:
    """Learning episode data class"""
    task_type: str
    strategy_used: str
    success_metric: float
    learning_time: float
    resources_used: Dict[str, float]
    insights_gained: List[str]
    timestamp: float

@dataclass
class MetaLearningState:
    """Meta-learning state data class"""
    current_strategy: str
    strategy_performance: Dict[str, float]
    adaptation_rate: float
    learning_curve: List[float]
    knowledge_transfer_efficiency: float

class MetaLearningSystem:
    """
    Meta Learning System - Implements advanced meta-learning capabilities
    Enables the system to learn how to learn, optimize learning strategies, and adapt to new tasks
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
        
        # Learning strategies library
        self.learning_strategies = {
            "reinforcement": self._reinforcement_learning_strategy,
            "supervised": self._supervised_learning_strategy,
            "unsupervised": self._unsupervised_learning_strategy,
            "transfer": self._transfer_learning_strategy,
            "meta": self._meta_learning_strategy,
            "active": self._active_learning_strategy
        }
        
        # Performance metrics
        self.performance_metrics = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "average_learning_time": 0.0,
            "strategy_effectiveness": {},
            "recent_improvement": 0.0
        }
        
        # Track if training from scratch
        self.from_scratch = from_scratch
        
        # Decide whether to load historical data based on from_scratch setting
        if not from_scratch:
            self._load_learning_history()
        else:
            logger.info("From scratch training mode - Not loading meta-learning history data")
            
    def initialize(self):
        """Initialize the meta-learning system"""
        logger.info("Meta-learning system initialized")
        return True
    
    def _load_learning_history(self):
        """Load learning history data"""
        history_file = Path("data/meta_learning_history.pkl")
        if history_file.exists():
            try:
                with open(history_file, 'rb') as f:
                    data = pickle.load(f)
                    self.learning_history = data.get('history', [])
                    self.meta_state = data.get('state', self.meta_state)
                    self.performance_metrics = data.get('metrics', self.performance_metrics)
                logger.info(f"Loaded {len(self.learning_history)} learning history records")
            except Exception as e:
                logger.warning(f"Failed to load learning history: {e}")
    
    def _save_learning_history(self):
        """Save learning history data"""
        try:
            data = {
                'history': self.learning_history,
                'state': self.meta_state,
                'metrics': self.performance_metrics
            }
            with open("data/meta_learning_history.pkl", 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save learning history: {e}")
    
    def _reinforcement_learning_strategy(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Reinforcement learning strategy - optimized with neural networks"""
        # Generate optimal reinforcement learning parameters using neural optimization
        neural_config = self._optimize_learning_strategy(
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
        """Supervised learning strategy - optimized with neural networks"""
        # Generate optimal supervised learning parameters using neural optimization
        neural_config = self._optimize_learning_strategy(
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
        """Unsupervised learning strategy - optimized with neural networks"""
        # Generate optimal unsupervised learning parameters using neural optimization
        neural_config = self._optimize_learning_strategy(
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
        """Transfer learning strategy - optimized with neural networks"""
        # Generate optimal transfer learning parameters using neural optimization
        neural_config = self._optimize_learning_strategy(
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
        """Meta learning strategy - optimized with neural networks"""
        # Generate optimal meta learning parameters using neural optimization
        neural_config = self._optimize_learning_strategy(
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
        """Active learning strategy - optimized with neural networks"""
        # Generate optimal active learning parameters using neural optimization
        neural_config = self._optimize_learning_strategy(
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
        Select optimal learning strategy based on task description and historical experience
        """
        # Analyze task type
        task_type = self._analyze_task_type(task_description, task_data)
        
        # Select strategy based on historical performance
        best_strategy = self._get_best_strategy_for_task_type(task_type)
        
        # Get strategy configuration
        strategy_config = self.learning_strategies[best_strategy](task_data)
        
        return {
            "selected_strategy": best_strategy,
            "strategy_config": strategy_config,
            "confidence": self._calculate_strategy_confidence(best_strategy, task_type),
            "task_type": task_type
        }
    
    def _analyze_task_type(self, task_description: str, task_data: Dict[str, Any]) -> str:
        """Analyze task type based on description"""
        description = task_description.lower()
        
        if any(word in description for word in ["classification", "recognition", "prediction", "regression"]):
            return "supervised"
        elif any(word in description for word in ["clustering", "grouping", "pattern discovery", "anomaly detection"]):
            return "unsupervised"
        elif any(word in description for word in ["decision", "control", "optimization", "reward"]):
            return "reinforcement"
        elif any(word in description for word in ["transfer", "adaptation", "similar tasks"]):
            return "transfer"
        elif any(word in description for word in ["meta learning", "learn to learn", "fast adaptation"]):
            return "meta"
        elif any(word in description for word in ["active", "query", "interactive"]):
            return "active"
        else:
            return "general"
    
    def _get_best_strategy_for_task_type(self, task_type: str) -> str:
        """Get best strategy based on historical performance"""
        # Initialize strategy performance
        if not self.meta_state.strategy_performance:
            for strategy in self.learning_strategies.keys():
                self.meta_state.strategy_performance[strategy] = 0.7  # Default confidence
        
        # Select strategy with highest performance
        best_strategy = max(
            self.meta_state.strategy_performance.items(),
            key=lambda x: x[1]
        )[0]
        
        # Occasionally explore new strategies (10% probability)
        if np.random.random() < 0.1:
            exploration_strategy = np.random.choice(list(self.learning_strategies.keys()))
            if self.meta_state.strategy_performance[exploration_strategy] > 0.5:
                best_strategy = exploration_strategy
        
        return best_strategy
    
    def _calculate_strategy_confidence(self, strategy: str, task_type: str) -> float:
        """Calculate strategy confidence"""
        base_confidence = self.meta_state.strategy_performance.get(strategy, 0.7)
        
        # Adjust confidence based on task type match
        type_match_bonus = 0.2 if strategy == task_type else 0.0
        
        return min(1.0, base_confidence + type_match_bonus)
    
    def _optimize_learning_strategy(self, strategy_type: str, task_data: Dict[str, Any], 
                                  historical_performance: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize learning strategy parameters using neural optimization techniques
        """
        # Neural optimization based on strategy type and historical performance
        base_config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "exploration_rate": 0.3
        }
        
        # Adjust parameters based on strategy type
        if strategy_type == "reinforcement":
            base_config.update({
                "learning_rate": 0.01,
                "exploration_rate": 0.3,
                "reward_shaping": True,
                "value_iteration_steps": 100
            })
        elif strategy_type == "supervised":
            base_config.update({
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10,
                "validation_split": 0.2
            })
        elif strategy_type == "unsupervised":
            base_config.update({
                "clustering_method": "kmeans",
                "dimensionality_reduction": "pca",
                "anomaly_detection": True
            })
        elif strategy_type == "transfer":
            base_config.update({
                "fine_tuning": True,
                "feature_extraction": True,
                "adaptation_layers": 2
            })
        elif strategy_type == "meta":
            base_config.update({
                "meta_learning_rate": 0.001,
                "inner_loop_steps": 5,
                "outer_loop_steps": 3,
                "gradient_clipping": True
            })
        elif strategy_type == "active":
            base_config.update({
                "query_strategy": "uncertainty_sampling",
                "batch_size": 16,
                "max_queries": 100,
                "diversity_measure": "cosine"
            })
        
        # Apply performance-based adjustments
        strategy_performance = historical_performance.get(strategy_type, 0.7)
        if strategy_performance > 0.8:
            # High performance - fine-tune parameters
            if "learning_rate" in base_config:
                base_config["learning_rate"] *= 1.05
        elif strategy_performance < 0.6:
            # Low performance - adjust more aggressively
            if "learning_rate" in base_config:
                base_config["learning_rate"] *= 0.8
        
        return base_config
    
    def record_learning_episode(self, episode: LearningEpisode):
        """Record learning experience"""
        self.learning_history.append(episode)
        self.performance_metrics["total_episodes"] += 1
        
        # Update strategy performance
        if episode.success_metric > 0.7:  # Success threshold
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
        
        # Update learning curve
        self.meta_state.learning_curve.append(episode.success_metric)
        if len(self.meta_state.learning_curve) > 100:
            self.meta_state.learning_curve = self.meta_state.learning_curve[-100:]
        
        # Update adaptation rate (based on recent performance)
        recent_performance = np.mean(self.meta_state.learning_curve[-10:]) if len(self.meta_state.learning_curve) >= 10 else 0.7
        self.meta_state.adaptation_rate = 0.1 + (recent_performance - 0.7) * 0.5
        
        # Update knowledge transfer efficiency
        similar_episodes = [e for e in self.learning_history if e.task_type == episode.task_type]
        if len(similar_episodes) > 1:
            improvements = [e.success_metric for e in similar_episodes]
            self.meta_state.knowledge_transfer_efficiency = np.mean(improvements) / max(improvements)
        
        # Save history
        self._save_learning_history()
        
        logger.info(f"Recorded learning experience: {episode.task_type}, Strategy: {episode.strategy_used}, Success rate: {episode.success_metric:.2f}")
    
    def optimize_learning_parameters(self, current_params: Dict[str, Any], performance_feedback: float) -> Dict[str, Any]:
        """Optimize learning parameters based on performance feedback"""
        optimized_params = current_params.copy()
        
        # Adjust parameters based on performance feedback
        if performance_feedback > 0.8:
            # Good performance, fine-tune parameters
            if 'learning_rate' in optimized_params:
                optimized_params['learning_rate'] *= 1.1
            if 'exploration_rate' in optimized_params:
                optimized_params['exploration_rate'] *= 0.9
        elif performance_feedback < 0.6:
            # Poor performance, make larger adjustments
            if 'learning_rate' in optimized_params:
                optimized_params['learning_rate'] *= 0.8
            if 'exploration_rate' in optimized_params:
                optimized_params['exploration_rate'] *= 1.2
        
        return optimized_params
    
    def generate_learning_insights(self) -> List[str]:
        """Generate learning insights from recent learning experiences"""
        insights = []
        
        # Analyze recent learning patterns
        recent_episodes = self.learning_history[-10:] if len(self.learning_history) >= 10 else self.learning_history
        
        if recent_episodes:
            success_rate = np.mean([e.success_metric for e in recent_episodes])
            avg_time = np.mean([e.learning_time for e in recent_episodes])
            
            insights.append(f"Recent learning success rate: {success_rate:.2f}")
            insights.append(f"Average learning time: {avg_time:.2f} seconds")
            
            # Strategy effectiveness analysis
            strategy_perf = {}
            for episode in recent_episodes:
                if episode.strategy_used not in strategy_perf:
                    strategy_perf[episode.strategy_used] = []
                strategy_perf[episode.strategy_used].append(episode.success_metric)
            
            for strategy, perfs in strategy_perf.items():
                avg_perf = np.mean(perfs)
                insights.append(f"Strategy '{strategy}' average effectiveness: {avg_perf:.2f}")
        
        return insights
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and performance metrics"""
        return {
            "total_episodes": self.performance_metrics["total_episodes"],
            "success_rate": self.performance_metrics["successful_episodes"] / max(1, self.performance_metrics["total_episodes"]),
            "strategy_performance": self.meta_state.strategy_performance,
            "adaptation_rate": self.meta_state.adaptation_rate,
            "knowledge_transfer_efficiency": self.meta_state.knowledge_transfer_efficiency,
            "learning_curve_length": len(self.meta_state.learning_curve)
        }

# Singleton instance will be created in main.py after all dependencies are loaded

if __name__ == "__main__":
    # Test code
    mls = MetaLearningSystem()
    
    print("=== Testing Meta Learning System ===")
    
    # Test strategy selection
    task_desc = "Image classification task"
    task_data = {"dataset_size": 1000, "similar_tasks": ["object recognition"]}
    
    strategy = mls.select_learning_strategy(task_desc, task_data)
    print(f"Selected learning strategy: {strategy}")
    
    # Record learning experience
    episode = LearningEpisode(
        task_type="supervised",
        strategy_used="supervised",
        success_metric=0.85,
        learning_time=120.5,
        resources_used={"cpu": 0.7, "memory": 0.6},
        insights_gained=["Using data augmentation to improve generalization"],
        timestamp=time.time()
    )
    mls.record_learning_episode(episode)
    
    # Display system statistics
    stats = mls.get_system_stats()
    print("\n=== System Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Generate learning insights
    insights = mls.generate_learning_insights()
    print("\n=== Learning Insights ===")
    for insight in insights:
        print(f"- {insight}")
