import zlib
"""
AGI自我学习系统 - 完全AGI级别的自我学习能力
集成元学习、迁移学习、因果推理、自我反思等高级认知功能
"""

import logging
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Set, Union, Callable
from datetime import datetime, timedelta
import hashlib
import json
import pickle
from collections import defaultdict, deque
import random
from pathlib import Path
import uuid
import threading
import time
from dataclasses import dataclass
import statistics
from core.error_handling import error_handler

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从autonomous_learning_manager.py集成的配置类

def _deterministic_randn(size, seed_prefix="default"):
    """Generate deterministic normal distribution using numpy RandomState"""
    import math
    if isinstance(size, int):
        size = (size,)
    total_elements = 1
    for dim in size:
        total_elements *= dim
    
    # Create deterministic seed from seed_prefix using adler32
    seed_hash = zlib.adler32(seed_prefix.encode('utf-8')) & 0xffffffff
    rng = np.random.RandomState(seed_hash)
    
    # Generate uniform random numbers
    u1 = rng.random_sample(total_elements)
    u2 = rng.random_sample(total_elements)
    
    # Apply Box-Muller transform
    u1 = np.maximum(u1, 1e-10)
    u2 = np.maximum(u2, 1e-10)
    z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)
    
    # Convert to torch tensor
    import torch
    result = torch.from_numpy(z0).float()
    
    return result.view(*size)

@dataclass
class AutonomousConfig:
    """自主学习配置"""
    training_interval: int = 3600  # 训练间隔（秒）
    optimization_interval: int = 1800  # 优化间隔（秒）
    monitoring_interval: int = 300  # 监控间隔（秒）
    learning_interval: int = 300  # 学习循环间隔（秒）
    min_improvement_threshold: float = 0.1  # 最小改进阈值
    max_training_iterations: int = 10  # 最大训练迭代次数
    enable_continuous_learning: bool = True  # 启用持续学习
    enable_pickle_backup: bool = False  # 启用pickle格式备份
    
    def get(self, key, default=None):
        """提供字典兼容的get方法"""
        return getattr(self, key, default)

class AGISelfLearningSystem:
    """
    AGI级自我学习系统 - 实现真正的通用人工智能学习能力
    集成神经科学启发的学习机制和高级认知功能
    """
    
    def __init__(self, from_scratch=False, model_registry=None):
        self.initialized = False
        self.learning_enabled = True
        self.last_learning_time = None
        self.creation_time = datetime.now()
        self.from_scratch = from_scratch
        if model_registry is None:
            try:
                from core.model_registry import get_model_registry
                self.model_registry = get_model_registry()
            except Exception as e:
                self.model_registry = None
                logger.warning(f"获取模型注册表失败: {e}")
        else:
            self.model_registry = model_registry
        
        # 高级学习统计
        self.learning_stats = {
            'total_learning_sessions': 0,
            'successful_learnings': 0,
            'failed_learnings': 0,
            'total_knowledge_gained': 0.0,
            'meta_learning_improvements': 0,
            'transfer_learning_successes': 0,
            'causal_discoveries': 0,
            'self_reflection_insights': 0,
            'long_term_goals_achieved': 0
        }
        
        # 高级知识架构
        self.knowledge_architecture = {
            'semantic_memory': defaultdict(dict),      # 语义记忆（概念、事实）
            'episodic_memory': deque(maxlen=10000),    # 情景记忆（经历、事件）
            'procedural_memory': defaultdict(dict),    # 程序记忆（技能、程序）
            'meta_memory': defaultdict(dict),          # 元记忆（关于记忆的记忆）
            'causal_models': defaultdict(dict),        # 因果模型
            'mental_models': defaultdict(dict)         # 心智模型
        }
        
        # 学习系统和参数
        self.learning_parameters = {
            'base_learning_rate': 0.01,
            'meta_learning_rate': 0.001,
            'exploration_rate': 0.15,
            'exploitation_rate': 0.85,
            'discount_factor': 0.95,
            'forgetting_rate': 0.001,
            'consolidation_threshold': 0.7,
            'transfer_learning_threshold': 0.6
        }
        
        # 经验回放和缓冲区
        self.experience_replay = deque(maxlen=5000)
        self.working_memory = deque(maxlen=100)
        self.long_term_memory = []
        
        # 学习目标和规划
        self.learning_goals = {
            'short_term': [],
            'medium_term': [],
            'long_term': []
        }
        
        # 自我监控和反思
        self.self_monitoring = {
            'learning_efficiency': 0.8,
            'knowledge_retention': 0.75,
            'knowledge_consolidation': 0.7,
            'problem_solving_ability': 0.7,
            'adaptability_score': 0.65,
            'creativity_level': 0.6
        }
        
        # 设备和优化设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 从autonomous_learning_manager.py集成的功能
        self.config = AutonomousConfig()
        self.running = False
        self.learning_thread = None
        self.performance_history = defaultdict(list)
        self.improvement_suggestions = []
        self.model_references = {}
        
        # 学习进度和状态跟踪
        self.learning_progress = 0.0
        self.current_learning_status = 'idle'  # idle, running, paused, completed
        self.learning_domains = []
        self.learning_priority = 'balanced'
        self.learning_logs = []
        self.max_logs = 50
        
        # 模型状态跟踪
        self.model_status_tracking = defaultdict(lambda: {
            'last_trained': None,
            'performance_score': 0.0,
            'improvement_rate': 0.0,
            'training_priority': 0
        })
        
        # 优先级队列和已实施的建议
        self.priority_queue = []
        self.implemented_suggestions = []
        
        # 性能评估器和学习策略选择器
        self.performance_evaluator = None
        self.learning_strategy = None
        
        # 知识模型和语言模型引用
        self.knowledge_model = None
        self.language_model = None
        
        # 设置学习基础设施
        self._setup_learning_infrastructure()
    
    def _execute_with_recovery(self, operation_name, operation_func, max_retries=2, 
                             recovery_func=None, fallback_value=None, **kwargs):
        """
        执行操作并提供错误恢复机制
        参数:
            operation_name: 操作名称
            operation_func: 操作函数
            max_retries: 最大重试次数
            recovery_func: 恢复函数
            fallback_value: 后备值
            **kwargs: 传递给操作函数的参数
        返回: 操作结果或后备值
        """
        retries = 0
        last_exception = None
        
        while retries <= max_retries:
            try:
                if retries > 0:
                    error_handler.log_info(f"重试 {operation_name}，第{retries}次尝试", "SelfLearning")
                    # 指数退避：每次重试等待更长时间
                    wait_time = min(1.0 * (2 ** (retries - 1)), 10.0)
                    time.sleep(wait_time)
                
                result = operation_func(**kwargs)
                if retries > 0:
                    error_handler.log_info(f"{operation_name} 在第{retries}次重试后成功", "SelfLearning")
                return result
                
            except Exception as e:
                last_exception = e
                retries += 1
                error_handler.log_warning(
                    f"{operation_name} 失败，第{retries-1}次尝试: {str(e)[:100]}",
                    "SelfLearning"
                )
        
        # 所有重试都失败了
        error_message = f"{operation_name} 在{max_retries}次重试后仍然失败"
        if last_exception:
            error_message += f": {last_exception}"
        
        error_handler.handle_error(Exception(error_message), "SelfLearning", f"{operation_name}失败")
        
        # 尝试恢复函数
        if recovery_func:
            try:
                error_handler.log_info(f"尝试执行{operation_name}的恢复函数", "SelfLearning")
                recovery_result = recovery_func()
                if recovery_result is not None:
                    return recovery_result
            except Exception as recovery_error:
                error_handler.handle_error(recovery_error, "SelfLearning", f"{operation_name}恢复失败")
        
        # 返回后备值
        if fallback_value is not None:
            error_handler.log_warning(f"{operation_name} 使用后备值", "SelfLearning")
            return fallback_value
        else:
            # 如果没有后备值，重新抛出最后一个异常
            raise last_exception if last_exception else Exception(f"{operation_name}失败")
    
    def set_cognitive_architecture(self, cognitive_architecture: Any) -> bool:
        """
        设置认知架构引用
        Sets reference to the cognitive architecture
        
        Args:
            cognitive_architecture: 认知架构对象
            
        Returns:
            bool: 设置是否成功
        """
        try:
            self.cognitive_architecture = cognitive_architecture
            logger.info("认知架构已设置 | Cognitive architecture set")
            return True
        except Exception as e:
            logger.error(f"设置认知架构失败: {e} | Failed to set cognitive architecture: {e}")
            return False
    
    def set_emotion_system(self, emotion_system: Any) -> bool:
        """
        设置情感系统引用
        Sets reference to the emotion system
        
        Args:
            emotion_system: 情感系统对象
            
        Returns:
            bool: 设置是否成功
        """
        try:
            self.emotion_system = emotion_system
            logger.info("情感系统已设置 | Emotion system set")
            return True
        except Exception as e:
            logger.error(f"设置情感系统失败: {e} | Failed to set emotion system: {e}")
            return False
    
    def initialize(self) -> bool:
        """
        初始化AGI自我学习系统
        返回: 成功为True，否则为False
        """
        try:
            # 集成情感意识系统
            try:
                from core.emotion_awareness import AGIEmotionAwarenessSystem
                self.emotion_system = AGIEmotionAwarenessSystem()
                logger.info("情感意识系统已集成")
            except ImportError as e:
                self.emotion_system = None
                error_handler.log_warning("情感意识系统不可用", "SelfLearning")
            
            # 集成价值对齐系统
            try:
                from core.value_alignment import AGIValueAlignmentSystem
                self.value_system = AGIValueAlignmentSystem()
                logger.info("价值对齐系统已集成")
            except ImportError as e:
                self.value_system = None
                error_handler.log_warning("价值对齐系统不可用", "SelfLearning")
            
            # 获取模型注册表（如果未设置）
            if self.model_registry is None:
                try:
                    from core.model_registry import get_model_registry
                    self.model_registry = get_model_registry()
                    logger.info("模型注册表已获取")
                except Exception as e:
                    logger.warning(f"获取模型注册表失败: {e}")
                    self.model_registry = None

            # 初始化神经学习模型
            self._initialize_neural_models()
            
            # 加载现有知识（如果存在）
            if not self.from_scratch:
                self._load_existing_knowledge()
            else:
                logger.info("AGI自我学习系统以从零开始训练模式启动，跳过现有知识加载")
            
            # 设置初始学习目标
            self._setup_initial_learning_goals()
            
            self.initialized = True
            self.last_learning_time = datetime.now()
            
            logger.info("AGI自我学习系统初始化成功")
            logger.info(f"知识架构: {self._get_knowledge_summary()}")
            
            return True
            
        except Exception as e:
            logger.error(f"初始化AGI自我学习系统失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _initialize_neural_models(self):
        """初始化神经学习模型"""
        # 元学习模型（学习如何学习）
        self.meta_learner = nn.ModuleDict({
            'feature_extractor': nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            ),
            'learning_strategy_predictor': nn.Linear(32, 10)
        }).to(self.device)
        
        # 迁移学习模型
        self.transfer_learner = nn.ModuleDict({
            'domain_encoder': nn.Linear(50, 25),
            'knowledge_transformer': nn.Transformer(d_model=25, nhead=5)
        }).to(self.device)
        
        # 优化器
        self.meta_optimizer = torch.optim.Adam(self.meta_learner.parameters(), 
                                              lr=self.learning_parameters['meta_learning_rate'])
        self.transfer_optimizer = torch.optim.Adam(self.transfer_learner.parameters(),
                                                  lr=self.learning_parameters['base_learning_rate'])
        
        logger.info("神经学习模型初始化完成")
    
    def _setup_initial_learning_goals(self):
        """设置初始学习目标"""
        # 短期目标（立即学习）
        self.learning_goals['short_term'] = [
            {'id': str(uuid.uuid4()), 'description': '掌握基础交互模式', 'priority': 0.9, 'deadline': datetime.now() + timedelta(hours=1)},
            {'id': str(uuid.uuid4()), 'description': '建立初始概念网络', 'priority': 0.8, 'deadline': datetime.now() + timedelta(hours=2)}
        ]
        
        # 中期目标（几天内）
        self.learning_goals['medium_term'] = [
            {'id': str(uuid.uuid4()), 'description': '发展因果推理能力', 'priority': 0.7, 'deadline': datetime.now() + timedelta(days=3)},
            {'id': str(uuid.uuid4()), 'description': '建立自我反思机制', 'priority': 0.75, 'deadline': datetime.now() + timedelta(days=5)}
        ]
        
        # 长期目标（几周内）
        self.learning_goals['long_term'] = [
            {'id': str(uuid.uuid4()), 'description': '实现通用问题解决能力', 'priority': 0.6, 'deadline': datetime.now() + timedelta(weeks=2)},
            {'id': str(uuid.uuid4()), 'description': '发展创造性思维', 'priority': 0.55, 'deadline': datetime.now() + timedelta(weeks=4)}
        ]

    def _setup_learning_infrastructure(self):
        """设置学习基础设施（从autonomous_learning_manager.py迁移）"""
        # 初始化模型引用
        self._initialize_model_references()
        
        # 创建必要的数据结构和组件
        self.performance_evaluator = self._create_performance_evaluator()
        self.learning_strategy = self._create_learning_strategy()
        
        # 初始化模型状态跟踪
        self.model_status_tracking = defaultdict(lambda: {
            'last_trained': None,
            'performance_score': 0.0,
            'improvement_rate': 0.0,
            'training_priority': 0
        })

    def _initialize_model_references(self):
        """初始化对其他模型的引用"""
        if self.model_registry is None:
            self.knowledge_model = None
            self.language_model = None
            self.model_references = {}
            logger.warning("模型注册表未设置，无法初始化模型引用")
            return
        
        # 获取关键模型的引用
        self.knowledge_model = self.model_registry.get_model('knowledge')
        self.language_model = self.model_registry.get_model('language')
        
        # 获取所有模型引用
        all_models = self.model_registry.get_all_models()
        for model_id, model in all_models.items():
            self.model_references[model_id] = model

    def _create_performance_evaluator(self):
        """创建性能评估器"""
        def evaluate_model(model, task):
            """评估模型性能"""
            try:
                # 获取模型的实际性能指标
                if hasattr(model, 'get_performance_metrics'):
                    metrics = model.get_performance_metrics()
                    # 计算综合性能分数
                    if metrics:
                        score = 0
                        weight_sum = 0
                        
                        # 为不同指标分配权重
                        weights = {
                            'accuracy': 0.3,
                            'precision': 0.2,
                            'recall': 0.2,
                            'f1_score': 0.2,
                            'speed': 0.1
                        }
                        
                        for metric, value in metrics.items():
                            if metric in weights and isinstance(value, (int, float)):
                                score += value * weights[metric]
                                weight_sum += weights[metric]
                        
                        return score / weight_sum if weight_sum > 0 else 0.5
                
                # 如果没有具体性能指标，使用基础评估
                if hasattr(model, 'evaluate'):
                    result = model.evaluate(task)
                    if isinstance(result, (int, float)):
                        return min(max(result, 0), 1)  # 确保在0-1范围内
                
                # 默认返回中间值
                return 0.5
            except Exception as e:
                error_handler.handle_error(e, "SelfLearning", "性能评估失败")
                return 0.3  # 出错时返回较低分数
        
        # 返回性能评估器实例
        return {'evaluate': evaluate_model}

    def _create_learning_strategy(self):
        """创建学习策略"""
        def select_next_task(model, performance):
            """根据模型性能和类型选择下一个学习任务"""
            try:
                model_id = model.model_id if hasattr(model, 'model_id') else 'unknown'
                
                # 根据模型类型和性能选择不同的学习任务
                # 性能较低的模型应该优先进行基础训练
                if performance < 0.4:
                    if model_id == 'knowledge':
                        return 'foundational_knowledge_acquisition'
                    elif model_id in ['language', 'vision_image', 'vision_video']:
                        return 'basic_skill_training'
                    else:
                        return 'fundamental_concept_learning'
                elif performance < 0.7:
                    # 中等性能模型进行知识增强和实践训练
                    if model_id == 'knowledge':
                        return 'knowledge_integration'
                    elif model_id == 'language':
                        return 'contextual_understanding_training'
                    elif model_id in ['vision_image', 'vision_video']:
                        return 'complex_pattern_recognition'
                    elif model_id in ['planning', 'prediction']:
                        return 'scenario_based_training'
                    else:
                        return 'task_specific_enhancement'
                else:
                    # 高性能模型进行高级学习和创新
                    if model_id == 'knowledge':
                        return 'knowledge_creation'
                    elif model_id == 'autonomous':
                        return 'meta_learning_optimization'
                    elif model_id == 'programming':
                        return 'advanced_algorithm_exploration'
                    else:
                        return 'cross_domain_knowledge_transfer'
            except Exception as e:
                error_handler.handle_error(e, "SelfLearning", "学习任务选择失败")
                return 'knowledge_enhancement'  # 默认返回知识增强任务
        
        # 返回学习策略实例
        return {'select_next_task': select_next_task}

    def _load_existing_knowledge(self):
        """加载现有知识 - 使用安全的JSON格式替代pickle"""
        knowledge_path = Path("data/knowledge/self_learning_knowledge.json")
        if knowledge_path.exists():
            try:
                with open(knowledge_path, 'r', encoding='utf-8') as f:
                    saved_knowledge = json.load(f)
                    
                # 验证加载的数据结构
                if self._validate_knowledge_structure(saved_knowledge):
                    self.knowledge_architecture.update(saved_knowledge)
                    logger.info("现有知识加载成功（JSON格式）")
                else:
                    logger.warning("知识文件结构验证失败，使用默认结构")
            except json.JSONDecodeError as e:
                error_handler.log_warning(f"JSON解析失败: {e}，尝试使用pickle格式作为后备", "SelfLearning")
                # 尝试加载旧的pickle格式作为后备
                self._load_pickle_knowledge_as_fallback()
            except Exception as e:
                error_handler.log_warning(f"加载现有知识失败: {e}", "SelfLearning")
        else:
            # 检查是否存在旧的pickle格式文件
            pickle_path = Path("data/knowledge/self_learning_knowledge.pkl")
            if pickle_path.exists():
                logger.info("检测到旧的pickle格式知识文件，尝试转换")
                self._convert_pickle_to_json(pickle_path, knowledge_path)
    
    def _load_pickle_knowledge_as_fallback(self) -> bool:
        """加载pickle格式知识作为后备方案（仅在JSON失败时使用）"""
        try:
            pickle_path = Path("data/knowledge/self_learning_knowledge.pkl")
            if not pickle_path.exists():
                logger.warning("pickle知识文件不存在，无法加载后备知识")
                return False
            
            # 使用pickle加载，但限制反序列化类型
            import pickle
            # 为了安全，我们可以使用限制反序列化类型的pickle加载器
            # 但这里我们只是简单加载，因为这是内部使用的备份文件
            with open(pickle_path, 'rb') as f:
                saved_knowledge = pickle.load(f)
            
            # 验证数据结构
            if self._validate_knowledge_structure(saved_knowledge):
                self.knowledge_architecture.update(saved_knowledge)
                logger.info("现有知识加载成功（pickle后备格式）")
                # 尝试转换为JSON格式以便将来使用
                self._convert_pickle_to_json(pickle_path, Path("data/knowledge/self_learning_knowledge.json"))
                return True
            else:
                logger.warning("pickle知识文件结构验证失败")
                return False
        except Exception as e:
            error_handler.log_warning(f"加载pickle后备知识失败: {e}", "SelfLearning")
            return False
    
    def _convert_pickle_to_json(self, pickle_path: Path, json_path: Path) -> bool:
        """将pickle格式知识转换为JSON格式"""
        try:
            import pickle
            # 确保目录存在
            json_path.parent.mkdir(exist_ok=True, parents=True)
            
            # 加载pickle数据
            with open(pickle_path, 'rb') as f:
                pickle_data = pickle.load(f)
            
            # 转换为可序列化的格式
            serializable_data = self._make_knowledge_serializable(pickle_data)
            
            # 保存为JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"成功将pickle知识转换为JSON格式: {pickle_path} -> {json_path}")
            return True
        except Exception as e:
            error_handler.log_warning(f"转换pickle到JSON失败: {e}", "SelfLearning")
            return False
    
    def _validate_knowledge_structure(self, knowledge_data: Dict[str, Any]) -> bool:
        """验证知识数据结构是否有效"""
        try:
            # 检查基本结构
            if not isinstance(knowledge_data, dict):
                return False
            
            # 检查必需的部分
            required_sections = ['semantic_memory', 'episodic_memory', 'procedural_memory', 
                                'meta_memory', 'causal_models', 'mental_models']
            
            for section in required_sections:
                if section not in knowledge_data:
                    return False
            
            # 检查每个部分的数据类型
            if not isinstance(knowledge_data['semantic_memory'], dict):
                return False
            if not isinstance(knowledge_data['episodic_memory'], list):
                return False
            if not isinstance(knowledge_data['procedural_memory'], dict):
                return False
            if not isinstance(knowledge_data['meta_memory'], dict):
                return False
            if not isinstance(knowledge_data['causal_models'], dict):
                return False
            if not isinstance(knowledge_data['mental_models'], dict):
                return False
            
            return True
        except Exception:
            return False
    
    def _make_knowledge_serializable(self, knowledge_data: Any) -> Any:
        """确保知识数据可被JSON序列化"""
        if isinstance(knowledge_data, dict):
            return {key: self._make_knowledge_serializable(value) for key, value in knowledge_data.items()}
        elif isinstance(knowledge_data, list):
            return [self._make_knowledge_serializable(item) for item in knowledge_data]
        elif isinstance(knowledge_data, (str, int, float, bool, type(None))):
            return knowledge_data
        elif isinstance(knowledge_data, (datetime,)):
            return knowledge_data.isoformat()
        elif hasattr(knowledge_data, '__dict__'):
            # 将对象转换为字典
            return self._make_knowledge_serializable(knowledge_data.__dict__)
        else:
            # 其他类型转换为字符串
            return str(knowledge_data)
    
    def _save_knowledge(self):
        """保存知识到文件 - 使用安全的JSON格式"""
        try:
            knowledge_path = Path("data/knowledge")
            knowledge_path.mkdir(exist_ok=True)
            
            # 保存为JSON格式
            json_path = knowledge_path / "self_learning_knowledge.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                # 确保数据可序列化
                serializable_knowledge = self._make_knowledge_serializable(self.knowledge_architecture)
                json.dump(serializable_knowledge, f, indent=2, ensure_ascii=False)
            
            logger.info("知识保存成功（JSON格式）")
            
            # 可选：同时保存一个pickle格式的备份（但标记为不安全）
            # 在实际生产环境中应该禁用pickle格式
            if self.config.get('enable_pickle_backup', False):
                pickle_path = knowledge_path / "self_learning_knowledge_backup.pkl"
                with open(pickle_path, 'wb') as f:
                    pickle.dump(self.knowledge_architecture, f)
                logger.info("知识备份保存成功（pickle格式）")
                
        except Exception as e:
            logger.error(f"保存知识失败: {e}")

    # 从autonomous_learning_manager.py集成的方法
    def start_autonomous_learning_cycle(self, domains=None, priority='balanced'):
        """启动自主学习循环"""
        if self.running:
            logger.info("自主学习循环已在运行中")
            return False
        
        # 设置学习参数
        self.learning_domains = domains or []
        self.learning_priority = priority
        
        # 重置进度和状态
        self.learning_progress = 0.0
        self.current_learning_status = 'running'
        self.learning_logs = []
        
        # 添加启动日志
        self._add_learning_log(f"开始自主学习，领域: {self.learning_domains}, 优先级: {self.learning_priority}")
        
        self.running = True
        self.learning_thread = threading.Thread(target=self._autonomous_learning_cycle)
        self.learning_thread.daemon = True
        self.learning_thread.start()
        
        logger.info("自主学习循环已启动")
        return True
        
    def stop_autonomous_learning_cycle(self):
        """停止自主学习循环"""
        self.running = False
        self.current_learning_status = 'idle'
        self._add_learning_log("自主学习已停止")
        
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=5.0)
            
        logger.info("自主学习循环已停止")
        return True
        
    def _autonomous_learning_cycle(self):
        """自主学习循环的内部实现"""
        try:
            total_cycles = self.config.max_training_iterations
            current_cycle = 0
            
            while self.running and current_cycle < total_cycles:
                try:
                    current_cycle += 1
                    
                    # 更新进度
                    self.learning_progress = min((current_cycle / total_cycles) * 100, 100)
                    
                    # 执行学习循环
                    self._execute_learning_cycle()
                    
                    # 等待下一个学习周期
                    wait_time = self.config.learning_interval
                    for i in range(wait_time // 1000):
                        if not self.running:
                            break
                        time.sleep(1)
                        # 每1秒小幅度更新进度
                        self.learning_progress = min(self.learning_progress + (100/(total_cycles*wait_time)), 100)
                except Exception as e:
                    logger.error(f"自主学习循环出错: {e}")
                    self._add_learning_log(f"学习循环出错: {str(e)}")
                    time.sleep(5)
            
            # 学习完成
            if self.running:
                self.learning_progress = 100
                self.current_learning_status = 'completed'
                self._add_learning_log("自主学习完成")
                self.running = False
        except Exception as e:
            logger.error(f"自主学习循环严重错误: {e}")
            self._add_learning_log(f"学习循环严重错误: {str(e)}")
            self.learning_progress = 0
            self.current_learning_status = 'idle'
            self.running = False
    
    def _learning_cycle(self):
        """自主学习循环的内部实现（从autonomous_learning_manager.py迁移）"""
        try:
            total_cycles = self.config.max_training_iterations
            current_cycle = 0
            
            while self.running and current_cycle < total_cycles:
                try:
                    current_cycle += 1
                    
                    # 更新进度
                    self.learning_progress = min((current_cycle / total_cycles) * 100, 100)
                    
                    # 检查每个模型的状态和性能
                    self._evaluate_all_models()
                    
                    # 选择最需要改进的模型和任务
                    model_id, task = self._select_next_improvement_target()
                    
                    if model_id and task:
                        # 执行改进任务
                        self._execute_improvement_task(model_id, task)
                        
                        # 添加任务执行日志
                        self._add_learning_log(f"已完成 {task} 任务对模型 {model_id}")
                    
                    # 生成学习报告
                    self._generate_learning_report()
                    
                    # 等待下一个学习周期
                    wait_time = self.config.learning_interval
                    for i in range(wait_time // 1000):
                        if not self.running:
                            break
                        time.sleep(1)
                        # 每1秒小幅度更新进度
                        self.learning_progress = min(self.learning_progress + (100/(total_cycles*wait_time)), 100)
                except Exception as e:
                    logger.error(f"自主学习循环出错: {e}")
                    self._add_learning_log(f"学习循环出错: {str(e)}")
                    # 继续运行，即使发生错误
                    time.sleep(5)
            
            # 学习完成
            if self.running:
                self.learning_progress = 100
                self.current_learning_status = 'completed'
                self._add_learning_log("自主学习完成")
                self.running = False
        except Exception as e:
            logger.error(f"自主学习循环严重错误: {e}")
            self._add_learning_log(f"学习循环严重错误: {str(e)}")
            self.learning_progress = 0
            self.current_learning_status = 'idle'
            self.running = False
    
    def _execute_learning_cycle(self):
        """执行单个学习循环"""
        # 评估当前知识状态
        self._evaluate_knowledge_state()
        
        # 选择学习目标
        learning_target = self._select_learning_target()
        
        if learning_target:
            # 执行学习任务
            self._execute_learning_task(learning_target)
            
            # 添加任务执行日志
            self._add_learning_log(f"已完成学习任务: {learning_target}")
        
        # 生成学习报告
        self._generate_learning_report()
    
    def _evaluate_knowledge_state(self):
        """评估当前知识状态"""
        # 评估知识架构的完整性
        knowledge_completeness = self._assess_knowledge_completeness()
        
        # 评估知识巩固效果
        knowledge_consolidation = self._verify_knowledge_consolidation()
        
        # 评估知识保留情况
        knowledge_retention = self._assess_learning_retention()
        
        # 评估学习效率
        learning_efficiency = self.self_monitoring['learning_efficiency']
        
        # 更新性能历史
        self.performance_history['knowledge_state'].append({
            'timestamp': datetime.now(),
            'completeness': knowledge_completeness,
            'consolidation': knowledge_consolidation,
            'retention': knowledge_retention,
            'efficiency': learning_efficiency
        })
        
        # 更新自我监控
        self.self_monitoring['knowledge_retention'] = knowledge_retention
        self.self_monitoring['knowledge_consolidation'] = knowledge_consolidation
    
    def _assess_knowledge_completeness(self):
        """评估知识完整性"""
        total_concepts = len(self.knowledge_architecture['semantic_memory']['concepts'])
        total_patterns = len(self.knowledge_architecture['semantic_memory']['patterns'])
        total_rules = len(self.knowledge_architecture['procedural_memory']['rules'])
        total_causal_models = len(self.knowledge_architecture['causal_models'])
        
        # 计算综合完整性分数
        completeness_score = min(1.0, (total_concepts * 0.3 + total_patterns * 0.2 + 
                                     total_rules * 0.3 + total_causal_models * 0.2) / 100)
        
        return completeness_score
    
    def _verify_knowledge_consolidation(self):
        """验证知识巩固效果"""
        # 测试知识的应用能力
        consolidation_score = 0.0
        test_count = 0
        
        # 测试概念理解
        concepts = list(self.knowledge_architecture['semantic_memory']['concepts'].values())
        if concepts:
            test_count += 1
            # 检查概念是否有相关概念和示例值
            concept_score = sum(1 for c in concepts if len(c.get('related_concepts', [])) > 0 and len(c.get('value_examples', [])) > 0) / len(concepts)
            consolidation_score += concept_score * 0.3
        
        # 测试规则应用
        rules = list(self.knowledge_architecture['procedural_memory']['rules'].values())
        if rules:
            test_count += 1
            # 检查规则是否有使用记录和置信度
            rule_score = sum(1 for r in rules if r.get('usage_count', 0) > 0 and r.get('confidence', 0) > 0.5) / len(rules)
            consolidation_score += rule_score * 0.3
        
        # 测试因果模型理解
        causal_models = list(self.knowledge_architecture['causal_models'].values())
        if causal_models:
            test_count += 1
            # 检查因果模型是否有证据支持和预测能力
            causal_score = sum(1 for m in causal_models if m.get('evidence_count', 0) > 0 and m.get('predictive_power', 0) > 0.5) / len(causal_models)
            consolidation_score += causal_score * 0.2
        
        # 测试模式识别能力
        patterns = list(self.knowledge_architecture['semantic_memory']['patterns'].values())
        if patterns:
            test_count += 1
            # 检查模式是否有上下文信息和高置信度
            pattern_score = sum(1 for p in patterns if p.get('context', {}) and p.get('pattern_score', 0) > 0.7) / len(patterns)
            consolidation_score += pattern_score * 0.2
        
        return consolidation_score / test_count if test_count > 0 else 0.0
    
    def _assess_learning_retention(self):
        """评估知识保留情况"""
        retention_score = 0.0
        category_count = 0
        
        # 检查概念的保留情况
        concepts = self.knowledge_architecture['semantic_memory']['concepts']
        if concepts:
            category_count += 1
            recent_concepts = sum(1 for c in concepts.values() if 
                                datetime.fromisoformat(c['last_encountered']) > datetime.now() - timedelta(days=30))
            retention_score += (recent_concepts / len(concepts)) * 0.3
        
        # 检查规则的保留情况
        rules = self.knowledge_architecture['procedural_memory']['rules']
        if rules:
            category_count += 1
            recent_rules = sum(1 for r in rules.values() if 
                              datetime.fromisoformat(r['last_used']) > datetime.now() - timedelta(days=30))
            retention_score += (recent_rules / len(rules)) * 0.3
        
        # 检查因果模型的保留情况
        causal_models = self.knowledge_architecture['causal_models']
        if causal_models:
            category_count += 1
            recent_models = sum(1 for m in causal_models.values() if 
                               datetime.fromisoformat(m['last_updated']) > datetime.now() - timedelta(days=30))
            retention_score += (recent_models / len(causal_models)) * 0.2
        
        # 检查经验的保留情况
        episodes = self.knowledge_architecture['episodic_memory']
        if episodes:
            category_count += 1
            recent_episodes = sum(1 for e in episodes if 
                                datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(days=30))
            retention_score += (recent_episodes / len(episodes)) * 0.2
        
        return retention_score / category_count if category_count > 0 else 0.0
    
    def _select_learning_target(self):
        """选择学习目标"""
        # 基于当前知识状态和学习优先级选择目标
        if self.learning_priority == 'exploration':
            return self._select_exploration_target()
        elif self.learning_priority == 'exploitation':
            return self._select_exploitation_target()
        else:  # balanced
            return self._select_balanced_target()
    
    def _select_exploration_target(self):
        """选择探索性学习目标"""
        # 优先选择新的、未充分探索的知识领域
        exploration_targets = [
            'causal_discovery',
            'pattern_identification', 
            'concept_abstraction',
            'meta_learning_optimization'
        ]
        return exploration_targets[(zlib.adler32(str(str(exploration_targets).encode('utf-8')) & 0xffffffff) + "exploration") % len(exploration_targets)]
    
    def _select_exploitation_target(self):
        """选择利用性学习目标"""
        # 优先优化现有知识和技能
        exploitation_targets = [
            'knowledge_consolidation',
            'skill_refinement',
            'performance_optimization',
            'error_correction'
        ]
        return exploitation_targets[(zlib.adler32(str(str(exploitation_targets).encode('utf-8')) & 0xffffffff) + "exploitation") % len(exploitation_targets)]
    
    def _select_balanced_target(self):
        """选择平衡性学习目标"""
        balanced_targets = [
            'knowledge_integration',
            'transfer_learning',
            'adaptive_learning',
            'reflective_learning'
        ]
        return balanced_targets[(zlib.adler32(str(str(balanced_targets).encode('utf-8')) & 0xffffffff) + "balanced") % len(balanced_targets)]
    
    def _execute_learning_task(self, task_type):
        """执行学习任务"""
        try:
            logger.info(f"开始执行学习任务: {task_type}")
            
            # 根据任务类型执行不同的学习策略
            if task_type == 'causal_discovery':
                self._execute_causal_discovery()
            elif task_type == 'pattern_identification':
                self._execute_pattern_identification()
            elif task_type == 'knowledge_integration':
                self._execute_knowledge_integration()
            elif task_type == 'transfer_learning':
                self._execute_transfer_learning()
            else:
                # 默认执行通用学习任务
                self._execute_general_learning()
            
            logger.info(f"完成学习任务: {task_type}")
        except Exception as e:
            logger.error(f"执行学习任务时出错: {task_type}, 错误: {e}")
    
    def _execute_causal_discovery(self):
        """执行因果发现任务"""
        # 从经验中提取潜在的因果关系
        recent_experiences = list(self.experience_replay)[-10:]
        for experience in recent_experiences:
            if 'interaction' in experience:
                self._causal_learning(experience['interaction'])
    
    def _execute_pattern_identification(self):
        """执行模式识别任务"""
        # 分析知识架构中的模式
        patterns = self.knowledge_architecture['semantic_memory']['patterns']
        if len(patterns) < 5:  # 如果模式较少，进行模式发现
            self._identify_advanced_patterns(
                torch.tensor([0.5] * 10, device=self.device),
                {'type': 'pattern_discovery', 'context': 'autonomous_learning'}
            )
    
    def _execute_knowledge_integration(self):
        """执行知识整合任务"""
        # 整合不同记忆系统中的知识
        self._consolidate_knowledge(
            {'type': 'knowledge_integration'},
            {'basic_learning': {'success': True, 'concepts_learned': 1}}
        )
    
    def _execute_transfer_learning(self):
        """执行迁移学习任务"""
        # 尝试将知识迁移到新领域
        self._transfer_learning(
            {'type': 'cross_domain_transfer', 'context': 'autonomous_learning'}
        )
    
    def _execute_general_learning(self):
        """执行通用学习任务"""
        # 执行基础学习循环
        interaction_data = {
            'type': 'autonomous_learning',
            'input': {'learning_goal': 'general_improvement'},
            'output': {'learning_result': 'success'},
            'context': {'autonomous': True}
        }
        self.learn_from_interaction(interaction_data)

    # 从autonomous_learning_manager.py迁移的独特方法
    
    def schedule_implementation(self, suggestion):
        """调度建议的实施
        
        Args:
            suggestion: 改进建议
        """
        try:
            self._add_learning_log(f"开始实施建议: {suggestion.get('content', '未命名建议')}")
            
            # 获取建议目标
            target = suggestion.get('target', 'system')
            suggestion_type = suggestion.get('type', 'general')
            content = suggestion.get('content', '')
            
            # 根据建议类型和目标执行不同的实施策略
            if target == 'system':
                # 系统级建议处理
                self._implement_system_suggestion(suggestion)
            elif target in self.model_references:
                # 模型级建议处理
                self._implement_model_suggestion(target, suggestion)
            else:
                # 通用建议处理
                self._implement_general_suggestion(suggestion)
            
            # 记录已实施的建议
            suggestion['implemented'] = True
            suggestion['implementation_time'] = datetime.now()
            self.implemented_suggestions.append(suggestion)
            
            # 如果建议在优先级队列中，移除它
            if suggestion in self.priority_queue:
                self.priority_queue.remove(suggestion)
            
            self._add_learning_log(f"建议实施完成: {suggestion.get('content', '未命名建议')}")
        except Exception as e:
            logger.error(f"调度实施建议时出错: {str(e)}")
    
    def _implement_system_suggestion(self, suggestion):
        """实施系统级建议
        
        Args:
            suggestion: 系统级建议
        """
        try:
            content = suggestion.get('content', '')
            if 'training_interval' in content or '优化间隔' in content:
                # 调整训练间隔
                if hasattr(self.config, 'training_interval'):
                    self.config.training_interval = suggestion.get('parameters', {}).get('training_interval', 3600)
            elif 'learning_interval' in content or '学习间隔' in content:
                # 调整学习间隔
                if hasattr(self.config, 'learning_interval'):
                    self.config.learning_interval = suggestion.get('parameters', {}).get('learning_interval', 300)
            # 其他系统级建议处理逻辑...
        except Exception as e:
            logger.error(f"实施系统级建议时出错: {str(e)}")
    
    def _implement_model_suggestion(self, model_id, suggestion):
        """实施模型级建议
        
        Args:
            model_id: 模型ID
            suggestion: 模型级建议
        """
        try:
            model = self.model_references.get(model_id)
            if not model:
                logger.warning(f"无法找到模型: {model_id}")
                return
            
            # 调用模型的improve方法
            if hasattr(model, 'improve'):
                model.improve(suggestion.get('content', ''), self.knowledge_model)
            else:
                logger.warning(f"模型 {model_id} 没有improve方法")
        except Exception as e:
            logger.error(f"实施模型级建议时出错: {str(e)}")
    
    def _implement_general_suggestion(self, suggestion):
        """实施通用建议
        
        Args:
            suggestion: 通用建议
        """
        try:
            content = suggestion.get('content', '')
            logger.info(f"实施通用建议: {content}")
            # 通用建议处理逻辑...
        except Exception as e:
            logger.error(f"实施通用建议时出错: {str(e)}")

    def evaluate_suggestion_impact(self, suggestion):
        """评估建议的影响程度
        
        Args:
            suggestion: 改进建议字典，包含type, target, content等字段
            
        Returns:
            float: 影响评分（0-1之间）
        """
        try:
            impact = 0.0
            
            # 1. 根据建议类型评估影响
            suggestion_type = suggestion.get('type', 'general')
            type_weights = {
                'core': 0.8,
                'model': 0.7,
                'optimization': 0.6,
                'knowledge': 0.5,
                'interface': 0.4,
                'general': 0.3
            }
            impact += type_weights.get(suggestion_type, 0.3) * 0.4
            
            # 2. 根据目标范围评估影响
            target = suggestion.get('target', 'system')
            if target == 'system':
                impact += 0.3  # 系统级影响
            elif target in self.model_references:
                impact += 0.2  # 模型级影响
            else:
                impact += 0.1  # 组件级影响
            
            # 3. 根据预期收益评估影响
            expected_benefit = suggestion.get('expected_benefit', 0.5)
            impact += expected_benefit * 0.2
            
            # 4. 根据实施难度反向评估影响（难度低，影响相对更大）
            implementation_difficulty = suggestion.get('implementation_difficulty', 0.5)
            impact += (1 - implementation_difficulty) * 0.1
            
            # 确保影响分数在0-1范围内
            return min(max(impact, 0), 1)
        except Exception as e:
            logger.error(f"评估建议影响时出错: {str(e)}")
            return 0.5  # 默认中间值

    def process_improvement(self, suggestion):
        """处理改进建议，包括影响评估、优先级排序和调度
        
        Args:
            suggestion: 改进建议
        """
        try:
            # 评估建议影响
            impact = self.evaluate_suggestion_impact(suggestion)
            suggestion['impact_score'] = impact
            
            # 添加时间戳
            suggestion['timestamp'] = datetime.now()
            
            # 如果是高影响建议（>0.6），加入优先级队列
            if impact > 0.6:
                self.priority_queue.append(suggestion)
                # 按影响分数排序优先级队列
                self.priority_queue.sort(key=lambda x: x['impact_score'], reverse=True)
                self._add_learning_log(f"高影响建议已加入优先级队列: {suggestion.get('content', '未命名建议')} (影响分数: {impact:.2f})")
                
                # 立即调度实施
                self.schedule_implementation(suggestion)
            else:
                # 低影响建议加入普通改进列表
                self.improvement_suggestions.append(suggestion)
                self._add_learning_log(f"普通影响建议已记录: {suggestion.get('content', '未命名建议')} (影响分数: {impact:.2f})")
        except Exception as e:
            logger.error(f"处理改进建议时出错: {str(e)}")

    def _calculate_overall_performance(self):
        """计算系统整体性能
        
        Returns:
            float: 整体性能分数
        """
        # 计算所有模型的平均性能
        performances = [status.get('performance_score', 0.0) for status in self.model_status_tracking.values()]
        if not performances:
            return 0.0
        
        return sum(performances) / len(performances)

    def get_performance_metrics(self):
        """获取性能指标
        
        Returns:
            dict: 性能指标字典
        """
        metrics = {
            'overall_performance': self._calculate_overall_performance(),
            'model_performances': {}
        }
        
        # 添加各模型性能
        for model_id, status in self.model_status_tracking.items():
            metrics['model_performances'][model_id] = status
        
        return metrics
        
    def get_status(self):
        """获取自主学习管理器的状态
        
        Returns:
            dict: 状态字典
        """
        return {
            'running': self.running,
            'config': self.config.__dict__,
            'models_managed': len(self.model_references),
            'last_evaluation_time': self._get_last_evaluation_time()
        }
        
    def _get_last_evaluation_time(self):
        """获取最后一次评估时间
        
        Returns:
            datetime or None: 最后一次评估时间或None
        """
        # 找出最近的评估时间
        latest_time = None
        for model_id, history in self.performance_history.items():
            if history:
                model_latest = max(history, key=lambda x: x['timestamp'])
                if not latest_time or model_latest['timestamp'] > latest_time:
                    latest_time = model_latest['timestamp']
        
        return latest_time
        
    def get_improvement_suggestions(self):
        """获取改进建议列表
        
        Returns:
            list: 改进建议列表
        """
        return self.improvement_suggestions.copy()

    def _select_next_improvement_target(self):
        """选择下一个需要改进的目标模型和任务
        
        Returns:
            tuple: (模型ID, 任务类型) 或 (None, None)
        """
        # 按优先级排序模型
        prioritized_models = sorted(
            self.model_status_tracking.items(),
            key=lambda x: x[1]['training_priority'],
            reverse=True
        )
        
        # 选择优先级最高的模型
        if prioritized_models:
            model_id, _ = prioritized_models[0]
            # 选择适合该模型的任务
            task = self._select_task_for_model(model_id)
            return model_id, task
        
        return None, None

    def _select_task_for_model(self, model_id):
        """为指定模型选择合适的任务
        
        Args:
            model_id: 模型ID
            
        Returns:
            str: 任务类型
        """
        # 根据模型类型选择任务
        task_map = {
            'language': 'language_enhancement',
            'knowledge': 'knowledge_enhancement',
            'audio': 'audio_enhancement',
            'vision_image': 'vision_enhancement',
            'vision_video': 'vision_enhancement',
            'spatial': 'spatial_enhancement',
            'stereo_spatial': 'spatial_enhancement',
            'sensor': 'sensor_enhancement',
            'computer': 'computer_enhancement',
            'motion': 'motion_enhancement',
            'programming': 'programming_enhancement'
        }
        
        return task_map.get(model_id, 'general_enhancement')

    def _execute_improvement_task(self, model_id, task):
        """执行模型改进任务
        
        Args:
            model_id: 模型ID
            task: 任务类型
        """
        try:
            logger.info(f"开始执行改进任务: {task} 对模型: {model_id}")
            
            # 获取模型引用
            model = self.model_references.get(model_id)
            if not model:
                logger.warning(f"无法找到模型: {model_id}")
                return
            
            # 执行具体任务
            if hasattr(model, 'improve'):
                # 如果模型有improve方法，调用它
                model.improve(task, self.knowledge_model)
            else:
                # 否则使用通用改进方法
                self._general_improvement(model_id, task)
            
            logger.info(f"完成改进任务: {task} 对模型: {model_id}")
        except Exception as e:
            logger.error(f"执行改进任务时出错: {task} 对模型: {model_id}, 错误: {e}")

    def _general_improvement(self, model_id, task):
        """通用模型改进方法
        
        Args:
            model_id: 模型ID
            task: 任务类型
        """
        try:
            logger.info(f"对模型 {model_id} 应用通用改进: {task}")
            
            model = self.model_references.get(model_id)
            if not model:
                return
            
            # 1. 从知识库获取相关知识
            relevant_knowledge = []
            if self.knowledge_model and hasattr(self.knowledge_model, 'search_knowledge'):
                # 根据模型ID和任务类型搜索相关知识
                search_query = f"{model_id} {task}"
                relevant_knowledge = self.knowledge_model.search_knowledge(search_query, limit=5)
                
            # 2. 根据不同的任务类型执行不同的改进
            if task.endswith('enhancement'):
                # 基础增强任务
                if hasattr(model, 'update_parameters'):
                    # 为模型提供相关知识作为更新参数
                    model.update_parameters(relevant_knowledge)
            elif task == 'foundational_knowledge_acquisition':
                # 基础知识获取
                if hasattr(model, 'acquire_knowledge'):
                    model.acquire_knowledge(relevant_knowledge)
            elif task == 'cross_domain_knowledge_transfer':
                # 跨领域知识迁移
                if hasattr(model, 'transfer_knowledge'):
                    model.transfer_knowledge(relevant_knowledge)
            
            # 3. 对于知识模型的特殊处理
            if model_id == 'knowledge' and hasattr(model, 'consolidate_knowledge'):
                model.consolidate_knowledge()
                
        except Exception as e:
            logger.error(f"执行通用改进任务时出错: {task} 对模型: {model_id}, 错误: {e}")

    def _evaluate_model(self, model_or_id):
        """评估单个模型的性能
        
        参数:
            model_or_id: 模型ID（字符串）或模型对象
            
        返回:
            float: 性能分数
        """
        try:
            if isinstance(model_or_id, str):
                # 参数是模型ID
                model_id = model_or_id
                model = self.model_references.get(model_id)
                if not model:
                    logger.warning(f"无法找到模型: {model_id}")
                    return 0.3  # 无法找到模型，返回低分
            else:
                # 参数是模型对象
                model = model_or_id
                model_id = model.model_id if hasattr(model, 'model_id') else 'unknown'
            
            # 使用实际的性能评估方法
            if hasattr(model, 'get_performance_metrics'):
                metrics = model.get_performance_metrics()
                if metrics:
                    # 计算综合性能分数
                    score = 0
                    weight_sum = 0
                    weights = {
                        'accuracy': 0.3,
                        'precision': 0.2,
                        'recall': 0.2,
                        'f1_score': 0.2,
                        'speed': 0.1
                    }
                    
                    for metric, value in metrics.items():
                        if metric in weights and isinstance(value, (int, float)):
                            score += value * weights[metric]
                            weight_sum += weights[metric]
                    
                    return score / weight_sum if weight_sum > 0 else 0.5
            elif hasattr(model, 'evaluate'):
                # 如果模型有evaluate方法，调用它
                result = model.evaluate()
                if isinstance(result, (int, float)):
                    return min(max(result, 0), 1)  # 确保在0-1范围内
            
            # 对于知识模型，我们可以特殊处理
            if model_id == 'knowledge' and hasattr(model, 'get_knowledge_coverage'):
                coverage = model.get_knowledge_coverage()
                if isinstance(coverage, (int, float)):
                    return min(max(coverage/100, 0), 1)
            
            # 如果没有具体性能指标，返回基础分
            return 0.5
        except Exception as e:
            logger.error(f"评估模型 {model_id} 性能时出错: {str(e)}")
            return 0.3

    def _evaluate_all_models(self):
        """评估所有模型的性能"""
        for model_id, model in self.model_references.items():
            try:
                # 评估模型性能
                performance = self._evaluate_model(model)
                
                # 更新性能历史
                self.performance_history[model_id].append({
                    'timestamp': datetime.now(),
                    'score': performance
                })
                
                # 计算改进率
                improvement_rate = self._calculate_improvement_rate(model_id)
                
                # 更新模型状态跟踪
                self.model_status_tracking[model_id] = {
                    'last_trained': datetime.now(),
                    'performance_score': performance,
                    'improvement_rate': improvement_rate,
                    'training_priority': self._calculate_training_priority(model_id)
                }
                
                logger.info(f"评估模型 {model_id}: 性能分数 {performance:.2f}")
            except Exception as e:
                logger.error(f"评估模型 {model_id} 时出错: {e}")
    
    def _calculate_improvement_rate(self, model_id):
        """计算模型性能的改进率"""
        history = self.performance_history.get(model_id, [])
        if len(history) < 2:
            return 0.0
        
        # 计算最近几次性能的平均改进率
        recent_history = history[-5:]  # 获取最近5次评估
        if len(recent_history) < 2:
            return 0.0
        
        # 计算改进率
        improvements = []
        for i in range(1, len(recent_history)):
            prev_score = recent_history[i-1]['score']
            curr_score = recent_history[i]['score']
            if prev_score > 0:
                improvement = (curr_score - prev_score) / prev_score
                improvements.append(improvement)
        
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    def _calculate_training_priority(self, model_id):
        """计算模型的训练优先级"""
        status = self.model_status_tracking.get(model_id, {})
        performance = status.get('performance_score', 0.0)
        improvement_rate = status.get('improvement_rate', 0.0)
        
        # 性能越低，优先级越高
        # 改进率越低，优先级越高
        priority = (1.0 - performance) * 0.7 + (1.0 - max(improvement_rate, 0.0)) * 0.3
        
        return priority
    
    def _select_next_improvement_target(self):
        """选择下一个需要改进的目标模型和任务"""
        # 按优先级排序模型
        prioritized_models = sorted(
            self.model_status_tracking.items(),
            key=lambda x: x[1]['training_priority'],
            reverse=True
        )
        
        # 选择优先级最高的模型
        if prioritized_models:
            model_id, _ = prioritized_models[0]
            # 选择适合该模型的任务
            task = self._select_task_for_model(model_id)
            return model_id, task
        
        return None, None
    
    def _select_task_for_model(self, model_id):
        """为指定模型选择合适的任务"""
        # 根据模型类型选择任务
        task_map = {
            'language': 'language_enhancement',
            'knowledge': 'knowledge_enhancement',
            'audio': 'audio_enhancement',
            'vision_image': 'vision_enhancement',
            'vision_video': 'vision_enhancement',
            'spatial': 'spatial_enhancement',
            'stereo_spatial': 'spatial_enhancement',
            'sensor': 'sensor_enhancement',
            'computer': 'computer_enhancement',
            'motion': 'motion_enhancement',
            'programming': 'programming_enhancement'
        }
        
        return task_map.get(model_id, 'general_enhancement')
    
    def _execute_improvement_task(self, model_id, task):
        """执行模型改进任务"""
        try:
            logger.info(f"开始执行改进任务: {task} 对模型: {model_id}")
            
            # 获取模型引用
            model = self.model_references.get(model_id)
            if not model:
                logger.warning(f"无法找到模型: {model_id}")
                return
            
            # 执行具体任务
            if hasattr(model, 'improve'):
                # 如果模型有improve方法，调用它
                model.improve(task, self.knowledge_model)
            else:
                # 否则使用通用改进方法
                self._general_improvement(model_id, task)
            
            logger.info(f"完成改进任务: {task} 对模型: {model_id}")
        except Exception as e:
            logger.error(f"执行改进任务时出错: {task} 对模型: {model_id}, 错误: {e}")
    
    def _general_improvement(self, model_id, task):
        """通用模型改进方法"""
        try:
            logger.info(f"对模型 {model_id} 应用通用改进: {task}")
            
            model = self.model_references.get(model_id)
            if not model:
                return
            
            # 1. 从知识库获取相关知识
            relevant_knowledge = []
            if self.knowledge_model and hasattr(self.knowledge_model, 'search_knowledge'):
                # 根据模型ID和任务类型搜索相关知识
                search_query = f"{model_id} {task}"
                relevant_knowledge = self.knowledge_model.search_knowledge(search_query, limit=5)
                
            # 2. 根据不同的任务类型执行不同的改进
            if task.endswith('enhancement'):
                # 基础增强任务
                if hasattr(model, 'update_parameters'):
                    # 为模型提供相关知识作为更新参数
                    model.update_parameters(relevant_knowledge)
            elif task == 'foundational_knowledge_acquisition':
                # 基础知识获取
                if hasattr(model, 'acquire_knowledge'):
                    model.acquire_knowledge(relevant_knowledge)
            elif task == 'cross_domain_knowledge_transfer':
                # 跨领域知识迁移
                if hasattr(model, 'transfer_knowledge'):
                    model.transfer_knowledge(relevant_knowledge)
            
            # 3. 对于知识模型的特殊处理
            if model_id == 'knowledge' and hasattr(model, 'consolidate_knowledge'):
                model.consolidate_knowledge()
                
        except Exception as e:
            logger.error(f"执行通用改进任务时出错: {task} 对模型: {model_id}, 错误: {e}")
    
    def reset_learning(self):
        """重置学习过程"""
        # 重置学习相关的数据结构
        self.performance_history = defaultdict(list)
        self.improvement_suggestions = []
        self.model_status_tracking = defaultdict(lambda: {
            'last_trained': None,
            'performance_score': 0.0,
            'improvement_rate': 0.0,
            'training_priority': 0
        })
        
        # 重置进度和状态
        self.learning_progress = 0.0
        self.current_learning_status = 'idle'
        self.learning_domains = []
        self.learning_priority = 'balanced'
        self.learning_logs = []
        
        logger.info("重置自主学习过程")

    def _generate_learning_report(self):
        """生成学习报告"""
        # 评估当前知识状态的各个维度
        knowledge_completeness = self._assess_knowledge_completeness()
        knowledge_consolidation = self._verify_knowledge_consolidation()
        knowledge_retention = self._assess_learning_retention()
        
        report = {
            'timestamp': datetime.now(),
            'learning_progress': self.learning_progress,
            'knowledge_completeness': knowledge_completeness,
            'knowledge_consolidation': knowledge_consolidation,
            'knowledge_retention': knowledge_retention,
            'learning_efficiency': self.self_monitoring['learning_efficiency'],
            'problem_solving_ability': self.self_monitoring['problem_solving_ability'],
            'improvement_suggestions': self.improvement_suggestions.copy()
        }
        
        # 清空改进建议，为下一轮做准备
        self.improvement_suggestions = []
        
        # 记录报告
        self._add_learning_log(f"学习报告生成: 进度{self.learning_progress}%, 完整性{knowledge_completeness:.2f}, 巩固度{knowledge_consolidation:.2f}, 保留率{knowledge_retention:.2f}")
    
    def suggest_improvement(self, suggestion):
        """添加改进建议"""
        self.improvement_suggestions.append(suggestion)
        logger.info(f"添加改进建议: {suggestion}")
        
    def get_learning_progress(self):
        """获取自主学习进度"""
        return {
            "progress": round(self.learning_progress, 2),
            "status": self.current_learning_status,
            "logs": self.learning_logs.copy(),
            "domains": self.learning_domains,
            "priority": self.learning_priority
        }
    
    def _add_learning_log(self, message):
        """添加学习日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message
        }
        
        self.learning_logs.append(log_entry)
        
        # 限制日志数量
        if len(self.learning_logs) > self.max_logs:
            self.learning_logs = self.learning_logs[-self.max_logs:]
    
    def update_config(self, config):
        """更新配置"""
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        logger.info(f"更新自主学习配置: {config}")
    
    def reset_autonomous_learning(self):
        """重置自主学习过程"""
        self.performance_history = defaultdict(list)
        self.improvement_suggestions = []
        self.model_status_tracking = defaultdict(lambda: {
            'last_trained': None,
            'performance_score': 0.0,
            'improvement_rate': 0.0,
            'training_priority': 0
        })
        
        # 重置进度和状态
        self.learning_progress = 0.0
        self.current_learning_status = 'idle'
        self.learning_domains = []
        self.learning_priority = 'balanced'
        self.learning_logs = []
        
        logger.info("重置自主学习过程")



    def start_autonomous_learning(self, config=None, model_id=None, autonomous_config=None):
        """启动自主学习（兼容性方法）
        
        Args:
            config: 配置字典（可能包含domains, priority等）
            model_id: 模型ID（可选）
            autonomous_config: 自主学习配置（可选）
            
        Returns:
            dict: 包含成功状态和消息的结果
        """
        try:
            # 处理不同的调用方式
            if config is not None:
                # 来自model_training_api.py的调用方式
                domains = config.get('domains', [])
                priority = config.get('priority', 'balanced')
                self.start_autonomous_learning_cycle(domains=domains, priority=priority)
                return {
                    'success': True,
                    'message': f"自主学习已启动，领域: {domains}, 优先级: {priority}"
                }
            elif model_id is not None and autonomous_config is not None:
                # 来自enhanced_training_system.py的调用方式
                domains = autonomous_config.get('domains', [])
                priority = autonomous_config.get('priority', 'balanced')
                self.start_autonomous_learning_cycle(domains=domains, priority=priority)
                return {
                    'success': True,
                    'message': f"模型 {model_id} 的自主学习已启动"
                }
            else:
                # 默认启动
                self.start_autonomous_learning_cycle()
                return {
                    'success': True,
                    'message': "自主学习已启动"
                }
        except Exception as e:
            logger.error(f"启动自主学习失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def learn_from_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        AGI级交互学习 - 从交互中学习并返回学习结果
        参数:
            interaction_data: 包含交互详情的字典
        返回: 包含学习结果和元数据的字典
        """
        if not self.initialized:
            error_handler.log_warning("自我学习模块未初始化", "SelfLearning")
            return {'success': False, 'reason': 'system_not_initialized'}
        
        if not self.learning_enabled:
            error_handler.log_warning("学习功能已禁用", "SelfLearning")
            return {'success': False, 'reason': 'learning_disabled'}
        
        try:
            self.learning_stats['total_learning_sessions'] += 1
            
            # 情感意识集成
            emotional_context = self._integrate_emotional_awareness(interaction_data)
            
            # 价值对齐检查
            value_alignment = self._check_value_alignment(interaction_data)
            
            # 多层级学习处理
            learning_results = {
                'basic_learning': self._basic_learning(interaction_data),
                'meta_learning': self._meta_learning(interaction_data),
                'transfer_learning': self._transfer_learning(interaction_data),
                'causal_learning': self._causal_learning(interaction_data),
                'reflective_learning': self._reflective_learning(interaction_data)
            }
            
            # 知识整合和巩固
            consolidation_result = self._consolidate_knowledge(interaction_data, learning_results)
            
            # 更新学习统计
            if all(result['success'] for result in learning_results.values() if result):
                self.learning_stats['successful_learnings'] += 1
                knowledge_gain = self._calculate_agi_knowledge_gain(interaction_data, learning_results)
                self.learning_stats['total_knowledge_gained'] += knowledge_gain
            else:
                self.learning_stats['failed_learnings'] += 1
            
            self.last_learning_time = datetime.now()
            
            # 存储经验并进行经验回放
            experience = self._create_agi_experience(interaction_data, learning_results, emotional_context, value_alignment)
            self.experience_replay.append(experience)
            
            # 定期进行经验回放学习
            if len(self.experience_replay) % 100 == 0:
                self._experience_replay_learning()
            
            # 自我监控更新
            self._update_self_monitoring(learning_results)
            
            # 保存知识
            self._save_knowledge()
            
            return {
                'success': True,
                'learning_results': learning_results,
                'knowledge_gain': knowledge_gain,
                'emotional_context': emotional_context,
                'value_alignment': value_alignment,
                'consolidation_result': consolidation_result
            }
            
        except Exception as e:
            logger.error(f"AGI交互学习错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.learning_stats['failed_learnings'] += 1
            return {'success': False, 'reason': str(e)}
    
    def record_comprehensive_experience(self, learning_experience: Dict[str, Any]) -> Dict[str, Any]:
        """记录全面学习经验
        
        Args:
            learning_experience: 学习经验字典，包含operation, input_data, result, context等信息
            
        Returns:
            记录结果字典
        """
        try:
            # 验证学习经验数据
            if not learning_experience or not isinstance(learning_experience, dict):
                logger.warning("无效的学习经验数据")
                return {'success': False, 'error': 'Invalid learning experience data'}
            
            # 提取关键信息
            operation = learning_experience.get('operation', 'unknown')
            input_data = learning_experience.get('input_data', {})
            result = learning_experience.get('result', {})
            cognitive_metrics = learning_experience.get('cognitive_metrics', {})
            agi_state = learning_experience.get('agi_state', {})
            
            # 创建交互数据格式
            interaction_data = {
                'type': 'learning_experience',
                'operation': operation,
                'input_data': input_data,
                'result': result,
                'timestamp': datetime.now().isoformat(),
                'cognitive_metrics': cognitive_metrics,
                'agi_state': agi_state,
                'source': 'record_comprehensive_experience'
            }
            
            # 使用现有的learn_from_interaction方法进行学习
            learning_result = self.learn_from_interaction(interaction_data)
            
            # 存储到经验回放缓冲区
            comprehensive_experience = {
                **learning_experience,
                'recorded_timestamp': datetime.now().isoformat(),
                'learning_result': learning_result,
                'processed_by': 'AGISelfLearningSystem.record_comprehensive_experience'
            }
            
            self.experience_replay.append(comprehensive_experience)
            
            # 定期进行经验回放学习
            if len(self.experience_replay) % 50 == 0:
                self._experience_replay_learning()
            
            logger.info(f"全面学习经验记录成功: {operation}")
            return {
                'success': True,
                'recorded': True,
                'operation': operation,
                'learning_result': learning_result,
                'experience_replay_size': len(self.experience_replay),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"记录全面学习经验失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'recorded': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def _integrate_emotional_awareness(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """集成情感意识"""
        if self.emotion_system:
            try:
                emotional_state = self.emotion_system.analyze_emotional_context(interaction_data)
                # 情感影响学习参数
                emotional_impact = emotional_state.get('overall_impact', 0.5)
                self.learning_parameters['base_learning_rate'] *= (0.8 + 0.4 * emotional_impact)
                return emotional_state
            except Exception as e:
                error_handler.log_warning(f"情感意识集成失败: {e}", "SelfLearning")
                return {'default_emotional_context': True}
        return {'emotional_integration': False}
    
    def _check_value_alignment(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """检查价值对齐"""
        if self.value_system:
            try:
                alignment_result = self.value_system.check_alignment(interaction_data)
                if not alignment_result['aligned']:
                    error_handler.log_warning(f"价值对齐问题: {alignment_result.get('issues', [])}", "SelfLearning")
                return alignment_result
            except Exception as e:
                error_handler.log_warning(f"价值对齐检查失败: {e}", "SelfLearning")
                return {'alignment_check': False}
        return {'value_integration': False}
    
    def _basic_learning(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """基础学习 - 模式识别和概念学习"""
        try:
            # 提取交互特征
            features = self._extract_interaction_features(interaction_data)
            
            # 模式识别
            patterns = self._identify_advanced_patterns(features, interaction_data)
            
            # 概念学习
            concepts = self._learn_advanced_concepts(features, interaction_data)
            
            # 规则提取
            rules = self._extract_advanced_rules(features, interaction_data)
            
            return {
                'success': True,
                'patterns_identified': len(patterns),
                'concepts_learned': len(concepts),
                'rules_extracted': len(rules),
                'features': features.shape if hasattr(features, 'shape') else 'non_tensor'
            }
        except Exception as e:
            logger.error(f"基础学习错误: {e}")
            return {'success': False, 'error': str(e)}
    
    def _meta_learning(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """元学习 - 学习如何学习"""
        try:
            # 提取元特征
            meta_features = self._extract_meta_features(interaction_data)
            
            # 调试日志：打印元特征的形状和内容
            logger.debug(f"meta_features shape: {meta_features.shape}, content: {meta_features.tolist()}")
            
            # 预测最佳学习策略
            strategy = self._predict_learning_strategy(meta_features)
            
            # 更新元学习模型
            loss = self._update_meta_learner(meta_features, strategy)
            
            self.learning_stats['meta_learning_improvements'] += 1
            
            return {
                'success': True,
                'strategy_predicted': strategy,
                'meta_loss': loss.item() if hasattr(loss, 'item') else loss,
                'meta_features_dim': meta_features.shape[0] if hasattr(meta_features, 'shape') else 'scalar'
            }
        except Exception as e:
            logger.error(f"元学习错误: {e}")
            return {'success': False, 'error': str(e)}
    
    def _transfer_learning(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """迁移学习 - 跨领域知识应用"""
        try:
            # 识别知识迁移机会
            transfer_opportunities = self._identify_transfer_opportunities(interaction_data)
            
            if transfer_opportunities:
                # 执行知识迁移
                transfer_results = self._execute_knowledge_transfer(interaction_data, transfer_opportunities)
                self.learning_stats['transfer_learning_successes'] += len(transfer_results['successful_transfers'])
                
                return {
                    'success': True,
                    'transfer_opportunities': len(transfer_opportunities),
                    'successful_transfers': len(transfer_results['successful_transfers']),
                    'transfer_gain': transfer_results['total_gain']
                }
            
            return {'success': True, 'transfer_opportunities': 0, 'no_transfer_needed': True}
            
        except Exception as e:
            logger.error(f"迁移学习错误: {e}")
            return {'success': False, 'error': str(e)}
    
    def _causal_learning(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """因果学习 - 发现因果关系"""
        try:
            # 因果发现
            causal_relations = self._discover_causal_relations(interaction_data)
            
            if causal_relations:
                # 构建因果模型
                causal_models = self._build_causal_models(causal_relations)
                self.learning_stats['causal_discoveries'] += len(causal_models)
                
                return {
                    'success': True,
                    'causal_relations_discovered': len(causal_relations),
                    'causal_models_built': len(causal_models),
                    'causal_strength': sum(model['strength'] for model in causal_models) / len(causal_models) if causal_models else 0
                }
            
            return {'success': True, 'causal_relations': 0, 'no_causality_detected': True}
            
        except Exception as e:
            logger.error(f"因果学习错误: {e}")
            return {'success': False, 'error': str(e)}
    
    def _reflective_learning(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """反思学习 - 自我反思和错误纠正"""
        try:
            # 自我反思
            reflection_insights = self._perform_self_reflection(interaction_data)
            
            # 错误检测和纠正
            error_corrections = self._detect_and_correct_errors(interaction_data)
            
            self.learning_stats['self_reflection_insights'] += len(reflection_insights)
            
            return {
                'success': True,
                'reflection_insights': len(reflection_insights),
                'errors_corrected': len(error_corrections),
                'learning_improvement': min(1.0, 0.1 * len(reflection_insights) + 0.2 * len(error_corrections))
            }
        except Exception as e:
            logger.error(f"反思学习错误: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_interaction_features(self, interaction_data: Dict[str, Any]) -> torch.Tensor:
        """提取交互特征"""
        # 简化的特征提取
        features = []
        
        # 基于交互类型、复杂性、新颖性等提取特征
        if 'type' in interaction_data:
            type_hash = (zlib.adler32(str(interaction_data['type']).encode('utf-8')) & 0xffffffff) % 100 / 100
            features.append(type_hash)
        
        if 'input' in interaction_data and isinstance(interaction_data['input'], dict):
            complexity = min(len(interaction_data['input']) / 20.0, 1.0)
            features.append(complexity)
        
        if 'context' in interaction_data and isinstance(interaction_data['context'], dict):
            context_richness = min(len(interaction_data['context']) / 15.0, 1.0)
            features.append(context_richness)
        
        # 确保有足够特征
        while len(features) < 10:
            features.append(0.0)
        
        return torch.tensor(features[:10], dtype=torch.float32, device=self.device)
    
    def _identify_advanced_patterns(self, features: torch.Tensor, interaction_data: Dict[str, Any]) -> List[Dict]:
        """识别高级模式"""
        patterns = []
        
        # 使用神经模型进行模式识别
        # 确保输入特征是10维的
        if features.shape[-1] != 10:
            error_handler.log_warning(f"特征维度不匹配: {features.shape[-1]}维，需要10维", "SelfLearning")
            # 调整特征维度
            if len(features.shape) == 1:
                # 一维张量，直接补零或截断
                if features.shape[0] < 10:
                    features = torch.cat([features, torch.zeros(10 - features.shape[0], device=features.device)])
                else:
                    features = features[:10]
            else:
                # 多维张量，只取前10个特征
                features = features[..., :10]
        pattern_features = self.meta_learner['feature_extractor'](features)
        pattern_score = torch.sigmoid(pattern_features.mean()).item()
        
        if pattern_score > 0.7:
            pattern_id = f"advanced_pattern_{hashlib.md5(str(interaction_data).encode()).hexdigest()[:12]}"
            pattern = {
                'id': pattern_id,
                'type': interaction_data.get('type', 'unknown'),
                'features': features.cpu().numpy().tolist(),
                'pattern_score': pattern_score,
                'timestamp': datetime.now().isoformat(),
                'context': interaction_data.get('context', {})
            }
            patterns.append(pattern)
            
            # 存储到知识架构
            self.knowledge_architecture['semantic_memory']['patterns'][pattern_id] = pattern
        
        return patterns
    
    def _learn_advanced_concepts(self, features: torch.Tensor, interaction_data: Dict[str, Any]) -> List[Dict]:
        """学习高级概念"""
        concepts = []
        
        if 'input' in interaction_data and isinstance(interaction_data['input'], dict):
            for key, value in interaction_data['input'].items():
                concept_id = f"concept_{hashlib.md5(str(key).encode()).hexdigest()[:10]}"
                
                concept = {
                    'id': concept_id,
                    'name': key,
                    'type': type(value).__name__,
                    'value_examples': [{'value': value, 'timestamp': datetime.now().isoformat()}],
                    'semantic_embedding': features.cpu().numpy().tolist(),
                    'frequency': 1,
                    'first_encountered': datetime.now().isoformat(),
                    'last_encountered': datetime.now().isoformat(),
                    'related_concepts': []
                }
                
                # 检查是否已有此概念
                if concept_id in self.knowledge_architecture['semantic_memory']['concepts']:
                    existing = self.knowledge_architecture['semantic_memory']['concepts'][concept_id]
                    existing['frequency'] += 1
                    existing['value_examples'].append({'value': value, 'timestamp': datetime.now().isoformat()})
                    existing['last_encountered'] = datetime.now().isoformat()
                    existing['semantic_embedding'] = (np.array(existing['semantic_embedding']) * 0.7 + 
                                                     np.array(concept['semantic_embedding']) * 0.3).tolist()
                else:
                    self.knowledge_architecture['semantic_memory']['concepts'][concept_id] = concept
                    concepts.append(concept)
        
        return concepts
    
    def _extract_advanced_rules(self, features: torch.Tensor, interaction_data: Dict[str, Any]) -> List[Dict]:
        """提取高级规则"""
        rules = []
        
        if 'input' in interaction_data and 'output' in interaction_data:
            input_data = interaction_data['input']
            output_data = interaction_data['output']
            
            if isinstance(input_data, dict) and isinstance(output_data, dict):
                rule_id = f"rule_{hashlib.md5(str({'input': input_data, 'output': output_data}).encode()).hexdigest()[:14]}"
                
                rule = {
                    'id': rule_id,
                    'conditions': input_data,
                    'actions': output_data,
                    'confidence': 0.8,  # 初始置信度
                    'usage_count': 0,
                    'success_count': 0,
                    'failure_count': 0,
                    'created': datetime.now().isoformat(),
                    'last_used': datetime.now().isoformat(),
                    'context': interaction_data.get('context', {}),
                    'semantic_features': features.cpu().numpy().tolist()
                }
                
                rules.append(rule)
                self.knowledge_architecture['procedural_memory']['rules'][rule_id] = rule
        
        return rules
    
    def _extract_meta_features(self, interaction_data: Dict[str, Any]) -> torch.Tensor:
        """提取元特征"""
        meta_features = []
        
        # 学习历史特征
        meta_features.append(min(self.learning_stats['total_learning_sessions'] / 1000.0, 1.0))
        meta_features.append(min(self.learning_stats['successful_learnings'] / max(1, self.learning_stats['total_learning_sessions']), 1.0))
        
        # 知识基础特征 - 安全访问
        try:
            total_concepts = len(self.knowledge_architecture['semantic_memory']['concepts'])
        except (KeyError, TypeError):
            total_concepts = 0
        meta_features.append(min(total_concepts / 500.0, 1.0))
        
        # 复杂性特征
        if 'input' in interaction_data:
            try:
                input_complexity = self._calculate_complexity(interaction_data['input'])
                meta_features.append(input_complexity)
            except Exception as e:
                logger.error(f"提取元特征时出错: {e}")
                meta_features.append(0.0)
        else:
            meta_features.append(0.0)
        
        # 新颖性特征
        try:
            novelty = self._calculate_novelty(interaction_data)
            meta_features.append(novelty)
        except Exception as e:
            logger.error(f"计算新颖性时出错: {e}")
            meta_features.append(0.0)
        
        # 添加额外的特征以确保总共有10个特征
        meta_features.append(min(self.learning_stats['meta_learning_improvements'] / 100.0, 1.0))
        meta_features.append(min(self.learning_stats['transfer_learning_successes'] / 100.0, 1.0))
        meta_features.append(min(self.learning_stats['causal_discoveries'] / 100.0, 1.0))
        meta_features.append(min(self.learning_stats['self_reflection_insights'] / 100.0, 1.0))
        meta_features.append(min(self.learning_stats['long_term_goals_achieved'] / 100.0, 1.0))
        
        # 确保有足够特征（额外的安全保障）
        while len(meta_features) < 10:
            meta_features.append(0.0)
        
        # 调试日志：打印元特征的长度和内容
        logger.debug(f"_extract_meta_features - meta_features length: {len(meta_features)}, content: {meta_features}")
        
        return torch.tensor(meta_features[:10], dtype=torch.float32, device=self.device)
    
    def _predict_learning_strategy(self, meta_features: torch.Tensor) -> str:
        """预测最佳学习策略"""
        # 确保输入特征是10维的
        if meta_features.shape[-1] != 10:
            error_handler.log_warning(f"元特征维度不匹配: {meta_features.shape[-1]}维，需要10维", "SelfLearning")
            # 调整特征维度
            if len(meta_features.shape) == 1:
                # 一维张量，直接补零或截断
                if meta_features.shape[0] < 10:
                    meta_features = torch.cat([meta_features, torch.zeros(10 - meta_features.shape[0], device=meta_features.device)])
                else:
                    meta_features = meta_features[:10]
            else:
                # 多维张量，只取前10个特征
                meta_features = meta_features[..., :10]
        strategy_logits = self.meta_learner['learning_strategy_predictor'](
            self.meta_learner['feature_extractor'](meta_features)
        )
        
        strategy_idx = torch.argmax(strategy_logits).item()
        strategies = [
            'deep_analysis', 'pattern_recognition', 'conceptual_abstraction',
            'procedural_learning', 'social_learning', 'experimental_learning',
            'reflective_learning', 'creative_synthesis', 'critical_evaluation',
            'integrative_learning'
        ]
        
        return strategies[strategy_idx % len(strategies)]
    
    def _update_meta_learner(self, meta_features: torch.Tensor, strategy: str) -> float:
        """更新元学习模型"""
        strategies = [
            'deep_analysis', 'pattern_recognition', 'conceptual_abstraction',
            'procedural_learning', 'social_learning', 'experimental_learning',
            'reflective_learning', 'creative_synthesis', 'critical_evaluation',
            'integrative_learning'
        ]
        
        target = torch.zeros(len(strategies), device=self.device)
        target[strategies.index(strategy) if strategy in strategies else 0] = 1.0
        
        # 确保输入特征是10维的
        if meta_features.shape[-1] != 10:
            error_handler.log_warning(f"元特征维度不匹配: {meta_features.shape[-1]}维，需要10维", "SelfLearning")
            # 调整特征维度
            if len(meta_features.shape) == 1:
                # 一维张量，直接补零或截断
                if meta_features.shape[0] < 10:
                    meta_features = torch.cat([meta_features, torch.zeros(10 - meta_features.shape[0], device=meta_features.device)])
                else:
                    meta_features = meta_features[:10]
            else:
                # 多维张量，只取前10个特征
                meta_features = meta_features[..., :10]
        
        strategy_logits = self.meta_learner['learning_strategy_predictor'](
            self.meta_learner['feature_extractor'](meta_features)
        )
        
        loss = nn.CrossEntropyLoss()(strategy_logits.unsqueeze(0), target.unsqueeze(0))
        
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()
        
        return loss
    
    def _identify_transfer_opportunities(self, interaction_data: Dict[str, Any]) -> List[Dict]:
        """识别迁移学习机会"""
        opportunities = []
        
        # 简化的迁移机会识别
        if 'input' in interaction_data and isinstance(interaction_data['input'], dict):
            input_keys = set(interaction_data['input'].keys())
            
            # 检查已有知识中的相似概念
            for concept_id, concept in self.knowledge_architecture['semantic_memory']['concepts'].items():
                if concept['name'] in input_keys:
                    opportunity = {
                        'source_concept': concept_id,
                        'target_domain': interaction_data.get('type', 'unknown'),
                        'similarity_score': 0.7,
                        'transfer_potential': 0.6
                    }
                    opportunities.append(opportunity)
        
        return opportunities
    
    def _execute_knowledge_transfer(self, interaction_data: Dict[str, Any], opportunities: List[Dict]) -> Dict[str, Any]:
        """执行知识迁移"""
        successful_transfers = []
        total_gain = 0.0
        
        for opportunity in opportunities:
            try:
                # 知识迁移增益计算 - 基于机会潜力而非随机值
                base_efficiency = 0.7  # 基础迁移效率
                quality_factor = opportunity.get('quality', 1.0)  # 机会质量因子
                transfer_gain = opportunity['transfer_potential'] * base_efficiency * quality_factor
                total_gain += transfer_gain
                
                successful_transfers.append({
                    'opportunity': opportunity,
                    'transfer_gain': transfer_gain,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                error_handler.log_warning(f"知识迁移失败: {e}", "SelfLearning")
        
        return {
            'successful_transfers': successful_transfers,
            'total_gain': total_gain
        }
    
    def _discover_causal_relations(self, interaction_data: Dict[str, Any]) -> List[Dict]:
        """发现因果关系"""
        causal_relations = []
        
        # 简化的因果发现
        if 'input' in interaction_data and 'output' in interaction_data:
            input_data = interaction_data['input']
            output_data = interaction_data['output']
            
            if isinstance(input_data, dict) and isinstance(output_data, dict):
                for input_key in input_data.keys():
                    for output_key in output_data.keys():
                        # 基于实际数据关联性计算因果强度，而非随机生成
                        causal_strength = self._calculate_causal_strength(
                            input_data[input_key], 
                            output_data[output_key],
                            input_key,
                            output_key
                        )
                        
                        if causal_strength > 0.6:  # 阈值
                            relation = {
                                'cause': input_key,
                                'effect': output_key,
                                'strength': causal_strength,
                                'context': interaction_data.get('context', {}),
                                'evidence_count': 1,
                                'first_observed': datetime.now().isoformat(),
                                'last_observed': datetime.now().isoformat()
                            }
                            causal_relations.append(relation)
        
        return causal_relations
    
    def _build_causal_models(self, causal_relations: List[Dict]) -> List[Dict]:
        """构建因果模型"""
        causal_models = []
        
        for relation in causal_relations:
            model_id = f"causal_model_{hashlib.md5((relation['cause'] + '_' + relation['effect']).encode()).hexdigest()[:10]}"
            
            causal_model = {
                'id': model_id,
                'cause': relation['cause'],
                'effect': relation['effect'],
                'strength': relation['strength'],
                'evidence_count': relation['evidence_count'],
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'predictive_power': relation['strength'] * 0.8,
                'explanatory_power': relation['strength'] * 0.7
            }
            
            # 存储到知识架构
            self.knowledge_architecture['causal_models'][model_id] = causal_model
            causal_models.append(causal_model)
        
        return causal_models
    
    def _perform_self_reflection(self, interaction_data: Dict[str, Any]) -> List[Dict]:
        """执行自我反思"""
        insights = []
        
        # 反思学习效率
        efficiency_insight = {
            'type': 'learning_efficiency',
            'insight': f"当前学习效率: {self.self_monitoring['learning_efficiency']:.2f}",
            'suggestion': '尝试不同的学习策略以提高效率',
            'confidence': 0.7,
            'timestamp': datetime.now().isoformat()
        }
        insights.append(efficiency_insight)
        
        # 反思知识保留
        retention_insight = {
            'type': 'knowledge_retention',
            'insight': f"知识保留率: {self.self_monitoring['knowledge_retention']:.2f}",
            'suggestion': '增加复习和巩固的频率',
            'confidence': 0.6,
            'timestamp': datetime.now().isoformat()
        }
        insights.append(retention_insight)
        
        # 存储到元记忆
        for insight in insights:
            insight_id = f"insight_{hashlib.md5(insight['insight'].encode()).hexdigest()[:8]}"
            self.knowledge_architecture['meta_memory']['insights'][insight_id] = insight
        
        return insights
    
    def _detect_and_correct_errors(self, interaction_data: Dict[str, Any]) -> List[Dict]:
        """检测和纠正错误"""
        corrections = []
        
        # 简化的错误检测
        if 'feedback' in interaction_data and isinstance(interaction_data['feedback'], dict):
            feedback = interaction_data['feedback']
            
            if 'error' in feedback or 'correction' in feedback:
                correction = {
                    'type': 'feedback_based_correction',
                    'original': interaction_data.get('output', {}),
                    'corrected': feedback.get('correction', {}),
                    'error_type': feedback.get('error', 'unknown'),
                    'learning_point': '从反馈中学习纠正',
                    'timestamp': datetime.now().isoformat()
                }
                corrections.append(correction)
        
        return corrections
    
    def _consolidate_knowledge(self, interaction_data: Dict[str, Any], learning_results: Dict[str, Any]) -> Dict[str, Any]:
        """巩固知识"""
        try:
            # 知识整合到长期记忆
            consolidated_items = 0
            
            # 整合概念
            if 'basic_learning' in learning_results and learning_results['basic_learning']['success']:
                consolidated_items += learning_results['basic_learning'].get('concepts_learned', 0)
            
            # 整合因果模型
            if 'causal_learning' in learning_results and learning_results['causal_learning']['success']:
                consolidated_items += learning_results['causal_learning'].get('causal_models_built', 0)
            
            # 整合模式
            if 'basic_learning' in learning_results and learning_results['basic_learning']['success']:
                consolidated_items += learning_results['basic_learning'].get('patterns_identified', 0)
            
            # 整合规则
            if 'basic_learning' in learning_results and learning_results['basic_learning']['success']:
                consolidated_items += learning_results['basic_learning'].get('rules_extracted', 0)
            
            # 验证知识巩固效果
            consolidation_quality = self._verify_knowledge_consolidation()
            
            # 更新知识巩固状态
            consolidation_strength = min(1.0, (consolidated_items / 10.0) * consolidation_quality)
            
            return {
                'success': True,
                'consolidated_items': consolidated_items,
                'consolidation_strength': consolidation_strength,
                'consolidation_quality': consolidation_quality,
                'memory_impact': consolidation_strength * 0.8
            }
            
        except Exception as e:
            logger.error(f"知识巩固错误: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_agi_knowledge_gain(self, interaction_data: Dict[str, Any], learning_results: Dict[str, Any]) -> float:
        """计算AGI级知识增益"""
        total_gain = 0.0
        
        # 基础学习增益
        if learning_results['basic_learning']['success']:
            base_gain = (learning_results['basic_learning'].get('patterns_identified', 0) * 0.3 +
                        learning_results['basic_learning'].get('concepts_learned', 0) * 0.4 +
                        learning_results['basic_learning'].get('rules_extracted', 0) * 0.3)
            total_gain += base_gain
        
        # 元学习增益
        if learning_results['meta_learning']['success']:
            meta_gain = 0.5 * (1.0 - learning_results['meta_learning'].get('meta_loss', 1.0))
            total_gain += meta_gain
        
        # 迁移学习增益
        if learning_results['transfer_learning']['success']:
            transfer_gain = learning_results['transfer_learning'].get('transfer_gain', 0.0)
            total_gain += transfer_gain
        
        # 因果学习增益
        if learning_results['causal_learning']['success']:
            causal_gain = learning_results['causal_learning'].get('causal_strength', 0.0) * 0.8
            total_gain += causal_gain
        
        # 反思学习增益
        if learning_results['reflective_learning']['success']:
            reflective_gain = learning_results['reflective_learning'].get('learning_improvement', 0.0)
            total_gain += reflective_gain
        
        return min(total_gain, 5.0)  # 限制最大增益
    
    def _create_agi_experience(self, interaction_data: Dict[str, Any], learning_results: Dict[str, Any],
                             emotional_context: Dict[str, Any], value_alignment: Dict[str, Any]) -> Dict[str, Any]:
        """创建AGI级经验"""
        experience = {
            'interaction': interaction_data,
            'learning_results': learning_results,
            'emotional_context': emotional_context,
            'value_alignment': value_alignment,
            'timestamp': datetime.now().isoformat(),
            'knowledge_gain': self._calculate_agi_knowledge_gain(interaction_data, learning_results),
            'self_monitoring_snapshot': self.self_monitoring.copy(),
            'learning_parameters_snapshot': self.learning_parameters.copy()
        }
        
        # 存储到情景记忆
        self.knowledge_architecture['episodic_memory'].append(experience)
        
        return experience
    
    def _experience_replay_learning(self):
        """经验回放学习"""
        if len(self.experience_replay) < 10:
            return
        
        try:
            # 确定性采样经验
            batch_size = min(32, len(self.experience_replay))
            experience_list = list(self.experience_replay)
            indices = list(range(len(experience_list)))
            sampled_indices = sorted(indices, key=lambda x: (zlib.adler32(str(str(experience_list).encode('utf-8')) & 0xffffffff) + str(x) + "sample"))[:batch_size]
            batch = [experience_list[i] for i in sampled_indices]
            
            total_replay_gain = 0.0
            
            for experience in batch:
                # 从经验中学习
                replay_learning = self.learn_from_interaction(experience['interaction'])
                if replay_learning['success']:
                    total_replay_gain += replay_learning.get('knowledge_gain', 0.0)
            
            logger.info(f"经验回放完成，增益: {total_replay_gain:.2f}")
            
        except Exception as e:
            logger.error(f"经验回放错误: {e}")
    
    def _update_self_monitoring(self, learning_results: Dict[str, Any]):
        """更新自我监控"""
        # 更新学习效率
        efficiency_improvement = 0.0
        if learning_results['meta_learning']['success']:
            efficiency_improvement += 0.1 * (1.0 - learning_results['meta_learning'].get('meta_loss', 1.0))
        
        if learning_results['reflective_learning']['success']:
            efficiency_improvement += 0.05 * learning_results['reflective_learning'].get('learning_improvement', 0.0)
        
        self.self_monitoring['learning_efficiency'] = min(1.0, self.self_monitoring['learning_efficiency'] + efficiency_improvement)
        
        # 更新知识保留
        if learning_results['basic_learning']['success']:
            retention_improvement = 0.02 * learning_results['basic_learning'].get('concepts_learned', 0)
            self.self_monitoring['knowledge_retention'] = min(1.0, self.self_monitoring['knowledge_retention'] + retention_improvement)
        
        # 更新问题解决能力
        if learning_results['causal_learning']['success']:
            problem_solving_improvement = 0.15 * learning_results['causal_learning'].get('causal_strength', 0.0)
            self.self_monitoring['problem_solving_ability'] = min(1.0, self.self_monitoring['problem_solving_ability'] + problem_solving_improvement)
    
    def _calculate_complexity(self, data: Any) -> float:
        """计算数据复杂性"""
        if isinstance(data, dict):
            return min(len(data) / 25.0, 1.0)
        elif isinstance(data, list):
            return min(len(data) / 50.0, 1.0)
        elif isinstance(data, str):
            return min(len(data) / 200.0, 1.0)
        return 0.3
    
    def _calculate_novelty(self, interaction_data: Dict[str, Any]) -> float:
        """计算新颖性"""
        novelty = 1.0
        
        if 'input' in interaction_data and isinstance(interaction_data['input'], dict):
            for key in interaction_data['input'].keys():
                # 检查概念是否已知
                concept_known = any(concept['name'] == key for concept in self.knowledge_architecture['semantic_memory']['concepts'].values())
                if concept_known:
                    novelty *= 0.85  # 已知概念降低新颖性
        
        # 基于交互类型的新颖性
        interaction_type = interaction_data.get('type', 'unknown')
        type_novelty = 1.0 - ((zlib.adler32(str(interaction_type).encode('utf-8')) & 0xffffffff) % 100) / 500.0
        novelty *= type_novelty
        
        return max(0.1, novelty)

    def _calculate_causal_strength(self, input_value: Any, output_value: Any, 
                                 input_key: str, output_key: str) -> float:
        """计算因果强度
        
        参数:
            input_value: 输入值
            output_value: 输出值
            input_key: 输入键名
            output_key: 输出键名
            
        返回:
            float: 因果强度 (0-1)
        """
        try:
            # 将输入和输出值转换为可比较的数值
            input_numeric = self._value_to_numeric(input_value)
            output_numeric = self._value_to_numeric(output_value)
            
            # 如果无法转换为数值，使用基于类型的简单启发式方法
            if input_numeric is None or output_numeric is None:
                return self._calculate_type_based_causal_strength(input_value, output_value, input_key, output_key)
            
            # 计算数值之间的相关性
            # 使用简单的线性关系检测
            if isinstance(input_numeric, (int, float)) and isinstance(output_numeric, (int, float)):
                # 如果两个都是数值，计算归一化的关系强度
                if input_numeric == 0:
                    return 0.5 if output_numeric != 0 else 0.0
                
                ratio = abs(output_numeric / input_numeric)
                # 将比率映射到0-1范围，理想比率接近1表示强关系
                strength = 1.0 - min(abs(ratio - 1.0), 1.0)
                return max(0.0, min(1.0, strength))
            
            # 对于列表或数组，计算相似度
            elif isinstance(input_numeric, (list, np.ndarray)) and isinstance(output_numeric, (list, np.ndarray)):
                if len(input_numeric) == 0 or len(output_numeric) == 0:
                    return 0.0
                
                # 计算余弦相似度或简单重叠
                try:
                    # 转换为numpy数组进行计算
                    input_arr = np.array(input_numeric)
                    output_arr = np.array(output_numeric)
                    
                    # 确保形状匹配
                    min_len = min(len(input_arr), len(output_arr))
                    input_arr = input_arr[:min_len]
                    output_arr = output_arr[:min_len]
                    
                    # 计算相关性
                    if min_len > 1:
                        correlation = np.corrcoef(input_arr, output_arr)[0, 1]
                        if np.isnan(correlation):
                            correlation = 0.0
                        # 将相关性转换为0-1范围
                        return (correlation + 1) / 2  # 将[-1,1]映射到[0,1]
                    else:
                        # 单个元素，比较是否相等
                        return 1.0 if input_arr[0] == output_arr[0] else 0.0
                except Exception as e:
                    logger.error(f"计算列表相关性时出错: {e}")
                    return 0.0
            else:
                # 其他类型，使用默认的相似度计算
                return self._calculate_type_based_causal_strength(input_value, output_value, input_key, output_key)
        except Exception as e:
            logger.error(f"计算因果强度时出错: {e}")
            return 0.0

    def _value_to_numeric(self, value: Any) -> Optional[Union[int, float, list, np.ndarray]]:
        """将任意值转换为数值形式"""
        try:
            if isinstance(value, (int, float)):
                return value
            elif isinstance(value, str):
                # 尝试转换为数字
                try:
                    return float(value)
                except ValueError:
                    # 如果是字符串，返回其哈希值（归一化到0-1）
                    return ((zlib.adler32(str(value).encode('utf-8')) & 0xffffffff) % 10000) / 10000.0
            elif isinstance(value, (list, np.ndarray)):
                # 递归转换列表中的每个元素
                numeric_list = []
                for item in value:
                    numeric_item = self._value_to_numeric(item)
                    if numeric_item is None:
                        return None
                    numeric_list.append(numeric_item)
                return numeric_list
            elif isinstance(value, dict):
                # 对于字典，使用其值的哈希值
                return ((zlib.adler32(str(str(sorted(value.items().encode('utf-8')) & 0xffffffff)))) % 10000) / 10000.0
            else:
                # 其他类型，尝试转换为字符串再哈希
                return ((zlib.adler32(str(str(value).encode('utf-8')) & 0xffffffff)) % 10000) / 10000.0
        except Exception as e:
            logger.error(f"转换值为数值时出错: {e}")
            return None

    def _calculate_type_based_causal_strength(self, input_value: Any, output_value: Any, 
                                            input_key: str, output_key: str) -> float:
        """基于类型计算因果强度"""
        # 基于输入和输出类型的简单启发式
        input_type = type(input_value).__name__
        output_type = type(output_value).__name__
        
        # 如果类型相同，给予较高的基础分数
        base_strength = 0.5 if input_type == output_type else 0.3
        
        # 基于键名相似度的调整
        key_similarity = self._calculate_key_similarity(input_key, output_key)
        
        # 基于值长度的调整（如果适用）
        length_factor = 1.0
        if isinstance(input_value, (str, list, dict)) and isinstance(output_value, (str, list, dict)):
            len_input = len(input_value)
            len_output = len(output_value)
            if len_input > 0 and len_output > 0:
                length_factor = 1.0 - min(abs(len_input - len_output) / max(len_input, len_output), 1.0)
        
        final_strength = base_strength * key_similarity * length_factor
        return max(0.0, min(1.0, final_strength))

    def _calculate_key_similarity(self, key1: str, key2: str) -> float:
        """计算两个键名的相似度"""
        # 简单的相似度计算：共同前缀长度和编辑距离
        common_prefix = 0
        min_len = min(len(key1), len(key2))
        for i in range(min_len):
            if key1[i] == key2[i]:
                common_prefix += 1
            else:
                break
        
        prefix_similarity = common_prefix / max(len(key1), len(key2))
        
        # 如果键名完全相同，返回1.0
        if key1 == key2:
            return 1.0
        
        return max(0.1, prefix_similarity)  # 确保至少0.1
    
    def get_learning_status(self) -> Dict[str, Any]:
        """
        获取详细学习状态和统计信息
        返回: 包含学习状态的字典
        """
        return {
            'system_status': {
                'initialized': self.initialized,
                'learning_enabled': self.learning_enabled,
                'system_uptime': str(datetime.now() - self.creation_time),
                'device': str(self.device)
            },
            'learning_statistics': self.learning_stats.copy(),
            'knowledge_architecture': self._get_knowledge_summary(),
            'self_monitoring': self.self_monitoring.copy(),
            'learning_parameters': self.learning_parameters.copy(),
            'integration_status': {
                'emotion_system_available': self.emotion_system is not None,
                'value_system_available': self.value_system is not None
            },
            'memory_usage': {
                'experience_replay_size': len(self.experience_replay),
                'working_memory_size': len(self.working_memory),
                'long_term_memory_size': len(self.long_term_memory)
            }
        }
    
    def _get_knowledge_summary(self) -> Dict[str, Any]:
        """获取知识架构摘要"""
        return {
            'semantic_memory': {
                'total_concepts': len(self.knowledge_architecture['semantic_memory']['concepts']),
                'total_patterns': len(self.knowledge_architecture['semantic_memory']['patterns']),
                'concept_categories': len(set(concept['name'].split('_')[0] for concept in self.knowledge_architecture['semantic_memory']['concepts'].values()))
            },
            'episodic_memory': {
                'total_experiences': len(self.knowledge_architecture['episodic_memory']),
                'time_span': self._get_memory_time_span()
            },
            'procedural_memory': {
                'total_rules': len(self.knowledge_architecture['procedural_memory']['rules']),
                'average_confidence': np.mean([rule['confidence'] for rule in self.knowledge_architecture['procedural_memory']['rules'].values()]) if self.knowledge_architecture['procedural_memory']['rules'] else 0
            },
            'causal_models': {
                'total_models': len(self.knowledge_architecture['causal_models']),
                'average_strength': np.mean([model['strength'] for model in self.knowledge_architecture['causal_models'].values()]) if self.knowledge_architecture['causal_models'] else 0
            },
            'meta_memory': {
                'total_insights': len(self.knowledge_architecture['meta_memory']['insights']),
                'recent_insights': len([insight for insight in self.knowledge_architecture['meta_memory']['insights'].values() 
                                      if datetime.fromisoformat(insight['timestamp']) > datetime.now() - timedelta(hours=24)])
            }
        }
    
    def _get_memory_time_span(self) -> str:
        """获取记忆时间跨度"""
        if not self.knowledge_architecture['episodic_memory']:
            return "无记忆"
        
        first_memory = min(exp['timestamp'] for exp in self.knowledge_architecture['episodic_memory'])
        last_memory = max(exp['timestamp'] for exp in self.knowledge_architecture['episodic_memory'])
        
        return f"{first_memory} 到 {last_memory}"
    
    def autonomous_learn(self, learning_intensity: float) -> Dict[str, Any]:
        """
        AGI级自主学习 - 基于学习强度进行自主知识获取和技能提升
        参数:
            learning_intensity: 学习强度 (0.0-1.0)
        返回: 包含学习结果的字典
        """
        try:
            if not self.initialized:
                error_handler.log_warning("AGI自我学习系统未初始化", "SelfLearning")
                return {'success': False, 'reason': 'system_not_initialized'}
            
            if not self.learning_enabled:
                error_handler.log_warning("学习功能已禁用", "SelfLearning")
                return {'success': False, 'reason': 'learning_disabled'}
            
            logger.info(f"开始自主学习，强度: {learning_intensity}")
            
            # 基于学习强度调整学习参数
            adjusted_intensity = max(0.1, min(learning_intensity, 1.0))
            self.learning_parameters['base_learning_rate'] = 0.01 * adjusted_intensity
            self.learning_parameters['exploration_rate'] = 0.15 * adjusted_intensity
            self.learning_parameters['exploitation_rate'] = 0.85 * adjusted_intensity
            
            # 生成自主学习任务
            autonomous_tasks = self._generate_autonomous_tasks(adjusted_intensity)
            
            # 执行学习任务
            learning_results = {}
            total_knowledge_gain = 0.0
            
            for task in autonomous_tasks:
                task_result = self._execute_autonomous_task(task, adjusted_intensity)
                learning_results[task['id']] = task_result
                
                if task_result.get('success', False):
                    total_knowledge_gain += task_result.get('knowledge_gain', 0.0)
            
            # 更新学习统计
            self.learning_stats['total_learning_sessions'] += 1
            self.learning_stats['successful_learnings'] += 1
            self.learning_stats['total_knowledge_gained'] += total_knowledge_gain
            
            # 更新自我监控
            self._update_self_monitoring_from_autonomous_learning(learning_results, adjusted_intensity)
            
            # 保存知识
            self._save_knowledge()
            
            logger.info(f"自主学习完成，知识增益: {total_knowledge_gain:.2f}")
            
            return {
                'success': True,
                'tasks_completed': len(autonomous_tasks),
                'total_knowledge_gain': total_knowledge_gain,
                'learning_results': learning_results,
                'adjusted_intensity': adjusted_intensity
            }
            
        except Exception as e:
            logger.error(f"自主学习错误: {e}")
            self.learning_stats['failed_learnings'] += 1
            return {'success': False, 'error': str(e)}
    
    def _generate_autonomous_tasks(self, intensity: float) -> List[Dict[str, Any]]:
        """基于学习强度生成自主学习任务"""
        tasks = []
        
        # 基础任务数量基于强度
        base_task_count = max(1, int(intensity * 5))
        
        # 任务类型分布
        task_types = [
            'pattern_discovery', 'concept_abstraction', 'causal_analysis',
            'knowledge_integration', 'skill_refinement', 'meta_learning'
        ]
        
        # 基于强度调整任务复杂度
        complexity_multiplier = 0.5 + (intensity * 0.5)
        
        for i in range(base_task_count):
            # 使用确定性选择而非随机选择
            task_type_index = i % len(task_types)
            task_type = task_types[task_type_index]
            task_id = f"autonomous_task_{int(time.time() * 1000)}_{i}"
            
            # 基于任务类型、索引和强度的确定性计算
            # 复杂性：基于任务类型和强度
            type_complexity_factor = (task_type_index + 1) / len(task_types)
            base_complexity = 0.3 + (type_complexity_factor * 0.4)  # 0.3-0.7范围
            complexity = min(1.0, base_complexity * complexity_multiplier)
            
            # 优先级：基于任务重要性和复杂性
            # 某些任务类型更重要（如meta_learning, causal_analysis）
            importance_factors = {
                'pattern_discovery': 0.7,
                'concept_abstraction': 0.8,
                'causal_analysis': 0.9,
                'knowledge_integration': 0.85,
                'skill_refinement': 0.75,
                'meta_learning': 0.95
            }
            importance = importance_factors.get(task_type, 0.8)
            priority = 0.5 + (importance * 0.5)  # 0.5-1.0范围
            
            # 估计持续时间：基于复杂性和强度
            estimated_duration = (1.0 + (complexity * 4.0)) * intensity  # 1.0-5.0范围 * 强度
            
            task = {
                'id': task_id,
                'type': task_type,
                'complexity': complexity,
                'priority': priority,
                'estimated_duration': estimated_duration,
                'created': datetime.now().isoformat(),
                'calculation_method': 'deterministic_based_on_type_and_intensity'
            }
            
            tasks.append(task)
        
        # 按优先级排序
        tasks.sort(key=lambda x: x['priority'], reverse=True)
        
        return tasks
    
    def _execute_autonomous_task(self, task: Dict[str, Any], intensity: float) -> Dict[str, Any]:
        """执行单个自主学习任务"""
        try:
            task_type = task['type']
            complexity = task['complexity']
            
            logger.info(f"执行自主学习任务: {task_type}, 复杂度: {complexity}")
            
            # 根据任务类型执行不同的学习策略
            if task_type == 'pattern_discovery':
                result = self._execute_pattern_discovery_task(complexity, intensity)
            elif task_type == 'concept_abstraction':
                result = self._execute_concept_abstraction_task(complexity, intensity)
            elif task_type == 'causal_analysis':
                result = self._execute_causal_analysis_task(complexity, intensity)
            elif task_type == 'knowledge_integration':
                result = self._execute_knowledge_integration_task(complexity, intensity)
            elif task_type == 'skill_refinement':
                result = self._execute_skill_refinement_task(complexity, intensity)
            elif task_type == 'meta_learning':
                result = self._execute_meta_learning_task(complexity, intensity)
            else:
                result = self._execute_general_learning_task(complexity, intensity)
            
            # 添加任务ID到结果
            result['task_id'] = task['id']
            result['task_type'] = task_type
            
            return result
            
        except Exception as e:
            logger.error(f"执行自主学习任务失败: {task['id']}, 错误: {e}")
            return {
                'success': False,
                'task_id': task['id'],
                'task_type': task['type'],
                'error': str(e)
            }
    
    def _execute_pattern_discovery_task(self, complexity: float, intensity: float) -> Dict[str, Any]:
        """执行模式发现任务"""
        
        synthetic_data = self._generate_synthetic_pattern_data(complexity, intensity)
        
        # 使用元学习器进行模式识别
        features = torch.tensor(synthetic_data, dtype=torch.float32, device=self.device)
        pattern_features = self.meta_learner['feature_extractor'](features)
        pattern_score = torch.sigmoid(pattern_features.mean()).item()
        
        # 如果发现显著模式，存储到知识架构
        patterns_discovered = 0
        if pattern_score > 0.6 + (complexity * 0.2):
            pattern_id = f"autonomous_pattern_{hashlib.md5(str(synthetic_data).encode()).hexdigest()[:10]}"
            pattern = {
                'id': pattern_id,
                'type': 'autonomous_discovery',
                'features': synthetic_data,
                'pattern_score': pattern_score,
                'complexity': complexity,
                'timestamp': datetime.now().isoformat(),
                'discovery_method': 'autonomous_learning'
            }
            
            self.knowledge_architecture['semantic_memory']['patterns'][pattern_id] = pattern
            patterns_discovered = 1
        
        knowledge_gain = patterns_discovered * (0.2 + complexity * 0.3)
        
        return {
            'success': True,
            'patterns_discovered': patterns_discovered,
            'pattern_score': pattern_score,
            'knowledge_gain': knowledge_gain,
            'complexity': complexity
        }
    
    def _execute_concept_abstraction_task(self, complexity: float, intensity: float) -> Dict[str, Any]:
        """执行概念抽象任务"""
        # 分析现有概念，尝试进行抽象
        existing_concepts = list(self.knowledge_architecture['semantic_memory']['concepts'].values())
        concepts_abstracted = 0
        
        if len(existing_concepts) >= 2:
            # 尝试找到可以抽象的概念对
            for i in range(min(3, len(existing_concepts))):
                if len(existing_concepts) > i + 1:
                    concept1 = existing_concepts[i]
                    concept2 = existing_concepts[i + 1]
                    
                    # 确定性概念抽象逻辑 - 基于概念相似性和复杂性
                    # 计算概念之间的语义相似性
                    concept_similarity = 0.0
                    if 'semantic_embedding' in concept1 and 'semantic_embedding' in concept2:
                        # 如果有语义嵌入，计算余弦相似性
                        emb1 = concept1['semantic_embedding']
                        emb2 = concept2['semantic_embedding']
                        if len(emb1) == len(emb2) and len(emb1) > 0:
                            # 简化的相似性计算（点积）
                            dot_product = sum(a * b for a, b in zip(emb1, emb2))
                            norm1 = math.sqrt(sum(a * a for a in emb1))
                            norm2 = math.sqrt(sum(b * b for b in emb2))
                            if norm1 > 0 and norm2 > 0:
                                concept_similarity = dot_product / (norm1 * norm2)
                    
                    # 确定性抽象检查：如果概念足够相似，则进行抽象
                    # 更高的复杂性允许抽象不太相似的概念
                    similarity_threshold = 0.5 - (complexity * 0.2)  # 复杂度越高，阈值越低
                    abstraction_success = concept_similarity >= similarity_threshold
                    
                    if abstraction_success:
                        abstract_concept_id = f"abstract_concept_{hashlib.md5((concept1['id'] + concept2['id']).encode()).hexdigest()[:8]}"
                        abstract_concept = {
                            'id': abstract_concept_id,
                            'name': f"abstract_{concept1['name']}_{concept2['name']}",
                            'type': 'abstract_concept',
                            'component_concepts': [concept1['id'], concept2['id']],
                            'abstraction_level': complexity,
                            'created': datetime.now().isoformat(),
                            'semantic_embedding': [(a + b) / 2 for a, b in zip(concept1['semantic_embedding'], concept2['semantic_embedding'])]
                        }
                        
                        self.knowledge_architecture['semantic_memory']['concepts'][abstract_concept_id] = abstract_concept
                        concepts_abstracted += 1
        
        knowledge_gain = concepts_abstracted * (0.3 + complexity * 0.4)
        
        return {
            'success': True,
            'concepts_abstracted': concepts_abstracted,
            'knowledge_gain': knowledge_gain,
            'complexity': complexity
        }
    
    def _execute_causal_analysis_task(self, complexity: float, intensity: float) -> Dict[str, Any]:
        """执行因果分析任务"""
        # 分析经验回放中的因果关系
        recent_experiences = list(self.experience_replay)[-min(10, len(self.experience_replay)):]
        causal_relations_discovered = 0
        
        for experience in recent_experiences:
            if 'interaction' in experience and 'learning_results' in experience:
                interaction = experience['interaction']
                learning_results = experience['learning_results']
                
                # 确定性因果分析 - 基于交互质量和复杂性
                # 计算因果发现概率：基于复杂性、交互深度和学习结果质量
                interaction_quality = len(str(interaction)) / 1000.0  # 简化的质量指标
                learning_result_quality = len(str(learning_results)) / 500.0
                base_discovery_probability = 0.2 + complexity * 0.3
                quality_factor = min(1.0, (interaction_quality + learning_result_quality) / 2.0)
                
                # 确定性检查：如果质量因子足够高，则发现因果关系
                if quality_factor >= (0.5 - complexity * 0.2):
                    # 关系强度基于复杂性和质量
                    base_strength = 0.5
                    strength_increment = complexity * 0.3
                    quality_adjustment = quality_factor * 0.1
                    relation_strength = base_strength + strength_increment + quality_adjustment
                    
                    causal_relation = {
                        'cause': f"interaction_{hashlib.md5(str(interaction).encode()).hexdigest()[:6]}",
                        'effect': f"learning_{hashlib.md5(str(learning_results).encode()).hexdigest()[:6]}",
                        'strength': min(0.9, max(0.5, relation_strength)),  # 保持在0.5-0.9范围内
                        'evidence_count': 1,
                        'discovered': datetime.now().isoformat(),
                        'analysis_complexity': complexity,
                        'quality_factor': quality_factor,
                        'discovery_method': 'deterministic_quality_based'
                    }
                    
                    # 存储因果关系
                    relation_id = f"causal_relation_{hashlib.md5(str(causal_relation).encode()).hexdigest()[:10]}"
                    self.knowledge_architecture['causal_models'][relation_id] = causal_relation
                    causal_relations_discovered += 1
        
        knowledge_gain = causal_relations_discovered * (0.4 + complexity * 0.3)
        
        return {
            'success': True,
            'causal_relations_discovered': causal_relations_discovered,
            'knowledge_gain': knowledge_gain,
            'complexity': complexity
        }
    
    def _execute_knowledge_integration_task(self, complexity: float, intensity: float) -> Dict[str, Any]:
        """执行知识整合任务"""
        # 整合不同记忆系统的知识
        integration_successes = 0
        
        # 尝试整合语义记忆和程序记忆
        concepts = list(self.knowledge_architecture['semantic_memory']['concepts'].values())
        rules = list(self.knowledge_architecture['procedural_memory']['rules'].values())
        
        if concepts and rules:
            # 简化的知识整合
            integration_attempts = min(3, len(concepts), len(rules))
            
            for i in range(integration_attempts):
                # 确定性知识整合 - 基于概念和规则质量
                # 评估概念和规则的整合兼容性
                concept = concepts[i]
                rule = rules[i]
                
                # 简化的质量指标：基于概念和规则的属性
                concept_quality = len(str(concept)) / 1000.0 if isinstance(concept, dict) else 0.5
                rule_quality = len(str(rule)) / 800.0 if isinstance(rule, dict) else 0.5
                compatibility_score = min(1.0, (concept_quality + rule_quality) / 2.0)
                
                # 确定性整合检查：如果兼容性足够高，则进行整合
                required_compatibility = 0.5 - (complexity * 0.1)  # 更复杂的任务要求更低的兼容性
                if compatibility_score >= required_compatibility:
                    # 整合强度基于兼容性和复杂性
                    base_strength = 0.6
                    compatibility_bonus = compatibility_score * 0.2
                    complexity_bonus = complexity * 0.1
                    integration_strength = base_strength + compatibility_bonus + complexity_bonus
                    
                    # 创建知识整合记录
                    integration_id = f"integration_{hashlib.md5((concepts[i]['id'] + rules[i]['id']).encode()).hexdigest()[:8]}"
                    integration_record = {
                        'id': integration_id,
                        'concept_id': concepts[i]['id'],
                        'rule_id': rules[i]['id'],
                        'integration_strength': min(0.9, max(0.6, integration_strength)),  # 保持在0.6-0.9范围内
                        'integrated_at': datetime.now().isoformat(),
                        'complexity': complexity,
                        'compatibility_score': compatibility_score,
                        'concept_quality': concept_quality,
                        'rule_quality': rule_quality,
                        'integration_method': 'deterministic_compatibility_based'
                    }
                    
                    # 存储到元记忆
                    self.knowledge_architecture['meta_memory']['integrations'][integration_id] = integration_record
                    integration_successes += 1
        
        knowledge_gain = integration_successes * (0.25 + complexity * 0.35)
        
        return {
            'success': True,
            'integrations_created': integration_successes,
            'knowledge_gain': knowledge_gain,
            'complexity': complexity
        }
    
    def _execute_skill_refinement_task(self, complexity: float, intensity: float) -> Dict[str, Any]:
        """执行技能精炼任务"""
        # 精炼现有规则和程序
        rules = list(self.knowledge_architecture['procedural_memory']['rules'].values())
        rules_refined = 0
        
        for rule in rules:
            # 确定性技能精炼 - 基于规则质量和复杂性
            # 评估规则质量（基于规则属性和当前置信度）
            rule_quality = 0.0
            if isinstance(rule, dict):
                # 规则质量基于多个因素
                current_confidence = rule.get('confidence', 0.5)
                rule_age = len(str(rule)) / 1000.0 if 'last_updated' in rule else 0.3
                
                # 质量指标：较高的置信度和适中的规则年龄表示更好的质量
                confidence_factor = current_confidence * 0.7
                age_factor = min(1.0, rule_age) * 0.3
                rule_quality = confidence_factor + age_factor
            
            # 确定性精炼检查：如果规则质量足够低（需要改进），则进行精炼
            # 更复杂的任务可以精炼质量较低的规则
            quality_threshold = 0.6 - (complexity * 0.2)  # 复杂度越高，阈值越低
            if rule_quality <= quality_threshold:
                # 置信度增量基于规则当前置信度和复杂性
                # 置信度较低的规则获得更大的增量
                current_confidence = rule.get('confidence', 0.5)
                base_increment = 0.05
                need_factor = (0.8 - current_confidence) * 0.2  # 需要越大，增量越大
                complexity_factor = complexity * 0.05
                confidence_increment = base_increment + need_factor + complexity_factor
                
                # 应用置信度增量
                rule['confidence'] = min(1.0, current_confidence + confidence_increment)
                rule['last_updated'] = datetime.now().isoformat()
                rule['refinement_method'] = 'deterministic_quality_based'
                rule['refinement_quality'] = rule_quality
                rules_refined += 1
        
        knowledge_gain = rules_refined * (0.15 + complexity * 0.25)
        
        return {
            'success': True,
            'rules_refined': rules_refined,
            'knowledge_gain': knowledge_gain,
            'complexity': complexity
        }
    
    def _execute_meta_learning_task(self, complexity: float, intensity: float) -> Dict[str, Any]:
        """执行元学习任务"""
        # 更新元学习模型
        try:
            # 生成元学习训练数据
            meta_features = _deterministic_randn((10,), seed_prefix="randn_default").to(self.device) * complexity
            target_strategy = torch.randint(0, 10, (1,), device=self.device)
            
            # 确保输入特征是10维的
            if meta_features.shape[-1] != 10:
                error_handler.log_warning(f"元特征维度不匹配: {meta_features.shape[-1]}维，需要10维", "SelfLearning")
                # 调整特征维度
                if len(meta_features.shape) == 1:
                    # 一维张量，直接补零或截断
                    if meta_features.shape[0] < 10:
                        meta_features = torch.cat([meta_features, torch.zeros(10 - meta_features.shape[0], device=meta_features.device)])
                    else:
                        meta_features = meta_features[:10]
                else:
                    # 多维张量，只取前10个特征
                    meta_features = meta_features[..., :10]
            
            # 前向传播
            features = self.meta_learner['feature_extractor'](meta_features)
            strategy_logits = self.meta_learner['learning_strategy_predictor'](features)
            
            # 计算损失
            loss = nn.CrossEntropyLoss()(strategy_logits.unsqueeze(0), target_strategy)
            
            # 反向传播
            self.meta_optimizer.zero_grad()
            loss.backward()
            self.meta_optimizer.step()
            
            knowledge_gain = (1.0 - loss.item()) * (0.3 + complexity * 0.4)
            
            return {
                'success': True,
                'meta_loss': loss.item(),
                'knowledge_gain': knowledge_gain,
                'complexity': complexity,
                'meta_learning_improvement': True
            }
            
        except Exception as e:
            logger.error(f"元学习任务失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'knowledge_gain': 0.0
            }
    
    def _execute_general_learning_task(self, complexity: float, intensity: float) -> Dict[str, Any]:
        """执行通用学习任务 - 基于真实学习逻辑"""
        try:
            # 基于复杂度和强度生成真实的学习目标
            learning_goals = [
                'pattern_recognition',
                'concept_formation', 
                'skill_acquisition',
                'knowledge_integration',
                'problem_solving'
            ]
            
            # 根据复杂度选择学习目标
            goal_index = min(int(complexity * len(learning_goals)), len(learning_goals) - 1)
            learning_goal = learning_goals[goal_index]
            
            # 创建真实的学习交互，基于参数而不是硬编码值
            real_interaction = {
                'type': 'autonomous_learning',
                'input': {
                    'learning_goal': learning_goal,
                    'complexity': complexity,
                    'intensity': intensity,
                    'timestamp': time.time()
                },
                'output': {
                    'learning_phase': 'execution',
                    'expected_complexity': complexity
                },
                'context': {
                    'autonomous': True,
                    'task_complexity': complexity,
                    'intensity_level': intensity,
                    'learning_focus': learning_goal
                }
            }
            
            # 使用现有的学习机制处理真实交互
            learning_result = self.learn_from_interaction(real_interaction)
            
            # 基于学习结果和参数计算知识增益
            if learning_result['success']:
                base_gain = learning_result.get('knowledge_gain', 0.1)
                # 根据复杂度和强度调整增益
                adjusted_gain = base_gain * complexity * (0.5 + intensity * 0.5)
                knowledge_gain = min(max(adjusted_gain, 0.01), 1.0)  # 限制在0.01-1.0之间
            else:
                # 学习失败时的最小增益
                knowledge_gain = 0.01 * complexity * intensity
            
            return {
                'success': learning_result['success'],
                'knowledge_gain': knowledge_gain,
                'complexity': complexity,
                'intensity': intensity,
                'learning_goal': learning_goal,
                'method': 'general_autonomous_learning'
            }
            
        except Exception as e:
            logger.error(f"通用学习任务执行失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'knowledge_gain': 0.01 * complexity,  # 失败时的最小增益
                'complexity': complexity,
                'method': 'general_autonomous_learning'
            }
    
    def _generate_synthetic_pattern_data(self, complexity: float, intensity: float) -> List[float]:
        """生成用于模式发现的合成数据"""
        data_points = 10
        base_pattern = [math.sin(i * 0.5) for i in range(data_points)]
        
        # 添加噪声基于复杂度
        noise_level = (1.0 - complexity) * 0.3
        # 确定性噪声生成 - 基于索引、复杂性和基础值
        synthetic_data = []
        for i, value in enumerate(base_pattern):
            # 确定性噪声函数：基于索引和复杂性的伪随机但确定性的值
            # 使用正弦函数创建确定性但变化的噪声
            noise_phase = (i * 0.7) + (complexity * 3.14159)
            deterministic_noise = math.sin(noise_phase) * noise_level
            synthetic_data.append(value + deterministic_noise)
        
        # 标准化到0-1范围
        min_val = min(synthetic_data)
        max_val = max(synthetic_data)
        if max_val > min_val:
            synthetic_data = [(x - min_val) / (max_val - min_val) for x in synthetic_data]
        
        return synthetic_data
    
    def _update_self_monitoring_from_autonomous_learning(self, learning_results: Dict[str, Any], intensity: float):
        """基于自主学习结果更新自我监控"""
        total_tasks = len(learning_results)
        successful_tasks = sum(1 for result in learning_results.values() if result.get('success', False))
        
        if total_tasks > 0:
            success_rate = successful_tasks / total_tasks
            
            # 更新学习效率
            efficiency_improvement = success_rate * 0.1 * intensity
            self.self_monitoring['learning_efficiency'] = min(1.0, 
                self.self_monitoring['learning_efficiency'] + efficiency_improvement)
            
            # 更新问题解决能力
            problem_solving_improvement = success_rate * 0.08 * intensity
            self.self_monitoring['problem_solving_ability'] = min(1.0,
                self.self_monitoring['problem_solving_ability'] + problem_solving_improvement)
            
            # 更新适应性评分
            adaptability_improvement = success_rate * 0.06 * intensity
            self.self_monitoring['adaptability_score'] = min(1.0,
                self.self_monitoring['adaptability_score'] + adaptability_improvement)
    
    def run_optimization(self, optimization_intensity: float) -> Dict[str, Any]:
        """
        运行系统优化
        参数:
            optimization_intensity: 优化强度 (0.0-1.0)
        返回: 包含优化结果的字典
        """
        # 使用错误恢复机制运行优化
        def run_optimization_logic():
            logger.info(f"开始系统优化，强度: {optimization_intensity}")
            
            adjusted_intensity = max(0.1, min(optimization_intensity, 1.0))
            
            # 执行优化任务，每个任务都使用错误恢复
            memory_opt_result = self._execute_with_recovery(
                operation_name="内存优化",
                operation_func=lambda: self._optimize_memory(adjusted_intensity),
                max_retries=1,
                fallback_value={'improvement': 0.0, 'success': False}
            )
            
            knowledge_opt_result = self._execute_with_recovery(
                operation_name="知识整合优化",
                operation_func=lambda: self._consolidate_knowledge_optimization(adjusted_intensity),
                max_retries=1,
                fallback_value={'improvement': 0.0, 'success': False}
            )
            
            learning_param_result = self._execute_with_recovery(
                operation_name="学习参数调优",
                operation_func=lambda: self._tune_learning_parameters(adjusted_intensity),
                max_retries=1,
                fallback_value={'improvement': 0.0, 'success': False}
            )
            
            optimization_results = {
                'memory_optimization': memory_opt_result,
                'knowledge_consolidation': knowledge_opt_result,
                'learning_parameter_tuning': learning_param_result
            }
            
            # 计算总体优化效果
            total_improvement = sum(
                result.get('improvement', 0.0) 
                for result in optimization_results.values() 
                if isinstance(result, dict)
            ) / len(optimization_results)
            
            logger.info(f"系统优化完成，总体改进: {total_improvement:.2f}")
            
            return {
                'success': True,
                'optimization_intensity': adjusted_intensity,
                'total_improvement': total_improvement,
                'optimization_results': optimization_results
            }
        
        # 恢复函数：尝试简化优化
        def recovery_func():
            try:
                error_handler.log_warning("优化失败，尝试简化优化", "SelfLearning")
                # 只执行最基本的内存清理
                simplified_optimization = {
                    'memory_optimization': self._execute_with_recovery(
                        operation_name="简化内存优化",
                        operation_func=self._optimize_memory,
                        max_retries=1,
                        fallback_value={'improvement': 0.0, 'success': False}
                    ),
                    'knowledge_consolidation': {'improvement': 0.0, 'success': False},
                    'learning_parameter_tuning': {'improvement': 0.0, 'success': False}
                }
                
                return {
                    'success': True,
                    'optimization_intensity': max(0.1, min(optimization_intensity, 0.3)),  # 降低强度
                    'total_improvement': 0.1,  # 最小改进
                    'optimization_results': simplified_optimization,
                    'note': 'simplified_optimization_due_to_error'
                }
            except Exception as recovery_error:
                error_handler.handle_error(recovery_error, "SelfLearning", "优化恢复失败")
                return {'success': False, 'error': 'Optimization recovery failed'}
        
        # 使用错误恢复机制执行
        return self._execute_with_recovery(
            operation_name="运行系统优化",
            operation_func=run_optimization_logic,
            max_retries=1,
            recovery_func=recovery_func,
            fallback_value={'success': False, 'error': 'Optimization failed'}
        )
    
    def _estimate_memory_usage(self) -> int:
        """估计当前内存使用（基于数据结构大小）"""
        total = 0
        # 经验回放
        total += len(self.experience_replay)
        # 工作记忆
        total += len(self.working_memory)
        # 长期记忆
        total += len(self.long_term_memory)
        # 知识架构
        for key, value in self.knowledge_architecture.items():
            if isinstance(value, dict):
                total += len(value)
            elif isinstance(value, deque):
                total += len(value)
            elif isinstance(value, list):
                total += len(value)
        # 性能历史
        for model_id, history in self.performance_history.items():
            total += len(history)
        # 改进建议
        total += len(self.improvement_suggestions)
        # 优先级队列
        total += len(self.priority_queue)
        # 已实施的建议
        total += len(self.implemented_suggestions)
        # 学习日志
        total += len(self.learning_logs)
        # 模型状态跟踪
        total += len(self.model_status_tracking)
        return total
    
    def _optimize_memory(self, intensity: float) -> Dict[str, Any]:
        """优化内存使用 - 增强版，清理多个数据结构"""
        try:
            original_total_memory = self._estimate_memory_usage()
            
            # 1. 清理经验回放
            original_experience_size = len(self.experience_replay)
            if original_experience_size > 1000:
                # 保留最近的1000条经验
                self.experience_replay = deque(
                    list(self.experience_replay)[-1000:], 
                    maxlen=1000
                )
            
            # 2. 清理工作记忆
            self.working_memory.clear()
            
            # 3. 清理长期记忆列表，保留最近500条
            if len(self.long_term_memory) > 500:
                self.long_term_memory = self.long_term_memory[-500:]
            
            # 4. 清理知识架构中的情景记忆（已有限制，但确保不超过）
            # episodic_memory 是 deque(maxlen=10000)，所以自动限制
            
            # 5. 清理性能历史，每个模型保留最近100条记录
            for model_id, history in self.performance_history.items():
                if len(history) > 100:
                    self.performance_history[model_id] = history[-100:]
            
            # 6. 清理改进建议列表
            if len(self.improvement_suggestions) > 50:
                self.improvement_suggestions = self.improvement_suggestions[-50:]
            
            # 7. 清理优先级队列
            if len(self.priority_queue) > 20:
                self.priority_queue = self.priority_queue[-20:]
            
            # 8. 清理已实施的建议
            if len(self.implemented_suggestions) > 100:
                self.implemented_suggestions = self.implemented_suggestions[-100:]
            
            # 9. 清理学习日志（已由 max_logs 控制，但确保）
            if len(self.learning_logs) > self.max_logs:
                self.learning_logs = self.learning_logs[-self.max_logs:]
            
            # 10. 清理模型状态跟踪中不存在的模型引用
            valid_model_ids = set(self.model_references.keys())
            for model_id in list(self.model_status_tracking.keys()):
                if model_id not in valid_model_ids:
                    del self.model_status_tracking[model_id]
            
            # 计算内存优化效果
            current_total_memory = self._estimate_memory_usage()
            memory_reduction = original_total_memory - current_total_memory
            improvement = min(1.0, memory_reduction / max(1, original_total_memory)) * intensity
            
            return {
                'success': True,
                'original_memory_estimate': original_total_memory,
                'current_memory_estimate': current_total_memory,
                'memory_reduction': memory_reduction,
                'improvement': improvement,
                'details': {
                    'experience_replay': original_experience_size - len(self.experience_replay),
                    'long_term_memory_reduced': max(0, len(self.long_term_memory) - 500),
                    'performance_history_cleaned': sum(max(0, len(h) - 100) for h in self.performance_history.values()),
                    'improvement_suggestions_reduced': max(0, len(self.improvement_suggestions) - 50),
                    'priority_queue_reduced': max(0, len(self.priority_queue) - 20),
                    'implemented_suggestions_reduced': max(0, len(self.implemented_suggestions) - 100),
                    'learning_logs_reduced': max(0, len(self.learning_logs) - self.max_logs),
                    'model_status_tracking_cleaned': len([mid for mid in self.model_status_tracking if mid not in valid_model_ids])
                }
            }
            
        except Exception as e:
            logger.error(f"内存优化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}
    
    def _consolidate_knowledge_optimization(self, intensity: float) -> Dict[str, Any]:
        """优化知识巩固"""
        try:
            # 合并相似概念
            concepts = self.knowledge_architecture['semantic_memory']['concepts']
            original_concept_count = len(concepts)
            
            # 简化的概念合并逻辑
            concepts_to_remove = []
            concept_names = {}
            
            for concept_id, concept in concepts.items():
                concept_name = concept['name']
                if concept_name in concept_names:
                    # 合并到现有概念
                    existing_concept = concepts[concept_names[concept_name]]
                    existing_concept['frequency'] += concept['frequency']
                    existing_concept['value_examples'].extend(concept['value_examples'])
                    concepts_to_remove.append(concept_id)
                else:
                    concept_names[concept_name] = concept_id
            
            # 移除重复概念
            for concept_id in concepts_to_remove:
                del concepts[concept_id]
            
            improvement = min(1.0, len(concepts_to_remove) / max(1, original_concept_count)) * intensity
            
            return {
                'success': True,
                'original_concept_count': original_concept_count,
                'current_concept_count': len(concepts),
                'concepts_merged': len(concepts_to_remove),
                'improvement': improvement
            }
            
        except Exception as e:
            logger.error(f"知识巩固优化失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _tune_learning_parameters(self, intensity: float) -> Dict[str, Any]:
        """调整学习参数"""
        try:
            original_parameters = self.learning_parameters.copy()
            
            # 基于性能调整参数
            success_rate = self.learning_stats['successful_learnings'] / max(1, self.learning_stats['total_learning_sessions'])
            
            if success_rate < 0.7:
                # 提高学习率
                self.learning_parameters['base_learning_rate'] *= (1.0 + intensity * 0.2)
                self.learning_parameters['exploration_rate'] *= (1.0 + intensity * 0.1)
            else:
                # 降低学习率，提高利用率
                self.learning_parameters['base_learning_rate'] *= (1.0 - intensity * 0.1)
                self.learning_parameters['exploitation_rate'] *= (1.0 + intensity * 0.1)
            
            improvement = abs(success_rate - 0.7) * intensity
            
            return {
                'success': True,
                'original_parameters': original_parameters,
                'adjusted_parameters': self.learning_parameters.copy(),
                'improvement': improvement,
                'success_rate': success_rate
            }
            
        except Exception as e:
            logger.error(f"学习参数调整失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def update_performance(self, model_id: str, performance_metrics: Dict[str, Any]) -> bool:
        """
        更新模型性能指标
        参数:
            model_id: 模型ID
            performance_metrics: 性能指标字典
        返回: 成功为True
        """
        # 使用错误恢复机制更新性能
        def update_performance_logic():
            # 更新模型状态跟踪
            current_time = datetime.now()
            
            self.model_status_tracking[model_id].update({
                'last_trained': current_time,
                'performance_score': performance_metrics.get('performance_score', 0.0),
                'improvement_rate': performance_metrics.get('improvement_rate', 0.0),
                'training_priority': performance_metrics.get('training_priority', 0)
            })
            
            # 记录性能历史
            self.performance_history[model_id].append({
                'timestamp': current_time,
                'metrics': performance_metrics,
                'model_id': model_id
            })
            
            # 限制历史记录数量
            if len(self.performance_history[model_id]) > 100:
                self.performance_history[model_id] = self.performance_history[model_id][-100:]
            
            logger.info(f"更新模型性能: {model_id}, 分数: {performance_metrics.get('performance_score', 0.0):.2f}")
            return True
        
        # 恢复函数：尝试简化更新或仅记录错误
        def recovery_func():
            try:
                error_handler.log_warning(f"性能更新失败，尝试简化更新: {model_id}", "SelfLearning")
                # 尝试只记录基本信息，不更新完整的状态
                current_time = datetime.now()
                simplified_metrics = {
                    'timestamp': current_time,
                    'model_id': model_id,
                    'note': 'simplified_update_due_to_error'
                }
                self.performance_history[model_id].append(simplified_metrics)
                
                # 限制历史记录数量
                if len(self.performance_history[model_id]) > 100:
                    self.performance_history[model_id] = self.performance_history[model_id][-100:]
                
                return True
            except Exception as recovery_error:
                error_handler.handle_error(recovery_error, "SelfLearning", "性能更新恢复失败")
                return False
        
        # 使用错误恢复机制执行
        return self._execute_with_recovery(
            operation_name=f"更新模型性能: {model_id}",
            operation_func=update_performance_logic,
            max_retries=1,
            recovery_func=recovery_func,
            fallback_value=False
        )
    
    def get_detailed_knowledge_report(self) -> Dict[str, Any]:
        """
        获取详细知识报告
        返回: 包含知识详细信息的字典
        """
        return {
            'knowledge_architecture': self.knowledge_architecture,
            'learning_evolution': {
                'knowledge_growth_rate': self._calculate_knowledge_growth_rate(),
                'learning_trajectory': self._get_learning_trajectory(),
                'competency_development': self._assess_competency_development()
            },
            'cognitive_abilities': {
                'pattern_recognition': self._assess_pattern_recognition(),
                'causal_reasoning': self._assess_causal_reasoning(),
                'abstract_thinking': self._assess_abstract_thinking(),
                'meta_cognition': self._assess_meta_cognition()
            }
        }
    
    def _calculate_knowledge_growth_rate(self) -> float:
        """计算知识增长率"""
        if self.learning_stats['total_learning_sessions'] == 0:
            return 0.0
        
        return self.learning_stats['total_knowledge_gained'] / self.learning_stats['total_learning_sessions']
    
    def _get_learning_trajectory(self) -> List[float]:
        """获取学习轨迹"""
        # 简化的学习轨迹（最近10次学习会话的知识增益）
        trajectory = []
        recent_experiences = list(self.experience_replay)[-10:]
        
        for exp in recent_experiences:
            trajectory.append(exp.get('knowledge_gain', 0.0))
        
        return trajectory
    
    def _assess_competency_development(self) -> Dict[str, float]:
        """评估能力发展"""
        return {
            'basic_learning': min(1.0, self.learning_stats['successful_learnings'] / max(1, self.learning_stats['total_learning_sessions']) * 1.2),
            'meta_learning': min(1.0, self.learning_stats['meta_learning_improvements'] / 50.0),
            'transfer_learning': min(1.0, self.learning_stats['transfer_learning_successes'] / 30.0),
            'causal_reasoning': min(1.0, self.learning_stats['causal_discoveries'] / 20.0),
            'self_reflection': min(1.0, self.learning_stats['self_reflection_insights'] / 40.0)
        }
    
    def _assess_pattern_recognition(self) -> float:
        """评估模式识别能力"""
        patterns = self.knowledge_architecture['semantic_memory']['patterns']
        if not patterns:
            return 0.3
        
        pattern_diversity = len(set(pattern['type'] for pattern in patterns.values())) / max(1, len(patterns))
        pattern_confidence = np.mean([pattern['pattern_score'] for pattern in patterns.values()])
        
        return min(1.0, (pattern_diversity + pattern_confidence) / 2)
    
    def _assess_causal_reasoning(self) -> float:
        """评估因果推理能力"""
        causal_models = self.knowledge_architecture['causal_models']
        if not causal_models:
            return 0.2
        
        model_strength = np.mean([model['strength'] for model in causal_models.values()])
        model_diversity = len(set((model['cause'], model['effect']) for model in causal_models.values())) / max(1, len(causal_models))
        
        return min(1.0, (model_strength + model_diversity) / 2)
    
    def _assess_abstract_thinking(self) -> float:
        """评估抽象思维能力"""
        concepts = self.knowledge_architecture['semantic_memory']['concepts']
        if not concepts:
            return 0.25
        
        abstraction_level = len([concept for concept in concepts.values() if len(concept['name'].split('_')) > 2]) / max(1, len(concepts))
        concept_relationships = sum(len(concept['related_concepts']) for concept in concepts.values()) / max(1, len(concepts))
        
        return min(1.0, (abstraction_level + concept_relationships) / 3)
    
    def _assess_meta_cognition(self) -> float:
        """评估元认知能力"""
        insights = self.knowledge_architecture['meta_memory']['insights']
        if not insights:
            return 0.3
        
        insight_depth = np.mean([insight['confidence'] for insight in insights.values()])
        insight_variety = len(set(insight['type'] for insight in insights.values())) / max(1, len(insights))
        
        return min(1.0, (insight_depth + insight_variety) / 2)
    
    def set_learning_goals(self, goals: Dict[str, List[Dict[str, Any]]]) -> bool:
        """
        设置学习目标
        参数:
            goals: 包含短期、中期、长期目标的字典
        返回: 成功为True
        """
        try:
            if 'short_term' in goals:
                self.learning_goals['short_term'] = goals['short_term']
            if 'medium_term' in goals:
                self.learning_goals['medium_term'] = goals['medium_term']
            if 'long_term' in goals:
                self.learning_goals['long_term'] = goals['long_term']
            
            logger.info("学习目标更新成功")
            return True
            
        except Exception as e:
            logger.error(f"设置学习目标失败: {e}")
            return False
    
    def get_learning_goals(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取当前学习目标
        返回: 学习目标字典
        """
        return self.learning_goals.copy()
    
    def enable_learning(self, enable: bool = True) -> None:
        """
        启用或禁用学习
        参数:
            enable: True启用，False禁用
        """
        self.learning_enabled = enable
        logger.info(f"学习功能{'启用' if enable else '禁用'}")
    
    def reset_learning_stats(self) -> None:
        """重置学习统计信息"""
        self.learning_stats = {
            'total_learning_sessions': 0,
            'successful_learnings': 0,
            'failed_learnings': 0,
            'total_knowledge_gained': 0.0,
            'meta_learning_improvements': 0,
            'transfer_learning_successes': 0,
            'causal_discoveries': 0,
            'self_reflection_insights': 0,
            'long_term_goals_achieved': 0
        }
        logger.info("学习统计信息已重置")
    
    def clear_knowledge_base(self, preserve_essential: bool = True) -> None:
        """
        清空知识库
        参数:
            preserve_essential: 是否保留基本知识
        """
        if preserve_essential:
            # 保留基本架构但清空内容
            for key in self.knowledge_architecture:
                if isinstance(self.knowledge_architecture[key], dict):
                    self.knowledge_architecture[key].clear()
                elif isinstance(self.knowledge_architecture[key], deque):
                    self.knowledge_architecture[key].clear()
        else:
            # 完全重置
            self.knowledge_architecture = {
                'semantic_memory': defaultdict(dict),
                'episodic_memory': deque(maxlen=10000),
                'procedural_memory': defaultdict(dict),
                'meta_memory': defaultdict(dict),
                'causal_models': defaultdict(dict),
                'mental_models': defaultdict(dict)
            }
        
        self.experience_replay.clear()
        self.working_memory.clear()
        self.long_term_memory.clear()
        
        logger.info("知识库已清空")

    def provide_learning_insights(self, operation: str, input_data: Dict[str, Any], 
                                  enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """提供学习见解和分析
        
        Args:
            operation: 正在执行的操作
            input_data: 操作的输入数据
            enhanced_result: 来自先前处理阶段的增强结果
            
        Returns:
            包含学习见解的字典
        """
        try:
            # 分析操作的学习潜力
            learning_potential = self._assess_learning_potential(operation, input_data)
            
            # 基于当前知识状态生成见解
            insights = {
                'operation': operation,
                'learning_potential': learning_potential,
                'knowledge_gaps': self._identify_knowledge_gaps(operation, input_data),
                'suggested_learning_strategies': self._suggest_learning_strategies(operation, learning_potential),
                'meta_learning_opportunities': self._find_meta_learning_opportunities(operation),
                'transfer_learning_possibilities': self._find_transfer_learning_possibilities(operation, input_data),
                'confidence_in_insights': min(0.9, self.learning_stats['successful_learnings'] / max(1, self.learning_stats['total_learning_sessions'])),
                'timestamp': datetime.now().isoformat()
            }
            
            # 如果操作是新的，添加特殊标记
            if learning_potential > 0.7:
                insights['high_learning_opportunity'] = True
                insights['recommended_learning_intensity'] = 0.8
            
            return insights
            
        except Exception as e:
            logger.error(f"生成学习见解失败: {e}")
            # 返回基础见解
            return {
                'operation': operation,
                'learning_potential': 0.3,
                'knowledge_gaps': [],
                'suggested_learning_strategies': ['observe', 'imitate'],
                'meta_learning_opportunities': [],
                'transfer_learning_possibilities': [],
                'confidence_in_insights': 0.5,
                'timestamp': datetime.now().isoformat(),
                '_error': str(e)
            }

    def get_operation_insights(self, operation: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """获取操作见解和元认知分析
        
        Args:
            operation: 要分析的操作
            context: 可选上下文信息
            
        Returns:
            包含操作见解的字典
        """
        try:
            # 分析操作在知识库中的表示
            operation_analysis = {
                'operation': operation,
                'familiarity': self._assess_operation_familiarity(operation),
                'complexity_estimate': self._estimate_operation_complexity(operation, context),
                'prerequisite_knowledge': self._identify_prerequisite_knowledge(operation),
                'related_operations': self._find_related_operations(operation),
                'performance_history': self.performance_history.get(operation, []),
                'optimal_learning_approach': self._determine_optimal_learning_approach(operation),
                'meta_cognitive_awareness': self._assess_meta_cognition(),
                'timestamp': datetime.now().isoformat()
            }
            
            # 如果上下文提供，添加更多分析
            if context:
                operation_analysis['contextual_relevance'] = self._assess_contextual_relevance(operation, context)
                operation_analysis['situational_adaptation'] = self._assess_situational_adaptation(operation, context)
            
            return operation_analysis
            
        except Exception as e:
            logger.error(f"获取操作见解失败: {e}")
            return {
                'operation': operation,
                'familiarity': 0.5,
                'complexity_estimate': 0.5,
                'prerequisite_knowledge': [],
                'related_operations': [],
                'performance_history': [],
                'optimal_learning_approach': 'trial_and_error',
                'meta_cognitive_awareness': 0.3,
                'timestamp': datetime.now().isoformat(),
                '_error': str(e)
            }

    def update_from_processing_experience(self, operation: str, data: Dict[str, Any]) -> bool:
        """Update learning system from processing experience
        
        Args:
            operation: The operation that was performed
            data: The processed data containing experience information
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Extract relevant learning information from the processing data
            learning_info = {
                'operation': operation,
                'timestamp': datetime.now().isoformat(),
                'processing_time': data.get('processing_time', 0),
                'success': data.get('success', False),
                'confidence': data.get('confidence', 0.5),
                'complexity_estimate': self._estimate_operation_complexity(operation, data),
                'key_insights': data.get('key_findings', []),
                'errors_encountered': data.get('errors', []),
                'improvements_suggested': data.get('improvement_suggestions', [])
            }
            
            # Store in experience replay buffer
            self.experience_replay.append(learning_info)
            
            # Update performance history
            if operation not in self.performance_history:
                self.performance_history[operation] = []
            
            # Calculate performance score based on success and confidence
            performance_score = (1.0 if learning_info['success'] else 0.3) * learning_info['confidence']
            self.performance_history[operation].append(performance_score)
            
            # Keep only recent performance records
            if len(self.performance_history[operation]) > 100:
                self.performance_history[operation] = self.performance_history[operation][-100:]
            
            # Update procedural memory with successful operations
            if learning_info['success'] and learning_info['confidence'] > 0.7:
                if operation not in self.knowledge_architecture['procedural_memory']:
                    self.knowledge_architecture['procedural_memory'][operation] = {}
                
                # Store successful execution parameters
                self.knowledge_architecture['procedural_memory'][operation].update({
                    'last_successful_execution': learning_info['timestamp'],
                    'average_confidence': (self.knowledge_architecture['procedural_memory'][operation]
                                           .get('average_confidence', 0.5) * 0.7 + learning_info['confidence'] * 0.3),
                    'execution_count': self.knowledge_architecture['procedural_memory'][operation]
                                       .get('execution_count', 0) + 1,
                    'recent_performance': performance_score
                })
            
            # Update semantic memory with operation characteristics
            if 'operation_type' not in self.knowledge_architecture['semantic_memory']:
                self.knowledge_architecture['semantic_memory']['operation_type'] = {}
            
            op_type = operation.split('_')[0] if '_' in operation else operation
            if op_type not in self.knowledge_architecture['semantic_memory']['operation_type']:
                self.knowledge_architecture['semantic_memory']['operation_type'][op_type] = []
            
            if operation not in self.knowledge_architecture['semantic_memory']['operation_type'][op_type]:
                self.knowledge_architecture['semantic_memory']['operation_type'][op_type].append(operation)
            
            # Update meta-memory with learning statistics
            self.learning_stats['total_learning_sessions'] += 1
            if learning_info['success']:
                self.learning_stats['successful_learnings'] += 1
            
            # Update causal models if causal relationships can be inferred
            if 'causal_factors' in data:
                self._update_causal_models(operation, data['causal_factors'])
            
            logger.info(f"Updated learning from {operation} processing experience")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update from processing experience: {e}")
            return False

    def _update_causal_models(self, operation: str, causal_factors: List[Dict[str, Any]]) -> None:
        """Update causal models based on observed factors"""
        if operation not in self.knowledge_architecture['causal_models']:
            self.knowledge_architecture['causal_models'][operation] = []
        
        for factor in causal_factors:
            # Add new causal factor or update existing one
            existing_factor = next(
                (f for f in self.knowledge_architecture['causal_models'][operation] 
                 if f.get('factor_id') == factor.get('factor_id')),
                None
            )
            
            if existing_factor:
                # Update confidence and evidence
                existing_factor['confidence'] = (existing_factor.get('confidence', 0.5) * 0.7 + 
                                               factor.get('confidence', 0.5) * 0.3)
                existing_factor['evidence_count'] = existing_factor.get('evidence_count', 0) + 1
                existing_factor['last_observed'] = datetime.now().isoformat()
            else:
                # Add new causal factor
                factor.update({
                    'evidence_count': 1,
                    'first_observed': datetime.now().isoformat(),
                    'last_observed': datetime.now().isoformat()
                })
                self.knowledge_architecture['causal_models'][operation].append(factor)
        
        # Keep only the most significant causal factors
        if len(self.knowledge_architecture['causal_models'][operation]) > 20:
            self.knowledge_architecture['causal_models'][operation].sort(
                key=lambda x: x.get('confidence', 0) * x.get('evidence_count', 0),
                reverse=True
            )
            self.knowledge_architecture['causal_models'][operation] = (
                self.knowledge_architecture['causal_models'][operation][:20]
            )

    def _assess_learning_potential(self, operation: str, input_data: Dict[str, Any]) -> float:
        """评估操作的学习潜力"""
        # 基于操作熟悉度和输入数据复杂度
        familiarity = self._assess_operation_familiarity(operation)
        data_complexity = len(str(input_data)) / 1000  # 简单启发式
        
        # 不熟悉的操作和复杂数据具有更高的学习潜力
        learning_potential = (1 - familiarity) * 0.6 + min(1.0, data_complexity) * 0.4
        return min(1.0, learning_potential)

    def _identify_knowledge_gaps(self, operation: str, input_data: Dict[str, Any]) -> List[str]:
        """识别知识缺口"""
        gaps = []
        
        # 检查操作是否在程序记忆中
        if operation not in self.knowledge_architecture['procedural_memory']:
            gaps.append(f"procedural_knowledge_for_{operation}")
        
        # 检查输入数据类型是否熟悉
        for key, value in input_data.items():
            value_type = type(value).__name__
            if value_type not in self.knowledge_architecture['semantic_memory'].get('data_types', {}):
                gaps.append(f"knowledge_about_{value_type}_data")
        
        return gaps

    def _suggest_learning_strategies(self, operation: str, learning_potential: float) -> List[str]:
        """建议学习策略"""
        strategies = []
        
        if learning_potential > 0.7:
            strategies.extend(['exploratory_learning', 'experimentation', 'hypothesis_testing'])
        elif learning_potential > 0.4:
            strategies.extend(['imitation', 'guided_practice', 'feedback_analysis'])
        else:
            strategies.extend(['reinforcement', 'pattern_recognition', 'incremental_improvement'])
        
        # 基于操作类型添加特定策略
        if 'video' in operation.lower() or 'image' in operation.lower():
            strategies.append('visual_analysis')
        if 'text' in operation.lower() or 'language' in operation.lower():
            strategies.append('linguistic_analysis')
        
        return strategies

    def _find_meta_learning_opportunities(self, operation: str) -> List[str]:
        """寻找元学习机会"""
        opportunities = []
        
        # 检查是否可以从这个操作中学习如何学习
        if self.learning_stats['total_learning_sessions'] > 10:
            opportunities.append('learning_strategy_optimization')
        
        if operation not in self.knowledge_architecture['meta_memory'].get('learned_operations', {}):
            opportunities.append('metacognitive_monitoring')
        
        return opportunities

    def _find_transfer_learning_possibilities(self, operation: str, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """寻找迁移学习可能性"""
        possibilities = []
        
        # 查找类似操作
        for known_op in self.knowledge_architecture['procedural_memory']:
            if known_op != operation and known_op[:3] == operation[:3]:  # 简单相似性检查
                possibilities.append({
                    'source_operation': known_op,
                    'target_operation': operation,
                    'similarity_estimate': 0.6,
                    'transferable_knowledge': ['procedural_steps', 'error_patterns']
                })
        
        return possibilities

    def _assess_operation_familiarity(self, operation: str) -> float:
        """评估操作熟悉度"""
        # 检查操作在各个记忆中的存在
        familiarity_score = 0.0
        
        if operation in self.knowledge_architecture['procedural_memory']:
            familiarity_score += 0.4
        if operation in self.knowledge_architecture['semantic_memory'].get('operations', {}):
            familiarity_score += 0.3
        if operation in [exp.get('operation') for exp in self.experience_replay]:
            familiarity_score += 0.2
        if operation in self.performance_history:
            familiarity_score += 0.1
        
        return min(1.0, familiarity_score)

    def _estimate_operation_complexity(self, operation: str, context: Dict[str, Any] = None) -> float:
        """估计操作复杂度"""
        # 基于操作名称长度和上下文大小
        complexity = len(operation.split('_')) / 5  # 每个单词增加复杂度
        
        if context:
            complexity += len(str(context)) / 5000  # 上下文大小贡献
        
        return min(1.0, complexity)

    def _identify_prerequisite_knowledge(self, operation: str) -> List[str]:
        """识别先决知识"""
        prerequisites = []
        
        # 基于操作名称推断
        if 'process' in operation.lower():
            prerequisites.append('data_processing_basics')
        if 'analyze' in operation.lower():
            prerequisites.append('analytical_thinking')
        if 'generate' in operation.lower():
            prerequisites.append('creative_thinking')
        
        return prerequisites

    def _find_related_operations(self, operation: str) -> List[str]:
        """查找相关操作"""
        related = []
        
        # 在程序记忆中查找相似操作
        for known_op in self.knowledge_architecture['procedural_memory']:
            if known_op != operation and known_op.split('_')[0] == operation.split('_')[0]:
                related.append(known_op)
        
        return related[:5]  # 限制返回数量

    def _determine_optimal_learning_approach(self, operation: str) -> str:
        """确定最优学习方法"""
        familiarity = self._assess_operation_familiarity(operation)
        
        if familiarity < 0.3:
            return 'exploratory_learning'
        elif familiarity < 0.6:
            return 'guided_practice'
        else:
            return 'refinement_and_optimization'

    def _assess_contextual_relevance(self, operation: str, context: Dict[str, Any]) -> float:
        """评估上下文相关性"""
        # 简单的启发式：上下文越大，相关性可能越高
        context_size = len(str(context))
        return min(1.0, context_size / 1000)

    def _assess_situational_adaptation(self, operation: str, context: Dict[str, Any]) -> float:
        """评估情境适应性"""
        # 基于历史性能
        if operation in self.performance_history:
            performances = self.performance_history[operation]
            if performances:
                return min(1.0, sum(performances) / len(performances))
        
        return 0.5

# 全局实例便于访问
agi_self_learning_system = AGISelfLearningSystem()

def initialize_agi_self_learning() -> bool:
    """初始化全局AGI自我学习系统"""
    return agi_self_learning_system.initialize()

def learn_from_interaction(interaction_data: Dict[str, Any]) -> Dict[str, Any]:
    """使用全局系统从交互中学习"""
    return agi_self_learning_system.learn_from_interaction(interaction_data)

def get_learning_status() -> Dict[str, Any]:
    """从全局系统获取学习状态"""
    return agi_self_learning_system.get_learning_status()

def get_knowledge_summary() -> Dict[str, Any]:
    """获取知识库摘要"""
    return agi_self_learning_system._get_knowledge_summary()

def get_detailed_knowledge_report() -> Dict[str, Any]:
    """获取详细知识报告"""
    return agi_self_learning_system.get_detailed_knowledge_report()

def enable_learning(enable: bool = True) -> None:
    """启用或禁用学习"""
    agi_self_learning_system.enable_learning(enable)

def reset_learning_stats() -> None:
    """重置学习统计信息"""
    agi_self_learning_system.reset_learning_stats()

def clear_knowledge_base(preserve_essential: bool = True) -> None:
    """清空知识库"""
    agi_self_learning_system.clear_knowledge_base(preserve_essential)

def set_learning_goals(goals: Dict[str, List[Dict[str, Any]]]) -> bool:
    """设置学习目标"""
    return agi_self_learning_system.set_learning_goals(goals)

def get_learning_goals() -> Dict[str, List[Dict[str, Any]]]:
    """获取学习目标"""
    return agi_self_learning_system.get_learning_goals()

# 下面是兼容性支持，为了支持旧版本的代码
class SelfLearningModule:
    """
    兼容性类 - 提供与旧版自我学习模块兼容的接口
    这是一个简单的适配器，将调用转发到新的AGI自我学习系统
    """
    
    def __init__(self):
        self.initialized = False
        self.learning_enabled = True
        self.last_learning_time = None
        self.learning_stats = {
            'total_learning_sessions': 0,
            'successful_learnings': 0,
            'failed_learnings': 0,
            'total_knowledge_gained': 0
        }
        self.knowledge_base = {
            'concepts': {},
            'patterns': defaultdict(list),
            'rules': [],
            'relationships': {},
            'q_values': {}
        }
        self.experience_replay = deque(maxlen=1000)
        self.max_experience_size = 1000
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.advanced_system = None
    
    def initialize(self) -> bool:
        """初始化自我学习模块"""
        # 使用新系统的初始化结果
        global agi_self_learning_system
        result = agi_self_learning_system.initialize()
        if result:
            self.initialized = True
            self.last_learning_time = datetime.now()
        return result
    
    def learn_from_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """从单个交互中学习"""
        if not self.initialized:
            error_handler.log_warning("自我学习模块未初始化", "SelfLearning")
            return False
        
        if not self.learning_enabled:
            error_handler.log_warning("学习功能已禁用", "SelfLearning")
            return False
        
        try:
            self.learning_stats['total_learning_sessions'] += 1
            
            # 转发到新系统
            result = agi_self_learning_system.learn_from_interaction(interaction_data)
            success = result.get('success', False)
            
            if success:
                self.learning_stats['successful_learnings'] += 1
                # 更新知识增益统计
                knowledge_gained = result.get('knowledge_gain', 0.0)
                self.learning_stats['total_knowledge_gained'] += knowledge_gained
            else:
                self.learning_stats['failed_learnings'] += 1
            
            self.last_learning_time = datetime.now()
            
            # 保存学习经验
            self._store_experience(interaction_data, success)
            
            return success
            
        except Exception as e:
            logger.error(f"从交互学习中出错: {e}")
            self.learning_stats['failed_learnings'] += 1
            return False
    
    def _store_experience(self, interaction_data: Dict[str, Any], success: bool):
        """存储学习经验"""
        experience = {
            'interaction': interaction_data,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'knowledge_gain': self._calculate_knowledge_gain(interaction_data)
        }
        
        self.experience_replay.append(experience)
        
        # 限制经验回放大小
        if len(self.experience_replay) > self.max_experience_size:
            self.experience_replay = self.experience_replay[-self.max_experience_size:]
    
    def _calculate_knowledge_gain(self, interaction_data: Dict[str, Any]) -> float:
        """计算知识增益"""
        # 简化的知识增益计算
        input_data = interaction_data.get('input', {})
        if isinstance(input_data, dict):
            return min(len(input_data) / 10.0, 1.0)
        return 0.1
    
    def get_learning_status(self) -> Dict[str, Any]:
        """获取当前学习状态和统计信息"""
        # 获取新系统的状态并转换为旧格式
        new_status = agi_self_learning_system.get_learning_status()
        
        return {
            'initialized': self.initialized,
            'learning_enabled': self.learning_enabled,
            'last_learning_time': self.last_learning_time.isoformat() if self.last_learning_time else None,
            'statistics': self.learning_stats.copy(),
            'advanced_system_available': new_status['integration_status']['emotion_system_available'] or \
                                          new_status['integration_status']['value_system_available'],
            'knowledge_base_size': new_status['knowledge_architecture']['semantic_memory']['total_concepts'] + \
                                   new_status['knowledge_architecture']['procedural_memory']['total_rules'],
            'experience_replay_size': new_status['memory_usage']['experience_replay_size']
        }
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """获取知识库摘要"""
        # 获取新系统的知识摘要并转换为旧格式
        new_summary = agi_self_learning_system._get_knowledge_summary()
        
        # 简化实现，返回近似的结果
        return {
            'total_concepts': new_summary['semantic_memory']['total_concepts'],
            'total_patterns': new_summary['semantic_memory']['total_patterns'],
            'total_rules': new_summary['procedural_memory']['total_rules'],
            'total_relationships': new_summary['causal_models']['total_models'],
            'most_common_concepts': []  # 为了兼容性返回空列表
        }
    
    def enable_learning(self, enable: bool = True) -> None:
        """启用或禁用学习"""
        self.learning_enabled = enable
        agi_self_learning_system.enable_learning(enable)
        logger.info(f"学习功能{'启用' if enable else '禁用'}")
    
    def reset_learning_stats(self) -> None:
        """重置学习统计信息"""
        self.learning_stats = {
            'total_learning_sessions': 0,
            'successful_learnings': 0,
            'failed_learnings': 0,
            'total_knowledge_gained': 0
        }
        agi_self_learning_system.reset_learning_stats()
        logger.info("学习统计信息已重置")
    
    def clear_knowledge_base(self) -> None:
        """清空知识库"""
        self.knowledge_base = {
            'concepts': {},
            'patterns': defaultdict(list),
            'rules': [],
            'relationships': {},
            'q_values': {}
        }
        self.experience_replay.clear()
        agi_self_learning_system.clear_knowledge_base()
        logger.info("知识库已清空")

# 全局实例便于访问
self_learning_module = SelfLearningModule()

def initialize_self_learning() -> bool:
    """初始化全局自我学习模块"""
    return self_learning_module.initialize()

def learn_from_interaction(interaction_data: Dict[str, Any]) -> bool:
    """使用全局模块从交互中学习"""
    return self_learning_module.learn_from_interaction(interaction_data)

def get_learning_status() -> Dict[str, Any]:
    """从全局模块获取学习状态"""
    return self_learning_module.get_learning_status()

def get_knowledge_summary() -> Dict[str, Any]:
    """获取知识库摘要"""
    return self_learning_module.get_knowledge_summary()

def enable_learning(enable: bool = True) -> None:
    """启用或禁用学习"""
    self_learning_module.enable_learning(enable)

def reset_learning_stats() -> None:
    """重置学习统计信息"""
    self_learning_module.reset_learning_stats()

def clear_knowledge_base() -> None:
    """清空知识库"""
    self_learning_module.clear_knowledge_base()
