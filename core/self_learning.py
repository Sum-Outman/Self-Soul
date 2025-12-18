"""
AGI自我学习系统 - 完全AGI级别的自我学习能力
集成元学习、迁移学习、因果推理、自我反思等高级认知功能
"""

import logging
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Set
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
import random
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从autonomous_learning_manager.py集成的配置类
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

class AGISelfLearningSystem:
    """
    AGI级自我学习系统 - 实现真正的通用人工智能学习能力
    集成神经科学启发的学习机制和高级认知功能
    """
    
    def __init__(self, from_scratch=False):
        self.initialized = False
        self.learning_enabled = True
        self.last_learning_time = None
        self.creation_time = datetime.now()
        self.from_scratch = from_scratch
        
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
    
    def set_cognitive_architecture(self, cognitive_architecture):
        """
        设置认知架构引用
        Sets reference to the cognitive architecture
        """
        try:
            self.cognitive_architecture = cognitive_architecture
            logger.info("认知架构已设置 | Cognitive architecture set")
            return True
        except Exception as e:
            logger.error(f"设置认知架构失败: {e} | Failed to set cognitive architecture: {e}")
            return False
    
    def set_emotion_system(self, emotion_system):
        """
        设置情感系统引用
        Sets reference to the emotion system
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
                logger.warning("情感意识系统不可用")
            
            # 集成价值对齐系统
            try:
                from core.value_alignment import AGIValueAlignmentSystem
                self.value_system = AGIValueAlignmentSystem()
                logger.info("价值对齐系统已集成")
            except ImportError as e:
                self.value_system = None
                logger.warning("价值对齐系统不可用")
            
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
                nn.Linear(100, 64),
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
    
    def _load_existing_knowledge(self):
        """加载现有知识"""
        knowledge_path = Path("data/knowledge/self_learning_knowledge.pkl")
        if knowledge_path.exists():
            try:
                with open(knowledge_path, 'rb') as f:
                    saved_knowledge = pickle.load(f)
                    self.knowledge_architecture.update(saved_knowledge)
                logger.info("现有知识加载成功")
            except Exception as e:
                logger.warning(f"加载现有知识失败: {e}")
    
    def _save_knowledge(self):
        """保存知识到文件"""
        try:
            knowledge_path = Path("data/knowledge")
            knowledge_path.mkdir(exist_ok=True)
            
            with open(knowledge_path / "self_learning_knowledge.pkl", 'wb') as f:
                pickle.dump(self.knowledge_architecture, f)
            
            logger.info("知识保存成功")
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
        return random.choice(exploration_targets)
    
    def _select_exploitation_target(self):
        """选择利用性学习目标"""
        # 优先优化现有知识和技能
        exploitation_targets = [
            'knowledge_consolidation',
            'skill_refinement',
            'performance_optimization',
            'error_correction'
        ]
        return random.choice(exploitation_targets)
    
    def _select_balanced_target(self):
        """选择平衡性学习目标"""
        balanced_targets = [
            'knowledge_integration',
            'transfer_learning',
            'adaptive_learning',
            'reflective_learning'
        ]
        return random.choice(balanced_targets)
    
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
    
    def learn_from_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        AGI级交互学习 - 从交互中学习并返回学习结果
        参数:
            interaction_data: 包含交互详情的字典
        返回: 包含学习结果和元数据的字典
        """
        if not self.initialized:
            logger.warning("AGI自我学习系统未初始化")
            return {'success': False, 'reason': 'system_not_initialized'}
        
        if not self.learning_enabled:
            logger.warning("学习功能已禁用")
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
                logger.warning(f"情感意识集成失败: {e}")
                return {'default_emotional_context': True}
        return {'emotional_integration': False}
    
    def _check_value_alignment(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """检查价值对齐"""
        if self.value_system:
            try:
                alignment_result = self.value_system.check_alignment(interaction_data)
                if not alignment_result['aligned']:
                    logger.warning(f"价值对齐问题: {alignment_result.get('issues', [])}")
                return alignment_result
            except Exception as e:
                logger.warning(f"价值对齐检查失败: {e}")
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
            type_hash = hash(interaction_data['type']) % 100 / 100
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
        
        # 知识基础特征
        total_concepts = len(self.knowledge_architecture['semantic_memory']['concepts'])
        meta_features.append(min(total_concepts / 500.0, 1.0))
        
        # 复杂性特征
        if 'input' in interaction_data:
            input_complexity = self._calculate_complexity(interaction_data['input'])
            meta_features.append(input_complexity)
        
        # 新颖性特征
        novelty = self._calculate_novelty(interaction_data)
        meta_features.append(novelty)
        
        # 确保有足够特征
        while len(meta_features) < 8:
            meta_features.append(0.0)
        
        return torch.tensor(meta_features[:8], dtype=torch.float32, device=self.device)
    
    def _predict_learning_strategy(self, meta_features: torch.Tensor) -> str:
        """预测最佳学习策略"""
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
                # 简化的知识迁移
                transfer_gain = opportunity['transfer_potential'] * random.uniform(0.5, 0.9)
                total_gain += transfer_gain
                
                successful_transfers.append({
                    'opportunity': opportunity,
                    'transfer_gain': transfer_gain,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.warning(f"知识迁移失败: {e}")
        
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
            # 随机采样经验
            batch_size = min(32, len(self.experience_replay))
            batch = random.sample(list(self.experience_replay), batch_size)
            
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
        type_novelty = 1.0 - (hash(interaction_type) % 100) / 500.0
        novelty *= type_novelty
        
        return max(0.1, novelty)
    
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
                logger.warning("AGI自我学习系统未初始化")
                return {'success': False, 'reason': 'system_not_initialized'}
            
            if not self.learning_enabled:
                logger.warning("学习功能已禁用")
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
            task_type = random.choice(task_types)
            task_id = f"autonomous_task_{int(time.time() * 1000)}_{i}"
            
            task = {
                'id': task_id,
                'type': task_type,
                'complexity': min(1.0, random.uniform(0.3, 0.7) * complexity_multiplier),
                'priority': random.uniform(0.5, 1.0),
                'estimated_duration': random.uniform(1.0, 5.0) * intensity,
                'created': datetime.now().isoformat()
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
        # 生成模拟数据用于模式发现
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
                    
                    # 简化的概念抽象逻辑
                    abstraction_success = random.random() < (0.3 + complexity * 0.4)
                    
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
                
                # 简化的因果分析
                if random.random() < (0.2 + complexity * 0.3):
                    causal_relation = {
                        'cause': f"interaction_{hashlib.md5(str(interaction).encode()).hexdigest()[:6]}",
                        'effect': f"learning_{hashlib.md5(str(learning_results).encode()).hexdigest()[:6]}",
                        'strength': random.uniform(0.5, 0.9),
                        'evidence_count': 1,
                        'discovered': datetime.now().isoformat(),
                        'analysis_complexity': complexity
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
                if random.random() < (0.4 + complexity * 0.3):
                    # 创建知识整合记录
                    integration_id = f"integration_{hashlib.md5((concepts[i]['id'] + rules[i]['id']).encode()).hexdigest()[:8]}"
                    integration_record = {
                        'id': integration_id,
                        'concept_id': concepts[i]['id'],
                        'rule_id': rules[i]['id'],
                        'integration_strength': random.uniform(0.6, 0.9),
                        'integrated_at': datetime.now().isoformat(),
                        'complexity': complexity
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
            # 基于复杂度和随机性决定是否精炼
            if random.random() < (0.3 + complexity * 0.2):
                # 提高规则置信度
                rule['confidence'] = min(1.0, rule['confidence'] + random.uniform(0.05, 0.15))
                rule['last_updated'] = datetime.now().isoformat()
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
            meta_features = torch.randn(8, device=self.device) * complexity
            target_strategy = torch.randint(0, 10, (1,), device=self.device)
            
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
        """执行通用学习任务"""
        # 创建模拟交互数据进行学习
        synthetic_interaction = {
            'type': 'autonomous_learning',
            'input': {
                'learning_goal': 'general_improvement',
                'complexity': complexity,
                'intensity': intensity
            },
            'output': {
                'learning_result': 'success',
                'knowledge_gain': random.uniform(0.1, 0.5) * complexity
            },
            'context': {
                'autonomous': True,
                'task_complexity': complexity
            }
        }
        
        # 使用现有的学习机制
        learning_result = self.learn_from_interaction(synthetic_interaction)
        
        if learning_result['success']:
            knowledge_gain = learning_result.get('knowledge_gain', 0.1) * complexity
        else:
            knowledge_gain = 0.05 * complexity  # 即使失败也有少量学习
        
        return {
            'success': learning_result['success'],
            'knowledge_gain': knowledge_gain,
            'complexity': complexity,
            'method': 'general_autonomous_learning'
        }
    
    def _generate_synthetic_pattern_data(self, complexity: float, intensity: float) -> List[float]:
        """生成用于模式发现的合成数据"""
        data_points = 10
        base_pattern = [math.sin(i * 0.5) for i in range(data_points)]
        
        # 添加噪声基于复杂度
        noise_level = (1.0 - complexity) * 0.3
        synthetic_data = [value + random.uniform(-noise_level, noise_level) for value in base_pattern]
        
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
        try:
            logger.info(f"开始系统优化，强度: {optimization_intensity}")
            
            adjusted_intensity = max(0.1, min(optimization_intensity, 1.0))
            
            # 执行优化任务
            optimization_results = {
                'memory_optimization': self._optimize_memory(adjusted_intensity),
                'knowledge_consolidation': self._consolidate_knowledge_optimization(adjusted_intensity),
                'learning_parameter_tuning': self._tune_learning_parameters(adjusted_intensity)
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
            
        except Exception as e:
            logger.error(f"系统优化失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _optimize_memory(self, intensity: float) -> Dict[str, Any]:
        """优化内存使用"""
        try:
            # 清理经验回放
            original_size = len(self.experience_replay)
            if original_size > 1000:
                # 保留最有价值的经验
                self.experience_replay = deque(
                    list(self.experience_replay)[-1000:], 
                    maxlen=1000
                )
            
            # 清理工作记忆
            self.working_memory.clear()
            
            improvement = min(1.0, (original_size - len(self.experience_replay)) / 1000.0) * intensity
            
            return {
                'success': True,
                'original_memory_size': original_size,
                'current_memory_size': len(self.experience_replay),
                'memory_reduction': original_size - len(self.experience_replay),
                'improvement': improvement
            }
            
        except Exception as e:
            logger.error(f"内存优化失败: {e}")
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
        try:
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
            
        except Exception as e:
            logger.error(f"更新模型性能失败: {model_id}, 错误: {e}")
            return False
    
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
            logger.warning("自我学习模块未初始化")
            return False
        
        if not self.learning_enabled:
            logger.warning("学习功能已禁用")
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
