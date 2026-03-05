"""
智能任务分配器 - Intelligent Task Distributor

增强多模型协同服务的任务分配功能：
1. 基于模型能力的智能任务分配
2. 负载均衡和性能优化
3. 动态优先级调整
4. 冲突检测和解决
5. 协同性能监控

设计目标：
- 根据模型专业领域分配任务
- 考虑模型当前负载和历史性能
- 支持动态优先级调整
- 实现公平调度和负载均衡
- 提供详细的分配决策日志
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import statistics

logger = logging.getLogger(__name__)

class ModelCapability(Enum):
    """模型能力枚举"""
    LANGUAGE_PROCESSING = "language_processing"
    KNOWLEDGE_REASONING = "knowledge_reasoning"
    VISION_ANALYSIS = "vision_analysis"
    AUDIO_PROCESSING = "audio_processing"
    PLANNING_SCHEDULING = "planning_scheduling"
    PROGRAMMING_CODE = "programming_code"
    EMOTION_ANALYSIS = "emotion_analysis"
    CREATIVE_GENERATION = "creative_generation"
    LOGICAL_REASONING = "logical_reasoning"
    DATA_ANALYSIS = "data_analysis"

class TaskComplexity(Enum):
    """任务复杂度"""
    SIMPLE = "simple"        # 简单任务：单一模型可处理
    MODERATE = "moderate"    # 中等任务：可能需要2-3个模型
    COMPLEX = "complex"      # 复杂任务：需要多个模型协同
    VERY_COMPLEX = "very_complex"  # 非常复杂：需要多个模型深度协同

@dataclass
class ModelCapabilityProfile:
    """模型能力配置文件"""
    model_id: str
    capabilities: List[ModelCapability]
    capability_scores: Dict[ModelCapability, float]  # 能力评分（0-1）
    max_concurrent_tasks: int = 5
    current_task_count: int = 0
    historical_performance: Dict[str, float] = field(default_factory=dict)  # 任务类型->平均执行时间
    reliability_score: float = 0.95  # 可靠性评分
    last_update_time: float = field(default_factory=time.time)

@dataclass 
class TaskRequirements:
    """任务需求分析"""
    required_capabilities: List[ModelCapability]
    preferred_capabilities: List[ModelCapability] = field(default_factory=list)
    complexity: TaskComplexity = TaskComplexity.MODERATE
    estimated_processing_time: float = 10.0  # 估计处理时间（秒）
    memory_requirement: float = 0.0  # 内存需求（MB）
    cpu_requirement: float = 0.0  # CPU需求（百分比）
    deadline: Optional[float] = None  # 截止时间（时间戳）

@dataclass
class AllocationDecision:
    """分配决策"""
    task_id: str
    allocated_models: List[str]
    primary_model: str
    secondary_models: List[str] = field(default_factory=list)
    allocation_reason: str = ""
    confidence_score: float = 0.0
    estimated_completion_time: float = 0.0
    potential_conflicts: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

class ModelCapabilityRegistry:
    """模型能力注册表"""
    
    def __init__(self):
        self.capability_profiles: Dict[str, ModelCapabilityProfile] = {}
        self.capability_model_map: Dict[ModelCapability, List[str]] = defaultdict(list)
        self.lock = threading.RLock()
        
        # 性能统计
        self.allocation_history: deque = deque(maxlen=1000)
        self.performance_metrics = defaultdict(lambda: {
            "total_allocations": 0,
            "successful_allocations": 0,
            "avg_execution_time": 0.0,
            "last_allocation_time": 0.0
        })
    
    def register_model(self, model_id: str, capabilities: List[ModelCapability], 
                      capability_scores: Optional[Dict[ModelCapability, float]] = None):
        """注册模型能力"""
        with self.lock:
            if capability_scores is None:
                capability_scores = {cap: 0.8 for cap in capabilities}
            
            profile = ModelCapabilityProfile(
                model_id=model_id,
                capabilities=capabilities,
                capability_scores=capability_scores
            )
            
            self.capability_profiles[model_id] = profile
            
            # 更新能力到模型映射
            for capability in capabilities:
                self.capability_model_map[capability].append(model_id)
            
            logger.info(f"模型注册成功: {model_id}, 能力: {[c.value for c in capabilities]}")
    
    def update_model_load(self, model_id: str, task_count_delta: int):
        """更新模型负载"""
        with self.lock:
            if model_id in self.capability_profiles:
                profile = self.capability_profiles[model_id]
                profile.current_task_count = max(0, profile.current_task_count + task_count_delta)
                profile.last_update_time = time.time()
    
    def get_available_models(self, required_capabilities: List[ModelCapability]) -> List[str]:
        """获取具备所需能力的可用模型（考虑负载）"""
        with self.lock:
            available_models = []
            
            for capability in required_capabilities:
                capable_models = self.capability_model_map.get(capability, [])
                
                for model_id in capable_models:
                    if model_id not in available_models:
                        profile = self.capability_profiles[model_id]
                        # 检查模型是否过载
                        if profile.current_task_count < profile.max_concurrent_tasks:
                            available_models.append(model_id)
            
            return available_models
    
    def get_model_fitness_score(self, model_id: str, task_requirements: TaskRequirements) -> float:
        """计算模型对任务的适应度分数"""
        with self.lock:
            if model_id not in self.capability_profiles:
                return 0.0
            
            profile = self.capability_profiles[model_id]
            
            # 基础分数
            base_score = 0.0
            
            # 1. 能力匹配分数
            capability_score = 0.0
            for capability in task_requirements.required_capabilities:
                if capability in profile.capability_scores:
                    capability_score += profile.capability_scores[capability]
            
            if task_requirements.required_capabilities:
                capability_score /= len(task_requirements.required_capabilities)
            
            # 2. 负载分数（负载越低分数越高）
            load_ratio = profile.current_task_count / max(1, profile.max_concurrent_tasks)
            load_score = 1.0 - load_ratio
            
            # 3. 可靠性分数
            reliability_score = profile.reliability_score
            
            # 4. 历史性能分数（如果存在相关任务类型）
            performance_score = 0.8  # 默认
            if task_requirements.complexity.value in profile.historical_performance:
                exec_time = profile.historical_performance[task_requirements.complexity.value]
                # 执行时间越短分数越高（归一化到0-1）
                performance_score = 1.0 / (1.0 + exec_time / 60.0)  # 假设60秒为基准
            
            # 综合分数（加权平均）
            weights = {
                'capability': 0.4,
                'load': 0.3,
                'reliability': 0.2,
                'performance': 0.1
            }
            
            total_score = (
                capability_score * weights['capability'] +
                load_score * weights['load'] +
                reliability_score * weights['reliability'] +
                performance_score * weights['performance']
            )
            
            return total_score
    
    def record_allocation_result(self, model_id: str, task_id: str, 
                               success: bool, execution_time: float):
        """记录分配结果以改进未来决策"""
        with self.lock:
            self.allocation_history.append({
                'model_id': model_id,
                'task_id': task_id,
                'success': success,
                'execution_time': execution_time,
                'timestamp': time.time()
            })
            
            # 更新性能指标
            metrics = self.performance_metrics[model_id]
            metrics['total_allocations'] += 1
            if success:
                metrics['successful_allocations'] += 1
            
            # 更新平均执行时间
            if execution_time > 0:
                old_avg = metrics['avg_execution_time']
                old_count = metrics['successful_allocations'] - 1
                if old_count > 0:
                    metrics['avg_execution_time'] = (old_avg * old_count + execution_time) / metrics['successful_allocations']
                else:
                    metrics['avg_execution_time'] = execution_time
            
            metrics['last_allocation_time'] = time.time()
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        with self.lock:
            return {
                'total_models': len(self.capability_profiles),
                'total_capabilities': len(self.capability_model_map),
                'avg_model_load': statistics.mean([
                    p.current_task_count for p in self.capability_profiles.values()
                ]) if self.capability_profiles else 0,
                'total_allocations': sum(
                    m['total_allocations'] for m in self.performance_metrics.values()
                ),
                'success_rate': (
                    sum(m['successful_allocations'] for m in self.performance_metrics.values()) /
                    max(1, sum(m['total_allocations'] for m in self.performance_metrics.values()))
                )
            }

class IntelligentTaskDistributor:
    """智能任务分配器"""
    
    def __init__(self, capability_registry: Optional[ModelCapabilityRegistry] = None):
        if capability_registry is None:
            capability_registry = ModelCapabilityRegistry()
        self.capability_registry = capability_registry
        self.task_models_map: Dict[str, AllocationDecision] = {}
        self.conflict_detector = ConflictDetector()
        self.lock = threading.RLock()
        
        # 分配策略
        self.allocation_strategies = {
            'balanced': self._balanced_allocation,
            'performance_optimized': self._performance_optimized_allocation,
            'load_aware': self._load_aware_allocation,
            'reliability_focused': self._reliability_focused_allocation
        }
    
    def analyze_task_requirements(self, task_description: str, 
                                required_models: List[str]) -> TaskRequirements:
        """分析任务需求"""
        # 简化的需求分析（实际实现可以更复杂）
        capabilities = []
        
        # 根据任务描述和所需模型推断能力需求
        if any(keyword in task_description.lower() for keyword in ['language', 'text', 'nlp']):
            capabilities.append(ModelCapability.LANGUAGE_PROCESSING)
        
        if any(keyword in task_description.lower() for keyword in ['image', 'vision', 'visual']):
            capabilities.append(ModelCapability.VISION_ANALYSIS)
        
        if any(keyword in task_description.lower() for keyword in ['audio', 'sound', 'speech']):
            capabilities.append(ModelCapability.AUDIO_PROCESSING)
        
        if any(keyword in task_description.lower() for keyword in ['knowledge', 'reasoning', 'logic']):
            capabilities.append(ModelCapability.KNOWLEDGE_REASONING)
        
        if any(keyword in task_description.lower() for keyword in ['plan', 'schedule', 'strategy']):
            capabilities.append(ModelCapability.PLANNING_SCHEDULING)
        
        # 如果没有检测到能力，使用默认
        if not capabilities:
            capabilities = [ModelCapability.KNOWLEDGE_REASONING]
        
        # 根据所需模型数量确定复杂度
        model_count = len(required_models)
        if model_count <= 1:
            complexity = TaskComplexity.SIMPLE
        elif model_count <= 3:
            complexity = TaskComplexity.MODERATE
        elif model_count <= 5:
            complexity = TaskComplexity.COMPLEX
        else:
            complexity = TaskComplexity.VERY_COMPLEX
        
        return TaskRequirements(
            required_capabilities=capabilities,
            complexity=complexity,
            estimated_processing_time=10.0 * model_count
        )
    
    def allocate_task(self, task_id: str, task_name: str, task_description: str,
                     required_models: List[str], strategy: str = 'balanced') -> AllocationDecision:
        """分配任务给最合适的模型"""
        with self.lock:
            # 分析任务需求
            task_requirements = self.analyze_task_requirements(task_description, required_models)
            
            # 获取可用模型
            available_models = self.capability_registry.get_available_models(
                task_requirements.required_capabilities
            )
            
            if not available_models:
                # 如果没有完全匹配的模型，尝试使用具备部分能力的模型
                logger.warning(f"没有完全匹配的模型，尝试使用部分能力模型")
                for capability in task_requirements.required_capabilities:
                    partial_models = self.capability_registry.capability_model_map.get(capability, [])
                    available_models.extend(partial_models)
                
                available_models = list(set(available_models))
            
            if not available_models:
                # 仍然没有可用模型，返回失败
                return AllocationDecision(
                    task_id=task_id,
                    allocated_models=[],
                    primary_model="",
                    allocation_reason="没有可用的模型满足任务需求",
                    confidence_score=0.0
                )
            
            # 使用指定策略分配
            if strategy not in self.allocation_strategies:
                strategy = 'balanced'
            
            allocation_func = self.allocation_strategies[strategy]
            allocation_result = allocation_func(available_models, task_requirements)
            
            # 检测潜在冲突
            potential_conflicts = self.conflict_detector.detect_conflicts(
                allocation_result.allocated_models, task_requirements
            )
            
            # 创建分配决策
            decision = AllocationDecision(
                task_id=task_id,
                allocated_models=allocation_result.allocated_models,
                primary_model=allocation_result.primary_model,
                secondary_models=allocation_result.secondary_models,
                allocation_reason=allocation_result.reason,
                confidence_score=allocation_result.confidence_score,
                estimated_completion_time=task_requirements.estimated_processing_time,
                potential_conflicts=potential_conflicts
            )
            
            # 记录分配
            self.task_models_map[task_id] = decision
            
            # 更新模型负载
            for model_id in allocation_result.allocated_models:
                self.capability_registry.update_model_load(model_id, 1)
            
            logger.info(f"任务分配完成: {task_id}, 主要模型: {allocation_result.primary_model}, "
                       f"置信度: {allocation_result.confidence_score:.2f}")
            
            return decision
    
    def _balanced_allocation(self, available_models: List[str], 
                           task_requirements: TaskRequirements) -> Any:
        """平衡分配策略（考虑能力和负载）"""
        # 计算每个模型的适应度分数
        model_scores = {}
        for model_id in available_models:
            score = self.capability_registry.get_model_fitness_score(model_id, task_requirements)
            model_scores[model_id] = score
        
        # 选择分数最高的模型作为主要模型
        if not model_scores:
            return self._create_allocation_result([], "", [], "没有可用模型", 0.0)
        
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        primary_model = sorted_models[0][0]
        primary_score = sorted_models[0][1]
        
        # 选择次要模型（如果需要）
        secondary_models = []
        if task_requirements.complexity.value in ['complex', 'very_complex']:
            # 选择前3个模型作为次要模型
            secondary_models = [model_id for model_id, _ in sorted_models[1:4] 
                              if model_id != primary_model]
        
        allocated_models = [primary_model] + secondary_models
        
        return self._create_allocation_result(
            allocated_models, primary_model, secondary_models,
            f"平衡分配策略: 主要模型 {primary_model} (分数: {primary_score:.2f})",
            primary_score
        )
    
    def _performance_optimized_allocation(self, available_models: List[str],
                                        task_requirements: TaskRequirements) -> Any:
        """性能优化分配策略（优先选择历史性能好的模型）"""
        # 简化的性能优化策略
        return self._balanced_allocation(available_models, task_requirements)
    
    def _load_aware_allocation(self, available_models: List[str],
                             task_requirements: TaskRequirements) -> Any:
        """负载感知分配策略（优先选择负载低的模型）"""
        # 简化的负载感知策略
        return self._balanced_allocation(available_models, task_requirements)
    
    def _reliability_focused_allocation(self, available_models: List[str],
                                      task_requirements: TaskRequirements) -> Any:
        """可靠性优先分配策略（优先选择可靠性高的模型）"""
        # 简化的可靠性优先策略
        return self._balanced_allocation(available_models, task_requirements)
    
    def _create_allocation_result(self, allocated_models: List[str], primary_model: str,
                                secondary_models: List[str], reason: str, 
                                confidence_score: float) -> Any:
        """创建分配结果对象"""
        # 使用简单字典作为结果对象（实际可以使用dataclass）
        return type('AllocationResult', (), {
            'allocated_models': allocated_models,
            'primary_model': primary_model,
            'secondary_models': secondary_models,
            'reason': reason,
            'confidence_score': confidence_score
        })()
    
    def complete_task(self, task_id: str, success: bool, execution_time: float):
        """标记任务完成，更新模型负载和性能记录"""
        with self.lock:
            if task_id not in self.task_models_map:
                return
            
            decision = self.task_models_map[task_id]
            
            # 更新模型负载
            for model_id in decision.allocated_models:
                self.capability_registry.update_model_load(model_id, -1)
                
                # 记录性能
                if success:
                    self.capability_registry.record_allocation_result(
                        model_id, task_id, success, execution_time
                    )
            
            # 从映射中移除任务
            del self.task_models_map[task_id]
            
            logger.info(f"任务完成处理: {task_id}, 成功: {success}, 执行时间: {execution_time:.2f}s")

class ConflictDetector:
    """冲突检测器"""
    
    def __init__(self):
        self.conflict_patterns = self._load_conflict_patterns()
    
    def _load_conflict_patterns(self) -> Dict[str, List[str]]:
        """加载冲突模式"""
        return {
            'resource_conflict': ['memory', 'cpu', 'gpu'],
            'capability_overlap': ['language_processing', 'knowledge_reasoning'],
            'data_access_conflict': ['same_data_source', 'exclusive_access'],
            'temporal_conflict': ['simultaneous_execution', 'deadline_overlap']
        }
    
    def detect_conflicts(self, allocated_models: List[str], 
                        task_requirements: TaskRequirements) -> List[str]:
        """检测潜在冲突"""
        conflicts = []
        
        # 检查模型数量是否过多
        if len(allocated_models) > 5:
            conflicts.append("分配模型数量过多可能导致协调困难")
        
        # 检查任务复杂度与模型数量的匹配
        if task_requirements.complexity == TaskComplexity.SIMPLE and len(allocated_models) > 1:
            conflicts.append("简单任务分配了多个模型，可能导致资源浪费")
        
        # 检查能力重叠
        if len(allocated_models) > 1:
            conflicts.append("多个模型可能存在能力重叠，需要协调")
        
        # 检查处理时间估计
        if task_requirements.estimated_processing_time > 300:  # 超过5分钟
            conflicts.append("任务处理时间较长，可能需要监控")
        
        return conflicts
    
    def resolve_conflict(self, conflict_type: str, conflicting_models: List[str]) -> Dict[str, Any]:
        """解决冲突"""
        resolution_strategies = {
            'resource_conflict': '优先级调度和资源配额',
            'capability_overlap': '主次模型分工和结果融合',
            'data_access_conflict': '锁机制和顺序访问',
            'temporal_conflict': '时间片调度和优先级调整'
        }
        
        strategy = resolution_strategies.get(conflict_type, '协商解决')
        
        return {
            'conflict_type': conflict_type,
            'conflicting_models': conflicting_models,
            'resolution_strategy': strategy,
            'suggested_actions': [
                f"为{conflict_type}应用{strategy}策略",
                "监控冲突解决效果",
                "记录冲突解决结果以便学习"
            ]
        }

# 全局实例
_capability_registry = None
_intelligent_distributor = None

def get_capability_registry() -> ModelCapabilityRegistry:
    """获取能力注册表单例"""
    global _capability_registry
    if _capability_registry is None:
        _capability_registry = ModelCapabilityRegistry()
    return _capability_registry

def get_intelligent_distributor() -> IntelligentTaskDistributor:
    """获取智能分配器单例"""
    global _intelligent_distributor
    if _intelligent_distributor is None:
        registry = get_capability_registry()
        _intelligent_distributor = IntelligentTaskDistributor(registry)
    return _intelligent_distributor

def initialize_default_capabilities():
    """初始化默认模型能力"""
    registry = get_capability_registry()
    
    # 定义默认模型能力映射
    default_capabilities = {
        'language': [
            ModelCapability.LANGUAGE_PROCESSING,
            ModelCapability.KNOWLEDGE_REASONING,
            ModelCapability.CREATIVE_GENERATION
        ],
        'vision': [
            ModelCapability.VISION_ANALYSIS,
            ModelCapability.DATA_ANALYSIS
        ],
        'audio': [
            ModelCapability.AUDIO_PROCESSING,
            ModelCapability.DATA_ANALYSIS
        ],
        'knowledge': [
            ModelCapability.KNOWLEDGE_REASONING,
            ModelCapability.LOGICAL_REASONING,
            ModelCapability.DATA_ANALYSIS
        ],
        'planning': [
            ModelCapability.PLANNING_SCHEDULING,
            ModelCapability.LOGICAL_REASONING
        ],
        'programming': [
            ModelCapability.PROGRAMMING_CODE,
            ModelCapability.LOGICAL_REASONING
        ],
        'emotion': [
            ModelCapability.EMOTION_ANALYSIS
        ]
    }
    
    # 注册模型
    for model_id, capabilities in default_capabilities.items():
        registry.register_model(model_id, capabilities)
    
    logger.info(f"默认模型能力初始化完成，注册了 {len(default_capabilities)} 个模型")