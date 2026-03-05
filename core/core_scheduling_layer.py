"""
核心调度层 - Core Scheduling Layer

实现报告要求的「任务解析器 + 能力调度中枢 + 结果验证器」架构
解决顶层调度缺失问题，支持动态能力组合和跨模型协同
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import re

from .capability_interface import (
    ICapabilityProvider, CapabilityType, CapabilityMetadata,
    TaskInput, TaskOutput, ExecutionMetrics, CapabilityLevel
)

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = "critical"      # 关键任务：系统核心功能
    HIGH = "high"              # 高优先级：用户直接请求
    MEDIUM = "medium"          # 中优先级：后台处理
    LOW = "low"                # 低优先级：维护任务
    BACKGROUND = "background"  # 后台任务：非紧急


class TaskComplexity(Enum):
    """任务复杂度"""
    SIMPLE = "simple"          # 简单任务：单一能力可处理
    COMPOUND = "compound"      # 复合任务：多个能力顺序执行
    COMPLEX = "complex"        # 复杂任务：多个能力协同执行
    AGI = "agi"               # AGI任务：需要推理和规划


@dataclass
class TaskAnalysis:
    """任务分析结果"""
    task_id: str
    original_description: str
    required_capabilities: List[CapabilityType]  # 必需能力
    preferred_capabilities: List[CapabilityType]  # 优选能力
    complexity: TaskComplexity
    estimated_duration: float  # 估计持续时间（秒）
    priority: TaskPriority
    domain_tags: List[str] = field(default_factory=list)  # 领域标签
    constraints: Dict[str, Any] = field(default_factory=dict)  # 约束条件


@dataclass
class ExecutionPlan:
    """执行计划"""
    task_id: str
    capability_sequence: List[Tuple[CapabilityType, str]]  # (能力类型, 模型ID)
    dependencies: Dict[int, List[int]] = field(default_factory=dict)  # 依赖关系
    expected_outputs: Dict[int, str] = field(default_factory=dict)  # 预期输出
    fallback_plans: List[List[Tuple[CapabilityType, str]]] = field(default_factory=list)  # 备用计划
    estimated_total_time: float = 0.0


@dataclass
class ValidationResult:
    """验证结果"""
    task_id: str
    success: bool
    quality_score: float  # 质量评分（0-1）
    completeness_score: float  # 完整性评分（0-1）
    correctness_score: float  # 正确性评分（0-1）
    efficiency_score: float  # 效率评分（0-1）
    feedback: Dict[str, Any] = field(default_factory=dict)  # 反馈信息
    improvement_suggestions: List[str] = field(default_factory=list)  # 改进建议


class TaskAnalyzer:
    """任务解析器 - 分析任务需求，识别所需能力"""
    
    def __init__(self):
        self.capability_patterns = self._load_capability_patterns()
        self.domain_keywords = self._load_domain_keywords()
        
    def _load_capability_patterns(self) -> Dict[CapabilityType, List[str]]:
        """加载能力匹配模式"""
        return {
            CapabilityType.LANGUAGE_PROCESSING: [
                r'处理.*文本', r'分析.*语言', r'翻译', r'总结.*内容',
                r'理解.*含义', r'生成.*文章', r'提取.*信息'
            ],
            CapabilityType.KNOWLEDGE_REASONING: [
                r'推理', r'推断', r'基于知识', r'专家系统',
                r'回答问题', r'解释概念', r'逻辑分析'
            ],
            CapabilityType.VISION_ANALYSIS: [
                r'分析.*图像', r'识别.*图片', r'处理.*视觉', 
                r'检测.*对象', r'理解.*场景'
            ],
            CapabilityType.AUDIO_PROCESSING: [
                r'处理.*音频', r'识别.*声音', r'分析.*语音',
                r'转录', r'语音.*识别'
            ],
            CapabilityType.PROGRAMMING_CODE: [
                r'生成.*代码', r'编写.*程序', r'修复.*bug',
                r'优化.*算法', r'分析.*代码'
            ],
            CapabilityType.EMOTION_ANALYSIS: [
                r'分析.*情感', r'理解.*情绪', r'情感.*识别',
                r'情绪.*分析'
            ],
            CapabilityType.DATA_ANALYSIS: [
                r'分析.*数据', r'处理.*数据', r'统计.*分析',
                r'可视化.*数据', r'预测.*趋势'
            ],
            CapabilityType.PLANNING_SCHEDULING: [
                r'规划', r'调度', r'安排.*任务', r'制定.*计划',
                r'优化.*流程'
            ],
            CapabilityType.DECISION_MAKING: [
                r'决策', r'做出.*决定', r'选择.*方案',
                r'评估.*选项'
            ]
        }
    
    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """加载领域关键词"""
        return {
            "engineering": ["工程", "机械", "电气", "结构", "设计"],
            "medical": ["医疗", "健康", "诊断", "治疗", "药物"],
            "financial": ["金融", "财务", "投资", "经济", "市场"],
            "education": ["教育", "学习", "教学", "课程", "培训"],
            "creative": ["创意", "创作", "艺术", "设计", "写作"]
        }
    
    def analyze_task(self, task_id: str, description: str, 
                    priority: TaskPriority = TaskPriority.MEDIUM) -> TaskAnalysis:
        """
        分析任务，识别所需能力
        
        Args:
            task_id: 任务ID
            description: 任务描述
            priority: 任务优先级
            
        Returns:
            任务分析结果
        """
        logger.info(f"开始分析任务: {task_id}")
        
        # 识别能力需求
        required_capabilities = []
        preferred_capabilities = []
        
        for capability_type, patterns in self.capability_patterns.items():
            for pattern in patterns:
                if re.search(pattern, description, re.IGNORECASE):
                    if capability_type in [CapabilityType.LANGUAGE_PROCESSING, 
                                         CapabilityType.KNOWLEDGE_REASONING]:
                        # 语言和知识推理通常是必需的
                        if capability_type not in required_capabilities:
                            required_capabilities.append(capability_type)
                    else:
                        # 其他能力作为优选
                        if capability_type not in preferred_capabilities:
                            preferred_capabilities.append(capability_type)
        
        # 如果没有识别到特定能力，添加默认能力
        if not required_capabilities:
            required_capabilities.append(CapabilityType.LANGUAGE_PROCESSING)
        
        # 识别领域标签
        domain_tags = []
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in description:
                    if domain not in domain_tags:
                        domain_tags.append(domain)
                    break
        
        # 评估复杂度
        complexity = self._assess_complexity(description, required_capabilities)
        
        # 估计持续时间
        estimated_duration = self._estimate_duration(complexity, len(required_capabilities))
        
        analysis = TaskAnalysis(
            task_id=task_id,
            original_description=description,
            required_capabilities=required_capabilities,
            preferred_capabilities=preferred_capabilities,
            complexity=complexity,
            estimated_duration=estimated_duration,
            priority=priority,
            domain_tags=domain_tags
        )
        
        logger.info(f"任务分析完成: {task_id}, 所需能力: {[c.value for c in required_capabilities]}")
        return analysis
    
    def _assess_complexity(self, description: str, 
                          capabilities: List[CapabilityType]) -> TaskComplexity:
        """评估任务复杂度"""
        word_count = len(description.split())
        capability_count = len(capabilities)
        
        if capability_count == 1 and word_count < 20:
            return TaskComplexity.SIMPLE
        elif capability_count <= 3:
            return TaskComplexity.COMPOUND
        elif capability_count <= 5:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.AGI
    
    def _estimate_duration(self, complexity: TaskComplexity, 
                          capability_count: int) -> float:
        """估计任务持续时间"""
        base_times = {
            TaskComplexity.SIMPLE: 5.0,
            TaskComplexity.COMPOUND: 15.0,
            TaskComplexity.COMPLEX: 30.0,
            TaskComplexity.AGI: 60.0
        }
        
        return base_times[complexity] * (1 + 0.2 * capability_count)


class CapabilityRegistry:
    """能力注册表 - 管理所有可用的能力提供者"""
    
    def __init__(self):
        self.providers: Dict[str, ICapabilityProvider] = {}  # 模型ID -> 提供者
        self.capability_index: Dict[CapabilityType, List[str]] = defaultdict(list)  # 能力类型 -> 模型ID列表
        self.performance_stats: Dict[str, Dict[CapabilityType, ExecutionMetrics]] = defaultdict(dict)
        self.lock = threading.RLock()
        
    def register_provider(self, model_id: str, provider: ICapabilityProvider):
        """注册能力提供者"""
        with self.lock:
            self.providers[model_id] = provider
            
            # 更新能力索引
            capabilities = provider.get_capabilities()
            for capability_type in capabilities.keys():
                if model_id not in self.capability_index[capability_type]:
                    self.capability_index[capability_type].append(model_id)
            
            logger.info(f"能力提供者注册成功: {model_id}, 能力: {list(capabilities.keys())}")
    
    def get_providers_for_capability(self, capability_type: CapabilityType) -> List[Tuple[str, ICapabilityProvider]]:
        """获取支持指定能力的所有提供者"""
        with self.lock:
            model_ids = self.capability_index.get(capability_type, [])
            providers = []
            
            for model_id in model_ids:
                if model_id in self.providers:
                    providers.append((model_id, self.providers[model_id]))
            
            return providers
    
    def get_best_provider(self, capability_type: CapabilityType, 
                         input_data: TaskInput) -> Optional[Tuple[str, ICapabilityProvider]]:
        """获取最适合处理指定任务的最佳提供者"""
        providers = self.get_providers_for_capability(capability_type)
        
        if not providers:
            return None
        
        # 评估每个提供者的适合度
        scored_providers = []
        for model_id, provider in providers:
            # 检查是否能处理
            if not provider.can_handle(capability_type, input_data):
                continue
            
            # 计算综合评分
            capability_score = provider.get_capability_score(capability_type)
            
            # 考虑性能历史
            perf_score = self._get_performance_score(model_id, capability_type)
            
            # 综合评分 = 能力评分 * 0.6 + 性能评分 * 0.4
            total_score = capability_score * 0.6 + perf_score * 0.4
            
            scored_providers.append((total_score, model_id, provider))
        
        if not scored_providers:
            return None
        
        # 返回评分最高的提供者
        scored_providers.sort(reverse=True)
        return (scored_providers[0][1], scored_providers[0][2])
    
    def _get_performance_score(self, model_id: str, 
                              capability_type: CapabilityType) -> float:
        """获取性能评分"""
        if model_id not in self.performance_stats:
            return 0.5  # 默认中等评分
        
        stats = self.performance_stats[model_id].get(capability_type)
        if not stats:
            return 0.5
        
        # 基于成功率、质量和效率计算评分
        success_rate = 1.0 if stats.success else 0.0
        quality = stats.result_quality
        efficiency = 1.0 / (1.0 + stats.end_time - stats.start_time)  # 时间越短得分越高
        
        return success_rate * 0.4 + quality * 0.4 + efficiency * 0.2
    
    def record_execution(self, model_id: str, capability_type: CapabilityType,
                        metrics: ExecutionMetrics):
        """记录执行指标"""
        with self.lock:
            self.performance_stats[model_id][capability_type] = metrics


class CapabilityScheduler:
    """能力调度中枢 - 根据任务分析创建和执行计划"""
    
    def __init__(self, capability_registry: CapabilityRegistry):
        self.registry = capability_registry
        self.task_analyzer = TaskAnalyzer()
        self.execution_history: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        
    def schedule_task(self, task_id: str, description: str,
                     input_data: Optional[Dict] = None,
                     priority: TaskPriority = TaskPriority.MEDIUM) -> ExecutionPlan:
        """
        调度任务，创建执行计划
        
        Args:
            task_id: 任务ID
            description: 任务描述
            input_data: 输入数据
            priority: 任务优先级
            
        Returns:
            执行计划
        """
        logger.info(f"开始调度任务: {task_id}")
        
        # 分析任务
        task_analysis = self.task_analyzer.analyze_task(task_id, description, priority)
        
        # 创建任务输入
        task_input = TaskInput(
            data=input_data or description,
            input_format="text" if isinstance(description, str) else "json",
            context={"task_id": task_id, "priority": priority.value}
        )
        
        # 生成执行计划
        execution_plan = self._create_execution_plan(task_analysis, task_input)
        
        # 存储计划
        with self.lock:
            self.execution_history[task_id] = {
                "analysis": task_analysis,
                "plan": execution_plan,
                "status": "scheduled",
                "created_at": time.time()
            }
        
        logger.info(f"任务调度完成: {task_id}, 计划步骤: {len(execution_plan.capability_sequence)}")
        return execution_plan
    
    def _create_execution_plan(self, task_analysis: TaskAnalysis,
                              task_input: TaskInput) -> ExecutionPlan:
        """创建执行计划"""
        capability_sequence = []
        dependencies = {}
        
        # 为每个必需能力选择最佳提供者
        for idx, capability_type in enumerate(task_analysis.required_capabilities):
            best_provider = self.registry.get_best_provider(capability_type, task_input)
            
            if best_provider:
                model_id, provider = best_provider
                capability_sequence.append((capability_type, model_id))
                
                # 设置依赖关系（当前为顺序执行）
                if idx > 0:
                    dependencies[idx] = [idx - 1]
            else:
                # 如果没有找到提供者，使用占位符
                capability_sequence.append((capability_type, "none"))
                logger.warning(f"没有找到能力提供者: {capability_type.value}")
        
        # 创建备用计划（使用不同提供者）
        fallback_plans = self._create_fallback_plans(task_analysis, task_input)
        
        # 估计总时间
        estimated_total_time = task_analysis.estimated_duration
        
        return ExecutionPlan(
            task_id=task_analysis.task_id,
            capability_sequence=capability_sequence,
            dependencies=dependencies,
            fallback_plans=fallback_plans,
            estimated_total_time=estimated_total_time
        )
    
    def _create_fallback_plans(self, task_analysis: TaskAnalysis,
                              task_input: TaskInput) -> List[List[Tuple[CapabilityType, str]]]:
        """创建备用计划"""
        fallback_plans = []
        
        # 为每个能力创建替代提供者列表
        alternative_providers = []
        
        for capability_type in task_analysis.required_capabilities:
            providers = self.registry.get_providers_for_capability(capability_type)
            if len(providers) > 1:
                # 取前3个替代提供者
                alternative_providers.append([
                    (model_id, provider) for model_id, provider in providers[:3]
                ])
            else:
                alternative_providers.append([])
        
        # 生成备用计划（如果有替代提供者）
        if any(alternative_providers):
            # 创建几个备用组合
            for i in range(min(3, len(alternative_providers[0]) if alternative_providers[0] else 1)):
                fallback_plan = []
                for j, capability_type in enumerate(task_analysis.required_capabilities):
                    if alternative_providers[j]:
                        model_id, _ = alternative_providers[j][i % len(alternative_providers[j])]
                        fallback_plan.append((capability_type, model_id))
                    else:
                        fallback_plan.append((capability_type, "none"))
                
                fallback_plans.append(fallback_plan)
        
        return fallback_plans
    
    def execute_plan(self, execution_plan: ExecutionPlan,
                    input_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行计划
        
        Args:
            execution_plan: 执行计划
            input_data: 输入数据
            
        Returns:
            执行结果
        """
        task_id = execution_plan.task_id
        logger.info(f"开始执行任务: {task_id}")
        
        results = {}
        execution_times = {}
        
        # 按顺序执行能力序列
        for idx, (capability_type, model_id) in enumerate(execution_plan.capability_sequence):
            if model_id == "none":
                logger.warning(f"步骤 {idx} 跳过: 没有提供者")
                results[f"step_{idx}"] = {
                    "success": False,
                    "error": "No provider available",
                    "capability": capability_type.value
                }
                continue
            
            # 获取提供者
            provider = self.registry.providers.get(model_id)
            if not provider:
                logger.error(f"提供者不存在: {model_id}")
                results[f"step_{idx}"] = {
                    "success": False,
                    "error": f"Provider {model_id} not found",
                    "capability": capability_type.value
                }
                continue
            
            # 准备输入
            step_input = TaskInput(
                data=input_data or {"task": "execute capability"},
                input_format="json",
                context={
                    "task_id": task_id,
                    "step": idx,
                    "total_steps": len(execution_plan.capability_sequence)
                }
            )
            
            try:
                # 执行能力
                start_time = time.time()
                output = provider.execute_capability(capability_type, step_input)
                end_time = time.time()
                
                # 记录执行指标
                metrics = ExecutionMetrics(
                    task_id=task_id,
                    capability_type=capability_type,
                    start_time=start_time,
                    end_time=end_time,
                    success=True,
                    result_quality=output.confidence,
                    resource_usage={}
                )
                
                self.registry.record_execution(model_id, capability_type, metrics)
                
                # 存储结果
                results[f"step_{idx}"] = {
                    "success": True,
                    "capability": capability_type.value,
                    "model": model_id,
                    "output": output.result,
                    "confidence": output.confidence,
                    "processing_time": output.processing_time
                }
                
                execution_times[f"step_{idx}"] = end_time - start_time
                
                logger.info(f"步骤 {idx} 完成: {capability_type.value} by {model_id}")
                
            except Exception as e:
                logger.error(f"步骤 {idx} 执行失败: {e}")
                results[f"step_{idx}"] = {
                    "success": False,
                    "error": str(e),
                    "capability": capability_type.value,
                    "model": model_id
                }
        
        # 更新执行历史
        with self.lock:
            if task_id in self.execution_history:
                self.execution_history[task_id].update({
                    "status": "completed",
                    "results": results,
                    "execution_times": execution_times,
                    "completed_at": time.time()
                })
        
        logger.info(f"任务执行完成: {task_id}")
        
        return {
            "task_id": task_id,
            "success": all(step.get("success", False) for step in results.values() if isinstance(step, dict)),
            "results": results,
            "execution_times": execution_times,
            "total_steps": len(execution_plan.capability_sequence)
        }


class ResultValidator:
    """结果验证器 - 评估任务执行质量"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """加载验证规则"""
        return {
            "completeness": {
                "min_required_fields": 1,
                "field_requirements": {
                    "text_output": ["result", "confidence"],
                    "json_output": ["result", "format"]
                }
            },
            "correctness": {
                "format_validation": True,
                "consistency_check": True
            },
            "efficiency": {
                "max_processing_time": 30.0,  # 秒
                "timeout_threshold": 60.0
            }
        }
    
    def validate_result(self, task_id: str, execution_result: Dict[str, Any]) -> ValidationResult:
        """
        验证执行结果
        
        Args:
            task_id: 任务ID
            execution_result: 执行结果
            
        Returns:
            验证结果
        """
        logger.info(f"开始验证任务结果: {task_id}")
        
        success = execution_result.get("success", False)
        
        if not success:
            return ValidationResult(
                task_id=task_id,
                success=False,
                quality_score=0.0,
                completeness_score=0.0,
                correctness_score=0.0,
                efficiency_score=0.0,
                feedback={"error": "Execution failed"},
                improvement_suggestions=["Check execution logs for errors"]
            )
        
        # 评估完整性
        completeness_score = self._assess_completeness(execution_result)
        
        # 评估正确性
        correctness_score = self._assess_correctness(execution_result)
        
        # 评估效率
        efficiency_score = self._assess_efficiency(execution_result)
        
        # 综合质量评分
        quality_score = completeness_score * 0.3 + correctness_score * 0.4 + efficiency_score * 0.3
        
        # 生成反馈和改进建议
        feedback, suggestions = self._generate_feedback(
            completeness_score, correctness_score, efficiency_score, execution_result
        )
        
        validation_result = ValidationResult(
            task_id=task_id,
            success=True,
            quality_score=quality_score,
            completeness_score=completeness_score,
            correctness_score=correctness_score,
            efficiency_score=efficiency_score,
            feedback=feedback,
            improvement_suggestions=suggestions
        )
        
        logger.info(f"结果验证完成: {task_id}, 质量评分: {quality_score:.2f}")
        return validation_result
    
    def _assess_completeness(self, execution_result: Dict[str, Any]) -> float:
        """评估完整性"""
        results = execution_result.get("results", {})
        
        if not results:
            return 0.0
        
        total_steps = execution_result.get("total_steps", 1)
        successful_steps = sum(1 for step in results.values() 
                              if isinstance(step, dict) and step.get("success", False))
        
        return successful_steps / total_steps if total_steps > 0 else 0.0
    
    def _assess_correctness(self, execution_result: Dict[str, Any]) -> float:
        """评估正确性"""
        results = execution_result.get("results", {})
        
        if not results:
            return 0.0
        
        confidence_scores = []
        for step_key, step_result in results.items():
            if isinstance(step_result, dict) and step_result.get("success", False):
                confidence = step_result.get("confidence", 0.5)
                confidence_scores.append(confidence)
        
        if not confidence_scores:
            return 0.0
        
        return sum(confidence_scores) / len(confidence_scores)
    
    def _assess_efficiency(self, execution_result: Dict[str, Any]) -> float:
        """评估效率"""
        execution_times = execution_result.get("execution_times", {})
        
        if not execution_times:
            return 0.5  # 默认中等评分
        
        total_time = sum(execution_times.values())
        max_time = self.validation_rules["efficiency"]["max_processing_time"]
        
        if total_time <= 0:
            return 1.0
        
        # 时间越短得分越高
        efficiency = max(0.0, 1.0 - (total_time / max_time))
        return min(1.0, efficiency)
    
    def _generate_feedback(self, completeness: float, correctness: float,
                          efficiency: float, execution_result: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """生成反馈和改进建议"""
        feedback = {
            "completeness_assessment": "优秀" if completeness > 0.9 else "良好" if completeness > 0.7 else "需要改进",
            "correctness_assessment": "高可信度" if correctness > 0.9 else "中等可信度" if correctness > 0.7 else "低可信度",
            "efficiency_assessment": "高效" if efficiency > 0.9 else "良好" if efficiency > 0.7 else "需要优化"
        }
        
        suggestions = []
        
        if completeness < 0.8:
            suggestions.append("部分步骤执行失败，建议检查模型可用性和输入数据")
        
        if correctness < 0.8:
            suggestions.append("结果置信度较低，建议使用更专业的模型或增加验证步骤")
        
        if efficiency < 0.7:
            suggestions.append("执行时间较长，建议优化任务分解或使用更高效的模型")
        
        if not suggestions:
            suggestions.append("执行质量良好，继续保持")
        
        return feedback, suggestions


class CoreSchedulingLayer:
    """核心调度层 - 整合所有组件"""
    
    def __init__(self):
        self.capability_registry = CapabilityRegistry()
        self.task_analyzer = TaskAnalyzer()
        self.capability_scheduler = CapabilityScheduler(self.capability_registry)
        self.result_validator = ResultValidator()
        self.lock = threading.RLock()
        
        logger.info("核心调度层初始化完成")
    
    def register_model(self, model_id: str, provider: ICapabilityProvider):
        """注册模型"""
        self.capability_registry.register_provider(model_id, provider)
    
    def process_task(self, task_id: str, description: str,
                    input_data: Optional[Dict] = None,
                    priority: TaskPriority = TaskPriority.MEDIUM) -> Dict[str, Any]:
        """
        处理完整任务流程
        
        Args:
            task_id: 任务ID
            description: 任务描述
            input_data: 输入数据
            priority: 任务优先级
            
        Returns:
            完整处理结果
        """
        logger.info(f"开始处理任务: {task_id}")
        
        # 1. 调度任务
        execution_plan = self.capability_scheduler.schedule_task(
            task_id, description, input_data, priority
        )
        
        # 2. 执行计划
        execution_result = self.capability_scheduler.execute_plan(
            execution_plan, input_data
        )
        
        # 3. 验证结果
        validation_result = self.result_validator.validate_result(
            task_id, execution_result
        )
        
        # 4. 返回综合结果
        result = {
            "task_id": task_id,
            "execution_result": execution_result,
            "validation_result": validation_result,
            "summary": {
                "success": execution_result["success"] and validation_result.success,
                "quality_score": validation_result.quality_score,
                "total_execution_time": sum(execution_result.get("execution_times", {}).values()),
                "steps_completed": len(execution_result.get("results", {}))
            }
        }
        
        logger.info(f"任务处理完成: {task_id}, 成功: {result['summary']['success']}")
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        with self.lock:
            return {
                "registered_models": len(self.capability_registry.providers),
                "available_capabilities": len(self.capability_registry.capability_index),
                "pending_tasks": len([h for h in self.capability_scheduler.execution_history.values() 
                                    if h.get("status") == "scheduled"]),
                "completed_tasks": len([h for h in self.capability_scheduler.execution_history.values() 
                                      if h.get("status") == "completed"]),
                "average_quality_score": self._calculate_average_quality_score()
            }
    
    def _calculate_average_quality_score(self) -> float:
        """计算平均质量评分"""
        quality_scores = []
        
        for task_history in self.capability_scheduler.execution_history.values():
            if task_history.get("status") == "completed" and "results" in task_history:
                # 简化的质量评估
                results = task_history["results"]
                successful_steps = sum(1 for step in results.values() 
                                      if isinstance(step, dict) and step.get("success", False))
                total_steps = len(results)
                
                if total_steps > 0:
                    quality_scores.append(successful_steps / total_steps)
        
        if not quality_scores:
            return 0.0
        
        return sum(quality_scores) / len(quality_scores)