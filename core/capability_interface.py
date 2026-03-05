"""
能力接口标准 - Capability Interface Standard

定义统一的AGI模型能力接口，支持核心调度层的动态发现和调用
"""

import abc
import json
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
import time


class CapabilityType(Enum):
    """能力类型枚举"""
    # 基础能力
    LANGUAGE_PROCESSING = "language_processing"          # 语言处理
    KNOWLEDGE_REASONING = "knowledge_reasoning"         # 知识推理
    VISION_ANALYSIS = "vision_analysis"                 # 视觉分析
    AUDIO_PROCESSING = "audio_processing"              # 音频处理
    SENSOR_DATA = "sensor_data"                        # 传感器数据处理
    SPATIAL_REASONING = "spatial_reasoning"            # 空间推理
    
    # 专业能力
    PROGRAMMING_CODE = "programming_code"              # 编程代码
    EMOTION_ANALYSIS = "emotion_analysis"              # 情感分析
    CREATIVE_GENERATION = "creative_generation"        # 创意生成
    LOGICAL_REASONING = "logical_reasoning"            # 逻辑推理
    DATA_ANALYSIS = "data_analysis"                    # 数据分析
    PREDICTIVE_MODELING = "predictive_modeling"        # 预测建模
    
    # 工程能力
    MECHANICAL_ENGINEERING = "mechanical_engineering"  # 机械工程
    ELECTRICAL_ENGINEERING = "electrical_engineering"  # 电气工程
    SYSTEM_OPTIMIZATION = "system_optimization"        # 系统优化
    
    # 医疗能力
    MEDICAL_ANALYSIS = "medical_analysis"              # 医疗分析
    DIAGNOSTIC_SUPPORT = "diagnostic_support"          # 诊断支持
    
    # 控制能力
    MOTION_CONTROL = "motion_control"                  # 运动控制
    COMPUTER_CONTROL = "computer_control"              # 计算机控制
    
    # AGI核心能力
    PLANNING_SCHEDULING = "planning_scheduling"        # 规划调度
    MULTIMODAL_FUSION = "multimodal_fusion"           # 多模态融合
    SELF_LEARNING = "self_learning"                   # 自主学习
    DECISION_MAKING = "decision_making"               # 决策制定
    COLLABORATION = "collaboration"                   # 协作协同


class CapabilityLevel(Enum):
    """能力水平枚举"""
    BASIC = "basic"           # 基础能力：能处理简单任务
    INTERMEDIATE = "intermediate"  # 中级能力：能处理标准任务
    ADVANCED = "advanced"     # 高级能力：能处理复杂任务
    EXPERT = "expert"         # 专家能力：能处理专业领域任务
    AGI = "agi"              # AGI能力：具备通用智能


@dataclass
class CapabilityMetadata:
    """能力元数据"""
    capability_type: CapabilityType
    capability_level: CapabilityLevel
    supported_formats: List[str] = field(default_factory=list)  # 支持的输入输出格式
    max_input_size: Optional[int] = None  # 最大输入大小
    max_output_size: Optional[int] = None  # 最大输出大小
    processing_speed: float = 1.0  # 处理速度系数（1.0为基准）
    accuracy_score: float = 0.8  # 准确度评分（0-1）
    reliability_score: float = 0.9  # 可靠性评分（0-1）
    resource_requirements: Dict[str, float] = field(default_factory=dict)  # 资源需求
    domain_specificity: List[str] = field(default_factory=list)  # 领域特异性


@dataclass
class TaskInput:
    """任务输入"""
    data: Union[str, Dict, List, bytes]  # 输入数据
    input_format: str  # 输入格式
    context: Optional[Dict[str, Any]] = None  # 上下文信息
    requirements: Optional[Dict[str, Any]] = None  # 特定要求


@dataclass
class TaskOutput:
    """任务输出"""
    result: Union[str, Dict, List, bytes]  # 输出结果
    output_format: str  # 输出格式
    confidence: float = 1.0  # 置信度（0-1）
    processing_time: float = 0.0  # 处理时间（秒）
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    errors: List[str] = field(default_factory=list)  # 错误信息


@dataclass
class ExecutionMetrics:
    """执行指标"""
    task_id: str
    capability_type: CapabilityType
    start_time: float
    end_time: float
    success: bool
    result_quality: float  # 结果质量评分（0-1）
    resource_usage: Dict[str, float]  # 资源使用情况
    error_messages: List[str] = field(default_factory=list)


class ICapabilityProvider(abc.ABC):
    """
    能力提供者接口
    所有模型需要实现此接口以支持核心调度层的动态调用
    """
    
    @abc.abstractmethod
    def get_capabilities(self) -> Dict[CapabilityType, CapabilityMetadata]:
        """
        获取模型支持的所有能力及其元数据
        
        Returns:
            能力类型到能力元数据的映射
        """
        pass
    
    @abc.abstractmethod
    def can_handle(self, capability_type: CapabilityType, 
                  input_data: TaskInput) -> bool:
        """
        检查是否能处理指定类型的任务输入
        
        Args:
            capability_type: 能力类型
            input_data: 任务输入
            
        Returns:
            是否能处理
        """
        pass
    
    @abc.abstractmethod
    def execute_capability(self, capability_type: CapabilityType,
                          input_data: TaskInput) -> TaskOutput:
        """
        执行指定能力类型的任务
        
        Args:
            capability_type: 能力类型
            input_data: 任务输入
            
        Returns:
            任务输出
        """
        pass
    
    @abc.abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        pass
    
    def get_capability_score(self, capability_type: CapabilityType) -> float:
        """
        获取指定能力的综合评分（可选实现）
        
        Args:
            capability_type: 能力类型
            
        Returns:
            综合评分（0-1）
        """
        capabilities = self.get_capabilities()
        if capability_type not in capabilities:
            return 0.0
        
        metadata = capabilities[capability_type]
        # 综合评分 = 准确度 * 0.4 + 可靠性 * 0.3 + 处理速度 * 0.3
        accuracy = metadata.accuracy_score
        reliability = metadata.reliability_score
        speed = min(metadata.processing_speed, 2.0) / 2.0  # 归一化到0-1
        
        return accuracy * 0.4 + reliability * 0.3 + speed * 0.3
    
    def get_resource_requirements(self, capability_type: CapabilityType) -> Dict[str, float]:
        """
        获取执行指定能力所需的资源需求（可选实现）
        
        Args:
            capability_type: 能力类型
            
        Returns:
            资源需求字典
        """
        capabilities = self.get_capabilities()
        if capability_type not in capabilities:
            return {}
        
        return capabilities[capability_type].resource_requirements


class CapabilityAdapter:
    """
    能力适配器
    将现有模型适配到统一的能力接口
    """
    
    def __init__(self, model_instance, adapter_mappings: Dict[CapabilityType, Callable]):
        """
        初始化适配器
        
        Args:
            model_instance: 模型实例
            adapter_mappings: 能力类型到适配函数的映射
        """
        self.model = model_instance
        self.adapter_mappings = adapter_mappings
        self.model_id = getattr(model_instance, 'model_id', 'unknown')
    
    def get_capabilities(self) -> Dict[CapabilityType, CapabilityMetadata]:
        """获取模型能力"""
        capabilities = {}
        
        for capability_type, adapter_func in self.adapter_mappings.items():
            # 根据模型类型和能力类型创建元数据
            metadata = CapabilityMetadata(
                capability_type=capability_type,
                capability_level=CapabilityLevel.INTERMEDIATE,
                accuracy_score=0.8,
                reliability_score=0.9,
                processing_speed=1.0
            )
            capabilities[capability_type] = metadata
        
        return capabilities
    
    def can_handle(self, capability_type: CapabilityType, 
                  input_data: TaskInput) -> bool:
        """检查是否能处理"""
        return capability_type in self.adapter_mappings
    
    def execute_capability(self, capability_type: CapabilityType,
                          input_data: TaskInput) -> TaskOutput:
        """执行能力"""
        if capability_type not in self.adapter_mappings:
            return TaskOutput(
                result=f"Unsupported capability: {capability_type}",
                output_format="error",
                confidence=0.0,
                errors=[f"Capability {capability_type} not supported"]
            )
        
        adapter_func = self.adapter_mappings[capability_type]
        try:
            start_time = time.time()
            # 调用适配函数
            result = adapter_func(self.model, input_data.data)
            end_time = time.time()
            
            return TaskOutput(
                result=result,
                output_format="json" if isinstance(result, (dict, list)) else "text",
                confidence=0.9,
                processing_time=end_time - start_time,
                metadata={
                    "model_id": self.model_id,
                    "capability_type": capability_type.value,
                    "adapter_used": True
                }
            )
        except Exception as e:
            return TaskOutput(
                result=f"Error executing capability: {str(e)}",
                output_format="error",
                confidence=0.0,
                processing_time=0.0,
                errors=[f"Execution error: {str(e)}"]
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_id": self.model_id,
            "model_type": type(self.model).__name__,
            "adapted": True,
            "capabilities": list(self.adapter_mappings.keys())
        }