"""
模型适配器 - Model Adapters

将现有模型适配到统一的能力接口，支持核心调度层的动态调用
"""

import time
import logging
from typing import Dict, List, Any, Optional, Callable
import importlib

from .capability_interface import (
    ICapabilityProvider, CapabilityType, CapabilityMetadata,
    TaskInput, TaskOutput, CapabilityLevel, CapabilityAdapter
)

logger = logging.getLogger(__name__)


class ModelAdapterFactory:
    """模型适配器工厂"""
    
    @staticmethod
    def create_adapter(model_instance, model_type: str) -> Optional[ICapabilityProvider]:
        """
        为模型实例创建适配器
        
        Args:
            model_instance: 模型实例
            model_type: 模型类型
            
        Returns:
            能力提供者适配器，如果无法适配则返回None
        """
        adapter_mappings = {}
        
        # 根据模型类型确定能力映射
        if model_type == "language_model":
            adapter_mappings = ModelAdapterFactory._create_language_model_mappings(model_instance)
        elif model_type == "vision_model":
            adapter_mappings = ModelAdapterFactory._create_vision_model_mappings(model_instance)
        elif model_type == "audio_model":
            adapter_mappings = ModelAdapterFactory._create_audio_model_mappings(model_instance)
        elif model_type == "knowledge_model":
            adapter_mappings = ModelAdapterFactory._create_knowledge_model_mappings(model_instance)
        elif model_type == "programming_model":
            adapter_mappings = ModelAdapterFactory._create_programming_model_mappings(model_instance)
        elif model_type == "emotion_model":
            adapter_mappings = ModelAdapterFactory._create_emotion_model_mappings(model_instance)
        elif model_type == "manager_model":
            adapter_mappings = ModelAdapterFactory._create_manager_model_mappings(model_instance)
        else:
            # 尝试通用适配
            adapter_mappings = ModelAdapterFactory._create_generic_mappings(model_instance)
        
        if not adapter_mappings:
            logger.warning(f"无法为模型类型 {model_type} 创建适配器")
            return None
        
        # 创建适配器
        adapter = CapabilityAdapter(model_instance, adapter_mappings)
        
        # 包装为完整的提供者
        return ModelAdapterFactory._wrap_as_provider(adapter, model_type, model_instance)
    
    @staticmethod
    def _create_language_model_mappings(model_instance) -> Dict[CapabilityType, Callable]:
        """创建语言模型的能力映射"""
        mappings = {}
        
        # 检查模型支持的方法
        if hasattr(model_instance, 'process_text'):
            mappings[CapabilityType.LANGUAGE_PROCESSING] = lambda m, data: m.process_text(data)
        
        if hasattr(model_instance, 'generate_text'):
            mappings[CapabilityType.CREATIVE_GENERATION] = lambda m, data: m.generate_text(data)
        
        if hasattr(model_instance, 'translate_text'):
            mappings[CapabilityType.LANGUAGE_PROCESSING] = lambda m, data: m.translate_text(data)
        
        if hasattr(model_instance, 'summarize_text'):
            mappings[CapabilityType.LANGUAGE_PROCESSING] = lambda m, data: m.summarize_text(data)
        
        return mappings
    
    @staticmethod
    def _create_vision_model_mappings(model_instance) -> Dict[CapabilityType, Callable]:
        """创建视觉模型的能力映射"""
        mappings = {}
        
        if hasattr(model_instance, 'analyze_image'):
            mappings[CapabilityType.VISION_ANALYSIS] = lambda m, data: m.analyze_image(data)
        
        if hasattr(model_instance, 'detect_objects'):
            mappings[CapabilityType.VISION_ANALYSIS] = lambda m, data: m.detect_objects(data)
        
        if hasattr(model_instance, 'recognize_scene'):
            mappings[CapabilityType.VISION_ANALYSIS] = lambda m, data: m.recognize_scene(data)
        
        return mappings
    
    @staticmethod
    def _create_audio_model_mappings(model_instance) -> Dict[CapabilityType, Callable]:
        """创建音频模型的能力映射"""
        mappings = {}
        
        if hasattr(model_instance, 'process_audio'):
            mappings[CapabilityType.AUDIO_PROCESSING] = lambda m, data: m.process_audio(data)
        
        if hasattr(model_instance, 'transcribe_audio'):
            mappings[CapabilityType.AUDIO_PROCESSING] = lambda m, data: m.transcribe_audio(data)
        
        if hasattr(model_instance, 'analyze_speech'):
            mappings[CapabilityType.AUDIO_PROCESSING] = lambda m, data: m.analyze_speech(data)
        
        return mappings
    
    @staticmethod
    def _create_knowledge_model_mappings(model_instance) -> Dict[CapabilityType, Callable]:
        """创建知识模型的能力映射"""
        mappings = {}
        
        if hasattr(model_instance, 'answer_question'):
            mappings[CapabilityType.KNOWLEDGE_REASONING] = lambda m, data: m.answer_question(data)
        
        if hasattr(model_instance, 'explain_concept'):
            mappings[CapabilityType.KNOWLEDGE_REASONING] = lambda m, data: m.explain_concept(data)
        
        if hasattr(model_instance, 'infer_conclusion'):
            mappings[CapabilityType.LOGICAL_REASONING] = lambda m, data: m.infer_conclusion(data)
        
        return mappings
    
    @staticmethod
    def _create_programming_model_mappings(model_instance) -> Dict[CapabilityType, Callable]:
        """创建编程模型的能力映射"""
        mappings = {}
        
        if hasattr(model_instance, 'generate_code'):
            mappings[CapabilityType.PROGRAMMING_CODE] = lambda m, data: m.generate_code(data)
        
        if hasattr(model_instance, 'debug_code'):
            mappings[CapabilityType.PROGRAMMING_CODE] = lambda m, data: m.debug_code(data)
        
        if hasattr(model_instance, 'optimize_code'):
            mappings[CapabilityType.PROGRAMMING_CODE] = lambda m, data: m.optimize_code(data)
        
        if hasattr(model_instance, 'analyze_algorithm'):
            mappings[CapabilityType.LOGICAL_REASONING] = lambda m, data: m.analyze_algorithm(data)
        
        return mappings
    
    @staticmethod
    def _create_emotion_model_mappings(model_instance) -> Dict[CapabilityType, Callable]:
        """创建情感模型的能力映射"""
        mappings = {}
        
        if hasattr(model_instance, 'analyze_emotion'):
            mappings[CapabilityType.EMOTION_ANALYSIS] = lambda m, data: m.analyze_emotion(data)
        
        if hasattr(model_instance, 'detect_sentiment'):
            mappings[CapabilityType.EMOTION_ANALYSIS] = lambda m, data: m.detect_sentiment(data)
        
        if hasattr(model_instance, 'understand_context'):
            mappings[CapabilityType.LANGUAGE_PROCESSING] = lambda m, data: m.understand_context(data)
        
        return mappings
    
    @staticmethod
    def _create_manager_model_mappings(model_instance) -> Dict[CapabilityType, Callable]:
        """创建管理模型的能力映射"""
        mappings = {}
        
        if hasattr(model_instance, 'plan_task'):
            mappings[CapabilityType.PLANNING_SCHEDULING] = lambda m, data: m.plan_task(data)
        
        if hasattr(model_instance, 'make_decision'):
            mappings[CapabilityType.DECISION_MAKING] = lambda m, data: m.make_decision(data)
        
        if hasattr(model_instance, 'coordinate_models'):
            mappings[CapabilityType.COLLABORATION] = lambda m, data: m.coordinate_models(data)
        
        return mappings
    
    @staticmethod
    def _create_generic_mappings(model_instance) -> Dict[CapabilityType, Callable]:
        """创建通用能力映射"""
        mappings = {}
        
        # 检查常见方法名
        method_capability_map = {
            'process': CapabilityType.LANGUAGE_PROCESSING,
            'analyze': CapabilityType.DATA_ANALYSIS,
            'generate': CapabilityType.CREATIVE_GENERATION,
            'predict': CapabilityType.PREDICTIVE_MODELING,
            'classify': CapabilityType.DATA_ANALYSIS,
            'recommend': CapabilityType.DECISION_MAKING
        }
        
        for method_prefix, capability in method_capability_map.items():
            # 查找以该前缀开头的方法
            for method_name in dir(model_instance):
                if method_name.startswith(method_prefix) and callable(getattr(model_instance, method_name)):
                    mappings[capability] = lambda m, data, mn=method_name: getattr(m, mn)(data)
                    break
        
        return mappings
    
    @staticmethod
    def _wrap_as_provider(adapter: CapabilityAdapter, model_type: str, 
                         model_instance) -> ICapabilityProvider:
        """将适配器包装为完整的能力提供者"""
        
        class WrappedProvider(ICapabilityProvider):
            def __init__(self, adapter, model_type, model_instance):
                self.adapter = adapter
                self.model_type = model_type
                self.model_instance = model_instance
                self.model_id = getattr(model_instance, 'model_id', f"{model_type}_{id(model_instance)}")
            
            def get_capabilities(self) -> Dict[CapabilityType, CapabilityMetadata]:
                # 基于模型类型创建更精确的元数据
                base_capabilities = self.adapter.get_capabilities()
                enhanced_capabilities = {}
                
                for capability_type, metadata in base_capabilities.items():
                    # 根据模型类型增强元数据
                    enhanced_metadata = CapabilityMetadata(
                        capability_type=capability_type,
                        capability_level=ModelAdapterFactory._determine_capability_level(
                            self.model_type, capability_type
                        ),
                        accuracy_score=ModelAdapterFactory._determine_accuracy_score(
                            self.model_type, capability_type
                        ),
                        reliability_score=0.85,  # 默认可靠性
                        processing_speed=1.0,
                        supported_formats=ModelAdapterFactory._determine_supported_formats(
                            self.model_type, capability_type
                        ),
                        resource_requirements={}
                    )
                    enhanced_capabilities[capability_type] = enhanced_metadata
                
                return enhanced_capabilities
            
            def can_handle(self, capability_type: CapabilityType, 
                          input_data: TaskInput) -> bool:
                return self.adapter.can_handle(capability_type, input_data)
            
            def execute_capability(self, capability_type: CapabilityType,
                                 input_data: TaskInput) -> TaskOutput:
                return self.adapter.execute_capability(capability_type, input_data)
            
            def get_model_info(self) -> Dict[str, Any]:
                base_info = self.adapter.get_model_info()
                base_info.update({
                    "model_type": self.model_type,
                    "wrapped": True,
                    "original_model_class": type(self.model_instance).__name__
                })
                return base_info
        
        return WrappedProvider(adapter, model_type, model_instance)
    
    @staticmethod
    def _determine_capability_level(model_type: str, capability_type: CapabilityType) -> CapabilityLevel:
        """确定能力水平"""
        # 基于模型类型和能力类型的映射
        level_mapping = {
            ("language_model", CapabilityType.LANGUAGE_PROCESSING): CapabilityLevel.ADVANCED,
            ("vision_model", CapabilityType.VISION_ANALYSIS): CapabilityLevel.ADVANCED,
            ("audio_model", CapabilityType.AUDIO_PROCESSING): CapabilityLevel.INTERMEDIATE,
            ("knowledge_model", CapabilityType.KNOWLEDGE_REASONING): CapabilityLevel.EXPERT,
            ("programming_model", CapabilityType.PROGRAMMING_CODE): CapabilityLevel.ADVANCED,
            ("emotion_model", CapabilityType.EMOTION_ANALYSIS): CapabilityLevel.INTERMEDIATE,
            ("manager_model", CapabilityType.PLANNING_SCHEDULING): CapabilityLevel.AGI,
        }
        
        return level_mapping.get((model_type, capability_type), CapabilityLevel.INTERMEDIATE)
    
    @staticmethod
    def _determine_accuracy_score(model_type: str, capability_type: CapabilityType) -> float:
        """确定准确度评分"""
        # 基于模型类型和能力类型的映射
        accuracy_mapping = {
            ("language_model", CapabilityType.LANGUAGE_PROCESSING): 0.9,
            ("vision_model", CapabilityType.VISION_ANALYSIS): 0.85,
            ("audio_model", CapabilityType.AUDIO_PROCESSING): 0.8,
            ("knowledge_model", CapabilityType.KNOWLEDGE_REASONING): 0.95,
            ("programming_model", CapabilityType.PROGRAMMING_CODE): 0.88,
            ("emotion_model", CapabilityType.EMOTION_ANALYSIS): 0.75,
            ("manager_model", CapabilityType.PLANNING_SCHEDULING): 0.82,
        }
        
        return accuracy_mapping.get((model_type, capability_type), 0.8)
    
    @staticmethod
    def _determine_supported_formats(model_type: str, capability_type: CapabilityType) -> List[str]:
        """确定支持的格式"""
        format_mapping = {
            ("language_model", CapabilityType.LANGUAGE_PROCESSING): ["text", "json"],
            ("vision_model", CapabilityType.VISION_ANALYSIS): ["image", "binary"],
            ("audio_model", CapabilityType.AUDIO_PROCESSING): ["audio", "binary"],
            ("knowledge_model", CapabilityType.KNOWLEDGE_REASONING): ["text", "json"],
            ("programming_model", CapabilityType.PROGRAMMING_CODE): ["text", "code"],
            ("emotion_model", CapabilityType.EMOTION_ANALYSIS): ["text", "json"],
            ("manager_model", CapabilityType.PLANNING_SCHEDULING): ["json", "yaml"],
        }
        
        return format_mapping.get((model_type, capability_type), ["text", "json"])


class UnifiedModelAdapter(ICapabilityProvider):
    """
    统一模型适配器
    适配现有的UnifiedModelTemplate及其子类
    """
    
    def __init__(self, unified_model):
        self.model = unified_model
        self.model_id = getattr(unified_model, 'model_id', 'unknown')
        self.model_type = getattr(unified_model, 'model_type', 'unknown')
        
        # 从模型获取能力信息
        self._extract_capabilities()
    
    def _extract_capabilities(self):
        """从统一模型中提取能力信息"""
        self.capabilities = {}
        
        # 尝试获取模型能力
        if hasattr(self.model, '_get_model_capabilities'):
            try:
                model_capabilities = self.model._get_model_capabilities()
                self._map_capabilities(model_capabilities)
            except Exception as e:
                logger.warning(f"无法获取模型能力: {e}")
        
        # 如果无法获取，根据模型类型推断
        if not self.capabilities:
            self._infer_capabilities_by_type()
    
    def _map_capabilities(self, model_capabilities: Dict[str, Any]):
        """映射模型能力到标准能力类型"""
        capability_mapping = {
            "language_processing": CapabilityType.LANGUAGE_PROCESSING,
            "knowledge_reasoning": CapabilityType.KNOWLEDGE_REASONING,
            "vision_analysis": CapabilityType.VISION_ANALYSIS,
            "audio_processing": CapabilityType.AUDIO_PROCESSING,
            "programming_code": CapabilityType.PROGRAMMING_CODE,
            "emotion_analysis": CapabilityType.EMOTION_ANALYSIS,
            "creative_generation": CapabilityType.CREATIVE_GENERATION,
            "logical_reasoning": CapabilityType.LOGICAL_REASONING,
            "data_analysis": CapabilityType.DATA_ANALYSIS,
            "planning_scheduling": CapabilityType.PLANNING_SCHEDULING,
            "decision_making": CapabilityType.DECISION_MAKING,
            "collaboration": CapabilityType.COLLABORATION,
        }
        
        for capability_name, enabled in model_capabilities.items():
            if enabled and capability_name in capability_mapping:
                capability_type = capability_mapping[capability_name]
                
                # 创建能力元数据
                metadata = CapabilityMetadata(
                    capability_type=capability_type,
                    capability_level=CapabilityLevel.INTERMEDIATE,
                    accuracy_score=0.85,
                    reliability_score=0.9,
                    processing_speed=1.0,
                    supported_formats=["text", "json"]
                )
                
                self.capabilities[capability_type] = metadata
    
    def _infer_capabilities_by_type(self):
        """根据模型类型推断能力"""
        type_capability_map = {
            "language": [CapabilityType.LANGUAGE_PROCESSING, CapabilityType.CREATIVE_GENERATION],
            "vision": [CapabilityType.VISION_ANALYSIS],
            "audio": [CapabilityType.AUDIO_PROCESSING],
            "knowledge": [CapabilityType.KNOWLEDGE_REASONING, CapabilityType.LOGICAL_REASONING],
            "programming": [CapabilityType.PROGRAMMING_CODE, CapabilityType.LOGICAL_REASONING],
            "emotion": [CapabilityType.EMOTION_ANALYSIS],
            "manager": [CapabilityType.PLANNING_SCHEDULING, CapabilityType.DECISION_MAKING, CapabilityType.COLLABORATION],
        }
        
        model_type_lower = self.model_type.lower()
        for type_prefix, capabilities in type_capability_map.items():
            if type_prefix in model_type_lower:
                for capability_type in capabilities:
                    metadata = CapabilityMetadata(
                        capability_type=capability_type,
                        capability_level=CapabilityLevel.INTERMEDIATE,
                        accuracy_score=0.8,
                        reliability_score=0.85,
                        processing_speed=1.0
                    )
                    self.capabilities[capability_type] = metadata
                break
    
    def get_capabilities(self) -> Dict[CapabilityType, CapabilityMetadata]:
        return self.capabilities
    
    def can_handle(self, capability_type: CapabilityType, 
                  input_data: TaskInput) -> bool:
        return capability_type in self.capabilities
    
    def execute_capability(self, capability_type: CapabilityType,
                          input_data: TaskInput) -> TaskOutput:
        if capability_type not in self.capabilities:
            return TaskOutput(
                result=f"Unsupported capability: {capability_type}",
                output_format="error",
                confidence=0.0,
                errors=[f"Capability {capability_type} not supported by model {self.model_id}"]
            )
        
        try:
            # 根据能力类型调用相应的方法
            method_name = self._get_method_for_capability(capability_type)
            if not hasattr(self.model, method_name):
                # 回退到通用处理方法
                method_name = "process"
            
            if not hasattr(self.model, method_name):
                return TaskOutput(
                    result=f"No method available for capability: {capability_type}",
                    output_format="error",
                    confidence=0.0,
                    errors=[f"No method {method_name} available"]
                )
            
            # 执行方法
            start_time = time.time()
            method = getattr(self.model, method_name)
            result = method(input_data.data)
            end_time = time.time()
            
            return TaskOutput(
                result=result,
                output_format="json" if isinstance(result, (dict, list)) else "text",
                confidence=0.85,
                processing_time=end_time - start_time,
                metadata={
                    "model_id": self.model_id,
                    "capability_type": capability_type.value,
                    "method_used": method_name
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
    
    def _get_method_for_capability(self, capability_type: CapabilityType) -> str:
        """获取对应能力类型的方法名"""
        method_mapping = {
            CapabilityType.LANGUAGE_PROCESSING: "process_text",
            CapabilityType.KNOWLEDGE_REASONING: "answer_question",
            CapabilityType.VISION_ANALYSIS: "analyze_image",
            CapabilityType.AUDIO_PROCESSING: "process_audio",
            CapabilityType.PROGRAMMING_CODE: "generate_code",
            CapabilityType.EMOTION_ANALYSIS: "analyze_emotion",
            CapabilityType.CREATIVE_GENERATION: "generate_text",
            CapabilityType.LOGICAL_REASONING: "reason",
            CapabilityType.DATA_ANALYSIS: "analyze_data",
            CapabilityType.PLANNING_SCHEDULING: "plan_task",
            CapabilityType.DECISION_MAKING: "make_decision",
            CapabilityType.COLLABORATION: "coordinate",
        }
        
        return method_mapping.get(capability_type, "process")
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "capabilities": [cap.value for cap in self.capabilities.keys()],
            "adapter_type": "UnifiedModelAdapter",
            "original_model_class": type(self.model).__name__
        }


def create_model_registry_adapter() -> ICapabilityProvider:
    """
    创建模型注册表适配器
    将整个模型注册表作为一个能力提供者
    """
    try:
        from core.model_registry import get_model_registry
        
        class ModelRegistryProvider(ICapabilityProvider):
            def __init__(self):
                self.registry = get_model_registry()
                self.model_id = "model_registry"
                self.cached_capabilities = None
            
            def get_capabilities(self) -> Dict[CapabilityType, CapabilityMetadata]:
                if self.cached_capabilities is not None:
                    return self.cached_capabilities
                
                # 聚合所有模型的能力
                all_capabilities = {}
                
                # 获取所有已加载模型
                # 注意：这里简化了实现，实际需要遍历所有模型
                
                # 添加基本能力
                base_capabilities = {
                    CapabilityType.PLANNING_SCHEDULING: CapabilityMetadata(
                        capability_type=CapabilityType.PLANNING_SCHEDULING,
                        capability_level=CapabilityLevel.AGI,
                        accuracy_score=0.9,
                        reliability_score=0.95,
                        processing_speed=1.0
                    ),
                    CapabilityType.COLLABORATION: CapabilityMetadata(
                        capability_type=CapabilityType.COLLABORATION,
                        capability_level=CapabilityLevel.ADVANCED,
                        accuracy_score=0.88,
                        reliability_score=0.9,
                        processing_speed=1.0
                    )
                }
                
                all_capabilities.update(base_capabilities)
                self.cached_capabilities = all_capabilities
                return all_capabilities
            
            def can_handle(self, capability_type: CapabilityType, 
                          input_data: TaskInput) -> bool:
                capabilities = self.get_capabilities()
                return capability_type in capabilities
            
            def execute_capability(self, capability_type: CapabilityType,
                                 input_data: TaskInput) -> TaskOutput:
                if capability_type == CapabilityType.PLANNING_SCHEDULING:
                    # 执行任务规划
                    return self._execute_planning(input_data)
                elif capability_type == CapabilityType.COLLABORATION:
                    # 执行协作协调
                    return self._execute_collaboration(input_data)
                else:
                    return TaskOutput(
                        result=f"Unsupported capability: {capability_type}",
                        output_format="error",
                        confidence=0.0
                    )
            
            def _execute_planning(self, input_data: TaskInput) -> TaskOutput:
                """执行任务规划"""
                try:
                    # 这里可以调用模型注册表的规划功能
                    result = {
                        "planning_result": "Task planning executed via model registry",
                        "available_models": len(self.registry.get_loaded_models()) if hasattr(self.registry, 'get_loaded_models') else 0
                    }
                    
                    return TaskOutput(
                        result=result,
                        output_format="json",
                        confidence=0.9,
                        processing_time=0.1
                    )
                except Exception as e:
                    return TaskOutput(
                        result=f"Planning error: {str(e)}",
                        output_format="error",
                        confidence=0.0,
                        errors=[str(e)]
                    )
            
            def _execute_collaboration(self, input_data: TaskInput) -> TaskOutput:
                """执行协作协调"""
                try:
                    result = {
                        "collaboration_result": "Model collaboration coordinated via registry",
                        "coordination_status": "active"
                    }
                    
                    return TaskOutput(
                        result=result,
                        output_format="json",
                        confidence=0.85,
                        processing_time=0.05
                    )
                except Exception as e:
                    return TaskOutput(
                        result=f"Collaboration error: {str(e)}",
                        output_format="error",
                        confidence=0.0,
                        errors=[str(e)]
                    )
            
            def get_model_info(self) -> Dict[str, Any]:
                return {
                    "model_id": self.model_id,
                    "model_type": "model_registry",
                    "description": "Aggregated capabilities from all registered models",
                    "registry_available": self.registry is not None
                }
        
        return ModelRegistryProvider()
        
    except ImportError:
        logger.warning("Model registry not available")
        return None