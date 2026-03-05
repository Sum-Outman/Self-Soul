"""
防循环统一配置系统 - Cycle Prevention Unified Configuration System

提供全项目统一的防循环配置管理，支持：
1. 模块化配置预设：为不同功能模块提供优化配置
2. 场景自适应：根据应用场景动态调整参数
3. 集中式管理：统一配置，便于维护和监控
4. 性能优化：基于历史性能数据优化参数

设计原则：
- 嵌入式思维：像单片机一样可靠，参数简单稳定
- 分层防护：基础层确保安全，高级层提供智能
- 统一接口：所有模块使用同一套配置体系
- 自适应调整：根据场景和性能动态优化
"""

import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SceneType(Enum):
    """场景类型枚举"""
    CREATIVE_WRITING = "creative_writing"        # 创意写作
    INDUSTRIAL_CONTROL = "industrial_control"    # 工业控制
    MEDICAL_ANALYSIS = "medical_analysis"        # 医疗分析
    FINANCIAL_ANALYSIS = "financial_analysis"    # 金融分析
    GENERAL_CONVERSATION = "general_conversation" # 通用对话
    CODE_GENERATION = "code_generation"          # 代码生成
    SCIENTIFIC_RESEARCH = "scientific_research"  # 科学研究
    EDUCATION = "education"                      # 教育辅导
    EMOTIONAL_SUPPORT = "emotional_support"      # 情感支持
    TECHNICAL_SUPPORT = "technical_support"      # 技术支持


@dataclass
class CyclePreventionConfig:
    """防循环基础配置类
    
    嵌入式思维配置参数，对应单片机硬件配置：
    - 缓冲区清理：环形缓冲区大小，防止上下文爆炸
    - 重复检测：故障检测阈值，连续N次重复=故障
    - 温度调节：PID阻尼系数，控制随机性
    - 重复惩罚：电机反电动势，抑制重复
    
    配置原则：
    1. 简单可靠：参数数量最小化，避免过度配置
    2. 范围保护：所有参数都有合理范围限制
    3. 增量调整：每次调整有上限，防止参数漂移
    4. 场景适配：不同场景使用不同参数预设
    """
    
    # ==================== 基础层参数（嵌入式思维）====================
    history_buffer_size: int = 10           # 对话历史缓冲区大小（最多存N轮）
    repeat_threshold: int = 3               # 重复检测阈值（连续N次重复触发防护）
    max_retry_attempts: int = 3             # 最大重试次数（看门狗重置次数）
    
    # ==================== 温度调节配置（对应PID阻尼系数）====================
    base_temperature: float = 0.7           # 基础温度（大厂最优值）
    temperature_increment: float = 0.1      # 温度增量（每次循环增加）
    max_temperature: float = 1.0            # 最大温度限制（防止过度随机）
    min_temperature: float = 0.1            # 最小温度限制（防止过度确定）
    
    # ==================== 重复惩罚配置（对应电机反电动势）====================
    base_repetition_penalty: float = 1.2    # 基础重复惩罚（抑制重复）
    penalty_increment: float = 0.05         # 惩罚增量（每次循环增加）
    max_repetition_penalty: float = 1.5     # 最大重复惩罚限制
    min_repetition_penalty: float = 1.0     # 最小重复惩罚限制（禁止负惩罚）
    
    # ==================== 场景自适应配置 ====================
    scene_type: SceneType = SceneType.GENERAL_CONVERSATION  # 场景类型
    enable_adaptive_layer: bool = True      # 是否启用场景自适应高级防护
    
    # ==================== 性能监控配置 ====================
    enable_performance_monitoring: bool = True  # 是否启用性能监控
    performance_history_size: int = 100     # 性能历史记录大小
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "history_buffer_size": self.history_buffer_size,
            "repeat_threshold": self.repeat_threshold,
            "max_retry_attempts": self.max_retry_attempts,
            "base_temperature": self.base_temperature,
            "temperature_increment": self.temperature_increment,
            "max_temperature": self.max_temperature,
            "min_temperature": self.min_temperature,
            "base_repetition_penalty": self.base_repetition_penalty,
            "penalty_increment": self.penalty_increment,
            "max_repetition_penalty": self.max_repetition_penalty,
            "min_repetition_penalty": self.min_repetition_penalty,
            "scene_type": self.scene_type.value,
            "enable_adaptive_layer": self.enable_adaptive_layer,
            "enable_performance_monitoring": self.enable_performance_monitoring,
            "performance_history_size": self.performance_history_size,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CyclePreventionConfig":
        """从字典创建配置实例"""
        # 处理scene_type转换
        if "scene_type" in config_dict and isinstance(config_dict["scene_type"], str):
            config_dict["scene_type"] = SceneType(config_dict["scene_type"])
        
        return cls(**config_dict)
    
    def validate(self) -> List[str]:
        """验证配置参数，返回错误列表"""
        errors = []
        
        # 验证整数参数
        if self.history_buffer_size < 1:
            errors.append(f"history_buffer_size必须大于0，当前值: {self.history_buffer_size}")
        if self.repeat_threshold < 1:
            errors.append(f"repeat_threshold必须大于0，当前值: {self.repeat_threshold}")
        if self.max_retry_attempts < 1:
            errors.append(f"max_retry_attempts必须大于0，当前值: {self.max_retry_attempts}")
        if self.performance_history_size < 1:
            errors.append(f"performance_history_size必须大于0，当前值: {self.performance_history_size}")
        
        # 验证温度参数
        if not 0.0 < self.base_temperature <= 2.0:
            errors.append(f"base_temperature必须在0.0-2.0之间，当前值: {self.base_temperature}")
        if not 0.0 < self.temperature_increment <= 0.5:
            errors.append(f"temperature_increment必须在0.0-0.5之间，当前值: {self.temperature_increment}")
        if not 0.0 < self.max_temperature <= 2.0:
            errors.append(f"max_temperature必须在0.0-2.0之间，当前值: {self.max_temperature}")
        if not 0.0 < self.min_temperature <= 2.0:
            errors.append(f"min_temperature必须在0.0-2.0之间，当前值: {self.min_temperature}")
        if self.min_temperature > self.max_temperature:
            errors.append(f"min_temperature不能大于max_temperature: {self.min_temperature} > {self.max_temperature}")
        
        # 验证重复惩罚参数
        if not 1.0 <= self.base_repetition_penalty <= 2.0:
            errors.append(f"base_repetition_penalty必须在1.0-2.0之间，当前值: {self.base_repetition_penalty}")
        if not 0.0 < self.penalty_increment <= 0.2:
            errors.append(f"penalty_increment必须在0.0-0.2之间，当前值: {self.penalty_increment}")
        if not 1.0 <= self.max_repetition_penalty <= 2.5:
            errors.append(f"max_repetition_penalty必须在1.0-2.5之间，当前值: {self.max_repetition_penalty}")
        if not 1.0 <= self.min_repetition_penalty <= 2.5:
            errors.append(f"min_repetition_penalty必须在1.0-2.5之间，当前值: {self.min_repetition_penalty}")
        if self.min_repetition_penalty > self.max_repetition_penalty:
            errors.append(f"min_repetition_penalty不能大于max_repetition_penalty: {self.min_repetition_penalty} > {self.max_repetition_penalty}")
        
        return errors


class CyclePreventionConfigFactory:
    """防循环配置工厂类
    
    为不同模块和场景提供优化配置预设
    """
    
    # ==================== 模块预设配置 ====================
    @staticmethod
    def get_creative_writing_config() -> CyclePreventionConfig:
        """创意写作模块配置"""
        return CyclePreventionConfig(
            history_buffer_size=15,           # 需要更大上下文
            repeat_threshold=4,               # 创意写作允许更多重复
            base_temperature=0.8,             # 更高温度增加创造性
            max_temperature=1.2,              # 允许更高温度
            base_repetition_penalty=1.1,      # 较低惩罚允许创意重复
            max_repetition_penalty=1.8,       # 但上限较高以防失控
            scene_type=SceneType.CREATIVE_WRITING,
            enable_adaptive_layer=True
        )
    
    @staticmethod
    def get_industrial_control_config() -> CyclePreventionConfig:
        """工业控制模块配置"""
        return CyclePreventionConfig(
            history_buffer_size=5,            # 工业指令上下文短
            repeat_threshold=2,               # 工业指令不允许重复
            base_temperature=0.3,             # 低温度确保确定性
            max_temperature=0.8,              # 严格限制随机性
            base_repetition_penalty=1.05,     # 很低惩罚，工业指令需要重复
            max_repetition_penalty=1.3,       # 上限也较低
            scene_type=SceneType.INDUSTRIAL_CONTROL,
            enable_adaptive_layer=True
        )
    
    @staticmethod
    def get_medical_analysis_config() -> CyclePreventionConfig:
        """医疗分析模块配置"""
        return CyclePreventionConfig(
            history_buffer_size=12,           # 需要医疗历史上下文
            repeat_threshold=3,               # 标准重复检测
            base_temperature=0.6,             # 中等温度平衡准确性和创造性
            max_temperature=1.0,              # 限制随机性
            base_repetition_penalty=1.15,     # 中等惩罚
            max_repetition_penalty=1.6,       # 中等上限
            scene_type=SceneType.MEDICAL_ANALYSIS,
            enable_adaptive_layer=True
        )
    
    @staticmethod
    def get_general_conversation_config() -> CyclePreventionConfig:
        """通用对话模块配置（默认）"""
        return CyclePreventionConfig(
            history_buffer_size=10,
            repeat_threshold=3,
            base_temperature=0.7,
            max_temperature=1.0,
            base_repetition_penalty=1.2,
            max_repetition_penalty=1.5,
            scene_type=SceneType.GENERAL_CONVERSATION,
            enable_adaptive_layer=True
        )
    
    @staticmethod
    def get_code_generation_config() -> CyclePreventionConfig:
        """代码生成模块配置"""
        return CyclePreventionConfig(
            history_buffer_size=20,           # 代码需要更长上下文
            repeat_threshold=2,               # 代码不允许重复
            base_temperature=0.5,             # 低温度确保代码准确性
            max_temperature=0.9,              # 限制随机性
            base_repetition_penalty=1.25,     # 较高惩罚防止重复代码
            max_repetition_penalty=2.0,       # 很高上限
            scene_type=SceneType.CODE_GENERATION,
            enable_adaptive_layer=True
        )
    
    @staticmethod
    def get_external_api_config() -> CyclePreventionConfig:
        """外部API模块配置（简化版）"""
        return CyclePreventionConfig(
            history_buffer_size=5,            # 外部API上下文短
            repeat_threshold=2,               # 快速检测重复
            base_temperature=0.7,
            max_temperature=1.2,
            base_repetition_penalty=1.1,
            max_repetition_penalty=1.8,
            scene_type=SceneType.GENERAL_CONVERSATION,
            enable_adaptive_layer=False       # 外部API不使用高级层（延迟问题）
        )
    
    @classmethod
    def get_config_for_module(cls, module_name: str) -> CyclePreventionConfig:
        """根据模块名获取配置"""
        # 模块映射表（在方法内部定义，避免lambda中的cls引用问题）
        MODULE_CONFIG_MAP = {
            # 模块类名 -> 配置工厂方法
            "CreativeWritingTool": cls.get_creative_writing_config,
            "AGILanguageModel": cls.get_general_conversation_config,
            "UnifiedLanguageModel": cls.get_general_conversation_config,
            "ExternalModelProxy": cls.get_external_api_config,
            "agi_text_processor": cls.get_general_conversation_config,
            "unified_language_model": cls.get_general_conversation_config,
            "external_model_proxy": cls.get_external_api_config,
        }
        
        if module_name in MODULE_CONFIG_MAP:
            return MODULE_CONFIG_MAP[module_name]()
        
        # 默认使用通用对话配置
        logger.info(f"模块 {module_name} 没有特定配置，使用通用对话配置")
        return cls.get_general_conversation_config()
    
    @classmethod
    def get_config_for_scene(cls, scene_type: SceneType) -> CyclePreventionConfig:
        """根据场景类型获取配置"""
        # 场景映射表（在方法内部定义，避免lambda中的cls引用问题）
        SCENE_CONFIG_MAP = {
            SceneType.CREATIVE_WRITING: cls.get_creative_writing_config,
            SceneType.INDUSTRIAL_CONTROL: cls.get_industrial_control_config,
            SceneType.MEDICAL_ANALYSIS: cls.get_medical_analysis_config,
            SceneType.GENERAL_CONVERSATION: cls.get_general_conversation_config,
            SceneType.CODE_GENERATION: cls.get_code_generation_config,
            SceneType.FINANCIAL_ANALYSIS: cls.get_medical_analysis_config,  # 复用医疗配置
            SceneType.SCIENTIFIC_RESEARCH: cls.get_medical_analysis_config,  # 复用医疗配置
            SceneType.EDUCATION: cls.get_general_conversation_config,
            SceneType.EMOTIONAL_SUPPORT: cls.get_creative_writing_config,    # 复用创意写作配置
            SceneType.TECHNICAL_SUPPORT: cls.get_code_generation_config,     # 复用代码生成配置
        }
        
        if scene_type in SCENE_CONFIG_MAP:
            return SCENE_CONFIG_MAP[scene_type]()
        
        # 默认使用通用对话配置
        logger.info(f"场景 {scene_type} 没有特定配置，使用通用对话配置")
        return cls.get_general_conversation_config()
    
    @classmethod
    def get_config(cls, 
                   module_name: Optional[str] = None,
                   scene_type: Optional[SceneType] = None) -> CyclePreventionConfig:
        """获取配置（优先级：scene_type > module_name > 默认）"""
        if scene_type is not None:
            return cls.get_config_for_scene(scene_type)
        elif module_name is not None:
            return cls.get_config_for_module(module_name)
        else:
            return cls.get_general_conversation_config()


class CyclePreventionConfigManager:
    """防循环配置管理器
    
    提供配置的加载、保存、验证和动态更新功能
    支持与项目配置管理器集成
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config_cache: Dict[str, CyclePreventionConfig] = {}
        
    def get_module_config(self, module_name: str, 
                          use_cache: bool = True) -> CyclePreventionConfig:
        """获取模块配置（带缓存）"""
        if use_cache and module_name in self.config_cache:
            return self.config_cache[module_name]
        
        config = CyclePreventionConfigFactory.get_config_for_module(module_name)
        
        if use_cache:
            self.config_cache[module_name] = config
        
        return config
    
    def update_config(self, module_name: str, 
                     config_updates: Dict[str, Any]) -> CyclePreventionConfig:
        """更新模块配置"""
        # 获取现有配置或默认配置
        current_config = self.get_module_config(module_name, use_cache=True)
        
        # 转换为字典，应用更新
        config_dict = current_config.to_dict()
        config_dict.update(config_updates)
        
        # 创建新配置
        new_config = CyclePreventionConfig.from_dict(config_dict)
        
        # 验证配置
        errors = new_config.validate()
        if errors:
            error_msg = "; ".join(errors)
            self.logger.error(f"配置验证失败: {error_msg}")
            raise ValueError(f"无效的配置更新: {error_msg}")
        
        # 更新缓存
        self.config_cache[module_name] = new_config
        
        self.logger.info(f"模块 {module_name} 配置已更新")
        return new_config
    
    def reset_config(self, module_name: str) -> CyclePreventionConfig:
        """重置模块配置为默认值"""
        if module_name in self.config_cache:
            del self.config_cache[module_name]
        
        return self.get_module_config(module_name, use_cache=True)
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取所有缓存配置"""
        return {
            module_name: config.to_dict()
            for module_name, config in self.config_cache.items()
        }


# 全局配置管理器实例
_global_config_manager: Optional[CyclePreventionConfigManager] = None

def get_global_config_manager() -> CyclePreventionConfigManager:
    """获取全局配置管理器实例（单例模式）"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = CyclePreventionConfigManager()
    return _global_config_manager

def get_module_config(module_name: str) -> CyclePreventionConfig:
    """便捷函数：获取模块配置"""
    return get_global_config_manager().get_module_config(module_name)