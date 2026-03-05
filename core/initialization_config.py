"""
初始化优化配置管理器
========================

专门用于管理模型初始化优化的配置系统。
支持从YAML配置文件中读取配置，并提供默认值和环境覆盖。
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field

# 导入错误处理
try:
    from core.error_handling import error_handler
except ImportError:
    error_handler = None


class InitializationStrategy(Enum):
    """初始化策略枚举"""
    STANDARD = "standard"      # 标准初始化（无优化）
    OPTIMIZED = "optimized"    # 优化初始化（默认）
    AGGRESSIVE = "aggressive"  # 激进优化（最大性能，可能牺牲稳定性）


class ComponentPriority(Enum):
    """组件优先级枚举"""
    CRITICAL = "critical"      # 关键组件，必须立即初始化
    ESSENTIAL = "essential"    # 重要组件，应尽快初始化
    IMPORTANT = "important"    # 重要组件，可稍后初始化
    BACKGROUND = "background"  # 后台组件，可延迟初始化
    OPTIONAL = "optional"      # 可选组件，按需初始化


@dataclass
class ComponentPolicy:
    """组件初始化策略"""
    name: str
    lazy: bool = True
    priority: ComponentPriority = ComponentPriority.IMPORTANT
    timeout_seconds: float = 5.0
    description: str = ""


@dataclass
class ModelSpecificOptimization:
    """模型特定优化配置"""
    model_name: str
    background_loading: bool = True
    lazy_imports: bool = True
    timeout_seconds: float = 30.0
    compression: str = "auto"  # auto, none, quantized
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WarmUpConfig:
    """预热配置"""
    enabled: bool = True
    pre_warm_models: List[Dict[str, Any]] = field(default_factory=list)
    warm_up_timeout_seconds: float = 30.0
    max_concurrent_warm_up: int = 2


@dataclass
class MonitoringConfig:
    """监控配置"""
    enabled: bool = True
    log_slow_initializations: bool = True
    slow_threshold_ms: float = 1000.0
    generate_reports: bool = True
    report_path: str = "logs/initialization_reports"
    retention_days: int = 30


@dataclass
class EnvironmentOverrides:
    """环境特定覆盖配置"""
    development: Dict[str, Any] = field(default_factory=dict)
    testing: Dict[str, Any] = field(default_factory=dict)
    production: Dict[str, Any] = field(default_factory=dict)


class InitializationConfig:
    """
    初始化优化配置管理器
    
    提供模型初始化优化的配置管理，支持：
    1. 从YAML配置文件加载配置
    2. 提供合理的默认值
    3. 环境特定的覆盖配置
    4. 配置验证和错误处理
    """
    
    # 默认配置
    DEFAULT_CONFIG = {
        "global": {
            "enabled": True,
            "default_strategy": "optimized",
            "enable_lazy_loading": True,
            "enable_parallel_init": True,
            "max_parallel_workers": 4,
            "enable_background_warmup": True,
            "enable_profiling": True
        },
        "component_policies": {
            "data_processor": {
                "lazy": False,
                "priority": "critical",
                "timeout_seconds": 1.0
            },
            "stream_manager": {
                "lazy": True,
                "priority": "essential",
                "timeout_seconds": 2.0
            },
            "external_api_service": {
                "lazy": True,
                "priority": "important",
                "timeout_seconds": 5.0
            },
            "multi_modal_processor": {
                "lazy": True,
                "priority": "important",
                "timeout_seconds": 3.0
            },
            "context_memory": {
                "lazy": True,
                "priority": "important",
                "timeout_seconds": 2.0
            },
            "agi_systems": {
                "lazy": True,
                "priority": "background",
                "timeout_seconds": 10.0
            },
            "from_scratch_trainer": {
                "lazy": True,
                "priority": "optional",
                "timeout_seconds": 15.0
            }
        },
        "model_specific_optimizations": {
            "language_model": {
                "bert_loading": {
                    "background": True,
                    "prefetch": False,
                    "compression": "auto",
                    "timeout_seconds": 30
                }
            },
            "vision_model": {
                "cv_libraries": {
                    "lazy_import": True,
                    "minimal_import": True,
                    "timeout_seconds": 10
                }
            },
            "audio_model": {
                "audio_libraries": {
                    "lazy_import": True,
                    "fallback_on_error": True,
                    "timeout_seconds": 5
                }
            }
        },
        "warm_up": {
            "enabled": True,
            "pre_warm_models": [
                {
                    "model_class": "core.models.language.unified_language_model.UnifiedLanguageModel",
                    "model_id": "default_language_model",
                    "config": {
                        "from_scratch": False,
                        "test_mode": False
                    },
                    "priority": "high"
                },
                {
                    "model_class": "core.models.vision.unified_vision_model.UnifiedVisionModel",
                    "model_id": "default_vision_model",
                    "config": {
                        "test_mode": False
                    },
                    "priority": "medium"
                },
                {
                    "model_class": "core.models.audio.unified_audio_model.UnifiedAudioModel",
                    "model_id": "default_audio_model",
                    "config": {
                        "test_mode": False
                    },
                    "priority": "medium"
                }
            ],
            "warm_up_timeout_seconds": 30,
            "max_concurrent_warm_up": 2
        },
        "monitoring": {
            "enabled": True,
            "log_slow_initializations": True,
            "slow_threshold_ms": 1000,
            "generate_reports": True,
            "report_path": "logs/initialization_reports",
            "retention_days": 30
        },
        "environment_overrides": {
            "development": {
                "enable_lazy_loading": True,
                "enable_profiling": True,
                "max_parallel_workers": 2
            },
            "testing": {
                "enable_lazy_loading": False,
                "enable_parallel_init": False,
                "enable_profiling": False
            },
            "production": {
                "enable_lazy_loading": True,
                "enable_parallel_init": True,
                "max_parallel_workers": 8,
                "enable_background_warmup": True,
                "enable_profiling": False
            }
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        
        # 加载配置
        self.config = self._load_config()
        
        # 应用环境覆盖
        self._apply_environment_overrides()
        
        # 验证配置
        self._validate_config()
        
        self.logger.info("InitializationConfig initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        config = self.DEFAULT_CONFIG.copy()
        
        # 如果提供了配置文件路径，尝试加载
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                
                # 合并配置
                if file_config and isinstance(file_config, dict):
                    # 递归合并配置
                    config = self._merge_configs(config, file_config)
                    self.logger.info(f"Loaded initialization config from {self.config_path}")
                else:
                    self.logger.warning(f"Config file {self.config_path} is empty or invalid, using defaults")
                    
            except Exception as e:
                self.logger.error(f"Failed to load config file {self.config_path}: {e}")
                if error_handler:
                    error_handler.log_error(
                        "config_load_error",
                        f"Failed to load initialization config: {e}",
                        component="InitializationConfig"
                    )
        
        # 应用环境变量覆盖
        config = self._apply_env_vars(config)
        
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """递归合并两个配置字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # 递归合并字典
                result[key] = self._merge_configs(result[key], value)
            else:
                # 直接覆盖
                result[key] = value
        
        return result
    
    def _apply_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量覆盖"""
        env_vars = {
            "SS_OPTIMIZATION_ENABLED": ("global", "enabled", bool),
            "SS_OPTIMIZATION_STRATEGY": ("global", "default_strategy", str),
            "SS_LAZY_LOADING": ("global", "enable_lazy_loading", bool),
            "SS_PARALLEL_INIT": ("global", "enable_parallel_init", bool),
            "SS_MAX_WORKERS": ("global", "max_parallel_workers", int),
            "SS_PROFILING": ("global", "enable_profiling", bool),
        }
        
        for env_var, (section, key, type_converter) in env_vars.items():
            if env_var in os.environ:
                try:
                    value = os.environ[env_var]
                    
                    # 类型转换
                    if type_converter == bool:
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    elif type_converter == int:
                        value = int(value)
                    
                    # 设置值
                    if section in config and isinstance(config[section], dict):
                        config[section][key] = value
                        self.logger.debug(f"Applied env var {env_var}={value} to {section}.{key}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to parse env var {env_var}: {e}")
        
        return config
    
    def _apply_environment_overrides(self):
        """应用环境特定的覆盖配置"""
        # 检测当前环境
        env = self._detect_environment()
        
        # 获取环境覆盖配置
        env_overrides = self.config.get("environment_overrides", {}).get(env, {})
        
        if env_overrides:
            # 应用覆盖到全局配置
            if "global" in self.config:
                for key, value in env_overrides.items():
                    if key in self.config["global"]:
                        old_value = self.config["global"][key]
                        self.config["global"][key] = value
                        self.logger.debug(f"Applied {env} override: {key}={value} (was: {old_value})")
        
        self.logger.info(f"Applied environment overrides for {env}")
    
    def _detect_environment(self) -> str:
        """检测当前环境"""
        # 检查环境变量
        env_var = os.environ.get("SS_ENVIRONMENT", "").lower()
        
        if env_var in ["development", "dev"]:
            return "development"
        elif env_var in ["testing", "test"]:
            return "testing"
        elif env_var in ["production", "prod"]:
            return "production"
        
        # 基于其他指标判断
        import sys
        
        # 检查是否在测试模式
        if "pytest" in sys.modules or "unittest" in sys.modules:
            return "testing"
        
        # 检查调试标志
        if hasattr(sys, 'gettrace') and sys.gettrace() is not None:
            return "development"
        
        # 默认为开发环境
        return "development"
    
    def _validate_config(self):
        """验证配置"""
        errors = []
        warnings = []
        
        # 验证全局配置
        global_config = self.config.get("global", {})
        
        # 检查必需字段
        required_fields = ["enabled", "default_strategy", "enable_lazy_loading"]
        for field in required_fields:
            if field not in global_config:
                errors.append(f"Missing required field: global.{field}")
        
        # 验证策略
        if "default_strategy" in global_config:
            strategy = global_config["default_strategy"]
            valid_strategies = [s.value for s in InitializationStrategy]
            if strategy not in valid_strategies:
                errors.append(f"Invalid strategy: {strategy}. Valid values: {valid_strategies}")
        
        # 验证组件策略
        component_policies = self.config.get("component_policies", {})
        for comp_name, policy in component_policies.items():
            if not isinstance(policy, dict):
                warnings.append(f"Invalid component policy for {comp_name}: expected dict")
                continue
            
            # 验证优先级
            if "priority" in policy:
                priority = policy["priority"]
                valid_priorities = [p.value for p in ComponentPriority]
                if priority not in valid_priorities:
                    warnings.append(f"Invalid priority for {comp_name}: {priority}. Valid: {valid_priorities}")
            
            # 验证超时
            if "timeout_seconds" in policy:
                timeout = policy["timeout_seconds"]
                if not isinstance(timeout, (int, float)) or timeout <= 0:
                    warnings.append(f"Invalid timeout for {comp_name}: {timeout}")
        
        # 记录验证结果
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(errors)
            self.logger.error(error_msg)
            if error_handler:
                error_handler.log_error(
                    "config_validation_error",
                    error_msg,
                    component="InitializationConfig"
                )
        
        if warnings:
            warning_msg = "Configuration validation warnings:\n" + "\n".join(warnings)
            self.logger.warning(warning_msg)
    
    def is_enabled(self) -> bool:
        """检查优化是否启用"""
        return self.config.get("global", {}).get("enabled", True)
    
    def get_strategy(self) -> InitializationStrategy:
        """获取初始化策略"""
        strategy_str = self.config.get("global", {}).get("default_strategy", "optimized")
        
        try:
            return InitializationStrategy(strategy_str)
        except ValueError:
            self.logger.warning(f"Invalid strategy {strategy_str}, falling back to OPTIMIZED")
            return InitializationStrategy.OPTIMIZED
    
    def get_component_policy(self, component_name: str) -> ComponentPolicy:
        """获取组件初始化策略"""
        policies = self.config.get("component_policies", {})
        
        if component_name in policies:
            policy_data = policies[component_name]
            
            # 解析优先级
            priority_str = policy_data.get("priority", "important")
            try:
                priority = ComponentPriority(priority_str)
            except ValueError:
                self.logger.warning(f"Invalid priority {priority_str} for {component_name}, using IMPORTANT")
                priority = ComponentPriority.IMPORTANT
            
            return ComponentPolicy(
                name=component_name,
                lazy=policy_data.get("lazy", True),
                priority=priority,
                timeout_seconds=policy_data.get("timeout_seconds", 5.0),
                description=f"Policy for {component_name}"
            )
        
        # 默认策略
        return ComponentPolicy(
            name=component_name,
            lazy=True,
            priority=ComponentPriority.IMPORTANT,
            timeout_seconds=5.0,
            description=f"Default policy for {component_name}"
        )
    
    def get_model_optimization(self, model_name: str) -> ModelSpecificOptimization:
        """获取模型特定优化配置"""
        optimizations = self.config.get("model_specific_optimizations", {})
        
        if model_name in optimizations:
            opt_data = optimizations[model_name]
            
            # 提取通用设置
            background_loading = opt_data.get("background_loading", True)
            lazy_imports = opt_data.get("lazy_imports", True)
            timeout_seconds = opt_data.get("timeout_seconds", 30.0)
            compression = opt_data.get("compression", "auto")
            
            # 特定配置
            custom_config = {}
            for key, value in opt_data.items():
                if key not in ["background_loading", "lazy_imports", "timeout_seconds", "compression"]:
                    custom_config[key] = value
            
            return ModelSpecificOptimization(
                model_name=model_name,
                background_loading=background_loading,
                lazy_imports=lazy_imports,
                timeout_seconds=timeout_seconds,
                compression=compression,
                custom_config=custom_config
            )
        
        # 默认配置
        return ModelSpecificOptimization(
            model_name=model_name,
            background_loading=True,
            lazy_imports=True,
            timeout_seconds=30.0,
            compression="auto"
        )
    
    def get_warm_up_config(self) -> WarmUpConfig:
        """获取预热配置"""
        warm_up_data = self.config.get("warm_up", {})
        
        return WarmUpConfig(
            enabled=warm_up_data.get("enabled", True),
            pre_warm_models=warm_up_data.get("pre_warm_models", []),
            warm_up_timeout_seconds=warm_up_data.get("warm_up_timeout_seconds", 30.0),
            max_concurrent_warm_up=warm_up_data.get("max_concurrent_warm_up", 2)
        )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """获取监控配置"""
        monitoring_data = self.config.get("monitoring", {})
        
        return MonitoringConfig(
            enabled=monitoring_data.get("enabled", True),
            log_slow_initializations=monitoring_data.get("log_slow_initializations", True),
            slow_threshold_ms=monitoring_data.get("slow_threshold_ms", 1000.0),
            generate_reports=monitoring_data.get("generate_reports", True),
            report_path=monitoring_data.get("report_path", "logs/initialization_reports"),
            retention_days=monitoring_data.get("retention_days", 30)
        )
    
    def get_global_config(self) -> Dict[str, Any]:
        """获取全局配置"""
        return self.config.get("global", {}).copy()
    
    def get_full_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.config.copy()
    
    def should_use_optimized_template(self) -> bool:
        """是否应该使用优化模板"""
        if not self.is_enabled():
            return False
        
        strategy = self.get_strategy()
        return strategy in [InitializationStrategy.OPTIMIZED, InitializationStrategy.AGGRESSIVE]
    
    def should_enable_lazy_loading(self) -> bool:
        """是否启用懒加载"""
        return self.config.get("global", {}).get("enable_lazy_loading", True)
    
    def should_enable_parallel_init(self) -> bool:
        """是否启用并行初始化"""
        return self.config.get("global", {}).get("enable_parallel_init", True)
    
    def get_max_parallel_workers(self) -> int:
        """获取最大并行工作线程数"""
        return self.config.get("global", {}).get("max_parallel_workers", 4)
    
    def should_enable_profiling(self) -> bool:
        """是否启用性能分析"""
        return self.config.get("global", {}).get("enable_profiling", True)


# 全局配置实例
_global_config: Optional[InitializationConfig] = None

def get_config(config_path: Optional[str] = None) -> InitializationConfig:
    """获取全局配置实例"""
    global _global_config
    
    if _global_config is None:
        # 如果没有提供路径，尝试从默认位置加载
        if config_path is None:
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config",
                "performance.yml"
            )
            config_path = default_path if os.path.exists(default_path) else None
        
        _global_config = InitializationConfig(config_path)
    
    return _global_config

def reset_config():
    """重置全局配置（主要用于测试）"""
    global _global_config
    _global_config = None