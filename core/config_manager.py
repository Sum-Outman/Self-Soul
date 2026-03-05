"""
统一配置管理器 - Unified Configuration Manager
提供集中式配置管理，支持配置验证、热重载和多源配置加载
"""

import abc
import os
import json
import logging
import threading
import time
import hashlib
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

# 条件导入可选依赖
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    error_handler = None  # 临时定义，将在后面导入

try:
    import watchdog.observers  # type: ignore
    import watchdog.events  # type: ignore
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

# 导入错误处理器（需要在条件导入后）
from core.error_handling import error_handler


# 配置类型枚举
class ConfigType(Enum):
    """配置类型枚举"""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"
    PYTHON = "python"


# 配置验证结果
@dataclass
class ValidationResult:
    """配置验证结果"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        """添加错误"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings
        }


# 配置变更事件
@dataclass
class ConfigChangeEvent:
    """配置变更事件"""
    timestamp: datetime
    config_source: str
    change_type: str  # 'added', 'modified', 'deleted', 'reloaded'
    config_key: Optional[str] = None
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    checksum: Optional[str] = None


class ConfigSchema:
    """配置模式定义类，用于配置验证"""
    
    def __init__(self, name: str, schema_def: Dict[str, Any]):
        """初始化配置模式
        
        Args:
            name: 模式名称
            schema_def: 模式定义字典
        """
        self.name = name
        self.schema_def = schema_def
        self.validators = self._build_validators()
    
    def _build_validators(self) -> Dict[str, Callable]:
        """构建验证器"""
        validators = {}
        
        # 根据模式定义构建验证器
        for key, rule in self.schema_def.items():
            if isinstance(rule, dict):
                if "type" in rule:
                    validators[key] = self._create_type_validator(key, rule)
                elif "schema" in rule:
                    # 嵌套模式
                    validators[key] = self._create_nested_validator(key, rule)
            elif callable(rule):
                validators[key] = rule
        
        return validators
    
    def _create_type_validator(self, key: str, rule: Dict[str, Any]) -> Callable:
        """创建类型验证器"""
        expected_type = rule["type"]
        required = rule.get("required", True)
        default = rule.get("default", None)
        min_value = rule.get("min", None)
        max_value = rule.get("max", None)
        choices = rule.get("choices", None)
        
        def validator(value: Any) -> Optional[str]:
            """验证器函数"""
            # 检查是否缺失但非必需
            if value is None:
                if not required:
                    return None
                return f"配置项 '{key}' 是必需的，但值为空"
            
            # 检查类型
            if expected_type == "int":
                if not isinstance(value, int):
                    return f"配置项 '{key}' 应为整数类型，实际为 {type(value).__name__}"
                if min_value is not None and value < min_value:
                    return f"配置项 '{key}' 值 {value} 小于最小值 {min_value}"
                if max_value is not None and value > max_value:
                    return f"配置项 '{key}' 值 {value} 大于最大值 {max_value}"
            
            elif expected_type == "float":
                if not isinstance(value, (int, float)):
                    return f"配置项 '{key}' 应为浮点数类型，实际为 {type(value).__name__}"
                if min_value is not None and value < min_value:
                    return f"配置项 '{key}' 值 {value} 小于最小值 {min_value}"
                if max_value is not None and value > max_value:
                    return f"配置项 '{key}' 值 {value} 大于最大值 {max_value}"
            
            elif expected_type == "str":
                if not isinstance(value, str):
                    return f"配置项 '{key}' 应为字符串类型，实际为 {type(value).__name__}"
                if choices is not None and value not in choices:
                    return f"配置项 '{key}' 值 '{value}' 不在允许的选项 {choices} 中"
            
            elif expected_type == "bool":
                if not isinstance(value, bool):
                    return f"配置项 '{key}' 应为布尔类型，实际为 {type(value).__name__}"
            
            elif expected_type == "list":
                if not isinstance(value, list):
                    return f"配置项 '{key}' 应为列表类型，实际为 {type(value).__name__}"
                if min_value is not None and len(value) < min_value:
                    return f"配置项 '{key}' 列表长度 {len(value)} 小于最小值 {min_value}"
                if max_value is not None and len(value) > max_value:
                    return f"配置项 '{key}' 列表长度 {len(value)} 大于最大值 {max_value}"
            
            elif expected_type == "dict":
                if not isinstance(value, dict):
                    return f"配置项 '{key}' 应为字典类型，实际为 {type(value).__name__}"
            
            return None
        
        return validator
    
    def _create_nested_validator(self, key: str, rule: Dict[str, Any]) -> Callable:
        """创建嵌套验证器"""
        nested_schema = ConfigSchema(f"{self.name}.{key}", rule["schema"])
        
        def validator(value: Any) -> Optional[str]:
            """验证器函数"""
            if value is None:
                if rule.get("required", True):
                    return f"配置项 '{key}' 是必需的，但值为空"
                return None
            
            if not isinstance(value, dict):
                return f"配置项 '{key}' 应为字典类型，实际为 {type(value).__name__}"
            
            # 验证嵌套配置
            result = nested_schema.validate(value)
            if not result.is_valid:
                return f"配置项 '{key}' 验证失败: {', '.join(result.errors)}"
            
            return None
        
        return validator
    
    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """验证配置"""
        result = ValidationResult(is_valid=True)
        
        # 检查必需字段
        for key, rule in self.schema_def.items():
            if isinstance(rule, dict):
                required = rule.get("required", True)
                if required and key not in config:
                    result.add_error(f"必需字段 '{key}' 缺失")
        
        # 验证每个字段
        for key, value in config.items():
            if key in self.validators:
                error = self.validators[key](value)
                if error:
                    result.add_error(error)
            else:
                # 未知字段，记录警告
                result.add_warning(f"未知配置字段 '{key}'")
        
        return result


class ConfigSource:
    """配置源基类"""
    
    def __init__(self, source_id: str, priority: int = 0):
        """初始化配置源
        
        Args:
            source_id: 源标识符
            priority: 优先级（数字越小优先级越高）
        """
        self.source_id = source_id
        self.priority = priority
        self.config = {}
        self.checksum = None
        self.last_loaded = None
    
    @abc.abstractmethod
    def load(self) -> bool:
        """加载配置，子类必须实现"""
        raise NotImplementedError
    
    def get_checksum(self) -> str:
        """计算配置校验和"""
        config_str = json.dumps(self.config, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()
    
    def has_changed(self) -> bool:
        """检查配置是否发生变化"""
        new_checksum = self.get_checksum()
        return new_checksum != self.checksum


class FileConfigSource(ConfigSource):
    """文件配置源"""
    
    def __init__(self, file_path: str, config_type: ConfigType, priority: int = 0):
        """初始化文件配置源
        
        Args:
            file_path: 文件路径
            config_type: 配置类型
            priority: 优先级
        """
        super().__init__(f"file:{file_path}", priority)
        self.file_path = Path(file_path)
        self.config_type = config_type
    
    def load(self) -> bool:
        """从文件加载配置"""
        try:
            if not self.file_path.exists():
                error_handler.log_warning(f"配置文件不存在: {self.file_path}", "ConfigManager")
                return False
            
            with open(self.file_path, 'r', encoding='utf-8') as f:
                if self.config_type == ConfigType.JSON:
                    self.config = json.load(f)
                elif self.config_type == ConfigType.YAML:
                    if YAML_AVAILABLE:
                        self.config = yaml.safe_load(f)
                    else:
                        error_handler.log_error(
                            f"无法加载YAML配置文件，PyYAML模块未安装: {self.file_path}", 
                            "ConfigManager"
                        )
                        return False
                else:
                    error_handler.log_error(f"不支持的配置文件类型: {self.config_type}", "ConfigManager")
                    return False
            
            self.checksum = self.get_checksum()
            self.last_loaded = datetime.now()
            error_handler.log_info(f"从文件加载配置: {self.file_path}", "ConfigManager")
            return True
            
        except Exception as e:
            error_handler.handle_error(e, "ConfigManager", f"加载配置文件失败: {self.file_path}")
            return False


class EnvConfigSource(ConfigSource):
    """环境变量配置源"""
    
    def __init__(self, prefix: str = "AGI_", priority: int = 10):
        """初始化环境变量配置源
        
        Args:
            prefix: 环境变量前缀
            priority: 优先级
        """
        super().__init__(f"env:{prefix}", priority)
        self.prefix = prefix
    
    def load(self) -> bool:
        """从环境变量加载配置"""
        try:
            self.config = {}
            
            # 收集所有以指定前缀开头的环境变量
            for key, value in os.environ.items():
                if key.startswith(self.prefix):
                    # 转换键名：移除前缀并转换为小写
                    config_key = key[len(self.prefix):].lower()
                    
                    # 尝试解析JSON值
                    try:
                        parsed_value = json.loads(value)
                        self.config[config_key] = parsed_value
                    except (json.JSONDecodeError, ValueError):
                        # 如果不是JSON，保持原始字符串值
                        self.config[config_key] = value
            
            self.checksum = self.get_checksum()
            self.last_loaded = datetime.now()
            error_handler.log_info(f"从环境变量加载配置，前缀: {self.prefix}", "ConfigManager")
            return True
            
        except Exception as e:
            error_handler.handle_error(e, "ConfigManager", "加载环境变量配置失败")
            return False


class UnifiedConfigManager:
    """统一配置管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    # 默认配置模式
    DEFAULT_SCHEMAS = {
        "model_ports": ConfigSchema("model_ports", {
            "manager": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "language": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "knowledge": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "vision": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "audio": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "autonomous": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "programming": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "planning": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "emotion": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "spatial": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "computer_vision": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "sensor": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "motion": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "prediction": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "advanced_reasoning": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "multi_model_collaboration": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "data_fusion": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "creative_problem_solving": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "meta_cognition": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "value_alignment": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "vision_image": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "vision_video": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "finance": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "medical": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "collaboration": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "optimization": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "computer": {"type": "int", "required": True, "min": 8001, "max": 8028},
            "mathematics": {"type": "int", "required": True, "min": 8001, "max": 8028}
        }),
        
        "main_api": ConfigSchema("main_api", {
            "port": {"type": "int", "required": True, "min": 8000, "max": 9000},
            "host": {"type": "str", "required": True, "default": "0.0.0.0"},
            "workers": {"type": "int", "required": True, "min": 1, "max": 10}
        }),
        
        "service_settings": ConfigSchema("service_settings", {
            "auto_start": {"type": "bool", "required": True, "default": True},
            "max_workers": {"type": "int", "required": True, "min": 1, "max": 50},
            "health_check_interval": {"type": "int", "required": True, "min": 5, "max": 300},
            "timeout": {"type": "int", "required": True, "min": 1, "max": 60},
            "retry_count": {"type": "int", "required": True, "min": 0, "max": 10}
        })
    }
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(UnifiedConfigManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """初始化配置管理器"""
        self.logger = logging.getLogger("ConfigManager")
        self.sources: List[ConfigSource] = []
        self.config: Dict[str, Any] = {}
        self.schemas: Dict[str, ConfigSchema] = copy.deepcopy(self.DEFAULT_SCHEMAS)
        self.change_listeners: List[Callable[[ConfigChangeEvent], None]] = []
        self.file_observer = None
        self.watch_thread = None
        self.watching = False
        self.config_history: List[ConfigChangeEvent] = []
        
        # 设置默认配置源
        self._setup_default_sources()
        
        # 加载初始配置
        self.reload_all()
        
        error_handler.log_info("统一配置管理器初始化完成", "ConfigManager")
    
    def _setup_default_sources(self):
        """设置默认配置源"""
        project_root = Path(__file__).parent.parent
        
        # 1. 模型服务配置文件（JSON）
        model_config_path = project_root / "config" / "model_services_config.json"
        if model_config_path.exists():
            self.add_source(FileConfigSource(str(model_config_path), ConfigType.JSON, priority=1))
        
        # 2. 性能配置文件（YAML）
        perf_config_path = project_root / "config" / "performance.yml"
        if perf_config_path.exists():
            self.add_source(FileConfigSource(str(perf_config_path), ConfigType.YAML, priority=2))
        
        # 3. 环境变量
        self.add_source(EnvConfigSource(prefix="AGI_", priority=100))
    
    def add_source(self, source: ConfigSource):
        """添加配置源"""
        self.sources.append(source)
        # 按优先级排序
        self.sources.sort(key=lambda s: s.priority)
    
    def add_schema(self, name: str, schema: ConfigSchema):
        """添加配置模式"""
        self.schemas[name] = schema
    
    def reload_all(self):
        """重新加载所有配置源"""
        self.logger.info("重新加载所有配置源")
        
        old_config = copy.deepcopy(self.config)
        new_config = {}
        
        # 按优先级顺序加载配置源
        for source in self.sources:
            if source.load():
                # 合并配置（高优先级覆盖低优先级）
                self._merge_configs(new_config, source.config)
        
        # 验证配置
        validation_result = self.validate_all(new_config)
        if not validation_result.is_valid:
            error_handler.log_error(f"配置验证失败: {validation_result.errors}", "ConfigManager")
            # 如果验证失败，回滚到旧配置
            self.config = old_config
            return False
        
        # 应用新配置
        self.config = new_config
        
        # 记录配置变更
        change_event = ConfigChangeEvent(
            timestamp=datetime.now(),
            config_source="all",
            change_type="reloaded",
            checksum=self.get_checksum()
        )
        self._notify_change(change_event)
        
        self.logger.info(f"配置重新加载完成，配置项总数: {self._count_config_items()}")
        return True
    
    def _merge_configs(self, base: Dict[str, Any], new: Dict[str, Any]):
        """合并配置字典"""
        for key, value in new.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # 递归合并字典
                self._merge_configs(base[key], value)
            else:
                # 覆盖或添加新值
                base[key] = value
    
    def validate_all(self, config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """验证所有配置"""
        config_to_validate = config if config is not None else self.config
        result = ValidationResult(is_valid=True)
        
        for section_name, section_config in config_to_validate.items():
            if section_name in self.schemas:
                schema_result = self.schemas[section_name].validate(section_config)
                if not schema_result.is_valid:
                    result.add_error(f"配置节 '{section_name}' 验证失败: {', '.join(schema_result.errors)}")
                result.warnings.extend(schema_result.warnings)
            else:
                result.add_warning(f"未知配置节 '{section_name}'，未验证")
        
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key_path: 配置键路径，使用点号分隔，例如 'model_ports.manager'
            default: 默认值
            
        Returns:
            配置值，如果不存在则返回默认值
        """
        keys = key_path.split('.')
        current = self.config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set(self, key_path: str, value: Any, save_to_source: bool = False) -> bool:
        """设置配置值
        
        Args:
            key_path: 配置键路径
            value: 配置值
            save_to_source: 是否保存到配置源
            
        Returns:
            是否成功
        """
        keys = key_path.split('.')
        current = self.config
        
        # 导航到目标位置，创建不存在的中间字典
        for i, key in enumerate(keys[:-1]):
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        # 记录旧值
        last_key = keys[-1]
        old_value = current.get(last_key) if last_key in current else None
        
        # 设置新值
        current[last_key] = value
        
        # 记录变更事件
        change_event = ConfigChangeEvent(
            timestamp=datetime.now(),
            config_source="internal",
            change_type="modified",
            config_key=key_path,
            old_value=old_value,
            new_value=value,
            checksum=self.get_checksum()
        )
        self._notify_change(change_event)
        
        # 如果需要保存到配置源
        if save_to_source:
            # 尝试保存到文件配置源
            saved = False
            for source in self.sources:
                if isinstance(source, FileConfigSource):
                    try:
                        # 构建要保存的配置
                        config_to_save = {}
                        
                        # 如果有特定的键路径，只保存该部分
                        if '.' in key_path:
                            keys = key_path.split('.')
                            current = config_to_save
                            for key in keys[:-1]:
                                current[key] = {}
                                current = current[key]
                            current[keys[-1]] = value
                        else:
                            config_to_save[key_path] = value
                        
                        # 保存到文件
                        self._save_config_to_file(source, config_to_save)
                        saved = True
                        self.logger.info(f"配置已保存到文件: {key_path} = {value}")
                        break
                    except Exception as e:
                        error_handler.handle_error(e, "ConfigManager", f"保存配置到文件失败: {key_path}")
            
            if not saved:
                self.logger.warning(f"未找到可保存的FileConfigSource，配置未持久化: {key_path}")
        
        return True
    
    def start_watching(self, interval_seconds: int = 5):
        """开始监视配置变化"""
        if self.watching:
            self.logger.warning("配置监视已经在运行")
            return
        
        self.watching = True
        self.watch_thread = threading.Thread(
            target=self._watch_loop,
            args=(interval_seconds,),
            daemon=True,
            name="ConfigWatcher"
        )
        self.watch_thread.start()
        self.logger.info(f"开始监视配置变化，检查间隔: {interval_seconds}秒")
    
    def stop_watching(self):
        """停止监视配置变化"""
        self.watching = False
        if self.watch_thread:
            self.watch_thread.join(timeout=5)
        self.logger.info("停止监视配置变化")
    
    def _watch_loop(self, interval_seconds: int):
        """监视循环"""
        while self.watching:
            try:
                # 检查每个配置源是否有变化
                for source in self.sources:
                    if isinstance(source, FileConfigSource) and source.has_changed():
                        self.logger.info(f"检测到配置源变化: {source.source_id}")
                        self.reload_all()
                        break
            except Exception as e:
                error_handler.handle_error(e, "ConfigManager", "配置监视循环出错")
            
            time.sleep(interval_seconds)
    
    def add_change_listener(self, listener: Callable[[ConfigChangeEvent], None]):
        """添加配置变更监听器"""
        self.change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable[[ConfigChangeEvent], None]):
        """移除配置变更监听器"""
        if listener in self.change_listeners:
            self.change_listeners.remove(listener)
    
    def _save_config_to_file(self, source: FileConfigSource, partial_config: Dict[str, Any] = None):
        """保存配置到文件配置源
        
        Args:
            source: 文件配置源
            partial_config: 部分配置（仅保存这部分配置）
        """
        try:
            # 确保文件目录存在
            source.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 确定要保存的配置
            if partial_config:
                # 加载现有配置
                existing_config = {}
                if source.file_path.exists():
                    with open(source.file_path, 'r', encoding='utf-8') as f:
                        if source.config_type == ConfigType.JSON:
                            existing_config = json.load(f)
                        elif source.config_type == ConfigType.YAML and YAML_AVAILABLE:
                            existing_config = yaml.safe_load(f)
                        else:
                            existing_config = {}
                
                # 合并部分配置到现有配置
                self._merge_partial_config(existing_config, partial_config)
                config_to_save = existing_config
            else:
                # 保存完整配置
                config_to_save = self.config
            
            # 保存到文件
            with open(source.file_path, 'w', encoding='utf-8') as f:
                if source.config_type == ConfigType.JSON:
                    json.dump(config_to_save, f, ensure_ascii=False, indent=2)
                elif source.config_type == ConfigType.YAML:
                    if YAML_AVAILABLE:
                        yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True)
                    else:
                        error_handler.log_error(
                            f"无法保存YAML配置文件，PyYAML模块未安装: {source.file_path}", 
                            "ConfigManager"
                        )
                        return False
                else:
                    error_handler.log_error(f"不支持的配置文件类型: {source.config_type}", "ConfigManager")
                    return False
            
            # 更新源的校验和
            source.checksum = source.get_checksum()
            source.last_loaded = datetime.now()
            
            self.logger.info(f"配置已保存到文件: {source.file_path}")
            return True
            
        except Exception as e:
            error_handler.handle_error(e, "ConfigManager", f"保存配置到文件失败: {source.file_path}")
            return False
    
    def _merge_partial_config(self, base: Dict[str, Any], partial: Dict[str, Any]):
        """合并部分配置到基础配置"""
        for key, value in partial.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                # 递归合并嵌套字典
                self._merge_partial_config(base[key], value)
            else:
                # 直接设置或覆盖
                base[key] = value
    
    def _notify_change(self, event: ConfigChangeEvent):
        """通知配置变更"""
        # 记录到历史
        self.config_history.append(event)
        # 限制历史记录大小
        if len(self.config_history) > 100:
            self.config_history = self.config_history[-100:]
        
        # 通知监听器
        for listener in self.change_listeners:
            try:
                listener(event)
            except Exception as e:
                error_handler.handle_error(e, "ConfigManager", f"配置变更监听器调用失败: {listener}")
    
    def get_checksum(self) -> str:
        """获取配置校验和"""
        config_str = json.dumps(self.config, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()
    
    def _count_config_items(self) -> int:
        """计算配置项数量"""
        def count_items(obj: Any) -> int:
            if isinstance(obj, dict):
                return sum(count_items(v) for v in obj.values())
            elif isinstance(obj, list):
                return sum(count_items(v) for v in obj)
            else:
                return 1
        
        return count_items(self.config)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            "total_sources": len(self.sources),
            "total_config_items": self._count_config_items(),
            "checksum": self.get_checksum(),
            "last_change": self.config_history[-1].timestamp if self.config_history else None,
            "change_count": len(self.config_history),
            "watching_enabled": self.watching
        }
    
    def save_to_file(self, file_path: str, config_type: ConfigType = ConfigType.JSON) -> bool:
        """保存配置到文件"""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if config_type == ConfigType.JSON:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                elif config_type == ConfigType.YAML:
                    if YAML_AVAILABLE:
                        yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                    else:
                        error_handler.log_error(
                            f"无法保存YAML配置文件，PyYAML模块未安装: {file_path}", 
                            "ConfigManager"
                        )
                        return False
                else:
                    error_handler.log_error(f"不支持的配置文件类型: {config_type}", "ConfigManager")
                    return False
            
            self.logger.info(f"配置已保存到文件: {file_path}")
            return True
            
        except Exception as e:
            error_handler.handle_error(e, "ConfigManager", f"保存配置到文件失败: {file_path}")
            return False


# 全局配置管理器实例
config_manager = UnifiedConfigManager()