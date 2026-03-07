"""
配置模块

提供系统配置管理和环境变量支持。
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config_data = {}
        self.defaults = self._get_default_config()
        
        # 加载配置
        self.load_config()
        
        logger.info("配置管理器已初始化")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'system': {
                'name': 'UnifiedCognitiveArchitecture',
                'version': '1.0.0',
                'environment': 'development',
                'log_level': 'INFO',
                'debug': False
            },
            'server': {
                'host': '0.0.0.0',
                'port': 9000,
                'workers': 1,
                'reload': True,
                'timeout': 30
            },
            'cognitive': {
                'embedding_dim': 1024,
                'max_shared_memory_mb': 1024,
                'enable_cache': True,
                'cache_size': 1000,
                'processing_mode': 'parallel'
            },
            'neural': {
                'communication_timeout': 5.0,
                'max_queue_size': 1000,
                'enable_monitoring': True,
                'monitoring_interval': 5.0
            },
            'security': {
                'max_input_length': 10000,
                'enable_validation': True,
                'enable_sanitization': True,
                'rate_limit_enabled': True,
                'rate_limit_requests': 100,
                'rate_limit_window': 60
            },
            'database': {
                'type': 'memory',  # memory, sqlite, postgresql
                'path': 'data/cognitive.db',
                'max_connections': 10
            },
            'monitoring': {
                'enable_metrics': True,
                'enable_tracing': True,
                'export_interval': 30,
                'metrics_port': 9090
            }
        }
    
    def load_config(self):
        """加载配置"""
        # 1. 从默认配置开始
        self.config_data = self.defaults.copy()
        
        # 2. 从配置文件加载（如果存在）
        if self.config_file and os.path.exists(self.config_file):
            self._load_from_file(self.config_file)
        
        # 3. 从环境变量加载
        self._load_from_env()
        
        # 4. 从默认配置文件加载
        self._load_default_files()
        
        logger.info(f"配置加载完成，环境: {self.get('system.environment')}")
    
    def _load_from_file(self, file_path: str):
        """从文件加载配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    file_config = json.load(f)
                elif file_path.endswith(('.yaml', '.yml')):
                    file_config = yaml.safe_load(f)
                else:
                    logger.warning(f"不支持的文件格式: {file_path}")
                    return
                
                self._merge_config(self.config_data, file_config)
                logger.info(f"从文件加载配置: {file_path}")
                
        except Exception as e:
            logger.error(f"加载配置文件失败 {file_path}: {e}")
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        env_prefix = 'AGI_'
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # 转换环境变量名为配置路径
                config_path = key[len(env_prefix):].lower().replace('__', '.')
                
                # 转换值类型
                typed_value = self._convert_env_value(value)
                
                # 设置配置值
                self._set_config_value(self.config_data, config_path, typed_value)
    
    def _load_default_files(self):
        """从默认配置文件加载"""
        # 检查当前目录和父目录的配置文件
        possible_paths = [
            'config.json',
            'config.yaml',
            'config.yml',
            '../config.json',
            '../config.yaml',
            '../config.yml',
            'config/default.json',
            'config/default.yaml',
            'config/default.yml'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self._load_from_file(path)
                break
    
    def _convert_env_value(self, value: str) -> Any:
        """转换环境变量值为适当类型"""
        # 尝试解析为JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # 检查布尔值
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False
        
        # 检查整数
        if value.isdigit():
            return int(value)
        
        # 检查浮点数
        try:
            return float(value)
        except ValueError:
            pass
        
        # 默认为字符串
        return value
    
    def _merge_config(self, base: Dict, update: Dict, path: str = ''):
        """递归合并配置"""
        for key, value in update.items():
            new_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                # 递归合并字典
                self._merge_config(base[key], value, new_path)
            else:
                # 设置新值
                base[key] = value
    
    def _set_config_value(self, config: Dict, path: str, value: Any):
        """设置配置值（支持点分隔路径）"""
        parts = path.split('.')
        current = config
        
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # 如果路径中间部分不是字典，则替换为字典
                current[part] = {}
            
            current = current[part]
        
        # 设置最终值
        current[parts[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        参数:
            key: 配置键，支持点分隔路径
            default: 默认值
        
        返回:
            配置值
        """
        parts = key.split('.')
        current = self.config_data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        参数:
            key: 配置键，支持点分隔路径
            value: 配置值
        """
        self._set_config_value(self.config_data, key, value)
    
    def save(self, file_path: Optional[str] = None):
        """保存配置到文件"""
        if file_path is None:
            file_path = self.config_file
        
        if file_path is None:
            logger.error("未指定配置文件路径")
            return
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    json.dump(self.config_data, f, indent=2, ensure_ascii=False)
                elif file_path.endswith(('.yaml', '.yml')):
                    yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)
                else:
                    logger.error(f"不支持的文件格式: {file_path}")
                    return
            
            logger.info(f"配置已保存到: {file_path}")
            
        except Exception as e:
            logger.error(f"保存配置文件失败 {file_path}: {e}")
    
    def reload(self):
        """重新加载配置"""
        logger.info("重新加载配置...")
        self.load_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """获取配置字典"""
        return self.config_data.copy()
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.get(key)


# 全局配置实例
_config_instance = None

def get_config(config_file: Optional[str] = None) -> Config:
    """
    获取全局配置实例
    
    参数:
        config_file: 配置文件路径
    
    返回:
        配置实例
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_file)
    
    return _config_instance