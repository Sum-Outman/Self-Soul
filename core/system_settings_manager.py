"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

"""
基础模型类 - 所有模型的基类
Base Model Class - Base class for all models

提供通用接口和功能，确保所有模型的一致性
Provides common interfaces and functionality to ensure consistency across all models
"""
"""
系统设置管理器：负责读取、保存和提供系统设置
"""
import json
import os
from typing import Dict, Any, List
import threading
from .error_handling import error_handler


"""
SystemSettingsManager类 - 中文类描述
SystemSettingsManager Class - English class description
"""
class SystemSettingsManager:
    """系统设置管理器类，负责管理所有系统设置的持久化存储和访问"""
    
    _instance = None
    _lock = threading.Lock()
    
    # 默认设置
    DEFAULT_SETTINGS = {
        "models": {
            "manager": {"type": "local", "api_url": "", "api_key": ""},
            "language": {"type": "local", "api_url": "", "api_key": ""},
            "audio": {"type": "local", "api_url": "", "api_key": ""},
            "image": {"type": "local", "api_url": "", "api_key": ""},
            "video": {"type": "local", "api_url": "", "api_key": ""},
            "spatial": {"type": "local", "api_url": "", "api_key": ""},
            "sensor": {"type": "local", "api_url": "", "api_key": ""},
            "computer": {"type": "local", "api_url": "", "api_key": ""},
            "motion": {"type": "local", "api_url": "", "api_key": ""},
            "knowledge": {"type": "local", "api_url": "", "api_key": ""},
            "programming": {"type": "local", "api_url": "", "api_key": ""}
        },
        "system": {
            "data_retention": 30,
            "auto_update": True,
            "log_level": "info",
            "max_memory_usage": 80,
            "max_cpu_usage": 90
        }
    }
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SystemSettingsManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """初始化系统设置管理器"""
        # 创建设置文件目录
        self.settings_dir = os.path.join(os.path.dirname(__file__), 'data', 'settings')
        os.makedirs(self.settings_dir, exist_ok=True)
        
        # 设置文件路径
        self.settings_file = os.path.join(self.settings_dir, 'system_settings.json')
        
        # 加载设置
        self.settings = self._load_settings()

    def _load_settings(self) -> Dict[str, Any]:
        """从文件加载系统设置"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    # 合并默认设置和加载的设置
                    return self._merge_settings(self.DEFAULT_SETTINGS, loaded_settings)
            else:
                # 如果文件不存在，返回默认设置并保存
                self._save_settings(self.DEFAULT_SETTINGS)
                return self.DEFAULT_SETTINGS
        except Exception as e:
            error_handler.handle_error(e, "SystemSettingsManager", "加载设置失败")
            return self.DEFAULT_SETTINGS

    def _merge_settings(self, default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
        """合并默认设置和自定义设置"""
        result = default.copy()
        
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                result[key] = self._merge_settings(result[key], value)
            else:
                result[key] = value
        
        return result

    def _save_settings(self, settings: Dict[str, Any]) -> bool:
        """将设置保存到文件"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            error_handler.handle_error(e, "SystemSettingsManager", "保存设置失败")
            return False

    def get_settings(self) -> Dict[str, Any]:
        """获取所有系统设置"""
        return self.settings.copy()

    def get_model_setting(self, model_id: str, setting_name: str = None, default=None) -> Any:
        """
        获取特定模型的设置
        :param model_id: 模型ID
        :param setting_name: 设置名称，如果为None则返回所有设置
        :param default: 默认值
        :return: 设置值
        """
        if model_id not in self.settings.get("models", {}):
            return default
        
        model_settings = self.settings["models"][model_id]
        if setting_name is None:
            return model_settings.copy()
        
        return model_settings.get(setting_name, default)

    def get_system_setting(self, setting_name: str = None, default=None) -> Any:
        """
        获取系统设置
        :param setting_name: 设置名称，如果为None则返回所有系统设置
        :param default: 默认值
        :return: 设置值
        """
        if setting_name is None:
            return self.settings.get("system", {}).copy()
        
        return self.settings.get("system", {}).get(setting_name, default)

    def update_model_setting(self, model_id: str, settings: Dict[str, Any]) -> bool:
        """
        更新特定模型的设置
        :param model_id: 模型ID
        :param settings: 要更新的设置
        :return: 是否更新成功
        """
        with self._lock:
            if model_id not in self.settings["models"]:
                # 创建新的模型设置
                self.settings["models"][model_id] = settings
            else:
                # 更新现有设置
                self.settings["models"][model_id].update(settings)
            
            return self._save_settings(self.settings)

    def update_system_setting(self, settings: Dict[str, Any]) -> bool:
        """
        更新系统设置
        :param settings: 要更新的系统设置
        :return: 是否更新成功
        """
        with self._lock:
            self.settings["system"].update(settings)
            return self._save_settings(self.settings)
    
    def reset_settings(self) -> bool:
        """重置所有设置为默认值"""
        with self._lock:
            self.settings = self.DEFAULT_SETTINGS.copy()
            return self._save_settings(self.settings)
    
    def get_model_type(self, model_id: str) -> str:
        """
        获取模型类型（本地或API）
        :param model_id: 模型ID
        :return: 'local' 或 'api'
        """
        return self.get_model_setting(model_id, "type", "local")
    
    def is_api_model(self, model_id: str) -> bool:
        """
        检查模型是否配置为API模式
        :param model_id: 模型ID
        :return: 是否为API模式
        """
        return self.get_model_type(model_id) == "api"
    
    def get_model_api_config(self, model_id: str) -> Dict[str, str]:
        """
        获取模型的API配置
        :param model_id: 模型ID
        :return: API配置（url和key）
        """
        model_settings = self.get_model_setting(model_id, default={})
        return {
            "api_url": model_settings.get("api_url", ""),
            "api_key": model_settings.get("api_key", "")
        }
    
    def get_all_model_types(self) -> Dict[str, str]:
        """
        获取所有模型的类型配置
        :return: {model_id: type}
        """
        result = {}
        for model_id, settings in self.settings.get("models", {}).items():
            result[model_id] = settings.get("type", "local")
        return result
    
    def save_model_config(self, model_id: str, config: Dict[str, Any]) -> bool:
        """
        保存模型配置
        :param model_id: 模型ID
        :param config: 模型配置
        :return: 是否保存成功
        """
        with self._lock:
            # 确保models字典存在
            if "models" not in self.settings:
                self.settings["models"] = {}
            
            # 更新或创建模型配置
            if model_id not in self.settings["models"]:
                self.settings["models"][model_id] = config
            else:
                self.settings["models"][model_id].update(config)
            
            # 保存设置
            return self._save_settings(self.settings)
    
    def get_model_config(self, model_id: str) -> Dict[str, Any]:
        """
        获取模型配置
        :param model_id: 模型ID
        :return: 模型配置
        """
        # 返回模型配置或空字典
        return self.settings.get("models", {}).get(model_id, {}).copy()
    
    def get_model_api_config(self, model_id: str) -> Dict[str, str]:
        """
        获取模型的API配置
        :param model_id: 模型ID
        :return: API配置（url和key）
        """
        model_settings = self.get_model_setting(model_id, default={})
        return {
            "api_url": model_settings.get("api_url", ""),
            "api_key": model_settings.get("api_key", ""),
            "model_name": model_settings.get("model_name", ""),
            "source": model_settings.get("source", "local")
        }
    
    def is_api_model(self, model_id: str) -> bool:
        """
        检查模型是否配置为API模式
        :param model_id: 模型ID
        :return: 是否为API模式
        """
        model_config = self.get_model_config(model_id)
        return model_config.get("source") == "external" or model_config.get("type") == "api"
    
    def save_settings(self, settings_data: Dict[str, Any]) -> bool:
        """
        保存系统设置
        :param settings_data: 要保存的设置数据
        :return: 是否保存成功
        """
        with self._lock:
            # 合并新设置到现有设置
            self.settings = self._merge_settings(self.settings, settings_data)
            return self._save_settings(self.settings)

    def load_settings(self) -> Dict[str, Any]:
        """
        加载系统设置
        :return: 加载的设置字典
        """
        self.settings = self._load_settings()
        return self.settings

# 创建全局实例
system_settings_manager = SystemSettingsManager()
