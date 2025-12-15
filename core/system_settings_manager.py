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
System Settings Manager: Responsible for reading, saving and providing system settings
"""
import json
import os
from typing import Dict, Any, List
import threading
from .error_handling import error_handler


"""
SystemSettingsManager Class
"""
class SystemSettingsManager:
    """System settings manager class responsible for managing persistent storage and access to all system settings"""
    
    _instance = None
    _lock = threading.Lock()
    
    # 默认设置
    DEFAULT_SETTINGS = {
        "models": {
            "manager": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "language": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "audio": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "image": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "video": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "spatial": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "sensor": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "computer": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "motion": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "knowledge": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "programming": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "vision_image": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "vision_video": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "emotion": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "planning": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "prediction": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "advanced_reasoning": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "data_fusion": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "creative_problem_solving": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "meta_cognition": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""},
            "value_alignment": {"type": "local", "api_url": "", "api_key": "", "model_name": "", "source": ""}
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
        """Initialize system settings manager"""
        # Create settings file directory
        self.settings_dir = os.path.join(os.path.dirname(__file__), 'data', 'settings')
        os.makedirs(self.settings_dir, exist_ok=True)
        
        # Settings file path
        self.settings_file = os.path.join(self.settings_dir, 'system_settings.json')
        
        # Load settings
        self.settings = self._load_settings()

    def _load_settings(self) -> Dict[str, Any]:
        """Load system settings from file"""
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
        """Merge default settings with custom settings"""
        result = default.copy()
        
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                result[key] = self._merge_settings(result[key], value)
            else:
                result[key] = value
        
        return result

    def _save_settings(self, settings: Dict[str, Any]) -> bool:
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            error_handler.handle_error(e, "SystemSettingsManager", "保存设置失败")
            return False

    def get_settings(self) -> Dict[str, Any]:
        """Get all system settings"""
        return self.settings.copy()

    def get_model_setting(self, model_id: str, setting_name: str = None, default=None) -> Any:
        """
        Get settings for a specific model
        :param model_id: Model ID
        :param setting_name: Setting name, if None returns all settings
        :param default: Default value
        :return: Setting value
        """
        if model_id not in self.settings.get("models", {}):
            return default
        
        model_settings = self.settings["models"][model_id]
        if setting_name is None:
            return model_settings.copy()
        
        return model_settings.get(setting_name, default)

    def get_system_setting(self, setting_name: str = None, default=None) -> Any:
        """
        Get system settings
        :param setting_name: Setting name, if None returns all system settings
        :param default: Default value
        :return: Setting value
        """
        if setting_name is None:
            return self.settings.get("system", {}).copy()
        
        return self.settings.get("system", {}).get(setting_name, default)

    def update_model_setting(self, model_id: str, settings: Dict[str, Any]) -> bool:
        """
        Update settings for a specific model
        :param model_id: Model ID
        :param settings: Settings to update
        :return: Whether update was successful
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
        Update system settings
        :param settings: System settings to update
        :return: Whether update was successful
        """
        with self._lock:
            self.settings["system"].update(settings)
            return self._save_settings(self.settings)
    
    def reset_settings(self) -> bool:
        """Reset all settings to default values"""
        with self._lock:
            self.settings = self.DEFAULT_SETTINGS.copy()
            return self._save_settings(self.settings)
    
    def get_model_type(self, model_id: str) -> str:
        """
        Get model type (local or API)
        :param model_id: Model ID
        :return: 'local' or 'api'
        """
        return self.get_model_setting(model_id, "type", "local")
    
    def get_models_config(self) -> Dict[str, Any]:
        """
        Get configurations for all models
        :return: All models configurations
        """
        return self.settings.get("models", {}).copy()
    

    
    def get_all_model_types(self) -> Dict[str, str]:
        """
        Get configuration of all model types
        :return: {model_id: type}
        """
        result = {}
        for model_id, settings in self.settings.get("models", {}).items():
            result[model_id] = settings.get("type", "local")
        return result
        
    def set_model_type(self, model_id: str, model_type: str) -> bool:
        """
        Set model type (local or api)
        :param model_id: Model ID
        :param model_type: 'local' or 'api'
        :return: Whether update was successful
        """
        return self.update_model_setting(model_id, {"type": model_type})
        
    def set_model_api_config(self, model_id: str, api_url: str, api_key: str, model_name: str = "", source: str = "") -> bool:
        """
        Set model API configuration
        :param model_id: Model ID
        :param api_url: API URL
        :param api_key: API Key
        :param model_name: Model name
        :param source: Source provider
        :return: Whether update was successful
        """
        settings = {
            "api_url": api_url,
            "api_key": api_key,
            "model_name": model_name,
            "source": source
        }
        return self.update_model_setting(model_id, settings)
    
    def save_model_config(self, model_id: str, config: Dict[str, Any]) -> bool:
        """
        Save model configuration
        :param model_id: Model ID
        :param config: Model configuration
        :return: Whether save was successful
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
        Get model configuration
        :param model_id: Model ID
        :return: Model configuration
        """
        # 返回模型配置或空字典
        return self.settings.get("models", {}).get(model_id, {}).copy()
    
    def get_model_api_config(self, model_id: str) -> Dict[str, str]:
        """
        Get model's API configuration
        :param model_id: Model ID
        :return: API configuration with all necessary fields
        """
        model_settings = self.get_model_setting(model_id, default={})
        return {
            "api_url": model_settings.get("api_url", ""),
            "api_key": model_settings.get("api_key", ""),
            "model_name": model_settings.get("model_name", ""),
            "source": model_settings.get("source", "local"),
            "endpoint": model_settings.get("endpoint", "")
        }
    
    def is_api_model(self, model_id: str) -> bool:
        """
        Check if model is configured in API mode
        :param model_id: Model ID
        :return: Whether it's in API mode
        """
        model_config = self.get_model_config(model_id)
        return model_config.get("source") == "external" or model_config.get("type") == "api"
    
    def save_settings(self, settings_data: Dict[str, Any]) -> bool:
        """
        Save system settings
        :param settings_data: Settings data to save
        :return: Whether save was successful
        """
        with self._lock:
            # 合并新设置到现有设置
            self.settings = self._merge_settings(self.settings, settings_data)
            return self._save_settings(self.settings)

    def load_settings(self) -> Dict[str, Any]:
        """
        Load system settings
        :return: Loaded settings dictionary
        """
        self.settings = self._load_settings()
        return self.settings

# 创建全局实例
system_settings_manager = SystemSettingsManager()
