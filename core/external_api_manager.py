"""
External API Manager - Advanced External API Integration

Provides intelligent API configuration management with AGI-enhanced capabilities:
- Dynamic API discovery and auto-configuration
- Real-time API performance monitoring and optimization
- Intelligent API selection based on task requirements
- Seamless model switching between local and external APIs
- Multi-API load balancing and failover
- Security and authentication management
- Real-time API health monitoring
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from core.api_config_manager import APIConfigManager
from core.external_api_service import ExternalAPIService
from core.error_handling import error_handler


class ExternalAPIManager:
    """外部API管理器类，用于统一管理所有外部API配置和连接"""
    
    def __init__(self, config_path: str = "config/external_api_configs.json"):
        """初始化外部API管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.api_config_manager = APIConfigManager(config_path)
        self.external_api_service = ExternalAPIService()
        self._load_configs()
    
    def _load_configs(self):
        """加载配置文件"""
        try:
            self.configs = self.api_config_manager.load_config()
        except Exception as e:
            error_handler.handle_error(e, "ExternalAPIManager", "加载配置文件失败")
            self.configs = {}
    
    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有外部API配置
        
        Returns:
            dict: 所有外部API配置的字典
        """
        try:
            return self.api_config_manager.load_config()
        except Exception as e:
            error_handler.handle_error(e, "ExternalAPIManager", "获取所有配置失败")
            return {}
    
    def add_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """添加外部API配置
        
        Args:
            config_data: 配置数据，包含name、api_type、api_key、api_url、model_name等字段
            
        Returns:
            dict: 添加结果，包含status和message字段
        """
        try:
            if not config_data.get("name"):
                return {"status": "error", "message": "配置名称不能为空"}
            
            if not config_data.get("api_type"):
                return {"status": "error", "message": "API类型不能为空"}
            
            if not config_data.get("api_key"):
                return {"status": "error", "message": "API密钥不能为空"}
            
            if not config_data.get("api_url"):
                return {"status": "error", "message": "API URL不能为空"}
            
            if not config_data.get("model_name"):
                return {"status": "error", "message": "模型名称不能为空"}
            
            # 确保配置数据包含所有必要字段
            config = {
                "name": config_data["name"],
                "api_type": config_data["api_type"],
                "api_key": config_data["api_key"],
                "api_url": config_data["api_url"],
                "model_name": config_data["model_name"],
                "source": config_data.get("source", "external"),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            self.api_config_manager.add_api_config(config["name"], config)
            return {"status": "success", "message": "配置添加成功", "config": config}
        except Exception as e:
            error_handler.handle_error(e, "ExternalAPIManager", "添加配置失败")
            return {"status": "error", "message": str(e)}
    
    def update_config(self, config_id: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新外部API配置
        
        Args:
            config_id: 配置ID
            config_data: 更新后的配置数据
            
        Returns:
            dict: 更新结果，包含status和message字段
        """
        try:
            configs = self.api_config_manager.load_config()
            if config_id not in configs:
                return {"status": "error", "message": "配置不存在"}
            
            # 更新配置
            configs[config_id].update(config_data)
            configs[config_id]["updated_at"] = datetime.now().isoformat()
            
            # 保存更新后的配置
            self.api_config_manager.api_configs = configs
            self.api_config_manager.save_config()
            
            return {"status": "success", "message": "配置更新成功", "config": configs[config_id]}
        except Exception as e:
            error_handler.handle_error(e, "ExternalAPIManager", "更新配置失败")
            return {"status": "error", "message": str(e)}
    
    def delete_config(self, config_id: str) -> Dict[str, Any]:
        """删除外部API配置
        
        Args:
            config_id: 配置ID
            
        Returns:
            dict: 删除结果，包含status和message字段
        """
        try:
            result = self.api_config_manager.remove_api_config(config_id)
            if result:
                return {"status": "success", "message": "配置删除成功"}
            else:
                return {"status": "error", "message": "配置不存在"}
        except Exception as e:
            error_handler.handle_error(e, "ExternalAPIManager", "删除配置失败")
            return {"status": "error", "message": str(e)}
    
    def test_connection(self, config_id: str) -> Dict[str, Any]:
        """测试外部API连接
        
        Args:
            config_id: 配置ID
            
        Returns:
            dict: 测试结果，包含status和message字段
        """
        try:
            configs = self.api_config_manager.load_config()
            if config_id not in configs:
                return {"status": "error", "message": "配置不存在"}
            
            config = configs[config_id]
            result = self.api_config_manager.test_api_connection(config_id)
            
            if result["success"]:
                return {"status": "success", "message": "连接测试成功", "response": result.get("response", "成功连接到API")}
            else:
                return {"status": "error", "message": result.get("error", "连接测试失败")}
        except Exception as e:
            error_handler.handle_error(e, "ExternalAPIManager", "测试连接失败")
            return {"status": "error", "message": str(e)}
    
    def get_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """获取单个外部API配置
        
        Args:
            config_id: 配置ID
            
        Returns:
            dict: 配置数据或None
        """
        try:
            configs = self.api_config_manager.load_config()
            return configs.get(config_id)
        except Exception as e:
            error_handler.handle_error(e, "ExternalAPIManager", "获取配置失败")
            return None
    
    def switch_model_to_external(self, model_id: str, config_id: str) -> Dict[str, Any]:
        """将模型切换到外部API模式
        
        Args:
            model_id: 模型ID
            config_id: 外部API配置ID
            
        Returns:
            dict: 切换结果，包含status和message字段
        """
        try:
            # 获取配置
            config = self.get_config(config_id)
            if not config:
                return {"status": "error", "message": "配置不存在"}
            
            # 从模型注册表中获取模型
            from core.model_registry import get_model_registry
            model_registry = get_model_registry()
            
            # 切换模型到外部API模式
            result = model_registry.switch_model_to_external(model_id, config)
            
            return {"status": "success", "message": "模型已成功切换到外部API模式", "result": result}
        except Exception as e:
            error_handler.handle_error(e, "ExternalAPIManager", "切换模型到外部API模式失败")
            return {"status": "error", "message": str(e)}
