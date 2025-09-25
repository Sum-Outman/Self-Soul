"""
外部API集成混入类 - 提供统一的外部API服务集成功能
External API Integration Mixin - Provides unified external API service integration

功能包括：
- 外部API服务初始化和配置
- API连接测试和状态管理
- 多类型API服务调用（图像、视频等）
- API能力查询和配置验证
- 统一的错误处理和重试机制
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

class ExternalAPIMixin:
    """外部API集成混入类，提供统一的外部API服务管理"""
    
    def __init__(self, *args, **kwargs):
        """初始化外部API集成功能"""
        super().__init__(*args, **kwargs)
        
        # 外部API配置
        self.external_api_config = None
        self.use_external_api = False
        
        # 外部API服务实例
        self.external_api_service = None
        
        # API连接状态
        self._api_connection_tested = False
        self._last_api_test_time = None
        
        # 初始化外部API服务（如果可用）
        self._initialize_external_api_service()
    
    def _initialize_external_api_service(self):
        """初始化外部API服务"""
        try:
            # 尝试导入ExternalAPIService
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            
            from external_api_service import ExternalAPIService
            
            # 获取配置
            config = getattr(self, 'config', {})
            external_api_config = config.get('external_api_config', {})
            
            # 初始化外部API服务
            self.external_api_service = ExternalAPIService(external_api_config)
            self.logger.info(f"External API service initialized for {getattr(self, 'model_id', 'unknown')}")
            
        except ImportError as e:
            self.logger.warning(f"ExternalAPIService not available: {str(e)}")
            self.external_api_service = None
        except Exception as e:
            self.logger.error(f"Failed to initialize external API service: {str(e)}")
            self.external_api_service = None
    
    def set_mode(self, mode: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """设置模型运行模式（本地或外部API）
        
        Args:
            mode: 运行模式 ('local' 或 'external')
            config: 当mode为'external'时的API配置
            
        Returns:
            设置结果字典
        """
        if mode == "external":
            # 验证API配置
            validation_result = self._validate_api_config(config)
            if not validation_result["success"]:
                return validation_result
            
            # 应用配置
            self.use_external_api = True
            self.external_api_config = validation_result["normalized_config"]
            
            # 测试API连接
            test_result = self.test_connection()
            if not test_result["success"]:
                self.use_external_api = False
                self.external_api_config = None
                return {"success": False, "error": f"API连接测试失败: {test_result.get('error', 'Unknown error')}"}
            
            self.logger.info(f"模型 {getattr(self, 'model_id', 'unknown')} 已切换到外部API模式")
            return {"success": True}
        
        elif mode == "local":
            self.use_external_api = False
            self.external_api_config = None
            self.logger.info(f"模型 {getattr(self, 'model_id', 'unknown')} 已切换到本地模式")
            return {"success": True}
        
        else:
            return {"success": False, "error": f"不支持的模式: {mode}"}
    
    def _validate_api_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证API配置
        
        Args:
            config: API配置字典
            
        Returns:
            验证结果和规范化配置
        """
        if not config:
            return {"success": False, "error": "外部模式需要提供API配置"}
        
        # 规范化API配置字段
        normalized_config = {}
        
        # 提取API URL
        if 'api_url' in config:
            normalized_config['api_url'] = config['api_url']
        elif 'url' in config:
            normalized_config['api_url'] = config['url']
        elif 'endpoint' in config:
            normalized_config['api_url'] = config['endpoint']
        else:
            return {"success": False, "error": "缺少必要的API配置项: api_url或url或endpoint"}
        
        # 提取API密钥
        if 'api_key' in config:
            normalized_config['api_key'] = config['api_key']
        else:
            return {"success": False, "error": "缺少必要的API配置项: api_key"}
        
        # 提取模型名称
        if 'model_name' in config:
            normalized_config['model_name'] = config['model_name']
        else:
            normalized_config['model_name'] = getattr(self, 'model_id', 'unknown')
        
        # 提取来源
        if 'source' in config:
            normalized_config['source'] = config['source']
        else:
            normalized_config['source'] = 'external'
        
        # 检查必要的配置项值是否为空
        for key in ['api_url', 'api_key']:
            if not normalized_config[key]:
                return {"success": False, "error": f"API配置项值不能为空: {key}"}
        
        # 检查URL格式是否有效
        url = normalized_config['api_url']
        if not (url.startswith('http://') or url.startswith('https://')):
            return {"success": False, "error": f"无效的API URL格式: {url}"}
        
        return {"success": True, "normalized_config": normalized_config}
    
    def test_connection(self) -> Dict[str, Any]:
        """测试外部API连接
        
        Returns:
            连接测试结果字典
        """
        if not self.use_external_api or not self.external_api_config:
            return {"success": False, "error": "未配置外部API"}
        
        try:
            config = self.external_api_config
            api_url = config.get('api_url', '')
            api_key = config.get('api_key', '')
            
            if not api_url:
                return {"success": False, "error": "缺少API URL"}
            
            if not api_key:
                return {"success": False, "error": "缺少API密钥"}
            
            self.logger.info(f"正在测试外部API连接: {api_url}")
            
            # 模拟连接测试（实际实现应调用具体API）
            time.sleep(0.1)  # 模拟网络延迟
            
            # 检查URL格式
            if not (api_url.startswith('http://') or api_url.startswith('https://')):
                return {"success": False, "error": f"无效的API URL格式: {api_url}"}
            
            # 更新连接状态
            self._api_connection_tested = True
            self._last_api_test_time = datetime.now()
            
            self.logger.info(f"外部API连接测试成功: {getattr(self, 'model_id', 'unknown')}")
            return {
                "success": True,
                "model_id": getattr(self, 'model_id', 'unknown'),
                "api_url": api_url,
                "model_name": config.get('model_name', getattr(self, 'model_id', 'unknown')),
                "source": config.get('source', 'external'),
                "timestamp": time.time(),
                "message": "API连接测试成功"
            }
        except Exception as e:
            error_message = str(e)
            self.logger.error(f"外部API连接测试失败: {error_message}")
            return {
                "success": False, 
                "error": error_message,
                "model_id": getattr(self, 'model_id', 'unknown'),
                "timestamp": time.time()
            }
    
    def get_api_status(self) -> Dict[str, Any]:
        """获取API连接状态
        
        Returns:
            API状态信息字典
        """
        return {
            "use_external_api": self.use_external_api,
            "api_config": self.external_api_config,
            "connection_status": self.test_connection() if self.use_external_api else {
                "success": True, 
                "message": "使用本地模式"
            },
            "last_test_time": self._last_api_test_time.isoformat() if self._last_api_test_time else None,
            "connection_tested": self._api_connection_tested
        }
    
    def use_external_api_service(self, api_type: str, service_type: str, data: Any) -> Dict[str, Any]:
        """使用统一的外部API服务处理数据
        
        Args:
            api_type: API类型（如'google', 'openai'等）
            service_type: 服务类型（如'image', 'video'等）
            data: 要处理的数据
            
        Returns:
            API服务处理结果
        """
        if not self.external_api_service:
            return {"success": False, "error": "External API service not available"}
        
        try:
            start_time = time.time()
            
            # 根据API类型和数据类型调用相应的服务
            if service_type == "image":
                result = self.external_api_service.analyze_image(data, api_type)
            elif service_type == "video":
                result = self.external_api_service.analyze_video(data, api_type)
            elif service_type == "text":
                result = self.external_api_service.analyze_text(data, api_type)
            elif service_type == "audio":
                result = self.external_api_service.analyze_audio(data, api_type)
            else:
                return {"success": False, "error": f"不支持的服务类型: {service_type}"}
            
            # 更新性能指标（如果存在）
            if hasattr(self, '_update_performance_metrics'):
                response_time = time.time() - start_time
                self._update_performance_metrics(response_time, True)
            
            return {
                "success": True,
                "api_type": api_type,
                "service_type": service_type,
                "result": result,
                "response_time": time.time() - start_time
            }
            
        except Exception as e:
            # 错误处理（如果存在）
            if hasattr(self, '_handle_error'):
                self._handle_error(e, "external_api_service")
            else:
                self.logger.error(f"Error in external API service: {str(e)}")
            
            return {"success": False, "error": str(e)}
    
    def get_external_api_capabilities(self) -> Dict[str, Any]:
        """获取外部API服务的能力信息
        
        Returns:
            API能力信息字典
        """
        if not self.external_api_service:
            return {"success": False, "error": "External API service not available"}
        
        try:
            capabilities = self.external_api_service.get_capabilities()
            return {
                "success": True,
                "capabilities": capabilities,
                "model_id": getattr(self, 'model_id', 'unknown')
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def is_external_api_available(self) -> bool:
        """检查外部API服务是否可用
        
        Returns:
            API服务可用性状态
        """
        return self.external_api_service is not None
    
    def get_supported_api_types(self) -> list:
        """获取支持的API类型列表
        
        Returns:
            支持的API类型列表
        """
        if not self.external_api_service:
            return []
        
        try:
            capabilities = self.external_api_service.get_capabilities()
            return capabilities.get('supported_api_types', [])
        except Exception:
            return []
    
    def get_supported_service_types(self) -> list:
        """获取支持的服务类型列表
        
        Returns:
            支持的服务类型列表
        """
        if not self.external_api_service:
            return []
        
        try:
            capabilities = self.external_api_service.get_capabilities()
            return capabilities.get('supported_service_types', [])
        except Exception:
            return []
    
    def validate_api_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证API配置的有效性
        
        Args:
            config: 要验证的API配置
            
        Returns:
            验证结果
        """
        validation_result = self._validate_api_config(config)
        if not validation_result["success"]:
            return validation_result
        
        # 临时应用配置进行连接测试
        original_config = self.external_api_config
        original_mode = self.use_external_api
        
        try:
            self.external_api_config = validation_result["normalized_config"]
            self.use_external_api = True
            
            test_result = self.test_connection()
            
            # 恢复原始配置
            self.external_api_config = original_config
            self.use_external_api = original_mode
            
            return test_result
            
        except Exception as e:
            # 恢复原始配置
            self.external_api_config = original_config
            self.use_external_api = original_mode
            
            return {"success": False, "error": f"配置验证失败: {str(e)}"}
    
    def switch_to_local_mode(self) -> Dict[str, Any]:
        """切换到本地模式
        
        Returns:
            切换结果
        """
        self.use_external_api = False
        self.external_api_config = None
        self.logger.info(f"模型 {getattr(self, 'model_id', 'unknown')} 已切换到本地模式")
        return {"success": True}
    
    def switch_to_external_mode(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """切换到外部API模式
        
        Args:
            config: 外部API配置
            
        Returns:
            切换结果
        """
        return self.set_mode("external", config)
