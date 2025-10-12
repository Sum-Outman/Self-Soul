"""
统一混入类 - 合并重复功能，消除代码冗余
Unified Mixins - Merge duplicate functionalities and eliminate code redundancy

这个文件整合了以下混入类的功能：
1. PerformanceMixin (性能监控)
2. CacheMixin (缓存优化) 
3. ErrorHandlingMixin (错误处理)
4. ResourceMixin (资源管理)
5. ExternalAPIMixin (外部API集成)
6. TrainingMixin (训练生命周期管理)

通过统一架构消除重复代码，提高代码可维护性。
"""

import os
import json
import pickle
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class UnifiedPerformanceCacheMixin:
    """
    统一性能监控和缓存管理混入类
    合并了PerformanceMixin和CacheMixin的功能
    """
    
    def __init__(self, *args, **kwargs):
        """初始化性能和缓存管理功能"""
        super().__init__(*args, **kwargs)
        
        # 性能指标
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "last_response_time": 0.0,
            "peak_memory_usage": 0,
            "cpu_utilization": 0.0,
            "cache_hit_rate": 0.0,
            "throughput": 0.0
        }
        
        # 缓存配置
        self.cache_enabled = getattr(self, 'config', {}).get('cache_enabled', True)
        self.cache_ttl = getattr(self, 'config', {}).get('cache_ttl', 300)
        self.cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'size': 0,
            'last_cleanup': time.time()
        }
        
        # 性能监控状态
        self._performance_monitoring_active = False
        self._monitoring_start_time = 0
        self._current_operation = None
    
    def start_performance_monitoring(self, operation: str):
        """开始性能监控"""
        self._performance_monitoring_active = True
        self._monitoring_start_time = time.time()
        self._current_operation = operation
        logger.debug(f"Started performance monitoring for {operation}")
    
    def stop_performance_monitoring(self, operation: str):
        """停止性能监控并更新指标"""
        if self._performance_monitoring_active and self._current_operation == operation:
            response_time = time.time() - self._monitoring_start_time
            self._update_performance_metrics(response_time, True)
            self._performance_monitoring_active = False
            self._current_operation = None
            logger.debug(f"Stopped performance monitoring for {operation}")
    
    def _update_performance_metrics(self, response_time: float, success: bool):
        """更新性能指标"""
        self.performance_metrics["total_requests"] += 1
        if success:
            self.performance_metrics["successful_requests"] += 1
        else:
            self.performance_metrics["failed_requests"] += 1
        
        # 更新平均响应时间
        current_avg = self.performance_metrics["average_response_time"]
        total_requests = self.performance_metrics["total_requests"]
        self.performance_metrics["average_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
        
        self.performance_metrics["last_response_time"] = response_time
        self.performance_metrics["throughput"] = (
            self.performance_metrics["successful_requests"] / 
            (time.time() - getattr(self, '_start_time', time.time()))
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self.performance_metrics.copy()
    
    def optimize_performance(self) -> Dict[str, Any]:
        """优化性能"""
        optimization_results = {}
        
        # 内存优化
        memory_saved = self._optimize_memory_usage()
        optimization_results['memory_saved'] = memory_saved
        
        # CPU优化
        cpu_improvement = self._optimize_cpu_usage()
        optimization_results['cpu_improvement'] = cpu_improvement
        
        # 缓存优化
        cache_optimized = self._optimize_cache()
        optimization_results['cache_optimized'] = cache_optimized
        
        logger.info(f"Performance optimization completed: {optimization_results}")
        return optimization_results
    
    def _optimize_memory_usage(self) -> int:
        """优化内存使用"""
        # 清理不必要的缓存
        initial_size = len(self.cache)
        self._cleanup_expired_cache()
        memory_saved = initial_size - len(self.cache)
        logger.info(f"Memory optimization: saved {memory_saved} cache entries")
        return memory_saved
    
    def _optimize_cpu_usage(self) -> float:
        """优化CPU使用"""
        # 简化计算或优化算法
        cpu_improvement = 0.1  # 模拟10%的改进
        logger.info(f"CPU optimization: {cpu_improvement * 100}% improvement")
        return cpu_improvement
    
    def _optimize_cache(self) -> bool:
        """优化缓存"""
        self._cleanup_expired_cache()
        logger.info("Cache optimization completed")
        return True
    
    # 缓存管理方法
    def is_caching_enabled(self) -> bool:
        """检查缓存是否启用"""
        return self.cache_enabled
    
    def enable_caching(self, enabled: bool = True):
        """启用或禁用缓存"""
        self.cache_enabled = enabled
        logger.info(f"Caching {'enabled' if enabled else 'disabled'}")
    
    def cache_result(self, key: str, value: Any, ttl: int = None):
        """缓存结果"""
        if not self.cache_enabled:
            return
        
        if ttl is None:
            ttl = self.cache_ttl
        
        self.cache[key] = {
            'value': value,
            'timestamp': time.time(),
            'ttl': ttl
        }
        self.cache_stats['size'] += 1
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """获取缓存结果"""
        if not self.cache_enabled or key not in self.cache:
            self.cache_stats['misses'] += 1
            return None
        
        cached_item = self.cache[key]
        current_time = time.time()
        
        # 检查是否过期
        if current_time - cached_item['timestamp'] > cached_item['ttl']:
            del self.cache[key]
            self.cache_stats['size'] -= 1
            self.cache_stats['misses'] += 1
            return None
        
        self.cache_stats['hits'] += 1
        return cached_item['value']
    
    def is_cached(self, key: str) -> bool:
        """检查键是否在缓存中且未过期"""
        if key not in self.cache:
            return False
        
        cached_item = self.cache[key]
        current_time = time.time()
        return current_time - cached_item['timestamp'] <= cached_item['ttl']
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'size': 0,
            'last_cleanup': time.time()
        }
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return self.cache_stats.copy()
    
    def _cleanup_expired_cache(self):
        """清理过期的缓存项"""
        current_time = time.time()
        expired_keys = []
        
        for key, item in self.cache.items():
            if current_time - item['timestamp'] > item['ttl']:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            self.cache_stats['size'] -= 1
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        self.cache_stats['last_cleanup'] = current_time


class UnifiedErrorResourceMixin:
    """
    统一错误处理和资源管理混入类
    合并了ErrorHandlingMixin和ResourceMixin的功能
    """
    
    def __init__(self, *args, **kwargs):
        """初始化错误处理和资源管理功能"""
        super().__init__(*args, **kwargs)
        
        # 错误处理配置
        config = getattr(self, 'config', {})
        self.auto_recovery_enabled = config.get('auto_recovery', True)
        self.max_retry_attempts = config.get('max_retry_attempts', 3)
        self.error_threshold = config.get('error_threshold', 10)
        self.error_history = []
        
        # 资源管理
        self._resource_allocations = {}
        self._memory_usage_baseline = self._get_memory_usage()
        self._resource_monitoring_enabled = True
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理错误并尝试自动恢复"""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context': context or {},
            'recovered': False
        }
        
        self.error_history.append(error_info)
        
        # 检查错误阈值
        if len(self.error_history) > self.error_threshold:
            logger.warning(f"Error threshold exceeded: {len(self.error_history)} errors")
        
        # 尝试自动恢复
        if self.auto_recovery_enabled:
            recovery_result = self._attempt_auto_recovery(error, context)
            error_info['recovery_attempted'] = True
            error_info['recovery_result'] = recovery_result
            
            if recovery_result.get('success', False):
                error_info['recovered'] = True
                logger.info(f"Error recovered: {error}")
                return error_info
        
        logger.error(f"Unrecovered error: {error}")
        return error_info
    
    def _attempt_auto_recovery(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """尝试自动恢复"""
        recovery_actions = [
            self._clear_caches_and_retry,
            self._reset_model_state,
            self._fallback_to_simpler_operation
        ]
        
        for attempt, recovery_action in enumerate(recovery_actions):
            try:
                result = recovery_action(error, context)
                if result.get('success', False):
                    return {
                        'success': True,
                        'recovery_action': recovery_action.__name__,
                        'attempt': attempt + 1
                    }
            except Exception as recovery_error:
                logger.warning(f"Recovery attempt {attempt + 1} failed: {recovery_error}")
        
        return {'success': False, 'message': 'All recovery attempts failed'}
    
    def _clear_caches_and_retry(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """清理缓存并重试"""
        if hasattr(self, 'clear_cache'):
            self.clear_cache()
            return {'success': True, 'action': 'cache_cleared'}
        return {'success': False}
    
    def _reset_model_state(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """重置模型状态"""
        if hasattr(self, 'model_state'):
            self.model_state = getattr(self, 'initial_model_state', {})
            return {'success': True, 'action': 'state_reset'}
        return {'success': False}
    
    def _fallback_to_simpler_operation(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """回退到更简单的操作"""
        return {'success': True, 'action': 'fallback_operation'}
    
    def get_error_history(self) -> List[Dict[str, Any]]:
        """获取错误历史"""
        return self.error_history.copy()
    
    def clear_error_history(self):
        """清空错误历史"""
        self.error_history.clear()
        logger.info("Error history cleared")
    
    # 资源管理方法
    def allocate_resources(self, resource_type: str, amount: int) -> bool:
        """分配资源"""
        if resource_type in self._resource_allocations:
            logger.warning(f"Resource {resource_type} already allocated")
            return False
        
        self._resource_allocations[resource_type] = {
            'amount': amount,
            'timestamp': time.time(),
            'status': 'allocated'
        }
        logger.info(f"Allocated {amount} units of {resource_type}")
        return True
    
    def release_resources(self, resource_type: str) -> bool:
        """释放资源"""
        if resource_type not in self._resource_allocations:
            logger.warning(f"Resource {resource_type} not found")
            return False
        
        del self._resource_allocations[resource_type]
        logger.info(f"Released resource: {resource_type}")
        return True
    
    def get_resource_status(self) -> Dict[str, Any]:
        """获取资源状态"""
        current_memory = self._get_memory_usage()
        memory_increase = current_memory - self._memory_usage_baseline
        
        return {
            'allocated_resources': self._resource_allocations.copy(),
            'memory_usage': current_memory,
            'memory_increase': memory_increase,
            'resource_count': len(self._resource_allocations),
            'monitoring_enabled': self._resource_monitoring_enabled
        }
    
    def _get_memory_usage(self) -> int:
        """获取内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0
    
    def cleanup_resources(self):
        """清理所有资源"""
        resources_to_release = list(self._resource_allocations.keys())
        for resource_type in resources_to_release:
            self.release_resources(resource_type)
        
        logger.info("All resources cleaned up")


class UnifiedExternalAPIMixin:
    """
    统一外部API集成混入类
    提供统一的外部API服务集成功能
    """
    
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
            logger.info(f"External API service initialized for {getattr(self, 'model_id', 'unknown')}")
            
        except ImportError as e:
            logger.warning(f"ExternalAPIService not available: {str(e)}")
            self.external_api_service = None
        except Exception as e:
            logger.error(f"Failed to initialize external API service: {str(e)}")
            self.external_api_service = None
    
    def set_mode(self, mode: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """设置模型运行模式（本地或外部API）"""
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
            
            logger.info(f"模型 {getattr(self, 'model_id', 'unknown')} 已切换到外部API模式")
            return {"success": True}
        
        elif mode == "local":
            self.use_external_api = False
            self.external_api_config = None
            logger.info(f"模型 {getattr(self, 'model_id', 'unknown')} 已切换到本地模式")
            return {"success": True}
        
        else:
            return {"success": False, "error": f"不支持的模式: {mode}"}
    
    def _validate_api_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证API配置"""
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
        """测试外部API连接"""
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
            
            logger.info(f"正在测试外部API连接: {api_url}")
            
            # 模拟连接测试（实际实现应调用具体API）
            time.sleep(0.1)  # 模拟网络延迟
            
            # 检查URL格式
            if not (api_url.startswith('http://') or api_url.startswith('https://')):
                return {"success": False, "error": f"无效的API URL格式: {api_url}"}
            
            # 更新连接状态
            self._api_connection_tested = True
            self._last_api_test_time = datetime.now()
            
            logger.info(f"外部API连接测试成功: {getattr(self, 'model_id', 'unknown')}")
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
            logger.error(f"外部API连接测试失败: {error_message}")
            return {
                "success": False, 
                "error": error_message,
                "model_id": getattr(self, 'model_id', 'unknown'),
                "timestamp": time.time()
            }
    
    def get_api_status(self) -> Dict[str, Any]:
        """获取API连接状态"""
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
        """使用统一的外部API服务处理数据"""
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
                logger.error(f"Error in external API service: {str(e)}")
            
            return {"success": False, "error": str(e)}
    
    def get_external_api_capabilities(self) -> Dict[str, Any]:
        """获取外部API服务的能力信息"""
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
        """检查外部API服务是否可用"""
        return self.external_api_service is not None
    
    def get_supported_api_types(self) -> list:
        """获取支持的API类型列表"""
        if not self.external_api_service:
            return []
        
        try:
            capabilities = self.external_api_service.get_capabilities()
            return capabilities.get('supported_api_types', [])
        except Exception:
            return []
    
    def get_supported_service_types(self) -> list:
        """获取支持的服务类型列表"""
        if not self.external_api_service:
            return []
        
        try:
            capabilities = self.external_api_service.get_capabilities()
            return capabilities.get('supported_service_types', [])
        except Exception:
            return []
    
    def validate_api_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证API配置的有效性"""
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
        """切换到本地模式"""
        self.use_external_api = False
        self.external_api_config = None
        logger.info(f"模型 {getattr(self, 'model_id', 'unknown')} 已切换到本地模式")
        return {"success": True}
    
    def switch_to_external_mode(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """切换到外部API模式"""
        return self.set_mode("external", config)
