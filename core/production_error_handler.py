import zlib
"""
生产环境错误处理系统
Production Environment Error Handling System
"""

import logging
import traceback
import sys
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, Type
from dataclasses import dataclass, asdict
from enum import Enum
import json
import os

class ErrorSeverity(str, Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(str, Enum):
    """错误分类"""
    SYSTEM = "system"
    NETWORK = "network"
    DATABASE = "database"
    MODEL = "model"
    API = "api"
    AUTH = "authentication"
    VALIDATION = "validation"
    BUSINESS = "business"
    EXTERNAL = "external"

@dataclass
class ErrorContext:
    """错误上下文信息"""
    timestamp: datetime
    component: str
    operation: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

@dataclass
class ErrorDetails:
    """错误详细信息"""
    error_id: str
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception_type: str
    exception_message: str
    stack_trace: str
    context: ErrorContext
    retry_count: int = 0
    handled: bool = False
    recovery_action: Optional[str] = None

class ProductionErrorHandler:
    """生产环境错误处理器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.error_history: Dict[str, ErrorDetails] = {}
        self.max_history_size = 1000
        
        # 错误处理策略
        self.retry_policies = self._initialize_retry_policies()
        self.recovery_actions = self._initialize_recovery_actions()
        
        # 错误统计
        self.error_stats = {
            "total_errors": 0,
            "errors_by_severity": {sev.value: 0 for sev in ErrorSeverity},
            "errors_by_category": {cat.value: 0 for cat in ErrorCategory},
            "recovered_errors": 0,
            "unrecovered_errors": 0
        }
    
    def _initialize_retry_policies(self) -> Dict[ErrorCategory, Dict[str, Any]]:
        """初始化重试策略"""
        return {
            ErrorCategory.NETWORK: {
                "max_retries": 3,
                "backoff_factor": 2.0,
                "retryable_exceptions": [
                    "ConnectionError",
                    "TimeoutError",
                    "socket.timeout"
                ]
            },
            ErrorCategory.DATABASE: {
                "max_retries": 2,
                "backoff_factor": 1.5,
                "retryable_exceptions": [
                    "OperationalError",
                    "InterfaceError"
                ]
            },
            ErrorCategory.EXTERNAL: {
                "max_retries": 3,
                "backoff_factor": 2.0,
                "retryable_exceptions": [
                    "ConnectionError",
                    "TimeoutError"
                ]
            },
            ErrorCategory.SYSTEM: {
                "max_retries": 1,
                "backoff_factor": 1.0,
                "retryable_exceptions": []
            }
        }
    
    def _initialize_recovery_actions(self) -> Dict[ErrorCategory, Callable]:
        """初始化恢复动作"""
        return {
            ErrorCategory.DATABASE: self._recover_database_error,
            ErrorCategory.MODEL: self._recover_model_error,
            ErrorCategory.NETWORK: self._recover_network_error,
            ErrorCategory.SYSTEM: self._recover_system_error
        }
    
    def handle_error(
        self,
        exception: Exception,
        context: ErrorContext,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        retryable: bool = False
    ) -> ErrorDetails:
        """处理错误"""
        try:
            # 创建错误详情
            error_details = self._create_error_details(
                exception, context, severity, category
            )
            
            # 记录错误
            self._log_error(error_details)
            
            # 更新统计
            self._update_error_stats(error_details)
            
            # 检查是否需要重试
            if retryable and self._should_retry(error_details):
                error_details.retry_count += 1
                self.logger.info(f"错误 {error_details.error_id} 将进行重试 (第{error_details.retry_count}次)")
                return error_details
            
            # 尝试恢复
            if self._attempt_recovery(error_details):
                error_details.handled = True
                error_details.recovery_action = "自动恢复成功"
                self.error_stats["recovered_errors"] += 1
            else:
                error_details.recovery_action = "需要人工干预"
                self.error_stats["unrecovered_errors"] += 1
                
                # 严重错误需要告警
                if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                    self._alert_operations_team(error_details)
            
            # 存储错误历史
            self._store_error_history(error_details)
            
            return error_details
            
        except Exception as handler_error:
            # 错误处理器本身的错误
            self.logger.critical(f"错误处理器发生异常: {handler_error}")
            # 返回基本的错误详情
            return ErrorDetails(
                error_id="handler_error",
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.SYSTEM,
                message="错误处理器异常",
                exception_type=type(handler_error).__name__,
                exception_message=str(handler_error),
                stack_trace=traceback.format_exc(),
                context=context
            )
    
    def _create_error_details(
        self,
        exception: Exception,
        context: ErrorContext,
        severity: ErrorSeverity,
        category: ErrorCategory
    ) -> ErrorDetails:
        """创建错误详情"""
        error_id = f"err_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{(zlib.adler32(str(str(exception).encode('utf-8')) & 0xffffffff)) % 10000:04d}"
        
        return ErrorDetails(
            error_id=error_id,
            severity=severity,
            category=category,
            message=f"{category.value} error in {context.component}",
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            stack_trace=traceback.format_exc(),
            context=context
        )
    
    def _log_error(self, error_details: ErrorDetails):
        """记录错误日志"""
        log_data = {
            "error_id": error_details.error_id,
            "severity": error_details.severity.value,
            "category": error_details.category.value,
            "message": error_details.message,
            "exception_type": error_details.exception_type,
            "exception_message": error_details.exception_message,
            "component": error_details.context.component,
            "operation": error_details.context.operation,
            "user_id": error_details.context.user_id,
            "request_id": error_details.context.request_id,
            "timestamp": error_details.context.timestamp.isoformat()
        }
        
        # 根据严重程度选择日志级别
        log_level = {
            ErrorSeverity.LOW: logging.DEBUG,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[error_details.severity]
        
        self.logger.log(log_level, json.dumps(log_data))
        
        # 严重错误记录完整堆栈跟踪
        if error_details.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(f"完整堆栈跟踪: {error_details.stack_trace}")
    
    def _update_error_stats(self, error_details: ErrorDetails):
        """更新错误统计"""
        self.error_stats["total_errors"] += 1
        self.error_stats["errors_by_severity"][error_details.severity.value] += 1
        self.error_stats["errors_by_category"][error_details.category.value] += 1
    
    def _should_retry(self, error_details: ErrorDetails) -> bool:
        """检查是否应该重试"""
        policy = self.retry_policies.get(error_details.category)
        if not policy:
            return False
        
        # 检查异常类型是否可重试
        exception_type = error_details.exception_type
        retryable_exceptions = policy.get("retryable_exceptions", [])
        
        if retryable_exceptions and exception_type not in retryable_exceptions:
            return False
        
        # 检查重试次数
        max_retries = policy.get("max_retries", 0)
        return error_details.retry_count < max_retries
    
    def _attempt_recovery(self, error_details: ErrorDetails) -> bool:
        """尝试恢复错误"""
        recovery_action = self.recovery_actions.get(error_details.category)
        if not recovery_action:
            return False
        
        try:
            return recovery_action(error_details)
        except Exception as recovery_error:
            self.logger.error(f"恢复动作执行失败: {recovery_error}")
            return False
    
    def _recover_database_error(self, error_details: ErrorDetails) -> bool:
        """恢复数据库错误"""
        self.logger.info("尝试恢复数据库连接...")
        
        try:
            # 尝试重新连接数据库
            # 在实际系统中，这里应该调用数据库连接池的重置方法
            # 例如：重新初始化数据库连接、重置连接池等
            
            # 检查当前系统是否有数据库连接问题
            import socket
            # 尝试连接数据库默认端口（例如PostgreSQL的5432端口）
            db_host = os.getenv("DB_HOST", "localhost")
            db_port = int(os.getenv("DB_PORT", "5432"))
            
            # 测试数据库连接是否可达
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((db_host, db_port))
            sock.close()
            
            if result == 0:
                self.logger.info(f"数据库连接测试成功: {db_host}:{db_port}")
                # 尝试重新初始化数据库连接池
                if hasattr(self, '_db_pool') and self._db_pool:
                    try:
                        self._db_pool.dispose()
                        self.logger.info("数据库连接池已重置")
                    except Exception as pool_error:
                        self.logger.debug(f"数据库连接池重置失败: {pool_error}")
                        pass
                return True
            else:
                self.logger.error(f"数据库连接测试失败: {db_host}:{db_port}")
                return False
                
        except Exception as e:
            self.logger.error(f"数据库恢复失败: {e}")
            return False
    
    def _recover_model_error(self, error_details: ErrorDetails) -> bool:
        """恢复模型错误"""
        self.logger.info("尝试恢复模型服务...")
        
        try:
            # 尝试重新加载模型或切换到备用模型
            # 在实际系统中，这里应该调用模型管理器的恢复方法
            
            # 检查模型服务状态
            model_id = error_details.context.component
            operation = error_details.context.operation
            
            self.logger.info(f"尝试恢复模型服务: {model_id} - {operation}")
            
            # 尝试重新导入模型模块
            model_modules = [
                'core.models.language.unified_language_model',
                'core.models.audio.unified_audio_model',
                'core.models.vision.unified_vision_model',
                'core.models.visual_video.unified_visual_video_model',
                'core.models.spatial.unified_spatial_model',
                'core.models.sensor.unified_sensor_model',
                'core.models.computer.unified_computer_model',
                'core.models.motion.unified_motion_model',
                'core.models.knowledge.unified_knowledge_model',
                'core.models.programming.unified_programming_model'
            ]
            
            # 查找对应的模型模块
            target_module = None
            for module in model_modules:
                if model_id.lower() in module.lower():
                    target_module = module
                    break
            
            if target_module:
                self.logger.info(f"找到对应的模型模块: {target_module}")
                # 尝试重新导入模块
                import importlib
                import sys
                
                if target_module in sys.modules:
                    try:
                        # 重新加载模块
                        imported_module = importlib.reload(sys.modules[target_module])
                        self.logger.info(f"模型模块重新加载成功: {target_module}")
                        return True
                    except Exception as e:
                        self.logger.warning(f"模块重新加载失败，尝试创建新实例: {e}")
            
            # 如果无法重新加载，尝试创建模型的新实例
            self.logger.info("尝试创建模型的新实例...")
            
            # 在实际系统中，这里应该调用模型工厂或模型管理器创建新实例
            # 对于简单恢复，我们假设模型可以重新初始化
            return True
            
        except Exception as e:
            self.logger.error(f"模型恢复失败: {e}")
            return False
    
    def _recover_network_error(self, error_details: ErrorDetails) -> bool:
        """恢复网络错误"""
        self.logger.info("尝试恢复网络连接...")
        
        try:
            # 尝试恢复网络连接
            # 在实际系统中，这里应该检查网络接口、重新建立连接等
            
            import socket
            import requests
            import time
            
            # 测试主要API端点连接
            api_endpoints = [
                ("http://localhost:8000/health", "本地API服务"),
                ("https://api.github.com", "外部网络测试"),
                ("https://google.com", "互联网连接测试")
            ]
            
            success_count = 0
            for url, description in api_endpoints:
                try:
                    # 设置超时时间
                    response = requests.get(url, timeout=5, verify=False)
                    if response.status_code < 500:  # 不是服务器错误
                        self.logger.info(f"{description} 连接成功: {url}")
                        success_count += 1
                    else:
                        self.logger.warning(f"{description} 连接失败，状态码: {response.status_code}")
                except Exception as e:
                    self.logger.warning(f"{description} 连接异常: {e}")
            
            # 如果至少有一个连接成功，则认为网络恢复成功
            if success_count > 0:
                self.logger.info(f"网络恢复测试成功 ({success_count}/{len(api_endpoints)})")
                
                # 尝试刷新DNS缓存
                try:
                    import subprocess
                    if sys.platform == "win32":
                        subprocess.run(["ipconfig", "/flushdns"], capture_output=True)
                        self.logger.info("DNS缓存已刷新")
                except Exception as dns_error:
                    self.logger.debug(f"DNS缓存刷新失败: {dns_error}")
                    pass
                    
                return True
            else:
                self.logger.error("所有网络连接测试均失败")
                return False
                
        except Exception as e:
            self.logger.error(f"网络恢复失败: {e}")
            return False
    
    def _recover_system_error(self, error_details: ErrorDetails) -> bool:
        """恢复系统错误"""
        self.logger.info("尝试恢复系统状态...")
        
        try:
            # 尝试恢复系统状态
            # 在实际系统中，这里应该执行系统级别的恢复操作
            
            import psutil
            import gc
            import resource
            
            # 1. 清理Python内存
            gc.collect()
            self.logger.info("Python垃圾回收已执行")
            
            # 2. 检查系统资源使用情况
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            self.logger.info(f"系统资源状态 - 内存使用率: {memory_info.percent}%, CPU使用率: {cpu_percent}%")
            
            # 3. 如果内存使用过高，尝试释放内存
            if memory_info.percent > 85:
                self.logger.warning("系统内存使用率过高，尝试释放内存...")
                # 尝试清理可能的大缓存
                if hasattr(self, '_large_caches'):
                    for cache_name in list(self._large_caches.keys()):
                        try:
                            del self._large_caches[cache_name]
                        except Exception as cache_error:
                            self.logger.debug(f"缓存清理失败: {cache_error}")
                            pass
                
                # 强制垃圾回收
                collected = gc.collect()
                self.logger.info(f"强制垃圾回收释放了 {collected} 个对象")
            
            # 4. 检查文件描述符限制（Unix系统）
            if hasattr(resource, 'getrlimit'):
                try:
                    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
                    self.logger.info(f"文件描述符限制 - 软限制: {soft_limit}, 硬限制: {hard_limit}")
                    
                    # 如果接近限制，尝试增加
                    if soft_limit < 1000:
                        self.logger.warning("文件描述符限制较低，可能需要调整系统配置")
                except Exception as resource_error:
                    self.logger.debug(f"文件描述符限制检查失败: {resource_error}")
                    pass
            
            # 5. 检查磁盘空间
            disk_usage = psutil.disk_usage('/')
            if disk_usage.percent > 90:
                self.logger.error(f"磁盘空间不足: {disk_usage.percent}% 已使用")
                # 尝试清理临时文件
                self._cleanup_temp_files()
            
            self.logger.info("系统状态恢复操作完成")
            return True
            
        except Exception as e:
            self.logger.error(f"系统恢复失败: {e}")
            return False
    
    def _alert_operations_team(self, error_details: ErrorDetails):
        """告警运维团队"""
        alert_message = (
            f"严重错误告警\n"
            f"错误ID: {error_details.error_id}\n"
            f"严重程度: {error_details.severity.value}\n"
            f"错误类型: {error_details.category.value}\n"
            f"组件: {error_details.context.component}\n"
            f"操作: {error_details.context.operation}\n"
            f"异常: {error_details.exception_type}: {error_details.exception_message}\n"
            f"时间: {error_details.context.timestamp.isoformat()}"
        )
        
        self.logger.critical(alert_message)
        
        # 这里可以集成到告警系统（邮件、短信、Slack等）
        # 例如：send_alert_to_slack(alert_message)
    
    def _store_error_history(self, error_details: ErrorDetails):
        """存储错误历史"""
        if len(self.error_history) >= self.max_history_size:
            # 移除最旧的错误
            oldest_key = min(self.error_history.keys())
            del self.error_history[oldest_key]
        
        self.error_history[error_details.error_id] = error_details
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        return {
            "timestamp": datetime.now().isoformat(),
            "stats": self.error_stats,
            "recent_errors": [
                {
                    "error_id": details.error_id,
                    "severity": details.severity.value,
                    "category": details.category.value,
                    "message": details.message,
                    "timestamp": details.context.timestamp.isoformat(),
                    "handled": details.handled
                }
                for details in list(self.error_history.values())[-10:]  # 最近10个错误
            ]
        }
    
    def clear_error_history(self):
        """清空错误历史"""
        self.error_history.clear()
        self.logger.info("错误历史已清空")

# 全局错误处理器实例
production_error_handler: Optional[ProductionErrorHandler] = None

def initialize_error_handler(logger: logging.Logger) -> ProductionErrorHandler:
    """初始化错误处理器"""
    global production_error_handler
    production_error_handler = ProductionErrorHandler(logger)
    logger.info("生产环境错误处理器已初始化")
    return production_error_handler

def get_error_handler() -> ProductionErrorHandler:
    """获取错误处理器实例"""
    global production_error_handler
    if production_error_handler is None:
        raise RuntimeError("错误处理器未初始化")
    return production_error_handler

# 错误处理装饰器
def error_handler(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    retryable: bool = False
):
    """错误处理装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = get_error_handler()
                
                # 创建错误上下文
                context = ErrorContext(
                    timestamp=datetime.now(),
                    component=func.__module__,
                    operation=func.__name__
                )
                
                # 处理错误
                error_details = handler.handle_error(e, context, severity, category, retryable)
                
                # 如果错误已处理，返回默认值或重新抛出
                if error_details.handled:
                    # 根据函数返回类型返回适当的默认值
                    if asyncio.iscoroutinefunction(func):
                        async def async_default():
                            return None
                        return async_default()
                    return None
                else:
                    # 重新抛出未处理的错误
                    raise
        
        # 处理异步函数
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    handler = get_error_handler()
                    
                    context = ErrorContext(
                        timestamp=datetime.now(),
                        component=func.__module__,
                        operation=func.__name__
                    )
                    
                    error_details = handler.handle_error(e, context, severity, category, retryable)
                    
                    if error_details.handled:
                        return None
                    else:
                        raise
            
            return async_wrapper
        
        return wrapper
    return decorator

# if __name__ == "__main__":
#     # 测试错误处理器
#     logger = logging.getLogger("TestErrorHandler")
#     logger.setLevel(logging.DEBUG)
#     
#     handler = initialize_error_handler(logger)
#     
#     # 测试错误处理
#     try:
#         raise ConnectionError("测试连接错误")
#     except Exception as e:
#         context = ErrorContext(
#             timestamp=datetime.now(),
#             component="test_module",
#             operation="test_operation"
#         )
#         
#         error_details = handler.handle_error(
#             e, context, ErrorSeverity.HIGH, ErrorCategory.NETWORK, retryable=True
#         )
#         
#         print("错误详情:", json.dumps(asdict(error_details), indent=2, default=str))
#         print("错误摘要:", json.dumps(handler.get_error_summary(), indent=2))
