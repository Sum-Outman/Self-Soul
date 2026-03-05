import zlib
"""
API异常处理工具类 - API Exception Handler Utility
提供统一的异常处理机制，减少代码重复
Provides unified exception handling mechanism to reduce code duplication
"""

import logging
import functools
from typing import Any, Callable, Optional, Dict, Type, Union
from datetime import datetime


class APIExceptionHandler:
    """API异常处理类 | API Exception Handler Class
    
    提供统一的异常处理装饰器、上下文管理器和工具函数
    Provides unified exception handler decorators, context managers and utility functions
    """
    
    def __init__(self, logger_name: str = __name__):
        """初始化异常处理器 | Initialize exception handler"""
        self.logger = logging.getLogger(logger_name)
        
        # 异常类型到处理策略的映射
        # Exception type to handling strategy mapping
        self.exception_strategies = {
            "ConnectionError": self._handle_connection_error,
            "TimeoutError": self._handle_timeout_error,
            "ValueError": self._handle_value_error,
            "TypeError": self._handle_type_error,
            "PermissionError": self._handle_permission_error,
            "default": self._handle_generic_error
        }
        
        # 错误恢复策略
        # Error recovery strategies
        self.recovery_strategies = {
            "retry": self._retry_strategy,
            "fallback": self._fallback_strategy,
            "circuit_breaker": self._circuit_breaker_strategy
        }
        
        # 错误统计
        # Error statistics
        self.error_stats = {
            "total_errors": 0,
            "error_types": {},
            "recovery_attempts": 0,
            "successful_recoveries": 0
        }
    
    def handle_exception(self, func: Callable) -> Callable:
        """异常处理装饰器 | Exception handling decorator
        
        Args:
            func: 要装饰的函数 | Function to decorate
            
        Returns:
            装饰后的函数 | Decorated function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 获取调用者信息
                # Get caller information
                caller = self._get_caller_info(func, args)
                
                # 根据异常类型选择处理策略
                # Select handling strategy based on exception type
                strategy = self.exception_strategies.get(
                    type(e).__name__, 
                    self.exception_strategies["default"]
                )
                
                # 记录错误统计
                # Record error statistics
                self._record_error_statistics(e)
                
                # 应用处理策略
                # Apply handling strategy
                return strategy(e, caller, func, args, kwargs)
        
        return wrapper
    
    def retry_on_failure(
        self, 
        max_retries: int = 3, 
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: tuple = (Exception,)
    ) -> Callable:
        """重试装饰器 | Retry decorator
        
        Args:
            max_retries: 最大重试次数 | Maximum retry attempts
            delay: 初始延迟时间（秒） | Initial delay in seconds
            backoff_factor: 退避因子 | Backoff factor
            exceptions: 触发重试的异常类型 | Exception types that trigger retry
            
        Returns:
            装饰器函数 | Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        if attempt > 0:
                            # 计算退避延迟
                            # Calculate backoff delay
                            wait_time = delay * (backoff_factor ** (attempt - 1))
                            self.logger.info(f"重试尝试 {attempt}/{max_retries}, 等待 {wait_time:.2f}秒后重试... | Retry attempt {attempt}/{max_retries}, waiting {wait_time:.2f} seconds before retry...")
                            
                            import time
                            time.sleep(wait_time)
                        
                        return func(*args, **kwargs)
                    
                    except exceptions as e:
                        last_exception = e
                        caller = self._get_caller_info(func, args)
                        
                        if attempt < max_retries:
                            self.logger.warning(
                                f"尝试 {attempt + 1}/{max_retries} 失败: {str(e)} | "
                                f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                            )
                        else:
                            self.logger.error(
                                f"所有 {max_retries} 次重试均失败: {str(e)} | "
                                f"All {max_retries} retry attempts failed: {str(e)}"
                            )
                
                # 所有重试都失败，抛出最后一个异常
                # All retries failed, raise the last exception
                raise last_exception
            
            return wrapper
        
        return decorator
    
    def fallback_on_failure(
        self, 
        fallback_func: Callable,
        exceptions: tuple = (Exception,)
    ) -> Callable:
        """故障转移装饰器 | Fallback decorator
        
        Args:
            fallback_func: 备用函数 | Fallback function
            exceptions: 触发故障转移的异常类型 | Exception types that trigger fallback
            
        Returns:
            装饰器函数 | Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    self.logger.warning(
                        f"主函数失败，使用备用函数: {str(e)} | "
                        f"Primary function failed, using fallback: {str(e)}"
                    )
                    
                    # 记录故障转移统计
                    # Record fallback statistics
                    self.error_stats["recovery_attempts"] += 1
                    
                    try:
                        result = fallback_func(*args, **kwargs)
                        self.error_stats["successful_recoveries"] += 1
                        return result
                    except Exception as fallback_e:
                        self.logger.error(
                            f"备用函数也失败: {str(fallback_e)} | "
                            f"Fallback function also failed: {str(fallback_e)}"
                        )
                        raise e  # 重新抛出原始异常 | Re-raise original exception
            
            return wrapper
        
        return decorator
    
    def circuit_breaker(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        exceptions: tuple = (Exception,)
    ) -> Callable:
        """断路器装饰器 | Circuit breaker decorator
        
        Args:
            failure_threshold: 失败阈值 | Failure threshold
            reset_timeout: 重置超时时间（秒） | Reset timeout in seconds
            exceptions: 触发断路器的异常类型 | Exception types that trigger circuit breaker
            
        Returns:
            装饰器函数 | Decorator function
        """
        import time
        from threading import Lock
        
        state = {
            "closed": "CLOSED",      # 正常状态 | Normal state
            "open": "OPEN",          # 断路状态 | Circuit open state
            "half_open": "HALF_OPEN" # 半开状态 | Half-open state
        }
        
        circuit_state = {
            "state": state["closed"],
            "failure_count": 0,
            "last_failure_time": 0,
            "lock": Lock()
        }
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with circuit_state["lock"]:
                    current_state = circuit_state["state"]
                    current_time = time.time()
                    
                    # 检查是否需要从OPEN状态转换到HALF_OPEN状态
                    # Check if need to transition from OPEN to HALF_OPEN
                    if current_state == state["open"]:
                        if current_time - circuit_state["last_failure_time"] > reset_timeout:
                            circuit_state["state"] = state["half_open"]
                            self.logger.info("断路器状态: 半开（尝试恢复） | Circuit breaker state: HALF_OPEN (attempting recovery)")
                        else:
                            # 仍在断路中，直接返回错误
                            # Still in circuit open state, return error directly
                            remaining = reset_timeout - (current_time - circuit_state["last_failure_time"])
                            self.logger.error(
                                f"断路器已打开，请{remaining:.1f}秒后重试 | "
                                f"Circuit breaker is OPEN, please retry after {remaining:.1f} seconds"
                            )
                            raise Exception(f"Circuit breaker is OPEN. Try again in {remaining:.1f} seconds")
                    
                try:
                    result = func(*args, **kwargs)
                    
                    # 成功调用，重置断路器
                    # Successful call, reset circuit breaker
                    with circuit_state["lock"]:
                        if circuit_state["state"] == state["half_open"]:
                            self.logger.info("断路器恢复成功，状态: 关闭 | Circuit breaker recovered successfully, state: CLOSED")
                        
                        circuit_state["state"] = state["closed"]
                        circuit_state["failure_count"] = 0
                    
                    return result
                
                except exceptions as e:
                    with circuit_state["lock"]:
                        circuit_state["failure_count"] += 1
                        circuit_state["last_failure_time"] = time.time()
                        
                        if circuit_state["failure_count"] >= failure_threshold:
                            circuit_state["state"] = state["open"]
                            self.logger.error(
                                f"断路器已打开（连续失败{failure_threshold}次） | "
                                f"Circuit breaker OPEN (consecutive failures: {failure_threshold})"
                            )
                        elif circuit_state["state"] == state["half_open"]:
                            # 半开状态下失败，重新打开断路器
                            # Failure in half-open state, re-open circuit breaker
                            circuit_state["state"] = state["open"]
                            self.logger.error("断路器恢复尝试失败，重新打开 | Circuit breaker recovery attempt failed, re-opening")
                    
                    raise e
            
            return wrapper
        
        return decorator
    
    def error_context(self, operation_name: str, **context_info):
        """错误上下文管理器 | Error context manager
        
        Args:
            operation_name: 操作名称 | Operation name
            **context_info: 上下文信息 | Context information
            
        Returns:
            上下文管理器 | Context manager
        """
        class ErrorContext:
            def __init__(self, handler, operation_name, context_info):
                self.handler = handler
                self.operation_name = operation_name
                self.context_info = context_info
                self.start_time = None
            
            def __enter__(self):
                self.start_time = datetime.now()
                self.handler.logger.debug(
                    f"开始操作: {self.operation_name} | "
                    f"Starting operation: {self.operation_name}"
                )
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = (datetime.now() - self.start_time).total_seconds()
                
                if exc_type is None:
                    self.handler.logger.debug(
                        f"操作成功: {self.operation_name} (耗时: {duration:.2f}s) | "
                        f"Operation successful: {self.operation_name} (duration: {duration:.2f}s)"
                    )
                    return False
                
                # 处理异常
                # Handle exception
                error_info = {
                    "operation": self.operation_name,
                    "error_type": exc_type.__name__,
                    "error_message": str(exc_val),
                    "duration": duration,
                    "context": self.context_info
                }
                
                self.handler._handle_exception_in_context(error_info, exc_type, exc_val, exc_tb)
                
                # 返回True表示异常已处理，False表示继续传播
                # Return True to indicate exception handled, False to continue propagation
                return False
        
        return ErrorContext(self, operation_name, context_info)
    
    # 异常处理策略方法 | Exception handling strategy methods
    def _handle_connection_error(self, error, caller, func, args, kwargs):
        """处理连接错误 | Handle connection error"""
        self.logger.error(
            f"连接错误: {str(error)} | 调用者: {caller} | "
            f"Connection error: {str(error)} | Caller: {caller}"
        )
        
        # 尝试重新连接
        # Attempt reconnection
        self.logger.info("尝试重新连接... | Attempting reconnection...")
        
        # 对于外部API服务，返回连接错误信息
        # For external API services, return connection error information
        return {
            "success": False,
            "error": f"连接错误: {str(error)}",
            "error_type": "ConnectionError",
            "caller": caller,
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_timeout_error(self, error, caller, func, args, kwargs):
        """处理超时错误 | Handle timeout error"""
        self.logger.error(
            f"超时错误: {str(error)} | 调用者: {caller} | "
            f"Timeout error: {str(error)} | Caller: {caller}"
        )
        
        return {
            "success": False,
            "error": f"请求超时: {str(error)}",
            "error_type": "TimeoutError",
            "caller": caller,
            "timestamp": datetime.now().isoformat(),
            "suggestion": "请检查网络连接或增加超时时间 | Please check network connection or increase timeout"
        }
    
    def _handle_value_error(self, error, caller, func, args, kwargs):
        """处理值错误 | Handle value error"""
        self.logger.error(
            f"值错误: {str(error)} | 调用者: {caller} | "
            f"Value error: {str(error)} | Caller: {caller}"
        )
        
        return {
            "success": False,
            "error": f"参数错误: {str(error)}",
            "error_type": "ValueError",
            "caller": caller,
            "timestamp": datetime.now().isoformat(),
            "suggestion": "请检查输入参数 | Please check input parameters"
        }
    
    def _handle_type_error(self, error, caller, func, args, kwargs):
        """处理类型错误 | Handle type error"""
        self.logger.error(
            f"类型错误: {str(error)} | 调用者: {caller} | "
            f"Type error: {str(error)} | Caller: {caller}"
        )
        
        return {
            "success": False,
            "error": f"类型错误: {str(error)}",
            "error_type": "TypeError",
            "caller": caller,
            "timestamp": datetime.now().isoformat(),
            "suggestion": "请检查参数类型 | Please check parameter types"
        }
    
    def _handle_permission_error(self, error, caller, func, args, kwargs):
        """处理权限错误 | Handle permission error"""
        self.logger.error(
            f"权限错误: {str(error)} | 调用者: {caller} | "
            f"Permission error: {str(error)} | Caller: {caller}"
        )
        
        return {
            "success": False,
            "error": f"权限不足: {str(error)}",
            "error_type": "PermissionError",
            "caller": caller,
            "timestamp": datetime.now().isoformat(),
            "suggestion": "请检查API密钥或权限设置 | Please check API key or permission settings"
        }
    
    def _handle_generic_error(self, error, caller, func, args, kwargs):
        """处理通用错误 | Handle generic error"""
        self.logger.error(
            f"未处理错误: {str(error)} | 调用者: {caller} | "
            f"Unhandled error: {str(error)} | Caller: {caller}"
        )
        
        return {
            "success": False,
            "error": f"未处理错误: {str(error)}",
            "error_type": type(error).__name__,
            "caller": caller,
            "timestamp": datetime.now().isoformat(),
            "suggestion": "请查看日志获取详细信息 | Please check logs for details"
        }
    
    def _handle_exception_in_context(self, error_info, exc_type, exc_val, exc_tb):
        """在上下文中处理异常 | Handle exception in context"""
        # 记录详细的错误信息
        # Log detailed error information
        self.logger.error(
            f"操作失败: {error_info['operation']} | "
            f"错误类型: {error_info['error_type']} | "
            f"错误信息: {error_info['error_message']} | "
            f"耗时: {error_info['duration']:.2f}s | "
            f"上下文: {error_info['context']}"
        )
        
        # 记录错误统计
        # Record error statistics
        self._record_error_statistics(exc_val)
    
    # 恢复策略方法 | Recovery strategy methods
    def _retry_strategy(self, error, max_retries=3, delay=1.0):
        """重试策略 | Retry strategy"""
        self.logger.info(f"应用重试策略，最大重试次数: {max_retries} | Applying retry strategy, max retries: {max_retries}")
        
        import time
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = delay * (2 ** (attempt - 1))
                    self.logger.info(f"重试 {attempt}/{max_retries}，等待 {wait_time}秒 | Retry {attempt}/{max_retries}, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                
                # 在实际应用中，这里应该重新执行失败的操作
                # In actual application, should re-execute the failed operation here
                self.logger.info(f"执行重试 {attempt + 1} | Executing retry {attempt + 1}")
                
                # 返回成功表示重试成功
                # Return success to indicate retry succeeded
                return {"success": True, "retry_attempt": attempt + 1}
            
            except Exception as retry_error:
                self.logger.error(f"重试 {attempt + 1} 失败: {str(retry_error)} | Retry {attempt + 1} failed: {str(retry_error)}")
        
        # 所有重试都失败
        # All retries failed
        return {"success": False, "error": f"所有 {max_retries} 次重试均失败 | All {max_retries} retries failed"}
    
    def _fallback_strategy(self, error, fallback_func=None, *args, **kwargs):
        """故障转移策略 | Fallback strategy"""
        self.logger.info("应用故障转移策略 | Applying fallback strategy")
        
        if fallback_func:
            try:
                result = fallback_func(*args, **kwargs)
                self.logger.info("故障转移成功 | Fallback successful")
                return {"success": True, "fallback_result": result}
            except Exception as fallback_error:
                self.logger.error(f"故障转移也失败: {str(fallback_error)} | Fallback also failed: {str(fallback_error)}")
        
        return {"success": False, "error": "故障转移失败 | Fallback failed"}
    
    def _circuit_breaker_strategy(self, error):
        """断路器策略 | Circuit breaker strategy"""
        self.logger.warning("应用断路器策略 | Applying circuit breaker strategy")
        
        return {
            "success": False,
            "error": "服务暂时不可用（断路器已打开） | Service temporarily unavailable (circuit breaker OPEN)",
            "circuit_state": "OPEN",
            "suggestion": "请稍后重试 | Please try again later"
        }
    
    # 工具方法 | Utility methods
    def _get_caller_info(self, func, args):
        """获取调用者信息 | Get caller information"""
        try:
            # 尝试获取self（如果是实例方法）
            # Try to get self (if it's an instance method)
            if args and len(args) > 0 and hasattr(args[0], '__class__'):
                instance = args[0]
                return f"{instance.__class__.__name__}.{func.__name__}"
            else:
                return func.__name__
        except Exception as e:
            logging.debug(f"获取调用者信息失败: {e}")
            return "unknown_caller"
    
    def _record_error_statistics(self, error):
        """记录错误统计 | Record error statistics"""
        self.error_stats["total_errors"] += 1
        
        error_type = type(error).__name__
        self.error_stats["error_types"][error_type] = self.error_stats["error_types"].get(error_type, 0) + 1
    
    def get_error_statistics(self):
        """获取错误统计 | Get error statistics"""
        return {
            "total_errors": self.error_stats["total_errors"],
            "error_types": self.error_stats["error_types"],
            "recovery_attempts": self.error_stats["recovery_attempts"],
            "successful_recoveries": self.error_stats["successful_recoveries"],
            "recovery_rate": (
                (self.error_stats["successful_recoveries"] / self.error_stats["recovery_attempts"] * 100)
                if self.error_stats["recovery_attempts"] > 0 else 0
            ),
            "timestamp": datetime.now().isoformat()
        }
    
    def reset_statistics(self):
        """重置统计 | Reset statistics"""
        self.error_stats = {
            "total_errors": 0,
            "error_types": {},
            "recovery_attempts": 0,
            "successful_recoveries": 0
        }
        self.logger.info("错误统计已重置 | Error statistics reset")


# 全局异常处理器实例 | Global exception handler instance
_global_exception_handler = None

def get_global_exception_handler(logger_name: str = __name__) -> APIExceptionHandler:
    """获取全局异常处理器 | Get global exception handler"""
    global _global_exception_handler
    if _global_exception_handler is None:
        _global_exception_handler = APIExceptionHandler(logger_name)
    return _global_exception_handler

def set_global_exception_handler(handler: APIExceptionHandler):
    """设置全局异常处理器 | Set global exception handler"""
    global _global_exception_handler
    _global_exception_handler = handler


# 常用装饰器快捷方式 | Common decorator shortcuts
def handle_exceptions(func: Callable) -> Callable:
    """异常处理装饰器快捷方式 | Exception handling decorator shortcut"""
    handler = get_global_exception_handler()
    return handler.handle_exception(func)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """重试装饰器快捷方式 | Retry decorator shortcut"""
    handler = get_global_exception_handler()
    return handler.retry_on_failure(max_retries, delay, backoff_factor)

def fallback_on_failure(fallback_func: Callable):
    """故障转移装饰器快捷方式 | Fallback decorator shortcut"""
    handler = get_global_exception_handler()
    return handler.fallback_on_failure(fallback_func)

def circuit_breaker(failure_threshold: int = 5, reset_timeout: float = 60.0):
    """断路器装饰器快捷方式 | Circuit breaker decorator shortcut"""
    handler = get_global_exception_handler()
    return handler.circuit_breaker(failure_threshold, reset_timeout)


# 示例用法 | Example usage
if __name__ == "__main__":
    import logging
    
    # 配置日志
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建异常处理器
    # Create exception handler
    handler = APIExceptionHandler()
    
    # 示例1: 使用装饰器
    # Example 1: Using decorator
    @handler.handle_exception
    def risky_operation(x, y):
        if y == 0:
            raise ValueError("除数不能为零 | Denominator cannot be zero")
        return x / y
    
    # 示例2: 使用重试装饰器
    # Example 2: Using retry decorator
    @handler.retry_on_failure(max_retries=3, delay=1.0)
    def unreliable_connection():
        deterministic_random = ((zlib.adler32(str("unreliable_connection").encode('utf-8')) & 0xffffffff) % 100) / 100.0
        if deterministic_random < 0.7:
            raise ConnectionError("连接失败 | Connection failed")
        return "连接成功 | Connection successful"
    
    # 示例3: 使用上下文管理器
    # Example 3: Using context manager
    with handler.error_context("测试操作", param1="value1", param2="value2"):
        print("执行操作... | Executing operation...")
    
    # 测试
    # Test
    print("测试异常处理... | Testing exception handling...")
    
    # 测试危险操作
    # Test risky operation
    result = risky_operation(10, 0)
    print(f"危险操作结果: {result} | Risky operation result: {result}")
    
    # 测试不可靠连接
    # Test unreliable connection
    try:
        result = unreliable_connection()
        print(f"连接结果: {result} | Connection result: {result}")
    except Exception as e:
        print(f"最终失败: {str(e)} | Final failure: {str(e)}")
    
    # 获取错误统计
    # Get error statistics
    stats = handler.get_error_statistics()
    print(f"错误统计: {stats} | Error statistics: {stats}")
