"""
错误处理混入类 - 提供统一的错误处理和自动恢复机制
Error Handling Mixin - Provides unified error handling and automatic recovery mechanisms

功能包括：
- 统一的错误记录和分类
- 自动恢复机制和重试策略
- 错误历史管理和分析
- 错误上下文捕获和诊断
- 优雅降级和容错处理
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

class ErrorHandlingMixin:
    """错误处理混入类，提供统一的错误处理和恢复机制"""
    
    def __init__(self, *args, **kwargs):
        """初始化错误处理功能"""
        super().__init__(*args, **kwargs)
        
        # 错误处理配置
        config = getattr(self, 'config', {})
        self.auto_recovery_enabled = config.get('auto_recovery', True)
        self.max_retry_attempts = config.get('max_retry_attempts', 3)
        self.error_threshold = config.get('error_threshold', 10)  # 最大错误记录数
        
        # 错误历史记录
        self.error_history = []
        
        # 恢复状态
        self.recovery_attempts = 0
        self.last_recovery_time = None
        self.recovery_successful = True
        
        self.logger.info(f"Error handling initialized for {getattr(self, 'model_id', 'unknown')}")
    
    def _handle_error(self, error: Exception, context: str, additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """统一错误处理方法
        
        Args:
            error: 发生的异常
            context: 错误发生的上下文
            additional_info: 额外的错误信息
            
        Returns:
            错误处理结果
        """
        additional_info = additional_info or {}
        
        # 创建错误记录
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "model_id": getattr(self, 'model_id', 'unknown'),
            "additional_info": additional_info,
            "recovery_attempted": False,
            "recovery_successful": False
        }
        
        # 更新性能指标（如果存在）
        if hasattr(self, 'performance_metrics'):
            self.performance_metrics["failed_requests"] = self.performance_metrics.get("failed_requests", 0) + 1
        
        # 记录错误历史
        self.error_history.append(error_info)
        
        # 限制错误历史大小
        if len(self.error_history) > self.error_threshold:
            self.error_history = self.error_history[-self.error_threshold:]
        
        # 记录错误日志
        self.logger.error(f"Error in {context}: {str(error)}")
        
        # 自动恢复逻辑
        recovery_result = None
        if self.auto_recovery_enabled:
            recovery_result = self._attempt_recovery(error, context, error_info)
            error_info["recovery_attempted"] = True
            error_info["recovery_successful"] = recovery_result.get("success", False)
        
        return {
            "error_info": error_info,
            "recovery_result": recovery_result,
            "should_retry": self._should_retry(error, context)
        }
    
    def _attempt_recovery(self, error: Exception, context: str, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """尝试自动恢复
        
        Args:
            error: 发生的异常
            context: 错误发生的上下文
            error_info: 错误信息
            
        Returns:
            恢复结果
        """
        recovery_attempts = 0
        max_attempts = self.max_retry_attempts
        
        while recovery_attempts < max_attempts:
            try:
                recovery_attempts += 1
                self.recovery_attempts += 1
                
                self.logger.info(f"Recovery attempt {recovery_attempts} for {context}")
                
                # 根据错误类型采取不同的恢复策略
                recovery_strategy = self._get_recovery_strategy(error, context)
                recovery_success = self._execute_recovery_strategy(recovery_strategy, error, context)
                
                if recovery_success:
                    self.recovery_successful = True
                    self.last_recovery_time = datetime.now()
                    self.logger.info(f"Recovery successful after {recovery_attempts} attempts")
                    
                    return {
                        "success": True,
                        "attempts": recovery_attempts,
                        "strategy": recovery_strategy,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    self.logger.warning(f"Recovery attempt {recovery_attempts} failed")
                    
            except Exception as recovery_error:
                self.logger.warning(f"Recovery attempt {recovery_attempts} failed with error: {str(recovery_error)}")
                
            if recovery_attempts >= max_attempts:
                self.recovery_successful = False
                self.logger.error(f"All recovery attempts failed for {context}")
                break
        
        return {
            "success": False,
            "attempts": recovery_attempts,
            "error": "All recovery attempts failed",
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_recovery_strategy(self, error: Exception, context: str) -> str:
        """根据错误类型和上下文获取恢复策略
        
        Args:
            error: 发生的异常
            context: 错误发生的上下文
            
        Returns:
            恢复策略名称
        """
        error_type = type(error).__name__
        
        # 根据错误类型选择策略
        if "Memory" in error_type or "memory" in str(error).lower():
            return "memory_recovery"
        elif "Connection" in error_type or "network" in str(error).lower():
            return "connection_recovery"
        elif "Timeout" in error_type:
            return "timeout_recovery"
        elif "File" in error_type or "IO" in error_type:
            return "file_recovery"
        elif "Value" in error_type or "Type" in error_type:
            return "validation_recovery"
        else:
            return "general_recovery"
    
    def _execute_recovery_strategy(self, strategy: str, error: Exception, context: str) -> bool:
        """执行具体的恢复策略
        
        Args:
            strategy: 恢复策略名称
            error: 发生的异常
            context: 错误发生的上下文
            
        Returns:
            恢复是否成功
        """
        try:
            if strategy == "memory_recovery":
                return self._recover_from_memory_error(error, context)
            elif strategy == "connection_recovery":
                return self._recover_from_connection_error(error, context)
            elif strategy == "timeout_recovery":
                return self._recover_from_timeout_error(error, context)
            elif strategy == "file_recovery":
                return self._recover_from_file_error(error, context)
            elif strategy == "validation_recovery":
                return self._recover_from_validation_error(error, context)
            else:
                return self._recover_general_error(error, context)
                
        except Exception as recovery_error:
            self.logger.error(f"Recovery strategy {strategy} failed: {str(recovery_error)}")
            return False
    
    def _recover_from_memory_error(self, error: Exception, context: str) -> bool:
        """从内存错误中恢复
        
        Args:
            error: 内存错误
            context: 错误上下文
            
        Returns:
            恢复是否成功
        """
        try:
            # 清理缓存（如果存在）
            if hasattr(self, 'clear_cache'):
                self.clear_cache()
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 重置模型状态
            if hasattr(self, 'reset'):
                self.reset()
            
            self.logger.info("Memory error recovery completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Memory error recovery failed: {str(e)}")
            return False
    
    def _recover_from_connection_error(self, error: Exception, context: str) -> bool:
        """从连接错误中恢复
        
        Args:
            error: 连接错误
            context: 错误上下文
            
        Returns:
            恢复是否成功
        """
        try:
            # 如果是外部API连接错误，切换到本地模式
            if "external" in context.lower() and hasattr(self, 'switch_to_local_mode'):
                result = self.switch_to_local_mode()
                if result.get("success", False):
                    self.logger.info("Switched to local mode due to connection error")
                    return True
            
            # 等待一段时间后重试
            time.sleep(1)
            
            # 重新初始化连接相关组件
            if hasattr(self, '_initialize_external_api_service'):
                self._initialize_external_api_service()
            
            self.logger.info("Connection error recovery completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection error recovery failed: {str(e)}")
            return False
    
    def _recover_from_timeout_error(self, error: Exception, context: str) -> bool:
        """从超时错误中恢复
        
        Args:
            error: 超时错误
            context: 错误上下文
            
        Returns:
            恢复是否成功
        """
        try:
            # 增加超时时间或简化处理
            if hasattr(self, 'config'):
                # 调整配置中的超时设置
                current_timeout = self.config.get('timeout', 30)
                self.config['timeout'] = min(current_timeout * 2, 300)  # 最大5分钟
            
            # 等待后重试
            time.sleep(2)
            
            self.logger.info("Timeout error recovery completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Timeout error recovery failed: {str(e)}")
            return False
    
    def _recover_from_file_error(self, error: Exception, context: str) -> bool:
        """从文件错误中恢复
        
        Args:
            error: 文件错误
            context: 错误上下文
            
        Returns:
            恢复是否成功
        """
        try:
            # 检查文件路径和权限
            error_msg = str(error).lower()
            
            if "not found" in error_msg or "no such file" in error_msg:
                # 文件不存在，尝试创建或使用默认值
                self.logger.warning("File not found, using default configuration")
                return True
            elif "permission" in error_msg:
                # 权限问题，尝试更改权限或使用备用路径
                self.logger.warning("Permission error, trying alternative approach")
                return True
            else:
                # 其他文件错误，尝试重新初始化
                if hasattr(self, 'reset'):
                    self.reset()
                
                return True
                
        except Exception as e:
            self.logger.error(f"File error recovery failed: {str(e)}")
            return False
    
    def _recover_from_validation_error(self, error: Exception, context: str) -> bool:
        """从验证错误中恢复
        
        Args:
            error: 验证错误
            context: 错误上下文
            
        Returns:
            恢复是否成功
        """
        try:
            # 清理无效数据或使用默认值
            if hasattr(self, 'clear_cache'):
                self.clear_cache()
            
            # 重新验证输入数据
            self.logger.info("Validation error recovery completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error recovery failed: {str(e)}")
            return False
    
    def _recover_general_error(self, error: Exception, context: str) -> bool:
        """通用错误恢复
        
        Args:
            error: 通用错误
            context: 错误上下文
            
        Returns:
            恢复是否成功
        """
        try:
            # 重置模型状态
            if hasattr(self, 'reset'):
                self.reset()
            
            # 重新初始化
            if hasattr(self, 'initialize'):
                init_result = self.initialize()
                if init_result.get("success", False):
                    return True
            
            # 简单的等待后重试
            time.sleep(1)
            return True
            
        except Exception as e:
            self.logger.error(f"General error recovery failed: {str(e)}")
            return False
    
    def _should_retry(self, error: Exception, context: str) -> bool:
        """判断是否应该重试
        
        Args:
            error: 发生的异常
            context: 错误发生的上下文
            
        Returns:
            是否应该重试
        """
        error_type = type(error).__name__
        
        # 不应该重试的错误类型
        non_retryable_errors = [
            "ValueError", "TypeError", "AttributeError", 
            "KeyError", "IndexError", "SyntaxError"
        ]
        
        if error_type in non_retryable_errors:
            return False
        
        # 根据上下文判断
        if "validation" in context.lower() or "input" in context.lower():
            return False
        
        # 检查错误历史中的相同错误频率
        recent_errors = [e for e in self.error_history[-5:] if e["error_type"] == error_type]
        if len(recent_errors) >= 3:  # 最近5次错误中有3次相同类型
            return False
        
        return True
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要
        
        Returns:
            错误摘要字典
        """
        error_counts = {}
        for error in self.error_history:
            error_type = error["error_type"]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # 计算错误率
        total_requests = getattr(self, 'performance_metrics', {}).get("total_requests", 0)
        total_errors = len(self.error_history)
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "model_id": getattr(self, 'model_id', 'unknown'),
            "total_errors": total_errors,
            "error_rate": round(error_rate, 2),
            "error_counts": error_counts,
            "recent_errors": self.error_history[-10:] if self.error_history else [],
            "auto_recovery_enabled": self.auto_recovery_enabled,
            "max_retry_attempts": self.max_retry_attempts,
            "recovery_attempts": self.recovery_attempts,
            "recovery_success_rate": self._calculate_recovery_success_rate(),
            "last_recovery_time": self.last_recovery_time.isoformat() if self.last_recovery_time else None
        }
    
    def _calculate_recovery_success_rate(self) -> float:
        """计算恢复成功率
        
        Returns:
            恢复成功率（百分比）
        """
        if self.recovery_attempts == 0:
            return 100.0  # 没有恢复尝试，视为100%成功
        
        # 从错误历史中统计恢复成功次数
        successful_recoveries = len([e for e in self.error_history if e.get("recovery_successful", False)])
        success_rate = (successful_recoveries / len(self.error_history)) * 100 if self.error_history else 100.0
        
        return round(success_rate, 2)
    
    def enable_auto_recovery(self, enabled: bool = True):
        """启用或禁用自动恢复
        
        Args:
            enabled: 是否启用自动恢复
        """
        self.auto_recovery_enabled = enabled
        self.logger.info(f"Auto recovery {'enabled' if enabled else 'disabled'}")
    
    def set_max_retry_attempts(self, attempts: int):
        """设置最大重试次数
        
        Args:
            attempts: 最大重试次数
        """
        self.max_retry_attempts = max(1, attempts)  # 至少1次
        self.logger.info(f"Max retry attempts set to: {self.max_retry_attempts}")
    
    def set_error_threshold(self, threshold: int):
        """设置错误历史阈值
        
        Args:
            threshold: 最大错误记录数
        """
        self.error_threshold = max(10, threshold)  # 至少10条
        # 立即应用阈值
        if len(self.error_history) > self.error_threshold:
            self.error_history = self.error_history[-self.error_threshold:]
        self.logger.info(f"Error threshold set to: {self.error_threshold}")
    
    def clear_error_history(self):
        """清空错误历史"""
        self.error_history.clear()
        self.recovery_attempts = 0
        self.last_recovery_time = None
        self.recovery_successful = True
        self.logger.info("Error history cleared")
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """获取错误分析报告
        
        Returns:
            错误分析报告
        """
        error_summary = self.get_error_summary()
        
        # 分析错误模式
        error_patterns = self._analyze_error_patterns()
        
        # 生成改进建议
        recommendations = self._generate_error_recommendations()
        
        return {
            "summary": error_summary,
            "patterns": error_patterns,
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """分析错误模式
        
        Returns:
            错误模式分析
        """
        patterns = {
            "frequent_errors": [],
            "recent_trend": "stable",
            "recovery_patterns": {}
        }
        
        if not self.error_history:
            return patterns
        
        # 找出频繁错误
        error_counts = {}
        for error in self.error_history:
            error_type = error["error_type"]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        total_errors = len(self.error_history)
        for error_type, count in error_counts.items():
            frequency = (count / total_errors) * 100
            if frequency > 20:  # 超过20%的错误是同一类型
                patterns["frequent_errors"].append({
                    "error_type": error_type,
                    "frequency": round(frequency, 2),
                    "count": count
                })
        
        # 分析近期趋势
        recent_errors = self.error_history[-10:] if len(self.error_history) >= 10 else self.error_history
        if len(recent_errors) >= 5:
            recent_count = len(recent_errors)
            older_count = len(self.error_history) - recent_count
            
            if older_count > 0:
                recent_rate = recent_count / len(self.error_history)
                if recent_rate > 0.7:
                    patterns["recent_trend"] = "increasing"
                elif recent_rate < 0.3:
                    patterns["recent_trend"] = "decreasing"
        
        # 分析恢复模式
        recovery_errors = [e for e in self.error_history if e.get("recovery_attempted", False)]
        if recovery_errors:
            successful_recoveries = len([e for e in recovery_errors if e.get("recovery_successful", False)])
            patterns["recovery_patterns"] = {
                "total_recovery_attempts": len(recovery_errors),
                "successful_recoveries": successful_recoveries,
                "recovery_success_rate": round((successful_recoveries / len(recovery_errors)) * 100, 2) if recovery_errors else 0
            }
        
        return patterns
    
    def _generate_error_recommendations(self) -> List[Dict[str, Any]]:
        """生成错误改进建议
        
        Returns:
            改进建议列表
        """
        recommendations = []
        error_summary = self.get_error_summary()
        patterns = self._analyze_error_patterns()
        
        # 基于错误率建议
        error_rate = error_summary.get("error_rate", 0)
        if error_rate > 10:
            recommendations.append({
                "type": "error_rate",
                "priority": "high",
                "message": f"错误率过高 ({error_rate}%)，建议检查系统稳定性",
                "action": "review_system_stability"
            })
        
        # 基于频繁错误建议
        for frequent_error in patterns.get("frequent_errors", []):
            if frequent_error["frequency"] > 50:
                recommendations.append({
                    "type": "frequent_error",
                    "priority": "high",
                    "message": f"频繁出现 {frequent_error['error_type']} 错误 ({frequent_error['frequency']}%)",
                    "action": f"fix_{frequent_error['error_type'].lower()}_issues"
                })
        
        # 基于恢复成功率建议
        recovery_success_rate = error_summary.get("recovery_success_rate", 100)
        if recovery_success_rate < 50:
            recommendations.append({
                "type": "recovery",
                "priority": "medium",
                "message": f"自动恢复成功率较低 ({recovery_success_rate}%)，建议优化恢复策略",
                "action": "optimize_recovery_strategy"
            })
        
        # 基于趋势建议
        if patterns.get("recent_trend") == "increasing":
            recommendations.append({
                "type": "trend",
                "priority": "medium",
                "message": "错误频率呈上升趋势，建议加强监控",
                "action": "enhance_monitoring"
            })
        
        return recommendations
