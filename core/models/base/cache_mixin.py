"""
缓存和优化混入类 - 提供智能缓存和性能优化功能
Cache and Optimization Mixin - Provides intelligent caching and performance optimization

功能包括：
- 智能缓存管理和TTL控制
- 缓存键生成和过期清理
- 性能优化策略（内存、CPU、响应时间）
- 资源使用监控和自动优化
- 缓存统计和清理机制
"""

import time
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime

class CacheMixin:
    """缓存和优化混入类，提供智能缓存和性能优化功能"""
    
    def __init__(self, *args, **kwargs):
        """初始化缓存和优化功能"""
        super().__init__(*args, **kwargs)
        
        # 缓存配置
        self.cache_enabled = getattr(self, 'config', {}).get('cache_enabled', True)
        self.cache_ttl = getattr(self, 'config', {}).get('cache_ttl', 300)  # 5分钟默认值
        self.cache = {}
        
        # 性能优化配置
        self.optimization_enabled = getattr(self, 'config', {}).get('optimization_enabled', True)
        self.performance_thresholds = {
            "memory_threshold": getattr(self, 'config', {}).get('memory_threshold', 1000000000),  # 1GB
            "cpu_threshold": getattr(self, 'config', {}).get('cpu_threshold', 80),  # 80%
            "response_threshold": getattr(self, 'config', {}).get('response_threshold', 5.0)  # 5秒
        }
        
        # 缓存统计
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
            "cache_size": 0,
            "last_cleanup": datetime.now()
        }
        
        self.logger.info(f"Cache and optimization initialized for {getattr(self, 'model_id', 'unknown')}")
    
    def _get_cache_key(self, data: Any) -> str:
        """生成缓存键
        
        Args:
            data: 要缓存的数据
            
        Returns:
            缓存键字符串
        """
        try:
            data_str = str(data).encode('utf-8')
            return hashlib.md5(data_str).hexdigest()
        except Exception as e:
            self.logger.warning(f"Cache key generation failed: {str(e)}")
            return str(hash(str(data)))
    
    def get_cached_result(self, data: Any) -> Optional[Dict[str, Any]]:
        """从缓存中获取结果
        
        Args:
            data: 要查询的数据
            
        Returns:
            缓存结果或None
        """
        if not self.cache_enabled:
            self.cache_stats["misses"] += 1
            self.cache_stats["total_requests"] += 1
            return None
        
        cache_key = self._get_cache_key(data)
        cached_item = self.cache.get(cache_key)
        
        self.cache_stats["total_requests"] += 1
        
        if cached_item:
            # 检查缓存是否过期
            current_time = time.time()
            if current_time - cached_item['timestamp'] < self.cache_ttl:
                self.cache_stats["hits"] += 1
                self.logger.debug(f"Cache hit for key: {cache_key}")
                return cached_item['result']
            else:
                # 缓存过期，删除
                del self.cache[cache_key]
                self.cache_stats["misses"] += 1
                self.logger.debug(f"Cache expired for key: {cache_key}")
        else:
            self.cache_stats["misses"] += 1
        
        return None
    
    def set_cached_result(self, data: Any, result: Dict[str, Any]):
        """设置缓存结果
        
        Args:
            data: 原始数据
            result: 要缓存的结果
        """
        if not self.cache_enabled:
            return
        
        cache_key = self._get_cache_key(data)
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time(),
            'size': len(str(result))
        }
        
        # 更新缓存大小统计
        self.cache_stats["cache_size"] = len(self.cache)
        
        # 定期清理过期缓存
        if len(self.cache) % 10 == 0:  # 每10次缓存操作清理一次
            self._cleanup_expired_cache()
    
    def _cleanup_expired_cache(self):
        """清理过期的缓存项"""
        current_time = time.time()
        expired_keys = []
        
        for key, item in self.cache.items():
            if current_time - item['timestamp'] > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            self.cache_stats["cache_size"] = len(self.cache)
            self.cache_stats["last_cleanup"] = datetime.now()
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.cache_stats.update({
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
            "cache_size": 0,
            "last_cleanup": datetime.now()
        })
        self.logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
        
        Returns:
            缓存统计字典
        """
        total_requests = self.cache_stats["total_requests"]
        hits = self.cache_stats["hits"]
        misses = self.cache_stats["misses"]
        
        hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "cache_size": self.cache_stats["cache_size"],
            "total_requests": total_requests,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hit_rate, 2),
            "last_cleanup": self.cache_stats["last_cleanup"].isoformat(),
            "optimization_enabled": self.optimization_enabled
        }
    
    def enable_cache(self, enabled: bool = True):
        """启用或禁用缓存
        
        Args:
            enabled: 是否启用缓存
        """
        self.cache_enabled = enabled
        if not enabled:
            self.clear_cache()
        self.logger.info(f"Cache {'enabled' if enabled else 'disabled'}")
    
    def set_cache_ttl(self, ttl_seconds: int):
        """设置缓存TTL（生存时间）
        
        Args:
            ttl_seconds: TTL秒数
        """
        self.cache_ttl = max(1, ttl_seconds)  # 至少1秒
        self.logger.info(f"Cache TTL set to {self.cache_ttl} seconds")
    
    def optimize_performance(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """优化模型性能基于当前指标
        
        Args:
            config: 优化配置
            
        Returns:
            优化结果字典
        """
        if not self.optimization_enabled:
            return {"success": False, "error": "Performance optimization is disabled"}
        
        config = config or {}
        optimizations = []
        
        try:
            # 获取当前性能指标（如果存在）
            performance_metrics = getattr(self, 'performance_metrics', {})
            resource_usage = getattr(self, 'resource_usage', {})
            
            # 内存优化
            memory_threshold = config.get("memory_threshold", self.performance_thresholds["memory_threshold"])
            peak_memory = performance_metrics.get("peak_memory_usage", 0)
            current_memory = resource_usage.get("memory_usage", 0)
            
            if peak_memory > memory_threshold or current_memory > memory_threshold:
                optimizations.append("memory_optimization")
                self._optimize_memory_usage()
            
            # CPU优化
            cpu_threshold = config.get("cpu_threshold", self.performance_thresholds["cpu_threshold"])
            cpu_utilization = performance_metrics.get("cpu_utilization", 0)
            
            if cpu_utilization > cpu_threshold:
                optimizations.append("cpu_optimization")
                self._optimize_cpu_usage()
            
            # 响应时间优化
            response_threshold = config.get("response_threshold", self.performance_thresholds["response_threshold"])
            avg_response_time = performance_metrics.get("average_response_time", 0)
            
            if avg_response_time > response_threshold:
                optimizations.append("response_time_optimization")
                self._optimize_response_time()
            
            # 缓存优化
            if self.cache_enabled:
                optimizations.append("cache_optimization")
                self._optimize_cache_usage()
            
            return {
                "success": True,
                "optimizations_applied": optimizations,
                "performance_metrics": performance_metrics,
                "cache_stats": self.get_cache_stats()
            }
            
        except Exception as e:
            self.logger.error(f"Performance optimization failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _optimize_memory_usage(self):
        """优化内存使用"""
        try:
            # 清理缓存
            if self.cache_enabled:
                old_size = len(self.cache)
                self.clear_cache()
                self.logger.info(f"Memory optimization: cleared cache with {old_size} items")
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 清理其他可能的内存占用
            if hasattr(self, 'cached_data'):
                old_size = len(getattr(self, 'cached_data', {}))
                getattr(self, 'cached_data', {}).clear()
                self.logger.info(f"Memory optimization: cleared cached data with {old_size} items")
            
            self.logger.info("Memory usage optimization completed")
            
        except Exception as e:
            self.logger.warning(f"Memory optimization partially failed: {str(e)}")
    
    def _optimize_cpu_usage(self):
        """优化CPU使用"""
        try:
            # 减少并行处理（如果适用）
            # 这是模型特定优化的占位符
            
            # 调整缓存策略以减少计算
            if self.cache_enabled:
                # 增加缓存TTL以减少重复计算
                original_ttl = self.cache_ttl
                self.cache_ttl = min(3600, self.cache_ttl * 2)  # 最大1小时
                self.logger.info(f"CPU optimization: increased cache TTL from {original_ttl} to {self.cache_ttl} seconds")
            
            self.logger.info("CPU usage optimization completed")
            
        except Exception as e:
            self.logger.warning(f"CPU optimization partially failed: {str(e)}")
    
    def _optimize_response_time(self):
        """优化响应时间"""
        try:
            # 实现缓存或其他响应时间优化
            if not self.cache_enabled:
                self.enable_cache(True)
                self.logger.info("Response time optimization: enabled caching")
            
            # 调整缓存策略
            if self.cache_ttl < 60:  # 如果TTL太小，增加它
                original_ttl = self.cache_ttl
                self.cache_ttl = 300  # 5分钟
                self.logger.info(f"Response time optimization: increased cache TTL from {original_ttl} to {self.cache_ttl} seconds")
            
            self.logger.info("Response time optimization completed")
            
        except Exception as e:
            self.logger.warning(f"Response time optimization partially failed: {str(e)}")
    
    def _optimize_cache_usage(self):
        """优化缓存使用"""
        try:
            current_stats = self.get_cache_stats()
            hit_rate = current_stats.get("hit_rate", 0)
            cache_size = current_stats.get("cache_size", 0)
            
            # 根据命中率调整缓存策略
            if hit_rate < 30:  # 低命中率
                # 减少缓存大小或禁用缓存
                if cache_size > 100:
                    self.clear_cache()
                    self.logger.info("Cache optimization: cleared cache due to low hit rate")
            
            elif hit_rate > 80:  # 高命中率
                # 增加缓存TTL
                if self.cache_ttl < 1800:  # 30分钟
                    original_ttl = self.cache_ttl
                    self.cache_ttl = min(3600, self.cache_ttl * 1.5)  # 增加50%，最大1小时
                    self.logger.info(f"Cache optimization: increased TTL from {original_ttl} to {self.cache_ttl} due to high hit rate")
            
            self.logger.info("Cache usage optimization completed")
            
        except Exception as e:
            self.logger.warning(f"Cache optimization partially failed: {str(e)}")
    
    def get_performance_recommendations(self) -> Dict[str, Any]:
        """获取性能优化建议
        
        Returns:
            优化建议字典
        """
        recommendations = []
        
        # 获取当前统计
        cache_stats = self.get_cache_stats()
        performance_metrics = getattr(self, 'performance_metrics', {})
        
        # 缓存建议
        hit_rate = cache_stats.get("hit_rate", 0)
        if hit_rate < 30 and self.cache_enabled:
            recommendations.append({
                "type": "cache",
                "priority": "high",
                "message": "缓存命中率低，考虑调整缓存策略或禁用缓存",
                "action": "review_cache_strategy"
            })
        elif hit_rate > 80 and self.cache_ttl < 600:
            recommendations.append({
                "type": "cache",
                "priority": "medium",
                "message": "缓存命中率高，考虑增加缓存TTL以提高性能",
                "action": "increase_cache_ttl"
            })
        
        # 内存建议
        memory_usage = performance_metrics.get("peak_memory_usage", 0)
        if memory_usage > self.performance_thresholds["memory_threshold"]:
            recommendations.append({
                "type": "memory",
                "priority": "high",
                "message": "内存使用过高，建议优化内存使用",
                "action": "optimize_memory"
            })
        
        # CPU建议
        cpu_usage = performance_metrics.get("cpu_utilization", 0)
        if cpu_usage > self.performance_thresholds["cpu_threshold"]:
            recommendations.append({
                "type": "cpu",
                "priority": "medium",
                "message": "CPU使用率较高，建议优化计算效率",
                "action": "optimize_cpu"
            })
        
        # 响应时间建议
        response_time = performance_metrics.get("average_response_time", 0)
        if response_time > self.performance_thresholds["response_threshold"]:
            recommendations.append({
                "type": "response_time",
                "priority": "high",
                "message": "响应时间较长，建议优化处理流程",
                "action": "optimize_response_time"
            })
        
        return {
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "high_priority": len([r for r in recommendations if r["priority"] == "high"]),
            "cache_stats": cache_stats,
            "performance_metrics": performance_metrics
        }
    
    def apply_recommendation(self, recommendation_type: str, action: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """应用性能优化建议
        
        Args:
            recommendation_type: 建议类型
            action: 要执行的动作
            parameters: 动作参数
            
        Returns:
            应用结果
        """
        parameters = parameters or {}
        
        try:
            if recommendation_type == "cache":
                if action == "increase_cache_ttl":
                    new_ttl = parameters.get("ttl", self.cache_ttl * 2)
                    self.set_cache_ttl(new_ttl)
                    return {"success": True, "message": f"Cache TTL increased to {new_ttl} seconds"}
                
                elif action == "review_cache_strategy":
                    self.enable_cache(parameters.get("enabled", False))
                    return {"success": True, "message": "Cache strategy reviewed and updated"}
            
            elif recommendation_type == "memory" and action == "optimize_memory":
                result = self.optimize_performance({"memory_threshold": parameters.get("threshold", 500000000)})  # 500MB
                return result
            
            elif recommendation_type == "cpu" and action == "optimize_cpu":
                result = self.optimize_performance({"cpu_threshold": parameters.get("threshold", 60)})  # 60%
                return result
            
            elif recommendation_type == "response_time" and action == "optimize_response_time":
                result = self.optimize_performance({"response_threshold": parameters.get("threshold", 2.0)})  # 2秒
                return result
            
            return {"success": False, "error": f"Unknown recommendation action: {action}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
