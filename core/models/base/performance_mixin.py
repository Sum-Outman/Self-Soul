"""
Performance Monitoring Mixin - Performance Tracking and Optimization

This module provides performance monitoring capabilities that can be mixed into
any model class. It handles performance metrics collection, analysis, and
optimization strategies.
"""

import time
from typing import Dict, Any, List
from datetime import datetime


class PerformanceMixin:
    """Mixin class for performance monitoring and optimization"""
    
    def __init__(self, *args, **kwargs):
        """Initialize performance monitoring capabilities"""
        super().__init__(*args, **kwargs)
        
        # Performance metrics tracking
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
        
        # Performance optimization settings
        self.performance_settings = {
            "monitoring_enabled": True,
            "auto_optimization": True,
            "optimization_threshold": 0.8,  # 80% utilization threshold
            "sampling_rate": 0.1  # 10% of requests are sampled for detailed metrics
        }
        
        # Historical performance data
        self.performance_history = []
        self.max_history_size = 1000

    # ===== PERFORMANCE MONITORING METHODS =====
    
    def _start_performance_monitoring(self, operation: str) -> Dict[str, Any]:
        """Start performance monitoring for an operation
        
        Args:
            operation: Name of the operation being monitored
            
        Returns:
            Dictionary containing monitoring context
        """
        return {
            "operation": operation,
            "start_time": time.time(),
            "start_timestamp": datetime.now().isoformat(),
            "initial_metrics": self.performance_metrics.copy()
        }

    def _end_performance_monitoring(self, monitoring_context: Dict[str, Any], 
                                  success: bool = True) -> Dict[str, Any]:
        """End performance monitoring and update metrics
        
        Args:
            monitoring_context: Context from _start_performance_monitoring
            success: Whether the operation was successful
            
        Returns:
            Dictionary containing performance results
        """
        end_time = time.time()
        response_time = end_time - monitoring_context["start_time"]
        
        # Update basic metrics
        self.performance_metrics["total_requests"] += 1
        self.performance_metrics["last_response_time"] = response_time
        
        if success:
            self.performance_metrics["successful_requests"] += 1
        else:
            self.performance_metrics["failed_requests"] += 1
        
        # Update average response time
        successful_requests = self.performance_metrics["successful_requests"]
        if successful_requests > 0:
            current_avg = self.performance_metrics["average_response_time"]
            self.performance_metrics["average_response_time"] = (
                (current_avg * (successful_requests - 1) + response_time) / successful_requests
            )
        
        # Update throughput (requests per second)
        if response_time > 0:
            self.performance_metrics["throughput"] = 1.0 / response_time
        
        # Record detailed performance data (sampled)
        if self._should_sample_performance():
            performance_record = {
                "timestamp": datetime.now().isoformat(),
                "operation": monitoring_context["operation"],
                "response_time": response_time,
                "success": success,
                "metrics_snapshot": self.performance_metrics.copy()
            }
            self._add_to_performance_history(performance_record)
        
        return {
            "response_time": response_time,
            "success": success,
            "metrics_updated": True
        }

    def _should_sample_performance(self) -> bool:
        """Determine if current operation should be sampled for detailed metrics"""
        import random
        return random.random() < self.performance_settings["sampling_rate"]

    def _add_to_performance_history(self, record: Dict[str, Any]):
        """Add performance record to history, maintaining size limits"""
        self.performance_history.append(record)
        
        # Trim history if it exceeds maximum size
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size:]

    # ===== PERFORMANCE ANALYSIS METHODS =====
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary
        
        Returns:
            Dictionary containing performance summary
        """
        total_requests = self.performance_metrics["total_requests"]
        successful_requests = self.performance_metrics["successful_requests"]
        failed_requests = self.performance_metrics["failed_requests"]
        
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "model_id": getattr(self, 'model_id', 'unknown'),
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": round(success_rate, 2),
            "average_response_time": round(self.performance_metrics["average_response_time"], 3),
            "last_response_time": round(self.performance_metrics["last_response_time"], 3),
            "peak_memory_usage": self.performance_metrics["peak_memory_usage"],
            "cpu_utilization": round(self.performance_metrics["cpu_utilization"], 2),
            "throughput": round(self.performance_metrics["throughput"], 2),
            "monitoring_enabled": self.performance_settings["monitoring_enabled"]
        }

    def analyze_performance_trends(self, window_size: int = 100) -> Dict[str, Any]:
        """Analyze performance trends over a window of recent operations
        
        Args:
            window_size: Number of recent operations to analyze
            
        Returns:
            Dictionary containing trend analysis
        """
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        # Get recent records
        recent_records = self.performance_history[-window_size:]
        
        if not recent_records:
            return {"error": "No recent performance data available"}
        
        # Calculate trends
        response_times = [r["response_time"] for r in recent_records]
        success_rates = [1 if r["success"] else 0 for r in recent_records]
        
        avg_response_time = sum(response_times) / len(response_times)
        success_rate = sum(success_rates) / len(success_rates) * 100
        
        # Detect trends
        trend_analysis = self._detect_performance_trends(recent_records)
        
        return {
            "window_size": len(recent_records),
            "average_response_time": round(avg_response_time, 3),
            "success_rate": round(success_rate, 2),
            "trend_analysis": trend_analysis,
            "recommendations": self._generate_performance_recommendations(trend_analysis)
        }

    def _detect_performance_trends(self, records: list) -> Dict[str, Any]:
        """Detect performance trends from historical records
        
        Args:
            records: List of performance records
            
        Returns:
            Dictionary containing trend analysis
        """
        if len(records) < 2:
            return {"status": "insufficient_data"}
        
        # Split records into halves for comparison
        half_point = len(records) // 2
        first_half = records[:half_point]
        second_half = records[half_point:]
        
        # Calculate averages for each half
        first_avg_time = sum(r["response_time"] for r in first_half) / len(first_half)
        second_avg_time = sum(r["response_time"] for r in second_half) / len(second_half)
        
        first_success_rate = sum(1 if r["success"] else 0 for r in first_half) / len(first_half) * 100
        second_success_rate = sum(1 if r["success"] else 0 for r in second_half) / len(second_half) * 100
        
        # Determine trends
        time_trend = "improving" if second_avg_time < first_avg_time else "degrading"
        success_trend = "improving" if second_success_rate > first_success_rate else "degrading"
        
        return {
            "response_time_trend": time_trend,
            "success_rate_trend": success_trend,
            "response_time_change": round(second_avg_time - first_avg_time, 3),
            "success_rate_change": round(second_success_rate - first_success_rate, 2),
            "confidence": "high" if len(records) >= 50 else "medium"
        }

    def _generate_performance_recommendations(self, trend_analysis: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on trend analysis
        
        Args:
            trend_analysis: Results from trend analysis
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if trend_analysis.get("response_time_trend") == "degrading":
            recommendations.append("Consider optimizing response time through caching")
            recommendations.append("Review resource utilization for bottlenecks")
        
        if trend_analysis.get("success_rate_trend") == "degrading":
            recommendations.append("Investigate recent error patterns")
            recommendations.append("Consider adding retry mechanisms for failed operations")
        
        if self.performance_metrics["average_response_time"] > 5.0:  # 5 seconds threshold
            recommendations.append("Response time exceeds recommended threshold - optimize processing")
        
        if self.performance_metrics["success_rate"] < 95.0:  # 95% success rate threshold
            recommendations.append("Success rate below target - improve error handling")
        
        return recommendations

    # ===== PERFORMANCE OPTIMIZATION METHODS =====
    
    def optimize_performance(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize model performance based on current metrics
        
        Args:
            config: Optimization configuration
            
        Returns:
            Dictionary containing optimization results
        """
        config = config or {}
        
        try:
            optimizations = []
            
            # Memory optimization
            memory_threshold = config.get("memory_threshold", 1000000000)  # 1GB
            if self.performance_metrics.get("peak_memory_usage", 0) > memory_threshold:
                optimizations.append("memory_optimization")
                self._optimize_memory_usage()
            
            # CPU optimization
            cpu_threshold = config.get("cpu_threshold", 80)  # 80% utilization
            if self.performance_metrics.get("cpu_utilization", 0) > cpu_threshold:
                optimizations.append("cpu_optimization")
                self._optimize_cpu_usage()
            
            # Response time optimization
            response_threshold = config.get("response_threshold", 5.0)  # 5 seconds
            avg_response = self.performance_metrics.get("average_response_time", 0)
            if avg_response > response_threshold:
                optimizations.append("response_time_optimization")
                self._optimize_response_time()
            
            # Cache optimization
            cache_threshold = config.get("cache_threshold", 0.3)  # 30% hit rate
            if self.performance_metrics.get("cache_hit_rate", 0) < cache_threshold:
                optimizations.append("cache_optimization")
                self._optimize_cache_strategy()
            
            return {
                "success": True,
                "optimizations_applied": optimizations,
                "performance_metrics": self.performance_metrics.copy()
            }
            
        except Exception as e:
            self.logger.error(f"Performance optimization failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _optimize_memory_usage(self):
        """Optimize memory usage"""
        # Clear cached data if available
        if hasattr(self, 'cache') and hasattr(self.cache, 'clear'):
            self.cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Update memory usage metrics
        self._update_resource_metrics()
        
        self.logger.info("Memory usage optimization completed")

    def _optimize_cpu_usage(self):
        """Optimize CPU usage"""
        # Reduce parallel processing if applicable
        # This is a placeholder for model-specific optimizations
        
        # Update CPU metrics
        self._update_resource_metrics()
        
        self.logger.info("CPU usage optimization completed")

    def _optimize_response_time(self):
        """Optimize response time"""
        # Implement response time optimizations
        # This is a placeholder for model-specific optimizations
        
        self.logger.info("Response time optimization completed")

    def _optimize_cache_strategy(self):
        """Optimize cache strategy"""
        if hasattr(self, 'cache_enabled'):
            # Adjust cache TTL or strategy
            if hasattr(self, 'cache_ttl'):
                # Increase cache TTL for better hit rates
                self.cache_ttl = min(self.cache_ttl * 1.5, 3600)  # Max 1 hour
        
        self.logger.info("Cache strategy optimization completed")

    def _update_resource_metrics(self):
        """Update resource usage metrics"""
        try:
            import psutil
            process = psutil.Process()
            
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            
            # Update metrics
            self.performance_metrics["cpu_utilization"] = cpu_percent
            
            # Update peak memory usage
            if memory_info.rss > self.performance_metrics["peak_memory_usage"]:
                self.performance_metrics["peak_memory_usage"] = memory_info.rss
                
        except ImportError:
            self.logger.warning("psutil not available for resource monitoring")

    # ===== PERFORMANCE SETTINGS MANAGEMENT =====
    
    def enable_performance_monitoring(self, enabled: bool = True):
        """Enable or disable performance monitoring
        
        Args:
            enabled: Whether to enable performance monitoring
        """
        self.performance_settings["monitoring_enabled"] = enabled
        self.logger.info(f"Performance monitoring {'enabled' if enabled else 'disabled'}")

    def set_performance_sampling_rate(self, rate: float):
        """Set performance sampling rate
        
        Args:
            rate: Sampling rate between 0.0 and 1.0
        """
        if 0.0 <= rate <= 1.0:
            self.performance_settings["sampling_rate"] = rate
            self.logger.info(f"Performance sampling rate set to: {rate}")
        else:
            self.logger.warning("Sampling rate must be between 0.0 and 1.0")

    def set_optimization_threshold(self, threshold: float):
        """Set optimization threshold
        
        Args:
            threshold: Threshold value for auto-optimization
        """
        if 0.0 <= threshold <= 1.0:
            self.performance_settings["optimization_threshold"] = threshold
            self.logger.info(f"Optimization threshold set to: {threshold}")
        else:
            self.logger.warning("Optimization threshold must be between 0.0 and 1.0")

    def export_performance_report(self) -> Dict[str, Any]:
        """Export comprehensive performance report
        
        Returns:
            Dictionary containing performance report
        """
        return {
            "model_id": getattr(self, 'model_id', 'unknown'),
            "performance_metrics": self.performance_metrics.copy(),
            "performance_settings": self.performance_settings.copy(),
            "recent_performance": self.performance_history[-100:] if self.performance_history else [],
            "trend_analysis": self.analyze_performance_trends(),
            "export_timestamp": datetime.now().isoformat()
        }
