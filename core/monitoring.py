"""
系统监控和健康检查模块
"""

import time
import psutil
import asyncio
import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from core.error_handling import error_handler

logger = logging.getLogger("Monitoring")


@dataclass
class SystemMetrics:
    """系统指标数据类"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    process_count: int
    model_loaded_count: int
    active_connections: int


class HealthChecker:
    """健康检查器"""
    
    def __init__(self):
        self.metrics_history: List[SystemMetrics] = []
        self.max_history_size = 1000
        self.health_thresholds = {
            "cpu": 90.0,      # CPU使用率阈值
            "memory": 85.0,   # 内存使用率阈值
            "disk": 90.0,     # 磁盘使用率阈值
            "processes": 500, # 进程数阈值
        }
    
    async def check_system_health(self) -> Dict[str, Any]:
        """检查系统健康状态"""
        try:
            # 收集系统指标
            metrics = await self._collect_metrics()
            
            # 检查健康状态
            health_status = self._evaluate_health(metrics)
            
            # 存储历史数据
            self._store_metrics(metrics)
            
            return {
                "status": "healthy" if health_status["is_healthy"] else "unhealthy",
                "timestamp": metrics.timestamp.isoformat(),
                "metrics": {
                    "cpu_percent": metrics.cpu_percent,
                    "memory_percent": metrics.memory_percent,
                    "disk_percent": metrics.disk_percent,
                    "process_count": metrics.process_count,
                    "model_loaded_count": metrics.model_loaded_count,
                    "active_connections": metrics.active_connections,
                },
                "health_checks": health_status["checks"],
                "warnings": health_status["warnings"],
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _collect_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        
        # 网络IO
        net_io = psutil.net_io_counters()
        network_io = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
        }
        
        # 进程数
        process_count = len(psutil.pids())
        
        # 模型加载数量（需要从模型注册表获取）
        model_loaded_count = 0
        try:
            from core.model_registry import model_registry
            if model_registry:
                model_loaded_count = len([m for m in model_registry.models.values() if m.get("loaded", False)])
        except Exception as e:
            logger.debug(f"Failed to get model registry: {e}")
        
        # 活跃连接数（需要从连接管理器获取）
        active_connections = 0
        try:
            from core.main import connection_manager
            if connection_manager:
                active_connections = len(connection_manager.active_connections)
        except Exception as e:
            logger.debug(f"Failed to get connection manager: {e}")
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            network_io=network_io,
            process_count=process_count,
            model_loaded_count=model_loaded_count,
            active_connections=active_connections,
        )
    
    def _evaluate_health(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """评估健康状态"""
        checks = {}
        warnings = []
        
        # CPU检查
        checks["cpu"] = metrics.cpu_percent <= self.health_thresholds["cpu"]
        if metrics.cpu_percent > 80:
            warnings.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # 内存检查
        checks["memory"] = metrics.memory_percent <= self.health_thresholds["memory"]
        if metrics.memory_percent > 80:
            warnings.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        # 磁盘检查
        checks["disk"] = metrics.disk_percent <= self.health_thresholds["disk"]
        if metrics.disk_percent > 80:
            warnings.append(f"High disk usage: {metrics.disk_percent:.1f}%")
        
        # 进程检查
        checks["processes"] = metrics.process_count <= self.health_thresholds["processes"]
        if metrics.process_count > 300:
            warnings.append(f"High process count: {metrics.process_count}")
        
        # 模型检查
        checks["models"] = metrics.model_loaded_count > 0
        if metrics.model_loaded_count == 0:
            warnings.append("No models loaded")
        
        is_healthy = all(checks.values())
        
        return {
            "is_healthy": is_healthy,
            "checks": checks,
            "warnings": warnings,
        }
    
    def _store_metrics(self, metrics: SystemMetrics):
        """存储指标数据"""
        self.metrics_history.append(metrics)
        
        # 限制历史数据大小
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取指定时间范围内的指标历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "cpu_percent": m.cpu_percent,
                "memory_percent": m.memory_percent,
                "disk_percent": m.disk_percent,
                "process_count": m.process_count,
                "model_loaded_count": m.model_loaded_count,
                "active_connections": m.active_connections,
            }
            for m in recent_metrics
        ]


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.request_times: List[float] = []
        self.max_request_history = 1000
        self.performance_thresholds = {
            "avg_response_time": 2.0,  # 平均响应时间阈值（秒）
            "p95_response_time": 5.0,  # 95%响应时间阈值
            "error_rate": 0.05,        # 错误率阈值
        }
    
    def record_request_time(self, duration: float):
        """记录请求处理时间"""
        self.request_times.append(duration)
        
        # 限制历史数据大小
        if len(self.request_times) > self.max_request_history:
            self.request_times = self.request_times[-self.max_request_history:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.request_times:
            return {"error": "No request data available"}
        
        sorted_times = sorted(self.request_times)
        n = len(sorted_times)
        
        return {
            "total_requests": n,
            "avg_response_time": sum(sorted_times) / n,
            "min_response_time": min(sorted_times),
            "max_response_time": max(sorted_times),
            "p50_response_time": sorted_times[int(n * 0.5)],
            "p95_response_time": sorted_times[int(n * 0.95)],
            "p99_response_time": sorted_times[int(n * 0.99)],
        }
    
    def check_performance_health(self) -> Dict[str, Any]:
        """检查性能健康状态"""
        stats = self.get_performance_stats()
        
        if "error" in stats:
            return {"status": "unknown", "error": stats["error"]}
        
        warnings = []
        
        # 检查平均响应时间
        if stats["avg_response_time"] > self.performance_thresholds["avg_response_time"]:
            warnings.append(f"High average response time: {stats['avg_response_time']:.2f}s")
        
        # 检查95%响应时间
        if stats["p95_response_time"] > self.performance_thresholds["p95_response_time"]:
            warnings.append(f"High 95th percentile response time: {stats['p95_response_time']:.2f}s")
        
        is_healthy = len(warnings) == 0
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "stats": stats,
            "warnings": warnings,
        }


# 全局监控实例
health_checker = HealthChecker()
performance_monitor = PerformanceMonitor()


async def start_monitoring_service():
    """启动监控服务"""
    logger.info("Starting monitoring service...")
    
    async def monitor_loop():
        """监控循环"""
        while True:
            try:
                # 定期检查系统健康
                health_status = await health_checker.check_system_health()
                
                # 记录性能统计
                performance_status = performance_monitor.check_performance_health()
                
                # 如果有警告，记录日志
                if health_status.get("warnings"):
                    for warning in health_status["warnings"]:
                        error_handler.log_warning(warning, "Monitoring")
                
                if performance_status.get("warnings"):
                    for warning in performance_status["warnings"]:
                        error_handler.log_warning(warning, "Monitoring")
                
                # 每30秒检查一次
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # 出错时等待更长时间
    
    # 启动监控循环
    asyncio.create_task(monitor_loop())
    logger.info("Monitoring service started")


def get_system_status() -> Dict[str, Any]:
    """获取系统状态摘要"""
    try:
        # 获取系统资源
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 获取模型状态
        model_status = {}
        try:
            from core.model_registry import model_registry
            if model_registry:
                for model_id, model_info in model_registry.models.items():
                    model_status[model_id] = {
                        "loaded": model_info.get("loaded", False),
                        "mode": model_info.get("mode", "local"),
                    }
        except Exception as e:
            logger.debug(f"Failed to get model status: {e}")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "uptime": time.time() - psutil.boot_time(),
            },
            "models": model_status,
            "performance": performance_monitor.get_performance_stats(),
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # 测试监控功能
    async def test_monitoring():
        health = await health_checker.check_system_health()
        print("Health Check:", health)
        
        status = get_system_status()
        print("System Status:", status)
    
    asyncio.run(test_monitoring())
