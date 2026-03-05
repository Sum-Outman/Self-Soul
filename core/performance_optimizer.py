import zlib
"""
性能优化和负载测试系统
Performance Optimization and Load Testing System

提供全面的性能优化和负载测试功能，确保系统在生产环境中的稳定性和可扩展性。
Provides comprehensive performance optimization and load testing capabilities to ensure system stability and scalability in production environments.
"""

import asyncio
import time
import psutil
import threading
import json
import statistics
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from pathlib import Path
from core.error_handling import error_handler

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime
    test_id: str
    
    # 系统指标
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    process_count: int
    
    # 应用指标
    response_time: float
    throughput: float
    error_rate: float
    concurrent_users: int
    
    # 模型指标
    model_load_time: float
    inference_time: float
    memory_consumption: float
    
    # 业务指标
    requests_per_second: float
    successful_requests: int
    failed_requests: int
    latency_p95: float
    latency_p99: float

@dataclass
class LoadTestConfig:
    """负载测试配置"""
    test_name: str
    target_url: str
    concurrent_users: int
    duration: int  # 秒
    ramp_up_time: int  # 秒
    request_timeout: int
    
    # 测试参数
    request_headers: Dict[str, str]
    request_body: Optional[Dict[str, Any]]
    test_scenarios: List[str]
    
    # 性能阈值
    max_response_time: float
    max_error_rate: float
    min_throughput: float

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_suggestions: List[str] = []
        
        # 性能基准
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        
        # 初始化系统监控
        self._initialize_monitoring()
        
        logger.info("Performance optimizer initialized")
    
    def _initialize_monitoring(self):
        """初始化系统监控"""
        self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitoring_thread.start()
    
    def _monitor_system(self):
        """系统监控循环"""
        while True:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # 保留最近1000条记录
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # 分析性能趋势
                self._analyze_performance_trends()
                
                time.sleep(5)  # 每5秒收集一次
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(10)
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """收集系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # 磁盘IO
        disk_io = psutil.disk_io_counters()
        disk_io_rate = disk_io.read_bytes + disk_io.write_bytes if disk_io else 0
        
        # 网络IO
        net_io = psutil.net_io_counters()
        net_io_rate = net_io.bytes_sent + net_io.bytes_recv if net_io else 0
        
        # 进程数
        process_count = len(psutil.pids())
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            test_id="system_monitor",
            cpu_usage=cpu_percent,
            memory_usage=memory_percent,
            disk_io=disk_io_rate,
            network_io=net_io_rate,
            process_count=process_count,
            response_time=0.0,
            throughput=0.0,
            error_rate=0.0,
            concurrent_users=0,
            model_load_time=0.0,
            inference_time=0.0,
            memory_consumption=0.0,
            requests_per_second=0.0,
            successful_requests=0,
            failed_requests=0,
            latency_p95=0.0,
            latency_p99=0.0
        )
    
    def _analyze_performance_trends(self):
        """分析性能趋势"""
        if len(self.metrics_history) < 10:
            return
        
        recent_metrics = self.metrics_history[-10:]
        
        # 检查CPU使用率趋势
        cpu_trend = self._calculate_trend([m.cpu_usage for m in recent_metrics])
        if cpu_trend > 0.1 and recent_metrics[-1].cpu_usage > 80:
            self.optimization_suggestions.append(
                "High CPU usage detected. Consider optimizing model inference or scaling horizontally."
            )
        
        # 检查内存使用率趋势
        memory_trend = self._calculate_trend([m.memory_usage for m in recent_metrics])
        if memory_trend > 0.1 and recent_metrics[-1].memory_usage > 85:
            self.optimization_suggestions.append(
                "High memory usage detected. Consider optimizing model memory usage or adding more RAM."
            )
    
    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势（线性回归斜率）"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        
        # 计算斜率
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    async def run_load_test(self, config: LoadTestConfig) -> Dict[str, Any]:
        """运行负载测试"""
        logger.info(f"Starting load test: {config.test_name}")
        
        test_results = {
            "test_name": config.test_name,
            "start_time": datetime.now().isoformat(),
            "config": asdict(config),
            "metrics": [],
            "summary": {}
        }
        
        # 创建测试任务
        tasks = []
        for user_id in range(config.concurrent_users):
            task = asyncio.create_task(
                self._execute_load_test_user(config, user_id)
            )
            tasks.append(task)
        
        # 运行测试
        start_time = time.time()
        
        # 收集测试期间的指标
        async def collect_metrics():
            while time.time() - start_time < config.duration:
                metrics = await self._collect_test_metrics(config)
                test_results["metrics"].append(metrics)
                await asyncio.sleep(1)
        
        # 并行运行测试和指标收集
        await asyncio.gather(
            *tasks,
            collect_metrics()
        )
        
        # 生成测试总结
        test_results["summary"] = self._generate_test_summary(test_results["metrics"])
        test_results["end_time"] = datetime.now().isoformat()
        
        logger.info(f"Load test completed: {config.test_name}")
        return test_results
    
    async def _execute_load_test_user(self, config: LoadTestConfig, user_id: int):
        """模拟单个用户行为"""
        import aiohttp
        import random
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < config.duration:
                try:
                    # 模拟不同类型的请求
                    request_type = ["chat", "training", "inference"][(zlib.adler32(str(str(time.time().encode('utf-8')) & 0xffffffff))) % 3]
                    
                    if request_type == "chat":
                        # 聊天请求
                        payload = {
                            "message": "Hello, how are you?",
                            "user_id": user_id
                        }
                        url = f"{config.target_url}/api/chat"
                    
                    elif request_type == "training":
                        # 训练请求
                        payload = {
                            "model_id": "test_model",
                            "dataset": "test_data"
                        }
                        url = f"{config.target_url}/api/train"
                    
                    else:  # inference
                        # 推理请求
                        payload = {
                            "model_id": "test_model",
                            "input_data": "test input"
                        }
                        url = f"{config.target_url}/api/inference"
                    
                    # 发送请求
                    async with session.post(
                        url,
                        json=payload,
                        headers=config.request_headers,
                        timeout=aiohttp.ClientTimeout(total=config.request_timeout)
                    ) as response:
                        if response.status != 200:
                            error_handler.log_warning(f"Request failed: {response.status}", "PerformanceOptimizer")
                    
                    # 随机等待时间模拟用户思考
                    await asyncio.sleep(0.5 + ((zlib.adler32(str(str(time.time().encode('utf-8')) & 0xffffffff)) + "sleep") % 150) * 0.01)
                    
                except Exception as e:
                    logger.error(f"User simulation error: {e}")
                    await asyncio.sleep(1)
    
    async def _collect_test_metrics(self, config: LoadTestConfig) -> Dict[str, Any]:
        """收集测试指标"""
        system_metrics = self._collect_system_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": asdict(system_metrics),
            "test_config": {
                "concurrent_users": config.concurrent_users,
                "duration_remaining": config.duration - (time.time() - getattr(self, '_test_start_time', time.time()))
            }
        }
    
    def _generate_test_summary(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成测试总结"""
        if not metrics:
            return {}
        
        response_times = [m.get("response_time", 0) for m in metrics if m.get("response_time")]
        error_rates = [m.get("error_rate", 0) for m in metrics if m.get("error_rate")]
        throughputs = [m.get("throughput", 0) for m in metrics if m.get("throughput")]
        
        summary = {
            "total_requests": len(metrics),
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "avg_error_rate": statistics.mean(error_rates) if error_rates else 0,
            "avg_throughput": statistics.mean(throughputs) if throughputs else 0,
            "p95_response_time": self._calculate_percentile(response_times, 95) if response_times else 0,
            "p99_response_time": self._calculate_percentile(response_times, 99) if response_times else 0
        }
        
        return summary
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100
        f = math.floor(k)
        c = math.ceil(k)
        
        if f == c:
            return sorted_values[int(k)]
        
        d0 = sorted_values[int(f)] * (c - k)
        d1 = sorted_values[int(c)] * (k - f)
        return d0 + d1
    
    def optimize_system(self) -> Dict[str, Any]:
        """系统性能优化"""
        optimization_results = {
            "timestamp": datetime.now().isoformat(),
            "optimizations_applied": [],
            "performance_improvement": {},
            "recommendations": []
        }
        
        # 分析当前性能状态
        current_metrics = self._collect_system_metrics()
        
        # 内存优化建议
        if current_metrics.memory_usage > 80:
            optimization_results["recommendations"].append(
                "High memory usage detected. Consider implementing model memory optimization."
            )
        
        # CPU优化建议
        if current_metrics.cpu_usage > 80:
            optimization_results["recommendations"].append(
                "High CPU usage detected. Consider optimizing model inference or scaling horizontally."
            )
        
        # 磁盘IO优化建议
        if current_metrics.disk_io > 100000000:  # 100MB/s
            optimization_results["recommendations"].append(
                "High disk I/O detected. Consider implementing caching or optimizing data access patterns."
            )
        
        return optimization_results
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        if not self.metrics_history:
            return {"error": "No performance data available"}
        
        recent_metrics = self.metrics_history[-100:]  # 最近100个数据点
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "time_period": {
                "start": recent_metrics[0].timestamp.isoformat(),
                "end": recent_metrics[-1].timestamp.isoformat(),
                "duration_hours": (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds() / 3600
            },
            "system_performance": {
                "avg_cpu_usage": statistics.mean([m.cpu_usage for m in recent_metrics]),
                "avg_memory_usage": statistics.mean([m.memory_usage for m in recent_metrics]),
                "peak_cpu_usage": max([m.cpu_usage for m in recent_metrics]),
                "peak_memory_usage": max([m.memory_usage for m in recent_metrics])
            },
            "performance_trends": self._analyze_long_term_trends(),
            "optimization_suggestions": self.optimization_suggestions[-10:],  # 最近10条建议
            "health_status": self._assess_system_health()
        }
        
        return report
    
    def _analyze_long_term_trends(self) -> Dict[str, Any]:
        """分析长期趋势"""
        if len(self.metrics_history) < 50:
            return {"insufficient_data": True}
        
        # 分析CPU使用率趋势
        cpu_trend = self._calculate_trend([m.cpu_usage for m in self.metrics_history[-50:]])
        
        # 分析内存使用率趋势
        memory_trend = self._calculate_trend([m.memory_usage for m in self.metrics_history[-50:]])
        
        return {
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "trend_analysis": "stable" if abs(cpu_trend) < 0.05 and abs(memory_trend) < 0.05 else "changing"
        }
    
    def _assess_system_health(self) -> str:
        """评估系统健康状态"""
        if not self.metrics_history:
            return "unknown"
        
        recent_metrics = self.metrics_history[-10:]
        
        avg_cpu = statistics.mean([m.cpu_usage for m in recent_metrics])
        avg_memory = statistics.mean([m.memory_usage for m in recent_metrics])
        
        if avg_cpu > 90 or avg_memory > 95:
            return "critical"
        elif avg_cpu > 80 or avg_memory > 85:
            return "warning"
        elif avg_cpu > 70 or avg_memory > 75:
            return "degraded"
        else:
            return "healthy"

# 性能测试API端点
class PerformanceAPI:
    """性能测试API"""
    
    def __init__(self, optimizer: PerformanceOptimizer):
        self.optimizer = optimizer
    
    async def run_load_test_endpoint(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """运行负载测试API端点"""
        try:
            config = LoadTestConfig(**test_config)
            results = await self.optimizer.run_load_test(config)
            return {"success": True, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告API端点"""
        try:
            report = self.optimizer.generate_performance_report()
            return {"success": True, "report": report}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def optimize_system_endpoint(self) -> Dict[str, Any]:
        """系统优化API端点"""
        try:
            results = self.optimizer.optimize_system()
            return {"success": True, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}

# 示例使用
if __name__ == "__main__":
    import math
    
    # 创建性能优化器
    optimizer = PerformanceOptimizer()
    
    # 示例负载测试配置
    test_config = LoadTestConfig(
        test_name="basic_load_test",
        target_url="http://localhost:8000",
        concurrent_users=10,
        duration=60,  # 1分钟
        ramp_up_time=10,
        request_timeout=30,
        request_headers={"Content-Type": "application/json"},
        request_body=None,
        test_scenarios=["chat", "training", "inference"],
        max_response_time=5.0,
        max_error_rate=0.05,
        min_throughput=10.0
    )
    
    # 运行负载测试
    async def run_test():
        results = await optimizer.run_load_test(test_config)
        print("Load test results:", json.dumps(results, indent=2, default=str))
    
    # 运行测试
    asyncio.run(run_test())
