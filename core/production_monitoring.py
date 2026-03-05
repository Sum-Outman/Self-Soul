"""
生产环境监控和日志系统
Production Environment Monitoring and Logging System
"""

import logging
import time
import psutil
import asyncio
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import os
from core.error_handling import error_handler

# 确保日志目录存在
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

@dataclass
class ProductionMetrics:
    """生产环境指标数据类"""
    timestamp: datetime
    
    # 系统指标
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    
    # 应用指标
    active_connections: int
    request_count: int
    error_count: int
    response_time_avg: float
    
    # 模型指标
    models_loaded: int
    models_active: int
    model_inference_time: float
    
    # 业务指标
    chat_messages_processed: int
    training_jobs_active: int
    knowledge_base_entries: int

class ProductionLogger:
    """生产环境日志管理器"""
    
    def __init__(self, log_level: str = "INFO"):
        self.log_level = getattr(logging, log_level.upper())
        self._configure_logging()
        self.logger = logging.getLogger("ProductionLogger")
        
        # 审计日志记录器
        self.audit_logger = logging.getLogger("audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # 性能日志记录器
        self.performance_logger = logging.getLogger("performance")
        self.performance_logger.setLevel(logging.INFO)
    
    def _configure_logging(self):
        """配置生产环境日志系统"""
        
        # 清除现有处理器
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # 主日志文件处理器 - 按时间轮转
        main_handler = TimedRotatingFileHandler(
            os.path.join(log_dir, 'application.log'),
            when='midnight',
            interval=1,
            backupCount=30,  # 保留30天
            encoding='utf-8'
        )
        main_handler.setFormatter(formatter)
        main_handler.setLevel(self.log_level)
        
        # 错误日志文件处理器
        error_handler = RotatingFileHandler(
            os.path.join(log_dir, 'error.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10,
            encoding='utf-8'
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        
        # 审计日志文件处理器
        audit_handler = RotatingFileHandler(
            os.path.join(log_dir, 'audit.log'),
            maxBytes=5*1024*1024,  # 5MB
            backupCount=15,
            encoding='utf-8'
        )
        audit_handler.setFormatter(formatter)
        
        # 性能日志文件处理器
        performance_handler = RotatingFileHandler(
            os.path.join(log_dir, 'performance.log'),
            maxBytes=5*1024*1024,  # 5MB
            backupCount=15,
            encoding='utf-8'
        )
        performance_handler.setFormatter(formatter)
        
        # 控制台处理器（生产环境只显示错误）
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.ERROR)
        
        # 配置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        root_logger.addHandler(main_handler)
        root_logger.addHandler(error_handler)
        root_logger.addHandler(console_handler)
        
        # 配置审计日志记录器
        audit_logger = logging.getLogger("audit")
        audit_logger.setLevel(logging.INFO)
        audit_logger.addHandler(audit_handler)
        audit_logger.propagate = False
        
        # 配置性能日志记录器
        performance_logger = logging.getLogger("performance")
        performance_logger.setLevel(logging.INFO)
        performance_logger.addHandler(performance_handler)
        performance_logger.propagate = False
    
    def log_audit(self, action: str, user: str, details: Dict[str, Any]):
        """记录审计日志"""
        audit_data = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user": user,
            "details": details
        }
        self.audit_logger.info(json.dumps(audit_data))
    
    def log_performance(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """记录性能指标"""
        perf_data = {
            "timestamp": datetime.now().isoformat(),
            "metric": metric_name,
            "value": value,
            "tags": tags or {}
        }
        self.performance_logger.info(json.dumps(perf_data))

class ProductionMonitor:
    """生产环境监控器"""
    
    def __init__(self, logger: ProductionLogger):
        self.logger = logger
        self.metrics_history: deque[ProductionMetrics] = deque(maxlen=1000)
        self.alert_thresholds = self._load_alert_thresholds()
        self.monitoring_thread = None
        self.is_running = False
        
        # 性能计数器
        self.request_counter = 0
        self.error_counter = 0
        self.response_times = deque(maxlen=100)
    
    def _load_alert_thresholds(self) -> Dict[str, float]:
        """加载告警阈值"""
        return {
            "cpu_percent": 90.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "response_time_avg": 5.0,  # 秒
            "error_rate": 5.0,  # 百分比
        }
    
    def start_monitoring(self):
        """启动监控"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.logger.info("Production monitoring started")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.logger.info("Production monitoring stopped")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 检查告警条件
                self._check_alerts(metrics)
                
                # 记录性能指标
                self._log_metrics(metrics)
                
                time.sleep(30)  # 每30秒收集一次指标
                
            except Exception as e:
                self.logger.logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # 出错后等待更长时间
    
    def _collect_metrics(self) -> ProductionMetrics:
        """收集系统指标"""
        # 系统指标
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net_io = psutil.net_io_counters()
        
        # 应用指标（需要从应用状态获取）
        active_connections = 0  # 需要从WebSocket管理器获取
        error_rate = (self.error_counter / max(self.request_counter, 1)) * 100
        response_time_avg = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        # 模型指标（需要从模型注册表获取）
        models_loaded = 0
        models_active = 0
        
        # 业务指标（需要从业务模块获取）
        chat_messages_processed = 0
        training_jobs_active = 0
        knowledge_base_entries = 0
        
        return ProductionMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            network_bytes_sent=net_io.bytes_sent,
            network_bytes_recv=net_io.bytes_recv,
            process_count=len(psutil.pids()),
            active_connections=active_connections,
            request_count=self.request_counter,
            error_count=self.error_counter,
            response_time_avg=response_time_avg,
            models_loaded=models_loaded,
            models_active=models_active,
            model_inference_time=0.0,
            chat_messages_processed=chat_messages_processed,
            training_jobs_active=training_jobs_active,
            knowledge_base_entries=knowledge_base_entries
        )
    
    def _check_alerts(self, metrics: ProductionMetrics):
        """检查告警条件"""
        alerts = []
        
        if metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append(f"CPU使用率过高: {metrics.cpu_percent}%")
        
        if metrics.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append(f"内存使用率过高: {metrics.memory_percent}%")
        
        if metrics.disk_percent > self.alert_thresholds["disk_percent"]:
            alerts.append(f"磁盘使用率过高: {metrics.disk_percent}%")
        
        if metrics.response_time_avg > self.alert_thresholds["response_time_avg"]:
            alerts.append(f"响应时间过长: {metrics.response_time_avg:.2f}s")
        
        error_rate = (metrics.error_count / max(metrics.request_count, 1)) * 100
        if error_rate > self.alert_thresholds["error_rate"]:
            alerts.append(f"错误率过高: {error_rate:.1f}%")
        
        if alerts:
            for alert in alerts:
                error_handler.log_warning(f"ALERT: {alert}", "ProductionMonitoring")
                # 这里可以集成到告警系统（邮件、短信、Slack等）
    
    def _log_metrics(self, metrics: ProductionMetrics):
        """记录性能指标"""
        # 记录系统指标
        self.logger.log_performance("cpu.percent", metrics.cpu_percent)
        self.logger.log_performance("memory.percent", metrics.memory_percent)
        self.logger.log_performance("disk.percent", metrics.disk_percent)
        
        # 记录应用指标
        self.logger.log_performance("requests.count", metrics.request_count)
        self.logger.log_performance("errors.count", metrics.error_count)
        self.logger.log_performance("response.time.avg", metrics.response_time_avg)
    
    def record_request(self, response_time: float, success: bool = True):
        """记录请求指标"""
        self.request_counter += 1
        self.response_times.append(response_time)
        
        if not success:
            self.error_counter += 1
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        
        return {
            "timestamp": latest.timestamp.isoformat(),
            "system": {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "disk_percent": latest.disk_percent,
                "process_count": latest.process_count
            },
            "application": {
                "active_connections": latest.active_connections,
                "request_count": latest.request_count,
                "error_count": latest.error_count,
                "response_time_avg": latest.response_time_avg
            },
            "business": {
                "chat_messages_processed": latest.chat_messages_processed,
                "training_jobs_active": latest.training_jobs_active,
                "knowledge_base_entries": latest.knowledge_base_entries
            }
        }

# 全局监控实例
production_logger = ProductionLogger()
production_monitor = ProductionMonitor(production_logger)

def initialize_production_monitoring():
    """初始化生产环境监控"""
    production_monitor.start_monitoring()
    production_logger.logger.info("Production monitoring initialized")

def shutdown_production_monitoring():
    """关闭生产环境监控"""
    production_monitor.stop_monitoring()
    production_logger.logger.info("Production monitoring shutdown")

if __name__ == "__main__":
    # 测试监控系统
    initialize_production_monitoring()
    
    try:
        # 模拟一些请求
        for i in range(10):
            production_monitor.record_request(0.1 + i * 0.01, success=True)
            time.sleep(1)
        
        # 获取指标摘要
        summary = production_monitor.get_metrics_summary()
        print("Metrics Summary:", json.dumps(summary, indent=2))
        
    finally:
        shutdown_production_monitoring()
