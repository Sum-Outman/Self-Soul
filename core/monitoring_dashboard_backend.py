"""
监控面板后端数据支持 - Monitoring Dashboard Backend Support

为前端监控面板提供全面的AGI系统后端数据支持，包括：
1. 系统资源监控数据（CPU、内存、GPU、磁盘、网络）
2. AGI模型状态和性能数据
3. 多模型协同调度状态
4. 动态知识引擎状态
5. 安全校验和异常监控
6. 实时数据流和历史数据分析

设计目标：
- 将前端UI雏形与后端数据采集连接
- 提供实时监控和异常诊断能力
- 支持决策触发和自动化响应
- 实现完整的数据采集、处理、存储和查询管道
"""

import asyncio
import time
import logging
import threading
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import threading
import queue

logger = logging.getLogger(__name__)

from core.monitoring import health_checker, performance_monitor, get_system_status
from core.performance_monitoring_dashboard import PerformanceMonitoringDashboard, create_performance_monitoring_dashboard
from core.collaboration.multi_model_scheduler import get_scheduler, MultiModelScheduler
from core.knowledge.dynamic_knowledge_engine import get_knowledge_engine, DynamicKnowledgeEngine

class DataSource(Enum):
    """数据源枚举"""
    SYSTEM_MONITOR = "system_monitor"          # 系统监控
    PERFORMANCE_DASHBOARD = "performance_dashboard"  # 性能仪表板
    MULTI_MODEL_SCHEDULER = "multi_model_scheduler"  # 多模型调度器
    KNOWLEDGE_ENGINE = "knowledge_engine"      # 知识引擎
    SECURITY_MONITOR = "security_monitor"      # 安全监控
    ERROR_HANDLER = "error_handler"            # 错误处理器

class DataType(Enum):
    """数据类型枚举"""
    REAL_TIME = "real_time"          # 实时数据
    HISTORICAL = "historical"        # 历史数据
    STATISTICAL = "statistical"      # 统计汇总
    ALERTS = "alerts"                # 警报数据
    EVENTS = "events"                # 事件数据
    METRICS = "metrics"              # 指标数据

@dataclass
class MonitoringData:
    """监控数据"""
    data_id: str
    source: DataSource
    data_type: DataType
    timestamp: float
    data: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    ttl: Optional[float] = None  # 生存时间（秒）

@dataclass
class AlertRule:
    """警报规则"""
    rule_id: str
    condition: str
    severity: str  # 'info', 'warning', 'critical'
    action: str
    enabled: bool = True
    last_triggered: Optional[float] = None
    trigger_count: int = 0

@dataclass
class DashboardMetric:
    """仪表板指标"""
    metric_id: str
    name: str
    value: float
    unit: str
    trend: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class MonitoringDashboardBackend:
    """监控面板后端服务"""
    
    def __init__(self, db_path: str = "monitoring_dashboard.db"):
        """初始化监控面板后端"""
        # 数据库连接
        self.db_path = db_path
        self.db_lock = threading.RLock()
        self._init_database()
        
        # 数据源
        self.health_checker = health_checker
        self.performance_monitor = performance_monitor
        self.performance_dashboard = create_performance_monitoring_dashboard()
        self.scheduler = get_scheduler()
        self.knowledge_engine = get_knowledge_engine()
        
        # 数据存储
        self.realtime_data: Dict[DataSource, Dict[str, Any]] = defaultdict(dict)
        self.historical_data = deque(maxlen=10000)
        self.alerts = deque(maxlen=1000)
        self.events = deque(maxlen=5000)
        
        # 指标缓存
        self.metrics_cache: Dict[str, DashboardMetric] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # 警报规则
        self.alert_rules: Dict[str, AlertRule] = {}
        self._init_alert_rules()
        
        # 数据采集配置
        self.collection_config = {
            "system_monitor_interval": 5.0,
            "performance_interval": 10.0,
            "scheduler_interval": 15.0,
            "knowledge_interval": 30.0,
            "metrics_aggregation_interval": 60.0,
            "data_retention_days": 30
        }
        
        # 数据采集线程
        self.collection_threads = {}
        self.running = False
        
        # 事件订阅
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # 初始化后端
        self._initialize_backend()
        
        logger.info("监控面板后端服务初始化完成")
    
    def _init_database(self):
        """初始化数据库"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建监控数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS monitoring_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    data_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建指标历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metric_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建警报表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    acknowledged BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建事件表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_monitoring_data_timestamp ON monitoring_data(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metric_history_metric_id ON metric_history(metric_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metric_history_timestamp ON metric_history(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)')
            
            conn.commit()
            conn.close()
    
    def _init_alert_rules(self):
        """初始化警报规则"""
        # 系统资源警报规则
        self.add_alert_rule(
            rule_id="high_cpu_usage",
            condition="system.cpu_percent > 85",
            severity="warning",
            action="notify_system_administrator"
        )
        
        self.add_alert_rule(
            rule_id="critical_cpu_usage",
            condition="system.cpu_percent > 95",
            severity="critical",
            action="trigger_emergency_response"
        )
        
        self.add_alert_rule(
            rule_id="high_memory_usage",
            condition="system.memory_percent > 85",
            severity="warning",
            action="suggest_memory_optimization"
        )
        
        # 调度器警报规则
        self.add_alert_rule(
            rule_id="scheduler_high_queue",
            condition="scheduler.pending_tasks > 50",
            severity="warning",
            action="increase_scheduler_capacity"
        )
        
        self.add_alert_rule(
            rule_id="scheduler_failure_rate_high",
            condition="scheduler.failure_rate > 0.1",
            severity="critical",
            action="investigate_scheduler_failures"
        )
        
        # 知识引擎警报规则
        self.add_alert_rule(
            rule_id="knowledge_conflicts_high",
            condition="knowledge.active_conflicts > 10",
            severity="warning",
            action="review_knowledge_conflicts"
        )
        
        logger.info(f"初始化了 {len(self.alert_rules)} 个警报规则")
    
    def _initialize_backend(self):
        """初始化后端服务"""
        # 启动数据采集
        self.running = True
        self._start_data_collection()
        
        # 启动指标聚合
        self._start_metrics_aggregation()
        
        # 启动警报检查
        self._start_alert_checking()
        
        # 启动数据清理
        self._start_data_cleanup()
    
    def _start_data_collection(self):
        """启动数据采集"""
        # 系统监控数据采集
        self.collection_threads["system_monitor"] = threading.Thread(
            target=self._collect_system_monitor_data,
            daemon=True,
            name="SystemMonitorCollector"
        )
        self.collection_threads["system_monitor"].start()
        
        # 性能仪表板数据采集
        self.collection_threads["performance_dashboard"] = threading.Thread(
            target=self._collect_performance_dashboard_data,
            daemon=True,
            name="PerformanceDashboardCollector"
        )
        self.collection_threads["performance_dashboard"].start()
        
        # 调度器数据采集
        self.collection_threads["scheduler"] = threading.Thread(
            target=self._collect_scheduler_data,
            daemon=True,
            name="SchedulerCollector"
        )
        self.collection_threads["scheduler"].start()
        
        # 知识引擎数据采集
        self.collection_threads["knowledge_engine"] = threading.Thread(
            target=self._collect_knowledge_engine_data,
            daemon=True,
            name="KnowledgeEngineCollector"
        )
        self.collection_threads["knowledge_engine"].start()
    
    def _start_metrics_aggregation(self):
        """启动指标聚合"""
        self.collection_threads["metrics_aggregation"] = threading.Thread(
            target=self._aggregate_metrics,
            daemon=True,
            name="MetricsAggregator"
        )
        self.collection_threads["metrics_aggregation"].start()
    
    def _start_alert_checking(self):
        """启动警报检查"""
        self.collection_threads["alert_checking"] = threading.Thread(
            target=self._check_alerts,
            daemon=True,
            name="AlertChecker"
        )
        self.collection_threads["alert_checking"].start()
    
    def _start_data_cleanup(self):
        """启动数据清理"""
        self.collection_threads["data_cleanup"] = threading.Thread(
            target=self._cleanup_old_data,
            daemon=True,
            name="DataCleanup"
        )
        self.collection_threads["data_cleanup"].start()
    
    def _collect_system_monitor_data(self):
        """采集系统监控数据"""
        while self.running:
            try:
                # 获取系统状态
                system_status = get_system_status()
                
                # 获取健康检查数据
                health_status = asyncio.run(self.health_checker.check_system_health())
                
                # 获取性能数据
                performance_stats = self.performance_monitor.get_performance_stats()
                
                # 创建监控数据
                data = {
                    "system_status": system_status,
                    "health_status": health_status,
                    "performance_stats": performance_stats,
                    "timestamp": time.time()
                }
                
                data_id = f"system_monitor_{int(time.time())}"
                monitoring_data = MonitoringData(
                    data_id=data_id,
                    source=DataSource.SYSTEM_MONITOR,
                    data_type=DataType.REAL_TIME,
                    timestamp=time.time(),
                    data=data,
                    tags=["system", "health", "performance"]
                )
                
                # 存储数据
                self._store_monitoring_data(monitoring_data)
                
                # 更新实时数据
                self.realtime_data[DataSource.SYSTEM_MONITOR] = data
                
                # 提取指标
                self._extract_system_metrics(data)
                
            except Exception as e:
                logger.error(f"系统监控数据采集失败: {e}")
            
            # 等待下一次采集
            time.sleep(self.collection_config["system_monitor_interval"])
    
    def _collect_performance_dashboard_data(self):
        """采集性能仪表板数据"""
        while self.running:
            try:
                # 获取性能仪表板状态
                dashboard_status = self.performance_dashboard.get_current_status()
                
                # 获取性能摘要
                performance_summary = self.performance_dashboard.get_performance_summary(hours=1)
                
                # 获取推理统计
                inference_statistics = self.performance_dashboard.get_inference_statistics()
                
                # 创建监控数据
                data = {
                    "dashboard_status": dashboard_status,
                    "performance_summary": performance_summary,
                    "inference_statistics": inference_statistics,
                    "timestamp": time.time()
                }
                
                data_id = f"performance_dashboard_{int(time.time())}"
                monitoring_data = MonitoringData(
                    data_id=data_id,
                    source=DataSource.PERFORMANCE_DASHBOARD,
                    data_type=DataType.REAL_TIME,
                    timestamp=time.time(),
                    data=data,
                    tags=["performance", "dashboard", "inference"]
                )
                
                # 存储数据
                self._store_monitoring_data(monitoring_data)
                
                # 更新实时数据
                self.realtime_data[DataSource.PERFORMANCE_DASHBOARD] = data
                
                # 提取指标
                self._extract_performance_metrics(data)
                
            except Exception as e:
                logger.error(f"性能仪表板数据采集失败: {e}")
            
            # 等待下一次采集
            time.sleep(self.collection_config["performance_interval"])
    
    def _collect_scheduler_data(self):
        """采集调度器数据"""
        while self.running:
            try:
                # 获取调度器指标
                scheduler_metrics = self.scheduler.get_scheduler_metrics()
                
                # 获取活跃任务状态
                active_tasks = {}
                for task_id, task_info in self.scheduler.tasks.items():
                    active_tasks[task_id] = {
                        "task_name": task_info.task_name,
                        "status": task_info.status.value,
                        "assigned_models": task_info.assigned_models,
                        "priority": task_info.priority,
                        "created_at": task_info.created_at
                    }
                
                # 创建监控数据
                data = {
                    "scheduler_metrics": scheduler_metrics,
                    "active_tasks": active_tasks,
                    "task_queue_size": len(self.scheduler.task_queue),
                    "completed_tasks_count": len(self.scheduler.completed_tasks),
                    "timestamp": time.time()
                }
                
                data_id = f"scheduler_{int(time.time())}"
                monitoring_data = MonitoringData(
                    data_id=data_id,
                    source=DataSource.MULTI_MODEL_SCHEDULER,
                    data_type=DataType.REAL_TIME,
                    timestamp=time.time(),
                    data=data,
                    tags=["scheduler", "tasks", "multi_model"]
                )
                
                # 存储数据
                self._store_monitoring_data(monitoring_data)
                
                # 更新实时数据
                self.realtime_data[DataSource.MULTI_MODEL_SCHEDULER] = data
                
                # 提取指标
                self._extract_scheduler_metrics(data)
                
            except Exception as e:
                logger.error(f"调度器数据采集失败: {e}")
            
            # 等待下一次采集
            time.sleep(self.collection_config["scheduler_interval"])
    
    def _collect_knowledge_engine_data(self):
        """采集知识引擎数据"""
        while self.running:
            try:
                # 获取知识引擎统计
                engine_statistics = self.knowledge_engine.get_engine_statistics()
                
                # 获取活跃冲突
                active_conflicts = {}
                for conflict_id, conflict in self.knowledge_engine.active_conflicts.items():
                    active_conflicts[conflict_id] = {
                        "entity_id": conflict.entity_id,
                        "attribute_name": conflict.attribute_name,
                        "conflicting_values": conflict.conflicting_values,
                        "detected_at": conflict.detected_at,
                        "resolved": conflict.resolved
                    }
                
                # 获取最近的知识更新
                recent_updates = []
                for update in list(self.knowledge_engine.update_history)[-10:]:
                    recent_updates.append({
                        "update_id": update.update_id,
                        "entity_id": update.entity_id,
                        "update_type": update.update_type,
                        "source": update.source.value if hasattr(update.source, 'value') else str(update.source),
                        "timestamp": update.timestamp
                    })
                
                # 创建监控数据
                data = {
                    "engine_statistics": engine_statistics,
                    "active_conflicts": active_conflicts,
                    "recent_updates": recent_updates,
                    "timestamp": time.time()
                }
                
                data_id = f"knowledge_engine_{int(time.time())}"
                monitoring_data = MonitoringData(
                    data_id=data_id,
                    source=DataSource.KNOWLEDGE_ENGINE,
                    data_type=DataType.REAL_TIME,
                    timestamp=time.time(),
                    data=data,
                    tags=["knowledge", "engine", "conflicts", "updates"]
                )
                
                # 存储数据
                self._store_monitoring_data(monitoring_data)
                
                # 更新实时数据
                self.realtime_data[DataSource.KNOWLEDGE_ENGINE] = data
                
                # 提取指标
                self._extract_knowledge_metrics(data)
                
            except Exception as e:
                logger.error(f"知识引擎数据采集失败: {e}")
            
            # 等待下一次采集
            time.sleep(self.collection_config["knowledge_interval"])
    
    def _extract_system_metrics(self, data: Dict[str, Any]):
        """提取系统指标"""
        try:
            # CPU使用率
            if "system_status" in data and "system" in data["system_status"]:
                system = data["system_status"]["system"]
                cpu_metric = DashboardMetric(
                    metric_id="system_cpu_percent",
                    name="CPU使用率",
                    value=system.get("cpu_percent", 0.0),
                    unit="%",
                    threshold_warning=80.0,
                    threshold_critical=95.0,
                    metadata={"cores": system.get("cores", 1)}
                )
                self._update_metric(cpu_metric)
            
            # 内存使用率
            if "system_status" in data and "system" in data["system_status"]:
                system = data["system_status"]["system"]
                memory_metric = DashboardMetric(
                    metric_id="system_memory_percent",
                    name="内存使用率",
                    value=system.get("memory_percent", 0.0),
                    unit="%",
                    threshold_warning=85.0,
                    threshold_critical=95.0,
                    metadata={
                        "total_gb": system.get("memory_total_gb", 0.0),
                        "used_gb": system.get("memory_used_gb", 0.0)
                    }
                )
                self._update_metric(memory_metric)
            
            # 系统正常运行时间
            if "system_status" in data and "system" in data["system_status"]:
                system = data["system_status"]["system"]
                uptime_metric = DashboardMetric(
                    metric_id="system_uptime",
                    name="系统运行时间",
                    value=system.get("uptime", 0.0),
                    unit="秒",
                    metadata={"hours": system.get("uptime", 0.0) / 3600}
                )
                self._update_metric(uptime_metric)
                
        except Exception as e:
            logger.error(f"提取系统指标失败: {e}")
    
    def _extract_performance_metrics(self, data: Dict[str, Any]):
        """提取性能指标"""
        try:
            # 平均响应时间
            if "performance_summary" in data and "metrics_summary" in data["performance_summary"]:
                metrics_summary = data["performance_summary"]["metrics_summary"]
                if "inference" in metrics_summary:
                    inference_metrics = metrics_summary["inference"]
                    avg_latency_metric = DashboardMetric(
                        metric_id="inference_avg_latency",
                        name="推理平均延迟",
                        value=inference_metrics.get("avg", 0.0),
                        unit="ms",
                        threshold_warning=5000.0,
                        threshold_critical=10000.0
                    )
                    self._update_metric(avg_latency_metric)
            
            # 推理成功率
            if "inference_statistics" in data and "overall" in data["inference_statistics"]:
                overall = data["inference_statistics"]["overall"]
                success_rate_metric = DashboardMetric(
                    metric_id="inference_success_rate",
                    name="推理成功率",
                    value=overall.get("success_rate", 0.0),
                    unit="%",
                    threshold_warning=95.0,
                    threshold_critical=90.0
                )
                self._update_metric(success_rate_metric)
                
        except Exception as e:
            logger.error(f"提取性能指标失败: {e}")
    
    def _extract_scheduler_metrics(self, data: Dict[str, Any]):
        """提取调度器指标"""
        try:
            # 活跃任务数
            if "active_tasks" in data:
                active_tasks_metric = DashboardMetric(
                    metric_id="scheduler_active_tasks",
                    name="活跃任务数",
                    value=len(data["active_tasks"]),
                    unit="个",
                    threshold_warning=20,
                    threshold_critical=50
                )
                self._update_metric(active_tasks_metric)
            
            # 任务队列大小
            if "task_queue_size" in data:
                queue_size_metric = DashboardMetric(
                    metric_id="scheduler_queue_size",
                    name="任务队列大小",
                    value=data["task_queue_size"],
                    unit="个",
                    threshold_warning=30,
                    threshold_critical=100
                )
                self._update_metric(queue_size_metric)
            
            # 调度成功率
            if "scheduler_metrics" in data:
                scheduler_metrics = data["scheduler_metrics"]
                success_rate = scheduler_metrics.get("success_rate", 0.0)
                success_rate_metric = DashboardMetric(
                    metric_id="scheduler_success_rate",
                    name="调度成功率",
                    value=success_rate,
                    unit="%",
                    threshold_warning=95.0,
                    threshold_critical=90.0
                )
                self._update_metric(success_rate_metric)
                
        except Exception as e:
            logger.error(f"提取调度器指标失败: {e}")
    
    def _extract_knowledge_metrics(self, data: Dict[str, Any]):
        """提取知识引擎指标"""
        try:
            # 活跃冲突数
            if "active_conflicts" in data:
                conflicts_metric = DashboardMetric(
                    metric_id="knowledge_active_conflicts",
                    name="活跃知识冲突",
                    value=len(data["active_conflicts"]),
                    unit="个",
                    threshold_warning=5,
                    threshold_critical=10
                )
                self._update_metric(conflicts_metric)
            
            # 知识更新成功率
            if "engine_statistics" in data:
                engine_stats = data["engine_statistics"]
                update_success_rate = engine_stats.get("update_success_rate", 0.0)
                update_success_metric = DashboardMetric(
                    metric_id="knowledge_update_success_rate",
                    name="知识更新成功率",
                    value=update_success_rate * 100,
                    unit="%",
                    threshold_warning=95.0,
                    threshold_critical=90.0
                )
                self._update_metric(update_success_metric)
                
        except Exception as e:
            logger.error(f"提取知识引擎指标失败: {e}")
    
    def _update_metric(self, metric: DashboardMetric):
        """更新指标"""
        # 计算趋势（如果历史数据存在）
        if metric.metric_id in self.metrics_history and len(self.metrics_history[metric.metric_id]) >= 2:
            historical_values = list(self.metrics_history[metric.metric_id])
            if len(historical_values) >= 2:
                recent_values = [m.value for m in historical_values[-2:]]
                if recent_values[0] != 0:
                    trend = (recent_values[1] - recent_values[0]) / recent_values[0] * 100
                    metric.trend = trend
        
        # 更新缓存
        self.metrics_cache[metric.metric_id] = metric
        
        # 添加到历史
        self.metrics_history[metric.metric_id].append(metric)
        
        # 存储到数据库
        self._store_metric_to_db(metric)
    
    def _store_monitoring_data(self, data: MonitoringData):
        """存储监控数据到数据库"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO monitoring_data 
                    (data_id, source, data_type, timestamp, data_json, tags_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    data.data_id,
                    data.source.value,
                    data.data_type.value,
                    data.timestamp,
                    json.dumps(data.data, default=str),
                    json.dumps(data.tags)
                ))
                
                conn.commit()
                conn.close()
                
                # 添加到内存历史
                self.historical_data.append(data)
                
        except Exception as e:
            logger.error(f"存储监控数据失败: {e}")
    
    def _store_metric_to_db(self, metric: DashboardMetric):
        """存储指标到数据库"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO metric_history 
                    (metric_id, name, value, unit, timestamp, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metric.metric_id,
                    metric.name,
                    metric.value,
                    metric.unit,
                    metric.timestamp,
                    json.dumps(metric.metadata, default=str)
                ))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"存储指标失败: {e}")
    
    def _aggregate_metrics(self):
        """聚合指标"""
        while self.running:
            try:
                # 计算每个指标的统计汇总
                aggregated_metrics = {}
                
                for metric_id, history in self.metrics_history.items():
                    if len(history) > 0:
                        values = [m.value for m in history]
                        timestamps = [m.timestamp for m in history]
                        
                        aggregated_metrics[metric_id] = {
                            "current": values[-1],
                            "average": np.mean(values) if values else 0.0,
                            "min": np.min(values) if values else 0.0,
                            "max": np.max(values) if values else 0.0,
                            "std": np.std(values) if len(values) > 1 else 0.0,
                            "count": len(values),
                            "last_updated": timestamps[-1] if timestamps else 0.0
                        }
                
                # 创建聚合数据
                data_id = f"aggregated_metrics_{int(time.time())}"
                monitoring_data = MonitoringData(
                    data_id=data_id,
                    source=DataSource.SYSTEM_MONITOR,
                    data_type=DataType.STATISTICAL,
                    timestamp=time.time(),
                    data=aggregated_metrics,
                    tags=["aggregated", "metrics", "statistics"]
                )
                
                # 存储聚合数据
                self._store_monitoring_data(monitoring_data)
                
                # 等待下一次聚合
                time.sleep(self.collection_config["metrics_aggregation_interval"])
                
            except Exception as e:
                logger.error(f"指标聚合失败: {e}")
                time.sleep(60)
    
    def _check_alerts(self):
        """检查警报"""
        while self.running:
            try:
                # 获取当前指标值
                current_metrics = self.get_current_metrics()
                
                # 检查每个警报规则
                for rule_id, rule in self.alert_rules.items():
                    if not rule.enabled:
                        continue
                    
                    try:
                        # 简化条件评估：检查相关指标是否超过阈值
                        # 这里实现简化的条件评估，实际应用中可以使用更复杂的规则引擎
                        condition_met = self._evaluate_alert_condition(rule.condition, current_metrics)
                        
                        if condition_met:
                            # 触发警报
                            self._trigger_alert(rule, current_metrics)
                            rule.last_triggered = time.time()
                            rule.trigger_count += 1
                            
                    except Exception as e:
                        logger.error(f"评估警报规则 {rule_id} 失败: {e}")
                
                # 等待下一次检查
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"警报检查失败: {e}")
                time.sleep(60)
    
    def _evaluate_alert_condition(self, condition: str, metrics: Dict[str, DashboardMetric]) -> bool:
        """评估警报条件（简化版）"""
        # 简化实现：解析条件字符串并检查指标值
        # 实际应用中应使用完整的规则引擎
        
        # 示例条件: "system.cpu_percent > 85"
        try:
            # 解析指标路径和阈值
            if ">" in condition:
                parts = condition.split(">")
                metric_path = parts[0].strip()
                threshold = float(parts[1].strip())
                
                # 根据路径获取指标值
                metric_value = self._get_metric_by_path(metric_path, metrics)
                
                return metric_value > threshold
            
            elif "<" in condition:
                parts = condition.split("<")
                metric_path = parts[0].strip()
                threshold = float(parts[1].strip())
                
                metric_value = self._get_metric_by_path(metric_path, metrics)
                
                return metric_value < threshold
            
            elif "==" in condition:
                parts = condition.split("==")
                metric_path = parts[0].strip()
                target_value = float(parts[1].strip())
                
                metric_value = self._get_metric_by_path(metric_path, metrics)
                
                return metric_value == target_value
            
        except Exception as e:
            logger.error(f"解析警报条件失败: {condition}, 错误: {e}")
        
        return False
    
    def _get_metric_by_path(self, path: str, metrics: Dict[str, DashboardMetric]) -> float:
        """根据路径获取指标值"""
        # 简化实现：根据指标ID直接查找
        # 实际应用中可能需要支持更复杂的路径解析
        
        # 映射路径到指标ID
        path_to_metric_id = {
            "system.cpu_percent": "system_cpu_percent",
            "system.memory_percent": "system_memory_percent",
            "scheduler.pending_tasks": "scheduler_queue_size",
            "scheduler.failure_rate": "scheduler_success_rate",  # 注意：这里是成功率，需要转换
            "knowledge.active_conflicts": "knowledge_active_conflicts"
        }
        
        metric_id = path_to_metric_id.get(path)
        if metric_id and metric_id in metrics:
            metric = metrics[metric_id]
            
            # 特殊处理：将成功率转换为失败率
            if path == "scheduler.failure_rate":
                return 100.0 - metric.value  # 失败率 = 100% - 成功率
            
            return metric.value
        
        return 0.0
    
    def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, DashboardMetric]):
        """触发警报"""
        alert_id = f"alert_{int(time.time())}_{rule.rule_id}"
        
        # 创建警报数据
        alert_data = {
            "rule_id": rule.rule_id,
            "condition": rule.condition,
            "metrics": {mid: m.value for mid, m in metrics.items()},
            "timestamp": time.time()
        }
        
        # 添加到内存警报列表
        self.alerts.append({
            "alert_id": alert_id,
            "severity": rule.severity,
            "message": f"警报触发: {rule.condition}",
            "data": alert_data,
            "timestamp": time.time(),
            "acknowledged": False
        })
        
        # 存储到数据库
        self._store_alert_to_db(alert_id, rule, alert_data)
        
        # 执行警报动作
        self._execute_alert_action(rule.action, alert_data)
        
        # 通知订阅者
        self._notify_subscribers("alert", {
            "alert_id": alert_id,
            "severity": rule.severity,
            "message": f"警报触发: {rule.condition}",
            "data": alert_data
        })
        
        logger.warning(f"警报触发: {rule.rule_id}, 条件: {rule.condition}")
    
    def _store_alert_to_db(self, alert_id: str, rule: AlertRule, alert_data: Dict[str, Any]):
        """存储警报到数据库"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO alerts 
                    (alert_id, source, severity, message, data_json, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    alert_id,
                    "alert_system",
                    rule.severity,
                    f"警报触发: {rule.condition}",
                    json.dumps(alert_data, default=str),
                    time.time()
                ))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"存储警报失败: {e}")
    
    def _execute_alert_action(self, action: str, alert_data: Dict[str, Any]):
        """执行警报动作"""
        # 根据动作类型执行相应操作
        action_handlers = {
            "notify_system_administrator": self._notify_system_administrator,
            "trigger_emergency_response": self._trigger_emergency_response,
            "suggest_memory_optimization": self._suggest_memory_optimization,
            "increase_scheduler_capacity": self._increase_scheduler_capacity,
            "investigate_scheduler_failures": self._investigate_scheduler_failures,
            "review_knowledge_conflicts": self._review_knowledge_conflicts
        }
        
        handler = action_handlers.get(action)
        if handler:
            try:
                handler(alert_data)
            except Exception as e:
                logger.error(f"执行警报动作 {action} 失败: {e}")
    
    def _notify_system_administrator(self, alert_data: Dict[str, Any]):
        """通知系统管理员"""
        logger.warning(f"系统管理员通知: {alert_data}")
        # 实际应用中，这里可以发送邮件、短信或调用通知系统
    
    def _trigger_emergency_response(self, alert_data: Dict[str, Any]):
        """触发紧急响应"""
        logger.critical(f"触发紧急响应: {alert_data}")
        # 实际应用中，这里可以触发紧急预案，如重启服务、切换备用系统等
    
    def _suggest_memory_optimization(self, alert_data: Dict[str, Any]):
        """建议内存优化"""
        logger.info(f"内存优化建议: {alert_data}")
        # 实际应用中，这里可以生成优化建议或触发自动优化
    
    def _increase_scheduler_capacity(self, alert_data: Dict[str, Any]):
        """增加调度器容量"""
        logger.info(f"增加调度器容量: {alert_data}")
        # 实际应用中，这里可以动态调整调度器参数或增加资源
    
    def _investigate_scheduler_failures(self, alert_data: Dict[str, Any]):
        """调查调度器失败"""
        logger.warning(f"调查调度器失败: {alert_data}")
        # 实际应用中，这里可以触发详细日志收集或诊断
    
    def _review_knowledge_conflicts(self, alert_data: Dict[str, Any]):
        """审查知识冲突"""
        logger.info(f"审查知识冲突: {alert_data}")
        # 实际应用中，这里可以触发知识冲突分析或人工审查
    
    def _cleanup_old_data(self):
        """清理旧数据"""
        while self.running:
            try:
                # 计算删除阈值时间
                retention_days = self.collection_config["data_retention_days"]
                cutoff_time = time.time() - (retention_days * 24 * 3600)
                
                with self.db_lock:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    # 删除旧监控数据
                    cursor.execute('DELETE FROM monitoring_data WHERE timestamp < ?', (cutoff_time,))
                    
                    # 删除旧指标历史
                    cursor.execute('DELETE FROM metric_history WHERE timestamp < ?', (cutoff_time,))
                    
                    # 删除旧警报（保留已确认的）
                    cursor.execute('DELETE FROM alerts WHERE timestamp < ? AND acknowledged = 1', (cutoff_time,))
                    
                    # 删除旧事件
                    cursor.execute('DELETE FROM events WHERE timestamp < ?', (cutoff_time,))
                    
                    conn.commit()
                    conn.close()
                
                logger.info(f"数据清理完成，删除了 {cutoff_time} 之前的数据")
                
                # 每周清理一次
                time.sleep(7 * 24 * 3600)
                
            except Exception as e:
                logger.error(f"数据清理失败: {e}")
                time.sleep(24 * 3600)  # 失败后等待一天
    
    def add_alert_rule(self, rule_id: str, condition: str, severity: str, action: str):
        """添加警报规则"""
        rule = AlertRule(
            rule_id=rule_id,
            condition=condition,
            severity=severity,
            action=action
        )
        self.alert_rules[rule_id] = rule
        
        logger.info(f"警报规则已添加: {rule_id}")
    
    def get_realtime_data(self, source: Optional[DataSource] = None) -> Dict[str, Any]:
        """获取实时数据"""
        if source:
            return self.realtime_data.get(source, {})
        else:
            return dict(self.realtime_data)
    
    def get_current_metrics(self) -> Dict[str, DashboardMetric]:
        """获取当前指标"""
        return dict(self.metrics_cache)
    
    def get_metric_history(self, metric_id: str, hours: int = 24) -> List[DashboardMetric]:
        """获取指标历史"""
        if metric_id not in self.metrics_history:
            return []
        
        cutoff_time = time.time() - (hours * 3600)
        return [
            metric for metric in self.metrics_history[metric_id]
            if metric.timestamp >= cutoff_time
        ]
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取最近警报"""
        return list(self.alerts)[-limit:]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认警报"""
        # 在内存中标记为已确认
        for alert in self.alerts:
            if alert.get("alert_id") == alert_id:
                alert["acknowledged"] = True
                break
        
        # 在数据库中标记为已确认
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('UPDATE alerts SET acknowledged = 1 WHERE alert_id = ?', (alert_id,))
                conn.commit()
                conn.close()
                
                return True
                
        except Exception as e:
            logger.error(f"确认警报失败: {e}")
            return False
    
    def subscribe(self, event_type: str, callback: Callable):
        """订阅事件"""
        self.subscribers[event_type].append(callback)
    
    def _notify_subscribers(self, event_type: str, data: Dict[str, Any]):
        """通知订阅者"""
        for callback in self.subscribers.get(event_type, []):
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"通知订阅者失败: {e}")
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """获取仪表板摘要"""
        # 收集各个组件状态
        system_health = asyncio.run(self.health_checker.check_system_health())
        performance_status = self.performance_monitor.check_performance_health()
        scheduler_metrics = self.scheduler.get_scheduler_metrics()
        engine_statistics = self.knowledge_engine.get_engine_statistics()
        
        # 计算总体健康状态
        overall_health = "healthy"
        if (system_health.get("status") == "unhealthy" or 
            performance_status.get("status") == "degraded" or
            scheduler_metrics.get("success_rate", 100) < 90.0):
            overall_health = "degraded"
        
        if system_health.get("status") == "error":
            overall_health = "unhealthy"
        
        return {
            "timestamp": time.time(),
            "overall_health": overall_health,
            "system_health": system_health.get("status", "unknown"),
            "performance_status": performance_status.get("status", "unknown"),
            "scheduler_status": "healthy" if scheduler_metrics.get("success_rate", 100) >= 95.0 else "degraded",
            "knowledge_engine_status": "healthy" if engine_statistics.get("update_success_rate", 1.0) >= 0.95 else "degraded",
            "active_alerts": len([a for a in self.alerts if not a.get("acknowledged", False)]),
            "recent_events": len(self.events),
            "metrics_count": len(self.metrics_cache)
        }
    
    def shutdown(self):
        """关闭后端服务"""
        self.running = False
        
        # 等待所有采集线程结束
        for thread_name, thread in self.collection_threads.items():
            if thread and thread.is_alive():
                thread.join(timeout=10.0)
        
        logger.info("监控面板后端服务已关闭")


# 全局后端实例
_dashboard_backend_instance = None

def get_dashboard_backend() -> MonitoringDashboardBackend:
    """获取监控面板后端实例"""
    global _dashboard_backend_instance
    if _dashboard_backend_instance is None:
        _dashboard_backend_instance = MonitoringDashboardBackend()
    return _dashboard_backend_instance

def start_monitoring_backend():
    """启动监控后端"""
    backend = get_dashboard_backend()
    logger.info("监控面板后端服务已启动")

def stop_monitoring_backend():
    """停止监控后端"""
    global _dashboard_backend_instance
    if _dashboard_backend_instance is not None:
        _dashboard_backend_instance.shutdown()
        _dashboard_backend_instance = None
    logger.info("监控面板后端服务已停止")
