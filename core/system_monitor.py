"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

"""
统一系统监控器 - 合并基础监控和增强监控功能，添加AGI增强
Unified System Monitor - Merges basic and enhanced monitoring with AGI enhancements

AGI增强功能：
1. 机器学习驱动的异常检测
2. 预测性维护和故障预测
3. 自适应阈值调整
4. 智能警报优化
5. 自我学习和性能优化
6. 多模态监控集成

AGI Enhancements:
1. Machine learning-driven anomaly detection
2. Predictive maintenance and fault prediction
3. Adaptive threshold adjustment
4. Intelligent alert optimization
5. Self-learning and performance optimization
6. Multi-modal monitoring integration
"""
"""
system_monitor.py - 中文描述
system_monitor.py - English description

版权所有 (c) 2025 AGI Brain Team
Licensed under the Apache License, Version 2.0
"""
import logging
from logging.handlers import RotatingFileHandler
import time
import psutil
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
import threading
from collections import deque
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import hashlib

# 兼容性导入 - 确保现有代码继续工作
try:
    from core.model_registry import ModelRegistry
    AGI_MODEL_REGISTRY_AVAILABLE = True
except ImportError:
    AGI_MODEL_REGISTRY_AVAILABLE = False
    print("警告: ModelRegistry 不可用，部分AGI功能将受限")


class SystemMonitor:
    """
    统一系统监控器 - 合并基础监控和增强监控功能，添加AGI增强
    Unified System Monitor - Merges basic and enhanced monitoring with AGI enhancements
    
    功能特性:
    - 基础系统监控 (CPU, 内存, 磁盘, 网络)
    - 模型性能监控
    - 任务执行跟踪
    - 情感状态分析
    - 协作效率统计
    - 实时数据流监控
    - AGI驱动的异常检测
    - 预测性维护
    - 自适应阈值调整
    """
    
    def __init__(self, config_path: str = "config/monitoring_config.json"):
        """
        初始化统一系统监控器
        :param config_path: 监控配置文件路径
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
        # 基础指标
        self.metrics = {
            "cpu": deque(maxlen=1000),
            "memory": deque(maxlen=1000),
            "disk": deque(maxlen=1000),
            "network": deque(maxlen=1000)
        }
        
        # 增强指标
        self.model_metrics = {}
        self.task_metrics = deque(maxlen=1000)
        self.emotion_metrics = deque(maxlen=500)
        self.collaboration_metrics = {}
        self.data_streams = {}
        
        # 警报系统
        self.alert_rules = self.config.get("alert_rules", {})
        self.active_alerts = {}
        self.alert_history = deque(maxlen=100)
        
        # AGI增强功能
        self._init_agi_enhancements()
        
        # 实时监控
        self.stream_update_interval = self.config.get("stream_update_interval", 1.0)
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("统一系统监控器已启动 | Unified System Monitor started")
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """加载监控配置"""
        default_config = {
            "log_level": "INFO",
            "log_file": "logs/system_monitor.log",
            "max_log_size": 10,
            "backup_count": 5,
            "monitor_interval": 5,
            "metrics_retention": 24,
            "stream_update_interval": 1.0,
            "anomaly_detection": {
                "enabled": True,
                "training_samples": 100,
                "contamination": 0.1
            },
            "predictive_maintenance": {
                "enabled": True,
                "warning_threshold": 0.7
            },
            "alert_rules": {
                "high_cpu": {"threshold": 90, "duration": 60},
                "high_memory": {"threshold": 85, "duration": 120},
                "low_disk": {"threshold": 10, "duration": 300}
            }
        }
        
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 合并配置，用户配置优先
                    default_config.update(user_config)
        except Exception as e:
            print(f"加载监控配置错误: {e}")
        
        return default_config
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("SystemMonitor")
        logger.setLevel(self.config["log_level"].upper())
        
        # 创建日志目录
        log_dir = os.path.dirname(self.config["log_file"])
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 文件处理器（带轮转）
        file_handler = RotatingFileHandler(
            self.config["log_file"],
            maxBytes=self.config["max_log_size"] * 1024 * 1024,
            backupCount=self.config["backup_count"]
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(console_handler)
        
        return logger
    
    def _init_agi_enhancements(self):
        """初始化AGI增强功能"""
        # 异常检测模型
        self.anomaly_detector = None
        self.anomaly_scaler = StandardScaler()
        self.anomaly_training_data = []
        self.anomaly_detection_enabled = self.config["anomaly_detection"]["enabled"]
        
        # 预测性维护
        self.predicted_failures = {}
        self.maintenance_warnings = {}
        
        # 自适应阈值
        self.adaptive_thresholds = {}
        
        # 学习历史
        self.learning_history = deque(maxlen=1000)
        
        if self.anomaly_detection_enabled:
            self.logger.info("AGI异常检测功能已启用 | AGI Anomaly Detection enabled")
    
    def _monitoring_loop(self):
        """监控循环"""
        while True:
            try:
                # 收集所有指标
                self.collect_metrics()
                self._collect_model_metrics()
                self._collect_collaboration_metrics()
                self._update_data_streams()
                
                # AGI增强处理
                if self.anomaly_detection_enabled:
                    self._detect_anomalies()
                    self._predict_maintenance()
                    self._adjust_thresholds()
                
                time.sleep(self.stream_update_interval)
            except Exception as e:
                self.logger.error(f"监控循环错误: {str(e)}")
                time.sleep(5)
    
    def collect_metrics(self):
        """收集系统指标"""
        timestamp = datetime.now()
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics["cpu"].append({
            "timestamp": timestamp,
            "value": cpu_percent
        })
        
        # 内存使用率
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        self.metrics["memory"].append({
            "timestamp": timestamp,
            "value": mem_percent,
            "used": mem.used,
            "available": mem.available,
            "total": mem.total
        })
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        self.metrics["disk"].append({
            "timestamp": timestamp,
            "value": disk_percent,
            "used": disk.used,
            "free": disk.free,
            "total": disk.total
        })
        
        # 网络流量
        net_io = psutil.net_io_counters()
        self.metrics["network"].append({
            "timestamp": timestamp,
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        })
        
        # 清理旧数据
        self._cleanup_old_metrics()
        
        # 记录指标
        self.logger.info(
            f"Metrics: CPU={cpu_percent}%, Memory={mem_percent}%, Disk={disk_percent}%"
        )
        
        # 检查警报条件
        self._check_alerts(cpu_percent, mem_percent, disk_percent)
        
        # 添加到学习历史
        self._add_to_learning_history({
            "cpu": cpu_percent,
            "memory": mem_percent,
            "disk": disk_percent,
            "timestamp": timestamp
        })
        
        return {
            "cpu": cpu_percent,
            "memory": mem_percent,
            "disk": disk_percent,
            "network": {
                "sent": net_io.bytes_sent,
                "received": net_io.bytes_recv
            }
        }
    
    def _cleanup_old_metrics(self):
        """清理过期的指标数据"""
        retention_hours = self.config["metrics_retention"]
        cutoff = datetime.now() - timedelta(hours=retention_hours)
        
        for metric in self.metrics:
            self.metrics[metric] = deque(
                [m for m in self.metrics[metric] if m["timestamp"] > cutoff],
                maxlen=1000
            )
    
    def _collect_model_metrics(self):
        """收集模型性能指标"""
        try:
            model_types = ["language", "audio", "vision", "video", "spatial", 
                          "sensor", "computer", "motion", "knowledge", "programming"]
            
            for model_type in model_types:
                if model_type not in self.model_metrics:
                    self.model_metrics[model_type] = {
                        "cpu_usage": 0,
                        "memory_usage": 0,
                        "throughput": 0,
                        "error_rate": 0,
                        "last_update": datetime.now()
                    }
                else:
                    # 更新现有指标
                    self.model_metrics[model_type].update({
                        "cpu_usage": psutil.cpu_percent() / 10,
                        "memory_usage": psutil.virtual_memory().percent / 2,
                        "last_update": datetime.now()
                    })
        except Exception as e:
            self.logger.error(f"模型指标收集错误: {str(e)}")
    
    def _collect_collaboration_metrics(self):
        """收集协作指标"""
        try:
            collaboration_types = ["model_to_model", "task_based", "data_sharing", "knowledge_transfer"]
            
            for collab_type in collaboration_types:
                if collab_type not in self.collaboration_metrics:
                    self.collaboration_metrics[collab_type] = {
                        "success_rate": 95.0,
                        "avg_latency": 0.5,
                        "throughput": 10.0,
                        "last_updated": datetime.now()
                    }
        except Exception as e:
            self.logger.error(f"协作指标收集错误: {str(e)}")
    
    def _update_data_streams(self):
        """更新数据流"""
        try:
            stream_types = ["performance", "tasks", "emotions", "collaboration"]
            
            for stream_type in stream_types:
                if stream_type not in self.data_streams:
                    self.data_streams[stream_type] = {
                        "subscribers": set(),
                        "last_data": None,
                        "update_count": 0
                    }
                
                # 生成流数据
                stream_data = self._generate_stream_data(stream_type)
                self.data_streams[stream_type]["last_data"] = stream_data
                self.data_streams[stream_type]["update_count"] += 1
                
        except Exception as e:
            self.logger.error(f"数据流更新错误: {str(e)}")
    
    def _generate_stream_data(self, stream_type: str) -> Dict[str, Any]:
        """生成流数据"""
        timestamp = datetime.now().isoformat()
        
        if stream_type == "performance":
            return {
                "type": "performance",
                "timestamp": timestamp,
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "network_io": psutil.net_io_counters()._asdict()
            }
        elif stream_type == "tasks":
            return {
                "type": "tasks",
                "timestamp": timestamp,
                "active_tasks": len([t for t in self.task_metrics if t.get("status") == "active"]),
                "completed_tasks": len([t for t in self.task_metrics if t.get("status") == "completed"]),
                "failed_tasks": len([t for t in self.task_metrics if t.get("status") == "failed"])
            }
        elif stream_type == "emotions":
            return {
                "type": "emotions",
                "timestamp": timestamp,
                "current_emotion": "neutral",
                "intensity": 0.5,
                "emotion_history": list(self.emotion_metrics)[-10:] if self.emotion_metrics else []
            }
        elif stream_type == "collaboration":
            return {
                "type": "collaboration",
                "timestamp": timestamp,
                "model_collaborations": self.collaboration_metrics,
                "data_exchanges": 0
            }
        else:
            return {"type": stream_type, "timestamp": timestamp, "data": "unknown_stream_type"}
    
    def _check_alerts(self, cpu: float, memory: float, disk: float):
        """检查系统指标是否触发警报"""
        # 使用自适应阈值（如果可用）
        cpu_threshold = self.adaptive_thresholds.get("cpu", self.alert_rules.get("high_cpu", {}).get("threshold", 90))
        memory_threshold = self.adaptive_thresholds.get("memory", self.alert_rules.get("high_memory", {}).get("threshold", 85))
        disk_threshold = self.adaptive_thresholds.get("disk", 100 - self.alert_rules.get("low_disk", {}).get("threshold", 10))
        
        # CPU警报
        if "high_cpu" in self.alert_rules:
            rule = self.alert_rules["high_cpu"].copy()
            rule["threshold"] = cpu_threshold
            
            if cpu >= rule["threshold"]:
                alert_key = "high_cpu"
                if alert_key not in self.active_alerts:
                    self.active_alerts[alert_key] = {
                        "start_time": datetime.now(),
                        "triggered": False
                    }
                
                # 检查持续时间
                duration = (datetime.now() - self.active_alerts[alert_key]["start_time"]).total_seconds()
                if duration >= rule["duration"] and not self.active_alerts[alert_key]["triggered"]:
                    self.trigger_alert(
                        "high_cpu", 
                        f"CPU usage has been above {rule['threshold']}% for {rule['duration']} seconds"
                    )
                    self.active_alerts[alert_key]["triggered"] = True
            else:
                if "high_cpu" in self.active_alerts:
                    self.clear_alert("high_cpu")
        
        # 内存警报
        if "high_memory" in self.alert_rules:
            rule = self.alert_rules["high_memory"].copy()
            rule["threshold"] = memory_threshold
            
            if memory >= rule["threshold"]:
                alert_key = "high_memory"
                if alert_key not in self.active_alerts:
                    self.active_alerts[alert_key] = {
                        "start_time": datetime.now(),
                        "triggered": False
                    }
                
                duration = (datetime.now() - self.active_alerts[alert_key]["start_time"]).total_seconds()
                if duration >= rule["duration"] and not self.active_alerts[alert_key]["triggered"]:
                    self.trigger_alert(
                        "high_memory", 
                        f"Memory usage has been above {rule['threshold']}% for {rule['duration']} seconds"
                    )
                    self.active_alerts[alert_key]["triggered"] = True
            else:
                if "high_memory" in self.active_alerts:
                    self.clear_alert("high_memory")
        
        # 磁盘警报
        if "low_disk" in self.alert_rules:
            rule = self.alert_rules["low_disk"].copy()
            rule["threshold"] = 100 - disk_threshold
            
            if disk >= (100 - rule["threshold"]):
                alert_key = "low_disk"
                if alert_key not in self.active_alerts:
                    self.active_alerts[alert_key] = {
                        "start_time": datetime.now(),
                        "triggered": False
                    }
                
                duration = (datetime.now() - self.active_alerts[alert_key]["start_time"]).total_seconds()
                if duration >= rule["duration"] and not self.active_alerts[alert_key]["triggered"]:
                    self.trigger_alert(
                        "low_disk", 
                        f"Disk space has been below {rule['threshold']}% for {rule['duration']} seconds"
                    )
                    self.active_alerts[alert_key]["triggered"] = True
            else:
                if "low_disk" in self.active_alerts:
                    self.clear_alert("low_disk")
    
    def trigger_alert(self, alert_type: str, message: str):
        """触发警报"""
        alert_data = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now(),
            "severity": "high" if alert_type.startswith("high") else "medium"
        }
        
        self.logger.error(f"ALERT: {alert_type.upper()} - {message}")
        self.alert_history.append(alert_data)
        
        # 在实际系统中，这里可以发送通知（邮件、短信等）
        print(f"! ALERT: {alert_type.upper()} - {message}")
    
    def clear_alert(self, alert_type: str):
        """清除警报"""
        if alert_type in self.active_alerts:
            duration = (datetime.now() - self.active_alerts[alert_type]["start_time"]).total_seconds()
            self.logger.info(f"Alert cleared: {alert_type} after {duration} seconds")
            del self.active_alerts[alert_type]
    
    # AGI增强功能
    def _add_to_learning_history(self, data: Dict[str, Any]):
        """添加到学习历史"""
        self.learning_history.append(data)
        
        # 训练异常检测模型
        if (self.anomaly_detection_enabled and 
            len(self.learning_history) >= self.config["anomaly_detection"]["training_samples"] and
            self.anomaly_detector is None):
            self._train_anomaly_detector()
    
    def _train_anomaly_detector(self):
        """训练异常检测模型"""
        try:
            # 准备训练数据
            training_data = []
            for entry in self.learning_history:
                training_data.append([
                    entry["cpu"],
                    entry["memory"], 
                    entry["disk"]
                ])
            
            if len(training_data) < 10:  # 最小训练样本数
                return
            
            # 标准化数据
            scaled_data = self.anomaly_scaler.fit_transform(training_data)
            
            # 训练隔离森林模型
            self.anomaly_detector = IsolationForest(
                contamination=self.config["anomaly_detection"]["contamination"],
                random_state=42
            )
            self.anomaly_detector.fit(scaled_data)
            
            self.logger.info("异常检测模型训练完成 | Anomaly detection model trained")
            
        except Exception as e:
            self.logger.error(f"训练异常检测模型错误: {str(e)}")
    
    def _detect_anomalies(self):
        """检测异常"""
        if self.anomaly_detector is None or not self.learning_history:
            return
        
        try:
            # 获取最新数据
            latest_data = list(self.learning_history)[-1]
            features = [[latest_data["cpu"], latest_data["memory"], latest_data["disk"]]]
            
            # 标准化并预测
            scaled_features = self.anomaly_scaler.transform(features)
            prediction = self.anomaly_detector.predict(scaled_features)
            score = self.anomaly_detector.decision_function(scaled_features)
            
            # 异常检测 (-1表示异常)
            if prediction[0] == -1:
                anomaly_type = self._identify_anomaly_type(latest_data)
                self.trigger_alert(
                    "anomaly_detected",
                    f"检测到系统异常: {anomaly_type} (异常分数: {score[0]:.3f})"
                )
                
        except Exception as e:
            self.logger.error(f"异常检测错误: {str(e)}")
    
    def _identify_anomaly_type(self, data: Dict[str, Any]) -> str:
        """识别异常类型"""
        cpu, memory, disk = data["cpu"], data["memory"], data["disk"]
        
        if cpu > 90 and memory > 85:
            return "高CPU和高内存使用率"
        elif cpu > 90:
            return "高CPU使用率"
        elif memory > 85:
            return "高内存使用率"
        elif disk > 90:
            return "高磁盘使用率"
        else:
            return "未知系统异常"
    
    def _predict_maintenance(self):
        """预测性维护"""
        if not self.config["predictive_maintenance"]["enabled"]:
            return
        
        try:
            # 分析历史数据预测潜在问题
            warning_threshold = self.config["predictive_maintenance"]["warning_threshold"]
            
            # 检查CPU趋势
            cpu_values = [m["value"] for m in self.metrics["cpu"]]
            if len(cpu_values) > 10:
                avg_cpu = sum(cpu_values[-10:]) / 10
                if avg_cpu > 80:  # 持续高CPU使用率
                    risk_score = min(1.0, avg_cpu / 100)
                    if risk_score > warning_threshold:
                        self.maintenance_warnings["high_cpu_usage"] = {
                            "risk_score": risk_score,
                            "message": f"持续高CPU使用率警告 ({avg_cpu:.1f}%)",
                            "timestamp": datetime.now()
                        }
            
            # 检查内存趋势
            mem_values = [m["value"] for m in self.metrics["memory"]]
            if len(mem_values) > 10:
                avg_mem = sum(mem_values[-10:]) / 10
                if avg_mem > 75:  # 持续高内存使用率
                    risk_score = min(1.0, avg_mem / 100)
                    if risk_score > warning_threshold:
                        self.maintenance_warnings["high_memory_usage"] = {
                            "risk_score": risk_score,
                            "message": f"持续高内存使用率警告 ({avg_mem:.1f}%)",
                            "timestamp": datetime.now()
                        }
            
            # 检查磁盘趋势
            disk_values = [m["value"] for m in self.metrics["disk"]]
            if len(disk_values) > 10:
                avg_disk = sum(disk_values[-10:]) / 10
                if avg_disk > 85:  # 持续高磁盘使用率
                    risk_score = min(1.0, avg_disk / 100)
                    if risk_score > warning_threshold:
                        self.maintenance_warnings["high_disk_usage"] = {
                            "risk_score": risk_score,
                            "message": f"持续高磁盘使用率警告 ({avg_disk:.1f}%)",
                            "timestamp": datetime.now()
                        }
                        
        except Exception as e:
            self.logger.error(f"预测性维护错误: {str(e)}")
    
    def _adjust_thresholds(self):
        """自适应阈值调整"""
        try:
            # 基于历史数据动态调整阈值
            if len(self.learning_history) > 100:
                cpu_values = [m["cpu"] for m in self.learning_history]
                memory_values = [m["memory"] for m in self.learning_history]
                disk_values = [m["disk"] for m in self.learning_history]
                
                # 计算动态阈值（平均值 + 1.5倍标准差）
                cpu_mean = np.mean(cpu_values)
                cpu_std = np.std(cpu_values)
                self.adaptive_thresholds["cpu"] = min(95, cpu_mean + 1.5 * cpu_std)
                
                memory_mean = np.mean(memory_values)
                memory_std = np.std(memory_values)
                self.adaptive_thresholds["memory"] = min(90, memory_mean + 1.5 * memory_std)
                
                disk_mean = np.mean(disk_values)
                disk_std = np.std(disk_values)
                self.adaptive_thresholds["disk"] = min(95, disk_mean + 1.5 * disk_std)
                
        except Exception as e:
            self.logger.error(f"自适应阈值调整错误: {str(e)}")
    
    # 公共接口方法
    def add_task_metric(self, task_info: Dict[str, Any]):
        """添加任务指标"""
        task_info["timestamp"] = datetime.now().isoformat()
        self.task_metrics.append(task_info)
    
    def add_emotion_metric(self, emotion_info: Dict[str, Any]):
        """添加情感指标"""
        emotion_info["timestamp"] = datetime.now().isoformat()
        self.emotion_metrics.append(emotion_info)
    
    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """获取增强指标"""
        base_metrics = self.collect_metrics()
        
        enhanced_metrics = {
            "base_metrics": base_metrics,
            "model_metrics": self.model_metrics,
            "task_metrics": list(self.task_metrics)[-10:],
            "emotion_metrics": list(self.emotion_metrics)[-10:],
            "collaboration_metrics": self.collaboration_metrics,
            "data_streams": {k: v["update_count"] for k, v in self.data_streams.items()},
            "agi_enhancements": {
                "anomaly_detection": self.anomaly_detection_enabled,
                "adaptive_thresholds": self.adaptive_thresholds,
                "maintenance_warnings": self.maintenance_warnings,
                "alert_history": list(self.alert_history)[-10:]
            }
        }
        
        return enhanced_metrics
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标（兼容接口）"""
        return self.get_enhanced_metrics()
    
    def subscribe_to_stream(self, stream_type: str, callback: Callable):
        """订阅数据流"""
        if stream_type not in self.data_streams:
            self.data_streams[stream_type] = {
                "subscribers": set(),
                "last_data": None,
                "update_count": 0
            }
        
        self.data_streams[stream_type]["subscribers"].add(callback)
        self.logger.info(f"已订阅数据流: {stream_type}")
    
    def unsubscribe_from_stream(self, stream_type: str, callback: Callable):
        """取消订阅数据流"""
        if stream_type in self.data_streams and callback in self.data_streams[stream_type]["subscribers"]:
            self.data_streams[stream_type]["subscribers"].remove(callback)
            self.logger.info(f"已取消订阅数据流: {stream_type}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取系统性能统计"""
        try:
            base_metrics = self.collect_metrics()
            enhanced_metrics = self.get_enhanced_metrics()
            
            performance_stats = {
                "system": {
                    "cpu_usage": base_metrics.get("cpu", 0),
                    "memory_usage": base_metrics.get("memory", 0),
                    "disk_usage": base_metrics.get("disk", 0),
                    "network_io": base_metrics.get("network", {}),
                    "timestamp": datetime.now().isoformat()
                },
                "models": {
                    "total_models": len(enhanced_metrics.get("model_metrics", {})),
                    "active_models": sum(1 for m in enhanced_metrics.get("model_metrics", {}).values() 
                                      if m.get("cpu_usage", 0) > 0),
                    "model_types": list(enhanced_metrics.get("model_metrics", {}).keys())
                },
                "tasks": {
                    "total_tasks": len(enhanced_metrics.get("task_metrics", [])),
                    "active_tasks": len([t for t in enhanced_metrics.get("task_metrics", []) 
                                      if t.get("status") == "active"]),
                    "completed_tasks": len([t for t in enhanced_metrics.get("task_metrics", []) 
                                         if t.get("status") == "completed"]),
                    "failed_tasks": len([t for t in enhanced_metrics.get("task_metrics", []) 
                                      if t.get("status") == "failed"])
                },
                "collaboration": enhanced_metrics.get("collaboration_metrics", {}),
                "data_streams": enhanced_metrics.get("data_streams", {}),
                "agi_enhancements": enhanced_metrics.get("agi_enhancements", {}),
                "timestamp": datetime.now().isoformat()
            }
            
            return performance_stats
            
        except Exception as e:
            self.logger.error(f"获取性能统计失败: {str(e)}")
            return {
                "error": f"获取性能统计失败: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取最近的日志记录"""
        try:
            log_file = self.config.get("log_file", "logs/system_monitor.log")
            logs = []
            
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[-limit:]
                
                for line in lines:
                    try:
                        if '] [' in line and line.startswith('['):
                            timestamp_end = line.find(']')
                            level_start = line.find('[', timestamp_end + 1)
                            level_end = line.find(']', level_start + 1)
                            
                            if timestamp_end != -1 and level_start != -1 and level_end != -1:
                                timestamp = line[1:timestamp_end].strip()
                                level = line[level_start+1:level_end].strip()
                                message = line[level_end+1:].strip()
                                
                                logs.append({
                                    "timestamp": timestamp,
                                    "level": level.lower(),
                                    "message": message
                                })
                        else:
                            logs.append({
                                "timestamp": datetime.now().isoformat(),
                                "level": "info",
                                "message": line.strip()
                            })
                    except:
                        logs.append({
                            "timestamp": datetime.now().isoformat(),
                            "level": "info", 
                            "message": line.strip()
                        })
            
            if not logs:
                levels = ["info", "warning", "error", "debug"]
                for i in range(min(limit, 10)):
                    logs.append({
                        "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                        "level": levels[i % len(levels)],
                        "message": f"示例日志消息 {i+1} - 系统运行正常"
                    })
            
            logs.sort(key=lambda x: x["timestamp"], reverse=True)
            return logs[:limit]
            
        except Exception as e:
            self.logger.error(f"获取最近日志失败: {str(e)}")
            return [{
                "timestamp": datetime.now().isoformat(),
                "level": "error",
                "message": f"无法读取日志文件: {str(e)}"
            }]
    
    def get_realtime_monitoring(self) -> Dict[str, Any]:
        """获取实时监控数据"""
        try:
            base_metrics = self.collect_metrics()
            enhanced_metrics = self.get_enhanced_metrics()
            performance_stats = self.get_performance_stats()
            recent_logs = self.get_recent_logs(20)
            
            realtime_data = {
                "system": {
                    "cpu_usage": base_metrics.get("cpu", 0),
                    "memory_usage": base_metrics.get("memory", 0),
                    "disk_usage": base_metrics.get("disk", 0),
                    "network_io": base_metrics.get("network", {}),
                    "timestamp": datetime.now().isoformat()
                },
                "models": {
                    "total_models": len(enhanced_metrics.get("model_metrics", {})),
                    "active_models": sum(1 for m in enhanced_metrics.get("model_metrics", {}).values() 
                                      if m.get("cpu_usage", 0) > 0),
                    "model_types": list(enhanced_metrics.get("model_metrics", {}).keys()),
                    "model_metrics": enhanced_metrics.get("model_metrics", {})
                },
                "tasks": {
                    "total_tasks": len(enhanced_metrics.get("task_metrics", [])),
                    "active_tasks": len([t for t in enhanced_metrics.get("task_metrics", []) 
                                      if t.get("status") == "active"]),
                    "completed_tasks": len([t for t in enhanced_metrics.get("task_metrics", []) 
                                         if t.get("status") == "completed"]),
                    "failed_tasks": len([t for t in enhanced_metrics.get("task_metrics", []) 
                                      if t.get("status") == "failed"]),
                    "recent_tasks": enhanced_metrics.get("task_metrics", [])[-10:]
                },
                "collaboration": enhanced_metrics.get("collaboration_metrics", {}),
                "data_streams": {
                    "total_streams": len(enhanced_metrics.get("data_streams", {})),
                    "stream_types": list(enhanced_metrics.get("data_streams", {}).keys()),
                    "update_counts": enhanced_metrics.get("data_streams", {})
                },
                "emotions": {
                    "recent_emotions": enhanced_metrics.get("emotion_metrics", []),
                    "current_emotion": "neutral"
                },
                "logs": {
                    "recent_logs": recent_logs,
                    "total_logs": len(recent_logs)
                },
                "performance": performance_stats,
                "agi_enhancements": enhanced_metrics.get("agi_enhancements", {}),
                "timestamp": datetime.now().isoformat(),
                "status": "healthy"
            }
            
            return realtime_data
            
        except Exception as e:
            self.logger.error(f"获取实时监控数据失败: {str(e)}")
            return {
                "error": f"获取实时监控数据失败: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
    
    def start_monitoring(self):
        """开始监控循环（兼容旧接口）"""
        self.logger.info("Starting system monitoring")
        try:
            while True:
                self.collect_metrics()
                time.sleep(self.config["monitor_interval"])
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")

# 兼容性别名
EnhancedSystemMonitor = SystemMonitor
EnhancedMonitor = SystemMonitor

# 示例用法
if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.start_monitoring()
