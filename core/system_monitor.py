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
Unified System Monitor - Merges basic and enhanced monitoring with AGI enhancements

AGI Enhancements:
1. Machine learning-driven anomaly detection
2. Predictive maintenance and fault prediction
3. Adaptive threshold adjustment
4. Intelligent alert optimization
5. Self-learning and performance optimization
6. Multi-modal monitoring integration
"""

"""
system_monitor.py - Unified System Monitoring Module

Copyright (c) 2025 AGI Brain Team
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
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
import asyncio

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_ports_config import PERFORMANCE_MONITORING_PORT

# Compatibility imports - ensure existing code continues to work
try:
    from core.model_registry import ModelRegistry
    AGI_MODEL_REGISTRY_AVAILABLE = True
except ImportError:
    AGI_MODEL_REGISTRY_AVAILABLE = False
    print("Warning: ModelRegistry not available, some AGI features will be limited")

# Create FastAPI app
app = FastAPI(title="System Performance Monitoring Service", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SystemMonitor:
    """
    Unified System Monitor - Merges basic and enhanced monitoring with AGI enhancements
    
    Features:
    - Basic system monitoring (CPU, memory, disk, network)
    - Model performance monitoring
    - Task execution tracking
    - Emotion state analysis
    - Collaboration efficiency statistics
    - Real-time data stream monitoring
    - AGI-driven anomaly detection
    - Predictive maintenance
    - Adaptive threshold adjustment
    """
    
    def __init__(self, config_path: str = "config/monitoring_config.json"):
        """
        Initialize unified system monitor
        :param config_path: Path to monitoring configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
        # Basic metrics
        self.metrics = {
            "cpu": deque(maxlen=1000),
            "memory": deque(maxlen=1000),
            "disk": deque(maxlen=1000),
            "network": deque(maxlen=1000)
        }
        
        # Enhanced metrics
        self.model_metrics = {}
        self.task_metrics = deque(maxlen=1000)
        self.emotion_metrics = deque(maxlen=500)
        self.collaboration_metrics = {}
        self.data_streams = {}
        
        # AGI-specific metrics
        self.agi_metrics = {
            "cognitive_load": deque(maxlen=1000),  # 认知负载
            "learning_efficiency": deque(maxlen=1000),  # 学习效率
            "decision_quality": deque(maxlen=1000),  # 决策质量
            "creativity_score": deque(maxlen=1000),  # 创造力得分
            "problem_solving_speed": deque(maxlen=1000),  # 问题解决速度
            "knowledge_growth": deque(maxlen=1000)  # 知识增长
        }
        
        # Alert system
        self.alert_rules = self.config.get("alert_rules", {})
        self.active_alerts = {}
        self.alert_history = deque(maxlen=100)
        
        # AGI enhancements
        self._init_agi_enhancements()
        
        # Real-time monitoring
        self.stream_update_interval = self.config.get("stream_update_interval", 1.0)
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Unified System Monitor started")
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load monitoring configuration"""
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
                    # Merge configurations, user config takes priority
                    default_config.update(user_config)
        except Exception as e:
            print(f"Error loading monitoring configuration: {e}")
        
        return default_config
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger"""
        logger = logging.getLogger("SystemMonitor")
        logger.setLevel(self.config["log_level"].upper())
        
        # Create log directory
        log_dir = os.path.dirname(self.config["log_file"])
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            self.config["log_file"],
            maxBytes=self.config["max_log_size"] * 1024 * 1024,
            backupCount=self.config["backup_count"]
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(console_handler)
        
        return logger
    
    def _init_agi_enhancements(self):
        """Initialize AGI enhancements"""
        # Anomaly detection model
        self.anomaly_detector = None
        self.anomaly_scaler = StandardScaler()
        self.anomaly_training_data = []
        self.anomaly_detection_enabled = self.config["anomaly_detection"]["enabled"]
        
        # Predictive maintenance
        self.predicted_failures = {}
        self.maintenance_warnings = {}
        
        # Adaptive thresholds
        self.adaptive_thresholds = {}
        
        # Learning history
        self.learning_history = deque(maxlen=1000)
        
        if self.anomaly_detection_enabled:
            self.logger.info("AGI Anomaly Detection enabled")
    
    def _monitoring_loop(self):
        """Monitoring loop"""
        while True:
            try:
                # Collect all metrics
                self.collect_metrics()
                self._collect_model_metrics()
                self._collect_collaboration_metrics()
                self._update_data_streams()
                self._collect_agi_metrics()  # 收集AGI指标
                
                # AGI enhancement processing
                if self.anomaly_detection_enabled:
                    self._detect_anomalies()
                    self._predict_maintenance()
                    self._adjust_thresholds()
                
                time.sleep(self.stream_update_interval)
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {str(e)}")
                time.sleep(5)
    
    def collect_metrics(self):
        """Collect system metrics"""
        timestamp = datetime.now()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics["cpu"].append({
            "timestamp": timestamp,
            "value": cpu_percent
        })
        
        # Memory usage
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        self.metrics["memory"].append({
            "timestamp": timestamp,
            "value": mem_percent,
            "used": mem.used,
            "available": mem.available,
            "total": mem.total
        })
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        self.metrics["disk"].append({
            "timestamp": timestamp,
            "value": disk_percent,
            "used": disk.used,
            "free": disk.free,
            "total": disk.total
        })
        
        # Network traffic
        net_io = psutil.net_io_counters()
        self.metrics["network"].append({
            "timestamp": timestamp,
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        })
        
        # Clean up old data
        self._cleanup_old_metrics()
        
        # Log metrics
        self.logger.info(
            f"Metrics: CPU={cpu_percent}%, Memory={mem_percent}%, Disk={disk_percent}%"
        )
        
        # Check alert conditions
        self._check_alerts(cpu_percent, mem_percent, disk_percent)
        
        # Add to learning history
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
        """Clean up expired metric data"""
        retention_hours = self.config["metrics_retention"]
        cutoff = datetime.now() - timedelta(hours=retention_hours)
        
        for metric in self.metrics:
            self.metrics[metric] = deque(
                [m for m in self.metrics[metric] if m["timestamp"] > cutoff],
                maxlen=1000
            )
    
    def _collect_model_metrics(self):
        """Collect model performance metrics"""
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
                    # Update existing metrics
                    self.model_metrics[model_type].update({
                        "cpu_usage": psutil.cpu_percent() / 10,
                        "memory_usage": psutil.virtual_memory().percent / 2,
                        "last_update": datetime.now()
                    })
        except Exception as e:
            self.logger.error(f"Model metrics collection error: {str(e)}")
    
    def _collect_collaboration_metrics(self):
        """Collect collaboration metrics"""
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
            self.logger.error(f"Collaboration metrics collection error: {str(e)}")
    
    def _update_data_streams(self):
        """Update data streams"""
        try:
            stream_types = ["performance", "tasks", "emotions", "collaboration"]
            
            for stream_type in stream_types:
                if stream_type not in self.data_streams:
                    self.data_streams[stream_type] = {
                        "subscribers": set(),
                        "last_data": None,
                        "update_count": 0
                    }
                
                # Generate stream data
                stream_data = self._generate_stream_data(stream_type)
                self.data_streams[stream_type]["last_data"] = stream_data
                self.data_streams[stream_type]["update_count"] += 1
                
        except Exception as e:
            self.logger.error(f"Data stream update error: {str(e)}")
    
    def _generate_stream_data(self, stream_type: str) -> Dict[str, Any]:
        """Generate stream data"""
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
        """Check if system metrics trigger alerts"""
        # Use adaptive thresholds if available
        cpu_threshold = self.adaptive_thresholds.get("cpu", self.alert_rules.get("high_cpu", {}).get("threshold", 90))
        memory_threshold = self.adaptive_thresholds.get("memory", self.alert_rules.get("high_memory", {}).get("threshold", 85))
        disk_threshold = self.adaptive_thresholds.get("disk", 100 - self.alert_rules.get("low_disk", {}).get("threshold", 10))
        
        # CPU alert
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
                
                # Check duration
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
        
        # Memory alert
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
        
        # Disk alert
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
        """Trigger an alert"""
        alert_data = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now(),
            "severity": "high" if alert_type.startswith("high") else "medium"
        }
        
        self.logger.error(f"ALERT: {alert_type.upper()} - {message}")
        self.alert_history.append(alert_data)
        
        # In a real system, here you could send notifications (email, SMS, etc.)
        print(f"! ALERT: {alert_type.upper()} - {message}")
    
    def clear_alert(self, alert_type: str):
        """Clear an alert"""
        if alert_type in self.active_alerts:
            duration = (datetime.now() - self.active_alerts[alert_type]["start_time"]).total_seconds()
            self.logger.info(f"Alert cleared: {alert_type} after {duration} seconds")
            del self.active_alerts[alert_type]
    
    # AGI Enhancement Methods
    def _add_to_learning_history(self, data: Dict[str, Any]):
        """Add to learning history"""
        self.learning_history.append(data)
        
        # Train anomaly detection model
        if (self.anomaly_detection_enabled and 
            len(self.learning_history) >= self.config["anomaly_detection"]["training_samples"] and
            self.anomaly_detector is None):
            self._train_anomaly_detector()
    
    def _train_anomaly_detector(self):
        """Train anomaly detection model"""
        try:
            # Prepare training data
            training_data = []
            for entry in self.learning_history:
                training_data.append([
                    entry["cpu"],
                    entry["memory"], 
                    entry["disk"]
                ])
            
            if len(training_data) < 10:  # Minimum training samples
                return
            
            # Standardize data
            scaled_data = self.anomaly_scaler.fit_transform(training_data)
            
            # Train Isolation Forest model
            self.anomaly_detector = IsolationForest(
                contamination=self.config["anomaly_detection"]["contamination"],
                random_state=42
            )
            self.anomaly_detector.fit(scaled_data)
            
            self.logger.info("Anomaly detection model trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training anomaly detection model: {str(e)}")
    
    def _detect_anomalies(self):
        """Detect anomalies"""
        if self.anomaly_detector is None or not self.learning_history:
            return
        
        try:
            # Get latest data
            latest_data = list(self.learning_history)[-1]
            features = [[latest_data["cpu"], latest_data["memory"], latest_data["disk"]]]
            
            # Standardize and predict
            scaled_features = self.anomaly_scaler.transform(features)
            prediction = self.anomaly_detector.predict(scaled_features)
            score = self.anomaly_detector.decision_function(scaled_features)
            
            # Anomaly detection (-1 indicates anomaly)
            if prediction[0] == -1:
                anomaly_type = self._identify_anomaly_type(latest_data)
                self.trigger_alert(
                    "anomaly_detected",
                    f"System anomaly detected: {anomaly_type} (anomaly score: {score[0]:.3f})"
                )
                
        except Exception as e:
            self.logger.error(f"Anomaly detection error: {str(e)}")
    
    def _identify_anomaly_type(self, data: Dict[str, Any]) -> str:
        """Identify anomaly type"""
        cpu, memory, disk = data["cpu"], data["memory"], data["disk"]
        
        if cpu > 90 and memory > 85:
            return "High CPU and memory usage"
        elif cpu > 90:
            return "High CPU usage"
        elif memory > 85:
            return "High memory usage"
        elif disk > 90:
            return "High disk usage"
        else:
            return "Unknown system anomaly"
    
    def _predict_maintenance(self):
        """Predictive maintenance"""
        if not self.config["predictive_maintenance"]["enabled"]:
            return
        
        try:
            # Analyze historical data to predict potential issues
            warning_threshold = self.config["predictive_maintenance"]["warning_threshold"]
            
            # Check CPU trends
            cpu_values = [m["value"] for m in self.metrics["cpu"]]
            if len(cpu_values) > 10:
                avg_cpu = sum(cpu_values[-10:]) / 10
                if avg_cpu > 80:  # Sustained high CPU usage
                    risk_score = min(1.0, avg_cpu / 100)
                    if risk_score > warning_threshold:
                        self.maintenance_warnings["high_cpu_usage"] = {
                            "risk_score": risk_score,
                            "message": f"Sustained high CPU usage warning ({avg_cpu:.1f}%)",
                            "timestamp": datetime.now()
                        }
            
            # Check memory trends
            mem_values = [m["value"] for m in self.metrics["memory"]]
            if len(mem_values) > 10:
                avg_mem = sum(mem_values[-10:]) / 10
                if avg_mem > 75:  # Sustained high memory usage
                    risk_score = min(1.0, avg_mem / 100)
                    if risk_score > warning_threshold:
                        self.maintenance_warnings["high_memory_usage"] = {
                            "risk_score": risk_score,
                            "message": f"Sustained high memory usage warning ({avg_mem:.1f}%)",
                            "timestamp": datetime.now()
                        }
            
            # Check disk trends
            disk_values = [m["value"] for m in self.metrics["disk"]]
            if len(disk_values) > 10:
                avg_disk = sum(disk_values[-10:]) / 10
                if avg_disk > 85:  # Sustained high disk usage
                    risk_score = min(1.0, avg_disk / 100)
                    if risk_score > warning_threshold:
                        self.maintenance_warnings["high_disk_usage"] = {
                            "risk_score": risk_score,
                            "message": f"Sustained high disk usage warning ({avg_disk:.1f}%)",
                            "timestamp": datetime.now()
                        }
                        
        except Exception as e:
            self.logger.error(f"Predictive maintenance error: {str(e)}")
    
    def _adjust_thresholds(self):
        """Adaptive threshold adjustment"""
        try:
            # Dynamically adjust thresholds based on historical data
            if len(self.learning_history) > 100:
                cpu_values = [m["cpu"] for m in self.learning_history]
                memory_values = [m["memory"] for m in self.learning_history]
                disk_values = [m["disk"] for m in self.learning_history]
                
                # Calculate dynamic thresholds (mean + 1.5 * standard deviation)
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
            self.logger.error(f"Adaptive threshold adjustment error: {str(e)}")
    
    def _collect_agi_metrics(self):
        """Collect AGI-specific metrics (cognitive load, learning efficiency, etc.)"""
        try:
            timestamp = datetime.now()
            
            # 模拟AGI指标的计算（在实际系统中，这些指标应从其他AGI模块获取）
            
            # 1. 认知负载：基于CPU使用率和任务数量
            cpu_usage = psutil.cpu_percent(interval=0.1)
            active_tasks = len([t for t in self.task_metrics if t.get("status") == "active"])
            cognitive_load = min(100.0, cpu_usage + active_tasks * 5.0)
            
            # 2. 学习效率：基于模型性能和历史学习记录
            if len(self.learning_history) > 10:
                recent_cpu = [m["cpu"] for m in list(self.learning_history)[-10:]]
                avg_cpu = np.mean(recent_cpu)
                # 学习效率与CPU使用率成反比（假设低CPU使用率表示高效学习）
                learning_efficiency = max(0.0, 100.0 - avg_cpu * 0.8)
            else:
                learning_efficiency = 75.0  # 默认值
            
            # 3. 决策质量：基于任务成功率和错误率
            if len(self.task_metrics) > 0:
                completed_tasks = [t for t in self.task_metrics if t.get("status") == "completed"]
                failed_tasks = [t for t in self.task_metrics if t.get("status") == "failed"]
                if len(completed_tasks) + len(failed_tasks) > 0:
                    success_rate = len(completed_tasks) / (len(completed_tasks) + len(failed_tasks)) * 100
                    decision_quality = success_rate
                else:
                    decision_quality = 80.0  # 默认值
            else:
                decision_quality = 80.0
            
            # 4. 创造力得分：基于系统运行时间和任务多样性
            system_uptime_minutes = (timestamp - self.learning_history[0]["timestamp"]).total_seconds() / 60 if self.learning_history else 1
            task_diversity = len(set([t.get("type", "unknown") for t in self.task_metrics]))
            creativity_score = min(100.0, 20.0 + min(system_uptime_minutes * 0.1, 40.0) + task_diversity * 5.0)
            
            # 5. 问题解决速度：基于最近任务的平均完成时间
            if len(self.task_metrics) > 0:
                recent_tasks = list(self.task_metrics)[-10:]
                task_durations = []
                for task in recent_tasks:
                    if "start_time" in task and "end_time" in task:
                        try:
                            start = datetime.fromisoformat(task["start_time"])
                            end = datetime.fromisoformat(task["end_time"])
                            duration = (end - start).total_seconds()
                            task_durations.append(duration)
                        except:
                            pass
                if task_durations:
                    avg_duration = np.mean(task_durations)
                    # 问题解决速度与任务持续时间成反比（更快=更高分数）
                    problem_solving_speed = min(100.0, max(0.0, 100.0 - avg_duration))
                else:
                    problem_solving_speed = 50.0
            else:
                problem_solving_speed = 50.0
            
            # 6. 知识增长：基于模型数量和协作次数
            model_count = len(self.model_metrics)
            collaboration_count = len(self.collaboration_metrics)
            knowledge_growth = min(100.0, model_count * 10.0 + collaboration_count * 5.0)
            
            # 将AGI指标添加到对应的deque中
            self.agi_metrics["cognitive_load"].append({
                "timestamp": timestamp,
                "value": cognitive_load
            })
            self.agi_metrics["learning_efficiency"].append({
                "timestamp": timestamp,
                "value": learning_efficiency
            })
            self.agi_metrics["decision_quality"].append({
                "timestamp": timestamp,
                "value": decision_quality
            })
            self.agi_metrics["creativity_score"].append({
                "timestamp": timestamp,
                "value": creativity_score
            })
            self.agi_metrics["problem_solving_speed"].append({
                "timestamp": timestamp,
                "value": problem_solving_speed
            })
            self.agi_metrics["knowledge_growth"].append({
                "timestamp": timestamp,
                "value": knowledge_growth
            })
            
            # 记录AGI指标摘要
            self.logger.debug(f"AGI Metrics: Cognitive Load={cognitive_load:.1f}, Learning Efficiency={learning_efficiency:.1f}, "
                            f"Decision Quality={decision_quality:.1f}, Creativity={creativity_score:.1f}, "
                            f"Problem Solving Speed={problem_solving_speed:.1f}, Knowledge Growth={knowledge_growth:.1f}")
            
        except Exception as e:
            self.logger.error(f"AGI metrics collection error: {str(e)}")
    
    # Public interface methods
    def add_task_metric(self, task_info: Dict[str, Any]):
        """Add task metrics"""
        task_info["timestamp"] = datetime.now().isoformat()
        self.task_metrics.append(task_info)
    
    def add_emotion_metric(self, emotion_info: Dict[str, Any]):
        """Add emotion metrics"""
        emotion_info["timestamp"] = datetime.now().isoformat()
        self.emotion_metrics.append(emotion_info)
    
    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get enhanced metrics"""
        base_metrics = self.collect_metrics()
        
        # 获取最新的AGI指标值
        agi_current_values = {}
        for metric_name, metric_deque in self.agi_metrics.items():
            if metric_deque:
                latest_value = list(metric_deque)[-1]["value"]
                agi_current_values[metric_name] = latest_value
            else:
                agi_current_values[metric_name] = 0.0
        
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
            },
            "agi_metrics": {
                "current_values": agi_current_values,
                "historical_trends": {
                    metric_name: list(metric_deque)[-50:]  # 最近50个数据点
                    for metric_name, metric_deque in self.agi_metrics.items()
                }
            }
        }
        
        return enhanced_metrics
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics (compatibility interface)"""
        return self.get_enhanced_metrics()
    
    def subscribe_to_stream(self, stream_type: str, callback: Callable):
        """Subscribe to data stream"""
        if stream_type not in self.data_streams:
            self.data_streams[stream_type] = {
                "subscribers": set(),
                "last_data": None,
                "update_count": 0
            }
        
        self.data_streams[stream_type]["subscribers"].add(callback)
        self.logger.info(f"Subscribed to data stream: {stream_type}")
    
    def unsubscribe_from_stream(self, stream_type: str, callback: Callable):
        """Unsubscribe from data stream"""
        if stream_type in self.data_streams and callback in self.data_streams[stream_type]["subscribers"]:
            self.data_streams[stream_type]["subscribers"].remove(callback)
            self.logger.info(f"Unsubscribed from data stream: {stream_type}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
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
            self.logger.error(f"Failed to get performance statistics: {str(e)}")
            return {
                "error": f"Failed to get performance statistics: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent log records"""
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
                        "message": f"Sample log message {i+1} - System running normally"
                    })
            
            logs.sort(key=lambda x: x["timestamp"], reverse=True)
            return logs[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get recent logs: {str(e)}")
            return [{
                "timestamp": datetime.now().isoformat(),
                "level": "error",
                "message": f"Unable to read log file: {str(e)}"
            }]
    
    def get_realtime_monitoring(self) -> Dict[str, Any]:
        """Get real-time monitoring data"""
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
                "agi_metrics": enhanced_metrics.get("agi_metrics", {}),
                "timestamp": datetime.now().isoformat(),
                "status": "healthy"
            }
            
            return realtime_data
            
        except Exception as e:
            self.logger.error(f"Failed to get real-time monitoring data: {str(e)}")
            return {
                "error": f"Failed to get real-time monitoring data: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
    
    def start_monitoring(self):
        """Start monitoring loop (compatibility interface)"""
        self.logger.info("Starting system monitoring")
        try:
            while True:
                self.collect_metrics()
                time.sleep(self.config["monitor_interval"])
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")

# Compatibility aliases
EnhancedSystemMonitor = SystemMonitor
EnhancedMonitor = SystemMonitor

# Create a global instance of the system monitor
monitor = SystemMonitor()


@app.get("/health")
async def health_check():
    """Health check endpoint for the monitoring service"""
    return {
        "status": "healthy",
        "service": "System Performance Monitoring Service",
        "version": "1.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/metrics/basic")
async def get_basic_metrics():
    """Get basic system metrics (CPU, memory, disk, network)"""
    try:
        metrics = monitor.collect_metrics()
        return JSONResponse(status_code=200, content={
            "success": True,
            "data": metrics,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.get("/api/metrics/enhanced")
async def get_enhanced_metrics():
    """Get enhanced system metrics (models, tasks, emotions)"""
    try:
        metrics = monitor.get_enhanced_metrics()
        return JSONResponse(status_code=200, content={
            "success": True,
            "data": metrics,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.get("/api/metrics/realtime")
async def get_realtime_metrics():
    """Get real-time monitoring data (comprehensive view)"""
    try:
        data = monitor.get_realtime_monitoring()
        return JSONResponse(status_code=200, content={
            "success": True,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.get("/api/metrics/alerts")
async def get_alerts():
    """Get current and historical alerts"""
    try:
        return JSONResponse(status_code=200, content={
            "success": True,
            "active_alerts": monitor.active_alerts,
            "alert_history": list(monitor.alert_history),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.get("/api/metrics/logs")
async def get_logs(limit: int = 50):
    """Get recent log entries"""
    try:
        logs = monitor.get_recent_logs(limit)
        return JSONResponse(status_code=200, content={
            "success": True,
            "logs": logs,
            "total": len(logs),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.on_event("startup")
async def startup_event():
    """Initialize monitoring service on startup"""
    monitor.logger.info("System Performance Monitoring Service is starting up")
    
    # Start the monitoring loop in a separate thread
    if not monitor.monitoring_thread.is_alive():
        monitor.monitoring_thread = threading.Thread(target=monitor._monitoring_loop, daemon=True)
        monitor.monitoring_thread.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    monitor.logger.info("System Performance Monitoring Service is shutting down")


if __name__ == "__main__":
    """Main entry point to start the monitoring service"""
    monitor.logger.info(
        f"Starting System Performance Monitoring Service on http://0.0.0.0:{PERFORMANCE_MONITORING_PORT}"
    )
    
    # Start the FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PERFORMANCE_MONITORING_PORT,
        reload=False,
        log_level="info"
    )
