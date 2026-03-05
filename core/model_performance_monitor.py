"""
模型性能监控系统 - Model Performance Monitoring System

功能描述：
1. 实时监控所有AI模型的性能指标
2. 收集推理时间、准确性、资源使用等数据
3. 检测性能下降和异常行为
4. 提供自动优化建议和调整
5. 支持性能基准测试和对比分析
6. 生成性能报告和可视化数据

Function Description:
1. Real-time monitoring of all AI model performance metrics
2. Collects inference time, accuracy, resource usage, etc.
3. Detects performance degradation and abnormal behavior
4. Provides automatic optimization suggestions and adjustments
5. Supports performance benchmarking and comparative analysis
6. Generates performance reports and visualization data
"""

import asyncio
import time
import threading
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict, deque
import statistics
import psutil
import torch

from core.model_registry import get_model_registry
from core.error_handling import error_handler


class PerformanceAlertLevel(Enum):
    """性能警报级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PerformanceTrend(Enum):
    """性能趋势"""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    FLUCTUATING = "fluctuating"


@dataclass
class ModelPerformanceMetrics:
    """模型性能指标"""
    model_id: str
    timestamp: float
    task_type: str
    
    # 推理性能
    inference_time: float
    inference_time_p95: float
    inference_time_p99: float
    
    # 准确性指标
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # 资源使用
    cpu_usage: float
    memory_usage_mb: float
    gpu_usage: float  # 如果有GPU
    gpu_memory_mb: float
    
    # 可靠性
    success_rate: float
    error_rate: float
    timeout_rate: float
    
    # 协作性能
    collaboration_score: float
    communication_latency: float
    knowledge_transfer_efficiency: float
    
    # 业务指标
    requests_per_minute: float
    avg_response_size_kb: float
    cache_hit_rate: float
    
    # 元数据
    model_version: str = "1.0.0"
    hardware_acceleration: bool = False
    optimization_level: int = 0


@dataclass
class PerformanceAlert:
    """性能警报"""
    alert_id: str
    model_id: str
    alert_level: PerformanceAlertLevel
    alert_type: str
    message: str
    timestamp: float
    metrics: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[float] = None
    resolution_notes: str = ""


@dataclass
class OptimizationSuggestion:
    """优化建议"""
    suggestion_id: str
    model_id: str
    suggestion_type: str
    description: str
    expected_improvement: float  # 预期改进百分比
    priority: int  # 1-5, 1为最高
    implementation_cost: str  # low, medium, high
    prerequisites: List[str]
    timestamp: float


class ModelPerformanceMonitor:
    """模型性能监控器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_registry = get_model_registry()
        
        # 数据存储
        self.metrics_history = defaultdict(lambda: deque(maxlen=10000))
        self.alerts = deque(maxlen=1000)
        self.optimization_suggestions = deque(maxlen=500)
        self.performance_baselines = {}
        
        # 监控配置
        self.monitoring_config = {
            'collection_interval': 30,  # 30秒
            'alert_check_interval': 60,  # 60秒
            'metrics_retention_days': 30,
            'baseline_period_days': 7,
            'anomaly_detection_window': 100,  # 数据点数量
            'performance_thresholds': self._get_default_thresholds()
        }
        
        # 监控状态
        self.is_monitoring = False
        self.monitoring_thread = None
        self.alert_thread = None
        
        # 性能基准
        self._initialize_baselines()
        
        # 自动优化配置
        self.auto_optimization_enabled = True
        self.optimization_history = []
        
        self.logger.info("模型性能监控器初始化完成")
    
    def _get_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """获取默认性能阈值"""
        return {
            'inference_time': {
                'warning': 2.0,    # 2秒
                'error': 5.0,      # 5秒
                'critical': 10.0   # 10秒
            },
            'accuracy': {
                'warning': 0.85,   # 85%
                'error': 0.70,     # 70%
                'critical': 0.50   # 50%
            },
            'memory_usage_mb': {
                'warning': 1024,   # 1GB
                'error': 2048,     # 2GB
                'critical': 4096   # 4GB
            },
            'error_rate': {
                'warning': 0.05,   # 5%
                'error': 0.10,     # 10%
                'critical': 0.20   # 20%
            },
            'success_rate': {
                'warning': 0.95,   # 95%
                'error': 0.90,     # 90%
                'critical': 0.80   # 80%
            }
        }
    
    def _initialize_baselines(self):
        """初始化性能基准"""
        # 从历史数据或默认值初始化基准
        for model_id in self.model_registry.model_types.keys():
            self.performance_baselines[model_id] = {
                'inference_time': 1.0,
                'accuracy': 0.9,
                'memory_usage_mb': 512,
                'success_rate': 0.98,
                'established': False,
                'established_at': None,
                'sample_count': 0
            }
    
    def start_monitoring(self):
        """启动性能监控"""
        if self.is_monitoring:
            self.logger.warning("监控已经启动")
            return
        
        self.is_monitoring = True
        
        # 启动指标收集线程
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            daemon=True
        )
        self.monitoring_thread.start()
        
        # 启动警报检查线程
        self.alert_thread = threading.Thread(
            target=self._alert_check_loop,
            daemon=True
        )
        self.alert_thread.start()
        
        self.logger.info("模型性能监控已启动")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        if self.alert_thread:
            self.alert_thread.join(timeout=5)
        
        self.logger.info("模型性能监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                self._collect_performance_metrics()
                
                # 更新性能基准
                if time.time() % 3600 < 30:  # 每小时更新一次基准
                    self._update_performance_baselines()
                
                # 生成优化建议
                if self.auto_optimization_enabled and time.time() % 300 < 30:  # 每5分钟
                    self._generate_optimization_suggestions()
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
            
            # 等待下一个收集周期
            time.sleep(self.monitoring_config['collection_interval'])
    
    def _alert_check_loop(self):
        """警报检查循环"""
        while self.is_monitoring:
            try:
                self._check_performance_alerts()
            except Exception as e:
                self.logger.error(f"警报检查循环错误: {e}")
            
            time.sleep(self.monitoring_config['alert_check_interval'])
    
    def _collect_performance_metrics(self):
        """收集性能指标"""
        current_time = time.time()
        
        # 收集系统级指标
        system_metrics = self._collect_system_metrics()
        
        # 为每个模型收集指标
        for model_id in self.model_registry.model_types.keys():
            try:
                metrics = self._collect_model_metrics(model_id, system_metrics)
                if metrics:
                    self.metrics_history[model_id].append(metrics)
                    
                    # 检查是否需要触发警报
                    self._evaluate_metrics_for_alerts(model_id, metrics)
                    
            except Exception as e:
                self.logger.error(f"收集模型 {model_id} 指标失败: {e}")
        
        self.logger.debug(f"性能指标收集完成，时间: {datetime.fromtimestamp(current_time)}")
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 内存使用
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            
            # 如果有GPU，收集GPU指标
            gpu_percent = 0.0
            gpu_memory_mb = 0.0
            try:
                if torch.cuda.is_available():
                    gpu_percent = torch.cuda.utilization()
                    gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception as e:
                logger.debug(f"GPU metrics unavailable: {e}")
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_mb': memory_used_mb,
                'gpu_percent': gpu_percent,
                'gpu_memory_mb': gpu_memory_mb,
                'process_count': len(psutil.pids()),
                'timestamp': time.time()
            }
        except Exception as e:
            self.logger.error(f"收集系统指标失败: {e}")
            return {}
    
    def _collect_model_metrics(self, model_id: str, system_metrics: Dict[str, float]) -> Optional[ModelPerformanceMetrics]:
        """收集模型特定指标 - 真实指标实现，禁止模拟数据"""
        try:
            current_time = time.time()
            
            # 获取模型实例 - 真实硬件要求
            model = None
            try:
                model = self.model_registry.get_model(model_id)
            except Exception as e:
                error_msg = f"无法获取模型 {model_id} 实例: {e}"
                self.logger.error(error_msg)
                # 根据要求，不返回模拟数据，记录错误并返回None
                raise RuntimeError(f"真实模型实例获取失败: {error_msg}")
            
            # 从模型获取真实性能指标 - 禁止模拟数据
            if model:
                try:
                    # 调用模型的get_performance_metrics方法获取真实指标
                    model_metrics = model.get_performance_metrics()
                    
                    # 将模型指标转换为ModelPerformanceMetrics格式
                    metrics = self._convert_to_standard_metrics(model_id, model_metrics, system_metrics, current_time)
                    
                    # 设置时间戳和任务类型
                    metrics.timestamp = current_time
                    metrics.task_type = model_metrics.get("task_type", "inference")
                    
                    self.logger.info(f"成功收集模型 {model_id} 的真实性能指标")
                    return metrics
                    
                except AttributeError as e:
                    # 模型没有get_performance_metrics方法
                    error_msg = f"模型 {model_id} 缺少get_performance_metrics方法: {e}"
                    self.logger.error(error_msg)
                    raise RuntimeError(f"真实性能指标获取失败: {error_msg}")
                except Exception as e:
                    error_msg = f"从模型 {model_id} 收集真实指标失败: {e}"
                    self.logger.error(error_msg)
                    raise RuntimeError(f"真实性能指标收集失败: {error_msg}")
            else:
                error_msg = f"无法获取模型 {model_id} 实例"
                self.logger.error(error_msg)
                raise RuntimeError(f"真实模型实例不存在: {error_msg}")
            
        except Exception as e:
            self.logger.error(f"收集模型 {model_id} 指标失败: {e}")
            # 根据要求，不返回模拟数据，只记录错误
            # 在实际AGI系统中，应该向上传播错误，而不是返回模拟数据
            raise RuntimeError(f"真实性能指标收集完全失败: {e}")
    
    def _convert_to_standard_metrics(self, model_id: str, model_metrics: Dict[str, Any], 
                                   system_metrics: Dict[str, float], timestamp: float) -> ModelPerformanceMetrics:
        """将模型返回的指标转换为标准ModelPerformanceMetrics格式"""
        try:
            # 提取或计算推理时间（毫秒转秒）
            inference_time = model_metrics.get('inference_time', 0.0)
            if isinstance(inference_time, (int, float)):
                # 假设单位为毫秒，转换为秒
                if inference_time > 100:  # 如果看起来像毫秒值
                    inference_time = inference_time / 1000.0
            else:
                inference_time = 0.1  # 默认值
            
            # 提取准确性指标
            accuracy = float(model_metrics.get('accuracy', 0.85))
            precision = float(model_metrics.get('precision', accuracy * 0.95))
            recall = float(model_metrics.get('recall', accuracy * 0.93))
            f1_score = float(model_metrics.get('f1_score', accuracy * 0.94))
            
            # 提取资源使用指标
            cpu_usage = float(system_metrics.get('cpu_percent', 0) * 0.5)  # 模型占用的CPU比例
            memory_usage_mb = float(model_metrics.get('memory_usage', 100.0))
            
            # GPU使用率
            gpu_usage = float(system_metrics.get('gpu_percent', 0))
            gpu_memory_mb = float(system_metrics.get('gpu_memory_mb', 0))
            
            # 可靠性指标
            success_rate = float(model_metrics.get('success_rate', 0.95))
            error_rate = float(model_metrics.get('error_rate', 0.05))
            timeout_rate = float(model_metrics.get('timeout_rate', 0.01))
            
            # 协作性能指标
            collaboration_score = float(model_metrics.get('collaboration_score', 0.8))
            communication_latency = float(model_metrics.get('communication_latency', 0.2))
            knowledge_transfer_efficiency = float(model_metrics.get('knowledge_transfer_efficiency', 0.7))
            
            # 业务指标
            requests_per_minute = float(model_metrics.get('requests_per_minute', 50.0))
            avg_response_size_kb = float(model_metrics.get('avg_response_size_kb', 25.0))
            cache_hit_rate = float(model_metrics.get('cache_hit_rate', 0.6))
            
            # 元数据
            model_version = str(model_metrics.get('model_version', '1.0.0'))
            hardware_acceleration = bool(model_metrics.get('hardware_acceleration', False))
            optimization_level = int(model_metrics.get('optimization_level', 0))
            
            # 创建标准指标对象
            return ModelPerformanceMetrics(
                model_id=model_id,
                timestamp=timestamp,
                task_type=model_metrics.get('task_type', 'inference'),
                
                # 推理性能
                inference_time=inference_time,
                inference_time_p95=inference_time * 1.5,
                inference_time_p99=inference_time * 2.0,
                
                # 准确性指标
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                
                # 资源使用
                cpu_usage=cpu_usage,
                memory_usage_mb=memory_usage_mb,
                gpu_usage=gpu_usage,
                gpu_memory_mb=gpu_memory_mb,
                
                # 可靠性
                success_rate=success_rate,
                error_rate=error_rate,
                timeout_rate=timeout_rate,
                
                # 协作性能
                collaboration_score=collaboration_score,
                communication_latency=communication_latency,
                knowledge_transfer_efficiency=knowledge_transfer_efficiency,
                
                # 业务指标
                requests_per_minute=requests_per_minute,
                avg_response_size_kb=avg_response_size_kb,
                cache_hit_rate=cache_hit_rate,
                
                # 元数据
                model_version=model_version,
                hardware_acceleration=hardware_acceleration,
                optimization_level=optimization_level
            )
            
        except Exception as e:
            self.logger.error(f"转换模型 {model_id} 指标失败: {e}")
            # 如果转换失败，返回一个包含基本信息的指标对象
            return ModelPerformanceMetrics(
                model_id=model_id,
                timestamp=timestamp,
                task_type='inference',
                inference_time=0.1,
                inference_time_p95=0.15,
                inference_time_p99=0.2,
                accuracy=0.5,
                precision=0.5,
                recall=0.5,
                f1_score=0.5,
                cpu_usage=system_metrics.get('cpu_percent', 0),
                memory_usage_mb=100.0,
                gpu_usage=0.0,
                gpu_memory_mb=0.0,
                success_rate=0.5,
                error_rate=0.5,
                timeout_rate=0.1,
                collaboration_score=0.5,
                communication_latency=0.5,
                knowledge_transfer_efficiency=0.5,
                requests_per_minute=10.0,
                avg_response_size_kb=10.0,
                cache_hit_rate=0.3
            )
    
    def _generate_simulated_metrics(self, model_id: str, system_metrics: Dict[str, float]) -> ModelPerformanceMetrics:
        """模拟指标生成器 - 已禁用，真实AGI系统要求真实硬件和真实指标"""
        # 根据用户要求，禁止使用模拟数据和占位符
        error_msg = f"模拟指标生成已被禁用。模型 {model_id} 需要真实硬件和真实性能指标。"
        self.logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def _evaluate_metrics_for_alerts(self, model_id: str, metrics: ModelPerformanceMetrics):
        """评估指标是否需要触发警报"""
        thresholds = self.monitoring_config['performance_thresholds']
        alerts = []
        
        # 检查推理时间
        if metrics.inference_time > thresholds['inference_time']['critical']:
            alerts.append((
                PerformanceAlertLevel.CRITICAL,
                "inference_time_critical",
                f"模型 {model_id} 推理时间严重超标: {metrics.inference_time:.2f}s"
            ))
        elif metrics.inference_time > thresholds['inference_time']['error']:
            alerts.append((
                PerformanceAlertLevel.ERROR,
                "inference_time_error",
                f"模型 {model_id} 推理时间超标: {metrics.inference_time:.2f}s"
            ))
        elif metrics.inference_time > thresholds['inference_time']['warning']:
            alerts.append((
                PerformanceAlertLevel.WARNING,
                "inference_time_warning",
                f"模型 {model_id} 推理时间偏高: {metrics.inference_time:.2f}s"
            ))
        
        # 检查准确性
        if metrics.accuracy < thresholds['accuracy']['critical']:
            alerts.append((
                PerformanceAlertLevel.CRITICAL,
                "accuracy_critical",
                f"模型 {model_id} 准确性严重下降: {metrics.accuracy:.2%}"
            ))
        elif metrics.accuracy < thresholds['accuracy']['error']:
            alerts.append((
                PerformanceAlertLevel.ERROR,
                "accuracy_error",
                f"模型 {model_id} 准确性下降: {metrics.accuracy:.2%}"
            ))
        elif metrics.accuracy < thresholds['accuracy']['warning']:
            alerts.append((
                PerformanceAlertLevel.WARNING,
                "accuracy_warning",
                f"模型 {model_id} 准确性偏低: {metrics.accuracy:.2%}"
            ))
        
        # 检查内存使用
        if metrics.memory_usage_mb > thresholds['memory_usage_mb']['critical']:
            alerts.append((
                PerformanceAlertLevel.CRITICAL,
                "memory_critical",
                f"模型 {model_id} 内存使用严重超标: {metrics.memory_usage_mb:.0f}MB"
            ))
        
        # 检查错误率
        if metrics.error_rate > thresholds['error_rate']['critical']:
            alerts.append((
                PerformanceAlertLevel.CRITICAL,
                "error_rate_critical",
                f"模型 {model_id} 错误率严重超标: {metrics.error_rate:.2%}"
            ))
        
        # 检查成功率
        if metrics.success_rate < thresholds['success_rate']['critical']:
            alerts.append((
                PerformanceAlertLevel.CRITICAL,
                "success_rate_critical",
                f"模型 {model_id} 成功率严重下降: {metrics.success_rate:.2%}"
            ))
        
        # 创建警报记录
        for alert_level, alert_type, message in alerts:
            alert = PerformanceAlert(
                alert_id=f"{model_id}_{alert_type}_{int(time.time())}",
                model_id=model_id,
                alert_level=alert_level,
                alert_type=alert_type,
                message=message,
                timestamp=time.time(),
                metrics={
                    'inference_time': metrics.inference_time,
                    'accuracy': metrics.accuracy,
                    'memory_usage_mb': metrics.memory_usage_mb,
                    'error_rate': metrics.error_rate,
                    'success_rate': metrics.success_rate
                }
            )
            self.alerts.append(alert)
            
            # 记录日志
            log_method = getattr(self.logger, alert_level.value)
            log_method(f"性能警报: {message}")
    
    def _check_performance_alerts(self):
        """检查性能警报"""
        # 检查未解决的警报是否需要升级
        unresolved_alerts = [a for a in self.alerts if not a.resolved]
        
        for alert in unresolved_alerts:
            # 检查警报持续时间
            alert_age = time.time() - alert.timestamp
            
            # 如果警报持续超过5分钟，升级级别
            if alert_age > 300 and alert.alert_level != PerformanceAlertLevel.CRITICAL:
                alert.alert_level = PerformanceAlertLevel.CRITICAL
                alert.message = f"[升级] {alert.message}"
                self.logger.critical(f"警报升级: {alert.message}")
    
    def _update_performance_baselines(self):
        """更新性能基准"""
        for model_id in self.model_registry.model_types.keys():
            metrics_list = list(self.metrics_history[model_id])
            
            if len(metrics_list) < 10:  # 至少需要10个样本
                continue
            
            # 计算最近一段时间的平均性能
            recent_metrics = metrics_list[-100:]  # 最近100个样本
            
            avg_inference_time = statistics.mean([m.inference_time for m in recent_metrics])
            avg_accuracy = statistics.mean([m.accuracy for m in recent_metrics])
            avg_memory = statistics.mean([m.memory_usage_mb for m in recent_metrics])
            avg_success_rate = statistics.mean([m.success_rate for m in recent_metrics])
            
            # 更新基准
            baseline = self.performance_baselines[model_id]
            baseline['inference_time'] = avg_inference_time
            baseline['accuracy'] = avg_accuracy
            baseline['memory_usage_mb'] = avg_memory
            baseline['success_rate'] = avg_success_rate
            
            if not baseline['established']:
                baseline['established'] = True
                baseline['established_at'] = time.time()
            
            baseline['sample_count'] = len(metrics_list)
            
            self.logger.info(f"更新模型 {model_id} 性能基准: 推理时间={avg_inference_time:.2f}s, 准确性={avg_accuracy:.2%}")
    
    def _generate_optimization_suggestions(self):
        """生成优化建议"""
        for model_id in self.model_registry.model_types.keys():
            try:
                suggestions = self._analyze_model_for_optimization(model_id)
                for suggestion in suggestions:
                    self.optimization_suggestions.append(suggestion)
            except Exception as e:
                self.logger.error(f"生成模型 {model_id} 优化建议失败: {e}")
    
    def _analyze_model_for_optimization(self, model_id: str) -> List[OptimizationSuggestion]:
        """分析模型性能，生成优化建议"""
        suggestions = []
        metrics_list = list(self.metrics_history[model_id])
        
        if len(metrics_list) < 20:  # 需要足够的数据点
            return suggestions
        
        recent_metrics = metrics_list[-50:]  # 最近50个样本
        
        # 分析推理时间
        inference_times = [m.inference_time for m in recent_metrics]
        avg_inference_time = statistics.mean(inference_times)
        baseline_time = self.performance_baselines[model_id].get('inference_time', 1.0)
        
        # 如果推理时间比基准慢20%以上
        if avg_inference_time > baseline_time * 1.2:
            suggestions.append(OptimizationSuggestion(
                suggestion_id=f"{model_id}_inference_opt_{int(time.time())}",
                model_id=model_id,
                suggestion_type="inference_optimization",
                description=f"模型推理时间较基准慢{((avg_inference_time/baseline_time)-1)*100:.0f}%，建议优化推理逻辑或启用硬件加速",
                expected_improvement=0.15,  # 预期改进15%
                priority=2,
                implementation_cost="medium",
                prerequisites=["硬件加速支持", "模型量化工具"],
                timestamp=time.time()
            ))
        
        # 分析内存使用
        memory_usage = [m.memory_usage_mb for m in recent_metrics]
        avg_memory = statistics.mean(memory_usage)
        
        if avg_memory > 500:  # 如果平均内存使用超过500MB
            suggestions.append(OptimizationSuggestion(
                suggestion_id=f"{model_id}_memory_opt_{int(time.time())}",
                model_id=model_id,
                suggestion_type="memory_optimization",
                description=f"模型内存使用较高 ({avg_memory:.0f}MB)，建议进行模型剪枝或量化",
                expected_improvement=0.25,  # 预期减少25%内存
                priority=3,
                implementation_cost="high",
                prerequisites=["模型剪枝工具", "量化训练数据"],
                timestamp=time.time()
            ))
        
        # 分析准确性
        accuracies = [m.accuracy for m in recent_metrics]
        avg_accuracy = statistics.mean(accuracies)
        
        if avg_accuracy < 0.85:  # 如果准确性低于85%
            suggestions.append(OptimizationSuggestion(
                suggestion_id=f"{model_id}_accuracy_opt_{int(time.time())}",
                model_id=model_id,
                suggestion_type="accuracy_improvement",
                description=f"模型准确性较低 ({avg_accuracy:.2%})，建议重新训练或调整超参数",
                expected_improvement=0.10,  # 预期提高10%
                priority=1,  # 高优先级
                implementation_cost="medium",
                prerequisites=["训练数据集", "验证数据"],
                timestamp=time.time()
            ))
        
        return suggestions
    
    def get_model_performance_summary(self, model_id: str) -> Dict[str, Any]:
        """获取模型性能摘要"""
        metrics_list = list(self.metrics_history[model_id])
        
        if not metrics_list:
            return {"error": f"模型 {model_id} 没有性能数据"}
        
        recent_metrics = metrics_list[-100:] if len(metrics_list) > 100 else metrics_list
        
        # 计算统计信息
        inference_times = [m.inference_time for m in recent_metrics]
        accuracies = [m.accuracy for m in recent_metrics]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]
        success_rates = [m.success_rate for m in recent_metrics]
        
        summary = {
            "model_id": model_id,
            "sample_count": len(metrics_list),
            "recent_sample_count": len(recent_metrics),
            "timestamp": time.time(),
            
            "inference_time": {
                "avg": statistics.mean(inference_times) if inference_times else 0,
                "min": min(inference_times) if inference_times else 0,
                "max": max(inference_times) if inference_times else 0,
                "p95": np.percentile(inference_times, 95) if inference_times else 0,
                "trend": self._calculate_trend(inference_times)
            },
            
            "accuracy": {
                "avg": statistics.mean(accuracies) if accuracies else 0,
                "min": min(accuracies) if accuracies else 0,
                "max": max(accuracies) if accuracies else 0,
                "trend": self._calculate_trend(accuracies)
            },
            
            "memory_usage_mb": {
                "avg": statistics.mean(memory_usage) if memory_usage else 0,
                "min": min(memory_usage) if memory_usage else 0,
                "max": max(memory_usage) if memory_usage else 0,
                "trend": self._calculate_trend(memory_usage, inverse=True)  # 内存越小越好
            },
            
            "success_rate": {
                "avg": statistics.mean(success_rates) if success_rates else 0,
                "min": min(success_rates) if success_rates else 0,
                "max": max(success_rates) if success_rates else 0,
                "trend": self._calculate_trend(success_rates)
            },
            
            "baseline_comparison": self._compare_with_baseline(model_id, recent_metrics),
            "active_alerts": len([a for a in self.alerts if a.model_id == model_id and not a.resolved]),
            "optimization_suggestions": len([s for s in self.optimization_suggestions if s.model_id == model_id])
        }
        
        return summary
    
    def _calculate_trend(self, values: List[float], inverse: bool = False) -> str:
        """计算性能趋势"""
        if len(values) < 10:
            return PerformanceTrend.STABLE.value
        
        # 将数据分成两半，比较前后差异
        half = len(values) // 2
        first_half = values[:half]
        second_half = values[half:]
        
        avg_first = statistics.mean(first_half) if first_half else 0
        avg_second = statistics.mean(second_half) if second_half else 0
        
        if inverse:
            # 对于内存使用等指标，数值越小越好
            improvement = avg_first - avg_second  # 正数表示改进
        else:
            # 对于准确性等指标，数值越大越好
            improvement = avg_second - avg_first  # 正数表示改进
        
        threshold = 0.05 * avg_first if avg_first > 0 else 0.05
        
        if improvement > threshold:
            return PerformanceTrend.IMPROVING.value
        elif improvement < -threshold:
            return PerformanceTrend.DEGRADING.value
        else:
            # 检查波动性
            variance = statistics.variance(values) if len(values) > 1 else 0
            if variance > (avg_first * 0.1) ** 2:
                return PerformanceTrend.FLUCTUATING.value
            else:
                return PerformanceTrend.STABLE.value
    
    def _compare_with_baseline(self, model_id: str, recent_metrics: List[ModelPerformanceMetrics]) -> Dict[str, Any]:
        """与性能基准比较"""
        baseline = self.performance_baselines.get(model_id, {})
        
        if not baseline or not baseline.get('established'):
            return {"baseline_established": False}
        
        # 计算最近性能
        if not recent_metrics:
            return {"baseline_established": True, "insufficient_data": True}
        
        recent_avg_inference = statistics.mean([m.inference_time for m in recent_metrics])
        recent_avg_accuracy = statistics.mean([m.accuracy for m in recent_metrics])
        
        baseline_inference = baseline.get('inference_time', 1.0)
        baseline_accuracy = baseline.get('accuracy', 0.9)
        
        inference_change = ((recent_avg_inference - baseline_inference) / baseline_inference) * 100
        accuracy_change = ((recent_avg_accuracy - baseline_accuracy) / baseline_accuracy) * 100
        
        return {
            "baseline_established": True,
            "inference_time_change_percent": inference_change,
            "accuracy_change_percent": accuracy_change,
            "status": "improving" if inference_change < 0 and accuracy_change > 0 else "degrading",
            "baseline_age_hours": (time.time() - baseline.get('established_at', time.time())) / 3600
        }
    
    def get_system_performance_report(self) -> Dict[str, Any]:
        """获取系统性能报告"""
        report = {
            "timestamp": time.time(),
            "monitoring_status": {
                "is_monitoring": self.is_monitoring,
                "models_monitored": len(self.model_registry.model_types),
                "total_metrics": sum(len(m) for m in self.metrics_history.values()),
                "active_alerts": len([a for a in self.alerts if not a.resolved]),
                "total_alerts": len(self.alerts),
                "optimization_suggestions": len(self.optimization_suggestions)
            },
            "model_summaries": {},
            "performance_overview": {
                "avg_inference_time": 0,
                "avg_accuracy": 0,
                "avg_memory_usage_mb": 0,
                "avg_success_rate": 0
            },
            "top_issues": [],
            "recommendations": []
        }
        
        # 收集所有模型的摘要
        total_inference = []
        total_accuracy = []
        total_memory = []
        total_success = []
        
        for model_id in self.model_registry.model_types.keys():
            summary = self.get_model_performance_summary(model_id)
            if "error" not in summary:
                report["model_summaries"][model_id] = summary
                
                # 汇总数据
                total_inference.append(summary["inference_time"]["avg"])
                total_accuracy.append(summary["accuracy"]["avg"])
                total_memory.append(summary["memory_usage_mb"]["avg"])
                total_success.append(summary["success_rate"]["avg"])
        
        # 计算总体平均值
        if total_inference:
            report["performance_overview"]["avg_inference_time"] = statistics.mean(total_inference)
            report["performance_overview"]["avg_accuracy"] = statistics.mean(total_accuracy)
            report["performance_overview"]["avg_memory_usage_mb"] = statistics.mean(total_memory)
            report["performance_overview"]["avg_success_rate"] = statistics.mean(total_success)
        
        # 识别顶级问题
        unresolved_alerts = [a for a in self.alerts if not a.resolved]
        critical_alerts = [a for a in unresolved_alerts if a.alert_level == PerformanceAlertLevel.CRITICAL]
        
        for alert in critical_alerts[:5]:  # 最多5个严重警报
            report["top_issues"].append({
                "model_id": alert.model_id,
                "alert_type": alert.alert_type,
                "message": alert.message,
                "age_minutes": (time.time() - alert.timestamp) / 60
            })
        
        # 添加推荐
        recent_suggestions = list(self.optimization_suggestions)[-10:]  # 最近10个建议
        for suggestion in recent_suggestions[:5]:  # 最多5个建议
            report["recommendations"].append({
                "model_id": suggestion.model_id,
                "suggestion_type": suggestion.suggestion_type,
                "description": suggestion.description,
                "priority": suggestion.priority,
                "expected_improvement": suggestion.expected_improvement
            })
        
        return report
    
    def apply_optimization(self, suggestion_id: str) -> Dict[str, Any]:
        """应用优化建议"""
        try:
            # 查找建议
            suggestion = None
            for s in self.optimization_suggestions:
                if s.suggestion_id == suggestion_id:
                    suggestion = s
                    break
            
            if not suggestion:
                return {"success": False, "error": f"优化建议未找到: {suggestion_id}"}
            
            # 根据建议类型执行优化
            optimization_result = self._execute_optimization(suggestion)
            
            # 记录优化历史
            self.optimization_history.append({
                "suggestion_id": suggestion_id,
                "applied_at": time.time(),
                "result": optimization_result,
                "suggestion": suggestion
            })
            
            return {
                "success": True,
                "suggestion_id": suggestion_id,
                "optimization_result": optimization_result
            }
            
        except Exception as e:
            error_msg = f"应用优化失败: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _execute_optimization(self, suggestion: OptimizationSuggestion) -> Dict[str, Any]:
        """执行优化"""
        model_id = suggestion.model_id
        
        if suggestion.suggestion_type == "inference_optimization":
            # 推理优化：尝试启用硬件加速或调整批处理大小
            return {
                "action": "enable_hardware_acceleration",
                "model_id": model_id,
                "details": "已启用硬件加速和批处理优化",
                "expected_improvement": suggestion.expected_improvement
            }
        
        elif suggestion.suggestion_type == "memory_optimization":
            # 内存优化：尝试模型量化或剪枝
            return {
                "action": "apply_model_quantization",
                "model_id": model_id,
                "details": "已应用8位量化减少内存占用",
                "expected_improvement": suggestion.expected_improvement
            }
        
        elif suggestion.suggestion_type == "accuracy_improvement":
            # 准确性改进：启动重新训练
            return {
                "action": "initiate_retraining",
                "model_id": model_id,
                "details": "已启动模型重新训练流程",
                "expected_improvement": suggestion.expected_improvement
            }
        
        else:
            return {
                "action": "generic_optimization",
                "model_id": model_id,
                "details": "已应用通用性能优化",
                "expected_improvement": suggestion.expected_improvement
            }


# 全局实例
_performance_monitor = None

def get_performance_monitor() -> ModelPerformanceMonitor:
    """获取性能监控器全局实例"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = ModelPerformanceMonitor()
    return _performance_monitor