"""
Monitoring and Observability for CMCP
====================================

Monitoring, logging, and observability components for cross-model communication.
"""

import time
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque

from .message_format import Message


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Metric definition"""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histogram
    quantiles: Optional[List[float]] = None  # For summary


@dataclass
class MetricValue:
    """Metric value with labels"""
    metric_name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class CommunicationMonitor:
    """
    Monitor for cross-model communication.
    
    Tracks metrics, logs, and traces for observability.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # Trace storage
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.spans: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Alert rules
        self.alert_rules: List[Dict[str, Any]] = []
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        
        # Log aggregation
        self.log_buffer: deque = deque(maxlen=10000)
        
        # Performance monitoring
        self.performance_data: Dict[str, List[float]] = defaultdict(list)
        
        # Initialize default metrics
        self._initialize_default_metrics()
        
        # Start background tasks
        self._start_background_tasks()
        
        self.logger.info("CommunicationMonitor initialized")
    
    def _initialize_default_metrics(self):
        """Initialize default metrics"""
        default_metrics = [
            MetricDefinition(
                name="cmcp_messages_total",
                type=MetricType.COUNTER,
                description="Total number of messages processed",
                labels=["source_model", "target_model", "operation", "status"]
            ),
            MetricDefinition(
                name="cmcp_message_latency_seconds",
                type=MetricType.HISTOGRAM,
                description="Message processing latency in seconds",
                labels=["source_model", "target_model", "operation"],
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
            ),
            MetricDefinition(
                name="cmcp_errors_total",
                type=MetricType.COUNTER,
                description="Total number of errors",
                labels=["error_type", "component", "recoverable"]
            ),
            MetricDefinition(
                name="cmcp_concurrent_requests",
                type=MetricType.GAUGE,
                description="Number of concurrent requests"
            ),
            MetricDefinition(
                name="cmcp_queue_size",
                type=MetricType.GAUGE,
                description="Message queue size",
                labels=["queue_name"]
            ),
            MetricDefinition(
                name="cmcp_protocol_usage",
                type=MetricType.COUNTER,
                description="Protocol usage counts",
                labels=["protocol", "direction"]
            ),
            MetricDefinition(
                name="cmcp_circuit_breaker_state",
                type=MetricType.GAUGE,
                description="Circuit breaker state (0=closed, 1=open, 0.5=half_open)",
                labels=["target"]
            )
        ]
        
        for metric in default_metrics:
            self.register_metric(metric)
    
    def register_metric(self, metric_def: MetricDefinition):
        """Register a new metric"""
        self.metric_definitions[metric_def.name] = metric_def
        self.logger.info(f"Registered metric: {metric_def.name}")
    
    def record_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        if metric_name not in self.metric_definitions:
            self.logger.warning(f"Recording unregistered metric: {metric_name}")
            # Auto-create a simple gauge metric
            self.register_metric(MetricDefinition(
                name=metric_name,
                type=MetricType.GAUGE,
                description="Auto-created metric"
            ))
        
        metric_value = MetricValue(
            metric_name=metric_name,
            value=value,
            labels=labels or {},
            timestamp=time.time()
        )
        
        self.metrics[metric_name].append(metric_value)
        
        # Keep only recent metrics (configurable)
        max_metrics = self.config.get('max_metrics_per_type', 1000)
        if len(self.metrics[metric_name]) > max_metrics:
            self.metrics[metric_name] = self.metrics[metric_name][-max_metrics:]
        
        # Check alert rules
        self._check_alerts(metric_name, metric_value)
    
    def increment_counter(self, metric_name: str, increment: float = 1.0,
                         labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        current_value = self._get_latest_metric_value(metric_name, labels or {})
        new_value = current_value + increment
        self.record_metric(metric_name, new_value, labels)
    
    def set_gauge(self, metric_name: str, value: float,
                 labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value"""
        self.record_metric(metric_name, value, labels)
    
    def observe_histogram(self, metric_name: str, value: float,
                         labels: Optional[Dict[str, str]] = None):
        """Observe a histogram metric value"""
        self.record_metric(metric_name, value, labels)
    
    def _get_latest_metric_value(self, metric_name: str, 
                                labels: Dict[str, str]) -> float:
        """Get latest value for a metric with specific labels"""
        if metric_name not in self.metrics:
            return 0.0
        
        # Find most recent matching metric
        for metric in reversed(self.metrics[metric_name]):
            if all(metric.labels.get(k) == v for k, v in labels.items()):
                return metric.value
        
        return 0.0
    
    def record_message_processing(self, message: Message, processing_time: float,
                                 success: bool, error_type: Optional[str] = None):
        """Record message processing metrics"""
        # Record message count
        self.increment_counter("cmcp_messages_total", labels={
            "source_model": message.source_model or "unknown",
            "target_model": message.target_models[0] if message.target_models else "unknown",
            "operation": message.operation or "unknown",
            "status": "success" if success else "error"
        })
        
        # Record latency
        if processing_time > 0:
            self.observe_histogram("cmcp_message_latency_seconds", processing_time, labels={
                "source_model": message.source_model or "unknown",
                "target_model": message.target_models[0] if message.target_models else "unknown",
                "operation": message.operation or "unknown"
            })
        
        # Record errors
        if not success:
            self.increment_counter("cmcp_errors_total", labels={
                "error_type": error_type or "unknown",
                "component": "message_processing",
                "recoverable": "true" if error_type in ["TIMEOUT", "CONNECTION_ERROR"] else "false"
            })
    
    def start_trace(self, trace_id: str, operation: str, 
                   source: Optional[str] = None) -> str:
        """Start a new trace"""
        span_id = f"span_{int(time.time() * 1000)}_{hash(trace_id) % 10000}"
        
        self.traces[trace_id] = {
            "trace_id": trace_id,
            "operation": operation,
            "source": source,
            "start_time": time.time(),
            "spans": [],
            "status": "in_progress"
        }
        
        self.logger.debug(f"Started trace {trace_id} for operation {operation}")
        
        return span_id
    
    def add_span(self, trace_id: str, span_id: str, name: str,
                parent_span_id: Optional[str] = None,
                attributes: Optional[Dict[str, Any]] = None):
        """Add a span to a trace"""
        if trace_id not in self.traces:
            self.logger.warning(f"Trace {trace_id} not found")
            return
        
        span = {
            "span_id": span_id,
            "name": name,
            "parent_span_id": parent_span_id,
            "start_time": time.time(),
            "attributes": attributes or {},
            "events": []
        }
        
        self.traces[trace_id]["spans"].append(span)
        self.spans[trace_id].append(span)
    
    def end_span(self, trace_id: str, span_id: str,
                status: str = "success",
                error: Optional[str] = None):
        """End a span"""
        if trace_id not in self.traces:
            return
        
        for span in self.traces[trace_id]["spans"]:
            if span["span_id"] == span_id:
                span["end_time"] = time.time()
                span["duration"] = span["end_time"] - span["start_time"]
                span["status"] = status
                if error:
                    span["error"] = error
                break
    
    def end_trace(self, trace_id: str, status: str = "success",
                 error: Optional[str] = None):
        """End a trace"""
        if trace_id not in self.traces:
            return
        
        self.traces[trace_id]["end_time"] = time.time()
        self.traces[trace_id]["duration"] = (
            self.traces[trace_id]["end_time"] - self.traces[trace_id]["start_time"]
        )
        self.traces[trace_id]["status"] = status
        
        if error:
            self.traces[trace_id]["error"] = error
        
        # Calculate trace statistics
        spans = self.traces[trace_id]["spans"]
        if spans:
            durations = [s.get("duration", 0) for s in spans if "duration" in s]
            if durations:
                self.traces[trace_id]["statistics"] = {
                    "span_count": len(spans),
                    "total_duration": sum(durations),
                    "avg_duration": sum(durations) / len(durations),
                    "max_duration": max(durations),
                    "min_duration": min(durations)
                }
        
        self.logger.debug(f"Ended trace {trace_id} with status {status}")
    
    def log_message(self, level: str, message: str, context: Optional[Dict[str, Any]] = None):
        """Log a message with context"""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            "context": context or {}
        }
        
        self.log_buffer.append(log_entry)
        
        # Also log to standard logger
        if level == "error":
            self.logger.error(f"{message} - Context: {context}")
        elif level == "warning":
            self.logger.warning(f"{message} - Context: {context}")
        elif level == "info":
            self.logger.info(f"{message} - Context: {context}")
        else:
            self.logger.debug(f"{message} - Context: {context}")
    
    def add_alert_rule(self, rule: Dict[str, Any]):
        """Add an alert rule"""
        self.alert_rules.append(rule)
        self.logger.info(f"Added alert rule: {rule.get('name', 'unnamed')}")
    
    def _check_alerts(self, metric_name: str, metric_value: MetricValue):
        """Check metric against alert rules"""
        for rule in self.alert_rules:
            if rule.get('metric') == metric_name:
                # Check condition
                condition_met = self._evaluate_alert_condition(rule, metric_value)
                
                if condition_met:
                    alert_id = f"{metric_name}_{rule.get('name', 'alert')}"
                    
                    if alert_id not in self.active_alerts:
                        # Trigger alert
                        alert = {
                            "id": alert_id,
                            "rule": rule,
                            "metric_value": metric_value.value,
                            "metric_labels": metric_value.labels,
                            "triggered_at": time.time(),
                            "acknowledged": False
                        }
                        
                        self.active_alerts[alert_id] = alert
                        
                        # Log alert
                        self.log_message("warning", 
                                        f"Alert triggered: {rule.get('name')}",
                                        {"metric": metric_name, 
                                         "value": metric_value.value,
                                         "threshold": rule.get('threshold')})
                        
                        # Call alert handler if defined
                        handler = rule.get('handler')
                        if handler and callable(handler):
                            try:
                                handler(alert)
                            except Exception as e:
                                self.logger.error(f"Alert handler failed: {e}")
    
    def _evaluate_alert_condition(self, rule: Dict[str, Any], 
                                 metric_value: MetricValue) -> bool:
        """Evaluate if alert condition is met"""
        condition = rule.get('condition', '>')
        threshold = rule.get('threshold')
        
        if threshold is None:
            return False
        
        try:
            if condition == '>':
                return metric_value.value > threshold
            elif condition == '>=':
                return metric_value.value >= threshold
            elif condition == '<':
                return metric_value.value < threshold
            elif condition == '<=':
                return metric_value.value <= threshold
            elif condition == '==':
                return metric_value.value == threshold
            elif condition == '!=':
                return metric_value.value != threshold
            else:
                return False
        except (TypeError, ValueError):
            return False
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]['acknowledged'] = True
            self.active_alerts[alert_id]['acknowledged_at'] = time.time()
            self.logger.info(f"Acknowledged alert: {alert_id}")
    
    def clear_alert(self, alert_id: str):
        """Clear an alert"""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            self.logger.info(f"Cleared alert: {alert_id}")
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        # Start metrics aggregation task
        asyncio.create_task(self._aggregate_metrics())
        
        # Start alert cleanup task
        asyncio.create_task(self._cleanup_old_alerts())
        
        # Start trace cleanup task
        asyncio.create_task(self._cleanup_old_traces())
    
    async def _aggregate_metrics(self):
        """Aggregate metrics periodically"""
        while True:
            try:
                await self._calculate_performance_metrics()
                await asyncio.sleep(60)  # Every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics aggregation error: {e}")
                await asyncio.sleep(60)
    
    async def _calculate_performance_metrics(self):
        """Calculate performance metrics from raw data"""
        # Calculate message rates
        message_metrics = self.metrics.get("cmcp_messages_total", [])
        if message_metrics:
            # Group by time window (last minute)
            current_time = time.time()
            recent_metrics = [
                m for m in message_metrics 
                if current_time - m.timestamp <= 60
            ]
            
            # Calculate rates
            success_count = sum(1 for m in recent_metrics 
                              if m.labels.get('status') == 'success')
            error_count = sum(1 for m in recent_metrics 
                            if m.labels.get('status') == 'error')
            total_count = len(recent_metrics)
            
            if total_count > 0:
                success_rate = success_count / total_count * 100
                self.set_gauge("cmcp_success_rate_percent", success_rate)
                self.set_gauge("cmcp_message_rate_per_minute", total_count)
    
    async def _cleanup_old_alerts(self):
        """Clean up old alerts"""
        while True:
            try:
                current_time = time.time()
                alert_retention = self.config.get('alert_retention_hours', 24) * 3600
                
                to_remove = []
                for alert_id, alert in self.active_alerts.items():
                    if current_time - alert['triggered_at'] > alert_retention:
                        to_remove.append(alert_id)
                
                for alert_id in to_remove:
                    del self.active_alerts[alert_id]
                
                if to_remove:
                    self.logger.debug(f"Cleaned up {len(to_remove)} old alerts")
                
                await asyncio.sleep(3600)  # Every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Alert cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_old_traces(self):
        """Clean up old traces"""
        while True:
            try:
                current_time = time.time()
                trace_retention = self.config.get('trace_retention_hours', 24) * 3600
                
                to_remove = []
                for trace_id, trace in self.traces.items():
                    end_time = trace.get('end_time', trace.get('start_time', 0))
                    if current_time - end_time > trace_retention:
                        to_remove.append(trace_id)
                
                for trace_id in to_remove:
                    del self.traces[trace_id]
                    if trace_id in self.spans:
                        del self.spans[trace_id]
                
                if to_remove:
                    self.logger.debug(f"Cleaned up {len(to_remove)} old traces")
                
                await asyncio.sleep(3600)  # Every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Trace cleanup error: {e}")
                await asyncio.sleep(3600)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        summary = {
            "metrics": {},
            "alerts": {
                "active": len(self.active_alerts),
                "rules": len(self.alert_rules)
            },
            "traces": {
                "active": len([t for t in self.traces.values() 
                             if t.get('status') == 'in_progress']),
                "completed": len([t for t in self.traces.values() 
                                if t.get('status') != 'in_progress']),
                "total": len(self.traces)
            },
            "logs": {
                "buffer_size": len(self.log_buffer)
            }
        }
        
        # Aggregate metrics by type
        for metric_name, values in self.metrics.items():
            if values:
                recent_values = [v for v in values if time.time() - v.timestamp <= 300]  # Last 5 minutes
                if recent_values:
                    numeric_values = [v.value for v in recent_values]
                    summary["metrics"][metric_name] = {
                        "count": len(numeric_values),
                        "latest": numeric_values[-1] if numeric_values else 0,
                        "average": sum(numeric_values) / len(numeric_values) if numeric_values else 0,
                        "min": min(numeric_values) if numeric_values else 0,
                        "max": max(numeric_values) if numeric_values else 0
                    }
        
        return summary
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        if format == "json":
            return json.dumps(self.get_metrics_summary(), indent=2)
        elif format == "prometheus":
            return self._export_prometheus_metrics()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        for metric_name, values in self.metrics.items():
            metric_def = self.metric_definitions.get(metric_name)
            if not metric_def:
                continue
            
            # Get latest value for each label combination
            latest_by_labels = {}
            for value in values:
                label_key = json.dumps(value.labels, sort_keys=True)
                if label_key not in latest_by_labels or value.timestamp > latest_by_labels[label_key]['timestamp']:
                    latest_by_labels[label_key] = {
                        'value': value.value,
                        'labels': value.labels,
                        'timestamp': value.timestamp
                    }
            
            # Format each value
            for data in latest_by_labels.values():
                labels_str = ""
                if data['labels']:
                    label_parts = [f'{k}="{v}"' for k, v in data['labels'].items()]
                    labels_str = "{" + ",".join(label_parts) + "}"
                
                lines.append(f"{metric_name}{labels_str} {data['value']}")
        
        return "\n".join(lines)


# Global monitor instance
_global_monitor: Optional[CommunicationMonitor] = None

def get_monitor(config: Optional[Dict[str, Any]] = None) -> CommunicationMonitor:
    """Get global communication monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = CommunicationMonitor(config)
    return _global_monitor

def record_message_event(message: Message, event_type: str, 
                        details: Optional[Dict[str, Any]] = None):
    """Helper to record message events"""
    monitor = get_monitor()
    monitor.log_message("info", f"Message {event_type}", {
        "message_id": message.message_id,
        "operation": message.operation,
        "source": message.source_model,
        "targets": message.target_models,
        "details": details or {}
    })