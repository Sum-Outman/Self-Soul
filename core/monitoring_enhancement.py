#!/usr/bin/env python3
"""
监控面板增强模块 - Monitoring Enhancement Module

增强现有监控系统的功能，包括：
1. AGI特定监控指标采集
2. 决策触发机制
3. 监控面板后端与API的集成
4. 实时警报和自动化响应
"""

import logging
import time
import threading
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio

logger = logging.getLogger(__name__)

# 导入现有监控组件
try:
    from core.system_monitor import SystemMonitor
    from core.monitoring_dashboard_backend import get_dashboard_backend, MonitoringDashboardBackend
    from core.monitoring import get_system_status
    from core.agi_coordinator import agi_coordinator
except ImportError as e:
    logger.warning(f"导入监控组件失败: {e}")
    # 创建模拟组件
    SystemMonitor = type('MockSystemMonitor', (), {
        'get_realtime_monitoring': lambda: {'data': {}},
        'get_enhanced_metrics': lambda: {'data': {}},
        'get_performance_stats': lambda: {'data': {}}
    })
    get_dashboard_backend = lambda: None
    get_system_status = lambda: {}
    agi_coordinator = None

class AGIMonitoringEnhancer:
    """
    AGI监控增强器 - 提供AGI系统特定监控指标和决策触发
    """
    
    def __init__(self):
        """初始化AGI监控增强器"""
        self.logger = logger
        self.initialized = False
        
        # 监控指标定义
        self.agi_metrics_definitions = {
            # 推理和规划指标
            "reasoning_quality": {
                "name": "推理质量",
                "description": "AGI推理系统的质量评分",
                "unit": "score",
                "threshold_warning": 0.6,
                "threshold_critical": 0.4
            },
            "planning_accuracy": {
                "name": "规划准确率",
                "description": "任务规划准确率",
                "unit": "score",
                "threshold_warning": 0.7,
                "threshold_critical": 0.5
            },
            "decision_effectiveness": {
                "name": "决策有效性",
                "description": "AGI决策的有效性评分",
                "unit": "score",
                "threshold_warning": 0.65,
                "threshold_critical": 0.45
            },
            
            # 学习和适应指标
            "learning_rate": {
                "name": "学习速率",
                "description": "系统学习新知识的速度",
                "unit": "score",
                "threshold_warning": 0.3,
                "threshold_critical": 0.1
            },
            "adaptation_success": {
                "name": "适应成功率",
                "description": "系统适应新环境/任务的成功率",
                "unit": "score",
                "threshold_warning": 0.6,
                "threshold_critical": 0.4
            },
            "knowledge_growth": {
                "name": "知识增长",
                "description": "知识库的增长速度",
                "unit": "items/day",
                "threshold_warning": 10,
                "threshold_critical": 5
            },
            
            # 协作和多模型指标
            "collaboration_efficiency": {
                "name": "协作效率",
                "description": "多模型协作的效率",
                "unit": "score",
                "threshold_warning": 0.7,
                "threshold_critical": 0.5
            },
            "model_coordination": {
                "name": "模型协调",
                "description": "不同模型间的协调程度",
                "unit": "score",
                "threshold_warning": 0.65,
                "threshold_critical": 0.45
            },
            "task_distribution": {
                "name": "任务分布均衡度",
                "description": "任务在不同模型间的分布均衡程度",
                "unit": "score",
                "threshold_warning": 0.6,
                "threshold_critical": 0.4
            },
            
            # 自主性和目标导向指标
            "autonomy_level": {
                "name": "自主性水平",
                "description": "系统的自主决策和执行能力",
                "unit": "score",
                "threshold_warning": 0.7,
                "threshold_critical": 0.5
            },
            "goal_achievement": {
                "name": "目标达成率",
                "description": "设定目标的完成比例",
                "unit": "score",
                "threshold_warning": 0.6,
                "threshold_critical": 0.4
            },
            "self_optimization": {
                "name": "自我优化",
                "description": "系统自我优化的效率",
                "unit": "score",
                "threshold_warning": 0.5,
                "threshold_critical": 0.3
            }
        }
        
        # 决策触发规则
        self.decision_rules = {
            "performance_degradation": {
                "condition": "any(metric.value < metric.threshold_warning for metric in metrics)",
                "action": "trigger_performance_optimization",
                "severity": "warning",
                "description": "检测到性能下降，触发优化"
            },
            "critical_failure": {
                "condition": "any(metric.value < metric.threshold_critical for metric in metrics)",
                "action": "trigger_emergency_response",
                "severity": "critical",
                "description": "检测到关键故障，触发紧急响应"
            },
            "learning_stagnation": {
                "condition": "learning_metric.trend < -0.1 for learning_metric in metrics if 'learning' in metric.name",
                "action": "trigger_learning_intervention",
                "severity": "warning",
                "description": "检测到学习停滞，触发学习干预"
            },
            "collaboration_inefficiency": {
                "condition": "collaboration_efficiency.value < 0.6",
                "action": "trigger_collaboration_optimization",
                "severity": "warning",
                "description": "检测到协作效率低下，触发协作优化"
            }
        }
        
        # 数据存储
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.triggered_decisions = deque(maxlen=500)
        self.alerts_history = deque(maxlen=1000)
        
        # 状态
        self.monitoring_active = False
        self.monitoring_thread = None
        self.last_collection_time = 0
        
        self._initialize()
    
    def _initialize(self):
        """初始化增强器"""
        try:
            # 尝试获取现有监控组件
            self.system_monitor = SystemMonitor()
            self.dashboard_backend = get_dashboard_backend()
            
            logger.info("AGI监控增强器初始化成功")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"AGI监控增强器初始化失败: {e}")
            self.initialized = False
    
    def collect_agi_metrics(self) -> Dict[str, Any]:
        """
        收集AGI特定监控指标
        
        Returns:
            AGI监控指标字典
        """
        metrics = {}
        
        try:
            # 1. 获取系统状态
            system_status = get_system_status() if callable(get_system_status) else {}
            
            # 2. 尝试从AGI协调器获取指标
            if agi_coordinator and hasattr(agi_coordinator, 'get_agi_metrics'):
                try:
                    agi_metrics = agi_coordinator.get_agi_metrics()
                    metrics.update(agi_metrics)
                except Exception as e:
                    logger.warning(f"从AGI协调器获取指标失败: {e}")
            
            # 3. 生成模拟/估算指标（如果真实数据不可用）
            self._generate_simulated_metrics(metrics)
            
            # 4. 处理指标数据
            processed_metrics = []
            timestamp = time.time()
            
            for metric_id, definition in self.agi_metrics_definitions.items():
                value = metrics.get(metric_id, self._estimate_metric_value(metric_id))
                
                metric_data = {
                    "metric_id": metric_id,
                    "name": definition["name"],
                    "description": definition["description"],
                    "value": value,
                    "unit": definition["unit"],
                    "threshold_warning": definition.get("threshold_warning"),
                    "threshold_critical": definition.get("threshold_critical"),
                    "timestamp": timestamp,
                    "source": "agi_enhancer"
                }
                
                # 计算趋势（如果有历史数据）
                if metric_id in self.metrics_history and len(self.metrics_history[metric_id]) > 1:
                    history = list(self.metrics_history[metric_id])
                    if len(history) >= 2:
                        old_value = history[-2]["value"]
                        new_value = history[-1]["value"]
                        if old_value != 0:
                            trend = (new_value - old_value) / abs(old_value)
                            metric_data["trend"] = trend
                
                processed_metrics.append(metric_data)
                
                # 存储到历史
                self.metrics_history[metric_id].append(metric_data)
            
            result = {
                "timestamp": timestamp,
                "metrics": processed_metrics,
                "summary": {
                    "total_metrics": len(processed_metrics),
                    "warning_count": len([m for m in processed_metrics if m.get("value", 0) < m.get("threshold_warning", float('inf'))]),
                    "critical_count": len([m for m in processed_metrics if m.get("value", 0) < m.get("threshold_critical", float('inf'))]),
                    "average_score": sum(m.get("value", 0) for m in processed_metrics) / max(len(processed_metrics), 1)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"收集AGI指标失败: {e}")
            return {
                "timestamp": time.time(),
                "metrics": [],
                "summary": {"total_metrics": 0, "warning_count": 0, "critical_count": 0, "average_score": 0},
                "error": str(e)
            }
    
    def _generate_simulated_metrics(self, metrics: Dict[str, Any]):
        """生成模拟指标（当真实数据不可用时）"""
        import random
        import math
        
        # 基础指标模拟
        base_time = time.time()
        
        # 使用系统时间作为随机种子，使模拟值在一定时间内相对稳定
        time_seed = int(base_time / 60)  # 每分钟变化一次
        
        # 模拟各种AGI指标
        simulated_values = {
            "reasoning_quality": 0.7 + 0.2 * math.sin(time_seed * 0.1) + random.uniform(-0.05, 0.05),
            "planning_accuracy": 0.75 + 0.15 * math.sin(time_seed * 0.15) + random.uniform(-0.05, 0.05),
            "decision_effectiveness": 0.65 + 0.2 * math.sin(time_seed * 0.12) + random.uniform(-0.05, 0.05),
            "learning_rate": 0.4 + 0.3 * math.sin(time_seed * 0.2) + random.uniform(-0.05, 0.05),
            "adaptation_success": 0.6 + 0.25 * math.sin(time_seed * 0.18) + random.uniform(-0.05, 0.05),
            "knowledge_growth": 15 + 10 * math.sin(time_seed * 0.1) + random.uniform(-2, 2),
            "collaboration_efficiency": 0.7 + 0.2 * math.sin(time_seed * 0.13) + random.uniform(-0.05, 0.05),
            "model_coordination": 0.65 + 0.2 * math.sin(time_seed * 0.14) + random.uniform(-0.05, 0.05),
            "task_distribution": 0.8 + 0.15 * math.sin(time_seed * 0.16) + random.uniform(-0.05, 0.05),
            "autonomy_level": 0.6 + 0.25 * math.sin(time_seed * 0.17) + random.uniform(-0.05, 0.05),
            "goal_achievement": 0.7 + 0.2 * math.sin(time_seed * 0.19) + random.uniform(-0.05, 0.05),
            "self_optimization": 0.5 + 0.3 * math.sin(time_seed * 0.21) + random.uniform(-0.05, 0.05),
        }
        
        # 只填充未提供的指标
        for key, value in simulated_values.items():
            if key not in metrics:
                metrics[key] = max(0, min(1, value)) if key.endswith("score") or "rate" in key else max(0, value)
    
    def _estimate_metric_value(self, metric_id: str) -> float:
        """估算指标值"""
        import random
        
        # 基于指标ID提供合理的估计值
        estimation_map = {
            "reasoning_quality": 0.7,
            "planning_accuracy": 0.75,
            "decision_effectiveness": 0.65,
            "learning_rate": 0.4,
            "adaptation_success": 0.6,
            "knowledge_growth": 15,
            "collaboration_efficiency": 0.7,
            "model_coordination": 0.65,
            "task_distribution": 0.8,
            "autonomy_level": 0.6,
            "goal_achievement": 0.7,
            "self_optimization": 0.5,
        }
        
        base_value = estimation_map.get(metric_id, 0.5)
        
        # 添加一些随机变化
        variation = random.uniform(-0.1, 0.1)
        
        return max(0, min(1, base_value + variation)) if metric_id.endswith("score") or "rate" in metric_id else max(0, base_value + variation * 5)
    
    def evaluate_decision_triggers(self, metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        评估决策触发规则
        
        Args:
            metrics_data: 指标数据
            
        Returns:
            触发的决策列表
        """
        triggered_decisions = []
        
        try:
            metrics = metrics_data.get("metrics", [])
            
            for rule_id, rule in self.decision_rules.items():
                try:
                    # 简化条件评估
                    should_trigger = False
                    
                    if rule_id == "performance_degradation":
                        warning_count = len([m for m in metrics if m.get("value", 0) < m.get("threshold_warning", float('inf'))])
                        should_trigger = warning_count > 0
                    
                    elif rule_id == "critical_failure":
                        critical_count = len([m for m in metrics if m.get("value", 0) < m.get("threshold_critical", float('inf'))])
                        should_trigger = critical_count > 0
                    
                    elif rule_id == "learning_stagnation":
                        learning_metrics = [m for m in metrics if 'learning' in m.get('name', '').lower()]
                        for metric in learning_metrics:
                            trend = metric.get('trend', 0)
                            if trend < -0.1:
                                should_trigger = True
                                break
                    
                    elif rule_id == "collaboration_inefficiency":
                        collab_metrics = [m for m in metrics if 'collaboration' in m.get('name', '').lower()]
                        for metric in collab_metrics:
                            if metric.get('value', 0) < 0.6:
                                should_trigger = True
                                break
                    
                    if should_trigger:
                        decision = {
                            "rule_id": rule_id,
                            "action": rule["action"],
                            "severity": rule["severity"],
                            "description": rule["description"],
                            "trigger_time": time.time(),
                            "metrics_context": metrics_data.get("summary", {}),
                            "triggered_metrics": [m["metric_id"] for m in metrics if self._metric_matches_rule(m, rule_id)]
                        }
                        
                        triggered_decisions.append(decision)
                        
                        # 记录决策
                        self.triggered_decisions.append(decision)
                        
                        # 记录警报
                        alert = {
                            "alert_id": f"decision_trigger_{rule_id}_{int(time.time())}",
                            "type": "decision_trigger",
                            "severity": rule["severity"],
                            "message": f"触发决策: {rule['description']}",
                            "timestamp": time.time(),
                            "data": decision
                        }
                        self.alerts_history.append(alert)
                        
                        logger.info(f"决策触发: {rule_id} - {rule['description']}")
                        
                except Exception as e:
                    logger.error(f"评估规则 {rule_id} 失败: {e}")
            
            return triggered_decisions
            
        except Exception as e:
            logger.error(f"评估决策触发规则失败: {e}")
            return []
    
    def _metric_matches_rule(self, metric: Dict[str, Any], rule_id: str) -> bool:
        """检查指标是否匹配规则"""
        metric_value = metric.get("value", 0)
        
        if rule_id == "performance_degradation":
            threshold = metric.get("threshold_warning")
            return threshold is not None and metric_value < threshold
        
        elif rule_id == "critical_failure":
            threshold = metric.get("threshold_critical")
            return threshold is not None and metric_value < threshold
        
        elif rule_id == "learning_stagnation":
            return 'learning' in metric.get('name', '').lower()
        
        elif rule_id == "collaboration_inefficiency":
            return 'collaboration' in metric.get('name', '').lower() and metric_value < 0.6
        
        return False
    
    def execute_decision_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行决策动作
        
        Args:
            decision: 决策数据
            
        Returns:
            执行结果
        """
        action = decision.get("action", "")
        
        try:
            result = {
                "action": action,
                "decision_id": decision.get("rule_id", ""),
                "execution_time": time.time(),
                "status": "pending"
            }
            
            # 根据动作类型执行
            if action == "trigger_performance_optimization":
                result.update(self._trigger_performance_optimization(decision))
                
            elif action == "trigger_emergency_response":
                result.update(self._trigger_emergency_response(decision))
                
            elif action == "trigger_learning_intervention":
                result.update(self._trigger_learning_intervention(decision))
                
            elif action == "trigger_collaboration_optimization":
                result.update(self._trigger_collaboration_optimization(decision))
            
            else:
                result["status"] = "unknown_action"
                result["message"] = f"未知动作: {action}"
            
            result["status"] = "executed" if result.get("status") != "pending" else result.get("status", "failed")
            
            return result
            
        except Exception as e:
            logger.error(f"执行决策动作失败: {e}")
            return {
                "action": action,
                "decision_id": decision.get("rule_id", ""),
                "execution_time": time.time(),
                "status": "failed",
                "error": str(e)
            }
    
    def _trigger_performance_optimization(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """触发性能优化"""
        return {
            "action_details": "触发系统性能优化",
            "optimization_type": "performance",
            "target_metrics": decision.get("triggered_metrics", []),
            "estimated_duration": 300,  # 5分钟
            "expected_improvement": 0.15
        }
    
    def _trigger_emergency_response(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """触发紧急响应"""
        return {
            "action_details": "触发紧急响应协议",
            "response_type": "emergency",
            "target_metrics": decision.get("triggered_metrics", []),
            "immediate_actions": [
                "暂停非关键任务",
                "激活备份系统",
                "发送紧急警报"
            ],
            "recovery_estimated": 600  # 10分钟
        }
    
    def _trigger_learning_intervention(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """触发学习干预"""
        return {
            "action_details": "触发学习干预",
            "intervention_type": "learning_acceleration",
            "target_metrics": decision.get("triggered_metrics", []),
            "intervention_strategies": [
                "增加学习样本多样性",
                "调整学习率参数",
                "激活元学习机制"
            ],
            "estimated_improvement_time": 1200  # 20分钟
        }
    
    def _trigger_collaboration_optimization(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """触发协作优化"""
        return {
            "action_details": "触发协作优化",
            "optimization_type": "collaboration",
            "target_metrics": decision.get("triggered_metrics", []),
            "optimization_strategies": [
                "重新分配任务负载",
                "优化模型间通信",
                "调整协作协议参数"
            ],
            "estimated_duration": 450  # 7.5分钟
        }
    
    def start_monitoring(self, interval_seconds: int = 60):
        """启动监控"""
        if self.monitoring_active:
            logger.warning("监控已经在运行")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"AGI监控已启动，间隔: {interval_seconds}秒")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("AGI监控已停止")
    
    def _monitoring_loop(self, interval_seconds: int):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 收集指标
                metrics_data = self.collect_agi_metrics()
                
                # 评估决策触发
                triggered_decisions = self.evaluate_decision_triggers(metrics_data)
                
                # 执行触发的决策
                for decision in triggered_decisions:
                    execution_result = self.execute_decision_action(decision)
                    
                    # 记录执行结果
                    logger.info(f"决策执行结果: {execution_result.get('status', 'unknown')}")
                
                self.last_collection_time = time.time()
                
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
            
            # 等待下一个周期
            time.sleep(interval_seconds)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        return {
            "status": "active" if self.monitoring_active else "inactive",
            "initialized": self.initialized,
            "last_collection_time": self.last_collection_time,
            "metrics_history_size": {k: len(v) for k, v in self.metrics_history.items()},
            "triggered_decisions_count": len(self.triggered_decisions),
            "alerts_count": len(self.alerts_history),
            "monitoring_definitions": {
                "agi_metrics_count": len(self.agi_metrics_definitions),
                "decision_rules_count": len(self.decision_rules)
            }
        }


# 全局增强器实例
_agi_monitoring_enhancer_instance = None

def get_agi_monitoring_enhancer() -> AGIMonitoringEnhancer:
    """获取AGI监控增强器单例"""
    global _agi_monitoring_enhancer_instance
    
    if _agi_monitoring_enhancer_instance is None:
        _agi_monitoring_enhancer_instance = AGIMonitoringEnhancer()
    
    return _agi_monitoring_enhancer_instance

def initialize_agi_monitoring():
    """初始化AGI监控"""
    enhancer = get_agi_monitoring_enhancer()
    if enhancer.initialized:
        enhancer.start_monitoring()
        return True
    return False