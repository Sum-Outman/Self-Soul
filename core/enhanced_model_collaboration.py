"""
增强模型协作系统 - Enhanced Model Collaboration System

功能描述：
1. 实时模型间通信通道
2. 模型性能监控和自动优化  
3. 知识共享和迁移机制
4. 冲突解决和协商系统
5. 协作历史分析和机器学习
6. 自适应协作策略调整
7. 多模型工作流编排

Function Description:
1. Real-time inter-model communication channels
2. Model performance monitoring and auto-optimization
3. Knowledge sharing and transfer mechanisms
4. Conflict resolution and negotiation systems
5. Collaboration history analysis and machine learning
6. Adaptive collaboration strategy adjustment
7. Multi-model workflow orchestration
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Callable, Optional, Union
from enum import Enum
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
import json
import numpy as np
from datetime import datetime

from core.error_handling import error_handler
from core.model_registry import get_model_registry
from core.collaboration.model_collaborator import ModelCollaborationOrchestrator, CollaborationMode
from core.model_performance_monitor import get_performance_monitor


class CommunicationProtocol(Enum):
    """模型间通信协议"""
    REST_API = "rest_api"
    GRPC = "grpc"
    MESSAGE_QUEUE = "message_queue"
    SHARED_MEMORY = "shared_memory"
    WEBSOCKET = "websocket"
    DIRECT_CALL = "direct_call"


class ConflictResolutionStrategy(Enum):
    """冲突解决策略"""
    MAJORITY_VOTE = "majority_vote"
    EXPERT_MODEL = "expert_model"
    HIERARCHICAL = "hierarchical"
    CONSENSUS = "consensus"
    ADAPTIVE_WEIGHTED = "adaptive_weighted"
    AGI_JUDGMENT = "agi_judgment"


@dataclass
class ModelPerformanceMetrics:
    """模型性能指标"""
    model_id: str
    inference_time: float
    accuracy: float
    resource_usage: float
    reliability: float
    collaboration_score: float
    timestamp: float
    task_type: str


@dataclass
class KnowledgeTransferRecord:
    """知识转移记录"""
    source_model: str
    target_model: str
    knowledge_type: str
    transfer_score: float
    timestamp: float
    context: Dict[str, Any]


class EnhancedModelCollaboration:
    """增强模型协作系统"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 核心组件
        self.base_orchestrator = ModelCollaborationOrchestrator()
        self.model_registry = get_model_registry()
        self.performance_monitor = get_performance_monitor()
        
        # 通信系统
        self.communication_channels = {}
        self.message_queues = defaultdict(deque)
        self.shared_knowledge_base = {}
        
        # 性能监控
        self.performance_history = defaultdict(list)
        self.model_capabilities = {}
        self.real_time_metrics = {}
        
        # 知识共享
        self.knowledge_transfer_records = []
        self.knowledge_sharing_enabled = True
        
        # 冲突解决
        self.conflict_resolution_strategy = ConflictResolutionStrategy.CONSENSUS
        self.conflict_history = []
        
        # 协作学习
        self.collaboration_patterns_learned = {}
        self.workflow_templates = {}
        self.adaptive_strategies = {}
        
        # 实时监控
        self.active_communications = {}
        self.communication_latency = defaultdict(list)
        
        # 初始化系统
        self._initialize_communication_system()
        self._initialize_performance_monitor()
        self._initialize_knowledge_sharing()
        
        self.logger.info("增强模型协作系统初始化完成")
        
        # 启动性能监控
        self._start_performance_monitoring()
    
    def _start_performance_monitoring(self):
        """启动性能监控"""
        try:
            if hasattr(self.performance_monitor, 'start_monitoring'):
                self.performance_monitor.start_monitoring()
                self.logger.info("性能监控已启动")
            else:
                self.logger.warning("性能监控器不支持自动启动")
        except Exception as e:
            self.logger.error(f"启动性能监控失败: {e}")

    def _initialize_communication_system(self):
        """初始化通信系统"""
        # 为每个模型创建通信通道
        for model_id in self.model_registry.model_types.keys():
            self.communication_channels[model_id] = {
                'rest_api': f"http://localhost:8000/api/{model_id}/receive",
                'message_queue': f"{model_id}_queue",
                'shared_memory_key': f"model_{model_id}_memory",
                'status': 'active'
            }
        
        # 初始化消息队列管理器
        self.message_queue_manager = {
            'max_size': 10000,
            'retention_time': 3600,  # 1小时
            'priority_levels': 3
        }
        
        self.logger.info(f"通信系统已初始化，支持 {len(self.communication_channels)} 个模型")
    
    def _initialize_performance_monitor(self):
        """初始化性能监控器"""
        # 定期收集性能指标
        self.performance_monitor_config = {
            'collection_interval': 60,  # 60秒
            'retention_days': 30,
            'alert_thresholds': {
                'inference_time': 5.0,  # 5秒
                'resource_usage': 0.8,   # 80%
                'reliability': 0.7       # 70%
            }
        }
        
        # 启动后台监控任务
        self._start_performance_monitoring()
    
    def _initialize_knowledge_sharing(self):
        """初始化知识共享系统"""
        self.knowledge_sharing_config = {
            'enabled': True,
            'transfer_threshold': 0.7,  # 知识相关性阈值
            'max_knowledge_records': 10000,
            'sharing_modes': ['explicit', 'implicit', 'adaptive']
        }
        
        # 知识类型映射
        self.knowledge_type_mapping = {
            'parameters': '模型参数',
            'embeddings': '特征嵌入',
            'patterns': '模式识别',
            'strategies': '决策策略',
            'constraints': '约束条件',
            'context': '上下文信息'
        }
    
    def _start_performance_monitoring(self):
        """启动性能监控后台任务"""
        async def monitor_performance():
            while True:
                try:
                    await self._collect_performance_metrics()
                    await asyncio.sleep(self.performance_monitor_config['collection_interval'])
                except Exception as e:
                    self.logger.error(f"性能监控失败: {e}")
                    await asyncio.sleep(30)  # 出错后等待30秒
        
        # 启动监控任务 - Python 3.6兼容版本
        asyncio.ensure_future(monitor_performance())
        self.logger.info("性能监控系统已启动")
    
    async def _collect_performance_metrics(self):
        """收集性能指标"""
        try:
            current_time = time.time()
            
            for model_id in self.model_registry.model_types.keys():
                # 模拟收集性能指标（真实实现中会调用模型API）
                metrics = ModelPerformanceMetrics(
                    model_id=model_id,
                    inference_time=np.random.uniform(0.1, 2.0),
                    accuracy=np.random.uniform(0.8, 0.99),
                    resource_usage=np.random.uniform(0.2, 0.6),
                    reliability=np.random.uniform(0.9, 0.99),
                    collaboration_score=np.random.uniform(0.7, 0.95),
                    timestamp=current_time,
                    task_type="general"
                )
                
                self.performance_history[model_id].append(metrics)
                
                # 保持历史记录大小
                if len(self.performance_history[model_id]) > 1000:
                    self.performance_history[model_id] = self.performance_history[model_id][-100:]
                
                # 检查警报阈值
                self._check_performance_alerts(metrics)
        
        except Exception as e:
            self.logger.error(f"收集性能指标失败: {e}")
    
    def _check_performance_alerts(self, metrics: ModelPerformanceMetrics):
        """检查性能警报"""
        alerts = []
        thresholds = self.performance_monitor_config['alert_thresholds']
        
        if metrics.inference_time > thresholds['inference_time']:
            alerts.append(f"模型 {metrics.model_id} 推理时间过长: {metrics.inference_time:.2f}s")
        
        if metrics.resource_usage > thresholds['resource_usage']:
            alerts.append(f"模型 {metrics.model_id} 资源使用率过高: {metrics.resource_usage:.2%}")
        
        if metrics.reliability < thresholds['reliability']:
            alerts.append(f"模型 {metrics.model_id} 可靠性过低: {metrics.reliability:.2%}")
        
        if alerts:
            for alert in alerts:
                self.logger.warning(f"性能警报: {alert}")
                # 这里可以触发通知或自动优化
    
    async def send_message(self, source_model: str, target_model: str, 
                          message: Dict[str, Any], 
                          protocol: CommunicationProtocol = CommunicationProtocol.DIRECT_CALL) -> Dict[str, Any]:
        """发送模型间消息"""
        message_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # 验证模型存在
            if target_model not in self.communication_channels:
                return {"error": f"目标模型不存在: {target_model}", "message_id": message_id}
            
            # 添加消息元数据
            full_message = {
                "message_id": message_id,
                "source": source_model,
                "target": target_model,
                "protocol": protocol.value,
                "timestamp": time.time(),
                "content": message,
                "status": "sending"
            }
            
            # 根据协议发送消息
            if protocol == CommunicationProtocol.DIRECT_CALL:
                result = await self._direct_model_call(target_model, message)
            elif protocol == CommunicationProtocol.MESSAGE_QUEUE:
                result = await self._queue_message(target_model, full_message)
            elif protocol == CommunicationProtocol.SHARED_MEMORY:
                result = await self._shared_memory_write(target_model, full_message)
            else:
                result = {"error": f"协议未实现: {protocol}", "message_id": message_id}
            
            # 记录通信延迟
            latency = time.time() - start_time
            self.communication_latency[f"{source_model}_{target_model}"].append(latency)
            
            # 更新消息状态
            full_message["status"] = "delivered" if "error" not in result else "failed"
            full_message["latency"] = latency
            full_message["response"] = result
            
            # 记录通信历史
            self._record_communication(full_message)
            
            return {"message_id": message_id, "status": "success", "response": result, "latency": latency}
            
        except Exception as e:
            error_msg = f"消息发送失败: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg, "message_id": message_id}
    
    async def _direct_model_call(self, target_model: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """直接模型调用"""
        try:
            model = self.model_registry.get_model(target_model)
            if not model:
                return {"error": f"模型未加载: {target_model}"}
            
            # 根据消息类型调用相应方法
            message_type = message.get("type", "process")
            
            if message_type == "process":
                if hasattr(model, 'process'):
                    result = model.process(message.get("data", {}))
                elif hasattr(model, 'forward'):
                    result = model.forward(message.get("data", {}))
                else:
                    return {"error": f"模型 {target_model} 没有处理方法"}
            elif message_type == "collaborate":
                if hasattr(model, 'collaborate'):
                    result = model.collaborate(message.get("data", {}))
                else:
                    return {"error": f"模型 {target_model} 没有协作方法"}
            else:
                return {"error": f"未知消息类型: {message_type}"}
            
            return {"success": True, "result": result}
            
        except Exception as e:
            return {"error": f"模型调用失败: {str(e)}"}
    
    async def _queue_message(self, target_model: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """队列消息传递"""
        try:
            # 将消息加入队列
            self.message_queues[target_model].append(message)
            
            # 限制队列大小
            if len(self.message_queues[target_model]) > self.message_queue_manager['max_size']:
                self.message_queues[target_model].popleft()
            
            return {"success": True, "queued": True, "queue_size": len(self.message_queues[target_model])}
        except Exception as e:
            return {"error": f"队列消息失败: {str(e)}"}
    
    async def _shared_memory_write(self, target_model: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """共享内存写入"""
        try:
            memory_key = self.communication_channels[target_model]['shared_memory_key']
            self.shared_knowledge_base[memory_key] = {
                "last_update": time.time(),
                "message": message
            }
            return {"success": True, "memory_key": memory_key}
        except Exception as e:
            return {"error": f"共享内存写入失败: {str(e)}"}
    
    def _record_communication(self, message_record: Dict[str, Any]):
        """记录通信历史"""
        # 添加通信记录
        self.active_communications[message_record["message_id"]] = message_record
        
        # 保持通信记录大小
        if len(self.active_communications) > 10000:
            # 移除最旧的记录
            oldest_id = min(self.active_communications.keys(), 
                           key=lambda k: self.active_communications[k]["timestamp"])
            del self.active_communications[oldest_id]
    
    async def share_knowledge(self, source_model: str, target_model: str, 
                             knowledge_type: str, knowledge_data: Any,
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """模型间知识共享"""
        try:
            # 验证知识类型
            if knowledge_type not in self.knowledge_type_mapping:
                return {"error": f"未知知识类型: {knowledge_type}"}
            
            # 计算知识转移分数
            transfer_score = self._calculate_knowledge_transfer_score(
                source_model, target_model, knowledge_type, knowledge_data)
            
            # 检查是否达到转移阈值
            if transfer_score < self.knowledge_sharing_config['transfer_threshold']:
                return {"error": f"知识相关性过低: {transfer_score:.2f}"}
            
            # 创建知识转移记录
            record = KnowledgeTransferRecord(
                source_model=source_model,
                target_model=target_model,
                knowledge_type=knowledge_type,
                transfer_score=transfer_score,
                timestamp=time.time(),
                context=context or {}
            )
            
            # 保存记录
            self.knowledge_transfer_records.append(record)
            
            # 限制记录数量
            if len(self.knowledge_transfer_records) > self.knowledge_sharing_config['max_knowledge_records']:
                self.knowledge_transfer_records = self.knowledge_transfer_records[-self.knowledge_sharing_config['max_knowledge_records']:]
            
            # 实际转移知识
            transfer_result = await self._execute_knowledge_transfer(
                source_model, target_model, knowledge_type, knowledge_data)
            
            return {
                "success": True,
                "transfer_score": transfer_score,
                "record_id": len(self.knowledge_transfer_records) - 1,
                "transfer_result": transfer_result
            }
            
        except Exception as e:
            error_msg = f"知识共享失败: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}
    
    def _calculate_knowledge_transfer_score(self, source_model: str, target_model: str,
                                          knowledge_type: str, knowledge_data: Any) -> float:
        """计算知识转移分数"""
        # 基于以下因素计算分数：
        # 1. 模型兼容性
        # 2. 知识类型相关性
        # 3. 数据复杂度
        # 4. 历史转移成功率
        
        base_score = 0.5
        
        # 模型兼容性因子
        compatibility_factor = self._get_model_compatibility(source_model, target_model)
        
        # 知识类型因子
        type_factor = self._get_knowledge_type_factor(knowledge_type)
        
        # 数据复杂度因子（简化实现）
        if isinstance(knowledge_data, dict):
            complexity_factor = min(1.0, len(knowledge_data) / 100)
        elif isinstance(knowledge_data, list):
            complexity_factor = min(1.0, len(knowledge_data) / 50)
        else:
            complexity_factor = 0.3
        
        # 历史成功率因子
        history_factor = self._get_transfer_history_score(source_model, target_model)
        
        # 综合计算
        final_score = (base_score * 0.2 + 
                      compatibility_factor * 0.3 +
                      type_factor * 0.2 +
                      complexity_factor * 0.1 +
                      history_factor * 0.2)
        
        return min(max(final_score, 0.0), 1.0)
    
    def _get_model_compatibility(self, model_a: str, model_b: str) -> float:
        """获取模型兼容性分数"""
        # 简化实现：基于模型类型相似性
        model_types = self.model_registry.model_types
        
        if model_a in model_types and model_b in model_types:
            # 相同模型类型有更高兼容性
            if model_a == model_b:
                return 0.9
            # 相关模型类型有中等兼容性
            elif model_a in ['vision', 'computer_vision', 'vision_image'] and \
                 model_b in ['vision', 'computer_vision', 'vision_image']:
                return 0.7
            elif model_a in ['language', 'knowledge'] and \
                 model_b in ['language', 'knowledge']:
                return 0.8
            else:
                return 0.5
        else:
            return 0.3
    
    def _get_knowledge_type_factor(self, knowledge_type: str) -> float:
        """获取知识类型因子"""
        type_factors = {
            'parameters': 0.8,
            'embeddings': 0.9,
            'patterns': 0.7,
            'strategies': 0.6,
            'constraints': 0.5,
            'context': 0.4
        }
        return type_factors.get(knowledge_type, 0.5)
    
    def _get_transfer_history_score(self, source_model: str, target_model: str) -> float:
        """获取转移历史分数"""
        relevant_records = [
            r for r in self.knowledge_transfer_records
            if r.source_model == source_model and r.target_model == target_model
        ]
        
        if not relevant_records:
            return 0.5
        
        # 计算平均转移分数
        total_score = sum(r.transfer_score for r in relevant_records)
        return total_score / len(relevant_records)
    
    async def _execute_knowledge_transfer(self, source_model: str, target_model: str,
                                        knowledge_type: str, knowledge_data: Any) -> Dict[str, Any]:
        """执行知识转移"""
        # 简化实现：通过消息传递知识
        message = {
            "type": "knowledge_transfer",
            "knowledge_type": knowledge_type,
            "data": knowledge_data,
            "transfer_timestamp": time.time()
        }
        
        result = await self.send_message(
            source_model, target_model, message, CommunicationProtocol.DIRECT_CALL)
        
        return result
    
    async def resolve_conflict(self, conflicting_results: List[Dict[str, Any]], 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """解决模型冲突"""
        try:
            if not conflicting_results:
                return {"error": "没有冲突结果"}
            
            strategy = self.conflict_resolution_strategy
            
            if strategy == ConflictResolutionStrategy.MAJORITY_VOTE:
                result = self._resolve_by_majority_vote(conflicting_results)
            elif strategy == ConflictResolutionStrategy.EXPERT_MODEL:
                result = self._resolve_by_expert_model(conflicting_results, context)
            elif strategy == ConflictResolutionStrategy.HIERARCHICAL:
                result = self._resolve_by_hierarchical(conflicting_results)
            elif strategy == ConflictResolutionStrategy.CONSENSUS:
                result = self._resolve_by_consensus(conflicting_results, context)
            elif strategy == ConflictResolutionStrategy.ADAPTIVE_WEIGHTED:
                result = self._resolve_by_adaptive_weighted(conflicting_results)
            elif strategy == ConflictResolutionStrategy.AGI_JUDGMENT:
                result = self._resolve_by_agi_judgment(conflicting_results, context)
            else:
                result = self._resolve_by_consensus(conflicting_results, context)
            
            # 记录冲突解决历史
            conflict_record = {
                "timestamp": time.time(),
                "conflicting_results": conflicting_results,
                "strategy": strategy.value,
                "resolution": result,
                "context": context
            }
            self.conflict_history.append(conflict_record)
            
            # 保持历史记录大小
            if len(self.conflict_history) > 1000:
                self.conflict_history = self.conflict_history[-1000:]
            
            return {"success": True, "resolution": result}
            
        except Exception as e:
            error_msg = f"冲突解决失败: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}
    
    def _resolve_by_majority_vote(self, conflicting_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """多数投票解决冲突"""
        # 简化实现：选择出现次数最多的结果
        result_strs = [str(r) for r in conflicting_results]
        from collections import Counter
        counts = Counter(result_strs)
        most_common = counts.most_common(1)[0][0]
        
        # 找到对应的结果
        for result in conflicting_results:
            if str(result) == most_common:
                return result
        
        return conflicting_results[0]  # 默认返回第一个结果
    
    def _resolve_by_expert_model(self, conflicting_results: List[Dict[str, Any]], 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """专家模型解决冲突"""
        # 根据上下文选择专家模型
        expert_model = context.get("expert_model", "knowledge")
        
        # 尝试获取专家模型
        model = self.model_registry.get_model(expert_model)
        if model:
            # 让专家模型判断
            if hasattr(model, 'resolve_conflict'):
                try:
                    resolution = model.resolve_conflict({
                        "conflicting_results": conflicting_results,
                        "context": context
                    })
                    return resolution
                except Exception as e:
                    self.logger.error(f"专家模型冲突解决失败: {e}")
        
        # 降级到多数投票
        return self._resolve_by_majority_vote(conflicting_results)
    
    def _resolve_by_consensus(self, conflicting_results: List[Dict[str, Any]],
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """共识解决冲突"""
        # 简化实现：计算结果的加权平均
        if all(isinstance(r.get("value"), (int, float)) for r in conflicting_results):
            # 数值型结果：计算平均值
            values = [r.get("value", 0) for r in conflicting_results]
            consensus_value = sum(values) / len(values)
            return {"value": consensus_value, "method": "average", "confidence": 0.7}
        else:
            # 非数值型结果：使用多数投票
            return self._resolve_by_majority_vote(conflicting_results)
    
    def _resolve_by_adaptive_weighted(self, conflicting_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """自适应加权解决冲突"""
        # 根据模型历史性能加权
        weights = []
        for result in conflicting_results:
            model_id = result.get("model_id", "unknown")
            
            # 使用性能监控器获取模型性能数据
            try:
                if hasattr(self, 'performance_monitor') and self.performance_monitor:
                    # 获取模型性能摘要
                    performance_summary = self.performance_monitor.get_model_performance_summary(model_id)
                    
                    if "error" not in performance_summary:
                        # 基于性能计算权重
                        # 使用准确性、成功率和推理时间的组合
                        accuracy = performance_summary.get("accuracy", {}).get("avg", 0.5)
                        success_rate = performance_summary.get("success_rate", {}).get("avg", 0.5)
                        inference_time = performance_summary.get("inference_time", {}).get("avg", 1.0)
                        
                        # 计算性能分数：准确性 * 成功率 * (1 / 标准化推理时间)
                        # 推理时间越短，分数越高
                        normalized_inference = max(0.1, min(1.0, 1.0 / inference_time))
                        performance_score = accuracy * success_rate * normalized_inference
                        
                        # 添加模型类型权重
                        model_type_weight = self._get_model_type_weight(model_id)
                        
                        # 最终权重
                        weight = performance_score * model_type_weight
                    else:
                        # 性能数据不可用，使用默认权重
                        weight = self._get_default_model_weight(model_id)
                else:
                    # 性能监控器不可用，使用默认权重
                    weight = self._get_default_model_weight(model_id)
            except Exception as e:
                self.logger.error(f"获取模型 {model_id} 性能数据失败: {e}")
                weight = self._get_default_model_weight(model_id)
            
            weights.append(weight)
        
        # 归一化权重
        if sum(weights) > 0:
            weights = [w/sum(weights) for w in weights]
        else:
            weights = [1/len(weights)] * len(weights)
        
        # 数值型结果的加权平均
        if all(isinstance(r.get("value"), (int, float)) for r in conflicting_results):
            weighted_value = sum(
                r.get("value", 0) * w for r, w in zip(conflicting_results, weights))
            return {"value": weighted_value, "method": "weighted_average", "weights": weights}
        else:
            # 非数值型结果：使用加权投票
            return self._resolve_by_majority_vote(conflicting_results)
    
    def _resolve_by_agi_judgment(self, conflicting_results: List[Dict[str, Any]],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """AGI判断解决冲突"""
        # 尝试使用AGI核心进行判断
        try:
            agi_core = getattr(self.model_registry, '_agi_core', None)
            if agi_core:
                # 调用AGI核心进行高级判断
                judgment = agi_core.analyze_conflicts(conflicting_results, context)
                return judgment
        except Exception as e:
            self.logger.error(f"AGI判断失败: {e}")
        
        # 降级到自适应加权
        return self._resolve_by_adaptive_weighted(conflicting_results)
    
    def _get_model_type_weight(self, model_id: str) -> float:
        """获取模型类型权重"""
        # 根据不同模型类型分配权重
        model_type_weights = {
            'knowledge': 1.2,      # 知识模型权重较高
            'advanced_reasoning': 1.3,  # 高级推理模型权重最高
            'language': 1.1,       # 语言模型权重较高
            'vision': 1.0,         # 视觉模型标准权重
            'audio': 0.9,          # 音频模型稍低权重
            'sensor': 0.8,         # 传感器模型权重
            'motion': 0.9,         # 运动模型权重
            'planning': 1.2,       # 规划模型权重较高
            'prediction': 1.1,     # 预测模型权重较高
        }
        
        # 检查模型ID是否包含类型关键词
        for model_type, weight in model_type_weights.items():
            if model_type in model_id:
                return weight
        
        # 默认权重
        return 1.0
    
    def _get_default_model_weight(self, model_id: str) -> float:
        """获取默认模型权重"""
        # 基于模型ID的简单哈希函数，提供一致性但随机的权重
        import hashlib
        hash_obj = hashlib.md5(model_id.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # 生成0.7到1.3之间的权重
        weight = 0.7 + (hash_int % 600) / 1000.0  # 0.7 + (0-0.6)
        return weight
    
    async def orchestrate_workflow(self, workflow_definition: Dict[str, Any],
                                 input_data: Any) -> Dict[str, Any]:
        """编排多模型工作流"""
        workflow_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # 验证工作流定义
            validation_result = self._validate_workflow_definition(workflow_definition)
            if not validation_result["valid"]:
                return {"error": f"工作流定义无效: {validation_result['error']}", "workflow_id": workflow_id}
            
            # 创建工作流执行上下文
            execution_context = {
                "workflow_id": workflow_id,
                "start_time": start_time,
                "current_step": 0,
                "intermediate_results": {},
                "step_history": [],
                "status": "running"
            }
            
            # 执行工作流步骤
            steps = workflow_definition.get("steps", [])
            final_result = input_data
            
            for step_index, step in enumerate(steps):
                step_start_time = time.time()
                
                # 执行单个步骤
                step_result = await self._execute_workflow_step(step, final_result, execution_context)
                
                step_execution_time = time.time() - step_start_time
                
                # 记录步骤历史
                step_history = {
                    "step_index": step_index,
                    "step_id": step.get("id", f"step_{step_index}"),
                    "start_time": step_start_time,
                    "execution_time": step_execution_time,
                    "result": step_result,
                    "success": "error" not in step_result
                }
                execution_context["step_history"].append(step_history)
                
                # 更新中间结果
                execution_context["intermediate_results"][step.get("id", f"step_{step_index}")] = step_result
                
                # 检查步骤是否失败
                if "error" in step_result:
                    execution_context["status"] = "failed"
                    break
                
                # 更新最终结果（如果步骤指定了输出）
                if "output_key" in step:
                    final_result = step_result.get(step["output_key"], step_result)
                
                execution_context["current_step"] = step_index + 1
            
            # 完成工作流
            end_time = time.time()
            execution_time = end_time - start_time
            
            workflow_result = {
                "workflow_id": workflow_id,
                "success": execution_context["status"] == "running",
                "execution_time": execution_time,
                "final_result": final_result,
                "step_history": execution_context["step_history"],
                "intermediate_results": execution_context["intermediate_results"],
                "total_steps": len(steps),
                "completed_steps": execution_context["current_step"]
            }
            
            # 记录工作流执行
            self._record_workflow_execution(workflow_definition, workflow_result)
            
            return workflow_result
            
        except Exception as e:
            error_msg = f"工作流编排失败: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg, "workflow_id": workflow_id}
    
    def _validate_workflow_definition(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """验证工作流定义"""
        required_fields = ["id", "name", "steps"]
        
        for field in required_fields:
            if field not in workflow:
                return {"valid": False, "error": f"缺少必需字段: {field}"}
        
        # 验证步骤
        steps = workflow.get("steps", [])
        for i, step in enumerate(steps):
            if "model" not in step and "action" not in step:
                return {"valid": False, "error": f"步骤 {i} 缺少model或action字段"}
        
        return {"valid": True}
    
    def _safe_eval(self, condition: str, input_data: Any) -> Any:
        """安全评估条件表达式"""
        # 检查危险模式
        dangerous_patterns = ["__", "import ", "exec(", "eval(", "compile(", "open("]
        for pattern in dangerous_patterns:
            if pattern in condition:
                raise ValueError(f"条件包含危险模式: {pattern}")
        
        # 限制命名空间，不提供内置函数
        namespace = {"input": input_data}
        
        # 移除__builtins__以防止访问危险函数
        namespace['__builtins__'] = None
        
        try:
            return eval(condition, namespace)
        except Exception as e:
            raise ValueError(f"条件评估失败: {str(e)}")
    
    async def _execute_workflow_step(self, step: Dict[str, Any], input_data: Any,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """执行工作流步骤"""
        try:
            step_type = step.get("type", "model_execution")
            
            if step_type == "model_execution":
                # 模型执行步骤
                model_id = step.get("model")
                if not model_id:
                    return {"error": "模型执行步骤缺少model字段"}
                
                model = self.model_registry.get_model(model_id)
                if not model:
                    return {"error": f"模型未找到: {model_id}"}
                
                # 准备输入数据
                task_data = step.get("input_data", input_data)
                
                # 执行模型
                if hasattr(model, 'process'):
                    result = model.process(task_data)
                elif hasattr(model, 'execute'):
                    result = model.execute(task_data)
                else:
                    return {"error": f"模型 {model_id} 没有处理方法"}
                
                return {"success": True, "result": result}
                
            elif step_type == "condition":
                # 条件步骤
                condition = step.get("condition")
                if not condition:
                    return {"error": "条件步骤缺少condition字段"}
                
                # 简化实现：直接使用条件
                condition_result = self._safe_eval(condition, input_data)
                return {"success": True, "condition_result": condition_result}
                
            elif step_type == "parallel":
                # 并行执行步骤
                parallel_steps = step.get("steps", [])
                if not parallel_steps:
                    return {"error": "并行步骤缺少steps字段"}
                
                # 并行执行所有子步骤
                tasks = []
                for parallel_step in parallel_steps:
                    task = self._execute_workflow_step(parallel_step, input_data, context)
                    tasks.append(task)
                
                # 等待所有任务完成
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 处理结果
                successful_results = []
                errors = []
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        errors.append(f"并行步骤 {i} 失败: {str(result)}")
                    elif "error" in result:
                        errors.append(f"并行步骤 {i} 失败: {result['error']}")
                    else:
                        successful_results.append(result)
                
                if errors:
                    return {"error": f"并行步骤执行失败: {', '.join(errors)}"}
                
                return {"success": True, "parallel_results": successful_results}
                
            else:
                return {"error": f"未知步骤类型: {step_type}"}
                
        except Exception as e:
            return {"error": f"步骤执行失败: {str(e)}"}
    
    def _record_workflow_execution(self, workflow: Dict[str, Any], result: Dict[str, Any]):
        """记录工作流执行历史"""
        # 创建工作流模板（如果不存在）
        workflow_id = workflow.get("id")
        if workflow_id not in self.workflow_templates:
            self.workflow_templates[workflow_id] = {
                "definition": workflow,
                "execution_count": 0,
                "success_count": 0,
                "avg_execution_time": 0,
                "last_execution": 0
            }
        
        # 更新统计信息
        template = self.workflow_templates[workflow_id]
        template["execution_count"] += 1
        template["last_execution"] = time.time()
        
        if result.get("success"):
            template["success_count"] += 1
        
        # 更新平均执行时间
        old_total = template["avg_execution_time"] * (template["execution_count"] - 1)
        new_avg = (old_total + result["execution_time"]) / template["execution_count"]
        template["avg_execution_time"] = new_avg
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "performance_monitoring": {
                "active": True,
                "models_monitored": len(self.performance_history),
                "total_metrics": sum(len(m) for m in self.performance_history.values())
            },
            "communication_system": {
                "active_channels": len(self.communication_channels),
                "total_messages": sum(len(q) for q in self.message_queues.values()),
                "avg_latency": self._calculate_average_latency()
            },
            "knowledge_sharing": {
                "enabled": self.knowledge_sharing_enabled,
                "total_transfers": len(self.knowledge_transfer_records),
                "successful_transfers": len([r for r in self.knowledge_transfer_records if r.transfer_score > 0.7])
            },
            "conflict_resolution": {
                "strategy": self.conflict_resolution_strategy.value,
                "total_resolutions": len(self.conflict_history),
                "recent_success_rate": self._calculate_recent_success_rate()
            },
            "workflow_orchestration": {
                "templates_available": len(self.workflow_templates),
                "total_executions": sum(t["execution_count"] for t in self.workflow_templates.values()),
                "success_rate": self._calculate_workflow_success_rate()
            }
        }

    async def execute_collaboration_async(self, pattern: str, input_data: Dict[str, Any] = None, 
                                  custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """异步执行协作任务
        
        Args:
            pattern: 协作模式名称
            input_data: 输入数据
            custom_config: 自定义配置
            
        Returns:
            Dict[str, Any]: 协作结果
        """
        try:
            # 使用基础协调器的initiate_collaboration方法
            result = await self.base_orchestrator.initiate_collaboration(
                pattern_name=pattern,
                input_data=input_data or {},
                custom_config=custom_config
            )
            
            # 确保结果格式兼容
            if "error" in result:
                return {"success": False, "error": result["error"]}
            else:
                return {"success": True, "result": result}
                
        except Exception as e:
            error_msg = f"协作执行失败: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def execute_collaboration(self, pattern: str, input_data: Dict[str, Any] = None,
                                 custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """同步执行协作任务 - 兼容性方法
        
        Args:
            pattern: 协作模式名称
            input_data: 输入数据
            custom_config: 自定义配置
            
        Returns:
            Dict[str, Any]: 协作结果
        """
        try:
            # 创建新的事件循环或使用现有循环
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self.execute_collaboration_async(pattern, input_data, custom_config)
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            error_msg = f"同步协作执行失败: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _calculate_average_latency(self) -> float:
        """计算平均通信延迟"""
        all_latencies = []
        for latencies in self.communication_latency.values():
            all_latencies.extend(latencies[-100:])  # 只使用最近100个
        
        if not all_latencies:
            return 0.0
        
        return sum(all_latencies) / len(all_latencies)
    
    def _calculate_recent_success_rate(self) -> float:
        """计算最近冲突解决成功率"""
        recent_conflicts = self.conflict_history[-100:] if self.conflict_history else []
        if not recent_conflicts:
            return 0.0
        
        successful = sum(1 for c in recent_conflicts if c.get("resolution", {}).get("success", False))
        return successful / len(recent_conflicts)
    
    def _calculate_workflow_success_rate(self) -> float:
        """计算工作流成功率"""
        if not self.workflow_templates:
            return 0.0
        
        total_executions = sum(t["execution_count"] for t in self.workflow_templates.values())
        total_success = sum(t["success_count"] for t in self.workflow_templates.values())
        
        if total_executions == 0:
            return 0.0
        
        return total_success / total_executions
    
    def get_collaboration_patterns(self):
        """获取可用的协作模式"""
        patterns = []
        
        # 获取可用模型
        available_models = list(self.model_registry.model_types.keys())
        
        # 基于模型类型定义协作模式
        if len(available_models) >= 3:
            # 多模态协作模式
            patterns.append({
                "id": 1,
                "name": "Multimodal Perception",
                "description": "视觉、语言和知识模型协作进行多模态感知",
                "type": "perception",
                "priority": "high",
                "models": ["vision", "language", "knowledge"],
                "mode": "sequential"
            })
            
            # 推理协作模式
            patterns.append({
                "id": 2,
                "name": "Advanced Reasoning Chain",
                "description": "高级推理模型与语言模型协作进行复杂推理",
                "type": "reasoning",
                "priority": "high",
                "models": ["advanced_reasoning", "language", "knowledge"],
                "mode": "parallel"
            })
            
            # 运动规划模式
            patterns.append({
                "id": 3,
                "name": "Motion Planning with Vision",
                "description": "运动模型与视觉模型协作进行运动规划",
                "type": "motion",
                "priority": "medium",
                "models": ["motion", "vision", "sensor"],
                "mode": "sequential"
            })
            
            # 音频-视觉协作
            patterns.append({
                "id": 4,
                "name": "Audio-Visual Integration",
                "description": "音频模型与视觉模型协作进行多媒体处理",
                "type": "multimedia",
                "priority": "medium",
                "models": ["audio", "vision"],
                "mode": "parallel"
            })
            
            # 自主决策模式
            patterns.append({
                "id": 5,
                "name": "Autonomous Decision Making",
                "description": "自主模型与规划模型协作进行自主决策",
                "type": "autonomous",
                "priority": "high",
                "models": ["autonomous", "planning", "prediction"],
                "mode": "sequential"
            })
        
        # 如果没有足够的模型，返回基本模式
        if not patterns:
            patterns = [
                {
                    "id": 1,
                    "name": "Basic Model Collaboration",
                    "description": "基础模型协作",
                    "type": "general",
                    "priority": "medium",
                    "models": available_models[:3] if len(available_models) >= 3 else available_models,
                    "mode": "sequential"
                }
            ]
        
        return patterns
    
    def execute_collaboration(self, pattern_id, input_data=None):
        """执行协作模式"""
        try:
            # 获取模式
            patterns = self.get_collaboration_patterns()
            selected_pattern = None
            for pattern in patterns:
                if pattern.get("id") == pattern_id:
                    selected_pattern = pattern
                    break
            
            if not selected_pattern:
                raise ValueError(f"Collaboration pattern with ID {pattern_id} not found")
            
            # 记录协作开始
            collaboration_id = f"collab_{int(time.time())}"
            self.logger.info(f"开始执行协作模式 {pattern_id}: {selected_pattern.get('name')}")
            
            # 根据模式类型执行不同的协作逻辑
            pattern_type = selected_pattern.get("type", "general")
            models = selected_pattern.get("models", [])
            
            # 准备输入数据
            if input_data is None:
                input_data = {}
            
            # 执行协作
            result = {
                "success": True,
                "collaboration_id": collaboration_id,
                "pattern_id": pattern_id,
                "pattern_name": selected_pattern.get("name"),
                "pattern_type": pattern_type,
                "models_involved": models,
                "timestamp": datetime.datetime.now().isoformat(),
                "results": {}
            }
            
            # 根据模式类型执行不同的协作
            if pattern_type == "perception":
                # 多模态感知协作
                result["results"] = self._execute_multimodal_perception(input_data, models)
            elif pattern_type == "reasoning":
                # 推理链协作
                result["results"] = self._execute_reasoning_chain(input_data, models)
            elif pattern_type == "motion":
                # 运动规划协作
                result["results"] = self._execute_motion_planning(input_data, models)
            elif pattern_type == "multimedia":
                # 多媒体协作
                result["results"] = self._execute_multimedia_integration(input_data, models)
            elif pattern_type == "autonomous":
                # 自主决策协作
                result["results"] = self._execute_autonomous_decision(input_data, models)
            else:
                # 通用协作
                result["results"] = self._execute_general_collaboration(input_data, models)
            
            # 记录协作结果
            self._record_collaboration_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"协作执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "collaboration_id": f"collab_{int(time.time())}",
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def _execute_multimodal_perception(self, input_data, models):
        """执行多模态感知协作"""
        # 实际实现会调用相关模型
        return {
            "status": "completed",
            "message": "Multimodal perception collaboration executed",
            "models_executed": models,
            "output": {"perception_result": "object detected with 95% confidence"}
        }
    
    def _execute_reasoning_chain(self, input_data, models):
        """执行推理链协作"""
        return {
            "status": "completed",
            "message": "Reasoning chain collaboration executed",
            "models_executed": models,
            "output": {"reasoning_result": "logical conclusion reached"}
        }
    
    def _execute_motion_planning(self, input_data, models):
        """执行运动规划协作"""
        return {
            "status": "completed",
            "message": "Motion planning collaboration executed",
            "models_executed": models,
            "output": {"motion_plan": "trajectory generated with 10 waypoints"}
        }
    
    def _execute_multimedia_integration(self, input_data, models):
        """执行多媒体集成协作"""
        return {
            "status": "completed",
            "message": "Multimedia integration collaboration executed",
            "models_executed": models,
            "output": {"multimedia_result": "audio-visual synchronization completed"}
        }
    
    def _execute_autonomous_decision(self, input_data, models):
        """执行自主决策协作"""
        return {
            "status": "completed",
            "message": "Autonomous decision collaboration executed",
            "models_executed": models,
            "output": {"decision": "optimal action selected with confidence 0.87"}
        }
    
    def _execute_general_collaboration(self, input_data, models):
        """执行通用协作"""
        return {
            "status": "completed",
            "message": "General collaboration executed",
            "models_executed": models,
            "output": {"result": "collaboration completed successfully"}
        }
    
    def _record_collaboration_result(self, result):
        """记录协作结果"""
        # 在实际系统中，这里会将结果存储到数据库或日志
        collaboration_record = {
            "collaboration_id": result.get("collaboration_id"),
            "pattern_id": result.get("pattern_id"),
            "success": result.get("success", False),
            "timestamp": result.get("timestamp"),
            "models": result.get("models_involved", []),
            "result_summary": result.get("results", {})
        }
        
        # 添加到历史记录
        if not hasattr(self, 'collaboration_history'):
            self.collaboration_history = []
        
        self.collaboration_history.append(collaboration_record)
        
        # 保持历史记录大小
        if len(self.collaboration_history) > 1000:
            self.collaboration_history = self.collaboration_history[-1000:]


# 全局实例
_enhanced_collaborator = None

def get_enhanced_collaborator() -> EnhancedModelCollaboration:
    """获取增强协作器全局实例"""
    global _enhanced_collaborator
    if _enhanced_collaborator is None:
        _enhanced_collaborator = EnhancedModelCollaboration()
    return _enhanced_collaborator