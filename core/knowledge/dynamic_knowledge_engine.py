"""
动态知识引擎 - Dynamic Knowledge Engine

实现知识库的动态更新、逻辑推理、自更新和纠错能力，解决现有知识库静态问题。
整合增强知识图谱和跨模型知识融合，提供完整的知识管理生命周期。

核心功能：
1. 动态知识获取与融合
2. 逻辑推理与关联发现
3. 知识自更新与演化
4. 冲突检测与智能纠错
5. 多源知识集成管理
6. 实时知识查询与推理

设计目标：
- 将静态JSON知识库转化为动态智能知识系统
- 实现知识关联、逻辑推理和自更新能力
- 提供跨领域知识融合与冲突解决
- 支持知识演化与持续优化
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
import random
import math

logger = logging.getLogger(__name__)

from core.knowledge.enhanced_knowledge_graph import (
    EnhancedKnowledgeGraph, KnowledgeEntity, EnhancedRelation, 
    Domain, RelationType, get_enhanced_knowledge_graph
)
from core.knowledge.cross_model_knowledge_fusion import CrossModelKnowledgeFusion

class UpdateSource(Enum):
    """知识更新来源"""
    MODEL_OUTPUT = "model_output"          # 模型输出
    EXTERNAL_API = "external_api"          # 外部API
    USER_INTERACTION = "user_interaction"  # 用户交互
    SYSTEM_DISCOVERY = "system_discovery"  # 系统发现
    KNOWLEDGE_FUSION = "knowledge_fusion"  # 知识融合

class KnowledgeQuality(Enum):
    """知识质量等级"""
    HIGH_CONFIDENCE = "high_confidence"    # 高置信度
    MEDIUM_CONFIDENCE = "medium_confidence" # 中置信度
    LOW_CONFIDENCE = "low_confidence"      # 低置信度
    UNVERIFIED = "unverified"              # 未验证
    CONFLICTING = "conflicting"            # 冲突

@dataclass
class KnowledgeUpdate:
    """知识更新记录"""
    update_id: str
    entity_id: str
    update_type: str
    old_value: Any
    new_value: Any
    source: UpdateSource
    confidence: float = 0.8
    timestamp: float = field(default_factory=time.time)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InferenceRule:
    """推理规则"""
    rule_id: str
    condition: str  # 条件表达式
    action: str     # 动作/结论
    priority: int = 1
    confidence: float = 0.9
    activation_count: int = 0
    last_activated: Optional[float] = None

@dataclass
class KnowledgeConflict:
    """知识冲突"""
    conflict_id: str
    entity_id: str
    attribute_name: str
    conflicting_values: List[Any]
    sources: List[UpdateSource]
    confidence_scores: List[float]
    detected_at: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_method: Optional[str] = None

class DynamicKnowledgeEngine:
    """动态知识引擎"""
    
    def __init__(self, existing_knowledge_path: Optional[str] = None):
        """初始化动态知识引擎"""
        # 核心知识组件
        self.knowledge_graph = get_enhanced_knowledge_graph()
        self.knowledge_fusion = CrossModelKnowledgeFusion()
        
        # 动态更新系统
        self.update_history: List[KnowledgeUpdate] = []
        self.pending_updates = deque()
        self.update_lock = threading.RLock()
        
        # 推理系统
        self.inference_rules: Dict[str, InferenceRule] = {}
        self.inference_cache: Dict[str, Any] = {}
        
        # 冲突管理系统
        self.active_conflicts: Dict[str, KnowledgeConflict] = {}
        self.resolved_conflicts = deque(maxlen=1000)
        
        # 质量监控
        self.quality_metrics = defaultdict(lambda: {
            "total_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "avg_confidence": 0.0,
            "last_update_time": 0.0
        })
        
        # 自更新配置
        self.auto_update_config = {
            "enable_auto_discovery": True,
            "enable_auto_correction": True,
            "enable_auto_optimization": True,
            "update_check_interval": 3600,  # 1小时
            "max_updates_per_cycle": 100,
            "min_confidence_threshold": 0.6
        }
        
        # 初始化引擎
        self._initialize_engine()
        
        # 加载现有知识（如果有）
        if existing_knowledge_path:
            self._load_existing_knowledge(existing_knowledge_path)
        
        # 启动监控线程
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("动态知识引擎初始化完成")
    
    def _initialize_engine(self):
        """初始化引擎"""
        # 注册默认推理规则
        self._register_default_rules()
        
        # 初始化知识融合
        self._initialize_knowledge_fusion()
        
        # 设置自动更新定时器
        self._setup_auto_update_timer()
    
    def _register_default_rules(self):
        """注册默认推理规则"""
        # 实体关联规则
        self.add_inference_rule(
            rule_id="associate_similar_entities",
            condition="entities_similarity > 0.7",
            action="create_cross_domain_relation",
            priority=2,
            confidence=0.8
        )
        
        # 知识更新规则
        self.add_inference_rule(
            rule_id="update_low_confidence_knowledge",
            condition="knowledge_confidence < 0.5 AND has_higher_confidence_source",
            action="update_knowledge_from_source",
            priority=1,
            confidence=0.9
        )
        
        # 冲突解决规则
        self.add_inference_rule(
            rule_id="resolve_attribute_conflict",
            condition="attribute_has_multiple_values AND values_conflict",
            action="resolve_conflict_by_confidence",
            priority=3,
            confidence=0.85
        )
        
        # 领域关联规则
        self.add_inference_rule(
            rule_id="discover_cross_domain_associations",
            condition="entity_has_multiple_domains AND no_cross_domain_relation",
            action="create_domain_bridge",
            priority=2,
            confidence=0.75
        )
        
        logger.info(f"已注册 {len(self.inference_rules)} 个默认推理规则")
    
    def _initialize_knowledge_fusion(self):
        """初始化知识融合"""
        # 注册默认模型图（如果有）
        # 在实际应用中，这里会从配置文件或数据库加载
        pass
    
    def _setup_auto_update_timer(self):
        """设置自动更新定时器"""
        if self.auto_update_config["enable_auto_discovery"]:
            # 创建一个后台线程定期检查更新
            self.auto_update_thread = threading.Thread(target=self._auto_update_loop, daemon=True)
            self.auto_update_thread.start()
    
    def _load_existing_knowledge(self, knowledge_path: str):
        """加载现有知识"""
        try:
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
            
            if isinstance(knowledge_data, dict):
                self._import_knowledge_from_dict(knowledge_data)
            elif isinstance(knowledge_data, list):
                for item in knowledge_data:
                    self._import_knowledge_item(item)
            
            logger.info(f"从 {knowledge_path} 加载了 {len(knowledge_data)} 个知识项")
            
        except Exception as e:
            logger.error(f"加载现有知识失败: {str(e)}")
    
    def _import_knowledge_from_dict(self, knowledge_dict: Dict[str, Any]):
        """从字典导入知识"""
        # 导入实体
        entities = knowledge_dict.get("entities", {})
        for entity_id, entity_data in entities.items():
            self._import_entity(entity_id, entity_data)
        
        # 导入关系
        relations = knowledge_dict.get("relations", {})
        for relation_id, relation_data in relations.items():
            self._import_relation(relation_id, relation_data)
    
    def _import_entity(self, entity_id: str, entity_data: Dict[str, Any]):
        """导入实体"""
        try:
            # 创建实体
            entity = KnowledgeEntity(
                entity_id=entity_id,
                name=entity_data.get("name", entity_id),
                entity_type=entity_data.get("type", "unknown"),
                attributes=entity_data.get("attributes", {}),
                domains=[Domain(d) for d in entity_data.get("domains", ["knowledge"])]
            )
            
            # 添加到知识图谱
            self.knowledge_graph.entities[entity_id] = entity
            
            # 更新索引
            self.knowledge_graph.entity_name_index[entity.name] = entity_id
            
            # 添加到领域索引
            for domain in entity.domains:
                self.knowledge_graph.domain_entities[domain].add(entity_id)
            
        except Exception as e:
            logger.error(f"导入实体失败 {entity_id}: {str(e)}")
    
    def _import_relation(self, relation_id: str, relation_data: Dict[str, Any]):
        """导入关系"""
        try:
            # 创建关系
            relation = EnhancedRelation(
                relation_id=relation_id,
                source_entity=relation_data["source"],
                target_entity=relation_data["target"],
                relation_type=RelationType(relation_data.get("type", "associated_with")),
                confidence=relation_data.get("confidence", 0.8),
                strength=relation_data.get("strength", 0.5),
                domains=[Domain(d) for d in relation_data.get("domains", ["knowledge"])]
            )
            
            # 添加到知识图谱
            self.knowledge_graph.relations[relation_id] = relation
            
            # 更新索引
            key = (relation.source_entity, relation.target_entity, relation.relation_type.value)
            self.knowledge_graph.relation_index[key] = relation_id
            
            # 更新实体关系映射
            self.knowledge_graph.entity_relations[relation.source_entity].append(relation_id)
            self.knowledge_graph.entity_relations[relation.target_entity].append(relation_id)
            
        except Exception as e:
            logger.error(f"导入关系失败 {relation_id}: {str(e)}")
    
    def add_inference_rule(self, rule_id: str, condition: str, action: str,
                         priority: int = 1, confidence: float = 0.8):
        """添加推理规则"""
        rule = InferenceRule(
            rule_id=rule_id,
            condition=condition,
            action=action,
            priority=priority,
            confidence=confidence
        )
        
        self.inference_rules[rule_id] = rule
        logger.info(f"推理规则已添加: {rule_id}")
    
    def update_knowledge(self, entity_id: str, attribute_name: str, new_value: Any,
                        source: UpdateSource, confidence: float = 0.8,
                        reason: str = "") -> Dict[str, Any]:
        """更新知识"""
        with self.update_lock:
            # 检查实体是否存在
            if entity_id not in self.knowledge_graph.entities:
                # 创建新实体
                entity = KnowledgeEntity(
                    entity_id=entity_id,
                    name=entity_id,
                    entity_type="unknown",
                    domains=[Domain.KNOWLEDGE]
                )
                self.knowledge_graph.entities[entity_id] = entity
                self.knowledge_graph.entity_name_index[entity_id] = entity_id
                
                old_value = None
            else:
                entity = self.knowledge_graph.entities[entity_id]
                old_value = entity.attributes.get(attribute_name)
            
            # 检查是否需要更新
            if old_value == new_value:
                return {
                    "success": False,
                    "message": "新值与旧值相同，无需更新",
                    "entity_id": entity_id
                }
            
            # 检查冲突
            conflict_detected = False
            if attribute_name in entity.attributes and old_value is not None:
                # 检查值是否冲突
                if self._values_conflict(old_value, new_value):
                    conflict_detected = True
                    
                    # 记录冲突
                    conflict_id = f"conflict_{len(self.active_conflicts)}"
                    conflict = KnowledgeConflict(
                        conflict_id=conflict_id,
                        entity_id=entity_id,
                        attribute_name=attribute_name,
                        conflicting_values=[old_value, new_value],
                        sources=[UpdateSource.SYSTEM_DISCOVERY, source],
                        confidence_scores=[self._get_attribute_confidence(entity, attribute_name), confidence]
                    )
                    
                    self.active_conflicts[conflict_id] = conflict
                    
                    # 尝试自动解决冲突
                    resolved = self._auto_resolve_conflict(conflict)
                    if resolved:
                        # 使用解决后的值
                        resolved_value = conflict.conflicting_values[0]  # 假设第一个是解决后的值
                        entity.attributes[attribute_name] = resolved_value
                        new_value = resolved_value
                        conflict_detected = False
                    else:
                        # 保留原值，等待人工解决
                        return {
                            "success": False,
                            "message": "检测到知识冲突，需要人工解决",
                            "conflict_id": conflict_id,
                            "entity_id": entity_id,
                            "attribute_name": attribute_name,
                            "old_value": old_value,
                            "new_value": new_value
                        }
            
            # 执行更新
            if not conflict_detected:
                entity.attributes[attribute_name] = new_value
                entity.updated_at = time.time()
                
                # 记录更新
                update_id = f"update_{len(self.update_history)}"
                update = KnowledgeUpdate(
                    update_id=update_id,
                    entity_id=entity_id,
                    update_type="attribute_update",
                    old_value=old_value,
                    new_value=new_value,
                    source=source,
                    confidence=confidence,
                    reason=reason
                )
                
                self.update_history.append(update)
                
                # 更新质量指标
                self._update_quality_metrics(entity_id, True, confidence)
                
                # 触发相关推理
                self._trigger_inference_rules(entity_id, attribute_name, new_value)
                
                logger.info(f"知识更新成功: {entity_id}.{attribute_name} = {new_value}")
                
                return {
                    "success": True,
                    "update_id": update_id,
                    "entity_id": entity_id,
                    "attribute_name": attribute_name,
                    "old_value": old_value,
                    "new_value": new_value,
                    "confidence": confidence
                }
    
    def _values_conflict(self, value1: Any, value2: Any) -> bool:
        """检查两个值是否冲突"""
        # 基本冲突检测
        if value1 is None or value2 is None:
            return False
        
        if isinstance(value1, str) and isinstance(value2, str):
            # 字符串值：检查是否矛盾
            contradictions = [
                ("是", "不是"),
                ("有", "没有"),
                ("真", "假"),
                ("正确", "错误"),
                ("存在", "不存在")
            ]
            
            for contra in contradictions:
                if (contra[0] in value1 and contra[1] in value2) or \
                   (contra[1] in value1 and contra[0] in value2):
                    return True
            
            # 检查数值差异（如果包含数字）
            import re
            numbers1 = re.findall(r'\d+\.?\d*', value1)
            numbers2 = re.findall(r'\d+\.?\d*', value2)
            
            if numbers1 and numbers2:
                try:
                    num1 = float(numbers1[0])
                    num2 = float(numbers2[0])
                    if abs(num1 - num2) > max(num1, num2) * 0.5:  # 差异超过50%
                        return True
                except:
                    pass
        
        elif isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            # 数值：差异超过阈值
            threshold = max(abs(value1), abs(value2)) * 0.3  # 30%差异
            if abs(value1 - value2) > threshold:
                return True
        
        return False
    
    def _get_attribute_confidence(self, entity: KnowledgeEntity, attribute_name: str) -> float:
        """获取属性置信度"""
        # 简化的置信度计算
        if attribute_name not in entity.attributes:
            return 0.0
        
        # 检查属性来源
        if "confidence" in entity.attributes:
            return entity.attributes.get("confidence", 0.5)
        
        # 默认置信度
        return 0.8
    
    def _auto_resolve_conflict(self, conflict: KnowledgeConflict) -> bool:
        """自动解决冲突"""
        # 策略1：使用置信度更高的值
        if len(conflict.confidence_scores) == 2:
            idx = 0 if conflict.confidence_scores[0] >= conflict.confidence_scores[1] else 1
            conflict.conflicting_values = [conflict.conflicting_values[idx]]
            conflict.resolved = True
            conflict.resolution_method = "confidence_based"
            return True
        
        # 策略2：使用最近的值（如果时间信息可用）
        # 这里可以扩展更多策略
        
        return False
    
    def _update_quality_metrics(self, entity_id: str, success: bool, confidence: float):
        """更新质量指标"""
        metrics = self.quality_metrics[entity_id]
        metrics["total_updates"] += 1
        
        if success:
            metrics["successful_updates"] += 1
        else:
            metrics["failed_updates"] += 1
        
        # 更新平均置信度
        old_avg = metrics["avg_confidence"]
        old_count = metrics["successful_updates"] - 1
        if old_count > 0:
            metrics["avg_confidence"] = (old_avg * old_count + confidence) / metrics["successful_updates"]
        else:
            metrics["avg_confidence"] = confidence
        
        metrics["last_update_time"] = time.time()
    
    def _trigger_inference_rules(self, entity_id: str, attribute_name: str, new_value: Any):
        """触发推理规则"""
        # 收集上下文信息
        context = {
            "entity_id": entity_id,
            "attribute_name": attribute_name,
            "new_value": new_value,
            "entity": self.knowledge_graph.entities.get(entity_id),
            "timestamp": time.time()
        }
        
        # 按优先级排序规则
        sorted_rules = sorted(self.inference_rules.values(), 
                            key=lambda r: (r.priority, r.confidence), 
                            reverse=True)
        
        activated_rules = []
        
        for rule in sorted_rules:
            try:
                # 检查条件是否满足（简化版）
                condition_met = self._evaluate_condition(rule.condition, context)
                
                if condition_met:
                    # 执行动作
                    self._execute_action(rule.action, context)
                    
                    # 更新规则统计
                    rule.activation_count += 1
                    rule.last_activated = time.time()
                    activated_rules.append(rule.rule_id)
                    
                    logger.debug(f"推理规则激活: {rule.rule_id}")
                    
            except Exception as e:
                logger.error(f"执行推理规则失败 {rule.rule_id}: {str(e)}")
        
        if activated_rules:
            logger.info(f"触发了 {len(activated_rules)} 个推理规则: {activated_rules}")
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """评估条件（简化版）"""
        # 在实际应用中，这里会有一个完整的条件评估引擎
        # 这里使用简化的关键字匹配
        
        entity = context.get("entity")
        if not entity:
            return False
        
        # 检查相似实体条件
        if "entities_similarity" in condition:
            # 寻找相似实体
            similar_entities = self._find_similar_entities(entity)
            if similar_entities:
                context["similar_entities"] = similar_entities
                context["entities_similarity"] = 0.8  # 示例相似度
                return True
        
        # 检查知识置信度条件
        if "knowledge_confidence" in condition:
            avg_confidence = self.quality_metrics.get(entity.entity_id, {}).get("avg_confidence", 0.0)
            context["knowledge_confidence"] = avg_confidence
            
            if "knowledge_confidence < 0.5" in condition:
                return avg_confidence < 0.5
        
        # 更多条件评估...
        
        return False
    
    def _find_similar_entities(self, entity: KnowledgeEntity, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """寻找相似实体"""
        similar_entities = []
        
        for other_id, other_entity in self.knowledge_graph.entities.items():
            if other_id == entity.entity_id:
                continue
            
            # 计算相似度（简化版）
            similarity = self._calculate_entity_similarity(entity, other_entity)
            
            if similarity > threshold:
                similar_entities.append({
                    "entity_id": other_id,
                    "entity": other_entity,
                    "similarity": similarity
                })
        
        return similar_entities
    
    def _calculate_entity_similarity(self, entity1: KnowledgeEntity, entity2: KnowledgeEntity) -> float:
        """计算实体相似度"""
        similarity = 0.0
        
        # 名称相似度
        if entity1.name and entity2.name:
            name_sim = self._calculate_text_similarity(entity1.name, entity2.name)
            similarity += name_sim * 0.3
        
        # 类型相似度
        if entity1.entity_type == entity2.entity_type:
            similarity += 0.2
        
        # 属性相似度
        attr_sim = self._calculate_attributes_similarity(entity1.attributes, entity2.attributes)
        similarity += attr_sim * 0.5
        
        return min(1.0, similarity)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        if not text1 or not text2:
            return 0.0
        
        # 简单实现：Jaccard相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_attributes_similarity(self, attrs1: Dict[str, Any], attrs2: Dict[str, Any]) -> float:
        """计算属性相似度"""
        if not attrs1 or not attrs2:
            return 0.0
        
        common_keys = set(attrs1.keys()).intersection(set(attrs2.keys()))
        total_keys = set(attrs1.keys()).union(set(attrs2.keys()))
        
        if not total_keys:
            return 0.0
        
        similarity = 0.0
        for key in common_keys:
            val1 = attrs1[key]
            val2 = attrs2[key]
            
            if val1 == val2:
                similarity += 1.0
            elif isinstance(val1, str) and isinstance(val2, str):
                similarity += self._calculate_text_similarity(val1, val2)
        
        return similarity / len(total_keys)
    
    def _execute_action(self, action: str, context: Dict[str, Any]):
        """执行动作"""
        entity = context.get("entity")
        if not entity:
            return
        
        if action == "create_cross_domain_relation":
            # 创建跨领域关系
            similar_entities = context.get("similar_entities", [])
            if similar_entities:
                # 与最相似的实体创建关系
                best_match = max(similar_entities, key=lambda x: x["similarity"])
                
                relation_id = f"inferred_relation_{len(self.knowledge_graph.relations)}"
                relation = EnhancedRelation(
                    relation_id=relation_id,
                    source_entity=entity.entity_id,
                    target_entity=best_match["entity_id"],
                    relation_type=RelationType.CROSS_DOMAIN,
                    confidence=context.get("entities_similarity", 0.7),
                    strength=0.6,
                    domains=list(set(entity.domains).union(best_match["entity"].domains)),
                    evidence=["推理规则生成"],
                    metadata={
                        "inference_rule": "create_cross_domain_relation",
                        "similarity_score": best_match["similarity"]
                    }
                )
                
                # 添加到知识图谱
                self.knowledge_graph.relations[relation_id] = relation
                self.knowledge_graph.entity_relations[entity.entity_id].append(relation_id)
                self.knowledge_graph.entity_relations[best_match["entity_id"]].append(relation_id)
                
                key = (entity.entity_id, best_match["entity_id"], RelationType.CROSS_DOMAIN.value)
                self.knowledge_graph.relation_index[key] = relation_id
        
        elif action == "update_knowledge_from_source":
            # 从高置信度源更新知识
            # 在实际应用中，这里会查询外部知识源
            pass
        
        elif action == "resolve_conflict_by_confidence":
            # 按置信度解决冲突
            # 已经在冲突检测中处理
            pass
        
        elif action == "create_domain_bridge":
            # 创建领域桥梁
            # 分析实体所属领域并创建跨领域关联
            if len(entity.domains) >= 2:
                # 创建跨领域实体标记
                entity.attributes["cross_domain_bridge"] = True
                entity.attributes["bridge_domains"] = [d.value for d in entity.domains]
    
    def query_knowledge(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """查询知识"""
        results = []
        
        # 简单查询：在实体名称和属性中搜索
        query_lower = query.lower()
        
        for entity_id, entity in self.knowledge_graph.entities.items():
            relevance = 0.0
            
            # 检查名称匹配
            if query_lower in entity.name.lower():
                relevance += 0.7
            
            # 检查属性匹配
            for key, value in entity.attributes.items():
                if isinstance(value, str) and query_lower in value.lower():
                    relevance += 0.3
                elif isinstance(value, (int, float)) and query_lower in str(value):
                    relevance += 0.2
            
            if relevance > 0:
                # 获取相关关系
                relations = []
                relation_ids = self.knowledge_graph.entity_relations.get(entity_id, [])
                for rel_id in relation_ids[:5]:  # 限制前5个关系
                    relation = self.knowledge_graph.relations[rel_id]
                    relations.append({
                        "id": relation.relation_id,
                        "type": relation.relation_type.value,
                        "target": relation.target_entity if relation.source_entity == entity_id else relation.source_entity,
                        "strength": relation.strength,
                        "confidence": relation.confidence
                    })
                
                results.append({
                    "entity_id": entity_id,
                    "name": entity.name,
                    "type": entity.entity_type,
                    "domains": [d.value for d in entity.domains],
                    "attributes": entity.attributes,
                    "relevance": relevance,
                    "relations": relations,
                    "importance_score": entity.importance_score,
                    "centrality_score": entity.centrality_score
                })
        
        # 按相关性排序
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        return results[:max_results]
    
    def infer_relationships(self, entity1_id: str, entity2_id: str, 
                          max_path_length: int = 3) -> List[List[str]]:
        """推理实体间关系"""
        return self.knowledge_graph.infer_indirect_relations(
            entity1_id, entity2_id, max_path_length
        )
    
    def discover_new_knowledge(self) -> Dict[str, Any]:
        """发现新知识"""
        discoveries = {
            "cross_domain_relations": [],
            "new_entities": [],
            "conflicts_resolved": 0,
            "quality_improvements": 0
        }
        
        # 发现跨领域关系
        if self.auto_update_config["enable_auto_discovery"]:
            new_relations = self.knowledge_graph.discover_cross_domain_relations()
            discoveries["cross_domain_relations"] = [
                {
                    "id": rel.relation_id,
                    "source": rel.source_entity,
                    "target": rel.target_entity,
                    "confidence": rel.confidence,
                    "strength": rel.strength
                }
                for rel in new_relations
            ]
        
        # 发现新实体（通过关系推断）
        # 这里可以扩展更多发现逻辑
        
        # 自动解决冲突
        if self.auto_update_config["enable_auto_correction"]:
            resolved = self._auto_resolve_all_conflicts()
            discoveries["conflicts_resolved"] = resolved
        
        # 知识质量改进
        if self.auto_update_config["enable_auto_optimization"]:
            improvements = self._optimize_knowledge_quality()
            discoveries["quality_improvements"] = improvements
        
        logger.info(f"知识发现完成: {len(discoveries['cross_domain_relations'])} 个新关系, "
                   f"{discoveries['conflicts_resolved']} 个冲突已解决")
        
        return discoveries
    
    def _auto_resolve_all_conflicts(self) -> int:
        """自动解决所有冲突"""
        resolved_count = 0
        
        for conflict_id, conflict in list(self.active_conflicts.items()):
            if not conflict.resolved:
                resolved = self._auto_resolve_conflict(conflict)
                if resolved:
                    conflict.resolved = True
                    self.resolved_conflicts.append(conflict)
                    del self.active_conflicts[conflict_id]
                    resolved_count += 1
        
        return resolved_count
    
    def _optimize_knowledge_quality(self) -> int:
        """优化知识质量"""
        improvements = 0
        
        # 检查低置信度知识
        for entity_id, metrics in self.quality_metrics.items():
            if metrics["avg_confidence"] < self.auto_update_config["min_confidence_threshold"]:
                # 尝试从其他源获取更高置信度的知识
                # 这里可以扩展实际的质量优化逻辑
                improvements += 1
        
        return improvements
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 更新实体中心性
                for entity_id in list(self.knowledge_graph.entities.keys())[:100]:  # 每次循环处理100个
                    self.knowledge_graph.calculate_entity_centrality(entity_id)
                
                # 清理旧缓存
                self._cleanup_old_cache()
                
                # 记录监控日志
                self._log_monitoring_stats()
                
                # 每10分钟运行一次
                time.sleep(600)
                
            except Exception as e:
                logger.error(f"监控循环出错: {str(e)}")
                time.sleep(60)
    
    def _cleanup_old_cache(self):
        """清理旧缓存"""
        current_time = time.time()
        old_keys = []
        
        for key, (timestamp, _) in self.inference_cache.items():
            if current_time - timestamp > 3600:  # 1小时过期
                old_keys.append(key)
        
        for key in old_keys:
            del self.inference_cache[key]
    
    def _log_monitoring_stats(self):
        """记录监控统计"""
        stats = self.get_engine_statistics()
        
        logger.info(f"知识引擎统计: {stats['entity_count']} 实体, "
                   f"{stats['relation_count']} 关系, "
                   f"{stats['active_conflicts']} 个活跃冲突, "
                   f"更新成功率: {stats['update_success_rate']:.1%}")
    
    def _auto_update_loop(self):
        """自动更新循环"""
        while self.running:
            try:
                # 等待下一个更新周期
                time.sleep(self.auto_update_config["update_check_interval"])
                
                # 执行知识发现
                discoveries = self.discover_new_knowledge()
                
                if discoveries["cross_domain_relations"] or discoveries["conflicts_resolved"] > 0:
                    logger.info(f"自动知识发现完成: {discoveries}")
                
            except Exception as e:
                logger.error(f"自动更新循环出错: {str(e)}")
                time.sleep(60)
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        graph_stats = self.knowledge_graph.get_graph_statistics()
        
        return {
            "entity_count": len(self.knowledge_graph.entities),
            "relation_count": len(self.knowledge_graph.relations),
            "update_count": len(self.update_history),
            "active_conflicts": len(self.active_conflicts),
            "resolved_conflicts": len(self.resolved_conflicts),
            "inference_rules": len(self.inference_rules),
            "update_success_rate": self._calculate_update_success_rate(),
            "average_confidence": self._calculate_average_confidence(),
            "knowledge_graph_stats": graph_stats,
            "timestamp": time.time()
        }
    
    def _calculate_update_success_rate(self) -> float:
        """计算更新成功率"""
        total_updates = 0
        successful_updates = 0
        
        for metrics in self.quality_metrics.values():
            total_updates += metrics["total_updates"]
            successful_updates += metrics["successful_updates"]
        
        if total_updates > 0:
            return successful_updates / total_updates
        return 0.0
    
    def _calculate_average_confidence(self) -> float:
        """计算平均置信度"""
        total_confidence = 0.0
        count = 0
        
        for metrics in self.quality_metrics.values():
            if metrics["successful_updates"] > 0:
                total_confidence += metrics["avg_confidence"]
                count += 1
        
        if count > 0:
            return total_confidence / count
        return 0.0
    
    def export_knowledge(self, format: str = "json") -> str:
        """导出知识"""
        if format == "json":
            export_data = {
                "metadata": {
                    "export_time": time.time(),
                    "entity_count": len(self.knowledge_graph.entities),
                    "relation_count": len(self.knowledge_graph.relations),
                    "engine_version": "1.0.0"
                },
                "entities": {},
                "relations": {},
                "statistics": self.get_engine_statistics()
            }
            
            # 导出实体
            for entity_id, entity in self.knowledge_graph.entities.items():
                export_data["entities"][entity_id] = {
                    "name": entity.name,
                    "type": entity.entity_type,
                    "domains": [d.value for d in entity.domains],
                    "attributes": entity.attributes,
                    "importance_score": entity.importance_score,
                    "centrality_score": entity.centrality_score,
                    "created_at": entity.created_at,
                    "updated_at": entity.updated_at
                }
            
            # 导出关系
            for relation_id, relation in self.knowledge_graph.relations.items():
                export_data["relations"][relation_id] = {
                    "source": relation.source_entity,
                    "target": relation.target_entity,
                    "type": relation.relation_type.value,
                    "confidence": relation.confidence,
                    "strength": relation.strength,
                    "domains": [d.value for d in relation.domains],
                    "created_at": relation.created_at,
                    "updated_at": relation.updated_at,
                    "metadata": relation.metadata
                }
            
            return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
        
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def shutdown(self):
        """关闭引擎"""
        self.running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        if hasattr(self, 'auto_update_thread'):
            self.auto_update_thread.join(timeout=5)
        
        logger.info("动态知识引擎已关闭")


# 全局实例
_knowledge_engine_instance = None

def get_knowledge_engine(existing_knowledge_path: Optional[str] = None) -> DynamicKnowledgeEngine:
    """获取知识引擎单例"""
    global _knowledge_engine_instance
    if _knowledge_engine_instance is None:
        _knowledge_engine_instance = DynamicKnowledgeEngine(existing_knowledge_path)
    return _knowledge_engine_instance

def initialize_knowledge_engine(existing_knowledge_path: Optional[str] = None):
    """初始化知识引擎"""
    global _knowledge_engine_instance
    if _knowledge_engine_instance is not None:
        _knowledge_engine_instance.shutdown()
    
    _knowledge_engine_instance = DynamicKnowledgeEngine(existing_knowledge_path)
    return _knowledge_engine_instance

def shutdown_knowledge_engine():
    """关闭知识引擎"""
    global _knowledge_engine_instance
    if _knowledge_engine_instance is not None:
        _knowledge_engine_instance.shutdown()
        _knowledge_engine_instance = None
