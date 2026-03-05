"""
Cross-Model Knowledge Graph Fusion

Enhanced knowledge graph fusion across different models to create unified
knowledge representations and enable cross-model reasoning and inference.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
import hashlib
import json
import time

logger = logging.getLogger(__name__)

from .enhanced_knowledge_graph import EnhancedKnowledgeGraph, Entity, RelationType, Domain, KnowledgeType

class CrossModelKnowledgeFusion:
    """跨模型知识图谱融合器"""
    
    def __init__(self):
        """初始化跨模型知识图谱融合器"""
        self.fusion_graph = EnhancedKnowledgeGraph()
        self.model_graphs: Dict[str, EnhancedKnowledgeGraph] = {}
        self.fusion_history: List[Dict[str, Any]] = []
        self.conflict_resolution_strategies = {
            'timestamp_newer': self._resolve_by_timestamp,
            'confidence_higher': self._resolve_by_confidence,
            'source_reliability': self._resolve_by_source_reliability,
            'majority_vote': self._resolve_by_majority_vote,
            'semantic_consistency': self._resolve_by_semantic_consistency
        }
        self.source_reliability_scores: Dict[str, float] = {}
        
        # 融合配置
        self.fusion_config = {
            'min_confidence_threshold': 0.6,
            'max_semantic_distance': 0.7,
            'enable_cross_domain_fusion': True,
            'enable_conflict_detection': True,
            'enable_consistency_check': True,
            'max_fusion_iterations': 3
        }
        
        logger.info("Cross-model knowledge graph fusion system initialized")
    
    def register_model_graph(self, model_id: str, knowledge_graph: EnhancedKnowledgeGraph,
                           reliability_score: float = 0.8):
        """
        注册模型知识图谱
        
        Args:
            model_id: 模型标识符
            knowledge_graph: 模型的知识图谱
            reliability_score: 模型可靠性分数 (0-1)
        """
        self.model_graphs[model_id] = knowledge_graph
        self.source_reliability_scores[model_id] = reliability_score
        logger.info(f"Model graph registered: {model_id}, reliability: {reliability_score}")
    
    def fuse_all_models(self, strategy: str = 'confidence_higher') -> Dict[str, Any]:
        """
        融合所有已注册的模型知识图谱
        
        Args:
            strategy: 冲突解决策略
            
        Returns:
            融合结果字典
        """
        logger.info(f"Starting fusion of {len(self.model_graphs)} model knowledge graphs")
        
        if not self.model_graphs:
            return {'success': False, 'error': 'No model graphs registered'}
        
        try:
            # 重置融合图谱
            self.fusion_graph = EnhancedKnowledgeGraph()
            fusion_stats = {
                'total_entities_fused': 0,
                'total_relations_fused': 0,
                'conflicts_resolved': 0,
                'cross_domain_relations_created': 0,
                'fusion_start_time': time.time()
            }
            
            # 第一步：融合所有实体
            entity_fusion_results = self._fuse_entities(strategy)
            fusion_stats['total_entities_fused'] = entity_fusion_results['entities_fused']
            
            # 第二步：融合所有关系
            relation_fusion_results = self._fuse_relations(strategy)
            fusion_stats['total_relations_fused'] = relation_fusion_results['relations_fused']
            fusion_stats['conflicts_resolved'] = relation_fusion_results['conflicts_resolved']
            
            # 第三步：发现和创建跨模型关系
            if self.fusion_config['enable_cross_domain_fusion']:
                cross_model_relations = self._discover_cross_model_relations()
                fusion_stats['cross_domain_relations_created'] = len(cross_model_relations)
            
            # 第四步：一致性检查
            if self.fusion_config['enable_consistency_check']:
                consistency_check = self._check_fusion_consistency()
                fusion_stats['consistency_score'] = consistency_check['consistency_score']
                fusion_stats['inconsistencies_found'] = consistency_check['inconsistencies_found']
            
            fusion_stats['fusion_end_time'] = time.time()
            fusion_stats['fusion_duration'] = fusion_stats['fusion_end_time'] - fusion_stats['fusion_start_time']
            
            # 记录融合历史
            fusion_record = {
                'timestamp': time.time(),
                'models_fused': list(self.model_graphs.keys()),
                'strategy_used': strategy,
                'stats': fusion_stats,
                'fusion_graph_size': {
                    'entities': len(self.fusion_graph.entities),
                    'relations': len(self.fusion_graph.relations)
                }
            }
            self.fusion_history.append(fusion_record)
            
            logger.info(f"Cross-model fusion completed. Stats: {fusion_stats}")
            
            return {
                'success': True,
                'stats': fusion_stats,
                'fusion_record': fusion_record,
                'fusion_graph': self.fusion_graph
            }
            
        except Exception as e:
            logger.error(f"Cross-model fusion failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _fuse_entities(self, strategy: str) -> Dict[str, Any]:
        """融合实体"""
        entities_fused = 0
        entity_conflicts = 0
        
        # 收集所有模型的所有实体
        all_entities: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for model_id, graph in self.model_graphs.items():
            for entity_id, entity in graph.entities.items():
                entity_data = {
                    'entity': entity,
                    'model_id': model_id,
                    'timestamp': getattr(entity, 'creation_time', time.time()),
                    'confidence': getattr(entity, 'confidence', 0.8),
                    'source_reliability': self.source_reliability_scores.get(model_id, 0.5)
                }
                all_entities[entity_id].append(entity_data)
        
        # 融合每个实体
        for entity_id, entity_versions in all_entities.items():
            if len(entity_versions) == 1:
                # 单一版本，直接添加
                entity_data = entity_versions[0]
                self._add_entity_to_fusion_graph(entity_data['entity'])
                entities_fused += 1
            else:
                # 多版本，需要融合
                fused_entity = self._fuse_entity_versions(entity_id, entity_versions, strategy)
                if fused_entity:
                    self.fusion_graph.add_entity(
                        entity_id=fused_entity.entity_id,
                        name=fused_entity.name,
                        entity_type=fused_entity.entity_type,
                        domains=fused_entity.domains,
                        knowledge_type=fused_entity.knowledge_type,
                        attributes=fused_entity.attributes
                    )
                    entities_fused += 1
                    entity_conflicts += 1
        
        return {
            'entities_fused': entities_fused,
            'entity_conflicts': entity_conflicts
        }
    
    def _fuse_entity_versions(self, entity_id: str, 
                            entity_versions: List[Dict[str, Any]],
                            strategy: str) -> Optional[Entity]:
        """融合实体多个版本"""
        if not entity_versions:
            return None
        
        # 选择最佳版本
        if strategy in self.conflict_resolution_strategies:
            resolution_func = self.conflict_resolution_strategies[strategy]
            selected_version = resolution_func(entity_versions, 'entity')
        else:
            # 默认使用置信度最高
            selected_version = max(entity_versions, 
                                 key=lambda x: x['confidence'] * x['source_reliability'])
        
        base_entity = selected_version['entity']
        
        # 创建融合实体
        fused_entity = Entity(
            entity_id=entity_id,
            name=base_entity.name,
            entity_type=base_entity.entity_type,
            domains=set(base_entity.domains),
            knowledge_type=base_entity.knowledge_type,
            attributes=dict(base_entity.attributes)
        )
        
        # 合并属性
        for version in entity_versions:
            if version is not selected_version:
                # 合并领域
                fused_entity.domains.update(version['entity'].domains)
                
                # 合并属性
                for key, value in version['entity'].attributes.items():
                    if key not in fused_entity.attributes:
                        fused_entity.attributes[key] = value
                    elif isinstance(value, list) and isinstance(fused_entity.attributes[key], list):
                        # 合并列表
                        fused_entity.attributes[key].extend(value)
                        # 去重
                        if fused_entity.attributes[key] and isinstance(fused_entity.attributes[key][0], str):
                            fused_entity.attributes[key] = list(set(fused_entity.attributes[key]))
        
        # 添加融合元数据
        fused_entity.attributes['fused_from_models'] = [v['model_id'] for v in entity_versions]
        fused_entity.attributes['fusion_timestamp'] = time.time()
        fused_entity.attributes['fusion_strategy'] = strategy
        
        return fused_entity
    
    def _fuse_relations(self, strategy: str) -> Dict[str, Any]:
        """融合关系"""
        relations_fused = 0
        conflicts_resolved = 0
        
        # 收集所有模型的所有关系
        all_relations: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        
        for model_id, graph in self.model_graphs.items():
            for relation in graph.relations.values():
                relation_key = (relation.source_entity, relation.target_entity, 
                              relation.relation_type.value)
                relation_data = {
                    'relation': relation,
                    'model_id': model_id,
                    'timestamp': getattr(relation, 'creation_time', time.time()),
                    'confidence': relation.confidence,
                    'strength': relation.strength,
                    'source_reliability': self.source_reliability_scores.get(model_id, 0.5)
                }
                all_relations[relation_key].append(relation_data)
        
        # 融合每个关系
        for relation_key, relation_versions in all_relations.items():
            source_entity, target_entity, relation_type = relation_key
            
            # 检查实体是否存在于融合图谱中
            if source_entity not in self.fusion_graph.entities:
                continue
            if target_entity not in self.fusion_graph.entities:
                continue
            
            if len(relation_versions) == 1:
                # 单一版本，直接添加
                relation_data = relation_versions[0]
                relation = relation_data['relation']
                self.fusion_graph.add_relation(
                    source_entity=relation.source_entity,
                    target_entity=relation.target_entity,
                    relation_type=relation.relation_type,
                    strength=relation.strength,
                    confidence=relation.confidence
                )
                relations_fused += 1
            else:
                # 多版本，需要融合
                fused_relation = self._fuse_relation_versions(relation_versions, strategy)
                if fused_relation:
                    self.fusion_graph.add_relation(
                        source_entity=source_entity,
                        target_entity=target_entity,
                        relation_type=fused_relation.relation_type,
                        strength=fused_relation.strength,
                        confidence=fused_relation.confidence
                    )
                    relations_fused += 1
                    conflicts_resolved += 1
        
        return {
            'relations_fused': relations_fused,
            'conflicts_resolved': conflicts_resolved
        }
    
    def _fuse_relation_versions(self, relation_versions: List[Dict[str, Any]],
                              strategy: str):
        """融合关系多个版本"""
        if not relation_versions:
            return None
        
        # 选择最佳版本
        if strategy in self.conflict_resolution_strategies:
            resolution_func = self.conflict_resolution_strategies[strategy]
            selected_version = resolution_func(relation_versions, 'relation')
        else:
            # 默认使用置信度最高
            selected_version = max(relation_versions,
                                 key=lambda x: x['confidence'] * x['source_reliability'])
        
        base_relation = selected_version['relation']
        
        # 计算平均强度和置信度
        avg_strength = np.mean([v['strength'] for v in relation_versions])
        avg_confidence = np.mean([v['confidence'] for v in relation_versions])
        
        # 创建融合关系
        fused_relation = type(base_relation)(
            relation_id=f"fused_{hashlib.md5(str(relation_versions).encode()).hexdigest()[:8]}",
            source_entity=base_relation.source_entity,
            target_entity=base_relation.target_entity,
            relation_type=base_relation.relation_type,
            strength=avg_strength,
            confidence=avg_confidence
        )
        
        # 添加融合元数据
        fused_relation.metadata = {
            'fused_from_models': [v['model_id'] for v in relation_versions],
            'fusion_timestamp': time.time(),
            'fusion_strategy': strategy,
            'original_confidences': [v['confidence'] for v in relation_versions],
            'original_strengths': [v['strength'] for v in relation_versions]
        }
        
        return fused_relation
    
    def _discover_cross_model_relations(self) -> List[Dict[str, Any]]:
        """发现跨模型关系"""
        discovered_relations = []
        
        # 获取所有实体的领域信息
        entity_domains: Dict[str, Set[Domain]] = {}
        for entity_id, entity in self.fusion_graph.entities.items():
            entity_domains[entity_id] = set(entity.domains)
        
        # 寻找跨领域实体对
        entity_ids = list(self.fusion_graph.entities.keys())
        
        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                entity1_id = entity_ids[i]
                entity2_id = entity_ids[j]
                
                # 检查是否已存在关系
                if self.fusion_graph._relation_exists(entity1_id, entity2_id):
                    continue
                
                # 检查领域差异
                domains1 = entity_domains[entity1_id]
                domains2 = entity_domains[entity2_id]
                
                if not domains1.isdisjoint(domains2):
                    continue  # 有重叠领域，可能不是跨模型
                
                # 计算语义相似度
                entity1 = self.fusion_graph.entities[entity1_id]
                entity2 = self.fusion_graph.entities[entity2_id]
                similarity = self._calculate_entity_similarity(entity1, entity2)
                
                if similarity > self.fusion_config['max_semantic_distance']:
                    # 创建跨模型关系
                    relation_type = RelationType.CROSS_DOMAIN
                    strength = similarity
                    confidence = 0.7 * similarity
                    
                    self.fusion_graph.add_relation(
                        source_entity=entity1_id,
                        target_entity=entity2_id,
                        relation_type=relation_type,
                        strength=strength,
                        confidence=confidence
                    )
                    
                    discovered_relations.append({
                        'source_entity': entity1_id,
                        'target_entity': entity2_id,
                        'relation_type': relation_type.value,
                        'strength': strength,
                        'confidence': confidence,
                        'domain1': [d.value for d in domains1],
                        'domain2': [d.value for d in domains2],
                        'similarity': similarity
                    })
        
        logger.info(f"Discovered {len(discovered_relations)} cross-model relations")
        return discovered_relations
    
    def _calculate_entity_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """计算实体相似度"""
        similarity = 0.0
        
        # 名称相似度
        if entity1.name and entity2.name:
            name_similarity = self._calculate_text_similarity(entity1.name, entity2.name)
            similarity += name_similarity * 0.3
        
        # 属性相似度
        attr_similarity = self._calculate_attributes_similarity(entity1.attributes, entity2.attributes)
        similarity += attr_similarity * 0.4
        
        # 领域重叠度
        domain_overlap = len(entity1.domains.intersection(entity2.domains))
        domain_total = len(entity1.domains.union(entity2.domains))
        if domain_total > 0:
            domain_similarity = domain_overlap / domain_total
            similarity += domain_similarity * 0.3
        
        return similarity
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简化版）"""
        if not text1 or not text2:
            return 0.0
        
        # 转换为小写
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # 简单单词重叠
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_attributes_similarity(self, attrs1: Dict[str, Any], 
                                       attrs2: Dict[str, Any]) -> float:
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
            
            if isinstance(val1, str) and isinstance(val2, str):
                key_similarity = self._calculate_text_similarity(val1, val2)
            elif isinstance(val1, list) and isinstance(val2, list):
                # 列表相似度
                set1 = set(str(item) for item in val1)
                set2 = set(str(item) for item in val2)
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                key_similarity = intersection / union if union > 0 else 0.0
            else:
                # 其他类型，检查相等性
                key_similarity = 1.0 if val1 == val2 else 0.0
            
            similarity += key_similarity
        
        return similarity / len(total_keys) if total_keys else 0.0
    
    def _check_fusion_consistency(self) -> Dict[str, Any]:
        """检查融合一致性"""
        inconsistencies = []
        
        # 检查循环关系
        try:
            # 这里可以实现循环检测
            pass
        except Exception as e:
            inconsistencies.append({'type': 'circular_relation', 'error': str(e)})
        
        # 检查关系完整性
        for relation in self.fusion_graph.relations.values():
            if relation.source_entity not in self.fusion_graph.entities:
                inconsistencies.append({
                    'type': 'missing_source_entity',
                    'relation_id': relation.relation_id,
                    'source_entity': relation.source_entity
                })
            if relation.target_entity not in self.fusion_graph.entities:
                inconsistencies.append({
                    'type': 'missing_target_entity',
                    'relation_id': relation.relation_id,
                    'target_entity': relation.target_entity
                })
        
        # 计算一致性分数
        total_checks = max(1, len(self.fusion_graph.relations))
        consistency_score = 1.0 - (len(inconsistencies) / total_checks)
        
        return {
            'consistency_score': consistency_score,
            'inconsistencies_found': len(inconsistencies),
            'inconsistencies': inconsistencies
        }
    
    def _add_entity_to_fusion_graph(self, entity: Entity):
        """添加实体到融合图谱"""
        self.fusion_graph.add_entity(
            entity_id=entity.entity_id,
            name=entity.name,
            entity_type=entity.entity_type,
            domains=entity.domains,
            knowledge_type=entity.knowledge_type,
            attributes=entity.attributes
        )
    
    # 冲突解决策略
    def _resolve_by_timestamp(self, versions: List[Dict[str, Any]], 
                            item_type: str) -> Dict[str, Any]:
        """按时间戳解决冲突（越新越好）"""
        return max(versions, key=lambda x: x['timestamp'])
    
    def _resolve_by_confidence(self, versions: List[Dict[str, Any]], 
                             item_type: str) -> Dict[str, Any]:
        """按置信度解决冲突（置信度越高越好）"""
        return max(versions, key=lambda x: x['confidence'])
    
    def _resolve_by_source_reliability(self, versions: List[Dict[str, Any]], 
                                     item_type: str) -> Dict[str, Any]:
        """按源可靠性解决冲突"""
        return max(versions, key=lambda x: x['source_reliability'])
    
    def _resolve_by_majority_vote(self, versions: List[Dict[str, Any]], 
                                item_type: str) -> Dict[str, Any]:
        """按多数投票解决冲突"""
        # 简化版：选择最常见的值
        if item_type == 'entity':
            # 按名称分组
            name_counts = defaultdict(int)
            for version in versions:
                name = version['entity'].name
                name_counts[name] += 1
            
            most_common_name = max(name_counts.items(), key=lambda x: x[1])[0]
            for version in versions:
                if version['entity'].name == most_common_name:
                    return version
        
        return versions[0]  # 默认返回第一个
    
    def _resolve_by_semantic_consistency(self, versions: List[Dict[str, Any]], 
                                       item_type: str) -> Dict[str, Any]:
        """按语义一致性解决冲突"""
        # 简化版：计算与所有版本的平均相似度，选择最相似的
        if len(versions) <= 1:
            return versions[0] if versions else None
        
        best_version = None
        best_avg_similarity = -1
        
        for i, version_i in enumerate(versions):
            total_similarity = 0
            count = 0
            
            for j, version_j in enumerate(versions):
                if i != j:
                    if item_type == 'entity':
                        similarity = self._calculate_entity_similarity(
                            version_i['entity'], version_j['entity']
                        )
                    else:  # relation
                        similarity = abs(version_i['strength'] - version_j['strength'])
                        similarity = 1.0 - min(similarity, 1.0)
                    
                    total_similarity += similarity
                    count += 1
            
            avg_similarity = total_similarity / count if count > 0 else 0
            
            if avg_similarity > best_avg_similarity:
                best_avg_similarity = avg_similarity
                best_version = version_i
        
        return best_version
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """获取融合统计信息"""
        if not self.fusion_history:
            return {'error': 'No fusion history available'}
        
        latest_fusion = self.fusion_history[-1]
        
        stats = {
            'total_fusions': len(self.fusion_history),
            'latest_fusion': latest_fusion['timestamp'],
            'models_in_latest_fusion': latest_fusion['models_fused'],
            'fusion_graph_size': latest_fusion['fusion_graph_size'],
            'fusion_stats': latest_fusion.get('stats', {})
        }
        
        return stats
    
    def export_fusion_graph(self, format: str = 'json') -> str:
        """导出融合图谱"""
        if format == 'json':
            graph_data = {
                'entities': {},
                'relations': {}
            }
            
            for entity_id, entity in self.fusion_graph.entities.items():
                graph_data['entities'][entity_id] = {
                    'name': entity.name,
                    'type': entity.entity_type,
                    'domains': [d.value for d in entity.domains],
                    'knowledge_type': entity.knowledge_type.value,
                    'attributes': entity.attributes
                }
            
            for relation_id, relation in self.fusion_graph.relations.items():
                graph_data['relations'][relation_id] = {
                    'source': relation.source_entity,
                    'target': relation.target_entity,
                    'type': relation.relation_type.value,
                    'strength': relation.strength,
                    'confidence': relation.confidence,
                    'metadata': getattr(relation, 'metadata', {})
                }
            
            return json.dumps(graph_data, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")