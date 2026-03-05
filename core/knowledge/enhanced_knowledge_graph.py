"""
增强知识图谱 - Enhanced Knowledge Graph

增强现有知识图谱功能，提供跨领域知识关联能力：
1. 跨领域关系发现和链接
2. 知识融合和冲突解决
3. 动态知识更新和演化
4. 关联强度计算和优化
5. 知识推理和查询增强

设计目标：
- 建立跨领域知识关联网络
- 支持动态知识演化
- 提供智能知识推理
- 实现多源知识融合
- 优化知识查询性能
"""

import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
import random
import math

logger = logging.getLogger(__name__)

class RelationType(Enum):
    """关系类型扩展"""
    IS_A = "is_a"                    # 属于关系
    PART_OF = "part_of"              # 部分关系
    HAS_PROPERTY = "has_property"    # 具有属性
    LOCATED_IN = "located_in"        # 位于
    CREATED_BY = "created_by"        # 由...创建
    USED_BY = "used_by"              # 被...使用
    SIMILAR_TO = "similar_to"        # 相似于
    OPPOSITE_OF = "opposite_of"      # 相反于
    CAUSES = "causes"                # 导致
    PRECEDES = "precedes"            # 先于
    INFLUENCES = "influences"        # 影响
    ASSOCIATED_WITH = "associated_with"  # 关联
    DEPENDS_ON = "depends_on"        # 依赖于
    EVOLVED_FROM = "evolved_from"    # 演化自
    CROSS_DOMAIN = "cross_domain"    # 跨领域关联

class Domain(Enum):
    """知识领域"""
    SCIENCE = "science"              # 科学
    TECHNOLOGY = "technology"        # 技术
    ART = "art"                      # 艺术
    HISTORY = "history"              # 历史
    PHILOSOPHY = "philosophy"        # 哲学
    MATHEMATICS = "mathematics"      # 数学
    LITERATURE = "literature"        # 文学
    MEDICINE = "medicine"            # 医学
    BUSINESS = "business"            # 商业
    SPORTS = "sports"                # 体育
    MUSIC = "music"                  # 音乐
    LAW = "law"                      # 法律
    PSYCHOLOGY = "psychology"        # 心理学
    SOCIOLOGY = "sociology"          # 社会学
    ECONOMICS = "economics"          # 经济学


class KnowledgeType(Enum):
    """知识类型"""
    FACT = "fact"                    # 事实
    CONCEPT = "concept"              # 概念
    PRINCIPLE = "principle"          # 原理
    PROCEDURE = "procedure"          # 过程
    METAKNOWLEDGE = "metaknowledge"  # 元知识
    CROSS_DOMAIN = "cross_domain"    # 跨领域知识

@dataclass
class EnhancedRelation:
    """增强关系"""
    relation_id: str
    source_entity: str
    target_entity: str
    relation_type: RelationType
    confidence: float = 1.0
    strength: float = 1.0              # 关联强度（0-1）
    domains: List[Domain] = field(default_factory=list)  # 涉及领域
    evidence: List[str] = field(default_factory=list)    # 证据
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeEntity:
    """知识实体增强"""
    entity_id: str
    name: str
    entity_type: str
    domains: List[Domain] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    centrality_score: float = 0.0      # 中心性分数
    importance_score: float = 0.0      # 重要性分数
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

@dataclass
class CrossDomainAssociation:
    """跨领域关联"""
    association_id: str
    source_domain: Domain
    target_domain: Domain
    bridging_entities: List[str]       # 桥接实体
    association_strength: float = 0.0
    relation_types: List[RelationType] = field(default_factory=list)
    evidence_count: int = 0
    created_at: float = field(default_factory=time.time)

class EnhancedKnowledgeGraph:
    """增强知识图谱"""
    
    def __init__(self, existing_graph=None):
        """初始化增强知识图谱"""
        self.entities: Dict[str, KnowledgeEntity] = {}
        self.relations: Dict[str, EnhancedRelation] = {}
        self.entity_relations: Dict[str, List[str]] = defaultdict(list)
        self.domain_entities: Dict[Domain, Set[str]] = defaultdict(set)
        self.cross_domain_associations: Dict[str, CrossDomainAssociation] = {}
        
        # 索引
        self.entity_name_index: Dict[str, str] = {}  # 名称->ID映射
        self.relation_index: Dict[Tuple[str, str, str], str] = {}  # (源,目标,类型)->关系ID
        
        # 性能优化
        self.cache_enabled = True
        self.cache: Dict[str, Any] = {}
        
        # 如果提供了现有图谱，合并数据
        if existing_graph:
            self._merge_existing_graph(existing_graph)
    
    def _merge_existing_graph(self, existing_graph):
        """合并现有知识图谱数据"""
        # 这个方法需要根据现有图谱的API进行调整
        # 这里假设现有图谱有entities和relations属性
        try:
            # 合并实体
            for entity_id, entity in existing_graph.entities.items():
                enhanced_entity = KnowledgeEntity(
                    entity_id=entity_id,
                    name=entity.get('name', entity_id),
                    entity_type=entity.get('type', 'unknown'),
                    attributes=entity.get('attributes', {}),
                    domains=self._infer_domains(entity)
                )
                self.entities[entity_id] = enhanced_entity
                self.entity_name_index[enhanced_entity.name] = entity_id
                
                # 添加到领域索引
                for domain in enhanced_entity.domains:
                    self.domain_entities[domain].add(entity_id)
            
            # 合并关系
            for rel_id, relation in existing_graph.relations.items():
                enhanced_relation = EnhancedRelation(
                    relation_id=rel_id,
                    source_entity=relation.get('source'),
                    target_entity=relation.get('target'),
                    relation_type=RelationType(relation.get('type', 'associated_with')),
                    confidence=relation.get('confidence', 1.0),
                    domains=self._infer_relation_domains(relation)
                )
                self.relations[rel_id] = enhanced_relation
                
                # 更新索引
                key = (enhanced_relation.source_entity, 
                      enhanced_relation.target_entity, 
                      enhanced_relation.relation_type.value)
                self.relation_index[key] = rel_id
                
                # 更新实体关系映射
                self.entity_relations[enhanced_relation.source_entity].append(rel_id)
                self.entity_relations[enhanced_relation.target_entity].append(rel_id)
                
            logger.info(f"合并现有知识图谱完成: {len(self.entities)} 个实体, {len(self.relations)} 个关系")
            
        except Exception as e:
            logger.error(f"合并现有知识图谱失败: {e}")
    
    def _infer_domains(self, entity: Dict[str, Any]) -> List[Domain]:
        """推断实体所属领域"""
        domains = []
        entity_type = entity.get('type', '').lower()
        entity_name = entity.get('name', '').lower()
        
        # 基于类型和名称推断领域
        domain_keywords = {
            Domain.SCIENCE: ['science', 'physics', 'chemistry', 'biology', 'scientific'],
            Domain.TECHNOLOGY: ['tech', 'computer', 'software', 'hardware', 'digital', 'ai'],
            Domain.ART: ['art', 'painting', 'sculpture', 'design', 'creative'],
            Domain.HISTORY: ['history', 'historical', 'ancient', 'medieval'],
            Domain.PHILOSOPHY: ['philosophy', 'philosopher', 'ethical', 'moral'],
            Domain.MATHEMATICS: ['math', 'mathematical', 'algebra', 'geometry', 'calculus'],
            Domain.LITERATURE: ['literature', 'book', 'novel', 'poem', 'author'],
            Domain.MEDICINE: ['medicine', 'medical', 'health', 'doctor', 'hospital'],
            Domain.BUSINESS: ['business', 'commerce', 'trade', 'market', 'company'],
            Domain.SPORTS: ['sports', 'sport', 'game', 'athlete', 'team'],
            Domain.MUSIC: ['music', 'song', 'musical', 'instrument', 'composer'],
            Domain.LAW: ['law', 'legal', 'court', 'justice', 'lawyer'],
            Domain.PSYCHOLOGY: ['psychology', 'psychological', 'mind', 'behavior'],
            Domain.SOCIOLOGY: ['sociology', 'society', 'social', 'community'],
            Domain.ECONOMICS: ['economics', 'economic', 'economy', 'financial']
        }
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in entity_type or keyword in entity_name:
                    domains.append(domain)
                    break
        
        # 如果没有推断出领域，使用默认领域
        if not domains:
            domains.append(Domain.KNOWLEDGE)
        
        return domains
    
    def _infer_relation_domains(self, relation: Dict[str, Any]) -> List[Domain]:
        """推断关系涉及领域"""
        # 简化的推断：基于关系类型
        domains = []
        rel_type = relation.get('type', '').lower()
        
        type_domain_mapping = {
            'is_a': [Domain.SCIENCE, Domain.PHILOSOPHY],
            'part_of': [Domain.SCIENCE, Domain.TECHNOLOGY],
            'has_property': [Domain.SCIENCE],
            'created_by': [Domain.ART, Domain.LITERATURE, Domain.MUSIC],
            'used_by': [Domain.TECHNOLOGY, Domain.BUSINESS],
            'similar_to': [Domain.SCIENCE, Domain.MATHEMATICS],
            'causes': [Domain.SCIENCE, Domain.MEDICINE],
            'influences': [Domain.HISTORY, Domain.PHILOSOPHY, Domain.PSYCHOLOGY]
        }
        
        for rel_pattern, domain_list in type_domain_mapping.items():
            if rel_pattern in rel_type:
                domains.extend(domain_list)
                break
        
        if not domains:
            domains.append(Domain.KNOWLEDGE)
        
        return list(set(domains))
    
    def discover_cross_domain_relations(self, max_relations: int = 100) -> List[EnhancedRelation]:
        """发现跨领域关系"""
        discovered_relations = []
        
        # 获取不同领域的实体
        domains = list(self.domain_entities.keys())
        
        if len(domains) < 2:
            logger.warning("需要至少两个领域才能发现跨领域关系")
            return discovered_relations
        
        # 为每个领域对寻找可能的关联
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                domain1 = domains[i]
                domain2 = domains[j]
                
                entities1 = list(self.domain_entities[domain1])
                entities2 = list(self.domain_entities[domain2])
                
                # 限制实体数量以提高性能
                if len(entities1) > 50:
                    entities1 = random.sample(entities1, 50)
                if len(entities2) > 50:
                    entities2 = random.sample(entities2, 50)
                
                # 寻找可能的关联
                for entity1_id in entities1:
                    entity1 = self.entities[entity1_id]
                    
                    for entity2_id in entities2:
                        entity2 = self.entities[entity2_id]
                        
                        # 检查是否已存在关系
                        if self._relation_exists(entity1_id, entity2_id):
                            continue
                        
                        # 计算关联可能性
                        similarity_score = self._calculate_entity_similarity(entity1, entity2)
                        
                        # 如果相似度超过阈值，创建跨领域关系
                        if similarity_score > 0.3:  # 可调整的阈值
                            relation_id = f"cross_domain_{len(self.relations)}"
                            domains_involved = list(set(entity1.domains + entity2.domains))
                            
                            relation = EnhancedRelation(
                                relation_id=relation_id,
                                source_entity=entity1_id,
                                target_entity=entity2_id,
                                relation_type=RelationType.CROSS_DOMAIN,
                                confidence=similarity_score,
                                strength=similarity_score * 0.5,  # 初始强度较低
                                domains=domains_involved,
                                evidence=["跨领域关联发现"],
                                metadata={
                                    "discovery_method": "similarity_based",
                                    "similarity_score": similarity_score,
                                    "source_domain": domain1.value,
                                    "target_domain": domain2.value
                                }
                            )
                            
                            # 添加到图谱
                            self.relations[relation_id] = relation
                            self.entity_relations[entity1_id].append(relation_id)
                            self.entity_relations[entity2_id].append(relation_id)
                            
                            # 更新索引
                            key = (entity1_id, entity2_id, RelationType.CROSS_DOMAIN.value)
                            self.relation_index[key] = relation_id
                            
                            discovered_relations.append(relation)
                            
                            if len(discovered_relations) >= max_relations:
                                return discovered_relations
        
        logger.info(f"发现 {len(discovered_relations)} 个跨领域关系")
        return discovered_relations
    
    def _relation_exists(self, entity1_id: str, entity2_id: str) -> bool:
        """检查两个实体之间是否存在关系"""
        # 检查两个方向
        key1 = (entity1_id, entity2_id, RelationType.CROSS_DOMAIN.value)
        key2 = (entity2_id, entity1_id, RelationType.CROSS_DOMAIN.value)
        
        return key1 in self.relation_index or key2 in self.relation_index
    
    def _calculate_entity_similarity(self, entity1: KnowledgeEntity, entity2: KnowledgeEntity) -> float:
        """计算实体相似度"""
        similarity = 0.0
        
        # 1. 名称相似度（简化的字符串匹配）
        name1 = entity1.name.lower()
        name2 = entity2.name.lower()
        
        if name1 == name2:
            similarity += 0.5
        elif name1 in name2 or name2 in name1:
            similarity += 0.3
        elif any(word in name2 for word in name1.split()):
            similarity += 0.2
        
        # 2. 类型相似度
        if entity1.entity_type == entity2.entity_type:
            similarity += 0.3
        
        # 3. 领域重叠
        common_domains = set(entity1.domains) & set(entity2.domains)
        if common_domains:
            similarity += 0.2 * len(common_domains)
        
        # 4. 属性相似度（如果有嵌入向量）
        if entity1.embedding and entity2.embedding:
            try:
                vec_similarity = self._cosine_similarity(entity1.embedding, entity2.embedding)
                similarity += vec_similarity * 0.4
            except:
                pass
        
        # 确保相似度在0-1之间
        return min(1.0, similarity)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def calculate_relation_strength(self, relation_id: str) -> float:
        """计算关系强度"""
        if relation_id not in self.relations:
            return 0.0
        
        relation = self.relations[relation_id]
        
        # 基础强度因子
        strength_factors = []
        
        # 1. 置信度因子
        strength_factors.append(relation.confidence * 0.3)
        
        # 2. 证据数量因子
        evidence_factor = min(1.0, len(relation.evidence) / 10.0)  # 最多10个证据
        strength_factors.append(evidence_factor * 0.2)
        
        # 3. 时间衰减因子（越新强度越高）
        time_decay = math.exp(-0.0001 * (time.time() - relation.created_at))  # 半衰期约6931秒
        strength_factors.append(time_decay * 0.2)
        
        # 4. 实体重要性因子
        source_importance = self.entities.get(relation.source_entity, KnowledgeEntity("", "", "")).importance_score
        target_importance = self.entities.get(relation.target_entity, KnowledgeEntity("", "", "")).importance_score
        importance_factor = (source_importance + target_importance) / 2.0
        strength_factors.append(importance_factor * 0.3)
        
        # 计算总强度
        total_strength = sum(strength_factors)
        
        # 更新关系强度
        relation.strength = total_strength
        relation.updated_at = time.time()
        
        return total_strength
    
    def fuse_knowledge_from_multiple_sources(self, entity_id: str, 
                                           new_attributes: Dict[str, Any],
                                           source: str = "unknown") -> bool:
        """融合多源知识"""
        if entity_id not in self.entities:
            logger.warning(f"实体不存在: {entity_id}")
            return False
        
        entity = self.entities[entity_id]
        conflicts = []
        fused_count = 0
        
        for key, new_value in new_attributes.items():
            if key in entity.attributes:
                old_value = entity.attributes[key]
                
                # 检查冲突
                if old_value != new_value:
                    conflicts.append({
                        'attribute': key,
                        'old_value': old_value,
                        'new_value': new_value,
                        'source': source
                    })
                    
                    # 冲突解决策略：保留更具体的值或更新
                    if isinstance(old_value, str) and isinstance(new_value, str):
                        # 如果新值更长（可能包含更多信息），则更新
                        if len(new_value) > len(old_value):
                            entity.attributes[key] = new_value
                            fused_count += 1
                    elif isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                        # 数值型属性：取平均值
                        entity.attributes[key] = (old_value + new_value) / 2.0
                        fused_count += 1
                    else:
                        # 其他情况：保留旧值，记录新值作为备选
                        if 'alternative_values' not in entity.metadata:
                            entity.metadata['alternative_values'] = {}
                        if key not in entity.metadata['alternative_values']:
                            entity.metadata['alternative_values'][key] = []
                        entity.metadata['alternative_values'][key].append({
                            'value': new_value,
                            'source': source,
                            'timestamp': time.time()
                        })
            else:
                # 新属性，直接添加
                entity.attributes[key] = new_value
                fused_count += 1
        
        # 更新实体
        entity.updated_at = time.time()
        
        # 记录融合结果
        if conflicts:
            logger.info(f"知识融合完成: {entity_id}, 融合 {fused_count} 个属性, {len(conflicts)} 个冲突")
        else:
            logger.info(f"知识融合完成: {entity_id}, 融合 {fused_count} 个属性, 无冲突")
        
        return fused_count > 0
    
    def infer_indirect_relations(self, entity1_id: str, entity2_id: str, 
                               max_path_length: int = 3) -> List[List[str]]:
        """推理间接关系（路径查找）"""
        if entity1_id not in self.entities or entity2_id not in self.entities:
            return []
        
        # 使用BFS查找路径
        paths = []
        queue = deque([(entity1_id, [entity1_id])])
        visited = set([entity1_id])
        
        while queue:
            current_entity, path = queue.popleft()
            
            # 如果路径过长，跳过
            if len(path) > max_path_length + 1:
                continue
            
            # 如果找到目标实体
            if current_entity == entity2_id and len(path) > 1:
                paths.append(path)
                continue
            
            # 探索邻居实体
            relation_ids = self.entity_relations.get(current_entity, [])
            for rel_id in relation_ids:
                relation = self.relations[rel_id]
                
                # 确定下一个实体
                if relation.source_entity == current_entity:
                    next_entity = relation.target_entity
                else:
                    next_entity = relation.source_entity
                
                if next_entity not in visited:
                    visited.add(next_entity)
                    queue.append((next_entity, path + [next_entity]))
        
        # 对路径按长度排序
        paths.sort(key=len)
        
        return paths
    
    def build_knowledge_subgraph(self, center_entity_id: str, 
                               radius: int = 2) -> Dict[str, Any]:
        """构建知识子图（以指定实体为中心）"""
        if center_entity_id not in self.entities:
            return {"error": f"实体不存在: {center_entity_id}"}
        
        subgraph_entities = set([center_entity_id])
        subgraph_relations = set()
        
        # BFS收集实体和关系
        queue = deque([(center_entity_id, 0)])
        visited = set([center_entity_id])
        
        while queue:
            current_entity, distance = queue.popleft()
            
            if distance >= radius:
                continue
            
            # 收集当前实体的关系
            relation_ids = self.entity_relations.get(current_entity, [])
            for rel_id in relation_ids:
                relation = self.relations[rel_id]
                subgraph_relations.add(rel_id)
                
                # 确定邻居实体
                if relation.source_entity == current_entity:
                    neighbor = relation.target_entity
                else:
                    neighbor = relation.source_entity
                
                subgraph_entities.add(neighbor)
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        
        # 构建子图数据
        subgraph_data = {
            "center_entity": center_entity_id,
            "radius": radius,
            "entity_count": len(subgraph_entities),
            "relation_count": len(subgraph_relations),
            "entities": {eid: self._entity_to_dict(eid) for eid in subgraph_entities},
            "relations": {rid: self._relation_to_dict(rid) for rid in subgraph_relations},
            "domains": self._extract_subgraph_domains(subgraph_entities),
            "density": len(subgraph_relations) / max(1, len(subgraph_entities))
        }
        
        return subgraph_data
    
    def _entity_to_dict(self, entity_id: str) -> Dict[str, Any]:
        """将实体转换为字典"""
        entity = self.entities[entity_id]
        return {
            "id": entity.entity_id,
            "name": entity.name,
            "type": entity.entity_type,
            "domains": [d.value for d in entity.domains],
            "attributes": entity.attributes,
            "importance_score": entity.importance_score,
            "centrality_score": entity.centrality_score
        }
    
    def _relation_to_dict(self, relation_id: str) -> Dict[str, Any]:
        """将关系转换为字典"""
        relation = self.relations[relation_id]
        return {
            "id": relation.relation_id,
            "source": relation.source_entity,
            "target": relation.target_entity,
            "type": relation.relation_type.value,
            "confidence": relation.confidence,
            "strength": relation.strength,
            "domains": [d.value for d in relation.domains]
        }
    
    def _extract_subgraph_domains(self, entity_ids: Set[str]) -> Dict[str, int]:
        """提取子图涉及的领域分布"""
        domain_counts = defaultdict(int)
        
        for entity_id in entity_ids:
            entity = self.entities[entity_id]
            for domain in entity.domains:
                domain_counts[domain.value] += 1
        
        return dict(domain_counts)
    
    def analyze_cross_domain_bridges(self) -> List[CrossDomainAssociation]:
        """分析跨领域桥梁（连接不同领域的实体）"""
        associations = []
        
        # 获取所有实体
        all_entities = list(self.entities.values())
        
        # 寻找连接多个领域的实体
        for entity in all_entities:
            if len(entity.domains) >= 2:
                # 这是一个跨领域实体
                for i in range(len(entity.domains)):
                    for j in range(i + 1, len(entity.domains)):
                        source_domain = entity.domains[i]
                        target_domain = entity.domains[j]
                        
                        # 检查是否已存在该领域对的关联
                        association_id = f"{source_domain.value}_{target_domain.value}"
                        
                        if association_id not in self.cross_domain_associations:
                            association = CrossDomainAssociation(
                                association_id=association_id,
                                source_domain=source_domain,
                                target_domain=target_domain,
                                bridging_entities=[entity.entity_id],
                                association_strength=0.5,  # 初始强度
                                evidence_count=1
                            )
                            self.cross_domain_associations[association_id] = association
                            associations.append(association)
                        else:
                            # 更新现有关联
                            association = self.cross_domain_associations[association_id]
                            if entity.entity_id not in association.bridging_entities:
                                association.bridging_entities.append(entity.entity_id)
                                association.evidence_count += 1
                                # 更新关联强度（基于桥接实体数量）
                                association.association_strength = min(1.0, 
                                    association.evidence_count / 10.0)
        
        logger.info(f"发现 {len(associations)} 个跨领域关联")
        return associations
    
    def calculate_entity_centrality(self, entity_id: str) -> float:
        """计算实体中心性（基于连接度）"""
        if entity_id not in self.entities:
            return 0.0
        
        # 获取实体的直接连接数
        direct_connections = len(self.entity_relations.get(entity_id, []))
        
        # 获取二级连接（邻居的邻居）
        secondary_connections = 0
        visited = set([entity_id])
        
        relation_ids = self.entity_relations.get(entity_id, [])
        for rel_id in relation_ids:
            relation = self.relations[rel_id]
            
            # 确定邻居实体
            if relation.source_entity == entity_id:
                neighbor = relation.target_entity
            else:
                neighbor = relation.source_entity
            
            if neighbor not in visited:
                visited.add(neighbor)
                # 计算邻居的连接数（不包括当前实体）
                neighbor_relations = self.entity_relations.get(neighbor, [])
                neighbor_connections = 0
                for n_rel_id in neighbor_relations:
                    n_relation = self.relations[n_rel_id]
                    if n_relation.source_entity != entity_id and n_relation.target_entity != entity_id:
                        neighbor_connections += 1
                secondary_connections += neighbor_connections
        
        # 计算综合中心性分数
        centrality = direct_connections * 0.7 + secondary_connections * 0.3
        
        # 更新实体中心性分数
        entity = self.entities[entity_id]
        entity.centrality_score = centrality
        
        return centrality
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        # 计算实体和关系数量
        entity_count = len(self.entities)
        relation_count = len(self.relations)
        
        # 计算平均连接度
        avg_degree = 0
        if entity_count > 0:
            total_degree = sum(len(relations) for relations in self.entity_relations.values())
            avg_degree = total_degree / entity_count
        
        # 计算领域分布
        domain_distribution = {}
        for domain, entity_set in self.domain_entities.items():
            domain_distribution[domain.value] = len(entity_set)
        
        # 计算跨领域关系比例
        cross_domain_relations = 0
        for relation in self.relations.values():
            if relation.relation_type == RelationType.CROSS_DOMAIN:
                cross_domain_relations += 1
        
        cross_domain_ratio = 0
        if relation_count > 0:
            cross_domain_ratio = cross_domain_relations / relation_count
        
        return {
            "entity_count": entity_count,
            "relation_count": relation_count,
            "average_degree": avg_degree,
            "domain_distribution": domain_distribution,
            "cross_domain_relations": cross_domain_relations,
            "cross_domain_ratio": cross_domain_ratio,
            "cross_domain_associations": len(self.cross_domain_associations),
            "timestamp": time.time()
        }

# 全局实例
_enhanced_knowledge_graph = None

def get_enhanced_knowledge_graph(existing_graph=None) -> EnhancedKnowledgeGraph:
    """获取增强知识图谱单例"""
    global _enhanced_knowledge_graph
    if _enhanced_knowledge_graph is None:
        _enhanced_knowledge_graph = EnhancedKnowledgeGraph(existing_graph)
    return _enhanced_knowledge_graph

# 为兼容性创建Entity别名
Entity = KnowledgeEntity
