"""
增强型知识整合模块 - 实现真正的AGI级知识表示和推理
集成深度学习、知识图谱、因果推理和常识理解
"""

import torch
import torch.nn as nn
import numpy as np
import requests
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, field
import networkx as nx
from collections import deque
import hashlib
from datetime import datetime
import pickle
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EnhancedKnowledgeRelation:
    """增强型知识关系表示 - 支持复杂关系建模"""
    # 基础关系信息
    subject: str
    relation: str
    object: str
    confidence: float
    source: str
    
    # 关系类型标记
    relation_type: str  # 时间、因果、层次、空间等
    
    # 动态属性
    created_at: datetime
    updated_at: datetime
    
    # 多维度关系属性
    temporal_context: Dict[str, Any]  # 时间关系：before, after, during, simultaneous
    causal_context: Dict[str, Any]    # 因果关系：strength, direction, mechanism
    hierarchical_context: Dict[str, Any]  # 层次关系：parent, child, part_of, whole_of
    spatial_context: Dict[str, Any]   # 空间关系：above, below, inside, outside
    
    # 可选动态属性
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    
    # 语义和推理属性
    semantic_embedding: Optional[torch.Tensor] = None
    causal_strength: float = 0.0
    utility_value: float = 0.0
    novelty_score: float = 0.0
    reliability_score: float = 0.0
    
    # 领域和上下文信息
    domain: str = "general"
    context_tags: List[str] = field(default_factory=list)

class DeepKnowledgeRepresentation(nn.Module):
    """深度知识表示网络 - 集成概率逻辑层"""
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 1024, from_scratch: bool = True):
        super(DeepKnowledgeRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 设置训练模式
        self.from_scratch = from_scratch
        
        # 使用简单的嵌入层而不是BERT
        logger.info("Using simple embedding layer (from_scratch mode only)")
        self.simple_embedding = nn.Embedding(10000, embedding_dim)  # Simple embedding for from_scratch
        # 使用随机初始化的token
        self.token_to_id = {}
        self.next_id = 0
        
        # Relation encoding network
        self.relation_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU()
        )
        
        # Causal reasoning network
        self.causal_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 概率分布层 - 输出均值和方差
        self.probability_mean = nn.Linear(hidden_dim // 2, embedding_dim)
        self.probability_logvar = nn.Linear(hidden_dim // 2, embedding_dim)
        
        # 概率逻辑层 - 处理不确定性传播
        self.probabilistic_logic_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 2),  # 输出概率和置信度
            nn.Sigmoid()
        )
        
        # Knowledge fusion layer
        self.fusion_network = nn.Linear(hidden_dim // 2, embedding_dim)
        
    def forward(self, subject: str, relation: str, object: str) -> Tuple[torch.Tensor, float, float, float]:
        """生成知识表示的深度嵌入、因果强度、概率和不确定性
        
        返回值:
            fused_knowledge: 知识融合嵌入
            causal_strength: 因果强度
            probability: 关系成立的概率
            uncertainty: 不确定性度量
        """
        # 编码文本到语义空间
        sub_embed = self._encode_text(subject)
        rel_embed = self._encode_text(relation)
        obj_embed = self._encode_text(object)
        
        # 关系编码
        combined = torch.cat([sub_embed, rel_embed, obj_embed], dim=-1)
        relation_encoded = self.relation_encoder(combined)
        
        # 因果推理
        causal_strength = self.causal_network(relation_encoded)
        
        # 概率分布表示
        mean = self.probability_mean(relation_encoded)
        logvar = self.probability_logvar(relation_encoded)
        uncertainty = torch.exp(logvar).mean().item()  # 不确定性度量
        
        # 知识融合
        fused_knowledge = self.fusion_network(relation_encoded)
        
        # 概率逻辑推理
        logic_input = torch.cat([mean, torch.exp(logvar)], dim=-1)
        logic_output = self.probabilistic_logic_layer(logic_input)
        probability = logic_output[0, 0].item()  # 关系成立的概率
        
        return fused_knowledge, causal_strength.item(), probability, uncertainty
    
    def probabilistic_reasoning(self, subject: str, relation: str, object: str) -> Dict[str, Any]:
        """概率推理 - 输出带不确定性的推理结果"""
        with torch.no_grad():
            embedding, causal_strength, probability, uncertainty = self.forward(subject, relation, object)
        
        return {
            "embedding": embedding,
            "causal_strength": causal_strength,
            "probability": probability,
            "uncertainty": uncertainty,
            "confidence": 1.0 - uncertainty,  # 置信度
            "explanation": self._generate_probabilistic_explanation(causal_strength, probability, uncertainty)
        }
    
    def _generate_probabilistic_explanation(self, causal_strength: float, probability: float, uncertainty: float) -> str:
        """生成概率推理的解释"""
        explanation = []
        
        if causal_strength > 0.7:
            explanation.append("强因果关系")
        elif causal_strength > 0.4:
            explanation.append("中等因果关系")
        else:
            explanation.append("弱因果关系")
        
        if probability > 0.8:
            explanation.append("高概率成立")
        elif probability > 0.5:
            explanation.append("中等概率成立")
        else:
            explanation.append("低概率成立")
        
        if uncertainty < 0.3:
            explanation.append("低不确定性")
        elif uncertainty < 0.6:
            explanation.append("中等不确定性")
        else:
            explanation.append("高不确定性")
        
        return ", ".join(explanation)
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """编码文本，根据模式使用不同的编码器"""
        if self.from_scratch:
            # In from_scratch mode, use simple embedding
            # Simple tokenization and embedding
            tokens = text.lower().split()
            token_ids = []
            for token in tokens:
                if token not in self.token_to_id:
                    self.token_to_id[token] = self.next_id
                    self.next_id += 1
                token_ids.append(self.token_to_id[token])
                
            # Ensure we have at least one token
            if not token_ids:
                token_ids = [0]  # Default token if text is empty
                
            # Truncate or pad to max length 128
            token_ids = token_ids[:128] + [0] * (128 - len(token_ids))
            
            # Convert to tensor and get embeddings
            token_tensor = torch.tensor([token_ids[:128]])  # Take first 128 tokens
            with torch.no_grad():
                embeddings = self.simple_embedding(token_tensor)
            return embeddings.mean(dim=1)  # 平均池化
        else:
            # Use BERT for encoding - lazy import to avoid download at module import
            try:
                from transformers import BertTokenizer, BertModel
                
                # Initialize tokenizer and model if not already done
                if not hasattr(self, 'tokenizer'):
                    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                if not hasattr(self, 'bert_model'):
                    self.bert_model = BertModel.from_pretrained('bert-base-uncased')
                
                inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                return outputs.last_hidden_state.mean(dim=1)  # 平均池化
            except ImportError:
                # Fallback to simple embedding if transformers not available
                logger.warning("BERT tokenizer not available, falling back to simple embedding")
                tokens = text.lower().split()
                token_ids = [hash(token) % 10000 for token in tokens[:128]]
                token_tensor = torch.tensor([token_ids + [0] * (128 - len(token_ids))])
                with torch.no_grad():
                    embeddings = nn.Embedding(10000, self.embedding_dim)(token_tensor)
                return embeddings.mean(dim=1)

class AGIKnowledgeIntegrator:
    """
    AGI级知识整合器 - 实现深度知识表示、因果推理和常识理解
    集成神经网络知识编码、动态知识图谱和多源知识融合
    """
    
    def __init__(self, knowledge_base_path: str = "data/agi_knowledge", from_scratch: bool = False):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化深度知识表示网络
        self.knowledge_encoder = DeepKnowledgeRepresentation(from_scratch=from_scratch)
        
        # 动态知识图谱
        self.knowledge_graph = nx.MultiDiGraph()
        self.semantic_index = {}
        self.temporal_context = {}
        self.causal_relationships = {}
        
        # 外部知识源配置
        self.external_sources = {
            "conceptnet": "http://api.conceptnet.io",
            "wordnet": "http://wordnet-rdf.princeton.edu",
            "dbpedia": "http://dbpedia.org/sparql",
            "wikidata": "https://query.wikidata.org/sparql"
        }
        
        # 缓存和统计
        self.cached_knowledge = {}
        self.performance_metrics = {
            "total_relations": 0,
            "semantic_queries": 0,
            "causal_inferences": 0,
            "knowledge_fusions": 0,
            "cache_hits": 0
        }
        
        # 记录是否从零开始训练
        self.from_scratch = from_scratch
        
        # 根据是否从零开始训练决定是否加载现有知识
        if not from_scratch:
            self._load_knowledge_base()
        else:
            logger.info("从零开始训练模式 - 不加载现有知识库")
        
        logger.info("AGI知识整合器初始化完成")
    
    def _load_knowledge_base(self):
        """加载知识库"""
        try:
            knowledge_file = self.knowledge_base_path / "agi_knowledge.pkl"
            if knowledge_file.exists():
                with open(knowledge_file, 'rb') as f:
                    data = pickle.load(f)
                    self.knowledge_graph = data['graph']
                    self.semantic_index = data['semantic_index']
                    self.temporal_context = data['temporal_context']
                    self.causal_relationships = data['causal_relationships']
                logger.info("AGI知识库加载成功")
        except Exception as e:
            logger.warning(f"加载知识库失败: {e}")
    
    def save_knowledge_base(self):
        """保存知识库"""
        try:
            data = {
                'graph': self.knowledge_graph,
                'semantic_index': self.semantic_index,
                'temporal_context': self.temporal_context,
                'causal_relationships': self.causal_relationships
            }
            with open(self.knowledge_base_path / "agi_knowledge.pkl", 'wb') as f:
                pickle.dump(data, f)
            logger.info("AGI知识库保存成功")
        except Exception as e:
            logger.error(f"保存知识库失败: {e}")
    
    def add_knowledge(self, subject: str, relation: str, object: str, 
                     confidence: float = 0.8, source: str = "system",
                     relation_type: str = "general",
                     domain: str = "general",
                     context: Optional[Dict[str, Any]] = None,
                     context_tags: Optional[List[str]] = None) -> str:
        """
        添加深度知识表示到知识图谱 - 支持复杂关系建模
        
        参数:
            relation_type: 关系类型 (时间、因果、层次、空间等)
            domain: 知识领域
            context: 上下文信息
            context_tags: 上下文标签
        """
        # 生成语义嵌入和因果强度
        with torch.no_grad():
            semantic_embedding, causal_strength, probability, uncertainty = self.knowledge_encoder(subject, relation, object)
        
        # 当前时间
        now = datetime.now()
        
        # 创建知识关系
        knowledge_id = hashlib.sha256(f"{subject}{relation}{object}{now.timestamp()}".encode()).hexdigest()[:16]
        
        # 自动检测关系类型
        detected_relation_type = self._detect_relation_type(relation)
        if relation_type == "general" and detected_relation_type:
            relation_type = detected_relation_type
        
        relation_data = EnhancedKnowledgeRelation(
            subject=subject,
            relation=relation,
            object=object,
            confidence=confidence,
            source=source,
            relation_type=relation_type,
            created_at=now,
            updated_at=now,
            
            # 多维度关系上下文
            temporal_context=self._build_temporal_context(context),
            causal_context=self._build_causal_context(relation, causal_strength, probability, uncertainty),
            hierarchical_context=self._build_hierarchical_context(relation),
            spatial_context=self._build_spatial_context(relation),
            
            semantic_embedding=semantic_embedding,
            causal_strength=causal_strength,
            utility_value=self._calculate_utility(confidence, causal_strength),
            reliability_score=self._calculate_reliability_score(source, confidence),
            
            domain=domain,
            context_tags=context_tags or []
        )
        
        # 添加到知识图谱
        if knowledge_id not in self.knowledge_graph:
            self.knowledge_graph.add_node(knowledge_id, **relation_data.__dict__)
        
        # 为每个概念创建节点（如果不存在）
        if subject not in self.knowledge_graph:
            self.knowledge_graph.add_node(subject, type='concept', name=subject)
        if object not in self.knowledge_graph:
            self.knowledge_graph.add_node(object, type='concept', name=object)
        
        # 创建从主题到知识节点的边
        if not self.knowledge_graph.has_edge(subject, knowledge_id):
            self.knowledge_graph.add_edge(subject, knowledge_id, type='has_knowledge', relation='subject')
        
        # 创建从知识节点到对象的边
        if not self.knowledge_graph.has_edge(knowledge_id, object):
            self.knowledge_graph.add_edge(knowledge_id, object, type='has_knowledge', relation='object')
        
        # 如果是因果关系，创建直接的因果边
        if relation_type == 'causal':
            if not self.knowledge_graph.has_edge(subject, object):
                self.knowledge_graph.add_edge(subject, object, 
                                             type='causal', 
                                             relation=relation, 
                                             strength=causal_strength, 
                                             confidence=confidence, 
                                             knowledge_id=knowledge_id)
        
        # 更新语义索引 - 使用每个概念的文本嵌入而不是融合后的知识嵌入
        with torch.no_grad():
            subject_embedding = self.knowledge_encoder._encode_text(subject)
            object_embedding = self.knowledge_encoder._encode_text(object)
            # 分别为subject和object更新语义索引
            self._update_semantic_index(subject, relation, object, subject_embedding, knowledge_id)
            self._update_semantic_index(object, relation, subject, object_embedding, knowledge_id)
        
        # 更新上下文
        if context:
            self.temporal_context[knowledge_id] = context
        
        # 添加知识到因果关系字典
        if relation_type == 'causal':
            if subject not in self.causal_relationships:
                self.causal_relationships[subject] = []
            self.causal_relationships[subject].append({
                'object': object,
                'relation': relation,
                'strength': causal_strength,
                'confidence': confidence
            })
        
        self.performance_metrics["total_relations"] += 1
        return knowledge_id
    
    def _update_semantic_index(self, subject: str, relation: str, object: str, 
                              embedding: torch.Tensor, knowledge_id: str):
        """更新语义索引"""
        # 为每个概念创建语义索引
        for concept in [subject, object]:
            if concept not in self.semantic_index:
                self.semantic_index[concept] = {
                    'embeddings': [],
                    'relations': [],
                    'knowledge_ids': []
                }
            self.semantic_index[concept]['embeddings'].append(embedding)
            self.semantic_index[concept]['relations'].append(relation)
            self.semantic_index[concept]['knowledge_ids'].append(knowledge_id)
        logger.debug(f"更新语义索引: subject={subject}, object={object}, knowledge_id={knowledge_id}")
        logger.debug(f"当前语义索引包含概念: {list(self.semantic_index.keys())}")
    
    def _calculate_utility(self, confidence: float, causal_strength: float) -> float:
        """计算知识效用值"""
        return min(1.0, confidence * 0.6 + causal_strength * 0.4)
    
    def _detect_relation_type(self, relation: str) -> Optional[str]:
        """自动检测关系类型"""
        relation_lower = relation.lower()
        
        # 时间关系检测
        temporal_keywords = ['before', 'after', 'during', 'simultaneous', 'when', 'while', 'then', 'previously']
        if any(keyword in relation_lower for keyword in temporal_keywords):
            return "temporal"
        
        # 因果关系检测
        causal_keywords = ['causes', 'because', 'due to', 'leads to', 'results in', 'affects', 'influences']
        if any(keyword in relation_lower for keyword in causal_keywords):
            return "causal"
        
        # 层次关系检测
        hierarchical_keywords = ['is a', 'type of', 'part of', 'composed of', 'belongs to', 'includes', 'contains']
        if any(keyword in relation_lower for keyword in hierarchical_keywords):
            return "hierarchical"
        
        # 空间关系检测
        spatial_keywords = ['above', 'below', 'inside', 'outside', 'near', 'far from', 'next to', 'behind']
        if any(keyword in relation_lower for keyword in spatial_keywords):
            return "spatial"
        
        return None
    
    def _build_temporal_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """构建时间上下文"""
        temporal_context = {
            'created': datetime.now(),
            'last_accessed': datetime.now(),
            'access_count': 1
        }
        
        if context:
            temporal_context.update(context.get('temporal', {}))
        
        return temporal_context
    
    def _build_causal_context(self, relation: str, causal_strength: float, probability: float = 0.0, uncertainty: float = 0.0) -> Dict[str, Any]:
        """构建因果上下文"""
        return {
            'strength': causal_strength,
            'direction': 'forward',
            'mechanism': relation,
            'confidence': min(1.0, causal_strength * 1.5),
            'probability': probability,
            'uncertainty': uncertainty,
            'reliability': 1.0 - uncertainty
        }
    
    def _build_hierarchical_context(self, relation: str) -> Dict[str, Any]:
        """构建层次上下文"""
        relation_lower = relation.lower()
        hierarchical_context = {
            'type': 'unknown',
            'depth': 0
        }
        
        if 'part of' in relation_lower or 'composed of' in relation_lower:
            hierarchical_context['type'] = 'part_of'
        elif 'is a' in relation_lower or 'type of' in relation_lower:
            hierarchical_context['type'] = 'type_of'
        elif 'includes' in relation_lower or 'contains' in relation_lower:
            hierarchical_context['type'] = 'contains'
        
        return hierarchical_context
    
    def _build_spatial_context(self, relation: str) -> Dict[str, Any]:
        """构建空间上下文"""
        relation_lower = relation.lower()
        spatial_context = {
            'type': 'unknown',
            'distance': 0.0
        }
        
        if 'above' in relation_lower:
            spatial_context['type'] = 'above'
        elif 'below' in relation_lower:
            spatial_context['type'] = 'below'
        elif 'inside' in relation_lower:
            spatial_context['type'] = 'inside'
        elif 'outside' in relation_lower:
            spatial_context['type'] = 'outside'
        elif 'near' in relation_lower or 'next to' in relation_lower:
            spatial_context['type'] = 'proximity'
            spatial_context['distance'] = 0.1
        
        return spatial_context
    
    def _calculate_reliability_score(self, source: str, confidence: float) -> float:
        """计算知识可靠性分数"""
        # 来源可信度权重
        source_weights = {
            'system': 0.9,
            'expert': 0.85,
            'verified': 0.8,
            'crowdsourced': 0.6,
            'unknown': 0.5
        }
        
        source_weight = source_weights.get(source, 0.5)
        return min(1.0, confidence * 0.7 + source_weight * 0.3)
    
    def semantic_query(self, query: str, max_results: int = 10, 
                      similarity_threshold: float = 0.3, domain_weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        语义查询 - 基于深度嵌入的相似性搜索，支持领域权重、置信度和时效性的增强相关性评分
        
        参数:
            query: 查询文本
            max_results: 最大返回结果数
            similarity_threshold: 相似度阈值
            domain_weights: 领域权重字典，用于调整不同领域的重要性
        """
        self.performance_metrics["semantic_queries"] += 1
        
        # 编码查询
        with torch.no_grad():
            # 由于knowledge_encoder是DeepKnowledgeRepresentation实例，它有_encode_text方法
            query_embedding = self.knowledge_encoder._encode_text(query)
        
        results = []
        
        # 在语义索引中搜索
        for concept, index_data in self.semantic_index.items():
            for i, embedding in enumerate(index_data['embeddings']):
                try:
                    # 确保嵌入向量形状兼容
                    if len(embedding.shape) == 1:
                        embedding = embedding.unsqueeze(0)
                    if len(query_embedding.shape) == 1:
                        query_embedding = query_embedding.unsqueeze(0)
                    
                    # 计算语义相似度
                    similarity = torch.cosine_similarity(query_embedding, embedding).mean().item()
                    
                    if similarity >= similarity_threshold:
                        knowledge_id = index_data['knowledge_ids'][i]
                        relation = index_data['relations'][i]
                        
                        node_data = self.knowledge_graph.nodes[knowledge_id]
                        
                        # 计算时效性因子 - 时间越近，权重越高
                        created_at = node_data['created_at']
                        time_diff_days = (datetime.now() - created_at).days
                        # 使用指数衰减模型，半衰期为30天
                        temporal_relevance = 2 ** (-time_diff_days / 30)
                        
                        # 计算领域权重
                        domain = node_data.get('domain', 'general')
                        domain_weight = domain_weights.get(domain, 1.0) if domain_weights else 1.0
                        
                        # 计算增强相关性评分 - 综合考虑多种因素
                        confidence = node_data['confidence']
                        causal_strength = node_data['causal_strength']
                        reliability = node_data['reliability_score']
                        
                        # 综合评分公式：语义相似度 * 0.4 + 置信度 * 0.2 + 领域权重 * 0.15 + 时效性 * 0.15 + 可靠性 * 0.1
                        enhanced_score = (similarity * 0.4 + confidence * 0.2 + domain_weight * 0.15 + 
                                         temporal_relevance * 0.15 + reliability * 0.1)
                        
                        results.append({
                            'concept': concept,
                            'relation': relation,
                            'similarity': similarity,
                            'enhanced_score': enhanced_score,
                            'confidence': confidence,
                            'causal_strength': causal_strength,
                            'reliability': reliability,
                            'domain': domain,
                            'temporal_relevance': temporal_relevance,
                            'created_at': created_at,
                            'knowledge_id': knowledge_id
                        })
                except Exception as e:
                    logger.error(f"计算相似度时出错: {e}")
                    continue
        
        # 按增强相关性评分排序
        results.sort(key=lambda x: x['enhanced_score'], reverse=True)
        return results[:max_results]
    
    def causal_reasoning(self, cause: str, effect: str, 
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        因果推理 - 分析因果关系强度和路径
        """
        self.performance_metrics["causal_inferences"] += 1
        
        # 查找相关知识
        cause_results = self.semantic_query(cause, max_results=5)
        effect_results = self.semantic_query(effect, max_results=5)
        
        if not cause_results or not effect_results:
            return {"strength": 0.0, "paths": [], "confidence": 0.0}
        
        # 分析因果路径
        causal_paths = self._find_causal_paths(cause_results, effect_results)
        overall_strength = self._calculate_overall_causal_strength(causal_paths)
        
        return {
            "strength": overall_strength,
            "paths": causal_paths,
            "confidence": min(1.0, overall_strength * 0.8),
            "temporal_context": context
        }
    
    def _find_causal_paths(self, cause_results: List[Dict[str, Any]], 
                          effect_results: List[Dict[str, Any]],
                          max_depth: int = 5,
                          max_paths: int = 10, 
                          evolutionary_factor: float = 0.1) -> List[Dict[str, Any]]:
        """查找因果路径 - 支持加权路径查找、动态演化和自适应学习
        
        参数:
            max_depth: 最大路径深度
            max_paths: 最大返回路径数
            evolutionary_factor: 演化因子，控制路径多样性和动态适应能力
        """
        paths = []
        visited_pairs = set()
        
        # 提取原因和效果的概念节点
        cause_concepts = {item['concept'] for item in cause_results}
        effect_concepts = {item['concept'] for item in effect_results}
        
        # 在概念节点之间查找路径
        for cause_concept in cause_concepts:
            for effect_concept in effect_concepts:
                if cause_concept not in self.knowledge_graph or effect_concept not in self.knowledge_graph:
                    continue
                    
                pair_key = (cause_concept, effect_concept)
                if pair_key in visited_pairs:
                    continue
                visited_pairs.add(pair_key)
                
                try:
                    # 找到所有可能的路径
                    all_paths = list(nx.all_simple_paths(self.knowledge_graph, 
                                                       cause_concept, 
                                                       effect_concept,
                                                       cutoff=max_depth))
                    
                    # 对每条路径进行评估
                    for path in all_paths[:max_paths * 2]:  # 先获取更多路径，再筛选
                        path_strength = 1.0
                        path_confidence = 1.0
                        path_specificity = 0.0
                        path_consistency = 1.0
                        path_info = []
                        
                        # 遍历路径中的边
                        for i in range(len(path) - 1):
                            u = path[i]
                            v = path[i+1]
                            
                            # 获取边数据 - 在MultiDiGraph中返回字典{edge_id: edge_data}
                            edges_data = self.knowledge_graph.get_edge_data(u, v)
                            if not edges_data:
                                continue
                            
                            # 如果这是一条知识边，获取知识节点数据
                            knowledge_node_id = None
                            edge_type = None
                            edge_relation = None
                            
                            # 遍历所有边数据（因为可能有多条边）
                            for edge_id, edge_data in edges_data.items():
                                if 'type' in edge_data:
                                    edge_type = edge_data['type']
                                    if edge_type == 'has_knowledge':
                                        edge_relation = edge_data.get('relation')
                                        # 确定知识节点
                                        if edge_relation == 'subject':
                                            knowledge_node_id = v
                                        else:  # object
                                            knowledge_node_id = u
                                        
                                        # 检查知识节点是否存在
                                        if knowledge_node_id in self.knowledge_graph.nodes:
                                            node_data = self.knowledge_graph.nodes[knowledge_node_id]
                                            
                                            # 计算边的综合权重
                                            edge_weight_value = self._calculate_edge_weight(u, v, node_data)
                                            
                                            # 获取核心参数
                                            causal_strength = node_data.get('causal_strength', 0.5)
                                            confidence = node_data.get('confidence', 0.5)
                                            reliability = node_data.get('reliability_score', 0.5)
                                            relation_type = node_data.get('relation_type', 'general')
                                            domain = node_data.get('domain', 'general')
                                            
                                            # 更新路径强度和置信度
                                            path_strength *= (causal_strength * 0.5 + confidence * 0.3 + reliability * 0.2)
                                            path_confidence *= confidence
                                            
                                            # 计算路径一致性
                                            if i > 0 and path_info:
                                                prev_relation_type = path_info[-1]['relation_type']
                                                prev_domain = path_info[-1]['domain']
                                                if relation_type != prev_relation_type:
                                                    path_consistency *= 0.95
                                                if domain != prev_domain:
                                                    path_consistency *= 0.9
                                            
                                            # 添加路径信息
                                            path_info.append({
                                                'from': node_data['subject'],
                                                'to': node_data['object'],
                                                'relation': node_data['relation'],
                                                'relation_type': relation_type,
                                                'strength': causal_strength,
                                                'confidence': confidence,
                                                'reliability': reliability,
                                                'domain': domain,
                                                'weight': edge_weight_value,
                                                'created_at': node_data.get('created_at', datetime.now()),
                                                'source': node_data.get('source', 'unknown')
                                            })
                                        break
                                elif edge_type == 'causal':
                                    # 处理直接因果边
                                    causal_strength = edge_data.get('strength', 0.5)
                                    confidence = edge_data.get('confidence', 0.5)
                                    
                                    # 更新路径强度和置信度
                                    path_strength *= causal_strength
                                    path_confidence *= confidence
                                    
                                    # 添加路径信息
                                    path_info.append({
                                        'from': u,
                                        'to': v,
                                        'relation': 'causal',
                                        'relation_type': 'causal',
                                        'strength': causal_strength,
                                        'confidence': confidence,
                                        'reliability': 0.5,  # 默认值
                                        'domain': 'general',  # 默认值
                                        'weight': 1.0,  # 默认值
                                        'created_at': datetime.now(),
                                        'source': 'direct'
                                    })
                                    break
                            
                        # 只有包含知识信息的路径才被考虑
                        if len(path_info) > 0:
                            # 计算路径的综合评分
                            path_score = self._calculate_path_score(path_strength, path_confidence, 
                                                                  len(path), path_info, 
                                                                  path_consistency, path_specificity, 
                                                                  evolutionary_factor)
                            
                            # 增强路径表示，包含演化信息
                            paths.append({
                                'strength': path_strength,
                                'confidence': path_confidence,
                                'score': path_score,
                                'length': len(path),
                                'depth': len(path) - 1,
                                'path': path_info,
                                'consistency': path_consistency,
                                'causal_density': sum(1 for step in path_info if step['relation_type'] == 'causal') / len(path_info),
                                'evolutionary_potential': self._calculate_evolutionary_potential(path_info)
                            })
                    
                except nx.NetworkXNoPath:
                    continue
        
        # 按路径评分排序，选择最佳路径
        if paths:
            paths.sort(key=lambda x: x['score'], reverse=True)
        
        # 确保路径多样性，避免只选择相似路径
        unique_paths = []
        seen_patterns = set()
        
        for path in paths:
            # 创建路径模式标识
            pattern = tuple((step['from'], step['to'], step['relation_type']) for step in path['path'])
            if pattern not in seen_patterns:
                seen_patterns.add(pattern)
                unique_paths.append(path)
                if len(unique_paths) >= max_paths:
                    break
        
        # 动态更新知识图谱，增强频繁使用的因果路径
        self._evolve_knowledge_graph(unique_paths, evolutionary_factor)
        
        return unique_paths
    
    def _calculate_overall_causal_strength(self, paths: List[Dict[str, Any]]) -> float:
        """计算整体因果强度"""
        if not paths:
            return 0.0
        
        strengths = [path['strength'] for path in paths]
        return max(strengths) if strengths else 0.0
    
    def _calculate_edge_weight(self, u: str, v: str, node_data: Dict[str, Any]) -> float:
        """计算边权重 - 综合考虑多种因素
        
        返回值: 权重值（越小越好）
        """
        # 基础因素
        causal_strength = node_data.get('causal_strength', 0.5)
        confidence = node_data.get('confidence', 0.5)
        reliability = node_data.get('reliability_score', 0.5)
        
        # 时效性因素
        created_at = node_data.get('created_at', datetime.now())
        age_days = (datetime.now() - created_at).days
        recency_factor = max(0.1, 1.0 - age_days / 365.0)  # 时间越近，权重越高
        
        # 领域因素
        domain = node_data.get('domain', 'general')
        domain_weight = self._get_domain_weight(domain)
        
        # 关系类型因素
        relation_type = node_data.get('relation_type', 'general')
        relation_type_weight = 1.0
        if relation_type == 'causal':
            relation_type_weight = 0.5  # 因果关系权重更低（更好）
        elif relation_type == 'temporal':
            relation_type_weight = 0.7  # 时间关系权重次低
        
        # 计算综合权重 - 转换为越小越好的权重值
        composite_score = (causal_strength * 0.4 + confidence * 0.2 + reliability * 0.2 + 
                          recency_factor * 0.1 + domain_weight * 0.05 + relation_type_weight * 0.05)
        
        # 转换为权重（越小越好）
        return 1.0 / composite_score if composite_score > 0 else 10.0
    
    def _calculate_path_score(self, path_strength: float, path_confidence: float, 
                            path_length: int, path_info: List[Dict[str, Any]], 
                            path_consistency: float = 1.0, path_specificity: float = 0.0, 
                            evolutionary_factor: float = 0.1) -> float:
        """计算路径的综合评分 - 增强版，支持动态演化和自适应学习
        
        返回值: 评分值（越大越好）
        """
        # 1. 基础评分 - 路径强度、置信度和一致性
        base_score = (path_strength * 0.5 + path_confidence * 0.3 + path_consistency * 0.2)
        
        # 2. 长度惩罚 - 路径越短，评分越高，但也要考虑路径完整性
        if path_length == 2:  # 直接因果关系
            length_penalty = 1.0
        elif path_length < 5:
            length_penalty = max(0.7, 1.0 - (path_length - 2) * 0.1)
        else:
            length_penalty = max(0.4, 1.0 - (path_length - 2) * 0.15)
        
        # 3. 因果密度评分 - 路径中因果关系的比例
        causal_relations = sum(1 for step in path_info if step['relation_type'] == 'causal')
        causal_density = causal_relations / len(path_info) if path_info else 0.0
        
        # 4. 领域专业性评分 - 专业领域知识权重更高
        domain_score = 0.0
        if path_info:
            domain_weights = [self._get_domain_weight(step['domain']) for step in path_info]
            domain_score = sum(domain_weights) / len(domain_weights)
        
        # 5. 时效性评分 - 知识越新，权重越高
        recency_scores = []
        for step in path_info:
            if 'created_at' in step:
                age_days = (datetime.now() - step['created_at']).days
                recency_scores.append(max(0.2, 1.0 - age_days / 365.0))
        recency_score = sum(recency_scores) / len(recency_scores) if recency_scores else 0.5
        
        # 6. 来源可靠性评分
        reliability_scores = [step.get('reliability', 0.5) for step in path_info]
        source_reliability = sum(reliability_scores) / len(reliability_scores) if reliability_scores else 0.5
        
        # 7. 演化潜力评分
        evolutionary_potential = self._calculate_evolutionary_potential(path_info)
        
        # 8. 多样性平衡 - 避免过度单一的路径
        relation_types = [step['relation_type'] for step in path_info]
        relation_diversity = len(set(relation_types)) / len(relation_types) if relation_types else 1.0
        domain_diversity = len(set(step['domain'] for step in path_info)) / len(path_info) if path_info else 1.0
        diversity_balance = (relation_diversity * 0.6 + domain_diversity * 0.4)
        
        # 9. 计算最终评分 - 综合考虑所有因素
        final_score = ((base_score * 0.4 + 
                      causal_density * 0.2 + 
                      domain_score * 0.15 + 
                      recency_score * 0.1 + 
                      source_reliability * 0.1 + 
                      evolutionary_potential * evolutionary_factor) * 
                      length_penalty * diversity_balance)
        
        return final_score
    
    def _calculate_evolutionary_potential(self, path_info: List[Dict[str, Any]]) -> float:
        """计算路径的演化潜力 - 评估路径的可扩展性和适应性
        
        返回值: 演化潜力分数（0-1之间）
        """
        if not path_info:
            return 0.0
        
        # 1. 知识新鲜度评估
        recency_scores = []
        for step in path_info:
            if 'created_at' in step:
                age_days = (datetime.now() - step['created_at']).days
                recency_scores.append(max(0.1, 1.0 - age_days / 180.0))  # 6个月内的知识有较高潜力
        recency_potential = sum(recency_scores) / len(recency_scores) if recency_scores else 0.5
        
        # 2. 关系类型多样性
        relation_types = set(step['relation_type'] for step in path_info)
        type_diversity = min(1.0, len(relation_types) / 3.0)  # 支持多种关系类型的路径更具扩展性
        
        # 3. 领域覆盖广度
        domains = set(step['domain'] for step in path_info)
        domain_coverage = min(1.0, len(domains) / 2.0)  # 跨领域路径更具适应性
        
        # 4. 知识来源多样性
        sources = set(step['source'] for step in path_info)
        source_diversity = min(1.0, len(sources) / 2.0)  # 多源知识更可靠且适应性强
        
        # 5. 因果关系密度与质量
        causal_steps = [step for step in path_info if step['relation_type'] == 'causal']
        causal_potential = 0.0
        if causal_steps:
            causal_strengths = [step['strength'] for step in causal_steps]
            causal_potential = sum(causal_strengths) / len(causal_strengths)
        
        # 综合演化潜力
        evolutionary_potential = (recency_potential * 0.3 + 
                                 type_diversity * 0.2 + 
                                 domain_coverage * 0.2 + 
                                 source_diversity * 0.15 + 
                                 causal_potential * 0.15)
        
        return evolutionary_potential
    
    def _evolve_knowledge_graph(self, top_paths: List[Dict[str, Any]], evolutionary_factor: float = 0.1):
        """动态演化知识图谱 - 根据路径评估结果更新知识图谱权重
        
        参数:
            top_paths: 经过排序的顶级路径列表
            evolutionary_factor: 演化因子，控制更新强度
        """
        if not top_paths:
            return
        
        # 1. 更新路径中节点的置信度
        for path in top_paths:
            if path['score'] < 0.3:  # 只演化高质量路径
                continue
            
            path_info = path['path']
            path_score = path['score']
            
            for step in path_info:
                # 查找对应的知识节点
                for u, v, d in self.knowledge_graph.edges(data=True):
                    if d.get('subject') == step['from'] and \
                       d.get('object') == step['to'] and \
                       d.get('relation') == step['relation']:
                        
                        # 增强因果强度
                        current_causal_strength = d.get('causal_strength', 0.5)
                        new_causal_strength = min(1.0, current_causal_strength + (1.0 - current_causal_strength) * evolutionary_factor * path_score)
                        d['causal_strength'] = new_causal_strength
                        
                        # 增强置信度
                        current_confidence = d.get('confidence', 0.5)
                        new_confidence = min(1.0, current_confidence + (1.0 - current_confidence) * evolutionary_factor * path_score * 0.7)
                        d['confidence'] = new_confidence
                        
                        # 更新最后使用时间
                        d['last_used'] = datetime.now()
                        
                        break
        
        # 2. 更新性能指标
        self.performance_metrics['evolutionary_updates'] = self.performance_metrics.get('evolutionary_updates', 0) + 1
        self.performance_metrics['top_path_scores'] = [path['score'] for path in top_paths[:3]]
    
    def _get_domain_weight(self, domain: str) -> float:
        """获取领域权重 - 重要领域权重更高"""
        domain_weights = {
            'medicine': 0.9,
            'physics': 0.85,
            'chemistry': 0.8,
            'biology': 0.8,
            'computer_science': 0.85,
            'mathematics': 0.9,
            'law': 0.75,
            'economics': 0.7,
            'psychology': 0.75,
            'sociology': 0.65,
            'history': 0.6,
            'humanities': 0.55,
            'general': 0.5
        }
        
        return domain_weights.get(domain, 0.5)
    
    def commonsense_inference(self, statement: str, 
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        常识推理 - 判断陈述的合理性
        """
        # 解析陈述
        parsed = self._parse_statement(statement)
        if not parsed:
            return {"valid": False, "confidence": 0.0, "reason": "无法解析陈述", "explanation": "无法解析输入的陈述"}
        
        subject, relation, obj = parsed
        
        # 查找相关知识
        subject_knowledge = self.semantic_query(subject, max_results=3)
        object_knowledge = self.semantic_query(obj, max_results=3)
        
        if not subject_knowledge or not object_knowledge:
            return {"valid": False, "confidence": 0.0, "reason": "缺乏相关知识", "explanation": "缺乏与主题或对象相关的知识"}
        
        # 计算合理性分数
        validity_score = self._calculate_validity_score(subject, relation, obj, 
                                                       subject_knowledge, object_knowledge)
        
        return {
            "valid": validity_score > 0.6,
            "confidence": validity_score,
            "subject": subject,
            "relation": relation,
            "object": obj,
            "explanation": self._generate_explanation(validity_score, subject_knowledge, object_knowledge)
        }
    
    def _parse_statement(self, statement: str) -> Optional[Tuple[str, str, str]]:
        """解析自然语言陈述"""
        # 简单的解析逻辑（实际应使用NLP解析器）
        words = statement.lower().split()
        if len(words) < 3:
            return None
        
        # 尝试提取主谓宾
        subject = words[0]
        object = words[-1]
        relation = " ".join(words[1:-1])
        
        return subject, relation, object
    
    def _calculate_validity_score(self, subject: str, relation: str, object: str,
                                subject_knowledge: List[Dict[str, Any]],
                                object_knowledge: List[Dict[str, Any]]) -> float:
        """计算合理性分数"""
        # 基于语义相似度和知识置信度
        subject_sim = max([item['similarity'] for item in subject_knowledge]) if subject_knowledge else 0.0
        object_sim = max([item['similarity'] for item in object_knowledge]) if object_knowledge else 0.0
        
        # 查找特定关系
        relation_strength = 0.0
        for item in subject_knowledge:
            if item['relation'] == relation:
                relation_strength = max(relation_strength, item['confidence'])
        
        return min(1.0, (subject_sim * 0.4 + object_sim * 0.3 + relation_strength * 0.3))
    
    def _generate_explanation(self, validity_score: float,
                             subject_knowledge: List[Dict[str, Any]],
                             object_knowledge: List[Dict[str, Any]]) -> str:
        """生成解释"""
        if validity_score > 0.8:
            return "基于强相关知识和高度置信的推理"
        elif validity_score > 0.6:
            return "基于相关知识和中等置信度的推理"
        else:
            return "缺乏足够证据或存在矛盾知识"
    
    def knowledge_fusion(self, knowledge_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        知识融合 - 整合多源知识
        """
        self.performance_metrics["knowledge_fusions"] += 1
        
        fused_knowledge = {
            'combined_confidence': 0.0,
            'sources_agreement': 0.0,
            'integrated_representation': None,
            'conflicts': []
        }
        
        if not knowledge_sources:
            return fused_knowledge
        
        # 计算平均置信度
        confidences = [source.get('confidence', 0.5) for source in knowledge_sources]
        fused_knowledge['combined_confidence'] = sum(confidences) / len(confidences)
        
        # 检查一致性
        unique_objects = len(set(source.get('object', '') for source in knowledge_sources))
        fused_knowledge['sources_agreement'] = 1.0 - (unique_objects - 1) / len(knowledge_sources)
        
        # 检测冲突
        if unique_objects > 1:
            fused_knowledge['conflicts'] = [{
                'type': 'object_conflict',
                'severity': (unique_objects - 1) / len(knowledge_sources)
            }]
        
        return fused_knowledge
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            **self.performance_metrics,
            "knowledge_graph_size": len(self.knowledge_graph),
            "semantic_index_size": len(self.semantic_index),
            "causal_relationships": len(self.causal_relationships)
        }

# 全局AGI知识整合器实例 - 启用从零开始训练模式以避免网络请求
agi_knowledge_integrator = AGIKnowledgeIntegrator(from_scratch=True)

if __name__ == "__main__":
    # 测试增强型知识整合器
    print("=== 测试AGI知识整合器 ===")
    
    # 使用从零开始训练模式避免网络请求
    integrator = AGIKnowledgeIntegrator(from_scratch=True)
    
    # 添加知识
    print("\n1. 添加知识...")
    knowledge_id1 = integrator.add_knowledge("水", "可以溶解", "盐", confidence=0.9, domain="chemistry")
    print(f"添加知识: 水 可以溶解 盐 (ID: {knowledge_id1}, 领域: chemistry)")
    
    knowledge_id2 = integrator.add_knowledge("盐", "是一种", "化合物", confidence=0.95, domain="chemistry")
    print(f"添加知识: 盐 是一种 化合物 (ID: {knowledge_id2}, 领域: chemistry)")
    
    knowledge_id3 = integrator.add_knowledge("水", "可以溶解", "糖", confidence=0.92, domain="chemistry")
    print(f"添加知识: 水 可以溶解 糖 (ID: {knowledge_id3}, 领域: chemistry)")
    
    knowledge_id4 = integrator.add_knowledge("糖", "可以被", "身体吸收", confidence=0.88, domain="biology")
    print(f"添加知识: 糖 可以被 身体吸收 (ID: {knowledge_id4}, 领域: biology)")
    
    knowledge_id5 = integrator.add_knowledge("身体吸收", "会产生", "能量", confidence=0.85, domain="biology")
    print(f"添加知识: 身体吸收 会产生 能量 (ID: {knowledge_id5}, 领域: biology)")
    
    knowledge_id6 = integrator.add_knowledge("能量", "可以提供", "动力", confidence=0.9, domain="physics")
    print(f"添加知识: 能量 可以提供 动力 (ID: {knowledge_id6}, 领域: physics)")
    
    # 语义查询
    print("\n2. 语义查询...")
    # 普通查询
    results = integrator.semantic_query("液体溶解", max_results=3)
    for result in results:
        print(f"概念: {result['concept']}, 关系: {result['relation']}, 相似度: {result['similarity']:.2f}, 增强评分: {result['enhanced_score']:.2f}")
    
    # 使用领域权重的查询
    print("\n2.1 使用领域权重的语义查询...")
    domain_weights = {'chemistry': 1.5, 'biology': 1.2}
    weighted_results = integrator.semantic_query("溶解", max_results=3, domain_weights=domain_weights)
    for result in weighted_results:
        print(f"概念: {result['concept']}, 领域: {result['domain']}, 关系: {result['relation']}, 增强评分: {result['enhanced_score']:.2f}")
    
    # 因果推理
    print("\n3. 因果推理...")
    causal_result = integrator.causal_reasoning("水", "能量")
    print(f"因果强度: {causal_result['strength']:.4f}, 置信度: {causal_result['confidence']:.4f}")
    print(f"发现路径数: {len(causal_result['paths'])}")
    
    # 常识推理
    print("\n4. 常识推理...")
    inference = integrator.commonsense_inference("水可以溶解糖")
    print(f"推理结果: 有效={inference['valid']}, 置信度={inference['confidence']:.4f}")
    print(f"解释: {inference['explanation']}")
    
    # 显示性能指标
    print("\n5. 性能指标:")
    metrics = integrator.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # 测试因果路径
    if causal_result['paths']:
        print("\n6. 因果路径分析:")
        for i, path in enumerate(causal_result['paths'][:3], 1):
            print(f"\n路径 {i}:")
            print(f"  强度: {path['strength']:.4f}, 置信度: {path['confidence']:.4f}, 评分: {path['score']:.4f}, 长度: {path['length']}")
            print("  路径详情:")
            for step in path['path']:
                print(f"    {step['from']} → {step['relation']} → {step['to']} (强度: {step['strength']:.2f}, 置信度: {step['confidence']:.2f})")
    
    print("\n=== 测试完成 ===")
