"""
语义关系图谱

构建和维护跨模态语义关系图谱，支持：
1. 概念节点创建和更新
2. 跨模态关系建立
3. 语义路径查询
4. 关系强度学习
5. 图谱可视化和分析

解决当前系统缺乏跨模态语义关联的问题。
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import logging
import json
import time
from collections import defaultdict, deque
import math

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("semantic_relation_graph")


class ConceptNode:
    """概念节点，表示跨模态语义概念"""
    
    def __init__(self, concept_id: str, concept_name: str, 
                 modality_types: List[str], embedding: np.ndarray):
        """
        初始化概念节点
        
        Args:
            concept_id: 概念ID
            concept_name: 概念名称
            modality_types: 涉及的模态类型列表
            embedding: 概念嵌入向量
        """
        self.concept_id = concept_id
        self.concept_name = concept_name
        self.modality_types = modality_types
        self.embedding = embedding
        self.metadata = {
            "created_at": time.time(),
            "update_count": 0,
            "access_count": 0,
            "confidence": 1.0
        }
        
        # 模态特定的信息
        self.modality_info = {modality: {} for modality in modality_types}
    
    def update_embedding(self, new_embedding: np.ndarray, learning_rate: float = 0.1):
        """更新概念嵌入"""
        self.embedding = self.embedding * (1 - learning_rate) + new_embedding * learning_rate
        self.metadata["update_count"] += 1
        self.metadata["confidence"] = min(1.0, self.metadata["confidence"] + 0.01)
    
    def add_modality_info(self, modality_type: str, info: Dict[str, Any]):
        """添加模态特定信息"""
        if modality_type in self.modality_info:
            self.modality_info[modality_type].update(info)
        else:
            self.modality_info[modality_type] = info
    
    def get_similarity(self, other_embedding: np.ndarray) -> float:
        """计算与另一个嵌入的相似度"""
        # 余弦相似度
        norm_a = np.linalg.norm(self.embedding)
        norm_b = np.linalg.norm(other_embedding)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(self.embedding, other_embedding) / (norm_a * norm_b)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "modality_types": self.modality_types,
            "embedding_shape": self.embedding.shape,
            "embedding_norm": float(np.linalg.norm(self.embedding)),
            "metadata": self.metadata,
            "modality_info": self.modality_info
        }


class SemanticRelation:
    """语义关系，连接两个概念节点"""
    
    def __init__(self, relation_id: str, source_id: str, target_id: str,
                 relation_type: str, strength: float = 1.0):
        """
        初始化语义关系
        
        Args:
            relation_id: 关系ID
            source_id: 源节点ID
            target_id: 目标节点ID
            relation_type: 关系类型
            strength: 关系强度 (0-1)
        """
        self.relation_id = relation_id
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.strength = strength
        self.metadata = {
            "created_at": time.time(),
            "update_count": 0,
            "access_count": 0,
            "modality_crossing": []  # 该关系涉及的模态交叉
        }
    
    def update_strength(self, new_strength: float, learning_rate: float = 0.1):
        """更新关系强度"""
        self.strength = self.strength * (1 - learning_rate) + new_strength * learning_rate
        self.strength = max(0.0, min(1.0, self.strength))
        self.metadata["update_count"] += 1
    
    def add_modality_crossing(self, source_modality: str, target_modality: str):
        """添加模态交叉信息"""
        crossing = f"{source_modality}→{target_modality}"
        if crossing not in self.metadata["modality_crossing"]:
            self.metadata["modality_crossing"].append(crossing)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "relation_id": self.relation_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "strength": float(self.strength),
            "metadata": self.metadata
        }


class SemanticRelationGraph:
    """
    语义关系图谱
    
    构建和维护跨模态语义关系，支持概念查询和关系推理。
    """
    
    def __init__(self, embedding_dim: int = 768, max_nodes: int = 10000,
                 similarity_threshold: float = 0.7):
        """
        初始化语义关系图谱
        
        Args:
            embedding_dim: 嵌入维度
            max_nodes: 最大节点数
            similarity_threshold: 相似度阈值，用于判断是否为同一概念
        """
        self.embedding_dim = embedding_dim
        self.max_nodes = max_nodes
        self.similarity_threshold = similarity_threshold
        
        # 概念节点存储
        self.concepts: Dict[str, ConceptNode] = {}
        
        # 语义关系存储
        self.relations: Dict[str, SemanticRelation] = {}
        
        # 概念名称到ID的映射
        self.name_to_id: Dict[str, str] = {}
        
        # 网络图表示（用于复杂查询）
        self.graph = nx.Graph()
        
        # 统计信息
        self.stats = {
            "total_concepts": 0,
            "total_relations": 0,
            "updates": 0,
            "queries": 0,
            "hits": 0,
            "misses": 0
        }
        
        # 预定义关系类型
        self.relation_types = [
            "is_a",          # 是一个（上位关系）
            "has_a",         # 有一个（部分关系）
            "related_to",    # 相关
            "similar_to",    # 相似
            "opposite_of",   # 相反
            "part_of",       # 部分
            "used_for",      # 用于
            "made_of",       # 由...制成
            "located_in",    # 位于
            "causes",        # 导致
            "modality_link"  # 模态链接（跨模态关联）
        ]
        
        # 初始化一些基础概念
        self._initialize_basic_concepts()
        
        logger.info(f"语义关系图谱初始化完成，嵌入维度: {embedding_dim}, 最大节点数: {max_nodes}")
    
    def _initialize_basic_concepts(self):
        """初始化一些基础概念"""
        basic_concepts = [
            ("object", "object", ["text", "image"]),
            ("color", "color", ["text", "image"]),
            ("shape", "shape", ["text", "image"]),
            ("action", "action", ["text", "audio"]),
            ("emotion", "emotion", ["text", "audio", "image"]),
            ("location", "location", ["text", "image"]),
            ("time", "time", ["text"]),
            ("quantity", "quantity", ["text"]),
            ("quality", "quality", ["text", "image", "audio"]),
        ]
        
        for concept_id, concept_name, modalities in basic_concepts:
            # 创建随机嵌入（实际应用中应该使用预训练嵌入）
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            
            self.add_concept(concept_id, concept_name, modalities, embedding)
        
        logger.info(f"初始化了{len(basic_concepts)}个基础概念")
    
    def add_concept(self, concept_id: str, concept_name: str, 
                   modality_types: List[str], embedding: np.ndarray) -> bool:
        """
        添加概念节点
        
        Args:
            concept_id: 概念ID
            concept_name: 概念名称
            modality_types: 模态类型列表
            embedding: 概念嵌入
            
        Returns:
            是否添加成功
        """
        if concept_id in self.concepts:
            logger.warning(f"概念 {concept_id} 已存在，将更新")
            return self.update_concept(concept_id, embedding)
        
        if len(self.concepts) >= self.max_nodes:
            logger.warning(f"达到最大节点数 {self.max_nodes}，无法添加新概念")
            return False
        
        # 创建概念节点
        concept = ConceptNode(concept_id, concept_name, modality_types, embedding)
        self.concepts[concept_id] = concept
        self.name_to_id[concept_name] = concept_id
        
        # 添加到网络图
        self.graph.add_node(concept_id, 
                           name=concept_name,
                           modalities=modality_types,
                           embedding_norm=np.linalg.norm(embedding))
        
        self.stats["total_concepts"] += 1
        logger.debug(f"添加概念: {concept_name} ({concept_id})")
        
        return True
    
    def update_concept(self, concept_id: str, new_embedding: np.ndarray,
                      learning_rate: float = 0.1) -> bool:
        """
        更新概念节点
        
        Args:
            concept_id: 概念ID
            new_embedding: 新的嵌入向量
            learning_rate: 学习率
            
        Returns:
            是否更新成功
        """
        if concept_id not in self.concepts:
            logger.warning(f"概念 {concept_id} 不存在")
            return False
        
        concept = self.concepts[concept_id]
        concept.update_embedding(new_embedding, learning_rate)
        
        self.stats["updates"] += 1
        logger.debug(f"更新概念: {concept.concept_name} ({concept_id})")
        
        return True
    
    def find_similar_concept(self, embedding: np.ndarray, 
                            modality_filter: Optional[List[str]] = None) -> Tuple[Optional[str], float]:
        """
        查找与给定嵌入最相似的概念
        
        Args:
            embedding: 查询嵌入
            modality_filter: 模态过滤器，只考虑这些模态的概念
            
        Returns:
            (概念ID, 相似度) 或 (None, 0.0)
        """
        self.stats["queries"] += 1
        
        best_concept_id = None
        best_similarity = -1.0
        
        for concept_id, concept in self.concepts.items():
            # 应用模态过滤器
            if modality_filter and not any(modality in concept.modality_types for modality in modality_filter):
                continue
            
            similarity = concept.get_similarity(embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_concept_id = concept_id
        
        if best_similarity >= self.similarity_threshold:
            self.stats["hits"] += 1
            logger.debug(f"找到相似概念: {best_concept_id}, 相似度: {best_similarity:.4f}")
        else:
            self.stats["misses"] += 1
            logger.debug(f"未找到足够相似的概念，最高相似度: {best_similarity:.4f}")
        
        return best_concept_id, best_similarity
    
    def add_relation(self, source_id: str, target_id: str, 
                    relation_type: str, strength: float = 1.0,
                    modality_crossing: Optional[Tuple[str, str]] = None) -> bool:
        """
        添加语义关系
        
        Args:
            source_id: 源概念ID
            target_id: 目标概念ID
            relation_type: 关系类型
            strength: 关系强度
            modality_crossing: 模态交叉信息 (源模态, 目标模态)
            
        Returns:
            是否添加成功
        """
        if source_id not in self.concepts or target_id not in self.concepts:
            logger.warning(f"概念不存在: source={source_id}, target={target_id}")
            return False
        
        if relation_type not in self.relation_types:
            logger.warning(f"不支持的关系类型: {relation_type}")
            return False
        
        # 生成关系ID
        relation_id = f"{source_id}_{relation_type}_{target_id}"
        
        if relation_id in self.relations:
            logger.debug(f"关系已存在: {relation_id}，将更新强度")
            relation = self.relations[relation_id]
            relation.update_strength(strength)
            return True
        
        # 创建关系
        relation = SemanticRelation(relation_id, source_id, target_id, relation_type, strength)
        
        # 添加模态交叉信息
        if modality_crossing:
            source_modality, target_modality = modality_crossing
            relation.add_modality_crossing(source_modality, target_modality)
        
        self.relations[relation_id] = relation
        
        # 添加到网络图
        self.graph.add_edge(source_id, target_id,
                           relation_type=relation_type,
                           strength=strength,
                           relation_id=relation_id,
                           modality_crossing=modality_crossing)
        
        self.stats["total_relations"] += 1
        logger.debug(f"添加关系: {self.concepts[source_id].concept_name} "
                    f"{relation_type} {self.concepts[target_id].concept_name}")
        
        return True
    
    def update_relation(self, relation_id: str, new_strength: float,
                       learning_rate: float = 0.1) -> bool:
        """
        更新语义关系强度
        
        Args:
            relation_id: 关系ID
            new_strength: 新的关系强度
            learning_rate: 学习率
            
        Returns:
            是否更新成功
        """
        if relation_id not in self.relations:
            logger.warning(f"关系 {relation_id} 不存在")
            return False
        
        relation = self.relations[relation_id]
        relation.update_strength(new_strength, learning_rate)
        
        # 更新网络图中的边属性
        if self.graph.has_edge(relation.source_id, relation.target_id):
            self.graph[relation.source_id][relation.target_id]["strength"] = relation.strength
        
        self.stats["updates"] += 1
        logger.debug(f"更新关系: {relation_id}, 新强度: {relation.strength:.4f}")
        
        return True
    
    def find_relations(self, concept_id: str, relation_type: Optional[str] = None,
                      min_strength: float = 0.0) -> List[Dict[str, Any]]:
        """
        查找与概念相关的所有关系
        
        Args:
            concept_id: 概念ID
            relation_type: 关系类型过滤器
            min_strength: 最小关系强度
            
        Returns:
            关系信息列表
        """
        if concept_id not in self.concepts:
            return []
        
        related_relations = []
        
        for relation_id, relation in self.relations.items():
            if relation.source_id == concept_id or relation.target_id == concept_id:
                if relation.strength < min_strength:
                    continue
                
                if relation_type and relation.relation_type != relation_type:
                    continue
                
                # 确定另一个概念
                other_id = relation.target_id if relation.source_id == concept_id else relation.source_id
                other_concept = self.concepts[other_id]
                
                relation_info = {
                    "relation_id": relation_id,
                    "relation_type": relation.relation_type,
                    "strength": relation.strength,
                    "other_concept_id": other_id,
                    "other_concept_name": other_concept.concept_name,
                    "other_modalities": other_concept.modality_types,
                    "direction": "outgoing" if relation.source_id == concept_id else "incoming",
                    "modality_crossing": relation.metadata["modality_crossing"]
                }
                
                related_relations.append(relation_info)
        
        return related_relations
    
    def find_semantic_path(self, source_id: str, target_id: str,
                          max_path_length: int = 3) -> List[List[str]]:
        """
        查找两个概念之间的语义路径
        
        Args:
            source_id: 源概念ID
            target_id: 目标概念ID
            max_path_length: 最大路径长度
            
        Returns:
            路径列表，每个路径是概念ID列表
        """
        if source_id not in self.concepts or target_id not in self.concepts:
            return []
        
        try:
            # 使用networkx查找所有简单路径
            all_paths = list(nx.all_simple_paths(self.graph, source_id, target_id, 
                                                cutoff=max_path_length))
            
            # 按路径长度排序
            all_paths.sort(key=len)
            
            # 转换为概念ID列表
            paths = []
            for path in all_paths[:10]:  # 最多返回10条路径
                paths.append(list(path))
            
            return paths
            
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            logger.error(f"查找语义路径时出错: {e}")
            return []
    
    def update(self, modality_types: List[str], features: List[torch.Tensor],
              unified_representation: torch.Tensor) -> Dict[str, Any]:
        """
        根据多模态特征更新语义关系图谱
        
        Args:
            modality_types: 模态类型列表
            features: 模态特征列表
            unified_representation: 统一语义表示
            
        Returns:
            更新信息字典
        """
        update_info = {
            "new_concepts": 0,
            "updated_concepts": 0,
            "new_relations": 0,
            "updated_relations": 0,
            "concept_similarities": []
        }
        
        batch_size, seq_len, dim = unified_representation.shape
        
        # 处理每个序列位置
        for seq_idx in range(min(seq_len, 5)):  # 最多处理5个位置
            # 提取当前位置的统一表示
            unified_embedding = unified_representation[:, seq_idx, :].mean(dim=0).detach().cpu().numpy()
            
            # 查找或创建概念
            concept_id, similarity = self.find_similar_concept(unified_embedding, modality_types)
            
            update_info["concept_similarities"].append({
                "seq_idx": seq_idx,
                "similarity": float(similarity),
                "found_concept": concept_id is not None
            })
            
            if concept_id is None or similarity < self.similarity_threshold:
                # 创建新概念
                new_concept_id = f"concept_{len(self.concepts) + 1}"
                concept_name = f"concept_{len(self.concepts) + 1}"
                
                if self.add_concept(new_concept_id, concept_name, modality_types, unified_embedding):
                    update_info["new_concepts"] += 1
                    concept_id = new_concept_id
                else:
                    continue
            else:
                # 更新现有概念
                self.update_concept(concept_id, unified_embedding)
                update_info["updated_concepts"] += 1
            
            # 建立模态间关系
            if len(modality_types) > 1:
                for i in range(len(modality_types)):
                    for j in range(i + 1, len(modality_types)):
                        modality_a = modality_types[i]
                        modality_b = modality_types[j]
                        
                        # 添加模态交叉关系
                        relation_type = "modality_link"
                        relation_id = f"{concept_id}_{modality_a}_to_{modality_b}"
                        
                        if self.add_relation(concept_id, concept_id, relation_type,
                                           strength=similarity,
                                           modality_crossing=(modality_a, modality_b)):
                            update_info["new_relations"] += 1
        
        logger.info(f"语义关系图谱更新: {update_info['new_concepts']}新概念, "
                   f"{update_info['updated_concepts']}更新概念, "
                   f"{update_info['new_relations']}新关系")
        
        return update_info
    
    def query_cross_modal_association(self, source_concept: str, source_modality: str,
                                     target_modality: str) -> List[Dict[str, Any]]:
        """
        查询跨模态关联
        
        Args:
            source_concept: 源概念名称
            source_modality: 源模态类型
            target_modality: 目标模态类型
            
        Returns:
            关联信息列表
        """
        if source_concept not in self.name_to_id:
            return []
        
        concept_id = self.name_to_id[source_concept]
        concept = self.concepts.get(concept_id)
        
        if not concept:
            return []
        
        # 查找所有包含目标模态的相关概念
        associations = []
        
        for relation_info in self.find_relations(concept_id):
            other_concept = self.concepts[relation_info["other_concept_id"]]
            
            # 检查是否涉及目标模态
            if target_modality in other_concept.modality_types:
                # 检查关系是否涉及模态交叉
                modality_crossing = relation_info.get("modality_crossing", [])
                relevant_crossing = any(
                    f"{source_modality}→{target_modality}" in crossing or
                    f"{target_modality}→{source_modality}" in crossing
                    for crossing in modality_crossing
                )
                
                if relevant_crossing:
                    association = {
                        "source_concept": concept.concept_name,
                        "source_modality": source_modality,
                        "target_concept": other_concept.concept_name,
                        "target_modality": target_modality,
                        "relation_type": relation_info["relation_type"],
                        "strength": relation_info["strength"],
                        "concept_similarity": concept.get_similarity(other_concept.embedding)
                    }
                    associations.append(association)
        
        # 按关联强度排序
        associations.sort(key=lambda x: x["strength"], reverse=True)
        
        return associations
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        # 计算模态分布
        modality_distribution = defaultdict(int)
        for concept in self.concepts.values():
            for modality in concept.modality_types:
                modality_distribution[modality] += 1
        
        # 计算关系类型分布
        relation_type_distribution = defaultdict(int)
        for relation in self.relations.values():
            relation_type_distribution[relation.relation_type] += 1
        
        # 计算图谱密度
        n_nodes = len(self.concepts)
        n_edges = len(self.relations)
        max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 0
        density = n_edges / max_edges if max_edges > 0 else 0
        
        # 计算平均关系强度
        avg_strength = np.mean([r.strength for r in self.relations.values()]) if self.relations else 0
        
        stats = {
            "total_concepts": self.stats["total_concepts"],
            "total_relations": self.stats["total_relations"],
            "concepts_by_modality": dict(modality_distribution),
            "relations_by_type": dict(relation_type_distribution),
            "graph_density": density,
            "average_relation_strength": float(avg_strength),
            "query_hit_rate": self.stats["hits"] / max(self.stats["queries"], 1),
            "updates": self.stats["updates"],
            "networkx_graph_info": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "connected_components": nx.number_connected_components(self.graph)
            }
        }
        
        return stats
    
    def save_to_file(self, filepath: str) -> bool:
        """保存图谱到文件"""
        try:
            data = {
                "concepts": {cid: concept.to_dict() for cid, concept in self.concepts.items()},
                "relations": {rid: relation.to_dict() for rid, relation in self.relations.items()},
                "name_to_id": self.name_to_id,
                "stats": self.stats,
                "metadata": {
                    "embedding_dim": self.embedding_dim,
                    "max_nodes": self.max_nodes,
                    "similarity_threshold": self.similarity_threshold,
                    "saved_at": time.time()
                }
            }
            
            # 转换为JSON可序列化格式
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.float32):
                    return float(obj)
                if isinstance(obj, np.int64):
                    return int(obj)
                return obj
            
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, default=convert_numpy, indent=2, ensure_ascii=False)
            
            logger.info(f"语义关系图谱保存到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"保存语义关系图谱失败: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """从文件加载图谱"""
        try:
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 清空现有数据
            self.concepts.clear()
            self.relations.clear()
            self.name_to_id.clear()
            self.graph.clear()
            
            # 加载概念
            for cid, concept_data in data["concepts"].items():
                concept = ConceptNode(
                    concept_id=cid,
                    concept_name=concept_data["concept_name"],
                    modality_types=concept_data["modality_types"],
                    embedding=np.array(concept_data.get("embedding", np.random.randn(self.embedding_dim)))
                )
                concept.metadata = concept_data.get("metadata", {})
                concept.modality_info = concept_data.get("modality_info", {})
                self.concepts[cid] = concept
                self.name_to_id[concept.concept_name] = cid
            
            # 加载关系
            for rid, relation_data in data["relations"].items():
                relation = SemanticRelation(
                    relation_id=rid,
                    source_id=relation_data["source_id"],
                    target_id=relation_data["target_id"],
                    relation_type=relation_data["relation_type"],
                    strength=relation_data["strength"]
                )
                relation.metadata = relation_data.get("metadata", {})
                self.relations[rid] = relation
            
            # 加载统计信息
            self.stats = data.get("stats", self.stats)
            
            # 重建网络图
            for cid, concept in self.concepts.items():
                self.graph.add_node(cid, 
                                   name=concept.concept_name,
                                   modalities=concept.modality_types,
                                   embedding_norm=np.linalg.norm(concept.embedding))
            
            for rid, relation in self.relations.items():
                self.graph.add_edge(relation.source_id, relation.target_id,
                                   relation_type=relation.relation_type,
                                   strength=relation.strength,
                                   relation_id=rid,
                                   modality_crossing=relation.metadata.get("modality_crossing", []))
            
            logger.info(f"从文件加载语义关系图谱: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"加载语义关系图谱失败: {e}")
            return False
    
    def visualize(self, max_nodes: int = 50) -> Dict[str, Any]:
        """
        可视化图谱（返回简化版本用于显示）
        
        Args:
            max_nodes: 最大显示节点数
            
        Returns:
            可视化数据
        """
        if not self.concepts:
            return {"nodes": [], "edges": []}
        
        # 选择最重要的节点（基于连接度）
        if len(self.concepts) > max_nodes:
            # 按连接度排序
            node_degrees = [(node, self.graph.degree(node)) for node in self.graph.nodes()]
            node_degrees.sort(key=lambda x: x[1], reverse=True)
            important_nodes = [node for node, _ in node_degrees[:max_nodes]]
        else:
            important_nodes = list(self.concepts.keys())
        
        # 构建节点列表
        nodes = []
        for node_id in important_nodes:
            concept = self.concepts[node_id]
            nodes.append({
                "id": node_id,
                "label": concept.concept_name,
                "modalities": concept.modality_types,
                "size": min(10 + self.graph.degree(node_id) * 2, 30)
            })
        
        # 构建边列表（只连接重要节点）
        edges = []
        for relation in self.relations.values():
            if relation.source_id in important_nodes and relation.target_id in important_nodes:
                edges.append({
                    "source": relation.source_id,
                    "target": relation.target_id,
                    "label": relation.relation_type,
                    "strength": relation.strength,
                    "width": relation.strength * 3
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(self.concepts),
            "total_edges": len(self.relations),
            "displayed_nodes": len(nodes),
            "displayed_edges": len(edges)
        }


def test_semantic_relation_graph():
    """测试语义关系图谱"""
    logger.info("测试语义关系图谱...")
    
    try:
        # 创建图谱
        graph = SemanticRelationGraph(embedding_dim=768, max_nodes=1000)
        
        # 测试添加概念
        test_embedding = np.random.randn(768).astype(np.float32)
        test_embedding = test_embedding / np.linalg.norm(test_embedding)
        
        success = graph.add_concept("test_concept_1", "杯子", ["text", "image"], test_embedding)
        assert success, "添加概念失败"
        
        # 测试查找相似概念
        similar_id, similarity = graph.find_similar_concept(test_embedding)
        assert similar_id == "test_concept_1", f"查找相似概念失败: {similar_id}"
        assert similarity > 0.99, f"相似度过低: {similarity}"
        
        # 测试添加关系
        success = graph.add_relation("object", "test_concept_1", "is_a", 0.8)
        assert success, "添加关系失败"
        
        # 测试查找关系
        relations = graph.find_relations("test_concept_1")
        assert len(relations) > 0, "查找关系失败"
        
        # 测试更新概念
        new_embedding = np.random.randn(768).astype(np.float32)
        new_embedding = new_embedding / np.linalg.norm(new_embedding)
        success = graph.update_concept("test_concept_1", new_embedding)
        assert success, "更新概念失败"
        
        # 测试查询跨模态关联
        associations = graph.query_cross_modal_association("object", "text", "image")
        # 可能没有关联，但不应该出错
        
        # 测试获取统计信息
        stats = graph.get_statistics()
        assert "total_concepts" in stats, "统计信息中缺少total_concepts"
        assert stats["total_concepts"] >= 1, f"概念数量错误: {stats['total_concepts']}"
        
        # 测试可视化
        visualization = graph.visualize(max_nodes=10)
        assert "nodes" in visualization, "可视化数据中缺少nodes"
        assert "edges" in visualization, "可视化数据中缺少edges"
        
        # 测试保存和加载（内存中）
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # 测试保存
            save_success = graph.save_to_file(temp_file)
            assert save_success, "保存图谱失败"
            
            # 创建新图谱并加载
            new_graph = SemanticRelationGraph(embedding_dim=768)
            load_success = new_graph.load_from_file(temp_file)
            assert load_success, "加载图谱失败"
            
            # 验证加载的数据
            new_stats = new_graph.get_statistics()
            assert new_stats["total_concepts"] == stats["total_concepts"], "加载后概念数量不匹配"
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        logger.info("✅ 语义关系图谱测试通过")
        
        return {
            "success": True,
            "total_concepts": stats["total_concepts"],
            "total_relations": stats["total_relations"],
            "visualization_nodes": len(visualization["nodes"]),
            "visualization_edges": len(visualization["edges"]),
            "message": "语义关系图谱测试完成"
        }
        
    except Exception as e:
        logger.error(f"❌ 语义关系图谱测试失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "语义关系图谱测试失败"
        }


# 导出主要类和方法
__all__ = [
    "SemanticRelationGraph",
    "ConceptNode",
    "SemanticRelation",
    "test_semantic_relation_graph"
]

if __name__ == "__main__":
    # 运行测试
    test_result = test_semantic_relation_graph()
    print(f"测试结果: {test_result}")

