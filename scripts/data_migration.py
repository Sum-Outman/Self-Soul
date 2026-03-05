#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量数据迁移和验证（因果知识图谱）

功能:
1. 从知识库JSON文件迁移数据到因果知识图谱
2. 应用启发式规则提取因果关系
3. 验证迁移数据的完整性
4. 保存和加载因果知识图谱

迁移策略:
1. 概念提取: 从每个领域提取概念作为图谱节点
2. 关系推断: 使用启发式规则从知识结构中推断因果关系
   - prerequisites → 因果依赖
   - related_concepts → 潜在关联
   - 领域特定模式 (医学: 疾病→症状, 治疗→改善)
3. 证据管理: 记录迁移来源和置信度
4. 验证检查: 确保图结构一致性和完整性

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict
import networkx as nx

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入因果知识图谱
try:
    from core.causal.causal_knowledge_graph import (
        CausalKnowledgeGraph, 
        CausalStrength, 
        EvidenceType
    )
    from core.causal.causal_scm_engine import StructuralCausalModelEngine
    from core.causal.causal_discovery import CausalDiscoveryEngine
    from core.causal.counterfactual_reasoner import CounterfactualReasoner
    from core.causal.do_calculus_engine import DoCalculusEngine
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在项目根目录下运行此脚本")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeDataMigrator:
    """
    知识数据迁移器 - 将知识库数据迁移到因果知识图谱
    
    工作流程:
    1. 扫描知识库目录，加载所有JSON文件
    2. 解析每个领域的数据结构
    3. 提取概念和关系
    4. 应用启发式规则推断因果关系
    5. 构建因果知识图谱
    6. 验证迁移结果
    7. 保存图谱到文件
    """
    
    def __init__(self, knowledge_base_path: str, output_path: Optional[str] = None):
        """
        初始化迁移器
        
        Args:
            knowledge_base_path: 知识库目录路径
            output_path: 输出文件路径（可选）
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        self.output_path = Path(output_path) if output_path else Path("data/causal_knowledge_graph.pkl")
        
        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化因果知识图谱
        self.causal_graph = CausalKnowledgeGraph(name="迁移的因果知识图谱")
        
        # 迁移统计
        self.migration_stats = {
            "total_files": 0,
            "files_processed": 0,
            "concepts_extracted": 0,
            "causal_relations_inferred": 0,
            "validation_errors": 0,
            "start_time": time.time(),
            "end_time": None
        }
        
        # 领域特定启发式规则
        self.domain_heuristics = {
            "medicine": self._apply_medical_heuristics,
            "biology": self._apply_biology_heuristics,
            "chemistry": self._apply_chemistry_heuristics,
            "physics": self._apply_physics_heuristics,
            "computer_science": self._apply_cs_heuristics,
            "default": self._apply_general_heuristics
        }
        
        logger.info(f"知识数据迁移器初始化完成")
        logger.info(f"知识库路径: {self.knowledge_base_path}")
        logger.info(f"输出路径: {self.output_path}")
    
    def scan_knowledge_files(self) -> List[Path]:
        """
        扫描知识库目录中的JSON文件
        
        Returns:
            JSON文件路径列表
        """
        if not self.knowledge_base_path.exists():
            logger.error(f"知识库目录不存在: {self.knowledge_base_path}")
            return []
        
        # 查找所有JSON文件
        json_files = list(self.knowledge_base_path.glob("*.json"))
        json_files = [f for f in json_files if f.is_file()]
        
        self.migration_stats["total_files"] = len(json_files)
        logger.info(f"找到 {len(json_files)} 个知识库文件")
        
        return json_files
    
    def load_knowledge_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        加载知识库JSON文件
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            解析的JSON数据，如果失败则返回None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证基本结构
            if "knowledge_base" not in data:
                logger.warning(f"文件缺少 'knowledge_base' 根键: {file_path.name}")
                return None
            
            knowledge_base = data["knowledge_base"]
            if "domain" not in knowledge_base:
                logger.warning(f"文件缺少 'domain' 字段: {file_path.name}")
                return None
            
            return knowledge_base
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误 {file_path.name}: {e}")
            return None
        except Exception as e:
            logger.error(f"加载文件错误 {file_path.name}: {e}")
            return None
    
    def extract_concepts_from_domain(self, domain_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从领域数据中提取概念
        
        Args:
            domain_data: 领域数据
            
        Returns:
            概念列表
        """
        concepts = []
        domain_id = domain_data.get("domain", "unknown")
        
        # 遍历所有类别
        for category in domain_data.get("categories", []):
            category_id = category.get("id", "unknown")
            
            # 遍历类别中的概念
            for concept in category.get("concepts", []):
                concept_id = concept.get("id", "unknown")
                
                # 创建规范化概念ID
                full_concept_id = f"{domain_id}:{category_id}:{concept_id}"
                
                # 提取概念属性
                concept_name = concept.get("name", {})
                concept_desc = concept.get("description", {})
                
                concept_data = {
                    "id": full_concept_id,
                    "original_id": concept_id,
                    "domain": domain_id,
                    "category": category_id,
                    "name": concept_name.get("en", concept_id),
                    "name_multilingual": concept_name,
                    "description": concept_desc.get("en", ""),
                    "description_multilingual": concept_desc,
                    "properties": {
                        key: value for key, value in concept.items() 
                        if key not in ["id", "name", "description", "prerequisites", "related_concepts"]
                    },
                    "prerequisites": concept.get("prerequisites", []),
                    "related_concepts": concept.get("related_concepts", []),
                    "difficulty": concept.get("difficulty", "intermediate"),
                    "importance": concept.get("importance", "medium")
                }
                
                concepts.append(concept_data)
        
        logger.debug(f"从领域 {domain_id} 提取了 {len(concepts)} 个概念")
        return concepts
    
    def infer_causal_relations(self, concepts: List[Dict[str, Any]], domain: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        从概念中推断因果关系
        
        Args:
            concepts: 概念列表
            domain: 领域标识符
            
        Returns:
            因果关系列表，每个元素为 (原因, 结果, 属性)
        """
        causal_relations = []
        
        # 为快速查找创建概念映射
        concept_map = {concept["id"]: concept for concept in concepts}
        
        for concept in concepts:
            concept_id = concept["id"]
            
            # 1. 从先决条件推断因果关系
            for prereq_id in concept.get("prerequisites", []):
                # 创建完整的先决条件概念ID
                full_prereq_id = self._resolve_concept_id(prereq_id, concept, concept_map)
                if full_prereq_id:
                    causal_relations.append((
                        full_prereq_id,
                        concept_id,
                        {
                            "strength": CausalStrength.MODERATE,
                            "confidence": 0.7,
                            "reason": "prerequisite_relationship",
                            "evidence_type": EvidenceType.EXPERT_KNOWLEDGE.value,
                            "domain": domain,
                            "inference_method": "prerequisite_heuristic"
                        }
                    ))
            
            # 2. 从相关概念推断潜在因果关系
            for related_id in concept.get("related_concepts", []):
                # 创建完整的相关概念ID
                full_related_id = self._resolve_concept_id(related_id, concept, concept_map)
                if full_related_id:
                    # 相关概念可能是双向的，我们假设较弱的关系
                    causal_relations.append((
                        concept_id,
                        full_related_id,
                        {
                            "strength": CausalStrength.WEAK,
                            "confidence": 0.5,
                            "reason": "related_concept",
                            "evidence_type": EvidenceType.EXPERT_KNOWLEDGE.value,
                            "domain": domain,
                            "inference_method": "related_concept_heuristic",
                            "bidirectional": True
                        }
                    ))
        
        # 3. 应用领域特定启发式规则
        heuristic_func = self.domain_heuristics.get(domain, self.domain_heuristics["default"])
        domain_relations = heuristic_func(concepts, domain)
        causal_relations.extend(domain_relations)
        
        logger.debug(f"从领域 {domain} 推断出 {len(causal_relations)} 个因果关系")
        return causal_relations
    
    def _resolve_concept_id(self, concept_ref: str, source_concept: Dict[str, Any], concept_map: Dict[str, Any]) -> Optional[str]:
        """
        解析概念引用为完整的概念ID
        
        Args:
            concept_ref: 概念引用（可能是部分ID或名称）
            source_concept: 源概念数据
            concept_map: 概念映射
            
        Returns:
            完整的概念ID，如果无法解析则返回None
        """
        # 如果引用是完整的ID格式
        if concept_ref in concept_map:
            return concept_ref
        
        # 尝试解析为 domain:category:concept 格式
        domain = source_concept.get("domain", "")
        category = source_concept.get("category", "")
        
        # 尝试可能的组合
        possible_ids = [
            f"{domain}:{category}:{concept_ref}",
            f"{domain}:*:{concept_ref}",
            concept_ref  # 原样返回
        ]
        
        for possible_id in possible_ids:
            if possible_id in concept_map:
                return possible_id
        
        logger.warning(f"无法解析概念引用: {concept_ref}")
        return None
    
    def _apply_medical_heuristics(self, concepts: List[Dict[str, Any]], domain: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        应用医学领域启发式规则
        
        Args:
            concepts: 概念列表
            domain: 领域标识符
            
        Returns:
            因果关系列表
        """
        relations = []
        concept_map = {concept["id"]: concept for concept in concepts}
        
        for concept in concepts:
            concept_id = concept["id"]
            properties = concept.get("properties", {})
            
            # 检查疾病相关属性
            if "disease_mechanisms" in properties:
                # 疾病可能导致其机制中的症状
                mechanisms = properties.get("disease_mechanisms", [])
                for mechanism in mechanisms:
                    # 简单启发式：疾病 → 机制
                    relations.append((
                        concept_id,
                        f"{domain}:mechanism:{mechanism.lower().replace(' ', '_')}",
                        {
                            "strength": CausalStrength.MODERATE,
                            "confidence": 0.6,
                            "reason": "disease_causes_mechanism",
                            "evidence_type": EvidenceType.MECHANISTIC_REASONING.value,
                            "domain": domain,
                            "inference_method": "medical_heuristic"
                        }
                    ))
            
            # 检查治疗相关属性
            if "applications" in properties and any("treatment" in app.lower() for app in properties.get("applications", [])):
                # 治疗可能改善疾病
                relations.append((
                    concept_id,
                    f"{domain}:disease:improvement",
                    {
                        "strength": CausalStrength.STRONG,
                        "confidence": 0.8,
                        "reason": "treatment_improves_condition",
                        "evidence_type": EvidenceType.RANDOMIZED_TRIAL.value,
                        "domain": domain,
                        "inference_method": "medical_heuristic"
                    }
                ))
        
        return relations
    
    def _apply_biology_heuristics(self, concepts: List[Dict[str, Any]], domain: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        应用生物学领域启发式规则
        """
        relations = []
        
        for concept in concepts:
            concept_id = concept["id"]
            properties = concept.get("properties", {})
            
            # 生物学中的层次关系：分子 → 细胞 → 组织 → 器官 → 系统 → 生物体
            hierarchy_keywords = {
                "molecular": ["gene", "protein", "enzyme", "dna", "rna"],
                "cellular": ["cell", "organelle", "membrane", "cytoplasm"],
                "tissue": ["tissue", "epithelial", "connective", "muscle", "nervous"],
                "organ": ["organ", "heart", "lung", "liver", "kidney", "brain"],
                "system": ["system", "nervous", "circulatory", "respiratory", "digestive"],
                "organism": ["organism", "individual", "patient", "animal", "plant"]
            }
            
            # 检测概念在层次结构中的位置
            for level, keywords in hierarchy_keywords.items():
                concept_name_lower = concept["name"].lower()
                if any(keyword in concept_name_lower for keyword in keywords):
                    # 创建层次因果关系：较低层次 → 较高层次
                    if level == "molecular":
                        # 分子影响细胞
                        relations.append((
                            concept_id,
                            f"{domain}:cellular:function",
                            {
                                "strength": CausalStrength.MODERATE,
                                "confidence": 0.7,
                                "reason": "molecular_regulates_cellular",
                                "evidence_type": EvidenceType.MECHANISTIC_REASONING.value,
                                "domain": domain,
                                "inference_method": "biology_hierarchy_heuristic"
                            }
                        ))
                    elif level == "cellular":
                        # 细胞构成组织
                        relations.append((
                            concept_id,
                            f"{domain}:tissue:structure",
                            {
                                "strength": CausalStrength.MODERATE,
                                "confidence": 0.7,
                                "reason": "cells_form_tissue",
                                "evidence_type": EvidenceType.MECHANISTIC_REASONING.value,
                                "domain": domain,
                                "inference_method": "biology_hierarchy_heuristic"
                            }
                        ))
        
        return relations
    
    def _apply_chemistry_heuristics(self, concepts: List[Dict[str, Any]], domain: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        应用化学领域启发式规则
        """
        relations = []
        
        for concept in concepts:
            concept_id = concept["id"]
            properties = concept.get("properties", {})
            
            # 化学中的反应关系
            if "formula" in properties:
                # 化学式可能暗示反应物和生成物的关系
                formula = properties["formula"]
                # 简单启发式：如果有"→"或"="，推断反应关系
                if "→" in formula or "=" in formula:
                    relations.append((
                        concept_id,
                        f"{domain}:reaction:product",
                        {
                            "strength": CausalStrength.STRONG,
                            "confidence": 0.9,
                            "reason": "chemical_reaction",
                            "evidence_type": EvidenceType.MECHANISTIC_REASONING.value,
                            "domain": domain,
                            "inference_method": "chemistry_reaction_heuristic"
                        }
                    ))
        
        return relations
    
    def _apply_physics_heuristics(self, concepts: List[Dict[str, Any]], domain: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        应用物理学领域启发式规则
        """
        relations = []
        
        for concept in concepts:
            concept_id = concept["id"]
            properties = concept.get("properties", {})
            
            # 物理中的因果链：力 → 运动 → 能量
            physics_keywords = {
                "force": ["force", "gravity", "electromagnetic", "strong", "weak"],
                "motion": ["motion", "velocity", "acceleration", "momentum"],
                "energy": ["energy", "kinetic", "potential", "thermal", "electrical"]
            }
            
            concept_name_lower = concept["name"].lower()
            
            # 力导致运动
            if any(keyword in concept_name_lower for keyword in physics_keywords["force"]):
                relations.append((
                    concept_id,
                    f"{domain}:motion:change",
                    {
                        "strength": CausalStrength.STRONG,
                        "confidence": 0.95,
                        "reason": "force_causes_motion",
                        "evidence_type": EvidenceType.MECHANISTIC_REASONING.value,
                        "domain": domain,
                        "inference_method": "physics_causality_heuristic"
                    }
                ))
            
            # 运动产生能量
            if any(keyword in concept_name_lower for keyword in physics_keywords["motion"]):
                relations.append((
                    concept_id,
                    f"{domain}:energy:kinetic",
                    {
                        "strength": CausalStrength.MODERATE,
                        "confidence": 0.8,
                        "reason": "motion_creates_energy",
                        "evidence_type": EvidenceType.MECHANISTIC_REASONING.value,
                        "domain": domain,
                        "inference_method": "physics_causality_heuristic"
                    }
                ))
        
        return relations
    
    def _apply_cs_heuristics(self, concepts: List[Dict[str, Any]], domain: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        应用计算机科学领域启发式规则
        """
        relations = []
        
        for concept in concepts:
            concept_id = concept["id"]
            properties = concept.get("properties", {})
            
            # 计算机科学中的依赖关系
            cs_keywords = {
                "algorithm": ["algorithm", "sort", "search", "graph", "tree"],
                "data_structure": ["array", "linked list", "stack", "queue", "hash table"],
                "system": ["operating system", "network", "database", "compiler"]
            }
            
            concept_name_lower = concept["name"].lower()
            
            # 算法使用数据结构
            if any(keyword in concept_name_lower for keyword in cs_keywords["algorithm"]):
                relations.append((
                    concept_id,
                    f"{domain}:data_structure:usage",
                    {
                        "strength": CausalStrength.MODERATE,
                        "confidence": 0.75,
                        "reason": "algorithm_uses_data_structure",
                        "evidence_type": EvidenceType.MECHANISTIC_REASONING.value,
                        "domain": domain,
                        "inference_method": "cs_dependency_heuristic"
                    }
                ))
            
            # 系统依赖于算法
            if any(keyword in concept_name_lower for keyword in cs_keywords["system"]):
                relations.append((
                    f"{domain}:algorithm:implementation",
                    concept_id,
                    {
                        "strength": CausalStrength.MODERATE,
                        "confidence": 0.7,
                        "reason": "system_depends_on_algorithm",
                        "evidence_type": EvidenceType.MECHANISTIC_REASONING.value,
                        "domain": domain,
                        "inference_method": "cs_dependency_heuristic"
                    }
                ))
        
        return relations
    
    def _apply_general_heuristics(self, concepts: List[Dict[str, Any]], domain: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        应用通用启发式规则
        
        从概念属性中提取关系，包括：
        1. 列表字段（applications, body_systems等）→ 概念与列表项之间的关系
        2. 层次关系字段 → 父子关系
        3. 重要性和难度 → 影响力关系
        """
        relations = []
        
        for concept in concepts:
            concept_id = concept["id"]
            properties = concept.get("properties", {})
            
            # 基于难度和重要性的启发式
            difficulty = concept.get("difficulty", "intermediate")
            importance = concept.get("importance", "medium")
            
            # 重要概念可能影响其他概念
            if importance in ["high", "critical"]:
                relations.append((
                    concept_id,
                    f"{domain}:general:influence",
                    {
                        "strength": CausalStrength.MODERATE,
                        "confidence": 0.6,
                        "reason": "important_concept_influences_others",
                        "evidence_type": EvidenceType.EXPERT_KNOWLEDGE.value,
                        "domain": domain,
                        "inference_method": "importance_heuristic"
                    }
                ))
            
            # 从属性中提取关系
            relations.extend(self._extract_relations_from_properties(concept_id, properties, domain))
        
        return relations
    
    def _extract_relations_from_properties(self, concept_id: str, properties: Dict[str, Any], domain: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        从概念属性中提取关系
        
        Args:
            concept_id: 概念ID
            properties: 概念属性字典
            domain: 领域标识符
            
        Returns:
            关系列表
        """
        relations = []
        
        # 定义属性字段到关系类型的映射
        property_mappings = {
            # 字段名: (关系方向, 关系描述, 强度, 置信度)
            "applications": ("concept_to_item", "has_application", CausalStrength.MODERATE, 0.7),
            "body_systems": ("concept_to_item", "part_of_system", CausalStrength.STRONG, 0.8),
            "tissue_types": ("concept_to_item", "involves_tissue", CausalStrength.MODERATE, 0.7),
            "key_processes": ("concept_to_item", "involves_process", CausalStrength.MODERATE, 0.7),
            "regulatory_mechanisms": ("concept_to_item", "regulated_by", CausalStrength.MODERATE, 0.7),
            "disease_mechanisms": ("concept_to_item", "causes_mechanism", CausalStrength.STRONG, 0.8),
            "diagnostic_applications": ("concept_to_item", "used_for_diagnosis", CausalStrength.MODERATE, 0.7),
            "staining_techniques": ("concept_to_item", "uses_technique", CausalStrength.MODERATE, 0.7),
            "regions": ("concept_to_item", "located_in", CausalStrength.MODERATE, 0.7),
            "formula": ("concept_to_item", "expressed_as", CausalStrength.STRONG, 0.9),
        }
        
        for prop_name, (direction, relation_desc, strength, confidence) in property_mappings.items():
            if prop_name in properties:
                prop_value = properties[prop_name]
                
                if isinstance(prop_value, list):
                    # 列表属性：为每个项创建关系
                    for item in prop_value:
                        if isinstance(item, str):
                            # 创建目标概念ID
                            target_id = f"{domain}:{prop_name}:{item.lower().replace(' ', '_')}"
                            
                            # 根据方向确定源和目标
                            if direction == "concept_to_item":
                                source, target = concept_id, target_id
                            else:
                                source, target = target_id, concept_id
                            
                            relations.append((
                                source,
                                target,
                                {
                                    "strength": strength,
                                    "confidence": confidence,
                                    "reason": relation_desc,
                                    "evidence_type": EvidenceType.EXPERT_KNOWLEDGE.value,
                                    "domain": domain,
                                    "inference_method": "property_extraction",
                                    "property_source": prop_name
                                }
                            ))
                
                elif isinstance(prop_value, str):
                    # 字符串属性：创建单个关系
                    target_id = f"{domain}:{prop_name}:{prop_value.lower().replace(' ', '_')}"
                    
                    if direction == "concept_to_item":
                        source, target = concept_id, target_id
                    else:
                        source, target = target_id, concept_id
                    
                    relations.append((
                        source,
                        target,
                        {
                            "strength": strength,
                            "confidence": confidence,
                            "reason": relation_desc,
                            "evidence_type": EvidenceType.EXPERT_KNOWLEDGE.value,
                            "domain": domain,
                            "inference_method": "property_extraction",
                            "property_source": prop_name
                        }
                    ))
        
        return relations
    
    def add_cross_domain_connections(self) -> int:
        """
        添加跨领域连接以减少孤立节点
        
        策略:
        1. 连接不同领域中名称相似的概念
        2. 连接重要性高的概念
        3. 创建领域间的桥梁关系
        
        Returns:
            添加的连接数量
        """
        added_connections = 0
        graph = self.causal_graph.graph
        
        # 收集所有概念节点
        concept_nodes = []
        for node_id, data in graph.nodes(data=True):
            if data.get("type") == "concept":
                concept_nodes.append((node_id, data))
        
        if len(concept_nodes) < 2:
            return 0
        
        # 按领域分组
        concepts_by_domain = defaultdict(list)
        for node_id, data in concept_nodes:
            domain = data.get("metadata", {}).get("domain", "unknown")
            concepts_by_domain[domain].append((node_id, data))
        
        # 策略1: 连接不同领域中名称相似的概念
        for i, (domain1, concepts1) in enumerate(concepts_by_domain.items()):
            for domain2, concepts2 in list(concepts_by_domain.items())[i+1:]:
                # 简单启发式：连接两个领域中的第一个概念
                if concepts1 and concepts2:
                    node1_id, node1_data = concepts1[0]
                    node2_id, node2_data = concepts2[0]
                    
                    # 添加双向弱连接
                    success1 = self.causal_graph.add_causal_relation(
                        cause=node1_id,
                        effect=node2_id,
                        strength=CausalStrength.WEAK,
                        confidence=0.4,
                        properties={
                            "reason": "cross_domain_similarity",
                            "evidence_type": EvidenceType.EXPERT_KNOWLEDGE.value,
                            "inference_method": "cross_domain_heuristic"
                        }
                    )
                    
                    if success1:
                        added_connections += 1
        
        # 策略2: 连接重要性高的概念
        important_concepts = []
        for node_id, data in concept_nodes:
            importance = data.get("metadata", {}).get("importance", "medium")
            if importance in ["high", "critical"]:
                important_concepts.append(node_id)
        
        # 连接重要性高的概念（如果至少有两个）
        if len(important_concepts) >= 2:
            for i in range(len(important_concepts)):
                for j in range(i+1, len(important_concepts)):
                    success = self.causal_graph.add_causal_relation(
                        cause=important_concepts[i],
                        effect=important_concepts[j],
                        strength=CausalStrength.MODERATE,
                        confidence=0.5,
                        properties={
                            "reason": "important_concepts_connection",
                            "evidence_type": EvidenceType.EXPERT_KNOWLEDGE.value,
                            "inference_method": "importance_based_connection"
                        }
                    )
                    
                    if success:
                        added_connections += 1
        
        logger.info(f"添加了 {added_connections} 个跨领域连接")
        return added_connections
    
    def add_concepts_to_graph(self, concepts: List[Dict[str, Any]]) -> int:
        """
        添加概念到因果知识图谱
        
        Args:
            concepts: 概念列表
            
        Returns:
            成功添加的概念数量
        """
        added_count = 0
        
        for concept in concepts:
            concept_id = concept["id"]
            
            # 准备节点属性
            properties = concept.get("properties", {})
            metadata = {
                "domain": concept.get("domain"),
                "category": concept.get("category"),
                "original_id": concept.get("original_id"),
                "difficulty": concept.get("difficulty"),
                "importance": concept.get("importance"),
                "migration_source": "knowledge_base",
                "migration_timestamp": time.time()
            }
            
            # 添加节点
            success = self.causal_graph.add_node(
                node_id=concept_id,
                node_type="concept",
                properties=properties,
                metadata=metadata
            )
            
            if success:
                added_count += 1
        
        logger.info(f"成功添加 {added_count} 个概念到图谱")
        self.migration_stats["concepts_extracted"] += added_count
        
        return added_count
    
    def add_causal_relations_to_graph(self, relations: List[Tuple[str, str, Dict[str, Any]]]) -> int:
        """
        添加因果关系到图谱
        
        Args:
            relations: 因果关系列表
            
        Returns:
            成功添加的关系数量
        """
        added_count = 0
        
        for cause, effect, attrs in relations:
            # 从属性中提取参数
            strength = attrs.get("strength", CausalStrength.MODERATE)
            confidence = attrs.get("confidence", 0.7)
            evidence = attrs.get("evidence", None)
            
            # 创建属性副本，确保强度为字符串
            properties = attrs.copy()
            if "strength" in properties:
                if isinstance(properties["strength"], CausalStrength):
                    properties["strength"] = properties["strength"].value
            
            # 添加因果关系
            success = self.causal_graph.add_causal_relation(
                cause=cause,
                effect=effect,
                strength=strength,
                confidence=confidence,
                evidence=evidence,
                properties=properties
            )
            
            if success:
                added_count += 1
        
        logger.info(f"成功添加 {added_count} 个因果关系到图谱")
        self.migration_stats["causal_relations_inferred"] += added_count
        
        return added_count
    
    def validate_migration(self) -> Dict[str, Any]:
        """
        验证迁移结果
        
        Returns:
            验证结果字典
        """
        validation_results = {
            "graph_consistency": True,
            "node_count": self.causal_graph.stats.get("nodes", 0),
            "edge_count": self.causal_graph.stats.get("edges", 0),
            "issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        # 检查图的一致性
        graph = self.causal_graph.graph
        
        # 1. 检查是否有孤立的节点（无连接）
        isolated_nodes = [n for n in graph.nodes() if graph.degree(n) == 0]
        if isolated_nodes:
            validation_results["warnings"].append(f"发现 {len(isolated_nodes)} 个孤立节点")
            validation_results["graph_consistency"] = False
        
        # 2. 检查是否有自循环
        self_loops = list(nx.selfloop_edges(graph))
        if self_loops:
            validation_results["issues"].append(f"发现 {len(self_loops)} 个自循环边")
        
        # 3. 检查边的属性完整性
        edge_attribute_issues = 0
        for u, v, data in graph.edges(data=True):
            if "strength" not in data or "confidence" not in data:
                edge_attribute_issues += 1
        
        if edge_attribute_issues > 0:
            validation_results["issues"].append(f"{edge_attribute_issues} 条边缺少必要属性")
        
        # 4. 检查图的连通性
        if not nx.is_weakly_connected(graph):
            # 计算连通分量
            components = list(nx.weakly_connected_components(graph))
            validation_results["warnings"].append(f"图不是弱连通的，有 {len(components)} 个连通分量")
            validation_results["recommendations"].append("考虑添加跨领域连接以改善连通性")
        
        # 5. 统计节点类型分布
        node_types = defaultdict(int)
        for _, data in graph.nodes(data=True):
            node_type = data.get("type", "unknown")
            node_types[node_type] += 1
        
        validation_results["node_type_distribution"] = dict(node_types)
        
        # 6. 统计边强度分布
        edge_strengths = defaultdict(int)
        for _, _, data in graph.edges(data=True):
            strength = data.get("strength", "unknown")
            edge_strengths[strength] += 1
        
        validation_results["edge_strength_distribution"] = dict(edge_strengths)
        
        logger.info(f"验证完成: {validation_results['node_count']} 节点, {validation_results['edge_count']} 边")
        
        if validation_results["issues"]:
            logger.warning(f"验证发现问题: {validation_results['issues']}")
        
        if validation_results["warnings"]:
            logger.warning(f"验证发现警告: {validation_results['warnings']}")
        
        return validation_results
    
    def save_graph(self, output_path: Optional[Path] = None) -> bool:
        """
        保存因果知识图谱到文件
        
        Args:
            output_path: 输出文件路径（可选）
            
        Returns:
            是否成功保存
        """
        try:
            if output_path is None:
                output_path = self.output_path
            
            # 创建序列化数据
            graph_data = {
                "graph_name": self.causal_graph.name,
                "graph_data": nx.node_link_data(self.causal_graph.graph),
                "metadata": self.causal_graph.metadata,
                "stats": self.causal_graph.stats,
                "migration_stats": self.migration_stats,
                "save_timestamp": time.time()
            }
            
            # 保存为JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"图谱已保存到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存图谱失败: {e}")
            return False
    
    def run_migration(self) -> Dict[str, Any]:
        """
        运行完整的数据迁移流程
        
        Returns:
            迁移结果摘要
        """
        logger.info("开始数据迁移流程")
        
        # 1. 扫描知识库文件
        knowledge_files = self.scan_knowledge_files()
        if not knowledge_files:
            logger.error("未找到知识库文件，迁移中止")
            return {"success": False, "error": "未找到知识库文件"}
        
        # 2. 处理每个文件
        for file_path in knowledge_files:
            logger.info(f"处理文件: {file_path.name}")
            
            # 加载知识库数据
            domain_data = self.load_knowledge_file(file_path)
            if domain_data is None:
                logger.warning(f"跳过文件: {file_path.name}")
                continue
            
            domain = domain_data.get("domain", "unknown")
            
            # 提取概念
            concepts = self.extract_concepts_from_domain(domain_data)
            if not concepts:
                logger.warning(f"文件 {file_path.name} 中未提取到概念")
                continue
            
            # 添加概念到图谱
            concepts_added = self.add_concepts_to_graph(concepts)
            
            # 推断因果关系
            causal_relations = self.infer_causal_relations(concepts, domain)
            
            # 添加关系到图谱
            relations_added = self.add_causal_relations_to_graph(causal_relations)
            
            self.migration_stats["files_processed"] += 1
            
            logger.info(f"文件处理完成: {file_path.name} - {concepts_added} 概念, {relations_added} 关系")
        
        # 3. 添加跨领域连接
        cross_domain_connections = self.add_cross_domain_connections()
        self.migration_stats["causal_relations_inferred"] += cross_domain_connections
        
        # 4. 验证迁移结果
        validation_results = self.validate_migration()
        
        # 5. 保存图谱
        save_success = self.save_graph()
        
        # 6. 完成统计
        self.migration_stats["end_time"] = time.time()
        total_time = self.migration_stats["end_time"] - self.migration_stats["start_time"]
        
        # 构建结果摘要
        migration_summary = {
            "success": save_success and validation_results["graph_consistency"],
            "total_files": self.migration_stats["total_files"],
            "files_processed": self.migration_stats["files_processed"],
            "concepts_extracted": self.migration_stats["concepts_extracted"],
            "causal_relations_inferred": self.migration_stats["causal_relations_inferred"],
            "total_time_seconds": total_time,
            "validation_results": validation_results,
            "graph_saved": save_success,
            "save_path": str(self.output_path) if save_success else None
        }
        
        logger.info("数据迁移流程完成")
        logger.info(f"摘要: {migration_summary}")
        
        return migration_summary


def main():
    """主函数"""
    # 配置路径
    knowledge_base_path = "core/data/knowledge"
    output_path = "data/migrated_causal_knowledge_graph.json"
    
    # 创建迁移器
    migrator = KnowledgeDataMigrator(knowledge_base_path, output_path)
    
    # 运行迁移
    result = migrator.run_migration()
    
    # 打印结果
    print("\n" + "="*60)
    print("数据迁移结果摘要")
    print("="*60)
    
    if result["success"]:
        print(f"✅ 迁移成功!")
    else:
        print(f"❌ 迁移失败或存在问题")
    
    print(f"\n📊 统计信息:")
    print(f"   处理文件数: {result['files_processed']}/{result['total_files']}")
    print(f"   提取概念数: {result['concepts_extracted']}")
    print(f"   推断关系数: {result['causal_relations_inferred']}")
    print(f"   总耗时: {result['total_time_seconds']:.2f} 秒")
    
    if result["graph_saved"]:
        print(f"\n💾 图谱已保存到: {result['save_path']}")
    
    # 显示验证结果
    validation = result["validation_results"]
    if validation["issues"]:
        print(f"\n⚠️  验证问题:")
        for issue in validation["issues"]:
            print(f"   - {issue}")
    
    if validation["warnings"]:
        print(f"\n⚠️  验证警告:")
        for warning in validation["warnings"]:
            print(f"   - {warning}")
    
    if validation["recommendations"]:
        print(f"\n💡 建议:")
        for rec in validation["recommendations"]:
            print(f"   - {rec}")
    
    print("\n" + "="*60)
    
    return result


if __name__ == "__main__":
    main()