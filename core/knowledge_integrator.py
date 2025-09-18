"""
常识知识库整合模块
集成外部常识知识库（如ConceptNet、WordNet）来增强系统的常识推理能力
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CommonSenseRelation:
    """常识关系数据类"""
    subject: str
    relation: str
    object: str
    weight: float
    source: str

class KnowledgeIntegrator:
    """
    知识整合器 - 集成外部常识知识库
    提供常识推理、关系查询、概念扩展等功能
    """
    
    def __init__(self):
        self.conceptnet_base_url = "http://api.conceptnet.io"
        self.wordnet_base_url = "http://wordnet-rdf.princeton.edu"
        self.cached_relations: Dict[str, List[CommonSenseRelation]] = {}
        self.common_sense_knowledge: Dict[str, Any] = self._initialize_common_sense_knowledge()
        
        # 知识库统计
        self.stats = {
            "conceptnet_queries": 0,
            "wordnet_queries": 0,
            "cache_hits": 0,
            "total_relations": 0
        }
    
    def _initialize_common_sense_knowledge(self) -> Dict[str, Any]:
        """初始化内置常识知识库"""
        return {
            "basic_facts": {
                "water": {"is_liquid": True, "freezes_at": 0, "boils_at": 100},
                "fire": {"is_hot": True, "produces_light": True, "can_burn": True},
                "human": {"needs_oxygen": True, "needs_water": True, "needs_food": True},
                "animal": {"can_move": True, "needs_food": True, "reproduces": True}
            },
            "causal_relationships": {
                "rain": {"causes": ["wet_ground", "flooding", "plant_growth"]},
                "fire": {"causes": ["heat", "smoke", "destruction"]},
                "eating": {"causes": ["satiety", "energy", "digestion"]}
            },
            "temporal_knowledge": {
                "day_night_cycle": {"day_follows_night": True, "24_hours": True},
                "seasons": {"spring": "growth", "summer": "warm", "autumn": "harvest", "winter": "cold"}
            }
        }
    
    def query_conceptnet(self, concept: str, limit: int = 10) -> List[CommonSenseRelation]:
        """查询ConceptNet知识库"""
        try:
            url = f"{self.conceptnet_base_url}/c/zh/{concept}?limit={limit}"
            response = requests.get(url, timeout=5)
            self.stats["conceptnet_queries"] += 1
            
            if response.status_code == 200:
                data = response.json()
                relations = []
                
                for edge in data.get('edges', []):
                    rel = CommonSenseRelation(
                        subject=edge.get('start', {}).get('label', ''),
                        relation=edge.get('rel', {}).get('label', ''),
                        object=edge.get('end', {}).get('label', ''),
                        weight=edge.get('weight', 0.5),
                        source="conceptnet"
                    )
                    relations.append(rel)
                    self.stats["total_relations"] += 1
                
                # 缓存结果
                self.cached_relations[concept] = relations
                return relations
                
        except Exception as e:
            logger.warning(f"ConceptNet查询失败: {e}")
        
        return []
    
    def get_common_sense_relations(self, concept: str, use_cache: bool = True) -> List[CommonSenseRelation]:
        """获取概念的常识关系"""
        # 检查缓存
        if use_cache and concept in self.cached_relations:
            self.stats["cache_hits"] += 1
            return self.cached_relations[concept]
        
        # 查询外部知识库
        external_relations = self.query_conceptnet(concept)
        
        # 合并内置知识
        all_relations = external_relations.copy()
        
        # 添加内置常识关系
        if concept in self.common_sense_knowledge["basic_facts"]:
            for attr, value in self.common_sense_knowledge["basic_facts"][concept].items():
                rel = CommonSenseRelation(
                    subject=concept,
                    relation=f"has_property",
                    object=str(value),
                    weight=0.9,
                    source="builtin"
                )
                all_relations.append(rel)
        
        return all_relations
    
    def infer_common_sense(self, subject: str, relation: str, object: str = None) -> float:
        """
        常识推理 - 判断关系是否合理
        返回置信度分数 (0.0-1.0)
        """
        relations = self.get_common_sense_relations(subject)
        
        if object:
            # 检查特定关系
            for rel in relations:
                if rel.relation == relation and rel.object.lower() == object.lower():
                    return rel.weight
            return 0.0
        else:
            # 检查关系是否存在
            for rel in relations:
                if rel.relation == relation:
                    return max(0.6, rel.weight)  # 至少0.6的置信度
            return 0.0
    
    def expand_concept(self, concept: str, depth: int = 2) -> Dict[str, Any]:
        """概念扩展 - 获取相关概念网络"""
        if depth <= 0:
            return {"concept": concept, "relations": []}
        
        relations = self.get_common_sense_relations(concept)
        expanded = {
            "concept": concept,
            "relations": [],
            "related_concepts": {}
        }
        
        for rel in relations:
            expanded["relations"].append({
                "relation": rel.relation,
                "target": rel.object,
                "weight": rel.weight,
                "source": rel.source
            })
            
            # 递归扩展相关概念
            if depth > 1 and rel.weight > 0.7:
                expanded["related_concepts"][rel.object] = self.expand_concept(rel.object, depth - 1)
        
        return expanded
    
    def validate_fact(self, fact: str) -> Dict[str, Any]:
        """验证事实的合理性"""
        # 简单的事实解析（实际应用中需要更复杂的NLP）
        parts = fact.lower().split()
        if len(parts) < 3:
            return {"valid": False, "confidence": 0.0, "reason": "Invalid fact format"}
        
        # 尝试提取主语、关系、宾语
        subject = parts[0]
        relation = " ".join(parts[1:-1])
        object = parts[-1]
        
        confidence = self.infer_common_sense(subject, relation, object)
        
        return {
            "valid": confidence > 0.6,
            "confidence": confidence,
            "subject": subject,
            "relation": relation,
            "object": object
        }
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return {
            **self.stats,
            "cached_concepts": len(self.cached_relations),
            "builtin_facts": len(self.common_sense_knowledge["basic_facts"])
        }

# 单例实例
knowledge_integrator = KnowledgeIntegrator()

if __name__ == "__main__":
    # 测试代码
    ki = KnowledgeIntegrator()
    
    # 测试常识查询
    print("=== 测试常识知识库 ===")
    relations = ki.get_common_sense_relations("水")
    for rel in relations[:5]:  # 显示前5个关系
        print(f"{rel.subject} --{rel.relation}--> {rel.object} ({rel.weight})")
    
    # 测试常识推理
    print("\n=== 测试常识推理 ===")
    test_cases = [
        ("水", "is_liquid", "True"),
        ("火", "is_hot", "True"), 
        ("鸟", "can_fly", "True"),
        ("鱼", "can_fly", "False")
    ]
    
    for subject, relation, obj in test_cases:
        confidence = ki.infer_common_sense(subject, relation, obj)
        print(f"{subject} {relation} {obj}: 置信度 {confidence:.2f}")
    
    # 显示统计信息
    print(f"\n=== 统计信息 ===")
    stats = ki.get_knowledge_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
