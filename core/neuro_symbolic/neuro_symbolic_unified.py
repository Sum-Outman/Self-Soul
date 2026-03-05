#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
神经符号统一框架 - 实现双向神经符号推理系统

核心功能:
1. 神经表示到符号命题的自动提取和学习
2. 符号命题到神经执行的约束指导和验证
3. 一阶逻辑推理和定理证明
4. 可微符号操作的端到端学习
5. 神经符号一致性检查和维护

架构设计:
- 神经编码器: 将原始输入转换为神经表示
- 符号提取器: 从神经表示中提取符号概念和关系
- 逻辑推理器: 执行符号逻辑推理
- 符号约束器: 将符号约束转换为神经网络的指导信号
- 一致性检查器: 确保神经和符号表示的一致性

技术特点:
- 可微符号推理: 符号操作通过松弛化实现可微分
- 双向信息流: 神经⇄符号的闭环信息流
- 分层抽象: 从具体感知到抽象概念的多层符号化
- 自适应学习: 从失败中学习改进符号提取规则

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx

# 导入错误处理
from core.error_handling import ErrorHandler

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()



def _deterministic_randn(size, seed_prefix="default"):
    """Generate deterministic normal distribution using numpy RandomState"""
    import math
    if isinstance(size, int):
        size = (size,)
    total_elements = 1
    for dim in size:
        total_elements *= dim
    
    # Create deterministic seed from seed_prefix using adler32
    seed_hash = zlib.adler32(seed_prefix.encode('utf-8')) & 0xffffffff
    rng = np.random.RandomState(seed_hash)
    
    # Generate uniform random numbers
    u1 = rng.random_sample(total_elements)
    u2 = rng.random_sample(total_elements)
    
    # Apply Box-Muller transform
    u1 = np.maximum(u1, 1e-10)
    u2 = np.maximum(u2, 1e-10)
    z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)
    
    # Convert to torch tensor
    import torch
    result = torch.from_numpy(z0).float()
    
    return result.view(*size)

class SymbolType(Enum):
    """符号类型枚举"""
    ENTITY = "entity"          # 实体: 对象、人、地点等
    PROPERTY = "property"      # 属性: 颜色、大小、形状等
    RELATION = "relation"      # 关系: 在...之上、是...的一部分等
    ACTION = "action"          # 动作: 移动、旋转、创建等
    EVENT = "event"            # 事件: 发生、开始、结束等
    CONCEPT = "concept"        # 概念: 抽象概念和类别


class LogicOperator(Enum):
    """逻辑运算符枚举"""
    AND = "and"        # 合取 (∧)
    OR = "or"          # 析取 (∨)
    NOT = "not"        # 否定 (¬)
    IMPLIES = "implies"  # 蕴含 (→)
    EQUIVALENT = "equivalent"  # 等价 (↔)
    FORALL = "forall"  # 全称量词 (∀)
    EXISTS = "exists"  # 存在量词 (∃)


class NeuralSymbolicUnifiedFramework:
    """
    神经符号统一框架 - 实现双向神经符号推理
    
    核心组件:
    1. NeuralEncoder: 神经编码器，将输入转换为神经表示
    2. SymbolExtractor: 符号提取器，从神经表示中提取符号
    3. LogicReasoner: 逻辑推理器，执行符号逻辑推理
    4. SymbolicConstraint: 符号约束器，将符号约束应用于神经网络
    5. ConsistencyChecker: 一致性检查器，确保神经符号一致性
    
    工作流程:
    输入 → NeuralEncoder → 神经表示 → SymbolExtractor → 符号命题
    符号命题 → LogicReasoner → 推理结果 → SymbolicConstraint → 神经约束
    神经约束 → 指导神经网络训练和推理 → 一致性检查 → 反馈改进
    """
    
    def __init__(self, 
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 symbol_dim: int = 128,
                 learning_rate: float = 0.001):
        """
        初始化神经符号统一框架
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            symbol_dim: 符号表示维度
            learning_rate: 学习率
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.symbol_dim = symbol_dim
        self.learning_rate = learning_rate
        
        # 训练状态
        self.training = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化神经网络组件
        self._init_neural_components()
        
        # 初始化符号组件
        self._init_symbolic_components()
        
        # 初始化优化器
        self._init_optimizers()
        
        # 符号知识库
        self.symbol_knowledge_base = {
            "entities": {},      # 实体符号
            "properties": {},    # 属性符号
            "relations": {},     # 关系符号
            "rules": [],         # 逻辑规则
            "constraints": []    # 约束条件
        }
        
        # 性能统计
        self.performance_stats = {
            "neural_to_symbol_translations": 0,
            "symbol_to_neural_constraints": 0,
            "logic_inferences": 0,
            "consistency_checks": 0,
            "learning_updates": 0
        }
        
        logger.info(f"神经符号统一框架初始化完成，设备: {self.device}")
    
    def _init_neural_components(self):
        """初始化神经网络组件"""
        # 神经编码器：输入 → 神经表示
        self.neural_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
        
        # 符号提取器：神经表示 → 符号特征
        self.symbol_extractor = nn.ModuleDict({
            "entity_detector": nn.Sequential(
                nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 4, self.symbol_dim)
            ),
            "relation_detector": nn.Sequential(
                nn.Linear(self.hidden_dim // 2 * 2, self.hidden_dim // 2),  # 两个实体的连接
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, self.symbol_dim)
            ),
            "property_detector": nn.Sequential(
                nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 4, self.symbol_dim)
            )
        })
        
        # 符号解码器：符号表示 → 神经表示
        self.symbol_decoder = nn.Sequential(
            nn.Linear(self.symbol_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim)
        )
        
        # 移动到设备
        self.neural_encoder.to(self.device)
        self.symbol_extractor.to(self.device)
        self.symbol_decoder.to(self.device)
    
    def _init_symbolic_components(self):
        """初始化符号组件"""
        # 符号到概念的映射
        self.symbol_to_concept = {}
        self.concept_to_symbol = {}
        
        # 逻辑推理规则库
        self.logic_rules = []
        
        # 符号相似度阈值
        self.symbol_similarity_threshold = 0.7
        
        # 初始化基础符号集
        self._initialize_basic_symbols()
    
    def _initialize_basic_symbols(self):
        """初始化基础符号集"""
        # 基础实体符号
        basic_entities = ["object", "person", "location", "time", "event", "action"]
        for entity in basic_entities:
            self._add_symbol(entity, SymbolType.ENTITY)
        
        # 基础关系符号
        basic_relations = ["part_of", "contains", "near", "above", "below", "inside", "outside"]
        for relation in basic_relations:
            self._add_symbol(relation, SymbolType.RELATION)
        
        # 基础属性符号
        basic_properties = ["color", "size", "shape", "texture", "weight", "temperature"]
        for prop in basic_properties:
            self._add_symbol(prop, SymbolType.PROPERTY)
        
        # 基础逻辑规则
        self._add_logic_rule("forall x, y: part_of(x, y) → contains(y, x)")
        self._add_logic_rule("forall x, y: above(x, y) → not below(x, y)")
        
        logger.info(f"初始化基础符号集: {len(basic_entities)}实体, {len(basic_relations)}关系, {len(basic_properties)}属性")
    
    def _init_optimizers(self):
        """初始化优化器"""
        # 神经网络参数
        neural_params = list(self.neural_encoder.parameters()) + \
                       list(self.symbol_extractor.parameters()) + \
                       list(self.symbol_decoder.parameters())
        
        self.neural_optimizer = optim.Adam(neural_params, lr=self.learning_rate)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.neural_optimizer, step_size=1000, gamma=0.9)
    
    def _add_symbol(self, symbol_name: str, symbol_type: SymbolType) -> str:
        """
        添加新符号到知识库
        
        Args:
            symbol_name: 符号名称
            symbol_type: 符号类型
            
        Returns:
            符号ID
        """
        symbol_id = f"{symbol_type.value}_{symbol_name}"
        
        if symbol_id not in self.symbol_knowledge_base["entities"]:
            # 创建符号向量表示（确定性生成）
            symbol_vector = self._generate_symbol_vector(symbol_name, symbol_type)
            
            # 存储到知识库
            if symbol_type == SymbolType.ENTITY:
                self.symbol_knowledge_base["entities"][symbol_id] = {
                    "name": symbol_name,
                    "type": symbol_type.value,
                    "vector": symbol_vector,
                    "properties": [],
                    "relations": []
                }
            elif symbol_type == SymbolType.RELATION:
                self.symbol_knowledge_base["relations"][symbol_id] = {
                    "name": symbol_name,
                    "type": symbol_type.value,
                    "vector": symbol_vector,
                    "arity": 2,  # 默认为二元关系
                    "symmetry": False,
                    "transitivity": False
                }
            elif symbol_type == SymbolType.PROPERTY:
                self.symbol_knowledge_base["properties"][symbol_id] = {
                    "name": symbol_name,
                    "type": symbol_type.value,
                    "vector": symbol_vector,
                    "value_type": "continuous",  # 或 "discrete", "categorical"
                    "value_range": None
                }
            
            # 更新映射
            self.symbol_to_concept[symbol_id] = symbol_name
            self.concept_to_symbol[symbol_name] = symbol_id
            
            logger.debug(f"添加符号: {symbol_id} ({symbol_type.value})")
        
        return symbol_id
    
    def _generate_symbol_vector(self, symbol_name: str, symbol_type: SymbolType) -> np.ndarray:
        """
        生成符号的向量表示
        
        Args:
            symbol_name: 符号名称
            symbol_type: 符号类型
            
        Returns:
            符号向量
        """
        # 确定性向量生成（基于名称和类型的哈希）
        vector = np.zeros(self.symbol_dim, dtype=np.float32)
        
        # 使用哈希函数生成确定性值
        for i in range(self.symbol_dim):
            # 结合名称、类型和维度的哈希
            hash_str = f"{symbol_name}_{symbol_type.value}_{i}"
            hash_val = abs((zlib.adler32(str(hash_str).encode('utf-8')) & 0xffffffff)) % 10000
            vector[i] = (hash_val / 10000.0) * 2 - 1  # 归一化到[-1, 1]
        
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _add_logic_rule(self, rule_expression: str):
        """
        添加逻辑规则
        
        Args:
            rule_expression: 规则表达式字符串
        """
        # 简化：存储原始表达式
        rule_id = f"rule_{len(self.symbol_knowledge_base['rules']) + 1}"
        rule = {
            "id": rule_id,
            "expression": rule_expression,
            "created": time.time(),
            "usage_count": 0
        }
        
        self.symbol_knowledge_base["rules"].append(rule)
        self.logic_rules.append(rule_expression)
        
        logger.debug(f"添加逻辑规则: {rule_expression}")
    
    def neural_to_symbol(self, 
                        neural_input: Union[torch.Tensor, np.ndarray],
                        extract_types: Optional[List[SymbolType]] = None) -> Dict[str, Any]:
        """
        神经表示到符号的转换
        
        Args:
            neural_input: 神经输入表示
            extract_types: 要提取的符号类型列表（如果为None，则提取所有类型）
            
        Returns:
            提取的符号信息
        """
        start_time = time.time()
        
        # 转换输入为张量
        if isinstance(neural_input, np.ndarray):
            neural_tensor = torch.tensor(neural_input, dtype=torch.float32).to(self.device)
        else:
            neural_tensor = neural_input.to(self.device)
        
        # 确保正确的维度
        if len(neural_tensor.shape) == 1:
            neural_tensor = neural_tensor.unsqueeze(0)  # 添加批次维度
        
        # 神经编码
        with torch.set_grad_enabled(self.training):
            encoded = self.neural_encoder(neural_tensor)
        
        # 符号提取
        symbols = {
            "entities": [],
            "relations": [],
            "properties": [],
            "confidence_scores": {}
        }
        
        # 实体提取
        if extract_types is None or SymbolType.ENTITY in extract_types:
            entity_features = self.symbol_extractor["entity_detector"](encoded)
            detected_entities = self._match_symbols(entity_features, SymbolType.ENTITY)
            symbols["entities"] = detected_entities
        
        # 属性提取
        if extract_types is None or SymbolType.PROPERTY in extract_types:
            property_features = self.symbol_extractor["property_detector"](encoded)
            detected_properties = self._match_symbols(property_features, SymbolType.PROPERTY)
            symbols["properties"] = detected_properties
        
        # 关系提取（需要多个实体）
        if extract_types is None or SymbolType.RELATION in extract_types:
            if len(symbols["entities"]) >= 2:
                # 对实体对提取关系
                for i in range(len(symbols["entities"])):
                    for j in range(i + 1, len(symbols["entities"])):
                        # 连接两个实体的特征
                        entity_i_feat = self._get_symbol_vector(symbols["entities"][i]["id"])
                        entity_j_feat = self._get_symbol_vector(symbols["entities"][j]["id"])
                        
                        if entity_i_feat is not None and entity_j_feat is not None:
                            # 转换为张量
                            entity_i_tensor = torch.tensor(entity_i_feat, dtype=torch.float32).to(self.device)
                            entity_j_tensor = torch.tensor(entity_j_feat, dtype=torch.float32).to(self.device)
                            
                            # 连接特征
                            pair_features = torch.cat([entity_i_tensor, entity_j_tensor], dim=0)
                            if len(pair_features.shape) == 1:
                                pair_features = pair_features.unsqueeze(0)
                            
                            # 关系检测
                            relation_features = self.symbol_extractor["relation_detector"](pair_features)
                            detected_relations = self._match_symbols(relation_features, SymbolType.RELATION)
                            
                            for rel in detected_relations:
                                rel["entities"] = [symbols["entities"][i]["id"], symbols["entities"][j]["id"]]
                                if "relations" not in symbols:
                                    symbols["relations"] = []
                                symbols["relations"].append(rel)
        
        # 计算置信度分数
        symbols["confidence_scores"] = {
            "entity_confidence": np.mean([e.get("similarity", 0) for e in symbols["entities"]]) if symbols["entities"] else 0,
            "property_confidence": np.mean([p.get("similarity", 0) for p in symbols["properties"]]) if symbols["properties"] else 0,
            "relation_confidence": np.mean([r.get("similarity", 0) for r in symbols.get("relations", [])]) if symbols.get("relations") else 0
        }
        
        elapsed_time = time.time() - start_time
        symbols["performance"] = {
            "extraction_time": elapsed_time,
            "neural_input_shape": list(neural_input.shape) if hasattr(neural_input, 'shape') else "unknown",
            "symbols_extracted": len(symbols["entities"]) + len(symbols["properties"]) + len(symbols.get("relations", []))
        }
        
        self.performance_stats["neural_to_symbol_translations"] += 1
        logger.debug(f"神经到符号转换完成: {len(symbols['entities'])}实体, {len(symbols['properties'])}属性, {len(symbols.get('relations', []))}关系")
        
        return symbols
    
    def _match_symbols(self, 
                      features: torch.Tensor, 
                      symbol_type: SymbolType) -> List[Dict[str, Any]]:
        """
        匹配特征到符号
        
        Args:
            features: 特征张量
            symbol_type: 符号类型
            
        Returns:
            匹配的符号列表
        """
        matched_symbols = []
        
        # 获取目标符号库
        if symbol_type == SymbolType.ENTITY:
            symbol_dict = self.symbol_knowledge_base["entities"]
        elif symbol_type == SymbolType.RELATION:
            symbol_dict = self.symbol_knowledge_base["relations"]
        elif symbol_type == SymbolType.PROPERTY:
            symbol_dict = self.symbol_knowledge_base["properties"]
        else:
            return matched_symbols
        
        # 转换为numpy用于相似度计算
        if features.is_cuda:
            features_np = features.cpu().detach().numpy()
        else:
            features_np = features.detach().numpy()
        
        # 如果特征是多维的，取平均值
        if len(features_np.shape) > 1:
            features_np = np.mean(features_np, axis=0)
        
        # 计算与每个符号的相似度
        for symbol_id, symbol_info in symbol_dict.items():
            symbol_vector = symbol_info.get("vector")
            if symbol_vector is not None:
                # 计算余弦相似度
                similarity = self._cosine_similarity(features_np, symbol_vector)
                
                if similarity > self.symbol_similarity_threshold:
                    matched_symbols.append({
                        "id": symbol_id,
                        "name": symbol_info.get("name", symbol_id),
                        "type": symbol_type.value,
                        "similarity": float(similarity),
                        "vector": symbol_vector.tolist() if hasattr(symbol_vector, 'tolist') else symbol_vector
                    })
        
        # 按相似度排序
        matched_symbols.sort(key=lambda x: x["similarity"], reverse=True)
        
        # 限制返回数量
        max_symbols = 5 if symbol_type == SymbolType.ENTITY else 3
        return matched_symbols[:max_symbols]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _get_symbol_vector(self, symbol_id: str) -> Optional[np.ndarray]:
        """获取符号向量"""
        # 在实体中查找
        if symbol_id in self.symbol_knowledge_base["entities"]:
            return self.symbol_knowledge_base["entities"][symbol_id].get("vector")
        
        # 在关系中查找
        if symbol_id in self.symbol_knowledge_base["relations"]:
            return self.symbol_knowledge_base["relations"][symbol_id].get("vector")
        
        # 在属性中查找
        if symbol_id in self.symbol_knowledge_base["properties"]:
            return self.symbol_knowledge_base["properties"][symbol_id].get("vector")
        
        return None
    
    def symbol_to_neural(self, 
                        symbol_expression: Union[str, Dict[str, Any]],
                        target_shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """
        符号到神经表示的转换（符号约束指导）
        
        Args:
            symbol_expression: 符号表达式或符号字典
            target_shape: 目标神经表示的形状
            
        Returns:
            神经约束或表示
        """
        start_time = time.time()
        
        # 解析符号表达式
        if isinstance(symbol_expression, str):
            symbols = self._parse_symbol_expression(symbol_expression)
        else:
            symbols = symbol_expression
        
        # 提取符号向量
        symbol_vectors = []
        symbol_ids = []
        
        # 收集实体向量
        if "entities" in symbols:
            for entity in symbols["entities"]:
                entity_id = entity if isinstance(entity, str) else entity.get("id")
                vec = self._get_symbol_vector(entity_id)
                if vec is not None:
                    symbol_vectors.append(vec)
                    symbol_ids.append(entity_id)
        
        # 收集属性向量
        if "properties" in symbols:
            for prop in symbols["properties"]:
                prop_id = prop if isinstance(prop, str) else prop.get("id")
                vec = self._get_symbol_vector(prop_id)
                if vec is not None:
                    symbol_vectors.append(vec)
                    symbol_ids.append(prop_id)
        
        # 收集关系向量
        if "relations" in symbols:
            for rel in symbols["relations"]:
                rel_id = rel if isinstance(rel, str) else rel.get("id")
                vec = self._get_symbol_vector(rel_id)
                if vec is not None:
                    symbol_vectors.append(vec)
                    symbol_ids.append(rel_id)
        
        if not symbol_vectors:
            logger.warning("符号到神经转换: 未找到有效的符号向量")
            # 返回零向量
            if target_shape:
                return torch.zeros(target_shape, device=self.device)
            else:
                return torch.zeros(self.input_dim, device=self.device)
        
        # 合并符号向量（平均池化）
        combined_vector = np.mean(symbol_vectors, axis=0)
        
        # 通过符号解码器转换为神经表示
        symbol_tensor = torch.tensor(combined_vector, dtype=torch.float32).to(self.device)
        if len(symbol_tensor.shape) == 1:
            symbol_tensor = symbol_tensor.unsqueeze(0)
        
        with torch.set_grad_enabled(self.training):
            neural_constraint = self.symbol_decoder(symbol_tensor)
        
        # 调整形状
        if target_shape:
            neural_constraint = neural_constraint.view(*target_shape)
        
        elapsed_time = time.time() - start_time
        
        self.performance_stats["symbol_to_neural_constraints"] += 1
        logger.debug(f"符号到神经转换完成: {len(symbol_ids)}个符号, 输出形状: {neural_constraint.shape}")
        
        return neural_constraint
    
    def _parse_symbol_expression(self, expression: str) -> Dict[str, List[str]]:
        """
        解析符号表达式
        
        Args:
            expression: 符号表达式字符串
            
        Returns:
            解析后的符号字典
        """
        # 简化解析：支持简单的符号列表
        # 例如: "entity:person,property:red,relation:above"
        symbols = {"entities": [], "properties": [], "relations": []}
        
        try:
            parts = expression.split(',')
            for part in parts:
                part = part.strip()
                if ':' in part:
                    symbol_type, symbol_name = part.split(':', 1)
                    symbol_type = symbol_type.strip().lower()
                    symbol_name = symbol_name.strip()
                    
                    if symbol_type == 'entity':
                        symbols["entities"].append(symbol_name)
                    elif symbol_type == 'property':
                        symbols["properties"].append(symbol_name)
                    elif symbol_type == 'relation':
                        symbols["relations"].append(symbol_name)
        except Exception as e:
            logger.warning(f"符号表达式解析失败: {expression}, 错误: {e}")
        
        return symbols
    
    def logical_inference(self, 
                         premises: List[str],
                         conclusion_template: Optional[str] = None) -> Dict[str, Any]:
        """
        逻辑推理
        
        Args:
            premises: 前提列表
            conclusion_template: 结论模板（可选）
            
        Returns:
            推理结果
        """
        start_time = time.time()
        
        # 简化逻辑推理实现
        # 实际实现应使用定理证明器或逻辑编程引擎
        
        # 解析前提
        parsed_premises = []
        for premise in premises:
            parsed = self._parse_logical_expression(premise)
            parsed_premises.append(parsed)
        
        # 应用逻辑规则
        inferred_conclusions = []
        for rule in self.logic_rules:
            # 简化：检查规则是否可应用
            rule_applicable = self._check_rule_applicability(rule, parsed_premises)
            if rule_applicable:
                conclusion = self._apply_rule(rule, parsed_premises)
                if conclusion:
                    inferred_conclusions.append(conclusion)
        
        # 生成结论
        if conclusion_template and inferred_conclusions:
            # 使用模板生成结论
            final_conclusion = conclusion_template
            for i, conc in enumerate(inferred_conclusions[:3]):  # 最多使用3个结论
                final_conclusion = final_conclusion.replace(f"{{conclusion_{i+1}}}", conc)
        elif inferred_conclusions:
            final_conclusion = inferred_conclusions[0]
        else:
            final_conclusion = "无法从给定前提推导出结论"
        
        # 计算置信度
        confidence = min(0.9, 0.5 + 0.1 * len(inferred_conclusions))
        
        elapsed_time = time.time() - start_time
        result = {
            "premises": premises,
            "conclusions": inferred_conclusions,
            "final_conclusion": final_conclusion,
            "confidence": confidence,
            "rules_applied": len(inferred_conclusions),
            "performance": {
                "inference_time": elapsed_time,
                "premises_count": len(premises),
                "rules_checked": len(self.logic_rules)
            }
        }
        
        self.performance_stats["logic_inferences"] += 1
        logger.info(f"逻辑推理完成: {len(premises)}前提 → {len(inferred_conclusions)}结论, 置信度: {confidence:.2f}")
        
        return result
    
    def _parse_logical_expression(self, expression: str) -> Dict[str, Any]:
        """解析逻辑表达式"""
        # 简化解析
        return {
            "original": expression,
            "tokens": expression.split(),
            "has_quantifiers": any(q in expression.lower() for q in ["forall", "exists", "∀", "∃"]),
            "has_operators": any(op in expression for op in ["∧", "∨", "¬", "→", "↔", "and", "or", "not", "implies"])
        }
    
    def _check_rule_applicability(self, rule: str, premises: List[Dict[str, Any]]) -> bool:
        """检查规则是否可应用"""
        # 简化检查：检查规则中的关键词是否出现在前提中
        rule_keywords = set(rule.lower().split())
        for premise in premises:
            premise_text = premise["original"].lower()
            premise_keywords = set(premise_text.split())
            if rule_keywords.intersection(premise_keywords):
                return True
        return False
    
    def _apply_rule(self, rule: str, premises: List[Dict[str, Any]]) -> Optional[str]:
        """应用规则推导结论"""
        # 简化实现：返回规则的一部分作为结论
        # 实际实现应执行真正的逻辑推导
        
        # 提取规则结论部分（简化：假设规则为"前提 → 结论"形式）
        if "→" in rule:
            parts = rule.split("→", 1)
            if len(parts) == 2:
                return parts[1].strip()
        elif "implies" in rule.lower():
            parts = rule.lower().split("implies", 1)
            if len(parts) == 2:
                return parts[1].strip()
        
        return None
    
    def enforce_symbolic_constraints(self,
                                    neural_output: torch.Tensor,
                                    constraints: List[str]) -> torch.Tensor:
        """
        强制执行符号约束
        
        Args:
            neural_output: 神经网络的输出
            constraints: 符号约束列表
            
        Returns:
            应用约束后的输出
        """
        start_time = time.time()
        
        # 将约束转换为神经指导信号
        constraint_signals = []
        for constraint in constraints:
            # 将符号约束转换为神经表示
            constraint_signal = self.symbol_to_neural(constraint, neural_output.shape)
            constraint_signals.append(constraint_signal)
        
        if constraint_signals:
            # 合并约束信号（平均）
            combined_constraint = torch.mean(torch.stack(constraint_signals), dim=0)
            
            # 应用约束：混合原始输出和约束信号
            # 使用可学习的混合权重（简化：固定权重）
            alpha = 0.3  # 约束强度
            constrained_output = (1 - alpha) * neural_output + alpha * combined_constraint
        else:
            constrained_output = neural_output
        
        elapsed_time = time.time() - start_time
        
        self.performance_stats["symbol_to_neural_constraints"] += len(constraints)
        logger.debug(f"符号约束强制执行: {len(constraints)}个约束, 耗时: {elapsed_time:.3f}秒")
        
        return constrained_output
    
    def check_consistency(self,
                         neural_representation: torch.Tensor,
                         symbolic_representation: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查神经符号一致性
        
        Args:
            neural_representation: 神经表示
            symbolic_representation: 符号表示
            
        Returns:
            一致性检查结果
        """
        start_time = time.time()
        
        # 从神经表示中提取符号
        extracted_symbols = self.neural_to_symbol(neural_representation)
        
        # 比较提取的符号与给定的符号
        consistency_scores = {}
        
        # 实体一致性
        given_entities = set(symbolic_representation.get("entities", []))
        extracted_entities = set([e["name"] for e in extracted_symbols.get("entities", [])])
        entity_consistency = len(given_entities.intersection(extracted_entities)) / max(1, len(given_entities))
        consistency_scores["entity_consistency"] = entity_consistency
        
        # 属性一致性
        given_properties = set(symbolic_representation.get("properties", []))
        extracted_properties = set([p["name"] for p in extracted_symbols.get("properties", [])])
        property_consistency = len(given_properties.intersection(extracted_properties)) / max(1, len(given_properties))
        consistency_scores["property_consistency"] = property_consistency
        
        # 关系一致性（简化）
        relation_consistency = 0.5  # 默认值
        consistency_scores["relation_consistency"] = relation_consistency
        
        # 总体一致性
        overall_consistency = np.mean(list(consistency_scores.values()))
        consistency_scores["overall_consistency"] = overall_consistency
        
        # 不一致的项
        inconsistencies = {
            "missing_entities": list(given_entities - extracted_entities),
            "extra_entities": list(extracted_entities - given_entities),
            "missing_properties": list(given_properties - extracted_properties),
            "extra_properties": list(extracted_properties - given_properties)
        }
        
        elapsed_time = time.time() - start_time
        result = {
            "consistency_scores": consistency_scores,
            "inconsistencies": inconsistencies,
            "extracted_symbols": extracted_symbols,
            "given_symbols": symbolic_representation,
            "performance": {
                "check_time": elapsed_time,
                "entities_compared": len(given_entities) + len(extracted_entities),
                "properties_compared": len(given_properties) + len(extracted_properties)
            }
        }
        
        self.performance_stats["consistency_checks"] += 1
        logger.info(f"一致性检查完成: 总体一致性={overall_consistency:.2f}, 实体={entity_consistency:.2f}, 属性={property_consistency:.2f}")
        
        return result
    
    def learn_from_feedback(self,
                           neural_input: torch.Tensor,
                           expected_symbols: Dict[str, Any],
                           learning_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        从反馈中学习，改进符号提取
        
        Args:
            neural_input: 神经输入
            expected_symbols: 期望的符号表示
            learning_rate: 学习率（如果为None则使用默认值）
            
        Returns:
            学习结果
        """
        start_time = time.time()
        
        if not self.training:
            logger.warning("学习模式已关闭，无法从反馈中学习")
            return {"success": False, "error": "training_mode_disabled"}
        
        # 设置训练模式
        self.neural_encoder.train()
        self.symbol_extractor.train()
        self.symbol_decoder.train()
        
        # 前向传播
        encoded = self.neural_encoder(neural_input)
        
        # 提取符号（用于计算损失）
        entity_features = self.symbol_extractor["entity_detector"](encoded)
        property_features = self.symbol_extractor["property_detector"](encoded)
        
        # 计算损失
        loss = 0.0
        loss_components = {}
        
        # 实体匹配损失
        if "entities" in expected_symbols and expected_symbols["entities"]:
            # 简化：鼓励提取的特征接近期望实体的向量
            entity_loss = self._compute_symbol_matching_loss(entity_features, expected_symbols["entities"], SymbolType.ENTITY)
            loss += entity_loss
            loss_components["entity_loss"] = float(entity_loss.item())
        
        # 属性匹配损失
        if "properties" in expected_symbols and expected_symbols["properties"]:
            property_loss = self._compute_symbol_matching_loss(property_features, expected_symbols["properties"], SymbolType.PROPERTY)
            loss += property_loss
            loss_components["property_loss"] = float(property_loss.item())
        
        # 重建损失（确保符号可解码回神经表示）
        if "original_input" in expected_symbols:
            original_input = expected_symbols["original_input"]
            # 通过符号解码器重建
            symbol_vector = torch.mean(torch.cat([entity_features, property_features], dim=1), dim=1, keepdim=True)
            reconstructed = self.symbol_decoder(symbol_vector)
            recon_loss = F.mse_loss(reconstructed, original_input)
            loss += recon_loss * 0.1  # 较小的权重
            loss_components["reconstruction_loss"] = float(recon_loss.item())
        
        # 反向传播和优化
        if loss > 0:
            self.neural_optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.neural_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.symbol_extractor.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.symbol_decoder.parameters(), max_norm=1.0)
            
            self.neural_optimizer.step()
            self.scheduler.step()
        
        elapsed_time = time.time() - start_time
        learning_result = {
            "success": True,
            "total_loss": float(loss.item()) if hasattr(loss, 'item') else float(loss),
            "loss_components": loss_components,
            "learning_rate": self.neural_optimizer.param_groups[0]['lr'] if self.neural_optimizer.param_groups else self.learning_rate,
            "performance": {
                "learning_time": elapsed_time,
                "parameters_updated": sum(p.numel() for p in self.neural_encoder.parameters() if p.requires_grad) +
                                     sum(p.numel() for p in self.symbol_extractor.parameters() if p.requires_grad) +
                                     sum(p.numel() for p in self.symbol_decoder.parameters() if p.requires_grad)
            }
        }
        
        self.performance_stats["learning_updates"] += 1
        logger.info(f"从反馈中学习完成: 总损失={learning_result['total_loss']:.4f}, 学习率={learning_result['learning_rate']:.6f}")
        
        return learning_result
    
    def _compute_symbol_matching_loss(self, 
                                     features: torch.Tensor,
                                     expected_symbols: List[str],
                                     symbol_type: SymbolType) -> torch.Tensor:
        """
        计算符号匹配损失
        
        Args:
            features: 提取的特征
            expected_symbols: 期望的符号列表
            symbol_type: 符号类型
            
        Returns:
            匹配损失
        """
        # 获取期望符号的向量
        target_vectors = []
        for symbol in expected_symbols:
            symbol_id = self.concept_to_symbol.get(symbol)
            if symbol_id:
                vec = self._get_symbol_vector(symbol_id)
                if vec is not None:
                    target_vectors.append(torch.tensor(vec, dtype=torch.float32).to(self.device))
        
        if not target_vectors:
            return torch.tensor(0.0, device=self.device)
        
        # 计算特征与目标向量的相似度损失
        target_tensor = torch.stack(target_vectors).mean(dim=0)  # 平均目标向量
        
        # 确保特征维度匹配
        if features.shape[-1] != target_tensor.shape[-1]:
            # 调整特征维度
            if features.shape[-1] > target_tensor.shape[-1]:
                features = features[:, :target_tensor.shape[-1]]
            else:
                padding = torch.zeros(features.shape[0], target_tensor.shape[-1] - features.shape[-1], 
                                     device=self.device)
                features = torch.cat([features, padding], dim=1)
        
        # 计算余弦相似度损失（鼓励高相似度）
        similarity = F.cosine_similarity(features, target_tensor.unsqueeze(0), dim=1)
        loss = 1.0 - similarity.mean()  # 1 - 相似度作为损失
        
        return loss
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = self.performance_stats.copy()
        stats["knowledge_base_stats"] = {
            "entities": len(self.symbol_knowledge_base["entities"]),
            "relations": len(self.symbol_knowledge_base["relations"]),
            "properties": len(self.symbol_knowledge_base["properties"]),
            "rules": len(self.symbol_knowledge_base["rules"]),
            "constraints": len(self.symbol_knowledge_base["constraints"])
        }
        stats["model_info"] = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "symbol_dim": self.symbol_dim,
            "device": str(self.device),
            "training_mode": self.training
        }
        return stats
    
    def train_mode(self, enabled: bool = True):
        """设置训练模式"""
        self.training = enabled
        if enabled:
            self.neural_encoder.train()
            self.symbol_extractor.train()
            self.symbol_decoder.train()
        else:
            self.neural_encoder.eval()
            self.symbol_extractor.eval()
            self.symbol_decoder.eval()
        logger.info(f"训练模式: {'启用' if enabled else '禁用'}")
    
    def save_model(self, filepath: str) -> bool:
        """保存模型到文件"""
        try:
            import pickle
            model_data = {
                "neural_encoder_state": self.neural_encoder.state_dict(),
                "symbol_extractor_state": self.symbol_extractor.state_dict(),
                "symbol_decoder_state": self.symbol_decoder.state_dict(),
                "symbol_knowledge_base": self.symbol_knowledge_base,
                "performance_stats": self.performance_stats,
                "symbol_to_concept": self.symbol_to_concept,
                "concept_to_symbol": self.concept_to_symbol,
                "logic_rules": self.logic_rules
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"神经符号模型保存到: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存神经符号模型失败: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """从文件加载模型"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # 加载状态字典
            self.neural_encoder.load_state_dict(model_data["neural_encoder_state"])
            self.symbol_extractor.load_state_dict(model_data["symbol_extractor_state"])
            self.symbol_decoder.load_state_dict(model_data["symbol_decoder_state"])
            
            # 加载知识库
            self.symbol_knowledge_base = model_data.get("symbol_knowledge_base", self.symbol_knowledge_base)
            self.performance_stats = model_data.get("performance_stats", self.performance_stats.copy())
            self.symbol_to_concept = model_data.get("symbol_to_concept", {})
            self.concept_to_symbol = model_data.get("concept_to_symbol", {})
            self.logic_rules = model_data.get("logic_rules", [])
            
            logger.info(f"神经符号模型从 {filepath} 加载")
            return True
        except Exception as e:
            logger.error(f"加载神经符号模型失败: {e}")
            return False


# 示例和测试函数
def create_example_framework() -> NeuralSymbolicUnifiedFramework:
    """创建示例神经符号框架"""
    framework = NeuralSymbolicUnifiedFramework(
        input_dim=256,
        hidden_dim=128,
        symbol_dim=64,
        learning_rate=0.001
    )
    return framework


def test_neural_symbolic_framework():
    """测试神经符号统一框架"""
    logger.info("开始测试神经符号统一框架")
    
    # 创建示例框架
    framework = create_example_framework()
    
    # 创建示例神经输入
    batch_size = 4
    input_dim = 256
    example_input = _deterministic_randn((batch_size, input_dim), seed_prefix="randn_default").to(framework.device)
    
    # 测试神经到符号转换
    logger.info("测试神经到符号转换...")
    symbols = framework.neural_to_symbol(example_input)
    logger.info(f"提取的符号: {len(symbols['entities'])}实体, {len(symbols['properties'])}属性")
    
    # 测试符号到神经转换
    logger.info("测试符号到神经转换...")
    symbol_expression = "entity:person,property:red,relation:above"
    neural_constraint = framework.symbol_to_neural(symbol_expression, example_input.shape)
    logger.info(f"生成的神经约束形状: {neural_constraint.shape}")
    
    # 测试逻辑推理
    logger.info("测试逻辑推理...")
    premises = [
        "所有人类都是哺乳动物",
        "苏格拉底是人类"
    ]
    inference_result = framework.logical_inference(premises, "因此{{conclusion_1}}")
    logger.info(f"逻辑推理结果: {inference_result['final_conclusion']}, 置信度: {inference_result['confidence']:.2f}")
    
    # 测试符号约束强制执行
    logger.info("测试符号约束强制执行...")
    neural_output = _deterministic_randn((batch_size, input_dim), seed_prefix="randn_default").to(framework.device)
    constraints = ["entity:object", "property:large"]
    constrained_output = framework.enforce_symbolic_constraints(neural_output, constraints)
    logger.info(f"约束应用前后形状: {neural_output.shape} → {constrained_output.shape}")
    
    # 测试一致性检查
    logger.info("测试一致性检查...")
    symbolic_rep = {
        "entities": ["person", "object"],
        "properties": ["red", "large"]
    }
    consistency_result = framework.check_consistency(example_input, symbolic_rep)
    logger.info(f"一致性检查: 总体一致性={consistency_result['consistency_scores']['overall_consistency']:.2f}")
    
    # 测试学习反馈
    logger.info("测试从反馈中学习...")
    expected_symbols = {
        "entities": ["person", "object"],
        "properties": ["red"],
        "original_input": example_input
    }
    learning_result = framework.learn_from_feedback(example_input, expected_symbols)
    logger.info(f"学习结果: 总损失={learning_result['total_loss']:.4f}")
    
    # 显示性能统计
    stats = framework.get_performance_stats()
    logger.info(f"性能统计: {stats}")
    
    logger.info("神经符号统一框架测试完成")
    return framework


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_framework = test_neural_symbolic_framework()