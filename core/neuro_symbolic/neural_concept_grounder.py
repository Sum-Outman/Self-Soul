#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
神经概念接地器 - 实现神经表示到符号概念的接地（Grounding）

核心功能:
1. 神经特征到符号概念的映射学习
2. 多模态感知的符号接地
3. 概念层次结构的构建和维护
4. 接地质量的评估和验证
5. 自适应概念学习和更新

接地类型:
- 感知接地: 视觉、听觉等感知特征到符号的映射
- 语义接地: 语言描述到概念的映射
- 功能接地: 对象功能到用途概念的映射
- 关系接地: 空间/时间关系到关系符号的映射

技术实现:
- 神经网络嵌入学习
- 符号相似度计算
- 聚类和分类算法
- 增量学习和概念演化

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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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

class GroundingType(Enum):
    """接地类型枚举"""
    PERCEPTUAL = "perceptual"      # 感知接地: 视觉、听觉特征
    SEMANTIC = "semantic"          # 语义接地: 语言描述
    FUNCTIONAL = "functional"      # 功能接地: 用途和功能
    RELATIONAL = "relational"      # 关系接地: 空间/时间关系
    AFFORDANCE = "affordance"      # 可供性接地: 交互可能性


class ConceptType(Enum):
    """概念类型枚举"""
    BASIC_LEVEL = "basic_level"    # 基础层次概念: 猫、狗、椅子
    SUPERORDINATE = "superordinate"  # 上位概念: 动物、家具
    SUBORDINATE = "subordinate"    # 下位概念: 波斯猫、办公椅
    ABSTRACT = "abstract"          # 抽象概念: 美丽、正义
    RELATIONAL = "relational"      # 关系概念: 之上、之内


class NeuralConceptGrounder:
    """
    神经概念接地器 - 实现神经表示到符号概念的接地
    
    核心组件:
    1. EmbeddingModel: 神经特征嵌入模型
    2. ConceptMapper: 概念映射器
    3. SimilarityCalculator: 相似度计算器
    4. ConceptHierarchy: 概念层次结构
    5. GroundingValidator: 接地验证器
    
    工作流程:
    神经特征 → EmbeddingModel → 特征向量 → ConceptMapper → 符号概念
    符号概念 → ConceptHierarchy → 概念关系 → GroundingValidator → 接地质量
    反馈信号 → 更新EmbeddingModel和ConceptMapper → 改进接地
    
    技术特性:
    - 多模态特征融合
    - 增量概念学习
    - 层次概念结构
    - 接地一致性检查
    - 自适应概念演化
    """
    
    def __init__(self,
                 feature_dim: int = 512,
                 embedding_dim: int = 256,
                 concept_dim: int = 128,
                 learning_rate: float = 0.001,
                 similarity_threshold: float = 0.7):
        """
        初始化神经概念接地器
        
        Args:
            feature_dim: 特征维度
            embedding_dim: 嵌入维度
            concept_dim: 概念维度
            learning_rate: 学习率
            similarity_threshold: 相似度阈值
        """
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.concept_dim = concept_dim
        self.learning_rate = learning_rate
        self.similarity_threshold = similarity_threshold
        
        # 初始化神经网络组件
        self._init_neural_components()
        
        # 初始化概念系统
        self._init_concept_system()
        
        # 初始化优化器
        self._init_optimizers()
        
        # 性能统计
        self.performance_stats = {
            "groundings_performed": 0,
            "concepts_learned": 0,
            "similarities_computed": 0,
            "hierarchy_updates": 0,
            "validation_checks": 0
        }
        
        # 训练状态
        self.training = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"神经概念接地器初始化完成，设备: {self.device}")
    
    def _init_neural_components(self):
        """初始化神经网络组件"""
        # 特征嵌入器：原始特征 → 嵌入空间
        self.feature_embedder = nn.Sequential(
            nn.Linear(self.feature_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.concept_dim)
        )
        
        # 概念投影器：概念符号 → 概念空间
        self.concept_projector = nn.Sequential(
            nn.Linear(self.concept_dim * 2, self.concept_dim),  # 概念+上下文
            nn.ReLU(),
            nn.Linear(self.concept_dim, self.concept_dim),
            nn.Tanh()  # 输出在[-1, 1]范围内
        )
        
        # 概念解码器：概念空间 → 特征重建
        self.concept_decoder = nn.Sequential(
            nn.Linear(self.concept_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.feature_dim)
        )
        
        # 移动到设备
        self.feature_embedder.to(self.device)
        self.concept_projector.to(self.device)
        self.concept_decoder.to(self.device)
    
    def _init_concept_system(self):
        """初始化概念系统"""
        # 概念库
        self.concept_library = {
            "concepts": {},           # 概念字典 {concept_id: concept_info}
            "embeddings": {},         # 概念嵌入 {concept_id: embedding_vector}
            "hierarchy": nx.DiGraph(),  # 概念层次图
            "grounding_examples": {}   # 接地示例 {concept_id: [example_features]}
        }
        
        # 概念计数器
        self.concept_counter = 0
        
        # 初始化基础概念
        self._initialize_basic_concepts()
    
    def _initialize_basic_concepts(self):
        """初始化基础概念"""
        basic_concepts = [
            # 基础层次概念
            {"name": "object", "type": ConceptType.BASIC_LEVEL, "grounding_type": GroundingType.PERCEPTUAL},
            {"name": "person", "type": ConceptType.BASIC_LEVEL, "grounding_type": GroundingType.PERCEPTUAL},
            {"name": "animal", "type": ConceptType.SUPERORDINATE, "grounding_type": GroundingType.PERCEPTUAL},
            {"name": "vehicle", "type": ConceptType.SUPERORDINATE, "grounding_type": GroundingType.FUNCTIONAL},
            
            # 属性概念
            {"name": "red", "type": ConceptType.BASIC_LEVEL, "grounding_type": GroundingType.PERCEPTUAL},
            {"name": "large", "type": ConceptType.BASIC_LEVEL, "grounding_type": GroundingType.PERCEPTUAL},
            {"name": "round", "type": ConceptType.BASIC_LEVEL, "grounding_type": GroundingType.PERCEPTUAL},
            
            # 关系概念
            {"name": "above", "type": ConceptType.RELATIONAL, "grounding_type": GroundingType.RELATIONAL},
            {"name": "inside", "type": ConceptType.RELATIONAL, "grounding_type": GroundingType.RELATIONAL},
            {"name": "near", "type": ConceptType.RELATIONAL, "grounding_type": GroundingType.RELATIONAL}
        ]
        
        for concept_info in basic_concepts:
            self.add_concept(
                name=concept_info["name"],
                concept_type=concept_info["type"],
                grounding_type=concept_info["grounding_type"]
            )
        
        # 构建基础层次结构
        self._build_basic_hierarchy()
        
        logger.info(f"初始化基础概念: {len(basic_concepts)}个概念")
    
    def _build_basic_hierarchy(self):
        """构建基础层次结构"""
        # 添加层次关系
        hierarchy_relations = [
            # 上位-下位关系
            ("animal", "person"),
            ("object", "vehicle"),
            
            # 部分-整体关系 (简化)
            ("vehicle", "wheel"),
            ("person", "head"),
            
            # 属性关系
            ("object", "red"),
            ("object", "large"),
            ("object", "round")
        ]
        
        for parent, child in hierarchy_relations:
            # 确保子概念存在
            if child not in self.concept_library["concepts"]:
                self.add_concept(child, ConceptType.BASIC_LEVEL, GroundingType.PERCEPTUAL)
            
            self.add_hierarchical_relation(parent, child, relation_type="is_a")
    
    def _init_optimizers(self):
        """初始化优化器"""
        # 神经网络参数
        neural_params = list(self.feature_embedder.parameters()) + \
                       list(self.concept_projector.parameters()) + \
                       list(self.concept_decoder.parameters())
        
        self.neural_optimizer = optim.Adam(neural_params, lr=self.learning_rate)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.neural_optimizer, step_size=500, gamma=0.95)
    
    def add_concept(self,
                   name: str,
                   concept_type: ConceptType = ConceptType.BASIC_LEVEL,
                   grounding_type: GroundingType = GroundingType.PERCEPTUAL,
                   initial_embedding: Optional[np.ndarray] = None) -> str:
        """
        添加新概念
        
        Args:
            name: 概念名称
            concept_type: 概念类型
            grounding_type: 接地类型
            initial_embedding: 初始嵌入向量（如果为None则随机生成）
            
        Returns:
            概念ID
        """
        concept_id = f"concept_{self.concept_counter}"
        self.concept_counter += 1
        
        # 生成概念嵌入
        if initial_embedding is not None:
            embedding = initial_embedding
        else:
            # 确定性生成基于名称的嵌入
            embedding = self._generate_concept_embedding(name, concept_type, grounding_type)
        
        # 归一化嵌入
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm > 0:
            embedding = embedding / embedding_norm
        
        # 存储概念信息
        self.concept_library["concepts"][concept_id] = {
            "id": concept_id,
            "name": name,
            "type": concept_type.value,
            "grounding_type": grounding_type.value,
            "creation_time": time.time(),
            "usage_count": 0,
            "grounding_examples": 0,
            "confidence_score": 1.0  # 初始置信度
        }
        
        # 存储概念嵌入
        self.concept_library["embeddings"][concept_id] = embedding
        
        # 添加到层次图
        self.concept_library["hierarchy"].add_node(concept_id)
        
        # 初始化接地示例
        self.concept_library["grounding_examples"][concept_id] = []
        
        self.performance_stats["concepts_learned"] += 1
        logger.debug(f"添加概念: {name} (ID: {concept_id}, 类型: {concept_type.value}, 接地: {grounding_type.value})")
        
        return concept_id
    
    def _generate_concept_embedding(self,
                                   name: str,
                                   concept_type: ConceptType,
                                   grounding_type: GroundingType) -> np.ndarray:
        """
        生成概念嵌入向量
        
        Args:
            name: 概念名称
            concept_type: 概念类型
            grounding_type: 接地类型
            
        Returns:
            概念嵌入向量
        """
        # 确定性生成基于名称、类型和接地类型的嵌入
        vector = np.zeros(self.concept_dim, dtype=np.float32)
        
        # 使用哈希函数生成确定性值
        base_seed = f"{name}_{concept_type.value}_{grounding_type.value}"
        
        for i in range(self.concept_dim):
            hash_str = f"{base_seed}_dim_{i}"
            hash_val = abs((zlib.adler32(str(hash_str).encode('utf-8')) & 0xffffffff)) % 10000
            vector[i] = (hash_val / 10000.0) * 2 - 1  # 归一化到[-1, 1]
        
        return vector
    
    def add_hierarchical_relation(self,
                                 parent_concept: str,
                                 child_concept: str,
                                 relation_type: str = "is_a",
                                 confidence: float = 1.0) -> bool:
        """
        添加层次关系
        
        Args:
            parent_concept: 父概念名称或ID
            child_concept: 子概念名称或ID
            relation_type: 关系类型 ("is_a", "part_of", "has_property", 等)
            confidence: 关系置信度
            
        Returns:
            是否成功添加
        """
        # 查找概念ID
        parent_id = self._find_concept_id(parent_concept)
        child_id = self._find_concept_id(child_concept)
        
        if not parent_id or not child_id:
            logger.warning(f"添加层次关系失败: 概念不存在 - 父: {parent_concept}, 子: {child_concept}")
            return False
        
        # 添加边
        self.concept_library["hierarchy"].add_edge(
            parent_id, child_id,
            relation_type=relation_type,
            confidence=confidence,
            added_time=time.time()
        )
        
        self.performance_stats["hierarchy_updates"] += 1
        logger.debug(f"添加层次关系: {parent_id} --{relation_type}--> {child_id} (置信度: {confidence})")
        
        return True
    
    def _find_concept_id(self, concept_identifier: str) -> Optional[str]:
        """
        查找概念ID
        
        Args:
            concept_identifier: 概念名称或ID
            
        Returns:
            概念ID或None（如果未找到）
        """
        # 检查是否为概念ID
        if concept_identifier in self.concept_library["concepts"]:
            return concept_identifier
        
        # 检查是否为概念名称
        for concept_id, concept_info in self.concept_library["concepts"].items():
            if concept_info["name"] == concept_identifier:
                return concept_id
        
        return None
    
    def ground_feature_to_concept(self,
                                 feature: Union[torch.Tensor, np.ndarray],
                                 grounding_type: Optional[GroundingType] = None,
                                 k: int = 3,
                                 return_similarities: bool = True) -> Dict[str, Any]:
        """
        将特征接地到概念
        
        Args:
            feature: 输入特征
            grounding_type: 接地类型（如果为None则考虑所有类型）
            k: 返回的top-k概念数量
            return_similarities: 是否返回相似度
            
        Returns:
            接地结果
        """
        start_time = time.time()
        
        # 转换输入为张量
        if isinstance(feature, np.ndarray):
            feature_tensor = torch.tensor(feature, dtype=torch.float32).to(self.device)
        else:
            feature_tensor = feature.to(self.device)
        
        # 确保正确的维度
        if len(feature_tensor.shape) == 1:
            feature_tensor = feature_tensor.unsqueeze(0)  # 添加批次维度
        
        # 特征嵌入
        with torch.set_grad_enabled(self.training):
            feature_embedding = self.feature_embedder(feature_tensor)
        
        # 计算与所有概念的相似度
        similarities = []
        
        for concept_id, concept_embedding in self.concept_library["embeddings"].items():
            # 获取概念信息
            concept_info = self.concept_library["concepts"][concept_id]
            
            # 检查接地类型过滤
            if grounding_type and concept_info["grounding_type"] != grounding_type.value:
                continue
            
            # 转换为张量
            concept_tensor = torch.tensor(concept_embedding, dtype=torch.float32).to(self.device)
            if len(concept_tensor.shape) == 1:
                concept_tensor = concept_tensor.unsqueeze(0)
            
            # 计算余弦相似度
            similarity = F.cosine_similarity(feature_embedding, concept_tensor, dim=1)
            avg_similarity = similarity.mean().item()
            
            # 检查是否超过阈值
            if avg_similarity >= self.similarity_threshold:
                similarities.append({
                    "concept_id": concept_id,
                    "concept_name": concept_info["name"],
                    "concept_type": concept_info["type"],
                    "grounding_type": concept_info["grounding_type"],
                    "similarity": avg_similarity,
                    "confidence": concept_info["confidence_score"]
                })
        
        # 按相似度排序
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # 限制返回数量
        top_k = similarities[:k]
        
        # 更新概念使用计数
        for result in top_k:
            concept_id = result["concept_id"]
            self.concept_library["concepts"][concept_id]["usage_count"] += 1
        
        # 构建结果
        grounding_result = {
            "feature_shape": list(feature.shape) if hasattr(feature, 'shape') else "unknown",
            "feature_embedding_shape": list(feature_embedding.shape),
            "top_concepts": top_k,
            "grounding_type": grounding_type.value if grounding_type else "any",
            "similarity_threshold": self.similarity_threshold,
            "performance": {
                "grounding_time": time.time() - start_time,
                "concepts_considered": len(similarities),
                "concepts_matched": len(top_k)
            }
        }
        
        if return_similarities:
            grounding_result["all_similarities"] = similarities
        
        self.performance_stats["groundings_performed"] += 1
        self.performance_stats["similarities_computed"] += len(similarities)
        
        logger.debug(f"特征接地完成: {len(top_k)}个概念匹配, 最高相似度: {top_k[0]['similarity'] if top_k else 0:.3f}")
        
        return grounding_result
    
    def ground_multimodal_features(self,
                                  feature_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
                                  fusion_method: str = "average") -> Dict[str, Any]:
        """
        多模态特征接地
        
        Args:
            feature_dict: 特征字典 {modality: feature}
            fusion_method: 融合方法 ("average", "concat", "weighted")
            
        Returns:
            多模态接地结果
        """
        start_time = time.time()
        
        # 处理每个模态
        modality_results = {}
        modality_embeddings = []
        
        for modality, feature in feature_dict.items():
            # 接地单个模态
            result = self.ground_feature_to_concept(feature, k=5, return_similarities=False)
            modality_results[modality] = result
            
            # 提取特征嵌入
            if isinstance(feature, np.ndarray):
                feature_tensor = torch.tensor(feature, dtype=torch.float32).to(self.device)
            else:
                feature_tensor = feature.to(self.device)
            
            if len(feature_tensor.shape) == 1:
                feature_tensor = feature_tensor.unsqueeze(0)
            
            with torch.set_grad_enabled(self.training):
                modality_embedding = self.feature_embedder(feature_tensor)
                modality_embeddings.append(modulation_embedding)
        
        # 融合模态嵌入
        if modality_embeddings:
            if fusion_method == "average":
                # 平均融合
                fused_embedding = torch.mean(torch.stack(modality_embeddings), dim=0)
            elif fusion_method == "concat":
                # 拼接融合
                fused_embedding = torch.cat(modality_embeddings, dim=1)
            elif fusion_method == "weighted":
                # 加权平均（简化：等权重）
                weights = torch.ones(len(modality_embeddings), device=self.device) / len(modality_embeddings)
                fused_embedding = torch.sum(torch.stack(modality_embeddings) * weights.view(-1, 1, 1), dim=0)
            else:
                # 默认：平均融合
                fused_embedding = torch.mean(torch.stack(modality_embeddings), dim=0)
            
            # 接地融合特征
            fused_result = self.ground_feature_to_concept(fused_embedding, k=5, return_similarities=False)
        else:
            fused_embedding = None
            fused_result = {"top_concepts": []}
        
        elapsed_time = time.time() - start_time
        
        # 构建多模态结果
        multimodal_result = {
            "modality_results": modality_results,
            "fused_result": fused_result,
            "fusion_method": fusion_method,
            "modalities_processed": list(feature_dict.keys()),
            "performance": {
                "total_time": elapsed_time,
                "modalities_count": len(feature_dict)
            }
        }
        
        logger.info(f"多模态接地完成: {len(feature_dict)}个模态, 融合方法: {fusion_method}")
        
        return multimodal_result
    
    def learn_from_grounding_example(self,
                                   feature: Union[torch.Tensor, np.ndarray],
                                   target_concept: str,
                                   learning_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        从接地示例中学习
        
        Args:
            feature: 输入特征
            target_concept: 目标概念名称或ID
            learning_rate: 学习率（如果为None则使用默认值）
            
        Returns:
            学习结果
        """
        start_time = time.time()
        
        if not self.training:
            logger.warning("学习模式已关闭，无法从示例中学习")
            return {"success": False, "error": "training_mode_disabled"}
        
        # 查找概念ID
        concept_id = self._find_concept_id(target_concept)
        if not concept_id:
            # 如果概念不存在，创建新概念
            concept_id = self.add_concept(
                name=target_concept,
                concept_type=ConceptType.BASIC_LEVEL,
                grounding_type=GroundingType.PERCEPTUAL
            )
            logger.info(f"创建新概念: {target_concept} (ID: {concept_id})")
        
        # 转换输入为张量
        if isinstance(feature, np.ndarray):
            feature_tensor = torch.tensor(feature, dtype=torch.float32).to(self.device)
        else:
            feature_tensor = feature.to(self.device)
        
        if len(feature_tensor.shape) == 1:
            feature_tensor = feature_tensor.unsqueeze(0)
        
        # 设置训练模式
        self.feature_embedder.train()
        self.concept_projector.train()
        self.concept_decoder.train()
        
        # 前向传播
        feature_embedding = self.feature_embedder(feature_tensor)
        
        # 获取目标概念嵌入
        target_embedding = self.concept_library["embeddings"][concept_id]
        target_tensor = torch.tensor(target_embedding, dtype=torch.float32).to(self.device)
        if len(target_tensor.shape) == 1:
            target_tensor = target_tensor.unsqueeze(0)
        
        # 计算损失
        # 1. 相似度损失：鼓励特征嵌入接近概念嵌入
        similarity_loss = 1.0 - F.cosine_similarity(feature_embedding, target_tensor, dim=1).mean()
        
        # 2. 重建损失：概念应该能够重建特征
        # 首先通过概念投影器
        concept_context = _deterministic_randn((feature_embedding.shape[0], self.concept_dim, device=self.device), seed_prefix="randn_default")  # 模拟上下文
        projected_concept = self.concept_projector(torch.cat([target_tensor, concept_context], dim=1))
        
        # 然后通过解码器重建特征
        reconstructed_feature = self.concept_decoder(projected_concept)
        reconstruction_loss = F.mse_loss(reconstructed_feature, feature_tensor)
        
        # 总损失
        total_loss = similarity_loss + 0.1 * reconstruction_loss
        
        # 反向传播和优化
        self.neural_optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.feature_embedder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.concept_projector.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.concept_decoder.parameters(), max_norm=1.0)
        
        self.neural_optimizer.step()
        self.scheduler.step()
        
        # 更新概念嵌入（可选：调整概念嵌入使其更接近特征）
        # 这里我们使用指数移动平均来平滑更新
        with torch.no_grad():
            # 计算新的概念嵌入（特征嵌入和原概念嵌入的加权平均）
            alpha = 0.1  # 更新强度
            new_embedding = (1 - alpha) * target_tensor + alpha * feature_embedding.mean(dim=0, keepdim=True)
            
            # 归一化
            new_embedding_norm = torch.norm(new_embedding, dim=1, keepdim=True)
            new_embedding = new_embedding / new_embedding_norm
            
            # 更新概念库
            self.concept_library["embeddings"][concept_id] = new_embedding.cpu().numpy().flatten()
        
        # 记录接地示例
        self.concept_library["grounding_examples"][concept_id].append({
            "feature": feature_tensor.cpu().numpy().tolist() if feature_tensor.is_cuda else feature_tensor.numpy().tolist(),
            "timestamp": time.time(),
            "loss": total_loss.item()
        })
        
        # 更新概念统计
        self.concept_library["concepts"][concept_id]["grounding_examples"] += 1
        
        # 限制示例数量
        max_examples = 100
        if len(self.concept_library["grounding_examples"][concept_id]) > max_examples:
            self.concept_library["grounding_examples"][concept_id] = \
                self.concept_library["grounding_examples"][concept_id][-max_examples:]
        
        elapsed_time = time.time() - start_time
        learning_result = {
            "success": True,
            "concept_id": concept_id,
            "concept_name": self.concept_library["concepts"][concept_id]["name"],
            "losses": {
                "similarity_loss": float(similarity_loss.item()),
                "reconstruction_loss": float(reconstruction_loss.item()),
                "total_loss": float(total_loss.item())
            },
            "embedding_updated": True,
            "examples_recorded": len(self.concept_library["grounding_examples"][concept_id]),
            "performance": {
                "learning_time": elapsed_time,
                "current_lr": self.neural_optimizer.param_groups[0]['lr'] if self.neural_optimizer.param_groups else self.learning_rate
            }
        }
        
        logger.info(f"从接地示例中学习完成: 概念={target_concept}, 总损失={total_loss.item():.4f}, 示例数={learning_result['examples_recorded']}")
        
        return learning_result
    
    def validate_grounding_consistency(self,
                                     concept_id: str,
                                     n_examples: int = 10) -> Dict[str, Any]:
        """
        验证接地一致性
        
        Args:
            concept_id: 概念ID
            n_examples: 使用的示例数量
            
        Returns:
            一致性验证结果
        """
        start_time = time.time()
        
        if concept_id not in self.concept_library["concepts"]:
            return {
                "success": False,
                "error": f"概念不存在: {concept_id}"
            }
        
        # 获取概念信息
        concept_info = self.concept_library["concepts"][concept_id]
        concept_embedding = self.concept_library["embeddings"][concept_id]
        
        # 获取接地示例
        examples = self.concept_library["grounding_examples"][concept_id]
        
        if not examples:
            return {
                "success": False,
                "error": f"概念 {concept_id} 没有接地示例",
                "concept_id": concept_id,
                "concept_name": concept_info["name"]
            }
        
        # 随机选择示例（如果示例数量足够）
        if len(examples) > n_examples:
            import random
            selected_examples = random.sample(examples, n_examples)
        else:
            selected_examples = examples
        
        # 计算一致性指标
        consistency_scores = []
        
        for example in selected_examples:
            feature_array = np.array(example["feature"])
            if len(feature_array.shape) == 1:
                feature_array = feature_array.reshape(1, -1)
            
            # 接地到概念
            feature_tensor = torch.tensor(feature_array, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                feature_embedding = self.feature_embedder(feature_tensor)
                
                # 转换为张量
                concept_tensor = torch.tensor(concept_embedding, dtype=torch.float32).to(self.device)
                if len(concept_tensor.shape) == 1:
                    concept_tensor = concept_tensor.unsqueeze(0)
                
                # 计算相似度
                similarity = F.cosine_similarity(feature_embedding, concept_tensor, dim=1).mean().item()
                consistency_scores.append(similarity)
        
        # 计算统计信息
        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
            std_consistency = np.std(consistency_scores)
            min_consistency = np.min(consistency_scores)
            max_consistency = np.max(consistency_scores)
            
            # 判断一致性是否足够
            is_consistent = avg_consistency >= self.similarity_threshold
        else:
            avg_consistency = std_consistency = min_consistency = max_consistency = 0.0
            is_consistent = False
        
        elapsed_time = time.time() - start_time
        validation_result = {
            "success": True,
            "concept_id": concept_id,
            "concept_name": concept_info["name"],
            "consistency_metrics": {
                "average_similarity": float(avg_consistency),
                "std_similarity": float(std_consistency),
                "min_similarity": float(min_consistency),
                "max_similarity": float(max_consistency),
                "is_consistent": is_consistent,
                "threshold": self.similarity_threshold
            },
            "examples_analyzed": len(selected_examples),
            "total_examples": len(examples),
            "performance": {
                "validation_time": elapsed_time
            }
        }
        
        self.performance_stats["validation_checks"] += 1
        logger.info(f"接地一致性验证完成: 概念={concept_info['name']}, 平均相似度={avg_consistency:.3f}, 一致性={is_consistent}")
        
        return validation_result
    
    def discover_new_concepts(self,
                            features: List[Union[torch.Tensor, np.ndarray]],
                            n_clusters: int = 5,
                            min_samples: int = 10) -> Dict[str, Any]:
        """
        从特征中发现新概念（聚类分析）
        
        Args:
            features: 特征列表
            n_clusters: 聚类数量
            min_samples: 最小样本数
            
        Returns:
            概念发现结果
        """
        start_time = time.time()
        
        if len(features) < min_samples:
            return {
                "success": False,
                "error": f"样本数量不足: {len(features)} < {min_samples}",
                "samples_available": len(features),
                "min_samples_required": min_samples
            }
        
        # 转换和嵌入特征
        feature_embeddings = []
        
        for feature in features:
            if isinstance(feature, np.ndarray):
                feature_tensor = torch.tensor(feature, dtype=torch.float32).to(self.device)
            else:
                feature_tensor = feature.to(self.device)
            
            if len(feature_tensor.shape) == 1:
                feature_tensor = feature_tensor.unsqueeze(0)
            
            with torch.no_grad():
                embedding = self.feature_embedder(feature_tensor)
                feature_embeddings.append(embedding.cpu().numpy().flatten())
        
        # 转换为numpy数组
        feature_matrix = np.array(feature_embeddings)
        
        # 降维（如果维度太高）
        if feature_matrix.shape[1] > 50:
            pca = PCA(n_components=50)
            feature_matrix_reduced = pca.fit_transform(feature_matrix)
        else:
            feature_matrix_reduced = feature_matrix
        
        # 聚类
        kmeans = KMeans(n_clusters=min(n_clusters, len(features)), random_state=42)
        cluster_labels = kmeans.fit_predict(feature_matrix_reduced)
        cluster_centers = kmeans.cluster_centers_
        
        # 分析每个聚类
        discovered_concepts = []
        
        for cluster_id in range(kmeans.n_clusters):
            # 获取聚类中的样本
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_size = len(cluster_indices)
            
            if cluster_size < 2:  # 太小，忽略
                continue
            
            # 计算聚类质量
            cluster_features = feature_matrix[cluster_indices]
            cluster_center = cluster_centers[cluster_id]
            
            # 计算类内距离（紧凑性）
            distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
            compactness = np.mean(distances)
            
            # 计算分离度（与其他聚类的距离）
            other_centers = [cluster_centers[i] for i in range(kmeans.n_clusters) if i != cluster_id]
            if other_centers:
                separations = [np.linalg.norm(cluster_center - other_center) for other_center in other_centers]
                separation = np.min(separations) if separations else 0.0
            else:
                separation = 0.0
            
            # 聚类质量分数
            quality_score = separation / (compactness + 1e-8)
            
            # 生成概念名称
            concept_name = f"discovered_cluster_{cluster_id}"
            
            # 创建概念嵌入（聚类中心）
            # 需要将降维后的中心映射回原始空间
            if feature_matrix.shape[1] > 50:
                # 使用PCA逆变换近似原始空间
                concept_embedding_approx = pca.inverse_transform(cluster_center)
            else:
                concept_embedding_approx = cluster_center
            
            # 归一化
            embedding_norm = np.linalg.norm(concept_embedding_approx)
            if embedding_norm > 0:
                concept_embedding = concept_embedding_approx / embedding_norm
            else:
                concept_embedding = concept_embedding_approx
            
            # 创建新概念
            concept_id = self.add_concept(
                name=concept_name,
                concept_type=ConceptType.BASIC_LEVEL,
                grounding_type=GroundingType.PERCEPTUAL,
                initial_embedding=concept_embedding
            )
            
            discovered_concepts.append({
                "concept_id": concept_id,
                "concept_name": concept_name,
                "cluster_id": cluster_id,
                "cluster_size": cluster_size,
                "compactness": float(compactness),
                "separation": float(separation),
                "quality_score": float(quality_score),
                "embedding_shape": concept_embedding.shape
            })
        
        elapsed_time = time.time() - start_time
        discovery_result = {
            "success": True,
            "discovered_concepts": discovered_concepts,
            "clustering_info": {
                "n_clusters_requested": n_clusters,
                "n_clusters_actual": kmeans.n_clusters,
                "n_features": len(features),
                "feature_dimension": feature_matrix.shape[1],
                "reduced_dimension": feature_matrix_reduced.shape[1] if feature_matrix.shape[1] > 50 else feature_matrix.shape[1]
            },
            "performance": {
                "discovery_time": elapsed_time
            }
        }
        
        logger.info(f"概念发现完成: 发现{len(discovered_concepts)}个新概念")
        
        return discovery_result
    
    def get_concept_hierarchy(self, 
                             root_concept: Optional[str] = None,
                             max_depth: int = 3) -> Dict[str, Any]:
        """
        获取概念层次结构
        
        Args:
            root_concept: 根概念（如果为None则返回整个层次结构）
            max_depth: 最大深度
            
        Returns:
            层次结构信息
        """
        hierarchy = self.concept_library["hierarchy"]
        
        if root_concept:
            root_id = self._find_concept_id(root_concept)
            if not root_id:
                return {
                    "success": False,
                    "error": f"根概念不存在: {root_concept}"
                }
            
            # 获取以root_id为根的子树
            if nx.has_path(hierarchy, root_id, root_id):  # 检查是否存在自环
                # 使用BFS获取子树
                subtree_nodes = set()
                queue = deque([(root_id, 0)])  # (node, depth)
                
                while queue:
                    node, depth = queue.popleft()
                    if depth > max_depth:
                        continue
                    
                    subtree_nodes.add(node)
                    
                    # 添加子节点
                    for child in hierarchy.successors(node):
                        queue.append((child, depth + 1))
                
                # 创建子图
                subgraph = hierarchy.subgraph(subtree_nodes)
            else:
                subgraph = nx.DiGraph()
        else:
            subgraph = hierarchy
        
        # 转换为可序列化的格式
        nodes = []
        for node in subgraph.nodes():
            concept_info = self.concept_library["concepts"].get(node, {})
            nodes.append({
                "id": node,
                "name": concept_info.get("name", "unknown"),
                "type": concept_info.get("type", "unknown"),
                "grounding_type": concept_info.get("grounding_type", "unknown")
            })
        
        edges = []
        for u, v, data in subgraph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "relation_type": data.get("relation_type", "unknown"),
                "confidence": data.get("confidence", 1.0)
            })
        
        return {
            "success": True,
            "hierarchy": {
                "nodes": nodes,
                "edges": edges,
                "is_dag": nx.is_directed_acyclic_graph(subgraph),
                "node_count": len(nodes),
                "edge_count": len(edges)
            },
            "root_concept": root_concept,
            "max_depth": max_depth
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = self.performance_stats.copy()
        stats["concept_library_stats"] = {
            "total_concepts": len(self.concept_library["concepts"]),
            "concept_types": defaultdict(int),
            "grounding_types": defaultdict(int),
            "hierarchy_nodes": self.concept_library["hierarchy"].number_of_nodes(),
            "hierarchy_edges": self.concept_library["hierarchy"].number_of_edges(),
            "total_grounding_examples": sum(len(examples) for examples in self.concept_library["grounding_examples"].values())
        }
        
        # 统计概念类型和接地类型
        for concept_info in self.concept_library["concepts"].values():
            stats["concept_library_stats"]["concept_types"][concept_info["type"]] += 1
            stats["concept_library_stats"]["grounding_types"][concept_info["grounding_type"]] += 1
        
        stats["model_info"] = {
            "feature_dim": self.feature_dim,
            "embedding_dim": self.embedding_dim,
            "concept_dim": self.concept_dim,
            "similarity_threshold": self.similarity_threshold,
            "device": str(self.device),
            "training_mode": self.training
        }
        
        return stats
    
    def train_mode(self, enabled: bool = True):
        """设置训练模式"""
        self.training = enabled
        if enabled:
            self.feature_embedder.train()
            self.concept_projector.train()
            self.concept_decoder.train()
        else:
            self.feature_embedder.eval()
            self.concept_projector.eval()
            self.concept_decoder.eval()
        logger.info(f"训练模式: {'启用' if enabled else '禁用'}")
    
    def save_model(self, filepath: str) -> bool:
        """保存模型到文件"""
        try:
            import pickle
            model_data = {
                "feature_embedder_state": self.feature_embedder.state_dict(),
                "concept_projector_state": self.concept_projector.state_dict(),
                "concept_decoder_state": self.concept_decoder.state_dict(),
                "concept_library": self.concept_library,
                "concept_counter": self.concept_counter,
                "performance_stats": self.performance_stats
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"神经概念接地器保存到: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存神经概念接地器失败: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """从文件加载模型"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # 加载状态字典
            self.feature_embedder.load_state_dict(model_data["feature_embedder_state"])
            self.concept_projector.load_state_dict(model_data["concept_projector_state"])
            self.concept_decoder.load_state_dict(model_data["concept_decoder_state"])
            
            # 加载概念系统
            self.concept_library = model_data.get("concept_library", self.concept_library)
            self.concept_counter = model_data.get("concept_counter", self.concept_counter)
            self.performance_stats = model_data.get("performance_stats", self.performance_stats.copy())
            
            logger.info(f"神经概念接地器从 {filepath} 加载")
            return True
        except Exception as e:
            logger.error(f"加载神经概念接地器失败: {e}")
            return False


# 示例和测试函数
def create_example_grounder() -> NeuralConceptGrounder:
    """创建示例概念接地器"""
    grounder = NeuralConceptGrounder(
        feature_dim=256,
        embedding_dim=128,
        concept_dim=64,
        learning_rate=0.001,
        similarity_threshold=0.6
    )
    return grounder


def test_concept_grounder():
    """测试神经概念接地器"""
    logger.info("开始测试神经概念接地器")
    
    # 创建示例接地器
    grounder = create_example_grounder()
    
    # 创建示例特征
    batch_size = 5
    feature_dim = 256
    example_features = _deterministic_randn((batch_size, feature_dim), seed_prefix="randn_default").to(grounder.device)
    
    # 测试特征接地
    logger.info("测试特征接地...")
    grounding_result = grounder.ground_feature_to_concept(example_features, k=3)
    logger.info(f"接地结果: {len(grounding_result['top_concepts'])}个概念匹配")
    for i, concept in enumerate(grounding_result['top_concepts'][:3]):
        logger.info(f"  概念{i+1}: {concept['concept_name']}, 相似度: {concept['similarity']:.3f}")
    
    # 测试多模态接地
    logger.info("测试多模态接地...")
    multimodal_features = {
        "visual": _deterministic_randn((batch_size, feature_dim), seed_prefix="randn_default").to(grounder.device),
        "audio": _deterministic_randn((batch_size, feature_dim // 2), seed_prefix="randn_default").to(grounder.device),
        "text": _deterministic_randn((batch_size, feature_dim // 4), seed_prefix="randn_default").to(grounder.device)
    }
    multimodal_result = grounder.ground_multimodal_features(multimodal_features)
    logger.info(f"多模态接地: {len(multimodal_result['modalities_processed'])}个模态")
    
    # 测试从示例中学习
    logger.info("测试从示例中学习...")
    target_concept = "unusual_pattern"
    learning_result = grounder.learn_from_grounding_example(example_features[0], target_concept)
    logger.info(f"学习结果: 概念={learning_result['concept_name']}, 总损失={learning_result['losses']['total_loss']:.4f}")
    
    # 测试接地一致性验证
    logger.info("测试接地一致性验证...")
    if learning_result['success']:
        concept_id = learning_result['concept_id']
        validation_result = grounder.validate_grounding_consistency(concept_id)
        logger.info(f"一致性验证: 平均相似度={validation_result['consistency_metrics']['average_similarity']:.3f}")
    
    # 测试概念发现
    logger.info("测试概念发现...")
    # 创建一些随机特征用于发现
    discovery_features = [_deterministic_randn((feature_dim,), seed_prefix="randn_default").to(grounder.device) for _ in range(20)]
    discovery_result = grounder.discover_new_concepts(discovery_features, n_clusters=3)
    logger.info(f"概念发现: {len(discovery_result.get('discovered_concepts', []))}个新概念")
    
    # 测试层次结构获取
    logger.info("测试概念层次结构...")
    hierarchy_result = grounder.get_concept_hierarchy("object", max_depth=2)
    if hierarchy_result['success']:
        logger.info(f"层次结构: {hierarchy_result['hierarchy']['node_count']}节点, {hierarchy_result['hierarchy']['edge_count']}边")
    
    # 显示性能统计
    stats = grounder.get_performance_stats()
    logger.info(f"性能统计: {stats}")
    
    logger.info("神经概念接地器测试完成")
    return grounder


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_grounder = test_concept_grounder()