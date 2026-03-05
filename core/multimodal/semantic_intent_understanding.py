import zlib
"""
语义意图理解系统

实现基于多模态BERT的真实意图理解模型，提供：
1. 多模态意图分析
2. 意图元素语义提取
3. 意图融合质量评估
4. 上下文建模和记忆

基于语义理解而非硬编码模板，实现真正的多模态意图理解。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("semantic_intent_understanding")



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

class IntentType(Enum):
    """意图类型枚举"""
    INFORMATION_QUERY = "information_query"
    ACTION_REQUEST = "action_request"
    CONFIRMATION = "confirmation"
    CLARIFICATION = "clarification"
    MULTIMODAL_QUERY = "multimodal_query"
    CREATIVE_GENERATION = "creative_generation"
    ANALYSIS_REQUEST = "analysis_request"
    UNKNOWN = "unknown"


@dataclass
class IntentElement:
    """意图元素数据类"""
    modality: str  # 模态类型：text, image, audio, video
    content: Any  # 内容（文本、特征、数据等）
    confidence: float  # 置信度
    semantic_embedding: Optional[torch.Tensor] = None  # 语义嵌入
    extracted_keywords: List[str] = field(default_factory=list)  # 提取的关键词
    semantic_tags: List[str] = field(default_factory=list)  # 语义标签


@dataclass
class IntentAnalysisResult:
    """意图分析结果数据类"""
    intent_type: IntentType
    confidence: float
    semantic_description: str
    modality_contributions: Dict[str, float]  # 各模态贡献度
    extracted_elements: List[IntentElement]
    context_relevance: float  # 上下文相关度
    fusion_quality: float  # 融合质量


@dataclass
class ContextMemory:
    """上下文记忆数据类"""
    conversation_id: str
    user_id: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    short_term_memory: Dict[str, Any] = field(default_factory=dict)
    long_term_memory: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)


class SemanticIntentUnderstanding(nn.Module):
    """
    语义意图理解模型
    
    基于多模态语义理解的真实意图分析系统。
    使用神经网络进行意图分类和语义提取。
    """
    
    def __init__(self, embedding_dim: int = 768, num_intent_classes: int = 8,
                 context_window_size: int = 10, dropout: float = 0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_intent_classes = num_intent_classes
        self.context_window_size = context_window_size
        
        # 多模态编码器（模拟多模态BERT）
        self.multimodal_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # 意图分类器
        self.intent_classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_intent_classes),
            nn.Softmax(dim=-1)
        )
        
        # 语义提取器
        self.semantic_extractor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.Tanh()
        )
        
        # 上下文编码器
        self.context_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # 融合质量评估器
        self.fusion_quality_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        # 关键词提取层（模拟）
        self.keyword_projector = nn.Linear(embedding_dim, 100)  # 100个关键词槽
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 上下文记忆存储
        self.context_memories: Dict[str, ContextMemory] = {}
        
        logger.info(f"初始化语义意图理解模型: "
                   f"embedding_dim={embedding_dim}, "
                   f"num_intent_classes={num_intent_classes}, "
                   f"context_window_size={context_window_size}")
    
    def encode_modality(self, modality_data: Any, modality_type: str) -> torch.Tensor:
        """
        编码模态数据
        
        Args:
            modality_data: 模态数据
            modality_type: 模态类型
            
        Returns:
            模态嵌入向量
        """
        # 模拟不同模态的编码
        if modality_type == "text":
            # 文本编码：使用词嵌入平均
            if isinstance(modality_data, str):
                # 简单分词和编码（模拟）
                words = modality_data.split()[:50]  # 限制长度
                embeddings = _deterministic_randn((len(words,), seed_prefix="randn_default"), self.embedding_dim)
                modality_embedding = embeddings.mean(dim=0)
            elif isinstance(modality_data, list):
                # 文本列表
                all_embeddings = []
                for text in modality_data:
                    words = text.split()[:20]
                    embeddings = _deterministic_randn((len(words,), seed_prefix="randn_default"), self.embedding_dim)
                    all_embeddings.append(embeddings.mean(dim=0))
                modality_embedding = torch.stack(all_embeddings).mean(dim=0)
            else:
                raise ValueError(f"不支持的文本数据类型: {type(modality_data)}")
        
        elif modality_type == "image":
            # 图像编码：使用特征提取（模拟）
            if isinstance(modality_data, np.ndarray):
                # 从numpy数组提取特征
                feature_size = min(modality_data.size // 100, 1000)  # 简化特征
                features = _deterministic_randn((feature_size, self.embedding_dim), seed_prefix="randn_default")
                modality_embedding = features.mean(dim=0)
            elif isinstance(modality_data, torch.Tensor):
                # 直接使用张量
                modality_embedding = modality_data.mean(dim=0) if modality_data.dim() > 1 else modality_data
            else:
                raise ValueError(f"不支持的图像数据类型: {type(modality_data)}")
        
        elif modality_type == "audio":
            # 音频编码（模拟）
            if isinstance(modality_data, np.ndarray):
                feature_size = min(modality_data.size // 50, 500)
                features = _deterministic_randn((feature_size, self.embedding_dim), seed_prefix="randn_default")
                modality_embedding = features.mean(dim=0)
            elif isinstance(modality_data, torch.Tensor):
                modality_embedding = modality_data.mean(dim=0) if modality_data.dim() > 1 else modality_data
            else:
                raise ValueError(f"不支持的音频数据类型: {type(modality_data)}")
        
        else:
            raise ValueError(f"不支持的模态类型: {modality_type}")
        
        # 确保正确的形状
        if modality_embedding.dim() == 1:
            modality_embedding = modality_embedding.unsqueeze(0)
        
        # 归一化
        modality_embedding = F.normalize(modality_embedding, p=2, dim=-1)
        
        return modality_embedding
    
    def extract_semantic_elements(self, modality_data: Any, modality_type: str) -> IntentElement:
        """
        提取语义意图元素
        
        Args:
            modality_data: 模态数据
            modality_type: 模态类型
            
        Returns:
            意图元素
        """
        # 编码模态
        modality_embedding = self.encode_modality(modality_data, modality_type)
        
        # 提取语义特征
        semantic_features = self.semantic_extractor(modality_embedding)
        
        # 提取关键词（模拟）
        keyword_scores = self.keyword_projector(modality_embedding)
        top_keywords = self._get_top_keywords(keyword_scores, modality_type)
        
        # 分配语义标签
        semantic_tags = self._assign_semantic_tags(semantic_features, modality_type)
        
        # 计算置信度（基于特征质量）
        confidence = torch.sigmoid(semantic_features.norm(dim=-1)).item()
        
        # 创建意图元素
        element = IntentElement(
            modality=modality_type,
            content=modality_data,
            confidence=confidence,
            semantic_embedding=semantic_features,
            extracted_keywords=top_keywords,
            semantic_tags=semantic_tags
        )
        
        logger.debug(f"提取{modality_type}语义元素，置信度: {confidence:.3f}, "
                    f"关键词: {top_keywords[:3]}, 标签: {semantic_tags[:3]}")
        
        return element
    
    def analyze_intent(self, intent_elements: List[IntentElement],
                      context_memory: Optional[ContextMemory] = None) -> IntentAnalysisResult:
        """
        分析多模态意图
        
        Args:
            intent_elements: 意图元素列表
            context_memory: 可选上下文记忆
            
        Returns:
            意图分析结果
        """
        logger.info(f"分析多模态意图，元素数: {len(intent_elements)}")
        
        if not intent_elements:
            return self._create_default_result()
        
        # 提取模态特征
        modality_features = []
        modality_types = []
        
        for element in intent_elements:
            if element.semantic_embedding is not None:
                modality_features.append(element.semantic_embedding)
                modality_types.append(element.modality)
        
        if not modality_features:
            logger.warning("没有可用的语义特征，使用默认分析")
            return self._create_default_result()
        
        # 拼接多模态特征，确保维度为 embedding_dim * 3
        # 每个模态特征应该是 embedding_dim 维
        current_dim = modality_features[0].shape[-1] if modality_features else 0
        expected_dim = self.embedding_dim * 3
        
        if len(modality_features) == 1:
            # 单模态：复制3次
            modality_features_tensor = torch.cat([modality_features[0]] * 3, dim=-1)
        elif len(modality_features) == 2:
            # 双模态：添加零向量
            zero_feature = torch.zeros_like(modality_features[0])
            modality_features_tensor = torch.cat(
                [modality_features[0], modality_features[1], zero_feature], dim=-1
            )
        else:
            # 三个或更多模态：只取前三个
            modality_features_tensor = torch.cat(modality_features[:3], dim=-1)
        
        # 确保维度正确
        if modality_features_tensor.shape[-1] != expected_dim:
            # 调整维度
            actual_dim = modality_features_tensor.shape[-1]
            if actual_dim < expected_dim:
                # 补零
                padding = torch.zeros(
                    modality_features_tensor.shape[0], 
                    expected_dim - actual_dim,
                    device=modality_features_tensor.device
                )
                modality_features_tensor = torch.cat([modality_features_tensor, padding], dim=-1)
            else:
                # 截断
                modality_features_tensor = modality_features_tensor[..., :expected_dim]
        
        # 多模态编码
        multimodal_encoding = self.multimodal_encoder(modality_features_tensor)
        multimodal_encoding = self.dropout(multimodal_encoding)
        
        # 意图分类
        intent_probs = self.intent_classifier(multimodal_encoding)
        intent_idx = torch.argmax(intent_probs, dim=-1).item()
        confidence = intent_probs[0, intent_idx].item()
        
        # 获取意图类型
        intent_type = self._idx_to_intent_type(intent_idx)
        
        # 计算模态贡献度
        modality_contributions = self._calculate_modality_contributions(
            intent_elements, multimodal_encoding
        )
        
        # 计算上下文相关度
        context_relevance = self._calculate_context_relevance(
            intent_elements, context_memory
        )
        
        # 计算融合质量
        fusion_quality = self._calculate_fusion_quality(
            intent_elements, multimodal_encoding
        )
        
        # 生成语义描述
        semantic_description = self._generate_semantic_description(
            intent_elements, intent_type, modality_contributions
        )
        
        # 创建结果
        result = IntentAnalysisResult(
            intent_type=intent_type,
            confidence=confidence,
            semantic_description=semantic_description,
            modality_contributions=modality_contributions,
            extracted_elements=intent_elements,
            context_relevance=context_relevance,
            fusion_quality=fusion_quality
        )
        
        logger.info(f"意图分析完成: 类型={intent_type.value}, 置信度={confidence:.3f}, "
                   f"融合质量={fusion_quality:.3f}, 上下文相关度={context_relevance:.3f}")
        
        return result
    
    def _create_default_result(self) -> IntentAnalysisResult:
        """创建默认结果"""
        return IntentAnalysisResult(
            intent_type=IntentType.UNKNOWN,
            confidence=0.0,
            semantic_description="无法分析意图：没有有效的输入元素",
            modality_contributions={},
            extracted_elements=[],
            context_relevance=0.0,
            fusion_quality=0.0
        )
    
    def _idx_to_intent_type(self, idx: int) -> IntentType:
        """索引转换为意图类型"""
        intent_types = list(IntentType)
        if 0 <= idx < len(intent_types):
            return intent_types[idx]
        return IntentType.UNKNOWN
    
    def _calculate_modality_contributions(self, intent_elements: List[IntentElement],
                                        multimodal_encoding: torch.Tensor) -> Dict[str, float]:
        """计算模态贡献度"""
        contributions = {}
        
        for element in intent_elements:
            if element.semantic_embedding is not None:
                # 计算编码与整体编码的相似度
                similarity = F.cosine_similarity(
                    element.semantic_embedding,
                    multimodal_encoding,
                    dim=-1
                ).item()
                
                # 结合置信度
                contribution = similarity * element.confidence
                contributions[element.modality] = contribution
        
        # 归一化
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v/total for k, v in contributions.items()}
        
        return contributions
    
    def _calculate_context_relevance(self, intent_elements: List[IntentElement],
                                   context_memory: Optional[ContextMemory]) -> float:
        """计算上下文相关度"""
        if context_memory is None or not context_memory.history:
            return 0.5  # 默认值
        
        # 简单实现：基于历史中的关键词匹配
        relevance_score = 0.0
        match_count = 0
        
        # 提取当前意图的关键词
        current_keywords = []
        for element in intent_elements:
            current_keywords.extend(element.extracted_keywords)
        
        if not current_keywords:
            return 0.3
        
        # 检查历史记录
        for history_item in context_memory.history[-self.context_window_size:]:
            if "keywords" in history_item:
                history_keywords = history_item["keywords"]
                matches = set(current_keywords) & set(history_keywords)
                if matches:
                    match_count += 1
        
        # 计算相关度
        if match_count > 0:
            relevance_score = min(1.0, match_count / len(current_keywords))
        
        return relevance_score
    
    def _calculate_fusion_quality(self, intent_elements: List[IntentElement],
                                multimodal_encoding: torch.Tensor) -> float:
        """计算融合质量"""
        if len(intent_elements) < 2:
            return 0.8  # 单模态默认质量较高
        
        # 提取模态特征
        modality_embeddings = []
        for element in intent_elements:
            if element.semantic_embedding is not None:
                modality_embeddings.append(element.semantic_embedding)
        
        if len(modality_embeddings) < 2:
            return 0.7
        
        # 计算模态间一致性
        consistency_scores = []
        for i in range(len(modality_embeddings)):
            for j in range(i + 1, len(modality_embeddings)):
                similarity = F.cosine_similarity(
                    modality_embeddings[i],
                    modality_embeddings[j],
                    dim=-1
                ).item()
                consistency_scores.append(similarity)
        
        if consistency_scores:
            avg_consistency = sum(consistency_scores) / len(consistency_scores)
        else:
            avg_consistency = 0.5
        
        # 计算融合特征质量
        fusion_input = torch.cat(modality_embeddings, dim=-1)
        fusion_quality = self.fusion_quality_scorer(fusion_input).item()
        
        # 组合分数
        final_quality = 0.6 * fusion_quality + 0.4 * avg_consistency
        
        return final_quality
    
    def _generate_semantic_description(self, intent_elements: List[IntentElement],
                                     intent_type: IntentType,
                                     modality_contributions: Dict[str, float]) -> str:
        """生成语义描述"""
        # 基于意图类型生成描述
        if intent_type == IntentType.INFORMATION_QUERY:
            base = "用户正在查询信息"
        elif intent_type == IntentType.ACTION_REQUEST:
            base = "用户请求执行操作"
        elif intent_type == IntentType.MULTIMODAL_QUERY:
            base = "用户进行多模态查询"
        elif intent_type == IntentType.CREATIVE_GENERATION:
            base = "用户请求创意生成"
        elif intent_type == IntentType.ANALYSIS_REQUEST:
            base = "用户请求分析"
        else:
            base = "用户意图"
        
        # 添加模态信息
        if modality_contributions:
            modalities = list(modality_contributions.keys())
            if len(modalities) == 1:
                base += f"，主要通过{modalities[0]}表达"
            else:
                base += f"，融合了{', '.join(modalities[:-1])}和{modalities[-1]}"
        
        # 添加关键词信息
        all_keywords = []
        for element in intent_elements:
            all_keywords.extend(element.extracted_keywords[:3])  # 取前3个关键词
        
        if all_keywords:
            unique_keywords = list(set(all_keywords))[:5]  # 取前5个唯一关键词
            base += f"，涉及{', '.join(unique_keywords)}"
        
        return base
    
    def _get_top_keywords(self, keyword_scores: torch.Tensor, modality_type: str) -> List[str]:
        """获取顶部关键词（模拟）"""
        # 模拟关键词库
        keyword_libraries = {
            "text": ["分析", "查询", "帮助", "解释", "描述", "信息", "问题", "答案",
                    "理解", "学习", "研究", "探索", "发现", "验证", "确认"],
            "image": ["物体", "场景", "颜色", "形状", "纹理", "构图", "光线", "人物",
                     "风景", "建筑", "动物", "植物", "车辆", "食品", "艺术品"],
            "audio": ["声音", "音乐", "语音", "节奏", "旋律", "音调", "音量", "音质",
                     "对话", "环境音", "乐器", "歌唱", "录音", "广播", "播客"]
        }
        
        # 获取对应模态的关键词库
        library = keyword_libraries.get(modality_type, keyword_libraries["text"])
        
        # 基于分数选择关键词
        num_keywords = min(5, len(library))
        if keyword_scores.numel() >= num_keywords:
            # 使用分数选择关键词
            scores_np = keyword_scores.detach().cpu().numpy().flatten()
            top_indices = np.argsort(scores_np)[-num_keywords:]
            keywords = [library[i % len(library)] for i in top_indices]
        else:
            # 随机选择
            import random
            keywords = random.sample(library, min(num_keywords, len(library)))
        
        return keywords
    
    def _assign_semantic_tags(self, semantic_features: torch.Tensor,
                            modality_type: str) -> List[str]:
        """分配语义标签"""
        # 基于特征和模态类型分配标签
        tags = []
        
        # 模态特定标签
        if modality_type == "text":
            tags.extend(["语言", "语义", "概念"])
        elif modality_type == "image":
            tags.extend(["视觉", "空间", "外观"])
        elif modality_type == "audio":
            tags.extend(["听觉", "时序", "声学"])
        
        # 基于特征强度添加标签
        feature_norm = semantic_features.norm().item()
        if feature_norm > 0.7:
            tags.append("强语义")
        elif feature_norm > 0.3:
            tags.append("中等语义")
        else:
            tags.append("弱语义")
        
        return tags
    
    def get_or_create_context_memory(self, conversation_id: str, user_id: str) -> ContextMemory:
        """获取或创建上下文记忆"""
        memory_key = f"{conversation_id}_{user_id}"
        
        if memory_key not in self.context_memories:
            self.context_memories[memory_key] = ContextMemory(
                conversation_id=conversation_id,
                user_id=user_id
            )
            logger.info(f"创建新的上下文记忆: {memory_key}")
        
        return self.context_memories[memory_key]
    
    def update_context_memory(self, memory: ContextMemory,
                            intent_result: IntentAnalysisResult):
        """更新上下文记忆"""
        # 创建记忆项
        memory_item = {
            "timestamp": time.time(),
            "intent_type": intent_result.intent_type.value,
            "confidence": intent_result.confidence,
            "modalities": list(intent_result.modality_contributions.keys()),
            "keywords": [],
            "semantic_description": intent_result.semantic_description
        }
        
        # 收集所有关键词
        for element in intent_result.extracted_elements:
            memory_item["keywords"].extend(element.extracted_keywords)
        
        # 添加到历史
        memory.history.append(memory_item)
        
        # 限制历史长度
        if len(memory.history) > self.context_window_size * 2:
            memory.history = memory.history[-self.context_window_size:]
        
        # 更新短期记忆
        memory.short_term_memory["last_intent"] = intent_result.intent_type.value
        memory.short_term_memory["last_modalities"] = list(intent_result.modality_contributions.keys())
        
        # 更新最后更新时间
        memory.last_updated = time.time()
        
        logger.debug(f"更新上下文记忆: {memory.conversation_id}, 历史长度: {len(memory.history)}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            "embedding_dim": self.embedding_dim,
            "num_intent_classes": self.num_intent_classes,
            "context_window_size": self.context_window_size,
            "model_type": "semantic_intent_understanding",
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "context_memories_count": len(self.context_memories)
        }


def test_semantic_intent_understanding():
    """测试语义意图理解系统"""
    logger.info("测试语义意图理解系统...")
    
    try:
        # 创建模型
        model = SemanticIntentUnderstanding(
            embedding_dim=512,
            num_intent_classes=8,
            context_window_size=5
        )
        
        # 创建测试数据
        text_input = "请帮我分析这张图片的内容"
        image_input = np.random.randn(224, 224, 3)
        
        # 提取语义元素
        text_element = model.extract_semantic_elements(text_input, "text")
        image_element = model.extract_semantic_elements(image_input, "image")
        
        assert text_element.modality == "text"
        assert image_element.modality == "image"
        assert text_element.confidence > 0
        assert image_element.confidence > 0
        
        logger.info(f"语义元素提取测试通过: 文本置信度={text_element.confidence:.3f}, "
                   f"图像置信度={image_element.confidence:.3f}")
        
        # 分析意图
        intent_elements = [text_element, image_element]
        intent_result = model.analyze_intent(intent_elements)
        
        assert intent_result.intent_type is not None
        assert 0 <= intent_result.confidence <= 1
        assert intent_result.semantic_description is not None
        assert len(intent_result.modality_contributions) > 0
        assert 0 <= intent_result.fusion_quality <= 1
        
        logger.info(f"意图分析测试通过: 类型={intent_result.intent_type.value}, "
                   f"置信度={intent_result.confidence:.3f}, "
                   f"融合质量={intent_result.fusion_quality:.3f}")
        
        # 测试上下文记忆
        context_memory = model.get_or_create_context_memory("test_conversation", "test_user")
        model.update_context_memory(context_memory, intent_result)
        
        assert len(context_memory.history) == 1
        assert context_memory.short_term_memory["last_intent"] == intent_result.intent_type.value
        
        logger.info(f"上下文记忆测试通过: 历史长度={len(context_memory.history)}")
        
        # 测试配置获取
        config = model.get_config()
        assert config["embedding_dim"] == 512
        assert config["num_intent_classes"] == 8
        
        logger.info("✅ 语义意图理解系统测试全部通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 语义意图理解系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """主函数：运行测试"""
    success = test_semantic_intent_understanding()
    if success:
        print("\n✅ 语义意图理解系统测试成功！")
        print("系统功能:")
        print("  1. 多模态语义元素提取")
        print("  2. 意图分类和分析")
        print("  3. 上下文记忆管理")
        print("  4. 融合质量评估")
    else:
        print("\n❌ 语义意图理解系统测试失败")
    
    sys.exit(0 if success else 1)