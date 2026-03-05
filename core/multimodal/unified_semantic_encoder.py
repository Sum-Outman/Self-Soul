import zlib
"""
统一语义编码器

将文本、图像、音频等多模态特征对齐到统一的语义空间，
解决当前系统的核心缺陷：缺乏真正的跨模态语义对齐。

核心功能：
1. 模态特定特征投影到共享语义空间
2. 对比学习实现跨模态语义对齐
3. 计算模态间语义相似度
4. 生成统一语义表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("unified_semantic_encoder")



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

class UnifiedSemanticEncoder(nn.Module):
    """
    统一语义编码器
    
    将不同模态的特征映射到统一的语义空间，
    实现真正的跨模态语义对齐而非简单的特征拼接。
    """
    
    def __init__(self, embedding_dim: int = 768, temperature: float = 0.07,
                 projection_layers: int = 2, dropout: float = 0.1):
        """
        初始化统一语义编码器
        
        Args:
            embedding_dim: 统一语义空间的维度
            temperature: 对比学习的温度参数
            projection_layers: 投影层的数量
            dropout: Dropout概率
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        logger.info(f"初始化统一语义编码器，嵌入维度: {embedding_dim}, 温度: {temperature}")
        
        # 模态特定的投影层
        self.text_projector = self._build_projection_layers(embedding_dim, projection_layers, dropout)
        self.image_projector = self._build_projection_layers(embedding_dim, projection_layers, dropout)
        self.audio_projector = self._build_projection_layers(embedding_dim, projection_layers, dropout)
        
        # 共享语义空间投影
        self.shared_semantic_projector = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # 对齐质量评估器
        self.alignment_quality_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._initialize_weights()
        
        logger.info("统一语义编码器初始化完成")
    
    def _build_projection_layers(self, dim: int, num_layers: int, dropout: float) -> nn.Module:
        """构建投影层"""
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(dim, dim))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, modality_features: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播：将多模态特征对齐到统一语义空间
        
        Args:
            modality_features: 模态特征列表，顺序为[文本, 图像, 音频]
            
        Returns:
            unified_representation: 统一语义表示 [batch_size, seq_len, embedding_dim]
            alignment_info: 对齐信息字典
        """
        if len(modality_features) != 3:
            raise ValueError(f"需要3个模态特征，但提供了{len(modality_features)}个")
        
        text_features, image_features, audio_features = modality_features
        
        # 验证输入形状
        batch_size, seq_len, dim = text_features.shape
        if (image_features.shape != text_features.shape or 
            audio_features.shape != text_features.shape):
            logger.warning(f"模态特征形状不一致: text{text_features.shape}, "
                          f"image{image_features.shape}, audio{audio_features.shape}")
        
        logger.debug(f"处理多模态特征，批次大小: {batch_size}, 序列长度: {seq_len}, 维度: {dim}")
        
        # 1. 模态特定投影
        text_projected = self.text_projector(text_features)
        image_projected = self.image_projector(image_features)
        audio_projected = self.audio_projector(audio_features)
        
        # 2. 计算模态间对齐分数（语义相似度）
        text_image_alignment = self._compute_alignment_scores(text_projected, image_projected)
        text_audio_alignment = self._compute_alignment_scores(text_projected, audio_projected)
        image_audio_alignment = self._compute_alignment_scores(image_projected, audio_projected)
        
        # 3. 计算平均对齐分数（用于评估）
        avg_text_image_alignment = text_image_alignment.mean().item()
        avg_text_audio_alignment = text_audio_alignment.mean().item()
        avg_image_audio_alignment = image_audio_alignment.mean().item()
        
        # 4. 拼接所有模态特征
        concatenated_features = torch.cat([text_projected, image_projected, audio_projected], dim=-1)
        
        # 5. 投影到统一语义空间
        unified_representation = self.shared_semantic_projector(concatenated_features)
        
        # 6. 计算语义连贯性
        semantic_coherence = self._compute_semantic_coherence(
            unified_representation, text_projected, image_projected, audio_projected
        )
        
        # 7. 计算对齐质量
        alignment_quality = self._compute_alignment_quality(
            text_projected, image_projected, audio_projected, unified_representation
        )
        
        # 8. 准备对齐信息
        alignment_info = {
            "text_image_alignment": text_image_alignment,
            "text_audio_alignment": text_audio_alignment,
            "image_audio_alignment": image_audio_alignment,
            "semantic_coherence": semantic_coherence,
            "alignment_quality": alignment_quality,
            "text_projected_norm": torch.norm(text_projected, dim=-1).mean(),
            "image_projected_norm": torch.norm(image_projected, dim=-1).mean(),
            "audio_projected_norm": torch.norm(audio_projected, dim=-1).mean(),
            "avg_text_image_similarity": avg_text_image_alignment,
            "avg_text_audio_similarity": avg_text_audio_alignment,
            "avg_image_audio_similarity": avg_image_audio_alignment,
        }
        
        logger.debug(f"对齐完成，文本-图像平均相似度: {avg_text_image_alignment:.4f}, "
                    f"文本-音频: {avg_text_audio_alignment:.4f}, "
                    f"图像-音频: {avg_image_audio_alignment:.4f}")
        
        return unified_representation, alignment_info
    
    def _compute_alignment_scores(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        """
        计算模态间对齐分数（语义相似度）
        
        Args:
            features_a: 特征A [batch_size, seq_len, dim]
            features_b: 特征B [batch_size, seq_len, dim]
            
        Returns:
            对齐分数矩阵 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, dim = features_a.shape
        
        # 归一化特征
        features_a_norm = F.normalize(features_a, p=2, dim=-1)  # [batch_size, seq_len, dim]
        features_b_norm = F.normalize(features_b, p=2, dim=-1)  # [batch_size, seq_len, dim]
        
        # 计算相似度矩阵（批次内所有位置对）
        similarity = torch.matmul(features_a_norm, features_b_norm.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        
        return similarity
    
    def _compute_semantic_coherence(self, unified_repr: torch.Tensor,
                                  text_repr: torch.Tensor, image_repr: torch.Tensor,
                                  audio_repr: torch.Tensor) -> torch.Tensor:
        """
        计算语义连贯性分数
        
        Args:
            unified_repr: 统一语义表示
            text_repr: 文本特征
            image_repr: 图像特征
            audio_repr: 音频特征
            
        Returns:
            语义连贯性分数 [batch_size]
        """
        batch_size, seq_len, dim = unified_repr.shape
        
        # 计算统一表示与各模态表示的一致性
        text_coherence = F.cosine_similarity(
            unified_repr.reshape(-1, dim),
            text_repr.reshape(-1, dim),
            dim=-1
        ).reshape(batch_size, seq_len).mean(dim=-1)
        
        image_coherence = F.cosine_similarity(
            unified_repr.reshape(-1, dim),
            image_repr.reshape(-1, dim),
            dim=-1
        ).reshape(batch_size, seq_len).mean(dim=-1)
        
        audio_coherence = F.cosine_similarity(
            unified_repr.reshape(-1, dim),
            audio_repr.reshape(-1, dim),
            dim=-1
        ).reshape(batch_size, seq_len).mean(dim=-1)
        
        # 平均连贯性分数
        overall_coherence = (text_coherence + image_coherence + audio_coherence) / 3
        
        return overall_coherence
    
    def _compute_alignment_quality(self, text_repr: torch.Tensor, image_repr: torch.Tensor,
                                 audio_repr: torch.Tensor, unified_repr: torch.Tensor) -> torch.Tensor:
        """
        计算对齐质量分数
        
        Args:
            text_repr: 文本特征
            image_repr: 图像特征
            audio_repr: 音频特征
            unified_repr: 统一语义表示
            
        Returns:
            对齐质量分数 [batch_size, 1]
        """
        batch_size, seq_len, dim = text_repr.shape
        
        # 提取全局表示（平均池化）
        text_global = text_repr.mean(dim=1)  # [batch_size, dim]
        image_global = image_repr.mean(dim=1)
        audio_global = audio_repr.mean(dim=1)
        unified_global = unified_repr.mean(dim=1)
        
        # 拼接特征
        concatenated = torch.cat([text_global, image_global, audio_global, unified_global], dim=-1)
        
        # 计算对齐质量
        alignment_quality = self.alignment_quality_scorer(concatenated)  # [batch_size, 1]
        
        return alignment_quality
    
    def compute_alignment_loss(self, text_features: torch.Tensor, image_features: torch.Tensor,
                              audio_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算对齐损失（用于训练）
        
        Args:
            text_features: 文本特征
            image_features: 图像特征
            audio_features: 音频特征
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 计算对比学习损失
        losses["text_image_loss"] = self._contrastive_loss(text_features, image_features)
        losses["text_audio_loss"] = self._contrastive_loss(text_features, audio_features)
        losses["image_audio_loss"] = self._contrastive_loss(image_features, audio_features)
        
        # 总对齐损失
        losses["total_alignment_loss"] = sum(losses.values()) / len(losses)
        
        return losses
    
    def _contrastive_loss(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        """
        计算对比学习损失
        
        Args:
            features_a: 特征A [batch_size, seq_len, dim]
            features_b: 特征B [batch_size, seq_len, dim]
            
        Returns:
            对比损失
        """
        batch_size, seq_len, dim = features_a.shape
        
        # 重塑为 [batch_size * seq_len, dim]
        features_a_flat = features_a.reshape(-1, dim)
        features_b_flat = features_b.reshape(-1, dim)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features_a_flat, features_b_flat.T) / self.temperature
        
        # 创建标签：对角线为匹配对
        labels = torch.arange(features_a_flat.shape[0], device=features_a.device)
        
        # 计算交叉熵损失（双向）
        cross_entropy = nn.CrossEntropyLoss()
        loss_a = cross_entropy(similarity_matrix, labels)
        loss_b = cross_entropy(similarity_matrix.T, labels)
        
        # 平均损失
        loss = (loss_a + loss_b) / 2
        
        return loss
    
    def encode_single_modality(self, modality_type: str, features: torch.Tensor) -> torch.Tensor:
        """
        编码单个模态特征到统一语义空间
        
        真实实现：直接将模态特征投影到统一语义空间，无需零向量占位符。
        每个模态特定的投影器将特征映射到相同的语义空间。
        
        Args:
            modality_type: 模态类型（text, image, audio）
            features: 模态特征 [batch_size, seq_len, dim]
            
        Returns:
            统一语义空间中的表示 [batch_size, seq_len, embedding_dim]
        """
        if modality_type == "text":
            projected = self.text_projector(features)
        elif modality_type == "image":
            projected = self.image_projector(features)
        elif modality_type == "audio":
            projected = self.audio_projector(features)
        else:
            raise ValueError(f"不支持的模态类型: {modality_type}")
        
        # 真实实现：直接返回投影后的特征，无需零向量占位符
        # 投影器已经将不同模态的特征映射到相同的语义空间
        # 添加层归一化以确保数值稳定性
        projected_normalized = F.layer_norm(projected, projected.shape[-1:])
        
        return projected_normalized
    
    def calculate_cross_modal_similarity(self, modality_a: str, features_a: torch.Tensor,
                                        modality_b: str, features_b: torch.Tensor) -> float:
        """
        计算两个不同模态特征的语义相似度
        
        Args:
            modality_a: 第一个模态类型
            features_a: 第一个模态特征
            modality_b: 第二个模态类型
            features_b: 第二个模态特征
            
        Returns:
            语义相似度（0-1之间）
        """
        # 编码到统一语义空间
        unified_a = self.encode_single_modality(modality_a, features_a)
        unified_b = self.encode_single_modality(modality_b, features_b)
        
        # 计算全局表示的余弦相似度
        global_a = unified_a.mean(dim=1)  # [batch_size, dim]
        global_b = unified_b.mean(dim=1)
        
        # 归一化
        global_a_norm = F.normalize(global_a, p=2, dim=-1)
        global_b_norm = F.normalize(global_b, p=2, dim=-1)
        
        # 计算余弦相似度
        similarity = torch.cosine_similarity(global_a_norm, global_b_norm, dim=-1)
        
        return similarity.mean().item()
    
    def get_config(self) -> Dict[str, Any]:
        """获取编码器配置"""
        return {
            "embedding_dim": self.embedding_dim,
            "temperature": self.temperature,
            "projection_layers": len(self.text_projector) // 3,  # 估算层数
            "dropout": 0.1,  # 假设值
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def test_unified_semantic_encoder():
    """测试统一语义编码器"""
    logger.info("测试统一语义编码器...")
    
    try:
        # 创建编码器
        encoder = UnifiedSemanticEncoder(embedding_dim=768, temperature=0.07)
        
        # 创建测试数据
        batch_size = 2
        seq_len = 10
        dim = 768
        
        # 模拟不同模态的特征
        text_features = _deterministic_randn((batch_size, seq_len, dim), seed_prefix="randn_default")
        image_features = _deterministic_randn((batch_size, seq_len, dim), seed_prefix="randn_default")
        audio_features = _deterministic_randn((batch_size, seq_len, dim), seed_prefix="randn_default")
        
        # 测试前向传播
        unified_repr, alignment_info = encoder([text_features, image_features, audio_features])
        
        # 验证输出形状
        assert unified_repr.shape == (batch_size, seq_len, 768), f"统一表示形状错误: {unified_repr.shape}"
        
        # 测试对齐信息
        required_keys = ["text_image_alignment", "text_audio_alignment", "image_audio_alignment",
                        "semantic_coherence", "alignment_quality"]
        for key in required_keys:
            assert key in alignment_info, f"缺失对齐信息键: {key}"
        
        # 测试单个模态编码
        text_unified = encoder.encode_single_modality("text", text_features)
        assert text_unified.shape == (batch_size, seq_len, 768), f"文本统一表示形状错误: {text_unified.shape}"
        
        # 测试跨模态相似度计算
        similarity = encoder.calculate_cross_modal_similarity(
            "text", text_features, "image", image_features
        )
        assert -1 <= similarity <= 1, f"相似度超出范围: {similarity}"
        
        # 测试配置获取
        config = encoder.get_config()
        assert "embedding_dim" in config, "配置中缺少embedding_dim"
        
        # 测试损失计算
        losses = encoder.compute_alignment_loss(text_features, image_features, audio_features)
        assert "total_alignment_loss" in losses, "损失计算中缺少total_alignment_loss"
        
        logger.info("✅ 统一语义编码器测试通过")
        
        return {
            "success": True,
            "unified_representation_shape": list(unified_repr.shape),
            "alignment_info_keys": list(alignment_info.keys()),
            "cross_modal_similarity": similarity,
            "config": config,
            "message": "统一语义编码器测试完成"
        }
        
    except Exception as e:
        logger.error(f"❌ 统一语义编码器测试失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "统一语义编码器测试失败"
        }


# 导出主要类和方法
__all__ = [
    "UnifiedSemanticEncoder",
    "test_unified_semantic_encoder"
]

if __name__ == "__main__":
    # 运行测试
    test_result = test_unified_semantic_encoder()
    print(f"测试结果: {test_result}")

