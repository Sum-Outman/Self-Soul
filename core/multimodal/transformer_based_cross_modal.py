import zlib
"""
基于Transformer的跨模态注意力系统

实现标准Transformer架构的跨模态注意力机制，支持：
1. 模态间双向注意力
2. 多头注意力机制
3. 位置编码
4. 层归一化和残差连接
5. 注意力可视化

基于标准的Transformer编码器-解码器架构，提供真正的跨模态语义关联。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("transformer_cross_modal")


@dataclass

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

class AttentionWeights:
    """注意力权重数据类"""
    text_to_image: Optional[torch.Tensor] = None
    image_to_text: Optional[torch.Tensor] = None
    text_to_audio: Optional[torch.Tensor] = None
    audio_to_text: Optional[torch.Tensor] = None
    image_to_audio: Optional[torch.Tensor] = None
    audio_to_image: Optional[torch.Tensor] = None


@dataclass
class CrossModalOutput:
    """跨模态输出数据类"""
    fused_features: torch.Tensor
    attention_weights: AttentionWeights
    alignment_scores: Dict[str, float]
    semantic_coherence: float


class PositionalEncoding(nn.Module):
    """位置编码层"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # 创建位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            添加位置编码后的张量
        """
        return x + self.pe[:x.size(1)]


class TransformerCrossModalEncoder(nn.Module):
    """Transformer跨模态编码器"""
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, n_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 模态特定投影
        self.text_projection = nn.Linear(d_model, d_model)
        self.image_projection = nn.Linear(d_model, d_model)
        self.audio_projection = nn.Linear(d_model, d_model)
        
        # 跨模态注意力质量评估
        self.alignment_scorer = nn.Sequential(
            nn.Linear(d_model * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"初始化Transformer跨模态编码器: "
                   f"d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
    
    def encode_modality(self, features: torch.Tensor, modality_type: str) -> torch.Tensor:
        """
        编码特定模态特征
        
        Args:
            features: 模态特征 [batch_size, seq_len, d_model]
            modality_type: 模态类型 ('text', 'image', 'audio')
            
        Returns:
            编码后的模态特征
        """
        # 应用模态特定投影
        if modality_type == 'text':
            projected = self.text_projection(features)
        elif modality_type == 'image':
            projected = self.image_projection(features)
        elif modality_type == 'audio':
            projected = self.audio_projection(features)
        else:
            raise ValueError(f"不支持的模态类型: {modality_type}")
        
        # 添加位置编码
        projected = self.positional_encoding(projected)
        
        # Transformer编码
        encoded = self.transformer_encoder(projected)
        
        return encoded
    
    def cross_modal_attention(self, query: torch.Tensor, key: torch.Tensor,
                             value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        跨模态注意力计算
        
        Args:
            query: 查询张量 [batch_size, query_len, d_model]
            key: 键张量 [batch_size, key_len, d_model]
            value: 值张量 [batch_size, key_len, d_model]
            
        Returns:
            attended_output: 注意力输出
            attention_weights: 注意力权重
        """
        batch_size, query_len, d_model = query.shape
        key_len = key.shape[1]
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_model)
        
        # 应用softmax获取注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重到值
        attended_output = torch.matmul(attention_weights, value)
        
        return attended_output, attention_weights
    
    def forward(self, text_features: Optional[torch.Tensor] = None,
                image_features: Optional[torch.Tensor] = None,
                audio_features: Optional[torch.Tensor] = None) -> CrossModalOutput:
        """
        前向传播：处理多模态输入
        
        Args:
            text_features: 文本特征 [batch_size, seq_len, d_model]
            image_features: 图像特征 [batch_size, seq_len, d_model]
            audio_features: 音频特征 [batch_size, seq_len, d_model]
            
        Returns:
            跨模态输出
        """
        batch_size = 1
        if text_features is not None:
            batch_size = text_features.shape[0]
        elif image_features is not None:
            batch_size = image_features.shape[0]
        elif audio_features is not None:
            batch_size = audio_features.shape[0]
        
        # 编码各个模态
        encoded_features = {}
        
        if text_features is not None:
            encoded_text = self.encode_modality(text_features, 'text')
            encoded_features['text'] = encoded_text
            logger.debug(f"编码文本特征，形状: {encoded_text.shape}")
        
        if image_features is not None:
            encoded_image = self.encode_modality(image_features, 'image')
            encoded_features['image'] = encoded_image
            logger.debug(f"编码图像特征，形状: {encoded_image.shape}")
        
        if audio_features is not None:
            encoded_audio = self.encode_modality(audio_features, 'audio')
            encoded_features['audio'] = encoded_audio
            logger.debug(f"编码音频特征，形状: {encoded_audio.shape}")
        
        # 计算跨模态注意力
        attention_weights = AttentionWeights()
        alignment_scores = {}
        
        # 文本→图像注意力
        if 'text' in encoded_features and 'image' in encoded_features:
            text_to_image_output, text_to_image_weights = self.cross_modal_attention(
                encoded_features['text'], encoded_features['image'], encoded_features['image']
            )
            attention_weights.text_to_image = text_to_image_weights
            
            # 计算对齐分数
            alignment_scores['text_image'] = self._calculate_alignment_score(
                encoded_features['text'], encoded_features['image'], text_to_image_weights
            )
        
        # 图像→文本注意力
        if 'image' in encoded_features and 'text' in encoded_features:
            image_to_text_output, image_to_text_weights = self.cross_modal_attention(
                encoded_features['image'], encoded_features['text'], encoded_features['text']
            )
            attention_weights.image_to_text = image_to_text_weights
            
            # 计算对齐分数
            if 'text_image' not in alignment_scores:
                alignment_scores['text_image'] = self._calculate_alignment_score(
                    encoded_features['image'], encoded_features['text'], image_to_text_weights
                )
        
        # 文本→音频注意力
        if 'text' in encoded_features and 'audio' in encoded_features:
            text_to_audio_output, text_to_audio_weights = self.cross_modal_attention(
                encoded_features['text'], encoded_features['audio'], encoded_features['audio']
            )
            attention_weights.text_to_audio = text_to_audio_weights
            
            # 计算对齐分数
            alignment_scores['text_audio'] = self._calculate_alignment_score(
                encoded_features['text'], encoded_features['audio'], text_to_audio_weights
            )
        
        # 融合特征
        fused_features = self._fuse_features(encoded_features)
        
        # 计算语义一致性
        semantic_coherence = self._calculate_semantic_coherence(alignment_scores)
        
        output = CrossModalOutput(
            fused_features=fused_features,
            attention_weights=attention_weights,
            alignment_scores=alignment_scores,
            semantic_coherence=semantic_coherence
        )
        
        logger.info(f"跨模态编码完成，融合特征形状: {fused_features.shape}, "
                   f"语义一致性: {semantic_coherence:.3f}")
        
        return output
    
    def _calculate_alignment_score(self, query: torch.Tensor, key: torch.Tensor,
                                  attention_weights: torch.Tensor) -> float:
        """
        计算对齐分数
        
        Args:
            query: 查询特征
            key: 键特征
            attention_weights: 注意力权重
            
        Returns:
            对齐分数 (0-1)
        """
        # 计算注意力集中度（越高表示注意力越集中）
        attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-10), dim=-1)
        max_entropy = math.log(attention_weights.shape[-1])
        attention_concentration = 1.0 - (attention_entropy.mean() / max_entropy)
        
        # 计算特征相似度
        query_global = query.mean(dim=1)
        key_global = key.mean(dim=1)
        feature_similarity = F.cosine_similarity(query_global, key_global, dim=-1).mean().item()
        
        # 组合分数
        alignment_score = 0.7 * attention_concentration + 0.3 * feature_similarity
        
        return alignment_score
    
    def _fuse_features(self, encoded_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        融合多模态特征
        
        Args:
            encoded_features: 编码后的特征字典
            
        Returns:
            融合后的特征
        """
        if not encoded_features:
            raise ValueError("没有可融合的特征")
        
        # 获取全局表示
        global_features = []
        for modality, features in encoded_features.items():
            global_feature = features.mean(dim=1)  # [batch_size, d_model]
            global_features.append(global_feature)
        
        # 连接全局特征
        fused = torch.cat(global_features, dim=-1)  # [batch_size, d_model * n_modalities]
        
        # 如果需要，可以添加融合层
        if len(encoded_features) > 1:
            # 简单的平均融合
            fused = fused.view(fused.shape[0], len(encoded_features), -1)
            fused = fused.mean(dim=1)  # [batch_size, d_model]
        
        return fused
    
    def _calculate_semantic_coherence(self, alignment_scores: Dict[str, float]) -> float:
        """
        计算语义一致性
        
        Args:
            alignment_scores: 对齐分数字典
            
        Returns:
            语义一致性分数
        """
        if not alignment_scores:
            return 0.0
        
        scores = list(alignment_scores.values())
        return sum(scores) / len(scores)
    
    def visualize_attention(self, attention_weights: torch.Tensor, title: str = "Attention Weights"):
        """
        可视化注意力权重
        
        Args:
            attention_weights: 注意力权重张量
            title: 图表标题
        """
        if attention_weights is None:
            logger.warning("无法可视化：注意力权重为空")
            return
        
        # 转换为numpy
        weights_np = attention_weights.detach().cpu().numpy()
        
        # 如果是多头注意力，取平均
        if len(weights_np.shape) == 4:  # [batch_size, n_heads, query_len, key_len]
            weights_np = weights_np.mean(axis=1)  # 平均所有头
        
        # 取第一个batch
        if len(weights_np.shape) == 3:  # [batch_size, query_len, key_len]
            weights_np = weights_np[0]
        
        # 创建热图
        plt.figure(figsize=(10, 8))
        plt.imshow(weights_np, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(title)
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        
        # 保存图像
        import os
        os.makedirs("attention_visualizations", exist_ok=True)
        filename = f"attention_visualizations/{title.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"注意力可视化已保存到: {filename}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "model_type": "transformer_cross_modal_encoder",
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def test_transformer_cross_modal():
    """测试Transformer跨模态编码器"""
    logger.info("测试Transformer跨模态编码器...")
    
    try:
        # 创建编码器
        encoder = TransformerCrossModalEncoder(
            d_model=512,
            n_heads=8,
            n_layers=4,
            dropout=0.1
        )
        
        # 创建测试数据
        batch_size = 2
        seq_len = 10
        d_model = 512
        
        text_features = _deterministic_randn((batch_size, seq_len, d_model), seed_prefix="randn_default")
        image_features = _deterministic_randn((batch_size, seq_len, d_model), seed_prefix="randn_default")
        audio_features = _deterministic_randn((batch_size, seq_len, d_model), seed_prefix="randn_default")
        
        logger.info(f"测试数据形状: 文本={text_features.shape}, 图像={image_features.shape}, 音频={audio_features.shape}")
        
        # 测试前向传播
        output = encoder(
            text_features=text_features,
            image_features=image_features,
            audio_features=audio_features
        )
        
        # 验证输出
        assert output.fused_features.shape == (batch_size, d_model), \
            f"融合特征形状错误: {output.fused_features.shape}"
        
        assert output.semantic_coherence >= 0 and output.semantic_coherence <= 1, \
            f"语义一致性超出范围: {output.semantic_coherence}"
        
        # 验证注意力权重
        if output.attention_weights.text_to_image is not None:
            assert output.attention_weights.text_to_image.shape == (batch_size, seq_len, seq_len), \
                f"文本到图像注意力形状错误: {output.attention_weights.text_to_image.shape}"
        
        # 验证对齐分数
        for modality_pair, score in output.alignment_scores.items():
            assert 0 <= score <= 1, f"对齐分数超出范围 ({modality_pair}): {score}"
        
        # 测试配置获取
        config = encoder.get_config()
        assert config["d_model"] == 512
        assert config["n_heads"] == 8
        
        # 测试注意力可视化（可选）
        if output.attention_weights.text_to_image is not None:
            encoder.visualize_attention(
                output.attention_weights.text_to_image,
                title="Text to Image Attention"
            )
        
        logger.info("✅ Transformer跨模态编码器测试全部通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ Transformer跨模态编码器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """主函数：运行测试"""
    success = test_transformer_cross_modal()
    if success:
        print("\n✅ Transformer跨模态编码器测试成功！")
        print("模型配置:")
        encoder = TransformerCrossModalEncoder()
        config = encoder.get_config()
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        print("\n❌ Transformer跨模态编码器测试失败")
    
    sys.exit(0 if success else 1)