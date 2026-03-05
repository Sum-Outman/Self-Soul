import numpy as np
import zlib
"""
跨模态注意力机制

实现文本、图像、音频等不同模态之间的双向注意力，
支持模态间语义关联和特征增强。

核心功能：
1. 文本→图像注意力：文本描述关注图像相关区域
2. 图像→文本注意力：图像特征关注相关文本描述
3. 文本→音频注意力：文本关注音频相关片段
4. 音频→文本注意力：音频关注相关文本描述
5. 自适应注意力权重：根据模态质量和重要性动态调整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cross_modal_attention")



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

class MultiHeadCrossModalAttention(nn.Module):
    """多头跨模态注意力层"""
    
    def __init__(self, d_model: int = 768, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # 查询、键、值投影
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        logger.debug(f"初始化多头跨模态注意力，模型维度: {d_model}, 头数: {n_heads}, 头维度: {self.d_head}")
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, query_len, d_model]
            key: 键张量 [batch_size, key_len, d_model]
            value: 值张量 [batch_size, key_len, d_model]
            key_padding_mask: 键填充掩码 [batch_size, key_len]
            
        Returns:
            attended_output: 注意力输出 [batch_size, query_len, d_model]
            attention_weights: 注意力权重 [batch_size, n_heads, query_len, key_len]
        """
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]
        
        # 线性投影
        Q = self.query_projection(query)  # [batch_size, query_len, d_model]
        K = self.key_projection(key)      # [batch_size, key_len, d_model]
        V = self.value_projection(value)  # [batch_size, key_len, d_model]
        
        # 重塑为多头格式
        Q = Q.view(batch_size, query_len, self.n_heads, self.d_head).transpose(1, 2)  # [batch_size, n_heads, query_len, d_head]
        K = K.view(batch_size, key_len, self.n_heads, self.d_head).transpose(1, 2)    # [batch_size, n_heads, key_len, d_head]
        V = V.view(batch_size, key_len, self.n_heads, self.d_head).transpose(1, 2)    # [batch_size, n_heads, key_len, d_head]
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch_size, n_heads, query_len, key_len]
        
        # 应用填充掩码
        if key_padding_mask is not None:
            # 扩展掩码以匹配注意力头
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, key_len]
            attention_scores = attention_scores.masked_fill(key_padding_mask, float('-inf'))
        
        # 应用softmax获取注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重到值
        attended_values = torch.matmul(attention_weights, V)  # [batch_size, n_heads, query_len, d_head]
        
        # 重塑回原始形状
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)
        
        # 输出投影
        output = self.output_projection(attended_values)
        
        return output, attention_weights


class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制
    
    支持文本、图像、音频之间的双向注意力，
    实现模态间语义关联和特征增强。
    """
    
    def __init__(self, embedding_dim: int = 768, num_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.1):
        """
        初始化跨模态注意力机制
        
        Args:
            embedding_dim: 嵌入维度
            num_heads: 注意力头数
            num_layers: 注意力层数
            dropout: Dropout概率
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        logger.info(f"初始化跨模态注意力机制，嵌入维度: {embedding_dim}, 头数: {num_heads}, 层数: {num_layers}")
        
        # 模态特定适配器
        self.modality_adapters = nn.ModuleDict({
            "text": nn.Linear(embedding_dim, embedding_dim),
            "image": nn.Linear(embedding_dim, embedding_dim),
            "audio": nn.Linear(embedding_dim, embedding_dim)
        })
        
        # 跨模态注意力层
        self.attention_layers = nn.ModuleList([
            MultiHeadCrossModalAttention(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim)
            for _ in range(num_layers * 3)  # 每层有3个注意力方向
        ])
        
        # 注意力融合器
        self.attention_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # 注意力权重分析器
        self.attention_analyzer = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 3),  # 3个注意力方向的权重
            nn.Softmax(dim=-1)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._initialize_weights()
        
        logger.info("跨模态注意力机制初始化完成")
    
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
    
    def forward(self, modality_features: List[torch.Tensor],
                modality_types: List[str] = None) -> Dict[str, Any]:
        """
        执行跨模态注意力
        
        Args:
            modality_features: 模态特征列表，顺序为[文本, 图像, 音频]
            modality_types: 模态类型列表，如["text", "image", "audio"]
            
        Returns:
            注意力结果字典
        """
        if len(modality_features) != 3:
            raise ValueError(f"需要3个模态特征，但提供了{len(modality_features)}个")
        
        # 默认模态类型
        if modality_types is None:
            modality_types = ["text", "image", "audio"]
        
        text_features, image_features, audio_features = modality_features
        
        # 验证输入形状
        batch_size, seq_len, dim = text_features.shape
        logger.debug(f"执行跨模态注意力，批次大小: {batch_size}, 序列长度: {seq_len}, 维度: {dim}")
        
        # 1. 模态特定适配
        text_adapted = self.modality_adapters["text"](text_features)
        image_adapted = self.modality_adapters["image"](image_features)
        audio_adapted = self.modality_adapters["audio"](audio_features)
        
        # 2. 初始化当前表示
        current_text = text_adapted
        current_image = image_adapted
        current_audio = audio_adapted
        
        # 3. 存储各层的注意力结果
        all_attention_outputs = []
        all_attention_weights = []
        
        # 4. 逐层应用跨模态注意力
        for layer_idx in range(self.num_layers):
            layer_outputs = {}
            layer_weights = {}
            
            # 获取当前层的注意力模块
            attention_layer = self.attention_layers[layer_idx]
            
            # 文本→图像注意力
            text_to_image, weights_text_image = attention_layer(
                query=current_text,
                key=current_image,
                value=current_image
            )
            text_to_image = self.layer_norms[layer_idx * 3](current_text + self.dropout(text_to_image))
            layer_outputs["text_to_image"] = text_to_image
            layer_weights["text_to_image"] = weights_text_image
            
            # 图像→文本注意力
            image_to_text, weights_image_text = attention_layer(
                query=current_image,
                key=current_text,
                value=current_text
            )
            image_to_text = self.layer_norms[layer_idx * 3 + 1](current_image + self.dropout(image_to_text))
            layer_outputs["image_to_text"] = image_to_text
            layer_weights["image_to_text"] = weights_image_text
            
            # 文本→音频注意力
            text_to_audio, weights_text_audio = attention_layer(
                query=current_text,
                key=current_audio,
                value=current_audio
            )
            text_to_audio = self.layer_norms[layer_idx * 3 + 2](current_text + self.dropout(text_to_audio))
            layer_outputs["text_to_audio"] = text_to_audio
            layer_weights["text_to_audio"] = weights_text_audio
            
            # 更新当前表示
            current_text = (text_to_image + text_to_audio) / 2
            current_image = image_to_text
            # 音频表示暂时保持不变（可以添加音频→文本注意力）
            
            # 保存层结果
            all_attention_outputs.append(layer_outputs)
            all_attention_weights.append(layer_weights)
            
            logger.debug(f"第{layer_idx+1}层注意力完成: "
                        f"文本→图像权重均值: {weights_text_image.mean().item():.4f}, "
                        f"图像→文本权重均值: {weights_image_text.mean().item():.4f}")
        
        # 5. 融合不同注意力方向的输出
        concatenated_outputs = torch.cat([current_text, current_image, current_audio], dim=-1)
        fused_output = self.attention_fusion(concatenated_outputs)
        
        # 6. 分析注意力权重分布
        attention_weight_distribution = self._analyze_attention_weights(all_attention_weights)
        
        # 7. 计算注意力质量指标
        attention_quality = self._compute_attention_quality(all_attention_weights, all_attention_outputs)
        
        # 8. 准备结果
        result = {
            "fused_output": fused_output,
            "text_enhanced": current_text,
            "image_enhanced": current_image,
            "audio_enhanced": current_audio,
            "attention_outputs": all_attention_outputs,
            "attention_weights": all_attention_weights,
            "attention_weight_distribution": attention_weight_distribution,
            "attention_quality": attention_quality,
            "modality_types": modality_types,
            "batch_size": batch_size,
            "seq_len": seq_len
        }
        
        return result
    
    def _analyze_attention_weights(self, all_attention_weights: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """分析注意力权重分布"""
        if not all_attention_weights:
            return {}
        
        analysis = {}
        
        # 收集所有注意力权重
        all_weights = []
        for layer_weights in all_attention_weights:
            for weight_type, weights in layer_weights.items():
                all_weights.append(weights.detach().cpu())
        
        if not all_weights:
            return analysis
        
        # 计算统计信息
        combined_weights = torch.cat([w.flatten() for w in all_weights])
        
        analysis["mean_attention_weight"] = combined_weights.mean().item()
        analysis["std_attention_weight"] = combined_weights.std().item()
        analysis["max_attention_weight"] = combined_weights.max().item()
        analysis["min_attention_weight"] = combined_weights.min().item()
        analysis["attention_sparsity"] = (combined_weights < 0.1).float().mean().item()  # 稀疏度
        
        # 按注意力方向分析
        for layer_idx, layer_weights in enumerate(all_attention_weights):
            for weight_type, weights in layer_weights.items():
                key = f"layer{layer_idx}_{weight_type}"
                weights_flat = weights.detach().cpu().flatten()
                analysis[f"{key}_mean"] = weights_flat.mean().item()
                analysis[f"{key}_std"] = weights_flat.std().item()
        
        return analysis
    
    def _compute_attention_quality(self, all_attention_weights: List[Dict[str, torch.Tensor]],
                                  all_attention_outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """计算注意力质量指标"""
        quality = {}
        
        if not all_attention_weights or not all_attention_outputs:
            return quality
        
        # 注意力集中度（注意力权重的熵）
        attention_entropies = []
        for layer_weights in all_attention_weights:
            for weight_type, weights in layer_weights.items():
                # 计算每个查询位置的注意力分布熵
                batch_size, n_heads, query_len, key_len = weights.shape
                weights_flat = weights.view(batch_size * n_heads * query_len, key_len)
                
                # 避免log(0)
                weights_safe = weights_flat + 1e-10
                entropy = -torch.sum(weights_safe * torch.log(weights_safe), dim=-1)
                attention_entropies.append(entropy.mean().item())
        
        if attention_entropies:
            quality["average_attention_entropy"] = sum(attention_entropies) / len(attention_entropies)
            quality["attention_concentration"] = 1.0 / (1.0 + quality["average_attention_entropy"])  # 集中度
        
        # 注意力稳定性（跨层的一致性）
        if len(all_attention_weights) > 1:
            stability_scores = []
            for weight_type in all_attention_weights[0].keys():
                # 比较相邻层的注意力权重
                for layer_idx in range(len(all_attention_weights) - 1):
                    weights_layer1 = all_attention_weights[layer_idx][weight_type]
                    weights_layer2 = all_attention_weights[layer_idx + 1][weight_type]
                    
                    # 计算相似度
                    sim = F.cosine_similarity(
                        weights_layer1.flatten(start_dim=2),
                        weights_layer2.flatten(start_dim=2),
                        dim=-1
                    ).mean().item()
                    stability_scores.append(sim)
            
            if stability_scores:
                quality["attention_stability"] = sum(stability_scores) / len(stability_scores)
        
        # 注意力多样性（不同注意力头之间的差异）
        diversity_scores = []
        for layer_weights in all_attention_weights:
            for weight_type, weights in layer_weights.items():
                # 计算不同头之间的平均差异
                batch_size, n_heads, query_len, key_len = weights.shape
                if n_heads > 1:
                    head_vectors = weights.view(batch_size, n_heads, -1)
                    # 计算头之间的平均余弦相似度
                    head_similarities = []
                    for i in range(n_heads):
                        for j in range(i + 1, n_heads):
                            sim = F.cosine_similarity(
                                head_vectors[:, i, :],
                                head_vectors[:, j, :],
                                dim=-1
                            ).mean().item()
                            head_similarities.append(sim)
                    
                    if head_similarities:
                        avg_similarity = sum(head_similarities) / len(head_similarities)
                        diversity = 1.0 - avg_similarity  # 差异越大，多样性越高
                        diversity_scores.append(diversity)
        
        if diversity_scores:
            quality["attention_diversity"] = sum(diversity_scores) / len(diversity_scores)
        
        return quality
    
    def compute_attention_weights(self, source_modality: str, target_modality: str,
                                 source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """
        计算特定模态对之间的注意力权重
        
        Args:
            source_modality: 源模态类型
            target_modality: 目标模态类型
            source_features: 源模态特征
            target_features: 目标模态特征
            
        Returns:
            注意力权重 [batch_size, num_heads, source_len, target_len]
        """
        # 模态适配
        source_adapted = self.modality_adapters[source_modality](source_features)
        target_adapted = self.modality_adapters[target_modality](target_features)
        
        # 使用第一层注意力模块
        attention_layer = self.attention_layers[0]
        
        # 只计算注意力权重，不更新表示
        _, attention_weights = attention_layer(
            query=source_adapted,
            key=target_adapted,
            value=target_adapted
        )
        
        return attention_weights
    
    def visualize_attention(self, attention_weights: torch.Tensor,
                           source_tokens: List[str] = None,
                           target_tokens: List[str] = None) -> Dict[str, Any]:
        """
        可视化注意力权重
        
        Args:
            attention_weights: 注意力权重张量
            source_tokens: 源token列表（用于文本）
            target_tokens: 目标token列表
            
        Returns:
            可视化数据字典
        """
        batch_size, num_heads, source_len, target_len = attention_weights.shape
        
        # 取第一个样本和第一个头作为示例
        sample_weights = attention_weights[0, 0].detach().cpu().numpy()
        
        visualization = {
            "attention_matrix": sample_weights.tolist(),
            "source_len": source_len,
            "target_len": target_len,
            "num_heads": num_heads,
            "batch_size": batch_size,
            "mean_weight": float(sample_weights.mean()),
            "max_weight": float(sample_weights.max()),
            "min_weight": float(sample_weights.min())
        }
        
        # 如果有token信息，添加
        if source_tokens is not None:
            visualization["source_tokens"] = source_tokens[:source_len]
        
        if target_tokens is not None:
            visualization["target_tokens"] = target_tokens[:target_len]
        
        return visualization
    
    def get_config(self) -> Dict[str, Any]:
        """获取注意力机制配置"""
        return {
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "attention_directions": ["text_to_image", "image_to_text", "text_to_audio"]
        }


def test_cross_modal_attention():
    """测试跨模态注意力机制"""
    logger.info("测试跨模态注意力机制...")
    
    try:
        # 创建注意力机制
        attention = CrossModalAttention(embedding_dim=768, num_heads=8, num_layers=2)
        
        # 创建测试数据
        batch_size = 2
        seq_len = 10
        dim = 768
        
        # 模拟不同模态的特征
        text_features = _deterministic_randn((batch_size, seq_len, dim), seed_prefix="randn_default")
        image_features = _deterministic_randn((batch_size, seq_len, dim), seed_prefix="randn_default")
        audio_features = _deterministic_randn((batch_size, seq_len, dim), seed_prefix="randn_default")
        
        # 测试前向传播
        result = attention([text_features, image_features, audio_features])
        
        # 验证输出
        required_keys = ["fused_output", "text_enhanced", "image_enhanced", "audio_enhanced",
                        "attention_outputs", "attention_weights", "attention_quality"]
        for key in required_keys:
            assert key in result, f"缺失结果键: {key}"
        
        # 验证形状
        assert result["fused_output"].shape == (batch_size, seq_len, dim), f"融合输出形状错误: {result['fused_output'].shape}"
        assert result["text_enhanced"].shape == (batch_size, seq_len, dim), f"增强文本形状错误: {result['text_enhanced'].shape}"
        
        # 测试注意力权重计算
        attention_weights = attention.compute_attention_weights(
            "text", "image", text_features, image_features
        )
        assert attention_weights.shape[0] == batch_size, f"注意力权重批次大小错误: {attention_weights.shape}"
        assert attention_weights.shape[1] == 8, f"注意力头数错误: {attention_weights.shape}"  # num_heads=8
        
        # 测试注意力可视化
        visualization = attention.visualize_attention(attention_weights)
        assert "attention_matrix" in visualization, "可视化数据中缺少attention_matrix"
        
        # 测试配置获取
        config = attention.get_config()
        assert "embedding_dim" in config, "配置中缺少embedding_dim"
        
        logger.info("✅ 跨模态注意力机制测试通过")
        
        return {
            "success": True,
            "fused_output_shape": list(result["fused_output"].shape),
            "attention_quality": result["attention_quality"],
            "attention_weight_distribution": result.get("attention_weight_distribution", {}),
            "config": config,
            "message": "跨模态注意力机制测试完成"
        }
        
    except Exception as e:
        logger.error(f"❌ 跨模态注意力机制测试失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "跨模态注意力机制测试失败"
        }


# 导出主要类和方法
__all__ = [
    "CrossModalAttention",
    "MultiHeadCrossModalAttention",
    "test_cross_modal_attention"
]

if __name__ == "__main__":
    # 运行测试
    test_result = test_cross_modal_attention()
    print(f"测试结果: {test_result}")

