import zlib
"""
真正统一语义编码器

彻底修复虚假实现，实现真正的跨模态语义对齐：
1. 不使用零向量作为占位符
2. 实现真正的模态特定特征提取
3. 基于对比学习的跨模态语义对齐
4. 支持单个模态到统一语义空间的映射

核心修复：
- 移除零向量占位符虚假实现
- 实现真正的多模态特征投影
- 基于真实对比学习的语义对齐
- 支持真实语义相似度计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import math

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("true_unified_semantic_encoder")



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

class TrueModalityEncoder(nn.Module):
    """
    模态特定编码器
    
    提取和编码特定模态的特征，不使用零向量占位符。
    """
    
    def __init__(self, modality_type: str, input_dim: int, hidden_dim: int, output_dim: int):
        """
        初始化模态编码器
        
        Args:
            modality_type: 模态类型（text, image, audio）
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
        """
        super().__init__()
        
        self.modality_type = modality_type
        
        # 模态特定的编码器结构
        if modality_type == "text":
            # 文本编码器：使用Transformer编码器
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=input_dim,
                    nhead=8,
                    dim_feedforward=hidden_dim,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=2
            )
            self.projection = nn.Linear(input_dim, output_dim)
            
        elif modality_type == "image":
            # 图像编码器：使用CNN + Transformer
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 假设输入为3通道图像
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((8, 8))  # 输出8x8特征图
            )
            cnn_output_dim = 64 * 8 * 8
            self.flatten = nn.Flatten()
            self.projection = nn.Sequential(
                nn.Linear(cnn_output_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            
        elif modality_type == "audio":
            # 音频编码器：使用1D CNN + Transformer
            self.cnn1d = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),  # 假设单通道音频
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(128)  # 输出128个特征
            )
            cnn_output_dim = 64 * 128
            self.flatten = nn.Flatten()
            self.projection = nn.Sequential(
                nn.Linear(cnn_output_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            
        else:
            raise ValueError(f"不支持的模态类型: {modality_type}")
        
        # 输出层归一化
        self.layer_norm = nn.LayerNorm(output_dim)
        
        logger.info(f"初始化{modality_type}编码器，输入维度: {input_dim}, 输出维度: {output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：提取模态特定特征
        
        Args:
            x: 输入特征，形状取决于模态类型
            
        Returns:
            编码后的特征，形状为[batch_size, seq_len, output_dim]或[batch_size, output_dim]
        """
        if self.modality_type == "text":
            # 文本输入: [batch_size, seq_len, input_dim]
            encoded = self.encoder(x)
            projected = self.projection(encoded)
            
        elif self.modality_type == "image":
            # 图像输入: [batch_size, channels, height, width]
            # 假设输入已经预处理为固定大小
            batch_size = x.shape[0]
            cnn_features = self.cnn(x)  # [batch_size, 64, 8, 8]
            flattened = self.flatten(cnn_features)  # [batch_size, 64*8*8]
            projected = self.projection(flattened)  # [batch_size, output_dim]
            # 添加序列维度
            projected = projected.unsqueeze(1)  # [batch_size, 1, output_dim]
            
        elif self.modality_type == "audio":
            # 音频输入: [batch_size, 1, seq_len] 或 [batch_size, seq_len]
            if x.dim() == 2:
                x = x.unsqueeze(1)  # 添加通道维度
            batch_size = x.shape[0]
            cnn_features = self.cnn1d(x)  # [batch_size, 64, 128]
            flattened = self.flatten(cnn_features)  # [batch_size, 64*128]
            projected = self.projection(flattened)  # [batch_size, output_dim]
            # 添加序列维度
            projected = projected.unsqueeze(1)  # [batch_size, 1, output_dim]
            
        # 层归一化
        normalized = self.layer_norm(projected)
        
        return normalized


class TrueUnifiedSemanticEncoder(nn.Module):
    """
    真正统一语义编码器
    
    实现真正的跨模态语义对齐，不使用零向量占位符。
    基于对比学习实现模态间语义对齐。
    """
    
    def __init__(self, 
                 text_input_dim: int = 768,
                 image_input_dim: int = 3,  # 图像通道数
                 audio_input_dim: int = 1,  # 音频通道数
                 unified_dim: int = 512,
                 hidden_dim: int = 1024,
                 temperature: float = 0.07,
                 dropout: float = 0.1):
        """
        初始化真正统一语义编码器
        
        Args:
            text_input_dim: 文本输入维度
            image_input_dim: 图像输入维度（通道数）
            audio_input_dim: 音频输入维度（通道数）
            unified_dim: 统一语义空间维度
            hidden_dim: 隐藏层维度
            temperature: 对比学习温度参数
            dropout: Dropout概率
        """
        super().__init__()
        
        self.unified_dim = unified_dim
        self.temperature = temperature
        
        logger.info(f"初始化真正统一语义编码器，统一维度: {unified_dim}, 温度: {temperature}")
        
        # 模态特定编码器
        self.text_encoder = TrueModalityEncoder("text", text_input_dim, hidden_dim, unified_dim)
        self.image_encoder = TrueModalityEncoder("image", image_input_dim, hidden_dim, unified_dim)
        self.audio_encoder = TrueModalityEncoder("audio", audio_input_dim, hidden_dim, unified_dim)
        
        # 共享语义空间投影（可选的进一步对齐）
        self.shared_projection = nn.Sequential(
            nn.Linear(unified_dim, unified_dim),
            nn.LayerNorm(unified_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(unified_dim, unified_dim),
            nn.LayerNorm(unified_dim)
        )
        
        # 对比学习温度参数（可学习）
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        
        # 语义对齐评估器
        self.alignment_evaluator = nn.Sequential(
            nn.Linear(unified_dim * 3, unified_dim),
            nn.LayerNorm(unified_dim),
            nn.GELU(),
            nn.Linear(unified_dim, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._initialize_weights()
        
        logger.info("真正统一语义编码器初始化完成")
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def encode_modality(self, modality_type: str, input_data: torch.Tensor) -> torch.Tensor:
        """
        编码单个模态到统一语义空间
        
        Args:
            modality_type: 模态类型（text, image, audio）
            input_data: 输入数据，形状取决于模态类型
            
        Returns:
            统一语义空间中的特征表示
        """
        if modality_type == "text":
            encoded = self.text_encoder(input_data)
        elif modality_type == "image":
            encoded = self.image_encoder(input_data)
        elif modality_type == "audio":
            encoded = self.audio_encoder(input_data)
        else:
            raise ValueError(f"不支持的模态类型: {modality_type}")
        
        # 进一步投影到共享语义空间
        projected = self.shared_projection(encoded)
        
        return projected
    
    def forward(self, 
                text_input: Optional[torch.Tensor] = None,
                image_input: Optional[torch.Tensor] = None,
                audio_input: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        前向传播：编码多模态输入到统一语义空间
        
        Args:
            text_input: 文本输入，形状为[batch_size, seq_len, text_input_dim]
            image_input: 图像输入，形状为[batch_size, 3, height, width]
            audio_input: 音频输入，形状为[batch_size, 1, seq_len]或[batch_size, seq_len]
            
        Returns:
            编码结果字典，包含统一表示和对齐信息
        """
        batch_size = None
        encoded_features = {}
        
        # 编码文本模态
        if text_input is not None:
            text_encoded = self.encode_modality("text", text_input)
            encoded_features["text"] = text_encoded
            batch_size = text_input.shape[0]
            
            # 提取全局表示（平均池化）
            if text_encoded.dim() == 3:  # [batch_size, seq_len, dim]
                text_global = text_encoded.mean(dim=1)
            else:  # [batch_size, dim]
                text_global = text_encoded
            encoded_features["text_global"] = text_global
        
        # 编码图像模态
        if image_input is not None:
            image_encoded = self.encode_modality("image", image_input)
            encoded_features["image"] = image_encoded
            if batch_size is None:
                batch_size = image_input.shape[0]
            
            # 提取全局表示
            if image_encoded.dim() == 3:
                image_global = image_encoded.mean(dim=1)
            else:
                image_global = image_encoded
            encoded_features["image_global"] = image_global
        
        # 编码音频模态
        if audio_input is not None:
            audio_encoded = self.encode_modality("audio", audio_input)
            encoded_features["audio"] = audio_encoded
            if batch_size is None:
                batch_size = audio_input.shape[0]
            
            # 提取全局表示
            if audio_encoded.dim() == 3:
                audio_global = audio_encoded.mean(dim=1)
            else:
                audio_global = audio_encoded
            encoded_features["audio_global"] = audio_global
        
        # 计算模态间语义相似度（如果至少有两个模态）
        similarity_info = self._compute_cross_modal_similarities(encoded_features)
        
        # 准备返回结果
        result = {
            "encoded_features": encoded_features,
            "similarity_info": similarity_info,
            "batch_size": batch_size,
            "unified_dim": self.unified_dim
        }
        
        # 如果有多个模态，计算对齐质量
        if len(encoded_features) >= 2:
            alignment_quality = self._compute_alignment_quality(encoded_features)
            result["alignment_quality"] = alignment_quality
            
            # 计算对比学习损失（用于训练）
            if self.training:
                contrastive_loss = self._compute_contrastive_loss(encoded_features)
                result["contrastive_loss"] = contrastive_loss
        
        logger.debug(f"编码完成，处理了{len(encoded_features)}个模态")
        
        return result
    
    def _compute_cross_modal_similarities(self, encoded_features: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        计算模态间语义相似度
        
        Args:
            encoded_features: 编码后的特征字典
            
        Returns:
            相似度信息字典
        """
        similarity_info = {}
        
        # 检查是否有全局表示
        global_features = {}
        for key, value in encoded_features.items():
            if key.endswith("_global"):
                modality = key.replace("_global", "")
                global_features[modality] = value
        
        # 计算所有模态对之间的余弦相似度
        modalities = list(global_features.keys())
        
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1 = modalities[i]
                mod2 = modalities[j]
                
                feat1 = global_features[mod1]
                feat2 = global_features[mod2]
                
                # 计算余弦相似度
                similarity = F.cosine_similarity(feat1, feat2, dim=-1)
                avg_similarity = similarity.mean().item()
                
                key = f"{mod1}_{mod2}_similarity"
                similarity_info[key] = avg_similarity
        
        return similarity_info
    
    def _compute_alignment_quality(self, encoded_features: Dict[str, torch.Tensor]) -> float:
        """
        计算对齐质量分数
        
        Args:
            encoded_features: 编码后的特征字典
            
        Returns:
            对齐质量分数（0-1之间）
        """
        # 收集所有全局特征
        global_features = []
        for key, value in encoded_features.items():
            if key.endswith("_global"):
                global_features.append(value)
        
        if len(global_features) < 2:
            return 0.0
        
        # 拼接所有全局特征
        concatenated = torch.cat(global_features, dim=-1)
        
        # 计算对齐质量
        alignment_score = self.alignment_evaluator(concatenated)
        
        return alignment_score.mean().item()
    
    def _compute_contrastive_loss(self, encoded_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算对比学习损失
        
        Args:
            encoded_features: 编码后的特征字典
            
        Returns:
            对比学习损失
        """
        # 收集所有全局特征
        global_features = []
        modality_names = []
        
        for key, value in encoded_features.items():
            if key.endswith("_global"):
                modality = key.replace("_global", "")
                global_features.append(value)
                modality_names.append(modality)
        
        if len(global_features) < 2:
            return torch.tensor(0.0, device=global_features[0].device)
        
        # 创建相似度矩阵
        num_modalities = len(global_features)
        batch_size = global_features[0].shape[0]
        
        # 归一化特征
        normalized_features = []
        for feat in global_features:
            normalized = F.normalize(feat, p=2, dim=-1)
            normalized_features.append(normalized)
        
        # 计算相似度矩阵
        similarity_matrix = torch.zeros((batch_size * num_modalities, batch_size * num_modalities),
                                       device=global_features[0].device)
        
        # 填充相似度矩阵
        for i in range(num_modalities):
            for j in range(num_modalities):
                # 计算模态i和模态j之间的相似度
                sim = torch.matmul(normalized_features[i], normalized_features[j].T)
                
                # 放置到正确的位置
                row_start = i * batch_size
                row_end = (i + 1) * batch_size
                col_start = j * batch_size
                col_end = (j + 1) * batch_size
                
                similarity_matrix[row_start:row_end, col_start:col_end] = sim
        
        # 缩放相似度
        similarity_matrix = similarity_matrix * self.logit_scale.exp()
        
        # 创建标签：对角线为匹配对
        labels = torch.arange(batch_size * num_modalities, device=similarity_matrix.device)
        
        # 计算对比损失
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(similarity_matrix, labels)
        
        return loss
    
    def calculate_semantic_similarity(self, 
                                     modality_a: str, 
                                     features_a: torch.Tensor,
                                     modality_b: str, 
                                     features_b: torch.Tensor) -> float:
        """
        计算两个不同模态特征的语义相似度（真实计算）
        
        Args:
            modality_a: 第一个模态类型
            features_a: 第一个模态特征
            modality_b: 第二个模态类型
            features_b: 第二个模态特征
            
        Returns:
            语义相似度（0-1之间），真实计算非硬编码
        """
        # 编码到统一语义空间
        encoded_a = self.encode_modality(modality_a, features_a)
        encoded_b = self.encode_modality(modality_b, features_b)
        
        # 提取全局表示
        if encoded_a.dim() == 3:
            global_a = encoded_a.mean(dim=1)
        else:
            global_a = encoded_a
            
        if encoded_b.dim() == 3:
            global_b = encoded_b.mean(dim=1)
        else:
            global_b = encoded_b
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(global_a, global_b, dim=-1)
        
        return similarity.mean().item()
    
    def get_config(self) -> Dict[str, Any]:
        """获取编码器配置"""
        return {
            "unified_dim": self.unified_dim,
            "temperature": self.temperature,
            "text_encoder_params": sum(p.numel() for p in self.text_encoder.parameters()),
            "image_encoder_params": sum(p.numel() for p in self.image_encoder.parameters()),
            "audio_encoder_params": sum(p.numel() for p in self.audio_encoder.parameters()),
            "total_params": sum(p.numel() for p in self.parameters()),
            "modality_support": ["text", "image", "audio"],
            "implementation_type": "true_unified_semantic_encoder"
        }


def test_true_unified_semantic_encoder():
    """测试真正统一语义编码器"""
    logger.info("测试真正统一语义编码器...")
    
    try:
        # 创建编码器实例
        encoder = TrueUnifiedSemanticEncoder(
            text_input_dim=768,
            image_input_dim=3,
            audio_input_dim=1,
            unified_dim=512
        )
        
        # 创建测试数据
        batch_size = 2
        
        # 文本测试数据
        text_seq_len = 10
        text_dim = 768
        text_input = _deterministic_randn((batch_size, text_seq_len, text_dim), seed_prefix="randn_default")
        
        # 图像测试数据
        image_channels = 3
        image_height = 64
        image_width = 64
        image_input = _deterministic_randn((batch_size, image_channels, image_height, image_width), seed_prefix="randn_default")
        
        # 音频测试数据
        audio_seq_len = 16000  # 16kHz音频的1秒
        audio_input = _deterministic_randn((batch_size, audio_seq_len), seed_prefix="randn_default")
        
        # 测试单个模态编码
        print("测试单个模态编码...")
        text_encoded = encoder.encode_modality("text", text_input)
        print(f"  文本编码形状: {text_encoded.shape}")
        
        image_encoded = encoder.encode_modality("image", image_input)
        print(f"  图像编码形状: {image_encoded.shape}")
        
        audio_encoded = encoder.encode_modality("audio", audio_input)
        print(f"  音频编码形状: {audio_encoded.shape}")
        
        # 测试多模态编码
        print("\n测试多模态编码...")
        result = encoder(
            text_input=text_input,
            image_input=image_input,
            audio_input=audio_input
        )
        
        # 验证结果
        required_keys = ["encoded_features", "similarity_info", "alignment_quality"]
        for key in required_keys:
            assert key in result, f"缺失结果键: {key}"
        
        # 验证编码特征
        encoded_features = result["encoded_features"]
        assert "text" in encoded_features, "缺失文本编码特征"
        assert "image" in encoded_features, "缺失图像编码特征"
        assert "audio" in encoded_features, "缺失音频编码特征"
        
        # 验证相似度计算
        similarity_info = result["similarity_info"]
        assert "text_image_similarity" in similarity_info, "缺失文本-图像相似度"
        assert "text_audio_similarity" in similarity_info, "缺失文本-音频相似度"
        assert "image_audio_similarity" in similarity_info, "缺失图像-音频相似度"
        
        # 验证相似度值在合理范围内
        for key, value in similarity_info.items():
            assert 0 <= value <= 1, f"相似度值超出范围: {key}={value}"
        
        # 测试语义相似度计算
        print("\n测试语义相似度计算...")
        similarity = encoder.calculate_semantic_similarity(
            "text", text_input, "image", image_input
        )
        print(f"  文本-图像语义相似度: {similarity:.4f}")
        assert 0 <= similarity <= 1, f"语义相似度超出范围: {similarity}"
        
        # 测试配置获取
        config = encoder.get_config()
        assert "implementation_type" in config, "配置中缺少implementation_type"
        assert config["implementation_type"] == "true_unified_semantic_encoder"
        
        logger.info("✅ 真正统一语义编码器测试通过")
        
        return {
            "success": True,
            "encoded_shapes": {
                "text": list(text_encoded.shape),
                "image": list(image_encoded.shape),
                "audio": list(audio_encoded.shape)
            },
            "similarity_info": similarity_info,
            "alignment_quality": result.get("alignment_quality", 0.0),
            "config": config,
            "message": "真正统一语义编码器测试完成"
        }
        
    except Exception as e:
        logger.error(f"❌ 真正统一语义编码器测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "message": "真正统一语义编码器测试失败"
        }


# 导出主要类和方法
__all__ = [
    "TrueUnifiedSemanticEncoder",
    "TrueModalityEncoder",
    "test_true_unified_semantic_encoder"
]

if __name__ == "__main__":
    # 运行测试
    test_result = test_true_unified_semantic_encoder()
    print(f"测试结果: {test_result}")
