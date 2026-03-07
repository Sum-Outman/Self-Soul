"""
统一认知架构的神经网络

包含统一认知架构使用的所有神经网络组件，
包括编码器、投影网络、注意力网络和记忆网络。

所有网络使用统一的张量维度以实现无缝通信。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Any


# ============================================================================
# 编码器网络
# ============================================================================

class TextEncoder(nn.Module):
    """统一表示空间的文本编码器"""
    
    def __init__(self, embedding_dim: int = 1024, vocab_size: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 词元嵌入
        self.embedding = nn.Embedding(vocab_size, embedding_dim // 4)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim // 4,
            nhead=8,
            dim_feedforward=embedding_dim // 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # 投影到统一维度
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim // 4, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(embedding_dim // 4)
    
    def forward(self, text_input) -> torch.Tensor:
        """
        编码文本输入。
        
        Args:
            text_input: 可以是词元张量或原始文本字符串
            
        Returns:
            编码的文本表示，形状为 (batch_size, embedding_dim)
        """
        # 处理不同类型的输入
        if isinstance(text_input, str):
            # 目前使用简单的分词
            tokens = self._tokenize_text(text_input)
        elif isinstance(text_input, list):
            tokens = torch.tensor(text_input).long()
        else:
            tokens = text_input
        
        # 如果需要添加批次维度
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)
        
        # 获取嵌入
        embeddings = self.embedding(tokens)
        
        # 添加位置编码
        embeddings = self.positional_encoding(embeddings)
        
        # 应用transformer
        encoded = self.transformer(embeddings)
        
        # 池化（平均池化）
        pooled = encoded.mean(dim=1)
        
        # 投影到统一维度
        projected = self.projection(pooled)
        
        return projected
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """简单的分词（实际中会使用适当的分词器）"""
        # 将字符转换为整数编码
        tokens = [ord(c) % 10000 for c in text[:100]]
        if len(tokens) < 100:
            tokens += [0] * (100 - len(tokens))
        return torch.tensor(tokens).long()


class ImageEncoder(nn.Module):
    """统一表示空间的图像编码器"""
    
    def __init__(self, embedding_dim: int = 1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 块1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 块2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 块3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 块4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 投影到统一维度
        self.projection = nn.Sequential(
            nn.Linear(512 * 4 * 4, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, image_input) -> torch.Tensor:
        """
        编码图像输入。
        
        Args:
            image_input: 可以是张量、numpy数组或图像路径
            
        Returns:
            编码的图像表示，形状为 (batch_size, embedding_dim)
        """
        # 处理不同类型的输入
        if isinstance(image_input, str):
            # 从文件加载图像（简化版）
            # 实际中会使用PIL或OpenCV
            image_tensor = torch.randn(1, 3, 224, 224)
        elif isinstance(image_input, torch.Tensor):
            image_tensor = image_input
        else:
            # 假设是numpy数组
            import numpy as np
            if isinstance(image_input, np.ndarray):
                image_tensor = torch.from_numpy(image_input).float()
            else:
                raise ValueError(f"不支持的图像输入类型: {type(image_input)}")
        
        # 如果需要添加批次维度
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # 如果需要调整大小（简化版）
        if image_tensor.shape[-2:] != (224, 224):
            image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear')
        
        # 应用卷积层
        features = self.conv_layers(image_tensor)
        
        # 展平
        features = features.view(features.size(0), -1)
        
        # 投影到统一维度
        encoded = self.projection(features)
        
        return encoded


class AudioEncoder(nn.Module):
    """统一表示空间的音频编码器"""
    
    def __init__(self, embedding_dim: int = 1024, sample_rate: int = 16000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sample_rate = sample_rate
        
        # 1D convolutional layers
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            # Block 3
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            # Block 4
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128)
        )
        
        # Projection to unified dimension
        self.projection = nn.Sequential(
            nn.Linear(512 * 128, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, audio_input) -> torch.Tensor:
        """
        Encode audio input.
        
        Args:
            audio_input: Can be tensor, numpy array, or audio path
            
        Returns:
            Encoded audio representation of shape (batch_size, embedding_dim)
        """
        # Handle different input types
        if isinstance(audio_input, str):
            # Load audio from file (simplified)
            # In practice would use librosa or torchaudio
            audio_tensor = torch.randn(1, 1, self.sample_rate)
        elif isinstance(audio_input, torch.Tensor):
            audio_tensor = audio_input
        else:
            # Assume numpy array
            import numpy as np
            if isinstance(audio_input, np.ndarray):
                audio_tensor = torch.from_numpy(audio_input).float()
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
        
        # Ensure correct shape
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, n_samples)
        elif len(audio_tensor.shape) == 2:
            audio_tensor = audio_tensor.unsqueeze(1)  # (batch, 1, n_samples)
        
        # Resample if needed (simplified)
        if audio_tensor.shape[-1] != self.sample_rate:
            audio_tensor = F.interpolate(
                audio_tensor, 
                size=self.sample_rate, 
                mode='linear'
            )
        
        # Apply convolutional layers
        features = self.conv_layers(audio_tensor)
        
        # Flatten
        features = features.view(features.size(0), -1)
        
        # Project to unified dimension
        encoded = self.projection(features)
        
        return encoded


class StructuredEncoder(nn.Module):
    """Structured data encoder for unified representation space"""
    
    def __init__(self, embedding_dim: int = 1024, num_features: int = 100):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(num_features, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, embedding_dim // 2),
            nn.ReLU()
        )
        
        # Self-attention for feature interactions
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim // 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Projection to unified dimension
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, structured_input) -> torch.Tensor:
        """
        Encode structured data input.
        
        Args:
            structured_input: Can be tensor, numpy array, or dictionary
            
        Returns:
            Encoded structured representation of shape (batch_size, embedding_dim)
        """
        # Handle different input types
        if isinstance(structured_input, dict):
            # Convert dictionary to tensor (simplified)
            # In practice would handle specific schema
            import numpy as np
            features = []
            for key, value in structured_input.items():
                if isinstance(value, (int, float)):
                    features.append(float(value))
                elif isinstance(value, bool):
                    features.append(1.0 if value else 0.0)
                else:
                    features.append(0.0)
            
            # Pad or truncate to num_features
            if len(features) > self.num_features:
                features = features[:self.num_features]
            else:
                features += [0.0] * (self.num_features - len(features))
            
            input_tensor = torch.tensor(features).float()
        
        elif isinstance(structured_input, torch.Tensor):
            input_tensor = structured_input
        else:
            # Assume numpy array
            import numpy as np
            if isinstance(structured_input, np.ndarray):
                input_tensor = torch.from_numpy(structured_input).float()
            else:
                raise ValueError(f"Unsupported structured input type: {type(structured_input)}")
        
        # Add batch dimension if needed
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Ensure correct feature dimension
        if input_tensor.shape[-1] > self.num_features:
            input_tensor = input_tensor[..., :self.num_features]
        elif input_tensor.shape[-1] < self.num_features:
            padding = torch.zeros(
                *input_tensor.shape[:-1], 
                self.num_features - input_tensor.shape[-1]
            )
            input_tensor = torch.cat([input_tensor, padding], dim=-1)
        
        # Process features
        processed = self.feature_processor(input_tensor)
        
        # Apply self-attention
        # Add sequence dimension for attention
        processed_seq = processed.unsqueeze(1)
        attended, _ = self.attention(processed_seq, processed_seq, processed_seq)
        attended = attended.squeeze(1)
        
        # Project to unified dimension
        encoded = self.projection(attended)
        
        return encoded


# ============================================================================
# 统一投影网络
# ============================================================================

class UnifiedProjectionNetwork(nn.Module):
    """Projects any modality representation to unified space"""
    
    def __init__(self, input_dim: int, output_dim: int = 1024):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(output_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.Tanh(),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class AttentionNetwork(nn.Module):
    """Attention network for hierarchical attention"""
    
    def __init__(self, embedding_dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Multi-head attention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attended, _ = self.multihead_attention(x, x, x)
        
        # Add & norm
        x = self.norm1(x + self.dropout(attended))
        
        # Feed-forward
        ff_out = self.ffn(x)
        
        # Add & norm
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class MemoryNetwork(nn.Module):
    """Memory network for episodic and semantic memory"""
    
    def __init__(self, embedding_dim: int = 1024, memory_size: int = 1000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        
        # Key-value memory
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Memory retrieval
        self.retrieval_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Memory update
        self.update_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim * 2, embedding_dim)
    
    def forward(self, query: torch.Tensor, 
                memory_keys: torch.Tensor, 
                memory_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve from memory.
        
        Args:
            query: Query tensor
            memory_keys: Memory keys
            memory_values: Memory values
            
        Returns:
            Retrieved memory and attention weights
        """
        # Project query to key space
        query_key = self.key_projection(query)
        
        # Retrieve from memory
        retrieved, attention_weights = self.retrieval_attention(
            query_key.unsqueeze(1),  # Add sequence dimension
            memory_keys,
            memory_values
        )
        
        retrieved = retrieved.squeeze(1)
        
        # Combine query and retrieved memory
        combined = torch.cat([query, retrieved], dim=-1)
        
        # Project to output dimension
        output = self.output_projection(combined)
        
        return output, attention_weights


# ============================================================================
# 实用网络
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class ResidualBlock(nn.Module):
    """Residual block for deep networks"""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.block(x)
        x = self.norm(x + residual)
        return x


# ============================================================================
# 工厂函数
# ============================================================================

def create_text_encoder(embedding_dim: int = 1024) -> TextEncoder:
    """Factory function for text encoder"""
    return TextEncoder(embedding_dim=embedding_dim)


def create_image_encoder(embedding_dim: int = 1024) -> ImageEncoder:
    """Factory function for image encoder"""
    return ImageEncoder(embedding_dim=embedding_dim)


def create_audio_encoder(embedding_dim: int = 1024) -> AudioEncoder:
    """Factory function for audio encoder"""
    return AudioEncoder(embedding_dim=embedding_dim)


def create_structured_encoder(embedding_dim: int = 1024) -> StructuredEncoder:
    """Factory function for structured encoder"""
    return StructuredEncoder(embedding_dim=embedding_dim)


def create_unified_projection(input_dim: int, output_dim: int = 1024) -> UnifiedProjectionNetwork:
    """Factory function for unified projection network"""
    return UnifiedProjectionNetwork(input_dim=input_dim, output_dim=output_dim)


def create_attention_network(embedding_dim: int = 1024) -> AttentionNetwork:
    """Factory function for attention network"""
    return AttentionNetwork(embedding_dim=embedding_dim)


def create_memory_network(embedding_dim: int = 1024) -> MemoryNetwork:
    """Factory function for memory network"""
    return MemoryNetwork(embedding_dim=embedding_dim)


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    # Encoders
    'TextEncoder',
    'ImageEncoder',
    'AudioEncoder',
    'StructuredEncoder',
    
    # Projection Networks
    'UnifiedProjectionNetwork',
    'AttentionNetwork',
    'MemoryNetwork',
    
    # Utility Networks
    'PositionalEncoding',
    'ResidualBlock',
    
    # Factory Functions
    'create_text_encoder',
    'create_image_encoder',
    'create_audio_encoder',
    'create_structured_encoder',
    'create_unified_projection',
    'create_attention_network',
    'create_memory_network'
]