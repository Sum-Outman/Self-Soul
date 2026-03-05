"""
AGI-Friendly Text Processing Module - From-Scratch Implementation

This module provides custom AGI-compliant text processing components that:
1. Support from-scratch training without external pretrained models
2. Provide lightweight but effective NLP capabilities
3. Maintain compatibility with existing AGI architecture
4. Enable autonomous learning and adaptation

Core Components:
- AGITokenizer: Character/subword-based tokenizer with dynamic vocabulary
- AGIEmbedding: Learnable embedding layer with position encoding  
- AGITransformerEncoder: Lightweight transformer for text encoding
- AGIEmotionAnalyzer: Emotion analysis without external dependencies
- AGILanguageModel: Simplified language modeling capability
"""

import re
import json
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging

# Cycle prevention for safe text generation
try:
    from .cycle_prevention_manager import CyclePreventionManager
except ImportError:
    from core.cycle_prevention_manager import CyclePreventionManager

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Configuration for AGI tokenizer"""
    vocab_size: int = 10000
    max_length: int = 512
    min_frequency: int = 2
    special_tokens: List[str] = None
    char_level: bool = False
    subword_level: bool = True
    unknown_token: str = "[UNK]"
    padding_token: str = "[PAD]"
    start_token: str = "[CLS]"
    separator_token: str = "[SEP]"
    mask_token: str = "[MASK]"
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = [
                self.padding_token,
                self.unknown_token,
                self.start_token,
                self.separator_token,
                self.mask_token
            ]


class AGITokenizer:
    """AGI-Friendly Tokenizer - From-scratch implementation
    
    Features:
    - Character and subword tokenization
    - Dynamic vocabulary building
    - Support for special tokens
    - Vocabulary persistence
    """
    
    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        self.vocab = {}
        self.reverse_vocab = {}
        self.word_frequencies = Counter()
        
        # Initialize with special tokens
        self._init_special_tokens()
        
    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary"""
        for i, token in enumerate(self.config.special_tokens):
            self.vocab[token] = i
            self.reverse_vocab[i] = token
            
    def build_vocab(self, texts: List[str], update: bool = False):
        """Build vocabulary from text corpus
        
        Args:
            texts: List of text strings
            update: Whether to update existing vocabulary
        """
        if not update:
            self.word_frequencies = Counter()
            
        # Tokenize texts and count frequencies
        for text in texts:
            if self.config.char_level:
                tokens = list(text.lower())
            else:
                # Simple word tokenization (can be enhanced)
                tokens = re.findall(r'\b\w+\b', text.lower())
            
            self.word_frequencies.update(tokens)
        
        # Filter by minimum frequency
        filtered_words = [
            word for word, freq in self.word_frequencies.items()
            if freq >= self.config.min_frequency
        ]
        
        # Limit vocabulary size (excluding special tokens)
        max_words = self.config.vocab_size - len(self.config.special_tokens)
        filtered_words = filtered_words[:max_words]
        
        # Add words to vocabulary
        current_idx = len(self.config.special_tokens)
        for word in filtered_words:
            if word not in self.vocab:
                self.vocab[word] = current_idx
                self.reverse_vocab[current_idx] = word
                current_idx += 1
                
        logger.info(f"Vocabulary built with {len(self.vocab)} tokens")
        
    def encode(self, text: str, add_special_tokens: bool = True, 
               max_length: int = None, padding: bool = True) -> List[int]:
        """Encode text to token IDs
        
        Args:
            text: Input text
            add_special_tokens: Whether to add [CLS] and [SEP] tokens
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            
        Returns:
            List of token IDs
        """
        if max_length is None:
            max_length = self.config.max_length
            
        # Tokenize
        if self.config.char_level:
            tokens = list(text.lower())
        else:
            tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab[self.config.unknown_token])
        
        # Add special tokens
        if add_special_tokens:
            cls_id = self.vocab[self.config.start_token]
            sep_id = self.vocab[self.config.separator_token]
            token_ids = [cls_id] + token_ids + [sep_id]
        
        # Truncate if needed
        if len(token_ids) > max_length:
            if add_special_tokens:
                # Keep [CLS] and [SEP]
                token_ids = [token_ids[0]] + token_ids[1:max_length-1] + [token_ids[-1]]
            else:
                token_ids = token_ids[:max_length]
        
        # Pad if needed
        if padding and len(token_ids) < max_length:
            pad_id = self.vocab[self.config.padding_token]
            token_ids = token_ids + [pad_id] * (max_length - len(token_ids))
            
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]
                if skip_special_tokens and token in self.config.special_tokens:
                    continue
                tokens.append(token)
        
        if self.config.char_level:
            return ''.join(tokens)
        else:
            return ' '.join(tokens)
    
    def save_vocab(self, filepath: str):
        """Save vocabulary to file
        
        Args:
            filepath: Path to save vocabulary
        """
        vocab_data = {
            'vocab': self.vocab,
            'config': self.config.__dict__,
            'word_frequencies': dict(self.word_frequencies)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Vocabulary saved to {filepath}")
    
    def load_vocab(self, filepath: str):
        """Load vocabulary from file
        
        Args:
            filepath: Path to vocabulary file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab = {k: int(v) for k, v in vocab_data['vocab'].items()}
        self.reverse_vocab = {int(v): k for k, v in vocab_data['vocab'].items()}
        self.word_frequencies = Counter(vocab_data['word_frequencies'])
        
        # Recreate config
        config_dict = vocab_data['config']
        self.config = TokenizerConfig(
            vocab_size=config_dict.get('vocab_size', 10000),
            max_length=config_dict.get('max_length', 512),
            min_frequency=config_dict.get('min_frequency', 2),
            special_tokens=config_dict.get('special_tokens'),
            char_level=config_dict.get('char_level', False),
            subword_level=config_dict.get('subword_level', True),
            unknown_token=config_dict.get('unknown_token', '[UNK]'),
            padding_token=config_dict.get('padding_token', '[PAD]'),
            start_token=config_dict.get('start_token', '[CLS]'),
            separator_token=config_dict.get('separator_token', '[SEP]'),
            mask_token=config_dict.get('mask_token', '[MASK]')
        )
        
        logger.info(f"Vocabulary loaded from {filepath} with {len(self.vocab)} tokens")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size including special tokens"""
        return len(self.vocab)
    
    def token_to_id(self, token: str) -> int:
        """Convert token to ID
        
        Args:
            token: Token string
            
        Returns:
            Token ID or unknown token ID
        """
        return self.vocab.get(token, self.vocab[self.config.unknown_token])
    
    def id_to_token(self, token_id: int) -> str:
        """Convert ID to token
        
        Args:
            token_id: Token ID
            
        Returns:
            Token string or unknown token
        """
        return self.reverse_vocab.get(token_id, self.config.unknown_token)


class AGIEmbedding(nn.Module):
    """AGI-Friendly Embedding Layer with Position Encoding
    
    Features:
    - Learnable token embeddings
    - Positional encoding (sinusoidal or learnable)
    - Layer normalization
    - Dropout for regularization
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, max_length: int = 512,
                 dropout: float = 0.1, use_positional_encoding: bool = True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.dropout_rate = dropout
        self.use_positional_encoding = use_positional_encoding
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        if use_positional_encoding:
            self.position_embedding = nn.Embedding(max_length, embedding_dim)
        else:
            self.register_buffer('position_ids', 
                                torch.arange(max_length).unsqueeze(0))
            self.position_encoding = self._create_sinusoidal_encoding()
            
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize embedding weights"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if hasattr(self, 'position_embedding'):
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
            
    def _create_sinusoidal_encoding(self) -> torch.Tensor:
        """Create sinusoidal positional encoding
        
        Returns:
            Positional encoding tensor
        """
        position = torch.arange(self.max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2) * 
                           -(math.log(10000.0) / self.embedding_dim))
        
        pe = torch.zeros(self.max_length, self.embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        return pe
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            token_ids: Token IDs tensor of shape (batch_size, seq_len)
            
        Returns:
            Embedded tensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(token_ids)
        
        # Positional embeddings
        if self.use_positional_encoding:
            position_ids = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
            position_embeds = self.position_embedding(position_ids)
        else:
            position_embeds = self.position_encoding[:, :seq_len, :]
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        
        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim


class SimpleAttention(nn.Module):
    """Simplified Attention Mechanism for AGI
    
    Features:
    - Multi-head attention
    - Scaled dot-product attention
    - Masking support
    """
    
    def __init__(self, embedding_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        assert embedding_dim % num_heads == 0, \
            f"Embedding dimension {embedding_dim} must be divisible by num_heads {num_heads}"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection
        self.output = nn.Linear(embedding_dim, embedding_dim)
        
        # Dropout
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            mask: Attention mask
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape for multi-head attention
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, v)
        
        # Reshape back
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embedding_dim)
        
        # Output projection
        output = self.output(context)
        output = self.output_dropout(output)
        
        return output


# Create global tokenizer instance for easy access
_default_tokenizer_config = TokenizerConfig(
    vocab_size=10000,
    max_length=512,
    min_frequency=2,
    char_level=False,
    subword_level=True
)

agi_tokenizer = AGITokenizer(_default_tokenizer_config)


class AGITransformerEncoderLayer(nn.Module):
    """AGI-Friendly Transformer Encoder Layer
    
    Features:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization
    - Residual connections
    """
    
    def __init__(self, embedding_dim: int, num_heads: int = 4, 
                 ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = embedding_dim * 4
            
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        # Self-attention
        self.attention = SimpleAttention(embedding_dim, num_heads, dropout)
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention_dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            mask: Attention mask
            
        Returns:
            Output tensor
        """
        # Self-attention with residual connection
        attention_output = self.attention(x, mask)
        x = self.attention_norm(x + self.attention_dropout(attention_output))
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + ffn_output)
        
        return x


class AGITransformerEncoder(nn.Module):
    """AGI-Friendly Transformer Encoder (BERT alternative)
    
    Features:
    - Multiple transformer layers
    - Pooling layer for sentence embeddings
    - Configurable depth and dimensions
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, 
                 num_layers: int = 4, num_heads: int = 4, 
                 ff_dim: int = None, max_length: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        
        # Embedding layer
        self.embedding = AGIEmbedding(vocab_size, embedding_dim, max_length, dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            AGITransformerEncoderLayer(embedding_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Pooling layer for sentence embeddings
        self.pooler = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh()
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, token_ids: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass
        
        Args:
            token_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Dictionary with:
            - last_hidden_state: Sequence output
            - pooler_output: Sentence embedding
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (token_ids != 0).float()
        
        # Get embeddings
        embeddings = self.embedding(token_ids)
        
        # Apply transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask.unsqueeze(1).unsqueeze(2))
        
        # Pooling (use [CLS] token representation)
        cls_token = hidden_states[:, 0, :]  # First token is [CLS]
        pooler_output = self.pooler(cls_token)
        
        # Apply output normalization
        hidden_states = self.output_norm(hidden_states)
        
        return {
            'last_hidden_state': hidden_states,
            'pooler_output': pooler_output
        }
    
    def get_text_embedding(self, text: str, tokenizer: AGITokenizer = None) -> torch.Tensor:
        """Get text embedding (alternative to BERT)
        
        Args:
            text: Input text
            tokenizer: AGITokenizer instance (uses global if None)
            
        Returns:
            Text embedding tensor
        """
        if tokenizer is None:
            tokenizer = agi_tokenizer
        
        # Tokenize
        token_ids = tokenizer.encode(text, add_special_tokens=True, padding=True)
        token_tensor = torch.tensor([token_ids])
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.forward(token_tensor)
            embedding = outputs['pooler_output']
        
        return embedding


class AGIEmotionAnalyzer(nn.Module):
    """AGI-Friendly Emotion Analyzer (replaces external sentiment analysis)
    
    Features:
    - Multi-emotion prediction (valence, arousal, dominance)
    - Based on AGI transformer encoder
    - From-scratch training support
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256,
                 num_layers: int = 3, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        # Transformer encoder for text understanding
        self.encoder = AGITransformerEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Emotion prediction heads
        self.emotion_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 3)  # valence, arousal, dominance
        )
        
        # Emotion classification (discrete emotions)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 9)  # 9 basic emotions
        )
        
    def forward(self, token_ids: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass
        
        Args:
            token_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Dictionary with emotion predictions
        """
        # Encode text
        encoder_outputs = self.encoder(token_ids, attention_mask)
        pooled_output = encoder_outputs['pooler_output']
        
        # Predict emotion dimensions
        emotion_dims = torch.sigmoid(self.emotion_head(pooled_output))
        
        # Predict emotion categories
        emotion_logits = self.emotion_classifier(pooled_output)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        
        # Map to emotion names
        emotion_names = ['happy', 'sad', 'angry', 'fear', 'surprised', 
                        'calm', 'curious', 'confident', 'neutral']
        
        return {
            'emotion_dimensions': emotion_dims,  # valence, arousal, dominance
            'emotion_probabilities': emotion_probs,
            'emotion_names': emotion_names,
            'pooler_output': pooled_output
        }
    
    def analyze_text(self, text: str, tokenizer: AGITokenizer = None) -> Dict[str, Any]:
        """Analyze emotion in text
        
        Args:
            text: Input text
            tokenizer: AGITokenizer instance
            
        Returns:
            Emotion analysis results
        """
        if tokenizer is None:
            tokenizer = agi_tokenizer
        
        # Tokenize
        token_ids = tokenizer.encode(text, add_special_tokens=True, padding=True)
        token_tensor = torch.tensor([token_ids])
        
        # Analyze
        with torch.no_grad():
            outputs = self.forward(token_tensor)
            
            # Get emotion with highest probability
            emotion_probs = outputs['emotion_probabilities'][0]
            emotion_idx = torch.argmax(emotion_probs).item()
            emotion_name = outputs['emotion_names'][emotion_idx]
            emotion_confidence = emotion_probs[emotion_idx].item()
            
            # Get emotion dimensions
            emotion_dims = outputs['emotion_dimensions'][0]
            valence, arousal, dominance = emotion_dims.tolist()
        
        return {
            'dominant_emotion': emotion_name,
            'confidence': emotion_confidence,
            'valence': valence,
            'arousal': arousal,
            'dominance': dominance,
            'all_emotions': {
                name: prob.item() for name, prob in zip(outputs['emotion_names'], emotion_probs[0])
            }
        }


class AGILanguageModel(nn.Module):
    """AGI-Friendly Language Model (GPT-2 alternative)
    
    Features:
    - Causal language modeling
    - Simplified transformer decoder
    - Text generation capability
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256,
                 num_layers: int = 4, num_heads: int = 4,
                 ff_dim: int = None, max_length: int = 512,
                 dropout: float = 0.1, enable_cycle_prevention: bool = True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.max_length = max_length
        self.enable_cycle_prevention = enable_cycle_prevention
        
        # Embedding layer (shared with output)
        self.embedding = AGIEmbedding(vocab_size, embedding_dim, max_length, dropout)
        
        # Causal transformer layers
        self.layers = nn.ModuleList([
            AGITransformerEncoderLayer(embedding_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection (tied to embedding weights)
        self.output_projection = nn.Linear(embedding_dim, vocab_size, bias=False)
        
        # Tie weights (input and output embeddings)
        self.output_projection.weight = self.embedding.token_embedding.weight
        
        # Layer normalization
        self.final_norm = nn.LayerNorm(embedding_dim)
        
        # Cycle prevention manager for safe generation
        if self.enable_cycle_prevention:
            try:
                self.cycle_prevention_manager = CyclePreventionManager(
                    config={
                        "history_buffer_size": 10,
                        "repeat_threshold": 3,
                        "base_temperature": 0.7,
                        "max_temperature": 1.2,
                        "base_repetition_penalty": 1.1,
                        "max_repetition_penalty": 1.8,
                    },
                    enable_adaptive_layer=True
                )
                logger.info("AGILanguageModel: Cycle prevention manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize cycle prevention manager: {e}")
                self.cycle_prevention_manager = None
        else:
            self.cycle_prevention_manager = None
        
    def forward(self, token_ids: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass for language modeling
        
        Args:
            token_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Causal attention mask
            
        Returns:
            Dictionary with logits and hidden states
        """
        batch_size, seq_len = token_ids.shape
        
        # Create causal attention mask
        if attention_mask is None:
            # Create causal mask (upper triangular)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
            attention_mask = causal_mask
        else:
            # Combine with causal mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) * causal_mask
        
        # Get embeddings
        embeddings = self.embedding(token_ids)
        
        # Apply transformer layers with causal masking
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        # Project to vocabulary
        logits = self.output_projection(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states
        }
    
    def generate(self, prompt: str, tokenizer: AGITokenizer = None,
                 max_length: int = 100, temperature: float = 1.0,
                 top_k: int = 50) -> str:
        """Generate text from prompt
        
        Args:
            prompt: Input prompt
            tokenizer: AGITokenizer instance
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text
        """
        if tokenizer is None:
            tokenizer = agi_tokenizer
        
        # Encode prompt
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids])
        
        # Generate tokens
        generated_ids = list(prompt_ids)
        
        for _ in range(max_length):
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(input_ids)
                logits = outputs['logits'][0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits[top_k_indices] = top_k_logits
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()
            
            # Check for end of sequence
            if next_token_id == tokenizer.token_to_id(tokenizer.config.separator_token):
                break
            
            # Add token and update input
            generated_ids.append(next_token_id)
            input_ids = torch.tensor([generated_ids[-self.max_length:]])
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
    def generate_safe(self, prompt: str, tokenizer: AGITokenizer = None,
                     max_length: int = 100, temperature: float = 1.0,
                     top_k: int = 50, repetition_penalty: float = 1.1) -> Tuple[str, Dict[str, Any]]:
        """
        Safe text generation with cycle prevention
        
        This method wraps the original generate method with cycle prevention:
        1. Basic layer: Embedded-style protection (buffer cleanup + repeat detection + watchdog)
        2. Adaptive layer: Scene-aware parameter adjustment + performance feedback
        
        Args:
            prompt: Input prompt
            tokenizer: AGITokenizer instance
            max_length: Maximum generation length
            temperature: Sampling temperature (can be overridden by adaptive layer)
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty coefficient (can be overridden by adaptive layer)
            
        Returns:
            Tuple of (generated_text, protection_info)
        """
        # If cycle prevention is disabled, use original generate method
        if not self.enable_cycle_prevention or self.cycle_prevention_manager is None:
            generated_text = self.generate(
                prompt=prompt,
                tokenizer=tokenizer,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k
            )
            return generated_text, {"protection_layer": "disabled"}
        
        # Define inner generation function for cycle prevention manager
        def inner_generate_func(context: str, params: Dict[str, Any]) -> str:
            """Inner generation function with dynamic parameters"""
            # Extract parameters from cycle prevention manager
            adaptive_temperature = params.get("temperature", temperature)
            adaptive_repetition_penalty = params.get("repetition_penalty", repetition_penalty)
            
            # For AGILanguageModel, we need to simulate repetition penalty
            # Since the model doesn't have built-in repetition penalty,
            # we'll adjust temperature based on repetition_penalty
            adjusted_temperature = adaptive_temperature * adaptive_repetition_penalty
            
            # Use original generate method with adjusted parameters
            return self.generate(
                prompt=context,
                tokenizer=tokenizer,
                max_length=max_length,
                temperature=adjusted_temperature,
                top_k=top_k
            )
        
        # Use cycle prevention manager for safe generation
        generated_text, protection_info = self.cycle_prevention_manager.generate_safe(
            prompt=prompt,
            generate_func=inner_generate_func,
            max_attempts=3
        )
        
        return generated_text, protection_info


# Create global instances for easy access
_default_encoder_config = {
    'vocab_size': 10000,
    'embedding_dim': 256,
    'num_layers': 4,
    'num_heads': 4,
    'max_length': 512,
    'dropout': 0.1
}

# Note: These will be initialized on-demand to avoid unnecessary memory usage
# Users should create instances with specific vocabulary sizes based on their tokenizer

def create_agi_encoder(vocab_size: int = None, config: Dict[str, Any] = None) -> AGITransformerEncoder:
    """Create AGI transformer encoder instance
    
    Args:
        vocab_size: Vocabulary size (required)
        config: Configuration dictionary
        
    Returns:
        AGITransformerEncoder instance
    """
    if config is None:
        config = _default_encoder_config.copy()
    
    if vocab_size is not None:
        config['vocab_size'] = vocab_size
    
    return AGITransformerEncoder(**config)

def create_agi_emotion_analyzer(vocab_size: int = None, config: Dict[str, Any] = None) -> AGIEmotionAnalyzer:
    """Create AGI emotion analyzer instance
    
    Args:
        vocab_size: Vocabulary size (required)
        config: Configuration dictionary
        
    Returns:
        AGIEmotionAnalyzer instance
    """
    if config is None:
        config = {
            'vocab_size': 10000,
            'embedding_dim': 256,
            'num_layers': 3,
            'num_heads': 4,
            'dropout': 0.1
        }
    
    if vocab_size is not None:
        config['vocab_size'] = vocab_size
    
    return AGIEmotionAnalyzer(**config)

def create_agi_language_model(vocab_size: int = None, config: Dict[str, Any] = None, 
                              enable_cycle_prevention: bool = True) -> AGILanguageModel:
    """Create AGI language model instance with optional cycle prevention
    
    Args:
        vocab_size: Vocabulary size (required)
        config: Configuration dictionary
        enable_cycle_prevention: Whether to enable cycle prevention for safe generation
        
    Returns:
        AGILanguageModel instance
    """
    if config is None:
        config = {
            'vocab_size': 10000,
            'embedding_dim': 256,
            'num_layers': 4,
            'num_heads': 4,
            'max_length': 512,
            'dropout': 0.1
        }
    
    if vocab_size is not None:
        config['vocab_size'] = vocab_size
    
    # Add cycle prevention parameter if not already in config
    if 'enable_cycle_prevention' not in config:
        config['enable_cycle_prevention'] = enable_cycle_prevention
    
    return AGILanguageModel(**config)