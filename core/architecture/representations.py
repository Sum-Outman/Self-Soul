"""
Representations module for unified cognitive architecture.
Includes tokenizers, encoders, and embedding spaces.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import re
import time
from typing import Dict, List, Any, Optional

# Import error handler from error_handling module to avoid circular dependency
from core.error_handling import error_handler


class SimpleOutput:
    """Simple output class for model outputs"""
    
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer models"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (non-trainable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class AdvancedTokenizer:
    """Advanced Text Tokenizer - Enhanced tokenization with semantic understanding"""
    
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.rev_vocab = {}
        self.word_frequency = {}
        self.subword_units = {}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.sep_token_id = 2
        self.cls_token_id = 3
        self.mask_token_id = 4
        self.special_tokens = ['[PAD]', '[UNK]', '[SEP]', '[CLS]', '[MASK]']
        self.max_sequence_length = 512
        self._initialize_advanced_vocab()
        
    def _initialize_advanced_vocab(self):
        """Initialize advanced vocabulary with semantic structure"""
        # Add special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.rev_vocab[i] = token
        
        # Initialize with common English words and semantic patterns
        common_words = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
            'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
            'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
            'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
            'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'
        ]
        
        # Add common words to vocabulary
        for i, word in enumerate(common_words, start=len(self.special_tokens)):
            if i < self.vocab_size:
                self.vocab[word] = i
                self.rev_vocab[i] = word
                self.word_frequency[word] = 1
    
    def tokenize(self, text):
        """Advanced text tokenization with semantic analysis"""
        # Text preprocessing
        text = self._preprocess_text(text)
        
        # Multi-level tokenization
        tokens = self._multi_level_tokenization(text)
        
        # Semantic chunking
        chunks = self._semantic_chunking(tokens)
        
        return chunks
    
    def _preprocess_text(self, text):
        """Preprocess text with enhanced cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle contractions and special characters
        text = self._normalize_contractions(text)
        
        return text
    
    def _normalize_contractions(self, text):
        """Normalize English contractions"""
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
            "'ve": " have", "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _multi_level_tokenization(self, text):
        """Multi-level tokenization with word and subword units"""
        # Word-level tokenization
        words = re.findall(r'\b\w+\b|[.,!?;":()\[\]{}]', text)
        
        # Subword analysis for unknown words
        enhanced_tokens = []
        for word in words:
            if word in self.vocab:
                enhanced_tokens.append(word)
            else:
                # Handle unknown words with subword segmentation
                subwords = self._subword_segmentation(word)
                enhanced_tokens.extend(subwords)
        
        return enhanced_tokens
    
    def _subword_segmentation(self, word):
        """Segment unknown words into subword units"""
        # Simple subword segmentation based on common patterns
        if len(word) <= 3:
            return [word]
        
        # Try common prefixes and suffixes
        prefixes = ['un', 're', 'pre', 'dis', 'mis', 'over', 'under']
        suffixes = ['ing', 'ed', 's', 'es', 'ly', 'ment', 'ness', 'ful', 'less']
        
        # Check for prefixes
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                return [prefix, word[len(prefix):]]
        
        # Check for suffixes
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return [word[:-len(suffix)], suffix]
        
        # Default: split into character n-grams
        return self._character_ngrams(word)
    
    def _character_ngrams(self, word, n=3):
        """Generate character n-grams for unknown words"""
        ngrams = []
        for i in range(0, len(word) - n + 1):
            ngram = word[i:i+n]
            ngrams.append(ngram)
        return ngrams if ngrams else [word]
    
    def _semantic_chunking(self, tokens):
        """Semantic chunking based on linguistic patterns"""
        chunks = []
        current_chunk = []
        
        for token in tokens:
            # Check for sentence boundaries
            if token in ['.', '!', '?']:
                if current_chunk:
                    chunks.extend(current_chunk)
                    chunks.append(token)
                    current_chunk = []
            else:
                current_chunk.append(token)
        
        # Add remaining tokens
        if current_chunk:
            chunks.extend(current_chunk)
        
        return chunks
    
    def encode(self, tokens, max_length=None):
        """Encode tokens to IDs with padding/truncation"""
        if max_length is None:
            max_length = self.max_sequence_length
        
        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # Handle unknown tokens
                if len(token) > 1:
                    # Try subword segmentation for unknown tokens
                    subwords = self._subword_segmentation(token)
                    for subword in subwords:
                        if subword in self.vocab:
                            token_ids.append(self.vocab[subword])
                        else:
                            token_ids.append(self.unk_token_id)
                else:
                    token_ids.append(self.unk_token_id)
        
        # Truncate or pad
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids += [self.pad_token_id] * (max_length - len(token_ids))
        
        return token_ids
    
    def decode(self, token_ids):
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.rev_vocab:
                token = self.rev_vocab[token_id]
                if token not in self.special_tokens:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    def update_vocabulary(self, new_tokens):
        """Update vocabulary with new tokens"""
        for token in new_tokens:
            if token not in self.vocab and len(self.vocab) < self.vocab_size:
                new_id = len(self.vocab)
                self.vocab[token] = new_id
                self.rev_vocab[new_id] = token
                self.word_frequency[token] = 1
            elif token in self.word_frequency:
                self.word_frequency[token] += 1
    
    def get_vocabulary_stats(self):
        """Get vocabulary statistics"""
        return {
            'total_vocab_size': len(self.vocab),
            'word_frequency_distribution': dict(sorted(self.word_frequency.items(), key=lambda x: x[1], reverse=True)[:20]),
            'special_tokens': self.special_tokens,
            'max_sequence_length': self.max_sequence_length
        }


class NeuralEmbeddingSpace:
    """Neural Embedding Space - Unified representation for all data types with dynamic dimensionality"""
    
    def __init__(self, representation_dim=768, adaptive_dimensions=True):
        # Lazy import to avoid circular dependency
        try:
            from core.memory_optimization import memory_optimizer
            lightweight_mode = memory_optimizer.lightweight_mode
        except (ImportError, AttributeError):
            # If memory optimization module is unavailable or missing attribute, use default
            lightweight_mode = False
        
        # Adjust representation dimension based on memory optimization settings
        self.representation_dim = representation_dim
        if lightweight_mode:
            self.representation_dim = min(representation_dim, 384)
            
        self.adaptive_dimensions = adaptive_dimensions
        self.tokenizer = AdvancedTokenizer()
        self.text_encoder = self._create_advanced_text_encoder()
        self.criterion = nn.MSELoss()
        
        # Optimizer parameter adjustment
        optimizer_params = {
            'lr': 1e-4,
            'weight_decay': 0.01  # Add weight decay for better generalization
        }
        
        if lightweight_mode:
            # In lightweight mode, use fewer parameters
            self.criterion = nn.L1Loss()  # More computationally efficient loss function
            optimizer_params['lr'] = 5e-5  # Smaller learning rate
            
        # Robust optimizer initialization
        try:
            self.optimizer = optim.Adam(self.text_encoder.parameters(), **optimizer_params)
        except Exception as e:
            error_handler.handle_error(e, "NeuralEmbeddingSpace", "Failed to initialize optimizer")
            # As a fallback, temporarily don't initialize optimizer
            self.optimizer = None
        self.training_mode = False
        self.dimensionality_history = []
        self.complexity_threshold = 0.7
        
    def _create_advanced_text_encoder(self):
        """Create advanced transformer-based text encoder with real neural architecture"""
        # Import here to avoid circular dependency
        try:
            from core.memory_optimization import memory_optimizer
        except ImportError:
            memory_optimizer = None
        
        class AdvancedTextEncoder(nn.Module):
            @staticmethod
            def get_lightweight_params():
                """Return parameters for lightweight mode"""
                return {
                    'd_model': 384,
                    'nhead': 6,
                    'num_layers': 3,
                    'dim_feedforward': 1536,
                    'vocab_size': 15000,
                    'dropout': 0.05
                }
                
            def __init__(self, vocab_size=30000, d_model=768, nhead=12, num_layers=6, dim_feedforward=3072, dropout=0.1):
                # Adjust parameters based on memory optimization settings
                if memory_optimizer and memory_optimizer.lightweight_mode:
                    lightweight_params = self.get_lightweight_params()
                    vocab_size = lightweight_params['vocab_size']
                    d_model = lightweight_params['d_model']
                    nhead = lightweight_params['nhead']
                    num_layers = lightweight_params['num_layers']
                    dim_feedforward = lightweight_params['dim_feedforward']
                    dropout = lightweight_params['dropout']
                    
                super(AdvancedTextEncoder, self).__init__()
                self.d_model = d_model
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.positional_encoding = PositionalEncoding(d_model, dropout)
                
                # Multi-layer transformer encoder with advanced configuration
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation='gelu'
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                # Layer normalization and output projection
                self.layer_norm = nn.LayerNorm(d_model)
                self.output_projection = nn.Linear(d_model, d_model)
                
                # Attention mechanisms for different linguistic levels
                self.word_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                self.sentence_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                
                # Hierarchical processing components
                # Skip LSTM components completely in lightweight mode to reduce memory usage
                self.use_lstm = not (memory_optimizer and memory_optimizer.lightweight_mode)
                self.word_level_encoder = None
                self.sentence_level_encoder = None
                if self.use_lstm:
                    try:
                        # Ensure d_model is at least 2 to avoid LSTM dimension 0
                        lstm_hidden_size = max(d_model // 2, 1)
                        self.word_level_encoder = nn.LSTM(d_model, lstm_hidden_size, batch_first=True, bidirectional=True)
                        self.sentence_level_encoder = nn.LSTM(d_model, lstm_hidden_size, batch_first=True, bidirectional=True)
                    except Exception as e:
                        error_handler.handle_error(e, "AdvancedTextEncoder", "Failed to initialize LSTM components")
                        self.use_lstm = False
                
                # Semantic enhancement layers
                self.semantic_enhancer = nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 2, d_model),
                    nn.LayerNorm(d_model)
                )
                
                # Contextual understanding components
                # Use fewer layers in lightweight mode
                context_layers = 1 if (memory_optimizer and memory_optimizer.lightweight_mode) else 2
                self.context_encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='gelu'),
                    num_layers=context_layers
                )
                
                self.dropout = nn.Dropout(dropout)
                self.init_weights()
                
            def init_weights(self):
                """Initialize weights with advanced initialization"""
                initrange = 0.1
                self.embedding.weight.data.uniform_(-initrange, initrange)
                
            def forward(self, input_ids, attention_mask=None):
                # Get embeddings
                embedded = self.embedding(input_ids) * math.sqrt(self.d_model)
                
                # Add positional encoding
                embedded = self.positional_encoding(embedded)
                
                # Apply transformer encoder with attention mask
                if attention_mask is not None:
                    # Convert attention mask to transformer format
                    mask = (attention_mask == 0)
                    transformer_output = self.transformer(embedded, src_key_padding_mask=mask)
                else:
                    transformer_output = self.transformer(embedded)
                
                # Apply layer normalization
                transformer_output = self.layer_norm(transformer_output)
                
                # Word-level attention
                word_attended, word_weights = self.word_attention(
                    transformer_output, transformer_output, transformer_output
                )
                
                # Semantic enhancement
                enhanced_output = self.semantic_enhancer(transformer_output)
                
                # Contextual understanding
                contextual_output = self.context_encoder(enhanced_output)
                
                # Hierarchical processing based on use_lstm flag
                if self.use_lstm and not (memory_optimizer and memory_optimizer.lightweight_mode):
                    # Full mode hierarchical processing
                    # Hierarchical processing: word to sentence level
                    word_lstm_out, _ = self.word_level_encoder(transformer_output)
                    
                    # Sentence-level attention (using mean pooling for sentence representation)
                    sentence_repr = word_lstm_out.mean(dim=1).unsqueeze(1)
                    sentence_attended, sentence_weights = self.sentence_attention(
                        sentence_repr, sentence_repr, sentence_repr
                    )
                    
                    # Combine outputs with LSTM processing
                    combined_output = transformer_output + word_attended + contextual_output + sentence_attended.mean(dim=1).unsqueeze(1)
                else:
                    # Simplified processing: don't use LSTM components
                    combined_output = transformer_output + word_attended + contextual_output
                
                # Final projection
                final_output = self.output_projection(combined_output)
                final_output = self.dropout(final_output)
                
                return SimpleOutput(last_hidden_state=final_output)
        
        return AdvancedTextEncoder()
        
    def enable_training(self):
        """Enable training mode"""
        self.training_mode = True
        self.text_encoder.train()
        
    def disable_training(self):
        """Disable training mode"""
        self.training_mode = False
        self.text_encoder.eval()
        
    def _encode_text(self, text):
        """Encode text data"""
        try:
            # Tokenize text
            tokens = self.tokenizer.tokenize(text)
            # Convert tokens to IDs (simple implementation)
            input_ids = [self.tokenizer.vocab.get(token, self.tokenizer.unk_token_id) for token in tokens]
            # Truncate or pad to 512 tokens
            if len(input_ids) > 512:
                input_ids = input_ids[:512]
            else:
                input_ids += [self.tokenizer.pad_token_id] * (512 - len(input_ids))
            # Convert to tensor
            input_tensor = torch.tensor([input_ids])
            
            # Inference
            if self.training_mode:
                outputs = self.text_encoder(input_tensor)
            else:
                with torch.no_grad():
                    outputs = self.text_encoder(input_tensor)
            
            return outputs.last_hidden_state.mean(dim=1).detach().numpy()
        except Exception as e:
            error_handler.handle_error(e, "NeuralEmbeddingSpace", "Text encoding failed")
            return np.zeros((1, self.representation_dim))  # Return default vector with correct dimension
            
    def train_step(self, text, target_embedding):
        """Perform a training step"""
        if not self.training_mode:
            raise RuntimeError("Training mode must be enabled to train")
            
        try:
            # Tokenize text
            tokens = self.tokenizer.tokenize(text)
            # Convert tokens to IDs
            input_ids = [self.tokenizer.vocab.get(token, self.tokenizer.unk_token_id) for token in tokens]
            # Truncate or pad to 512 tokens
            if len(input_ids) > 512:
                input_ids = input_ids[:512]
            else:
                input_ids += [self.tokenizer.pad_token_id] * (512 - len(input_ids))
            # Convert to tensor
            input_tensor = torch.tensor([input_ids])
            
            # Forward pass
            output = self.text_encoder(input_tensor)
            predicted_embedding = output.last_hidden_state.mean(dim=1)
            
            # Calculate loss
            target_tensor = torch.tensor(target_embedding).unsqueeze(0)
            loss = self.criterion(predicted_embedding, target_tensor)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
        except Exception as e:
            error_handler.handle_error(e, "NeuralEmbeddingSpace", "Training step failed")
            return float('inf')
    
    def _encode_image(self, image_data):
        """Encode image data using real vision model"""
        try:
            # Implement real image encoder
            if isinstance(image_data, np.ndarray):
                # If input is numpy array, process directly
                image_tensor = torch.tensor(image_data).float()
            else:
                # Otherwise assume file path or PIL image
                import PIL.Image
                if isinstance(image_data, str):
                    image = PIL.Image.open(image_data)
                else:
                    image = image_data
                
                # Convert to tensor and preprocess
                image_tensor = torch.tensor(np.array(image)).float()
            
            # Use simple CNN encoder
            if not hasattr(self, 'image_encoder'):
                self.image_encoder = self._create_image_encoder()
            
            # Ensure image tensor has correct shape (C, H, W)
            if len(image_tensor.shape) == 2:  # (H, W)
                image_tensor = image_tensor.unsqueeze(0)  # (1, H, W)
            elif len(image_tensor.shape) == 3 and image_tensor.shape[0] != 3:  # (H, W, C)
                image_tensor = image_tensor.permute(2, 0, 1)  # (C, H, W)
            
            # Resize to standard size
            if image_tensor.shape[1] != 224 or image_tensor.shape[2] != 224:
                image_tensor = torch.nn.functional.interpolate(
                    image_tensor.unsqueeze(0), size=(224, 224), mode='bilinear'
                ).squeeze(0)
            
            # Encode
            with torch.no_grad():
                embedding = self.image_encoder(image_tensor.unsqueeze(0))
            
            return embedding.detach().numpy()
            
        except Exception as e:
            error_handler.handle_error(e, "NeuralEmbeddingSpace", "Image encoding failed")
            return np.zeros((1, self.representation_dim))  # Use dynamic dimension
    
    def _create_image_encoder(self):
        """Create real image encoder"""
        class SimpleImageEncoder(nn.Module):
            def __init__(self, input_channels=3, output_dim=768):
                super(SimpleImageEncoder, self).__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.fc = nn.Linear(512, output_dim)
                
            def forward(self, x):
                x = self.conv_layers(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        return SimpleImageEncoder(output_dim=self.representation_dim)
    
    def _encode_audio(self, audio_data):
        """Encode audio data using real audio model"""
        try:
            # Implement real audio encoder
            if isinstance(audio_data, np.ndarray):
                # If input is numpy array, process directly
                audio_tensor = torch.tensor(audio_data).float()
            else:
                # Otherwise assume file path
                import librosa
                audio, sr = librosa.load(audio_data, sr=16000)
                audio_tensor = torch.tensor(audio).float()
            
            # Use simple audio encoder
            if not hasattr(self, 'audio_encoder'):
                self.audio_encoder = self._create_audio_encoder()
            
            # Preprocess audio data
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # (1, samples)
            
            # Extract mel spectrogram
            mel_spec = self._extract_mel_spectrogram(audio_tensor)
            
            # Encode
            with torch.no_grad():
                embedding = self.audio_encoder(mel_spec.unsqueeze(1))  # Add channel dimension
            
            return embedding.detach().numpy()
            
        except Exception as e:
            error_handler.handle_error(e, "NeuralEmbeddingSpace", "Audio encoding failed")
            return np.zeros((1, self.representation_dim))  # Use dynamic dimension
    
    def _create_audio_encoder(self):
        """Create real audio encoder"""
        class SimpleAudioEncoder(nn.Module):
            def __init__(self, input_dim=128, output_dim=768):
                super(SimpleAudioEncoder, self).__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.fc = nn.Linear(128, output_dim)
                
            def forward(self, x):
                x = self.conv_layers(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        return SimpleAudioEncoder(output_dim=self.representation_dim)
    
    def _extract_mel_spectrogram(self, audio_tensor, n_mels=128, n_fft=2048, hop_length=512):
        """Extract mel spectrogram"""
        import librosa
        audio_np = audio_tensor.numpy()
        
        # Calculate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_np[0], sr=16000, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return torch.tensor(log_mel_spec).float()
    
    def encode(self, data, data_type):
        """Encode any type of data into unified representation"""
        if data_type == 'text':
            return self._encode_text(data)
        elif data_type == 'image':
            return self._encode_image(data)
        elif data_type == 'audio':
            return self._encode_audio(data)
        else:
            error_handler.log_warning(f"Unsupported data type: {data_type}", "NeuralEmbeddingSpace")
            return np.zeros((1, 768))
