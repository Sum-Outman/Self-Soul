"""
Advanced Emotion Analysis Module - AGI Compliant Version
Supports from-scratch training without external pre-trained models
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
import json
import os


class TextEmbeddingModel(nn.Module):
    """Text embedding model trained from scratch for emotion analysis"""
    
    def __init__(self, vocab_size: int = 30000, embedding_dim: int = 256, hidden_dim: int = 512):
        super(TextEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeddings)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        return context_vector


class AdvancedEmotionAnalyzer:
    """Advanced Emotion Analyzer with from-scratch training capability"""
    
    def __init__(self, from_scratch: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.from_scratch = from_scratch
        
        # Vocabulary and tokenization
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        
        # Text model
        self.text_model = TextEmbeddingModel(
            vocab_size=self.vocab_size,
            embedding_dim=256,
            hidden_dim=512
        ).to(self.device)
        
        # Emotion classifier
        self.emotion_classifier = nn.Sequential(
            nn.Linear(1024, 512),  # 512 * 2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6)  # 6 basic emotions
        ).to(self.device)
        
        self.emotion_labels = ['happy', 'sad', 'angry', 'fearful', 'surprised', 'neutral']
        self.training_data = []
        self.is_trained = False
        
    def _build_vocab(self) -> List[str]:
        """Build basic vocabulary from scratch"""
        base_vocab = [
            # Basic emotional words
            'happy', 'sad', 'angry', 'fear', 'surprise', 'neutral',
            'love', 'hate', 'joy', 'pain', 'excited', 'bored',
            'good', 'bad', 'great', 'terrible', 'wonderful', 'awful',
            # Common words
            'the', 'and', 'is', 'are', 'was', 'were', 'this', 'that',
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'very', 'really', 'quite', 'extremely', 'somewhat'
        ]
        return base_vocab + [f'<unk_{i}>' for i in range(1000)]
    
    def tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text using from-scratch vocabulary"""
        words = text.lower().split()
        token_ids = []
        for word in words:
            if word in self.word_to_idx:
                token_ids.append(self.word_to_idx[word])
            else:
                # Use unknown token
                token_ids.append(self.word_to_idx.get('<unk_0>', 0))
        
        if not token_ids:
            token_ids = [0]  # Default to first token
        
        return torch.tensor([token_ids], device=self.device)
    
    def train_from_scratch(self, training_data: List[Dict[str, Any]], epochs: int = 10):
        """Train emotion analyzer from scratch"""
        if not training_data:
            raise ValueError("Training data is required for from-scratch training")
        
        optimizer = torch.optim.Adam(
            list(self.text_model.parameters()) + list(self.emotion_classifier.parameters()),
            lr=0.001
        )
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for data_point in training_data:
                text = data_point.get('text', '')
                emotion_label = data_point.get('emotion', 'neutral')
                
                if not text:
                    continue
                
                # Tokenize and process
                token_ids = self.tokenize_text(text)
                embeddings = self.text_model(token_ids)
                predictions = self.emotion_classifier(embeddings)
                
                # Convert emotion label to index
                label_idx = self.emotion_labels.index(emotion_label) if emotion_label in self.emotion_labels else 5
                target = torch.tensor([label_idx], device=self.device)
                
                # Calculate loss and update
                loss = criterion(predictions, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(training_data):.4f}")
        
        self.is_trained = True
        self.training_data = training_data
    
    def analyze_text_emotion(self, text: str) -> Dict[str, float]:
        """Analyze text emotion using from-scratch model"""
        if not self.is_trained and self.from_scratch:
            # Return neutral if not trained
            return {'neutral': 1.0}
        
        if not text.strip():
            return {'neutral': 1.0}
        
        # Tokenize and process
        token_ids = self.tokenize_text(text)
        with torch.no_grad():
            embeddings = self.text_model(token_ids)
            emotion_logits = self.emotion_classifier(embeddings)
            emotion_probs = torch.softmax(emotion_logits, dim=1)
        
        # Convert to probability dictionary
        emotion_dict = {}
        for i, label in enumerate(self.emotion_labels):
            emotion_dict[label] = emotion_probs[0][i].item()
        
        return emotion_dict
    
    def analyze_audio_emotion(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Analyze audio emotion - placeholder for AGI integration"""
        # This would be implemented with from-scratch audio processing
        # For now, return neutral as base state
        return {'neutral': 1.0}
    
    def analyze_visual_emotion(self, image_data: np.ndarray) -> Dict[str, float]:
        """Analyze visual emotion - placeholder for AGI integration"""
        # This would be implemented with from-scratch vision processing
        # For now, return neutral as base state
        return {'neutral': 1.0}
    
    def fuse_multimodal_emotions(self, text_emotion: Dict[str, float], 
                                audio_emotion: Dict[str, float], 
                                visual_emotion: Dict[str, float]) -> Dict[str, float]:
        """Fuse multimodal emotions with adaptive weighting"""
        # Adaptive weighting based on confidence
        text_conf = max(text_emotion.values()) if text_emotion else 0
        audio_conf = max(audio_emotion.values()) if audio_emotion else 0
        visual_conf = max(visual_emotion.values()) if visual_emotion else 0
        
        total_conf = text_conf + audio_conf + visual_conf
        if total_conf == 0:
            weights = {'text': 0.33, 'audio': 0.33, 'visual': 0.34}
        else:
            weights = {
                'text': text_conf / total_conf,
                'audio': audio_conf / total_conf,
                'visual': visual_conf / total_conf
            }
        
        # Fuse emotions
        fused_emotion = {}
        for emotion in self.emotion_labels:
            fused_value = (text_emotion.get(emotion, 0) * weights['text'] +
                          audio_emotion.get(emotion, 0) * weights['audio'] +
                          visual_emotion.get(emotion, 0) * weights['visual'])
            fused_emotion[emotion] = fused_value
        
        # Normalize
        total = sum(fused_emotion.values())
        if total > 0:
            for emotion in fused_emotion:
                fused_emotion[emotion] /= total
        
        return fused_emotion
    
    def save_model(self, path: str):
        """Save model state for from-scratch training"""
        state = {
            'text_model_state': self.text_model.state_dict(),
            'classifier_state': self.emotion_classifier.state_dict(),
            'vocab': self.vocab,
            'is_trained': self.is_trained,
            'training_data': self.training_data
        }
        torch.save(state, path)
    
    def load_model(self, path: str):
        """Load model state for from-scratch training"""
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.text_model.load_state_dict(state['text_model_state'])
            self.emotion_classifier.load_state_dict(state['classifier_state'])
            self.vocab = state['vocab']
            self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
            self.is_trained = state['is_trained']
            self.training_data = state.get('training_data', [])
