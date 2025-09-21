"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

"""
Language Model Training Program

Function: Train emotion-aware language model
"""

import os
import json
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from datetime import datetime

# Import from-scratch training framework
from core.training.from_scratch_training import AGIFromScratchModel, FromScratchDataset, FromScratchTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageDataset(Dataset):
    """Language model training dataset"""
    
    def __init__(self, data_dir, max_length=512):
        self.data = []
        self.max_length = max_length
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        
        # Load English training data
        file_path = os.path.join(data_dir, "language_en.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        
        logger.info(f"Loaded language training data: {len(self.data)} samples")
    
    def _build_vocab(self):
        """Build basic vocabulary for from-scratch training"""
        # Basic vocabulary including common words and special tokens
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        
        # Add common words
        common_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'as', 'was', 'on', 'are', 'this', 'be', 'by', 'have', 'from']
        for i, word in enumerate(common_words, start=len(vocab)):
            vocab[word] = i
        
        return vocab
    
    def _text_to_ids(self, text):
        """Convert text to token IDs using simple vocabulary"""
        words = text.lower().split()
        ids = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        # Truncate or pad to max_length
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        else:
            ids = ids + [self.vocab['<PAD>']] * (self.max_length - len(ids))
        
        return torch.tensor(ids, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Build training sample
        text = item['text']
        emotion_label = item.get('emotion_label', 'neutral')
        
        # Convert text to token IDs
        input_ids = self._text_to_ids(text)
        
        # Emotion label mapping
        emotion_mapping = {
            'happy': 0, 'sad': 1, 'angry': 2, 'surprised': 3,
            'fear': 4, 'neutral': 5, 'excited': 6, 'calm': 7
        }
        
        return {
            'input_ids': input_ids,
            'labels': input_ids,  # For language modeling, labels are same as input
            'emotion_label': torch.tensor(emotion_mapping.get(emotion_label, 5))
        }


class LanguageModelTrainer:
    """Language model trainer"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize from-scratch model for emotion classification
        self.model = AGIFromScratchModel(
            input_size=512,  # Input size based on sequence length
            hidden_sizes=[256, 128],  # Hidden layers
            output_size=8,  # Output for 8 emotions
            model_type="emotion"
        ).to(self.device)
        
        # Loss function for emotion classification
        self.emotion_loss = nn.CrossEntropyLoss()
    
    
    def load_data(self, data_dir):
        """Load language data"""
        dataset = LanguageDataset(data_dir)
        self.vocab = dataset.vocab  # Store vocab for saving
        return dataset
    
    def create_data_loader(self, dataset, batch_size=8, shuffle=True):
        """Create data loader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4
        )
    
    
    def train_epoch(self, train_loader, optimizer):
        """Single epoch training"""
        self.model.train()
        
        total_loss = 0
        total_accuracy = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            emotion_labels = batch['emotion_label'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass - use input_ids as features for emotion classification
            # Flatten the input sequence for the fully connected network
            batch_size = input_ids.size(0)
            features = input_ids.view(batch_size, -1).float()  # Flatten to [batch_size, 512]
            
            emotion_logits = self.model(features)
            loss = self.emotion_loss(emotion_logits, emotion_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            emotion_preds = emotion_logits.argmax(dim=1)
            total_accuracy += (emotion_preds == emotion_labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader.dataset)
        
        return avg_loss, avg_accuracy
    
    
    def evaluate(self, test_loader):
        """Model evaluation"""
        self.model.eval()
        
        total_loss = 0
        total_accuracy = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                emotion_labels = batch['emotion_label'].to(self.device)
                
                # Forward pass - use input_ids as features for emotion classification
                batch_size = input_ids.size(0)
                features = input_ids.view(batch_size, -1).float()  # Flatten to [batch_size, 512]
                
                emotion_logits = self.model(features)
                loss = self.emotion_loss(emotion_logits, emotion_labels)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                emotion_preds = emotion_logits.argmax(dim=1)
                total_accuracy += (emotion_preds == emotion_labels).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        avg_accuracy = total_accuracy / len(test_loader.dataset)
        
        return avg_loss, avg_accuracy
    
    
    def save_model(self, path):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab if hasattr(self, 'vocab') else None
        }, path)
        logger.info(f"Language model saved to {path}")
    
    
    def full_training(self, data_dir, epochs=10):
        """Full training pipeline"""
        # Load data
        dataset = self.load_data(data_dir)
        
        # Split train and test sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        # Create data loaders
        train_loader = self.create_data_loader(train_dataset, batch_size=self.config['batch_size'])
        test_loader = self.create_data_loader(test_dataset, batch_size=self.config['batch_size'])
        
        # Set up optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        
        # Training loop
        best_accuracy = 0
        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_epoch(train_loader, optimizer)
            test_loss, test_accuracy = self.evaluate(test_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}"
            )
            
            # Save best model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                self.save_model(self.config['model_save_path'])
        
        logger.info("Language model training completed")

# Example configuration
if __name__ == "__main__":
    config = {
        'batch_size': 4,
        'learning_rate': 5e-5,
        'model_save_path': 'models/language_model.pth'
    }
    
    # Language data directory
    language_data_dir = 'data/language/'
    
    # Create data directory if it doesn't exist
    os.makedirs(language_data_dir, exist_ok=True)
    
    trainer = LanguageModelTrainer(config)
    trainer.full_training(language_data_dir, epochs=15)
