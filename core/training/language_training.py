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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageDataset(Dataset):
    """Language model training dataset"""
    
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load English training data
        file_path = os.path.join(data_dir, "language_en.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        
        logger.info(f"Loaded language training data: {len(self.data)} samples")
    
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Build training sample
        text = item['text']
        emotion_label = item.get('emotion_label', 'neutral')
        
        # Encode text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Emotion label mapping
        emotion_mapping = {
            'happy': 0, 'sad': 1, 'angry': 2, 'surprised': 3,
            'fear': 4, 'neutral': 5, 'excited': 6, 'calm': 7
        }
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
            'emotion_label': torch.tensor(emotion_mapping.get(emotion_label, 5))
        }


class LanguageModelTrainer:
    """Language model trainer"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            'gpt2-medium'
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Emotion classification head
        self.emotion_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 8)  # 8 emotions
        ).to(self.device)
        
        # Loss functions
        self.lm_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.emotion_loss = nn.CrossEntropyLoss()
    
    
    def load_data(self, data_dir):
        """Load language data"""
        dataset = LanguageDataset(data_dir, self.tokenizer)
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
        self.emotion_classifier.train()
        
        total_lm_loss = 0
        total_emotion_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            emotion_labels = batch['emotion_label'].to(self.device)
            
            optimizer.zero_grad()
            
            # Language model forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            lm_loss = outputs.loss
            
            # Emotion analysis
            hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
            cls_embeddings = hidden_states[:, 0, :]  # CLS token embeddings
            emotion_logits = self.emotion_classifier(cls_embeddings)
            emotion_loss = self.emotion_loss(emotion_logits, emotion_labels)
            
            # Total loss
            total_loss = lm_loss + 0.3 * emotion_loss  # Weighted loss
            
            total_loss.backward()
            optimizer.step()
            
            total_lm_loss += lm_loss.item()
            total_emotion_loss += emotion_loss.item()
        
        return total_lm_loss / len(train_loader), total_emotion_loss / len(train_loader)
    
    
    def evaluate(self, test_loader):
        """Model evaluation"""
        self.model.eval()
        self.emotion_classifier.eval()
        
        total_lm_loss = 0
        total_emotion_loss = 0
        emotion_accuracy = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                emotion_labels = batch['emotion_label'].to(self.device)
                
                # Language model forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                lm_loss = outputs.loss
                
                # Emotion analysis
                hidden_states = outputs.hidden_states[-1]
                cls_embeddings = hidden_states[:, 0, :]
                emotion_logits = self.emotion_classifier(cls_embeddings)
                emotion_loss = self.emotion_loss(emotion_logits, emotion_labels)
                
                # Calculate emotion accuracy
                emotion_preds = emotion_logits.argmax(dim=1)
                emotion_accuracy += (emotion_preds == emotion_labels).sum().item()
                
                total_lm_loss += lm_loss.item()
                total_emotion_loss += emotion_loss.item()
        
        avg_lm_loss = total_lm_loss / len(test_loader)
        avg_emotion_loss = total_emotion_loss / len(test_loader)
        emotion_acc = emotion_accuracy / len(test_loader.dataset)
        
        return avg_lm_loss, avg_emotion_loss, emotion_acc
    
    
    def save_model(self, path):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'emotion_classifier_state_dict': self.emotion_classifier.state_dict(),
            'tokenizer': self.tokenizer
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
        optimizer = optim.AdamW([
            {'params': self.model.parameters()},
            {'params': self.emotion_classifier.parameters()}
        ], lr=self.config['learning_rate'])
        
        # Training loop
        best_emotion_acc = 0
        for epoch in range(epochs):
            train_lm_loss, train_emotion_loss = self.train_epoch(train_loader, optimizer)
            test_lm_loss, test_emotion_loss, test_emotion_acc = self.evaluate(test_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train LM Loss: {train_lm_loss:.4f} | Train Emotion Loss: {train_emotion_loss:.4f} | "
                f"Test LM Loss: {test_lm_loss:.4f} | Test Emotion Loss: {test_emotion_loss:.4f} | "
                f"Emotion Accuracy: {test_emotion_acc:.4f}"
            )
            
            # Save best model
            if test_emotion_acc > best_emotion_acc:
                best_emotion_acc = test_emotion_acc
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
