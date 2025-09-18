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
基础模型类 - 所有模型的基类
Base Model Class - Base class for all models

提供通用接口和功能，确保所有模型的一致性
Provides common interfaces and functionality to ensure consistency across all models
"""
"""
大语言模型训练程序
Language Model Training Program

功能：训练多语言情感感知语言模型
Function: Train multilingual emotion-aware language model
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


"""
LanguageDataset类 - 中文类描述
LanguageDataset Class - English class description
"""
class LanguageDataset(Dataset):
    """语言模型训练数据集 | Language model training dataset"""
    
    
"""
__init__函数 - 中文函数描述
__init__ Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def __init__(self, data_dir, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载多语言训练数据
        languages = ['zh', 'en', 'de', 'ja', 'ru']
        
        for lang in languages:
            file_path = os.path.join(data_dir, f"language_{lang}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    lang_data = json.load(f)
                    self.data.extend(lang_data)
        
        logger.info(f"加载语言模型训练数据: {len(self.data)} 条样本 | "
                    f"Loaded language training data: {len(self.data)} samples")
    
    
"""
__len__函数 - 中文函数描述
__len__ Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def __len__(self):
        return len(self.data)
    
    
"""
__getitem__函数 - 中文函数描述
__getitem__ Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建训练样本
        text = item['text']
        emotion_label = item.get('emotion_label', 'neutral')
        language = item.get('language', 'zh')
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 情感标签映射
        emotion_mapping = {
            'happy': 0, 'sad': 1, 'angry': 2, 'surprised': 3,
            'fear': 4, 'neutral': 5, 'excited': 6, 'calm': 7
        }
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
            'emotion_label': torch.tensor(emotion_mapping.get(emotion_label, 5)),
            'language': language
        }


"""
LanguageModelTrainer类 - 中文类描述
LanguageModelTrainer Class - English class description
"""
class LanguageModelTrainer:
    """语言模型训练器 | Language model trainer"""
    
    
"""
__init__函数 - 中文函数描述
__init__ Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型和tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            'gpt2-medium'
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 情感分类头
        self.emotion_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 8)  # 8种情感
        ).to(self.device)
        
        # 损失函数
        self.lm_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.emotion_loss = nn.CrossEntropyLoss()
    
    
"""
load_data函数 - 中文函数描述
load_data Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def load_data(self, data_dir):
        """加载语言数据 | Load language data"""
        dataset = LanguageDataset(data_dir, self.tokenizer)
        return dataset
    
    
"""
create_data_loader函数 - 中文函数描述
create_data_loader Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def create_data_loader(self, dataset, batch_size=8, shuffle=True):
        """创建数据加载器 | Create data loader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4
        )
    
    
"""
train_epoch函数 - 中文函数描述
train_epoch Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def train_epoch(self, train_loader, optimizer):
        """单轮训练 | Single epoch training"""
        self.model.train()
        self.emotion_classifier.train()
        
        total_lm_loss = 0
        total_emotion_loss = 0
        
        for batch in tqdm(train_loader, desc="训练中 | Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            emotion_labels = batch['emotion_label'].to(self.device)
            
            optimizer.zero_grad()
            
            # 语言模型前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            lm_loss = outputs.loss
            
            # 情感分析
            hidden_states = outputs.hidden_states[-1]  # 最后一层隐藏状态
            cls_embeddings = hidden_states[:, 0, :]  # CLS token嵌入
            emotion_logits = self.emotion_classifier(cls_embeddings)
            emotion_loss = self.emotion_loss(emotion_logits, emotion_labels)
            
            # 总损失
            total_loss = lm_loss + 0.3 * emotion_loss  # 加权损失
            
            total_loss.backward()
            optimizer.step()
            
            total_lm_loss += lm_loss.item()
            total_emotion_loss += emotion_loss.item()
        
        return total_lm_loss / len(train_loader), total_emotion_loss / len(train_loader)
    
    
"""
evaluate函数 - 中文函数描述
evaluate Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def evaluate(self, test_loader):
        """模型评估 | Model evaluation"""
        self.model.eval()
        self.emotion_classifier.eval()
        
        total_lm_loss = 0
        total_emotion_loss = 0
        emotion_accuracy = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估中 | Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                emotion_labels = batch['emotion_label'].to(self.device)
                
                # 语言模型前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                lm_loss = outputs.loss
                
                # 情感分析
                hidden_states = outputs.hidden_states[-1]
                cls_embeddings = hidden_states[:, 0, :]
                emotion_logits = self.emotion_classifier(cls_embeddings)
                emotion_loss = self.emotion_loss(emotion_logits, emotion_labels)
                
                # 计算情感准确率
                emotion_preds = emotion_logits.argmax(dim=1)
                emotion_accuracy += (emotion_preds == emotion_labels).sum().item()
                
                total_lm_loss += lm_loss.item()
                total_emotion_loss += emotion_loss.item()
        
        avg_lm_loss = total_lm_loss / len(test_loader)
        avg_emotion_loss = total_emotion_loss / len(test_loader)
        emotion_acc = emotion_accuracy / len(test_loader.dataset)
        
        return avg_lm_loss, avg_emotion_loss, emotion_acc
    
    
"""
save_model函数 - 中文函数描述
save_model Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def save_model(self, path):
        """保存模型 | Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'emotion_classifier_state_dict': self.emotion_classifier.state_dict(),
            'tokenizer': self.tokenizer
        }, path)
        logger.info(f"语言模型已保存至 {path} | Language model saved to {path}")
    
    
"""
full_training函数 - 中文函数描述
full_training Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def full_training(self, data_dir, epochs=10):
        """完整训练流程 | Full training pipeline"""
        # 加载数据
        dataset = self.load_data(data_dir)
        
        # 划分训练集和测试集
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        # 创建数据加载器
        train_loader = self.create_data_loader(train_dataset, batch_size=self.config['batch_size'])
        test_loader = self.create_data_loader(test_dataset, batch_size=self.config['batch_size'])
        
        # 设置优化器
        optimizer = optim.AdamW([
            {'params': self.model.parameters()},
            {'params': self.emotion_classifier.parameters()}
        ], lr=self.config['learning_rate'])
        
        # 训练循环
        best_emotion_acc = 0
        for epoch in range(epochs):
            train_lm_loss, train_emotion_loss = self.train_epoch(train_loader, optimizer)
            test_lm_loss, test_emotion_loss, test_emotion_acc = self.evaluate(test_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"训练LM损失: {train_lm_loss:.4f} | 训练情感损失: {train_emotion_loss:.4f} | "
                f"测试LM损失: {test_lm_loss:.4f} | 测试情感损失: {test_emotion_loss:.4f} | "
                f"情感准确率: {test_emotion_acc:.4f} | "
                f"Train LM Loss: {train_lm_loss:.4f} | Train Emotion Loss: {train_emotion_loss:.4f} | "
                f"Test LM Loss: {test_lm_loss:.4f} | Test Emotion Loss: {test_emotion_loss:.4f} | "
                f"Emotion Accuracy: {test_emotion_acc:.4f}"
            )
            
            # 保存最佳模型
            if test_emotion_acc > best_emotion_acc:
                best_emotion_acc = test_emotion_acc
                self.save_model(self.config['model_save_path'])
        
        logger.info("语言模型训练完成 | Language model training completed")

# 配置示例 | Example configuration
if __name__ == "__main__":
    config = {
        'batch_size': 4,
        'learning_rate': 5e-5,
        'model_save_path': 'models/language_model.pth'
    }
    
    # 语言数据目录
    language_data_dir = 'data/language/'
    
    # 创建数据目录（如果不存在）
    os.makedirs(language_data_dir, exist_ok=True)
    
    trainer = LanguageModelTrainer(config)
    trainer.full_training(language_data_dir, epochs=15)
