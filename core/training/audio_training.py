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
音频处理模型训练程序
Audio Processing Model Training Program

功能：训练多模态音频处理模型，支持语音识别、语调分析、音乐识别、噪音识别和音频合成
Function: Train multimodal audio processing model supporting speech recognition, tone analysis, music recognition, noise recognition, and audio synthesis
"""

import os
import json
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from tqdm import tqdm
import logging
import librosa
import soundfile as sf
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
AudioDataset类 - 中文类描述
AudioDataset Class - English class description
"""
class AudioDataset(Dataset):
    """音频模型训练数据集 | Audio model training dataset"""
    
    
"""
__init__函数 - 中文函数描述
__init__ Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def __init__(self, data_dir, feature_extractor, max_length=16000 * 5):  # 5秒音频
        self.data = []
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        
        # 加载多模态音频数据
        audio_types = ['speech', 'music', 'noise', 'tone', 'effects']
        
        for audio_type in audio_types:
            file_path = os.path.join(data_dir, f"audio_{audio_type}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    audio_data = json.load(f)
                    self.data.extend(audio_data)
        
        logger.info(f"加载音频模型训练数据: {len(self.data)} 条样本 | "
                    f"Loaded audio training data: {len(self.data)} samples")
    
    
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
        
        # 加载音频文件
        audio_path = item['audio_path']
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # 截断或填充到固定长度
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            else:
                audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
            
            # 提取特征
            inputs = self.feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length
            )
            
            # 标签处理
            labels = {
                'speech_text': item.get('transcript', ''),
                'emotion_label': item.get('emotion', 'neutral'),
                'audio_type': item.get('audio_type', 'speech'),
                'tone_features': torch.tensor(item.get('tone_features', [0.5, 0.5, 0.5])),
                'music_genre': item.get('music_genre', 'unknown'),
                'noise_type': item.get('noise_type', 'none')
            }
            
            return {
                'input_values': inputs.input_values.squeeze(),
                'attention_mask': inputs.attention_mask.squeeze(),
                'labels': labels
            }
            
        except Exception as e:
            logger.error(f"音频加载错误 {audio_path}: {str(e)}")
            # 返回空数据
            return {
                'input_values': torch.zeros(self.max_length),
                'attention_mask': torch.ones(self.max_length),
                'labels': {
                    'speech_text': '',
                    'emotion_label': 'neutral',
                    'audio_type': 'speech',
                    'tone_features': torch.tensor([0.5, 0.5, 0.5]),
                    'music_genre': 'unknown',
                    'noise_type': 'none'
                }
            }


"""
AudioModelTrainer类 - 中文类描述
AudioModelTrainer Class - English class description
"""
class AudioModelTrainer:
    """音频模型训练器 | Audio model trainer"""
    
    
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
        
        # 初始化语音识别模型
        self.speech_model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base-960h"
        ).to(self.device)
        
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        
        # 初始化音频分类模型
        self.classification_model = AutoModelForAudioClassification.from_pretrained(
            "superb/hubert-base-superb-ks",
            num_labels=10  # 10种音频类型
        ).to(self.device)
        
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "superb/hubert-base-superb-ks"
        )
        
        # 情感分析头
        self.emotion_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 8)  # 8种情感
        ).to(self.device)
        
        # 音调分析头
        self.tone_analyzer = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 音调特征: pitch, intensity, timbre
        ).to(self.device)
        
        # 损失函数
        self.ctc_loss = nn.CTCLoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.emotion_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
    
    
"""
load_data函数 - 中文函数描述
load_data Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def load_data(self, data_dir):
        """加载音频数据 | Load audio data"""
        dataset = AudioDataset(data_dir, self.feature_extractor)
        return dataset
    
    
"""
create_data_loader函数 - 中文函数描述
create_data_loader Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def create_data_loader(self, dataset, batch_size=4, shuffle=True):
        """创建数据加载器 | Create data loader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2  # 音频处理需要较少workers
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
        self.speech_model.train()
        self.classification_model.train()
        self.emotion_classifier.train()
        self.tone_analyzer.train()
        
        total_ctc_loss = 0
        total_class_loss = 0
        total_emotion_loss = 0
        total_tone_loss = 0
        
        for batch in tqdm(train_loader, desc="训练中 | Training"):
            input_values = batch['input_values'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels']
            
            optimizer.zero_grad()
            
            # 语音识别前向传播
            speech_outputs = self.speech_model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=self.processor(labels['speech_text'], return_tensors="pt").input_ids.to(self.device)
            )
            ctc_loss = speech_outputs.loss
            
            # 音频分类
            classification_outputs = self.classification_model(
                input_values=input_values,
                attention_mask=attention_mask
            )
            
            # 情感分析
            hidden_states = classification_outputs.hidden_states[-1]
            cls_embeddings = hidden_states[:, 0, :]
            emotion_logits = self.emotion_classifier(cls_embeddings)
            
            # 音调分析
            tone_features = self.tone_analyzer(cls_embeddings)
            
            # 计算各种损失
            emotion_labels = torch.tensor([self._emotion_to_idx(e) for e in labels['emotion_label']]).to(self.device)
            emotion_loss = self.emotion_loss(emotion_logits, emotion_labels)
            
            tone_targets = torch.stack([l['tone_features'] for l in labels]).to(self.device)
            tone_loss = self.regression_loss(tone_features, tone_targets)
            
            # 总损失
            total_loss = ctc_loss + 0.5 * emotion_loss + 0.3 * tone_loss
            
            total_loss.backward()
            optimizer.step()
            
            total_ctc_loss += ctc_loss.item()
            total_emotion_loss += emotion_loss.item()
            total_tone_loss += tone_loss.item()
        
        return (total_ctc_loss / len(train_loader), 
                total_emotion_loss / len(train_loader),
                total_tone_loss / len(train_loader))
    
    
"""
_emotion_to_idx函数 - 中文函数描述
_emotion_to_idx Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _emotion_to_idx(self, emotion):
        """情感标签映射 | Emotion label mapping"""
        emotion_mapping = {
            'happy': 0, 'sad': 1, 'angry': 2, 'surprised': 3,
            'fear': 4, 'neutral': 5, 'excited': 6, 'calm': 7
        }
        return emotion_mapping.get(emotion, 5)
    
    
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
        self.speech_model.eval()
        self.classification_model.eval()
        self.emotion_classifier.eval()
        self.tone_analyzer.eval()
        
        total_ctc_loss = 0
        total_emotion_loss = 0
        total_tone_loss = 0
        emotion_accuracy = 0
        tone_accuracy = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估中 | Evaluating"):
                input_values = batch['input_values'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                # 语音识别
                speech_outputs = self.speech_model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    labels=self.processor(labels['speech_text'], return_tensors="pt").input_ids.to(self.device)
                )
                ctc_loss = speech_outputs.loss
                
                # 音频分类和情感分析
                classification_outputs = self.classification_model(
                    input_values=input_values,
                    attention_mask=attention_mask
                )
                
                hidden_states = classification_outputs.hidden_states[-1]
                cls_embeddings = hidden_states[:, 0, :]
                emotion_logits = self.emotion_classifier(cls_embeddings)
                tone_features = self.tone_analyzer(cls_embeddings)
                
                # 计算情感准确率
                emotion_labels = torch.tensor([self._emotion_to_idx(e) for e in labels['emotion_label']]).to(self.device)
                emotion_loss = self.emotion_loss(emotion_logits, emotion_labels)
                emotion_preds = emotion_logits.argmax(dim=1)
                emotion_accuracy += (emotion_preds == emotion_labels).sum().item()
                
                # 计算音调损失
                tone_targets = torch.stack([l['tone_features'] for l in labels]).to(self.device)
                tone_loss = self.regression_loss(tone_features, tone_targets)
                
                total_ctc_loss += ctc_loss.item()
                total_emotion_loss += emotion_loss.item()
                total_tone_loss += tone_loss.item()
        
        avg_ctc_loss = total_ctc_loss / len(test_loader)
        avg_emotion_loss = total_emotion_loss / len(test_loader)
        avg_tone_loss = total_tone_loss / len(test_loader)
        emotion_acc = emotion_accuracy / len(test_loader.dataset)
        
        return avg_ctc_loss, avg_emotion_loss, avg_tone_loss, emotion_acc
    
    
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
            'speech_model_state_dict': self.speech_model.state_dict(),
            'classification_model_state_dict': self.classification_model.state_dict(),
            'emotion_classifier_state_dict': self.emotion_classifier.state_dict(),
            'tone_analyzer_state_dict': self.tone_analyzer.state_dict(),
            'processor': self.processor,
            'feature_extractor': self.feature_extractor
        }, path)
        logger.info(f"音频模型已保存至 {path} | Audio model saved to {path}")
    
    
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
            {'params': self.speech_model.parameters()},
            {'params': self.classification_model.parameters()},
            {'params': self.emotion_classifier.parameters()},
            {'params': self.tone_analyzer.parameters()}
        ], lr=self.config['learning_rate'])
        
        # 训练循环
        best_emotion_acc = 0
        for epoch in range(epochs):
            train_ctc_loss, train_emotion_loss, train_tone_loss = self.train_epoch(train_loader, optimizer)
            test_ctc_loss, test_emotion_loss, test_tone_loss, test_emotion_acc = self.evaluate(test_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"训练CTC损失: {train_ctc_loss:.4f} | 训练情感损失: {train_emotion_loss:.4f} | 训练音调损失: {train_tone_loss:.4f} | "
                f"测试CTC损失: {test_ctc_loss:.4f} | 测试情感损失: {test_emotion_loss:.4f} | 测试音调损失: {test_tone_loss:.4f} | "
                f"情感准确率: {test_emotion_acc:.4f} | "
                f"Train CTC Loss: {train_ctc_loss:.4f} | Train Emotion Loss: {train_emotion_loss:.4f} | Train Tone Loss: {train_tone_loss:.4f} | "
                f"Test CTC Loss: {test_ctc_loss:.4f} | Test Emotion Loss: {test_emotion_loss:.4f} | Test Tone Loss: {test_tone_loss:.4f} | "
                f"Emotion Accuracy: {test_emotion_acc:.4f}"
            )
            
            # 保存最佳模型
            if test_emotion_acc > best_emotion_acc:
                best_emotion_acc = test_emotion_acc
                self.save_model(self.config['model_save_path'])
        
        logger.info("音频模型训练完成 | Audio model training completed")

# 配置示例 | Example configuration
if __name__ == "__main__":
    config = {
        'batch_size': 2,  # 音频处理需要较小批次
        'learning_rate': 3e-5,
        'model_save_path': 'models/audio_model.pth'
    }
    
    # 音频数据目录
    audio_data_dir = 'data/audio/'
    
    # 创建数据目录（如果不存在）
    os.makedirs(audio_data_dir, exist_ok=True)
    
    trainer = AudioModelTrainer(config)
    trainer.full_training(audio_data_dir, epochs=10)
