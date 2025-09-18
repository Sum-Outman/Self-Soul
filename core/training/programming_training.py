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
编程模型训练程序
Programming Model Training Program

功能：训练编程模型，具有辅助编程能力，可以根据主模型需求进行自主编程改进本地所有模型和环境的能力
Function: Train programming model with auxiliary programming capabilities, able to autonomously program and improve all local models and environments based on main model requirements
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
import re
import ast
import tokenize
from io import StringIO

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
ProgrammingDataset类 - 中文类描述
ProgrammingDataset Class - English class description
"""
class ProgrammingDataset(Dataset):
    """编程数据集 | Programming dataset"""
    
    
"""
__init__函数 - 中文函数描述
__init__ Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def __init__(self, data_dir, max_sequence_length=100):
        self.data = []
        self.max_sequence_length = max_sequence_length
        self.vocab = self._build_vocab()
        
        # 加载编程数据
        programming_types = ['code_generation', 'code_optimization', 'bug_fixing', 
                           'refactoring', 'documentation', 'model_integration']
        
        for programming_type in programming_types:
            file_path = os.path.join(data_dir, f"programming_{programming_type}.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        programming_data = json.load(f)
                        self.data.extend(programming_data)
                except Exception as e:
                    logger.error(f"编程数据加载错误 {file_path}: {str(e)}")
        
        logger.info(f"加载编程训练数据: {len(self.data)} 条样本 | "
                    f"Loaded programming training data: {len(self.data)} samples")
    
    
"""
_build_vocab函数 - 中文函数描述
_build_vocab Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _build_vocab(self):
        """构建编程词汇表 | Build programming vocabulary"""
        # 基础编程词汇
        base_vocab = {
            'def': 0, 'class': 1, 'import': 2, 'from': 3, 'return': 4,
            'if': 5, 'else': 6, 'elif': 7, 'for': 8, 'while': 9,
            'try': 10, 'except': 11, 'finally': 12, 'with': 13, 'as': 14,
            'pass': 15, 'break': 16, 'continue': 17, 'yield': 18, 'async': 19,
            'await': 20, 'global': 21, 'nonlocal': 22, 'lambda': 23, 'assert': 24,
            'raise': 25, 'del': 26, 'in': 27, 'is': 28, 'not': 29,
            'and': 30, 'or': 31, 'True': 32, 'False': 33, 'None': 34,
            'self': 35, 'super': 36, 'init': 37, 'str': 38, 'int': 39,
            'float': 40, 'list': 41, 'dict': 42, 'tuple': 43, 'set': 44,
            'bool': 45, 'range': 46, 'len': 47, 'print': 48, 'input': 49,
            'open': 50, 'close': 51, 'read': 52, 'write': 53, 'append': 54,
            'extend': 55, 'pop': 56, 'remove': 57, 'sort': 58, 'reverse': 59,
            'keys': 60, 'values': 61, 'items': 62, 'get': 63, 'update': 64,
            'join': 65, 'split': 66, 'strip': 67, 'replace': 68, 'find': 69,
            'startswith': 70, 'endswith': 71, 'format': 72, 'encode': 73, 'decode': 74,
            'math': 75, 'random': 76, 'time': 77, 'datetime': 78, 'os': 79,
            'sys': 80, 'json': 81, 're': 82, 'torch': 83, 'numpy': 84,
            'pandas': 85, 'matplotlib': 86, 'seaborn': 87, 'sklearn': 88, 'tensorflow': 89,
            'keras': 90, 'transformers': 91, 'PIL': 92, 'cv2': 93, 'requests': 94,
            'BeautifulSoup': 95, 'flask': 96, 'django': 97, 'fastapi': 98, 'uvicorn': 99,
            '<UNK>': 100, '<PAD>': 101, '<BOS>': 102, '<EOS>': 103
        }
        return base_vocab
    
    
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
        
        try:
            # 编程任务描述
            task_description = item.get('task_description', '')
            
            # 输入代码
            input_code = item.get('input_code', '')
            
            # 目标代码
            target_code = item.get('target_code', '')
            
            # 编程语言
            programming_language = item.get('language', 'python')
            
            # 代码复杂度
            complexity = item.get('complexity', 'simple')
            
            # 编码任务描述
            encoded_task = self._encode_text(task_description)
            
            # 编码输入代码
            encoded_input = self._encode_code(input_code)
            
            # 编码目标代码
            encoded_target = self._encode_code(target_code)
            
            # 截断或填充序列
            encoded_task = self._pad_sequence(encoded_task, self.max_sequence_length)
            encoded_input = self._pad_sequence(encoded_input, self.max_sequence_length)
            encoded_target = self._pad_sequence(encoded_target, self.max_sequence_length)
            
            return {
                'task_description': encoded_task,
                'input_code': encoded_input,
                'target_code': encoded_target,
                'language': programming_language,
                'complexity': complexity,
                'metadata': {
                    'task_type': item.get('task_type', 'code_generation'),
                    'quality_score': item.get('quality_score', 1.0),
                    'efficiency_score': item.get('efficiency_score', 1.0)
                }
            }
            
        except Exception as e:
            logger.error(f"编程数据处理错误: {str(e)}")
            # 返回空数据
            return {
                'task_description': [self.vocab['<PAD>']] * self.max_sequence_length,
                'input_code': [self.vocab['<PAD>']] * self.max_sequence_length,
                'target_code': [self.vocab['<PAD>']] * self.max_sequence_length,
                'language': 'python',
                'complexity': 'simple',
                'metadata': {
                    'task_type': 'code_generation',
                    'quality_score': 1.0,
                    'efficiency_score': 1.0
                }
            }
    
    
"""
_encode_text函数 - 中文函数描述
_encode_text Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _encode_text(self, text):
        """编码文本 | Encode text"""
        tokens = text.lower().split()
        encoded = []
        for token in tokens:
            if token in self.vocab:
                encoded.append(self.vocab[token])
            else:
                # 处理未知词汇
                if token.isidentifier():
                    # 标识符映射到特殊token
                    encoded.append(self.vocab['<UNK>'])
                elif token.isdigit():
                    # 数字统一处理
                    encoded.append(self.vocab['int'])
                else:
                    encoded.append(self.vocab['<UNK>'])
        return encoded
    
    
"""
_encode_code函数 - 中文函数描述
_encode_code Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _encode_code(self, code):
        """编码代码 | Encode code"""
        try:
            # 使用Python tokenizer解析代码
            tokens = []
            for tok in tokenize.generate_tokens(StringIO(code).readline):
                token_type, token_string, _, _, _ = tok
                if token_type == tokenize.NAME:
                    if token_string in self.vocab:
                        tokens.append(self.vocab[token_string])
                    else:
                        tokens.append(self.vocab['<UNK>'])
                elif token_type == tokenize.NUMBER:
                    tokens.append(self.vocab['int'])
                elif token_type == tokenize.STRING:
                    tokens.append(self.vocab['str'])
                elif token_type == tokenize.OP:
                    # 操作符映射
                    if token_string in ['+', '-', '*', '/', '%', '**', '//']:
                        tokens.append(self.vocab['math'])
                    elif token_string in ['=', '==', '!=', '<', '>', '<=', '>=']:
                        tokens.append(self.vocab['if'])
                    else:
                        tokens.append(self.vocab['<UNK>'])
            
            return tokens
        except:
            # 如果解析失败，使用简单的文本编码
            return self._encode_text(code)
    
    
"""
_pad_sequence函数 - 中文函数描述
_pad_sequence Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _pad_sequence(self, sequence, max_length):
        """填充序列 | Pad sequence"""
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence + [self.vocab['<PAD>']] * (max_length - len(sequence))


"""
ProgrammingModelTrainer类 - 中文类描述
ProgrammingModelTrainer Class - English class description
"""
class ProgrammingModelTrainer:
    """编程模型训练器 | Programming model trainer"""
    
    
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
        self.vocab_size = config['vocab_size']
        
        # 任务描述编码器
        self.task_encoder = nn.Sequential(
            nn.Embedding(self.vocab_size, config['embedding_dim']),
            nn.LSTM(input_size=config['embedding_dim'], 
                   hidden_size=256, 
                   num_layers=2, 
                   batch_first=True,
                   dropout=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(self.device)
        
        # 输入代码编码器
        self.code_encoder = nn.Sequential(
            nn.Embedding(self.vocab_size, config['embedding_dim']),
            nn.LSTM(input_size=config['embedding_dim'], 
                   hidden_size=256, 
                   num_layers=2, 
                   batch_first=True,
                   dropout=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(self.device)
        
        # 代码生成解码器
        self.code_decoder = nn.Sequential(
            nn.LSTM(input_size=256,  # 任务特征 + 代码特征
                   hidden_size=512, 
                   num_layers=3, 
                   batch_first=True,
                   dropout=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.vocab_size),
            nn.LogSoftmax(dim=2)
        ).to(self.device)
        
        # 代码质量评估网络
        self.quality_assessor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 质量评分: 语法正确性, 逻辑正确性, 效率
            nn.Sigmoid()
        ).to(self.device)
        
        # 代码优化网络
        self.optimization_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 优化建议: 重构, 性能提升
            nn.Sigmoid()
        ).to(self.device)
        
        # 损失函数
        self.generation_loss = nn.NLLLoss(ignore_index=self.vocab['<PAD>'])
        self.quality_loss = nn.MSELoss()
        self.optimization_loss = nn.BCELoss()
    
    
"""
load_data函数 - 中文函数描述
load_data Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def load_data(self, data_dir):
        """加载编程数据 | Load programming data"""
        dataset = ProgrammingDataset(data_dir)
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
            num_workers=2
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
        self.task_encoder.train()
        self.code_encoder.train()
        self.code_decoder.train()
        self.quality_assessor.train()
        self.optimization_net.train()
        
        total_generation_loss = 0
        total_quality_loss = 0
        total_optimization_loss = 0
        
        for batch in tqdm(train_loader, desc="训练中 | Training"):
            task_descriptions = batch['task_description'].to(self.device)
            input_codes = batch['input_code'].to(self.device)
            target_codes = batch['target_code'].to(self.device)
            
            optimizer.zero_grad()
            
            # 编码任务描述
            task_embeddings = self.task_encoder[0](task_descriptions)
            task_features, _ = self.task_encoder[1](task_embeddings)
            encoded_task = self.task_encoder[2:](task_features[:, -1, :])
            
            # 编码输入代码
            code_embeddings = self.code_encoder[0](input_codes)
            code_features, _ = self.code_encoder[1](code_embeddings)
            encoded_code = self.code_encoder[2:](code_features[:, -1, :])
            
            # 融合特征
            fused_features = torch.cat([encoded_task, encoded_code], dim=1)
            
            # 代码生成
            decoder_input = torch.cat([
                fused_features.unsqueeze(1), 
                torch.zeros(batch_size, self.config['max_sequence_length'] - 1, 256).to(self.device)
            ], dim=1)
            
            decoder_output, _ = self.code_decoder[0](decoder_input)
            generated_code = self.code_decoder[1:](decoder_output)
            
            # 代码质量评估
            quality_pred = self.quality_assessor(fused_features)
            
            # 代码优化建议
            optimization_pred = self.optimization_net(fused_features)
            
            total_loss = 0
            
            # 计算各种损失
            generation_loss = self.generation_loss(
                generated_code.view(-1, self.vocab_size),
                target_codes.view(-1)
            )
            
            quality_loss = self.quality_loss(
                quality_pred,
                torch.ones_like(quality_pred) * 0.8  # 简化目标质量分数
            )
            
            optimization_loss = self.optimization_loss(
                optimization_pred,
                torch.ones_like(optimization_pred) * 0.5  # 简化目标
            )
            
            # 总损失
            total_batch_loss = generation_loss + 0.3 * quality_loss + 0.2 * optimization_loss
            
            total_batch_loss.backward()
            optimizer.step()
            
            total_generation_loss += generation_loss.item()
            total_quality_loss += quality_loss.item()
            total_optimization_loss += optimization_loss.item()
        
        return (total_generation_loss / len(train_loader),
                total_quality_loss / len(train_loader),
                total_optimization_loss / len(train_loader))
    
    
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
        self.task_encoder.eval()
        self.code_encoder.eval()
        self.code_decoder.eval()
        self.quality_assessor.eval()
        self.optimization_net.eval()
        
        total_generation_loss = 0
        total_quality_loss = 0
        total_optimization_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估中 | Evaluating"):
                task_descriptions = batch['task_description'].to(self.device)
                input_codes = batch['input_code'].to(self.device)
                target_codes = batch['target_code'].to(self.device)
                
                # 编码任务描述
                task_embeddings = self.task_encoder[0](task_descriptions)
                task_features, _ = self.task_encoder[1](task_embeddings)
                encoded_task = self.task_encoder[2:](task_features[:, -1, :])
                
                # 编码输入代码
                code_embeddings = self.code_encoder[0](input_codes)
                code_features, _ = self.code_encoder[1](code_embeddings)
                encoded_code = self.code_encoder[2:](code_features[:, -1, :])
                
                # 融合特征
                fused_features = torch.cat([encoded_task, encoded_code], dim=1)
                
                # 代码生成
                decoder_input = torch.cat([
                    fused_features.unsqueeze(1), 
                    torch.zeros(batch_size, self.config['max_sequence_length'] - 1, 256).to(self.device)
                ], dim=1)
                
                decoder_output, _ = self.code_decoder[0](decoder_input)
                generated_code = self.code_decoder[1:](decoder_output)
                
                # 代码质量评估
                quality_pred = self.quality_assessor(fused_features)
                
                # 代码优化建议
                optimization_pred = self.optimization_net(fused_features)
                
                # 计算各种损失
                generation_loss = self.generation_loss(
                    generated_code.view(-1, self.vocab_size),
                    target_codes.view(-1)
                )
                
                quality_loss = self.quality_loss(
                    quality_pred,
                    torch.ones_like(quality_pred) * 0.8
                )
                
                optimization_loss = self.optimization_loss(
                    optimization_pred,
                    torch.ones_like(optimization_pred) * 0.5
                )
                
                total_generation_loss += generation_loss.item()
                total_quality_loss += quality_loss.item()
                total_optimization_loss += optimization_loss.item()
        
        return (total_generation_loss / len(test_loader),
                total_quality_loss / len(test_loader),
                total_optimization_loss / len(test_loader))
    
    
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
            'task_encoder_state_dict': self.task_encoder.state_dict(),
            'code_encoder_state_dict': self.code_encoder.state_dict(),
            'code_decoder_state_dict': self.code_decoder.state_dict(),
            'quality_assessor_state_dict': self.quality_assessor.state_dict(),
            'optimization_net_state_dict': self.optimization_net.state_dict(),
            'vocab': self.vocab
        }, path)
        logger.info(f"编程模型已保存至 {path} | Programming model saved to {path}")
    
    
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
            {'params': self.task_encoder.parameters()},
            {'params': self.code_encoder.parameters()},
            {'params': self.code_decoder.parameters()},
            {'params': self.quality_assessor.parameters()},
            {'params': self.optimization_net.parameters()}
        ], lr=self.config['learning_rate'])
        
        # 训练循环
        for epoch in range(epochs):
            (train_gen_loss, train_qual_loss, 
             train_opt_loss) = self.train_epoch(train_loader, optimizer)
            
            (test_gen_loss, test_qual_loss, 
             test_opt_loss) = self.evaluate(test_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"训练生成损失: {train_gen_loss:.4f} | 训练质量损失: {train_qual_loss:.4f} | "
                f"训练优化损失: {train_opt_loss:.4f} | "
                f"测试生成损失: {test_gen_loss:.4f} | 测试质量损失: {test_qual_loss:.4f} | "
                f"测试优化损失: {test_opt_loss:.4f} | "
                f"Train Generation Loss: {train_gen_loss:.4f} | Train Quality Loss: {train_qual_loss:.4f} | "
                f"Train Optimization Loss: {train_opt_loss:.4f} | "
                f"Test Generation Loss: {test_gen_loss:.4f} | Test Quality Loss: {test_qual_loss:.4f} | "
                f"Test Optimization Loss: {test_opt_loss:.4f}"
            )
            
            # 每5轮保存一次模型
            if (epoch + 1) % 5 == 0:
                self.save_model(f"{self.config['model_save_path']}_epoch_{epoch+1}.pth")
        
        # 保存最终模型
        self.save_model(self.config['model_save_path'])
        logger.info("编程模型训练完成 | Programming model training completed")

# 配置示例 | Example configuration
if __name__ == "__main__":
    config = {
        'vocab_size': 104,  # 词汇表大小
        'embedding_dim': 128,
        'max_sequence_length': 100,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'model_save_path': 'models/programming_model.pth'
    }
    
    # 编程数据目录
    programming_data_dir = 'data/programming/'
    
    # 创建数据目录（如果不存在）
    os.makedirs(programming_data_dir, exist_ok=True)
    
    trainer = ProgrammingModelTrainer(config)
    trainer.full_training(programming_data_dir, epochs=12)
