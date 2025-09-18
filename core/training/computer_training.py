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
计算机控制模型训练程序
Computer Control Model Training Program

功能：训练计算机控制模型，支持多系统兼容性控制，通过命令控制电脑完成所有操作，支持MCP服务器集成
Function: Train computer control model supporting multi-system compatibility control, controlling computer operations through commands, and supporting MCP server integration
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
import subprocess
import platform

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
ComputerControlDataset类 - 中文类描述
ComputerControlDataset Class - English class description
"""
class ComputerControlDataset(Dataset):
    """计算机控制数据集 | Computer control dataset"""
    
    
"""
__init__函数 - 中文函数描述
__init__ Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def __init__(self, data_dir, max_sequence_length=50):
        self.data = []
        self.max_sequence_length = max_sequence_length
        
        # 加载计算机控制数据
        control_types = ['system_commands', 'file_operations', 'process_management', 
                        'network_operations', 'system_config', 'mcp_integration']
        
        for control_type in control_types:
            file_path = os.path.join(data_dir, f"computer_{control_type}.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        control_data = json.load(f)
                        self.data.extend(control_data)
                except Exception as e:
                    logger.error(f"计算机控制数据加载错误 {file_path}: {str(e)}")
        
        logger.info(f"加载计算机控制训练数据: {len(self.data)} 条样本 | "
                    f"Loaded computer control training data: {len(self.data)} samples")
    
    
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
            # 命令序列处理
            command_sequence = item.get('command_sequence', [])
            if len(command_sequence) > self.max_sequence_length:
                command_sequence = command_sequence[:self.max_sequence_length]
            
            # 填充或截断序列
            padded_sequence = command_sequence + [''] * (self.max_sequence_length - len(command_sequence))
            
            # 目标系统信息
            target_system = item.get('target_system', 'windows')
            system_encoding = self._get_system_encoding(target_system)
            
            # 预期结果
            expected_result = item.get('expected_result', {})
            
            # MCP集成信息
            mcp_integration = item.get('mcp_integration', {})
            
            return {
                'command_sequence': padded_sequence,
                'target_system': target_system,
                'system_encoding': system_encoding,
                'expected_result': expected_result,
                'mcp_integration': mcp_integration,
                'metadata': {
                    'complexity': item.get('complexity', 'simple'),
                    'privilege_level': item.get('privilege_level', 'user'),
                    'execution_timeout': item.get('execution_timeout', 30)
                }
            }
            
        except Exception as e:
            logger.error(f"计算机控制数据处理错误: {str(e)}")
            # 返回空数据
            return {
                'command_sequence': [''] * self.max_sequence_length,
                'target_system': 'windows',
                'system_encoding': 0,
                'expected_result': {},
                'mcp_integration': {},
                'metadata': {
                    'complexity': 'simple',
                    'privilege_level': 'user',
                    'execution_timeout': 30
                }
            }
    
    
"""
_get_system_encoding函数 - 中文函数描述
_get_system_encoding Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _get_system_encoding(self, system_name):
        """获取系统编码 | Get system encoding"""
        system_mapping = {
            'windows': 0,
            'linux': 1,
            'macos': 2,
            'android': 3,
            'ios': 4,
            'unknown': 5
        }
        return system_mapping.get(system_name.lower(), 5)


"""
ComputerControlModelTrainer类 - 中文类描述
ComputerControlModelTrainer Class - English class description
"""
class ComputerControlModelTrainer:
    """计算机控制模型训练器 | Computer control model trainer"""
    
    
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
        
        # 命令序列编码器
        self.command_encoder = nn.Sequential(
            nn.LSTM(input_size=config['embedding_dim'], 
                   hidden_size=256, 
                   num_layers=2, 
                   batch_first=True,
                   dropout=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(self.device)
        
        # 系统兼容性网络
        self.system_compatibility_net = nn.Sequential(
            nn.Linear(128 + 6, 64),  # 128维特征 + 6维系统编码
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),  # 6种系统兼容性分数
            nn.Softmax(dim=1)
        ).to(self.device)
        
        # 命令执行预测网络
        self.command_execution_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3种执行状态: 成功, 失败, 需要权限
            nn.Softmax(dim=1)
        ).to(self.device)
        
        # 执行时间预测网络
        self.execution_time_net = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # 预测执行时间
            nn.ReLU()
        ).to(self.device)
        
        # MCP集成网络
        self.mcp_integration_net = nn.Sequential(
            nn.Linear(128 + 10, 64),  # 128维特征 + 10维MCP信息
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # 是否需要MCP集成
            nn.Sigmoid()
        ).to(self.device)
        
        # 命令生成网络
        self.command_generation_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config['embedding_dim']),
            nn.Tanh()
        ).to(self.device)
        
        # 损失函数
        self.compatibility_loss = nn.CrossEntropyLoss()
        self.execution_loss = nn.CrossEntropyLoss()
        self.time_loss = nn.MSELoss()
        self.mcp_loss = nn.BCELoss()
        self.generation_loss = nn.MSELoss()
    
    
"""
load_data函数 - 中文函数描述
load_data Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def load_data(self, data_dir):
        """加载计算机控制数据 | Load computer control data"""
        dataset = ComputerControlDataset(data_dir)
        return dataset
    
    
"""
create_data_loader函数 - 中文函数描述
create_data_loader Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def create_data_loader(self, dataset, batch_size=16, shuffle=True):
        """创建数据加载器 | Create data loader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2
        )
    
    
"""
encode_commands函数 - 中文函数描述
encode_commands Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def encode_commands(self, command_sequences):
        """编码命令序列 | Encode command sequences"""
        # 这里使用简单的词嵌入，实际应用中可以使用预训练的语言模型
        batch_size, seq_length = command_sequences.shape
        embeddings = torch.zeros(batch_size, seq_length, self.config['embedding_dim']).to(self.device)
        
        for i in range(batch_size):
            for j in range(seq_length):
                command = command_sequences[i, j]
                if command:
                    # 简单的词频编码
                    embedding = self._simple_command_embedding(command)
                    embeddings[i, j] = torch.FloatTensor(embedding).to(self.device)
        
        return embeddings
    
    
"""
_simple_command_embedding函数 - 中文函数描述
_simple_command_embedding Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _simple_command_embedding(self, command):
        """简单命令嵌入 | Simple command embedding"""
        # 这里使用简单的特征提取，实际应用中可以使用更复杂的方法
        features = np.zeros(self.config['embedding_dim'])
        
        # 命令长度特征
        features[0] = min(len(command) / 100, 1.0)
        
        # 系统命令特征
        system_keywords = {
            'ls': 1, 'dir': 2, 'cd': 3, 'mkdir': 4, 'rm': 5, 'del': 6,
            'cp': 7, 'copy': 8, 'mv': 9, 'move': 10, 'chmod': 11, 'attrib': 12,
            'ps': 13, 'tasklist': 14, 'kill': 15, 'taskkill': 16, 'ping': 17,
            'ipconfig': 18, 'ifconfig': 19, 'netstat': 20, 'ssh': 21, 'scp': 22
        }
        
        for keyword, idx in system_keywords.items():
            if keyword in command.lower():
                features[idx] = 1.0
        
        # 权限级别特征
        if any(word in command.lower() for word in ['sudo', 'admin', 'root', 'runas']):
            features[23] = 1.0
        
        # 文件操作特征
        if any(word in command.lower() for word in ['file', 'read', 'write', 'open', 'close']):
            features[24] = 1.0
        
        # 网络操作特征
        if any(word in command.lower() for word in ['network', 'http', 'ftp', 'tcp', 'udp']):
            features[25] = 1.0
        
        return features
    
    
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
        self.command_encoder.train()
        self.system_compatibility_net.train()
        self.command_execution_net.train()
        self.execution_time_net.train()
        self.mcp_integration_net.train()
        self.command_generation_net.train()
        
        total_compatibility_loss = 0
        total_execution_loss = 0
        total_time_loss = 0
        total_mcp_loss = 0
        total_generation_loss = 0
        
        for batch in tqdm(train_loader, desc="训练中 | Training"):
            command_sequences = batch['command_sequence']
            target_systems = batch['target_system']
            system_encodings = batch['system_encoding']
            expected_results = batch['expected_result']
            mcp_integrations = batch['mcp_integration']
            
            optimizer.zero_grad()
            
            # 编码命令序列
            command_embeddings = self.encode_commands(command_sequences)
            
            # 命令序列特征提取
            features, _ = self.command_encoder(command_embeddings)
            sequence_features = features[:, -1, :]  # 取最后一个时间步的特征
            
            total_loss = 0
            
            # 系统兼容性预测
            system_input = torch.cat([
                sequence_features, 
                torch.eye(6)[system_encodings].to(self.device)
            ], dim=1)
            compatibility_pred = self.system_compatibility_net(system_input)
            
            # 执行状态预测
            execution_pred = self.command_execution_net(sequence_features)
            
            # 执行时间预测
            time_pred = self.execution_time_net(sequence_features)
            
            # MCP集成预测
            mcp_features = torch.cat([
                sequence_features,
                torch.zeros(sequence_features.size(0), 10).to(self.device)  # 简化处理
            ], dim=1)
            mcp_pred = self.mcp_integration_net(mcp_features)
            
            # 命令生成
            generated_commands = self.command_generation_net(sequence_features)
            
            # 计算各种损失
            compatibility_loss = self.compatibility_loss(
                compatibility_pred, 
                torch.ones_like(compatibility_pred).argmax(dim=1)  # 简化目标
            )
            execution_loss = self.execution_loss(
                execution_pred,
                torch.ones_like(execution_pred).argmax(dim=1)  # 简化目标
            )
            time_loss = self.time_loss(
                time_pred,
                torch.ones_like(time_pred) * 5.0  # 简化目标时间
            )
            mcp_loss = self.mcp_loss(
                mcp_pred,
                torch.ones_like(mcp_pred) * 0.5  # 简化目标
            )
            generation_loss = self.generation_loss(
                generated_commands,
                command_embeddings.mean(dim=1)  # 使用平均嵌入作为目标
            )
            
            # 总损失
            total_batch_loss = (compatibility_loss + execution_loss + 
                              time_loss + mcp_loss + generation_loss)
            
            total_batch_loss.backward()
            optimizer.step()
            
            total_compatibility_loss += compatibility_loss.item()
            total_execution_loss += execution_loss.item()
            total_time_loss += time_loss.item()
            total_mcp_loss += mcp_loss.item()
            total_generation_loss += generation_loss.item()
        
        return (total_compatibility_loss / len(train_loader),
                total_execution_loss / len(train_loader),
                total_time_loss / len(train_loader),
                total_mcp_loss / len(train_loader),
                total_generation_loss / len(train_loader))
    
    
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
        self.command_encoder.eval()
        self.system_compatibility_net.eval()
        self.command_execution_net.eval()
        self.execution_time_net.eval()
        self.mcp_integration_net.eval()
        self.command_generation_net.eval()
        
        total_compatibility_loss = 0
        total_execution_loss = 0
        total_time_loss = 0
        total_mcp_loss = 0
        total_generation_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估中 | Evaluating"):
                command_sequences = batch['command_sequence']
                target_systems = batch['target_system']
                system_encodings = batch['system_encoding']
                expected_results = batch['expected_result']
                mcp_integrations = batch['mcp_integration']
                
                # 编码命令序列
                command_embeddings = self.encode_commands(command_sequences)
                
                # 命令序列特征提取
                features, _ = self.command_encoder(command_embeddings)
                sequence_features = features[:, -1, :]
                
                # 系统兼容性预测
                system_input = torch.cat([
                    sequence_features, 
                    torch.eye(6)[system_encodings].to(self.device)
                ], dim=1)
                compatibility_pred = self.system_compatibility_net(system_input)
                
                # 执行状态预测
                execution_pred = self.command_execution_net(sequence_features)
                
                # 执行时间预测
                time_pred = self.execution_time_net(sequence_features)
                
                # MCP集成预测
                mcp_features = torch.cat([
                    sequence_features,
                    torch.zeros(sequence_features.size(0), 10).to(self.device)
                ], dim=1)
                mcp_pred = self.mcp_integration_net(mcp_features)
                
                # 命令生成
                generated_commands = self.command_generation_net(sequence_features)
                
                # 计算各种损失
                compatibility_loss = self.compatibility_loss(
                    compatibility_pred, 
                    torch.ones_like(compatibility_pred).argmax(dim=1)
                )
                execution_loss = self.execution_loss(
                    execution_pred,
                    torch.ones_like(execution_pred).argmax(dim=1)
                )
                time_loss = self.time_loss(
                    time_pred,
                    torch.ones_like(time_pred) * 5.0
                )
                mcp_loss = self.mcp_loss(
                    mcp_pred,
                    torch.ones_like(mcp_pred) * 0.5
                )
                generation_loss = self.generation_loss(
                    generated_commands,
                    command_embeddings.mean(dim=1)
                )
                
                total_compatibility_loss += compatibility_loss.item()
                total_execution_loss += execution_loss.item()
                total_time_loss += time_loss.item()
                total_mcp_loss += mcp_loss.item()
                total_generation_loss += generation_loss.item()
        
        return (total_compatibility_loss / len(test_loader),
                total_execution_loss / len(test_loader),
                total_time_loss / len(test_loader),
                total_mcp_loss / len(test_loader),
                total_generation_loss / len(test_loader))
    
    
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
            'command_encoder_state_dict': self.command_encoder.state_dict(),
            'system_compatibility_net_state_dict': self.system_compatibility_net.state_dict(),
            'command_execution_net_state_dict': self.command_execution_net.state_dict(),
            'execution_time_net_state_dict': self.execution_time_net.state_dict(),
            'mcp_integration_net_state_dict': self.mcp_integration_net.state_dict(),
            'command_generation_net_state_dict': self.command_generation_net.state_dict()
        }, path)
        logger.info(f"计算机控制模型已保存至 {path} | Computer control model saved to {path}")
    
    
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
            {'params': self.command_encoder.parameters()},
            {'params': self.system_compatibility_net.parameters()},
            {'params': self.command_execution_net.parameters()},
            {'params': self.execution_time_net.parameters()},
            {'params': self.mcp_integration_net.parameters()},
            {'params': self.command_generation_net.parameters()}
        ], lr=self.config['learning_rate'])
        
        # 训练循环
        for epoch in range(epochs):
            (train_comp_loss, train_exec_loss, train_time_loss, 
             train_mcp_loss, train_gen_loss) = self.train_epoch(train_loader, optimizer)
            
            (test_comp_loss, test_exec_loss, test_time_loss, 
             test_mcp_loss, test_gen_loss) = self.evaluate(test_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"训练兼容性损失: {train_comp_loss:.4f} | 训练执行损失: {train_exec_loss:.4f} | "
                f"训练时间损失: {train_time_loss:.4f} | 训练MCP损失: {train_mcp_loss:.4f} | "
                f"训练生成损失: {train_gen_loss:.4f} | "
                f"测试兼容性损失: {test_comp_loss:.4f} | 测试执行损失: {test_exec_loss:.4f} | "
                f"测试时间损失: {test_time_loss:.4f} | 测试MCP损失: {test_mcp_loss:.4f} | "
                f"测试生成损失: {test_gen_loss:.4f} | "
                f"Train Compatibility Loss: {train_comp_loss:.4f} | Train Execution Loss: {train_exec_loss:.4f} | "
                f"Train Time Loss: {train_time_loss:.4f} | Train MCP Loss: {train_mcp_loss:.4f} | "
                f"Train Generation Loss: {train_gen_loss:.4f} | "
                f"Test Compatibility Loss: {test_comp_loss:.4f} | Test Execution Loss: {test_exec_loss:.4f} | "
                f"Test Time Loss: {test_time_loss:.4f} | Test MCP Loss: {test_mcp_loss:.4f} | "
                f"Test Generation Loss: {test_gen_loss:.4f}"
            )
            
            # 每5轮保存一次模型
            if (epoch + 1) % 5 == 0:
                self.save_model(f"{self.config['model_save_path']}_epoch_{epoch+1}.pth")
        
        # 保存最终模型
        self.save_model(self.config['model_save_path'])
        logger.info("计算机控制模型训练完成 | Computer control model training completed")

# 配置示例 | Example configuration
if __name__ == "__main__":
    config = {
        'embedding_dim': 50,  # 命令嵌入维度
        'batch_size': 16,
        'learning_rate': 1e-4,
        'model_save_path': 'models/computer_control_model.pth'
    }
    
    # 计算机控制数据目录
    computer_data_dir = 'data/computer_control/'
    
    # 创建数据目录（如果不存在）
    os.makedirs(computer_data_dir, exist_ok=True)
    
    trainer = ComputerControlModelTrainer(config)
    trainer.full_training(computer_data_dir, epochs=12)
