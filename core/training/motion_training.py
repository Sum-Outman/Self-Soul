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
运动和执行器控制模型训练程序
Motion and Actuator Control Model Training Program

功能：训练运动和执行器控制模型，根据其他感知模型数据和需求完成各种复杂的控制，多端口输出，多信号通讯形式兼容
Function: Train motion and actuator control model to complete various complex controls based on other perceptual model data and requirements, with multi-port output and multi-signal communication compatibility
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

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
MotionDataset类 - 中文类描述
MotionDataset Class - English class description
"""
class MotionDataset(Dataset):
    """运动控制数据集 | Motion control dataset"""
    
    
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
        
        # 加载运动控制数据
        motion_types = ['motor_control', 'servo_control', 'actuator_control', 
                       'robotic_arm', 'locomotion', 'grasping', 'manipulation']
        
        for motion_type in motion_types:
            file_path = os.path.join(data_dir, f"motion_{motion_type}.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        motion_data = json.load(f)
                        self.data.extend(motion_data)
                except Exception as e:
                    logger.error(f"运动控制数据加载错误 {file_path}: {str(e)}")
        
        logger.info(f"加载运动控制训练数据: {len(self.data)} 条样本 | "
                    f"Loaded motion control training data: {len(self.data)} samples")
    
    
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
            # 感知数据输入
            sensor_data = item.get('sensor_input', {})
            spatial_data = item.get('spatial_input', {})
            vision_data = item.get('vision_input', {})
            
            # 控制命令序列
            control_sequence = item.get('control_sequence', [])
            if len(control_sequence) > self.max_sequence_length:
                control_sequence = control_sequence[:self.max_sequence_length]
            
            # 填充或截断序列
            padded_sequence = control_sequence + [0] * (self.max_sequence_length - len(control_sequence))
            
            # 预期输出
            expected_output = item.get('expected_output', {})
            
            # 执行器类型
            actuator_type = item.get('actuator_type', 'motor')
            
            # 通讯协议
            communication_protocol = item.get('communication_protocol', 'pwm')
            
            return {
                'sensor_input': self._process_sensor_data(sensor_data),
                'spatial_input': self._process_spatial_data(spatial_data),
                'vision_input': self._process_vision_data(vision_data),
                'control_sequence': padded_sequence,
                'expected_output': expected_output,
                'actuator_type': actuator_type,
                'communication_protocol': communication_protocol,
                'metadata': {
                    'complexity': item.get('complexity', 'simple'),
                    'safety_level': item.get('safety_level', 'low'),
                    'precision_requirement': item.get('precision_requirement', 0.1)
                }
            }
            
        except Exception as e:
            logger.error(f"运动控制数据处理错误: {str(e)}")
            # 返回空数据
            return {
                'sensor_input': np.zeros(20),
                'spatial_input': np.zeros(10),
                'vision_input': np.zeros(100),
                'control_sequence': [0] * self.max_sequence_length,
                'expected_output': {},
                'actuator_type': 'motor',
                'communication_protocol': 'pwm',
                'metadata': {
                    'complexity': 'simple',
                    'safety_level': 'low',
                    'precision_requirement': 0.1
                }
            }
    
    
"""
_process_sensor_data函数 - 中文函数描述
_process_sensor_data Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _process_sensor_data(self, sensor_data):
        """处理传感器数据 | Process sensor data"""
        # 提取各种传感器数据并转换为特征向量
        features = []
        
        # 加速度传感器
        accel = sensor_data.get('acceleration', [0, 0, 0])
        features.extend(accel)
        
        # 陀螺仪
        gyro = sensor_data.get('gyroscope', [0, 0, 0])
        features.extend(gyro)
        
        # 距离传感器
        distance = sensor_data.get('distance', 0)
        features.append(distance)
        
        # 压力传感器
        pressure = sensor_data.get('pressure', 0)
        features.append(pressure)
        
        # 温度传感器
        temperature = sensor_data.get('temperature', 25)
        features.append(temperature)
        
        # 填充到固定长度
        if len(features) < 20:
            features.extend([0] * (20 - len(features)))
        else:
            features = features[:20]
            
        return np.array(features, dtype=np.float32)
    
    
"""
_process_spatial_data函数 - 中文函数描述
_process_spatial_data Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _process_spatial_data(self, spatial_data):
        """处理空间数据 | Process spatial data"""
        # 提取空间定位数据
        features = []
        
        # 位置坐标
        position = spatial_data.get('position', [0, 0, 0])
        features.extend(position)
        
        # 方向向量
        orientation = spatial_data.get('orientation', [0, 0, 0, 1])  # 四元数
        features.extend(orientation)
        
        # 运动速度
        velocity = spatial_data.get('velocity', [0, 0, 0])
        features.extend(velocity)
        
        # 填充到固定长度
        if len(features) < 10:
            features.extend([0] * (10 - len(features)))
        else:
            features = features[:10]
            
        return np.array(features, dtype=np.float32)
    
    
"""
_process_vision_data函数 - 中文函数描述
_process_vision_data Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _process_vision_data(self, vision_data):
        """处理视觉数据 | Process vision data"""
        # 提取视觉特征数据
        features = []
        
        # 目标检测结果
        objects = vision_data.get('detected_objects', [])
        for obj in objects:
            # 位置和置信度
            features.extend(obj.get('bbox', [0, 0, 0, 0]))
            features.append(obj.get('confidence', 0))
        
        # 填充到固定长度
        if len(features) < 100:
            features.extend([0] * (100 - len(features)))
        else:
            features = features[:100]
            
        return np.array(features, dtype=np.float32)


"""
MotionModelTrainer类 - 中文类描述
MotionModelTrainer Class - English class description
"""
class MotionModelTrainer:
    """运动控制模型训练器 | Motion control model trainer"""
    
    
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
        
        # 传感器数据编码器
        self.sensor_encoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(self.device)
        
        # 空间数据编码器
        self.spatial_encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(self.device)
        
        # 视觉数据编码器
        self.vision_encoder = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(self.device)
        
        # 特征融合网络
        self.feature_fusion = nn.Sequential(
            nn.Linear(32 + 16 + 64, 128),  # 传感器32 + 空间16 + 视觉64
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        ).to(self.device)
        
        # 控制序列生成器
        self.control_generator = nn.Sequential(
            nn.LSTM(input_size=64, 
                   hidden_size=128, 
                   num_layers=2, 
                   batch_first=True,
                   dropout=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config['output_dim']),
            nn.Tanh()  # 输出在-1到1之间，适合控制信号
        ).to(self.device)
        
        # 执行器类型适配器
        self.actuator_adapter = nn.Sequential(
            nn.Linear(64 + 10, 32),  # 64维特征 + 10维执行器编码
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),  # 8种执行器类型适配
            nn.Softmax(dim=1)
        ).to(self.device)
        
        # 通讯协议适配器
        self.protocol_adapter = nn.Sequential(
            nn.Linear(64 + 8, 32),  # 64维特征 + 8维协议编码
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6),  # 6种通讯协议
            nn.Softmax(dim=1)
        ).to(self.device)
        
        # 安全性评估网络
        self.safety_assessor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # 3种安全级别: 低, 中, 高
            nn.Softmax(dim=1)
        ).to(self.device)
        
        # 损失函数
        self.control_loss = nn.MSELoss()
        self.actuator_loss = nn.CrossEntropyLoss()
        self.protocol_loss = nn.CrossEntropyLoss()
        self.safety_loss = nn.CrossEntropyLoss()
    
    
"""
load_data函数 - 中文函数描述
load_data Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def load_data(self, data_dir):
        """加载运动控制数据 | Load motion control data"""
        dataset = MotionDataset(data_dir)
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
_encode_actuator_type函数 - 中文函数描述
_encode_actuator_type Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _encode_actuator_type(self, actuator_type):
        """编码执行器类型 | Encode actuator type"""
        actuator_mapping = {
            'motor': 0, 'servo': 1, 'stepper': 2, 'linear_actuator': 3,
            'pneumatic': 4, 'hydraulic': 5, 'solenoid': 6, 'piezo': 7
        }
        encoding = torch.zeros(10)
        idx = actuator_mapping.get(actuator_type, 0)
        encoding[idx] = 1.0
        return encoding
    
    
"""
_encode_protocol函数 - 中文函数描述
_encode_protocol Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _encode_protocol(self, protocol):
        """编码通讯协议 | Encode communication protocol"""
        protocol_mapping = {
            'pwm': 0, 'i2c': 1, 'spi': 2, 'uart': 3, 'can': 4, 'ethernet': 5
        }
        encoding = torch.zeros(8)
        idx = protocol_mapping.get(protocol, 0)
        encoding[idx] = 1.0
        return encoding
    
    
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
        self.sensor_encoder.train()
        self.spatial_encoder.train()
        self.vision_encoder.train()
        self.feature_fusion.train()
        self.control_generator.train()
        self.actuator_adapter.train()
        self.protocol_adapter.train()
        self.safety_assessor.train()
        
        total_control_loss = 0
        total_actuator_loss = 0
        total_protocol_loss = 0
        total_safety_loss = 0
        
        for batch in tqdm(train_loader, desc="训练中 | Training"):
            sensor_inputs = batch['sensor_input'].float().to(self.device)
            spatial_inputs = batch['spatial_input'].float().to(self.device)
            vision_inputs = batch['vision_input'].float().to(self.device)
            control_targets = batch['control_sequence'].float().to(self.device)
            actuator_types = batch['actuator_type']
            protocols = batch['communication_protocol']
            
            optimizer.zero_grad()
            
            # 编码各种输入数据
            sensor_features = self.sensor_encoder(sensor_inputs)
            spatial_features = self.spatial_encoder(spatial_inputs)
            vision_features = self.vision_encoder(vision_inputs)
            
            # 特征融合
            fused_features = torch.cat([sensor_features, spatial_features, vision_features], dim=1)
            context_features = self.feature_fusion(fused_features)
            
            # 生成控制序列
            control_input = context_features.unsqueeze(1).repeat(1, self.config['max_sequence_length'], 1)
            control_output, _ = self.control_generator[0](control_input)
            generated_controls = self.control_generator[1:](control_output)
            
            # 执行器类型适配
            actuator_encodings = torch.stack([self._encode_actuator_type(at).to(self.device) for at in actuator_types])
            actuator_input = torch.cat([context_features, actuator_encodings], dim=1)
            actuator_pred = self.actuator_adapter(actuator_input)
            
            # 通讯协议适配
            protocol_encodings = torch.stack([self._encode_protocol(p).to(self.device) for p in protocols])
            protocol_input = torch.cat([context_features, protocol_encodings], dim=1)
            protocol_pred = self.protocol_adapter(protocol_input)
            
            # 安全性评估
            safety_pred = self.safety_assessor(context_features)
            
            total_loss = 0
            
            # 计算各种损失
            control_loss = self.control_loss(
                generated_controls,
                control_targets.unsqueeze(2)  # 添加维度匹配
            )
            
            actuator_loss = self.actuator_loss(
                actuator_pred,
                torch.ones_like(actuator_pred).argmax(dim=1)  # 简化目标
            )
            
            protocol_loss = self.protocol_loss(
                protocol_pred,
                torch.ones_like(protocol_pred).argmax(dim=1)  # 简化目标
            )
            
            safety_loss = self.safety_loss(
                safety_pred,
                torch.ones_like(safety_pred).argmax(dim=1)  # 简化目标
            )
            
            # 总损失
            total_batch_loss = (control_loss + 0.5 * actuator_loss + 
                              0.3 * protocol_loss + 0.2 * safety_loss)
            
            total_batch_loss.backward()
            optimizer.step()
            
            total_control_loss += control_loss.item()
            total_actuator_loss += actuator_loss.item()
            total_protocol_loss += protocol_loss.item()
            total_safety_loss += safety_loss.item()
        
        return (total_control_loss / len(train_loader),
                total_actuator_loss / len(train_loader),
                total_protocol_loss / len(train_loader),
                total_safety_loss / len(train_loader))
    
    
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
        self.sensor_encoder.eval()
        self.spatial_encoder.eval()
        self.vision_encoder.eval()
        self.feature_fusion.eval()
        self.control_generator.eval()
        self.actuator_adapter.eval()
        self.protocol_adapter.eval()
        self.safety_assessor.eval()
        
        total_control_loss = 0
        total_actuator_loss = 0
        total_protocol_loss = 0
        total_safety_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估中 | Evaluating"):
                sensor_inputs = batch['sensor_input'].float().to(self.device)
                spatial_inputs = batch['spatial_input'].float().to(self.device)
                vision_inputs = batch['vision_input'].float().to(self.device)
                control_targets = batch['control_sequence'].float().to(self.device)
                actuator_types = batch['actuator_type']
                protocols = batch['communication_protocol']
                
                # 编码各种输入数据
                sensor_features = self.sensor_encoder(sensor_inputs)
                spatial_features = self.spatial_encoder(spatial_inputs)
                vision_features = self.vision_encoder(vision_inputs)
                
                # 特征融合
                fused_features = torch.cat([sensor_features, spatial_features, vision_features], dim=1)
                context_features = self.feature_fusion(fused_features)
                
                # 生成控制序列
                control_input = context_features.unsqueeze(1).repeat(1, self.config['max_sequence_length'], 1)
                control_output, _ = self.control_generator[0](control_input)
                generated_controls = self.control_generator[1:](control_output)
                
                # 执行器类型适配
                actuator_encodings = torch.stack([self._encode_actuator_type(at).to(self.device) for at in actuator_types])
                actuator_input = torch.cat([context_features, actuator_encodings], dim=1)
                actuator_pred = self.actuator_adapter(actuator_input)
                
                # 通讯协议适配
                protocol_encodings = torch.stack([self._encode_protocol(p).to(self.device) for p in protocols])
                protocol_input = torch.cat([context_features, protocol_encodings], dim=1)
                protocol_pred = self.protocol_adapter(protocol_input)
                
                # 安全性评估
                safety_pred = self.safety_assessor(context_features)
                
                # 计算各种损失
                control_loss = self.control_loss(
                    generated_controls,
                    control_targets.unsqueeze(2)
                )
                
                actuator_loss = self.actuator_loss(
                    actuator_pred,
                    torch.ones_like(actuator_pred).argmax(dim=1)
                )
                
                protocol_loss = self.protocol_loss(
                    protocol_pred,
                    torch.ones_like(protocol_pred).argmax(dim=1)
                )
                
                safety_loss = self.safety_loss(
                    safety_pred,
                    torch.ones_like(safety_pred).argmax(dim=1)
                )
                
                total_control_loss += control_loss.item()
                total_actuator_loss += actuator_loss.item()
                total_protocol_loss += protocol_loss.item()
                total_safety_loss += safety_loss.item()
        
        return (total_control_loss / len(test_loader),
                total_actuator_loss / len(test_loader),
                total_protocol_loss / len(test_loader),
                total_safety_loss / len(test_loader))
    
    
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
            'sensor_encoder_state_dict': self.sensor_encoder.state_dict(),
            'spatial_encoder_state_dict': self.spatial_encoder.state_dict(),
            'vision_encoder_state_dict': self.vision_encoder.state_dict(),
            'feature_fusion_state_dict': self.feature_fusion.state_dict(),
            'control_generator_state_dict': self.control_generator.state_dict(),
            'actuator_adapter_state_dict': self.actuator_adapter.state_dict(),
            'protocol_adapter_state_dict': self.protocol_adapter.state_dict(),
            'safety_assessor_state_dict': self.safety_assessor.state_dict()
        }, path)
        logger.info(f"运动控制模型已保存至 {path} | Motion control model saved to {path}")
    
    
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
            {'params': self.sensor_encoder.parameters()},
            {'params': self.spatial_encoder.parameters()},
            {'params': self.vision_encoder.parameters()},
            {'params': self.feature_fusion.parameters()},
            {'params': self.control_generator.parameters()},
            {'params': self.actuator_adapter.parameters()},
            {'params': self.protocol_adapter.parameters()},
            {'params': self.safety_assessor.parameters()}
        ], lr=self.config['learning_rate'])
        
        # 训练循环
        for epoch in range(epochs):
            (train_ctrl_loss, train_act_loss, 
             train_prot_loss, train_safe_loss) = self.train_epoch(train_loader, optimizer)
            
            (test_ctrl_loss, test_act_loss, 
             test_prot_loss, test_safe_loss) = self.evaluate(test_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"训练控制损失: {train_ctrl_loss:.4f} | 训练执行器损失: {train_act_loss:.4f} | "
                f"训练协议损失: {train_prot_loss:.4f} | 训练安全损失: {train_safe_loss:.4f} | "
                f"测试控制损失: {test_ctrl_loss:.4f} | 测试执行器损失: {test_act_loss:.4f} | "
                f"测试协议损失: {test_prot_loss:.4f} | 测试安全损失: {test_safe_loss:.4f} | "
                f"Train Control Loss: {train_ctrl_loss:.4f} | Train Actuator Loss: {train_act_loss:.4f} | "
                f"Train Protocol Loss: {train_prot_loss:.4f} | Train Safety Loss: {train_safe_loss:.4f} | "
                f"Test Control Loss: {test_ctrl_loss:.4f} | Test Actuator Loss: {test_act_loss:.4f} | "
                f"Test Protocol Loss: {test_prot_loss:.4f} | Test Safety Loss: {test_safe_loss:.4f}"
            )
            
            # 每5轮保存一次模型
            if (epoch + 1) % 5 == 0:
                self.save_model(f"{self.config['model_save_path']}_epoch_{epoch+1}.pth")
        
        # 保存最终模型
        self.save_model(self.config['model_save_path'])
        logger.info("运动控制模型训练完成 | Motion control model training completed")

# 配置示例 | Example configuration
if __name__ == "__main__":
    config = {
        'output_dim': 1,  # 控制信号输出维度
        'max_sequence_length': 50,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'model_save_path': 'models/motion_control_model.pth'
    }
    
    # 运动控制数据目录
    motion_data_dir = 'data/motion_control/'
    
    # 创建数据目录（如果不存在）
    os.makedirs(motion_data_dir, exist_ok=True)
    
    trainer = MotionModelTrainer(config)
    trainer.full_training(motion_data_dir, epochs=12)
