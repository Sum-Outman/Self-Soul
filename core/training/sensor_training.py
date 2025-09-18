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
传感器感知模型训练程序
Sensor Perception Model Training Program

功能：训练多模态传感器感知模型，支持温湿度、加速度、速度、位移、六轴陀螺仪、压力、气压、测距、红外、味觉、烟雾、光传感器等多种传感器数据处理
Function: Train multimodal sensor perception model supporting temperature/humidity, acceleration, velocity, displacement, 6-axis gyroscope, pressure, barometric pressure, distance measurement, infrared, taste, smoke, light sensors and other sensor data processing
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
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SensorDataset(Dataset):
    """传感器数据集 | Sensor dataset"""
    
    def __init__(self, data_dir, sequence_length=100):
        """初始化传感器数据集 | Initialize sensor dataset
        
        Args:
            data_dir: 数据目录路径 | Data directory path
            sequence_length: 序列长度 | Sequence length
        """
        self.data = []
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
        # 加载多模态传感器数据
        sensor_types = [
            'temperature_humidity', 'acceleration', 'velocity', 'displacement',
            'gyroscope', 'pressure', 'barometric', 'distance', 
            'infrared', 'taste', 'smoke', 'light'
        ]
        
        for sensor_type in sensor_types:
            file_path = os.path.join(data_dir, f"sensor_{sensor_type}.csv")
            if os.path.exists(file_path):
                try:
                    # 读取CSV数据
                    df = pd.read_csv(file_path)
                    
                    # 数据预处理
                    sensor_data = self._preprocess_sensor_data(df, sensor_type)
                    self.data.extend(sensor_data)
                    
                except Exception as e:
                    logger.error(f"传感器数据加载错误 {file_path}: {str(e)} | Sensor data loading error {file_path}: {str(e)}")
        
        logger.info(f"加载传感器训练数据: {len(self.data)} 条样本 | Loaded sensor training data: {len(self.data)} samples")
    
    def _preprocess_sensor_data(self, df, sensor_type):
        """预处理传感器数据 | Preprocess sensor data
        
        Args:
            df: 数据框 | DataFrame
            sensor_type: 传感器类型 | Sensor type
            
        Returns:
            处理后的数据列表 | Processed data list
        """
        processed_data = []
        
        # 根据传感器类型进行特定处理
        if sensor_type == 'temperature_humidity':
            # 温湿度传感器数据
            for _, row in df.iterrows():
                processed_data.append({
                    'sensor_type': 'temperature_humidity',
                    'data': np.array([row.get('temperature', 25.0), row.get('humidity', 50.0)]),
                    'timestamp': row.get('timestamp', 0),
                    'environment': row.get('environment', 'normal')
                })
        
        elif sensor_type == 'acceleration':
            # 加速度传感器数据
            for _, row in df.iterrows():
                processed_data.append({
                    'sensor_type': 'acceleration',
                    'data': np.array([
                        row.get('accel_x', 0.0), 
                        row.get('accel_y', 0.0), 
                        row.get('accel_z', 0.0)
                    ]),
                    'timestamp': row.get('timestamp', 0),
                    'motion_state': row.get('motion_state', 'stationary')
                })
        
        elif sensor_type == 'gyroscope':
            # 六轴陀螺仪数据
            for _, row in df.iterrows():
                processed_data.append({
                    'sensor_type': 'gyroscope',
                    'data': np.array([
                        row.get('gyro_x', 0.0), 
                        row.get('gyro_y', 0.0), 
                        row.get('gyro_z', 0.0),
                        row.get('accel_x', 0.0), 
                        row.get('accel_y', 0.0), 
                        row.get('accel_z', 0.0)
                    ]),
                    'timestamp': row.get('timestamp', 0),
                    'orientation': row.get('orientation', 'unknown')
                })
        
        # 其他传感器类型的处理...
        else:
            # 通用传感器数据处理
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            for _, row in df.iterrows():
                sensor_data = [row[col] for col in numeric_columns if col != 'timestamp']
                processed_data.append({
                    'sensor_type': sensor_type,
                    'data': np.array(sensor_data),
                    'timestamp': row.get('timestamp', 0),
                    'status': row.get('status', 'normal')
                })
        
        return processed_data
    
    def __len__(self):
        """获取数据集长度 | Get dataset length
        
        Returns:
            数据集长度 | Dataset length
        """
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        """获取数据项 | Get data item
        
        Args:
            idx: 索引 | Index
            
        Returns:
            包含传感器数据、标签和类型的字典 | Dictionary containing sensor data, labels and type
        """
        # 获取时间序列数据
        sequence_data = []
        labels = []
        
        for i in range(self.sequence_length):
            item = self.data[idx + i]
            sequence_data.append(item['data'])
            
            # 根据传感器类型提取标签
            if item['sensor_type'] == 'temperature_humidity':
                labels.append({
                    'temperature': item['data'][0],
                    'humidity': item['data'][1],
                    'environment': item.get('environment', 'normal')
                })
            elif item['sensor_type'] == 'acceleration':
                labels.append({
                    'accel_x': item['data'][0],
                    'accel_y': item['data'][1],
                    'accel_z': item['data'][2],
                    'motion_state': item.get('motion_state', 'stationary')
                })
            elif item['sensor_type'] == 'gyroscope':
                labels.append({
                    'gyro_x': item['data'][0],
                    'gyro_y': item['data'][1],
                    'gyro_z': item['data'][2],
                    'accel_x': item['data'][3],
                    'accel_y': item['data'][4],
                    'accel_z': item['data'][5],
                    'orientation': item.get('orientation', 'unknown')
                })
            else:
                labels.append({
                    'sensor_values': item['data'],
                    'status': item.get('status', 'normal')
                })
        
        # 转换为张量
        sequence_tensor = torch.FloatTensor(np.array(sequence_data))
        
        return {
            'sensor_data': sequence_tensor,
            'labels': labels,
            'sensor_type': self.data[idx]['sensor_type']
        }


class SensorModelTrainer:
    """传感器模型训练器 | Sensor model trainer"""
    
    def __init__(self, config):
        """初始化训练器 | Initialize trainer
        
        Args:
            config: 配置字典 | Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 传感器特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.LSTM(input_size=config['input_size'], 
                   hidden_size=128, 
                   num_layers=2, 
                   batch_first=True,
                   dropout=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(self.device)
        
        # 温度湿度预测网络
        self.temperature_humidity_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # 温度和湿度
            nn.Sigmoid()
        ).to(self.device)
        
        # 运动状态分类网络
        self.motion_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),  # 5种运动状态
            nn.Softmax(dim=1)
        ).to(self.device)
        
        # 姿态估计网络
        self.pose_estimator = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),  # 6个姿态参数
            nn.Tanh()
        ).to(self.device)
        
        # 环境状态分类网络
        self.environment_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # 4种环境状态
            nn.Softmax(dim=1)
        ).to(self.device)
        
        # 异常检测网络
        self.anomaly_detector = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # 异常分数
            nn.Sigmoid()
        ).to(self.device)
        
        # 损失函数
        self.regression_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.anomaly_loss = nn.BCELoss()
    
    def load_data(self, data_dir):
        """加载传感器数据 | Load sensor data
        
        Args:
            data_dir: 数据目录 | Data directory
            
        Returns:
            传感器数据集 | Sensor dataset
        """
        dataset = SensorDataset(data_dir)
        return dataset
    
    def create_data_loader(self, dataset, batch_size=32, shuffle=True):
        """创建数据加载器 | Create data loader
        
        Args:
            dataset: 数据集 | Dataset
            batch_size: 批次大小 | Batch size
            shuffle: 是否打乱 | Whether to shuffle
            
        Returns:
            数据加载器 | Data loader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4
        )
    
    def train_epoch(self, train_loader, optimizer):
        """单轮训练 | Single epoch training
        
        Args:
            train_loader: 训练数据加载器 | Training data loader
            optimizer: 优化器 | Optimizer
            
        Returns:
            各种损失值 | Various loss values
        """
        self.feature_extractor.train()
        self.temperature_humidity_predictor.train()
        self.motion_classifier.train()
        self.pose_estimator.train()
        self.environment_classifier.train()
        self.anomaly_detector.train()
        
        total_temp_hum_loss = 0
        total_motion_loss = 0
        total_pose_loss = 0
        total_env_loss = 0
        total_anomaly_loss = 0
        
        for batch in tqdm(train_loader, desc="训练中 | Training"):
            sensor_data = batch['sensor_data'].to(self.device)
            labels = batch['labels']
            sensor_type = batch['sensor_type']
            
            optimizer.zero_grad()
            
            # 特征提取
            features, _ = self.feature_extractor(sensor_data)
            features = features[:, -1, :]  # 取最后一个时间步的特征
            
            total_loss = 0
            temp_hum_loss = 0
            motion_loss = 0
            pose_loss = 0
            env_loss = 0
            anomaly_loss = 0
            
            # 根据传感器类型计算相应的损失
            for i in range(len(sensor_type)):
                if sensor_type[i] == 'temperature_humidity':
                    # 温湿度预测
                    pred = self.temperature_humidity_predictor(features[i])
                    target = torch.FloatTensor([
                        labels[i][0]['temperature'] / 100.0,  # 归一化
                        labels[i][0]['humidity'] / 100.0
                    ]).to(self.device)
                    loss = self.regression_loss(pred, target)
                    temp_hum_loss += loss.item()
                    total_loss += loss
                
                elif sensor_type[i] == 'acceleration':
                    # 运动状态分类
                    pred = self.motion_classifier(features[i])
                    motion_states = ['stationary', 'walking', 'running', 'falling', 'jumping']
                    target_state = labels[i][0].get('motion_state', 'stationary')
                    target_idx = motion_states.index(target_state) if target_state in motion_states else 0
                    target = torch.LongTensor([target_idx]).to(self.device)
                    loss = self.classification_loss(pred.unsqueeze(0), target)
                    motion_loss += loss.item()
                    total_loss += loss
                
                elif sensor_type[i] == 'gyroscope':
                    # 姿态估计
                    pred = self.pose_estimator(features[i])
                    target = torch.FloatTensor([
                        labels[i][0].get('gyro_x', 0.0) / 10.0,  # 归一化
                        labels[i][0].get('gyro_y', 0.0) / 10.0,
                        labels[i][0].get('gyro_z', 0.0) / 10.0,
                        labels[i][0].get('accel_x', 0.0) / 20.0,
                        labels[i][0].get('accel_y', 0.0) / 20.0,
                        labels[i][0].get('accel_z', 0.0) / 20.0
                    ]).to(self.device)
                    loss = self.regression_loss(pred, target)
                    pose_loss += loss.item()
                    total_loss += loss
                
                else:
                    # 环境状态分类和异常检测
                    env_pred = self.environment_classifier(features[i])
                    environments = ['normal', 'abnormal', 'dangerous', 'unknown']
                    target_env = labels[i][0].get('status', 'normal')
                    target_env_idx = environments.index(target_env) if target_env in environments else 0
                    env_target = torch.LongTensor([target_env_idx]).to(self.device)
                    env_loss_val = self.classification_loss(env_pred.unsqueeze(0), env_target)
                    env_loss += env_loss_val.item()
                    total_loss += env_loss_val
                    
                    # 异常检测
                    anomaly_pred = self.anomaly_detector(features[i])
                    anomaly_target = torch.FloatTensor([1.0 if target_env != 'normal' else 0.0]).to(self.device)
                    anomaly_loss_val = self.anomaly_loss(anomaly_pred, anomaly_target)
                    anomaly_loss += anomaly_loss_val.item()
                    total_loss += anomaly_loss_val
            
            total_loss.backward()
            optimizer.step()
            
            total_temp_hum_loss += temp_hum_loss / len(sensor_type)
            total_motion_loss += motion_loss / len(sensor_type)
            total_pose_loss += pose_loss / len(sensor_type)
            total_env_loss += env_loss / len(sensor_type)
            total_anomaly_loss += anomaly_loss / len(sensor_type)
        
        return (total_temp_hum_loss / len(train_loader),
                total_motion_loss / len(train_loader),
                total_pose_loss / len(train_loader),
                total_env_loss / len(train_loader),
                total_anomaly_loss / len(train_loader))
    
    def evaluate(self, test_loader):
        """模型评估 | Model evaluation
        
        Args:
            test_loader: 测试数据加载器 | Test data loader
            
        Returns:
            各种损失值 | Various loss values
        """
        self.feature_extractor.eval()
        self.temperature_humidity_predictor.eval()
        self.motion_classifier.eval()
        self.pose_estimator.eval()
        self.environment_classifier.eval()
        self.anomaly_detector.eval()
        
        total_temp_hum_loss = 0
        total_motion_loss = 0
        total_pose_loss = 0
        total_env_loss = 0
        total_anomaly_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估中 | Evaluating"):
                sensor_data = batch['sensor_data'].to(self.device)
                labels = batch['labels']
                sensor_type = batch['sensor_type']
                
                # 特征提取
                features, _ = self.feature_extractor(sensor_data)
                features = features[:, -1, :]
                
                temp_hum_loss = 0
                motion_loss = 0
                pose_loss = 0
                env_loss = 0
                anomaly_loss = 0
                
                for i in range(len(sensor_type)):
                    if sensor_type[i] == 'temperature_humidity':
                        pred = self.temperature_humidity_predictor(features[i])
                        target = torch.FloatTensor([
                            labels[i][0]['temperature'] / 100.0,
                            labels[i][0]['humidity'] / 100.0
                        ]).to(self.device)
                        temp_hum_loss += self.regression_loss(pred, target).item()
                    
                    elif sensor_type[i] == 'acceleration':
                        pred = self.motion_classifier(features[i])
                        motion_states = ['stationary', 'walking', 'running', 'falling', 'jumping']
                        target_state = labels[i][0].get('motion_state', 'stationary')
                        target_idx = motion_states.index(target_state) if target_state in motion_states else 0
                        target = torch.LongTensor([target_idx]).to(self.device)
                        motion_loss += self.classification_loss(pred.unsqueeze(0), target).item()
                    
                    elif sensor_type[i] == 'gyroscope':
                        pred = self.pose_estimator(features[i])
                        target = torch.FloatTensor([
                            labels[i][0].get('gyro_x', 0.0) / 10.0,
                            labels[i][0].get('gyro_y', 0.0) / 10.0,
                            labels[i][0].get('gyro_z', 0.0) / 10.0,
                            labels[i][0].get('accel_x', 0.0) / 20.0,
                            labels[i][0].get('accel_y', 0.0) / 20.0,
                            labels[i][0].get('accel_z', 0.0) / 20.0
                        ]).to(self.device)
                        pose_loss += self.regression_loss(pred, target).item()
                    
                    else:
                        # 环境状态分类和异常检测
                        env_pred = self.environment_classifier(features[i])
                        environments = ['normal', 'abnormal', 'dangerous', 'unknown']
                        target_env = labels[i][0].get('status', 'normal')
                        target_env_idx = environments.index(target_env) if target_env in environments else 0
                        env_target = torch.LongTensor([target_env_idx]).to(self.device)
                        env_loss += self.classification_loss(env_pred.unsqueeze(0), env_target).item()
                        
                        # 异常检测
                        anomaly_pred = self.anomaly_detector(features[i])
                        anomaly_target = torch.FloatTensor([1.0 if target_env != 'normal' else 0.0]).to(self.device)
                        anomaly_loss += self.anomaly_loss(anomaly_pred, anomaly_target).item()
                
                total_temp_hum_loss += temp_hum_loss / len(sensor_type)
                total_motion_loss += motion_loss / len(sensor_type)
                total_pose_loss += pose_loss / len(sensor_type)
                total_env_loss += env_loss / len(sensor_type)
                total_anomaly_loss += anomaly_loss / len(sensor_type)
        
        return (total_temp_hum_loss / len(test_loader),
                total_motion_loss / len(test_loader),
                total_pose_loss / len(test_loader),
                total_env_loss / len(test_loader),
                total_anomaly_loss / len(test_loader))
    
    def train(self, train_dir, val_dir=None, epochs=100, learning_rate=0.001):
        """训练传感器模型 | Train sensor model
        
        Args:
            train_dir: 训练数据目录 | Training data directory
            val_dir: 验证数据目录 | Validation data directory
            epochs: 训练轮数 | Number of epochs
            learning_rate: 学习率 | Learning rate
            
        Returns:
            训练历史 | Training history
        """
        # 加载数据
        train_dataset = self.load_data(train_dir)
        train_loader = self.create_data_loader(train_dataset)
        
        if val_dir:
            val_dataset = self.load_data(val_dir)
            val_loader = self.create_data_loader(val_dataset, shuffle=False)
        else:
            val_loader = None
        
        # 优化器
        optimizer = optim.Adam([
            {'params': self.feature_extractor.parameters()},
            {'params': self.temperature_humidity_predictor.parameters()},
            {'params': self.motion_classifier.parameters()},
            {'params': self.pose_estimator.parameters()},
            {'params': self.environment_classifier.parameters()},
            {'params': self.anomaly_detector.parameters()}
        ], lr=learning_rate)
        
        # 训练历史
        history = {
            'train_temp_hum_loss': [],
            'train_motion_loss': [],
            'train_pose_loss': [],
            'train_env_loss': [],
            'train_anomaly_loss': [],
            'val_temp_hum_loss': [],
            'val_motion_loss': [],
            'val_pose_loss': [],
            'val_env_loss': [],
            'val_anomaly_loss': []
        }
        
        # 训练循环
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs} | 轮次 {epoch+1}/{epochs}")
            
            # 训练
            train_losses = self.train_epoch(train_loader, optimizer)
            history['train_temp_hum_loss'].append(train_losses[0])
            history['train_motion_loss'].append(train_losses[1])
            history['train_pose_loss'].append(train_losses[2])
            history['train_env_loss'].append(train_losses[3])
            history['train_anomaly_loss'].append(train_losses[4])
            
            # 验证
            if val_loader:
                val_losses = self.evaluate(val_loader)
                history['val_temp_hum_loss'].append(val_losses[0])
                history['val_motion_loss'].append(val_losses[1])
                history['val_pose_loss'].append(val_losses[2])
                history['val_env_loss'].append(val_losses[3])
                history['val_anomaly_loss'].append(val_losses[4])
                
                logger.info(f"验证损失 - 温湿度: {val_losses[0]:.4f}, 运动: {val_losses[1]:.4f}, "
                          f"姿态: {val_losses[2]:.4f}, 环境: {val_losses[3]:.4f}, 异常: {val_losses[4]:.4f} | "
                          f"Validation Loss - Temp/Hum: {val_losses[0]:.4f}, Motion: {val_losses[1]:.4f}, "
                          f"Pose: {val_losses[2]:.4f}, Env: {val_losses[3]:.4f}, Anomaly: {val_losses[4]:.4f}")
            
            # 保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
        
        return history
    
    def save_checkpoint(self, filepath):
        """保存模型检查点 | Save model checkpoint
        
        Args:
            filepath: 文件路径 | File path
        """
        checkpoint = {
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'temperature_humidity_predictor_state_dict': self.temperature_humidity_predictor.state_dict(),
            'motion_classifier_state_dict': self.motion_classifier.state_dict(),
            'pose_estimator_state_dict': self.pose_estimator.state_dict(),
            'environment_classifier_state_dict': self.environment_classifier.state_dict(),
            'anomaly_detector_state_dict': self.anomaly_detector.state_dict(),
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"模型检查点已保存: {filepath} | Model checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """加载模型检查点 | Load model checkpoint
        
        Args:
            filepath: 文件路径 | File path
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.temperature_humidity_predictor.load_state_dict(checkpoint['temperature_humidity_predictor_state_dict'])
        self.motion_classifier.load_state_dict(checkpoint['motion_classifier_state_dict'])
        self.pose_estimator.load_state_dict(checkpoint['pose_estimator_state_dict'])
        self.environment_classifier.load_state_dict(checkpoint['environment_classifier_state_dict'])
        self.anomaly_detector.load_state_dict(checkpoint['anomaly_detector_state_dict'])
        
        logger.info(f"模型检查点已加载: {filepath} | Model checkpoint loaded: {filepath}")
    
    def predict(self, sensor_data, sensor_type):
        """预测传感器数据 | Predict sensor data
        
        Args:
            sensor_data: 传感器数据 | Sensor data
            sensor_type: 传感器类型 | Sensor type
            
        Returns:
            预测结果 | Prediction results
        """
        self.feature_extractor.eval()
        
        with torch.no_grad():
            sensor_tensor = torch.FloatTensor(sensor_data).unsqueeze(0).to(self.device)
            features, _ = self.feature_extractor(sensor_tensor)
            features = features[:, -1, :]
            
            result = {}
            
            if sensor_type == 'temperature_humidity':
                pred = self.temperature_humidity_predictor(features)
                result['temperature'] = pred[0][0].item() * 100.0  # 反归一化
                result['humidity'] = pred[0][1].item() * 100.0
            
            elif sensor_type == 'acceleration':
                pred = self.motion_classifier(features)
                motion_states = ['stationary', 'walking', 'running', 'falling', 'jumping']
                result['motion_state'] = motion_states[torch.argmax(pred).item()]
                result['confidence'] = torch.max(pred).item()
            
            elif sensor_type == 'gyroscope':
                pred = self.pose_estimator(features)
                result['gyro_x'] = pred[0][0].item() * 10.0
                result['gyro_y'] = pred[0][1].item() * 10.0
                result['gyro_z'] = pred[0][2].item() * 10.0
                result['accel_x'] = pred[0][3].item() * 20.0
                result['accel_y'] = pred[0][4].item() * 20.0
                result['accel_z'] = pred[0][5].item() * 20.0
            
            else:
                env_pred = self.environment_classifier(features)
                environments = ['normal', 'abnormal', 'dangerous', 'unknown']
                result['environment'] = environments[torch.argmax(env_pred).item()]
                result['environment_confidence'] = torch.max(env_pred).item()
                
                anomaly_pred = self.anomaly_detector(features)
                result['anomaly_score'] = anomaly_pred.item()
                result['is_anomaly'] = anomaly_pred.item() > 0.5
            
            return result


# 主训练函数
def main():
    """主训练函数 | Main training function"""
    config = {
        'input_size': 6,  # 假设最大传感器特征维度
        'hidden_size': 128,
        'num_layers': 2,
        'output_size': 1
    }
    
    trainer = SensorModelTrainer(config)
    
    # 训练数据目录
    train_data_dir = "data/sensor/train"
    val_data_dir = "data/sensor/val"
    
    # 开始训练
    history = trainer.train(train_data_dir, val_data_dir, epochs=50, learning_rate=0.001)
    
    # 保存最终模型
    trainer.save_checkpoint("sensor_model_final.pth")
    
    logger.info("传感器模型训练完成！ | Sensor model training completed!")


if __name__ == "__main__":
    main()
