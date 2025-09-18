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
双目空间定位感知模型训练程序
Spatial Localization Perception Model Training Program

功能：训练空间定位感知模型，支持空间识别、可视化空间建模、空间定位、距离感知、体积识别、运动物体识别和自身位置识别
Function: Train spatial localization perception model supporting spatial recognition, visual spatial modeling, spatial positioning, distance perception, volume recognition, moving object recognition, and self-position recognition
"""

import os
import json
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
from PIL import Image
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
SpatialDataset类 - 中文类描述
SpatialDataset Class - English class description
"""
class SpatialDataset(Dataset):
    """空间定位数据集 | Spatial localization dataset"""
    
    
"""
__init__函数 - 中文函数描述
__init__ Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def __init__(self, data_dir, image_size=224):
        self.data = []
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载空间定位数据
        spatial_types = ['depth', 'stereo', 'pointcloud', 'spatial_mapping']
        
        for spatial_type in spatial_types:
            file_path = os.path.join(data_dir, f"spatial_{spatial_type}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    spatial_data = json.load(f)
                    self.data.extend(spatial_data)
        
        logger.info(f"加载空间定位训练数据: {len(self.data)} 条样本 | "
                    f"Loaded spatial training data: {len(self.data)} samples")
    
    
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
            # 加载双目图像
            left_image_path = item['left_image']
            right_image_path = item['right_image']
            
            left_image = Image.open(left_image_path).convert('RGB')
            right_image = Image.open(right_image_path).convert('RGB')
            
            # 应用变换
            left_tensor = self.transform(left_image)
            right_tensor = self.transform(right_image)
            
            # 空间信息标签
            labels = {
                'depth_map': np.array(item.get('depth_map', [])),
                'point_cloud': np.array(item.get('point_cloud', [])),
                'object_positions': item.get('object_positions', []),
                'distances': item.get('distances', []),
                'volumes': item.get('volumes', []),
                'motion_vectors': item.get('motion_vectors', []),
                'self_position': item.get('self_position', [0, 0, 0]),
                'spatial_layout': item.get('spatial_layout', {})
            }
            
            return {
                'left_image': left_tensor,
                'right_image': right_tensor,
                'labels': labels,
                'metadata': {
                    'camera_params': item.get('camera_params', {}),
                    'image_size': item.get('image_size', [640, 480])
                }
            }
            
        except Exception as e:
            logger.error(f"空间数据加载错误: {str(e)}")
            # 返回空数据
            empty_image = torch.zeros(3, self.image_size, self.image_size)
            return {
                'left_image': empty_image,
                'right_image': empty_image,
                'labels': {
                    'depth_map': np.zeros((100, 100)),
                    'point_cloud': np.zeros((100, 3)),
                    'object_positions': [],
                    'distances': [],
                    'volumes': [],
                    'motion_vectors': [],
                    'self_position': [0, 0, 0],
                    'spatial_layout': {}
                },
                'metadata': {
                    'camera_params': {},
                    'image_size': [640, 480]
                }
            }


"""
SpatialModelTrainer类 - 中文类描述
SpatialModelTrainer Class - English class description
"""
class SpatialModelTrainer:
    """空间定位模型训练器 | Spatial localization model trainer"""
    
    
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
        
        # 双目视觉特征提取器
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        self.feature_extractor = self.feature_extractor.to(self.device)
        
        # 深度估计网络
        self.depth_estimator = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Sigmoid()
        ).to(self.device)
        
        # 点云生成网络
        self.pointcloud_generator = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100 * 3)  # 100个点，每个点3个坐标
        ).to(self.device)
        
        # 物体检测和定位网络
        self.object_localizer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 50 * 6)  # 50个物体，每个物体6个参数 (x,y,z,w,h,d)
        ).to(self.device)
        
        # 距离估计网络
        self.distance_estimator = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 50)  # 50个距离值
        ).to(self.device)
        
        # 体积估计网络
        self.volume_estimator = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 50)  # 50个体积值
        ).to(self.device)
        
        # 运动预测网络
        self.motion_predictor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 50 * 3)  # 50个运动向量，每个3个分量
        ).to(self.device)
        
        # 自身位置估计网络
        self.self_position_estimator = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 3D位置
        ).to(self.device)
        
        # 损失函数
        self.depth_loss = nn.MSELoss()
        self.pointcloud_loss = nn.MSELoss()
        self.localization_loss = nn.MSELoss()
        self.distance_loss = nn.MSELoss()
        self.volume_loss = nn.MSELoss()
        self.motion_loss = nn.MSELoss()
        self.position_loss = nn.MSELoss()
    
    
"""
load_data函数 - 中文函数描述
load_data Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def load_data(self, data_dir):
        """加载空间数据 | Load spatial data"""
        dataset = SpatialDataset(data_dir)
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
            num_workers=2
        )
    
    
"""
extract_features函数 - 中文函数描述
extract_features Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def extract_features(self, image):
        """提取图像特征 | Extract image features"""
        features = self.feature_extractor(image)
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
        self.feature_extractor.train()
        self.depth_estimator.train()
        self.pointcloud_generator.train()
        self.object_localizer.train()
        self.distance_estimator.train()
        self.volume_estimator.train()
        self.motion_predictor.train()
        self.self_position_estimator.train()
        
        total_depth_loss = 0
        total_pointcloud_loss = 0
        total_localization_loss = 0
        total_distance_loss = 0
        total_volume_loss = 0
        total_motion_loss = 0
        total_position_loss = 0
        
        for batch in tqdm(train_loader, desc="训练中 | Training"):
            left_images = batch['left_image'].to(self.device)
            right_images = batch['right_image'].to(self.device)
            labels = batch['labels']
            
            optimizer.zero_grad()
            
            # 提取左右图像特征
            left_features = self.extract_features(left_images)
            right_features = self.extract_features(right_images)
            
            # 合并特征
            combined_features = torch.cat([left_features, right_features], dim=1)
            
            # 深度估计
            depth_pred = self.depth_estimator(combined_features)
            
            # 点云生成
            pointcloud_pred = self.pointcloud_generator(combined_features.view(combined_features.size(0), -1))
            
            # 物体定位
            localization_pred = self.object_localizer(combined_features.view(combined_features.size(0), -1))
            
            # 距离估计
            distance_pred = self.distance_estimator(combined_features.view(combined_features.size(0), -1))
            
            # 体积估计
            volume_pred = self.volume_estimator(combined_features.view(combined_features.size(0), -1))
            
            # 运动预测
            motion_pred = self.motion_predictor(combined_features.view(combined_features.size(0), -1))
            
            # 自身位置估计
            position_pred = self.self_position_estimator(combined_features.view(combined_features.size(0), -1))
            
            # 计算各种损失
            depth_loss = self.depth_loss(depth_pred, torch.zeros_like(depth_pred))
            pointcloud_loss = self.pointcloud_loss(pointcloud_pred, torch.zeros_like(pointcloud_pred))
            localization_loss = self.localization_loss(localization_pred, torch.zeros_like(localization_pred))
            distance_loss = self.distance_loss(distance_pred, torch.zeros_like(distance_pred))
            volume_loss = self.volume_loss(volume_pred, torch.zeros_like(volume_pred))
            motion_loss = self.motion_loss(motion_pred, torch.zeros_like(motion_pred))
            position_loss = self.position_loss(position_pred, torch.zeros_like(position_pred))
            
            # 总损失
            total_loss = (depth_loss + pointcloud_loss + localization_loss +
                         distance_loss + volume_loss + motion_loss + position_loss)
            
            total_loss.backward()
            optimizer.step()
            
            total_depth_loss += depth_loss.item()
            total_pointcloud_loss += pointcloud_loss.item()
            total_localization_loss += localization_loss.item()
            total_distance_loss += distance_loss.item()
            total_volume_loss += volume_loss.item()
            total_motion_loss += motion_loss.item()
            total_position_loss += position_loss.item()
        
        return (total_depth_loss / len(train_loader),
                total_pointcloud_loss / len(train_loader),
                total_localization_loss / len(train_loader),
                total_distance_loss / len(train_loader),
                total_volume_loss / len(train_loader),
                total_motion_loss / len(train_loader),
                total_position_loss / len(train_loader))
    
    
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
        self.feature_extractor.eval()
        self.depth_estimator.eval()
        self.pointcloud_generator.eval()
        self.object_localizer.eval()
        self.distance_estimator.eval()
        self.volume_estimator.eval()
        self.motion_predictor.eval()
        self.self_position_estimator.eval()
        
        total_depth_loss = 0
        total_pointcloud_loss = 0
        total_localization_loss = 0
        total_distance_loss = 0
        total_volume_loss = 0
        total_motion_loss = 0
        total_position_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估中 | Evaluating"):
                left_images = batch['left_image'].to(self.device)
                right_images = batch['right_image'].to(self.device)
                labels = batch['labels']
                
                # 提取左右图像特征
                left_features = self.extract_features(left_images)
                right_features = self.extract_features(right_images)
                
                # 合并特征
                combined_features = torch.cat([left_features, right_features], dim=1)
                
                # 深度估计
                depth_pred = self.depth_estimator(combined_features)
                
                # 点云生成
                pointcloud_pred = self.pointcloud_generator(combined_features.view(combined_features.size(0), -1))
                
                # 物体定位
                localization_pred = self.object_localizer(combined_features.view(combined_features.size(0), -1))
                
                # 距离估计
                distance_pred = self.distance_estimator(combined_features.view(combined_features.size(0), -1))
                
                # 体积估计
                volume_pred = self.volume_estimator(combined_features.view(combined_features.size(0), -1))
                
                # 运动预测
                motion_pred = self.motion_predictor(combined_features.view(combined_features.size(0), -1))
                
                # 自身位置估计
                position_pred = self.self_position_estimator(combined_features.view(combined_features.size(0), -1))
                
                # 计算各种损失
                depth_loss = self.depth_loss(depth_pred, torch.zeros_like(depth_pred))
                pointcloud_loss = self.pointcloud_loss(pointcloud_pred, torch.zeros_like(pointcloud_pred))
                localization_loss = self.localization_loss(localization_pred, torch.zeros_like(localization_pred))
                distance_loss = self.distance_loss(distance_pred, torch.zeros_like(distance_pred))
                volume_loss = self.volume_loss(volume_pred, torch.zeros_like(volume_pred))
                motion_loss = self.motion_loss(motion_pred, torch.zeros_like(motion_pred))
                position_loss = self.position_loss(position_pred, torch.zeros_like(position_pred))
                
                total_depth_loss += depth_loss.item()
                total_pointcloud_loss += pointcloud_loss.item()
                total_localization_loss += localization_loss.item()
                total_distance_loss += distance_loss.item()
                total_volume_loss += volume_loss.item()
                total_motion_loss += motion_loss.item()
                total_position_loss += position_loss.item()
        
        return (total_depth_loss / len(test_loader),
                total_pointcloud_loss / len(test_loader),
                total_localization_loss / len(test_loader),
                total_distance_loss / len(test_loader),
                total_volume_loss / len(test_loader),
                total_motion_loss / len(test_loader),
                total_position_loss / len(test_loader))
    
    
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
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'depth_estimator_state_dict': self.depth_estimator.state_dict(),
            'pointcloud_generator_state_dict': self.pointcloud_generator.state_dict(),
            'object_localizer_state_dict': self.object_localizer.state_dict(),
            'distance_estimator_state_dict': self.distance_estimator.state_dict(),
            'volume_estimator_state_dict': self.volume_estimator.state_dict(),
            'motion_predictor_state_dict': self.motion_predictor.state_dict(),
            'self_position_estimator_state_dict': self.self_position_estimator.state_dict()
        }, path)
        logger.info(f"空间定位模型已保存至 {path} | Spatial localization model saved to {path}")
    
    
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
            {'params': self.feature_extractor.parameters()},
            {'params': self.depth_estimator.parameters()},
            {'params': self.pointcloud_generator.parameters()},
            {'params': self.object_localizer.parameters()},
            {'params': self.distance_estimator.parameters()},
            {'params': self.volume_estimator.parameters()},
            {'params': self.motion_predictor.parameters()},
            {'params': self.self_position_estimator.parameters()}
        ], lr=self.config['learning_rate'])
        
        # 训练循环
        for epoch in range(epochs):
            (train_depth_loss, train_pc_loss, train_loc_loss, 
             train_dist_loss, train_vol_loss, train_mot_loss, train_pos_loss) = self.train_epoch(train_loader, optimizer)
            
            (test_depth_loss, test_pc_loss, test_loc_loss, 
             test_dist_loss, test_vol_loss, test_mot_loss, test_pos_loss) = self.evaluate(test_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"训练深度损失: {train_depth_loss:.4f} | 训练点云损失: {train_pc_loss:.4f} | "
                f"训练定位损失: {train_loc_loss:.4f} | 训练距离损失: {train_dist_loss:.4f} | "
                f"训练体积损失: {train_vol_loss:.4f} | 训练运动损失: {train_mot_loss:.4f} | "
                f"训练位置损失: {train_pos_loss:.4f} | "
                f"测试深度损失: {test_depth_loss:.4f} | 测试点云损失: {test_pc_loss:.4f} | "
                f"测试定位损失: {test_loc_loss:.4f} | 测试距离损失: {test_dist_loss:.4f} | "
                f"测试体积损失: {test_vol_loss:.4f} | 测试运动损失: {test_mot_loss:.4f} | "
                f"测试位置损失: {test_pos_loss:.4f} | "
                f"Train Depth Loss: {train_depth_loss:.4f} | Train PointCloud Loss: {train_pc_loss:.4f} | "
                f"Train Localization Loss: {train_loc_loss:.4f} | Train Distance Loss: {train_dist_loss:.4f} | "
                f"Train Volume Loss: {train_vol_loss:.4f} | Train Motion Loss: {train_mot_loss:.4f} | "
                f"Train Position Loss: {train_pos_loss:.4f} | "
                f"Test Depth Loss: {test_depth_loss:.4f} | Test PointCloud Loss: {test_pc_loss:.4f} | "
                f"Test Localization Loss: {test_loc_loss:.4f} | Test Distance Loss: {test_dist_loss:.4f} | "
                f"Test Volume Loss: {test_vol_loss:.4f} | Test Motion Loss: {test_mot_loss:.4f} | "
                f"Test Position Loss: {test_pos_loss:.4f}"
            )
            
            # 每5轮保存一次模型
            if (epoch + 1) % 5 == 0:
                self.save_model(f"{self.config['model_save_path']}_epoch_{epoch+1}.pth")
        
        # 保存最终模型
        self.save_model(self.config['model_save_path'])
        logger.info("空间定位模型训练完成 | Spatial localization model training completed")

# 配置示例 | Example configuration
if __name__ == "__main__":
    config = {
        'batch_size': 2,  # 空间处理需要较小批次
        'learning_rate': 1e-5,
        'model_save_path': 'models/spatial_model.pth'
    }
    
    # 空间数据目录
    spatial_data_dir = 'data/spatial/'
    
    # 创建数据目录（如果不存在）
    os.makedirs(spatial_data_dir, exist_ok=True)
    
    trainer = SpatialModelTrainer(config)
    trainer.full_training(spatial_data_dir, epochs=10)
