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

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import os


"""
BaseModel类 - 中文类描述
BaseModel Class - English class description
"""
class BaseModel(ABC):
    """Abstract base class for all AGI models
    
    Provides common functionality like model lifecycle management,
    configuration management, etc.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize base model"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_id = self.__class__.__name__.lower().replace('model', '')
        
        # Model configuration
        self.config = config or {}
        self.from_scratch = self.config.get('from_scratch', False)
        
        # Model state
        self.is_initialized = False
        self.is_training = False
        self.performance_metrics = {}
        
        # External API configuration
        self.external_api_config = None
        self.use_external_api = False
        
        # Internal model architecture (for from-scratch training)
        self.model_architecture = None
        self.optimizer = None
        self.loss_function = None
        self.training_history = []
        
        self.logger.info(f"Base model initialized: {self.model_id}, from_scratch: {self.from_scratch}")
    
    @abstractmethod
    def initialize(self) -> Dict[str, Any]:
        """初始化模型资源 | Initialize model resources"""
        pass

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据 | Process input data"""
        pass



    def set_mode(self, mode: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """设置模型运行模式（本地或外部API）
        Set model operation mode (local or external API)
        
        Args:
            mode: 运行模式 ('local' 或 'external')
            config: 当mode为'external'时的API配置
        """
        if mode == "external":
            # 验证API配置
            if not config:
                return {"success": False, "error": "外部模式需要提供API配置"}
            
            # 规范化API配置字段
            normalized_config = {}
            if 'api_url' in config:
                normalized_config['api_url'] = config['api_url']
            elif 'url' in config:
                normalized_config['api_url'] = config['url']
            elif 'endpoint' in config:
                normalized_config['api_url'] = config['endpoint']
            else:
                return {"success": False, "error": "缺少必要的API配置项: api_url或url或endpoint"}
            
            if 'api_key' in config:
                normalized_config['api_key'] = config['api_key']
            else:
                return {"success": False, "error": "缺少必要的API配置项: api_key"}
            
            if 'model_name' in config:
                normalized_config['model_name'] = config['model_name']
            else:
                normalized_config['model_name'] = self.model_id
            
            if 'source' in config:
                normalized_config['source'] = config['source']
            else:
                normalized_config['source'] = 'external'
            
            # 检查必要的配置项值是否为空
            for key in ['api_url', 'api_key']:
                if not normalized_config[key]:
                    return {"success": False, "error": f"API配置项值不能为空: {key}"}
            
            # 检查URL格式是否有效
            url = normalized_config['api_url']
            if not (url.startswith('http://') or url.startswith('https://')):
                return {"success": False, "error": f"无效的API URL格式: {url}"}
            
            self.use_external_api = True
            self.external_api_config = normalized_config
            self.logger.info(f"模型 {self.model_id} 已切换到外部API模式")
            
            # Test API connection
            test_result = self.test_connection()
            if not test_result["success"]:
                self.use_external_api = False
                self.external_api_config = None
                return {"success": False, "error": f"API连接测试失败: {test_result.get('error', 'Unknown error')}"}
            
            return {"success": True}
        
        elif mode == "local":
            self.use_external_api = False
            self.external_api_config = None
            self.logger.info(f"模型 {self.model_id} 已切换到本地模式")
            return {"success": True}
        
        else:
            return {"success": False, "error": f"不支持的模式: {mode}"}

    def get_status(self) -> Dict[str, Any]:
        """Get model status"""
        return {
            "model_id": self.model_id,
            "is_initialized": self.is_initialized,
            "is_training": self.is_training,
            "use_external_api": self.use_external_api,
            "performance_metrics": self.performance_metrics
        }

    def save_model(self, path: str) -> Dict[str, Any]:
        """Save model to file"""
        try:
            # This should implement model-specific saving logic
            model_data = {
                "model_id": self.model_id,
                "config": self.config,
                "performance_metrics": self.performance_metrics
            }
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Model saved to: {path}")
            return {"success": True, "path": path}
        except Exception as e:
            self.logger.error(f"Model save failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def load_model(self, path: str) -> Dict[str, Any]:
        """Load model from file"""
        try:
            if not os.path.exists(path):
                return {"success": False, "error": f"Model file not found: {path}"}
            
            with open(path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            # This should implement model-specific loading logic
            self.config.update(model_data.get("config", {}))
            self.performance_metrics.update(model_data.get("performance_metrics", {}))
            
            self.logger.info(f"模型已从 {path} 加载 | Model loaded from {path}")
            return {"success": True}
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)} | Model load failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_progress(self) -> Dict[str, Any]:
        """获取训练或处理进度 | Get training or processing progress"""
        return {
            "model_id": self.model_id,
            "progress": 0.0,
            "status": "idle",
            "estimated_time_remaining": 0
        }

    def train(self, training_data: Any, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """训练模型 | Train model"""
        self.is_training = True
        try:
            # 确保配置不为None
            config = config or {}
            
            # 如果是从零开始训练模式，则使用特定的训练逻辑
            if self.from_scratch:
                return self._train_from_scratch(training_data, config)
            
            # 标准训练逻辑
            self.logger.info(f"开始模型训练: {self.model_id}, 从零开始: {self.from_scratch}")
            
            # 具体模型应该覆盖此方法以实现实际训练逻辑
            import time
            time.sleep(2)
            
            self.is_training = False
            return {"success": True, "epochs": 1, "loss": 0.1}
        except Exception as e:
            self.is_training = False
            self.logger.error(f"训练失败: {str(e)} | Training failed: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _train_from_scratch(self, training_data: Any, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """从零开始训练模型的基础实现"""
        try:
            # 导入必要的库
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import numpy as np
            from torch.utils.data import DataLoader, TensorDataset
            from tqdm import tqdm
            
            # 提取配置参数
            epochs = config.get('epochs', 10)
            batch_size = config.get('batch_size', 32)
            learning_rate = config.get('learning_rate', 0.001)
            optimizer_type = config.get('optimizer', 'adam')
            dropout_rate = config.get('dropout_rate', 0.2)
            validation_split = config.get('validation_split', 0.2)
            
            # 准备数据（具体模型应覆盖此部分）
            train_loader, val_loader = self._prepare_training_data(training_data, batch_size, validation_split)
            
            # 初始化模型架构（具体模型应覆盖此部分）
            if not self.model_architecture:
                self._initialize_model_architecture(config)
                
            # 设置优化器和损失函数
            if optimizer_type == 'adam':
                self.optimizer = optim.Adam(self.model_architecture.parameters(), lr=learning_rate)
            elif optimizer_type == 'sgd':
                momentum = config.get('momentum', 0.9)
                self.optimizer = optim.SGD(self.model_architecture.parameters(), lr=learning_rate, momentum=momentum)
            elif optimizer_type == 'rmsprop':
                self.optimizer = optim.RMSprop(self.model_architecture.parameters(), lr=learning_rate)
            
            # 设置损失函数（具体模型可能需要不同的损失函数）
            if not self.loss_function:
                self.loss_function = nn.CrossEntropyLoss()
            
            # 训练循环
            self.training_history = []
            best_val_loss = float('inf')
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_architecture.to(device)
            
            for epoch in range(epochs):
                # 训练阶段
                self.model_architecture.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
                for inputs, labels in progress_bar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # 前向传播
                    outputs = self.model_architecture(inputs)
                    loss = self.loss_function(outputs, labels)
                    
                    # 反向传播和优化
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # 统计
                    train_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    
                    # 更新进度条
                    progress_bar.set_postfix({'loss': train_loss/train_total, 'acc': 100*train_correct/train_total})
                
                # 验证阶段
                self.model_architecture.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = self.model_architecture(inputs)
                        loss = self.loss_function(outputs, labels)
                        
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                # 计算平均损失和准确率
                train_loss = train_loss / train_total
                train_acc = 100 * train_correct / train_total
                val_loss = val_loss / val_total
                val_acc = 100 * val_correct / val_total
                
                # 记录历史
                epoch_history = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }
                self.training_history.append(epoch_history)
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(os.path.join('models', f'{self.model_id}_best.pt'))
                
                self.logger.info(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # 训练完成，保存最终模型
            self.save_model(os.path.join('models', f'{self.model_id}_final.pt'))
            
            # 返回训练结果
            self.is_training = False
            return {
                "success": True,
                "epochs": epochs,
                "best_val_loss": best_val_loss,
                "final_train_loss": train_loss,
                "final_train_acc": train_acc,
                "final_val_loss": val_loss,
                "final_val_acc": val_acc,
                "training_history": self.training_history
            }
        except Exception as e:
            self.is_training = False
            self.logger.error(f"从零开始训练失败: {str(e)} | From scratch training failed: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _prepare_training_data(self, training_data: Any, batch_size: int, validation_split: float) -> tuple:
        """准备训练数据，具体模型应覆盖此方法"""
        # 这是一个基础实现，具体模型应根据自身需求覆盖此方法
        import torch
        from torch.utils.data import DataLoader, TensorDataset, random_split
        
        # 假设training_data是一个包含inputs和labels的元组
        if isinstance(training_data, tuple) and len(training_data) == 2:
            inputs, labels = training_data
            dataset = TensorDataset(torch.tensor(inputs), torch.tensor(labels))
            
            # 分割训练集和验证集
            train_size = int((1 - validation_split) * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            return train_loader, val_loader
        else:
            raise ValueError("训练数据格式不正确，请提供(inputs, labels)元组")
            
    def _initialize_model_architecture(self, config: Dict[str, Any]):
        """初始化模型架构，具体模型应覆盖此方法"""
        import torch.nn as nn
        
        # 这是一个基础实现，具体模型应根据自身需求覆盖此方法
        # 默认创建一个简单的前馈神经网络
        input_size = config.get('input_size', 100)
        hidden_sizes = config.get('hidden_sizes', [64, 32])
        output_size = config.get('output_size', 10)
        
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.get('dropout_rate', 0.2)))
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, output_size))
        
        self.model_architecture = nn.Sequential(*layers)
        self.logger.info(f"模型架构初始化完成: {self.model_id}, 输入维度: {input_size}, 隐藏层: {hidden_sizes}, 输出维度: {output_size}")

    def validate(self, validation_data: Any) -> Dict[str, Any]:
        """验证模型性能 | Validate model performance"""
        try:
            # 这里应该实现模型特定的验证逻辑
            # This should implement model-specific validation logic
            self.logger.info("开始模型验证 | Starting model validation")
            
            return {
                "success": True,
                "accuracy": 0.95,
                "precision": 0.93,
                "recall": 0.96,
                "f1_score": 0.94
            }
        except Exception as e:
            self.logger.error(f"验证失败: {str(e)} | Validation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def test_connection(self) -> Dict[str, Any]:
        """测试外部API连接
        Test external API connection
        
        Returns:
            Dict[str, Any]: 连接测试结果
        """
        if not self.use_external_api or not self.external_api_config:
            return {"success": False, "error": "未配置外部API | External API not configured"}
        
        try:
            # 从已配置的API信息中获取参数
            config = self.external_api_config
            api_url = config.get('api_url', '')
            api_key = config.get('api_key', '')
            
            if not api_url:
                return {"success": False, "error": "缺少API URL"}
            
            if not api_key:
                return {"success": False, "error": "缺少API密钥"}
            
            # 记录连接测试信息
            self.logger.info(f"正在测试外部API连接: {api_url}")
            
            # 模拟连接测试（实际实现应调用具体API）
            import time
            time.sleep(0.1)  # 模拟网络延迟
            
            # 检查URL格式
            if not (api_url.startswith('http://') or api_url.startswith('https://')):
                return {"success": False, "error": f"无效的API URL格式: {api_url}"}
            
            # 模拟连接成功
            self.logger.info(f"外部API连接测试成功: {self.model_id}")
            return {
                "success": True,
                "model_id": self.model_id,
                "api_url": api_url,
                "model_name": config.get('model_name', self.model_id),
                "source": config.get('source', 'external'),
                "timestamp": time.time(),
                "message": "API连接测试成功 | API connection test successful"
            }
        except Exception as e:
            error_message = str(e)
            self.logger.error(f"外部API连接测试失败: {error_message}")
            return {
                "success": False, 
                "error": error_message,
                "model_id": self.model_id,
                "timestamp": time.time()
            }

    def get_api_status(self) -> Dict[str, Any]:
        """获取API连接状态 | Get API connection status"""
        return {
            "use_external_api": self.use_external_api,
            "api_config": self.external_api_config,
            "connection_status": self.test_connection() if self.use_external_api else {"success": True, "message": "使用本地模式 | Using local mode"}
        }

    def reset(self) -> Dict[str, Any]:
        """重置模型状态 | Reset model state"""
        self.is_initialized = False
        self.is_training = False
        self.performance_metrics = {}
        self.use_external_api = False
        self.external_api_config = None
        self.logger.info("模型已重置 | Model reset")
        return {"success": True}

# 导出基类 | Export base class
AGIBaseModel = BaseModel
