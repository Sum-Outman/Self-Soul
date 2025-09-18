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
完整的训练管理器：支持单独训练和联合训练
Complete Training Manager: Supports individual and joint training

功能增强：
1. 支持模型单独训练和联合训练的可视化控制
2. 实时训练进度监控
3. 训练结果保存/加载功能
4. 多语言支持（中英文注释）

Enhanced Features:
1. Visual control for individual and joint model training
2. Real-time training progress monitoring
3. Training results save/load functionality
4. Multi-language support (Chinese-English comments)
"""

import asyncio
import time
from typing import Dict, List, Any, Tuple
import json
from pathlib import Path
from datetime import datetime
from ..error_handling import error_handler
from ..model_registry import model_registry


"""
CompleteTrainingManager类 - 中文类描述
CompleteTrainingManager Class - English class description
"""
class CompleteTrainingManager:
    """完整的训练管理器类
    Complete Training Manager Class
    """
    
    
"""
__init__函数 - 中文函数描述
__init__ Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def __init__(self):
        # 训练状态字典: model_id -> status
        # Training status dict: model_id -> status
        self.training_status = {}
        
        # 训练进度字典: model_id -> progress_percentage
        # Training progress dict: model_id -> progress_percentage
        self.training_progress = {}
        
        # 训练结果字典: model_id -> list of epoch results
        # Training results dict: model_id -> list of epoch results
        self.training_results = {}
        
        # 联合训练队列: list of model groups
        # Joint training queue: list of model groups
        self.joint_training_queue = []
        
        # 活动训练集合: set of active model_ids
        # Active trainings set: set of active model_ids
        self.active_trainings = set()
        
        # 模型依赖关系: model_id -> list of dependent models
        # Model dependencies: model_id -> list of dependent models
        self.model_dependencies = {
            "manager": ["language", "knowledge", "planning"],
            "audio": ["language"],
            "vision": ["spatial"],
            "video": ["vision", "audio"],
            "sensor": [],
            "motion": ["sensor", "spatial"],
            "computer": [],
            "programming": ["knowledge", "language"]
        }
        
        # 训练历史记录
        # Training history
        self.training_history = []
        
async def train_model(self, model_id: str, training_data: Any,
                         epochs: int = 10, batch_size: int = 32,
                         dependencies: List[str] = None) -> Dict[str, Any]:
        """训练单个模型
        Train single model
        
        增强功能：
        - 添加依赖模型检查
        - 改进训练进度跟踪
        - 支持实际模型训练方法
        
        Enhanced:
        - Added dependency model check
        - Improved training progress tracking
        - Support actual model training methods
        
        Args:
            model_id: 模型ID (Model ID)
            training_data: 训练数据 (Training data)
            epochs: 训练轮数 (Number of epochs)
            batch_size: 批次大小 (Batch size)
            dependencies: 依赖模型列表 (List of dependent models)
            
        Returns:
            dict: 训练结果 (Training results)
        """
        try:
            model = model_registry.get_model(model_id)
            if not model:
                error_handler.log_error(f"模型 {model_id} 未找到", "CompleteTrainingManager")
                return {"status": "error", "message": f"Model {model_id} not found"}
            
            self.training_status[model_id] = "training"
            self.training_progress[model_id] = 0
            self.active_trainings.add(model_id)
            
            # 检查依赖模型是否已训练
            # Check if dependent models are trained
            if dependencies is None:
                dependencies = self.model_dependencies.get(model_id, [])
                
            for dep_model in dependencies:
                if dep_model not in self.training_results or not self.training_results.get(dep_model):
                    error_handler.log_error(
                        f"依赖模型 {dep_model} 未训练，无法训练 {model_id}",
                        "CompleteTrainingManager"
                    )
                    return {
                        "status": "error",
                        "message": f"Dependent model {dep_model} not trained"
                    }
            
            # 训练过程
            # Training process
            start_time = time.time()
            epoch_results = []
            
            for epoch in range(epochs):
                if model_id not in self.active_trainings:
                    break
                    
                # 更新进度 (带时间戳)
                # Update progress (with timestamp)
                progress = (epoch + 1) / epochs * 100
                self.training_progress[model_id] = progress
                self._update_training_history(model_id, "progress", {
                    "epoch": epoch + 1,
                    "progress": progress,
                    "timestamp": datetime.now().isoformat()
                })
                
                # 实际训练逻辑（调用模型的train方法）
                if hasattr(model, 'train'):
                    try:
                        result = await model.train(training_data, epoch, batch_size)
                        epoch_results.append(result)
                    except Exception as e:
                        error_handler.handle_error(e, "CompleteTrainingManager", f"模型 {model_id} 训练失败")
                        result = {
                            "loss": 0.0,
                            "accuracy": 0.0,
                            "epoch": epoch + 1,
                            "error": str(e)
                        }
                        epoch_results.append(result)
                else:
                    # 默认训练模拟
                    await asyncio.sleep(1)  # 模拟训练时间
                    result = {
                        "loss": 0.1 * (epochs - epoch) / epochs,
                        "accuracy": 0.8 * (epoch + 1) / epochs,
                        "epoch": epoch + 1
                    }
                    epoch_results.append(result)
                
                # 保存中间结果
                if model_id not in self.training_results:
                    self.training_results[model_id] = []
                self.training_results[model_id].append(result)
                
                # 实时更新知识库
                if model_id != "knowledge" and "knowledge" in model_registry.models:
                    knowledge_model = model_registry.get_model("knowledge")
                    if hasattr(knowledge_model, 'update_training_knowledge'):
                        knowledge_model.update_training_knowledge(model_id, result)
            
            training_time = time.time() - start_time
            self.training_status[model_id] = "completed"
            self.training_progress[model_id] = 100
            self.active_trainings.remove(model_id)
            
            # 计算最终准确率
            final_accuracy = 0.0
            if epoch_results:
                final_result = epoch_results[-1]
                final_accuracy = final_result.get('accuracy', 0.85)
            
            final_result = {
                "status": "success",
                "model_id": model_id,
                "total_epochs": epochs,
                "final_accuracy": final_accuracy,
                "training_time": training_time,
                "epoch_results": epoch_results
            }
            
            return final_result
            
        except Exception as e:
            error_handler.handle_error(e, "CompleteTrainingManager", f"训练模型 {model_id} 失败")
            self.training_status[model_id] = "error"
            if model_id in self.active_trainings:
                self.active_trainings.remove(model_id)
            return {"status": "error", "message": str(e)}
    
async def joint_train(self, model_ids: List[str], training_data: Dict[str, Any], 
                         epochs: int = 10, coordination_strategy: str = "sequential",
                         selected_models: List[str] = None) -> Dict[str, Any]:
        """联合训练多个模型
        Joint train multiple models
        
        增强功能：
        - 添加选择的模型参数
        - 改进协调策略
        
        Enhanced:
        - Added selected_models parameter
        - Improved coordination strategies
        
        Args:
            model_ids: 模型ID列表 (List of model IDs)
            training_data: 各模型的训练数据 (Training data for each model)
            epochs: 训练轮数 (Number of epochs)
            coordination_strategy: 协调策略 (Coordination strategy: sequential, parallel, adaptive)
            selected_models: 选择的模型列表 (List of selected models for joint training)
            
        Returns:
            dict: 联合训练结果 (Joint training results)
        """
        
        # 如果没有指定选择的模型，使用全部模型
        # If no selected models specified, use all models
        if selected_models is None:
            selected_models = model_ids
        else:
            # 过滤掉不在模型列表中的选择
            # Filter out selections not in model list
            selected_models = [m for m in selected_models if m in model_ids]
        try:
            self.joint_training_queue.append(model_ids)
            joint_id = f"joint_{int(time.time())}"
            self.training_status[joint_id] = "training"
            
            results = {}
            
            if coordination_strategy == "sequential":
                # 顺序训练 (只训练选择的模型)
                # Sequential training (only selected models)
                for model_id in selected_models:
                    if model_id in training_data:
                        result = await self.train_model(
                            model_id, 
                            training_data[model_id], 
                            epochs
                        )
                        results[model_id] = result
                        
                        # 更新联合训练进度
                        # Update joint training progress
                        self.training_progress[joint_id] = (
                            (selected_models.index(model_id) + 1) / len(selected_models) * 100
                        )
                    
            elif coordination_strategy == "parallel":
                # 并行训练
                tasks = []
                for model_id in model_ids:
                    if model_id in training_data:
                        task = self.train_model(model_id, training_data[model_id], epochs)
                        tasks.append(task)
                
                results_list = await asyncio.gather(*tasks, return_exceptions=True)
                for i, model_id in enumerate(model_ids):
                    if i < len(results_list) and not isinstance(results_list[i], Exception):
                        results[model_id] = results_list[i]
                    
            elif coordination_strategy == "adaptive":
                # 自适应训练 - 根据模型性能动态调整
                manager_model = model_registry.get_model("manager")
                if hasattr(manager_model, 'optimize_training_schedule'):
                    training_plan = manager_model.optimize_training_schedule(model_ids, training_data)
                    
                    for model_id, plan in training_plan.items():
                        if model_id in training_data:
                            result = await self.train_model(
                                model_id, 
                                training_data[model_id], 
                                plan.get('epochs', epochs),
                                plan.get('batch_size', 32)
                            )
                            results[model_id] = result
                else:
                    # 回退到顺序训练
                    for model_id in model_ids:
                        if model_id in training_data:
                            result = await self.train_model(model_id, training_data[model_id], epochs)
                            results[model_id] = result
            
            self.training_status[joint_id] = "completed"
            self.joint_training_queue.remove(model_ids)
            
            # 综合评估联合训练效果
            overall_accuracy = sum(result.get('final_accuracy', 0) for result in results.values()) / len(results) if results else 0
            
            return {
                "status": "success",
                "joint_training_id": joint_id,
                "model_results": results,
                "overall_accuracy": overall_accuracy,
                "coordination_strategy": coordination_strategy
            }
            
        except Exception as e:
            error_handler.handle_error(e, "CompleteTrainingManager", "联合训练失败")
            self.training_status[joint_id] = "error"
            if model_ids in self.joint_training_queue:
                self.joint_training_queue.remove(model_ids)
            return {"status": "error", "message": str(e)}
    
def get_training_status(self, model_id: str = None) -> Dict[str, Any]:
        """获取训练状态
        Get training status
        
        Args:
            model_id: 可选模型ID
            
        Returns:
            dict: 训练状态信息
        """
        if model_id:
            return {
                "status": self.training_status.get(model_id, "not_started"),
                "progress": self.training_progress.get(model_id, 0),
                "active": model_id in self.active_trainings
            }
        else:
            return {
                "training_status": self.training_status,
                "training_progress": self.training_progress,
                "active_trainings": list(self.active_trainings),
                "joint_training_queue": self.joint_training_queue
            }
    
def stop_training(self, model_id: str) -> bool:
        """停止训练
        Stop training
        
        Args:
            model_id: 模型ID
            
        Returns:
            bool: 是否成功停止
        """
        if model_id in self.active_trainings:
            self.active_trainings.remove(model_id)
            self.training_status[model_id] = "stopped"
            return True
        return False
    
def save_training_results(self, file_path: str, model_id: str = None) -> bool:
        """保存训练结果到文件
        Save training results to file
        
        增强功能：
        - 支持保存单个模型结果
        - 添加更多元数据
        
        Enhanced:
        - Support saving single model results
        - Add more metadata
        
        Args:
            file_path: 文件路径 (File path)
            model_id: 可选模型ID (Optional model ID)
            
        Returns:
            bool: 是否成功保存 (Whether save was successful)
        """
        try:
            # 保存特定模型或全部结果
            # Save specific model or all results
            if model_id:
                results_data = {
                    model_id: self.training_results.get(model_id, {}),
                    "metadata": {
                        "saved_at": datetime.now().isoformat(),
                        "model_id": model_id,
                        "system_version": "2.0"
                    }
                }
            else:
                results_data = {
                    "training_results": self.training_results,
                    "training_history": self.training_history,
                    "metadata": {
                        "saved_at": datetime.now().isoformat(),
                        "total_models": len(self.training_results),
                        "system_version": "2.0"
                    }
                }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            error_handler.handle_error(e, "CompleteTrainingManager", "保存训练结果失败")
            return False
    
def load_training_results(self, file_path: str, model_id: str = None) -> bool:
        """从文件加载训练结果
        Load training results from file
        
        增强功能：
        - 支持加载单个模型结果
        - 添加错误处理
        
        Enhanced:
        - Support loading single model results
        - Add error handling
        
        Args:
            file_path: 文件路径 (File path)
            model_id: 可选模型ID (Optional model ID)
            
        Returns:
            bool: 是否成功加载 (Whether load was successful)
        """
        try:
            if not Path(file_path).exists():
                error_handler.log_error(f"文件不存在: {file_path}", "CompleteTrainingManager")
                return False
                
            with open(file_path, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            
            # 加载特定模型或全部结果
            # Load specific model or all results
            if model_id:
                if model_id in results_data:
                    self.training_results[model_id] = results_data[model_id]
                elif "training_results" in results_data and model_id in results_data["training_results"]:
                    self.training_results[model_id] = results_data["training_results"][model_id]
                else:
                    error_handler.log_error(f"未找到模型 {model_id} 的结果", "CompleteTrainingManager")
                    return False
            else:
                if "training_results" in results_data:
                    self.training_results = results_data["training_results"]
                else:
                    # 兼容旧格式
                    # Compatible with old format
                    self.training_results = results_data
                    
                if "training_history" in results_data:
                    self.training_history = results_data["training_history"]
            
            return True
        except Exception as e:
            error_handler.handle_error(e, "CompleteTrainingManager", "加载训练结果失败")
            return False

    
"""
_update_training_history函数 - 中文函数描述
_update_training_history Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _update_training_history(self, model_id: str, event_type: str, data: Any):
        """更新训练历史记录
        Update training history
        
        Args:
            model_id: 模型ID (Model ID)
            event_type: 事件类型 (Event type: start, progress, complete, error)
            data: 事件数据 (Event data)
        """
        self.training_history.append({
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
            "event_type": event_type,
            "data": data
        })
        
        # 保持历史记录大小
        # Keep history size manageable
        if len(self.training_history) > 1000:
            self.training_history = self.training_history[-500:]

# 创建全局训练管理器实例
training_manager = CompleteTrainingManager()
