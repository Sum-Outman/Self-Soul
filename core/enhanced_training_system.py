"""
增强型模型训练系统
Enhanced Model Training System

提供完整的模型训练功能，支持从零开始训练、联合训练、自主学习和真实有效的模型训练
Provides complete model training functionality with support for from-scratch training, joint training, autonomous learning, and real effective model training
"""

import asyncio
import json
import time
import uuid
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import threading
import queue

from core.model_registry import get_model_registry
from core.training_manager import TrainingManager
from core.joint_training_coordinator import JointTrainingCoordinator
from core.self_learning import AGISelfLearningSystem
from core.external_api_service import ExternalAPIService

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTrainingSystem:
    """增强型模型训练系统"""
    
    def __init__(self):
        self.model_registry = get_model_registry()
        self.training_manager = TrainingManager()
        self.joint_training_coordinator = None  # Will be initialized when needed
        self.autonomous_learning_manager = AGISelfLearningSystem()
        self.external_api_service = ExternalAPIService()
        
        # 训练状态跟踪
        self.active_trainings: Dict[str, Dict[str, Any]] = {}
        self.training_results: Dict[str, Dict[str, Any]] = {}
        
        # 训练数据存储
        self.training_data_dir = Path("./data/training")
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型训练配置
        self.default_training_configs = self._initialize_default_configs()
        
        logger.info("EnhancedTrainingSystem initialized successfully")
    
    def _initialize_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """初始化默认训练配置"""
        return {
            "language": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "loss_function": "cross_entropy",
                "validation_split": 0.2,
                "early_stopping_patience": 10
            },
            "vision": {
                "epochs": 50,
                "batch_size": 16,
                "learning_rate": 0.0001,
                "optimizer": "adam",
                "loss_function": "mse",
                "validation_split": 0.2,
                "early_stopping_patience": 5
            },
            "audio": {
                "epochs": 80,
                "batch_size": 8,
                "learning_rate": 0.0005,
                "optimizer": "adam",
                "loss_function": "binary_cross_entropy",
                "validation_split": 0.15,
                "early_stopping_patience": 8
            },
            "sensor": {
                "epochs": 30,
                "batch_size": 64,
                "learning_rate": 0.01,
                "optimizer": "sgd",
                "loss_function": "mse",
                "validation_split": 0.1,
                "early_stopping_patience": 3
            },
            "spatial": {
                "epochs": 40,
                "batch_size": 24,
                "learning_rate": 0.0002,
                "optimizer": "adam",
                "loss_function": "huber",
                "validation_split": 0.15,
                "early_stopping_patience": 6
            }
        }
    
    async def start_training(self, model_id: str, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """启动模型训练"""
        try:
            # 生成训练ID
            training_id = str(uuid.uuid4())
            
            # 验证模型是否存在
            if not self.model_registry.is_model_registered(model_id):
                raise ValueError(f"Model '{model_id}' not found")
            
            # 获取模型信息
            model_info = self.model_registry.get_model_info(model_id)
            model_type = model_info.get("model_type", "general")
            
            # 检查是否为增量训练
            training_type = training_config.get("type", "normal")
            
            if training_type == "incremental":
                # 增量训练配置
                merged_config = {
                    **training_config,
                    "model_type": model_type,
                    "epochs": training_config.get("epochs", 1),
                    "batch_size": training_config.get("batch_size", 1)
                }
            else:
                # 合并默认配置
                default_config = self.default_training_configs.get(model_type, {})
                merged_config = {**default_config, **training_config}
            
            # 记录训练开始
            self.active_trainings[training_id] = {
                "training_id": training_id,
                "model_id": model_id,
                "model_type": model_type,
                "training_type": training_type,
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "config": merged_config,
                "progress": 0.0,
                "current_epoch": 0,
                "loss": 0.0,
                "accuracy": 0.0
            }
            
            # 启动训练线程
            training_thread = threading.Thread(
                target=self._run_training,
                args=(training_id, model_id, merged_config)
            )
            training_thread.daemon = True
            training_thread.start()
            
            logger.info(f"Training started for model '{model_id}' (type: {training_type}) with ID: {training_id}")
            
            return {
                "success": True,
                "training_id": training_id,
                "message": f"Training started successfully for model '{model_id}'",
                "estimated_duration": self._estimate_training_duration(merged_config)
            }
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_training(self, training_id: str, model_id: str, config: Dict[str, Any]):
        """运行真实训练过程"""
        try:
            training_type = config.get("type", "normal")
            
            # 获取模型实例进行训练
            model_instance = self._get_model_instance(model_id)
            if model_instance is None:
                raise ValueError(f"Cannot get model instance for model_id: {model_id}")
            
            # 准备训练数据
            training_data = self._prepare_training_data(model_id, config)
            
            if training_type == "incremental":
                # 真实增量训练逻辑
                logger.info(f"Starting real incremental training for model '{model_id}'")
                
                # 获取增量训练样本
                sample = config.get("sample", {})
                if not sample:
                    raise ValueError("Incremental training requires a sample")
                
                # 保存增量训练样本到数据目录
                sample_file = self.training_data_dir / f"incremental_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(sample_file, "w") as f:
                    json.dump(sample, f, indent=2)
                
                # 配置增量训练参数
                incremental_config = config.copy()
                incremental_config["training_mode"] = "incremental"
                incremental_config["epochs"] = config.get("epochs", 1)
                
                # 调用模型的实际增量训练方法
                training_result = model_instance.train_model(
                    training_data=sample, 
                    training_config=incremental_config
                )
                
                # 更新训练状态
                self.active_trainings[training_id].update({
                    "current_epoch": incremental_config["epochs"],
                    "progress": 100.0,
                    "loss": training_result.get("final_loss", 0.0),
                    "accuracy": training_result.get("final_accuracy", 0.0),
                    "training_result": training_result
                })
                
            else:
                # 真实常规训练逻辑
                logger.info(f"Starting real training for model '{model_id}'")
                
                # 配置训练参数
                training_config = config.copy()
                training_config["training_mode"] = "normal"
                
                # 调用模型的实际训练方法
                training_result = model_instance.train_model(
                    training_data=training_data,
                    training_config=training_config
                )
                
                # 更新训练状态
                if training_result.get("success", False):
                    epochs = config.get("epochs", training_result.get("epochs_completed", 100))
                    self.active_trainings[training_id].update({
                        "current_epoch": epochs,
                        "progress": 100.0,
                        "loss": training_result.get("final_loss", 0.0),
                        "accuracy": training_result.get("final_accuracy", 0.0),
                        "training_result": training_result
                    })
                else:
                    raise RuntimeError(f"Training failed: {training_result.get('error', 'Unknown error')}")
            
            # 训练完成
            self.active_trainings[training_id]["status"] = "completed"
            self.active_trainings[training_id]["completed_at"] = datetime.now().isoformat()
            
            # 保存训练结果
            self._save_training_results(training_id, model_id, config)
            
            logger.info(f"Training {training_id} completed successfully using real neural network training")
            
        except Exception as e:
            logger.error(f"Training {training_id} failed: {e}")
            self.active_trainings[training_id]["status"] = "failed"
            self.active_trainings[training_id]["error"] = str(e)
            self.active_trainings[training_id]["failed_at"] = datetime.now().isoformat()
    
    def _get_model_instance(self, model_id: str):
        """获取模型实例进行训练"""
        try:
            # 尝试通过模型注册表获取模型实例
            model_info = self.model_registry.get_model_info(model_id)
            if model_info and "model_class" in model_info:
                model_class = model_info["model_class"]
                model_config = model_info.get("config", {})
                return model_class(model_config)
            else:
                # 尝试通过模型服务管理器获取
                from core.model_service_manager import model_service_manager
                return model_service_manager.get_model(model_id)
        except Exception as e:
            logger.error(f"Cannot get model instance for {model_id}: {e}")
            return None
    
    def _prepare_training_data(self, model_id: str, config: Dict[str, Any]):
        """准备训练数据"""
        try:
            # 检查配置中是否有训练数据
            if "training_data" in config:
                return config["training_data"]
            
            # 尝试从数据集中加载训练数据
            model_type = config.get("model_type", "general")
            dataset_name = config.get("dataset", f"{model_type}_dataset")
            
            from core.dataset_manager import DatasetManager
            dataset_manager = DatasetManager()
            dataset = dataset_manager.get_dataset(dataset_name)
            
            if dataset:
                return dataset.get_training_split()
            else:
                # 返回空数据集，模型应该处理这种情况
                return []
                
        except Exception as e:
            logger.error(f"Cannot prepare training data for {model_id}: {e}")
            return []
    
    def _calculate_loss(self, epoch: int, total_epochs: int) -> float:
        """计算损失值"""
        # 模拟损失下降曲线
        base_loss = 2.0
        decay_rate = 0.95
        return base_loss * (decay_rate ** epoch)
    
    def _calculate_accuracy(self, epoch: int, total_epochs: int) -> float:
        """计算准确率"""
        # 模拟准确率上升曲线
        base_accuracy = 0.1
        growth_rate = 0.98
        max_accuracy = 0.95
        
        accuracy = base_accuracy + (max_accuracy - base_accuracy) * (1 - growth_rate ** epoch)
        return min(accuracy, max_accuracy)
    
    def _should_stop_training(self, training_id: str) -> bool:
        """检查是否应该停止训练"""
        training_info = self.active_trainings.get(training_id, {})
        return training_info.get("status") == "stopping"
    
    def _save_training_results(self, training_id: str, model_id: str, config: Dict[str, Any]):
        """保存训练结果"""
        training_info = self.active_trainings[training_id]
        
        results = {
            "training_id": training_id,
            "model_id": model_id,
            "config": config,
            "final_loss": training_info["loss"],
            "final_accuracy": training_info["accuracy"],
            "total_epochs": training_info["current_epoch"],
            "start_time": training_info["start_time"],
            "end_time": training_info["completed_at"],
            "training_duration": self._calculate_training_duration(training_info)
        }
        
        # 保存到文件
        results_file = self.training_data_dir / f"{training_id}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # 保存到内存
        self.training_results[training_id] = results
    
    def _calculate_training_duration(self, training_info: Dict[str, Any]) -> float:
        """计算训练时长"""
        start_time = datetime.fromisoformat(training_info["start_time"])
        end_time = datetime.fromisoformat(training_info["completed_at"])
        return (end_time - start_time).total_seconds()
    
    async def start_joint_training(self, model_ids: List[str], joint_config: Dict[str, Any]) -> Dict[str, Any]:
        """启动联合训练"""
        try:
            # 生成联合训练ID
            joint_training_id = str(uuid.uuid4())
            
            # 验证所有模型是否存在
            for model_id in model_ids:
                if not self.model_registry.is_model_registered(model_id):
                    raise ValueError(f"Model '{model_id}' not found")
            
            # 初始化联合训练协调器（如果需要）
            if self.joint_training_coordinator is None:
                from core.joint_training_coordinator import JointTrainingCoordinator
                self.joint_training_coordinator = JointTrainingCoordinator(
                    model_ids=model_ids,
                    parameters=joint_config
                )
            
            # 启动联合训练
            result = await self.joint_training_coordinator.start_joint_training(
                model_ids=model_ids,
                joint_config=joint_config
            )
            
            # 记录联合训练
            self.active_trainings[joint_training_id] = {
                "training_id": joint_training_id,
                "model_ids": model_ids,
                "training_type": "joint",
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "config": joint_config
            }
            
            logger.info(f"Joint training started for models: {model_ids}")
            
            return {
                "success": True,
                "joint_training_id": joint_training_id,
                "message": f"Joint training started for {len(model_ids)} models",
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Failed to start joint training: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def start_autonomous_learning(self, model_id: str, autonomous_config: Dict[str, Any]) -> Dict[str, Any]:
        """启动自主学习"""
        try:
            # 验证模型是否存在
            if not self.model_registry.is_model_registered(model_id):
                raise ValueError(f"Model '{model_id}' not found")
            
            # 启动自主学习
            result = await self.autonomous_learning_manager.start_autonomous_learning(
                model_id=model_id,
                autonomous_config=autonomous_config
            )
            
            logger.info(f"Autonomous learning started for model '{model_id}'")
            
            return {
                "success": True,
                "message": f"Autonomous learning started for model '{model_id}'",
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Failed to start autonomous learning: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_training_status(self, training_id: str) -> Dict[str, Any]:
        """获取训练状态"""
        if training_id not in self.active_trainings:
            return {
                "success": False,
                "error": f"Training '{training_id}' not found"
            }
        
        training_info = self.active_trainings[training_id]
        
        # 构建状态响应
        status_response = {
            "training_id": training_id,
            "model_id": training_info.get("model_id"),
            "model_ids": training_info.get("model_ids", []),
            "status": training_info["status"],
            "progress": training_info.get("progress", 0.0),
            "current_epoch": training_info.get("current_epoch", 0),
            "loss": training_info.get("loss", 0.0),
            "accuracy": training_info.get("accuracy", 0.0),
            "start_time": training_info["start_time"]
        }
        
        # 添加完成时间信息
        if training_info["status"] == "completed":
            status_response["completed_at"] = training_info.get("completed_at")
        elif training_info["status"] == "failed":
            status_response["error"] = training_info.get("error")
        
        return {
            "success": True,
            "training_status": status_response
        }
    
    async def stop_training(self, training_id: str) -> Dict[str, Any]:
        """停止训练"""
        if training_id not in self.active_trainings:
            return {
                "success": False,
                "error": f"Training '{training_id}' not found"
            }
        
        training_info = self.active_trainings[training_id]
        
        if training_info["status"] != "running":
            return {
                "success": False,
                "error": f"Training '{training_id}' is not running"
            }
        
        # 标记为停止中
        training_info["status"] = "stopping"
        
        # 等待训练停止
        max_wait_time = 30  # 最大等待时间（秒）
        start_time = time.time()
        
        while training_info["status"] == "stopping" and time.time() - start_time < max_wait_time:
            await asyncio.sleep(1)
        
        if training_info["status"] == "stopping":
            # 强制停止
            training_info["status"] = "stopped"
            training_info["stopped_at"] = datetime.now().isoformat()
        
        logger.info(f"Training {training_id} stopped successfully")
        
        return {
            "success": True,
            "message": f"Training '{training_id}' stopped successfully"
        }
    
    async def list_trainings(self, status: Optional[str] = None) -> Dict[str, Any]:
        """列出所有训练任务"""
        trainings = []
        
        for training_id, training_info in self.active_trainings.items():
            # 过滤条件
            if status and training_info["status"] != status:
                continue
            
            training_data = {
                "training_id": training_id,
                "model_id": training_info.get("model_id"),
                "model_ids": training_info.get("model_ids", []),
                "training_type": training_info.get("training_type", "individual"),
                "status": training_info["status"],
                "progress": training_info.get("progress", 0.0),
                "start_time": training_info["start_time"],
                "completed_at": training_info.get("completed_at"),
                "failed_at": training_info.get("failed_at")
            }
            
            trainings.append(training_data)
        
        return {
            "success": True,
            "trainings": trainings,
            "total_count": len(trainings),
            "running_count": len([t for t in trainings if t["status"] == "running"]),
            "completed_count": len([t for t in trainings if t["status"] == "completed"]),
            "failed_count": len([t for t in trainings if t["status"] == "failed"])
        }
    
    async def get_training_results(self, training_id: str) -> Dict[str, Any]:
        """获取训练结果"""
        if training_id not in self.training_results:
            # 尝试从文件加载
            results_file = self.training_data_dir / f"{training_id}_results.json"
            
            if results_file.exists():
                with open(results_file, "r") as f:
                    self.training_results[training_id] = json.load(f)
            else:
                return {
                    "success": False,
                    "error": f"Training results for '{training_id}' not found"
                }
        
        return {
            "success": True,
            "training_results": self.training_results[training_id]
        }
    
    def _estimate_training_duration(self, config: Dict[str, Any]) -> int:
        """估计训练时长（秒）"""
        epochs = config.get("epochs", 100)
        batch_size = config.get("batch_size", 32)
        dataset_size = config.get("dataset_size", 10000)
        
        # 简单的估计公式
        batches_per_epoch = dataset_size / batch_size
        time_per_batch = 0.1  # 假设每批次0.1秒
        
        total_time = epochs * batches_per_epoch * time_per_batch
        return int(total_time)
    
    async def export_training_data(self, training_id: str, export_format: str = "json") -> Dict[str, Any]:
        """导出训练数据"""
        try:
            if training_id not in self.training_results:
                return {
                    "success": False,
                    "error": f"Training results for '{training_id}' not found"
                }
            
            results = self.training_results[training_id]
            
            if export_format == "json":
                export_data = json.dumps(results, indent=2)
                file_extension = "json"
            elif export_format == "csv":
                # 简化的CSV导出
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # 写入标题
                writer.writerow(["Metric", "Value"])
                
                # 写入数据
                for key, value in results.items():
                    if isinstance(value, (str, int, float)):
                        writer.writerow([key, str(value)])
                
                export_data = output.getvalue()
                file_extension = "csv"
            else:
                return {
                    "success": False,
                    "error": f"Unsupported export format: {export_format}"
                }
            
            # 保存导出文件
            export_file = self.training_data_dir / f"{training_id}_export.{file_extension}"
            with open(export_file, "w") as f:
                f.write(export_data)
            
            return {
                "success": True,
                "export_file": str(export_file),
                "export_format": export_format,
                "file_size": len(export_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to export training data: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# 创建全局实例
enhanced_training_system = EnhancedTrainingSystem()
