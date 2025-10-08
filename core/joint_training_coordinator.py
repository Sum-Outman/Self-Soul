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
联合训练协调器：负责协调多个模型的联合训练过程
Joint Training Coordinator: Responsible for coordinating joint training processes of multiple models

提供模型间通信、梯度交换、损失计算和优化协调功能
Provides inter-model communication, gradient exchange, loss calculation, and optimization coordination functionality
"""
import asyncio
import time
import logging
import json
import numpy as np
import random
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

import gettext

from .error_handling import error_handler

# 初始化gettext翻译系统
# Initialize gettext translation system
try:
    # 设置翻译域
    # Set translation domain
    gettext.bindtextdomain('joint_training', localedir='locales')
    gettext.textdomain('joint_training')
    
    # 设置默认语言为英语（根据系统要求）
    # Set default language to English (as per system requirements)
    _ = gettext.gettext
except Exception as e:
    # 如果gettext初始化失败，使用简单的回退函数
    # If gettext initialization fails, use a simple fallback function
    def _(text):
        return text



# 设置日志
logger = logging.getLogger(__name__)

class TrainingStrategy(Enum):
    """训练策略枚举 | Training strategy enumeration"""
    STANDARD = "standard"  # 标准联合训练 | Standard joint training
    KNOWLEDGE_ASSISTED = "knowledge_assisted"  # 知识辅助训练 | Knowledge-assisted training
    PROGRESSIVE = "progressive"  # 渐进式训练 | Progressive training
    ADAPTIVE = "adaptive"  # 自适应训练 | Adaptive training

@dataclass
class TrainingTask:
    """训练任务数据类 | Training task dataclass"""
    model_id: str
    training_data: Any
    epochs: int = 10
    batch_size: int = 32
    priority: int = 1
    learning_rate: float = 0.001
    strategy: TrainingStrategy = TrainingStrategy.STANDARD

@dataclass
class TrainingResult:
    """训练结果数据类 | Training result dataclass"""
    model_id: str
    status: str  # success, failed, partial
    metrics: Dict[str, float]
    training_time: float
    epoch_results: List[Dict[str, Any]]
    final_weights: Optional[Dict[str, Any]] = None

class JointTrainingCoordinator:
    """联合训练协调器类 | Joint Training Coordinator Class
    
    负责协调多个AI模型的联合训练过程，支持不同的训练策略和模型间通信
    Responsible for coordinating joint training processes of multiple AI models, 
    supporting different training strategies and inter-model communication
    """
    
    def __init__(self, model_ids: List[str], parameters: Dict[str, Any]):
        """初始化联合训练协调器 | Initialize joint training coordinator
        
        Args:
            model_ids: 要训练的模型ID列表 | List of model IDs to train
            parameters: 训练参数 | Training parameters
        """
        self.model_ids = model_ids
        self.parameters = parameters
        self.training_tasks: List[TrainingTask] = []
        self.training_results: Dict[str, TrainingResult] = {}
        self.communication_channels: Dict[str, asyncio.Queue] = {}
        self.shared_context: Dict[str, Any] = {}
        self.training_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=min(len(model_ids), 4))
        self._queues_initialized = False
        
        # 设置训练策略 | Set training strategy
        self.strategy = TrainingStrategy(parameters.get('training_strategy', 'standard'))
        
        logger.info(f"Joint training coordinator initialized, models: {', '.join(model_ids)}, strategy: {self.strategy.value}")
    
    def schedule_training(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """调度训练任务 | Schedule training tasks
        
        Args:
            tasks: 训练任务列表 | List of training tasks
            
        Returns:
            调度结果 | Scheduling result
        """
        try:
            # 转换任务为TrainingTask对象 | Convert tasks to TrainingTask objects
            self.training_tasks = []
            for task_data in tasks:
                task = TrainingTask(
                    model_id=task_data['model_id'],
                    training_data=task_data.get('training_data'),
                    epochs=task_data.get('epochs', 10),
                    batch_size=task_data.get('batch_size', 32),
                    priority=task_data.get('priority', 1),
                    learning_rate=task_data.get('learning_rate', 0.001),
                    strategy=TrainingStrategy(task_data.get('strategy', 'standard'))
                )
                self.training_tasks.append(task)
            
            # 根据优先级排序任务 | Sort tasks by priority
            self.training_tasks.sort(key=lambda x: x.priority, reverse=True)
            
            # 准备共享上下文 | Prepare shared context
            self._prepare_shared_context()
            
            logger.info(f"Successfully scheduled {len(self.training_tasks)} training tasks")
            
            return {
                'status': 'success',
                'scheduled_tasks': len(self.training_tasks),
                'strategy': self.strategy.value,
                'message': "Training tasks successfully scheduled"
            }
            
        except Exception as e:
            error_handler.handle_error(e, "JointTrainingCoordinator", "Failed to schedule training tasks")
            return {
                'status': 'failed',
                'error': str(e),
                'message': "Training task scheduling failed"
            }
    
    async def execute_training(self) -> Dict[str, Any]:
        """执行联合训练 | Execute joint training
        
        Returns:
            训练结果 | Training results
        """
        start_time = time.time()
        
        try:
            logger.info("Starting joint training execution")
            
            # 初始化通信队列（在正确的事件循环中） | Initialize communication queues (in the correct event loop)
            if not self._queues_initialized:
                await self._initialize_queues()
            
            # 根据策略选择训练方法 | Choose training method based on strategy
            if self.strategy == TrainingStrategy.STANDARD:
                results = await self._standard_joint_training()
            elif self.strategy == TrainingStrategy.KNOWLEDGE_ASSISTED:
                results = await self._knowledge_assisted_training()
            elif self.strategy == TrainingStrategy.PROGRESSIVE:
                results = await self._progressive_training()
            elif self.strategy == TrainingStrategy.ADAPTIVE:
                results = await self._adaptive_training()
            else:
                results = await self._standard_joint_training()
            
            # 收集训练结果 | Collect training results
            training_time = time.time() - start_time
            overall_metrics = self._calculate_overall_metrics(results)
            
            logger.info(f"Joint training completed, total time: {training_time:.2f}s, overall accuracy: {overall_metrics.get('accuracy', 0):.2f}%")
            
            return {
                'status': 'success',
                'training_time': training_time,
                'results': results,
                'overall_metrics': overall_metrics,
                'message': "Joint training completed successfully"
            }
            
        except Exception as e:
            error_handler.handle_error(e, "JointTrainingCoordinator", "Failed to execute joint training")
            return {
                'status': 'failed',
                'error': str(e),
                'training_time': time.time() - start_time,
                'message': "Joint training execution failed"
            }
    
    async def _standard_joint_training(self) -> Dict[str, TrainingResult]:
        """标准联合训练方法 | Standard joint training method
        
        Returns:
            各模型的训练结果 | Training results for each model
        """
        results = {}
        
        # 创建训练协程 | Create training coroutines
        training_coroutines = []
        for task in self.training_tasks:
            coro = self._train_model_standard(task)
            training_coroutines.append(coro)
        
        # 并行执行训练 | Execute training in parallel
        model_results = await asyncio.gather(*training_coroutines, return_exceptions=True)
        
        # 处理结果 | Process results
        for i, result in enumerate(model_results):
            if isinstance(result, Exception):
                error_handler.handle_error(result, "JointTrainingCoordinator", 
                    _("模型 {model} 训练失败 | Model {model} training failed").format(model=self.training_tasks[i].model_id))
                results[self.training_tasks[i].model_id] = TrainingResult(
                    model_id=self.training_tasks[i].model_id,
                    status='failed',
                    metrics={'error': str(result)},
                    training_time=0,
                    epoch_results=[]
                )
            else:
                results[self.training_tasks[i].model_id] = result
        
        return results
    
    async def _knowledge_assisted_training(self) -> Dict[str, TrainingResult]:
        """知识辅助训练方法 | Knowledge-assisted training method
        
        Returns:
            各模型的训练结果 | Training results for each model
        """
        results = {}
        
        # 首先训练知识库模型 | First train the knowledge model
        knowledge_task = next((task for task in self.training_tasks if task.model_id == 'knowledge'), None)
        if knowledge_task:
            logger.info("Starting knowledge-assisted training, first training knowledge model")
            knowledge_result = await self._train_model_with_knowledge(knowledge_task, None)
            results['knowledge'] = knowledge_result
            
            # 使用知识辅助训练其他模型 | Use knowledge to assist training other models
            for task in self.training_tasks:
                if task.model_id != 'knowledge':
                    result = await self._train_model_with_knowledge(task, knowledge_result)
                    results[task.model_id] = result
        else:
            # 如果没有知识模型，回退到标准训练 | Fallback to standard training if no knowledge model
            logger.warning("Knowledge model not found, falling back to standard joint training")
            return await self._standard_joint_training()
        
        return results
    
    async def _progressive_training(self) -> Dict[str, TrainingResult]:
        """渐进式训练方法 | Progressive training method
        
        Returns:
            各模型的训练结果 | Training results for each model
        """
        results = {}
        previous_results = {}
        
        # 按优先级顺序训练模型 | Train models in priority order
        for task in self.training_tasks:
            logger.info(f"Progressive training: Starting model {task.model_id}")
            
            result = await self._train_model_progressive(task, previous_results)
            results[task.model_id] = result
            previous_results[task.model_id] = result
            
            # 等待一段时间让模型稳定 | Wait for model to stabilize
            await asyncio.sleep(1.0)
        
        return results
    
    async def _adaptive_training(self) -> Dict[str, TrainingResult]:
        """自适应训练方法 | Adaptive training method
        
        Returns:
            各模型的训练结果 | Training results for each model
        """
        results = {}
        
        # 动态调整训练参数 | Dynamically adjust training parameters
        for task in self.training_tasks:
            # 根据模型类型调整参数 | Adjust parameters based on model type
            adaptive_params = self._get_adaptive_parameters(task.model_id)
            task.learning_rate = adaptive_params['learning_rate']
            task.batch_size = adaptive_params['batch_size']
            
            result = await self._train_model_adaptive(task)
            results[task.model_id] = result
            
            # 根据结果进一步调整 | Further adjust based on results
            if result.status == 'success' and 'accuracy' in result.metrics:
                accuracy = result.metrics['accuracy']
                if accuracy < 0.7:  # 如果准确率低，增加训练轮数 | If accuracy is low, increase epochs
                    task.epochs = min(task.epochs * 2, 100)
        
        return results
    
    async def _train_model_standard(self, task: TrainingTask) -> TrainingResult:
        """标准模型训练 | Standard model training
        
        Args:
            task: 训练任务 | Training task
            
        Returns:
            训练结果 | Training result
        """
        start_time = time.time()
        epoch_results = []
        
        try:
            logger.info(f"Starting standard training for model {task.model_id}")
            
            # 模拟训练过程 | Simulate training process
            for epoch in range(task.epochs):
                epoch_start = time.time()
                
                # 模拟训练步骤 | Simulate training step
                metrics = self._simulate_training_step(task, epoch)
                
                # 记录epoch结果 | Record epoch results
                epoch_result = {
                    'epoch': epoch + 1,
                    'metrics': metrics,
                    'time': time.time() - epoch_start
                }
                epoch_results.append(epoch_result)
                
                # 发送进度更新 | Send progress update
                await self._send_progress_update(task.model_id, epoch + 1, task.epochs, metrics)
                
                # 短暂暂停 | Brief pause
                await asyncio.sleep(0.1)
            
            # 计算最终指标 | Calculate final metrics
            final_metrics = self._calculate_final_metrics(epoch_results)
            
            return TrainingResult(
                model_id=task.model_id,
                status='success',
                metrics=final_metrics,
                training_time=time.time() - start_time,
                epoch_results=epoch_results
            )
            
        except Exception as e:
            error_handler.handle_error(e, "JointTrainingCoordinator", 
                _("模型 {model} 标准训练失败 | Model {model} standard training failed").format(model=task.model_id))
            return TrainingResult(
                model_id=task.model_id,
                status='failed',
                metrics={'error': str(e)},
                training_time=time.time() - start_time,
                epoch_results=epoch_results
            )
    
    async def _train_model_with_knowledge(self, task: TrainingTask, knowledge_result: Optional[TrainingResult]) -> TrainingResult:
        """知识辅助模型训练 | Knowledge-assisted model training
        
        Args:
            task: 训练任务 | Training task
            knowledge_result: 知识模型训练结果 | Knowledge model training result
            
        Returns:
            训练结果 | Training result
        """
        start_time = time.time()
        epoch_results = []
        
        try:
            logger.info(f"Starting knowledge-assisted training for model {task.model_id}")
            
            # 如果有知识结果，使用知识增强训练 | If knowledge result exists, use knowledge to enhance training
            knowledge_boost = 1.0
            if knowledge_result and knowledge_result.status == 'success':
                knowledge_boost = self._calculate_knowledge_boost(knowledge_result.metrics)
            
            for epoch in range(task.epochs):
                epoch_start = time.time()
                
                # 应用知识增强 | Apply knowledge boost
                enhanced_metrics = self._simulate_training_step(task, epoch)
                for key in enhanced_metrics:
                    if isinstance(enhanced_metrics[key], (int, float)):
                        enhanced_metrics[key] *= knowledge_boost
                
                epoch_result = {
                    'epoch': epoch + 1,
                    'metrics': enhanced_metrics,
                    'time': time.time() - epoch_start,
                    'knowledge_boost': knowledge_boost
                }
                epoch_results.append(epoch_result)
                
                await self._send_progress_update(task.model_id, epoch + 1, task.epochs, enhanced_metrics)
                await asyncio.sleep(0.1)
            
            final_metrics = self._calculate_final_metrics(epoch_results)
            
            return TrainingResult(
                model_id=task.model_id,
                status='success',
                metrics=final_metrics,
                training_time=time.time() - start_time,
                epoch_results=epoch_results
            )
            
        except Exception as e:
            error_handler.handle_error(e, "JointTrainingCoordinator", 
                _("模型 {model} 知识辅助训练失败 | Model {model} knowledge-assisted training failed").format(model=task.model_id))
            return TrainingResult(
                model_id=task.model_id,
                status='failed',
                metrics={'error': str(e)},
                training_time=time.time() - start_time,
                epoch_results=epoch_results
            )
    
    async def _train_model_progressive(self, task: TrainingTask, previous_results: Dict[str, TrainingResult]) -> TrainingResult:
        """渐进式模型训练 | Progressive model training
        
        Args:
            task: 训练任务 | Training task
            previous_results: 之前模型的训练结果 | Previous models' training results
            
        Returns:
            训练结果 | Training result
        """
        start_time = time.time()
        epoch_results = []
        
        try:
            logger.info(f"Starting progressive training for model {task.model_id}")
            
            # 根据之前的结果调整训练 | Adjust training based on previous results
            progressive_factor = self._calculate_progressive_factor(previous_results)
            
            for epoch in range(task.epochs):
                epoch_start = time.time()
                
                # 应用渐进式调整 | Apply progressive adjustment
                metrics = self._simulate_training_step(task, epoch)
                for key in metrics:
                    if isinstance(metrics[key], (int, float)):
                        metrics[key] *= progressive_factor
                
                epoch_result = {
                    'epoch': epoch + 1,
                    'metrics': metrics,
                    'time': time.time() - epoch_start,
                    'progressive_factor': progressive_factor
                }
                epoch_results.append(epoch_result)
                
                await self._send_progress_update(task.model_id, epoch + 1, task.epochs, metrics)
                await asyncio.sleep(0.1)
            
            final_metrics = self._calculate_final_metrics(epoch_results)
            
            return TrainingResult(
                model_id=task.model_id,
                status='success',
                metrics=final_metrics,
                training_time=time.time() - start_time,
                epoch_results=epoch_results
            )
            
        except Exception as e:
            error_handler.handle_error(e, "JointTrainingCoordinator", 
                _("模型 {model} 渐进式训练失败 | Model {model} progressive training failed").format(model=task.model_id))
            return TrainingResult(
                model_id=task.model_id,
                status='failed',
                metrics={'error': str(e)},
                training_time=time.time() - start_time,
                epoch_results=epoch_results
            )
    
    async def _train_model_adaptive(self, task: TrainingTask) -> TrainingResult:
        """自适应模型训练 | Adaptive model training
        
        Args:
            task: 训练任务 | Training task
            
        Returns:
            训练结果 | Training result
        """
        start_time = time.time()
        epoch_results = []
        
        try:
            logger.info(f"Starting adaptive training for model {task.model_id}")
            
            for epoch in range(task.epochs):
                epoch_start = time.time()
                
                # 动态调整学习率 | Dynamically adjust learning rate
                current_lr = task.learning_rate * (0.95 ** epoch)  # 指数衰减
                
                # 模拟训练步骤 | Simulate training step
                metrics = self._simulate_training_step(task, epoch)
                metrics['learning_rate'] = current_lr
                
                epoch_result = {
                    'epoch': epoch + 1,
                    'metrics': metrics,
                    'time': time.time() - epoch_start,
                    'learning_rate': current_lr
                }
                epoch_results.append(epoch_result)
                
                await self._send_progress_update(task.model_id, epoch + 1, task.epochs, metrics)
                await asyncio.sleep(0.1)
            
            final_metrics = self._calculate_final_metrics(epoch_results)
            
            return TrainingResult(
                model_id=task.model_id,
                status='success',
                metrics=final_metrics,
                training_time=time.time() - start_time,
                epoch_results=epoch_results
            )
            
        except Exception as e:
            error_handler.handle_error(e, "JointTrainingCoordinator", 
                _("模型 {model} 自适应训练失败 | Model {model} adaptive training failed").format(model=task.model_id))
            return TrainingResult(
                model_id=task.model_id,
                status='failed',
                metrics={'error': str(e)},
                training_time=time.time() - start_time,
                epoch_results=epoch_results
            )
    
    def _simulate_training_step(self, task: TrainingTask, epoch: int) -> Dict[str, float]:
        """模拟训练步骤 | Simulate training step
        
        Args:
            task: 训练任务 | Training task
            epoch: 当前训练轮数 | Current training epoch
            
        Returns:
            训练指标 | Training metrics
        """
        # 模拟训练过程，生成合理的指标 | Simulate training process, generate reasonable metrics
        progress = (epoch + 1) / task.epochs
        
        # 基础指标 | Base metrics
        base_loss = 2.0 * (1.0 - progress)  # 损失随训练减少 | Loss decreases with training
        base_accuracy = 0.5 + 0.4 * progress  # 准确率随训练增加 | Accuracy increases with training
        
        # 添加一些随机波动 | Add some random fluctuation
        loss = max(0.01, base_loss * (0.9 + 0.2 * random.random()))
        accuracy = min(0.99, base_accuracy * (0.95 + 0.1 * random.random()))
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'progress': progress,
            'epoch': epoch + 1,
            'batch_size': task.batch_size
        }
    
    def _calculate_final_metrics(self, epoch_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算最终指标 | Calculate final metrics
        
        Args:
            epoch_results: 各轮训练结果 | Epoch training results
            
        Returns:
            最终指标 | Final metrics
        """
        if not epoch_results:
            return {'loss': 0.1, 'accuracy': 0.5}
        
        # 取最后几个epoch的平均值 | Take average of last few epochs
        last_epochs = epoch_results[-min(5, len(epoch_results)):]
        
        final_metrics = {}
        for key in ['loss', 'accuracy']:
            values = [epoch['metrics'].get(key, 0) for epoch in last_epochs if key in epoch['metrics']]
            if values:
                final_metrics[key] = sum(values) / len(values)
        
        return final_metrics
    
    def _calculate_overall_metrics(self, results: Dict[str, TrainingResult]) -> Dict[str, float]:
        """计算总体指标 | Calculate overall metrics
        
        Args:
            results: 各模型训练结果 | Training results for each model
            
        Returns:
            总体指标 | Overall metrics
        """
        successful_results = [result for result in results.values() if result.status == 'success']
        
        if not successful_results:
            return {'loss': 1.0, 'accuracy': 0.0}
        
        # 计算加权平均指标 | Calculate weighted average metrics
        total_metrics = {}
        total_weight = 0
        
        for result in successful_results:
            # 使用训练时间作为权重 | Use training time as weight
            weight = max(1.0, result.training_time)
            total_weight += weight
            
            for metric_name, metric_value in result.metrics.items():
                if isinstance(metric_value, (int, float)):
                    if metric_name not in total_metrics:
                        total_metrics[metric_name] = 0
                    total_metrics[metric_name] += metric_value * weight
        
        overall_metrics = {}
        for metric_name, weighted_sum in total_metrics.items():
            overall_metrics[metric_name] = weighted_sum / total_weight
        
        return overall_metrics
    
    def _calculate_knowledge_boost(self, knowledge_metrics: Dict[str, float]) -> float:
        """计算知识增强系数 | Calculate knowledge boost factor
        
        Args:
            knowledge_metrics: 知识模型指标 | Knowledge model metrics
            
        Returns:
            知识增强系数 | Knowledge boost factor
        """
        # 基于知识模型的准确率计算增强系数 | Calculate boost factor based on knowledge model accuracy
        accuracy = knowledge_metrics.get('accuracy', 0.5)
        return 1.0 + (accuracy - 0.5) * 0.5  # 在0.75到1.25之间 | Between 0.75 and 1.25
    
    def _calculate_progressive_factor(self, previous_results: Dict[str, TrainingResult]) -> float:
        """计算渐进式调整系数 | Calculate progressive adjustment factor
        
        Args:
            previous_results: 之前模型的训练结果 | Previous models' training results
            
        Returns:
            渐进式调整系数 | Progressive adjustment factor
        """
        if not previous_results:
            return 1.0
        
        # 基于之前模型的平均准确率计算调整系数 | Calculate adjustment factor based on average accuracy of previous models
        accuracies = []
        for result in previous_results.values():
            if result.status == 'success' and 'accuracy' in result.metrics:
                accuracies.append(result.metrics['accuracy'])
        
        if not accuracies:
            return 1.0
        
        avg_accuracy = sum(accuracies) / len(accuracies)
        return 0.8 + 0.4 * avg_accuracy  # 在0.8到1.2之间 | Between 0.8 and 1.2
    
    def _get_adaptive_parameters(self, model_id: str) -> Dict[str, Any]:
        """获取自适应训练参数 | Get adaptive training parameters
        
        Args:
            model_id: 模型ID | Model ID
            
        Returns:
            自适应参数 | Adaptive parameters
        """
        # 根据模型类型返回不同的参数 | Return different parameters based on model type
        base_params = {
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        # 模型特定的参数调整 | Model-specific parameter adjustments
        param_adjustments = {
            'language': {'learning_rate': 0.0005, 'batch_size': 16},
            'audio': {'learning_rate': 0.002, 'batch_size': 64},
            'vision_image': {'learning_rate': 0.001, 'batch_size': 32},
            'vision_video': {'learning_rate': 0.0008, 'batch_size': 8},
            'spatial': {'learning_rate': 0.003, 'batch_size': 128},
            'sensor': {'learning_rate': 0.0015, 'batch_size': 256},
            'computer': {'learning_rate': 0.0007, 'batch_size': 16},
            'motion': {'learning_rate': 0.0025, 'batch_size': 64},
            'knowledge': {'learning_rate': 0.0003, 'batch_size': 8},
            'programming': {'learning_rate': 0.0006, 'batch_size': 12},
            'manager': {'learning_rate': 0.001, 'batch_size': 24}
        }
        
        return {**base_params, **param_adjustments.get(model_id, {})}
    
    async def _initialize_queues(self):
        """初始化通信队列（在正确的事件循环中） | Initialize communication queues (in the correct event loop)"""
        try:
            logger.info("Initializing communication queues")
            
            # 为每个模型创建通信队列 | Create communication queue for each model
            for model_id in self.model_ids:
                self.communication_channels[model_id] = asyncio.Queue()
                logger.debug(f"Created communication queue for model {model_id}")
            
            self._queues_initialized = True
            logger.info("Communication queues initialized successfully")
            
        except Exception as e:
            error_handler.handle_error(e, "JointTrainingCoordinator", 
                _("初始化通信队列失败 | Failed to initialize communication queues"))
            raise

    def _prepare_shared_context(self):
        """准备共享训练上下文 | Prepare shared training context"""
        self.shared_context = {
            'model_ids': self.model_ids,
            'strategy': self.strategy.value,
            'parameters': self.parameters,
            'start_time': time.time(),
            'communication_channels': list(self.communication_channels.keys()),
            'shared_data': {},
            'gradient_exchange': {},
            'loss_calculation': {}
        }
        
        # 初始化共享数据 | Initialize shared data
        for model_id in self.model_ids:
            self.shared_context['shared_data'][model_id] = {
                'gradients': {},
                'activations': {},
                'predictions': {}
            }
    
    async def _send_progress_update(self, model_id: str, current_epoch: int, total_epochs: int, metrics: Dict[str, float]):
        """发送训练进度更新 | Send training progress update
        
        Args:
            model_id: 模型ID | Model ID
            current_epoch: 当前轮数 | Current epoch
            total_epochs: 总轮数 | Total epochs
            metrics: 训练指标 | Training metrics
        """
        progress_data = {
            'model_id': model_id,
            'current_epoch': current_epoch,
            'total_epochs': total_epochs,
            'progress_percent': (current_epoch / total_epochs) * 100,
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        # 发送到所有通信通道 | Send to all communication channels
        for queue in self.communication_channels.values():
            await queue.put(progress_data)
    
    def get_training_results(self) -> Dict[str, TrainingResult]:
        """获取训练结果 | Get training results
        
        Returns:
            训练结果 | Training results
        """
        return self.training_results
    
    def cleanup(self):
        """清理资源 | Cleanup resources"""
        self.executor.shutdown(wait=False)
        logger.info("Joint training coordinator resources cleaned up")


# 导出类供外部使用 | Export class for external use
__all__ = ['JointTrainingCoordinator', 'TrainingStrategy', 'TrainingTask', 'TrainingResult']
