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
联合训练协调器：管理多模型协同训练
Joint Training Coordinator: Manages multi-model collaborative training

提供多种训练策略和模型协同训练管理功能
Provides multiple training strategies and model collaborative training management functionality
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Set
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from ..error_handling import error_handler
from ..model_registry import model_registry

# 配置日志记录器
logger = logging.getLogger(__name__)


"""
TrainingStrategy类 - 中文类描述
TrainingStrategy Class - English class description
"""
class TrainingStrategy(Enum):
    """训练策略枚举
    Training Strategy Enumeration
    """
    SEQUENTIAL = "sequential"      # 顺序训练
    PARALLEL = "parallel"          # 并行训练  
    ADAPTIVE = "adaptive"          # 自适应训练
    PIPELINE = "pipeline"          # 流水线训练

"""
TrainingTask类 - 中文类描述
TrainingTask Class - English class description
"""
@dataclass
class TrainingTask:
    """训练任务数据类
    Training Task Data Class
    """
    model_id: str
    training_data: Any
    epochs: int = 10
    batch_size: int = 32
    priority: int = 1
    dependencies: List[str] = None


"""
JointTrainingCoordinator类 - 中文类描述
JointTrainingCoordinator Class - English class description
"""
class JointTrainingCoordinator:
    """联合训练协调器类
    Joint Training Coordinator Class
    """

    def __init__(self, model_ids: List[str] = None, config: Dict = None):
        self.training_queue: List[TrainingTask] = []
        self.active_tasks: Set[str] = set()
        self.completed_tasks: Dict[str, Any] = {}
        self.training_strategy = TrainingStrategy.ADAPTIVE
        self.performance_metrics = {}
        self.resource_allocations = {}
        self.knowledge_assistant_enabled = True  # 是否启用知识库辅助训练
        # Whether to enable knowledge base assisted training
        
        # 存储传入的模型ID和配置 | Store passed model IDs and config
        self.model_ids = model_ids or []
        self.config = config or {}
        
        # 根据配置初始化训练策略 | Initialize training strategy based on config
        if config and "training_strategy" in config:
            try:
                self.training_strategy = TrainingStrategy(config["training_strategy"])
            except ValueError:
                logger.warning(f"无效的训练策略: {config['training_strategy']}, 使用默认策略")
                self.training_strategy = TrainingStrategy.ADAPTIVE
        
    def schedule_training(self, tasks: List[TrainingTask], 
                         strategy: TrainingStrategy = TrainingStrategy.ADAPTIVE) -> Dict[str, Any]:
        """调度训练任务
        Schedule training tasks
        
        Args:
            tasks: 训练任务列表
            strategy: 训练策略
            
        Returns:
            dict: 调度结果
        """
        self.training_strategy = strategy
        self.training_queue.extend(tasks)
        
        # 根据策略优化任务顺序
        if strategy == TrainingStrategy.ADAPTIVE:
            optimized_tasks = self._optimize_task_order(tasks)
            self.training_queue = optimized_tasks
        elif strategy == TrainingStrategy.PIPELINE:
            pipelined_tasks = self._create_pipeline(tasks)
            self.training_queue = pipelined_tasks
            
        return {
            "scheduled_tasks": len(self.training_queue),
            "strategy": strategy.value,
            "estimated_time": self._estimate_training_time()
        }
    
    async def execute_training(self) -> Dict[str, Any]:
        """执行训练任务
        Execute training tasks
        
        Returns:
            dict: 训练执行结果
        """
        results = {}
        
        try:
            if self.training_strategy == TrainingStrategy.SEQUENTIAL:
                results = await self._execute_sequential()
            elif self.training_strategy == TrainingStrategy.PARALLEL:
                results = await self._execute_parallel()
            elif self.training_strategy == TrainingStrategy.ADAPTIVE:
                results = await self._execute_adaptive()
            elif self.training_strategy == TrainingStrategy.PIPELINE:
                results = await self._execute_pipeline()
                
            # 训练后优化和知识整合
            await self._post_training_optimization(results)
            
        except Exception as e:
            error_handler.handle_error(e, "JointTrainingCoordinator", "联合训练执行失败")
            return {"status": "error", "message": str(e)}
        
        return {"status": "success", "results": results}
    
    async def _execute_sequential(self) -> Dict[str, Any]:
        """顺序执行训练任务
        Execute training tasks sequentially
        """
        results = {}
        for task in self.training_queue:
            if task.model_id not in self.active_tasks:
                self.active_tasks.add(task.model_id)
                result = await self._train_single_model(task)
                results[task.model_id] = result
                self.completed_tasks[task.model_id] = result
                self.active_tasks.remove(task.model_id)
        return results
    
    async def _execute_parallel(self) -> Dict[str, Any]:
        """并行执行训练任务  
        Execute training tasks in parallel
        """
        tasks = []
        for task in self.training_queue:
            if task.model_id not in self.active_tasks:
                self.active_tasks.add(task.model_id)
                tasks.append(self._train_single_model(task))
        
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        results = {}
        
        for i, task in enumerate(self.training_queue):
            if i < len(results_list) and not isinstance(results_list[i], Exception):
                results[task.model_id] = results_list[i]
                self.completed_tasks[task.model_id] = results_list[i]
            self.active_tasks.discard(task.model_id)
            
        return results
    
    async def _execute_adaptive(self) -> Dict[str, Any]:
        """自适应执行训练任务
        Execute training tasks adaptively
        """
        results = {}
        manager_model = model_registry.get_model("manager")
        
        if manager_model and hasattr(manager_model, 'optimize_training_plan'):
            # 使用管理模型优化训练计划
            training_plan = manager_model.optimize_training_plan(
                [task.model_id for task in self.training_queue],
                self.performance_metrics
            )
            
            for task in self.training_queue:
                if task.model_id in training_plan:
                    optimized_task = TrainingTask(
                        model_id=task.model_id,
                        training_data=task.training_data,
                        epochs=training_plan[task.model_id].get('epochs', task.epochs),
                        batch_size=training_plan[task.model_id].get('batch_size', task.batch_size),
                        priority=training_plan[task.model_id].get('priority', task.priority)
                    )
                    
                    self.active_tasks.add(optimized_task.model_id)
                    result = await self._train_single_model(optimized_task)
                    results[optimized_task.model_id] = result
                    self.completed_tasks[optimized_task.model_id] = result
                    self.active_tasks.remove(optimized_task.model_id)
        else:
            # 回退到并行训练
            results = await self._execute_parallel()
            
        return results
    
    async def _execute_pipeline(self) -> Dict[str, Any]:
        """流水线执行训练任务
        Execute training tasks in pipeline
        """
        results = {}
        # 实现基于数据流的流水线训练
        # 这里需要根据模型依赖关系创建训练流水线
        for task in self.training_queue:
            # 检查依赖是否完成
            if task.dependencies:
                deps_ready = all(dep in self.completed_tasks for dep in task.dependencies)
                if not deps_ready:
                    continue
            
            self.active_tasks.add(task.model_id)
            result = await self._train_single_model(task)
            results[task.model_id] = result
            self.completed_tasks[task.model_id] = result
            self.active_tasks.remove(task.model_id)
            
        return results
    
    async def _train_single_model(self, task: TrainingTask) -> Dict[str, Any]:
        """训练单个模型
        Train single model
        
        Args:
            task: 训练任务
            
        Returns:
            dict: 训练结果
        """
        try:
            model = model_registry.get_model(task.model_id)
            if not model:
                return {"status": "error", "message": f"Model {task.model_id} not found"}
            
            # 检查模型是否支持训练
            if not hasattr(model, 'train') or not callable(getattr(model, 'train')):
                return {"status": "error", "message": f"Model {task.model_id} does not support training"}
            
            start_time = time.time()
            
            # 知识库辅助训练（如果启用）
            # Knowledge base assisted training (if enabled)
            if self.knowledge_assistant_enabled:
                knowledge_model = model_registry.get_model("knowledge")
                if knowledge_model and hasattr(knowledge_model, 'assist_training'):
                    # 获取相关知识辅助训练
                    # Get relevant knowledge to assist training
                    # 获取训练数据元数据 | Get training data metadata
                    metadata = task.training_data.metadata if hasattr(task.training_data, 'metadata') else {}
                    
                    knowledge_context = await knowledge_model.assist_training(
                        model_id=task.model_id,
                        training_data_metadata=metadata
                    )
                    # 将知识上下文注入训练数据
                    # Inject knowledge context into training data
                    if hasattr(task.training_data, 'enhance_with_knowledge'):
                        task.training_data.enhance_with_knowledge(knowledge_context)
            
            # 调用模型的训练方法
            training_parameters = {
                "epochs": task.epochs,
                "batch_size": task.batch_size,
                "priority": task.priority
            }
            training_result = model.train(
                task.training_data, 
                training_parameters
            )
            
            training_time = time.time() - start_time
            
            # 记录性能指标
            self._update_performance_metrics(task.model_id, training_result, training_time)
            
            return {
                "status": "success",
                "model_id": task.model_id,
                "training_time": training_time,
                "result": training_result
            }
            
        except Exception as e:
            error_handler.handle_error(e, "JointTrainingCoordinator", f"训练模型 {task.model_id} 失败")
            return {"status": "error", "message": str(e)}
    
    
    def _optimize_task_order(self, tasks: List[TrainingTask]) -> List[TrainingTask]:
        """优化任务顺序
        Optimize task order
        """
        # 基于模型依赖关系和优先级排序
        sorted_tasks = sorted(tasks, key=lambda x: (
            -x.priority,  # 优先级降序
            len(x.dependencies) if x.dependencies else 0  # 依赖少的优先
        ))
        return sorted_tasks
    
    def _create_pipeline(self, tasks: List[TrainingTask]) -> List[TrainingTask]:
        """创建训练流水线
        Create training pipeline
        """
        # 基于模型依赖关系创建流水线
        pipeline_tasks = []
        processed_models = set()
        
        # 首先处理无依赖的任务
        for task in tasks:
            if not task.dependencies or all(dep in processed_models for dep in task.dependencies):
                pipeline_tasks.append(task)
                processed_models.add(task.model_id)
        
        # 然后处理有依赖的任务
        remaining_tasks = [t for t in tasks if t not in pipeline_tasks]
        while remaining_tasks:
            for task in remaining_tasks[:]:
                if all(dep in processed_models for dep in task.dependencies):
                    pipeline_tasks.append(task)
                    processed_models.add(task.model_id)
                    remaining_tasks.remove(task)
        
        return pipeline_tasks
    
    def _estimate_training_time(self) -> float:
        """估计训练时间
        Estimate training time
        """
        total_time = 0
        for task in self.training_queue:
            # 基于历史性能数据估计
            avg_time = self.performance_metrics.get(task.model_id, {}).get('avg_training_time', 60)
            total_time += avg_time * task.epochs
        
        return total_time
    
    
    def _update_performance_metrics(self, model_id: str, result: Dict[str, Any], training_time: float):
        """更新性能指标
        Update performance metrics
        """
        if model_id not in self.performance_metrics:
            self.performance_metrics[model_id] = {}
        
        metrics = self.performance_metrics[model_id]
        metrics['last_training_time'] = training_time
        metrics['avg_training_time'] = (
            metrics.get('avg_training_time', 0) * 0.8 + training_time * 0.2
        )
        metrics['last_accuracy'] = result.get('final_accuracy', 0)
        metrics['best_accuracy'] = max(metrics.get('best_accuracy', 0), result.get('final_accuracy', 0))
        metrics['training_count'] = metrics.get('training_count', 0) + 1
    

    async def _post_training_optimization(self, results: Dict[str, Any]):
        """训练后优化
        Post-training optimization
        """
        # 更新知识库
        knowledge_model = model_registry.get_model("knowledge")
        if knowledge_model and hasattr(knowledge_model, 'integrate_training_results'):
            await knowledge_model.integrate_training_results(results)
        
        # 优化模型协同
        manager_model = model_registry.get_model("manager")
        if manager_model and hasattr(manager_model, 'optimize_model_collaboration'):
            await manager_model.optimize_model_collaboration(results, self.performance_metrics)
        
        # 增强知识库学习能力
        if knowledge_model and hasattr(knowledge_model, 'enhance_knowledge_learning'):
            # 从训练结果中提取学习数据
            learning_data = {
                "text_sources": [],
                "structured_sources": [],
                "external_sources": []
            }
            
            for model_id, result in results.items():
                if "training_data" in result and result["training_data"]:
                    learning_data["text_sources"].append({
                        "type": "training_result",
                        "content": f"训练模型 {model_id} 的结果: {result}",
                        "domain": model_id
                    })
            
            # 应用增强学习
            enhancement_result = knowledge_model.enhance_knowledge_learning(learning_data)
            logger.info(f"知识库增强学习完成: {enhancement_result}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """获取训练状态
        Get training status
        """
        return {
            "queue_size": len(self.training_queue),
            "active_tasks": list(self.active_tasks),
            "completed_tasks": list(self.completed_tasks.keys()),
            "strategy": self.training_strategy.value,
            "performance_metrics": self.performance_metrics
        }
    
    
    def clear_queue(self):
        """清空训练队列
        Clear training queue
        """
        self.training_queue.clear()
        self.active_tasks.clear()
        self.completed_tasks.clear()

    def get_available_models(self) -> List[str]:
        """获取可用模型列表
        Get available models list
        
        Returns:
            List[str]: 可用模型ID列表
        """
        return [
            "manager",           # A - 管理模型
            "language",          # B - 大语言模型
            "audio",             # C - 音频处理模型
            "vision_image",      # D - 图片视觉处理模型
            "vision_video",      # E - 视频流视觉处理模型
            "spatial",           # F - 双目空间定位感知模型
            "sensor",            # G - 传感器感知模型
            "computer",          # H - 计算机控制模型
            "motion",            # I - 运动和执行器控制模型
            "knowledge",         # J - 知识库专家模型
            "programming"        # K - 编程模型
        ]

    def get_model_groups(self) -> Dict[str, List[str]]:
        """获取模型分组配置
        Get model groups configuration
        
        Returns:
            Dict[str, List[str]]: 模型分组字典
        """
        return {
            "basic_models": ["language", "audio", "vision_image", "vision_video"],
            "perception_models": ["sensor", "spatial", "audio", "vision_image", "vision_video"],
            "action_models": ["computer", "motion", "programming"],
            "cognitive_models": ["language", "knowledge", "manager"],
            "all_models": self.get_available_models()
        }

    def validate_model_combination(self, model_ids: List[str]) -> Dict[str, Any]:
        """验证模型组合的有效性
        Validate model combination effectiveness
        
        Args:
            model_ids: 要验证的模型ID列表
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        if not model_ids:
            return {"valid": False, "message": "模型列表不能为空"}
        
        # 检查所有模型是否可用
        available_models = self.get_available_models()
        invalid_models = [model_id for model_id in model_ids if model_id not in available_models]
        
        if invalid_models:
            return {
                "valid": False, 
                "message": f"无效的模型ID: {invalid_models}",
                "invalid_models": invalid_models
            }
        
        # 检查模型依赖关系
        dependencies = self._get_model_dependencies()
        missing_deps = []
        
        for model_id in model_ids:
            if model_id in dependencies:
                required_deps = dependencies[model_id]
                missing = [dep for dep in required_deps if dep not in model_ids]
                if missing:
                    missing_deps.append({
                        "model": model_id,
                        "missing_dependencies": missing
                    })
        
        if missing_deps:
            return {
                "valid": False,
                "message": "模型依赖关系不满足",
                "missing_dependencies": missing_deps
            }
        
        return {"valid": True, "message": "模型组合有效"}

    def _get_model_dependencies(self) -> Dict[str, List[str]]:
        """获取模型依赖关系
        Get model dependencies
        
        Returns:
            Dict[str, List[str]]: 模型依赖关系字典
        """
        return {
            "manager": ["language", "knowledge"],
            "vision_video": ["vision_image"],
            "spatial": ["vision_image"],
            "motion": ["spatial", "sensor"],
            "programming": ["language", "knowledge"]
        }

    def get_recommended_combinations(self) -> Dict[str, List[str]]:
        """获取推荐的模型组合
        Get recommended model combinations
        
        Returns:
            Dict[str, List[str]]: 推荐组合字典
        """
        return {
            "basic_interaction": ["manager", "language", "audio"],
            "visual_processing": ["manager", "vision_image", "vision_video"],
            "sensor_analysis": ["manager", "sensor", "spatial"],
            "complete_system": self.get_available_models(),
            "knowledge_programming": ["manager", "language", "knowledge", "programming"],
            "autonomous_control": ["manager", "motion", "computer", "sensor", "spatial"]
        }

# 创建全局联合训练协调器实例
joint_training_coordinator = JointTrainingCoordinator()
