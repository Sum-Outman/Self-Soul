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
Joint Training Coordinator: Manages multi-model collaborative training

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

# Configure logger
logger = logging.getLogger(__name__)


"""
TrainingStrategy Class
"""
class TrainingStrategy(Enum):
    """Training Strategy Enumeration"""
    SEQUENTIAL = "sequential"      # Sequential training
    PARALLEL = "parallel"          # Parallel training  
    ADAPTIVE = "adaptive"          # Adaptive training
    PIPELINE = "pipeline"          # Pipeline training

"""
TrainingTask Class
"""
@dataclass
class TrainingTask:
    """Training Task Data Class"""
    model_id: str
    training_data: Any
    epochs: int = 10
    batch_size: int = 32
    priority: int = 1
    dependencies: List[str] = None


"""
JointTrainingCoordinator Class
"""
class JointTrainingCoordinator:
    """Joint Training Coordinator Class"""

    def __init__(self, model_ids: List[str] = None, config: Dict = None):
        self.training_queue: List[TrainingTask] = []
        self.active_tasks: Set[str] = set()
        self.completed_tasks: Dict[str, Any] = {}
        self.training_strategy = TrainingStrategy.ADAPTIVE
        self.performance_metrics = {}
        self.resource_allocations = {}
        self.knowledge_assistant_enabled = True  # Whether to enable knowledge base assisted training
        
        # Store passed model IDs and config
        self.model_ids = model_ids or []
        self.config = config or {}
        
        # Initialize training strategy based on config
        if config and "training_strategy" in config:
            try:
                self.training_strategy = TrainingStrategy(config["training_strategy"])
            except ValueError:
                logger.warning(f"Invalid training strategy: {config['training_strategy']}, using default strategy")
                self.training_strategy = TrainingStrategy.ADAPTIVE
        
    def schedule_training(self, tasks: List[TrainingTask], 
                         strategy: TrainingStrategy = TrainingStrategy.ADAPTIVE) -> Dict[str, Any]:
        """Schedule training tasks
        
        Args:
            tasks: List of training tasks
            strategy: Training strategy
            
        Returns:
            dict: Scheduling result
        """
        self.training_strategy = strategy
        self.training_queue.extend(tasks)
        
        # Optimize task order based on strategy
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
        """Execute training tasks
        
        Returns:
            dict: Training execution result
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
                
            # Post-training optimization and knowledge integration
            await self._post_training_optimization(results)
            
        except Exception as e:
            error_handler.handle_error(e, "JointTrainingCoordinator", "Joint training execution failed")
            return {"status": "error", "message": str(e)}
        
        return {"status": "success", "results": results}
    
    async def _execute_sequential(self) -> Dict[str, Any]:
        """Execute training tasks sequentially"""
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
        """Execute training tasks in parallel"""
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
        """Execute training tasks adaptively"""
        results = {}
        manager_model = model_registry.get_model("manager")
        
        if manager_model and hasattr(manager_model, 'optimize_training_plan'):
            # Use manager model to optimize training plan
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
            # Fallback to parallel training
            results = await self._execute_parallel()
            
        return results
    
    async def _execute_pipeline(self) -> Dict[str, Any]:
        """Execute training tasks in pipeline"""
        results = {}
        # Implement dataflow-based pipeline training
        # Create training pipeline based on model dependencies
        for task in self.training_queue:
            # Check if dependencies are completed
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
        """Train single model
        
        Args:
            task: Training task
            
        Returns:
            dict: Training result
        """
        try:
            model = model_registry.get_model(task.model_id)
            if not model:
                return {"status": "error", "message": f"Model {task.model_id} not found"}
            
            # Check if model supports training
            if not hasattr(model, 'train') or not callable(getattr(model, 'train')):
                return {"status": "error", "message": f"Model {task.model_id} does not support training"}
            
            start_time = time.time()
            
            # Knowledge base assisted training (if enabled)
            if self.knowledge_assistant_enabled:
                knowledge_model = model_registry.get_model("knowledge")
                if knowledge_model and hasattr(knowledge_model, 'assist_training'):
                    # Get relevant knowledge to assist training
                    # Get training data metadata
                    metadata = task.training_data.metadata if hasattr(task.training_data, 'metadata') else {}
                    
                    knowledge_context = await knowledge_model.assist_training(
                        model_id=task.model_id,
                        training_data_metadata=metadata
                    )
                    # Inject knowledge context into training data
                    if hasattr(task.training_data, 'enhance_with_knowledge'):
                        task.training_data.enhance_with_knowledge(knowledge_context)
            
            # Call model's training method
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
            
            # Record performance metrics
            self._update_performance_metrics(task.model_id, training_result, training_time)
            
            return {
                "status": "success",
                "model_id": task.model_id,
                "training_time": training_time,
                "result": training_result
            }
            
        except Exception as e:
            error_handler.handle_error(e, "JointTrainingCoordinator", f"Training model {task.model_id} failed")
            return {"status": "error", "message": str(e)}
    
    
    def _optimize_task_order(self, tasks: List[TrainingTask]) -> List[TrainingTask]:
        """Optimize task order"""
        # Sort based on model dependencies and priority
        sorted_tasks = sorted(tasks, key=lambda x: (
            -x.priority,  # Priority descending
            len(x.dependencies) if x.dependencies else 0  # Fewer dependencies first
        ))
        return sorted_tasks
    
    def _create_pipeline(self, tasks: List[TrainingTask]) -> List[TrainingTask]:
        """Create training pipeline"""
        # Create pipeline based on model dependencies
        pipeline_tasks = []
        processed_models = set()
        
        # First process tasks without dependencies
        for task in tasks:
            if not task.dependencies or all(dep in processed_models for dep in task.dependencies):
                pipeline_tasks.append(task)
                processed_models.add(task.model_id)
        
        # Then process tasks with dependencies
        remaining_tasks = [t for t in tasks if t not in pipeline_tasks]
        while remaining_tasks:
            for task in remaining_tasks[:]:
                if all(dep in processed_models for dep in task.dependencies):
                    pipeline_tasks.append(task)
                    processed_models.add(task.model_id)
                    remaining_tasks.remove(task)
        
        return pipeline_tasks
    
    def _estimate_training_time(self) -> float:
        """Estimate training time"""
        total_time = 0
        for task in self.training_queue:
            # Estimate based on historical performance data
            avg_time = self.performance_metrics.get(task.model_id, {}).get('avg_training_time', 60)
            total_time += avg_time * task.epochs
        
        return total_time
    
    
    def _update_performance_metrics(self, model_id: str, result: Dict[str, Any], training_time: float):
        """Update performance metrics"""
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
        """Post-training optimization"""
        # Update knowledge base
        knowledge_model = model_registry.get_model("knowledge")
        if knowledge_model and hasattr(knowledge_model, 'integrate_training_results'):
            await knowledge_model.integrate_training_results(results)
        
        # Optimize model collaboration
        manager_model = model_registry.get_model("manager")
        if manager_model and hasattr(manager_model, 'optimize_model_collaboration'):
            await manager_model.optimize_model_collaboration(results, self.performance_metrics)
        
        # Enhance knowledge base learning capability
        if knowledge_model and hasattr(knowledge_model, 'enhance_knowledge_learning'):
            # Extract learning data from training results
            learning_data = {
                "text_sources": [],
                "structured_sources": [],
                "external_sources": []
            }
            
            for model_id, result in results.items():
                if "training_data" in result and result["training_data"]:
                    learning_data["text_sources"].append({
                        "type": "training_result",
                        "content": f"Training result for model {model_id}: {result}",
                        "domain": model_id
                    })
            
            # Apply enhanced learning
            enhancement_result = knowledge_model.enhance_knowledge_learning(learning_data)
            logger.info(f"Knowledge base enhancement completed: {enhancement_result}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get training status"""
        return {
            "queue_size": len(self.training_queue),
            "active_tasks": list(self.active_tasks),
            "completed_tasks": list(self.completed_tasks.keys()),
            "strategy": self.training_strategy.value,
            "performance_metrics": self.performance_metrics
        }
    
    
    def clear_queue(self):
        """Clear training queue"""
        self.training_queue.clear()
        self.active_tasks.clear()
        self.completed_tasks.clear()

    def get_available_models(self) -> List[str]:
        """Get available models list
        
        Returns:
            List[str]: List of available model IDs
        """
        return [
            "manager",           # A - Management model
            "language",          # B - Large language model
            "audio",             # C - Audio processing model
            "vision_image",      # D - Image vision processing model
            "vision_video",      # E - Video stream vision processing model
            "spatial",           # F - Binocular spatial positioning perception model
            "sensor",            # G - Sensor perception model
            "computer",          # H - Computer control model
            "motion",            # I - Motion and actuator control model
            "knowledge",         # J - Knowledge base expert model
            "programming"        # K - Programming model
        ]

    def get_model_groups(self) -> Dict[str, List[str]]:
        """Get model groups configuration
        
        Returns:
            Dict[str, List[str]]: Model groups dictionary
        """
        return {
            "basic_models": ["language", "audio", "vision_image", "vision_video"],
            "perception_models": ["sensor", "spatial", "audio", "vision_image", "vision_video"],
            "action_models": ["computer", "motion", "programming"],
            "cognitive_models": ["language", "knowledge", "manager"],
            "all_models": self.get_available_models()
        }

    def validate_model_combination(self, model_ids: List[str]) -> Dict[str, Any]:
        """Validate model combination effectiveness
        
        Args:
            model_ids: List of model IDs to validate
            
        Returns:
            Dict[str, Any]: Validation result
        """
        if not model_ids:
            return {"valid": False, "message": "Model list cannot be empty"}
        
        # Check if all models are available
        available_models = self.get_available_models()
        invalid_models = [model_id for model_id in model_ids if model_id not in available_models]
        
        if invalid_models:
            return {
                "valid": False, 
                "message": f"Invalid model IDs: {invalid_models}",
                "invalid_models": invalid_models
            }
        
        # Check model dependencies
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
                "message": "Model dependencies not satisfied",
                "missing_dependencies": missing_deps
            }
        
        return {"valid": True, "message": "Model combination is valid"}

    def _get_model_dependencies(self) -> Dict[str, List[str]]:
        """Get model dependencies
        
        Returns:
            Dict[str, List[str]]: Model dependencies dictionary
        """
        return {
            "manager": ["language", "knowledge"],
            "vision_video": ["vision_image"],
            "spatial": ["vision_image"],
            "motion": ["spatial", "sensor"],
            "programming": ["language", "knowledge"]
        }

    def get_recommended_combinations(self) -> Dict[str, List[str]]:
        """Get recommended model combinations
        
        Returns:
            Dict[str, List[str]]: Recommended combinations dictionary
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
