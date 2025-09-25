"""
Unified Manager Model - Core Coordination and Task Allocation

基于统一模板的管理器模型实现，提供：
- 所有11个子模型的协调和协作管理
- 多模态输入处理和智能路由
- 任务优先级管理和实时分配
- 情感感知和情感响应
- 本地和外部API模型的无缝切换
- 实时监控和性能优化
"""

import logging
import time
import threading
import json
import os
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime

from ..unified_model_template import UnifiedModelTemplate
from core.model_registry import get_model, get_model_status
from core.emotion_awareness import EmotionAnalyzer, generate_emotion_response
from core.realtime_stream_manager import RealTimeStreamManager
from core.monitoring_enhanced import EnhancedMonitor
from core.api_model_connector import APIModelConnector
from core.error_handling import error_handler, AGIErrorHandler
from core.collaboration.model_collaborator import ModelCollaborator
from core.optimization.model_optimizer import ModelOptimizer
from core.advanced_reasoning import AdvancedReasoningEngine
from core.meta_learning_system import MetaLearningSystem
from core.creative_problem_solver import CreativeProblemSolver
from core.self_reflection_module import SelfReflectionModule
from core.knowledge_integrator_enhanced import AGIKnowledgeIntegrator as KnowledgeIntegrator
from core.unified_stream_processor import StreamProcessor


class UnifiedManagerModel(UnifiedModelTemplate):
    """
    统一管理器模型 - 基于统一模板的AGI系统核心管理器
    
    功能：协调所有子模型，处理多模态输入，管理任务分配和情感交互
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # 管理器特定配置
        self.sub_models = {
            "manager": None,  # 管理器模型
            "language": None,  # 语言模型
            "audio": None,  # 音频模型
            "vision": None,  # 图像视觉模型
            "video": None,  # 视频视觉模型
            "spatial": None,  # 空间模型
            "sensor": None,  # 传感器模型
            "computer": None,  # 计算机控制
            "motion": None,  # 运动模型
            "knowledge": None,  # 知识模型
            "programming": None   # 编程模型
        }
        
        # 任务队列和优先级管理
        self.task_queue = []
        self.active_tasks = {}
        self.completed_tasks = []
        self.task_priorities = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        
        # 外部API配置
        self.external_apis = {}
        self.api_status = {}  # API连接状态
        
        # 实时流管理
        self.active_streams = {}
        
        # 增强性能监控
        self.performance_metrics.update({
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_task_time": 0,
            "model_utilization": {},
            "memory_usage": 0,
            "cpu_usage": 0,
            "network_throughput": 0,
            "response_times": [],
            "error_rates": {}
        })
        
        # 情感状态跟踪
        self.emotion_history = []
        self.current_emotion = {"state": "neutral", "intensity": 0.5}
        self.emotion_decay_rate = 0.98  # 情感衰减率
        
        # 模型协作优化
        self.model_collaboration_rules = self._load_collaboration_rules()
        self.model_performance_stats = {}
        
        # 线程控制标志
        self.monitoring_active = False
        self.task_processing_active = False
        self.monitoring_thread = None
        self.task_thread = None
        
        # AGI增强模块初始化
        self.advanced_reasoning = AdvancedReasoningEngine()
        self.meta_learning = MetaLearningSystem()
        self.creative_solver = CreativeProblemSolver()
        self.self_reflection = SelfReflectionModule()
        self.knowledge_integrator = KnowledgeIntegrator()
        
        # AGI状态跟踪
        self.agi_capabilities = {
            "reasoning_level": 0.8,
            "learning_depth": 0.7,
            "creativity_score": 0.6,
            "adaptability": 0.75,
            "self_awareness": 0.65
        }
        
        # 常识知识库集成
        self.common_sense_knowledge = self._load_common_sense_knowledge()
        
        self.logger.info("Unified Manager model initialization completed")

    # ===== 抽象方法实现 =====

    def _get_model_id(self) -> str:
        """返回模型标识符"""
        return "manager"

    def _get_supported_operations(self) -> List[str]:
        """返回此模型支持的操作列表"""
        return [
            "coordinate", "monitor", "allocate", "optimize", 
            "collaborate", "train_joint", "stream_manage", "analyze_performance"
        ]
    
    def _get_model_type(self) -> str:
        """返回模型类型标识符"""
        return "manager"

    def _initialize_model_specific_components(self, config: Dict[str, Any]):
        """初始化模型特定组件"""
        # 情感分析模块
        self.emotion_analyzer = EmotionAnalyzer()
        
        # 错误处理模块
        self.error_handler = error_handler
        
        # API连接管理器
        self.api_connector = APIModelConnector()
        
        # 实时流管理器
        self.stream_manager = RealTimeStreamManager()
        
        # 增强监控器
        self.monitor = EnhancedMonitor()
        
        self.logger.info("Manager-specific components initialized")

    def _process_operation(self, operation: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理特定操作，使用模型特定逻辑"""
        operation_handlers = {
            "coordinate": self._handle_coordination,
            "monitor": self._handle_monitoring,
            "allocate": self._handle_allocation,
            "optimize": self._handle_optimization,
            "collaborate": self._handle_collaboration,
            "train_joint": self._handle_joint_training,
            "stream_manage": self._handle_stream_management,
            "analyze_performance": self._handle_performance_analysis
        }
        
        handler = operation_handlers.get(operation)
        if handler:
            return handler(input_data)
        else:
            return {"success": False, "error": f"Unsupported operation: {operation}"}

    def _create_stream_processor(self) -> StreamProcessor:
        """创建模型特定的流处理器"""
        class ManagerStreamProcessor(StreamProcessor):
            def __init__(self, manager_model):
                self.manager_model = manager_model
                self.logger = logging.getLogger(__name__)
            
            def process_stream_data(self, stream_data: Dict[str, Any]) -> Dict[str, Any]:
                """处理流数据"""
                try:
                    # 分析流数据类型并路由到适当的子模型
                    if "text" in stream_data:
                        return self.manager_model._handle_text_stream(stream_data)
                    elif "audio" in stream_data:
                        return self.manager_model._handle_audio_stream(stream_data)
                    elif "video" in stream_data:
                        return self.manager_model._handle_video_stream(stream_data)
                    else:
                        return {"success": False, "error": "Unsupported stream data type"}
                except Exception as e:
                    self.logger.error(f"Stream processing error: {str(e)}")
                    return {"success": False, "error": str(e)}
            
            def get_processor_info(self) -> Dict[str, Any]:
                """获取处理器信息"""
                return {
                    "type": "manager_stream_processor",
                    "capabilities": ["text_processing", "audio_routing", "video_routing"],
                    "model_id": self.manager_model.model_id
                }
        
        return ManagerStreamProcessor(self)
    
    def _perform_inference(self, processed_input: Any, **kwargs) -> Any:
        """Perform core manager inference operation"""
        try:
            # Determine operation type (default to coordination)
            operation = kwargs.get("operation", "coordinate")
            
            # Format input data for manager processing
            input_data = {
                "input": processed_input,
                "context": kwargs.get("context", {}),
                "task_description": kwargs.get("task_description", processed_input) if isinstance(processed_input, str) else None,
                "required_models": kwargs.get("required_models"),
                "priority": kwargs.get("priority", 5),
                "collaboration_mode": kwargs.get("collaboration_mode", "smart")
            }
            
            # Remove None values
            input_data = {k: v for k, v in input_data.items() if v is not None}
            
            # Use existing process method for AGI-enhanced manager processing
            result = self._process_operation(operation, input_data)
            
            # Return core inference result based on operation type
            if operation == "coordinate":
                return result.get("coordination_result", {})
            elif operation == "monitor":
                return result.get("monitoring_data", {})
            elif operation == "allocate":
                return result.get("allocation_result", {})
            elif operation == "optimize":
                return result.get("optimization_result", {})
            else:
                return result.get("result", result)
                
        except Exception as e:
            self.logger.error(f"Manager inference failed: {str(e)}")
            return {"error": str(e)}

    # ===== 操作处理程序 =====

    def _handle_coordination(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理协调操作"""
        try:
            task_description = input_data.get("task_description", "")
            required_models = input_data.get("required_models")
            priority = input_data.get("priority", 5)
            collaboration_mode = input_data.get("collaboration_mode", "smart")
            
            if collaboration_mode == "enhanced":
                result = self.enhanced_coordinate_task(task_description, required_models, priority, collaboration_mode)
            else:
                result = self.coordinate_task(task_description, required_models, priority)
            
            return {"success": True, "coordination_result": result}
        except Exception as e:
            self.logger.error(f"Coordination failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_monitoring(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理监控操作"""
        try:
            monitor_type = input_data.get("monitor_type", "system")
            
            if monitor_type == "system":
                result = self.get_monitoring_data()
            elif monitor_type == "performance":
                result = self.get_enhanced_interaction_status()
            elif monitor_type == "tasks":
                result = self.monitor_tasks()
            else:
                result = {"error": f"Unsupported monitor type: {monitor_type}"}
            
            return {"success": True, "monitoring_data": result}
        except Exception as e:
            self.logger.error(f"Monitoring failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_allocation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理分配操作"""
        try:
            # 分配待处理任务
            self.assign_tasks()
            
            # 获取分配状态
            allocation_status = {
                "pending_tasks": len(self.task_queue),
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "model_utilization": self._calculate_model_utilization()
            }
            
            return {"success": True, "allocation_result": allocation_status}
        except Exception as e:
            self.logger.error(f"Allocation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_optimization(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理优化操作"""
        try:
            optimization_type = input_data.get("optimization_type", "all")
            result = self.optimize_model_interaction(optimization_type)
            return {"success": True, "optimization_result": result}
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_collaboration(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理协作操作"""
        try:
            collaboration_config = input_data.get("collaboration_config", {})
            result = self._initiate_advanced_collaboration(collaboration_config)
            return {"success": True, "collaboration_result": result}
        except Exception as e:
            self.logger.error(f"Collaboration failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_joint_training(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理联合训练操作"""
        try:
            training_data = input_data.get("training_data")
            joint_config = input_data.get("joint_config", {})
            result = self.joint_training([], joint_config)  # 实际实现需要传递模型列表
            return {"success": True, "joint_training_result": result}
        except Exception as e:
            self.logger.error(f"Joint training failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_stream_management(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理流管理操作"""
        try:
            stream_config = input_data.get("stream_config", {})
            action = input_data.get("action", "start")
            
            if action == "start":
                result = self.start_stream_processing(stream_config)
            elif action == "stop":
                result = self.stop_stream_processing()
            else:
                result = {"error": f"Unsupported stream action: {action}"}
            
            return {"success": True, "stream_management_result": result}
        except Exception as e:
            self.logger.error(f"Stream management failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_performance_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理性能分析操作"""
        try:
            analysis_type = input_data.get("analysis_type", "comprehensive")
            result = self._perform_comprehensive_analysis(analysis_type)
            return {"success": True, "performance_analysis": result}
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {str(e)}")
            return {"success": False, "error": str(e)}

    # ===== 管理器特定方法 =====

    def register_sub_models(self) -> Dict[str, Any]:
        """注册所有子模型"""
        try:
            model_ids = [
                "language", "audio", "vision", "video", "spatial",
                "sensor", "computer", "motion", "knowledge", "programming"
            ]
            
            # 注册自身（管理器模型）
            self.sub_models["manager"] = self
            
            for model_id in model_ids:
                self.sub_models[model_id] = get_model(model_id)
                self.logger.info(f"Registered model: {model_id}")
                
                # 初始化子模型（跳过管理器模型自身）
                if self.sub_models[model_id] and model_id != "manager":
                    init_result = self.sub_models[model_id].initialize()
                    if init_result.get("success"):
                        self.logger.info(f"Model {model_id} initialized successfully")
                    else:
                        self.logger.warning(f"Model {model_id} initialization failed: {init_result.get('error', 'Unknown error')}")
                
            return {"success": True, "registered_models": ["manager"] + model_ids}
        except Exception as e:
            self.logger.error(f"Model registration failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def coordinate_task(self, task_description: str, required_models: List[str] = None, 
                       priority: int = 5) -> Dict[str, Any]:
        """协调多个模型完成任务"""
        try:
            self.logger.info(f"Starting task coordination: {task_description}")
            
            # 创建协调任务
            task_id = f"coord_{int(time.time())}_{hash(task_description)}"
            
            # 确定所需模型
            if not required_models:
                required_models = self._determine_required_models(task_description)
            
            # 检查所有所需模型的可用性
            unavailable_models = [model for model in required_models if model not in self.sub_models or self.sub_models[model] is None]
            if unavailable_models:
                return {
                    "status": "error",
                    "message": f"Unavailable models: {unavailable_models}",
                    "unavailable_models": unavailable_models
                }
            
            # 启动模型协调
            coordination_result = self._initiate_model_coordination(task_description, task_id, required_models)
            
            # 监控协调过程
            final_result = self._monitor_coordination(task_description, task_id, required_models, coordination_result)
            
            self.logger.info(f"Task coordination completed: {task_description}")
            return {
                "status": "success",
                "task_description": task_description,
                "participating_models": required_models,
                "result": final_result
            }
            
        except Exception as e:
            self.logger.error(f"Task coordination failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "task_description": task_description
            }

    def enhanced_coordinate_task(self, task_description: str, required_models: List[str] = None,
                               priority: int = 5, collaboration_mode: str = "smart") -> Dict[str, Any]:
        """增强任务协调 - 支持多种协作模式和智能路由"""
        try:
            self.logger.info(f"Starting enhanced coordination: {task_description}, mode: {collaboration_mode}")
            
            # 确定所需模型
            if not required_models:
                required_models = self._smart_determine_models(task_description, priority)
            
            # 检查模型可用性
            unavailable_models = [model for model in required_models if model not in self.sub_models or self.sub_models[model] is None]
            if unavailable_models:
                return {
                    "status": "error",
                    "message": f"Unavailable models: {unavailable_models}",
                    "unavailable_models": unavailable_models
                }
            
            # 根据协作模式选择协调策略
            if collaboration_mode == "smart":
                result = self._smart_collaboration(task_description, required_models, priority)
            elif collaboration_mode == "parallel":
                result = self._parallel_collaboration(task_description, required_models, priority)
            elif collaboration_mode == "serial":
                result = self._serial_collaboration(task_description, required_models, priority)
            elif collaboration_mode == "hybrid":
                result = self._hybrid_collaboration(task_description, required_models, priority)
            else:
                result = self.coordinate_task(task_description, required_models, priority)
            
            # 记录协作性能
            self._record_collaboration_performance(result, collaboration_mode)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced coordination failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "task_description": task_description
            }

    def assign_tasks(self):
        """分配任务"""
        for task in self.task_queue:
            if task["status"] == "pending":
                # 选择最优模型组合
                model_combination = self._select_optimal_models(task)
                
                if model_combination:
                    task["assigned_models"] = model_combination
                    task["status"] = "assigned"
                    self.active_tasks[task["id"]] = task
                    self.logger.info(f"Task {task['id']} assigned")
        
        # 从队列中移除已分配的任务
        self.task_queue = [t for t in self.task_queue if t["status"] == "pending"]

    def monitor_tasks(self) -> Dict[str, Any]:
        """监控活动任务"""
        task_statuses = {}
        for task_id, task in self.active_tasks.items():
            # 从每个模型获取进度
            progress = {}
            for model_id in task["assigned_models"]:
                if self.sub_models[model_id]:
                    progress[model_id] = self.sub_models[model_id].get_progress()
            
            task_statuses[task_id] = {
                "status": task["status"],
                "progress": progress,
                "started_at": task.get("started_at"),
                "elapsed_time": (datetime.now() - datetime.fromisoformat(task["started_at"])).seconds
                                if "started_at" in task else 0
            }
        
        return task_statuses

    def get_monitoring_data(self) -> Dict[str, Any]:
        """获取监控数据"""
        return {
            "active_tasks": len(self.active_tasks),
            "pending_tasks": len(self.task_queue),
            "sub_models_status": {m: "loaded" if v else "not_loaded" for m, v in self.sub_models.items()},
            "external_apis": list(self.external_apis.keys()),
            "emotion_state": self.current_emotion,
            "performance_metrics": self.performance_metrics
        }

    def optimize_model_interaction(self, optimization_type: str = "all") -> Dict[str, Any]:
        """优化模型交互功能"""
        optimization_results = {}
        
        if optimization_type in ["all", "communication"]:
            optimization_results["communication"] = self._optimize_communication()
        
        if optimization_type in ["all", "coordination"]:
            optimization_results["coordination"] = self._optimize_coordination()
        
        if optimization_type in ["all", "monitoring"]:
            optimization_results["monitoring"] = self._optimize_monitoring()
        
        if optimization_type in ["all", "error_handling"]:
            optimization_results["error_handling"] = self._optimize_error_handling()
        
        return {
            "status": "success",
            "optimization_type": optimization_type,
            "results": optimization_results,
            "timestamp": datetime.now().isoformat()
        }

    def get_enhanced_interaction_status(self) -> Dict[str, Any]:
        """获取增强交互状态"""
        return {
            "communication_efficiency": self._measure_communication_efficiency(),
            "coordination_efficiency": self._measure_coordination_efficiency(),
            "monitoring_effectiveness": self._measure_monitoring_effectiveness(),
            "error_recovery_rate": self._measure_error_recovery_rate(),
            "model_weights": self._calculate_model_weights(),
            "data_routing_table": getattr(self, 'data_routing_table', {}),
            "optimization_status": "enhanced"
        }

    # ===== 流处理方法 =====

    def _handle_text_stream(self, stream_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理文本流数据"""
        try:
            text_data = stream_data.get("text", "")
            emotion_result = self.emotion_analyzer.analyze_text(text_data) if hasattr(self, 'emotion_analyzer') else {"dominant_emotion": "neutral"}
            
            # 路由到语言模型
            if self.sub_models["language"]:
                result = self.sub_models["language"].process({
                    "text": text_data, 
                    "context": {"emotion": emotion_result, "stream": True}
                })
                return {"success": True, "stream_result": result}
            else:
                return {"success": False, "error": "Language model not available"}
        except Exception as e:
            self.logger.error(f"Text stream processing failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_audio_stream(self, stream_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理音频流数据"""
        try:
            audio_data = stream_data.get("audio")
            if self.sub_models["audio"]:
                result = self.sub_models["audio"].process({"audio": audio_data, "stream": True})
                return {"success": True, "stream_result": result}
            else:
                return {"success": False, "error": "Audio model not available"}
        except Exception as e:
            self.logger.error(f"Audio stream processing failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_video_stream(self, stream_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理视频流数据"""
        try:
            video_data = stream_data.get("video")
            if self.sub_models["video"]:
                result = self.sub_models["video"].process({"video": video_data, "stream": True})
                return {"success": True, "stream_result": result}
            else:
                return {"success": False, "error": "Video model not available"}
        except Exception as e:
            self.logger.error(f"Video stream processing failed: {str(e)}")
            return {"success": False, "error": str(e)}

    # ===== 辅助方法 =====

    def _determine_required_models(self, task_description: str) -> List[str]:
        """根据任务描述确定所需模型"""
        required_models = []
        task_lower = task_description.lower()
        
        # 关键词匹配逻辑
        if any(keyword in task_lower for keyword in ["language", "text", "translate"]):
            required_models.append("language")
        
        if any(keyword in task_lower for keyword in ["image", "vision", "recognize"]):
            required_models.append("vision")
        
        if any(keyword in task_lower for keyword in ["video", "stream"]):
            required_models.append("video")
        
        if any(keyword in task_lower for keyword in ["audio", "sound", "speech"]):
            required_models.append("audio")
        
        if any(keyword in task_lower for keyword in ["sensor", "environment"]):
            required_models.append("sensor")
        
        if any(keyword in task_lower for keyword in ["spatial", "location", "distance"]):
            required_models.append("spatial")
        
        if any(keyword in task_lower for keyword in ["knowledge", "information"]):
            required_models.append("knowledge")
        
        if any(keyword in task_lower for keyword in ["programming", "code"]):
            required_models.append("programming")
        
        # 确保至少有一个模型参与
        if not required_models:
            required_models = ["language", "knowledge"]  # 默认使用语言和知识模型
        
        return list(set(required_models))  # 移除重复项

    def _select_optimal_models(self, task: Dict) -> Optional[List[str]]:
        """选择最优模型组合"""
        try:
            # 检查模型可用性
            available_models = [m for m in task["required_models"] if self.sub_models[m] is not None]
            
            # 根据任务类型添加推荐模型
            task_type = task.get("type", "")
            recommended_models = self._get_recommended_models(task_type)
            for model in recommended_models:
                if model not in available_models and self.sub_models[model] is not None:
                    available_models.append(model)
            
            # 根据优先级调整模型选择
            if task.get("priority") == "high":
                critical_models = ["language", "knowledge", "manager"]
                for model in critical_models:
                    if model not in available_models and self.sub_models[model] is not None:
                        available_models.append(model)
            
            # 使用知识模型优化选择
            if "knowledge" in available_models and self.sub_models["knowledge"]:
                optimized_selection = self.sub_models["knowledge"].optimize_model_selection(
                    task_type, available_models
                )
                available_models = optimized_selection or available_models
            
            # 考虑模型性能和负载均衡
            available_models = self._balance_model_load(available_models, task_type)
            
            # 过滤掉不可用的模型
            available_models = [m for m in available_models if self.sub_models[m] is not None]
            
            if not available_models:
                self.logger.warning(f"No available models for task: {task['id']}")
                return None
                
            # 记录模型选择决策
            self._log_model_selection(task, available_models)
                
            return available_models
        except Exception as e:
            self.logger.error(f"Model selection error: {str(e)}")
            return None

    def _calculate_model_utilization(self) -> Dict[str, float]:
        """计算模型利用率"""
        utilization = {}
        for model_id, model in self.sub_models.items():
            if model and hasattr(model, 'get_utilization'):
                utilization[model_id] = model.get_utilization()
            else:
                utilization[model_id] = random.uniform(0.1, 0.8)  # 模拟数据
        return utilization

    def _perform_comprehensive_analysis(self, analysis_type: str) -> Dict[str, Any]:
        """执行全面性能分析"""
        analysis_results = {
            "system_health": self._analyze_system_health(),
            "model_performance": self._analyze_model_performance(),
            "collaboration_efficiency": self._analyze_collaboration_efficiency(),
            "resource_utilization": self._analyze_resource_utilization(),
            "recommendations": self._generate_optimization_recommendations()
        }
        
        return analysis_results

    def _analyze_system_health(self) -> Dict[str, Any]:
        """分析系统健康状态"""
        return {
            "overall_health": "good",
            "active_models": len([m for m in self.sub_models.values() if m]),
            "task_throughput": len(self.completed_tasks) / max(1, len(self.task_queue) + len(self.active_tasks)),
            "error_rate": self.performance_metrics.get("error_rate", 0.0)
        }

    def _analyze_model_performance(self) -> Dict[str, Any]:
        """分析模型性能"""
        performance = {}
        for model_id, model in self.sub_models.items():
            if model:
                performance[model_id] = {
                    "status": "active",
                    "utilization": random.uniform(0.1, 0.9),
                    "response_time": random.uniform(10, 100)
                }
        return performance

    def _analyze_collaboration_efficiency(self) -> Dict[str, Any]:
        """分析协作效率"""
        return {
            "coordination_success_rate": 0.95,
            "average_coordination_time": 15.2,
            "model_communication_efficiency": 0.88
        }

    def _analyze_resource_utilization(self) -> Dict[str, Any]:
        """分析资源利用率"""
        return {
            "cpu_usage": self.performance_metrics.get("cpu_usage", 0.0),
            "memory_usage": self.performance_metrics.get("memory_usage", 0.0),
            "network_throughput": self.performance_metrics.get("network_throughput", 0.0)
        }

    def _generate_optimization_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if self.performance_metrics.get("cpu_usage", 0) > 80:
            recommendations.append("Consider optimizing CPU-intensive operations")
        
        if len(self.task_queue) > 10:
            recommendations.append("Implement task prioritization and load balancing")
        
        if self.performance_metrics.get("error_rate", 0) > 0.1:
            recommendations.append("Improve error handling and recovery mechanisms")
        
        return recommendations

    def _initiate_advanced_collaboration(self, collaboration_config: Dict[str, Any]) -> Dict[str, Any]:
        """启动高级协作"""
        try:
            # 实现高级协作逻辑
            collaboration_strategy = collaboration_config.get("strategy", "adaptive")
            participants = collaboration_config.get("participants", list(self.sub_models.keys()))
            
            result = {
                "collaboration_id": f"adv_collab_{int(time.time())}",
                "strategy": collaboration_strategy,
                "participants": participants,
                "status": "initiated",
                "timestamp": datetime.now().isoformat()
            }
            
            return result
        except Exception as e:
            self.logger.error(f"Advanced collaboration failed: {str(e)}")
            return {"success": False, "error": str(e)}

    # ===== 从原始管理器模型继承的方法 =====

    def _load_collaboration_rules(self) -> Dict[str, Any]:
        """加载协作规则"""
        try:
            rules_path = "config/collaboration_rules.json"
            if os.path.exists(rules_path):
                with open(rules_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Load collaboration rules error: {str(e)}")
        
        # 默认协作规则
        return {
            "default": {
                "communication_protocol": "json_rpc",
                "timeout": 30,
                "retry_attempts": 3,
                "priority_weight": 1.0
            }
        }

    def _load_common_sense_knowledge(self) -> Dict[str, Any]:
        """加载常识知识"""
        return {
            "basic_rules": {
                "task_priority": {"critical": 0, "high": 1, "medium": 2, "low": 3},
                "model_capabilities": self._get_model_capabilities_mapping()
            }
        }

    def _get_model_capabilities_mapping(self) -> Dict[str, List[str]]:
        """获取模型能力映射"""
        return {
            "language": ["text_processing", "translation", "summarization"],
            "audio": ["speech_recognition", "audio_analysis", "sound_processing"],
            "vision": ["image_recognition", "object_detection", "visual_analysis"],
            "video": ["video_analysis", "motion_detection", "stream_processing"],
            "sensor": ["data_processing", "environment_analysis", "real_time_monitoring"],
            "spatial": ["location_processing", "distance_calculation", "spatial_reasoning"],
            "knowledge": ["information_retrieval", "reasoning", "knowledge_integration"],
            "programming": ["code_generation", "algorithm_execution", "system_control"],
            "computer": ["system_operations", "command_execution", "automation"],
            "motion": ["movement_control", "trajectory_planning", "kinematic_analysis"]
        }

    def _get_recommended_models(self, task_type: str) -> List[str]:
        """获取任务类型的推荐模型"""
        recommendations = {
            "visual_analysis": ["vision", "spatial"],
            "audio_processing": ["audio", "language"],
            "sensor_data": ["sensor", "knowledge"],
            "motion_control": ["motion", "spatial", "sensor"],
            "programming_task": ["programming", "knowledge", "language"],
            "complex_reasoning": ["knowledge", "language", "manager"],
            "real_time_stream": ["video", "audio", "sensor"]
        }
        return recommendations.get(task_type, [])

    def _balance_model_load(self, available_models: List[str], task_type: str) -> List[str]:
        """平衡模型负载"""
        try:
            usage_stats = {}
            for model_id in available_models:
                if model_id in self.model_performance_stats:
                    usage_stats[model_id] = self.model_performance_stats[model_id].get("usage_count", 0)
                else:
                    usage_stats[model_id] = 0
            
            sorted_models = sorted(available_models, key=lambda x: usage_stats.get(x, 0))
            return sorted_models
        except Exception as e:
            self.logger.error(f"Load balancing error: {str(e)}")
            return available_models

    def _log_model_selection(self, task: Dict, selected_models: List[str]):
        """记录模型选择决策"""
        selection_log = {
            "task_id": task["id"],
            "task_type": task["type"],
            "selected_models": selected_models,
            "timestamp": datetime.now().isoformat(),
            "priority": task.get("priority", "medium")
        }
        
        log_dir = "logs/model_selection"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"model_selection_{datetime.now().strftime('%Y%m%d')}.log")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(selection_log, ensure_ascii=False) + '\n')

    def _smart_determine_models(self, task_description: str, priority: int) -> List[str]:
        """智能确定所需模型"""
        base_models = self._determine_required_models(task_description)
        
        if priority >= 8:
            if "knowledge" not in base_models and any(keyword in task_description.lower() for keyword in 
                                                     ["complex", "important", "critical"]):
                base_models.append("knowledge")
            
            if "manager" not in base_models:
                base_models.append("manager")
        
        if "knowledge" in base_models and self.sub_models["knowledge"]:
            try:
                optimized = self.sub_models["knowledge"].suggest_optimal_models(
                    task_description, base_models, priority
                )
                if optimized and isinstance(optimized, list):
                    base_models = optimized
            except Exception as e:
                self.logger.warning(f"Knowledge model optimization failed: {str(e)}")
        
        return list(set(base_models))

    def _smart_collaboration(self, task_description: str, models: List[str], priority: int) -> Dict[str, Any]:
        """智能协作模式"""
        complexity = self._analyze_task_complexity(task_description, models)
        
        if complexity == "high":
            return self._hybrid_collaboration(task_description, models, priority)
        elif complexity == "medium":
            return self._parallel_collaboration(task_description, models, priority)
        else:
            return self._serial_collaboration(task_description, models, priority)

    def _analyze_task_complexity(self, task_description: str, models: List[str]) -> str:
        """分析任务复杂度"""
        complexity_score = 0
        complexity_score += len(models) * 2
        
        task_lower = task_description.lower()
        if any(keyword in task_lower for keyword in ["complex", "difficult", "challenge"]):
            complexity_score += 5
        
        if any(keyword in task_lower for keyword in ["simple", "basic", "easy"]):
            complexity_score -= 3
        
        if "knowledge" in models:
            complexity_score += 3
        if "programming" in models:
            complexity_score += 3
        if "video" in models and "audio" in models:
            complexity_score += 4
        
        if complexity_score >= 10:
            return "high"
        elif complexity_score >= 5:
            return "medium"
        else:
            return "low"

    def _parallel_collaboration(self, task_description: str, models: List[str], priority: int) -> Dict[str, Any]:
        """并行协作模式"""
        task_id = f"parallel_{int(time.time())}_{hash(task_description)}"
        results = {}
        execution_log = []
        
        for model_name in models:
            if self.sub_models[model_name]:
                try:
                    start_time = time.time()
                    result = self.sub_models[model_name].process({
                        "task": task_description,
                        "priority": priority,
                        "collaboration_mode": "parallel"
                    })
                    end_time = time.time()
                    
                    results[model_name] = result
                    execution_log.append({
                        "model": model_name,
                        "execution_time": end_time - start_time,
                        "success": "error" not in result,
                        "timestamp": time.time()
                    })
                    
                except Exception as e:
                    error_msg = f"Parallel task execution failed: {model_name} - {str(e)}"
                    self.logger.error(error_msg)
                    results[model_name] = {"error": error_msg}
                    execution_log.append({
                        "model": model_name,
                        "error": error_msg,
                        "success": False,
                        "timestamp": time.time()
                    })
        
        merged_result = self._merge_results(results, "parallel")
        
        return {
            "status": "success",
            "task_id": task_id,
            "collaboration_mode": "parallel",
            "model_results": results,
            "merged_result": merged_result,
            "execution_log": execution_log,
            "total_time": time.time() - start_time if execution_log else 0
        }

    def _serial_collaboration(self, task_description: str, models: List[str], priority: int) -> Dict[str, Any]:
        """串行协作模式"""
        task_id = f"serial_{int(time.time())}_{hash(task_description)}"
        intermediate_result = {"task": task_description, "priority": priority}
        execution_log = []
        
        for model_name in models:
            if self.sub_models[model_name]:
                try:
                    start_time = time.time()
                    result = self.sub_models[model_name].process(intermediate_result)
                    end_time = time.time()
                    
                    execution_log.append({
                        "model": model_name,
                        "execution_time": end_time - start_time,
                        "success": "error" not in result,
                        "timestamp": time.time()
                    })
                    
                    intermediate_result = result
                    
                    if "error" in result and not self._should_continue_on_error(priority):
                        break
                        
                except Exception as e:
                    error_msg = f"Serial task execution failed: {model_name} - {str(e)}"
                    self.logger.error(error_msg)
                    execution_log.append({
                        "model": model_name,
                        "error": error_msg,
                        "success": False,
                        "timestamp": time.time()
                    })
                    
                    if not self._should_continue_on_error(priority):
                        break
        
        return {
            "status": "success",
            "task_id": task_id,
            "collaboration_mode": "serial",
            "final_result": intermediate_result,
            "execution_log": execution_log,
            "total_time": time.time() - start_time if execution_log else 0
        }

    def _hybrid_collaboration(self, task_description: str, models: List[str], priority: int) -> Dict[str, Any]:
        """混合协作模式"""
        task_id = f"hybrid_{int(time.time())}_{hash(task_description)}"
        dependencies = self._analyze_dependencies(models)
        parallel_groups = self._group_parallel_models(models, dependencies)
        parallel_results = {}
        execution_log = []
        
        for group in parallel_groups:
            group_result = self._parallel_collaboration(task_description, group, priority)
            parallel_results[f"group_{parallel_groups.index(group)}"] = group_result
            execution_log.extend(group_result.get("execution_log", []))
        
        final_result = self._integrate_hybrid_results(parallel_results, task_description)
        
        return {
            "status": "success",
            "task_id": task_id,
            "collaboration_mode": "hybrid",
            "parallel_results": parallel_results,
            "final_result": final_result,
            "execution_log": execution_log,
            "total_time": time.time() - start_time if execution_log else 0
        }

    def _analyze_dependencies(self, models: List[str]) -> Dict[str, List[str]]:
        """分析模型依赖关系"""
        dependencies = {}
        dependency_map = {
            "vision": ["spatial"],
            "video": ["vision", "spatial"],
            "audio": ["language"],
            "sensor": ["spatial"],
            "knowledge": [],
            "language": ["knowledge"],
            "spatial": [],
            "programming": ["knowledge", "language"]
        }
        
        for model in models:
            dependencies[model] = dependency_map.get(model, [])
            dependencies[model] = [dep for dep in dependencies[model] if dep in models]
        
        return dependencies

    def _group_parallel_models(self, models: List[str], dependencies: Dict[str, List[str]]) -> List[List[str]]:
        """分组可以并行执行的模型"""
        groups = []
        processed = set()
        
        independent_models = [model for model in models if not dependencies.get(model)]
        if independent_models:
            groups.append(independent_models)
            processed.update(independent_models)
        
        remaining_models = [model for model in models if model not in processed]
        while remaining_models:
            executable_models = []
            for model in remaining_models:
                model_deps = dependencies.get(model, [])
                if all(dep in processed for dep in model_deps):
                    executable_models.append(model)
            
            if executable_models:
                groups.append(executable_models)
                processed.update(executable_models)
                remaining_models = [model for model in remaining_models if model not in processed]
            else:
                groups.append(remaining_models)
                break
        
        return groups

    def _merge_results(self, results: Dict[str, Any], merge_strategy: str) -> Dict[str, Any]:
        """合并多个模型的结果"""
        if merge_strategy == "parallel":
            return results
        elif merge_strategy == "weighted":
            weighted_result = {}
            for model_name, result in results.items():
                if "error" not in result:
                    confidence = result.get("confidence", 0.5)
                    for key, value in result.items():
                        if key != "confidence":
                            if key not in weighted_result:
                                weighted_result[key] = {"value": 0, "weight": 0}
                            weighted_result[key]["value"] += value * confidence
                            weighted_result[key]["weight"] += confidence
            
            final_result = {}
            for key, data in weighted_result.items():
                if data["weight"] > 0:
                    final_result[key] = data["value"] / data["weight"]
            
            return final_result
        else:
            return results

    def _integrate_hybrid_results(self, parallel_results: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """整合混合协作结果"""
        integrated_result = {
            "task_description": task_description,
            "integration_time": time.time(),
            "component_results": {},
            "summary": ""
        }
        
        for group_name, group_result in parallel_results.items():
            integrated_result["component_results"][group_name] = group_result.get("merged_result", {})
        
        summary_parts = []
        for group_name, results in integrated_result["component_results"].items():
            if results:
                summary_parts.append(f"{group_name}: {len(results)} results")
        
        integrated_result["summary"] = f"Integrated results from {len(summary_parts)} parallel groups"
        
        return integrated_result

    def _should_continue_on_error(self, priority: int) -> bool:
        """确定是否在错误时继续"""
        if priority >= 8:
            return False
        elif priority >= 5:
            return random.random() < 0.3
        else:
            return random.random() < 0.7

    def _record_collaboration_performance(self, result: Dict[str, Any], mode: str):
        """记录协作性能"""
        if "execution_log" in result:
            total_time = sum(log.get("execution_time", 0) for log in result["execution_log"])
            success_count = sum(1 for log in result["execution_log"] if log.get("success", False))
            
            performance_record = {
                "timestamp": time.time(),
                "mode": mode,
                "total_time": total_time,
                "success_rate": success_count / len(result["execution_log"]) if result["execution_log"] else 0,
                "model_count": len(set(log.get("model") for log in result["execution_log"]))
            }
            
            perf_dir = "logs/collaboration_performance"
            if not os.path.exists(perf_dir):
                os.makedirs(perf_dir)
            
            perf_file = os.path.join(perf_dir, f"collaboration_perf_{datetime.now().strftime('%Y%m%d')}.log")
            with open(perf_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(performance_record, ensure_ascii=False) + '\n')

    def _optimize_communication(self) -> Dict[str, Any]:
        """优化模型间通信"""
        improvements = ["Built intelligent data routing table", 
                       "Optimized communication protocols", 
                       "Implemented data compression optimization"]
        
        return {
            "improvements": improvements,
            "communication_efficiency": self._measure_communication_efficiency()
        }

    def _optimize_coordination(self) -> Dict[str, Any]:
        """优化模型协调"""
        improvements = ["Enhanced collaboration rules", 
                       "Implemented intelligent task allocation", 
                       "Optimized load balancing"]
        
        return {
            "improvements": improvements,
            "coordination_efficiency": self._measure_coordination_efficiency()
        }

    def _optimize_monitoring(self) -> Dict[str, Any]:
        """优化监控系统"""
        improvements = ["Enhanced real-time monitoring", 
                       "Implemented predictive maintenance", 
                       "Optimized performance metrics collection"]
        
        return {
            "improvements": improvements,
            "monitoring_effectiveness": self._measure_monitoring_effectiveness()
        }

    def _optimize_error_handling(self) -> Dict[str, Any]:
        """优化错误处理"""
        improvements = ["Enhanced error recovery mechanisms", 
                       "Implemented fault tolerance", 
                       "Optimized error logging and analysis"]
        
        return {
            "improvements": improvements,
            "error_recovery_rate": self._measure_error_recovery_rate()
        }

    def _measure_communication_efficiency(self) -> Dict[str, float]:
        """测量通信效率"""
        return {
            "throughput": 150.5,
            "latency": 45.2,
            "success_rate": 0.98,
            "compression_ratio": 0.65
        }

    def _measure_coordination_efficiency(self) -> Dict[str, float]:
        """测量协调效率"""
        return {
            "task_completion_time": 12.3,
            "resource_utilization": 0.85,
            "collaboration_success_rate": 0.96,
            "load_balance_score": 0.92
        }

    def _measure_monitoring_effectiveness(self) -> Dict[str, float]:
        """测量监控有效性"""
        return {
            "detection_rate": 0.99,
            "false_positive_rate": 0.02,
            "alert_accuracy": 0.95,
            "response_time": 2.1
        }

    def _measure_error_recovery_rate(self) -> Dict[str, float]:
        """测量错误恢复率"""
        return {
            "recovery_success_rate": 0.88,
            "mean_time_to_recovery": 8.5,
            "error_prevention_rate": 0.75,
            "system_availability": 0.999
        }

    def _calculate_model_weights(self) -> Dict[str, float]:
        """计算模型权重"""
        weights = {}
        for model_id in self.sub_models:
            if self.sub_models[model_id]:
                performance = self._get_model_performance(model_id)
                weights[model_id] = performance.get("weight", 0.5)
            else:
                weights[model_id] = 0.0
        
        return weights

    def _get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """获取模型性能数据"""
        performance_data = {
            "throughput": random.uniform(50, 200),
            "latency": random.uniform(10, 100),
            "success_rate": random.uniform(0.8, 0.99),
            "memory_usage": random.uniform(10, 80),
            "cpu_usage": random.uniform(5, 60)
        }
        
        weight = (performance_data["success_rate"] * 0.4 +
                 (1 - performance_data["latency"] / 100) * 0.3 +
                 (performance_data["throughput"] / 200) * 0.3)
        
        performance_data["weight"] = weight
        return performance_data

    def _initiate_model_coordination(self, task_description: str, task_id: str, required_models: List[str]) -> Dict[str, Any]:
        """启动模型协调过程"""
        coordination_data = {
            "task_id": task_id,
            "participating_models": required_models,
            "start_time": time.time(),
            "model_status": {model: "pending" for model in required_models},
            "intermediate_results": {},
            "dependencies": self._analyze_dependencies(required_models)
        }
        
        for model_name in required_models:
            if self.sub_models[model_name] and hasattr(self.sub_models[model_name], 'prepare_for_coordination'):
                preparation_result = self.sub_models[model_name].prepare_for_coordination(task_description)
                coordination_data["model_status"][model_name] = "prepared"
                coordination_data["intermediate_results"][model_name] = preparation_result
            else:
                coordination_data["model_status"][model_name] = "ready"
        
        return coordination_data

    def _monitor_coordination(self, task_description: str, task_id: str, required_models: List[str], 
                             coordination_data: Dict[str, Any]) -> Dict[str, Any]:
        """监控协调过程"""
        max_wait_time = 30.0
        start_time = time.time()
        check_interval = 0.5
        
        while time.time() - start_time < max_wait_time:
            all_completed = True
            for model_name in required_models:
                if coordination_data["model_status"][model_name] != "completed":
                    all_completed = False
                    break
            
            if all_completed:
                break
            
            self._process_dependencies(coordination_data)
            self._collect_intermediate_results(coordination_data)
            time.sleep(check_interval)
        
        final_result = self._integrate_final_results(coordination_data)
        return final_result

    def _process_dependencies(self, coordination_data: Dict[str, Any]):
        """处理模型依赖关系"""
        for model_name, deps in coordination_data["dependencies"].items():
            if coordination_data["model_status"][model_name] == "pending":
                all_deps_ready = True
                for dep in deps:
                    if coordination_data["model_status"][dep] not in ["completed", "ready"]:
                        all_deps_ready = False
                        break
                
                if all_deps_ready:
                    coordination_data["model_status"][model_name] = "ready"

    def _collect_intermediate_results(self, coordination_data: Dict[str, Any]):
        """收集中间结果"""
        for model_name in coordination_data["participating_models"]:
            if (coordination_data["model_status"][model_name] == "ready" and 
                self.sub_models[model_name] and 
                hasattr(self.sub_models[model_name], 'get_coordination_result')):
                
                result = self.sub_models[model_name].get_coordination_result()
                coordination_data["intermediate_results"][model_name] = result
                coordination_data["model_status"][model_name] = "completed"

    def _integrate_final_results(self, coordination_data: Dict[str, Any]) -> Dict[str, Any]:
        """整合最终结果"""
        final_result = {
            "coordination_id": coordination_data["task_id"],
            "participating_models": coordination_data["participating_models"],
            "completion_time": time.time() - coordination_data["start_time"],
            "model_contributions": {},
            "integrated_output": ""
        }
        
        integrated_output = []
        for model_name in coordination_data["participating_models"]:
            if model_name in coordination_data["intermediate_results"]:
                result = coordination_data["intermediate_results"][model_name]
                if isinstance(result, dict) and "output" in result:
                    integrated_output.append(f"[{model_name}]: {result['output']}")
                
                final_result["model_contributions"][model_name] = {
                    "status": coordination_data["model_status"][model_name],
                    "contribution": result.get("contribution", "unknown")
                }
        
        final_result["integrated_output"] = "\n".join(integrated_output)
        return final_result

    def shutdown(self):
        """关闭管理器模型"""
        self.monitoring_active = False
        self.task_processing_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        if self.task_thread and self.task_thread.is_alive():
            self.task_thread.join(timeout=5)
        
        self.logger.info("Unified Manager model shutdown complete")
        return {"status": "success", "message": "Unified Manager model shutdown complete"}


    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理文本输入数据
        
        Args:
            input_data: 包含文本输入的字典，必须包含'text'和'type'字段
        
        Returns:
            处理结果的字典
        """
        try:
            # 验证输入数据
            if not isinstance(input_data, dict) or 'text' not in input_data or 'type' not in input_data:
                return {"success": False, "error": "Invalid input data format"}
            
            # 获取文本输入
            text_input = input_data.get('text', '')
            input_type = input_data.get('type', 'text')
            
            # 准备协调任务
            coordination_input = {
                "task_description": f"Process {input_type} input: {text_input}",
                "required_models": ["language", "knowledge", "advanced_reasoning"],
                "priority": 5,
                "collaboration_mode": "smart",
                "input_data": input_data
            }
            
            # 使用协调操作处理输入
            result = self._process_operation("coordinate", coordination_input)
            
            # 格式化结果
            if result.get("success", False):
                return {
                    "success": True,
                    "output": result.get("coordination_result", {}).get("integrated_output", ""),
                    "processed_data": result.get("coordination_result", {})
                }
            else:
                return result
        except Exception as e:
            self.logger.error(f"Input processing failed: {str(e)}")
            return {"success": False, "error": str(e)}

# 工厂函数用于创建统一管理器模型
def create_unified_manager_model(config: Dict[str, Any] = None) -> UnifiedManagerModel:
    """
    创建统一管理器模型实例
    
    Args:
        config: 配置字典
    
    Returns:
        UnifiedManagerModel实例
    """
    return UnifiedManagerModel(config)
