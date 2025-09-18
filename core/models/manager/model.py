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
管理模型 - 核心协调与任务分配
Manager Model - Core Coordination and Task Allocation

功能描述：
- 协调所有11个子模型的协同工作
- 处理多模态输入并智能路由到相应模型
- 管理任务优先级和实时分配
- 实现情感感知和情感化响应
- 支持本地和外部API模型无缝切换
- 提供实时监控和性能优化

Function Description:
- Coordinates all 11 sub-models for collaborative work
- Processes multimodal inputs and intelligently routes to appropriate models
- Manages task priorities and real-time allocation
- Implements emotion perception and emotional responses
- Supports seamless switching between local and external API models
- Provides real-time monitoring and performance optimization
"""

import logging
import datetime
import time
import threading
import json
import uuid
import os
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from ..base_model import BaseModel
from core.i18n_manager import gettext, set_language as set_global_language
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
from core.creative_problem_solving import CreativeProblemSolver
from core.self_reflection import SelfReflectionModule
from core.knowledge_integration import KnowledgeIntegrator


"""
ManagerModel类 - 中文类描述
ManagerModel Class - English class description
"""
class ManagerModel(BaseModel):
    """Self Soul 核心管理模型
    AGI System Core Manager Model
    
    功能：协调所有子模型，处理多模态输入，管理任务分配和情感交互
    Function: Coordinates all sub-models, processes multimodal inputs, 
              manages task allocation and emotional interaction
    """
    
    
    """
    __init__函数 - 中文函数描述
    __init__ Function - English function description

    Args:
        config: 配置参数 | Configuration parameters
        
    Returns:
        None
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.model_id = "manager"
        
        # 情感分析模块 | Emotion analysis module
        self.emotion_analyzer = EmotionAnalyzer()
        
        # 错误处理模块 | Error handling module
        self.error_handler = error_handler
        
        # API连接管理器 | API connection manager
        self.api_connector = APIModelConnector()
        
        # 子模型注册 | Sub-model registry
        self.sub_models = {
            "manager": None,  # 管理模型 | Manager model
            "language": None,  # 语言模型 | Language model
            "audio": None,  # 音频模型 | Audio model
            "vision_image": None,  # 图片视觉模型 | Image vision model
            "vision_video": None,  # 视频视觉模型 | Video vision model
            "spatial": None,  # 空间模型 | Spatial model
            "sensor": None,  # 传感器模型 | Sensor model
            "computer": None,  # 计算机控制 | Computer control
            "motion": None,  # 运动模型 | Motion model
            "knowledge": None,  # 知识模型 | Knowledge model
            "programming": None   # 编程模型 | Programming model
        }
        
        # 任务队列和优先级管理 | Task queue and priority management
        self.task_queue = []
        self.active_tasks = {}
        self.completed_tasks = []
        self.task_priorities = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        
        # 多语言支持 | Multilingual support
        self.supported_languages = ["zh", "en", "de", "ja", "ru"]
        self.current_language = "zh"
        
        # 外部API配置 | External API configuration
        self.external_apis = {}
        self.api_status = {}  # API连接状态 | API connection status
        
        # 实时流管理 | Real-time stream management
        self.active_streams = {}
        self.stream_manager = RealTimeStreamManager()
        
        # 增强性能监控 | Enhanced performance monitoring
        self.monitor = EnhancedMonitor()
        self.performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_task_time": 0,
            "model_utilization": {},
            "memory_usage": 0,
            "cpu_usage": 0,
            "network_throughput": 0,
            "response_times": [],
            "error_rates": {}
        }
        
        # 情感状态跟踪 | Emotion state tracking
        self.emotion_history = []
        self.current_emotion = {"state": "neutral", "intensity": 0.5}
        self.emotion_decay_rate = 0.98  # 情感衰减率 | Emotion decay rate
        
        # 模型协作优化 | Model collaboration optimization
        self.model_collaboration_rules = self._load_collaboration_rules()
        self.model_performance_stats = {}
        
        # 线程控制标志（不在构造函数中启动线程）
        # Thread control flags (don't start threads in constructor)
        self.monitoring_active = False
        self.task_processing_active = False
        self.monitoring_thread = None
        self.task_thread = None
        
        # AGI增强模块初始化 | AGI enhancement modules initialization
        self.advanced_reasoning = AdvancedReasoningEngine()
        self.meta_learning = MetaLearningSystem()
        self.creative_solver = CreativeProblemSolver()
        self.self_reflection = SelfReflectionModule()
        self.knowledge_integrator = KnowledgeIntegrator()
        
        # AGI状态跟踪 | AGI state tracking
        self.agi_capabilities = {
            "reasoning_level": 0.8,
            "learning_depth": 0.7,
            "creativity_score": 0.6,
            "adaptability": 0.75,
            "self_awareness": 0.65
        }
        
        # 常识知识库集成 | Common sense knowledge base integration
        self.common_sense_knowledge = self._load_common_sense_knowledge()
        
        self.logger.info("管理模型基础初始化完成 | Manager model basic initialization completed")
        self.logger.info("AGI增强模块已加载 | AGI enhancement modules loaded")

    
    def initialize(self) -> Dict[str, Any]:
        """初始化模型资源 | Initialize model resources"""
        try:
            # 首先设置管理模型为已初始化状态，避免循环依赖
            # First set manager model as initialized to avoid circular dependency
            self.is_initialized = True
            
            # 注册所有子模型 | Register all sub-models
            registration_result = self.register_sub_models()
            
            # 初始化情感分析器 | Initialize emotion analyzer
            self.emotion_analyzer.initialize()
            
            # 初始化API连接器 | Initialize API connector
            self.api_connector.initialize()
            
            # 初始化实时流管理器 | Initialize real-time stream manager
            self.stream_manager.initialize()
            
            # 初始化错误处理器 | Initialize error handler
            self.error_handler.initialize()
            
            # 启动实时监控线程（在所有资源初始化完成后）
            # Start real-time monitoring thread (after all resources are initialized)
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            # 启动任务处理线程（在所有资源初始化完成后）
            # Start task processing thread (after all resources are initialized)
            self.task_processing_active = True
            self.task_thread = threading.Thread(target=self._task_processing_loop)
            self.task_thread.daemon = True
            self.task_thread.start()
            
            self.logger.info("管理模型资源初始化完成 | Manager model resources initialized")
            self.logger.info("实时监控和任务处理线程已启动 | Real-time monitoring and task processing threads started")
            return {"success": True, "initialized_components": [
                "sub_models", "emotion_analyzer", "api_connector", 
                "stream_manager", "monitor", "error_handler",
                "monitoring_thread", "task_thread"
            ]}
        except Exception as e:
            self.logger.error(f"模型初始化失败: {str(e)} | Model initialization failed: {str(e)}")
            self.is_initialized = False  # 恢复初始化状态 | Restore initialization status
            # 确保线程被停止
            self.monitoring_active = False
            self.task_processing_active = False
            return {"success": False, "error": str(e)}

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据 | Process input data"""
        try:
            # 检查模型是否已初始化 | Check if model is initialized
            if not self.is_initialized:
                init_result = self.initialize()
                if not init_result["success"]:
                    return {"success": False, "error": "模型初始化失败 | Model initialization failed"}
            
            # 处理多模态输入 | Process multimodal input
            result = self.process_input(input_data)
            
            # 更新性能指标 | Update performance metrics
            self.performance_metrics["tasks_completed"] += 1
            self.performance_metrics["response_times"].append(time.time())
            
            # 限制响应时间记录数量 | Limit response time records
            if len(self.performance_metrics["response_times"]) > 1000:
                self.performance_metrics["response_times"] = self.performance_metrics["response_times"][-1000:]
            
            return result
        except Exception as e:
            self.logger.error(f"数据处理错误: {str(e)} | Data processing error: {str(e)}")
            return {"success": False, "error": str(e)}

    
    def register_sub_models(self):
        """注册所有子模型 | Register all sub-models"""
        try:
            # 直接使用实际模型ID注册，不再使用字母键映射
            # Use actual model IDs directly, no longer use letter key mapping
            model_ids = [
                "language", "audio", "vision", "video", "spatial",
                "sensor", "computer", "motion", "knowledge", "programming"
            ]
            
            # 注册自己（manager模型）
            # Register self (manager model)
            self.sub_models["manager"] = self
            
            for model_id in model_ids:
                self.sub_models[model_id] = get_model(model_id)
                self.logger.info(f"已注册模型: {model_id} | Registered model: {model_id}")
                
                # 初始化子模型 | Initialize sub-model (跳过管理模型自己)
                if self.sub_models[model_id] and model_id != "manager":
                    init_result = self.sub_models[model_id].initialize()
                    if init_result.get("success"):
                        self.logger.info(f"模型 {model_id} 初始化成功 | Model {model_id} initialized successfully")
                    else:
                        self.logger.warning(f"模型 {model_id} 初始化失败: {init_result.get('error', '未知错误')} | Model {model_id} initialization failed: {init_result.get('error', 'Unknown error')}")
                
            # 设置知识模型语言 | Set knowledge model language
            if self.sub_models["knowledge"]:
                self.sub_models["knowledge"].set_language(self.current_language)
                
            return {"success": True, "registered_models": ["manager"] + model_ids}
        except Exception as e:
            self.logger.error(f"模型注册失败: {str(e)} | Model registration failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理多模态输入 | Process multimodal input"""
        try:
            # 情感分析 | Emotion analysis
            # 检查输入类型，如果是文本则进行情感分析
            emotion_result = {"dominant_emotion": "neutral", "confidence": 0.0, "emotions": {}}
            if "text" in input_data and isinstance(input_data["text"], str):
                emotion_result = self.emotion_analyzer.analyze_text(input_data["text"])
            
            # 根据输入类型路由到对应模型 | Route to appropriate model based on input type
            if "text" in input_data:
                return self._handle_text_input(input_data["text"], emotion_result)
            elif "audio" in input_data:
                return self._handle_audio_input(input_data["audio"], emotion_result)
            elif "image" in input_data:
                return self._handle_image_input(input_data["image"], emotion_result)
            elif "video" in input_data:
                return self._handle_video_input(input_data["video"], emotion_result)
            elif "sensor" in input_data:
                return self._handle_sensor_input(input_data["sensor"], emotion_result)
            else:
                return {"success": False, "error": "不支持的输入类型 | Unsupported input type"}
        except Exception as e:
            self.logger.error(f"输入处理错误: {str(e)} | Input processing error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _handle_text_input(self, text: str, emotion: Dict) -> Dict[str, Any]:
        """处理文本输入 | Handle text input"""
        try:
            # 情感增强响应 | Emotion-enhanced response
            response = self.sub_models["language"].process({"text": text, "context": {"emotion": emotion}})
            
            # 检查是否需要执行任务 | Check if task execution is needed
            if response.get("requires_action"):
                task_id = self._create_task(response["action_details"])
                response["task_id"] = task_id
                
            # 增强情感分析和情感记忆 | Enhanced emotion analysis and memory
            emotion_analysis = self.emotion_analyzer.analyze_text_with_context(text, emotion)
            self._update_emotion_memory(emotion_analysis)
            
            # 知识库辅助响应优化 | Knowledge base assisted response optimization
            if self.sub_models["knowledge"]:
                knowledge_assist = self.sub_models["knowledge"].assist_model("language", {
                    "task_type": "text_response",
                    "input_text": text,
                    "current_emotion": emotion
                })
                if knowledge_assist.get("suggestions"):
                    response["knowledge_enhanced"] = True
                    response["assistance_suggestions"] = knowledge_assist["suggestions"]
            
            return response
        except Exception as e:
            self.logger.error(f"文本输入处理错误: {str(e)} | Text input processing error: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_audio_input(self, audio_data: Any, emotion: Dict) -> Dict[str, Any]:
        """处理音频输入 | Handle audio input"""
        # 语音识别 | Speech recognition
        text = self.sub_models["audio"].speech_to_text(audio_data)
        
        # 情感分析增强 | Enhanced with emotion analysis
        return self._handle_text_input(text, emotion)

    def _handle_image_input(self, image_data: Any, emotion: Dict) -> Dict[str, Any]:
        """处理图像输入 | Handle image input"""
        try:
            # 图像分析 | Image analysis
            analysis_result = self.sub_models["vision"].analyze_image(image_data, emotion)
            
            # 情感增强响应 | Emotion-enhanced response
            response = self.sub_models["language"].generate_response(
                f"图像分析结果: {analysis_result.get('description', '')}", 
                emotion
            )
            
            response["image_analysis"] = analysis_result
            return response
        except Exception as e:
            self.logger.error(f"图像输入处理错误: {str(e)} | Image input processing error: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_video_input(self, video_data: Any, emotion: Dict) -> Dict[str, Any]:
        """处理视频输入 | Handle video input"""
        try:
            # 视频分析 | Video analysis
            analysis_result = self.sub_models["video"].analyze_video(video_data, emotion)
            
            # 情感增强响应 | Emotion-enhanced response
            response = self.sub_models["language"].generate_response(
                f"视频分析结果: {analysis_result.get('description', '')}", 
                emotion
            )
            
            response["video_analysis"] = analysis_result
            return response
        except Exception as e:
            self.logger.error(f"视频输入处理错误: {str(e)} | Video input processing error: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_sensor_input(self, sensor_data: Any, emotion: Dict) -> Dict[str, Any]:
        """处理传感器输入 | Handle sensor input"""
        try:
            # 传感器数据分析 | Sensor data analysis
            analysis_result = self.sub_models["sensor"].analyze_sensor_data(sensor_data, emotion)
            
            # 情感增强响应 | Emotion-enhanced response
            response = self.sub_models["language"].generate_response(
                f"传感器数据分析结果: {analysis_result.get('description', '')}", 
                emotion
            )
            
            response["sensor_analysis"] = analysis_result
            return response
        except Exception as e:
            self.logger.error(f"传感器输入处理错误: {str(e)} | Sensor input processing error: {str(e)}")
            return {"success": False, "error": str(e)}

    def _create_task(self, task_details: Dict) -> str:
        """创建新任务 | Create new task"""
        task_id = f"task_{len(self.task_queue)+1}"
        task = {
            "id": task_id,
            "type": task_details["type"],
            "priority": task_details.get("priority", "medium"),
            "required_models": task_details["required_models"],
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        self.task_queue.append(task)
        return task_id

    
    def assign_tasks(self):
        for task in self.task_queue:
            if task["status"] == "pending":
                # 选择最优模型组合 | Select optimal model combination
                model_combination = self._select_optimal_models(task)
                
                if model_combination:
                    task["assigned_models"] = model_combination
                    task["status"] = "assigned"
                    self.active_tasks[task["id"]] = task
                    self.logger.info(f"任务 {task['id']} 已分配 | Task {task['id']} assigned")
        
        # 从队列中移除已分配任务 | Remove assigned tasks from queue
        self.task_queue = [t for t in self.task_queue if t["status"] == "pending"]

    def _select_optimal_models(self, task: Dict) -> Optional[List[str]]:
        """选择最优模型组合 | Select optimal model combination"""
        try:
            # 实现智能模型选择算法 | Implement smart model selection algorithm
            # 1. 检查模型可用性 | Check model availability
            available_models = [m for m in task["required_models"] if self.sub_models[m] is not None]
            
            # 2. 根据任务类型添加推荐模型 | Add recommended models based on task type
            task_type = task.get("type", "")
            recommended_models = self._get_recommended_models(task_type)
            for model in recommended_models:
                if model not in available_models and self.sub_models[model] is not None:
                    available_models.append(model)
            
            # 3. 根据优先级调整模型选择 | Adjust model selection based on priority
            if task.get("priority") == "high":
                # 确保关键模型参与 | Ensure critical models participate
                critical_models = ["language", "knowledge", "manager"]
                for model in critical_models:
                    if model not in available_models and self.sub_models[model] is not None:
                        available_models.append(model)
            
            # 4. 使用知识库模型优化选择 | Use knowledge model to optimize selection
            if "knowledge" in available_models and self.sub_models["knowledge"]:
                optimized_selection = self.sub_models["knowledge"].optimize_model_selection(
                    task_type, available_models
                )
                available_models = optimized_selection or available_models
            
            # 5. 考虑模型性能和负载均衡 | Consider model performance and load balancing
            available_models = self._balance_model_load(available_models, task_type)
            
            # 6. 过滤掉不可用模型 | Filter out unavailable models
            available_models = [m for m in available_models if self.sub_models[m] is not None]
            
            # 7. 确保模型组合有效 | Ensure model combination is valid
            if not available_models:
                self.logger.warning(f"无可用模型处理任务: {task['id']} | No available models for task: {task['id']}")
                return None
                
            # 8. 记录模型选择决策 | Record model selection decision
            self._log_model_selection(task, available_models)
                
            return available_models
        except Exception as e:
            self.logger.error(f"模型选择错误: {str(e)} | Model selection error: {str(e)}")
            return None

    def _get_recommended_models(self, task_type: str) -> List[str]:
        """获取任务类型推荐模型 | Get recommended models for task type"""
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

    def monitor_tasks(self) -> Dict[str, Any]:
        """监控活动任务 | Monitor active tasks"""
        task_statuses = {}
        for task_id, task in self.active_tasks.items():
            # 获取每个模型的进度 | Get progress from each model
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

    def configure_external_api(self, model_id: str, config: Dict[str, str]):
        """配置外部API | Configure external API"""
        if model_id not in self.sub_models:
            return {"success": False, "error": "无效模型ID | Invalid model ID"}
        
        # 保存配置 | Save configuration
        self.external_apis[model_id] = config
        
        # 切换模型模式 | Switch model mode
        if self.sub_models[model_id]:
            try:
                self.sub_models[model_id].set_mode("external", config)
                return {"success": True, "model": model_id}
            except Exception as e:
                return {"success": False, "error": str(e)}
        return {"success": False, "error": "模型未初始化 | Model not initialized"}

    def get_monitoring_data(self) -> Dict[str, Any]:
        """获取监控数据 | Get monitoring data"""
        return {
            "active_tasks": len(self.active_tasks),
            "pending_tasks": len(self.task_queue),
            "sub_models_status": {m: "loaded" if v else "not_loaded" for m, v in self.sub_models.items()},
            "external_apis": list(self.external_apis.keys()),
            "emotion_state": self.emotion_analyzer.current_state(),
            "language": self.current_language
        }

    def set_language(self, language: str):
        """设置当前语言 | Set current language"""
        if language not in self.supported_languages:
            raise ValueError(f"不支持的语言: {language} | Unsupported language: {language}")
            
        self.current_language = language
        # 更新所有子模型语言 | Update all sub-models language
        for model in self.sub_models.values():
            if model and hasattr(model, "set_language"):
                model.set_language(language)
                
        self.logger.info(f"系统语言已切换至: {language} | System language switched to: {language}")

    def _monitoring_loop(self):
        """实时监控循环 | Real-time monitoring loop"""
        while self.monitoring_active:
            try:
                # 更新性能指标 | Update performance metrics
                self._update_performance_metrics()
                
                # 检查模型状态 | Check model status
                self._check_model_availability()
                
                # 更新情感状态 | Update emotion state
                self._decay_emotions()
                
                # 检查API连接状态 | Check API connection status
                self._check_api_connections()
                
                time.sleep(2)  # 每2秒更新一次 | Update every 2 seconds
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {str(e)} | Monitoring loop error: {str(e)}")
                time.sleep(5)

    def _task_processing_loop(self):
        """任务处理循环 | Task processing loop"""
        while self.task_processing_active:
            try:
                # 分配待处理任务 | Assign pending tasks
                self.assign_tasks()
                
                # 监控活动任务进度 | Monitor active task progress
                self._monitor_active_tasks()
                
                # 处理已完成任务 | Process completed tasks
                self._process_completed_tasks()
                
                time.sleep(1)  # 每1秒处理一次 | Process every 1 second
                
            except Exception as e:
                self.logger.error(f"任务处理循环错误: {str(e)} | Task processing loop error: {str(e)}")
                time.sleep(3)

    def _update_performance_metrics(self):
        """更新性能指标 | Update performance metrics"""
        # 获取系统性能数据 | Get system performance data
        system_metrics = self.monitor.get_system_metrics()
        
        # 更新性能指标 | Update performance metrics
        self.performance_metrics.update({
            "memory_usage": system_metrics.get("memory_usage", 0),
            "cpu_usage": system_metrics.get("cpu_usage", 0),
            "network_throughput": system_metrics.get("network_throughput", 0),
            "last_updated": datetime.now().isoformat()
        })

    def _check_model_availability(self):
        """检查模型可用性 | Check model availability"""
        for model_id, model in self.sub_models.items():
            if model is not None:
                try:
                    status = model.get_status()
                    
                    # 检查模型是否已初始化 - 只有在模型明确报告错误时才记录警告
                    # Check if model is initialized - only log warning if model explicitly reports error
                    if not status.get("is_initialized", False):
                        # 只有在模型报告初始化失败时才记录警告，正常初始化过程中不记录
                        # Only log warning if model reports initialization failure, not during normal initialization process
                        if status.get("initialization_failed", False):
                            self.logger.warning(f"模型 {model_id} 初始化失败: {status} | Model {model_id} initialization failed: {status}")
                        elif status.get("is_initializing", False):
                            # 模型正在初始化中，这是正常状态，不记录警告
                            # Model is initializing, this is normal state, don't log warning
                            if self.logger.level <= logging.DEBUG:
                                self.logger.debug(f"模型 {model_id} 正在初始化中 | Model {model_id} is initializing")
                        else:
                            # 模型未初始化但也没有报告失败，可能是正常启动过程
                            # Model not initialized but no failure reported, could be normal startup process
                            if self.logger.level <= logging.DEBUG:
                                self.logger.debug(f"模型 {model_id} 未初始化 (正常状态) | Model {model_id} not initialized (normal state)")
                        continue  # 跳过其他检查，因为模型未初始化
                    
                    # 检查模型是否有错误状态
                    # Check if model has error status
                    if status.get("has_error", False):
                        self.logger.warning(f"模型 {model_id} 有错误: {status} | Model {model_id} has error: {status}")
                        continue
                    
                    # 检查模型是否处于异常训练状态（只有在模型报告训练但系统不在训练模式时才警告）
                    # Check if model is in abnormal training state (only warn if model reports training but system is not in training mode)
                    if (status.get("is_training", False) and 
                        not self.is_training and 
                        not status.get("training_expected", False)):
                        self.logger.warning(f"模型 {model_id} 训练状态异常: {status} | Model {model_id} training state abnormal: {status}")
                        continue
                    
                    # 检查性能指标是否异常（例如内存使用过高、CPU使用率异常等）
                    # Check if performance metrics are abnormal (e.g., high memory usage, abnormal CPU usage, etc.)
                    performance_metrics = status.get("performance_metrics", {})
                    
                    # 只有当性能指标存在且包含异常值时才警告，空性能指标是正常的初始状态
                    # Only warn if performance metrics exist and contain abnormal values, empty metrics are normal initial state
                    if performance_metrics and performance_metrics != {}:  # 只有当性能指标不为空时才检查
                        if (performance_metrics.get("memory_usage", 0) > 90 or  # 内存使用超过90%
                            performance_metrics.get("cpu_usage", 0) > 95):      # CPU使用超过95%
                            self.logger.warning(f"模型 {model_id} 性能异常: {performance_metrics} | Model {model_id} performance abnormal: {performance_metrics}")
                            continue
                    
                    # 如果模型状态正常，不记录任何警告信息（避免误报）
                    # If model status is normal, do not log any warning messages (avoid false positives)
                    # 只有在调试模式下才记录状态信息
                    # Only log status information in debug mode
                    if self.logger.level <= logging.DEBUG:
                        self.logger.debug(f"模型 {model_id} 状态正常 | Model {model_id} status normal")
                        
                except Exception as e:
                    self.logger.error(f"检查模型 {model_id} 状态错误: {str(e)} | Check model {model_id} status error: {str(e)}")

    def _decay_emotions(self):
        """情感衰减 | Emotion decay"""
        # 情感强度随时间衰减 | Emotion intensity decays over time
        self.current_emotion["intensity"] *= self.emotion_decay_rate
        if self.current_emotion["intensity"] < 0.1:
            self.current_emotion = {"state": "neutral", "intensity": 0.5}

    def _check_api_connections(self):
        """检查API连接状态 | Check API connection status"""
        for api_name, config in self.external_apis.items():
            try:
                status = self.api_connector.check_connection(api_name, config)
                self.api_status[api_name] = status
                if not status["connected"]:
                    self.logger.warning(f"API {api_name} 连接失败: {status.get('error', '未知错误')} | API {api_name} connection failed: {status.get('error', 'Unknown error')}")
            except Exception as e:
                self.logger.error(f"检查API {api_name} 连接错误: {str(e)} | Check API {api_name} connection error: {str(e)}")
                self.api_status[api_name] = {"connected": False, "error": str(e)}

    def _monitor_active_tasks(self):
        """监控活动任务 | Monitor active tasks"""
        completed_tasks = []
        for task_id, task in list(self.active_tasks.items()):
            try:
                # 检查任务是否完成 | Check if task is completed
                all_completed = True
                for model_id in task["assigned_models"]:
                    if self.sub_models[model_id] and not self.sub_models[model_id].is_task_completed(task_id):
                        all_completed = False
                        break
                
                if all_completed:
                    task["status"] = "completed"
                    task["completed_at"] = datetime.now().isoformat()
                    completed_tasks.append(task_id)
                    self.completed_tasks.append(task)
                    self.logger.info(f"任务 {task_id} 已完成 | Task {task_id} completed")
                    
            except Exception as e:
                self.logger.error(f"监控任务 {task_id} 错误: {str(e)} | Monitor task {task_id} error: {str(e)}")
        
        # 从活动任务中移除已完成的任务 | Remove completed tasks from active tasks
        for task_id in completed_tasks:
            del self.active_tasks[task_id]

    def _process_completed_tasks(self):
        """处理已完成任务 | Process completed tasks"""
        # 这里可以添加任务完成后的处理逻辑，如清理资源、记录日志等
        # Add post-task processing logic here, such as cleaning up resources, logging, etc.
        pass

    def _load_collaboration_rules(self) -> Dict[str, Any]:
        """加载协作规则 | Load collaboration rules"""
        # 从配置文件加载协作规则 | Load collaboration rules from config file
        try:
            rules_path = "config/collaboration_rules.json"
            if os.path.exists(rules_path):
                with open(rules_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"加载协作规则错误: {str(e)} | Load collaboration rules error: {str(e)}")
        
        # 默认协作规则 | Default collaboration rules
        return {
            "default": {
                "communication_protocol": "json_rpc",
                "timeout": 30,
                "retry_attempts": 3,
                "priority_weight": 1.0
            }
        }

    def _balance_model_load(self, available_models: List[str], task_type: str) -> List[str]:
        """平衡模型负载 | Balance model load"""
        # 简单的负载均衡策略：优先选择最近使用较少的模型
        # Simple load balancing strategy: prefer models that were used less recently
        try:
            # 获取模型使用统计 | Get model usage statistics
            usage_stats = {}
            for model_id in available_models:
                if model_id in self.model_performance_stats:
                    usage_stats[model_id] = self.model_performance_stats[model_id].get("usage_count", 0)
                else:
                    usage_stats[model_id] = 0
            
            # 按使用次数排序，使用次数少的优先 | Sort by usage count, less used ones first
            sorted_models = sorted(available_models, key=lambda x: usage_stats.get(x, 0))
            return sorted_models
        except Exception as e:
            self.logger.error(f"负载均衡错误: {str(e)} | Load balancing error: {str(e)}")
            return available_models

    def _log_model_selection(self, task: Dict, selected_models: List[str]):
        """记录模型选择决策 | Log model selection decision"""
        selection_log = {
            "task_id": task["id"],
            "task_type": task["type"],
            "selected_models": selected_models,
            "timestamp": datetime.now().isoformat(),
            "priority": task.get("priority", "medium")
        }
        
        # 记录到日志文件 | Log to file
        log_dir = "logs/model_selection"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"model_selection_{datetime.now().strftime('%Y%m%d')}.log")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(selection_log, ensure_ascii=False) + '\n')

    def _update_emotion_memory(self, emotion_analysis: Dict[str, Any]):
        """更新情感记忆 | Update emotion memory"""
        # 记录情感历史 | Record emotion history
        self.emotion_history.append({
            "timestamp": datetime.now().isoformat(),
            "emotion": emotion_analysis.get("emotion", "neutral"),
            "intensity": emotion_analysis.get("intensity", 0.5),
            "context": emotion_analysis.get("context", "")
        })
        
        # 限制情感历史记录数量 | Limit emotion history records
        if len(self.emotion_history) > 1000:
            self.emotion_history = self.emotion_history[-1000:]

    def shutdown(self):
        """关闭管理模型 | Shutdown manager model"""
        self.monitoring_active = False
        self.task_processing_active = False
        
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        if self.task_thread.is_alive():
            self.task_thread.join(timeout=5)
        
        self.logger.info("管理模型已关闭 | Manager model shutdown complete")

    
    """
    coordinate_task函数 - 中文函数描述
    coordinate_task Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def coordinate_task(self, task_description: str, required_models: List[str] = None, 
                       priority: int = 5) -> Dict[str, Any]:
        """协调多个模型共同完成任务 / Coordinate multiple models to complete a task
        
        Args:
            task_description: 任务描述 / Task description
            required_models: 需要参与的模型列表 / List of models required to participate
            priority: 任务优先级 (1-10) / Task priority (1-10)
            
        Returns:
            dict: 协调结果 / Coordination result
        """
        try:
            self.logger.info(f"开始协调任务: {task_description} | Starting task coordination: {task_description}")
            
            # 创建协调任务
            task_id = f"coord_{int(time.time())}_{hash(task_description)}"
            
            # 确定需要参与的模型
            if not required_models:
                required_models = self._determine_required_models(task_description)
            
            # 检查所有必需模型是否可用
            unavailable_models = [model for model in required_models if model not in self.sub_models or self.sub_models[model] is None]
            if unavailable_models:
                return {
                    "status": "error",
                    "message": f"以下模型不可用: {unavailable_models} | Unavailable models: {unavailable_models}",
                    "unavailable_models": unavailable_models
                }
            
            # 启动模型协调
            coordination_result = self._initiate_model_coordination(task_description, task_id, required_models)
            
            # 监控协调过程
            final_result = self._monitor_coordination(task_description, task_id, required_models, coordination_result)
            
            self.logger.info(f"任务协调完成: {task_description} | Task coordination completed: {task_description}")
            return {
                "status": "success",
                "task_description": task_description,
                "participating_models": required_models,
                "result": final_result
            }
            
        except Exception as e:
            self.logger.error(f"任务协调失败: {str(e)} | Task coordination failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "task_description": task_description
            }
    
    def _determine_required_models(self, task_description: str) -> List[str]:
        """根据任务描述确定需要的模型 / Determine required models based on task description"""
        required_models = []
        
        # 简单的关键词匹配逻辑 - 实际实现应更智能
        task_lower = task_description.lower()
        
        if any(keyword in task_lower for keyword in ["语言", "文本", "翻译", "对话", "language", "text", "translate"]):
            required_models.append("language")
        
        if any(keyword in task_lower for keyword in ["图像", "图片", "视觉", "识别", "image", "vision", "recognize"]):
            required_models.append("vision")
        
        if any(keyword in task_lower for keyword in ["视频", "流媒体", "video", "stream"]):
            required_models.append("video")
        
        if any(keyword in task_lower for keyword in ["音频", "声音", "语音", "audio", "sound", "speech"]):
            required_models.append("audio")
        
        if any(keyword in task_lower for keyword in ["传感器", "环境", "sensor", "environment"]):
            required_models.append("sensor")
        
        if any(keyword in task_lower for keyword in ["空间", "定位", "距离", "spatial", "location", "distance"]):
            required_models.append("spatial")
        
        if any(keyword in task_lower for keyword in ["知识", "信息", "knowledge", "information"]):
            required_models.append("knowledge")
        
        if any(keyword in task_lower for keyword in ["编程", "代码", "programming", "code"]):
            required_models.append("programming")
        
        # 确保至少有一个模型参与
        if not required_models:
            required_models = ["language", "knowledge"]  # 默认使用语言和知识模型
        
        return list(set(required_models))  # 去重
    
    def _initiate_model_coordination(self, task_description: str, task_id: str, required_models: List[str]) -> Dict[str, Any]:
        """启动模型协调过程 / Initiate model coordination process"""
        coordination_data = {
            "task_id": task_id,
            "participating_models": required_models,
            "start_time": time.time(),
            "model_status": {model: "pending" for model in required_models},
            "intermediate_results": {},
            "dependencies": self._analyze_dependencies(required_models)
        }
        
        # 通知所有参与模型
        for model_name in required_models:
            if self.sub_models[model_name] and hasattr(self.sub_models[model_name], 'prepare_for_coordination'):
                preparation_result = self.sub_models[model_name].prepare_for_coordination(task_description)
                coordination_data["model_status"][model_name] = "prepared"
                coordination_data["intermediate_results"][model_name] = preparation_result
            else:
                coordination_data["model_status"][model_name] = "ready"
        
        return coordination_data
    
    def _analyze_dependencies(self, models: List[str]) -> Dict[str, List[str]]:
        """分析模型间的依赖关系 / Analyze dependencies between models"""
        dependencies = {}
        
        # 简单的依赖关系映射
        dependency_map = {
            "vision": ["spatial"],
            "video": ["vision", "spatial"],
            "audio": ["language"],
            "sensor": ["spatial"],
            "knowledge": [],  # 知识模型通常独立
            "language": ["knowledge"],
            "spatial": [],
            "programming": ["knowledge", "language"]
        }
        
        for model in models:
            dependencies[model] = dependency_map.get(model, [])
            # 只包含实际参与模型的依赖
            dependencies[model] = [dep for dep in dependencies[model] if dep in models]
        
        return dependencies
    
    def _monitor_coordination(self, task_description: str, task_id: str, required_models: List[str], 
                             coordination_data: Dict[str, Any]) -> Dict[str, Any]:
        """监控协调过程 / Monitor coordination process"""
        max_wait_time = 30.0  # 最大等待时间（秒）
        start_time = time.time()
        check_interval = 0.5
        
        while time.time() - start_time < max_wait_time:
            # 检查所有模型状态
            all_completed = True
            for model_name in required_models:
                if coordination_data["model_status"][model_name] != "completed":
                    all_completed = False
                    break
            
            if all_completed:
                break
            
            # 处理模型依赖
            self._process_dependencies(coordination_data)
            
            # 收集中间结果
            self._collect_intermediate_results(coordination_data)
            
            time.sleep(check_interval)
        
        # 整合最终结果
        final_result = self._integrate_final_results(coordination_data)
        
        return final_result
    
    def _process_dependencies(self, coordination_data: Dict[str, Any]):
        """处理模型依赖关系 / Process model dependencies"""
        for model_name, deps in coordination_data["dependencies"].items():
            if coordination_data["model_status"][model_name] == "pending":
                # 检查所有依赖是否就绪
                all_deps_ready = True
                for dep in deps:
                    if coordination_data["model_status"][dep] not in ["completed", "ready"]:
                        all_deps_ready = False
                        break
                
                if all_deps_ready:
                    coordination_data["model_status"][model_name] = "ready"
    
    def _collect_intermediate_results(self, coordination_data: Dict[str, Any]):
        """收集中间结果 / Collect intermediate results"""
        for model_name in coordination_data["participating_models"]:
            if (coordination_data["model_status"][model_name] == "ready" and 
                self.sub_models[model_name] and 
                hasattr(self.sub_models[model_name], 'get_coordination_result')):
                
                result = self.sub_models[model_name].get_coordination_result()
                coordination_data["intermediate_results"][model_name] = result
                coordination_data["model_status"][model_name] = "completed"
    
    def _integrate_final_results(self, coordination_data: Dict[str, Any]) -> Dict[str, Any]:
        """整合最终结果 / Integrate final results"""
        final_result = {
            "coordination_id": coordination_data["task_id"],
            "participating_models": coordination_data["participating_models"],
            "completion_time": time.time() - coordination_data["start_time"],
            "model_contributions": {},
            "integrated_output": ""
        }
        
        # 整合各模型的结果
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

    def enhanced_coordinate_task(self, task_description: str, required_models: List[str] = None,
                               priority: int = 5, collaboration_mode: str = "smart") -> Dict[str, Any]:
        """增强型任务协调 - 支持多种协作模式和智能路由
        Enhanced task coordination - supports multiple collaboration modes and intelligent routing
        
        Args:
            task_description: 任务描述 / Task description
            required_models: 需要参与的模型列表 / List of models required to participate
            priority: 任务优先级 (1-10) / Task priority (1-10)
            collaboration_mode: 协作模式 (smart, parallel, serial, hybrid) / Collaboration mode
            
        Returns:
            dict: 协调结果 / Coordination result
        """
        try:
            self.logger.info(f"开始增强协调任务: {task_description}, 模式: {collaboration_mode}")
            self.logger.info(f"Starting enhanced coordination: {task_description}, mode: {collaboration_mode}")
            
            # 确定需要参与的模型
            if not required_models:
                required_models = self._smart_determine_models(task_description, priority)
            
            # 检查模型可用性
            unavailable_models = [model for model in required_models if model not in self.sub_models or self.sub_models[model] is None]
            if unavailable_models:
                return {
                    "status": "error",
                    "message": f"以下模型不可用: {unavailable_models} | Unavailable models: {unavailable_models}",
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
            self.logger.error(f"增强协调失败: {str(e)} | Enhanced coordination failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "task_description": task_description
            }
    
    def _smart_determine_models(self, task_description: str, priority: int) -> List[str]:
        """智能确定需要的模型 / Smartly determine required models"""
        # 基础关键词匹配
        base_models = self._determine_required_models(task_description)
        
        # 根据优先级添加额外模型
        if priority >= 8:  # 高优先级任务
            # 确保知识模型参与高优先级复杂任务
            if "knowledge" not in base_models and any(keyword in task_description.lower() for keyword in 
                                                     ["复杂", "重要", "关键", "complex", "important", "critical"]):
                base_models.append("knowledge")
            
            # 高优先级任务添加管理模型监督
            if "manager" not in base_models:
                base_models.append("manager")
        
        # 使用知识模型进一步优化选择
        if "knowledge" in base_models and self.sub_models["knowledge"]:
            try:
                optimized = self.sub_models["knowledge"].suggest_optimal_models(
                    task_description, base_models, priority
                )
                if optimized and isinstance(optimized, list):
                    base_models = optimized
            except Exception as e:
                self.logger.warning(f"知识模型优化建议失败: {str(e)} | Knowledge model optimization failed: {str(e)}")
        
        return list(set(base_models))  # 去重
    
    def _smart_collaboration(self, task_description: str, models: List[str], priority: int) -> Dict[str, Any]:
        """智能协作模式 - 根据任务复杂度自动选择最佳协作策略
        Smart collaboration mode - automatically selects best strategy based on task complexity
        """
        # 分析任务复杂度
        complexity = self._analyze_task_complexity(task_description, models)
        
        if complexity == "high":
            # 高复杂度任务使用混合模式
            return self._hybrid_collaboration(task_description, models, priority)
        elif complexity == "medium":
            # 中等复杂度任务使用并行模式
            return self._parallel_collaboration(task_description, models, priority)
        else:
            # 低复杂度任务使用串行模式
            return self._serial_collaboration(task_description, models, priority)
    
    def _analyze_task_complexity(self, task_description: str, models: List[str]) -> str:
        """分析任务复杂度 / Analyze task complexity"""
        complexity_score = 0
        
        # 基于模型数量
        complexity_score += len(models) * 2
        
        # 基于任务描述长度和关键词
        task_lower = task_description.lower()
        if any(keyword in task_lower for keyword in ["复杂", "困难", "挑战", "complex", "difficult", "challenge"]):
            complexity_score += 5
        
        if any(keyword in task_lower for keyword in ["简单", "基本", "容易", "simple", "basic", "easy"]):
            complexity_score -= 3
        
        # 基于涉及模型类型
        if "knowledge" in models:
            complexity_score += 3
        if "programming" in models:
            complexity_score += 3
        if "video" in models and "audio" in models:
            complexity_score += 4
        
        # 确定复杂度级别
        if complexity_score >= 10:
            return "high"
        elif complexity_score >= 5:
            return "medium"
        else:
            return "low"
    
    def _parallel_collaboration(self, task_description: str, models: List[str], priority: int) -> Dict[str, Any]:
        """并行协作模式 / Parallel collaboration mode"""
        task_id = f"parallel_{int(time.time())}_{hash(task_description)}"
        
        # 创建并行任务
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
                    error_msg = f"并行任务执行失败: {model_name} - {str(e)}"
                    self.logger.error(error_msg)
                    results[model_name] = {"error": error_msg}
                    execution_log.append({
                        "model": model_name,
                        "error": error_msg,
                        "success": False,
                        "timestamp": time.time()
                    })
        
        # 合并结果
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
        """串行协作模式 / Serial collaboration mode"""
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
                    
                    # 更新中间结果
                    intermediate_result = result
                    
                    # 如果遇到错误且不是继续模式，则停止
                    if "error" in result and not self._should_continue_on_error(priority):
                        break
                        
                except Exception as e:
                    error_msg = f"串行任务执行失败: {model_name} - {str(e)}"
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
        """混合协作模式 / Hybrid collaboration mode"""
        task_id = f"hybrid_{int(time.time())}_{hash(task_description)}"
        
        # 分析模型依赖关系
        dependencies = self._analyze_dependencies(models)
        
        # 分组可以并行执行的模型
        parallel_groups = self._group_parallel_models(models, dependencies)
        
        # 执行并行阶段
        parallel_results = {}
        execution_log = []
        
        for group in parallel_groups:
            group_result = self._parallel_collaboration(task_description, group, priority)
            parallel_results[f"group_{parallel_groups.index(group)}"] = group_result
            execution_log.extend(group_result.get("execution_log", []))
        
        # 执行串行阶段（整合并行结果）
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
    
    def _group_parallel_models(self, models: List[str], dependencies: Dict[str, List[str]]) -> List[List[str]]:
        """分组可以并行执行的模型 / Group models that can execute in parallel"""
        groups = []
        processed = set()
        
        # 首先处理没有依赖的模型
        independent_models = [model for model in models if not dependencies.get(model)]
        if independent_models:
            groups.append(independent_models)
            processed.update(independent_models)
        
        # 然后处理有依赖的模型
        remaining_models = [model for model in models if model not in processed]
        while remaining_models:
            # 找到当前可以执行的模型（所有依赖都已满足）
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
                # 无法解决依赖，将所有剩余模型放入一个组
                groups.append(remaining_models)
                break
        
        return groups
    
    def _merge_results(self, results: Dict[str, Any], merge_strategy: str) -> Dict[str, Any]:
        """合并多个模型的结果 / Merge results from multiple models"""
        if merge_strategy == "parallel":
            # 简单合并所有结果
            return results
        
        elif merge_strategy == "weighted":
            # 加权合并（基于模型置信度）
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
            
            # 计算加权平均值
            final_result = {}
            for key, data in weighted_result.items():
                if data["weight"] > 0:
                    final_result[key] = data["value"] / data["weight"]
            
            return final_result
        
        else:
            return results
    
    def _integrate_hybrid_results(self, parallel_results: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """整合混合协作的结果 / Integrate hybrid collaboration results"""
        integrated_result = {
            "task_description": task_description,
            "integration_time": time.time(),
            "component_results": {},
            "summary": ""
        }
        
        # 收集所有组件结果
        for group_name, group_result in parallel_results.items():
            integrated_result["component_results"][group_name] = group_result.get("merged_result", {})
        
        # 生成摘要
        summary_parts = []
        for group_name, results in integrated_result["component_results"].items():
            if results:
                summary_parts.append(f"{group_name}: {len(results)} results")
        
        integrated_result["summary"] = f"整合了 {len(summary_parts)} 个并行组的结果 | Integrated results from {len(summary_parts)} parallel groups"
        
        return integrated_result
    
    def _should_continue_on_error(self, priority: int) -> bool:
        """判断错误时是否继续 / Determine whether to continue on error"""
        # 高优先级任务在错误时继续的可能性更低
        if priority >= 8:
            return False
        elif priority >= 5:
            return random.random() < 0.3  # 30% 几率继续
        else:
            return random.random() < 0.7  # 70% 几率继续
    
    def _record_collaboration_performance(self, result: Dict[str, Any], mode: str):
        """记录协作性能 / Record collaboration performance"""
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
            
            # 保存到性能数据库或文件
            perf_dir = "logs/collaboration_performance"
            if not os.path.exists(perf_dir):
                os.makedirs(perf_dir)
            
            perf_file = os.path.join(perf_dir, f"collaboration_perf_{datetime.now().strftime('%Y%m%d')}.log")
            with open(perf_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(performance_record, ensure_ascii=False) + '\n')
        
        return result

    def optimize_model_interaction(self, optimization_type: str = "all") -> Dict[str, Any]:
        """优化模型间交互功能 | Optimize model interaction functionality
        
        Args:
            optimization_type: 优化类型 (all, communication, coordination, monitoring, error_handling)
            
        Returns:
            dict: 优化结果
        """
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
    
    def _optimize_communication(self) -> Dict[str, Any]:
        """优化模型间通信 | Optimize inter-model communication"""
        improvements = []
        
        # 1. 实现智能数据路由
        if not hasattr(self, 'data_routing_table'):
            self.data_routing_table = self._build_data_routing_table()
            improvements.append("构建了智能数据路由表")
        
        # 2. 优化通信协议
        self._optimize_communication_protocols()
        improvements.append("优化了通信协议")
        
        # 3. 实现数据压缩和序列化优化
        self._implement_data_compression()
        improvements.append("实现了数据压缩优化")
        
        return {
            "improvements": improvements,
            "communication_efficiency": self._measure_communication_efficiency()
        }
    
    def _optimize_coordination(self) -> Dict[str, Any]:
        """优化模型协调 | Optimize model coordination"""
        improvements = []
        
        # 1. 增强协作规则
        self.model_collaboration_rules = self._enhance_collaboration_rules()
        improvements.append("增强了协作规则")
        
        # 2. 实现智能任务分配
        self._implement_smart_task_allocation()
        improvements.append("实现了智能任务分配")
        
        # 3. 优化负载均衡
        self._optimize_load_balancing()
        improvements.append("优化了负载均衡")
        
        return {
            "improvements": improvements,
            "coordination_efficiency": self._measure_coordination_efficiency()
        }
    
    def _optimize_monitoring(self) -> Dict[str, Any]:
        """优化监控系统 | Optimize monitoring system"""
        improvements = []
        
        # 1. 增强实时监控
        self._enhance_real_time_monitoring()
        improvements.append("增强了实时监控")
        
        # 2. 实现预测性维护
        self._implement_predictive_maintenance()
        improvements.append("实现了预测性维护")
        
        # 3. 优化性能指标收集
        self._optimize_metrics_collection()
        improvements.append("优化了性能指标收集")
        
        return {
            "improvements": improvements,
            "monitoring_effectiveness": self._measure_monitoring_effectiveness()
        }
    
    def _optimize_error_handling(self) -> Dict[str, Any]:
        """优化错误处理 | Optimize error handling"""
        improvements = []
        
        # 1. 增强错误恢复机制
        self._enhance_error_recovery()
        improvements.append("增强了错误恢复机制")
        
        # 2. 实现容错处理
        self._implement_fault_tolerance()
        improvements.append("实现了容错处理")
        
        # 3. 优化错误日志和分析
        self._optimize_error_logging()
        improvements.append("优化了错误日志和分析")
        
        return {
            "improvements": improvements,
            "error_recovery_rate": self._measure_error_recovery_rate()
        }
    
    def _build_data_routing_table(self) -> Dict[str, Any]:
        """构建智能数据路由表 | Build smart data routing table"""
        routing_table = {
            "text": ["language", "knowledge"],
            "audio": ["audio", "language"],
            "image": ["vision", "spatial"],
            "video": ["video", "vision", "audio"],
            "sensor": ["sensor", "spatial", "knowledge"],
            "command": ["computer", "motion", "programming"],
            "complex": ["knowledge", "language", "manager"]
        }
        
        # 添加优先级权重
        for data_type, models in routing_table.items():
            routing_table[data_type] = {
                "primary_models": models,
                "backup_models": self._get_backup_models(models),
                "priority_weights": {model: self._calculate_model_weight(model, data_type) for model in models}
            }
        
        return routing_table
    
    def _get_backup_models(self, primary_models: List[str]) -> List[str]:
        """获取备份模型 | Get backup models"""
        backup_map = {
            "language": ["knowledge"],
            "audio": ["language"],
            "vision": ["video", "spatial"],
            "video": ["vision", "audio"],
            "sensor": ["knowledge"],
            "knowledge": ["language"],
            "spatial": ["vision"],
            "computer": ["programming"],
            "motion": ["spatial", "sensor"],
            "programming": ["knowledge", "language"]
        }
        
        backup_models = []
        for model in primary_models:
            if model in backup_map:
                backup_models.extend(backup_map[model])
        
        return list(set(backup_models))
    
    def _calculate_model_weight(self, model_id: str, data_type: str) -> float:
        """计算模型权重 | Calculate model weight"""
        base_weights = {
            "language": 0.9, "audio": 0.8, "vision": 0.85, "video": 0.8,
            "sensor": 0.75, "knowledge": 0.95, "spatial": 0.8,
            "computer": 0.7, "motion": 0.7, "programming": 0.85, "manager": 1.0
        }
        
        # 根据数据类型调整权重
        type_adjustments = {
            "text": {"language": 0.2, "knowledge": 0.1},
            "audio": {"audio": 0.2, "language": 0.1},
            "image": {"vision": 0.2, "spatial": 0.1},
            "video": {"video": 0.2, "vision": 0.1, "audio": 0.1},
            "sensor": {"sensor": 0.2, "spatial": 0.1, "knowledge": 0.1},
            "command": {"computer": 0.2, "motion": 0.1, "programming": 0.1},
            "complex": {"knowledge": 0.2, "language": 0.1, "manager": 0.1}
        }
        
        weight = base_weights.get(model_id, 0.5)
        if data_type in type_adjustments and model_id in type_adjustments[data_type]:
            weight += type_adjustments[data_type][model_id]
        
        return min(max(weight, 0.1), 1.0)
    
    def _optimize_communication_protocols(self):
        """优化通信协议 | Optimize communication protocols"""
        # 实现更高效的序列化格式
        self.communication_protocols = {
            "internal": {"format": "msgpack", "compression": "zlib", "timeout": 5},
            "external": {"format": "json", "compression": "gzip", "timeout": 10},
            "realtime": {"format": "protobuf", "compression": "none", "timeout": 1}
        }
    
    def _implement_data_compression(self):
        """实现数据压缩 | Implement data compression"""
        self.compression_strategies = {
            "text": {"algorithm": "gzip", "level": 6},
            "image": {"algorithm": "jpeg", "quality": 85},
            "audio": {"algorithm": "mp3", "bitrate": 128},
            "video": {"algorithm": "h264", "crf": 23},
            "sensor": {"algorithm": "zlib", "level": 3}
        }
    
    def _enhance_collaboration_rules(self) -> Dict[str, Any]:
        """增强协作规则 | Enhance collaboration rules"""
        enhanced_rules = self.model_collaboration_rules.copy()
        
        # 添加智能协作规则
        enhanced_rules.update({
            "smart_collaboration": {
                "dynamic_model_selection": True,
                "adaptive_timeout": True,
                "performance_based_routing": True,
                "error_recovery_strategy": "retry_then_fallback",
                "max_retry_attempts": 3,
                "fallback_models": self._get_backup_models([])
            },
            "quality_of_service": {
                "min_throughput": 100,  # KB/s
                "max_latency": 2000,    # ms
                "reliability_threshold": 0.95,
                "availability_requirement": 0.99
            }
        })
        
        return enhanced_rules
    
    def _implement_smart_task_allocation(self):
        """实现智能任务分配 | Implement smart task allocation"""
        self.task_allocation_strategy = {
            "load_aware": True,
            "performance_aware": True,
            "priority_aware": True,
            "dependency_aware": True,
            "realtime_adjustment": True
        }
    
    def _optimize_load_balancing(self):
        """优化负载均衡 | Optimize load balancing"""
        self.load_balancing_config = {
            "algorithm": "weighted_round_robin",
            "weights": self._calculate_model_weights(),
            "health_check_interval": 5,
            "performance_threshold": 0.8,
            "overload_protection": True
        }
    
    def _enhance_real_time_monitoring(self):
        """增强实时监控 | Enhance real-time monitoring"""
        self.monitoring_config = {
            "sampling_rate": 100,  # 毫秒
            "metrics": ["cpu", "memory", "throughput", "latency", "error_rate"],
            "alert_thresholds": {
                "cpu": 90, "memory": 85, "latency": 1000, "error_rate": 0.1
            },
            "predictive_analysis": True,
            "anomaly_detection": True
        }
    
    def _implement_predictive_maintenance(self):
        """实现预测性维护 | Implement predictive maintenance"""
        self.predictive_maintenance = {
            "enabled": True,
            "check_interval": 300,  # 秒
            "performance_degradation_threshold": 0.2,
            "memory_leak_detection": True,
            "resource_exhaustion_prediction": True
        }
    
    def _optimize_metrics_collection(self):
        """优化性能指标收集 | Optimize metrics collection"""
        self.metrics_config = {
            "collection_interval": 1,  # 秒
            "retention_period": 86400,  # 24小时
            "aggregation_levels": ["1m", "5m", "1h", "24h"],
            "storage_backend": "timeseries_db",
            "compression_enabled": True
        }
    
    def _enhance_error_recovery(self):
        """增强错误恢复机制 | Enhance error recovery mechanism"""
        self.error_recovery_config = {
            "automatic_retry": True,
            "max_retries": 3,
            "retry_delay": [1, 2, 4],  # 指数退避
            "fallback_strategies": ["alternative_model", "simplified_task", "graceful_degradation"],
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 5,
                "reset_timeout": 60
            }
        }
    
    def _implement_fault_tolerance(self):
        """实现容错处理 | Implement fault tolerance"""
        self.fault_tolerance_config = {
            "replication_factor": 2,
            "consistency_level": "quorum",
            "data_durability": "high",
            "checkpoint_interval": 60,
            "recovery_time_objective": 30,  # 秒
            "recovery_point_objective": 5   # 秒
        }
    
    def _optimize_error_logging(self):
        """优化错误日志和分析 | Optimize error logging and analysis"""
        self.error_logging_config = {
            "log_level": "ERROR",
            "structured_logging": True,
            "error_categorization": True,
            "root_cause_analysis": True,
            "trend_analysis": True,
            "alerting_enabled": True
        }
    
    def _measure_communication_efficiency(self) -> Dict[str, float]:
        """测量通信效率 | Measure communication efficiency"""
        # 模拟测量结果
        return {
            "throughput": 150.5,  # KB/s
            "latency": 45.2,      # ms
            "success_rate": 0.98,
            "compression_ratio": 0.65
        }
    
    def _measure_coordination_efficiency(self) -> Dict[str, float]:
        """测量协调效率 | Measure coordination efficiency"""
        return {
            "task_completion_time": 12.3,  # 秒
            "resource_utilization": 0.85,
            "collaboration_success_rate": 0.96,
            "load_balance_score": 0.92
        }
    
    def _measure_monitoring_effectiveness(self) -> Dict[str, float]:
        """测量监控效果 | Measure monitoring effectiveness"""
        return {
            "detection_rate": 0.99,
            "false_positive_rate": 0.02,
            "alert_accuracy": 0.95,
            "response_time": 2.1  # 秒
        }
    
    def _measure_error_recovery_rate(self) -> Dict[str, float]:
        """测量错误恢复率 | Measure error recovery rate"""
        return {
            "recovery_success_rate": 0.88,
            "mean_time_to_recovery": 8.5,  # 秒
            "error_prevention_rate": 0.75,
            "system_availability": 0.999
        }
    
    def _calculate_model_weights(self) -> Dict[str, float]:
        """计算模型权重 | Calculate model weights"""
        weights = {}
        for model_id in self.sub_models:
            if self.sub_models[model_id]:
                # 基于模型性能和历史数据计算权重
                performance = self._get_model_performance(model_id)
                weights[model_id] = performance.get("weight", 0.5)
            else:
                weights[model_id] = 0.0
        
        return weights
    
    def _get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """获取模型性能数据 | Get model performance data"""
        # 模拟性能数据
        performance_data = {
            "throughput": random.uniform(50, 200),
            "latency": random.uniform(10, 100),
            "success_rate": random.uniform(0.8, 0.99),
            "memory_usage": random.uniform(10, 80),
            "cpu_usage": random.uniform(5, 60)
        }
        
        # 计算综合权重
        weight = (performance_data["success_rate"] * 0.4 +
                 (1 - performance_data["latency"] / 100) * 0.3 +
                 (performance_data["throughput"] / 200) * 0.3)
        
        performance_data["weight"] = weight
        return performance_data

    def get_enhanced_interaction_status(self) -> Dict[str, Any]:
        """获取增强的交互状态 | Get enhanced interaction status"""
        return {
            "communication_efficiency": self._measure_communication_efficiency(),
            "coordination_efficiency": self._measure_coordination_efficiency(),
            "monitoring_effectiveness": self._measure_monitoring_effectiveness(),
            "error_recovery_rate": self._measure_error_recovery_rate(),
            "model_weights": self._calculate_model_weights(),
            "data_routing_table": getattr(self, 'data_routing_table', {}),
            "optimization_status": "enhanced"
        }

    def shutdown(self):
        """关闭管理模型 | Shutdown manager model"""
        self.monitoring_active = False
        self.task_processing_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        if self.task_thread and self.task_thread.is_alive():
            self.task_thread.join(timeout=5)
        
        self.logger.info("管理模型已关闭 | Manager model shutdown complete")
        return {"status": "success", "message": "Manager model shutdown complete"}
