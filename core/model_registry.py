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
AGI模型注册表：管理所有AI模型的加载、注册和生命周期，支持从零开始训练和AGI级别的协作
AGI Model Registry: Manages loading, registration and lifecycle of all AI models with from-scratch training and AGI-level collaboration
"""
import importlib
import os
import time
import threading
import random
import asyncio
import json
from typing import Dict, Any, Type, List, Optional, Tuple, Set, Callable
from core.error_handling import error_handler
from core.api_config_manager import APIConfigManager
from core.external_api_service import ExternalAPIService
from core.agi_core import AGICore  # AGI核心组件
from core.meta_learning_system import MetaLearningSystem  # 元学习系统
from core.adaptive_learning_engine import AdaptiveLearningEngine  # 自适应学习引擎
# 延迟导入以避免循环依赖
# from core.autonomous_learning_manager import AutonomousLearningManager  # 自主学习管理器
# AGI协调器导入
# 延迟导入以避免循环依赖
# AGICoordinator将在需要时动态导入
from core.unified_cognitive_architecture import UnifiedCognitiveArchitecture  # 统一认知架构
from core.self_learning import AGISelfLearningSystem  # 自我学习系统
from core.context_memory import ContextMemoryManager  # 上下文记忆管理器
from core.intrinsic_motivation_system import IntrinsicMotivationSystem  # 内在动机系统
from core.creative_problem_solver import EnhancedCreativeProblemSolver  # 创造性问题解决器
from core.value_alignment import ValueAlignment  # 值对齐系统
from concurrent.futures import ThreadPoolExecutor


"""
ExternalModelProxy类 - 中文类描述
ExternalModelProxy Class - English class description
"""
class ExternalModelProxy:
    """外部模型代理类，用于连接和管理外部API模型
    External Model Proxy Class, used to connect and manage external API models
    """
    
    def __init__(self, model_id, api_config):
        self.model_id = model_id
        self.api_config = api_config
        self.status = "disconnected"
        self.last_connection_time = None
        self.performance_metrics = {}
        # APIConfigManager用于管理全局API配置，不应在此实例化
        # 直接使用传入的api_config进行连接
        
    def connect(self):
        """连接到外部API模型 - 修复连接测试逻辑
        Connect to external API model - fixed connection test logic
        """
        try:
            # 检查必要的配置项
            required_keys = ['url', 'api_key']
            for key in required_keys:
                if key not in self.api_config or not self.api_config[key]:
                    self.status = "error"
                    error_handler.log_warning(f"缺少必要的API配置项或配置值为空: {key}", "ExternalModelProxy")
                    return False
            
            # 检查URL格式是否有效
            url = self.api_config['url']
            if not (url.startswith('http://') or url.startswith('https://')):
                self.status = "error"
                error_handler.log_warning(f"无效的API URL格式: {url}", "ExternalModelProxy")
                return False
            
            # 记录连接尝试
            error_handler.log_info(f"正在连接到外部模型: {self.model_id} ({url})", "ExternalModelProxy")
            
            # 真实连接测试 - 使用requests库进行实际API调用
            import requests
            headers = {
                'Authorization': f'Bearer {self.api_config["api_key"]}',
                'Content-Type': 'application/json'
            }
            
            # 根据模型类型和提供商准备更智能的测试数据
            model_type = self.api_config.get('model_type', 'general')
            provider = self.api_config.get('provider', 'unknown')
            test_data = {}
            
            # 根据提供商和模型类型生成合适的测试数据
            if provider in ['openai', 'anthropic', 'cohere']:
                # 通用语言模型测试数据
                test_data = {
                    'prompt': 'Hello, this is a connection test. Please respond with "OK" if you receive this.',
                    'max_tokens': 5,
                    'temperature': 0.1
                }
            elif provider in ['huggingface', 'local']:
                # HuggingFace或本地模型测试数据
                test_data = {
                    'inputs': 'Hello, connection test.',
                    'parameters': {'max_length': 10}
                }
            elif model_type == 'vision':
                # 视觉模型测试数据 - 使用更安全的占位符
                test_data = {
                    'task': 'describe',
                    'image': 'base64_placeholder_for_test'
                }
            elif model_type == 'audio':
                # 音频模型测试数据
                test_data = {
                    'audio': 'base64_placeholder_for_test',
                    'task': 'transcribe'
                }
            else:
                # 通用测试数据
                test_data = {'test': True, 'message': 'Connection test'}
            
            # 发送真实连接测试请求，增加超时和重试机制
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    response = requests.post(
                        url,
                        headers=headers,
                        json=test_data,
                        timeout=15  # 减少超时时间
                    )
                    
                    # 检查连接状态 - 更宽松的成功条件
                    if response.status_code in [200, 201, 202]:
                        self.status = "connected"
                        self.last_connection_time = time.time()
                        
                        # 记录连接成功的详细信息
                        connection_info = {
                            'model_id': self.model_id,
                            'url': url,
                            'model_name': self.api_config.get('model_name', self.model_id),
                            'source': self.api_config.get('source', 'external'),
                            'connection_time': self.last_connection_time,
                            'response_status': response.status_code,
                            'provider': provider
                        }
                        
                        error_handler.log_info(f"成功连接到外部模型: {self.model_id} (提供商: {provider})", "ExternalModelProxy")
                        return True
                    else:
                        # 如果是最后一次尝试，记录详细错误
                        if attempt == max_retries:
                            self.status = "error"
                            error_details = {
                                'model_id': self.model_id,
                                'status_code': response.status_code,
                                'response_text': response.text[:200] if response.text else 'No response body',
                                'url': url,
                                'attempt': attempt + 1
                            }
                            error_handler.log_warning(f"外部模型连接失败: {error_details}", "ExternalModelProxy")
                            return False
                        else:
                            # 重试前等待
                            time.sleep(1)
                            
                except requests.exceptions.Timeout:
                    if attempt == max_retries:
                        self.status = "error"
                        error_handler.log_warning(f"外部模型连接超时: {self.model_id} (尝试 {attempt + 1})", "ExternalModelProxy")
                        return False
                    time.sleep(1)
                except requests.exceptions.ConnectionError:
                    if attempt == max_retries:
                        self.status = "error"
                        error_handler.log_warning(f"外部模型连接错误: {self.model_id} (尝试 {attempt + 1})", "ExternalModelProxy")
                        return False
                    time.sleep(1)
                except Exception as e:
                    if attempt == max_retries:
                        self.status = "error"
                        error_handler.handle_error(e, "ExternalModelProxy", f"连接外部模型 {self.model_id} 失败 (尝试 {attempt + 1})")
                        return False
                    time.sleep(1)
            
            # 如果所有重试都失败
            self.status = "error"
            return False
            
        except Exception as e:
            self.status = "error"
            error_handler.handle_error(e, "ExternalModelProxy", f"连接外部模型 {self.model_id} 失败")
            return False
            
    def process(self, input_data):
        """处理输入数据
        Process input data
        """
        try:
            if self.status != "connected":
                error_handler.log_warning(f"尝试使用未连接的外部模型: {self.model_id}", "ExternalModelProxy")
                return {"error": "Model not connected", "status": "error"}
            
            # 根据不同模型类型处理输入数据
            model_type = self.api_config.get('model_type', 'general')
            headers = {
                'Authorization': f'Bearer {self.api_config.get("api_key", "")}',
                'Content-Type': 'application/json'
            }
            
            # 准备请求数据
            request_data = {
                'input': input_data,
                'model': self.api_config.get('model_name', self.model_id)
            }
            
            # 添加特定模型类型的参数
            if model_type == 'language':
                request_data['temperature'] = self.api_config.get('temperature', 0.7)
                request_data['max_tokens'] = self.api_config.get('max_tokens', 1000)
            elif model_type == 'vision':
                # 图像处理需要特殊处理
                pass
            
            # 调用外部API
            import requests
            response = requests.post(
                self.api_config['url'],
                headers=headers,
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                # 更新性能指标
                self.performance_metrics['last_response_time'] = time.time()
                self.performance_metrics['success_rate'] = (self.performance_metrics.get('success_rate', 0) * 9 + 1) / 10  # 平滑计算
                return {"result": result, "status": "success"}
            else:
                error_handler.log_warning(f"外部模型调用失败: {self.model_id}, 状态码: {response.status_code}", "ExternalModelProxy")
                self.performance_metrics['error_count'] = self.performance_metrics.get('error_count', 0) + 1
                return {"error": f"API call failed with status {response.status_code}", "status": "error"}
        except Exception as e:
            error_handler.handle_error(e, "ExternalModelProxy", f"处理外部模型请求失败: {self.model_id}")
            self.performance_metrics['error_count'] = self.performance_metrics.get('error_count', 0) + 1
            return {"error": str(e), "status": "error"}
        
    def get_status(self):
        """获取模型状态
        Get model status
        """
        return {
            "status": self.status,
            "last_connection_time": self.last_connection_time,
            "api_endpoint": self.api_config.get('url', 'unknown')
        }
        
    def on_access(self):
        """访问回调方法
        Access callback method
        """
        # 记录访问时间或其他统计信息
        # Record access time or other statistics
        pass


class ModelRegistry:
    """模型注册表类，负责管理所有AI模型
    Model Registry Class, responsible for managing all AI models
    """
    
    def __init__(self):
        self.models = {}
        self.model_configs = {}
        self._models_lock = threading.RLock()  # 添加线程锁防止并发问题
        self.model_types = {
            'manager': 'core.models.manager.unified_manager_model.UnifiedManagerModel',           # A管理模型
            'language': 'core.models.language.unified_language_model.UnifiedLanguageModel',        # B大语言模型
            'audio': 'core.models.audio.unified_audio_model.UnifiedAudioModel',       # C音频处理模型
            'vision_image': 'core.models.vision.unified_vision_model.UnifiedVisionModel',  # D图片视觉处理模型
            'vision_video': 'core.models.video.unified_video_model.UnifiedVideoModel',  # E视频流视觉处理模型
            'spatial': 'core.models.spatial.unified_spatial_model.UnifiedSpatialModel',           # F双目空间定位感知模型
            'sensor': 'core.models.sensor.unified_sensor_model.UnifiedSensorModel',    # G传感器感知模型
            'computer': 'core.models.computer.unified_computer_model.UnifiedComputerModel',        # H计算机控制模型
            'motion': 'core.models.motion.unified_motion_model.UnifiedMotionModel',              # I运动和执行器控制模型
            'knowledge': 'core.models.knowledge.unified_knowledge_model.UnifiedKnowledgeModel',     # J知识库专家模型
            'programming': 'core.models.programming.unified_programming_model.UnifiedProgrammingModel', # K编程模型
            'planning': 'core.models.planning.unified_planning_model.UnifiedPlanningModel',        # 规划模型
            'emotion': 'core.models.emotion.unified_emotion_model.UnifiedEmotionModel',        # 情感模型
            'finance': 'core.models.finance.unified_finance_model.UnifiedFinanceModel',        # 金融模型
            'medical': 'core.models.medical.unified_medical_model.UnifiedMedicalModel',        # 医疗模型
            'prediction': 'core.models.prediction.unified_prediction_model.UnifiedPredictionModel',        # 预测模型
            'collaboration': 'core.models.collaboration.unified_collaboration_model.UnifiedCollaborationModel',        # 协作模型
            'optimization': 'core.models.optimization.unified_optimization_model.UnifiedOptimizationModel',        # 优化模型
            'autonomous': 'core.models.autonomous.unified_autonomous_model.UnifiedAutonomousModel',        # 自主模型
            'value_alignment': 'core.value_alignment.ValueAlignment'        # 值对齐模型
        }
        # 简化模型依赖关系管理，避免循环依赖
        self.model_dependencies = {
            'manager': [],  # 完全移除依赖，让manager成为独立模型
            'language': [],  # 移除依赖，改为按需加载
            'audio': [],  # 移除依赖，改为按需加载
            'vision_image': [],  # 移除依赖，改为按需加载
            'vision_video': [],  # 移除依赖，改为按需加载
            'spatial': [],  # 移除依赖，改为按需加载
            'sensor': [],
            'computer': [],  # 移除依赖，改为按需加载
            'motion': [],  # 移除依赖，改为按需加载
            'knowledge': [],  # 基础模型无依赖
            'programming': [],  # 移除依赖，改为按需加载
            'emotion': [],  # 移除依赖，改为按需加载
            'finance': [],  # 移除依赖，改为按需加载
            'medical': [],  # 移除依赖，改为按需加载
            'planning': [],  # 移除依赖，改为按需加载
            'prediction': [],  # 移除依赖，改为按需加载
            'collaboration': [],  # 移除依赖，改为按需加载
            'optimization': [],  # 移除依赖，改为按需加载
            'autonomous': [],  # 移除依赖，改为按需加载
            'value_alignment': []  # 移除依赖，改为按需加载
        }
        # 模型性能评估和训练状态 | Model performance evaluation and training status
        self.performance_metrics = {}
        self.training_status = {}  # 新增训练状态跟踪
        self.joint_training_coordinator = None  # 联合训练协调器
        self.training_history = {}  # 训练历史记录
        
        # AGI级别组件延迟初始化
        self._agi_core = None
        self._agi_coordinator = None
        self._cognitive_architecture = None
        self._self_learning_system = None
        self._context_memory = None
        self._intrinsic_motivation = None
        self._creative_solver = None
        self._value_alignment = None
        
        # 新增：认知融合引擎和跨模型知识迁移
        self.cognitive_fusion_engine = None  # 认知融合引擎
        self.knowledge_transfer_engine = None  # 知识迁移引擎
        self.context_manager = None  # 上下文管理器
        self.active_workflows = {}  # 活跃的工作流
        self.workflow_lock = threading.RLock()  # 工作流锁
        self._executor = None  # 延迟初始化线程池
        self.conflict_resolution_strategies = {
            'majority_vote': self._resolve_conflict_majority,
            'expert_model': self._resolve_conflict_expert,
            'hierarchical': self._resolve_conflict_hierarchical,
            'agi_consensus': self._resolve_conflict_agi_consensus  # AGI共识策略
        }
        self.default_conflict_strategy = 'majority_vote'  # 使用更简单的默认策略
        
        # 简化AGI状态跟踪
        self.agi_state = {
            'consciousness_level': 0.1,  # 意识水平（0-1）
            'learning_capability': 0.8,  # 学习能力
            'problem_solving_ability': 0.7,  # 问题解决能力
            'creativity_level': 0.6,  # 创造力水平
            'ethical_alignment': 0.9,  # 伦理对齐度
            'last_self_reflection': time.time(),  # 上次自我反思时间
            'total_interactions': 0,  # 总交互次数
            'knowledge_accumulation': 0.0  # 知识积累度
        }
        
        # 从零开始训练的支持
        self.from_scratch_training_enabled = True  # 启用从零开始训练
        self.training_progress = {}  # 训练进度跟踪
        self.knowledge_base_integration = {}  # 知识库集成状态
        
        # 内存优化跟踪
        self._loaded_dependencies = set()  # 跟踪已加载的依赖
        self._loading_stack = []  # 跟踪当前加载堆栈，防止循环依赖
        self._max_recursion_depth = 5  # 限制最大递归深度，防止内存溢出
        self._max_dependencies = 20  # 限制最大依赖数量
        
        # 新增内存管理参数
        self._last_memory_cleanup = time.time()
        self._memory_cleanup_interval = 3600  # 1小时清理一次
        self._max_workflows = 100  # 最大工作流数量
        self._max_performance_records = 1000  # 最大性能记录数量
        self._max_training_history_per_model = 50  # 每个模型最大训练历史记录数
        self._max_loaded_models = 20  # 最大加载模型数量
        self._max_collaboration_records = 500  # 最大协作记录数量
        
        # 内存清理监控
        self._memory_cleanup_count = 0
        self._last_memory_usage_report = time.time()
        self._memory_usage_interval = 1800  # 30分钟报告一次内存使用情况
        
        # 启动内存清理监控线程
        self._cleanup_thread = None
        self._stop_cleanup_thread = False
        self._start_cleanup_monitor()
        
    def _start_cleanup_monitor(self):
        """启动内存清理监控线程
        Start memory cleanup monitoring thread
        """
        try:
            if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
                self._stop_cleanup_thread = False
                self._cleanup_thread = threading.Thread(target=self._cleanup_monitor_loop, daemon=True)
                self._cleanup_thread.start()
                error_handler.log_info("内存清理监控线程已启动", "ModelRegistry")
        except Exception as e:
            error_handler.handle_error(e, "ModelRegistry", "启动内存清理监控线程失败")
            
    def _cleanup_monitor_loop(self):
        """内存清理监控循环
        Memory cleanup monitoring loop
        """
        while not self._stop_cleanup_thread:
            try:
                current_time = time.time()
                
                # 检查是否需要清理内存
                if current_time - self._last_memory_cleanup > self._memory_cleanup_interval:
                    self._cleanup_memory()
                    self._last_memory_cleanup = current_time
                    
                # 检查是否需要报告内存使用情况
                if current_time - self._last_memory_usage_report > self._memory_usage_interval:
                    self._report_memory_usage()
                    self._last_memory_usage_report = current_time
                    
                # 每30秒检查一次
                time.sleep(30)
                
            except Exception as e:
                error_handler.handle_error(e, "ModelRegistry", "内存清理监控循环出错")
                time.sleep(60)  # 出错时等待更长时间
                
    def _cleanup_memory(self):
        """执行实际的内存清理操作
        Perform actual memory cleanup operations
        """
        try:
            start_time = time.time()
            cleanup_stats = {
                'workflows_cleaned': 0,
                'performance_records_cleaned': 0,
                'training_history_cleaned': 0,
                'collaboration_records_cleaned': 0
            }
            
            error_handler.log_info("开始内存清理操作", "ModelRegistry")
            
            # 1. 清理过期的工作流
            with self.workflow_lock:
                workflows_to_clean = []
                current_time = time.time()
                for workflow_id, workflow_info in self.active_workflows.items():
                    # 清理已完成、失败或取消且超过1小时的工作流
                    if (workflow_info.get('status') in ['completed', 'failed', 'cancelled'] and 
                        current_time - workflow_info.get('end_time', 0) > 3600):
                        workflows_to_clean.append(workflow_id)
                
                for workflow_id in workflows_to_clean:
                    del self.active_workflows[workflow_id]
                    cleanup_stats['workflows_cleaned'] += 1
                
                # 限制活跃工作流数量
                if len(self.active_workflows) > self._max_workflows:
                    # 按结束时间排序，删除最旧的
                    sorted_workflows = sorted(
                        self.active_workflows.items(),
                        key=lambda x: x[1].get('end_time', 0)
                    )
                    excess_count = len(self.active_workflows) - self._max_workflows
                    for i in range(excess_count):
                        workflow_id, _ = sorted_workflows[i]
                        del self.active_workflows[workflow_id]
                        cleanup_stats['workflows_cleaned'] += 1
            
            # 2. 清理性能记录
            if len(self.performance_metrics) > self._max_performance_records:
                # 按最后活动时间排序，删除最不活跃的
                sorted_metrics = sorted(
                    self.performance_metrics.items(),
                    key=lambda x: x[1].get('last_collaboration_time', 0)
                )
                excess_count = len(self.performance_metrics) - self._max_performance_records
                for i in range(excess_count):
                    model_id, _ = sorted_metrics[i]
                    del self.performance_metrics[model_id]
                    cleanup_stats['performance_records_cleaned'] += 1
            
            # 3. 清理训练历史记录
            for model_id in list(self.training_history.keys()):
                history = self.training_history[model_id]
                if len(history) > self._max_training_history_per_model:
                    # 保留最近的记录
                    self.training_history[model_id] = history[-self._max_training_history_per_model:]
                    cleanup_stats['training_history_cleaned'] += len(history) - len(self.training_history[model_id])
            
            # 4. 清理协作记录（如果有的话）
            # 这里假设协作记录存储在某个地方，实际实现中可能需要调整
            if hasattr(self, '_collaboration_records'):
                if len(self._collaboration_records) > self._max_collaboration_records:
                    # 按时间戳排序，删除最旧的
                    sorted_records = sorted(
                        self._collaboration_records,
                        key=lambda x: x.get('timestamp', 0)
                    )
                    excess_count = len(self._collaboration_records) - self._max_collaboration_records
                    for i in range(excess_count):
                        self._collaboration_records.remove(sorted_records[i])
                        cleanup_stats['collaboration_records_cleaned'] += 1
            
            # 5. 清理加载的模型（如果有太多）
            if len(self.models) > self._max_loaded_models:
                # 按最后访问时间排序，卸载最不活跃的模型
                model_access_times = {}
                for model_id, model in self.models.items():
                    if hasattr(model, 'last_access_time'):
                        model_access_times[model_id] = model.last_access_time
                    else:
                        model_access_times[model_id] = 0
                
                sorted_models = sorted(
                    model_access_times.items(),
                    key=lambda x: x[1]
                )
                
                excess_count = len(self.models) - self._max_loaded_models
                for i in range(excess_count):
                    model_id, _ = sorted_models[i]
                    self.unload_model(model_id)
            
            # 6. 强制垃圾回收
            import gc
            gc.collect()
            
            # 记录清理统计
            end_time = time.time()
            cleanup_stats['duration'] = end_time - start_time
            cleanup_stats['timestamp'] = time.time()
            self._memory_cleanup_count += 1
            
            error_handler.log_info(
                f"内存清理完成: 工作流清理{cleanup_stats['workflows_cleaned']}个, "
                f"性能记录清理{cleanup_stats['performance_records_cleaned']}个, "
                f"训练历史清理{cleanup_stats['training_history_cleaned']}条, "
                f"协作记录清理{cleanup_stats['collaboration_records_cleaned']}条, "
                f"耗时{cleanup_stats['duration']:.2f}秒",
                "ModelRegistry"
            )
            
            return cleanup_stats
            
        except Exception as e:
            error_handler.handle_error(e, "ModelRegistry", "内存清理操作失败")
            return {'error': str(e)}
            
    def _report_memory_usage(self):
        """报告内存使用情况
        Report memory usage
        """
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            memory_stats = {
                'rss_mb': memory_info.rss / 1024 / 1024,  # 常驻内存
                'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存
                'memory_percent': process.memory_percent(),
                'models_count': len(self.models),
                'workflows_count': len(self.active_workflows),
                'performance_records_count': len(self.performance_metrics),
                'training_history_count': sum(len(h) for h in self.training_history.values()),
                'cleanup_count': self._memory_cleanup_count,
                'timestamp': time.time()
            }
            
            error_handler.log_info(
                f"内存使用报告: RSS={memory_stats['rss_mb']:.1f}MB, "
                f"VMS={memory_stats['vms_mb']:.1f}MB, "
                f"使用率={memory_stats['memory_percent']:.1f}%, "
                f"模型数={memory_stats['models_count']}, "
                f"工作流数={memory_stats['workflows_count']}, "
                f"清理次数={memory_stats['cleanup_count']}",
                "ModelRegistry"
            )
            
            return memory_stats
            
        except ImportError:
            # psutil不可用，使用基础报告
            memory_stats = {
                'models_count': len(self.models),
                'workflows_count': len(self.active_workflows),
                'performance_records_count': len(self.performance_metrics),
                'training_history_count': sum(len(h) for h in self.training_history.values()),
                'cleanup_count': self._memory_cleanup_count,
                'timestamp': time.time(),
                'psutil_available': False
            }
            
            error_handler.log_info(
                f"基础内存报告: 模型数={memory_stats['models_count']}, "
                f"工作流数={memory_stats['workflows_count']}, "
                f"清理次数={memory_stats['cleanup_count']}",
                "ModelRegistry"
            )
            
            return memory_stats
        except Exception as e:
            error_handler.handle_error(e, "ModelRegistry", "内存使用报告失败")
            return {'error': str(e)}
            
    def stop_cleanup_monitor(self):
        """停止内存清理监控线程
        Stop memory cleanup monitoring thread
        """
        try:
            self._stop_cleanup_thread = True
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=5)
            error_handler.log_info("内存清理监控线程已停止", "ModelRegistry")
        except Exception as e:
            error_handler.handle_error(e, "ModelRegistry", "停止内存清理监控线程失败")
        
    def set_agi_coordinator(self, agi_coordinator):
        """设置AGI协调器引用，避免循环依赖
        Set AGI coordinator reference to avoid circular dependencies
        """
        self._agi_coordinator = agi_coordinator
        error_handler.log_info("AGI协调器引用已设置", "ModelRegistry")

    @property
    def agi_coordinator(self):
        """AGI协调器属性 - 修复循环依赖问题
        AGI Coordinator property - fixed circular dependency issue
        """
        if self._agi_coordinator is None:
            # 避免循环依赖，不在这里创建AGICoordinator实例
            error_handler.log_warning("AGI协调器未设置，使用备用协调器", "ModelRegistry")
            return self._create_fallback_agi_coordinator()
        return self._agi_coordinator

    def _create_fallback_agi_coordinator(self):
        """创建简化版的AGI协调器作为备用
        Create simplified AGI coordinator as fallback
        """
        class FallbackAGICoordinator:
            def __init__(self):
                self.models = {}
                
            def on_model_registered(self, model_id, model_instance):
                error_handler.log_info(f"备用协调器: 模型 {model_id} 已注册", "FallbackAGICoordinator")
                
            def on_model_loaded(self, model_id):
                error_handler.log_info(f"备用协调器: 模型 {model_id} 已加载", "FallbackAGICoordinator")
                
            def get_model_coordination(self, model_id):
                return {"status": "fallback", "message": "使用备用协调器"}
                
        return FallbackAGICoordinator()

    def register_model(self, model_id: str, model_class: Type, config: Dict[str, Any] = None):
        """AGI增强版模型注册方法，支持从零开始训练和深度集成
        AGI-enhanced model registration method with from-scratch training and deep integration support
        
        Args:
            model_id: 模型ID / Model ID
            model_class: 模型类 / Model class
            config: 配置字典 / Configuration dictionary
            
        Returns:
            object: 模型实例或None / Model instance or None
        """
        if model_id in self.models:
            error_handler.log_warning(f"模型 {model_id} 已存在，将被替换", "ModelRegistry")
        
        try:
            # AGI级别的模型配置增强
            enhanced_config = config or {}
            
            # 添加AGI系统集成配置
            enhanced_config.update({
                'agi_core': self._agi_core,
                'cognitive_architecture': self._cognitive_architecture,
                'self_learning_system': self._self_learning_system,
                'context_memory': self._context_memory,
                'from_scratch': self.from_scratch_training_enabled,
                'model_registry': self  # 传递模型注册表引用
            })
            
            # 创建模型实例 - 传递单个config参数而不是展开
            model_instance = model_class(enhanced_config)
            self.models[model_id] = model_instance
            self.model_configs[model_id] = enhanced_config
            
            # AGI级别的模型初始化
            self._initialize_agi_model(model_id, model_instance)
            
            error_handler.log_info(f"AGI级别成功注册模型: {model_id}", "ModelRegistry")
            return model_instance
        except Exception as e:
            error_handler.handle_error(e, "ModelRegistry", f"注册模型 {model_id} 失败")
            return None
            
    def _initialize_agi_model(self, model_id: str, model_instance):
        """AGI级别的模型初始化
        AGI-level model initialization
        
        Args:
            model_id: 模型ID
            model_instance: 模型实例
        """
        try:
            # 初始化训练进度跟踪
            self.training_progress[model_id] = {
                'status': 'initialized',
                'epochs_completed': 0,
                'total_epochs': 0,
                'accuracy': 0.0,
                'loss': float('inf'),
                'last_training_time': time.time(),
                'from_scratch': self.from_scratch_training_enabled
            }
            
            # 初始化知识库集成状态
            self.knowledge_base_integration[model_id] = {
                'integrated': False,
                'knowledge_loaded': 0,
                'last_learning_time': 0,
                'learning_efficiency': 0.0
            }
            
            # 如果模型支持AGI方法，进行深度集成
            if hasattr(model_instance, 'initialize_agi'):
                model_instance.initialize_agi(self._agi_core)
                
            # 通知AGI协调器新模型已注册
            self.agi_coordinator.on_model_registered(model_id, model_instance)
            
            # 更新AGI状态
            self.agi_state['total_interactions'] += 1
            
        except Exception as e:
            error_handler.handle_error(e, "ModelRegistry", f"初始化AGI模型 {model_id} 失败")
            
    def get_all_registered_models(self) -> List[str]:
        """获取所有已注册的模型ID列表
        Get list of all registered model IDs
        
        Returns:
            List[str]: 所有已注册的模型ID列表
        """
        return list(self.models.keys())
        
    def load_model(self, model_id: str, config: Dict[str, Any] = None, force_reload: bool = False, priority: int = 1, from_scratch: bool = None):
        """AGI增强版模型加载方法，支持从零开始训练和深度认知集成
        AGI-enhanced model loading method with from-scratch training and deep cognitive integration support
        
        Args:
            model_id: 模型ID / Model ID
            config: 配置字典 / Configuration dictionary
            force_reload: 是否强制重新加载 / Whether to force reload
            priority: 加载优先级，数字越小优先级越高 / Loading priority, smaller number means higher priority
            from_scratch: 是否从零开始训练，None表示使用全局设置 / Whether to train from scratch, None means use global setting
            
        Returns:
            object: 模型实例或None / Model instance or None
        """
        # 检查模型是否已加载
        if model_id in self.models and not force_reload:
            # 如果模型支持访问回调，则调用它
            if hasattr(self.models[model_id], 'on_access'):
                self.models[model_id].on_access()
            return self.models[model_id]
        
        if model_id not in self.model_types:
            error_handler.log_warning(f"未知模型类型: {model_id}", "ModelRegistry")
            return None
        
        # 检查内存使用情况，避免内存溢出
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_percent = psutil.virtual_memory().percent
            
            # 如果内存使用率超过70%，拒绝加载非核心模型
            if memory_percent > 70 and model_id != 'manager':
                error_handler.log_warning(f"内存使用率过高 ({memory_percent:.1f}%)，拒绝加载非核心模型: {model_id}", "ModelRegistry")
                return None
                
            # 如果内存使用率超过80%，即使是核心模型也拒绝加载
            if memory_percent > 80:
                error_handler.log_error(f"内存使用率过高 ({memory_percent:.1f}%)，拒绝加载任何模型: {model_id}", "ModelRegistry")
                return None
                
        except ImportError:
            # psutil不可用，跳过内存检查
            error_handler.log_warning("psutil不可用，跳过内存检查", "ModelRegistry")
        except Exception as e:
            error_handler.handle_error(e, "ModelRegistry", "内存检查失败")
        
        # 防止循环依赖和重复加载 - 检查当前模型是否正在加载中
        if model_id in self._loading_stack:
            error_handler.log_warning(f"检测到循环依赖或重复加载: {model_id} (加载堆栈: {self._loading_stack})", "ModelRegistry")
            return None
        
        # 检查递归深度，防止无限递归
        if priority > 10:  # 最大递归深度为10
            error_handler.log_error(f"模型加载递归深度过大: {model_id} (优先级: {priority})", "ModelRegistry")
            return None
        
        # 将当前模型添加到加载堆栈
        self._loading_stack.append(model_id)
        
        try:
            # AGI级别的依赖解析 - 考虑认知依赖关系
            dependency_order = self._get_agi_dependency_loading_order(model_id)
            
            # 按依赖顺序加载模型，但避免重复加载
            for dep_id in dependency_order:
                if dep_id not in self.models or force_reload:
                    # 检查依赖是否已经在加载堆栈中（防止循环依赖）
                    if dep_id in self._loading_stack:
                        error_handler.log_warning(f"跳过循环依赖: {dep_id} 已在加载堆栈中", "ModelRegistry")
                        continue
                    
                    # 检查依赖是否已经加载过
                    if dep_id in self._loaded_dependencies:
                        error_handler.log_info(f"依赖 {dep_id} 已加载过，跳过重复加载", "ModelRegistry")
                        continue
                    
                    error_handler.log_info(f"AGI级别加载模型 {model_id} 的认知依赖: {dep_id}", "ModelRegistry")
                    
                    # 递归加载依赖，但不重复加载当前模型
                    if dep_id != model_id:
                        dep_model = self.load_model(dep_id, force_reload=force_reload, priority=priority + 1, from_scratch=from_scratch)
                        if dep_model:
                            # 标记依赖已加载
                            self._loaded_dependencies.add(dep_id)
                        else:
                            error_handler.log_warning(f"加载依赖 {dep_id} 失败，但继续加载主模型 {model_id}", "ModelRegistry")
        except Exception as e:
            error_handler.log_warning(f"依赖加载过程中出现错误，但继续加载主模型 {model_id}: {str(e)}", "ModelRegistry")
        
        try:
            # 解析模块路径和类名
            module_path, class_name = self.model_types[model_id].rsplit('.', 1)
            # 动态导入模块
            module = importlib.import_module(module_path)
            # 获取类
            model_class = getattr(module, class_name)
            
            # AGI级别的配置增强
            if config is None:
                config = {}
                
            # 设置从零开始训练标志
            if from_scratch is None:
                from_scratch = self.from_scratch_training_enabled
            config['from_scratch'] = from_scratch
            
            # 增强配置 - 添加AGI系统集成
            config.update({
                'related_models': self._get_agi_related_models(model_id),  # 获取AGI级别的相关模型
                'priority': priority,
    'agi_system': self._agi_core,
                'cognitive_context': self._get_cognitive_context(model_id),
                'learning_capability': self.agi_state['learning_capability']
            })
            
            # AGI优先级加载日志
            if priority <= 1:
                error_handler.log_info(f"AGI高优先级加载模型: {model_id} (从零开始: {from_scratch})", "ModelRegistry")
            
            # 注册并返回模型实例
            model = self.register_model(model_id, model_class, config)
            
            # AGI级别的模型加载通知
            self._notify_agi_model_loaded(model_id, from_scratch)
            
            return model
            
        except Exception as e:
            # 更详细的错误日志记录，特别关注JSON相关错误
            import traceback
            error_details = {
                'model_id': model_id,
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': time.time(),
                'stack_trace': traceback.format_exc(),
                'agi_state': self.agi_state.copy(),
                'config_keys': list(config.keys()) if config else [],
                'model_types_keys': list(self.model_types.keys()) if self.model_types else []
            }
            
            # 特别检查JSON相关错误
            if 'JSON' in str(e) or 'json' in str(e).lower():
                error_details['json_error_detected'] = True
                # 检查可能的JSON配置问题
                if config:
                    error_details['config_has_json_keys'] = any('json' in key.lower() for key in config.keys())
                error_handler.log_error(f"JSON相关错误检测到: {str(e)}", "ModelRegistry")
            
            # 记录详细错误信息（确保只包含可序列化的基本类型）
            safe_error_details = {}
            for key, value in error_details.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    safe_error_details[key] = value
                else:
                    # 对于复杂对象，转换为字符串
                    try:
                        safe_error_details[key] = str(value)
                    except:
                        safe_error_details[key] = "unserializable_value"
            
            self._log_error(safe_error_details)
            
            # 输出更详细的错误信息到控制台
            print(f"DEBUG - ModelRegistry加载模型 {model_id} 失败:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print(f"堆栈跟踪: {traceback.format_exc()}")
            
            error_handler.handle_error(e, "ModelRegistry", f"AGI级别加载模型 {model_id} 失败")
            return None
            
    def _get_agi_dependency_loading_order(self, model_id: str) -> List[str]:
        """AGI级别的依赖加载顺序，考虑认知层次和功能依赖
        AGI-level dependency loading order considering cognitive hierarchy and functional dependencies
        
        Args:
            model_id: 模型ID / Model ID
            
        Returns:
            list: 按AGI认知顺序排列的模型ID列表 / List of model IDs in AGI cognitive order
        """
        # AGI认知层次：基础认知 -> 专业认知 -> 高级认知
        agi_cognitive_hierarchy = {
            'knowledge': 100,      # 知识基础
            'language': 90,        # 语言理解
            'vision_image': 85,    # 视觉感知
            'audio': 80,           # 听觉感知
            'emotion': 75,         # 情感理解
            'planning': 70,        # 规划能力
            'prediction': 65,      # 预测能力
            'collaboration': 60,   # 协作能力
            'optimization': 55,    # 优化能力
            'autonomous': 50,      # 自主能力
            'manager': 40,         # 管理能力
            'value_alignment': 30  # 值对齐（最高层次）
        }
        
        visited = set()
        order = []
        
        def cognitive_dfs(current_id, depth=0):
            if current_id in visited or depth > 10:  # 防止无限递归
                return
            visited.add(current_id)
            
            # 先加载认知层次更高的依赖
            dependencies = self.model_dependencies.get(current_id, [])
            # 按认知层次排序依赖
            dependencies.sort(key=lambda x: agi_cognitive_hierarchy.get(x, 0), reverse=True)
            
            for dep_id in dependencies:
                if dep_id not in visited:
                    cognitive_dfs(dep_id, depth + 1)
            
            order.append(current_id)
        
        cognitive_dfs(model_id)
        return order
        
    def _get_agi_related_models(self, model_id: str) -> Dict[str, Any]:
        """获取AGI级别的相关模型，包括认知关联和功能互补
        Get AGI-level related models including cognitive associations and functional complements
        
        Args:
            model_id: 模型ID / Model ID
            
        Returns:
            dict: AGI级别的相关模型字典 / Dictionary of AGI-level related models
        """
        related = {}
        
        # 获取认知关联模型（基于AGI认知层次）
        cognitive_hierarchy = self._get_cognitive_hierarchy()
        current_level = cognitive_hierarchy.get(model_id, 0)
        
        for other_id, level in cognitive_hierarchy.items():
            if other_id != model_id and abs(level - current_level) <= 20:  # 相近认知层次
                if other_id in self.models:
                    related[f"cognitive_peer_{other_id}"] = self.models[other_id]
        
        # 获取功能互补模型
        functional_complements = self._get_functional_complements(model_id)
        for comp_id in functional_complements:
            if comp_id in self.models:
                related[f"functional_complement_{comp_id}"] = self.models[comp_id]
        
        return related
        
    def _get_cognitive_hierarchy(self) -> Dict[str, int]:
        """获取模型的认知层次评分
        Get cognitive hierarchy scores for models
        
        Returns:
            dict: 模型ID到认知层次评分的映射 / Mapping of model IDs to cognitive hierarchy scores
        """
        hierarchy = {
            'knowledge': 100,      # 知识基础
            'language': 90,        # 语言理解
            'vision_image': 85,    # 视觉感知
            'audio': 80,           # 听觉感知
            'emotion': 75,         # 情感理解
            'planning': 70,        # 规划能力
            'prediction': 65,      # 预测能力
            'collaboration': 60,   # 协作能力
            'optimization': 55,    # 优化能力
            'autonomous': 50,      # 自主能力
            'manager': 40,         # 管理能力
            'value_alignment': 30  # 值对齐（最高层次）
        }
        return hierarchy
        
    def _get_functional_complements(self, model_id: str) -> List[str]:
        """获取功能互补模型列表
        Get list of functionally complementary models
        
        Args:
            model_id: 模型ID / Model ID
            
        Returns:
            list: 功能互补模型ID列表 / List of functionally complementary model IDs
        """
        # 定义功能互补关系
        complements = {
            'language': ['knowledge', 'emotion'],
            'vision_image': ['knowledge', 'spatial'],
            'audio': ['language', 'emotion'],
            'planning': ['knowledge', 'prediction'],
            'prediction': ['knowledge', 'optimization'],
            'manager': ['knowledge', 'language', 'planning']
        }
        return complements.get(model_id, [])
        
    def _get_cognitive_context(self, model_id: str) -> Dict[str, Any]:
        """获取模型的认知上下文
        Get cognitive context for model
        
        Args:
            model_id: 模型ID / Model ID
            
        Returns:
            dict: 认知上下文信息 / Cognitive context information
        """
        return {
            'model_id': model_id,
            'cognitive_level': self._get_cognitive_hierarchy().get(model_id, 50),
            'agi_state': self.agi_state.copy(),
            'available_models': list(self.models.keys()),
            'timestamp': time.time()
        }
        
    def _notify_agi_model_loaded(self, model_id: str, from_scratch: bool):
        """AGI级别的模型加载通知 - 完全修复JSON序列化问题
        AGI-level model loaded notification - completely fixed JSON serialization issues
        
        Args:
            model_id: 模型ID / Model ID
            from_scratch: 是否从零开始训练 / Whether training from scratch
        """
        try:
            # 使用error_handler记录信息，避免print可能带来的问题
            error_handler.log_info(f"AGI system integrating model: {model_id} (from_scratch: {from_scratch})", "ModelRegistry")
            
            # 安全的AGI状态更新 - 只使用基本数据类型，确保完全可序列化
            try:
                # 直接使用基本数值操作，避免任何复杂对象
                # 确保所有值都是基本类型，可以直接JSON序列化
                knowledge_acc = float(self.agi_state.get('knowledge_accumulation', 0.0))
                consciousness = float(self.agi_state.get('consciousness_level', 0.1))
                interactions = int(self.agi_state.get('total_interactions', 0))
                learning_capability = float(self.agi_state.get('learning_capability', 0.8))
                problem_solving = float(self.agi_state.get('problem_solving_ability', 0.7))
                creativity = float(self.agi_state.get('creativity_level', 0.6))
                ethical_alignment = float(self.agi_state.get('ethical_alignment', 0.9))
                last_reflection = float(self.agi_state.get('last_self_reflection', time.time()))
                
                # 更新状态，确保所有值都是基本类型且完全可序列化
                self.agi_state = {
                    'consciousness_level': float(min(1.0, consciousness + 0.05)),
                    'learning_capability': float(learning_capability),
                    'problem_solving_ability': float(problem_solving),
                    'creativity_level': float(creativity),
                    'ethical_alignment': float(ethical_alignment),
                    'last_self_reflection': float(time.time()),
                    'total_interactions': int(interactions + 1),
                    'knowledge_accumulation': float(min(1.0, knowledge_acc + 0.1))
                }
                
                # 验证AGI状态完全可序列化
                try:
                    json.dumps(self.agi_state)
                except Exception as json_error:
                    error_handler.log_warning(f"AGI state JSON serialization failed: {json_error}", "ModelRegistry")
                    # 如果JSON序列化失败，使用最安全的基本值
                    self.agi_state = {
                        'consciousness_level': 0.1,
                        'learning_capability': 0.8,
                        'problem_solving_ability': 0.7,
                        'creativity_level': 0.6,
                        'ethical_alignment': 0.9,
                        'last_self_reflection': float(time.time()),
                        'total_interactions': 0,
                        'knowledge_accumulation': 0.0
                    }
                        
            except Exception as e:
                error_handler.log_warning(f"AGI state update failed for {model_id}: {type(e).__name__}", "ModelRegistry")
                # 如果状态更新失败，重置为安全状态 - 只包含基本类型
                self.agi_state = {
                    'consciousness_level': 0.1,
                    'learning_capability': 0.8,
                    'problem_solving_ability': 0.7,
                    'creativity_level': 0.6,
                    'ethical_alignment': 0.9,
                    'last_self_reflection': float(time.time()),
                    'total_interactions': 0,
                    'knowledge_accumulation': 0.0
                }
            
            error_handler.log_info(f"AGI system successfully integrated model: {model_id}", "ModelRegistry")
        except Exception as e:
            # 使用最简单的错误处理
            error_handler.log_warning(f"Error in AGI model load notification for {model_id}: {type(e).__name__}", "ModelRegistry")
            # 确保AGI状态是安全的 - 只包含基本类型
            self.agi_state = {
                'consciousness_level': 0.1,
                'learning_capability': 0.8,
                'problem_solving_ability': 0.7,
                'creativity_level': 0.6,
                'ethical_alignment': 0.9,
                'last_self_reflection': float(time.time()),
                'total_interactions': 0,
                'knowledge_accumulation': 0.0
            }
        
    def is_model_registered(self, model_id: str) -> bool:
        """检查模型是否已注册
        Check if model is registered
        
        Args:
            model_id: 模型ID
            
        Returns:
            bool: 如果模型已注册返回True，否则False
        """
        return model_id in self.models

    def get_model(self, model_id: str):
        """获取已注册的模型实例
        Get registered model instance
        
        Args:
            model_id: 模型ID / Model ID
            
        Returns:
            object: 模型实例或None / Model instance or None
        """
        model = self.models.get(model_id)
        # 如果模型支持访问回调，则调用它
        # If the model supports access callback, call it
        if model and hasattr(model, 'on_access'):
            model.on_access()
        return model
        
    def get_all_models(self):
        """获取所有已注册的模型
        Get all registered models
        
        Returns:
            dict: 模型字典 / Models dictionary
        """
        return self.models.copy()
    
    def get_registered_models(self):
        """获取所有已注册的模型（别名方法）
        Get all registered models (alias method)
        
        Returns:
            dict: 模型字典 / Models dictionary
        """
        return self.get_all_models()
        
    def get_all_model_types(self):
        """获取所有模型类型
        Get all model types
        
        Returns:
            list: 模型类型列表 / List of model types
        """
        return list(self.model_types.keys())
        
    def unload_model(self, model_id: str):
        """卸载指定模型
        Unload specified model
        
        Args:
            model_id: 模型ID / Model ID
            
        Returns:
            bool: 卸载是否成功 / Whether unloading was successful
        """
        if model_id in self.models:
            del self.models[model_id]
            if model_id in self.model_configs:
                del self.model_configs[model_id]
            error_handler.log_info(f"已卸载模型: {model_id}", "ModelRegistry")
            return True
        return False
        
    def _get_dependency_loading_order(self, model_id: str) -> List[str]:
        """获取模型依赖加载顺序，避免循环依赖
        Get model dependency loading order, avoiding circular dependencies
        
        Args:
            model_id: 模型ID / Model ID
            
        Returns:
            list: 按顺序排列的模型ID列表 / List of model IDs in loading order
        """
        visited = set()
        order = []
        
        def dfs(current_id):
            if current_id in visited:
                return
            visited.add(current_id)
            
            if current_id in self.model_dependencies:
                for dep_id in self.model_dependencies[current_id]:
                    if dep_id not in visited:
                        dfs(dep_id)
            
            order.append(current_id)
        
        dfs(model_id)
        return order
    
    def _get_related_models(self, model_id: str) -> Dict[str, Any]:
        """获取与指定模型相关的其他模型
        Get other models related to the specified model
        
        Args:
            model_id: 模型ID / Model ID
            
        Returns:
            dict: 相关模型字典 / Dictionary of related models
        """
        related = {}
        
        # 获取依赖此模型的模型
        for m_id, deps in self.model_dependencies.items():
            if model_id in deps:
                if m_id in self.models:
                    related[f"used_by_{m_id}"] = self.models[m_id]
        
        # 获取此模型依赖的模型
        if model_id in self.model_dependencies:
            for dep_id in self.model_dependencies[model_id]:
                if dep_id in self.models:
                    related[f"depends_on_{dep_id}"] = self.models[dep_id]
        
        return related
    
    def _notify_model_loaded(self, model_id: str):
        """通知系统新模型已加载
        Notify system that a new model has been loaded
        
        Args:
            model_id: 模型ID / Model ID
        """
        # 初始化认知融合引擎和知识迁移引擎（如果尚未初始化）
        if self.cognitive_fusion_engine is None:
            self._initialize_cognitive_fusion_engine()
        
        if self.knowledge_transfer_engine is None:
            self._initialize_knowledge_transfer_engine()
        
        if self.context_manager is None:
            self._initialize_context_manager()
        
        # 记录模型加载事件
        error_handler.log_info(f"模型加载事件通知: {model_id}", "ModelRegistry")
    
    def switch_model_to_external(self, model_id: str, api_config: Dict[str, Any]) -> Dict[str, Any]:
        """Switch model to external API mode
        
        Args:
            model_id: Model ID
            api_config: API configuration including api_url, api_key, model_name, etc.
            
        Returns:
            Dict[str, Any]: Switch result
        """
        try:
            # 验证模型存在
            if model_id not in self.models:
                return {"status": "error", "message": f"模型{model_id}不存在"}
            
            # 规范化API配置字段
            normalized_api_config = {}
            if 'api_url' in api_config:
                normalized_api_config['api_url'] = api_config['api_url']
            elif 'url' in api_config:
                normalized_api_config['api_url'] = api_config['url']
            else:
                return {"status": "error", "message": "缺少必要的API配置项: api_url或url"}
                
            if 'api_key' in api_config:
                normalized_api_config['api_key'] = api_config['api_key']
            else:
                return {"status": "error", "message": "缺少必要的API配置项: api_key"}
                
            if 'model_name' in api_config:
                normalized_api_config['model_name'] = api_config['model_name']
            else:
                normalized_api_config['model_name'] = model_id
                
            if 'source' in api_config:
                normalized_api_config['source'] = api_config['source']
            else:
                normalized_api_config['source'] = 'external'
                
            if 'endpoint' in api_config:
                normalized_api_config['endpoint'] = api_config['endpoint']
                
            # 保存配置到系统设置
            from .system_settings_manager import system_settings_manager
            system_settings_manager.update_model_setting(model_id, {
                "type": "api",
                "source": "external",
                "api_url": normalized_api_config['api_url'],
                "api_key": normalized_api_config['api_key'],
                "model_name": normalized_api_config['model_name'],
                "endpoint": normalized_api_config.get('endpoint', '')
            })
            
            # 加载外部模型
            external_model = self.load_external_model(model_id, normalized_api_config)
            if not external_model:
                return {"status": "error", "message": f"无法加载外部模型: {model_id}"}
            
            # 检查连接状态
            status = external_model.get_status()
            if status.get('status') == 'connected':
                error_handler.log_info(f"成功将模型{model_id}切换到外部API模式", "ModelRegistry")
                return {
                    "status": "success", 
                    "message": f"模型{model_id}已成功切换到外部API模式",
                    "model": model_id,
                    "api_status": status
                }
            else:
                # 连接失败，恢复原模型
                self.unload_model(model_id)
                self.load_model(model_id)
                return {"status": "error", "message": f"外部API连接失败: {status.get('error', '未知错误')}"}
        except Exception as e:
            error_handler.handle_error(e, "ModelRegistry", f"切换模型{model_id}到外部API模式失败")
            return {"status": "error", "message": str(e)}
            
    def switch_model_to_local(self, model_id: str) -> Dict[str, Any]:
        """Switch model back to local mode
        
        Args:
            model_id: Model ID
            
        Returns:
            Dict[str, Any]: Switch result
        """
        try:
            # 验证模型存在
            if model_id not in self.models:
                return {"status": "error", "message": f"模型{model_id}不存在"}
            
            # 保存配置到系统设置
            from .system_settings_manager import system_settings_manager
            system_settings_manager.update_model_setting(model_id, {"type": "local", "source": "local"})
            
            # 卸载并重新加载本地模型
            self.unload_model(model_id)
            local_model = self.load_model(model_id)
            
            if local_model:
                error_handler.log_info(f"成功将模型{model_id}切换到本地模式", "ModelRegistry")
                return {"status": "success", "message": f"模型{model_id}已成功切换到本地模式", "model": model_id}
            else:
                return {"status": "error", "message": f"无法加载本地模型: {model_id}"}
        except Exception as e:
            error_handler.handle_error(e, "ModelRegistry", f"切换模型{model_id}到本地模式失败")
            return {"status": "error", "message": str(e)}
            
    def get_model_mode(self, model_id: str) -> str:
        """获取模型当前的运行模式
        Get current model operation mode
        
        Args:
            model_id: 模型ID
            
        Returns:
            str: 'local' 或 'external'
        """
        from .system_settings_manager import system_settings_manager
        model_config = system_settings_manager.get_model_config(model_id)
        
        if model_config.get("source") == "external" or model_config.get("type") == "api":
            return "external"
        return "local"
    
    def _log_error(self, error_details: Dict[str, Any]):
        """记录详细的错误信息
        Log detailed error information
        
        Args:
            error_details: 错误详情字典 / Dictionary of error details
        """
        # 这里可以扩展为将错误信息写入专门的日志文件或错误跟踪系统
        error_handler.log_error(f"详细错误: {error_details}", "ModelRegistry")
    
    def _resolve_conflict_majority(self, model_responses: List[Tuple[str, Any]]) -> Any:
        """使用多数投票解决模型响应冲突
        Resolve model response conflicts using majority vote
        
        Args:
            model_responses: 模型响应列表，每个元素是(模型ID, 响应)的元组
            
        Returns:
            解决冲突后的结果 / Result after conflict resolution
        """
        # 简单多数投票实现，实际应用中可能需要更复杂的算法
        from collections import Counter
        
        # 提取所有响应值
        responses = [response for _, response in model_responses]
        
        # 处理可哈希的响应
        if all(isinstance(r, (str, int, float, bool, type(None))) for r in responses):
            counter = Counter(responses)
            most_common = counter.most_common(1)
            if most_common:
                return most_common[0][0]
        
        # 默认返回第一个响应
        return responses[0] if responses else None
    
    def _resolve_conflict_expert(self, model_responses: List[Tuple[str, Any]]) -> Any:
        """使用专家模型策略解决冲突
        Resolve conflicts using expert model strategy
        
        Args:
            model_responses: 模型响应列表，每个元素是(模型ID, 响应)的元组
            
        Returns:
            解决冲突后的结果 / Result after conflict resolution
        """
        # 基于模型性能指标选择专家模型
        best_model_id = None
        best_score = -1
        
        for model_id, response in model_responses:
            if model_id in self.performance_metrics:
                # 计算综合得分（可以根据具体需求调整权重）
                metrics = self.performance_metrics[model_id]
                score = (metrics.get('success_rate', 0) * 0.4 +
                         metrics.get('accuracy', 0) * 0.3 +
                         metrics.get('collaboration_score', 0) * 0.3)
                
                if score > best_score:
                    best_score = score
                    best_model_id = model_id
        
        # 返回专家模型的响应
        for model_id, response in model_responses:
            if model_id == best_model_id:
                return response
        
        # 默认返回第一个响应
        return model_responses[0][1] if model_responses else None
    
    def _resolve_conflict_hierarchical(self, model_responses: List[Tuple[str, Any]]) -> Any:
        """使用层次化策略解决冲突
        Resolve conflicts using hierarchical strategy
        
        Args:
            model_responses: 模型响应列表，每个元素是(模型ID, 响应)的元组
            
        Returns:
            解决冲突后的结果 / Result after conflict resolution
        """
        # 定义模型层次优先级（可以根据具体需求调整）
        hierarchy = {
            'manager': 100,
            'knowledge': 90,
            'language': 80,
            'planning': 70,
            'prediction': 60,
            'collaboration': 50,
            'emotion': 40,
            'vision_image': 30,
            'audio': 20,
            'sensor': 10
        }
        
        # 找出层次最高的模型
        highest_priority = -1
        highest_response = None
        
        for model_id, response in model_responses:
            priority = hierarchy.get(model_id, 0)
            if priority > highest_priority:
                highest_priority = priority
                highest_response = response
        
        return highest_response
    
    def _resolve_conflict_agi_consensus(self, model_responses: List[Tuple[str, Any]]) -> Any:
        """使用AGI共识策略解决冲突，结合多个模型的智能和协作能力
        Resolve conflicts using AGI consensus strategy, combining intelligence and collaboration of multiple models
        
        Args:
            model_responses: 模型响应列表，每个元素是(模型ID, 响应)的元组
            
        Returns:
            解决冲突后的结果 / Result after conflict resolution
        """
        if not model_responses:
            return None
        
        # 如果只有一个响应，直接返回
        if len(model_responses) == 1:
            return model_responses[0][1]
        
        try:
            # 使用AGI协调器进行共识决策
            if self._agi_coordinator is not None:
                # 准备模型响应数据
                model_data = []
                for model_id, response in model_responses:
                    model_data.append({
                        'model_id': model_id,
                        'response': response,
                        'performance': self.performance_metrics.get(model_id, {}),
                        'collaboration_score': self.performance_metrics.get(model_id, {}).get('collaboration_score', 0.5)
                    })
                
                # 尝试使用AGI协调器达成共识
                if hasattr(self._agi_coordinator, 'resolve_conflicts'):
                    consensus_result = self._agi_coordinator.resolve_conflicts(model_data)
                    if consensus_result is not None:
                        return consensus_result
            
            # 如果AGI协调器不可用或没有提供结果，使用加权共识算法
            return self._weighted_consensus(model_responses)
            
        except Exception as e:
            error_handler.handle_error(e, "ModelRegistry", "AGI共识策略失败，回退到加权共识")
            return self._weighted_consensus(model_responses)
    
    def _weighted_consensus(self, model_responses: List[Tuple[str, Any]]) -> Any:
        """加权共识算法，基于模型性能和协作能力
        Weighted consensus algorithm based on model performance and collaboration capability
        
        Args:
            model_responses: 模型响应列表，每个元素是(模型ID, 响应)的元组
            
        Returns:
            加权共识结果 / Weighted consensus result
        """
        if not model_responses:
            return None
        
        # 计算每个模型的权重
        weights = {}
        total_weight = 0
        
        for model_id, response in model_responses:
            # 基础权重
            weight = 1.0
            
            # 基于性能指标调整权重
            if model_id in self.performance_metrics:
                metrics = self.performance_metrics[model_id]
                # 综合权重 = 成功率(40%) + 协作能力(30%) + 最近表现(30%)
                success_weight = metrics.get('success_rate', 0.5) * 0.4
                collaboration_weight = metrics.get('collaboration_score', 0.5) * 0.3
                recent_performance_weight = self._calculate_recent_performance(model_id) * 0.3
                weight = success_weight + collaboration_weight + recent_performance_weight
            
            weights[model_id] = weight
            total_weight += weight
        
        # 如果所有权重都是0，使用平均权重
        if total_weight == 0:
            equal_weight = 1.0 / len(model_responses)
            for model_id in weights:
                weights[model_id] = equal_weight
            total_weight = 1.0
        
        # 归一化权重
        for model_id in weights:
            weights[model_id] /= total_weight
        
        # 对于数值型响应，计算加权平均
        if all(isinstance(r, (int, float)) for _, r in model_responses):
            weighted_sum = 0
            for model_id, response in model_responses:
                weighted_sum += response * weights[model_id]
            return weighted_sum
        
        # 对于字符串响应，选择权重最高的响应
        elif all(isinstance(r, str) for _, r in model_responses):
            best_model_id = max(weights, key=weights.get)
            for model_id, response in model_responses:
                if model_id == best_model_id:
                    return response
        
        # 对于字典响应，合并所有响应，按权重调整
        elif all(isinstance(r, dict) for _, r in model_responses):
            merged_result = {}
            for model_id, response in model_responses:
                for key, value in response.items():
                    if key not in merged_result:
                        merged_result[key] = value * weights[model_id]
                    else:
                        merged_result[key] += value * weights[model_id]
            return merged_result
        
        # 默认返回权重最高的响应
        best_model_id = max(weights, key=weights.get)
        for model_id, response in model_responses:
            if model_id == best_model_id:
                return response
    
    def _calculate_recent_performance(self, model_id: str) -> float:
        """计算模型的近期表现评分
        Calculate recent performance score for a model
        
        Args:
            model_id: 模型ID
            
        Returns:
            float: 近期表现评分 (0-1)
        """
        if model_id not in self.performance_metrics:
            return 0.5
        
        metrics = self.performance_metrics[model_id]
        
        # 检查最近活动时间
        last_activity = metrics.get('last_collaboration_time', 0)
        current_time = time.time()
        
        # 如果超过1小时没有活动，降低评分
        time_decay = 1.0
        if current_time - last_activity > 3600:  # 1小时
            time_decay = max(0.1, 1.0 - (current_time - last_activity - 3600) / 36000)  # 每10小时衰减0.1
        
        # 基于最近的成功率和错误率计算表现
        success_rate = metrics.get('success_rate', 0.5)
        error_rate = metrics.get('error_rate', 0)
        
        # 表现评分 = 成功率 * (1 - 错误率) * 时间衰减
        performance_score = success_rate * (1 - error_rate) * time_decay
        
        return max(0, min(1, performance_score))
    
    def _initialize_cognitive_fusion_engine(self):
        """初始化认知融合引擎
        Initialize cognitive fusion engine
        """
        try:
            # 尝试导入并初始化认知融合引擎
            from .cognitive_fusion import CognitiveFusionEngine  # type: ignore
            self.cognitive_fusion_engine = CognitiveFusionEngine(self)
            error_handler.log_info("认知融合引擎初始化成功", "ModelRegistry")
        except ImportError:
            error_handler.log_warning("认知融合引擎模块不可用，将使用内置简化版本", "ModelRegistry")
            # 创建简化版本的认知融合引擎
            self.cognitive_fusion_engine = {
                'fuse_results': self._simple_fusion
            }
    
    def _initialize_knowledge_transfer_engine(self):
        """初始化知识迁移引擎
        Initialize knowledge transfer engine
        """
        try:
            # 尝试导入并初始化知识迁移引擎
            from .knowledge_transfer import KnowledgeTransferEngine  # type: ignore
            self.knowledge_transfer_engine = KnowledgeTransferEngine(self)
            error_handler.log_info("知识迁移引擎初始化成功", "ModelRegistry")
        except ImportError:
            error_handler.log_warning("知识迁移引擎模块不可用，将使用内置简化版本", "ModelRegistry")
            # 创建简化版本的知识迁移引擎
            self.knowledge_transfer_engine = {
                'transfer_knowledge': self._simple_knowledge_transfer
            }
    
    def _initialize_context_manager(self):
        """初始化上下文管理器
        Initialize context manager
        """
        try:
            # 尝试导入并初始化上下文管理器
            from .context_management import ContextManager  # type: ignore
            self.context_manager = ContextManager(self)
            error_handler.log_info("上下文管理器初始化成功", "ModelRegistry")
        except ImportError:
            error_handler.log_warning("上下文管理器模块不可用，将使用内置简化版本", "ModelRegistry")
            # 创建简化版本的上下文管理器
            self.context_manager = {
                'get_context': self._simple_get_context,
                'update_context': self._simple_update_context
            }
    
    def _simple_fusion(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """简化版结果融合实现
        Simple result fusion implementation
        
        Args:
            results: 来自不同模型的结果列表
            
        Returns:
            融合后的结果 / Fused result
        """
        fused_result = {}
        
        # 简单合并所有结果
        for result in results:
            for key, value in result.items():
                if key not in fused_result:
                    fused_result[key] = value
                elif isinstance(value, dict) and isinstance(fused_result[key], dict):
                    # 递归合并字典
                    fused_result[key].update(value)
        
        return fused_result
    
    def _simple_knowledge_transfer(self, source_model_id: str, target_model_id: str, knowledge_type: str = None):
        """简化版知识迁移实现
        Simple knowledge transfer implementation
        
        Args:
            source_model_id: 源模型ID / Source model ID
            target_model_id: 目标模型ID / Target model ID
            knowledge_type: 知识类型（可选） / Knowledge type (optional)
        """
        if source_model_id not in self.models or target_model_id not in self.models:
            return False
        
        source_model = self.models[source_model_id]
        target_model = self.models[target_model_id]
        
        # 尝试调用模型的知识迁移方法
        try:
            if hasattr(source_model, 'transfer_knowledge'):
                source_model.transfer_knowledge(target_model, knowledge_type)
                return True
            if hasattr(target_model, 'receive_knowledge'):
                if hasattr(source_model, 'get_knowledge'):
                    knowledge = source_model.get_knowledge(knowledge_type)
                    target_model.receive_knowledge(knowledge)
                    return True
        except Exception as e:
            error_handler.handle_error(e, "ModelRegistry", f"知识迁移失败: {source_model_id} -> {target_model_id}")
        
        return False
    
    def _simple_get_context(self, context_id: str = 'default') -> Dict[str, Any]:
        """简化版上下文获取实现
        Simple context retrieval implementation
        
        Args:
            context_id: 上下文ID / Context ID
            
        Returns:
            上下文数据 / Context data
        """
        # 简单实现，实际应用中可能需要更复杂的上下文管理
        return {'models': list(self.models.keys()), 'timestamp': time.time()}
    
    def _simple_update_context(self, context_id: str = 'default', updates: Dict[str, Any] = None):
        """简化版上下文更新实现
        Simple context update implementation
        
        Args:
            context_id: 上下文ID / Context ID
            updates: 要更新的上下文数据 / Context data to update
        """
        # 简单实现，实际应用中可能需要更复杂的上下文管理
        pass
    
    def _optimize_task_allocation(self, subtasks: Dict[str, Any]) -> Dict[str, Any]:
        """基于模型性能和当前负载优化任务分配
        Optimize task allocation based on model performance and current load
        
        Args:
            subtasks: 原始任务分配 / Original task allocation
            
        Returns:
            dict: 优化后的任务分配 / Optimized task allocation
        """
        # 获取实时指标
        metrics = self.get_realtime_metrics()
        
        # 构建模型性能评分
        model_scores = {}
        for model_id in set(subtasks.keys()):
            if model_id in metrics:
                perf = metrics[model_id].get('performance', {})
                # 综合评分 = 成功率 * 0.3 + (1-资源使用率) * 0.3 + (1-延迟) * 0.2 + 协作能力 * 0.2
                score = (perf.get('success_rate', 0) * 0.3 +
                        (1 - min(1, perf.get('resource_usage', 0) / 100)) * 0.3 +
                        (1 - min(1, perf.get('latency', 0) / 1000)) * 0.2 +
                        perf.get('collaboration_score', 0.5) * 0.2)
                model_scores[model_id] = score
            else:
                model_scores[model_id] = 0.5  # 默认中等评分
        
        # 如果有多个任务分配给同一模型，考虑负载均衡
        task_count = {model_id: 0 for model_id in subtasks.keys()}
        for model_id in subtasks.keys():
            task_count[model_id] += 1
        
        # 这里可以实现更复杂的任务重分配逻辑
        # 对于简单版本，我们保持原始分配，但可以根据需要扩展
        
        return subtasks
    
    def _execute_subtasks_parallel(self, subtasks: Dict[str, Any], context: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """并行执行多个子任务
        Execute multiple subtasks in parallel
        
        Args:
            subtasks: 子任务字典 / Subtasks dictionary
            context: 上下文信息 / Context information
            
        Returns:
            tuple: (结果字典, 执行时间字典) / (Results dictionary, Execution times dictionary)
        """
        results = {}
        execution_times = {}
        
        # 定义任务执行函数
        def execute_task(model_id, task):
            start_time = time.time()
            try:
                model = self.get_model(model_id)
                if model:
                    # 添加上下文信息到任务中
                    if context:
                        if isinstance(task, dict):
                            task['context'] = context
                    result = model.execute(task)
                    exec_time = time.time() - start_time
                    return model_id, result, exec_time
                else:
                    return model_id, None, time.time() - start_time
            except Exception as e:
                error_handler.handle_error(e, "ModelRegistry", f"执行子任务失败: {model_id}")
                return model_id, None, time.time() - start_time
        
        # 使用线程池并行执行任务
        futures = []
        for model_id, task in subtasks.items():
            future = self.executor.submit(execute_task, model_id, task)
            futures.append(future)
        
        # 收集结果
        for future in futures:
            model_id, result, exec_time = future.result()
            results[model_id] = result
            execution_times[model_id] = exec_time
        
        return results, execution_times
    
    def _execute_subtasks_serial(self, subtasks: Dict[str, Any], context: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """串行执行多个子任务
        Execute multiple subtasks serially
        
        Args:
            subtasks: 子任务字典 / Subtasks dictionary
            context: 上下文信息 / Context information
            
        Returns:
            tuple: (结果字典, 执行时间字典) / (Results dictionary, Execution times dictionary)
        """
        results = {}
        execution_times = {}
        
        # 串行执行每个任务
        for model_id, task in subtasks.items():
            start_time = time.time()
            try:
                model = self.get_model(model_id)
                if model:
                    # 添加上下文信息到任务中
                    if context:
                        if isinstance(task, dict):
                            task['context'] = context
                    results[model_id] = model.execute(task)
                    execution_times[model_id] = time.time() - start_time
                else:
                    results[model_id] = None
                    execution_times[model_id] = time.time() - start_time
            except Exception as e:
                error_handler.handle_error(e, "ModelRegistry", f"执行子任务失败: {model_id}")
                results[model_id] = None
                execution_times[model_id] = time.time() - start_time
        
        return results, execution_times
    
    def _fuse_results(self, results: Dict[str, Any], conflict_strategy: str) -> Any:
        """融合多个模型的结果
        Fuse results from multiple models
        
        Args:
            results: 模型结果字典 / Model results dictionary
            conflict_strategy: 冲突解决策略 / Conflict resolution strategy
            
        Returns:
            融合后的结果 / Fused result
        """
        # 过滤掉None结果
        valid_results = [(model_id, result) for model_id, result in results.items() if result is not None]
        
        if not valid_results:
            return None
        
        # 如果只有一个有效结果，直接返回
        if len(valid_results) == 1:
            return valid_results[0][1]
        
        # 使用认知融合引擎（如果可用）
        if self.cognitive_fusion_engine:
            try:
                # 准备结果列表
                results_list = []
                for model_id, result in valid_results:
                    if isinstance(result, dict):
                        result_with_meta = result.copy()
                        result_with_meta['model_id'] = model_id
                        results_list.append(result_with_meta)
                    else:
                        # 为非字典结果添加元数据
                        results_list.append({'data': result, 'model_id': model_id})
                
                if isinstance(self.cognitive_fusion_engine, dict):
                    return self.cognitive_fusion_engine['fuse_results'](results_list)
                else:
                    return self.cognitive_fusion_engine.fuse_results(results_list, strategy=conflict_strategy)
            except Exception as e:
                error_handler.handle_error(e, "ModelRegistry", "使用认知融合引擎失败，回退到基础融合方法")
        
        # 如果认知融合引擎不可用或失败，使用指定的冲突解决策略
        if conflict_strategy in self.conflict_resolution_strategies:
            return self.conflict_resolution_strategies[conflict_strategy](valid_results)
        else:
            # 如果指定的策略不可用，使用默认策略
            return self.conflict_resolution_strategies[self.default_conflict_strategy](valid_results)
    
    def _update_collaboration_metrics(self, results: Dict[str, Any], execution_times: Dict[str, float]):
        """更新模型协作性能指标
        Update model collaboration performance metrics
        
        Args:
            results: 模型结果字典 / Model results dictionary
            execution_times: 执行时间字典 / Execution times dictionary
        """
        for model_id, result in results.items():
            if model_id not in self.performance_metrics:
                self.performance_metrics[model_id] = {
                    'success_rate': 0.0,
                    'latency': 0.0,
                    'accuracy': 0.0,
                    'resource_usage': 0.0,
                    'collaboration_score': 0.0,
                    'calls': 0
                }
            
            # 更新性能指标
            metrics = self.performance_metrics[model_id]
            calls = metrics.get('calls', 0) + 1
            
            # 更新延迟
            if model_id in execution_times:
                metrics['latency'] = ((metrics.get('latency', 0) * (calls - 1)) + execution_times[model_id]) / calls
            
            # 更新成功率（假设非None结果为成功）
            success = 1.0 if result is not None else 0.0
            metrics['success_rate'] = ((metrics.get('success_rate', 0) * (calls - 1)) + success) / calls
            
            # 更新协作评分（这里简单实现，可以根据实际情况扩展）
            # 例如：考虑模型在协作中的贡献度、兼容性等
            metrics['collaboration_score'] = min(1.0, metrics.get('collaboration_score', 0) + 0.01)
            
            metrics['calls'] = calls
            metrics['last_collaboration_time'] = time.time()
    
    def _trigger_knowledge_transfer(self, results: Dict[str, Any]):
        """根据模型结果触发知识迁移
        Trigger knowledge transfer based on model results
        
        Args:
            results: 模型结果字典 / Model results dictionary
        """
        try:
            # 找出表现最好和最差的模型
            best_model = None
            best_score = -1
            worst_model = None
            worst_score = 2
            
            for model_id in results.keys():
                if model_id in self.performance_metrics:
                    metrics = self.performance_metrics[model_id]
                    # 计算综合评分
                    score = (metrics.get('success_rate', 0) * 0.4 +
                            metrics.get('accuracy', 0) * 0.3 +
                            metrics.get('collaboration_score', 0) * 0.3)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model_id
                    
                    if score < worst_score:
                        worst_score = score
                        worst_model = model_id
            
            # 如果存在明显的表现差异，触发知识迁移
            if best_model and worst_model and (best_score - worst_score) > 0.2:
                # 从表现好的模型向表现差的模型迁移知识
                if self.knowledge_transfer_engine:
                    try:
                        if isinstance(self.knowledge_transfer_engine, dict):
                            success = self.knowledge_transfer_engine['transfer_knowledge'](best_model, worst_model)
                        else:
                            success = self.knowledge_transfer_engine.transfer_knowledge(best_model, worst_model)
                        
                        if success:
                            error_handler.log_info(f"成功从模型 {best_model} 向模型 {worst_model} 迁移知识", "ModelRegistry")
                    except Exception as e:
                        error_handler.handle_error(e, "ModelRegistry", f"知识迁移失败: {best_model} -> {worst_model}")
        except Exception as e:
            error_handler.handle_error(e, "ModelRegistry", "触发知识迁移过程失败")
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """获取指定工作流的状态
        Get the status of a specific workflow
        
        Args:
            workflow_id: 工作流ID / Workflow ID
            
        Returns:
            dict: 工作流状态信息 / Workflow status information
        """
        with self.workflow_lock:
            if workflow_id in self.active_workflows:
                return self.active_workflows[workflow_id].copy()
            else:
                error_handler.log_warning(f"找不到工作流: {workflow_id}", "ModelRegistry")
                return {}
    
    def get_all_workflows(self) -> Dict[str, Dict[str, Any]]:
        """获取所有工作流的信息
        Get information about all workflows
        
        Returns:
            dict: 工作流ID到状态信息的映射 / Mapping of workflow IDs to status information
        """
        with self.workflow_lock:
            return {k: v.copy() for k, v in self.active_workflows.items()}
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """取消指定的工作流
        Cancel a specific workflow
        
        Args:
            workflow_id: 工作流ID / Workflow ID
            
        Returns:
            bool: 取消是否成功 / Whether cancellation was successful
        """
        with self.workflow_lock:
            if workflow_id in self.active_workflows:
                if self.active_workflows[workflow_id]['status'] == 'active':
                    self.active_workflows[workflow_id]['status'] = 'cancelled'
                    self.active_workflows[workflow_id]['end_time'] = time.time()
                    error_handler.log_info(f"工作流已取消: {workflow_id}", "ModelRegistry")
                    return True
                else:
                    error_handler.log_warning(f"工作流 {workflow_id} 不在活动状态，无法取消", "ModelRegistry")
                    return False
            else:
                error_handler.log_warning(f"找不到工作流: {workflow_id}", "ModelRegistry")
                return False
    
    def clean_workflows(self, older_than_seconds: int = 3600) -> int:
        """清理指定时间之前的工作流
        Clean up workflows older than the specified time
        
        Args:
            older_than_seconds: 清理多少秒之前的工作流 / Clean up workflows older than this number of seconds
            
        Returns:
            int: 清理的工作流数量 / Number of workflows cleaned up
        """
        cutoff_time = time.time() - older_than_seconds
        workflows_to_clean = []
        
        with self.workflow_lock:
            for workflow_id, workflow_info in self.active_workflows.items():
                # 只清理已完成、已失败或已取消的工作流
                if (workflow_info.get('status') in ['completed', 'failed', 'cancelled'] and 
                    workflow_info.get('end_time', 0) < cutoff_time):
                    workflows_to_clean.append(workflow_id)
            
            for workflow_id in workflows_to_clean:
                del self.active_workflows[workflow_id]
        
        error_handler.log_info(f"已清理 {len(workflows_to_clean)} 个过期工作流", "ModelRegistry")
        return len(workflows_to_clean)
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """获取工作流执行的整体指标
        Get overall metrics for workflow execution
        
        Returns:
            dict: 工作流执行指标 / Workflow execution metrics
        """
        metrics = {
            'total_workflows': 0,
            'active_workflows': 0,
            'completed_workflows': 0,
            'failed_workflows': 0,
            'cancelled_workflows': 0,
            'average_execution_time': 0,
            'success_rate': 0
        }
        
        execution_times = []
        
        with self.workflow_lock:
            metrics['total_workflows'] = len(self.active_workflows)
            
            for workflow_info in self.active_workflows.values():
                if workflow_info.get('status') == 'active':
                    metrics['active_workflows'] += 1
                elif workflow_info.get('status') == 'completed':
                    metrics['completed_workflows'] += 1
                    # 计算执行时间
                    if 'start_time' in workflow_info and 'end_time' in workflow_info:
                        execution_time = workflow_info['end_time'] - workflow_info['start_time']
                        execution_times.append(execution_time)
                elif workflow_info.get('status') == 'failed':
                    metrics['failed_workflows'] += 1
                elif workflow_info.get('status') == 'cancelled':
                    metrics['cancelled_workflows'] += 1
        
        # 计算平均执行时间
        if execution_times:
            metrics['average_execution_time'] = sum(execution_times) / len(execution_times)
        
        # 计算成功率（已完成的工作流占总工作流的比例，不包括活跃的）
        completed = metrics['completed_workflows']
        total_non_active = metrics['total_workflows'] - metrics['active_workflows']
        if total_non_active > 0:
            metrics['success_rate'] = completed / total_non_active
        
        return metrics
    
    def load_all_models(self, configs: Dict[str, Dict[str, Any]] = None):
        """加载所有模型
        Load all models
        
        Args:
            configs: 模型配置字典 / Models configuration dictionary
            
        Returns:
            list: 成功加载的模型ID列表 / List of successfully loaded model IDs
        """
        configs = configs or {}
        loaded_models = []
        
        for model_id in self.model_types:
            model = self.load_model(model_id, configs.get(model_id))
            if model:
                loaded_models.append(model_id)
        
        return loaded_models
    
    def load_external_model(self, model_id: str, api_config: Dict[str, Any]):
        """加载外部API模型
        Load external API model
        
        Args:
            model_id: 模型ID
            api_config: API配置，包含api_url、api_key、model_name等信息
            
        Returns:
            object: 模型实例或API客户端
        """
        try:
            # 创建外部API服务实例
            external_service = ExternalAPIService()
            
            # 验证API配置并规范化字段命名
            normalized_config = {}
            if 'api_url' in api_config:
                normalized_config['url'] = api_config['api_url']
            elif 'url' in api_config:
                normalized_config['url'] = api_config['url']
            else:
                error_handler.log_error(f"缺少必要的API配置项: api_url或url", "ModelRegistry")
                return None
                
            if 'api_key' in api_config:
                normalized_config['api_key'] = api_config['api_key']
            else:
                error_handler.log_error(f"缺少必要的API配置项: api_key", "ModelRegistry")
                return None
                
            if 'model_name' in api_config:
                normalized_config['model_name'] = api_config['model_name']
            else:
                normalized_config['model_name'] = model_id
                
            # 添加额外的配置字段
            if 'source' in api_config:
                normalized_config['source'] = api_config['source']
            else:
                normalized_config['source'] = 'external'
                
            if 'endpoint' in api_config:
                normalized_config['endpoint'] = api_config['endpoint']
                
            if 'provider' in api_config:
                normalized_config['provider'] = api_config['provider']
            else:
                # 自动检测提供商
                normalized_config['provider'] = external_service.detect_provider(normalized_config['url'])
            
            # 初始化API服务
            init_result = external_service.initialize_api_service(
                normalized_config['provider'],
                normalized_config
            )
            
            if not init_result:
                error_handler.log_error(f"初始化外部API服务失败: {model_id}", "ModelRegistry")
                return None
            
            # 测试连接
            test_result = external_service.test_connection()
            if not test_result.get('success', False):
                error_handler.log_error(f"无法连接到外部模型: {model_id}, 错误: {test_result.get('error', '未知错误')}", "ModelRegistry")
                return None
                
            # 注册外部服务实例
            self.models[model_id] = external_service
            self.model_configs[model_id] = normalized_config
            
            error_handler.log_info(f"成功加载并连接外部模型: {model_id} (提供商: {normalized_config['provider']})", "ModelRegistry")
            return external_service
            
        except Exception as e:
            error_handler.handle_error(e, "ModelRegistry", f"加载外部模型 {model_id} 失败")
            return None
    
    def switch_model_source(self, model_id: str, source: str, config: Dict[str, Any] = None):
        """切换模型来源（本地或外部）
        Switch model source (local or external)
        
        Args:
            model_id: 模型ID
            source: 来源类型 ('local' 或 'external')
            config: 配置信息（对于外部模型需要api_config）
            
        Returns:
            bool: 切换是否成功
        """
        try:
            # 卸载当前模型
            self.unload_model(model_id)
            
            if source == 'local':
                # 加载本地模型
                return self.load_model(model_id, config) is not None
            elif source == 'external':
                # 加载外部模型
                return self.load_external_model(model_id, config) is not None
            else:
                error_handler.log_warning(f"未知的模型来源: {source}", "ModelRegistry")
                return False
                
        except Exception as e:
            error_handler.handle_error(e, "ModelRegistry", f"切换模型来源失败: {model_id}")
            return False
    
    def update_model_config(self, model_id: str, config: Dict[str, Any]):
        """更新模型配置并重启模型
        Update model configuration and restart the model
        
        Args:
            model_id: 模型ID / Model ID
            config: 新的配置字典 / New configuration dictionary
            
        Returns:
            object: 更新后的模型实例或None / Updated model instance or None
        """
        if model_id in self.models:
            # 合并新配置和旧配置
            # Merge new config with old config
            updated_config = {**self.model_configs.get(model_id, {}), **config}
            # 卸载旧模型
            # Unload old model
            self.unload_model(model_id)
            # 重新加载模型
            # Reload model
            return self.load_model(model_id, updated_config)
        return None
    
    # 新增：注册模型依赖关系
    
    def register_model_dependency(self, dependent_model_id, required_model_id):
        """注册模型间的依赖关系"""
        if dependent_model_id not in self.model_dependencies:
            self.model_dependencies[dependent_model_id] = []
        
        if required_model_id not in self.model_dependencies[dependent_model_id]:
            self.model_dependencies[dependent_model_id].append(required_model_id)
        
        error_handler.log_info(f"已注册模型依赖: {dependent_model_id} -> {required_model_id}", "ModelRegistry")
    
    # 新增：获取模型依赖树
    
    def get_model_dependencies(self, model_id):
        """获取指定模型的依赖树
        Get dependency tree for specified model
        
        Args:
            model_id: 模型ID / Model ID
            
        Returns:
            list: 依赖模型ID列表 / List of dependent model IDs
        """
        dependencies = self.model_dependencies.get(model_id, [])
        full_dependencies = set(dependencies)
        
        # 递归获取所有间接依赖
        # Recursively get all indirect dependencies
        for dep in dependencies:
            full_dependencies.update(self.get_model_dependencies(dep))
        
        return list(full_dependencies)
    
    # 新增：更新模型性能指标
    
    def update_model_performance(self, model_id, metrics):
        """
        更新模型性能指标
        Update model performance metrics
        """
        if model_id not in self.performance_metrics:
            self.performance_metrics[model_id] = {}
        
        self.performance_metrics[model_id].update({
            **metrics,
            'timestamp': time.time()
        })
        
        # 如果知识库模型已加载，使用它分析性能数据并提供优化建议
        # If knowledge model is loaded, use it to analyze performance data and provide optimization suggestions
        if "knowledge" in self.models:
            knowledge_model = self.get_model("knowledge")
            optimization_suggestions = knowledge_model.analyze_performance(
                model_id, 
                metrics
            )
            
            # 记录优化建议到日志
            # Log optimization suggestions
            if optimization_suggestions:
                error_handler.log_info(
                    f"知识库模型为 {model_id} 提供的优化建议: {optimization_suggestions}",
                    "ModelRegistry"
                )
                
                # 将建议添加到性能指标中
                # Add suggestions to performance metrics
                self.performance_metrics[model_id]['optimization_suggestions'] = (
                    optimization_suggestions
                )
    
    # 新增：获取最优模型（基于性能）
    
    def get_best_model_for_task(self, task_type, criteria=None):
        """根据任务类型和标准推荐最优模型
        Recommend the best model based on task type and criteria
        
        Args:
            task_type: 任务类型 / Task type
            criteria: 评估标准 / Evaluation criteria
            
        Returns:
            str: 最优模型ID或None / Best model ID or None
        """
        # 增强知识库模型的学习能力 - 添加主动知识获取
        # Enhance knowledge model learning capability - add active knowledge acquisition
        if "knowledge" in self.models:
            if task_type == "knowledge_enhancement":
                self.models["knowledge"].active_learning()
            else:
                # 知识库模型为其他模型提供实时辅助
                # Knowledge model provides real-time assistance to other models
                self.models["knowledge"].assist_other_models(task_type)
        
        # 多维度评估模型性能 - 添加协作能力评估
        # Multi-dimensional model performance evaluation - add collaboration capability
        eligible_models = []
        for model_id, metrics in self.performance_metrics.items():
            if metrics.get('task_type') == task_type:
                # 综合评分 = 成功率(40%) + 效率(25%) + 资源消耗(15%) + 协作能力(20%)
                # Comprehensive score = success_rate(40%) + efficiency(25%) + resource_usage(15%) + collaboration(20%)
                success_rate = metrics.get('success_rate', 0)
                efficiency = metrics.get('efficiency', 0)
                resource_usage = 1 - min(1, metrics.get('resource_usage', 0)/100)
                collaboration = metrics.get('collaboration_score', 0.5)  # 默认0.5
                score = (success_rate * 0.4) + (efficiency * 0.25) + (resource_usage * 0.15) + (collaboration * 0.2)
                eligible_models.append((model_id, score))
        
        if not eligible_models:
            # 尝试知识库模型作为默认解决方案
            # Try knowledge model as default solution
            if "knowledge" in self.models:
                return "knowledge"
            return None
        
        # 按综合评分排序并返回最优模型
        # Sort by comprehensive score and return the best model
        eligible_models.sort(key=lambda x: x[1], reverse=True)
        return eligible_models[0][0]
        
    
    def collaborative_execution(self, task_description, workflow_id=None, conflict_strategy=None, use_parallel_execution=True, context=None):
        """增强版模型协作执行方法，支持并行执行和高级结果融合
        Enhanced model collaborative execution method with parallel execution and advanced result fusion
        
        Args:
            task_description: 任务描述 / Task description
            workflow_id: 工作流ID（用于跟踪和管理长期任务） / Workflow ID (for tracking and managing long-term tasks)
            conflict_strategy: 冲突解决策略 ('majority_vote', 'expert_model', 'hierarchical') / Conflict resolution strategy
            use_parallel_execution: 是否使用并行执行 / Whether to use parallel execution
            context: 上下文信息 / Context information
            
        Returns:
            dict: 包含各模型输出、最终结果和执行元数据 / Contains outputs from each model, final result, and execution metadata
        """
        result = {
            'success': False,
            'data': None,
            'execution_time': 0,
            'details': {}
        }
        
        start_time = time.time()
        
        try:
            # 如果没有指定工作流ID，生成一个
            if workflow_id is None:
                workflow_id = f"workflow_{time.time()}_{hash(str(task_description)) % 10000}"
            
            # 如果没有指定冲突解决策略，使用默认策略
            if conflict_strategy is None:
                conflict_strategy = self.default_conflict_strategy
            
            # 保存工作流信息
            with self.workflow_lock:
                self.active_workflows[workflow_id] = {
                    'task_description': task_description,
                    'start_time': start_time,
                    'status': 'active',
                    'context': context or {},
                    'parallel_execution': use_parallel_execution,
                    'conflict_strategy': conflict_strategy,
                    'models_count': 0,
                    'completed_models': 0
                }
            
            error_handler.log_info(f"开始执行工作流: {workflow_id}", "ModelRegistry")
            
            # 定期清理过期工作流
            try:
                if random.random() < 0.1:  # 10%的概率触发清理
                    cleaned_count = self.clean_workflows()
                    if cleaned_count > 0:
                        error_handler.log_info(f"清理了 {cleaned_count} 个过期工作流", "ModelRegistry")
            except Exception as cleanup_error:
                error_handler.handle_error(cleanup_error, "ModelRegistry", "清理工作流时出错")
            
            # 使用管理模型分解任务
            manager = self.get_model("manager")
            if not manager:
                error_handler.log_error("管理模型未加载", "ModelRegistry")
                with self.workflow_lock:
                    self.active_workflows[workflow_id]['status'] = 'failed'
                    self.active_workflows[workflow_id]['error'] = 'Manager model not loaded'
                result['error'] = 'Manager model not loaded'
                return result
                
            # 任务分解和分配
            subtasks = manager.decompose_task(task_description, context=context)
            results = {}
            execution_times = {}
            
            # 验证模型列表
            valid_subtasks = {}
            for model_id, task in subtasks.items():
                if self.get_model(model_id):
                    valid_subtasks[model_id] = task
                else:
                    error_handler.log_warning(f"模型 {model_id} 不存在或未加载，将被跳过", "ModelRegistry")
            
            if not valid_subtasks:
                raise ValueError("没有有效的模型可用于协作执行")
            
            # 更新工作流中的模型数量
            with self.workflow_lock:
                self.active_workflows[workflow_id]['models_count'] = len(valid_subtasks)
            
            # 增强子任务创建 - 添加模型特定信息
            enhanced_subtasks = {}
            for model_id, task in valid_subtasks.items():
                model = self.get_model(model_id)
                if model:
                    # 为每个模型创建适合其专长的子任务
                    subtask = task.copy() if isinstance(task, dict) else {"task_data": task}
                    # 添加模型特定信息
                    subtask['model_specific'] = True
                    subtask['target_model'] = model_id
                    subtask['workflow_id'] = workflow_id
                    # 如果模型有特定专长，添加到任务中
                    if hasattr(model, 'specialties'):
                        subtask['model_specialties'] = model.specialties
                    # 添加模型性能数据
                    subtask['model_performance'] = self.performance_metrics.get(model_id, {})
                    enhanced_subtasks[model_id] = subtask
            
            # 自适应任务分配 - 基于模型性能和当前负载
            optimized_subtasks = self._optimize_task_allocation(enhanced_subtasks)
            
            # 获取上下文信息（如果可用）
            if self.context_manager:
                context = self.context_manager['get_context'](workflow_id) if isinstance(self.context_manager, dict) else self.context_manager.get_context(workflow_id)
            
            # 获取实时指标并将其添加到上下文中
            realtime_metrics = self.get_realtime_metrics()
            context_with_metrics = context.copy() if context else {}
            context_with_metrics['realtime_metrics'] = realtime_metrics
            
            # 执行子任务
            if use_parallel_execution and self.executor:
                # 并行执行子任务
                results, execution_times = self._execute_subtasks_parallel(optimized_subtasks, context_with_metrics)
            else:
                # 串行执行子任务
                results, execution_times = self._execute_subtasks_serial(optimized_subtasks, context_with_metrics)
            
            # 实时更新知识库
            if "knowledge" in self.models:
                for model_id, task in optimized_subtasks.items():
                    if model_id in results:
                        try:
                            # 使用上下文感知的知识更新
                            self.models["knowledge"].update_knowledge(task, results[model_id], context=context_with_metrics)
                        except Exception as e:
                            error_handler.handle_error(e, "ModelRegistry", f"更新知识库失败: {model_id}")
            
            # 更新上下文
            if context and context.get('workflow_id'):
                context_updates = {}
                for model_id, result_data in results.items():
                    if result_data is not None:
                        # 添加上下文更新，包含每个模型的结果
                        context_updates[f'result_{model_id}'] = result_data
                        context_updates[f'execution_time_{model_id}'] = execution_times.get(model_id, 0)
                self._simple_update_context(context['workflow_id'], context_updates)
            
            # 高级结果融合
            final_result = self._fuse_results(results, conflict_strategy)
            
            # 更新工作流状态
            with self.workflow_lock:
                self.active_workflows[workflow_id].update({
                    'end_time': time.time(),
                    'status': 'completed',
                    'results': results,
                    'final_result': final_result,
                    'execution_times': execution_times,
                    'completed_models': len(results)
                })
            
            # 记录性能指标
            self._update_collaboration_metrics(results, execution_times)
            
            # 如果结果表明需要，启动知识迁移过程
            self._trigger_knowledge_transfer(results)
            
            # 构建最终结果
            result['success'] = True
            result['data'] = final_result
            result['execution_time'] = time.time() - start_time
            result['details'] = {
                "workflow_id": workflow_id,
                "subtask_results": results,
                "execution_times": execution_times,
                "context": context,
                "timestamp": time.time(),
                "conflict_strategy": conflict_strategy,
                "parallel_execution": use_parallel_execution,
                "models_used": list(optimized_subtasks.keys()),
                "workflow_status": self.get_workflow_status(workflow_id)
            }
            
            error_handler.log_info(f"工作流 {workflow_id} 执行完成", "ModelRegistry")
        except Exception as e:
            error_handler.handle_error(e, "ModelRegistry", f"协作执行任务失败: {str(e)}")
            result['error'] = str(e)
            # 更新工作流状态为失败
            if workflow_id:
                with self.workflow_lock:
                    if workflow_id in self.active_workflows:
                        self.active_workflows[workflow_id]['status'] = 'failed'
                        self.active_workflows[workflow_id]['error'] = str(e)
                        self.active_workflows[workflow_id]['end_time'] = time.time()
        
        return result
        
    
    def get_realtime_metrics(self):
        """获取所有模型的实时监控数据 (用于仪表盘)
        Get real-time monitoring data for all models (for dashboard)
        
        Returns:
            dict: 模型ID到监控数据的映射 / Mapping of model IDs to monitoring data
        """
        metrics = {}
        for model_id, model in self.models.items():
            if hasattr(model, 'get_realtime_metrics'):
                model_metrics = model.get_realtime_metrics()
                # 添加训练状态信息 | Add training status information
                model_metrics['training_status'] = self.training_status.get(model_id, 'idle')
                metrics[model_id] = model_metrics
            else:
                # 基础监控数据 - 添加训练状态
                # Basic monitoring data - add training status
                metrics[model_id] = {
                    "status": "active",
                    "training_status": self.training_status.get(model_id, 'idle'),
                    "last_activity": time.time(),
                    "performance": self.performance_metrics.get(model_id, {})
                }
            
            # 添加协作能力指标 | Add collaboration metrics
            if 'collaboration_score' not in metrics[model_id]['performance']:
                metrics[model_id]['performance']['collaboration_score'] = 0.7  # 默认值
                
        return metrics

    def get_model_status(self, model_id: str):
        """获取模型状态（增强版）
        Get model status (enhanced version)
        
        Args:
            model_id: 模型ID / Model ID
            
        Returns:
            dict: 模型状态字典，包含详细的状态信息和健康评分 / Model status dictionary with detailed status information and health score
        """
        status = {
            "model_id": model_id,
            "timestamp": time.time(),
            "last_updated": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "status": "unknown",
            "health_score": 0,
            "health_status": "unknown",
            "details": {}
        }
        
        # 检查模型是否已加载
        model = self.get_model(model_id)
        if not model:
            status["status"] = "not_loaded"
            status["health_score"] = 0
            status["health_status"] = "critical"
            status["details"]["reason"] = "模型未加载或不存在"
            return status
        
        try:
            # 检查是否在训练中
            training_status = self.training_status.get(model_id, 'idle')
            status["details"]["training_status"] = training_status
            
            # 获取性能指标
            performance_metrics = self.performance_metrics.get(model_id, {})
            status["details"]["performance"] = performance_metrics
            
            # 尝试获取模型状态（如果模型支持）
            if hasattr(model, 'get_status'):
                try:
                    model_status = model.get_status()
                    # 合并模型返回的状态
                    status.update(model_status)
                    # 确保基本字段存在
                    status["model_id"] = model_id
                    status["status"] = model_status.get("status", "loaded")
                except Exception as e:
                    error_handler.log_warning(f"模型 {model_id} 的 get_status 方法调用失败: {str(e)}", "ModelRegistry")
                    status["status"] = "loaded_with_warnings"
                    status["details"]["status_method_error"] = str(e)
            else:
                # 对于没有get_status方法的模型，返回基础状态
                status["status"] = "loaded"
                status["details"]["is_initialized"] = hasattr(model, 'is_initialized') and model.is_initialized
                status["details"]["is_training"] = hasattr(model, 'is_training') and model.is_training
            
            # 计算健康评分
            health_score = 100  # 满分100分
            
            # 检查初始化状态
            if not status["details"].get("is_initialized", True):
                health_score -= 40
            
            # 检查训练状态
            if status["details"]["training_status"] == 'failed':
                health_score -= 50
            elif status["details"]["training_status"] == 'idle':
                health_score -= 10  # 空闲状态扣分较少
            
            # 基于性能指标扣分
            if performance_metrics:
                # 成功率低于70%扣分
                success_rate = performance_metrics.get('success_rate', 1)
                if success_rate < 0.7:
                    health_score -= (0.7 - success_rate) * 100
                
                # 错误率高于10%扣分
                error_rate = performance_metrics.get('error_rate', 0)
                health_score -= min(error_rate * 500, 30)  # 最多扣30分
                
                # 响应时间过长扣分
                avg_response_time = performance_metrics.get('avg_response_time', 0)
                if avg_response_time > 5:  # 响应时间超过5秒
                    health_score -= min((avg_response_time - 5) * 5, 20)  # 最多扣20分
            
            # 限制健康评分范围
            health_score = max(0, min(100, health_score))
            status["health_score"] = round(health_score, 2)
            
            # 确定健康状态
            if health_score >= 85:
                status["health_status"] = "excellent"
            elif health_score >= 70:
                status["health_status"] = "good"
            elif health_score >= 50:
                status["health_status"] = "fair"
            elif health_score >= 30:
                status["health_status"] = "poor"
            else:
                status["health_status"] = "critical"
            
            # 添加协作能力指标
            collaboration_score = performance_metrics.get('collaboration_score', 0.7)
            status["details"]["collaboration_score"] = collaboration_score
            
            # 添加模型类型信息
            if model_id in self.model_types:
                status["details"]["model_type"] = self.model_types[model_id]
            
            # 添加资源使用信息（如果可用）
            if hasattr(model, 'resource_usage'):
                status["details"]["resource_usage"] = model.resource_usage
            
            return status
        except Exception as e:
            error_handler.handle_error(e, "ModelRegistry", f"获取模型 {model_id} 状态失败")
            status["status"] = "error"
            status["error"] = str(e)
            status["health_score"] = 0
            status["health_status"] = "critical"
            return status

    def get_model_health_summary(self) -> Dict[str, Any]:
        """获取所有模型的健康状况摘要
        Get health summary of all models
        
        Returns:
            dict: 模型健康摘要信息 / Model health summary information
        """
        all_statuses = self.get_all_models_status()
        summary = {
            "total_models": len(all_statuses),
            "status_counts": {
                "loaded": 0,
                "not_loaded": 0,
                "error": 0,
                "training": 0,
                "idle": 0
            },
            "health_counts": {
                "excellent": 0,
                "good": 0,
                "fair": 0,
                "poor": 0,
                "critical": 0
            },
            "average_health_score": 0,
            "problematic_models": [],
            "training_models": [],
            "timestamp": time.time()
        }
        
        health_scores = []
        
        for model_id, status in all_statuses.items():
            # 更新状态计数
            if status.get("status") == "not_loaded":
                summary["status_counts"]["not_loaded"] += 1
            elif status.get("status") == "error":
                summary["status_counts"]["error"] += 1
            else:
                summary["status_counts"]["loaded"] += 1
            
            # 更新训练状态计数
            training_status = status.get("details", {}).get("training_status", "idle")
            if training_status in ["training", "finetuning"]:
                summary["status_counts"]["training"] += 1
                summary["training_models"].append(model_id)
            else:
                summary["status_counts"]["idle"] += 1
            
            # 更新健康状态计数
            health_status = status.get("health_status", "unknown")
            if health_status in summary["health_counts"]:
                summary["health_counts"][health_status] += 1
            
            # 收集健康评分
            health_score = status.get("health_score", 0)
            health_scores.append(health_score)
            
            # 记录有问题的模型
            if health_score < 50:
                summary["problematic_models"].append({
                    "model_id": model_id,
                    "health_score": health_score,
                    "health_status": health_status,
                    "status": status.get("status", "unknown")
                })
        
        # 计算平均健康评分
        if health_scores:
            summary["average_health_score"] = round(sum(health_scores) / len(health_scores), 2)
        
        # 按健康评分排序问题模型
        summary["problematic_models"].sort(key=lambda x: x["health_score"])
        
        return summary

    def get_all_models_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模型的状态 (增强版)
        Get status of all models (enhanced version)
        
        Returns:
            dict: 所有模型的详细状态字典 / Detailed dictionary of all models' status
        """
        statuses = {}
        
        # 首先检查是否有模型加载
        if not self.models and not self.model_types:
            error_handler.log_info("没有已注册或加载的模型", "ModelRegistry")
            return statuses
        
        # 同时检查已加载的模型和已注册但未加载的模型
        all_model_ids = set(self.models.keys()).union(self.model_types.keys())
        
        for model_id in all_model_ids:
            try:
                # 使用增强版get_model_status获取详细状态
                status = self.get_model_status(model_id)
                
                # 对于已注册但未加载的模型，添加额外信息
                if model_id in self.model_types and model_id not in self.models:
                    status["is_registered"] = True
                    status["details"]["model_type"] = self.model_types[model_id]
                    status["details"]["is_loaded"] = False
                else:
                    status["is_registered"] = model_id in self.model_types
                    status["details"]["is_loaded"] = model_id in self.models
                
                # 添加上次更新时间（更友好的格式）
                status["last_updated_friendly"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(status["timestamp"]))
                
                # 添加资源使用信息（如果有）
                if hasattr(self, 'resource_monitor') and self.resource_monitor:
                    try:
                        if isinstance(self.resource_monitor, dict):
                            resource_usage = self.resource_monitor.get(model_id, {})
                        else:
                            resource_usage = self.resource_monitor.get_resource_usage(model_id)
                        
                        if resource_usage:
                            status["details"]["resource_usage"] = resource_usage
                    except Exception as e:
                        error_handler.log_warning(f"获取模型 {model_id} 资源使用情况失败: {str(e)}", "ModelRegistry")
                
                # 添加协作能力指标（如果没有的话）
                if "collaboration_score" not in status["details"]:
                    performance = self.performance_metrics.get(model_id, {})
                    status["details"]["collaboration_score"] = performance.get('collaboration_score', 0.7)
                
                statuses[model_id] = status
            except Exception as e:
                error_handler.handle_error(e, "ModelRegistry", f"获取模型 {model_id} 状态失败")
                statuses[model_id] = {
                    "model_id": model_id,
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time(),
                    "health_score": 0,
                    "health_status": "critical"
                }
        
        return statuses

    def get_all_model_ids(self):
        """获取所有已注册模型的ID列表
        Get list of all registered model IDs
        
        Returns:
            list: 模型ID列表 / List of model IDs
        """
        return list(self.models.keys())

    

# 创建全局模型注册表实例（延迟初始化）
# Create global model registry instance (lazy initialization)
_model_registry_instance = None

# 全局模型注册表实例，供直接导入使用
# Global model registry instance for direct import
model_registry = None

def get_model_registry():
    """获取模型注册表实例（懒加载）
    Get model registry instance (lazy loading)
    
    Returns:
        ModelRegistry: 模型注册表实例
    """
    global _model_registry_instance
    if _model_registry_instance is None:
        try:
            _model_registry_instance = ModelRegistry()
            print("ModelRegistry initialized successfully")
        except Exception as e:
            print(f"Error initializing ModelRegistry: {e}")
            # 创建简化版本的注册表作为备用
            class FallbackModelRegistry:
                def __init__(self):
                    self.models = {}
                    self.model_configs = {}
                def get_model(self, model_id):
                    return self.models.get(model_id)
                def get_model_status(self, model_id):
                    return {"status": "fallback", "model_id": model_id}
            _model_registry_instance = FallbackModelRegistry()
            print("Using fallback ModelRegistry")
    return _model_registry_instance

# 模块级函数：提供对全局实例方法的直接访问
# Module-level functions: provide direct access to global instance methods
def get_model(model_id: str):
    """获取已注册的模型实例
    Get registered model instance
    
    Args:
        model_id: 模型ID / Model ID
        
    Returns:
        object: 模型实例或None / Model instance or None
    """
    return get_model_registry().get_model(model_id)

# 获取模型状态
def get_model_status(model_id: str):
    """获取模型状态
    Get model status
    
    Args:
        model_id: 模型ID / Model ID
        
    Returns:
        dict: 模型状态字典 / Model status dictionary
    """
    return get_model_registry().get_model_status(model_id)

# 初始化模型注册表（兼容性函数）
# Initialize model registry (compatibility function)
def initialize():
    """初始化模型注册表
    Initialize model registry
    
    Returns:
        ModelRegistry: 模型注册表实例
    """
    return get_model_registry()

# ====== 新增训练协调方法 ====== | ====== Added training coordination methods ======

"""
init_joint_training_coordinator函数 - 中文函数描述
init_joint_training_coordinator Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def init_joint_training_coordinator():
    """初始化联合训练协调器
    Initialize joint training coordinator
    """
    from core.training.joint_training_coordinator import JointTrainingCoordinator  # type: ignore
    model_registry.joint_training_coordinator = JointTrainingCoordinator()
    error_handler.log_info("联合训练协调器已初始化", "ModelRegistry")


"""
start_training函数 - 中文函数描述
start_training Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def start_training(model_ids, training_mode='individual', config=None):
    """启动模型训练
    Start model training
    
    Args:
        model_ids: 要训练的模型ID列表 | List of model IDs to train
        training_mode: 训练模式 (individual/joint) | Training mode
        config: 训练配置 | Training configuration
        
    Returns:
        dict: 训练结果 | Training results
    """
    if not model_ids:
        error_handler.log_error("未指定训练模型", "start_training")
        return {"status": "error", "message": "未指定训练模型"}
    
    # 确保model_ids是列表格式
    if isinstance(model_ids, str):
        model_ids = [model_ids]
    
    # 检查模型是否存在
    for model_id in model_ids:
        if not model_registry.get_model(model_id):
            error_handler.log_error(f"模型{model_id}不存在", "start_training")
            return {"status": "error", "message": f"模型{model_id}不存在"}
    
    # 获取模型健康状态，确保模型适合训练
    for model_id in model_ids:
        model_status = model_registry.get_model_status(model_id)
        if model_status.get("health_status") == "critical":
            error_handler.log_warning(f"模型{model_id}健康状态为critical，不建议训练", "start_training")
    
    # 初始化训练协调器（如果需要） | Initialize training coordinator if needed
    if training_mode == 'joint' and model_registry.joint_training_coordinator is None:
        init_joint_training_coordinator()
    
    # 设置训练状态 | Set training status
    for model_id in model_ids:
        model_registry.training_status[model_id] = 'preparing'
    
    # 记录开始训练时间
    start_time = time.time()
    
    try:
        if training_mode == 'joint':
            # 联合训练 | Joint training
            # 首先创建训练任务
            from core.training.joint_training_coordinator import TrainingTask  # type: ignore
            training_tasks = []
            for model_id in model_ids:
                training_tasks.append(TrainingTask(
                    model_id=model_id,
                    training_data=config.get('training_data', {}),
                    epochs=config.get('epochs', 10),
                    batch_size=config.get('batch_size', 32)
                ))
            
            # 调度训练任务
            schedule_result = model_registry.joint_training_coordinator.schedule_training(training_tasks)
            
            # 执行训练
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(model_registry.joint_training_coordinator.execute_training())
            loop.close()
            
        else:
            # 单独训练 | Individual training
            results = {}
            for model_id in model_ids:
                model = model_registry.get_model(model_id)
                if model and hasattr(model, 'train'):
                    model_registry.training_status[model_id] = 'training'
                    results[model_id] = model.train(config)
                    model_registry.training_status[model_id] = 'completed'
                else:
                    results[model_id] = {"status": "error", "message": f"模型{model_id}不支持训练"}
                    model_registry.training_status[model_id] = 'error'
        
        # 记录训练历史
        duration = time.time() - start_time
        
        # 初始化训练历史记录（如果不存在）
        if not hasattr(model_registry, 'training_history'):
            model_registry.training_history = {}
        
        for model_id in model_ids:
            if model_id not in model_registry.training_history:
                model_registry.training_history[model_id] = []
            
            # 获取模型状态
            model_status = model_registry.get_model_status(model_id)
            
            # 记录训练历史
            training_record = {
                'timestamp': int(time.time()),
                'duration': duration,
                'training_mode': training_mode,
                'config': config,
                'status': model_registry.training_status.get(model_id, 'unknown'),
                'results': results.get(model_id, {}),
                'model_health_before': model_status
            }
            
            model_registry.training_history[model_id].append(training_record)
            
            # 限制历史记录数量，保留最近100条
            if len(model_registry.training_history[model_id]) > 100:
                model_registry.training_history[model_id].pop(0)
        
        error_handler.log_info(f"模型训练完成：{model_ids}", "start_training")
        return results
    except Exception as e:
        error_handler.handle_error(e, "Training", "训练过程中出错")
        for model_id in model_ids:
            model_registry.training_status[model_id] = 'error'
        
        # 即使出错也要记录训练历史
        duration = time.time() - start_time
        
        # 初始化训练历史记录（如果不存在）
        if not hasattr(model_registry, 'training_history'):
            model_registry.training_history = {}
        
        for model_id in model_ids:
            if model_id not in model_registry.training_history:
                model_registry.training_history[model_id] = []
            
            training_record = {
                'timestamp': int(time.time()),
                'duration': duration,
                'training_mode': training_mode,
                'config': config,
                'status': 'error',
                'error_message': str(e)
            }
            
            model_registry.training_history[model_id].append(training_record)
            
            # 限制历史记录数量，保留最近100条
            if len(model_registry.training_history[model_id]) > 100:
                model_registry.training_history[model_id].pop(0)
        
        return {"status": "error", "message": str(e)}

def get_model_training_history(model_id):
    """获取模型的训练历史记录
    Get model's training history
    
    Args:
        model_id: 模型ID
        
    Returns:
        list: 训练历史记录列表
    """
    if model_id not in model_registry.models:
        error_handler.log_error(f"模型{model_id}不存在", "get_model_training_history")
        return {"status": "error", "message": f"模型{model_id}不存在"}
    
    if model_id not in model_registry.training_history:
        model_registry.training_history[model_id] = []
    
    return model_registry.training_history[model_id]

def clear_model_training_history(model_id, keep_last_n=None):
    """清理模型的训练历史记录
    Clear model's training history
    
    Args:
        model_id: 模型ID
        keep_last_n: 保留最近的n条记录，如果为None则清空所有记录
        
    Returns:
        dict: 操作结果
    """
    if model_id not in model_registry.models:
        error_handler.log_error(f"模型{model_id}不存在", "clear_model_training_history")
        return {"status": "error", "message": f"模型{model_id}不存在"}
    
    if model_id not in model_registry.training_history:
        return {"status": "success", "message": "模型没有训练历史记录"}
    
    original_count = len(model_registry.training_history[model_id])
    
    if keep_last_n is not None and isinstance(keep_last_n, int) and keep_last_n > 0:
        # 只保留最近的n条记录
        if len(model_registry.training_history[model_id]) > keep_last_n:
            model_registry.training_history[model_id] = model_registry.training_history[model_id][-keep_last_n:]
            error_handler.log_info(f"已清理模型{model_id}的训练历史，保留最近{keep_last_n}条记录", "clear_model_training_history")
            return {"status": "success", "message": f"已清理模型{model_id}的训练历史，保留最近{keep_last_n}条记录", 
                    "original_count": original_count, "remaining_count": len(model_registry.training_history[model_id])}
        else:
            return {"status": "success", "message": f"模型{model_id}的训练历史记录数量已少于或等于{keep_last_n}，无需清理"}
    else:
        # 清空所有记录
        model_registry.training_history[model_id] = []
        error_handler.log_info(f"已清空模型{model_id}的所有训练历史记录", "clear_model_training_history")
        return {"status": "success", "message": f"已清空模型{model_id}的所有训练历史记录", 
                "original_count": original_count, "remaining_count": 0}

def get_all_training_stats():
    """获取所有模型的训练统计信息
    Get training statistics for all models
    
    Returns:
        dict: 训练统计信息字典
    """
    stats = {
        'total_trainings': 0,
        'models_with_history': 0,
        'success_rate': 0.0,
        'avg_training_duration': 0.0,
        'training_modes': {},
        'models': {}
    }
    
    total_success_count = 0
    total_duration = 0
    total_count = 0
    
    for model_id, history in model_registry.training_history.items():
        if not history:
            continue
        
        stats['models_with_history'] += 1
        model_stats = {
            'training_count': len(history),
            'success_count': 0,
            'failure_count': 0,
            'avg_duration': 0.0,
            'last_training_time': 0,
            'last_training_status': 'unknown'
        }
        
        model_duration = 0
        
        for record in history:
            total_count += 1
            model_duration += record.get('duration', 0)
            
            # 统计训练模式
            mode = record.get('training_mode', 'unknown')
            if mode not in stats['training_modes']:
                stats['training_modes'][mode] = 0
            stats['training_modes'][mode] += 1
            
            # 统计成功/失败次数
            status = record.get('status', 'unknown')
            if status == 'completed' or (isinstance(status, dict) and status.get('status') == 'completed'):
                model_stats['success_count'] += 1
                total_success_count += 1
            elif status == 'error' or (isinstance(status, dict) and status.get('status') == 'error'):
                model_stats['failure_count'] += 1
            
            # 更新最后训练时间和状态
            if record.get('timestamp', 0) > model_stats['last_training_time']:
                model_stats['last_training_time'] = record.get('timestamp', 0)
                model_stats['last_training_status'] = status
        
        # 计算模型平均训练时间
        if model_stats['training_count'] > 0:
            model_stats['avg_duration'] = model_duration / model_stats['training_count']
        
        stats['models'][model_id] = model_stats
        total_duration += model_duration
    
    # 计算总体统计
    stats['total_trainings'] = total_count
    if total_count > 0:
        stats['success_rate'] = total_success_count / total_count
        stats['avg_training_duration'] = total_duration / total_count
    
    return stats

# 独立的load_all_models函数
def load_all_models(model_registry=None, configs=None):
    """加载所有模型
    Load all models
    
    Args:
        model_registry: 模型注册表实例，默认为全局实例
        configs: 模型配置字典
        
    Returns:
        list: 成功加载的模型ID列表
    """
    if model_registry is None:
        model_registry = globals().get('model_registry')
        if model_registry is None:
            model_registry = ModelRegistry()
            error_handler.log_info("创建了新的模型注册表实例", "load_all_models")
    
    configs = configs or {}
    return model_registry.load_all_models(configs)

# 在ModelRegistry类中添加initialize方法
# 这是为了兼容之前的代码
# Add initialize method to ModelRegistry class for backward compatibility
ModelRegistry.initialize = lambda self: None

# 初始化全局模型注册表实例
# Initialize global model registry instance
try:
    model_registry = get_model_registry()
except Exception as e:
    # 如果初始化失败，创建一个空的占位符
    # Create an empty placeholder if initialization fails
    class PlaceholderModelRegistry:
        def __getattr__(self, name):
            def placeholder_method(*args, **kwargs):
                print(f"Warning: model_registry.{name} called but not initialized")
                return None
            return placeholder_method
    model_registry = PlaceholderModelRegistry()
    print(f"Warning: Failed to initialize model_registry: {e}")

# 全局函数：切换模型到外部API模式
def switch_model_to_external(model_id: str, api_config: Dict[str, Any]) -> Dict[str, Any]:
    """Switch specified model to external API mode
    
    Args:
        model_id: Model ID
        api_config: API configuration including url, api_key, model_name, etc.
        
    Returns:
        Dict[str, Any]: Switch result
    """
    return model_registry.switch_model_to_external(model_id, api_config)

# 全局函数：切换模型回本地模式
def switch_model_to_local(model_id: str) -> Dict[str, Any]:
    """Switch specified model back to local mode
    
    Args:
        model_id: Model ID
        
    Returns:
        Dict[str, Any]: Switch result
    """
    return model_registry.switch_model_to_local(model_id)

# 全局函数：获取模型当前运行模式
def get_model_mode(model_id: str) -> str:
    """获取指定模型当前的运行模式
    Get current operation mode of specified model
    
    Args:
        model_id: 模型ID
        
    Returns:
        str: 'local' 或 'external'
    """
    return model_registry.get_model_mode(model_id)

# 全局函数：测试外部API连接
def test_external_api_connection(model_id: str, api_config: Dict[str, Any]) -> Dict[str, Any]:
    """测试外部API连接
    Test external API connection
    
    Args:
        model_id: 模型ID
        api_config: API配置信息
        
    Returns:
        Dict[str, Any]: 连接测试结果
    """
    try:
        # 创建ExternalAPIService实例进行连接测试
        external_service = ExternalAPIService()
        
        # 验证API配置并规范化字段命名
        normalized_config = {}
        if 'api_url' in api_config:
            normalized_config['url'] = api_config['api_url']
        elif 'url' in api_config:
            normalized_config['url'] = api_config['url']
        else:
            return {"status": "error", "message": "缺少必要的API配置项: api_url或url"}
            
        if 'api_key' in api_config:
            normalized_config['api_key'] = api_config['api_key']
        else:
            return {"status": "error", "message": "缺少必要的API配置项: api_key"}
            
        if 'model_name' in api_config:
            normalized_config['model_name'] = api_config['model_name']
        else:
            normalized_config['model_name'] = model_id
            
        # 添加额外的配置字段
        if 'source' in api_config:
            normalized_config['source'] = api_config['source']
        else:
            normalized_config['source'] = 'external'
            
        if 'endpoint' in api_config:
            normalized_config['endpoint'] = api_config['endpoint']
            
        if 'provider' in api_config:
            normalized_config['provider'] = api_config['provider']
        else:
            # 自动检测提供商
            normalized_config['provider'] = external_service.detect_provider(normalized_config['url'])
        
        # 初始化API服务
        init_result = external_service.initialize_api_service(
            normalized_config['provider'],
            normalized_config
        )
        
        if not init_result:
            return {"status": "error", "message": f"初始化外部API服务失败: {model_id}"}
        
        # 测试连接
        test_result = external_service.test_connection()
        
        if test_result.get('success', False):
            return {
                "status": "success", 
                "message": f"外部API连接成功 (提供商: {normalized_config['provider']})", 
                "api_status": test_result
            }
        else:
            return {
                "status": "error", 
                "message": f"外部API连接失败: {test_result.get('error', '未知错误')}",
                "details": test_result
            }
    except Exception as e:
        error_handler.handle_error(e, "ModelRegistry", f"测试外部API连接失败: {model_id}")
        return {"status": "error", "message": str(e)}

# 全局函数：获取所有模型的运行模式
def get_all_models_mode() -> Dict[str, str]:
    """获取所有模型的运行模式
    Get operation modes of all models
    
    Returns:
        Dict[str, str]: 模型ID到运行模式的映射
    """
    result = {}
    for model_id in model_registry.models:
        result[model_id] = get_model_mode(model_id)
    return result

# 全局函数：批量切换模型模式
def batch_switch_model_modes(mode_switches: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """批量切换多个模型的运行模式
    Batch switch operation modes of multiple models
    
    Args:
        mode_switches: 模型ID到模式配置的映射
                      例如: {"model_id1": {"mode": "external", "config": {...}}, "model_id2": {"mode": "local"}}
        
    Returns:
        Dict[str, Dict[str, Any]]: 每个模型的切换结果
    """
    results = {}
    
    for model_id, switch_config in mode_switches.items():
        if switch_config.get("mode") == "external" and "config" in switch_config:
            results[model_id] = switch_model_to_external(model_id, switch_config["config"])
        elif switch_config.get("mode") == "local":
            results[model_id] = switch_model_to_local(model_id)
        else:
            results[model_id] = {"status": "error", "message": "无效的模式配置"}
    
    return results
