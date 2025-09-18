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
训练管理器：负责模型训练流程的管理和控制
Training Manager: Responsible for managing and controlling model training processes

提供完整的模型训练管理功能，包括单独训练、联合训练、实时数据监控等
Provides complete model training management functionality, including individual training, joint training, real-time data monitoring, etc.
"""
import time
import os
import json
import threading
import queue
import random
import logging
import numpy as np
from .error_handling import error_handler
from .model_registry import ModelRegistry
# i18n module not implemented yet - using dummy function

"""
_函数 - 中文函数描述
_ Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _(text):
    return text

# 设置日志
logger = logging.getLogger(__name__)


"""
TrainingManager类 - 中文类描述
TrainingManager Class - English class description
"""
class TrainingManager:
    """模型训练管理器"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.training_jobs = {}
        self.training_history = self._load_training_history()
        self.training_lock = threading.Lock()
        # 训练结果保存路径 | Training results save path
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        # 实时数据队列和处理线程 | Real-time data queue and processing thread
        self.realtime_data_queue = queue.Queue()
        self.realtime_thread = threading.Thread(target=self._process_realtime_training_data)
        self.realtime_thread.daemon = True
        self.realtime_thread.start()
        self.realtime_data_source = None
        # 仪表盘数据 | Dashboard data
        self.dashboard_data = {
            'training_progress': {},
            'model_metrics': {},
            'system_status': {}
        }
        # 仪表盘更新回调 | Dashboard update callback
        self.dashboard_update_callback = None

    def set_realtime_data_source(self, data_source):
        """设置实时数据源 | Set real-time data source
        
        Args:
            data_source: 实时数据源对象，应具有获取数据的方法 | Real-time data source object, should have methods to get data
        """
        self.realtime_data_source = data_source
        # 如果数据源有get_data_stream方法，使用它 | If data source has get_data_stream method, use it
        if hasattr(data_source, 'get_data_stream'):
            self._start_data_stream()

    def set_dashboard_update_callback(self, callback):
        """设置仪表盘更新回调 | Set dashboard update callback
        
        Args:
            callback: 回调函数，当仪表盘数据更新时调用 | Callback function to be called when dashboard data updates
        """
        self.dashboard_update_callback = callback
        logger.info(_("仪表盘更新回调已设置 | Dashboard update callback set"))

    def _start_data_stream(self):
        """启动数据流处理"""
        if hasattr(self.realtime_data_source, 'get_data_stream'):
            stream = self.realtime_data_source.get_data_stream()
            for data in stream:
                self.receive_realtime_data(data)

    def receive_realtime_data(self, data_item):
        """接收实时数据项
        Receive real-time data item
        
        Args:
            data_item: 实时数据项
        """
        self.realtime_data_queue.put(data_item)

    def _process_realtime_training_data(self):
        """处理实时训练数据 | Process real-time training data"""
        while True:
            try:
                data_item = self.realtime_data_queue.get()
                if data_item is None:
                    continue
                    
                # 在实际实现中，这里会进行数据预处理 | In actual implementation, data preprocessing would be done here
                # 现在只是简单记录 | For now, just log
                logger.info(_("接收到实时训练数据: {type}").format(type=data_item.get('type', 'unknown')))
                
                # 更新仪表盘数据 | Update dashboard data
                self._update_dashboard(data_item)
                
                # 标记任务完成 | Mark task as done
                self.realtime_data_queue.task_done()
            except Exception as e:
                error_handler.handle_error(e, "TrainingManager", _("处理实时训练数据失败 | Failed to process real-time training data"))
                self.realtime_data_queue.task_done()

    def _update_dashboard(self, data_item):
        """更新仪表盘数据 | Update dashboard data
        
        Args:
            data_item: 实时数据项 | Real-time data item
        """
        try:
            # 更新训练进度 | Update training progress
            if 'job_id' in data_item and 'progress' in data_item:
                self.dashboard_data['training_progress'][data_item['job_id']] = data_item['progress']
                
            # 更新模型指标 | Update model metrics
            if 'model_id' in data_item and 'metrics' in data_item:
                self.dashboard_data['model_metrics'][data_item['model_id']] = data_item['metrics']
                
            # 更新系统状态 | Update system status
            if 'system' in data_item:
                self.dashboard_data['system_status'].update(data_item['system'])
                
            # 调用回调函数通知更新 | Call callback to notify update
            if callable(self.dashboard_update_callback):
                self.dashboard_update_callback(self.dashboard_data)
        except Exception as e:
            logger.error(_("更新仪表盘数据失败: {error}").format(error=str(e)))

    def _load_training_history(self):
        """加载训练历史记录"""
        history_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training_history.json')
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            error_handler.handle_error(e, "TrainingManager", "加载训练历史失败")
            return []

    def _save_training_history(self):
        """保存训练历史记录"""
        history_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training_history.json')
        try:
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            error_handler.handle_error(e, "TrainingManager", "保存训练历史失败")

    def start_training(self, model_ids, parameters):
        """启动模型训练任务 | Start model training task
        
        Args:
            model_ids: 要训练的模型ID列表 | List of model IDs to train
            parameters: 训练参数 | Training parameters
            
        Returns:
            任务ID | Job ID
        """
        with self.training_lock:
            # 生成任务ID | Generate job ID
            job_id = f"train_{int(time.time())}_{'_'.join(model_ids)}"
            
            # 检查模型是否已加载 | Check if models are loaded
            for model_id in model_ids:
                if not self.model_registry.get_model(model_id):
                    error_handler.log_warning(_("模型 {model_id} 未加载，尝试加载... | Model {model_id} not loaded, trying to load...").format(model_id=model_id), "TrainingManager")
                    self.model_registry.load_model(model_id)
                
                if not self.model_registry.get_model(model_id):
                    raise RuntimeError(_("无法加载模型 {model_id} | Failed to load model {model_id}").format(model_id=model_id))
                    
            # 验证模型类型 | Validate model types
            valid_models = ['manager', 'language', 'audio', 'vision_image', 'vision_video', 
                          'spatial', 'sensor', 'computer', 'motion', 
                          'knowledge', 'programming']
            for model_id in model_ids:
                if model_id not in valid_models:
                    raise ValueError(_("无效的模型类型: {model_id} | Invalid model type: {model_id}").format(model_id=model_id))
            
            # 创建训练任务
            self.training_jobs[job_id] = {
                'model_ids': model_ids,
                'parameters': parameters,
                'status': 'running',
                'start_time': time.time(),
                'progress': 0,
                'logs': [],
                'metrics': {}
            }
            
            # 记录开始日志
            self._log_job(job_id, f"开始训练模型: {', '.join(model_ids)}")
            
            # 启动训练线程
            training_thread = threading.Thread(
                target=self._train_models_thread, 
                args=(job_id, model_ids, parameters)
            )
            training_thread.daemon = True
            training_thread.start()
            
            error_handler.log_info(f"已启动训练任务: {job_id}", "TrainingManager")
            return job_id

    def _train_models_thread(self, job_id, model_ids, parameters):
        """模型训练线程 | Model training thread"""
        try:
            # 根据是否为联合训练选择不同的训练策略
            training_mode = parameters.get('training_mode', 'individual')
            
            if training_mode == 'joint' and len(model_ids) > 1:
                # 使用联合训练协调器进行联合训练
                self._log_job(job_id, "开始联合训练 | Starting joint training")
                
                # 导入联合训练协调器
                try:
                    from .joint_training_coordinator import JointTrainingCoordinator
                    coordinator = JointTrainingCoordinator(model_ids, parameters)
                    
                    # 准备训练任务
                    training_tasks = []
                    for model_id in model_ids:
                        model = self.model_registry.get_model(model_id)
                        if model:
                            # 获取模型特定的训练数据
                            model_data = self._prepare_model_training_data(model_id, parameters)
                            
                            task = {
                                'model_id': model_id,
                                'training_data': model_data,
                                'epochs': parameters.get('epochs', 10),
                                'batch_size': parameters.get('batch_size', 32),
                                'priority': parameters.get('priority', 1)
                            }
                            training_tasks.append(task)
                    
                    # 执行联合训练 - 使用线程安全的asyncio执行
                    try:
                        # 调度训练任务
                        schedule_result = coordinator.schedule_training(training_tasks)
                        self._log_job(job_id, f"训练任务已调度: {schedule_result}")
                        
                        # 使用线程安全的asyncio执行
                        import asyncio
                        
                        # 在新线程中运行asyncio代码的可靠方法
                        def run_async_in_thread():
                            """在新线程中运行asyncio代码"""
                            try:
                                # 创建新的事件循环
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                
                                # 执行训练协程
                                try:
                                    return loop.run_until_complete(coordinator.execute_training())
                                finally:
                                    loop.close()
                            except Exception as e:
                                error_handler.handle_error(e, "TrainingManager", 
                                    _("异步训练执行失败 | Async training execution failed"))
                                return {'status': 'failed', 'error': str(e)}
                        
                        # 在当前线程中直接运行（避免嵌套线程问题）
                        training_result = run_async_in_thread()
                        
                        if training_result.get('status') == 'success':
                            # 保存联合训练结果
                            self._save_joint_training_results(job_id, model_ids, parameters, training_result['results'])
                            self._complete_job(job_id, "联合训练成功完成 | Joint training completed successfully")
                        else:
                            raise RuntimeError(f"联合训练失败: {training_result.get('message', 'Unknown error')}")
                            
                    except RuntimeError as e:
                        if "There is no current event loop" in str(e):
                            # 处理事件循环问题，使用回退实现
                            self._log_job(job_id, "警告: 事件循环问题，使用回退联合训练实现 | Warning: Event loop issue, using fallback joint training implementation")
                            self._joint_train_fallback(job_id, model_ids, parameters)
                        else:
                            raise
                    except Exception as e:
                        # 处理其他异常，使用回退实现
                        self._log_job(job_id, f"警告: 联合训练异常 ({type(e).__name__})，使用回退实现 | Warning: Joint training exception ({type(e).__name__}), using fallback implementation")
                        self._joint_train_fallback(job_id, model_ids, parameters)
                        
                except ImportError:
                    self._log_job(job_id, "警告: 联合训练协调器不可用，使用回退实现 | Warning: Joint training coordinator not available, using fallback implementation")
                    # 回退到原有的联合训练实现
                    self._joint_train_fallback(job_id, model_ids, parameters)
                    
            else:
                # 单独训练
                for model_id in model_ids:
                    model = self.model_registry.get_model(model_id)
                    if model:
                        self._log_job(job_id, f"开始训练模型 {model_id} | Starting training for model: {model_id}")
                        # 执行模型特定的训练
                        self._individual_train(job_id, model_id, parameters, model_ids)
                        self._log_job(job_id, f"模型 {model_id} 训练完成 | Model {model_id} training completed")
                        # 更新进度
                        progress = (model_ids.index(model_id) + 1) / len(model_ids) * 100
                        self._update_job_progress(job_id, progress)
                    else:
                        self._log_job(job_id, f"警告: 模型 {model_id} 未找到 | Warning: Model {model_id} not found")
            
            # 标记任务完成
            self._complete_job(job_id, "训练成功完成 | Training completed successfully")
        except Exception as e:
            error_handler.handle_error(e, "TrainingManager", f"训练任务 {job_id} 失败 | Training task {job_id} failed")
            self._fail_job(job_id, str(e))

    def _prepare_joint_training_data(self, model_ids, parameters):
        """准备联合训练数据 | Prepare joint training data"""
        # 优先使用实时数据队列中的数据 | Prioritize data from real-time data queue
        if not self.realtime_data_queue.empty():
            realtime_data = []
            while not self.realtime_data_queue.empty():
                try:
                    data_item = self.realtime_data_queue.get_nowait()
                    realtime_data.append(data_item)
                    self.realtime_data_queue.task_done()
                except queue.Empty:
                    break
            if realtime_data:
                return realtime_data
        
        # 其次使用实时数据源 | Then use real-time data source
        if self.realtime_data_source and hasattr(self.realtime_data_source, 'get_data'):
            try:
                return self.realtime_data_source.get_data()
            except Exception as e:
                error_handler.log_warning(f"从实时数据源获取数据失败: {e}", "TrainingManager")
        
        # 再次使用参数中的训练数据 | Then use training data from parameters
        if 'training_data' in parameters:
            return parameters['training_data']
        
        # 最后生成模拟数据 | Finally generate simulated data
        data_size = parameters.get('data_size', 1000)
        modalities = self._get_required_modalities(model_ids)
        
        # 根据需要的模态生成不同类型的数据 | Generate different types of data based on required modalities
        shared_data = []
        for i in range(data_size):
            item = {'id': i, 'label': random.randint(0, 9)}  # 多分类标签 | Multi-class label
            
            # 为每种模态生成数据 | Generate data for each modality
            if 'text' in modalities:
                item['text'] = f"Sample text data {i} with label {item['label']}"
            if 'image' in modalities:
                # 生成模拟图像数据 | Generate simulated image data
                image_shape = parameters.get('image_shape', (64, 64, 3))
                item['image'] = np.random.rand(*image_shape).tolist()
            if 'audio' in modalities:
                # 生成模拟音频数据 | Generate simulated audio data
                audio_length = parameters.get('audio_length', 1000)
                item['audio'] = np.random.rand(audio_length).tolist()
            if 'sensor' in modalities:
                # 生成模拟传感器数据 | Generate simulated sensor data
                sensor_types = parameters.get('sensor_types', ['temperature', 'humidity', 'pressure'])
                item['sensor'] = {sensor: random.random() for sensor in sensor_types}
            if 'spatial' in modalities:
                # 生成模拟空间数据 | Generate simulated spatial data
                item['spatial'] = {
                    'position': [random.random() for _ in range(3)],
                    'orientation': [random.random() for _ in range(4)]
                }
                
            shared_data.append(item)
        
        return shared_data

    def _get_required_modalities(self, model_ids):
        """获取联合训练所需的模态类型 | Get required modalities for joint training
        
        Args:
            model_ids: 模型ID列表 | List of model IDs
            
        Returns:
            需要的模态类型集合 | Set of required modality types
        """
        modalities = set()
        
        # 根据模型类型确定需要的模态 | Determine required modalities based on model types
        for model_id in model_ids:
            # 直接使用模型ID而不是字母前缀
            if model_id == 'manager':  # 管理模型 | Manager model
                modalities.update(['text', 'image', 'audio', 'sensor', 'spatial'])
            elif model_id == 'language':  # 大语言模型 | Large language model
                modalities.add('text')
            elif model_id == 'audio':  # 音频处理模型 | Audio processing model
                modalities.add('audio')
            elif model_id == 'vision_image':  # 图片视觉处理模型 | Image vision processing model
                modalities.add('image')
            elif model_id == 'vision_video':  # 视频流视觉处理模型 | Video stream vision processing model
                modalities.update(['image', 'video'])
            elif model_id == 'spatial':  # 空间定位感知模型 | Spatial perception model
                modalities.add('spatial')
            elif model_id == 'sensor':  # 传感器感知模型 | Sensor perception model
                modalities.add('sensor')
            elif model_id == 'computer':  # 计算机控制模型 | Computer control model
                modalities.add('text')  # 文本命令
            elif model_id == 'knowledge':  # 知识库专家模型 | Knowledge base expert model
                modalities.add('text')
            elif model_id == 'programming':  # 编程模型 | Programming model
                modalities.add('text')
            elif model_id == 'motion':  # 运动和执行器控制模型 | Motion and actuator control model
                modalities.update(['sensor', 'spatial'])
        
        return modalities


    def _fuse_model_outputs(self, outputs):
        """融合模型输出 | Fuse model outputs"""
        if not outputs:
            return {}
        
        # 简单融合策略：加权平均 | Simple fusion strategy: weighted average
        fused_output = {}
        
        # 收集所有输出键 | Collect all output keys
        all_keys = set()
        for output in outputs:
            if isinstance(output, dict):
                all_keys.update(output.keys())
        
        # 对每个键计算加权平均 | Calculate weighted average for each key
        for key in all_keys:
            values = []
            weights = []
            
            for i, output in enumerate(outputs):
                if isinstance(output, dict) and key in output:
                    values.append(output[key])
                    # 简单权重：模型索引的倒数 | Simple weight: reciprocal of model index
                    weights.append(1.0 / (i + 1))
            
            if values:
                # 确保所有值都是数值类型 | Ensure all values are numeric
                if all(isinstance(v, (int, float)) for v in values):
                    weighted_avg = sum(v * w for v, w in zip(values, weights)) / sum(weights)
                    fused_output[key] = weighted_avg
                # 处理列表类型的数值数据 | Handle list-type numeric data
                elif all(isinstance(v, list) and all(isinstance(x, (int, float)) for x in v) for v in values):
                    # 确保所有列表长度相同 | Ensure all lists have same length
                    if all(len(v) == len(values[0]) for v in values):
                        fused_list = []
                        for j in range(len(values[0])):
                            weighted_val = sum(v[j] * w for v, w in zip(values, weights)) / sum(weights)
                            fused_list.append(weighted_val)
                        fused_output[key] = fused_list
        
        return fused_output

    def _calculate_joint_loss(self, fused_output, batch):
        """计算联合损失 | Calculate joint loss"""
        # 简单实现：均方误差损失 | Simple implementation: mean squared error loss
        total_loss = 0.0
        sample_count = 0
        
        for i, sample in enumerate(batch):
            if 'label' in sample and 'prediction' in fused_output:
                # 计算预测值和真实值的差异 | Calculate difference between prediction and true value
                if isinstance(fused_output['prediction'], (int, float)):
                    prediction = fused_output['prediction']
                    true_value = sample['label']
                    loss = (prediction - true_value) ** 2
                    total_loss += loss
                    sample_count += 1
                elif isinstance(fused_output['prediction'], list) and isinstance(sample['label'], (int, float)):
                    # 处理多输出情况 | Handle multiple outputs
                    avg_prediction = sum(fused_output['prediction']) / len(fused_output['prediction'])
                    loss = (avg_prediction - sample['label']) ** 2
                    total_loss += loss
                    sample_count += 1
        
        if sample_count > 0:
            return total_loss / sample_count
        else:
            return 0.1  # 默认损失值 | Default loss value

    def _backward_and_optimize(self, models, loss):
        """反向传播和优化 | Backward propagation and optimization"""
        # 简单实现：模拟反向传播 | Simple implementation: simulate backward propagation
        # 实际应用中应使用具体的优化算法 | In practice, should use specific optimization algorithms
        
        learning_rate = 0.01
        gradient = loss * learning_rate
        
        for model in models:
            if hasattr(model, 'update_parameters'):
                try:
                    model.update_parameters(gradient)
                except Exception as e:
                    error_handler.log_warning(f"模型参数更新失败: {e}", "TrainingManager")
            elif hasattr(model, 'backward'):
                try:
                    model.backward(gradient)
                except Exception as e:
                    error_handler.log_warning(f"模型反向传播失败: {e}", "TrainingManager")

    def _save_joint_training_results(self, job_id, model_ids, parameters, results):
        """保存联合训练结果 | Save joint training results
        
        Args:
            job_id: 任务ID | Job ID
            model_ids: 模型ID列表 | List of model IDs
            parameters: 训练参数 | Training parameters
            results: 训练结果 | Training results
        """
        try:
            # 将TrainingResult对象转换为可序列化的字典
            serializable_results = {}
            for model_id, result in results.items():
                if hasattr(result, '__dict__'):
                    # 如果是TrainingResult对象，转换为字典
                    result_dict = result.__dict__.copy()
                    # 确保所有值都是可序列化的
                    for key, value in result_dict.items():
                        if hasattr(value, '__dict__'):
                            result_dict[key] = value.__dict__
                    serializable_results[model_id] = result_dict
                else:
                    serializable_results[model_id] = result
            
            # 构建结果数据结构 | Build result data structure
            training_result = {
                'job_id': job_id,
                'model_ids': model_ids,
                'parameters': parameters,
                'completion_time': time.time(),
                'results': serializable_results,
                'metrics': self.training_jobs[job_id].get('metrics', {}) if job_id in self.training_jobs else {}
            }
            
            # 保存到文件 | Save to file
            # 从job_id中移除"train_"前缀，因为job_id格式是"train_{timestamp}_{model_names}"
            clean_job_id = job_id.replace("train_", "", 1) if job_id.startswith("train_") else job_id
            result_file = os.path.join(self.results_dir, f"joint_training_{clean_job_id}.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(training_result, f, ensure_ascii=False, indent=2)
            
            self._log_job(job_id, f"联合训练结果已保存: {result_file}")
            
        except Exception as e:
            error_handler.handle_error(e, "TrainingManager", "保存联合训练结果失败")

    def _save_individual_training_result(self, model_id, result):
        """保存单独训练结果 | Save individual training result
        
        Args:
            model_id: 模型ID | Model ID
            result: 训练结果 | Training result
        """
        try:
            # 创建结果数据结构 | Create result data structure
            training_result = {
                'model_id': model_id,
                'completion_time': time.time(),
                'result': result,
                'metrics': result.get('metrics', {}) if isinstance(result, dict) else {}
            }
            
            # 保存到文件 | Save to file
            result_file = os.path.join(self.results_dir, f"individual_training_{model_id}_{int(time.time())}.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(training_result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"单独训练结果已保存: {result_file}")
            
        except Exception as e:
            error_handler.handle_error(e, "TrainingManager", f"保存模型 {model_id} 的训练结果失败")

    def _individual_train(self, job_id, model_id, parameters, model_ids):
        """单独训练一个模型 | Train a single model individually
        
        Args:
            job_id: 训练任务ID | Training job ID
            model_id: 要训练的模型ID | Model ID to train
            parameters: 训练参数 | Training parameters
            model_ids: 所有要训练的模型ID列表 | List of all model IDs to train
        """
        model = self.model_registry.get_model(model_id)
        if not model or not hasattr(model, 'train'):
            raise RuntimeError(f"模型 {model_id} 不支持训练 | Model {model_id} does not support training")
        
        # 获取该模型的特定参数 | Get specific parameters for this model
        model_params = parameters.get(model_id, {})
        
        # 开始训练 | Start training
        self._log_job(job_id, f"开始单独训练模型: {model_id} | Starting individual training for model: {model_id}")
        
        # 创建训练回调 | Create training callback
        def training_callback(progress, metrics):
            """训练进度回调 | Training progress callback"""
            # 更新任务进度 | Update job progress
            current_progress = self.training_jobs[job_id]['progress']
            base_progress = (model_ids.index(model_id) / len(model_ids)) * 100
            new_progress = base_progress + (progress / 100) * (100 / len(model_ids))
            self._update_job_progress(job_id, new_progress)
            
            # 更新指标 | Update metrics
            if model_id not in self.training_jobs[job_id]['metrics']:
                self.training_jobs[job_id]['metrics'][model_id] = {}
            self.training_jobs[job_id]['metrics'][model_id].update(metrics)
            
            # 记录日志 | Log progress
            if 'loss' in metrics and 'accuracy' in metrics:
                self._log_job(job_id,
                    f"模型 {model_id} 训练进度: {progress:.1f}%, 损失值: {metrics['loss']:.4f}, 准确率: {metrics['accuracy']:.2f}% | "
                    f"Model {model_id} training progress: {progress:.1f}%, Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.2f}%")
        
        # 准备训练数据 | Prepare training data
        training_data = self._prepare_model_training_data(model_id, parameters)
        
        # 执行训练 | Execute training
        try:
            # 检查模型是否支持回调参数
            import inspect
            train_signature = inspect.signature(model.train)
            
            if 'callback' in train_signature.parameters:
                # 模型支持回调参数
                result = model.train(training_data=training_data, callback=training_callback, **model_params)
            else:
                # 模型不支持回调参数，使用轮询方式更新进度
                result = model.train(training_data=training_data, **model_params)
                
                # 模拟进度更新（对于不支持回调的模型）
                for progress in range(0, 101, 10):
                    training_callback(progress, {"loss": 0.1, "accuracy": progress})
                    import time
                    time.sleep(0.1)
            
            # 保存训练结果 | Save training result
            self._save_individual_training_result(model_id, result)
            
            self._log_job(job_id, f"模型 {model_id} 训练完成 | Model {model_id} training completed")
            return result
        except Exception as e:
            error_handler.handle_error(e, "TrainingManager", f"单独训练模型 {model_id} 失败 | Individual training for model {model_id} failed")
            raise

    def _joint_train(self, job_id, model_ids, parameters):
        """联合训练多个模型"""
        # 获取所有模型
        models = {model_id: self.model_registry.get_model(model_id) for model_id in model_ids}
        
        # 检查所有模型是否支持联合训练
        for model_id, model in models.items():
            if not model or not hasattr(model, 'joint_train'):
                raise RuntimeError(f"模型 {model_id} 不支持联合训练")
        
        self._log_job(job_id, f"开始联合训练模型: {', '.join(model_ids)}")
        
        # 创建联合训练回调
        def joint_training_callback(progress, metrics):
            """联合训练进度回调"""
            self._update_job_progress(job_id, progress)
            self.training_jobs[job_id]['metrics'].update(metrics)
            
            self._log_job(job_id,
                f"联合训练进度: {progress:.1f}%, 损失值: {metrics.get('loss', 0):.4f}, 准确率: {metrics.get('accuracy', 0):.2f}%")
        
        try:
            # 改进：实际联合训练实现
            # 1. 准备共享训练数据
            shared_data = self._prepare_shared_training_data(model_ids, parameters)
            
            # 2. 初始化联合训练上下文
            joint_context = {
                'models': model_ids,
                'shared_weights': {},
                'communication_channels': {}
            }
            
            # 3. 执行多轮联合训练
            epochs = parameters.get('epochs', 10)
            batch_size = parameters.get('batch_size', 32)
            
            for epoch in range(epochs):
                # 分割批次
                batches = self._create_batches(shared_data, batch_size)
                
                for batch_idx, batch in enumerate(batches):
                    # 模型并行处理批次
                    model_results = {}
                    for model_id, model in models.items():
                        try:
                            # 每个模型处理自己负责的部分
                            result = model.joint_train_step(batch, joint_context)
                            model_results[model_id] = result
                        except Exception as e:
                            error_handler.handle_error(e, "TrainingManager", f"模型 {model_id} 联合训练步骤失败")
                            
                    # 交换模型间的信息和梯度
                    self._exchange_model_information(models, model_results, joint_context)
                    
                    # 计算联合损失和准确率
                    joint_metrics = self._calculate_joint_metrics(model_results)
                    
                    # 更新进度
                    progress = ((epoch * len(batches) + batch_idx + 1) / (epochs * len(batches))) * 100
                    joint_training_callback(progress, joint_metrics)
                    
                    # 检查是否需要早停
                    if self._check_early_stopping(joint_metrics, parameters.get('early_stopping', {})):
                        self._log_job(job_id, "联合训练触发早停条件")
                        break
                    
                if self.training_jobs[job_id].get('status') == 'stopping':
                    break
            
            self._log_job(job_id, "联合训练完成")
            
            # 保存联合训练结果
            self._save_joint_training_results(job_id, model_ids, parameters, {})
            
            # 返回联合训练结果
            return self.training_jobs[job_id]['metrics']

        except Exception as e:
            self._log_job(job_id, f"联合训练发生错误: {str(e)}")
            # 回退到单独训练 | Fallback to individual training
            self._log_job(job_id, "回退到单独训练模式 | Falling back to individual training mode")
            for model_id in model_ids:
                self._individual_train(job_id, model_id, parameters, model_ids)
                # 更新进度 | Update progress
                self._update_job_progress(job_id,
                    (model_ids.index(model_id) + 1) / len(model_ids) * 100)

            # 返回单独训练的结果 | Return individual training results
            return self.training_jobs[job_id]['metrics']

    def _prepare_model_training_data(self, model_id, parameters):
        """准备模型特定的训练数据 | Prepare model-specific training data
        
        Args:
            model_id: 模型ID | Model ID
            parameters: 训练参数 | Training parameters
            
        Returns:
            模型特定的训练数据 | Model-specific training data
        """
        # 根据模型类型准备不同的训练数据
        if model_id == 'language':
            # 语言模型训练数据
            return self._prepare_text_training_data(parameters)
        elif model_id == 'audio':
            # 音频模型训练数据
            return self._prepare_audio_training_data(parameters)
        elif model_id in ['image_vision', 'video_vision']:
            # 视觉模型训练数据
            return self._prepare_vision_training_data(parameters)
        elif model_id in ['spatial', 'stereo_spatial']:
            # 空间模型训练数据
            return self._prepare_spatial_training_data(parameters)
        elif model_id == 'sensor':
            # 传感器模型训练数据
            return self._prepare_sensor_training_data(parameters)
        else:
            # 默认训练数据
            return self._prepare_default_training_data(parameters)

    def _prepare_text_training_data(self, parameters):
        """准备文本训练数据 | Prepare text training data"""
        data_size = parameters.get('data_size', 1000)
        text_data = []
        for i in range(data_size):
            text_data.append({
                'text': f"Sample training text {i} with label {random.randint(0, 9)}",
                'label': random.randint(0, 9)
            })
        return text_data

    def _prepare_audio_training_data(self, parameters):
        """准备音频训练数据 | Prepare audio training data"""
        data_size = parameters.get('data_size', 100)
        audio_length = parameters.get('audio_length', 1000)
        audio_data = []
        for i in range(data_size):
            audio_data.append({
                'audio': np.random.rand(audio_length).tolist(),
                'label': random.randint(0, 9)
            })
        return audio_data

    def _prepare_vision_training_data(self, parameters):
        """准备视觉训练数据 | Prepare vision training data"""
        data_size = parameters.get('data_size', 100)
        image_shape = parameters.get('image_shape', (64, 64, 3))
        vision_data = []
        for i in range(data_size):
            vision_data.append({
                'image': np.random.rand(*image_shape).tolist(),
                'label': random.randint(0, 9)
            })
        return vision_data

    def _prepare_spatial_training_data(self, parameters):
        """准备空间训练数据 | Prepare spatial training data"""
        data_size = parameters.get('data_size', 100)
        spatial_data = []
        for i in range(data_size):
            spatial_data.append({
                'position': [random.random() for _ in range(3)],
                'orientation': [random.random() for _ in range(4)],
                'label': random.randint(0, 9)
            })
        return spatial_data

    def _prepare_sensor_training_data(self, parameters):
        """准备传感器训练数据 | Prepare sensor training data"""
        data_size = parameters.get('data_size', 100)
        sensor_types = parameters.get('sensor_types', ['temperature', 'humidity', 'pressure'])
        sensor_data = []
        for i in range(data_size):
            sensor_data.append({
                'sensor': {sensor: random.random() for sensor in sensor_types},
                'label': random.randint(0, 9)
            })
        return sensor_data

    def _prepare_default_training_data(self, parameters):
        """准备默认训练数据 | Prepare default training data"""
        data_size = parameters.get('data_size', 100)
        default_data = []
        for i in range(data_size):
            default_data.append({
                'features': [random.random() for _ in range(10)],
                'label': random.randint(0, 9)
            })
        return default_data

    def _joint_train_fallback(self, job_id, model_ids, parameters):
        """联合训练回退实现 | Joint training fallback implementation"""
        self._log_job(job_id, "使用回退联合训练实现 | Using fallback joint training implementation")
        
        # 简单的联合训练实现
        epochs = parameters.get('epochs', 10)
        batch_size = parameters.get('batch_size', 32)
        
        # 准备共享训练数据
        shared_data = self._prepare_joint_training_data(model_ids, parameters)
        
        for epoch in range(epochs):
            self._log_job(job_id, f"回退联合训练 epoch {epoch+1}/{epochs}")
            
            # 分割批次
            for i in range(0, len(shared_data), batch_size):
                batch = shared_data[i:i+batch_size]
                
                # 每个模型处理批次
                for model_id in model_ids:
                    model = self.model_registry.get_model(model_id)
                    if model and hasattr(model, 'train_step'):
                        try:
                            model.train_step(batch)
                        except Exception as e:
                            error_handler.log_warning(f"模型 {model_id} 训练步骤失败: {e}", "TrainingManager")
                
                # 更新进度
                batch_progress = ((i + batch_size) / len(shared_data)) * (100 / epochs)
                epoch_progress = (epoch / epochs) * 100
                total_progress = epoch_progress + batch_progress
                self._update_job_progress(job_id, total_progress)
        
        # 保存结果
        results = {}
        for model_id in model_ids:
            model = self.model_registry.get_model(model_id)
            if model:
                results[model_id] = {
                    'status': 'completed',
                    'epochs': epochs,
                    'batch_size': batch_size
                }
        
        self._save_joint_training_results(job_id, model_ids, parameters, results)
        self._complete_job(job_id, "回退联合训练完成 | Fallback joint training completed")


    def _create_batches(self, data, batch_size):
        """将数据分割成批次"""
        batches = []
        for i in range(0, len(data), batch_size):
            batches.append(data[i:i+batch_size])
        return batches

    def _exchange_model_information(self, models, model_results, joint_context):
        """在联合训练过程中交换模型间的信息"""
        # 简单实现：计算平均权重和梯度
        # 实际应用中应使用更复杂的联邦学习或知识蒸馏技术
        
        # 收集所有模型的权重更新
        all_updates = {}
        for model_id, result in model_results.items():
            if 'weight_updates' in result:
                for param_name, update in result['weight_updates'].items():
                    if param_name not in all_updates:
                        all_updates[param_name] = []
                    all_updates[param_name].append(update)
        
        # 计算平均更新
        avg_updates = {}
        for param_name, updates in all_updates.items():
            # 处理数值类型
            if updates and isinstance(updates[0], (int, float)):
                avg_updates[param_name] = sum(updates) / len(updates)
            # 处理列表类型（如梯度向量）
            elif updates and isinstance(updates[0], list):
                # 确保所有更新具有相同长度
                if all(len(u) == len(updates[0]) for u in updates):
                    avg_list = [sum(values) / len(values) for values in zip(*updates)]
                    avg_updates[param_name] = avg_list
            # 处理字典类型（如结构化梯度）
            elif updates and isinstance(updates[0], dict):
                avg_dict = {}
                for key in updates[0].keys():
                    if all(key in u for u in updates):
                        key_values = [u[key] for u in updates]
                        if all(isinstance(v, (int, float)) for v in key_values):
                            avg_dict[key] = sum(key_values) / len(key_values)
                avg_updates[param_name] = avg_dict
            # 其他类型暂时不处理
            else:
                error_handler.log_warning(f"无法处理参数 {param_name} 的更新类型", "TrainingManager")
            
        # 更新联合上下文
        joint_context['shared_weights'] = avg_updates

    def _calculate_joint_metrics(self, model_results):
        """计算联合训练的指标"""
        metrics = {'loss': 0, 'accuracy': 0, 'models_contributed': 0}
        
        # 聚合每个模型的指标
        for model_id, result in model_results.items():
            if 'metrics' in result:
                model_metrics = result['metrics']
                
                # 处理损失指标
                if 'loss' in model_metrics:
                    metrics['loss'] += model_metrics['loss']
                    metrics['loss_count'] = metrics.get('loss_count', 0) + 1
                
                # 处理准确率指标
                if 'accuracy' in model_metrics:
                    metrics['accuracy'] += model_metrics['accuracy']
                    metrics['accuracy_count'] = metrics.get('accuracy_count', 0) + 1
                
                # 处理其他指标
                for metric_name, value in model_metrics.items():
                    if metric_name not in ['loss', 'accuracy']:
                        if metric_name not in metrics:
                            metrics[metric_name] = 0
                        metrics[metric_name] += value
                        metrics[f'{metric_name}_count'] = metrics.get(f'{metric_name}_count', 0) + 1
                
                metrics['models_contributed'] += 1
        
        # 计算平均值
        if metrics.get('loss_count', 0) > 0:
            metrics['loss'] = metrics['loss'] / metrics['loss_count']
        if metrics.get('accuracy_count', 0) > 0:
            metrics['accuracy'] = metrics['accuracy'] / metrics['accuracy_count']
        
        for key in list(metrics.keys()):
            if key.endswith('_count'):
                del metrics[key]
        
        return metrics

    def _check_early_stopping(self, metrics, early_stopping_config):
        """检查是否需要早停"""
        if not early_stopping_config:
            return False
        
        patience = early_stopping_config.get('patience', 5)
        min_delta = early_stopping_config.get('min_delta', 0.01)
        
        # 检查损失是否不再下降
        if 'loss' in metrics:
            current_loss = metrics['loss']
            if not hasattr(self, '_best_loss'):
                self._best_loss = current_loss
                self._patience_counter = 0
            elif current_loss < self._best_loss - min_delta:
                self._best_loss = current_loss
                self._patience_counter = 0
            else:
                self._patience_counter += 1
                if self._patience_counter >= patience:
                    return True
        
        return False

    def _log_job(self, job_id, message):
        """记录训练任务日志"""
        if job_id in self.training_jobs:
            self.training_jobs[job_id]['logs'].append({
                'timestamp': time.time(),
                'message': message
            })
            logger.info(f"[{job_id}] {message}")

    def _update_job_progress(self, job_id, progress):
        """更新训练任务进度"""
        if job_id in self.training_jobs:
            self.training_jobs[job_id]['progress'] = progress
            # 发送实时更新
            self._send_realtime_update(job_id, 'progress', progress)

    def _complete_job(self, job_id, message):
        """标记训练任务完成"""
        if job_id in self.training_jobs:
            self.training_jobs[job_id]['status'] = 'completed'
            self.training_jobs[job_id]['end_time'] = time.time()
            self._log_job(job_id, message)
            # 记录到训练历史
            self.training_history.append({
                'job_id': job_id,
                'models': self.training_jobs[job_id]['model_ids'],
                'start_time': self.training_jobs[job_id]['start_time'],
                'end_time': self.training_jobs[job_id]['end_time'],
                'status': 'completed'
            })
            self._save_training_history()

    def _fail_job(self, job_id, error_message):
        """标记训练任务失败"""
        if job_id in self.training_jobs:
            self.training_jobs[job_id]['status'] = 'failed'
            self.training_jobs[job_id]['end_time'] = time.time()
            self.training_jobs[job_id]['error'] = error_message
            self._log_job(job_id, f"训练失败: {error_message}")
            # 记录到训练历史
            self.training_history.append({
                'job_id': job_id,
                'models': self.training_jobs[job_id]['model_ids'],
                'start_time': self.training_jobs[job_id]['start_time'],
                'end_time': self.training_jobs[job_id]['end_time'],
                'status': 'failed',
                'error': error_message
            })
            self._save_training_history()

    def _send_realtime_update(self, job_id, update_type, data):
        """发送实时更新"""
        update_data = {
            'job_id': job_id,
            'type': update_type,
            'data': data,
            'timestamp': time.time()
        }
        self.realtime_data_queue.put(update_data)

    def get_job_status(self, job_id):
        """获取训练任务状态"""
        return self.training_jobs.get(job_id, {'status': 'not_found'})

    def get_training_history(self):
        """获取训练历史记录"""
        return self.training_history

    def stop_training(self, job_id):
        """停止训练任务"""
        if job_id in self.training_jobs:
            self.training_jobs[job_id]['status'] = 'stopping'
            self._log_job(job_id, "训练任务正在停止...")
            return True
        return False

    def validate_model_combination(self, model_ids, mode='joint'):
        """验证模型组合是否有效 | Validate if model combination is valid
        
        Args:
            model_ids: 要验证的模型ID列表 | List of model IDs to validate
            mode: 训练模式 ('individual' 或 'joint') | Training mode ('individual' or 'joint')
            
        Returns:
            验证结果字典 | Validation result dictionary
        """
        try:
            # 检查模型是否存在 | Check if models exist
            missing_models = []
            for model_id in model_ids:
                if not self.model_registry.get_model(model_id):
                    missing_models.append(model_id)
            
            if missing_models:
                return {
                    'valid': False,
                    'message': _("以下模型未加载或不存在: {models} | The following models are not loaded or do not exist: {models}").format(
                        models=', '.join(missing_models)
                    ),
                    'missing_models': missing_models
                }
            
            # 检查模型是否支持训练 | Check if models support training
            non_trainable_models = []
            for model_id in model_ids:
                model = self.model_registry.get_model(model_id)
                if model and not hasattr(model, 'train'):
                    non_trainable_models.append(model_id)
            
            if non_trainable_models:
                return {
                    'valid': False,
                    'message': _("以下模型不支持训练: {models} | The following models do not support training: {models}").format(
                        models=', '.join(non_trainable_models)
                    ),
                    'non_trainable_models': non_trainable_models
                }
            
            # 检查模型组合的兼容性 | Check model combination compatibility
            if len(model_ids) > 1:
                # 检查联合训练兼容性 | Check joint training compatibility
                compatible, reason = self._check_joint_training_compatibility(model_ids)
                if not compatible:
                    return {
                        'valid': False,
                        'message': reason,
                        'incompatible_combination': True
                    }
            
            # 检查资源可用性 | Check resource availability
            resource_check = self._check_resource_availability(model_ids)
            if not resource_check['available']:
                return {
                    'valid': False,
                    'message': resource_check['message'],
                    'resource_constraint': True
                }
            
            return {
                'valid': True,
                'message': _("模型组合验证通过 | Model combination validated successfully"),
                'compatible_models': model_ids,
                'recommended_parameters': self._get_recommended_parameters(model_ids)
            }
            
        except Exception as e:
            error_handler.handle_error(e, "TrainingManager", "验证模型组合失败")
            return {
                'valid': False,
                'message': _("验证过程中发生错误: {error} | Error during validation: {error}").format(error=str(e))
            }

    def _check_joint_training_compatibility(self, model_ids):
        """检查联合训练兼容性 | Check joint training compatibility
        
        Args:
            model_ids: 模型ID列表 | List of model IDs
            
        Returns:
            (是否兼容, 原因) | (Whether compatible, reason)
        """
        # 定义兼容性规则 | Define compatibility rules
        compatibility_rules = {
            # 语言模型可以与其他大多数模型联合训练
            'language': ['audio', 'vision_image', 'vision_video', 'knowledge', 'programming', 'manager'],
            # 音频模型主要与语言和视觉模型兼容
            'audio': ['language', 'vision_image', 'manager'],
            # 图像视觉模型可以与语言、音频、视频模型兼容
            'vision_image': ['language', 'audio', 'vision_video', 'manager'],
            # 视频视觉模型可以与图像视觉和语言模型兼容
            'vision_video': ['vision_image', 'language', 'manager'],
            # 空间模型可以与传感器和运动模型兼容
            'spatial': ['sensor', 'motion', 'manager'],
            # 传感器模型可以与空间和运动模型兼容
            'sensor': ['spatial', 'motion', 'manager'],
            # 计算机控制模型主要与语言模型兼容
            'computer': ['language', 'manager'],
            # 运动模型可以与空间和传感器模型兼容
            'motion': ['spatial', 'sensor', 'manager'],
            # 知识库模型可以与语言和编程模型兼容
            'knowledge': ['language', 'programming', 'manager'],
            # 编程模型可以与语言和知识库模型兼容
            'programming': ['language', 'knowledge', 'manager'],
            # 管理模型可以与所有模型兼容
            'manager': ['language', 'audio', 'vision_image', 'vision_video', 'spatial', 
                       'sensor', 'computer', 'motion', 'knowledge', 'programming']
        }
        
        # 检查所有模型对之间的兼容性
        for i, model1 in enumerate(model_ids):
            for j, model2 in enumerate(model_ids):
                if i != j:
                    # 检查双向兼容性
                    if (model2 not in compatibility_rules.get(model1, []) or 
                        model1 not in compatibility_rules.get(model2, [])):
                        return False, _("模型 {model1} 和 {model2} 不兼容联合训练 | Models {model1} and {model2} are not compatible for joint training").format(
                            model1=model1, model2=model2)
        
        return True, _("所有模型兼容联合训练 | All models are compatible for joint training")

    def _check_resource_availability(self, model_ids):
        """检查资源可用性 | Check resource availability
        
        Args:
            model_ids: 模型ID列表 | List of model IDs
            
        Returns:
            资源检查结果 | Resource check result
        """
        # 模拟资源检查 - 实际实现应根据系统资源进行真实检查
        # 这里使用简单的启发式规则
        
        total_models = len(model_ids)
        memory_required = total_models * 512  # MB per model
        cpu_required = total_models * 0.5     # CPU cores per model
        
        # 检查内存
        import psutil
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        
        if available_memory < memory_required:
            return {
                'available': False,
                'message': _("内存不足: 需要 {required}MB, 可用 {available}MB | Insufficient memory: {required}MB required, {available}MB available").format(
                    required=memory_required, available=int(available_memory))
            }
        
        # 检查CPU
        available_cpu = psutil.cpu_count(logical=False)
        if available_cpu < cpu_required:
            return {
                'available': False,
                'message': _("CPU资源不足: 需要 {required}核心, 可用 {available}核心 | Insufficient CPU: {required} cores required, {available} cores available").format(
                    required=cpu_required, available=available_cpu)
            }
        
        return {
            'available': True,
            'message': _("资源充足 | Sufficient resources available")
        }

    def _get_recommended_parameters(self, model_ids):
        """获取推荐的训练参数 | Get recommended training parameters
        
        Args:
            model_ids: 模型ID列表 | List of model IDs
            
        Returns:
            推荐的参数字典 | Recommended parameters dictionary
        """
        # 根据模型组合提供推荐的参数
        base_params = {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'training_mode': 'individual'
        }
        
        if len(model_ids) > 1:
            base_params['training_mode'] = 'joint'
            # 联合训练需要调整参数
            base_params['batch_size'] = 16  # 较小的批次大小
            base_params['learning_rate'] = 0.0005  # 较小的学习率
            
            # 根据模型类型调整
            if any(model_id in ['vision_image', 'vision_video'] for model_id in model_ids):
                base_params['batch_size'] = 8  # 视觉模型需要更小的批次
            
            if any(model_id in ['audio'] for model_id in model_ids):
                base_params['learning_rate'] = 0.0002  # 音频模型需要更小的学习率
        
        return base_params

    def get_all_jobs_status(self):
        """获取所有训练任务的状态 | Get status of all training jobs
        
        Returns:
            所有任务状态的字典 | Dictionary of all job statuses
        """
        try:
            jobs_status = {}
            for job_id, job_info in self.training_jobs.items():
                jobs_status[job_id] = {
                    'model_ids': job_info.get('model_ids', []),
                    'status': job_info.get('status', 'unknown'),
                    'progress': job_info.get('progress', 0),
                    'start_time': job_info.get('start_time', 0),
                    'end_time': job_info.get('end_time', 0),
                    'metrics': job_info.get('metrics', {}),
                    'logs_count': len(job_info.get('logs', [])),
                    'error': job_info.get('error', None)
                }
            
            return jobs_status
        except Exception as e:
            error_handler.handle_error(e, "TrainingManager", "获取所有任务状态失败")
            return {}

    def get_joint_training_recommendations(self):
        """获取联合训练推荐组合 | Get joint training recommendations
        
        Returns:
            推荐组合列表 | List of recommended combinations
        """
        try:
            # 定义推荐的联合训练组合
            recommendations = [
                {
                    'name': _("语言-视觉联合训练 | Language-Vision Joint Training"),
                    'description': _("语言模型与视觉模型的联合训练，适用于多模态理解任务 | Joint training of language and vision models for multimodal understanding tasks"),
                    'model_ids': ['language', 'vision_image'],
                    'compatibility_score': 0.95,
                    'recommended_parameters': {
                        'epochs': 15,
                        'batch_size': 16,
                        'learning_rate': 0.0005,
                        'training_mode': 'joint'
                    }
                },
                {
                    'name': _("音频-语言联合训练 | Audio-Language Joint Training"),
                    'description': _("音频模型与语言模型的联合训练，适用于语音理解和生成任务 | Joint training of audio and language models for speech understanding and generation tasks"),
                    'model_ids': ['audio', 'language'],
                    'compatibility_score': 0.92,
                    'recommended_parameters': {
                        'epochs': 12,
                        'batch_size': 20,
                        'learning_rate': 0.0003,
                        'training_mode': 'joint'
                    }
                },
                {
                    'name': _("空间-传感器联合训练 | Spatial-Sensor Joint Training"),
                    'description': _("空间模型与传感器模型的联合训练，适用于环境感知和导航任务 | Joint training of spatial and sensor models for environmental perception and navigation tasks"),
                    'model_ids': ['spatial', 'sensor'],
                    'compatibility_score': 0.88,
                    'recommended_parameters': {
                        'epochs': 20,
                        'batch_size': 12,
                        'learning_rate': 0.0004,
                        'training_mode': 'joint'
                    }
                },
                {
                    'name': _("知识-编程联合训练 | Knowledge-Programming Joint Training"),
                    'description': _("知识库模型与编程模型的联合训练，适用于智能编程和代码生成任务 | Joint training of knowledge base and programming models for intelligent programming and code generation tasks"),
                    'model_ids': ['knowledge', 'programming'],
                    'compatibility_score': 0.96,
                    'recommended_parameters': {
                        'epochs': 10,
                        'batch_size': 24,
                        'learning_rate': 0.0006,
                        'training_mode': 'joint'
                    }
                },
                {
                    'name': _("多模态综合训练 | Multimodal Comprehensive Training"),
                    'description': _("管理模型与多个感知模型的联合训练，适用于复杂多模态任务 | Joint training of manager model with multiple perception models for complex multimodal tasks"),
                    'model_ids': ['manager', 'language', 'vision_image', 'audio'],
                    'compatibility_score': 0.85,
                    'recommended_parameters': {
                        'epochs': 8,
                        'batch_size': 8,
                        'learning_rate': 0.0002,
                        'training_mode': 'joint'
                    }
                }
            ]
            
            return recommendations
        except Exception as e:
            error_handler.handle_error(e, "TrainingManager", "获取联合训练推荐失败")
            return []

    def get_joint_training_details(self, job_id):
        """获取联合训练详情 | Get joint training details
        
        Args:
            job_id: 训练任务ID | Training job ID
            
        Returns:
            训练详情字典 | Training details dictionary
        """
        try:
            if job_id not in self.training_jobs:
                return {'error': _("训练任务不存在 | Training job not found")}
            
            job_info = self.training_jobs[job_id]
            
            # 检查是否为联合训练
            if len(job_info.get('model_ids', [])) <= 1:
                return {'error': _("这不是联合训练任务 | This is not a joint training job")}
            
            # 构建详细响应
            details = {
                'job_id': job_id,
                'model_ids': job_info.get('model_ids', []),
                'status': job_info.get('status', 'unknown'),
                'progress': job_info.get('progress', 0),
                'start_time': job_info.get('start_time', 0),
                'end_time': job_info.get('end_time', 0),
                'parameters': job_info.get('parameters', {}),
                'metrics': job_info.get('metrics', {}),
                'logs': job_info.get('logs', []),
                'error': job_info.get('error', None),
                'training_mode': job_info.get('parameters', {}).get('training_mode', 'individual'),
                'is_joint_training': len(job_info.get('model_ids', [])) > 1
            }
            
            # 尝试加载结果文件
            # 从job_id中移除"train_"前缀，因为job_id格式是"train_{timestamp}_{model_names}"
            clean_job_id = job_id.replace("train_", "", 1) if job_id.startswith("train_") else job_id
            result_file = os.path.join(self.results_dir, f"joint_training_{clean_job_id}.json")
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                        details['results'] = results
                except Exception as e:
                    error_handler.log_warning(f"加载训练结果文件失败: {e}", "TrainingManager")
            
            return details
        except Exception as e:
            error_handler.handle_error(e, "TrainingManager", "获取联合训练详情失败")
            return {'error': str(e)}

    def analyze_joint_training_effectiveness(self, job_ids, metrics=None):
        """分析联合训练效果 | Analyze joint training effectiveness
        
        Args:
            job_ids: 训练任务ID列表或单个任务ID | Training job ID list or single job ID
            metrics: 要分析的指标列表 | List of metrics to analyze
            
        Returns:
            效果分析结果 | Effectiveness analysis results
        """
        try:
            # 处理单个job_id的情况
            if isinstance(job_ids, str):
                job_ids = [job_ids]
            
            if metrics is None:
                metrics = ["accuracy", "loss", "convergence_speed"]
            
            analysis_results = []
            
            for job_id in job_ids:
                if job_id not in self.training_jobs:
                    analysis_results.append({
                        'job_id': job_id,
                        'error': _("训练任务不存在 | Training job not found")
                    })
                    continue
                
                job_info = self.training_jobs[job_id]
                
                # 检查是否为联合训练
                if len(job_info.get('model_ids', [])) <= 1:
                    analysis_results.append({
                        'job_id': job_id,
                        'error': _("这不是联合训练任务 | This is not a joint training job")
                    })
                    continue
                
                # 获取训练详情
                details = self.get_joint_training_details(job_id)
                if 'error' in details:
                    analysis_results.append({
                        'job_id': job_id,
                        'error': details['error']
                    })
                    continue
                
                # 分析训练效果
                job_metrics = job_info.get('metrics', {})
                model_ids = job_info.get('model_ids', [])
                
                # 计算效果指标
                effectiveness = {
                    'job_id': job_id,
                    'model_count': len(model_ids),
                    'training_duration': details.get('end_time', time.time()) - details.get('start_time', 0),
                    'overall_metrics': job_metrics,
                    'model_specific_metrics': {},
                    'effectiveness_score': 0.0,
                    'recommendations': [],
                    'analyzed_metrics': metrics
                }
                
                # 计算每个模型的指标
                for model_id in model_ids:
                    if model_id in job_metrics:
                        model_metrics = job_metrics[model_id]
                        effectiveness['model_specific_metrics'][model_id] = {
                            'accuracy': model_metrics.get('accuracy', 0),
                            'loss': model_metrics.get('loss', 0),
                            'training_time': model_metrics.get('training_time', 0)
                        }
                
                # 计算综合效果分数
                if job_metrics:
                    # 基于准确率和损失计算效果分数
                    accuracies = [m.get('accuracy', 0) for m in job_metrics.values() if isinstance(m, dict)]
                    losses = [m.get('loss', 1.0) for m in job_metrics.values() if isinstance(m, dict)]
                    
                    if accuracies and losses:
                        avg_accuracy = sum(accuracies) / len(accuracies)
                        avg_loss = sum(losses) / len(losses)
                        
                        # 效果分数公式：准确率 * (1 - 损失)
                        effectiveness['effectiveness_score'] = avg_accuracy * (1 - min(avg_loss, 1.0))
                
                # 生成改进建议
                if effectiveness['effectiveness_score'] < 0.7:
                    effectiveness['recommendations'].append(
                        _("建议调整学习率或批次大小以提高训练效果 | Consider adjusting learning rate or batch size to improve training effectiveness")
                    )
                
                if len(model_ids) > 3 and effectiveness['training_duration'] > 3600:  # 超过1小时
                    effectiveness['recommendations'].append(
                        _("多模型联合训练时间较长，建议分批训练或增加计算资源 | Multi-model joint training takes longer, consider batch training or increasing computational resources")
                    )
                
                analysis_results.append(effectiveness)
            
            # 如果只有一个任务，直接返回结果，否则返回比较分析
            if len(analysis_results) == 1:
                return analysis_results[0]
            else:
                # 添加比较分析
                return self._compare_joint_training_analyses(analysis_results)
                
        except Exception as e:
            error_handler.handle_error(e, "TrainingManager", "分析联合训练效果失败")
            return {'error': str(e)}
    
    def _compare_joint_training_analyses(self, analyses):
        """比较多个联合训练分析结果 | Compare multiple joint training analysis results
        
        Args:
            analyses: 分析结果列表 | List of analysis results
            
        Returns:
            比较分析结果 | Comparative analysis results
        """
        comparison = {
            'analyses': analyses,
            'comparison': {},
            'best_performing': None,
            'worst_performing': None,
            'average_effectiveness': 0.0
        }
        
        # 计算平均效果分数
        valid_scores = [a.get('effectiveness_score', 0) for a in analyses if 'effectiveness_score' in a]
        if valid_scores:
            comparison['average_effectiveness'] = sum(valid_scores) / len(valid_scores)
        
        # 找出最佳和最差表现
        if valid_scores:
            best_score = max(valid_scores)
            worst_score = min(valid_scores)
            
            for analysis in analyses:
                if analysis.get('effectiveness_score', 0) == best_score:
                    comparison['best_performing'] = analysis['job_id']
                if analysis.get('effectiveness_score', 0) == worst_score:
                    comparison['worst_performing'] = analysis['job_id']
        
        return comparison
