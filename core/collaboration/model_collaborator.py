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
模型协作协调器 - 管理多个AI模型之间的复杂协作任务
Model Collaboration Orchestrator - Manages complex collaboration tasks between multiple AI models

功能描述：
- 支持多种协作模式（串行、并行、混合）
- 管理模型间的任务分配和数据流
- 提供协作性能监控和优化
- 支持动态协作策略调整
- 实现错误处理和恢复机制

Function Description:
- Supports multiple collaboration modes (serial, parallel, hybrid)
- Manages task allocation and data flow between models
- Provides collaboration performance monitoring and optimization
- Supports dynamic collaboration strategy adjustment
- Implements error handling and recovery mechanisms
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Callable, Optional, Union
from enum import Enum
import logging
from dataclasses import dataclass
from collections import deque
import json

from core.error_handling import error_handler
from core.model_registry import get_model_registry


"""
CollaborationMode类 - 中文类描述
CollaborationMode Class - English class description
"""
class CollaborationMode(Enum):
    """协作模式枚举
    Collaboration Mode Enumeration
    """
    SERIAL = "serial"          # 串行处理：一个模型接一个模型顺序处理
    PARALLEL = "parallel"      # 并行处理：多个模型同时处理相同输入
    HYBRID = "hybrid"          # 混合模式：结合串行和并行处理
    CONDITIONAL = "conditional" # 条件分支：根据条件选择不同的处理路径

"""
CollaborationConfig类 - 中文类描述
CollaborationConfig Class - English class description
"""
@dataclass
class CollaborationConfig:
    """协作配置数据类
    Collaboration Configuration Data Class
    """
    collaboration_id: str
    model_ids: List[str]
    mode: CollaborationMode
    task_description: str
    parameters: Dict[str, Any]
    timeout: float = 30.0  # 超时时间（秒）
    max_retries: int = 3   # 最大重试次数

@dataclass
class CollaborationStats:
    """协作统计信息数据类
    Collaboration Statistics Data Class
    """
    start_time: float = 0.0
    end_time: float = 0.0
    execution_time: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    total_collaborations: int = 0
    avg_execution_time: float = 0.0


class ModelCollaborationOrchestrator:
    """模型协作协调器
    Model Collaboration Orchestrator
    
    功能：管理和协调多个AI模型之间的协作任务，优化任务处理流程
    Function: Manages and coordinates collaboration tasks between multiple AI models, 
              optimizing task processing workflows
    
    __init__函数 - 中文函数描述
    __init__ Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def __init__(self):
        self.active_collaborations: Dict[str, CollaborationConfig] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
        self.performance_stats: CollaborationStats = CollaborationStats()
        self.collaboration_patterns: Dict[str, Dict[str, Any]] = {}
        self.shared_memory: Dict[str, Any] = {}
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
        
        # 初始化默认协作模式
        self._initialize_default_patterns()
        
        self.logger.info("模型协作协调器初始化完成")
        self.logger.info("Model Collaboration Orchestrator initialized")
    
    def _initialize_default_patterns(self):
        """初始化默认协作模式
        Initialize default collaboration patterns
        """
        # 视觉-语言协作模式
        self.collaboration_patterns["vision_language"] = {
            "models": ["vision", "language"],
            "workflow": [
                {"model": "vision", "task": "analyze_image", "share_result": True},
                {"model": "language", "task": "generate_description", "depends_on": "vision"}
            ],
            "mode": CollaborationMode.SERIAL,
            "description": "视觉分析后生成语言描述"
        }
        
        # 多模态融合模式
        self.collaboration_patterns["multimodal_fusion"] = {
            "models": ["vision", "audio", "language"],
            "workflow": [
                {"model": "vision", "task": "analyze_video", "share_result": True},
                {"model": "audio", "task": "analyze_audio", "share_result": True},
                {"model": "language", "task": "fuse_modalities", "depends_on": ["vision", "audio"]}
            ],
            "mode": CollaborationMode.PARALLEL,
            "description": "并行处理视觉和音频，然后融合结果"
        }
        
        # 知识增强模式
        self.collaboration_patterns["knowledge_enhanced"] = {
            "models": ["language", "knowledge"],
            "workflow": [
                {"model": "language", "task": "understand_query", "share_result": True},
                {"model": "knowledge", "task": "retrieve_information", "depends_on": "language"},
                {"model": "language", "task": "generate_response", "depends_on": "knowledge"}
            ],
            "mode": CollaborationMode.SERIAL,
            "description": "语言理解后检索知识，再生成响应"
        }
    
    async def initiate_collaboration(self, pattern_name: str, input_data: Any, 
                                   custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """启动协作任务
        Initiate collaboration task
        
        Args:
            pattern_name: 协作模式名称 | Collaboration pattern name
            input_data: 输入数据 | Input data
            custom_config: 自定义配置 | Custom configuration
            
        Returns:
            Dict[str, Any]: 协作结果 | Collaboration result
        """
        collaboration_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # 获取协作模式配置
            if pattern_name not in self.collaboration_patterns:
                error_msg = f"未知的协作模式: {pattern_name}"
                self.logger.error(error_msg)
                return {"error": error_msg, "collaboration_id": collaboration_id}
            
            pattern = self.collaboration_patterns[pattern_name]
            model_ids = pattern["models"]
            mode = pattern["mode"]
            
            # 创建协作配置
            config = CollaborationConfig(
                collaboration_id=collaboration_id,
                model_ids=model_ids,
                mode=mode,
                task_description=pattern["description"],
                parameters=custom_config or {},
                timeout=custom_config.get('timeout', 30.0) if custom_config else 30.0,
                max_retries=custom_config.get('max_retries', 3) if custom_config else 3
            )
            
            # 记录协作开始
            async with self.lock:
                self.active_collaborations[collaboration_id] = config
                self.performance_stats.total_collaborations += 1
            
            self.logger.info(f"启动协作任务: {collaboration_id}, 模式: {pattern_name}")
            self.logger.info(f"Initiating collaboration: {collaboration_id}, pattern: {pattern_name}")
            
            # 根据协作模式执行
            if mode == CollaborationMode.SERIAL:
                result = await self._execute_serial_collaboration(config, input_data)
            elif mode == CollaborationMode.PARALLEL:
                result = await self._execute_parallel_collaboration(config, input_data)
            elif mode == CollaborationMode.HYBRID:
                result = await self._execute_hybrid_collaboration(config, input_data)
            elif mode == CollaborationMode.CONDITIONAL:
                result = await self._execute_conditional_collaboration(config, input_data)
            else:
                error_msg = f"不支持的协作模式: {mode}"
                self.logger.error(error_msg)
                result = {"error": error_msg}
            
            # 记录协作完成
            end_time = time.time()
            execution_time = end_time - start_time
            
            async with self.lock:
                if collaboration_id in self.active_collaborations:
                    del self.active_collaborations[collaboration_id]
                
                # 更新统计信息
                self.performance_stats.execution_time += execution_time
                self.performance_stats.avg_execution_time = (
                    self.performance_stats.execution_time / 
                    self.performance_stats.total_collaborations
                )
                
                if "error" not in result:
                    self.performance_stats.success_count += 1
                else:
                    self.performance_stats.failure_count += 1
            
            # 记录历史
            collaboration_record = {
                "id": collaboration_id,
                "pattern": pattern_name,
                "start_time": start_time,
                "end_time": end_time,
                "execution_time": execution_time,
                "input_data": str(input_data)[:1000],  # 限制长度
                "result": result,
                "success": "error" not in result
            }
            self.collaboration_history.append(collaboration_record)
            
            # 保持历史记录大小
            if len(self.collaboration_history) > 1000:
                self.collaboration_history = self.collaboration_history[-1000:]
            
            return result
            
        except Exception as e:
            error_msg = f"协作任务执行失败: {str(e)}"
            self.logger.error(error_msg)
            error_handler.handle_error(e, "ModelCollaborationOrchestrator", error_msg)
            
            # 清理协作任务
            async with self.lock:
                if collaboration_id in self.active_collaborations:
                    del self.active_collaborations[collaboration_id]
                self.performance_stats.failure_count += 1
            
            return {"error": error_msg, "collaboration_id": collaboration_id}
    
    async def _execute_serial_collaboration(self, config: CollaborationConfig, 
                                          input_data: Any) -> Dict[str, Any]:
        """执行串行协作
        Execute serial collaboration
        
        Args:
            config: 协作配置 | Collaboration configuration
            input_data: 输入数据 | Input data
            
        Returns:
            Dict[str, Any]: 协作结果 | Collaboration result
        """
        intermediate_result = input_data
        execution_log = []
        
        for model_id in config.model_ids:
            try:
                model = get_model_registry().get_model(model_id)
                if not model:
                    error_msg = f"模型未找到: {model_id}"
                    self.logger.error(error_msg)
                    return {"error": error_msg, "execution_log": execution_log}
                
                # 执行模型任务
                start_time = time.time()
                if asyncio.iscoroutinefunction(model.execute_task):
                    result = await model.execute_task(intermediate_result, config.parameters)
                else:
                    result = model.execute_task(intermediate_result, config.parameters)
                end_time = time.time()
                
                # 记录执行日志
                execution_log.append({
                    "model": model_id,
                    "execution_time": end_time - start_time,
                    "success": True,
                    "timestamp": time.time()
                })
                
                # 更新中间结果
                intermediate_result = result
                
                # 共享中间结果到共享内存
                self.shared_memory[f"{config.collaboration_id}_{model_id}"] = result
                
            except Exception as e:
                error_msg = f"模型执行失败: {model_id} - {str(e)}"
                self.logger.error(error_msg)
                execution_log.append({
                    "model": model_id,
                    "error": error_msg,
                    "success": False,
                    "timestamp": time.time()
                })
                
                # 根据配置决定是否继续
                if config.parameters.get("continue_on_error", False):
                    continue
                else:
                    return {"error": error_msg, "execution_log": execution_log}
        
        return {
            "result": intermediate_result,
            "execution_log": execution_log,
            "collaboration_id": config.collaboration_id
        }
    
    async def _execute_parallel_collaboration(self, config: CollaborationConfig, 
                                            input_data: Any) -> Dict[str, Any]:
        """执行并行协作
        Execute parallel collaboration
        
        Args:
            config: 协作配置 | Collaboration configuration
            input_data: 输入数据 | Input data
            
        Returns:
            Dict[str, Any]: 协作结果 | Collaboration result
        """
        results = {}
        execution_log = []
        tasks = []
        
        # 创建并行任务
        for model_id in config.model_ids:
            model = get_model_registry().get_model(model_id)
            if not model:
                error_msg = f"模型未找到: {model_id}"
                self.logger.error(error_msg)
                results[model_id] = {"error": error_msg}
                continue
            
            # 创建异步任务
            if asyncio.iscoroutinefunction(model.execute_task):
                task = asyncio.create_task(
                    self._execute_model_task(model, model_id, input_data, config.parameters)
                )
            else:
                # 对于同步模型，使用线程池
                task = asyncio.get_event_loop().run_in_executor(
                    None, self._execute_model_task_sync, model, model_id, input_data, config.parameters
                )
            tasks.append((model_id, task))
        
        # 等待所有任务完成
        for model_id, task in tasks:
            try:
                start_time = time.time()
                result = await task
                end_time = time.time()
                
                results[model_id] = result
                execution_log.append({
                    "model": model_id,
                    "execution_time": end_time - start_time,
                    "success": "error" not in result,
                    "timestamp": time.time()
                })
                
                # 共享结果到共享内存
                self.shared_memory[f"{config.collaboration_id}_{model_id}"] = result
                
            except Exception as e:
                error_msg = f"并行任务执行失败: {model_id} - {str(e)}"
                self.logger.error(error_msg)
                results[model_id] = {"error": error_msg}
                execution_log.append({
                    "model": model_id,
                    "error": error_msg,
                    "success": False,
                    "timestamp": time.time()
                })
        
        # 合并结果
        merged_result = self._merge_parallel_results(results, config.parameters)
        
        return {
            "model_results": results,
            "merged_result": merged_result,
            "execution_log": execution_log,
            "collaboration_id": config.collaboration_id
        }
    
    async def _execute_hybrid_collaboration(self, config: CollaborationConfig, 
                                          input_data: Any) -> Dict[str, Any]:
        """执行混合协作
        Execute hybrid collaboration
        
        Args:
            config: 协作配置 | Collaboration configuration
            input_data: 输入数据 | Input data
            
        Returns:
            Dict[str, Any]: 协作结果 | Collaboration result
        """
        # 解析混合配置
        parallel_models = config.parameters.get("parallel_models", [])
        serial_models = config.parameters.get("serial_models", [])
        
        if not parallel_models and not serial_models:
            error_msg = "混合协作需要指定并行和串行模型"
            self.logger.error(error_msg)
            return {"error": error_msg}
        
        # 执行并行阶段
        parallel_config = CollaborationConfig(
            collaboration_id=config.collaboration_id + "_parallel",
            model_ids=parallel_models,
            mode=CollaborationMode.PARALLEL,
            task_description="Parallel phase of hybrid collaboration",
            parameters=config.parameters
        )
        
        parallel_result = await self._execute_parallel_collaboration(parallel_config, input_data)
        if "error" in parallel_result:
            return parallel_result
        
        # 准备串行阶段输入
        serial_input = {
            "parallel_results": parallel_result,
            "original_input": input_data
        }
        
        # 执行串行阶段
        serial_config = CollaborationConfig(
            collaboration_id=config.collaboration_id + "_serial",
            model_ids=serial_models,
            mode=CollaborationMode.SERIAL,
            task_description="Serial phase of hybrid collaboration",
            parameters=config.parameters
        )
        
        serial_result = await self._execute_serial_collaboration(serial_config, serial_input)
        if "error" in serial_result:
            return serial_result
        
        # 合并最终结果
        final_result = {
            "parallel_phase": parallel_result,
            "serial_phase": serial_result,
            "final_result": serial_result.get("result", {}),
            "timestamp": time.time()
        }
        
        return final_result
    
    async def _execute_conditional_collaboration(self, config: CollaborationConfig, 
                                               input_data: Any) -> Dict[str, Any]:
        """执行条件分支协作
        Execute conditional collaboration
        
        Args:
            config: 协作配置 | Collaboration configuration
            input_data: 输入数据 | Input data
            
        Returns:
            Dict[str, Any]: 协作结果 | Collaboration result
        """
        # 获取条件判断模型
        condition_model_id = config.parameters.get("condition_model", config.model_ids[0])
        condition_model = get_model_registry().get_model(condition_model_id)
        
        if not condition_model:
            error_msg = f"条件模型未找到: {condition_model_id}"
            self.logger.error(error_msg)
            return {"error": error_msg}
        
        try:
            # 执行条件判断
            if asyncio.iscoroutinefunction(condition_model.execute_task):
                condition_result = await condition_model.execute_task(input_data, config.parameters)
            else:
                condition_result = condition_model.execute_task(input_data, config.parameters)
            
            # 根据条件选择分支
            branch_name = self._evaluate_condition(condition_result, config.parameters)
            
            if branch_name not in config.parameters.get("branches", {}):
                error_msg = f"未知的分支: {branch_name}"
                self.logger.error(error_msg)
                return {"error": error_msg}
            
            # 执行选定分支
            branch_config = config.parameters["branches"][branch_name]
            branch_models = branch_config.get("models", [])
            branch_mode = CollaborationMode(branch_config.get("mode", "serial"))
            
            branch_collaboration_config = CollaborationConfig(
                collaboration_id=config.collaboration_id + f"_{branch_name}",
                model_ids=branch_models,
                mode=branch_mode,
                task_description=f"Branch: {branch_name}",
                parameters=branch_config
            )
            
            if branch_mode == CollaborationMode.SERIAL:
                branch_result = await self._execute_serial_collaboration(
                    branch_collaboration_config, input_data
                )
            elif branch_mode == CollaborationMode.PARALLEL:
                branch_result = await self._execute_parallel_collaboration(
                    branch_collaboration_config, input_data
                )
            else:
                error_msg = f"不支持的分支模式: {branch_mode}"
                self.logger.error(error_msg)
                return {"error": error_msg}
            
            return {
                "condition_result": condition_result,
                "selected_branch": branch_name,
                "branch_result": branch_result,
                "timestamp": time.time()
            }
            
        except Exception as e:
            error_msg = f"条件协作执行失败: {str(e)}"
            self.logger.error(error_msg)
            error_handler.handle_error(e, "ModelCollaborationOrchestrator", error_msg)
            return {"error": error_msg}
    
    async def _execute_model_task(self, model, model_id: str, input_data: Any, 
                                parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行模型任务（异步）
        Execute model task (async)
        
        Args:
            model: 模型实例 | Model instance
            model_id: 模型ID | Model ID
            input_data: 输入数据 | Input data
            parameters: 参数 | Parameters
            
        Returns:
            Dict[str, Any]: 任务结果 | Task result
        """
        try:
            result = await model.execute_task(input_data, parameters)
            return result
        except Exception as e:
            error_msg = f"模型任务执行失败: {model_id} - {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}
    
    def _execute_model_task_sync(self, model, model_id: str, input_data: Any, 
                               parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行模型任务(同步)
        Execute model task (sync)
        
        Args:
            model: 模型实例 | Model instance
            model_id: 模型ID | Model ID
            input_data: 输入数据 | Input data
            parameters: 参数 | Parameters
            
        Returns:
            Dict[str, Any]: 任务结果 | Task result
        """
        try:
            result = model.execute_task(input_data, parameters)
            return result
        except Exception as e:
            error_msg = f"模型任务执行失败: {model_id} - {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}
    
    def _merge_parallel_results(self, results: Dict[str, Any], 
                              parameters: Dict[str, Any]) -> Dict[str, Any]:
        """合并并行处理的结果
        Merge parallel processing results
        
        Args:
            results: 各模型结果 | Model results
            parameters: 合并参数 | Merge parameters
            
        Returns:
            Dict[str, Any]: 合并后的结果 | Merged result
        """
        merge_strategy = parameters.get("merge_strategy", "combine")
        
        if merge_strategy == "combine":
            # 简单合并所有结果
            return results
        
        elif merge_strategy == "vote":
            # 投票机制（适用于分类任务）
            return self._merge_by_voting(results)
        
        elif merge_strategy == "weighted_average":
            # 加权平均（适用于数值结果）
            return self._merge_by_weighted_average(results, parameters)
        
        elif merge_strategy == "confidence_based":
            # 基于置信度的合并
            return self._merge_by_confidence(results)
        
        else:
            self.logger.warning(f"未知的合并策略: {merge_strategy}, 使用默认合并")
            return results
    
    def _merge_by_voting(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """通过投票合并结果
        Merge results by voting
        
        Args:
            results: 各模型结果 | Model results
            
        Returns:
            Dict[str, Any]: 投票结果 | Voting result
        """
        # 实现简单的投票机制
        votes = {}
        for model_id, result in results.items():
            if "error" not in result and "prediction" in result:
                prediction = result["prediction"]
                if prediction not in votes:
                    votes[prediction] = 0
                votes[prediction] += 1
        
        if votes:
            # 选择票数最多的预测
            selected_prediction = max(votes.items(), key=lambda x: x[1])[0]
            return {
                "merged_prediction": selected_prediction,
                "vote_counts": votes,
                "total_votes": sum(votes.values())
            }
        else:
            return {"error": "无法进行投票合并，无有效预测"}
    
    def _merge_by_weighted_average(self, results: Dict[str, Any], 
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """通过加权平均合并结果
        Merge results by weighted average
        
        Args:
            results: 各模型结果 | Model results
            parameters: 权重参数 | Weight parameters
            
        Returns:
            Dict[str, Any]: 加权平均结果 | Weighted average result
        """
        weights = parameters.get("weights", {})
        default_weight = parameters.get("default_weight", 1.0)
        
        weighted_sum = 0.0
        total_weight = 0.0
        valid_results = 0
        
        for model_id, result in results.items():
            if "error" not in result and "value" in result:
                weight = weights.get(model_id, default_weight)
                weighted_sum += result["value"] * weight
                total_weight += weight
                valid_results += 1
        
        if valid_results > 0 and total_weight > 0:
            return {
                "weighted_average": weighted_sum / total_weight,
                "total_weight": total_weight,
                "valid_results": valid_results
            }
        else:
            return {"error": "无法进行加权平均合并，无有效数值结果"}
    
    def _merge_by_confidence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """通过置信度合并结果
        Merge results by confidence
        
        Args:
            results: 各模型结果 | Model results
            
        Returns:
            Dict[str, Any]: 置信度合并结果 | Confidence-based merged result
        """
        best_result = None
        best_confidence = -1.0
        
        for model_id, result in results.items():
            if "error" not in result and "confidence" in result:
                confidence = result["confidence"]
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = result
        
        if best_result is not None:
            return {
                "merged_result": best_result,
                "selected_by_confidence": best_confidence
            }
        else:
            return {"error": "无法进行置信度合并，无有效置信度数据"}
    
    def _evaluate_condition(self, condition_result: Any, 
                          parameters: Dict[str, Any]) -> str:
        """评估条件结果
        Evaluate condition result
        
        Args:
            condition_result: 条件判断结果 | Condition result
            parameters: 条件参数 | Condition parameters
            
        Returns:
            str: 分支名称 | Branch name
        """
        # 默认实现：根据条件结果的类型选择分支
        if isinstance(condition_result, dict):
            if "decision" in condition_result:
                return condition_result["decision"]
            elif "branch" in condition_result:
                return condition_result["branch"]
        
        # 尝试从参数中获取条件映射
        condition_mapping = parameters.get("condition_mapping", {})
        for branch_name, condition_value in condition_mapping.items():
            if condition_result == condition_value:
                return branch_name
        
        # 默认分支
        return parameters.get("default_branch", "default")
    
    def register_collaboration_pattern(self, pattern_name: str, 
                                     pattern_config: Dict[str, Any]) -> bool:
        """注册协作模式
        Register collaboration pattern
        
        Args:
            pattern_name: 模式名称 | Pattern name
            pattern_config: 模式配置 | Pattern configuration
            
        Returns:
            bool: 是否成功注册 | Whether registration was successful
        """
        try:
            # 验证模式配置
            required_fields = ["models", "workflow", "mode", "description"]
            for field in required_fields:
                if field not in pattern_config:
                    error_msg = f"协作模式缺少必需字段: {field}"
                    self.logger.error(error_msg)
                    return False
            
            self.collaboration_patterns[pattern_name] = pattern_config
            self.logger.info(f"协作模式已注册: {pattern_name}")
            self.logger.info(f"Collaboration pattern registered: {pattern_name}")
            return True
            
        except Exception as e:
            error_msg = f"注册协作模式失败: {pattern_name} - {str(e)}"
            self.logger.error(error_msg)
            error_handler.handle_error(e, "ModelCollaborationOrchestrator", error_msg)
            return False
    
    def get_collaboration_status(self, collaboration_id: str = None) -> Dict[str, Any]:
        """获取协作状态
        Get collaboration status
        
        Args:
            collaboration_id: 可选协作ID | Optional collaboration ID
            
        Returns:
            Dict[str, Any]: 状态信息 | Status information
        """
        try:
            if collaboration_id:
                if collaboration_id in self.active_collaborations:
                    config = self.active_collaborations[collaboration_id]
                    return {
                        "id": collaboration_id,
                        "status": "active",
                        "mode": config.mode.value,
                        "models": config.model_ids,
                        "start_time": config.parameters.get("start_time", time.time())
                    }
                else:
                    # 检查历史记录
                    for record in self.collaboration_history:
                        if record["id"] == collaboration_id:
                            return {
                                "id": collaboration_id,
                                "status": "completed",
                                "execution_time": record["execution_time"],
                                "success": record["success"]
                            }
                    return {"status": "not_found"}
            else:
                return {
                    "active_collaborations": len(self.active_collaborations),
                    "total_collaborations": self.performance_stats.total_collaborations,
                    "success_rate": (
                        self.performance_stats.success_count / 
                        self.performance_stats.total_collaborations * 100
                        if self.performance_stats.total_collaborations > 0 else 0
                    ),
                    "avg_execution_time": self.performance_stats.avg_execution_time
                }
                
        except Exception as e:
            error_msg = f"获取协作状态失败: {str(e)}"
            self.logger.error(error_msg)
            error_handler.handle_error(e, "ModelCollaborationOrchestrator", error_msg)
            return {"error": error_msg}
    
    def get_shared_memory(self, key: str = None) -> Any:
        """获取共享内存数据
        Get shared memory data
        
        Args:
            key: 可选键 | Optional key
            
        Returns:
            Any: 共享数据 | Shared data
        """
        if key:
            return self.shared_memory.get(key)
        else:
            # 返回所有共享内存数据的副本
            return dict(self.shared_memory)
    

    
    def add_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """添加性能指标
        Add performance metrics
        
        Args:
            metrics: 性能指标数据 | Performance metrics data
        """
        try:
            asyncio.run(self._add_performance_metrics_async(metrics))
        except RuntimeError:
            # 如果已经在事件循环中
            asyncio.create_task(self._add_performance_metrics_async(metrics))
    
    async def _add_performance_metrics_async(self, metrics: Dict[str, Any]) -> None:
        """异步添加性能指标
        Add performance metrics asynchronously
        
        Args:
            metrics: 性能指标数据 | Performance metrics data
        """
        async with self.lock:
            # 更新性能统计
            if "success" in metrics and metrics["success"]:
                self.performance_stats.success_count += 1
            elif "success" in metrics:
                self.performance_stats.failure_count += 1
            
            if "execution_time" in metrics:
                self.performance_stats.execution_time += metrics["execution_time"]
                self.performance_stats.avg_execution_time = (
                    self.performance_stats.execution_time / 
                    self.performance_stats.total_collaborations
                    if self.performance_stats.total_collaborations > 0 else 0
                )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告
        Get performance report

        Returns:
            Dict[str, Any]: 性能报告数据 | Performance report data
        """
        return {
            "total_collaborations": self.performance_stats.total_collaborations,
            "success_count": self.performance_stats.success_count,
            "failure_count": self.performance_stats.failure_count,
            "success_rate": (
                self.performance_stats.success_count / 
                self.performance_stats.total_collaborations * 100
                if self.performance_stats.total_collaborations > 0 else 0
            ),
            "avg_execution_time": self.performance_stats.avg_execution_time,
            "active_collaborations": len(self.active_collaborations)
        }
    
    def clear_shared_memory(self, key: str = None) -> bool:
        """清除共享内存数据
        Clear shared memory data
        
        Args:
            key: 可选键 | Optional key
            
        Returns:
            bool: 是否成功清除 | Whether clearance was successful
        """
        try:
            if key:
                if key in self.shared_memory:
                    del self.shared_memory[key]
                    return True
                return False
            else:
                self.shared_memory.clear()
                return True
                
        except Exception as e:
            error_msg = f"清除共享内存失败: {str(e)}"
            self.logger.error(error_msg)
            error_handler.handle_error(e, "ModelCollaborationOrchestrator", error_msg)
            return False

# 创建全局协作协调器实例
collaboration_orchestrator = ModelCollaborationOrchestrator()

# 启动函数
async def start_collaboration_orchestrator():
    """启动协作协调器
    Start collaboration orchestrator
    """
    # 目前不需要特殊的启动逻辑
    collaboration_orchestrator.logger.info("协作协调器已就绪")
    collaboration_orchestrator.logger.info("Collaboration orchestrator ready")

# 停止函数
async def stop_collaboration_orchestrator():
    """停止协作协调器
    Stop collaboration orchestrator
    """
    # 清理资源
    collaboration_orchestrator.active_collaborations.clear()
    collaboration_orchestrator.logger.info("协作协调器已停止")
    collaboration_orchestrator.logger.info("Collaboration orchestrator stopped")

# 为了向后兼容，提供别名
ModelCollaborator = ModelCollaborationOrchestrator
