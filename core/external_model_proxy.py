"""
外部模型代理 - External Model Proxy
用于连接和管理外部API模型，提供统一的接口供系统调用
For connecting and managing external API models, providing unified interface for system calls
"""

import logging
import json
import time
from typing import Dict, Any, Optional, Union, List
from core.error_handling import error_handler
from core.api_model_connector import APIModelConnector
from core.external_api_service import ExternalAPIService

# Cycle prevention for safe text generation
try:
    from .cycle_prevention_manager import CyclePreventionManager
except ImportError:
    from core.cycle_prevention_manager import CyclePreventionManager


class ExternalModelProxy:
    """外部模型代理类 | External Model Proxy Class
    
    功能：代理外部API模型，为系统调用提供统一接口
    Function: Proxy external API models, providing unified interface for system calls
    """
    
    def __init__(self, enable_cycle_prevention: bool = True):
        """初始化外部模型代理 | Initialize external model proxy
        
        Args:
            enable_cycle_prevention: 是否启用防循环保护
        """
        self.logger = logging.getLogger(__name__)
        self.api_connector = APIModelConnector()
        self.api_service = ExternalAPIService()
        
        # 缓存已连接的模型
        self.connected_models: Dict[str, Dict[str, Any]] = {}
        
        # 防循环管理器
        self.enable_cycle_prevention = enable_cycle_prevention
        if self.enable_cycle_prevention:
            try:
                self.cycle_prevention_manager = CyclePreventionManager(
                    config={
                        "history_buffer_size": 10,
                        "repeat_threshold": 3,
                        "base_temperature": 0.7,
                        "max_temperature": 1.2,
                        "base_repetition_penalty": 1.1,
                        "max_repetition_penalty": 1.8,
                    },
                    enable_adaptive_layer=True
                )
                self.logger.info("ExternalModelProxy: Cycle prevention manager initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize cycle prevention manager: {e}")
                self.cycle_prevention_manager = None
        else:
            self.cycle_prevention_manager = None
        
        self.logger.info("外部模型代理初始化完成 | External model proxy initialized")
    
    async def inference(self, model_id: str, input_data: Union[str, Dict[str, Any], List[Any]], 
                       config: Dict[str, Any] = None) -> Dict[str, Any]:
        """外部模型推理接口 | External model inference interface
        
        Args:
            model_id: 模型ID
            input_data: 输入数据
            config: 推理配置
            
        Returns:
            推理结果
        """
        try:
            # 确保模型已连接
            if model_id not in self.connected_models:
                connect_result = self.api_connector.connect(model_id)
                if not connect_result.get("success", False):
                    return {
                        "success": False,
                        "message": f"无法连接到模型 {model_id}: {connect_result.get('message', 'Unknown error')}",
                        "model_id": model_id
                    }
                self.connected_models[model_id] = connect_result
            
            # 构建推理请求参数
            inference_params = {
                "input": input_data
            }
            if config:
                inference_params.update(config)
            
            # 调用外部API
            result = self.api_connector.call_model(
                model_id=model_id,
                method="inference",
                params=inference_params
            )
            
            if result.get("success", False):
                return {
                    "success": True,
                    "result": result.get("result", {}),
                    "model_id": model_id,
                    "response_time": result.get("_model_metadata", {}).get("response_time", 0)
                }
            else:
                return {
                    "success": False,
                    "message": f"推理失败: {result.get('message', 'Unknown error')}",
                    "model_id": model_id
                }
                
        except Exception as e:
            error_handler.handle_error(e, "ExternalModelProxy", f"推理过程中出错: {str(e)}")
            return {
                "success": False,
                "message": f"推理异常: {str(e)}",
                "model_id": model_id
            }
    
    async def chat(self, model_id: str, message: str, conversation_id: Optional[str] = None,
                  config: Dict[str, Any] = None) -> Dict[str, Any]:
        """外部模型对话接口 | External model chat interface
        
        Args:
            model_id: 模型ID
            message: 消息内容
            conversation_id: 对话ID（可选）
            config: 对话配置
            
        Returns:
            对话结果
        """
        try:
            # 确保模型已连接
            if model_id not in self.connected_models:
                connect_result = self.api_connector.connect(model_id)
                if not connect_result.get("success", False):
                    return {
                        "success": False,
                        "message": f"无法连接到模型 {model_id}: {connect_result.get('message', 'Unknown error')}",
                        "model_id": model_id
                    }
                self.connected_models[model_id] = connect_result
            
            # 构建对话请求参数
            chat_params = {
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                "conversation_id": conversation_id
            }
            if config:
                chat_params.update(config)
            
            # 调用外部API
            result = self.api_connector.call_model(
                model_id=model_id,
                method="chat/completions",
                params=chat_params
            )
            
            if result.get("success", False):
                response_data = result.get("result", {})
                
                # 提取回复内容
                if isinstance(response_data, dict):
                    choices = response_data.get("choices", [])
                    if choices:
                        message_content = choices[0].get("message", {}).get("content", "")
                    else:
                        message_content = response_data.get("text", "")
                else:
                    message_content = str(response_data)
                
                return {
                    "success": True,
                    "response": message_content,
                    "conversation_id": conversation_id,
                    "model_id": model_id,
                    "response_time": result.get("_model_metadata", {}).get("response_time", 0)
                }
            else:
                return {
                    "success": False,
                    "message": f"对话失败: {result.get('message', 'Unknown error')}",
                    "model_id": model_id
                }
                
        except Exception as e:
            error_handler.handle_error(e, "ExternalModelProxy", f"对话过程中出错: {str(e)}")
            return {
                "success": False,
                "message": f"对话异常: {str(e)}",
                "model_id": model_id
            }
    
    async def generate(self, model_id: str, prompt: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """外部模型生成接口 | External model generation interface
        
        Args:
            model_id: 模型ID
            prompt: 生成提示
            config: 生成配置
            
        Returns:
            生成结果
        """
        try:
            # 确保模型已连接
            if model_id not in self.connected_models:
                connect_result = self.api_connector.connect(model_id)
                if not connect_result.get("success", False):
                    return {
                        "success": False,
                        "message": f"无法连接到模型 {model_id}: {connect_result.get('message', 'Unknown error')}",
                        "model_id": model_id
                    }
                self.connected_models[model_id] = connect_result
            
            # 构建生成请求参数
            generate_params = {
                "prompt": prompt
            }
            if config:
                generate_params.update(config)
            
            # 调用外部API
            result = self.api_connector.call_model(
                model_id=model_id,
                method="completions",
                params=generate_params
            )
            
            if result.get("success", False):
                response_data = result.get("result", {})
                
                # 提取生成内容
                if isinstance(response_data, dict):
                    choices = response_data.get("choices", [])
                    if choices:
                        generated_text = choices[0].get("text", "")
                    else:
                        generated_text = response_data.get("text", "")
                else:
                    generated_text = str(response_data)
                
                return {
                    "success": True,
                    "generated_text": generated_text,
                    "model_id": model_id,
                    "response_time": result.get("_model_metadata", {}).get("response_time", 0)
                }
            else:
                return {
                    "success": False,
                    "message": f"生成失败: {result.get('message', 'Unknown error')}",
                    "model_id": model_id
                }
                
        except Exception as e:
            error_handler.handle_error(e, "ExternalModelProxy", f"生成过程中出错: {str(e)}")
            return {
                "success": False,
                "message": f"生成异常: {str(e)}",
                "model_id": model_id
            }
    
    async def generate_safe(self, model_id: str, prompt: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        安全生成接口 - 使用简化防循环逻辑包装外部模型生成
        
        简化防循环逻辑（针对外部API优化）：
        1. 缓冲区清理：只保留最近5轮对话
        2. 重复检测：连续2次相同输出触发防护
        3. 参数调整：检测到循环时调整temperature和repetition_penalty
        4. 重试机制：最多重试2次
        
        Args:
            model_id: 模型ID
            prompt: 生成提示
            config: 生成配置
            
        Returns:
            生成结果（包含防护信息）
        """
        # 检查是否启用防循环
        if not self.enable_cycle_prevention:
            # 如果不启用防循环，直接使用原始generate方法
            return await self.generate(model_id, prompt, config)
        
        try:
            import re
            from collections import deque
            
            # 初始化防循环参数
            history_buffer = deque(maxlen=5)  # 只保留最近5轮
            last_outputs = deque(maxlen=2)    # 用于检测重复
            
            # 初始参数
            current_config = config.copy() if config else {}
            temperature = current_config.get("temperature", 0.7)
            repetition_penalty = current_config.get("repetition_penalty", 1.1)
            
            # 更新历史
            history_buffer.append(prompt)
            
            # 尝试生成（最多重试2次）
            max_attempts = 2
            for attempt in range(max_attempts):
                # 构建生成参数
                generate_config = current_config.copy()
                
                # 如果是重试，调整参数
                if attempt > 0:
                    # 增加温度（增加随机性）
                    temperature = min(1.2, temperature + 0.1)
                    generate_config["temperature"] = temperature
                    
                    # 增加重复惩罚
                    repetition_penalty = min(1.8, repetition_penalty + 0.05)
                    generate_config["repetition_penalty"] = repetition_penalty
                    
                    self.logger.info(f"防循环重试 {attempt}: 调整参数 temperature={temperature}, repetition_penalty={repetition_penalty}")
                
                # 生成文本
                result = await self.generate(model_id, prompt, generate_config)
                
                if not result.get("success", False):
                    # 生成失败，直接返回
                    result["protection_layer"] = "failed_no_protection"
                    return result
                
                generated_text = result.get("generated_text", "")
                
                # 清理文本用于重复检测
                def clean_text(text: str) -> str:
                    if not text:
                        return ""
                    # 移除标点符号和空格，只保留核心内容
                    return re.sub(r'[^\w\s]', '', text).strip().lower()
                
                clean_generated = clean_text(generated_text)
                
                # 检查是否重复
                is_repeat = False
                if clean_generated and len(clean_generated) > 10:  # 只检查足够长的文本
                    for last_output in last_outputs:
                        if clean_text(last_output) == clean_generated:
                            is_repeat = True
                            break
                
                if is_repeat and attempt < max_attempts - 1:
                    # 检测到重复，将输出加入队列并重试
                    last_outputs.append(generated_text)
                    self.logger.warning(f"检测到重复输出，进行第{attempt+1}次重试")
                    continue
                else:
                    # 无重复或已达最大重试次数，返回结果
                    history_buffer.append(generated_text)
                    last_outputs.append(generated_text)
                    
                    protection_info = {
                        "attempts": attempt + 1,
                        "temperature": temperature,
                        "repetition_penalty": repetition_penalty,
                        "repeat_detected": is_repeat,
                        "history_size": len(history_buffer),
                        "last_outputs_count": len(last_outputs)
                    }
                    
                    result.update({
                        "protection_layer": "simplified_cycle_prevention",
                        "protection_info": protection_info
                    })
                    return result
            
            # 理论上不会执行到这里，但为了安全起见
            return await self.generate(model_id, prompt, config)
            
        except Exception as e:
            error_handler.handle_error(e, "ExternalModelProxy", f"安全生成过程中出错: {str(e)}")
            # 降级到原始generate方法
            fallback_result = await self.generate(model_id, prompt, config)
            if fallback_result.get("success", False):
                fallback_result["protection_layer"] = "fallback_no_protection"
            return fallback_result
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """获取模型状态 | Get model status
        
        Args:
            model_id: 模型ID
            
        Returns:
            模型状态信息
        """
        try:
            # 检查连接状态
            if model_id in self.connected_models:
                connection = self.connected_models[model_id]
                
                # 测试连接
                test_result = self.api_connector.call_model(
                    model_id=model_id,
                    method="status",
                    params={}
                )
                
                return {
                    "model_id": model_id,
                    "status": "connected",
                    "connected_at": connection.get("connected_at", time.time()),
                    "api_url": connection.get("api_url", ""),
                    "model_name": connection.get("model_name", ""),
                    "last_test": time.time(),
                    "test_result": test_result.get("success", False)
                }
            else:
                return {
                    "model_id": model_id,
                    "status": "disconnected",
                    "message": "模型未连接"
                }
                
        except Exception as e:
            error_handler.handle_error(e, "ExternalModelProxy", f"获取模型状态失败: {str(e)}")
            return {
                "model_id": model_id,
                "status": "error",
                "message": f"状态检查失败: {str(e)}"
            }
    
    def disconnect(self, model_id: str) -> Dict[str, Any]:
        """断开模型连接 | Disconnect model
        
        Args:
            model_id: 模型ID
            
        Returns:
            断开连接结果
        """
        try:
            if model_id in self.connected_models:
                del self.connected_models[model_id]
                
                # 通知API连接器断开连接
                # 注意：APIModelConnector目前没有disconnect方法，需要后续实现
                
                return {
                    "success": True,
                    "message": f"已断开模型 {model_id} 的连接"
                }
            else:
                return {
                    "success": True,
                    "message": f"模型 {model_id} 未连接"
                }
                
        except Exception as e:
            error_handler.handle_error(e, "ExternalModelProxy", f"断开连接失败: {str(e)}")
            return {
                "success": False,
                "message": f"断开连接异常: {str(e)}"
            }
    
    def list_connected_models(self) -> Dict[str, Any]:
        """列出已连接的模型 | List connected models
        
        Returns:
            已连接模型列表
        """
        try:
            connected_list = []
            
            for model_id, connection in self.connected_models.items():
                status = self.get_model_status(model_id)
                connected_list.append({
                    "model_id": model_id,
                    "api_url": connection.get("api_url", ""),
                    "model_name": connection.get("model_name", ""),
                    "connected_at": connection.get("connected_at", 0),
                    "status": status.get("status", "unknown")
                })
            
            return {
                "success": True,
                "connected_models": connected_list,
                "total_count": len(connected_list)
            }
            
        except Exception as e:
            error_handler.handle_error(e, "ExternalModelProxy", f"列出连接模型失败: {str(e)}")
            return {
                "success": False,
                "message": f"列出连接模型异常: {str(e)}",
                "connected_models": []
            }


# 导出类
ExternalModelProxy = ExternalModelProxy