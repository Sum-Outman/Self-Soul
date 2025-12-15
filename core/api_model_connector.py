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
API模型连接器：负责连接和调用外部API模型
"""
import requests
import json
import time
from typing import Dict, Any, Optional
from .error_handling import error_handler
from .system_settings_manager import system_settings_manager


"""
APIModelConnector类 - 中文类描述
APIModelConnector Class - English class description
"""
class APIModelConnector:
    """API模型连接器类，负责管理与外部API模型的连接和通信"""
    
    def __init__(self):
        # 连接池缓存
        self.connections = {}
        # 请求超时设置
        self.timeout = 10
        # 重试次数
        self.max_retries = 3
        # 请求间隔（防止速率限制）
        self.request_interval = 0.5
        # 上次请求时间记录
        self.last_request_time = {}
    
    def connect(self, model_id: str) -> Dict[str, Any]:
        """
        连接到指定的外部API模型
        :param model_id: 模型ID
        :return: 连接结果
        """
        try:
            # 检查模型是否配置为API模式
            if not system_settings_manager.is_api_model(model_id):
                return {"success": False, "message": f"模型 {model_id} 未配置为API模式"}
            
            # 获取API配置
            api_config = system_settings_manager.get_model_api_config(model_id)
            api_url = api_config["api_url"]
            api_key = api_config["api_key"]
            model_name = api_config.get("model_name", "")
            source = api_config.get("source", "external")
            endpoint = api_config.get("endpoint", "")
            
            # 验证核心配置
            if not api_url or not api_key:
                return {"success": False, "message": f"模型 {model_id} 的API配置不完整（缺少URL或密钥）"}
            
            # 使用endpoint覆盖api_url（如果提供）
            if endpoint:
                api_url = endpoint
            
            # 检查连接是否已存在
            if model_id in self.connections:
                return {"success": True, "message": f"已连接到模型 {model_id}", "connection_id": model_id, "config": api_config}
            
            # 测试连接
            test_result = self._test_connection(api_url, api_key, model_name)
            if not test_result["success"]:
                return test_result
            
            # 创建连接并存储完整配置
            self.connections[model_id] = {
                "api_url": api_url,
                "api_key": api_key,
                "model_name": model_name,
                "source": source,
                "connected_at": time.time(),
                "status": "connected"
            }
            
            error_handler.log_info(f"成功连接到API模型: {model_id} (模型名称: {model_name}), API端点: {api_url}", "APIModelConnector")
            return {"success": True, "message": f"成功连接到模型 {model_id}", "connection_id": model_id, "config": api_config}
        
        except Exception as e:
            error_handler.handle_error(e, "APIModelConnector", f"连接模型 {model_id} 失败")
            return {"success": False, "message": f"连接失败: {str(e)}"}
    
    def _test_connection(self, api_url: str, api_key: str, model_name: str = None) -> Dict[str, Any]:
        """
        测试API连接 - 支持多种API类型测试
        Test API connection - Supports multiple API type testing
        :param api_url: API端点URL | API endpoint URL
        :param api_key: API密钥 | API key
        :param model_name: 模型名称 | Model name
        :return: 测试结果 | Test result
        """
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # 尝试多种常见的API测试方法 | Try multiple common API testing methods
            test_methods = [
                lambda url, h: self._test_openai_style_api(url, h, model_name),
                self._test_huggingface_style_api,
                self._test_generic_api,
                self._test_simple_endpoint
            ]
            
            for test_method in test_methods:
                result = test_method(api_url, headers)
                if result["success"]:
                    return result
            
            return {"success": False, "message": "所有API测试方法都失败 | All API test methods failed"}
        
        except Exception as e:
            return {"success": False, "message": f"连接测试异常: {str(e)} | Connection test exception: {str(e)}"}
    
    def _test_openai_style_api(self, api_url: str, headers: Dict[str, str], model_name: str = None) -> Dict[str, Any]:
        """测试OpenAI风格的API | Test OpenAI-style API"""
        try:
            # 尝试列出模型端点 | Try models endpoint
            models_url = f"{api_url}/models"
            response = requests.get(models_url, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                return {"success": True, "message": "OpenAI风格API连接成功 | OpenAI-style API connection successful"}
        except:
            pass
        
        # 尝试聊天完成端点 | Try chat completion endpoint
        try:
            chat_url = f"{api_url}/chat/completions"
            # 使用提供的模型名称或默认值
            model_to_use = model_name if model_name else "gpt-3.5-turbo"
            test_payload = {
                "model": model_to_use,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5
            }
            response = requests.post(chat_url, json=test_payload, headers=headers, timeout=self.timeout)
            
            if response.status_code in [200, 201]:
                return {"success": True, "message": f"OpenAI风格API连接成功，使用模型: {model_to_use} | OpenAI-style API connection successful, using model: {model_to_use}"}
            elif response.status_code == 401:
                return {"success": False, "message": "API密钥无效 | Invalid API key"}
            elif response.status_code == 404:
                return {"success": False, "message": "API端点不存在 | API endpoint not found"}
            elif response.status_code == 400 and "model_not_found" in response.text:
                return {"success": False, "message": f"模型不存在: {model_to_use} | Model not found: {model_to_use}"}
        except:
            pass
        
        return {"success": False, "message": "OpenAI风格API测试失败 | OpenAI-style API test failed"}
    
    def _test_huggingface_style_api(self, api_url: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """测试HuggingFace风格的API | Test HuggingFace-style API"""
        try:
            # 尝试推理端点 | Try inference endpoint
            inference_url = f"{api_url}"
            test_payload = {"inputs": "Hello, how are you?"}
            response = requests.post(inference_url, json=test_payload, headers=headers, timeout=self.timeout)
            
            if response.status_code in [200, 201]:
                return {"success": True, "message": "HuggingFace风格API连接成功 | HuggingFace-style API connection successful"}
            elif response.status_code == 401:
                return {"success": False, "message": "API密钥无效 | Invalid API key"}
            elif response.status_code == 404:
                return {"success": False, "message": "API端点不存在 | API endpoint not found"}
        except:
            pass
        
        return {"success": False, "message": "HuggingFace风格API测试失败 | HuggingFace-style API test failed"}
    
    def _test_generic_api(self, api_url: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """测试通用API | Test generic API"""
        try:
            # 尝试简单的GET请求 | Try simple GET request
            response = requests.get(api_url, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                return {"success": True, "message": "通用API连接成功 | Generic API connection successful"}
            
            # 尝试简单的POST请求 | Try simple POST request
            response = requests.post(api_url, json={"test": True}, headers=headers, timeout=self.timeout)
            
            if response.status_code in [200, 201]:
                return {"success": True, "message": "通用API连接成功 | Generic API connection successful"}
        except:
            pass
        
        return {"success": False, "message": "通用API测试失败 | Generic API test failed"}
    
    def _test_simple_endpoint(self, api_url: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """测试简单端点 | Test simple endpoint"""
        try:
            # 尝试/test端点 | Try /test endpoint
            test_url = f"{api_url}/test"
            response = requests.post(test_url, json={"test": True}, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                return {"success": True, "message": "测试端点连接成功 | Test endpoint connection successful"}
        except:
            pass
        
        return {"success": False, "message": "简单端点测试失败 | Simple endpoint test failed"}
    
    def call_model(self, model_id: str, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        调用外部API模型的方法
        :param model_id: 模型ID
        :param method: 要调用的方法名
        :param params: 方法参数
        :return: 调用结果
        """
        try:
            # 检查连接是否存在
            if model_id not in self.connections:
                # 尝试自动连接
                connect_result = self.connect(model_id)
                if not connect_result["success"]:
                    return connect_result
            
            # 获取连接配置
            connection = self.connections[model_id]
            api_url = connection["api_url"]
            api_key = connection["api_key"]
            model_name = connection.get("model_name", "")
            
            # 速率限制控制
            self._rate_limit_control(model_id)
            
            # 构建请求
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建请求URL
            request_url = f"{api_url}/{method}"
            
            # 如果参数中没有model字段但我们有model_name，添加它
            if params is None:
                params = {}
            if "model" not in params and model_name:
                params["model"] = model_name
                error_handler.log_debug(f"为API调用添加模型名称: {model_name}", "APIModelConnector")
            
            # 发送请求（带重试）
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        request_url,
                        json=params or {},
                        headers=headers,
                        timeout=self.timeout
                    )
                    
                    # 更新最后请求时间
                    self.last_request_time[model_id] = time.time()
                    
                    # 检查响应
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            # 添加模型元数据到结果
                            result["_model_metadata"] = {
                                "model_id": model_id,
                                "model_name": model_name,
                                "api_url": api_url,
                                "response_time": time.time() - self.last_request_time[model_id]
                            }
                            return {"success": True, "result": result}
                        except json.JSONDecodeError:
                            return {"success": True, "result": response.text}
                    elif response.status_code == 429:  # 速率限制
                        wait_time = min(2 ** attempt, 10)  # 指数退避
                        time.sleep(wait_time)
                        continue  # 重试
                    else:
                        return {"success": False, "message": f"API调用失败: HTTP {response.status_code} - {response.text}"}
                except requests.exceptions.Timeout:
                    if attempt == self.max_retries - 1:
                        return {"success": False, "message": "API调用超时"}
                    time.sleep(1)  # 等待1秒后重试
                except requests.exceptions.RequestException as e:
                    return {"success": False, "message": f"API调用异常: {str(e)}"}
            
            return {"success": False, "message": "API调用失败，已达到最大重试次数"}
        
        except Exception as e:
            error_handler.handle_error(e, "APIModelConnector", f"调用模型 {model_id} 方法 {method} 失败")
            return {"success": False, "message": f"调用失败: {str(e)}"}
    
    
    def _rate_limit_control(self, model_id: str):
        """_rate_limit_control函数 - 中文函数描述
        _rate_limit_control Function - English function description
        
        速率限制控制
        :param model_id: 模型ID
        """
        if model_id in self.last_request_time:
            elapsed = time.time() - self.last_request_time[model_id]
            if elapsed < self.request_interval:
                time.sleep(self.request_interval - elapsed)
    
    def disconnect(self, model_id: str) -> bool:
        """
        断开与外部API模型的连接
        :param model_id: 模型ID
        :return: 是否断开成功
        """
        if model_id in self.connections:
            del self.connections[model_id]
            if model_id in self.last_request_time:
                del self.last_request_time[model_id]
            error_handler.log_info(f"已断开与API模型的连接: {model_id}", "APIModelConnector")
            return True
        return False
    
    def get_connection_status(self, model_id: str) -> Dict[str, Any]:
        """
        获取连接状态
        :param model_id: 模型ID
        :return: 连接状态信息
        """
        if model_id in self.connections:
            connection = self.connections[model_id]
            return {
                "status": connection["status"],
                "connected_at": connection["connected_at"],
                "api_url": connection["api_url"],
                "model_name": connection.get("model_name", ""),
                "source": connection.get("source", "external")
            }
        return {"status": "disconnected"}
    
    def get_all_connections(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有活动连接
        :return: 所有连接信息
        """
        result = {}
        for model_id, connection in self.connections.items():
            result[model_id] = {
                "status": connection["status"],
                "connected_at": connection["connected_at"],
                "api_url": connection["api_url"],
                "model_name": connection.get("model_name", ""),
                "source": connection.get("source", "external")
            }
        return result
    
    def execute_task(self, model_id: str, task_description: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行任务（适配与本地模型相同的接口）
        :param model_id: 模型ID
        :param task_description: 任务描述
        :return: 任务执行结果
        """
        # 将任务转换为API调用
        method = "execute_task"
        params = {
            "task": task_description
        }
        
        # 调用API
        result = self.call_model(model_id, method, params)
        
        # 转换结果格式以匹配本地模型的输出
        if result["success"]:
            return result["result"]
        else:
            return {"error": result["message"]}
    
    def generate_response(self, message: str, conversation_history: list = None) -> str:
        """
        生成响应（特别用于语言模型的对话）
        :param message: 用户消息
        :param conversation_history: 对话历史
        :return: 生成的响应文本
        """
        try:
            # 获取连接配置
            if "language" not in self.connections:
                # 尝试自动连接
                connect_result = self.connect("language")
                if not connect_result.get("success", False):
                    error_handler.log_error(f"无法连接到语言模型: {connect_result.get('message', 'Unknown error')}", "APIModelConnector")
                    return "Failed to connect to language model"
            
            connection = self.connections["language"]
            api_type = connection.get("api_type", "generic")
            model_name = connection.get("model_name", "gpt-3.5-turbo")
            
            # 根据API类型格式化请求
            if api_type == "openai" or model_name.startswith("gpt-"):
                # OpenAI风格的API
                messages = []
                if conversation_history:
                    messages = conversation_history.copy()
                messages.append({"role": "user", "content": message})
                
                params = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": 1000
                }
                
                result = self.call_model("language", "chat/completions", params)
                
                if result["success"]:
                    return result["result"].get("choices", [{}])[0].get("message", {}).get("content", "No response")
            elif api_type == "huggingface":
                # HuggingFace风格的API
                params = {
                    "inputs": message,
                    "parameters": {
                        "max_new_tokens": 1000
                    }
                }
                
                result = self.call_model("language", "", params)
                
                if result["success"]:
                    if isinstance(result["result"], list) and len(result["result"]) > 0:
                        return result["result"][0].get("generated_text", "No response")
                    return str(result["result"])
            else:
                # 通用API
                params = {
                    "message": message,
                    "conversation_history": conversation_history or [],
                    "model_name": model_name
                }
                
                result = self.call_model("language", "generate", params)
                
                if result["success"]:
                    return result["result"].get("response", "No response")
            
            error_handler.log_error(f"无法生成响应: {result.get('message', 'Unknown error')}", "APIModelConnector")
            return "Failed to generate response"
        except Exception as e:
            error_handler.handle_error(e, "APIModelConnector", "生成响应失败")
            return f"Error generating response: {str(e)}"

    def initialize(self) -> Dict[str, Any]:
        """
        初始化API模型连接器
        Initialize API model connector
        
        Returns:
            Dict[str, Any]: 初始化结果，包含成功状态和详细信息
        """
        try:
            # 初始化连接池
            self.connections = {}
            self.last_request_time = {}
            
            # 加载系统设置中的API配置
            # 这里可以添加预加载常用API配置的逻辑
            
            error_handler.log_info("API模型连接器初始化成功", "APIModelConnector")
            return {
                "success": True,
                "message": "API模型连接器初始化成功 | API model connector initialized successfully",
                "initialized_components": ["connection_pool", "request_tracker"]
            }
        except Exception as e:
            error_handler.handle_error(e, "APIModelConnector", "初始化失败")
            return {
                "success": False,
                "error": f"API模型连接器初始化失败: {str(e)} | API model connector initialization failed: {str(e)}"
            }
# 创建全局实例
api_model_connector = APIModelConnector()
