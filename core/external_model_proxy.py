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
基础模型类 - 所有模型的基类
Base Model Class - Base class for all models

提供通用接口和功能，确保所有模型的一致性
Provides common interfaces and functionality to ensure consistency across all models
"""
"""
外部模型代理 - 用于连接和管理外部API模型
External Model Proxy - For connecting and managing external API models
"""

import requests
import json
import logging
from typing import Dict, Any, Optional
from .error_handling import error_handler


"""
ExternalModelProxy类 - 中文类描述
ExternalModelProxy Class - English class description
"""
class ExternalModelProxy:
    """外部模型代理类
    External Model Proxy Class
    
    功能：代理外部API模型，提供统一的接口供系统调用
    Function: Proxy external API models, providing unified interface for system calls
    """
    
    
"""
__init__函数 - 中文函数描述
__init__ Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def __init__(self, model_id: str, api_config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.model_id = model_id
        self.api_config = api_config
        self.base_url = api_config.get('url', '')
        self.api_key = api_config.get('api_key', '')
        self.model_name = api_config.get('model_name', model_id)
        self.timeout = api_config.get('timeout', 30)
        
        # 验证配置 | Validate configuration
        if not self.base_url:
            error_handler.log_warning("外部模型URL未配置 | External model URL not configured", "ExternalModelProxy")
        
        self.logger.info(f"外部模型代理初始化: {model_id} | External model proxy initialized: {model_id}")
    
def call_api(self, endpoint: str, data: Dict[str, Any], method: str = 'POST') -> Optional[Dict[str, Any]]:
        """调用外部API
        Call external API
        
        Args:
            endpoint: API端点 | API endpoint
            data: 请求数据 | Request data
            method: HTTP方法 | HTTP method
            
        Returns:
            Optional[Dict[str, Any]]: API响应 | API response
        """
        try:
            url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            if method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
            elif method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=data, timeout=self.timeout)
            else:
                error_handler.log_warning(f"不支持的HTTP方法: {method} | Unsupported HTTP method: {method}", "ExternalModelProxy")
                return None
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            error_handler.handle_error(e, "ExternalModelProxy", f"调用外部API失败: {endpoint} | Failed to call external API: {endpoint}")
            return None
        except Exception as e:
            error_handler.handle_error(e, "ExternalModelProxy", f"处理API响应时发生错误 | Error occurred while processing API response")
            return None
    
def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理输入数据（通用接口）
        Process input data (general interface)
        
        Args:
            input_data: 输入数据 | Input data
            
        Returns:
            Optional[Dict[str, Any]]: 处理结果 | Processing result
        """
        # 根据模型类型调用相应的API端点
        # Call corresponding API endpoint based on model type
        if self.model_id.startswith('language'):
            return self.process_text(input_data)
        elif self.model_id.startswith('audio'):
            return self.process_audio(input_data)
        elif self.model_id.startswith('image'):
            return self.process_image(input_data)
        elif self.model_id.startswith('video'):
            return self.process_video(input_data)
        else:
            # 默认处理 | Default processing
            return self.call_api('/process', input_data)
    
def process_text(self, text_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理文本数据
        Process text data
        
        Args:
            text_data: 文本数据 | Text data
            
        Returns:
            Optional[Dict[str, Any]]: 处理结果 | Processing result
        """
        return self.call_api('/text/process', text_data)
    
def process_audio(self, audio_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理音频数据
        Process audio data
        
        Args:
            audio_data: 音频数据 | Audio data
            
        Returns:
            Optional[Dict[str, Any]]: 处理结果 | Processing result
        """
        return self.call_api('/audio/process', audio_data)
    
def process_image(self, image_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理图像数据
        Process image data
        
        Args:
            image_data: 图像数据 | Image data
            
        Returns:
            Optional[Dict[str, Any]]: 处理结果 | Processing result
        """
        return self.call_api('/image/process', image_data)
    
def process_video(self, video_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理视频数据
        Process video data
        
        Args:
            video_data: 视频数据 | Video data
            
        Returns:
            Optional[Dict[str, Any]]: 处理结果 | Processing result
        """
        return self.call_api('/video/process', video_data)
    
def train(self, training_data: Dict[str, Any] = None, callback=None) -> Dict[str, Any]:
        """训练模型（外部模型通常不支持训练）
        Train model (external models typically don't support training)
        
        Args:
            training_data: 训练数据 | Training data
            callback: 进度回调 | Progress callback
            
        Returns:
            Dict[str, Any]: 训练结果 | Training results
        """
        error_handler.log_warning(f"外部模型 {self.model_id} 不支持训练 | External model {self.model_id} does not support training", "ExternalModelProxy")
        
        if callback:
            callback(100, {'status': 'not_supported', 'message': 'External models cannot be trained locally'})
        
        return {
            'status': 'not_supported',
            'message': 'External models cannot be trained locally',
            'model_id': self.model_id
        }
    
def get_status(self) -> Dict[str, Any]:
        """获取模型状态
        Get model status
        
        Returns:
            Dict[str, Any]: 状态信息 | Status information
        """
        # 检查API连接状态 | Check API connection status
        test_data = {'test': 'ping'}
        response = self.call_api('/status', test_data, 'GET')
        
        if response and 'status' in response:
            return {
                'model_id': self.model_id,
                'status': 'connected',
                'api_status': response['status'],
                'model_type': 'external',
                'model_name': self.model_name
            }
        else:
            return {
                'model_id': self.model_id,
                'status': 'disconnected',
                'model_type': 'external',
                'model_name': self.model_name,
                'error': 'API connection failed'
            }
    
    
"""
cleanup函数 - 中文函数描述
cleanup Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def cleanup(self):
        """清理资源
        Cleanup resources
        """
        # 外部模型代理不需要清理物理资源
        # External model proxy doesn't need to cleanup physical resources
        self.logger.info(f"外部模型代理清理完成: {self.model_id} | External model proxy cleanup completed: {self.model_id}")

# 导出类 | Export class
ExternalModelProxy = ExternalModelProxy