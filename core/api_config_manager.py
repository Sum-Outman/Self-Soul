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
api_config_manager.py - 中文描述
api_config_manager.py - English description

版权所有 (c) 2025 AGI Brain Team
Licensed under the Apache License, Version 2.0
"""
import json
import os
from typing import Dict, List, Any
import requests


class APIConfigManager:
    """API配置管理器 / API Configuration Manager"""
    
    def __init__(self, config_path: str = "config/api_config.json"):
        """初始化API配置管理器
        Initialize API configuration manager
        
        Args:
            config_path: 配置文件路径 | Configuration file path
        """
        self.config_path = config_path
        self.api_configs = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """加载API配置 / Load API configuration"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading API config: {e}")
                
        return {}
        
    def save_config(self):
        """保存API配置 / Save API configuration"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.api_configs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving API config: {e}")
            
    def add_api_config(self, api_name: str, config: Dict[str, Any]) -> bool:
        """添加API配置 / Add API configuration"""
        self.api_configs[api_name] = config
        self.save_config()
        return True
        
    def remove_api_config(self, api_name: str) -> bool:
        """移除API配置 / Remove API configuration"""
        if api_name in self.api_configs:
            del self.api_configs[api_name]
            self.save_config()
            return True
        return False
        
    def test_api_connection(self, api_name: str) -> Dict[str, Any]:
        """测试API连接 / Test API connection"""
        if api_name not in self.api_configs:
            return {'success': False, 'error': 'API not configured'}
            
        config = self.api_configs[api_name]
        api_type = config.get('type')
        
        try:
            if api_type == 'openai':
                return self.test_openai_connection(config)
            elif api_type == 'huggingface':
                return self.test_huggingface_connection(config)
            elif api_type == 'custom':
                return self.test_custom_connection(config)
            else:
                return {'success': False, 'error': f'Unknown API type: {api_type}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    def test_openai_connection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """测试OpenAI API连接 / Test OpenAI API connection"""
        import openai
        
        openai.api_key = config.get('api_key', '')
        try:
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt="Test connection",
                max_tokens=5
            )
            return {'success': True, 'response': response.choices[0].text.strip()}
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    def test_huggingface_connection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """测试HuggingFace API连接 / Test HuggingFace API connection"""
        api_key = config.get('api_key', '')
        model_name = config.get('model_name', '')

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'inputs': 'Test connection'
        }

        try:
            response = requests.post(
                f'https://api-inference.huggingface.co/models/{model_name}',
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                return {'success': True, 'response': response.json()}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}: {response.text}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    def test_custom_connection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """测试自定义API连接 / Test custom API connection"""
        url = config.get('endpoint', '')
        api_key = config.get('api_key', '')

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'prompt': 'Test connection'
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                return {'success': True, 'response': response.json()}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}: {response.text}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}