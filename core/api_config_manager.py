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

# Try to import openai module
try:
    import openai  # type: ignore
except ImportError:
    # If openai is not installed, create a mock object
    class MockOpenAI:
        api_key = ''
        class ChatCompletion:
            @staticmethod
            def create(**kwargs):
                return {"choices": [{"message": {"content": "Mock OpenAI response"}}]}
        class Completion:
            @staticmethod
            def create(**kwargs):
                return {"choices": [{"text": "Mock OpenAI response"}]}
    openai = MockOpenAI()


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
        # 使用文件顶部已定义的openai变量
        # 如果是Mock对象，说明库未安装
        if isinstance(openai, type) and openai.__name__ == 'MockOpenAI':
            return {
                'success': False, 
                'error': 'OpenAI library not installed. Please install with: pip install openai',
                'status_code': 500
            }
        
        openai.api_key = config.get('api_key', '')
        # 支持api_base字段设置自定义API端点
        if 'api_url' in config:
            openai.api_base = config['api_url']
        
        model_name = config.get('model_name', 'text-davinci-002')
        source = config.get('source', 'openai')
        
        try:
            # 优先使用较新的ChatCompletion接口
            if model_name.startswith('gpt-'):
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "Test connection"}],
                    max_tokens=5
                )
                result_text = response.choices[0].message.content.strip()
            else:
                # 回退到Completion接口
                response = openai.Completion.create(
                    engine=model_name,
                    prompt="Test connection",
                    max_tokens=5
                )
                result_text = response.choices[0].text.strip()
            
            result = {'success': True, 'response': result_text}
            # 添加元数据到结果中
            result['metadata'] = {
                'model_name': model_name,
                'source': source,
                'api_base': openai.api_base
            }
            return result
        except openai.error.AuthenticationError:
            return {'success': False, 'error': 'Invalid API key', 'status_code': 401}
        except openai.error.APIError as e:
            return {'success': False, 'error': f'API error: {str(e)}', 'status_code': 500}
        except Exception as e:
            return {'success': False, 'error': str(e), 'status_code': 500}
            
    def test_huggingface_connection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """测试HuggingFace API连接 / Test HuggingFace API connection"""
        api_key = config.get('api_key', '')
        model_name = config.get('model_name', '')
        # 支持自定义API URL
        api_url = config.get('api_url', '')
        source = config.get('source', 'huggingface')

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'inputs': 'Test connection'
        }

        # 构建请求URL
        if api_url:
            url = f'{api_url}/models/{model_name}' if not api_url.endswith('/') else f'{api_url}models/{model_name}'
        else:
            url = f'https://api-inference.huggingface.co/models/{model_name}'

        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = {'success': True, 'response': response.json()}
                # 添加元数据到结果中
                result['metadata'] = {
                    'model_name': model_name,
                    'source': source,
                    'api_url': url,
                    'status_code': response.status_code
                }
                return result
            elif response.status_code == 401:
                return {'success': False, 'error': 'Invalid API key', 'status_code': 401}
            elif response.status_code == 404:
                return {'success': False, 'error': 'API endpoint or model not found', 'status_code': 404}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}: {response.text}', 'status_code': response.status_code}
                
        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'API connection timeout', 'status_code': 408}
        except requests.exceptions.ConnectionError:
            return {'success': False, 'error': 'Failed to connect to API endpoint', 'status_code': 503}
        except Exception as e:
            return {'success': False, 'error': str(e), 'status_code': 500}
            
    def test_custom_connection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """测试自定义API连接 / Test custom API connection"""
        # 支持api_url或endpoint作为URL字段，保持一致性
        url = config.get('api_url', config.get('endpoint', ''))
        api_key = config.get('api_key', '')
        source = config.get('source', 'external')
        model_name = config.get('model_name', '')

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        # 构建测试数据，如果有model_name则添加
        data = {
            'prompt': 'Test connection'
        }
        if model_name:
            data['model'] = model_name

        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = {'success': True, 'response': response.json()}
                # 添加元数据到结果中
                result['metadata'] = {
                    'api_url': url,
                    'source': source,
                    'model_name': model_name,
                    'status_code': response.status_code
                }
                return result
            elif response.status_code == 401:
                return {'success': False, 'error': 'Invalid API key', 'status_code': 401}
            elif response.status_code == 404:
                return {'success': False, 'error': 'API endpoint not found', 'status_code': 404}
            elif response.status_code == 400 and 'model_not_found' in response.text:
                return {'success': False, 'error': f'Model not found: {model_name}', 'status_code': 400}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}: {response.text}', 'status_code': response.status_code}
                
        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'API connection timeout', 'status_code': 408}
        except requests.exceptions.ConnectionError:
            return {'success': False, 'error': 'Failed to connect to API endpoint', 'status_code': 503}
        except Exception as e:
            return {'success': False, 'error': str(e), 'status_code': 500}
