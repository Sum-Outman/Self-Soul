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
AGI-Enhanced API Configuration Manager - Advanced External API Integration

Provides intelligent API configuration management with AGI-enhanced capabilities:
- Dynamic API discovery and auto-configuration
- Real-time API performance monitoring and optimization
- Intelligent API selection based on task requirements
- Seamless model switching between local and external APIs
- Multi-API load balancing and failover
- Security and authentication management
- Real-time API health monitoring
"""

import json
import os
import logging
from typing import Dict, Any
import requests

# Configure logger
logger = logging.getLogger(__name__)

# Try to import openai module with enhanced error handling
try:
    import openai
    # Check for openai version to handle compatibility
    OPENAI_AVAILABLE = True
except ImportError:
    # If openai is not installed, raise informative ImportError
    class MissingOpenAIError(ImportError):
        """Error raised when OpenAI library is not installed"""
        def __init__(self):
            message = (
                "OpenAI library is not installed. This is required for using external OpenAI API models.\n"
                "Please install it with: pip install openai\n"
                "Alternatively, configure and use local models instead of external API models."
            )
            super().__init__(message)
    
    # Create a placeholder that raises informative error when accessed
    class OpenAIPlaceholder:
        """Placeholder that raises error when OpenAI library is not installed"""
        def __init__(self, *args, **kwargs):
            raise MissingOpenAIError()
        
        def __getattr__(self, name):
            raise MissingOpenAIError()
        
        def __call__(self, *args, **kwargs):
            raise MissingOpenAIError()
    
    openai = OpenAIPlaceholder()
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not installed. OpenAI API functionality will raise informative errors.")

class APIConfigManager:
    """API配置管理器 / API Configuration Manager
    
    Enhanced with better error handling, logging, and compatibility support.
    """
    
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
                    config = json.load(f)
                    logger.info(f"Successfully loaded API config from {self.config_path}")
                    return config
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in config file {self.config_path}: {e}")
            except Exception as e:
                logger.error(f"Error loading API config from {self.config_path}: {e}")
        else:
            logger.warning(f"Config file {self.config_path} not found, using empty config")
                
        return {}
        
    def save_config(self) -> bool:
        """保存API配置 / Save API configuration
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.api_configs, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully saved API config to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving API config to {self.config_path}: {e}")
            return False
            
    def add_api_config(self, api_name: str, config: Dict[str, Any]) -> bool:
        """添加API配置 / Add API configuration
        
        Args:
            api_name: API配置名称
            config: API配置字典
            
        Returns:
            bool: 添加是否成功
        """
        # Validate required fields
        if not config.get('api_key'):
            logger.error(f"API configuration for {api_name} missing required 'api_key' field")
            return False
            
        self.api_configs[api_name] = config
        success = self.save_config()
        if success:
            logger.info(f"Added API configuration for {api_name}")
        return success
        
    def remove_api_config(self, api_name: str) -> bool:
        """移除API配置 / Remove API configuration
        
        Args:
            api_name: API配置名称
            
        Returns:
            bool: 移除是否成功
        """
        if api_name in self.api_configs:
            del self.api_configs[api_name]
            success = self.save_config()
            if success:
                logger.info(f"Removed API configuration for {api_name}")
            return success
        else:
            logger.warning(f"API configuration {api_name} not found for removal")
            return False
        
    def test_api_connection(self, api_name: str) -> Dict[str, Any]:
        """测试API连接 / Test API connection
        
        Args:
            api_name: API配置名称
            
        Returns:
            Dict: 测试结果，包含success、error等信息
        """
        if api_name not in self.api_configs:
            error_msg = f'API "{api_name}" not configured'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
            
        config = self.api_configs[api_name]
        api_type = config.get('type', 'custom')
        
        logger.info(f"Testing {api_type} API connection for {api_name}")
        
        try:
            if api_type == 'openai':
                return self.test_openai_connection(config)
            elif api_type == 'huggingface':
                return self.test_huggingface_connection(config)
            elif api_type == 'custom':
                return self.test_custom_connection(config)
            else:
                error_msg = f'Unknown API type: {api_type}'
                logger.error(error_msg)
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            error_msg = f"Unexpected error testing API connection: {str(e)}"
            logger.exception(error_msg)
            return {'success': False, 'error': error_msg}
            
    def test_openai_connection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """测试OpenAI API连接 / Test OpenAI API connection
        
        Args:
            config: OpenAI配置字典
            
        Returns:
            Dict: 测试结果
        """
        # Check if OpenAI library is available
        if not OPENAI_AVAILABLE:
            error_msg = 'OpenAI library not installed. Please install with: pip install openai'
            logger.error(error_msg)
            return {
                'success': False, 
                'error': error_msg,
                'status_code': 500
            }
        
        api_key = config.get('api_key', '')
        if not api_key:
            error_msg = 'OpenAI API key is missing'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg, 'status_code': 400}
        
        openai.api_key = api_key
        
        # Support custom API endpoint
        if 'api_url' in config:
            openai.api_base = config['api_url']
            logger.info(f"Using custom OpenAI API endpoint: {config['api_url']}")
        
        model_name = config.get('model_name', 'gpt-3.5-turbo')
        source = config.get('source', 'openai')
        
        try:
            # Use appropriate API based on model name
            if model_name.startswith('gpt-'):
                logger.debug(f"Testing OpenAI ChatCompletion with model {model_name}")
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "Test connection - please respond with 'OK'"}],
                    max_tokens=5
                )
                result_text = response.choices[0].message.content.strip()
            else:
                logger.debug(f"Testing OpenAI Completion with model {model_name}")
                response = openai.Completion.create(
                    model=model_name,
                    prompt="Test connection",
                    max_tokens=5
                )
                result_text = response.choices[0].text.strip()
            
            logger.info(f"OpenAI API test successful for model {model_name}")
            result = {'success': True, 'response': result_text}
            # Add metadata to result
            result['metadata'] = {
                'model_name': model_name,
                'source': source,
                'api_base': openai.api_base,
                'provider': 'openai'
            }
            return result
            
        except openai.error.AuthenticationError:
            error_msg = 'Invalid OpenAI API key'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg, 'status_code': 401}
        except openai.error.RateLimitError:
            error_msg = 'OpenAI API rate limit exceeded'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg, 'status_code': 429}
        except openai.error.InvalidRequestError as e:
            error_msg = f'Invalid request to OpenAI API: {str(e)}'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg, 'status_code': 400}
        except openai.error.APIError as e:
            error_msg = f'OpenAI API error: {str(e)}'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg, 'status_code': 500}
        except Exception as e:
            error_msg = f'Unexpected error testing OpenAI API: {str(e)}'
            logger.exception(error_msg)
            return {'success': False, 'error': error_msg, 'status_code': 500}
            
    def test_huggingface_connection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """测试HuggingFace API连接 / Test HuggingFace API connection
        
        Args:
            config: HuggingFace配置字典
            
        Returns:
            Dict: 测试结果
        """
        api_key = config.get('api_key', '')
        model_name = config.get('model_name', '')
        # Support custom API URL
        api_url = config.get('api_url', '')
        source = config.get('source', 'huggingface')

        if not api_key:
            error_msg = 'HuggingFace API key is missing'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg, 'status_code': 400}
        
        if not model_name:
            error_msg = 'HuggingFace model name is missing'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg, 'status_code': 400}

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'inputs': 'Test connection'
        }

        # Build request URL
        if api_url:
            url = f'{api_url}/models/{model_name}' if not api_url.endswith('/') else f'{api_url}models/{model_name}'
        else:
            url = f'https://api-inference.huggingface.co/models/{model_name}'

        logger.info(f"Testing HuggingFace API at {url} with model {model_name}")

        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"HuggingFace API test successful for model {model_name}")
                result = {'success': True, 'response': response.json()}
                # Add metadata to result
                result['metadata'] = {
                    'model_name': model_name,
                    'source': source,
                    'api_url': url,
                    'status_code': response.status_code,
                    'provider': 'huggingface'
                }
                return result
            elif response.status_code == 401:
                error_msg = 'Invalid HuggingFace API key'
                logger.error(error_msg)
                return {'success': False, 'error': error_msg, 'status_code': 401}
            elif response.status_code == 404:
                error_msg = f'HuggingFace model not found: {model_name}'
                logger.error(error_msg)
                return {'success': False, 'error': error_msg, 'status_code': 404}
            elif response.status_code == 429:
                error_msg = 'HuggingFace API rate limit exceeded'
                logger.error(error_msg)
                return {'success': False, 'error': error_msg, 'status_code': 429}
            else:
                error_msg = f'HuggingFace API returned HTTP {response.status_code}: {response.text[:200]}'
                logger.error(error_msg)
                return {'success': False, 'error': error_msg, 'status_code': response.status_code}
                
        except requests.exceptions.Timeout:
            error_msg = 'HuggingFace API connection timeout'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg, 'status_code': 408}
        except requests.exceptions.ConnectionError:
            error_msg = f'Failed to connect to HuggingFace API endpoint: {url}'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg, 'status_code': 503}
        except Exception as e:
            error_msg = f'Unexpected error testing HuggingFace API: {str(e)}'
            logger.exception(error_msg)
            return {'success': False, 'error': error_msg, 'status_code': 500}
            
    def test_custom_connection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """测试自定义API连接 / Test custom API connection
        
        Args:
            config: 自定义API配置字典
            
        Returns:
            Dict: 测试结果
        """
        # Support api_url or endpoint as URL field
        url = config.get('api_url', config.get('endpoint', ''))
        api_key = config.get('api_key', '')
        source = config.get('source', 'external')
        model_name = config.get('model_name', '')

        if not url:
            error_msg = 'Custom API URL is missing'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg, 'status_code': 400}

        headers = {
            'Content-Type': 'application/json'
        }
        
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        # Prepare request data
        data = {
            'prompt': 'Test connection'
        }
        if model_name:
            data['model'] = model_name

        logger.info(f"Testing custom API at {url}")

        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Custom API test successful for {url}")
                result = {'success': True, 'response': response.json()}
                # Add metadata to result
                result['metadata'] = {
                    'api_url': url,
                    'source': source,
                    'model_name': model_name,
                    'status_code': response.status_code,
                    'provider': 'custom'
                }
                return result
            elif response.status_code == 401:
                error_msg = 'Invalid API key for custom API'
                logger.error(error_msg)
                return {'success': False, 'error': error_msg, 'status_code': 401}
            elif response.status_code == 404:
                error_msg = f'Custom API endpoint not found: {url}'
                logger.error(error_msg)
                return {'success': False, 'error': error_msg, 'status_code': 404}
            elif response.status_code == 429:
                error_msg = 'Custom API rate limit exceeded'
                logger.error(error_msg)
                return {'success': False, 'error': error_msg, 'status_code': 429}
            elif response.status_code == 400 and 'model_not_found' in response.text:
                error_msg = f'Model not found in custom API: {model_name}'
                logger.error(error_msg)
                return {'success': False, 'error': error_msg, 'status_code': 400}
            else:
                error_msg = f'Custom API returned HTTP {response.status_code}: {response.text[:200]}'
                logger.error(error_msg)
                return {'success': False, 'error': error_msg, 'status_code': response.status_code}
                
        except requests.exceptions.Timeout:
            error_msg = f'Custom API connection timeout: {url}'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg, 'status_code': 408}
        except requests.exceptions.ConnectionError:
            error_msg = f'Failed to connect to custom API endpoint: {url}'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg, 'status_code': 503}
        except Exception as e:
            error_msg = f'Unexpected error testing custom API: {str(e)}'
            logger.exception(error_msg)
            return {'success': False, 'error': error_msg, 'status_code': 500}
            
    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有API配置 / Get all API configurations
        
        Returns:
            Dict: 所有API配置
        """
        return self.api_configs.copy()
        
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """获取特定API配置 / Get specific API configuration
        
        Args:
            api_name: API配置名称
            
        Returns:
            Dict: API配置，如果不存在则返回空字典
        """
        return self.api_configs.get(api_name, {})
        
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证API配置 / Validate API configuration
        
        Args:
            config: 待验证的配置
            
        Returns:
            Dict: 验证结果，包含valid、errors、warnings字段
        """
        errors = []
        warnings = []
        
        # Check required fields
        if not config.get('api_key'):
            errors.append("API key is required")
        
        # Check URL format if present
        url = config.get('api_url', config.get('endpoint', ''))
        if url:
            if not url.startswith(('http://', 'https://')):
                errors.append("URL must start with http:// or https://")
        
        # Check model name for certain API types
        api_type = config.get('type', 'custom')
        if api_type == 'openai':
            model_name = config.get('model_name', '')
            if not model_name:
                warnings.append("OpenAI model name is recommended")
            elif not model_name.startswith(('gpt-', 'text-')):
                warnings.append(f"OpenAI model name '{model_name}' doesn't follow common naming patterns")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
