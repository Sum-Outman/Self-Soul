"""
API客户端工厂类 - API Client Factory
提供统一的API客户端创建和管理，减少重复的初始化代码
Provides unified API client creation and management, reducing duplicate initialization code
"""

import logging
import functools
import os
from typing import Dict, Any, Optional, Type, Union, Callable
import importlib.util
import sys

logger = logging.getLogger(__name__)

class APIClientFactory:
    """API客户端工厂类 | API Client Factory Class
    
    使用工厂模式统一创建和管理各种API客户端，减少重复的初始化代码
    Uses factory pattern to uniformly create and manage various API clients, reducing duplicate initialization code
    """
    
    def __init__(self):
        """初始化客户端工厂 | Initialize client factory"""
        self.clients = {}
        self.client_configs = {}
        self.client_cache = {}
        
        # 注册所有支持的API提供商
        # Register all supported API providers
        self._register_all_providers()
    
    def _register_all_providers(self):
        """注册所有API提供商 | Register all API providers"""
        self.provider_registry = {
            # OpenAI
            "openai": {
                "module_name": "openai",
                "client_class": "OpenAI",
                "config_keys": ["api_key", "base_url", "organization", "project"],
                "optional_keys": ["timeout", "max_retries"],
                "validation_rules": {
                    "api_key": {"required": True, "type": str, "min_length": 10},
                    "base_url": {"required": False, "type": str, "default": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")}
                }
            },
            # Anthropic
            "anthropic": {
                "module_name": "anthropic",
                "client_class": "Anthropic",
                "config_keys": ["api_key"],
                "optional_keys": ["base_url", "timeout", "max_retries"],
                "validation_rules": {
                    "api_key": {"required": True, "type": str, "min_length": 10}
                }
            },
            # Google Generative AI
            "google_genai": {
                "module_name": "google.generativeai",
                "client_class": "GenerativeModel",
                "config_keys": ["api_key"],
                "optional_keys": ["model_name"],
                "validation_rules": {
                    "api_key": {"required": True, "type": str, "min_length": 10}
                }
            },
            # AWS Boto3 (通用)
            "aws_boto3": {
                "module_name": "boto3",
                "client_class": "client",
                "config_keys": ["service_name"],
                "optional_keys": ["region_name", "aws_access_key_id", "aws_secret_access_key"],
                "validation_rules": {
                    "service_name": {"required": True, "type": str}
                }
            },
            # Azure OpenAI
            "azure_openai": {
                "module_name": "openai",
                "client_class": "AzureOpenAI",
                "config_keys": ["api_key", "azure_endpoint", "api_version"],
                "optional_keys": ["azure_deployment"],
                "validation_rules": {
                    "api_key": {"required": True, "type": str, "min_length": 10},
                    "azure_endpoint": {"required": True, "type": str},
                    "api_version": {"required": True, "type": str}
                }
            },
            # Cohere
            "cohere": {
                "module_name": "cohere",
                "client_class": "Client",
                "config_keys": ["api_key"],
                "optional_keys": ["base_url", "client_name", "check_api_key", "timeout"],
                "validation_rules": {
                    "api_key": {"required": True, "type": str, "min_length": 10}
                }
            },
            # Mistral AI
            "mistral": {
                "module_name": "mistralai",
                "client_class": "Mistral",
                "config_keys": ["api_key"],
                "optional_keys": ["endpoint"],
                "validation_rules": {
                    "api_key": {"required": True, "type": str, "min_length": 10}
                }
            },
            # HuggingFace Inference
            "huggingface": {
                "module_name": "huggingface_hub",
                "client_class": "InferenceClient",
                "config_keys": [],
                "optional_keys": ["token", "model", "timeout"],
                "validation_rules": {
                    "token": {"required": False, "type": str}
                }
            },
            # Replicate
            "replicate": {
                "module_name": "replicate",
                "client_class": "Client",
                "config_keys": [],
                "optional_keys": ["api_token", "base_url"],
                "validation_rules": {
                    "api_token": {"required": True, "type": str, "min_length": 10}
                }
            },
            # Ollama
            "ollama": {
                "module_name": "ollama",
                "client_class": "Client",
                "config_keys": [],
                "optional_keys": ["host"],
                "validation_rules": {
                    "host": {"required": False, "type": str, "default": os.environ.get("OLLAMA_HOST", "http://localhost:11434")}
                }
            }
        }
        
        # 国内API提供商
        # Domestic API providers
        self.domestic_providers = {
            # DeepSeek
            "deepseek": {
                "module_name": "openai",  # 兼容OpenAI API
                "client_class": "OpenAI",
                "config_keys": ["api_key"],
                "optional_keys": ["base_url"],
                "validation_rules": {
                    "api_key": {"required": True, "type": str, "min_length": 10},
                    "base_url": {"required": True, "type": str, "default": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")}
                }
            },
            # SiliconFlow
            "siliconflow": {
                "module_name": "openai",  # 兼容OpenAI API
                "client_class": "OpenAI",
                "config_keys": ["api_key"],
                "optional_keys": ["base_url"],
                "validation_rules": {
                    "api_key": {"required": True, "type": str, "min_length": 10},
                    "base_url": {"required": True, "type": str, "default": os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")}
                }
            },
            # 智谱AI (Zhipu)
            "zhipu": {
                "module_name": "zhipuai",
                "client_class": "ZhipuAI",
                "config_keys": ["api_key"],
                "optional_keys": ["base_url"],
                "validation_rules": {
                    "api_key": {"required": True, "type": str, "min_length": 10}
                }
            },
            # 百度文心 (Baidu ERNIE)
            "baidu": {
                "module_name": "qianfan",
                "client_class": "ChatCompletion",
                "config_keys": ["api_key", "secret_key"],
                "optional_keys": ["base_url"],
                "validation_rules": {
                    "api_key": {"required": True, "type": str, "min_length": 10},
                    "secret_key": {"required": True, "type": str, "min_length": 10}
                }
            },
            # 阿里通义千问 (Alibaba Qwen)
            "alibaba": {
                "module_name": "dashscope",
                "client_class": "Generation",
                "config_keys": ["api_key"],
                "optional_keys": ["base_url"],
                "validation_rules": {
                    "api_key": {"required": True, "type": str, "min_length": 10}
                }
            },
            # 月之暗面 (Moonshot)
            "moonshot": {
                "module_name": "openai",  # 兼容OpenAI API
                "client_class": "OpenAI",
                "config_keys": ["api_key"],
                "optional_keys": ["base_url"],
                "validation_rules": {
                    "api_key": {"required": True, "type": str, "min_length": 10},
                    "base_url": {"required": True, "type": str, "default": os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")}
                }
            },
            # 零一万物 (Yi)
            "yi": {
                "module_name": "openai",  # 兼容OpenAI API
                "client_class": "OpenAI",
                "config_keys": ["api_key"],
                "optional_keys": ["base_url"],
                "validation_rules": {
                    "api_key": {"required": True, "type": str, "min_length": 10},
                    "base_url": {"required": True, "type": str, "default": os.environ.get("LINGYIWANWU_BASE_URL", "https://api.lingyiwanwu.com/v1")}
                }
            },
            # 腾讯混元 (Tencent Hunyuan)
            "tencent": {
                "module_name": "tencentcloud",
                "client_class": "HunyuanClient",
                "config_keys": ["secret_id", "secret_key"],
                "optional_keys": ["region"],
                "validation_rules": {
                    "secret_id": {"required": True, "type": str, "min_length": 10},
                    "secret_key": {"required": True, "type": str, "min_length": 10},
                    "region": {"required": True, "type": str, "default": "ap-guangzhou"}
                }
            }
        }
        
        # 合并所有提供商
        # Merge all providers
        self.provider_registry.update(self.domestic_providers)
    
    def create_client(self, provider: str, config: Dict[str, Any]) -> Any:
        """创建API客户端 | Create API client
        
        Args:
            provider: API提供商名称
            config: 客户端配置
            
        Returns:
            API客户端实例 | API client instance
        """
        try:
            # 检查是否已有缓存
            # Check if already cached
            cache_key = self._get_cache_key(provider, config)
            if cache_key in self.client_cache:
                logger.debug(f"使用缓存的{provider}客户端 | Using cached {provider} client")
                return self.client_cache[cache_key]
            
            # 验证提供商是否支持
            # Verify if provider is supported
            if provider not in self.provider_registry:
                raise ValueError(f"不支持的API提供商: {provider} | Unsupported API provider: {provider}")
            
            provider_info = self.provider_registry[provider]
            
            # 验证配置
            # Validate configuration
            self._validate_config(provider, config, provider_info)
            
            # 导入模块
            # Import module
            module = self._import_module(provider_info["module_name"], provider)
            
            # 创建客户端
            # Create client
            client = self._instantiate_client(module, provider_info["client_class"], config, provider)
            
            # 缓存客户端
            # Cache client
            self.client_cache[cache_key] = client
            self.client_configs[cache_key] = config
            
            logger.info(f"成功创建{provider}客户端 | Successfully created {provider} client")
            return client
            
        except Exception as e:
            logger.error(f"创建{provider}客户端失败: {str(e)} | Failed to create {provider} client: {str(e)}")
            
            # No mock client fallback - real API clients required
            raise RuntimeError(
                f"Failed to create real API client for provider '{provider}'. "
                f"Please check API credentials, network connectivity, and required libraries. "
                f"Error: {str(e)}"
            )
    
    def _get_cache_key(self, provider: str, config: Dict[str, Any]) -> str:
        """生成缓存键 | Generate cache key"""
        import json
        import hashlib
        
        # 创建配置的哈希值
        # Create hash of configuration
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        return f"{provider}:{config_hash}"
    
    def _validate_config(self, provider: str, config: Dict[str, Any], provider_info: Dict[str, Any]):
        """验证配置 | Validate configuration"""
        validation_rules = provider_info.get("validation_rules", {})
        
        for field, rules in validation_rules.items():
            is_required = rules.get("required", False)
            field_type = rules.get("type", str)
            min_length = rules.get("min_length", 0)
            default_value = rules.get("default", None)
            
            # 检查必需字段
            # Check required fields
            if is_required and field not in config:
                if default_value is not None:
                    config[field] = default_value
                else:
                    raise ValueError(f"缺少必需字段: {field} | Missing required field: {field}")
            
            # 检查字段类型
            # Check field type
            if field in config and not isinstance(config[field], field_type):
                try:
                    # 尝试类型转换
                    # Try type conversion
                    config[field] = field_type(config[field])
                except (ValueError, TypeError):
                    raise ValueError(f"字段{field}类型错误，应为{field_type} | Field {field} type error, should be {field_type}")
            
            # 检查最小长度
            # Check minimum length
            if field in config and isinstance(config[field], str) and len(config[field]) < min_length:
                raise ValueError(f"字段{field}长度过短，至少需要{min_length}个字符 | Field {field} too short, minimum {min_length} characters required")
    
    def _import_module(self, module_name: str, provider: str) -> Any:
        """导入模块 | Import module"""
        try:
            # 尝试导入模块
            # Try to import module
            module = importlib.import_module(module_name)
            logger.debug(f"成功导入{module_name}模块 | Successfully imported {module_name} module")
            return module
        except ImportError as e:
            error_msg = f"无法导入{module_name}模块。请安装相应的Python包。错误: {str(e)} | Cannot import {module_name} module. Please install the required Python package. Error: {str(e)}"
            logger.error(error_msg)
            raise ImportError(error_msg)
    
    def _instantiate_client(self, module: Any, client_class: str, config: Dict[str, Any], provider: str) -> Any:
        """实例化客户端 | Instantiate client"""
        try:
            # 获取客户端类
            # Get client class
            if hasattr(module, client_class):
                cls = getattr(module, client_class)
            else:
                # 如果模块本身就是一个客户端类
                # If module itself is a client class
                cls = module
            
            # 根据提供商类型使用不同的初始化方式
            # Use different initialization methods based on provider type
            if provider == "aws_boto3":
                # AWS Boto3 客户端特殊处理
                # AWS Boto3 client special handling
                service_name = config.pop("service_name", None)
                if not service_name:
                    raise ValueError("AWS客户端需要service_name参数 | AWS client requires service_name parameter")
                
                client = cls(service_name, **config)
            
            elif provider in ["openai", "deepseek", "siliconflow", "moonshot", "yi"]:
                # OpenAI兼容客户端
                # OpenAI compatible clients
                client = cls(**config)
            
            elif provider == "anthropic":
                # Anthropic客户端
                # Anthropic client
                client = cls(**config)
            
            elif provider == "google_genai":
                # Google Generative AI特殊处理
                # Google Generative AI special handling
                try:
                    import google.generativeai as genai  # type: ignore
                    genai.configure(api_key=config.get("api_key"))
                    model_name = config.get("model_name", "gemini-pro")
                    client = cls(model_name)
                except ImportError as e:
                    error_msg = f"无法导入Google Generative AI模块。请安装google-generativeai包。错误: {str(e)} | Cannot import Google Generative AI module. Please install google-generativeai package. Error: {str(e)}"
                    logger.warning(error_msg)
                    raise ImportError(error_msg)
            
            elif provider == "azure_openai":
                # Azure OpenAI特殊处理
                # Azure OpenAI special handling
                import openai
                client = openai.AzureOpenAI(**config)
            
            elif provider == "cohere":
                # Cohere客户端
                # Cohere client
                client = cls(**config)
            
            elif provider == "mistral":
                # Mistral AI客户端
                # Mistral AI client
                client = cls(**config)
            
            elif provider == "huggingface":
                # HuggingFace Inference客户端
                # HuggingFace Inference client
                token = config.get("token", None)
                model = config.get("model", None)
                timeout = config.get("timeout", None)
                
                client_args = {}
                if token:
                    client_args["token"] = token
                if model:
                    client_args["model"] = model
                if timeout:
                    client_args["timeout"] = timeout
                
                client = cls(**client_args)
            
            elif provider == "replicate":
                # Replicate客户端
                # Replicate client
                api_token = config.get("api_token", None)
                base_url = config.get("base_url", None)
                
                try:
                    import replicate as replicate_module  # type: ignore
                    if api_token:
                        replicate_module.api_token = api_token
                        client = replicate_module
                    else:
                        client = replicate_module
                except ImportError as e:
                    error_msg = f"无法导入Replicate模块。请安装replicate包。错误: {str(e)} | Cannot import Replicate module. Please install replicate package. Error: {str(e)}"
                    logger.warning(error_msg)
                    raise ImportError(error_msg)
            
            elif provider == "ollama":
                # Ollama客户端
                # Ollama client
                host = config.get("host", os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
                client = cls(host=host)
            
            elif provider == "zhipu":
                # 智谱AI客户端
                # Zhipu AI client
                client = cls(**config)
            
            elif provider == "baidu":
                # 百度文心客户端
                # Baidu ERNIE client
                # 百度API需要特殊处理
                # Baidu API requires special handling
                try:
                    from qianfan import QfResponse  # type: ignore
                    client = cls()
                    # 保存配置供后续使用
                    # Save configuration for later use
                    client._api_key = config.get("api_key")
                    client._secret_key = config.get("secret_key")
                except ImportError as e:
                    error_msg = f"无法导入百度千帆模块。请安装qianfan包。错误: {str(e)} | Cannot import Baidu Qianfan module. Please install qianfan package. Error: {str(e)}"
                    logger.warning(error_msg)
                    raise ImportError(error_msg)
            
            elif provider == "alibaba":
                # 阿里通义千问客户端
                # Alibaba Qwen client
                try:
                    import dashscope  # type: ignore
                    dashscope.api_key = config.get("api_key")
                    client = dashscope
                except ImportError as e:
                    error_msg = f"无法导入阿里达摩院Dashscope模块。请安装dashscope包。错误: {str(e)} | Cannot import Alibaba Dashscope module. Please install dashscope package. Error: {str(e)}"
                    logger.warning(error_msg)
                    raise ImportError(error_msg)
            
            elif provider == "tencent":
                # 腾讯混元客户端
                # Tencent Hunyuan client
                try:
                    from tencentcloud.common import credential  # type: ignore
                    from tencentcloud.hunyuan.v20230901 import hunyuan_client  # type: ignore
                    
                    cred = credential.Credential(
                        config.get("secret_id"),
                        config.get("secret_key")
                    )
                    client = hunyuan_client.HunyuanClient(cred, config.get("region", "ap-guangzhou"))
                except ImportError as e:
                    error_msg = f"无法导入腾讯云SDK模块。请安装tencentcloud-sdk-python包。错误: {str(e)} | Cannot import Tencent Cloud SDK module. Please install tencentcloud-sdk-python package. Error: {str(e)}"
                    logger.warning(error_msg)
                    raise ImportError(error_msg)
            
            else:
                # 默认初始化方式
                # Default initialization method
                client = cls(**config)
            
            return client
            
        except Exception as e:
            logger.error(f"实例化{provider}客户端失败: {str(e)} | Failed to instantiate {provider} client: {str(e)}")
            raise

    def _create_mock_client(self, provider: str, config: Dict[str, Any]) -> Any:
        """Mock clients are not supported - real API clients required"""
        raise RuntimeError(
            f"Mock clients are not supported for provider '{provider}'. "
            f"Real API clients are required. Please provide valid API credentials and ensure required libraries are installed. "
            f"Simulated responses are not allowed in production systems."
        )
    
    def get_or_create_client(self, provider: str, config: Dict[str, Any]) -> Any:
        """获取或创建API客户端 | Get or create API client
        
        如果已存在相同配置的客户端，则返回缓存的客户端，否则创建新客户端
        If a client with the same configuration exists, return cached client, otherwise create new client
        """
        cache_key = self._get_cache_key(provider, config)
        
        if cache_key in self.client_cache:
            logger.debug(f"返回缓存的{provider}客户端 | Returning cached {provider} client")
            return self.client_cache[cache_key]
        
        return self.create_client(provider, config)
    
    def clear_cache(self, provider: str = None):
        """清除客户端缓存 | Clear client cache
        
        Args:
            provider: 可选的提供商名称，如果提供则只清除该提供商的缓存
                     Optional provider name, if provided only clear cache for that provider
        """
        if provider:
            # 清除特定提供商的缓存
            # Clear cache for specific provider
            keys_to_delete = [k for k in self.client_cache.keys() if k.startswith(f"{provider}:")]
            for key in keys_to_delete:
                del self.client_cache[key]
                if key in self.client_configs:
                    del self.client_configs[key]
            
            logger.info(f"已清除{provider}客户端缓存 | Cleared {provider} client cache")
        else:
            # 清除所有缓存
            # Clear all cache
            self.client_cache.clear()
            self.client_configs.clear()
            logger.info("已清除所有客户端缓存 | Cleared all client cache")
    
    def list_clients(self) -> Dict[str, Any]:
        """列出所有已创建的客户端 | List all created clients
        
        Returns:
            客户端列表信息 | Client list information
        """
        clients_info = {}
        
        for cache_key, client in self.client_cache.items():
            provider, config_hash = cache_key.split(":", 1)
            config = self.client_configs.get(cache_key, {})
            
            if provider not in clients_info:
                clients_info[provider] = []
            
            # 隐藏敏感信息
            # Hide sensitive information
            safe_config = {}
            for key, value in config.items():
                if "key" in key.lower() or "secret" in key.lower() or "token" in key.lower():
                    safe_config[key] = "[REDACTED]"
                else:
                    safe_config[key] = value
            
            clients_info[provider].append({
                "config_hash": config_hash,
                "config": safe_config,
                "client_type": type(client).__name__
            })
        
        return {
            "total_clients": len(self.client_cache),
            "providers": clients_info,
            "timestamp": self._get_current_timestamp()
        }
    
    def _get_current_timestamp(self) -> str:
        """获取当前时间戳 | Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def test_client_connection(self, provider: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """测试客户端连接 | Test client connection
        
        Args:
            provider: API提供商名称
            config: 客户端配置
            
        Returns:
            连接测试结果 | Connection test result
        """
        try:
            # 创建客户端
            # Create client
            client = self.create_client(provider, config)
            
            # 根据提供商类型执行不同的测试
            # Execute different tests based on provider type
            if provider in ["openai", "deepseek", "siliconflow", "moonshot", "yi", "azure_openai"]:
                # OpenAI兼容API测试
                # OpenAI compatible API test
                test_result = self._test_openai_compatible_client(client, provider)
            
            elif provider == "anthropic":
                # Anthropic API测试
                # Anthropic API test
                test_result = self._test_anthropic_client(client)
            
            elif provider == "google_genai":
                # Google Generative AI测试
                # Google Generative AI test
                test_result = self._test_google_genai_client(client)
            
            elif provider == "cohere":
                # Cohere API测试
                # Cohere API test
                test_result = self._test_cohere_client(client)
            
            elif provider == "mistral":
                # Mistral AI测试
                # Mistral AI test
                test_result = self._test_mistral_client(client)
            
            elif provider in ["zhipu", "baidu", "alibaba", "tencent"]:
                # 国内API提供商测试
                # Domestic API provider test
                test_result = self._test_domestic_client(client, provider)
            
            else:
                # 通用测试
                # Generic test
                test_result = {"connected": True, "message": f"{provider}客户端已创建 | {provider} client created"}
            
            return {
                "success": True,
                "provider": provider,
                "connected": True,
                "test_result": test_result,
                "timestamp": self._get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"测试{provider}客户端连接失败: {str(e)} | Failed to test {provider} client connection: {str(e)}")
            return {
                "success": False,
                "provider": provider,
                "connected": False,
                "error": str(e),
                "timestamp": self._get_current_timestamp()
            }
    
    def _test_openai_compatible_client(self, client, provider: str) -> Dict[str, Any]:
        """测试OpenAI兼容客户端 | Test OpenAI compatible client"""
        try:
            # 尝试调用聊天完成API
            # Try to call chat completions API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            return {
                "connected": True,
                "message": f"{provider} API连接成功 | {provider} API connection successful",
                "response_type": type(response).__name__,
                "has_content": hasattr(response.choices[0].message, 'content') if hasattr(response, 'choices') else False
            }
            
        except Exception as e:
            # 如果标准API调用失败，检查是否为模拟客户端
            # If standard API call fails, check if it's a mock client
            if hasattr(client, 'provider_name') and "Mock" in str(type(client)):
                return {
                    "connected": False,
                    "message": f"{provider} 模拟客户端检测到 - 真实API客户端为必需 | {provider} mock client detected - real API client required",
                    "is_mock": True,
                    "error": "Mock clients are not allowed. Real API credentials and libraries required."
                }
            else:
                raise e
    
    def _test_anthropic_client(self, client) -> Dict[str, Any]:
        """测试Anthropic客户端 | Test Anthropic client"""
        try:
            # Anthropic API测试
            # Anthropic API test
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=5,
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            return {
                "connected": True,
                "message": "Anthropic API连接成功 | Anthropic API connection successful",
                "has_content": hasattr(response, 'content') and len(response.content) > 0
            }
            
        except Exception as e:
            # 检查是否为模拟客户端
            # Check if it's a mock client
            if hasattr(client, 'provider_name') and "Mock" in str(type(client)):
                return {
                    "connected": False,
                    "message": "Anthropic模拟客户端检测到 - 真实API客户端为必需 | Anthropic mock client detected - real API client required",
                    "is_mock": True,
                    "error": "Mock clients are not allowed. Real API credentials and libraries required."
                }
            else:
                raise e
    
    def _test_google_genai_client(self, client) -> Dict[str, Any]:
        """测试Google Generative AI客户端 | Test Google Generative AI client"""
        try:
            # Google Generative AI测试
            # Google Generative AI test
            response = client.generate_content("Hello")
            
            return {
                "connected": True,
                "message": "Google Generative AI连接成功 | Google Generative AI connection successful",
                "has_text": hasattr(response, 'text') and bool(response.text)
            }
            
        except Exception as e:
            # 检查是否为模拟客户端
            # Check if it's a mock client
            if hasattr(client, 'model_name') and "Mock" in str(type(client)):
                return {
                    "connected": False,
                    "message": "Google Generative AI模拟客户端检测到 - 真实API客户端为必需 | Google Generative AI mock client detected - real API client required",
                    "is_mock": True,
                    "error": "Mock clients are not allowed. Real API credentials and libraries required."
                }
            else:
                raise e
    
    def _test_cohere_client(self, client) -> Dict[str, Any]:
        """测试Cohere客户端 | Test Cohere client"""
        try:
            # Cohere API测试
            # Cohere API test
            response = client.chat(
                model="command",
                message="Hello",
                max_tokens=5
            )
            
            return {
                "connected": True,
                "message": "Cohere API连接成功 | Cohere API connection successful",
                "has_text": hasattr(response, 'text') and bool(response.text)
            }
            
        except Exception as e:
            # 检查是否为模拟客户端
            # Check if it's a mock client
            if hasattr(client, 'provider_name') and "Mock" in str(type(client)):
                return {
                    "connected": False,
                    "message": "Cohere模拟客户端检测到 - 真实API客户端为必需 | Cohere mock client detected - real API client required",
                    "is_mock": True,
                    "error": "Mock clients are not allowed. Real API credentials and libraries required."
                }
            else:
                raise e
    
    def _test_mistral_client(self, client) -> Dict[str, Any]:
        """测试Mistral AI客户端 | Test Mistral AI client"""
        try:
            # Mistral AI API测试
            # Mistral AI API test
            response = client.chat.completions.create(
                model="mistral-tiny",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            return {
                "connected": True,
                "message": "Mistral AI API连接成功 | Mistral AI API connection successful",
                "has_content": hasattr(response.choices[0].message, 'content') if hasattr(response, 'choices') else False
            }
            
        except Exception as e:
            # 检查是否为模拟客户端
            # Check if it's a mock client
            if hasattr(client, 'provider_name') and "Mock" in str(type(client)):
                return {
                    "connected": False,
                    "message": "Mistral AI模拟客户端检测到 - 真实API客户端为必需 | Mistral AI mock client detected - real API client required",
                    "is_mock": True,
                    "error": "Mock clients are not allowed. Real API credentials and libraries required."
                }
            else:
                raise e
    
    def _test_domestic_client(self, client, provider: str) -> Dict[str, Any]:
        """测试国内API客户端 | Test domestic API client"""
        # 国内API提供商测试逻辑
        # Domestic API provider test logic
        if hasattr(client, 'provider_name') and "Mock" in str(type(client)):
            return {
                "connected": False,
                "message": f"{provider}模拟客户端检测到 - 真实API客户端为必需 | {provider} mock client detected - real API client required",
                "is_mock": True,
                "error": "Mock clients are not allowed. Real API credentials and libraries required."
            }
        else:
            return {
                "connected": True,
                "message": f"{provider}客户端已创建 | {provider} client created",
                "client_type": type(client).__name__
            }

# 全局客户端工厂实例 | Global client factory instance
_global_client_factory = None

def get_global_client_factory() -> APIClientFactory:
    """获取全局客户端工厂实例 | Get global client factory instance"""
    global _global_client_factory
    if _global_client_factory is None:
        _global_client_factory = APIClientFactory()
    return _global_client_factory

def set_global_client_factory(factory: APIClientFactory):
    """设置全局客户端工厂实例 | Set global client factory instance"""
    global _global_client_factory
    _global_client_factory = factory

# 装饰器：自动创建和管理API客户端 | Decorator: Auto-create and manage API clients
def with_api_client(provider: str, config_key: str = "api_config"):
    """API客户端装饰器 | API client decorator
    
    自动为方法提供API客户端，减少重复的客户端创建代码
    Automatically provide API client for methods, reducing duplicate client creation code
    
    Args:
        provider: API提供商名称
        config_key: 配置参数在kwargs中的键名
        
    Returns:
        装饰器函数 | Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取配置
            # Get configuration
            config = kwargs.get(config_key, {})
            if not config and len(args) > 1:
                # 尝试从位置参数获取配置
                # Try to get configuration from positional arguments
                for arg in args:
                    if isinstance(arg, dict):
                        config = arg
                        break
            
            # 创建或获取客户端
            # Create or get client
            factory = get_global_client_factory()
            client = factory.get_or_create_client(provider, config)
            
            # 将客户端添加到kwargs
            # Add client to kwargs
            kwargs["api_client"] = client
            
            # 调用原始函数
            # Call original function
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

# 示例用法 | Example usage
if __name__ == "__main__":
    import logging
    
    # 配置日志
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建客户端工厂
    # Create client factory
    factory = APIClientFactory()
    
    # 示例1: 创建OpenAI客户端
    # Example 1: Create OpenAI client
    logger.info("示例1: 创建OpenAI客户端 | Example 1: Create OpenAI client")
    openai_config = {
        "api_key": os.environ.get("OPENAI_API_KEY", "sk-test1234567890abcdefghijklmnopqrstuvwxyz"),
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    }
    
    try:
        openai_client = factory.create_client("openai", openai_config)
        logger.info(f"OpenAI客户端创建成功: {type(openai_client)} | OpenAI client created successfully: {type(openai_client)}")
    except Exception as e:
        logger.error(f"OpenAI客户端创建失败: {e} | OpenAI client creation failed: {e}")
    
    # 示例2: 测试连接
    # Example 2: Test connection
    logger.info("\n示例2: 测试客户端连接 | Example 2: Test client connection")
    test_result = factory.test_client_connection("openai", openai_config)
    logger.info(f"连接测试结果: {test_result} | Connection test result: {test_result}")
    
    # 示例3: 列出所有客户端
    # Example 3: List all clients
    logger.info("\n示例3: 列出所有客户端 | Example 3: List all clients")
    clients_info = factory.list_clients()
    logger.info(f"客户端信息: {clients_info} | Client info: {clients_info}")
    
    # 示例4: 使用装饰器
    # Example 4: Using decorator
    logger.info("\n示例4: 使用API客户端装饰器 | Example 4: Using API client decorator")
    
    @with_api_client("openai", "api_config")
    def chat_with_openai(api_client=None, **kwargs):
        """使用OpenAI客户端聊天 | Chat using OpenAI client"""
        if api_client:
            try:
                response = api_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello, this is a test."}],
                    max_tokens=10
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"聊天失败: {str(e)} | Chat failed: {str(e)}"
        else:
            return "未找到API客户端 | API client not found"
    
    # 测试装饰器函数
    # Test decorator function
    result = chat_with_openai(api_config=openai_config)
    logger.info(f"装饰器测试结果: {result} | Decorator test result: {result}")
    
    # 示例5: 清除缓存
    # Example 5: Clear cache
    logger.info("\n示例5: 清除客户端缓存 | Example 5: Clear client cache")
    factory.clear_cache("openai")
    logger.info("OpenAI客户端缓存已清除 | OpenAI client cache cleared")
    
    logger.info("\nAPI客户端工厂示例完成 | API client factory example completed")
