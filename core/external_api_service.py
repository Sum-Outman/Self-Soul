"""
统一外部API服务类 - Unified External API Service
提供统一的主流AI API服务接口，支持OpenAI、Anthropic、Google AI、AWS、Azure等
Provides unified mainstream AI API service interface, supporting OpenAI, Anthropic, Google AI, AWS, Azure, etc.
"""

import logging
import json
import os
import requests
import threading
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass

# 导入核心模块
# Import core modules
try:
    from core.error_handling import error_handler
    from core.system_settings_manager import system_settings_manager
    from core.model_registry import get_model_registry
except ImportError:
    # 提供默认的模拟实现，以防核心模块不可用
    # Provide default mock implementations in case core modules are not available
    class ErrorHandler:
        def handle_error(self, error, module, message):
            logging.error(f"{module} - {message}: {str(error)}")
        def log_info(self, message, module):
            logging.info(f"{module} - {message}")
    error_handler = ErrorHandler()
    
    class SystemSettingsManager:
        def __init__(self):
            self.settings = {}
        def get_system_setting(self, key, default=None):
            return self.settings.get(key, default)
        def update_system_setting(self, key, value):
            self.settings[key] = value
        def get_model_setting(self, model_id, default=None, use_cache=False):
            return self.settings.get(f"model_{model_id}", default)
        def update_model_setting(self, model_id, settings):
            self.settings[f"model_{model_id}"] = settings
    system_settings_manager = SystemSettingsManager()
    
    class ModelRegistry:
        def __init__(self):
            self.models = {}
        def is_model_registered(self, model_id):
            return model_id in self.models
        def register_model(self, model_id, settings):
            self.models[model_id] = settings
    _model_registry = ModelRegistry()
    def get_model_registry():
        return _model_registry

# API特定导入
# API specific imports
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

try:
    import anthropic  # type: ignore
except ImportError:
    # If anthropic is not installed, create a mock object
    class MockAnthropic:
        class Client:
            def __init__(self, api_key):
                self.api_key = api_key
            def messages_create(self, **kwargs):
                return {"content": [{"text": "Mock Anthropic response"}]}
    anthropic = MockAnthropic()

try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    # If google.generativeai is not installed, create a mock object
    class MockGenAI:
        class GenerativeModel:
            def __init__(self, model_name):
                self.model_name = model_name
            def generate_content(self, content, **kwargs):
                class MockResponse:
                    def __init__(self):
                        self.text = "Mock Google AI response"
                return MockResponse()
    genai = MockGenAI()

try:
    import boto3  # type: ignore
except ImportError:
    # If boto3 is not installed, create a mock object
    class MockBoto3:
        def client(self, service_name, **kwargs):
            class MockClient:
                def detect_labels(self, **kwargs):
                    return {"Labels": []}
                def detect_text(self, **kwargs):
                    return {"TextDetections": []}
                def start_label_detection(self, **kwargs):
                    return {"JobId": "mock_job_id"}
            return MockClient()
    boto3 = MockBoto3()

try:
    import replicate  # type: ignore
except ImportError:
    # If replicate is not installed, create a mock object
    class MockReplicate:
        def run(self, model, **kwargs):
            return "Mock Replicate response"
    replicate = MockReplicate()

try:
    import ollama  # type: ignore
except ImportError:
    # If ollama is not installed, create a mock object
    class MockOllama:
        def chat(self, **kwargs):
            return {"message": {"content": "Mock Ollama response"}}
        def generate(self, **kwargs):
            return {"response": "Mock Ollama generation"}
    ollama = MockOllama()


@dataclass
class APIConnectionStatus:
    """API连接状态数据类
    Data class for tracking API connection status"""
    connected: bool = False
    last_check: Optional[str] = None
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    api_version: Optional[str] = None
    rate_limit_info: Optional[Dict[str, Any]] = None

class ExternalAPIService:
    """统一外部API服务
    Unified External API Service
    
    功能：统一管理所有主流AI API的配置、认证和调用，支持每个模型单独配置外部API
    Function: Unified management of all mainstream AI APIs configuration, authentication and calls, supporting per-model external API configuration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化外部API服务 | Initialize external API service"""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # 支持的API提供商
        # Supported API providers
        self.supported_providers = ["openai", "anthropic", "google", "aws", "azure", "huggingface", "cohere", "mistral", "replicate", "ollama",
                                    "deepseek", "siliconflow", "zhipu", "baidu", "alibaba", "moonshot", "yi", "tencent"]
        
        # API服务状态 | API service status
        self.services = {
            "openai": {
                "chat": None,
                "vision": None,
                "configured": False
            },
            "anthropic": {
                "chat": None,
                "configured": False
            },
            "google": {
                "ai": None,
                "vision": None,
                "video": None,
                "configured": False
            },
            "aws": {
                "bedrock": None,
                "rekognition": None,
                "rekognition_video": None,
                "configured": False
            },
            "azure": {
                "openai": None,
                "vision": None,
                "video": None,
                "configured": False
            },
            "huggingface": {
                "inference": None,
                "configured": False
            },
            "cohere": {
                "chat": None,
                "configured": False
            },
            "mistral": {
                "chat": None,
                "configured": False
            },
            "replicate": {
                "inference": None,
                "configured": False
            },
            "ollama": {
                "inference": None,
                "configured": False
            },
            "deepseek": {
                "chat": None,
                "configured": False
            },
            "siliconflow": {
                "chat": None,
                "configured": False
            },
            "zhipu": {
                "chat": None,
                "configured": False
            },
            "baidu": {
                "chat": None,
                "configured": False
            },
            "alibaba": {
                "chat": None,
                "configured": False
            },
            "moonshot": {
                "chat": None,
                "configured": False
            },
            "yi": {
                "chat": None,
                "configured": False
            },
            "tencent": {
                "chat": None,
                "configured": False
            }
        }
        
        # API配置缓存 | API configuration cache
        self.api_configs = {}
        
        # 连接状态跟踪 | Connection status tracking
        self.connection_status = {}
        self.api_clients = {}
        
        # 线程锁 | Thread lock
        self.lock = threading.Lock()
        
        # 模型注册表实例
        # Model registry instance
        self.model_registry = get_model_registry()
        
        # 初始化默认配置
        # Initialize default configurations
        self._initialize_default_configs()
        
        # 初始化所有API服务 | Initialize all API services
        self._initialize_all_services()
        
        self.logger.info("统一外部API服务初始化完成 | Unified external API service initialized")
    
    def initialize(self):
        """初始化外部API服务（兼容接口） | Initialize external API service (compatibility interface)"""
        try:
            # 检查是否已经初始化
            # Check if already initialized
            if hasattr(self, 'initialized') and self.initialized:
                return True
                
            # 确保所有服务都已初始化
            # Ensure all services are initialized
            if not hasattr(self, 'services') or not self.services:
                self._initialize_all_services()
                
            self.initialized = True
            self.logger.info("外部API服务已初始化 | External API service initialized")
            return True
        except Exception as e:
            self.logger.error(f"外部API服务初始化失败: {str(e)} | External API service initialization failed: {str(e)}")
            return False
        
    def _initialize_default_configs(self):
        """初始化默认API配置 | Initialize default API configurations"""
        # 为每个支持的提供商创建默认配置
        # Create default configuration for each supported provider
        self.api_configs = {
            "openai": {
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o",
                "timeout": 60,
                "max_tokens": 4096,
                "temperature": 0.7
            },
            "anthropic": {
                "api_key": "",
                "model": "claude-3-opus-20240229",
                "timeout": 60,
                "max_tokens": 4096,
                "temperature": 0.7
            },
            "google": {
                "api_key": "",
                "model": "gemini-pro",
                "timeout": 60,
                "max_tokens": 4096,
                "temperature": 0.7
            },
            "aws": {
                "access_key_id": "",
                "secret_access_key": "",
                "region": "us-east-1",
                "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                "timeout": 60
            },
            "azure": {
                "api_key": "",
                "endpoint": "",
                "deployment_id": "",
                "api_version": "2024-02-15-preview",
                "timeout": 60
            },
            "huggingface": {
                "api_key": "",
                "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "timeout": 120
            },
            "cohere": {
                "api_key": "",
                "model": "command-r-plus",
                "timeout": 60,
                "max_tokens": 4096,
                "temperature": 0.7
            },
            "mistral": {
                "api_key": "",
                "model": "mistral-large-latest",
                "timeout": 60,
                "max_tokens": 4096,
                "temperature": 0.7
            },
            "replicate": {
                "api_key": "",
                "model": "meta/llama-3-70b-instruct",
                "timeout": 120
            },
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "llama3",
                "timeout": 60
            },
            "deepseek": {
                "api_key": "",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "timeout": 60,
                "max_tokens": 4096,
                "temperature": 0.7
            },
            "siliconflow": {
                "api_key": "",
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "Qwen2.5-7B-Instruct",
                "timeout": 60,
                "max_tokens": 4096,
                "temperature": 0.7
            },
            "zhipu": {
                "api_key": "",
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "model": "glm-4",
                "timeout": 60,
                "max_tokens": 4096,
                "temperature": 0.7
            },
            "baidu": {
                "api_key": "",
                "base_url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop",
                "model": "ERNIE-Bot-4",
                "timeout": 60,
                "max_tokens": 4096,
                "temperature": 0.7
            },
            "alibaba": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen-max",
                "timeout": 60,
                "max_tokens": 4096,
                "temperature": 0.7
            },
            "moonshot": {
                "api_key": "",
                "base_url": "https://api.moonshot.cn/v1",
                "model": "moonshot-v1-8k",
                "timeout": 60,
                "max_tokens": 4096,
                "temperature": 0.7
            },
            "yi": {
                "api_key": "",
                "base_url": "https://api.lingyiwanwu.com/v1",
                "model": "yi-large",
                "timeout": 60,
                "max_tokens": 4096,
                "temperature": 0.7
            },
            "tencent": {
                "api_key": "",
                "base_url": "https://hunyuan.cloud.tencent.com",
                "model": "hunyuan-standard",
                "timeout": 60,
                "max_tokens": 4096,
                "temperature": 0.7
            }
        }
        
        # 从系统设置加载已保存的配置
        # Load saved configurations from system settings
        self._load_saved_configs()
        
    def _load_saved_configs(self):
        """从系统设置加载已保存的API配置 | Load saved API configurations from system settings"""
        try:
            # 尝试从系统设置管理器加载配置
            # Try to load configuration from system settings manager
            for provider in self.supported_providers:
                try:
                    saved_config = system_settings_manager.get_model_setting(
                        f"external_api_{provider}", default=None
                    )
                    if saved_config:
                        # 更新API配置
                        # Update API configuration
                        self.api_configs[provider].update(saved_config)
                        self.logger.info(f"已加载{provider}的保存配置 | Loaded saved configuration for {provider}")
                except Exception as e:
                    self.logger.warning(f"加载{provider}配置失败: {str(e)} | Failed to load {provider} configuration: {str(e)}")
        except Exception as e:
            self.logger.error(f"加载保存的API配置失败: {str(e)} | Failed to load saved API configurations: {str(e)}")
    
    def _initialize_all_services(self):
        """初始化所有API服务 | Initialize all API services"""
        try:
            # 从配置加载API设置 | Load API settings from config
            api_config = self.config.get("external_apis", {})
            
            # 初始化OpenAI服务 | Initialize OpenAI services
            self._initialize_openai_services(api_config.get("openai", {}))
            
            # 初始化Anthropic服务 | Initialize Anthropic services
            self._initialize_anthropic_services(api_config.get("anthropic", {}))
            
            # 初始化Google AI服务 | Initialize Google AI services
            self._initialize_google_ai_services(api_config.get("google_ai", {}))
            
            # 初始化Google服务 | Initialize Google services
            self._initialize_google_services(api_config.get("google", {}))
            
            # 初始化AWS服务 | Initialize AWS services
            self._initialize_aws_services(api_config.get("aws", {}))
            
            # 初始化Azure服务 | Initialize Azure services
            self._initialize_azure_services(api_config.get("azure", {}))
            
            # 初始化HuggingFace服务 | Initialize HuggingFace services
            self._initialize_huggingface_services(api_config.get("huggingface", {}))
            
            # 初始化Cohere服务 | Initialize Cohere services
            self._initialize_cohere_services(api_config.get("cohere", {}))
            
            # 初始化Mistral服务 | Initialize Mistral services
            self._initialize_mistral_services(api_config.get("mistral", {}))
            
            # 初始化Ollama服务 | Initialize Ollama services
            self._initialize_ollama_services(api_config.get("ollama", {}))
            
            # 初始化国内供应商服务 | Initialize domestic provider services
            self._initialize_deepseek_services(api_config.get("deepseek", {}))
            self._initialize_siliconflow_services(api_config.get("siliconflow", {}))
            self._initialize_zhipu_services(api_config.get("zhipu", {}))
            self._initialize_baidu_services(api_config.get("baidu", {}))
            self._initialize_alibaba_services(api_config.get("alibaba", {}))
            self._initialize_moonshot_services(api_config.get("moonshot", {}))
            self._initialize_yi_services(api_config.get("yi", {}))
            self._initialize_tencent_services(api_config.get("tencent", {}))
            
        except Exception as e:
            self.logger.error(f"初始化API服务失败: {str(e)} | Failed to initialize API services: {str(e)}")
    
    def _initialize_google_services(self, google_config: Dict[str, Any]):
        """初始化Google API服务 | Initialize Google API services"""
        try:
            # Google Vision API
            vision_config = google_config.get("vision", {})
            if vision_config.get("api_key"):
                try:
                    from google.cloud import vision
                    client = vision.ImageAnnotatorClient.from_service_account_info({
                        "type": "service_account",
                        "project_id": vision_config.get("project_id", ""),
                        "private_key_id": vision_config.get("private_key_id", ""),
                        "private_key": vision_config.get("private_key", ""),
                        "client_email": vision_config.get("client_email", ""),
                        "client_id": vision_config.get("client_id", ""),
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                        "client_x509_cert_url": vision_config.get("client_x509_cert_url", "")
                    })
                    self.services["google"]["vision"] = client
                    self.services["google"]["configured"] = True
                    self.logger.info("Google Vision API配置完成 | Google Vision API configured")
                except ImportError:
                    self.logger.warning("google-cloud-vision未安装，无法使用Google Vision API | google-cloud-vision not installed, cannot use Google Vision API")
                except Exception as e:
                    self.logger.error(f"Google Vision API配置失败: {str(e)} | Google Vision API configuration failed: {str(e)}")
            
            # Google Video Intelligence API
            video_config = google_config.get("video", {})
            if video_config.get("api_key"):
                try:
                    from google.cloud import videointelligence
                    client = videointelligence.VideoIntelligenceServiceClient.from_service_account_info({
                        "type": "service_account",
                        "project_id": video_config.get("project_id", ""),
                        "private_key_id": video_config.get("private_key_id", ""),
                        "private_key": video_config.get("private_key", ""),
                        "client_email": video_config.get("client_email", ""),
                        "client_id": video_config.get("client_id", ""),
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                        "client_x509_cert_url": video_config.get("client_x509_cert_url", "")
                    })
                    self.services["google"]["video"] = client
                    self.services["google"]["configured"] = True
                    self.logger.info("Google Video API配置完成 | Google Video API configured")
                except ImportError:
                    self.logger.warning("google-cloud-videointelligence未安装，无法使用Google Video API | google-cloud-videointelligence not installed, cannot use Google Video API")
                except Exception as e:
                    self.logger.error(f"Google Video API配置失败: {str(e)} | Google Video API configuration failed: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Google服务初始化失败: {str(e)} | Google service initialization failed: {str(e)}")
    
    def _initialize_aws_services(self, aws_config: Dict[str, Any]):
        """初始化AWS API服务 | Initialize AWS API services"""
        try:
            # AWS Rekognition
            rekognition_config = aws_config.get("rekognition", {})
            if rekognition_config.get("access_key") and rekognition_config.get("secret_key"):
                try:
                    if boto3 is not None:
                        self.services["aws"]["rekognition"] = boto3.client(
                            'rekognition',
                            aws_access_key_id=rekognition_config["access_key"],
                            aws_secret_access_key=rekognition_config["secret_key"],
                            region_name=rekognition_config.get("region", "us-east-1")
                        )
                        self.services["aws"]["configured"] = True
                        self.logger.info("AWS Rekognition配置完成 | AWS Rekognition configured")
                    else:
                        self.logger.warning("boto3未安装，无法使用AWS Rekognition | boto3 not installed, cannot use AWS Rekognition")
                except Exception as e:
                    self.logger.error(f"AWS Rekognition配置失败: {str(e)} | AWS Rekognition configuration failed: {str(e)}")
            
            # AWS Rekognition Video
            video_config = aws_config.get("rekognition_video", {})
            if video_config.get("access_key") and video_config.get("secret_key"):
                try:
                    if boto3 is not None:
                        self.services["aws"]["rekognition_video"] = boto3.client(
                            'rekognition',
                            aws_access_key_id=video_config["access_key"],
                            aws_secret_access_key=video_config["secret_key"],
                            region_name=video_config.get("region", "us-east-1")
                        )
                        self.services["aws"]["configured"] = True
                        self.logger.info("AWS Rekognition Video配置完成 | AWS Rekognition Video configured")
                    else:
                        self.logger.warning("boto3未安装，无法使用AWS Rekognition Video | boto3 not installed, cannot use AWS Rekognition Video")
                except Exception as e:
                    self.logger.error(f"AWS Rekognition Video配置失败: {str(e)} | AWS Rekognition Video configuration failed: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"AWS服务初始化失败: {str(e)} | AWS service initialization failed: {str(e)}")
    
    def _initialize_openai_services(self, openai_config: Dict[str, Any]):
        """初始化OpenAI API服务 | Initialize OpenAI API services"""
        try:
            # OpenAI Chat API
            chat_config = openai_config.get("chat", {})
            if chat_config.get("api_key"):
                self.services["openai"]["chat"] = {
                    "api_key": chat_config["api_key"],
                    "base_url": chat_config.get("base_url", "https://api.openai.com/v1"),
                    "model": chat_config.get("model", "gpt-4"),
                    "configured": True
                }
                self.services["openai"]["configured"] = True
                self.logger.info("OpenAI Chat API配置完成 | OpenAI Chat API configured")
            
            # OpenAI Vision API
            vision_config = openai_config.get("vision", {})
            if vision_config.get("api_key"):
                self.services["openai"]["vision"] = {
                    "api_key": vision_config["api_key"],
                    "base_url": vision_config.get("base_url", "https://api.openai.com/v1"),
                    "model": vision_config.get("model", "gpt-4-vision-preview"),
                    "configured": True
                }
                self.services["openai"]["configured"] = True
                self.logger.info("OpenAI Vision API配置完成 | OpenAI Vision API configured")
                
        except Exception as e:
            self.logger.error(f"OpenAI服务初始化失败: {str(e)} | OpenAI service initialization failed: {str(e)}")
    
    def _initialize_anthropic_services(self, anthropic_config: Dict[str, Any]):
        """初始化Anthropic API服务 | Initialize Anthropic API services"""
        try:
            # Anthropic Chat API
            chat_config = anthropic_config.get("chat", {})
            if chat_config.get("api_key"):
                self.services["anthropic"]["chat"] = {
                    "api_key": chat_config["api_key"],
                    "base_url": chat_config.get("base_url", "https://api.anthropic.com"),
                    "model": chat_config.get("model", "claude-3-opus-20240229"),
                    "configured": True
                }
                self.services["anthropic"]["configured"] = True
                self.logger.info("Anthropic Chat API配置完成 | Anthropic Chat API configured")
                
        except Exception as e:
            self.logger.error(f"Anthropic服务初始化失败: {str(e)} | Anthropic service initialization failed: {str(e)}")
    
    def _initialize_google_ai_services(self, google_ai_config: Dict[str, Any]):
        """初始化Google AI API服务 | Initialize Google AI API services"""
        try:
            # Google AI API (Gemini)
            ai_config = google_ai_config.get("ai", {})
            if ai_config.get("api_key"):
                self.services["google"]["ai"] = {
                    "api_key": ai_config["api_key"],
                    "base_url": ai_config.get("base_url", "https://generativelanguage.googleapis.com/v1beta"),
                    "model": ai_config.get("model", "gemini-pro"),
                    "configured": True
                }
                self.services["google"]["configured"] = True
                self.logger.info("Google AI API配置完成 | Google AI API configured")
                
        except Exception as e:
            self.logger.error(f"Google AI服务初始化失败: {str(e)} | Google AI service initialization failed: {str(e)}")
    
    def _initialize_huggingface_services(self, huggingface_config: Dict[str, Any]):
        """初始化HuggingFace API服务 | Initialize HuggingFace API services"""
        try:
            # HuggingFace Inference API
            inference_config = huggingface_config.get("inference", {})
            if inference_config.get("api_key"):
                self.services["huggingface"]["inference"] = {
                    "api_key": inference_config["api_key"],
                    "base_url": inference_config.get("base_url", "https://api-inference.huggingface.co"),
                    "configured": True
                }
                self.services["huggingface"]["configured"] = True
                self.logger.info("HuggingFace Inference API配置完成 | HuggingFace Inference API configured")
                
        except Exception as e:
            self.logger.error(f"HuggingFace服务初始化失败: {str(e)} | HuggingFace service initialization failed: {str(e)}")
    
    def _initialize_cohere_services(self, cohere_config: Dict[str, Any]):
        """初始化Cohere API服务 | Initialize Cohere API services"""
        try:
            # Cohere Chat API
            chat_config = cohere_config.get("chat", {})
            if chat_config.get("api_key"):
                self.services["cohere"]["chat"] = {
                    "api_key": chat_config["api_key"],
                    "base_url": chat_config.get("base_url", "https://api.cohere.ai/v1"),
                    "model": chat_config.get("model", "command"),
                    "configured": True
                }
                self.services["cohere"]["configured"] = True
                self.logger.info("Cohere Chat API配置完成 | Cohere Chat API configured")
                
        except Exception as e:
            self.logger.error(f"Cohere服务初始化失败: {str(e)} | Cohere service initialization failed: {str(e)}")
    
    def _initialize_mistral_services(self, mistral_config: Dict[str, Any]):
        """初始化Mistral API服务 | Initialize Mistral API services"""
        try:
            # Mistral Chat API
            chat_config = mistral_config.get("chat", {})
            if chat_config.get("api_key"):
                self.services["mistral"]["chat"] = {
                    "api_key": chat_config["api_key"],
                    "base_url": chat_config.get("base_url", "https://api.mistral.ai/v1"),
                    "model": chat_config.get("model", "mistral-large-latest"),
                    "configured": True
                }
                self.services["mistral"]["configured"] = True
                self.logger.info("Mistral Chat API配置完成 | Mistral Chat API configured")
                
        except Exception as e:
            self.logger.error(f"Mistral服务初始化失败: {str(e)} | Mistral service initialization failed: {str(e)}")
    
    def _initialize_ollama_services(self, ollama_config: Dict[str, Any]):
        """初始化Ollama API服务 | Initialize Ollama API services"""
        try:
            # Ollama API
            if ollama_config.get("base_url") or ollama_config.get("model"):
                self.services["ollama"]["inference"] = {
                    "base_url": ollama_config.get("base_url", "http://localhost:11434"),
                    "model": ollama_config.get("model", "llama3"),
                    "configured": True
                }
                self.services["ollama"]["configured"] = True
                self.logger.info("Ollama API配置完成 | Ollama API configured")
                
        except Exception as e:
            self.logger.error(f"Ollama服务初始化失败: {str(e)} | Ollama service initialization failed: {str(e)}")
    
    def _initialize_deepseek_services(self, deepseek_config: Dict[str, Any]):
        """初始化DeepSeek API服务 | Initialize DeepSeek API services"""
        try:
            # DeepSeek Chat API
            chat_config = deepseek_config.get("chat", {})
            if chat_config.get("api_key"):
                self.services["deepseek"]["chat"] = {
                    "api_key": chat_config["api_key"],
                    "base_url": chat_config.get("base_url", "https://api.deepseek.com"),
                    "model": chat_config.get("model", "deepseek-chat"),
                    "configured": True
                }
                self.services["deepseek"]["configured"] = True
                self.logger.info("DeepSeek Chat API配置完成 | DeepSeek Chat API configured")
                
        except Exception as e:
            self.logger.error(f"DeepSeek服务初始化失败: {str(e)} | DeepSeek service initialization failed: {str(e)}")
    
    def _initialize_siliconflow_services(self, siliconflow_config: Dict[str, Any]):
        """初始化SiliconFlow API服务 | Initialize SiliconFlow API services"""
        try:
            # SiliconFlow Chat API
            chat_config = siliconflow_config.get("chat", {})
            if chat_config.get("api_key"):
                self.services["siliconflow"]["chat"] = {
                    "api_key": chat_config["api_key"],
                    "base_url": chat_config.get("base_url", "https://api.siliconflow.cn/v1"),
                    "model": chat_config.get("model", "Qwen2.5-7B-Instruct"),
                    "configured": True
                }
                self.services["siliconflow"]["configured"] = True
                self.logger.info("SiliconFlow Chat API配置完成 | SiliconFlow Chat API configured")
                
        except Exception as e:
            self.logger.error(f"SiliconFlow服务初始化失败: {str(e)} | SiliconFlow service initialization failed: {str(e)}")
    
    def _initialize_zhipu_services(self, zhipu_config: Dict[str, Any]):
        """初始化Zhipu AI API服务 | Initialize Zhipu AI API services"""
        try:
            # Zhipu AI Chat API
            chat_config = zhipu_config.get("chat", {})
            if chat_config.get("api_key"):
                self.services["zhipu"]["chat"] = {
                    "api_key": chat_config["api_key"],
                    "base_url": chat_config.get("base_url", "https://open.bigmodel.cn/api/paas/v4"),
                    "model": chat_config.get("model", "glm-4"),
                    "configured": True
                }
                self.services["zhipu"]["configured"] = True
                self.logger.info("Zhipu AI Chat API配置完成 | Zhipu AI Chat API configured")
                
        except Exception as e:
            self.logger.error(f"Zhipu AI服务初始化失败: {str(e)} | Zhipu AI service initialization failed: {str(e)}")
    
    def _initialize_baidu_services(self, baidu_config: Dict[str, Any]):
        """初始化Baidu ERNIE API服务 | Initialize Baidu ERNIE API services"""
        try:
            # Baidu ERNIE Chat API
            chat_config = baidu_config.get("chat", {})
            if chat_config.get("api_key"):
                self.services["baidu"]["chat"] = {
                    "api_key": chat_config["api_key"],
                    "base_url": chat_config.get("base_url", "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop"),
                    "model": chat_config.get("model", "ERNIE-Bot-4"),
                    "configured": True
                }
                self.services["baidu"]["configured"] = True
                self.logger.info("Baidu ERNIE Chat API配置完成 | Baidu ERNIE Chat API configured")
                
        except Exception as e:
            self.logger.error(f"Baidu ERNIE服务初始化失败: {str(e)} | Baidu ERNIE service initialization failed: {str(e)}")
    
    def _initialize_alibaba_services(self, alibaba_config: Dict[str, Any]):
        """初始化Alibaba Qwen API服务 | Initialize Alibaba Qwen API services"""
        try:
            # Alibaba Qwen Chat API
            chat_config = alibaba_config.get("chat", {})
            if chat_config.get("api_key"):
                self.services["alibaba"]["chat"] = {
                    "api_key": chat_config["api_key"],
                    "base_url": chat_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                    "model": chat_config.get("model", "qwen-max"),
                    "configured": True
                }
                self.services["alibaba"]["configured"] = True
                self.logger.info("Alibaba Qwen Chat API配置完成 | Alibaba Qwen Chat API configured")
                
        except Exception as e:
            self.logger.error(f"Alibaba Qwen服务初始化失败: {str(e)} | Alibaba Qwen service initialization failed: {str(e)}")
    
    def _initialize_moonshot_services(self, moonshot_config: Dict[str, Any]):
        """初始化Moonshot API服务 | Initialize Moonshot API services"""
        try:
            # Moonshot Chat API
            chat_config = moonshot_config.get("chat", {})
            if chat_config.get("api_key"):
                self.services["moonshot"]["chat"] = {
                    "api_key": chat_config["api_key"],
                    "base_url": chat_config.get("base_url", "https://api.moonshot.cn/v1"),
                    "model": chat_config.get("model", "moonshot-v1-8k"),
                    "configured": True
                }
                self.services["moonshot"]["configured"] = True
                self.logger.info("Moonshot Chat API配置完成 | Moonshot Chat API configured")
                
        except Exception as e:
            self.logger.error(f"Moonshot服务初始化失败: {str(e)} | Moonshot service initialization failed: {str(e)}")
    
    def _initialize_yi_services(self, yi_config: Dict[str, Any]):
        """初始化Yi API服务 | Initialize Yi API services"""
        try:
            # Yi Chat API
            chat_config = yi_config.get("chat", {})
            if chat_config.get("api_key"):
                self.services["yi"]["chat"] = {
                    "api_key": chat_config["api_key"],
                    "base_url": chat_config.get("base_url", "https://api.lingyiwanwu.com/v1"),
                    "model": chat_config.get("model", "yi-large"),
                    "configured": True
                }
                self.services["yi"]["configured"] = True
                self.logger.info("Yi Chat API配置完成 | Yi Chat API configured")
                
        except Exception as e:
            self.logger.error(f"Yi服务初始化失败: {str(e)} | Yi service initialization failed: {str(e)}")
    
    def _initialize_tencent_services(self, tencent_config: Dict[str, Any]):
        """初始化Tencent Hunyuan API服务 | Initialize Tencent Hunyuan API services"""
        try:
            # Tencent Hunyuan Chat API
            chat_config = tencent_config.get("chat", {})
            if chat_config.get("api_key"):
                self.services["tencent"]["chat"] = {
                    "api_key": chat_config["api_key"],
                    "base_url": chat_config.get("base_url", "https://hunyuan.cloud.tencent.com"),
                    "model": chat_config.get("model", "hunyuan-standard"),
                    "configured": True
                }
                self.services["tencent"]["configured"] = True
                self.logger.info("Tencent Hunyuan Chat API配置完成 | Tencent Hunyuan Chat API configured")
                
        except Exception as e:
            self.logger.error(f"Tencent Hunyuan服务初始化失败: {str(e)} | Tencent Hunyuan service initialization failed: {str(e)}")
    
    def _initialize_azure_services(self, azure_config: Dict[str, Any]):
        """初始化Azure API服务 | Initialize Azure API services"""
        try:
            # Azure Computer Vision
            vision_config = azure_config.get("vision", {})
            if vision_config.get("endpoint") and vision_config.get("subscription_key"):
                self.services["azure"]["vision"] = {
                    "endpoint": vision_config["endpoint"],
                    "subscription_key": vision_config["subscription_key"],
                    "configured": True
                }
                self.logger.info("Azure Vision配置完成 | Azure Vision configured")
            
            # Azure Video Analyzer
            video_config = azure_config.get("video", {})
            if video_config.get("endpoint") and video_config.get("subscription_key"):
                self.services["azure"]["video"] = {
                    "endpoint": video_config["endpoint"],
                    "subscription_key": video_config["subscription_key"],
                    "configured": True
                }
                self.logger.info("Azure Video配置完成 | Azure Video configured")
                
        except Exception as e:
            self.logger.error(f"Azure服务初始化失败: {str(e)} | Azure service initialization failed: {str(e)}")
    
    def analyze_image(self, image_data: Any, api_type: str = "google") -> Dict[str, Any]:
        """使用外部API分析图像 | Analyze image using external API
        
        Args:
            image_data: 图像数据 | Image data
            api_type: API类型 (google/aws/azure) | API type (google/aws/azure)
            
        Returns:
            分析结果 | Analysis result
        """
        try:
            if api_type == "google":
                return self._google_vision_analyze(image_data)
            elif api_type == "aws":
                return self._aws_rekognition_analyze(image_data)
            elif api_type == "azure":
                return self._azure_vision_analyze(image_data)
            else:
                return {"error": f"不支持的API类型: {api_type} | Unsupported API type: {api_type}"}
                
        except Exception as e:
            self.logger.error(f"图像分析失败: {str(e)} | Image analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def analyze_video(self, video_data: Any, api_type: str = "google") -> Dict[str, Any]:
        """使用外部API分析视频 | Analyze video using external API
        
        Args:
            video_data: 视频数据 | Video data
            api_type: API类型 (google/aws/azure) | API type (google/aws/azure)
            
        Returns:
            分析结果 | Analysis result
        """
        try:
            if api_type == "google":
                return self._google_video_analyze(video_data)
            elif api_type == "aws":
                return self._aws_rekognition_video_analyze(video_data)
            elif api_type == "azure":
                return self._azure_video_analyze(video_data)
            else:
                return {"error": f"不支持的API类型: {api_type} | Unsupported API type: {api_type}"}
                
        except Exception as e:
            self.logger.error(f"视频分析失败: {str(e)} | Video analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _google_vision_analyze(self, image_data: Any) -> Dict[str, Any]:
        """使用Google Vision API分析图像 | Analyze image using Google Vision API"""
        if not self.services["google"]["vision"]:
            return {"error": "Google Vision API未配置 | Google Vision API not configured"}
        
        try:
            # 创建图像对象 | Create image object
            image = self.services["google"]["vision"].types.Image(content=image_data)
            
            # 配置特征检测 | Configure feature detection
            features = [
                self.services["google"]["vision"].types.Feature.Type.LABEL_DETECTION,
                self.services["google"]["vision"].types.Feature.Type.OBJECT_LOCALIZATION,
                self.services["google"]["vision"].types.Feature.Type.FACE_DETECTION,
                self.services["google"]["vision"].types.Feature.Type.TEXT_DETECTION
            ]
            
            # 执行分析 | Perform analysis
            response = self.services["google"]["vision"].annotate_image({
                'image': image,
                'features': [{'type': feature} for feature in features]
            })
            
            # 解析结果 | Parse results
            result = {
                "labels": [],
                "objects": [],
                "faces": [],
                "text": []
            }
            
            # 提取标签 | Extract labels
            for label in response.label_annotations:
                result["labels"].append({
                    "description": label.description,
                    "score": label.score,
                    "mid": label.mid
                })
            
            # 提取对象 | Extract objects
            for obj in response.localized_object_annotations:
                result["objects"].append({
                    "name": obj.name,
                    "score": obj.score,
                    "bounding_poly": [(vertex.x, vertex.y) for vertex in obj.bounding_poly.normalized_vertices]
                })
            
            # 提取人脸 | Extract faces
            for face in response.face_annotations:
                result["faces"].append({
                    "detection_confidence": face.detection_confidence,
                    "joy_likelihood": face.joy_likelihood,
                    "sorrow_likelihood": face.sorrow_likelihood,
                    "anger_likelihood": face.anger_likelihood,
                    "surprise_likelihood": face.surprise_likelihood
                })
            
            # 提取文本 | Extract text
            for text in response.text_annotations:
                result["text"].append({
                    "description": text.description,
                    "bounding_poly": [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
                })
            
            return {"success": True, "result": result, "source": "google_vision"}
            
        except Exception as e:
            self.logger.error(f"Google Vision API调用失败: {str(e)} | Google Vision API call failed: {str(e)}")
            return {"error": str(e)}
    
    def _aws_rekognition_analyze(self, image_data: Any) -> Dict[str, Any]:
        """使用AWS Rekognition分析图像 | Analyze image using AWS Rekognition"""
        if not self.services["aws"]["rekognition"]:
            return {"error": "AWS Rekognition未配置 | AWS Rekognition not configured"}
        
        try:
            # 调用AWS Rekognition API | Call AWS Rekognition API
            response = self.services["aws"]["rekognition"].detect_labels(
                Image={'Bytes': image_data},
                MaxLabels=10,
                MinConfidence=70.0
            )
            
            # 解析结果 | Parse results
            result = {
                "labels": [],
                "objects": []
            }
            
            for label in response['Labels']:
                result["labels"].append({
                    "name": label['Name'],
                    "confidence": label['Confidence'],
                    "instances": len(label.get('Instances', [])),
                    "parents": [parent['Name'] for parent in label.get('Parents', [])]
                })
                
                # 如果有实例，则视为检测到的对象
                if 'Instances' in label and label['Instances']:
                    for instance in label['Instances']:
                        result["objects"].append({
                            "name": label['Name'],
                            "confidence": instance['Confidence'],
                            "bounding_box": instance['BoundingBox']
                        })
            
            return {"success": True, "result": result, "source": "aws_rekognition"}
            
        except Exception as e:
            self.logger.error(f"AWS Rekognition API调用失败: {str(e)} | AWS Rekognition API call failed: {str(e)}")
            return {"error": str(e)}
    
    def _azure_vision_analyze(self, image_data: Any) -> Dict[str, Any]:
        """使用Azure Vision分析图像 | Analyze image using Azure Vision"""
        if not self.services["azure"]["vision"]:
            return {"error": "Azure Vision未配置 | Azure Vision not configured"}
        
        try:
            import requests
            
            endpoint = self.services["azure"]["vision"]["endpoint"]
            subscription_key = self.services["azure"]["vision"]["subscription_key"]
            
            # 构建API请求 | Build API request
            headers = {
                'Ocp-Apim-Subscription-Key': subscription_key,
                'Content-Type': 'application/octet-stream'
            }
            
            # 发送分析请求 | Send analysis request
            response = requests.post(
                f"{endpoint}/analyze",
                headers=headers,
                data=image_data,
                params={'visualFeatures': 'Categories,Description,Objects'}
            )
            
            if response.status_code != 200:
                return {"error": f"Azure API错误: {response.status_code} | Azure API error: {response.status_code}"}
            
            # 解析响应 | Parse response
            result = response.json()
            
            return {
                "success": True, 
                "result": result, 
                "source": "azure_vision"
            }
            
        except Exception as e:
            self.logger.error(f"Azure Vision API调用失败: {str(e)} | Azure Vision API call failed: {str(e)}")
            return {"error": str(e)}
    
    def _google_video_analyze(self, video_data: Any) -> Dict[str, Any]:
        """使用Google Video API分析视频 | Analyze video using Google Video API"""
        if not self.services["google"]["video"]:
            return {"error": "Google Video API未配置 | Google Video API not configured"}
        
        try:
            # 配置特征检测 | Configure feature detection
            features = [
                self.services["google"]["video"].enums.Feature.LABEL_DETECTION,
                self.services["google"]["video"].enums.Feature.OBJECT_TRACKING,
                self.services["google"]["video"].enums.Feature.SHOT_CHANGE_DETECTION
            ]
            
            # 执行视频分析 | Perform video analysis
            operation = self.services["google"]["video"].annotate_video(
                request={
                    "features": features,
                    "input_content": video_data
                }
            )
            
            # 等待操作完成 | Wait for operation to complete
            result = operation.result(timeout=90)
            
            # 解析结果 | Parse results
            annotations = result.annotation_results[0]
            
            actions = []
            for segment_label in annotations.segment_label_annotations:
                for segment in segment_label.segments:
                    confidence = segment.confidence
                    if confidence > 0.5:
                        actions.append({
                            "action": segment_label.entity.description,
                            "confidence": confidence
                        })
            
            objects = []
            for object_annotation in annotations.object_annotations:
                objects.append({
                    "object": object_annotation.entity.description,
                    "confidence": object_annotation.confidence
                })
            
            return {
                "success": True,
                "actions": actions,
                "objects": objects,
                "source": "google_video"
            }
            
        except Exception as e:
            self.logger.error(f"Google Video API调用失败: {str(e)} | Google Video API call failed: {str(e)}")
            return {"error": str(e)}
    
    def _aws_rekognition_video_analyze(self, video_data: Any) -> Dict[str, Any]:
        """使用AWS Rekognition Video分析视频 | Analyze video using AWS Rekognition Video"""
        if not self.services["aws"]["rekognition_video"]:
            return {"error": "AWS Rekognition Video未配置 | AWS Rekognition Video not configured"}
        
        try:
            # 启动标签检测 | Start label detection
            response = self.services["aws"]["rekognition_video"].start_label_detection(
                Video={'Bytes': video_data},
                MinConfidence=50.0
            )
            
            job_id = response['JobId']
            
            # 等待作业完成 | Wait for job to complete
            import time
            max_attempts = 30
            for attempt in range(max_attempts):
                result = self.services["aws"]["rekognition_video"].get_label_detection(JobId=job_id)
                status = result['JobStatus']
                
                if status == 'SUCCEEDED':
                    break
                elif status == 'FAILED':
                    return {"error": "AWS Rekognition作业失败 | AWS Rekognition job failed"}
                
                time.sleep(2)
            
            # 解析结果 | Parse results
            actions = []
            objects = []
            
            for label in result.get('Labels', []):
                label_name = label['Label']['Name']
                confidence = label['Label']['Confidence']
                
                if label_name.lower() in ['walking', 'running', 'jumping', 'dancing']:
                    actions.append({
                        "action": label_name,
                        "confidence": confidence / 100.0
                    })
                else:
                    objects.append({
                        "object": label_name,
                        "confidence": confidence / 100.0
                    })
            
            return {
                "success": True,
                "actions": actions,
                "objects": objects,
                "source": "aws_rekognition_video"
            }
            
        except Exception as e:
            self.logger.error(f"AWS Rekognition Video API调用失败: {str(e)} | AWS Rekognition Video API call failed: {str(e)}")
            return {"error": str(e)}
    
    def _azure_video_analyze(self, video_data: Any) -> Dict[str, Any]:
        """使用Azure Video分析视频 | Analyze video using Azure Video"""
        if not self.services["azure"]["video"]:
            return {"error": "Azure Video未配置 | Azure Video not configured"}
        
        try:
            import requests
            
            endpoint = self.services["azure"]["video"]["endpoint"]
            subscription_key = self.services["azure"]["video"]["subscription_key"]
            
            # 构建API请求 | Build API request
            headers = {
                'Ocp-Apim-Subscription-Key': subscription_key,
                'Content-Type': 'application/octet-stream'
            }
            
            # 发送分析请求 | Send analysis request
            response = requests.post(
                f"{endpoint}/analyze",
                headers=headers,
                data=video_data,
                params={'visualFeatures': 'Categories,Description,Objects'}
            )
            
            if response.status_code != 200:
                return {"error": f"Azure API错误: {response.status_code} | Azure API error: {response.status_code}"}
            
            # 解析响应 | Parse response
            result = response.json()
            
            return {
                "success": True,
                "result": result,
                "source": "azure_video"
            }
            
        except Exception as e:
            self.logger.error(f"Azure Video API调用失败: {str(e)} | Azure Video API call failed: {str(e)}")
            return {"error": str(e)}
    
    def generate_text(self, prompt: str, api_type: str = "openai", **kwargs) -> Dict[str, Any]:
        """使用外部API生成文本 | Generate text using external API
        
        Args:
            prompt: 提示文本 | Prompt text
            api_type: API类型 (openai/anthropic/google_ai/huggingface/cohere/mistral/deepseek/siliconflow/zhipu/baidu/alibaba/moonshot/yi/tencent/ollama) | API type
            **kwargs: 额外参数 | Additional parameters
            
        Returns:
            生成的文本结果 | Generated text result
        """
        try:
            if api_type == "openai":
                return self._openai_chat_analyze(prompt, **kwargs)
            elif api_type == "anthropic":
                return self._anthropic_chat_analyze(prompt, **kwargs)
            elif api_type == "google_ai":
                return self._google_ai_chat_analyze(prompt, **kwargs)
            elif api_type == "huggingface":
                return self._huggingface_inference_analyze(prompt, **kwargs)
            elif api_type == "cohere":
                return self._cohere_chat_analyze(prompt, **kwargs)
            elif api_type == "mistral":
                return self._mistral_chat_analyze(prompt, **kwargs)
            elif api_type == "deepseek":
                return self._deepseek_chat_analyze(prompt, **kwargs)
            elif api_type == "siliconflow":
                return self._siliconflow_chat_analyze(prompt, **kwargs)
            elif api_type == "zhipu":
                return self._zhipu_chat_analyze(prompt, **kwargs)
            elif api_type == "baidu":
                return self._baidu_chat_analyze(prompt, **kwargs)
            elif api_type == "alibaba":
                return self._alibaba_chat_analyze(prompt, **kwargs)
            elif api_type == "moonshot":
                return self._moonshot_chat_analyze(prompt, **kwargs)
            elif api_type == "yi":
                return self._yi_chat_analyze(prompt, **kwargs)
            elif api_type == "tencent":
                return self._tencent_chat_analyze(prompt, **kwargs)
            elif api_type == "ollama":
                return self._ollama_chat_analyze(prompt, **kwargs)
            else:
                return {"error": f"不支持的API类型: {api_type} | Unsupported API type: {api_type}"}
                
        except Exception as e:
            self.logger.error(f"文本生成失败: {str(e)} | Text generation failed: {str(e)}")
            return {"error": str(e)}
    
    def _openai_chat_analyze(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """使用OpenAI Chat API生成文本 | Generate text using OpenAI Chat API"""
        if not self.services["openai"]["chat"]:
            return {"error": "OpenAI Chat API未配置 | OpenAI Chat API not configured"}
        
        try:
            import requests
            
            config = self.services["openai"]["chat"]
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": kwargs.get("model", config["model"]),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            response = requests.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=kwargs.get("timeout", 30)
            )
            
            if response.status_code != 200:
                return {"error": f"OpenAI API错误: {response.status_code} | OpenAI API error: {response.status_code}"}
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            
            return {
                "success": True,
                "text": text,
                "usage": result.get("usage", {}),
                "source": "openai_chat"
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI Chat API调用失败: {str(e)} | OpenAI Chat API call failed: {str(e)}")
            return {"error": str(e)}
    
    def _anthropic_chat_analyze(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """使用Anthropic Chat API生成文本 | Generate text using Anthropic Chat API"""
        if not self.services["anthropic"]["chat"]:
            return {"error": "Anthropic Chat API未配置 | Anthropic Chat API not configured"}
        
        try:
            import requests
            
            config = self.services["anthropic"]["chat"]
            headers = {
                "x-api-key": config["api_key"],
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": kwargs.get("model", config["model"]),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            response = requests.post(
                f"{config['base_url']}/messages",
                headers=headers,
                json=data,
                timeout=kwargs.get("timeout", 30)
            )
            
            if response.status_code != 200:
                return {"error": f"Anthropic API错误: {response.status_code} | Anthropic API error: {response.status_code}"}
            
            result = response.json()
            text = result["content"][0]["text"]
            
            return {
                "success": True,
                "text": text,
                "usage": result.get("usage", {}),
                "source": "anthropic_chat"
            }
            
        except Exception as e:
            self.logger.error(f"Anthropic Chat API调用失败: {str(e)} | Anthropic Chat API call failed: {str(e)}")
            return {"error": str(e)}
    
    def _google_ai_chat_analyze(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """使用Google AI API生成文本 | Generate text using Google AI API"""
        if not self.services["google"]["ai"]:
            return {"error": "Google AI API未配置 | Google AI API not configured"}
        
        try:
            import requests
            
            config = self.services["google"]["ai"]
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "maxOutputTokens": kwargs.get("max_tokens", 1000)
                }
            }
            
            response = requests.post(
                f"{config['base_url']}/models/{config['model']}:generateContent?key={config['api_key']}",
                headers=headers,
                json=data,
                timeout=kwargs.get("timeout", 30)
            )
            
            if response.status_code != 200:
                return {"error": f"Google AI API错误: {response.status_code} | Google AI API error: {response.status_code}"}
            
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            
            return {
                "success": True,
                "text": text,
                "usage": result.get("usageMetadata", {}),
                "source": "google_ai"
            }
            
        except Exception as e:
            self.logger.error(f"Google AI API调用失败: {str(e)} | Google AI API call failed: {str(e)}")
            return {"error": str(e)}
    
    def _huggingface_inference_analyze(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """使用HuggingFace Inference API生成文本 | Generate text using HuggingFace Inference API"""
        if not self.services["huggingface"]["inference"]:
            return {"error": "HuggingFace Inference API未配置 | HuggingFace Inference API not configured"}
        
        try:
            import requests
            
            config = self.services["huggingface"]["inference"]
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            model = kwargs.get("model", "microsoft/DialoGPT-large")
            data = {
                "inputs": prompt,
                "parameters": {
                    "max_length": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7),
                    "do_sample": True
                }
            }
            
            response = requests.post(
                f"{config['base_url']}/models/{model}",
                headers=headers,
                json=data,
                timeout=kwargs.get("timeout", 30)
            )
            
            if response.status_code != 200:
                return {"error": f"HuggingFace API错误: {response.status_code} | HuggingFace API error: {response.status_code}"}
            
            result = response.json()
            text = result[0]["generated_text"] if isinstance(result, list) else result.get("generated_text", "")
            
            return {
                "success": True,
                "text": text,
                "source": "huggingface_inference"
            }
            
        except Exception as e:
            self.logger.error(f"HuggingFace Inference API调用失败: {str(e)} | HuggingFace Inference API call failed: {str(e)}")
            return {"error": str(e)}
    
    def _cohere_chat_analyze(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """使用Cohere Chat API生成文本 | Generate text using Cohere Chat API"""
        if not self.services["cohere"]["chat"]:
            return {"error": "Cohere Chat API未配置 | Cohere Chat API not configured"}
        
        try:
            import requests
            
            config = self.services["cohere"]["chat"]
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": kwargs.get("model", config["model"]),
                "message": prompt,
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            response = requests.post(
                f"{config['base_url']}/chat",
                headers=headers,
                json=data,
                timeout=kwargs.get("timeout", 30)
            )
            
            if response.status_code != 200:
                return {"error": f"Cohere API错误: {response.status_code} | Cohere API error: {response.status_code}"}
            
            result = response.json()
            text = result["text"]
            
            return {
                "success": True,
                "text": text,
                "source": "cohere_chat"
            }
            
        except Exception as e:
            self.logger.error(f"Cohere Chat API调用失败: {str(e)} | Cohere Chat API call failed: {str(e)}")
            return {"error": str(e)}
    
    def _mistral_chat_analyze(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """使用Mistral Chat API生成文本 | Generate text using Mistral Chat API"""
        if not self.services["mistral"]["chat"]:
            return {"error": "Mistral Chat API未配置 | Mistral Chat API not configured"}
        
        try:
            import requests
            
            config = self.services["mistral"]["chat"]
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": kwargs.get("model", config["model"]),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            response = requests.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=kwargs.get("timeout", 30)
            )
            
            if response.status_code != 200:
                return {"error": f"Mistral API错误: {response.status_code} | Mistral API error: {response.status_code}"}
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            
            return {
                "success": True,
                "text": text,
                "usage": result.get("usage", {}),
                "source": "mistral_chat"
            }
            
        except Exception as e:
            self.logger.error(f"Mistral Chat API调用失败: {str(e)} | Mistral Chat API call failed: {str(e)}")
            return {"error": str(e)}
    
    def _deepseek_chat_analyze(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """使用DeepSeek Chat API生成文本 | Generate text using DeepSeek Chat API"""
        if not self.services["deepseek"]["chat"]:
            return {"error": "DeepSeek Chat API未配置 | DeepSeek Chat API not configured"}
        
        try:
            import requests
            
            config = self.services["deepseek"]["chat"]
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": kwargs.get("model", config["model"]),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            response = requests.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=kwargs.get("timeout", 30)
            )
            
            if response.status_code != 200:
                return {"error": f"DeepSeek API错误: {response.status_code} | DeepSeek API error: {response.status_code}"}
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            
            return {
                "success": True,
                "text": text,
                "usage": result.get("usage", {}),
                "source": "deepseek_chat"
            }
            
        except Exception as e:
            self.logger.error(f"DeepSeek Chat API调用失败: {str(e)} | DeepSeek Chat API call failed: {str(e)}")
            return {"error": str(e)}

    def _siliconflow_chat_analyze(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """使用SiliconFlow Chat API生成文本 | Generate text using SiliconFlow Chat API"""
        if not self.services["siliconflow"]["chat"]:
            return {"error": "SiliconFlow Chat API未配置 | SiliconFlow Chat API not configured"}
        
        try:
            import requests
            
            config = self.services["siliconflow"]["chat"]
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": kwargs.get("model", config["model"]),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            response = requests.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=kwargs.get("timeout", 30)
            )
            
            if response.status_code != 200:
                return {"error": f"SiliconFlow API错误: {response.status_code} | SiliconFlow API error: {response.status_code}"}
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            
            return {
                "success": True,
                "text": text,
                "usage": result.get("usage", {}),
                "source": "siliconflow_chat"
            }
            
        except Exception as e:
            self.logger.error(f"SiliconFlow Chat API调用失败: {str(e)} | SiliconFlow Chat API call failed: {str(e)}")
            return {"error": str(e)}

    def _zhipu_chat_analyze(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """使用Zhipu AI Chat API生成文本 | Generate text using Zhipu AI Chat API"""
        if not self.services["zhipu"]["chat"]:
            return {"error": "Zhipu AI Chat API未配置 | Zhipu AI Chat API not configured"}
        
        try:
            import requests
            
            config = self.services["zhipu"]["chat"]
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": kwargs.get("model", config["model"]),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            response = requests.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=kwargs.get("timeout", 30)
            )
            
            if response.status_code != 200:
                return {"error": f"Zhipu AI API错误: {response.status_code} | Zhipu AI API error: {response.status_code}"}
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            
            return {
                "success": True,
                "text": text,
                "usage": result.get("usage", {}),
                "source": "zhipu_chat"
            }
            
        except Exception as e:
            self.logger.error(f"Zhipu AI Chat API调用失败: {str(e)} | Zhipu AI Chat API call failed: {str(e)}")
            return {"error": str(e)}

    def _baidu_chat_analyze(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """使用Baidu ERNIE Chat API生成文本 | Generate text using Baidu ERNIE Chat API"""
        if not self.services["baidu"]["chat"]:
            return {"error": "Baidu ERNIE Chat API未配置 | Baidu ERNIE Chat API not configured"}
        
        try:
            import requests
            
            config = self.services["baidu"]["chat"]
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": kwargs.get("model", config["model"]),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            response = requests.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=kwargs.get("timeout", 30)
            )
            
            if response.status_code != 200:
                return {"error": f"Baidu ERNIE API错误: {response.status_code} | Baidu ERNIE API error: {response.status_code}"}
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            
            return {
                "success": True,
                "text": text,
                "usage": result.get("usage", {}),
                "source": "baidu_chat"
            }
            
        except Exception as e:
            self.logger.error(f"Baidu ERNIE Chat API调用失败: {str(e)} | Baidu ERNIE Chat API call failed: {str(e)}")
            return {"error": str(e)}

    def _alibaba_chat_analyze(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """使用Alibaba Qwen Chat API生成文本 | Generate text using Alibaba Qwen Chat API"""
        if not self.services["alibaba"]["chat"]:
            return {"error": "Alibaba Qwen Chat API未配置 | Alibaba Qwen Chat API not configured"}
        
        try:
            import requests
            
            config = self.services["alibaba"]["chat"]
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": kwargs.get("model", config["model"]),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            response = requests.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=kwargs.get("timeout", 30)
            )
            
            if response.status_code != 200:
                return {"error": f"Alibaba Qwen API错误: {response.status_code} | Alibaba Qwen API error: {response.status_code}"}
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            
            return {
                "success": True,
                "text": text,
                "usage": result.get("usage", {}),
                "source": "alibaba_chat"
            }
            
        except Exception as e:
            self.logger.error(f"Alibaba Qwen Chat API调用失败: {str(e)} | Alibaba Qwen Chat API call failed: {str(e)}")
            return {"error": str(e)}

    def _moonshot_chat_analyze(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """使用Moonshot Chat API生成文本 | Generate text using Moonshot Chat API"""
        if not self.services["moonshot"]["chat"]:
            return {"error": "Moonshot Chat API未配置 | Moonshot Chat API not configured"}
        
        try:
            import requests
            
            config = self.services["moonshot"]["chat"]
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": kwargs.get("model", config["model"]),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            response = requests.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=kwargs.get("timeout", 30)
            )
            
            if response.status_code != 200:
                return {"error": f"Moonshot API错误: {response.status_code} | Moonshot API error: {response.status_code}"}
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            
            return {
                "success": True,
                "text": text,
                "usage": result.get("usage", {}),
                "source": "moonshot_chat"
            }
            
        except Exception as e:
            self.logger.error(f"Moonshot Chat API调用失败: {str(e)} | Moonshot Chat API call failed: {str(e)}")
            return {"error": str(e)}

    def _yi_chat_analyze(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """使用Yi Chat API生成文本 | Generate text using Yi Chat API"""
        if not self.services["yi"]["chat"]:
            return {"error": "Yi Chat API未配置 | Yi Chat API not configured"}
        
        try:
            import requests
            
            config = self.services["yi"]["chat"]
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": kwargs.get("model", config["model"]),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            response = requests.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=kwargs.get("timeout", 30)
            )
            
            if response.status_code != 200:
                return {"error": f"Yi API错误: {response.status_code} | Yi API error: {response.status_code}"}
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            
            return {
                "success": True,
                "text": text,
                "usage": result.get("usage", {}),
                "source": "yi_chat"
            }
            
        except Exception as e:
            self.logger.error(f"Yi Chat API调用失败: {str(e)} | Yi Chat API call failed: {str(e)}")
            return {"error": str(e)}

    def _tencent_chat_analyze(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """使用Tencent Hunyuan Chat API生成文本 | Generate text using Tencent Hunyuan Chat API"""
        if not self.services["tencent"]["chat"]:
            return {"error": "Tencent Hunyuan Chat API未配置 | Tencent Hunyuan Chat API not configured"}
        
        try:
            import requests
            
            config = self.services["tencent"]["chat"]
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": kwargs.get("model", config["model"]),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            response = requests.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=kwargs.get("timeout", 30)
            )
            
            if response.status_code != 200:
                return {"error": f"Tencent Hunyuan API错误: {response.status_code} | Tencent Hunyuan API error: {response.status_code}"}
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            
            return {
                "success": True,
                "text": text,
                "usage": result.get("usage", {}),
                "source": "tencent_chat"
            }
            
        except Exception as e:
            self.logger.error(f"Tencent Hunyuan Chat API调用失败: {str(e)} | Tencent Hunyuan Chat API call failed: {str(e)}")
            return {"error": str(e)}

    def _ollama_chat_analyze(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """使用Ollama API生成文本 | Generate text using Ollama API"""
        if not self.services["ollama"]["inference"]:
            return {"error": "Ollama API未配置 | Ollama API not configured"}
        
        try:
            import requests
            
            config = self.services["ollama"]["inference"]
            base_url = config['base_url']
            # 确保base_url以/v1结尾
            if not base_url.endswith('/v1'):
                base_url = base_url.rstrip('/') + '/v1'
            
            # Ollama不需要API密钥，但如果有则加上
            headers = {
                "Content-Type": "application/json"
            }
            if config.get('api_key'):
                headers['Authorization'] = f"Bearer {config['api_key']}"
            
            data = {
                "model": kwargs.get("model", config["model"]),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=kwargs.get("timeout", 30)
            )
            
            if response.status_code != 200:
                return {"error": f"Ollama API错误: {response.status_code} | Ollama API error: {response.status_code}"}
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            
            return {
                "success": True,
                "text": text,
                "usage": result.get("usage", {}),
                "source": "ollama_chat"
            }
            
        except Exception as e:
            self.logger.error(f"Ollama API调用失败: {str(e)} | Ollama API call failed: {str(e)}")
            return {"error": str(e)}

    def get_service_status(self) -> Dict[str, Any]:
        """获取API服务状态 | Get API service status"""
        status = {}
        
        for provider, services in self.services.items():
            status[provider] = {
                "configured": services["configured"],
                "services_available": []
            }
            
            for service_name, service in services.items():
                if service_name not in ["configured"] and service is not None:
                    status[provider]["services_available"].append(service_name)
        
        return status
        
    def test_connection(self, provider: str, service_type: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """测试外部API连接 | Test external API connection
        
        Args:
            provider: API提供商 (openai/anthropic/google等) | API provider
            service_type: 服务类型 (chat/vision/video等) | Service type
            config: 可选的API配置，如果提供则使用该配置进行测试 | Optional API config
        
        Returns:
            连接测试结果 | Connection test result
        """
        try:
            if provider not in self.supported_providers:
                return {"success": False, "error": f"不支持的API提供商: {provider} | Unsupported API provider: {provider}"}
                
            # 如果未提供配置，使用已加载的配置
            if config is None:
                if not self.services[provider][service_type]:
                    return {"success": False, "error": f"{provider} {service_type} API未配置 | {provider} {service_type} API not configured"}
                
                # 使用现有配置进行测试
                test_config = self.services[provider][service_type]
            else:
                # 使用提供的配置进行测试
                test_config = config
                
            self.logger.info(f"正在测试{provider} {service_type} API连接 | Testing {provider} {service_type} API connection")
            
            # 针对不同的API提供商和服务类型进行特定的连接测试
            if provider == "openai" and service_type == "chat":
                return self._test_openai_connection(test_config)
            elif provider == "anthropic" and service_type == "chat":
                return self._test_anthropic_connection(test_config)
            elif provider == "google_ai" or (provider == "google" and service_type == "ai"):
                return self._test_google_ai_connection(test_config)
            elif provider == "custom":
                # 自定义API连接测试
                return self._test_generic_api_connection(test_config)
            else:
                # 对于其他API类型，使用通用的连接测试方法
                return self._test_generic_api_connection(test_config)
                
        except Exception as e:
            error_message = str(e)
            self.logger.error(f"API连接测试失败: {error_message} | API connection test failed: {error_message}")
            return {
                "success": False, 
                "error": error_message,
                "error_type": type(e).__name__, 
                "timestamp": time.time()
            }
            
    def _test_openai_connection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """测试OpenAI API连接 | Test OpenAI API connection"""
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            # 构建一个简单的测试请求
            data = {
                "model": config.get("model", "gpt-4"),
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5
            }
            
            response = requests.post(
                f"{config.get('base_url', 'https://api.openai.com/v1')}/chat/completions",
                headers=headers,
                json=data,
                timeout=5
            )
            
            if response.status_code == 200:
                self.logger.info("OpenAI API连接测试成功 | OpenAI API connection test successful")
                return {
                    "success": True,
                    "message": "OpenAI API连接测试成功 | OpenAI API connection test successful",
                    "status_code": response.status_code,
                    "timestamp": time.time()
                }
            else:
                error_msg = f"OpenAI API连接失败: HTTP {response.status_code} | OpenAI API connection failed: HTTP {response.status_code}"
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "status_code": response.status_code,
                    "error_type": "HTTPError",
                    "timestamp": time.time()
                }
                
        except requests.exceptions.RequestException as e:
            error_msg = f"OpenAI API连接异常: {str(e)} | OpenAI API connection exception: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "error_type": type(e).__name__,
                "timestamp": time.time()
            }
            
    def _test_anthropic_connection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """测试Anthropic API连接 | Test Anthropic API connection"""
        try:
            import requests
            
            headers = {
                "x-api-key": config['api_key'],
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            # 构建一个简单的测试请求
            data = {
                "model": config.get("model", "claude-3-opus-20240229"),
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5
            }
            
            response = requests.post(
                f"{config.get('base_url', 'https://api.anthropic.com')}/messages",
                headers=headers,
                json=data,
                timeout=5
            )
            
            if response.status_code == 200:
                self.logger.info("Anthropic API连接测试成功 | Anthropic API connection test successful")
                return {
                    "success": True,
                    "message": "Anthropic API连接测试成功 | Anthropic API connection test successful",
                    "status_code": response.status_code,
                    "timestamp": time.time()
                }
            else:
                error_msg = f"Anthropic API连接失败: HTTP {response.status_code} | Anthropic API connection failed: HTTP {response.status_code}"
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "status_code": response.status_code,
                    "error_type": "HTTPError",
                    "timestamp": time.time()
                }
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Anthropic API连接异常: {str(e)} | Anthropic API connection exception: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "error_type": type(e).__name__,
                "timestamp": time.time()
            }
            
    def _test_google_ai_connection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """测试Google AI API连接 | Test Google AI API connection"""
        try:
            import requests
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # 构建一个简单的测试请求
            data = {
                "contents": [{
                    "parts": [{"text": "ping"}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 5
                }
            }
            
            model = config.get("model", "gemini-pro")
            response = requests.post(
                f"{config.get('base_url', 'https://generativelanguage.googleapis.com/v1beta')}/models/{model}:generateContent?key={config['api_key']}",
                headers=headers,
                json=data,
                timeout=5
            )
            
            if response.status_code == 200:
                self.logger.info("Google AI API连接测试成功 | Google AI API connection test successful")
                return {
                    "success": True,
                    "message": "Google AI API连接测试成功 | Google AI API connection test successful",
                    "status_code": response.status_code,
                    "timestamp": time.time()
                }
            else:
                error_msg = f"Google AI API连接失败: HTTP {response.status_code} | Google AI API connection failed: HTTP {response.status_code}"
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "status_code": response.status_code,
                    "error_type": "HTTPError",
                    "timestamp": time.time()
                }
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Google AI API连接异常: {str(e)} | Google AI API connection exception: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "error_type": type(e).__name__,
                "timestamp": time.time()
            }
            
    def _test_generic_api_connection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """测试通用API连接 | Test generic API connection
        
        支持测试自定义或未预定义的API连接，尝试多种常见的端点格式
        """
        try:
            import requests
            
            api_url = config.get('api_url')
            api_key = config.get('api_key')
            
            if not api_url:
                return {"success": False, "error": "缺少API URL | Missing API URL", "error_type": "ConfigError"}
                
            if not (api_url.startswith('http://') or api_url.startswith('https://')):
                return {"success": False, "error": f"无效的API URL格式: {api_url} | Invalid API URL format", "error_type": "ConfigError"}
                
            # 构建请求头
            headers = {}
            if api_key:
                # 尝试常见的API密钥格式
                if 'Authorization' not in headers:
                    headers['Authorization'] = f"Bearer {api_key}"
                    
            # 尝试多种常见的端点格式
            test_endpoints = []
            
            # 1. 直接使用提供的URL
            test_endpoints.append(api_url)
            
            # 2. 添加常见的测试端点后缀
            common_endpoints = ['/ping', '/health', '/v1/chat/completions', '/chat/completions']
            for endpoint in common_endpoints:
                if not api_url.endswith('/'):
                    test_endpoints.append(f"{api_url}{endpoint}")
                else:
                    test_endpoints.append(f"{api_url[:-1]}{endpoint}")
                    
            # 3. 确保唯一性
            test_endpoints = list(set(test_endpoints))
            
            # 构建测试数据
            test_data = {
                "model": config.get("model_name", config.get("model", "default")),
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5
            }
            
            # 尝试每个端点
            for url in test_endpoints:
                try:
                    # 先尝试GET请求（可能是健康检查端点）
                    response = requests.get(url, headers=headers, timeout=2)
                    if response.status_code == 200:
                        self.logger.info(f"通用API连接测试成功 (GET): {url} | Generic API connection test successful (GET)")
                        return {
                            "success": True,
                            "message": f"API连接测试成功 | API connection test successful",
                            "status_code": response.status_code,
                            "endpoint": url,
                            "method": "GET",
                            "timestamp": time.time()
                        }
                        
                    # 如果GET失败，尝试POST请求（可能是聊天完成端点）
                    response = requests.post(url, headers=headers, json=test_data, timeout=3)
                    if response.status_code in [200, 400]:  # 400可能表示请求格式错误但连接正常
                        self.logger.info(f"通用API连接测试成功 (POST): {url} | Generic API connection test successful (POST)")
                        return {
                            "success": True,
                            "message": f"API连接测试成功 | API connection test successful",
                            "status_code": response.status_code,
                            "endpoint": url,
                            "method": "POST",
                            "timestamp": time.time()
                        }
                        
                except requests.exceptions.RequestException:
                    # 这个端点失败，尝试下一个
                    continue
                    
            # 所有端点都失败
            self.logger.error("所有通用API端点测试失败 | All generic API endpoints test failed")
            return {
                "success": False,
                "error": "API连接测试失败，所有尝试的端点都无法访问 | API connection test failed, all attempted endpoints are inaccessible",
                "error_type": "ConnectionError",
                "attempted_endpoints": test_endpoints,
                "timestamp": time.time()
            }
            
        except Exception as e:
            error_message = str(e)
            self.logger.error(f"通用API连接测试异常: {error_message} | Generic API connection test exception: {error_message}")
            return {
                "success": False,
                "error": error_message,
                "error_type": type(e).__name__, 
                "timestamp": time.time()
            }
            
    def set_model_api_config(self, model_id: str, api_config: Dict[str, Any]) -> Dict[str, Any]:
        """设置特定模型的API配置 | Set API configuration for a specific model
        
        Args:
            model_id: 模型ID | Model ID
            api_config: API配置 | API configuration
        
        Returns:
            操作结果 | Operation result
        """
        try:
            with self.lock:
                # 保存模型的API配置
                self.api_configs[model_id] = api_config
                
                # 从系统设置管理器获取模型设置
                if hasattr(self, 'system_settings_manager'):
                    model_setting = self.system_settings_manager.get_model_setting(model_id)
                    if model_setting:
                        # 更新模型设置中的API配置
                        model_setting['external_api_config'] = api_config
                        model_setting['use_external_api'] = True
                        self.system_settings_manager.update_model_setting(model_id, model_setting)
                        
            self.logger.info(f"模型 {model_id} 的API配置已设置 | API configuration for model {model_id} has been set")
            return {
                "success": True,
                "model_id": model_id,
                "message": "API配置已成功设置 | API configuration has been set successfully",
                "timestamp": time.time()
            }
            
        except Exception as e:
            error_message = str(e)
            self.logger.error(f"设置模型 {model_id} 的API配置失败: {error_message} | Failed to set API configuration for model {model_id}: {error_message}")
            return {
                "success": False,
                "error": error_message,
                "error_type": type(e).__name__, 
                "timestamp": time.time()
            }
            
    def get_model_api_config(self, model_id: str) -> Dict[str, Any]:
        """获取特定模型的API配置 | Get API configuration for a specific model
        
        Args:
            model_id: 模型ID | Model ID
        
        Returns:
            API配置 | API configuration
        """
        with self.lock:
            # 首先从缓存中获取
            if model_id in self.api_configs:
                return self.api_configs[model_id]
                
            # 如果缓存中没有，尝试从系统设置管理器获取
            if hasattr(self, 'system_settings_manager'):
                model_setting = self.system_settings_manager.get_model_setting(model_id)
                if model_setting and 'external_api_config' in model_setting:
                    return model_setting['external_api_config']
                    
            # 如果都没有，返回空配置
            return {}
            
    def get_model_api_status(self, model_id: str) -> Dict[str, Any]:
        """获取特定模型的API连接状态 | Get API connection status for a specific model
        
        Args:
            model_id: 模型ID | Model ID
        
        Returns:
            API连接状态信息 | API connection status information
        """
        try:
            # 获取模型配置
            model_config = self.get_model_api_config(model_id)
            
            # 检查是否配置了外部API
            if not model_config:
                return {
                    "model_id": model_id,
                    "use_external_api": False,
                    "configured": False,
                    "status": "未配置 | Not configured",
                    "timestamp": time.time()
                }
                
            # 检查模型是否使用外部API（从系统设置中获取）
            use_external_api = False
            if hasattr(self, 'system_settings_manager'):
                model_setting = self.system_settings_manager.get_model_setting(model_id)
                if model_setting:
                    use_external_api = model_setting.get('use_external_api', False)
                    
            # 获取API提供商和服务类型
            provider = model_config.get('source', 'custom')
            service_type = model_config.get('service_type', 'chat')
            
            # 如果未启用外部API，返回配置但未启用的状态
            if not use_external_api:
                return {
                    "model_id": model_id,
                    "use_external_api": False,
                    "configured": True,
                    "provider": provider,
                    "service_type": service_type,
                    "status": "已配置但未启用 | Configured but not enabled",
                    "config": model_config,
                    "timestamp": time.time()
                }
                
            # 测试连接
            test_result = self.test_connection(provider, service_type, model_config)
            
            # 合并测试结果和状态信息
            status_info = {
                "model_id": model_id,
                "use_external_api": True,
                "configured": True,
                "provider": provider,
                "service_type": service_type,
                "config": model_config,
                "connection_test": test_result,
                "timestamp": time.time()
            }
            
            if test_result.get('success', False):
                status_info["status"] = "已连接 | Connected"
            else:
                status_info["status"] = "连接失败 | Connection failed"
                
            return status_info
            
        except Exception as e:
            error_message = str(e)
            self.logger.error(f"获取模型 {model_id} 的API状态失败: {error_message} | Failed to get API status for model {model_id}: {error_message}")
            return {
                "model_id": model_id,
                "success": False,
                "error": error_message,
                "error_type": type(e).__name__, 
                "timestamp": time.time()
            }
    
    def save_configuration(self, filepath: str = "config/external_apis_config.json"):
        """保存API配置到文件 | Save API configuration to file"""
        try:
            import os
            # 确保目录存在 | Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 构建完整的配置信息 | Build complete configuration information
            config_to_save = {
                "external_apis": {
                    "openai": {
                        "chat": {
                            "api_key": "[REDACTED]" if self.services["openai"]["chat"] else None,
                            "base_url": self.services["openai"]["chat"]["base_url"] if self.services["openai"]["chat"] else None,
                            "model": self.services["openai"]["chat"]["model"] if self.services["openai"]["chat"] else None,
                            "configured": self.services["openai"]["chat"] is not None
                        },
                        "vision": {
                            "api_key": "[REDACTED]" if self.services["openai"]["vision"] else None,
                            "base_url": self.services["openai"]["vision"]["base_url"] if self.services["openai"]["vision"] else None,
                            "model": self.services["openai"]["vision"]["model"] if self.services["openai"]["vision"] else None,
                            "configured": self.services["openai"]["vision"] is not None
                        }
                    },
                    "anthropic": {
                        "chat": {
                            "api_key": "[REDACTED]" if self.services["anthropic"]["chat"] else None,
                            "base_url": self.services["anthropic"]["chat"]["base_url"] if self.services["anthropic"]["chat"] else None,
                            "model": self.services["anthropic"]["chat"]["model"] if self.services["anthropic"]["chat"] else None,
                            "configured": self.services["anthropic"]["chat"] is not None
                        }
                    },
                    "google_ai": {
                        "ai": {
                            "api_key": "[REDACTED]" if self.services["google"]["ai"] else None,
                            "base_url": self.services["google"]["ai"]["base_url"] if self.services["google"]["ai"] else None,
                            "model": self.services["google"]["ai"]["model"] if self.services["google"]["ai"] else None,
                            "configured": self.services["google"]["ai"] is not None
                        }
                    },
                    "google": {
                        "vision": {
                            "api_key": "[REDACTED]" if self.services["google"]["vision"] else None,
                            "project_id": "[REDACTED]" if self.services["google"]["vision"] else None,
                            "configured": self.services["google"]["vision"] is not None
                        },
                        "video": {
                            "api_key": "[REDACTED]" if self.services["google"]["video"] else None,
                            "project_id": "[REDACTED]" if self.services["google"]["video"] else None,
                            "configured": self.services["google"]["video"] is not None
                        }
                    },
                    "aws": {
                        "rekognition": {
                            "access_key": "[REDACTED]" if self.services["aws"]["rekognition"] else None,
                            "region": "us-east-1",
                            "configured": self.services["aws"]["rekognition"] is not None
                        },
                        "rekognition_video": {
                            "access_key": "[REDACTED]" if self.services["aws"]["rekognition_video"] else None,
                            "region": "us-east-1",
                            "configured": self.services["aws"]["rekognition_video"] is not None
                        }
                    },
                    "azure": {
                        "vision": {
                            "endpoint": self.services["azure"]["vision"]["endpoint"] if self.services["azure"]["vision"] else None,
                            "subscription_key": "[REDACTED]" if self.services["azure"]["vision"] else None,
                            "configured": self.services["azure"]["vision"] is not None
                        },
                        "video": {
                            "endpoint": self.services["azure"]["video"]["endpoint"] if self.services["azure"]["video"] else None,
                            "subscription_key": "[REDACTED]" if self.services["azure"]["video"] else None,
                            "configured": self.services["azure"]["video"] is not None
                        }
                    },
                    "huggingface": {
                        "inference": {
                            "api_key": "[REDACTED]" if self.services["huggingface"]["inference"] else None,
                            "base_url": self.services["huggingface"]["inference"]["base_url"] if self.services["huggingface"]["inference"] else None,
                            "configured": self.services["huggingface"]["inference"] is not None
                        }
                    },
                    "cohere": {
                        "chat": {
                            "api_key": "[REDACTED]" if self.services["cohere"]["chat"] else None,
                            "base_url": self.services["cohere"]["chat"]["base_url"] if self.services["cohere"]["chat"] else None,
                            "model": self.services["cohere"]["chat"]["model"] if self.services["cohere"]["chat"] else None,
                            "configured": self.services["cohere"]["chat"] is not None
                        }
                    },
                    "mistral": {
                        "chat": {
                            "api_key": "[REDACTED]" if self.services["mistral"]["chat"] else None,
                            "base_url": self.services["mistral"]["chat"]["base_url"] if self.services["mistral"]["chat"] else None,
                            "model": self.services["mistral"]["chat"]["model"] if self.services["mistral"]["chat"] else None,
                            "configured": self.services["mistral"]["chat"] is not None
                        }
                    }
                },
                "last_updated": datetime.now().isoformat(),
                "service_status": self.get_service_status()
            }
            
            # 保存配置（敏感信息已脱敏） | Save configuration (sensitive information redacted)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"API配置已保存: {filepath} | API configuration saved: {filepath}")
            return {"success": True, "filepath": filepath}
            
        except Exception as e:
            self.logger.error(f"保存API配置失败: {str(e)} | Failed to save API configuration: {str(e)}")
            return {"error": str(e)}
    
    def load_configuration(self, filepath: str = "config/external_apis_config.json"):
        """从文件加载API配置 | Load API configuration from file"""
        try:
            if not os.path.exists(filepath):
                return {"error": f"配置文件不存在: {filepath} | Configuration file does not exist: {filepath}"}
            
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 重新初始化服务 | Reinitialize services
            self.config = config
            self._initialize_all_services()
            
            self.logger.info(f"API配置已加载: {filepath} | API configuration loaded: {filepath}")
            return {"success": True, "config": config}
            
        except Exception as e:
            self.logger.error(f"加载API配置失败: {str(e)} | Failed to load API configuration: {str(e)}")
            return {"error": str(e)}


# 全局API服务实例 | Global API service instance
_global_api_service = None

def get_global_api_service(config: Dict[str, Any] = None) -> ExternalAPIService:
    """获取全局API服务实例 | Get global API service instance"""
    global _global_api_service
    if _global_api_service is None:
        _global_api_service = ExternalAPIService(config)
    return _global_api_service

def set_global_api_service(service: ExternalAPIService):
    """设置全局API服务实例 | Set global API service instance"""
    global _global_api_service
    _global_api_service = service
