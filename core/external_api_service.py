"""
统一外部API服务类 - Unified External API Service
提供统一的主流AI API服务接口，支持OpenAI、Anthropic、Google AI、AWS、Azure等
Provides unified mainstream AI API service interface, supporting OpenAI, Anthropic, Google AI, AWS, Azure, etc.
"""

import logging
import json
import os
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime


class ExternalAPIService:
    """统一外部API服务
    Unified External API Service
    
    功能：统一管理所有主流AI API的配置、认证和调用
    Function: Unified management of all mainstream AI APIs configuration, authentication and calls
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化外部API服务 | Initialize external API service"""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
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
            }
        }
        
        # API配置缓存 | API configuration cache
        self.api_configs = {}
        
        # 初始化所有API服务 | Initialize all API services
        self._initialize_all_services()
        
        self.logger.info("统一外部API服务初始化完成 | Unified external API service initialized")
    
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
                    import boto3
                    self.services["aws"]["rekognition"] = boto3.client(
                        'rekognition',
                        aws_access_key_id=rekognition_config["access_key"],
                        aws_secret_access_key=rekognition_config["secret_key"],
                        region_name=rekognition_config.get("region", "us-east-1")
                    )
                    self.services["aws"]["configured"] = True
                    self.logger.info("AWS Rekognition配置完成 | AWS Rekognition configured")
                except ImportError:
                    self.logger.warning("boto3未安装，无法使用AWS Rekognition | boto3 not installed, cannot use AWS Rekognition")
            
            # AWS Rekognition Video
            video_config = aws_config.get("rekognition_video", {})
            if video_config.get("access_key") and video_config.get("secret_key"):
                try:
                    import boto3
                    self.services["aws"]["rekognition_video"] = boto3.client(
                        'rekognition',
                        aws_access_key_id=video_config["access_key"],
                        aws_secret_access_key=video_config["secret_key"],
                        region_name=video_config.get("region", "us-east-1")
                    )
                    self.services["aws"]["configured"] = True
                    self.logger.info("AWS Rekognition Video配置完成 | AWS Rekognition Video configured")
                except ImportError:
                    self.logger.warning("boto3未安装，无法使用AWS Rekognition Video | boto3 not installed, cannot use AWS Rekognition Video")
                    
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
            api_type: API类型 (openai/anthropic/google_ai/huggingface/cohere/mistral) | API type
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
