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
数据集管理器 - 处理训练数据的上传、验证和管理
Dataset Manager - Handle training data upload, validation and management
"""

import os
import json
import logging
import shutil
import tempfile
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from .error_handling import error_handler

class DatasetManager:
    """数据集管理器类
    Dataset Manager Class
    
    功能：管理训练数据集的上传、验证、存储和检索
    Function: Manage training dataset upload, validation, storage and retrieval
    """
    
    def __init__(self, base_data_dir: str = "data/datasets"):
        """初始化数据集管理器
        Initialize dataset manager
        
        Args:
            base_data_dir: 数据集存储的基础目录 | Base directory for dataset storage
        """
        self.logger = logging.getLogger(__name__)
        self.base_data_dir = base_data_dir
        self.supported_formats = self._get_supported_formats()
        
        # 确保基础目录存在 | Ensure base directory exists
        os.makedirs(base_data_dir, exist_ok=True)
        
        self.logger.info("数据集管理器初始化完成 | Dataset manager initialized")
    
    def _get_supported_formats(self) -> Dict[str, List[str]]:
        """获取各模型支持的数据格式
        Get supported data formats for each model
        
        Returns:
            各模型支持的文件格式字典 | Dictionary of supported file formats for each model
        """
        return {
            "manager": ["json", "csv", "txt"],
            "language": ["json", "txt", "csv", "jsonl", "parquet"],
            "audio": ["wav", "mp3", "flac", "ogg", "json", "csv"],
            "vision_image": ["jpg", "jpeg", "png", "bmp", "tiff", "json", "csv"],
            "vision_video": ["mp4", "avi", "mov", "mkv", "json", "csv"],
            "spatial": ["json", "csv", "txt", "ply", "obj", "gltf"],
            "sensor": ["csv", "json", "txt", "bin"],
            "computer": ["json", "csv", "txt", "yaml", "xml"],
            "motion": ["json", "csv", "txt", "bin"],
            "knowledge": ["json", "csv", "txt", "pdf", "docx", "html"],
            "programming": ["json", "csv", "txt", "py", "js", "java", "cpp"],
            "planning": ["json", "csv", "txt", "yaml"],
            "finance": ["json", "csv", "xlsx", "txt"],
            "medical": ["json", "csv", "txt", "dicom", "nii"],
            "prediction": ["json", "csv", "txt", "parquet"],
            "emotion": ["json", "csv", "txt"],
            "stereo_vision": ["json", "csv", "txt", "png", "jpg", "bin"]
        }
    
    def get_model_supported_formats(self, model_id: str) -> List[str]:
        """获取指定模型支持的数据格式
        Get supported data formats for specified model
        
        Args:
            model_id: 模型ID | Model ID
            
        Returns:
            支持的格式列表 | List of supported formats
        """
        return self.supported_formats.get(model_id, ["json", "csv", "txt"])
    
    def get_all_supported_formats(self) -> Dict[str, List[str]]:
        """获取所有模型支持的数据格式
        Get all supported data formats for all models
        
        Returns:
            各模型支持格式的字典 | Dictionary of supported formats for all models
        """
        return self.supported_formats
    
    def validate_dataset(self, file_path: str, model_id: str) -> Dict[str, Any]:
        """验证数据集文件
        Validate dataset file
        
        Args:
            file_path: 文件路径 | File path
            model_id: 目标模型ID | Target model ID
            
        Returns:
            验证结果 | Validation result
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return {
                    "valid": False,
                    "error": f"文件不存在: {file_path}",
                    "details": {}
                }
            
            # 获取文件扩展名
            file_ext = Path(file_path).suffix.lower().lstrip('.')
            
            # 检查格式是否支持
            supported_formats = self.get_model_supported_formats(model_id)
            if file_ext not in supported_formats:
                return {
                    "valid": False,
                    "error": f"模型 {model_id} 不支持 {file_ext} 格式",
                    "details": {
                        "supported_formats": supported_formats,
                        "file_format": file_ext
                    }
                }
            
            # 根据模型类型进行特定验证
            validation_result = self._validate_by_model_type(file_path, file_ext, model_id)
            
            if not validation_result["valid"]:
                return validation_result
            
            # 获取文件统计信息
            file_stats = os.stat(file_path)
            
            return {
                "valid": True,
                "file_size": file_stats.st_size,
                "file_format": file_ext,
                "last_modified": datetime.fromtimestamp(file_stats.st_mtime),
                "details": validation_result.get("details", {})
            }
            
        except Exception as e:
            self.logger.error(f"数据集验证失败: {str(e)} | Dataset validation failed: {str(e)}")
            return {
                "valid": False,
                "error": f"验证过程中发生错误: {str(e)}",
                "details": {}
            }
    
    def _validate_by_model_type(self, file_path: str, file_ext: str, model_id: str) -> Dict[str, Any]:
        """根据模型类型进行特定验证
        Perform model-specific validation
        
        Args:
            file_path: 文件路径 | File path
            file_ext: 文件扩展名 | File extension
            model_id: 模型ID | Model ID
            
        Returns:
            验证结果 | Validation result
        """
        validation_methods = {
            "language": self._validate_language_data,
            "audio": self._validate_audio_data,
            "vision_image": self._validate_vision_image_data,
            "vision_video": self._validate_vision_video_data,
            "knowledge": self._validate_knowledge_data,
            # 其他模型的验证方法...
        }
        
        # 使用通用验证作为默认
        validator = validation_methods.get(model_id, self._validate_general_data)
        return validator(file_path, file_ext)
    
    def _validate_general_data(self, file_path: str, file_ext: str) -> Dict[str, Any]:
        """通用数据验证
        General data validation
        
        Args:
            file_path: 文件路径 | File path
            file_ext: 文件扩展名 | File extension
            
        Returns:
            验证结果 | Validation result
        """
        try:
            # 基本文件检查
            if os.path.getsize(file_path) == 0:
                return {
                    "valid": False,
                    "error": "文件为空",
                    "details": {}
                }
            
            # 根据文件类型进行基本验证
            if file_ext == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)  # 尝试解析JSON
                return {"valid": True, "details": {"format": "json"}}
                
            elif file_ext == "csv":
                # 简单的CSV检查：至少有一行数据
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) < 2:  # 至少标题行+一行数据
                        return {
                            "valid": False,
                            "error": "CSV文件需要至少包含标题行和一行数据",
                            "details": {"line_count": len(lines)}
                        }
                return {"valid": True, "details": {"line_count": len(lines)}}
                
            elif file_ext == "txt":
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        return {
                            "valid": False,
                            "error": "文本文件为空",
                            "details": {}
                        }
                return {"valid": True, "details": {"content_length": len(content)}}
            
            # 其他格式的简单验证...
            return {"valid": True, "details": {"format": file_ext}}
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"文件验证失败: {str(e)}",
                "details": {"format": file_ext}
            }
    
    def _validate_language_data(self, file_path: str, file_ext: str) -> Dict[str, Any]:
        """语言模型数据验证
        Language model data validation
        
        Args:
            file_path: 文件路径 | File path
            file_ext: 文件扩展名 | File extension
            
        Returns:
            验证结果 | Validation result
        """
        result = self._validate_general_data(file_path, file_ext)
        if not result["valid"]:
            return result
        
        try:
            if file_ext == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查JSON结构是否适合语言模型训练
                if isinstance(data, list):
                    # 检查列表中的每个项目是否有文本内容
                    for i, item in enumerate(data):
                        if not isinstance(item, dict) or "text" not in item:
                            return {
                                "valid": False,
                                "error": f"JSON列表中的项目 {i} 缺少 'text' 字段",
                                "details": {"item_index": i}
                            }
                elif isinstance(data, dict):
                    # 检查是否有训练数据相关的字段
                    if "text" not in data and "conversations" not in data:
                        return {
                            "valid": False,
                            "error": "JSON对象缺少 'text' 或 'conversations' 字段",
                            "details": {"keys": list(data.keys())}
                        }
                
                return {
                    "valid": True,
                    "details": {
                        "format": "json",
                        "data_type": "language_training",
                        "item_count": len(data) if isinstance(data, list) else 1
                    }
                }
            
            return result
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"语言数据验证失败: {str(e)}",
                "details": {"format": file_ext}
            }
    
    def _validate_audio_data(self, file_path: str, file_ext: str) -> Dict[str, Any]:
        """音频数据验证
        Audio data validation
        
        Args:
            file_path: 文件路径 | File path
            file_ext: 文件扩展名 | File extension
            
        Returns:
            验证结果 | Validation result
        """
        result = self._validate_general_data(file_path, file_ext)
        if not result["valid"]:
            return result
        
        # 音频文件的特定验证
        if file_ext in ["wav", "mp3", "flac", "ogg"]:
            # 这里可以添加音频文件的具体验证逻辑
            # 例如检查文件头、采样率等
            try:
                file_size = os.path.getsize(file_path)
                if file_size < 1024:  # 最小文件大小检查
                    return {
                        "valid": False,
                        "error": "音频文件太小，可能损坏",
                        "details": {"file_size": file_size}
                    }
                
                return {
                    "valid": True,
                    "details": {
                        "format": file_ext,
                        "file_size": file_size,
                        "data_type": "audio"
                    }
                }
                
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"音频文件验证失败: {str(e)}",
                    "details": {"format": file_ext}
                }
        
        return result
    
    def _validate_vision_image_data(self, file_path: str, file_ext: str) -> Dict[str, Any]:
        """视觉图像数据验证
        Vision image data validation
        
        Args:
            file_path: 文件路径 | File path
            file_ext: 文件扩展名 | File extension
            
        Returns:
            验证结果 | Validation result
        """
        result = self._validate_general_data(file_path, file_ext)
        if not result["valid"]:
            return result
        
        if file_ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
            try:
                file_size = os.path.getsize(file_path)
                # 图像文件的最小大小检查
                if file_size < 100:  # 100字节
                    return {
                        "valid": False,
                        "error": "图像文件太小，可能损坏",
                        "details": {"file_size": file_size}
                    }
                
                return {
                    "valid": True,
                    "details": {
                        "format": file_ext,
                        "file_size": file_size,
                        "data_type": "image"
                    }
                }
                
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"图像文件验证失败: {str(e)}",
                    "details": {"format": file_ext}
                }
        
        return result
    
    def _validate_vision_video_data(self, file_path: str, file_ext: str) -> Dict[str, Any]:
        """视觉视频数据验证
        Vision video data validation
        
        Args:
            file_path: 文件路径 | File path
            file_ext: 文件扩展名 | File extension
            
        Returns:
            验证结果 | Validation result
        """
        result = self._validate_general_data(file_path, file_ext)
        if not result["valid"]:
            return result
        
        if file_ext in ["mp4", "avi", "mov", "mkv"]:
            try:
                file_size = os.path.getsize(file_path)
                # 视频文件的最小大小检查
                if file_size < 1024:  # 1KB
                    return {
                        "valid": False,
                        "error": "视频文件太小，可能损坏",
                        "details": {"file_size": file_size}
                    }
                
                return {
                    "valid": True,
                    "details": {
                        "format": file_ext,
                        "file_size": file_size,
                        "data_type": "video"
                    }
                }
                
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"视频文件验证失败: {str(e)}",
                    "details": {"format": file_ext}
                }
        
        return result
    
    def _validate_knowledge_data(self, file_path: str, file_ext: str) -> Dict[str, Any]:
        """知识库数据验证
        Knowledge data validation
        
        Args:
            file_path: 文件路径 | File path
            file_ext: 文件扩展名 | File extension
            
        Returns:
            验证结果 | Validation result
        """
        result = self._validate_general_data(file_path, file_ext)
        if not result["valid"]:
            return result
        
        if file_ext in ["pdf", "docx"]:
            # 这里可以添加PDF/DOCX文件的特定验证
            try:
                file_size = os.path.getsize(file_path)
                return {
                    "valid": True,
                    "details": {
                        "format": file_ext,
                        "file_size": file_size,
                        "data_type": "document"
                    }
                }
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"文档文件验证失败: {str(e)}",
                    "details": {"format": file_ext}
                }
        
        return result
    
    def save_dataset(self, file_path: str, model_id: str, dataset_name: str = None) -> Dict[str, Any]:
        """保存验证通过的数据集
        Save validated dataset
        
        Args:
            file_path: 源文件路径 | Source file path
            model_id: 模型ID | Model ID
            dataset_name: 数据集名称 (可选) | Dataset name (optional)
            
        Returns:
            保存结果 | Save result
        """
        try:
            # 验证数据集
            validation_result = self.validate_dataset(file_path, model_id)
            if not validation_result["valid"]:
                return validation_result
            
            # 生成数据集ID和名称
            if not dataset_name:
                dataset_name = Path(file_path).stem
            dataset_id = f"{model_id}_{dataset_name}_{int(datetime.now().timestamp())}"
            
            # 创建模型特定的数据集目录
            model_data_dir = os.path.join(self.base_data_dir, model_id)
            os.makedirs(model_data_dir, exist_ok=True)
            
            # 目标文件路径
            target_path = os.path.join(model_data_dir, f"{dataset_id}{Path(file_path).suffix}")
            
            # 复制文件
            shutil.copy2(file_path, target_path)
            
            # 创建元数据文件
            # 准备可序列化的验证结果
            serializable_validation = validation_result.copy()
            # 将datetime对象转换为字符串
            if "last_modified" in serializable_validation:
                if hasattr(serializable_validation["last_modified"], 'isoformat'):
                    serializable_validation["last_modified"] = serializable_validation["last_modified"].isoformat()
            
            metadata = {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "dataset_name": dataset_name,
                "original_filename": Path(file_path).name,
                "file_format": validation_result["file_format"],
                "file_size": validation_result["file_size"],
                "upload_time": datetime.now().isoformat(),
                "validation_result": serializable_validation
            }
            
            metadata_path = os.path.join(model_data_dir, f"{dataset_id}_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"数据集保存成功: {dataset_id} | Dataset saved successfully: {dataset_id}")
            
            return {
                "success": True,
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "file_path": target_path,
                "metadata_path": metadata_path,
                "validation_result": validation_result
            }
            
        except Exception as e:
            self.logger.error(f"数据集保存失败: {str(e)} | Dataset save failed: {str(e)}")
            return {
                "success": False,
                "error": f"保存过程中发生错误: {str(e)}",
                "details": {}
            }
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """获取数据集信息
        Get dataset information
        
        Args:
            dataset_id: 数据集ID | Dataset ID
            
        Returns:
            数据集信息或None | Dataset information or None
        """
        try:
            # 查找数据集文件
            for model_id in os.listdir(self.base_data_dir):
                model_dir = os.path.join(self.base_data_dir, model_id)
                if not os.path.isdir(model_dir):
                    continue
                
                metadata_path = os.path.join(model_dir, f"{dataset_id}_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # 添加文件存在性检查
                    file_path = metadata.get("validation_result", {}).get("file_path")
                    if file_path and os.path.exists(file_path):
                        metadata["file_exists"] = True
                        metadata["file_size"] = os.path.getsize(file_path)
                    else:
                        metadata["file_exists"] = False
                    
                    return metadata
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取数据集信息失败: {str(e)} | Get dataset info failed: {str(e)}")
            return None
    
    def list_datasets(self, model_id: str = None) -> List[Dict[str, Any]]:
        """列出所有数据集
        List all datasets
        
        Args:
            model_id: 特定模型的ID (可选) | Specific model ID (optional)
            
        Returns:
            数据集列表 | List of datasets
        """
        datasets = []
        
        try:
            model_dirs = []
            if model_id:
                model_dirs = [model_id]
            else:
                model_dirs = [d for d in os.listdir(self.base_data_dir) 
                             if os.path.isdir(os.path.join(self.base_data_dir, d))]
            
            for model_dir in model_dirs:
                full_model_dir = os.path.join(self.base_data_dir, model_dir)
                if not os.path.isdir(full_model_dir):
                    continue
                
                for file_name in os.listdir(full_model_dir):
                    if file_name.endswith('_metadata.json'):
                        metadata_path = os.path.join(full_model_dir, file_name)
                        try:
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            
                            # 检查数据文件是否存在
                            dataset_id = metadata.get("dataset_id")
                            file_pattern = f"{dataset_id}.*"
                            data_files = [f for f in os.listdir(full_model_dir) 
                                        if f.startswith(dataset_id) and not f.endswith('_metadata.json')]
                            
                            if data_files:
                                metadata["data_file_exists"] = True
                                metadata["data_file"] = data_files[0]
                            else:
                                metadata["data_file_exists"] = False
                            
                            datasets.append(metadata)
                            
                        except Exception as e:
                            self.logger.warning(f"读取元数据文件失败 {file_name}: {str(e)}")
            
            # 按上传时间排序
            datasets.sort(key=lambda x: x.get("upload_time", ""), reverse=True)
            return datasets
            
        except Exception as e:
            self.logger.error(f"列出数据集失败: {str(e)} | List datasets failed: {str(e)}")
            return []
    
    def delete_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """删除数据集
        Delete dataset
        
        Args:
            dataset_id: 数据集ID | Dataset ID
            
        Returns:
            删除结果 | Delete result
        """
        try:
            dataset_info = self.get_dataset_info(dataset_id)
            if not dataset_info:
                return {
                    "success": False,
                    "error": f"数据集不存在: {dataset_id}",
                    "details": {}
                }
            
            model_id = dataset_info.get("model_id")
            model_dir = os.path.join(self.base_data_dir, model_id)
            
            # 删除数据文件
            data_files = [f for f in os.listdir(model_dir) 
                         if f.startswith(dataset_id) and not f.endswith('_metadata.json')]
            
            # 删除元数据文件
            metadata_file = f"{dataset_id}_metadata.json"
            metadata_path = os.path.join(model_dir, metadata_file)
            
            deleted_files = []
            
            # 删除数据文件
            for data_file in data_files:
                data_path = os.path.join(model_dir, data_file)
                if os.path.exists(data_path):
                    os.remove(data_path)
                    deleted_files.append(data_file)
            
            # 删除元数据文件
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                deleted_files.append(metadata_file)
            
            self.logger.info(f"数据集删除成功: {dataset_id} | Dataset deleted successfully: {dataset_id}")
            
            return {
                "success": True,
                "dataset_id": dataset_id,
                "deleted_files": deleted_files,
                "message": f"成功删除数据集 {dataset_id}"
            }
            
        except Exception as e:
            self.logger.error(f"数据集删除失败: {str(e)} | Dataset deletion failed: {str(e)}")
            return {
                "success": False,
                "error": f"删除过程中发生错误: {str(e)}",
                "details": {}
            }
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息
        Get dataset statistics
        
        Returns:
            统计信息 | Statistics
        """
        try:
            stats = {
                "total_datasets": 0,
                "by_model": {},
                "by_format": {},
                "total_size": 0
            }
            
            datasets = self.list_datasets()
            stats["total_datasets"] = len(datasets)
            
            for dataset in datasets:
                model_id = dataset.get("model_id")
                file_format = dataset.get("file_format")
                file_size = dataset.get("file_size", 0)
                
                # 按模型统计
                if model_id not in stats["by_model"]:
                    stats["by_model"][model_id] = {"count": 0, "size": 0}
                stats["by_model"][model_id]["count"] += 1
                stats["by_model"][model_id]["size"] += file_size
                
                # 按格式统计
                if file_format not in stats["by_format"]:
                    stats["by_format"][file_format] = {"count": 0, "size": 0}
                stats["by_format"][file_format]["count"] += 1
                stats["by_format"][file_format]["size"] += file_size
                
                # 总大小
                stats["total_size"] += file_size
            
            return stats
            
        except Exception as e:
            self.logger.error(f"获取数据集统计失败: {str(e)} | Get dataset stats failed: {str(e)}")
            return {
                "total_datasets": 0,
                "by_model": {},
                "by_format": {},
                "total_size": 0,
                "error": str(e)
            }
    
    def get_training_dataset_for_model(self, model_id: str, dataset_name: str = None) -> Dict[str, Any]:
        """
        根据模型ID获取适合的训练数据集
        
        :param model_id: 模型ID
        :param dataset_name: 可选的数据集名称
        :return: 数据集信息字典
        """
        try:
            # 获取模型类型
            from core.system_settings_manager import system_settings_manager
            model_type = system_settings_manager.get_model_setting(model_id, "type", "local")
            
            # 获取数据集存储路径
            dataset_path = os.path.join(self.base_dir, model_id)
            
            # 如果指定了数据集名称，尝试直接加载
            if dataset_name:
                dataset_file = os.path.join(dataset_path, f"{dataset_name}.json")
                if os.path.exists(dataset_file):
                    with open(dataset_file, 'r', encoding='utf-8') as f:
                        dataset_content = json.load(f)
                    return {
                        "success": True,
                        "dataset_name": dataset_name,
                        "content": dataset_content,
                        "model_id": model_id
                    }
                else:
                    return {"success": False, "message": f"Dataset {dataset_name} not found"}
            
            # 检查是否存在适合该模型的数据集
            if not os.path.exists(dataset_path):
                return {"success": False, "message": f"No dataset found for model {model_id}"}
            
            # 查找最近创建的数据集
            datasets = []
            for filename in os.listdir(dataset_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(dataset_path, filename)
                    file_time = os.path.getmtime(file_path)
                    datasets.append((file_time, filename))
            
            if not datasets:
                return {"success": False, "message": f"No datasets found for model {model_id}"}
            
            # 按创建时间排序，选择最新的数据集
            datasets.sort(reverse=True)
            latest_dataset = datasets[0][1]
            
            # 加载数据集内容
            with open(os.path.join(dataset_path, latest_dataset), 'r', encoding='utf-8') as f:
                dataset_content = json.load(f)
            
            return {
                "success": True,
                "dataset_name": latest_dataset.replace('.json', ''),
                "content": dataset_content,
                "model_id": model_id
            }
            
        except Exception as e:
            error_handler.handle_error(e, "DatasetManager", f"Failed to get training dataset for model {model_id}")
            return {"success": False, "message": str(e)}
    
    def create_basic_dataset(self, model_id: str) -> Dict[str, Any]:
        """
        为模型创建基本的训练数据集
        
        :param model_id: 模型ID
        :return: 数据集创建结果
        """
        try:
            # 获取模型类型
            from core.system_settings_manager import system_settings_manager
            model_type = system_settings_manager.get_model_setting(model_id, "type", "local")
            
            # 根据模型类型生成基本数据集
            basic_dataset = self._generate_basic_dataset_for_model_type(model_id, model_type)
            
            # 确保模型目录存在
            model_dir = os.path.join(self.base_dir, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # 生成数据集名称和文件路径
            dataset_name = f"basic_dataset_{int(time.time())}"
            dataset_file = os.path.join(model_dir, f"{dataset_name}.json")
            
            # 保存数据集
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(basic_dataset, f, ensure_ascii=False, indent=2)
            
            # 创建元数据
            metadata_file = os.path.join(model_dir, f"{dataset_name}_metadata.json")
            metadata = {
                "dataset_id": dataset_name,
                "model_id": model_id,
                "model_type": model_type,
                "creation_time": datetime.now().isoformat(),
                "size": len(json.dumps(basic_dataset)),
                "description": "Basic training dataset for model",
                "source": "generated_basic",
                "format": self._get_supported_formats()[model_type][0] if model_type in self._get_supported_formats() else "json"
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 记录日志
            error_handler.log_info(f"Created basic dataset {dataset_name} for model {model_id}", "DatasetManager")
            
            return {
                "success": True,
                "dataset_name": dataset_name,
                "content": basic_dataset,
                "model_id": model_id,
                "metadata": metadata
            }
            
        except Exception as e:
            error_handler.handle_error(e, "DatasetManager", f"Failed to create basic dataset for model {model_id}")
            return {"success": False, "message": str(e)}
    
    def _generate_basic_dataset_for_model_type(self, model_id: str, model_type: str) -> Dict[str, Any]:
        """
        根据模型类型生成基本数据集
        
        :param model_id: 模型ID
        :param model_type: 模型类型
        :return: 生成的数据集内容
        """
        import numpy as np
        
        # 根据模型类型生成适当的基本数据集
        if model_type in ["language", "programming", "knowledge", "manager", "emotion"]:
            # 文本类模型的基本数据集
            texts = [
                "This is a sample text for training.",
                "The quick brown fox jumps over the lazy dog.",
                "Artificial intelligence is transforming the world.",
                "Machine learning models can learn from data.",
                "Training deep neural networks requires a lot of data.",
                "Natural language processing is a subfield of AI.",
                "Computer vision enables machines to see.",
                "Reinforcement learning uses rewards to guide behavior.",
                "Data preprocessing is an important step in machine learning.",
                "Model evaluation helps measure performance."
            ]
            
            # 根据模型ID调整数据集内容
            if model_id.startswith("programming"):
                texts = [
                    "def hello_world():",
                    "    print('Hello, World!')",
                    "x = 5",
                    "y = 10",
                    "result = x + y",
                    "for i in range(5):",
                    "    print(i)",
                    "if x > y:",
                    "    print('x is greater')",
                    "else:",
                    "    print('y is greater')"
                ]
            elif model_id.startswith("knowledge"):
                texts = [
                    "The capital of France is Paris.",
                    "Water is composed of hydrogen and oxygen.",
                    "The human body has 206 bones.",
                    "The Earth orbits around the Sun.",
                    "Photosynthesis converts light energy into chemical energy.",
                    "The speed of light is approximately 299,792,458 meters per second.",
                    "The periodic table contains 118 elements.",
                    "Gravity is a fundamental force of nature.",
                    "DNA carries genetic information.",
                    "The square of the hypotenuse is equal to the sum of the squares of the other two sides."
                ]
            
            # 生成随机标签
            labels = np.random.randint(0, 2, size=len(texts)).tolist()
            
            return {"texts": texts, "labels": labels}
            
        elif model_type in ["vision", "vision_image"]:
            # 视觉类模型的基本数据集
            # 创建模拟图像数据 (10个32x32的3通道图像)
            num_samples = 10
            image_size = 32
            
            images = []
            for _ in range(num_samples):
                # 创建随机图像数据
                image = np.random.randint(0, 256, size=(image_size, image_size, 3)).tolist()
                images.append(image)
            
            # 生成随机标签
            labels = np.random.randint(0, 10, size=num_samples).tolist()
            
            return {"images": images, "labels": labels}
            
        elif model_type in ["audio"]:
            # 音频类模型的基本数据集
            # 创建模拟音频数据
            num_samples = 10
            audio_length = 1000  # 每个音频样本的长度
            
            audio = []
            for _ in range(num_samples):
                # 创建随机音频数据
                audio_sample = np.random.rand(audio_length).tolist()
                audio.append(audio_sample)
            
            # 生成随机标签
            labels = np.random.randint(0, 5, size=num_samples).tolist()
            
            return {"audio": audio, "labels": labels}
            
        elif model_type in ["sensor", "motion"]:
            # 传感器和运动控制模型的基本数据集
            num_samples = 10
            feature_dim = 6  # 6个传感器特征 (如加速度计的x, y, z轴)
            
            features = []
            targets = []
            
            for _ in range(num_samples):
                # 创建随机特征数据
                feature = np.random.rand(feature_dim).tolist()
                # 创建相应的目标数据
                target = np.random.rand(3).tolist()  # 3个控制输出
                
                features.append(feature)
                targets.append(target)
            
            return {"features": features, "targets": targets}
            
        else:
            # 默认通用数据集
            num_samples = 10
            feature_dim = 10
            
            features = []
            labels = []
            
            for _ in range(num_samples):
                feature = np.random.rand(feature_dim).tolist()
                label = np.random.randint(0, 2)
                
                features.append(feature)
                labels.append(label)
            
            return {"features": features, "labels": labels}

# Global dataset manager instance
dataset_manager = DatasetManager()
