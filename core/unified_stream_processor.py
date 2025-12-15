"""
统一流处理框架 - Unified Stream Processing Framework

提供统一的流数据处理功能，消除各个模型中的重复流处理逻辑
Provides unified stream data processing functionality to eliminate duplicate stream processing logic across models

主要功能：
- 实时数据流管理
- 流数据处理管道
- 性能监控和优化
- 错误处理和自动恢复
- 多模态流数据支持
"""

import logging
import time
import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
import queue
import json
import numpy as np
from abc import ABC, abstractmethod


class StreamProcessor(ABC):
    """抽象流处理器基类 | Abstract stream processor base class"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.processor_id = self.__class__.__name__.lower()
        
        # 流处理状态
        self.is_running = False
        self.is_paused = False
        self.stream_start_time = None
        self.stream_end_time = None
        
        # 性能监控
        self.performance_metrics = {
            "total_frames_processed": 0,
            "frames_per_second": 0.0,
            "average_processing_time": 0.0,
            "peak_memory_usage": 0,
            "error_count": 0,
            "recovery_count": 0
        }
        
        # 数据缓冲区
        self.buffer_size = self.config.get('buffer_size', 1000)
        self.data_buffer = queue.Queue(maxsize=self.buffer_size)
        
        # 处理管道
        self.processing_pipeline = []
        self._initialize_pipeline()
        
        # 错误处理
        self.error_history = []
        self.auto_recovery_enabled = self.config.get('auto_recovery', True)
        
        self.logger.info(f"Stream processor initialized: {self.processor_id}")
    
    @abstractmethod
    def _initialize_pipeline(self):
        """初始化处理管道 | Initialize processing pipeline"""
        pass
    
    @abstractmethod
    def process_frame(self, frame_data: Any) -> Dict[str, Any]:
        """处理单个数据帧 | Process single data frame"""
        pass
    
    def start_stream(self) -> Dict[str, Any]:
        """开始流处理 | Start stream processing"""
        try:
            if self.is_running:
                return {"success": False, "error": "Stream is already running"}
            
            self.is_running = True
            self.is_paused = False
            self.stream_start_time = datetime.now()
            
            # 启动处理线程
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.logger.info(f"Stream processing started: {self.processor_id}")
            return {"success": True, "start_time": self.stream_start_time.isoformat()}
            
        except Exception as e:
            self._handle_error(e, "start_stream")
            return {"success": False, "error": str(e)}
    
    def stop_stream(self) -> Dict[str, Any]:
        """停止流处理 | Stop stream processing"""
        try:
            if not self.is_running:
                return {"success": False, "error": "Stream is not running"}
            
            self.is_running = False
            self.stream_end_time = datetime.now()
            
            # 等待处理线程结束
            if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
            
            self.logger.info(f"Stream processing stopped: {self.processor_id}")
            return {
                "success": True, 
                "end_time": self.stream_end_time.isoformat(),
                "total_frames": self.performance_metrics["total_frames_processed"]
            }
            
        except Exception as e:
            self._handle_error(e, "stop_stream")
            return {"success": False, "error": str(e)}
    
    def pause_stream(self) -> Dict[str, Any]:
        """暂停流处理 | Pause stream processing"""
        self.is_paused = True
        self.logger.info(f"Stream processing paused: {self.processor_id}")
        return {"success": True}
    
    def resume_stream(self) -> Dict[str, Any]:
        """恢复流处理 | Resume stream processing"""
        self.is_paused = False
        self.logger.info(f"Stream processing resumed: {self.processor_id}")
        return {"success": True}
    
    def add_frame(self, frame_data: Any) -> Dict[str, Any]:
        """添加数据帧到缓冲区 | Add data frame to buffer"""
        try:
            if not self.is_running:
                return {"success": False, "error": "Stream is not running"}
            
            if self.data_buffer.full():
                return {"success": False, "error": "Buffer is full"}
            
            # 添加时间戳
            framed_data = {
                "data": frame_data,
                "timestamp": datetime.now().isoformat(),
                "frame_id": self.performance_metrics["total_frames_processed"] + 1
            }
            
            self.data_buffer.put(framed_data, block=False)
            return {"success": True, "frame_id": framed_data["frame_id"]}
            
        except queue.Full:
            return {"success": False, "error": "Buffer is full"}
        except Exception as e:
            self._handle_error(e, "add_frame")
            return {"success": False, "error": str(e)}
    
    def _processing_loop(self):
        """处理循环 | Processing loop"""
        frame_times = []
        
        while self.is_running:
            try:
                # 检查是否暂停
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # 从缓冲区获取数据
                try:
                    framed_data = self.data_buffer.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                start_time = time.time()
                
                # 处理数据帧
                result = self.process_frame(framed_data["data"])
                result["frame_id"] = framed_data["frame_id"]
                result["timestamp"] = framed_data["timestamp"]
                result["processing_time"] = time.time() - start_time
                
                # 更新性能指标
                self._update_performance_metrics(result["processing_time"])
                
                # 执行处理管道
                for processor in self.processing_pipeline:
                    try:
                        result = processor(result)
                    except Exception as e:
                        self.logger.warning(f"Pipeline processor failed: {str(e)}")
                
                # 记录处理结果
                self._on_frame_processed(result)
                
                # 更新帧率计算
                frame_times.append(time.time())
                if len(frame_times) > 10:
                    frame_times.pop(0)
                
                if len(frame_times) >= 2:
                    time_diff = frame_times[-1] - frame_times[0]
                    self.performance_metrics["frames_per_second"] = len(frame_times) / time_diff if time_diff > 0 else 0
                
            except Exception as e:
                self._handle_error(e, "processing_loop")
                time.sleep(0.1)  # 防止错误循环
    
    def _update_performance_metrics(self, processing_time: float):
        """更新性能指标 | Update performance metrics"""
        self.performance_metrics["total_frames_processed"] += 1
        
        # 更新平均处理时间
        total_frames = self.performance_metrics["total_frames_processed"]
        current_avg = self.performance_metrics["average_processing_time"]
        self.performance_metrics["average_processing_time"] = (
            (current_avg * (total_frames - 1) + processing_time) / total_frames
        )
    
    def _handle_error(self, error: Exception, context: str):
        """处理错误 | Handle errors"""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "processor_id": self.processor_id
        }
        
        self.error_history.append(error_info)
        self.performance_metrics["error_count"] += 1
        
        self.logger.error(f"Error in {context}: {str(error)}")
        
        # 自动恢复
        if self.auto_recovery_enabled:
            self._attempt_recovery(error, context)
    
    def _attempt_recovery(self, error: Exception, context: str):
        """尝试自动恢复 | Attempt automatic recovery"""
        try:
            self.performance_metrics["recovery_count"] += 1
            
            # 清空缓冲区
            while not self.data_buffer.empty():
                try:
                    self.data_buffer.get_nowait()
                except queue.Empty:
                    break
            
            self.logger.info(f"Auto-recovery completed for {context}")
            
        except Exception as recovery_error:
            self.logger.error(f"Auto-recovery failed: {str(recovery_error)}")
    
    def _on_frame_processed(self, result: Dict[str, Any]):
        """数据帧处理完成回调 | Frame processed callback"""
        # 子类可以重写此方法来自定义处理完成后的行为
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """获取流处理器状态 | Get stream processor status"""
        return {
            "processor_id": self.processor_id,
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "buffer_size": self.data_buffer.qsize(),
            "performance_metrics": self.performance_metrics,
            "stream_duration": self._get_stream_duration()
        }
    
    def _get_stream_duration(self) -> float:
        """获取流处理持续时间 | Get stream processing duration"""
        if not self.stream_start_time:
            return 0.0
        
        end_time = self.stream_end_time or datetime.now()
        return (end_time - self.stream_start_time).total_seconds()
    
    def add_processor_to_pipeline(self, processor: Callable):
        """添加处理器到管道 | Add processor to pipeline"""
        self.processing_pipeline.append(processor)
        self.logger.info(f"Processor added to pipeline: {processor.__name__}")
    
    def clear_pipeline(self):
        """清空处理管道 | Clear processing pipeline"""
        self.processing_pipeline.clear()
        self.logger.info("Processing pipeline cleared")


class VideoStreamProcessor(StreamProcessor):
    """视频流处理器 | Video stream processor"""
    
    def _initialize_pipeline(self):
        """初始化视频处理管道 | Initialize video processing pipeline"""
        # 默认的视频处理管道
        self.processing_pipeline = [
            self._preprocess_frame,
            self._detect_objects,
            self._analyze_motion,
            self._postprocess_results
        ]
    
    def process_frame(self, frame_data: Any) -> Dict[str, Any]:
        """处理视频帧 | Process video frame"""
        try:
            # 基础视频帧处理逻辑
            result = {
                "frame_type": "video",
                "width": frame_data.get('width', 0),
                "height": frame_data.get('height', 0),
                "format": frame_data.get('format', 'unknown'),
                "objects_detected": [],
                "motion_analysis": {},
                "processing_stages": ["initial"]
            }
            
            return result
            
        except Exception as e:
            self._handle_error(e, "process_frame")
            return {"success": False, "error": str(e)}
    
    def _preprocess_frame(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """预处理视频帧 | Preprocess video frame"""
        try:
            # 图像增强、噪声去除等预处理
            frame_data["processing_stages"].append("preprocessed")
            return frame_data
        except Exception as e:
            self.logger.warning(f"Frame preprocessing failed: {str(e)}")
            return frame_data
    
    def _detect_objects(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """目标检测 | Object detection"""
        try:
            # 模拟目标检测逻辑
            frame_data["objects_detected"] = [
                {"class": "person", "confidence": 0.95, "bbox": [100, 100, 200, 300]},
                {"class": "car", "confidence": 0.87, "bbox": [300, 150, 400, 250]}
            ]
            frame_data["processing_stages"].append("object_detection")
            return frame_data
        except Exception as e:
            self.logger.warning(f"Object detection failed: {str(e)}")
            return frame_data
    
    def _analyze_motion(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """运动分析 | Motion analysis"""
        try:
            # 模拟运动分析逻辑
            frame_data["motion_analysis"] = {
                "motion_detected": True,
                "motion_vectors": [[1.2, 0.8], [-0.5, 1.1]],
                "motion_intensity": 0.75
            }
            frame_data["processing_stages"].append("motion_analysis")
            return frame_data
        except Exception as e:
            self.logger.warning(f"Motion analysis failed: {str(e)}")
            return frame_data
    
    def _postprocess_results(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """后处理结果 | Postprocess results"""
        try:
            # 结果整理和格式化
            frame_data["processing_stages"].append("postprocessed")
            frame_data["success"] = True
            return frame_data
        except Exception as e:
            self.logger.warning(f"Postprocessing failed: {str(e)}")
            return frame_data


class AudioStreamProcessor(StreamProcessor):
    """音频流处理器 | Audio stream processor"""
    
    def _initialize_pipeline(self):
        """初始化音频处理管道 | Initialize audio processing pipeline"""
        self.processing_pipeline = [
            self._preprocess_audio,
            self._extract_features,
            self._analyze_speech,
            self._postprocess_audio
        ]
    
    def process_frame(self, audio_data: Any) -> Dict[str, Any]:
        """处理音频帧 | Process audio frame"""
        try:
            result = {
                "frame_type": "audio",
                "sample_rate": audio_data.get('sample_rate', 44100),
                "channels": audio_data.get('channels', 2),
                "duration": audio_data.get('duration', 0.1),
                "features": {},
                "speech_analysis": {},
                "processing_stages": ["initial"]
            }
            
            return result
            
        except Exception as e:
            self._handle_error(e, "process_frame")
            return {"success": False, "error": str(e)}
    
    def _preprocess_audio(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """预处理音频 | Preprocess audio"""
        try:
            # 音频增强、降噪等
            audio_data["processing_stages"].append("preprocessed")
            return audio_data
        except Exception as e:
            self.logger.warning(f"Audio preprocessing failed: {str(e)}")
            return audio_data
    
    def _extract_features(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取音频特征 | Extract audio features"""
        try:
            # 模拟特征提取
            audio_data["features"] = {
                "mfcc": [0.1, 0.2, 0.15, 0.3],
                "spectral_centroid": 4500,
                "zero_crossing_rate": 0.12
            }
            audio_data["processing_stages"].append("feature_extraction")
            return audio_data
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {str(e)}")
            return audio_data
    
    def _analyze_speech(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """语音分析 | Speech analysis"""
        try:
            # 模拟语音分析
            audio_data["speech_analysis"] = {
                "speech_detected": True,
                "language": "english",
                "confidence": 0.89,
                "words": ["hello", "world"]
            }
            audio_data["processing_stages"].append("speech_analysis")
            return audio_data
        except Exception as e:
            self.logger.warning(f"Speech analysis failed: {str(e)}")
            return audio_data
    
    def _postprocess_audio(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """后处理音频 | Postprocess audio"""
        try:
            audio_data["processing_stages"].append("postprocessed")
            audio_data["success"] = True
            return audio_data
        except Exception as e:
            self.logger.warning(f"Audio postprocessing failed: {str(e)}")
            return audio_data


class TextStreamProcessor(StreamProcessor):
    """文本流处理器 | Text stream processor"""
    
    def _initialize_pipeline(self):
        """初始化文本处理管道 | Initialize text processing pipeline"""
        self.processing_pipeline = [
            self._preprocess_text,
            self._analyze_sentiment,
            self._extract_entities,
            self._postprocess_text
        ]
    
    def process_frame(self, text_data: Any) -> Dict[str, Any]:
        """处理文本帧 | Process text frame"""
        try:
            result = {
                "frame_type": "text",
                "text": str(text_data),
                "length": len(str(text_data)),
                "sentiment": {},
                "entities": [],
                "processing_stages": ["initial"]
            }
            
            return result
            
        except Exception as e:
            self._handle_error(e, "process_frame")
            return {"success": False, "error": str(e)}
    
    def _preprocess_text(self, text_data: Dict[str, Any]) -> Dict[str, Any]:
        """预处理文本 | Preprocess text"""
        try:
            # 文本清洗、标准化等
            text_data["processing_stages"].append("preprocessed")
            return text_data
        except Exception as e:
            self.logger.warning(f"Text preprocessing failed: {str(e)}")
            return text_data
    
    def _analyze_sentiment(self, text_data: Dict[str, Any]) -> Dict[str, Any]:
        """情感分析 | Sentiment analysis"""
        try:
            # 模拟情感分析
            text_data["sentiment"] = {
                "polarity": 0.75,
                "subjectivity": 0.6,
                "label": "positive"
            }
            text_data["processing_stages"].append("sentiment_analysis")
            return text_data
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed: {str(e)}")
            return text_data
    
    def _extract_entities(self, text_data: Dict[str, Any]) -> Dict[str, Any]:
        """实体提取 | Entity extraction"""
        try:
            # 模拟实体提取
            text_data["entities"] = [
                {"text": "John", "type": "PERSON", "start": 0, "end": 4},
                {"text": "New York", "type": "LOCATION", "start": 10, "end": 18}
            ]
            text_data["processing_stages"].append("entity_extraction")
            return text_data
        except Exception as e:
            self.logger.warning(f"Entity extraction failed: {str(e)}")
            return text_data
    
    def _postprocess_text(self, text_data: Dict[str, Any]) -> Dict[str, Any]:
        """后处理文本 | Postprocess text"""
        try:
            text_data["processing_stages"].append("postprocessed")
            text_data["success"] = True
            return text_data
        except Exception as e:
            self.logger.warning(f"Text postprocessing failed: {str(e)}")
            return text_data


class UnifiedStreamManager:
    """统一流管理器 | Unified stream manager"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 流处理器注册表
        self.stream_processors = {}
        
        # 性能监控
        self.manager_metrics = {
            "total_streams_managed": 0,
            "active_streams": 0,
            "total_errors": 0,
            "average_stream_duration": 0.0
        }
        
        self.logger.info("Unified stream manager initialized")
    
    def register_processor(self, processor_id: str, processor: StreamProcessor) -> Dict[str, Any]:
        """注册流处理器 | Register stream processor"""
        try:
            if processor_id in self.stream_processors:
                return {"success": False, "error": f"Processor {processor_id} already registered"}
            
            self.stream_processors[processor_id] = processor
            self.manager_metrics["total_streams_managed"] += 1
            
            self.logger.info(f"Stream processor registered: {processor_id}")
            return {"success": True, "processor_id": processor_id}
            
        except Exception as e:
            self._handle_error(e, "register_processor")
            return {"success": False, "error": str(e)}
    
    def start_processor(self, processor_id: str) -> Dict[str, Any]:
        """启动流处理器 | Start stream processor"""
        try:
            if processor_id not in self.stream_processors:
                return {"success": False, "error": f"Processor {processor_id} not found"}
            
            processor = self.stream_processors[processor_id]
            result = processor.start_stream()
            
            if result["success"]:
                self.manager_metrics["active_streams"] += 1
            
            return result
            
        except Exception as e:
            self._handle_error(e, "start_processor")
            return {"success": False, "error": str(e)}
    
    def stop_processor(self, processor_id: str) -> Dict[str, Any]:
        """停止流处理器 | Stop stream processor"""
        try:
            if processor_id not in self.stream_processors:
                return {"success": False, "error": f"Processor {processor_id} not found"}
            
            processor = self.stream_processors[processor_id]
            result = processor.stop_stream()
            
            if result["success"]:
                self.manager_metrics["active_streams"] = max(0, self.manager_metrics["active_streams"] - 1)
            
            return result
            
        except Exception as e:
            self._handle_error(e, "stop_processor")
            return {"success": False, "error": str(e)}
    
    def get_processor_status(self, processor_id: str) -> Dict[str, Any]:
        """获取处理器状态 | Get processor status"""
        try:
            if processor_id not in self.stream_processors:
                return {"success": False, "error": f"Processor {processor_id} not found"}
            
            processor = self.stream_processors[processor_id]
            status = processor.get_status()
            status["processor_id"] = processor_id
            
            return {"success": True, "status": status}
            
        except Exception as e:
            self._handle_error(e, "get_processor_status")
            return {"success": False, "error": str(e)}
    
    def get_all_status(self) -> Dict[str, Any]:
        """获取所有处理器状态 | Get all processors status"""
        try:
            statuses = {}
            for processor_id, processor in self.stream_processors.items():
                statuses[processor_id] = processor.get_status()
            
            return {
                "success": True,
                "statuses": statuses,
                "manager_metrics": self.manager_metrics
            }
            
        except Exception as e:
            self._handle_error(e, "get_all_status")
            return {"success": False, "error": str(e)}
    
    def _handle_error(self, error: Exception, context: str):
        """处理错误 | Handle errors"""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        
        self.manager_metrics["total_errors"] += 1
        self.logger.error(f"Error in {context}: {str(error)}")


# 工厂函数创建流处理器 | Factory functions to create stream processors
def create_video_stream_processor(config: Dict[str, Any] = None) -> VideoStreamProcessor:
    """创建视频流处理器 | Create video stream processor"""
    return VideoStreamProcessor(config)

def create_audio_stream_processor(config: Dict[str, Any] = None) -> AudioStreamProcessor:
    """创建音频流处理器 | Create audio stream processor"""
    return AudioStreamProcessor(config)

def create_text_stream_processor(config: Dict[str, Any] = None) -> TextStreamProcessor:
    """创建文本流处理器 | Create text stream processor"""
    return TextStreamProcessor(config)

def create_unified_stream_manager(config: Dict[str, Any] = None) -> UnifiedStreamManager:
    """创建统一流管理器 | Create unified stream manager"""
    return UnifiedStreamManager(config)


# 导出主要类 | Export main classes
__all__ = [
    'StreamProcessor',
    'VideoStreamProcessor', 
    'AudioStreamProcessor',
    'TextStreamProcessor',
    'UnifiedStreamManager',
    'create_video_stream_processor',
    'create_audio_stream_processor',
    'create_text_stream_processor',
    'create_unified_stream_manager'
]
