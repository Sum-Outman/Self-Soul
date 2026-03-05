"""
自然混合输入接口

支持用户以自然方式混合输入多种模态，如：
"边说语音边传图片边手绘草图"，实现真正的自然交互。

核心功能：
1. 实时模态识别和内容提取
2. 时间同步和内容关联
3. 支持手势、眼动等新兴模态
4. 提供统一的输入流处理
"""

import time
import threading
import queue
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("natural_hybrid_input")


class ModalityType(Enum):
    """模态类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    GESTURE = "gesture"
    EYE_TRACKING = "eye_tracking"
    SKETCH = "sketch"
    SCREENSHARE = "screenshare"
    UNKNOWN = "unknown"


class InputState(Enum):
    """输入状态枚举"""
    WAITING = "waiting"
    RECEIVING = "receiving"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ModalityChunk:
    """模态数据块"""
    modality_type: ModalityType
    data: Any
    timestamp: float
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "modality_type": self.modality_type.value,
            "data_type": type(self.data).__name__,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "metadata": self.metadata,
            "data_size": len(str(self.data)) if hasattr(self.data, '__len__') else 0
        }


@dataclass
class HybridInputGroup:
    """混合输入组"""
    chunks: List[ModalityChunk]
    start_time: float
    end_time: Optional[float] = None
    correlation_score: float = 0.0
    group_id: str = ""
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.group_id:
            self.group_id = f"group_{int(time.time() * 1000)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "group_id": self.group_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "correlation_score": self.correlation_score,
            "chunk_count": len(self.chunks),
            "modalities": [chunk.modality_type.value for chunk in self.chunks]
        }


class NaturalHybridInputInterface:
    """
    自然混合输入接口
    
    核心功能：
    1. 支持"边说语音边传图片边手绘草图"的自然交互
    2. 实现实时模态识别和内容关联
    3. 添加手势、眼动等新兴模态支持
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化自然混合输入接口
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 输入队列
        self.input_queue = queue.Queue()
        
        # 处理线程
        self.processing_thread = None
        self.is_processing = False
        
        # 时间窗口配置（毫秒）
        self.time_window_ms = self.config.get("time_window_ms", 5000)
        self.max_gap_ms = self.config.get("max_gap_ms", 1000)
        
        # 模态检测器映射
        self.modality_detectors = {}
        self._initialize_modality_detectors()
        
        # 关联分析器
        self.correlation_analyzer = CorrelationAnalyzer()
        
        # 状态管理
        self.current_group: Optional[HybridInputGroup] = None
        self.groups: List[HybridInputGroup] = []
        
        # 统计信息
        self.stats = {
            "total_chunks_received": 0,
            "total_groups_created": 0,
            "average_chunks_per_group": 0.0,
            "average_correlation_score": 0.0,
            "supported_modalities": [m.value for m in ModalityType if m != ModalityType.UNKNOWN]
        }
        
        logger.info(f"自然混合输入接口初始化完成，时间窗口: {self.time_window_ms}ms，最大间隙: {self.max_gap_ms}ms")
    
    def _initialize_modality_detectors(self) -> None:
        """初始化模态检测器"""
        # 文本检测器
        self.modality_detectors[ModalityType.TEXT] = TextDetector()
        
        # 图像检测器
        self.modality_detectors[ModalityType.IMAGE] = ImageDetector()
        
        # 音频检测器
        self.modality_detectors[ModalityType.AUDIO] = AudioDetector()
        
        # 视频检测器
        self.modality_detectors[ModalityType.VIDEO] = VideoDetector()
        
        # 手势检测器
        self.modality_detectors[ModalityType.GESTURE] = GestureDetector()
        
        # 眼动检测器
        self.modality_detectors[ModalityType.EYE_TRACKING] = EyeTrackingDetector()
        
        # 草图检测器
        self.modality_detectors[ModalityType.SKETCH] = SketchDetector()
        
        # 屏幕共享检测器
        self.modality_detectors[ModalityType.SCREENSHARE] = ScreenshareDetector()
        
        logger.info(f"初始化了 {len(self.modality_detectors)} 种模态检测器")
    
    def start_processing(self) -> None:
        """开始处理输入流"""
        if self.is_processing:
            logger.warning("处理线程已经在运行")
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_input_stream, daemon=True)
        self.processing_thread.start()
        logger.info("开始处理输入流")
    
    def stop_processing(self) -> None:
        """停止处理输入流"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            logger.info("停止处理输入流")
    
    def receive_input(self, data: Any, modality_hint: Optional[str] = None) -> str:
        """
        接收输入数据
        
        Args:
            data: 输入数据
            modality_hint: 模态提示（可选）
            
        Returns:
            数据块ID
        """
        current_time = time.time()
        
        # 检测模态类型
        modality_type = self._detect_modality(data, modality_hint)
        
        # 创建数据块
        chunk = ModalityChunk(
            modality_type=modality_type,
            data=data,
            timestamp=current_time,
            metadata={
                "detection_confidence": 1.0,
                "modality_hint": modality_hint
            }
        )
        
        # 添加到队列
        chunk_id = f"chunk_{int(current_time * 1000)}_{modality_type.value}"
        self.input_queue.put((chunk_id, chunk))
        
        self.stats["total_chunks_received"] += 1
        
        logger.debug(f"接收输入数据，ID: {chunk_id}, 模态: {modality_type.value}")
        
        return chunk_id
    
    def _detect_modality(self, data: Any, modality_hint: Optional[str] = None) -> ModalityType:
        """
        检测数据类型对应的模态
        
        Args:
            data: 输入数据
            modality_hint: 模态提示
            
        Returns:
            模态类型
        """
        # 如果有提示，优先使用提示
        if modality_hint:
            try:
                return ModalityType(modality_hint.lower())
            except ValueError:
                logger.warning(f"无效的模态提示: {modality_hint}")
        
        # 根据数据类型检测
        if isinstance(data, str) and len(data) < 10000:  # 假设为文本
            return ModalityType.TEXT
        
        # 检查是否是图像数据
        if hasattr(data, 'shape') and len(getattr(data, 'shape', [])) in [2, 3, 4]:
            return ModalityType.IMAGE
        
        # 检查是否是音频数据
        if hasattr(data, 'shape') and len(getattr(data, 'shape', [])) == 1:
            return ModalityType.AUDIO
        
        # 默认返回未知
        logger.warning(f"无法检测数据类型: {type(data)}")
        return ModalityType.UNKNOWN
    
    def _process_input_stream(self) -> None:
        """处理输入流"""
        buffer = []
        
        while self.is_processing:
            try:
                # 从队列获取数据（带超时）
                chunk_id, chunk = self.input_queue.get(timeout=0.1)
                buffer.append((chunk_id, chunk))
                
                # 处理缓冲区
                self._process_buffer(buffer)
                
            except queue.Empty:
                # 检查缓冲区是否有需要处理的旧数据
                if buffer:
                    self._process_buffer(buffer)
                continue
            except Exception as e:
                logger.error(f"处理输入流时出错: {e}")
                continue
    
    def _process_buffer(self, buffer: List[Tuple[str, ModalityChunk]]) -> None:
        """处理缓冲区中的数据"""
        if not buffer:
            return
        
        current_time = time.time()
        
        # 检查是否需要开始新组
        if self.current_group is None:
            # 开始新组
            chunk_id, first_chunk = buffer[0]
            self.current_group = HybridInputGroup(
                chunks=[first_chunk],
                start_time=first_chunk.timestamp
            )
            buffer.pop(0)
            logger.debug(f"开始新输入组: {self.current_group.group_id}")
        
        # 处理缓冲区中的剩余数据
        i = 0
        while i < len(buffer):
            chunk_id, chunk = buffer[i]
            
            # 检查时间差
            time_diff_ms = (chunk.timestamp - self.current_group.chunks[-1].timestamp) * 1000
            
            if time_diff_ms <= self.time_window_ms:
                # 在时间窗口内，添加到当前组
                self.current_group.chunks.append(chunk)
                buffer.pop(i)
                
                # 更新相关性分数
                self._update_correlation_score()
            else:
                # 超出时间窗口，完成当前组
                self._complete_current_group()
                i += 1
        
        # 检查是否需要完成当前组
        if self.current_group and len(self.current_group.chunks) > 0:
            last_chunk_time = self.current_group.chunks[-1].timestamp
            time_since_last = (current_time - last_chunk_time) * 1000
            
            if time_since_last > self.max_gap_ms:
                self._complete_current_group()
    
    def _update_correlation_score(self) -> None:
        """更新相关性分数"""
        if not self.current_group or len(self.current_group.chunks) < 2:
            return
        
        # 使用关联分析器计算相关性
        correlation_score = self.correlation_analyzer.analyze(
            self.current_group.chunks
        )
        
        self.current_group.correlation_score = correlation_score
    
    def _complete_current_group(self) -> None:
        """完成当前输入组"""
        if not self.current_group or len(self.current_group.chunks) == 0:
            return
        
        self.current_group.end_time = time.time()
        
        # 计算最终相关性分数
        self._update_correlation_score()
        
        # 添加到组列表
        self.groups.append(self.current_group)
        
        # 更新统计
        self.stats["total_groups_created"] += 1
        self.stats["average_chunks_per_group"] = (
            self.stats["average_chunks_per_group"] * (self.stats["total_groups_created"] - 1) +
            len(self.current_group.chunks)
        ) / self.stats["total_groups_created"]
        
        self.stats["average_correlation_score"] = (
            self.stats["average_correlation_score"] * (self.stats["total_groups_created"] - 1) +
            self.current_group.correlation_score
        ) / self.stats["total_groups_created"]
        
        logger.info(f"完成输入组: {self.current_group.group_id}, "
                   f"模态数: {len(self.current_group.chunks)}, "
                   f"相关性: {self.current_group.correlation_score:.2f}")
        
        # 重置当前组
        self.current_group = None
    
    def get_latest_group(self) -> Optional[Dict[str, Any]]:
        """获取最新的输入组"""
        if not self.groups:
            return None
        
        latest_group = self.groups[-1]
        result = latest_group.to_dict()
        result["chunks"] = [chunk.to_dict() for chunk in latest_group.chunks]
        
        return result
    
    def get_all_groups(self) -> List[Dict[str, Any]]:
        """获取所有输入组"""
        result = []
        for group in self.groups:
            group_data = group.to_dict()
            group_data["chunks"] = [chunk.to_dict() for chunk in group.chunks]
            result.append(group_data)
        
        return result
    
    def clear_groups(self) -> None:
        """清除所有输入组"""
        self.groups.clear()
        self.current_group = None
        logger.info("已清除所有输入组")


# ==================== 模态检测器实现 ====================

class ModalityDetector:
    """模态检测器基类"""
    
    def detect(self, data: Any) -> Tuple[bool, float]:
        """
        检测数据是否属于该模态
        
        Args:
            data: 输入数据
            
        Returns:
            (是否属于该模态, 置信度)
        """
        raise NotImplementedError


class TextDetector(ModalityDetector):
    """文本检测器"""
    
    def detect(self, data: Any) -> Tuple[bool, float]:
        if isinstance(data, str):
            # 简单文本检测：检查是否主要是可打印字符
            if len(data) > 0 and all(c.isprintable() or c in '\n\t\r' for c in data):
                return True, 0.9
        return False, 0.0


class ImageDetector(ModalityDetector):
    """图像检测器"""
    
    def detect(self, data: Any) -> Tuple[bool, float]:
        try:
            import numpy as np
            import PIL.Image
            
            if isinstance(data, np.ndarray):
                # NumPy数组
                shape = data.shape
                if len(shape) in [2, 3] and shape[0] > 10 and shape[1] > 10:
                    return True, 0.8
            
            if isinstance(data, PIL.Image.Image):
                # PIL图像
                return True, 0.9
                
        except ImportError:
            pass
        
        return False, 0.0


class AudioDetector(ModalityDetector):
    """音频检测器"""
    
    def detect(self, data: Any) -> Tuple[bool, float]:
        try:
            import numpy as np
            
            if isinstance(data, np.ndarray):
                # 一维或二维数组，可能是音频
                shape = data.shape
                if len(shape) in [1, 2] and shape[0] > 100:
                    return True, 0.7
                    
        except ImportError:
            pass
        
        return False, 0.0


class VideoDetector(ModalityDetector):
    """视频检测器"""
    
    def detect(self, data: Any) -> Tuple[bool, float]:
        return False, 0.0  # 简化实现


class GestureDetector(ModalityDetector):
    """手势检测器"""
    
    def detect(self, data: Any) -> Tuple[bool, float]:
        return False, 0.0  # 简化实现


class EyeTrackingDetector(ModalityDetector):
    """眼动检测器"""
    
    def detect(self, data: Any) -> Tuple[bool, float]:
        return False, 0.0  # 简化实现


class SketchDetector(ModalityDetector):
    """草图检测器"""
    
    def detect(self, data: Any) -> Tuple[bool, float]:
        return False, 0.0  # 简化实现


class ScreenshareDetector(ModalityDetector):
    """屏幕共享检测器"""
    
    def detect(self, data: Any) -> Tuple[bool, float]:
        return False, 0.0  # 简化实现


# ==================== 关联分析器 ====================

class CorrelationAnalyzer:
    """关联分析器"""
    
    def __init__(self):
        """初始化关联分析器"""
        pass
    
    def analyze(self, chunks: List[ModalityChunk]) -> float:
        """
        分析模态块之间的相关性
        
        Args:
            chunks: 模态块列表
            
        Returns:
            相关性分数 (0.0-1.0)
        """
        if len(chunks) < 2:
            return 1.0
        
        # 计算时间相关性
        time_scores = []
        for i in range(len(chunks) - 1):
            time_diff = abs(chunks[i+1].timestamp - chunks[i].timestamp)
            # 时间差越小，相关性越高（假设在10秒内）
            time_score = max(0.0, 1.0 - (time_diff / 10.0))
            time_scores.append(time_score)
        
        # 计算模态组合相关性
        modality_scores = []
        modality_pairs = [
            (ModalityType.TEXT, ModalityType.IMAGE),  # 文本和图像相关性高
            (ModalityType.AUDIO, ModalityType.IMAGE),  # 音频和图像相关性高
            (ModalityType.TEXT, ModalityType.AUDIO),  # 文本和音频相关性高
        ]
        
        for i in range(len(chunks)):
            for j in range(i+1, len(chunks)):
                mod1 = chunks[i].modality_type
                mod2 = chunks[j].modality_type
                
                # 检查是否为高相关性模态对
                for pair in modality_pairs:
                    if (mod1 == pair[0] and mod2 == pair[1]) or (mod1 == pair[1] and mod2 == pair[0]):
                        modality_scores.append(0.8)
                        break
                else:
                    modality_scores.append(0.5)  # 默认相关性
        
        # 计算总分数
        if time_scores:
            time_avg = sum(time_scores) / len(time_scores)
        else:
            time_avg = 0.5
        
        if modality_scores:
            modality_avg = sum(modality_scores) / len(modality_scores)
        else:
            modality_avg = 0.5
        
        # 加权平均
        total_score = (time_avg * 0.6 + modality_avg * 0.4)
        
        return min(1.0, max(0.0, total_score))


# ==================== 测试函数 ====================

def test_natural_hybrid_input() -> None:
    """测试自然混合输入接口"""
    print("测试自然混合输入接口...")
    
    # 创建接口实例
    interface = NaturalHybridInputInterface({
        "time_window_ms": 3000,
        "max_gap_ms": 1000
    })
    
    # 开始处理
    interface.start_processing()
    
    # 模拟混合输入
    print("模拟混合输入...")
    
    # 1. 文本输入
    interface.receive_input("请帮我分析这张图片", "text")
    
    # 2. 图像输入
    import numpy as np
    test_image = np.random.rand(224, 224, 3)
    interface.receive_input(test_image, "image")
    
    # 3. 音频输入
    test_audio = np.random.randn(16000)
    interface.receive_input(test_audio, "audio")
    
    # 获取结果
    latest_group = interface.get_latest_group()
    if latest_group:
        print(f"生成的输入组: {latest_group['group_id']}")
        print(f"包含模态: {latest_group['modalities']}")
        print(f"相关性分数: {latest_group['correlation_score']:.2f}")
        print(f"数据块数: {latest_group['chunk_count']}")
    else:
        print("未生成输入组")
    
    # 停止处理
    interface.stop_processing()
    
    # 打印统计信息
    print(f"\n统计信息:")
    for key, value in interface.stats.items():
        print(f"  {key}: {value}")
    
    print("测试完成！")


if __name__ == "__main__":
    test_natural_hybrid_input()