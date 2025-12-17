"""
工作记忆系统模块 - 实现AGI级的认知工作记忆
支持多模态信息的暂存、整合和检索
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import hashlib
from collections import OrderedDict, deque

# 导入多模态处理器
try:
    import os
    import sys
    # 添加当前目录到Python路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from multimodal_processor import MultimodalProcessor
except ImportError as e:
    MultimodalProcessor = None
    logging.warning(f"无法导入多模态处理器，将使用模拟嵌入: {e}")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    """工作记忆项数据结构"""
    memory_id: str
    content: Any
    modality: str
    embedding: Optional[np.ndarray] = None
    confidence: float = 0.0
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    activation_level: float = 1.0  # 激活水平，影响记忆的可访问性
    relevance_score: float = 1.0  # 与当前任务的相关性得分
    source: str = "system"  # 记忆来源
    decay_rate: float = 0.01  # 记忆衰减率
    last_accessed: datetime = None
    access_count: int = 0
    associative_links: List[str] = None  # 与其他记忆项的关联链接
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
        if self.metadata is None:
            self.metadata = {}
        if self.associative_links is None:
            self.associative_links = []

class WorkingMemory:
    """
    工作记忆系统 - 实现多模态信息的暂存、整合和动态检索
    基于Baddeley的工作记忆模型，包含中央执行系统、语音环路、视觉空间画板和情景缓冲器
    """
    
    def __init__(self, capacity: int = 7, decay_factor: float = 0.99, activation_threshold: float = 0.1):
        """初始化工作记忆系统"""
        # 工作记忆容量配置（基于Miller的7±2理论）
        self.capacity = capacity
        self.decay_factor = decay_factor  # 记忆衰减因子
        self.activation_threshold = activation_threshold  # 记忆激活阈值
        
        # 工作记忆组件
        self.central_executive = {}  # 中央执行系统 - 管理整体记忆
        self.phonological_loop = OrderedDict()  # 语音环路 - 处理语言信息
        self.visuospatial_sketchpad = OrderedDict()  # 视觉空间画板 - 处理视觉空间信息
        self.episodic_buffer = OrderedDict()  # 情景缓冲器 - 整合多模态信息
        
        # 记忆索引
        self.memory_index = {}
        self.association_graph = {}
        
        # 性能统计
        self.stats = {
            'total_items': 0,
            'add_operations': 0,
            'retrieve_operations': 0,
            'forget_operations': 0,
            'association_count': 0
        }
        
        # 初始化多模态处理器
        self.multimodal_processor = None
        if MultimodalProcessor is not None:
            try:
                self.multimodal_processor = MultimodalProcessor()
                logger.info("多模态处理器已集成到工作记忆系统")
            except Exception as e:
                logger.error(f"初始化多模态处理器失败: {e}")
        
        logger.info("工作记忆系统初始化完成")
    
    def add_memory(self, content: Any, modality: str, embedding: Optional[np.ndarray] = None,
                  confidence: float = 0.0, metadata: Optional[Dict[str, Any]] = None,
                  source: str = "system") -> str:
        """
        添加记忆项到工作记忆
        
        参数:
            content: 记忆内容
            modality: 记忆模态
            embedding: 内容的嵌入表示
            confidence: 记忆的置信度
            metadata: 记忆的元数据
            source: 记忆来源
        
        返回:
            记忆项ID
        """
        # 生成唯一记忆ID
        memory_id = hashlib.sha256(f"{content}{modality}{datetime.now().timestamp()}".encode()).hexdigest()[:16]
        
        # 如果没有提供嵌入，尝试使用多模态处理器生成
        if embedding is None and self.multimodal_processor is not None:
            try:
                processed = self.multimodal_processor.process_single_modality(content, modality)
                if processed and hasattr(processed, 'embedding') and processed.embedding is not None:
                    embedding = processed.embedding
                    confidence = processed.confidence
            except Exception as e:
                logger.debug(f"使用多模态处理器生成嵌入失败: {e}")
        
        # 创建记忆项
        memory_item = MemoryItem(
            memory_id=memory_id,
            content=content,
            modality=modality,
            embedding=embedding,
            confidence=confidence,
            metadata=metadata,
            source=source
        )
        
        # 添加到相应的工作记忆组件
        if modality == 'text':
            self._add_to_phonological_loop(memory_id, memory_item)
        elif modality in ['image', 'sensor', 'audio']:
            self._add_to_visuospatial_sketchpad(memory_id, memory_item)
        
        # 添加到情景缓冲器
        self._add_to_episodic_buffer(memory_id, memory_item)
        
        # 更新中央执行系统
        self.central_executive[memory_id] = memory_item
        self.memory_index[memory_id] = memory_item
        
        # 触发记忆整合
        self._integrate_memory(memory_item)
        
        # 更新统计信息
        self.stats['total_items'] += 1
        self.stats['add_operations'] += 1
        
        logger.debug(f"添加记忆项: {memory_id}, 模态: {modality}")
        return memory_id
    
    def _add_to_phonological_loop(self, memory_id: str, memory_item: MemoryItem):
        """添加到语音环路"""
        if len(self.phonological_loop) >= self.capacity:
            # 移除最旧的记忆项
            oldest_id, _ = self.phonological_loop.popitem(last=False)
            self._forget_memory(oldest_id)
        
        self.phonological_loop[memory_id] = memory_item
    
    def _add_to_visuospatial_sketchpad(self, memory_id: str, memory_item: MemoryItem):
        """添加到视觉空间画板"""
        if len(self.visuospatial_sketchpad) >= self.capacity:
            # 移除最旧的记忆项
            oldest_id, _ = self.visuospatial_sketchpad.popitem(last=False)
            self._forget_memory(oldest_id)
        
        self.visuospatial_sketchpad[memory_id] = memory_item
    
    def _add_to_episodic_buffer(self, memory_id: str, memory_item: MemoryItem):
        """添加到情景缓冲器"""
        if len(self.episodic_buffer) >= self.capacity * 2:  # 情景缓冲器容量更大
            # 移除激活水平最低的记忆项
            least_activated = min(self.episodic_buffer.values(), key=lambda x: x.activation_level)
            self._forget_memory(least_activated.memory_id)
        
        self.episodic_buffer[memory_id] = memory_item
    
    def _integrate_memory(self, new_memory: MemoryItem):
        """整合新记忆与现有记忆，建立关联"""
        # 寻找相似记忆
        similar_memories = self._find_similar_memories(new_memory)
        
        # 计算动态阈值
        dynamic_threshold = self._calculate_dynamic_threshold()
        
        for similar_id, similarity_score in similar_memories:
            # 计算多因素关联评分
            association_score = self._calculate_association_score(new_memory, self.memory_index[similar_id], similarity_score)
            
            if association_score > dynamic_threshold:
                # 建立关联
                self._create_association(new_memory.memory_id, similar_id, association_score)
    
    def _find_similar_memories(self, memory: MemoryItem) -> List[Tuple[str, float]]:
        """查找相似记忆项"""
        if memory.embedding is None:
            return []
        
        similar = []
        for mem_id, mem_item in self.memory_index.items():
            if mem_id == memory.memory_id or mem_item.embedding is None:
                continue
            
            # 计算余弦相似度
            similarity = self._cosine_similarity(memory.embedding, mem_item.embedding)
            similar.append((mem_id, similarity))
        
        # 按相似度排序
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:3]  # 返回前3个最相似的记忆
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        # 确保向量是一维的
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        
        # 处理不同维度的向量
        min_dim = min(len(vec1), len(vec2))
        if min_dim == 0:
            return 0.0
        
        # 截断到相同维度
        vec1 = vec1[:min_dim]
        vec2 = vec2[:min_dim]
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_dynamic_threshold(self) -> float:
        """计算动态关联阈值，根据当前记忆状态调整"""
        if self.stats['total_items'] < 5:
            return 0.3  # 初始阶段使用较低阈值，促进关联形成
        
        # 根据记忆数量和关联密度动态调整阈值
        association_density = self.stats['association_count'] / max(1, self.stats['total_items'])
        
        # 动态阈值范围：0.4-0.7
        threshold = 0.4 + (association_density / 10) * 0.3
        return min(0.7, max(0.4, threshold))
    
    def _calculate_association_score(self, memory1: MemoryItem, memory2: MemoryItem, similarity_score: float) -> float:
        """计算多因素关联评分"""
        # 基础相似度评分
        base_score = similarity_score
        
        # 模态兼容性评分
        modality_score = 1.0
        if memory1.modality == memory2.modality:
            modality_score = 1.2  # 相同模态的记忆关联更强
        elif (memory1.modality in ['text'] and memory2.modality in ['text']):
            modality_score = 1.1
        # 增强跨模态关联的合理性
        elif ((memory1.modality == 'text' and memory2.modality in ['image', 'audio']) or 
              (memory2.modality == 'text' and memory1.modality in ['image', 'audio'])):
            modality_score = 1.05  # 文本与其他模态的关联有一定增强
        
        # 时间接近性评分
        time_diff = abs((memory1.timestamp - memory2.timestamp).total_seconds()) / 60.0  # 分钟数
        temporal_score = max(0.5, 1.0 - (time_diff / 30.0))  # 30分钟内的记忆关联更强
        
        # 置信度一致性评分
        confidence_diff = abs(memory1.confidence - memory2.confidence)
        confidence_score = max(0.8, 1.0 - confidence_diff)
        
        # 主题一致性评分
        topic_score = 1.0
        if memory1.modality == 'text' and memory2.modality == 'text':
            # 简单的文本主题一致性检查
            text1 = str(memory1.content).lower()
            text2 = str(memory2.content).lower()
            # 检查共同关键词数量
            common_words = set(text1.split()) & set(text2.split())
            if len(common_words) >= 2:
                topic_score = 1.3
            elif len(common_words) == 1:
                topic_score = 1.1
        
        # 组合评分
        association_score = base_score * modality_score * temporal_score * confidence_score * topic_score
        
        # 归一化到0-1范围
        return min(1.0, max(0.0, association_score))
    
    def _create_association(self, memory_id1: str, memory_id2: str, strength: float):
        """建立两个记忆项之间的关联"""
        # 检查记忆是否存在
        if memory_id1 not in self.memory_index or memory_id2 not in self.memory_index:
            return
        
        # 更新记忆项的关联链接
        if memory_id2 not in self.memory_index[memory_id1].associative_links:
            self.memory_index[memory_id1].associative_links.append(memory_id2)
            
        if memory_id1 not in self.memory_index[memory_id2].associative_links:
            self.memory_index[memory_id2].associative_links.append(memory_id1)
        
        # 更新关联图，支持强度更新
        if memory_id1 not in self.association_graph:
            self.association_graph[memory_id1] = {}
            
        if memory_id2 not in self.association_graph:
            self.association_graph[memory_id2] = {}
        
        # 如果关联已存在，根据强度变化进行更新
        if memory_id2 in self.association_graph[memory_id1]:
            existing_strength = self.association_graph[memory_id1][memory_id2]
            # 关联强度随时间动态变化：新强度与旧强度的加权平均
            # 这样既考虑了新的相似度，又保留了历史关联强度
            new_strength = existing_strength * 0.6 + strength * 0.4
        else:
            new_strength = strength
            self.stats['association_count'] += 1  # 只有新关联才增加计数
        
        # 确保强度在0-1范围内
        new_strength = min(1.0, max(0.0, new_strength))
        
        self.association_graph[memory_id1][memory_id2] = new_strength
        self.association_graph[memory_id2][memory_id1] = new_strength
    
    def retrieve_memory(self, query: Any, modality: str = None, limit: int = 5) -> List[MemoryItem]:
        """
        检索记忆项
        
        参数:
            query: 检索查询
            modality: 限制记忆模态
            limit: 返回结果数量限制
        
        返回:
            匹配的记忆项列表
        """
        self.stats['retrieve_operations'] += 1
        
        # 计算查询嵌入
        query_embedding = self._get_embedding(query, modality)
        
        # 检索匹配的记忆
        if query_embedding is not None:
            return self._semantic_retrieve(query_embedding, modality, limit)
        else:
            return self._exact_retrieve(query, modality, limit)
    
    def _semantic_retrieve(self, query_embedding: np.ndarray, modality: str = None, limit: int = 5) -> List[MemoryItem]:
        """基于语义嵌入的记忆检索"""
        results = []
        
        for mem_id, mem_item in self.memory_index.items():
            if modality and mem_item.modality != modality:
                continue
            
            if mem_item.embedding is None:
                continue
            
            # 计算相似度
            similarity = self._cosine_similarity(query_embedding, mem_item.embedding)
            
            # 更新记忆激活水平
            mem_item.activation_level = min(1.0, mem_item.activation_level + similarity * 0.3)
            mem_item.last_accessed = datetime.now()
            mem_item.access_count += 1
            
            results.append((mem_item, similarity))
        
        # 按相似度和激活水平排序
        results.sort(key=lambda x: (x[1] * 0.7 + x[0].activation_level * 0.3), reverse=True)
        
        # 扩散激活：提高检索结果相关联记忆的激活水平
        retrieved_memories = [result[0] for result in results[:limit]]
        self._spread_activation(retrieved_memories)
        
        return retrieved_memories
    
    def _exact_retrieve(self, query: Any, modality: str = None, limit: int = 5) -> List[MemoryItem]:
        """精确匹配的记忆检索"""
        results = []
        
        for mem_id, mem_item in self.memory_index.items():
            if modality and mem_item.modality != modality:
                continue
            
            if str(query) in str(mem_item.content):
                # 更新记忆激活水平
                mem_item.activation_level = min(1.0, mem_item.activation_level + 0.5)
                mem_item.last_accessed = datetime.now()
                mem_item.access_count += 1
                
                results.append(mem_item)
        
        # 按激活水平和访问时间排序
        results.sort(key=lambda x: (x.activation_level, x.last_accessed), reverse=True)
        
        # 扩散激活：提高检索结果相关联记忆的激活水平
        retrieved_memories = results[:limit]
        self._spread_activation(retrieved_memories)
        
        return retrieved_memories
    
    def _get_embedding(self, content: Any, modality: str = None) -> Optional[np.ndarray]:
        """获取内容的嵌入表示"""
        # 如果多模态处理器可用，使用真实特征提取
        if self.multimodal_processor is not None and modality is not None:
            try:
                processed = self.multimodal_processor.process_single_modality(content, modality)
                if processed and hasattr(processed, 'embedding') and processed.embedding is not None:
                    return processed.embedding
            except Exception as e:
                logger.debug(f"多模态处理器获取嵌入失败: {e}")
        
        # 回退到模拟嵌入生成
        if isinstance(content, str) and modality == 'text':
            # 使用哈希生成简单嵌入
            hash_value = hashlib.md5(content.encode()).hexdigest()
            embedding = np.array([int(hash_value[i:i+2], 16) / 255.0 for i in range(0, 32, 2)])
            return embedding
        elif modality in ['image', 'sensor', 'audio']:
            # 生成随机嵌入
            return np.random.randn(64)
        
        return None
    
    def update_memory_activation(self, memory_id: str, activation_change: float):
        """更新记忆项的激活水平"""
        if memory_id in self.memory_index:
            mem_item = self.memory_index[memory_id]
            mem_item.activation_level = max(0.0, min(1.0, mem_item.activation_level + activation_change))
    
    def _spread_activation(self, retrieved_memories: List[MemoryItem]):
        """扩散激活机制：提高检索到的记忆的关联记忆的激活水平"""
        # 扩散激活的参数
        activation_spread = 0.2  # 初始激活扩散强度
        decay_factor = 0.5  # 激活衰减因子
        max_spread_depth = 2  # 最大扩散深度
        
        for mem_item in retrieved_memories:
            # 激活扩散队列，存储(记忆ID, 当前激活强度, 当前深度)
            activation_queue = deque()
            visited = set()  # 避免循环激活
            
            # 添加初始记忆
            activation_queue.append((mem_item.memory_id, activation_spread, 0))
            visited.add(mem_item.memory_id)
            
            while activation_queue:
                current_id, current_activation, depth = activation_queue.popleft()
                
                # 获取当前记忆项
                current_mem = self.memory_index.get(current_id)
                if not current_mem:
                    continue
                
                # 遍历所有关联的记忆
                for linked_id in current_mem.associative_links:
                    if linked_id in visited or linked_id not in self.memory_index:
                        continue
                    
                    # 获取关联强度
                    association_strength = self.association_graph.get(current_id, {}).get(linked_id, 0.0)
                    
                    # 计算新的激活强度：当前激活 * 关联强度 * 衰减因子
                    new_activation = current_activation * association_strength * decay_factor
                    
                    # 更新关联记忆的激活水平
                    linked_mem = self.memory_index[linked_id]
                    linked_mem.activation_level = min(1.0, linked_mem.activation_level + new_activation)
                    linked_mem.last_accessed = datetime.now()  # 更新访问时间
                    
                    # 如果深度小于最大深度，继续扩散
                    if depth < max_spread_depth:
                        activation_queue.append((linked_id, new_activation, depth + 1))
                        visited.add(linked_id)
    
    def _forget_memory(self, memory_id: str):
        """忘记记忆项"""
        if memory_id in self.memory_index:
            # 从所有组件中移除
            if memory_id in self.central_executive:
                del self.central_executive[memory_id]
            if memory_id in self.phonological_loop:
                del self.phonological_loop[memory_id]
            if memory_id in self.visuospatial_sketchpad:
                del self.visuospatial_sketchpad[memory_id]
            if memory_id in self.episodic_buffer:
                del self.episodic_buffer[memory_id]
            
            # 更新关联图
            if memory_id in self.association_graph:
                for associated_id in self.association_graph[memory_id]:
                    if associated_id in self.association_graph:
                        if memory_id in self.association_graph[associated_id]:
                            del self.association_graph[associated_id][memory_id]
                del self.association_graph[memory_id]
            
            # 更新统计信息
            del self.memory_index[memory_id]
            self.stats['total_items'] -= 1
            self.stats['forget_operations'] += 1
            
            logger.debug(f"遗忘记忆项: {memory_id}")
    
    def decay_memories(self):
        """记忆衰减处理"""
        to_forget = []
        current_time = datetime.now()
        
        for mem_id, mem_item in self.memory_index.items():
            # 计算时间差（分钟数）
            time_diff = (current_time - mem_item.last_accessed).total_seconds() / 60.0
            
            # 根据记忆类型调整衰减因子
            modality_decay_factor = self._get_modality_decay_factor(mem_item.modality)
            
            # 计算衰减速率，考虑激活水平、访问频率和置信度
            decay_rate = self._calculate_decay_rate(mem_item, time_diff, modality_decay_factor)
            
            # 更新激活水平
            mem_item.activation_level = max(0.0, mem_item.activation_level * decay_rate)
            
            # 检查是否需要遗忘
            if mem_item.activation_level < self.activation_threshold:
                to_forget.append(mem_id)
        
        # 遗忘激活水平低于阈值的记忆
        for mem_id in to_forget:
            self._forget_memory(mem_id)
    
    def _get_modality_decay_factor(self, modality: str) -> float:
        """
        根据记忆模态获取模态特定的衰减因子
        
        参数:
            modality: 记忆的模态类型
            
        返回:
            模态特定的衰减因子，值越大衰减越快
        """
        # 不同模态的记忆有不同的衰减特性
        # 文本记忆相对稳定，衰减较慢
        # 图像和音频记忆衰减适中
        # 传感器数据衰减较快
        modality_decay_map = {
            'text': 0.98,       # 文本记忆衰减最慢
            'image': 0.95,      # 图像记忆衰减适中
            'audio': 0.94,      # 音频记忆衰减适中
            'sensor': 0.92      # 传感器数据衰减最快
        }
        
        # 默认衰减因子
        return modality_decay_map.get(modality, 0.95)
    
    def _calculate_decay_rate(self, memory_item: MemoryItem, time_diff: float, modality_decay_factor: float) -> float:
        """
        计算记忆衰减速率
        
        参数:
            memory_item: 记忆项
            time_diff: 自上次访问以来的时间差（分钟）
            modality_decay_factor: 模态特定的衰减因子
            
        返回:
            计算得到的衰减速率
        """
        # 基础衰减率基于时间差和模态衰减因子
        base_decay = modality_decay_factor ** time_diff
        
        # 激活水平影响衰减速率：激活水平高的记忆衰减较慢
        activation_factor = 1.0 - (1.0 - memory_item.activation_level) * 0.3
        
        # 访问频率影响衰减速率：访问频率高的记忆衰减较慢
        # 将访问频率归一化到0-1范围
        normalized_access_count = min(1.0, memory_item.access_count / 10.0)
        access_factor = 1.0 - (1.0 - normalized_access_count) * 0.2
        
        # 置信度影响衰减速率：置信度高的记忆衰减较慢
        confidence_factor = memory_item.confidence * 0.1 + 0.9
        
        # 综合衰减速率：所有因子的乘积
        decay_rate = base_decay * activation_factor * access_factor * confidence_factor
        
        # 确保衰减速率在合理范围内
        return max(0.5, min(0.999, decay_rate))
    
    def clear_memory(self):
        """清空工作记忆"""
        self.central_executive.clear()
        self.phonological_loop.clear()
        self.visuospatial_sketchpad.clear()
        self.episodic_buffer.clear()
        self.memory_index.clear()
        self.association_graph.clear()
        
        logger.info("工作记忆已清空")
    
    def get_memory_stats(self) -> Dict[str, int]:
        """获取工作记忆统计信息"""
        return {
            'total_items': self.stats['total_items'],
            'add_operations': self.stats['add_operations'],
            'retrieve_operations': self.stats['retrieve_operations'],
            'forget_operations': self.stats['forget_operations'],
            'association_count': self.stats['association_count'],
            'phonological_items': len(self.phonological_loop),
            'visuospatial_items': len(self.visuospatial_sketchpad),
            'episodic_items': len(self.episodic_buffer)
        }
    
    def get_associated_memories(self, memory_id: str, limit: int = 5) -> List[MemoryItem]:
        """获取关联的记忆项"""
        if memory_id not in self.association_graph:
            return []
        
        associated = self.association_graph[memory_id]
        
        # 按关联强度排序
        sorted_associated = sorted(associated.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        results = []
        for mem_id, _ in sorted_associated:
            if mem_id in self.memory_index:
                mem_item = self.memory_index[mem_id]
                mem_item.access_count += 1
                results.append(mem_item)
        
        return results

# 测试代码
if __name__ == "__main__":
    print("测试工作记忆系统...")
    
    # 初始化工作记忆
    wm = WorkingMemory()
    
    # 添加测试记忆
    wm.add_memory("水可以溶解糖", "text", confidence=0.9)
    wm.add_memory("盐在水中的溶解度随温度升高而增加", "text", confidence=0.85)
    wm.add_memory("水的化学分子式是H2O", "text", confidence=0.99)
    wm.add_memory("图像: 水分子结构", "image", confidence=0.8)
    wm.add_memory("音频: 水流动的声音", "audio", confidence=0.75)
    
    # 检索记忆
    print("\n检索与'水'相关的记忆:")
    results = wm.retrieve_memory("水", "text")
    for i, result in enumerate(results):
        print(f"{i+1}. {result.content} (置信度: {result.confidence:.2f}, 激活水平: {result.activation_level:.2f})")
    
    # 获取记忆统计
    print("\n工作记忆统计:")
    stats = wm.get_memory_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 测试记忆衰减
    print("\n模拟记忆衰减...")
    wm.decay_factor = 0.5  # 设置更快的衰减
    for i in range(5):
        wm.decay_memories()
        print(f"衰减后记忆数量: {wm.get_memory_stats()['total_items']}")
    
    print("\n测试完成!")