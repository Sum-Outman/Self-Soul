#!/usr/bin/env python3
"""
数据集生成基础模块
包含通用的确定性随机数生成器和基础数据生成函数
"""

import hashlib
from typing import Any, List, Sequence, Optional
import time

class DeterministicRandom:
    """
    确定性随机数生成器
    提供可重复的随机数生成，用于测试和可重复的数据集生成
    """
    
    def __init__(self, seed: int = 42):
        """
        初始化确定性随机数生成器
        
        Args:
            seed: 随机种子，默认42
        """
        self.seed = seed
        self.counter = 0
    
    def _get_hash(self, *args) -> int:
        """
        生成确定性哈希值
        
        Args:
            *args: 任意参数
            
        Returns:
            哈希值整数
        """
        self.counter += 1
        input_str = str(args) + str(self.seed) + str(self.counter)
        return int(hashlib.sha256(input_str.encode()).hexdigest(), 16)
    
    def choice(self, seq: Sequence) -> Any:
        """
        从序列中随机选择一个元素
        
        Args:
            seq: 输入序列
            
        Returns:
            随机选择的元素
            
        Raises:
            ValueError: 如果序列为空
        """
        if not seq:
            raise ValueError("Cannot choose from empty sequence")
        return seq[self._get_hash('choice', tuple(seq)) % len(seq)]
    
    def randint(self, a: int, b: int) -> int:
        """
        生成区间[a, b]内的随机整数
        
        Args:
            a: 最小值
            b: 最大值
            
        Returns:
            随机整数
        """
        return a + self._get_hash('randint', a, b) % (b - a + 1)
    
    def uniform(self, a: float, b: float) -> float:
        """
        生成区间[a, b]内的随机浮点数
        
        Args:
            a: 最小值
            b: 最大值
            
        Returns:
            随机浮点数
        """
        return a + (self._get_hash('uniform', a, b) % 10000) / 10000.0 * (b - a)
    
    def sample(self, population: Sequence, k: int) -> List:
        """
        从总体中随机抽取k个不重复的元素
        
        Args:
            population: 总体序列
            k: 抽取数量
            
        Returns:
            抽取的元素列表
        """
        # 简化实现：按确定性哈希排序后取前k个
        indices = sorted(range(len(population)), key=lambda i: self._get_hash('sample', i))
        return [population[i] for i in indices[:k]]
    
    def random(self) -> float:
        """
        生成[0.0, 1.0)区间的随机浮点数
        
        Returns:
            随机浮点数
        """
        return self.uniform(0.0, 1.0)
    
    def shuffle(self, x: List) -> None:
        """
        原地打乱列表（确定性）
        
        Args:
            x: 要打乱的列表
        """
        # 使用确定性哈希进行打乱
        n = len(x)
        for i in range(n - 1, 0, -1):
            j = self.randint(0, i)
            x[i], x[j] = x[j], x[i]

def create_deterministic_random(seed: int = 42) -> DeterministicRandom:
    """
    创建确定性随机数生成器实例
    
    Args:
        seed: 随机种子
        
    Returns:
        DeterministicRandom实例
    """
    return DeterministicRandom(seed)

# 预定义常量
DEFAULT_SEED = 42
STANDARD_DATASET_SIZE = 5000
SUPER_LARGE_DATASET_SIZE = 50000
MIN_DIALOGUES = 20000
MIN_SUPER_DIALOGUES = 100000
DIALOGUE_VARIANTS = 30

# 模型类型定义
MODEL_TYPES = [
    "manager", "language", "audio", "vision_image", "vision_video",
    "spatial", "sensor", "computer", "motion", "knowledge",
    "programming", "planning", "autonomous", "emotion", "prediction",
    "collaboration", "optimization", "finance", "medical", "value_alignment",
    "stereo_vision"
]

# 知识库领域定义
KNOWLEDGE_DOMAINS = [
    "computer_science",
    "mathematics", 
    "physics",
    "chemistry",
    "biology",
    "medicine",
    "engineering",
    "economics",
    "psychology",
    "sociology",
    "philosophy",
    "history",
    "literature",
    "art",
    "music",
    "geography",
    "environmental_science",
    "political_science",
    "law",
    "education"
]

def get_dataset_size(dataset_type: str) -> int:
    """
    根据数据集类型获取数据规模
    
    Args:
        dataset_type: 数据集类型 ('standard' 或 'super_large')
        
    Returns:
        数据规模
        
    Raises:
        ValueError: 如果数据集类型无效
    """
    if dataset_type == 'standard':
        return STANDARD_DATASET_SIZE
    elif dataset_type == 'super_large':
        return SUPER_LARGE_DATASET_SIZE
    else:
        raise ValueError(f"无效的数据集类型: {dataset_type}")

def get_dataset_directory(dataset_type: str) -> str:
    """
    根据数据集类型获取输出目录
    
    Args:
        dataset_type: 数据集类型 ('standard' 或 'super_large')
        
    Returns:
        输出目录路径
    """
    if dataset_type == 'standard':
        return "data/datasets"
    else:  # super_large
        return "training_data_super_large"

def generate_dialogue_variants(base_dialogues: List[str], num_variants: int, 
                              random_gen: DeterministicRandom) -> List[str]:
    """
    生成对话变体
    
    Args:
        base_dialogues: 基础对话列表
        num_variants: 每个对话的变体数量
        random_gen: 随机数生成器
        
    Returns:
        对话变体列表
    """
    variants = []
    for dialogue in base_dialogues:
        for _ in range(num_variants):
            variants.append(dialogue)
    return variants

def ensure_minimum_data(data: List, min_size: int, 
                       fallback_generator: callable,
                       random_gen: DeterministicRandom) -> List:
    """
    确保数据达到最小规模
    
    Args:
        data: 现有数据列表
        min_size: 最小数据规模
        fallback_generator: 回退生成函数
        random_gen: 随机数生成器
        
    Returns:
        达到最小规模的数据列表
    """
    while len(data) < min_size:
        data.append(fallback_generator(random_gen))
    return data

def safe_random_choice(seq: Sequence, random_gen: DeterministicRandom, 
                      default: Optional[Any] = None) -> Any:
    """
    安全的随机选择，处理空序列
    
    Args:
        seq: 输入序列
        random_gen: 随机数生成器
        default: 序列为空时的默认值
        
    Returns:
        选择的元素或默认值
        
    Raises:
        ValueError: 如果序列为空且未提供默认值
    """
    if not seq:
        if default is not None:
            return default
        raise ValueError("Sequence must not be empty")
    return random_gen.choice(seq)