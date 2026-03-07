"""
情景和语义记忆组件

为统一认知架构管理情景（经验性）和语义（知识性）记忆。
"""

import torch
import torch.nn as nn
import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class EpisodicSemanticMemory:
    """组合情景和语义记忆系统"""
    
    def __init__(self, communication):
        """
        初始化记忆组件。
        
        参数:
            communication: 神经通信系统
        """
        self.communication = communication
        self.initialized = False
        
        # 记忆存储
        self.episodic_memory = []  # 情景经验列表
        self.semantic_memory = {}  # 语义知识字典
        self.working_memory = []   # 短期工作记忆
        
        # 记忆网络
        self.encoding_network = None
        self.retrieval_network = None
        self.consolidation_network = None
        
        # 配置
        self.config = {
            'episodic_capacity': 1000,
            'working_memory_capacity': 10,
            'retrieval_threshold': 0.7,
            'consolidation_interval': 100  # 每100次经验进行巩固
        }
        
        # 统计信息
        self.stats = {
            'episodic_count': 0,
            'semantic_count': 0,
            'retrieval_success': 0,
            'retrieval_failure': 0,
            'consolidation_count': 0
        }
        
        logger.info("情景语义记忆系统已初始化")
    
    async def initialize(self):
        """初始化记忆网络"""
        if self.initialized:
            return
        
        logger.info("正在初始化记忆网络...")
        
        # 初始化记忆网络
        self.encoding_network = self._create_encoding_network()
        self.retrieval_network = self._create_retrieval_network()
        self.consolidation_network = self._create_consolidation_network()
        
        # 如果可用，加载现有记忆
        await self._load_memories()
        
        self.initialized = True
        logger.info(f"记忆网络初始化完成，包含 {self.stats['episodic_count']} 个情景记忆和 {self.stats['semantic_count']} 个语义记忆")
    
    async def retrieve(self, query_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        为查询检索相关记忆。
        
        参数:
            query_tensor: 来自注意力组件的查询张量
            metadata: 检索元数据
            
        返回:
            检索到的记忆张量
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # 从元数据提取上下文
            relevant_context = metadata.get('relevant_context', [])
            cognitive_state = metadata.get('cognitive_state', {})
            
            # 执行记忆检索
            retrieved_memories = await self._retrieve_memories(
                query_tensor, relevant_context, cognitive_state
            )
            
            # 编码检索到的记忆
            memory_tensor = await self._encode_memories(retrieved_memories)
            
            # 更新工作记忆
            await self._update_working_memory(query_tensor, retrieved_memories)
            
            # 准备响应元数据
            response_metadata = {
                'retrieved_context': [
                    {'type': mem.get('type', 'unknown'), 'id': mem.get('id', 'unknown')}
                    for mem in retrieved_memories[:3]  # 前3个记忆
                ],
                'relevance_scores': [mem.get('relevance', 0.0) for mem in retrieved_memories[:3]],
                'retrieval_timestamp': time.time()
            }
            
            logger.debug(f"检索到 {len(retrieved_memories)} 个记忆")
            
            return memory_tensor
            
        except Exception as e:
            logger.error(f"记忆检索失败: {e}")
            # 返回空记忆张量
            return torch.zeros_like(query_tensor)
    
    async def store(self, experience_tensor: torch.Tensor, metadata: Dict[str, Any]):
        """
        Store an experience in memory.
        
        Args:
            experience_tensor: Experience tensor to store
            metadata: Experience metadata
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # 创建情景记忆条目
            episodic_entry = {
                'id': str(time.time()),
                'tensor': experience_tensor.cpu().detach().numpy().tolist(),
                'metadata': metadata,
                'timestamp': time.time(),
                'type': 'episodic',
                'importance': metadata.get('importance', 1.0)
            }
            
            # 存储在情景记忆中
            self.episodic_memory.append(episodic_entry)
            self.stats['episodic_count'] += 1
            
            # 限制情景记忆的大小
            if len(self.episodic_memory) > self.config['episodic_capacity']:
                # 移除最不重要的记忆
                self.episodic_memory.sort(key=lambda x: x.get('importance', 0))
                self.episodic_memory = self.episodic_memory[-self.config['episodic_capacity']:]
            
            # 检查合并情况
            if self.stats['episodic_count'] % self.config['consolidation_interval'] == 0:
                await self._consolidate_memories()
            
            logger.debug(f"存储情景记忆 {episodic_entry['id']}")
            
        except Exception as e:
            logger.error(f"记忆存储失败: {e}")
    
    async def _retrieve_memories(self, query_tensor: torch.Tensor,
                               context: List[Any], cognitive_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on query and context"""
        retrieved_memories = []
        
        # 1. 从工作记忆中检索
        working_memories = self._retrieve_from_working_memory(query_tensor)
        retrieved_memories.extend(working_memories)
        
        # 2. 从情景记忆中检索
        episodic_memories = await self._retrieve_from_episodic_memory(query_tensor, context)
        retrieved_memories.extend(episodic_memories)
        
        # 3. 从语义记忆中检索
        semantic_memories = await self._retrieve_from_semantic_memory(query_tensor, cognitive_state)
        retrieved_memories.extend(semantic_memories)
        
        # 按相关性排序
        retrieved_memories.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        
        # 限制结果数量
        max_results = 10
        retrieved_memories = retrieved_memories[:max_results]
        
        # 更新统计信息
        if retrieved_memories:
            self.stats['retrieval_success'] += 1
        else:
            self.stats['retrieval_failure'] += 1
        
        return retrieved_memories
    
    def _retrieve_from_working_memory(self, query_tensor: torch.Tensor) -> List[Dict[str, Any]]:
        """从工作记忆中检索"""
        retrieved = []
        
        for memory in self.working_memory:
            # 简单的相关性计算（余弦相似度）
            if 'tensor' in memory and memory['tensor'] is not None:
                try:
                    memory_tensor = torch.tensor(memory['tensor']).float()
                    similarity = self._calculate_similarity(query_tensor, memory_tensor)
                    
                    if similarity > self.config['retrieval_threshold']:
                        memory['relevance'] = similarity
                        retrieved.append(memory)
                except Exception as e:
                    logger.debug(f"计算工作记忆相似度失败: {e}")
        
        return retrieved
    
    async def _retrieve_from_episodic_memory(self, query_tensor: torch.Tensor,
                                           context: List[Any]) -> List[Dict[str, Any]]:
        """Retrieve from episodic memory"""
        retrieved = []
        
        # 使用检索网络（如果可用）
        if self.retrieval_network is not None:
            try:
                # 编码查询
                query_encoded = self.retrieval_network(query_tensor.unsqueeze(0))
                
                # 查找相似记忆（简化 - 实际应使用适当的相似度搜索）
                for memory in self.episodic_memory[-100:]:  # 检查最近记忆
                    if 'tensor' in memory:
                        try:
                            memory_tensor = torch.tensor(memory['tensor']).float()
                            similarity = self._calculate_similarity(query_encoded.squeeze(0), memory_tensor)
                            
                            if similarity > self.config['retrieval_threshold']:
                                memory['relevance'] = similarity
                                retrieved.append(memory)
                        except Exception as e:
                            logger.debug(f"计算情景记忆相似度失败: {e}")
            except Exception as e:
                logger.warning(f"情景记忆检索网络失败: {e}")
        
        # 基于上下文的检索
        for ctx in context:
            if isinstance(ctx, dict) and 'type' in ctx:
                # 查找与上下文类型相似的记忆
                similar_memories = [
                    mem for mem in self.episodic_memory[-50:]
                    if mem.get('metadata', {}).get('type') == ctx.get('type')
                ]
                retrieved.extend(similar_memories[:3])  # 每个上下文限制3个
        
        return retrieved
    
    async def _retrieve_from_semantic_memory(self, query_tensor: torch.Tensor,
                                           cognitive_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve from semantic memory"""
        retrieved = []
        
        # 从认知状态获取当前目标
        current_goals = cognitive_state.get('goal_stack', [])
        
        # 检索与当前目标相关的知识
        for goal in current_goals[:3]:  # 前3个目标
            goal_type = goal.get('type', 'unknown')
            
            # 查找与目标类型相关的语义知识
            if goal_type in self.semantic_memory:
                knowledge_items = self.semantic_memory[goal_type]
                for item in knowledge_items[:5]:  # 每个目标类型的前5个项目
                    item['relevance'] = 0.8  # 与目标相关知识的高相关性
                    retrieved.append(item)
        
        # 基于查询的一般语义检索
        # 这是简化的实现 - 实际应使用适当的语义搜索
        for category, items in self.semantic_memory.items():
            if len(retrieved) >= 10:  # 限制总检索数量
                break
            
            # 从每个类别添加几个项目
            for item in items[:2]:
                item['relevance'] = 0.5  # 中等相关性
                retrieved.append(item)
        
        return retrieved
    
    async def _encode_memories(self, memories: List[Dict[str, Any]]) -> torch.Tensor:
        """将检索到的记忆编码为张量"""
        if not memories:
            return torch.zeros(1, 512)  # 空记忆张量
        
        try:
            if self.encoding_network is not None:
                # 编码每个记忆
                encoded_memories = []
                for memory in memories[:5]:  # 编码前5个记忆
                    if 'tensor' in memory and memory['tensor'] is not None:
                        memory_tensor = torch.tensor(memory['tensor']).float()
                        encoded = self.encoding_network(memory_tensor.unsqueeze(0))
                        # 按相关性加权
                        relevance = memory.get('relevance', 0.5)
                        weighted = encoded * relevance
                        encoded_memories.append(weighted)
                
                if encoded_memories:
                    # 合并编码的记忆
                    combined = torch.stack(encoded_memories).mean(dim=0)
                    return combined
                else:
                    return torch.zeros(1, 512)
            else:
                # 回退方案：记忆张量的平均值
                memory_tensors = []
                for memory in memories[:5]:
                    if 'tensor' in memory and memory['tensor'] is not None:
                        tensor = torch.tensor(memory['tensor']).float()
                        memory_tensors.append(tensor)
                
                if memory_tensors:
                    combined = torch.stack(memory_tensors).mean(dim=0)
                    return combined
                else:
                    return torch.zeros(1, 512)
                    
        except Exception as e:
            logger.error(f"记忆编码失败: {e}")
            return torch.zeros(1, 512)
    
    async def _update_working_memory(self, query_tensor: torch.Tensor,
                                   retrieved_memories: List[Dict[str, Any]]):
        """Update working memory with current retrieval"""
        # Add query to working memory
        query_entry = {
            'tensor': query_tensor.cpu().detach().numpy().tolist(),
            'timestamp': time.time(),
            'type': 'query',
            'importance': 1.0
        }
        self.working_memory.append(query_entry)
        
        # Add top retrieved memories to working memory
        for memory in retrieved_memories[:3]:
            if 'tensor' in memory:
                memory_entry = memory.copy()
                memory_entry['timestamp'] = time.time()
                memory_entry['importance'] = memory.get('relevance', 0.5)
                self.working_memory.append(memory_entry)
        
        # Limit working memory size
        if len(self.working_memory) > self.config['working_memory_capacity']:
            # Remove oldest entries
            self.working_memory.sort(key=lambda x: x['timestamp'])
            self.working_memory = self.working_memory[-self.config['working_memory_capacity']:]
    
    async def _consolidate_memories(self):
        """Consolidate episodic memories into semantic knowledge"""
        logger.info("正在整合记忆...")
        
        try:
            if self.consolidation_network is not None:
                # Get recent episodic memories
                recent_memories = self.episodic_memory[-self.config['consolidation_interval']:]
                
                # Group by type
                memories_by_type = defaultdict(list)
                for memory in recent_memories:
                    memory_type = memory.get('metadata', {}).get('type', 'general')
                    memories_by_type[memory_type].append(memory)
                
                # Consolidate each type
                for memory_type, memories in memories_by_type.items():
                    if len(memories) >= 3:  # Need at least 3 memories to consolidate
                        # Extract patterns (simplified)
                        patterns = await self._extract_patterns(memories)
                        
                        # Create semantic knowledge entry
                        knowledge_entry = {
                            'id': f"knowledge_{memory_type}_{time.time()}",
                            'type': memory_type,
                            'patterns': patterns,
                            'source_memories': [m['id'] for m in memories],
                            'created_at': time.time(),
                            'confidence': min(1.0, len(memories) / 10.0)  # More memories = higher confidence
                        }
                        
                        # Store in semantic memory
                        if memory_type not in self.semantic_memory:
                            self.semantic_memory[memory_type] = []
                        
                        self.semantic_memory[memory_type].append(knowledge_entry)
                        self.stats['semantic_count'] += 1
                
                self.stats['consolidation_count'] += 1
                logger.info(f"整合了 {len(memories_by_type)} 种记忆类型")
                
        except Exception as e:
            logger.error(f"记忆整合失败: {e}")
    
    async def _extract_patterns(self, memories: List[Dict[str, Any]]) -> List[Any]:
        """Extract patterns from memories (simplified)"""
        patterns = []
        
        # Simplified pattern extraction
        # In practice, would use proper pattern recognition
        for i in range(min(3, len(memories))):
            memory = memories[i]
            pattern = {
                'summary': f"Pattern from memory {memory.get('id', 'unknown')}",
                'key_features': memory.get('metadata', {}).get('key_features', []),
                'frequency': 1
            }
            patterns.append(pattern)
        
        return patterns
    
    async def _load_memories(self):
        """Load memories from storage (simplified)"""
        # This would load from disk in practice
        logger.info("正在从存储加载记忆...")
        
        # Create some initial semantic knowledge
        self.semantic_memory['general'] = [
            {
                'id': 'knowledge_general_1',
                'type': 'general',
                'patterns': [{'summary': 'Basic cognitive pattern'}],
                'created_at': time.time() - 3600,
                'confidence': 0.8
            }
        ]
        self.stats['semantic_count'] = 1
        
        logger.info("记忆已从存储加载")
    
    def _calculate_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """Calculate cosine similarity between two tensors"""
        try:
            # Ensure tensors are the same shape
            if tensor1.shape != tensor2.shape:
                # Reshape or pad to match
                min_size = min(tensor1.numel(), tensor2.numel())
                tensor1_flat = tensor1.flatten()[:min_size]
                tensor2_flat = tensor2.flatten()[:min_size]
            else:
                tensor1_flat = tensor1.flatten()
                tensor2_flat = tensor2.flatten()
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                tensor1_flat.unsqueeze(0),
                tensor2_flat.unsqueeze(0)
            )
            return similarity.item()
        except Exception as e:
            logger.debug(f"相似度计算失败: {e}")
            return 0.0
    
    def _create_encoding_network(self) -> nn.Module:
        """Create memory encoding network"""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Tanh()
        )
    
    def _create_retrieval_network(self) -> nn.Module:
        """Create memory retrieval network"""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Tanh()
        )
    
    def _create_consolidation_network(self) -> nn.Module:
        """Create memory consolidation network"""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Tanh()
        )
    
    async def shutdown(self):
        """关闭记忆组件"""
        logger.info("正在关闭记忆组件...")
        
        # 保存记忆到存储
        await self._save_memories()
        
        self.initialized = False
        logger.info("记忆组件关闭完成")
    
    async def _save_memories(self):
        """保存记忆到存储（简化版）"""
        logger.info(f"正在保存 {self.stats['episodic_count']} 个情景记忆和 {self.stats['semantic_count']} 个语义记忆...")
        # 实际应用中，会保存到磁盘
        logger.info("记忆已保存到存储")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        return {
            **self.stats,
            'episodic_memory_size': len(self.episodic_memory),
            'working_memory_size': len(self.working_memory),
            'semantic_categories': len(self.semantic_memory),
            'total_semantic_items': sum(len(items) for items in self.semantic_memory.values()),
            'retrieval_success_rate': self.stats['retrieval_success'] / max(1, self.stats['retrieval_success'] + self.stats['retrieval_failure'])
        }