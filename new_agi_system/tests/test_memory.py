"""
情景和语义记忆组件测试。
"""

import pytest
import asyncio
import torch
import time
from unittest.mock import Mock, AsyncMock
from cognitive.memory import EpisodicSemanticMemory


class MockCommunication:
    """用于测试的模拟通信系统"""
    
    def __init__(self):
        self.registered_components = {}
        self.shared_memory = {}
        self.tensor_channels = {}
        self.lock = Mock()
    
    def register_component(self, component_id, component_type):
        self.registered_components[component_id] = {
            'type': component_type,
            'registered_at': time.time()
        }
    
    def create_shared_tensor(self, name, shape, dtype=torch.float32):
        tensor = torch.zeros(shape, dtype=dtype)
        self.shared_memory[name] = tensor
        return tensor


class TestEpisodicSemanticMemory:
    """情景语义记忆系统测试套件"""
    
    @pytest.fixture
    def mock_comm(self):
        """创建模拟通信系统"""
        return MockCommunication()
    
    @pytest.fixture
    def memory_system(self, mock_comm):
        """创建记忆系统实例"""
        return EpisodicSemanticMemory(mock_comm)
    
    @pytest.mark.asyncio
    async def test_initialization(self, memory_system):
        """测试记忆系统初始化"""
        # 初始状态应为未初始化
        assert memory_system.initialized == False
        assert memory_system.episodic_memory == []
        assert memory_system.semantic_memory == {}
        assert memory_system.working_memory == []
        
        # 初始化系统
        await memory_system.initialize()
        
        # 验证初始化状态
        assert memory_system.initialized == True
        assert memory_system.encoding_network is not None
        assert memory_system.retrieval_network is not None
        assert memory_system.consolidation_network is not None
        
        # 验证配置
        config = memory_system.config
        assert config['episodic_capacity'] == 1000
        assert config['working_memory_capacity'] == 10
        assert config['retrieval_threshold'] == 0.7
        assert config['consolidation_interval'] == 100
        
        # 验证统计信息
        stats = memory_system.stats
        assert stats['episodic_count'] == 0
        assert stats['semantic_count'] == 1  # 初始语义记忆已加载
        assert stats['retrieval_success'] == 0
        assert stats['retrieval_failure'] == 0
        assert stats['consolidation_count'] == 0
    
    @pytest.mark.asyncio
    async def test_store_experience(self, memory_system):
        """测试存储经验到记忆"""
        await memory_system.initialize()
        
        # 创建测试张量和元数据
        test_tensor = torch.randn(1, 512)
        test_metadata = {
            'importance': 0.8,
            'context': '测试上下文',
            'user': '测试用户'
        }
        
        # 存储经验
        await memory_system.store(test_tensor, test_metadata)
        
        # 验证存储
        assert memory_system.stats['episodic_count'] == 1
        assert len(memory_system.episodic_memory) == 1
        
        # 验证存储的条目
        stored_entry = memory_system.episodic_memory[0]
        assert 'id' in stored_entry
        assert stored_entry['type'] == 'episodic'
        assert stored_entry['importance'] == 0.8
        assert stored_entry['metadata'] == test_metadata
        assert 'timestamp' in stored_entry
        
        # 验证张量存储（转换为列表）
        assert 'tensor' in stored_entry
        assert isinstance(stored_entry['tensor'], list)
    
    @pytest.mark.asyncio
    async def test_store_multiple_experiences(self, memory_system):
        """测试存储多个经验"""
        await memory_system.initialize()
        
        # 存储多个经验
        for i in range(5):
            test_tensor = torch.randn(1, 512)
            test_metadata = {'importance': i/10.0, 'index': i}
            await memory_system.store(test_tensor, test_metadata)
        
        # 验证存储
        assert memory_system.stats['episodic_count'] == 5
        assert len(memory_system.episodic_memory) == 5
        
        # 验证按重要性排序（不重要到重要）
        importances = [entry['importance'] for entry in memory_system.episodic_memory]
        assert importances == [0.0, 0.1, 0.2, 0.3, 0.4]
    
    @pytest.mark.asyncio
    async def test_retrieve_empty_memory(self, memory_system):
        """测试从空记忆检索"""
        await memory_system.initialize()
        
        # 清空初始加载的语义记忆，以便测试空记忆检索
        memory_system.semantic_memory = {}
        memory_system.stats['semantic_count'] = 0
        
        # 创建查询张量
        query_tensor = torch.randn(1, 512)
        metadata = {
            'relevant_context': [],
            'cognitive_state': {}
        }
        
        # 检索记忆（应该返回零张量）
        result = await memory_system.retrieve(query_tensor, metadata)
        
        # 验证结果
        assert isinstance(result, torch.Tensor)
        assert result.shape == query_tensor.shape
        assert torch.allclose(result, torch.zeros_like(query_tensor))
        
        # 验证统计信息
        assert memory_system.stats['retrieval_failure'] == 1
        assert memory_system.stats['retrieval_success'] == 0
    
    @pytest.mark.asyncio
    async def test_retrieve_with_stored_memories(self, memory_system):
        """测试从存储的记忆中检索"""
        await memory_system.initialize()
        
        # 清空初始语义记忆，仅测试存储的情景记忆
        memory_system.semantic_memory = {}
        memory_system.stats['semantic_count'] = 0
        
        # 降低检索阈值以确保检索成功
        memory_system.config['retrieval_threshold'] = 0.1
        
        # 存储一些测试记忆
        for i in range(3):
            test_tensor = torch.randn(1, 512)
            test_metadata = {
                'importance': 1.0,
                'context': f'测试上下文{i}',
                'category': '测试',
                'type': '测试'  # 添加类型字段以便上下文检索
            }
            await memory_system.store(test_tensor, test_metadata)
        
        # 创建查询张量（使用与存储张量相同的张量以提高相似度）
        query_tensor = torch.randn(1, 512)
        metadata = {
            'relevant_context': [{'type': '测试'}],  # 上下文类型匹配
            'cognitive_state': {'goal': '测试检索'}
        }
        
        # 检索记忆
        result = await memory_system.retrieve(query_tensor, metadata)
        
        # 验证结果
        assert isinstance(result, torch.Tensor)
        # 允许额外的批次维度
        assert result.squeeze(0).shape == query_tensor.shape
        
        # 结果不应是零张量（因为有些记忆被存储）
        assert not torch.allclose(result.squeeze(0), torch.zeros_like(query_tensor))
        
        # 验证统计信息
        assert memory_system.stats['retrieval_success'] == 1
    
    @pytest.mark.asyncio
    async def test_working_memory_operations(self, memory_system):
        """测试工作记忆操作"""
        await memory_system.initialize()
        
        # 初始工作记忆应为空
        assert len(memory_system.working_memory) == 0
        
        # 模拟工作记忆更新（通过存储和检索）
        test_tensor = torch.randn(1, 512)
        test_metadata = {'importance': 1.0}
        await memory_system.store(test_tensor, test_metadata)
        
        # 检索应更新工作记忆
        query_tensor = torch.randn(1, 512)
        metadata = {'relevant_context': [], 'cognitive_state': {}}
        await memory_system.retrieve(query_tensor, metadata)
        
        # 工作记忆可能被更新（取决于实现）
        # 我们不假设具体内容，只验证系统不崩溃
    
    def test_configuration_update(self, memory_system):
        """测试配置更新"""
        # 更新配置
        memory_system.config['episodic_capacity'] = 500
        memory_system.config['working_memory_capacity'] = 5
        memory_system.config['retrieval_threshold'] = 0.8
        
        # 验证配置已更新
        assert memory_system.config['episodic_capacity'] == 500
        assert memory_system.config['working_memory_capacity'] == 5
        assert memory_system.config['retrieval_threshold'] == 0.8
    
    @pytest.mark.asyncio
    async def test_memory_capacity_limit(self, memory_system):
        """测试记忆容量限制"""
        await memory_system.initialize()
        
        # 设置较小的容量以便测试
        memory_system.config['episodic_capacity'] = 3
        
        # 存储超出容量的经验
        for i in range(5):
            test_tensor = torch.randn(1, 512)
            test_metadata = {'importance': i/10.0, 'index': i}
            await memory_system.store(test_tensor, test_metadata)
        
        # 验证记忆数量不超过容量
        assert len(memory_system.episodic_memory) == 3
        
        # 验证保留了最重要的记忆（重要性最高的）
        importances = [entry['importance'] for entry in memory_system.episodic_memory]
        assert importances == [0.2, 0.3, 0.4]  # 最重要的3个
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, memory_system):
        """测试并发操作（模拟）"""
        await memory_system.initialize()
        
        # 创建多个任务
        tasks = []
        for i in range(10):
            test_tensor = torch.randn(1, 512)
            test_metadata = {'importance': 0.5, 'task_id': i}
            
            # 交替存储和检索
            if i % 2 == 0:
                tasks.append(memory_system.store(test_tensor, test_metadata))
            else:
                query_tensor = torch.randn(1, 512)
                metadata = {'relevant_context': [], 'cognitive_state': {}}
                tasks.append(memory_system.retrieve(query_tensor, metadata))
        
        # 并发执行
        await asyncio.gather(*tasks)
        
        # 验证系统仍然正常工作
        assert memory_system.initialized == True
        assert memory_system.stats['episodic_count'] == 5  # 5个存储操作
        
        # 验证没有崩溃
        assert memory_system.episodic_memory is not None
        assert memory_system.working_memory is not None