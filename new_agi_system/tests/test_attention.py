"""
层级注意力组件测试。
"""

import pytest
import asyncio
import torch
import time
from unittest.mock import Mock
from cognitive.attention import HierarchicalAttention


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


class TestHierarchicalAttention:
    """层级注意力系统测试套件"""
    
    @pytest.fixture
    def mock_comm(self):
        """创建模拟通信系统"""
        return MockCommunication()
    
    @pytest.fixture
    def attention_system(self, mock_comm):
        """创建注意力系统实例"""
        return HierarchicalAttention(mock_comm)
    
    @pytest.mark.asyncio
    async def test_initialization(self, attention_system):
        """测试注意力系统初始化"""
        # 初始状态应为未初始化
        assert attention_system.initialized == False
        
        # 初始化系统
        await attention_system.initialize()
        
        # 验证初始化状态
        assert attention_system.initialized == True
        assert attention_system.spatial_attention is not None
        assert attention_system.temporal_attention is not None
        assert attention_system.semantic_attention is not None
        
        # 验证配置
        config = attention_system.config
        assert config['attention_heads'] == 8
        assert config['attention_dim'] == 512
        assert config['hierarchy_levels'] == 3
    
    @pytest.mark.asyncio
    async def test_process_attention(self, attention_system):
        """测试注意力处理"""
        await attention_system.initialize()
        
        # 创建输入张量和元数据
        input_tensor = torch.randn(1, 512)
        metadata = {
            'cognitive_state': {
                'attention_weights': {
                    'spatial': 0.5,
                    'temporal': 0.3,
                    'semantic': 0.2
                }
            }
        }
        
        # 处理注意力
        result = await attention_system.process(input_tensor, metadata)
        
        # 验证结果
        assert isinstance(result, torch.Tensor)
        assert result.shape == input_tensor.shape
        
        # 验证系统正常工作
        assert attention_system.initialized == True
    
    @pytest.mark.asyncio
    async def test_process_without_attention_weights(self, attention_system):
        """测试无注意力权重的处理"""
        await attention_system.initialize()
        
        # 创建输入张量，无注意力权重
        input_tensor = torch.randn(1, 512)
        metadata = {
            'cognitive_state': {}
        }
        
        # 处理注意力
        result = await attention_system.process(input_tensor, metadata)
        
        # 验证结果
        assert isinstance(result, torch.Tensor)
        assert result.shape == input_tensor.shape
    
    @pytest.mark.asyncio
    async def test_process_multiple_inputs(self, attention_system):
        """测试处理多个输入"""
        await attention_system.initialize()
        
        # 处理多个输入
        for i in range(3):
            input_tensor = torch.randn(1, 512)
            metadata = {
                'cognitive_state': {
                    'attention_weights': {
                        'spatial': i/10.0,
                        'temporal': (i+1)/10.0,
                        'semantic': (i+2)/10.0
                    }
                }
            }
            result = await attention_system.process(input_tensor, metadata)
            assert isinstance(result, torch.Tensor)
            assert result.shape == input_tensor.shape
    
    @pytest.mark.asyncio
    async def test_process_different_shapes(self, attention_system):
        """测试处理不同形状的输入"""
        await attention_system.initialize()
        
        # 测试不同形状的输入
        shapes = [(1, 512), (2, 512), (1, 256)]
        
        for shape in shapes:
            input_tensor = torch.randn(shape)
            metadata = {
                'cognitive_state': {
                    'attention_weights': {'spatial': 0.5}
                }
            }
            result = await attention_system.process(input_tensor, metadata)
            assert isinstance(result, torch.Tensor)
            assert result.shape == input_tensor.shape
    
    def test_configuration_update(self, attention_system):
        """测试配置更新"""
        # 更新配置
        attention_system.config['attention_heads'] = 16
        attention_system.config['attention_dim'] = 256
        attention_system.config['hierarchy_levels'] = 4
        
        # 验证配置已更新
        assert attention_system.config['attention_heads'] == 16
        assert attention_system.config['attention_dim'] == 256
        assert attention_system.config['hierarchy_levels'] == 4
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, attention_system):
        """测试并发处理"""
        await attention_system.initialize()
        
        # 创建多个任务
        tasks = []
        for i in range(5):
            input_tensor = torch.randn(1, 512)
            metadata = {
                'cognitive_state': {
                    'attention_weights': {'spatial': i/10.0}
                }
            }
            tasks.append(attention_system.process(input_tensor, metadata))
        
        # 并发执行
        results = await asyncio.gather(*tasks)
        
        # 验证所有任务完成
        assert len(results) == 5
        for result in results:
            assert isinstance(result, torch.Tensor)
            assert result.shape == (1, 512)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, attention_system):
        """测试错误处理"""
        await attention_system.initialize()
        
        # 创建无效输入（空张量）
        input_tensor = torch.tensor([])
        metadata = {'cognitive_state': {}}
        
        # 处理应该处理错误而不崩溃
        result = await attention_system.process(input_tensor, metadata)
        
        # 验证系统仍然工作
        assert attention_system.initialized == True
        assert isinstance(result, torch.Tensor)
    
    @pytest.mark.asyncio
    async def test_network_reinitialization(self, attention_system):
        """测试网络重新初始化"""
        await attention_system.initialize()
        
        # 保存初始网络引用
        initial_spatial_attention = attention_system.spatial_attention
        initial_temporal_attention = attention_system.temporal_attention
        
        # 再次初始化（应该没有变化）
        await attention_system.initialize()
        
        # 验证网络引用未改变
        assert attention_system.spatial_attention is initial_spatial_attention
        assert attention_system.temporal_attention is initial_temporal_attention
        
        # 验证初始化状态
        assert attention_system.initialized == True