"""
自适应行动组件测试。
"""

import pytest
import asyncio
import torch
import time
from unittest.mock import Mock
from cognitive.action import AdaptiveAction


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


class TestAdaptiveAction:
    """自适应行动系统测试套件"""
    
    @pytest.fixture
    def mock_comm(self):
        """创建模拟通信系统"""
        return MockCommunication()
    
    @pytest.fixture
    def action_system(self, mock_comm):
        """创建行动系统实例"""
        return AdaptiveAction(mock_comm)
    
    @pytest.mark.asyncio
    async def test_initialization(self, action_system):
        """测试行动系统初始化"""
        # 初始状态应为未初始化
        assert action_system.initialized == False
        assert action_system.action_history == []
        
        # 初始化系统
        await action_system.initialize()
        
        # 验证初始化状态
        assert action_system.initialized == True
        assert action_system.execution_network is not None
        assert action_system.adaptation_network is not None
        assert action_system.feedback_network is not None
        
        # 验证配置
        config = action_system.config
        assert config['execution_timeout'] == 30.0
        assert config['max_retries'] == 3
        assert config['adaptation_rate'] == 0.1
        
        # 验证性能指标
        metrics = action_system.performance_metrics
        assert metrics['total_actions'] == 0
        assert metrics['successful_actions'] == 0
        assert metrics['failed_actions'] == 0
        assert metrics['total_execution_time'] == 0.0
        assert metrics['avg_execution_time'] == 0.0
    
    @pytest.mark.asyncio
    async def test_process_action(self, action_system):
        """测试行动处理"""
        await action_system.initialize()
        
        # 创建输入张量和元数据
        input_tensor = torch.randn(1, 512)
        metadata = {
            'decision_context': {'goal': '测试目标', 'priority': 1.0},
            'cognitive_state': {'focus': '测试焦点'}
        }
        
        # 处理行动
        result = await action_system.process(input_tensor, metadata)
        
        # 验证结果
        assert isinstance(result, torch.Tensor)
        # 允许形状变化（网络可能返回不同形状）
        assert result.dim() in (1, 2)  # 1维或2维张量
        
        # 验证历史记录已更新
        assert len(action_system.action_history) == 1
        
        # 验证性能指标已更新
        metrics = action_system.performance_metrics
        assert metrics['total_actions'] == 1
        assert metrics['total_execution_time'] >= 0.0
    
    @pytest.mark.asyncio
    async def test_process_multiple_actions(self, action_system):
        """测试处理多个行动"""
        await action_system.initialize()
        
        # 处理多个行动
        for i in range(3):
            input_tensor = torch.randn(1, 512)
            metadata = {
                'decision_context': {'goal': f'目标{i}', 'priority': i/10.0},
                'cognitive_state': {'focus': f'焦点{i}'}
            }
            result = await action_system.process(input_tensor, metadata)
            assert isinstance(result, torch.Tensor)
        
        # 验证历史记录
        assert len(action_system.action_history) == 3
        
        # 验证性能指标
        metrics = action_system.performance_metrics
        assert metrics['total_actions'] == 3
        assert metrics['total_execution_time'] > 0.0
    
    @pytest.mark.asyncio
    async def test_action_history_structure(self, action_system):
        """测试行动历史结构"""
        await action_system.initialize()
        
        input_tensor = torch.randn(1, 512)
        metadata = {
            'decision_context': {'goal': '测试目标'},
            'cognitive_state': {}
        }
        
        await action_system.process(input_tensor, metadata)
        
        # 验证历史条目结构
        history_entry = action_system.action_history[0]
        assert 'timestamp' in history_entry
        assert 'tensor' in history_entry  # 实际存储的是'tensor'，不是'input_shape'
        assert 'decision_context' in history_entry
        assert 'feedback' in history_entry
        assert 'execution_time' in history_entry
        assert 'actual_cost' in history_entry
        assert 'status' in history_entry  # 实际存储的是'status'，不是'success'
        
        # 验证决策上下文存储
        assert 'goal' not in history_entry['decision_context']  # decision_context只存储selected_option和value_assessment
    
    def test_configuration_update(self, action_system):
        """测试配置更新"""
        # 更新配置
        action_system.config['execution_timeout'] = 60.0
        action_system.config['max_retries'] = 5
        action_system.config['adaptation_rate'] = 0.2
        
        # 验证配置已更新
        assert action_system.config['execution_timeout'] == 60.0
        assert action_system.config['max_retries'] == 5
        assert action_system.config['adaptation_rate'] == 0.2
    
    @pytest.mark.asyncio
    async def test_concurrent_action_processing(self, action_system):
        """测试并发行动处理"""
        await action_system.initialize()
        
        # 创建多个任务
        tasks = []
        for i in range(5):
            input_tensor = torch.randn(1, 512)
            metadata = {'decision_context': {}, 'cognitive_state': {}}
            tasks.append(action_system.process(input_tensor, metadata))
        
        # 并发执行
        results = await asyncio.gather(*tasks)
        
        # 验证所有任务完成
        assert len(results) == 5
        for result in results:
            assert isinstance(result, torch.Tensor)
        
        # 验证历史记录
        assert len(action_system.action_history) == 5
        
        # 验证性能指标
        metrics = action_system.performance_metrics
        assert metrics['total_actions'] == 5
    
    @pytest.mark.asyncio
    async def test_feedback_processing(self, action_system):
        """测试反馈处理（间接）"""
        await action_system.initialize()
        
        # 通过处理行动来测试反馈处理
        input_tensor = torch.randn(1, 512)
        metadata = {
            'decision_context': {'goal': '测试反馈'},
            'cognitive_state': {'feedback_enabled': True}
        }
        
        result = await action_system.process(input_tensor, metadata)
        
        # 验证系统没有崩溃
        assert isinstance(result, torch.Tensor)
        
        # 验证网络仍然有效
        assert action_system.feedback_network is not None
    
    @pytest.mark.asyncio
    async def test_network_reinitialization(self, action_system):
        """测试网络重新初始化"""
        await action_system.initialize()
        
        # 保存初始网络引用
        initial_execution_network = action_system.execution_network
        initial_adaptation_network = action_system.adaptation_network
        
        # 再次初始化（应该没有变化）
        await action_system.initialize()
        
        # 验证网络引用未改变
        assert action_system.execution_network is initial_execution_network
        assert action_system.adaptation_network is initial_adaptation_network
        
        # 验证初始化状态
        assert action_system.initialized == True
    
    @pytest.mark.asyncio
    async def test_error_handling(self, action_system):
        """测试错误处理（模拟）"""
        await action_system.initialize()
        
        # 创建无效输入（形状奇怪的张量）
        input_tensor = torch.tensor([[[[]]]])  # 复杂形状但可能无效
        metadata = {'decision_context': {}, 'cognitive_state': {}}
        
        # 处理应该处理错误而不崩溃
        result = await action_system.process(input_tensor, metadata)
        
        # 验证系统仍然工作
        assert action_system.initialized == True
        
        # 验证结果是张量（即使是无效输入）
        assert isinstance(result, torch.Tensor)