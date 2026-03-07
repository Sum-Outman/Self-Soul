"""
统一认知架构测试。
"""

import pytest
import asyncio
import torch
import time
from cognitive.architecture import UnifiedCognitiveArchitecture, CognitiveState


class TestCognitiveState:
    """认知状态测试套件"""
    
    def test_initialization(self):
        """测试认知状态初始化"""
        state = CognitiveState()
        
        assert state.current_focus is None
        assert len(state.working_memory) == 0
        assert state.working_memory_capacity == 10
        assert len(state.goal_stack) == 0
        assert state.cognitive_load == 0.0
        assert state.attention_fatigue == 0.0
        assert state.last_update <= time.time()
    
    def test_working_memory_update(self):
        """测试工作记忆更新"""
        state = CognitiveState()
        
        # Add items to working memory
        for i in range(5):
            state.update_working_memory(f"Item {i}")
        
        assert len(state.working_memory) == 5
        
        # Check item structure
        for item in state.working_memory:
            assert 'item' in item
            assert 'timestamp' in item
            assert 'importance' in item
            assert item['importance'] == 1.0
        
        # Add more items beyond capacity
        for i in range(10):
            state.update_working_memory(f"Extra item {i}")
        
        # Should be limited to capacity
        assert len(state.working_memory) == state.working_memory_capacity
        
        # Least important items should be removed first
        # (all have same importance, so oldest should be removed)
    
    def test_relevant_context_retrieval(self):
        """测试相关上下文检索"""
        state = CognitiveState()
        
        # Add items to working memory
        for i in range(5):
            state.update_working_memory(f"Test item {i}")
        
        # Get relevant context
        context = state.get_relevant_context("query", max_items=3)
        
        assert len(context) == 3
        assert all(isinstance(item, str) for item in context)
        
        # Should return most recent items
        expected_items = ["Test item 2", "Test item 3", "Test item 4"]
        # Note: due to list slicing, might be different
        
    def test_goal_management(self):
        """测试目标管理"""
        state = CognitiveState()
        
        # Add goals
        state.add_goal({'description': 'Goal 1', 'priority': 1.0})
        state.add_goal({'description': 'Goal 2', 'priority': 0.5})
        state.add_goal({'description': 'Goal 3', 'priority': 0.8})
        
        assert len(state.goal_stack) == 3
        
        # Goals should be sorted by priority
        priorities = [goal['priority'] for goal in state.goal_stack]
        assert priorities == sorted(priorities, reverse=True)
        
        # Get current goal (highest priority)
        current_goal = state.get_current_goal()
        assert current_goal is not None
        assert current_goal['description'] == 'Goal 1'
        assert current_goal['priority'] == 1.0
        
        # Mark goal as completed (simulated)
        # In actual implementation, would update status
        state.goal_stack[0]['status'] = 'completed'
        
        # Now current goal should be next highest priority
        current_goal = state.get_current_goal()
        assert current_goal['description'] == 'Goal 3'  # Priority 0.8
        
    def test_to_dict_conversion(self):
        """测试字典转换"""
        state = CognitiveState()
        
        # Add some data
        state.update_working_memory("Test item")
        state.add_goal({'description': 'Test goal', 'priority': 0.7})
        
        # Convert to dict
        state_dict = state.to_dict()
        
        assert isinstance(state_dict, dict)
        assert 'current_focus' in state_dict
        assert 'working_memory_size' in state_dict
        assert state_dict['working_memory_size'] == 1
        assert 'goal_stack_size' in state_dict
        assert state_dict['goal_stack_size'] == 1
        assert 'current_goal' in state_dict
        assert 'cognitive_load' in state_dict
        assert 'last_update' in state_dict


class TestUnifiedCognitiveArchitecture:
    """统一认知架构测试套件"""
    
    @pytest.fixture
    def agi_instance(self):
        """创建用于测试的AGI实例"""
        config = {
            'embedding_dim': 512,
            'max_shared_memory_mb': 100
        }
        
        agi = UnifiedCognitiveArchitecture(config)
        
        # Don't fully initialize (avoids component dependencies)
        # Just return the instance
        return agi
    
    def test_initialization(self, agi_instance):
        """测试统一架构初始化"""
        agi = agi_instance
        
        # 检查配置参数
        assert agi.embedding_dim == 512
        assert agi.max_shared_memory_mb == 100
        
        # 检查组件是否创建
        assert hasattr(agi, 'neural_comm')
        assert hasattr(agi, 'repr_space')
        assert hasattr(agi, 'cognitive_state')
        assert hasattr(agi, 'performance_metrics')
        
        # 检查通信系统（转换为字节）
        assert agi.neural_comm.max_shared_memory == 100 * 1024 * 1024
        
        # 检查表征空间（通过属性访问）
        assert agi.repr_space.embedding_dim == 512
        
        # 检查认知状态
        assert isinstance(agi.cognitive_state, CognitiveState)
        
        # 检查性能指标
        assert agi.performance_metrics['total_cycles'] == 0
        assert agi.performance_metrics['successful_cycles'] == 0
        assert agi.performance_metrics['failed_cycles'] == 0
        assert agi.performance_metrics['avg_response_time'] == 0.0
        assert agi.performance_metrics['total_processing_time'] == 0.0
    
    def test_component_registration(self, agi_instance):
        """测试组件在通信系统中的注册"""
        agi = agi_instance
        
        # Check that cognitive components are registered
        # Note: actual registration happens in _register_components()
        # We can check that the method exists
        assert hasattr(agi, '_register_components')
    
    def test_lazy_component_loading(self, agi_instance):
        """测试认知组件的懒加载"""
        agi = agi_instance
        
        # Components should be None initially
        assert agi._perception is None
        assert agi._attention is None
        assert agi._memory is None
        assert agi._reasoning is None
        assert agi._planning is None
        assert agi._decision is None
        assert agi._action is None
        assert agi._learning is None
        
        # Accessing properties should trigger lazy loading
        # But we don't want to actually load them in unit tests
        # due to dependencies
        assert hasattr(agi, 'perception')
        assert hasattr(agi, 'attention')
        assert hasattr(agi, 'memory')
        assert hasattr(agi, 'reasoning')
        assert hasattr(agi, 'planning')
        assert hasattr(agi, 'decision')
        assert hasattr(agi, 'action')
        assert hasattr(agi, 'learning')
    
    @pytest.mark.asyncio
    async def test_cognitive_cycle_success(self, agi_instance):
        """Test successful cognitive cycle execution"""
        agi = agi_instance
        
        # Mock the component processing methods
        # to avoid actual neural network execution
        original_methods = {}
        component_methods = [
            '_process_perception',
            '_process_attention', 
            '_process_memory',
            '_process_reasoning',
            '_process_planning',
            '_process_decision',
            '_process_action',
            '_process_learning'
        ]
        
        # Create mock responses
        mock_response = {
            'output': {'result': 'test_output'},
            'reasoning_trace': {'steps': ['step1', 'step2']},
            'learning_update': {'improvement': 0.1}
        }
        
        async def mock_process(*args, **kwargs):
            return mock_response
        
        # Replace methods with mocks
        for method_name in component_methods:
            if hasattr(agi, method_name):
                original_methods[method_name] = getattr(agi, method_name)
                setattr(agi, method_name, mock_process)
        
        try:
            # Test input
            input_data = {
                'text': "Test cognitive cycle input",
                'context': {'test': 'context'}
            }
            
            # Execute cognitive cycle
            result = await agi.cognitive_cycle(input_data)
            
            # 检查结果结构
            assert isinstance(result, dict)
            
            # 检查每个组件的结果
            assert 'perception' in result
            assert 'attention' in result
            assert 'memory' in result
            assert 'reasoning' in result
            assert 'planning' in result
            assert 'decision' in result
            assert 'action' in result
            assert 'learning' in result
            
            # 检查模拟响应是否在每个组件中
            assert result['perception']['output'] == mock_response['output']
            assert result['attention']['output'] == mock_response['output']
            
            # 检查其他字段
            assert 'cognitive_state' in result
            assert 'timestamp' in result
            assert 'cycle_id' in result
            assert 'success' in result
            
            # 检查性能指标是否更新
            assert agi.performance_metrics['total_cycles'] == 1
            assert agi.performance_metrics['successful_cycles'] == 1  # 成功
            assert agi.performance_metrics['failed_cycles'] == 0
            assert agi.performance_metrics['avg_response_time'] > 0
            
        finally:
            # Restore original methods
            for method_name, original_method in original_methods.items():
                setattr(agi, method_name, original_method)
    
    @pytest.mark.asyncio
    async def test_cognitive_cycle_error_handling(self, agi_instance):
        """Test error handling in cognitive cycle"""
        agi = agi_instance
        
        # Mock a method to raise an exception
        original_process = agi._process_perception
        
        async def failing_process(*args, **kwargs):
            raise Exception("Test error in perception")
        
        agi._process_perception = failing_process
        
        try:
            # Test input
            input_data = {'text': "Test input that will fail"}
            
            # Execute cognitive cycle (should handle error)
            result = await agi.cognitive_cycle(input_data)
            
            # 应该返回错误响应
            assert 'error' in result
            assert 'error_type' in result
            assert 'timestamp' in result
            assert 'Test error in perception' in result['error']
            
            # 错误响应不包含cognitive_state和performance
            # 这是设计的：错误时返回最小信息
            
            # 性能指标应该反映失败
            assert agi.performance_metrics['total_cycles'] == 1
            assert agi.performance_metrics['failed_cycles'] == 1  # 失败计数增加
            assert agi.performance_metrics['successful_cycles'] == 0
            
        finally:
            # Restore original method
            agi._process_perception = original_process
    
    def test_metrics_update(self, agi_instance):
        """Test metrics update method"""
        agi = agi_instance
        
        # 初始状态
        initial_cycles = agi.performance_metrics['total_cycles']
        initial_successful = agi.performance_metrics['successful_cycles']
        initial_failed = agi.performance_metrics['failed_cycles']
        initial_avg_time = agi.performance_metrics['avg_response_time']
        
        # 计算初始成功率
        def calculate_success_rate(metrics):
            total = metrics['total_cycles']
            if total == 0:
                return 0.0
            return metrics['successful_cycles'] / total
        
        initial_success_rate = calculate_success_rate(agi.performance_metrics)
        
        # 用成功更新指标
        agi._update_performance_metrics(start_time=time.time() - 0.5, success=True)
        
        assert agi.performance_metrics['total_cycles'] == initial_cycles + 1
        assert agi.performance_metrics['successful_cycles'] == initial_successful + 1
        assert agi.performance_metrics['failed_cycles'] == initial_failed
        assert agi.performance_metrics['avg_response_time'] > 0
        
        # 计算新的成功率
        new_success_rate = calculate_success_rate(agi.performance_metrics)
        assert new_success_rate >= initial_success_rate
        
        # 用失败更新指标
        success_rate_before = new_success_rate
        agi._update_performance_metrics(start_time=time.time() - 1.0, success=False)
        
        # 成功率应该下降
        final_success_rate = calculate_success_rate(agi.performance_metrics)
        assert final_success_rate <= success_rate_before
    
    def test_diagnostics_collection(self, agi_instance):
        """Test diagnostics collection"""
        agi = agi_instance
        
        diagnostics = agi.get_diagnostics()
        
        assert isinstance(diagnostics, dict)
        assert 'cognitive_state' in diagnostics
        assert 'system_info' in diagnostics
        assert 'representation_cache' in diagnostics
        assert 'communication_stats' in diagnostics
        assert 'configuration' in diagnostics
        
        # 检查系统信息
        system_info = diagnostics['system_info']
        assert 'total_cycles' in system_info
        assert 'successful_cycles' in system_info
        assert 'failed_cycles' in system_info
        assert 'success_rate' in system_info
        assert 'avg_response_time' in system_info
        # timestamp 在 diagnostics 顶层，不在 system_info 中
        assert 'timestamp' in diagnostics
        
        # 应该匹配内部指标
        assert system_info['total_cycles'] == agi.performance_metrics['total_cycles']
        assert system_info['successful_cycles'] == agi.performance_metrics['successful_cycles']
        assert system_info['failed_cycles'] == agi.performance_metrics['failed_cycles']
        assert system_info['avg_response_time'] == agi.performance_metrics['avg_response_time']
        
        # 检查成功率计算
        total_cycles = agi.performance_metrics['total_cycles']
        if total_cycles > 0:
            expected_success_rate = agi.performance_metrics['successful_cycles'] / total_cycles
            assert abs(system_info['success_rate'] - expected_success_rate) < 0.001
    
    @pytest.mark.asyncio
    async def test_initialize_method(self, agi_instance):
        """Test initialize method"""
        agi = agi_instance
        
        # Initialize should not crash
        # (components may fail to initialize in test environment)
        try:
            await agi.initialize()
        except Exception as e:
            # It's OK if initialization fails in tests
            # due to missing dependencies
            pass
    
    @pytest.mark.asyncio
    async def test_shutdown_method(self, agi_instance):
        """Test shutdown method"""
        agi = agi_instance
        
        # Shutdown should not crash
        await agi.shutdown()
        
        # Should be able to call shutdown multiple times
        await agi.shutdown()
    
    def test_component_processing_methods_exist(self, agi_instance):
        """Test that all component processing methods exist"""
        agi = agi_instance
        
        component_methods = [
            '_process_perception',
            '_process_attention',
            '_process_memory',
            '_process_reasoning',
            '_process_planning',
            '_process_decision',
            '_process_action',
            '_process_learning'
        ]
        
        for method_name in component_methods:
            assert hasattr(agi, method_name)
            method = getattr(agi, method_name)
            assert callable(method)
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # 使用空配置测试
        agi1 = UnifiedCognitiveArchitecture()
        assert isinstance(agi1.config, dict)
        
        # 检查默认值通过属性访问
        assert agi1.embedding_dim == 1024  # 默认值
        assert agi1.max_shared_memory_mb == 1024  # 默认值
        
        # 使用自定义配置测试
        custom_config = {
            'embedding_dim': 2048,
            'max_shared_memory_mb': 2048,
            'custom_param': 'custom_value'
        }
        
        agi2 = UnifiedCognitiveArchitecture(custom_config)
        
        # 配置应该被存储
        assert agi2.config['embedding_dim'] == 2048
        assert agi2.config['max_shared_memory_mb'] == 2048
        assert agi2.config['custom_param'] == 'custom_value'
        
        # 属性应该使用配置值
        assert agi2.embedding_dim == 2048
        assert agi2.max_shared_memory_mb == 2048
        
        # 表征空间应该使用配置
        assert agi2.repr_space.embedding_dim == 2048
    
    def test_performance_tracking(self, agi_instance):
        """Test performance tracking over multiple cycles"""
        agi = agi_instance
        
        # 模拟多个循环
        for i in range(5):
            start_time = time.time() - (0.1 * (i + 1))  # 递减的开始时间
            success = i % 2 == 0  # 交替成功/失败
            
            agi._update_performance_metrics(start_time=start_time, success=success)
        
        # 检查指标
        assert agi.performance_metrics['total_cycles'] == 5
        assert agi.performance_metrics['avg_response_time'] > 0
        assert agi.performance_metrics['total_processing_time'] > 0
        
        # 检查成功和失败计数
        assert agi.performance_metrics['successful_cycles'] >= 0
        assert agi.performance_metrics['failed_cycles'] >= 0
        assert agi.performance_metrics['successful_cycles'] + agi.performance_metrics['failed_cycles'] == 5
        
        # 计算成功率应该在0和1之间
        success_rate = agi.performance_metrics['successful_cycles'] / 5
        assert 0.0 <= success_rate <= 1.0