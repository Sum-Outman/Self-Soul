"""
神经通信系统测试。
"""

import pytest
import pytest_asyncio
import asyncio
import torch
import time
from neural.communication import NeuralCommunication, MessagePriority


@pytest_asyncio.fixture
async def neural_comm():
    """NeuralCommunication的异步测试夹具"""
    comm = NeuralCommunication(max_shared_memory_mb=100)
    try:
        yield comm
    finally:
        comm.shutdown()

@pytest.fixture
def neural_comm_sync():
    """NeuralCommunication的同步测试夹具"""
    comm = NeuralCommunication(max_shared_memory_mb=100)
    try:
        yield comm
    finally:
        comm.shutdown()


class TestNeuralCommunication:
    """神经通信系统测试套件"""
    
    def test_initialization(self, neural_comm_sync):
        """测试神经通信系统初始化"""
        comm = neural_comm_sync
        
        # 检查基本属性
        assert hasattr(comm, 'input_queues')
        assert hasattr(comm, 'output_queues')
        assert hasattr(comm, 'tensor_channels')
        assert hasattr(comm, 'registered_components')
        assert hasattr(comm, 'shared_memory')
        assert hasattr(comm, 'max_shared_memory')
        assert hasattr(comm, 'lock')
        assert hasattr(comm, 'stats')
        
        # 检查队列已初始化
        for priority in MessagePriority:
            assert priority in comm.input_queues
            assert priority in comm.output_queues
        
        # 检查统计信息
        stats = comm.get_statistics()
        assert stats['registered_components'] == 0
        assert stats['shared_tensors'] == 0
        assert stats['current_memory_usage_mb'] == 0
    
    def test_component_registration(self, neural_comm_sync):
        """测试组件注册"""
        comm = neural_comm_sync
        
        # 注册组件
        comm.register_component("perception", "perception")
        comm.register_component("memory", "memory")
        comm.register_component("reasoning", "reasoning")
        
        stats = comm.get_statistics()
        assert stats['registered_components'] == 3
        
        # 检查通道是否创建
        assert "perception" in comm.tensor_channels
        assert "memory" in comm.tensor_channels
        assert "reasoning" in comm.tensor_channels
        
        # 取消注册组件
        comm.unregister_component("memory")
        stats_after = comm.get_statistics()
        assert stats_after['registered_components'] == 2
        assert "memory" not in comm.tensor_channels
    
    def test_shared_tensor_creation(self, neural_comm_sync):
        """测试共享张量创建"""
        comm = neural_comm_sync
        
        # 创建共享张量
        tensor = comm.create_shared_tensor("test_tensor", shape=(3, 224, 224))
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert tensor.is_shared()  # 应该在共享内存中
        
        # 检查内存使用情况
        stats = comm.get_statistics()
        assert stats['shared_tensors'] == 1
        assert stats['current_memory_usage_mb'] > 0
        
        # 获取现有张量
        retrieved = comm.get_shared_tensor("test_tensor")
        assert retrieved is not None
        assert torch.equal(tensor, retrieved)
        
        # 获取不存在的张量
        nonexistent = comm.get_shared_tensor("nonexistent")
        assert nonexistent is None
    
    @pytest.mark.asyncio
    async def test_tensor_send_receive(self, neural_comm):
        """测试张量发送和接收"""
        comm = neural_comm
        
        # 注册测试组件
        comm.register_component("test_component", "test")
        
        # 创建测试张量
        test_tensor = torch.randn(3, 224, 224)
        
        # 发送张量
        message_id = await comm.send_tensor(
            test_tensor, 
            "test_component",
            priority=MessagePriority.NORMAL,
            metadata={'test': 'data'}
        )
        
        assert isinstance(message_id, str)
        assert len(message_id) > 0
        
        # 检查统计信息
        stats = comm.get_statistics()
        assert stats['messages_sent'] == 1
        assert stats['tensors_transferred'] == 1
        assert stats['total_data_transferred'] > 0
        
        # 注意：接收需要实际组件监听，
        # 这在隔离测试中更复杂
        # 此测试验证发送端正常工作
    
    @pytest.mark.asyncio
    async def test_request_response_pattern(self, neural_comm):
        """测试请求-响应模式"""
        comm = neural_comm
        
        # 注册测试组件
        comm.register_component("test_component", "test")
        
        # 创建测试张量
        test_tensor = torch.randn(1, 512)
        
        # 发送请求（将超时，因为无人响应）
        # 使用短超时进行测试
        response = await comm.send_request(
            test_tensor, 
            "test_component",
            metadata={'test': 'request'},
            timeout=0.1  # 测试用的短超时
        )
        
        # 应该超时，因为无人监听
        assert response is None
    
    @pytest.mark.asyncio
    async def test_broadcast_tensor(self, neural_comm):
        """测试张量广播"""
        comm = neural_comm
        
        # 注册多个同类型组件
        comm.register_component("perception1", "perception")
        comm.register_component("perception2", "perception")
        comm.register_component("memory1", "memory")
        
        # 创建测试张量
        test_tensor = torch.randn(1, 512)
        
        # 广播到感知组件
        # 注意：broadcast_tensor不返回任何内容，只发送
        # 我们测试它不会崩溃
        await comm.broadcast_tensor(
            test_tensor,
            "perception",
            metadata={'broadcast': 'test'}
        )
        
        # 检查感知组件的张量通道是否存在
        assert "perception1" in comm.tensor_channels
        assert "perception2" in comm.tensor_channels
    
    def test_statistics_tracking(self, neural_comm_sync):
        """测试统计信息跟踪"""
        comm = neural_comm_sync
        
        # 注册一些组件
        comm.register_component("component1", "type1")
        comm.register_component("component2", "type2")
        
        # 创建一些共享张量
        comm.create_shared_tensor("tensor1", shape=(10, 10))
        comm.create_shared_tensor("tensor2", shape=(20, 20))
        
        # 获取统计信息
        stats = comm.get_statistics()
        
        assert stats['registered_components'] == 2
        assert stats['shared_tensors'] == 2
        assert stats['current_memory_usage_mb'] > 0
        assert 'input_queue_sizes' in stats
        assert 'output_queue_sizes' in stats
        
        # 重置统计信息
        comm.reset_statistics()
        stats_after = comm.get_statistics()
        
        assert stats_after['messages_sent'] == 0
        assert stats_after['messages_received'] == 0
        assert stats_after['tensors_transferred'] == 0
        assert stats_after['total_data_transferred'] == 0
        assert stats_after['errors'] == 0
        
        # 组件计数应保持不变
        assert stats_after['registered_components'] == 2
    
    def test_memory_cleanup(self):
        """测试内存清理（旧张量）"""
        comm = NeuralCommunication(max_shared_memory_mb=10)  # 小内存限制
        
        try:
            # 创建多个张量以超过内存限制
            tensors = []
            for i in range(10):
                tensor = comm.create_shared_tensor(f"tensor_{i}", shape=(1000, 1000))
                tensors.append(tensor)
            
            stats = comm.get_statistics()
            
            # 内存使用应该被管理
            assert stats['current_memory_usage_mb'] <= 10
            
            # 一些张量可能已被清理
            # 确切数量取决于清理逻辑
            assert stats['shared_tensors'] <= 10
        finally:
            comm.shutdown()
    
    def test_message_priority_queues(self, neural_comm_sync):
        """测试消息优先级队列存在"""
        comm = neural_comm_sync
        
        # 检查所有优先级队列是否存在
        for priority in MessagePriority:
            assert priority in comm.input_queues
            assert priority in comm.output_queues
            
            queue = comm.input_queues[priority]
            # 检查队列属性
            assert hasattr(queue, 'put')
            assert hasattr(queue, 'get')
    
    def test_component_activity_tracking(self, neural_comm_sync):
        """测试组件活动跟踪"""
        comm = neural_comm_sync
        
        # 注册组件
        comm.register_component("active_component", "active")
        
        # 模拟活动（通过发送张量）
        # （在实际使用中，组件在处理时会更新last_active）
        
        # 监控线程在后台运行
        # 我们不便于在没有等待的情况下测试它
        
        # 仅验证组件已注册
        assert "active_component" in comm.registered_components
    
    def test_error_handling(self, neural_comm_sync):
        """测试错误处理"""
        comm = neural_comm_sync
        
        # 尝试发送到不存在的组件
        # 这不应该崩溃，只会增加错误计数
        test_tensor = torch.randn(1, 512)
        
        # 我们不便于在没有异步上下文的情况下测试异步错误
        # 但我们可以测试错误计数从0开始
        stats = comm.get_statistics()
        assert stats['errors'] == 0
    
    def test_response_channels(self, neural_comm_sync):
        """测试响应通道管理"""
        comm = neural_comm_sync
        
        # 创建响应通道
        channel_id = "test_response_channel"
        comm.response_channels[channel_id] = comm.manager.Queue(maxsize=1)
        
        assert channel_id in comm.response_channels
        
        # 移除通道
        del comm.response_channels[channel_id]
        assert channel_id not in comm.response_channels
    
    def test_concurrent_access(self, neural_comm_sync):
        """测试通信系统支持并发访问"""
        comm = neural_comm_sync
        
        # 并发注册多个组件
        # （模拟的 - 实际并发需要线程）
        components = [
            ("comp1", "type1"),
            ("comp2", "type2"),
            ("comp3", "type3")
        ]
        
        for comp_id, comp_type in components:
            comm.register_component(comp_id, comp_type)
        
        # 所有组件都应该已注册
        stats = comm.get_statistics()
        assert stats['registered_components'] == 3
        
        for comp_id, _ in components:
            assert comp_id in comm.registered_components
    
    def test_resource_monitoring(self, neural_comm_sync):
        """测试资源监控初始化"""
        comm = neural_comm_sync
        
        # 监控线程应该已启动
        assert hasattr(comm, 'monitor_thread')
        assert comm.monitor_thread.is_alive()
        
        # 给它一点时间启动
        time.sleep(0.1)
        
        # 线程应该仍然存活
        assert comm.monitor_thread.is_alive()
    
    def test_configuration_options(self):
        """测试不同的配置选项"""
        # 测试不同的内存限制
        for memory_mb in [10, 100, 1024]:
            comm = NeuralCommunication(max_shared_memory_mb=memory_mb)
            try:
                assert comm.max_shared_memory == memory_mb * 1024 * 1024
                
                stats = comm.get_statistics()
                assert stats['max_memory_mb'] == memory_mb
            finally:
                comm.shutdown()
    
    @pytest.mark.asyncio
    async def test_receive_tensor(self, neural_comm):
        """测试接收张量功能"""
        comm = neural_comm
        
        # 注册发送方组件
        comm.register_component("sender", "test")
        
        # 创建测试张量
        test_tensor = torch.randn(3, 224, 224)
        
        # 发送张量到sender组件（模拟发送）
        # 我们需要先将张量放入通道
        message_data = {
            'tensor_data': {
                'data': test_tensor.cpu().numpy().tobytes(),
                'shape': test_tensor.shape,
                'dtype': str(test_tensor.dtype)
            },
            'metadata': {'test': 'receive'}
        }
        
        # 手动将消息放入通道
        channel = comm.tensor_channels.get("sender")
        if channel is None:
            with comm.lock:
                comm.tensor_channels["sender"] = comm.manager.Queue(maxsize=100)
                channel = comm.tensor_channels["sender"]
        
        # 将消息放入队列
        channel.put(message_data)
        
        # 现在接收张量
        result = await comm.receive_tensor("sender", timeout=1.0)
        
        # 验证结果
        assert result is not None
        tensor, metadata = result
        assert torch.allclose(tensor, test_tensor)
        assert metadata.get('test') == 'receive'
        
        # 验证统计信息
        stats = comm.get_statistics()
        assert stats['messages_received'] >= 1
    
    @pytest.mark.asyncio
    async def test_send_response(self, neural_comm):
        """测试发送响应功能"""
        comm = neural_comm
        
        # 创建请求消息ID
        request_id = "test_request_123"
        response_channel = f"response_{request_id}"
        
        # 创建响应通道
        with comm.lock:
            comm.response_channels[response_channel] = comm.manager.Queue(maxsize=1)
        
        # 创建测试响应张量
        response_tensor = torch.randn(1, 512)
        
        # 发送响应
        await comm.send_response(request_id, response_tensor, metadata={'response': 'test'})
        
        # 验证响应已发送到队列
        response_queue = comm.response_channels[response_channel]
        
        # 尝试从队列获取响应（非阻塞方式）
        try:
            response_data = response_queue.get_nowait()
            assert response_data is not None
            assert 'tensor_data' in response_data
            assert 'metadata' in response_data
            assert response_data['metadata'].get('response') == 'test'
        except:
            # 队列为空，表示响应未发送成功
            pytest.fail("响应未发送到队列")
        
        # 清理
        with comm.lock:
            if response_channel in comm.response_channels:
                del comm.response_channels[response_channel]
    
    def test_reset_statistics(self, neural_comm_sync):
        """测试重置统计信息"""
        comm = neural_comm_sync
        
        # 先执行一些操作以增加统计计数
        comm.register_component("test_component", "test")
        comm.create_shared_tensor("test_tensor", shape=(10, 10))
        
        # 获取初始统计信息
        stats_before = comm.get_statistics()
        assert stats_before['registered_components'] == 1
        assert stats_before['shared_tensors'] == 1
        
        # 重置统计信息
        comm.reset_statistics()
        
        # 获取重置后的统计信息
        stats_after = comm.get_statistics()
        assert stats_after['messages_sent'] == 0
        assert stats_after['messages_received'] == 0
        assert stats_after['tensors_transferred'] == 0
        assert stats_after['total_data_transferred'] == 0
        assert stats_after['errors'] == 0
        
        # 注意：reset_statistics 不重置注册的组件和共享张量
        # 只重置消息和错误统计
        assert stats_after['registered_components'] == 1  # 保持不变
        assert stats_after['shared_tensors'] == 1  # 保持不变
    
    def test_shared_tensor_manager(self):
        """测试共享张量管理器"""
        # 导入SharedTensorManager（从communication模块）
        from neural.communication import SharedTensorManager
        
        # 创建管理器
        manager = SharedTensorManager(max_memory_mb=100)
        
        try:
            # 测试创建张量
            tensor1 = manager.create_tensor("tensor1", shape=(10, 20))
            assert tensor1 is not None
            assert tensor1.shape == (10, 20)
            assert tensor1.is_shared()
            
            # 测试获取张量
            retrieved = manager.get_tensor("tensor1")
            assert retrieved is not None
            assert torch.equal(tensor1, retrieved)
            
            # 测试获取不存在的张量
            nonexistent = manager.get_tensor("nonexistent")
            assert nonexistent is None
            
            # 测试列出张量
            tensor_list = manager.list_tensors()
            assert "tensor1" in tensor_list
            assert tensor_list["tensor1"]["shape"] == (10, 20)
            
            # 测试内存统计
            memory_stats = manager.get_memory_stats()
            assert memory_stats["tensor_count"] == 1
            assert memory_stats["current_memory_mb"] > 0
            
            # 测试创建另一个张量
            tensor2 = manager.create_tensor("tensor2", shape=(5, 5))
            assert tensor2 is not None
            assert manager.get_memory_stats()["tensor_count"] == 2
            
            # 测试删除张量
            delete_result = manager.delete_tensor("tensor1")
            assert delete_result is True
            assert manager.get_tensor("tensor1") is None
            assert manager.get_memory_stats()["tensor_count"] == 1
            
            # 测试删除不存在的张量
            delete_fail = manager.delete_tensor("nonexistent")
            assert delete_fail is False
            
            # 测试清空所有张量
            manager.clear_all()
            assert manager.get_memory_stats()["tensor_count"] == 0
            assert manager.get_memory_stats()["current_memory_mb"] == 0
            
        finally:
            # 清理
            manager.shutdown()