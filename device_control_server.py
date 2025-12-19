import asyncio
import websockets
import json
import os
from core.training_manager import TrainingManager
from core.model_registry import ModelRegistry
from core.knowledge_integrator_enhanced import AGIKnowledgeIntegrator

# 全局组件实例
training_manager = None
model_registry = None
knowledge_integrator = None

# 连接管理
connected_clients = set()

def initialize_components():
    """初始化核心组件"""
    global training_manager, model_registry, knowledge_integrator
    
    # 初始化模型注册表
    model_registry = ModelRegistry()
    
    # 初始化知识库集成器
    knowledge_integrator = AGIKnowledgeIntegrator()
    
    # 初始化训练管理器
    training_manager = TrainingManager(
        model_registry=model_registry,
        from_scratch=True  # 设置为从零开始训练
    )
    
    # 初始化所有模型客户端为从零开始模式
    training_manager._initialize_model_clients(from_scratch=True)
    
    print("所有核心组件已初始化完成，模型已准备好进行训练")
    
    # 设置模型初始状态为准备训练
    if hasattr(training_manager, 'set_model_status'):
        for model_id in model_registry.models:
            training_manager.set_model_status(model_id, 'PREPARING')

async def handle_client(websocket, path):
    """处理WebSocket客户端连接"""
    client_id = id(websocket)
    connected_clients.add(websocket)
    print(f"客户端连接: {client_id}")
    
    try:
        # 发送服务器初始化成功消息
        await websocket.send(json.dumps({
            'type': 'INITIALIZATION_SUCCESS',
            'message': '设备控制服务器已初始化',
            'models_ready': True
        }))
        
        # 处理客户端消息
        async for message in websocket:
            try:
                data = json.loads(message)
                await process_client_message(websocket, data)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    'type': 'ERROR',
                    'message': '无效的JSON格式'
                }))
    except websockets.exceptions.ConnectionClosed as e:
        print(f"客户端断开连接: {client_id}, 原因: {e}")
    finally:
        connected_clients.remove(websocket)

async def process_client_message(websocket, data):
    """处理客户端发送的消息"""
    message_type = data.get('type', '')
    
    if message_type == 'GET_MODEL_STATUS':
        # 获取模型状态
        model_id = data.get('model_id')
        model_status = get_model_status(model_id)
        await websocket.send(json.dumps({
            'type': 'MODEL_STATUS',
            'model_id': model_id,
            'status': model_status
        }))
        
    elif message_type == 'START_TRAINING':
        # 开始训练
        model_ids = data.get('model_ids', [])
        parameters = data.get('parameters', {})
        job_id = start_training(model_ids, parameters)
        await websocket.send(json.dumps({
            'type': 'TRAINING_STARTED',
            'job_id': job_id,
            'model_ids': model_ids
        }))
        
    elif message_type == 'GET_TRAINING_PROGRESS':
        # 获取训练进度
        job_id = data.get('job_id')
        progress = get_training_progress(job_id)
        await websocket.send(json.dumps({
            'type': 'TRAINING_PROGRESS',
            'job_id': job_id,
            'progress': progress
        }))
        
    elif message_type == 'GET_ALL_MODELS':
        # 获取所有模型
        models = get_all_models()
        await websocket.send(json.dumps({
            'type': 'ALL_MODELS',
            'models': models
        }))
        
    elif message_type == 'PREPARE_MODEL_FOR_TRAINING':
        # 准备模型训练
        model_id = data.get('model_id')
        result = prepare_model_for_training(model_id)
        await websocket.send(json.dumps({
            'type': 'MODEL_PREPARATION_RESULT',
            'model_id': model_id,
            'success': result['success'],
            'message': result['message'],
            'progress': result.get('progress', 0)
        }))
        
    else:
        await websocket.send(json.dumps({
            'type': 'ERROR',
            'message': f'未知的消息类型: {message_type}'
        }))

def get_model_status(model_id):
    """获取指定模型的状态"""
    if training_manager and model_registry:
        model = model_registry.get_model(model_id)
        if model:
            # 检查是否有状态管理器
            if hasattr(training_manager, 'get_model_status'):
                return training_manager.get_model_status(model_id)
            else:
                # 默认返回准备状态
                return {
                    'status': 'PREPARING',
                    'from_scratch': getattr(model, 'from_scratch', False),
                    'training_available': hasattr(model, 'train'),
                    'progress': 0
                }
    return {'status': 'not_found'}

def prepare_model_for_training(model_id):
    """准备模型训练"""
    if training_manager and model_registry:
        model = model_registry.get_model(model_id)
        if model:
            try:
                # 执行模型准备操作
                # 这里可以添加实际的模型准备逻辑
                print(f"正在准备模型 {model_id} 进行训练...")
                
                # 如果training_manager有prepare_model方法，则调用它
                if hasattr(training_manager, 'prepare_model'):
                    result = training_manager.prepare_model(model_id)
                    return result
                
                # 模拟准备过程，实际应用中应替换为真实的准备逻辑
                # 准备完成后更新状态
                if hasattr(training_manager, 'set_model_status'):
                    training_manager.set_model_status(model_id, 'PREPARED')
                    
                return {
                    'success': True,
                    'message': f"模型 {model_id} 准备成功",
                    'progress': 100
                }
            except Exception as e:
                return {
                    'success': False,
                    'message': f"模型 {model_id} 准备失败: {str(e)}",
                    'progress': 0
                }
    return {
        'success': False,
        'message': f"未找到模型 {model_id}"
    }

def start_training(model_ids, parameters):
    """启动模型训练"""
    if training_manager:
        # 确保from_scratch设置为True
        parameters['from_scratch'] = True
        
        # 检查所有模型是否准备就绪
        for model_id in model_ids:
            status = get_model_status(model_id)
            if status.get('status') not in ['PREPARED', 'ready']:
                # 如果模型未准备好，先准备模型
                prepare_result = prepare_model_for_training(model_id)
                if not prepare_result['success']:
                    print(f"模型 {model_id} 准备失败，无法开始训练")
                    return None
        
        return training_manager.start_training(model_ids, parameters)
    return None

def get_training_progress(job_id):
    """获取训练进度"""
    if training_manager and hasattr(training_manager, 'dashboard_data'):
        return training_manager.dashboard_data.get('training_progress', {}).get(job_id, 0)
    return 0

def get_all_models():
    """获取所有可用模型"""
    models = []
    if model_registry:
        for model_id, model in model_registry.models.items():
            models.append({
                'id': model_id,
                'type': model_id,
                'status': 'ready',
                'from_scratch': getattr(model, 'from_scratch', False)
            })
    return models

async def broadcast_status_update():
    """广播状态更新到所有客户端"""
    while True:
        if connected_clients and training_manager:
            # 准备状态更新数据
            status_data = {
                'type': 'SYSTEM_STATUS',
                'training_jobs': len(training_manager.training_jobs) if hasattr(training_manager, 'training_jobs') else 0,
                'active_clients': len(connected_clients),
                'models_ready': True,
                'models_prepared': 0
            }
            
            # 获取所有模型的准备状态
            if model_registry:
                models_prepared = 0
                for model_id in model_registry.models:
                    status = get_model_status(model_id)
                    if status.get('status') in ['PREPARED', 'ready']:
                        models_prepared += 1
                status_data['models_prepared'] = models_prepared
                status_data['total_models'] = len(model_registry.models)
            
            # 发送状态更新到所有连接的客户端
            for client in list(connected_clients):
                try:
                    await client.send(json.dumps(status_data))
                except websockets.exceptions.ConnectionClosed:
                    pass
        
        # 每5秒更新一次
        await asyncio.sleep(5)

async def main():
    """主函数"""
    # 初始化核心组件
    initialize_components()
    
    # 启动WebSocket服务器
    server = await websockets.serve(handle_client, "localhost", 8765)
    print("设备控制WebSocket服务器已启动在 ws://localhost:8765")
    
    # 启动状态广播任务
    broadcast_task = asyncio.create_task(broadcast_status_update())
    
    # 保持服务器运行
    await server.wait_closed()
    
    # 取消广播任务
    broadcast_task.cancel()
    await broadcast_task

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("服务器已停止")
