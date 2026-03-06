#!/usr/bin/env python3
"""
测试单个模型服务启动
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.model_service_manager import model_service_manager
from core.error_handling import error_handler

def test_single_model_service():
    """测试单个模型服务启动"""
    print("=== 测试单个模型服务启动 ===")
    
    model_id = "language"
    print(f"测试模型: {model_id}")
    
    # 检查模型是否在注册表中
    if model_id not in model_service_manager.model_registry.model_types:
        print(f"❌ 模型 {model_id} 不在注册表中")
        return False
    
    print(f"✅ 模型 {model_id} 在注册表中")
    
    # 检查端口配置
    from core.model_ports_config import get_model_port
    port = get_model_port(model_id)
    print(f"端口配置: {port}")
    
    if not port:
        print(f"❌ 模型 {model_id} 没有端口配置")
        return False
    
    # 检查端口可用性
    print(f"检查端口 {port} 可用性...")
    if not model_service_manager._check_port_available(port):
        print(f"❌ 端口 {port} 不可用")
        return False
    print(f"✅ 端口 {port} 可用")
    
    # 检查是否已有服务
    if model_id in model_service_manager.model_services:
        print(f"模型 {model_id} 的服务已存在")
        service_info = model_service_manager.model_services[model_id]
        print(f"服务信息: 端口={service_info.get('port')}, 运行状态={service_info.get('is_running')}")
    
    # 启动模型服务
    print(f"\n启动模型服务 {model_id}...")
    start_time = time.time()
    success = model_service_manager.start_model_service(model_id)
    elapsed = time.time() - start_time
    
    print(f"启动结果: {success}, 耗时: {elapsed:.2f}秒")
    
    if success:
        print(f"等待2秒让服务器稳定...")
        time.sleep(2)
        
        # 验证服务是否运行
        print(f"验证服务是否在端口 {port} 上运行...")
        if model_service_manager._verify_service_running(port):
            print(f"✅ 模型服务 {model_id} 在端口 {port} 上成功运行")
            
            # 测试HTTP连接
            import requests
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=5)
                print(f"健康检查响应: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"健康检查失败: {e}")
        else:
            print(f"❌ 模型服务 {model_id} 在端口 {port} 上未运行")
            
            # 检查线程状态
            if model_id in model_service_manager.model_services:
                service_info = model_service_manager.model_services[model_id]
                thread = service_info.get("thread")
                if thread:
                    print(f"线程状态: 存活={thread.is_alive()}")
                else:
                    print(f"线程: None")
    else:
        print(f"❌ 模型服务 {model_id} 启动失败")
        
        # 检查可能的原因
        print(f"\n检查可能的原因:")
        # 检查服务信息
        if model_id in model_service_manager.model_services:
            service_info = model_service_manager.model_services[model_id]
            print(f"服务信息: {service_info}")
    
    return success

if __name__ == "__main__":
    try:
        success = test_single_model_service()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)