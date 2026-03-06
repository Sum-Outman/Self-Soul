#!/usr/bin/env python3
"""
测试模型服务启动
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.model_service_manager import model_service_manager
from core.error_handling import error_handler
import time

def test_model_service_startup():
    """测试模型服务启动"""
    print("=== 测试模型服务启动 ===")
    
    # 检查模型注册表中的模型类型
    print(f"模型注册表中的模型类型数量: {len(model_service_manager.model_registry.model_types)}")
    print("模型类型列表:", list(model_service_manager.model_registry.model_types.keys())[:10])
    
    # 测试启动单个模型服务（语言模型，端口8002）
    model_id = "language"
    print(f"\n=== 测试启动模型服务: {model_id} ===")
    
    # 先检查端口
    port = 8002
    print(f"检查端口 {port} 是否可用...")
    if not model_service_manager._check_port_available(port):
        print(f"端口 {port} 不可用")
    else:
        print(f"端口 {port} 可用")
    
    # 启动模型服务
    print(f"启动模型服务 {model_id}...")
    success = model_service_manager.start_model_service(model_id)
    print(f"启动结果: {success}")
    
    if success:
        print("等待3秒让服务器完全启动...")
        time.sleep(3)
        
        # 验证服务是否运行
        print(f"验证服务是否在端口 {port} 上运行...")
        if model_service_manager._verify_service_running(port):
            print(f"✅ 模型服务 {model_id} 在端口 {port} 上成功运行")
        else:
            print(f"❌ 模型服务 {model_id} 在端口 {port} 上未运行")
        
        # 获取服务状态
        status = model_service_manager.get_service_status(model_id)
        print(f"服务状态: {status}")
    else:
        print(f"❌ 模型服务 {model_id} 启动失败")
    
    # 测试启动所有模型服务
    print(f"\n=== 测试启动所有模型服务 ===")
    print("注意：这可能会启动多个服务，需要一些时间...")
    
    try:
        results = model_service_manager.start_all_model_services()
        success_count = sum(1 for success in results.values() if success)
        total_models = len(results)
        
        print(f"启动结果: {success_count}/{total_models} 个模型服务启动成功")
        
        # 列出失败的模型
        failed_models = [model_id for model_id, success in results.items() if not success]
        if failed_models:
            print(f"失败的模型: {failed_models}")
            
            # 检查每个失败模型的原因
            for model_id in failed_models[:5]:  # 只检查前5个
                print(f"\n检查失败模型 {model_id}:")
                # 尝试检查端口
                try:
                    from core.model_ports_config import get_model_port
                    port = get_model_port(model_id)
                    print(f"  端口配置: {port}")
                    if port:
                        print(f"  端口 {port} 可用性: {model_service_manager._check_port_available(port)}")
                except Exception as e:
                    print(f"  检查端口时出错: {e}")
    except Exception as e:
        print(f"启动所有模型服务时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n=== 测试完成 ===")

if __name__ == "__main__":
    test_model_service_startup()