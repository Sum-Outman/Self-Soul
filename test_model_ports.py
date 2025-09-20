#!/usr/bin/env python3
"""
模型服务测试脚本
用于测试finance和medical模型服务是否能够正常加载和运行
"""
import sys
import os
import time
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入所需的模块
from core.model_ports_config import get_model_port, load_config_from_file
from core.model_service_manager import model_service_manager

class ModelPortTester:
    """模型端口测试类"""
    
    def __init__(self):
        # 加载最新的配置
        load_config_from_file()
        self.test_results = {}
    
    def test_model(self, model_id: str) -> Dict[str, Any]:
        """测试指定模型的服务"""
        print(f"\n===== 开始测试模型: {model_id} =====")
        
        # 获取模型端口
        port = get_model_port(model_id)
        if not port:
            result = {
                "model_id": model_id,
                "status": "error",
                "message": f"模型 {model_id} 未配置端口"
            }
            print(f"错误: {result['message']}")
            return result
        
        print(f"模型 {model_id} 的端口配置为: {port}")
        
        # 检查模型服务是否已运行
        service_status = model_service_manager.get_service_status(model_id)
        if service_status and service_status["is_running"]:
            print(f"模型 {model_id} 的服务已经在运行")
        else:
            # 启动模型服务
            print(f"正在启动模型 {model_id} 的服务...")
            start_time = time.time()
            success = model_service_manager.start_model_service(model_id)
            elapsed_time = time.time() - start_time
            
            if success:
                print(f"成功启动模型 {model_id} 的服务，耗时: {elapsed_time:.2f} 秒")
            else:
                result = {
                    "model_id": model_id,
                    "status": "error",
                    "port": port,
                    "message": f"启动模型 {model_id} 的服务失败"
                }
                print(f"错误: {result['message']}")
                return result
        
        # 获取服务状态
        service_status = model_service_manager.get_service_status(model_id)
        if service_status:
            result = {
                "model_id": model_id,
                "status": "success" if service_status["is_running"] else "error",
                "port": service_status["port"],
                "is_running": service_status["is_running"],
                "message": "服务运行正常" if service_status["is_running"] else "服务未运行"
            }
            print(f"测试结果: {result['message']}, 运行状态: {result['is_running']}")
        else:
            result = {
                "model_id": model_id,
                "status": "error",
                "port": port,
                "message": "无法获取服务状态"
            }
            print(f"错误: {result['message']}")
        
        print(f"===== 测试模型: {model_id} 完成 =====")
        return result
    
    def test_all_models(self) -> Dict[str, Dict[str, Any]]:
        """测试所有关键模型"""
        models_to_test = ["finance", "medical", "value_alignment"]
        
        for model_id in models_to_test:
            self.test_results[model_id] = self.test_model(model_id)
        
        return self.test_results
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n\n===== 测试摘要 =====")
        success_count = sum(1 for result in self.test_results.values() if result["status"] == "success")
        error_count = len(self.test_results) - success_count
        
        print(f"总测试模型数: {len(self.test_results)}")
        print(f"成功: {success_count}")
        print(f"失败: {error_count}")
        
        if error_count > 0:
            print("\n失败的模型:")
            for model_id, result in self.test_results.items():
                if result["status"] == "error":
                    print(f"  - {model_id}: {result['message']}")
        
        print("===================")
        
        return self.test_results

if __name__ == "__main__":
    print("模型服务端口测试工具")
    print("====================")
    
    # 创建测试器实例
    tester = ModelPortTester()
    
    # 测试所有模型
    tester.test_all_models()
    
    # 打印测试摘要
    tester.print_summary()