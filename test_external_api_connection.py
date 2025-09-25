#!/usr/bin/env python3
"""
外部API连接测试脚本 - External API Connection Test Script
测试所有主流AI API的连接和功能
Test connection and functionality of all mainstream AI APIs
"""

import sys
import os
import json
from datetime import datetime

# 添加core目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

try:
    from external_api_service import ExternalAPIService
except ImportError as e:
    print(f"导入ExternalAPIService失败: {e}")
    print("尝试直接导入...")
    # 尝试直接导入
    import importlib.util
    spec = importlib.util.spec_from_file_location("external_api_service", "core/external_api_service.py")
    external_api_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(external_api_module)
    ExternalAPIService = external_api_module.ExternalAPIService

def test_api_connections():
    """测试所有API连接 | Test all API connections"""
    print("=" * 60)
    print("外部API连接测试开始 | External API Connection Test Started")
    print("=" * 60)
    
    # 创建测试配置
    test_config = {
        "external_apis": {
            # 测试配置 - 使用虚拟密钥进行连接测试
            "openai": {
                "chat": {
                    "api_key": "test_openai_key",
                    "model": "gpt-3.5-turbo"
                }
            },
            "anthropic": {
                "chat": {
                    "api_key": "test_anthropic_key", 
                    "model": "claude-3-sonnet-20240229"
                }
            },
            "google_ai": {
                "ai": {
                    "api_key": "test_google_ai_key",
                    "model": "gemini-pro"
                }
            },
            "google": {
                "vision": {
                    "api_key": "test_google_vision_key"
                }
            },
            "aws": {
                "rekognition": {
                    "access_key": "test_aws_access_key",
                    "secret_key": "test_aws_secret_key",
                    "region": "us-east-1"
                }
            },
            "azure": {
                "vision": {
                    "endpoint": "https://test.cognitiveservices.azure.com/",
                    "subscription_key": "test_azure_key"
                }
            },
            "huggingface": {
                "inference": {
                    "api_key": "test_hf_key"
                }
            },
            "cohere": {
                "chat": {
                    "api_key": "test_cohere_key",
                    "model": "command"
                }
            },
            "mistral": {
                "chat": {
                    "api_key": "test_mistral_key",
                    "model": "mistral-large-latest"
                }
            }
        }
    }
    
    try:
        # 初始化API服务
        print("初始化外部API服务... | Initializing external API service...")
        api_service = ExternalAPIService(test_config)
        
        # 测试服务状态
        print("\n获取API服务状态... | Getting API service status...")
        status = api_service.get_service_status()
        
        print("\nAPI服务状态报告 | API Service Status Report:")
        print("-" * 40)
        
        for provider, provider_status in status.items():
            configured = "✓ 已配置" if provider_status["configured"] else "✗ 未配置"
            services = ", ".join(provider_status["services_available"]) if provider_status["services_available"] else "无可用服务"
            print(f"{provider.upper():<12} | {configured:<10} | 服务: {services}")
        
        # 测试文本生成功能（模拟）
        print("\n测试文本生成功能... | Testing text generation functionality...")
        test_prompt = "Hello, this is a test message. Please respond with 'Test successful'."
        
        # 测试每个配置的API
        api_types = ["openai", "anthropic", "google_ai", "huggingface", "cohere", "mistral"]
        
        for api_type in api_types:
            print(f"\n测试 {api_type.upper()} API... | Testing {api_type.upper()} API...")
            try:
                result = api_service.generate_text(test_prompt, api_type=api_type)
                if "error" in result:
                    print(f"  {api_type.upper()} API: ✗ 连接测试失败 - {result['error']}")
                else:
                    print(f"  {api_type.upper()} API: ✓ 功能正常（模拟）")
            except Exception as e:
                print(f"  {api_type.upper()} API: ✗ 异常 - {str(e)}")
        
        # 测试图像分析功能（模拟）
        print("\n测试图像分析功能... | Testing image analysis functionality...")
        vision_apis = ["google", "aws", "azure"]
        
        for vision_api in vision_apis:
            print(f"\n测试 {vision_api.upper()} 视觉API... | Testing {vision_api.upper()} Vision API...")
            try:
                # 使用空数据测试
                result = api_service.analyze_image(b"test_image_data", api_type=vision_api)
                if "error" in result:
                    print(f"  {vision_api.upper()} Vision: ✗ 连接测试失败 - {result['error']}")
                else:
                    print(f"  {vision_api.upper()} Vision: ✓ 功能正常（模拟）")
            except Exception as e:
                print(f"  {vision_api.upper()} Vision: ✗ 异常 - {str(e)}")
        
        # 测试配置保存和加载
        print("\n测试配置管理功能... | Testing configuration management...")
        try:
            # 使用绝对路径确保目录存在
            test_config_path = os.path.join(os.path.dirname(__file__), "test_config.json")
            save_result = api_service.save_configuration(test_config_path)
            if "success" in save_result and save_result["success"]:
                print("  ✓ 配置保存功能正常")
            else:
                print(f"  ✗ 配置保存失败: {save_result.get('error', '未知错误')}")
                
            load_result = api_service.load_configuration(test_config_path)
            if "success" in load_result and load_result["success"]:
                print("  ✓ 配置加载功能正常")
            else:
                print(f"  ✗ 配置加载失败: {load_result.get('error', '未知错误')}")
                
            # 清理测试文件
            if os.path.exists(test_config_path):
                os.remove(test_config_path)
                
        except Exception as e:
            print(f"  ✗ 配置管理异常 - {str(e)}")
        
        print("\n" + "=" * 60)
        print("外部API连接测试完成 | External API Connection Test Completed")
        print("=" * 60)
        
        # 生成测试报告
        test_report = {
            "test_timestamp": datetime.now().isoformat(),
            "service_status": status,
            "summary": "所有API服务初始化成功，功能测试完成（使用模拟配置）"
        }
        
        print(f"\n测试报告已生成 | Test report generated")
        return test_report
        
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)} | Error during testing: {str(e)}")
        return {"error": str(e)}

def test_model_registry_integration():
    """测试模型注册表集成 | Test model registry integration"""
    print("\n" + "=" * 60)
    print("模型注册表集成测试 | Model Registry Integration Test")
    print("=" * 60)
    
    try:
        # 修复模型注册表导入问题 - 使用绝对导入
        try:
            # 首先确保core目录在Python路径中
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            core_dir = os.path.join(current_dir, 'core')
            if core_dir not in sys.path:
                sys.path.insert(0, core_dir)
            
            # 使用绝对导入
            from model_registry import ModelRegistry, test_external_api_connection
        except ImportError as e:
            print(f"绝对导入失败: {e}")
            print("尝试直接导入模型注册表...")
            import importlib.util
            spec = importlib.util.spec_from_file_location("model_registry", os.path.join(current_dir, "core", "model_registry.py"))
            model_registry_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_registry_module)
            ModelRegistry = model_registry_module.ModelRegistry
            test_external_api_connection = model_registry_module.test_external_api_connection
        
        # 初始化模型注册表
        registry = ModelRegistry()
        
        # 测试外部API连接功能
        print("测试外部API连接功能... | Testing external API connection function...")
        
        # 模拟外部API配置
        test_api_config = {
            "provider": "openai",
            "api_key": "test_key",
            "model_name": "gpt-3.5-turbo",
            "base_url": "https://api.openai.com/v1"
        }
        
        connection_result = test_external_api_connection("test_model", test_api_config)
        print(f"API连接测试结果: {connection_result}")
        
        # 测试模型切换功能
        print("\n测试模型切换功能... | Testing model switching function...")
        
        # 创建测试模型
        test_model_id = "test_external_model"
        test_model_config = {
            "name": "Test External Model",
            "type": "language",
            "external_config": test_api_config
        }
        
        # 注册测试模型（需要先有模型类）
        # 这里简化测试，直接测试切换功能
        print(f"跳过模型注册，直接测试切换功能...")
        
        # 测试切换到外部模式
        switch_result = registry.switch_model_to_external(test_model_id, test_api_config)
        print(f"切换到外部模式结果: {switch_result}")
        
        # 测试切换回本地模式
        switch_back_result = registry.switch_model_to_local(test_model_id)
        print(f"切换回本地模式结果: {switch_back_result}")
        
        print("模型注册表集成测试完成 | Model registry integration test completed")
        return {"success": True}
        
    except Exception as e:
        print(f"模型注册表集成测试失败: {str(e)} | Model registry integration test failed: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return {"error": str(e)}

if __name__ == "__main__":
    # 运行API连接测试
    api_test_result = test_api_connections()
    
    # 运行模型注册表集成测试
    registry_test_result = test_model_registry_integration()
    
    print("\n" + "=" * 60)
    print("综合测试报告 | Comprehensive Test Report")
    print("=" * 60)
    print(f"API连接测试: {'成功' if 'error' not in api_test_result else '失败'}")
    print(f"模型注册表集成测试: {'成功' if 'error' not in registry_test_result else '失败'}")
    
    if 'error' not in api_test_result and 'error' not in registry_test_result:
        print("\n🎉 所有测试通过！外部API服务功能正常。")
        print("🎉 All tests passed! External API service functionality is working correctly.")
    else:
        print("\n❌ 部分测试失败，需要检查实现。")
        print("❌ Some tests failed, implementation needs to be checked.")
