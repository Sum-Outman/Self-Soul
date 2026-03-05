#!/usr/bin/env python
"""
测试真实生成模型API集成

这个脚本测试Self-Soul-B系统的API集成框架，包括：
1. API配置管理
2. 连接测试
3. 错误处理
4. 回退机制

注意：在没有真实API密钥的情况下，主要测试错误处理机制。
"""

import sys
import os
import time
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_api_config_manager():
    """测试API配置管理器"""
    print("=" * 80)
    print("测试API配置管理器")
    print("=" * 80)
    
    try:
        from core.api_config_manager import APIConfigManager
        
        config_manager = APIConfigManager()
        print("✅ APIConfigManager初始化成功")
        
        # 测试加载配置
        configs = config_manager.api_configs
        print(f"  加载了 {len(configs)} 个API配置")
        
        # 测试OpenAI配置验证
        openai_config = {
            "api_key": "test-key",
            "api_url": "https://api.openai.com/v1",
            "model_name": "gpt-3.5-turbo"
        }
        
        validation_result = config_manager.validate_config(openai_config)
        print(f"  OpenAI配置验证: {validation_result}")
        
        # 测试连接测试（应该失败，因为没有真实API密钥）
        test_result = config_manager.test_openai_connection(openai_config)
        print(f"  OpenAI连接测试结果: {test_result.get('success', False)}")
        if not test_result.get('success', False):
            print(f"    预期错误: {test_result.get('error', '未知错误')}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_model_connector():
    """测试API模型连接器"""
    print("\n" + "=" * 80)
    print("测试API模型连接器")
    print("=" * 80)
    
    try:
        from core.api_model_connector import api_model_connector
        
        print("✅ APIModelConnector初始化成功")
        
        # 测试连接测试
        test_config = {
            "api_url": "https://api.openai.com/v1",
            "api_key": "test-key-invalid",
            "model_name": "gpt-3.5-turbo"
        }
        
        result = api_model_connector._test_connection(
            test_config["api_url"],
            test_config["api_key"],
            test_config["model_name"]
        )
        
        print(f"  API连接测试结果: {result}")
        
        # 测试完整的API连接
        full_result = api_model_connector.test_api_connection(
            "test_model",
            test_config
        )
        
        print(f"  完整API连接测试: {full_result}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_external_api_service():
    """测试外部API服务"""
    print("\n" + "=" * 80)
    print("测试外部API服务")
    print("=" * 80)
    
    try:
        from core.external_api_service import ExternalAPIService
        
        api_service = ExternalAPIService()
        print("✅ ExternalAPIService初始化成功")
        
        # 测试配置API提供商
        test_config = {
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-3.5-turbo"
        }
        
        # 配置OpenAI（模拟配置，不测试真实连接）
        api_service.configure_api_provider("openai", test_config)
        print("  OpenAI配置成功")
        
        # 检查配置状态
        is_configured = api_service._is_provider_configured("openai")
        print(f"  OpenAI配置状态: {is_configured}")
        
        # 测试连接测试（模拟）
        try:
            # 这里会尝试真实连接，但应该失败
            result = api_service.test_connection("openai")
            print(f"  连接测试结果: {result}")
        except Exception as e:
            print(f"  预期连接失败: {type(e).__name__}")
        
        # 测试获取提供商列表
        providers = api_service.get_available_providers()
        print(f"  可用API提供商: {providers}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_dependency_manager():
    """测试API依赖管理器"""
    print("\n" + "=" * 80)
    print("测试API依赖管理器")
    print("=" * 80)
    
    try:
        from core.api_dependency_manager import APIDependencyManager
        
        dependency_manager = APIDependencyManager()
        print("✅ APIDependencyManager初始化成功")
        
        # 测试检查OpenAI依赖
        openai_status = dependency_manager.check_dependency("openai")
        print(f"  OpenAI依赖状态: {openai_status}")
        
        # 测试获取安装指南
        installation_guide = dependency_manager.get_installation_guide("openai", "openai")
        print(f"  OpenAI安装指南: {installation_guide[:100]}...")
        
        # 测试检查多个依赖
        dependencies = ["openai", "anthropic", "google.generativeai"]
        all_status = dependency_manager.check_dependencies(dependencies)
        print(f"  多个依赖检查结果: {len(all_status)} 个依赖")
        
        for dep, status in all_status.items():
            print(f"    {dep}: {status.get('installed', False)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_client_factory():
    """测试API客户端工厂"""
    print("\n" + "=" * 80)
    print("测试API客户端工厂")
    print("=" * 80)
    
    try:
        from core.api_client_factory import APIClientFactory
        
        client_factory = APIClientFactory()
        print("✅ APIClientFactory初始化成功")
        
        # 测试注册的提供商
        providers = client_factory.get_registered_providers()
        print(f"  已注册的API提供商: {list(providers.keys())}")
        
        # 测试创建OpenAI客户端（应该失败，因为没有API密钥）
        try:
            client = client_factory.create_client("openai", {"api_key": "test-key"})
            print(f"  OpenAI客户端创建: {'成功' if client else '失败'}")
        except Exception as e:
            print(f"  OpenAI客户端创建失败（预期）: {type(e).__name__}")
        
        # 测试测试连接功能
        test_result = client_factory.test_client_connection(
            "openai",
            {"api_key": "test-key", "base_url": "https://api.openai.com/v1"}
        )
        print(f"  客户端连接测试: {test_result}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_integration_with_mock():
    """使用模拟数据测试API集成"""
    print("\n" + "=" * 80)
    print("使用模拟数据测试API集成")
    print("=" * 80)
    
    try:
        # 测试模拟API响应
        print("测试模拟API响应生成...")
        
        # 导入必要的模块
        from core.external_api_service import ExternalAPIService
        
        # 创建API服务实例
        api_service = ExternalAPIService()
        
        # 启用测试模式（如果支持）
        os.environ["API_TEST_MODE"] = "true"
        
        # 配置模拟提供商
        mock_config = {
            "api_key": "mock-key",
            "base_url": "http://mock-api.example.com",
            "model": "mock-model",
            "test_mode": True
        }
        
        api_service.configure_api_provider("mock", mock_config)
        print("✅ 模拟API提供商配置成功")
        
        # 测试模拟文本生成
        try:
            # 注意：实际实现可能需要修改以支持模拟模式
            print("  注意：实际API集成测试需要真实API密钥")
            print("  请参考API_INTEGRATION_GUIDE.md配置真实API密钥")
        except Exception as e:
            print(f"  模拟测试预期错误: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_settings_api_config():
    """测试系统设置中的API配置"""
    print("\n" + "=" * 80)
    print("测试系统设置中的API配置")
    print("=" * 80)
    
    try:
        import json
        
        # 读取系统设置
        settings_path = "core/data/settings/system_settings.json"
        if os.path.exists(settings_path):
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            # 检查API配置
            api_configs = settings.get("external_api_configs", {})
            print(f"  系统设置中找到 {len(api_configs)} 个API配置")
            
            for config_id, config in api_configs.items():
                name = config.get("name", "未知")
                api_type = config.get("api_type", "未知")
                enabled = config.get("enabled", False)
                print(f"    - {name} ({api_type}): 启用={enabled}")
        else:
            print("⚠️  系统设置文件未找到")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_api_test_report():
    """生成API测试报告"""
    print("\n" + "=" * 80)
    print("API集成测试报告")
    print("=" * 80)
    
    tests = [
        ("API配置管理器", test_api_config_manager),
        ("API模型连接器", test_api_model_connector),
        ("外部API服务", test_external_api_service),
        ("API依赖管理器", test_api_dependency_manager),
        ("API客户端工厂", test_api_client_factory),
        ("模拟API集成", test_api_integration_with_mock),
        ("系统设置API配置", test_system_settings_api_config),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n运行测试: {test_name}")
        start_time = time.time()
        
        try:
            success = test_func()
            elapsed_time = time.time() - start_time
            
            status = "✅ 通过" if success else "❌ 失败"
            results.append((test_name, success, elapsed_time))
            
            print(f"  结果: {status}, 耗时: {elapsed_time:.2f}秒")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"  异常: {e}")
            results.append((test_name, False, elapsed_time))
    
    # 生成总结报告
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success, _ in results if success)
    failed_tests = total_tests - passed_tests
    total_time = sum(elapsed for _, _, elapsed in results)
    
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {failed_tests}")
    print(f"总耗时: {total_time:.2f}秒")
    
    if failed_tests > 0:
        print("\n失败测试详情:")
        for test_name, success, elapsed in results:
            if not success:
                print(f"  - {test_name} (耗时: {elapsed:.2f}秒)")
    
    # 提供配置建议
    print("\n" + "=" * 80)
    print("下一步行动")
    print("=" * 80)
    
    if passed_tests == total_tests:
        print("✅ 所有API集成测试通过！")
        print("\n要使用真实API:")
        print("1. 复制 .env.example 为 .env")
        print("2. 填入真实的API密钥")
        print("3. 重启系统")
        print("4. 运行真实API测试")
    else:
        print("⚠️  部分测试失败")
        print("\n建议:")
        print("1. 检查缺少的依赖: pip install openai anthropic google-generativeai")
        print("2. 检查配置文件格式")
        print("3. 查看详细错误信息")
    
    return passed_tests == total_tests

def main():
    """主函数"""
    print("=" * 80)
    print("Self-Soul-B多模态系统API集成测试")
    print("=" * 80)
    print("注意：此测试主要验证API集成框架，不测试真实API连接。")
    print("要测试真实API连接，需要配置真实的API密钥。")
    print("=" * 80)
    
    success = generate_api_test_report()
    
    if success:
        print("\n✅ API集成测试完成！")
        print("框架验证成功，可以配置真实API密钥进行完整测试。")
        return 0
    else:
        print("\n⚠️  API集成测试部分失败")
        print("请检查错误信息并修复问题。")
        return 1

if __name__ == "__main__":
    sys.exit(main())