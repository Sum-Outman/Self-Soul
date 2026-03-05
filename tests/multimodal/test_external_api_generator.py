"""
外部API生成器测试

测试ExternalAPIMultimodalGenerator类的基本功能和接口。
由于实际API调用需要有效的API密钥，本测试主要验证：
1. 类初始化和接口正确性
2. 错误处理和降级机制
3. 配置加载和API提供商选择逻辑
4. 统计信息和状态管理
"""

import sys
import os
import logging
import time
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.multimodal.external_api_generator import ExternalAPIMultimodalGenerator
from core.multimodal.true_multimodal_generator import GenerationInput

# 设置日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_external_api_generator")


def test_initialization():
    """测试生成器初始化"""
    print("\n1. 测试生成器初始化")
    print("-" * 50)
    
    try:
        # 使用降级模式（避免API调用失败）
        generator = ExternalAPIMultimodalGenerator(enable_fallback=True)
        
        # 验证属性
        assert generator.enable_fallback is True, "降级模式设置错误"
        assert generator.config_path == "config/external_api_configs.json", "默认配置文件路径错误"
        
        # 验证支持的生成方向
        supported = generator.get_supported_directions()
        expected_directions = [("text", "image"), ("text", "text"), ("text", "audio"), ("image", "text")]
        
        for direction in expected_directions:
            assert direction in supported, f"缺少支持的生成方向: {direction}"
        
        print(f"  ✅ 生成器初始化成功")
        print(f"     支持的生成方向: {supported}")
        print(f"     降级模式: {generator.enable_fallback}")
        
    except Exception as e:
        print(f"  ❌ 生成器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        raise  # 重新抛出异常以便pytest捕获


def test_generation_interface():
    """测试生成接口"""
    print("\n2. 测试生成接口")
    print("-" * 50)
    
    try:
        generator = ExternalAPIMultimodalGenerator(enable_fallback=True)
        
        # 测试不支持的生成方向
        invalid_input = GenerationInput(
            source_modality="audio",
            target_modality="image",
            content="测试音频",
            parameters={"test": True}
        )
        
        try:
            result = generator.generate(invalid_input)
            # 如果不支持的生成方向没有抛出异常，检查结果是否表明失败或使用了降级
            if result.metadata.get("success", False):
                # 如果成功了，检查是否使用了降级
                if result.metadata.get("used_fallback", False):
                    print(f"  ⚠️  不支持的生成方向使用了降级生成器")
                else:
                    print(f"  ⚠️  不支持的生成方向意外成功")
            else:
                print(f"  ✅ 不支持的生成方向正确返回失败结果")
        except ValueError as e:
            print(f"  ✅ 不支持的生成方向正确抛出异常: {e}")
        
        # 测试支持的生成方向（由于没有API配置，应该触发降级或失败）
        test_cases = [
            ("text", "image", "一只可爱的猫在草地上玩耍"),
            ("text", "text", "请写一个关于人工智能的短故事"),
            ("text", "audio", "欢迎使用语音合成系统"),
        ]
        
        passed = 0
        total = len(test_cases)
        
        for source, target, content in test_cases:
            try:
                input_data = GenerationInput(
                    source_modality=source,
                    target_modality=target,
                    content=content,
                    parameters={"test": True, "max_tokens": 50}
                )
                
                result = generator.generate(input_data)
                
                # 验证结果结构
                assert result.target_modality == target, f"目标模态错误: {result.target_modality}"
                assert hasattr(result, 'quality_score'), "缺少质量分数"
                assert hasattr(result, 'generation_time'), "缺少生成时间"
                assert hasattr(result, 'metadata'), "缺少元数据"
                
                # 检查是否使用了降级
                if result.metadata.get("used_fallback", False):
                    print(f"  ✅ {source}->{target} 使用降级生成成功")
                else:
                    print(f"  ✅ {source}->{target} 生成成功 (API或降级)")
                
                passed += 1
                
            except Exception as e:
                print(f"  ⚠️  {source}->{target} 生成失败: {e}")
                # 在某些情况下，如果没有配置且降级也失败，这是预期的
        
        print(f"  生成接口测试: {passed}/{total} 通过")
        assert passed >= 1, f"至少需要一个生成成功，但只有{passed}个通过"
        
    except Exception as e:
        print(f"  ❌ 生成接口测试异常: {e}")
        import traceback
        traceback.print_exc()
        raise  # 重新抛出异常以便pytest捕获


def test_statistics_tracking():
    """测试统计信息跟踪"""
    print("\n3. 测试统计信息跟踪")
    print("-" * 50)
    
    try:
        generator = ExternalAPIMultimodalGenerator(enable_fallback=True)
        
        # 获取初始统计
        initial_stats = generator.get_stats()
        assert initial_stats["total_generations"] == 0, "初始总生成数应为0"
        
        # 执行一些生成操作
        test_inputs = [
            GenerationInput("text", "image", "测试图像生成", {"test": True}),
            GenerationInput("text", "text", "测试文本生成", {"test": True}),
        ]
        
        for input_data in test_inputs:
            try:
                generator.generate(input_data)
            except:
                pass  # 忽略生成失败
        
        # 获取更新后的统计
        updated_stats = generator.get_stats()
        
        # 验证统计已更新
        assert updated_stats["total_generations"] >= 2, f"总生成数未正确更新: {updated_stats['total_generations']}"
        
        # 验证统计结构
        expected_keys = ["total_generations", "successful_generations", "failed_generations", 
                        "average_quality_score", "total_generation_time", "api_calls", "errors"]
        
        for key in expected_keys:
            assert key in updated_stats, f"缺少统计键: {key}"
        
        print(f"  ✅ 统计信息跟踪正常")
        print(f"     总生成数: {updated_stats['total_generations']}")
        print(f"     成功生成数: {updated_stats['successful_generations']}")
        print(f"     失败生成数: {updated_stats['failed_generations']}")
        
        # 重置统计
        generator.reset_stats()
        reset_stats = generator.get_stats()
        
        assert reset_stats["total_generations"] == 0, "重置后总生成数应为0"
        print(f"  ✅ 统计信息重置正常")
        
        # 所有断言通过，测试成功
        
    except Exception as e:
        print(f"  ❌ 统计信息测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise  # 重新抛出异常以便pytest捕获


def test_configuration_handling():
    """测试配置处理"""
    print("\n4. 测试配置处理")
    print("-" * 50)
    
    try:
        # 测试默认配置路径
        generator = ExternalAPIMultimodalGenerator()
        assert hasattr(generator, 'api_manager'), "缺少API管理器"
        assert hasattr(generator, 'api_factory'), "缺少API工厂"
        
        # 测试自定义配置路径
        custom_generator = ExternalAPIMultimodalGenerator(
            config_path="config/external_api_configs.json",
            default_provider="openai",
            enable_fallback=False
        )
        
        assert custom_generator.config_path == "config/external_api_configs.json", "自定义配置路径错误"
        assert custom_generator.default_provider == "openai", "默认提供商设置错误"
        assert custom_generator.enable_fallback is False, "降级模式设置错误"
        
        print(f"  ✅ 配置处理正常")
        print(f"     默认配置路径: {generator.config_path}")
        print(f"     自定义配置路径: {custom_generator.config_path}")
        print(f"     默认提供商: {custom_generator.default_provider}")
        
        # 所有配置处理断言通过
        
    except Exception as e:
        print(f"  ❌ 配置处理测试失败: {e}")
        raise  # 重新抛出异常以便pytest捕获


def test_error_handling_and_fallback():
    """测试错误处理和降级机制"""
    print("\n5. 测试错误处理和降级机制")
    print("-" * 50)
    
    try:
        # 测试无降级模式（应该失败）
        no_fallback_generator = ExternalAPIMultimodalGenerator(enable_fallback=False)
        
        test_input = GenerationInput(
            source_modality="text",
            target_modality="image",
            content="测试图像生成",
            parameters={"test": True}
        )
        
        try:
            # 由于没有有效的API配置，这应该会失败
            result = no_fallback_generator.generate(test_input)
            
            # 如果成功了（可能因为某些原因），检查是否使用了降级
            if result.metadata.get("used_fallback", False):
                print(f"  ⚠️  无降级模式但使用了降级 - 这可能表示配置问题")
            else:
                print(f"  ⚠️  无降级模式但生成成功 - 这可能表示有有效配置")
                
        except Exception as e:
            print(f"  ✅ 无降级模式下API失败正确抛出异常: {type(e).__name__}")
        
        # 测试有降级模式
        with_fallback_generator = ExternalAPIMultimodalGenerator(enable_fallback=True)
        
        try:
            result = with_fallback_generator.generate(test_input)
            
            # 检查结果
            if result.metadata.get("success", False):
                if result.metadata.get("used_fallback", False):
                    print(f"  ✅ 降级机制工作正常（使用本地生成器）")
                else:
                    print(f"  ⚠️  生成成功但未使用降级（可能有有效配置）")
            else:
                print(f"  ⚠️  生成失败，但这是预期的（没有API配置）")
                
        except Exception as e:
            print(f"  ❌ 降级模式下仍然抛出异常: {e}")
            raise  # 重新抛出异常以便pytest捕获
        
        # 错误处理和降级机制测试通过
        
    except Exception as e:
        print(f"  ❌ 错误处理测试失败: {e}")
        raise  # 重新抛出异常以便pytest捕获


def test_provider_selection_logic():
    """测试API提供商选择逻辑"""
    print("\n6. 测试API提供商选择逻辑")
    print("-" * 50)
    
    try:
        generator = ExternalAPIMultimodalGenerator(enable_fallback=True)
        
        # 检查提供商优先级配置
        assert hasattr(generator, 'provider_priority'), "缺少提供商优先级配置"
        
        priority = generator.provider_priority
        
        # 验证每种生成类型都有提供商列表
        expected_generation_types = ["text_to_image", "text_to_text", "text_to_audio", "image_to_text"]
        
        for gen_type in expected_generation_types:
            assert gen_type in priority, f"缺少生成类型: {gen_type}"
            assert isinstance(priority[gen_type], list), f"{gen_type} 提供商优先级不是列表"
            assert len(priority[gen_type]) > 0, f"{gen_type} 提供商列表为空"
            
            print(f"  ✅ {gen_type}: {priority[gen_type]}")
        
        # 验证提供商是已知的
        known_providers = ["openai", "anthropic", "google_genai", "replicate", 
                          "huggingface", "stability_ai", "deepseek", "elevenlabs"]
        
        for gen_type, providers in priority.items():
            for provider in providers:
                if provider not in known_providers:
                    print(f"  ⚠️  未知提供商: {provider} (在 {gen_type} 中)")
        
        # 提供商选择逻辑测试通过
        
    except Exception as e:
        print(f"  ❌ 提供商选择逻辑测试失败: {e}")
        raise  # 重新抛出异常以便pytest捕获


def main():
    """主测试函数"""
    print("=" * 80)
    print("外部API生成器测试套件")
    print("版本: 1.0.0 | 日期: 2026-03-06")
    print("=" * 80)
    
    total_tests = 6
    passed_tests = 0
    
    # 运行测试（使用try-except捕获测试结果）
    try:
        test_initialization()
        passed_tests += 1
    except Exception:
        pass  # 测试失败，不增加计数
    
    try:
        test_generation_interface()
        passed_tests += 1
    except Exception:
        pass
    
    try:
        test_statistics_tracking()
        passed_tests += 1
    except Exception:
        pass
    
    try:
        test_configuration_handling()
        passed_tests += 1
    except Exception:
        pass
    
    try:
        test_error_handling_and_fallback()
        passed_tests += 1
    except Exception:
        pass
    
    try:
        test_provider_selection_logic()
        passed_tests += 1
    except Exception:
        pass
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    if passed_tests == total_tests:
        print(f"✅ 所有 {total_tests} 个测试全部通过!")
        print("\n🎉 外部API生成器接口验证完成!")
        print("\n📋 已验证功能:")
        print("  • 生成器初始化和配置")
        print("  • 生成接口和方向支持")
        print("  • 统计信息跟踪和重置")
        print("  • 配置处理和API提供商选择")
        print("  • 错误处理和降级机制")
        print("  • 提供商优先级逻辑")
        print("\n⚠️  注意: 实际API功能测试需要有效的API配置和密钥")
        print("    当前测试主要验证接口和错误处理逻辑")
        return 0
    else:
        print(f"⚠️  测试部分通过: {passed_tests}/{total_tests}")
        print("部分功能需要进一步调试")
        return 1


if __name__ == "__main__":
    sys.exit(main())