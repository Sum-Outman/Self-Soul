"""
增强图像处理器测试

测试增强的真实图像处理器，验证：
1. 真实图像格式解码功能
2. 图像质量检查
3. 尺寸调整和标准化
4. 性能基准测试
"""

import torch
import numpy as np
import logging
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.multimodal.true_data_processor import TrueImageProcessor, TrueMultimodalDataProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_enhanced_image_processor")


def create_test_image_data(format_name: str, width: int = 400, height: int = 300) -> bytes:
    """
    创建测试图像数据
    
    Args:
        format_name: 图像格式名称
        width: 图像宽度
        height: 图像高度
        
    Returns:
        图像字节数据
    """
    import io
    from PIL import Image
    
    # 创建测试图像（渐变）
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(3):  # RGB通道
        for y in range(height):
            gradient[y, :, i] = np.linspace(0, 255, width) * (i + 1) / 3
    
    img = Image.fromarray(gradient, 'RGB')
    
    # 保存为指定格式
    img_bytes = io.BytesIO()
    
    if format_name == "jpeg":
        img.save(img_bytes, format='JPEG', quality=90)
    elif format_name == "png":
        img.save(img_bytes, format='PNG')
    elif format_name == "gif":
        img.save(img_bytes, format='GIF')
    elif format_name == "bmp":
        img.save(img_bytes, format='BMP')
    elif format_name == "webp":
        img.save(img_bytes, format='WEBP', quality=90)
    elif format_name == "tiff":
        img.save(img_bytes, format='TIFF')
    else:
        # 默认使用PNG
        img.save(img_bytes, format='PNG')
    
    return img_bytes.getvalue()


def test_format_detection():
    """测试格式检测功能"""
    print("\n1. 测试格式检测")
    print("-" * 50)
    
    processor = TrueImageProcessor()
    passed = 0
    total = 0
    
    test_formats = ["jpeg", "png", "gif"]  # 简化测试集
    
    for format_name in test_formats:
        total += 1
        try:
            # 创建测试图像
            image_data = create_test_image_data(format_name, 200, 150)
            
            # 检测格式
            detected = processor.detect_format(image_data)
            
            # 验证检测结果
            if detected == format_name:
                print(f"  ✅ {format_name.upper()}格式检测成功: {detected}")
                passed += 1
            else:
                print(f"  ❌ {format_name.upper()}格式检测失败: 期望={format_name}, 实际={detected}")
                
        except Exception as e:
            print(f"  ❌ {format_name.upper()}格式检测异常: {e}")
    
    print(f"  格式检测: {passed}/{total} 通过")
    assert passed == total, f"格式检测失败: {passed}/{total} 通过"


def test_image_decoding():
    """测试图像解码功能"""
    print("\n2. 测试图像解码")
    print("-" * 50)
    
    processor = TrueImageProcessor()
    passed = 0
    total = 0
    
    test_formats = ["jpeg", "png"]  # 测试主要格式
    
    for format_name in test_formats:
        total += 1
        try:
            # 创建测试图像
            image_data = create_test_image_data(format_name, 400, 300)
            
            # 解码图像
            start_time = time.time()
            image_tensor = processor.preprocess_image(image_data, format_hint=format_name)
            decode_time = time.time() - start_time
            
            # 验证结果
            if isinstance(image_tensor, torch.Tensor):
                # 检查形状 [C, H, W]
                if len(image_tensor.shape) == 3:
                    channels, height, width = image_tensor.shape
                    if channels == 3 and height == 224 and width == 224:
                        print(f"  ✅ {format_name.upper()}解码成功: {width}x{height}x{channels}, 耗时: {decode_time:.3f}s")
                        passed += 1
                    else:
                        print(f"  ❌ {format_name.upper()}解码形状错误: {width}x{height}x{channels}")
                else:
                    print(f"  ❌ {format_name.upper()}解码维度错误: {image_tensor.shape}")
            else:
                print(f"  ❌ {format_name.upper()}解码返回类型错误: {type(image_tensor)}")
                
        except Exception as e:
            print(f"  ❌ {format_name.upper()}解码异常: {e}")
    
    print(f"  图像解码: {passed}/{total} 通过")
    assert passed == total, f"图像解码失败: {passed}/{total} 通过"


def test_image_quality_validation():
    """测试图像质量验证"""
    print("\n3. 测试图像质量验证")
    print("-" * 50)
    
    processor = TrueImageProcessor()
    passed = 0
    total = 3
    
    try:
        # 测试1: 有效图像
        image_data = create_test_image_data("png", 400, 300)
        image_tensor = processor.preprocess_image(image_data)
        print(f"  ✅ 测试1 - 有效图像处理成功")
        passed += 1
        
        # 测试2: 无效数据（过小）
        try:
            invalid_data = b"invalid"
            processor.preprocess_image(invalid_data)
            print(f"  ❌ 测试2 - 无效数据未抛出异常")
        except (ValueError, RuntimeError):
            print(f"  ✅ 测试2 - 无效数据正确抛出异常")
            passed += 1
        
        # 测试3: 空数据
        try:
            empty_data = b""
            processor.preprocess_image(empty_data)
            print(f"  ❌ 测试3 - 空数据未抛出异常")
        except (ValueError, RuntimeError):
            print(f"  ✅ 测试3 - 空数据正确抛出异常")
            passed += 1
            
    except Exception as e:
        print(f"  ❌ 图像质量验证测试异常: {e}")
    
    print(f"  质量验证: {passed}/{total} 通过")
    assert passed == total, f"图像质量验证失败: {passed}/{total} 通过"


def test_resize_options():
    """测试尺寸调整选项"""
    print("\n4. 测试尺寸调整选项")
    print("-" * 50)
    
    passed = 0
    total = 2
    
    try:
        # 测试1: 保持宽高比
        processor_aspect = TrueImageProcessor(
            target_size=(300, 200),
            maintain_aspect_ratio=True,
            interpolation_mode='bilinear'
        )
        
        image_data = create_test_image_data("png", 400, 300)  # 原始4:3
        image_tensor = processor_aspect.preprocess_image(image_data)
        
        if image_tensor.shape == (3, 200, 300):  # C, H, W
            print(f"  ✅ 测试1 - 保持宽高比成功: {image_tensor.shape}")
            passed += 1
        else:
            print(f"  ❌ 测试1 - 保持宽高比失败: {image_tensor.shape}")
        
        # 测试2: 直接调整尺寸
        processor_direct = TrueImageProcessor(
            target_size=(150, 100),
            maintain_aspect_ratio=False,
            interpolation_mode='bilinear'
        )
        
        image_tensor2 = processor_direct.preprocess_image(image_data)
        
        if image_tensor2.shape == (3, 100, 150):
            print(f"  ✅ 测试2 - 直接调整尺寸成功: {image_tensor2.shape}")
            passed += 1
        else:
            print(f"  ❌ 测试2 - 直接调整尺寸失败: {image_tensor2.shape}")
            
    except Exception as e:
        print(f"  ❌ 尺寸调整测试异常: {e}")
    
    print(f"  尺寸调整: {passed}/{total} 通过")
    assert passed == total, f"尺寸调整测试失败: {passed}/{total} 通过"


def test_performance_benchmark():
    """测试性能基准"""
    print("\n5. 测试性能基准")
    print("-" * 50)
    
    processor = TrueImageProcessor()
    test_sizes = [(200, 150), (400, 300), (800, 600)]
    
    results = []
    
    for width, height in test_sizes:
        # 创建测试图像
        image_data = create_test_image_data("png", width, height)
        
        # 多次测试取平均
        num_runs = 5
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            image_tensor = processor.preprocess_image(image_data)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results.append({
            'size': f"{width}x{height}",
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'tensor_shape': image_tensor.shape
        })
        
        print(f"  📊 {width}x{height}: {avg_time*1000:.1f} ± {std_time*1000:.1f} ms")
    
    # 性能要求：每张图像处理时间 < 100ms
    all_passed = all(r['avg_time_ms'] < 100 for r in results)
    
    if all_passed:
        print(f"  ✅ 所有尺寸处理时间 < 100ms")
    else:
        print(f"  ⚠️  部分尺寸处理时间 > 100ms")
    
    assert all_passed, "性能测试失败：部分图像处理时间超过100ms"


def test_integration_with_multimodal_processor():
    """测试与多模态处理器的集成（专注于图像处理）"""
    print("\n6. 测试多模态处理器集成（图像处理）")
    print("-" * 50)
    
    try:
        # 创建多模态处理器（禁用向量存储以避免外部依赖）
        multimodal_processor = TrueMultimodalDataProcessor(enable_vector_store=False)
        
        # 创建测试数据 - 只测试图像处理，不测试文本编码
        test_input = {
            "image_data": create_test_image_data("png", 400, 300)
        }
        
        metadata = {
            "source": "integration_test",
            "timestamp": "2026-03-06T00:00:00",
            "test_purpose": "image_processing_only"
        }
        
        # 处理输入 - 只调用图像处理，不涉及文本编码
        start_time = time.time()
        
        # 直接调用图像处理器（避免文本编码的联网问题）
        image_processor = multimodal_processor.image_processor
        image_data = test_input["image_data"]
        
        # 测试格式检测
        detected_format = image_processor.detect_format(image_data)
        
        # 测试图像预处理
        image_tensor = image_processor.preprocess_image(image_data)
        
        process_time = time.time() - start_time
        
        # 验证结果
        assert detected_format == "png", f"图像格式检测失败: 期望='png', 实际='{detected_format}'"
        print(f"  ✅ 图像格式检测成功: {detected_format}")
        
        # 验证图像张量
        assert isinstance(image_tensor, torch.Tensor), f"图像处理返回类型错误: {type(image_tensor)}"
        
        channels, height, width = image_tensor.shape
        assert channels == 3 and height == 224 and width == 224, f"图像张量形状错误: {width}x{height}x{channels}，期望224x224x3"
        print(f"  ✅ 图像处理成功: {width}x{height}x{channels}, 耗时: {process_time:.3f}s")
        
        # 验证张量值范围（标准化后应在合理范围内）
        min_val = image_tensor.min().item()
        max_val = image_tensor.max().item()
        mean_val = image_tensor.mean().item()
        
        print(f"     张量统计: 最小值={min_val:.3f}, 最大值={max_val:.3f}, 均值={mean_val:.3f}")
        
        # 验证张量没有NaN或Inf值
        assert not torch.isnan(image_tensor).any() and not torch.isinf(image_tensor).any(), "图像张量包含NaN或Inf值"
        print(f"  ✅ 图像张量质量检查通过")
            
    except Exception as e:
        print(f"  ❌ 多模态集成测试异常: {e}")
        import traceback
        traceback.print_exc()
        raise  # 重新抛出异常以便pytest捕获


def main():
    """主测试函数"""
    print("=" * 80)
    print("增强图像处理器测试套件")
    print("版本: 1.0.0 | 日期: 2026-03-06")
    print("=" * 80)
    
    # 检查依赖
    try:
        from PIL import Image
        print("✅ PIL/Pillow可用")
    except ImportError:
        print("❌ PIL/Pillow不可用，部分测试将跳过")
    
    total_tests = 6
    passed_tests = 0
    
    # 运行测试（使用try-except捕获测试结果）
    try:
        test_format_detection()
        passed_tests += 1
    except Exception:
        pass  # 测试失败，不增加计数
    
    try:
        test_image_decoding()
        passed_tests += 1
    except Exception:
        pass
    
    try:
        test_image_quality_validation()
        passed_tests += 1
    except Exception:
        pass
    
    try:
        test_resize_options()
        passed_tests += 1
    except Exception:
        pass
    
    try:
        test_performance_benchmark()
        passed_tests += 1
    except Exception:
        pass
    
    try:
        test_integration_with_multimodal_processor()
        passed_tests += 1
    except Exception:
        pass
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    if passed_tests == total_tests:
        print(f"✅ 所有 {total_tests} 个测试全部通过!")
        print("\n🎉 增强图像处理器验证完成!")
        print("\n📋 已验证功能:")
        print("  • 图像格式检测 (JPEG, PNG, GIF等)")
        print("  • 真实图像解码 (PIL/OpenCV)")
        print("  • 图像质量验证和错误处理")
        print("  • 尺寸调整选项 (保持宽高比/直接调整)")
        print("  • 性能基准测试 (<100ms/图像)")
        print("  • 多模态处理器集成（图像处理验证）")
        return 0
    else:
        print(f"⚠️  测试部分通过: {passed_tests}/{total_tests}")
        print("部分功能需要进一步调试")
        return 1


if __name__ == "__main__":
    sys.exit(main())