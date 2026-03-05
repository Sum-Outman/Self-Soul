#!/usr/bin/env python
"""
运行性能基准测试
"""

import sys
import os
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_performance_benchmark():
    """测试性能基准测试"""
    try:
        from tests.multimodal.performance_benchmark import PerformanceBenchmark
        
        print("创建性能基准测试实例...")
        benchmark = PerformanceBenchmark(num_iterations=2)  # 使用少量迭代
        
        print("运行性能基准测试...")
        start_time = time.time()
        results = benchmark.run_benchmark()
        elapsed_time = time.time() - start_time
        
        print(f"性能测试完成，耗时: {elapsed_time:.2f}秒")
        print(f"测试配置: {results['test_config']}")
        
        # 输出简要结果
        if 'single_modality_tests' in results:
            print("\n单模态测试结果:")
            for name, test_result in results['single_modality_tests'].items():
                avg_time = test_result.get('total_avg_time', 0)
                print(f"  {name}: 平均时间 {avg_time*1000:.2f}ms")
        
        if 'multimodal_tests' in results:
            print("\n多模态测试结果:")
            for name, test_result in results['multimodal_tests'].items():
                avg_time = test_result.get('total_avg_time', 0)
                print(f"  {name}: 平均时间 {avg_time*1000:.2f}ms")
        
        if 'summary' in results:
            print("\n性能总结:")
            for key, value in results['summary'].items():
                print(f"  {key}: {value}")
        
        return True
        
    except ImportError as e:
        print(f"导入错误: {e}")
        return False
    except Exception as e:
        print(f"运行性能测试时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_test_suite():
    """测试性能测试套件"""
    try:
        from tests.multimodal.performance_test_suite import PerformanceTestSuite
        
        print("\n创建性能测试套件实例...")
        test_suite = PerformanceTestSuite(num_iterations=1)  # 使用少量迭代
        
        print("运行性能测试套件...")
        start_time = time.time()
        results = test_suite.run_all_tests()
        elapsed_time = time.time() - start_time
        
        print(f"性能测试套件完成，耗时: {elapsed_time:.2f}秒")
        print(f"测试配置: {results['test_config']}")
        
        # 输出简要结果
        if 'summary' in results:
            print("\n性能总结:")
            for key, value in results['summary'].items():
                print(f"  {key}: {value}")
        
        return True
        
    except ImportError as e:
        print(f"导入错误: {e}")
        return False
    except Exception as e:
        print(f"运行性能测试套件时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("Self-Soul-B多模态系统性能测试")
    print("=" * 80)
    
    success1 = test_performance_benchmark()
    success2 = test_performance_test_suite()
    
    if success1 and success2:
        print("\n✅ 所有性能测试完成!")
    else:
        print("\n❌ 部分性能测试失败!")
        sys.exit(1)