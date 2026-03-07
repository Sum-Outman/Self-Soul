#!/usr/bin/env python
"""
运行统一认知架构测试。

Usage:
    python run_tests.py [test_module] [options]
    
Examples:
    python run_tests.py                          # Run all tests
    python run_tests.py test_representation      # Run representation tests only
    python run_tests.py -v                       # Verbose output
    python run_tests.py --coverage               # Run with coverage
"""

import sys
import os
import argparse
import subprocess
import pytest

# 添加源目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def run_pytest_tests(test_module=None, verbose=False, coverage=False):
    """
    使用pytest运行测试。
    
    Args:
        test_module: 要运行的特定测试模块（None表示全部）
        verbose: 启用详细输出
        coverage: 启用覆盖率报告
    """
    # Build pytest arguments
    pytest_args = []
    
    if verbose:
        pytest_args.append('-v')
    
    if coverage:
        pytest_args.extend([
            '--cov=src',
            '--cov-report=term',
            '--cov-report=html:coverage_html'
        ])
    
    # Add test module if specified
    if test_module:
        test_path = os.path.join(os.path.dirname(__file__), f"{test_module}.py")
        if os.path.exists(test_path):
            pytest_args.append(test_path)
        else:
            print(f"Error: Test module '{test_module}' not found.")
            print(f"Available test modules:")
            for f in os.listdir(os.path.dirname(__file__)):
                if f.startswith('test_') and f.endswith('.py'):
                    print(f"  - {f[5:-3]}")
            return False
    
    # Run pytest
    print(f"Running tests with arguments: {pytest_args}")
    print("=" * 80)
    
    result = pytest.main(pytest_args)
    
    return result == 0


def run_quick_test():
    """Run a quick sanity test without pytest"""
    print("Running quick sanity test...")
    
    try:
        # Test imports
        from cognitive.representation import UnifiedRepresentationSpace
        from neural.communication import NeuralCommunication
        
        print("✓ 导入成功")
        
        # 测试基本功能
        repr_space = UnifiedRepresentationSpace(embedding_dim=512)
        print(f"✓ 统一表征空间已初始化 (维度={repr_space.embedding_dim})")
        
        comm = NeuralCommunication(max_shared_memory_mb=100)
        print(f"✓ 神经通信系统已初始化 (最大内存={comm.max_shared_memory/(1024*1024)}MB)")
        
        # 测试编码
        test_input = {'text': "快速测试输入"}
        encoded = repr_space.encode(test_input, use_cache=False)
        print(f"✓ 编码成功 (形状={encoded.shape})")
        
        print("\n✅ 所有快速测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """检查所需依赖是否已安装"""
    required_packages = [
        'torch',
        'numpy',
        'pytest',
        'pytest_asyncio'
    ]
    
    print("检查依赖...")
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} (缺失)")
    
    if missing_packages:
        print(f"\n缺失的包: {', '.join(missing_packages)}")
        print("安装命令: pip install " + " ".join(missing_packages))
        return False
    
    print("\n✅ 所有依赖可用")
    return True


def main():
    """主入口点"""
    parser = argparse.ArgumentParser(description='运行统一认知架构测试')
    parser.add_argument('test_module', nargs='?', help='要运行的特定测试模块')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    parser.add_argument('-q', '--quick', action='store_true', help='仅运行快速健全性测试')
    parser.add_argument('-c', '--coverage', action='store_true', help='运行覆盖率报告')
    parser.add_argument('-d', '--deps', action='store_true', help='仅检查依赖')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("统一认知架构 - 测试运行器")
    print("=" * 80)
    
    # 检查依赖
    if not check_dependencies():
        if not args.deps:
            print("\n缺少必要依赖，无法运行测试。")
            return 1
    
    if args.deps:
        return 0
    
    # Run quick test if requested
    if args.quick:
        success = run_quick_test()
        return 0 if success else 1
    
    # Run full tests
    success = run_pytest_tests(
        test_module=args.test_module,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())