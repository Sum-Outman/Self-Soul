#!/usr/bin/env python3
"""
测试six模块是否能被正确导入和使用
"""
import sys

print(f"Python版本: {sys.version}")
print(f"Python路径: {sys.executable}")

try:
    import six
    print("成功导入six模块")
    
    try:
        from six.moves import urllib
        print("成功导入six.moves.urllib")
        
        try:
            from six.moves.urllib.request import build_opener, install_opener, getproxies
            print("成功导入six.moves.urllib.request中的函数")
            
            # 测试一个简单的操作
            opener = build_opener()
            print("成功创建opener对象")
            
            proxies = getproxies()
            print(f"获取代理设置: {proxies}")
            
        except ImportError as e:
            print(f"无法导入six.moves.urllib.request: {e}")
        
    except ImportError as e:
        print(f"无法导入six.moves.urllib: {e}")
        
except ImportError as e:
    print(f"无法导入six模块: {e}")
    
    # 检查模块路径
    import os
    print("Python模块搜索路径:")
    for path in sys.path:
        if os.path.exists(os.path.join(path, "six.py")) or os.path.exists(os.path.join(path, "six")):
            print(f"  - {path} (包含six模块)")
        else:
            print(f"  - {path}")

print("\n测试完成")