#!/usr/bin/env python3
"""
检查 main.py 中的导入错误和 NameError
"""

import sys
import os
import ast
import importlib

def find_undefined_names_in_main():
    """分析 main.py 中未定义的名称"""
    main_path = "d:\\2026\\20260101\\Self-Soul-B\\core\\main.py"
    
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析AST
    tree = ast.parse(content)
    
    # 收集所有导入的名称
    imported_names = set()
    undefined_names = []
    
    class ImportVisitor(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                imported_names.add(alias.name)
        
        def visit_ImportFrom(self, node):
            if node.module:
                for alias in node.names:
                    full_name = f"{node.module}.{alias.name}" if node.module else alias.name
                    imported_names.add(full_name)
                    imported_names.add(alias.name)
    
    # 收集导入的名称
    visitor = ImportVisitor()
    visitor.visit(tree)
    
    # 检查全局变量定义
    class GlobalDefVisitor(ast.NodeVisitor):
        def __init__(self):
            self.global_vars = set()
        
        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.global_vars.add(target.id)
        
        def visit_AnnAssign(self, node):
            if isinstance(node.target, ast.Name):
                self.global_vars.add(node.target.id)
    
    global_visitor = GlobalDefVisitor()
    global_visitor.visit(tree)
    
    # 收集所有名称引用
    class NameVisitor(ast.NodeVisitor):
        def __init__(self, imported_names, global_vars):
            self.imported_names = imported_names
            self.global_vars = global_vars
            self.undefined = []
        
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                name = node.id
                # 检查是否是内置函数/变量
                if (name not in self.imported_names and 
                    name not in self.global_vars and
                    name not in dir(__builtins__) and
                    not name.startswith('_') and
                    name not in ['self', 'cls']):
                    self.undefined.append(name)
    
    name_visitor = NameVisitor(imported_names, global_visitor.global_vars)
    name_visitor.visit(tree)
    
    return list(set(name_visitor.undefined))

def test_import_manager_modules():
    """测试各种管理器模块的导入"""
    manager_modules = [
        'core.training_manager',
        'core.agi_core', 
        'core.hardware.camera_manager',
        'core.dataset_manager',
        'core.system_settings_manager',
        'core.system_monitor',
        'core.emotion_awareness',
        'core.unified_cognitive_architecture',
        'core.self_learning',
        'core.enhanced_meta_cognition',
        'core.intrinsic_motivation_system',
        'core.explainable_ai',
        'core.value_alignment',
        'core.agi_coordinator',
        'core.external_api_service',
        'core.api_model_connector',
        'core.model_service_manager',
        'core.hardware.external_device_interface',
        'core.memory_optimization',
    ]
    
    results = []
    for module_name in manager_modules:
        try:
            module = importlib.import_module(module_name)
            results.append((module_name, True, None))
        except Exception as e:
            results.append((module_name, False, str(e)))
    
    return results

def check_model_registry_imports():
    """检查 model_registry.py 中的导入"""
    try:
        from core.model_registry import ModelRegistry
        model_registry = ModelRegistry()
        return True, "ModelRegistry 导入和实例化成功"
    except Exception as e:
        return False, f"ModelRegistry 错误: {type(e).__name__}: {e}"

def main():
    """主函数"""
    print("检查导入错误和 NameError...")
    print("=" * 60)
    
    # 1. 查找未定义的名称
    print("\n1. 查找 main.py 中的未定义名称:")
    undefined_names = find_undefined_names_in_main()
    if undefined_names:
        print(f"  找到 {len(undefined_names)} 个可能未定义的名称:")
        for name in sorted(undefined_names):
            print(f"  - {name}")
    else:
        print("  未找到未定义的名称")
    
    # 2. 测试管理器模块导入
    print("\n2. 测试管理器模块导入:")
    import_results = test_import_manager_modules()
    for module_name, success, error in import_results:
        if success:
            print(f"  ✓ {module_name}")
        else:
            print(f"  ✗ {module_name}: {error}")
    
    # 3. 检查 ModelRegistry
    print("\n3. 检查 ModelRegistry:")
    success, message = check_model_registry_imports()
    if success:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")
    
    # 4. 尝试直接运行 main.py 的导入部分
    print("\n4. 尝试导入 main.py 的全局导入部分:")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 导入 main.py 中的导入部分
    try:
        # 只导入前几行，避免执行整个文件
        import_text = """
import os
import sys
import time
import tempfile
import asyncio
import uvicorn
import threading
import argparse
import json
import logging
from datetime import datetime
import numpy as np
import uuid
import cv2

# Add the root directory to sys.path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator, root_validator, model_validator
from typing import List, Optional, Dict, Any

from core.error_handling import error_handler
from core.model_ports_config import MAIN_API_PORT, MODEL_PORTS
"""
        exec(import_text)
        print("  ✓ main.py 基础导入成功")
    except Exception as e:
        print(f"  ✗ main.py 基础导入失败: {type(e).__name__}: {e}")
    
    print("\n" + "=" * 60)
    print("检查完成")

if __name__ == "__main__":
    main()