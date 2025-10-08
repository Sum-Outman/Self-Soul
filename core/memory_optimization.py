import gc
import torch
import psutil
import os
from typing import Dict, Any, Optional
from functools import lru_cache

# 全局组件注册表 - 单例模式管理核心组件
global_components_registry = {}

class MemoryOptimizer:
    """内存优化器，用于监控和优化系统内存使用"""
    def __init__(self):
        self.max_memory_usage = 75  # 最大内存使用率百分比
        self.enable_optimization = True
        self.lazy_loading = True
        self.lightweight_mode = False  # 轻量模式，使用更小的模型参数

    def check_memory_usage(self) -> Dict[str, float]:
        """检查当前系统内存使用情况"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_percent = psutil.virtual_memory().percent
        
        return {
            "used": memory_info.rss / (1024 * 1024),  # MB
            "percent": memory_percent,
            "available": psutil.virtual_memory().available / (1024 * 1024)  # MB
        }

    def should_optimize(self) -> bool:
        """根据内存使用情况决定是否需要优化"""
        if not self.enable_optimization:
            return False
        
        memory_usage = self.check_memory_usage()
        return memory_usage["percent"] > self.max_memory_usage

    def optimize_memory(self):
        """执行内存优化操作"""
        # 运行垃圾回收
        gc.collect()
        
        # 如果使用PyTorch，清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # 在CPU上也清理缓存
        if hasattr(torch, 'empty_cache'):
            torch.empty_cache()

# 创建全局内存优化器实例
memory_optimizer = MemoryOptimizer()

class ComponentFactory:
    """组件工厂，管理核心组件的创建和共享"""
    @staticmethod
    def get_component(component_name: str, component_class, *args, **kwargs) -> Any:
        """获取或创建组件实例，实现单例模式"""
        if component_name not in global_components_registry:
            # 检查内存使用情况，必要时优化
            if memory_optimizer.should_optimize():
                memory_optimizer.optimize_memory()
                
            # 如果在轻量模式下，传递轻量参数
            if memory_optimizer.lightweight_mode:
                if hasattr(component_class, 'get_lightweight_params'):
                    lightweight_kwargs = component_class.get_lightweight_params()
                    kwargs.update(lightweight_kwargs)
                # 为常见模型组件设置轻量参数
                if 'representation_dim' in kwargs:
                    kwargs['representation_dim'] = min(kwargs['representation_dim'], 384)
                if 'num_layers' in kwargs:
                    kwargs['num_layers'] = max(1, kwargs['num_layers'] // 2)
                if 'nhead' in kwargs:
                    kwargs['nhead'] = max(4, kwargs['nhead'] // 2)
                
            # 创建组件实例
            global_components_registry[component_name] = component_class(*args, **kwargs)
        
        return global_components_registry[component_name]
    
    @staticmethod
    def reset_component(component_name: str):
        """重置指定组件"""
        if component_name in global_components_registry:
            del global_components_registry[component_name]
            gc.collect()

class ModelConfigurationManager:
    """模型配置管理器，管理模型参数和优化设置"""
    @staticmethod
    def get_optimized_model_params(model_type: str, default_params: Dict[str, Any]) -> Dict[str, Any]:
        """根据模型类型和系统状态返回优化的模型参数"""
        optimized_params = default_params.copy()
        
        # 根据轻量模式调整参数
        if memory_optimizer.lightweight_mode:
            # 文本编码器优化
            if model_type == 'text_encoder':
                optimized_params.update({
                    'd_model': 384,  # 从768减少
                    'nhead': 6,      # 从12减少
                    'num_layers': 3, # 从6减少
                    'dim_feedforward': 1536  # 从3072减少
                })
            
            # LSTM模型优化
            if model_type == 'lstm':
                if 'hidden_size' in optimized_params:
                    optimized_params['hidden_size'] = optimized_params['hidden_size'] // 2
            
        return optimized_params

# 配置系统参数的函数
def configure_memory_optimization(
    enable_optimization: bool = True,
    lightweight_mode: bool = False,
    max_memory_usage: int = 75
):
    """配置内存优化参数"""
    memory_optimizer.enable_optimization = enable_optimization
    memory_optimizer.lightweight_mode = lightweight_mode
    memory_optimizer.max_memory_usage = max_memory_usage

# 获取全局组件的便捷函数
def get_global_component(component_name: str, component_class=None, *args, **kwargs) -> Any:
    """获取全局组件实例，如不存在则创建"""
    if component_name not in global_components_registry:
        if component_class is None:
            raise ValueError(f"Component '{component_name}' not found and no class provided")
        return ComponentFactory.get_component(component_name, component_class, *args, **kwargs)
    return global_components_registry[component_name]

# 初始化系统配置
def initialize_system_configuration(config_path: Optional[str] = None):
    """从配置文件初始化系统设置"""
    # 默认配置
    config = {
        'lightweight_mode': False,
        'enable_memory_optimization': True,
        'max_memory_usage': 75
    }
    
    # 这里可以添加从配置文件加载设置的代码
    
    # 应用配置
    configure_memory_optimization(
        enable_optimization=config['enable_memory_optimization'],
        lightweight_mode=config['lightweight_mode'],
        max_memory_usage=config['max_memory_usage']
    )

# 默认初始化系统配置
initialize_system_configuration()