import gc
import torch
import psutil
import os
import time
from typing import Dict, Any, Optional, List, Tuple
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
    
    def aggressive_memory_cleanup(self, memory_threshold: int = 80):
        """
        积极的内存清理，当内存使用超过阈值时执行
        
        Args:
            memory_threshold: 内存使用率阈值，超过此值执行积极清理
        """
        memory_usage = self.check_memory_usage()
        current_percent = memory_usage["percent"]
        
        if current_percent <= memory_threshold:
            return {"success": True, "message": f"Memory usage {current_percent}% <= threshold {memory_threshold}%, skipping aggressive cleanup"}
        
        print(f"⚠️ Performing aggressive memory cleanup - current usage: {current_percent}% > threshold: {memory_threshold}%")
        
        # 第一步：深度垃圾回收
        for i in range(3):  # 运行3次GC以确保清理彻底
            collected = gc.collect(generation=i)
            if collected > 0:
                print(f"  GC generation {i} collected {collected} objects")
        
        # 第二步：清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print(f"  Cleared PyTorch CUDA cache")
        
        # 第三步：清理全局组件注册表中非必要的组件
        essential_components = ['memory_optimizer', 'model_registry', 'training_manager']
        components_to_remove = []
        
        for component_name in list(global_components_registry.keys()):
            if component_name not in essential_components:
                components_to_remove.append(component_name)
        
        # 避免在迭代过程中修改字典
        for component_name in components_to_remove:
            try:
                del global_components_registry[component_name]
                print(f"  Removed non-essential component: {component_name}")
            except Exception as e:
                print(f"  Failed to remove component {component_name}: {e}")
        
        # 第四步：尝试卸载非必要的模型
        try:
            # 动态导入模型注册表以避免循环依赖
            from core.model_registry import get_model_registry
            model_registry = get_model_registry()
            
            # 获取所有已加载的模型
            all_models = model_registry.list_models()
            loaded_models = []
            
            for model in all_models:
                model_id = model.get('model_id')
                # 检查模型是否已加载（简化检查）
                try:
                    # 这里需要根据实际实现检查模型是否已加载
                    # 暂时假设所有模型都已加载
                    loaded_models.append(model_id)
                except Exception as e:
                    # 模型检查失败，跳过此模型
                    pass
            
            # 定义模型优先级：必要模型 > 最近使用模型 > 其他模型
            essential_models = ['manager', 'knowledge', 'language']
            recent_models = []  # 可以扩展为记录最近使用时间
            
            # 卸载非必要模型
            models_to_unload = []
            for model_id in loaded_models:
                if model_id not in essential_models and model_id not in recent_models:
                    models_to_unload.append(model_id)
            
            # 限制每次最多卸载3个模型，避免过度卸载
            models_to_unload = models_to_unload[:3]
            
            for model_id in models_to_unload:
                try:
                    model_registry.unload_model(model_id)
                    print(f"  Unloaded non-essential model: {model_id}")
                except Exception as e:
                    print(f"  Failed to unload model {model_id}: {e}")
                    
        except ImportError:
            print("  Model registry not available for unloading models")
        except Exception as e:
            print(f"  Error during model unloading: {e}")
        
        # 检查清理后的内存使用情况
        memory_usage_after = self.check_memory_usage()
        reduction = current_percent - memory_usage_after["percent"]
        
        return {
            "success": True,
            "message": f"Aggressive memory cleanup completed",
            "memory_before": f"{current_percent}%",
            "memory_after": f"{memory_usage_after['percent']}%",
            "reduction": f"{reduction:.1f}%",
            "components_removed": len(components_to_remove),
            "models_unloaded": len(models_to_unload) if 'models_to_unload' in locals() else 0
        }
    
    def set_memory_threshold(self, threshold: int):
        """设置内存使用率阈值"""
        if 10 <= threshold <= 95:
            self.max_memory_usage = threshold
            print(f"Memory optimization threshold set to {threshold}%")
            return True
        else:
            print(f"Warning: Invalid threshold {threshold}%, must be between 10 and 95")
            return False
    
    def enable_lightweight_mode(self, enable: bool = True):
        """启用或禁用轻量模式"""
        self.lightweight_mode = enable
        mode_status = "enabled" if enable else "disabled"
        print(f"Lightweight mode {mode_status}")
        return True

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
                
            # 深拷贝kwargs以避免修改原始参数
            optimized_kwargs = kwargs.copy()
            
            # 如果在轻量模式下，传递轻量参数
            if memory_optimizer.lightweight_mode:
                if hasattr(component_class, 'get_lightweight_params'):
                    try:
                        lightweight_kwargs = component_class.get_lightweight_params()
                        optimized_kwargs.update(lightweight_kwargs)
                    except Exception as e:
                        print(f"Warning: Failed to get lightweight params for {component_class.__name__}: {e}")
                # 为常见模型组件设置轻量参数
                if 'representation_dim' in optimized_kwargs:
                    optimized_kwargs['representation_dim'] = min(optimized_kwargs['representation_dim'], 384)
                if 'num_layers' in optimized_kwargs:
                    optimized_kwargs['num_layers'] = max(1, optimized_kwargs['num_layers'] // 2)
                if 'nhead' in optimized_kwargs:
                    optimized_kwargs['nhead'] = max(4, optimized_kwargs['nhead'] // 2)
                if 'hidden_size' in optimized_kwargs:
                    optimized_kwargs['hidden_size'] = max(32, optimized_kwargs['hidden_size'] // 2)
                if 'num_heads' in optimized_kwargs:
                    optimized_kwargs['num_heads'] = max(2, optimized_kwargs['num_heads'] // 2)
            
            # 创建组件实例
            try:
                component_instance = component_class(*args, **optimized_kwargs)
                global_components_registry[component_name] = component_instance
            except Exception as e:
                print(f"Error: Failed to create component {component_name}: {e}")
                raise
        
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
    max_memory_usage: int = 75,
    aggressive_cleanup_threshold: int = 80,
    enable_auto_cleanup: bool = True,
    agi_mode: bool = False
):
    """配置内存优化参数"""
    memory_optimizer.enable_optimization = enable_optimization
    memory_optimizer.lightweight_mode = lightweight_mode
    memory_optimizer.max_memory_usage = max_memory_usage
    # AGI模式参数暂未使用，为兼容性保留
    
    # 记录配置
    print(f"Memory optimization configured:")
    print(f"  - Optimization enabled: {enable_optimization}")
    print(f"  - Lightweight mode: {lightweight_mode}")
    print(f"  - Max memory usage threshold: {max_memory_usage}%")
    print(f"  - Aggressive cleanup threshold: {aggressive_cleanup_threshold}%")
    print(f"  - Auto cleanup enabled: {enable_auto_cleanup}")

def perform_memory_maintenance():
    """执行内存维护，如果内存使用超过阈值则进行清理"""
    if not memory_optimizer.enable_optimization:
        return {"status": "disabled", "message": "Memory optimization is disabled"}
    
    memory_usage = memory_optimizer.check_memory_usage()
    current_percent = memory_usage["percent"]
    
    if memory_optimizer.should_optimize():
        print(f"⚠️ Memory usage {current_percent}% exceeds threshold {memory_optimizer.max_memory_usage}%, performing optimization")
        memory_optimizer.optimize_memory()
        
        # 检查优化后的内存使用情况
        memory_after = memory_optimizer.check_memory_usage()
        reduction = current_percent - memory_after["percent"]
        
        return {
            "status": "optimized",
            "message": f"Memory optimized: {current_percent}% → {memory_after['percent']}%",
            "reduction": f"{reduction:.1f}%",
            "memory_before": current_percent,
            "memory_after": memory_after["percent"]
        }
    else:
        return {
            "status": "normal",
            "message": f"Memory usage normal: {current_percent}% (threshold: {memory_optimizer.max_memory_usage}%)",
            "memory_usage": current_percent
        }

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
        'max_memory_usage': 75,
        'aggressive_cleanup_threshold': 80,
        'enable_auto_cleanup': True
    }
    
    # 尝试从performance.yml配置文件加载设置
    try:
        import yaml
        config_file_path = config_path or 'config/performance.yml'
        
        if os.path.exists(config_file_path):
            with open(config_file_path, 'r', encoding='utf-8') as f:
                perf_config = yaml.safe_load(f)
            
            # 从性能配置中提取内存优化设置
            if 'memory_optimization' in perf_config:
                mem_config = perf_config['memory_optimization']
                
                # 模型加载策略
                if 'model_loading' in mem_config:
                    loading_config = mem_config['model_loading']
                    if 'memory_threshold' in loading_config:
                        config['max_memory_usage'] = loading_config['memory_threshold']
                
                # 垃圾回收配置
                if 'garbage_collection' in mem_config:
                    gc_config = mem_config['garbage_collection']
                    if 'enabled' in gc_config:
                        config['enable_memory_optimization'] = gc_config['enabled']
                    if 'threshold' in gc_config:
                        config['aggressive_cleanup_threshold'] = gc_config['threshold']
                        
            print(f"Loaded memory optimization settings from {config_file_path}")
        else:
            print(f"Performance config file {config_file_path} not found, using defaults")
            
    except ImportError:
        print("YAML module not available, using default memory optimization settings")
    except Exception as e:
        print(f"Error loading performance configuration: {e}, using defaults")
    
    # 应用配置
    configure_memory_optimization(
        enable_optimization=config['enable_memory_optimization'],
        lightweight_mode=config['lightweight_mode'],
        max_memory_usage=config['max_memory_usage'],
        aggressive_cleanup_threshold=config['aggressive_cleanup_threshold'],
        enable_auto_cleanup=config['enable_auto_cleanup']
    )

# 默认初始化系统配置
initialize_system_configuration()
