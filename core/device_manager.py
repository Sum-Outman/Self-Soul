"""
设备管理器 - Device Manager
用于检测和管理CPU/GPU设备，支持训练过程中的设备切换
For detecting and managing CPU/GPU devices, supporting device switching during training
"""

import logging
import torch
import psutil
import platform
from typing import Dict, Any, List, Optional


class DeviceManager:
    """设备管理器类 | Device Manager Class
    
    功能：检测和管理CPU/GPU设备，支持训练过程中的设备切换
    Function: Detect and manage CPU/GPU devices, support device switching during training
    """
    
    def __init__(self):
        """初始化设备管理器 | Initialize device manager"""
        self.logger = logging.getLogger(__name__)
        self.available_devices = {}
        self.current_device = None
        self.device_memory = {}
        
        # 检测可用设备
        self._detect_devices()
        
        self.logger.info("设备管理器初始化完成 | Device manager initialized")
    
    def _detect_devices(self):
        """检测可用设备 | Detect available devices"""
        try:
            # CPU设备
            cpu_info = {
                "name": "CPU",
                "type": "cpu",
                "available": True,
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "platform": platform.system()
            }
            self.available_devices["cpu"] = cpu_info
            
            # CUDA设备 (NVIDIA GPU)
            if torch.cuda.is_available():
                cuda_info = {
                    "name": "CUDA",
                    "type": "cuda",
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "devices": []
                }
                
                for i in range(torch.cuda.device_count()):
                    device_props = {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3),
                        "compute_capability": f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
                    }
                    cuda_info["devices"].append(device_props)
                
                self.available_devices["cuda"] = cuda_info
            else:
                self.available_devices["cuda"] = {
                    "name": "CUDA",
                    "type": "cuda",
                    "available": False,
                    "reason": "CUDA not available"
                }
            
            # MPS设备 (Apple Silicon GPU)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                mps_info = {
                    "name": "MPS",
                    "type": "mps",
                    "available": True,
                    "device_name": "Apple Silicon GPU"
                }
                self.available_devices["mps"] = mps_info
            else:
                self.available_devices["mps"] = {
                    "name": "MPS",
                    "type": "mps",
                    "available": False,
                    "reason": "MPS not available"
                }
            
            # 设置默认设备
            self.current_device = self._get_best_device()
            
            self.logger.info(f"检测到设备: {list(self.available_devices.keys())}")
            self.logger.info(f"默认设备: {self.current_device}")
            
        except Exception as e:
            self.logger.error(f"设备检测失败: {e}")
            # 设置默认CPU设备
            self.available_devices["cpu"] = {
                "name": "CPU",
                "type": "cpu",
                "available": True
            }
            self.current_device = "cpu"
    
    def _get_best_device(self) -> str:
        """获取最佳可用设备 | Get best available device
        
        Returns:
            最佳设备类型
        """
        # 优先级: CUDA > MPS > CPU
        if self.available_devices.get("cuda", {}).get("available", False):
            return "cuda"
        elif self.available_devices.get("mps", {}).get("available", False):
            return "mps"
        else:
            return "cpu"
    
    def get_available_devices(self) -> Dict[str, bool]:
        """获取可用设备状态 | Get available device status
        
        Returns:
            设备可用性字典
        """
        return {
            device_type: device_info.get("available", False)
            for device_type, device_info in self.available_devices.items()
        }
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """获取硬件配置 | Get hardware configuration
        
        Returns:
            硬件配置信息
        """
        return {
            "available_devices": self.available_devices,
            "current_device": self.current_device,
            "device_memory": self.device_memory,
            "total_devices": len(self.available_devices),
            "has_cuda": torch.cuda.is_available() if hasattr(torch.cuda, 'is_available') else False,
            "has_cpu": True,
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__ if hasattr(torch, '__version__') else "unknown"
        }
    
    def get_device_info(self, device_type: str) -> Dict[str, Any]:
        """获取设备详细信息 | Get detailed device information
        
        Args:
            device_type: 设备类型
            
        Returns:
            设备详细信息
        """
        return self.available_devices.get(device_type, {})
    
    def get_current_device(self) -> str:
        """获取当前设备 | Get current device
        
        Returns:
            当前设备类型
        """
        return self.current_device
    
    def set_device(self, device_type: str) -> Dict[str, Any]:
        """设置设备 | Set device
        
        Args:
            device_type: 设备类型
            
        Returns:
            设置结果
        """
        try:
            if device_type not in self.available_devices:
                return {
                    "success": False,
                    "message": f"未知设备类型: {device_type}",
                    "available_devices": list(self.available_devices.keys())
                }
            
            if not self.available_devices[device_type].get("available", False):
                return {
                    "success": False,
                    "message": f"设备 {device_type} 不可用",
                    "reason": self.available_devices[device_type].get("reason", "Unknown reason")
                }
            
            # 设置设备
            self.current_device = device_type
            
            self.logger.info(f"设备已设置为: {device_type}")
            
            return {
                "success": True,
                "message": f"设备已设置为 {device_type}",
                "device": device_type,
                "device_info": self.available_devices[device_type]
            }
            
        except Exception as e:
            self.logger.error(f"设置设备失败: {e}")
            return {
                "success": False,
                "message": f"设置设备失败: {str(e)}"
            }
    
    def get_torch_device(self) -> torch.device:
        """获取PyTorch设备对象 | Get PyTorch device object
        
        Returns:
            PyTorch设备对象
        """
        if self.current_device == "cuda":
            return torch.device("cuda")
        elif self.current_device == "mps":
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def get_device_memory_info(self) -> Dict[str, Any]:
        """获取设备内存信息 | Get device memory information
        
        Returns:
            内存信息字典
        """
        memory_info = {}
        
        try:
            # CPU内存
            cpu_memory = psutil.virtual_memory()
            memory_info["cpu"] = {
                "total_gb": cpu_memory.total / (1024**3),
                "available_gb": cpu_memory.available / (1024**3),
                "used_gb": cpu_memory.used / (1024**3),
                "percent": cpu_memory.percent
            }
            
            # GPU内存 (如果可用)
            if self.available_devices.get("cuda", {}).get("available", False):
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    memory_cached = torch.cuda.memory_reserved(i) / (1024**3)
                    memory_info[f"cuda_{i}"] = {
                        "allocated_gb": memory_allocated,
                        "cached_gb": memory_cached,
                        "total_gb": self.available_devices["cuda"]["devices"][i]["memory_gb"]
                    }
            
            # MPS内存 (如果可用)
            if self.available_devices.get("mps", {}).get("available", False):
                # MPS内存信息有限，使用系统内存作为参考
                memory_info["mps"] = {
                    "total_gb": psutil.virtual_memory().total / (1024**3),
                    "available_gb": psutil.virtual_memory().available / (1024**3)
                }
            
        except Exception as e:
            self.logger.error(f"获取内存信息失败: {e}")
        
        return memory_info
    
    def optimize_for_device(self, device_type: str) -> Dict[str, Any]:
        """为特定设备优化配置 | Optimize configuration for specific device
        
        Args:
            device_type: 设备类型
            
        Returns:
            优化配置
        """
        optimizations = {
            "cpu": {
                "batch_size": 16,
                "mixed_precision": False,
                "gradient_accumulation": 1,
                "num_workers": 4,
                "pin_memory": False,
                "recommendations": [
                    "使用较小的批次大小",
                    "禁用混合精度训练",
                    "使用多个数据加载器进程"
                ]
            },
            "cuda": {
                "batch_size": 64,
                "mixed_precision": True,
                "gradient_accumulation": 4,
                "num_workers": 8,
                "pin_memory": True,
                "recommendations": [
                    "使用较大的批次大小",
                    "启用混合精度训练",
                    "使用梯度累积",
                    "启用内存固定"
                ]
            },
            "mps": {
                "batch_size": 32,
                "mixed_precision": True,
                "gradient_accumulation": 2,
                "num_workers": 4,
                "pin_memory": False,
                "recommendations": [
                    "使用中等批次大小",
                    "启用混合精度训练",
                    "使用适度的梯度累积"
                ]
            },
            "auto": {
                "batch_size": 32,
                "mixed_precision": True,
                "gradient_accumulation": 2,
                "num_workers": 4,
                "pin_memory": True,
                "recommendations": [
                    "自动优化配置",
                    "根据可用设备调整参数"
                ]
            }
        }
        
        optimization = optimizations.get(device_type, optimizations["auto"])
        
        return {
            "success": True,
            "device": device_type,
            "optimization": optimization,
            "message": f"已为{device_type}设备优化配置"
        }
    
    def transfer_model_to_device(self, model, device_type: str) -> Dict[str, Any]:
        """将模型转移到指定设备 | Transfer model to specified device
        
        Args:
            model: PyTorch模型
            device_type: 目标设备类型
            
        Returns:
            转移结果
        """
        try:
            if device_type not in self.available_devices:
                return {
                    "success": False,
                    "message": f"未知设备类型: {device_type}"
                }
            
            if not self.available_devices[device_type].get("available", False):
                return {
                    "success": False,
                    "message": f"设备 {device_type} 不可用"
                }
            
            # 获取目标设备
            target_device = self.get_torch_device()
            
            # 转移模型到目标设备
            model.to(target_device)
            
            self.logger.info(f"模型已转移到设备: {device_type}")
            
            return {
                "success": True,
                "message": f"模型已转移到{device_type}设备",
                "device": device_type,
                "model_device": str(target_device)
            }
            
        except Exception as e:
            self.logger.error(f"模型转移失败: {e}")
            return {
                "success": False,
                "message": f"模型转移失败: {str(e)}"
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息 | Get system information
        
        Returns:
            系统信息字典
        """
        try:
            system_info = {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "architecture": platform.architecture(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "Not available",
                "available_devices": self.get_available_devices(),
                "current_device": self.current_device,
                "memory_info": self.get_device_memory_info()
            }
            
            return system_info
            
        except Exception as e:
            self.logger.error(f"获取系统信息失败: {e}")
            return {
                "error": f"获取系统信息失败: {str(e)}"
            }


# 全局设备管理器实例
_device_manager = None

def get_device_manager() -> DeviceManager:
    """获取设备管理器实例 | Get device manager instance
    
    Returns:
        设备管理器实例
    """
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager