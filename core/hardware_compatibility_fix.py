"""
硬件兼容性快速修复模块 - 解决评估报告中的硬件兼容性问题

核心问题修复：
1. 设备自动检测和回退机制不足
2. 硬件抽象层不够完善
3. 协议兼容性问题
4. 资源管理优化不足

关键改进：
- 增强的设备检测与回退
- 统一的硬件抽象接口
- 协议适配和转换层
- 智能资源分配和优化
"""

import logging
import sys
import os
import platform
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import threading
import psutil

logger = logging.getLogger(__name__)


@dataclass
class HardwareDeviceInfo:
    """硬件设备信息"""
    device_id: str
    device_type: str  # cpu, gpu, camera, sensor, motor, etc.
    name: str
    available: bool
    capabilities: Dict[str, Any]
    fallback_devices: List[str] = None  # 备用设备列表
    compatibility_score: float = 1.0  # 兼容性评分 (0-1)
    error_reason: str = ""
    
    def __post_init__(self):
        if self.fallback_devices is None:
            self.fallback_devices = []


@dataclass
class ProtocolInfo:
    """协议信息"""
    protocol_id: str
    protocol_name: str
    supported_platforms: List[str]
    priority: int = 1  # 优先级 (1-10, 10最高)
    fallback_protocols: List[str] = None
    compatibility_mode: bool = False  # 兼容模式
    
    def __post_init__(self):
        if self.fallback_protocols is None:
            self.fallback_protocols = []


class HardwareCompatibilityFix:
    """硬件兼容性快速修复管理器"""
    
    def __init__(self, enable_auto_fallback: bool = True):
        """初始化硬件兼容性修复管理器
        
        Args:
            enable_auto_fallback: 是否启用自动回退机制
        """
        self.enable_auto_fallback = enable_auto_fallback
        self.device_registry: Dict[str, HardwareDeviceInfo] = {}
        self.protocol_registry: Dict[str, ProtocolInfo] = {}
        self.resource_allocations: Dict[str, Any] = {}
        self.compatibility_cache: Dict[str, float] = {}
        
        # 初始化注册表
        self._initialize_device_registry()
        self._initialize_protocol_registry()
        
        # 资源监控
        self.resource_monitor_enabled = True
        self.monitoring_thread = None
        
        logger.info("硬件兼容性快速修复管理器初始化完成")
    
    def _initialize_device_registry(self):
        """初始化设备注册表"""
        try:
            # CPU设备
            cpu_info = HardwareDeviceInfo(
                device_id="cpu_primary",
                device_type="cpu",
                name=f"{platform.processor()} CPU",
                available=True,
                capabilities={
                    "cores": psutil.cpu_count(logical=False),
                    "threads": psutil.cpu_count(logical=True),
                    "memory_gb": psutil.virtual_memory().total / (1024**3),
                    "architecture": platform.machine()
                },
                compatibility_score=1.0
            )
            self.device_registry[cpu_info.device_id] = cpu_info
            
            # GPU设备检测
            self._detect_gpu_devices()
            
            # 内存设备
            memory_info = HardwareDeviceInfo(
                device_id="system_memory",
                device_type="memory",
                name="系统内存",
                available=True,
                capabilities={
                    "total_gb": psutil.virtual_memory().total / (1024**3),
                    "available_gb": psutil.virtual_memory().available / (1024**3),
                    "swap_gb": psutil.swap_memory().total / (1024**3) if hasattr(psutil, 'swap_memory') else 0
                },
                compatibility_score=1.0
            )
            self.device_registry[memory_info.device_id] = memory_info
            
            logger.info(f"初始化了 {len(self.device_registry)} 个设备")
            
        except Exception as e:
            logger.error(f"设备注册表初始化失败: {e}")
    
    def _detect_gpu_devices(self):
        """检测GPU设备"""
        gpu_detection_methods = [
            self._detect_gpu_torch,
            self._detect_gpu_nvidia,
            self._detect_gpu_amd,
            self._detect_gpu_intel,
            self._detect_gpu_apple
        ]
        
        gpu_devices_found = 0
        
        for detection_method in gpu_detection_methods:
            try:
                devices_detected = detection_method()
                gpu_devices_found += len(devices_detected)
            except Exception as e:
                logger.debug(f"GPU检测方法 {detection_method.__name__} 失败: {e}")
        
        if gpu_devices_found == 0:
            logger.warning("未检测到GPU设备，将使用CPU回退模式")
    
    def _detect_gpu_torch(self) -> List[HardwareDeviceInfo]:
        """使用PyTorch检测GPU"""
        devices = []
        try:
            import torch
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device_name = torch.cuda.get_device_name(i)
                    device_props = torch.cuda.get_device_properties(i)
                    
                    gpu_info = HardwareDeviceInfo(
                        device_id=f"cuda_gpu_{i}",
                        device_type="gpu",
                        name=f"NVIDIA {device_name}",
                        available=True,
                        capabilities={
                            "memory_gb": device_props.total_memory / (1024**3),
                            "compute_capability": f"{device_props.major}.{device_props.minor}",
                            "multiprocessors": device_props.multi_processor_count,
                            "cuda_support": True,
                            "device_index": i
                        },
                        compatibility_score=0.9,  # CUDA通常兼容性好
                        fallback_devices=["cpu_primary"]
                    )
                    devices.append(gpu_info)
                    self.device_registry[gpu_info.device_id] = gpu_info
            
            # 检测Apple Silicon GPU (MPS)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                mps_info = HardwareDeviceInfo(
                    device_id="apple_silicon_gpu",
                    device_type="gpu",
                    name="Apple Silicon GPU",
                    available=True,
                    capabilities={
                        "mps_support": True,
                        "architecture": "arm64"
                    },
                    compatibility_score=0.85,
                    fallback_devices=["cpu_primary"]
                )
                devices.append(mps_info)
                self.device_registry[mps_info.device_id] = mps_info
                
        except ImportError:
            logger.warning("PyTorch未安装，跳过GPU检测")
        
        return devices
    
    def _detect_gpu_nvidia(self) -> List[HardwareDeviceInfo]:
        """检测NVIDIA GPU (备用方法)"""
        devices = []
        try:
            # 尝试使用nvidia-smi
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            memory = parts[1].strip()
                            
                            gpu_info = HardwareDeviceInfo(
                                device_id=f"nvidia_gpu_{i}",
                                device_type="gpu",
                                name=f"NVIDIA {name}",
                                available=True,
                                capabilities={
                                    "memory": memory,
                                    "vendor": "nvidia",
                                    "detection_method": "nvidia-smi"
                                },
                                compatibility_score=0.8,
                                fallback_devices=["cpu_primary"]
                            )
                            devices.append(gpu_info)
                            self.device_registry[gpu_info.device_id] = gpu_info
                            
        except Exception as e:
            logger.debug(f"NVIDIA GPU检测失败: {e}")
        
        return devices
    
    def _detect_gpu_amd(self) -> List[HardwareDeviceInfo]:
        """检测AMD GPU"""
        # AMD检测实现占位符
        return []
    
    def _detect_gpu_intel(self) -> List[HardwareDeviceInfo]:
        """检测Intel GPU"""
        # Intel检测实现占位符
        return []
    
    def _detect_gpu_apple(self) -> List[HardwareDeviceInfo]:
        """检测Apple GPU (除MPS外的其他方法)"""
        # Apple GPU检测实现占位符
        return []
    
    def _initialize_protocol_registry(self):
        """初始化协议注册表"""
        protocols = [
            ProtocolInfo(
                protocol_id="ethernet_tcp",
                protocol_name="以太网 TCP/IP",
                supported_platforms=["linux", "windows", "macos"],
                priority=10,
                fallback_protocols=["ethernet_udp", "serial"]
            ),
            ProtocolInfo(
                protocol_id="ethernet_udp",
                protocol_name="以太网 UDP",
                supported_platforms=["linux", "windows", "macos"],
                priority=8,
                fallback_protocols=["ethernet_tcp"]
            ),
            ProtocolInfo(
                protocol_id="serial_rs232",
                protocol_name="串口 RS-232",
                supported_platforms=["linux", "windows", "macos"],
                priority=6,
                compatibility_mode=True  # 旧设备兼容模式
            ),
            ProtocolInfo(
                protocol_id="serial_usb",
                protocol_name="USB串口",
                supported_platforms=["linux", "windows", "macos"],
                priority=7
            ),
            ProtocolInfo(
                protocol_id="i2c",
                protocol_name="I2C总线",
                supported_platforms=["linux", "raspberry_pi"],
                priority=5,
                fallback_protocols=["spi", "serial"]
            ),
            ProtocolInfo(
                protocol_id="spi",
                protocol_name="SPI总线",
                supported_platforms=["linux", "raspberry_pi"],
                priority=5
            ),
            ProtocolInfo(
                protocol_id="can_bus",
                protocol_name="CAN总线",
                supported_platforms=["linux", "embedded"],
                priority=4,
                compatibility_mode=True
            ),
            ProtocolInfo(
                protocol_id="pwm",
                protocol_name="PWM控制",
                supported_platforms=["linux", "raspberry_pi", "arduino"],
                priority=3
            ),
            ProtocolInfo(
                protocol_id="ros",
                protocol_name="ROS协议",
                supported_platforms=["linux"],
                priority=9,
                fallback_protocols=["ethernet_tcp"]
            ),
            ProtocolInfo(
                protocol_id="ros2",
                protocol_name="ROS2协议",
                supported_platforms=["linux", "windows", "macos"],
                priority=9
            )
        ]
        
        for protocol in protocols:
            self.protocol_registry[protocol.protocol_id] = protocol
        
        logger.info(f"初始化了 {len(self.protocol_registry)} 个协议")
    
    def get_compatible_device(self, device_type: str, requirements: Dict[str, Any] = None) -> Optional[HardwareDeviceInfo]:
        """获取兼容的设备
        
        Args:
            device_type: 设备类型
            requirements: 设备要求
            
        Returns:
            兼容的设备信息，如果没有则返回None
        """
        requirements = requirements or {}
        
        # 查找匹配的设备
        compatible_devices = []
        
        for device_id, device_info in self.device_registry.items():
            if device_info.device_type != device_type:
                continue
            
            if not device_info.available:
                continue
            
            # 检查兼容性
            compatibility = self._check_device_compatibility(device_info, requirements)
            
            if compatibility >= 0.5:  # 兼容性阈值
                compatible_devices.append((device_info, compatibility))
        
        if not compatible_devices:
            logger.warning(f"未找到兼容的 {device_type} 设备")
            return None
        
        # 按兼容性排序
        compatible_devices.sort(key=lambda x: x[1], reverse=True)
        
        best_device, best_compatibility = compatible_devices[0]
        
        logger.info(f"选择设备: {best_device.name} (兼容性: {best_compatibility:.2f})")
        
        # 缓存兼容性分数
        self.compatibility_cache[best_device.device_id] = best_compatibility
        
        return best_device
    
    def _check_device_compatibility(self, device: HardwareDeviceInfo, requirements: Dict[str, Any]) -> float:
        """检查设备兼容性
        
        Args:
            device: 设备信息
            requirements: 设备要求
            
        Returns:
            兼容性分数 (0-1)
        """
        base_compatibility = device.compatibility_score
        
        # 检查设备能力是否满足要求
        capability_match = 1.0
        
        for req_key, req_value in requirements.items():
            if req_key in device.capabilities:
                device_value = device.capabilities[req_key]
                
                # 简单匹配逻辑
                if isinstance(req_value, (int, float)) and isinstance(device_value, (int, float)):
                    if req_value <= device_value:
                        match_score = min(1.0, device_value / max(req_value, 1))
                    else:
                        match_score = 0.3  # 设备能力不足
                else:
                    # 非数值匹配
                    match_score = 1.0 if str(device_value) == str(req_value) else 0.5
                
                capability_match = min(capability_match, match_score)
        
        # 综合兼容性分数
        compatibility = base_compatibility * capability_match
        
        return compatibility
    
    def get_compatible_protocol(self, device_type: str, platform: str = None) -> Optional[ProtocolInfo]:
        """获取兼容的协议
        
        Args:
            device_type: 设备类型
            platform: 目标平台
            
        Returns:
            兼容的协议信息
        """
        platform = platform or sys.platform
        
        compatible_protocols = []
        
        for protocol_id, protocol_info in self.protocol_registry.items():
            # 检查平台支持
            if platform not in protocol_info.supported_platforms:
                continue
            
            # 设备类型特定逻辑
            if device_type == "gpu":
                # GPU通常使用高速协议
                if protocol_id in ["ethernet_tcp", "ethernet_udp", "pcie"]:
                    compatible_protocols.append((protocol_info, protocol_info.priority))
            elif device_type in ["sensor", "camera"]:
                if protocol_id in ["i2c", "spi", "serial_usb", "ethernet_tcp"]:
                    compatible_protocols.append((protocol_info, protocol_info.priority))
            elif device_type in ["motor", "servo"]:
                if protocol_id in ["pwm", "serial_rs232", "can_bus", "ethernet_tcp"]:
                    compatible_protocols.append((protocol_info, protocol_info.priority))
            else:
                # 通用设备
                compatible_protocols.append((protocol_info, protocol_info.priority))
        
        if not compatible_protocols:
            logger.warning(f"未找到兼容的协议用于 {device_type} 设备")
            return None
        
        # 按优先级排序
        compatible_protocols.sort(key=lambda x: x[1], reverse=True)
        
        best_protocol, priority = compatible_protocols[0]
        
        logger.info(f"选择协议: {best_protocol.protocol_name} (优先级: {priority})")
        
        return best_protocol
    
    def allocate_resources(self, task_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """分配系统资源
        
        Args:
            task_requirements: 任务资源要求
            
        Returns:
            资源分配结果
        """
        try:
            # 分析当前资源状态
            resource_status = self._analyze_resource_status()
            
            # 资源分配策略
            allocation = {
                "cpu_cores": self._allocate_cpu_cores(task_requirements, resource_status),
                "memory_gb": self._allocate_memory(task_requirements, resource_status),
                "gpu_devices": self._allocate_gpu_devices(task_requirements, resource_status),
                "network_bandwidth": self._allocate_network_bandwidth(task_requirements, resource_status),
                "storage_space": self._allocate_storage_space(task_requirements, resource_status)
            }
            
            # 验证分配
            if self._validate_resource_allocation(allocation, resource_status):
                # 记录分配
                allocation_id = f"alloc_{int(time.time())}"
                self.resource_allocations[allocation_id] = {
                    "allocation": allocation,
                    "timestamp": time.time(),
                    "task_requirements": task_requirements
                }
                
                logger.info(f"资源分配成功: {allocation_id}")
                
                return {
                    "success": True,
                    "allocation_id": allocation_id,
                    "allocation": allocation,
                    "resource_status": resource_status
                }
            else:
                logger.warning("资源分配验证失败")
                return {
                    "success": False,
                    "error": "资源分配验证失败",
                    "suggestions": self._generate_resource_suggestions(task_requirements, resource_status)
                }
                
        except Exception as e:
            logger.error(f"资源分配失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _analyze_resource_status(self) -> Dict[str, Any]:
        """分析当前资源状态"""
        try:
            import psutil
            
            # CPU状态
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            
            # 内存状态
            memory = psutil.virtual_memory()
            
            # 磁盘状态
            disk = psutil.disk_usage('/')
            
            # 网络状态
            net_io = psutil.net_io_counters()
            
            return {
                "cpu": {
                    "cores_physical": psutil.cpu_count(logical=False),
                    "cores_logical": psutil.cpu_count(logical=True),
                    "usage_percent": cpu_percent,
                    "frequency_mhz": cpu_freq.current if cpu_freq else None
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_percent": memory.percent
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "used_percent": disk.percent
                },
                "network": {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv
                }
            }
            
        except Exception as e:
            logger.error(f"资源状态分析失败: {e}")
            return {}
    
    def _allocate_cpu_cores(self, requirements: Dict[str, Any], status: Dict[str, Any]) -> int:
        """分配CPU核心"""
        requested_cores = requirements.get("cpu_cores", 1)
        available_cores = status.get("cpu", {}).get("cores_logical", 1)
        cpu_usage = status.get("cpu", {}).get("usage_percent", 0)
        
        # 考虑CPU使用率
        available_for_allocation = max(1, int(available_cores * (1 - cpu_usage / 100)))
        
        allocated_cores = min(requested_cores, available_for_allocation)
        
        return allocated_cores
    
    def _allocate_memory(self, requirements: Dict[str, Any], status: Dict[str, Any]) -> float:
        """分配内存 (GB)"""
        requested_memory = requirements.get("memory_gb", 1.0)
        available_memory = status.get("memory", {}).get("available_gb", 1.0)
        
        # 保留系统内存
        system_reserve = 2.0  # 保留2GB给系统
        allocatable_memory = max(0.5, available_memory - system_reserve)
        
        allocated_memory = min(requested_memory, allocatable_memory)
        
        return allocated_memory
    
    def _allocate_gpu_devices(self, requirements: Dict[str, Any], status: Dict[str, Any]) -> List[str]:
        """分配GPU设备"""
        requested_gpus = requirements.get("gpu_count", 0)
        
        if requested_gpus <= 0:
            return []
        
        # 查找可用的GPU设备
        available_gpus = []
        for device_id, device_info in self.device_registry.items():
            if device_info.device_type == "gpu" and device_info.available:
                available_gpus.append(device_id)
        
        # 按兼容性排序
        sorted_gpus = sorted(
            available_gpus,
            key=lambda x: self.compatibility_cache.get(x, 0.5),
            reverse=True
        )
        
        allocated_gpus = sorted_gpus[:min(requested_gpus, len(sorted_gpus))]
        
        return allocated_gpus
    
    def _allocate_network_bandwidth(self, requirements: Dict[str, Any], status: Dict[str, Any]) -> float:
        """分配网络带宽 (Mbps)"""
        requested_bandwidth = requirements.get("network_bandwidth_mbps", 10.0)
        
        # 简化分配：返回请求值，实际系统中需要更复杂的网络监控
        return min(requested_bandwidth, 1000.0)  # 限制为1Gbps
    
    def _allocate_storage_space(self, requirements: Dict[str, Any], status: Dict[str, Any]) -> float:
        """分配存储空间 (GB)"""
        requested_storage = requirements.get("storage_gb", 10.0)
        available_storage = status.get("disk", {}).get("free_gb", 50.0)
        
        # 保留系统存储
        system_reserve = 10.0  # 保留10GB给系统
        allocatable_storage = max(1.0, available_storage - system_reserve)
        
        allocated_storage = min(requested_storage, allocatable_storage)
        
        return allocated_storage
    
    def _validate_resource_allocation(self, allocation: Dict[str, Any], status: Dict[str, Any]) -> bool:
        """验证资源分配是否合理"""
        try:
            # 检查CPU分配
            cpu_allocated = allocation.get("cpu_cores", 0)
            cpu_available = status.get("cpu", {}).get("cores_logical", 1)
            cpu_usage = status.get("cpu", {}).get("usage_percent", 0)
            
            if cpu_allocated > cpu_available * 0.8:  # 不超过80%的CPU核心
                logger.warning(f"CPU分配过多: {cpu_allocated}/{cpu_available}")
                return False
            
            # 检查内存分配
            memory_allocated = allocation.get("memory_gb", 0)
            memory_available = status.get("memory", {}).get("available_gb", 1)
            
            if memory_allocated > memory_available * 0.7:  # 不超过70%的可用内存
                logger.warning(f"内存分配过多: {memory_allocated:.1f}/{memory_available:.1f} GB")
                return False
            
            # 检查存储分配
            storage_allocated = allocation.get("storage_space", 0)
            storage_available = status.get("disk", {}).get("free_gb", 1)
            
            if storage_allocated > storage_available * 0.8:  # 不超过80%的可用存储
                logger.warning(f"存储分配过多: {storage_allocated:.1f}/{storage_available:.1f} GB")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"资源分配验证失败: {e}")
            return False
    
    def _generate_resource_suggestions(self, requirements: Dict[str, Any], status: Dict[str, Any]) -> List[str]:
        """生成资源建议"""
        suggestions = []
        
        # CPU建议
        cpu_required = requirements.get("cpu_cores", 1)
        cpu_available = status.get("cpu", {}).get("cores_logical", 1)
        
        if cpu_required > cpu_available:
            suggestions.append(f"CPU需求过高: 需要 {cpu_required} 核心，但只有 {cpu_available} 个可用")
            suggestions.append("建议: 减少CPU核心需求或升级硬件")
        
        # 内存建议
        memory_required = requirements.get("memory_gb", 1.0)
        memory_available = status.get("memory", {}).get("available_gb", 1.0)
        
        if memory_required > memory_available * 0.7:
            suggestions.append(f"内存需求过高: 需要 {memory_required:.1f} GB，但只有 {memory_available:.1f} GB 可用")
            suggestions.append("建议: 减少内存需求或增加系统内存")
        
        # 存储建议
        storage_required = requirements.get("storage_gb", 10.0)
        storage_available = status.get("disk", {}).get("free_gb", 50.0)
        
        if storage_required > storage_available * 0.8:
            suggestions.append(f"存储需求过高: 需要 {storage_required:.1f} GB，但只有 {storage_available:.1f} GB 可用")
            suggestions.append("建议: 清理磁盘空间或增加存储容量")
        
        return suggestions
    
    def get_system_compatibility_report(self) -> Dict[str, Any]:
        """获取系统兼容性报告"""
        report = {
            "timestamp": time.time(),
            "system_info": {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "device_compatibility": {},
            "protocol_compatibility": {},
            "resource_status": self._analyze_resource_status(),
            "issues_found": [],
            "recommendations": []
        }
        
        # 设备兼容性分析
        for device_id, device_info in self.device_registry.items():
            report["device_compatibility"][device_id] = {
                "name": device_info.name,
                "type": device_info.device_type,
                "available": device_info.available,
                "compatibility_score": device_info.compatibility_score,
                "error_reason": device_info.error_reason
            }
            
            if not device_info.available and device_info.error_reason:
                report["issues_found"].append(f"设备不可用: {device_info.name} - {device_info.error_reason}")
        
        # 协议兼容性分析
        current_platform = platform.system().lower()
        
        for protocol_id, protocol_info in self.protocol_registry.items():
            platform_supported = current_platform in protocol_info.supported_platforms
            
            report["protocol_compatibility"][protocol_id] = {
                "name": protocol_info.protocol_name,
                "platform_supported": platform_supported,
                "priority": protocol_info.priority,
                "compatibility_mode": protocol_info.compatibility_mode
            }
            
            if not platform_supported:
                report["issues_found"].append(f"协议不受支持: {protocol_info.protocol_name} 在当前平台 {current_platform}")
        
        # 生成建议
        if not report["issues_found"]:
            report["recommendations"].append("系统兼容性良好，未发现重大问题")
        else:
            report["recommendations"].append(f"发现 {len(report['issues_found'])} 个兼容性问题，建议修复")
        
        # 资源建议
        resource_status = report["resource_status"]
        if resource_status.get("memory", {}).get("used_percent", 0) > 90:
            report["recommendations"].append("内存使用率过高，建议优化或增加内存")
        
        if resource_status.get("disk", {}).get("used_percent", 0) > 90:
            report["recommendations"].append("磁盘使用率过高，建议清理空间")
        
        return report


def create_hardware_compatibility_fix() -> HardwareCompatibilityFix:
    """创建硬件兼容性修复管理器实例（工厂函数）"""
    return HardwareCompatibilityFix(enable_auto_fallback=True)