#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
崩溃快照管理器 - Crash Snapshot Manager

解决用户指出的"无崩溃快照"问题：
系统崩溃时仅输出"崩溃"，不保存崩溃前的上下文、硬件状态、演化进度

本模块提供：
1. 系统崩溃时自动捕获关键状态信息
2. 保存上下文、硬件状态、演化进度到快照文件
3. 支持手动触发快照（用于测试和调试）
4. 快照文件管理和分析工具
"""

import os
import sys
import json
import traceback
import signal
import atexit
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from collections import defaultdict
import psutil
import inspect

logger = logging.getLogger(__name__)


class CrashSnapshotManager:
    """
    崩溃快照管理器
    在系统崩溃时捕获和保存关键系统状态
    """
    
    def __init__(
        self,
        snapshot_dir: Optional[str] = None,
        max_snapshots: int = 50,
        enable_auto_capture: bool = True
    ):
        """
        初始化崩溃快照管理器
        
        Args:
            snapshot_dir: 快照保存目录
            max_snapshots: 最大快照数量
            enable_auto_capture: 是否启用自动捕获
        """
        # 确定快照目录
        if snapshot_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            snapshot_dir = os.path.join(base_dir, "crash_snapshots")
        
        self.snapshot_dir = snapshot_dir
        self.max_snapshots = max_snapshots
        self.enable_auto_capture = enable_auto_capture
        
        # 确保快照目录存在
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # 系统组件引用（延迟设置）
        self.context_memory = None
        self.evolution_engine = None
        self.core_metrics_collector = None
        self.scene_adaptive_params = None
        self.performance_dashboard = None
        
        # 注册的组件列表
        self.registered_components = {}
        
        # 设置信号处理器和退出处理器
        if self.enable_auto_capture:
            self._setup_signal_handlers()
            self._setup_atexit_handler()
        
        # 快照计数器
        self.snapshot_counter = 0
        
        logger.info(f"CrashSnapshotManager initialized (snapshot_dir: {self.snapshot_dir})")
    
    def _setup_signal_handlers(self) -> None:
        """设置信号处理器"""
        # 捕获常见崩溃信号
        signals = [
            signal.SIGINT,    # Ctrl+C
            signal.SIGTERM,   # 终止信号
            signal.SIGSEGV,   # 段错误
            signal.SIGABRT,   # 中止信号
            signal.SIGFPE,    # 浮点异常
            signal.SIGILL,    # 非法指令
        ]
        
        original_handlers = {}
        
        def signal_handler(sig, frame):
            """统一信号处理器"""
            signal_name = signal.Signals(sig).name if hasattr(signal, 'Signals') else str(sig)
            logger.critical(f"Received signal {signal_name}, creating crash snapshot...")
            
            # 恢复原始处理器
            if sig in original_handlers:
                signal.signal(sig, original_handlers[sig])
            
            # 创建崩溃快照
            snapshot_data = self.create_crash_snapshot(
                crash_type=f"signal_{signal_name}",
                signal_info={"signal": sig, "signal_name": signal_name},
                frame_info=frame
            )
            
            # 保存快照
            snapshot_file = self.save_snapshot(snapshot_data, prefix=f"signal_{signal_name}")
            
            logger.critical(f"Crash snapshot saved to: {snapshot_file}")
            
            # 重新引发信号
            if sig == signal.SIGINT:
                raise KeyboardInterrupt
            else:
                os.kill(os.getpid(), sig)
        
        # 注册信号处理器
        for sig in signals:
            try:
                original_handlers[sig] = signal.signal(sig, signal_handler)
            except (ValueError, OSError):
                # 某些信号在某些平台上可能不可用
                pass
        
        self.original_signal_handlers = original_handlers
    
    def _setup_atexit_handler(self) -> None:
        """设置退出处理器"""
        def atexit_handler():
            """退出时处理器"""
            # 检查是否因异常退出
            exc_info = sys.exc_info()
            if exc_info[0] is not None:
                # 因异常退出，创建崩溃快照
                logger.critical(f"Exiting due to exception: {exc_info[0].__name__}: {exc_info[1]}")
                
                snapshot_data = self.create_crash_snapshot(
                    crash_type=f"exception_{exc_info[0].__name__}",
                    exception_info={
                        "type": exc_info[0].__name__,
                        "message": str(exc_info[1]),
                        "traceback": traceback.format_exception(*exc_info)
                    }
                )
                
                snapshot_file = self.save_snapshot(snapshot_data, prefix=f"exception_{exc_info[0].__name__}")
                logger.critical(f"Crash snapshot saved to: {snapshot_file}")
        
        atexit.register(atexit_handler)
    
    def register_component(self, component_name: str, component: Any) -> None:
        """
        注册系统组件
        
        Args:
            component_name: 组件名称
            component: 组件实例
        """
        self.registered_components[component_name] = component
        
        # 同时设置特定属性以便快速访问
        if component_name == "context_memory":
            self.context_memory = component
        elif component_name == "evolution_engine":
            self.evolution_engine = component
        elif component_name == "core_metrics_collector":
            self.core_metrics_collector = component
        elif component_name == "scene_adaptive_params":
            self.scene_adaptive_params = component
        elif component_name == "performance_dashboard":
            self.performance_dashboard = component
        
        logger.debug(f"Component registered: {component_name}")
    
    def create_crash_snapshot(
        self,
        crash_type: str = "manual",
        exception_info: Optional[Dict[str, Any]] = None,
        signal_info: Optional[Dict[str, Any]] = None,
        frame_info: Any = None
    ) -> Dict[str, Any]:
        """
        创建崩溃快照
        
        Args:
            crash_type: 崩溃类型
            exception_info: 异常信息
            signal_info: 信号信息
            frame_info: 帧信息
            
        Returns:
            崩溃快照数据
        """
        snapshot_start_time = time.time()
        logger.info(f"Creating crash snapshot (type: {crash_type})...")
        
        snapshot_data = {
            "metadata": self._collect_metadata(crash_type),
            "system_state": self._collect_system_state(),
            "hardware_state": self._collect_hardware_state(),
            "context_state": self._collect_context_state(),
            "evolution_state": self._collect_evolution_state(),
            "metrics_state": self._collect_metrics_state(),
            "crash_info": self._collect_crash_info(exception_info, signal_info, frame_info),
            "registered_components": list(self.registered_components.keys())
        }
        
        snapshot_duration = time.time() - snapshot_start_time
        snapshot_data["metadata"]["snapshot_creation_time_seconds"] = snapshot_duration
        
        logger.info(f"Crash snapshot created in {snapshot_duration:.3f} seconds")
        return snapshot_data
    
    def _collect_metadata(self, crash_type: str) -> Dict[str, Any]:
        """收集元数据"""
        return {
            "snapshot_type": "crash_snapshot",
            "crash_type": crash_type,
            "timestamp": datetime.now().isoformat(),
            "timestamp_unix": time.time(),
            "pid": os.getpid(),
            "process_name": psutil.Process().name(),
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": os.getcwd(),
            "script_path": sys.argv[0] if len(sys.argv) > 0 else None,
            "command_line_args": sys.argv[1:] if len(sys.argv) > 1 else []
        }
    
    def _collect_system_state(self) -> Dict[str, Any]:
        """收集系统状态"""
        try:
            process = psutil.Process()
            
            # 进程信息
            process_info = {
                "status": process.status(),
                "create_time": process.create_time(),
                "cpu_times": process.cpu_times()._asdict(),
                "memory_info": process.memory_info()._asdict(),
                "num_threads": process.num_threads(),
                "num_fds": process.num_fds() if hasattr(process, 'num_fds') else None,
                "connections": len(process.connections()) if hasattr(process, 'connections') else 0
            }
            
            # 环境信息
            env_vars = {}
            for key in os.environ:
                if not key.startswith(('SECRET_', 'PASSWORD_', 'KEY_', 'TOKEN_')):
                    env_vars[key] = os.environ[key]
            
            # 线程信息
            threads = []
            for thread_id, thread in threading._active.items():
                if thread is threading.current_thread():
                    continue
                
                thread_info = {
                    "thread_id": thread_id,
                    "name": thread.name,
                    "daemon": thread.daemon,
                    "alive": thread.is_alive()
                }
                
                # 获取线程堆栈
                try:
                    frame = sys._current_frames().get(thread_id)
                    if frame:
                        stack = traceback.extract_stack(frame)
                        thread_info["stack"] = [{"filename": s.filename, "lineno": s.lineno, "name": s.name, "line": s.line} for s in stack[-10:]]
                except:
                    pass
                
                threads.append(thread_info)
            
            return {
                "process": process_info,
                "environment": env_vars,
                "threads": threads,
                "imported_modules": list(sys.modules.keys()),
                "python_path": sys.path
            }
            
        except Exception as e:
            logger.warning(f"Failed to collect system state: {e}")
            return {"error": str(e)}
    
    def _collect_hardware_state(self) -> Dict[str, Any]:
        """收集硬件状态"""
        try:
            hardware_state = {}
            
            # CPU信息
            cpu_info = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "current_usage_percent": psutil.cpu_percent(interval=0.1),
                "per_cpu_usage": psutil.cpu_percent(interval=0.1, percpu=True),
                "cpu_freq": psutil.cpu_freq()._asdict() if hasattr(psutil, 'cpu_freq') else None,
                "cpu_stats": psutil.cpu_stats()._asdict() if hasattr(psutil, 'cpu_stats') else None
            }
            hardware_state["cpu"] = cpu_info
            
            # 内存信息
            virtual_memory = psutil.virtual_memory()
            swap_memory = psutil.swap_memory()
            
            memory_info = {
                "virtual": {
                    "total_gb": virtual_memory.total / (1024**3),
                    "available_gb": virtual_memory.available / (1024**3),
                    "used_gb": virtual_memory.used / (1024**3),
                    "percent": virtual_memory.percent,
                    "free_gb": virtual_memory.free / (1024**3)
                },
                "swap": {
                    "total_gb": swap_memory.total / (1024**3),
                    "used_gb": swap_memory.used / (1024**3),
                    "free_gb": swap_memory.free / (1024**3),
                    "percent": swap_memory.percent
                }
            }
            hardware_state["memory"] = memory_info
            
            # 磁盘信息
            disk_info = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "fstype": partition.fstype,
                        "total_gb": usage.total / (1024**3),
                        "used_gb": usage.used / (1024**3),
                        "free_gb": usage.free / (1024**3),
                        "percent": usage.percent
                    })
                except:
                    pass
            
            hardware_state["disks"] = disk_info
            
            # 网络信息
            try:
                net_io = psutil.net_io_counters()
                net_info = {
                    "bytes_sent_mb": net_io.bytes_sent / (1024**2),
                    "bytes_recv_mb": net_io.bytes_recv / (1024**2),
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
                hardware_state["network"] = net_info
            except:
                hardware_state["network"] = {"error": "Failed to collect network info"}
            
            # GPU信息（如果可用）
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                gpu_info = []
                for gpu in gpus:
                    gpu_info.append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": gpu.load,
                        "memory_used_mb": gpu.memoryUsed,
                        "memory_total_mb": gpu.memoryTotal,
                        "temperature": gpu.temperature
                    })
                hardware_state["gpu"] = gpu_info
            except:
                hardware_state["gpu"] = {"error": "GPU monitoring not available"}
            
            return hardware_state
            
        except Exception as e:
            logger.warning(f"Failed to collect hardware state: {e}")
            return {"error": str(e)}
    
    def _collect_context_state(self) -> Dict[str, Any]:
        """收集上下文状态"""
        try:
            if self.context_memory:
                # 假设context_memory有获取状态的方法
                if hasattr(self.context_memory, "get_context_state"):
                    return self.context_memory.get_context_state()
                elif hasattr(self.context_memory, "context_history"):
                    return {
                        "context_history": self.context_memory.context_history,
                        "context_count": len(self.context_memory.context_history)
                    }
            
            # 检查是否有其他上下文管理器
            for name, component in self.registered_components.items():
                if "context" in name.lower() or "memory" in name.lower():
                    if hasattr(component, "get_state"):
                        return component.get_state()
                    elif hasattr(component, "get_context"):
                        return {"context": component.get_context()}
            
            return {"info": "No context manager found or context not accessible"}
            
        except Exception as e:
            logger.warning(f"Failed to collect context state: {e}")
            return {"error": str(e)}
    
    def _collect_evolution_state(self) -> Dict[str, Any]:
        """收集演化状态"""
        try:
            if self.evolution_engine:
                # 假设evolution_engine有获取状态的方法
                if hasattr(self.evolution_engine, "get_evolution_state"):
                    return self.evolution_engine.get_evolution_state()
                elif hasattr(self.evolution_engine, "evolution_state"):
                    state = self.evolution_engine.evolution_state.copy() if hasattr(self.evolution_engine.evolution_state, 'copy') else self.evolution_engine.evolution_state
                    # 移除可能的大数据
                    if "population" in state:
                        state["population_summary"] = f"{len(state['population'])} individuals"
                        del state["population"]
                    return state
            
            # 检查注册组件
            for name, component in self.registered_components.items():
                if "evolution" in name.lower():
                    if hasattr(component, "get_state"):
                        return component.get_state()
            
            return {"info": "No evolution engine found or evolution state not accessible"}
            
        except Exception as e:
            logger.warning(f"Failed to collect evolution state: {e}")
            return {"error": str(e)}
    
    def _collect_metrics_state(self) -> Dict[str, Any]:
        """收集指标状态"""
        try:
            if self.core_metrics_collector:
                if hasattr(self.core_metrics_collector, "get_current_metrics"):
                    return self.core_metrics_collector.get_current_metrics()
                elif hasattr(self.core_metrics_collector, "collect_core_metrics"):
                    return self.core_metrics_collector.collect_core_metrics()
            
            if self.scene_adaptive_params:
                scene_metrics = {}
                if hasattr(self.scene_adaptive_params, "get_current_parameters"):
                    scene_metrics["current_parameters"] = self.scene_adaptive_params.get_current_parameters()
                if hasattr(self.scene_adaptive_params, "get_scene_statistics"):
                    scene_metrics["scene_statistics"] = self.scene_adaptive_params.get_scene_statistics()
                
                return {"scene_adaptive_metrics": scene_metrics}
            
            return {"info": "No metrics collector found"}
            
        except Exception as e:
            logger.warning(f"Failed to collect metrics state: {e}")
            return {"error": str(e)}
    
    def _collect_crash_info(
        self,
        exception_info: Optional[Dict[str, Any]],
        signal_info: Optional[Dict[str, Any]],
        frame_info: Any
    ) -> Dict[str, Any]:
        """收集崩溃信息"""
        crash_info = {}
        
        # 异常信息
        if exception_info:
            crash_info["exception"] = exception_info
        else:
            # 捕获当前异常（如果有）
            exc_info = sys.exc_info()
            if exc_info[0] is not None:
                crash_info["exception"] = {
                    "type": exc_info[0].__name__,
                    "message": str(exc_info[1]),
                    "traceback": traceback.format_exception(*exc_info)
                }
        
        # 信号信息
        if signal_info:
            crash_info["signal"] = signal_info
        
        # 帧/堆栈信息
        if frame_info:
            try:
                stack = traceback.extract_stack(frame_info)
                crash_info["stack"] = [{"filename": s.filename, "lineno": s.lineno, "name": s.name, "line": s.line} for s in stack[-20:]]
            except:
                pass
        
        # 当前堆栈
        current_stack = traceback.extract_stack()
        crash_info["current_stack"] = [{"filename": s.filename, "lineno": s.lineno, "name": s.name, "line": s.line} for s in current_stack[-15:]]
        
        return crash_info
    
    def save_snapshot(self, snapshot_data: Dict[str, Any], prefix: str = "crash") -> str:
        """
        保存快照到文件
        
        Args:
            snapshot_data: 快照数据
            prefix: 文件前缀
            
        Returns:
            保存的文件路径
        """
        try:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}_{self.snapshot_counter:06d}.json"
            filepath = os.path.join(self.snapshot_dir, filename)
            
            # 增加计数器
            self.snapshot_counter += 1
            
            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(snapshot_data, f, ensure_ascii=False, indent=2)
            
            # 清理旧快照
            self._cleanup_old_snapshots()
            
            logger.info(f"Snapshot saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            # 尝试备用位置
            try:
                backup_path = os.path.join(os.path.expanduser("~"), f"crash_snapshot_backup_{int(time.time())}.json")
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(snapshot_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Snapshot saved to backup location: {backup_path}")
                return backup_path
            except:
                logger.critical("Failed to save snapshot to both primary and backup locations")
                return ""
    
    def _cleanup_old_snapshots(self) -> None:
        """清理旧快照"""
        try:
            # 获取所有快照文件
            snapshot_files = []
            for filename in os.listdir(self.snapshot_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.snapshot_dir, filename)
                    if os.path.isfile(filepath):
                        snapshot_files.append((filepath, os.path.getmtime(filepath)))
            
            # 按修改时间排序
            snapshot_files.sort(key=lambda x: x[1])
            
            # 删除超过限制的旧文件
            if len(snapshot_files) > self.max_snapshots:
                files_to_delete = snapshot_files[:len(snapshot_files) - self.max_snapshots]
                for filepath, _ in files_to_delete:
                    try:
                        os.remove(filepath)
                        logger.debug(f"Deleted old snapshot: {filepath}")
                    except Exception as e:
                        logger.warning(f"Failed to delete old snapshot {filepath}: {e}")
                        
        except Exception as e:
            logger.warning(f"Failed to cleanup old snapshots: {e}")
    
    def create_manual_snapshot(self, description: str = "Manual snapshot") -> str:
        """
        创建手动快照（用于测试和调试）
        
        Args:
            description: 快照描述
            
        Returns:
            保存的文件路径
        """
        logger.info(f"Creating manual snapshot: {description}")
        
        snapshot_data = self.create_crash_snapshot(
            crash_type="manual",
            exception_info={"description": description, "type": "manual_snapshot"}
        )
        
        # 添加手动快照特定信息
        snapshot_data["manual_snapshot"] = {
            "description": description,
            "purpose": "debugging_or_testing"
        }
        
        filepath = self.save_snapshot(snapshot_data, prefix="manual")
        logger.info(f"Manual snapshot saved to: {filepath}")
        return filepath
    
    def get_snapshot_files(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取快照文件列表
        
        Args:
            limit: 限制返回的文件数量
            
        Returns:
            快照文件信息列表
        """
        try:
            snapshot_files = []
            for filename in os.listdir(self.snapshot_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.snapshot_dir, filename)
                    if os.path.isfile(filepath):
                        file_info = {
                            "filename": filename,
                            "filepath": filepath,
                            "size_bytes": os.path.getsize(filepath),
                            "modified_time": os.path.getmtime(filepath),
                            "modified_time_iso": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                        }
                        snapshot_files.append(file_info)
            
            # 按修改时间倒序排序
            snapshot_files.sort(key=lambda x: x["modified_time"], reverse=True)
            
            return snapshot_files[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get snapshot files: {e}")
            return []
    
    def load_snapshot(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        加载快照文件
        
        Args:
            filepath: 快照文件路径
            
        Returns:
            快照数据或None
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                snapshot_data = json.load(f)
            
            logger.info(f"Snapshot loaded from: {filepath}")
            return snapshot_data
            
        except Exception as e:
            logger.error(f"Failed to load snapshot {filepath}: {e}")
            return None
    
    def analyze_snapshot(self, snapshot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析快照数据
        
        Args:
            snapshot_data: 快照数据
            
        Returns:
            分析结果
        """
        analysis = {
            "snapshot_info": {
                "crash_type": snapshot_data.get("metadata", {}).get("crash_type", "unknown"),
                "timestamp": snapshot_data.get("metadata", {}).get("timestamp", "unknown"),
                "snapshot_creation_time": snapshot_data.get("metadata", {}).get("snapshot_creation_time_seconds", 0)
            },
            "crash_analysis": {},
            "system_health": {},
            "recommendations": []
        }
        
        # 分析崩溃信息
        crash_info = snapshot_data.get("crash_info", {})
        if "exception" in crash_info:
            exception = crash_info["exception"]
            analysis["crash_analysis"]["exception_type"] = exception.get("type", "unknown")
            analysis["crash_analysis"]["exception_message"] = exception.get("message", "")
            
            # 分析堆栈跟踪
            if "traceback" in exception:
                traceback_text = "\n".join(exception["traceback"])
                analysis["crash_analysis"]["traceback"] = traceback_text[-1000:]  # 最后1000个字符
        
        # 分析系统状态
        system_state = snapshot_data.get("system_state", {})
        if "process" in system_state:
            process_info = system_state["process"]
            if "memory_info" in process_info:
                memory_used_mb = process_info["memory_info"].get("rss", 0) / (1024**2)
                analysis["system_health"]["memory_used_mb"] = memory_used_mb
        
        # 分析硬件状态
        hardware_state = snapshot_data.get("hardware_state", {})
        if "memory" in hardware_state:
            memory = hardware_state["memory"]
            if "virtual" in memory:
                virtual_memory = memory["virtual"]
                analysis["system_health"]["memory_usage_percent"] = virtual_memory.get("percent", 0)
        
        if "cpu" in hardware_state:
            cpu = hardware_state["cpu"]
            analysis["system_health"]["cpu_usage_percent"] = cpu.get("current_usage_percent", 0)
        
        # 生成建议
        recommendations = []
        
        # 内存使用建议
        if analysis["system_health"].get("memory_usage_percent", 0) > 90:
            recommendations.append({
                "issue": "内存使用率过高",
                "recommendation": "检查内存泄漏，优化内存使用，增加系统内存",
                "priority": "high"
            })
        
        # CPU使用建议
        if analysis["system_health"].get("cpu_usage_percent", 0) > 90:
            recommendations.append({
                "issue": "CPU使用率过高",
                "recommendation": "优化CPU密集型任务，考虑负载均衡或增加CPU资源",
                "priority": "high"
            })
        
        # 异常类型建议
        crash_type = analysis["snapshot_info"].get("crash_type", "")
        if "MemoryError" in crash_type or "memory" in crash_type.lower():
            recommendations.append({
                "issue": "内存相关崩溃",
                "recommendation": "实施内存监控和限制，优化大内存操作，使用流式处理",
                "priority": "critical"
            })
        elif "Timeout" in crash_type or "timeout" in crash_type.lower():
            recommendations.append({
                "issue": "超时相关崩溃",
                "recommendation": "增加超时设置，优化性能瓶颈，实施重试机制",
                "priority": "medium"
            })
        
        analysis["recommendations"] = recommendations
        
        return analysis


# 单例实例
_crash_snapshot_instance = None

def get_crash_snapshot_manager(
    snapshot_dir: Optional[str] = None,
    max_snapshots: int = 50,
    enable_auto_capture: bool = True
) -> CrashSnapshotManager:
    """
    获取崩溃快照管理器实例（单例模式）
    
    Args:
        snapshot_dir: 快照保存目录
        max_snapshots: 最大快照数量
        enable_auto_capture: 是否启用自动捕获
        
    Returns:
        CrashSnapshotManager实例
    """
    global _crash_snapshot_instance
    
    if _crash_snapshot_instance is None:
        _crash_snapshot_instance = CrashSnapshotManager(
            snapshot_dir=snapshot_dir,
            max_snapshots=max_snapshots,
            enable_auto_capture=enable_auto_capture
        )
    
    return _crash_snapshot_instance


# 全局异常钩子
def setup_global_exception_hook():
    """设置全局异常钩子"""
    original_excepthook = sys.excepthook
    
    def global_exception_hook(exc_type, exc_value, exc_traceback):
        """全局异常处理器"""
        # 调用原始处理器
        original_excepthook(exc_type, exc_value, exc_traceback)
        
        # 创建崩溃快照
        try:
            snapshot_manager = get_crash_snapshot_manager()
            
            snapshot_data = snapshot_manager.create_crash_snapshot(
                crash_type=f"unhandled_exception_{exc_type.__name__}",
                exception_info={
                    "type": exc_type.__name__,
                    "message": str(exc_value),
                    "traceback": traceback.format_exception(exc_type, exc_value, exc_traceback)
                }
            )
            
            snapshot_file = snapshot_manager.save_snapshot(snapshot_data, prefix=f"unhandled_{exc_type.__name__}")
            logger.critical(f"Unhandled exception snapshot saved to: {snapshot_file}")
        except Exception as e:
            logger.critical(f"Failed to create crash snapshot for unhandled exception: {e}")
    
    sys.excepthook = global_exception_hook
    logger.info("Global exception hook setup complete")


# 示例使用
if __name__ == "__main__":
    # 测试崩溃快照管理器
    print("=" * 80)
    print("测试崩溃快照管理器")
    print("=" * 80)
    
    # 创建管理器
    snapshot_manager = CrashSnapshotManager(
        snapshot_dir="./test_snapshots",
        max_snapshots=5,
        enable_auto_capture=False  # 测试时禁用自动捕获
    )
    
    print(f"\n1. 快照目录: {snapshot_manager.snapshot_dir}")
    
    print("\n2. 创建手动快照:")
    snapshot_file = snapshot_manager.create_manual_snapshot("测试手动快照")
    print(f"   快照保存到: {snapshot_file}")
    
    print("\n3. 获取快照文件列表:")
    snapshot_files = snapshot_manager.get_snapshot_files()
    for i, file_info in enumerate(snapshot_files):
        print(f"   {i+1}. {file_info['filename']} ({file_info['size_bytes']} bytes)")
    
    if snapshot_files:
        print("\n4. 加载并分析快照:")
        snapshot_data = snapshot_manager.load_snapshot(snapshot_files[0]["filepath"])
        if snapshot_data:
            analysis = snapshot_manager.analyze_snapshot(snapshot_data)
            print(f"   崩溃类型: {analysis['snapshot_info'].get('crash_type', 'unknown')}")
            print(f"   时间戳: {analysis['snapshot_info'].get('timestamp', 'unknown')}")
            
            if analysis['recommendations']:
                print(f"   建议数量: {len(analysis['recommendations'])}")
                for rec in analysis['recommendations']:
                    print(f"     - {rec['issue']}: {rec['recommendation']}")
    
    print("\n5. 测试全局异常钩子:")
    setup_global_exception_hook()
    print("   全局异常钩子已设置（测试完成时不会实际触发）")
    
    print("\n✓ 崩溃快照管理器测试完成")
    
    # 清理测试目录
    import shutil
    if os.path.exists("./test_snapshots"):
        shutil.rmtree("./test_snapshots")
        print("测试目录已清理")