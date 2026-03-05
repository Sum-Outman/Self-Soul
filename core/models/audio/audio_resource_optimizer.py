"""
Audio Resource Optimizer

This module provides advanced resource management and optimization for audio processing.
It implements context managers, resource pooling, and automatic cleanup to prevent
resource leaks and improve performance.
"""

import logging
import threading
import time
import weakref
import gc
import numpy as np
from typing import Dict, Any, Optional, List, Set, Union
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ResourceMetrics:
    """Metrics for tracking resource usage"""
    name: str
    acquired_count: int = 0
    released_count: int = 0
    current_count: int = 0
    total_hold_time: float = 0.0
    max_hold_time: float = 0.0
    last_acquired: Optional[datetime] = None
    last_released: Optional[datetime] = None


class AudioResourcePool:
    """Pool for managing reusable audio resources"""
    
    def __init__(self, name: str, max_size: int = 10, logger=None):
        self.name = name
        self.max_size = max_size
        self.logger = logger or logging.getLogger(__name__)
        self._available = []
        self._in_use = set()
        self._lock = threading.RLock()
        self._creation_count = 0
        self._metrics = ResourceMetrics(name=f"pool_{name}")
        
    def acquire(self, create_func, *args, **kwargs):
        """Acquire a resource from the pool"""
        with self._lock:
            # Try to get from available pool first
            if self._available:
                resource = self._available.pop()
                self.logger.debug(f"Reusing resource from pool {self.name}")
            else:
                # Create new resource if pool not full
                if len(self._in_use) < self.max_size:
                    resource = create_func(*args, **kwargs)
                    self._creation_count += 1
                    self.logger.debug(f"Created new resource for pool {self.name}")
                else:
                    # Wait for resource to become available (simple implementation)
                    self.logger.warning(f"Resource pool {self.name} exhausted, waiting...")
                    return None
            
            self._in_use.add(resource)
            self._metrics.acquired_count += 1
            self._metrics.current_count = len(self._in_use)
            self._metrics.last_acquired = datetime.now()
            
            return resource
    
    def release(self, resource):
        """Release a resource back to the pool"""
        with self._lock:
            if resource in self._in_use:
                self._in_use.remove(resource)
                self._available.append(resource)
                self._metrics.released_count += 1
                self._metrics.current_count = len(self._in_use)
                self._metrics.last_released = datetime.now()
                self.logger.debug(f"Released resource to pool {self.name}")
            else:
                self.logger.warning(f"Attempted to release unknown resource to pool {self.name}")
    
    def cleanup(self):
        """Clean up all resources in the pool"""
        with self._lock:
            total_resources = len(self._available) + len(self._in_use)
            self._available.clear()
            self._in_use.clear()
            self.logger.info(f"Cleaned up {total_resources} resources from pool {self.name}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pool metrics"""
        with self._lock:
            return {
                "name": self.name,
                "available": len(self._available),
                "in_use": len(self._in_use),
                "max_size": self.max_size,
                "creation_count": self._creation_count,
                "utilization": len(self._in_use) / self.max_size if self.max_size > 0 else 0,
                "reuse_rate": (self._metrics.acquired_count - self._creation_count) / 
                              self._metrics.acquired_count if self._metrics.acquired_count > 0 else 0
            }


class AudioResourceMonitor:
    """Monitor for tracking audio resource usage"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self._resources = {}
        self._lock = threading.RLock()
        self._monitor_thread = None
        self._running = False
        self._check_interval = 30  # seconds
        
    def register_resource(self, resource_name: str, resource_obj):
        """Register a resource for monitoring"""
        with self._lock:
            if resource_name not in self._resources:
                self._resources[resource_name] = {
                    "object": weakref.ref(resource_obj),
                    "created_at": datetime.now(),
                    "access_count": 0,
                    "last_accessed": datetime.now(),
                    "is_active": True
                }
                self.logger.debug(f"Registered resource: {resource_name}")
    
    def unregister_resource(self, resource_name: str):
        """Unregister a resource from monitoring"""
        with self._lock:
            if resource_name in self._resources:
                del self._resources[resource_name]
                self.logger.debug(f"Unregistered resource: {resource_name}")
    
    def mark_accessed(self, resource_name: str):
        """Mark a resource as accessed"""
        with self._lock:
            if resource_name in self._resources:
                self._resources[resource_name]["access_count"] += 1
                self._resources[resource_name]["last_accessed"] = datetime.now()
    
    def start_monitoring(self):
        """Start the resource monitoring thread"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self.logger.warning("Resource monitor already running")
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="AudioResourceMonitor"
        )
        self._monitor_thread.start()
        self.logger.info("Started audio resource monitor")
    
    def stop_monitoring(self):
        """Stop the resource monitoring thread"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self.logger.info("Stopped audio resource monitor")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            time.sleep(self._check_interval)
            self._check_resources()
    
    def _check_resources(self):
        """Check resource status and clean up inactive ones"""
        with self._lock:
            inactive_resources = []
            current_time = datetime.now()
            
            for name, info in self._resources.items():
                # Check if object still exists
                obj = info["object"]()
                if obj is None:
                    inactive_resources.append(name)
                    continue
                
                # Check for stale resources (not accessed in last 5 minutes)
                last_accessed = info["last_accessed"]
                if (current_time - last_accessed) > timedelta(minutes=5):
                    self.logger.warning(f"Resource {name} has not been accessed in 5 minutes")
                    info["is_active"] = False
            
            # Clean up inactive resources
            for name in inactive_resources:
                del self._resources[name]
                self.logger.info(f"Cleaned up inactive resource: {name}")
    
    def get_resource_report(self) -> Dict[str, Any]:
        """Get a report of all monitored resources"""
        with self._lock:
            report = {
                "total_resources": len(self._resources),
                "active_resources": 0,
                "inactive_resources": 0,
                "resources": []
            }
            
            for name, info in self._resources.items():
                resource_info = {
                    "name": name,
                    "created_at": info["created_at"].isoformat(),
                    "access_count": info["access_count"],
                    "last_accessed": info["last_accessed"].isoformat(),
                    "is_active": info["is_active"],
                    "age_seconds": (datetime.now() - info["created_at"]).total_seconds()
                }
                
                if info["is_active"]:
                    report["active_resources"] += 1
                else:
                    report["inactive_resources"] += 1
                
                report["resources"].append(resource_info)
            
            return report


@contextmanager
def managed_audio_resource(resource_name: str, acquire_func, release_func, *args, **kwargs):
    """
    Context manager for audio resources with automatic cleanup
    
    Args:
        resource_name: Name of the resource for logging
        acquire_func: Function to acquire the resource
        release_func: Function to release the resource
        *args, **kwargs: Arguments to pass to acquire_func
    """
    resource = None
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Acquiring audio resource: {resource_name}")
        resource = acquire_func(*args, **kwargs)
        yield resource
    except Exception as e:
        logger.error(f"Error using audio resource {resource_name}: {e}")
        raise
    finally:
        if resource is not None:
            try:
                logger.info(f"Releasing audio resource: {resource_name}")
                release_func(resource)
            except Exception as e:
                logger.error(f"Error releasing audio resource {resource_name}: {e}")


class AudioMemoryManager:
    """Manager for audio memory optimization"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self._audio_buffers = {}
        self._buffer_lock = threading.RLock()
        self._max_buffer_size = 100 * 1024 * 1024  # 100MB
        self._current_buffer_size = 0
        
    def cache_audio(self, key: str, audio_data: Union[np.ndarray, bytes], 
                   metadata: Optional[Dict[str, Any]] = None):
        """
        Cache audio data in memory
        
        Args:
            key: Unique key for the audio data
            audio_data: Audio data to cache
            metadata: Optional metadata about the audio
        """
        with self._buffer_lock:
            # Calculate size
            if isinstance(audio_data, np.ndarray):
                size = audio_data.nbytes
            elif isinstance(audio_data, bytes):
                size = len(audio_data)
            else:
                size = 1024  # Default estimate
            
            # Check if we need to make space
            if self._current_buffer_size + size > self._max_buffer_size:
                self._make_space(size)
            
            # Store in cache
            self._audio_buffers[key] = {
                "data": audio_data,
                "metadata": metadata or {},
                "size": size,
                "last_accessed": datetime.now(),
                "access_count": 0
            }
            
            self._current_buffer_size += size
            self.logger.debug(f"Cached audio {key} ({size} bytes)")
    
    def get_audio(self, key: str) -> Optional[Union[np.ndarray, bytes]]:
        """Get audio data from cache"""
        with self._buffer_lock:
            if key in self._audio_buffers:
                buffer_info = self._audio_buffers[key]
                buffer_info["last_accessed"] = datetime.now()
                buffer_info["access_count"] += 1
                self.logger.debug(f"Retrieved audio from cache: {key}")
                return buffer_info["data"]
            return None
    
    def remove_audio(self, key: str):
        """Remove audio data from cache"""
        with self._buffer_lock:
            if key in self._audio_buffers:
                size = self._audio_buffers[key]["size"]
                del self._audio_buffers[key]
                self._current_buffer_size -= size
                self.logger.debug(f"Removed audio from cache: {key}")
    
    def _make_space(self, required_size: int):
        """Make space in the cache by removing least recently used items"""
        with self._buffer_lock:
            # Sort buffers by last accessed time
            sorted_buffers = sorted(
                self._audio_buffers.items(),
                key=lambda x: x[1]["last_accessed"]
            )
            
            removed_size = 0
            for key, buffer_info in sorted_buffers:
                if self._current_buffer_size - removed_size + required_size <= self._max_buffer_size:
                    break
                
                self.logger.debug(f"Evicting audio from cache: {key}")
                removed_size += buffer_info["size"]
                del self._audio_buffers[key]
            
            self._current_buffer_size -= removed_size
            self.logger.info(f"Made space in audio cache: removed {removed_size} bytes")
    
    def clear_cache(self):
        """Clear all cached audio data"""
        with self._buffer_lock:
            cleared_size = self._current_buffer_size
            self._audio_buffers.clear()
            self._current_buffer_size = 0
            self.logger.info(f"Cleared audio cache: {cleared_size} bytes")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._buffer_lock:
            total_accesses = sum(buf["access_count"] for buf in self._audio_buffers.values())
            avg_accesses = total_accesses / len(self._audio_buffers) if self._audio_buffers else 0
            
            return {
                "total_buffers": len(self._audio_buffers),
                "total_size": self._current_buffer_size,
                "max_size": self._max_buffer_size,
                "utilization": self._current_buffer_size / self._max_buffer_size,
                "total_accesses": total_accesses,
                "average_accesses": avg_accesses,
                "oldest_buffer": min((buf["last_accessed"] for buf in self._audio_buffers.values()), 
                                   default=None)
            }


class AudioThreadPool:
    """Thread pool for audio processing tasks"""
    
    def __init__(self, max_workers: int = 4, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.max_workers = max_workers
        self._workers = []
        self._task_queue = []
        self._result_queue = []
        self._lock = threading.RLock()
        self._running = False
        self._task_id_counter = 0
        
    def submit(self, task_func, *args, **kwargs) -> int:
        """
        Submit a task to the thread pool
        
        Args:
            task_func: Function to execute
            *args, **kwargs: Arguments to pass to task_func
            
        Returns:
            Task ID for tracking
        """
        with self._lock:
            task_id = self._task_id_counter
            self._task_id_counter += 1
            
            task = {
                "id": task_id,
                "func": task_func,
                "args": args,
                "kwargs": kwargs,
                "submitted_at": datetime.now(),
                "status": "pending"
            }
            
            self._task_queue.append(task)
            self.logger.debug(f"Submitted audio task {task_id}")
            
            # Start workers if not already running
            if not self._running:
                self._start_workers()
            
            return task_id
    
    def _start_workers(self):
        """Start worker threads"""
        self._running = True
        
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"AudioWorker-{i}"
            )
            worker.start()
            self._workers.append(worker)
        
        self.logger.info(f"Started {self.max_workers} audio worker threads")
    
    def _worker_loop(self):
        """Worker thread main loop"""
        while self._running:
            task = None
            
            with self._lock:
                if self._task_queue:
                    task = self._task_queue.pop(0)
                    task["status"] = "running"
                    task["started_at"] = datetime.now()
            
            if task:
                try:
                    result = task["func"](*task["args"], **task["kwargs"])
                    task["status"] = "completed"
                    task["result"] = result
                    task["completed_at"] = datetime.now()
                    
                    with self._lock:
                        self._result_queue.append(task)
                    
                    self.logger.debug(f"Audio task {task['id']} completed")
                except Exception as e:
                    task["status"] = "failed"
                    task["error"] = str(e)
                    task["completed_at"] = datetime.now()
                    
                    with self._lock:
                        self._result_queue.append(task)
                    
                    self.logger.error(f"Audio task {task['id']} failed: {e}")
            else:
                # No tasks, sleep a bit
                time.sleep(0.1)
    
    def get_task_result(self, task_id: int, timeout: float = None) -> Optional[Any]:
        """
        Get the result of a completed task
        
        Args:
            task_id: ID of the task
            timeout: Maximum time to wait (None = wait forever)
            
        Returns:
            Task result or None if not found/timed out
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                # Check result queue
                for i, task in enumerate(self._result_queue):
                    if task["id"] == task_id:
                        result_task = self._result_queue.pop(i)
                        return result_task.get("result")
                
                # Check if task is still in queue
                for task in self._task_queue:
                    if task["id"] == task_id:
                        break
                else:
                    # Task not found
                    return None
            
            # Check timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                return None
            
            # Wait a bit before checking again
            time.sleep(0.05)
    
    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool"""
        self._running = False
        
        if wait:
            for worker in self._workers:
                worker.join(timeout=5.0)
        
        self._workers.clear()
        self.logger.info("Audio thread pool shutdown complete")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics"""
        with self._lock:
            pending = len([t for t in self._task_queue if t["status"] == "pending"])
            running = len([t for t in self._task_queue if t["status"] == "running"])
            completed = len([t for t in self._result_queue if t["status"] == "completed"])
            failed = len([t for t in self._result_queue if t["status"] == "failed"])
            
            return {
                "max_workers": self.max_workers,
                "active_workers": len([w for w in self._workers if w.is_alive()]),
                "tasks_pending": pending,
                "tasks_running": running,
                "tasks_completed": completed,
                "tasks_failed": failed,
                "total_tasks": pending + running + completed + failed
            }


# Global instances for easy access
_resource_monitor = AudioResourceMonitor()
_memory_manager = AudioMemoryManager()
_thread_pool = AudioThreadPool()

def get_resource_monitor() -> AudioResourceMonitor:
    """Get the global resource monitor instance"""
    return _resource_monitor

def get_memory_manager() -> AudioMemoryManager:
    """Get the global memory manager instance"""
    return _memory_manager

def get_thread_pool() -> AudioThreadPool:
    """Get the global thread pool instance"""
    return _thread_pool
