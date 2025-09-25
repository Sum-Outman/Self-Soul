"""
Resource Management Mixin for AGI Models

This mixin provides resource management capabilities including memory management,
GPU resource allocation, and cleanup operations. It is designed to be mixed into
model classes to handle resource lifecycle efficiently.
"""

import psutil
import gc
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ResourceManagementMixin:
    """
    Mixin class for managing computational resources in AGI models.
    Provides methods for memory monitoring, GPU resource management, and cleanup.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize resource management capabilities."""
        super().__init__(*args, **kwargs)
        self._resource_allocations = {}
        self._memory_usage_baseline = self._get_memory_usage()
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / 1024 / 1024,  # RSS in MB
            'vms': memory_info.vms / 1024 / 1024,  # VMS in MB
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available / 1024 / 1024,
            'total': psutil.virtual_memory().total / 1024 / 1024
        }
    
    def monitor_resource_usage(self) -> Dict[str, Any]:
        """
        Monitor current resource usage and return detailed statistics.
        
        Returns:
            Dictionary containing memory, CPU, and GPU usage information
        """
        resource_stats = {
            'memory': self._get_memory_usage(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'timestamp': self._get_timestamp()
        }
        
        # Add GPU monitoring if available
        gpu_stats = self._get_gpu_usage()
        if gpu_stats:
            resource_stats['gpu'] = gpu_stats
        
        logger.debug(f"Resource usage: {resource_stats}")
        return resource_stats
    
    def _get_gpu_usage(self) -> Optional[Dict[str, Any]]:
        """
        Get GPU usage statistics if GPUs are available.
        
        Returns:
            GPU usage information or None if no GPUs available
        """
        try:
            import torch
            if torch.cuda.is_available():
                gpu_stats = {}
                for i in range(torch.cuda.device_count()):
                    gpu_stats[f'gpu_{i}'] = {
                        'memory_allocated': torch.cuda.memory_allocated(i) / 1024 / 1024,
                        'memory_cached': torch.cuda.memory_cached(i) / 1024 / 1024,
                        'utilization': torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0
                    }
                return gpu_stats
        except ImportError:
            pass
        
        return None
    
    def allocate_resources(self, resource_type: str, amount: float, **kwargs) -> bool:
        """
        Allocate specific resources for model operations.
        
        Args:
            resource_type: Type of resource to allocate (e.g., 'memory', 'gpu')
            amount: Amount of resource to allocate
            **kwargs: Additional allocation parameters
            
        Returns:
            True if allocation successful, False otherwise
        """
        allocation_id = f"{resource_type}_{self._get_timestamp()}"
        
        try:
            if resource_type == 'memory':
                # For memory, we track the allocation but don't pre-allocate
                self._resource_allocations[allocation_id] = {
                    'type': resource_type,
                    'amount': amount,
                    'timestamp': self._get_timestamp()
                }
                logger.info(f"Memory allocation tracked: {amount}MB")
                
            elif resource_type == 'gpu':
                # GPU resource allocation tracking
                self._resource_allocations[allocation_id] = {
                    'type': resource_type,
                    'amount': amount,
                    'device': kwargs.get('device', 'cuda:0'),
                    'timestamp': self._get_timestamp()
                }
                logger.info(f"GPU allocation tracked: {amount}MB on {kwargs.get('device')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Resource allocation failed: {e}")
            return False
    
    def release_resources(self, allocation_id: Optional[str] = None) -> bool:
        """
        Release allocated resources.
        
        Args:
            allocation_id: Specific allocation to release, or None to release all
            
        Returns:
            True if release successful, False otherwise
        """
        try:
            if allocation_id:
                if allocation_id in self._resource_allocations:
                    released = self._resource_allocations.pop(allocation_id)
                    logger.info(f"Released resource allocation: {released}")
                else:
                    logger.warning(f"Allocation ID not found: {allocation_id}")
                    return False
            else:
                # Release all allocations
                released_count = len(self._resource_allocations)
                self._resource_allocations.clear()
                logger.info(f"Released all resource allocations: {released_count} allocations")
                
                # Force garbage collection
                gc.collect()
            
            return True
            
        except Exception as e:
            logger.error(f"Resource release failed: {e}")
            return False
    
    def optimize_memory_usage(self) -> Dict[str, float]:
        """
        Optimize memory usage by cleaning up and compacting memory.
        
        Returns:
            Memory usage statistics after optimization
        """
        logger.info("Optimizing memory usage...")
        
        # Force garbage collection
        gc.collect()
        
        # Clear any cached data if possible
        if hasattr(self, 'clear_caches'):
            self.clear_caches()
        
        # Get memory usage after optimization
        post_optimization = self._get_memory_usage()
        reduction = self._memory_usage_baseline['rss'] - post_optimization['rss']
        
        logger.info(f"Memory optimized. Reduction: {reduction:.2f}MB")
        return post_optimization
    
    def get_resource_status(self) -> Dict[str, Any]:
        """
        Get comprehensive resource status report.
        
        Returns:
            Detailed resource status information
        """
        current_usage = self.monitor_resource_usage()
        baseline = self._memory_usage_baseline
        
        status = {
            'current_usage': current_usage,
            'baseline_usage': baseline,
            'active_allocations': len(self._resource_allocations),
            'allocations_list': list(self._resource_allocations.keys()),
            'memory_increase_mb': current_usage['memory']['rss'] - baseline['rss'],
            'memory_increase_percent': ((current_usage['memory']['rss'] - baseline['rss']) / baseline['rss']) * 100
        }
        
        return status
    
    def _get_timestamp(self) -> float:
        """Get current timestamp for resource tracking."""
        import time
        return time.time()
    
    def cleanup(self):
        """Clean up all resources before destruction."""
        logger.info("Cleaning up resources...")
        self.release_resources()  # Release all allocations
        super().cleanup() if hasattr(super(), 'cleanup') else None
    
    def __del__(self):
        """Destructor to ensure resource cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Avoid exceptions during destruction
