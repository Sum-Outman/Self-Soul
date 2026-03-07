"""
Neural Network Optimization

Optimization and compilation tools for neural networks in the
unified cognitive architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import logging
import time

logger = logging.getLogger(__name__)


class CognitiveCompiler:
    """
    Compiler for neural network components.
    
    Optimizes neural networks for specific hardware targets and
    cognitive tasks, including fusion, quantization, and graph optimization.
    """
    
    def __init__(self, target_device: str = "auto"):
        """
        Initialize cognitive compiler.
        
        Args:
            target_device: Target device ('cpu', 'cuda', or 'auto')
        """
        if target_device == "auto":
            self.target_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.target_device = target_device
        
        logger.info(f"CognitiveCompiler initialized for {self.target_device}")
    
    def compile_model(self, model: nn.Module, input_shapes: Dict[str, Tuple]) -> nn.Module:
        """
        Compile a neural network model for optimal performance.
        
        Args:
            model: PyTorch model to compile
            input_shapes: Dictionary of input shapes
            
        Returns:
            Compiled model
        """
        logger.info(f"Compiling model for {self.target_device}")
        
        # Move model to target device
        model.to(self.target_device)
        
        # Create example inputs for tracing
        example_inputs = self._create_example_inputs(input_shapes)
        
        try:
            # Try to compile with torch.compile (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                compiled_model = torch.compile(
                    model,
                    mode='reduce-overhead',
                    fullgraph=False,
                    dynamic=False
                )
                logger.info("Model compiled with torch.compile")
                return compiled_model
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
        
        # Fallback to jit tracing
        try:
            traced_model = torch.jit.trace(model, example_inputs)
            traced_model.to(self.target_device)
            logger.info("Model compiled with torch.jit.trace")
            return traced_model
        except Exception as e:
            logger.warning(f"jit.trace failed: {e}")
        
        # Return original model if compilation fails
        logger.warning("Compilation failed, returning original model")
        return model
    
    def optimize_inference(self, model: nn.Module, use_fp16: bool = True) -> nn.Module:
        """
        Optimize model for inference.
        
        Args:
            model: Model to optimize
            use_fp16: Use mixed precision if available
            
        Returns:
            Optimized model
        """
        # Set to evaluation mode
        model.eval()
        
        # Apply optimizations
        with torch.no_grad():
            # Disable gradient computation
            for param in model.parameters():
                param.requires_grad = False
            
            # Enable mixed precision if requested and available
            if use_fp16 and self.target_device == "cuda":
                try:
                    model = model.half()
                    logger.info("Converted model to FP16 for inference")
                except Exception as e:
                    logger.warning(f"FP16 conversion failed: {e}")
        
        return model
    
    def _create_example_inputs(self, input_shapes: Dict[str, Tuple]) -> Any:
        """
        Create example inputs for tracing.
        
        Args:
            input_shapes: Dictionary of input shapes
            
        Returns:
            Example inputs suitable for tracing
        """
        example_inputs = []
        
        for shape in input_shapes.values():
            # Create random tensor with appropriate shape
            tensor = torch.randn(*shape)
            example_inputs.append(tensor)
        
        if len(example_inputs) == 1:
            return example_inputs[0]
        return tuple(example_inputs)
    
    def get_compilation_stats(self, model: nn.Module) -> Dict[str, Any]:
        """
        Get model compilation statistics.
        
        Args:
            model: Compiled model
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'device': str(next(model.parameters()).device),
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'compiled': hasattr(model, '_orig_mod') or isinstance(model, torch.jit.ScriptModule),
            'dtype': str(next(model.parameters()).dtype)
        }
        
        # Add memory usage if on CUDA
        if self.target_device == "cuda":
            torch.cuda.synchronize()
            stats['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
        
        return stats


class GPUMemoryManager:
    """
    GPU memory manager for neural network components.
    
    Manages GPU memory allocation, sharing, and cleanup for
    efficient multi-model execution.
    """
    
    def __init__(self, max_gpu_memory_mb: Optional[int] = None):
        """
        Initialize GPU memory manager.
        
        Args:
            max_gpu_memory_mb: Maximum GPU memory to use (None = all available)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if max_gpu_memory_mb is None and self.device.type == "cuda":
            # Use 90% of available memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            self.max_memory = int(total_memory * 0.9)
        else:
            self.max_memory = (max_gpu_memory_mb * 1024 * 1024) if max_gpu_memory_mb else 0
        
        self.allocated_memory = 0
        self.model_memory = {}  # model_name -> memory_used
        
        logger.info(f"GPUMemoryManager initialized for {self.device}, max memory: {self.max_memory/(1024*1024):.1f}MB")
    
    def allocate_for_model(self, model: nn.Module, model_name: str, input_size: Tuple) -> bool:
        """
        Allocate GPU memory for a model.
        
        Args:
            model: Model to allocate memory for
            model_name: Name for tracking
            input_size: Expected input size
            
        Returns:
            True if allocation successful, False otherwise
        """
        if self.device.type != "cuda":
            logger.warning(f"Not on GPU, skipping memory allocation for {model_name}")
            return True
        
        # Estimate memory requirement
        estimated_memory = self._estimate_model_memory(model, input_size)
        
        # Check if enough memory available
        if self.allocated_memory + estimated_memory > self.max_memory:
            logger.warning(f"Insufficient GPU memory for {model_name}: need {estimated_memory/(1024*1024):.1f}MB, have {(self.max_memory - self.allocated_memory)/(1024*1024):.1f}MB")
            return False
        
        # Move model to GPU
        model.to(self.device)
        
        # Update memory tracking
        self.allocated_memory += estimated_memory
        self.model_memory[model_name] = estimated_memory
        
        logger.info(f"Allocated {estimated_memory/(1024*1024):.1f}MB GPU memory for {model_name}")
        return True
    
    def release_model_memory(self, model_name: str):
        """Release memory allocated for a model"""
        if model_name in self.model_memory:
            released_memory = self.model_memory[model_name]
            self.allocated_memory -= released_memory
            del self.model_memory[model_name]
            
            logger.info(f"Released {released_memory/(1024*1024):.1f}MB GPU memory for {model_name}")
    
    def clear_all_memory(self):
        """Clear all GPU memory allocations"""
        self.model_memory.clear()
        self.allocated_memory = 0
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("Cleared all GPU memory and cache")
    
    def _estimate_model_memory(self, model: nn.Module, input_size: Tuple) -> int:
        """
        Estimate GPU memory requirement for a model.
        
        Args:
            model: Model to estimate
            input_size: Expected input size
            
        Returns:
            Estimated memory in bytes
        """
        # Count parameters
        param_memory = 0
        for param in model.parameters():
            param_memory += param.numel() * param.element_size()
        
        # Estimate activation memory (rough estimate)
        # For simplicity, assume activations are 2x input size
        activation_memory = torch.prod(torch.tensor(input_size)).item() * 2 * 4  # 4 bytes per float
        
        # Add 20% overhead
        total_memory = int((param_memory + activation_memory) * 1.2)
        
        return total_memory
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        stats = {
            'device': str(self.device),
            'max_memory_mb': self.max_memory / (1024 * 1024),
            'allocated_memory_mb': self.allocated_memory / (1024 * 1024),
            'available_memory_mb': (self.max_memory - self.allocated_memory) / (1024 * 1024),
            'memory_usage_percent': (self.allocated_memory / self.max_memory) * 100 if self.max_memory > 0 else 0,
            'model_count': len(self.model_memory)
        }
        
        # Add CUDA-specific stats
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            stats.update({
                'cuda_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                'cuda_reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024),
                'cuda_max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024)
            })
        
        return stats


# 简单测试
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试CognitiveCompiler
    compiler = CognitiveCompiler()
    print(f"Compiler initialized for: {compiler.target_device}")
    
    # 测试GPUMemoryManager
    memory_manager = GPUMemoryManager()
    print(f"Memory manager stats: {memory_manager.get_memory_stats()}")
    
    print("Optimization module test complete.")