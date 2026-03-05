#!/usr/bin/env python3
"""
Error Handling Utilities for AGI Model Training

This module provides utility functions for consistent error handling and logging
across all model training operations.
"""

import logging
import traceback
import sys
import json
from typing import Dict, Any, Optional, Callable, TypeVar, cast
from functools import wraps
import time

# Type variable for generic function wrapping
T = TypeVar('T')

def get_model_logger(model_name: str) -> logging.Logger:
    """
    Get a standardized logger for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"model.{model_name}")
    
    # Configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

def log_training_error(logger: logging.Logger, 
                      error: Exception, 
                      context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Log a training error with context.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context information
        
    Returns:
        Error information dictionary
    """
    error_info = {
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "error_traceback": traceback.format_exc(),
        "timestamp": time.time(),
        "context": context or {},
    }
    
    logger.error(f"Training error: {error.__class__.__name__}: {str(error)}")
    logger.debug(f"Error details: {json.dumps(error_info, default=str)}")
    
    return error_info

def retry_on_failure(max_retries: int = 3, 
                    delay: float = 1.0,
                    exponential_backoff: bool = True,
                    retry_on: tuple = (Exception,)):
    """
    Decorator for retrying functions on failure.
    
    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
        exponential_backoff: Whether to use exponential backoff
        retry_on: Exception types to retry on
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        # Log retry attempt
                        logger = getattr(args[0], 'logger', None) if args else None
                        if logger:
                            logger.warning(
                                f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                                f"after error: {e.__class__.__name__}: {str(e)}"
                            )
                        
                        # Wait before retry
                        time.sleep(current_delay)
                        
                        # Increase delay for exponential backoff
                        if exponential_backoff:
                            current_delay *= 2
                    else:
                        # Max retries exceeded
                        if logger:
                            logger.error(
                                f"Max retries ({max_retries}) exceeded for {func.__name__}. "
                                f"Last error: {e.__class__.__name__}: {str(e)}"
                            )
                        raise
            
            # This should never be reached, but just in case
            raise last_exception if last_exception else RuntimeError("Unexpected error in retry logic")
        
        return wrapper
    return decorator

def safe_training_execution(logger: logging.Logger, 
                          fallback_result: Optional[Dict[str, Any]] = None):
    """
    Decorator for safe training execution with error recovery.
    
    Args:
        logger: Logger instance
        fallback_result: Result to return if training fails
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error
                error_info = log_training_error(logger, e, {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs),
                })
                
                # Return fallback result if provided
                if fallback_result is not None:
                    fallback = fallback_result.copy()
                    fallback.update({
                        "error": error_info,
                        "success": False,
                        "training_completed": False,
                    })
                    return fallback
                
                # Re-raise if no fallback
                raise
        
        return wrapper
    return decorator

def validate_training_data(data: Any, 
                          logger: Optional[logging.Logger] = None) -> bool:
    """
    Validate training data before training.
    
    Args:
        data: Training data to validate
        logger: Optional logger for validation messages
        
    Returns:
        True if data is valid, False otherwise
    """
    if data is None:
        if logger:
            logger.error("Training data is None")
        return False
    
    if hasattr(data, '__len__'):
        if len(data) == 0:
            if logger:
                logger.error("Training data is empty")
            return False
    
    # Check for PyTorch tensors
    try:
        import torch
        if isinstance(data, torch.Tensor):
            if torch.any(torch.isnan(data)):
                if logger:
                    logger.error("Training data contains NaN values")
                return False
            if torch.any(torch.isinf(data)):
                if logger:
                    logger.error("Training data contains infinite values")
                return False
    except ImportError:
        pass  # PyTorch not available, skip tensor checks
    
    # Check for numpy arrays
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            if np.any(np.isnan(data)):
                if logger:
                    logger.error("Training data contains NaN values")
                return False
            if np.any(np.isinf(data)):
                if logger:
                    logger.error("Training data contains infinite values")
                return False
    except ImportError:
        pass  # NumPy not available, skip array checks
    
    if logger:
        logger.info("Training data validation passed")
    
    return True

def log_training_progress(logger: logging.Logger,
                         epoch: int,
                         total_epochs: int,
                         loss: float,
                         metrics: Optional[Dict[str, float]] = None,
                         learning_rate: Optional[float] = None):
    """
    Log training progress in a standardized format.
    
    Args:
        logger: Logger instance
        epoch: Current epoch number
        total_epochs: Total number of epochs
        loss: Current loss value
        metrics: Additional metrics dictionary
        learning_rate: Current learning rate
    """
    progress_percent = (epoch + 1) / total_epochs * 100
    
    log_message = (
        f"Epoch {epoch + 1}/{total_epochs} ({progress_percent:.1f}%) - "
        f"Loss: {loss:.6f}"
    )
    
    if learning_rate is not None:
        log_message += f" - LR: {learning_rate:.6f}"
    
    if metrics:
        metrics_str = ", ".join(f"{k}: {v:.6f}" for k, v in metrics.items())
        log_message += f" - Metrics: {metrics_str}"
    
    logger.info(log_message)

def create_error_response(error: Exception,
                         context: Dict[str, Any] = None,
                         include_traceback: bool = False) -> Dict[str, Any]:
    """
    Create a standardized error response for training functions.
    
    Args:
        error: Exception that occurred
        context: Additional context information
        include_traceback: Whether to include traceback in response
        
    Returns:
        Standardized error response dictionary
    """
    response = {
        "success": False,
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "timestamp": time.time(),
        "context": context or {},
    }
    
    if include_traceback:
        response["traceback"] = traceback.format_exc()
    
    return response

def benchmark_training_step(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to benchmark training step execution time.
    
    Args:
        func: Training step function
        
    Returns:
        Decorated function with benchmarking
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log benchmark if logger is available
            if args and hasattr(args[0], 'logger'):
                args[0].logger.debug(
                    f"Training step '{func.__name__}' took {execution_time:.4f} seconds"
                )
            
            # Add benchmark info to result if it's a dictionary
            if isinstance(result, dict):
                result = result.copy()
                result["_benchmark"] = {
                    "step_name": func.__name__,
                    "execution_time_seconds": execution_time,
                    "timestamp": start_time,
                }
            
            return result
        except Exception as e:
            # Log benchmark even on error
            execution_time = time.time() - start_time
            if args and hasattr(args[0], 'logger'):
                args[0].logger.error(
                    f"Training step '{func.__name__}' failed after {execution_time:.4f} seconds: "
                    f"{e.__class__.__name__}: {str(e)}"
                )
            raise
    
    return wrapper


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example logger
    logger = get_model_logger("example_model")
    
    # Example of safe training execution
    @safe_training_execution(logger, fallback_result={"success": False})
    def example_training_function(data):
        if not validate_training_data(data, logger):
            raise ValueError("Invalid training data")
        
        # Simulate training
        logger.info("Training started")
        time.sleep(0.1)
        
        # Simulate error
        if data == "error":
            raise RuntimeError("Simulated training error")
        
        return {
            "success": True,
            "loss": 0.123,
            "accuracy": 0.95,
        }
    
    # Test successful training
    print("Test 1: Successful training")
    result = example_training_function("valid_data")
    print(f"Result: {result}")
    
    print("\nTest 2: Training with error (should return fallback)")
    result = example_training_function("error")
    print(f"Result: {result}")
    
    # Test retry decorator
    @retry_on_failure(max_retries=2, delay=0.1)
    def example_retry_function(attempt_num):
        if attempt_num < 2:
            raise ConnectionError(f"Attempt {attempt_num} failed")
        return f"Attempt {attempt_num} succeeded"
    
    print("\nTest 3: Retry function")
    try:
        result = example_retry_function(0)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nError handling utilities test completed.")