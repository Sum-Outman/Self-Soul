"""
Model Training API System

Provides complete model training management API with support for from-scratch training, joint training, autonomous learning, and external API integration
"""

import asyncio
import json
import time
import uuid
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from pydantic import BaseModel
import logging

# Compatibility patch: Python 3.6 does not have datetime.fromisoformat
# Create a safe fromisoformat function that works across Python versions
def safe_fromisoformat(date_string: str) -> datetime:
    """Parse ISO format datetime string (compatible implementation for Python < 3.7)"""
    # If datetime.fromisoformat exists, use it
    if hasattr(datetime, 'fromisoformat') and callable(datetime.fromisoformat):
        return datetime.fromisoformat(date_string)
    
    # Fallback implementation for Python < 3.7
    try:
        # Try to parse with datetime.strptime for common ISO format
        # Format: YYYY-MM-DDTHH:MM:SS[.ffffff][+HH:MM[:SS[.ffffff]]]
        # Simplified version that handles most common cases
        if 'T' in date_string:
            # Contains time component
            date_part, time_part = date_string.split('T', 1)
            
            # Handle timezone offset if present
            if '+' in time_part:
                time_part, tz_part = time_part.split('+', 1)
                # Ignore timezone for now, just parse naive datetime
            elif '-' in time_part and time_part.count('-') >= 3:
                # Might be timezone with negative offset, handle carefully
                # This is a simplified approach
                pass
            
            # Parse date
            year, month, day = map(int, date_part.split('-'))
            
            # Parse time
            if '.' in time_part:
                time_without_fraction, fraction = time_part.split('.', 1)
                # Remove any trailing Z
                if fraction.endswith('Z'):
                    fraction = fraction[:-1]
                microsecond = int(float(f'0.{fraction}') * 1_000_000)
            else:
                time_without_fraction = time_part
                microsecond = 0
                if time_without_fraction.endswith('Z'):
                    time_without_fraction = time_without_fraction[:-1]
            
            hour, minute, second = map(int, time_without_fraction.split(':'))
            
            return datetime(year, month, day, hour, minute, second, microsecond)
        else:
            # Just date part
            year, month, day = map(int, date_string.split('-'))
            return datetime(year, month, day)
    except Exception as e:
        # Fallback to using dateutil if available, otherwise raise
        try:
            from dateutil.parser import parse
            return parse(date_string)
        except ImportError:
            raise ValueError(f"Failed to parse ISO format string: {date_string}. Error: {e}")

# Compatibility patch: Python 3.6 does not support asyncio.to_thread
if not hasattr(asyncio, 'to_thread'):
    import functools
    import concurrent.futures
    async def to_thread(func, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
    asyncio.to_thread = to_thread

from core.training_manager import TrainingManager
from core.enhanced_training_manager import EnhancedTrainingManager
from core.model_registry import get_model_registry
from core.joint_training_coordinator import JointTrainingCoordinator
from core.self_learning import AGISelfLearningSystem
from core.external_api_service import ExternalAPIService
from core.knowledge_manager import knowledge_manager
from core.dataset_manager import DatasetManager
from core.device_manager import get_device_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/training", tags=["training"])

# Global training manager instances - lazy initialization
_base_training_manager = None
_training_manager = None
_model_registry = None
_joint_training_coordinator = None
_autonomous_learning_manager = None
_external_api_service = None
_dataset_manager = None

def execute_with_recovery(operation_name: str, operation_func: Callable[..., Any], max_retries: int = 2, 
                         recovery_func: Optional[Callable[[], Any]] = None, fallback_value: Any = None, **kwargs: Any) -> Any:
    """
    Execute training operations with error recovery mechanism
    
    Args:
        operation_name: Operation name (for logging)
        operation_func: Operation function to execute
        max_retries: Maximum retry attempts
        recovery_func: Recovery function (called after all retries fail)
        fallback_value: Fallback return value (used when all recovery fails)
        **kwargs: Arguments passed to the operation function
        
    Returns:
        Operation result or fallback value
    """
    retries = 0
    last_exception = None
    
    while retries <= max_retries:
        try:
            if retries > 0:
                logger.info(f"Retrying {operation_name}, attempt {retries}")
                # Exponential backoff: wait longer for each retry
                wait_time = min(1.0 * (2 ** (retries - 1)), 10.0)
                time.sleep(wait_time)
            
            result = operation_func(**kwargs)
            if retries > 0:
                logger.info(f"{operation_name} succeeded after {retries} retries")
            return result
            
        except asyncio.CancelledError:
            # Task cancelled, should not retry, re-raise directly
            logger.info(f"{operation_name} was cancelled, not retrying")
            raise
        except concurrent.futures.CancelledError:
            # Concurrent task cancelled, should not retry, re-raise directly
            logger.info(f"{operation_name} was cancelled (concurrent), not retrying")
            raise
        except KeyboardInterrupt:
            # User interrupted, should not retry, re-raise directly
            logger.info(f"{operation_name} was interrupted by user, not retrying")
            raise
        except Exception as e:
            last_exception = e
            retries += 1
            logger.warning(
                f"{operation_name} failed, attempt {retries-1}: {str(e)[:100]}"
            )
    
    # All retries failed
    error_message = f"{operation_name} failed after {max_retries} retries"
    if last_exception:
        error_message += f": {last_exception}"
    
    logger.error(error_message)
    
    # Try recovery function
    if recovery_func:
        try:
            logger.info(f"Attempting to execute recovery function for {operation_name}")
            recovery_result = recovery_func()
            if recovery_result is not None:
                return recovery_result
        except Exception as recovery_error:
            logger.error(f"Recovery function failed: {recovery_error}")
    
    # Return fallback value
    if fallback_value is not None:
        logger.warning(f"{operation_name} using fallback value")
        return fallback_value
    else:
            # If no fallback value, re-raise the last exception
            raise last_exception if last_exception else Exception(f"{operation_name} failed")

async def async_execute_with_recovery(operation_name: str, operation_func: Callable[..., Any], max_retries: int = 2, 
                                    recovery_func: Optional[Callable[[], Any]] = None, fallback_value: Any = None, **kwargs: Any) -> Any:
    """
    Execute training operations with error recovery mechanism (asynchronous version)
    
    Args:
        operation_name: Operation name (for logging)
        operation_func: Operation function to execute (can be sync or async)
        max_retries: Maximum retry attempts
        recovery_func: Recovery function (called after all retries fail)
        fallback_value: Fallback return value (used when all recovery fails)
        **kwargs: Arguments passed to the operation function
        
    Returns:
        Operation result or fallback value
    """
    retries = 0
    last_exception = None
    
    while retries <= max_retries:
        try:
            if retries > 0:
                logger.info(f"Retrying {operation_name}, attempt {retries}")
                # Exponential backoff: wait longer for each retry (non-blocking)
                wait_time = min(1.0 * (2 ** (retries - 1)), 10.0)
                await asyncio.sleep(wait_time)
            
            # Check if operation_func is a coroutine function
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(**kwargs)
            else:
                # Run synchronous function in thread pool to avoid blocking
                result = await asyncio.to_thread(operation_func, **kwargs)
            
            if retries > 0:
                logger.info(f"{operation_name} succeeded after {retries} retries")
            return result
            
        except asyncio.CancelledError:
            # Task cancelled, should not retry, re-raise directly
            logger.info(f"{operation_name} was cancelled, not retrying")
            raise
        except concurrent.futures.CancelledError:
            # Concurrent task cancelled, should not retry, re-raise directly
            logger.info(f"{operation_name} was cancelled (concurrent), not retrying")
            raise
        except KeyboardInterrupt:
            # User interrupted, should not retry, re-raise directly
            logger.info(f"{operation_name} was interrupted by user, not retrying")
            raise
        except Exception as e:
            last_exception = e
            retries += 1
            logger.warning(
                f"{operation_name} failed, attempt {retries-1}: {str(e)[:100]}"
            )
    
    # All retries failed
    error_message = f"{operation_name} failed after {max_retries} retries"
    if last_exception:
        error_message += f": {last_exception}"
    
    logger.error(error_message)
    
    # Try recovery function
    if recovery_func:
        try:
            logger.info(f"Attempting to execute recovery function for {operation_name}")
            # Check if recovery_func is a coroutine function
            if asyncio.iscoroutinefunction(recovery_func):
                recovery_result = await recovery_func()
            else:
                recovery_result = recovery_func()
            
            if recovery_result is not None:
                return recovery_result
        except Exception as recovery_error:
            logger.error(f"Recovery function failed: {recovery_error}")
    
    # Return fallback value
    if fallback_value is not None:
        logger.warning(f"{operation_name} using fallback value")
        return fallback_value
    else:
        # If no fallback value, re-raise the last exception
        raise last_exception if last_exception else Exception(f"{operation_name} failed")

def get_base_training_manager() -> TrainingManager:
    global _base_training_manager
    if _base_training_manager is None:
        _base_training_manager = TrainingManager()
    return _base_training_manager

def get_training_manager() -> EnhancedTrainingManager:
    global _training_manager, _base_training_manager
    if _training_manager is None:
        if _base_training_manager is None:
            _base_training_manager = TrainingManager()
        _training_manager = EnhancedTrainingManager(training_manager=_base_training_manager)
    return _training_manager

def get_model_registry_instance() -> Any:
    global _model_registry
    if _model_registry is None:
        _model_registry = get_model_registry()
    return _model_registry

def get_joint_training_coordinator() -> JointTrainingCoordinator:
    global _joint_training_coordinator
    if _joint_training_coordinator is None:
        _joint_training_coordinator = JointTrainingCoordinator(model_ids=[], parameters={'training_strategy': 'standard'})
    return _joint_training_coordinator

def get_autonomous_learning_manager() -> AGISelfLearningSystem:
    global _autonomous_learning_manager
    if _autonomous_learning_manager is None:
        _autonomous_learning_manager = AGISelfLearningSystem()
    return _autonomous_learning_manager

def get_external_api_service() -> ExternalAPIService:
    global _external_api_service
    if _external_api_service is None:
        _external_api_service = ExternalAPIService()
    return _external_api_service

def get_dataset_manager() -> DatasetManager:
    global _dataset_manager
    if _dataset_manager is None:
        _dataset_manager = DatasetManager()
    return _dataset_manager

# Currently running training tasks
active_trainings: Dict[str, Dict[str, Any]] = {}
active_trainings_lock = asyncio.Lock()

# Autonomous learning task storage
active_learnings: Dict[str, Dict[str, Any]] = {}
active_learnings_lock = asyncio.Lock()

class TrainingRequest(BaseModel):
    """Training request model"""
    model_id: str
    training_type: str  # from_scratch, fine_tune, joint, autonomous
    dataset_config: Dict[str, Any]
    training_config: Dict[str, Any]
    
    # Optional parameters
    device: Optional[str] = None  # cpu, cuda, mps, auto
    external_api_config: Optional[Dict[str, Any]] = None
    external_model_assistance: Optional[Dict[str, Any]] = None
    joint_models: Optional[List[str]] = None
    autonomous_learning_config: Optional[Dict[str, Any]] = None

class TrainingStartRequest(BaseModel):
    """Training start request model - matches frontend request format"""
    models: List[str]
    dataset_id: str
    parameters: Dict[str, Any]
    training_mode: str  # individual, joint
    from_scratch: bool = False
    device: str = 'auto'
    external_model_assistance: bool = False
    external_model_id: Optional[str] = None

class KnowledgeLearningRequest(BaseModel):
    """Knowledge base learning request model"""
    model: str
    domains: List[str]
    priority: str  # balanced, exploration, exploitation
    intensity: float  # 0.1 to 1.0
    max_time: int  # minutes

class KnowledgeTransferRequest(BaseModel):
    """Knowledge transfer request model"""
    source_domain: str
    target_domain: str
    transfer_strategy: str = "semantic_similarity"  # semantic_similarity, structural_analogy, conceptual_mapping, cross_domain_inference
    transfer_intensity: float = 0.7  # 0.1 to 1.0
    max_items: Optional[int] = None  # Maximum number of transfer items, None means automatic selection

class TrainingStatus(BaseModel):
    """Training status model"""
    training_id: str
    model_id: str
    status: str  # pending, running, completed, failed, stopped
    progress: float  # 0-100
    current_epoch: int
    total_epochs: int
    loss: float
    accuracy: float
    start_time: str
    device: Optional[str] = None  # cpu, cuda, mps, auto
    device_info: Optional[Dict[str, Any]] = None  # Device detailed information
    estimated_completion: Optional[str] = None
    error_message: Optional[str] = None

class ModelConfig(BaseModel):
    """Model configuration model"""
    model_type: str  # language, vision, audio, etc.
    model_name: str
    config: Dict[str, Any]
    training_config: Dict[str, Any]
    external_api_config: Optional[Dict[str, Any]] = None

class DatasetConfig(BaseModel):
    """Dataset configuration model"""
    dataset_type: str  # text, image, audio, multimodal
    dataset_path: str
    preprocessing_config: Dict[str, Any]
    split_config: Dict[str, Any]  # train, validation, test splits

@router.post("/start")
async def start_training(request: TrainingStartRequest, background_tasks: BackgroundTasks):
    """Start model training (matches frontend request format)"""
    try:
        import sys
        logger.debug(f"[API_START_TRAINING] Training API called - models={request.models}, dataset_id={request.dataset_id}, training_mode={request.training_mode}")
        logger.info(f"[API_START_TRAINING] Training API called - models={request.models}, dataset_id={request.dataset_id}, training_mode={request.training_mode}")
        logger.info(f"Training start request received: models={request.models}, dataset_id={request.dataset_id}, training_mode={request.training_mode}")
        # Generate training ID
        training_id = str(uuid.uuid4())
        
        # Validate all models exist
        logger.info(f"Validating models: {request.models}")
        # Get model registry instance
        model_registry = get_model_registry_instance()
        for model_id in request.models:
            logger.info(f"Checking if model '{model_id}' is registered")
            if not model_registry.is_model_registered(model_id):
                logger.error(f"Model '{model_id}' not found in registry")
                raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        logger.info("All models validated successfully")
        
        # Merge parameters: combine TrainingStartRequest fields into parameters dictionary
        merged_parameters = request.parameters.copy()
        merged_parameters.update({
            'dataset_id': request.dataset_id,
            'training_mode': request.training_mode,
            'from_scratch': request.from_scratch,
            'device': request.device,
            'external_model_assistance': request.external_model_assistance,
            'external_model_id': request.external_model_id
        })
        
        # Record training start
        # Get device information
        device_manager = get_device_manager()
        device_type = request.device
        if device_type == 'auto':
            device_type = device_manager.get_current_device()
        
        device_info = device_manager.get_device_info(device_type)
        
        active_trainings[training_id] = {
            "training_id": training_id,
            "model_ids": request.models,
            "model_id": request.models[0] if request.models else None,
            "training_mode": request.training_mode,
            "training_type": request.training_mode,
            "status": "starting",
            "started_at": datetime.now().isoformat(),
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": request.parameters.get("epochs", 100),
            "loss": 0.0,
            "accuracy": 0.0,
            "device": request.device,
            "device_type": device_type,
            "device_info": device_info,
            "from_scratch": request.from_scratch
        }
        
        logger.info(f"Training {training_id} registered in active_trainings")
        
        # Run training in background
        async def run_training():
            try:
                import sys
                logger.debug(f"[RUN_TRAINING_START] Training background task started - training_id={training_id}, model_id={request.models[0] if request.models else 'unknown'}")
                logger.info(f"[TRAINING_START] Training background task started - training_id={training_id}, model_id={request.models[0] if request.models else 'unknown'}")
                logger.info(f"[DEBUG] Background training task started for {training_id}")
                logger.info(f"Background training task started for {training_id}")
                # Record training parameter information
                logger.info(f"[TRAINING_INFO] Training mode: {request.training_mode}, Device: {request.device}, External model assistance: {request.external_model_assistance}")
                # Select appropriate training manager based on training mode
                if request.training_mode == "joint" and len(request.models) > 1:
                    # Use joint training coordinator for joint training
                    logger.info(f"Starting joint training for models: {request.models}")
                    
                    # Create joint training coordinator instance
                    strategy = merged_parameters.get('strategy', 'standard')
                    joint_coordinator = JointTrainingCoordinator(
                        model_ids=request.models,
                        parameters={'training_strategy': strategy}
                    )
                    
                    # Prepare training tasks with real data
                    training_tasks = []
                    dataset_manager = get_dataset_manager()
                    
                    for model_id in request.models:
                        # Get training data for each model
                        training_data = {}
                        dataset_result = dataset_manager.get_training_dataset_for_model(model_id, request.dataset_id)
                        
                        if dataset_result.get("success", False):
                            training_data = dataset_result.get("content", {})
                            logger.info(f"Using dataset {request.dataset_id} for model {model_id}")
                        else:
                            # Create basic dataset if specific dataset not found
                            basic_dataset_result = dataset_manager.create_basic_dataset(model_id)
                            if basic_dataset_result.get("success", False):
                                training_data = basic_dataset_result.get("content", {})
                                logger.info(f"Using basic dataset for model {model_id}")
                            else:
                                logger.warning(f"Failed to get training data for model {model_id}, using empty data")
                        
                        task = {
                            'model_id': model_id,
                            'training_data': training_data,
                            'epochs': merged_parameters.get('epochs', 10),
                            'batch_size': merged_parameters.get('batch_size', 32),
                            'priority': merged_parameters.get('priority', 1),
                            'learning_rate': merged_parameters.get('learning_rate', 0.001),
                            'strategy': strategy
                        }
                        training_tasks.append(task)
                    
                    # Schedule training tasks
                    schedule_result = joint_coordinator.schedule_training(training_tasks)
                    if not schedule_result.get('status') == 'success':
                        raise Exception(f"Failed to schedule joint training: {schedule_result.get('message')}")
                    
                    # Execute training
                    joint_result = await joint_coordinator.execute_training()
                    
                    # Generate a virtual job_id for tracking
                    job_id = f"joint_{training_id}"
                    
                    # Update training status
                    if joint_result.get('status') == 'success':
                        active_trainings[training_id]["status"] = "completed"
                        active_trainings[training_id]["result"] = joint_result
                        active_trainings[training_id]["completed_at"] = datetime.now().isoformat()
                        logger.info(f"Joint training {training_id} completed successfully")
                    else:
                        active_trainings[training_id]["status"] = "failed"
                        active_trainings[training_id]["error"] = joint_result.get('error', 'Unknown error')
                        active_trainings[training_id]["failed_at"] = datetime.now().isoformat()
                        logger.error(f"Joint training {training_id} failed: {joint_result.get('error')}")
                else:
                    # Individual model training
                    model_id = request.models[0] if request.models else None
                    if not model_id:
                        raise Exception("No model specified for individual training")
                    
                    # Check if external model assistance is enabled
                    if request.external_model_assistance:
                        logger.info(f"Starting training with external model assistance for model: {model_id}")
                        
                        # Prepare training data
                        training_data = {
                            "dataset_id": request.dataset_id,
                            "parameters": merged_parameters,
                            "model_id": model_id,
                            "training_mode": request.training_mode
                        }
                        
                        # Get external model ID
                        external_model_id = request.external_model_id
                        if not external_model_id:
                            # Automatically select external model based on model type
                            external_model_id = training_manager._get_recommended_external_model(model_id)
                        
                        # Use external model assistance to train local model
                        training_result = await training_manager.train_with_external_model_assistance(
                            model_name=model_id,
                            training_data=training_data,
                            external_model_id=external_model_id
                        )
                        
                        if training_result.get("success", False):
                            job_id = training_result.get("training_job_id")
                            active_trainings[training_id]["status"] = "running"
                            active_trainings[training_id]["job_id"] = job_id
                            active_trainings[training_id]["started_at"] = datetime.now().isoformat()
                            active_trainings[training_id]["external_model_used"] = external_model_id
                            logger.info(f"Training {training_id} started with external model assistance, job_id: {job_id}")
                            
                            # Update training status to completed (because train_with_external_model_assistance is synchronous)
                            active_trainings[training_id]["status"] = "completed"
                            active_trainings[training_id]["result"] = training_result
                            active_trainings[training_id]["completed_at"] = datetime.now().isoformat()
                            logger.info(f"Training with external model assistance {training_id} completed successfully")
                        else:
                            active_trainings[training_id]["status"] = "failed"
                            active_trainings[training_id]["error"] = training_result.get("message", "External model assistance training failed")
                            active_trainings[training_id]["failed_at"] = datetime.now().isoformat()
                            logger.error(f"Training with external model assistance {training_id} failed: {training_result.get('message')}")
                    else:
                        # Call training manager to start training (without external model assistance) - use async thread to avoid blocking
                        
                        # Define helper function to handle dataset creation and training startup
                        async def start_training_with_dataset(model_id, parameters, dataset_id):
                            """Start training and automatically create dataset (if needed)"""
                            try:
                                print(f"[PRINT_DEBUG] start_training_with_dataset called: model_id={model_id}, dataset_id={dataset_id}")
                                logger.info(f"[DEBUG] start_training_with_dataset start: model_id={model_id}, dataset_id={dataset_id}")
                                
                                # Check memory usage before starting training
                                try:
                                    import psutil
                                    memory_info = psutil.virtual_memory()
                                    memory_percent = memory_info.percent
                                    
                                    # Get memory threshold from memory optimizer if available
                                    memory_threshold = 85  # Default threshold
                                    try:
                                        from core.memory_optimization import memory_optimizer
                                        memory_threshold = memory_optimizer.max_memory_usage
                                        logger.debug(f"[MEMORY_THRESHOLD] Using memory optimizer threshold: {memory_threshold}%")
                                    except (ImportError, AttributeError):
                                        logger.debug(f"[MEMORY_THRESHOLD] Using default threshold: {memory_threshold}%")
                                    
                                    logger.info(f"[MEMORY_CHECK] Current memory usage: {memory_percent:.1f}% (threshold: {memory_threshold}%)")
                                    
                                    if memory_percent > memory_threshold:
                                        logger.warning(f"[MEMORY_WARNING] Memory usage {memory_percent:.1f}% exceeds threshold {memory_threshold}%")
                                        
                                        # Try to perform memory optimization if available
                                        try:
                                            from core.memory_optimization import memory_optimizer
                                            if memory_optimizer.should_optimize():
                                                logger.info(f"[MEMORY_OPTIMIZATION] Performing memory optimization before training")
                                                memory_optimizer.optimize_memory()
                                                
                                                # Check memory again after optimization
                                                memory_info = psutil.virtual_memory()
                                                memory_percent = memory_info.percent
                                                logger.info(f"[MEMORY_AFTER_OPTIMIZATION] Memory usage after optimization: {memory_percent:.1f}%")
                                        except ImportError:
                                            logger.debug(f"[MEMORY_OPTIMIZATION] Memory optimizer not available")
                                        except Exception as e:
                                            logger.warning(f"[MEMORY_OPTIMIZATION_ERROR] Error during memory optimization: {e}")
                                        
                                        # If memory is still too high, log warning but continue
                                        if memory_percent > memory_threshold:
                                            logger.warning(f"[MEMORY_HIGH] Memory usage still high ({memory_percent:.1f}%), proceeding with training anyway")
                                except ImportError:
                                    logger.debug(f"[MEMORY_CHECK] psutil not available, skipping memory check")
                                except Exception as e:
                                    logger.warning(f"[MEMORY_CHECK_ERROR] Error checking memory: {e}")
                                
                                # 1. Prepare dataset configuration with error recovery
                                data_config = None
                                dataset_manager = get_dataset_manager()
                                
                                # Try to get or create dataset with recovery mechanism
                                if dataset_id:
                                    logger.info(f"[DEBUG] Trying to get existing dataset: {dataset_id}")
                                    
                                    # Define dataset retrieval function with recovery
                                    def get_dataset_func():
                                        nonlocal dataset_manager
                                        return dataset_manager.get_training_dataset_for_model(model_id, dataset_id)
                                    
                                    # Define recovery function for dataset retrieval failure
                                    def dataset_retrieval_recovery():
                                        logger.warning(f"Dataset retrieval failed, will create basic dataset instead")
                                        return {"success": False, "message": "Dataset retrieval failed, fallback to basic dataset"}
                                    
                                    # Execute dataset retrieval with recovery (asynchronous)
                                    dataset_result = await async_execute_with_recovery(
                                        operation_name=f"Dataset retrieval for {dataset_id}",
                                        operation_func=get_dataset_func,
                                        max_retries=2,
                                        recovery_func=dataset_retrieval_recovery,
                                        fallback_value={"success": False, "message": "Dataset retrieval failed"}
                                    )
                                    
                                    print(f"[PRINT_DEBUG] dataset_result: {dataset_result}")
                                    logger.info(f"[DEBUG] Dataset retrieval result: {dataset_result}")
                                    if dataset_result.get("success", False):
                                        data_config = {
                                            "dataset_id": dataset_id,
                                            "dataset_name": dataset_result.get("dataset_name"),
                                            "content": dataset_result.get("content", {})
                                        }
                                        print(f"[PRINT_DEBUG] data_config created: {data_config}")
                                        logger.info(f"Using existing dataset {dataset_id} for model {model_id}")
                                    else:
                                        logger.warning(f"Failed to get dataset {dataset_id}, creating basic dataset: {dataset_result.get('message')}")
                                        # Fallback to create basic dataset
                                        dataset_id = None  # Clear dataset_id to trigger basic dataset creation
                                
                                # If no dataset ID or retrieval failed, create basic dataset with recovery
                                if not dataset_id or data_config is None:
                                    logger.info(f"[DEBUG] Creating basic dataset for model {model_id}")
                                    
                                    # Define dataset creation function
                                    def create_dataset_func():
                                        nonlocal dataset_manager
                                        return dataset_manager.create_basic_dataset(model_id)
                                    
                                    # Execute dataset creation with recovery (asynchronous)
                                    dataset_result = await async_execute_with_recovery(
                                        operation_name=f"Basic dataset creation for model {model_id}",
                                        operation_func=create_dataset_func,
                                        max_retries=2,
                                        recovery_func=lambda: {"success": False, "message": "Dataset creation failed"},
                                        fallback_value={"success": False, "message": "Dataset creation failed after retries"}
                                    )
                                    
                                    logger.info(f"[DEBUG] Basic dataset creation result: {dataset_result}")
                                    if dataset_result.get("success", False):
                                        data_config = {
                                            "dataset_id": dataset_result.get("dataset_name"),
                                            "dataset_name": dataset_result.get("dataset_name"),
                                            "content": dataset_result.get("content", {})
                                        }
                                        logger.info(f"Created basic dataset {data_config['dataset_id']} for model {model_id}")
                                    else:
                                        logger.error(f"Failed to create basic dataset: {dataset_result.get('message')}")
                                        return {"success": False, "message": f"Failed to create basic dataset: {dataset_result.get('message')}"}
                                
                                # Ensure data_config is not None
                                if data_config is None:
                                    logger.error("Failed to prepare data configuration")
                                    return {"success": False, "message": "Failed to prepare data configuration"}
                                
                                logger.info(f"[DEBUG] Calling base_training_manager.start_training, model_id={model_id}")
                                # Get base training manager instance
                                base_training_manager = get_base_training_manager()
                                logger.info(f"[DEBUG] base_training_manager obtained: {type(base_training_manager)}")
                                
                                # Define training start function with recovery
                                def start_training_func():
                                    logger.info(f"[DEBUG] start_training_func called, model_id={model_id}")
                                    print(f"[PRINT_DEBUG] start_training_func called, model_id={model_id}, data_config={data_config}, parameters={parameters}")
                                    try:
                                        result = base_training_manager.start_training(
                                            model_id=model_id,
                                            data_config=data_config,
                                            training_params=parameters
                                        )
                                        print(f"[PRINT_DEBUG] start_training_func success, result: {result}")
                                        logger.info(f"[DEBUG] start_training_func result: {result}")
                                        return result
                                    except Exception as e:
                                        print(f"[PRINT_DEBUG] start_training_func EXCEPTION: {e}")
                                        import traceback
                                        print(f"[PRINT_DEBUG] traceback: {traceback.format_exc()}")
                                        raise
                                
                                # Define recovery function for training start failure
                                def training_start_recovery():
                                    logger.warning(f"Training start failed, returning error result")
                                    return {"success": False, "message": "Training failed to start", "job_id": None}
                                
                                # Execute training start with recovery (asynchronous)
                                print(f"[PRINT_DEBUG] Before async_execute_with_recovery")
                                result = await async_execute_with_recovery(
                                    operation_name=f"Training start for model {model_id}",
                                    operation_func=start_training_func,
                                    max_retries=1,  # Training start is critical, but we don't want infinite retries
                                    recovery_func=training_start_recovery,
                                    fallback_value={"success": False, "message": "Training failed to start after retries", "job_id": None}
                                )
                                print(f"[PRINT_DEBUG] After async_execute_with_recovery, result: {result}")
                                
                                logger.info(f"[DEBUG] async_execute_with_recovery return result: {result}")
                                logger.info(f"[DEBUG] result type: {type(result)}, keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                                if isinstance(result, dict):
                                    logger.info(f"[DEBUG] result.success: {result.get('success')}")
                                    logger.info(f"[DEBUG] result.job_id: {result.get('job_id')}")
                                    logger.info(f"[DEBUG] result.message: {result.get('message')}")
                                    print(f"[PRINT_DEBUG] result dict - success: {result.get('success')}, job_id: {result.get('job_id')}, message: {result.get('message')}")
                                else:
                                    print(f"[PRINT_DEBUG] result is not a dict: {type(result)}")
                                return result
                            except Exception as e:
                                logger.error(f"Error in start_training_with_dataset: {e}")
                                return {"success": False, "message": str(e)}
                        
                        logger.info(f"[DEBUG] Preparing to start training (asynchronous)")
                        
                        # Add retry mechanism for CancelledError
                        max_retries = 3
                        retry_count = 0
                        start_result = None
                        
                        while retry_count < max_retries:
                            try:
                                # Directly call the async function without run_in_executor
                                start_result = await start_training_with_dataset(model_id, merged_parameters, request.dataset_id)
                                break  # Success, exit retry loop
                            except asyncio.CancelledError as e:
                                retry_count += 1
                                logger.warning(f"[CANCELLED_ERROR] Training task cancelled (attempt {retry_count}/{max_retries}): {e}")
                                if retry_count < max_retries:
                                    # Exponential backoff before retry (non-blocking)
                                    wait_time = 0.5 * (2 ** (retry_count - 1))
                                    logger.info(f"[RETRY] Waiting {wait_time:.2f} seconds before retry {retry_count}")
                                    await asyncio.sleep(wait_time)
                                else:
                                    logger.error(f"[CANCELLED_ERROR_MAX_RETRIES] Max retries reached for cancelled task")
                                    start_result = {"success": False, "message": f"Training task cancelled after {max_retries} retries"}
                            except Exception as e:
                                logger.error(f"[TRAINING_ERROR] Error in training task: {e}")
                                start_result = {"success": False, "message": f"Training error: {str(e)}"}
                                break  # Non-cancellation error, don't retry
                        
                        # If all retries failed and start_result is still None
                        if start_result is None:
                            start_result = {"success": False, "message": "Failed to start training after retries"}
                        logger.info(f"[DEBUG] Training start result: {start_result}")
                        logger.info(f"[DEBUG_START_RESULT] start_result full content: {start_result}")
                        logger.info(f"[DEBUG_START_RESULT_KEYS] start_result keys: {list(start_result.keys()) if start_result else 'None'}")
                        
                        if not start_result.get('success', False):
                            logger.error(f"[DEBUG] Training startup failed: {start_result}")
                            logger.error(f"[DEBUG_FAILURE_DETAILS] Failure details - success field: {start_result.get('success')}, message: {start_result.get('message')}, job_id: {start_result.get('job_id')}")
                            active_trainings[training_id]["status"] = "failed"
                            active_trainings[training_id]["error"] = start_result.get('message', 'Failed to start training')
                            active_trainings[training_id]["failed_at"] = datetime.now().isoformat()
                            logger.error(f"Failed to start training {training_id}: {start_result.get('message')}")
                            return
                        
                        job_id = start_result.get('job_id')
                        logger.info(f"[DEBUG] job_id obtained from training manager: {job_id}")
                        logger.info(f"[DEBUG_JOB_ID_DETAILS] job_id type: {type(job_id)}, job_id value: '{job_id}'")
                        if not job_id:
                            logger.error(f"[DEBUG] Training manager did not return job_id, start_result: {start_result}")
                            logger.error(f"[DEBUG_NO_JOB_ID] start_result full content: {start_result}")
                            logger.error(f"[DEBUG_NO_JOB_ID_KEYS] start_result keys: {list(start_result.keys()) if start_result else 'None'}")
                            active_trainings[training_id]["status"] = "failed"
                            active_trainings[training_id]["error"] = "No job_id returned from training manager"
                            active_trainings[training_id]["failed_at"] = datetime.now().isoformat()
                            logger.error(f"Failed to start training {training_id}: No job_id returned")
                            return
                        
                        # Update training status
                        logger.info(f"[DEBUG] Update training status to running, training_id={training_id}, job_id={job_id}")
                        active_trainings[training_id]["status"] = "running"
                        active_trainings[training_id]["job_id"] = job_id
                        active_trainings[training_id]["started_at"] = datetime.now().isoformat()
                        
                        logger.info(f"[TRAINING_SUCCESS] Training successfully started - training_id={training_id}, job_id={job_id}, model_id={model_id}")
                        logger.info(f"Training {training_id} started with job_id: {job_id}")
                        
                        # Training has been started in background, no need to wait for completion
                        # Training status will be queried through other endpoints
                        logger.info(f"Training {training_id} started in background with job_id: {job_id}")
                        # Can start an async task here to periodically check training status (optional)
                        # But for response speed, we let training run in background, query status through other endpoints
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                active_trainings[training_id]["status"] = "failed"
                active_trainings[training_id]["error"] = str(e)
                active_trainings[training_id]["failed_at"] = datetime.now().isoformat()
                logger.error(f"[TRAINING_ERROR] Training failed - training_id={training_id}, error={e}", exc_info=True)
                logger.error(f"Training {training_id} failed: {e}\nError details:\n{error_details}")
        
        import sys
        logger.debug(f"[BACKGROUND_TASKS] Using background_tasks.add_task to schedule background training task, training_id={training_id}")
        logger.info(f"[DEBUG] Using background_tasks.add_task to schedule background training task, training_id={training_id}")
        # Start background task - using FastAPI's BackgroundTasks
        background_tasks.add_task(run_training)
        logger.info(f"[DEBUG] Training task has been added to background_tasks, training_id={training_id}")
        
        return {
            "success": True,
            "training_id": training_id,
            "message": f"Training started successfully for models: {', '.join(request.models)}",
            "device": request.device,
            "estimated_duration": router._estimate_training_duration(request.parameters)
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Failed to start training: {e}\nError details:\n{error_details}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{training_id}")
async def get_training_status(training_id: str):
    """Get training status"""
    logger.info(f"[STATUS_REQUEST] Get training status request - training_id={training_id}")
    start_time = time.time()
    
    try:
        # Set timeout to prevent endpoint blocking
        async def get_status():
            logger.info(f"[STATUS_GET_START] Start getting training status - training_id={training_id}")
            # Use lock to ensure thread-safe access to active_trainings
            async with active_trainings_lock:
                if training_id not in active_trainings:
                    logger.warning(f"Training '{training_id}' not found")
                    raise HTTPException(status_code=404, detail=f"Training '{training_id}' not found")
                
                training_info = active_trainings[training_id]
                logger.info(f"Training {training_id} status: {training_info['status']}, progress: {training_info['progress']}, job_id: {training_info.get('job_id')}")
            
            # Get real-time status from training manager (if job_id exists)
            real_time_status = None
            job_id = training_info.get("job_id")
            if job_id:
                logger.info(f"[REAL_TIME_STATUS] Getting real-time status from training manager - job_id={job_id}")
                try:
                    real_time_status = training_manager.get_job_status(job_id)
                    if real_time_status:
                        logger.info(f"[REAL_TIME_STATUS_SUCCESS] Got real-time status: {real_time_status}")
                        
                        # Use real-time status to update training_info
                        if 'progress' in real_time_status:
                            training_info["progress"] = real_time_status.get('progress', training_info["progress"])
                        if 'current_epoch' in real_time_status:
                            training_info["current_epoch"] = real_time_status.get('current_epoch', training_info["current_epoch"])
                        if 'total_epochs' in real_time_status:
                            training_info["total_epochs"] = real_time_status.get('total_epochs', training_info["total_epochs"])
                        if 'latest_metrics' in real_time_status and real_time_status['latest_metrics']:
                            latest_metrics = real_time_status['latest_metrics']
                            training_info["loss"] = latest_metrics.get('loss', training_info["loss"])
                            training_info["accuracy"] = latest_metrics.get('accuracy', training_info["accuracy"])
                        
                        # Always use real-time status from training manager to update status
                        if real_time_status.get('status'):
                            training_info["status"] = real_time_status.get('status')
                            # Also update status in active_trainings
                            async with active_trainings_lock:
                                active_trainings[training_id]["status"] = real_time_status.get('status')
                            
                    else:
                        logger.warning(f"[REAL_TIME_STATUS_NULL] Training manager returned null status - job_id={job_id}")
                except Exception as e:
                    logger.error(f"[REAL_TIME_STATUS_ERROR] Failed to get status from training manager - job_id={job_id}, error={e}")
            
            # Build training status response
            status_response = TrainingStatus(
                training_id=training_id,
                model_id=training_info["model_id"],
                status=training_info["status"],
                progress=training_info["progress"],
                current_epoch=training_info["current_epoch"],
                total_epochs=training_info["total_epochs"],
                loss=training_info["loss"],
                accuracy=training_info["accuracy"],
                start_time=training_info["started_at"],
                device=training_info.get("device"),
                device_info=training_info.get("device_info")
            )
            
            # If training is completed, add result information
            if training_info["status"] == "completed":
                status_response.estimated_completion = training_info.get("completed_at")
                
            # If training failed, add error information
            elif training_info["status"] == "failed":
                status_response.error_message = training_info.get("error")
            
            # If training is running, calculate estimated completion time
            elif training_info["status"] == "running":
                start_time = safe_fromisoformat(training_info["started_at"])
                elapsed_time = datetime.now() - start_time
                
                if training_info["progress"] > 0:
                    total_estimated = elapsed_time / (training_info["progress"] / 100)
                    estimated_completion = start_time + total_estimated
                    status_response.estimated_completion = estimated_completion.isoformat()
            
            logger.info(f"[STATUS_RESPONSE] Training status response prepared - training_id={training_id}, status={training_info['status']}, progress={training_info['progress']}")
            logger.info(f"Training status response prepared for {training_id}")
            return {
                "success": True,
                "training_status": status_response.dict()
            }
        
        # Set 3 second timeout (shortened timeout)
        try:
            result = await asyncio.wait_for(get_status(), timeout=3.0)
            elapsed_time = time.time() - start_time
            logger.info(f"[STATUS_COMPLETE] Training status query completed - training_id={training_id}, processing time={elapsed_time:.3f} seconds")
            return result
        except asyncio.TimeoutError:
            elapsed_time = time.time() - start_time
            logger.error(f"[STATUS_TIMEOUT] Training status query timeout - training_id={training_id}, waiting time={elapsed_time:.3f} seconds")
            logger.error(f"Timeout getting training status for {training_id}")
            raise HTTPException(status_code=408, detail=f"Request timeout for training status")
        
    except HTTPException:
        # Re-raise HTTP exception
        raise
    except Exception as e:
        logger.error(f"Error getting training status for {training_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/list")
async def list_trainings(
    status: Optional[str] = None,
    model_id: Optional[str] = None,
    limit: int = 50
):
    """List training tasks"""
    trainings = []
    
    for training_id, training_info in active_trainings.items():
        # Filter conditions
        if status and training_info["status"] != status:
            continue
        if model_id and training_info["model_id"] != model_id:
            continue
        
        training_data = {
            "training_id": training_id,
            "model_id": training_info["model_id"],
            "training_type": training_info["training_type"],
            "status": training_info["status"],
            "progress": training_info["progress"],
            "started_at": training_info["started_at"],
            "completed_at": training_info.get("completed_at"),
            "failed_at": training_info.get("failed_at")
        }
        
        trainings.append(training_data)
    
    # Limit return count
    trainings = trainings[-limit:]
    
    return {
        "success": True,
        "trainings": trainings,
        "total_count": len(trainings),
        "running_count": len([t for t in trainings if t["status"] == "running"]),
        "completed_count": len([t for t in trainings if t["status"] == "completed"]),
        "failed_count": len([t for t in trainings if t["status"] == "failed"])
    }

@router.delete("/clean/{training_id}")
async def clean_training(training_id: str):
    """Clean stuck training tasks (regardless of status)"""
    async with active_trainings_lock:
        if training_id not in active_trainings:
            raise HTTPException(status_code=404, detail=f"Training '{training_id}' not found")
        
        training_info = active_trainings[training_id]
        status = training_info["status"]
        
        # If training is running, try to stop it first
        if status == "running":
            try:
                training_manager.stop_training(training_id)
                logger.info(f"Stopped running training {training_id} before cleaning")
            except Exception as e:
                logger.warning(f"Failed to stop training {training_id}: {e}")
        
        # Remove from active_trainings
        del active_trainings[training_id]
        
        logger.info(f"Cleaned training {training_id} (status: {status})")
        
        return {
            "success": True,
            "message": f"Training '{training_id}' cleaned successfully",
            "previous_status": status
        }

@router.post("/stop/{training_id}")
async def stop_training(training_id: str):
    """Stop training task"""
    if training_id not in active_trainings:
        raise HTTPException(status_code=404, detail=f"Training '{training_id}' not found")
    
    training_info = active_trainings[training_id]
    
    if training_info["status"] != "running":
        raise HTTPException(status_code=400, detail=f"Training '{training_id}' is not running")
    
    try:
        # Stop training (actual implementation requires more complex logic)
        training_manager.stop_training(training_id)
        
        # Update training status
        training_info["status"] = "stopped"
        training_info["stopped_at"] = datetime.now().isoformat()
        
        return {
            "success": True,
            "message": f"Training '{training_id}' stopped successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to stop training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-devices")
async def get_available_devices():
    """Get available training devices"""
    try:
        # Use enhanced training manager to get available device information
        devices_info = get_training_manager().get_training_status()
        raw_devices = devices_info.get("available_devices", {})
        
        # Convert device format to array format expected by frontend
        devices_list = []
        
        # CPU device is always available
        devices_list.append({
            "id": "cpu",
            "name": "CPU",
            "description": "Central Processing Unit - Universal compatibility",
            "icon": "💻",
            "available": True,
            "recommended": False
        })
        
        # CUDA device
        cuda_available = raw_devices.get("cuda", False)
        devices_list.append({
            "id": "cuda",
            "name": "GPU (CUDA)",
            "description": f"NVIDIA GPU with CUDA support - Best for deep learning",
            "icon": "🚀",
            "available": cuda_available,
            "recommended": cuda_available  # Recommended if available
        })
        
        # MPS device (Apple Silicon)
        mps_available = raw_devices.get("mps", False)
        devices_list.append({
            "id": "mps",
            "name": "GPU (MPS)",
            "description": f"Apple Silicon GPU with Metal Performance Shaders",
            "icon": "🍎",
            "available": mps_available,
            "recommended": False
        })
        
        # Auto device (automatic selection)
        devices_list.append({
            "id": "auto",
            "name": "Auto Select",
            "description": "Automatically select the best available device",
            "icon": "⚙️",
            "available": True,
            "recommended": True
        })
        
        return {
            "status": "success",
            "devices": devices_list,
            "success": True,
            "mode": "real"
        }
    except Exception as e:
        logger.error(f"Failed to get available devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/switch-device")
async def switch_training_device(switch_request: dict):
    """Switch training device"""
    try:
        training_id = switch_request.get("job_id")
        new_device = switch_request.get("new_device")
        
        if not new_device:
            raise HTTPException(status_code=400, detail="Missing new_device")
        
        # Call enhanced training manager to switch device
        # Note: EnhancedTrainingManager's switch_training_device method does not receive training_id parameter
        # It switches the global training device settings
        result = get_training_manager().switch_training_device(new_device)
        
        if result.get("success", False):
            # If training_id is provided, also update device information in active_trainings
            if training_id and training_id in active_trainings:
                active_trainings[training_id]["device"] = new_device
            
            return {
                "status": "success",
                "success": True,
                "message": result.get("message", f"Device switched to {new_device} successfully"),
                "device": new_device,
                "mode": "real"
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to switch device"))
        
    except Exception as e:
        logger.error(f"Failed to switch training device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/devices")
async def get_system_devices():
    """Get system device information (compatibility endpoint)"""
    try:
        # Use enhanced training manager to get available device information
        devices_info = get_training_manager().get_training_status()
        raw_devices = devices_info.get("available_devices", {})
        
        # Convert device format to array format expected by frontend
        devices_list = []
        
        # CPU device is always available
        devices_list.append({
            "id": "cpu",
            "name": "CPU",
            "description": "Central Processing Unit - Universal compatibility",
            "icon": "💻",
            "available": True,
            "recommended": False
        })
        
        # CUDA device
        cuda_available = raw_devices.get("cuda", False)
        devices_list.append({
            "id": "cuda",
            "name": "GPU (CUDA)",
            "description": f"NVIDIA GPU with CUDA support - Best for deep learning",
            "icon": "🚀",
            "available": cuda_available,
            "recommended": cuda_available  # Recommended if available
        })
        
        # MPS device (Apple Silicon)
        mps_available = raw_devices.get("mps", False)
        devices_list.append({
            "id": "mps",
            "name": "GPU (MPS)",
            "description": f"Apple Silicon GPU with Metal Performance Shaders",
            "icon": "🍎",
            "available": mps_available,
            "recommended": False
        })
        
        # Auto device (automatic selection)
        devices_list.append({
            "id": "auto",
            "name": "Auto Select",
            "description": "Automatically select the best available device",
            "icon": "⚙️",
            "available": True,
            "recommended": True
        })
        
        return {
            "status": "success",
            "devices": devices_list,
            "success": True,
            "mode": "real"
        }
    except Exception as e:
        logger.error(f"Failed to get system devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/register")
async def register_model(config: ModelConfig):
    """Register new model"""
    try:
        # Generate model ID
        model_id = f"{config.model_type}_{config.model_name}_{uuid.uuid4().hex[:8]}"
        
        # Get model registry instance
        model_registry = get_model_registry_instance()
        
        # Register model
        model_registry.register_model(
            model_id=model_id,
            model_type=config.model_type,
            model_config=config.config,
            training_config=config.training_config,
            external_api_config=config.external_api_config
        )
        
        return {
            "success": True,
            "model_id": model_id,
            "message": f"Model '{model_id}' registered successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/list")
async def list_models(model_type: Optional[str] = None):
    """List all models"""
    try:
        model_registry = get_model_registry_instance()
        models = model_registry.list_models()
        
        # Filter by type
        if model_type:
            models = [m for m in models if m["model_type"] == model_type]
        
        return {
            "success": True,
            "models": models,
            "total_count": len(models),
            "model_types": list(set(m["model_type"] for m in models))
        }
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dataset/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_type: str = Form(...),
    preprocessing_config: str = Form("{}")
):
    """Upload training dataset"""
    try:
        # Parse preprocessing configuration
        try:
            preprocess_config = json.loads(preprocessing_config)
        except json.JSONDecodeError:
            preprocess_config = {}
        
        # Generate dataset ID
        dataset_id = f"{dataset_type}_{uuid.uuid4().hex[:8]}"
        
        # Save dataset file
        dataset_path = f"./data/datasets/{dataset_id}_{file.filename}"
        
        with open(dataset_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Register dataset
        dataset_config = {
            "dataset_id": dataset_id,
            "dataset_type": dataset_type,
            "dataset_path": dataset_path,
            "preprocessing_config": preprocess_config,
            "uploaded_at": datetime.now().isoformat()
        }
        
        # Save dataset configuration
        datasets_file = "./data/datasets/datasets.json"
        try:
            with open(datasets_file, "r") as f:
                datasets = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            datasets = []
        
        datasets.append(dataset_config)
        
        with open(datasets_file, "w") as f:
            json.dump(datasets, f, indent=2)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "message": f"Dataset '{dataset_id}' uploaded successfully",
            "file_size": len(content),
            "dataset_config": dataset_config
        }
        
    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset/list")
async def list_datasets(dataset_type: Optional[str] = None):
    """List all datasets"""
    try:
        datasets_file = "./data/datasets/datasets.json"
        
        try:
            with open(datasets_file, "r") as f:
                datasets = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            datasets = []
        
        # Filter by type
        if dataset_type:
            datasets = [d for d in datasets if d["dataset_type"] == dataset_type]
        
        return {
            "success": True,
            "datasets": datasets,
            "total_count": len(datasets),
            "dataset_types": list(set(d["dataset_type"] for d in datasets))
        }
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/external-api/connect")
async def connect_external_api(api_config: Dict[str, Any]):
    """Connect external API"""
    try:
        # Test connection
        connection_result = await external_api_service.test_connection(api_config)
        
        if not connection_result["success"]:
            raise HTTPException(status_code=400, detail=connection_result["error"])
        
        # Save API configuration
        api_id = external_api_service.register_api(api_config)
        
        return {
            "success": True,
            "api_id": api_id,
            "message": "External API connected successfully",
            "connection_test": connection_result
        }
        
    except Exception as e:
        logger.error(f"Failed to connect external API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/external-api/list")
async def list_external_apis():
    """List all external APIs"""
    try:
        apis = external_api_service.list_apis()
        
        return {
            "success": True,
            "apis": apis,
            "total_count": len(apis)
        }
        
    except Exception as e:
        logger.error(f"Failed to list external APIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/external-model-stats")
async def get_external_model_stats():
    """Get external model usage statistics"""
    try:
        # Get usage statistics from external API service
        external_api_service_instance = get_external_api_service()
        usage_stats = external_api_service_instance.usage_monitor.get_usage_statistics()
        
        # Convert data format to match frontend expectations
        provider_stats = usage_stats.get("provider_stats", {})
        
        # Build external model statistics list
        external_model_stats = []
        for provider, stats in provider_stats.items():
            # Map provider name to model name
            model_name = provider
            if provider == "openai":
                model_name = "GPT-4"
            elif provider == "anthropic":
                model_name = "Claude 3"
            elif provider == "google":
                model_name = "Google AI"
            elif provider == "huggingface":
                model_name = "HuggingFace"
            elif provider == "cohere":
                model_name = "Cohere"
            elif provider == "mistral":
                model_name = "Mistral"
            elif provider == "replicate":
                model_name = "Replicate"
            elif provider == "ollama":
                model_name = "Ollama"
            elif provider == "deepseek":
                model_name = "DeepSeek"
            elif provider == "siliconflow":
                model_name = "SiliconFlow"
            elif provider == "zhipu":
                model_name = "Zhipu AI"
            elif provider == "baidu":
                model_name = "Baidu ERNIE"
            elif provider == "alibaba":
                model_name = "Alibaba Qwen"
            elif provider == "moonshot":
                model_name = "Moonshot"
            elif provider == "yi":
                model_name = "Yi"
            elif provider == "tencent":
                model_name = "Tencent Hunyuan"
            
            external_model_stats.append({
                "model": model_name,
                "count": stats.get("total_calls", 0),
                "lastUsed": None  # Temporarily None, can be obtained from history records
            })
        
        # Sort by usage count
        external_model_stats.sort(key=lambda x: x["count"], reverse=True)
        
        return {
            "status": "success",
            "stats": external_model_stats,
            "total_models": len(external_model_stats),
            "total_calls": usage_stats.get("overall_stats", {}).get("total_calls", 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to get external model stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/autonomous/start")
async def start_autonomous_learning(config: Dict[str, Any], background_tasks: BackgroundTasks):
    """Start autonomous learning"""
    try:
        # Generate learning ID
        learning_id = str(uuid.uuid4())
        
        # Start autonomous learning in background
        async def run_autonomous_learning():
            try:
                result = await autonomous_learning_manager.start_autonomous_learning(config)
                
                # Update learning status
                active_learnings[learning_id]["status"] = "completed"
                active_learnings[learning_id]["result"] = result
                active_learnings[learning_id]["completed_at"] = datetime.now().isoformat()
                
                logger.info(f"Autonomous learning {learning_id} completed successfully")
                
            except Exception as e:
                active_learnings[learning_id]["status"] = "failed"
                active_learnings[learning_id]["error"] = str(e)
                active_learnings[learning_id]["failed_at"] = datetime.now().isoformat()
                logger.error(f"Autonomous learning {learning_id} failed: {e}")
        
        # Start background task
        background_tasks.add_task(run_autonomous_learning)
        
        # Record learning start
        active_learnings[learning_id] = {
            "learning_id": learning_id,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "config": config
        }
        
        return {
            "success": True,
            "learning_id": learning_id,
            "message": "Autonomous learning started successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to start autonomous learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/autonomous/status/{learning_id}")
async def get_autonomous_learning_status(learning_id: str):
    """Get autonomous learning status"""
    if learning_id not in active_learnings:
        raise HTTPException(status_code=404, detail=f"Autonomous learning '{learning_id}' not found")
    
    learning_info = active_learnings[learning_id]
    
    return {
        "success": True,
        "learning_id": learning_id,
        "status": learning_info["status"],
        "started_at": learning_info["started_at"],
        "completed_at": learning_info.get("completed_at"),
        "failed_at": learning_info.get("failed_at"),
        "error": learning_info.get("error")
    }

@router.get("/autonomous/list")
async def list_autonomous_learnings():
    """List all autonomous learning tasks"""
    return {
        "success": True,
        "learnings": list(active_learnings.values()),
        "total_count": len(active_learnings)
    }

# Helper methods
@router.post("/estimate-duration")
async def estimate_training_duration(config: Dict[str, Any]):
    """Estimate training duration"""
    try:
        duration = router._estimate_training_duration(config)
        
        return {
            "success": True,
            "estimated_duration_seconds": duration,
            "estimated_duration_human": str(timedelta(seconds=duration))
        }
        
    except Exception as e:
        logger.error(f"Failed to estimate training duration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _estimate_training_duration(self, training_config: Dict[str, Any]) -> int:
    """Estimate training duration (seconds) - Dynamic calculation based on actual system performance"""
    import psutil
    import platform
    
    # Try to import cpuinfo and GPUtil, use default values if not available
    try:
        import cpuinfo
        cpuinfo_available = True
    except ImportError:
        cpuinfo_available = False
        logger.warning("cpuinfo module not installed, using default CPU information")
    
    try:
        import GPUtil
        GPUtil_available = True
    except ImportError:
        GPUtil_available = False
        logger.warning("GPUtil module not installed, cannot get GPU information")
    
    epochs = training_config.get("epochs", 100)
    batch_size = training_config.get("batch_size", 32)
    dataset_size = training_config.get("dataset_size", 10000)
    
    # Get actual system performance metrics
    try:
        # Get CPU information
        if cpuinfo_available:
            cpu_info = cpuinfo.get_cpu_info()
            cpu_name = cpu_info.get('brand_raw', 'Unknown CPU')
        else:
            cpu_name = 'Unknown CPU'
            
        cpu_cores = psutil.cpu_count(logical=False)  # Physical core count
        cpu_logical_cores = psutil.cpu_count(logical=True)  # Logical core count
        cpu_freq = psutil.cpu_freq()
        cpu_max_freq = cpu_freq.max if cpu_freq else 3.0  # GHz
        
        # Get memory information
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        
        # Get GPU information (if available)
        gpu_available = False
        gpu_memory_gb = 0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass
        
        # Determine computational complexity based on model type
        model_type = training_config.get("model_type", "language")
        complexity_factors = {
            "language": 1.0,
            "vision": 2.0,
            "audio": 1.5,
            "multimodal": 2.5
        }
        complexity = complexity_factors.get(model_type, 1.0)
        
        # Dynamically calculate per-batch time based on system performance
        # Basic formula: time_per_batch = base_time * complexity / performance_factor
        
        # Calculate performance factor
        cpu_performance = (cpu_cores * cpu_max_freq) / 3.0  # Performance relative to 3GHz single core
        memory_factor = min(1.0, total_memory_gb / 16.0)  # Factor relative to 16GB memory
        
        if gpu_available:
            # GPU accelerated training
            gpu_factor = min(2.0, gpu_memory_gb / 8.0)  # Factor relative to 8GB GPU memory
            performance_factor = cpu_performance * memory_factor * gpu_factor
            base_time = 0.02  # Base time for GPU acceleration
        else:
            # CPU training
            performance_factor = cpu_performance * memory_factor
            base_time = 0.1  # Base time for CPU training
        
        # Consider the impact of batch size
        batch_size_factor = max(0.1, min(2.0, 32.0 / batch_size))
        
        # Calculate dynamic per-batch time
        time_per_batch = (base_time * complexity * batch_size_factor) / max(0.1, performance_factor)
        
        # Ensure time is within reasonable range
        time_per_batch = max(0.01, min(1.0, time_per_batch))
        
        logger.info(f"Dynamically calculated training time parameters: CPU cores={cpu_cores}, frequency={cpu_max_freq:.2f}GHz, "
                   f"memory={total_memory_gb:.1f}GB, GPU available={gpu_available}, "
                   f"performance factor={performance_factor:.2f}, per-batch time={time_per_batch:.4f} seconds")
        
    except Exception as e:
        logger.warning(f"Cannot get system performance metrics, using conservative estimate: {str(e)}")
        # Conservative estimate
        time_per_batch = 0.15  # Default 0.15 seconds per batch
    
    # Calculate total time
    batches_per_epoch = dataset_size / batch_size
    total_time = epochs * batches_per_epoch * time_per_batch
    
    # Add buffer time (20%)
    total_time_with_buffer = total_time * 1.2
    
    return int(total_time_with_buffer)

@router.post("/knowledge-learning/start")
async def start_knowledge_learning(request: KnowledgeLearningRequest, background_tasks: BackgroundTasks):
    """Start knowledge base content learning - integrated with real knowledge manager"""
    try:
        # Generate learning ID
        learning_id = str(uuid.uuid4())
        
        # Validate learning model
        supported_models = ["transformer", "graph", "memory", "hybrid"]
        if request.model not in supported_models:
            raise HTTPException(status_code=400, detail=f"Model '{request.model}' not supported. Supported models: {', '.join(supported_models)}")
        
        # Validate learning intensity
        if not 0.1 <= request.intensity <= 1.0:
            raise HTTPException(status_code=400, detail="Learning intensity must be between 0.1 and 1.0")
        
        # Validate maximum learning time
        if not 1 <= request.max_time <= 240:
            raise HTTPException(status_code=400, detail="Max learning time must be between 1 and 240 minutes")
        
        # Validate priority
        valid_priorities = ["balanced", "exploration", "exploitation"]
        if request.priority not in valid_priorities:
            raise HTTPException(status_code=400, detail=f"Priority must be one of: {', '.join(valid_priorities)}")
        
        # Use real knowledge manager to start autonomous learning
        learning_result = knowledge_manager.start_autonomous_learning(
            model_id=request.model,
            domains=request.domains,
            priority=request.priority
        )
        
        if not learning_result.get('success'):
            raise HTTPException(status_code=500, detail=learning_result.get('message', 'Failed to start knowledge learning'))
        
        # Run knowledge learning progress tracking in background
        async def run_knowledge_learning_progress():
            try:
                start_time = datetime.now()
                max_duration_seconds = request.max_time * 60  # Convert to seconds
                
                while True:
                    # Check if timeout
                    elapsed_time = (datetime.now() - start_time).seconds
                    if elapsed_time > max_duration_seconds:
                        # Timeout stop learning
                        knowledge_manager.stop_autonomous_learning()
                        active_learnings[learning_id]["status"] = "completed"
                        active_learnings[learning_id]["completed_at"] = datetime.now().isoformat()
                        active_learnings[learning_id]["active"] = False
                        active_learnings[learning_id]["logs"].append({
                            "timestamp": datetime.now().isoformat(),
                            "message": f"Knowledge learning completed (timeout after {request.max_time} minutes)",
                            "level": "info"
                        })
                        break
                    
                    # Get knowledge manager status
                    knowledge_status = knowledge_manager.get_autonomous_learning_status()
                    
                    # Update progress
                    if knowledge_status.get('active'):
                        # Time-based progress estimate
                        time_progress = min(100, (elapsed_time * 100) / max_duration_seconds)
                        
                        # Update progress with actual knowledge statistics
                        knowledge_stats = knowledge_manager.get_knowledge_stats()
                        stats_progress = min(100, knowledge_stats.get('total_facts', 0) / 1000 * 100)
                        
                        # Combined progress
                        progress = (time_progress + stats_progress) / 2
                        
                        active_learnings[learning_id]["progress"] = progress
                        
                        # Update status
                        if progress < 30:
                            status = "Extracting knowledge..."
                        elif progress < 60:
                            status = "Processing information..."
                        elif progress < 90:
                            status = "Integrating knowledge..."
                        else:
                            status = "Finalizing..."
                        
                        active_learnings[learning_id]["status"] = status
                        
                        # Get recent updates
                        recent_updates = knowledge_manager.get_recent_updates()
                        if recent_updates and len(recent_updates) > 0:
                            for update in recent_updates[:3]:  # Only show the latest 3 updates
                                active_learnings[learning_id]["logs"].append({
                                    "timestamp": update.get('timestamp', datetime.now().isoformat()),
                                    "message": f"{update.get('domain', 'Unknown')}: {update.get('action', 'update')} - {update.get('items_added', 0)} items",
                                    "level": "info"
                                })
                        
                        # Update knowledge graph
                        knowledge_manager.build_knowledge_graph()
                        
                        # Update knowledge freshness
                        knowledge_manager.update_knowledge_freshness()
                        
                        # Knowledge source fusion
                        knowledge_manager.fuse_knowledge_sources()
                    else:
                        # Learning has stopped
                        active_learnings[learning_id]["status"] = "completed"
                        active_learnings[learning_id]["completed_at"] = datetime.now().isoformat()
                        active_learnings[learning_id]["active"] = False
                        active_learnings[learning_id]["logs"].append({
                            "timestamp": datetime.now().isoformat(),
                            "message": f"Knowledge learning completed for domains: {', '.join(request.domains)}",
                            "level": "success"
                        })
                        break
                    
                    await asyncio.sleep(5)  # Update progress every 5 seconds
                
                logger.info(f"Knowledge learning {learning_id} completed successfully")
                
            except Exception as e:
                active_learnings[learning_id]["status"] = "failed"
                active_learnings[learning_id]["error"] = str(e)
                active_learnings[learning_id]["failed_at"] = datetime.now().isoformat()
                active_learnings[learning_id]["active"] = False
                logger.error(f"Knowledge learning {learning_id} failed: {e}")
        
        # Start background task
        background_tasks.add_task(run_knowledge_learning_progress)
        
        # Record learning start
        active_learnings[learning_id] = {
            "learning_id": learning_id,
            "model": request.model,
            "domains": request.domains,
            "priority": request.priority,
            "intensity": request.intensity,
            "max_time_minutes": request.max_time,
            "status": "running",
            "progress": 0.0,
            "started_at": datetime.now().isoformat(),
            "active": True,
            "logs": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": f"Starting knowledge learning with model: {request.model}",
                    "level": "info"
                },
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": f"Domains: {', '.join(request.domains)}",
                    "level": "info"
                },
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": f"Priority: {request.priority}, Intensity: {request.intensity}",
                    "level": "info"
                },
                {
                    "timestamp": datetime.now().isoformat(),
                    "message": f"Connected to real Knowledge Manager: {learning_result.get('message', 'Learning started')}",
                    "level": "success"
                }
            ]
        }
        
        return {
            "status": "success",
            "learning_id": learning_id,
            "message": "Knowledge learning started successfully with real Knowledge Manager",
            "model": request.model,
            "domains": request.domains,
            "priority": request.priority,
            "estimated_duration_minutes": request.max_time,
            "knowledge_manager_result": learning_result
        }
        
    except Exception as e:
        logger.error(f"Failed to start knowledge learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-learning/status")
async def get_knowledge_learning_status():
    """Get knowledge learning status
    
    Returns:
        Current knowledge learning status
    """
    try:
        # Find active learning tasks
        active_learning = None
        learning_id = None
        
        for lid, learning in active_learnings.items():
            if learning.get("active") and learning.get("status") == "running":
                active_learning = learning
                learning_id = lid
                break
        
        if not active_learning:
            # No active learning session found
            return {
                "status": "success",
                "active": False,
                "message": "No active knowledge learning session",
                "learning_id": None,
                "progress": 0,
                "duration_minutes": 0,
                "knowledge_stats": {"total_domains": 0, "total_facts": 0}
            }
        
        # Calculate duration
        started_at = safe_fromisoformat(active_learning["started_at"])
        duration_minutes = (datetime.now() - started_at).seconds / 60
        
        # Get knowledge stats
        knowledge_stats = knowledge_manager.get_knowledge_stats()
        
        return {
            "status": "success",
            "active": True,
            "learning_id": learning_id,
            "progress": active_learning.get("progress", 0),
            "duration_minutes": round(duration_minutes, 2),
            "model": active_learning.get("model"),
            "domains": active_learning.get("domains", []),
            "priority": active_learning.get("priority"),
            "current_status": active_learning.get("status"),
            "knowledge_stats": knowledge_stats,
            "logs": active_learning.get("logs", [])[-10:]  # Last 10 logs
        }
        
    except Exception as e:
        logger.error(f"Failed to get knowledge learning status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge-learning/stop")
async def stop_knowledge_learning():
    """Stop knowledge base content learning - integrated with real knowledge manager"""
    try:
        # Find active knowledge learning tasks
        active_learning = None
        learning_id = None
        
        for lid, learning in active_learnings.items():
            if learning.get("active") and learning.get("status") == "running":
                active_learning = learning
                learning_id = lid
                break
        
        if not active_learning:
            # No active learning session found, return success anyway
            return {
                "status": "success",
                "message": "No active knowledge learning session found",
                "learning_id": None,
                "progress": 0,
                "duration_minutes": 0,
                "knowledge_stats": {"total_domains": 0, "total_facts": 0},
                "knowledge_manager_result": {"success": True, "message": "No active session to stop"}
            }
        
        # Use real knowledge manager to stop autonomous learning
        stop_result = knowledge_manager.stop_autonomous_learning()
        
        if not stop_result.get('success'):
            logger.warning(f"Failed to stop knowledge manager: {stop_result.get('message')}")
        
        # Stop learning task
        active_learning["active"] = False
        active_learning["status"] = "stopped"
        active_learning["stopped_at"] = datetime.now().isoformat()
        
        # Get final knowledge statistics
        final_stats = knowledge_manager.get_knowledge_stats()
        
        # Add stop log
        active_learning["logs"].append({
            "timestamp": datetime.now().isoformat(),
            "message": "Knowledge learning stopped by user request",
            "level": "warning"
        })
        
        active_learning["logs"].append({
            "timestamp": datetime.now().isoformat(),
            "message": f"Final knowledge statistics: {final_stats.get('total_domains', 0)} domains, {final_stats.get('total_facts', 0)} facts",
            "level": "info"
        })
        
        logger.info(f"Knowledge learning {learning_id} stopped successfully")
        
        return {
            "status": "success",
            "message": "Knowledge learning stopped successfully with real Knowledge Manager",
            "learning_id": learning_id,
            "progress": active_learning["progress"],
            "duration_minutes": (safe_fromisoformat(active_learning["stopped_at"]) - 
                                safe_fromisoformat(active_learning["started_at"])).seconds / 60,
            "knowledge_stats": final_stats,
            "knowledge_manager_result": stop_result
        }
        
    except Exception as e:
        logger.error(f"Failed to stop knowledge learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge-transfer/start")
async def start_knowledge_transfer(request: KnowledgeTransferRequest):
    """Start knowledge transfer"""
    try:
        logger.info(f"Starting knowledge transfer: {request.source_domain} -> {request.target_domain}")
        
        # Call knowledge manager's knowledge transfer function
        transfer_result = knowledge_manager.transfer_knowledge(
            source_domain=request.source_domain,
            target_domain=request.target_domain,
            transfer_strategy=request.transfer_strategy
        )
        
        if not transfer_result.get('success'):
            raise HTTPException(status_code=400, detail=transfer_result.get('message'))
        
        # Generate transfer ID
        transfer_id = str(uuid.uuid4())
        
        # Record transfer history
        transfer_record = {
            'transfer_id': transfer_id,
            'source_domain': request.source_domain,
            'target_domain': request.target_domain,
            'strategy': request.transfer_strategy,
            'intensity': request.transfer_intensity,
            'started_at': datetime.now().isoformat(),
            'result': transfer_result,
            'status': 'completed'
        }
        
        # Get knowledge statistics
        knowledge_stats = knowledge_manager.get_knowledge_stats()
        
        return {
            'status': 'success',
            'transfer_id': transfer_id,
            'message': transfer_result.get('message'),
            'transfer_result': transfer_result,
            'knowledge_stats': knowledge_stats,
            'transfer_efficiency': transfer_result.get('transfer_efficiency', 0.0),
            'items_transferred': transfer_result.get('items_transferred', 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to start knowledge transfer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-transfer/history")
async def get_knowledge_transfer_history():
    """Get knowledge transfer history records"""
    try:
        # Call knowledge manager to get transfer history
        transfer_history = knowledge_manager.get_transfer_history()
        
        return {
            'status': 'success',
            'transfer_history': transfer_history,
            'total_transfers': len(transfer_history)
        }
    except Exception as e:
        logger.error(f"Failed to get knowledge transfer history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-transfer/efficiency/{source_domain}/{target_domain}")
async def get_transfer_efficiency(source_domain: str, target_domain: str):
    """Get knowledge transfer efficiency analysis between domains"""
    try:
        # Call knowledge manager to calculate transfer efficiency
        efficiency_analysis = knowledge_manager.calculate_transfer_efficiency(
            source_domain=source_domain,
            target_domain=target_domain
        )
        
        if not efficiency_analysis.get('success'):
            raise HTTPException(status_code=400, detail=efficiency_analysis.get('message'))
        
        return {
            'status': 'success',
            'efficiency_analysis': efficiency_analysis,
            'source_domain': source_domain,
            'target_domain': target_domain
        }
    except Exception as e:
        logger.error(f"Failed to get transfer efficiency: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-transfer/available-domains")
async def get_available_knowledge_domains():
    """Get available knowledge domains"""
    try:
        # Call knowledge manager to get knowledge statistics
        knowledge_stats = knowledge_manager.get_knowledge_stats()
        
        # Get all knowledge domains
        knowledge_domains = knowledge_manager.knowledge_bases.keys()
        
        return {
            'status': 'success',
            'available_domains': list(knowledge_domains),
            'total_domains': knowledge_stats.get('total_domains', 0),
            'total_facts': knowledge_stats.get('total_facts', 0)
        }
    except Exception as e:
        logger.error(f"Failed to get available knowledge domains: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Bind helper method to class
router._estimate_training_duration = _estimate_training_duration.__get__(router, type(router))

@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify API routes are working"""
    return {"message": "Test endpoint works", "success": True}
