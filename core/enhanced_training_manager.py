"""
Enhanced Training Manager
Supports local model training when external models are enabled and CPU/GPU training switching
"""

import logging
import time
import threading
import torch
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from .error_handling import error_handler
from .external_model_proxy import ExternalModelProxy
from .training_manager import TrainingManager

class EnhancedTrainingManager:
    """Enhanced Training Manager Class
    
    Function: Supports local model training when external models are enabled and CPU/GPU training switching
    """
    
    def __init__(self, training_manager: TrainingManager = None):
        """Initialize enhanced training manager"""
        self.logger = logging.getLogger(__name__)
        
        # Use existing training manager or create new
        if training_manager:
            self.training_manager = training_manager
        else:
            self.training_manager = TrainingManager()
        
        # External model proxy
        self.external_model_proxy = ExternalModelProxy()
        
        # Training configuration
        self.training_config = {
            "device_type": "auto",  # auto, cpu, cuda, mps
            "mixed_precision": True,
            "distributed_training": False,
            "memory_optimization": True,
            "checkpoint_frequency": 1000
        }
        
        # External model to local model mapping
        self.external_to_local_mapping = {
            "gpt-4": "language_model",
            "claude-3": "language_model", 
            "dall-e-3": "vision_model",
            "stable-diffusion": "vision_model",
            "whisper": "audio_model",
            "midjourney": "vision_model"
        }
        
        # Training status tracking
        self.training_status = {
            "current_device": None,
            "active_training_jobs": {},
            "external_model_usage": {},
            "training_metrics": {}
        }
        
        # Device detection
        self._detect_available_devices()
        
        self.logger.info("Enhanced training manager initialized")
    
    def _detect_available_devices(self):
        """Detect available devices"""
        self.available_devices = {
            "cpu": True,
            "cuda": torch.cuda.is_available(),
            "mps": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }
        
        if self.available_devices["cuda"]:
            self.available_devices["cuda_device_count"] = torch.cuda.device_count()
            self.logger.info(f"Detected {torch.cuda.device_count()} CUDA devices")
        
        if self.available_devices["mps"]:
            self.logger.info("Detected MPS device")
    
    def set_training_device(self, device_type: str) -> Dict[str, Any]:
        """Set training device
        
        Args:
            device_type: Device type (auto, cpu, cuda, mps)
            
        Returns:
            Setting result
        """
        try:
            if device_type == "auto":
                # Automatically select the best device
                if self.available_devices["cuda"]:
                    device = "cuda"
                elif self.available_devices["mps"]:
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = device_type
            
            # Validate device availability
            if device == "cuda" and not self.available_devices["cuda"]:
                return {"success": False, "message": "CUDA device not available"}
            elif device == "mps" and not self.available_devices["mps"]:
                return {"success": False, "message": "MPS device not available"}
            
            self.training_config["device_type"] = device
            self.training_status["current_device"] = device
            
            self.logger.info(f"Training device set to: {device}")
            
            return {
                "success": True, 
                "device": device,
                "message": f"Training device set to {device}"
            }
            
        except Exception as e:
            error_handler.handle_error(e, "EnhancedTrainingManager", f"Failed to set training device: {str(e)}")
            return {"success": False, "message": f"Failed to set training device: {str(e)}"}
    
    async def train_with_external_model_assistance(self, model_name: str, training_data: Dict[str, Any], 
                                                  external_model_id: str = None) -> Dict[str, Any]:
        """Train local model with external model assistance
        
        Args:
            model_name: Local model name
            training_data: Training data
            external_model_id: External model ID
            
        Returns:
            Training result
        """
        try:
            # If no external model is specified, automatically select based on model type
            if not external_model_id:
                external_model_id = self._get_recommended_external_model(model_name)
            
            # Validate external model availability
            if not await self._validate_external_model(external_model_id):
                return {"success": False, "message": f"External model {external_model_id} not available"}
            
            # Start training process
            training_job_id = f"{model_name}_{external_model_id}_{int(time.time())}"
            self.training_status["active_training_jobs"][training_job_id] = {
                "model_name": model_name,
                "external_model_id": external_model_id,
                "status": "initializing",
                "start_time": datetime.now().isoformat(),
                "metrics": {}
            }
            
            # Generate training guidance using external model
            guidance_result = await self._generate_training_guidance(external_model_id, model_name, training_data)
            
            if not guidance_result.get("success", False):
                return {"success": False, "message": "Unable to generate training guidance"}
            
            # Update training status
            self.training_status["active_training_jobs"][training_job_id]["status"] = "generating_guidance"
            
            # Perform knowledge distillation using external model
            distillation_result = await self._perform_knowledge_distillation(
                external_model_id, model_name, training_data, guidance_result["guidance"]
            )
            
            if not distillation_result.get("success", False):
                return {"success": False, "message": "Knowledge distillation failed"}
            
            # Update training status
            self.training_status["active_training_jobs"][training_job_id]["status"] = "distilling_knowledge"
            
            # Start local model training
            local_training_result = await self._train_local_model(
                model_name, training_data, distillation_result["distilled_knowledge"]
            )
            
            # Complete training
            self.training_status["active_training_jobs"][training_job_id]["status"] = "completed"
            self.training_status["active_training_jobs"][training_job_id]["end_time"] = datetime.now().isoformat()
            self.training_status["active_training_jobs"][training_job_id]["metrics"] = local_training_result.get("metrics", {})
            
            # Record external model usage
            if external_model_id not in self.training_status["external_model_usage"]:
                self.training_status["external_model_usage"][external_model_id] = 0
            self.training_status["external_model_usage"][external_model_id] += 1
            
            return {
                "success": True,
                "training_job_id": training_job_id,
                "local_training_result": local_training_result,
                "external_model_used": external_model_id,
                "message": "Training with external model assistance completed"
            }
            
        except Exception as e:
            error_handler.handle_error(e, "EnhancedTrainingManager", f"External model assisted training failed: {str(e)}")
            return {"success": False, "message": f"External model assisted training failed: {str(e)}"}
    
    def _get_recommended_external_model(self, model_name: str) -> str:
        """Get recommended external model
        
        Args:
            model_name: Local model name
            
        Returns:
            Recommended external model ID
        """
        model_mapping = {
            "language_model": "gpt-4",
            "vision_model": "dall-e-3", 
            "audio_model": "whisper",
            "knowledge_model": "gpt-4",
            "manager_model": "gpt-4"
        }
        
        return model_mapping.get(model_name, "gpt-4")
    
    async def _validate_external_model(self, external_model_id: str) -> bool:
        """Validate external model availability
        
        Args:
            external_model_id: External model ID
            
        Returns:
            Whether available
        """
        try:
            # Simple connection test
            test_result = await self.external_model_proxy.inference(
                external_model_id, 
                {"test": "connection_test"}
            )
            return test_result.get("success", False)
        except Exception as e:
            self.logger.error(f"External model connection test failed: {e}")
            return False
    
    async def _generate_training_guidance(self, external_model_id: str, model_name: str, 
                                         training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate training guidance using external model
        
        Args:
            external_model_id: External model ID
            model_name: Local model name
            training_data: Training data
            
        Returns:
            Training guidance result
        """
        try:
            guidance_prompt = f"""
            Please provide training guidance for the {model_name} model.
            Training data types: {list(training_data.keys())}
            Please provide the following guidance:
            1. Optimal learning rate settings
            2. Batch size recommendations
            3. Training epoch recommendations
            4. Optimizer selection
            5. Regularization strategies
            6. Data augmentation suggestions
            """
            
            guidance_result = await self.external_model_proxy.inference(
                external_model_id,
                guidance_prompt
            )
            
            if guidance_result.get("success", False):
                return {
                    "success": True,
                    "guidance": guidance_result.get("result", {}),
                    "message": "Training guidance generated successfully"
                }
            else:
                return {"success": False, "message": "Failed to generate training guidance"}
                
        except Exception as e:
            error_handler.handle_error(e, "EnhancedTrainingManager", f"Failed to generate training guidance: {str(e)}")
            return {"success": False, "message": f"Failed to generate training guidance: {str(e)}"}
    
    async def _perform_knowledge_distillation(self, external_model_id: str, model_name: str,
                                             training_data: Dict[str, Any], guidance: Dict[str, Any]) -> Dict[str, Any]:
        """Perform knowledge distillation
        
        Args:
            external_model_id: External model ID
            model_name: Local model name
            training_data: Training data
            guidance: Training guidance
            
        Returns:
            Knowledge distillation result
        """
        try:
            # Use external model to process training data and generate distilled knowledge
            distillation_prompt = f"""
            Please perform knowledge distillation for the {model_name} model.
            Based on the following training data, generate knowledge representations suitable for local model learning:
            {json.dumps(training_data, ensure_ascii=False, indent=2)[:1000]}...
            
            Training guidance:
            {json.dumps(guidance, ensure_ascii=False, indent=2)}
            """
            
            distillation_result = await self.external_model_proxy.inference(
                external_model_id,
                distillation_prompt
            )
            
            if distillation_result.get("success", False):
                return {
                    "success": True,
                    "distilled_knowledge": distillation_result.get("result", {}),
                    "message": "Knowledge distillation completed"
                }
            else:
                return {"success": False, "message": "Knowledge distillation failed"}
                
        except Exception as e:
            error_handler.handle_error(e, "EnhancedTrainingManager", f"Knowledge distillation failed: {str(e)}")
            return {"success": False, "message": f"Knowledge distillation failed: {str(e)}"}
    
    async def _train_local_model(self, model_name: str, training_data: Dict[str, Any],
                                distilled_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Train local model
        
        Args:
            model_name: Model name
            training_data: Training data
            distilled_knowledge: Distilled knowledge
            
        Returns:
            Training result
        """
        try:
            # Get current device
            device = self.training_status["current_device"] or "cpu"
            
            # Build training configuration - adapt to new TrainingManager interface
            training_config = {
                "device": device,
                "mixed_precision": self.training_config["mixed_precision"],
                "distilled_knowledge": distilled_knowledge,
                **self.training_config
            }
            
            # Convert to format expected by new TrainingManager
            data_config = {
                "dataset_path": "generated_distilled_data",
                "data_type": "distilled",
                "distilled_knowledge": distilled_knowledge,
                "original_data": training_data
            }
            
            training_params = {
                "epochs": self.training_config.get("checkpoint_frequency", 1000) // 100,
                "batch_size": self.training_config.get("batch_size", 32),
                "learning_rate": 0.001,
                "device": device,
                "mixed_precision": self.training_config.get("mixed_precision", True),
                "distilled_training": True
            }
            
            # Call refactored TrainingManager for training
            training_result = self.training_manager.start_training(
                model_id=model_name,
                data_config=data_config,
                training_params=training_params
            )
            
            if not training_result.get("success", False):
                return {
                    "success": False,
                    "message": f"Training startup failed: {training_result.get('message', 'Unknown error')}"
                }
            
            job_id = training_result.get("job_id")
            
            # Wait for training completion and get real metrics
            import asyncio
            max_wait_time = 3600  # Maximum wait time: 1 hour
            check_interval = 5    # Check interval: 5 seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                # Get training status
                status_result = self.training_manager.get_training_status(job_id)
                
                if not status_result.get("success", False):
                    return {
                        "success": False,
                        "message": f"Failed to get training status: {status_result.get('message', 'Unknown error')}"
                    }
                
                job_status = status_result.get("status", {})
                current_status = job_status.get("status", "unknown")
                
                # Check if training is completed
                if current_status in ["completed", "failed", "stopped"]:
                    # Get monitoring metrics
                    monitor_metrics = status_result.get("metrics", {})
                    latest_metrics = monitor_metrics.get("latest_metrics", {})
                    
                    if current_status == "completed":
                        # Return real training metrics
                        real_metrics = {
                            "loss": latest_metrics.get("loss", 0.0),
                            "accuracy": latest_metrics.get("accuracy", 0.0),
                            "epochs_completed": job_status.get("current_epoch", 0),
                            "training_time_seconds": time.time() - start_time,
                            "device_used": device,
                            "progress": job_status.get("progress", 0.0)
                        }
                        
                        return {
                            "success": True,
                            "job_id": job_id,
                            "metrics": real_metrics,
                            "training_time": real_metrics["training_time_seconds"],
                            "message": "Local model training completed"
                        }
                    else:
                        return {
                            "success": False,
                            "job_id": job_id,
                            "message": f"Training failed with status: {current_status}"
                        }
                
                # Wait for next check
                await asyncio.sleep(check_interval)
            
            # Timeout handling
            return {
                "success": False,
                "job_id": job_id,
                "message": "Training timeout, not completed within the specified time"
            }
            
        except Exception as e:
            error_handler.handle_error(e, "EnhancedTrainingManager", f"Local model training failed: {str(e)}")
            return {"success": False, "message": f"Local model training failed: {str(e)}"}
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get training status
        
        Returns:
            Training status information
        """
        return {
            "current_device": self.training_status["current_device"],
            "available_devices": self.available_devices,
            "active_jobs_count": len(self.training_status["active_training_jobs"]),
            "external_model_usage": self.training_status["external_model_usage"],
            "training_config": self.training_config
        }
    
    def switch_training_device(self, new_device: str) -> Dict[str, Any]:
        """Switch training device
        
        Args:
            new_device: New device type
            
        Returns:
            Switch result
        """
        try:
            # Check if there are active training jobs
            active_jobs = [job for job in self.training_status["active_training_jobs"].values() 
                          if job["status"] in ["training", "initializing", "generating_guidance", "distilling_knowledge"]]
            
            if active_jobs:
                return {
                    "success": False, 
                    "message": "Cannot switch device while active training jobs exist"
                }
            
            # Set new device
            result = self.set_training_device(new_device)
            
            if result["success"]:
                self.logger.info(f"Training device switched to: {new_device}")
            
            return result
            
        except Exception as e:
            error_handler.handle_error(e, "EnhancedTrainingManager", f"Failed to switch training device: {str(e)}")
            return {"success": False, "message": f"Failed to switch training device: {str(e)}"}
    
    def optimize_training_for_device(self, device_type: str) -> Dict[str, Any]:
        """Optimize training configuration for specific device
        
        Args:
            device_type: Device type
            
        Returns:
            Optimization result
        """
        try:
            optimization_configs = {
                "cpu": {
                    "batch_size": 32,
                    "mixed_precision": False,
                    "gradient_accumulation": 1,
                    "num_workers": 4
                },
                "cuda": {
                    "batch_size": 64,
                    "mixed_precision": True,
                    "gradient_accumulation": 4,
                    "num_workers": 8
                },
                "mps": {
                    "batch_size": 32,
                    "mixed_precision": True,
                    "gradient_accumulation": 2,
                    "num_workers": 4
                }
            }
            
            if device_type not in optimization_configs:
                return {"success": False, "message": f"Unsupported device type: {device_type}"}
            
            # Update training configuration
            self.training_config.update(optimization_configs[device_type])
            
            self.logger.info(f"Training configuration optimized for {device_type} device")
            
            return {
                "success": True,
                "optimized_config": self.training_config,
                "message": f"Training configuration optimized for {device_type} device"
            }
            
        except Exception as e:
            error_handler.handle_error(e, "EnhancedTrainingManager", f"Failed to optimize training configuration: {str(e)}")
            return {"success": False, "message": f"Failed to optimize training configuration: {str(e)}"}
    
    def stop_training(self, training_job_id: str) -> Dict[str, Any]:
        """Stop training job
        
        Args:
            training_job_id: Training job ID
            
        Returns:
            Stop result
        """
        try:
            # Check if training job exists
            if training_job_id not in self.training_status["active_training_jobs"]:
                return {"success": False, "message": f"Training job {training_job_id} does not exist"}
            
            # Update training job status
            self.training_status["active_training_jobs"][training_job_id]["status"] = "stopped"
            self.training_status["active_training_jobs"][training_job_id]["end_time"] = datetime.now().isoformat()
            
            # Call training manager's stop method
            stop_result = self.training_manager.stop_model_training(training_job_id)
            
            self.logger.info(f"Training job {training_job_id} stopped")
            
            return {
                "success": True,
                "training_job_id": training_job_id,
                "message": f"Training job {training_job_id} stopped",
                "stop_result": stop_result
            }
            
        except Exception as e:
            error_handler.handle_error(e, "EnhancedTrainingManager", f"Failed to stop training job: {str(e)}")
            return {"success": False, "message": f"Failed to stop training job: {str(e)}"}
    
    def start_training(self, model_id: Union[str, List[str]], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start training job
        
        Args:
            model_id: Model ID or list of model IDs
            parameters: Training parameters
            
        Returns:
            Training startup result
        """
        try:
            # Delegate to internal training manager's start_training method
            # Note: internal TrainingManager expects data_config and training_params parameters
            # We pass parameters as training_params and set data_config to None
            result = self.training_manager.start_training(
                model_id=model_id,
                data_config=None,
                training_params=parameters
            )
            return result
        except Exception as e:
            error_handler.handle_error(e, "EnhancedTrainingManager", f"Failed to start training job: {str(e)}")
            return {"success": False, "message": f"Failed to start training job: {str(e)}", "job_id": None}
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get training job status
        
        Args:
            job_id: Training job ID
            
        Returns:
            Training job status information
        """
        try:
            # Delegate to internal training manager's get_job_status method
            return self.training_manager.get_job_status(job_id)
        except Exception as e:
            error_handler.handle_error(e, "EnhancedTrainingManager", f"Failed to get training job status: {str(e)}")
            return {"success": False, "message": f"Failed to get training job status: {str(e)}"}
    
    def get_active_training_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get all active training jobs
        
        Returns:
            Active training jobs dictionary
        """
        try:
            return {
                "success": True,
                "active_jobs": self.training_status["active_training_jobs"]
            }
        except Exception as e:
            error_handler.handle_error(e, "EnhancedTrainingManager", f"Failed to get active training jobs: {str(e)}")
            return {"success": False, "message": f"Failed to get active training jobs: {str(e)}"}
    
    def get_joint_training_recommendations(self) -> Dict[str, Any]:
        """Get joint training recommended combinations
        
        Returns:
            Dictionary with recommended combinations
        """
        try:
            # Get available models from model registry
            available_models = list(self.training_manager.model_registry.model_types.keys())
            
            # Map model IDs to letter IDs based on fixed mapping from frontend
            # This mapping must match modelIdMapper.js in the frontend
            model_id_to_letter_map = {
                'manager': 'A',
                'language': 'B',
                'knowledge': 'C',
                'vision': 'D',
                'audio': 'E',
                'autonomous': 'F',
                'programming': 'G',
                'planning': 'H',
                'emotion': 'I',
                'spatial': 'J',
                'computer_vision': 'K',
                'sensor': 'L',
                'motion': 'M',
                'prediction': 'N',
                'advanced_reasoning': 'O',
                'data_fusion': 'P',
                'creative_problem_solving': 'Q',
                'meta_cognition': 'R',
                'value_alignment': 'S',
                'vision_image': 'T',
                'vision_video': 'U',
                'finance': 'V',
                'medical': 'W',
                'collaboration': 'X',
                'optimization': 'Y',
                'computer': 'Z',
                'mathematics': 'AA'
            }
            
            # Create model_to_letter mapping only for available models
            model_to_letter = {}
            for model_id in available_models:
                if model_id in model_id_to_letter_map:
                    model_to_letter[model_id] = model_id_to_letter_map[model_id]
                else:
                    # Fallback: use model_id itself if not in mapping
                    model_to_letter[model_id] = model_id
            
            # Debug logging
            error_handler.log_info(f"[EnhancedTrainingManager] Available models: {available_models}", "Debug")
            error_handler.log_info(f"[EnhancedTrainingManager] Model to letter mapping: {model_to_letter}", "Debug")
            
            # Define model categories based on model ID patterns and names
            model_categories = {
                "core_models": ["manager", "language", "vision", "knowledge", "emotion"],
                "specialized_models": ["computer_vision", "optimization", "mathematics", "autonomous", "medical", "finance", "collaboration"],
                "sensor_models": ["sensor_processor", "audio_processor", "video_processor"],
                "reasoning_models": ["logical_reasoning", "causal_inference", "planning"]
            }
            
            # Filter available models in each category
            actual_models = {}
            for category, model_ids in model_categories.items():
                actual_models[category] = [model_id for model_id in model_ids if model_id in available_models]
            
            # Create recommendations based on actual available models
            # For combinations, prioritize core models first
            recommendations = {}
            
            # Basic interaction: manager + language + knowledge (if available)
            basic_models = []
            if "manager" in available_models:
                basic_models.append(model_to_letter["manager"])
            if "language" in available_models:
                basic_models.append(model_to_letter["language"])
            if "knowledge" in available_models:
                basic_models.append(model_to_letter["knowledge"])
            if basic_models:
                recommendations["basic_interaction"] = basic_models
            
            # Visual processing: vision/computer_vision + manager + language
            visual_models = []
            if "vision" in available_models:
                visual_models.append(model_to_letter["vision"])
            elif "computer_vision" in available_models:
                visual_models.append(model_to_letter["computer_vision"])
            if "manager" in available_models:
                visual_models.append(model_to_letter["manager"])
            if "language" in available_models:
                visual_models.append(model_to_letter["language"])
            if visual_models:
                recommendations["visual_processing"] = visual_models
            
            # Sensor analysis: sensor models + manager
            sensor_models = []
            sensor_model_ids = ["sensor_processor", "audio_processor", "video_processor"]
            for sensor_id in sensor_model_ids:
                if sensor_id in available_models:
                    sensor_models.append(model_to_letter[sensor_id])
            if "manager" in available_models and sensor_models:
                sensor_models.append(model_to_letter["manager"])
            if sensor_models:
                recommendations["sensor_analysis"] = sensor_models
            
            # Knowledge intensive: knowledge + language + manager
            knowledge_models = []
            if "knowledge" in available_models:
                knowledge_models.append(model_to_letter["knowledge"])
            if "language" in available_models:
                knowledge_models.append(model_to_letter["language"])
            if "manager" in available_models:
                knowledge_models.append(model_to_letter["manager"])
            if knowledge_models:
                recommendations["knowledge_intensive"] = knowledge_models
            
            # Emotional intelligence: emotion + manager + language
            emotional_models = []
            if "emotion" in available_models:
                emotional_models.append(model_to_letter["emotion"])
            if "manager" in available_models:
                emotional_models.append(model_to_letter["manager"])
            if "language" in available_models:
                emotional_models.append(model_to_letter["language"])
            if emotional_models:
                recommendations["emotional_intelligence"] = emotional_models
            
            # Complete system: all core models
            complete_models = []
            for model_id in ["manager", "language", "vision", "knowledge", "emotion"]:
                if model_id in available_models:
                    complete_models.append(model_to_letter[model_id])
            if "computer_vision" in available_models and "vision" not in available_models:
                complete_models.append(model_to_letter["computer_vision"])
            if complete_models:
                recommendations["complete_system"] = complete_models
            
            # Autonomous control: autonomous + manager + planning (if available)
            autonomous_models = []
            if "autonomous" in available_models:
                autonomous_models.append(model_to_letter["autonomous"])
            if "manager" in available_models:
                autonomous_models.append(model_to_letter["manager"])
            if "planning" in available_models:
                autonomous_models.append(model_to_letter["planning"])
            elif "logical_reasoning" in available_models:
                autonomous_models.append(model_to_letter["logical_reasoning"])
            if autonomous_models:
                recommendations["autonomous_control"] = autonomous_models
            
            # Cognitive processing: language + manager + logical_reasoning + knowledge
            cognitive_models = []
            if "language" in available_models:
                cognitive_models.append(model_to_letter["language"])
            if "manager" in available_models:
                cognitive_models.append(model_to_letter["manager"])
            if "logical_reasoning" in available_models:
                cognitive_models.append(model_to_letter["logical_reasoning"])
            if "knowledge" in available_models:
                cognitive_models.append(model_to_letter["knowledge"])
            if cognitive_models:
                recommendations["cognitive_processing"] = cognitive_models
            
            # Multimodal perception: vision + audio_processor + sensor_processor + manager + language
            multimodal_models = []
            if "vision" in available_models:
                multimodal_models.append(model_to_letter["vision"])
            elif "computer_vision" in available_models:
                multimodal_models.append(model_to_letter["computer_vision"])
            if "audio_processor" in available_models:
                multimodal_models.append(model_to_letter["audio_processor"])
            if "sensor_processor" in available_models:
                multimodal_models.append(model_to_letter["sensor_processor"])
            if "manager" in available_models:
                multimodal_models.append(model_to_letter["manager"])
            if "language" in available_models:
                multimodal_models.append(model_to_letter["language"])
            if multimodal_models:
                recommendations["multimodal_perception"] = multimodal_models
            
            # Full system: all available models
            full_models = []
            for model_id in available_models:
                # Exclude test models
                if not model_id.startswith("test_"):
                    full_models.append(model_to_letter.get(model_id, model_id))
            if full_models:
                recommendations["full_system"] = full_models
            
            # Remove any empty recommendations
            recommendations = {k: v for k, v in recommendations.items() if v}
            
            return {
                "success": True,
                "recommendations": recommendations
            }
        except Exception as e:
            error_handler.handle_error(e, "EnhancedTrainingManager", f"Failed to get joint training recommendations: {str(e)}")
            return {"success": False, "message": f"Failed to get joint training recommendations: {str(e)}"}
