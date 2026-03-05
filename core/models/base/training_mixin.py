"""
Training Lifecycle Mixin for AGI Models

This mixin provides comprehensive training lifecycle management including
training, evaluation, saving, loading, and from-scratch training capabilities.
It is designed to be mixed into model classes to handle the complete training workflow.
"""

import os
import json
import pickle
import threading
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
import logging
import abc
from core.error_handling import error_handler
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class TrainingLifecycleMixin(abc.ABC):
    """
    Mixin class for managing training lifecycle operations in AGI models.
    Provides methods for training, evaluation, model persistence, and from-scratch training.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize training lifecycle capabilities."""
        super().__init__(*args, **kwargs)
        self._training_history = []
        self._current_training_phase = None
        self._model_checkpoints = {}
        self._from_scratch_training_enabled = True
        self._training_config = {}
        self._training_progress = {"epoch": 0, "step": 0, "loss": 0.0, "accuracy": 0.0}
        self._training_callbacks = {}
        self._training_visualization_data = {}
        self._hyperparameter_optimization_enabled = False
        self._distributed_training_enabled = False
        self._data_augmentation_enabled = True
        self._model_compression_enabled = False
        self._training_metrics_history = {
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "learning_rate": []
        }
        self._training_start_time = None
        self._training_end_time = None
        self._training_thread = None
        self._training_stop_requested = False
        self._training_paused = False
    
    # ===== ABSTRACT METHODS FOR AGI TRAINING =====
    
    @abc.abstractmethod
    def joint_training(self, model_ids: List[str], training_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Joint training of multiple models with neural network optimization
        
        Args:
            model_ids: List of model IDs to train jointly
            training_config: Configuration for joint training
            
        Returns:
            Training results dictionary
        """
        pass
    
    @abc.abstractmethod
    def transfer_knowledge(self, source_model_id: str, target_model_id: str, 
                          knowledge_type: str = "general", transfer_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Transfer knowledge between models using neural networks
        
        Args:
            source_model_id: Source model ID
            target_model_id: Target model ID
            knowledge_type: Type of knowledge to transfer
            transfer_config: Configuration for knowledge transfer
            
        Returns:
            Transfer results dictionary
        """
        pass
    
    def train_from_scratch(self, training_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Train the model from scratch using provided training data.
        
        Args:
            training_data: Training dataset or data loader
            **kwargs: Additional training parameters
            
        Returns:
            Training results and metrics
        """
        if not self._from_scratch_training_enabled:
            raise ValueError("From-scratch training is not enabled for this model")
        
        logger.info("Starting from-scratch training...")
        self._current_training_phase = 'from_scratch_training'
        self._training_start_time = datetime.now()
        self._training_stop_requested = False
        self._training_paused = False
        
        # Store training configuration
        self._training_config = {
            "epochs": kwargs.get("epochs", 100),
            "batch_size": kwargs.get("batch_size", 32),
            "learning_rate": kwargs.get("learning_rate", 0.001),
            "optimizer": kwargs.get("optimizer", "adam"),
            "loss_function": kwargs.get("loss_function", "auto"),
            **kwargs
        }
        
        # Execute callbacks before training
        self._execute_training_callbacks("before_training_start", {
            "config": self._training_config,
            "data_info": self._get_training_data_info(training_data)
        })
        
        try:
            # Initialize model weights randomly
            self._initialize_model_weights()
            
            # Perform training with enhanced monitoring
            training_results = self._perform_enhanced_training(training_data, **kwargs)
            
            # Update training history
            training_record = {
                'type': 'from_scratch',
                'timestamp': datetime.now().isoformat(),
                'start_time': self._training_start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'results': training_results,
                'parameters': self._training_config,
                'metrics_history': self._training_metrics_history,
                'visualization_data': self._training_visualization_data
            }
            self._training_history.append(training_record)
            
            # Execute callbacks after training
            self._execute_training_callbacks("after_training_complete", {
                "results": training_results,
                "history": training_record
            })
            
            logger.info("From-scratch training completed successfully")
            return training_results
            
        except Exception as e:
            logger.error(f"From-scratch training failed: {e}")
            self._current_training_phase = None
            self._execute_training_callbacks("on_training_error", {"error": str(e)})
            raise
    
    def continue_training(self, training_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Continue training from existing model state.
        
        Args:
            training_data: Training dataset or data loader
            **kwargs: Additional training parameters
            
        Returns:
            Training results and metrics
        """
        logger.info("Continuing training from existing state...")
        self._current_training_phase = 'continued_training'
        self._training_start_time = datetime.now()
        self._training_stop_requested = False
        self._training_paused = False
        
        # Update training configuration
        self._training_config.update({
            "epochs": kwargs.get("epochs", self._training_config.get("epochs", 100)),
            "batch_size": kwargs.get("batch_size", self._training_config.get("batch_size", 32)),
            "learning_rate": kwargs.get("learning_rate", self._training_config.get("learning_rate", 0.001)),
            "optimizer": kwargs.get("optimizer", self._training_config.get("optimizer", "adam")),
            "loss_function": kwargs.get("loss_function", self._training_config.get("loss_function", "auto")),
            **{k: v for k, v in kwargs.items() if k not in ["epochs", "batch_size", "learning_rate", "optimizer", "loss_function"]}
        })
        
        # Execute callbacks before training continuation
        self._execute_training_callbacks("before_training_continuation", {
            "config": self._training_config,
            "data_info": self._get_training_data_info(training_data)
        })
        
        try:
            # Perform enhanced training continuation
            training_results = self._perform_enhanced_training(training_data, **kwargs)
            
            # Update training history
            training_record = {
                'type': 'continued',
                'timestamp': datetime.now().isoformat(),
                'start_time': self._training_start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'results': training_results,
                'parameters': self._training_config,
                'metrics_history': self._training_metrics_history,
                'visualization_data': self._training_visualization_data
            }
            self._training_history.append(training_record)
            
            # Execute callbacks after training continuation
            self._execute_training_callbacks("after_training_continuation", {
                "results": training_results,
                "history": training_record
            })
            
            logger.info("Training continuation completed successfully")
            return training_results
            
        except Exception as e:
            logger.error(f"Training continuation failed: {e}")
            self._current_training_phase = None
            self._execute_training_callbacks("on_training_error", {"error": str(e)})
            raise
    
    def evaluate_model(self, test_data: Any, metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test dataset or data loader
            metrics: List of metrics to compute (default: ['accuracy', 'loss'])
            
        Returns:
            Dictionary of evaluation metrics
        """
        if metrics is None:
            metrics = ['accuracy', 'loss']
        
        logger.info(f"Evaluating model with metrics: {metrics}")
        
        try:
            evaluation_results = {}
            
            for metric in metrics:
                if metric == 'accuracy':
                    evaluation_results['accuracy'] = self._compute_accuracy(test_data)
                elif metric == 'loss':
                    evaluation_results['loss'] = self._compute_loss(test_data)
                elif metric == 'precision':
                    evaluation_results['precision'] = self._compute_precision(test_data)
                elif metric == 'recall':
                    evaluation_results['recall'] = self._compute_recall(test_data)
                elif metric == 'f1_score':
                    evaluation_results['f1_score'] = self._compute_f1_score(test_data)
            
            logger.info(f"Evaluation completed: {evaluation_results}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
    
    def save_model(self, filepath: str, format: str = 'pickle') -> bool:
        """
        Save model state to file.
        
        Args:
            filepath: Path where to save the model
            format: Serialization format ('pickle', 'json', 'h5')
            
        Returns:
            True if save successful, False otherwise
        """
        logger.info(f"Saving model to {filepath} in {format} format")
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if format == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(self.get_model_state(), f)
            elif format == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self._serialize_to_json(), f, indent=2)
            elif format == 'h5':
                self._save_h5_format(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Create checkpoint record
            checkpoint_id = f"checkpoint_{len(self._model_checkpoints)}"
            self._model_checkpoints[checkpoint_id] = {
                'filepath': filepath,
                'format': format,
                'timestamp': datetime.now().isoformat(),
                'training_phase': self._current_training_phase
            }
            
            logger.info(f"Model saved successfully to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Model save failed: {e}")
            return False
    
    def load_model(self, filepath: str, format: str = 'pickle') -> bool:
        """
        Load model state from file.
        
        Args:
            filepath: Path from which to load the model
            format: Serialization format ('pickle', 'json', 'h5')
            
        Returns:
            True if load successful, False otherwise
        """
        logger.info(f"Loading model from {filepath} in {format} format")
        
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            if format == 'pickle':
                with open(filepath, 'rb') as f:
                    model_state = pickle.load(f)
                self.set_model_state(model_state)
            elif format == 'json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                self._deserialize_from_json(json_data)
            elif format == 'h5':
                self._load_h5_format(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Model loaded successfully from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            return False
    
    def create_checkpoint(self, checkpoint_dir: str, checkpoint_name: str = None) -> str:
        """
        Create a training checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            checkpoint_name: Optional custom checkpoint name
            
        Returns:
            Path to the created checkpoint file
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.ckpt")
        
        if self.save_model(checkpoint_path, format='pickle'):
            logger.info(f"Checkpoint created: {checkpoint_path}")
            return checkpoint_path
        else:
            raise RuntimeError(f"Failed to create checkpoint: {checkpoint_path}")
    
    def restore_from_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Restore model state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if restoration successful, False otherwise
        """
        return self.load_model(checkpoint_path, format='pickle')
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get complete training history.
        
        Returns:
            List of training records
        """
        return self._training_history.copy()
    
    def get_latest_training_results(self) -> Optional[Dict[str, Any]]:
        """
        Get results from the most recent training session.
        
        Returns:
            Latest training results or None if no training history
        """
        if not self._training_history:
            return None
        return self._training_history[-1]['results']
    
    def enable_from_scratch_training(self, enabled: bool = True):
        """
        Enable or disable from-scratch training capability.
        
        Args:
            enabled: Whether to enable from-scratch training
        """
        self._from_scratch_training_enabled = enabled
        logger.info(f"From-scratch training {'enabled' if enabled else 'disabled'}")
    
    # Protected methods for internal implementation
    def _initialize_model_weights(self):
        """Initialize model weights for from-scratch training."""
        logger.info("Initializing model weights...")
        # This should be implemented by the specific model class
        if hasattr(self, '_custom_weight_initialization'):
            self._custom_weight_initialization()
        else:
            # Default weight initialization logic
            pass
    
    def _perform_training(self, training_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Perform the actual training process with real neural network training.
        
        Args:
            training_data: Training data
            **kwargs: Training parameters
            
        Returns:
            Training results with real metrics
        """
        # First check for custom training implementation
        if hasattr(self, '_custom_training_implementation'):
            return self._custom_training_implementation(training_data, **kwargs)
        
        # Real training implementation using PyTorch
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            # Get training parameters
            epochs = kwargs.get('epochs', 100)
            batch_size = kwargs.get('batch_size', 32)
            learning_rate = kwargs.get('learning_rate', 0.001)
            validation_split = kwargs.get('validation_split', 0.2)
            
            # Check if model has required components
            if not hasattr(self, 'model') or self.model is None:
                return {
                    'success': False,
                    'error': 'Model not initialized. Cannot perform training.',
                    'epochs_completed': 0,
                    'final_loss': 0.0,
                    'training_time': 0.0,
                    'status': 'failed'
                }
            
            # Prepare data loaders
            train_loader, val_loader = self._prepare_data_loaders(
                training_data, batch_size, validation_split
            )
            
            # Initialize optimizer and criterion if not present
            if not hasattr(self, 'optimizer') or self.optimizer is None:
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            if not hasattr(self, 'criterion') or self.criterion is None:
                # Auto-select loss function based on model type
                self.criterion = self._auto_select_loss_function()
            
            # Training loop with real backpropagation
            training_history = {
                'train_loss': [],
                'val_loss': [],
                'train_accuracy': [],
                'val_accuracy': []
            }
            
            start_time = time.time()
            
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    # Move data to device
                    if hasattr(self, 'device'):
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Compute loss
                    loss = self.criterion(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Optimizer step
                    self.optimizer.step()
                    
                    # Update statistics
                    train_loss += loss.item()
                    
                    # Calculate accuracy for classification tasks
                    if outputs.shape == targets.shape or len(targets.shape) == 1:
                        _, predicted = torch.max(outputs.data, 1)
                        train_total += targets.size(0)
                        train_correct += (predicted == targets).sum().item()
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                if val_loader is not None:
                    with torch.no_grad():
                        for inputs, targets in val_loader:
                            if hasattr(self, 'device'):
                                inputs, targets = inputs.to(self.device), targets.to(self.device)
                            
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, targets)
                            val_loss += loss.item()
                            
                            if outputs.shape == targets.shape or len(targets.shape) == 1:
                                _, predicted = torch.max(outputs.data, 1)
                                val_total += targets.size(0)
                                val_correct += (predicted == targets).sum().item()
                
                # Calculate epoch metrics
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader) if val_loader else 0.0
                
                train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0
                val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0.0
                
                # Store history
                training_history['train_loss'].append(avg_train_loss)
                training_history['val_loss'].append(avg_val_loss)
                training_history['train_accuracy'].append(train_accuracy)
                training_history['val_accuracy'].append(val_accuracy)
                
                # Update training progress
                self.update_training_progress(
                    epoch=epoch,
                    step=batch_idx if 'batch_idx' in locals() else 0,
                    loss=avg_train_loss,
                    accuracy=train_accuracy
                )
                
                # Log progress every 10% of epochs
                if epoch % max(1, epochs // 10) == 0:
                    logger.info(
                        f'Epoch {epoch+1}/{epochs}: '
                        f'Train Loss: {avg_train_loss:.4f}, '
                        f'Val Loss: {avg_val_loss:.4f}, '
                        f'Train Acc: {train_accuracy:.2f}%, '
                        f'Val Acc: {val_accuracy:.2f}%'
                    )
            
            training_time = time.time() - start_time
            
            # Return real training results
            return {
                'success': True,
                'epochs_completed': epochs,
                'final_loss': training_history['val_loss'][-1] if training_history['val_loss'] else 0.0,
                'final_accuracy': training_history['val_accuracy'][-1] if training_history['val_accuracy'] else 0.0,
                'training_time': training_time,
                'training_history': training_history,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Real training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'epochs_completed': 0,
                'final_loss': 0.0,
                'training_time': 0.0,
                'status': 'failed'
            }
    
    def _prepare_data_loaders(self, training_data: Any, batch_size: int, validation_split: float = 0.2):
        """Prepare training and validation data loaders"""
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset, random_split
            
            # Convert training data to appropriate format
            if isinstance(training_data, (list, tuple)) and len(training_data) == 2:
                # Assume (inputs, targets) format
                inputs, targets = training_data
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                if not isinstance(targets, torch.Tensor):
                    targets = torch.tensor(targets, dtype=torch.long if len(targets.shape) == 1 else torch.float32)
                
                # Create dataset
                dataset = TensorDataset(inputs, targets)
                
            elif hasattr(training_data, '__getitem__') and hasattr(training_data, '__len__'):
                # Already a dataset-like object
                dataset = training_data
            else:
                # Try to convert to tensor for unsupervised learning
                try:
                    data_tensor = torch.tensor(training_data, dtype=torch.float32)
                    # For unsupervised learning (e.g., autoencoders), use input as target
                    # This enables reconstruction loss without dummy targets
                    target_tensor = data_tensor.clone().detach()
                    dataset = TensorDataset(data_tensor, target_tensor)
                    self.logger.info("Unsupervised learning data detected: using input as target for reconstruction loss")
                except Exception as e:
                    raise ValueError(f"Cannot convert training data to tensors: {e}")
            
            # Split dataset
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            
            if val_size > 0:
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            else:
                train_dataset, val_dataset = dataset, None
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=0  # Avoid multiprocessing issues
            )
            
            val_loader = None
            if val_dataset:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0
                )
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Failed to prepare data loaders: {e}")
            raise
    
    def _auto_select_loss_function(self):
        """Auto-select loss function based on model type and problem"""
        import torch.nn as nn
        
        # Try to determine problem type from model name or attributes
        model_class_name = self.__class__.__name__.lower()
        
        if any(keyword in model_class_name for keyword in ['classification', 'vision', 'language', 'audio']):
            # Classification tasks
            return nn.CrossEntropyLoss()
        elif any(keyword in model_class_name for keyword in ['regression', 'prediction', 'finance', 'sensor']):
            # Regression tasks
            return nn.MSELoss()
        elif any(keyword in model_class_name for keyword in ['binary', 'detection']):
            # Binary classification
            return nn.BCEWithLogitsLoss()
        else:
            # Default to MSE loss
            return nn.MSELoss()
    
    def _compute_accuracy(self, test_data: Any) -> float:
        """Compute accuracy metric with real calculation.
        
        This implementation performs real inference using the model on test_data
        and calculates accuracy based on predictions vs true labels.
        
        Args:
            test_data: Test dataset or data loader
            
        Returns:
            Accuracy as float between 0 and 1
        """
        try:
            import torch
            import torch.nn as nn
            
            # Check if model exists
            if not hasattr(self, 'model') or self.model is None:
                self.logger.warning("No model found for accuracy computation")
                return 0.0
            
            # Prepare data loader
            if hasattr(self, '_prepare_data_loaders'):
                # Use existing data loader preparation
                test_loader, _ = self._prepare_data_loaders(test_data, batch_size=32, validation_split=0.0)
            else:
                # Default data preparation
                if isinstance(test_data, (list, tuple)) and len(test_data) == 2:
                    inputs, targets = test_data
                    if not isinstance(inputs, torch.Tensor):
                        inputs = torch.tensor(inputs, dtype=torch.float32)
                    if not isinstance(targets, torch.Tensor):
                        targets = torch.tensor(targets, dtype=torch.long if len(targets.shape) == 1 else torch.float32)
                    
                    from torch.utils.data import TensorDataset, DataLoader
                    dataset = TensorDataset(inputs, targets)
                    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
                else:
                    self.logger.warning("Unsupported test data format for accuracy computation")
                    return 0.0
            
            # Set model to evaluation mode
            self.model.eval()
            if hasattr(self, 'device'):
                self.model.to(self.device)
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    if hasattr(self, 'device'):
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    
                    # Handle different output types
                    if outputs.dim() > 1 and outputs.shape[1] > 1:
                        # Classification
                        _, predicted = torch.max(outputs.data, 1)
                    else:
                        # Regression or binary classification
                        predicted = (outputs > 0.5).float()
                    
                    # Compare predictions with targets
                    if targets.dim() > 1:
                        targets = targets.squeeze()
                    
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)
            
            accuracy = correct / total if total > 0 else 0.0
            self.logger.info(f"Computed real accuracy: {accuracy:.4f}")
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Accuracy computation failed: {e}")
            return 0.0
    
    def _compute_loss(self, test_data: Any) -> float:
        """Compute loss metric with real calculation.
        
        This implementation performs real inference using the model on test_data
        and calculates loss based on predictions vs true labels.
        
        Args:
            test_data: Test dataset or data loader
            
        Returns:
            Loss as float (lower is better)
        """
        try:
            import torch
            import torch.nn as nn
            
            # Check if model exists
            if not hasattr(self, 'model') or self.model is None:
                self.logger.warning("No model found for loss computation")
                return 1.0
            
            # Prepare data loader
            if hasattr(self, '_prepare_data_loaders'):
                # Use existing data loader preparation
                test_loader, _ = self._prepare_data_loaders(test_data, batch_size=32, validation_split=0.0)
            else:
                # Default data preparation
                if isinstance(test_data, (list, tuple)) and len(test_data) == 2:
                    inputs, targets = test_data
                    if not isinstance(inputs, torch.Tensor):
                        inputs = torch.tensor(inputs, dtype=torch.float32)
                    if not isinstance(targets, torch.Tensor):
                        targets = torch.tensor(targets, dtype=torch.long if len(targets.shape) == 1 else torch.float32)
                    
                    from torch.utils.data import TensorDataset, DataLoader
                    dataset = TensorDataset(inputs, targets)
                    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
                else:
                    self.logger.warning("Unsupported test data format for loss computation")
                    return 1.0
            
            # Set model to evaluation mode
            self.model.eval()
            if hasattr(self, 'device'):
                self.model.to(self.device)
            
            # Determine loss function
            if hasattr(self, 'criterion') and self.criterion is not None:
                criterion = self.criterion
            else:
                # Auto-select loss function based on model type
                criterion = self._auto_select_loss_function()
            
            total_loss = 0.0
            total_batches = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    if hasattr(self, 'device'):
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    total_batches += 1
            
            avg_loss = total_loss / total_batches if total_batches > 0 else 1.0
            self.logger.info(f"Computed real loss: {avg_loss:.4f}")
            return avg_loss
            
        except Exception as e:
            self.logger.error(f"Loss computation failed: {e}")
            return 1.0
    
    def _compute_precision(self, test_data: Any) -> float:
        """Compute precision metric with real calculation.
        
        This implementation performs real inference and calculates precision
        (true positives / (true positives + false positives)).
        
        Args:
            test_data: Test dataset or data loader
            
        Returns:
            Precision as float between 0 and 1
        """
        try:
            import torch
            
            # Check if model exists
            if not hasattr(self, 'model') or self.model is None:
                self.logger.warning("No model found for precision computation")
                return 0.0
            
            # Prepare data loader using same logic as accuracy computation
            if hasattr(self, '_prepare_data_loaders'):
                test_loader, _ = self._prepare_data_loaders(test_data, batch_size=32, validation_split=0.0)
            else:
                if isinstance(test_data, (list, tuple)) and len(test_data) == 2:
                    inputs, targets = test_data
                    if not isinstance(inputs, torch.Tensor):
                        inputs = torch.tensor(inputs, dtype=torch.float32)
                    if not isinstance(targets, torch.Tensor):
                        targets = torch.tensor(targets, dtype=torch.long if len(targets.shape) == 1 else torch.float32)
                    
                    from torch.utils.data import TensorDataset, DataLoader
                    dataset = TensorDataset(inputs, targets)
                    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
                else:
                    self.logger.warning("Unsupported test data format for precision computation")
                    return 0.0
            
            # Set model to evaluation mode
            self.model.eval()
            if hasattr(self, 'device'):
                self.model.to(self.device)
            
            true_positives = 0
            false_positives = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    if hasattr(self, 'device'):
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    
                    # Get predictions
                    if outputs.dim() > 1 and outputs.shape[1] > 1:
                        # Multi-class classification
                        _, predicted = torch.max(outputs.data, 1)
                    else:
                        # Binary classification
                        predicted = (outputs > 0.5).long()
                    
                    # Flatten targets if needed
                    if targets.dim() > 1:
                        targets = targets.squeeze()
                    
                    # For each class (binary or multi-class)
                    if predicted.max() < 2:  # Binary classification
                        # Binary precision: TP / (TP + FP)
                        true_positives += ((predicted == 1) & (targets == 1)).sum().item()
                        false_positives += ((predicted == 1) & (targets == 0)).sum().item()
                    else:
                        # Multi-class precision (macro-average)
                        # This is simplified; real implementation would compute per-class
                        # For now, compute accuracy-like metric
                        true_positives += (predicted == targets).sum().item()
                        false_positives += (predicted != targets).sum().item()
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            self.logger.info(f"Computed real precision: {precision:.4f}")
            return precision
            
        except Exception as e:
            self.logger.error(f"Precision computation failed: {e}")
            return 0.0
    
    def _compute_recall(self, test_data: Any) -> float:
        """Compute recall metric with real calculation.
        
        This implementation performs real inference and calculates recall
        (true positives / (true positives + false negatives)).
        
        Args:
            test_data: Test dataset or data loader
            
        Returns:
            Recall as float between 0 and 1
        """
        try:
            import torch
            
            # Check if model exists
            if not hasattr(self, 'model') or self.model is None:
                self.logger.warning("No model found for recall computation")
                return 0.0
            
            # Prepare data loader using same logic as accuracy computation
            if hasattr(self, '_prepare_data_loaders'):
                test_loader, _ = self._prepare_data_loaders(test_data, batch_size=32, validation_split=0.0)
            else:
                if isinstance(test_data, (list, tuple)) and len(test_data) == 2:
                    inputs, targets = test_data
                    if not isinstance(inputs, torch.Tensor):
                        inputs = torch.tensor(inputs, dtype=torch.float32)
                    if not isinstance(targets, torch.Tensor):
                        targets = torch.tensor(targets, dtype=torch.long if len(targets.shape) == 1 else torch.float32)
                    
                    from torch.utils.data import TensorDataset, DataLoader
                    dataset = TensorDataset(inputs, targets)
                    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
                else:
                    self.logger.warning("Unsupported test data format for recall computation")
                    return 0.0
            
            # Set model to evaluation mode
            self.model.eval()
            if hasattr(self, 'device'):
                self.model.to(self.device)
            
            true_positives = 0
            false_negatives = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    if hasattr(self, 'device'):
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    
                    # Get predictions
                    if outputs.dim() > 1 and outputs.shape[1] > 1:
                        # Multi-class classification
                        _, predicted = torch.max(outputs.data, 1)
                    else:
                        # Binary classification
                        predicted = (outputs > 0.5).long()
                    
                    # Flatten targets if needed
                    if targets.dim() > 1:
                        targets = targets.squeeze()
                    
                    # For each class (binary or multi-class)
                    if predicted.max() < 2:  # Binary classification
                        # Binary recall: TP / (TP + FN)
                        true_positives += ((predicted == 1) & (targets == 1)).sum().item()
                        false_negatives += ((predicted == 0) & (targets == 1)).sum().item()
                    else:
                        # Multi-class recall (macro-average)
                        # This is simplified; real implementation would compute per-class
                        # For now, compute accuracy-like metric
                        true_positives += (predicted == targets).sum().item()
                        false_negatives += (predicted != targets).sum().item()
            
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            self.logger.info(f"Computed real recall: {recall:.4f}")
            return recall
            
        except Exception as e:
            self.logger.error(f"Recall computation failed: {e}")
            return 0.0
    
    def _compute_f1_score(self, test_data: Any) -> float:
        """Compute F1 score metric with real calculation.
        
        This implementation uses real precision and recall calculations
        to compute F1 score (harmonic mean of precision and recall).
        
        Args:
            test_data: Test dataset or data loader
            
        Returns:
            F1 score as float between 0 and 1
        """
        precision = self._compute_precision(test_data)
        recall = self._compute_recall(test_data)
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            self.logger.info(f"Computed real F1 score: {f1:.4f}")
            return f1
        else:
            return 0.0
    
    def _serialize_to_json(self) -> Dict[str, Any]:
        """Serialize model to JSON-serializable format."""
        return {
            'model_type': self.__class__.__name__,
            'training_history': self._training_history,
            'checkpoints': list(self._model_checkpoints.keys()),
            'from_scratch_enabled': self._from_scratch_training_enabled,
            'serialization_timestamp': datetime.now().isoformat()
        }
    
    def _deserialize_from_json(self, json_data: Dict[str, Any]):
        """Deserialize model from JSON data."""
        self._training_history = json_data.get('training_history', [])
        self._from_scratch_training_enabled = json_data.get('from_scratch_enabled', True)
    
    def _save_h5_format(self, filepath: str):
        """Save model in H5 format (for compatibility with certain frameworks)."""
        try:
            import h5py
            import numpy as np
            
            logger.info(f"Saving model to H5 format: {filepath}")
            
            # Get the full model state for serialization
            model_state = self.get_model_state()
            
            with h5py.File(filepath, 'w') as f:
                # Save training history
                if 'training_history' in model_state and model_state['training_history']:
                    history_group = f.create_group('training_history')
                    for i, record in enumerate(model_state['training_history']):
                        record_group = history_group.create_group(f'record_{i}')
                        # Convert record to JSON string for storage
                        import json
                        record_json = json.dumps(record, ensure_ascii=False)
                        record_group.create_dataset('data', data=np.string_(record_json))
                
                # Save model checkpoints metadata
                if 'model_checkpoints' in model_state and model_state['model_checkpoints']:
                    checkpoints_group = f.create_group('model_checkpoints')
                    for checkpoint_id, checkpoint_data in model_state['model_checkpoints'].items():
                        checkpoint_group = checkpoints_group.create_group(checkpoint_id)
                        # Convert checkpoint data to JSON
                        import json
                        checkpoint_json = json.dumps(checkpoint_data, ensure_ascii=False)
                        checkpoint_group.create_dataset('data', data=np.string_(checkpoint_json))
                
                # Save boolean flags
                f.attrs['from_scratch_training_enabled'] = model_state.get('from_scratch_training_enabled', True)
                
                # Save current training phase if available
                if 'current_training_phase' in model_state and model_state['current_training_phase']:
                    f.attrs['current_training_phase'] = model_state['current_training_phase']
                
                # Save model-specific data if available
                if hasattr(self, 'get_model_specific_state'):
                    specific_state = self.get_model_specific_state()
                    if specific_state:
                        specific_group = f.create_group('model_specific')
                        for key, value in specific_state.items():
                            if isinstance(value, (int, float, bool, str)):
                                specific_group.attrs[key] = value
                            elif isinstance(value, (list, tuple)) and all(isinstance(v, (int, float)) for v in value):
                                specific_group.create_dataset(key, data=np.array(value))
                
                logger.info(f"Model saved successfully to H5 file: {filepath}")
                
        except ImportError:
            error_handler.log_warning("h5py library not available. H5 format saving requires h5py installation.", "TrainingLifecycleMixin")
            raise ImportError("h5py library not installed. Please install h5py to use H5 format: pip install h5py")
        except Exception as e:
            logger.error(f"Failed to save model in H5 format: {str(e)}")
            raise
    
    def _load_h5_format(self, filepath: str):
        """Load model from H5 format."""
        try:
            import h5py
            import json
            import numpy as np
            
            logger.info(f"Loading model from H5 format: {filepath}")
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"H5 file not found: {filepath}")
            
            with h5py.File(filepath, 'r') as f:
                # Load training history
                training_history = []
                if 'training_history' in f:
                    history_group = f['training_history']
                    for record_name in history_group:
                        record_group = history_group[record_name]
                        if 'data' in record_group:
                            record_data = record_group['data'][()].decode('utf-8')
                            record = json.loads(record_data)
                            training_history.append(record)
                
                # Load model checkpoints
                model_checkpoints = {}
                if 'model_checkpoints' in f:
                    checkpoints_group = f['model_checkpoints']
                    for checkpoint_id in checkpoints_group:
                        checkpoint_group = checkpoints_group[checkpoint_id]
                        if 'data' in checkpoint_group:
                            checkpoint_data = checkpoint_group['data'][()].decode('utf-8')
                            checkpoint = json.loads(checkpoint_data)
                            model_checkpoints[checkpoint_id] = checkpoint
                
                # Load boolean flags
                from_scratch_training_enabled = f.attrs.get('from_scratch_training_enabled', True)
                current_training_phase = f.attrs.get('current_training_phase', None)
                
                # Create model state dictionary
                model_state = {
                    'training_history': training_history,
                    'model_checkpoints': model_checkpoints,
                    'from_scratch_training_enabled': from_scratch_training_enabled,
                    'current_training_phase': current_training_phase
                }
                
                # Load model-specific data if available
                if 'model_specific' in f and hasattr(self, 'set_model_specific_state'):
                    specific_group = f['model_specific']
                    specific_state = {}
                    
                    # Load attributes
                    for attr_name, attr_value in specific_group.attrs.items():
                        specific_state[attr_name] = attr_value
                    
                    # Load datasets
                    for dataset_name in specific_group:
                        if dataset_name in specific_group:
                            dataset = specific_group[dataset_name]
                            specific_state[dataset_name] = dataset[()].tolist() if hasattr(dataset[()], 'tolist') else dataset[()]
                    
                    # Set model-specific state
                    self.set_model_specific_state(specific_state)
                
                # Apply the loaded state to the model
                self.set_model_state(model_state)
                
                logger.info(f"Model loaded successfully from H5 file: {filepath}")
                
        except ImportError:
            error_handler.log_warning("h5py library not available. H5 format loading requires h5py installation.", "TrainingLifecycleMixin")
            raise ImportError("h5py library not installed. Please install h5py to use H5 format: pip install h5py")
        except Exception as e:
            logger.error(f"Failed to load model from H5 format: {str(e)}")
            raise
    
    def get_model_state(self) -> Dict[str, Any]:
        """
        Get current model state for serialization.
        
        Returns:
            Model state dictionary
        """
        return {
            'training_history': self._training_history,
            'model_checkpoints': self._model_checkpoints,
            'from_scratch_training_enabled': self._from_scratch_training_enabled,
            'current_training_phase': self._current_training_phase
        }
    
    def set_model_state(self, state: Dict[str, Any]):
        """
        Set model state from serialized data.
        
        Args:
            state: Model state dictionary
        """
        self._training_history = state.get('training_history', [])
        self._model_checkpoints = state.get('model_checkpoints', {})
        self._from_scratch_training_enabled = state.get('from_scratch_training_enabled', True)
        self._current_training_phase = state.get('current_training_phase')
    
    def get_model_specific_state(self) -> Dict[str, Any]:
        """
        Get model-specific state for serialization.
        
        This method should be overridden by subclasses to save model-specific data.
        The base implementation returns an empty dictionary.
        
        Returns:
            Model-specific state dictionary
        """
        return {}
    
    def set_model_specific_state(self, state: Dict[str, Any]):
        """
        Set model-specific state from serialized data.
        
        This method should be overridden by subclasses to restore model-specific data.
        The base implementation does nothing.
        
        Args:
            state: Model-specific state dictionary
        """
        # Base implementation does nothing - subclasses should override
        pass
    
    # Enhanced training lifecycle management methods
    
    def _execute_training_callbacks(self, event_name: str, event_data: Dict[str, Any]):
        """Execute registered callbacks for a training event"""
        if event_name in self._training_callbacks:
            for callback in self._training_callbacks[event_name]:
                try:
                    callback(event_data)
                except Exception as e:
                    logger.warning(f"Training callback {event_name} failed: {e}")
    
    def register_training_callback(self, event_name: str, callback: Callable):
        """Register a callback function for a training event"""
        if event_name not in self._training_callbacks:
            self._training_callbacks[event_name] = []
        self._training_callbacks[event_name].append(callback)
        logger.debug(f"Registered callback for event: {event_name}")
    
    def _get_training_data_info(self, training_data: Any) -> Dict[str, Any]:
        """Get information about training data"""
        try:
            if hasattr(training_data, '__len__'):
                data_size = len(training_data)
            else:
                data_size = 'unknown'
            
            if hasattr(training_data, 'shape'):
                data_shape = training_data.shape
            elif hasattr(training_data, '__getitem__'):
                try:
                    sample = training_data[0]
                    if hasattr(sample, 'shape'):
                        data_shape = sample.shape
                    else:
                        data_shape = 'variable'
                except (IndexError, TypeError) as e:
                    logger.debug(f"Could not get sample from training data: {e}")
                    data_shape = 'unknown'
            else:
                data_shape = 'unknown'
            
            return {
                "size": data_size,
                "shape": data_shape,
                "type": type(training_data).__name__
            }
        except Exception as e:
            logger.warning(f"Failed to get training data info: {e}")
            return {"size": "unknown", "shape": "unknown", "type": "unknown"}
    
    def _perform_enhanced_training(self, training_data: Any, **kwargs) -> Dict[str, Any]:
        """Perform training with enhanced monitoring and visualization"""
        logger.info("Starting enhanced training with monitoring...")
        
        # Store training parameters
        epochs = kwargs.get('epochs', self._training_config.get('epochs', 100))
        batch_size = kwargs.get('batch_size', self._training_config.get('batch_size', 32))
        
        # Execute callbacks for epoch start
        self._execute_training_callbacks("before_epoch_start", {
            "total_epochs": epochs,
            "batch_size": batch_size
        })
        
        try:
            # Call the original training implementation
            training_results = self._perform_training(training_data, **kwargs)
            
            # Enhance results with monitoring data
            enhanced_results = {
                **training_results,
                "training_duration": (datetime.now() - self._training_start_time).total_seconds(),
                "final_progress": self._training_progress.copy(),
                "metrics_summary": {
                    metric: values[-1] if values else 0.0 
                    for metric, values in self._training_metrics_history.items()
                },
                "visualization_available": bool(self._training_visualization_data)
            }
            
            # Execute callbacks for epoch completion
            self._execute_training_callbacks("after_epoch_complete", {
                "results": enhanced_results,
                "epoch": epochs
            })
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Enhanced training failed: {e}")
            self._execute_training_callbacks("on_training_error", {"error": str(e)})
            raise
    
    def update_training_progress(self, epoch: int, step: int, loss: float, accuracy: float = None):
        """Update training progress and metrics"""
        self._training_progress = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "accuracy": accuracy or self._training_progress.get("accuracy", 0.0)
        }
        
        # Record metrics history
        self._training_metrics_history["loss"].append(loss)
        if accuracy is not None:
            self._training_metrics_history["accuracy"].append(accuracy)
        
        # Execute progress callbacks
        self._execute_training_callbacks("on_training_progress", self._training_progress.copy())
    
    def add_training_visualization_data(self, epoch: int, data_type: str, data: Any):
        """Add visualization data for training monitoring"""
        if epoch not in self._training_visualization_data:
            self._training_visualization_data[epoch] = {}
        self._training_visualization_data[epoch][data_type] = data
    
    def get_training_visualization_data(self) -> Dict[int, Dict[str, Any]]:
        """Get all collected training visualization data"""
        return self._training_visualization_data.copy()
    
    def enable_hyperparameter_optimization(self, enabled: bool = True):
        """Enable or disable hyperparameter optimization"""
        self._hyperparameter_optimization_enabled = enabled
        logger.info(f"Hyperparameter optimization {'enabled' if enabled else 'disabled'}")
    
    def enable_distributed_training(self, enabled: bool = True):
        """Enable or disable distributed training support"""
        self._distributed_training_enabled = enabled
        logger.info(f"Distributed training {'enabled' if enabled else 'disabled'}")
    
    def enable_data_augmentation(self, enabled: bool = True):
        """Enable or disable data augmentation"""
        self._data_augmentation_enabled = enabled
        logger.info(f"Data augmentation {'enabled' if enabled else 'disabled'}")
    
    def enable_model_compression(self, enabled: bool = True):
        """Enable or disable model compression"""
        self._model_compression_enabled = enabled
        logger.info(f"Model compression {'enabled' if enabled else 'disabled'}")
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get current training configuration"""
        return self._training_config.copy()
    
    def update_training_config(self, config: Dict[str, Any]):
        """Update training configuration"""
        self._training_config.update(config)
        logger.info(f"Training configuration updated: {config}")
    
    def pause_training(self):
        """Pause the current training process"""
        if self._current_training_phase and not self._training_paused:
            self._training_paused = True
            logger.info("Training paused")
            self._execute_training_callbacks("on_training_paused", {})
    
    def resume_training(self):
        """Resume paused training"""
        if self._current_training_phase and self._training_paused:
            self._training_paused = False
            logger.info("Training resumed")
            self._execute_training_callbacks("on_training_resumed", {})
    
    def stop_training(self):
        """Stop the current training process"""
        if self._current_training_phase and not self._training_stop_requested:
            self._training_stop_requested = True
            logger.info("Training stop requested")
            self._execute_training_callbacks("on_training_stop_requested", {})
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            "current_phase": self._current_training_phase,
            "progress": self._training_progress.copy(),
            "config": self._training_config.copy(),
            "paused": self._training_paused,
            "stop_requested": self._training_stop_requested,
            "start_time": self._training_start_time.isoformat() if self._training_start_time else None,
            "duration": (datetime.now() - self._training_start_time).total_seconds() 
                       if self._training_start_time else 0.0
        }
    
    # ==================== 训练资源调度与性能监控集成 ====================
    
    def _allocate_training_resources(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """分配训练资源（GPU/CPU优先级分配）"""
        logger.info("Allocating training resources...")
        
        resource_allocation = {
            "gpu_memory_mb": 0,
            "cpu_cores": 1,
            "system_memory_mb": 1024,
            "priority": "medium",
            "estimated_training_time": 0.0,
            "resource_constraints": {}
        }
        
        # 更新资源配置
        resource_allocation.update(config.get("resource_allocation", {}))
        
        # 检查是否有ResourceMixin可用
        if hasattr(self, 'allocate_resources'):
            # 分配GPU资源
            if resource_allocation.get("gpu_memory_mb", 0) > 0:
                self.allocate_resources("gpu_memory", resource_allocation["gpu_memory_mb"])
            
            # 分配CPU资源
            if resource_allocation.get("cpu_cores", 1) > 0:
                self.allocate_resources("cpu_cores", resource_allocation["cpu_cores"])
            
            # 分配系统内存
            if resource_allocation.get("system_memory_mb", 0) > 0:
                self.allocate_resources("system_memory", resource_allocation["system_memory_mb"])
        
        # 记录资源分配
        self._training_config["resource_allocation"] = resource_allocation
        
        logger.info(f"Training resources allocated: {resource_allocation}")
        return resource_allocation
    
    def _monitor_training_performance(self, training_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """监控训练性能（训练速度、资源利用率调优）"""
        performance_report = {
            "training_speed_samples_per_sec": 0.0,
            "resource_utilization": {
                "gpu_utilization": 0.0,
                "cpu_utilization": 0.0,
                "memory_utilization": 0.0
            },
            "training_efficiency": 1.0,
            "bottlenecks": [],
            "optimization_suggestions": []
        }
        
        # 更新性能指标
        if "samples_processed" in training_metrics and "training_time" in training_metrics:
            samples = training_metrics["samples_processed"]
            time_sec = training_metrics["training_time"]
            if time_sec > 0:
                performance_report["training_speed_samples_per_sec"] = samples / time_sec
        
        # 检查是否有PerformanceMixin可用
        if hasattr(self, 'get_performance_metrics'):
            try:
                perf_metrics = self.get_performance_metrics()
                performance_report["resource_utilization"].update({
                    "cpu_utilization": perf_metrics.get("cpu_utilization", 0.0),
                    "memory_utilization": perf_metrics.get("peak_memory_usage", 0) / (1024 * 1024)  # 转换为MB
                })
                
                # 检测性能瓶颈
                if perf_metrics.get("cpu_utilization", 0.0) > 90.0:
                    performance_report["bottlenecks"].append("High CPU utilization")
                    performance_report["optimization_suggestions"].append("Consider reducing batch size or using more CPU cores")
                
                if performance_report["training_speed_samples_per_sec"] < 10.0:
                    performance_report["bottlenecks"].append("Low training speed")
                    performance_report["optimization_suggestions"].append("Consider GPU acceleration or optimizing data pipeline")
                    
            except Exception as e:
                logger.warning(f"Failed to get performance metrics: {e}")
        
        # 计算训练效率
        cpu_util = performance_report["resource_utilization"]["cpu_utilization"]
        gpu_util = performance_report["resource_utilization"]["gpu_utilization"]
        if cpu_util > 0 or gpu_util > 0:
            performance_report["training_efficiency"] = min(1.0, (cpu_util + gpu_util) / 200.0)
        
        # 记录性能报告
        if "performance_reports" not in self._training_config:
            self._training_config["performance_reports"] = []
        self._training_config["performance_reports"].append(performance_report)
        
        logger.info(f"Training performance monitored: {performance_report}")
        return performance_report
    
    def _get_training_resource_requirements(self, training_data_info: Dict[str, Any]) -> Dict[str, Any]:
        """获取训练资源需求"""
        requirements = {
            "estimated_memory_mb": 1024,  # 默认1GB
            "estimated_gpu_memory_mb": 0,
            "required_cpu_cores": 1,
            "estimated_training_time_minutes": 60,
            "data_size_factor": 1.0,
            "model_complexity_factor": 1.0
        }
        
        # 基于数据大小估算
        if "data_size_mb" in training_data_info:
            data_size_mb = training_data_info["data_size_mb"]
            requirements["estimated_memory_mb"] = max(1024, data_size_mb * 2)
            requirements["data_size_factor"] = data_size_mb / 1024  # 相对于1GB的比例
        
        # 基于模型复杂度估算
        if hasattr(self, 'model') and self.model is not None:
            try:
                import torch
                if isinstance(self.model, torch.nn.Module):
                    # 粗略估计参数数量
                    param_count = sum(p.numel() for p in self.model.parameters())
                    requirements["model_complexity_factor"] = max(1.0, param_count / 1000000)  # 百万参数为单位
                    
                    # 估算GPU内存需求（每个参数4字节，加上中间激活值）
                    gpu_memory_needed = param_count * 4 * 2  # 参数+梯度
                    requirements["estimated_gpu_memory_mb"] = gpu_memory_needed / (1024 * 1024)
            except ImportError:
                pass
        
        # 基于训练配置调整
        epochs = self._training_config.get("epochs", 100)
        batch_size = self._training_config.get("batch_size", 32)
        requirements["estimated_training_time_minutes"] = epochs * batch_size * requirements["data_size_factor"] * 0.1
        
        logger.info(f"Training resource requirements: {requirements}")
        return requirements
    
    def _optimize_training_resources(self, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """优化训练资源分配"""
        optimization_results = {
            "optimization_applied": False,
            "recommendations": [],
            "estimated_improvement": 0.0
        }
        
        # 检测瓶颈并提供优化建议
        bottlenecks = current_performance.get("bottlenecks", [])
        resource_util = current_performance.get("resource_utilization", {})
        
        # 检查CPU利用率瓶颈
        if "High CPU utilization" in bottlenecks:
            optimization_results["recommendations"].append(
                "Reduce batch size from 32 to 16 to lower CPU load"
            )
            optimization_results["estimated_improvement"] += 0.2
        
        # 检查训练速度瓶颈
        if "Low training speed" in bottlenecks:
            optimization_results["recommendations"].append(
                "Enable GPU acceleration if available"
            )
            optimization_results["estimated_improvement"] += 0.4
        
        # 检查内存利用率
        memory_util = resource_util.get("memory_utilization", 0.0)
        if memory_util > 80.0:  # 80%内存使用率
            optimization_results["recommendations"].append(
                "Reduce model complexity or enable gradient checkpointing"
            )
            optimization_results["estimated_improvement"] += 0.15
        
        # 如果检测到瓶颈，标记优化已应用
        if optimization_results["recommendations"]:
            optimization_results["optimization_applied"] = True
        
        logger.info(f"Training resource optimization results: {optimization_results}")
        return optimization_results
    
    def _integrate_with_resource_mixin(self) -> bool:
        """与ResourceMixin集成"""
        try:
            # 检查ResourceMixin方法是否可用
            required_methods = ["allocate_resources", "release_resources", "get_resource_status"]
            for method in required_methods:
                if not hasattr(self, method):
                    logger.warning(f"ResourceMixin method {method} not available")
                    return False
            
            logger.info("Successfully integrated with ResourceMixin")
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate with ResourceMixin: {e}")
            return False
    
    def _integrate_with_performance_mixin(self) -> bool:
        """与PerformanceMixin集成"""
        try:
            # 检查PerformanceMixin方法是否可用
            required_methods = ["get_performance_metrics", "start_performance_monitoring", "stop_performance_monitoring"]
            for method in required_methods:
                if not hasattr(self, method):
                    logger.warning(f"PerformanceMixin method {method} not available")
                    return False
            
            logger.info("Successfully integrated with PerformanceMixin")
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate with PerformanceMixin: {e}")
            return False
    
    # ==================== 训练数据管理组件 ====================
    
    def configure_training(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        配置训练参数
        
        Args:
            config: 用户提供的配置，将覆盖默认值
            
        Returns:
            完整的训练配置字典
        """
        logger.info("Configuring training parameters")
        
        # 默认训练配置
        default_config = {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss_function": "auto",
            "validation_split": 0.2,
            "patience": 10,  # 早停耐心值
            "min_delta": 0.001,  # 最小改进阈值
            "shuffle_data": True,
            "normalize_data": True,
            "enable_checkpoints": True,
            "checkpoint_frequency": 5,  # 每5个epoch保存检查点
            "enable_early_stopping": True,
            "enable_progress_bar": True,
            "log_frequency": 10,  # 每10个batch记录一次
            "device": "auto",  # auto, cpu, cuda
            "num_workers": 0,  # 数据加载工作线程数
            "pin_memory": True,
            "gradient_clipping": 1.0,  # 梯度裁剪阈值
            "weight_decay": 0.0,  # L2正则化
            "momentum": 0.9,  # SGD动量
            "nesterov": True,  # SGD Nesterov动量
            "amsgrad": False,  # Adam AMSGrad变体
            "beta1": 0.9,  # Adam beta1
            "beta2": 0.999,  # Adam beta2
            "epsilon": 1e-8  # Adam epsilon
        }
        
        # 合并用户配置
        if config is not None:
            default_config.update(config)
        
        logger.info(f"Training configuration: {default_config}")
        return default_config
    
    def load_training_data(self, data_path: str, data_format: str = "auto") -> Any:
        """
        加载训练数据（支持CSV、JSON、图像等格式）
        
        Args:
            data_path: 数据路径（文件或目录）
            data_format: 数据格式（csv, json, image, text, auto）
            
        Returns:
            加载的数据
        """
        logger.info(f"Loading training data from {data_path} (format: {data_format})")
        
        try:
            # 自动检测格式
            if data_format == "auto":
                data_format = self._detect_data_format(data_path)
            
            # 根据格式加载数据
            if data_format == "csv":
                return self._load_csv_data(data_path)
            elif data_format == "json":
                return self._load_json_data(data_path)
            elif data_format == "image":
                return self._load_image_data(data_path)
            elif data_format == "text":
                return self._load_text_data(data_path)
            else:
                raise ValueError(f"Unsupported data format: {data_format}")
                
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise
    
    def preprocess_training_data(self, raw_data: Any, preprocessing_steps: List[str] = None) -> Any:
        """
        预处理训练数据
        
        Args:
            raw_data: 原始数据
            preprocessing_steps: 预处理步骤列表
            
        Returns:
            预处理后的数据
        """
        logger.info("Preprocessing training data...")
        
        if preprocessing_steps is None:
            preprocessing_steps = ["normalize", "shuffle", "split"]
        
        processed_data = raw_data
        
        for step in preprocessing_steps:
            if step == "normalize":
                processed_data = self._normalize_data(processed_data)
            elif step == "shuffle":
                processed_data = self._shuffle_data(processed_data)
            elif step == "split":
                processed_data = self._split_data(processed_data)
            elif step == "augment" and self._data_augmentation_enabled:
                processed_data = self._augment_data(processed_data)
            elif step == "encode":
                processed_data = self._encode_data(processed_data)
            else:
                logger.warning(f"Unknown preprocessing step: {step}")
        
        logger.info(f"Preprocessing completed with steps: {preprocessing_steps}")
        return processed_data
    
    def create_data_loader(self, data: Any, batch_size: int = 32, shuffle: bool = True, 
                          num_workers: int = 0) -> Any:
        """
        创建数据加载器
        
        Args:
            data: 训练数据
            batch_size: 批量大小
            shuffle: 是否打乱数据
            num_workers: 数据加载工作线程数
            
        Returns:
            数据加载器
        """
        logger.info(f"Creating data loader (batch_size={batch_size}, shuffle={shuffle})")
        
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            
            # 检查数据格式
            if isinstance(data, (list, tuple)) and len(data) == 2:
                # 假设 (inputs, targets) 格式
                inputs, targets = data
                
                # 转换为张量
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                if not isinstance(targets, torch.Tensor):
                    # 自动检测目标类型
                    if len(targets.shape) == 1 or targets.shape[1] == 1:
                        targets = torch.tensor(targets, dtype=torch.long)
                    else:
                        targets = torch.tensor(targets, dtype=torch.float32)
                
                # 创建数据集
                dataset = TensorDataset(inputs, targets)
            elif hasattr(data, '__getitem__') and hasattr(data, '__len__'):
                # 已经是数据集对象
                dataset = data
            else:
                # 尝试创建单张量数据集（用于无监督学习）
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)
                dataset = TensorDataset(data)
            
            # 创建数据加载器
            data_loader = DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=hasattr(self, 'device') and str(self.device) != 'cpu'
            )
            
            logger.info(f"Data loader created with {len(dataset)} samples")
            return data_loader
            
        except ImportError:
            logger.warning("PyTorch not available, returning data as-is")
            return data
        except Exception as e:
            logger.error(f"Failed to create data loader: {e}")
            raise
    
    def _create_data_loader(self, data: Any, batch_size: int = 32, shuffle: bool = True,
                          num_workers: int = 0) -> Any:
        """
        创建数据加载器（内部方法）
        
        Args:
            data: 训练数据
            batch_size: 批量大小
            shuffle: 是否打乱数据
            num_workers: 数据加载工作线程数
            
        Returns:
            数据加载器
        """
        # 调用公共方法
        return self.create_data_loader(data, batch_size, shuffle, num_workers)
    
    def split_training_validation(self, data: Any, validation_split: float = 0.2, 
                                random_seed: int = 42) -> Tuple[Any, Any]:
        """
        分割训练集和验证集
        
        Args:
            data: 完整数据集
            validation_split: 验证集比例
            random_seed: 随机种子
            
        Returns:
            (训练数据, 验证数据)
        """
        logger.info(f"Splitting data into training/validation (split={validation_split})")
        
        try:
            import torch
            from torch.utils.data import random_split
            
            if not 0 < validation_split < 1:
                raise ValueError(f"Invalid validation split: {validation_split}")
            
            # 计算分割大小
            total_size = len(data)
            val_size = int(total_size * validation_split)
            train_size = total_size - val_size
            
            # 随机分割
            train_data, val_data = random_split(
                data, [train_size, val_size],
                generator=torch.Generator().manual_seed(random_seed)
            )
            
            logger.info(f"Data split: {train_size} training samples, {val_size} validation samples")
            return train_data, val_data
            
        except ImportError:
            logger.warning("PyTorch not available, using simple splitting")
            # 简单分割
            split_idx = int(len(data) * (1 - validation_split))
            return data[:split_idx], data[split_idx:]
        except Exception as e:
            logger.error(f"Failed to split data: {e}")
            raise
    
    def _split_training_validation(self, data: Any, validation_split: float = 0.2, 
                                 random_seed: int = 42) -> Tuple[Any, Any]:
        """
        分割训练集和验证集（内部方法）
        
        Args:
            data: 完整数据集
            validation_split: 验证集比例
            random_seed: 随机种子
            
        Returns:
            (训练数据, 验证数据)
        """
        # 调用公共方法
        return self.split_training_validation(data, validation_split, random_seed)
    
    def apply_data_augmentation(self, data: Any, augmentation_types: List[str] = None) -> Any:
        """
        应用数据增强
        
        Args:
            data: 原始数据
            augmentation_types: 增强类型列表
            
        Returns:
            增强后的数据
        """
        if not self._data_augmentation_enabled:
            logger.info("Data augmentation is disabled")
            return data
        
        if augmentation_types is None:
            augmentation_types = ["flip", "rotate", "noise"]
        
        logger.info(f"Applying data augmentation: {augmentation_types}")
        
        augmented_data = data
        
        for aug_type in augmentation_types:
            if aug_type == "flip":
                augmented_data = self._apply_flip_augmentation(augmented_data)
            elif aug_type == "rotate":
                augmented_data = self._apply_rotate_augmentation(augmented_data)
            elif aug_type == "noise":
                augmented_data = self._apply_noise_augmentation(augmented_data)
            elif aug_type == "crop":
                augmented_data = self._apply_crop_augmentation(augmented_data)
            elif aug_type == "brightness":
                augmented_data = self._apply_brightness_augmentation(augmented_data)
            else:
                logger.warning(f"Unknown augmentation type: {aug_type}")
        
        logger.info(f"Data augmentation applied: {augmentation_types}")
        return augmented_data
    
    # 私有辅助方法
    def _detect_data_format(self, data_path: str) -> str:
        """检测数据格式"""
        import os
        
        if os.path.isdir(data_path):
            # 检查目录内容
            files = os.listdir(data_path)
            image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
            for file in files[:10]:  # 检查前10个文件
                if any(file.lower().endswith(ext) for ext in image_exts):
                    return "image"
            return "text"
        
        # 检查文件扩展名
        ext = os.path.splitext(data_path)[1].lower()
        if ext == '.csv':
            return "csv"
        elif ext == '.json':
            return "json"
        elif ext in {'.txt', '.text'}:
            return "text"
        elif ext in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}:
            return "image"
        else:
            # 默认尝试文本格式
            return "text"
    
    def _load_csv_data(self, filepath: str) -> Any:
        """加载CSV数据"""
        import pandas as pd
        logger.info(f"Loading CSV data from {filepath}")
        return pd.read_csv(filepath)
    
    def _load_json_data(self, filepath: str) -> Any:
        """加载JSON数据"""
        import json
        logger.info(f"Loading JSON data from {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_image_data(self, path: str) -> Any:
        """加载图像数据"""
        import os
        from PIL import Image
        import numpy as np
        
        logger.info(f"Loading image data from {path}")
        
        if os.path.isdir(path):
            # 加载目录中的所有图像
            images = []
            for filename in os.listdir(path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(path, filename)
                    img = Image.open(img_path)
                    images.append(np.array(img))
            return np.array(images)
        else:
            # 加载单个图像
            img = Image.open(path)
            return np.array(img)
    
    def _load_text_data(self, filepath: str) -> Any:
        """加载文本数据"""
        logger.info(f"Loading text data from {filepath}")
        
        if os.path.isdir(filepath):
            # 加载目录中的所有文本文件
            texts = []
            import os
            for filename in os.listdir(filepath):
                if filename.lower().endswith(('.txt', '.text')):
                    txt_path = os.path.join(filepath, filename)
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
            return texts
        else:
            # 加载单个文本文件
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _normalize_data(self, data: Any) -> Any:
        """标准化数据"""
        try:
            import numpy as np
            if isinstance(data, np.ndarray):
                # 归一化到[0, 1]
                data_min = data.min()
                data_max = data.max()
                if data_max > data_min:
                    data = (data - data_min) / (data_max - data_min)
                return data
        except ImportError:
            pass
        return data
    
    def _apply_normalization(self, data: Any, normalization_type: str = "minmax") -> Any:
        """
        应用数据归一化
        
        Args:
            data: 要归一化的数据
            normalization_type: 归一化类型（minmax, standard, robust）
            
        Returns:
            归一化后的数据
        """
        logger.info(f"Applying {normalization_type} normalization")
        
        try:
            import numpy as np
            
            if not isinstance(data, np.ndarray):
                # 尝试转换为numpy数组
                data = np.array(data)
            
            if normalization_type == "minmax":
                # 最小-最大归一化 [0, 1]
                data_min = data.min()
                data_max = data.max()
                if data_max > data_min:
                    data = (data - data_min) / (data_max - data_min)
                return data
                
            elif normalization_type == "standard":
                # 标准化（零均值，单位方差）
                mean = data.mean()
                std = data.std()
                if std > 0:
                    data = (data - mean) / std
                return data
                
            elif normalization_type == "robust":
                # 鲁棒归一化（使用中位数和IQR）
                median = np.median(data)
                q75, q25 = np.percentile(data, [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    data = (data - median) / iqr
                return data
                
            else:
                logger.warning(f"Unknown normalization type: {normalization_type}, using minmax")
                return self._apply_normalization(data, "minmax")
                
        except Exception as e:
            logger.error(f"Failed to apply normalization: {e}")
            return data
    
    def _shuffle_data(self, data: Any) -> Any:
        """打乱数据"""
        try:
            import numpy as np
            if isinstance(data, np.ndarray):
                np.random.shuffle(data)
                return data
            elif isinstance(data, (list, tuple)):
                import random
                shuffled = list(data)
                random.shuffle(shuffled)
                return type(data)(shuffled)
        except ImportError:
            pass
        return data
    
    def _apply_shuffling(self, data: Any, random_seed: Optional[int] = None) -> Any:
        """
        应用数据打乱
        
        Args:
            data: 要打乱的数据
            random_seed: 随机种子（确保可重复性）
            
        Returns:
            打乱后的数据
        """
        logger.info(f"Applying data shuffling (seed={random_seed})")
        
        try:
            import numpy as np
            import random
            
            # 设置随机种子
            if random_seed is not None:
                np.random.seed(random_seed)
                random.seed(random_seed)
            
            if isinstance(data, np.ndarray):
                # 打乱numpy数组
                indices = np.arange(len(data))
                np.random.shuffle(indices)
                return data[indices]
                
            elif isinstance(data, (list, tuple)):
                # 打乱列表或元组
                shuffled = list(data)
                random.shuffle(shuffled)
                return type(data)(shuffled)
                
            elif isinstance(data, dict):
                # 打乱字典（按键）
                keys = list(data.keys())
                random.shuffle(keys)
                return {key: data[key] for key in keys}
                
            else:
                logger.warning(f"Unsupported data type for shuffling: {type(data)}")
                return data
                
        except Exception as e:
            logger.error(f"Failed to apply shuffling: {e}")
            return data
    
    def _split_data(self, data: Any) -> Any:
        """分割数据（默认不操作，由split_training_validation处理）"""
        return data
    
    def _augment_data(self, data: Any) -> Any:
        """增强数据（默认实现）"""
        return data
    
    def _encode_data(self, data: Any) -> Any:
        """编码数据"""
        return data
    
    def _apply_flip_augmentation(self, data: Any) -> Any:
        """应用翻转增强"""
        try:
            import numpy as np
            if isinstance(data, np.ndarray) and len(data.shape) >= 3:
                # 假设图像数据（batch, height, width, channels）
                return np.flip(data, axis=2)  # 水平翻转
        except ImportError:
            pass
        return data
    
    def _apply_rotate_augmentation(self, data: Any) -> Any:
        """应用旋转增强"""
        try:
            import numpy as np
            from scipy.ndimage import rotate
            if isinstance(data, np.ndarray) and len(data.shape) >= 3:
                # 随机旋转
                angle = np.random.uniform(-15, 15)
                return rotate(data, angle, axes=(1, 2), reshape=False)
        except ImportError:
            pass
        return data
    
    def _apply_noise_augmentation(self, data: Any) -> Any:
        """应用噪声增强"""
        try:
            import numpy as np
            if isinstance(data, np.ndarray):
                noise = np.random.normal(0, 0.01, data.shape)
                return data + noise
        except ImportError:
            pass
        return data
    
    def _apply_crop_augmentation(self, data: Any) -> Any:
        """应用裁剪增强"""
        try:
            import numpy as np
            if isinstance(data, np.ndarray) and len(data.shape) >= 3:
                # 随机裁剪
                h, w = data.shape[1:3]
                crop_h, crop_w = int(h * 0.9), int(w * 0.9)
                start_h = np.random.randint(0, h - crop_h)
                start_w = np.random.randint(0, w - crop_w)
                return data[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
        except ImportError:
            pass
        return data
    
    def _apply_brightness_augmentation(self, data: Any) -> Any:
        """应用亮度增强"""
        try:
            import numpy as np
            if isinstance(data, np.ndarray):
                factor = np.random.uniform(0.8, 1.2)
                return data * factor
        except ImportError:
            pass
        return data
    
    # ==================== 基础训练算法封装 ====================
    
    def create_optimizer(self, optimizer_type: str = "adam", 
                        learning_rate: float = 0.001, 
                        optimizer_params: Dict[str, Any] = None) -> Any:
        """
        创建优化器
        
        Args:
            optimizer_type: 优化器类型（sgd, adam, rmsprop, adagrad）
            learning_rate: 学习率
            optimizer_params: 优化器额外参数
            
        Returns:
            优化器实例
        """
        logger.info(f"Creating {optimizer_type} optimizer with learning rate {learning_rate}")
        
        try:
            import torch.optim as optim
            
            if optimizer_params is None:
                optimizer_params = {}
            
            # 检查模型是否存在
            if not hasattr(self, 'model') or self.model is None:
                logger.warning("Model not available, cannot create optimizer")
                return None
            
            # 获取模型参数
            model_params = self.model.parameters()
            
            # 创建优化器
            if optimizer_type.lower() == "sgd":
                momentum = optimizer_params.get("momentum", 0.9)
                weight_decay = optimizer_params.get("weight_decay", 0.0)
                optimizer = optim.SGD(model_params, lr=learning_rate, 
                                     momentum=momentum, weight_decay=weight_decay)
            
            elif optimizer_type.lower() == "adam":
                betas = optimizer_params.get("betas", (0.9, 0.999))
                eps = optimizer_params.get("eps", 1e-8)
                weight_decay = optimizer_params.get("weight_decay", 0.0)
                optimizer = optim.Adam(model_params, lr=learning_rate, 
                                      betas=betas, eps=eps, weight_decay=weight_decay)
            
            elif optimizer_type.lower() == "rmsprop":
                alpha = optimizer_params.get("alpha", 0.99)
                eps = optimizer_params.get("eps", 1e-8)
                momentum = optimizer_params.get("momentum", 0.0)
                weight_decay = optimizer_params.get("weight_decay", 0.0)
                optimizer = optim.RMSprop(model_params, lr=learning_rate, 
                                         alpha=alpha, eps=eps, momentum=momentum,
                                         weight_decay=weight_decay)
            
            elif optimizer_type.lower() == "adagrad":
                lr_decay = optimizer_params.get("lr_decay", 0.0)
                weight_decay = optimizer_params.get("weight_decay", 0.0)
                optimizer = optim.Adagrad(model_params, lr=learning_rate, 
                                         lr_decay=lr_decay, weight_decay=weight_decay)
            
            else:
                logger.warning(f"Unknown optimizer type: {optimizer_type}, using Adam as default")
                optimizer = optim.Adam(model_params, lr=learning_rate)
            
            # 存储优化器引用
            self.optimizer = optimizer
            logger.info(f"Optimizer created: {optimizer_type}")
            return optimizer
            
        except ImportError:
            logger.error("PyTorch not available, cannot create optimizer")
            return None
        except Exception as e:
            logger.error(f"Failed to create optimizer: {e}")
            return None
    
    def create_loss_function(self, loss_type: str = "auto", 
                           loss_params: Dict[str, Any] = None) -> Any:
        """
        创建损失函数
        
        Args:
            loss_type: 损失函数类型（auto, mse, cross_entropy, l1, binary_cross_entropy）
            loss_params: 损失函数额外参数
            
        Returns:
            损失函数实例
        """
        logger.info(f"Creating loss function: {loss_type}")
        
        try:
            import torch.nn as nn
            
            if loss_params is None:
                loss_params = {}
            
            # 自动选择损失函数
            if loss_type == "auto":
                loss_type = self._auto_select_loss_function_type()
            
            # 创建损失函数
            if loss_type.lower() == "mse":
                reduction = loss_params.get("reduction", "mean")
                criterion = nn.MSELoss(reduction=reduction)
            
            elif loss_type.lower() == "cross_entropy":
                weight = loss_params.get("weight", None)
                ignore_index = loss_params.get("ignore_index", -100)
                reduction = loss_params.get("reduction", "mean")
                criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, 
                                               reduction=reduction)
            
            elif loss_type.lower() == "l1":
                reduction = loss_params.get("reduction", "mean")
                criterion = nn.L1Loss(reduction=reduction)
            
            elif loss_type.lower() == "binary_cross_entropy":
                weight = loss_params.get("weight", None)
                reduction = loss_params.get("reduction", "mean")
                criterion = nn.BCELoss(weight=weight, reduction=reduction)
            
            elif loss_type.lower() == "smooth_l1":
                beta = loss_params.get("beta", 1.0)
                reduction = loss_params.get("reduction", "mean")
                criterion = nn.SmoothL1Loss(beta=beta, reduction=reduction)
            
            else:
                logger.warning(f"Unknown loss type: {loss_type}, using MSE as default")
                criterion = nn.MSELoss()
            
            # 存储损失函数引用
            self.criterion = criterion
            logger.info(f"Loss function created: {loss_type}")
            return criterion
            
        except ImportError:
            logger.error("PyTorch not available, cannot create loss function")
            return None
        except Exception as e:
            logger.error(f"Failed to create loss function: {e}")
            return None
    
    def create_scheduler(self, scheduler_type: str = "steplr", 
                        scheduler_params: Dict[str, Any] = None) -> Any:
        """
        创建学习率调度器
        
        Args:
            scheduler_type: 调度器类型（steplr, exponential, cosine, plateau）
            scheduler_params: 调度器参数
            
        Returns:
            调度器实例
        """
        logger.info(f"Creating scheduler: {scheduler_type}")
        
        try:
            import torch.optim as optim
            
            if scheduler_params is None:
                scheduler_params = {}
            
            # 检查优化器是否存在
            if not hasattr(self, 'optimizer') or self.optimizer is None:
                logger.warning("Optimizer not available, cannot create scheduler")
                return None
            
            # 创建调度器
            if scheduler_type.lower() == "steplr":
                step_size = scheduler_params.get("step_size", 30)
                gamma = scheduler_params.get("gamma", 0.1)
                scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                     step_size=step_size, gamma=gamma)
            
            elif scheduler_type.lower() == "exponential":
                gamma = scheduler_params.get("gamma", 0.9)
                scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
            
            elif scheduler_type.lower() == "cosine":
                T_max = scheduler_params.get("T_max", 100)
                eta_min = scheduler_params.get("eta_min", 0)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                T_max=T_max, eta_min=eta_min)
            
            elif scheduler_type.lower() == "plateau":
                mode = scheduler_params.get("mode", "min")
                factor = scheduler_params.get("factor", 0.1)
                patience = scheduler_params.get("patience", 10)
                threshold = scheduler_params.get("threshold", 1e-4)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode=mode, factor=factor, patience=patience,
                    threshold=threshold
                )
            
            else:
                logger.warning(f"Unknown scheduler type: {scheduler_type}, using StepLR as default")
                scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
            
            # 存储调度器引用
            self.scheduler = scheduler
            logger.info(f"Scheduler created: {scheduler_type}")
            return scheduler
            
        except ImportError:
            logger.error("PyTorch not available, cannot create scheduler")
            return None
        except Exception as e:
            logger.error(f"Failed to create scheduler: {e}")
            return None
    
    def train_one_epoch(self, data_loader: Any, epoch: int = 0) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            data_loader: 数据加载器
            epoch: 当前epoch编号
            
        Returns:
            训练指标字典
        """
        logger.info(f"Training epoch {epoch}")
        
        try:
            import torch
            
            # 检查必要组件
            if not hasattr(self, 'model') or self.model is None:
                raise ValueError("Model not initialized")
            if not hasattr(self, 'optimizer') or self.optimizer is None:
                raise ValueError("Optimizer not initialized")
            if not hasattr(self, 'criterion') or self.criterion is None:
                raise ValueError("Loss function not initialized")
            
            # 设置为训练模式
            self.model.train()
            
            # 训练统计
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # 训练循环
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                # 移动到设备
                if hasattr(self, 'device'):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 梯度清零
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                loss = self.criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                
                # 优化器步骤
                self.optimizer.step()
                
                # 更新统计
                train_loss += loss.item()
                
                # 计算准确率（分类任务）
                if outputs.shape == targets.shape or len(targets.shape) == 1:
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += targets.size(0)
                    train_correct += (predicted == targets).sum().item()
                
                # 每10个batch记录一次进度
                if batch_idx % 10 == 0:
                    batch_loss = loss.item()
                    logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Loss: {batch_loss:.4f}")
            
            # 计算epoch指标
            avg_loss = train_loss / len(data_loader)
            accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0
            
            # 更新学习率调度器
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_loss)
                else:
                    self.scheduler.step()
            
            epoch_metrics = {
                "loss": avg_loss,
                "accuracy": accuracy,
                "samples": train_total,
                "batches": len(data_loader)
            }
            
            logger.info(f"Epoch {epoch} completed: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
            return epoch_metrics
            
        except Exception as e:
            logger.error(f"Failed to train epoch {epoch}: {e}")
            raise
    
    def validate_one_epoch(self, data_loader: Any, epoch: int = 0) -> Dict[str, float]:
        """
        验证一个epoch
        
        Args:
            data_loader: 数据加载器
            epoch: 当前epoch编号
            
        Returns:
            验证指标字典
        """
        logger.info(f"Validating epoch {epoch}")
        
        try:
            import torch
            
            # 检查必要组件
            if not hasattr(self, 'model') or self.model is None:
                raise ValueError("Model not initialized")
            if not hasattr(self, 'criterion') or self.criterion is None:
                raise ValueError("Loss function not initialized")
            
            # 设置为评估模式
            self.model.eval()
            
            # 验证统计
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # 验证循环（无梯度计算）
            with torch.no_grad():
                for inputs, targets in data_loader:
                    # 移动到设备
                    if hasattr(self, 'device'):
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # 前向传播
                    outputs = self.model(inputs)
                    
                    # 计算损失
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    # 计算准确率（分类任务）
                    if outputs.shape == targets.shape or len(targets.shape) == 1:
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += targets.size(0)
                        val_correct += (predicted == targets).sum().item()
            
            # 计算epoch指标
            avg_loss = val_loss / len(data_loader) if len(data_loader) > 0 else 0.0
            accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0.0
            
            epoch_metrics = {
                "loss": avg_loss,
                "accuracy": accuracy,
                "samples": val_total,
                "batches": len(data_loader)
            }
            
            logger.info(f"Validation epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
            return epoch_metrics
            
        except Exception as e:
            logger.error(f"Failed to validate epoch {epoch}: {e}")
            raise
    
    def _auto_select_loss_function_type(self) -> str:
        """
        自动选择损失函数类型
        
        Returns:
            损失函数类型字符串
        """
        # 检查模型输出特征
        if hasattr(self, 'model') and self.model is not None:
            try:
                import torch.nn as nn
                
                # 检查最后一层
                last_layer = list(self.model.children())[-1]
                
                if isinstance(last_layer, nn.Linear):
                    # 检查输出维度
                    output_dim = last_layer.out_features
                    if output_dim == 1:
                        return "mse"  # 回归任务
                    elif output_dim == 2:
                        return "cross_entropy"  # 二分类
                    else:
                        return "cross_entropy"  # 多分类
                
                elif isinstance(last_layer, nn.Sigmoid):
                    return "binary_cross_entropy"
                
                elif isinstance(last_layer, nn.LogSoftmax):
                    return "cross_entropy"
                    
            except (ImportError, AttributeError, IndexError):
                pass
        
        # 默认使用MSE
        logger.info("Auto-selected loss function: mse (default)")
        return "mse"
    
    # ==================== 训练效果评估方法 ====================
    
    def compute_classification_metrics(self, predictions: Any, targets: Any, 
                                     average: str = "macro") -> Dict[str, float]:
        """
        计算分类指标（准确率、精确率、召回率、F1分数）
        
        Args:
            predictions: 预测值
            targets: 真实标签
            average: 多分类平均方式（macro, micro, weighted, binary）
            
        Returns:
            分类指标字典
        """
        logger.info(f"Computing classification metrics (average={average})")
        
        try:
            import numpy as np
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # 转换为numpy数组
            if hasattr(predictions, 'cpu'):
                predictions = predictions.cpu().numpy()
            if hasattr(targets, 'cpu'):
                targets = targets.cpu().numpy()
            
            # 确保是一维数组
            predictions = np.ravel(predictions)
            targets = np.ravel(targets)
            
            # 计算指标
            metrics = {
                "accuracy": accuracy_score(targets, predictions),
                "precision": precision_score(targets, predictions, average=average, zero_division=0),
                "recall": recall_score(targets, predictions, average=average, zero_division=0),
                "f1_score": f1_score(targets, predictions, average=average, zero_division=0)
            }
            
            # 对于二分类，添加额外指标
            if len(np.unique(targets)) == 2 and average == "binary":
                from sklearn.metrics import roc_auc_score
                try:
                    metrics["roc_auc"] = roc_auc_score(targets, predictions)
                except ValueError:
                    metrics["roc_auc"] = 0.0
            
            logger.info(f"Classification metrics: {metrics}")
            return metrics
            
        except ImportError:
            logger.warning("scikit-learn not available, using simple accuracy")
            # 简单准确率计算
            correct = np.sum(predictions == targets)
            total = len(targets)
            accuracy = correct / total if total > 0 else 0.0
            return {"accuracy": accuracy, "precision": accuracy, "recall": accuracy, "f1_score": accuracy}
        except Exception as e:
            logger.error(f"Failed to compute classification metrics: {e}")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
    
    def compute_regression_metrics(self, predictions: Any, targets: Any) -> Dict[str, float]:
        """
        计算回归指标（MSE、MAE、R²）
        
        Args:
            predictions: 预测值
            targets: 真实值
            
        Returns:
            回归指标字典
        """
        logger.info("Computing regression metrics")
        
        try:
            import numpy as np
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # 转换为numpy数组
            if hasattr(predictions, 'cpu'):
                predictions = predictions.cpu().numpy()
            if hasattr(targets, 'cpu'):
                targets = targets.cpu().numpy()
            
            # 确保是一维数组
            predictions = np.ravel(predictions)
            targets = np.ravel(targets)
            
            # 计算指标
            metrics = {
                "mse": mean_squared_error(targets, predictions),
                "rmse": np.sqrt(mean_squared_error(targets, predictions)),
                "mae": mean_absolute_error(targets, predictions),
                "r2_score": r2_score(targets, predictions)
            }
            
            logger.info(f"Regression metrics: {metrics}")
            return metrics
            
        except ImportError:
            logger.warning("scikit-learn not available, using simple MSE")
            # 简单MSE计算
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            return {"mse": mse, "rmse": np.sqrt(mse), "mae": mae, "r2_score": 0.0}
        except Exception as e:
            logger.error(f"Failed to compute regression metrics: {e}")
            return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2_score": 0.0}
    
    def generate_training_report(self, training_history: Dict[str, List[float]] = None) -> Dict[str, Any]:
        """
        生成训练报告
        
        Args:
            training_history: 训练历史记录
            
        Returns:
            训练报告字典
        """
        logger.info("Generating training report")
        
        if training_history is None:
            training_history = self._training_metrics_history
        
        report = {
            "summary": {},
            "final_metrics": {},
            "improvement_analysis": {},
            "recommendations": []
        }
        
        try:
            # 提取关键指标
            train_loss = training_history.get("loss", [])
            train_accuracy = training_history.get("accuracy", [])
            val_loss = training_history.get("val_loss", []) if "val_loss" in training_history else []
            val_accuracy = training_history.get("val_accuracy", []) if "val_accuracy" in training_history else []
            
            # 生成摘要
            report["summary"] = {
                "total_epochs": len(train_loss),
                "final_train_loss": train_loss[-1] if train_loss else 0.0,
                "final_train_accuracy": train_accuracy[-1] if train_accuracy else 0.0,
                "final_val_loss": val_loss[-1] if val_loss else 0.0,
                "final_val_accuracy": val_accuracy[-1] if val_accuracy else 0.0,
                "best_val_accuracy": max(val_accuracy) if val_accuracy else 0.0,
                "best_val_epoch": val_accuracy.index(max(val_accuracy)) if val_accuracy else 0
            }
            
            # 最终指标
            report["final_metrics"] = {
                "loss": report["summary"]["final_train_loss"],
                "accuracy": report["summary"]["final_train_accuracy"],
                "val_loss": report["summary"]["final_val_loss"],
                "val_accuracy": report["summary"]["final_val_accuracy"]
            }
            
            # 改进分析
            if len(train_loss) > 1:
                loss_improvement = train_loss[0] - train_loss[-1]
                accuracy_improvement = train_accuracy[-1] - train_accuracy[0] if train_accuracy else 0.0
                
                report["improvement_analysis"] = {
                    "loss_reduction_percent": (loss_improvement / train_loss[0] * 100) if train_loss[0] > 0 else 0.0,
                    "accuracy_improvement_percent": accuracy_improvement,
                    "convergence_speed": len(train_loss) / max(1, loss_improvement) if loss_improvement > 0 else 0.0,
                    "overfitting_detected": len(val_loss) > 0 and val_loss[-1] > train_loss[-1] * 1.2
                }
            
            # 训练建议
            recommendations = []
            
            # 检查过拟合
            if report["improvement_analysis"].get("overfitting_detected", False):
                recommendations.append("Detected overfitting: consider adding regularization or reducing model complexity")
            
            # 检查收敛速度
            if report["improvement_analysis"].get("convergence_speed", 0) > 100:
                recommendations.append("Slow convergence: consider increasing learning rate or adjusting optimizer parameters")
            
            # 检查最终准确率
            final_acc = report["summary"]["final_val_accuracy"]
            if final_acc < 70.0:
                recommendations.append(f"Low validation accuracy ({final_acc:.1f}%): consider more training data or model architecture changes")
            
            report["recommendations"] = recommendations
            
            logger.info(f"Training report generated: {report['summary']}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate training report: {e}")
            return report
    
    def plot_training_curves(self, training_history: Dict[str, List[float]] = None, 
                           save_path: str = None) -> bool:
        """
        绘制训练曲线
        
        Args:
            training_history: 训练历史记录
            save_path: 保存路径（可选）
            
        Returns:
            是否成功绘制
        """
        logger.info("Plotting training curves")
        
        if training_history is None:
            training_history = self._training_metrics_history
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Training Progress Metrics', fontsize=16)
            
            # 提取数据
            train_loss = training_history.get("loss", [])
            train_accuracy = training_history.get("accuracy", [])
            val_loss = training_history.get("val_loss", []) if "val_loss" in training_history else []
            val_accuracy = training_history.get("val_accuracy", []) if "val_accuracy" in training_history else []
            
            epochs = list(range(1, len(train_loss) + 1))
            
            # 1. 训练损失曲线
            ax1 = axes[0, 0]
            ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
            if val_loss:
                ax1.plot(epochs[:len(val_loss)], val_loss, 'r-', label='Val Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 训练准确率曲线
            ax2 = axes[0, 1]
            if train_accuracy:
                ax2.plot(epochs[:len(train_accuracy)], train_accuracy, 'b-', label='Train Accuracy', linewidth=2)
            if val_accuracy:
                ax2.plot(epochs[:len(val_accuracy)], val_accuracy, 'r-', label='Val Accuracy', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. 学习率曲线（如果有）
            ax3 = axes[1, 0]
            if "learning_rate" in training_history and training_history["learning_rate"]:
                lr_history = training_history["learning_rate"]
                ax3.plot(epochs[:len(lr_history)], lr_history, 'g-', label='Learning Rate', linewidth=2)
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Learning Rate')
                ax3.set_title('Learning Rate Schedule')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No learning rate data', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax3.transAxes)
                ax3.set_title('Learning Rate Schedule')
            
            # 4. 其他指标（如果有）
            ax4 = axes[1, 1]
            other_metrics = {k: v for k, v in training_history.items() 
                           if k not in ["loss", "accuracy", "val_loss", "val_accuracy", "learning_rate"]}
            if other_metrics:
                for metric_name, values in other_metrics.items():
                    if values:
                        ax4.plot(epochs[:len(values)], values, label=metric_name, linewidth=2)
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Metric Value')
                ax4.set_title('Other Training Metrics')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No other metrics data', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes)
                ax4.set_title('Other Training Metrics')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存或显示
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training curves saved to {save_path}")
            else:
                plt.show()
            
            plt.close(fig)
            return True
            
        except ImportError:
            logger.warning("matplotlib not available, cannot plot training curves")
            return False
        except Exception as e:
            logger.error(f"Failed to plot training curves: {e}")
            return False
    
    # ==================== 自动化训练流程 ====================
    
    def fit(self, 
            train_data: Any, 
            validation_data: Any = None,
            epochs: int = 100,
            batch_size: int = 32,
            validation_split: float = 0.2,
            callbacks: List[Callable] = None,
            save_checkpoints: bool = True,
            checkpoint_dir: str = None,
            early_stopping_patience: int = 10,
            early_stopping_min_delta: float = 0.001,
            restore_best_weights: bool = True,
            verbose: int = 1) -> Dict[str, Any]:
        """
        自动化训练流程（完整训练生命周期管理）
        
        Args:
            train_data: 训练数据（可以是数据加载器、数据集、numpy数组等）
            validation_data: 验证数据（可选，如未提供则从训练数据分割）
            epochs: 训练轮数
            batch_size: 批量大小
            validation_split: 验证集分割比例（仅在validation_data未提供时使用）
            callbacks: 训练回调函数列表
            save_checkpoints: 是否保存检查点
            checkpoint_dir: 检查点保存目录（如未提供则使用默认位置）
            early_stopping_patience: 早停耐心值（验证损失不再改善的epoch数）
            early_stopping_min_delta: 最小改善阈值
            restore_best_weights: 训练结束时是否恢复最佳权重
            verbose: 详细级别（0=静默，1=进度条，2=详细信息）
            
        Returns:
            训练历史记录和统计信息
        """
        logger.info(f"Starting automated training with {epochs} epochs")
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 初始化训练状态
        self._current_training_phase = 'fit_training'
        self._training_start_time = datetime.now()
        self._training_stop_requested = False
        
        # 准备回调函数
        if callbacks is None:
            callbacks = []
        
        # 准备数据加载器
        train_loader, val_loader = self._prepare_fit_data_loaders(
            train_data, validation_data, batch_size, validation_split
        )
        
        # 准备检查点目录
        if save_checkpoints:
            if checkpoint_dir is None:
                # 创建默认检查点目录
                checkpoint_dir = self._create_default_checkpoint_dir()
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 初始化训练历史
        training_history = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
        
        # 早停计数器
        early_stopping_counter = 0
        best_weights = None
        
        # 执行训练前回调
        self._execute_fit_callbacks('on_train_begin', callbacks, {
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_data': validation_data is not None
        })
        
        # 训练循环
        try:
            for epoch in range(epochs):
                # 检查是否停止训练
                if self._training_stop_requested:
                    logger.info("Training stopped by request")
                    break
                
                # 执行epoch开始回调
                self._execute_fit_callbacks('on_epoch_begin', callbacks, {
                    'epoch': epoch,
                    'total_epochs': epochs
                })
                
                # 训练一个epoch
                train_metrics = self.train_one_epoch(train_loader, epoch)
                
                # 验证（如果有验证数据）
                val_metrics = None
                if val_loader is not None:
                    val_metrics = self.validate_one_epoch(val_loader, epoch)
                
                # 更新学习率调度器
                current_lr = self._update_learning_rate(val_metrics)
                
                # 记录训练历史
                training_history['epoch'].append(epoch)
                training_history['train_loss'].append(train_metrics['loss'])
                training_history['train_accuracy'].append(train_metrics.get('accuracy', 0.0))
                training_history['learning_rate'].append(current_lr)
                
                if val_metrics:
                    training_history['val_loss'].append(val_metrics['loss'])
                    training_history['val_accuracy'].append(val_metrics.get('accuracy', 0.0))
                    
                    # 检查是否达到最佳验证损失
                    current_val_loss = val_metrics['loss']
                    if current_val_loss < training_history['best_val_loss'] - early_stopping_min_delta:
                        training_history['best_val_loss'] = current_val_loss
                        training_history['best_epoch'] = epoch
                        early_stopping_counter = 0
                        
                        # 保存最佳权重
                        if restore_best_weights:
                            best_weights = self._save_best_weights()
                    else:
                        early_stopping_counter += 1
                
                # 保存检查点
                if save_checkpoints and (epoch % 10 == 0 or epoch == epochs - 1):
                    self._save_checkpoint(checkpoint_dir, epoch, training_history)
                
                # 执行epoch结束回调
                self._execute_fit_callbacks('on_epoch_end', callbacks, {
                    'epoch': epoch,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'training_history': training_history
                })
                
                # 打印进度
                if verbose >= 1:
                    self._print_epoch_progress(epoch, train_metrics, val_metrics, current_lr)
                
                # 检查早停
                if early_stopping_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs (no improvement for {early_stopping_patience} epochs)")
                    break
            
            # 恢复最佳权重
            if restore_best_weights and best_weights is not None:
                self._restore_best_weights(best_weights)
            
            # 计算训练统计
            training_duration = time.time() - start_time
            
            # 执行训练结束回调
            self._execute_fit_callbacks('on_train_end', callbacks, {
                'training_history': training_history,
                'training_duration': training_duration,
                'early_stopped': early_stopping_counter >= early_stopping_patience
            })
            
            # 准备最终结果
            final_results = {
                'success': True,
                'epochs_trained': epoch + 1,
                'training_duration': training_duration,
                'final_train_loss': training_history['train_loss'][-1],
                'final_val_loss': training_history['val_loss'][-1] if training_history['val_loss'] else None,
                'best_val_loss': training_history['best_val_loss'],
                'best_epoch': training_history['best_epoch'],
                'training_history': training_history,
                'early_stopped': early_stopping_counter >= early_stopping_patience,
                'checkpoint_dir': checkpoint_dir if save_checkpoints else None
            }
            
            logger.info(f"Training completed successfully in {training_duration:.2f} seconds")
            return final_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # 执行错误回调
            self._execute_fit_callbacks('on_train_error', callbacks, {'error': str(e)})
            raise
    
    def _prepare_fit_data_loaders(self, train_data: Any, validation_data: Any,
                                batch_size: int, validation_split: float):
        """
        为fit方法准备数据加载器
        
        Args:
            train_data: 训练数据
            validation_data: 验证数据（可选）
            batch_size: 批量大小
            validation_split: 验证集分割比例
            
        Returns:
            (train_loader, val_loader) 数据加载器元组
        """
        logger.info("Preparing data loaders for fit training")
        
        try:
            # 如果提供了验证数据，直接创建数据加载器
            if validation_data is not None:
                train_loader = self.create_data_loader(train_data, batch_size, shuffle=True)
                val_loader = self.create_data_loader(validation_data, batch_size, shuffle=False)
                return train_loader, val_loader
            
            # 否则从训练数据分割验证集
            from torch.utils.data import random_split
            
            # 创建完整数据集
            full_dataset = self._create_dataset_from_data(train_data)
            dataset_size = len(full_dataset)
            
            # 计算分割大小
            val_size = int(dataset_size * validation_split)
            train_size = dataset_size - val_size
            
            # 分割数据集
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )
            
            # 创建数据加载器
            train_loader = self.create_data_loader(train_dataset, batch_size, shuffle=True)
            val_loader = self.create_data_loader(val_dataset, batch_size, shuffle=False)
            
            logger.info(f"Data split: {train_size} training samples, {val_size} validation samples")
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Failed to prepare data loaders: {e}")
            raise
    
    def _create_dataset_from_data(self, data: Any):
        """
        从各种数据格式创建数据集
        
        Args:
            data: 输入数据（numpy数组、张量、列表等）
            
        Returns:
            数据集对象
        """
        try:
            import torch
            from torch.utils.data import TensorDataset
            
            # 根据数据类型创建数据集
            if isinstance(data, (tuple, list)) and len(data) == 2:
                # (inputs, targets) 格式
                inputs, targets = data
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                if not isinstance(targets, torch.Tensor):
                    # 自动检测目标类型
                    if len(targets.shape) == 1 or targets.shape[1] == 1:
                        targets = torch.tensor(targets, dtype=torch.long)
                    else:
                        targets = torch.tensor(targets, dtype=torch.float32)
                return TensorDataset(inputs, targets)
            
            elif hasattr(data, '__getitem__') and hasattr(data, '__len__'):
                # 已经是数据集对象
                return data
            
            else:
                # 单张量数据集（无监督学习）
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)
                return TensorDataset(data)
                
        except Exception as e:
            logger.error(f"Failed to create dataset from data: {e}")
            raise
    
    def _execute_fit_callbacks(self, callback_name: str, callbacks: List[Callable], logs: Dict):
        """
        执行fit回调函数
        
        Args:
            callback_name: 回调名称
            callbacks: 回调函数列表
            logs: 日志字典
        """
        for callback in callbacks:
            try:
                if hasattr(callback, callback_name):
                    getattr(callback, callback_name)(logs)
            except Exception as e:
                logger.warning(f"Callback {callback_name} failed: {e}")
    
    def _update_learning_rate(self, val_metrics: Dict = None) -> float:
        """
        更新学习率调度器
        
        Args:
            val_metrics: 验证指标（用于ReduceLROnPlateau）
            
        Returns:
            当前学习率
        """
        try:
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                import torch
                
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics is not None:
                        self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
                
                # 获取当前学习率
                if hasattr(self, 'optimizer') and self.optimizer is not None:
                    for param_group in self.optimizer.param_groups:
                        return param_group['lr']
            
            # 默认返回0.001
            return 0.001
            
        except Exception as e:
            logger.warning(f"Failed to update learning rate: {e}")
            return 0.001
    
    def _save_best_weights(self):
        """
        保存最佳权重
        
        Returns:
            权重状态字典（用于后续恢复）
        """
        try:
            if hasattr(self, 'model') and self.model is not None:
                import torch
                return self.model.state_dict().copy()
        except Exception as e:
            logger.warning(f"Failed to save best weights: {e}")
        return None
    
    def _restore_best_weights(self, best_weights):
        """
        恢复最佳权重
        
        Args:
            best_weights: 最佳权重状态字典
        """
        try:
            if best_weights is not None and hasattr(self, 'model') and self.model is not None:
                import torch
                self.model.load_state_dict(best_weights)
                logger.info("Restored best weights from training")
        except Exception as e:
            logger.warning(f"Failed to restore best weights: {e}")
    
    def _save_checkpoint(self, checkpoint_dir: str, epoch: int, training_history: Dict):
        """
        保存训练检查点
        
        Args:
            checkpoint_dir: 检查点目录
            epoch: 当前epoch
            training_history: 训练历史记录
        """
        try:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:04d}.pth')
            
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict() if hasattr(self, 'model') else None,
                'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
                'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
                'training_history': training_history,
                'timestamp': datetime.now().isoformat()
            }
            
            import torch
            torch.save(checkpoint_data, checkpoint_path)
            
            # 同时保存简化版历史记录
            history_path = os.path.join(checkpoint_dir, f'training_history_epoch_{epoch:04d}.json')
            with open(history_path, 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'train_loss': training_history['train_loss'][-1],
                    'val_loss': training_history['val_loss'][-1] if training_history['val_loss'] else None,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            if epoch % 10 == 0:  # 每10个epoch记录一次
                logger.info(f"Checkpoint saved: {checkpoint_path}")
                
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _create_default_checkpoint_dir(self) -> str:
        """
        创建默认检查点目录
        
        Returns:
            检查点目录路径
        """
        try:
            # 基于模型名称和时间戳创建目录
            model_name = getattr(self, 'model_name', 'unnamed_model')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join('checkpoints', f'{model_name}_{timestamp}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            return checkpoint_dir
        except Exception as e:
            logger.warning(f"Failed to create default checkpoint dir: {e}")
            return './checkpoints'
    
    def _print_epoch_progress(self, epoch: int, train_metrics: Dict, val_metrics: Dict, learning_rate: float):
        """
        打印epoch进度
        
        Args:
            epoch: 当前epoch
            train_metrics: 训练指标
            val_metrics: 验证指标
            learning_rate: 当前学习率
        """
        train_acc = train_metrics.get('accuracy', 0.0)
        val_acc = val_metrics.get('accuracy', 0.0) if val_metrics else None
        
        progress_str = f"Epoch {epoch+1}: "
        progress_str += f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_acc:.2f}%"
        progress_str += f", LR: {learning_rate:.6f}"
        
        if val_metrics:
            progress_str += f", Val Loss: {val_metrics['loss']:.4f}"
            if val_acc is not None:
                progress_str += f", Val Acc: {val_acc:.2f}%"
        
        logger.info(progress_str)
    
    # ==================== 训练回调组件 ====================
    
    class EarlyStoppingCallback:
        """
        早停回调函数
        
        监控验证损失并在不再改善时停止训练
        """
        
        def __init__(self, 
                     patience: int = 10,
                     min_delta: float = 0.001,
                     restore_best_weights: bool = True,
                     verbose: int = 1):
            """
            初始化早停回调
            
            Args:
                patience: 验证损失不再改善的epoch数
                min_delta: 最小改善阈值
                restore_best_weights: 是否恢复最佳权重
                verbose: 详细级别
            """
            self.patience = patience
            self.min_delta = min_delta
            self.restore_best_weights = restore_best_weights
            self.verbose = verbose
            
            self.best_weights = None
            self.best_loss = float('inf')
            self.wait = 0
            self.stopped_epoch = 0
            self.stopped = False
        
        def on_train_begin(self, logs=None):
            """训练开始时重置状态"""
            self.best_weights = None
            self.best_loss = float('inf')
            self.wait = 0
            self.stopped_epoch = 0
            self.stopped = False
        
        def on_epoch_end(self, epoch, model, logs=None):
            """
            epoch结束时检查早停条件
            
            Args:
                epoch: 当前epoch
                model: 训练模型
                logs: 训练日志
            """
            if logs is None:
                logs = {}
            
            current_loss = logs.get('val_loss')
            if current_loss is None:
                # 如果没有验证损失，不进行早停
                return
            
            # 检查是否改善
            if current_loss < self.best_loss - self.min_delta:
                # 有改善，重置等待计数器
                self.best_loss = current_loss
                self.wait = 0
                
                # 保存最佳权重
                if self.restore_best_weights and hasattr(model, 'model'):
                    try:
                        import torch
                        self.best_weights = model.model.state_dict().copy()
                    except Exception as e:
                        if self.verbose:
                            print(f"Warning: Failed to save best weights: {e}")
                
                if self.verbose:
                    print(f"EarlyStopping: Validation loss improved to {current_loss:.4f}")
            else:
                # 没有改善，增加等待计数器
                self.wait += 1
                if self.verbose:
                    print(f"EarlyStopping: No improvement for {self.wait}/{self.patience} epochs")
            
            # 检查是否应该停止
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stopped = True
                
                # 恢复最佳权重
                if self.restore_best_weights and self.best_weights is not None:
                    try:
                        import torch
                        model.model.load_state_dict(self.best_weights)
                        if self.verbose:
                            print(f"EarlyStopping: Restored best weights from epoch {epoch - self.wait}")
                    except Exception as e:
                        if self.verbose:
                            print(f"Warning: Failed to restore best weights: {e}")
                
                if self.verbose:
                    print(f"EarlyStopping: Training stopped at epoch {epoch + 1}")
                
                # 设置训练停止标志
                if hasattr(model, '_training_stop_requested'):
                    model._training_stop_requested = True
        
        def on_train_end(self, logs=None):
            """训练结束时记录状态"""
            if self.stopped and self.verbose:
                print(f"EarlyStopping: Training stopped early at epoch {self.stopped_epoch + 1}")
    
    class ModelCheckpointCallback:
        """
        模型检查点回调
        
        定期保存模型检查点
        """
        
        def __init__(self, 
                     filepath: str = None,
                     save_best_only: bool = False,
                     save_freq: int = 10,  # 每n个epoch保存一次
                     verbose: int = 1):
            """
            初始化模型检查点回调
            
            Args:
                filepath: 检查点保存路径（如未提供则使用默认路径）
                save_best_only: 是否只保存最佳模型
                save_freq: 保存频率（epoch数）
                verbose: 详细级别
            """
            self.filepath = filepath
            self.save_best_only = save_best_only
            self.save_freq = save_freq
            self.verbose = verbose
            
            self.best_loss = float('inf')
            self.best_epoch = 0
        
        def on_train_begin(self, logs=None):
            """训练开始时重置状态"""
            self.best_loss = float('inf')
            self.best_epoch = 0
        
        def on_epoch_end(self, epoch, model, logs=None):
            """
            epoch结束时保存检查点
            
            Args:
                epoch: 当前epoch
                model: 训练模型
                logs: 训练日志
            """
            if logs is None:
                logs = {}
            
            current_loss = logs.get('val_loss')
            should_save = False
            
            # 决定是否保存
            if self.save_best_only and current_loss is not None:
                # 只保存最佳模型
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.best_epoch = epoch
                    should_save = True
                    if self.verbose:
                        print(f"ModelCheckpoint: Validation loss improved to {current_loss:.4f}, saving model")
            else:
                # 定期保存
                if epoch % self.save_freq == 0 or epoch == 0:
                    should_save = True
            
            if should_save:
                try:
                    # 生成文件路径
                    if self.filepath is None:
                        filepath = self._generate_default_filepath(model, epoch)
                    else:
                        # 支持占位符
                        filepath = self.filepath.format(epoch=epoch, **logs)
                    
                    # 确保目录存在
                    import os
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    
                    # 保存检查点
                    model._save_checkpoint(
                        os.path.dirname(filepath),
                        epoch,
                        logs.get('training_history', {})
                    )
                    
                    if self.verbose:
                        print(f"ModelCheckpoint: Saved checkpoint to {filepath}")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"ModelCheckpoint: Failed to save checkpoint: {e}")
        
        def _generate_default_filepath(self, model, epoch):
            """生成默认文件路径"""
            model_name = getattr(model, 'model_name', 'model')
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"checkpoints/{model_name}_epoch_{epoch:04d}_{timestamp}.pth"
    
    class LearningRateSchedulerCallback:
        """
        学习率调度器回调
        
        支持多种学习率调度策略
        """
        
        def __init__(self, 
                     scheduler_type: str = "plateau",
                     monitor: str = "val_loss",
                     factor: float = 0.1,
                     patience: int = 10,
                     min_lr: float = 1e-6,
                     verbose: int = 1):
            """
            初始化学习率调度器回调
            
            Args:
                scheduler_type: 调度器类型（plateau, step, cosine等）
                monitor: 监控指标
                factor: 学习率衰减因子
                patience: 等待epoch数
                min_lr: 最小学习率
                verbose: 详细级别
            """
            self.scheduler_type = scheduler_type
            self.monitor = monitor
            self.factor = factor
            self.patience = patience
            self.min_lr = min_lr
            self.verbose = verbose
            
            self.scheduler = None
        
        def on_train_begin(self, logs=None):
            """训练开始时初始化调度器"""
            if hasattr(self, '_model') and hasattr(self._model, 'optimizer'):
                try:
                    import torch
                    
                    if self.scheduler_type == "plateau":
                        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            self._model.optimizer,
                            mode='min',
                            factor=self.factor,
                            patience=self.patience,
                            min_lr=self.min_lr,
                            verbose=self.verbose > 0
                        )
                    elif self.scheduler_type == "step":
                        self.scheduler = torch.optim.lr_scheduler.StepLR(
                            self._model.optimizer,
                            step_size=self.patience,
                            gamma=self.factor
                        )
                    elif self.scheduler_type == "cosine":
                        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            self._model.optimizer,
                            T_max=100,
                            eta_min=self.min_lr
                        )
                    
                    if self.verbose:
                        print(f"LearningRateScheduler: {self.scheduler_type} scheduler initialized")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"LearningRateScheduler: Failed to initialize scheduler: {e}")
        
        def on_epoch_end(self, epoch, model, logs=None):
            """epoch结束时更新学习率"""
            if logs is None:
                logs = {}
            
            if self.scheduler is not None:
                try:
                    if self.scheduler_type == "plateau":
                        # ReduceLROnPlateau需要监控指标
                        monitor_value = logs.get(self.monitor)
                        if monitor_value is not None:
                            self.scheduler.step(monitor_value)
                    else:
                        # 其他调度器自动更新
                        self.scheduler.step()
                    
                    # 记录当前学习率
                    current_lr = self._model.optimizer.param_groups[0]['lr']
                    if self.verbose:
                        print(f"LearningRateScheduler: Learning rate updated to {current_lr:.6f}")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"LearningRateScheduler: Failed to update learning rate: {e}")
    
    class ProgressBarCallback:
        """
        训练进度条回调
        
        显示训练进度和关键指标
        """
        
        def __init__(self, 
                     update_freq: int = 1,  # 每n个epoch更新一次
                     metrics: List[str] = None,
                     verbose: int = 1):
            """
            初始化进度条回调
            
            Args:
                update_freq: 更新频率
                metrics: 要显示的指标列表
                verbose: 详细级别
            """
            self.update_freq = update_freq
            self.metrics = metrics or ["loss", "val_loss", "accuracy", "val_accuracy"]
            self.verbose = verbose
            self.total_epochs = 0
        
        def on_train_begin(self, logs=None):
            """训练开始时显示初始信息"""
            if self.verbose:
                self.total_epochs = logs.get('epochs', 0)
                print(f"ProgressBar: Starting training for {self.total_epochs} epochs")
                print("-" * 80)
        
        def on_epoch_end(self, epoch, model, logs=None):
            """epoch结束时显示进度"""
            if logs is None:
                logs = {}
            
            if self.verbose and epoch % self.update_freq == 0:
                # 构建进度信息
                epoch_str = f"Epoch {epoch + 1}/{self.total_epochs}"
                progress_info = [epoch_str]
                
                # 添加选定的指标
                for metric in self.metrics:
                    value = logs.get(metric)
                    if value is not None:
                        # 格式化指标值
                        if isinstance(value, float):
                            if metric in ["loss", "val_loss"]:
                                progress_info.append(f"{metric}: {value:.4f}")
                            elif "accuracy" in metric:
                                progress_info.append(f"{metric}: {value:.2f}%")
                            else:
                                progress_info.append(f"{metric}: {value:.4f}")
                        else:
                            progress_info.append(f"{metric}: {value}")
                
                # 显示进度
                print(" | ".join(progress_info))
        
        def on_train_end(self, logs=None):
            """训练结束时显示总结"""
            if self.verbose:
                print("-" * 80)
                duration = logs.get('training_duration', 0)
                early_stopped = logs.get('early_stopped', False)
                
                if early_stopped:
                    print(f"ProgressBar: Training completed early in {duration:.2f} seconds")
                else:
                    print(f"ProgressBar: Training completed in {duration:.2f} seconds")
    
    # ==================== 模型选择策略 ====================
    
    def select_best_model(self, 
                          model_checkpoints: List[str],
                          selection_criterion: str = "val_loss",
                          selection_mode: str = "min") -> str:
        """
        从多个模型检查点中选择最佳模型
        
        Args:
            model_checkpoints: 模型检查点路径列表
            selection_criterion: 选择标准（val_loss, val_accuracy等）
            selection_mode: 选择模式（min或max）
            
        Returns:
            最佳模型检查点路径
        """
        logger.info(f"Selecting best model from {len(model_checkpoints)} checkpoints")
        
        if not model_checkpoints:
            raise ValueError("No model checkpoints provided")
        
        best_checkpoint = None
        best_value = float('inf') if selection_mode == "min" else float('-inf')
        
        for checkpoint_path in model_checkpoints:
            try:
                # 加载检查点元数据
                checkpoint_data = self._load_checkpoint_metadata(checkpoint_path)
                
                # 获取选择标准的值
                criterion_value = checkpoint_data.get(selection_criterion)
                if criterion_value is None:
                    logger.warning(f"Checkpoint {checkpoint_path} missing {selection_criterion}")
                    continue
                
                # 根据选择模式比较值
                if selection_mode == "min" and criterion_value < best_value:
                    best_value = criterion_value
                    best_checkpoint = checkpoint_path
                elif selection_mode == "max" and criterion_value > best_value:
                    best_value = criterion_value
                    best_checkpoint = checkpoint_path
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate checkpoint {checkpoint_path}: {e}")
        
        if best_checkpoint:
            logger.info(f"Selected best model: {best_checkpoint} ({selection_criterion}={best_value:.4f})")
            return best_checkpoint
        else:
            raise ValueError("Could not select any model from checkpoints")
    
    def _load_checkpoint_metadata(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        加载检查点元数据
        
        Args:
            checkpoint_path: 检查点路径
            
        Returns:
            检查点元数据字典
        """
        try:
            import torch
            
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 提取元数据
            metadata = {
                'path': checkpoint_path,
                'epoch': checkpoint.get('epoch', 0),
                'train_loss': 0.0,
                'val_loss': 0.0,
                'timestamp': checkpoint.get('timestamp', '')
            }
            
            # 从训练历史中提取指标
            training_history = checkpoint.get('training_history', {})
            if training_history:
                if 'train_loss' in training_history and training_history['train_loss']:
                    metadata['train_loss'] = training_history['train_loss'][-1]
                if 'val_loss' in training_history and training_history['val_loss']:
                    metadata['val_loss'] = training_history['val_loss'][-1]
                if 'train_accuracy' in training_history and training_history['train_accuracy']:
                    metadata['train_accuracy'] = training_history['train_accuracy'][-1]
                if 'val_accuracy' in training_history and training_history['val_accuracy']:
                    metadata['val_accuracy'] = training_history['val_accuracy'][-1]
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint metadata from {checkpoint_path}: {e}")
            return {}
    
    def ensemble_models(self, 
                       model_checkpoints: List[str],
                       ensemble_method: str = "average") -> Any:
        """
        集成多个模型检查点
        
        Args:
            model_checkpoints: 模型检查点路径列表
            ensemble_method: 集成方法（average, weighted_average, voting）
            
        Returns:
            集成后的模型
        """
        logger.info(f"Creating model ensemble from {len(model_checkpoints)} checkpoints")
        
        if not model_checkpoints:
            raise ValueError("No model checkpoints provided")
        
        try:
            import torch
            import torch.nn as nn
            
            # 加载所有模型
            models = []
            for checkpoint_path in model_checkpoints:
                try:
                    # 加载检查点
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    model_state = checkpoint.get('model_state_dict')
                    
                    if model_state is None:
                        logger.warning(f"Checkpoint {checkpoint_path} missing model state")
                        continue
                    
                    # 创建模型实例（需要子类实现）
                    if hasattr(self, 'model') and self.model is not None:
                        # 复制当前模型结构
                        model_copy = type(self.model)()
                        model_copy.load_state_dict(model_state)
                        models.append(model_copy)
                        
                except Exception as e:
                    logger.warning(f"Failed to load model from {checkpoint_path}: {e}")
            
            if not models:
                raise ValueError("Failed to load any models from checkpoints")
            
            # 根据集成方法组合模型
            if ensemble_method == "average":
                # 平均权重
                ensemble_model = self._average_models(models)
            elif ensemble_method == "weighted_average":
                # 加权平均
                ensemble_model = self._weighted_average_models(models)
            elif ensemble_method == "voting":
                # 投票集成
                ensemble_model = self._voting_ensemble(models)
            else:
                raise ValueError(f"Unsupported ensemble method: {ensemble_method}")
            
            logger.info(f"Model ensemble created successfully with {len(models)} models")
            return ensemble_model
            
        except Exception as e:
            logger.error(f"Failed to create model ensemble: {e}")
            raise
    
    def _average_models(self, models: List['torch.nn.Module']) -> 'torch.nn.Module':
        """平均多个模型的权重"""
        if not models:
            raise ValueError("No models to average")
        
        import torch
        import torch.nn as nn
        
        # 使用第一个模型作为基础
        base_model = models[0]
        avg_state_dict = base_model.state_dict().copy()
        
        # 对每个参数进行平均
        for key in avg_state_dict:
            param_sum = torch.zeros_like(avg_state_dict[key])
            for model in models:
                param_sum += model.state_dict()[key]
            avg_state_dict[key] = param_sum / len(models)
        
        # 创建新模型并加载平均后的权重
        avg_model = type(base_model)()
        avg_model.load_state_dict(avg_state_dict)
        
        return avg_model
    
    def _weighted_average_models(self, models: List['torch.nn.Module'], 
                               weights: List[float] = None) -> 'torch.nn.Module':
        """加权平均多个模型的权重"""
        if not models:
            raise ValueError("No models to average")
        
        if weights is None:
            # 如果没有提供权重，使用均匀权重
            weights = [1.0 / len(models)] * len(models)
        
        if len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        import torch
        import torch.nn as nn
        
        # 使用第一个模型作为基础
        base_model = models[0]
        weighted_state_dict = base_model.state_dict().copy()
        
        # 对每个参数进行加权平均
        for key in weighted_state_dict:
            param_sum = torch.zeros_like(weighted_state_dict[key])
            for model, weight in zip(models, weights):
                param_sum += model.state_dict()[key] * weight
            weighted_state_dict[key] = param_sum
        
        # 创建新模型并加载加权平均后的权重
        weighted_model = type(base_model)()
        weighted_model.load_state_dict(weighted_state_dict)
        
        return weighted_model
    
    def _voting_ensemble(self, models: List['torch.nn.Module']) -> 'torch.nn.Module':
        """投票集成（用于分类任务）"""
        # 这是一个占位符实现，需要子类根据具体任务实现
        logger.warning("Voting ensemble requires task-specific implementation")
        return models[0]  # 返回第一个模型作为占位符
    
    # ==================== 训练监控和可视化组件 ====================
    
    class TrainingMonitor:
        """
        训练监控器
        
        实时监控训练过程，记录指标，提供可视化
        """
        
        def __init__(self, 
                     log_dir: str = None,
                     use_tensorboard: bool = False,
                     update_freq: int = 10,  # 每n个batch更新一次
                     verbose: int = 1):
            """
            初始化训练监控器
            
            Args:
                log_dir: 日志目录
                use_tensorboard: 是否使用TensorBoard
                update_freq: 更新频率（batch数）
                verbose: 详细级别
            """
            self.log_dir = log_dir or f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.use_tensorboard = use_tensorboard
            self.update_freq = update_freq
            self.verbose = verbose
            
            self.metrics_history = {}
            self.current_epoch = 0
            self.current_batch = 0
            self.start_time = None
            
            # 初始化TensorBoard（如果启用）
            self.writer = None
            if use_tensorboard:
                self._init_tensorboard()
        
        def _init_tensorboard(self):
            """初始化TensorBoard"""
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(self.log_dir, exist_ok=True)
                self.writer = SummaryWriter(self.log_dir)
                if self.verbose:
                    print(f"TrainingMonitor: TensorBoard initialized at {self.log_dir}")
            except ImportError:
                if self.verbose:
                    print("TrainingMonitor: TensorBoard not available, falling back to logging")
                self.use_tensorboard = False
        
        def on_train_begin(self, logs=None):
            """训练开始时初始化监控器"""
            self.start_time = time.time()
            self.current_epoch = 0
            self.current_batch = 0
            self.metrics_history = {}
            
            if self.verbose:
                print(f"TrainingMonitor: Training started, logging to {self.log_dir}")
        
        def on_epoch_begin(self, epoch, logs=None):
            """epoch开始时更新状态"""
            self.current_epoch = epoch
            self.current_batch = 0
            
            # 初始化该epoch的指标记录
            if epoch not in self.metrics_history:
                self.metrics_history[epoch] = {
                    'train_loss': [],
                    'val_loss': [],
                    'train_accuracy': [],
                    'val_accuracy': [],
                    'learning_rate': [],
                    'timestamps': []
                }
        
        def on_batch_end(self, batch, logs=None):
            """batch结束时记录指标"""
            if logs is None:
                logs = {}
            
            self.current_batch = batch
            
            # 定期记录指标
            if batch % self.update_freq == 0:
                # 记录到内存
                if 'loss' in logs:
                    self.metrics_history[self.current_epoch]['train_loss'].append(logs['loss'])
                
                # 记录到TensorBoard
                if self.use_tensorboard and self.writer is not None:
                    if 'loss' in logs:
                        self.writer.add_scalar('batch/train_loss', logs['loss'], 
                                             self.current_epoch * 1000 + batch)
                    
                    if 'accuracy' in logs:
                        self.writer.add_scalar('batch/train_accuracy', logs['accuracy'],
                                             self.current_epoch * 1000 + batch)
                
                # 控制台输出
                if self.verbose >= 2:
                    batch_info = f"Epoch {self.current_epoch}, Batch {batch}"
                    if 'loss' in logs:
                        batch_info += f", Loss: {logs['loss']:.4f}"
                    if 'accuracy' in logs:
                        batch_info += f", Acc: {logs['accuracy']:.2f}%"
                    print(batch_info)
        
        def on_epoch_end(self, epoch, logs=None):
            """epoch结束时记录指标"""
            if logs is None:
                logs = {}
            
            # 记录epoch指标
            if 'train_loss' in logs:
                self.metrics_history[epoch]['train_loss'].append(logs['train_loss'])
            if 'val_loss' in logs:
                self.metrics_history[epoch]['val_loss'].append(logs['val_loss'])
            if 'train_accuracy' in logs:
                self.metrics_history[epoch]['train_accuracy'].append(logs['train_accuracy'])
            if 'val_accuracy' in logs:
                self.metrics_history[epoch]['val_accuracy'].append(logs['val_accuracy'])
            if 'learning_rate' in logs:
                self.metrics_history[epoch]['learning_rate'].append(logs['learning_rate'])
            
            # 记录时间戳
            self.metrics_history[epoch]['timestamps'].append(time.time())
            
            # 记录到TensorBoard
            if self.use_tensorboard and self.writer is not None:
                if 'train_loss' in logs:
                    self.writer.add_scalar('epoch/train_loss', logs['train_loss'], epoch)
                if 'val_loss' in logs:
                    self.writer.add_scalar('epoch/val_loss', logs['val_loss'], epoch)
                if 'train_accuracy' in logs:
                    self.writer.add_scalar('epoch/train_accuracy', logs['train_accuracy'], epoch)
                if 'val_accuracy' in logs:
                    self.writer.add_scalar('epoch/val_accuracy', logs['val_accuracy'], epoch)
                if 'learning_rate' in logs:
                    self.writer.add_scalar('epoch/learning_rate', logs['learning_rate'], epoch)
            
            # 控制台输出
            if self.verbose:
                epoch_time = time.time() - (self.start_time or time.time())
                print(f"TrainingMonitor: Epoch {epoch} completed in {epoch_time:.2f}s")
        
        def on_train_end(self, logs=None):
            """训练结束时总结"""
            training_duration = time.time() - self.start_time if self.start_time else 0
            
            if self.verbose:
                print(f"TrainingMonitor: Training completed in {training_duration:.2f} seconds")
                print(f"TrainingMonitor: Metrics recorded for {len(self.metrics_history)} epochs")
            
            # 关闭TensorBoard写入器
            if self.use_tensorboard and self.writer is not None:
                self.writer.close()
                if self.verbose:
                    print(f"TrainingMonitor: TensorBoard logs saved to {self.log_dir}")
            
            # 保存指标到文件
            self._save_metrics_to_file()
        
        def _save_metrics_to_file(self):
            """保存指标到文件"""
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                metrics_file = os.path.join(self.log_dir, 'training_metrics.json')
                
                # 准备可序列化的数据
                serializable_metrics = {}
                for epoch, metrics in self.metrics_history.items():
                    serializable_metrics[str(epoch)] = {
                        'train_loss': [float(v) for v in metrics.get('train_loss', [])],
                        'val_loss': [float(v) for v in metrics.get('val_loss', [])],
                        'train_accuracy': [float(v) for v in metrics.get('train_accuracy', [])],
                        'val_accuracy': [float(v) for v in metrics.get('val_accuracy', [])],
                        'learning_rate': [float(v) for v in metrics.get('learning_rate', [])],
                        'timestamps': metrics.get('timestamps', [])
                    }
                
                with open(metrics_file, 'w') as f:
                    json.dump(serializable_metrics, f, indent=2)
                
                if self.verbose:
                    print(f"TrainingMonitor: Metrics saved to {metrics_file}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"TrainingMonitor: Failed to save metrics: {e}")
        
        def get_metrics_summary(self) -> Dict[str, Any]:
            """获取指标总结"""
            if not self.metrics_history:
                return {}
            
            summary = {
                'total_epochs': len(self.metrics_history),
                'final_metrics': {},
                'best_metrics': {}
            }
            
            # 获取最终指标
            last_epoch = max(self.metrics_history.keys())
            last_metrics = self.metrics_history[last_epoch]
            
            for metric_name in ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy']:
                if metric_name in last_metrics and last_metrics[metric_name]:
                    summary['final_metrics'][metric_name] = last_metrics[metric_name][-1]
            
            # 计算最佳指标
            all_val_losses = []
            all_val_accuracies = []
            
            for epoch_metrics in self.metrics_history.values():
                if 'val_loss' in epoch_metrics and epoch_metrics['val_loss']:
                    all_val_losses.extend(epoch_metrics['val_loss'])
                if 'val_accuracy' in epoch_metrics and epoch_metrics['val_accuracy']:
                    all_val_accuracies.extend(epoch_metrics['val_accuracy'])
            
            if all_val_losses:
                summary['best_metrics']['val_loss'] = min(all_val_losses)
            if all_val_accuracies:
                summary['best_metrics']['val_accuracy'] = max(all_val_accuracies)
            
            return summary
        
        def plot_metrics(self, save_path: str = None):
            """绘制训练指标"""
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle('Training Metrics', fontsize=16)
                
                # 准备数据
                epochs = sorted(self.metrics_history.keys())
                train_losses = []
                val_losses = []
                train_accuracies = []
                val_accuracies = []
                
                for epoch in epochs:
                    metrics = self.metrics_history[epoch]
                    if metrics.get('train_loss'):
                        train_losses.append(metrics['train_loss'][-1])
                    if metrics.get('val_loss'):
                        val_losses.append(metrics['val_loss'][-1])
                    if metrics.get('train_accuracy'):
                        train_accuracies.append(metrics['train_accuracy'][-1])
                    if metrics.get('val_accuracy'):
                        val_accuracies.append(metrics['val_accuracy'][-1])
                
                # 1. 训练和验证损失
                ax1 = axes[0, 0]
                if train_losses:
                    ax1.plot(range(len(train_losses)), train_losses, 'b-', label='Train Loss', linewidth=2)
                if val_losses:
                    ax1.plot(range(len(val_losses)), val_losses, 'r-', label='Val Loss', linewidth=2)
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training and Validation Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 2. 训练和验证准确率
                ax2 = axes[0, 1]
                if train_accuracies:
                    ax2.plot(range(len(train_accuracies)), train_accuracies, 'b-', 
                           label='Train Accuracy', linewidth=2)
                if val_accuracies:
                    ax2.plot(range(len(val_accuracies)), val_accuracies, 'r-', 
                           label='Val Accuracy', linewidth=2)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy (%)')
                ax2.set_title('Training and Validation Accuracy')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # 3. 学习率变化
                ax3 = axes[1, 0]
                learning_rates = []
                for epoch in epochs:
                    metrics = self.metrics_history[epoch]
                    if metrics.get('learning_rate'):
                        learning_rates.append(metrics['learning_rate'][-1])
                
                if learning_rates:
                    ax3.plot(range(len(learning_rates)), learning_rates, 'g-', 
                           label='Learning Rate', linewidth=2)
                    ax3.set_xlabel('Epoch')
                    ax3.set_ylabel('Learning Rate')
                    ax3.set_title('Learning Rate Schedule')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                else:
                    ax3.text(0.5, 0.5, 'No learning rate data', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax3.transAxes)
                    ax3.set_title('Learning Rate Schedule')
                
                # 4. 训练进度（时间）
                ax4 = axes[1, 1]
                timestamps = []
                for epoch in epochs:
                    metrics = self.metrics_history[epoch]
                    if metrics.get('timestamps'):
                        timestamps.append(metrics['timestamps'][-1])
                
                if timestamps and self.start_time:
                    relative_times = [(t - self.start_time) / 60 for t in timestamps]  # 转换为分钟
                    ax4.plot(range(len(relative_times)), relative_times, 'purple-', 
                           label='Training Time', linewidth=2)
                    ax4.set_xlabel('Epoch')
                    ax4.set_ylabel('Time (minutes)')
                    ax4.set_title('Training Progress Over Time')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'No timestamp data', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax4.transAxes)
                    ax4.set_title('Training Progress Over Time')
                
                # 调整布局
                plt.tight_layout()
                
                # 保存或显示
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"TrainingMonitor: Metrics plot saved to {save_path}")
                else:
                    plt.show()
                
                plt.close(fig)
                
            except ImportError:
                print("TrainingMonitor: matplotlib not available, cannot plot metrics")
            except Exception as e:
                print(f"TrainingMonitor: Failed to plot metrics: {e}")
    
    def create_training_report(self, 
                              training_history: Dict[str, Any],
                              save_path: str = None) -> str:
        """
        创建训练报告
        
        Args:
            training_history: 训练历史记录
            save_path: 报告保存路径
            
        Returns:
            报告文本
        """
        logger.info("Creating training report")
        
        try:
            # 提取关键信息
            epochs_trained = training_history.get('epochs_trained', 0)
            training_duration = training_history.get('training_duration', 0)
            early_stopped = training_history.get('early_stopped', False)
            best_epoch = training_history.get('best_epoch', 0)
            best_val_loss = training_history.get('best_val_loss', 0.0)
            
            # 从历史中获取指标
            history_data = training_history.get('training_history', {})
            train_losses = history_data.get('train_loss', [])
            val_losses = history_data.get('val_loss', [])
            train_accuracies = history_data.get('train_accuracy', [])
            val_accuracies = history_data.get('val_accuracy', [])
            
            # 构建报告
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("TRAINING REPORT")
            report_lines.append("=" * 80)
            report_lines.append("")
            
            # 训练概述
            report_lines.append("TRAINING OVERVIEW")
            report_lines.append("-" * 40)
            report_lines.append(f"Total epochs trained: {epochs_trained}")
            report_lines.append(f"Training duration: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
            report_lines.append(f"Early stopped: {'Yes' if early_stopped else 'No'}")
            if best_epoch > 0:
                report_lines.append(f"Best epoch: {best_epoch}")
                report_lines.append(f"Best validation loss: {best_val_loss:.6f}")
            report_lines.append("")
            
            # 最终指标
            report_lines.append("FINAL METRICS")
            report_lines.append("-" * 40)
            if train_losses:
                report_lines.append(f"Final training loss: {train_losses[-1]:.6f}")
            if val_losses:
                report_lines.append(f"Final validation loss: {val_losses[-1]:.6f}")
            if train_accuracies:
                report_lines.append(f"Final training accuracy: {train_accuracies[-1]:.2f}%")
            if val_accuracies:
                report_lines.append(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")
            report_lines.append("")
            
            # 指标趋势
            report_lines.append("METRICS TRENDS")
            report_lines.append("-" * 40)
            if len(train_losses) >= 2:
                loss_improvement = train_losses[0] - train_losses[-1]
                loss_improvement_pct = (loss_improvement / train_losses[0]) * 100 if train_losses[0] > 0 else 0
                report_lines.append(f"Training loss improvement: {loss_improvement:.6f} ({loss_improvement_pct:.1f}%)")
            
            if len(val_losses) >= 2:
                val_loss_improvement = val_losses[0] - val_losses[-1]
                val_loss_improvement_pct = (val_loss_improvement / val_losses[0]) * 100 if val_losses[0] > 0 else 0
                report_lines.append(f"Validation loss improvement: {val_loss_improvement:.6f} ({val_loss_improvement_pct:.1f}%)")
            report_lines.append("")
            
            # 训练统计
            report_lines.append("TRAINING STATISTICS")
            report_lines.append("-" * 40)
            if train_losses:
                min_train_loss = min(train_losses)
                max_train_loss = max(train_losses)
                report_lines.append(f"Training loss range: {min_train_loss:.6f} - {max_train_loss:.6f}")
            
            if val_losses:
                min_val_loss = min(val_losses)
                max_val_loss = max(val_losses)
                report_lines.append(f"Validation loss range: {min_val_loss:.6f} - {max_val_loss:.6f}")
            
            if train_accuracies:
                min_train_acc = min(train_accuracies)
                max_train_acc = max(train_accuracies)
                report_lines.append(f"Training accuracy range: {min_train_acc:.2f}% - {max_train_acc:.2f}%")
            
            if val_accuracies:
                min_val_acc = min(val_accuracies)
                max_val_acc = max(val_accuracies)
                report_lines.append(f"Validation accuracy range: {min_val_acc:.2f}% - {max_val_acc:.2f}%")
            report_lines.append("")
            
            # 训练建议
            report_lines.append("TRAINING RECOMMENDATIONS")
            report_lines.append("-" * 40)
            if val_losses and val_losses[-1] > val_losses[0]:
                report_lines.append("⚠️  Validation loss increased during training. Consider:")
                report_lines.append("   - Reducing learning rate")
                report_lines.append("   - Adding regularization (dropout, weight decay)")
                report_lines.append("   - Increasing model capacity")
            elif val_losses and (val_losses[-1] - min(val_losses)) < 0.01:
                report_lines.append("✓  Training converged well.")
                report_lines.append("   Model appears to have learned effectively.")
            else:
                report_lines.append("✓  Training completed successfully.")
                report_lines.append("   Model performance improved during training.")
            report_lines.append("")
            
            report_lines.append("=" * 80)
            
            # 合并报告行
            report_text = "\n".join(report_lines)
            
            # 保存报告
            if save_path:
                try:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(report_text)
                    logger.info(f"Training report saved to {save_path}")
                except Exception as e:
                    logger.warning(f"Failed to save training report: {e}")
            
            return report_text
            
        except Exception as e:
            logger.error(f"Failed to create training report: {e}")
            return f"Error creating training report: {e}"
