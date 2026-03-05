"""
Base Audio Neural Network Model

This module provides the base class for all audio neural network models
to reduce code duplication and ensure consistency across audio models.

Key Features:
1. Device management (GPU/CPU detection)
2. Weight initialization
3. Logging setup
4. Basic prediction interface
5. Model state management
"""

import logging
import torch
from typing import Dict, Any


class BaseAudioNeuralModel(torch.nn.Module):
    """Base class for all audio neural network models to reduce code duplication
    
    This base class provides common functionality for:
    1. Device management (GPU/CPU detection)
    2. Weight initialization
    3. Logging setup
    4. Basic prediction interface
    5. Model state management
    """
    
    def __init__(self, model_name: str = "BaseAudioModel", config: Dict[str, Any] = None):
        """Initialize base audio neural network model
        
        Args:
            model_name: Name of the model for logging
            config: Configuration dictionary
        """
        super().__init__()
        self.model_name = model_name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
        
        # Device detection and optimization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"{model_name} initialized on device: {self.device}")
        
        # Initialize model parameters
        self._init_model_parameters()
        
        # Move model to appropriate device
        self.to(self.device)
    
    def _init_model_parameters(self):
        """Initialize model-specific parameters (to be overridden by subclasses)"""
        self.model_params = {}
    
    def _initialize_weights(self):
        """Initialize model weights using appropriate initialization methods
        
        This method provides common weight initialization strategies that can be
        used or overridden by subclasses.
        """
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.LayerNorm)):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def predict(self, features, **kwargs):
        """Base prediction method (to be overridden by subclasses)
        
        Args:
            features: Input features for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Prediction result
        """
        self.logger.warning(f"predict method called on base class {self.model_name}, providing default implementation")
        
        # Provide a default implementation that returns basic information
        # This allows the system to continue functioning even if subclasses don't override
        if isinstance(features, torch.Tensor):
            shape = list(features.shape)
            device = features.device
        else:
            shape = str(type(features))
            device = "unknown"
        
        return {
            "prediction": "default_prediction",
            "features_shape": shape,
            "features_device": str(device),
            "model_name": self.model_name,
            "warning": "predict method not implemented in subclass, using base implementation"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for debugging and monitoring
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "model_params": self.model_params
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'config': self.config,
            'model_params': self.model_params
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.model_params = checkpoint.get('model_params', {})
        self.logger.info(f"Model checkpoint loaded from {filepath}")
    
    def to(self, device=None, **kwargs):
        """Move model to specified device with enhanced logging"""
        if device is None:
            device = self.device
        result = super().to(device, **kwargs)
        self.device = device
        self.logger.info(f"Model moved to device: {device}")
        return result