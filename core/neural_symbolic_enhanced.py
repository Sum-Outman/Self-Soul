#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
Enhanced Neural Symbolic Model and Joint Training Coordinator

This module provides:
1. Enhanced NeuralSymbolicModel with advanced neural network architectures and training capabilities.
2. JointTrainingCoordinator for coordinating multiple models during training.

Key enhancements:
- Transformer-based neural architecture
- Attention mechanisms for better symbolic reasoning
- Full training pipeline with optimizer and loss functions
- Support for multi-task and joint training
- Real-time performance monitoring
- Model checkpointing and saving

Copyright (c) 2025 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, deque
import json
import pickle
import os

logger = logging.getLogger(__name__)

class EnhancedNeuralSymbolicModel(nn.Module):
    """
    Enhanced neural-symbolic model with advanced neural network architectures,
    training capabilities, and attention mechanisms.
    
    Features:
    - Transformer-based neural encoder
    - Multi-head attention for symbolic reasoning
    - Residual connections and layer normalization
    - Dropout for regularization
    - Full training pipeline with Adam optimizer
    - Support for symbolic constraints during training
    - Model checkpointing and loading
    """
    
    def __init__(self, 
                 input_dim: int = 384,
                 symbolic_dim: int = 256,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout_rate: float = 0.1,
                 use_transformer: bool = True):
        super(EnhancedNeuralSymbolicModel, self).__init__()
        
        self.input_dim = input_dim
        self.symbolic_dim = symbolic_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_transformer = use_transformer
        
        # Neural encoder with advanced architecture
        if use_transformer:
            # Transformer-based encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout_rate,
                activation='gelu',
                batch_first=True
            )
            self.neural_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.encoder_norm = nn.LayerNorm(hidden_dim)
        else:
            # Multi-layer perceptron encoder
            self.neural_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_rate),
            )
        
        # Symbolic projection layer
        self.symbolic_projector = nn.Linear(hidden_dim, symbolic_dim)
        
        # Multi-head attention for symbolic reasoning
        self.symbolic_attention = nn.MultiheadAttention(
            embed_dim=symbolic_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Symbolic reasoning layers with residual connections
        self.symbolic_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(symbolic_dim, symbolic_dim),
                nn.ReLU(),
                nn.LayerNorm(symbolic_dim),
                nn.Dropout(dropout_rate)
            ) for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_attention = nn.MultiheadAttention(
            embed_dim=symbolic_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(symbolic_dim, symbolic_dim),
            nn.ReLU(),
            nn.LayerNorm(symbolic_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(symbolic_dim, input_dim)
        )
        
        # Training-related attributes
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        # Device management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized EnhancedNeuralSymbolicModel with input_dim={input_dim}, "
                   f"symbolic_dim={symbolic_dim}, hidden_dim={hidden_dim}, "
                   f"num_layers={num_layers}, device={self.device}")
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x: torch.Tensor, symbolic_constraints: Optional[List[Dict]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional symbolic constraints.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            symbolic_constraints: List of constraint dictionaries
            
        Returns:
            output: Reconstructed input of shape (batch_size, input_dim)
            symbolic_features: Symbolic representation of shape (batch_size, symbolic_dim)
        """
        batch_size = x.size(0)
        
        # Neural encoding
        if self.use_transformer:
            # Project input to hidden dimension
            x_proj = self.input_projection(x).unsqueeze(1)  # (batch_size, 1, hidden_dim)
            x_proj = self.encoder_norm(x_proj)
            
            # Transformer encoding
            neural_features = self.neural_encoder(x_proj).squeeze(1)  # (batch_size, hidden_dim)
        else:
            neural_features = self.neural_encoder(x)
        
        # Symbolic projection
        symbolic_features = self.symbolic_projector(neural_features)
        
        # Apply symbolic attention
        symbolic_features_attn, _ = self.symbolic_attention(
            symbolic_features.unsqueeze(1),
            symbolic_features.unsqueeze(1),
            symbolic_features.unsqueeze(1)
        )
        symbolic_features = symbolic_features + symbolic_features_attn.squeeze(1)
        
        # Apply symbolic layers with residual connections
        for layer in self.symbolic_layers:
            residual = symbolic_features
            symbolic_features = layer(symbolic_features)
            symbolic_features = symbolic_features + residual
        
        # Apply symbolic constraints if provided
        if symbolic_constraints is not None:
            symbolic_features = self._apply_symbolic_constraints(symbolic_features, symbolic_constraints)
        
        # Apply output attention
        output_features, _ = self.output_attention(
            symbolic_features.unsqueeze(1),
            symbolic_features.unsqueeze(1),
            symbolic_features.unsqueeze(1)
        )
        symbolic_features = symbolic_features + output_features.squeeze(1)
        
        # Decode back to input space
        output = self.output_layer(symbolic_features)
        
        return output, symbolic_features
    
    def _apply_symbolic_constraints(self, features: torch.Tensor, constraints: List[Dict]) -> torch.Tensor:
        """
        Apply symbolic constraints to neural features.
        
        Args:
            features: Tensor of shape (batch_size, symbolic_dim)
            constraints: List of constraint dictionaries
            
        Returns:
            Constrained features tensor
        """
        constrained_features = features.clone()
        
        for constraint in constraints:
            constraint_type = constraint.get("type", "")
            
            if constraint_type == "equality":
                idx1, idx2 = constraint["indices"]
                # Enforce equality by averaging
                avg = (constrained_features[:, idx1] + constrained_features[:, idx2]) / 2
                constrained_features[:, idx1] = avg
                constrained_features[:, idx2] = avg
            
            elif constraint_type == "inequality":
                idx1, idx2 = constraint["indices"]
                diff = constrained_features[:, idx1] - constrained_features[:, idx2]
                
                if constraint.get("greater", True):
                    # Ensure idx1 > idx2
                    mask = diff <= 0
                    constrained_features[mask, idx1] = constrained_features[mask, idx2] + 0.1
                else:
                    # Ensure idx1 < idx2
                    mask = diff >= 0
                    constrained_features[mask, idx1] = constrained_features[mask, idx2] - 0.1
            
            elif constraint_type == "range":
                idx = constraint["index"]
                min_val = constraint.get("min", -1.0)
                max_val = constraint.get("max", 1.0)
                # Clip values to range
                constrained_features[:, idx] = torch.clamp(constrained_features[:, idx], min_val, max_val)
        
        return constrained_features
    
    def configure_optimizer(self, 
                           optimizer_type: str = "adam",
                           learning_rate: float = 0.001,
                           weight_decay: float = 1e-5,
                           scheduler_type: str = "reduce_on_plateau"):
        """
        Configure optimizer and learning rate scheduler.
        
        Args:
            optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            scheduler_type: Type of scheduler ('reduce_on_plateau', 'step', 'cosine')
        """
        # Configure optimizer
        if optimizer_type == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_type == "adamw":
            self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Configure scheduler
        if scheduler_type == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        elif scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        elif scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=50, eta_min=1e-6
            )
        else:
            self.scheduler = None
        
        logger.info(f"Configured {optimizer_type} optimizer with lr={learning_rate}, "
                   f"scheduler={scheduler_type}")
    
    def train_step(self, batch: torch.Tensor, symbolic_constraints: Optional[List[Dict]] = None) -> float:
        """
        Perform a single training step.
        
        Args:
            batch: Input batch tensor of shape (batch_size, input_dim)
            symbolic_constraints: Optional symbolic constraints
            
        Returns:
            loss: Training loss for this batch
        """
        if self.optimizer is None:
            self.configure_optimizer()
        
        self.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = batch.to(self.device)
        
        # Forward pass
        reconstructed, _ = self(batch, symbolic_constraints)
        
        # Compute loss
        loss = self.criterion(reconstructed, batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader: DataLoader, symbolic_constraints: Optional[List[Dict]] = None) -> float:
        """
        Validate the model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
            symbolic_constraints: Optional symbolic constraints
            
        Returns:
            val_loss: Average validation loss
        """
        self.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, tuple):
                    batch = batch[0]  # Assume first element is input
                
                batch = batch.to(self.device)
                reconstructed, _ = self(batch, symbolic_constraints)
                loss = self.criterion(reconstructed, batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def fit(self, 
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 100,
            symbolic_constraints: Optional[List[Dict]] = None,
            checkpoint_dir: Optional[str] = None,
            early_stopping_patience: int = 20) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of training epochs
            symbolic_constraints: Optional symbolic constraints
            checkpoint_dir: Directory to save model checkpoints
            early_stopping_patience: Patience for early stopping
            
        Returns:
            training_history: Dictionary containing training metrics
        """
        if self.optimizer is None:
            self.configure_optimizer()
        
        # Initialize training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        best_val_loss = float('inf')
        early_stopping_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = 0.0
            num_train_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                if isinstance(batch, tuple):
                    batch = batch[0]  # Assume first element is input
                
                batch_loss = self.train_step(batch, symbolic_constraints)
                train_loss += batch_loss
                num_train_batches += 1
                
                # Log progress every 10% of batches
                if batch_idx % max(1, len(train_loader) // 10) == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                               f"Loss: {batch_loss:.6f}")
            
            avg_train_loss = train_loss / max(num_train_batches, 1)
            self.training_history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                avg_val_loss = self.validate(val_loader, symbolic_constraints)
                self.training_history['val_loss'].append(avg_val_loss)
                
                # Update scheduler if using ReduceLROnPlateau
                if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)
            else:
                avg_val_loss = None
                self.training_history['val_loss'].append(None)
                
                # Update scheduler if not ReduceLROnPlateau
                if self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
            
            # Record epoch time and learning rate
            epoch_time = time.time() - epoch_start_time
            self.training_history['epoch_times'].append(epoch_time)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['learning_rates'].append(current_lr)
            
            # Log epoch results
            if avg_val_loss is not None:
                logger.info(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s, "
                           f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                           f"LR: {current_lr:.6f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s, "
                           f"Train Loss: {avg_train_loss:.6f}, LR: {current_lr:.6f}")
            
            # Save checkpoint if validation loss improved
            if checkpoint_dir is not None and avg_val_loss is not None:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    early_stopping_counter = 0
                    self.save_model(os.path.join(checkpoint_dir, f"best_model_epoch_{epoch+1}.pth"))
                    logger.info(f"Saved best model with validation loss: {best_val_loss:.6f}")
                else:
                    early_stopping_counter += 1
            
            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        logger.info("Training completed")
        return self.training_history
    
    def predict(self, 
                data: Union[torch.Tensor, np.ndarray],
                symbolic_constraints: Optional[List[Dict]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the trained model.
        
        Args:
            data: Input data as torch Tensor or numpy array
            symbolic_constraints: Optional symbolic constraints
            
        Returns:
            reconstructed: Reconstructed input as numpy array
            symbolic_features: Symbolic features as numpy array
        """
        self.eval()
        
        # Convert to tensor if necessary
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data)
        
        data = data.to(self.device)
        
        with torch.no_grad():
            reconstructed, symbolic_features = self(data, symbolic_constraints)
        
        return reconstructed.cpu().numpy(), symbolic_features.cpu().numpy()
    
    def save_model(self, filepath: str):
        """
        Save model weights and training history.
        
        Args:
            filepath: Path to save model
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_history': self.training_history,
            'model_config': {
                'input_dim': self.input_dim,
                'symbolic_dim': self.symbolic_dim,
                'hidden_dim': self.hidden_dim,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'dropout_rate': self.dropout_rate,
                'use_transformer': self.use_transformer
            }
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model weights and training history.
        
        Args:
            filepath: Path to load model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if checkpoint['optimizer_state_dict'] is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if checkpoint['scheduler_state_dict'] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training history
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'input_dim': self.input_dim,
            'symbolic_dim': self.symbolic_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'use_transformer': self.use_transformer,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'optimizer': self.optimizer.__class__.__name__ if self.optimizer else None,
            'scheduler': self.scheduler.__class__.__name__ if self.scheduler else None,
            'training_history_keys': list(self.training_history.keys())
        }

class JointTrainingCoordinator:
    """
    Coordinator for joint training of multiple neural-symbolic models.
    
    Features:
    - Coordinate training of multiple models with shared constraints
    - Handle data distribution across models
    - Manage communication between models during training
    - Optimize joint loss functions
    - Support for federated learning scenarios
    """
    
    def __init__(self, models: List[EnhancedNeuralSymbolicModel], coordinator_config: Dict[str, Any] = None):
        """
        Initialize joint training coordinator.
        
        Args:
            models: List of EnhancedNeuralSymbolicModel instances
            coordinator_config: Configuration dictionary for coordinator
        """
        self.models = models
        self.config = coordinator_config or {}
        
        # Coordinator settings
        self.communication_frequency = self.config.get('communication_frequency', 5)
        self.joint_loss_weight = self.config.get('joint_loss_weight', 0.5)
        self.coordination_strategy = self.config.get('coordination_strategy', 'average')
        
        # Training state
        self.training_history = {
            'joint_loss': [],
            'individual_losses': defaultdict(list),
            'communication_steps': [],
            'coordination_metrics': []
        }
        
        # Communication buffers
        self.communication_buffer = {}
        self._initialize_communication_buffer()
        
        logger.info(f"Initialized JointTrainingCoordinator with {len(models)} models, "
                   f"strategy: {self.coordination_strategy}")
    
    def _initialize_communication_buffer(self):
        """Initialize communication buffer for model parameters"""
        for i, model in enumerate(self.models):
            self.communication_buffer[f'model_{i}'] = {
                'parameters': None,
                'gradients': None,
                'loss': None,
                'last_update': 0
            }
    
    def coordinate_training(self, 
                           train_loaders: List[DataLoader],
                           val_loaders: Optional[List[DataLoader]] = None,
                           epochs: int = 100,
                           symbolic_constraints: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Coordinate joint training of multiple models.
        
        Args:
            train_loaders: List of DataLoaders for each model
            val_loaders: Optional list of DataLoaders for validation
            epochs: Number of training epochs
            symbolic_constraints: Optional symbolic constraints
            
        Returns:
            joint_training_history: Dictionary containing training metrics
        """
        if len(train_loaders) != len(self.models):
            raise ValueError(f"Number of train_loaders ({len(train_loaders)}) must "
                           f"match number of models ({len(self.models)})")
        
        if val_loaders is not None and len(val_loaders) != len(self.models):
            raise ValueError(f"Number of val_loaders ({len(val_loaders)}) must "
                           f"match number of models ({len(self.models)})")
        
        logger.info(f"Starting joint training for {epochs} epochs")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Individual model training
            individual_losses = []
            
            for i, (model, train_loader) in enumerate(zip(self.models, train_loaders)):
                logger.info(f"Training model {i+1}/{len(self.models)} in epoch {epoch+1}")
                
                # Train model for one epoch
                model.train()
                total_loss = 0.0
                num_batches = 0
                
                for batch in train_loader:
                    if isinstance(batch, tuple):
                        batch = batch[0]
                    
                    loss = model.train_step(batch, symbolic_constraints)
                    total_loss += loss
                    num_batches += 1
                
                avg_loss = total_loss / max(num_batches, 1)
                individual_losses.append(avg_loss)
                
                # Update communication buffer
                self.communication_buffer[f'model_{i}']['loss'] = avg_loss
                self.communication_buffer[f'model_{i}']['last_update'] = epoch
            
            # Calculate joint loss
            joint_loss = self._calculate_joint_loss(individual_losses)
            self.training_history['joint_loss'].append(joint_loss)
            
            # Store individual losses
            for i, loss in enumerate(individual_losses):
                self.training_history['individual_losses'][f'model_{i}'].append(loss)
            
            # Coordinate models (parameter sharing/averaging)
            if (epoch + 1) % self.communication_frequency == 0:
                coordination_metrics = self._coordinate_models()
                self.training_history['coordination_metrics'].append(coordination_metrics)
                self.training_history['communication_steps'].append(epoch)
                
                logger.info(f"Model coordination completed at epoch {epoch+1}")
            
            # Validation phase
            if val_loaders is not None:
                val_losses = []
                
                for i, (model, val_loader) in enumerate(zip(self.models, val_loaders)):
                    val_loss = model.validate(val_loader, symbolic_constraints)
                    val_losses.append(val_loss)
                
                avg_val_loss = sum(val_losses) / len(val_losses)
                logger.info(f"Epoch {epoch+1}/{epochs} - Joint Loss: {joint_loss:.6f}, "
                           f"Avg Val Loss: {avg_val_loss:.6f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Joint Loss: {joint_loss:.6f}")
            
            # Record epoch time
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        
        logger.info("Joint training completed")
        return self.training_history
    
    def _calculate_joint_loss(self, individual_losses: List[float]) -> float:
        """
        Calculate joint loss from individual model losses.
        
        Args:
            individual_losses: List of individual model losses
            
        Returns:
            joint_loss: Combined joint loss
        """
        if self.coordination_strategy == 'average':
            return sum(individual_losses) / len(individual_losses)
        elif self.coordination_strategy == 'weighted':
            # Weight by model performance (better models get higher weight)
            weights = [1.0 / (loss + 1e-8) for loss in individual_losses]
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            joint_loss = sum(loss * weight for loss, weight in zip(individual_losses, normalized_weights))
            return joint_loss
        elif self.coordination_strategy == 'min':
            return min(individual_losses)
        elif self.coordination_strategy == 'max':
            return max(individual_losses)
        else:
            return sum(individual_losses) / len(individual_losses)
    
    def _coordinate_models(self) -> Dict[str, Any]:
        """
        Coordinate models by sharing parameters based on coordination strategy.
        
        Returns:
            coordination_metrics: Metrics about the coordination process
        """
        coordination_metrics = {
            'strategy': self.coordination_strategy,
            'parameter_changes': [],
            'model_contributions': []
        }
        
        if self.coordination_strategy == 'average':
            # Average parameters across all models
            avg_params = {}
            
            # Collect all parameters
            for i, model in enumerate(self.models):
                for name, param in model.named_parameters():
                    if name not in avg_params:
                        avg_params[name] = []
                    avg_params[name].append(param.data.clone())
            
            # Calculate averages
            for name, param_list in avg_params.items():
                if len(param_list) > 0:
                    avg_tensor = torch.stack(param_list).mean(dim=0)
                    
                    # Apply averaged parameters to all models
                    for model in self.models:
                        model_param = dict(model.named_parameters())[name]
                        param_change = torch.norm(model_param.data - avg_tensor).item()
                        coordination_metrics['parameter_changes'].append(param_change)
                        model_param.data.copy_(avg_tensor)
            
            coordination_metrics['model_contributions'] = [1.0 / len(self.models)] * len(self.models)
        
        elif self.coordination_strategy == 'weighted':
            # Weighted average based on model performance
            weights = []
            for i, model in enumerate(self.models):
                loss = self.communication_buffer[f'model_{i}']['loss']
                if loss is None:
                    weight = 1.0
                else:
                    weight = 1.0 / (loss + 1e-8)
                weights.append(weight)
            
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # Weighted average of parameters
            weighted_params = {}
            
            for i, (model, weight) in enumerate(zip(self.models, normalized_weights)):
                for name, param in model.named_parameters():
                    if name not in weighted_params:
                        weighted_params[name] = []
                    weighted_params[name].append(param.data.clone() * weight)
            
            # Calculate weighted averages
            for name, param_list in weighted_params.items():
                if len(param_list) > 0:
                    weighted_avg = torch.stack(param_list).sum(dim=0)
                    
                    # Apply weighted averages to all models
                    for model in self.models:
                        model_param = dict(model.named_parameters())[name]
                        model_param.data.copy_(weighted_avg)
            
            coordination_metrics['model_contributions'] = normalized_weights
        
        elif self.coordination_strategy == 'best_model':
            # Use parameters from the best performing model
            losses = []
            for i, model in enumerate(self.models):
                loss = self.communication_buffer[f'model_{i}']['loss']
                if loss is None:
                    loss = float('inf')
                losses.append(loss)
            
            best_model_idx = losses.index(min(losses))
            best_model = self.models[best_model_idx]
            
            # Copy parameters from best model to all other models
            for i, model in enumerate(self.models):
                if i != best_model_idx:
                    for (name1, param1), (name2, param2) in zip(
                        best_model.named_parameters(), model.named_parameters()
                    ):
                        if name1 == name2:
                            param2.data.copy_(param1.data)
            
            coordination_metrics['model_contributions'] = [
                1.0 if i == best_model_idx else 0.0 for i in range(len(self.models))
            ]
        
        coordination_metrics['average_parameter_change'] = (
            sum(coordination_metrics['parameter_changes']) / 
            max(len(coordination_metrics['parameter_changes']), 1)
        )
        
        return coordination_metrics
    
    def save_coordinator_state(self, filepath: str):
        """
        Save coordinator state.
        
        Args:
            filepath: Path to save coordinator state
        """
        coordinator_state = {
            'config': self.config,
            'training_history': self.training_history,
            'communication_buffer': self.communication_buffer,
            'coordination_strategy': self.coordination_strategy,
            'communication_frequency': self.communication_frequency,
            'joint_loss_weight': self.joint_loss_weight
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(coordinator_state, f)
        
        logger.info(f"Coordinator state saved to {filepath}")
    
    def load_coordinator_state(self, filepath: str):
        """
        Load coordinator state.
        
        Args:
            filepath: Path to load coordinator state from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Coordinator state file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            coordinator_state = pickle.load(f)
        
        self.config = coordinator_state.get('config', {})
        self.training_history = coordinator_state.get('training_history', self.training_history)
        self.communication_buffer = coordinator_state.get('communication_buffer', self.communication_buffer)
        self.coordination_strategy = coordinator_state.get('coordination_strategy', self.coordination_strategy)
        self.communication_frequency = coordinator_state.get('communication_frequency', self.communication_frequency)
        self.joint_loss_weight = coordinator_state.get('joint_loss_weight', self.joint_loss_weight)
        
        logger.info(f"Coordinator state loaded from {filepath}")
    
    def get_coordinator_info(self) -> Dict[str, Any]:
        """
        Get information about the coordinator.
        
        Returns:
            Dictionary containing coordinator information
        """
        return {
            'num_models': len(self.models),
            'coordination_strategy': self.coordination_strategy,
            'communication_frequency': self.communication_frequency,
            'joint_loss_weight': self.joint_loss_weight,
            'training_history_keys': list(self.training_history.keys()),
            'model_types': [model.__class__.__name__ for model in self.models],
            'model_devices': [str(model.device) for model in self.models]
        }

class SyntheticDataset(Dataset):
    """
    Synthetic dataset for training neural-symbolic models.
    
    Generates synthetic data with controllable complexity and patterns.
    """
    
    def __init__(self, 
                 num_samples: int = 1000,
                 input_dim: int = 384,
                 complexity: float = 0.5,
                 noise_level: float = 0.1,
                 pattern_type: str = 'linear'):
        """
        Initialize synthetic dataset.
        
        Args:
            num_samples: Number of samples in dataset
            input_dim: Dimensionality of input vectors
            complexity: Complexity of generated patterns (0.0 to 1.0)
            noise_level: Amount of Gaussian noise to add
            pattern_type: Type of pattern to generate ('linear', 'nonlinear', 'mixed')
        """
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.complexity = complexity
        self.noise_level = noise_level
        self.pattern_type = pattern_type
        
        # Generate synthetic data
        self.data = self._generate_data()
    
    def _deterministic_randn(self, size, seed_prefix=""):
        """Generate deterministic normal distribution using Box-Muller transform"""
        if isinstance(size, int):
            size = (size,)
        total_elements = 1
        for dim in size:
            total_elements *= dim
        
        result = np.zeros(total_elements, dtype=np.float32)
        for i in range(total_elements):
            seed_str = f"{seed_prefix}_{i}"
            u1 = ((zlib.adler32(str(seed_str + "_u1").encode('utf-8')) & 0xffffffff) % 10000) / 10000.0
            u2 = ((zlib.adler32(str(seed_str + "_u2").encode('utf-8')) & 0xffffffff) % 10000) / 10000.0
            u1 = max(u1, 1e-10)
            u2 = max(u2, 1e-10)
            z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            result[i] = z0
        
        return result.reshape(size)
    
    def _deterministic_rand(self, size, seed_prefix=""):
        """Generate deterministic uniform distribution between 0 and 1"""
        if isinstance(size, int):
            size = (size,)
        total_elements = 1
        for dim in size:
            total_elements *= dim
        
        result = np.zeros(total_elements, dtype=np.float32)
        for i in range(total_elements):
            seed_str = f"{seed_prefix}_{i}"
            value = ((zlib.adler32(str(seed_str).encode('utf-8')) & 0xffffffff) % 10000) / 10000.0
            result[i] = value
        
        return result.reshape(size)
    
    def _generate_data(self) -> np.ndarray:
        """Generate synthetic data with specified patterns"""
        data = np.zeros((self.num_samples, self.input_dim))
        
        # Generate base patterns
        if self.pattern_type == 'linear':
            # Linear combinations of random vectors
            num_components = max(2, int(self.complexity * 10))
            components = self._deterministic_randn((num_components, self.input_dim), "linear_components")
            coefficients = self._deterministic_rand((self.num_samples, num_components), "linear_coefficients")
            
            for i in range(self.num_samples):
                data[i] = np.dot(coefficients[i], components)
        
        elif self.pattern_type == 'nonlinear':
            # Nonlinear transformations of random vectors
            for i in range(self.num_samples):
                base = self._deterministic_randn(self.input_dim, f"nonlinear_base_{i}")
                # Apply nonlinear transformations
                data[i] = np.sin(base * self.complexity) + np.cos(base * (1 - self.complexity))
        
        elif self.pattern_type == 'mixed':
            # Mixture of linear and nonlinear patterns
            half_dim = self.input_dim // 2
            
            # Linear part
            num_components = max(2, int(self.complexity * 5))
            components = self._deterministic_randn((num_components, half_dim), "mixed_components")
            coefficients = self._deterministic_rand((self.num_samples, num_components), "mixed_coefficients")
            
            # Nonlinear part
            for i in range(self.num_samples):
                linear_part = np.dot(coefficients[i], components)
                nonlinear_base = self._deterministic_randn(self.input_dim - half_dim, f"mixed_nonlinear_base_{i}")
                nonlinear_part = np.tanh(nonlinear_base * self.complexity)
                
                data[i, :half_dim] = linear_part
                data[i, half_dim:] = nonlinear_part
        
        else:
            # Random data
            data = self._deterministic_randn((self.num_samples, self.input_dim), "random_data")
        
        # Add noise
        if self.noise_level > 0:
            noise = self._deterministic_randn(data.shape, "noise") * self.noise_level
            data += noise
        
        # Normalize data
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        
        return data.astype(np.float32)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.FloatTensor(self.data[idx])
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset"""
        return {
            'num_samples': self.num_samples,
            'input_dim': self.input_dim,
            'complexity': self.complexity,
            'noise_level': self.noise_level,
            'pattern_type': self.pattern_type,
            'data_shape': self.data.shape,
            'data_mean': float(self.data.mean()),
            'data_std': float(self.data.std())
        }

# Example usage and test function
def test_enhanced_neural_symbolic_model():
    """Test function for EnhancedNeuralSymbolicModel"""
    print("Testing EnhancedNeuralSymbolicModel...")
    
    # Create synthetic dataset
    dataset = SyntheticDataset(num_samples=100, input_dim=384, complexity=0.7)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize model
    model = EnhancedNeuralSymbolicModel(
        input_dim=384,
        symbolic_dim=256,
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        use_transformer=True
    )
    
    # Configure optimizer
    model.configure_optimizer(
        optimizer_type='adam',
        learning_rate=0.001,
        scheduler_type='reduce_on_plateau'
    )
    
    # Get model info
    model_info = model.get_model_info()
    print(f"Model Info: {json.dumps(model_info, indent=2, default=str)}")
    
    # Test forward pass
    test_batch = next(iter(data_loader))
    reconstructed, symbolic_features = model(test_batch)
    print(f"Input shape: {test_batch.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Symbolic features shape: {symbolic_features.shape}")
    
    # Test training step
    loss = model.train_step(test_batch)
    print(f"Training step loss: {loss:.6f}")
    
    # Test prediction
    test_data = test_batch[:5]  # First 5 samples
    reconstructed_np, symbolic_np = model.predict(test_data)
    print(f"Prediction - Reconstructed shape: {reconstructed_np.shape}")
    print(f"Prediction - Symbolic shape: {symbolic_np.shape}")
    
    print("EnhancedNeuralSymbolicModel test completed successfully!")

def test_joint_training_coordinator():
    """Test function for JointTrainingCoordinator"""
    print("\nTesting JointTrainingCoordinator...")
    
    # Create multiple models
    models = [
        EnhancedNeuralSymbolicModel(input_dim=384, symbolic_dim=256, hidden_dim=512),
        EnhancedNeuralSymbolicModel(input_dim=384, symbolic_dim=256, hidden_dim=512),
        EnhancedNeuralSymbolicModel(input_dim=384, symbolic_dim=256, hidden_dim=512)
    ]
    
    # Configure optimizers for each model
    for model in models:
        model.configure_optimizer(learning_rate=0.001)
    
    # Create synthetic datasets for each model
    train_loaders = []
    for i in range(len(models)):
        dataset = SyntheticDataset(
            num_samples=200,
            input_dim=384,
            complexity=0.5 + 0.2 * i,  # Varying complexity
            pattern_type='mixed'
        )
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        train_loaders.append(train_loader)
    
    # Initialize coordinator
    coordinator_config = {
        'communication_frequency': 3,
        'joint_loss_weight': 0.7,
        'coordination_strategy': 'average'
    }
    
    coordinator = JointTrainingCoordinator(models, coordinator_config)
    
    # Get coordinator info
    coordinator_info = coordinator.get_coordinator_info()
    print(f"Coordinator Info: {json.dumps(coordinator_info, indent=2, default=str)}")
    
    # Test coordination training (short version)
    print("Starting short joint training (2 epochs)...")
    training_history = coordinator.coordinate_training(train_loaders, epochs=2)
    
    print(f"Joint training completed. Joint loss history: {training_history['joint_loss']}")
    print("JointTrainingCoordinator test completed successfully!")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run tests
    test_enhanced_neural_symbolic_model()
    test_joint_training_coordinator()
    
    print("\nAll tests completed successfully!")
