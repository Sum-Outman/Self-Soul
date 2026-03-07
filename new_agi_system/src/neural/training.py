"""
Neural Network Training

Training systems for neural network components in the
unified cognitive architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, List, Tuple, Callable
import logging
import time
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Training phase enumeration"""
    PRETRAINING = "pretraining"
    FINE_TUNING = "fine_tuning"
    CONTINUAL_LEARNING = "continual_learning"
    META_LEARNING = "meta_learning"


@dataclass
class TrainingMetrics:
    """Training metrics container"""
    loss: float
    accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: int = 0
    step: int = 0
    timestamp: float = 0.0


class NeuralTrainer:
    """
    Neural network trainer for cognitive components.
    
    Provides standardized training procedures for various
    neural network architectures in the unified system.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize neural trainer.
        
        Args:
            device: Training device ('cpu', 'cuda', or 'auto')
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.current_phase = TrainingPhase.PRETRAINING
        self.metrics_history = []
        
        logger.info(f"NeuralTrainer initialized for {self.device}")
    
    def train_epoch(self, model: nn.Module, train_loader, 
                   criterion: nn.Module, optimizer: optim.Optimizer,
                   phase: TrainingPhase = TrainingPhase.FINE_TUNING,
                   clip_grad_norm: Optional[float] = None) -> TrainingMetrics:
        """
        Train model for one epoch.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            phase: Training phase
            clip_grad_norm: Gradient clipping norm
            
        Returns:
            Training metrics for the epoch
        """
        model.train()
        model.to(self.device)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            # Optimization step
            optimizer.step()
            
            # Calculate accuracy
            if output.shape == target.shape:
                # For regression tasks
                pred = output
            else:
                # For classification tasks
                pred = output.argmax(dim=1, keepdim=True)
                target = target.view_as(pred)
                correct = pred.eq(target).sum().item()
                total_correct += correct
            
            total_loss += loss.item()
            total_samples += len(data)
            
            # Log progress every 100 batches
            if batch_idx % 100 == 0:
                logger.info(f"Phase {phase.value}: Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Calculate metrics
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else None
        
        metrics = TrainingMetrics(
            loss=avg_loss,
            accuracy=accuracy,
            learning_rate=optimizer.param_groups[0]['lr'] if optimizer.param_groups else None,
            epoch=len(self.metrics_history) + 1,
            step=(len(self.metrics_history) + 1) * len(train_loader),
            timestamp=time.time()
        )
        
        self.metrics_history.append(metrics)
        self.current_phase = phase
        
        logger.info(f"Epoch completed: loss={avg_loss:.4f}, accuracy={accuracy:.4f if accuracy else 'N/A'}, time={epoch_time:.2f}s")
        
        return metrics
    
    def validate(self, model: nn.Module, val_loader, criterion: nn.Module) -> TrainingMetrics:
        """
        Validate model performance.
        
        Args:
            model: Model to validate
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Validation metrics
        """
        model.eval()
        model.to(self.device)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                
                if output.shape == target.shape:
                    # For regression tasks
                    pred = output
                else:
                    # For classification tasks
                    pred = output.argmax(dim=1, keepdim=True)
                    target = target.view_as(pred)
                    correct = pred.eq(target).sum().item()
                    total_correct += correct
                
                total_samples += len(data)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else None
        
        metrics = TrainingMetrics(
            loss=avg_loss,
            accuracy=accuracy,
            timestamp=time.time()
        )
        
        logger.info(f"Validation: loss={avg_loss:.4f}, accuracy={accuracy:.4f if accuracy else 'N/A'}")
        
        return metrics
    
    def create_optimizer(self, model: nn.Module, phase: TrainingPhase) -> optim.Optimizer:
        """
        Create appropriate optimizer for training phase.
        
        Args:
            model: Model to optimize
            phase: Training phase
            
        Returns:
            Optimizer
        """
        if phase == TrainingPhase.PRETRAINING:
            # AdamW for pretraining
            optimizer = optim.AdamW(
                model.parameters(),
                lr=1e-4,
                betas=(0.9, 0.999),
                weight_decay=0.01
            )
        elif phase == TrainingPhase.FINE_TUNING:
            # SGD for fine-tuning
            optimizer = optim.SGD(
                model.parameters(),
                lr=0.01,
                momentum=0.9,
                weight_decay=0.0005
            )
        elif phase == TrainingPhase.CONTINUAL_LEARNING:
            # Adam for continual learning
            optimizer = optim.Adam(
                model.parameters(),
                lr=1e-3,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:  # META_LEARNING
            # Adam with lower learning rate
            optimizer = optim.Adam(
                model.parameters(),
                lr=1e-4,
                betas=(0.9, 0.999)
            )
        
        logger.info(f"Created {type(optimizer).__name__} optimizer for {phase.value}")
        return optimizer
    
    def create_scheduler(self, optimizer: optim.Optimizer, phase: TrainingPhase) -> optim.lr_scheduler._LRScheduler:
        """
        Create learning rate scheduler for training phase.
        
        Args:
            optimizer: Optimizer
            phase: Training phase
            
        Returns:
            Learning rate scheduler
        """
        if phase == TrainingPhase.PRETRAINING:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2
            )
        elif phase == TrainingPhase.FINE_TUNING:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.5
            )
        elif phase == TrainingPhase.CONTINUAL_LEARNING:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:  # META_LEARNING
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=50
            )
        
        logger.info(f"Created {type(scheduler).__name__} scheduler for {phase.value}")
        return scheduler
    
    def get_training_history(self) -> List[TrainingMetrics]:
        """Get training history"""
        return self.metrics_history
    
    def clear_history(self):
        """Clear training history"""
        self.metrics_history.clear()
        logger.info("Cleared training history")


class MetaTrainer:
    """
    Meta-learning trainer for cognitive components.
    
    Implements meta-learning algorithms for fast adaptation
    and continuous learning in the unified architecture.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize meta-trainer.
        
        Args:
            device: Training device
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.inner_lr = 0.01
        self.meta_lr = 0.001
        self.num_inner_steps = 5
        
        logger.info(f"MetaTrainer initialized for {self.device}")
    
    def meta_train_step(self, model: nn.Module, support_set, query_set,
                       criterion: nn.Module) -> Dict[str, float]:
        """
        Perform one meta-training step (MAML-style).
        
        Args:
            model: Model to meta-train
            support_set: Support set for inner loop
            query_set: Query set for outer loop
            criterion: Loss function
            
        Returns:
            Meta-training metrics
        """
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Inner loop: adapt to support set
        adapted_params = self._inner_loop(model, support_set, criterion)
        
        # Outer loop: evaluate on query set with adapted params
        meta_loss = self._outer_loop(model, adapted_params, query_set, criterion)
        
        # Update original parameters with meta-gradient
        self._meta_update(model, original_params, meta_loss)
        
        metrics = {
            'meta_loss': meta_loss.item(),
            'inner_lr': self.inner_lr,
            'meta_lr': self.meta_lr
        }
        
        logger.info(f"Meta-training step completed: meta_loss={meta_loss.item():.4f}")
        
        return metrics
    
    def _inner_loop(self, model: nn.Module, support_set, criterion: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Inner loop adaptation (fast weights).
        
        Args:
            model: Model to adapt
            support_set: Support set data
            criterion: Loss function
            
        Returns:
            Adapted parameters
        """
        # Clone original parameters
        adapted_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Perform few steps of gradient descent on support set
        for step in range(self.num_inner_steps):
            # Forward pass on support set
            data, target = support_set
            data, target = data.to(self.device), target.to(self.device)
            
            # Compute loss with current adapted params
            loss = self._compute_loss_with_params(model, adapted_params, data, target, criterion)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
            
            # Update adapted parameters
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    adapted_params[name] = param - self.inner_lr * grad
        
        return adapted_params
    
    def _outer_loop(self, model: nn.Module, adapted_params: Dict[str, torch.Tensor],
                   query_set, criterion: nn.Module) -> torch.Tensor:
        """
        Outer loop evaluation.
        
        Args:
            model: Model
            adapted_params: Adapted parameters from inner loop
            query_set: Query set data
            criterion: Loss function
            
        Returns:
            Meta-loss on query set
        """
        data, target = query_set
        data, target = data.to(self.device), target.to(self.device)
        
        # Compute loss with adapted parameters
        loss = self._compute_loss_with_params(model, adapted_params, data, target, criterion)
        
        return loss
    
    def _compute_loss_with_params(self, model: nn.Module, params: Dict[str, torch.Tensor],
                                 data: torch.Tensor, target: torch.Tensor,
                                 criterion: nn.Module) -> torch.Tensor:
        """
        Compute loss with specific parameters.
        
        Args:
            model: Model
            params: Parameters to use
            data: Input data
            target: Target data
            criterion: Loss function
            
        Returns:
            Loss
        """
        # Save original parameters
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Temporarily set model parameters
        for name, param in model.named_parameters():
            if name in params:
                param.data.copy_(params[name])
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Restore original parameters
        for name, param in model.named_parameters():
            if name in original_params:
                param.data.copy_(original_params[name])
        
        return loss
    
    def _meta_update(self, model: nn.Module, original_params: Dict[str, torch.Tensor],
                    meta_loss: torch.Tensor):
        """
        Update model parameters with meta-gradient.
        
        Args:
            model: Model
            original_params: Original parameters before inner loop
            meta_loss: Meta-loss
        """
        # Compute meta-gradient
        meta_grads = torch.autograd.grad(meta_loss, model.parameters(), allow_unused=True)
        
        # Update parameters
        with torch.no_grad():
            for (name, param), grad in zip(model.named_parameters(), meta_grads):
                if grad is not None:
                    param.data = original_params[name] - self.meta_lr * grad


# 简单测试
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试NeuralTrainer
    trainer = NeuralTrainer()
    print(f"NeuralTrainer initialized for: {trainer.device}")
    
    # 测试MetaTrainer
    meta_trainer = MetaTrainer()
    print(f"MetaTrainer initialized for: {meta_trainer.device}")
    
    print("Training module test complete.")