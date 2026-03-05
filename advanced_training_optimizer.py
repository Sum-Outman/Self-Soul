#!/usr/bin/env python3
"""
Advanced Training Optimizer for AGI Models

This module provides advanced PyTorch training optimizations that can be integrated
into any model's training pipeline. Features include mixed precision training,
learning rate schedulers, gradient accumulation, checkpointing, and more.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, List, Tuple, Callable
import time
import json
import logging
from datetime import datetime
import numpy as np
import zlib

class AdvancedTrainingOptimizer:
    """Advanced training optimization toolkit for PyTorch models."""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: Optional[optim.Optimizer] = None,
                 device: Optional[torch.device] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the advanced training optimizer.
        
        Args:
            model: PyTorch model to optimize
            optimizer: PyTorch optimizer (will be created if None)
            device: Torch device (auto-detected if None)
            config: Configuration dictionary with optimization settings
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device or self._auto_detect_device()
        self.config = config or self._default_config()
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize optimization components
        self.scaler = None
        self.scheduler = None
        self.early_stopping = None
        self.tensorboard_writer = None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = None
        self.best_model_state = None
        self.training_history = []
        
        self.logger.info(f"AdvancedTrainingOptimizer initialized on {self.device}")
    
    def _deterministic_randn(self, size, seed_prefix="default"):
        """Generate deterministic normal distribution using numpy RandomState"""
        import math
        if isinstance(size, int):
            size = (size,)
        total_elements = 1
        for dim in size:
            total_elements *= dim
        
        # Create deterministic seed from seed_prefix using adler32
        seed_hash = zlib.adler32(seed_prefix.encode('utf-8')) & 0xffffffff
        rng = np.random.RandomState(seed_hash)
        
        # Generate uniform random numbers
        u1 = rng.random_sample(total_elements)
        u2 = rng.random_sample(total_elements)
        
        # Apply Box-Muller transform
        u1 = np.maximum(u1, 1e-10)
        u2 = np.maximum(u2, 1e-10)
        z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)
        
        # Convert to torch tensor
        import torch
        result = torch.from_numpy(z0).float()
        
        return result.view(*size)
    
    def _auto_detect_device(self) -> torch.device:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            self.logger.info("Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU")
        return device
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default optimization configuration."""
        return {
            # Mixed precision
            "use_mixed_precision": True,
            "mixed_precision_enabled": torch.cuda.is_available(),
            
            # Learning rate scheduling
            "use_scheduler": True,
            "scheduler_type": "cosine",  # cosine, reduce_on_plateau, step, onecycle
            "lr": 0.001,
            "weight_decay": 1e-4,
            
            # Checkpointing
            "checkpoint_enabled": True,
            "checkpoint_dir": "./checkpoints",
            "save_best_only": True,
            "checkpoint_interval": 5,  # epochs
            
            # Early stopping
            "early_stopping_enabled": True,
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 1e-4,
            
            # Gradient handling
            "gradient_accumulation_steps": 1,
            "gradient_clip_enabled": True,
            "gradient_clip_value": 1.0,
            "gradient_clip_norm": None,
            
            # Training
            "num_epochs": 100,
            "batch_size": 32,
            "validation_split": 0.2,
            
            # Logging
            "tensorboard_logging": True,
            "log_dir": "./logs",
            "verbose": True,
            
            # Performance
            "profile_training": False,
            "use_cudnn_benchmark": True,
        }
    
    def setup_mixed_precision(self) -> None:
        """Setup mixed precision training with gradient scaling."""
        if self.config["use_mixed_precision"] and self.config["mixed_precision_enabled"]:
            if hasattr(torch.cuda, 'amp') and torch.cuda.is_available():
                from torch.cuda.amp import GradScaler, autocast
                self.scaler = GradScaler()
                self.autocast = autocast
                self.logger.info("Mixed precision training (AMP) enabled")
            else:
                self.logger.warning("Mixed precision not available, using FP32")
                self.config["use_mixed_precision"] = False
        else:
            self.logger.info("Mixed precision disabled")
    
    def setup_optimizer(self) -> None:
        """Setup optimizer with default or provided configuration."""
        if self.optimizer is None:
            # Default Adam optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"]
            )
            self.logger.info(f"Created AdamW optimizer with lr={self.config['lr']}")
        
        # Setup learning rate scheduler
        if self.config["use_scheduler"]:
            self.setup_scheduler()
    
    def setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        scheduler_type = self.config.get("scheduler_type", "cosine").lower()
        
        if scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["num_epochs"],
                eta_min=self.config["lr"] * 0.01  # 1% of initial LR
            )
            self.logger.info("Cosine annealing scheduler configured")
            
        elif scheduler_type == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=self.config["verbose"]
            )
            self.logger.info("ReduceLROnPlateau scheduler configured")
            
        elif scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
            self.logger.info("StepLR scheduler configured")
            
        elif scheduler_type == "onecycle":
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config["lr"],
                total_steps=self.config["num_epochs"],
                verbose=self.config["verbose"]
            )
            self.logger.info("OneCycleLR scheduler configured")
        else:
            self.logger.warning(f"Unknown scheduler type: {scheduler_type}")
            self.config["use_scheduler"] = False
    
    def setup_early_stopping(self) -> None:
        """Setup early stopping mechanism."""
        if self.config["early_stopping_enabled"]:
            self.early_stopping = {
                "patience": self.config["early_stopping_patience"],
                "min_delta": self.config["early_stopping_min_delta"],
                "counter": 0,
                "best_loss": float('inf'),
                "best_epoch": 0,
                "should_stop": False
            }
            self.logger.info(f"Early stopping enabled (patience={self.config['early_stopping_patience']})")
    
    def setup_checkpointing(self) -> None:
        """Setup model checkpointing."""
        if self.config["checkpoint_enabled"]:
            checkpoint_dir = self.config["checkpoint_dir"]
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.logger.info(f"Checkpointing enabled, saving to {checkpoint_dir}")
    
    def setup_tensorboard(self) -> None:
        """Setup TensorBoard logging if available."""
        if self.config["tensorboard_logging"]:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = os.path.join(self.config["log_dir"], 
                                      f"{self.model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(log_dir, exist_ok=True)
                self.tensorboard_writer = SummaryWriter(log_dir=log_dir)
                self.logger.info(f"TensorBoard logging enabled, logs at {log_dir}")
            except ImportError:
                self.logger.warning("TensorBoard not available, skipping")
                self.config["tensorboard_logging"] = False
    
    def setup_performance_optimizations(self) -> None:
        """Setup performance optimizations."""
        if self.config["use_cudnn_benchmark"] and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.logger.info("CuDNN benchmark enabled for faster convolutions")
    
    def setup_all(self) -> None:
        """Setup all optimization components."""
        self.logger.info("Setting up advanced training optimizations...")
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizations in order
        self.setup_optimizer()
        self.setup_mixed_precision()
        self.setup_scheduler()
        self.setup_early_stopping()
        self.setup_checkpointing()
        self.setup_tensorboard()
        self.setup_performance_optimizations()
        
        self.logger.info("All optimizations setup complete")
    
    def train_step(self, 
                   batch: Any, 
                   loss_fn: Callable,
                   metrics_fn: Optional[Callable] = None) -> Dict[str, float]:
        """
        Perform a single training step with optimizations.
        
        Args:
            batch: Training batch
            loss_fn: Loss function
            metrics_fn: Optional metrics function
            
        Returns:
            Dictionary with loss and metrics
        """
        self.model.train()
        
        # Unpack batch if needed
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        else:
            inputs = batch.to(self.device)
            targets = None
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if self.config["use_mixed_precision"] and self.scaler is not None:
            with self.autocast():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets) if targets is not None else loss_fn(outputs)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if self.current_step % self.config["gradient_accumulation_steps"] == 0:
                # Gradient clipping
                if self.config["gradient_clip_enabled"]:
                    self.scaler.unscale_(self.optimizer)
                    if self.config["gradient_clip_norm"]:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config["gradient_clip_norm"]
                        )
                    elif self.config["gradient_clip_value"]:
                        torch.nn.utils.clip_grad_value_(
                            self.model.parameters(),
                            self.config["gradient_clip_value"]
                        )
                
                # Optimizer step with scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            # Standard FP32 training
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets) if targets is not None else loss_fn(outputs)
            loss.backward()
            
            # Gradient accumulation
            if self.current_step % self.config["gradient_accumulation_steps"] == 0:
                # Gradient clipping
                if self.config["gradient_clip_enabled"]:
                    if self.config["gradient_clip_norm"]:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config["gradient_clip_norm"]
                        )
                    elif self.config["gradient_clip_value"]:
                        torch.nn.utils.clip_grad_value_(
                            self.model.parameters(),
                            self.config["gradient_clip_value"]
                        )
                
                # Optimizer step
                self.optimizer.step()
        
        # Calculate metrics
        metrics = {"loss": loss.item()}
        if metrics_fn is not None:
            metrics.update(metrics_fn(outputs, targets))
        
        return metrics
    
    def validate(self, 
                 val_loader: torch.utils.data.DataLoader,
                 loss_fn: Callable,
                 metrics_fn: Optional[Callable] = None) -> Dict[str, float]:
        """
        Perform validation.
        
        Args:
            val_loader: Validation data loader
            loss_fn: Loss function
            metrics_fn: Optional metrics function
            
        Returns:
            Dictionary with validation loss and metrics
        """
        self.model.eval()
        total_loss = 0
        all_metrics = {}
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = None
                
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets) if targets is not None else loss_fn(outputs)
                total_loss += loss.item()
                
                if metrics_fn is not None:
                    batch_metrics = metrics_fn(outputs, targets)
                    for key, value in batch_metrics.items():
                        if key not in all_metrics:
                            all_metrics[key] = []
                        all_metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {"val_loss": total_loss / len(val_loader)}
        for key, values in all_metrics.items():
            avg_metrics[f"val_{key}"] = np.mean(values)
        
        return avg_metrics
    
    def train(self,
              train_loader: torch.utils.data.DataLoader,
              val_loader: Optional[torch.utils.data.DataLoader] = None,
              loss_fn: Optional[Callable] = None,
              metrics_fn: Optional[Callable] = None,
              num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Main training loop with all optimizations.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            loss_fn: Loss function (default: nn.MSELoss)
            metrics_fn: Metrics function (optional)
            num_epochs: Number of epochs (overrides config)
            
        Returns:
            Training results dictionary
        """
        # Setup if not already done
        if self.optimizer is None:
            self.setup_all()
        
        # Use provided or default values
        loss_fn = loss_fn or nn.MSELoss()
        num_epochs = num_epochs or self.config["num_epochs"]
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "epoch_times": []
        }
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            epoch_train_loss = 0
            self.model.train()
            
            for batch_idx, batch in enumerate(train_loader):
                self.current_step = batch_idx
                metrics = self.train_step(batch, loss_fn, metrics_fn)
                epoch_train_loss += metrics["loss"]
                
                # Log batch progress
                if self.config["verbose"] and batch_idx % 10 == 0:
                    self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {metrics['loss']:.4f}")
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_metrics = self.validate(val_loader, loss_fn, metrics_fn)
                val_loss = val_metrics["val_loss"]
                history["val_loss"].append(val_loss)
            else:
                val_loss = avg_train_loss
                history["val_loss"].append(val_loss)
            
            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]['lr']
            history["learning_rate"].append(current_lr)
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping check
            if self.early_stopping is not None:
                if val_loss < self.early_stopping["best_loss"] - self.early_stopping["min_delta"]:
                    self.early_stopping["best_loss"] = val_loss
                    self.early_stopping["best_epoch"] = epoch
                    self.early_stopping["counter"] = 0
                    
                    # Save best model
                    if self.config["save_best_only"]:
                        self.best_model_state = self.model.state_dict().copy()
                        self.logger.info(f"New best model at epoch {epoch+1} with val_loss={val_loss:.4f}")
                else:
                    self.early_stopping["counter"] += 1
                    
                    if self.early_stopping["counter"] >= self.early_stopping["patience"]:
                        self.early_stopping["should_stop"] = True
                        self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
            
            # Checkpointing
            if self.config["checkpoint_enabled"]:
                if (epoch + 1) % self.config["checkpoint_interval"] == 0:
                    self.save_checkpoint(epoch, val_loss)
            
            # TensorBoard logging
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('Loss/train', avg_train_loss, epoch)
                self.tensorboard_writer.add_scalar('Loss/val', val_loss, epoch)
                self.tensorboard_writer.add_scalar('LearningRate', current_lr, epoch)
            
            # Epoch timing
            epoch_time = time.time() - epoch_start_time
            history["epoch_times"].append(epoch_time)
            
            # Log epoch results
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                           f"Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, "
                           f"LR: {current_lr:.6f}, "
                           f"Time: {epoch_time:.2f}s")
            
            # Check for early stopping
            if self.early_stopping is not None and self.early_stopping["should_stop"]:
                break
        
        # Restore best model if early stopping was used
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info(f"Restored best model from epoch {self.early_stopping['best_epoch'] + 1}")
        
        # Close TensorBoard writer
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
        
        # Prepare results
        results = {
            "model": self.model,
            "history": history,
            "best_val_loss": self.early_stopping["best_loss"] if self.early_stopping else val_loss,
            "total_epochs": self.current_epoch + 1,
            "final_loss": val_loss,
            "config": self.config
        }
        
        self.logger.info(f"Training completed. Best val loss: {results['best_val_loss']:.4f}")
        return results
    
    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        if not self.config["checkpoint_enabled"]:
            return
        
        checkpoint_dir = self.config["checkpoint_dir"]
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{self.model.__class__.__name__}_epoch{epoch+1}_loss{val_loss:.4f}.pt"
        )
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'history': self.training_history
        }
        
        # Save scaler state if using mixed precision
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint.get('history', [])
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} with val_loss={checkpoint['val_loss']:.4f}")
        return checkpoint
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of enabled optimizations."""
        summary = {
            "device": str(self.device),
            "mixed_precision": self.config["use_mixed_precision"] and self.config["mixed_precision_enabled"],
            "learning_rate_scheduler": self.config["use_scheduler"] and self.scheduler is not None,
            "scheduler_type": self.config.get("scheduler_type", "none"),
            "early_stopping": self.config["early_stopping_enabled"],
            "gradient_accumulation": self.config["gradient_accumulation_steps"] > 1,
            "gradient_clipping": self.config["gradient_clip_enabled"],
            "checkpointing": self.config["checkpoint_enabled"],
            "tensorboard": self.config["tensorboard_logging"] and self.tensorboard_writer is not None,
            "performance_optimizations": self.config["use_cudnn_benchmark"],
            "current_learning_rate": self.optimizer.param_groups[0]['lr'] if self.optimizer else None
        }
        return summary


def example_usage():
    """Example usage of AdvancedTrainingOptimizer."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 50)
            self.fc2 = nn.Linear(50, 20)
            self.fc3 = nn.Linear(20, 1)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Generate dummy data
    torch.manual_seed(42)
    
    # Deterministic random number generator
    def _deterministic_randn(size, seed_prefix="default"):
        import math
        import numpy as np
        import zlib
        if isinstance(size, int):
            size = (size,)
        total_elements = 1
        for dim in size:
            total_elements *= dim
        
        # Create deterministic seed from seed_prefix using adler32
        seed_hash = zlib.adler32(seed_prefix.encode('utf-8')) & 0xffffffff
        rng = np.random.RandomState(seed_hash)
        
        # Generate uniform random numbers
        u1 = rng.random_sample(total_elements)
        u2 = rng.random_sample(total_elements)
        
        # Apply Box-Muller transform
        u1 = np.maximum(u1, 1e-10)
        u2 = np.maximum(u2, 1e-10)
        z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)
        
        # Convert to torch tensor
        import torch
        result = torch.from_numpy(z0).float()
        
        return result.view(*size)
    
    X_train = _deterministic_randn((1000, 10), seed_prefix="X_train")
    y_train = _deterministic_randn((1000, 1), seed_prefix="y_train")
    X_val = _deterministic_randn((200, 10), seed_prefix="X_val")
    y_val = _deterministic_randn((200, 1), seed_prefix="y_val")
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create model and optimizer
    model = SimpleModel()
    
    # Create advanced training optimizer
    config = {
        "use_mixed_precision": True,
        "scheduler_type": "cosine",
        "early_stopping_enabled": True,
        "gradient_accumulation_steps": 2,
        "num_epochs": 20,
        "verbose": True
    }
    
    optimizer = AdvancedTrainingOptimizer(model, config=config)
    
    # Setup all optimizations
    optimizer.setup_all()
    
    # Print optimization summary
    summary = optimizer.get_optimization_summary()
    print("\nOptimization Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Train the model
    print("\nStarting training...")
    results = optimizer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=nn.MSELoss(),
        num_epochs=10
    )
    
    print(f"\nTraining completed. Best val loss: {results['best_val_loss']:.4f}")
    return results


if __name__ == "__main__":
    example_usage()