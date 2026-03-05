#!/usr/bin/env python3
"""
Example script demonstrating proper shebang and argparse usage.
This serves as a template for converting experimental scripts.

Usage:
    python example_with_argparse.py --model-type language --dataset data/train.json
    python example_with_argparse.py --help
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to sys.path to allow importing core modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import core modules (example)
try:
    from core.training_manager import TrainingManager
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")
    print("Make sure you're running from the project root directory.")
    CORE_AVAILABLE = False

def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train models with configurable parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--model-type',
        required=True,
        choices=['language', 'knowledge', 'manager', 'programming', 'planning', 'other'],
        help='Type of model to train'
    )
    
    parser.add_argument(
        '--dataset',
        required=True,
        type=Path,
        help='Path to training dataset (JSON format)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for optimizer'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/models'),
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Fraction of data to use for validation'
    )
    
    # Flags
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate arguments without actually training'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint if available'
    )
    
    return parser.parse_args()

def validate_arguments(args, logger):
    """Validate provided arguments."""
    errors = []
    
    # Check dataset exists
    if not args.dataset.exists():
        errors.append(f"Dataset file not found: {args.dataset}")
    
    # Check dataset is JSON
    if args.dataset.suffix.lower() != '.json':
        logger.warning(f"Dataset file may not be JSON: {args.dataset}")
    
    # Check output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.output_dir.is_dir():
        errors.append(f"Output directory is not a directory: {args.output_dir}")
    
    # Check numerical values
    if args.epochs <= 0:
        errors.append(f"Epochs must be positive: {args.epochs}")
    if args.batch_size <= 0:
        errors.append(f"Batch size must be positive: {args.batch_size}")
    if args.learning_rate <= 0:
        errors.append(f"Learning rate must be positive: {args.learning_rate}")
    if not (0 < args.validation_split < 1):
        errors.append(f"Validation split must be between 0 and 1: {args.validation_split}")
    
    return errors

def train_model(args, logger):
    """Main training function."""
    logger.info(f"Starting training for model type: {args.model_type}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Learning rate: {args.learning_rate}")
    
    if not CORE_AVAILABLE:
        logger.error("Core modules not available. Cannot proceed with training.")
        return False
    
    if args.dry_run:
        logger.info("Dry run mode - validation successful, skipping actual training")
        return True
    
    try:
        # Example training logic (replace with actual implementation)
        training_manager = TrainingManager()
        
        training_params = {
            "model_type": args.model_type,
            "dataset_path": str(args.dataset),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "validation_split": args.validation_split,
            "save_dir": str(args.output_dir / args.model_type),
            "resume": args.resume
        }
        
        logger.info(f"Training parameters: {training_params}")
        
        # Create training task
        task_id = training_manager.create_training_task(
            model_type=args.model_type,
            parameters=training_params,
            priority=1
        )
        
        if task_id:
            logger.info(f"Training task created: {task_id}")
            # Start training
            training_manager.start_training_task(task_id)
            logger.info("Training started successfully")
            return True
        else:
            logger.error("Failed to create training task")
            return False
            
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return False

def main():
    """Main entry point."""
    args = parse_arguments()
    logger = setup_logging(args.verbose)
    
    logger.debug(f"Arguments: {args}")
    
    # Validate arguments
    errors = validate_arguments(args, logger)
    if errors:
        for error in errors:
            logger.error(error)
        logger.error("Argument validation failed")
        sys.exit(1)
    
    # Execute training
    success = train_model(args, logger)
    
    if success:
        logger.info("Training completed successfully")
        sys.exit(0)
    else:
        logger.error("Training failed")
        sys.exit(1)

if __name__ == "__main__":
    main()