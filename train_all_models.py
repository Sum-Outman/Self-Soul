#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to train all Self Soul AGI models from scratch
This script will initialize and start training for all registered models
"""
import os
import sys
import time
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.from_scratch_training import from_scratch_training_manager
from core.model_registry import get_model_registry


def main():
    """Main function to train all models from scratch"""
    print("=" * 80)
    print("      Self Soul AGI - All Models From Scratch Training      ")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nPreparing to train all models from scratch...\n")
    
    try:
        # Initialize training manager if not already initialized
        if not from_scratch_training_manager.initialized:
            init_result = from_scratch_training_manager.initialize()
            if not init_result["success"]:
                print(f"Error initializing training manager: {init_result.get('error', 'Unknown error')}")
                return 1
            print("Training manager initialized successfully.")
        
        # List all registered models first
        model_registry = get_model_registry()
        
        # Load all models
        print("Loading all models...")
        loaded_models = model_registry.load_all_models()
        print(f"Successfully loaded {len(loaded_models)} models")
        
        registered_models = model_registry.get_all_registered_models()
        
        print(f"Found {len(registered_models)} registered models:")
        for i, model_id in enumerate(registered_models, 1):
            print(f"  {i}. {model_id}")
        print()
        
        # Start training all models
        print("Starting from-scratch training for all models...")
        start_time = time.time()
        result = from_scratch_training_manager.initialize_all_models_from_scratch()
        
        if result["success"]:
            print(f"\n{result['message']}")
            
            # Display detailed results
            details = result.get('details', {})
            print(f"\nSuccessfully started training for {len(details.get('succeeded', []))} models:")
            for item in details.get('succeeded', []):
                print(f"  - {item['model_id']} (Task ID: {item['task_id']})")
            
            if details.get('failed', []):
                print(f"\nFailed to start training for {len(details.get('failed', []))} models:")
                for item in details.get('failed', []):
                    print(f"  - {item['model_id']}: {item['message']}")
            
            print(f"\nOperation completed in {time.time() - start_time:.2f} seconds")
            print(f"\nYou can check training status by running: python check_training_status.ps1")
        else:
            print(f"\nFailed to start training: {result.get('message', 'Unknown error')}")
            return 1
        
    except Exception as e:
        print(f"\nError during training initialization: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("All models training process has been initiated. Training will continue in the background.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
