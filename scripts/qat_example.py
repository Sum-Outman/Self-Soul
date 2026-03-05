#!/usr/bin/env python3
"""
Quantization-Aware Training (QAT) Example Script
量化感知训练示例脚本

This script demonstrates how to use the Quantization-Aware Training (QAT) 
features implemented in the AGI system. It shows:
1. How to enable QAT mode in the ModelRegistry
2. How to train models with QAT support
3. How to monitor QAT-specific metrics during training

此脚本演示了如何在AGI系统中使用量化感知训练功能。它展示了：
1. 如何在ModelRegistry中启用QAT模式
2. 如何使用QAT支持训练模型
3. 如何在训练期间监控QAT特定指标
"""

import sys
import os
import logging
import zlib
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _deterministic_randn(size, seed_prefix="default"):
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

def demonstrate_qat_basic():
    """演示基本的QAT功能
    Demonstrate basic QAT functionality
    """
    logger.info("=== 量化感知训练（QAT）基础演示 ===")
    logger.info("=== Quantization-Aware Training (QAT) Basic Demo ===")
    
    try:
        # 1. 导入必要的模块
        # Import necessary modules
        from core.model_registry import ModelRegistry, QATModelWrapper
        from core.training_manager import TrainingManager
        
        logger.info("1. 初始化模型注册表...")
        logger.info("1. Initializing Model Registry...")
        
        # 创建模型注册表实例
        registry = ModelRegistry()
        
        # 2. 启用QAT模式
        logger.info("2. 启用QAT模式...")
        logger.info("2. Enabling QAT mode...")
        
        registry.quantization_mode = 'qat'
        logger.info(f"量化模式已设置为: {registry.quantization_mode}")
        logger.info(f"Quantization mode set to: {registry.quantization_mode}")
        
        # 3. 配置QAT参数
        logger.info("3. 配置QAT参数...")
        logger.info("3. Configuring QAT parameters...")
        
        qat_config = {
            'observer_type': 'histogram',
            'quantization_scheme': 'per_tensor_affine',
            'training_steps': 500,
            'calibration_steps': 50,
            'fuse_modules': True,
            'fuse_patterns': [('conv', 'bn', 'relu'), ('linear', 'relu')]
        }
        
        registry.qat_config.update(qat_config)
        logger.info(f"QAT配置: {registry.qat_config}")
        logger.info(f"QAT configuration: {registry.qat_config}")
        
        # 4. 加载一个模型（例如语言模型）进行QAT演示
        logger.info("4. 加载语言模型进行QAT演示...")
        logger.info("4. Loading language model for QAT demo...")
        
        model_id = 'language'
        model = registry.get_model(model_id)
        
        if model is None:
            logger.error(f"无法加载模型: {model_id}")
            logger.error(f"Failed to load model: {model_id}")
            return False
        
        logger.info(f"成功加载模型: {model_id}")
        logger.info(f"Successfully loaded model: {model_id}")
        
        # 检查模型是否是QAT包装器
        if hasattr(model, '__class__') and model.__class__.__name__ == 'QATModelWrapper':
            logger.info(f"模型 {model_id} 已使用QAT包装器包装")
            logger.info(f"Model {model_id} is wrapped with QAT wrapper")
            
            # 演示QAT包装器功能
            logger.info("5. 演示QAT包装器功能...")
            logger.info("5. Demonstrating QAT wrapper functionality...")
            
            # 准备QAT训练
            model.prepare_qat()
            logger.info("QAT包装器已准备就绪")
            logger.info("QAT wrapper prepared")
            
            # 检查QAT状态
            qat_state = model.get_state()
            logger.info(f"QAT状态: 当前步数={qat_state.get('current_step')}, "
                      f"校准中={qat_state.get('is_calibrating')}, "
                      f"已量化={qat_state.get('is_quantized')}")
            logger.info(f"QAT state: current_step={qat_state.get('current_step')}, "
                      f"is_calibrating={qat_state.get('is_calibrating')}, "
                      f"is_quantized={qat_state.get('is_quantized')}")
            
            # 6. 创建一个简单的数据集进行演示
            logger.info("6. 创建简单数据集进行演示...")
            logger.info("6. Creating simple dataset for demo...")
            
            # 创建简单的合成数据
            batch_size = 4
            input_size = 10
            num_samples = 100
            
            synthetic_inputs = _deterministic_randn((num_samples, input_size), seed_prefix="synthetic_inputs")
            synthetic_targets = torch.randint(0, 2, (num_samples,))
            
            logger.info(f"合成数据集: 输入形状={synthetic_inputs.shape}, 目标形状={synthetic_targets.shape}")
            logger.info(f"Synthetic dataset: inputs shape={synthetic_inputs.shape}, targets shape={synthetic_targets.shape}")
            
            # 7. 演示训练步骤
            logger.info("7. 演示QAT训练步骤...")
            logger.info("7. Demonstrating QAT training step...")
            
            # 创建优化器和损失函数
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.CrossEntropyLoss()
            
            # 执行几个训练步骤
            num_demo_steps = 5
            for step in range(num_demo_steps):
                # 选择一个小批量
                idx = step % (num_samples // batch_size)
                start_idx = idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_inputs = synthetic_inputs[start_idx:end_idx]
                batch_targets = synthetic_targets[start_idx:end_idx]
                
                # 执行训练步骤
                outputs, loss_value = model.train_step(
                    (batch_inputs, batch_targets),
                    optimizer,
                    loss_fn
                )
                
                if loss_value is not None:
                    logger.info(f"  步骤 {step+1}: 损失={loss_value:.4f}")
                    logger.info(f"  Step {step+1}: loss={loss_value:.4f}")
                else:
                    logger.info(f"  步骤 {step+1}: 校准步骤 (无损失)")
                    logger.info(f"  Step {step+1}: calibration step (no loss)")
            
            # 8. 检查最终状态
            final_state = model.get_state()
            logger.info(f"最终QAT状态: 当前步数={final_state.get('current_step')}, "
                      f"校准中={final_state.get('is_calibrating')}, "
                      f"已量化={final_state.get('is_quantized')}")
            logger.info(f"Final QAT state: current_step={final_state.get('current_step')}, "
                      f"is_calibrating={final_state.get('is_calibrating')}, "
                      f"is_quantized={final_state.get('is_quantized')}")
            
        else:
            logger.warning(f"模型 {model_id} 不是QAT包装器，可能是由于配置问题")
            logger.warning(f"Model {model_id} is not a QAT wrapper, possibly due to configuration")
            
            # 检查量化模式
            logger.info(f"当前量化模式: {registry.quantization_mode}")
            logger.info(f"Current quantization mode: {registry.quantization_mode}")
            
            # 尝试直接创建QAT包装器
            logger.info("尝试直接创建QAT包装器...")
            logger.info("Attempting to create QAT wrapper directly...")
            
            qat_wrapper = QATModelWrapper(model, registry.qat_config)
            logger.info(f"成功创建QAT包装器: {type(qat_wrapper).__name__}")
            logger.info(f"Successfully created QAT wrapper: {type(qat_wrapper).__name__}")
        
        # 9. 演示TrainingManager中的QAT训练
        logger.info("8. 演示TrainingManager中的QAT训练...")
        logger.info("8. Demonstrating QAT training in TrainingManager...")
        
        training_manager = TrainingManager()
        
        # 检查train_with_qat方法是否存在
        if hasattr(training_manager, 'train_with_qat'):
            logger.info("TrainingManager具有train_with_qat方法")
            logger.info("TrainingManager has train_with_qat method")
            
            # 注意：实际训练可能需要更长时间，这里只是演示API调用
            logger.info("要使用完整的QAT训练，请调用:")
            logger.info("  training_manager.train_with_qat(")
            logger.info("      model_ids=['language', 'vision'],")
            logger.info("      epochs=10,")
            logger.info("      batch_size=32,")
            logger.info("      learning_rate=0.001")
            logger.info("  )")
            
        else:
            logger.warning("TrainingManager没有train_with_qat方法")
            logger.warning("TrainingManager does not have train_with_qat method")
        
        logger.info("=== QAT演示完成 ===")
        logger.info("=== QAT Demo Completed ===")
        
        return True
        
    except Exception as e:
        logger.error(f"QAT演示过程中出错: {e}", exc_info=True)
        logger.error(f"Error during QAT demonstration: {e}", exc_info=True)
        return False

def demonstrate_qat_advanced_features():
    """演示高级QAT功能
    Demonstrate advanced QAT features
    """
    logger.info("\n=== 高级QAT功能演示 ===")
    logger.info("\n=== Advanced QAT Features Demo ===")
    
    try:
        from core.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        
        # 1. 演示不同的量化模式
        logger.info("1. 演示不同的量化模式...")
        logger.info("1. Demonstrating different quantization modes...")
        
        quantization_modes = ['none', 'dynamic', 'qat']
        
        for mode in quantization_modes:
            registry.quantization_mode = mode
            logger.info(f"  量化模式 '{mode}':")
            logger.info(f"    - 启用量化: {registry.quantization_enabled}")
            logger.info(f"    - QAT启用: {registry.qat_config.get('enabled', False)}")
        
        # 2. 演示QAT配置选项
        logger.info("2. 演示QAT配置选项...")
        logger.info("2. Demonstrating QAT configuration options...")
        
        registry.quantization_mode = 'qat'
        
        # 不同的观察器类型
        observer_types = ['minmax', 'histogram']
        for obs_type in observer_types:
            registry.qat_config['observer_type'] = obs_type
            logger.info(f"  观察器类型 '{obs_type}': 已设置")
        
        # 不同的量化方案
        schemes = ['per_tensor_affine', 'per_channel_affine']
        for scheme in schemes:
            registry.qat_config['quantization_scheme'] = scheme
            logger.info(f"  量化方案 '{scheme}': 已设置")
        
        # 3. 演示性能优化组合
        logger.info("3. 演示性能优化组合...")
        logger.info("3. Demonstrating performance optimization combinations...")
        
        # 启用编译优化
        registry.compile_enabled = True
        logger.info(f"  编译优化启用: {registry.compile_enabled}")
        
        # 4. 演示核心模型配置
        logger.info("4. 演示核心模型配置...")
        logger.info("4. Demonstrating core models configuration...")
        
        core_models = ['manager', 'language', 'vision', 'audio', 'knowledge']
        registry.core_models = core_models
        logger.info(f"  核心模型: {registry.core_models}")
        
        logger.info("=== 高级QAT演示完成 ===")
        logger.info("=== Advanced QAT Demo Completed ===")
        
        return True
        
    except Exception as e:
        logger.error(f"高级QAT演示过程中出错: {e}", exc_info=True)
        logger.error(f"Error during advanced QAT demonstration: {e}", exc_info=True)
        return False

def main():
    """主函数
    Main function
    """
    logger.info("启动QAT示例脚本...")
    logger.info("Starting QAT example script...")
    
    # 演示基本QAT功能
    basic_success = demonstrate_qat_basic()
    
    if basic_success:
        # 演示高级QAT功能
        advanced_success = demonstrate_qat_advanced_features()
    
    logger.info("\n=== 总结 ===")
    logger.info("\n=== Summary ===")
    
    if basic_success:
        logger.info("✓ 基本QAT功能演示成功")
        logger.info("✓ Basic QAT functionality demo successful")
    else:
        logger.error("✗ 基本QAT功能演示失败")
        logger.error("✗ Basic QAT functionality demo failed")
    
    logger.info("\n使用说明:")
    logger.info("1. 要在训练中启用QAT，将ModelRegistry.quantization_mode设置为'qat'")
    logger.info("2. 使用training_manager.train_with_qat()进行QAT训练")
    logger.info("3. 监控qat_states以跟踪校准和量化进度")
    logger.info("\nUsage instructions:")
    logger.info("1. To enable QAT in training, set ModelRegistry.quantization_mode to 'qat'")
    logger.info("2. Use training_manager.train_with_qat() for QAT training")
    logger.info("3. Monitor qat_states to track calibration and quantization progress")
    
    return 0 if basic_success else 1

if __name__ == "__main__":
    sys.exit(main())