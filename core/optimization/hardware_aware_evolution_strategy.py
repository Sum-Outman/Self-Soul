#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬件感知演化策略 - Hardware-Aware Evolution Strategy

将硬件感知能力集成到演化策略中，根据当前硬件环境自动调整演化参数。
支持动态硬件适配和跨硬件优化。
"""

import sys
import os
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.evolution_strategies import (
        BaseEvolutionStrategy, 
        EvolutionStrategyConfig,
        EvolutionResult,
        EvolutionStrategyType
    )
    from core.optimization.hardware_aware_evolution import (
        HardwareAwareEvolutionModule,
        HardwareOptimizationLevel,
        HardwareType,
        create_hardware_aware_evolution_module
    )
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在项目根目录中运行此脚本")
    sys.exit(1)

logger = logging.getLogger(__name__)


class HardwareAwareEvolutionStrategy(BaseEvolutionStrategy):
    """硬件感知演化策略
    
    包装现有的演化策略，根据硬件特性自动调整参数。
    支持动态硬件适配和跨硬件优化。
    """
    
    def __init__(self, 
                 base_strategy: BaseEvolutionStrategy,
                 hardware_config: Optional[Dict[str, Any]] = None,
                 task_complexity: float = 0.5):
        """初始化硬件感知演化策略
        
        Args:
            base_strategy: 基础演化策略实例
            hardware_config: 硬件感知配置
            task_complexity: 任务复杂性 (0.0-1.0)
        """
        # 使用基础策略的配置
        super().__init__(base_strategy.config)
        
        self.base_strategy = base_strategy
        self.task_complexity = task_complexity
        
        # 创建硬件感知模块
        self.hardware_module = create_hardware_aware_evolution_module(hardware_config)
        
        # 获取硬件感知参数
        self.hardware_params = self.hardware_module.get_hardware_aware_evolution_parameters(task_complexity)
        
        # 调整基础策略的参数
        self._adjust_base_strategy_parameters()
        
        logger.info(f"硬件感知演化策略初始化完成")
        logger.info(f"  硬件类型: {self.hardware_module.hardware_type.value if self.hardware_module.hardware_type else 'unknown'}")
        logger.info(f"  优化级别: {self.hardware_module.config.optimization_level.value}")
    
    def _adjust_base_strategy_parameters(self) -> None:
        """根据硬件特性调整基础策略参数"""
        # 调整种群大小
        if "population_size" in self.hardware_params:
            self.base_strategy.config.population_size = self.hardware_params["population_size"]
        
        # 调整突变率
        if "mutation_rate" in self.hardware_params:
            self.base_strategy.config.mutation_rate = self.hardware_params["mutation_rate"]
        
        # 调整交叉率
        if "crossover_rate" in self.hardware_params:
            self.base_strategy.config.crossover_rate = self.hardware_params["crossover_rate"]
        
        # 对于PSO策略，调整惯性权重
        if hasattr(self.base_strategy.config, 'pso_inertia_weight'):
            # 根据硬件类型调整PSO参数
            if self.hardware_module.hardware_type in [HardwareType.HIGH_END_GPU, HardwareType.MID_RANGE_GPU]:
                self.base_strategy.config.pso_inertia_weight = 0.9  # GPU上使用更高的惯性
            else:
                self.base_strategy.config.pso_inertia_weight = 0.7  # CPU/边缘设备使用较低的惯性
        
        # 更新自适应参数
        if hasattr(self.base_strategy, 'update_adaptive_parameters'):
            self.base_strategy.update_adaptive_parameters()
    
    def initialize_population(self, population_size: Optional[int] = None) -> List[Any]:
        """初始化种群
        
        Args:
            population_size: 种群大小，如果为None则使用配置中的值
            
        Returns:
            初始种群
        """
        # 使用硬件感知的种群大小
        effective_size = population_size or self.hardware_params.get("population_size", self.config.population_size)
        
        # 调用基础策略的初始化方法
        return self.base_strategy.initialize_population(effective_size)
    
    def evaluate_fitness(self, individual: Any) -> float:
        """评估个体适应度
        
        Args:
            individual: 个体
            
        Returns:
            适应度分数
        """
        # 对于硬件感知策略，可以添加硬件特定的适应度评估
        base_fitness = self.base_strategy.evaluate_fitness(individual)
        
        # 根据硬件特性调整适应度
        adjusted_fitness = self._adjust_fitness_for_hardware(base_fitness, individual)
        
        return adjusted_fitness
    
    def _adjust_fitness_for_hardware(self, base_fitness: float, individual: Any) -> float:
        """根据硬件特性调整适应度
        
        Args:
            base_fitness: 基础适应度
            individual: 个体
            
        Returns:
            调整后的适应度
        """
        # 默认不调整
        adjusted_fitness = base_fitness
        
        # 检查个体是否包含架构信息
        if isinstance(individual, dict) and "architecture" in individual:
            architecture = individual["architecture"]
            
            # 根据硬件优化架构
            optimized_architecture = self.hardware_module.optimize_architecture_for_hardware(architecture)
            
            # 根据优化程度调整适应度
            # 这里可以添加更复杂的适应度调整逻辑
            adjustment_factor = 1.0
            
            # 例如：边缘设备优化可以加分
            if self.hardware_module.hardware_type == HardwareType.EDGE_DEVICE:
                if "edge_optimized" in optimized_architecture.get("optimizations", []):
                    adjustment_factor = 1.1  # 边缘优化加分10%
            
            adjusted_fitness = base_fitness * adjustment_factor
        
        return adjusted_fitness
    
    def evolve_population(self, population: List[Any], fitness_scores: Dict[str, float]) -> List[Any]:
        """演化种群
        
        Args:
            population: 当前种群
            fitness_scores: 适应度分数字典
            
        Returns:
            新的种群
        """
        # 在演化前动态调整参数（基于演化进度）
        self._dynamic_parameter_adjustment()
        
        # 调用基础策略的演化方法
        new_population = self.base_strategy.evolve_population(population, fitness_scores)
        
        return new_population
    
    def _dynamic_parameter_adjustment(self) -> None:
        """动态参数调整
        
        基于演化进度和硬件状态动态调整参数
        """
        # 获取当前硬件状态
        hardware_info = self.hardware_module.get_hardware_info()
        
        # 根据硬件利用率调整参数
        # 这里可以添加更复杂的动态调整逻辑
        if hasattr(self.base_strategy, 'config'):
            # 简单示例：根据演化进度调整突变率
            if self.generation > 0 and self.generation % 10 == 0:
                # 随着代数增加，逐渐降低突变率
                decay_factor = max(0.5, 1.0 - self.generation / 100.0)
                self.base_strategy.config.mutation_rate = self.hardware_params.get("mutation_rate", 0.1) * decay_factor
    
    def optimize_architecture(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """优化网络架构
        
        Args:
            architecture: 原始网络架构
            
        Returns:
            优化后的网络架构
        """
        return self.hardware_module.optimize_architecture_for_hardware(architecture)
    
    def get_hardware_aware_parameters(self) -> Dict[str, Any]:
        """获取硬件感知参数
        
        Returns:
            硬件感知参数
        """
        return self.hardware_params.copy()
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """获取硬件信息
        
        Returns:
            硬件信息
        """
        return self.hardware_module.get_hardware_info()
    
    def update_task_complexity(self, task_complexity: float) -> None:
        """更新任务复杂性
        
        Args:
            task_complexity: 新的任务复杂性 (0.0-1.0)
        """
        self.task_complexity = max(0.0, min(1.0, task_complexity))
        
        # 重新获取硬件感知参数
        self.hardware_params = self.hardware_module.get_hardware_aware_evolution_parameters(task_complexity)
        
        # 重新调整基础策略参数
        self._adjust_base_strategy_parameters()
        
        logger.info(f"任务复杂性更新为: {task_complexity}")
        logger.info(f"  新的种群大小: {self.base_strategy.config.population_size}")
        logger.info(f"  新的突变率: {self.base_strategy.config.mutation_rate:.3f}")


class HardwareAwareEvolutionStrategyFactory:
    """硬件感知演化策略工厂"""
    
    @staticmethod
    def create_strategy(
        strategy_type: EvolutionStrategyType,
        hardware_config: Optional[Dict[str, Any]] = None,
        task_complexity: float = 0.5,
        base_config: Optional[EvolutionStrategyConfig] = None
    ) -> HardwareAwareEvolutionStrategy:
        """创建硬件感知演化策略
        
        Args:
            strategy_type: 基础策略类型
            hardware_config: 硬件感知配置
            task_complexity: 任务复杂性
            base_config: 基础策略配置
            
        Returns:
            硬件感知演化策略实例
        """
        from core.evolution_strategies import EvolutionStrategyFactory
        
        # 创建基础策略
        base_strategy = EvolutionStrategyFactory.create_strategy(strategy_type, base_config)
        
        # 创建硬件感知包装器
        hardware_aware_strategy = HardwareAwareEvolutionStrategy(
            base_strategy=base_strategy,
            hardware_config=hardware_config,
            task_complexity=task_complexity
        )
        
        return hardware_aware_strategy
    
    @staticmethod
    def create_hardware_aware_config(
        hardware_type: HardwareType,
        optimization_level: HardwareOptimizationLevel = HardwareOptimizationLevel.BALANCED,
        task_complexity: float = 0.5
    ) -> Dict[str, Any]:
        """创建硬件感知配置
        
        Args:
            hardware_type: 硬件类型
            optimization_level: 优化级别
            task_complexity: 任务复杂性
            
        Returns:
            硬件感知配置
        """
        return {
            "hardware_type": hardware_type.value,
            "optimization_level": optimization_level.value,
            "task_complexity": task_complexity,
            "max_model_size_mb": 500.0,
            "max_memory_usage_gb": 32.0,
        }


# 测试代码
if __name__ == "__main__":
    import sys
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("硬件感知演化策略演示")
    print("=" * 80)
    
    try:
        print("\n1. 测试硬件检测和策略创建:")
        
        # 测试不同的优化级别
        optimization_levels = [
            ("最大性能", "max_performance"),
            ("平衡", "balanced"),
            ("功耗优先", "power_efficient"),
            ("边缘优化", "edge_optimized")
        ]
        
        for level_name, level_value in optimization_levels:
            print(f"\n  优化级别: {level_name}")
            
            # 创建硬件感知配置
            hardware_config = {
                "optimization_level": level_value
            }
            
            # 创建硬件感知遗传算法策略
            strategy = HardwareAwareEvolutionStrategyFactory.create_strategy(
                strategy_type=EvolutionStrategyType.GENETIC_ALGORITHM,
                hardware_config=hardware_config,
                task_complexity=0.5
            )
            
            # 获取硬件信息
            hardware_info = strategy.get_hardware_info()
            print(f"    检测到的硬件类型: {hardware_info['hardware_type']}")
            print(f"    内存: {hardware_info['features']['memory_gb']:.1f} GB")
            print(f"    计算单元: {hardware_info['features']['compute_units']}")
        
        print("\n2. 测试任务复杂性调整:")
        
        # 创建默认策略
        strategy = HardwareAwareEvolutionStrategyFactory.create_strategy(
            strategy_type=EvolutionStrategyType.GENETIC_ALGORITHM,
            task_complexity=0.3
        )
        
        hardware_params = strategy.get_hardware_aware_parameters()
        print(f"  任务复杂性 0.3:")
        print(f"    种群大小: {hardware_params.get('population_size', 'N/A')}")
        print(f"    突变率: {hardware_params.get('mutation_rate', 0.0):.3f}")
        
        # 更新任务复杂性
        strategy.update_task_complexity(0.8)
        hardware_params = strategy.get_hardware_aware_parameters()
        print(f"  任务复杂性 0.8:")
        print(f"    种群大小: {hardware_params.get('population_size', 'N/A')}")
        print(f"    突变率: {hardware_params.get('mutation_rate', 0.0):.3f}")
        
        print("\n3. 测试架构优化:")
        
        sample_architecture = {
            "type": "classification",
            "layers": [
                {"type": "linear", "size": 512, "activation": "relu"},
                {"type": "linear", "size": 256, "activation": "relu"},
                {"type": "linear", "size": 128, "activation": "sigmoid"}
            ]
        }
        
        optimized_architecture = strategy.optimize_architecture(sample_architecture)
        print(f"  原始架构层数: {len(sample_architecture['layers'])}")
        print(f"  优化后架构层数: {len(optimized_architecture['layers'])}")
        print(f"  优化标志: {optimized_architecture.get('optimizations', [])}")
        
        print("\n4. 测试不同演化策略:")
        
        strategy_types = [
            ("遗传算法", EvolutionStrategyType.GENETIC_ALGORITHM),
            ("粒子群优化", EvolutionStrategyType.PARTICLE_SWARM_OPTIMIZATION),
            ("差分进化", EvolutionStrategyType.DIFFERENTIAL_EVOLUTION)
        ]
        
        for strategy_name, strategy_type in strategy_types:
            print(f"\n  {strategy_name}:")
            
            strategy = HardwareAwareEvolutionStrategyFactory.create_strategy(
                strategy_type=strategy_type,
                task_complexity=0.5
            )
            
            hardware_params = strategy.get_hardware_aware_parameters()
            print(f"    种群大小: {hardware_params.get('population_size', 'N/A')}")
            print(f"    适应度评估可用: {hasattr(strategy.base_strategy, 'evaluate_fitness')}")
        
        print("\n✓ 硬件感知演化策略演示完成")
        print("\n总结:")
        print("  1. 成功创建硬件感知演化策略")
        print("  2. 支持实时硬件检测和参数调整")
        print("  3. 支持任务复杂性动态调整")
        print("  4. 支持多种演化策略类型")
        print("  5. 提供架构优化功能")
        
    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()