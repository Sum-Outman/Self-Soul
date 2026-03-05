#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演化模块实现 - Evolution Module Implementation

实现IEvolutionModule接口，提供神经网络架构的自主演化功能。
基于现有的ArchitectureEvolutionEngine，适配到标准化接口。

主要功能：
1. 架构演化：evolve_architecture - 演化神经网络架构以适应任务需求
2. 状态管理：get_evolution_status - 获取演化状态信息
3. 演化控制：stop_evolution, rollback_evolution - 控制演化过程
4. 统计监控：get_evolution_statistics - 获取演化统计信息
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
import threading

from core.module_interfaces import (
    IEvolutionModule,
    EvolutionResult,
    EvolutionParameters,
)
from core.architecture_evolution_engine import (
    ArchitectureEvolutionEngine,
    NetworkArchitecture,
)

# 配置日志
logger = logging.getLogger(__name__)


class EvolutionModule(IEvolutionModule):
    """
    演化模块实现类
    实现IEvolutionModule接口，提供完整的架构演化功能
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化演化模块

        Args:
            config: 配置字典，可选
        """
        self.logger = logger
        self.config = config or {}

        # 演化引擎实例
        self.evolution_engine = ArchitectureEvolutionEngine(config)

        # 演化状态
        self.evolution_active = False
        self.evolution_thread = None
        self.stop_requested = False

        # 演化历史
        self.evolution_history: List[Dict[str, Any]] = []
        self.max_history_size = self.config.get("max_history_size", 100)

        # 统计信息
        self.statistics = {
            "total_evolutions": 0,
            "successful_evolutions": 0,
            "failed_evolutions": 0,
            "total_evolution_time": 0.0,
            "average_evolution_time": 0.0,
            "best_accuracy": 0.0,
            "best_efficiency": 0.0,
        }

        # 回滚点
        self.rollback_points: Dict[int, Dict[str, Any]] = {}
        self.next_rollback_id = 1

        self.logger.info("演化模块初始化完成")

    def evolve_architecture(
        self,
        base_architecture: Dict[str, Any],
        performance_targets: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> EvolutionResult:
        """
        演化神经网络架构

        Args:
            base_architecture: 基础架构
            performance_targets: 性能目标
            constraints: 约束条件

        Returns:
            演化结果
        """
        start_time = time.time()
        evolution_id = f"evo_{int(start_time)}_{len(self.evolution_history)}"

        try:
            self.logger.info(f"开始演化架构 (ID: {evolution_id})")
            self.logger.info(f"性能目标: {performance_targets}")

            # 检查演化是否已在运行
            if self.evolution_active:
                self.logger.warning("演化已在运行中，等待当前演化完成")
                return EvolutionResult(
                    success=False,
                    evolved_architecture={},
                    performance_metrics={},
                    generation_info={"error": "evolution_already_active"},
                    error_message="演化已在运行中，请等待当前演化完成",
                )

            # 创建演化任务需求
            task_requirements = {
                "architecture_type": base_architecture.get("type", "generic"),
                "performance_targets": performance_targets,
                "constraints": constraints or {},
            }

            # 设置性能反馈（如果没有提供，使用默认值）
            performance_feedback = performance_targets.copy()

            # 设置资源约束
            resource_constraints = (
                constraints.get("resource_constraints", {}) if constraints else {}
            )

            # 开始演化
            self.evolution_active = True
            self.stop_requested = False

            # 在实际系统中，这里应该启动一个线程进行演化
            # 为简化，我们同步调用演化引擎
            evolved_architecture = self.evolution_engine.evolve_architecture(
                task_requirements=task_requirements,
                performance_feedback=performance_feedback,
                resource_constraints=resource_constraints,
            )

            # 检查熔断机制
            fuse_status = self.evolution_engine.get_fuse_status()
            if fuse_status.get("fuse_triggered", False):
                self.logger.warning(
                    f"熔断机制已触发: {fuse_status.get('last_failure_reason', 'unknown')}"
                )

                # 尝试获取最佳架构作为回退
                best_arch = self.evolution_engine.get_best_architecture()
                if best_arch:
                    evolved_architecture = best_arch
                    self.logger.info("使用最佳架构作为回退方案")
                else:
                    self.logger.error("熔断触发且无可用最佳架构")
                    return EvolutionResult(
                        success=False,
                        evolved_architecture={},
                        performance_metrics={},
                        generation_info={
                            "error": "fuse_triggered",
                            "fuse_status": fuse_status,
                        },
                        error_message=f"演化熔断触发: {fuse_status.get('last_failure_reason', 'unknown')}",
                    )

            # 获取演化状态
            evolution_status = self.evolution_engine.get_evolution_status()

            # 创建演化结果
            evolved_arch_dict = self._architecture_to_dict(evolved_architecture)

            # 计算性能指标
            performance_metrics = {
                "accuracy": evolved_architecture.performance_metrics.get(
                    "accuracy", 0.0
                ),
                "efficiency": evolved_architecture.performance_metrics.get(
                    "efficiency", 0.0
                ),
                "robustness": evolved_architecture.performance_metrics.get(
                    "robustness", 0.0
                ),
                "adaptability": evolved_architecture.performance_metrics.get(
                    "adaptability", 0.0
                ),
                "resource_usage": evolved_architecture.performance_metrics.get(
                    "resource_usage", 0.0
                ),
                "overall_score": evolved_architecture.fitness_score,
            }

            # 创建代信息
            generation_info = {
                "generation_id": evolution_id,
                "generation_number": evolution_status.get("total_generations", 0),
                "population_size": evolution_status.get("population_size", 0),
                "best_fitness": evolution_status.get("best_fitness", 0.0),
                "evaluation_count": evolution_status.get("evaluation_count", 0),
                "total_computation_time": evolution_status.get(
                    "total_computation_time", 0.0
                ),
                "timestamp": time.time(),
            }

            # 保存回滚点
            rollback_id = self._create_rollback_point(
                architecture=evolved_arch_dict,
                performance_metrics=performance_metrics,
                generation_info=generation_info,
            )
            generation_info["rollback_id"] = rollback_id

            # 更新统计信息
            self._update_statistics(
                performance_metrics, time.time() - start_time, success=True
            )

            # 记录演化历史
            self._record_evolution_history(
                evolution_id=evolution_id,
                base_architecture=base_architecture,
                evolved_architecture=evolved_arch_dict,
                performance_metrics=performance_metrics,
                generation_info=generation_info,
                success=True,
            )

            result = EvolutionResult(
                success=True,
                evolved_architecture=evolved_arch_dict,
                performance_metrics=performance_metrics,
                generation_info=generation_info,
            )

            self.logger.info(
                f"演化完成 (ID: {evolution_id}, 耗时: {time.time() - start_time:.2f}s)"
            )
            return result

        except Exception as e:
            self.logger.error(f"演化失败: {str(e)}", exc_info=True)

            # 更新失败统计
            self._update_statistics({}, time.time() - start_time, success=False)

            # 记录失败历史
            self._record_evolution_history(
                evolution_id=evolution_id,
                base_architecture=base_architecture,
                evolved_architecture={},
                performance_metrics={},
                generation_info={"error": str(e), "timestamp": time.time()},
                success=False,
            )

            return EvolutionResult(
                success=False,
                evolved_architecture={},
                performance_metrics={},
                generation_info={"error": str(e), "timestamp": time.time()},
                error_message=str(e),
            )
        finally:
            self.evolution_active = False

    def get_evolution_status(self) -> Dict[str, Any]:
        """获取演化状态"""
        try:
            engine_status = self.evolution_engine.get_evolution_status()
            fuse_status = self.evolution_engine.get_fuse_status()

            status = {
                "module_status": {
                    "evolution_active": self.evolution_active,
                    "stop_requested": self.stop_requested,
                    "history_size": len(self.evolution_history),
                    "rollback_points": len(self.rollback_points),
                },
                "engine_status": engine_status,
                "fuse_status": fuse_status,
                "statistics": self.statistics,
                "timestamp": time.time(),
            }

            return status

        except Exception as e:
            self.logger.error(f"获取演化状态失败: {str(e)}")
            return {
                "error": str(e),
                "module_status": {
                    "evolution_active": self.evolution_active,
                    "stop_requested": self.stop_requested,
                },
                "timestamp": time.time(),
            }

    def stop_evolution(self) -> bool:
        """停止演化过程"""
        try:
            if not self.evolution_active:
                self.logger.warning("没有正在运行的演化过程")
                return False

            self.stop_requested = True
            self.logger.info("演化停止请求已发送")

            # 在实际系统中，这里应该向演化线程发送停止信号
            # 为简化，我们只是设置标志

            return True

        except Exception as e:
            self.logger.error(f"停止演化失败: {str(e)}")
            return False

    def rollback_evolution(self, generation: int = -1) -> bool:
        """回滚到指定代数的架构"""
        try:
            if generation == -1:
                # 回滚到最新稳定版本
                if not self.rollback_points:
                    self.logger.error("没有可用的回滚点")
                    return False

                # 获取最新的回滚点
                latest_id = max(self.rollback_points.keys())
                rollback_point = self.rollback_points[latest_id]
                self.logger.info(f"回滚到最新回滚点 (ID: {latest_id})")

            else:
                # 回滚到指定代
                if generation not in self.rollback_points:
                    self.logger.error(f"回滚点 {generation} 不存在")
                    return False

                rollback_point = self.rollback_points[generation]
                self.logger.info(f"回滚到回滚点 {generation}")

            # 在实际系统中，这里应该应用回滚点的架构
            # 为简化，我们记录回滚操作

            rollback_info = {
                "rollback_point_id": generation if generation != -1 else latest_id,
                "rollback_point": rollback_point,
                "timestamp": time.time(),
            }

            self.evolution_history.append(
                {"type": "rollback", "info": rollback_info, "timestamp": time.time()}
            )

            # 限制历史记录大小
            if len(self.evolution_history) > self.max_history_size:
                self.evolution_history = self.evolution_history[
                    -self.max_history_size :
                ]

            self.logger.info(f"回滚完成: {rollback_info}")
            return True

        except Exception as e:
            self.logger.error(f"回滚失败: {str(e)}")
            return False

    def get_evolution_statistics(self) -> Dict[str, float]:
        """获取演化统计信息"""
        return self.statistics.copy()

    def _architecture_to_dict(
        self, architecture: NetworkArchitecture
    ) -> Dict[str, Any]:
        """将NetworkArchitecture转换为字典"""
        if not architecture:
            return {}

        try:
            return (
                architecture.to_dict()
                if hasattr(architecture, "to_dict")
                else {
                    "architecture_id": getattr(
                        architecture, "architecture_id", "unknown"
                    ),
                    "layers": getattr(architecture, "layers", []),
                    "attention_mechanisms": getattr(
                        architecture, "attention_mechanisms", []
                    ),
                    "activation_functions": getattr(
                        architecture, "activation_functions", []
                    ),
                    "fusion_strategies": getattr(architecture, "fusion_strategies", []),
                    "connection_patterns": getattr(
                        architecture, "connection_patterns", {}
                    ),
                    "performance_metrics": getattr(
                        architecture, "performance_metrics", {}
                    ),
                    "resource_requirements": getattr(
                        architecture, "resource_requirements", {}
                    ),
                    "fitness_score": getattr(architecture, "fitness_score", 0.0),
                }
            )
        except Exception as e:
            self.logger.error(f"转换架构到字典失败: {str(e)}")
            return {}

    def _create_rollback_point(
        self,
        architecture: Dict[str, Any],
        performance_metrics: Dict[str, float],
        generation_info: Dict[str, Any],
    ) -> int:
        """创建回滚点"""
        rollback_id = self.next_rollback_id
        self.next_rollback_id += 1

        self.rollback_points[rollback_id] = {
            "architecture": architecture,
            "performance_metrics": performance_metrics,
            "generation_info": generation_info,
            "timestamp": time.time(),
        }

        # 限制回滚点数量
        max_rollback_points = self.config.get("max_rollback_points", 10)
        if len(self.rollback_points) > max_rollback_points:
            # 删除最旧的回滚点
            oldest_id = min(self.rollback_points.keys())
            del self.rollback_points[oldest_id]

        return rollback_id

    def _update_statistics(
        self,
        performance_metrics: Dict[str, float],
        evolution_time: float,
        success: bool,
    ):
        """更新统计信息"""
        self.statistics["total_evolutions"] += 1

        if success:
            self.statistics["successful_evolutions"] += 1

            # 更新最佳性能
            if "accuracy" in performance_metrics:
                self.statistics["best_accuracy"] = max(
                    self.statistics["best_accuracy"], performance_metrics["accuracy"]
                )

            if "efficiency" in performance_metrics:
                self.statistics["best_efficiency"] = max(
                    self.statistics["best_efficiency"],
                    performance_metrics["efficiency"],
                )
        else:
            self.statistics["failed_evolutions"] += 1

        # 更新时间统计
        self.statistics["total_evolution_time"] += evolution_time
        if self.statistics["successful_evolutions"] > 0:
            self.statistics["average_evolution_time"] = (
                self.statistics["total_evolution_time"]
                / self.statistics["successful_evolutions"]
            )

    def _record_evolution_history(
        self,
        evolution_id: str,
        base_architecture: Dict[str, Any],
        evolved_architecture: Dict[str, Any],
        performance_metrics: Dict[str, float],
        generation_info: Dict[str, Any],
        success: bool,
    ):
        """记录演化历史"""
        history_entry = {
            "evolution_id": evolution_id,
            "timestamp": time.time(),
            "success": success,
            "base_architecture": base_architecture,
            "evolved_architecture": evolved_architecture,
            "performance_metrics": performance_metrics,
            "generation_info": generation_info,
        }

        self.evolution_history.append(history_entry)

        # 限制历史记录大小
        if len(self.evolution_history) > self.max_history_size:
            self.evolution_history = self.evolution_history[-self.max_history_size :]


# 全局演化模块实例
_global_evolution_module: Optional[IEvolutionModule] = None


def _create_basic_evolution_module(config: Dict[str, Any]) -> IEvolutionModule:
    """创建基础版演化模块实例"""
    # 确保配置包含必要字段
    final_config = config.copy()
    
    # 确保包含search_space（ArchitectureEvolutionEngine必需）
    if "search_space" not in final_config:
        final_config["search_space"] = {
            "layer_types": ["dense", "conv1d", "conv2d", "lstm", "gru", "attention"],
            "layer_sizes": [32, 64, 128, 256, 512],
            "activation_functions": ["relu", "sigmoid", "tanh", "leaky_relu", "gelu"],
            "attention_types": ["scaled_dot_product", "multi_head", "local"],
            "normalization_types": ["batch_norm", "layer_norm", "instance_norm", "group_norm", "none"],
            "fusion_strategies": ["concatenate", "add", "multiply", "average"],
            "max_layers": 10,
            "min_layers": 1,
        }
    
    # 确保包含其他必要字段
    if "mutation_rate" not in final_config:
        final_config["mutation_rate"] = 0.1
    
    if "crossover_rate" not in final_config:
        final_config["crossover_rate"] = 0.8
    
    if "population_size" not in final_config:
        final_config["population_size"] = 100
    
    if "max_generations" not in final_config:
        final_config["max_generations"] = 50
    
    if "elite_size" not in final_config:
        final_config["elite_size"] = 5
    
    return EvolutionModule(final_config)


def _create_enhanced_evolution_module(config: Dict[str, Any]) -> IEvolutionModule:
    """创建增强版演化模块实例"""
    try:
        # 尝试导入增强演化模块
        from core.enhanced_evolution_module import EnhancedEvolutionModule
        
        # 提取增强功能配置
        enhanced_config = config.copy()
        enhanced_features = enhanced_config.pop("enhanced_features", {})
        
        # 合并配置
        final_config = {
            **enhanced_config,
            **enhanced_features
        }
        
        # 创建增强模块实例
        return EnhancedEvolutionModule(final_config)
        
    except ImportError as e:
        raise ImportError("增强演化模块不可用，请确保core.enhanced_evolution_module模块存在") from e
    except Exception as e:
        raise RuntimeError(f"创建增强演化模块失败: {e}") from e


def _create_federated_evolution_module(config: Dict[str, Any]) -> IEvolutionModule:
    """创建联邦版演化模块实例"""
    try:
        # 尝试导入联邦演化模块
        from core.optimization.federated_evolution import create_federated_evolution_module
        
        # 提取联邦功能配置
        federated_config = config.copy()
        federated_features = federated_config.pop("federated_features", {})
        
        # 合并配置
        final_config = {
            **federated_config,
            **federated_features
        }
        
        # 创建联邦模块实例
        return create_federated_evolution_module(final_config)
        
    except ImportError as e:
        raise ImportError("联邦演化模块不可用，请确保core.optimization.federated_evolution模块存在") from e
    except Exception as e:
        raise RuntimeError(f"创建联邦演化模块失败: {e}") from e


def get_evolution_module(config: Optional[Dict[str, Any]] = None) -> IEvolutionModule:
    """
    获取全局演化模块实例（增强版）
    支持基础版、增强版和联邦版演化模块，根据配置自动选择

    Args:
        config: 配置字典，可选，可包含以下配置：
            - evolution_module_type: "basic", "enhanced", "federated", 或 "auto"（默认）
            - enhanced_features: 增强功能配置字典（仅增强版有效）
            - federated_features: 联邦功能配置字典（仅联邦版有效）
            - 其他配置将传递给具体的演化模块

    Returns:
        演化模块实例（基础版、增强版或联邦版）
    """
    global _global_evolution_module

    if _global_evolution_module is None:
        # 解析配置
        config_dict = config or {}
        module_type = config_dict.get("evolution_module_type", "auto").lower()
        
        # 根据模块类型创建实例
        if module_type == "basic":
            # 强制使用基础版
            _global_evolution_module = _create_basic_evolution_module(config_dict)
            logger.info("使用基础版演化模块")
            
        elif module_type == "enhanced":
            # 强制使用增强版
            _global_evolution_module = _create_enhanced_evolution_module(config_dict)
            logger.info("使用增强版演化模块")
            
        elif module_type == "federated":
            # 强制使用联邦版
            _global_evolution_module = _create_federated_evolution_module(config_dict)
            logger.info("使用联邦版演化模块")
            
        else:  # "auto" 或未知类型
            # 自动选择：优先尝试增强版，失败则回退到基础版
            try:
                _global_evolution_module = _create_enhanced_evolution_module(config_dict)
                logger.info("自动选择：使用增强版演化模块")
            except Exception as e:
                logger.warning(f"增强版演化模块创建失败，回退到基础版: {e}")
                _global_evolution_module = _create_basic_evolution_module(config_dict)
                logger.info("自动选择：使用基础版演化模块")
    
    return _global_evolution_module


def reset_evolution_module():
    """重置全局演化模块实例"""
    global _global_evolution_module
    _global_evolution_module = None


# 测试代码
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 80)
    print("演化模块测试")
    print("=" * 80)

    try:
        # 创建演化模块实例
        evolution_module = EvolutionModule()

        print("\n1. 测试演化状态获取:")
        status = evolution_module.get_evolution_status()
        print(
            f"   初始状态: active={status['module_status']['evolution_active']}, "
            f"history_size={status['module_status']['history_size']}"
        )

        print("\n2. 测试架构演化:")
        base_architecture = {
            "type": "generic",
            "layers": [
                {"type": "linear", "size": 128},
                {"type": "relu"},
                {"type": "linear", "size": 64},
                {"type": "relu"},
                {"type": "linear", "size": 32},
            ],
        }

        performance_targets = {"accuracy": 0.85, "efficiency": 0.7, "robustness": 0.8}

        constraints = {
            "resource_constraints": {"memory_mb": 500.0, "compute_gflops": 5.0}
        }

        result = evolution_module.evolve_architecture(
            base_architecture=base_architecture,
            performance_targets=performance_targets,
            constraints=constraints,
        )

        if result.success:
            print(f"   演化成功!")
            print(
                f"   架构ID: {result.evolved_architecture.get('architecture_id', 'unknown')}"
            )
            print(
                f"   性能指标: 准确率={result.performance_metrics.get('accuracy', 0.0):.3f}, "
                f"效率={result.performance_metrics.get('efficiency', 0.0):.3f}"
            )
            print(
                f"   代信息: 代数={result.generation_info.get('generation_number', 0)}, "
                f"最佳适应度={result.generation_info.get('best_fitness', 0.0):.3f}"
            )
        else:
            print(f"   演化失败: {result.error_message}")

        print("\n3. 测试演化状态:")
        status = evolution_module.get_evolution_status()
        print(f"   演化状态: active={status['module_status']['evolution_active']}")
        print(
            f"   统计信息: 总演化次数={status['statistics']['total_evolutions']}, "
            f"成功次数={status['statistics']['successful_evolutions']}"
        )

        print("\n4. 测试演化统计:")
        stats = evolution_module.get_evolution_statistics()
        print(f"   统计: {stats}")

        print("\n✓ 演化模块测试完成")

    except Exception as e:
        print(f"✗ 测试失败: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
