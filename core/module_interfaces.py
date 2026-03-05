#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块接口定义 - Module Interface Definitions

解决用户指出的"核心模块耦合严重"问题：
修改"演化温度参数"需改推理模块代码，无标准化接口

本模块定义核心模块之间的标准化接口，实现模块解耦：
1. IEvolutionModule - 演化模块接口
2. IInferenceModule - 推理模块接口
3. IAdaptationModule - 适配模块接口
4. IParameterManager - 参数管理器接口
5. IMonitoringService - 监控服务接口

通过接口隔离和依赖注入实现模块解耦。
"""

import abc
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EvolutionParameters:
    """演化参数数据类"""
    temperature: float = 0.7
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    population_size: int = 100
    max_generations: int = 50
    selection_pressure: float = 1.5
    elitism_count: int = 2
    diversity_weight: float = 0.3


@dataclass
class InferenceParameters:
    """推理参数数据类"""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    max_length: int = 512
    min_length: int = 1
    num_beams: int = 1
    do_sample: bool = True


@dataclass
class AdaptationParameters:
    """适配参数数据类"""
    hardware_type: str = "auto"  # auto, cpu, cuda, rocm
    memory_limit_mb: int = 4096
    batch_size: int = 1
    precision: str = "mixed"  # full, mixed, half
    optimization_level: int = 1
    fallback_enabled: bool = True


@dataclass
class EvolutionResult:
    """演化结果数据类"""
    success: bool
    evolved_architecture: Dict[str, Any]
    performance_metrics: Dict[str, float]
    generation_info: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class InferenceResult:
    """推理结果数据类"""
    success: bool
    generated_text: Optional[str] = None
    logits: Optional[List[float]] = None
    latency_ms: float = 0.0
    tokens_generated: int = 0
    error_message: Optional[str] = None


@dataclass 
class AdaptationResult:
    """适配结果数据类"""
    success: bool
    hardware_config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    compatibility_status: Dict[str, bool]
    error_message: Optional[str] = None


class IParameterManager(abc.ABC):
    """
    参数管理器接口
    统一管理演化、推理、适配参数
    """
    
    @abc.abstractmethod
    def get_evolution_parameters(self) -> EvolutionParameters:
        """获取演化参数"""
        pass
    
    @abc.abstractmethod
    def get_inference_parameters(self, scene: Optional[str] = None) -> InferenceParameters:
        """获取推理参数
        
        Args:
            scene: 可选场景名称，用于场景自适应参数
            
        Returns:
            推理参数
        """
        pass
    
    @abc.abstractmethod
    def get_adaptation_parameters(self) -> AdaptationParameters:
        """获取适配参数"""
        pass
    
    @abc.abstractmethod
    def update_evolution_parameters(self, params: EvolutionParameters) -> bool:
        """更新演化参数
        
        Args:
            params: 新的演化参数
            
        Returns:
            是否更新成功
        """
        pass
    
    @abc.abstractmethod
    def update_inference_parameters(self, params: InferenceParameters) -> bool:
        """更新推理参数
        
        Args:
            params: 新的推理参数
            
        Returns:
            是否更新成功
        """
        pass
    
    @abc.abstractmethod
    def update_adaptation_parameters(self, params: AdaptationParameters) -> bool:
        """更新适配参数
        
        Args:
            params: 新的适配参数
            
        Returns:
            是否更新成功
        """
        pass
    
    @abc.abstractmethod
    def reset_parameters(self, parameter_type: Optional[str] = None) -> bool:
        """重置参数为默认值
        
        Args:
            parameter_type: 参数类型（evolution/inference/adaptation），如果为None则重置所有参数
            
        Returns:
            是否重置成功
        """
        pass


class IEvolutionModule(abc.ABC):
    """
    演化模块接口
    负责神经网络架构的自主演化
    """
    
    @abc.abstractmethod
    def evolve_architecture(
        self,
        base_architecture: Dict[str, Any],
        performance_targets: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None
    ) -> EvolutionResult:
        """演化神经网络架构
        
        Args:
            base_architecture: 基础架构
            performance_targets: 性能目标
            constraints: 约束条件
            
        Returns:
            演化结果
        """
        pass
    
    @abc.abstractmethod
    def get_evolution_status(self) -> Dict[str, Any]:
        """获取演化状态
        
        Returns:
            演化状态信息
        """
        pass
    
    @abc.abstractmethod
    def stop_evolution(self) -> bool:
        """停止演化过程
        
        Returns:
            是否停止成功
        """
        pass
    
    @abc.abstractmethod
    def rollback_evolution(self, generation: int = -1) -> bool:
        """回滚到指定代数的架构
        
        Args:
            generation: 代数（-1表示回滚到上一稳定版本）
            
        Returns:
            是否回滚成功
        """
        pass
    
    @abc.abstractmethod
    def get_evolution_statistics(self) -> Dict[str, float]:
        """获取演化统计信息
        
        Returns:
            演化统计信息，包括成功率、平均性能等
        """
        pass


class IInferenceModule(abc.ABC):
    """
    推理模块接口
    负责模型推理和文本生成
    """
    
    @abc.abstractmethod
    def generate_text(
        self,
        prompt: str,
        parameters: Optional[InferenceParameters] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> InferenceResult:
        """生成文本
        
        Args:
            prompt: 输入提示
            parameters: 推理参数，如果为None则使用默认参数
            context: 上下文信息
            
        Returns:
            推理结果
        """
        pass
    
    @abc.abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            模型信息
        """
        pass
    
    @abc.abstractmethod
    def load_model(self, model_path: str) -> bool:
        """加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            是否加载成功
        """
        pass
    
    @abc.abstractmethod
    def unload_model(self) -> bool:
        """卸载模型
        
        Returns:
            是否卸载成功
        """
        pass
    
    @abc.abstractmethod
    def get_inference_statistics(self) -> Dict[str, float]:
        """获取推理统计信息
        
        Returns:
            推理统计信息，包括成功率、平均延迟等
        """
        pass


class IAdaptationModule(abc.ABC):
    """
    适配模块接口
    负责硬件适配和资源管理
    """
    
    @abc.abstractmethod
    def adapt_to_hardware(
        self,
        target_hardware: str,
        performance_constraints: Optional[Dict[str, float]] = None
    ) -> AdaptationResult:
        """适配到目标硬件
        
        Args:
            target_hardware: 目标硬件类型
            performance_constraints: 性能约束
            
        Returns:
            适配结果
        """
        pass
    
    @abc.abstractmethod
    def optimize_resource_usage(
        self,
        resource_constraints: Dict[str, float]
    ) -> AdaptationResult:
        """优化资源使用
        
        Args:
            resource_constraints: 资源约束
            
        Returns:
            优化结果
        """
        pass
    
    @abc.abstractmethod
    def get_hardware_info(self) -> Dict[str, Any]:
        """获取硬件信息
        
        Returns:
            硬件信息
        """
        pass
    
    @abc.abstractmethod
    def monitor_resource_usage(self) -> Dict[str, float]:
        """监控资源使用
        
        Returns:
            资源使用情况
        """
        pass
    
    @abc.abstractmethod
    def get_compatibility_status(self) -> Dict[str, bool]:
        """获取兼容性状态
        
        Returns:
            兼容性状态
        """
        pass


class IMonitoringService(abc.ABC):
    """
    监控服务接口
    负责系统监控和指标收集
    """
    
    @abc.abstractmethod
    def collect_metrics(self) -> Dict[str, Any]:
        """收集指标
        
        Returns:
            系统指标
        """
        pass
    
    @abc.abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态
        
        Returns:
            健康状态信息
        """
        pass
    
    @abc.abstractmethod
    def check_thresholds(self) -> List[Dict[str, Any]]:
        """检查阈值
        
        Returns:
            阈值违规列表
        """
        pass
    
    @abc.abstractmethod
    def generate_report(self, report_type: str = "health") -> Dict[str, Any]:
        """生成报告
        
        Args:
            report_type: 报告类型
            
        Returns:
            报告数据
        """
        pass
    
    @abc.abstractmethod
    def register_component(self, component_name: str, component: Any) -> bool:
        """注册组件
        
        Args:
            component_name: 组件名称
            component: 组件实例
            
        Returns:
            是否注册成功
        """
        pass


class ModuleCoordinator:
    """
    模块协调器
    通过接口协调各个模块，实现模块解耦
    """
    
    def __init__(
        self,
        parameter_manager: IParameterManager,
        evolution_module: Optional[IEvolutionModule] = None,
        inference_module: Optional[IInferenceModule] = None,
        adaptation_module: Optional[IAdaptationModule] = None,
        monitoring_service: Optional[IMonitoringService] = None
    ):
        """
        初始化模块协调器
        
        Args:
            parameter_manager: 参数管理器
            evolution_module: 演化模块
            inference_module: 推理模块
            adaptation_module: 适配模块
            monitoring_service: 监控服务
        """
        self.parameter_manager = parameter_manager
        self.evolution_module = evolution_module
        self.inference_module = inference_module
        self.adaptation_module = adaptation_module
        self.monitoring_service = monitoring_service
        
        # 如果提供了监控服务，注册各个模块
        if self.monitoring_service:
            if self.evolution_module:
                self.monitoring_service.register_component("evolution_module", self.evolution_module)
            if self.inference_module:
                self.monitoring_service.register_component("inference_module", self.inference_module)
            if self.adaptation_module:
                self.monitoring_service.register_component("adaptation_module", self.adaptation_module)
    
    def perform_adaptive_inference(
        self,
        prompt: str,
        scene: Optional[str] = None
    ) -> InferenceResult:
        """
        执行自适应推理（使用场景自适应参数）
        
        Args:
            prompt: 输入提示
            scene: 场景名称
            
        Returns:
            推理结果
        """
        if not self.inference_module:
            return InferenceResult(
                success=False,
                error_message="推理模块未初始化"
            )
        
        # 获取场景自适应推理参数
        inference_params = self.parameter_manager.get_inference_parameters(scene)
        
        # 执行推理
        result = self.inference_module.generate_text(
            prompt=prompt,
            parameters=inference_params
        )
        
        # 记录到监控服务（如果可用）
        if self.monitoring_service and result.success:
            self.monitoring_service.collect_metrics()
        
        return result
    
    def coordinate_evolution_and_inference(
        self,
        base_architecture: Dict[str, Any],
        test_prompts: List[str]
    ) -> Dict[str, Any]:
        """
        协调演化和推理（演化后测试新架构）
        
        Args:
            base_architecture: 基础架构
            test_prompts: 测试提示列表
            
        Returns:
            协调结果
        """
        if not self.evolution_module or not self.inference_module:
            return {
                "success": False,
                "error": "演化模块或推理模块未初始化"
            }
        
        results = {
            "evolution_results": [],
            "inference_results": [],
            "overall_success": False
        }
        
        try:
            # 执行演化
            evolution_result = self.evolution_module.evolve_architecture(
                base_architecture=base_architecture,
                performance_targets={"accuracy": 0.8, "latency": 100.0}
            )
            
            results["evolution_results"].append(evolution_result)
            
            if not evolution_result.success:
                results["error"] = f"演化失败: {evolution_result.error_message}"
                return results
            
            # 使用新架构进行推理测试
            for prompt in test_prompts[:3]:  # 限制测试数量
                inference_result = self.perform_adaptive_inference(prompt)
                results["inference_results"].append({
                    "prompt": prompt,
                    "result": inference_result
                })
            
            results["overall_success"] = True
            
        except Exception as e:
            results["error"] = f"协调过程异常: {str(e)}"
        
        return results
    
    def optimize_system_performance(self) -> Dict[str, Any]:
        """
        优化系统性能（协调适配和监控）
        
        Returns:
            优化结果
        """
        if not self.adaptation_module or not self.monitoring_service:
            return {
                "success": False,
                "error": "适配模块或监控服务未初始化"
            }
        
        try:
            # 获取当前资源使用情况
            resource_usage = self.adaptation_module.monitor_resource_usage()
            
            # 设置资源约束
            resource_constraints = {
                "cpu_percent": 80.0,
                "memory_percent": 80.0,
                "gpu_memory_percent": 80.0
            }
            
            # 优化资源使用
            optimization_result = self.adaptation_module.optimize_resource_usage(
                resource_constraints=resource_constraints
            )
            
            # 生成健康报告
            health_report = self.monitoring_service.get_health_status()
            
            return {
                "success": optimization_result.success,
                "optimization_result": optimization_result,
                "health_report": health_report,
                "original_resource_usage": resource_usage
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"性能优化失败: {str(e)}"
            }
    
    def update_parameters_centrally(
        self,
        evolution_params: Optional[EvolutionParameters] = None,
        inference_params: Optional[InferenceParameters] = None,
        adaptation_params: Optional[AdaptationParameters] = None
    ) -> Dict[str, bool]:
        """
        集中更新参数（解决参数耦合问题）
        
        Args:
            evolution_params: 演化参数
            inference_params: 推理参数
            adaptation_params: 适配参数
            
        Returns:
            各参数更新结果
        """
        results = {}
        
        if evolution_params:
            results["evolution"] = self.parameter_manager.update_evolution_parameters(evolution_params)
        
        if inference_params:
            results["inference"] = self.parameter_manager.update_inference_parameters(inference_params)
        
        if adaptation_params:
            results["adaptation"] = self.parameter_manager.update_adaptation_parameters(adaptation_params)
        
        return results


# 适配器类（用于将现有模块适配到接口）
class SceneAdaptiveParameterAdapter(IParameterManager):
    """
    SceneAdaptiveParameters适配器
    将SceneAdaptiveParameters适配到IParameterManager接口
    """
    
    def __init__(self, scene_adaptive_params):
        self.scene_adaptive_params = scene_adaptive_params
        
        # 默认参数
        self.default_evolution_params = EvolutionParameters()
        self.default_adaptation_params = AdaptationParameters()
    
    def get_evolution_parameters(self) -> EvolutionParameters:
        """获取演化参数"""
        # SceneAdaptiveParameters主要处理推理参数
        # 这里返回默认演化参数
        return self.default_evolution_params
    
    def get_inference_parameters(self, scene: Optional[str] = None) -> InferenceParameters:
        """获取推理参数"""
        if scene:
            # 获取指定场景的参数
            scene_params = self.scene_adaptive_params.get_scene_parameters(scene)
            
            return InferenceParameters(
                temperature=scene_params.get("temperature", 0.7),
                repetition_penalty=scene_params.get("repetition_penalty", 1.2),
                top_p=0.95,  # 默认值
                top_k=50,
                max_length=512,
                min_length=1,
                num_beams=1,
                do_sample=True
            )
        else:
            # 获取当前参数
            current_params = self.scene_adaptive_params.get_current_parameters()
            
            return InferenceParameters(
                temperature=current_params.get("temperature", 0.7),
                repetition_penalty=current_params.get("repetition_penalty", 1.2),
                top_p=0.95,
                top_k=50,
                max_length=512,
                min_length=1,
                num_beams=1,
                do_sample=True
            )
    
    def get_adaptation_parameters(self) -> AdaptationParameters:
        """获取适配参数"""
        return self.default_adaptation_params
    
    def update_evolution_parameters(self, params: EvolutionParameters) -> bool:
        """更新演化参数"""
        # SceneAdaptiveParameters不管理演化参数
        # 更新默认值
        self.default_evolution_params = params
        return True
    
    def update_inference_parameters(self, params: InferenceParameters) -> bool:
        """更新推理参数"""
        # SceneAdaptiveParameters目前不支持直接更新
        # 可以扩展SceneAdaptiveParameters以支持此功能
        return False
    
    def update_adaptation_parameters(self, params: AdaptationParameters) -> bool:
        """更新适配参数"""
        self.default_adaptation_params = params
        return True
    
    def reset_parameters(self, parameter_type: Optional[str] = None) -> bool:
        """重置参数"""
        if parameter_type is None or parameter_type == "inference":
            # 重置场景自适应参数
            self.scene_adaptive_params.reset_scene_statistics()
        
        if parameter_type is None or parameter_type == "evolution":
            self.default_evolution_params = EvolutionParameters()
        
        if parameter_type is None or parameter_type == "adaptation":
            self.default_adaptation_params = AdaptationParameters()
        
        return True


# 示例使用
if __name__ == "__main__":
    print("=" * 80)
    print("模块接口定义测试")
    print("=" * 80)
    
    print("\n1. 定义参数数据类:")
    evolution_params = EvolutionParameters(temperature=0.8, population_size=200)
    inference_params = InferenceParameters(temperature=0.7, repetition_penalty=1.3)
    adaptation_params = AdaptationParameters(hardware_type="cuda", memory_limit_mb=8192)
    
    print(f"   演化参数: temperature={evolution_params.temperature}, population_size={evolution_params.population_size}")
    print(f"   推理参数: temperature={inference_params.temperature}, repetition_penalty={inference_params.repetition_penalty}")
    print(f"   适配参数: hardware_type={adaptation_params.hardware_type}, memory_limit_mb={adaptation_params.memory_limit_mb}")
    
    print("\n2. 测试适配器模式:")
    # 创建模拟的SceneAdaptiveParameters
    class MockSceneAdaptiveParams:
        def get_scene_parameters(self, scene):
            return {"temperature": 0.6, "repetition_penalty": 1.1}
        def get_current_parameters(self):
            return {"temperature": 0.7, "repetition_penalty": 1.2}
        def reset_scene_statistics(self):
            pass
    
    mock_params = MockSceneAdaptiveParams()
    adapter = SceneAdaptiveParameterAdapter(mock_params)
    
    inference_params_from_adapter = adapter.get_inference_parameters("industrial_control")
    print(f"   工业控制场景推理参数: temperature={inference_params_from_adapter.temperature}, "
          f"repetition_penalty={inference_params_from_adapter.repetition_penalty}")
    
    print("\n3. 测试模块协调器:")
    coordinator = ModuleCoordinator(parameter_manager=adapter)
    
    # 测试集中更新参数
    update_results = coordinator.update_parameters_centrally(
        evolution_params=EvolutionParameters(temperature=0.9),
        inference_params=InferenceParameters(temperature=0.6)
    )
    
    print(f"   参数更新结果: {update_results}")
    
    print("\n✓ 模块接口定义测试完成")
    print("\n说明:")
    print("  1. 接口定义了演化、推理、适配模块之间的标准契约")
    print("  2. 通过参数管理器统一管理各模块参数，解决参数耦合问题")
    print("  3. 模块协调器通过接口协调各模块，实现模块解耦")
    print("  4. 适配器模式允许现有模块逐步迁移到新接口")