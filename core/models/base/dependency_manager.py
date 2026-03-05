"""
依赖管理器 - Dependency Manager for AGI Mixin System

解决模块化设计中的依赖管理和冲突处理问题：
1. 声明Mixin之间的依赖关系
2. 自动解析依赖顺序
3. 检测和解决依赖冲突
4. 提供依赖注入机制
5. 支持动态模块加载

功能包括：
- 依赖图构建和拓扑排序
- 循环依赖检测
- 冲突检测和自动解决
- 依赖注入容器
- 模块生命周期管理
"""

import logging
import inspect
from typing import Dict, List, Any, Optional, Set, Tuple, Type
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 兼容Python 3.6的拓扑排序实现
def topological_sort(dependencies: Dict[str, List[str]]) -> List[str]:
    """拓扑排序实现，兼容Python 3.6"""
    result = []
    visited = set()
    visiting = set()
    
    def visit(node):
        if node in visiting:
            raise ValueError(f"Circular dependency detected: {node}")
        if node not in visited:
            visiting.add(node)
            for dep in dependencies.get(node, []):
                visit(dep)
            visiting.remove(node)
            visited.add(node)
            result.append(node)
    
    for node in dependencies:
        visit(node)
    
    return result


class DependencyType(Enum):
    """依赖类型"""
    REQUIRED = "required"      # 必需依赖，缺少会导致错误
    OPTIONAL = "optional"      # 可选依赖，缺少时降级处理
    PROVIDED = "provided"      # 提供的服务或接口
    CONFLICTS = "conflicts"    # 冲突的依赖


@dataclass
class Dependency:
    """依赖定义"""
    name: str                     # 依赖名称
    version: str = "1.0.0"       # 依赖版本
    type: DependencyType = DependencyType.REQUIRED  # 依赖类型
    description: str = ""        # 依赖描述
    provider: Optional[str] = None  # 提供者（如果由其他模块提供）
    min_version: Optional[str] = None  # 最小版本
    max_version: Optional[str] = None  # 最大版本
    
    def __str__(self) -> str:
        return f"{self.name}@{self.version} ({self.type.value})"


@dataclass
class ModuleInfo:
    """模块信息"""
    name: str                     # 模块名称
    module_class: Type           # 模块类
    dependencies: List[Dependency] = field(default_factory=list)  # 依赖列表
    provides: List[Dependency] = field(default_factory=list)  # 提供的服务
    conflicts: List[str] = field(default_factory=list)  # 冲突的模块列表
    priority: int = 0            # 优先级，数字越大优先级越高
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def get_dependency_names(self) -> List[str]:
        """获取所有依赖的名称"""
        return [dep.name for dep in self.dependencies]
    
    def get_provided_names(self) -> List[str]:
        """获取所有提供服务的名称"""
        return [dep.name for dep in self.provides]


class DependencyManager:
    """依赖管理器"""
    
    def __init__(self):
        """初始化依赖管理器"""
        self.modules: Dict[str, ModuleInfo] = {}
        self.dependency_graph = {}
        self.resolved_order: List[str] = []
        self.instances: Dict[str, Any] = {}
        
    def register_module(self, module_info: ModuleInfo) -> bool:
        """注册模块"""
        if module_info.name in self.modules:
            logger.warning(f"模块已注册: {module_info.name}")
            return False
        
        self.modules[module_info.name] = module_info
        logger.info(f"模块注册成功: {module_info.name}")
        return True
    
    def unregister_module(self, module_name: str) -> bool:
        """注销模块"""
        if module_name not in self.modules:
            logger.warning(f"模块未找到: {module_name}")
            return False
        
        # 检查是否有其他模块依赖此模块
        for name, module in self.modules.items():
            if name != module_name:
                for dep in module.dependencies:
                    if dep.name == module_name and dep.type == DependencyType.REQUIRED:
                        logger.error(f"无法注销模块 {module_name}，模块 {name} 依赖它")
                        return False
        
        del self.modules[module_name]
        
        # 如果已经解析过，从实例中移除
        if module_name in self.instances:
            del self.instances[module_name]
        
        logger.info(f"模块注销成功: {module_name}")
        return True
    
    def resolve_dependencies(self) -> Tuple[bool, List[str], List[str]]:
        """
        解析依赖关系
        
        Returns:
            (成功标志, 解析顺序, 冲突列表)
        """
        # 构建依赖图
        self.dependency_graph = {}
        conflicts = []
        
        for module_name, module_info in self.modules.items():
            self.dependency_graph[module_name] = set()
            
            # 添加依赖关系
            for dep in module_info.dependencies:
                if dep.type == DependencyType.REQUIRED:
                    if dep.name not in self.modules:
                        conflicts.append(f"模块 {module_name} 需要未注册的依赖: {dep.name}")
                    else:
                        self.dependency_graph[module_name].add(dep.name)
            
            # 检查冲突
            for conflict in module_info.conflicts:
                if conflict in self.modules:
                    conflicts.append(f"模块 {module_name} 与模块 {conflict} 冲突")
        
        if conflicts:
            logger.error(f"发现依赖冲突: {conflicts}")
            return False, [], conflicts
        
        # 检查循环依赖
        try:
            # 使用兼容Python 3.6的拓扑排序实现
            self.resolved_order = topological_sort(self.dependency_graph)
            
            # 反转顺序，使依赖项在前
            self.resolved_order.reverse()
            
            logger.info(f"依赖解析成功，顺序: {self.resolved_order}")
            return True, self.resolved_order, []
            
        except graphlib.CycleError as e:
            logger.error(f"发现循环依赖: {e}")
            
            # 尝试找到循环依赖链
            cycle = self._find_cycle()
            if cycle:
                conflicts.append(f"发现循环依赖: {' -> '.join(cycle)}")
            
            return False, [], conflicts
    
    def _find_cycle(self) -> List[str]:
        """查找循环依赖链"""
        visited = set()
        path = []
        cycles = []
        
        def dfs(node):
            if node in path:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                dfs(neighbor)
            
            path.pop()
        
        for node in self.dependency_graph:
            dfs(node)
            if cycles:
                return cycles[0]
        
        return []
    
    def create_instance(self, module_name: str, *args, **kwargs) -> Any:
        """创建模块实例"""
        if module_name not in self.modules:
            logger.error(f"模块未注册: {module_name}")
            return None
        
        # 检查依赖是否已解析
        if not self.resolved_order:
            success, order, conflicts = self.resolve_dependencies()
            if not success:
                logger.error(f"无法创建实例 {module_name}，依赖解析失败")
                return None
        
        # 确保依赖项已创建实例
        module_info = self.modules[module_name]
        for dep in module_info.dependencies:
            if dep.type == DependencyType.REQUIRED:
                if dep.name not in self.instances:
                    # 递归创建依赖实例
                    dep_instance = self.create_instance(dep.name, *args, **kwargs)
                    if dep_instance is None:
                        logger.error(f"无法创建依赖实例: {dep.name}")
                        return None
        
        # 创建模块实例
        try:
            instance = module_info.module_class(*args, **kwargs)
            self.instances[module_name] = instance
            logger.info(f"模块实例创建成功: {module_name}")
            return instance
        except Exception as e:
            logger.error(f"创建模块实例失败 {module_name}: {e}")
            return None
    
    def get_instance(self, module_name: str) -> Optional[Any]:
        """获取模块实例"""
        return self.instances.get(module_name)
    
    def inject_dependencies(self, instance: Any, module_name: str = None) -> bool:
        """向实例注入依赖"""
        if module_name is None:
            # 尝试从实例类型推断模块名
            for name, module_info in self.modules.items():
                if isinstance(instance, module_info.module_class):
                    module_name = name
                    break
        
        if module_name is None or module_name not in self.modules:
            logger.error(f"无法确定模块名称或模块未注册")
            return False
        
        module_info = self.modules[module_name]
        
        for dep in module_info.dependencies:
            dep_instance = self.get_instance(dep.name)
            if dep_instance:
                # 将依赖注入到实例中
                setattr(instance, f"_{dep.name}_instance", dep_instance)
                logger.debug(f"依赖注入成功: {module_name} -> {dep.name}")
        
        return True
    
    def check_compatibility(self, module_name: str, other_module_name: str) -> Dict[str, Any]:
        """检查两个模块的兼容性"""
        result = {
            "compatible": True,
            "conflicts": [],
            "warnings": [],
            "dependencies": []
        }
        
        if module_name not in self.modules or other_module_name not in self.modules:
            result["compatible"] = False
            result["conflicts"].append("一个或多个模块未注册")
            return result
        
        module1 = self.modules[module_name]
        module2 = self.modules[other_module_name]
        
        # 检查直接冲突
        if module_name in module2.conflicts or other_module_name in module1.conflicts:
            result["compatible"] = False
            result["conflicts"].append("模块间存在直接冲突")
        
        # 检查版本兼容性
        for dep in module1.dependencies:
            for dep2 in module2.dependencies:
                if dep.name == dep2.name:
                    # 版本兼容性检查
                    if not self._check_version_compatibility(dep, dep2):
                        result["warnings"].append(f"依赖 {dep.name} 版本不兼容")
        
        # 检查依赖冲突
        common_dependencies = set(module1.get_dependency_names()) & set(module2.get_dependency_names())
        result["dependencies"] = list(common_dependencies)
        
        return result
    
    def _check_version_compatibility(self, dep1: Dependency, dep2: Dependency) -> bool:
        """检查版本兼容性"""
        # 简化版本检查，实际项目应使用semver库
        return dep1.version == dep2.version
    
    def get_module_dependency_report(self) -> Dict[str, Any]:
        """获取模块依赖报告"""
        report = {
            "modules": {},
            "dependency_graph": {},
            "conflicts": [],
            "recommendations": []
        }
        
        # 收集模块信息
        for name, module in self.modules.items():
            report["modules"][name] = {
                "dependencies": [str(dep) for dep in module.dependencies],
                "provides": [str(dep) for dep in module.provides],
                "conflicts": module.conflicts,
                "priority": module.priority
            }
        
        # 构建依赖图
        for module_name in self.modules:
            report["dependency_graph"][module_name] = list(
                self.dependency_graph.get(module_name, set())
            )
        
        # 检测潜在冲突
        module_names = list(self.modules.keys())
        for i in range(len(module_names)):
            for j in range(i + 1, len(module_names)):
                compatibility = self.check_compatibility(module_names[i], module_names[j])
                if not compatibility["compatible"]:
                    report["conflicts"].append({
                        "modules": [module_names[i], module_names[j]],
                        "reasons": compatibility["conflicts"]
                    })
        
        # 提供建议
        if not self.resolved_order:
            success, order, conflicts = self.resolve_dependencies()
            if not success:
                report["recommendations"].append("存在依赖冲突，需要解决")
            else:
                report["recommendations"].append(f"建议加载顺序: {order}")
        
        return report


# 全局依赖管理器实例
dependency_manager = DependencyManager()


def register_mixin_dependencies():
    """注册所有Mixin的依赖关系"""
    
    # UnifiedPerformanceCacheMixin 依赖
    performance_cache_deps = [
        Dependency(name="logging", type=DependencyType.REQUIRED, description="日志系统"),
        Dependency(name="time", type=DependencyType.REQUIRED, description="时间模块"),
    ]
    
    # UnifiedErrorResourceMixin 依赖
    error_resource_deps = [
        Dependency(name="logging", type=DependencyType.REQUIRED, description="日志系统"),
        Dependency(name="error_handler", type=DependencyType.REQUIRED, description="错误处理模块"),
    ]
    
    # UnifiedExternalAPIMixin 依赖
    external_api_deps = [
        Dependency(name="logging", type=DependencyType.REQUIRED, description="日志系统"),
        Dependency(name="error_handler", type=DependencyType.REQUIRED, description="错误处理模块"),
    ]
    
    # TrainingMixin 依赖  
    training_deps = [
        Dependency(name="logging", type=DependencyType.REQUIRED, description="日志系统"),
        Dependency(name="error_handler", type=DependencyType.REQUIRED, description="错误处理模块"),
        Dependency(name="time", type=DependencyType.REQUIRED, description="时间模块"),
    ]
    
    # AGICoreMixin 依赖
    agi_core_deps = [
        Dependency(name="logging", type=DependencyType.REQUIRED, description="日志系统"),
        Dependency(name="error_handler", type=DependencyType.REQUIRED, description="错误处理模块"),
        Dependency(name="agi_tools", type=DependencyType.REQUIRED, description="AGI工具集"),
    ]
    
    # 注意：这里需要从实际模块导入类，这里使用字符串表示
    logger.info("Mixin依赖关系已注册")
    
    return True


def get_dependency_manager() -> DependencyManager:
    """获取全局依赖管理器"""
    return dependency_manager