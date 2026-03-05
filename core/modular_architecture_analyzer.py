"""
模块化架构分析器 - 识别架构问题并设计重构方案

功能：
1. 模块依赖分析
2. 循环依赖检测
3. 接口一致性检查
4. 模块职责分析
5. 重构建议生成

基于评估报告中的架构同质化和模块化不足问题设计
"""

import ast
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, deque
import networkx as nx

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("modular_architecture_analyzer")


class ModuleDependencyAnalyzer:
    """模块依赖分析器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.dependency_graph = nx.DiGraph()
        self.module_files = {}
        self.import_relationships = defaultdict(set)
        self.module_stats = {}  # 存储模块统计信息：代码行数、函数数、类数等
        
    def analyze_project(self) -> Dict[str, Any]:
        """分析整个项目的模块依赖"""
        try:
            # 收集所有Python文件
            python_files = list(self.project_root.rglob("*.py"))
            
            # 过滤掉虚拟环境目录
            python_files = [f for f in python_files if ".venv" not in str(f)]
            
            logger.info(f"开始分析 {len(python_files)} 个Python文件")
            
            # 分析每个文件的导入关系
            for file_path in python_files:
                self._analyze_file_imports(file_path)
            
            logger.info(f"成功分析 {len(self.module_files)} 个模块")
            
            # 构建依赖图
            self._build_dependency_graph()
            logger.info(f"构建依赖图: {len(self.dependency_graph.nodes())} 个节点, {len(self.dependency_graph.edges())} 条边")
            
            # 分析架构问题
            architecture_issues = self._analyze_architecture_issues()
            logger.info(f"发现 {len(architecture_issues)} 个架构问题")
            
            # 生成重构建议
            refactoring_suggestions = self._generate_refactoring_suggestions(architecture_issues)
            logger.info(f"生成 {len(refactoring_suggestions)} 条重构建议")
            
            # 计算模块统计
            module_statistics = self._calculate_module_statistics()
            
            return {
                "project_root": str(self.project_root),
                "total_files": len(python_files),
                "analyzed_files": len(self.module_files),
                "dependency_graph": self._serialize_dependency_graph(),
                "architecture_issues": architecture_issues,
                "refactoring_suggestions": refactoring_suggestions,
                "module_statistics": module_statistics,
                "module_stats": self.module_stats
            }
            
        except Exception as e:
            logger.error(f"分析项目失败: {e}")
            # 返回部分结果或错误信息
            return {
                "project_root": str(self.project_root),
                "total_files": 0,
                "analyzed_files": 0,
                "dependency_graph": {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0},
                "architecture_issues": [],
                "refactoring_suggestions": [],
                "module_statistics": {},
                "module_stats": {},
                "error": str(e)
            }
    
    def _analyze_file_imports(self, file_path: Path):
        """分析文件的导入关系并收集模块统计信息"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析AST
            tree = ast.parse(content, filename=str(file_path))
            
            # 获取模块路径
            module_path = self._get_module_path(file_path)
            self.module_files[module_path] = str(file_path.relative_to(self.project_root))
            
            # 收集模块统计信息
            stats = self._collect_module_statistics(tree, content, file_path)
            self.module_stats[module_path] = stats
            
            # 分析导入语句
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_module = alias.name
                        self._record_import_relationship(module_path, imported_module)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported_module = node.module
                        self._record_import_relationship(module_path, imported_module)
            
        except Exception as e:
            logger.warning(f"分析文件 {file_path} 失败: {e}")
    
    def _collect_module_statistics(self, tree: ast.AST, content: str, file_path: Path) -> Dict[str, Any]:
        """收集模块统计信息"""
        # 计算代码行数
        lines = content.split('\n')
        total_lines = len(lines)
        code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        blank_lines = len([line for line in lines if not line.strip()])
        
        # 统计AST元素
        function_count = 0
        class_count = 0
        import_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_count += 1
            elif isinstance(node, ast.AsyncFunctionDef):
                function_count += 1
            elif isinstance(node, ast.ClassDef):
                class_count += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                import_count += 1
        
        return {
            "file_path": str(file_path.relative_to(self.project_root)),
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "blank_lines": blank_lines,
            "function_count": function_count,
            "class_count": class_count,
            "import_count": import_count,
            "comment_ratio": comment_lines / max(1, code_lines) * 100,
            "file_size_bytes": len(content.encode('utf-8'))
        }
    
    def _get_module_path(self, file_path: Path) -> str:
        """获取模块路径"""
        # 转换为相对于项目根目录的路径
        relative_path = file_path.relative_to(self.project_root)
        
        # 移除.py扩展名，将路径分隔符替换为点
        module_path = str(relative_path).replace('.py', '').replace(os.sep, '.')
        
        # 移除可能的__init__
        if module_path.endswith('.__init__'):
            module_path = module_path[:-9]
        
        return module_path
    
    def _resolve_relative_import(self, source_module: str, relative_import: str) -> str:
        """解析相对导入路径
        
        Args:
            source_module: 源模块路径 (如 'a.b.c')
            relative_import: 相对导入路径 (如 '.d', '..e', '...f')
            
        Returns:
            解析后的绝对模块路径
        """
        if not relative_import.startswith('.'):
            return relative_import
        
        # 计算相对导入的深度
        dot_count = 0
        while dot_count < len(relative_import) and relative_import[dot_count] == '.':
            dot_count += 1
        
        # 获取相对部分
        relative_part = relative_import[dot_count:] or ''
        if relative_part.startswith('.'):
            # 处理多个点的情况
            return relative_import  # 无法解析，返回原样
        
        # 分割源模块路径
        source_parts = source_module.split('.')
        
        # 计算新模块路径
        # 点数量减1表示向上移动的层级
        new_depth = len(source_parts) - (dot_count - 1) if dot_count > 1 else len(source_parts)
        if new_depth < 0:
            # 超出范围，返回原样
            return relative_import
        
        # 构建新的模块路径
        if relative_part:
            new_parts = source_parts[:new_depth] + [relative_part]
        else:
            new_parts = source_parts[:new_depth]
        
        return '.'.join(new_parts) if new_parts else ''
    
    def _record_import_relationship(self, source_module: str, target_module: str):
        """记录导入关系"""
        # 处理相对导入
        if target_module.startswith('.'):
            # 解析相对导入
            resolved_module = self._resolve_relative_import(source_module, target_module)
            if resolved_module and not resolved_module.startswith('.'):
                # 只记录成功解析的相对导入
                self.import_relationships[source_module].add(resolved_module)
            return
        
        # 记录关系
        self.import_relationships[source_module].add(target_module)
    
    def _build_dependency_graph(self):
        """构建依赖图"""
        # 添加节点
        for module in self.module_files.keys():
            self.dependency_graph.add_node(module, type="module")
        
        # 添加边
        for source, targets in self.import_relationships.items():
            for target in targets:
                # 只添加在图中存在的目标节点
                if target in self.dependency_graph:
                    self.dependency_graph.add_edge(source, target, type="import")
    
    def _serialize_dependency_graph(self) -> Dict[str, Any]:
        """序列化依赖图"""
        nodes = []
        edges = []
        
        for node in self.dependency_graph.nodes():
            nodes.append({
                "id": node,
                "type": self.dependency_graph.nodes[node].get("type", "unknown"),
                "file": self.module_files.get(node, "")
            })
        
        for source, target, data in self.dependency_graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "type": data.get("type", "unknown")
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges)
        }
    
    def _analyze_architecture_issues(self) -> List[Dict[str, Any]]:
        """分析架构问题"""
        issues = []
        
        # 1. 检测循环依赖
        cycles = list(nx.simple_cycles(self.dependency_graph))
        if cycles:
            for i, cycle in enumerate(cycles[:5]):  # 最多显示5个循环
                issues.append({
                    "type": "circular_dependency",
                    "severity": "high",
                    "description": f"循环依赖 #{i+1}: {' -> '.join(cycle)} -> {cycle[0]}",
                    "modules": list(cycle),
                    "cycle_length": len(cycle)
                })
        
        # 2. 检测高度耦合模块
        degree_centrality = nx.degree_centrality(self.dependency_graph)
        highly_coupled = [(node, centrality) for node, centrality in degree_centrality.items() 
                         if centrality > 0.3]  # 30%以上的中心度
        
        for node, centrality in sorted(highly_coupled, key=lambda x: x[1], reverse=True)[:10]:
            issues.append({
                "type": "high_coupling",
                "severity": "medium",
                "description": f"模块 {node} 耦合度过高 (中心度: {centrality:.3f})",
                "module": node,
                "centrality": centrality,
                "dependencies": list(self.dependency_graph.successors(node)),
                "dependents": list(self.dependency_graph.predecessors(node))
            })
        
        # 3. 检测大模块（过多的导入）
        for node in self.dependency_graph.nodes():
            out_degree = self.dependency_graph.out_degree(node)
            in_degree = self.dependency_graph.in_degree(node)
            
            if out_degree > 10:  # 导入超过10个其他模块
                issues.append({
                    "type": "large_module",
                    "severity": "medium",
                    "description": f"模块 {node} 导入过多 ({out_degree} 个导入)",
                    "module": node,
                    "out_degree": out_degree,
                    "in_degree": in_degree
                })
            
            if in_degree > 15:  # 被超过15个模块导入
                issues.append({
                    "type": "central_module",
                    "severity": "low",
                    "description": f"模块 {node} 被过多模块依赖 ({in_degree} 个依赖)",
                    "module": node,
                    "out_degree": out_degree,
                    "in_degree": in_degree
                })
        
        # 4. 检测孤立模块
        isolated_nodes = [node for node in self.dependency_graph.nodes() 
                         if self.dependency_graph.degree(node) == 0]
        
        if isolated_nodes:
            issues.append({
                "type": "isolated_modules",
                "severity": "low",
                "description": f"发现 {len(isolated_nodes)} 个孤立模块",
                "modules": isolated_nodes,
                "count": len(isolated_nodes)
            })
        
        return issues
    
    def _generate_refactoring_suggestions(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成重构建议"""
        suggestions = []
        
        # 按问题类型分组
        issue_types = defaultdict(list)
        for issue in issues:
            issue_types[issue["type"]].append(issue)
        
        # 循环依赖重构建议
        if "circular_dependency" in issue_types:
            cycles = issue_types["circular_dependency"]
            
            for cycle_issue in cycles:
                modules = cycle_issue["modules"]
                cycle_length = cycle_issue["cycle_length"]
                
                suggestions.append({
                    "type": "break_circular_dependency",
                    "priority": "high",
                    "description": f"打破循环依赖: {cycle_issue['description']}",
                    "action": "引入接口抽象或依赖反转",
                    "details": [
                        f"涉及模块: {', '.join(modules)}",
                        f"循环长度: {cycle_length}",
                        "建议: 提取公共接口，使用依赖注入，或引入中间层"
                    ],
                    "affected_modules": modules
                })
        
        # 高耦合重构建议
        if "high_coupling" in issue_types:
            high_coupling_issues = issue_types["high_coupling"]
            
            for coupling_issue in high_coupling_issues[:3]:  # 只处理前3个
                module = coupling_issue["module"]
                centrality = coupling_issue["centrality"]
                
                suggestions.append({
                    "type": "reduce_coupling",
                    "priority": "medium",
                    "description": f"降低模块 {module} 的耦合度",
                    "action": "提取子模块或引入接口",
                    "details": [
                        f"当前中心度: {centrality:.3f}",
                        f"依赖模块数: {len(coupling_issue['dependencies'])}",
                        f"被依赖模块数: {len(coupling_issue['dependents'])}",
                        "建议: 提取独立的功能到子模块，使用接口隔离实现"
                    ],
                    "affected_modules": [module]
                })
        
        # 大模块重构建议
        if "large_module" in issue_types:
            large_module_issues = issue_types["large_module"]
            
            for module_issue in large_module_issues[:5]:  # 只处理前5个
                module = module_issue["module"]
                out_degree = module_issue["out_degree"]
                
                suggestions.append({
                    "type": "split_large_module",
                    "priority": "medium",
                    "description": f"拆分大模块 {module}",
                    "action": "按功能职责拆分为多个小模块",
                    "details": [
                        f"当前导入数: {out_degree}",
                        "建议: 根据单一职责原则拆分模块，每个模块专注于一个功能领域"
                    ],
                    "affected_modules": [module]
                })
        
        # 中心模块重构建议
        if "central_module" in issue_types:
            central_module_issues = issue_types["central_module"]
            
            for module_issue in central_module_issues[:3]:  # 只处理前3个
                module = module_issue["module"]
                in_degree = module_issue["in_degree"]
                
                suggestions.append({
                    "type": "abstract_central_module",
                    "priority": "low",
                    "description": f"为中心模块 {module} 创建抽象接口",
                    "action": "定义稳定接口，分离实现",
                    "details": [
                        f"被依赖数: {in_degree}",
                        "建议: 定义清晰的接口，将实现细节隐藏在接口后面"
                    ],
                    "affected_modules": [module]
                })
        
        # 通用架构建议
        suggestions.extend([
            {
                "type": "define_module_boundaries",
                "priority": "high",
                "description": "定义清晰的模块边界",
                "action": "制定模块化设计规范",
                "details": [
                    "定义模块的职责范围",
                    "制定模块间通信协议",
                    "建立模块依赖管理规则"
                ],
                "affected_modules": ["all"]
            },
            {
                "type": "implement_dependency_injection",
                "priority": "medium",
                "description": "实现依赖注入容器",
                "action": "引入依赖注入框架",
                "details": [
                    "减少模块间的硬编码依赖",
                    "提高代码的可测试性",
                    "支持动态配置和替换"
                ],
                "affected_modules": ["all"]
            },
            {
                "type": "create_module_registry",
                "priority": "medium",
                "description": "创建模块注册表",
                "action": "实现服务发现机制",
                "details": [
                    "支持动态模块加载",
                    "提供模块发现功能",
                    "支持插件化架构"
                ],
                "affected_modules": ["all"]
            }
        ])
        
        return suggestions
    
    def _calculate_module_statistics(self) -> Dict[str, Any]:
        """计算模块统计信息"""
        if not self.dependency_graph.nodes():
            return {}
        
        # 计算图指标
        try:
            density = nx.density(self.dependency_graph)
        except:
            density = 0.0
        
        try:
            avg_clustering = nx.average_clustering(self.dependency_graph.to_undirected())
        except:
            avg_clustering = 0.0
        
        # 计算模块大小分布
        out_degrees = [self.dependency_graph.out_degree(node) for node in self.dependency_graph.nodes()]
        in_degrees = [self.dependency_graph.in_degree(node) for node in self.dependency_graph.nodes()]
        
        return {
            "total_modules": len(self.dependency_graph.nodes()),
            "total_dependencies": len(self.dependency_graph.edges()),
            "graph_density": density,
            "average_clustering": avg_clustering,
            "avg_out_degree": sum(out_degrees) / max(1, len(out_degrees)),
            "avg_in_degree": sum(in_degrees) / max(1, len(in_degrees)),
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "max_in_degree": max(in_degrees) if in_degrees else 0
        }


class ModularArchitectureDesigner:
    """模块化架构设计师"""
    
    def __init__(self, analysis_results: Dict[str, Any]):
        self.analysis_results = analysis_results
        self.design_proposals = []
        
    def generate_design_proposals(self) -> List[Dict[str, Any]]:
        """生成架构设计提案"""
        proposals = []
        
        # 1. 基于分析结果的针对性设计
        issues = self.analysis_results.get("architecture_issues", [])
        suggestions = self.analysis_results.get("refactoring_suggestions", [])
        
        # 高优先级问题解决方案
        high_priority_issues = [issue for issue in issues if issue.get("severity") == "high"]
        if high_priority_issues:
            proposals.append(self._create_circular_dependency_solution(high_priority_issues))
        
        # 2. 模块化分层架构
        proposals.append(self._create_layered_architecture_design())
        
        # 3. 插件化架构设计
        proposals.append(self._create_plugin_architecture_design())
        
        # 4. 微服务化演进路径
        proposals.append(self._create_microservices_evolution_path())
        
        # 5. 接口标准化方案
        proposals.append(self._create_interface_standardization())
        
        self.design_proposals = proposals
        return proposals
    
    def _create_circular_dependency_solution(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建循环依赖解决方案"""
        circular_issues = [issue for issue in issues if issue["type"] == "circular_dependency"]
        
        return {
            "design_name": "循环依赖解决方案",
            "priority": "high",
            "description": "解决模块间的循环依赖问题",
            "approach": "依赖反转和接口隔离",
            "implementation_steps": [
                "1. 识别循环依赖链中的核心抽象",
                "2. 提取公共接口到独立模块",
                "3. 使用依赖注入替换直接导入",
                "4. 引入中间层或消息总线",
                "5. 重构测试以验证解耦效果"
            ],
            "benefits": [
                "消除循环依赖，提高代码稳定性",
                "提高模块独立性和可测试性",
                "支持更灵活的模块组合"
            ],
            "challenges": [
                "需要修改现有模块接口",
                "可能增加初始架构复杂度",
                "需要更新相关测试"
            ],
            "estimated_effort": "medium",  # low, medium, high
            "affected_modules": list(set([module for issue in circular_issues for module in issue.get("modules", [])]))
        }
    
    def _create_layered_architecture_design(self) -> Dict[str, Any]:
        """创建分层架构设计"""
        return {
            "design_name": "分层模块化架构",
            "priority": "high",
            "description": "将系统划分为清晰的分层架构",
            "approach": "经典分层架构（表示层、业务层、数据层）",
            "layers": [
                {
                    "name": "接口层",
                    "responsibilities": ["API接口", "协议适配", "输入验证"],
                    "allowed_dependencies": ["业务层"],
                    "example_modules": ["api_handlers", "controllers", "presenters"]
                },
                {
                    "name": "业务层",
                    "responsibilities": ["业务逻辑", "领域模型", "工作流"],
                    "allowed_dependencies": ["数据层", "工具层"],
                    "example_modules": ["services", "managers", "domain_models"]
                },
                {
                    "name": "数据层",
                    "responsibilities": ["数据访问", "持久化", "缓存"],
                    "allowed_dependencies": ["工具层"],
                    "example_modules": ["repositories", "daos", "data_models"]
                },
                {
                    "name": "工具层",
                    "responsibilities": ["通用工具", "基础设施", "第三方集成"],
                    "allowed_dependencies": [],  # 基础层，不依赖其他层
                    "example_modules": ["utils", "helpers", "external_apis"]
                }
            ],
            "implementation_steps": [
                "1. 定义各层职责和依赖规则",
                "2. 将现有模块归类到相应层次",
                "3. 重构违反依赖规则的模块",
                "4. 建立层间通信机制",
                "5. 添加架构验证工具"
            ],
            "benefits": [
                "清晰的关注点分离",
                "可维护性和可测试性提高",
                "团队分工更明确"
            ],
            "challenges": [
                "需要大规模重构",
                "可能引入性能开销",
                "需要团队架构共识"
            ],
            "estimated_effort": "high",
            "affected_modules": ["all"]
        }
    
    def _create_plugin_architecture_design(self) -> Dict[str, Any]:
        """创建插件化架构设计"""
        return {
            "design_name": "插件化架构",
            "priority": "medium",
            "description": "支持动态加载和卸载的功能模块",
            "approach": "基于接口的插件系统",
            "core_components": [
                {
                    "name": "插件管理器",
                    "responsibilities": ["插件加载", "生命周期管理", "依赖解析"]
                },
                {
                    "name": "插件接口",
                    "responsibilities": ["定义插件契约", "提供扩展点", "版本兼容"]
                },
                {
                    "name": "插件注册表",
                    "responsibilities": ["插件发现", "元数据管理", "服务注册"]
                },
                {
                    "name": "插件上下文",
                    "responsibilities": ["运行时环境", "资源管理", "事件总线"]
                }
            ],
            "implementation_steps": [
                "1. 定义插件接口和扩展点",
                "2. 实现插件管理器和注册表",
                "3. 将核心功能重构为插件",
                "4. 实现插件热加载机制",
                "5. 添加插件配置和验证"
            ],
            "benefits": [
                "系统可扩展性极大提高",
                "支持动态功能增减",
                "便于第三方扩展开发"
            ],
            "challenges": [
                "插件接口设计需要前瞻性",
                "插件兼容性管理复杂",
                "安全性和稳定性要求高"
            ],
            "estimated_effort": "high",
            "affected_modules": ["core", "models", "services"]
        }
    
    def _create_microservices_evolution_path(self) -> Dict[str, Any]:
        """创建微服务化演进路径"""
        return {
            "design_name": "微服务化演进路径",
            "priority": "low",
            "description": "从单体到微服务的渐进式演进方案",
            "approach": "绞杀者模式 + 领域驱动设计",
            "evolution_stages": [
                {
                    "stage": 1,
                    "name": "模块化单体",
                    "description": "在单体内部实现清晰模块边界",
                    "duration": "1-2个月",
                    "deliverables": ["模块化架构", "接口标准化", "内部API"]
                },
                {
                    "stage": 2,
                    "name": "独立部署单元",
                    "description": "将模块打包为独立部署单元",
                    "duration": "2-3个月",
                    "deliverables": ["独立服务", "服务发现", "基础监控"]
                },
                {
                    "stage": 3,
                    "name": "微服务架构",
                    "description": "完全解耦的微服务系统",
                    "duration": "3-6个月",
                    "deliverables": ["完整微服务", "分布式追踪", "自动化运维"]
                }
            ],
            "implementation_steps": [
                "1. 识别领域边界（领域驱动设计）",
                "2. 定义服务契约和API",
                "3. 提取第一个微服务（绞杀者模式）",
                "4. 建立服务通信和治理",
                "5. 逐步迁移其他模块"
            ],
            "benefits": [
                "渐进式演进，风险可控",
                "团队独立开发和部署",
                "技术栈灵活性提高"
            ],
            "challenges": [
                "分布式系统复杂性",
                "数据一致性问题",
                "运维和监控复杂度增加"
            ],
            "estimated_effort": "very_high",
            "affected_modules": ["all"]
        }
    
    def _create_interface_standardization(self) -> Dict[str, Any]:
        """创建接口标准化方案"""
        return {
            "design_name": "接口标准化方案",
            "priority": "medium",
            "description": "统一模块间通信接口标准",
            "approach": "基于协议和契约的接口设计",
            "standards": [
                {
                    "type": "RESTful API",
                    "scope": "外部接口",
                    "specification": "OpenAPI 3.0",
                    "tools": ["FastAPI", "Swagger"]
                },
                {
                    "type": "内部服务调用",
                    "scope": "模块间通信",
                    "specification": "gRPC/protobuf",
                    "tools": ["grpc", "protobuf"]
                },
                {
                    "type": "事件通信",
                    "scope": "异步通知",
                    "specification": "CloudEvents",
                    "tools": ["Kafka", "RabbitMQ"]
                },
                {
                    "type": "数据交换",
                    "scope": "数据传递",
                    "specification": "JSON Schema",
                    "tools": ["pydantic", "marshmallow"]
                }
            ],
            "implementation_steps": [
                "1. 制定接口标准和规范",
                "2. 创建接口代码生成工具",
                "3. 重构现有接口符合标准",
                "4. 添加接口验证和测试",
                "5. 建立接口文档自动生成"
            ],
            "benefits": [
                "接口一致性和可预测性",
                "自动化的接口验证",
                "简化的集成测试"
            ],
            "challenges": [
                "现有接口迁移工作量大",
                "标准制定需要跨团队共识",
                "工具链建设和维护"
            ],
            "estimated_effort": "medium",
            "affected_modules": ["api", "services", "models"]
        }
    
    def generate_implementation_roadmap(self) -> Dict[str, Any]:
        """生成实施路线图"""
        if not self.design_proposals:
            self.generate_design_proposals()
        
        # 按优先级排序
        priority_order = {"high": 1, "medium": 2, "low": 3}
        sorted_proposals = sorted(self.design_proposals, 
                                 key=lambda x: priority_order.get(x.get("priority", "low"), 3))
        
        roadmap = {
            "phases": [],
            "total_estimated_duration": "6-12个月",
            "key_milestones": []
        }
        
        # 定义实施阶段
        phases = [
            {
                "phase": 1,
                "name": "架构分析和规划",
                "duration": "1个月",
                "objectives": ["理解现状", "制定架构原则", "优先解决高风险问题"],
                "designs": [p for p in sorted_proposals if p["priority"] == "high"][:2]
            },
            {
                "phase": 2,
                "name": "核心架构重构",
                "duration": "2-3个月",
                "objectives": ["解决循环依赖", "建立分层架构", "接口标准化"],
                "designs": [p for p in sorted_proposals if p["priority"] in ["high", "medium"]][:3]
            },
            {
                "phase": 3,
                "name": "架构优化和扩展",
                "duration": "2-3个月",
                "objectives": ["插件化支持", "性能优化", "监控和治理"],
                "designs": [p for p in sorted_proposals if p["priority"] in ["medium", "low"]]
            },
            {
                "phase": 4,
                "name": "持续改进和演进",
                "duration": "持续",
                "objectives": ["架构演进", "技术债务管理", "团队能力建设"],
                "designs": []
            }
        ]
        
        roadmap["phases"] = phases
        
        # 关键里程碑
        roadmap["key_milestones"] = [
            {
                "milestone": "M1",
                "name": "架构分析完成",
                "phase": 1,
                "success_criteria": ["架构问题报告完成", "重构方案获得批准", "团队共识建立"]
            },
            {
                "milestone": "M2",
                "name": "核心依赖问题解决",
                "phase": 2,
                "success_criteria": ["循环依赖消除", "模块边界清晰", "构建时间改善20%"]
            },
            {
                "milestone": "M3",
                "name": "分层架构实施完成",
                "phase": 2,
                "success_criteria": ["分层架构就绪", "接口标准化完成", "代码可测试性提高"]
            },
            {
                "milestone": "M4",
                "name": "插件化架构就绪",
                "phase": 3,
                "success_criteria": ["插件系统运行", "动态加载支持", "扩展开发指南完成"]
            }
        ]
        
        return roadmap


def analyze_modular_architecture(project_root: str = ".") -> Dict[str, Any]:
    """分析模块化架构（主函数）"""
    logger.info("开始模块化架构分析...")
    
    analyzer = ModuleDependencyAnalyzer(project_root)
    analysis_results = analyzer.analyze_project()
    
    logger.info(f"分析完成: {analysis_results['total_files']} 个文件, {analysis_results['analyzed_files']} 个模块")
    logger.info(f"发现 {len(analysis_results['architecture_issues'])} 个架构问题")
    
    return analysis_results


def design_modular_refactoring(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """设计模块化重构方案"""
    logger.info("开始设计模块化重构方案...")
    
    designer = ModularArchitectureDesigner(analysis_results)
    design_proposals = designer.generate_design_proposals()
    implementation_roadmap = designer.generate_implementation_roadmap()
    
    logger.info(f"生成了 {len(design_proposals)} 个架构设计提案")
    logger.info(f"制定了 {len(implementation_roadmap['phases'])} 阶段实施路线图")
    
    return {
        "design_proposals": design_proposals,
        "implementation_roadmap": implementation_roadmap,
        "analysis_summary": {
            "total_issues": len(analysis_results.get("architecture_issues", [])),
            "high_priority_issues": len([i for i in analysis_results.get("architecture_issues", []) 
                                       if i.get("severity") == "high"]),
            "suggestions": len(analysis_results.get("refactoring_suggestions", []))
        }
    }