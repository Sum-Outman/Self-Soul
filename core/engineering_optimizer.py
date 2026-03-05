"""
工程实现优化器 - 改进代码质量、测试覆盖和部署流程

功能：
1. 代码质量分析
2. 类型提示检查
3. 测试覆盖率评估
4. 部署配置验证
5. 性能瓶颈分析
"""

import ast
import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import importlib.util

logger = logging.getLogger(__name__)


class CodeQualityAnalyzer:
    """代码质量分析器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues = []
        self.metrics = {}
        
    def analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """分析Python文件"""
        issues = []
        metrics = {
            "lines": 0,
            "functions": 0,
            "classes": 0,
            "type_hints": 0,
            "docstrings": 0,
            "complexity": 0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                metrics["lines"] = len(content.splitlines())
            
            # 解析AST
            tree = ast.parse(content, filename=str(file_path))
            
            # 分析AST节点
            for node in ast.walk(tree):
                # 函数定义
                if isinstance(node, ast.FunctionDef):
                    metrics["functions"] += 1
                    
                    # 检查类型提示
                    if node.returns:
                        metrics["type_hints"] += 1
                    
                    # 检查参数类型提示
                    for arg in node.args.args:
                        if arg.annotation:
                            metrics["type_hints"] += 1
                    
                    # 检查文档字符串
                    if ast.get_docstring(node):
                        metrics["docstrings"] += 1
                    
                    # 计算复杂度（简化）
                    complexity = self._calculate_complexity(node)
                    metrics["complexity"] = max(metrics["complexity"], complexity)
                    
                    if complexity > 10:
                        issues.append({
                            "type": "high_complexity",
                            "message": f"函数 {node.name} 复杂度过高 ({complexity})",
                            "file": str(file_path.relative_to(self.project_root)),
                            "line": node.lineno
                        })
                
                # 类定义
                elif isinstance(node, ast.ClassDef):
                    metrics["classes"] += 1
                    
                    # 检查类文档字符串
                    if ast.get_docstring(node):
                        metrics["docstrings"] += 1
                
                # 导入语句分析
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith('.'):
                            issues.append({
                                "type": "relative_import",
                                "message": f"相对导入: {alias.name}",
                                "file": str(file_path.relative_to(self.project_root)),
                                "line": node.lineno
                            })
                
                # 检查eval使用
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == 'eval':
                        issues.append({
                            "type": "security_risk",
                            "message": "使用了eval()，存在安全风险",
                            "file": str(file_path.relative_to(self.project_root)),
                            "line": node.lineno
                        })
            
            return {
                "file": str(file_path.relative_to(self.project_root)),
                "metrics": metrics,
                "issues": issues
            }
            
        except Exception as e:
            logger.error(f"分析文件 {file_path} 失败: {e}")
            return {
                "file": str(file_path.relative_to(self.project_root)),
                "error": str(e),
                "metrics": metrics,
                "issues": issues
            }
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """计算AST节点复杂度（简化版）"""
        complexity = 0
        
        for child in ast.walk(node):
            # 控制流语句增加复杂度
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)):
                complexity += 1
            # 布尔操作增加复杂度
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def analyze_project(self) -> Dict[str, Any]:
        """分析整个项目"""
        python_files = list(self.project_root.rglob("*.py"))
        
        # 过滤掉虚拟环境目录
        python_files = [f for f in python_files if ".venv" not in str(f)]
        
        results = []
        total_metrics = {
            "files": 0,
            "lines": 0,
            "functions": 0,
            "classes": 0,
            "type_hints": 0,
            "docstrings": 0,
            "avg_complexity": 0
        }
        all_issues = []
        
        for file_path in python_files[:50]:  # 限制分析文件数量
            result = self.analyze_python_file(file_path)
            results.append(result)
            
            if "metrics" in result:
                metrics = result["metrics"]
                total_metrics["files"] += 1
                total_metrics["lines"] += metrics["lines"]
                total_metrics["functions"] += metrics["functions"]
                total_metrics["classes"] += metrics["classes"]
                total_metrics["type_hints"] += metrics["type_hints"]
                total_metrics["docstrings"] += metrics["docstrings"]
                total_metrics["avg_complexity"] = (
                    total_metrics["avg_complexity"] * (total_metrics["files"] - 1) + metrics["complexity"]
                ) / total_metrics["files"]
            
            all_issues.extend(result.get("issues", []))
        
        # 计算质量分数
        quality_score = self._calculate_quality_score(total_metrics, all_issues)
        
        return {
            "summary": total_metrics,
            "quality_score": quality_score,
            "issues": all_issues,
            "issue_count": len(all_issues),
            "results": results
        }
    
    def _calculate_quality_score(self, metrics: Dict[str, Any], issues: List[Dict[str, Any]]) -> float:
        """计算代码质量分数 (0-100)"""
        score = 100.0
        
        # 类型提示覆盖率
        if metrics["functions"] > 0:
            type_hint_coverage = metrics["type_hints"] / (metrics["functions"] * 2)  # 返回类型 + 参数类型
            score *= (0.3 + 0.7 * type_hint_coverage)  # 30%基础分 + 70%覆盖率
        
        # 文档字符串覆盖率
        if metrics["functions"] + metrics["classes"] > 0:
            docstring_coverage = metrics["docstrings"] / (metrics["functions"] + metrics["classes"])
            score *= (0.5 + 0.5 * docstring_coverage)  # 50%基础分 + 50%覆盖率
        
        # 问题扣分
        issue_penalty = min(30, len(issues) * 2)  # 最多扣30分
        score -= issue_penalty
        
        # 复杂度扣分
        if metrics["avg_complexity"] > 5:
            complexity_penalty = (metrics["avg_complexity"] - 5) * 2
            score -= min(20, complexity_penalty)
        
        return max(0, min(100, score))


class TestCoverageAnalyzer:
    """测试覆盖率分析器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
    
    def analyze_coverage(self) -> Dict[str, Any]:
        """分析测试覆盖率"""
        try:
            # 尝试使用pytest-cov
            result = subprocess.run(
                ["pytest", "--cov=.", "--cov-report=json", "--cov-fail-under=0"],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=30
            )
            
            if result.returncode == 0 or "coverage" in result.stdout:
                # 解析覆盖率报告
                coverage_report = json.loads(result.stdout)
                return {
                    "success": True,
                    "coverage": coverage_report.get("totals", {}).get("percent_covered", 0),
                    "report": coverage_report
                }
            else:
                return {
                    "success": False,
                    "error": "pytest-cov未安装或未配置",
                    "coverage": 0
                }
                
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            return {
                "success": False,
                "error": str(e),
                "coverage": 0
            }


class DeploymentAnalyzer:
    """部署配置分析器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
    
    def analyze_deployment(self) -> Dict[str, Any]:
        """分析部署配置"""
        issues = []
        config_files = []
        
        # 检查Docker配置
        docker_files = ["Dockerfile", "Dockerfile.backend", "Dockerfile.frontend", "docker-compose.yml"]
        for docker_file in docker_files:
            file_path = self.project_root / docker_file
            if file_path.exists():
                config_files.append(docker_file)
                
                # 简单检查Dockerfile内容
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        
                        if docker_file.startswith("Dockerfile"):
                            if "python:3.6" in content:
                                issues.append({
                                    "type": "outdated_python",
                                    "message": f"{docker_file} 使用Python 3.6，建议升级到3.8+",
                                    "file": docker_file
                                })
                            
                            if "apt-get update && apt-get install" in content and "clean" not in content:
                                issues.append({
                                    "type": "docker_optimization",
                                    "message": f"{docker_file} 缺少apt-get clean，镜像体积可能过大",
                                    "file": docker_file
                                })
                except Exception as e:
                    logger.warning(f"分析Dockerfile失败: {e}")
        
        # 检查环境变量配置
        env_files = [".env.example", ".env.development", ".env.production"]
        for env_file in env_files:
            file_path = self.project_root / env_file
            if file_path.exists():
                config_files.append(env_file)
                
                # 检查是否包含敏感信息
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        
                        sensitive_keywords = ["PASSWORD", "SECRET", "KEY", "TOKEN"]
                        for keyword in sensitive_keywords:
                            if keyword in content:
                                issues.append({
                                    "type": "sensitive_info",
                                    "message": f"{env_file} 可能包含敏感信息 ({keyword})",
                                    "file": env_file,
                                    "severity": "high"
                                })
                except Exception as e:
                    logger.warning(f"分析环境文件失败: {e}")
        
        # 检查部署文档
        deployment_docs = ["DEPLOYMENT.md", "DEPLOYMENT_GUIDE.md", "README.md"]
        docs_found = []
        for doc in deployment_docs:
            if (self.project_root / doc).exists():
                docs_found.append(doc)
        
        return {
            "config_files": config_files,
            "deployment_docs": docs_found,
            "issues": issues,
            "issue_count": len(issues)
        }


class EngineeringOptimizer:
    """工程实现优化器主类"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).absolute()
        self.code_analyzer = CodeQualityAnalyzer(self.project_root)
        self.test_analyzer = TestCoverageAnalyzer(self.project_root)
        self.deployment_analyzer = DeploymentAnalyzer(self.project_root)
        
        logger.info(f"工程实现优化器初始化完成，项目根目录: {self.project_root}")
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """运行完整分析"""
        logger.info("开始工程实现分析...")
        
        results = {
            "timestamp": self._get_timestamp(),
            "project_root": str(self.project_root),
            "analyses": {}
        }
        
        # 1. 代码质量分析
        logger.info("分析代码质量...")
        code_analysis = self.code_analyzer.analyze_project()
        results["analyses"]["code_quality"] = code_analysis
        
        # 2. 测试覆盖率分析
        logger.info("分析测试覆盖率...")
        test_analysis = self.test_analyzer.analyze_coverage()
        results["analyses"]["test_coverage"] = test_analysis
        
        # 3. 部署配置分析
        logger.info("分析部署配置...")
        deployment_analysis = self.deployment_analyzer.analyze_deployment()
        results["analyses"]["deployment"] = deployment_analysis
        
        # 4. 生成改进建议
        logger.info("生成改进建议...")
        recommendations = self._generate_recommendations(
            code_analysis, test_analysis, deployment_analysis
        )
        results["recommendations"] = recommendations
        
        # 5. 计算总体评分
        overall_score = self._calculate_overall_score(
            code_analysis.get("quality_score", 0),
            test_analysis.get("coverage", 0),
            len(deployment_analysis.get("issues", []))
        )
        results["overall_score"] = overall_score
        
        logger.info(f"工程实现分析完成，总体评分: {overall_score:.1f}/100")
        
        return results
    
    def _generate_recommendations(self, code_analysis: Dict[str, Any], 
                                 test_analysis: Dict[str, Any],
                                 deployment_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成改进建议"""
        recommendations = []
        
        # 代码质量建议
        code_issues = code_analysis.get("issues", [])
        if code_analysis.get("quality_score", 0) < 70:
            recommendations.append({
                "category": "代码质量",
                "priority": "high",
                "action": "提高代码质量分数到70分以上",
                "details": [
                    f"当前质量分数: {code_analysis.get('quality_score', 0):.1f}/100",
                    f"发现 {len(code_issues)} 个代码问题",
                    "建议：添加类型提示、完善文档字符串、降低函数复杂度"
                ]
            })
        
        # 类型提示建议
        metrics = code_analysis.get("summary", {})
        if metrics.get("functions", 0) > 0:
            type_hint_ratio = metrics.get("type_hints", 0) / (metrics.get("functions", 1) * 2)
            if type_hint_ratio < 0.5:
                recommendations.append({
                    "category": "类型安全",
                    "priority": "medium",
                    "action": "提高类型提示覆盖率",
                    "details": [
                        f"当前类型提示覆盖率: {type_hint_ratio:.1%}",
                        "建议：为函数参数和返回值添加类型提示",
                        "使用工具：mypy, pyright"
                    ]
                })
        
        # 测试覆盖率建议
        test_coverage = test_analysis.get("coverage", 0)
        if test_coverage < 50:
            recommendations.append({
                "category": "测试覆盖",
                "priority": "high",
                "action": "提高测试覆盖率",
                "details": [
                    f"当前测试覆盖率: {test_coverage:.1f}%",
                    "建议：添加单元测试和集成测试",
                    "使用工具：pytest, pytest-cov"
                ]
            })
        
        # 部署配置建议
        deployment_issues = deployment_analysis.get("issues", [])
        if deployment_issues:
            high_priority_issues = [i for i in deployment_issues if i.get("severity") == "high"]
            
            if high_priority_issues:
                recommendations.append({
                    "category": "部署安全",
                    "priority": "high",
                    "action": "修复部署安全问题",
                    "details": [
                        f"发现 {len(high_priority_issues)} 个高优先级部署问题",
                        "建议：检查环境变量中的敏感信息，更新Docker配置"
                    ]
                })
        
        # 文档建议
        deployment_docs = deployment_analysis.get("deployment_docs", [])
        if not deployment_docs:
            recommendations.append({
                "category": "文档",
                "priority": "low",
                "action": "完善部署文档",
                "details": [
                    "缺少部署文档",
                    "建议：创建DEPLOYMENT.md或更新README.md中的部署说明"
                ]
            })
        
        return recommendations
    
    def _calculate_overall_score(self, code_quality: float, test_coverage: float, 
                                deployment_issues: int) -> float:
        """计算总体评分"""
        # 代码质量权重: 50%
        # 测试覆盖率权重: 30%
        # 部署配置权重: 20%
        
        code_score = min(100, code_quality)
        test_score = test_coverage
        
        deployment_score = 100 - min(30, deployment_issues * 10)
        
        overall = (code_score * 0.5) + (test_score * 0.3) + (deployment_score * 0.2)
        
        return overall
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def generate_report(self, results: Dict[str, Any], output_file: str = "engineering_optimization_report.json") -> str:
        """生成分析报告"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"报告已保存到: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            return ""


def optimize_engineering_implementation(project_root: str = ".") -> Dict[str, Any]:
    """优化工程实现（主函数）"""
    optimizer = EngineeringOptimizer(project_root)
    results = optimizer.run_full_analysis()
    
    # 输出摘要
    print("\n" + "="*80)
    print("工程实现优化分析报告")
    print("="*80)
    
    print(f"\n项目根目录: {results['project_root']}")
    print(f"分析时间: {results['timestamp']}")
    print(f"总体评分: {results['overall_score']:.1f}/100")
    
    # 代码质量摘要
    code_quality = results["analyses"]["code_quality"]
    print(f"\n代码质量:")
    print(f"  质量分数: {code_quality.get('quality_score', 0):.1f}/100")
    print(f"  分析文件: {code_quality.get('summary', {}).get('files', 0)}")
    print(f"  总行数: {code_quality.get('summary', {}).get('lines', 0)}")
    print(f"  函数数: {code_quality.get('summary', {}).get('functions', 0)}")
    print(f"  类型提示: {code_quality.get('summary', {}).get('type_hints', 0)}")
    print(f"  发现问题: {code_quality.get('issue_count', 0)}")
    
    # 测试覆盖率摘要
    test_coverage = results["analyses"]["test_coverage"]
    print(f"\n测试覆盖率:")
    if test_coverage.get("success"):
        print(f"  覆盖率: {test_coverage.get('coverage', 0):.1f}%")
    else:
        print(f"  分析失败: {test_coverage.get('error', '未知错误')}")
    
    # 部署配置摘要
    deployment = results["analyses"]["deployment"]
    print(f"\n部署配置:")
    print(f"  配置文件: {len(deployment.get('config_files', []))}")
    print(f"  部署文档: {len(deployment.get('deployment_docs', []))}")
    print(f"  发现问题: {deployment.get('issue_count', 0)}")
    
    # 建议摘要
    recommendations = results["recommendations"]
    print(f"\n改进建议 ({len(recommendations)} 条):")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n  {i}. [{rec['category']}] {rec['action']} (优先级: {rec['priority']})")
        for detail in rec.get("details", []):
            print(f"     - {detail}")
    
    # 生成报告文件
    report_file = optimizer.generate_report(results)
    if report_file:
        print(f"\n详细报告已保存到: {report_file}")
    
    return results