#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因果知识图谱验证脚本

功能:
1. 加载迁移后的因果知识图谱
2. 验证图谱结构和属性完整性
3. 检查因果关系的一致性和合理性
4. 生成验证报告

验证内容:
1. 图谱基本属性（节点数、边数、连通性）
2. 节点属性完整性
3. 边属性完整性（强度、置信度、证据）
4. 因果图性质（无环性、方向一致性）
5. 领域覆盖和分布
6. 性能指标评估

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import os
import json
import logging
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CausalGraphValidator:
    """
    因果知识图谱验证器
    
    验证维度:
    1. 结构验证: 图的基本属性和拓扑结构
    2. 属性验证: 节点和边的属性完整性
    3. 语义验证: 因果关系的逻辑一致性
    4. 领域验证: 领域覆盖和分布合理性
    5. 性能验证: 查询和推理性能基准测试
    """
    
    def __init__(self, graph_data: Optional[Dict[str, Any]] = None, 
                 graph_file_path: Optional[str] = None):
        """
        初始化验证器
        
        Args:
            graph_data: 图谱数据字典（可选）
            graph_file_path: 图谱文件路径（可选）
        """
        self.graph_data = graph_data
        self.graph_file_path = Path(graph_file_path) if graph_file_path else None
        self.graph = None
        self.metadata = None
        self.stats = None
        
        # 验证结果
        self.validation_results = {
            "overall_score": 0.0,
            "structural_validation": {},
            "attribute_validation": {},
            "semantic_validation": {},
            "domain_validation": {},
            "performance_validation": {},
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "passed_tests": 0,
            "total_tests": 0
        }
        
        # 加载图谱数据
        if graph_file_path and not graph_data:
            self.load_graph_from_file()
        elif graph_data:
            self.load_graph_from_data()
    
    def load_graph_from_file(self) -> bool:
        """
        从文件加载图谱数据
        
        Returns:
            是否成功加载
        """
        try:
            if not self.graph_file_path or not self.graph_file_path.exists():
                logger.error(f"图谱文件不存在: {self.graph_file_path}")
                return False
            
            with open(self.graph_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.graph_data = data
            
            # 从node_link_data重建图
            if "graph_data" in data:
                self.graph = nx.node_link_graph(data["graph_data"])
                self.metadata = data.get("metadata", {})
                self.stats = data.get("stats", {})
                logger.info(f"图谱加载成功: {data.get('graph_name', '未命名')}")
                logger.info(f"节点数: {self.graph.number_of_nodes()}, 边数: {self.graph.number_of_edges()}")
                return True
            else:
                logger.error("图谱数据缺少 'graph_data' 字段")
                return False
                
        except Exception as e:
            logger.error(f"加载图谱文件失败: {e}")
            return False
    
    def load_graph_from_data(self) -> bool:
        """
        从数据字典加载图谱
        
        Returns:
            是否成功加载
        """
        try:
            if not self.graph_data or "graph_data" not in self.graph_data:
                logger.error("图谱数据无效")
                return False
            
            self.graph = nx.node_link_graph(self.graph_data["graph_data"])
            self.metadata = self.graph_data.get("metadata", {})
            self.stats = self.graph_data.get("stats", {})
            
            logger.info(f"图谱加载成功: {self.metadata.get('name', '未命名')}")
            logger.info(f"节点数: {self.graph.number_of_nodes()}, 边数: {self.graph.number_of_edges()}")
            return True
            
        except Exception as e:
            logger.error(f"加载图谱数据失败: {e}")
            return False
    
    def run_full_validation(self) -> Dict[str, Any]:
        """
        运行完整的验证流程
        
        Returns:
            验证结果
        """
        logger.info("开始因果知识图谱验证流程")
        
        if self.graph is None:
            logger.error("图谱未加载，无法验证")
            self.validation_results["issues"].append("图谱未加载")
            return self.validation_results
        
        # 1. 结构验证
        self._validate_structure()
        
        # 2. 属性验证
        self._validate_attributes()
        
        # 3. 语义验证
        self._validate_semantics()
        
        # 4. 领域验证
        self._validate_domains()
        
        # 5. 性能验证
        self._validate_performance()
        
        # 计算总体评分
        self._calculate_overall_score()
        
        logger.info(f"验证完成: 通过 {self.validation_results['passed_tests']}/{self.validation_results['total_tests']} 个测试")
        
        return self.validation_results
    
    def _validate_structure(self) -> None:
        """验证图谱结构"""
        logger.info("执行结构验证")
        
        structural_results = {
            "basic_stats": {},
            "connectivity": {},
            "topology": {},
            "issues": [],
            "passed": 0,
            "total": 0
        }
        
        # 基本统计
        structural_results["basic_stats"]["node_count"] = self.graph.number_of_nodes()
        structural_results["basic_stats"]["edge_count"] = self.graph.number_of_edges()
        structural_results["basic_stats"]["density"] = nx.density(self.graph)
        
        structural_results["total"] += 3
        if self.graph.number_of_nodes() > 0:
            structural_results["passed"] += 1
        if self.graph.number_of_edges() > 0:
            structural_results["passed"] += 1
        if structural_results["basic_stats"]["density"] > 0:
            structural_results["passed"] += 1
        
        # 连通性检查
        if nx.is_weakly_connected(self.graph):
            structural_results["connectivity"]["weakly_connected"] = True
            structural_results["passed"] += 1
        else:
            structural_results["connectivity"]["weakly_connected"] = False
            structural_results["connectivity"]["weak_components"] = nx.number_weakly_connected_components(self.graph)
            structural_results["issues"].append(f"图不是弱连通的，有 {structural_results['connectivity']['weak_components']} 个连通分量")
        
        structural_results["total"] += 1
        
        # 检查孤立节点
        isolated_nodes = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
        if isolated_nodes:
            structural_results["connectivity"]["isolated_nodes"] = len(isolated_nodes)
            structural_results["issues"].append(f"发现 {len(isolated_nodes)} 个孤立节点")
        
        # 检查自循环
        self_loops = list(nx.selfloop_edges(self.graph))
        if self_loops:
            structural_results["topology"]["self_loops"] = len(self_loops)
            structural_results["issues"].append(f"发现 {len(self_loops)} 个自循环边")
        
        # 检查环（有向环）
        try:
            is_dag = nx.is_directed_acyclic_graph(self.graph)
            structural_results["topology"]["is_dag"] = is_dag
            if not is_dag:
                structural_results["issues"].append("图中存在有向环，可能违反因果时序性")
        except Exception as e:
            logger.warning(f"有向环检查失败: {e}")
            structural_results["topology"]["is_dag"] = "检查失败"
        
        structural_results["total"] += 2  # 孤立节点和环检查
        
        self.validation_results["structural_validation"] = structural_results
        self.validation_results["total_tests"] += structural_results["total"]
        self.validation_results["passed_tests"] += structural_results["passed"]
        
        # 添加问题和警告
        self.validation_results["issues"].extend(structural_results["issues"])
    
    def _validate_attributes(self) -> None:
        """验证节点和边属性"""
        logger.info("执行属性验证")
        
        attribute_results = {
            "node_attributes": {},
            "edge_attributes": {},
            "issues": [],
            "passed": 0,
            "total": 0
        }
        
        # 节点属性验证
        required_node_attrs = ["id", "type"]
        node_attr_issues = 0
        
        for node, data in self.graph.nodes(data=True):
            for attr in required_node_attrs:
                if attr not in data:
                    node_attr_issues += 1
                    break
        
        attribute_results["node_attributes"]["missing_required"] = node_attr_issues
        attribute_results["node_attributes"]["total_nodes"] = self.graph.number_of_nodes()
        
        if node_attr_issues == 0:
            attribute_results["passed"] += 1
            attribute_results["node_attributes"]["all_have_required"] = True
        else:
            attribute_results["issues"].append(f"{node_attr_issues} 个节点缺少必要属性")
            attribute_results["node_attributes"]["all_have_required"] = False
        
        attribute_results["total"] += 1
        
        # 边属性验证
        required_edge_attrs = ["strength", "confidence"]
        edge_attr_issues = 0
        
        for u, v, data in self.graph.edges(data=True):
            for attr in required_edge_attrs:
                if attr not in data:
                    edge_attr_issues += 1
                    break
        
        attribute_results["edge_attributes"]["missing_required"] = edge_attr_issues
        attribute_results["edge_attributes"]["total_edges"] = self.graph.number_of_edges()
        
        if edge_attr_issues == 0:
            attribute_results["passed"] += 1
            attribute_results["edge_attributes"]["all_have_required"] = True
        else:
            attribute_results["issues"].append(f"{edge_attr_issues} 条边缺少必要属性")
            attribute_results["edge_attributes"]["all_have_required"] = False
        
        attribute_results["total"] += 1
        
        # 检查属性值范围
        edge_strength_issues = 0
        edge_confidence_issues = 0
        
        for u, v, data in self.graph.edges(data=True):
            # 检查强度值
            if "strength" in data:
                strength = data["strength"]
                if isinstance(strength, str):
                    valid_strengths = ["strong", "moderate", "weak", "very_weak", "uncertain"]
                    if strength not in valid_strengths:
                        edge_strength_issues += 1
            
            # 检查置信度范围
            if "confidence" in data:
                confidence = data["confidence"]
                if isinstance(confidence, (int, float)):
                    if confidence < 0.0 or confidence > 1.0:
                        edge_confidence_issues += 1
        
        attribute_results["edge_attributes"]["invalid_strength"] = edge_strength_issues
        attribute_results["edge_attributes"]["invalid_confidence"] = edge_confidence_issues
        
        if edge_strength_issues == 0 and edge_confidence_issues == 0:
            attribute_results["passed"] += 1
        else:
            if edge_strength_issues > 0:
                attribute_results["issues"].append(f"{edge_strength_issues} 条边的强度值无效")
            if edge_confidence_issues > 0:
                attribute_results["issues"].append(f"{edge_confidence_issues} 条边的置信度超出范围[0,1]")
        
        attribute_results["total"] += 1
        
        self.validation_results["attribute_validation"] = attribute_results
        self.validation_results["total_tests"] += attribute_results["total"]
        self.validation_results["passed_tests"] += attribute_results["passed"]
        
        # 添加问题和警告
        self.validation_results["issues"].extend(attribute_results["issues"])
    
    def _validate_semantics(self) -> None:
        """验证语义一致性"""
        logger.info("执行语义验证")
        
        semantic_results = {
            "causal_chains": {},
            "transitivity": {},
            "issues": [],
            "passed": 0,
            "total": 0
        }
        
        # 检查因果链长度
        max_path_length = 0
        long_paths = []
        
        # 抽样检查一些节点对的最长路径
        nodes = list(self.graph.nodes())
        sample_size = min(10, len(nodes))
        
        import random
        if len(nodes) >= 2:
            random.seed(42)  # 固定随机种子以便复现
            samples = random.sample(nodes, min(sample_size * 2, len(nodes)))
            
            for i in range(0, len(samples), 2):
                if i+1 < len(samples):
                    source = samples[i]
                    target = samples[i+1]
                    
                    try:
                        if nx.has_path(self.graph, source, target):
                            path_length = nx.shortest_path_length(self.graph, source, target)
                            max_path_length = max(max_path_length, path_length)
                            
                            if path_length > 10:  # 认为长度超过10的路径可能有问题
                                long_paths.append((source, target, path_length))
                    except nx.NetworkXNoPath:
                        pass
        
        semantic_results["causal_chains"]["max_path_length"] = max_path_length
        
        if max_path_length <= 20:
            semantic_results["passed"] += 1
            semantic_results["causal_chains"]["reasonable_length"] = True
        else:
            semantic_results["issues"].append(f"发现过长的因果链（最大长度: {max_path_length}）")
            semantic_results["causal_chains"]["reasonable_length"] = False
        
        semantic_results["total"] += 1
        
        # 检查基本因果性质
        # 简化：检查是否存在明显的矛盾（如双向强因果关系可能表示混淆）
        strong_bidirectional = 0
        for u, v in self.graph.edges():
            if self.graph.has_edge(v, u):
                # 双向边
                data_uv = self.graph[u][v]
                data_vu = self.graph[v][u]
                
                # 如果双向都是强因果关系，可能有问题
                if (data_uv.get("strength") == "strong" and 
                    data_vu.get("strength") == "strong"):
                    strong_bidirectional += 1
        
        semantic_results["transitivity"]["strong_bidirectional"] = strong_bidirectional
        
        if strong_bidirectional == 0:
            semantic_results["passed"] += 1
            semantic_results["transitivity"]["no_strong_bidirectional"] = True
        else:
            semantic_results["issues"].append(f"发现 {strong_bidirectional} 对双向强因果关系，可能表示混淆")
            semantic_results["transitivity"]["no_strong_bidirectional"] = False
        
        semantic_results["total"] += 1
        
        self.validation_results["semantic_validation"] = semantic_results
        self.validation_results["total_tests"] += semantic_results["total"]
        self.validation_results["passed_tests"] += semantic_results["passed"]
        
        # 添加问题和警告
        self.validation_results["issues"].extend(semantic_results["issues"])
    
    def _validate_domains(self) -> None:
        """验证领域覆盖和分布"""
        logger.info("执行领域验证")
        
        domain_results = {
            "domain_coverage": {},
            "domain_connectivity": {},
            "issues": [],
            "passed": 0,
            "total": 0
        }
        
        # 统计领域分布
        domain_counts = {}
        node_domains = {}
        
        for node, data in self.graph.nodes(data=True):
            domain = "unknown"
            if "metadata" in data:
                domain = data["metadata"].get("domain", "unknown")
            elif "domain" in data:
                domain = data.get("domain", "unknown")
            
            node_domains[node] = domain
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        domain_results["domain_coverage"]["domains"] = domain_counts
        domain_results["domain_coverage"]["unique_domains"] = len(domain_counts)
        
        # 检查领域覆盖
        if len(domain_counts) >= 5:  # 至少覆盖5个领域
            domain_results["passed"] += 1
            domain_results["domain_coverage"]["adequate_coverage"] = True
        else:
            domain_results["issues"].append(f"领域覆盖不足，仅覆盖 {len(domain_counts)} 个领域")
            domain_results["domain_coverage"]["adequate_coverage"] = False
        
        domain_results["total"] += 1
        
        # 检查跨领域连接
        cross_domain_edges = 0
        total_edges = self.graph.number_of_edges()
        
        for u, v in self.graph.edges():
            domain_u = node_domains.get(u, "unknown")
            domain_v = node_domains.get(v, "unknown")
            
            if domain_u != domain_v:
                cross_domain_edges += 1
        
        if total_edges > 0:
            cross_domain_ratio = cross_domain_edges / total_edges
            domain_results["domain_connectivity"]["cross_domain_edges"] = cross_domain_edges
            domain_results["domain_connectivity"]["cross_domain_ratio"] = cross_domain_ratio
            
            if cross_domain_ratio > 0.1:  # 至少10%的跨领域连接
                domain_results["passed"] += 1
                domain_results["domain_connectivity"]["adequate_cross_domain"] = True
            else:
                domain_results["issues"].append(f"跨领域连接不足，仅 {cross_domain_ratio:.2%} 的边连接不同领域")
                domain_results["domain_connectivity"]["adequate_cross_domain"] = False
        else:
            domain_results["domain_connectivity"]["cross_domain_edges"] = 0
            domain_results["domain_connectivity"]["cross_domain_ratio"] = 0.0
            domain_results["issues"].append("图中无边，无法评估跨领域连接")
        
        domain_results["total"] += 1
        
        self.validation_results["domain_validation"] = domain_results
        self.validation_results["total_tests"] += domain_results["total"]
        self.validation_results["passed_tests"] += domain_results["passed"]
        
        # 添加问题和警告
        self.validation_results["issues"].extend(domain_results["issues"])
    
    def _validate_performance(self) -> None:
        """验证性能指标"""
        logger.info("执行性能验证")
        
        performance_results = {
            "query_performance": {},
            "memory_usage": {},
            "issues": [],
            "passed": 0,
            "total": 0
        }
        
        # 简化的性能测试：测量基本图操作的时间
        import time
        
        # 测试1: 邻居查询性能
        start_time = time.time()
        neighbor_queries = 0
        sample_nodes = list(self.graph.nodes())[:10]  # 取前10个节点
        
        for node in sample_nodes:
            try:
                neighbors = list(self.graph.neighbors(node))
                neighbor_queries += 1
            except:
                pass
        
        neighbor_query_time = time.time() - start_time
        if neighbor_queries > 0:
            avg_neighbor_query_time = neighbor_query_time / neighbor_queries
        else:
            avg_neighbor_query_time = 0
        
        performance_results["query_performance"]["avg_neighbor_query_time"] = avg_neighbor_query_time
        
        if avg_neighbor_query_time < 0.001:  # 1毫秒
            performance_results["passed"] += 1
            performance_results["query_performance"]["neighbor_query_fast"] = True
        else:
            performance_results["issues"].append(f"邻居查询较慢，平均 {avg_neighbor_query_time:.6f} 秒")
            performance_results["query_performance"]["neighbor_query_fast"] = False
        
        performance_results["total"] += 1
        
        # 测试2: 路径查询性能
        if len(sample_nodes) >= 2:
            start_time = time.time()
            path_queries = 0
            
            for i in range(len(sample_nodes)-1):
                try:
                    has_path = nx.has_path(self.graph, sample_nodes[i], sample_nodes[i+1])
                    path_queries += 1
                except:
                    pass
            
            path_query_time = time.time() - start_time
            if path_queries > 0:
                avg_path_query_time = path_query_time / path_queries
            else:
                avg_path_query_time = 0
            
            performance_results["query_performance"]["avg_path_query_time"] = avg_path_query_time
            
            if avg_path_query_time < 0.01:  # 10毫秒
                performance_results["passed"] += 1
                performance_results["query_performance"]["path_query_fast"] = True
            else:
                performance_results["issues"].append(f"路径查询较慢，平均 {avg_path_query_time:.6f} 秒")
                performance_results["query_performance"]["path_query_fast"] = False
        else:
            performance_results["query_performance"]["avg_path_query_time"] = 0
            performance_results["issues"].append("节点不足，无法测试路径查询性能")
        
        performance_results["total"] += 1
        
        # 内存使用估算
        import sys
        graph_size = sys.getsizeof(self.graph)
        
        # 粗略估计节点和边数据大小
        node_data_size = 0
        for node, data in self.graph.nodes(data=True):
            node_data_size += sys.getsizeof(data)
        
        edge_data_size = 0
        for u, v, data in self.graph.edges(data=True):
            edge_data_size += sys.getsizeof(data)
        
        total_estimated_size = graph_size + node_data_size + edge_data_size
        performance_results["memory_usage"]["estimated_size_bytes"] = total_estimated_size
        performance_results["memory_usage"]["estimated_size_mb"] = total_estimated_size / (1024 * 1024)
        
        if total_estimated_size < 100 * 1024 * 1024:  # 100MB
            performance_results["passed"] += 1
            performance_results["memory_usage"]["reasonable_size"] = True
        else:
            performance_results["issues"].append(f"图谱内存占用较大，估计 {total_estimated_size / (1024 * 1024):.2f} MB")
            performance_results["memory_usage"]["reasonable_size"] = False
        
        performance_results["total"] += 1
        
        self.validation_results["performance_validation"] = performance_results
        self.validation_results["total_tests"] += performance_results["total"]
        self.validation_results["passed_tests"] += performance_results["passed"]
        
        # 添加问题和警告
        self.validation_results["issues"].extend(performance_results["issues"])
    
    def _calculate_overall_score(self) -> None:
        """计算总体评分"""
        total_tests = self.validation_results["total_tests"]
        passed_tests = self.validation_results["passed_tests"]
        
        if total_tests > 0:
            overall_score = passed_tests / total_tests
        else:
            overall_score = 0.0
        
        self.validation_results["overall_score"] = overall_score
        
        # 根据评分确定验证状态
        if overall_score >= 0.8:
            self.validation_results["verification_status"] = "PASSED"
            self.validation_results["verification_message"] = "因果知识图谱验证通过"
        elif overall_score >= 0.6:
            self.validation_results["verification_status"] = "WARNING"
            self.validation_results["verification_message"] = "因果知识图谱验证警告，存在一些问题"
        else:
            self.validation_results["verification_status"] = "FAILED"
            self.validation_results["verification_message"] = "因果知识图谱验证失败，存在严重问题"
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        生成验证报告
        
        Args:
            output_path: 输出文件路径（可选）
            
        Returns:
            报告内容
        """
        report_lines = []
        
        report_lines.append("=" * 70)
        report_lines.append("因果知识图谱验证报告")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # 总体信息
        report_lines.append(f"📊 总体评分: {self.validation_results['overall_score']:.2%}")
        report_lines.append(f"📈 验证状态: {self.validation_results.get('verification_status', 'UNKNOWN')}")
        report_lines.append(f"📝 验证消息: {self.validation_results.get('verification_message', '')}")
        report_lines.append(f"✅ 通过测试: {self.validation_results['passed_tests']}/{self.validation_results['total_tests']}")
        report_lines.append("")
        
        # 图谱基本信息
        if self.graph:
            report_lines.append("📋 图谱基本信息:")
            report_lines.append(f"   节点数: {self.graph.number_of_nodes()}")
            report_lines.append(f"   边数: {self.graph.number_of_edges()}")
            report_lines.append(f"   密度: {nx.density(self.graph):.6f}")
            report_lines.append("")
        
        # 结构验证结果
        structural = self.validation_results.get("structural_validation", {})
        if structural:
            report_lines.append("🏗️  结构验证结果:")
            report_lines.append(f"   通过测试: {structural.get('passed', 0)}/{structural.get('total', 0)}")
            
            basic_stats = structural.get("basic_stats", {})
            if basic_stats:
                report_lines.append(f"   节点数: {basic_stats.get('node_count', 0)}")
                report_lines.append(f"   边数: {basic_stats.get('edge_count', 0)}")
                report_lines.append(f"   密度: {basic_stats.get('density', 0):.6f}")
            
            connectivity = structural.get("connectivity", {})
            if connectivity:
                if connectivity.get("weakly_connected", False):
                    report_lines.append("   连通性: 弱连通 ✓")
                else:
                    report_lines.append(f"   连通性: 非弱连通 ({connectivity.get('weak_components', 0)} 个分量)")
                
                if "isolated_nodes" in connectivity:
                    report_lines.append(f"   孤立节点: {connectivity['isolated_nodes']}")
            
            issues = structural.get("issues", [])
            if issues:
                report_lines.append("   问题:")
                for issue in issues:
                    report_lines.append(f"     - {issue}")
            
            report_lines.append("")
        
        # 属性验证结果
        attribute = self.validation_results.get("attribute_validation", {})
        if attribute:
            report_lines.append("🏷️  属性验证结果:")
            report_lines.append(f"   通过测试: {attribute.get('passed', 0)}/{attribute.get('total', 0)}")
            
            node_attrs = attribute.get("node_attributes", {})
            if node_attrs:
                if node_attrs.get("all_have_required", False):
                    report_lines.append("   节点属性: 完整性 ✓")
                else:
                    report_lines.append(f"   节点属性: {node_attrs.get('missing_required', 0)} 个节点缺少必要属性")
            
            edge_attrs = attribute.get("edge_attributes", {})
            if edge_attrs:
                if edge_attrs.get("all_have_required", False):
                    report_lines.append("   边属性: 完整性 ✓")
                else:
                    report_lines.append(f"   边属性: {edge_attrs.get('missing_required', 0)} 条边缺少必要属性")
                
                if edge_attrs.get("invalid_strength", 0) > 0:
                    report_lines.append(f"   边属性: {edge_attrs['invalid_strength']} 条边的强度值无效")
                
                if edge_attrs.get("invalid_confidence", 0) > 0:
                    report_lines.append(f"   边属性: {edge_attrs['invalid_confidence']} 条边的置信度超出范围")
            
            issues = attribute.get("issues", [])
            if issues:
                report_lines.append("   问题:")
                for issue in issues:
                    report_lines.append(f"     - {issue}")
            
            report_lines.append("")
        
        # 领域验证结果
        domain = self.validation_results.get("domain_validation", {})
        if domain:
            report_lines.append("🌐 领域验证结果:")
            report_lines.append(f"   通过测试: {domain.get('passed', 0)}/{domain.get('total', 0)}")
            
            domain_coverage = domain.get("domain_coverage", {})
            if domain_coverage:
                report_lines.append(f"   覆盖领域数: {domain_coverage.get('unique_domains', 0)}")
                
                domains = domain_coverage.get("domains", {})
                if domains:
                    report_lines.append("   领域分布:")
                    for domain_name, count in sorted(domains.items(), key=lambda x: x[1], reverse=True)[:10]:  # 前10个
                        report_lines.append(f"     - {domain_name}: {count} 个节点")
            
            domain_connectivity = domain.get("domain_connectivity", {})
            if domain_connectivity:
                cross_domain_ratio = domain_connectivity.get("cross_domain_ratio", 0)
                report_lines.append(f"   跨领域连接比例: {cross_domain_ratio:.2%}")
            
            issues = domain.get("issues", [])
            if issues:
                report_lines.append("   问题:")
                for issue in issues:
                    report_lines.append(f"     - {issue}")
            
            report_lines.append("")
        
        # 性能验证结果
        performance = self.validation_results.get("performance_validation", {})
        if performance:
            report_lines.append("⚡ 性能验证结果:")
            report_lines.append(f"   通过测试: {performance.get('passed', 0)}/{performance.get('total', 0)}")
            
            query_perf = performance.get("query_performance", {})
            if query_perf:
                report_lines.append(f"   平均邻居查询时间: {query_perf.get('avg_neighbor_query_time', 0):.6f} 秒")
                report_lines.append(f"   平均路径查询时间: {query_perf.get('avg_path_query_time', 0):.6f} 秒")
            
            memory_usage = performance.get("memory_usage", {})
            if memory_usage:
                report_lines.append(f"   估计内存占用: {memory_usage.get('estimated_size_mb', 0):.2f} MB")
            
            issues = performance.get("issues", [])
            if issues:
                report_lines.append("   问题:")
                for issue in issues:
                    report_lines.append(f"     - {issue}")
            
            report_lines.append("")
        
        # 总体建议
        if self.validation_results.get("recommendations"):
            report_lines.append("💡 改进建议:")
            for recommendation in self.validation_results["recommendations"]:
                report_lines.append(f"   - {recommendation}")
            report_lines.append("")
        
        report_lines.append("=" * 70)
        report_lines.append("验证报告结束")
        report_lines.append("=" * 70)
        
        report_content = "\n".join(report_lines)
        
        # 保存到文件
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"验证报告已保存到: {output_path}")
            except Exception as e:
                logger.error(f"保存验证报告失败: {e}")
        
        return report_content


def main():
    """主函数"""
    # 配置路径
    graph_file_path = "data/migrated_causal_knowledge_graph.json"
    report_output_path = "data/causal_graph_validation_report.txt"
    
    print(f"加载图谱文件: {graph_file_path}")
    
    # 创建验证器
    validator = CausalGraphValidator(graph_file_path=graph_file_path)
    
    # 运行验证
    print("运行因果知识图谱验证...")
    validation_results = validator.run_full_validation()
    
    # 生成报告
    print("生成验证报告...")
    report = validator.generate_report(report_output_path)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("验证结果摘要")
    print("=" * 60)
    
    print(f"总体评分: {validation_results['overall_score']:.2%}")
    print(f"验证状态: {validation_results.get('verification_status', 'UNKNOWN')}")
    print(f"通过测试: {validation_results['passed_tests']}/{validation_results['total_tests']}")
    
    if validation_results.get("issues"):
        print(f"\n发现 {len(validation_results['issues'])} 个问题:")
        for i, issue in enumerate(validation_results["issues"][:5], 1):  # 显示前5个问题
            print(f"  {i}. {issue}")
        
        if len(validation_results["issues"]) > 5:
            print(f"  ... 和 {len(validation_results['issues']) - 5} 个其他问题")
    
    if validation_results.get("warnings"):
        print(f"\n发现 {len(validation_results['warnings'])} 个警告:")
        for i, warning in enumerate(validation_results["warnings"][:3], 1):
            print(f"  {i}. {warning}")
    
    print(f"\n详细报告已保存到: {report_output_path}")
    print("=" * 60)
    
    return validation_results


if __name__ == "__main__":
    main()