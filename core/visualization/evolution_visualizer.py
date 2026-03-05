#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演化可视化分析工具 - Evolution Visualization and Analysis Tool

提供演化历史、性能指标和架构变化的可视化分析功能。
支持多种输出格式：文本报告、ASCII图表、HTML报告和JSON数据导出。

主要功能：
1. 演化历史可视化：展示演化过程的性能趋势、架构变化和决策点
2. 性能指标分析：多维度性能指标的可视化对比
3. 架构差异比较：可视化展示不同架构版本之间的差异
4. 演化决策解释：提供演化决策的可解释性分析
5. 实时监控仪表盘：提供实时演化状态的文本仪表盘

设计原则：
- 轻量级：不依赖复杂的外部库，核心功能使用纯Python实现
- 可扩展：支持多种输出格式和后端渲染引擎
- 交互式：支持命令行交互和参数配置
- 可嵌入：可以轻松集成到现有监控和日志系统中
"""

import json
import time
import datetime
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvolutionDataPoint:
    """演化数据点"""
    timestamp: float
    generation: int
    performance_metrics: Dict[str, float]
    architecture_id: str
    architecture_summary: Dict[str, Any]
    decision_info: Optional[Dict[str, Any]] = None


@dataclass
class ArchitectureComparison:
    """架构比较结果"""
    version_a: str
    version_b: str
    similarity_score: float
    added_components: List[str]
    removed_components: List[str]
    modified_components: List[Tuple[str, str, str]]  # (组件名, 旧值, 新值)
    performance_delta: Dict[str, float]


@dataclass
class VisualizationConfig:
    """可视化配置"""
    output_format: str = "text"  # text, html, json
    max_data_points: int = 100
    show_details: bool = True
    show_ascii_charts: bool = True
    chart_width: int = 60
    chart_height: int = 20
    color_enabled: bool = True


class ASCIIChartRenderer:
    """ASCII图表渲染器"""
    
    @staticmethod
    def line_chart(values: List[float], width: int = 60, height: int = 20) -> str:
        """生成线状ASCII图表
        
        Args:
            values: 数值列表
            width: 图表宽度（字符）
            height: 图表高度（行数）
            
        Returns:
            ASCII图表字符串
        """
        if not values:
            return "无数据"
        
        # 归一化数值
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            # 所有值相同
            normalized = [height // 2] * len(values)
        else:
            normalized = [
                int((val - min_val) / (max_val - min_val) * (height - 1))
                for val in values
            ]
        
        # 创建画布
        canvas = [[' ' for _ in range(width)] for _ in range(height)]
        
        # 绘制Y轴
        for y in range(height):
            canvas[y][0] = '│'
        
        # 绘制X轴
        for x in range(width):
            canvas[height-1][x] = '─'
        
        # 角标
        canvas[height-1][0] = '└'
        
        # 绘制数据点
        x_step = max(1, len(values) / width)
        
        for i in range(width):
            idx = min(int(i * x_step), len(values) - 1)
            y = height - 1 - normalized[idx]
            
            if 0 <= y < height and 0 <= i < width:
                canvas[y][i] = '●'
        
        # 连接数据点（简单版本）
        for i in range(width - 1):
            idx1 = min(int(i * x_step), len(values) - 1)
            idx2 = min(int((i + 1) * x_step), len(values) - 1)
            
            y1 = height - 1 - normalized[idx1]
            y2 = height - 1 - normalized[idx2]
            
            # 简单的线性插值
            steps = max(1, abs(y2 - y1))
            for step in range(steps + 1):
                y = int(y1 + (y2 - y1) * step / steps)
                x = i
                
                if 0 <= y < height and 0 <= x < width:
                    if canvas[y][x] == ' ':
                        canvas[y][x] = '·'
        
        # 转换为字符串
        lines = []
        for row in canvas:
            lines.append(''.join(row))
        
        # 添加标题和坐标轴标签
        result = []
        result.append(f"数值范围: {min_val:.3f} - {max_val:.3f}")
        result.append('')
        result.extend(lines)
        result.append('')
        result.append(f"数据点: {len(values)}")
        
        return '\n'.join(result)
    
    @staticmethod
    def bar_chart(data: Dict[str, float], width: int = 60, height: int = 20) -> str:
        """生成柱状ASCII图表
        
        Args:
            data: 数据字典 {标签: 值}
            width: 图表宽度（字符）
            height: 图表高度（行数）
            
        Returns:
            ASCII图表字符串
        """
        if not data:
            return "无数据"
        
        # 归一化数值
        values = list(data.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            normalized = [height // 2] * len(values)
        else:
            normalized = [
                int((val - min_val) / (max_val - min_val) * (height - 1))
                for val in values
            ]
        
        # 计算条形宽度
        bar_width = max(1, width // len(data) - 2)
        if bar_width < 1:
            bar_width = 1
        
        # 创建画布
        canvas = [[' ' for _ in range(width)] for _ in range(height)]
        
        # 绘制Y轴
        for y in range(height):
            canvas[y][0] = '│'
        
        # 绘制X轴
        for x in range(width):
            canvas[height-1][x] = '─'
        
        canvas[height-1][0] = '└'
        
        # 绘制条形
        labels = list(data.keys())
        
        for i, (label, value) in enumerate(data.items()):
            bar_height = normalized[i]
            x_start = 2 + i * (bar_width + 1)
            
            # 确保不超过宽度
            if x_start >= width:
                break
            
            # 绘制条形
            for y in range(height - bar_height, height):
                for x in range(x_start, min(x_start + bar_width, width)):
                    canvas[y][x] = '█'
            
            # 添加标签（简化）
            if bar_width >= 3:
                label_chars = label[:bar_width]
                y_label = height - 2
                for j, char in enumerate(label_chars):
                    x = x_start + j
                    if x < width and y_label < height:
                        canvas[y_label][x] = char
        
        # 转换为字符串
        lines = []
        for row in canvas:
            lines.append(''.join(row))
        
        # 添加标题
        result = []
        result.append(f"数值范围: {min_val:.3f} - {max_val:.3f}")
        result.append('')
        result.extend(lines)
        result.append('')
        result.append(f"数据项: {len(data)}")
        
        return '\n'.join(result)
    
    @staticmethod
    def sparkline(values: List[float]) -> str:
        """生成迷你Sparkline图表
        
        Args:
            values: 数值列表
            
        Returns:
            Sparkline字符串
        """
        if not values:
            return ""
        
        # 使用Unicode字符
        chars = "▁▂▃▄▅▆▇█"
        
        # 归一化数值
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            normalized = [len(chars) // 2] * len(values)
        else:
            normalized = [
                int((val - min_val) / (max_val - min_val) * (len(chars) - 1))
                for val in values
            ]
        
        # 生成Sparkline
        sparkline = ''.join(chars[idx] for idx in normalized)
        
        return sparkline


class EvolutionVisualizer:
    """演化可视化器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化演化可视化器
        
        Args:
            config: 配置字典
        """
        self.config = VisualizationConfig(**(config or {}))
        self.data_points: List[EvolutionDataPoint] = []
        self.renderer = ASCIIChartRenderer()
    
    def add_data_point(self, data_point: EvolutionDataPoint):
        """添加数据点"""
        self.data_points.append(data_point)
        
        # 限制数据点数量
        if len(self.data_points) > self.config.max_data_points:
            self.data_points = self.data_points[-self.config.max_data_points:]
    
    def add_data_from_dict(self, data_dict: Dict[str, Any]):
        """从字典添加数据点"""
        data_point = EvolutionDataPoint(
            timestamp=data_dict.get("timestamp", time.time()),
            generation=data_dict.get("generation", len(self.data_points)),
            performance_metrics=data_dict.get("performance_metrics", {}),
            architecture_id=data_dict.get("architecture_id", ""),
            architecture_summary=data_dict.get("architecture_summary", {}),
            decision_info=data_dict.get("decision_info")
        )
        self.add_data_point(data_point)
    
    def generate_text_report(self) -> str:
        """生成文本报告"""
        if not self.data_points:
            return "无演化数据"
        
        report_lines = []
        
        # 标题
        report_lines.append("=" * 80)
        report_lines.append("演化分析报告")
        report_lines.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"数据点数: {len(self.data_points)}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 摘要统计
        report_lines.append("摘要统计")
        report_lines.append("-" * 40)
        
        if self.data_points:
            first_dp = self.data_points[0]
            last_dp = self.data_points[-1]
            
            # 时间范围
            time_range = last_dp.timestamp - first_dp.timestamp
            time_str = str(datetime.timedelta(seconds=int(time_range)))
            
            report_lines.append(f"时间范围: {time_str}")
            report_lines.append(f"演化代数: {first_dp.generation} → {last_dp.generation}")
            
            # 性能改进
            if "accuracy" in first_dp.performance_metrics and "accuracy" in last_dp.performance_metrics:
                acc_improvement = last_dp.performance_metrics["accuracy"] - first_dp.performance_metrics["accuracy"]
                report_lines.append(f"准确率改进: {acc_improvement:.3f}")
            
            # 架构变化
            if first_dp.architecture_id != last_dp.architecture_id:
                report_lines.append(f"架构变化: {first_dp.architecture_id} → {last_dp.architecture_id}")
        
        report_lines.append("")
        
        # 性能趋势
        if len(self.data_points) >= 2 and self.config.show_ascii_charts:
            report_lines.append("性能趋势")
            report_lines.append("-" * 40)
            
            # 提取准确率数据
            accuracies = []
            valid_points = []
            
            for dp in self.data_points:
                if "accuracy" in dp.performance_metrics:
                    accuracies.append(dp.performance_metrics["accuracy"])
                    valid_points.append(dp)
            
            if accuracies:
                chart = self.renderer.line_chart(
                    accuracies,
                    width=self.config.chart_width,
                    height=self.config.chart_height
                )
                report_lines.append(chart)
                
                # 添加趋势分析
                if len(accuracies) >= 5:
                    recent_avg = sum(accuracies[-5:]) / 5
                    overall_avg = sum(accuracies) / len(accuracies)
                    
                    if recent_avg > overall_avg * 1.05:
                        trend = "上升"
                    elif recent_avg < overall_avg * 0.95:
                        trend = "下降"
                    else:
                        trend = "稳定"
                    
                    report_lines.append(f"趋势分析: {trend} (最近5代平均: {recent_avg:.3f})")
            else:
                report_lines.append("无准确率数据")
            
            report_lines.append("")
        
        # 演化历史详情
        if self.config.show_details:
            report_lines.append("演化历史详情")
            report_lines.append("-" * 40)
            
            # 按代分组
            for i, dp in enumerate(self.data_points[-10:]):  # 显示最近10个点
                time_str = datetime.datetime.fromtimestamp(dp.timestamp).strftime('%H:%M:%S')
                
                report_lines.append(f"第{dp.generation}代 ({time_str})")
                report_lines.append(f"  架构: {dp.architecture_id}")
                
                # 性能指标
                perf_summary = []
                for metric, value in dp.performance_metrics.items():
                    perf_summary.append(f"{metric}: {value:.3f}")
                
                if perf_summary:
                    report_lines.append(f"  性能: {', '.join(perf_summary)}")
                
                # 决策信息
                if dp.decision_info:
                    decision_type = dp.decision_info.get("type", "未知")
                    decision_reason = dp.decision_info.get("reason", "")
                    report_lines.append(f"  决策: {decision_type} - {decision_reason[:50]}...")
                
                report_lines.append("")
        
        # 架构比较
        if len(self.data_points) >= 2:
            report_lines.append("架构比较")
            report_lines.append("-" * 40)
            
            # 比较第一个和最后一个架构
            first_arch = self.data_points[0].architecture_summary
            last_arch = self.data_points[-1].architecture_summary
            
            if first_arch and last_arch:
                comparison = self._compare_architectures(first_arch, last_arch)
                
                report_lines.append(f"架构 {self.data_points[0].architecture_id} → {self.data_points[-1].architecture_id}")
                report_lines.append(f"相似度: {comparison.similarity_score:.3f}")
                
                if comparison.added_components:
                    report_lines.append(f"新增组件: {', '.join(comparison.added_components)}")
                
                if comparison.removed_components:
                    report_lines.append(f"移除组件: {', '.join(comparison.removed_components)}")
                
                if comparison.modified_components:
                    report_lines.append("修改组件:")
                    for comp, old_val, new_val in comparison.modified_components[:3]:  # 最多显示3个
                        report_lines.append(f"  {comp}: {old_val} → {new_val}")
                    
                    if len(comparison.modified_components) > 3:
                        report_lines.append(f"  ... 共{len(comparison.modified_components)}个组件被修改")
                
                # 性能差异
                if comparison.performance_delta:
                    delta_str = []
                    for metric, delta in comparison.performance_delta.items():
                        if delta >= 0:
                            delta_str.append(f"{metric}: +{delta:.3f}")
                        else:
                            delta_str.append(f"{metric}: {delta:.3f}")
                    
                    if delta_str:
                        report_lines.append(f"性能变化: {', '.join(delta_str)}")
            
            report_lines.append("")
        
        # 建议和改进点
        report_lines.append("建议和改进点")
        report_lines.append("-" * 40)
        
        suggestions = self._generate_suggestions()
        for i, suggestion in enumerate(suggestions, 1):
            report_lines.append(f"{i}. {suggestion}")
        
        report_lines.append("")
        
        # 结尾
        report_lines.append("=" * 80)
        report_lines.append("报告结束")
        
        return '\n'.join(report_lines)
    
    def generate_html_report(self) -> str:
        """生成HTML报告（简化版）"""
        # 简化实现：返回包含JSON数据的简单HTML
        # 在实际系统中，可以使用Jinja2模板引擎
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>演化分析报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .data-point { margin: 10px 0; padding: 10px; background: #f9f9f9; border-left: 4px solid #007acc; }
                .metric { display: inline-block; margin-right: 20px; }
                .chart { font-family: monospace; white-space: pre; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>演化分析报告</h1>
                <p>生成时间: """ + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                <p>数据点数: """ + str(len(self.data_points)) + """</p>
            </div>
        """
        
        if self.data_points:
            # 数据点列表
            html += '<div class="section"><h2>演化历史</h2>'
            for dp in self.data_points[-20:]:  # 最多显示20个
                time_str = datetime.datetime.fromtimestamp(dp.timestamp).strftime('%H:%M:%S')
                
                html += f"""
                <div class="data-point">
                    <h3>第{dp.generation}代 ({time_str})</h3>
                    <p><strong>架构:</strong> {dp.architecture_id}</p>
                    <div class="metrics">
                """
                
                for metric, value in dp.performance_metrics.items():
                    html += f'<span class="metric"><strong>{metric}:</strong> {value:.3f}</span>'
                
                html += '</div></div>'
            
            html += '</div>'
        
        html += """
            <div class="section">
                <h2>ASCII图表</h2>
                <div class="chart">
        """
        
        # 添加ASCII图表
        if len(self.data_points) >= 2:
            accuracies = []
            for dp in self.data_points:
                if "accuracy" in dp.performance_metrics:
                    accuracies.append(dp.performance_metrics["accuracy"])
            
            if accuracies:
                chart = self.renderer.line_chart(
                    accuracies,
                    width=60,
                    height=15
                )
                html += f"<pre>{chart}</pre>"
        
        html += """
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def generate_json_report(self) -> str:
        """生成JSON报告"""
        report_data = {
            "metadata": {
                "generated_at": datetime.datetime.now().isoformat(),
                "data_points_count": len(self.data_points),
                "config": self.config.__dict__
            },
            "summary": self._generate_summary(),
            "data_points": [
                {
                    "timestamp": dp.timestamp,
                    "generation": dp.generation,
                    "architecture_id": dp.architecture_id,
                    "performance_metrics": dp.performance_metrics,
                    "architecture_summary": dp.architecture_summary,
                    "decision_info": dp.decision_info
                }
                for dp in self.data_points
            ],
            "analysis": self._generate_analysis()
        }
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)
    
    def generate_report(self, format: Optional[str] = None) -> str:
        """生成报告
        
        Args:
            format: 报告格式 (text, html, json)
            
        Returns:
            报告字符串
        """
        report_format = format or self.config.output_format
        
        if report_format == "html":
            return self.generate_html_report()
        elif report_format == "json":
            return self.generate_json_report()
        else:  # text
            return self.generate_text_report()
    
    def _compare_architectures(self, arch_a: Dict[str, Any], arch_b: Dict[str, Any]) -> ArchitectureComparison:
        """比较两个架构"""
        # 简化实现
        components_a = set(str(k) for k in arch_a.keys())
        components_b = set(str(k) for k in arch_b.keys())
        
        added = list(components_b - components_a)
        removed = list(components_a - components_b)
        
        # 查找修改的组件
        modified = []
        common = components_a & components_b
        
        for comp in common:
            val_a = arch_a.get(comp)
            val_b = arch_b.get(comp)
            
            if str(val_a) != str(val_b):
                modified.append((comp, str(val_a), str(val_b)))
        
        # 计算相似度
        total_components = len(components_a | components_b)
        if total_components == 0:
            similarity = 1.0
        else:
            similar_components = len(common)
            similarity = similar_components / total_components
        
        # 性能差异（简化）
        performance_delta = {}
        
        return ArchitectureComparison(
            version_a="A",
            version_b="B",
            similarity_score=similarity,
            added_components=added,
            removed_components=removed,
            modified_components=modified,
            performance_delta=performance_delta
        )
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成摘要"""
        if not self.data_points:
            return {"message": "无数据"}
        
        summary = {
            "total_generations": self.data_points[-1].generation - self.data_points[0].generation + 1,
            "time_range_seconds": self.data_points[-1].timestamp - self.data_points[0].timestamp,
            "unique_architectures": len(set(dp.architecture_id for dp in self.data_points))
        }
        
        # 性能统计
        performance_stats = {}
        for metric in ["accuracy", "efficiency", "robustness"]:
            values = []
            for dp in self.data_points:
                if metric in dp.performance_metrics:
                    values.append(dp.performance_metrics[metric])
            
            if values:
                performance_stats[metric] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "latest": values[-1] if values else None
                }
        
        summary["performance_stats"] = performance_stats
        
        return summary
    
    def _generate_analysis(self) -> Dict[str, Any]:
        """生成分析"""
        analysis = {
            "trends": {},
            "insights": [],
            "recommendations": []
        }
        
        if len(self.data_points) >= 3:
            # 趋势分析
            accuracies = []
            for dp in self.data_points:
                if "accuracy" in dp.performance_metrics:
                    accuracies.append(dp.performance_metrics["accuracy"])
            
            if accuracies:
                # 简单趋势检测
                first_half_avg = sum(accuracies[:len(accuracies)//2]) / (len(accuracies)//2)
                second_half_avg = sum(accuracies[len(accuracies)//2:]) / (len(accuracies) - len(accuracies)//2)
                
                if second_half_avg > first_half_avg * 1.1:
                    analysis["trends"]["accuracy"] = "显著上升"
                elif second_half_avg > first_half_avg * 1.01:
                    analysis["trends"]["accuracy"] = "缓慢上升"
                elif second_half_avg < first_half_avg * 0.9:
                    analysis["trends"]["accuracy"] = "显著下降"
                elif second_half_avg < first_half_avg * 0.99:
                    analysis["trends"]["accuracy"] = "缓慢下降"
                else:
                    analysis["trends"]["accuracy"] = "稳定"
        
        # 生成洞察
        if len(self.data_points) > 10:
            analysis["insights"].append("演化过程已运行较长时间，建议分析收敛性")
        
        unique_arch_count = len(set(dp.architecture_id for dp in self.data_points))
        if unique_arch_count < len(self.data_points) * 0.5:
            analysis["insights"].append("架构变化较少，可能需要增加探索性演化")
        
        # 生成建议
        analysis["recommendations"].extend(self._generate_suggestions())
        
        return analysis
    
    def _generate_suggestions(self) -> List[str]:
        """生成建议列表"""
        suggestions = []
        
        if not self.data_points:
            suggestions.append("收集更多演化数据以进行分析")
            return suggestions
        
        # 检查数据量
        if len(self.data_points) < 10:
            suggestions.append("数据点较少，建议运行更多演化代以获得更可靠的分析")
        
        # 检查性能趋势
        if len(self.data_points) >= 5:
            recent_accuracies = []
            for dp in self.data_points[-5:]:
                if "accuracy" in dp.performance_metrics:
                    recent_accuracies.append(dp.performance_metrics["accuracy"])
            
            if recent_accuracies:
                recent_avg = sum(recent_accuracies) / len(recent_accuracies)
                overall_accuracies = []
                for dp in self.data_points:
                    if "accuracy" in dp.performance_metrics:
                        overall_accuracies.append(dp.performance_metrics["accuracy"])
                
                if overall_accuracies:
                    overall_avg = sum(overall_accuracies) / len(overall_accuracies)
                    
                    if recent_avg < overall_avg * 0.95:
                        suggestions.append("最近几代性能下降，考虑调整演化策略或增加多样性")
                    elif recent_avg > overall_avg * 1.05:
                        suggestions.append("最近几代性能提升明显，当前演化策略有效")
        
        # 架构多样性建议
        unique_arch_count = len(set(dp.architecture_id for dp in self.data_points))
        arch_diversity_ratio = unique_arch_count / len(self.data_points)
        
        if arch_diversity_ratio < 0.3:
            suggestions.append("架构多样性较低，建议增加突变率或引入更多演化算子")
        elif arch_diversity_ratio > 0.8:
            suggestions.append("架构多样性较高，可能缺乏收敛，建议增加选择压力")
        
        # 默认建议
        suggestions.append("定期监控演化过程，确保目标对齐")
        suggestions.append("考虑引入多目标优化以平衡不同性能指标")
        
        return suggestions


def create_evolution_visualizer(config: Optional[Dict[str, Any]] = None) -> EvolutionVisualizer:
    """创建演化可视化器实例
    
    Args:
        config: 配置字典
        
    Returns:
        演化可视化器实例
    """
    return EvolutionVisualizer(config)


# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("演化可视化分析工具测试")
    print("=" * 80)
    
    try:
        # 创建可视化器
        config = {
            "output_format": "text",
            "show_ascii_charts": True,
            "chart_width": 70,
            "chart_height": 15
        }
        
        visualizer = create_evolution_visualizer(config)
        
        print("演化可视化器创建成功")
        
        # 添加模拟数据
        print("\n添加模拟演化数据...")
        
        import random
        for i in range(20):
            data_point = EvolutionDataPoint(
                timestamp=time.time() - (20 - i) * 3600,  # 模拟过去20小时
                generation=i + 1,
                performance_metrics={
                    "accuracy": 0.7 + (i * 0.015) + (random.random() * 0.05 - 0.025),
                    "efficiency": 0.6 + (i * 0.01) + (random.random() * 0.03 - 0.015),
                    "robustness": 0.5 + (i * 0.02) + (random.random() * 0.04 - 0.02)
                },
                architecture_id=f"arch_{i // 5 + 1}",  # 每5代更换架构
                architecture_summary={
                    "type": "classification",
                    "layers": i % 5 + 3,
                    "activation": "relu" if i % 2 == 0 else "sigmoid",
                    "parameters": 1000 * (i % 3 + 1)
                },
                decision_info={
                    "type": "mutation" if i % 3 != 0 else "crossover",
                    "reason": "性能优化" if i % 2 == 0 else "架构调整"
                } if i % 2 == 0 else None
            )
            
            visualizer.add_data_point(data_point)
        
        print(f"已添加 {len(visualizer.data_points)} 个数据点")
        
        # 生成文本报告
        print("\n" + "=" * 80)
        print("生成文本报告")
        print("=" * 80)
        
        text_report = visualizer.generate_text_report()
        print(text_report)
        
        # 生成JSON报告
        print("\n" + "=" * 80)
        print("生成JSON报告")
        print("=" * 80)
        
        json_report = visualizer.generate_json_report()
        print("JSON报告长度:", len(json_report), "字符")
        
        # 解析JSON以显示摘要
        json_data = json.loads(json_report)
        summary = json_data.get("summary", {})
        
        print(f"\n摘要统计:")
        print(f"  总演化代数: {summary.get('total_generations', 'N/A')}")
        print(f"  时间范围: {summary.get('time_range_seconds', 'N/A'):.0f} 秒")
        print(f"  唯一架构数: {summary.get('unique_architectures', 'N/A')}")
        
        # 生成HTML报告
        print("\n" + "=" * 80)
        print("生成HTML报告")
        print("=" * 80)
        
        html_report = visualizer.generate_html_report()
        print("HTML报告长度:", len(html_report), "字符")
        
        # 保存HTML报告到文件
        html_file = "evolution_report.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_report)
        
        print(f"HTML报告已保存到: {html_file}")
        
        print("\n" + "=" * 80)
        print("✓ 演化可视化分析工具测试完成")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()