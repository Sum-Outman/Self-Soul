"""
全链路可解释性系统

修复计划第五阶段：完善用户体验（自然交互+多模态反馈）
任务5.3：创建全链路可解释性系统

核心功能：
1. 提供从输入到输出的完整解释
2. 可视化：为什么选择这个输出模态？如何融合多个意图？生成逻辑是什么？
3. 支持调试和优化，提升系统透明度
"""

import sys
import os
import logging
import time
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import inspect

# 导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 配置日志
logger = logging.getLogger("end_to_end_explainability")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class ExplanationLevel(Enum):
    """解释级别"""
    BRIEF = "brief"        # 简要：仅核心决策点
    DETAILED = "detailed"  # 详细：主要处理步骤
    DEBUG = "debug"        # 调试：所有细节，包括内部状态
    VISUAL = "visual"      # 可视化：适合图形展示


class ExplanationType(Enum):
    """解释类型"""
    INPUT_PARSING = "input_parsing"      # 输入解析
    INTENT_FUSION = "intent_fusion"      # 意图融合
    MODALITY_SELECTION = "modality_selection"  # 模态选择
    GENERATION_LOGIC = "generation_logic"  # 生成逻辑
    OUTPUT_OPTIMIZATION = "output_optimization"  # 输出优化
    ERROR_HANDLING = "error_handling"    # 错误处理
    PERFORMANCE = "performance"          # 性能分析


@dataclass
class ExplanationStep:
    """解释步骤"""
    step_id: str
    explanation_type: ExplanationType
    level: ExplanationLevel
    timestamp: float
    component: str
    operation: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    reasoning: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "step_id": self.step_id,
            "explanation_type": self.explanation_type.value,
            "level": self.level.value,
            "timestamp": self.timestamp,
            "component": self.component,
            "operation": self.operation,
            "input_summary": self._summarize_data(self.input_data),
            "output_summary": self._summarize_data(self.output_data),
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    def _summarize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """总结数据"""
        summary = {}
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)):
                summary[key] = value
            elif isinstance(value, list):
                summary[key] = f"list[{len(value)}]"
            elif isinstance(value, dict):
                summary[key] = f"dict[{len(value)} keys]"
            elif hasattr(value, '__class__'):
                summary[key] = value.__class__.__name__
            else:
                summary[key] = str(type(value))
        
        return summary


@dataclass
class ExplanationFlow:
    """解释流程"""
    flow_id: str
    start_time: float
    end_time: Optional[float] = None
    steps: List[ExplanationStep] = field(default_factory=list)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    overall_confidence: float = 1.0
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.flow_id:
            self.flow_id = f"flow_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    def add_step(self, step: ExplanationStep) -> None:
        """添加步骤"""
        self.steps.append(step)
    
    def complete(self) -> None:
        """完成流程"""
        self.end_time = time.time()
        
        # 计算总体置信度
        if self.steps:
            confidences = [step.confidence for step in self.steps]
            self.overall_confidence = sum(confidences) / len(confidences)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "flow_id": self.flow_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time if self.end_time else None,
            "step_count": len(self.steps),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "overall_confidence": self.overall_confidence,
            "steps": [step.to_dict() for step in self.steps]
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        step_types = {}
        for step in self.steps:
            step_type = step.explanation_type.value
            step_types[step_type] = step_types.get(step_type, 0) + 1
        
        return {
            "flow_id": self.flow_id,
            "duration": self.end_time - self.start_time if self.end_time else None,
            "step_count": len(self.steps),
            "step_types": step_types,
            "overall_confidence": self.overall_confidence
        }


class EndToEndExplainability:
    """
    全链路可解释性系统
    
    核心功能：
    1. 提供从输入到输出的完整解释
    2. 可视化：为什么选择这个输出模态？如何融合多个意图？生成逻辑是什么？
    3. 支持调试和优化，提升系统透明度
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化可解释性系统
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 流程存储
        self.flows: Dict[str, ExplanationFlow] = {}
        self.active_flows: Dict[str, ExplanationFlow] = {}
        
        # 组件追踪器
        self.component_trackers = {}
        
        # 可视化生成器
        self.visualization_generator = VisualizationGenerator()
        
        # 统计分析器
        self.statistics_analyzer = StatisticsAnalyzer()
        
        # 配置
        self.default_level = ExplanationLevel(self.config.get("default_level", "detailed"))
        self.auto_capture = self.config.get("auto_capture", True)
        
        # 统计信息
        self.stats = {
            "total_flows": 0,
            "active_flows": 0,
            "total_steps": 0,
            "average_steps_per_flow": 0.0,
            "average_confidence": 0.0,
            "explanation_types": {}
        }
        
        logger.info(f"全链路可解释性系统初始化完成，默认级别: {self.default_level.value}")
    
    def start_flow(self, user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  initial_data: Optional[Dict[str, Any]] = None) -> str:
        """
        开始新的解释流程
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            initial_data: 初始数据
            
        Returns:
            流程ID
        """
        flow = ExplanationFlow(
            flow_id="",
            start_time=time.time(),
            user_id=user_id,
            session_id=session_id
        )
        
        self.flows[flow.flow_id] = flow
        self.active_flows[flow.flow_id] = flow
        
        self.stats["total_flows"] += 1
        self.stats["active_flows"] = len(self.active_flows)
        
        # 记录初始步骤（如果提供初始数据）
        if initial_data:
            self.record_step(
                flow_id=flow.flow_id,
                explanation_type=ExplanationType.INPUT_PARSING,
                component="ExplainabilitySystem",
                operation="start_flow",
                input_data={"initial_data": initial_data},
                output_data={"flow_id": flow.flow_id},
                reasoning=f"开始新的解释流程，用户: {user_id or 'anonymous'}",
                level=ExplanationLevel.BRIEF
            )
        
        logger.info(f"开始解释流程: {flow.flow_id}, 用户: {user_id or 'anonymous'}")
        
        return flow.flow_id
    
    def end_flow(self, flow_id: str) -> Optional[ExplanationFlow]:
        """
        结束解释流程
        
        Args:
            flow_id: 流程ID
            
        Returns:
            完成的流程（如果存在）
        """
        if flow_id not in self.active_flows:
            logger.warning(f"流程 {flow_id} 不存在或未激活")
            return None
        
        flow = self.active_flows[flow_id]
        flow.complete()
        
        # 从激活流程中移除
        del self.active_flows[flow_id]
        
        # 更新统计
        self.stats["active_flows"] = len(self.active_flows)
        self.stats["average_steps_per_flow"] = (
            self.stats["average_steps_per_flow"] * (self.stats["total_flows"] - 1) + len(flow.steps)
        ) / self.stats["total_flows"]
        
        if flow.steps:
            confidences = [step.confidence for step in flow.steps]
            flow_confidence = sum(confidences) / len(confidences)
            self.stats["average_confidence"] = (
                self.stats["average_confidence"] * (self.stats["total_flows"] - 1) + flow_confidence
            ) / self.stats["total_flows"]
        
        # 更新解释类型统计
        for step in flow.steps:
            step_type = step.explanation_type.value
            self.stats["explanation_types"][step_type] = self.stats["explanation_types"].get(step_type, 0) + 1
        
        logger.info(f"结束解释流程: {flow_id}, 步骤数: {len(flow.steps)}, 置信度: {flow.overall_confidence:.2f}")
        
        return flow
    
    def record_step(self, flow_id: str, explanation_type: ExplanationType,
                   component: str, operation: str,
                   input_data: Dict[str, Any], output_data: Dict[str, Any],
                   reasoning: str, confidence: float = 1.0,
                   level: Optional[ExplanationLevel] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Optional[ExplanationStep]:
        """
        记录解释步骤
        
        Args:
            flow_id: 流程ID
            explanation_type: 解释类型
            component: 组件名称
            operation: 操作名称
            input_data: 输入数据
            output_data: 输出数据
            reasoning: 解释说明
            confidence: 置信度
            level: 解释级别
            metadata: 元数据
            
        Returns:
            创建的步骤（如果成功）
        """
        if flow_id not in self.active_flows:
            logger.warning(f"流程 {flow_id} 不存在或未激活")
            return None
        
        flow = self.active_flows[flow_id]
        
        # 确定级别
        if level is None:
            level = self.default_level
        
        # 创建步骤
        step = ExplanationStep(
            step_id=f"step_{len(flow.steps)}_{int(time.time() * 1000)}",
            explanation_type=explanation_type,
            level=level,
            timestamp=time.time(),
            component=component,
            operation=operation,
            input_data=input_data,
            output_data=output_data,
            reasoning=reasoning,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        # 添加到流程
        flow.add_step(step)
        
        self.stats["total_steps"] += 1
        
        logger.debug(f"记录步骤: {step.step_id}, 类型: {explanation_type.value}, 组件: {component}")
        
        return step
    
    def get_flow(self, flow_id: str, include_steps: bool = True) -> Optional[Dict[str, Any]]:
        """
        获取流程详情
        
        Args:
            flow_id: 流程ID
            include_steps: 是否包含步骤详情
            
        Returns:
            流程详情字典
        """
        if flow_id not in self.flows:
            return None
        
        flow = self.flows[flow_id]
        result = flow.to_dict()
        
        if not include_steps:
            result["steps"] = f"{len(flow.steps)} steps"
        
        return result
    
    def get_flow_summary(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """
        获取流程摘要
        
        Args:
            flow_id: 流程ID
            
        Returns:
            流程摘要字典
        """
        if flow_id not in self.flows:
            return None
        
        flow = self.flows[flow_id]
        return flow.get_summary()
    
    def analyze_flow(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """
        分析流程
        
        Args:
            flow_id: 流程ID
            
        Returns:
            分析结果
        """
        if flow_id not in self.flows:
            return None
        
        flow = self.flows[flow_id]
        
        # 使用统计分析器
        analysis = self.statistics_analyzer.analyze(flow)
        
        # 生成可视化建议
        visualization_suggestions = self.visualization_generator.get_suggestions(flow)
        
        result = {
            "flow_id": flow_id,
            "summary": flow.get_summary(),
            "analysis": analysis,
            "visualization_suggestions": visualization_suggestions,
            "improvement_recommendations": self._generate_improvement_recommendations(flow)
        }
        
        return result
    
    def _generate_improvement_recommendations(self, flow: ExplanationFlow) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 检查置信度
        if flow.overall_confidence < 0.7:
            recommendations.append("总体置信度较低，建议检查决策逻辑")
        
        # 检查步骤数量
        if len(flow.steps) < 3:
            recommendations.append("处理步骤较少，可能缺少关键决策点")
        elif len(flow.steps) > 20:
            recommendations.append("处理步骤过多，可能影响性能")
        
        # 检查错误处理步骤
        error_steps = [s for s in flow.steps if s.explanation_type == ExplanationType.ERROR_HANDLING]
        if error_steps:
            recommendations.append(f"检测到 {len(error_steps)} 个错误处理步骤，建议优化容错机制")
        
        return recommendations
    
    def generate_explanation(self, flow_id: str, level: ExplanationLevel = None) -> Optional[str]:
        """
        生成自然语言解释
        
        Args:
            flow_id: 流程ID
            level: 解释级别
            
        Returns:
            自然语言解释
        """
        if flow_id not in self.flows:
            return None
        
        if level is None:
            level = self.default_level
        
        flow = self.flows[flow_id]
        
        explanation_parts = []
        
        # 开头
        explanation_parts.append(f"## 处理流程解释 (流程ID: {flow_id})")
        explanation_parts.append(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(flow.start_time))}")
        
        if flow.end_time:
            duration = flow.end_time - flow.start_time
            explanation_parts.append(f"处理时长: {duration:.2f}秒")
        
        explanation_parts.append(f"总步骤数: {len(flow.steps)}")
        explanation_parts.append(f"总体置信度: {flow.overall_confidence:.2%}")
        
        # 按步骤解释
        explanation_parts.append("\n## 处理步骤:")
        
        for i, step in enumerate(flow.steps, 1):
            if level == ExplanationLevel.BRIEF and step.explanation_type not in [
                ExplanationType.INPUT_PARSING,
                ExplanationType.INTENT_FUSION,
                ExplanationType.MODALITY_SELECTION,
                ExplanationType.GENERATION_LOGIC
            ]:
                continue
            
            step_explanation = f"\n{i}. **{step.component} - {step.operation}**"
            step_explanation += f"\n   类型: {step.explanation_type.value}"
            step_explanation += f"\n   解释: {step.reasoning}"
            step_explanation += f"\n   置信度: {step.confidence:.2%}"
            
            if level in [ExplanationLevel.DETAILED, ExplanationLevel.DEBUG]:
                step_explanation += f"\n   输入: {self._format_data_summary(step.input_data)}"
                step_explanation += f"\n   输出: {self._format_data_summary(step.output_data)}"
            
            explanation_parts.append(step_explanation)
        
        # 总结
        explanation_parts.append("\n## 总结:")
        
        # 统计不同类型步骤
        type_counts = {}
        for step in flow.steps:
            step_type = step.explanation_type.value
            type_counts[step_type] = type_counts.get(step_type, 0) + 1
        
        for step_type, count in type_counts.items():
            explanation_parts.append(f"- {step_type}: {count}个步骤")
        
        # 关键决策点
        key_decisions = [s for s in flow.steps if s.confidence >= 0.8]
        if key_decisions:
            explanation_parts.append(f"- 关键决策点: {len(key_decisions)}个")
        
        return "\n".join(explanation_parts)
    
    def _format_data_summary(self, data: Dict[str, Any]) -> str:
        """格式化数据摘要"""
        if not data:
            return "无"
        
        items = []
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)):
                if isinstance(value, str) and len(value) > 50:
                    items.append(f"{key}: '{value[:50]}...'")
                else:
                    items.append(f"{key}: {value}")
            elif isinstance(value, list):
                items.append(f"{key}: 列表[{len(value)}项]")
            elif isinstance(value, dict):
                items.append(f"{key}: 字典[{len(value)}键]")
            else:
                items.append(f"{key}: {type(value).__name__}")
        
        return ", ".join(items)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 添加流程统计
        stats["completed_flows"] = stats["total_flows"] - stats["active_flows"]
        
        # 计算平均持续时间
        durations = []
        for flow_id, flow in self.flows.items():
            if flow.end_time:
                durations.append(flow.end_time - flow.start_time)
        
        if durations:
            stats["average_duration"] = sum(durations) / len(durations)
        else:
            stats["average_duration"] = 0.0
        
        return stats
    
    def clear_old_flows(self, max_age_hours: float = 24.0) -> int:
        """
        清理旧流程
        
        Args:
            max_age_hours: 最大保留时间（小时）
            
        Returns:
            清理的流程数量
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        to_delete = []
        for flow_id, flow in self.flows.items():
            if flow.end_time and (current_time - flow.end_time) > max_age_seconds:
                to_delete.append(flow_id)
        
        for flow_id in to_delete:
            del self.flows[flow_id]
            if flow_id in self.active_flows:
                del self.active_flows[flow_id]
        
        logger.info(f"清理了 {len(to_delete)} 个旧流程")
        
        return len(to_delete)


# ==================== 辅助类实现 ====================

class VisualizationGenerator:
    """可视化生成器"""
    
    def __init__(self):
        """初始化可视化生成器"""
        pass
    
    def get_suggestions(self, flow: ExplanationFlow) -> List[Dict[str, Any]]:
        """获取可视化建议"""
        suggestions = []
        
        # 检查是否需要流程图
        if len(flow.steps) >= 5:
            suggestions.append({
                "type": "flow_chart",
                "description": "生成处理流程图",
                "priority": "high",
                "reason": "步骤数量多，适合可视化展示处理流程"
            })
        
        # 检查是否需要置信度图表
        confidences = [step.confidence for step in flow.steps]
        if any(c < 0.7 for c in confidences):
            suggestions.append({
                "type": "confidence_chart",
                "description": "生成置信度变化图表",
                "priority": "medium",
                "reason": "存在低置信度步骤，需要可视化分析"
            })
        
        # 检查是否需要类型分布图
        type_counts = {}
        for step in flow.steps:
            step_type = step.explanation_type.value
            type_counts[step_type] = type_counts.get(step_type, 0) + 1
        
        if len(type_counts) >= 3:
            suggestions.append({
                "type": "type_distribution",
                "description": "生成步骤类型分布图",
                "priority": "low",
                "reason": "多种步骤类型，适合展示分布"
            })
        
        return suggestions
    
    def generate_flow_chart(self, flow: ExplanationFlow) -> Dict[str, Any]:
        """生成流程图数据"""
        nodes = []
        edges = []
        
        for i, step in enumerate(flow.steps):
            # 创建节点
            node = {
                "id": step.step_id,
                "label": f"{step.component}\n{step.operation}",
                "type": step.explanation_type.value,
                "confidence": step.confidence,
                "metadata": {
                    "timestamp": step.timestamp,
                    "reasoning": step.reasoning[:100] + "..." if len(step.reasoning) > 100 else step.reasoning
                }
            }
            nodes.append(node)
            
            # 创建边（连接到前一个节点）
            if i > 0:
                prev_step = flow.steps[i-1]
                edge = {
                    "from": prev_step.step_id,
                    "to": step.step_id,
                    "label": f"步骤 {i} → {i+1}"
                }
                edges.append(edge)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "flow_id": flow.flow_id,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "generated_at": time.time()
            }
        }


class StatisticsAnalyzer:
    """统计分析器"""
    
    def __init__(self):
        """初始化统计分析器"""
        pass
    
    def analyze(self, flow: ExplanationFlow) -> Dict[str, Any]:
        """分析流程"""
        analysis = {
            "basic_statistics": self._calculate_basic_stats(flow),
            "confidence_analysis": self._analyze_confidence(flow),
            "performance_analysis": self._analyze_performance(flow),
            "pattern_detection": self._detect_patterns(flow),
            "anomaly_detection": self._detect_anomalies(flow)
        }
        
        return analysis
    
    def _calculate_basic_stats(self, flow: ExplanationFlow) -> Dict[str, Any]:
        """计算基础统计"""
        stats = {
            "step_count": len(flow.steps),
            "duration": flow.end_time - flow.start_time if flow.end_time else None,
            "component_count": len(set(step.component for step in flow.steps)),
            "operation_count": len(set(step.operation for step in flow.steps))
        }
        
        return stats
    
    def _analyze_confidence(self, flow: ExplanationFlow) -> Dict[str, Any]:
        """分析置信度"""
        if not flow.steps:
            return {"average": 0.0, "min": 0.0, "max": 0.0, "distribution": []}
        
        confidences = [step.confidence for step in flow.steps]
        
        return {
            "average": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences),
            "std_dev": self._calculate_std_dev(confidences),
            "low_confidence_steps": sum(1 for c in confidences if c < 0.7),
            "high_confidence_steps": sum(1 for c in confidences if c >= 0.9)
        }
    
    def _analyze_performance(self, flow: ExplanationFlow) -> Dict[str, Any]:
        """分析性能"""
        if len(flow.steps) < 2:
            return {"average_step_duration": 0.0, "total_duration": 0.0}
        
        durations = []
        for i in range(1, len(flow.steps)):
            duration = flow.steps[i].timestamp - flow.steps[i-1].timestamp
            durations.append(duration)
        
        total_duration = flow.steps[-1].timestamp - flow.steps[0].timestamp if flow.steps else 0
        
        return {
            "average_step_duration": sum(durations) / len(durations) if durations else 0.0,
            "total_duration": total_duration,
            "slowest_step": max(durations) if durations else 0.0,
            "fastest_step": min(durations) if durations else 0.0
        }
    
    def _detect_patterns(self, flow: ExplanationFlow) -> List[str]:
        """检测模式"""
        patterns = []
        
        # 检测连续的相同组件
        for i in range(len(flow.steps) - 2):
            if (flow.steps[i].component == flow.steps[i+1].component == 
                flow.steps[i+2].component):
                patterns.append(f"连续3个步骤使用相同组件: {flow.steps[i].component}")
                break
        
        # 检测置信度下降模式
        for i in range(len(flow.steps) - 2):
            if (flow.steps[i].confidence > flow.steps[i+1].confidence > 
                flow.steps[i+2].confidence):
                patterns.append("检测到置信度连续下降模式")
                break
        
        return patterns
    
    def _detect_anomalies(self, flow: ExplanationFlow) -> List[Dict[str, Any]]:
        """检测异常"""
        anomalies = []
        
        for i, step in enumerate(flow.steps):
            # 检测极低置信度
            if step.confidence < 0.3:
                anomalies.append({
                    "type": "low_confidence",
                    "step_index": i,
                    "step_id": step.step_id,
                    "confidence": step.confidence,
                    "description": f"步骤 {i+1} 置信度过低: {step.confidence:.2f}"
                })
            
            # 检测长时间步骤（假设超过2秒为异常）
            if i > 0:
                duration = step.timestamp - flow.steps[i-1].timestamp
                if duration > 2.0:
                    anomalies.append({
                        "type": "long_duration",
                        "step_index": i,
                        "step_id": step.step_id,
                        "duration": duration,
                        "description": f"步骤 {i+1} 处理时间过长: {duration:.2f}秒"
                    })
        
        return anomalies
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        
        return variance ** 0.5


# ==================== 测试函数 ====================

def test_end_to_end_explainability() -> None:
    """测试全链路可解释性系统"""
    print("测试全链路可解释性系统...")
    
    # 创建可解释性系统实例
    explainer = EndToEndExplainability({
        "default_level": "detailed",
        "auto_capture": True
    })
    
    # 开始流程
    flow_id = explainer.start_flow(
        user_id="test_user_001",
        session_id="test_session_001",
        initial_data={"input_text": "请帮我分析这张图片"}
    )
    
    print(f"开始流程: {flow_id}")
    
    # 模拟输入解析步骤
    explainer.record_step(
        flow_id=flow_id,
        explanation_type=ExplanationType.INPUT_PARSING,
        component="NaturalHybridInputInterface",
        operation="receive_input",
        input_data={"raw_input": "请帮我分析这张图片", "modality_hint": "text"},
        output_data={"parsed_modality": "text", "content": "请帮我分析这张图片"},
        reasoning="检测到文本输入，解析为文本模态",
        confidence=0.95
    )
    
    # 模拟意图融合步骤
    explainer.record_step(
        flow_id=flow_id,
        explanation_type=ExplanationType.INTENT_FUSION,
        component="IntentFusionEngine",
        operation="fuse_intents",
        input_data={"text_intent": "分析图片", "image_present": True},
        output_data={"fused_intent": "分析用户提供的图片", "confidence": 0.88},
        reasoning="融合文本意图和图像存在信息，生成完整意图",
        confidence=0.88
    )
    
    # 模拟模态选择步骤
    explainer.record_step(
        flow_id=flow_id,
        explanation_type=ExplanationType.MODALITY_SELECTION,
        component="IntelligentOutputSelector",
        operation="select_output_modality",
        input_data={"available_modalities": ["text", "image", "multimodal"], "context": {"environment": "quiet_indoor"}},
        output_data={"selected_modality": "multimodal", "confidence": 0.82},
        reasoning="环境安静，用户需要详细分析，选择多模态输出（文本+图像）",
        confidence=0.82
    )
    
    # 模拟生成逻辑步骤
    explainer.record_step(
        flow_id=flow_id,
        explanation_type=ExplanationType.GENERATION_LOGIC,
        component="CrossModalConsistentGenerator",
        operation="generate_multimodal",
        input_data={"intent": "分析图片", "modality": "multimodal"},
        output_data={"text_output": "图片分析结果：...", "image_output": "annotated_image"},
        reasoning="基于融合意图生成一致的多模态输出",
        confidence=0.90
    )
    
    # 模拟输出优化步骤
    explainer.record_step(
        flow_id=flow_id,
        explanation_type=ExplanationType.OUTPUT_OPTIMIZATION,
        component="AdaptiveOutputOptimizer",
        operation="optimize_output",
        input_data={"text_output": "原始文本", "image_output": "原始图像"},
        output_data={"optimized_text": "优化后文本", "optimized_image": "优化后图像"},
        reasoning="根据用户偏好和环境优化输出格式和质量",
        confidence=0.85
    )
    
    # 结束流程
    flow = explainer.end_flow(flow_id)
    
    print(f"结束流程，步骤数: {len(flow.steps)}")
    
    # 生成解释
    explanation = explainer.generate_explanation(flow_id, level=ExplanationLevel.DETAILED)
    print("\n生成的解释:")
    print(explanation[:500] + "..." if len(explanation) > 500 else explanation)
    
    # 获取流程详情
    flow_details = explainer.get_flow(flow_id, include_steps=False)
    print(f"\n流程摘要:")
    for key, value in flow_details.items():
        if key != "steps":
            print(f"  {key}: {value}")
    
    # 分析流程
    analysis = explainer.analyze_flow(flow_id)
    print(f"\n流程分析 - 基础统计:")
    for key, value in analysis["analysis"]["basic_statistics"].items():
        print(f"  {key}: {value}")
    
    print(f"\n流程分析 - 置信度分析:")
    for key, value in analysis["analysis"]["confidence_analysis"].items():
        print(f"  {key}: {value}")
    
    # 获取统计信息
    stats = explainer.get_statistics()
    print(f"\n系统统计:")
    for key, value in stats.items():
        if key != "explanation_types":
            print(f"  {key}: {value}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_end_to_end_explainability()