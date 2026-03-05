"""
系统可视化器 - 为AGI系统提供可解释性和可视化功能
System Visualizer - Provides interpretability and visualization capabilities for AGI systems
"""

import os
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging

# 在导入matplotlib之前设置后端以避免Qt兼容性问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib不可用，某些可视化功能将受限")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.subplots as sp
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("警告: plotly不可用，交互式可视化功能将受限")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

logger = logging.getLogger(__name__)


class SystemVisualizer:
    """系统可视化器 - 为AGI系统提供可解释性和可视化功能"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化可视化器"""
        self.config = config or {}
        self.output_dir = self.config.get("output_dir", "visualizations")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 可视化选项
        self.use_plotly = self.config.get("use_plotly", True) and PLOTLY_AVAILABLE
        self.use_matplotlib = self.config.get("use_matplotlib", True) and MATPLOTLIB_AVAILABLE
        self.use_seaborn = self.config.get("use_seaborn", True) and SEABORN_AVAILABLE
        
        # 样式配置
        self.colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "warning": "#d62728",
            "info": "#9467bd",
            "light": "#7f7f7f",
            "dark": "#17becf"
        }
        
        logger.info(f"系统可视化器初始化完成，输出目录: {self.output_dir}")
    
    def visualize_architecture_evolution(
        self, 
        evolution_history: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """可视化架构演化过程
        
        Args:
            evolution_history: 演化历史记录，包含每代的架构和适应度得分
            output_path: 输出文件路径，如果为None则自动生成
            
        Returns:
            包含可视化信息的字典
        """
        if not evolution_history:
            return {"error": "没有演化历史数据"}
        
        # 提取数据
        generations = list(range(len(evolution_history)))
        fitness_scores = [gen.get("best_fitness", 0.0) for gen in evolution_history]
        architecture_sizes = []
        attention_types = []
        fusion_strategies = []
        
        for gen in evolution_history:
            best_arch = gen.get("best_architecture", {})
            if isinstance(best_arch, dict):
                architecture_sizes.append(best_arch.get("num_layers", 0))
                attention_types.append(best_arch.get("attention_type", "unknown"))
                fusion_strategies.append(best_arch.get("fusion_strategy", "unknown"))
            else:
                # 如果是对象
                architecture_sizes.append(getattr(best_arch, "num_layers", 0))
                attention_types.append(getattr(best_arch, "attention_type", "unknown"))
                fusion_strategies.append(getattr(best_arch, "fusion_strategy", "unknown"))
        
        # 生成输出路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"architecture_evolution_{timestamp}")
        
        visualizations = {}
        
        # 1. 适应度得分演化图
        if self.use_plotly and PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=generations,
                y=fitness_scores,
                mode='lines+markers',
                name='适应度得分',
                line=dict(color=self.colors["primary"], width=2),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="架构演化 - 适应度得分变化",
                xaxis_title="代数",
                yaxis_title="适应度得分",
                template="plotly_white",
                height=500
            )
            
            plotly_path = f"{output_path}_fitness.html"
            fig.write_html(plotly_path)
            visualizations["fitness_plotly"] = plotly_path
        
        # 2. 架构大小变化图
        if self.use_matplotlib and MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(generations, architecture_sizes, marker='o', color=self.colors["secondary"], linewidth=2)
            ax.set_xlabel("代数")
            ax.set_ylabel("架构层数")
            ax.set_title("架构演化 - 层数变化")
            ax.grid(True, alpha=0.3)
            
            matplotlib_path = f"{output_path}_architecture_size.png"
            fig.savefig(matplotlib_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            visualizations["architecture_size_matplotlib"] = matplotlib_path
        
        # 3. 注意力机制和融合策略分布
        if self.use_plotly and PLOTLY_AVAILABLE and attention_types and fusion_strategies:
            # 注意力类型分布
            attention_counts = {}
            for at in attention_types:
                if isinstance(at, str):
                    attention_counts[at] = attention_counts.get(at, 0) + 1
            
            # 融合策略分布
            fusion_counts = {}
            for fs in fusion_strategies:
                if isinstance(fs, str):
                    fusion_counts[fs] = fusion_counts.get(fs, 0) + 1
            
            # 创建子图
            fig = sp.make_subplots(
                rows=1, cols=2,
                subplot_titles=("注意力机制分布", "融合策略分布"),
                specs=[[{"type": "pie"}, {"type": "pie"}]]
            )
            
            if attention_counts:
                fig.add_trace(
                    go.Pie(
                        labels=list(attention_counts.keys()),
                        values=list(attention_counts.values()),
                        name="注意力机制"
                    ),
                    row=1, col=1
                )
            
            if fusion_counts:
                fig.add_trace(
                    go.Pie(
                        labels=list(fusion_counts.keys()),
                        values=list(fusion_counts.values()),
                        name="融合策略"
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(
                title="架构组件分布",
                height=500,
                showlegend=True
            )
            
            components_path = f"{output_path}_components.html"
            fig.write_html(components_path)
            visualizations["components_plotly"] = components_path
        
        # 4. 创建综合报告
        report = {
            "total_generations": len(evolution_history),
            "best_fitness": max(fitness_scores) if fitness_scores else 0.0,
            "best_fitness_generation": np.argmax(fitness_scores) if fitness_scores else 0,
            "average_architecture_size": np.mean(architecture_sizes) if architecture_sizes else 0,
            "visualizations": visualizations,
            "generated_at": datetime.now().isoformat()
        }
        
        return report
    
    def visualize_multimodal_semantic_space(
        self,
        embeddings: Dict[str, np.ndarray],
        similarity_matrix: Optional[np.ndarray] = None,
        modality_labels: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """可视化多模态语义空间
        
        Args:
            embeddings: 嵌入字典，键为模态标识，值为嵌入向量
            similarity_matrix: 相似度矩阵（可选）
            modality_labels: 模态标签列表（可选）
            output_path: 输出文件路径
            
        Returns:
            包含可视化信息的字典
        """
        if not embeddings:
            return {"error": "没有嵌入数据"}
        
        # 生成输出路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"multimodal_semantic_space_{timestamp}")
        
        visualizations = {}
        modalities = list(embeddings.keys())
        
        # 1. 嵌入可视化（使用PCA降维到2D）
        try:
            from sklearn.decomposition import PCA
            
            # 准备数据 - 正确处理不同形状的嵌入
            all_embeddings_list = []
            modality_indices = []
            
            for i, (modality, emb) in enumerate(embeddings.items()):
                if isinstance(emb, np.ndarray):
                    # 确保emb是2D数组 (n_samples, embedding_dim)
                    if emb.ndim == 1:
                        emb = emb.reshape(1, -1)
                    elif emb.ndim == 2:
                        # 已经是2D，直接使用
                        pass
                    else:
                        # 高维数组，展平或跳过
                        logger.warning(f"模态 {modality} 的嵌入维度 {emb.ndim} 不支持，跳过")
                        continue
                    
                    # 将嵌入添加到列表
                    n_samples = emb.shape[0]
                    if n_samples > 0:
                        all_embeddings_list.append(emb)
                        modality_indices.extend([i] * n_samples)
                elif isinstance(emb, list):
                    # 处理列表类型
                    for item in emb:
                        if isinstance(item, np.ndarray):
                            if item.ndim == 1:
                                item = item.reshape(1, -1)
                                all_embeddings_list.append(item)
                                modality_indices.append(i)
                            elif item.ndim == 2:
                                n_samples = item.shape[0]
                                all_embeddings_list.append(item)
                                modality_indices.extend([i] * n_samples)
                        else:
                            # 尝试转换为numpy数组
                            try:
                                item_arr = np.array(item)
                                if item_arr.ndim == 1:
                                    item_arr = item_arr.reshape(1, -1)
                                    all_embeddings_list.append(item_arr)
                                    modality_indices.append(i)
                            except:
                                logger.warning(f"无法处理模态 {modality} 的嵌入项: {type(item)}")
            
            if not all_embeddings_list:
                return {"error": "没有有效的嵌入数据"}
            
            # 连接所有嵌入
            X = np.vstack(all_embeddings_list)
            
            # 检查数据形状
            if X.shape[0] < 2:
                logger.warning(f"样本数太少 ({X.shape[0]})，无法进行PCA")
                return {"error": "样本数不足，无法进行PCA"}
            
            # 应用PCA
            pca = PCA(n_components=min(2, X.shape[0], X.shape[1]))
            X_2d = pca.fit_transform(X)
            
            # 解释方差比
            explained_variance = pca.explained_variance_ratio_
            
            if self.use_plotly and PLOTLY_AVAILABLE:
                # 创建散点图
                fig = go.Figure()
                
                # 为每个模态添加轨迹
                for i, modality in enumerate(modalities):
                    # 获取该模态的数据点索引
                    indices = [idx for idx, mod_idx in enumerate(modality_indices) if mod_idx == i]
                    if not indices:
                        continue
                    
                    x_vals = X_2d[indices, 0]
                    y_vals = X_2d[indices, 1]
                    
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='markers',
                        name=modality,
                        marker=dict(size=8, opacity=0.7)
                    ))
                
                fig.update_layout(
                    title=f"多模态语义空间 (PCA: {explained_variance[0]:.1%} + {explained_variance[1]:.1%})",
                    xaxis_title="主成分 1",
                    yaxis_title="主成分 2",
                    template="plotly_white",
                    height=600
                )
                
                pca_path = f"{output_path}_pca.html"
                fig.write_html(pca_path)
                visualizations["pca_plotly"] = pca_path
            
            if self.use_matplotlib and MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # 为每个模态绘制散点
                for i, modality in enumerate(modalities):
                    indices = [idx for idx, mod_idx in enumerate(modality_indices) if mod_idx == i]
                    if not indices:
                        continue
                    
                    x_vals = X_2d[indices, 0]
                    y_vals = X_2d[indices, 1]
                    
                    ax.scatter(x_vals, y_vals, label=modality, alpha=0.7, s=50)
                
                ax.set_xlabel(f"主成分 1 ({explained_variance[0]:.1%})")
                ax.set_ylabel(f"主成分 2 ({explained_variance[1]:.1%})")
                ax.set_title("多模态语义空间 - PCA可视化")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                matplotlib_path = f"{output_path}_pca.png"
                fig.savefig(matplotlib_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                visualizations["pca_matplotlib"] = matplotlib_path
                
        except ImportError:
            logger.warning("sklearn不可用，跳过PCA可视化")
        except Exception as e:
            logger.error(f"PCA可视化失败: {e}")
        
        # 2. 相似度矩阵热图
        if similarity_matrix is not None and isinstance(similarity_matrix, np.ndarray):
            if self.use_plotly and PLOTLY_AVAILABLE:
                # 创建热图
                fig = go.Figure(data=go.Heatmap(
                    z=similarity_matrix,
                    x=modality_labels if modality_labels else list(range(similarity_matrix.shape[1])),
                    y=modality_labels if modality_labels else list(range(similarity_matrix.shape[0])),
                    colorscale='Viridis',
                    showscale=True
                ))
                
                fig.update_layout(
                    title="跨模态相似度矩阵",
                    xaxis_title="模态",
                    yaxis_title="模态",
                    height=600
                )
                
                heatmap_path = f"{output_path}_similarity.html"
                fig.write_html(heatmap_path)
                visualizations["similarity_plotly"] = heatmap_path
            
            if self.use_seaborn and SEABORN_AVAILABLE and self.use_matplotlib and MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # 使用seaborn热图
                sns.heatmap(
                    similarity_matrix,
                    ax=ax,
                    cmap='viridis',
                    annot=True,
                    fmt='.2f',
                    xticklabels=modality_labels if modality_labels else list(range(similarity_matrix.shape[1])),
                    yticklabels=modality_labels if modality_labels else list(range(similarity_matrix.shape[0])),
                    cbar_kws={'label': '相似度'}
                )
                
                ax.set_title("跨模态相似度矩阵")
                ax.set_xlabel("模态")
                ax.set_ylabel("模态")
                
                seaborn_path = f"{output_path}_similarity_seaborn.png"
                fig.savefig(seaborn_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                visualizations["similarity_seaborn"] = seaborn_path
        
        # 3. 创建综合报告
        report = {
            "modalities": modalities,
            "total_embeddings": sum(len(emb) if isinstance(emb, list) else 1 for emb in embeddings.values()),
            "embedding_dimensions": embeddings[modalities[0]].shape[1] if isinstance(embeddings[modalities[0]], np.ndarray) else "未知",
            "visualizations": visualizations,
            "generated_at": datetime.now().isoformat()
        }
        
        return report
    
    def visualize_multi_path_reasoning(
        self,
        reasoning_paths: List[Dict[str, Any]],
        voting_results: Optional[Dict[str, Any]] = None,
        confidence_scores: Optional[List[float]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """可视化多路径推理过程
        
        Args:
            reasoning_paths: 推理路径列表
            voting_results: 投票结果（可选）
            confidence_scores: 置信度分数列表（可选）
            output_path: 输出文件路径
            
        Returns:
            包含可视化信息的字典
        """
        if not reasoning_paths:
            return {"error": "没有推理路径数据"}
        
        # 生成输出路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"multi_path_reasoning_{timestamp}")
        
        visualizations = {}
        
        # 1. 推理路径图
        if self.use_plotly and PLOTLY_AVAILABLE:
            # 创建网络图
            fig = go.Figure()
            
            # 提取节点和边
            nodes = []
            edges = []
            node_sizes = []
            node_colors = []
            
            for i, path in enumerate(reasoning_paths):
                path_id = path.get("path_id", f"path_{i}")
                steps = path.get("steps", [])
                
                # 添加节点
                for j, step in enumerate(steps):
                    node_id = f"{path_id}_step_{j}"
                    nodes.append(node_id)
                    
                    # 节点大小基于置信度
                    confidence = step.get("confidence", 0.5)
                    node_sizes.append(10 + confidence * 20)
                    
                    # 节点颜色基于路径
                    node_colors.append(i)
                
                # 添加边
                for j in range(len(steps) - 1):
                    source = f"{path_id}_step_{j}"
                    target = f"{path_id}_step_{j + 1}"
                    edges.append((source, target))
            
            if nodes:
                # 创建节点位置（简单布局）
                import networkx as nx
                G = nx.Graph()
                G.add_nodes_from(nodes)
                G.add_edges_from(edges)
                
                # 使用spring布局
                pos = nx.spring_layout(G, seed=42)
                
                # 提取位置
                x_nodes = [pos[node][0] for node in nodes]
                y_nodes = [pos[node][1] for node in nodes]
                
                # 添加节点轨迹
                fig.add_trace(go.Scatter(
                    x=x_nodes,
                    y=y_nodes,
                    mode='markers',
                    marker=dict(
                        size=node_sizes,
                        color=node_colors,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="路径索引")
                    ),
                    text=nodes,
                    hoverinfo='text',
                    name="推理节点"
                ))
                
                # 添加边
                for edge in edges:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    
                    fig.add_trace(go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=1, color='gray'),
                        hoverinfo='none',
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title="多路径推理网络",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    showlegend=True,
                    height=600
                )
                
                network_path = f"{output_path}_network.html"
                fig.write_html(network_path)
                visualizations["network_plotly"] = network_path
        
        # 2. 置信度比较图
        if confidence_scores or (reasoning_paths and all('confidence' in path for path in reasoning_paths)):
            # 提取置信度分数
            if confidence_scores:
                confidences = confidence_scores
                path_labels = [f"路径 {i+1}" for i in range(len(confidence_scores))]
            else:
                confidences = [path.get("confidence", 0.0) for path in reasoning_paths]
                path_labels = [path.get("path_id", f"路径 {i+1}") for i, path in enumerate(reasoning_paths)]
            
            if self.use_plotly and PLOTLY_AVAILABLE:
                fig = go.Figure(data=go.Bar(
                    x=path_labels,
                    y=confidences,
                    marker_color=[self.colors["primary"] if i == np.argmax(confidences) else self.colors["light"] 
                                 for i in range(len(confidences))]
                ))
                
                fig.update_layout(
                    title="推理路径置信度比较",
                    xaxis_title="推理路径",
                    yaxis_title="置信度",
                    template="plotly_white",
                    height=400
                )
                
                confidence_path = f"{output_path}_confidence.html"
                fig.write_html(confidence_path)
                visualizations["confidence_plotly"] = confidence_path
            
            if self.use_matplotlib and MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                bars = ax.bar(range(len(confidences)), confidences, 
                             color=[self.colors["primary"] if i == np.argmax(confidences) else self.colors["light"] 
                                   for i in range(len(confidences))])
                
                ax.set_xlabel("推理路径")
                ax.set_ylabel("置信度")
                ax.set_title("推理路径置信度比较")
                ax.set_xticks(range(len(confidences)))
                ax.set_xticklabels(path_labels, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom')
                
                matplotlib_path = f"{output_path}_confidence.png"
                fig.savefig(matplotlib_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                visualizations["confidence_matplotlib"] = matplotlib_path
        
        # 3. 投票结果可视化
        if voting_results:
            if self.use_plotly and PLOTLY_AVAILABLE:
                # 提取投票数据
                voting_method = voting_results.get("voting_method", "未知")
                selected_path = voting_results.get("selected_path", "未知")
                voting_details = voting_results.get("voting_details", {})
                
                # 创建投票结果图
                fig = go.Figure()
                
                if "path_votes" in voting_details:
                    path_votes = voting_details["path_votes"]
                    paths = list(path_votes.keys())
                    votes = list(path_votes.values())
                    
                    fig.add_trace(go.Bar(
                        x=paths,
                        y=votes,
                        name="投票数",
                        marker_color=self.colors["secondary"]
                    ))
                    
                    # 标记选中的路径
                    if selected_path in paths:
                        selected_idx = paths.index(selected_path)
                        fig.add_trace(go.Scatter(
                            x=[paths[selected_idx]],
                            y=[votes[selected_idx]],
                            mode='markers',
                            marker=dict(
                                size=15,
                                color=self.colors["success"],
                                symbol='star'
                            ),
                            name="选中路径"
                        ))
                
                fig.update_layout(
                    title=f"投票结果 ({voting_method})",
                    xaxis_title="推理路径",
                    yaxis_title="投票数",
                    template="plotly_white",
                    height=400
                )
                
                voting_path = f"{output_path}_voting.html"
                fig.write_html(voting_path)
                visualizations["voting_plotly"] = voting_path
        
        # 4. 创建综合报告
        report = {
            "total_paths": len(reasoning_paths),
            "max_confidence": max(confidences) if confidence_scores or reasoning_paths else 0.0,
            "average_confidence": np.mean(confidences) if confidence_scores or reasoning_paths else 0.0,
            "visualizations": visualizations,
            "generated_at": datetime.now().isoformat()
        }
        
        if voting_results:
            report["voting_method"] = voting_results.get("voting_method", "未知")
            report["selected_path"] = voting_results.get("selected_path", "未知")
        
        return report
    
    def visualize_resource_usage(
        self,
        performance_metrics: List[Dict[str, Any]],
        memory_snapshots: List[Dict[str, Any]],
        optimization_suggestions: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """可视化资源使用情况
        
        Args:
            performance_metrics: 性能指标列表
            memory_snapshots: 内存快照列表
            optimization_suggestions: 优化建议列表（可选）
            output_path: 输出文件路径
            
        Returns:
            包含可视化信息的字典
        """
        if not performance_metrics and not memory_snapshots:
            return {"error": "没有资源使用数据"}
        
        # 生成输出路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"resource_usage_{timestamp}")
        
        visualizations = {}
        
        # 1. CPU和内存使用时间序列
        if performance_metrics:
            # 提取时间序列数据
            timestamps = []
            cpu_percent = []
            memory_mb = []
            
            for metric in performance_metrics:
                if isinstance(metric, dict):
                    timestamps.append(metric.get("timestamp", 0))
                    cpu_percent.append(metric.get("cpu_percent", 0.0))
                    memory_mb.append(metric.get("memory_mb", 0.0))
                else:
                    # 如果是对象
                    timestamps.append(getattr(metric, "timestamp", 0))
                    cpu_percent.append(getattr(metric, "cpu_percent", 0.0))
                    memory_mb.append(getattr(metric, "memory_mb", 0.0))
            
            if timestamps and cpu_percent and memory_mb:
                if self.use_plotly and PLOTLY_AVAILABLE:
                    # 创建子图
                    fig = sp.make_subplots(
                        rows=2, cols=1,
                        subplot_titles=("CPU使用率", "内存使用"),
                        vertical_spacing=0.15
                    )
                    
                    # CPU使用率
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=cpu_percent,
                            mode='lines',
                            name='CPU使用率',
                            line=dict(color=self.colors["primary"], width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # 内存使用
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=memory_mb,
                            mode='lines',
                            name='内存使用 (MB)',
                            line=dict(color=self.colors["secondary"], width=2)
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        title="资源使用时间序列",
                        height=600,
                        showlegend=True
                    )
                    
                    fig.update_xaxes(title_text="时间", row=2, col=1)
                    fig.update_yaxes(title_text="CPU (%)", row=1, col=1)
                    fig.update_yaxes(title_text="内存 (MB)", row=2, col=1)
                    
                    timeseries_path = f"{output_path}_timeseries.html"
                    fig.write_html(timeseries_path)
                    visualizations["timeseries_plotly"] = timeseries_path
                
                if self.use_matplotlib and MATPLOTLIB_AVAILABLE:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                    
                    # CPU使用率
                    ax1.plot(timestamps, cpu_percent, color=self.colors["primary"], linewidth=2)
                    ax1.set_ylabel("CPU使用率 (%)")
                    ax1.set_title("CPU使用率时间序列")
                    ax1.grid(True, alpha=0.3)
                    
                    # 内存使用
                    ax2.plot(timestamps, memory_mb, color=self.colors["secondary"], linewidth=2)
                    ax2.set_xlabel("时间")
                    ax2.set_ylabel("内存使用 (MB)")
                    ax2.set_title("内存使用时间序列")
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    timeseries_matplotlib_path = f"{output_path}_timeseries.png"
                    fig.savefig(timeseries_matplotlib_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    visualizations["timeseries_matplotlib"] = timeseries_matplotlib_path
        
        # 2. 内存使用详情
        if memory_snapshots:
            # 提取内存数据
            timestamps = []
            process_memory_mb = []
            system_memory_percent = []
            
            for snapshot in memory_snapshots:
                if isinstance(snapshot, dict):
                    timestamps.append(snapshot.get("timestamp", 0))
                    process_memory_mb.append(snapshot.get("process_memory_mb", 0.0))
                    system_memory_percent.append(snapshot.get("system_memory_percent", 0.0))
                else:
                    # 如果是对象
                    timestamps.append(getattr(snapshot, "timestamp", 0))
                    process_memory_mb.append(getattr(snapshot, "process_memory_mb", 0.0))
                    system_memory_percent.append(getattr(snapshot, "system_memory_percent", 0.0))
            
            if timestamps and process_memory_mb and system_memory_percent:
                if self.use_plotly and PLOTLY_AVAILABLE:
                    fig = sp.make_subplots(
                        rows=2, cols=1,
                        subplot_titles=("进程内存使用", "系统内存使用率"),
                        vertical_spacing=0.15
                    )
                    
                    # 进程内存
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=process_memory_mb,
                            mode='lines',
                            name='进程内存 (MB)',
                            line=dict(color=self.colors["warning"], width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # 系统内存
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=system_memory_percent,
                            mode='lines',
                            name='系统内存 (%)',
                            line=dict(color=self.colors["info"], width=2)
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        title="内存使用详情",
                        height=600,
                        showlegend=True
                    )
                    
                    fig.update_xaxes(title_text="时间", row=2, col=1)
                    fig.update_yaxes(title_text="进程内存 (MB)", row=1, col=1)
                    fig.update_yaxes(title_text="系统内存 (%)", row=2, col=1)
                    
                    memory_detail_path = f"{output_path}_memory_detail.html"
                    fig.write_html(memory_detail_path)
                    visualizations["memory_detail_plotly"] = memory_detail_path
        
        # 3. 优化建议可视化
        if optimization_suggestions:
            if self.use_plotly and PLOTLY_AVAILABLE:
                # 按优先级分组
                suggestions_by_priority = {"high": [], "medium": [], "low": []}
                for suggestion in optimization_suggestions:
                    if isinstance(suggestion, dict):
                        priority = suggestion.get("priority", "low").lower()
                        suggestions_by_priority[priority].append(suggestion)
                
                # 创建条形图
                priorities = ["high", "medium", "low"]
                counts = [len(suggestions_by_priority[p]) for p in priorities]
                
                fig = go.Figure(data=go.Bar(
                    x=priorities,
                    y=counts,
                    marker_color=[self.colors["warning"], self.colors["info"], self.colors["light"]]
                ))
                
                fig.update_layout(
                    title="优化建议分布",
                    xaxis_title="优先级",
                    yaxis_title="建议数量",
                    template="plotly_white",
                    height=400
                )
                
                suggestions_path = f"{output_path}_suggestions.html"
                fig.write_html(suggestions_path)
                visualizations["suggestions_plotly"] = suggestions_path
        
        # 4. 创建综合报告
        report = {
            "performance_metrics_count": len(performance_metrics),
            "memory_snapshots_count": len(memory_snapshots),
            "optimization_suggestions_count": len(optimization_suggestions) if optimization_suggestions else 0,
            "visualizations": visualizations,
            "generated_at": datetime.now().isoformat()
        }
        
        if performance_metrics:
            cpu_values = [m.get("cpu_percent", 0.0) if isinstance(m, dict) else getattr(m, "cpu_percent", 0.0) 
                         for m in performance_metrics]
            memory_values = [m.get("memory_mb", 0.0) if isinstance(m, dict) else getattr(m, "memory_mb", 0.0) 
                           for m in performance_metrics]
            
            report["cpu_stats"] = {
                "avg": np.mean(cpu_values) if cpu_values else 0.0,
                "max": np.max(cpu_values) if cpu_values else 0.0,
                "min": np.min(cpu_values) if cpu_values else 0.0
            }
            
            report["memory_stats"] = {
                "avg": np.mean(memory_values) if memory_values else 0.0,
                "max": np.max(memory_values) if memory_values else 0.0,
                "min": np.min(memory_values) if memory_values else 0.0
            }
        
        return report
    
    def create_system_dashboard(
        self,
        architecture_data: Optional[Dict[str, Any]] = None,
        multimodal_data: Optional[Dict[str, Any]] = None,
        reasoning_data: Optional[Dict[str, Any]] = None,
        resource_data: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建系统综合仪表板
        
        Args:
            architecture_data: 架构演化数据
            multimodal_data: 多模态数据
            reasoning_data: 推理数据
            resource_data: 资源数据
            output_path: 输出文件路径
            
        Returns:
            包含仪表板信息的字典
        """
        # 生成输出路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"system_dashboard_{timestamp}")
        
        dashboard_data = {
            "generated_at": datetime.now().isoformat(),
            "components": {}
        }
        
        # 生成各组件可视化
        if architecture_data:
            # 提取演化历史数据
            evolution_history = architecture_data.get("evolution_history", [])
            if evolution_history:
                arch_viz = self.visualize_architecture_evolution(evolution_history, f"{output_path}_architecture")
                dashboard_data["components"]["architecture"] = arch_viz
        
        if multimodal_data:
            multimodal_viz = self.visualize_multimodal_semantic_space(**multimodal_data, output_path=f"{output_path}_multimodal")
            dashboard_data["components"]["multimodal"] = multimodal_viz
        
        if reasoning_data:
            reasoning_viz = self.visualize_multi_path_reasoning(**reasoning_data, output_path=f"{output_path}_reasoning")
            dashboard_data["components"]["reasoning"] = reasoning_viz
        
        if resource_data:
            resource_viz = self.visualize_resource_usage(**resource_data, output_path=f"{output_path}_resource")
            dashboard_data["components"]["resource"] = resource_viz
        
        # 创建HTML仪表板
        if self.use_plotly and PLOTLY_AVAILABLE:
            dashboard_html = self._generate_dashboard_html(dashboard_data)
            
            dashboard_file = f"{output_path}.html"
            with open(dashboard_file, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)
            
            dashboard_data["dashboard_html"] = dashboard_file
        
        # 创建JSON报告之前，确保所有数据都是JSON可序列化的
        serializable_dashboard_data = self._convert_to_serializable(dashboard_data)
        
        json_file = f"{output_path}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_dashboard_data, f, indent=2, ensure_ascii=False)
        
        dashboard_data["dashboard_json"] = json_file
        
        return dashboard_data
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """将对象转换为JSON可序列化的格式
        
        处理numpy数组、标量和其他非标准类型
        """
        if obj is None:
            return None
        
        # 处理numpy标量
        if hasattr(obj, 'item'):
            try:
                return obj.item()  # 将numpy标量转换为Python标量
            except:
                pass
        
        # 处理numpy数组
        if hasattr(obj, 'tolist'):
            try:
                return obj.tolist()  # 将numpy数组转换为Python列表
            except:
                pass
        
        # 处理字典
        if isinstance(obj, dict):
            return {str(k): self._convert_to_serializable(v) for k, v in obj.items()}
        
        # 处理列表、元组等可迭代对象
        if isinstance(obj, (list, tuple, set)):
            return [self._convert_to_serializable(item) for item in obj]
        
        # 处理基本类型
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        # 处理datetime对象
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # 尝试转换为字符串
        try:
            return str(obj)
        except:
            # 如果无法转换，返回None或空字符串
            return None
    
    def _generate_dashboard_html(self, dashboard_data: Dict[str, Any]) -> str:
        """生成HTML仪表板"""
        html_template = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AGI系统综合仪表板</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background-color: #f5f5f5;
                    padding: 20px;
                }
                
                .container {
                    max-width: 1400px;
                    margin: 0 auto;
                }
                
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                
                .header h1 {
                    font-size: 2.5rem;
                    margin-bottom: 10px;
                }
                
                .header p {
                    font-size: 1.1rem;
                    opacity: 0.9;
                }
                
                .dashboard-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                
                .dashboard-card {
                    background: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                    transition: transform 0.3s ease;
                }
                
                .dashboard-card:hover {
                    transform: translateY(-5px);
                }
                
                .card-header {
                    display: flex;
                    align-items: center;
                    margin-bottom: 20px;
                    padding-bottom: 15px;
                    border-bottom: 2px solid #f0f0f0;
                }
                
                .card-icon {
                    font-size: 1.5rem;
                    margin-right: 10px;
                }
                
                .card-title {
                    font-size: 1.3rem;
                    font-weight: 600;
                    color: #333;
                }
                
                .card-content {
                    margin-top: 15px;
                }
                
                .plot-container {
                    width: 100%;
                    height: 400px;
                }
                
                .status-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                    margin-top: 20px;
                }
                
                .status-item {
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                }
                
                .status-label {
                    font-size: 0.9rem;
                    color: #666;
                    margin-bottom: 5px;
                }
                
                .status-value {
                    font-size: 1.2rem;
                    font-weight: 600;
                    color: #333;
                }
                
                .suggestions-list {
                    list-style-type: none;
                }
                
                .suggestion-item {
                    padding: 10px;
                    margin-bottom: 10px;
                    background: #fff3cd;
                    border-left: 4px solid #ffc107;
                    border-radius: 4px;
                }
                
                .suggestion-high {
                    background: #f8d7da;
                    border-left-color: #dc3545;
                }
                
                .suggestion-medium {
                    background: #fff3cd;
                    border-left-color: #ffc107;
                }
                
                .suggestion-low {
                    background: #d1ecf1;
                    border-left-color: #17a2b8;
                }
                
                .footer {
                    text-align: center;
                    margin-top: 40px;
                    padding: 20px;
                    color: #666;
                    font-size: 0.9rem;
                    border-top: 1px solid #eee;
                }
                
                @media (max-width: 768px) {
                    .dashboard-grid {
                        grid-template-columns: 1fr;
                    }
                    
                    .header h1 {
                        font-size: 2rem;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 AGI系统综合仪表板</h1>
                    <p>系统可视化与可解释性报告 | 生成时间: {generated_at}</p>
                </div>
                
                <div class="dashboard-grid">
                    {dashboard_cards}
                </div>
                
                <div class="footer">
                    <p>© 2025 AGI Soul System - 系统可视化与可解释性报告</p>
                    <p>本报告由系统可视化器自动生成 | 版本: 1.0.0</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # 生成卡片内容
        dashboard_cards = ""
        
        # 架构演化卡片
        if "architecture" in dashboard_data["components"]:
            arch_data = dashboard_data["components"]["architecture"]
            if "visualizations" in arch_data and "fitness_plotly" in arch_data["visualizations"]:
                arch_card = """
                <div class="dashboard-card">
                    <div class="card-header">
                        <div class="card-icon">🏗️</div>
                        <div class="card-title">架构演化分析</div>
                    </div>
                    <div class="card-content">
                        <div class="plot-container" id="architecture-plot"></div>
                        <div class="status-grid">
                            <div class="status-item">
                                <div class="status-label">总代数</div>
                                <div class="status-value">{total_generations}</div>
                            </div>
                            <div class="status-item">
                                <div class="status-label">最佳适应度</div>
                                <div class="status-value">{best_fitness:.3f}</div>
                            </div>
                            <div class="status-item">
                                <div class="status-label">平均架构大小</div>
                                <div class="status-value">{avg_size:.1f} 层</div>
                            </div>
                        </div>
                    </div>
                </div>
                """.format(
                    total_generations=arch_data.get("total_generations", 0),
                    best_fitness=arch_data.get("best_fitness", 0.0),
                    avg_size=arch_data.get("average_architecture_size", 0.0)
                )
                dashboard_cards += arch_card
        
        # 多模态语义空间卡片
        if "multimodal" in dashboard_data["components"]:
            multimodal_data = dashboard_data["components"]["multimodal"]
            if "visualizations" in multimodal_data and "pca_plotly" in multimodal_data["visualizations"]:
                multimodal_card = """
                <div class="dashboard-card">
                    <div class="card-header">
                        <div class="card-icon">🎨</div>
                        <div class="card-title">多模态语义空间</div>
                    </div>
                    <div class="card-content">
                        <div class="plot-container" id="multimodal-plot"></div>
                        <div class="status-grid">
                            <div class="status-item">
                                <div class="status-label">模态数量</div>
                                <div class="status-value">{modality_count}</div>
                            </div>
                            <div class="status-item">
                                <div class="status-label">总嵌入数</div>
                                <div class="status-value">{embedding_count}</div>
                            </div>
                            <div class="status-item">
                                <div class="status-label">嵌入维度</div>
                                <div class="status-value">{embedding_dim}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """.format(
                    modality_count=len(multimodal_data.get("modalities", [])),
                    embedding_count=multimodal_data.get("total_embeddings", 0),
                    embedding_dim=multimodal_data.get("embedding_dimensions", "未知")
                )
                dashboard_cards += multimodal_card
        
        # 多路径推理卡片
        if "reasoning" in dashboard_data["components"]:
            reasoning_data = dashboard_data["components"]["reasoning"]
            if "visualizations" in reasoning_data and "confidence_plotly" in reasoning_data["visualizations"]:
                reasoning_card = """
                <div class="dashboard-card">
                    <div class="card-header">
                        <div class="card-icon">🧠</div>
                        <div class="card-title">多路径推理分析</div>
                    </div>
                    <div class="card-content">
                        <div class="plot-container" id="reasoning-plot"></div>
                        <div class="status-grid">
                            <div class="status-item">
                                <div class="status-label">总路径数</div>
                                <div class="status-value">{total_paths}</div>
                            </div>
                            <div class="status-item">
                                <div class="status-label">最高置信度</div>
                                <div class="status-value">{max_confidence:.3f}</div>
                            </div>
                            <div class="status-item">
                                <div class="status-label">平均置信度</div>
                                <div class="status-value">{avg_confidence:.3f}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """.format(
                    total_paths=reasoning_data.get("total_paths", 0),
                    max_confidence=reasoning_data.get("max_confidence", 0.0),
                    avg_confidence=reasoning_data.get("average_confidence", 0.0)
                )
                dashboard_cards += reasoning_card
        
        # 资源使用卡片
        if "resource" in dashboard_data["components"]:
            resource_data = dashboard_data["components"]["resource"]
            if "visualizations" in resource_data and "timeseries_plotly" in resource_data["visualizations"]:
                # 提取优化建议
                suggestions_html = ""
                if resource_data.get("optimization_suggestions_count", 0) > 0:
                    suggestions_html = """
                    <div style="margin-top: 20px;">
                        <h4>优化建议 ({suggestion_count}条)</h4>
                        <ul class="suggestions-list">
                    """.format(suggestion_count=resource_data.get("optimization_suggestions_count", 0))
                    
                    # 这里可以添加具体的建议，但为了简化，只显示计数
                    suggestions_html += """
                            <li class="suggestion-item">查看详细报告获取具体优化建议</li>
                        </ul>
                    </div>
                    """
                
                resource_card = """
                <div class="dashboard-card">
                    <div class="card-header">
                        <div class="card-icon">⚡</div>
                        <div class="card-title">资源使用监控</div>
                    </div>
                    <div class="card-content">
                        <div class="plot-container" id="resource-plot"></div>
                        <div class="status-grid">
                            <div class="status-item">
                                <div class="status-label">CPU平均使用率</div>
                                <div class="status-value">{cpu_avg:.1f}%</div>
                            </div>
                            <div class="status-item">
                                <div class="status-label">内存平均使用</div>
                                <div class="status-value">{memory_avg:.1f}MB</div>
                            </div>
                            <div class="status-item">
                                <div class="status-label">监控样本数</div>
                                <div class="status-value">{sample_count}</div>
                            </div>
                        </div>
                        {suggestions}
                    </div>
                </div>
                """.format(
                    cpu_avg=resource_data.get("cpu_stats", {}).get("avg", 0.0),
                    memory_avg=resource_data.get("memory_stats", {}).get("avg", 0.0),
                    sample_count=resource_data.get("performance_metrics_count", 0),
                    suggestions=suggestions_html
                )
                dashboard_cards += resource_card
        
        # 填充模板 - 使用字符串替换避免CSS花括号被误解
        generated_at = dashboard_data.get("generated_at", "未知时间")
        html_content = html_template.replace("{generated_at}", generated_at).replace("{dashboard_cards}", dashboard_cards)
        
        return html_content


def create_system_visualizer(config: Optional[Dict[str, Any]] = None) -> SystemVisualizer:
    """创建系统可视化器实例"""
    return SystemVisualizer(config)