"""
多模态特征融合

为统一认知架构提供高级多模态特征融合功能。
"""

import torch
import torch.nn as nn
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class MultimodalFusion:
    """多模态特征融合系统"""
    
    def __init__(self, communication):
        """
        初始化多模态融合组件。
        
        参数:
            communication: 神经通信系统
        """
        self.communication = communication
        self.initialized = False
        
        # 融合网络
        self.fusion_network = None
        self.attention_network = None
        self.alignment_network = None
        
        # 融合配置
        self.fusion_config = {
            'fusion_method': 'attention_weighted',  # attention_weighted, concatenation, cross_attention
            'enable_alignment': True,
            'alignment_threshold': 0.8,
            'max_modalities': 5,
            'fusion_dimension': 1024
        }
        
        # 模态权重
        self.modality_weights = {
            'text': 0.35,
            'image': 0.30,
            'audio': 0.20,
            'sensor': 0.15
        }
        
        # 上下文历史
        self.context_history = []
        self.max_history_size = 10
        
        # 融合统计
        self.fusion_stats = {
            'total_fusions': 0,
            'successful_fusions': 0,
            'failed_fusions': 0,
            'avg_fusion_time': 0.0
        }
        
        logger.info("多模态特征融合系统已初始化")
    
    async def initialize(self):
        """初始化融合网络"""
        if self.initialized:
            return
        
        logger.info("初始化多模态融合网络...")
        
        # 初始化融合网络
        self.fusion_network = self._create_fusion_network()
        self.attention_network = self._create_attention_network()
        self.alignment_network = self._create_alignment_network()
        
        self.initialized = True
        logger.info("多模态融合网络初始化完成")
    
    async def fuse_modalities(self, modality_features: Dict[str, torch.Tensor], 
                             context: Dict[str, Any] = None) -> torch.Tensor:
        """
        融合多模态特征。
        
        参数:
            modality_features: 模态特征字典 {模态名: 特征张量}
            context: 融合上下文信息
            
        返回:
            融合后的特征张量
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 验证输入
            if not modality_features:
                raise ValueError("未提供模态特征")
            
            # 对齐模态特征
            aligned_features = await self._align_modalities(modality_features)
            
            # 根据融合方法进行融合
            fusion_method = self.fusion_config['fusion_method']
            
            if fusion_method == 'attention_weighted':
                fused_features = await self._attention_weighted_fusion(aligned_features, context)
            elif fusion_method == 'concatenation':
                fused_features = await self._concatenation_fusion(aligned_features)
            elif fusion_method == 'cross_attention':
                fused_features = await self._cross_attention_fusion(aligned_features, context)
            else:
                # 默认使用注意力加权融合
                fused_features = await self._attention_weighted_fusion(aligned_features, context)
            
            # 更新统计信息
            self._update_fusion_stats(start_time, success=True)
            
            # 更新上下文历史
            self._update_context_history(modality_features.keys(), fused_features)
            
            return fused_features
            
        except Exception as e:
            logger.error(f"多模态融合失败: {e}")
            self._update_fusion_stats(start_time, success=False)
            raise
    
    async def _align_modalities(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """对齐模态特征到统一空间"""
        if not self.fusion_config['enable_alignment']:
            return modality_features
        
        aligned_features = {}
        
        for modality, features in modality_features.items():
            # 简单对齐：确保特征维度一致
            if len(features.shape) == 1:
                # 一维特征，扩展到二维 (1, features)
                aligned = features.unsqueeze(0)
            elif len(features.shape) == 2:
                # 已经是二维特征
                aligned = features
            else:
                # 更高维特征，展平
                aligned = features.view(features.shape[0], -1)
            
            # 如果对齐网络可用，使用对齐网络
            if self.alignment_network is not None:
                aligned = self.alignment_network(aligned)
            
            aligned_features[modality] = aligned
        
        return aligned_features
    
    async def _attention_weighted_fusion(self, aligned_features: Dict[str, torch.Tensor], 
                                        context: Dict[str, Any] = None) -> torch.Tensor:
        """注意力加权融合"""
        # 提取特征列表
        features_list = list(aligned_features.values())
        modality_names = list(aligned_features.keys())
        
        if not features_list:
            raise ValueError("没有可融合的特征")
        
        # 计算注意力权重
        if self.attention_network is not None and context is not None:
            # 使用注意力网络计算权重
            context_tensor = self._create_context_tensor(context)
            attention_weights = self.attention_network(context_tensor)
        else:
            # 使用预定义权重
            attention_weights = []
            for modality in modality_names:
                weight = self.modality_weights.get(modality, 0.1)
                attention_weights.append(weight)
            
            # 归一化权重
            total_weight = sum(attention_weights)
            attention_weights = [w / total_weight for w in attention_weights]
        
        # 加权融合
        fused = torch.zeros_like(features_list[0])
        
        for i, features in enumerate(features_list):
            # 确保特征维度一致
            if features.shape != fused.shape:
                features = self._resize_features(features, fused.shape)
            
            # 应用权重
            weight = attention_weights[i]
            fused += features * weight
        
        return fused
    
    async def _concatenation_fusion(self, aligned_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """拼接融合"""
        features_list = list(aligned_features.values())
        
        if not features_list:
            raise ValueError("没有可融合的特征")
        
        # 确保所有特征维度一致
        resized_features = []
        for features in features_list:
            # 展平特征
            flattened = features.view(features.shape[0], -1)
            resized_features.append(flattened)
        
        # 拼接所有特征
        fused = torch.cat(resized_features, dim=1)
        
        return fused
    
    async def _cross_attention_fusion(self, aligned_features: Dict[str, torch.Tensor], 
                                     context: Dict[str, Any] = None) -> torch.Tensor:
        """交叉注意力融合"""
        # 使用融合网络进行交叉注意力融合
        if self.fusion_network is None:
            # 回退到注意力加权融合
            return await self._attention_weighted_fusion(aligned_features, context)
        
        # 准备输入
        features_list = list(aligned_features.values())
        
        # 将特征列表转换为张量
        features_tensor = torch.stack(features_list, dim=0)
        
        # 应用融合网络
        fused = self.fusion_network(features_tensor)
        
        return fused
    
    def _create_context_tensor(self, context: Dict[str, Any]) -> torch.Tensor:
        """创建上下文张量"""
        # 简单实现：从上下文中提取数值特征
        context_features = []
        
        # 添加时间特征
        if 'timestamp' in context:
            context_features.append(float(context['timestamp']))
        
        # 添加任务类型特征
        if 'task_type' in context:
            # 简单的one-hot编码
            task_types = ['perception', 'reasoning', 'planning', 'action', 'learning']
            if context['task_type'] in task_types:
                idx = task_types.index(context['task_type'])
                one_hot = [0] * len(task_types)
                one_hot[idx] = 1
                context_features.extend(one_hot)
        
        # 添加认知状态特征
        if 'cognitive_state' in context:
            cognitive_state = context['cognitive_state']
            if isinstance(cognitive_state, dict):
                # 提取数值特征
                for key, value in cognitive_state.items():
                    if isinstance(value, (int, float)):
                        context_features.append(float(value))
        
        # 转换为张量
        if not context_features:
            # 默认上下文特征
            context_features = [0.5] * 10
        
        context_tensor = torch.tensor(context_features, dtype=torch.float32)
        return context_tensor.unsqueeze(0)  # 添加批次维度
    
    def _resize_features(self, features: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """调整特征维度以匹配目标形状"""
        if features.shape == target_shape:
            return features
        
        # 简单的调整方法：使用线性插值
        if len(features.shape) == 2 and len(target_shape) == 2:
            # 2D特征
            return nn.functional.interpolate(
                features.unsqueeze(0).unsqueeze(0), 
                size=target_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        else:
            # 展平后调整
            flattened = features.view(-1)
            target_elements = target_shape.numel()
            
            if len(flattened) >= target_elements:
                # 截断
                resized = flattened[:target_elements]
            else:
                # 填充
                padding = torch.zeros(target_elements - len(flattened))
                resized = torch.cat([flattened, padding])
            
            return resized.view(target_shape)
    
    def _update_fusion_stats(self, start_time: float, success: bool):
        """更新融合统计"""
        fusion_time = asyncio.get_event_loop().time() - start_time
        self.fusion_stats['total_fusions'] += 1
        
        if success:
            self.fusion_stats['successful_fusions'] += 1
        else:
            self.fusion_stats['failed_fusions'] += 1
        
        # 更新平均融合时间
        total_successful = self.fusion_stats['successful_fusions']
        if total_successful > 0:
            current_avg = self.fusion_stats['avg_fusion_time']
            self.fusion_stats['avg_fusion_time'] = (
                current_avg * (total_successful - 1) + fusion_time
            ) / total_successful
    
    def _update_context_history(self, modality_names: List[str], fused_features: torch.Tensor):
        """更新上下文历史"""
        history_entry = {
            'modalities': modality_names,
            'timestamp': asyncio.get_event_loop().time(),
            'features_shape': fused_features.shape
        }
        
        self.context_history.append(history_entry)
        
        # 限制历史大小
        if len(self.context_history) > self.max_history_size:
            self.context_history = self.context_history[-self.max_history_size:]
    
    def _create_fusion_network(self) -> nn.Module:
        """创建融合网络"""
        return nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.fusion_config['fusion_dimension']),
            nn.Tanh()
        )
    
    def _create_attention_network(self) -> nn.Module:
        """创建注意力网络"""
        return nn.Sequential(
            nn.Linear(20, 64),  # 上下文特征维度
            nn.ReLU(),
            nn.Linear(64, 10),  # 最大模态数
            nn.Softmax(dim=1)
        )
    
    def _create_alignment_network(self) -> nn.Module:
        """创建对齐网络"""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.fusion_config['fusion_dimension']),
            nn.LayerNorm(self.fusion_config['fusion_dimension'])
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取融合统计信息"""
        return {
            'fusion_stats': self.fusion_stats,
            'modality_weights': self.modality_weights,
            'context_history_size': len(self.context_history),
            'fusion_config': self.fusion_config
        }