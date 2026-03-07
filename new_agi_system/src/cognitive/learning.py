"""
元学习系统

为统一认知架构实现元学习。
"""

import torch
import torch.nn as nn
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class MetaLearningSystem:
    """元学习系统"""
    
    def __init__(self, communication):
        """
        初始化学习组件。
        
        参数:
            communication: 神经通信系统
        """
        self.communication = communication
        self.initialized = False
        
        # 学习网络
        self.meta_learning_network = None
        self.optimization_network = None
        self.experience_integration_network = None
        
        # 配置
        self.config = {
            'learning_rate': 0.001,
            'meta_batch_size': 10,
            'experience_buffer_size': 1000,
            'adaptation_steps': 5
        }
        
        # 经验缓冲区
        self.experience_buffer = []
        
        # 学习统计
        self.learning_stats = {
            'total_experiences': 0,
            'learning_cycles': 0,
            'performance_improvement': 0.0,
            'adaptation_successes': 0,
            'adaptation_failures': 0
        }
        
        # 学习到的模式
        self.learned_patterns = []
        
        logger.info("元学习系统已初始化")
    
    async def initialize(self):
        """初始化学习网络"""
        if self.initialized:
            return
        
        logger.info("正在初始化学习网络...")
        
        # 初始化学习网络
        self.meta_learning_network = self._create_meta_learning_network()
        self.optimization_network = self._create_optimization_network()
        self.experience_integration_network = self._create_experience_integration_network()
        
        # 加载学习到的模式
        await self._load_learned_patterns()
        
        self.initialized = True
        logger.info(f"学习网络初始化完成，包含 {len(self.learned_patterns)} 个模式")
    
    async def process(self, input_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Apply meta-learning.
        
        Args:
            input_tensor: Input tensor (combined experience)
            metadata: Learning metadata
            
        Returns:
            Learning update tensor
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            input_data = metadata.get('input_data', {})
            action_result = metadata.get('action_result', {})
            cognitive_state = metadata.get('cognitive_state', {})
            
            # Store experience
            await self._store_experience(input_data, action_result, cognitive_state)
            
            # Apply meta-learning
            learning_update = await self._apply_meta_learning(
                input_tensor, input_data, action_result
            )
            
            # Extract lessons learned
            lessons_learned = await self._extract_lessons(
                input_data, action_result, learning_update
            )
            
            # Optimize parameters
            parameter_updates = await self._optimize_parameters(
                lessons_learned, cognitive_state
            )
            
            # Calculate performance improvement
            performance_improvement = self._calculate_performance_improvement(
                action_result, lessons_learned
            )
            
            # Update learning statistics
            self.learning_stats['learning_cycles'] += 1
            self.learning_stats['performance_improvement'] = (
                self.learning_stats['performance_improvement'] * 0.9 +
                performance_improvement * 0.1
            )
            
            if performance_improvement > 0:
                self.learning_stats['adaptation_successes'] += 1
            else:
                self.learning_stats['adaptation_failures'] += 1
            
            # Prepare response metadata
            response_metadata = {
                'lessons_learned': lessons_learned,
                'parameter_updates': parameter_updates,
                'performance_improvement': performance_improvement,
                'experience_buffer_size': len(self.experience_buffer),
                'learning_timestamp': time.time()
            }
            
            logger.debug(f"应用元学习，性能提升: {performance_improvement:.3f}")
            
            return learning_update
            
        except Exception as e:
            logger.error(f"元学习失败: {e}")
            # Return zero tensor as fallback
            return torch.zeros_like(input_tensor)
    
    async def _store_experience(self, input_data: Dict[str, Any],
                              action_result: Dict[str, Any],
                              cognitive_state: Dict[str, Any]):
        """Store experience in buffer"""
        experience = {
            'input_data': input_data,
            'action_result': action_result,
            'cognitive_state': cognitive_state,
            'timestamp': time.time(),
            'value': self._calculate_experience_value(input_data, action_result)
        }
        
        self.experience_buffer.append(experience)
        self.learning_stats['total_experiences'] += 1
        
        # Limit buffer size
        if len(self.experience_buffer) > self.config['experience_buffer_size']:
            # Remove lowest value experiences
            self.experience_buffer.sort(key=lambda x: x['value'])
            self.experience_buffer = self.experience_buffer[-self.config['experience_buffer_size']:]
    
    async def _apply_meta_learning(self, input_tensor: torch.Tensor,
                                 input_data: Dict[str, Any],
                                 action_result: Dict[str, Any]) -> torch.Tensor:
        """Apply meta-learning"""
        if self.meta_learning_network:
            try:
                # Extract feedback from action result
                feedback = action_result.get('feedback', {})
                feedback_quality = feedback.get('quality', 0.5)
                execution_time = feedback.get('execution_time', 1.0)
                
                # Create learning context
                context_vector = torch.tensor([
                    feedback_quality,
                    min(1.0, execution_time / 30.0),  # Normalized
                    len(self.experience_buffer) / self.config['experience_buffer_size']
                ]).float()
                
                # Combine with input tensor
                learning_input = torch.cat([input_tensor, context_vector])
                
                # Apply meta-learning
                learning_update = self.meta_learning_network(learning_input.unsqueeze(0))
                return learning_update.squeeze(0)
                
            except Exception as e:
                logger.warning(f"元学习失败: {e}")
                return torch.zeros_like(input_tensor)
        else:
            return torch.zeros_like(input_tensor)
    
    async def _extract_lessons(self, input_data: Dict[str, Any],
                             action_result: Dict[str, Any],
                             learning_update: torch.Tensor) -> List[Dict[str, Any]]:
        """Extract lessons learned"""
        lessons = []
        
        # Extract from action result feedback
        feedback = action_result.get('feedback', {})
        if feedback:
            lessons.append({
                'type': 'feedback_analysis',
                'insight': f"Action quality: {feedback.get('quality', 0.5):.2f}",
                'confidence': min(1.0, feedback.get('quality', 0.5)),
                'applicability': 'general'
            })
        
        # Extract from execution metrics
        execution_time = feedback.get('execution_time', 0.0)
        if execution_time > 0:
            efficiency = feedback.get('efficiency', 0.5)
            lessons.append({
                'type': 'efficiency_analysis',
                'insight': f"Execution efficiency: {efficiency:.2f} (time: {execution_time:.2f}s)",
                'confidence': efficiency,
                'applicability': 'execution'
            })
        
        # Extract from learning update
        if learning_update is not None:
            # Analyze learning update characteristics
            update_magnitude = torch.norm(learning_update).item()
            update_variance = torch.var(learning_update).item() if learning_update.numel() > 1 else 0.0
            
            lessons.append({
                'type': 'learning_pattern',
                'insight': f"Learning update magnitude: {update_magnitude:.3f}, variance: {update_variance:.3f}",
                'confidence': min(1.0, update_magnitude * 10),  # Scale to reasonable range
                'applicability': 'meta_learning'
            })
        
        # Add generic lessons if few lessons extracted
        if len(lessons) < 2:
            lessons.append({
                'type': 'general_learning',
                'insight': "Continuously adapting based on experience",
                'confidence': 0.7,
                'applicability': 'general'
            })
        
        # Limit number of lessons
        return lessons[:5]
    
    async def _optimize_parameters(self, lessons_learned: List[Dict[str, Any]],
                                 cognitive_state: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize parameters based on lessons learned"""
        if not self.optimization_network:
            return {'optimized': False, 'reason': 'optimization_network_not_available'}
        
        try:
            # Create optimization input from lessons
            lesson_confidence = sum(lesson.get('confidence', 0.0) for lesson in lessons_learned)
            avg_confidence = lesson_confidence / len(lessons_learned) if lessons_learned else 0.0
            
            # Incorporate cognitive state
            cognitive_load = cognitive_state.get('cognitive_load', 0.0)
            
            optimization_input = torch.tensor([
                avg_confidence,
                cognitive_load,
                len(lessons_learned) / 5.0  # Normalized
            ]).float()
            
            # Apply optimization
            optimization_result = self.optimization_network(optimization_input.unsqueeze(0))
            
            # Extract parameter updates
            param_updates = {
                'learning_rate_adjustment': torch.sigmoid(optimization_result[0][0]).item(),
                'exploration_adjustment': torch.sigmoid(optimization_result[0][1]).item(),
                'adaptation_rate_adjustment': torch.sigmoid(optimization_result[0][2]).item(),
                'confidence_threshold_adjustment': torch.sigmoid(optimization_result[0][3]).item()
            }
            
            # Apply updates to configuration
            self.config['learning_rate'] *= param_updates['learning_rate_adjustment']
            self.config['learning_rate'] = max(0.0001, min(0.01, self.config['learning_rate']))
            
            logger.debug(f"优化的参数: {param_updates}")
            
            return {
                'optimized': True,
                'parameter_updates': param_updates,
                'new_learning_rate': self.config['learning_rate']
            }
            
        except Exception as e:
            logger.warning(f"参数优化失败: {e}")
            return {'optimized': False, 'reason': str(e)}
    
    async def _integrate_experience(self):
        """Integrate experiences into learned patterns"""
        if not self.experience_integration_network or len(self.experience_buffer) < 10:
            return
        
        try:
            # Sample experiences for integration
            sample_size = min(10, len(self.experience_buffer))
            sampled_experiences = self.experience_buffer[-sample_size:]
            
            # Extract experience features
            experience_features = []
            for exp in sampled_experiences:
                value = exp.get('value', 0.5)
                timestamp = exp.get('timestamp', 0.0)
                recency = 1.0 - min(1.0, (time.time() - timestamp) / 3600)  # 1 hour decay
                
                features = torch.tensor([value, recency]).float()
                experience_features.append(features)
            
            if not experience_features:
                return
            
            # Combine features
            combined_features = torch.stack(experience_features).mean(dim=0)
            
            # Integrate experiences
            integration_result = self.experience_integration_network(combined_features.unsqueeze(0))
            
            # Extract pattern
            pattern_strength = torch.sigmoid(integration_result[0][0]).item()
            pattern_novelty = torch.sigmoid(integration_result[0][1]).item()
            
            # Create new learned pattern if significant
            if pattern_strength > 0.7:
                new_pattern = {
                    'id': f"pattern_{int(time.time())}",
                    'strength': pattern_strength,
                    'novelty': pattern_novelty,
                    'based_on_experiences': sample_size,
                    'created_at': time.time(),
                    'applications': 0
                }
                
                self.learned_patterns.append(new_pattern)
                
                # Limit number of patterns
                if len(self.learned_patterns) > 50:
                    # Remove weakest patterns
                    self.learned_patterns.sort(key=lambda x: x['strength'])
                    self.learned_patterns = self.learned_patterns[-50:]
                
                logger.debug(f"将 {sample_size} 个经验整合为新模式（强度: {pattern_strength:.2f}）")
            
        except Exception as e:
            logger.warning(f"经验整合失败: {e}")
    
    def _calculate_experience_value(self, input_data: Dict[str, Any],
                                  action_result: Dict[str, Any]) -> float:
        """Calculate value of experience"""
        # Extract feedback
        feedback = action_result.get('feedback', {})
        quality = feedback.get('quality', 0.5)
        efficiency = feedback.get('efficiency', 0.5)
        learning_value = feedback.get('learning_value', 0.3)
        
        # Calculate value
        value = (
            quality * 0.4 +
            efficiency * 0.3 +
            learning_value * 0.3
        )
        
        # Adjust based on novelty
        input_type = list(input_data.keys())[0] if input_data else 'unknown'
        novelty_bonus = 0.2 if input_type not in ['text', 'image'] else 0.0
        
        return min(1.0, max(0.0, value + novelty_bonus))
    
    def _calculate_performance_improvement(self, action_result: Dict[str, Any],
                                         lessons_learned: List[Dict[str, Any]]) -> float:
        """Calculate performance improvement"""
        # Extract feedback quality
        feedback = action_result.get('feedback', {})
        current_quality = feedback.get('quality', 0.5)
        
        # Calculate lesson confidence
        lesson_confidence = sum(lesson.get('confidence', 0.0) for lesson in lessons_learned)
        avg_lesson_confidence = lesson_confidence / len(lessons_learned) if lessons_learned else 0.0
        
        # Estimate improvement
        # Higher quality + higher lesson confidence = more improvement
        improvement = (
            current_quality * 0.6 +
            avg_lesson_confidence * 0.4
        ) - 0.5  # Center around 0
        
        return improvement
    
    async def _load_learned_patterns(self):
        """Load learned patterns (simplified)"""
        self.learned_patterns = [
            {
                'id': 'pattern_cognitive_cycle',
                'strength': 0.8,
                'novelty': 0.3,
                'based_on_experiences': 50,
                'created_at': time.time() - 3600,
                'applications': 10,
                'description': 'Efficient cognitive cycle pattern'
            },
            {
                'id': 'pattern_attention_optimization',
                'strength': 0.7,
                'novelty': 0.5,
                'based_on_experiences': 30,
                'created_at': time.time() - 1800,
                'applications': 5,
                'description': 'Optimized attention allocation'
            }
        ]
    
    def _create_meta_learning_network(self) -> nn.Module:
        """Create meta-learning network"""
        return nn.Sequential(
            nn.Linear(515, 256),  # 512 + 3 for context
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Tanh()
        )
    
    def _create_optimization_network(self) -> nn.Module:
        """Create optimization network"""
        return nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4 parameter adjustments
            nn.Sigmoid()
        )
    
    def _create_experience_integration_network(self) -> nn.Module:
        """Create experience integration network"""
        return nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Pattern strength and novelty
            nn.Sigmoid()
        )
    
    async def shutdown(self):
        """Shutdown learning component"""
        logger.info("正在关闭学习组件...")
        
        # 整合剩余经验
        await self._integrate_experience()
        
        # 保存学习到的模式（简化版）
        logger.info(f"正在保存 {len(self.learned_patterns)} 个学习到的模式...")
        
        self.initialized = False
        logger.info("学习组件关闭完成")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get learning summary"""
        total_cycles = self.learning_stats['learning_cycles']
        adaptation_success_rate = (
            self.learning_stats['adaptation_successes'] /
            max(1, self.learning_stats['adaptation_successes'] + self.learning_stats['adaptation_failures'])
        )
        
        return {
            **self.learning_stats,
            'adaptation_success_rate': adaptation_success_rate,
            'experience_buffer_utilization': len(self.experience_buffer) / self.config['experience_buffer_size'],
            'learned_patterns_count': len(self.learned_patterns),
            'strong_patterns': len([p for p in self.learned_patterns if p.get('strength', 0) > 0.7]),
            'learning_efficiency': self.config['learning_rate'] * 1000  # Scaled for readability
        }