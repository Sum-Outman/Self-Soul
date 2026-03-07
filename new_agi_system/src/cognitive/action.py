"""
自适应行动组件

为统一认知架构实现自适应行动执行。
"""

import torch
import torch.nn as nn
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class AdaptiveAction:
    """自适应行动执行系统"""
    
    def __init__(self, communication):
        """
        初始化行动组件。
        
        参数:
            communication: 神经通信系统
        """
        self.communication = communication
        self.initialized = False
        
        # 行动网络
        self.execution_network = None
        self.adaptation_network = None
        self.feedback_network = None
        
        # 配置
        self.config = {
            'execution_timeout': 30.0,
            'max_retries': 3,
            'adaptation_rate': 0.1
        }
        
        # 行动历史
        self.action_history = []
        
        # 性能跟踪
        self.performance_metrics = {
            'total_actions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'total_execution_time': 0.0,
            'avg_execution_time': 0.0
        }
        
        logger.info("自适应行动系统已初始化")
    
    async def initialize(self):
        """初始化行动网络"""
        if self.initialized:
            return
        
        logger.info("正在初始化行动网络...")
        
        # 初始化行动网络
        self.execution_network = self._create_execution_network()
        self.adaptation_network = self._create_adaptation_network()
        self.feedback_network = self._create_feedback_network()
        
        self.initialized = True
        logger.info("行动网络初始化完成")
    
    async def process(self, input_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Execute adaptive action.
        
        Args:
            input_tensor: Input tensor (decision)
            metadata: Action metadata
            
        Returns:
            Action result tensor
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            decision_context = metadata.get('decision_context', {})
            cognitive_state = metadata.get('cognitive_state', {})
            
            # Execute action
            action_result = await self._execute_action(
                input_tensor, decision_context, cognitive_state
            )
            
            # Process feedback
            feedback = await self._process_feedback(
                action_result, decision_context, start_time
            )
            
            # Adapt based on results
            adaptation = await self._adapt_execution(
                action_result, feedback, decision_context
            )
            
            # Calculate actual cost
            execution_time = time.time() - start_time
            actual_cost = self._calculate_actual_cost(execution_time, feedback)
            
            # Update performance metrics
            self._update_performance_metrics(execution_time, feedback)
            
            # Record action in history
            action_entry = {
                'tensor': action_result.cpu().detach().numpy().tolist(),
                'decision_context': {
                    'selected_option': decision_context.get('selected_option', 0),
                    'value_assessment': decision_context.get('value_assessment', {})
                },
                'feedback': feedback,
                'execution_time': execution_time,
                'actual_cost': actual_cost,
                'timestamp': time.time(),
                'status': feedback.get('quality', 0.5) > 0.5
            }
            self.action_history.append(action_entry)
            
            # Limit history size
            if len(self.action_history) > 100:
                self.action_history = self.action_history[-100:]
            
            # Prepare response metadata
            response_metadata = {
                'execution_status': 'completed',
                'feedback': feedback,
                'actual_cost': actual_cost,
                'execution_time': execution_time,
                'adaptation_applied': adaptation is not None,
                'action_timestamp': time.time()
            }
            
            logger.debug(f"执行动作完成，耗时 {execution_time:.3f}秒，反馈质量: {feedback.get('quality', 0.5):.2f}")
            
            return action_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Update failure metrics
            self.performance_metrics['failed_actions'] += 1
            self.performance_metrics['total_actions'] += 1
            
            logger.error(f"动作执行失败: {e}")
            
            # Return input tensor as fallback
            return input_tensor
    
    async def _execute_action(self, input_tensor: torch.Tensor,
                            decision_context: Dict[str, Any],
                            cognitive_state: Dict[str, Any]) -> torch.Tensor:
        """Execute action"""
        if self.execution_network:
            try:
                # Incorporate decision context
                selected_option = decision_context.get('selected_option', 0)
                value_assessment = decision_context.get('value_assessment', {}).get('overall_value', 0.5)
                
                context_vector = torch.tensor([selected_option, value_assessment]).float()
                
                # Incorporate cognitive state
                cognitive_load = cognitive_state.get('cognitive_load', 0.0)
                state_vector = torch.tensor([cognitive_load]).float()
                
                # Combine with input tensor
                # 展平输入张量（从(batch, features)到(features,)）
                flattened_input = input_tensor.view(-1)
                combined = torch.cat([flattened_input, context_vector, state_vector])
                
                # Execute action
                result = self.execution_network(combined.unsqueeze(0))
                return result.squeeze(0)
            except Exception as e:
                logger.warning(f"动作执行失败: {e}")
                return input_tensor
        else:
            return input_tensor
    
    async def _process_feedback(self, action_result: torch.Tensor,
                              decision_context: Dict[str, Any],
                              start_time: float) -> Dict[str, Any]:
        """Process action feedback"""
        execution_time = time.time() - start_time
        
        try:
            if self.feedback_network:
                # Generate feedback tensor
                # 展平动作结果张量（从(batch, features)到(features,)）
                flattened_result = action_result.view(-1)
                feedback_tensor = torch.cat([
                    flattened_result,
                    torch.tensor([execution_time]).float()
                ])
                
                # Process feedback
                feedback_output = self.feedback_network(feedback_tensor.unsqueeze(0)).squeeze(0)
                
                # Extract feedback components
                quality = torch.sigmoid(feedback_output[0]).item()
                efficiency = torch.sigmoid(feedback_output[1]).item()
                learning_value = torch.sigmoid(feedback_output[2]).item()
                
                feedback = {
                    'quality': quality,
                    'efficiency': efficiency,
                    'learning_value': learning_value,
                    'execution_time': execution_time,
                    'timestamp': time.time()
                }
            else:
                # Default feedback
                expected_value = decision_context.get('value_assessment', {}).get('overall_value', 0.5)
                quality = min(1.0, expected_value * (1.0 - min(1.0, execution_time / 10.0)))
                
                feedback = {
                    'quality': quality,
                    'efficiency': 1.0 - min(1.0, execution_time / 30.0),
                    'learning_value': 0.3,
                    'execution_time': execution_time,
                    'timestamp': time.time()
                }
            
            return feedback
            
        except Exception as e:
            logger.warning(f"反馈处理失败: {e}")
            return {
                'quality': 0.5,
                'efficiency': 0.5,
                'learning_value': 0.1,
                'execution_time': execution_time,
                'error': str(e)
            }
    
    async def _adapt_execution(self, action_result: torch.Tensor,
                             feedback: Dict[str, Any],
                             decision_context: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Adapt execution based on feedback"""
        if not self.adaptation_network:
            return None
        
        try:
            # Check if adaptation is needed
            feedback_quality = feedback.get('quality', 0.5)
            if feedback_quality > 0.7:
                # Good performance, minimal adaptation
                return None
            
            # Prepare adaptation input
            feedback_tensor = torch.tensor([
                feedback_quality,
                feedback.get('efficiency', 0.5),
                feedback.get('learning_value', 0.3)
            ]).float()
            
            # Combine with action result
            # 展平动作结果张量（从(batch, features)到(features,)）
            flattened_result = action_result.view(-1)
            adaptation_input = torch.cat([flattened_result, feedback_tensor])
            
            # Generate adaptation
            adaptation = self.adaptation_network(adaptation_input.unsqueeze(0))
            
            logger.debug(f"应用自适应调整，反馈质量: {feedback_quality:.2f}")
            
            return adaptation.squeeze(0)
            
        except Exception as e:
            logger.warning(f"自适应调整失败: {e}")
            return None
    
    def _calculate_actual_cost(self, execution_time: float,
                             feedback: Dict[str, Any]) -> float:
        """Calculate actual cost of action"""
        # Base cost from execution time
        time_cost = min(1.0, execution_time / 60.0)  # Normalize to 1 minute
        
        # Quality cost (lower quality = higher cost)
        quality = feedback.get('quality', 0.5)
        quality_cost = 1.0 - quality
        
        # Efficiency cost
        efficiency = feedback.get('efficiency', 0.5)
        efficiency_cost = 1.0 - efficiency
        
        # Weighted total cost
        total_cost = (
            time_cost * 0.4 +
            quality_cost * 0.4 +
            efficiency_cost * 0.2
        )
        
        return min(1.0, max(0.0, total_cost))
    
    def _update_performance_metrics(self, execution_time: float,
                                  feedback: Dict[str, Any]):
        """Update performance metrics"""
        self.performance_metrics['total_actions'] += 1
        self.performance_metrics['total_execution_time'] += execution_time
        
        # Update average execution time
        self.performance_metrics['avg_execution_time'] = (
            self.performance_metrics['total_execution_time'] /
            self.performance_metrics['total_actions']
        )
        
        # Check if action was successful
        if feedback.get('quality', 0.5) > 0.5:
            self.performance_metrics['successful_actions'] += 1
        else:
            self.performance_metrics['failed_actions'] += 1
    
    def _create_execution_network(self) -> nn.Module:
        """Create execution network"""
        return nn.Sequential(
            nn.Linear(515, 256),  # 512 + 2 for context + 1 for state
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Tanh()
        )
    
    def _create_adaptation_network(self) -> nn.Module:
        """Create adaptation network"""
        return nn.Sequential(
            nn.Linear(515, 256),  # 512 + 3 for feedback
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Tanh()
        )
    
    def _create_feedback_network(self) -> nn.Module:
        """Create feedback network"""
        return nn.Sequential(
            nn.Linear(513, 256),  # 512 + 1 for execution time
            nn.ReLU(),
            nn.Linear(256, 3),  # Quality, efficiency, learning value
            nn.Sigmoid()
        )
    
    async def shutdown(self):
        """Shutdown action component"""
        logger.info("正在关闭行动组件...")
        self.initialized = False
        logger.info("行动组件关闭完成")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total = self.performance_metrics['total_actions']
        successful = self.performance_metrics['successful_actions']
        
        success_rate = successful / total if total > 0 else 0.0
        
        return {
            **self.performance_metrics,
            'success_rate': success_rate,
            'recent_actions': len(self.action_history[-10:]),
            'avg_feedback_quality': (
                sum(action.get('feedback', {}).get('quality', 0.5) for action in self.action_history[-10:]) /
                len(self.action_history[-10:]) if len(self.action_history[-10:]) > 0 else 0.5
            ),
            'adaptation_effectiveness': 'high' if success_rate > 0.7 else 'medium' if success_rate > 0.5 else 'low'
        }