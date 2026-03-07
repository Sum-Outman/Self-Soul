"""
层级规划组件

为统一认知架构实现层级规划。
"""

import torch
import torch.nn as nn
import asyncio
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class HierarchicalPlanning:
    """层级规划系统"""
    
    def __init__(self, communication):
        """
        初始化规划组件。
        
        参数:
            communication: 神经通信系统
        """
        self.communication = communication
        self.initialized = False
        
        # 规划网络
        self.strategic_planning = None
        self.tactical_planning = None
        self.operational_planning = None
        
        # 配置
        self.config = {
            'planning_horizon': 10,
            'planning_depth': 3,
            'max_plan_steps': 20
        }
        
        # 规划库
        self.plan_library = []
        
        logger.info("层级规划系统已初始化")
    
    async def initialize(self):
        """初始化规划网络"""
        if self.initialized:
            return
        
        logger.info("正在初始化规划网络...")
        
        # 初始化规划网络
        self.strategic_planning = self._create_strategic_planning()
        self.tactical_planning = self._create_tactical_planning()
        self.operational_planning = self._create_operational_planning()
        
        # 加载规划库
        await self._load_plan_library()
        
        self.initialized = True
        logger.info(f"规划网络初始化完成，包含 {len(self.plan_library)} 个规划")
    
    async def process(self, input_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Generate hierarchical plan.
        
        Args:
            input_tensor: Input tensor (reasoning result)
            metadata: Planning metadata
            
        Returns:
            Plan tensor
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            reasoning_context = metadata.get('reasoning_context', {})
            current_goal = metadata.get('current_goal')
            cognitive_state = metadata.get('cognitive_state', {})
            
            # Generate plan
            plan = await self._generate_plan(
                input_tensor, reasoning_context, current_goal, cognitive_state
            )
            
            # Generate plan steps
            plan_steps = self._generate_plan_steps(plan, reasoning_context)
            
            # Calculate plan metrics
            estimated_cost = self._calculate_estimated_cost(plan_steps)
            success_probability = self._calculate_success_probability(plan, reasoning_context)
            
            # Prepare response metadata
            response_metadata = {
                'plan_steps': plan_steps,
                'estimated_cost': estimated_cost,
                'success_probability': success_probability,
                'planning_level': 'hierarchical',
                'planning_timestamp': asyncio.get_event_loop().time()
            }
            
            logger.debug(f"生成了包含 {len(plan_steps)} 个步骤的计划，成功概率: {success_probability:.2f}")

            return plan

        except Exception as e:
            logger.error(f"规划失败: {e}")
            # Return input tensor as fallback
            return input_tensor
    
    async def _generate_plan(self, input_tensor: torch.Tensor,
                           reasoning_context: Dict[str, Any],
                           current_goal: Optional[Dict[str, Any]],
                           cognitive_state: Dict[str, Any]) -> torch.Tensor:
        """Generate hierarchical plan"""
        # Level 1: Strategic planning
        strategic_plan = await self._apply_strategic_planning(
            input_tensor, current_goal, cognitive_state
        )
        
        # Level 2: Tactical planning
        tactical_plan = await self._apply_tactical_planning(
            strategic_plan, reasoning_context
        )
        
        # Level 3: Operational planning
        operational_plan = await self._apply_operational_planning(
            tactical_plan, cognitive_state
        )
        
        # Combine planning levels
        combined = torch.stack([
            strategic_plan,
            tactical_plan,
            operational_plan
        ]).mean(dim=0)
        
        return combined
    
    async def _apply_strategic_planning(self, tensor: torch.Tensor,
                                      current_goal: Optional[Dict[str, Any]],
                                      cognitive_state: Dict[str, Any]) -> torch.Tensor:
        """Apply strategic planning"""
        if self.strategic_planning:
            try:
                # Incorporate goal information
                if current_goal:
                    goal_vector = torch.tensor([
                        current_goal.get('priority', 1.0),
                        len(cognitive_state.get('goal_stack', []))
                    ]).float()
                else:
                    goal_vector = torch.tensor([0.0, 0.0])
                
                # Combine with input tensor
                combined = torch.cat([tensor, goal_vector])
                
                # Apply strategic planning
                result = self.strategic_planning(combined.unsqueeze(0))
                return result.squeeze(0)
            except Exception as e:
                logger.warning(f"战略规划失败: {e}")
                return tensor
        else:
            return tensor
    
    async def _apply_tactical_planning(self, tensor: torch.Tensor,
                                     reasoning_context: Dict[str, Any]) -> torch.Tensor:
        """Apply tactical planning"""
        if self.tactical_planning:
            try:
                # Incorporate reasoning context
                reasoning_confidence = reasoning_context.get('confidence', 0.5)
                inference_steps = len(reasoning_context.get('inference_steps', []))
                
                context_vector = torch.tensor([reasoning_confidence, inference_steps]).float()
                
                # Combine with input tensor
                combined = torch.cat([tensor, context_vector])
                
                # Apply tactical planning
                result = self.tactical_planning(combined.unsqueeze(0))
                return result.squeeze(0)
            except Exception as e:
                logger.warning(f"战术规划失败: {e}")
                return tensor
        else:
            return tensor
    
    async def _apply_operational_planning(self, tensor: torch.Tensor,
                                        cognitive_state: Dict[str, Any]) -> torch.Tensor:
        """Apply operational planning"""
        if self.operational_planning:
            try:
                # Incorporate cognitive state
                cognitive_load = cognitive_state.get('cognitive_load', 0.0)
                working_memory_size = cognitive_state.get('working_memory_size', 0)
                
                state_vector = torch.tensor([cognitive_load, working_memory_size]).float()
                
                # Combine with input tensor
                combined = torch.cat([tensor, state_vector])
                
                # Apply operational planning
                result = self.operational_planning(combined.unsqueeze(0))
                return result.squeeze(0)
            except Exception as e:
                logger.warning(f"操作规划失败: {e}")
                return tensor
        else:
            return tensor
    
    def _generate_plan_steps(self, plan: torch.Tensor,
                           reasoning_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate plan steps from plan tensor"""
        # Simplified plan step generation
        # In practice, would decode plan tensor to concrete steps
        
        plan_steps = []
        
        # Generate steps based on reasoning confidence
        reasoning_confidence = reasoning_context.get('confidence', 0.5)
        num_steps = int(reasoning_confidence * 10)  # 1-10 steps based on confidence
        
        for i in range(min(num_steps, self.config['max_plan_steps'])):
            step = {
                'step_id': i + 1,
                'description': f"Plan step {i + 1}",
                'type': 'operation',
                'estimated_duration': 1.0,
                'dependencies': [] if i == 0 else [i],
                'resources': ['cognitive', 'computational']
            }
            plan_steps.append(step)
        
        return plan_steps
    
    def _calculate_estimated_cost(self, plan_steps: List[Dict[str, Any]]) -> float:
        """Calculate estimated cost of plan"""
        # Simplified cost calculation
        base_cost = len(plan_steps) * 0.5
        
        # Add complexity cost
        complexity = sum(len(step.get('dependencies', [])) for step in plan_steps)
        complexity_cost = complexity * 0.1
        
        # Add resource cost
        resource_cost = 0.0
        for step in plan_steps:
            resources = step.get('resources', [])
            resource_cost += len(resources) * 0.05
        
        total_cost = base_cost + complexity_cost + resource_cost
        return total_cost
    
    def _calculate_success_probability(self, plan: torch.Tensor,
                                     reasoning_context: Dict[str, Any]) -> float:
        """Calculate success probability of plan"""
        # Simplified success probability calculation
        
        # 1. Reasoning confidence factor
        reasoning_confidence = reasoning_context.get('confidence', 0.5)
        
        # 2. Plan stability factor
        if hasattr(plan, 'var'):
            plan_stability = 1.0 - min(1.0, plan.var().item())
        else:
            plan_stability = 0.7
        
        # 3. Plan library matching factor
        library_match = 0.0
        if self.plan_library:
            library_match = 0.3  # Simplified
        
        # Combine factors
        success_probability = (
            reasoning_confidence * 0.5 +
            plan_stability * 0.3 +
            library_match * 0.2
        )
        
        return min(1.0, max(0.0, success_probability))
    
    async def _load_plan_library(self):
        """Load plan library (simplified)"""
        self.plan_library = [
            {
                'name': 'default_cognitive_plan',
                'type': 'cognitive',
                'steps': 5,
                'success_rate': 0.8,
                'complexity': 'medium'
            },
            {
                'name': 'learning_optimization_plan',
                'type': 'learning',
                'steps': 8,
                'success_rate': 0.7,
                'complexity': 'high'
            },
            {
                'name': 'simple_execution_plan',
                'type': 'execution',
                'steps': 3,
                'success_rate': 0.9,
                'complexity': 'low'
            }
        ]
    
    def _create_strategic_planning(self) -> nn.Module:
        """Create strategic planning network"""
        return nn.Sequential(
            nn.Linear(514, 256),  # 512 + 2 for goal
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Tanh()
        )
    
    def _create_tactical_planning(self) -> nn.Module:
        """Create tactical planning network"""
        return nn.Sequential(
            nn.Linear(514, 256),  # 512 + 2 for context
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Tanh()
        )
    
    def _create_operational_planning(self) -> nn.Module:
        """Create operational planning network"""
        return nn.Sequential(
            nn.Linear(514, 256),  # 512 + 2 for cognitive state
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Tanh()
        )
    
    async def shutdown(self):
        """Shutdown planning component"""
        logger.info("正在关闭规划组件...")
        self.initialized = False
        logger.info("规划组件关闭完成")