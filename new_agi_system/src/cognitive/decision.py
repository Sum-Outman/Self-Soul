"""
价值基础决策组件

为统一认知架构实现价值基础决策。
"""

import torch
import torch.nn as nn
import asyncio
import logging
from typing import Dict, List, Any, Optional
import random

logger = logging.getLogger(__name__)


class ValueBasedDecision:
    """价值基础决策系统"""
    
    def __init__(self, communication):
        """
        初始化决策组件。
        
        参数:
            communication: 神经通信系统
        """
        self.communication = communication
        self.initialized = False
        
        # 决策网络
        self.value_network = None
        self.risk_network = None
        self.preference_network = None
        
        # 配置
        self.config = {
            'decision_threshold': 0.6,
            'exploration_rate': 0.1,
            'risk_aversion': 0.3
        }
        
        # 决策历史
        self.decision_history = []
        
        # 价值框架
        self.value_framework = {
            'utility': 1.0,
            'safety': 0.8,
            'efficiency': 0.7,
            'ethics': 0.9,
            'learning': 0.6
        }
        
        logger.info("价值基础决策系统已初始化")
    
    async def initialize(self):
        """初始化决策网络"""
        if self.initialized:
            return
        
        logger.info("正在初始化决策网络...")
        
        # 初始化决策网络
        self.value_network = self._create_value_network()
        self.risk_network = self._create_risk_network()
        self.preference_network = self._create_preference_network()
        
        self.initialized = True
        logger.info("决策网络初始化完成")
    
    async def process(self, input_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Make value-based decision.
        
        Args:
            input_tensor: Input tensor (plan)
            metadata: Decision metadata
            
        Returns:
            Decision tensor
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            plan_context = metadata.get('plan_context', {})
            cognitive_state = metadata.get('cognitive_state', {})
            
            # Make decision
            decision = await self._make_decision(
                input_tensor, plan_context, cognitive_state
            )
            
            # Evaluate options
            options = self._generate_decision_options(decision, plan_context)
            selected_option = self._select_option(options, cognitive_state)
            
            # Assess value and risk
            value_assessment = self._assess_value(selected_option, options)
            risk_assessment = self._assess_risk(selected_option, plan_context)
            
            # Prepare response metadata
            response_metadata = {
                'selected_option': selected_option,
                'value_assessment': value_assessment,
                'risk_assessment': risk_assessment,
                'decision_confidence': value_assessment.get('overall_value', 0.5),
                'decision_timestamp': asyncio.get_event_loop().time()
            }
            
            logger.debug(f"做出决策，有 {len(options)} 个选项，已选择: {selected_option}")
            
            return decision
            
        except Exception as e:
            logger.error(f"决策失败: {e}")
            # Return input tensor as fallback
            return input_tensor
    
    async def _make_decision(self, input_tensor: torch.Tensor,
                           plan_context: Dict[str, Any],
                           cognitive_state: Dict[str, Any]) -> torch.Tensor:
        """Make value-based decision"""
        # Step 1: Value assessment
        value_result = await self._apply_value_assessment(input_tensor, plan_context)
        
        # Step 2: Risk assessment
        risk_result = await self._apply_risk_assessment(value_result, plan_context)
        
        # Step 3: Preference integration
        preference_result = await self._apply_preference_integration(
            risk_result, cognitive_state
        )
        
        # Combine decision factors
        combined = torch.stack([
            value_result,
            risk_result,
            preference_result
        ]).mean(dim=0)
        
        # Record decision in history
        decision_entry = {
            'tensor': combined.cpu().detach().numpy().tolist(),
            'plan_context': {
                'success_probability': plan_context.get('success_probability', 0.5),
                'estimated_cost': plan_context.get('estimated_cost', 1.0)
            },
            'timestamp': asyncio.get_event_loop().time(),
            'cognitive_state': {
                'cognitive_load': cognitive_state.get('cognitive_load', 0.0),
                'current_goal': cognitive_state.get('current_goal')
            }
        }
        self.decision_history.append(decision_entry)
        
        # Limit history size
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]
        
        return combined
    
    async def _apply_value_assessment(self, tensor: torch.Tensor,
                                    plan_context: Dict[str, Any]) -> torch.Tensor:
        """Apply value assessment"""
        if self.value_network:
            try:
                # Incorporate plan context
                success_prob = plan_context.get('success_probability', 0.5)
                estimated_cost = plan_context.get('estimated_cost', 1.0)
                
                context_vector = torch.tensor([success_prob, estimated_cost]).float()
                
                # Combine with input tensor
                combined = torch.cat([tensor, context_vector])
                
                # Apply value assessment
                result = self.value_network(combined.unsqueeze(0))
                return result.squeeze(0)
            except Exception as e:
                logger.warning(f"价值评估失败: {e}")
                return tensor
        else:
            return tensor
    
    async def _apply_risk_assessment(self, tensor: torch.Tensor,
                                   plan_context: Dict[str, Any]) -> torch.Tensor:
        """Apply risk assessment"""
        if self.risk_network:
            try:
                # Calculate risk factors
                plan_steps = len(plan_context.get('plan_steps', []))
                complexity = sum(len(step.get('dependencies', [])) for step in plan_context.get('plan_steps', []))
                
                risk_factors = torch.tensor([plan_steps, complexity]).float()
                
                # Combine with input tensor
                combined = torch.cat([tensor, risk_factors])
                
                # Apply risk assessment
                result = self.risk_network(combined.unsqueeze(0))
                return result.squeeze(0)
            except Exception as e:
                logger.warning(f"风险评估失败: {e}")
                return tensor
        else:
            return tensor
    
    async def _apply_preference_integration(self, tensor: torch.Tensor,
                                          cognitive_state: Dict[str, Any]) -> torch.Tensor:
        """Apply preference integration"""
        if self.preference_network:
            try:
                # Get preferences from cognitive state
                preferences = cognitive_state.get('long_term_context', {}).get('preferences', {})
                
                # Create preference vector
                pref_vector = torch.tensor([
                    preferences.get('risk_tolerance', 0.5),
                    preferences.get('efficiency_preference', 0.7),
                    preferences.get('learning_preference', 0.6)
                ]).float()
                
                # Combine with input tensor
                combined = torch.cat([tensor, pref_vector])
                
                # Apply preference integration
                result = self.preference_network(combined.unsqueeze(0))
                return result.squeeze(0)
            except Exception as e:
                logger.warning(f"偏好整合失败: {e}")
                return tensor
        else:
            return tensor
    
    def _generate_decision_options(self, decision: torch.Tensor,
                                 plan_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate decision options"""
        options = []
        
        # Number of options based on plan complexity
        plan_steps = len(plan_context.get('plan_steps', []))
        num_options = min(5, max(2, plan_steps // 2))
        
        for i in range(num_options):
            option = {
                'option_id': i,
                'description': f"Decision option {i}",
                'type': 'action',
                'estimated_value': random.uniform(0.3, 0.9),
                'estimated_risk': random.uniform(0.1, 0.7),
                'resource_requirements': ['cognitive', 'computational'],
                'compatibility': random.uniform(0.5, 1.0)
            }
            options.append(option)
        
        return options
    
    def _select_option(self, options: List[Dict[str, Any]],
                      cognitive_state: Dict[str, Any]) -> int:
        """Select best option"""
        if not options:
            return 0
        
        # Get current preferences
        preferences = cognitive_state.get('long_term_context', {}).get('preferences', {})
        risk_tolerance = preferences.get('risk_tolerance', 0.5)
        efficiency_preference = preferences.get('efficiency_preference', 0.7)
        
        # Score each option
        scored_options = []
        for i, option in enumerate(options):
            # Calculate score
            value_score = option.get('estimated_value', 0.5)
            risk_score = 1.0 - option.get('estimated_risk', 0.5) * risk_tolerance
            efficiency_score = option.get('compatibility', 0.5) * efficiency_preference
            
            # Weighted sum
            total_score = (
                value_score * self.value_framework['utility'] +
                risk_score * self.value_framework['safety'] +
                efficiency_score * self.value_framework['efficiency']
            )
            
            scored_options.append((i, total_score, option))
        
        # Sort by score
        scored_options.sort(key=lambda x: x[1], reverse=True)
        
        # Exploration: sometimes choose a random option
        if random.random() < self.config['exploration_rate']:
            selected = random.choice(range(len(options)))
        else:
            selected = scored_options[0][0]
        
        return selected
    
    def _assess_value(self, selected_option: int,
                     options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess value of selected option"""
        if not options or selected_option >= len(options):
            return {'overall_value': 0.5, 'components': {}}
        
        option = options[selected_option]
        
        # Calculate value components
        utility_value = option.get('estimated_value', 0.5)
        safety_value = 1.0 - option.get('estimated_risk', 0.5)
        efficiency_value = option.get('compatibility', 0.5)
        
        # Apply value framework weights
        overall_value = (
            utility_value * self.value_framework['utility'] +
            safety_value * self.value_framework['safety'] +
            efficiency_value * self.value_framework['efficiency']
        )
        
        return {
            'overall_value': min(1.0, max(0.0, overall_value)),
            'components': {
                'utility': utility_value,
                'safety': safety_value,
                'efficiency': efficiency_value,
                'ethics': self.value_framework['ethics'],  # Fixed for now
                'learning': self.value_framework['learning']  # Fixed for now
            },
            'selected_option_details': option
        }
    
    def _assess_risk(self, selected_option: int,
                    plan_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk of selected option"""
        if not plan_context:
            return {'risk_level': 'medium', 'risk_score': 0.5, 'factors': {}}
        
        # Calculate risk factors
        plan_success_prob = plan_context.get('success_probability', 0.5)
        plan_cost = plan_context.get('estimated_cost', 1.0)
        plan_steps = len(plan_context.get('plan_steps', []))
        
        # Risk score calculation
        risk_score = (
            (1.0 - plan_success_prob) * 0.4 +
            min(1.0, plan_cost / 5.0) * 0.3 +
            min(1.0, plan_steps / 10.0) * 0.3
        )
        
        # Determine risk level
        if risk_score < 0.3:
            risk_level = 'low'
        elif risk_score < 0.7:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'factors': {
                'success_probability': plan_success_prob,
                'estimated_cost': plan_cost,
                'plan_complexity': plan_steps
            },
            'risk_mitigation': ['monitoring', 'fallback_plan'] if risk_level == 'high' else ['monitoring']
        }
    
    def _create_value_network(self) -> nn.Module:
        """Create value assessment network"""
        return nn.Sequential(
            nn.Linear(514, 256),  # 512 + 2 for context
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Sigmoid()  # Values between 0 and 1
        )
    
    def _create_risk_network(self) -> nn.Module:
        """Create risk assessment network"""
        return nn.Sequential(
            nn.Linear(514, 256),  # 512 + 2 for risk factors
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )
    
    def _create_preference_network(self) -> nn.Module:
        """Create preference integration network"""
        return nn.Sequential(
            nn.Linear(515, 256),  # 512 + 3 for preferences
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Tanh()
        )
    
    async def shutdown(self):
        """Shutdown decision component"""
        logger.info("正在关闭决策组件...")
        self.initialized = False
        logger.info("决策组件关闭完成")
    
    def get_decision_history_summary(self) -> Dict[str, Any]:
        """Get decision history summary"""
        if not self.decision_history:
            return {'total_decisions': 0, 'average_confidence': 0.0}
        
        # Calculate average success probability from history
        success_probs = []
        for decision in self.decision_history:
            success_prob = decision.get('plan_context', {}).get('success_probability', 0.5)
            success_probs.append(success_prob)
        
        avg_success_prob = sum(success_probs) / len(success_probs) if success_probs else 0.5
        
        return {
            'total_decisions': len(self.decision_history),
            'average_success_probability': avg_success_prob,
            'recent_decisions': len(self.decision_history[-10:]),
            'decision_making_style': 'value_based'
        }