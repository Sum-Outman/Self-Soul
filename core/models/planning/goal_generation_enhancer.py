#!/usr/bin/env python3
"""
目标生成和任务规划增强模块
为现有PlanningModel提供端到端的目标生成和任务规划功能

解决审计报告中的核心问题：缺乏端到端的目标生成和任务规划能力
实现从高层次描述到具体可执行任务的完整流程
"""

import os
import sys
import json
import time
import random
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import logging
from collections import defaultdict, deque, OrderedDict
from datetime import datetime, timedelta
import heapq
import hashlib

logger = logging.getLogger(__name__)

class GoalGenerationEnhancer:
    """
    目标生成和任务规划增强器
    
    核心功能：
    1. 从高层次描述生成具体目标
    2. 目标分解和优先级排序
    3. 任务规划和调度
    4. 端到端执行监控
    5. 动态调整和优化
    6. 学习与自我改进
    """
    
    def __init__(self, unified_planning_model=None):
        """
        初始化增强器
        
        Args:
            unified_planning_model: UnifiedPlanningModel实例（可选）
        """
        self.model = unified_planning_model
        self.logger = logger
        
        # 目标类型分类
        self.goal_types = {
            "achievement": "成果型目标（如完成项目、达到指标）",
            "improvement": "改进型目标（如提升性能、优化流程）",
            "learning": "学习型目标（如掌握技能、获取知识）",
            "exploration": "探索型目标（如研究新领域、发现新模式）",
            "maintenance": "维护型目标（如保持系统稳定、定期检查）",
            "creative": "创造型目标（如创作内容、设计方案）",
            "problem_solving": "问题解决型目标（如解决问题、消除障碍）",
            "decision_making": "决策型目标（如做出选择、确定方向）"
        }
        
        # 目标生成策略
        self.goal_generation_strategies = {
            "requirement_analysis": "需求分析生成",
            "gap_analysis": "差距分析生成",
            "opportunity_identification": "机会识别生成",
            "problem_decomposition": "问题分解生成",
            "resource_based": "资源导向生成",
            "constraint_based": "约束导向生成",
            "value_based": "价值导向生成",
            "vision_based": "愿景导向生成"
        }
        
        # 目标质量标准
        self.goal_quality_metrics = {
            "specificity": "具体性",
            "measurability": "可衡量性",
            "achievability": "可实现性",
            "relevance": "相关性",
            "time_bound": "时限性",
            "clarity": "清晰度",
            "alignment": "对齐性",
            "flexibility": "灵活性"
        }
        
        # 任务规划模板
        self.task_planning_templates = {
            "sequential_linear": {
                "description": "顺序线性规划",
                "pattern": "start → task1 → task2 → ... → end",
                "use_case": "简单线性依赖任务"
            },
            "parallel_independent": {
                "description": "并行独立规划",
                "pattern": "task1, task2, task3 → merge_results",
                "use_case": "相互独立的任务"
            },
            "hierarchical_decomposition": {
                "description": "层次分解规划",
                "pattern": "goal → subgoal1 → task1, task2 → subgoal2 → task3, task4",
                "use_case": "复杂多层次目标"
            },
            "adaptive_iterative": {
                "description": "自适应迭代规划",
                "pattern": "plan → execute → evaluate → adjust → repeat",
                "use_case": "不确定环境下的目标"
            },
            "conditional_branching": {
                "description": "条件分支规划",
                "pattern": "if condition1: path1 else: path2",
                "use_case": "需要条件判断的目标"
            },
            "time_constrained": {
                "description": "时间约束规划",
                "pattern": "task1[time1] → task2[time2] → ... → deadline",
                "use_case": "有时间限制的目标"
            }
        }
        
        # 目标知识库
        self.goal_knowledge_base = {
            "common_goal_patterns": {
                "performance_improvement": ["baseline_measurement", "identify_bottlenecks", "implement_solutions", "measure_results"],
                "skill_acquisition": ["assess_current_level", "identify_gaps", "practice_exercises", "evaluate_progress"],
                "problem_solution": ["define_problem", "analyze_causes", "generate_solutions", "implement_best"],
                "project_completion": ["define_scope", "create_plan", "execute_tasks", "deliver_results"]
            },
            "goal_decomposition_rules": {
                "by_function": "按功能分解",
                "by_time": "按时序分解",
                "by_resource": "按资源分解",
                "by_difficulty": "按难度分解",
                "by_dependency": "按依赖关系分解"
            },
            "success_criteria_templates": {
                "quantitative": "达到{metric}的{value}{unit}",
                "qualitative": "实现{quality}的{level}水平",
                "completion": "完成{task_count}个任务中的{completion_rate}%",
                "time_based": "在{time_limit}内完成",
                "resource_based": "使用不超过{resource_limit}的{resource_type}"
            }
        }
        
        # 初始化神经网络组件
        self._initialize_neural_components()
        
        # 学习和适应数据
        self.learning_data = {
            "generated_goals": [],
            "successful_plans": [],
            "failed_plans": [],
            "adaptation_patterns": [],
            "strategy_effectiveness": {},
            "quality_metrics_history": []
        }
        
        logger.info("目标生成和任务规划增强器初始化完成")
    
    def _initialize_neural_components(self):
        """初始化神经网络组件"""
        try:
            # 目标生成网络
            self.goal_generation_network = GoalGenerationNetwork()
            
            # 目标分类网络
            self.goal_classification_network = GoalClassificationNetwork()
            
            # 任务分解网络
            self.task_decomposition_network = TaskDecompositionNetwork()
            
            # 规划质量评估网络
            self.planning_quality_network = PlanningQualityNetwork()
            
            # 移动到GPU（如果可用）
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.goal_generation_network.to(self.device)
            self.goal_classification_network.to(self.device)
            self.task_decomposition_network.to(self.device)
            self.planning_quality_network.to(self.device)
            
            logger.info(f"神经网络组件初始化完成，使用设备: {self.device}")
            
        except Exception as e:
            logger.warning(f"神经网络组件初始化失败，使用基础方法: {e}")
            self.goal_generation_network = None
            self.goal_classification_network = None
            self.task_decomposition_network = None
            self.planning_quality_network = None
    
    def enhance_planning_model(self):
        """增强PlanningModel，提供端到端目标生成和任务规划功能"""
        logger.info(f"开始增强规划模型，self.model: {self.model}")
        
        if self.model is None:
            logger.error("无法增强PlanningModel：未提供模型实例")
            return False
        
        try:
            logger.info("步骤1：添加目标生成方法")
            self._add_goal_generation_methods()
            
            logger.info("步骤2：添加目标分析方法")
            self._add_goal_analysis_methods()
            
            logger.info("步骤3：添加任务规划方法")
            self._add_task_planning_methods()
            
            logger.info("步骤4：添加执行监控方法")
            self._add_execution_monitoring_methods()
            
            logger.info("步骤5：添加优化改进方法")
            self._add_optimization_improvement_methods()
            
            logger.info("PlanningModel增强完成，添加了端到端目标生成和任务规划功能")
            return True
            
        except Exception as e:
            logger.error(f"增强PlanningModel失败: {e}")
            import traceback
            logger.error(f"详细错误跟踪: {traceback.format_exc()}")
            return False
    
    def _add_goal_generation_methods(self):
        """添加目标生成方法"""
        # 1. 从描述生成目标
        if not hasattr(self.model, 'generate_goal_from_description'):
            self.model.generate_goal_from_description = self._generate_goal_from_description
        
        # 2. 从需求生成目标
        if not hasattr(self.model, 'generate_goal_from_requirements'):
            self.model.generate_goal_from_requirements = self._generate_goal_from_requirements
        
        # 3. 从问题生成目标
        if not hasattr(self.model, 'generate_goal_from_problem'):
            self.model.generate_goal_from_problem = self._generate_goal_from_problem
        
        # 4. 从机会生成目标
        if not hasattr(self.model, 'generate_goal_from_opportunity'):
            self.model.generate_goal_from_opportunity = self._generate_goal_from_opportunity
        
        # 5. 生成目标层次结构
        if not hasattr(self.model, 'generate_goal_hierarchy'):
            self.model.generate_goal_hierarchy = self._generate_goal_hierarchy
        
        # 6. 生成SMART目标
        if not hasattr(self.model, 'generate_smart_goal'):
            self.model.generate_smart_goal = self._generate_smart_goal
        
        self.logger.info("添加了目标生成方法")
    
    def _add_goal_analysis_methods(self):
        """添加目标分析方法"""
        # 1. 分析目标质量
        if not hasattr(self.model, 'analyze_goal_quality'):
            self.model.analyze_goal_quality = self._analyze_goal_quality
        
        # 2. 评估目标可行性
        if not hasattr(self.model, 'assess_goal_feasibility'):
            self.model.assess_goal_feasibility = self._assess_goal_feasibility
        
        # 3. 识别目标依赖关系
        if not hasattr(self.model, 'identify_goal_dependencies'):
            self.model.identify_goal_dependencies = self._identify_goal_dependencies
        
        # 4. 分析目标复杂度
        if not hasattr(self.model, 'analyze_goal_complexity'):
            self.model.analyze_goal_complexity = self._analyze_goal_complexity
        
        # 5. 评估目标价值
        if not hasattr(self.model, 'evaluate_goal_value'):
            self.model.evaluate_goal_value = self._evaluate_goal_value
        
        # 6. 识别目标风险
        if not hasattr(self.model, 'identify_goal_risks'):
            self.model.identify_goal_risks = self._identify_goal_risks
        
        self.logger.info("添加了目标分析方法")
    
    def _add_task_planning_methods(self):
        """添加任务规划方法"""
        # 1. 从目标生成任务
        if not hasattr(self.model, 'generate_tasks_from_goal'):
            self.model.generate_tasks_from_goal = self._generate_tasks_from_goal
        
        # 2. 创建详细执行计划
        if not hasattr(self.model, 'create_detailed_execution_plan'):
            self.model.create_detailed_execution_plan = self._create_detailed_execution_plan
        
        # 3. 优化任务顺序
        if not hasattr(self.model, 'optimize_task_sequence'):
            self.model.optimize_task_sequence = self._optimize_task_sequence
        
        # 4. 分配任务资源
        if not hasattr(self.model, 'allocate_task_resources'):
            self.model.allocate_task_resources = self._allocate_task_resources
        
        # 5. 估算任务时间
        if not hasattr(self.model, 'estimate_task_times'):
            self.model.estimate_task_times = self._estimate_task_times
        
        # 6. 识别关键路径
        if not hasattr(self.model, 'identify_critical_path'):
            self.model.identify_critical_path = self._identify_critical_path
        
        self.logger.info("添加了任务规划方法")
    
    def _add_execution_monitoring_methods(self):
        """添加执行监控方法"""
        # 1. 监控执行进度
        if not hasattr(self.model, 'monitor_execution_progress'):
            self.model.monitor_execution_progress = self._monitor_execution_progress
        
        # 2. 检测执行偏差
        if not hasattr(self.model, 'detect_execution_deviations'):
            self.model.detect_execution_deviations = self._detect_execution_deviations
        
        # 3. 调整执行计划
        if not hasattr(self.model, 'adjust_execution_plan'):
            self.model.adjust_execution_plan = self._adjust_execution_plan
        
        # 4. 报告执行状态
        if not hasattr(self.model, 'report_execution_status'):
            self.model.report_execution_status = self._report_execution_status
        
        # 5. 评估执行效果
        if not hasattr(self.model, 'evaluate_execution_effectiveness'):
            self.model.evaluate_execution_effectiveness = self._evaluate_execution_effectiveness
        
        self.logger.info("添加了执行监控方法")
    
    def _add_optimization_improvement_methods(self):
        """添加优化改进方法"""
        # 1. 优化目标质量
        if not hasattr(self.model, 'optimize_goal_quality'):
            self.model.optimize_goal_quality = self._optimize_goal_quality
        
        # 2. 改进规划策略
        if not hasattr(self.model, 'improve_planning_strategies'):
            self.model.improve_planning_strategies = self._improve_planning_strategies
        
        # 3. 学习执行经验
        if not hasattr(self.model, 'learn_from_execution_experience'):
            self.model.learn_from_execution_experience = self._learn_from_execution_experience
        
        # 4. 适应环境变化
        if not hasattr(self.model, 'adapt_to_environment_changes'):
            self.model.adapt_to_environment_changes = self._adapt_to_environment_changes
        
        # 5. 自我评估改进
        if not hasattr(self.model, 'self_assess_and_improve'):
            self.model.self_assess_and_improve = self._self_assess_and_improve
        
        self.logger.info("添加了优化改进方法")
    
    # ==================== 目标生成方法 ====================
    
    def _generate_goal_from_description(self, description: str, context: Dict = None) -> Dict[str, Any]:
        """
        从高层次描述生成具体目标
        
        Args:
            description: 高层次目标描述
            context: 上下文信息（资源、约束、环境等）
            
        Returns:
            生成的目标字典
        """
        try:
            # 分析描述类型
            description_type = self._classify_description_type(description)
            
            # 提取关键信息
            key_elements = self._extract_key_elements(description)
            
            # 生成目标结构
            goal_structure = self._generate_goal_structure(description_type, key_elements)
            
            # 应用SMART原则
            smart_goal = self._apply_smart_principles(goal_structure, context)
            
            # 评估目标质量
            quality_metrics = self._evaluate_goal_quality(smart_goal)
            
            # 生成完整目标
            goal = {
                "id": f"goal_{int(time.time())}_{hashlib.md5(description.encode()).hexdigest()[:8]}",
                "description": description,
                "structured_goal": smart_goal,
                "type": description_type,
                "key_elements": key_elements,
                "quality_metrics": quality_metrics,
                "metrics": quality_metrics,
                "success_criteria": [],
                "generation_timestamp": time.time(),
                "generation_method": "from_description",
                "context": context or {}
            }
            
            # 记录学习数据
            self.learning_data["generated_goals"].append({
                "goal_id": goal["id"],
                "description": description,
                "type": description_type,
                "quality_score": quality_metrics.get("overall_score", 0),
                "timestamp": time.time()
            })
            
            logger.info(f"从描述生成目标: {description[:50]}..., 类型: {description_type}, 质量: {quality_metrics.get('overall_score', 0):.2f}")
            
            return goal
            
        except Exception as e:
            logger.error(f"从描述生成目标失败: {e}")
            return {
                "error": str(e),
                "description": description,
                "generation_method": "from_description"
            }
    
    def generate_goals_from_description(self, description: str, context: Dict = None) -> List[Dict[str, Any]]:
        """
        公有方法：从描述生成目标列表
        
        Args:
            description: 高层次目标描述
            context: 上下文信息
            
        Returns:
            生成的目标列表（包含多个目标变体）
        """
        try:
            # 生成主要目标
            main_goal = self._generate_goal_from_description(description, context)
            
            # 生成替代目标变体
            alternative_goals = []
            
            # 基于不同解释生成变体
            interpretations = self._generate_alternative_interpretations(description)
            
            for interpretation in interpretations[:2]:  # 最多生成2个变体
                alt_goal = self._generate_goal_from_description(interpretation, context)
                if "error" not in alt_goal:
                    alternative_goals.append(alt_goal)
            
            # 构建结果列表
            all_goals = [main_goal] + alternative_goals
            
            # 添加元信息
            result = {
                "original_description": description,
                "total_goals": len(all_goals),
                "generation_timestamp": time.time(),
                "goals": all_goals,
                "main_goal_id": main_goal.get("id", ""),
                "context": context or {}
            }
            
            logger.info(f"从描述生成{len(all_goals)}个目标: {description[:50]}...")
            
            return all_goals
            
        except Exception as e:
            logger.error(f"生成目标列表失败: {e}")
            return [{
                "error": str(e),
                "description": description,
                "generation_method": "from_description"
            }]
    
    def _generate_goal_from_requirements(self, requirements: List[str], constraints: Dict = None) -> Dict[str, Any]:
        """
        从需求列表生成目标
        
        Args:
            requirements: 需求列表
            constraints: 约束条件
            
        Returns:
            生成的目标字典
        """
        try:
            # 分析需求类型
            requirement_types = [self._classify_requirement_type(req) for req in requirements]
            
            # 识别核心需求
            core_requirements = self._identify_core_requirements(requirements)
            
            # 生成目标层次
            goal_hierarchy = self._create_goal_hierarchy_from_requirements(core_requirements)
            
            # 应用约束条件
            constrained_goal = self._apply_constraints_to_goal(goal_hierarchy, constraints or {})
            
            # 生成完整目标
            goal = {
                "id": f"goal_req_{int(time.time())}",
                "requirements": requirements,
                "requirement_types": requirement_types,
                "core_requirements": core_requirements,
                "goal_hierarchy": constrained_goal,
                "constraints": constraints or {},
                "generation_timestamp": time.time(),
                "generation_method": "from_requirements"
            }
            
            logger.info(f"从需求生成目标: {len(requirements)}个需求, {len(core_requirements)}个核心需求")
            
            return goal
            
        except Exception as e:
            logger.error(f"从需求生成目标失败: {e}")
            return {
                "error": str(e),
                "requirements": requirements,
                "generation_method": "from_requirements"
            }
    
    def _generate_goal_from_problem(self, problem_description: str, current_state: Dict = None) -> Dict[str, Any]:
        """
        从问题描述生成解决目标
        
        Args:
            problem_description: 问题描述
            current_state: 当前状态信息
            
        Returns:
            生成的目标字典
        """
        try:
            # 分析问题类型
            problem_type = self._classify_problem_type(problem_description)
            
            # 识别问题根源
            root_causes = self._identify_root_causes(problem_description, current_state)
            
            # 生成解决方案目标
            solution_goals = self._generate_solution_goals(problem_type, root_causes)
            
            # 优先级排序
            prioritized_goals = self._prioritize_goals(solution_goals, current_state)
            
            # 生成完整目标
            goal = {
                "id": f"goal_prob_{int(time.time())}",
                "problem_description": problem_description,
                "problem_type": problem_type,
                "root_causes": root_causes,
                "solution_goals": prioritized_goals,
                "current_state": current_state or {},
                "generation_timestamp": time.time(),
                "generation_method": "from_problem"
            }
            
            logger.info(f"从问题生成目标: {problem_description[:50]}..., 类型: {problem_type}, {len(prioritized_goals)}个解决方案目标")
            
            return goal
            
        except Exception as e:
            logger.error(f"从问题生成目标失败: {e}")
            return {
                "error": str(e),
                "problem_description": problem_description,
                "generation_method": "from_problem"
            }
    
    def _generate_goal_hierarchy(self, main_goal: str, depth: int = 3) -> Dict[str, Any]:
        """
        生成目标层次结构
        
        Args:
            main_goal: 主要目标
            depth: 层次深度
            
        Returns:
            目标层次结构字典
        """
        try:
            # 生成顶层目标
            top_level_goal = self._generate_goal_from_description(main_goal)
            
            # 分解为子目标
            subgoals = self._decompose_goal_into_subgoals(top_level_goal, depth)
            
            # 构建层次结构
            hierarchy = self._build_goal_hierarchy(top_level_goal, subgoals)
            
            # 评估层次质量
            hierarchy_quality = self._evaluate_hierarchy_quality(hierarchy)
            
            # 生成完整层次结构
            result = {
                "main_goal": main_goal,
                "top_level_goal": top_level_goal,
                "subgoals": subgoals,
                "hierarchy": hierarchy,
                "hierarchy_quality": hierarchy_quality,
                "depth": depth,
                "total_goals": len(subgoals) + 1,
                "generation_timestamp": time.time()
            }
            
            logger.info(f"生成目标层次结构: {main_goal[:30]}..., 深度: {depth}, 目标总数: {result['total_goals']}")
            
            return result
            
        except Exception as e:
            logger.error(f"生成目标层次结构失败: {e}")
            return {
                "error": str(e),
                "main_goal": main_goal,
                "depth": depth
            }
    
    def _generate_smart_goal(self, raw_goal: str, context: Dict = None) -> Dict[str, Any]:
        """
        生成符合SMART原则的目标
        
        Args:
            raw_goal: 原始目标描述
            context: 上下文信息
            
        Returns:
            SMART目标字典
        """
        try:
            # 应用SMART原则
            smart_components = self._apply_smart_components(raw_goal, context)
            
            # 构建SMART目标
            smart_goal = self._build_smart_goal(smart_components)
            
            # 评估SMART合规性
            smart_compliance = self._evaluate_smart_compliance(smart_goal)
            
            # 生成完整SMART目标
            result = {
                "raw_goal": raw_goal,
                "smart_goal": smart_goal,
                "smart_components": smart_components,
                "smart_compliance": smart_compliance,
                "quality_score": smart_compliance.get("overall_score", 0),
                "generation_timestamp": time.time(),
                "context": context or {}
            }
            
            logger.info(f"生成SMART目标: {raw_goal[:30]}..., 质量分数: {result['quality_score']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"生成SMART目标失败: {e}")
            return {
                "error": str(e),
                "raw_goal": raw_goal
            }
    
    # ==================== 辅助方法 ====================
    
    def _classify_description_type(self, description: str) -> str:
        """分类描述类型"""
        desc_lower = description.lower()
        
        # 检查关键词
        if any(word in desc_lower for word in ["achieve", "complete", "finish", "accomplish"]):
            return "achievement"
        elif any(word in desc_lower for word in ["improve", "enhance", "optimize", "increase", "reduce"]):
            return "improvement"
        elif any(word in desc_lower for word in ["learn", "study", "understand", "master", "acquire"]):
            return "learning"
        elif any(word in desc_lower for word in ["explore", "discover", "investigate", "research"]):
            return "exploration"
        elif any(word in desc_lower for word in ["maintain", "keep", "preserve", "sustain"]):
            return "maintenance"
        elif any(word in desc_lower for word in ["create", "design", "develop", "build", "write"]):
            return "creative"
        elif any(word in desc_lower for word in ["solve", "fix", "resolve", "address"]):
            return "problem_solving"
        elif any(word in desc_lower for word in ["decide", "choose", "select", "determine"]):
            return "decision_making"
        else:
            return "general"
    
    def _extract_key_elements(self, description: str) -> Dict[str, Any]:
        """提取关键元素"""
        elements = {
            "actions": [],
            "objects": [],
            "metrics": [],
            "time_frames": [],
            "conditions": []
        }
        
        # 简单正则匹配提取
        import re
        
        # 提取动作词
        action_words = ["achieve", "complete", "create", "build", "improve", "learn", 
                       "explore", "maintain", "solve", "decide", "generate", "develop"]
        for word in action_words:
            if word in description.lower():
                elements["actions"].append(word)
        
        # 提取数字和度量
        metric_patterns = [
            r'(\d+)\s*%',  # 百分比
            r'(\d+)\s*(times|hours|days|weeks|months|years)',  # 时间和次数
            r'(\d+\.?\d*)\s*(points|score|rating)',  # 分数
            r'increase\s+by\s+(\d+)',  # 增加量
            r'reduce\s+by\s+(\d+)'  # 减少量
        ]
        
        for pattern in metric_patterns:
            matches = re.findall(pattern, description.lower())
            for match in matches:
                if isinstance(match, tuple):
                    elements["metrics"].append(" ".join(match))
                else:
                    elements["metrics"].append(match)
        
        # 提取时间框架
        time_patterns = [
            r'by\s+(\w+\s+\d+)',  # by December 2024
            r'in\s+(\d+)\s*(days|weeks|months|years)',  # in 3 months
            r'within\s+(\d+)\s*(days|weeks|months|years)',  # within 2 weeks
            r'before\s+(\w+\s+\d+)'  # before March 1st
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, description.lower())
            for match in matches:
                if isinstance(match, tuple):
                    elements["time_frames"].append(" ".join(match))
                else:
                    elements["time_frames"].append(match)
        
        return elements
    
    def _generate_alternative_interpretations(self, description: str) -> List[str]:
        """
        生成替代解释
        
        Args:
            description: 原始描述
            
        Returns:
            替代解释列表
        """
        interpretations = []
        
        # 添加原始描述作为第一个解释
        interpretations.append(description)
        
        # 生成基于不同重点的解释
        desc_lower = description.lower()
        
        # 如果描述包含"improve"，生成不同方面的改进解释
        if "improve" in desc_lower:
            aspects = ["performance", "efficiency", "quality", "reliability", "usability"]
            for aspect in aspects[:2]:  # 最多2个方面
                new_desc = description.replace("improve", f"improve {aspect}")
                if new_desc != description:
                    interpretations.append(new_desc)
        
        # 如果描述包含时间范围，生成不同时间框架的解释
        time_phrases = ["next quarter", "next month", "within 3 months", "by the end of the year"]
        for time_phrase in time_phrases:
            if time_phrase in desc_lower:
                # 保持原始时间框架
                interpretations.append(description)
                
                # 添加其他时间框架变体
                for alt_time in ["within 1 month", "within 6 months", "as soon as possible"]:
                    if alt_time != time_phrase:
                        new_desc = description.replace(time_phrase, alt_time)
                        interpretations.append(new_desc)
                break
        
        # 如果描述包含度量指标，生成不同度量目标的解释
        if "%" in description or "times" in description:
            # 保持原始度量
            interpretations.append(description)
            
            # 添加调整度量的变体
            import re
            percent_match = re.search(r'(\d+)\s*%', description)
            if percent_match:
                percent_value = int(percent_match.group(1))
                if percent_value > 10:
                    # 生成更保守的目标
                    conservative_value = max(10, percent_value - 10)
                    conservative_desc = description.replace(f"{percent_value}%", f"{conservative_value}%")
                    interpretations.append(conservative_desc)
                    
                    # 生成更积极的目标
                    ambitious_value = min(100, percent_value + 10)
                    ambitious_desc = description.replace(f"{percent_value}%", f"{ambitious_value}%")
                    interpretations.append(ambitious_desc)
        
        # 确保唯一性
        unique_interpretations = []
        seen = set()
        for interp in interpretations:
            if interp not in seen:
                seen.add(interp)
                unique_interpretations.append(interp)
        
        return unique_interpretations[:5]  # 最多返回5个解释
    
    def _generate_goal_structure(self, goal_type: str, key_elements: Dict) -> Dict[str, Any]:
        """生成目标结构"""
        structure = {
            "type": goal_type,
            "key_elements": key_elements,
            "components": {}
        }
        
        # 根据目标类型添加特定组件
        if goal_type == "achievement":
            structure["components"] = {
                "target": "待确定的目标",
                "success_criteria": ["完成所有必要任务"],
                "completion_indicators": ["100%任务完成"]
            }
        elif goal_type == "improvement":
            structure["components"] = {
                "current_state": "待评估",
                "target_state": "待确定",
                "improvement_metrics": key_elements.get("metrics", []),
                "measurement_method": "待确定"
            }
        elif goal_type == "learning":
            structure["components"] = {
                "skill_topic": "待确定",
                "learning_objectives": ["理解基本概念", "掌握核心技能"],
                "assessment_method": ["测试", "实践", "项目"]
            }
        
        return structure
    
    def _apply_smart_principles(self, goal_structure: Dict, context: Dict) -> Dict[str, Any]:
        """应用SMART原则"""
        smart_goal = {
            "specific": self._make_specific(goal_structure, context),
            "measurable": self._make_measurable(goal_structure, context),
            "achievable": self._make_achievable(goal_structure, context),
            "relevant": self._make_relevant(goal_structure, context),
            "time_bound": self._make_time_bound(goal_structure, context)
        }
        
        # 构建完整SMART目标描述
        smart_description = f"{smart_goal['specific']}. " \
                          f"It will be measured by {smart_goal['measurable']}. " \
                          f"It is achievable because {smart_goal['achievable']}. " \
                          f"It is relevant because {smart_goal['relevant']}. " \
                          f"It will be completed by {smart_goal['time_bound']}."
        
        smart_goal["description"] = smart_description
        smart_goal["original_structure"] = goal_structure
        
        return smart_goal
    
    def _evaluate_goal_quality(self, goal: Dict) -> Dict[str, Any]:
        """评估目标质量"""
        quality_metrics = {}
        
        # 计算各维度分数
        quality_metrics["specificity"] = self._calculate_specificity_score(goal)
        quality_metrics["measurability"] = self._calculate_measurability_score(goal)
        quality_metrics["achievability"] = self._calculate_achievability_score(goal)
        quality_metrics["relevance"] = self._calculate_relevance_score(goal)
        quality_metrics["time_bound"] = self._calculate_time_bound_score(goal)
        quality_metrics["clarity"] = self._calculate_clarity_score(goal)
        
        # 计算总体分数
        scores = [quality_metrics[key] for key in ["specificity", "measurability", "achievability", 
                                                  "relevance", "time_bound", "clarity"]]
        quality_metrics["overall_score"] = sum(scores) / len(scores)
        
        # 质量等级
        overall_score = quality_metrics["overall_score"]
        if overall_score >= 0.8:
            quality_metrics["quality_level"] = "excellent"
        elif overall_score >= 0.6:
            quality_metrics["quality_level"] = "good"
        elif overall_score >= 0.4:
            quality_metrics["quality_level"] = "fair"
        else:
            quality_metrics["quality_level"] = "poor"
        
        # 改进建议
        quality_metrics["improvement_suggestions"] = self._generate_improvement_suggestions(quality_metrics)
        
        return quality_metrics
    
    # ==================== SMART原则实现方法 ====================
    
    def _make_specific(self, goal_structure: Dict, context: Dict) -> str:
        """使目标具体化"""
        goal_type = goal_structure.get("type", "general")
        key_elements = goal_structure.get("key_elements", {})
        
        actions = key_elements.get("actions", [])
        if not actions:
            actions = ["achieve"]
        
        # 构建具体描述
        specific_parts = []
        
        # 添加动作
        specific_parts.append(actions[0] if actions else "achieve")
        
        # 添加对象/主题
        if goal_type == "improvement":
            specific_parts.append("performance improvement")
        elif goal_type == "learning":
            specific_parts.append("new skill acquisition")
        elif goal_type == "problem_solving":
            specific_parts.append("identified problem resolution")
        else:
            specific_parts.append("the defined objective")
        
        # 添加具体指标（如果有）
        metrics = key_elements.get("metrics", [])
        if metrics:
            specific_parts.append(f"measured by {metrics[0]}")
        
        return " ".join(specific_parts)
    
    def _make_measurable(self, goal_structure: Dict, context: Dict) -> str:
        """使目标可衡量"""
        key_elements = goal_structure.get("key_elements", {})
        metrics = key_elements.get("metrics", [])
        
        if metrics:
            # 使用提取的度量
            return f"tracking progress through {', '.join(metrics[:2])}"
        else:
            # 默认度量
            goal_type = goal_structure.get("type", "general")
            if goal_type == "achievement":
                return "completion percentage and quality assessment"
            elif goal_type == "learning":
                return "skill assessment tests and practical demonstrations"
            elif goal_type == "improvement":
                return "performance metrics comparison before and after"
            else:
                return "specific milestones and completion criteria"
    
    def _make_achievable(self, goal_structure: Dict, context: Dict) -> str:
        """使目标可实现"""
        resources = context.get("resources", {}) if context else {}
        
        if resources:
            resource_count = len(resources)
            return f"available resources include {resource_count} key capabilities"
        else:
            return "it aligns with current capabilities and can be broken down into manageable steps"
    
    def _make_relevant(self, goal_structure: Dict, context: Dict) -> str:
        """使目标相关"""
        goal_type = goal_structure.get("type", "general")
        
        relevance_map = {
            "achievement": "it contributes to overall mission success",
            "improvement": "it enhances system performance and efficiency",
            "learning": "it builds capabilities for future challenges",
            "problem_solving": "it addresses critical issues affecting operations",
            "creative": "it enables innovation and new possibilities",
            "exploration": "it expands knowledge and discovers new opportunities"
        }
        
        return relevance_map.get(goal_type, "it aligns with strategic objectives and priorities")
    
    def _make_time_bound(self, goal_structure: Dict, context: Dict) -> str:
        """使目标有时限"""
        key_elements = goal_structure.get("key_elements", {})
        time_frames = key_elements.get("time_frames", [])
        
        if time_frames:
            return f"{time_frames[0]}"
        else:
            # 默认时间框架
            goal_type = goal_structure.get("type", "general")
            if goal_type in ["achievement", "problem_solving"]:
                return "the next 30 days"
            elif goal_type == "learning":
                return "the next 3 months"
            elif goal_type == "improvement":
                return "the next quarter"
            else:
                return "a reasonable timeframe based on complexity"
    
    # ==================== 质量评估方法 ====================
    
    def _calculate_specificity_score(self, goal: Dict) -> float:
        """计算具体性分数"""
        smart_goal = goal.get("smart_goal", {})
        specific = smart_goal.get("specific", "")
        
        # 基于描述长度和具体词汇
        if not specific:
            return 0.3
        
        score = 0.5
        
        # 检查是否有具体动作
        specific_actions = ["achieve", "complete", "create", "build", "improve", "learn"]
        if any(action in specific.lower() for action in specific_actions):
            score += 0.2
        
        # 检查是否有具体对象
        if len(specific.split()) > 5:
            score += 0.2
        
        # 检查是否有量化指标
        if any(char.isdigit() for char in specific):
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_measurability_score(self, goal: Dict) -> float:
        """计算可衡量性分数"""
        smart_goal = goal.get("smart_goal", {})
        measurable = smart_goal.get("measurable", "")
        
        if not measurable:
            return 0.3
        
        score = 0.5
        
        # 检查是否有度量词汇
        measure_words = ["measure", "track", "assess", "evaluate", "quantify", "percentage"]
        if any(word in measurable.lower() for word in measure_words):
            score += 0.2
        
        # 检查是否有具体指标
        if any(char.isdigit() for char in measurable):
            score += 0.2
        
        # 检查是否有比较基准
        if "before" in measurable.lower() or "after" in measurable.lower():
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_achievability_score(self, goal: Dict) -> float:
        """计算可实现性分数"""
        smart_goal = goal.get("smart_goal", {})
        achievable = smart_goal.get("achievable", "")
        
        if not achievable:
            return 0.5  # 默认中等分数
        
        score = 0.6
        
        # 检查是否有资源提及
        resource_words = ["resource", "capability", "skill", "tool", "support"]
        if any(word in achievable.lower() for word in resource_words):
            score += 0.2
        
        # 检查是否有可行性说明
        if "manageable" in achievable.lower() or "feasible" in achievable.lower():
            score += 0.1
        
        # 检查是否有步骤分解提及
        if "step" in achievable.lower() or "break down" in achievable.lower():
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_relevance_score(self, goal: Dict) -> float:
        """计算相关性分数"""
        smart_goal = goal.get("smart_goal", {})
        relevant = smart_goal.get("relevant", "")
        
        if not relevant:
            return 0.6  # 默认中等分数
        
        score = 0.7
        
        # 检查是否有战略对齐词汇
        alignment_words = ["align", "contribute", "support", "mission", "strategic", "priority"]
        if any(word in relevant.lower() for word in alignment_words):
            score += 0.2
        
        # 检查是否有价值说明
        if "value" in relevant.lower() or "benefit" in relevant.lower():
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_time_bound_score(self, goal: Dict) -> float:
        """计算时限性分数"""
        smart_goal = goal.get("smart_goal", {})
        time_bound = smart_goal.get("time_bound", "")
        
        if not time_bound:
            return 0.4
        
        score = 0.5
        
        # 检查是否有具体时间
        time_patterns = [
            r'\d+\s*(days|weeks|months|years)',
            r'\w+\s+\d+',  # December 2024
            r'Q\d',  # Q1, Q2
            r'end of \w+'  # end of month
        ]
        
        import re
        for pattern in time_patterns:
            if re.search(pattern, time_bound.lower()):
                score += 0.3
                break
        
        # 检查是否有时间框架词
        time_words = ["by", "within", "before", "deadline", "timeframe"]
        if any(word in time_bound.lower() for word in time_words):
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_clarity_score(self, goal: Dict) -> float:
        """计算清晰度分数"""
        smart_goal = goal.get("smart_goal", {})
        description = smart_goal.get("description", "")
        
        if not description:
            return 0.3
        
        score = 0.5
        
        # 基于句子结构
        sentences = description.split('.')
        if len(sentences) >= 3:  # 至少有3个句子
            score += 0.2
        
        # 检查是否有明确的主谓宾结构
        if "will be" in description or "is to" in description:
            score += 0.2
        
        # 检查是否有连接词（表示逻辑清晰）
        connectives = ["because", "through", "by", "and", "which"]
        connective_count = sum(1 for word in connectives if word in description.lower())
        score += min(0.1, connective_count * 0.05)
        
        return min(1.0, score)
    
    def _generate_improvement_suggestions(self, quality_metrics: Dict) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        # 检查各维度分数，为低分维度生成建议
        if quality_metrics.get("specificity", 0) < 0.6:
            suggestions.append("Make the goal more specific by adding concrete actions and objects")
        
        if quality_metrics.get("measurability", 0) < 0.6:
            suggestions.append("Add measurable criteria or metrics to track progress")
        
        if quality_metrics.get("achievability", 0) < 0.6:
            suggestions.append("Break down the goal into smaller, more manageable steps")
        
        if quality_metrics.get("relevance", 0) < 0.6:
            suggestions.append("Clearly explain how this goal aligns with broader objectives")
        
        if quality_metrics.get("time_bound", 0) < 0.6:
            suggestions.append("Add a specific deadline or timeframe for completion")
        
        if quality_metrics.get("clarity", 0) < 0.6:
            suggestions.append("Improve clarity by using simpler language and clearer structure")
        
        return suggestions
    
    # ==================== 神经网络定义 ====================
    
    class GoalGenerationNetwork(nn.Module):
        """目标生成神经网络"""
        def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=256):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=2, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, vocab_size)  # 双向LSTM
            
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            output = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步
            return output
    
    class GoalClassificationNetwork(nn.Module):
        """目标分类神经网络"""
        def __init__(self, input_dim=256, hidden_dim=128, num_classes=8):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
            self.dropout = nn.Dropout(0.3)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return self.softmax(x)
    
    class TaskDecompositionNetwork(nn.Module):
        """任务分解神经网络"""
        def __init__(self, input_dim=256, hidden_dim=128, max_tasks=10):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, max_tasks * 3)  # 每个任务：类型、时长、依赖
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x.view(-1, 10, 3)  # 重塑为(max_tasks, 3)格式
    
    class PlanningQualityNetwork(nn.Module):
        """规划质量评估神经网络"""
        def __init__(self, input_dim=256, hidden_dim=128):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, 6)  # 6个质量维度
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return self.sigmoid(x)

# 主函数：测试增强器
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建增强器实例
    enhancer = GoalGenerationEnhancer()
    
    # 测试目标生成
    test_description = "Improve system performance by 20% within the next quarter"
    goal = enhancer._generate_goal_from_description(test_description)
    
    print("测试目标生成结果:")
    print(f"原始描述: {test_description}")
    print(f"生成的目标ID: {goal.get('id', 'N/A')}")
    print(f"目标类型: {goal.get('type', 'N/A')}")
    print(f"质量分数: {goal.get('quality_metrics', {}).get('overall_score', 0):.2f}")
    print(f"质量等级: {goal.get('quality_metrics', {}).get('quality_level', 'N/A')}")
    
    # 测试SMART目标生成
    smart_goal = enhancer._generate_smart_goal("Learn Python programming")
    print("\n测试SMART目标生成结果:")
    print(f"原始目标: Learn Python programming")
    print(f"SMART目标描述: {smart_goal.get('smart_goal', {}).get('description', 'N/A')[:100]}...")
    print(f"质量分数: {smart_goal.get('quality_score', 0):.2f}")
    
    print("\n目标生成增强器测试完成")