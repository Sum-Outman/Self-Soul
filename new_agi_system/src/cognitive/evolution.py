"""
自主演化系统

为统一认知架构实现自主演化能力。
基于原有Self-Soul系统的演化功能，提供架构级自主演化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio
import logging
import time
import json
import random
import copy
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib

logger = logging.getLogger(__name__)


class EvolutionComponentType(Enum):
    """演化组件类型"""
    NETWORK_LAYER = "network_layer"
    ATTENTION_MECHANISM = "attention_mechanism"
    ACTIVATION_FUNCTION = "activation_function"
    CONNECTION_PATTERN = "connection_pattern"
    MODALITY_FUSION = "modality_fusion"
    POOLING_OPERATION = "pooling_operation"
    NORMALIZATION_LAYER = "normalization_layer"
    MEMORY_MECHANISM = "memory_mechanism"
    REASONING_MODULE = "reasoning_module"
    DECISION_MODULE = "decision_module"


class AttentionEvolutionType(Enum):
    """注意力演化类型"""
    STANDARD = "standard"           # 标准注意力
    MULTI_HEAD = "multi_head"       # 多头注意力
    SPARSE = "sparse"               # 稀疏注意力
    LOCAL = "local"                 # 局部注意力
    GLOBAL = "global"               # 全局注意力
    CROSS_MODAL = "cross_modal"     # 跨模态注意力
    HIERARCHICAL = "hierarchical"   # 层次注意力
    ADAPTIVE = "adaptive"           # 自适应注意力


class FusionEvolutionStrategy(Enum):
    """融合演化策略"""
    EARLY_FUSION = "early_fusion"           # 早期融合
    LATE_FUSION = "late_fusion"             # 晚期融合
    HYBRID_FUSION = "hybrid_fusion"         # 混合融合
    ATTENTION_FUSION = "attention_fusion"   # 注意力融合
    TRANSFORMER_FUSION = "transformer_fusion"  # Transformer融合
    ADAPTIVE_FUSION = "adaptive_fusion"     # 自适应融合


@dataclass
class ArchitectureGene:
    """架构基因表示"""
    component_type: EvolutionComponentType
    component_id: str
    parameters: Dict[str, Any]
    performance_score: float = 0.0
    resource_cost: float = 0.0
    complexity_score: float = 0.0
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0
    mutation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "component_type": self.component_type.value,
            "component_id": self.component_id,
            "parameters": self.parameters,
            "performance_score": self.performance_score,
            "resource_cost": self.resource_cost,
            "complexity_score": self.complexity_score,
            "last_used": self.last_used,
            "usage_count": self.usage_count,
            "mutation_count": len(self.mutation_history)
        }


@dataclass
class EvolutionResult:
    """演化结果"""
    success: bool
    evolved_architecture: Dict[str, Any]
    performance_improvement: float
    resource_change: float
    evolution_steps: List[Dict[str, Any]]
    generation: int
    best_gene_id: str
    total_time: float
    error: Optional[str] = None


class ArchitectureEvolutionEngine:
    """架构演化引擎"""
    
    def __init__(self, communication):
        """初始化架构演化引擎"""
        self.communication = communication
        self.initialized = False
        
        # 基因库
        self.gene_pool: Dict[str, ArchitectureGene] = {}
        self.active_genes: List[str] = []
        
        # 演化配置
        self.config = {
            'mutation_rate': 0.15,
            'crossover_rate': 0.25,
            'selection_pressure': 0.6,
            'population_size': 20,
            'max_generations': 50,
            'performance_threshold': 0.85,
            'resource_constraint': 0.8,  # 资源使用率上限
            'complexity_penalty': 0.2,   # 复杂度惩罚系数
            'innovation_reward': 0.3,    # 创新奖励系数
            'stability_bonus': 0.1       # 稳定性奖励
        }
        
        # 演化历史
        self.evolution_history: deque = deque(maxlen=100)
        self.generation = 1
        
        # 性能评估缓存
        self.performance_cache: Dict[str, float] = {}
        
        # 统计信息
        self.evolution_stats = {
            'total_evolutions': 0,
            'successful_evolutions': 0,
            'failed_evolutions': 0,
            'performance_improvement_sum': 0.0,
            'average_improvement': 0.0,
            'best_performance': 0.0,
            'total_computation_time': 0.0,
            'last_evolution_time': 0.0
        }
        
        logger.info("架构演化引擎已初始化")
    
    async def initialize(self):
        """初始化演化引擎"""
        if self.initialized:
            return
        
        logger.info("初始化架构演化引擎...")
        
        # 初始化基础基因库
        await self._initialize_base_gene_pool()
        
        # 加载演化历史（如果有）
        await self._load_evolution_history()
        
        self.initialized = True
        logger.info(f"架构演化引擎初始化完成，基因库大小: {len(self.gene_pool)}")
    
    async def evolve_architecture(self, target_component: str, 
                                 performance_targets: Dict[str, float],
                                 constraints: Dict[str, Any] = None) -> EvolutionResult:
        """
        演化指定组件的架构。
        
        参数:
            target_component: 目标组件名称
            performance_targets: 性能目标字典
            constraints: 约束条件
            
        返回:
            演化结果
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            logger.info(f"开始演化组件: {target_component}")
            
            # 准备演化参数
            actual_constraints = constraints or {}
            
            # 获取当前架构
            current_architecture = await self._get_current_architecture(target_component)
            
            # 生成初始种群
            population = await self._generate_initial_population(
                target_component, current_architecture, performance_targets
            )
            
            # 主演化循环
            best_individual = None
            best_fitness = -float('inf')
            evolution_steps = []
            
            for generation in range(self.config['max_generations']):
                # 评估种群
                fitness_scores = await self._evaluate_population(
                    population, target_component, performance_targets, actual_constraints
                )
                
                # 找到最佳个体
                generation_best_idx = max(fitness_scores, key=fitness_scores.get)
                generation_best_fitness = fitness_scores[generation_best_idx]
                
                if generation_best_fitness > best_fitness:
                    best_fitness = generation_best_fitness
                    best_individual = population[generation_best_idx]
                
                # 记录演化步骤
                evolution_steps.append({
                    'generation': generation + 1,
                    'best_fitness': generation_best_fitness,
                    'population_size': len(population),
                    'average_fitness': sum(fitness_scores.values()) / len(fitness_scores),
                    'diversity': await self._calculate_population_diversity(population)
                })
                
                # 检查收敛条件
                if generation_best_fitness >= self.config['performance_threshold']:
                    logger.info(f"演化在代 {generation + 1} 收敛，适应度: {generation_best_fitness:.4f}")
                    break
                
                # 选择下一代
                selected = await self._select_parents(population, fitness_scores)
                
                # 生成下一代
                new_population = []
                while len(new_population) < self.config['population_size']:
                    if random.random() < self.config['crossover_rate'] and len(selected) >= 2:
                        # 交叉操作
                        parent1, parent2 = random.sample(selected, 2)
                        child = await self._crossover(parent1, parent2)
                    else:
                        # 克隆操作
                        child = copy.deepcopy(random.choice(selected))
                    
                    # 变异操作
                    if random.random() < self.config['mutation_rate']:
                        child = await self._mutate(child, target_component)
                    
                    new_population.append(child)
                
                population = new_population
            
            # 计算演化结果
            total_time = time.time() - start_time
            
            if best_individual:
                # 应用最佳架构
                evolved_architecture = await self._apply_architecture(
                    target_component, best_individual
                )
                
                # 计算性能改进
                current_performance = await self._evaluate_architecture(
                    current_architecture, target_component, performance_targets, actual_constraints
                )
                performance_improvement = best_fitness - current_performance
                
                # 更新基因库
                best_gene_id = await self._update_gene_pool(best_individual, best_fitness)
                
                # 更新统计信息
                self._update_evolution_stats(True, performance_improvement, total_time)
                
                # 保存演化历史
                evolution_record = {
                    'target_component': target_component,
                    'generation': self.generation,
                    'performance_targets': performance_targets,
                    'performance_improvement': performance_improvement,
                    'best_fitness': best_fitness,
                    'total_time': total_time,
                    'timestamp': time.time(),
                    'best_gene_id': best_gene_id
                }
                self.evolution_history.append(evolution_record)
                self.generation += 1
                
                result = EvolutionResult(
                    success=True,
                    evolved_architecture=evolved_architecture,
                    performance_improvement=performance_improvement,
                    resource_change=0.0,  # 简化实现
                    evolution_steps=evolution_steps,
                    generation=self.generation - 1,
                    best_gene_id=best_gene_id,
                    total_time=total_time
                )
                
                logger.info(f"组件 {target_component} 演化成功，性能改进: {performance_improvement:.4f}")
                return result
            else:
                raise ValueError("演化未产生有效结果")
                
        except Exception as e:
            logger.error(f"组件 {target_component} 演化失败: {e}")
            total_time = time.time() - start_time
            self._update_evolution_stats(False, 0.0, total_time)
            
            return EvolutionResult(
                success=False,
                evolved_architecture={},
                performance_improvement=0.0,
                resource_change=0.0,
                evolution_steps=[],
                generation=self.generation,
                best_gene_id="",
                total_time=total_time,
                error=str(e)
            )
    
    async def _initialize_base_gene_pool(self):
        """初始化基础基因库"""
        # 注意力机制基因
        attention_genes = [
            ArchitectureGene(
                component_type=EvolutionComponentType.ATTENTION_MECHANISM,
                component_id="standard_attention",
                parameters={
                    "type": AttentionEvolutionType.STANDARD.value,
                    "dimension": 512,
                    "num_heads": 8,
                    "dropout": 0.1
                },
                performance_score=0.7,
                resource_cost=0.5,
                complexity_score=0.3
            ),
            ArchitectureGene(
                component_type=EvolutionComponentType.ATTENTION_MECHANISM,
                component_id="multi_head_attention",
                parameters={
                    "type": AttentionEvolutionType.MULTI_HEAD.value,
                    "dimension": 512,
                    "num_heads": 8,
                    "dropout": 0.1
                },
                performance_score=0.75,
                resource_cost=0.6,
                complexity_score=0.4
            ),
            ArchitectureGene(
                component_type=EvolutionComponentType.ATTENTION_MECHANISM,
                component_id="hierarchical_attention",
                parameters={
                    "type": AttentionEvolutionType.HIERARCHICAL.value,
                    "dimension": 512,
                    "num_levels": 3,
                    "num_heads": 8,
                    "dropout": 0.1
                },
                performance_score=0.8,
                resource_cost=0.7,
                complexity_score=0.6
            )
        ]
        
        # 融合策略基因
        fusion_genes = [
            ArchitectureGene(
                component_type=EvolutionComponentType.MODALITY_FUSION,
                component_id="attention_fusion",
                parameters={
                    "type": FusionEvolutionStrategy.ATTENTION_FUSION.value,
                    "fusion_dimension": 1024,
                    "attention_heads": 8,
                    "dropout": 0.1
                },
                performance_score=0.75,
                resource_cost=0.6,
                complexity_score=0.5
            ),
            ArchitectureGene(
                component_type=EvolutionComponentType.MODALITY_FUSION,
                component_id="transformer_fusion",
                parameters={
                    "type": FusionEvolutionStrategy.TRANSFORMER_FUSION.value,
                    "fusion_dimension": 1024,
                    "num_layers": 3,
                    "num_heads": 8,
                    "dropout": 0.1
                },
                performance_score=0.8,
                resource_cost=0.7,
                complexity_score=0.7
            ),
            ArchitectureGene(
                component_type=EvolutionComponentType.MODALITY_FUSION,
                component_id="adaptive_fusion",
                parameters={
                    "type": FusionEvolutionStrategy.ADAPTIVE_FUSION.value,
                    "fusion_dimension": 1024,
                    "attention_heads": 8,
                    "adaptation_rate": 0.01,
                    "dropout": 0.1
                },
                performance_score=0.85,
                resource_cost=0.75,
                complexity_score=0.8
            )
        ]
        
        # 添加所有基因到基因库
        for gene in attention_genes + fusion_genes:
            self.gene_pool[gene.component_id] = gene
        
        logger.info(f"基础基因库初始化完成，包含 {len(self.gene_pool)} 个基因")
    
    async def _load_evolution_history(self):
        """加载演化历史"""
        # 简化实现：从文件加载历史（如果有）
        # 这里只是记录日志
        logger.info("演化历史加载功能待实现")
    
    async def _get_current_architecture(self, component_name: str) -> Dict[str, Any]:
        """获取当前组件架构"""
        # 简化实现：返回基础架构
        if component_name == "attention":
            return {
                "type": "standard_attention",
                "dimension": 512,
                "num_heads": 8,
                "dropout": 0.1
            }
        elif component_name == "fusion":
            return {
                "type": "attention_fusion",
                "fusion_dimension": 1024,
                "attention_heads": 8,
                "dropout": 0.1
            }
        else:
            return {
                "type": "generic",
                "parameters": {"size": 512}
            }
    
    async def _generate_initial_population(self, component_name: str, 
                                         base_architecture: Dict[str, Any],
                                         performance_targets: Dict[str, float]) -> List[Dict[str, Any]]:
        """生成初始种群"""
        population = []
        
        # 基于基因库生成个体
        relevant_genes = [
            gene for gene in self.gene_pool.values()
            if (component_name == "attention" and gene.component_type == EvolutionComponentType.ATTENTION_MECHANISM) or
               (component_name == "fusion" and gene.component_type == EvolutionComponentType.MODALITY_FUSION)
        ]
        
        if not relevant_genes:
            # 如果没有相关基因，使用基础架构
            relevant_genes = [ArchitectureGene(
                component_type=EvolutionComponentType.NETWORK_LAYER,
                component_id=f"base_{component_name}",
                parameters=base_architecture,
                performance_score=0.5,
                resource_cost=0.5,
                complexity_score=0.5
            )]
        
        # 生成种群
        for i in range(self.config['population_size']):
            # 随机选择或变异基因
            if random.random() < 0.7 and relevant_genes:
                base_gene = random.choice(relevant_genes)
                individual = copy.deepcopy(base_gene.parameters)
                
                # 应用随机变异
                if random.random() < 0.3:
                    individual = await self._mutate_parameters(individual, component_name)
            else:
                # 使用基础架构
                individual = copy.deepcopy(base_architecture)
            
            population.append(individual)
        
        return population
    
    async def _evaluate_population(self, population: List[Dict[str, Any]],
                                 component_name: str,
                                 performance_targets: Dict[str, float],
                                 constraints: Dict[str, Any]) -> Dict[int, float]:
        """评估种群适应度"""
        fitness_scores = {}
        
        for idx, individual in enumerate(population):
            # 计算适应度
            performance_score = await self._evaluate_architecture(
                individual, component_name, performance_targets, constraints
            )
            
            # 计算资源成本
            resource_cost = await self._calculate_resource_cost(individual, component_name)
            
            # 计算复杂度惩罚
            complexity_penalty = await self._calculate_complexity_penalty(individual, component_name)
            
            # 总适应度 = 性能 - 资源成本 * 权重 - 复杂度惩罚
            resource_weight = self.config.get('resource_constraint', 0.8)
            complexity_weight = self.config.get('complexity_penalty', 0.2)
            
            fitness = (performance_score * 0.7 + 
                      (1.0 - resource_cost) * 0.2 * resource_weight - 
                      complexity_penalty * complexity_weight)
            
            # 添加创新奖励（如果是个新结构）
            if await self._is_novel_architecture(individual, component_name):
                fitness += self.config.get('innovation_reward', 0.3)
            
            fitness_scores[idx] = max(0.0, min(1.0, fitness))
        
        return fitness_scores
    
    async def _evaluate_architecture(self, architecture: Dict[str, Any],
                                   component_name: str,
                                   performance_targets: Dict[str, float],
                                   constraints: Dict[str, Any]) -> float:
        """评估架构性能"""
        # 简化实现：基于启发式规则评估
        base_score = 0.5
        
        # 根据组件类型调整
        if component_name == "attention":
            if architecture.get("type") == "hierarchical_attention":
                base_score += 0.2
            elif architecture.get("type") == "multi_head_attention":
                base_score += 0.15
            elif architecture.get("type") == "adaptive_attention":
                base_score += 0.25
        
        elif component_name == "fusion":
            if architecture.get("type") == "adaptive_fusion":
                base_score += 0.2
            elif architecture.get("type") == "transformer_fusion":
                base_score += 0.15
            elif architecture.get("type") == "attention_fusion":
                base_score += 0.1
        
        # 添加随机噪声模拟真实评估
        noise = random.uniform(-0.1, 0.1)
        final_score = base_score + noise
        
        return max(0.0, min(1.0, final_score))
    
    async def _calculate_resource_cost(self, architecture: Dict[str, Any], 
                                     component_name: str) -> float:
        """计算资源成本"""
        # 简化实现：基于参数数量估算
        param_count = 0
        
        if component_name == "attention":
            dimension = architecture.get("dimension", 512)
            num_heads = architecture.get("num_heads", 8)
            param_count = dimension * dimension * 3  # Q, K, V 投影
            
            if architecture.get("type") == "hierarchical_attention":
                num_levels = architecture.get("num_levels", 3)
                param_count *= num_levels
        elif component_name == "fusion":
            fusion_dim = architecture.get("fusion_dimension", 1024)
            num_heads = architecture.get("attention_heads", 8)
            param_count = fusion_dim * fusion_dim * num_heads
        
        # 归一化到 [0, 1]
        max_params = 1000000  # 假设最大100万参数
        cost = min(1.0, param_count / max_params)
        
        return cost
    
    async def _calculate_complexity_penalty(self, architecture: Dict[str, Any],
                                          component_name: str) -> float:
        """计算复杂度惩罚"""
        # 简化实现：基于架构复杂性
        complexity = 0.3  # 基础复杂度
        
        if component_name == "attention":
            att_type = architecture.get("type", "standard")
            if att_type == "hierarchical":
                complexity += 0.3
            elif att_type == "adaptive":
                complexity += 0.4
            elif att_type == "multi_head":
                complexity += 0.2
        
        elif component_name == "fusion":
            fusion_type = architecture.get("type", "attention_fusion")
            if fusion_type == "transformer_fusion":
                complexity += 0.3
            elif fusion_type == "adaptive_fusion":
                complexity += 0.4
            elif fusion_type == "hybrid_fusion":
                complexity += 0.25
        
        return min(1.0, complexity)
    
    async def _is_novel_architecture(self, architecture: Dict[str, Any],
                                   component_name: str) -> bool:
        """判断是否为新颖架构"""
        # 简化实现：检查是否在基因库中
        arch_hash = self._hash_architecture(architecture)
        
        for gene in self.gene_pool.values():
            gene_hash = self._hash_architecture(gene.parameters)
            if arch_hash == gene_hash:
                return False
        
        return True
    
    async def _calculate_population_diversity(self, population: List[Dict[str, Any]]) -> float:
        """计算种群多样性"""
        if len(population) <= 1:
            return 0.0
        
        # 计算架构哈希值
        hashes = [self._hash_architecture(ind) for ind in population]
        
        # 计算唯一哈希比例
        unique_hashes = len(set(hashes))
        diversity = unique_hashes / len(population)
        
        return diversity
    
    async def _select_parents(self, population: List[Dict[str, Any]],
                            fitness_scores: Dict[int, float]) -> List[Dict[str, Any]]:
        """选择父代"""
        # 锦标赛选择
        tournament_size = max(2, len(population) // 4)
        selected = []
        
        while len(selected) < len(population) // 2:  # 选择一半作为父代
            # 随机选择锦标赛参与者
            tournament_indices = random.sample(list(fitness_scores.keys()), 
                                              min(tournament_size, len(fitness_scores)))
            
            # 选择适应度最高的
            best_idx = max(tournament_indices, key=lambda idx: fitness_scores[idx])
            selected.append(population[best_idx])
        
        return selected
    
    async def _crossover(self, parent1: Dict[str, Any], 
                        parent2: Dict[str, Any]) -> Dict[str, Any]:
        """交叉操作"""
        # 均匀交叉：随机选择每个参数来自哪个父代
        child = {}
        
        all_keys = set(parent1.keys()) | set(parent2.keys())
        
        for key in all_keys:
            if key in parent1 and key in parent2:
                # 两个父代都有这个参数，随机选择
                if random.random() < 0.5:
                    child[key] = parent1[key]
                else:
                    child[key] = parent2[key]
            elif key in parent1:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        
        return child
    
    async def _mutate(self, individual: Dict[str, Any], 
                     component_name: str) -> Dict[str, Any]:
        """变异操作"""
        mutated = copy.deepcopy(individual)
        
        # 随机选择一个参数进行变异
        if mutated:
            param_to_mutate = random.choice(list(mutated.keys()))
            
            if isinstance(mutated[param_to_mutate], (int, float)):
                # 数值参数：添加随机噪声
                if isinstance(mutated[param_to_mutate], int):
                    mutation = random.randint(-2, 2)
                    mutated[param_to_mutate] = max(1, mutated[param_to_mutate] + mutation)
                else:
                    mutation = random.uniform(-0.2, 0.2)
                    mutated[param_to_mutate] = max(0.01, mutated[param_to_mutate] + mutation)
            elif isinstance(mutated[param_to_mutate], str) and param_to_mutate == "type":
                # 类型参数：随机切换到其他类型
                if component_name == "attention":
                    all_types = [t.value for t in AttentionEvolutionType]
                    mutated[param_to_mutate] = random.choice(all_types)
                elif component_name == "fusion":
                    all_types = [t.value for t in FusionEvolutionStrategy]
                    mutated[param_to_mutate] = random.choice(all_types)
        
        return mutated
    
    async def _mutate_parameters(self, individual: Dict[str, Any],
                               component_name: str) -> Dict[str, Any]:
        """变异参数"""
        return await self._mutate(individual, component_name)
    
    async def _apply_architecture(self, component_name: str,
                                architecture: Dict[str, Any]) -> Dict[str, Any]:
        """应用架构到组件"""
        # 简化实现：返回架构信息
        # 实际实现中应该更新对应组件的实际参数
        logger.info(f"应用架构到组件 {component_name}: {architecture}")
        
        # 在实际系统中，这里应该调用通信系统更新组件配置
        if self.communication:
            try:
                # 尝试通过通信系统更新组件
                update_message = {
                    "component": component_name,
                    "action": "update_architecture",
                    "architecture": architecture,
                    "timestamp": time.time()
                }
                
                # 发送更新消息
                await self.communication.send_message("architecture_updates", update_message)
                logger.info(f"已发送架构更新消息给 {component_name}")
            except Exception as e:
                logger.warning(f"通过通信系统更新架构失败: {e}")
        
        return architecture
    
    async def _update_gene_pool(self, architecture: Dict[str, Any],
                              fitness: float) -> str:
        """更新基因库"""
        # 为架构生成唯一ID
        arch_hash = self._hash_architecture(architecture)
        gene_id = f"gene_{arch_hash[:8]}"
        
        # 确定组件类型（简化实现）
        component_type = EvolutionComponentType.NETWORK_LAYER
        if "type" in architecture:
            att_type = architecture["type"]
            if any(att_type == t.value for t in AttentionEvolutionType):
                component_type = EvolutionComponentType.ATTENTION_MECHANISM
            elif any(att_type == t.value for t in FusionEvolutionStrategy):
                component_type = EvolutionComponentType.MODALITY_FUSION
        
        # 创建或更新基因
        if gene_id in self.gene_pool:
            # 更新现有基因
            gene = self.gene_pool[gene_id]
            gene.performance_score = (gene.performance_score + fitness) / 2
            gene.usage_count += 1
            gene.last_used = time.time()
            
            # 记录变异历史
            gene.mutation_history.append({
                "timestamp": time.time(),
                "previous_score": gene.performance_score,
                "new_score": fitness,
                "fitness_change": fitness - gene.performance_score
            })
        else:
            # 创建新基因
            gene = ArchitectureGene(
                component_type=component_type,
                component_id=gene_id,
                parameters=architecture,
                performance_score=fitness,
                resource_cost=await self._calculate_resource_cost(architecture, "generic"),
                complexity_score=await self._calculate_complexity_penalty(architecture, "generic"),
                last_used=time.time(),
                usage_count=1
            )
            self.gene_pool[gene_id] = gene
        
        self.active_genes.append(gene_id)
        
        # 限制活跃基因数量
        if len(self.active_genes) > 50:
            # 移除最久未使用的基因
            oldest_gene = min(self.active_genes, 
                            key=lambda gid: self.gene_pool[gid].last_used)
            self.active_genes.remove(oldest_gene)
        
        return gene_id
    
    def _update_evolution_stats(self, success: bool, 
                              performance_improvement: float,
                              computation_time: float):
        """更新演化统计信息"""
        self.evolution_stats['total_evolutions'] += 1
        self.evolution_stats['total_computation_time'] += computation_time
        
        if success:
            self.evolution_stats['successful_evolutions'] += 1
            self.evolution_stats['performance_improvement_sum'] += performance_improvement
            
            # 更新最佳性能
            if performance_improvement > self.evolution_stats['best_performance']:
                self.evolution_stats['best_performance'] = performance_improvement
        else:
            self.evolution_stats['failed_evolutions'] += 1
        
        # 计算平均改进
        if self.evolution_stats['successful_evolutions'] > 0:
            self.evolution_stats['average_improvement'] = (
                self.evolution_stats['performance_improvement_sum'] / 
                self.evolution_stats['successful_evolutions']
            )
        
        self.evolution_stats['last_evolution_time'] = computation_time
    
    def _hash_architecture(self, architecture: Dict[str, Any]) -> str:
        """计算架构哈希值"""
        # 将架构转换为可哈希字符串
        arch_str = json.dumps(architecture, sort_keys=True)
        return hashlib.md5(arch_str.encode()).hexdigest()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取演化统计信息"""
        return {
            'evolution_stats': self.evolution_stats,
            'gene_pool_size': len(self.gene_pool),
            'active_genes': len(self.active_genes),
            'generation': self.generation,
            'evolution_history_size': len(self.evolution_history),
            'config': self.config
        }
    
    def get_gene_pool_info(self) -> List[Dict[str, Any]]:
        """获取基因库信息"""
        genes_info = []
        
        for gene_id, gene in self.gene_pool.items():
            if gene_id in self.active_genes:
                gene_info = gene.to_dict()
                gene_info['active'] = True
                genes_info.append(gene_info)
        
        # 按性能分数排序
        genes_info.sort(key=lambda x: x['performance_score'], reverse=True)
        
        return genes_info[:20]  # 返回前20个基因


class AutonomousEvolutionSystem:
    """自主演化系统（主类）"""
    
    def __init__(self, communication):
        """初始化自主演化系统"""
        self.communication = communication
        
        # 核心引擎
        self.architecture_engine = ArchitectureEvolutionEngine(communication)
        
        # 演化状态
        self.evolution_status = {
            'initialized': False,
            'active_evolutions': 0,
            'total_successful_evolutions': 0,
            'system_health': 'healthy'
        }
        
        # 演化任务队列
        self.evolution_queue = []
        self.max_concurrent_evolutions = 2
        
        logger.info("自主演化系统已初始化")
    
    async def initialize(self):
        """初始化自主演化系统"""
        if self.evolution_status['initialized']:
            return
        
        # 初始化架构演化引擎
        await self.architecture_engine.initialize()
        
        self.evolution_status['initialized'] = True
        logger.info("自主演化系统初始化完成")
    
    async def request_evolution(self, component: str, 
                               performance_targets: Dict[str, float],
                               priority: str = "normal") -> Dict[str, Any]:
        """
        请求演化任务。
        
        参数:
            component: 要演化的组件
            performance_targets: 性能目标
            priority: 优先级 (low, normal, high)
            
        返回:
            演化任务信息
        """
        if not self.evolution_status['initialized']:
            await self.initialize()
        
        # 创建演化任务
        evolution_id = f"evo_{int(time.time())}_{random.randint(1000, 9999)}"
        
        task = {
            'id': evolution_id,
            'component': component,
            'performance_targets': performance_targets,
            'priority': priority,
            'status': 'pending',
            'created_time': time.time(),
            'start_time': None,
            'completion_time': None,
            'result': None
        }
        
        # 添加到队列
        self.evolution_queue.append(task)
        
        # 检查是否可以立即开始
        if self.evolution_status['active_evolutions'] < self.max_concurrent_evolutions:
            # 立即开始执行
            asyncio.create_task(self._execute_evolution(task))
            task['status'] = 'running'
            task['start_time'] = time.time()
            self.evolution_status['active_evolutions'] += 1
            
            response = {
                'evolution_id': evolution_id,
                'status': 'started',
                'message': '演化任务已开始执行',
                'queue_position': 0
            }
        else:
            # 排队等待
            queue_position = len([t for t in self.evolution_queue 
                                if t['status'] == 'pending'])
            response = {
                'evolution_id': evolution_id,
                'status': 'queued',
                'message': '演化任务已加入队列',
                'queue_position': queue_position
            }
        
        return response
    
    async def _execute_evolution(self, task: Dict[str, Any]):
        """执行演化任务"""
        try:
            logger.info(f"开始执行演化任务: {task['id']}")
            
            # 执行架构演化
            result = await self.architecture_engine.evolve_architecture(
                task['component'],
                task['performance_targets']
            )
            
            # 更新任务状态
            task['status'] = 'completed' if result.success else 'failed'
            task['completion_time'] = time.time()
            task['result'] = {
                'success': result.success,
                'performance_improvement': result.performance_improvement,
                'generation': result.generation,
                'total_time': result.total_time,
                'error': result.error
            }
            
            if result.success:
                self.evolution_status['total_successful_evolutions'] += 1
                logger.info(f"演化任务 {task['id']} 成功完成")
            else:
                logger.warning(f"演化任务 {task['id']} 失败: {result.error}")
                
        except Exception as e:
            logger.error(f"演化任务 {task['id']} 执行异常: {e}")
            task['status'] = 'failed'
            task['completion_time'] = time.time()
            task['result'] = {
                'success': False,
                'error': str(e)
            }
        finally:
            # 减少活跃演化计数
            self.evolution_status['active_evolutions'] -= 1
            
            # 检查队列中是否有等待的任务
            await self._process_next_evolution()
    
    async def _process_next_evolution(self):
        """处理下一个等待的演化任务"""
        # 找到第一个等待中的任务
        for task in self.evolution_queue:
            if task['status'] == 'pending':
                # 检查是否有可用槽位
                if self.evolution_status['active_evolutions'] < self.max_concurrent_evolutions:
                    # 开始执行
                    asyncio.create_task(self._execute_evolution(task))
                    task['status'] = 'running'
                    task['start_time'] = time.time()
                    self.evolution_status['active_evolutions'] += 1
                break
    
    async def get_evolution_status(self, evolution_id: str) -> Dict[str, Any]:
        """获取演化任务状态"""
        # 查找任务
        for task in self.evolution_queue:
            if task['id'] == evolution_id:
                return {
                    'evolution_id': evolution_id,
                    'status': task['status'],
                    'component': task['component'],
                    'created_time': task['created_time'],
                    'start_time': task['start_time'],
                    'completion_time': task['completion_time'],
                    'result': task.get('result'),
                    'queue_position': self._get_queue_position(evolution_id)
                }
        
        raise ValueError(f"未找到演化任务: {evolution_id}")
    
    def _get_queue_position(self, evolution_id: str) -> int:
        """获取队列位置"""
        pending_tasks = [t for t in self.evolution_queue 
                        if t['status'] == 'pending']
        
        for i, task in enumerate(pending_tasks):
            if task['id'] == evolution_id:
                return i + 1
        
        return 0  # 不在队列中或正在执行
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        # 收集架构引擎统计信息
        arch_stats = self.architecture_engine.get_statistics()
        
        # 收集任务队列信息
        queue_stats = {
            'total_tasks': len(self.evolution_queue),
            'pending_tasks': len([t for t in self.evolution_queue 
                                 if t['status'] == 'pending']),
            'running_tasks': len([t for t in self.evolution_queue 
                                 if t['status'] == 'running']),
            'completed_tasks': len([t for t in self.evolution_queue 
                                   if t['status'] == 'completed']),
            'failed_tasks': len([t for t in self.evolution_queue 
                                if t['status'] == 'failed'])
        }
        
        return {
            'system_status': self.evolution_status,
            'architecture_engine': arch_stats,
            'queue_stats': queue_stats,
            'timestamp': time.time()
        }
    
    def get_gene_pool(self) -> List[Dict[str, Any]]:
        """获取基因库信息"""
        return self.architecture_engine.get_gene_pool_info()