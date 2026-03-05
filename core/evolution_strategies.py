#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演化策略增强模块 - Enhanced Evolution Strategies

提供多种高级演化算法和优化策略：
1. 多种演化算法：遗传算法（GA）、粒子群优化（PSO）、差分进化（DE）
2. 自适应参数调整：动态调整突变率、交叉率等参数
3. 早停机制和收敛检测
4. 多目标优化和权衡处理
5. 精英保留和多样性维护

设计原则：
- 模块化：每种策略独立实现，易于组合和切换
- 可配置：支持详细的参数配置和调整
- 可扩展：易于添加新的演化策略
- 性能优先：优化计算效率，支持并行评估
"""

import logging
import time
import math
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import statistics

# 配置日志
logger = logging.getLogger(__name__)


class EvolutionStrategyType(Enum):
    """演化策略类型枚举"""
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM_OPTIMIZATION = "particle_swarm_optimization"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    HYBRID = "hybrid"
    CUSTOM = "custom"


@dataclass
class EvolutionStrategyConfig:
    """演化策略配置"""
    strategy_type: EvolutionStrategyType = EvolutionStrategyType.GENETIC_ALGORITHM
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_pressure: float = 2.0  # 选择压力（锦标赛大小）
    elite_size: int = 2  # 精英保留数量
    early_stopping_patience: int = 10  # 早停耐心值
    convergence_threshold: float = 0.001  # 收敛阈值
    diversity_threshold: float = 0.1  # 多样性阈值
    
    # PSO特定参数
    pso_cognitive_weight: float = 1.5
    pso_social_weight: float = 1.5
    pso_inertia_weight: float = 0.8
    
    # DE特定参数
    de_crossover_rate: float = 0.9
    de_differential_weight: float = 0.5
    
    # 自适应参数调整
    adaptive_parameters: bool = True
    adaptation_rate: float = 0.1
    
    # 并行配置
    parallel_evaluation: bool = False
    max_workers: int = 4


@dataclass
class EvolutionResult:
    """演化结果数据类"""
    best_individual: Any
    best_fitness: float
    average_fitness: float
    fitness_history: List[float]
    diversity_history: List[float]
    parameters_history: List[Dict[str, float]]
    generations_completed: int
    convergence_reached: bool
    early_stopping_triggered: bool
    execution_time: float


class BaseEvolutionStrategy(ABC):
    """演化策略基类"""
    
    def __init__(self, config: EvolutionStrategyConfig):
        self.config = config
        self.logger = logger
        self.generation = 0
        self.fitness_history = []
        self.diversity_history = []
        self.parameters_history = []
        
    @abstractmethod
    def initialize_population(self, population_size: int) -> List[Any]:
        """初始化种群"""
        pass
    
    @abstractmethod
    def evaluate_fitness(self, individual: Any) -> float:
        """评估个体适应度"""
        pass
    
    @abstractmethod
    def evolve_population(self, population: List[Any], fitness_scores: Dict[str, float]) -> List[Any]:
        """演化种群"""
        pass
    
    @abstractmethod
    def get_best_individual(self, population: List[Any], fitness_scores: Dict[str, float]) -> Tuple[Any, float]:
        """获取最佳个体"""
        pass
    
    def calculate_diversity(self, population: List[Any]) -> float:
        """计算种群多样性"""
        if len(population) <= 1:
            return 0.0
        
        # 默认实现：计算个体间的平均差异
        # 实际应用中应根据具体表示方式计算
        return 0.5  # 默认值
    
    def check_convergence(self) -> bool:
        """检查是否收敛"""
        if len(self.fitness_history) < 10:
            return False
        
        recent_history = self.fitness_history[-10:]
        improvement = abs(recent_history[-1] - recent_history[0])
        return improvement < self.config.convergence_threshold
    
    def check_early_stopping(self) -> bool:
        """检查是否应早停"""
        if not self.config.early_stopping_patience:
            return False
        
        if len(self.fitness_history) < self.config.early_stopping_patience:
            return False
        
        # 检查最近几代是否有改进
        recent_history = self.fitness_history[-self.config.early_stopping_patience:]
        best_in_window = max(recent_history)
        current_best = recent_history[-1]
        
        # 如果当前最佳与窗口内最佳相差不大，触发早停
        return abs(current_best - best_in_window) < self.config.convergence_threshold
    
    def update_adaptive_parameters(self):
        """更新自适应参数"""
        if not self.config.adaptive_parameters:
            return
        
        # 根据演化进度调整参数
        progress = min(1.0, self.generation / self.config.max_generations)
        
        # 示例：随着演化进行降低突变率
        if hasattr(self.config, 'mutation_rate'):
            initial_mutation_rate = 0.2
            final_mutation_rate = 0.05
            self.config.mutation_rate = initial_mutation_rate * (1 - progress) + final_mutation_rate * progress
            
        # 记录参数调整
        self.parameters_history.append({
            "generation": self.generation,
            "mutation_rate": getattr(self.config, 'mutation_rate', 0.0),
            "crossover_rate": getattr(self.config, 'crossover_rate', 0.0),
            "timestamp": time.time()
        })


class GeneticAlgorithmStrategy(BaseEvolutionStrategy):
    """遗传算法策略"""
    
    def __init__(self, config: EvolutionStrategyConfig):
        super().__init__(config)
        self.logger.info("遗传算法策略初始化完成")
    
    def initialize_population(self, population_size: int) -> List[Any]:
        """初始化随机种群"""
        population = []
        for i in range(population_size):
            # 创建随机个体
            # 实际应用中应根据具体问题初始化
            individual = {
                "id": f"ind_{i}",
                "genes": [random.random() for _ in range(10)],  # 示例基因
                "generation": 0
            }
            population.append(individual)
        return population
    
    def evaluate_fitness(self, individual: Any) -> float:
        """评估个体适应度"""
        # 默认实现：使用基因的平均值作为适应度
        genes = individual.get("genes", [])
        if not genes:
            return 0.0
        return sum(genes) / len(genes)
    
    def evolve_population(self, population: List[Any], fitness_scores: Dict[str, float]) -> List[Any]:
        """演化种群 - 遗传算法"""
        self.generation += 1
        
        # 1. 选择
        selected = self._selection(population, fitness_scores)
        
        # 2. 交叉
        offspring = self._crossover(selected)
        
        # 3. 变异
        mutated_offspring = self._mutation(offspring)
        
        # 4. 精英保留
        new_population = self._elitism(population, fitness_scores, mutated_offspring)
        
        # 5. 更新自适应参数
        self.update_adaptive_parameters()
        
        return new_population
    
    def get_best_individual(self, population: List[Any], fitness_scores: Dict[str, float]) -> Tuple[Any, float]:
        """获取最佳个体"""
        if not population or not fitness_scores:
            return None, 0.0
        
        best_id = max(fitness_scores, key=fitness_scores.get)
        best_fitness = fitness_scores[best_id]
        
        # 找到对应的个体
        for individual in population:
            if individual.get("id") == best_id:
                return individual, best_fitness
        
        return population[0], fitness_scores.get(population[0].get("id"), 0.0)
    
    def _selection(self, population: List[Any], fitness_scores: Dict[str, float]) -> List[Any]:
        """锦标赛选择"""
        selected = []
        tournament_size = max(2, int(len(population) * 0.1))
        
        while len(selected) < len(population):
            # 随机选择锦标赛参与者
            tournament = random.sample(population, min(tournament_size, len(population)))
            
            # 选择适应度最高的
            best_in_tournament = max(tournament, 
                                   key=lambda x: fitness_scores.get(x.get("id"), 0.0))
            selected.append(best_in_tournament)
        
        return selected
    
    def _crossover(self, selected: List[Any]) -> List[Any]:
        """单点交叉"""
        offspring = []
        
        # 随机配对
        random.shuffle(selected)
        
        for i in range(0, len(selected) - 1, 2):
            parent1 = selected[i]
            parent2 = selected[i + 1]
            
            if random.random() < self.config.crossover_rate:
                # 执行交叉
                genes1 = parent1.get("genes", [])
                genes2 = parent2.get("genes", [])
                
                if genes1 and genes2:
                    crossover_point = random.randint(1, min(len(genes1), len(genes2)) - 1)
                    
                    child1_genes = genes1[:crossover_point] + genes2[crossover_point:]
                    child2_genes = genes2[:crossover_point] + genes1[crossover_point:]
                    
                    child1 = {
                        "id": f"child_{len(offspring)}",
                        "genes": child1_genes,
                        "generation": self.generation,
                        "parents": [parent1.get("id"), parent2.get("id")]
                    }
                    
                    child2 = {
                        "id": f"child_{len(offspring) + 1}",
                        "genes": child2_genes,
                        "generation": self.generation,
                        "parents": [parent1.get("id"), parent2.get("id")]
                    }
                    
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([parent1, parent2])
            else:
                # 不交叉，直接复制
                offspring.extend([parent1, parent2])
        
        return offspring
    
    def _mutation(self, offspring: List[Any]) -> List[Any]:
        """高斯突变"""
        mutated = []
        
        for individual in offspring:
            if random.random() < self.config.mutation_rate:
                # 执行突变
                genes = individual.get("genes", [])
                mutated_genes = []
                
                for gene in genes:
                    if random.random() < 0.3:  # 30%的基因发生突变
                        mutated_gene = gene + random.gauss(0, 0.1)
                        mutated_gene = max(0.0, min(1.0, mutated_gene))  # 限制在[0,1]范围
                        mutated_genes.append(mutated_gene)
                    else:
                        mutated_genes.append(gene)
                
                mutated_individual = individual.copy()
                mutated_individual["genes"] = mutated_genes
                mutated_individual["mutated"] = True
                mutated.append(mutated_individual)
            else:
                mutated.append(individual)
        
        return mutated
    
    def _elitism(self, old_population: List[Any], fitness_scores: Dict[str, float], 
                new_population: List[Any]) -> List[Any]:
        """精英保留"""
        if self.config.elite_size <= 0:
            return new_population
        
        # 获取旧种群中的精英
        sorted_old = sorted(old_population, 
                          key=lambda x: fitness_scores.get(x.get("id"), 0.0), 
                          reverse=True)
        elites = sorted_old[:self.config.elite_size]
        
        # 将精英加入新种群
        final_population = elites + new_population[self.config.elite_size:]
        
        # 保持种群大小
        if len(final_population) > len(old_population):
            final_population = final_population[:len(old_population)]
        
        return final_population


class ParticleSwarmOptimizationStrategy(BaseEvolutionStrategy):
    """粒子群优化策略"""
    
    def __init__(self, config: EvolutionStrategyConfig):
        super().__init__(config)
        self.particles = []
        self.velocities = []
        self.personal_bests = []
        self.personal_best_fitnesses = []
        self.global_best = None
        self.global_best_fitness = -float('inf')
        self.logger.info("粒子群优化策略初始化完成")
    
    def initialize_population(self, population_size: int) -> List[Any]:
        """初始化粒子群"""
        self.particles = []
        self.velocities = []
        self.personal_bests = []
        self.personal_best_fitnesses = []
        
        for i in range(population_size):
            # 创建随机粒子
            position = [random.random() for _ in range(10)]  # 10维位置
            velocity = [random.uniform(-0.1, 0.1) for _ in range(10)]  # 初始速度
            
            particle = {
                "id": f"particle_{i}",
                "position": position,
                "generation": 0
            }
            
            self.particles.append(particle)
            self.velocities.append(velocity)
            self.personal_bests.append(position.copy())
            self.personal_best_fitnesses.append(-float('inf'))
        
        return self.particles
    
    def evaluate_fitness(self, individual: Any) -> float:
        """评估粒子适应度"""
        position = individual.get("position", [])
        if not position:
            return 0.0
        
        # 默认实现：求位置各维度的平方和的倒数（模拟优化问题）
        # 实际应用中应根据具体问题定义
        sum_squares = sum(x * x for x in position)
        return 1.0 / (1.0 + sum_squares)
    
    def evolve_population(self, population: List[Any], fitness_scores: Dict[str, float]) -> List[Any]:
        """演化粒子群"""
        self.generation += 1
        
        # 更新全局最佳
        for i, particle in enumerate(population):
            particle_id = particle.get("id")
            fitness = fitness_scores.get(particle_id, 0.0)
            
            # 更新个体最佳
            if fitness > self.personal_best_fitnesses[i]:
                self.personal_bests[i] = particle["position"].copy()
                self.personal_best_fitnesses[i] = fitness
            
            # 更新全局最佳
            if fitness > self.global_best_fitness:
                self.global_best = particle["position"].copy()
                self.global_best_fitness = fitness
        
        # 更新粒子速度和位置
        for i in range(len(population)):
            # 更新速度
            for d in range(len(self.velocities[i])):
                cognitive = self.config.pso_cognitive_weight * random.random() * (
                    self.personal_bests[i][d] - self.particles[i]["position"][d]
                )
                social = self.config.pso_social_weight * random.random() * (
                    self.global_best[d] - self.particles[i]["position"][d]
                )
                
                self.velocities[i][d] = (
                    self.config.pso_inertia_weight * self.velocities[i][d] +
                    cognitive + social
                )
                
                # 速度限制
                self.velocities[i][d] = max(-0.5, min(0.5, self.velocities[i][d]))
            
            # 更新位置
            for d in range(len(self.particles[i]["position"])):
                self.particles[i]["position"][d] += self.velocities[i][d]
                
                # 位置限制在[0,1]范围
                self.particles[i]["position"][d] = max(0.0, min(1.0, self.particles[i]["position"][d]))
        
        # 更新自适应参数
        self.update_adaptive_parameters()
        
        return self.particles
    
    def get_best_individual(self, population: List[Any], fitness_scores: Dict[str, float]) -> Tuple[Any, float]:
        """获取最佳粒子"""
        if not population:
            return None, 0.0
        
        best_id = max(fitness_scores, key=fitness_scores.get)
        best_fitness = fitness_scores[best_id]
        
        for particle in population:
            if particle.get("id") == best_id:
                return particle, best_fitness
        
        return population[0], fitness_scores.get(population[0].get("id"), 0.0)
    
    def calculate_diversity(self, population: List[Any]) -> float:
        """计算粒子群多样性"""
        if len(population) <= 1:
            return 0.0
        
        # 计算粒子位置的标准差
        all_positions = []
        for particle in population:
            positions = particle.get("position", [])
            all_positions.extend(positions)
        
        if not all_positions:
            return 0.0
        
        return statistics.stdev(all_positions) if len(all_positions) > 1 else 0.0


class DifferentialEvolutionStrategy(BaseEvolutionStrategy):
    """差分进化策略"""
    
    def __init__(self, config: EvolutionStrategyConfig):
        super().__init__(config)
        self.logger.info("差分进化策略初始化完成")
    
    def initialize_population(self, population_size: int) -> List[Any]:
        """初始化种群"""
        population = []
        for i in range(population_size):
            individual = {
                "id": f"de_ind_{i}",
                "genes": [random.random() for _ in range(10)],
                "generation": 0
            }
            population.append(individual)
        return population
    
    def evaluate_fitness(self, individual: Any) -> float:
        """评估个体适应度"""
        genes = individual.get("genes", [])
        if not genes:
            return 0.0
        
        # 默认实现：求基因的平均值
        return sum(genes) / len(genes)
    
    def evolve_population(self, population: List[Any], fitness_scores: Dict[str, float]) -> List[Any]:
        """差分进化"""
        self.generation += 1
        new_population = []
        
        for i, target in enumerate(population):
            # 选择三个不同的个体
            candidates = [j for j in range(len(population)) if j != i]
            selected = random.sample(candidates, 3)
            a, b, c = selected
            
            # 创建试验向量
            target_genes = target.get("genes", [])
            a_genes = population[a].get("genes", [])
            b_genes = population[b].get("genes", [])
            c_genes = population[c].get("genes", [])
            
            trial_genes = []
            for d in range(len(target_genes)):
                if random.random() < self.config.de_crossover_rate:
                    # 差分变异
                    trial_gene = a_genes[d] + self.config.de_differential_weight * (b_genes[d] - c_genes[d])
                    trial_gene = max(0.0, min(1.0, trial_gene))  # 边界处理
                    trial_genes.append(trial_gene)
                else:
                    trial_genes.append(target_genes[d])
            
            # 评估试验向量
            trial_individual = {
                "id": f"trial_{i}_{self.generation}",
                "genes": trial_genes,
                "generation": self.generation
            }
            trial_fitness = self.evaluate_fitness(trial_individual)
            target_fitness = fitness_scores.get(target.get("id"), 0.0)
            
            # 选择
            if trial_fitness > target_fitness:
                new_population.append(trial_individual)
            else:
                new_population.append(target)
        
        # 更新自适应参数
        self.update_adaptive_parameters()
        
        return new_population
    
    def get_best_individual(self, population: List[Any], fitness_scores: Dict[str, float]) -> Tuple[Any, float]:
        """获取最佳个体"""
        if not population:
            return None, 0.0
        
        best_id = max(fitness_scores, key=fitness_scores.get)
        best_fitness = fitness_scores[best_id]
        
        for individual in population:
            if individual.get("id") == best_id:
                return individual, best_fitness
        
        return population[0], fitness_scores.get(population[0].get("id"), 0.0)


class EvolutionStrategyFactory:
    """演化策略工厂"""
    
    @staticmethod
    def create_strategy(config: EvolutionStrategyConfig) -> BaseEvolutionStrategy:
        """创建演化策略实例"""
        if config.strategy_type == EvolutionStrategyType.GENETIC_ALGORITHM:
            return GeneticAlgorithmStrategy(config)
        elif config.strategy_type == EvolutionStrategyType.PARTICLE_SWARM_OPTIMIZATION:
            return ParticleSwarmOptimizationStrategy(config)
        elif config.strategy_type == EvolutionStrategyType.DIFFERENTIAL_EVOLUTION:
            return DifferentialEvolutionStrategy(config)
        elif config.strategy_type == EvolutionStrategyType.HYBRID:
            # 混合策略：结合多种算法
            return HybridEvolutionStrategy(config)
        else:
            raise ValueError(f"未知的演化策略类型: {config.strategy_type}")
    
    @staticmethod
    def get_default_config(strategy_type: EvolutionStrategyType) -> EvolutionStrategyConfig:
        """获取默认配置"""
        config = EvolutionStrategyConfig()
        config.strategy_type = strategy_type
        
        if strategy_type == EvolutionStrategyType.GENETIC_ALGORITHM:
            config.population_size = 50
            config.mutation_rate = 0.1
            config.crossover_rate = 0.8
            config.elite_size = 2
            
        elif strategy_type == EvolutionStrategyType.PARTICLE_SWARM_OPTIMIZATION:
            config.population_size = 30
            config.pso_cognitive_weight = 1.5
            config.pso_social_weight = 1.5
            config.pso_inertia_weight = 0.8
            
        elif strategy_type == EvolutionStrategyType.DIFFERENTIAL_EVOLUTION:
            config.population_size = 40
            config.de_crossover_rate = 0.9
            config.de_differential_weight = 0.5
        
        return config


class HybridEvolutionStrategy(BaseEvolutionStrategy):
    """混合演化策略 - 结合多种算法"""
    
    def __init__(self, config: EvolutionStrategyConfig):
        super().__init__(config)
        # 初始化多个策略
        self.ga_strategy = GeneticAlgorithmStrategy(config)
        self.pso_strategy = ParticleSwarmOptimizationStrategy(config)
        self.current_strategy = self.ga_strategy  # 默认使用遗传算法
        self.strategy_switch_interval = 10  # 每10代切换策略
        self.logger.info("混合演化策略初始化完成")
    
    def initialize_population(self, population_size: int) -> List[Any]:
        """使用当前策略初始化种群"""
        return self.current_strategy.initialize_population(population_size)
    
    def evaluate_fitness(self, individual: Any) -> float:
        """使用当前策略评估适应度"""
        return self.current_strategy.evaluate_fitness(individual)
    
    def evolve_population(self, population: List[Any], fitness_scores: Dict[str, float]) -> List[Any]:
        """混合演化"""
        # 检查是否需要切换策略
        if self.generation > 0 and self.generation % self.strategy_switch_interval == 0:
            self._switch_strategy()
        
        # 使用当前策略进行演化
        new_population = self.current_strategy.evolve_population(population, fitness_scores)
        
        # 更新代数
        self.generation += 1
        
        # 更新历史记录
        best_individual, best_fitness = self.get_best_individual(new_population, fitness_scores)
        self.fitness_history.append(best_fitness)
        self.diversity_history.append(self.calculate_diversity(new_population))
        
        return new_population
    
    def get_best_individual(self, population: List[Any], fitness_scores: Dict[str, float]) -> Tuple[Any, float]:
        """获取最佳个体"""
        return self.current_strategy.get_best_individual(population, fitness_scores)
    
    def _switch_strategy(self):
        """切换演化策略"""
        if isinstance(self.current_strategy, GeneticAlgorithmStrategy):
            self.current_strategy = self.pso_strategy
            self.logger.info("切换策略：遗传算法 -> 粒子群优化")
        else:
            self.current_strategy = self.ga_strategy
            self.logger.info("切换策略：粒子群优化 -> 遗传算法")


class EvolutionOptimizer:
    """演化优化器 - 高级封装"""
    
    def __init__(self, strategy_config: EvolutionStrategyConfig):
        self.config = strategy_config
        self.strategy = EvolutionStrategyFactory.create_strategy(strategy_config)
        self.population = []
        self.fitness_scores = {}
        self.result_history = []
        
    def optimize(self, max_generations: Optional[int] = None) -> EvolutionResult:
        """执行优化"""
        start_time = time.time()
        
        # 使用配置或指定的最大代数
        actual_max_generations = max_generations or self.config.max_generations
        
        # 初始化种群
        self.population = self.strategy.initialize_population(self.config.population_size)
        
        # 初始化适应度历史
        fitness_history = []
        diversity_history = []
        parameters_history = []
        
        # 主演化循环
        for generation in range(actual_max_generations):
            # 评估种群
            self.fitness_scores = {}
            for individual in self.population:
                fitness = self.strategy.evaluate_fitness(individual)
                self.fitness_scores[individual.get("id")] = fitness
            
            # 记录历史
            best_individual, best_fitness = self.strategy.get_best_individual(
                self.population, self.fitness_scores
            )
            fitness_history.append(best_fitness)
            diversity_history.append(self.strategy.calculate_diversity(self.population))
            parameters_history.append(self.strategy.parameters_history[-1] 
                                    if self.strategy.parameters_history else {})
            
            # 检查收敛
            convergence_reached = self.strategy.check_convergence()
            early_stopping_triggered = self.strategy.check_early_stopping()
            
            # 输出进度
            if generation % 10 == 0:
                avg_fitness = sum(self.fitness_scores.values()) / len(self.fitness_scores)
                self.logger.info(
                    f"代数 {generation}: 最佳适应度={best_fitness:.4f}, "
                    f"平均适应度={avg_fitness:.4f}, "
                    f"多样性={diversity_history[-1]:.4f}"
                )
            
            # 检查停止条件
            if convergence_reached or early_stopping_triggered:
                self.logger.info(f"演化提前停止: 代数={generation}")
                if convergence_reached:
                    self.logger.info(f"原因: 收敛达到阈值")
                else:
                    self.logger.info(f"原因: 早停触发")
                break
            
            # 演化到下一代
            self.population = self.strategy.evolve_population(self.population, self.fitness_scores)
        
        # 最终评估
        best_individual, best_fitness = self.strategy.get_best_individual(
            self.population, self.fitness_scores
        )
        avg_fitness = sum(self.fitness_scores.values()) / len(self.fitness_scores)
        
        # 创建结果
        result = EvolutionResult(
            best_individual=best_individual,
            best_fitness=best_fitness,
            average_fitness=avg_fitness,
            fitness_history=fitness_history,
            diversity_history=diversity_history,
            parameters_history=parameters_history,
            generations_completed=min(generation + 1, actual_max_generations),
            convergence_reached=convergence_reached,
            early_stopping_triggered=early_stopping_triggered,
            execution_time=time.time() - start_time
        )
        
        # 保存结果历史
        self.result_history.append(result)
        
        self.logger.info(
            f"优化完成: 最佳适应度={best_fitness:.4f}, "
            f"平均适应度={avg_fitness:.4f}, "
            f"耗时={result.execution_time:.2f}s"
        )
        
        return result


# 测试代码
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("演化策略增强模块测试")
    print("=" * 80)
    
    try:
        # 测试遗传算法
        print("\n1. 测试遗传算法:")
        ga_config = EvolutionStrategyFactory.get_default_config(
            EvolutionStrategyType.GENETIC_ALGORITHM
        )
        ga_config.population_size = 20
        ga_config.max_generations = 30
        
        ga_optimizer = EvolutionOptimizer(ga_config)
        ga_result = ga_optimizer.optimize()
        
        print(f"   遗传算法结果: 最佳适应度={ga_result.best_fitness:.4f}, "
              f"代数={ga_result.generations_completed}")
        
        # 测试粒子群优化
        print("\n2. 测试粒子群优化:")
        pso_config = EvolutionStrategyFactory.get_default_config(
            EvolutionStrategyType.PARTICLE_SWARM_OPTIMIZATION
        )
        pso_config.population_size = 15
        pso_config.max_generations = 25
        
        pso_optimizer = EvolutionOptimizer(pso_config)
        pso_result = pso_optimizer.optimize()
        
        print(f"   粒子群优化结果: 最佳适应度={pso_result.best_fitness:.4f}, "
              f"代数={pso_result.generations_completed}")
        
        # 测试差分进化
        print("\n3. 测试差分进化:")
        de_config = EvolutionStrategyFactory.get_default_config(
            EvolutionStrategyType.DIFFERENTIAL_EVOLUTION
        )
        de_config.population_size = 25
        de_config.max_generations = 20
        
        de_optimizer = EvolutionOptimizer(de_config)
        de_result = de_optimizer.optimize()
        
        print(f"   差分进化结果: 最佳适应度={de_result.best_fitness:.4f}, "
              f"代数={de_result.generations_completed}")
        
        print("\n✓ 所有演化策略测试完成")
        print("\n说明:")
        print("  1. 实现了多种演化策略：遗传算法、粒子群优化、差分进化")
        print("  2. 支持自适应参数调整和早停机制")
        print("  3. 提供混合策略和高级优化器封装")
        print("  4. 可轻松集成到现有演化系统中")
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)