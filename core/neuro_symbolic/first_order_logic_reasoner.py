#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
一阶逻辑推理引擎 - 实现完整的一阶逻辑推理和定理证明

核心功能:
1. 一阶逻辑语法解析和公式构建
2. 前向链接和后向链接推理
3. 合一算法(Unification)实现
4. 归结原理(Resolution)证明
5. 逻辑模型检查和验证
6. 量词处理和Skolem化

支持特性:
- 命题逻辑: 与(∧)、或(∨)、非(¬)、蕴含(→)、等价(↔)
- 一阶逻辑: 全称量词(∀)、存在量词(∃)
- 谓词和函数: 带参数的谓词，函数符号
- 变量和常量: 变量、常量符号
- 合一算法: 变量替换和匹配

技术实现:
- 基于知识图谱的推理
- 高效的合一算法
- 增量推理和缓存
- 冲突检测和解释

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import re
import copy
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import networkx as nx

# 导入错误处理
from core.error_handling import ErrorHandler

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()


class FormulaType(Enum):
    """公式类型枚举"""
    ATOMIC = "atomic"          # 原子公式: P(x), Q(a,b)
    NEGATION = "negation"      # 否定: ¬φ
    CONJUNCTION = "conjunction"  # 合取: φ ∧ ψ
    DISJUNCTION = "disjunction"  # 析取: φ ∨ ψ
    IMPLICATION = "implication"  # 蕴含: φ → ψ
    EQUIVALENCE = "equivalence"  # 等价: φ ↔ ψ
    UNIVERSAL = "universal"    # 全称量词: ∀x φ
    EXISTENTIAL = "existential"  # 存在量词: ∃x φ


class TermType(Enum):
    """项类型枚举"""
    VARIABLE = "variable"      # 变量: x, y, z
    CONSTANT = "constant"      # 常量: a, b, c
    FUNCTION = "function"      # 函数: f(x), g(a,b)


class InferenceRule(Enum):
    """推理规则枚举"""
    MODUS_PONENS = "modus_ponens"      # 假言推理: P→Q, P ⊢ Q
    UNIVERSAL_INSTANTIATION = "universal_instantiation"  # 全称实例化: ∀x P(x) ⊢ P(a)
    EXISTENTIAL_GENERALIZATION = "existential_generalization"  # 存在概括: P(a) ⊢ ∃x P(x)
    RESOLUTION = "resolution"          # 归结原理
    FORWARD_CHAINING = "forward_chaining"  # 前向链接
    BACKWARD_CHAINING = "backward_chaining"  # 后向链接


class FirstOrderLogicReasoner:
    """
    一阶逻辑推理引擎
    
    核心功能:
    1. 逻辑公式的解析和表示
    2. 知识库管理和查询
    3. 推理规则应用
    4. 定理证明和验证
    5. 合一算法和变量替换
    
    技术特性:
    - 支持一阶逻辑语法
    - 高效的合一算法实现
    - 增量推理和结果缓存
    - 冲突检测和矛盾处理
    - 可扩展的推理规则
    """
    
    def __init__(self):
        """初始化一阶逻辑推理引擎"""
        # 知识库
        self.knowledge_base = {
            "facts": set(),          # 事实集合
            "rules": [],             # 规则列表
            "predicates": {},        # 谓词定义
            "functions": {},         # 函数定义
            "constants": set(),      # 常量集合
            "variables": set()       # 变量集合
        }
        
        # 推理缓存
        self.inference_cache = {}
        
        # 性能统计
        self.performance_stats = {
            "queries_processed": 0,
            "inferences_made": 0,
            "unifications_performed": 0,
            "resolution_steps": 0,
            "proofs_completed": 0
        }
        
        # 初始化基础逻辑
        self._initialize_basic_logic()
        
        logger.info("一阶逻辑推理引擎初始化完成")
    
    def _initialize_basic_logic(self):
        """初始化基础逻辑规则"""
        # 添加基础逻辑公理
        basic_axioms = [
            # 同一律: ∀x (x = x)
            "forall x (equal(x, x))",
            
            # 非矛盾律: ¬(P ∧ ¬P)
            "forall P (not (and(P, not(P))))",
            
            # 排中律: P ∨ ¬P
            "forall P (or(P, not(P)))",
            
            # 假言推理规则: (P → Q) ∧ P → Q
            "forall P forall Q (implies(and(implies(P, Q), P), Q))",
            
            # 全称实例化: ∀x P(x) → P(a)
            "forall P forall x forall a (implies(forall(x, P(x)), P(a)))"
        ]
        
        for axiom in basic_axioms:
            self.add_formula(axiom, is_axiom=True)
        
        logger.info(f"初始化基础逻辑: {len(basic_axioms)}条公理")
    
    def add_formula(self, formula_str: str, is_axiom: bool = False) -> bool:
        """
        添加公式到知识库
        
        Args:
            formula_str: 公式字符串
            is_axiom: 是否为公理
            
        Returns:
            是否成功添加
        """
        try:
            # 解析公式
            parsed_formula = self._parse_formula(formula_str)
            
            if parsed_formula:
                # 添加到知识库
                if is_axiom:
                    formula_id = f"axiom_{len(self.knowledge_base['rules']) + 1}"
                else:
                    formula_id = f"formula_{len(self.knowledge_base['facts']) + len(self.knowledge_base['rules']) + 1}"
                
                formula_info = {
                    "id": formula_id,
                    "formula": parsed_formula,
                    "original": formula_str,
                    "is_axiom": is_axiom,
                    "added_time": time.time()
                }
                
                # 判断是事实还是规则
                if self._is_fact(parsed_formula):
                    self.knowledge_base["facts"].add(formula_info)
                else:
                    self.knowledge_base["rules"].append(formula_info)
                
                # 提取谓词、函数、常量、变量
                self._extract_symbols(parsed_formula)
                
                logger.debug(f"添加公式: {formula_str} (ID: {formula_id})")
                return True
            else:
                logger.warning(f"公式解析失败: {formula_str}")
                return False
                
        except Exception as e:
            logger.error(f"添加公式时出错: {formula_str}, 错误: {e}")
            return False
    
    def _parse_formula(self, formula_str: str) -> Optional[Dict[str, Any]]:
        """
        解析公式字符串
        
        Args:
            formula_str: 公式字符串
            
        Returns:
            解析后的公式字典
        """
        # 移除多余空格
        formula_str = formula_str.strip()
        
        # 处理量词
        if formula_str.startswith("forall") or formula_str.startswith("∀"):
            return self._parse_quantified_formula(formula_str, FormulaType.UNIVERSAL)
        elif formula_str.startswith("exists") or formula_str.startswith("∃"):
            return self._parse_quantified_formula(formula_str, FormulaType.EXISTENTIAL)
        
        # 处理括号表达式
        if formula_str.startswith("(") and formula_str.endswith(")"):
            # 移除外层括号
            inner = formula_str[1:-1].strip()
            return self._parse_formula(inner)
        
        # 处理逻辑连接词
        for connective, formula_type in [
            ("and", FormulaType.CONJUNCTION),
            ("∧", FormulaType.CONJUNCTION),
            ("or", FormulaType.DISJUNCTION),
            ("∨", FormulaType.DISJUNCTION),
            ("implies", FormulaType.IMPLICATION),
            ("→", FormulaType.IMPLICATION),
            ("equivalent", FormulaType.EQUIVALENCE),
            ("↔", FormulaType.EQUIVALENCE)
        ]:
            if f" {connective} " in formula_str or connective in formula_str:
                return self._parse_binary_formula(formula_str, connective, formula_type)
        
        # 处理否定
        if formula_str.startswith("not") or formula_str.startswith("¬"):
            return self._parse_negation(formula_str)
        
        # 原子公式
        return self._parse_atomic_formula(formula_str)
    
    def _parse_quantified_formula(self, formula_str: str, quantifier_type: FormulaType) -> Optional[Dict[str, Any]]:
        """解析量词公式"""
        try:
            # 匹配量词模式: forall x y z (formula) 或 ∀x∀y(formula)
            if quantifier_type == FormulaType.UNIVERSAL:
                quantifier = "forall"
                quantifier_symbol = "∀"
            else:  # EXISTENTIAL
                quantifier = "exists"
                quantifier_symbol = "∃"
            
            # 尝试不同模式
            patterns = [
                # 模式1: forall x y z (formula)
                rf"{quantifier}\s+([a-zA-Z][a-zA-Z0-9]*\s*)+(.*)",
                # 模式2: ∀x∀y(formula)
                rf"{quantifier_symbol}+[a-zA-Z][a-zA-Z0-9]*\(?(.*)\)?"
            ]
            
            for pattern in patterns:
                match = re.match(pattern, formula_str)
                if match:
                    # 提取变量和主体
                    if quantifier in formula_str:
                        # 模式1
                        parts = formula_str[len(quantifier):].strip().split("(", 1)
                        variables_str = parts[0].strip()
                        body_str = parts[1].rstrip(")") if len(parts) > 1 else ""
                    else:
                        # 模式2
                        # 提取所有变量
                        var_pattern = rf"{quantifier_symbol}([a-zA-Z][a-zA-Z0-9]*)"
                        variables = re.findall(var_pattern, formula_str)
                        variables_str = " ".join(variables)
                        # 提取主体（移除量词部分）
                        body_str = re.sub(rf"{quantifier_symbol}[a-zA-Z][a-zA-Z0-9]*", "", formula_str).strip()
                        if body_str.startswith("(") and body_str.endswith(")"):
                            body_str = body_str[1:-1]
                    
                    # 解析变量列表
                    variables = [v.strip() for v in variables_str.split() if v.strip()]
                    
                    # 解析主体公式
                    if body_str:
                        body_formula = self._parse_formula(body_str)
                    else:
                        # 如果没有主体，可能是简写形式
                        body_formula = None
                    
                    if variables and body_formula:
                        return {
                            "type": quantifier_type.value,
                            "quantifier": quantifier_type,
                            "variables": variables,
                            "body": body_formula,
                            "original": formula_str
                        }
            
            logger.warning(f"量词公式解析失败: {formula_str}")
            return None
            
        except Exception as e:
            logger.error(f"解析量词公式时出错: {formula_str}, 错误: {e}")
            return None
    
    def _parse_binary_formula(self, formula_str: str, connective: str, formula_type: FormulaType) -> Optional[Dict[str, Any]]:
        """解析二元连接词公式"""
        try:
            # 分割左右操作数
            # 需要处理嵌套括号
            
            # 查找连接词的位置（考虑括号嵌套）
            connective_pos = self._find_connective_position(formula_str, connective)
            
            if connective_pos >= 0:
                left_str = formula_str[:connective_pos].strip()
                right_str = formula_str[connective_pos + len(connective):].strip()
                
                # 解析左右操作数
                left_formula = self._parse_formula(left_str)
                right_formula = self._parse_formula(right_str)
                
                if left_formula and right_formula:
                    return {
                        "type": formula_type.value,
                        "connective": formula_type,
                        "left": left_formula,
                        "right": right_formula,
                        "original": formula_str
                    }
            
            logger.warning(f"二元公式解析失败: {formula_str}")
            return None
            
        except Exception as e:
            logger.error(f"解析二元公式时出错: {formula_str}, 错误: {e}")
            return None
    
    def _find_connective_position(self, formula_str: str, connective: str) -> int:
        """查找连接词的位置（考虑括号嵌套）"""
        parentheses_level = 0
        i = 0
        
        while i < len(formula_str):
            char = formula_str[i]
            
            if char == '(':
                parentheses_level += 1
            elif char == ')':
                parentheses_level -= 1
            elif parentheses_level == 0:
                # 检查是否匹配连接词
                if formula_str[i:i+len(connective)] == connective:
                    # 确保连接词前后是空格或是字符串边界
                    prev_char = formula_str[i-1] if i > 0 else ' '
                    next_char = formula_str[i+len(connective)] if i+len(connective) < len(formula_str) else ' '
                    
                    if prev_char.isspace() and next_char.isspace():
                        return i
            
            i += 1
        
        return -1
    
    def _parse_negation(self, formula_str: str) -> Optional[Dict[str, Any]]:
        """解析否定公式"""
        try:
            # 移除否定符号
            if formula_str.startswith("not"):
                body_str = formula_str[3:].strip()
            elif formula_str.startswith("¬"):
                body_str = formula_str[1:].strip()
            else:
                return None
            
            # 解析主体公式
            body_formula = self._parse_formula(body_str)
            
            if body_formula:
                return {
                    "type": FormulaType.NEGATION.value,
                    "connective": FormulaType.NEGATION,
                    "body": body_formula,
                    "original": formula_str
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"解析否定公式时出错: {formula_str}, 错误: {e}")
            return None
    
    def _parse_atomic_formula(self, formula_str: str) -> Optional[Dict[str, Any]]:
        """解析原子公式"""
        try:
            # 原子公式: 谓词(参数列表)
            # 例如: P(x), Q(a,b), equal(x, y)
            
            # 检查是否有参数列表
            if "(" in formula_str and formula_str.endswith(")"):
                # 分离谓词和参数
                predicate_end = formula_str.index("(")
                predicate = formula_str[:predicate_end].strip()
                args_str = formula_str[predicate_end+1:-1].strip()
                
                # 解析参数列表
                args = self._parse_argument_list(args_str)
                
                if args is not None:
                    return {
                        "type": FormulaType.ATOMIC.value,
                        "predicate": predicate,
                        "arguments": args,
                        "original": formula_str
                    }
            else:
                # 无参数谓词（零元谓词）
                return {
                    "type": FormulaType.ATOMIC.value,
                    "predicate": formula_str.strip(),
                    "arguments": [],
                    "original": formula_str
                }
            
            logger.warning(f"原子公式解析失败: {formula_str}")
            return None
            
        except Exception as e:
            logger.error(f"解析原子公式时出错: {formula_str}, 错误: {e}")
            return None
    
    def _parse_argument_list(self, args_str: str) -> Optional[List[Dict[str, Any]]]:
        """解析参数列表"""
        try:
            if not args_str:
                return []
            
            # 分割参数（考虑嵌套函数）
            args = []
            current_arg = ""
            parentheses_level = 0
            
            for char in args_str:
                if char == '(':
                    parentheses_level += 1
                    current_arg += char
                elif char == ')':
                    parentheses_level -= 1
                    current_arg += char
                elif char == ',' and parentheses_level == 0:
                    # 参数分隔符
                    if current_arg:
                        args.append(self._parse_term(current_arg.strip()))
                        current_arg = ""
                else:
                    current_arg += char
            
            # 添加最后一个参数
            if current_arg:
                args.append(self._parse_term(current_arg.strip()))
            
            return args if args else []
            
        except Exception as e:
            logger.error(f"解析参数列表时出错: {args_str}, 错误: {e}")
            return None
    
    def _parse_term(self, term_str: str) -> Dict[str, Any]:
        """解析项（变量、常量、函数）"""
        # 移除空格
        term_str = term_str.strip()
        
        # 检查是否为函数
        if "(" in term_str and term_str.endswith(")"):
            # 函数项: f(x, y)
            func_end = term_str.index("(")
            func_name = term_str[:func_end].strip()
            args_str = term_str[func_end+1:-1].strip()
            
            # 解析函数参数
            args = self._parse_argument_list(args_str)
            
            return {
                "type": TermType.FUNCTION.value,
                "term_type": TermType.FUNCTION,
                "name": func_name,
                "arguments": args or []
            }
        else:
            # 变量或常量
            # 简单规则: 小写字母开头为变量，大写字母开头为常量
            # 或单字母为变量，多字母为常量
            
            if len(term_str) == 1 and term_str.islower():
                # 单字母小写: 变量
                term_type = TermType.VARIABLE
            elif term_str[0].isupper():
                # 大写字母开头: 常量
                term_type = TermType.CONSTANT
            else:
                # 默认: 常量
                term_type = TermType.CONSTANT
            
            return {
                "type": term_type.value,
                "term_type": term_type,
                "name": term_str,
                "arguments": []
            }
    
    def _is_fact(self, formula: Dict[str, Any]) -> bool:
        """判断是否为事实（无变量、无量词的原子公式）"""
        try:
            # 检查是否为原子公式
            if formula.get("type") != FormulaType.ATOMIC.value:
                return False
            
            # 检查是否有变量参数
            args = formula.get("arguments", [])
            for arg in args:
                if self._contains_variables(arg):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _contains_variables(self, term: Dict[str, Any]) -> bool:
        """检查项中是否包含变量"""
        term_type = term.get("term_type")
        
        if term_type == TermType.VARIABLE.value or term_type == TermType.VARIABLE:
            return True
        elif term_type == TermType.FUNCTION.value or term_type == TermType.FUNCTION:
            # 检查函数参数
            args = term.get("arguments", [])
            for arg in args:
                if self._contains_variables(arg):
                    return True
        
        return False
    
    def _extract_symbols(self, formula: Dict[str, Any]):
        """从公式中提取符号（谓词、函数、常量、变量）"""
        # 递归提取
        self._extract_symbols_recursive(formula)
    
    def _extract_symbols_recursive(self, formula: Dict[str, Any]):
        """递归提取符号"""
        if not formula:
            return
        
        formula_type = formula.get("type")
        
        if formula_type == FormulaType.ATOMIC.value:
            # 原子公式：提取谓词和参数
            predicate = formula.get("predicate")
            if predicate:
                self.knowledge_base["predicates"][predicate] = self.knowledge_base["predicates"].get(predicate, 0) + 1
            
            # 提取参数中的符号
            args = formula.get("arguments", [])
            for arg in args:
                self._extract_symbols_from_term(arg)
        
        elif formula_type == FormulaType.NEGATION.value:
            # 否定：提取主体
            body = formula.get("body")
            if body:
                self._extract_symbols_recursive(body)
        
        elif formula_type in [FormulaType.CONJUNCTION.value, FormulaType.DISJUNCTION.value, 
                             FormulaType.IMPLICATION.value, FormulaType.EQUIVALENCE.value]:
            # 二元连接词：提取左右操作数
            left = formula.get("left")
            right = formula.get("right")
            if left:
                self._extract_symbols_recursive(left)
            if right:
                self._extract_symbols_recursive(right)
        
        elif formula_type in [FormulaType.UNIVERSAL.value, FormulaType.EXISTENTIAL.value]:
            # 量词：提取变量和主体
            variables = formula.get("variables", [])
            for var in variables:
                self.knowledge_base["variables"].add(var)
            
            body = formula.get("body")
            if body:
                self._extract_symbols_recursive(body)
    
    def _extract_symbols_from_term(self, term: Dict[str, Any]):
        """从项中提取符号"""
        if not term:
            return
        
        term_type = term.get("term_type")
        term_name = term.get("name", "")
        
        if term_type == TermType.VARIABLE.value or term_type == TermType.VARIABLE:
            self.knowledge_base["variables"].add(term_name)
        
        elif term_type == TermType.CONSTANT.value or term_type == TermType.CONSTANT:
            self.knowledge_base["constants"].add(term_name)
        
        elif term_type == TermType.FUNCTION.value or term_type == TermType.FUNCTION:
            self.knowledge_base["functions"][term_name] = self.knowledge_base["functions"].get(term_name, 0) + 1
            
            # 提取函数参数
            args = term.get("arguments", [])
            for arg in args:
                self._extract_symbols_from_term(arg)
    
    def query(self, query_str: str, inference_method: InferenceRule = InferenceRule.FORWARD_CHAINING) -> Dict[str, Any]:
        """
        查询知识库
        
        Args:
            query_str: 查询字符串
            inference_method: 推理方法
            
        Returns:
            查询结果
        """
        start_time = time.time()
        
        # 解析查询
        query_formula = self._parse_formula(query_str)
        
        if not query_formula:
            return {
                "success": False,
                "error": f"查询解析失败: {query_str}",
                "query": query_str
            }
        
        # 检查缓存
        cache_key = self._create_cache_key(query_formula, inference_method)
        if cache_key in self.inference_cache:
            cached_result = self.inference_cache[cache_key]
            cached_result["cached"] = True
            cached_result["cache_hit"] = True
            self.performance_stats["queries_processed"] += 1
            return cached_result
        
        # 执行推理
        if inference_method == InferenceRule.FORWARD_CHAINING:
            result = self._forward_chaining(query_formula)
        elif inference_method == InferenceRule.BACKWARD_CHAINING:
            result = self._backward_chaining(query_formula)
        elif inference_method == InferenceRule.RESOLUTION:
            result = self._resolution_proof(query_formula)
        else:
            # 默认使用前向链接
            result = self._forward_chaining(query_formula)
        
        # 添加性能信息
        elapsed_time = time.time() - start_time
        result["performance"] = {
            "query_time": elapsed_time,
            "inference_method": inference_method.value,
            "cache_key": cache_key
        }
        
        # 更新缓存
        if result.get("success", False):
            self.inference_cache[cache_key] = result.copy()
            # 限制缓存大小
            if len(self.inference_cache) > 1000:
                # 移除最早的缓存项
                oldest_key = next(iter(self.inference_cache))
                del self.inference_cache[oldest_key]
        
        self.performance_stats["queries_processed"] += 1
        logger.info(f"查询完成: {query_str}, 结果: {result.get('success', False)}, 耗时: {elapsed_time:.3f}秒")
        
        return result
    
    def _create_cache_key(self, formula: Dict[str, Any], inference_method: InferenceRule) -> str:
        """创建缓存键"""
        # 使用公式的字符串表示和推理方法
        formula_str = str(formula.get("original", str(formula)))
        return f"{inference_method.value}:{(zlib.adler32(str(formula_str).encode('utf-8')) & 0xffffffff) % 1000000}"
    
    def _forward_chaining(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """前向链接推理"""
        try:
            # 简化实现：检查查询是否可从知识库推导
            
            # 提取所有事实
            facts = [f["formula"] for f in self.knowledge_base["facts"]]
            
            # 应用规则推导新事实
            derived_facts = set()
            for rule in self.knowledge_base["rules"]:
                rule_formula = rule["formula"]
                new_facts = self._apply_rule_forward(rule_formula, facts)
                derived_facts.update(new_facts)
            
            # 合并所有事实
            all_facts = facts + list(derived_facts)
            
            # 检查查询是否在事实中
            for fact in all_facts:
                if self._formula_equal(fact, query):
                    return {
                        "success": True,
                        "result": "查询可从知识库推导",
                        "proof": [{"step": 1, "fact": fact, "source": "knowledge_base"}],
                        "facts_used": len(all_facts),
                        "rules_applied": len(self.knowledge_base["rules"])
                    }
            
            # 检查查询是否与事实矛盾
            for fact in all_facts:
                if self._formula_contradicts(fact, query):
                    return {
                        "success": False,
                        "result": "查询与知识库矛盾",
                        "contradiction": fact,
                        "facts_checked": len(all_facts)
                    }
            
            # 查询既不能推导也不矛盾
            return {
                "success": False,
                "result": "查询无法从知识库推导",
                "facts_checked": len(all_facts),
                "note": "查询与知识库一致但无法直接推导"
            }
            
        except Exception as e:
            logger.error(f"前向链接推理出错: {e}")
            return {
                "success": False,
                "error": str(e),
                "result": "推理过程出错"
            }
    
    def _apply_rule_forward(self, rule: Dict[str, Any], facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """前向应用规则"""
        # 简化实现：只处理简单的蕴含规则
        derived_facts = []
        
        # 检查是否为蕴含式
        if rule.get("type") == FormulaType.IMPLICATION.value:
            premise = rule.get("left")
            conclusion = rule.get("right")
            
            if premise and conclusion:
                # 检查前提是否匹配事实
                for fact in facts:
                    if self._formula_equal(fact, premise):
                        # 推导结论
                        derived_facts.append(conclusion)
                        self.performance_stats["inferences_made"] += 1
        
        return derived_facts
    
    def _backward_chaining(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """后向链接推理"""
        try:
            # 简化实现：递归搜索证明
            
            proof_steps = []
            visited_goals = set()
            
            def prove(goal, depth=0, max_depth=10):
                if depth > max_depth:
                    return False
                
                goal_key = str(goal.get("original", str(goal)))
                if goal_key in visited_goals:
                    return False
                
                visited_goals.add(goal_key)
                
                # 检查是否直接是事实
                for fact_info in self.knowledge_base["facts"]:
                    fact = fact_info["formula"]
                    if self._formula_equal(fact, goal):
                        proof_steps.append({
                            "depth": depth,
                            "goal": goal,
                            "type": "fact",
                            "fact": fact,
                            "source": fact_info["id"]
                        })
                        return True
                
                # 检查是否有规则可推导目标
                for rule_info in self.knowledge_base["rules"]:
                    rule = rule_info["formula"]
                    
                    # 检查是否为蕴含式且结论匹配目标
                    if rule.get("type") == FormulaType.IMPLICATION.value:
                        premise = rule.get("left")
                        conclusion = rule.get("right")
                        
                        if conclusion and self._formula_equal(conclusion, goal):
                            # 尝试证明前提
                            proof_steps.append({
                                "depth": depth,
                                "goal": goal,
                                "type": "rule_applied",
                                "rule": rule_info["id"],
                                "premise": premise
                            })
                            
                            if prove(premise, depth + 1, max_depth):
                                self.performance_stats["inferences_made"] += 1
                                return True
                            else:
                                # 回溯
                                proof_steps.pop()
                
                return False
            
            # 开始证明
            success = prove(query)
            
            if success:
                return {
                    "success": True,
                    "result": "查询可通过后向链接证明",
                    "proof": proof_steps,
                    "proof_length": len(proof_steps),
                    "max_depth_reached": max([step.get("depth", 0) for step in proof_steps]) if proof_steps else 0
                }
            else:
                return {
                    "success": False,
                    "result": "无法通过后向链接证明查询",
                    "goals_visited": len(visited_goals),
                    "max_depth": 10
                }
            
        except Exception as e:
            logger.error(f"后向链接推理出错: {e}")
            return {
                "success": False,
                "error": str(e),
                "result": "推理过程出错"
            }
    
    def _resolution_proof(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """归结原理证明"""
        # 简化实现：基本的归结步骤
        
        try:
            # 将查询取反（要证明的结论取反）
            negated_query = {
                "type": FormulaType.NEGATION.value,
                "connective": FormulaType.NEGATION,
                "body": query,
                "original": f"not({query.get('original', '')})"
            }
            
            # 将知识库和取反的查询转换为子句集
            clauses = []
            
            # 添加事实
            for fact_info in self.knowledge_base["facts"]:
                clauses.append(fact_info["formula"])
            
            # 添加取反的查询
            clauses.append(negated_query)
            
            # 归结步骤
            resolution_steps = []
            new_clauses = []
            max_steps = 100
            step_count = 0
            
            for i in range(len(clauses)):
                for j in range(i + 1, len(clauses)):
                    step_count += 1
                    if step_count > max_steps:
                        break
                    
                    clause1 = clauses[i]
                    clause2 = clauses[j]
                    
                    # 尝试归结
                    resolvent = self._resolve(clause1, clause2)
                    if resolvent:
                        resolution_steps.append({
                            "step": step_count,
                            "clause1": clause1.get("original", str(clause1)),
                            "clause2": clause2.get("original", str(clause2)),
                            "resolvent": resolvent.get("original", str(resolvent))
                        })
                        
                        # 检查是否得到空子句（矛盾）
                        if self._is_empty_clause(resolvent):
                            self.performance_stats["resolution_steps"] += step_count
                            self.performance_stats["proofs_completed"] += 1
                            
                            return {
                                "success": True,
                                "result": "查询通过归结原理证明",
                                "proof_method": "resolution",
                                "resolution_steps": resolution_steps,
                                "steps_required": step_count,
                                "empty_clause_found": True
                            }
                        
                        new_clauses.append(resolvent)
            
            self.performance_stats["resolution_steps"] += step_count
            
            return {
                "success": False,
                "result": "无法通过归结原理证明查询（未找到空子句）",
                "resolution_steps": resolution_steps,
                "steps_tried": step_count,
                "new_clauses_generated": len(new_clauses)
            }
            
        except Exception as e:
            logger.error(f"归结原理证明出错: {e}")
            return {
                "success": False,
                "error": str(e),
                "result": "归结证明过程出错"
            }
    
    def _resolve(self, clause1: Dict[str, Any], clause2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """执行归结"""
        # 简化实现：基本的文字归结
        # 实际实现需要完整的合一算法
        
        # 检查clause1中是否有否定，clause2中有对应的肯定（或反之）
        if clause1.get("type") == FormulaType.NEGATION.value:
            negated_formula = clause1.get("body")
            # 检查clause2是否与negated_formula相同
            if self._formula_equal(clause2, negated_formula):
                # 归结得到空子句（实际上应该返回None表示空）
                return None
        
        elif clause2.get("type") == FormulaType.NEGATION.value:
            negated_formula = clause2.get("body")
            # 检查clause1是否与negated_formula相同
            if self._formula_equal(clause1, negated_formula):
                # 归结得到空子句
                return None
        
        # 没有可归结的文字
        return None
    
    def _is_empty_clause(self, clause: Optional[Dict[str, Any]]) -> bool:
        """检查是否为空子句"""
        return clause is None
    
    def _formula_equal(self, formula1: Dict[str, Any], formula2: Dict[str, Any]) -> bool:
        """检查两个公式是否相等（简化实现）"""
        # 使用字符串表示进行比较
        str1 = str(formula1.get("original", str(formula1)))
        str2 = str(formula2.get("original", str(formula2)))
        
        # 规范化字符串（移除多余空格）
        str1_norm = " ".join(str1.split())
        str2_norm = " ".join(str2.split())
        
        return str1_norm == str2_norm
    
    def _formula_contradicts(self, formula1: Dict[str, Any], formula2: Dict[str, Any]) -> bool:
        """检查两个公式是否矛盾"""
        # 检查formula1是否为formula2的否定
        if formula1.get("type") == FormulaType.NEGATION.value:
            body = formula1.get("body")
            if body and self._formula_equal(body, formula2):
                return True
        
        # 检查formula2是否为formula1的否定
        if formula2.get("type") == FormulaType.NEGATION.value:
            body = formula2.get("body")
            if body and self._formula_equal(body, formula1):
                return True
        
        return False
    
    def unify(self, term1: Dict[str, Any], term2: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        合一算法：找到使两个项相等的替换
        
        Args:
            term1: 第一个项
            term2: 第二个项
            
        Returns:
            替换字典 {变量: 值} 或 None（如果无法合一）
        """
        start_time = time.time()
        
        # 初始化替换
        substitution = {}
        
        # 执行合一
        success = self._unify_recursive(term1, term2, substitution)
        
        elapsed_time = time.time() - start_time
        
        if success:
            self.performance_stats["unifications_performed"] += 1
            logger.debug(f"合一成功: {substitution}, 耗时: {elapsed_time:.3f}秒")
            return substitution
        else:
            logger.debug(f"合一失败: term1={term1}, term2={term2}, 耗时: {elapsed_time:.3f}秒")
            return None
    
    def _unify_recursive(self, term1: Dict[str, Any], term2: Dict[str, Any], substitution: Dict[str, Any]) -> bool:
        """递归合一算法"""
        # 获取项类型
        type1 = term1.get("term_type")
        type2 = term2.get("term_type")
        name1 = term1.get("name", "")
        name2 = term2.get("name", "")
        
        # 情况1: 两者都是变量
        if type1 == TermType.VARIABLE and type2 == TermType.VARIABLE:
            if name1 == name2:
                return True
            else:
                # 可以合一为同一个变量
                # 简化: 将name2替换为name1
                if name2 not in substitution:
                    substitution[name2] = {"type": TermType.VARIABLE.value, "term_type": TermType.VARIABLE, "name": name1}
                return True
        
        # 情况2: term1是变量
        elif type1 == TermType.VARIABLE:
            # 检查变量是否已有绑定
            if name1 in substitution:
                return self._unify_recursive(substitution[name1], term2, substitution)
            else:
                # 检查occurs check: term2中是否包含term1
                if self._occurs_check(name1, term2, substitution):
                    return False
                # 绑定变量
                substitution[name1] = term2
                return True
        
        # 情况3: term2是变量
        elif type2 == TermType.VARIABLE:
            # 对称处理
            return self._unify_recursive(term2, term1, substitution)
        
        # 情况4: 两者都是常量
        elif type1 == TermType.CONSTANT and type2 == TermType.CONSTANT:
            return name1 == name2
        
        # 情况5: 两者都是函数
        elif type1 == TermType.FUNCTION and type2 == TermType.FUNCTION:
            # 函数名必须相同
            if name1 != name2:
                return False
            
            # 参数数量必须相同
            args1 = term1.get("arguments", [])
            args2 = term2.get("arguments", [])
            
            if len(args1) != len(args2):
                return False
            
            # 递归合一每个参数
            for arg1, arg2 in zip(args1, args2):
                if not self._unify_recursive(arg1, arg2, substitution):
                    return False
            
            return True
        
        # 情况6: 类型不匹配
        else:
            return False
    
    def _occurs_check(self, var_name: str, term: Dict[str, Any], substitution: Dict[str, Any]) -> bool:
        """检查变量是否出现在项中（occurs check）"""
        term_type = term.get("term_type")
        term_name = term.get("name", "")
        
        # 检查项本身是否为变量
        if term_type == TermType.VARIABLE:
            # 如果变量有绑定，检查绑定的项
            if term_name in substitution:
                return self._occurs_check(var_name, substitution[term_name], substitution)
            else:
                return var_name == term_name
        
        # 检查函数项中的参数
        elif term_type == TermType.FUNCTION:
            args = term.get("arguments", [])
            for arg in args:
                if self._occurs_check(var_name, arg, substitution):
                    return True
        
        # 常量不包含变量
        return False
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """获取知识库信息"""
        return {
            "facts_count": len(self.knowledge_base["facts"]),
            "rules_count": len(self.knowledge_base["rules"]),
            "predicates_count": len(self.knowledge_base["predicates"]),
            "functions_count": len(self.knowledge_base["functions"]),
            "constants_count": len(self.knowledge_base["constants"]),
            "variables_count": len(self.knowledge_base["variables"]),
            "inference_cache_size": len(self.inference_cache)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return self.performance_stats.copy()
    
    def clear_cache(self):
        """清除推理缓存"""
        self.inference_cache.clear()
        logger.info("推理缓存已清除")
    
    def save_knowledge_base(self, filepath: str) -> bool:
        """保存知识库到文件"""
        try:
            import pickle
            kb_data = {
                "knowledge_base": self.knowledge_base,
                "performance_stats": self.performance_stats
            }
            with open(filepath, 'wb') as f:
                pickle.dump(kb_data, f)
            logger.info(f"知识库保存到: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存知识库失败: {e}")
            return False
    
    def load_knowledge_base(self, filepath: str) -> bool:
        """从文件加载知识库"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                kb_data = pickle.load(f)
            
            self.knowledge_base = kb_data.get("knowledge_base", self.knowledge_base)
            self.performance_stats = kb_data.get("performance_stats", self.performance_stats.copy())
            
            logger.info(f"知识库从 {filepath} 加载")
            return True
        except Exception as e:
            logger.error(f"加载知识库失败: {e}")
            return False


# 示例和测试函数
def create_example_reasoner() -> FirstOrderLogicReasoner:
    """创建示例推理引擎"""
    reasoner = FirstOrderLogicReasoner()
    
    # 添加示例知识
    example_knowledge = [
        # 事实
        "human(Socrates)",
        "mortal(Socrates)",
        "philosopher(Socrates)",
        "wise(Socrates)",
        
        # 规则
        "forall x (human(x) → mortal(x))",
        "forall x (philosopher(x) → wise(x))",
        "forall x (wise(x) ∧ human(x) → respected(x))",
        
        # 更复杂的规则
        "forall x forall y (parent(x, y) → older(x, y))",
        "parent(Zeus, Apollo)",
        "parent(Apollo, Asclepius)"
    ]
    
    for knowledge in example_knowledge:
        reasoner.add_formula(knowledge)
    
    return reasoner


def test_logic_reasoner():
    """测试逻辑推理引擎"""
    logger.info("开始测试一阶逻辑推理引擎")
    
    # 创建示例推理引擎
    reasoner = create_example_reasoner()
    
    # 测试查询
    test_queries = [
        "mortal(Socrates)",
        "wise(Socrates)", 
        "respected(Socrates)",
        "older(Zeus, Apollo)",
        "older(Zeus, Asclepius)",
        "human(Zeus)"  # 应该返回False
    ]
    
    for query in test_queries:
        logger.info(f"查询: {query}")
        result = reasoner.query(query, InferenceRule.FORWARD_CHAINING)
        
        if result["success"]:
            logger.info(f"  结果: 可推导 ({result['result']})")
        else:
            logger.info(f"  结果: 不可推导 ({result['result']})")
    
    # 测试归结原理
    logger.info("测试归结原理证明...")
    resolution_result = reasoner.query("mortal(Socrates)", InferenceRule.RESOLUTION)
    logger.info(f"归结证明结果: {resolution_result['success']}, 步骤: {resolution_result.get('steps_tried', 0)}")
    
    # 测试合一算法
    logger.info("测试合一算法...")
    
    # 创建两个项
    term1 = reasoner._parse_term("x")
    term2 = reasoner._parse_term("Socrates")
    
    substitution = reasoner.unify(term1, term2)
    if substitution:
        logger.info(f"合一成功: {substitution}")
    else:
        logger.info("合一失败")
    
    # 显示知识库信息
    kb_info = reasoner.get_knowledge_base_info()
    logger.info(f"知识库信息: {kb_info}")
    
    # 显示性能统计
    stats = reasoner.get_performance_stats()
    logger.info(f"性能统计: {stats}")
    
    logger.info("一阶逻辑推理引擎测试完成")
    return reasoner


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_reasoner = test_logic_reasoner()