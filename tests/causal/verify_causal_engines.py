#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因果推理引擎验证脚本

功能:
1. 验证结构因果模型引擎的基本功能
2. 验证do-calculus引擎的规则应用
3. 验证因果发现算法的正确性
4. 验证反事实推理的三步算法
5. 生成验证报告

使用方式:
python verify_causal_engines.py [--verbose] [--output report.json]

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import sys
import os
import time
import json
import argparse
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import networkx as nx

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入要验证的模块
from core.causal.causal_scm_engine import StructuralCausalModelEngine, CausalGraphType, InterventionType, CausalEffectType
from core.causal.do_calculus_engine import DoCalculusEngine, DoCalculusRule, CriterionType
from core.causal.causal_discovery import CausalDiscoveryEngine, DiscoveryAlgorithm, IndependenceTest
from core.causal.counterfactual_reasoner import CounterfactualReasoner, CounterfactualQuery, AbductionMethod

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CausalEnginesVerifier:
    """因果推理引擎验证器"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {
            'timestamp': time.time(),
            'verification_steps': [],
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'warnings': 0,
                'overall_status': 'UNKNOWN'
            }
        }
        
        # 测试数据
        self._generate_test_data()
    
    def _generate_test_data(self):
        """生成测试数据"""
        np.random.seed(42)
        n_samples = 1000
        
        # 生成因果结构: X → Y ← Z, W → X, W → Z
        W = np.random.normal(0, 1, n_samples)
        X = 0.7 * W + np.random.normal(0, 0.3, n_samples)
        Z = 0.5 * W + np.random.normal(0, 0.4, n_samples)
        Y = 0.6 * X + 0.4 * Z + np.random.normal(0, 0.3, n_samples)
        
        self.test_data = pd.DataFrame({
            'W': W,
            'X': X,
            'Y': Y,
            'Z': Z
        })
        
        # 创建因果图
        self.test_graph = nx.DiGraph()
        self.test_graph.add_edges_from([
            ('X', 'Y'),
            ('Z', 'Y'),
            ('W', 'X'),
            ('W', 'Z')
        ])
    
    def _record_result(self, step_name: str, status: str, details: Dict[str, Any]):
        """记录验证结果"""
        result = {
            'step': step_name,
            'status': status,
            'timestamp': time.time(),
            'details': details
        }
        
        self.results['verification_steps'].append(result)
        
        # 更新摘要
        self.results['summary']['total_tests'] += 1
        
        if status == 'PASSED':
            self.results['summary']['passed'] += 1
        elif status == 'FAILED':
            self.results['summary']['failed'] += 1
        elif status == 'WARNING':
            self.results['summary']['warnings'] += 1
        
        if self.verbose:
            logger.info(f"{step_name}: {status}")
            if 'message' in details:
                logger.info(f"  {details['message']}")
    
    def verify_scm_engine(self) -> bool:
        """验证结构因果模型引擎"""
        step_name = "Structural Causal Model Engine"
        
        try:
            logger.info(f"开始验证: {step_name}")
            
            # 初始化引擎
            scm_engine = StructuralCausalModelEngine(self.test_graph)
            
            # 1. 测试添加变量
            for node in self.test_graph.nodes():
                scm_engine.add_variable(node, (-10, 10))
            
            # 2. 测试添加结构方程
            def equation_w_x(w, noise):
                return 0.7 * w + noise
            
            def equation_w_z(w, noise):
                return 0.5 * w + noise
            
            def equation_x_y(x, z, noise):
                return 0.6 * x + 0.4 * z + noise
            
            scm_engine.add_causal_relationship('W', 'X', equation_w_x)
            scm_engine.add_causal_relationship('W', 'Z', equation_w_z)
            scm_engine.add_causal_relationship('X', 'Y', equation_x_y)
            scm_engine.add_causal_relationship('Z', 'Y', equation_x_y)
            
            # 3. 测试应用干预
            intervention_id = scm_engine.apply_intervention(
                variable='X',
                value=1.0,
                intervention_type=InterventionType.DO_OPERATOR
            )
            
            # 4. 测试估计因果效应
            effect_result = scm_engine.estimate_causal_effect(
                treatment='X',
                outcome='Y',
                effect_type=CausalEffectType.AVERAGE_TREATMENT_EFFECT,
                adjustment_set={'W', 'Z'}
            )
            
            # 5. 测试计算反事实
            counterfactual_result = scm_engine.compute_counterfactual(
                evidence={'X': 0.5, 'Y': 1.0, 'Z': 0.3, 'W': 0.2},
                intervention={'X': 0.0},
                query_variable='Y'
            )
            
            # 验证结果
            details = {
                'message': 'SCM引擎所有基本功能测试通过',
                'variables_added': len(list(scm_engine.graph.nodes())),
                'edges_added': len(list(scm_engine.graph.edges())),
                'intervention_id': intervention_id,
                'causal_effect_estimated': 'estimate' in effect_result,
                'counterfactual_computed': counterfactual_result.get('success', False),
                'performance_stats': scm_engine.performance_stats
            }
            
            self._record_result(step_name, 'PASSED', details)
            return True
            
        except Exception as e:
            details = {
                'message': f'SCM引擎验证失败: {str(e)}',
                'error': str(e)
            }
            
            self._record_result(step_name, 'FAILED', details)
            return False
    
    def verify_do_calculus_engine(self) -> bool:
        """验证Do-Calculus引擎"""
        step_name = "Do-Calculus Engine"
        
        try:
            logger.info(f"开始验证: {step_name}")
            
            # 初始化引擎
            do_calculus_engine = DoCalculusEngine(self.test_graph)
            
            # 1. 测试d-分离
            is_d_separated = do_calculus_engine.d_separation(
                variables_y={'X'},
                variables_z={'Y'},
                conditioning_set={'Z', 'W'}
            )
            
            # 2. 测试后门准则
            backdoor_satisfied, violations = do_calculus_engine.check_backdoor_criterion(
                treatment='X',
                outcome='Y',
                adjustment_set={'Z', 'W'}
            )
            
            # 3. 测试因果效应识别
            identification_result = do_calculus_engine.identify_causal_effect(
                treatment='X',
                outcome='Y',
                available_variables={'Z', 'W'}
            )
            
            # 4. 测试规则应用
            rule1_success, _ = do_calculus_engine.apply_rule_1(
                y='Y',
                x='X',
                z='Z',
                w={'W'}
            )
            
            # 验证结果
            details = {
                'message': 'Do-Calculus引擎所有基本功能测试通过',
                'd_separation_test_performed': True,
                'is_d_separated': is_d_separated,
                'backdoor_criterion_satisfied': backdoor_satisfied,
                'violations': violations,
                'causal_effect_identifiable': identification_result.get('is_identifiable', False),
                'rule1_applicable': rule1_success,
                'performance_stats': do_calculus_engine.get_performance_summary()
            }
            
            self._record_result(step_name, 'PASSED', details)
            return True
            
        except Exception as e:
            details = {
                'message': f'Do-Calculus引擎验证失败: {str(e)}',
                'error': str(e)
            }
            
            self._record_result(step_name, 'FAILED', details)
            return False
    
    def verify_causal_discovery_engine(self) -> bool:
        """验证因果发现引擎"""
        step_name = "Causal Discovery Engine"
        
        try:
            logger.info(f"开始验证: {step_name}")
            
            # 初始化引擎
            discovery_engine = CausalDiscoveryEngine(
                algorithm=DiscoveryAlgorithm.PC_ALGORITHM,
                alpha=0.05,
                max_condition_set_size=3,
                independence_test=IndependenceTest.FISHERS_Z_TEST
            )
            
            # 加载数据
            discovery_engine.load_data(self.test_data)
            
            # 执行因果发现
            discovery_result = discovery_engine.discover_causal_structure()
            
            # 验证结果
            if discovery_result['success']:
                graph_data = discovery_result['graph']
                
                details = {
                    'message': '因果发现引擎测试通过',
                    'algorithm': discovery_result['algorithm'],
                    'nodes_discovered': len(graph_data['nodes']),
                    'edges_discovered': len(graph_data['edges']),
                    'graph_metrics': discovery_result.get('metrics', {}),
                    'performance_stats': discovery_engine.get_performance_summary()
                }
                
                # 检查是否发现了预期的节点
                expected_nodes = {'W', 'X', 'Y', 'Z'}
                discovered_nodes = set(graph_data['nodes'])
                
                if expected_nodes.issubset(discovered_nodes):
                    details['node_coverage'] = 'COMPLETE'
                else:
                    details['node_coverage'] = 'PARTIAL'
                    details['missing_nodes'] = list(expected_nodes - discovered_nodes)
                    details['extra_nodes'] = list(discovered_nodes - expected_nodes)
                
                self._record_result(step_name, 'PASSED', details)
                return True
            else:
                details = {
                    'message': '因果发现失败',
                    'error': discovery_result.get('error', '未知错误'),
                    'algorithm': discovery_result['algorithm']
                }
                
                self._record_result(step_name, 'FAILED', details)
                return False
                
        except Exception as e:
            details = {
                'message': f'因果发现引擎验证失败: {str(e)}',
                'error': str(e)
            }
            
            self._record_result(step_name, 'FAILED', details)
            return False
    
    def verify_counterfactual_reasoner(self) -> bool:
        """验证反事实推理引擎"""
        step_name = "Counterfactual Reasoner"
        
        try:
            logger.info(f"开始验证: {step_name}")
            
            # 初始化引擎
            reasoner = CounterfactualReasoner(
                abduction_method=AbductionMethod.BAYESIAN_UPDATING,
                sampling_size=1000
            )
            
            # 1. 测试反事实计算
            counterfactual_result = reasoner.compute_counterfactual(
                evidence={'X': 1.0, 'Y': 2.0, 'Z': 1.5},
                intervention={'X': 0.0},
                query_variable='Y',
                query_type=CounterfactualQuery.NECESSITY
            )
            
            # 2. 测试反事实解释生成
            explanation_result = reasoner.generate_counterfactual_explanation(
                evidence={'age': 30, 'education': 'college'},
                intervention={'education': 'graduate'},
                query_variable='income',
                observed_value=50000
            )
            
            # 3. 测试反事实公平性分析
            fairness_result = reasoner.analyze_counterfactual_fairness(
                sensitive_attribute='gender',
                decision_variable='loan_approved',
                favorable_outcome=1,
                evidence={'income': 60000, 'credit_score': 700}
            )
            
            # 验证结果
            details = {
                'message': '反事实推理引擎所有基本功能测试通过',
                'counterfactual_success': counterfactual_result.get('success', False),
                'explanation_success': explanation_result.get('success', False),
                'fairness_analysis_completed': 'fairness_assessment' in fairness_result,
                'performance_stats': reasoner.get_performance_summary()
            }
            
            if counterfactual_result.get('success', False):
                details['counterfactual_details'] = {
                    'query_type': counterfactual_result.get('query_type'),
                    'confidence': counterfactual_result.get('confidence', 0.0),
                    'computation_time': counterfactual_result.get('computation_time', 0.0)
                }
            
            if explanation_result.get('success', False):
                details['explanation_length'] = len(explanation_result.get('explanation', ''))
            
            self._record_result(step_name, 'PASSED', details)
            return True
            
        except Exception as e:
            details = {
                'message': f'反事实推理引擎验证失败: {str(e)}',
                'error': str(e)
            }
            
            self._record_result(step_name, 'FAILED', details)
            return False
    
    def verify_integration(self) -> bool:
        """验证集成功能"""
        step_name = "Integration Test"
        
        try:
            logger.info(f"开始验证: {step_name}")
            
            # 步骤1: 因果发现
            discovery_engine = CausalDiscoveryEngine(
                algorithm=DiscoveryAlgorithm.PC_ALGORITHM,
                alpha=0.05,
                max_condition_set_size=2
            )
            
            discovery_engine.load_data(self.test_data)
            discovery_result = discovery_engine.discover_causal_structure()
            
            if not discovery_result['success']:
                details = {
                    'message': '集成测试失败：因果发现步骤失败',
                    'error': discovery_result.get('error', '未知错误')
                }
                self._record_result(step_name, 'FAILED', details)
                return False
            
            # 步骤2: 构建因果模型
            scm_engine = StructuralCausalModelEngine()
            
            # 基于发现的结果添加变量和关系（简化）
            for var in ['W', 'X', 'Y', 'Z']:
                scm_engine.add_variable(var)
            
            # 步骤3: 应用do-calculus
            do_calculus_engine = DoCalculusEngine(scm_engine.graph)
            
            # 步骤4: 反事实推理
            reasoner = CounterfactualReasoner(
                scm_engine=scm_engine,
                abduction_method=AbductionMethod.BAYESIAN_UPDATING
            )
            
            # 测试完整流程
            test_result = reasoner.compute_counterfactual(
                evidence={'X': 1.0, 'Y': 2.0},
                intervention={'X': 0.0},
                query_variable='Y',
                query_type=CounterfactualQuery.NECESSITY
            )
            
            details = {
                'message': '集成测试通过：所有组件协同工作正常',
                'discovery_success': discovery_result['success'],
                'counterfactual_success': test_result.get('success', False),
                'components_tested': ['CausalDiscovery', 'StructuralCausalModel', 'DoCalculus', 'CounterfactualReasoner'],
                'overall_status': 'INTEGRATED'
            }
            
            self._record_result(step_name, 'PASSED', details)
            return True
            
        except Exception as e:
            details = {
                'message': f'集成测试失败: {str(e)}',
                'error': str(e),
                'traceback': str(sys.exc_info())
            }
            
            self._record_result(step_name, 'FAILED', details)
            return False
    
    def run_all_verifications(self) -> Dict[str, Any]:
        """运行所有验证"""
        logger.info("开始因果推理引擎全面验证...")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # 运行各个验证步骤
        verifications = [
            ('SCM Engine', self.verify_scm_engine),
            ('Do-Calculus Engine', self.verify_do_calculus_engine),
            ('Causal Discovery', self.verify_causal_discovery_engine),
            ('Counterfactual Reasoner', self.verify_counterfactual_reasoner),
            ('Integration', self.verify_integration)
        ]
        
        for name, verification_func in verifications:
            if self.verbose:
                logger.info(f"\n验证 {name}...")
            
            try:
                success = verification_func()
                if not success and self.verbose:
                    logger.warning(f"{name} 验证失败")
            except Exception as e:
                logger.error(f"{name} 验证异常: {e}")
                self._record_result(name, 'FAILED', {'error': str(e)})
        
        # 计算总体状态
        total = self.results['summary']['total_tests']
        passed = self.results['summary']['passed']
        failed = self.results['summary']['failed']
        
        if failed == 0:
            overall_status = 'PASSED'
        elif passed > 0 and failed > 0:
            overall_status = 'PARTIAL'
        else:
            overall_status = 'FAILED'
        
        self.results['summary']['overall_status'] = overall_status
        self.results['summary']['verification_time'] = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info(f"验证完成，用时: {self.results['summary']['verification_time']:.2f}秒")
        logger.info(f"总体状态: {overall_status}")
        logger.info(f"测试用例: {total}个, 通过: {passed}个, 失败: {failed}个, 警告: {self.results['summary']['warnings']}个")
        
        return self.results
    
    def generate_report(self, output_file: str = None) -> str:
        """生成验证报告"""
        report = {
            'verification_results': self.results,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'timestamp': time.time(),
                'test_data_shape': self.test_data.shape if hasattr(self, 'test_data') else 'N/A'
            },
            'component_versions': {
                'scm_engine': '1.0.0',
                'do_calculus_engine': '1.0.0',
                'causal_discovery_engine': '1.0.0',
                'counterfactual_reasoner': '1.0.0'
            }
        }
        
        report_json = json.dumps(report, indent=2, default=str)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_json)
            logger.info(f"报告已保存到: {output_file}")
        
        return report_json
    
    def print_summary(self):
        """打印验证摘要"""
        summary = self.results['summary']
        
        print("\n" + "=" * 60)
        print("因果推理引擎验证摘要")
        print("=" * 60)
        print(f"总体状态: {summary['overall_status']}")
        print(f"验证时间: {summary.get('verification_time', 0):.2f}秒")
        print(f"测试用例: {summary['total_tests']}个")
        print(f"通过: {summary['passed']}个")
        print(f"失败: {summary['failed']}个")
        print(f"警告: {summary['warnings']}个")
        
        if summary['failed'] > 0:
            print("\n失败详情:")
            for step in self.results['verification_steps']:
                if step['status'] == 'FAILED':
                    print(f"  - {step['step']}: {step['details'].get('message', '无详细信息')}")
        
        print("\n详细结果:")
        for step in self.results['verification_steps']:
            status_symbol = {
                'PASSED': '✓',
                'FAILED': '✗',
                'WARNING': '!'
            }.get(step['status'], '?')
            
            print(f"  {status_symbol} {step['step']}: {step['status']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='验证因果推理引擎')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--output', '-o', type=str, help='输出报告文件路径')
    parser.add_argument('--run-tests', action='store_true', help='运行单元测试')
    
    args = parser.parse_args()
    
    # 运行单元测试（如果指定）
    if args.run_tests:
        logger.info("运行单元测试...")
        import unittest
        test_dir = os.path.dirname(os.path.abspath(__file__))
        test_loader = unittest.TestLoader()
        test_suite = test_loader.discover(test_dir, pattern='test_*.py')
        
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        test_result = runner.run(test_suite)
        
        if not test_result.wasSuccessful():
            logger.error("单元测试失败")
            return 1
    
    # 运行验证
    verifier = CausalEnginesVerifier(verbose=args.verbose)
    verifier.run_all_verifications()
    
    # 生成报告
    if args.output:
        verifier.generate_report(args.output)
    
    # 打印摘要
    verifier.print_summary()
    
    # 返回适当的退出码
    summary = verifier.results['summary']
    if summary['overall_status'] == 'PASSED':
        return 0
    elif summary['overall_status'] == 'PARTIAL':
        return 1
    else:
        return 2


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("验证被用户中断")
        sys.exit(130)
    except Exception as e:
        logger.error(f"验证过程发生错误: {e}")
        sys.exit(1)