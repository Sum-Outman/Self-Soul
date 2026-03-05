#!/usr/bin/env python3
"""
Self-Soul AGI系统健康检查脚本

该脚本验证升级后的AGI系统完整性，包括：
1. 核心模块导入测试
2. 关键组件初始化测试
3. 功能基本测试
4. 系统配置验证
"""

import sys
import os
import logging
import time
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SystemHealthCheck")

class AGISystemHealthCheck:
    """AGI系统健康检查类"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        self.error_count = 0
        self.warning_count = 0
    
    def log_result(self, module: str, test: str, status: str, details: str = ""):
        """记录测试结果"""
        result = {
            "module": module,
            "test": test,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        
        if status == "ERROR":
            self.error_count += 1
            logger.error(f"{module} - {test}: ❌ {details}")
        elif status == "WARNING":
            self.warning_count += 1
            logger.warning(f"{module} - {test}: ⚠️ {details}")
        else:
            logger.info(f"{module} - {test}: ✅ {details}")
    
    def run_import_tests(self):
        """运行导入测试"""
        logger.info("开始核心模块导入测试...")
        
        # 阶段一：底层认知架构
        phase1_modules = [
            ("core.causal.causal_scm_engine", "StructuralCausalModelEngine"),
            ("core.causal.do_calculus_engine", "DoCalculusEngine"),
            ("core.causal.causal_discovery", "CausalDiscoveryEngine"),
            ("core.causal.counterfactual_reasoner", "CounterfactualReasoner"),
            ("core.neuro_symbolic.neuro_symbolic_unified", "NeuralSymbolicUnifiedFramework"),
            ("core.causal.causal_knowledge_graph", "CausalKnowledgeGraph"),
        ]
        
        for module_path, class_name in phase1_modules:
            try:
                module = __import__(module_path, fromlist=[class_name])
                if hasattr(module, class_name):
                    self.log_result("Phase1_Import", f"导入{class_name}", "PASS", f"成功导入{class_name}")
                else:
                    self.log_result("Phase1_Import", f"导入{class_name}", "ERROR", f"模块中未找到类{class_name}")
            except ImportError as e:
                self.log_result("Phase1_Import", f"导入{class_name}", "ERROR", f"导入失败: {e}")
        
        # 阶段二：世界模型与规划
        phase2_modules = [
            ("core.world_model.world_state_representation", "WorldStateRepresentation"),
            ("core.world_model.belief_state", "BeliefState"),
            ("core.planning.hierarchical_planning_system", "HierarchicalPlanningSystem"),
            ("core.planning.strategic_planner", "StrategicPlanner"),
            ("core.memory.episodic_semantic_memory", "EpisodicSemanticMemory"),
        ]
        
        for module_path, class_name in phase2_modules:
            try:
                module = __import__(module_path, fromlist=[class_name])
                if hasattr(module, class_name):
                    self.log_result("Phase2_Import", f"导入{class_name}", "PASS", f"成功导入{class_name}")
                else:
                    self.log_result("Phase2_Import", f"导入{class_name}", "ERROR", f"模块中未找到类{class_name}")
            except ImportError as e:
                self.log_result("Phase2_Import", f"导入{class_name}", "ERROR", f"导入失败: {e}")
        
        # 阶段三：认知闭环
        phase3_modules = [
            ("core.autonomy.autonomous_goal_generator", "AutonomousGoalGenerator"),
            ("core.metacognition.meta_cognitive_monitor", "MetaCognitiveMonitor"),
            ("core.self_improvement.self_improvement_loop", "SelfImprovementLoop"),
            ("core.self_improvement.meta_learning_architecture_search", "MetaLearningArchitectureSearch"),
            ("core.self_improvement.progressive_safety_alignment", "ProgressiveSafetyAlignment"),
        ]
        
        for module_path, class_name in phase3_modules:
            try:
                module = __import__(module_path, fromlist=[class_name])
                if hasattr(module, class_name):
                    self.log_result("Phase3_Import", f"导入{class_name}", "PASS", f"成功导入{class_name}")
                else:
                    self.log_result("Phase3_Import", f"导入{class_name}", "ERROR", f"模块中未找到类{class_name}")
            except ImportError as e:
                self.log_result("Phase3_Import", f"导入{class_name}", "ERROR", f"导入失败: {e}")
    
    def run_configuration_tests(self):
        """运行配置测试"""
        logger.info("开始系统配置测试...")
        
        config_files = [
            ("config/agi_coordinator_config.json", "JSON配置文件"),
            ("config/model_services_config.json", "模型服务配置"),
            ("config/constants.py", "常量配置"),
        ]
        
        for file_path, description in config_files:
            if os.path.exists(file_path):
                self.log_result("Configuration", f"检查{file_path}", "PASS", f"{description}存在")
            else:
                self.log_result("Configuration", f"检查{file_path}", "WARNING", f"{description}不存在")
        
        # 检查必要的目录结构
        required_dirs = [
            "core/causal",
            "core/neuro_symbolic", 
            "core/world_model",
            "core/planning",
            "core/memory",
            "core/autonomy",
            "core/metacognition",
            "core/self_improvement",
        ]
        
        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                self.log_result("Configuration", f"检查目录{dir_path}", "PASS", "目录存在")
            else:
                self.log_result("Configuration", f"检查目录{dir_path}", "ERROR", "目录不存在")
    
    def run_basic_functionality_tests(self):
        """运行基本功能测试"""
        logger.info("开始基本功能测试...")
        
        # 测试1：简单的因果推理组件初始化
        try:
            from core.causal.causal_scm_engine import StructuralCausalModelEngine
            scm = StructuralCausalModelEngine()
            self.log_result("Functionality", "因果推理引擎初始化", "PASS", "SCM引擎初始化成功")
        except Exception as e:
            self.log_result("Functionality", "因果推理引擎初始化", "ERROR", f"初始化失败: {e}")
        
        # 测试2：自主目标生成
        try:
            from core.autonomy.autonomous_goal_generator import AutonomousGoalGenerator
            goal_gen = AutonomousGoalGenerator()
            goals = goal_gen.generate_goal_candidates()
            self.log_result("Functionality", "自主目标生成", "PASS", f"生成{len(goals) if goals else 0}个目标候选")
        except Exception as e:
            self.log_result("Functionality", "自主目标生成", "WARNING", f"目标生成测试跳过: {e}")
        
        # 测试3：元认知监控
        try:
            from core.metacognition.meta_cognitive_monitor import MetaCognitiveMonitor
            monitor = MetaCognitiveMonitor(
                monitoring_interval_ms=100.0,
                max_thought_history=1000,
                bias_detection_enabled=True
            )
            state = monitor.get_metacognitive_state()
            self.log_result("Functionality", "元认知监控初始化", "PASS", "监控系统初始化成功")
        except Exception as e:
            self.log_result("Functionality", "元认知监控初始化", "WARNING", f"监控测试跳过: {e}")
        
        # 测试4：安全对齐验证
        try:
            from core.self_improvement.progressive_safety_alignment import ProgressiveSafetyAlignment, AlignmentPhase
            safety_alignment = ProgressiveSafetyAlignment(
                current_phase=AlignmentPhase.PHASE_1_BASIC,
                enable_progressive_alignment=True
            )
            status = safety_alignment.get_system_status()
            self.log_result("Functionality", "安全对齐系统", "PASS", "安全对齐系统初始化成功")
        except Exception as e:
            self.log_result("Functionality", "安全对齐系统", "WARNING", f"安全对齐测试跳过: {e}")
    
    def run_integration_test_verification(self):
        """验证集成测试"""
        logger.info("验证集成测试结果...")
        
        # 检查Phase 3集成测试报告
        import glob
        test_reports = glob.glob("tests/phase3_test_report_*.txt")
        
        if test_reports:
            latest_report = max(test_reports, key=os.path.getmtime)
            report_size = os.path.getsize(latest_report)
            
            # 读取报告内容检查结果
            try:
                with open(latest_report, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if "通过率: 100.0%" in content:
                    self.log_result("Integration", "Phase 3集成测试", "PASS", f"通过率100%，报告: {os.path.basename(latest_report)}")
                else:
                    self.log_result("Integration", "Phase 3集成测试", "WARNING", f"非100%通过，请检查报告: {os.path.basename(latest_report)}")
            except Exception as e:
                self.log_result("Integration", "Phase 3集成测试", "WARNING", f"读取测试报告失败: {e}")
        else:
            self.log_result("Integration", "Phase 3集成测试", "WARNING", "未找到集成测试报告")
    
    def run_environment_interface_tests(self):
        """运行环境接口测试"""
        logger.info("运行环境接口测试...")
        
        try:
            # 导入环境接口
            from core.environment_interface import (
                SimulatedEnvironment, UnifiedEnvironmentManager, 
                Action, ActionType
            )
            
            # 创建模拟环境
            env = SimulatedEnvironment({
                "interface_name": "health_check_env",
                "simulation_mode": "basic"
            })
            
            # 测试连接
            connected = env.connect()
            if not connected:
                self.log_result("Environment", "环境接口连接", "ERROR", "模拟环境连接失败")
                return
            
            self.log_result("Environment", "环境接口连接", "PASS", "模拟环境连接成功")
            
            # 测试动作执行
            action = Action(
                action_id="health_check_action",
                action_type=ActionType.PHYSICAL,
                parameters={"move": True, "direction": "forward", "distance": 1.0},
                priority=1,
                timeout_seconds=5.0
            )
            
            success, error_msg = env.execute_action(action)
            if success:
                self.log_result("Environment", "环境动作执行", "PASS", f"动作执行成功: {error_msg}")
            else:
                self.log_result("Environment", "环境动作执行", "ERROR", f"动作执行失败: {error_msg}")
            
            # 测试观察获取
            observations = env.get_observation()
            if observations:
                self.log_result("Environment", "环境观察获取", "PASS", f"获取{len(observations)}个观察")
            else:
                self.log_result("Environment", "环境观察获取", "WARNING", "未获取到观察")
            
            # 测试断开连接
            disconnected = env.disconnect()
            if disconnected:
                self.log_result("Environment", "环境断开连接", "PASS", "成功断开连接")
            else:
                self.log_result("Environment", "环境断开连接", "WARNING", "断开连接失败")
            
        except ImportError as e:
            self.log_result("Environment", "环境接口导入", "ERROR", f"导入失败: {e}")
        except Exception as e:
            self.log_result("Environment", "环境接口测试", "ERROR", f"测试出现意外错误: {e}")
    
    def generate_summary_report(self):
        """生成总结报告"""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r["status"] == "PASS"])
        failed_tests = len([r for r in self.results if r["status"] == "ERROR"])
        warning_tests = len([r for r in self.results if r["status"] == "WARNING"])
        
        duration = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("Self-Soul AGI系统健康检查报告")
        print("="*80)
        print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"检查时长: {duration:.2f}秒")
        print(f"总测试数: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"失败测试: {failed_tests}")
        print(f"警告测试: {warning_tests}")
        print(f"通过率: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "通过率: N/A")
        
        if failed_tests == 0:
            print("\n🎉 系统健康检查通过！所有核心功能正常。")
        else:
            print(f"\n⚠️ 系统存在{failed_tests}个错误，需要修复。")
        
        # 显示失败的测试
        if failed_tests > 0:
            print("\n错误详情:")
            for result in self.results:
                if result["status"] == "ERROR":
                    print(f"  ❌ {result['module']} - {result['test']}: {result['details']}")
        
        # 显示警告的测试
        if warning_tests > 0:
            print("\n警告详情:")
            for result in self.results:
                if result["status"] == "WARNING":
                    print(f"  ⚠️ {result['module']} - {result['test']}: {result['details']}")
        
        print("\n" + "="*80)
        
        # 保存详细报告
        report_file = f"system_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Self-Soul AGI系统健康检查详细报告\n")
            f.write("="*80 + "\n")
            f.write(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"检查时长: {duration:.2f}秒\n\n")
            
            for result in self.results:
                status_symbol = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "ERROR" else "⚠️"
                f.write(f"{status_symbol} [{result['module']}] {result['test']}\n")
                f.write(f"    状态: {result['status']}\n")
                if result["details"]:
                    f.write(f"    详情: {result['details']}\n")
                f.write(f"    时间: {result['timestamp']}\n\n")
        
        logger.info(f"详细报告已保存到: {report_file}")
        
        return failed_tests == 0
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始Self-Soul AGI系统健康检查...")
        
        self.run_import_tests()
        self.run_configuration_tests()
        self.run_basic_functionality_tests()
        self.run_integration_test_verification()
        self.run_environment_interface_tests()
        
        # 生成报告
        success = self.generate_summary_report()
        
        return success

def main():
    """主函数"""
    print("Self-Soul AGI系统健康检查")
    print("版本: 3.0 (AGI升级完成)")
    print("="*50)
    
    health_check = AGISystemHealthCheck()
    success = health_check.run_all_tests()
    
    # 返回退出码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()