#!/usr/bin/env python3
"""
环境接口测试脚本

测试环境交互接口的核心功能：
1. 接口初始化与连接
2. 动作执行与验证
3. 观察获取与处理
4. 奖励计算与反馈
5. 状态管理与维护
"""

import sys
import os
import time
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnvironmentInterfaceTest")

# 导入环境接口
from core.environment_interface import (
    Action, ActionType, Observation, ObservationType, Reward, EnvironmentState,
    SimulatedEnvironment, UnifiedEnvironmentManager, initialize_environment_system
)


class EnvironmentInterfaceTest:
    """环境接口测试类"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始环境接口测试")
        
        # 测试1：接口导入测试
        self.test_results['import_tests'] = self.run_import_tests()
        
        # 测试2：模拟环境基础测试
        if self.test_results['import_tests']:
            self.test_results['simulated_environment_tests'] = self.run_simulated_environment_tests()
        else:
            self.test_results['simulated_environment_tests'] = False
            logger.warning("跳过模拟环境测试（导入测试失败）")
        
        # 测试3：统一管理器测试
        if self.test_results['import_tests']:
            self.test_results['unified_manager_tests'] = self.run_unified_manager_tests()
        else:
            self.test_results['unified_manager_tests'] = False
            logger.warning("跳过统一管理器测试（导入测试失败）")
        
        # 测试4：完整工作流程测试
        if all([
            self.test_results['import_tests'],
            self.test_results['simulated_environment_tests'],
            self.test_results['unified_manager_tests']
        ]):
            self.test_results['complete_workflow_tests'] = self.run_complete_workflow_tests()
        else:
            self.test_results['complete_workflow_tests'] = False
            logger.warning("跳过完整工作流程测试（部分前置测试失败）")
        
        # 生成测试报告
        self.generate_test_report()
        
        return self.test_results
    
    def run_import_tests(self) -> bool:
        """运行导入测试"""
        logger.info("运行导入测试...")
        
        try:
            # 测试类导入
            from core.environment_interface import Action, ActionType
            from core.environment_interface import Observation, ObservationType
            from core.environment_interface import Reward, EnvironmentState
            from core.environment_interface import SimulatedEnvironment, UnifiedEnvironmentManager
            
            logger.info("✅ 所有环境接口类导入成功")
            return True
            
        except ImportError as e:
            logger.error(f"❌ 导入测试失败: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ 导入测试出现意外错误: {e}")
            return False
    
    def run_simulated_environment_tests(self) -> bool:
        """运行模拟环境测试"""
        logger.info("运行模拟环境测试...")
        
        try:
            # 创建模拟环境
            env = SimulatedEnvironment({
                "simulation_mode": "test",
                "max_objects": 5,
                "enable_rewards": True
            })
            
            # 测试连接
            connected = env.connect()
            if not connected:
                logger.error("❌ 模拟环境连接失败")
                return False
            logger.info("✅ 模拟环境连接成功")
            
            # 测试获取能力信息
            capabilities = env.get_capabilities()
            if not capabilities:
                logger.error("❌ 获取能力信息失败")
                return False
            logger.info(f"✅ 获取能力信息成功: {capabilities['interface_name']}")
            
            # 测试动作执行
            action = Action(
                action_id="test_move_forward",
                action_type=ActionType.PHYSICAL,
                parameters={"move": True, "direction": "forward", "distance": 1.0},
                priority=1,
                timeout_seconds=5.0
            )
            
            success, error_msg = env.execute_action(action)
            if not success:
                logger.error(f"❌ 动作执行失败: {error_msg}")
                return False
            logger.info(f"✅ 动作执行成功: {error_msg}")
            
            # 测试观察获取
            observations = env.get_observation()
            if not observations:
                logger.error("❌ 获取观察失败")
                return False
            logger.info(f"✅ 获取观察成功，共 {len(observations)} 个观察")
            
            # 测试奖励获取
            rewards = env.get_reward()
            logger.info(f"✅ 获取奖励成功，共 {len(rewards)} 个奖励")
            
            # 测试状态获取
            state = env.get_state()
            if not state:
                logger.error("❌ 获取状态失败")
                return False
            logger.info(f"✅ 获取状态成功: {state.state_id}")
            
            # 测试重置
            reset_success = env.reset()
            if not reset_success:
                logger.error("❌ 环境重置失败")
                return False
            logger.info("✅ 环境重置成功")
            
            # 测试断开连接
            disconnected = env.disconnect()
            if not disconnected:
                logger.error("❌ 断开连接失败")
                return False
            logger.info("✅ 断开连接成功")
            
            # 获取统计信息
            stats = env.get_statistics()
            logger.info(f"✅ 获取统计信息成功: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 模拟环境测试出现意外错误: {e}")
            return False
    
    def run_unified_manager_tests(self) -> bool:
        """运行统一管理器测试"""
        logger.info("运行统一管理器测试...")
        
        try:
            # 创建管理器
            manager = UnifiedEnvironmentManager()
            
            # 创建并注册模拟环境
            env1 = SimulatedEnvironment({"interface_name": "test_env_1"})
            env2 = SimulatedEnvironment({"interface_name": "test_env_2"})
            
            # 测试注册接口
            reg1 = manager.register_interface(env1)
            reg2 = manager.register_interface(env2)
            
            if not reg1 or not reg2:
                logger.error("❌ 接口注册失败")
                return False
            logger.info("✅ 接口注册成功")
            
            # 测试设置活跃接口
            set_active = manager.set_active_interface("test_env_1")
            if not set_active:
                logger.error("❌ 设置活跃接口失败")
                return False
            logger.info("✅ 设置活跃接口成功")
            
            # 测试获取活跃接口
            active_interface = manager.get_active_interface()
            if not active_interface:
                logger.error("❌ 获取活跃接口失败")
                return False
            logger.info(f"✅ 获取活跃接口成功: {active_interface.interface_name}")
            
            # 测试连接所有接口
            connection_results = manager.connect_all()
            if not all(connection_results.values()):
                logger.error(f"❌ 连接所有接口失败: {connection_results}")
                return False
            logger.info(f"✅ 连接所有接口成功: {connection_results}")
            
            # 测试执行动作
            action = Action(
                action_id="manager_test_action",
                action_type=ActionType.PHYSICAL,
                parameters={"move": True, "direction": "forward", "distance": 2.0},
                priority=1,
                timeout_seconds=5.0
            )
            
            success, error_msg, observations, rewards = manager.execute_action(action)
            if not success:
                logger.error(f"❌ 通过管理器执行动作失败: {error_msg}")
                return False
            logger.info(f"✅ 通过管理器执行动作成功: {error_msg}")
            
            # 测试获取观察
            observations = manager.get_observation()
            logger.info(f"✅ 通过管理器获取观察成功，共 {len(observations)} 个观察")
            
            # 测试获取统计信息
            all_stats = manager.get_all_statistics()
            logger.info(f"✅ 获取所有统计信息成功，共 {len(all_stats)} 个接口")
            
            # 测试获取能力信息
            all_capabilities = manager.get_all_capabilities()
            logger.info(f"✅ 获取所有能力信息成功，共 {len(all_capabilities)} 个接口")
            
            # 测试断开所有接口
            disconnect_results = manager.disconnect_all()
            if not all(disconnect_results.values()):
                logger.warning(f"⚠️ 断开所有接口部分失败: {disconnect_results}")
            else:
                logger.info(f"✅ 断开所有接口成功: {disconnect_results}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 统一管理器测试出现意外错误: {e}")
            return False
    
    def run_complete_workflow_tests(self) -> bool:
        """运行完整工作流程测试"""
        logger.info("运行完整工作流程测试...")
        
        try:
            # 使用全局初始化函数
            manager = initialize_environment_system()
            
            # 测试完整交互循环
            test_steps = 5
            total_reward = 0.0
            
            for step in range(test_steps):
                logger.info(f"=== 交互步骤 {step + 1}/{test_steps} ===")
                
                # 创建动作
                if step % 2 == 0:
                    action = Action(
                        action_id=f"step_{step}_move",
                        action_type=ActionType.PHYSICAL,
                        parameters={"move": True, "direction": "forward", "distance": 1.0},
                        priority=1,
                        timeout_seconds=5.0
                    )
                else:
                    action = Action(
                        action_id=f"step_{step}_rotate",
                        action_type=ActionType.PHYSICAL,
                        parameters={"rotate": True, "axis": "yaw", "angle": 45.0},
                        priority=1,
                        timeout_seconds=5.0
                    )
                
                # 执行动作
                success, error_msg, observations, rewards = manager.execute_action(action)
                
                if not success:
                    logger.warning(f"⚠️ 步骤 {step + 1} 动作执行失败: {error_msg}")
                else:
                    logger.info(f"✅ 步骤 {step + 1} 动作执行成功: {error_msg}")
                
                # 记录观察
                logger.info(f"  获得 {len(observations)} 个观察，{len(rewards)} 个奖励")
                
                # 累计奖励
                step_reward = sum(r.value for r in rewards)
                total_reward += step_reward
                logger.info(f"  步骤奖励: {step_reward:.2f}, 累计奖励: {total_reward:.2f}")
                
                # 短暂等待
                time.sleep(0.1)
            
            # 获取最终统计信息
            stats = manager.get_all_statistics()
            logger.info(f"✅ 完整工作流程完成，总奖励: {total_reward:.2f}")
            logger.info(f"最终统计信息: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 完整工作流程测试出现意外错误: {e}")
            return False
    
    def generate_test_report(self):
        """生成测试报告"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        duration = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("环境接口测试报告")
        print("="*80)
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"测试时长: {duration:.2f}秒")
        print(f"总测试数: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"失败测试: {failed_tests}")
        print(f"通过率: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "通过率: N/A")
        
        # 详细测试结果
        print("\n详细测试结果:")
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {status}: {test_name}")
        
        if passed_tests == total_tests:
            print("\n🎉 所有环境接口测试通过！")
        else:
            print(f"\n⚠️ 环境接口测试存在 {failed_tests} 个失败项，需要修复。")
        
        print("="*80)
        
        # 保存测试报告
        report_file = f"environment_interface_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("环境接口测试报告\n")
            f.write("="*80 + "\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试时长: {duration:.2f}秒\n")
            f.write(f"总测试数: {total_tests}\n")
            f.write(f"通过测试: {passed_tests}\n")
            f.write(f"失败测试: {failed_tests}\n")
            f.write(f"通过率: {passed_tests/total_tests*100:.1f}%\n\n")
            
            f.write("详细测试结果:\n")
            for test_name, result in self.test_results.items():
                status = "✅ 通过" if result else "❌ 失败"
                f.write(f"{status}: {test_name}\n")
        
        logger.info(f"测试报告已保存到: {report_file}")


def main():
    """主函数"""
    print("环境接口测试")
    print("版本: 1.0")
    print("="*50)
    
    test = EnvironmentInterfaceTest()
    results = test.run_all_tests()
    
    # 返回退出码
    if all(results.values()):
        print("\n✅ 所有测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()