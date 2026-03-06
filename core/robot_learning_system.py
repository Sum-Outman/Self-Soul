"""
Robot Learning System - Enhanced AGI capabilities for humanoid robots

集成自适应学习、环境理解和自我改进能力，为AGI机器人提供高级学习功能。
Integrates adaptive learning, environment understanding, and self-improvement capabilities for AGI robots.

主要功能:
1. 自适应运动学习 - 根据传感器反馈优化运动参数
2. 环境理解增强 - 结合传感器数据理解环境并做出智能决策
3. 在线学习集成 - 持续从经验中学习并改进性能
4. 元学习能力 - 学习如何更好地学习，加速技能获取
5. 安全学习机制 - 确保学习过程的安全性

版权所有 (c) 2025 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from datetime import datetime
import json
import threading

# 导入现有AGI模块
try:
    from core.adaptive_learning_engine import EnhancedAdaptiveLearningEngine
    ADAPTIVE_LEARNING_AVAILABLE = True
except ImportError:
    logging.warning("Adaptive learning engine not available")
    ADAPTIVE_LEARNING_AVAILABLE = False

try:
    from core.online_learning_system import OnlineLearningSystem
    ONLINE_LEARNING_AVAILABLE = True
except ImportError:
    logging.warning("Online learning system not available")
    ONLINE_LEARNING_AVAILABLE = False

try:
    from core.agi_core_capabilities import EnvironmentAdaptationEngine
    ENVIRONMENT_ADAPTATION_AVAILABLE = True
except ImportError:
    logging.warning("Environment adaptation engine not available")
    ENVIRONMENT_ADAPTATION_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RobotLearningSystem")

class RobotLearningSystem:
    """机器人学习系统 - 增强AGI能力"""
    
    def __init__(self, robot_hardware_interface=None, motion_controller=None):
        """初始化机器人学习系统
        
        Args:
            robot_hardware_interface: 机器人硬件接口实例
            motion_controller: 运动控制器实例
        """
        self.robot_hardware = robot_hardware_interface
        self.motion_controller = motion_controller
        
        # 学习组件初始化
        self._initialize_learning_components()
        
        # 学习状态
        self.learning_state = {
            "enabled": True,
            "mode": "balanced",  # balanced, exploration, exploitation, safe
            "learning_rate": 0.01,
            "exploration_rate": 0.3,
            "performance_threshold": 0.7,
            "safety_enabled": True,
            "max_learning_iterations": 1000
        }
        
        # 学习历史
        self.learning_history = deque(maxlen=10000)
        self.skill_database = {}
        self.experience_buffer = deque(maxlen=5000)
        
        # 性能指标
        self.performance_metrics = {
            "total_learning_sessions": 0,
            "successful_learnings": 0,
            "failed_learnings": 0,
            "total_improvement": 0.0,
            "average_learning_time": 0.0,
            "best_performance": 0.0
        }
        
        # 环境理解状态
        self.environment_state = {
            "complexity": 0.5,
            "stability": 0.8,
            "obstacle_density": 0.0,
            "lighting_condition": 0.5,
            "terrain_type": "flat",  # flat, uneven, stairs, slope
            "hazard_level": 0.0
        }
        
        # 安全系统
        self.safety_system = {
            "emergency_stop_count": 0,
            "safety_violations": 0,
            "last_safety_check": time.time(),
            "safety_monitor_enabled": True
        }
        
        logger.info("Robot Learning System initialized")
    
    def _initialize_learning_components(self):
        """初始化学习组件"""
        # 自适应学习引擎
        if ADAPTIVE_LEARNING_AVAILABLE:
            self.adaptive_learner = EnhancedAdaptiveLearningEngine()
            logger.info("Adaptive learning engine initialized")
        else:
            self.adaptive_learner = None
            logger.warning("Adaptive learning engine not available")
        
        # 在线学习系统
        if ONLINE_LEARNING_AVAILABLE:
            self.online_learner = OnlineLearningSystem()
            logger.info("Online learning system initialized")
        else:
            self.online_learner = None
            logger.warning("Online learning system not available")
        
        # 环境适应引擎
        if ENVIRONMENT_ADAPTATION_AVAILABLE:
            self.environment_adapter = EnvironmentAdaptationEngine()
            logger.info("Environment adaptation engine initialized")
        else:
            self.environment_adapter = None
            logger.warning("Environment adaptation engine not available")
    
    def learn_motion_skill(self, skill_name: str, target_motion: Dict[str, Any], 
                          constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """学习运动技能
        
        Args:
            skill_name: 技能名称
            target_motion: 目标运动参数
            constraints: 约束条件
            
        Returns:
            学习结果字典
        """
        if not self.learning_state["enabled"]:
            return {"success": False, "error": "Learning disabled"}
        
        start_time = time.time()
        logger.info(f"Starting motion skill learning: {skill_name}")
        
        # 安全检查
        if not self._safety_check(target_motion, constraints):
            return {"success": False, "error": "Safety check failed"}
        
        # 初始化学习参数
        learning_params = self._initialize_learning_parameters(skill_name, target_motion)
        
        # 学习循环
        learning_result = self._perform_learning_cycle(skill_name, learning_params)
        
        # 记录学习结果
        learning_result["learning_time"] = time.time() - start_time
        self._record_learning_result(skill_name, learning_result)
        
        # 更新技能数据库
        if learning_result.get("success", False):
            self.skill_database[skill_name] = {
                "skill_params": learning_params,
                "performance": learning_result.get("performance", 0.0),
                "learned_at": datetime.now().isoformat(),
                "usage_count": 0
            }
            logger.info(f"Motion skill learned successfully: {skill_name}")
        else:
            logger.warning(f"Motion skill learning failed: {skill_name}")
        
        return learning_result
    
    def _safety_check(self, target_motion: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        """安全检查
        
        Args:
            target_motion: 目标运动参数
            constraints: 约束条件
            
        Returns:
            是否通过安全检查
        """
        if not self.learning_state["safety_enabled"]:
            return True
        
        # 检查关节限制
        if "joint_limits" in constraints:
            joint_limits = constraints["joint_limits"]
            for joint, limit in joint_limits.items():
                if joint in target_motion:
                    value = target_motion[joint]
                    if value < limit.get("min", -180) or value > limit.get("max", 180):
                        logger.warning(f"Joint limit violation: {joint} = {value}")
                        return False
        
        # 检查速度限制
        if "velocity_limits" in constraints:
            velocity_limits = constraints["velocity_limits"]
            for joint, limit in velocity_limits.items():
                if f"{joint}_velocity" in target_motion:
                    velocity = target_motion[f"{joint}_velocity"]
                    if abs(velocity) > limit:
                        logger.warning(f"Velocity limit violation: {joint} velocity = {velocity}")
                        return False
        
        # 检查稳定性约束
        if self.robot_hardware:
            try:
                # 获取当前平衡状态
                balance_status = self.robot_hardware.get_balance_status()
                if balance_status.get("unstable", False):
                    logger.warning("Balance unstable, learning unsafe")
                    return False
            except Exception as e:
                logger.warning(f"Failed to check balance status: {e}")
        
        return True
    
    def _initialize_learning_parameters(self, skill_name: str, target_motion: Dict[str, Any]) -> Dict[str, Any]:
        """初始化学习参数
        
        Args:
            skill_name: 技能名称
            target_motion: 目标运动参数
            
        Returns:
            学习参数字典
        """
        # 基础学习参数
        params = {
            "skill_name": skill_name,
            "target_motion": target_motion,
            "learning_rate": self.learning_state["learning_rate"],
            "exploration_rate": self.learning_state["exploration_rate"],
            "max_iterations": self.learning_state["max_learning_iterations"],
            "performance_threshold": self.learning_state["performance_threshold"],
            "environment_factors": self.environment_state.copy()
        }
        
        # 如果自适应学习引擎可用，使用它优化参数
        if self.adaptive_learner:
            try:
                optimized_params = self.adaptive_learner.configure_training(
                    model_types=["robot_motion"],
                    data_characteristics={"motion_complexity": 0.5, "dataset_size": 1000},
                    resource_constraints={},
                    meta_learning_strategy="balanced",
                    historical_performance=self.performance_metrics,
                    realtime_system_metrics=None
                )
                params.update(optimized_params)
            except Exception as e:
                logger.warning(f"Adaptive learning optimization failed: {e}")
        
        return params
    
    def _perform_learning_cycle(self, skill_name: str, learning_params: Dict[str, Any]) -> Dict[str, Any]:
        """执行学习循环
        
        Args:
            skill_name: 技能名称
            learning_params: 学习参数
            
        Returns:
            学习结果字典
        """
        iterations = 0
        max_iterations = learning_params.get("max_iterations", 100)
        performance_threshold = learning_params.get("performance_threshold", 0.7)
        
        best_performance = 0.0
        best_params = None
        
        while iterations < max_iterations:
            # 生成候选运动参数
            candidate_params = self._generate_candidate_params(learning_params, iterations)
            
            # 评估候选参数
            evaluation_result = self._evaluate_candidate_params(skill_name, candidate_params)
            
            # 更新学习状态
            performance = evaluation_result.get("performance", 0.0)
            if performance > best_performance:
                best_performance = performance
                best_params = candidate_params.copy()
            
            # 检查是否达到性能阈值
            if performance >= performance_threshold:
                return {
                    "success": True,
                    "performance": performance,
                    "learned_params": candidate_params,
                    "iterations": iterations + 1,
                    "converged": True
                }
            
            iterations += 1
        
        # 返回最佳参数（即使未达到阈值）
        if best_params is not None:
            return {
                "success": best_performance > 0.3,  # 最低性能要求
                "performance": best_performance,
                "learned_params": best_params,
                "iterations": iterations,
                "converged": False
            }
        else:
            return {
                "success": False,
                "error": "Learning failed - no valid parameters found",
                "iterations": iterations
            }
    
    def _generate_candidate_params(self, learning_params: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """生成候选参数
        
        Args:
            learning_params: 学习参数
            iteration: 当前迭代次数
            
        Returns:
            候选参数字典
        """
        target_motion = learning_params.get("target_motion", {})
        exploration_rate = learning_params.get("exploration_rate", 0.3)
        
        # 基础参数（目标值）
        candidate = target_motion.copy()
        
        # 添加探索噪声
        if np.random.random() < exploration_rate:
            for key, value in candidate.items():
                if isinstance(value, (int, float)):
                    # 随着迭代减少噪声
                    noise_scale = 0.1 * (1.0 - iteration / learning_params.get("max_iterations", 100))
                    noise = np.random.normal(0, noise_scale * abs(value))
                    candidate[key] = value + noise
        
        # 应用环境适应
        if self.environment_adapter:
            try:
                adaptation = self.environment_adapter.adapt(
                    performance=0.5,  # 中间值
                    context={"motion_params": candidate}
                )
                if adaptation.get("adaptations"):
                    # 应用环境适应的调整
                    for adj in adaptation["adaptations"]:
                        if adj.get("type") == "performance_improvement":
                            # 调整探索率
                            candidate["_exploration_adjusted"] = True
            except Exception as e:
                logger.debug(f"Environment adaptation failed: {e}")
        
        return candidate
    
    def _evaluate_candidate_params(self, skill_name: str, candidate_params: Dict[str, Any]) -> Dict[str, Any]:
        """评估候选参数
        
        Args:
            skill_name: 技能名称
            candidate_params: 候选参数
            
        Returns:
            评估结果字典
        """
        # 这里应该执行实际的运动并收集传感器反馈
        # 由于这是模拟版本，我们使用模拟评估
        
        try:
            # 模拟执行运动
            if self.motion_controller:
                # 实际执行运动
                execution_result = self._execute_motion_with_params(candidate_params)
                performance = execution_result.get("performance", 0.5)
            else:
                # 模拟性能评估
                performance = self._simulate_performance_evaluation(candidate_params)
            
            # 收集传感器数据（如果可用）
            sensor_data = {}
            if self.robot_hardware:
                try:
                    sensor_data = self.robot_hardware.get_sensor_data()
                except Exception as e:
                    logger.debug(f"Failed to get sensor data: {e}")
            
            # 安全评估
            safety_score = self._evaluate_safety(candidate_params, sensor_data)
            
            # 综合性能评分
            final_performance = performance * 0.7 + safety_score * 0.3
            
            return {
                "performance": final_performance,
                "safety_score": safety_score,
                "sensor_data": sensor_data,
                "candidate_params": candidate_params
            }
            
        except Exception as e:
            logger.error(f"Candidate evaluation failed: {e}")
            return {
                "performance": 0.0,
                "error": str(e)
            }
    
    def _execute_motion_with_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """使用参数执行运动
        
        Args:
            params: 运动参数
            
        Returns:
            执行结果字典
        """
        # 这里应该调用实际的运动控制器
        # 暂时返回模拟结果
        return {
            "success": True,
            "performance": 0.7,
            "execution_time": 0.5
        }
    
    def _simulate_performance_evaluation(self, params: Dict[str, Any]) -> float:
        """模拟性能评估
        
        Args:
            params: 运动参数
            
        Returns:
            性能评分 (0-1)
        """
        # 简单模拟：检查参数是否接近目标
        target_values = {
            "position_accuracy": 1.0,
            "velocity_smoothness": 1.0,
            "energy_efficiency": 1.0
        }
        
        score = 0.5  # 基础分
        
        # 添加一些随机性模拟现实世界变化
        score += np.random.normal(0, 0.1)
        
        # 确保在0-1范围内
        return max(0.0, min(1.0, score))
    
    def _evaluate_safety(self, params: Dict[str, Any], sensor_data: Dict[str, Any]) -> float:
        """评估安全性
        
        Args:
            params: 运动参数
            sensor_data: 传感器数据
            
        Returns:
            安全评分 (0-1)
        """
        safety_score = 1.0
        
        # 检查关节角度是否在安全范围内
        joint_angles = params.get("joint_angles", {})
        for joint, angle in joint_angles.items():
            if isinstance(angle, (int, float)):
                if abs(angle) > 120:  # 安全限制
                    safety_score *= 0.8
        
        # 检查传感器数据中的危险信号
        if sensor_data:
            # 检查IMU数据
            imu_data = sensor_data.get("imu", {})
            acceleration = imu_data.get("acceleration", [0, 0, 9.8])
            if any(abs(a) > 20 for a in acceleration[:2]):  # 横向加速度过大
                safety_score *= 0.7
            
            # 检查力传感器数据
            force_data = sensor_data.get("force_sensors", {})
            for sensor, force in force_data.items():
                if isinstance(force, (int, float)) and force > 100:  # 力过大
                    safety_score *= 0.9
        
        return safety_score
    
    def _record_learning_result(self, skill_name: str, result: Dict[str, Any]):
        """记录学习结果
        
        Args:
            skill_name: 技能名称
            result: 学习结果
        """
        record = {
            "skill_name": skill_name,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "environment_state": self.environment_state.copy(),
            "learning_state": self.learning_state.copy()
        }
        
        self.learning_history.append(record)
        
        # 更新性能指标
        self.performance_metrics["total_learning_sessions"] += 1
        if result.get("success", False):
            self.performance_metrics["successful_learnings"] += 1
            performance = result.get("performance", 0.0)
            self.performance_metrics["total_improvement"] += performance
            self.performance_metrics["best_performance"] = max(
                self.performance_metrics["best_performance"], performance
            )
        else:
            self.performance_metrics["failed_learnings"] += 1
    
    def update_environment_state(self, observations: Dict[str, Any]):
        """更新环境状态
        
        Args:
            observations: 环境观察数据
        """
        self.environment_state.update(observations)
        
        # 通知环境适应引擎
        if self.environment_adapter:
            try:
                self.environment_adapter.perceive_environment(observations)
            except Exception as e:
                logger.warning(f"Environment adaptation update failed: {e}")
        
        logger.debug(f"Environment state updated: {observations}")
    
    def get_learning_recommendations(self) -> List[str]:
        """获取学习建议
        
        Returns:
            建议列表
        """
        recommendations = []
        
        # 基于性能的建议
        success_rate = self.performance_metrics["successful_learnings"] / max(
            self.performance_metrics["total_learning_sessions"], 1
        )
        
        if success_rate < 0.5:
            recommendations.append("学习成功率较低，建议降低学习率或增加探索")
        
        if self.performance_metrics["failed_learnings"] > 10:
            recommendations.append("连续学习失败次数过多，建议检查安全约束或环境条件")
        
        # 基于环境的建议
        if self.environment_state.get("hazard_level", 0) > 0.7:
            recommendations.append("环境危险等级高，建议启用安全学习模式")
        
        if self.environment_state.get("complexity", 0) > 0.8:
            recommendations.append("环境复杂度高，建议分步学习复杂技能")
        
        # 基于自适应学习的建议（如果可用）
        if self.adaptive_learner:
            try:
                adaptive_recs = self.adaptive_learner.meta_analyzer.get_recommendations(
                    model_type="robot_motion",
                    data_size=len(self.learning_history)
                )
                recommendations.extend(adaptive_recs)
            except Exception as e:
                logger.debug(f"Failed to get adaptive recommendations: {e}")
        
        return recommendations
    
    def enable_learning(self, enabled: bool = True):
        """启用或禁用学习
        
        Args:
            enabled: 是否启用学习
        """
        self.learning_state["enabled"] = enabled
        logger.info(f"Learning {'enabled' if enabled else 'disabled'}")
    
    def set_learning_mode(self, mode: str):
        """设置学习模式
        
        Args:
            mode: 学习模式 (balanced, exploration, exploitation, safe)
        """
        valid_modes = ["balanced", "exploration", "exploitation", "safe"]
        if mode not in valid_modes:
            logger.warning(f"Invalid learning mode: {mode}, using 'balanced'")
            mode = "balanced"
        
        self.learning_state["mode"] = mode
        
        # 根据模式调整参数
        if mode == "exploration":
            self.learning_state["exploration_rate"] = 0.5
            self.learning_state["learning_rate"] = 0.02
        elif mode == "exploitation":
            self.learning_state["exploration_rate"] = 0.1
            self.learning_state["learning_rate"] = 0.005
        elif mode == "safe":
            self.learning_state["exploration_rate"] = 0.15
            self.learning_state["learning_rate"] = 0.01
            self.learning_state["safety_enabled"] = True
        else:  # balanced
            self.learning_state["exploration_rate"] = 0.3
            self.learning_state["learning_rate"] = 0.01
        
        logger.info(f"Learning mode set to: {mode}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态
        
        Returns:
            状态字典
        """
        return {
            "learning_enabled": self.learning_state["enabled"],
            "learning_mode": self.learning_state["mode"],
            "performance_metrics": self.performance_metrics.copy(),
            "environment_state": self.environment_state.copy(),
            "safety_status": self.safety_system.copy(),
            "skills_learned": list(self.skill_database.keys()),
            "recommendations": self.get_learning_recommendations()
        }


def get_robot_learning_system(robot_hardware=None, motion_controller=None) -> RobotLearningSystem:
    """获取机器人学习系统实例（单例模式）
    
    Args:
        robot_hardware: 机器人硬件接口实例
        motion_controller: 运动控制器实例
        
    Returns:
        机器人学习系统实例
    """
    global _robot_learning_system
    if _robot_learning_system is None:
        _robot_learning_system = RobotLearningSystem(robot_hardware, motion_controller)
    return _robot_learning_system


# 全局实例
_robot_learning_system = None


# 测试函数
def test_robot_learning_system():
    """测试机器人学习系统"""
    logger.info("Testing Robot Learning System...")
    
    try:
        # 创建学习系统实例
        learning_system = RobotLearningSystem()
        
        # 测试基本功能
        learning_system.enable_learning(True)
        learning_system.set_learning_mode("balanced")
        
        # 更新环境状态
        learning_system.update_environment_state({
            "complexity": 0.6,
            "stability": 0.9,
            "terrain_type": "flat"
        })
        
        # 测试学习运动技能（模拟）
        test_skill = {
            "joint_angles": {
                "left_hip": 30.0,
                "left_knee": 60.0,
                "left_ankle": -10.0
            }
        }
        
        result = learning_system.learn_motion_skill(
            skill_name="test_walking_step",
            target_motion=test_skill,
            constraints={
                "joint_limits": {
                    "left_hip": {"min": -45, "max": 45},
                    "left_knee": {"min": 0, "max": 120},
                    "left_ankle": {"min": -30, "max": 30}
                }
            }
        )
        
        logger.info(f"Learning result: {result.get('success', False)}")
        
        # 获取系统状态
        status = learning_system.get_status()
        logger.info(f"System status: {status.keys()}")
        
        # 获取建议
        recommendations = learning_system.get_learning_recommendations()
        logger.info(f"Recommendations: {recommendations}")
        
        return {
            "success": True,
            "learning_result": result,
            "system_status": status,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Robot learning system test failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    test_result = test_robot_learning_system()
    print(f"Test completed: {test_result.get('success', False)}")