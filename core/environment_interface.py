#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境交互接口 - 统一的环境动作执行和观察获取接口

核心功能:
1. 动作执行: 将抽象动作转换为具体环境操作
2. 观察获取: 从环境获取感知输入
3. 奖励处理: 接收环境反馈奖励
4. 状态维护: 跟踪环境状态变化
5. 交互循环: 实现标准的"感知→决策→行动→学习"循环

设计原则:
1. 统一接口: 为不同环境提供一致的操作接口
2. 抽象层次: 支持从低级物理动作到高级抽象任务
3. 模块化: 易于集成到现有认知架构中
4. 可扩展: 支持新环境类型和传感器

接口层次:
1. 物理层接口: 直接硬件控制（机器人、传感器等）
2. 模拟层接口: 虚拟环境交互（游戏、仿真等）
3. 抽象层接口: 符号化环境操作（任务规划、知识操作等）

集成点:
1. 世界状态表示: 更新环境状态
2. 分层规划系统: 执行规划动作
3. 自主探索: 指导探索行为
4. 学习系统: 从交互经验中学习

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# 导入错误处理
from core.error_handling import ErrorHandler

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()


class ActionType(Enum):
    """动作类型枚举"""
    PHYSICAL = "physical"          # 物理动作（移动、操作等）
    SENSORY = "sensory"            # 感知动作（观察、测量等）
    COMMUNICATION = "communication" # 通信动作（说话、发送消息等）
    COGNITIVE = "cognitive"        # 认知动作（思考、推理等）
    ABSTRACT = "abstract"          # 抽象动作（执行计划、完成任务等）


class ObservationType(Enum):
    """观察类型枚举"""
    VISUAL = "visual"              # 视觉观察（图像、视频等）
    AUDITORY = "auditory"          # 听觉观察（声音、语音等）
    TACTILE = "tactile"            # 触觉观察（触摸、压力等）
    PROPRIOCEPTIVE = "proprioceptive" # 本体感知（位置、姿态等）
    SEMANTIC = "semantic"          # 语义观察（文本、概念等）
    NUMERIC = "numeric"            # 数值观察（测量值、状态值等）


@dataclass
class Action:
    """动作数据类"""
    action_id: str
    action_type: ActionType
    parameters: Dict[str, Any]
    priority: int = 1
    timeout_seconds: float = 10.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """观察数据类"""
    observation_id: str
    observation_type: ObservationType
    data: Any
    timestamp: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Reward:
    """奖励数据类"""
    reward_id: str
    value: float
    source: str
    timestamp: float
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentState:
    """环境状态数据类"""
    state_id: str
    state_type: str
    variables: Dict[str, Any]
    timestamp: float
    uncertainty: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnvironmentInterface(ABC):
    """
    环境交互接口抽象基类
    
    所有具体环境实现都应继承此类
    """
    
    def __init__(self, interface_name: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化环境接口
        
        Args:
            interface_name: 接口名称
            config: 配置参数
        """
        self.interface_name = interface_name
        self.config = config or {}
        self.is_connected = False
        self.last_interaction_time = time.time()
        self.interaction_count = 0
        self.error_count = 0
        
        # 统计信息
        self.stats = {
            "total_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "total_observations": 0,
            "total_rewards": 0,
            "total_interaction_time": 0.0
        }
        
        logger.info(f"初始化环境接口: {interface_name}")
    
    @abstractmethod
    def connect(self) -> bool:
        """
        连接到环境
        
        Returns:
            连接是否成功
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开与环境连接
        
        Returns:
            断开是否成功
        """
        pass
    
    @abstractmethod
    def execute_action(self, action: Action) -> Tuple[bool, Optional[str]]:
        """
        执行动作
        
        Args:
            action: 要执行的动作
            
        Returns:
            (成功标志, 错误信息)
        """
        pass
    
    @abstractmethod
    def get_observation(self, observation_type: Optional[ObservationType] = None) -> List[Observation]:
        """
        获取观察
        
        Args:
            observation_type: 观察类型，为None时获取所有可用观察
            
        Returns:
            观察列表
        """
        pass
    
    @abstractmethod
    def get_reward(self) -> List[Reward]:
        """
        获取奖励
        
        Returns:
            奖励列表
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Optional[EnvironmentState]:
        """
        获取环境状态
        
        Returns:
            环境状态，如果不可用则返回None
        """
        pass
    
    @abstractmethod
    def reset(self) -> bool:
        """
        重置环境
        
        Returns:
            重置是否成功
        """
        pass
    
    def validate_action(self, action: Action) -> Tuple[bool, List[str]]:
        """
        验证动作的可行性
        
        Args:
            action: 要验证的动作
            
        Returns:
            (是否有效, 错误消息列表)
        """
        errors = []
        
        # 检查必要字段
        if not action.action_id:
            errors.append("动作ID不能为空")
        
        if not action.action_type:
            errors.append("动作类型不能为空")
        
        # 检查超时设置
        if action.timeout_seconds <= 0:
            errors.append("超时时间必须为正数")
        
        # 检查优先级
        if action.priority < 1:
            errors.append("优先级必须至少为1")
        
        return (len(errors) == 0, errors)
    
    def log_interaction(self, action: Optional[Action] = None, 
                       observations: Optional[List[Observation]] = None,
                       rewards: Optional[List[Reward]] = None,
                       success: bool = True):
        """
        记录交互信息
        
        Args:
            action: 执行的动作
            observations: 获取的观察
            rewards: 获得的奖励
            success: 是否成功
        """
        self.interaction_count += 1
        self.last_interaction_time = time.time()
        
        if action:
            self.stats["total_actions"] += 1
            if success:
                self.stats["successful_actions"] += 1
            else:
                self.stats["failed_actions"] += 1
                self.error_count += 1
        
        if observations:
            self.stats["total_observations"] += len(observations)
        
        if rewards:
            self.stats["total_rewards"] += len(rewards)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取接口统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.stats.copy()
        stats.update({
            "interface_name": self.interface_name,
            "is_connected": self.is_connected,
            "interaction_count": self.interaction_count,
            "error_count": self.error_count,
            "last_interaction_time": self.last_interaction_time,
            "success_rate": (self.stats["successful_actions"] / self.stats["total_actions"] 
                           if self.stats["total_actions"] > 0 else 0.0)
        })
        return stats
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        获取接口能力信息
        
        Returns:
            能力信息字典
        """
        return {
            "interface_name": self.interface_name,
            "supported_action_types": [action_type.value for action_type in ActionType],
            "supported_observation_types": [obs_type.value for obs_type in ObservationType],
            "configurable_parameters": list(self.config.keys()),
            "max_concurrent_actions": 1,  # 默认支持单个动作
            "real_time_capable": False,   # 默认非实时
            "supports_reset": True        # 默认支持重置
        }


class UnifiedEnvironmentManager:
    """
    统一环境管理器
    
    管理多个环境接口，提供统一的交互入口
    """
    
    def __init__(self):
        """初始化统一环境管理器"""
        self.interfaces: Dict[str, EnvironmentInterface] = {}
        self.active_interface: Optional[str] = None
        self.interaction_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
        logger.info("初始化统一环境管理器")
    
    def register_interface(self, interface: EnvironmentInterface) -> bool:
        """
        注册环境接口
        
        Args:
            interface: 环境接口实例
            
        Returns:
            注册是否成功
        """
        interface_name = interface.interface_name
        
        if interface_name in self.interfaces:
            logger.warning(f"接口 {interface_name} 已注册，将替换")
        
        self.interfaces[interface_name] = interface
        
        # 如果没有活跃接口，设置第一个注册的接口为活跃接口
        if self.active_interface is None:
            self.active_interface = interface_name
        
        logger.info(f"注册环境接口: {interface_name}")
        return True
    
    def set_active_interface(self, interface_name: str) -> bool:
        """
        设置活跃接口
        
        Args:
            interface_name: 接口名称
            
        Returns:
            设置是否成功
        """
        if interface_name not in self.interfaces:
            logger.error(f"接口 {interface_name} 未注册")
            return False
        
        self.active_interface = interface_name
        logger.info(f"设置活跃接口为: {interface_name}")
        return True
    
    def get_active_interface(self) -> Optional[EnvironmentInterface]:
        """
        获取活跃接口
        
        Returns:
            活跃接口实例，如果没有则返回None
        """
        if self.active_interface and self.active_interface in self.interfaces:
            return self.interfaces[self.active_interface]
        return None
    
    def connect_all(self) -> Dict[str, bool]:
        """
        连接所有已注册接口
        
        Returns:
            各接口连接结果字典
        """
        results = {}
        for name, interface in self.interfaces.items():
            try:
                success = interface.connect()
                results[name] = success
                logger.info(f"连接接口 {name}: {'成功' if success else '失败'}")
            except Exception as e:
                results[name] = False
                logger.error(f"连接接口 {name} 时出错: {e}")
        
        return results
    
    def disconnect_all(self) -> Dict[str, bool]:
        """
        断开所有已注册接口
        
        Returns:
            各接口断开结果字典
        """
        results = {}
        for name, interface in self.interfaces.items():
            try:
                success = interface.disconnect()
                results[name] = success
                logger.info(f"断开接口 {name}: {'成功' if success else '失败'}")
            except Exception as e:
                results[name] = False
                logger.error(f"断开接口 {name} 时出错: {e}")
        
        return results
    
    def execute_action(self, action: Action, 
                      interface_name: Optional[str] = None) -> Tuple[bool, Optional[str], List[Observation], List[Reward]]:
        """
        通过指定接口执行动作
        
        Args:
            action: 要执行的动作
            interface_name: 接口名称，为None时使用活跃接口
            
        Returns:
            (成功标志, 错误信息, 观察列表, 奖励列表)
        """
        # 确定使用哪个接口
        if interface_name:
            if interface_name not in self.interfaces:
                return False, f"接口 {interface_name} 未注册", [], []
            interface = self.interfaces[interface_name]
        elif self.active_interface:
            interface = self.interfaces[self.active_interface]
        else:
            return False, "没有可用的活跃接口", [], []
        
        # 验证动作
        is_valid, errors = interface.validate_action(action)
        if not is_valid:
            error_msg = "; ".join(errors)
            return False, f"动作验证失败: {error_msg}", [], []
        
        try:
            # 执行动作
            success, error_msg = interface.execute_action(action)
            
            # 获取观察和奖励
            observations = interface.get_observation()
            rewards = interface.get_reward()
            
            # 记录交互
            interface.log_interaction(action, observations, rewards, success)
            
            # 记录到历史
            self._record_interaction({
                "timestamp": time.time(),
                "interface": interface.interface_name,
                "action": action,
                "success": success,
                "error": error_msg,
                "observations_count": len(observations),
                "rewards_count": len(rewards)
            })
            
            return success, error_msg, observations, rewards
            
        except Exception as e:
            logger.error(f"执行动作时出错: {e}")
            interface.log_interaction(action, None, None, False)
            return False, str(e), [], []
    
    def get_observation(self, interface_name: Optional[str] = None) -> List[Observation]:
        """
        从指定接口获取观察
        
        Args:
            interface_name: 接口名称，为None时使用活跃接口
            
        Returns:
            观察列表
        """
        if interface_name:
            if interface_name not in self.interfaces:
                logger.error(f"接口 {interface_name} 未注册")
                return []
            interface = self.interfaces[interface_name]
        elif self.active_interface:
            interface = self.interfaces[self.active_interface]
        else:
            logger.error("没有可用的活跃接口")
            return []
        
        try:
            observations = interface.get_observation()
            interface.log_interaction(None, observations, None, True)
            return observations
        except Exception as e:
            logger.error(f"获取观察时出错: {e}")
            return []
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有接口的统计信息
        
        Returns:
            接口名称到统计信息的映射
        """
        stats = {}
        for name, interface in self.interfaces.items():
            stats[name] = interface.get_statistics()
        return stats
    
    def get_all_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有接口的能力信息
        
        Returns:
            接口名称到能力信息的映射
        """
        capabilities = {}
        for name, interface in self.interfaces.items():
            capabilities[name] = interface.get_capabilities()
        return capabilities
    
    def _record_interaction(self, interaction: Dict[str, Any]):
        """
        记录交互历史
        
        Args:
            interaction: 交互记录
        """
        self.interaction_history.append(interaction)
        
        # 限制历史大小
        if len(self.interaction_history) > self.max_history_size:
            self.interaction_history = self.interaction_history[-self.max_history_size:]
    
    def get_interaction_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取交互历史
        
        Args:
            limit: 限制返回的记录数，为None时返回全部
            
        Returns:
            交互历史列表
        """
        if limit:
            return self.interaction_history[-limit:]
        return self.interaction_history


class SimulatedEnvironment(EnvironmentInterface):
    """
    模拟环境实现
    
    用于测试和开发，模拟基本的环境交互
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化模拟环境"""
        # 从配置中获取接口名称，如果没有则使用默认值
        interface_name = "simulated_environment"
        if config and "interface_name" in config:
            interface_name = config["interface_name"]
        
        super().__init__(interface_name, config)
        
        # 模拟环境状态
        self.simulated_state = {
            "position": {"x": 0, "y": 0, "z": 0},
            "orientation": {"roll": 0, "pitch": 0, "yaw": 0},
            "objects": [],
            "time": 0
        }
        
        # 模拟对象
        self.objects = [
            {"id": "obj1", "type": "cube", "position": {"x": 1, "y": 0, "z": 0}, "color": "red"},
            {"id": "obj2", "type": "sphere", "position": {"x": -1, "y": 0, "z": 0}, "color": "blue"},
            {"id": "obj3", "type": "cylinder", "position": {"x": 0, "y": 1, "z": 0}, "color": "green"}
        ]
        
        self.simulated_state["objects"] = self.objects.copy()
    
    def connect(self) -> bool:
        """连接模拟环境（总是成功）"""
        self.is_connected = True
        logger.info("连接到模拟环境")
        return True
    
    def disconnect(self) -> bool:
        """断开模拟环境连接"""
        self.is_connected = False
        logger.info("断开模拟环境连接")
        return True
    
    def execute_action(self, action: Action) -> Tuple[bool, Optional[str]]:
        """执行模拟动作"""
        if not self.is_connected:
            return False, "未连接到环境"
        
        action_type = action.action_type
        params = action.parameters
        
        try:
            if action_type == ActionType.PHYSICAL:
                # 模拟物理动作
                if "move" in params:
                    direction = params.get("direction", "forward")
                    distance = params.get("distance", 1.0)
                    
                    if direction == "forward":
                        self.simulated_state["position"]["x"] += distance
                    elif direction == "backward":
                        self.simulated_state["position"]["x"] -= distance
                    elif direction == "left":
                        self.simulated_state["position"]["y"] += distance
                    elif direction == "right":
                        self.simulated_state["position"]["y"] -= distance
                    
                    return True, f"移动 {direction} {distance} 单位"
                
                elif "rotate" in params:
                    axis = params.get("axis", "yaw")
                    angle = params.get("angle", 90.0)
                    
                    self.simulated_state["orientation"][axis] += angle
                    return True, f"旋转 {axis} {angle} 度"
                
            elif action_type == ActionType.SENSORY:
                # 模拟感知动作
                return True, "感知动作执行成功"
            
            elif action_type == ActionType.COMMUNICATION:
                # 模拟通信动作
                message = params.get("message", "")
                return True, f"发送消息: {message}"
            
            elif action_type == ActionType.COGNITIVE:
                # 模拟认知动作
                return True, "认知动作执行成功"
            
            elif action_type == ActionType.ABSTRACT:
                # 模拟抽象动作
                task = params.get("task", "")
                return True, f"执行任务: {task}"
            
            else:
                return False, f"不支持的动作类型: {action_type}"
                
        except Exception as e:
            return False, f"执行动作时出错: {e}"
    
    def get_observation(self, observation_type: Optional[ObservationType] = None) -> List[Observation]:
        """获取模拟观察"""
        if not self.is_connected:
            return []
        
        observations = []
        current_time = time.time()
        
        # 视觉观察
        if observation_type is None or observation_type == ObservationType.VISUAL:
            visual_data = {
                "position": self.simulated_state["position"],
                "orientation": self.simulated_state["orientation"],
                "objects": self.objects
            }
            observations.append(Observation(
                observation_id=f"visual_{current_time}",
                observation_type=ObservationType.VISUAL,
                data=visual_data,
                timestamp=current_time,
                confidence=0.95
            ))
        
        # 本体感知观察
        if observation_type is None or observation_type == ObservationType.PROPRIOCEPTIVE:
            proprioceptive_data = {
                "position": self.simulated_state["position"],
                "orientation": self.simulated_state["orientation"]
            }
            observations.append(Observation(
                observation_id=f"proprioceptive_{current_time}",
                observation_type=ObservationType.PROPRIOCEPTIVE,
                data=proprioceptive_data,
                timestamp=current_time,
                confidence=0.98
            ))
        
        # 数值观察
        if observation_type is None or observation_type == ObservationType.NUMERIC:
            numeric_data = {
                "time": self.simulated_state["time"],
                "object_count": len(self.objects)
            }
            observations.append(Observation(
                observation_id=f"numeric_{current_time}",
                observation_type=ObservationType.NUMERIC,
                data=numeric_data,
                timestamp=current_time,
                confidence=1.0
            ))
        
        # 更新模拟时间
        self.simulated_state["time"] += 1
        
        return observations
    
    def get_reward(self) -> List[Reward]:
        """获取模拟奖励"""
        if not self.is_connected:
            return []
        
        rewards = []
        current_time = time.time()
        
        # 基于位置的奖励
        x_pos = self.simulated_state["position"]["x"]
        y_pos = self.simulated_state["position"]["y"]
        
        # 距离原点的负奖励
        distance = (x_pos**2 + y_pos**2)**0.5
        position_reward = -distance * 0.1
        
        rewards.append(Reward(
            reward_id=f"position_{current_time}",
            value=position_reward,
            source="position_penalty",
            timestamp=current_time,
            description=f"基于位置的奖励: {position_reward:.2f}"
        ))
        
        # 探索奖励（发现新对象）
        explored_objects = len([obj for obj in self.objects 
                              if abs(obj["position"]["x"] - x_pos) < 2 and 
                                 abs(obj["position"]["y"] - y_pos) < 2])
        exploration_reward = explored_objects * 0.5
        
        if exploration_reward > 0:
            rewards.append(Reward(
                reward_id=f"exploration_{current_time}",
                value=exploration_reward,
                source="object_exploration",
                timestamp=current_time,
                description=f"探索奖励: {exploration_reward:.2f}"
            ))
        
        return rewards
    
    def get_state(self) -> Optional[EnvironmentState]:
        """获取模拟环境状态"""
        if not self.is_connected:
            return None
        
        return EnvironmentState(
            state_id=f"simulated_state_{time.time()}",
            state_type="simulated",
            variables=self.simulated_state.copy(),
            timestamp=time.time(),
            uncertainty=0.1
        )
    
    def reset(self) -> bool:
        """重置模拟环境"""
        # 重置状态
        self.simulated_state = {
            "position": {"x": 0, "y": 0, "z": 0},
            "orientation": {"roll": 0, "pitch": 0, "yaw": 0},
            "objects": self.objects,
            "time": 0
        }
        
        logger.info("重置模拟环境")
        return True


# 全局环境管理器实例
global_environment_manager = UnifiedEnvironmentManager()


def initialize_environment_system() -> UnifiedEnvironmentManager:
    """
    初始化环境系统
    
    Returns:
        统一环境管理器实例
    """
    logger.info("初始化环境系统")
    
    # 创建并注册模拟环境
    simulated_env = SimulatedEnvironment({
        "simulation_mode": "basic",
        "max_objects": 10,
        "enable_rewards": True
    })
    
    global_environment_manager.register_interface(simulated_env)
    
    # 连接所有接口
    connection_results = global_environment_manager.connect_all()
    logger.info(f"环境接口连接结果: {connection_results}")
    
    return global_environment_manager


def get_environment_manager() -> UnifiedEnvironmentManager:
    """
    获取全局环境管理器
    
    Returns:
        统一环境管理器实例
    """
    return global_environment_manager