"""
人形机器人AGI系统

整合人形机器人的所有高级功能，包括平衡控制、双足行走、社交交互和环境适应。
作为统一认知架构的人形机器人扩展模块。
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# 导入子模块
from .balance_control import BalanceControlSystem, BalanceState
from .walking_gait import WalkingGaitSystem, WalkingDirection, GaitPhase


class HumanoidTaskType(Enum):
    """人形机器人任务类型"""
    WALK_TO_LOCATION = "walk_to_location"      # 走到指定位置
    PICK_UP_OBJECT = "pick_up_object"          # 捡起物体
    PLACE_OBJECT = "place_object"              # 放置物体
    INTERACT_WITH_HUMAN = "interact_with_human"  # 与人交互
    NAVIGATE_ENVIRONMENT = "navigate_environment"  # 环境导航
    MAINTAIN_BALANCE = "maintain_balance"      # 保持平衡
    PERFORM_GESTURE = "perform_gesture"        # 执行手势


class HumanoidAGISystem:
    """人形机器人AGI系统"""
    
    def __init__(self, communication):
        """
        初始化人形机器人AGI系统。
        
        参数:
            communication: 神经通信系统
        """
        self.communication = communication
        self.initialized = False
        
        # 子系统
        self.balance_control = BalanceControlSystem(communication)
        self.walking_gait = WalkingGaitSystem(communication)
        
        # 状态管理
        self.current_task = None
        self.task_history: List[Dict[str, Any]] = []
        self.system_state = {
            'operational_mode': 'normal',  # normal, cautious, emergency, maintenance
            'autonomy_level': 0.7,         # 自主性水平 (0-1)
            'safety_status': 'safe',       # safe, warning, critical
            'battery_level': 0.8,          # 电池电量 (0-1)
            'temperature': 25.0,           # 系统温度 (°C)
            'last_update': time.time()
        }
        
        # 任务队列
        self.task_queue: List[Dict[str, Any]] = []
        self.active_tasks: List[Dict[str, Any]] = []
        
        # 性能指标
        self.performance_metrics = {
            'total_tasks_completed': 0,
            'total_walking_distance': 0.0,
            'total_operating_time': 0.0,
            'balance_maintenance_rate': 0.0,
            'task_success_rate': 0.0,
            'emergency_stops': 0,
            'human_interactions': 0
        }
        
        # 配置参数
        self.config = {
            'max_walking_speed': 1.0,           # 最大行走速度 (m/s)
            'max_object_weight': 2.0,           # 最大物体重量 (kg)
            'interaction_distance': 1.5,        # 交互距离 (米)
            'safety_distance': 0.5,             # 安全距离 (米)
            'battery_warning_threshold': 0.2,   # 电池警告阈值
            'temperature_warning_threshold': 60.0,  # 温度警告阈值 (°C)
            'emergency_response_time': 0.1,     # 紧急响应时间 (秒)
        }
        
        # 监控任务
        self.monitoring_task = None
        self.monitoring_active = False
        
        logger.info("人形机器人AGI系统已初始化")
    
    async def initialize(self):
        """初始化人形机器人AGI系统"""
        if self.initialized:
            return
        
        logger.info("初始化人形机器人AGI系统...")
        
        # 初始化子系统
        await self.balance_control.initialize()
        await self.walking_gait.initialize()
        
        # 在通信系统中注册
        await self.communication.register_component(
            component_name="humanoid_agi",
            component_type="humanoid"
        )
        
        # 启动监控任务
        await self._start_monitoring()
        
        self.initialized = True
        logger.info("人形机器人AGI系统初始化完成")
    
    async def _start_monitoring(self):
        """启动监控任务"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("人形机器人监控任务已启动")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 更新系统状态
                await self._update_system_state()
                
                # 检查安全状态
                await self._check_safety()
                
                # 监控任务执行
                await self._monitor_tasks()
                
                # 更新性能指标
                await self._update_performance_metrics()
                
                # 等待下一个周期
                await asyncio.sleep(1.0)  # 1秒间隔
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(5.0)
    
    async def _update_system_state(self):
        """更新系统状态"""
        try:
            # 这里应该从硬件获取实际数据
            # 简化实现：模拟数据更新
            
            current_time = time.time()
            
            # 模拟电池消耗
            time_elapsed = current_time - self.system_state['last_update']
            battery_drain = time_elapsed / 36000.0  # 10小时耗尽
            self.system_state['battery_level'] = max(0.0, self.system_state['battery_level'] - battery_drain)
            
            # 模拟温度变化
            activity_factor = 1.0 if self.walking_gait.is_walking else 0.3
            temperature_increase = activity_factor * time_elapsed / 600.0  # 10分钟升温
            self.system_state['temperature'] = min(80.0, self.system_state['temperature'] + temperature_increase)
            
            # 冷却效果
            ambient_temperature = 25.0
            cooling_rate = 0.1 * time_elapsed / 60.0  # 每分钟冷却0.1度
            if self.system_state['temperature'] > ambient_temperature:
                self.system_state['temperature'] = max(
                    ambient_temperature,
                    self.system_state['temperature'] - cooling_rate
                )
            
            self.system_state['last_update'] = current_time
            self.system_state['total_operating_time'] = self.performance_metrics['total_operating_time'] + time_elapsed
            
        except Exception as e:
            logger.error(f"更新系统状态失败: {e}")
    
    async def _check_safety(self):
        """检查安全状态"""
        try:
            # 检查电池
            if self.system_state['battery_level'] < self.config['battery_warning_threshold']:
                self.system_state['safety_status'] = 'warning'
                logger.warning(f"电池电量低: {self.system_state['battery_level']:.2f}")
            
            # 检查温度
            if self.system_state['temperature'] > self.config['temperature_warning_threshold']:
                self.system_state['safety_status'] = 'critical'
                logger.warning(f"系统温度过高: {self.system_state['temperature']:.1f}°C")
                
                # 温度过高时停止行走
                if self.walking_gait.is_walking:
                    await self.walking_gait.emergency_stop()
                    logger.warning("因温度过高停止行走")
            
            # 检查平衡状态
            balance_report = await self.balance_control.get_balance_report()
            if balance_report['balance_state'] == 'critical':
                self.system_state['safety_status'] = 'warning'
            elif balance_report['balance_state'] == 'falling':
                self.system_state['safety_status'] = 'critical'
                
                # 紧急停止所有活动
                await self.emergency_stop()
            
            # 如果没有问题，设置为安全
            if (self.system_state['battery_level'] >= self.config['battery_warning_threshold'] and
                self.system_state['temperature'] <= self.config['temperature_warning_threshold'] and
                balance_report['balance_state'] not in ['critical', 'falling']):
                self.system_state['safety_status'] = 'safe'
                
        except Exception as e:
            logger.error(f"检查安全状态失败: {e}")
    
    async def _monitor_tasks(self):
        """监控任务执行"""
        try:
            # 检查活跃任务
            tasks_to_remove = []
            
            for task in self.active_tasks:
                task_id = task.get('task_id')
                task_type = task.get('type')
                start_time = task.get('start_time')
                
                # 检查任务超时
                if time.time() - start_time > task.get('timeout', 300.0):
                    logger.warning(f"任务 {task_id} 超时")
                    task['status'] = 'timeout'
                    tasks_to_remove.append(task)
                    
                    # 更新性能指标
                    self.performance_metrics['task_success_rate'] = (
                        self.performance_metrics['task_success_rate'] * 
                        self.performance_metrics['total_tasks_completed'] / 
                        max(1, self.performance_metrics['total_tasks_completed'] + 1)
                    )
                    continue
                
                # 根据任务类型监控进度
                if task_type == HumanoidTaskType.WALK_TO_LOCATION.value:
                    await self._monitor_walking_task(task)
                elif task_type == HumanoidTaskType.MAINTAIN_BALANCE.value:
                    await self._monitor_balance_task(task)
                # 其他任务类型的监控...
            
            # 移除已完成或失败的任务
            for task in tasks_to_remove:
                if task in self.active_tasks:
                    self.active_tasks.remove(task)
                    self.task_history.append(task)
            
        except Exception as e:
            logger.error(f"监控任务失败: {e}")
    
    async def _monitor_walking_task(self, task: Dict[str, Any]):
        """监控行走任务"""
        try:
            # 获取行走报告
            walking_report = await self.walking_gait.get_walking_report()
            
            # 更新任务状态
            task['current_step'] = walking_report['current_step']
            task['total_steps'] = walking_report['total_steps']
            task['progress'] = walking_report['current_step'] / max(1, walking_report['total_steps'])
            
            # 检查是否完成
            if not walking_report['is_walking'] and task['progress'] >= 0.99:
                task['status'] = 'completed'
                task['completion_time'] = time.time()
                task['result'] = {'success': True, 'distance': walking_report['performance_metrics']['walking_distance']}
                
                # 更新性能指标
                self.performance_metrics['total_tasks_completed'] += 1
                self.performance_metrics['total_walking_distance'] += walking_report['performance_metrics']['walking_distance']
                self.performance_metrics['task_success_rate'] = (
                    (self.performance_metrics['task_success_rate'] * 
                     (self.performance_metrics['total_tasks_completed'] - 1) + 1) / 
                    self.performance_metrics['total_tasks_completed']
                )
                
                logger.info(f"行走任务 {task.get('task_id')} 完成")
            
        except Exception as e:
            logger.error(f"监控行走任务失败: {e}")
            task['status'] = 'failed'
            task['error'] = str(e)
    
    async def _monitor_balance_task(self, task: Dict[str, Any]):
        """监控平衡任务"""
        try:
            # 获取平衡报告
            balance_report = await self.balance_control.get_balance_report()
            
            # 更新任务状态
            task['balance_state'] = balance_report['balance_state']
            task['stability_margin'] = balance_report['current_metrics']['stability_margin']
            
            # 检查是否达到平衡目标
            target_margin = task.get('target_stability_margin', 0.05)
            duration = task.get('duration', 10.0)
            
            if balance_report['current_metrics']['stability_margin'] >= target_margin:
                elapsed_time = time.time() - task['start_time']
                
                if elapsed_time >= duration:
                    task['status'] = 'completed'
                    task['completion_time'] = time.time()
                    task['result'] = {'success': True, 'duration': elapsed_time}
                    
                    # 更新性能指标
                    self.performance_metrics['total_tasks_completed'] += 1
                    self.performance_metrics['balance_maintenance_rate'] = (
                        (self.performance_metrics['balance_maintenance_rate'] * 
                         (self.performance_metrics['total_tasks_completed'] - 1) + 1) / 
                        self.performance_metrics['total_tasks_completed']
                    )
                    
                    logger.info(f"平衡任务 {task.get('task_id')} 完成，持续时间: {elapsed_time:.1f}秒")
            
        except Exception as e:
            logger.error(f"监控平衡任务失败: {e}")
            task['status'] = 'failed'
            task['error'] = str(e)
    
    async def _update_performance_metrics(self):
        """更新性能指标"""
        try:
            # 更新操作时间
            current_time = time.time()
            if self.system_state['last_update'] > 0:
                time_elapsed = current_time - self.system_state['last_update']
                self.performance_metrics['total_operating_time'] += time_elapsed
            
        except Exception as e:
            logger.error(f"更新性能指标失败: {e}")
    
    async def execute_task(self, task_type: HumanoidTaskType, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行人形机器人任务。
        
        参数:
            task_type: 任务类型
            parameters: 任务参数
            
        返回:
            任务执行结果
        """
        try:
            logger.info(f"执行人形机器人任务: {task_type.value}")
            
            # 创建任务记录
            task_id = f"task_{int(time.time())}_{len(self.task_history)}"
            task_record = {
                'task_id': task_id,
                'type': task_type.value,
                'parameters': parameters,
                'start_time': time.time(),
                'status': 'running',
                'assigned_to': 'humanoid_agi'
            }
            
            # 根据任务类型执行
            if task_type == HumanoidTaskType.WALK_TO_LOCATION:
                result = await self._execute_walk_to_location(parameters, task_record)
            elif task_type == HumanoidTaskType.MAINTAIN_BALANCE:
                result = await self._execute_maintain_balance(parameters, task_record)
            elif task_type == HumanoidTaskType.PICK_UP_OBJECT:
                result = await self._execute_pick_up_object(parameters, task_record)
            elif task_type == HumanoidTaskType.INTERACT_WITH_HUMAN:
                result = await self._execute_interact_with_human(parameters, task_record)
            else:
                result = {
                    'success': False,
                    'error': f'不支持的任务类型: {task_type.value}'
                }
            
            # 更新任务记录
            task_record['result'] = result
            if result.get('success', False):
                task_record['status'] = 'completed'
                task_record['completion_time'] = time.time()
            else:
                task_record['status'] = 'failed'
                task_record['error'] = result.get('error', '未知错误')
            
            # 添加到历史记录
            self.task_history.append(task_record)
            
            return {
                'task_id': task_id,
                'success': result.get('success', False),
                'result': result,
                'task_record': task_record
            }
            
        except Exception as e:
            logger.error(f"执行任务失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _execute_walk_to_location(self, parameters: Dict[str, Any], task_record: Dict[str, Any]) -> Dict[str, Any]:
        """执行走到指定位置任务"""
        try:
            # 提取参数
            target_location = parameters.get('target_location', [0.0, 0.0, 0.0])
            direction = parameters.get('direction', 'forward')
            distance = parameters.get('distance')
            
            # 转换为行走方向
            if direction == 'forward':
                walking_direction = WalkingDirection.FORWARD
            elif direction == 'backward':
                walking_direction = WalkingDirection.BACKWARD
            elif direction == 'left':
                walking_direction = WalkingDirection.LEFT
            elif direction == 'right':
                walking_direction = WalkingDirection.RIGHT
            elif direction == 'turn_left':
                walking_direction = WalkingDirection.TURN_LEFT
            elif direction == 'turn_right':
                walking_direction = WalkingDirection.TURN_RIGHT
            else:
                walking_direction = WalkingDirection.FORWARD
            
            # 开始行走
            start_result = await self.walking_gait.start_walking(walking_direction, distance)
            
            if not start_result.get('success', False):
                return {
                    'success': False,
                    'error': start_result.get('error', '开始行走失败')
                }
            
            # 将任务添加到活跃任务列表
            task_record['walking_direction'] = walking_direction.value
            task_record['distance'] = distance
            task_record['timeout'] = (start_result.get('estimated_duration', 30.0) + 10.0)  # 增加10秒缓冲
            self.active_tasks.append(task_record)
            
            return {
                'success': True,
                'message': '行走任务已开始',
                'estimated_duration': start_result.get('estimated_duration'),
                'footstep_count': start_result.get('footstep_count')
            }
            
        except Exception as e:
            logger.error(f"执行走到指定位置任务失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _execute_maintain_balance(self, parameters: Dict[str, Any], task_record: Dict[str, Any]) -> Dict[str, Any]:
        """执行保持平衡任务"""
        try:
            # 提取参数
            duration = parameters.get('duration', 10.0)
            target_margin = parameters.get('target_stability_margin', 0.05)
            
            # 将任务添加到活跃任务列表
            task_record['duration'] = duration
            task_record['target_stability_margin'] = target_margin
            task_record['timeout'] = duration + 5.0  # 增加5秒缓冲
            self.active_tasks.append(task_record)
            
            return {
                'success': True,
                'message': '平衡任务已开始',
                'duration': duration,
                'target_stability_margin': target_margin
            }
            
        except Exception as e:
            logger.error(f"执行保持平衡任务失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _execute_pick_up_object(self, parameters: Dict[str, Any], task_record: Dict[str, Any]) -> Dict[str, Any]:
        """执行捡起物体任务"""
        try:
            # 简化实现：先走到物体位置，然后执行抓取动作
            object_location = parameters.get('object_location', [1.0, 0.0, 0.5])
            object_weight = parameters.get('weight', 0.5)
            
            # 检查物体重量是否在允许范围内
            if object_weight > self.config['max_object_weight']:
                return {
                    'success': False,
                    'error': f'物体重量 {object_weight}kg 超过最大允许重量 {self.config["max_object_weight"]}kg'
                }
            
            # 计算需要行走的距离（简化）
            distance = object_location[0]  # 假设物体在x轴方向
            
            # 先走到物体位置
            walk_result = await self._execute_walk_to_location({
                'target_location': object_location,
                'distance': distance,
                'direction': 'forward'
            }, task_record)
            
            if not walk_result.get('success', False):
                return walk_result
            
            # 更新任务记录
            task_record['object_location'] = object_location
            task_record['object_weight'] = object_weight
            
            return {
                'success': True,
                'message': '捡起物体任务已开始（需要先走到物体位置）',
                'object_location': object_location,
                'object_weight': object_weight
            }
            
        except Exception as e:
            logger.error(f"执行捡起物体任务失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _execute_interact_with_human(self, parameters: Dict[str, Any], task_record: Dict[str, Any]) -> Dict[str, Any]:
        """执行与人交互任务"""
        try:
            # 简化实现
            interaction_type = parameters.get('interaction_type', 'greeting')
            human_location = parameters.get('human_location', [1.0, 0.0, 0.0])
            
            # 计算需要行走的距离
            distance = human_location[0]
            
            # 先走到人附近
            walk_result = await self._execute_walk_to_location({
                'target_location': human_location,
                'distance': distance,
                'direction': 'forward'
            }, task_record)
            
            if not walk_result.get('success', False):
                return walk_result
            
            # 更新任务记录
            task_record['interaction_type'] = interaction_type
            task_record['human_location'] = human_location
            
            # 更新性能指标
            self.performance_metrics['human_interactions'] += 1
            
            return {
                'success': True,
                'message': f'与人交互任务已开始（{interaction_type}）',
                'interaction_type': interaction_type
            }
            
        except Exception as e:
            logger.error(f"执行与人交互任务失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def update_humanoid_state(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新人形机器人状态。
        
        参数:
            sensor_data: 传感器数据
            
        返回:
            状态更新结果
        """
        try:
            # 更新平衡控制
            balance_output = await self.balance_control.update_balance(sensor_data)
            
            # 如果正在行走，更新行走状态
            walking_output = {}
            if self.walking_gait.is_walking:
                walking_output = await self.walking_gait.update_walking(sensor_data)
            
            # 更新系统状态
            await self._update_system_state()
            
            return {
                'success': True,
                'balance_output': balance_output,
                'walking_output': walking_output,
                'system_state': self.system_state.copy(),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"更新人形机器人状态失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_humanoid_report(self) -> Dict[str, Any]:
        """获取人形机器人报告"""
        try:
            # 获取子系统报告
            balance_report = await self.balance_control.get_balance_report()
            walking_report = await self.walking_gait.get_walking_report()
            
            return {
                'system_state': self.system_state.copy(),
                'performance_metrics': self.performance_metrics.copy(),
                'balance_report': balance_report,
                'walking_report': walking_report,
                'active_tasks': len(self.active_tasks),
                'task_history_count': len(self.task_history),
                'task_success_rate': self.performance_metrics['task_success_rate'],
                'operational_mode': self.system_state['operational_mode'],
                'safety_status': self.system_state['safety_status'],
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"获取人形机器人报告失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def emergency_stop(self):
        """紧急停止"""
        try:
            logger.warning("人形机器人AGI系统紧急停止")
            
            # 停止所有子系统
            await self.balance_control.emergency_stop()
            await self.walking_gait.emergency_stop()
            
            # 清空任务队列
            self.active_tasks.clear()
            
            # 更新系统状态
            self.system_state['operational_mode'] = 'emergency'
            self.system_state['safety_status'] = 'critical'
            
            # 更新性能指标
            self.performance_metrics['emergency_stops'] += 1
            
            return {
                'success': True,
                'message': '紧急停止已执行',
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"紧急停止失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def adjust_autonomy_level(self, level: float):
        """调整自主性水平"""
        try:
            if level < 0 or level > 1:
                return {'success': False, 'error': '自主性水平必须在0到1之间'}
            
            self.system_state['autonomy_level'] = level
            
            # 根据自主性水平调整配置
            if level < 0.3:
                self.system_state['operational_mode'] = 'cautious'
                self.config['max_walking_speed'] = 0.3
            elif level < 0.7:
                self.system_state['operational_mode'] = 'normal'
                self.config['max_walking_speed'] = 0.7
            else:
                self.system_state['operational_mode'] = 'autonomous'
                self.config['max_walking_speed'] = 1.0
            
            logger.info(f"自主性水平调整为 {level:.2f}，操作模式: {self.system_state['operational_mode']}")
            
            return {
                'success': True,
                'autonomy_level': level,
                'operational_mode': self.system_state['operational_mode'],
                'max_walking_speed': self.config['max_walking_speed']
            }
            
        except Exception as e:
            logger.error(f"调整自主性水平失败: {e}")
            return {'success': False, 'error': str(e)}
    
    async def shutdown(self):
        """关闭人形机器人AGI系统"""
        if not self.initialized:
            return
        
        logger.info("关闭人形机器人AGI系统...")
        
        # 停止监控任务
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # 关闭子系统
        await self.balance_control.shutdown()
        await self.walking_gait.shutdown()
        
        # 注销组件
        try:
            await self.communication.unregister_component("humanoid_agi")
        except Exception as e:
            logger.warning(f"注销组件失败: {e}")
        
        self.initialized = False
        logger.info("人形机器人AGI系统已关闭")