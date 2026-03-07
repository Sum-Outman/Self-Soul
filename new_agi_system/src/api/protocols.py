"""
通信协议定义

定义API和WebSocket通信的协议、消息格式和验证规则。
"""

import json
import time
from typing import Dict, Any, List, Optional, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, validator
import hashlib


class MessageType(Enum):
    """消息类型枚举"""
    COGNITIVE_REQUEST = "cognitive_request"
    COGNITIVE_RESPONSE = "cognitive_response"
    DIAGNOSTICS_REQUEST = "diagnostics_request"
    DIAGNOSTICS_RESPONSE = "diagnostics_response"
    CONNECTION_ESTABLISHED = "connection_established"
    SUBSCRIPTION_REQUEST = "subscribe"
    UNSUBSCRIPTION_REQUEST = "unsubscribe"
    SUBSCRIPTION_CONFIRMATION = "subscription_confirmation"
    UNSUBSCRIPTION_CONFIRMATION = "unsubscription_confirmation"
    COGNITIVE_STATE_UPDATE = "cognitive_state_update"
    PERFORMANCE_UPDATE = "performance_update"
    SYSTEM_EVENT = "system_event"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"


class CognitiveModality(Enum):
    """认知模态类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    STRUCTURED = "structured"
    MULTIMODAL = "multimodal"


class PriorityLevel(Enum):
    """优先级级别"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    REALTIME = "realtime"


class BaseMessage(BaseModel):
    """基础消息模型"""
    type: str
    timestamp: float = Field(default_factory=time.time)
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """验证时间戳"""
        if v <= 0:
            raise ValueError("时间戳必须为正数")
        if v > time.time() + 3600:  # 不能超过当前时间1小时
            raise ValueError("时间戳不能超过未来1小时")
        return v
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return self.json()
    
    @classmethod
    def from_json(cls, json_str: str):
        """从JSON字符串创建"""
        return cls.parse_raw(json_str)


class CognitiveRequestMessage(BaseMessage):
    """认知请求消息"""
    type: Literal[MessageType.COGNITIVE_REQUEST.value] = MessageType.COGNITIVE_REQUEST.value
    
    # 输入数据
    data: Dict[str, Any]
    
    # 请求参数
    priority: PriorityLevel = Field(default=PriorityLevel.NORMAL)
    modalities: List[CognitiveModality] = Field(default_factory=list)
    timeout: Optional[float] = Field(default=None, ge=0.1, le=300)  # 0.1到300秒
    
    # 上下文信息
    context: Dict[str, Any] = Field(default_factory=dict)
    goal: Optional[str] = None
    
    @validator('data')
    def validate_data(cls, v):
        """验证数据"""
        if not v:
            raise ValueError("数据不能为空")
        return v
    
    @validator('modalities')
    def validate_modalities(cls, v, values):
        """验证模态"""
        if not v:
            # 如果没有指定模态，从数据中推断
            data = values.get('data', {})
            modalities = []
            
            if 'text' in data and data['text']:
                modalities.append(CognitiveModality.TEXT)
            if 'image' in data and data['image']:
                modalities.append(CognitiveModality.IMAGE)
            if 'audio' in data and data['audio']:
                modalities.append(CognitiveModality.AUDIO)
            if 'structured' in data and data['structured']:
                modalities.append(CognitiveModality.STRUCTURED)
            
            if not modalities:
                modalities.append(CognitiveModality.TEXT)  # 默认为文本
            
            return modalities
        return v


class CognitiveResponseMessage(BaseMessage):
    """认知响应消息"""
    type: Literal[MessageType.COGNITIVE_RESPONSE.value] = MessageType.COGNITIVE_RESPONSE.value
    
    # 响应数据
    data: Dict[str, Any]
    
    # 性能指标
    performance: Dict[str, Any] = Field(default_factory=dict)
    
    # 推理痕迹
    reasoning_trace: Optional[Dict[str, Any]] = None
    
    # 错误信息
    error: Optional[str] = None
    
    @validator('data')
    def validate_data(cls, v):
        """验证响应数据"""
        # 必须包含output字段
        if 'output' not in v:
            raise ValueError("响应数据必须包含output字段")
        return v


class DiagnosticsRequestMessage(BaseMessage):
    """诊断请求消息"""
    type: Literal[MessageType.DIAGNOSTICS_REQUEST.value] = MessageType.DIAGNOSTICS_REQUEST.value
    
    # 请求的诊断信息类型
    include_cognitive_state: bool = Field(default=True)
    include_performance_metrics: bool = Field(default=True)
    include_representation_cache: bool = Field(default=False)
    include_communication_stats: bool = Field(default=False)
    include_system_info: bool = Field(default=True)


class DiagnosticsResponseMessage(BaseMessage):
    """诊断响应消息"""
    type: Literal[MessageType.DIAGNOSTICS_RESPONSE.value] = MessageType.DIAGNOSTICS_RESPONSE.value
    
    # 诊断数据
    cognitive_state: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    representation_cache: Dict[str, Any] = Field(default_factory=dict)
    communication_stats: Dict[str, Any] = Field(default_factory=dict)
    system_info: Dict[str, Any] = Field(default_factory=dict)


class SubscriptionMessage(BaseMessage):
    """订阅消息"""
    type: Literal[MessageType.SUBSCRIPTION_REQUEST.value] = MessageType.SUBSCRIPTION_REQUEST.value
    
    # 订阅的频道
    channels: List[str]
    
    @validator('channels')
    def validate_channels(cls, v):
        """验证频道"""
        valid_channels = ['cognitive_state', 'performance_metrics', 'system_events']
        
        for channel in v:
            if channel not in valid_channels:
                raise ValueError(f"无效的频道: {channel}。有效频道: {valid_channels}")
        
        return v


class UnsubscriptionMessage(BaseMessage):
    """取消订阅消息"""
    type: Literal[MessageType.UNSUBSCRIPTION_REQUEST.value] = MessageType.UNSUBSCRIPTION_REQUEST.value
    
    # 取消订阅的频道
    channels: List[str]


class CognitiveStateUpdateMessage(BaseMessage):
    """认知状态更新消息"""
    type: Literal[MessageType.COGNITIVE_STATE_UPDATE.value] = MessageType.COGNITIVE_STATE_UPDATE.value
    
    # 状态数据
    state: Dict[str, Any]
    
    @validator('state')
    def validate_state(cls, v):
        """验证状态数据"""
        required_fields = ['current_focus', 'working_memory', 'goal_stack']
        
        for field in required_fields:
            if field not in v:
                raise ValueError(f"状态数据必须包含 {field} 字段")
        
        return v


class PerformanceUpdateMessage(BaseMessage):
    """性能更新消息"""
    type: Literal[MessageType.PERFORMANCE_UPDATE.value] = MessageType.PERFORMANCE_UPDATE.value
    
    # 性能数据
    metrics: Dict[str, Any]
    
    @validator('metrics')
    def validate_metrics(cls, v):
        """验证性能指标"""
        if not v:
            raise ValueError("性能指标不能为空")
        return v


class SystemEventMessage(BaseMessage):
    """系统事件消息"""
    type: Literal[MessageType.SYSTEM_EVENT.value] = MessageType.SYSTEM_EVENT.value
    
    # 事件类型
    event_type: str
    
    # 事件数据
    event_data: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('event_type')
    def validate_event_type(cls, v):
        """验证事件类型"""
        valid_types = [
            'component_initialized',
            'component_error',
            'cognitive_cycle_completed',
            'goal_achieved',
            'system_warning',
            'system_error'
        ]
        
        if v not in valid_types:
            raise ValueError(f"无效的事件类型: {v}。有效类型: {valid_types}")
        
        return v


class PingMessage(BaseMessage):
    """Ping消息"""
    type: Literal[MessageType.PING.value] = MessageType.PING.value


class PongMessage(BaseMessage):
    """Pong消息"""
    type: Literal[MessageType.PONG.value] = MessageType.PONG.value
    
    # 原始ping消息的时间戳
    original_timestamp: Optional[float] = None


class ErrorMessage(BaseMessage):
    """错误消息"""
    type: Literal[MessageType.ERROR.value] = MessageType.ERROR.value
    
    # 错误信息
    message: str
    
    # 错误代码
    error_code: Optional[str] = None
    
    # 错误详情
    details: Optional[Dict[str, Any]] = None


# 消息类型映射
MESSAGE_TYPE_TO_CLASS = {
    MessageType.COGNITIVE_REQUEST.value: CognitiveRequestMessage,
    MessageType.COGNITIVE_RESPONSE.value: CognitiveResponseMessage,
    MessageType.DIAGNOSTICS_REQUEST.value: DiagnosticsRequestMessage,
    MessageType.DIAGNOSTICS_RESPONSE.value: DiagnosticsResponseMessage,
    MessageType.SUBSCRIPTION_REQUEST.value: SubscriptionMessage,
    MessageType.UNSUBSCRIPTION_REQUEST.value: UnsubscriptionMessage,
    MessageType.COGNITIVE_STATE_UPDATE.value: CognitiveStateUpdateMessage,
    MessageType.PERFORMANCE_UPDATE.value: PerformanceUpdateMessage,
    MessageType.SYSTEM_EVENT.value: SystemEventMessage,
    MessageType.PING.value: PingMessage,
    MessageType.PONG.value: PongMessage,
    MessageType.ERROR.value: ErrorMessage
}


def parse_message(json_data: Union[str, Dict]) -> BaseMessage:
    """
    解析消息JSON
    
    参数:
        json_data: JSON字符串或字典
    
    返回:
        消息对象
    """
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # 获取消息类型
    message_type = data.get('type')
    
    if not message_type:
        raise ValueError("消息缺少type字段")
    
    # 查找对应的消息类
    message_class = MESSAGE_TYPE_TO_CLASS.get(message_type)
    
    if not message_class:
        # 对于未知类型，返回基础消息
        return BaseMessage(**data)
    
    # 创建消息对象
    return message_class(**data)


def validate_message(message_dict: Dict[str, Any]) -> bool:
    """
    验证消息格式
    
    参数:
        message_dict: 消息字典
    
    返回:
        是否有效
    """
    try:
        parse_message(message_dict)
        return True
    except Exception:
        return False


def create_message_hash(message: BaseMessage) -> str:
    """
    创建消息哈希
    
    参数:
        message: 消息对象
    
    返回:
        哈希值
    """
    # 转换为字典并排除某些字段
    data = message.dict(exclude={'timestamp', 'metadata', 'request_id'})
    
    # 排序键以确保一致性
    sorted_data = json.dumps(data, sort_keys=True)
    
    # 创建哈希
    return hashlib.sha256(sorted_data.encode()).hexdigest()


class ProtocolValidator:
    """协议验证器"""
    
    def __init__(self):
        self.message_history = {}  # 消息历史（用于重复检测）
    
    def validate_and_parse(self, json_data: Union[str, Dict]) -> BaseMessage:
        """
        验证并解析消息
        
        参数:
            json_data: JSON字符串或字典
        
        返回:
            消息对象
        
        抛出:
            ValueError: 如果消息无效
        """
        try:
            message = parse_message(json_data)
            
            # 检查重复消息
            message_hash = create_message_hash(message)
            if message_hash in self.message_history:
                raise ValueError("重复消息")
            
            # 记录消息
            self.message_history[message_hash] = time.time()
            
            # 清理旧记录
            self._cleanup_history()
            
            return message
            
        except Exception as e:
            raise ValueError(f"消息验证失败: {str(e)}")
    
    def _cleanup_history(self, max_age_seconds: float = 3600):
        """清理历史记录"""
        current_time = time.time()
        to_delete = []
        
        for msg_hash, timestamp in self.message_history.items():
            if current_time - timestamp > max_age_seconds:
                to_delete.append(msg_hash)
        
        for msg_hash in to_delete:
            del self.message_history[msg_hash]