"""
API模块

包含API服务器和通信协议：
- 服务器：用于AGI请求的FastAPI服务器（单个端点）
- WebSocket：实时流式通信
- 协议：通信协议和数据格式
"""

# 从server.py导入可用的类
try:
    from .server import app
    from .server import CognitiveRequest, CognitiveResponse
    from .server import DiagnosticsResponse
except ImportError:
    # 如果导入失败，创建占位符
    app = None
    CognitiveRequest = None
    CognitiveResponse = None
    DiagnosticsResponse = None

# 导入WebSocket管理器
try:
    from .websocket import WebSocketManager, get_websocket_manager
except ImportError:
    WebSocketManager = None
    get_websocket_manager = None

# 导入协议
try:
    from .protocols import (
        BaseMessage,
        CognitiveRequestMessage,
        CognitiveResponseMessage,
        DiagnosticsRequestMessage,
        DiagnosticsResponseMessage,
        SubscriptionMessage,
        UnsubscriptionMessage,
        CognitiveStateUpdateMessage,
        PerformanceUpdateMessage,
        SystemEventMessage,
        PingMessage,
        PongMessage,
        ErrorMessage,
        parse_message,
        validate_message,
        ProtocolValidator
    )
except ImportError:
    BaseMessage = None
    CognitiveRequestMessage = None
    CognitiveResponseMessage = None
    DiagnosticsRequestMessage = None
    DiagnosticsResponseMessage = None
    SubscriptionMessage = None
    UnsubscriptionMessage = None
    CognitiveStateUpdateMessage = None
    PerformanceUpdateMessage = None
    SystemEventMessage = None
    PingMessage = None
    PongMessage = None
    ErrorMessage = None
    parse_message = None
    validate_message = None
    ProtocolValidator = None

__all__ = [
    'app',
    'CognitiveRequest',
    'CognitiveResponse',
    'DiagnosticsResponse',
    'WebSocketManager',
    'get_websocket_manager',
    'BaseMessage',
    'CognitiveRequestMessage',
    'CognitiveResponseMessage',
    'DiagnosticsRequestMessage',
    'DiagnosticsResponseMessage',
    'SubscriptionMessage',
    'UnsubscriptionMessage',
    'CognitiveStateUpdateMessage',
    'PerformanceUpdateMessage',
    'SystemEventMessage',
    'PingMessage',
    'PongMessage',
    'ErrorMessage',
    'parse_message',
    'validate_message',
    'ProtocolValidator'
]