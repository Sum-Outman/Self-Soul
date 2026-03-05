"""
Cross-Model Communication Protocol (CMCP)
========================================

Unified communication framework for AGI model collaboration.

This module provides a standardized protocol for communication between
different AI models in the Self-Soul AGI system.
"""

from .cmcp_gateway import CMCPGateway
from .protocols import (
    CommunicationProtocol,
    DirectCallHandler,
    RESTHandler,
    WebSocketHandler,
    MessageQueueHandler,
    SharedMemoryHandler
)
from .message_format import (
    Message,
    RequestMessage,
    ResponseMessage,
    ErrorMessage
)
from .model_adapter import ModelCommunicationAdapter
from .registry import ModelRegistryClient
from .monitoring import CommunicationMonitor

__all__ = [
    'CMCPGateway',
    'CommunicationProtocol',
    'DirectCallHandler',
    'RESTHandler', 
    'WebSocketHandler',
    'MessageQueueHandler',
    'SharedMemoryHandler',
    'Message',
    'RequestMessage',
    'ResponseMessage',
    'ErrorMessage',
    'ModelCommunicationAdapter',
    'ModelRegistryClient',
    'CommunicationMonitor'
]

__version__ = '1.0.0'