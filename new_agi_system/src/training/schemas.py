"""
训练相关数据模式
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class TrainingRequest(BaseModel):
    """训练请求"""
    type: str = Field(..., description="训练类型: cognitive_model, neural_network, from_scratch")
    model_name: str = Field(..., description="模型名称")
    dataset: Dict[str, Any] = Field(default_factory=dict, description="数据集配置")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="超参数")
    priority: str = Field("normal", description="优先级: low, normal, high")


class TrainingResponse(BaseModel):
    """训练响应"""
    job_id: str = Field(..., description="训练任务ID")
    status: str = Field(..., description="状态: pending, running, completed, failed, cancelled")
    message: str = Field(..., description="状态消息")
    queue_position: Optional[int] = Field(None, description="队列位置（如果排队中）")


class TrainingStatus(BaseModel):
    """训练状态"""
    job_id: str = Field(..., description="训练任务ID")
    status: str = Field(..., description="状态")
    progress: float = Field(..., description="进度 (0.0 - 1.0)")
    start_time: Optional[float] = Field(None, description="开始时间戳")
    end_time: Optional[float] = Field(None, description="结束时间戳")
    config: Dict[str, Any] = Field(..., description="训练配置")
    result: Optional[Dict[str, Any]] = Field(None, description="训练结果")
    error: Optional[str] = Field(None, description="错误信息")


class TrainingStatistics(BaseModel):
    """训练统计信息"""
    total_jobs: int = Field(..., description="总任务数")
    completed_jobs: int = Field(..., description="已完成任务数")
    failed_jobs: int = Field(..., description="失败任务数")
    cancelled_jobs: int = Field(..., description="取消任务数")
    total_training_time: float = Field(..., description="总训练时间（秒）")
    avg_training_time: float = Field(..., description="平均训练时间（秒）")
    active_jobs: int = Field(..., description="活动任务数")
    max_concurrent_jobs: int = Field(..., description="最大并发任务数")


class ModelTrainingConfig(BaseModel):
    """模型训练配置"""
    model_type: str = Field(..., description="模型类型")
    architecture: Dict[str, Any] = Field(..., description="架构配置")
    loss_function: str = Field(..., description="损失函数")
    optimizer: str = Field(..., description="优化器")
    learning_rate: float = Field(..., description="学习率")
    batch_size: int = Field(..., description="批量大小")
    epochs: int = Field(..., description="训练轮数")
    validation_split: float = Field(0.2, description="验证集比例")


class DatasetConfig(BaseModel):
    """数据集配置"""
    name: str = Field(..., description="数据集名称")
    path: str = Field(..., description="数据集路径")
    format: str = Field(..., description="数据格式: json, csv, image_folder等")
    preprocessing: Dict[str, Any] = Field(default_factory=dict, description="预处理配置")
    split: Dict[str, float] = Field(default_factory=dict, description="数据划分比例")