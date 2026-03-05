#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控Web界面 - 提供实时性能监控数据的Web API和仪表板

功能：
1. REST API端点提供实时性能数据
2. WebSocket用于实时数据推送
3. HTML仪表板显示实时性能图表
4. 警报管理和历史数据查询

设计目标：
- 提供轻量级的Web界面，便于远程监控系统性能
- 支持实时数据更新和可视化
- 易于集成到现有系统中
- 提供API接口供其他系统调用
"""

import sys
import os
import json
import time
import threading
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import asyncio

# FastAPI imports
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError as e:
    print(f"警告: FastAPI不可用，Web界面功能将受限: {e}")
    FASTAPI_AVAILABLE = False

# 导入性能监控仪表板
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.performance_monitoring_dashboard import (
        create_performance_monitoring_dashboard,
        PerformanceMonitoringDashboard,
        SystemMetric,
        PerformanceAlert,
        MetricType,
        AlertLevel
    )
    DASHBOARD_AVAILABLE = True
except ImportError as e:
    print(f"警告: 性能监控仪表板不可用: {e}")
    DASHBOARD_AVAILABLE = False

logger = logging.getLogger(__name__)

# 数据模型
class PerformanceDataResponse(BaseModel):
    """性能数据响应模型"""
    timestamp: float
    metrics: Dict[str, float]
    alerts: List[Dict[str, Any]]
    summary: Dict[str, Any]
    
class AlertResponse(BaseModel):
    """警报响应模型"""
    id: str
    level: str
    metric_type: str
    message: str
    timestamp: str
    value: float
    threshold: float
    suggestions: List[str]

class WebSocketManager:
    """WebSocket连接管理器"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """向所有连接的客户端广播消息"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"WebSocket发送失败: {e}")
                disconnected.append(connection)
        
        # 移除断开连接的客户端
        for connection in disconnected:
            self.disconnect(connection)

@dataclass
class PerformanceMonitoringWebServer:
    """性能监控Web服务器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化Web服务器"""
        self.config = config or self._get_default_config()
        
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI不可用，无法启动Web服务器")
        
        if not DASHBOARD_AVAILABLE:
            raise RuntimeError("性能监控仪表板不可用，无法启动Web服务器")
        
        # 创建性能监控仪表板
        self.dashboard = create_performance_monitoring_dashboard(
            self.config.get("dashboard_config", {})
        )
        
        # 创建FastAPI应用
        self.app = FastAPI(
            title="性能监控Web界面",
            description="实时监控系统性能的Web界面",
            version="1.0.0"
        )
        
        # WebSocket管理器
        self.websocket_manager = WebSocketManager()
        
        # 模板目录
        self.templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir)
        
        self.templates = Jinja2Templates(directory=self.templates_dir)
        
        # 静态文件目录
        self.static_dir = os.path.join(os.path.dirname(__file__), "static")
        if not os.path.exists(self.static_dir):
            os.makedirs(self.static_dir)
        
        self.app.mount("/static", StaticFiles(directory=self.static_dir), name="static")
        
        # 注册路由
        self._setup_routes()
        
        # 监控线程
        self.monitoring_thread = None
        self.monitoring_running = False
        
        # 广播间隔
        self.broadcast_interval = self.config.get("broadcast_interval", 2.0)
        
        logger.info("性能监控Web服务器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "host": "127.0.0.1",
            "port": 8088,
            "dashboard_config": {
                "monitor_interval": 2.0,
                "history_size": 1000,
                "output_dir": "performance_dashboard",
                "enable_cpu_monitoring": True,
                "enable_memory_monitoring": True,
                "enable_gpu_monitoring": False,
                "enable_disk_monitoring": True,
                "enable_network_monitoring": True,
                "enable_inference_monitoring": True,
                "enable_task_monitoring": True,
                "alert_thresholds": {
                    "cpu": {"warning": 70.0, "critical": 85.0},
                    "memory": {"warning": 75.0, "critical": 90.0},
                    "disk": {"warning": 85.0, "critical": 95.0},
                    "inference_latency": {"warning": 2000.0, "critical": 5000.0},
                    "task_throughput": {"warning": 10.0, "critical": 5.0}
                }
            },
            "broadcast_interval": 2.0,
            "enable_auto_start_monitoring": True
        }
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard(request: Request):
            """获取仪表板页面"""
            template_path = os.path.join(self.templates_dir, "dashboard.html")
            if not os.path.exists(template_path):
                # 如果模板不存在，返回一个简单的HTML页面
                html_content = """
                <!DOCTYPE html>
                <html lang="zh-CN">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>性能监控仪表板</title>
                    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
                        .metrics-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                        .metric-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                        .chart-container { height: 300px; }
                        .alerts-container { margin-top: 20px; }
                        .alert { padding: 10px; margin: 5px 0; border-radius: 3px; }
                        .alert-warning { background: #fff3cd; border: 1px solid #ffc107; }
                        .alert-critical { background: #f8d7da; border: 1px solid #dc3545; }
                        .alert-info { background: #d1ecf1; border: 1px solid #0dcaf0; }
                        .stats-panel { margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; border: 1px solid #e9ecef; }
                        .stats-row { display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 14px; }
                        .stats-label { font-weight: bold; color: #495057; }
                        .stats-value { color: #212529; }
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>性能监控仪表板</h1>
                        <p>实时监控系统性能指标</p>
                        <div>
                            <button onclick="startMonitoring()">启动监控</button>
                            <button onclick="stopMonitoring()">停止监控</button>
                            <span id="status">监控状态: 未启动</span>
                        </div>
                    </div>
                    
                    <div class="metrics-container">
                        <div class="metric-card">
                            <h3>CPU使用率</h3>
                            <div id="cpu-chart" class="chart-container"></div>
                            <div id="cpu-value">当前: --%</div>
                        </div>
                        <div class="metric-card">
                            <h3>内存使用率</h3>
                            <div id="memory-chart" class="chart-container"></div>
                            <div id="memory-value">当前: --%</div>
                        </div>
                        <div class="metric-card">
                            <h3>磁盘使用率</h3>
                            <div id="disk-chart" class="chart-container"></div>
                            <div id="disk-value">当前: --%</div>
                        </div>
                        <div class="metric-card">
                            <h3>推理延迟</h3>
                            <div id="inference-chart" class="chart-container"></div>
                            <div id="inference-value">当前: -- ms</div>
                            <div id="inference-stats" class="stats-panel">
                                <div class="stats-row">
                                    <span class="stats-label">成功率:</span>
                                    <span id="inference-success-rate" class="stats-value">--%</span>
                                </div>
                                <div class="stats-row">
                                    <span class="stats-label">样本数:</span>
                                    <span id="inference-sample-count" class="stats-value">--</span>
                                </div>
                                <div class="stats-row">
                                    <span class="stats-label">模型类型:</span>
                                    <span id="inference-model-types" class="stats-value">--</span>
                                </div>
                            </div>
                        </div>
                        <div class="metric-card">
                            <h3>任务吞吐量</h3>
                            <div id="task-chart" class="chart-container"></div>
                            <div id="task-value">当前: -- 任务/秒</div>
                        </div>
                    </div>
                    
                    <div class="alerts-container">
                        <h3>性能警报</h3>
                        <div id="alerts-list"></div>
                    </div>
                    
                    <script>
                        let ws = null;
                        let charts = {};
                        
                        function connectWebSocket() {
                            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                            const wsUrl = `${protocol}//${window.location.host}/ws/performance`;
                            ws = new WebSocket(wsUrl);
                            
                            ws.onopen = function() {
                                console.log('WebSocket连接已建立');
                                updateStatus('监控状态: 已连接');
                            };
                            
                            ws.onmessage = function(event) {
                                const data = JSON.parse(event.data);
                                updateCharts(data);
                                updateAlerts(data.alerts);
                            };
                            
                            ws.onclose = function() {
                                console.log('WebSocket连接已关闭');
                                updateStatus('监控状态: 已断开');
                                // 5秒后重连
                                setTimeout(connectWebSocket, 5000);
                            };
                            
                            ws.onerror = function(error) {
                                console.error('WebSocket错误:', error);
                                updateStatus('监控状态: 连接错误');
                            };
                        }
                        
                        function startMonitoring() {
                            fetch('/api/performance/start', { method: 'POST' })
                                .then(response => response.json())
                                .then(data => {
                                    if (data.status === 'success') {
                                        updateStatus('监控状态: 已启动');
                                        connectWebSocket();
                                    }
                                });
                        }
                        
                        function stopMonitoring() {
                            fetch('/api/performance/stop', { method: 'POST' })
                                .then(response => response.json())
                                .then(data => {
                                    if (data.status === 'success') {
                                        updateStatus('监控状态: 已停止');
                                        if (ws) ws.close();
                                    }
                                });
                        }
                        
                        function updateStatus(message) {
                            document.getElementById('status').textContent = message;
                        }
                        
                        function updateCharts(data) {
                            // 更新CPU图表
                            updateChart('cpu-chart', 'CPU使用率 (%)', data.metrics.cpu || 0, 0, 100);
                            document.getElementById('cpu-value').textContent = `当前: ${data.metrics.cpu || 0}%`;
                            
                            // 更新内存图表
                            updateChart('memory-chart', '内存使用率 (%)', data.metrics.memory || 0, 0, 100);
                            document.getElementById('memory-value').textContent = `当前: ${data.metrics.memory || 0}%`;
                            
                            // 更新磁盘图表
                            updateChart('disk-chart', '磁盘使用率 (%)', data.metrics.disk || 0, 0, 100);
                            document.getElementById('disk-value').textContent = `当前: ${data.metrics.disk || 0}%`;
                            
                            // 更新推理延迟图表
                            let inferenceValue = 0;
                            let inferenceMetadata = {};
                            
                            // 处理推理数据（可能是对象或数值）
                            if (data.metrics.inference && typeof data.metrics.inference === 'object') {
                                inferenceValue = data.metrics.inference.value || 0;
                                inferenceMetadata = data.metrics.inference.metadata || {};
                            } else {
                                inferenceValue = data.metrics.inference || 0;
                            }
                            
                            updateChart('inference-chart', '推理延迟 (ms)', inferenceValue, 0, 15000);
                            document.getElementById('inference-value').textContent = `当前: ${inferenceValue} ms`;
                            
                            // 更新推理统计信息
                            updateInferenceStats(inferenceMetadata);
                            
                            // 更新任务吞吐量图表
                            updateChart('task-chart', '任务吞吐量 (任务/秒)', data.metrics.task || 0, 0, 50);
                            document.getElementById('task-value').textContent = `当前: ${data.metrics.task || 0} 任务/秒`;
                        }
                        
                        function updateChart(chartId, title, value, yMin, yMax) {
                            if (!charts[chartId]) {
                                charts[chartId] = {
                                    data: [],
                                    layout: {
                                        title: title,
                                        xaxis: { title: '时间' },
                                        yaxis: { title: title, range: [yMin, yMax] }
                                    },
                                    config: { responsive: true }
                                };
                            }
                            
                            // 添加新数据点
                            charts[chartId].data.push({
                                x: [new Date().toLocaleTimeString()],
                                y: [value],
                                type: 'scatter',
                                mode: 'lines+markers',
                                name: '当前值'
                            });
                            
                            // 保持最近20个数据点
                            if (charts[chartId].data.length > 20) {
                                charts[chartId].data.shift();
                            }
                            
                            // 更新图表
                            Plotly.react(chartId, charts[chartId].data, charts[chartId].layout, charts[chartId].config);
                        }
                        
                        function updateInferenceStats(metadata) {
                            // 更新成功率
                            const successRate = metadata.success_rate || 0;
                            document.getElementById('inference-success-rate').textContent = `${successRate.toFixed(1)}%`;
                            
                            // 更新样本数
                            const sampleCount = metadata.sample_count || 0;
                            document.getElementById('inference-sample-count').textContent = sampleCount;
                            
                            // 更新模型类型
                            const modelStats = metadata.model_statistics || {};
                            const modelTypes = Object.keys(modelStats);
                            if (modelTypes.length > 0) {
                                document.getElementById('inference-model-types').textContent = modelTypes.slice(0, 3).join(', ');
                                if (modelTypes.length > 3) {
                                    document.getElementById('inference-model-types').textContent += `... (${modelTypes.length} 种模型)`;
                                }
                            } else {
                                document.getElementById('inference-model-types').textContent = '无数据';
                            }
                        }
                        
                        function updateAlerts(alerts) {
                            const alertsList = document.getElementById('alerts-list');
                            alertsList.innerHTML = '';
                            
                            alerts.forEach(alert => {
                                const alertDiv = document.createElement('div');
                                alertDiv.className = `alert alert-${alert.level}`;
                                alertDiv.innerHTML = `
                                    <strong>${alert.level.toUpperCase()}:</strong> 
                                    ${alert.message} (${alert.value} ${alert.metric_type})
                                    <br><small>${new Date(alert.timestamp).toLocaleString()}</small>
                                `;
                                alertsList.appendChild(alertDiv);
                            });
                        }
                        
                        // 页面加载时连接WebSocket
                        window.onload = function() {
                            connectWebSocket();
                        };
                    </script>
                </body>
                </html>
                """
                return HTMLResponse(content=html_content)
            
            return self.templates.TemplateResponse("dashboard.html", {"request": request})
        
        @self.app.get("/api/performance/status", response_model=Dict[str, Any])
        async def get_performance_status():
            """获取性能监控状态"""
            try:
                status = self.dashboard.get_current_status()
                return {"status": "success", "data": status}
            except Exception as e:
                logger.error(f"获取性能状态失败: {e}")
                raise HTTPException(status_code=500, detail="获取性能状态失败")
        
        @self.app.get("/api/performance/metrics", response_model=Dict[str, Any])
        async def get_performance_metrics():
            """获取当前性能指标"""
            try:
                # 获取最新的指标数据
                metrics_data = {}
                for metric_type in MetricType:
                    history = self.dashboard.metrics_history.get(metric_type, [])
                    if history:
                        latest = history[-1]
                        metrics_data[metric_type.value] = {
                            "value": latest.value,
                            "unit": latest.unit,
                            "timestamp": latest.timestamp.isoformat()
                        }
                
                return {"status": "success", "data": metrics_data}
            except Exception as e:
                logger.error(f"获取性能指标失败: {e}")
                raise HTTPException(status_code=500, detail="获取性能指标失败")
        
        @self.app.get("/api/performance/alerts", response_model=Dict[str, Any])
        async def get_performance_alerts():
            """获取性能警报"""
            try:
                alerts_data = []
                for alert in self.dashboard.alerts[-50:]:  # 最近50条警报
                    alerts_data.append({
                        "id": alert.alert_id,
                        "level": alert.level.value,
                        "metric_type": alert.metric_type.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "value": alert.value,
                        "threshold": alert.threshold,
                        "suggestions": alert.suggestions
                    })
                
                return {"status": "success", "data": alerts_data}
            except Exception as e:
                logger.error(f"获取性能警报失败: {e}")
                raise HTTPException(status_code=500, detail="获取性能警报失败")
        
        @self.app.post("/api/performance/start")
        async def start_performance_monitoring():
            """启动性能监控"""
            try:
                self.dashboard.start_monitoring()
                return {"status": "success", "message": "性能监控已启动"}
            except Exception as e:
                logger.error(f"启动性能监控失败: {e}")
                raise HTTPException(status_code=500, detail="启动性能监控失败")
        
        @self.app.post("/api/performance/stop")
        async def stop_performance_monitoring():
            """停止性能监控"""
            try:
                self.dashboard.stop_monitoring()
                return {"status": "success", "message": "性能监控已停止"}
            except Exception as e:
                logger.error(f"停止性能监控失败: {e}")
                raise HTTPException(status_code=500, detail="停止性能监控失败")
        
        @self.app.get("/api/performance/history")
        async def get_performance_history(
            metric_type: Optional[str] = None,
            limit: int = 100
        ):
            """获取历史性能数据"""
            try:
                history_data = {}
                
                if metric_type:
                    # 获取特定指标的历史数据
                    try:
                        mt = MetricType(metric_type)
                        history = list(self.dashboard.metrics_history.get(mt, []))[-limit:]
                        history_data[metric_type] = [
                            {
                                "value": metric.value,
                                "timestamp": metric.timestamp.isoformat(),
                                "unit": metric.unit
                            }
                            for metric in history
                        ]
                    except ValueError:
                        raise HTTPException(status_code=400, detail=f"无效的指标类型: {metric_type}")
                else:
                    # 获取所有指标的历史数据
                    for mt in MetricType:
                        history = list(self.dashboard.metrics_history.get(mt, []))[-limit:]
                        history_data[mt.value] = [
                            {
                                "value": metric.value,
                                "timestamp": metric.timestamp.isoformat(),
                                "unit": metric.unit
                            }
                            for metric in history
                        ]
                
                return {"status": "success", "data": history_data}
            except Exception as e:
                logger.error(f"获取历史数据失败: {e}")
                raise HTTPException(status_code=500, detail="获取历史数据失败")
        
        @self.app.websocket("/ws/performance")
        async def websocket_performance(websocket: WebSocket):
            """性能数据WebSocket端点"""
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    # 等待客户端消息（可选）
                    data = await websocket.receive_text()
                    # 可以处理客户端请求，这里简单忽略
                    pass
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket错误: {e}")
                self.websocket_manager.disconnect(websocket)
        
        # 健康检查端点
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    async def _broadcast_performance_data(self):
        """广播性能数据到所有WebSocket客户端"""
        while self.monitoring_running:
            try:
                # 获取当前性能数据
                metrics_data = {}
                for metric_type in MetricType:
                    history = self.dashboard.metrics_history.get(metric_type, [])
                    if history:
                        latest = history[-1]
                        # 对于推理指标，包含完整的元数据
                        if metric_type == MetricType.INFERENCE:
                            metrics_data[metric_type.value] = {
                                "value": latest.value,
                                "unit": latest.unit,
                                "metadata": latest.metadata
                            }
                        else:
                            metrics_data[metric_type.value] = latest.value
                
                # 获取最近的警报
                alerts_data = []
                for alert in self.dashboard.alerts[-10:]:  # 最近10条警报
                    alerts_data.append({
                        "id": alert.alert_id,
                        "level": alert.level.value,
                        "metric_type": alert.metric_type.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "value": alert.value,
                        "threshold": alert.threshold
                    })
                
                # 准备广播消息
                message = {
                    "type": "performance_update",
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics_data,
                    "alerts": alerts_data
                }
                
                # 广播消息
                await self.websocket_manager.broadcast(message)
                
            except Exception as e:
                logger.error(f"广播性能数据失败: {e}")
            
            # 等待下一次广播
            await asyncio.sleep(self.broadcast_interval)
    
    def start_monitoring_broadcast(self):
        """启动监控数据广播"""
        if self.monitoring_running:
            return
        
        self.monitoring_running = True
        
        async def broadcast_loop():
            await self._broadcast_performance_data()
        
        # 在新线程中运行广播循环
        # 使用兼容Python 3.6的异步事件循环
        def run_broadcast_loop():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(broadcast_loop())
            except Exception as e:
                logger.error(f"广播循环错误: {e}")
        
        self.monitoring_thread = threading.Thread(
            target=run_broadcast_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("性能监控数据广播已启动")
    
    def stop_monitoring_broadcast(self):
        """停止监控数据广播"""
        self.monitoring_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("性能监控数据广播已停止")
    
    def run(self, host: Optional[str] = None, port: Optional[int] = None):
        """运行Web服务器"""
        host = host or self.config.get("host", "127.0.0.1")
        port = port or self.config.get("port", 8088)
        
        # 启动性能监控
        if self.config.get("enable_auto_start_monitoring", True):
            self.dashboard.start_monitoring()
            self.start_monitoring_broadcast()
            logger.info("性能监控已自动启动")
        
        logger.info(f"性能监控Web服务器启动在 http://{host}:{port}")
        
        # 运行FastAPI应用
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )


def create_performance_monitoring_web_server(config: Optional[Dict[str, Any]] = None):
    """创建性能监控Web服务器实例"""
    return PerformanceMonitoringWebServer(config)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="性能监控Web服务器")
    parser.add_argument("--host", default="127.0.0.1", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8088, help="服务器端口")
    parser.add_argument("--no-auto-start", action="store_true", help="不自动启动性能监控")
    
    args = parser.parse_args()
    
    config = {
        "host": args.host,
        "port": args.port,
        "enable_auto_start_monitoring": not args.no_auto_start
    }
    
    try:
        server = create_performance_monitoring_web_server(config)
        server.run(args.host, args.port)
    except Exception as e:
        print(f"启动性能监控Web服务器失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()