# Self-Soul-B多模态AGI系统监控体系

本文档描述了Self-Soul-B系统的完整监控体系架构、组件和配置指南。

## 📋 目录
- [监控体系概述](#监控体系概述)
- [监控架构](#监控架构)
- [监控组件](#监控组件)
- [配置指南](#配置指南)
- [告警系统](#告警系统)
- [仪表板](#仪表板)
- [故障排除](#故障排除)
- [生产级增强](#生产级增强)

## 🎯 监控体系概述

Self-Soul-B采用多层次、多维度的监控体系，确保多模态AGI系统的稳定运行和性能优化。

### 监控目标
1. **系统健康**: 确保系统组件正常运行
2. **性能优化**: 监控和处理性能瓶颈
3. **故障预警**: 提前发现潜在问题
4. **资源管理**: 优化CPU、内存、磁盘、网络使用
5. **业务洞察**: 跟踪关键业务指标

### 监控层次
| 层次 | 监控对象 | 关键指标 |
|------|----------|----------|
| **基础设施层** | 服务器、网络、存储 | CPU、内存、磁盘、网络IO |
| **应用层** | 应用进程、服务 | 响应时间、错误率、吞吐量 |
| **业务层** | 多模态处理流程 | 处理成功率、延迟、质量 |
| **用户层** | 用户体验 | 用户满意度、交互成功率 |

## 🏗️ 监控架构

### 架构图
```
┌─────────────────────────────────────────────────────────┐
│                   监控仪表板 (Grafana)                   │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                 监控数据聚合 (Prometheus)                │
└─────────────────┬─────────────────┬─────────────────────┘
                  │                 │
    ┌─────────────▼─────┐ ┌─────────▼─────────────┐
    │  应用指标导出     │ │   节点指标导出        │
    │  (FastAPI Metrics)│ │   (Node Exporter)     │
    └─────────┬─────────┘ └─────────┬─────────────┘
              │                     │
    ┌─────────▼─────────────────────▼─────────┐
    │        Self-Soul-B多模态系统            │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐│
    │  │  文本处理│ │  图像处理│ │  音频处理││
    │  └──────────┘ └──────────┘ └──────────┘│
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐│
    │  │  视频处理│ │跨模态融合│ │机器人控制││
    │  └──────────┘ └──────────┘ └──────────┘│
    └─────────────────────────────────────────┘
```

### 数据流
1. **指标收集**: 应用和系统组件生成监控指标
2. **指标导出**: 通过HTTP端点暴露指标数据
3. **指标抓取**: Prometheus定期抓取指标
4. **数据存储**: Prometheus TSDB存储历史数据
5. **数据可视化**: Grafana展示监控仪表板
6. **告警触发**: Alertmanager处理告警规则

## 🔧 监控组件

### 1. 系统监控模块 (`core/monitoring.py`)

#### 功能
- 系统健康检查
- 资源使用监控（CPU、内存、磁盘、网络）
- 性能指标收集
- 历史数据存储

#### 关键类
- `HealthChecker`: 健康检查器
- `PerformanceMonitor`: 性能监控器
- `SystemMetrics`: 系统指标数据类

#### 使用示例
```python
from core.monitoring import health_checker, performance_monitor

# 检查系统健康
health_status = await health_checker.check_system_health()

# 获取性能统计
performance_stats = performance_monitor.get_performance_stats()
```

### 2. 增强监控模块 (`core/monitoring_enhanced.py`)

#### 功能
- 增强的系统指标收集
- 实时监控数据
- 模型性能监控
- 协作任务监控

#### 关键方法
- `get_realtime_monitoring()`: 获取实时监控数据
- `get_enhanced_metrics()`: 获取增强指标
- `get_performance_stats()`: 获取性能统计

### 3. 核心指标收集器 (`core/core_metrics_collector.py`)

#### 功能
- 核心AGI指标收集
- 演化成功率监控
- 推理延迟分析
- 硬件资源占用监控
- 日志审计和分析

#### 关键指标
- `evolution_success_rate`: 演化成功率
- `inference_latency_jitter`: 推理延迟抖动
- `hardware_utilization`: 硬件资源占用率
- `repetition_output_rate`: 重复输出率

### 4. 监控仪表板后端 (`core/monitoring_dashboard_backend.py`)

#### 功能
- 监控数据聚合
- 实时数据推送
- 历史数据查询
- 监控数据存储

### 5. 性能监控器 (`tests/performance_monitor.py`)

#### 功能
- 函数级性能监控
- 资源使用测量
- 性能分析装饰器
- 监控数据导出

## ⚙️ 配置指南

### Prometheus配置 (`monitoring/prometheus.yml`)

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'self-soul-backend'
    static_configs:
      - targets: ['self-soul-backend:8000']
    metrics_path: '/api/metrics'
    scrape_interval: 30s
    scrape_timeout: 10s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### 应用指标端点

#### 健康检查端点
```bash
# 基础健康检查
GET /api/health

# 详细健康检查
GET /api/health/detailed

# 性能指标
GET /api/metrics/performance

# Prometheus指标
GET /api/metrics
```

#### 指标格式示例
```json
{
  "status": "healthy",
  "timestamp": "2026-03-06T00:42:39.732469",
  "metrics": {
    "cpu_percent": 45.2,
    "memory_percent": 68.5,
    "disk_percent": 32.1,
    "process_count": 42,
    "model_loaded_count": 15,
    "active_connections": 8
  }
}
```

### Docker Compose监控配置

在`docker-compose.yml`中添加监控服务：

```yaml
services:
  # Prometheus监控
  prometheus:
    image: prom/prometheus:latest
    container_name: self-soul-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped
    networks:
      - self-soul-network

  # Grafana仪表板
  grafana:
    image: grafana/grafana:latest
    container_name: self-soul-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - self-soul-network

  # Node Exporter（系统指标）
  node-exporter:
    image: prom/node-exporter:latest
    container_name: self-soul-node-exporter
    ports:
      - "9100:9100"
    restart: unless-stopped
    networks:
      - self-soul-network

volumes:
  prometheus_data:
  grafana_data:
```

## 🚨 告警系统

### 告警规则配置

创建`monitoring/alert_rules.yml`:

```yaml
groups:
  - name: self_soul_alerts
    rules:
      # CPU使用率告警
      - alert: HighCPUUsage
        expr: process_cpu_seconds_total > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "高CPU使用率"
          description: "CPU使用率超过80%持续5分钟"
      
      # 内存使用率告警
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / machine_memory_bytes > 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "高内存使用率"
          description: "内存使用率超过85%持续5分钟"
      
      # 服务宕机告警
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "服务宕机"
          description: "{{ $labels.job }} 服务已宕机"
      
      # 响应时间告警
      - alert: HighResponseTime
        expr: http_request_duration_seconds{quantile="0.95"} > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "高响应时间"
          description: "95%请求响应时间超过2秒"
```

### 告警通知渠道

配置Alertmanager支持多种通知渠道：

1. **电子邮件通知**
2. **Slack/Teams通知**
3. **Webhook通知**
4. **短信通知**（通过第三方服务）
5. **电话通知**（通过PagerDuty等）

## 📊 仪表板

### Grafana仪表板配置

创建多模态系统监控仪表板：

#### 1. 系统资源仪表板
- CPU使用率（按核心）
- 内存使用趋势
- 磁盘IO性能
- 网络流量监控

#### 2. 应用性能仪表板
- 请求响应时间（P50, P95, P99）
- 请求吞吐量（QPS）
- 错误率和异常统计
- 并发连接数

#### 3. 多模态处理仪表板
- 各模态处理延迟（文本、图像、音频、视频）
- 跨模态融合成功率
- 模型推理性能
- 硬件加速器使用率

#### 4. 业务指标仪表板
- 用户会话统计
- 处理任务成功率
- 资源使用效率
- 成本效益分析

### 仪表板示例

```json
{
  "dashboard": {
    "title": "Self-Soul-B多模态系统监控",
    "panels": [
      {
        "title": "系统资源",
        "type": "row",
        "panels": [
          {
            "title": "CPU使用率",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(process_cpu_seconds_total[5m]) * 100",
                "legendFormat": "{{instance}}"
              }
            ]
          },
          {
            "title": "内存使用",
            "type": "graph",
            "targets": [
              {
                "expr": "process_resident_memory_bytes",
                "legendFormat": "内存使用"
              }
            ]
          }
        ]
      }
    ]
  }
}
```

## 🔍 故障排除

### 常见监控问题

#### 1. 指标无法收集
**症状**: Prometheus显示"DOWN"状态
**解决方案**:
```bash
# 检查应用是否运行
curl http://localhost:8000/api/health

# 检查指标端点
curl http://localhost:8000/api/metrics

# 检查Prometheus配置
docker-compose logs prometheus
```

#### 2. 监控数据不准确
**症状**: 指标值异常或缺失
**解决方案**:
1. 检查指标导出逻辑
2. 验证指标标签格式
3. 检查时间同步（NTP）
4. 验证采样间隔配置

#### 3. 告警不触发
**症状**: 条件满足但未触发告警
**解决方案**:
1. 检查告警规则语法
2. 验证指标名称和标签
3. 检查Prometheus规则文件加载
4. 验证Alertmanager配置

#### 4. 性能监控延迟高
**症状**: 监控数据更新延迟
**解决方案**:
1. 优化指标收集频率
2. 增加Prometheus资源
3. 使用更高效的数据结构
4. 考虑分片监控

### 监控调试命令

```bash
# 检查Prometheus目标状态
curl http://localhost:9090/api/v1/targets

# 检查告警规则
curl http://localhost:9090/api/v1/rules

# 查询特定指标
curl "http://localhost:9090/api/v1/query?query=process_cpu_seconds_total"

# 检查Grafana数据源
curl -u admin:admin http://localhost:3000/api/datasources
```

## 🏭 生产级增强

### 1. 分布式监控

对于大规模部署，考虑以下增强：

#### 监控联邦
```yaml
# prometheus.yml
federate:
  scrape_configs:
    - job_name: 'federate'
      honor_labels: true
      metrics_path: '/federate'
      params:
        'match[]':
          - '{job="self-soul-backend"}'
      static_configs:
        - targets:
          - 'prometheus-central:9090'
```

#### 时序数据库扩展
- 使用Thanos或Cortex进行长期存储
- 实现监控数据分片
- 添加数据压缩和降采样

### 2. 智能告警

#### 动态阈值调整
```python
class AdaptiveAlerting:
    """自适应告警"""
    
    def adjust_thresholds(self, historical_data):
        """根据历史数据调整阈值"""
        # 基于季节性和趋势调整
        pass
    
    def detect_anomalies(self, current_metrics):
        """异常检测"""
        # 使用机器学习检测异常模式
        pass
```

#### 告警关联分析
- 关联相关告警减少告警风暴
- 根本原因分析
- 告警优先级动态调整

### 3. 监控即代码

#### 基础设施即代码
```hcl
# Terraform监控配置
resource "grafana_dashboard" "self_soul" {
  config_json = file("${path.module}/dashboards/self-soul.json")
}

resource "prometheus_rule_group" "alerts" {
  name = "self-soul-alerts"
  
  rule {
    alert = "HighCPUUsage"
    expr  = "process_cpu_seconds_total > 0.8"
    for   = "5m"
  }
}
```

#### GitOps监控管理
- 监控配置版本控制
- 自动部署监控变更
- 监控配置审计跟踪

### 4. 安全监控

#### 安全事件监控
- 异常访问模式检测
- API滥用检测
- 数据泄露监控
- 合规性监控

#### 监控数据保护
- 监控数据加密
- 访问控制审计
- 监控数据保留策略
- GDPR合规性

### 5. 成本监控

#### 资源成本优化
```python
class CostMonitor:
    """成本监控器"""
    
    def calculate_resource_cost(self, usage_metrics):
        """计算资源成本"""
        # 基于云提供商定价计算成本
        pass
    
    def optimize_cost(self, usage_patterns):
        """成本优化建议"""
        # 提供资源优化建议
        pass
```

#### 成本告警
- 预算超支告警
- 异常成本检测
- 成本效益分析

## 📈 性能优化

### 监控系统自身优化

#### 1. 指标采样优化
```python
# 自适应采样
def adaptive_sampling(metric_value, historical_variance):
    """根据指标变化率调整采样频率"""
    if historical_variance > threshold:
        return high_frequency
    else:
        return low_frequency
```

#### 2. 数据存储优化
- 使用列式存储优化查询性能
- 实施数据分区和索引
- 定期清理过期数据

#### 3. 查询优化
- 预计算常用查询
- 使用查询缓存
- 优化查询语法

### 监控数据质量

#### 数据验证
```python
def validate_metrics(metrics_data):
    """验证监控数据质量"""
    checks = [
        check_timestamp_freshness,
        check_value_range,
        check_consistency,
        check_completeness
    ]
    
    for check in checks:
        if not check(metrics_data):
            raise MetricValidationError(f"检查失败: {check.__name__}")
```

#### 数据修复
- 自动数据插值
- 异常数据检测和修复
- 数据一致性维护

## 🤝 集成指南

### 与CI/CD集成

#### 监控驱动的部署
```yaml
# GitHub Actions工作流
jobs:
  deploy:
    steps:
      - name: 部署前健康检查
        run: |
          python scripts/health_check.py
      
      - name: 部署
        run: |
          docker-compose up -d
      
      - name: 部署后监控验证
        run: |
          python scripts/verify_monitoring.py
```

#### 性能基准测试集成
```python
# 性能测试与监控集成
def performance_test_with_monitoring():
    """带监控的性能测试"""
    start_monitoring()
    run_performance_tests()
    results = collect_monitoring_data()
    generate_performance_report(results)
```

### 与运维工具集成

#### 与Kubernetes集成
- 使用Prometheus Operator
- 自动服务发现
- 动态监控配置

#### 与消息队列集成
- 监控消息处理延迟
- 队列深度监控
- 消费者性能监控

## 📚 最佳实践

### 监控设计原则

1. **可观测性原则**: 监控应提供完整的系统可观测性
2. **最小特权原则**: 监控系统应有最小必要权限
3. **数据质量原则**: 确保监控数据的准确性和完整性
4. **性能影响原则**: 监控不应显著影响系统性能
5. **可维护性原则**: 监控系统应易于维护和扩展

### 监控配置最佳实践

1. **标准化指标命名**: 使用一致的指标命名约定
2. **合理设置阈值**: 基于历史数据设置合理阈值
3. **分层告警策略**: 实现分层告警减少告警风暴
4. **定期审查监控**: 定期审查和优化监控配置
5. **监控文档化**: 完整记录监控配置和决策

### 监控运维最佳实践

1. **监控系统自监控**: 监控系统自身健康状况
2. **容量规划**: 规划监控系统容量
3. **备份和恢复**: 实施监控数据备份策略
4. **安全加固**: 加强监控系统安全
5. **性能优化**: 持续优化监控系统性能

---

*本文档最后更新: 2026-03-06*
*版本: v1.0.0*