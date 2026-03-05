# Self Soul AGI 系统运维指南

## 概述

本运维指南提供Self Soul AGI系统的日常运维管理、监控、故障排除和性能优化指导。

## 日常运维任务

### 1. 系统状态检查

#### 每日检查清单

```bash
#!/bin/bash
# daily_check.sh

echo "=== Self Soul AGI 系统每日检查 ==="

# 检查服务状态
echo "1. 检查服务状态..."
docker-compose ps

# 检查系统资源
echo "2. 检查系统资源..."
free -h
df -h

# 检查API健康状态
echo "3. 检查API健康状态..."
curl -s http://localhost:8000/api/health | jq .

# 检查详细健康状态
echo "4. 检查详细健康状态..."
curl -s http://localhost:8000/api/health/detailed | jq .

# 检查性能指标
echo "5. 检查性能指标..."
curl -s http://localhost:8000/api/metrics/performance | jq .

echo "=== 检查完成 ==="
```

#### 自动化监控脚本

```python
#!/usr/bin/env python3
# monitoring_script.py

import requests
import json
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/self-soul-agi/logs/monitoring.log'),
        logging.StreamHandler()
    ]
)

def check_system_health():
    """检查系统健康状态"""
    try:
        # 检查基础健康状态
        response = requests.get('http://localhost:8000/api/health', timeout=10)
        if response.status_code != 200:
            logging.error(f"健康检查失败: {response.status_code}")
            return False
        
        # 检查详细健康状态
        response = requests.get('http://localhost:8000/api/health/detailed', timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            
            # 检查关键指标
            if health_data['health']['status'] != 'healthy':
                logging.warning("系统健康状态异常")
                return False
                
            # 检查性能指标
            cpu_usage = health_data['health']['metrics']['cpu_percent']
            memory_usage = health_data['health']['metrics']['memory_percent']
            
            if cpu_usage > 90:
                logging.warning(f"CPU使用率过高: {cpu_usage}%")
            
            if memory_usage > 85:
                logging.warning(f"内存使用率过高: {memory_usage}%")
        
        logging.info("系统健康检查通过")
        return True
        
    except Exception as e:
        logging.error(f"健康检查异常: {e}")
        return False

def check_model_status():
    """检查模型状态"""
    try:
        response = requests.get('http://localhost:8000/api/models/status', timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            
            for model in models_data.get('models', []):
                if not model.get('loaded', False):
                    logging.warning(f"模型未加载: {model.get('name', 'unknown')}")
            
            logging.info("模型状态检查完成")
            return True
        
    except Exception as e:
        logging.error(f"模型状态检查异常: {e}")
        return False

if __name__ == "__main__":
    logging.info("开始系统监控检查")
    
    health_ok = check_system_health()
    models_ok = check_model_status()
    
    if health_ok and models_ok:
        logging.info("所有检查通过")
    else:
        logging.error("部分检查失败")
```

### 2. 日志管理

#### 日志文件说明

```
logs/
├── production.log          # 生产环境日志
├── access.log             # 访问日志
├── error.log              # 错误日志
├── monitoring.log         # 监控日志
└── server_startup.log     # 服务器启动日志
```

#### 日志轮转配置

```bash
# 配置logrotate
sudo nano /etc/logrotate.d/self-soul-agi
```

```
/opt/self-soul-agi/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 appuser appuser
    postrotate
        docker-compose restart backend
    endscript
}
```

#### 日志分析命令

```bash
# 查看最新错误
sudo tail -100 logs/error.log | grep -E "(ERROR|CRITICAL)"

# 统计API调用频率
sudo awk '{print $4}' logs/access.log | cut -d: -f2 | sort | uniq -c | sort -nr

# 查找慢请求
sudo grep "WARNING" logs/production.log | grep -i "slow\|timeout"

# 监控实时日志
sudo tail -f logs/production.log | grep -E "(ERROR|WARNING|CRITICAL)"
```

### 3. 备份管理

#### 自动化备份脚本

```python
#!/usr/bin/env python3
# auto_backup.py

import os
import shutil
import tarfile
import datetime
import subprocess
from pathlib import Path

class BackupManager:
    def __init__(self, backup_dir="/backup/self-soul-agi"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def backup_database(self, backup_path):
        """备份数据库"""
        try:
            # 如果有数据库，备份数据库
            # pg_dump self_soul_db > backup_path/database.sql
            pass
        except Exception as e:
            print(f"数据库备份失败: {e}")
    
    def backup_models(self, backup_path):
        """备份模型数据"""
        try:
            models_dir = Path("data/models")
            if models_dir.exists():
                shutil.copytree(models_dir, backup_path / "models")
        except Exception as e:
            print(f"模型备份失败: {e}")
    
    def backup_config(self, backup_path):
        """备份配置文件"""
        try:
            config_files = [".env.production", "docker-compose.yml", "requirements.txt"]
            for file in config_files:
                if Path(file).exists():
                    shutil.copy2(file, backup_path / file)
        except Exception as e:
            print(f"配置备份失败: {e}")
    
    def create_backup(self):
        """创建完整备份"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = self.backup_dir / f"temp_{timestamp}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # 执行备份
            self.backup_database(temp_dir)
            self.backup_models(temp_dir)
            self.backup_config(temp_dir)
            
            # 创建压缩包
            backup_file = self.backup_dir / f"self-soul-agi_{timestamp}.tar.gz"
            with tarfile.open(backup_file, "w:gz") as tar:
                tar.add(temp_dir, arcname="")
            
            # 清理临时文件
            shutil.rmtree(temp_dir)
            
            print(f"备份创建成功: {backup_file}")
            return backup_file
            
        except Exception as e:
            print(f"备份失败: {e}")
            # 清理临时文件
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            return None
    
    def cleanup_old_backups(self, keep_days=30):
        """清理旧备份"""
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=keep_days)
        
        for backup_file in self.backup_dir.glob("self-soul-agi_*.tar.gz"):
            file_time = datetime.datetime.fromtimestamp(backup_file.stat().st_mtime)
            if file_time < cutoff_time:
                backup_file.unlink()
                print(f"删除旧备份: {backup_file}")

if __name__ == "__main__":
    backup_manager = BackupManager()
    backup_manager.create_backup()
    backup_manager.cleanup_old_backups()
```

## 性能优化

### 1. 内存优化

#### 监控内存使用

```bash
# 实时监控内存使用
watch -n 5 'free -h && ps aux --sort=-%mem | head -10'

# 检查内存泄漏
valgrind --leak-check=full python core/main.py --lightweight
```

#### 优化配置

```python
# 在.env.production中配置内存优化
MAX_MEMORY_USAGE=80
WORKER_COUNT=2
MODEL_LOAD_THRESHOLD=70
ENABLE_MEMORY_OPTIMIZATION=true
```

### 2. CPU优化

#### 负载均衡配置

```yaml
# docker-compose.yml中的负载均衡配置
services:
  backend:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

#### CPU亲和性设置

```bash
# 设置CPU亲和性
taskset -c 0,1,2,3 python core/main.py --production
```

### 3. 网络优化

#### 连接池配置

```python
# 在production_config.py中配置连接池
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 30
API_CONNECTION_TIMEOUT = 30
API_READ_TIMEOUT = 60
```

#### 压缩和缓存

```python
# 启用GZIP压缩和缓存
ENABLE_GZIP = true
ENABLE_CACHING = true
CACHE_TTL = 3600  # 1小时
```

## 故障排除

### 1. 常见问题解决

#### 服务无法启动

```bash
# 检查端口占用
netstat -tulpn | grep :8000

# 检查依赖
pip list | grep -E "(fastapi|uvicorn|torch|transformers)"

# 检查日志
tail -f logs/server_startup.log
```

#### 模型加载失败

```bash
# 检查模型文件
ls -la data/models/

# 检查模型权限
ls -la data/models/ | head -10

# 重新初始化模型
curl -X POST http://localhost:8000/api/models/reinitialize
```

#### API响应缓慢

```bash
# 检查性能指标
curl -s http://localhost:8000/api/metrics/performance | jq .

# 检查数据库连接
ps aux | grep postgres

# 检查网络延迟
ping localhost
```

### 2. 调试工具

#### 性能分析

```python
# 使用cProfile进行性能分析
import cProfile
import pstats

def profile_function():
    # 需要分析的函数
    pass

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 执行需要分析的代码
    profile_function()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # 显示前10个最耗时的函数
```

#### 内存分析

```python
# 使用memory_profiler分析内存使用
from memory_profiler import profile

@profile
def memory_intensive_function():
    # 内存密集型操作
    large_list = [i for i in range(1000000)]
    return sum(large_list)

if __name__ == "__main__":
    memory_intensive_function()
```

## 安全运维

### 1. 安全审计

#### 定期安全检查

```bash
# 检查系统漏洞
sudo apt update && sudo apt upgrade

# 检查容器安全
docker scan self-soul-agi-backend

# 检查依赖安全
pip-audit
npm audit
```

#### 安全配置检查

```bash
# 检查防火墙状态
sudo ufw status

# 检查SSL证书有效期
openssl x509 -in /path/to/certificate.crt -noout -dates

# 检查API密钥轮换
grep "API_KEY" .env.production
```

### 2. 访问控制

#### API访问限制

```python
# 在security.py中配置访问控制
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
```

#### 用户权限管理

```python
# 用户权限配置示例
USER_PERMISSIONS = {
    "admin": ["read", "write", "delete", "manage"],
    "user": ["read", "write"],
    "viewer": ["read"]
}
```

## 扩展和升级

### 1. 系统扩展

#### 水平扩展

```yaml
# docker-compose.yml中的水平扩展配置
services:
  backend:
    image: self-soul-agi-backend:latest
    deploy:
      mode: replicated
      replicas: 3
      placement:
        constraints:
          - node.role == worker
    
  frontend:
    image: self-soul-agi-frontend:latest
    deploy:
      mode: global
```

#### 垂直扩展

```bash
# 增加资源限制
docker service update --limit-cpu 4 --limit-memory 8G self-soul-agi_backend
```

### 2. 系统升级

#### 滚动升级

```bash
# 执行滚动升级
docker service update --image self-soul-agi-backend:new-version self-soul-agi_backend

# 监控升级过程
docker service ps self-soul-agi_backend
```

#### 回滚策略

```bash
# 回滚到上一个版本
docker service rollback self-soul-agi_backend

# 检查回滚状态
docker service ps self-soul-agi_backend
```

## 紧急响应

### 1. 紧急情况处理

#### 服务中断

```bash
# 立即重启服务
docker-compose down && docker-compose up -d

# 检查服务状态
docker-compose ps
curl -s http://localhost:8000/api/health
```

#### 安全事件

```bash
# 立即停止服务
docker-compose down

# 备份日志和配置
cp -r logs/ /backup/emergency/
cp .env.production /backup/emergency/

# 检查安全日志
grep -E "(unauthorized|failed|error)" logs/access.log
```

### 2. 恢复流程

#### 数据恢复

```bash
# 从备份恢复数据
tar -xzf /backup/self-soul-agi_20240101_120000.tar.gz -C /tmp/restore/
cp -r /tmp/restore/models data/
cp /tmp/restore/.env.production .

# 重启服务
docker-compose up -d
```

#### 配置恢复

```bash
# 恢复配置文件
cp /backup/emergency/.env.production .
cp /backup/emergency/docker-compose.yml .

# 验证配置
docker-compose config
```

## 监控和告警

### 1. 监控配置

#### Prometheus配置

```yaml
# monitoring/prometheus.yml
scrape_configs:
  - job_name: 'self-soul-agi'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/api/metrics'
    scrape_interval: 30s
```

#### Grafana仪表板

创建监控仪表板，监控以下指标：
- CPU使用率
- 内存使用率
- 磁盘使用率
- API响应时间
- 错误率
- 模型加载状态

### 2. 告警设置

#### 关键指标告警

```yaml
# monitoring/alert_rules.yml
groups:
- name: self-soul-agi
  rules:
  - alert: HighCPUUsage
    expr: 100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 90
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "高CPU使用率"
      
  - alert: HighMemoryUsage
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 85
    for: 5m
    labels:
      severity: warning
```

---

**注意**: 本运维指南需要根据实际部署环境和需求进行调整。定期更新和维护是确保系统稳定运行的关键。