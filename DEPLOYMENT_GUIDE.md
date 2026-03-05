# Self-Soul AGI 系统部署指南

## 概述

本文档提供了Self-Soul AGI系统的部署指南，重点关注新的配置管理系统和安全升级后的部署流程。

## 目录

1. [环境要求](#环境要求)
2. [配置管理系统](#配置管理系统)
3. [前端部署](#前端部署)
4. [后端部署](#后端部署)
5. [安全配置](#安全配置)
6. [环境变量配置](#环境变量配置)
7. [生产环境部署](#生产环境部署)
8. [故障排除](#故障排除)

## 环境要求

### 系统要求
- **操作系统**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 10.15+
- **Python**: 3.8+
- **Node.js**: 16+
- **内存**: 最低8GB，推荐16GB
- **磁盘空间**: 最低10GB可用空间

### 软件依赖
- **后端**: FastAPI, Uvicorn, 各种AI模型库
- **前端**: Vue 3, Vite, Pinia
- **数据库**: SQLite (默认), 可选PostgreSQL/MySQL

## 配置管理系统

### 架构概述

系统现在包含两个独立的配置管理系统：

1. **后端配置管理器** (`core/config_manager.py`)
   - 统一配置管理
   - 支持JSON、YAML、环境变量等多种配置源
   - 配置验证和热重载
   - 优先级: 环境变量 > 配置文件 > 默认值

2. **前端配置管理器** (`app/src/utils/config/configManager.js`)
   - Vue插件形式提供全局配置访问
   - 支持环境变量和localStorage
   - 类型安全的配置访问

### 重要修复

#### 配置验证修复
在v1.1.0版本中，修复了以下配置验证问题：

1. **模型端口验证**: 修复了`data_fusion`端口8028超出验证范围(8001-8026)的问题
   - 解决方案: 将端口验证范围扩展到8001-8028
   - 文件: `core/config_manager.py`

2. **缺少的模型配置**: 添加了缺失的模型配置键
   - 添加了: `multi_model_collaboration` 和 `mathematics`
   - 确保配置验证与`config/model_services_config.json`保持一致

3. **配置验证增强**: 配置管理器现在会验证所有配置项，确保配置的完整性和一致性

#### 安全配置修复
1. **主机绑定安全**: 默认绑定到`127.0.0.1`而非`0.0.0.0`
2. **eval()替换**: 使用安全的表达式求值器替代危险的`eval()`调用
3. **CORS安全**: 基于环境的CORS配置，生产环境严格限制来源

### 配置文件结构

```
项目根目录/
├── .env                    # 后端环境变量
├── .env.development       # 开发环境配置
├── .env.production        # 生产环境配置
├── config/
│   ├── model_services_config.json  # 模型服务配置
│   └── performance.yml             # 性能配置
├── app/
│   ├── .env.development   # 前端开发环境
│   └── .env.production    # 前端生产环境
└── core/data/settings/
    └── system_settings.json  # 系统设置
```

## 前端部署

### 开发环境

1. **安装依赖**
   ```bash
   cd app
   npm install
   ```

2. **配置环境变量**
   ```bash
   # 复制环境变量文件
   cp .env.development .env
   
   # 或手动创建.env文件
   # 内容参考: app/.env.development
   ```

3. **启动开发服务器**
   ```bash
   npm run dev
   ```
   - 访问: http://localhost:5175
   - 支持热重载和开发工具

### 生产环境构建

1. **构建前端应用**
   ```bash
   cd app
   npm run build
   ```
   - 构建结果位于 `app/dist` 目录

2. **配置生产环境变量**
   ```bash
   # 复制生产环境配置
   cp .env.production .env
   
   # 根据实际环境修改配置
   ```

3. **预览构建结果**
   ```bash
   npm run preview
   ```

### 配置说明

#### 前端环境变量 (Vite)

所有前端环境变量必须以 `VITE_` 开头：

```env
# API配置
VITE_API_BASE_URL=

# 系统配置
VITE_FRONTEND_PORT=5175
VITE_BACKEND_PORT=8000
VITE_ENVIRONMENT=development
VITE_DEBUG=true
VITE_LOG_LEVEL=debug

# 服务配置
VITE_REALTIME_STREAM_HOST=localhost
VITE_REALTIME_STREAM_PORT=8025
VITE_VALUE_ALIGNMENT_HOST=localhost
VITE_VALUE_ALIGNMENT_PORT=8019
```

## 后端部署

### 开发环境

1. **创建虚拟环境**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置环境变量**
   ```bash
   # 复制环境变量文件
   cp .env.example .env
   
   # 编辑.env文件，设置实际值
   ```

4. **启动后端服务器**
   ```bash
   python -m progressive_server
   ```
   - API文档: http://localhost:8000/docs
   - 健康检查: http://localhost:8000/health

### 生产环境

1. **使用生产环境变量**
   ```bash
   cp .env.production .env
   # 编辑.env文件，设置生产环境值
   ```

2. **使用Gunicorn部署 (Linux/macOS)**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker progressive_server:app
   ```

3. **使用Windows服务部署**
   - 使用NSSM创建Windows服务
   - 或使用IIS + wfastcgi

### 后端环境变量

```env
# CORS安全配置
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5175

# API密钥配置
MODEL_SERVICE_API_KEY=your_model_service_key_here
SYSTEM_MONITOR_API_KEY=your_system_monitor_key_here
REALTIME_STREAM_API_KEY=your_realtime_stream_key_here

# 服务器配置
SERVER_HOST=127.0.0.1  # 安全绑定，避免0.0.0.0
VALUE_ALIGNMENT_HOST=127.0.0.1

# 其他配置
DEBUG=False
LOG_LEVEL=INFO
ENVIRONMENT=production  # 或 development
```

## 安全配置

### 关键安全升级

1. **消除eval()风险**
   - 使用`SafeExpressionEvaluator`替代危险的`eval()`调用
   - 位置: `core/causal_reasoning_enhancer.py`

2. **修复主机绑定**
   - 默认绑定到`127.0.0.1`而非`0.0.0.0`
   - 支持环境变量覆盖: `SERVER_HOST`, `VALUE_ALIGNMENT_HOST`

3. **CORS安全配置**
   - 基于环境的CORS配置
   - 生产环境严格限制来源
   - 位置: `core/main.py`

4. **输入验证增强**
   - 数据库字段名验证防止SQL注入
   - 前端XSS防护 (使用textContent替代innerHTML)

### 安全最佳实践

1. **API密钥管理**
   ```env
   # 使用环境变量存储敏感信息
   MODEL_SERVICE_API_KEY=your_secure_key_here
   ```

2. **HTTPS配置**
   ```bash
   # 生产环境必须使用HTTPS
   # 配置SSL证书
   ```

3. **防火墙规则**
   - 只开放必要的端口 (8000, 5175等)
   - 使用反向代理 (Nginx/Apache)

## 环境变量配置

### 环境类型

系统支持三种环境类型：

1. **development** - 开发环境
   - 宽松的CORS配置
   - 详细日志
   - 热重载支持

2. **staging** - 预生产环境
   - 接近生产环境的配置
   - 用于测试

3. **production** - 生产环境
   - 严格的安全配置
   - 性能优化
   - 最小化日志

### 环境变量优先级

1. 环境变量 (最高优先级)
2. 配置文件 (`config/` 目录)
3. 默认值 (代码中定义)

### 重要环境变量

| 变量名 | 默认值 | 描述 |
|--------|--------|------|
| `ENVIRONMENT` | `development` | 运行环境 |
| `SERVER_HOST` | `127.0.0.1` | 服务器绑定地址 |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5175` | 允许的CORS来源 |
| `DEBUG` | `False` | 调试模式 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

## 生产环境部署

### Docker部署 (推荐)

1. **构建Docker镜像**
   ```dockerfile
   # Dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   CMD ["python", "-m", "progressive_server"]
   ```

2. **Docker Compose配置**
   ```yaml
   # docker-compose.yml
   version: '3.8'
   services:
     backend:
       build: .
       ports:
         - "8000:8000"
       environment:
         - ENVIRONMENT=production
         - SERVER_HOST=0.0.0.0
       volumes:
         - ./data:/app/data
   
     frontend:
       build: ./app
       ports:
         - "5175:5175"
       depends_on:
         - backend
   ```

### 手动部署步骤

1. **准备服务器**
   ```bash
   # 更新系统
   sudo apt update && sudo apt upgrade -y
   
   # 安装必要软件
   sudo apt install python3-pip nginx -y
   ```

2. **部署后端**
   ```bash
   # 克隆代码
   git clone <repository-url>
   cd self-soul-b
   
   # 设置虚拟环境
   python3 -m venv venv
   source venv/bin/activate
   
   # 安装依赖
   pip install -r requirements.txt
   
   # 配置环境变量
   cp .env.production .env
   nano .env  # 编辑配置
   
   # 启动服务
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker progressive_server:app
   ```

3. **部署前端**
   ```bash
   cd app
   npm install
   npm run build
   
   # 配置Nginx服务静态文件
   sudo cp nginx.conf /etc/nginx/sites-available/self-soul
   sudo ln -s /etc/nginx/sites-available/self-soul /etc/nginx/sites-enabled/
   sudo systemctl restart nginx
   ```

### 监控和维护

1. **日志管理**
   ```bash
   # 查看后端日志
   tail -f logs/app.log
   
   # 查看错误日志
   tail -f logs/error.log
   ```

2. **性能监控**
   - 使用`/api/health/detailed`端点监控系统健康状态
   - 监控CPU、内存、磁盘使用率

3. **备份策略**
   ```bash
   # 备份数据库
   cp data/self_soul.db backup/self_soul-$(date +%Y%m%d).db
   
   # 备份配置文件
   tar -czf backup/config-$(date +%Y%m%d).tar.gz config/
   ```

## 故障排除

### 常见问题

1. **端口占用**
   ```
   错误: [Errno 98] Address already in use
   解决方案: 更改端口或停止占用端口的进程
   ```

2. **CORS错误**
   ```
   错误: CORS policy blocked request
   解决方案: 检查CORS_ALLOWED_ORIGINS配置
   ```

3. **数据库连接失败**
   ```
   错误: Unable to connect to database
   解决方案: 检查数据库文件权限和路径
   ```

4. **前端无法连接后端**
   ```
   错误: Connection refused
   解决方案: 
   1. 确保后端服务正在运行
   2. 检查VITE_API_BASE_URL配置
   3. 检查网络防火墙设置
   ```

### 调试技巧

1. **启用详细日志**
   ```env
   DEBUG=True
   LOG_LEVEL=DEBUG
   ```

2. **检查配置加载**
   ```bash
   # 后端配置摘要
   curl http://localhost:8000/api/config/summary
   
   # 前端配置检查
   # 在浏览器控制台中输入: configManager.getAll()
   ```

3. **验证服务状态**
   ```bash
   # 健康检查
   curl http://localhost:8000/health
   
   # 详细状态
   curl http://localhost:8000/api/health/detailed
   ```

### 获取帮助

1. **查看日志文件**
   - `logs/app.log` - 应用日志
   - `logs/error.log` - 错误日志

2. **查阅文档**
   - API文档: http://localhost:8000/docs
   - 帮助页面: http://localhost:5175/#/help

3. **系统信息收集**
   ```bash
   # 收集部署信息
   python -c "import sys; print(f'Python {sys.version}')"
   pip list
   node --version
   npm --version
   ```

---

## 更新记录

### v1.1.0 - 安全升级和配置管理
- ✅ 实施统一配置管理系统
- ✅ 修复安全漏洞 (eval(), XSS, CORS等)
- ✅ 增强前端配置管理
- ✅ 改进类型提示和代码质量
- ✅ 更新部署文档

### 下一步计划
- 实现完整的Docker化部署
- 添加CI/CD流水线
- 增强监控和告警功能
- 优化AGI核心能力

---

**重要提醒**: 生产环境部署前，请确保:
1. 所有敏感信息使用环境变量存储
2. 启用HTTPS加密通信
3. 配置适当的防火墙规则
4. 设置定期备份策略
5. 监控系统性能和安全性