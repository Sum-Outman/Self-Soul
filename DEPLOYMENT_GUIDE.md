# Self Soul - Installation and Deployment Guide

## 多语言版本 / Multi-language Versions

- [中文 (Chinese)](#中文安装部署指南)
- [English](#english-installation-and-deployment-guide)

---

## 中文安装部署指南

### 系统要求

#### 硬件要求
- **最低配置**: 8GB RAM, 4核处理器, 10GB存储空间
- **推荐配置**: 16GB RAM, 8核处理器, 20GB存储空间 (用于所有模型的最佳性能)

#### 软件要求
- Windows 10+/macOS 11+/Linux (Ubuntu 20.04+/Debian 11+)
- Python 3.9+ (推荐: 3.11)
- Node.js 16+ (推荐: 18)
- Git

### 安装步骤

#### 1. 克隆仓库
```bash
git clone https://github.com/Sum-Outman/Self-Soul
cd Self-Soul
```

#### 2. 创建并激活Python虚拟环境
```bash
# 创建虚拟环境
python -m venv .venv

# Windows激活虚拟环境
.venv\Scripts\activate

# Linux/macOS激活虚拟环境
source .venv/bin/activate
```

#### 3. 安装Python依赖
```bash
pip install -r requirements.txt
```

#### 4. 安装前端依赖
```bash
cd app
npm install
cd ..
```

#### 5. 配置环境变量
创建一个 `.env` 文件，添加以下内容：
```
API_KEY=your-api-key-here
MODEL_PATH=./models
DATA_PATH=./data
```

### 启动系统

#### 方法一：分别启动前后端服务

1. **启动后端服务**
```bash
python core/main.py
```

2. **在新终端启动前端服务**
```bash
cd app
npm run dev
```

#### 方法二：使用脚本启动（仅Windows）
```powershell
.\start-app.ps1
```

### 访问系统

打开浏览器，访问：http://localhost:5175

### 服务端口配置

系统使用以下端口：

| 服务 | 端口 | 描述 |
|------|------|------|
| 主API网关 | 8000 | 系统主要入口点，提供RESTful API接口 |
| 前端应用 | 5175 | 用户界面，可通过浏览器访问 |
| 实时数据流管理器 | 8765 | 管理实时数据流和模型间通信 |
| 性能监控服务 | 8081 | 监控系统性能和资源使用情况 |

### 模型端口（8001-8019）
每个AI模型分配了独立端口：
- Manager Model: 8001
- Language Model: 8002
- Knowledge Model: 8003
- Vision Model: 8004
- Audio Model: 8005
- Autonomous Model: 8006
- Programming Model: 8007
- Planning Model: 8008
- Emotion Model: 8009
- Spatial Model: 8010
- Computer Vision Model: 8011
- Sensor Model: 8012
- Motion Model: 8013
- Prediction Model: 8014
- Advanced Reasoning Model: 8015
- Data Fusion Model: 8016
- Creative Problem Solving Model: 8017
- Meta Cognition Model: 8018
- Value Alignment Model: 8019

### 常见问题及解决方案

#### 虚拟环境问题
如果虚拟环境出现问题：
```bash
python -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
python -m venv .venv
```

#### 依赖安装问题
如果pip安装失败：
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 前端启动问题
如果npm run dev失败：
```bash
cd app
rm -rf node_modules
npm install
npm run dev
```

---

## English Installation and Deployment Guide

### System Requirements

#### Hardware Requirements
- **Minimum**: 8GB RAM, 4-core processor, 10GB storage space
- **Recommended**: 16GB RAM, 8-core processor, 20GB storage space (for optimal performance with all models)

#### Software Requirements
- Windows 10+/macOS 11+/Linux (Ubuntu 20.04+/Debian 11+)
- Python 3.9+ (recommended: 3.11)
- Node.js 16+ (recommended: 18)
- Git

### Installation Steps

#### 1. Clone the Repository
```bash
git clone https://github.com/Sum-Outman/Self-Soul
cd Self-Soul
```

#### 2. Create and Activate Python Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on Linux/macOS
source .venv/bin/activate
```

#### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Install Frontend Dependencies
```bash
cd app
npm install
cd ..
```

#### 5. Configure Environment Variables
Create a `.env` file with the following content:
```
API_KEY=your-api-key-here
MODEL_PATH=./models
DATA_PATH=./data
```

### Starting the System

#### Method 1: Start Services Separately

1. **Start Backend Service**
```bash
python core/main.py
```

2. **Start Frontend Service in New Terminal**
```bash
cd app
npm run dev
```

#### Method 2: Use Script (Windows Only)
```powershell
.\start-app.ps1
```

### Accessing the System

Open your browser and visit: http://localhost:5175

### Service Port Configuration

The system uses the following ports:

| Service | Port | Description |
|---------|------|-------------|
| Main API Gateway | 8000 | System's primary entry point, providing RESTful API interface |
| Frontend Application | 5175 | User interface accessible via web browser |
| Realtime Stream Manager | 8765 | Manages real-time data streams and inter-model communication |
| Performance Monitoring | 8081 | Monitors system performance and resource usage |

### Model Ports (8001-8019)
Each AI model has a dedicated port:
- Manager Model: 8001
- Language Model: 8002
- Knowledge Model: 8003
- Vision Model: 8004
- Audio Model: 8005
- Autonomous Model: 8006
- Programming Model: 8007
- Planning Model: 8008
- Emotion Model: 8009
- Spatial Model: 8010
- Computer Vision Model: 8011
- Sensor Model: 8012
- Motion Model: 8013
- Prediction Model: 8014
- Advanced Reasoning Model: 8015
- Data Fusion Model: 8016
- Creative Problem Solving Model: 8017
- Meta Cognition Model: 8018
- Value Alignment Model: 8019

### Troubleshooting

#### Virtual Environment Issues
If you encounter issues with the virtual environment:
```bash
python -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
python -m venv .venv
```

#### Dependency Installation Issues
If pip installation fails:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Frontend Startup Problems
If npm run dev fails:
```bash
cd app
rm -rf node_modules
npm install
npm run dev
```

---

## 联系方式 / Contact Information

For any questions or issues, please contact:
- Email: silencecrowtom@qq.com
- GitHub Repository: https://github.com/Sum-Outman/Self-Soul

© 2025 Self Soul Team. All rights reserved.
