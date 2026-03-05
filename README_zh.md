# Self Soul - 高级AGI系统

**开发团队**: Self Soul Team  
**开发人员邮箱**: silencecrowtom@qq.com  
**仓库地址**: https://github.com/Sum-Outman/Self-Soul

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Stars](https://img.shields.io/github/stars/Sum-Outman/Self-Soul)](https://github.com/Sum-Outman/Self-Soul/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/Sum-Outman/Self-Soul)](https://github.com/Sum-Outman/Self-Soul/issues)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Vue.js Version](https://img.shields.io/badge/Vue.js-3.4%2B-green.svg)](https://vuejs.org/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.4%2B-orange.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-20.10%2B-blue.svg)](https://www.docker.com/)

## 目录

- [最新更新](#最新更新)
- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [模型架构](#模型架构)
- [安装与部署](#安装与部署)
- [配置说明](#配置说明)
- [安全特性](#安全特性)
- [性能优化](#性能优化)
- [API文档](#api文档)
- [端口配置](#端口配置)
- [硬件集成](#硬件集成)
- [使用指南](#使用指南)
- [故障排除](#故障排除)
- [贡献指南](#贡献指南)
- [许可证](#许可证)
- [致谢](#致谢)

## 最新更新 (2026年3月)

Self Soul系统已进行了重大改进和增强：

### 🚀 **主要性能优化**
- **智能内存管理**: 动态模型加载与LRU卸载策略
- **8模型限制**: 最大同时加载8个模型，防止内存溢出(OOM)
- **核心模型保护**: 优先保护管理器、语言和知识模型
- **访问时间跟踪**: 所有模型跟踪最后访问时间以优化卸载决策
- **Torch编译支持**: 模型编译优化以加速推理

### 🔒 **增强的安全特性**
- **WebSocket认证**: WebSocket连接的API密钥和JWT令牌认证
- **生产环境安全强制执行**: 生产环境必须设置数据库密码
- **非root容器**: Docker容器以非root用户运行以提高安全性
- **细粒度Docker权限**: 使用特定功能替代特权模式(SYS_RAWIO, SYS_ADMIN等)
- **Redis安全**: 生产环境中的Redis连接密码验证

### 🎯 **部署和硬件改进**
- **Python 3.11升级**: Docker镜像更新至Python 3.11以获得更好的兼容性
- **硬件访问组**: 容器用户添加到dialout、video和gpio组
- **现代化前端**: Vue.js依赖更新(vue-i18n v11.0.0)
- **类型安全**: 完整的TypeScript迁移和全面类型检查

### 🧠 **AGI核心能力完善**
- **完整训练循环**: 集成DataLoader和完整的基于epoch的训练周期
- **联合模型训练**: 能够协同训练所有27个模型
- **自进化架构**: 神经架构搜索(NAS)与层变异
- **维度一致性**: 所有神经网络中的统一张量维度
- **跨模态融合**: 用于图像和音频处理的ResNet和LSTM编码器

### 📊 **代码质量和维护**
- **实验脚本组织**: 整理到scripts/目录并标准化CLI接口
- **导入错误修复**: 全面的导入验证和错误处理
- **文档同步**: 更新英文和中文文档

## 项目概述

Self Soul是一个复杂的通用人工智能(AGI)平台，将**27个专业AI模型**集成到统一的认知架构中，并支持**18个外部API提供商**以实现灵活的部署选项。这个开源系统提供了全面的多模态智能能力，包括自然语言处理、计算机视觉、音频分析、情感智能、自主学习和高级推理。

### 设计理念

Self Soul基于一个核心原则构建：真正的AGI需要一个凝聚、集成的架构，而不是孤立的模型。我们的设计理念强调：

- **统一认知架构**: 所有27个模型通过中央协调系统协同工作，实现超越单个组件总和的涌现智能
- **从零开始训练**: 通过从头开始训练所有模型而不依赖预训练基础，我们保持对模型开发、伦理对齐和AGI合规性的完全控制
- **以人为本的智能**: 集成情感智能和价值对齐，确保AI行为负责任、符合伦理并与人类价值观一致
- **模块化扩展性**: 灵活的架构允许轻松集成新的AI能力，同时保持系统一致性

### 技术亮点

- **27个专业AI模型**: 全面覆盖从基本感知到高级推理的认知能力
- **先进的多模态集成**: 无缝处理和融合文本、图像、音频和视频数据
- **自适应学习引擎**: 基于性能指标实时优化学习策略和训练参数
- **分布式处理**: 每个模型在专用端口(8001-8027)上运行，实现并行处理和可扩展性
- **现代UI/UX**: 基于Vue.js的直观仪表板，用于系统管理和监控
- **全面的API**: RESTful接口，用于与外部系统和应用程序集成
- **多模态数据集支持**: 扩展的Multimodal Dataset v1支持所有27个模型的全面训练
- **全面的外部API支持**: 集成18个外部API提供商，包括OpenAI、Anthropic、Google AI、AWS、Azure、HuggingFace以及国内提供商如DeepSeek、智谱AI和百度文心

## 核心特性

### 🧠 **智能内存管理**
- **动态模型加载**: 根据任务需求按需加载模型
- **LRU卸载策略**: 当内存限制达到时卸载最近最少使用的模型
- **核心模型保护**: 对基本模型(管理器、语言、知识)提供优先保护
- **8模型限制**: 最大同时加载8个模型，防止内存耗尽
- **访问时间跟踪**: 自动跟踪模型使用情况以优化卸载决策

### 🔐 **高级安全**
- **WebSocket认证**: 实时流的双重认证(API密钥 + JWT令牌)
- **生产环境安全**: 生产环境数据库的强制性密码要求
- **非root执行**: Docker容器以最小权限的非root用户运行
- **细粒度能力**: 使用特定Linux能力替代特权模式
- **基于环境的安全**: 开发和生产环境的不同安全配置

### 🚀 **性能优化**
- **惰性加载**: 仅在需要时加载模型，减少初始内存占用
- **模型量化**: 支持动态量化以减少模型大小
- **Torch编译**: JIT编译优化以加速模型执行
- **基于任务的加载**: 根据任务需求智能选择模型
- **资源监控**: CPU、内存和GPU使用情况的实时监控

### 🎯 **AGI核心能力**
- **完整训练系统**: 包含DataLoader、epochs和验证的完整训练循环
- **联合训练**: 所有27个模型可以协同训练
- **神经架构搜索**: 模型架构的自动优化
- **跨模态融合**: 视觉、音频和文本处理的集成
- **自进化**: 模型可以通过变异操作进化自己的架构

### 🌐 **硬件集成**
- **多摄像头支持**: 同时处理多个摄像头输入
- **传感器集成**: 支持各种环境和运动传感器
- **机器人控制**: 用于控制机器人系统和执行器的API
- **实时通信**: 基于WebSocket的实时数据流
- **协议支持**: 串行、TCP/IP、UDP和摄像头接口

## 系统架构

### 核心组件

```
Self Soul AGI系统
├── 中央协调器 (端口 8000)
│   ├── 模型注册表 (动态模型管理)
│   ├── 任务调度器 (智能任务分配)
│   ├── 安全管理器 (认证与授权)
│   └── 性能监视器 (实时指标)
├── 27个专业模型 (端口 8001-8027)
│   ├── 管理器模型 (端口 8001) - 系统协调
│   ├── 语言模型 (端口 8002) - 自然语言处理
│   ├── 知识模型 (端口 8003) - 知识管理
│   ├── 视觉模型 (端口 8004) - 计算机视觉
│   ├── 音频模型 (端口 8005) - 音频处理
│   └── ... (23个其他专业模型)
├── 前端仪表板 (端口 5175)
│   ├── 系统监控
│   ├── 模型管理
│   ├── 训练界面
│   └── 硬件控制
└── 外部服务
    ├── PostgreSQL数据库 (可选)
    ├── Redis缓存 (可选)
    └── 外部API提供商 (18个提供商)
```

### 模型协调架构

系统采用复杂的模型协调机制：

1. **任务分析**: 分析输入任务以确定所需能力
2. **模型选择**: 基于任务要求，选择5-8个相关模型
3. **动态加载**: 将选定模型加载到内存中(如果尚未加载)
4. **协同处理**: 模型协同工作处理任务
5. **结果融合**: 将各个模型输出融合成一致响应
6. **内存管理**: 基于LRU策略卸载未使用的模型

### 内存管理架构

智能内存管理系统确保最优的资源利用：

```
内存管理流程:
1. 接收任务 → 识别所需模型
2. 检查已加载模型 → 加载缺失模型
3. 如果>8个模型已加载 → 卸载LRU非核心模型
4. 处理任务 → 更新模型访问时间
5. 监控内存使用 → 必要时触发清理
6. 定期维护 → 清理过期的工作流和记录
```

## 模型架构

### 完整模型列表 (27个模型)

| 模型名称 | 端口 | 描述 | 核心模型 |
|----------|------|------|----------|
| 管理器 | 8001 | 系统协调和任务分发 | ✅ |
| 语言 | 8002 | 自然语言处理和生成 | ✅ |
| 知识 | 8003 | 知识管理和检索 | ✅ |
| 视觉 | 8004 | 计算机视觉和图像处理 | |
| 音频 | 8005 | 音频处理和语音识别 | |
| 自主 | 8006 | 自主决策制定 | |
| 编程 | 8007 | 代码生成和分析 | |
| 规划 | 8008 | 任务规划和调度 | |
| 情感 | 8009 | 情感智能和分析 | |
| 空间 | 8010 | 空间推理和导航 | |
| 计算机视觉 | 8011 | 高级计算机视觉 | |
| 传感器 | 8012 | 传感器数据处理 | |
| 运动 | 8013 | 运动规划和控制 | |
| 预测 | 8014 | 预测分析 | |
| 高级推理 | 8015 | 高级逻辑推理 | |
| 多模型协作 | 8016 | 跨模型协调 | |
| 数据融合 | 8028 | 多源数据融合 | |
| 创造性问题解决 | 8017 | 创造性解决方案生成 | |
| 元认知 | 8018 | 自我意识和反思 | |
| 价值对齐 | 8019 | 伦理对齐 | |
| 视觉图像 | 8020 | 图像特定处理 | |
| 视觉视频 | 8021 | 视频处理 | |
| 金融 | 8022 | 金融分析 | |
| 医疗 | 8023 | 医疗数据分析 | |
| 协作 | 8024 | 人机协作 | |
| 优化 | 8025 | 优化算法 | |
| 计算机 | 8026 | 计算机系统管理 | |
| 数学 | 8027 | 数学计算 | |

### 模型实现细节

所有27个模型都实现了以下核心结构：

```python
class UnifiedXXXModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 神经网络层
        self.layer1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.layer2 = nn.Linear(config.hidden_dim, config.output_dim)
        
    def forward(self, x):
        # 前向传播实现
        x = self.layer1(x)
        x = torch.relu(x)
        return self.layer2(x)
        
    def train_step(self, batch, optimizer):
        # 训练步骤实现
        inputs, targets = batch
        predictions = self(inputs)
        loss = nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
```

每个模型包括：
- **PyTorch nn.Module继承**: 标准神经网络架构
- **Forward方法**: 具有适当维度的张量处理
- **Train step方法**: 完整训练循环集成
- **维度处理**: 跨模型的一致张量形状
- **错误处理**: 全面的错误恢复机制

## 安装与部署

### 先决条件

- **Python**: 3.8+ (推荐: 3.11)
- **Node.js**: 16+ (用于前端开发)
- **Docker**: 20.10+ (用于容器化部署)
- **内存**: 最低8GB，推荐16GB+
- **存储**: 最低10GB可用空间

### Docker快速开始 (推荐)

```bash
# 克隆仓库
git clone https://github.com/Sum-Outman/Self-Soul.git
cd Self-Soul

# 使用Docker Compose启动
docker-compose up -d

# 检查服务状态
docker-compose ps

# 查看日志
docker-compose logs -f backend
```

系统将在以下位置可用：
- **前端仪表板**: http://localhost:5175
- **主API**: http://localhost:8000
- **API文档**: http://localhost:8000/docs

### 手动安装

#### 后端安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 初始化配置
cp .env.example .env
# 编辑.env文件配置您的设置

# 启动后端
python -m core.main
```

#### 前端安装

```bash
cd app
npm install
npm run dev
```

### 生产环境部署

对于生产环境部署，使用提供的Docker配置和环境特定设置：

```bash
# 设置生产环境
export ENVIRONMENT=production

# 设置必需的生产变量
export DB_PASSWORD=your_secure_password
export REDIS_PASSWORD=your_redis_password
export REALTIME_STREAM_API_KEY=your_api_key

# 使用Docker Compose部署
docker-compose -f docker-compose.yml up -d
```

## 配置说明

### 环境变量

在项目根目录创建`.env`文件：

```env
# 环境
ENVIRONMENT=development  # 或 production

# 数据库
DB_TYPE=sqlite  # 或 postgresql
DB_HOST=localhost
DB_PORT=5432
DB_NAME=selfsoul
DB_USER=selfsoul
DB_PASSWORD=  # 生产环境必需

# 安全
REALTIME_STREAM_API_KEY=your_api_key_here
REALTIME_STREAM_AUTH_TYPE=jwt  # 或 api_key
JWT_SECRET_KEY=your_jwt_secret

# 性能
MAX_LOADED_MODELS=8
LAZY_LOAD_ENABLED=true
QUANTIZATION_MODE=none  # none, dynamic, qat
COMPILE_ENABLED=false
```

### 端口配置

| 服务 | 端口 | 描述 |
|------|------|------|
| 主API | 8000 | 主要REST API端点 |
| 实时流管理器 | 8766 | WebSocket流服务 |
| 模型服务 | 8001-8027 | 单个模型端点 |
| 前端 | 5175 | Vue.js仪表板 |

### 模型服务配置

模型服务配置在`config/model_services_config.json`中定义：

```json
{
  "model_ports": {
    "manager": 8001,
    "language": 8002,
    "knowledge": 8003,
    "vision": 8004,
    "audio": 8005,
    // ... 其他模型
  },
  "main_api": {
    "port": 8000,
    "host": "127.0.0.1"
  }
}
```

## 安全特性

### 认证与授权

1. **WebSocket认证**:
   - 开发环境的API密钥认证
   - 生产环境的JWT令牌认证
   - 生产环境要求认证

2. **数据库安全**:
   - 生产环境数据库需要密码
   - 连接加密支持
   - 带加密的自动备份

3. **容器安全**:
   - 非root用户执行
   - 使用特定Linux能力替代特权模式
   - 通过组成员身份进行硬件访问

4. **API安全**:
   - 公共端点的速率限制
   - 输入验证和清理
   - 基于环境的CORS配置

### 生产环境安全检查清单

- [ ] 在.env中设置`ENVIRONMENT=production`
- [ ] 配置强数据库密码
- [ ] 设置WebSocket认证密钥
- [ ] 启用JWT认证
- [ ] 为生产环境配置HTTPS
- [ ] 为暴露端口设置防火墙规则
- [ ] 定期安全更新和补丁

## 性能优化

### 内存管理

系统实现智能内存管理以高效处理27个模型：

```python
# 模型注册表配置
self._max_loaded_models = 8  # 最大同时加载模型数
self.lazy_load_enabled = True  # 按需加载模型
self.quantization_mode = 'none'  # 'none', 'dynamic', 或 'qat'
self.compile_enabled = False  # 启用torch.compile优化
```

### 优化技术

1. **惰性加载**: 模型仅在任务需要时加载
2. **LRU卸载**: 当达到内存限制时卸载最近最少使用的模型
3. **核心模型保护**: 保护基本模型不被卸载
4. **模型量化**: 可选量化以减少模型大小
5. **Torch编译**: JIT编译以加速模型执行
6. **基于任务的加载**: 每个任务仅加载相关模型

### 性能监控

通过API可获取实时性能指标：

```bash
# 检查系统健康
curl http://localhost:8000/api/health

# 获取详细指标
curl http://localhost:8000/api/health/detailed

# 监控模型性能
curl http://localhost:8000/api/metrics/performance
```

## API文档

### 核心端点

| 端点 | 方法 | 描述 | 认证 |
|------|------|------|------|
| `/api/health` | GET | 系统健康检查 | 无 |
| `/api/health/detailed` | GET | 详细健康指标 | 无 |
| `/api/models` | GET | 列出可用模型 | 无 |
| `/api/models/{model_id}` | GET | 获取模型详情 | 无 |
| `/api/tasks` | POST | 提交任务进行处理 | JWT |
| `/api/tasks/{task_id}` | GET | 获取任务状态 | JWT |
| `/api/training/start` | POST | 开始模型训练 | JWT |
| `/api/training/status` | GET | 训练状态 | JWT |
| `/api/hardware/devices` | GET | 列出硬件设备 | JWT |
| `/api/hardware/control` | POST | 控制硬件设备 | JWT |

### WebSocket端点

| 端点 | 描述 | 认证 |
|------|------|------|
| `/ws/streams/{stream_id}` | 实时数据流 | API密钥或JWT |
| `/ws/notifications` | 系统通知 | API密钥或JWT |
| `/ws/hardware/{device_id}` | 硬件设备控制 | JWT |

### API使用示例

```python
import requests

# 健康检查
response = requests.get("http://localhost:8000/api/health")
print(response.json())

# 提交任务
task_data = {
    "task": "分析此图像并描述您看到的内容",
    "modality": "image",
    "data": "base64_encoded_image_data"
}

headers = {"Authorization": "Bearer your_jwt_token"}
response = requests.post(
    "http://localhost:8000/api/tasks",
    json=task_data,
    headers=headers
)
print(response.json())
```

## 端口配置

### 默认端口分配

| 端口 | 服务 | 描述 |
|------|------|------|
| 8000 | 主API | 主要REST API服务 |
| 8766 | 实时流管理器 | WebSocket流服务 |
| 8001 | 管理器模型 | 系统协调模型 |
| 8002 | 语言模型 | 自然语言处理 |
| 8003 | 知识模型 | 知识管理 |
| 8004 | 视觉模型 | 计算机视觉 |
| 8005 | 音频模型 | 音频处理 |
| 8006 | 自主模型 | 自主决策制定 |
| 8007 | 编程模型 | 代码生成和分析 |
| 8008 | 规划模型 | 任务规划和调度 |
| 8009 | 情感模型 | 情感智能 |
| 8010 | 空间模型 | 空间推理 |
| 8011 | 计算机视觉模型 | 高级计算机视觉 |
| 8012 | 传感器模型 | 传感器数据处理 |
| 8013 | 运动模型 | 运动规划和控制 |
| 8014 | 预测模型 | 预测分析 |
| 8015 | 高级推理模型 | 高级逻辑推理 |
| 8016 | 多模型协作模型 | 跨模型协调 |
| 8017 | 创造性问题解决模型 | 创造性解决方案生成 |
| 8018 | 元认知模型 | 自我意识和反思 |
| 8019 | 价值对齐模型 | 伦理对齐 |
| 8020 | 视觉图像模型 | 图像特定处理 |
| 8021 | 视觉视频模型 | 视频处理 |
| 8022 | 金融模型 | 金融分析 |
| 8023 | 医疗模型 | 医疗数据分析 |
| 8024 | 协作模型 | 人机协作 |
| 8025 | 优化模型 | 优化算法 |
| 8026 | 计算机模型 | 计算机系统管理 |
| 8027 | 数学模型 | 数学计算 |
| 8028 | 数据融合模型 | 多源数据融合 |
| 5175 | 前端仪表板 | Vue.js用户界面 |

### 端口自定义

要更改端口分配，修改`config/model_services_config.json`：

```json
{
  "model_ports": {
    "manager": 9001,
    "language": 9002,
    // ... 其他模型
  },
  "main_api": {
    "port": 9000,
    "host": "127.0.0.1"
  }
}
```

## 硬件集成

### 支持的硬件

1. **摄像头**:
   - USB摄像头
   - 以太网摄像头(IP摄像头)
   - CSI摄像头(树莓派)
   - 多摄像头同时支持

2. **传感器**:
   - 温度和湿度传感器
   - 运动传感器(加速度计、陀螺仪)
   - 环境传感器(压力、光线、烟雾)
   - 距离传感器(超声波、红外线)

3. **执行器**:
   - 机器人手臂和机械手
   - 电机控制器
   - LED控制器
   - 继电器模块

4. **通信协议**:
   - 串行(UART)
   - I2C
   - SPI
   - TCP/IP
   - UDP
   - WebSocket

### 硬件配置

硬件配置在`config/model_services_config.json`中定义：

```json
{
  "hardware_config": {
    "camera_settings": {
      "max_cameras": 4,
      "default_resolution": "1280x720",
      "supported_interfaces": ["usb", "ethernet", "csi"]
    },
    "sensor_settings": {
      "supported_sensors": ["temperature", "humidity", "accelerometer"],
      "polling_interval": 1000
    }
  }
}
```

### Docker硬件访问

对于Docker容器中的硬件访问：

```yaml
# docker-compose.yml
services:
  backend:
    devices:
      - /dev/video0:/dev/video0  # 摄像头访问
      - /dev/ttyUSB0:/dev/ttyUSB0  # 串行设备访问
      - /dev/dri:/dev/dri  # GPU加速
    cap_add:
      - SYS_RAWIO  # 原始I/O访问
      - SYS_ADMIN  # 系统管理
      - DAC_OVERRIDE  # 文件权限覆盖
    group_add:
      - dialout  # 串行端口访问
      - video  # 视频设备访问
      - gpio  # GPIO访问(如果可用)
```

## 使用指南

### 启动系统

```bash
# 使用Docker Compose(推荐)
docker-compose up -d

# 直接使用Python
python -m core.main

# 使用开发脚本
python scripts/start-dev.py
```

### 访问仪表板

1. 在浏览器中打开`http://localhost:5175`
2. 使用默认凭据登录(如果启用认证)
3. 导航到仪表板部分：
   - **系统概览**: 整体系统状态
   - **模型管理**: 查看和控制AI模型
   - **训练界面**: 启动和监控训练会话
   - **硬件控制**: 管理连接的硬件设备
   - **API文档**: 交互式API文档

### 基本任务

1. **文本分析**:
   ```bash
   curl -X POST http://localhost:8000/api/tasks \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"task": "分析此文本", "modality": "text", "data": "要分析的示例文本"}'
   ```

2. **图像处理**:
   ```bash
   # 首先将图像转换为base64
   base64_image=$(base64 -i image.jpg)
   
   curl -X POST http://localhost:8000/api/tasks \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d "{\"task\": \"描述此图像\", \"modality\": \"image\", \"data\": \"$base64_image\"}"
   ```

3. **训练模型**:
   ```bash
   curl -X POST http://localhost:8000/api/training/start \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"model_ids": ["language", "vision"], "epochs": 10, "batch_size": 32}'
   ```

### 监控与维护

1. **检查系统健康**:
   ```bash
   curl http://localhost:8000/api/health
   ```

2. **查看日志**:
   ```bash
   # Docker日志
   docker-compose logs -f backend
   
   # 文件日志
   tail -f logs/system.log
   ```

3. **性能监控**:
   ```bash
   curl http://localhost:8000/api/metrics/performance
   ```

## 故障排除

### 常见问题

#### 问题: "ImportError" 或 "NameError"
**解决方案**: 运行导入验证脚本：
```bash
python check_imports.py
python test_models_import.py
```

#### 问题: Docker权限错误
**解决方案**: 确保Docker有适当的权限：
```bash
# 将用户添加到docker组
sudo usermod -aG docker $USER

# 重启Docker服务
sudo systemctl restart docker
```

#### 问题: WebSocket连接失败
**解决方案**: 检查认证配置：
```bash
# 验证环境变量
echo $REALTIME_STREAM_API_KEY
echo $ENVIRONMENT

# 测试WebSocket连接
python -c "import websocket; ws = websocket.WebSocket(); ws.connect('ws://localhost:8766/ws/streams/test?api_key=YOUR_KEY')"
```

#### 问题: 内存耗尽(OOM)
**解决方案**: 调整内存管理设置：
```python
# 在model_registry.py中或通过环境
export MAX_LOADED_MODELS=5  # 从8减少到5
export LAZY_LOAD_ENABLED=true
```

### 性能优化提示

1. **减少加载的模型**: 对于RAM有限的系统，设置`MAX_LOADED_MODELS=5`
2. **启用量化**: 设置`QUANTIZATION_MODE=dynamic`以减少模型大小
3. **使用GPU加速**: 确保CUDA可用并配置
4. **监控资源使用**: 使用性能监控仪表板
5. **优化批处理大小**: 根据可用内存调整

### 调试技术

1. **启用调试日志**:
   ```bash
   export LOG_LEVEL=DEBUG
   docker-compose restart backend
   ```

2. **检查服务状态**:
   ```bash
   docker-compose ps
   docker-compose logs --tail=100 backend
   ```

3. **测试单个模型**:
   ```bash
   curl http://localhost:8001/health  # 管理器模型
   curl http://localhost:8002/health  # 语言模型
   ```

## 贡献指南

我们欢迎对Self Soul的贡献！请参阅我们的贡献指南：

1. **Fork仓库**
2. **创建功能分支**: `git checkout -b feature/amazing-feature`
3. **提交更改**: `git commit -m '添加惊人的功能'`
4. **推送到分支**: `git push origin feature/amazing-feature`
5. **打开Pull Request**

### 开发指南

- 遵循现有的代码风格和约定
- 为新功能添加测试
- 为更改更新文档
- 确保向后兼容性
- 遵循安全最佳实践

### 测试

```bash
# 运行导入测试
python check_imports.py

# 运行模型导入测试
python test_models_import.py

# 运行安全测试
python test_deployment_fixes.py

# 前端类型检查
cd app
npm run type-check
```

## 许可证

本项目采用Apache License 2.0许可证 - 详见[LICENSE](LICENSE)文件。

## 致谢

- **开源社区**: 为让本项目成为可能的令人惊叹的工具和库
- **AI研究社区**: 为推进人工智能领域
- **贡献者**: 每个贡献代码、问题或想法的人
- **用户**: 为测试、反馈和实际部署

### 特别感谢

1. **PyTorch团队**: 为优秀的深度学习框架
2. **FastAPI团队**: 为高性能Web框架
3. **Vue.js团队**: 为渐进式JavaScript框架
4. **Docker团队**: 为容器化技术
5. **所有外部API提供商**: 为使AI可访问

---

**Self Soul团队** - 构建通用人工智能的未来

*如有问题、问题或贡献，请在GitHub上创建issue或联系silencecrowtom@qq.com*