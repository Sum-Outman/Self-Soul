# Self Soul
# Self Soul - 通用人工智能大脑系统

**Developed by**: Self Soul Team
**开发者邮箱**: silencecrowtom@qq.com

## Project Overview
## 项目概述

Self Soul is a comprehensive Artificial General Intelligence (AGI) system designed to integrate multiple advanced AI capabilities into a unified architecture. This system combines language processing, visual recognition, audio analysis, emotional understanding, and autonomous learning into a cohesive platform.

Self Soul是一个全面的通用人工智能(AGI)系统，旨在将多种先进的AI能力整合到一个统一的架构中。该系统将语言处理、视觉识别、音频分析、情感理解和自主学习等能力整合到一个协调的平台中。

### Key Features
### 核心功能

- **Multi-Modal Integration**: Seamlessly combines text, image, audio, and video processing
- **Autonomous Learning**: Continuously improves through self-directed learning and adaptation
- **Advanced Language Processing**: Supports 5 languages with context-aware understanding
- **Real-time Monitoring**: Tracks system performance and model metrics
- **Interactive Dashboard**: Provides intuitive interface for system management
- **Extensible Architecture**: Easy to add new models and capabilities
- **Knowledge Management**: Organizes and utilizes structured and unstructured knowledge

- **多模态整合**：无缝结合文本、图像、音频和视频处理
- **自主学习**：通过自我导向的学习和适应不断改进
- **先进语言处理**：支持5种语言，具有上下文感知理解能力
- **实时监控**：跟踪系统性能和模型指标
- **交互式仪表盘**：提供直观的系统管理界面
- **可扩展架构**：易于添加新模型和功能
- **知识管理**：组织和利用结构化和非结构化知识

## System Architecture
## 系统架构

Self Soul employs a layered architecture that separates core AI capabilities from the user interface, enabling modular development and deployment.

Self Soul采用分层架构，将核心AI能力与用户界面分离，实现模块化开发和部署。

### Architecture Diagram
### 架构图

```
Self Soul /
├── core/                     # Core backend system
│   ├── models/               # 15+ specialized AI models
│   ├── training/             # Model training infrastructure
│   ├── knowledge/            # Knowledge base and management
│   └── main.py               # Backend entry point
├── app/                      # Frontend application
│   ├── src/                  # Vue.js source code
│   │   ├── views/            # Main application views
│   │   ├── components/       # Reusable UI components
│   │   └── locales/          # Multi-language support
│   └── package.json          # Frontend dependencies
├── data/                     # Training and knowledge data
└── config/                   # System configuration files
```

## Core Models
## 核心模型

Self Soul includes 15+ specialized AI models working together to provide comprehensive intelligence capabilities:

Self Soul包含15+个专业AI模型，它们协同工作以提供全面的智能功能：

### Foundational Models
### 基础模型

- **Manager Model**: Orchestrates other models and manages system resources
- **Language Model**: Processes natural language and generates human-like responses
- **Knowledge Model**: Organizes and retrieves structured and unstructured knowledge
- **Vision Model**: Analyzes images and video content
- **Audio Model**: Processes and understands sound and speech

- **管理模型**：协调其他模型并管理系统资源
- **语言模型**：处理自然语言并生成类人响应
- **知识模型**：组织和检索结构化和非结构化知识
- **视觉模型**：分析图像和视频内容
- **音频模型**：处理和理解声音和语音

### Specialized Models
### 专业模型

- **Autonomous Model**: Manages self-directed learning and adaptation
- **Programming Model**: Generates and optimizes code
- **Planning Model**: Creates and executes complex plans
- **Emotion Model**: Recognizes and responds to emotional cues
- **Spatial Model**: Processes spatial relationships and navigation
- **Computer Vision Model**: Advanced visual understanding
- **Sensor Model**: Processes data from various sensors
- **Motion Model**: Analyzes and predicts movement patterns
- **Prediction Model**: Makes data-driven forecasts and predictions

- **自主模型**：管理自我导向的学习和适应
- **编程模型**：生成和优化代码
- **规划模型**：创建和执行复杂计划
- **情感模型**：识别和响应情感线索
- **空间模型**：处理空间关系和导航
- **计算机视觉模型**：高级视觉理解
- **传感器模型**：处理来自各种传感器的数据
- **运动模型**：分析和预测运动模式
- **预测模型**：进行数据驱动的预测

## Installation Guide
## 安装指南

### System Requirements
### 系统要求

- Windows 10/11, Linux, or macOS
- Python 3.9+ (recommended: 3.11)
- Node.js 16+ (recommended: 18)
- 8GB RAM (minimum), 16GB+ recommended
- 10GB free disk space

- Windows 10/11、Linux或macOS
- Python 3.9+（推荐：3.11）
- Node.js 16+（推荐：18）
- 8GB内存（最低），推荐16GB以上
- 10GB可用磁盘空间

### Installation Steps
### 安装步骤

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/self-soul.git
   cd self-soul
   ```

2. **Set up Python virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - Windows:
   ```bash
   .venv\Scripts\activate
   ```
   - Linux/macOS:
   ```bash
   source .venv/bin/activate
   ```

4. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install frontend dependencies**
   ```bash
   cd app
   npm install
   ```

6. **Configure environment variables**
   Create a `.env` file and add the following:
   ```
   API_KEY=your-api-key-here
   MODEL_PATH=./models
   DATA_PATH=./data
   ```

7. **Start the system**
   ```bash
   # Start backend service
   python core/main.py
   
   # In a new terminal, start frontend application
   cd app && npm run dev
   ```

8. **Access the system**
   Open your browser and visit: http://localhost:5177

1. **克隆代码库**
   ```bash
   git clone https://github.com/your-username/self-soul.git
   cd self-soul
   ```

2. **设置Python虚拟环境**
   ```bash
   python -m venv .venv
   ```

3. **激活虚拟环境**
   - Windows:
   ```bash
   .venv\Scripts\activate
   ```
   - Linux/macOS:
   ```bash
   source .venv/bin/activate
   ```

4. **安装Python依赖**
   ```bash
   pip install -r requirements.txt
   ```

5. **安装前端依赖**
   ```bash
   cd app
   npm install
   ```

6. **配置环境变量**
   创建一个`.env`文件并添加以下内容：
   ```
   API_KEY=your-api-key-here
   MODEL_PATH=./models
   DATA_PATH=./data
   ```

7. **启动系统**
   ```bash
   # 启动后端服务
   python core/main.py
   
   # 在新终端中启动前端应用
   cd app && npm run dev
   ```

8. **访问系统**
   打开浏览器并访问：http://localhost:5177

## Usage Guide
## 使用指南

### Main Features
### 主要功能

1. **Dashboard**: Monitor system performance and active models
2. **Model Management**: Configure and control individual AI models
3. **Knowledge Management**: Import, browse, and manage knowledge base
4. **Training**: Train and fine-tune models with custom datasets
5. **Settings**: Configure system preferences and language settings
6. **Help Center**: Access documentation and usage guides

1. **仪表盘**：监控系统性能和活跃模型
2. **模型管理**：配置和控制各个AI模型
3. **知识管理**：导入、浏览和管理知识库
4. **训练**：使用自定义数据集训练和微调模型
5. **设置**：配置系统偏好和语言设置
6. **帮助中心**：访问文档和使用指南

### Multi-language Support
### 多语言支持

Self Soul supports 5 languages out of the box:
- English
- Chinese (Simplified)
- German
- Japanese
- Russian

You can switch languages from the settings page or using the language switcher in the header.

Self Soul默认支持5种语言：
- 英语
- 中文（简体）
- 德语
- 日语
- 俄语

您可以从设置页面或使用头部的语言切换器切换语言。

## API Documentation
## API文档

The system provides a RESTful API for integration with other applications. API documentation is available at:

系统提供RESTful API，用于与其他应用程序集成。API文档可在以下位置获取：

http://localhost:8000/docs (when backend is running)

## Common Issues and Solutions
## 常见问题和解决方案

### Virtual Environment Problems
### 虚拟环境问题

If you encounter issues with the virtual environment:
```bash
python -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
python -m venv .venv
```

### Dependency Installation Issues
### 依赖安装问题

If pip installation fails:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Frontend Startup Problems
### 前端启动问题

If npm run dev fails:
```bash
cd app
rm -rf node_modules
npm install
npm run dev
```

## Contributing
## 贡献指南

We welcome contributions to the Self Soul project. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

我们欢迎对Self Soul项目的贡献。请按照以下步骤进行：

1. Fork代码库
2. 创建功能分支
3. 提交您的更改
4. 推送到分支
5. 创建拉取请求

请确保您的代码符合项目的编码标准，并包含适当的测试。

## License
## 许可证

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

本项目采用Apache License 2.0许可证。详情请参阅[LICENSE](LICENSE)文件。

## Acknowledgements
## 鸣谢

Self Soul is built with contributions from the open-source community and leverages various AI research and technologies.

Self Soul是由开源社区贡献构建的，并利用了各种AI研究和技术。

---

© 2025 Self Soul Team. All rights reserved.
© 2025 Self Soul团队。保留所有权利。