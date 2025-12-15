# Self Soul - Advanced AGI System

**Developed by**: Self Soul Team  
**Developer Email**: silencecrowtom@qq.com  
**Repository**: https://github.com/Sum-Outman/Self-Soul

## Project Overview

Self Soul is a sophisticated Artificial General Intelligence (AGI) platform that integrates 19 specialized AI models into a unified cognitive architecture. This open-source system provides comprehensive multi-modal intelligence capabilities including natural language processing, computer vision, audio analysis, emotional intelligence, autonomous learning, and advanced reasoning. The system is designed to support true training from scratch, multi-camera vision capabilities, external device integration, and seamless switching between local and external API models.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Stars](https://img.shields.io/github/stars/Sum-Outman/Self-Soul)](https://github.com/Sum-Outman/Self-Soul/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/Sum-Outman/Self-Soul)](https://github.com/Sum-Outman/Self-Soul/issues)

### Core Features

- **Multi-Modal Integration**: Seamlessly combines text, image, audio, and video processing with unified cognitive architecture
- **Autonomous Learning**: Self-directed learning and continuous adaptation through meta-cognition
- **Advanced Language Processing**: Supports 5 languages with deep contextual understanding and reasoning
- **Real-time Monitoring**: Comprehensive system performance tracking and model metrics visualization
- **Interactive Dashboard**: Modern Vue.js-based interface for intuitive system management
- **Extensible Architecture**: Modular design allowing easy integration of new AI capabilities
- **Knowledge Management**: Advanced knowledge base with structured and unstructured data integration
- **Emotional Intelligence**: Emotion recognition and response capabilities for human-like interactions
- **Advanced Reasoning**: Complex logical reasoning and creative problem-solving abilities
- **Value Alignment**: Ethical guidelines integration to ensure responsible AI behavior

## System Architecture

Self Soul employs a layered architecture that separates core AI capabilities from the user interface, enabling modular development and deployment.

### Architecture Diagram

```
Self Soul /
├── core/                     # Core backend system
│   ├── main.py               # Backend entry point and API endpoints
│   ├── model_service_manager.py # Model service creation and management
│   ├── model_registry.py     # Model registration and lifecycle management
│   ├── training_manager.py   # Training coordination and management
│   ├── autonomous_learning_manager.py # Self-learning capabilities
│   ├── joint_training_coordinator.py  # Multi-model training coordination
│   └── error_handling.py     # Error handling and logging system
├── app/                      # Frontend application
│   ├── src/                  # Frontend source code
│   │   ├── views/            # Vue components for different views
│   │   ├── components/       # Reusable UI components
│   │   └── assets/           # Static assets
│   ├── public/               # Publicly accessible files
│   └── package.json          # Frontend dependencies
├── config/                   # System configuration files
│   └── model_services_config.json # Model service port configuration
├── data/                     # Data storage for models and training
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Core Models

Self Soul includes 19 specialized AI models working together to provide comprehensive intelligence capabilities:

### Foundational Models

- **Manager Model**: Orchestrates other models and manages system resources
- **Language Model**: Processes natural language and generates human-like responses
- **Knowledge Model**: Organizes and retrieves structured and unstructured knowledge
- **Vision Model**: Analyzes images and video content
- **Audio Model**: Processes and understands sound and speech
- **Autonomous Model**: Manages self-directed learning and adaptation

### Advanced Models

- **Programming Model**: Generates and optimizes code
- **Planning Model**: Creates and executes complex plans
- **Emotion Model**: Recognizes and responds to emotional cues
- **Spatial Model**: Processes spatial relationships and navigation
- **Computer Vision Model**: Advanced visual understanding
- **Sensor Model**: Processes data from various sensors
- **Motion Model**: Analyzes and predicts movement patterns
- **Prediction Model**: Makes data-driven forecasts and predictions
- **Advanced Reasoning Model**: Performs complex logical reasoning and problem-solving tasks
- **Data Fusion Model**: Integrates information from multiple sources for comprehensive understanding
- **Creative Problem Solving Model**: Develops innovative solutions to complex challenges
- **Meta Cognition Model**: Monitors and optimizes the system's own cognitive processes
- **Value Alignment Model**: Ensures system behaviors align with defined ethical guidelines and values

## Installation Guide

### System Requirements

- Windows 10/11, Linux, or macOS
- Python 3.9+ (recommended: 3.11)
- Node.js 16+ (recommended: 18)
- 8GB RAM (minimum), 16GB+ recommended
- 10GB free disk space

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sum-Outman/Self-Soul
   cd Self-Soul
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
   Open your browser and visit: http://localhost:5175

## Usage Guide

### Main Features

1. **Dashboard**: Monitor system performance and active models
2. **Model Management**: Configure and control individual AI models
3. **Knowledge Management**: Import, browse, and manage knowledge base
4. **Training**: Train and fine-tune models with custom datasets
5. **Settings**: Configure system preferences and language settings
6. **Help Center**: Access documentation and usage guides

### Multi-language Support

Self Soul supports 5 languages out of the box:
- English
- Chinese (Simplified)
- German
- Japanese
- Russian

You can switch languages from the settings page or using the language switcher in the header.

## Port Configuration

Self Soul system uses the following port configuration for running various services and models:

### Main Service Ports

| 主API网关 | Main API Gateway | 8000 | 系统的主要入口点，提供RESTful API接口 |
|---------|-----------------|------|----------------------------------|
| 前端应用 | Frontend Application | 5175 | 用户界面，可通过浏览器访问 |
| 实时数据流管理器 | Realtime Stream Manager | 8766 | 管理实时数据流和模型间通信 |
| 性能监控服务 | Performance Monitoring | 8080 | 监控系统性能和资源使用情况 |

### Model Port Configuration

系统为每个AI模型分配了独立的端口，范围从8001到8019：

| 端口号 | 模型类型 | 英文模型类型 |
|-------|---------|------------|
| 8001 | 管理模型 | Manager Model |
| 8002 | 语言模型 | Language Model |
| 8003 | 知识模型 | Knowledge Model |
| 8004 | 视觉模型 | Vision Model |
| 8005 | 音频模型 | Audio Model |
| 8006 | 自主模型 | Autonomous Model |
| 8007 | 编程模型 | Programming Model |
| 8008 | 规划模型 | Planning Model |
| 8009 | 情感模型 | Emotion Model |
| 8010 | 空间模型 | Spatial Model |
| 8011 | 计算机视觉模型 | Computer Vision Model |
| 8012 | 传感器模型 | Sensor Model |
| 8013 | 运动模型 | Motion Model |
| 8014 | 预测模型 | Prediction Model |
| 8015 | 高级推理模型 | Advanced Reasoning Model |
| 8016 | 数据融合模型 | Data Fusion Model |
| 8017 | 创造性问题解决模型 | Creative Problem Solving Model |
| 8018 | 元认知模型 | Meta Cognition Model |
| 8019 | 值对齐模型 | Value Alignment Model |

端口配置存储在`config/model_services_config.json`文件中，系统启动时会自动加载这些配置。

## API Documentation

The system provides a RESTful API for integration with other applications. API documentation is available at:

http://localhost:8000/docs (when backend is running)

## Common Issues and Solutions

### Virtual Environment Problems

If you encounter issues with the virtual environment:
```bash
python -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
python -m venv .venv
```

### Dependency Installation Issues

If pip installation fails:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Frontend Startup Problems

If npm run dev fails:
```bash
cd app
rm -rf node_modules
npm install
npm run dev
```

## Contributing

We welcome contributions to the Self Soul project. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Self Soul is built with contributions from the open-source community and leverages various AI research and technologies.

---

© 2025 Self Soul Team. All rights reserved.
