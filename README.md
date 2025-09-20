# Self Soul

**Developed by**: Self Soul Team
**Developer Email**: silencecrowtom@qq.com

## Project Overview

Self Soul is a comprehensive Artificial General Intelligence (AGI) system designed to integrate multiple advanced AI capabilities into a unified architecture. This system combines language processing, visual recognition, audio analysis, emotional understanding, and autonomous learning into a cohesive platform.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

### Core Features

- **Multi-Modal Integration**: Seamlessly combines text, image, audio, and video processing
- **Autonomous Learning**: Continuously improves through self-directed learning and adaptation
- **Advanced Language Processing**: Supports 5 languages with context-aware understanding
- **Real-time Monitoring**: Tracks system performance and model metrics
- **Interactive Dashboard**: Provides intuitive interface for system management
- **Extensible Architecture**: Easy to add new models and capabilities
- **Knowledge Management**: Organizes and utilizes structured and unstructured knowledge

## System Architecture

Self Soul employs a layered architecture that separates core AI capabilities from the user interface, enabling modular development and deployment.

### Architecture Diagram

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

Self Soul includes 15+ specialized AI models working together to provide comprehensive intelligence capabilities:

### Foundational Models

- **Manager Model**: Orchestrates other models and manages system resources
- **Language Model**: Processes natural language and generates human-like responses
- **Knowledge Model**: Organizes and retrieves structured and unstructured knowledge
- **Vision Model**: Analyzes images and video content
- **Audio Model**: Processes and understands sound and speech

### Specialized Models

- **Autonomous Model**: Manages self-directed learning and adaptation
- **Programming Model**: Generates and optimizes code
- **Planning Model**: Creates and executes complex plans
- **Emotion Model**: Recognizes and responds to emotional cues
- **Spatial Model**: Processes spatial relationships and navigation
- **Computer Vision Model**: Advanced visual understanding
- **Sensor Model**: Processes data from various sensors
- **Motion Model**: Analyzes and predicts movement patterns
- **Prediction Model**: Makes data-driven forecasts and predictions

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

| Service Name | English Name | Port | Description |
|-------------|-------------|------|------------|
| Service | English Name | Port | Description |
| ------- | ------------ | ---- | ----------- |
| Main API Gateway | Main API Gateway | 8000 | Main entry point of the system, providing RESTful API interface |
| Frontend Application | Frontend Application | 5175 | User interface, accessible through browser |
| Realtime Stream Manager | Realtime Stream Manager | 8765 | Manages real-time data streams and inter-model communication |
| Performance Monitoring | Performance Monitoring | 8081 | Monitors system performance and resource usage |

### Model Port Configuration

The system assigns independent ports to each AI model, ranging from 8001 to 8019:

| Port Number | Model Type (Chinese) | English Model Type |
|------------|---------------------|------------------|
| Port | Model Type | English Model Type |
| ---- | --------- | ------------------ |
| 8001 | Manager Model | Manager Model |
| 8002 | Language Model | Language Model |
| 8003 | Knowledge Model | Knowledge Model |
| 8004 | Vision Model | Vision Model |
| 8005 | Audio Model | Audio Model |
| 8006 | Autonomous Model | Autonomous Model |
| 8007 | Programming Model | Programming Model |
| 8008 | Planning Model | Planning Model |
| 8009 | Emotion Model | Emotion Model |
| 8010 | Spatial Model | Spatial Model |
| 8011 | Computer Vision Model | Computer Vision Model |
| 8012 | Sensor Model | Sensor Model |
| 8013 | Motion Model | Motion Model |
| 8014 | Prediction Model | Prediction Model |
| 8015 | Advanced Reasoning Model | Advanced Reasoning Model |
| 8016 | Data Fusion Model | Data Fusion Model |
| 8017 | Creative Problem Solving Model | Creative Problem Solving Model |
| 8018 | Meta Cognition Model | Meta Cognition Model |
| 8019 | Value Alignment Model | Value Alignment Model |

Port configuration is stored in the `config/model_services_config.json` file, which is automatically loaded when the system starts.

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

