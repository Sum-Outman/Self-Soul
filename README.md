# Self Soul - Advanced AGI System

**Developed by**: Self Soul Team  
**Developer Email**: silencecrowtom@qq.com  
**Repository**: https://github.com/Sum-Outman/Self-Soul

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Stars](https://img.shields.io/github/stars/Sum-Outman/Self-Soul)](https://github.com/Sum-Outman/Self-Soul/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/Sum-Outman/Self-Soul)](https://github.com/Sum-Outman/Self-Soul/issues)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Vue.js Version](https://img.shields.io/badge/Vue.js-3.4%2B-green.svg)](https://vuejs.org/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.4%2B-orange.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-20.10%2B-blue.svg)](https://www.docker.com/)

## Table of Contents

- [Latest Updates](#latest-updates)
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Model Architecture](#model-architecture)
- [Installation and Deployment](#installation-and-deployment)
- [Configuration](#configuration)
- [Security Features](#security-features)
- [Performance Optimization](#performance-optimization)
- [API Documentation](#api-documentation)
- [Port Configuration](#port-configuration)
- [Hardware Integration](#hardware-integration)
- [Usage Guide](#usage-guide)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Latest Updates (March 2026)

Self Soul system has undergone significant improvements and enhancements:

### 🚀 **Major Performance Optimizations**
- **Intelligent Memory Management**: Dynamic model loading with LRU unloading strategy
- **8-Model Limit**: Maximum of 8 models loaded concurrently to prevent OOM
- **Core Model Protection**: Priority protection for manager, language, and knowledge models
- **Access Time Tracking**: All models track last access time for optimal unloading decisions
- **Torch Compile Support**: Model compilation optimization for faster inference

### 🔒 **Enhanced Security Features**
- **WebSocket Authentication**: API key and JWT token authentication for WebSocket connections
- **Production Security Enforcement**: Database passwords required in production environment
- **Non-Root Container**: Docker containers run as non-root user for improved security
- **Granular Docker Permissions**: Replaced privileged mode with specific capabilities (SYS_RAWIO, SYS_ADMIN, etc.)
- **Redis Security**: Password validation for Redis connections in production

### 🎯 **Deployment and Hardware Improvements**
- **Python 3.11 Upgrade**: Docker images updated to Python 3.11 for better compatibility
- **Hardware Access Groups**: Container users added to dialout, video, and gpio groups
- **Modern Frontend**: Vue.js dependencies updated (vue-i18n v11.0.0)
- **Type Safety**: Full TypeScript migration with comprehensive type checking

### 🧠 **AGI Core Capability Completion**
- **Complete Training Loop**: Integrated DataLoader and full epoch-based training cycles
- **Joint Model Training**: Capability to train all 27 models collaboratively
- **Self-Evolution Architecture**: Neural Architecture Search (NAS) with layer mutation
- **Dimension Consistency**: Unified tensor dimensions across all neural networks
- **Cross-Modal Fusion**: ResNet and LSTM encoders for image and audio processing

### 📊 **Code Quality and Maintenance**
- **Experimental Scripts**: Organized into scripts/ directory with standardized CLI interfaces
- **Import Error Fixes**: Comprehensive import validation and error handling
- **Documentation Synchronization**: Updated English and Chinese documentation

## Project Overview

Self Soul is a sophisticated Artificial General Intelligence (AGI) platform that integrates **27 specialized AI models** into a unified cognitive architecture, supported by **18 external API providers** for flexible deployment options. This open-source system provides comprehensive multi-modal intelligence capabilities including natural language processing, computer vision, audio analysis, emotional intelligence, autonomous learning, and advanced reasoning.

### Design Philosophy

Self Soul is built on the core principle that true AGI requires a cohesive, integrated architecture rather than isolated models. Our design philosophy emphasizes:

- **Unified Cognitive Architecture**: All 27 models work synergistically through a central coordination system, enabling emergent intelligence greater than the sum of individual components
- **From-Scratch Training**: By training all models from scratch without pre-trained foundations, we maintain full control over model development, ethical alignment, and AGI compliance
- **Human-Centered Intelligence**: Integrating emotional intelligence and value alignment to ensure AI behavior is responsible, ethical, and aligned with human values
- **Modular Extensibility**: A flexible architecture that allows easy integration of new AI capabilities while maintaining system coherence

### Technical Highlights

- **27 Specialized AI Models**: Comprehensive coverage of cognitive capabilities from basic perception to advanced reasoning
- **Advanced Multi-Modal Integration**: Seamless processing and fusion of text, image, audio, and video data
- **Adaptive Learning Engine**: Real-time optimization of learning strategies and training parameters based on performance metrics
- **Distributed Processing**: Each model runs on a dedicated port (8001-8027) enabling parallel processing and scalability
- **Modern UI/UX**: Intuitive Vue.js-based dashboard for system management and monitoring
- **Comprehensive API**: RESTful interface for integration with external systems and applications
- **Multimodal Dataset Support**: Expanded Multimodal Dataset v1 supports all 27 models for comprehensive training
- **Comprehensive External API Support**: Integration with 18 external API providers including OpenAI, Anthropic, Google AI, AWS, Azure, HuggingFace, and domestic providers like DeepSeek, Zhipu AI, and Baidu ERNIE

## Key Features

### 🧠 **Intelligent Memory Management**
- **Dynamic Model Loading**: Models are loaded on-demand based on task requirements
- **LRU Unloading Strategy**: Least Recently Used models are unloaded when memory limits are reached
- **Core Model Protection**: Essential models (manager, language, knowledge) receive priority protection
- **8-Model Limit**: Maximum of 8 models loaded concurrently to prevent memory exhaustion
- **Access Time Tracking**: Automatic tracking of model usage for optimal unloading decisions

### 🔐 **Advanced Security**
- **WebSocket Authentication**: Dual authentication (API key + JWT tokens) for real-time streams
- **Production Security**: Mandatory password requirements for databases in production
- **Non-Root Execution**: Docker containers run as non-root user with minimal privileges
- **Granular Capabilities**: Specific Linux capabilities instead of privileged mode
- **Environment-Based Security**: Different security configurations for development and production

### 🚀 **Performance Optimization**
- **Lazy Loading**: Models are loaded only when needed, reducing initial memory footprint
- **Model Quantization**: Support for dynamic quantization to reduce model size
- **Torch Compile**: JIT compilation optimization for faster model execution
- **Task-Based Loading**: Intelligent model selection based on task requirements
- **Resource Monitoring**: Real-time monitoring of CPU, memory, and GPU usage

### 🎯 **AGI Core Capabilities**
- **Complete Training System**: Full training loop with DataLoader, epochs, and validation
- **Joint Training**: All 27 models can be trained collaboratively
- **Neural Architecture Search**: Automatic optimization of model architectures
- **Cross-Modal Fusion**: Integration of vision, audio, and text processing
- **Self-Evolution**: Models can evolve their own architectures through mutation operations

### 🌐 **Hardware Integration**
- **Multi-Camera Support**: Simultaneous processing from multiple camera inputs
- **Sensor Integration**: Support for various environmental and motion sensors
- **Robotic Control**: APIs for controlling robotic systems and actuators
- **Real-time Communication**: WebSocket-based real-time data streaming
- **Protocol Support**: Serial, TCP/IP, UDP, and camera interfaces

## System Architecture

### Core Components

```
Self Soul AGI System
├── Central Coordinator (Port 8000)
│   ├── Model Registry (Dynamic Model Management)
│   ├── Task Scheduler (Intelligent Task Distribution)
│   ├── Security Manager (Authentication & Authorization)
│   └── Performance Monitor (Real-time Metrics)
├── 27 Specialized Models (Ports 8001-8027)
│   ├── Manager Model (Port 8001) - System coordination
│   ├── Language Model (Port 8002) - Natural language processing
│   ├── Knowledge Model (Port 8003) - Knowledge management
│   ├── Vision Model (Port 8004) - Computer vision
│   ├── Audio Model (Port 8005) - Audio processing
│   └── ... (23 additional specialized models)
├── Frontend Dashboard (Port 5175)
│   ├── System Monitoring
│   ├── Model Management
│   ├── Training Interface
│   └── Hardware Control
└── External Services
    ├── PostgreSQL Database (Optional)
    ├── Redis Cache (Optional)
    └── External API Providers (18 providers)
```

### Model Coordination Architecture

The system employs a sophisticated model coordination mechanism:

1. **Task Analysis**: Input tasks are analyzed to determine required capabilities
2. **Model Selection**: Based on task requirements, 5-8 relevant models are selected
3. **Dynamic Loading**: Selected models are loaded into memory (if not already loaded)
4. **Collaborative Processing**: Models work together to process the task
5. **Result Fusion**: Individual model outputs are fused into a coherent response
6. **Memory Management**: Unused models are unloaded based on LRU strategy

### Memory Management Architecture

The intelligent memory management system ensures optimal resource utilization:

```
Memory Management Flow:
1. Task Received → Identify Required Models
2. Check Loaded Models → Load Missing Models
3. If >8 models loaded → Unload LRU non-core models
4. Process Task → Update Model Access Times
5. Monitor Memory Usage → Trigger Cleanup if Needed
6. Regular Maintenance → Clean expired workflows & records
```

## Model Architecture

### Complete Model List (27 Models)

| Model Name | Port | Description | Core Model |
|------------|------|-------------|------------|
| Manager | 8001 | System coordination and task distribution | ✅ |
| Language | 8002 | Natural language processing and generation | ✅ |
| Knowledge | 8003 | Knowledge management and retrieval | ✅ |
| Vision | 8004 | Computer vision and image processing | |
| Audio | 8005 | Audio processing and speech recognition | |
| Autonomous | 8006 | Autonomous decision making | |
| Programming | 8007 | Code generation and analysis | |
| Planning | 8008 | Task planning and scheduling | |
| Emotion | 8009 | Emotional intelligence and analysis | |
| Spatial | 8010 | Spatial reasoning and navigation | |
| Computer Vision | 8011 | Advanced computer vision | |
| Sensor | 8012 | Sensor data processing | |
| Motion | 8013 | Motion planning and control | |
| Prediction | 8014 | Predictive analytics | |
| Advanced Reasoning | 8015 | Advanced logical reasoning | |
| Multi-Model Collaboration | 8016 | Cross-model coordination | |
| Data Fusion | 8028 | Multi-source data fusion | |
| Creative Problem Solving | 8017 | Creative solution generation | |
| Meta Cognition | 8018 | Self-awareness and reflection | |
| Value Alignment | 8019 | Ethical alignment | |
| Vision Image | 8020 | Image-specific processing | |
| Vision Video | 8021 | Video processing | |
| Finance | 8022 | Financial analysis | |
| Medical | 8023 | Medical data analysis | |
| Collaboration | 8024 | Human-AI collaboration | |
| Optimization | 8025 | Optimization algorithms | |
| Computer | 8026 | Computer system management | |
| Mathematics | 8027 | Mathematical computation | |

### Model Implementation Details

All 27 models implement the following core structure:

```python
class UnifiedXXXModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Neural network layers
        self.layer1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.layer2 = nn.Linear(config.hidden_dim, config.output_dim)
        
    def forward(self, x):
        # Forward pass implementation
        x = self.layer1(x)
        x = torch.relu(x)
        return self.layer2(x)
        
    def train_step(self, batch, optimizer):
        # Training step implementation
        inputs, targets = batch
        predictions = self(inputs)
        loss = nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
```

Each model includes:
- **PyTorch nn.Module inheritance**: Standard neural network architecture
- **Forward method**: Tensor processing with appropriate dimensions
- **Train step method**: Complete training loop integration
- **Dimension handling**: Consistent tensor shapes across models
- **Error handling**: Comprehensive error recovery mechanisms

## Installation and Deployment

### Prerequisites

- **Python**: 3.8+ (Recommended: 3.11)
- **Node.js**: 16+ (for frontend development)
- **Docker**: 20.10+ (for containerized deployment)
- **Memory**: Minimum 8GB, Recommended 16GB+
- **Storage**: Minimum 10GB free space

### Quick Start with Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/Sum-Outman/Self-Soul.git
cd Self-Soul

# Start with Docker Compose
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f backend
```

The system will be available at:
- **Frontend Dashboard**: http://localhost:5175
- **Main API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Manual Installation

#### Backend Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize configuration
cp .env.example .env
# Edit .env with your configuration

# Start the backend
python -m core.main
```

#### Frontend Installation

```bash
cd app
npm install
npm run dev
```

### Production Deployment

For production deployment, use the provided Docker configuration with environment-specific settings:

```bash
# Set production environment
export ENVIRONMENT=production

# Set required production variables
export DB_PASSWORD=your_secure_password
export REDIS_PASSWORD=your_redis_password
export REALTIME_STREAM_API_KEY=your_api_key

# Deploy with Docker Compose
docker-compose -f docker-compose.yml up -d
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Environment
ENVIRONMENT=development  # or production

# Database
DB_TYPE=sqlite  # or postgresql
DB_HOST=localhost
DB_PORT=5432
DB_NAME=selfsoul
DB_USER=selfsoul
DB_PASSWORD=  # Required in production

# Security
REALTIME_STREAM_API_KEY=your_api_key_here
REALTIME_STREAM_AUTH_TYPE=jwt  # or api_key
JWT_SECRET_KEY=your_jwt_secret

# Performance
MAX_LOADED_MODELS=8
LAZY_LOAD_ENABLED=true
QUANTIZATION_MODE=none  # none, dynamic, qat
COMPILE_ENABLED=false
```

### Port Configuration

| Service | Port | Description |
|---------|------|-------------|
| Main API | 8000 | Primary REST API endpoint |
| Realtime Stream Manager | 8766 | WebSocket streaming service |
| Model Services | 8001-8027 | Individual model endpoints |
| Frontend | 5175 | Vue.js dashboard |

### Model Services Configuration

The model services configuration is defined in `config/model_services_config.json`:

```json
{
  "model_ports": {
    "manager": 8001,
    "language": 8002,
    "knowledge": 8003,
    "vision": 8004,
    "audio": 8005,
    // ... additional models
  },
  "main_api": {
    "port": 8000,
    "host": "127.0.0.1"
  }
}
```

## Security Features

### Authentication and Authorization

1. **WebSocket Authentication**:
   - API key authentication for development
   - JWT token authentication for production
   - Production environment requires authentication

2. **Database Security**:
   - Password required for production databases
   - Connection encryption support
   - Automatic backup with encryption

3. **Container Security**:
   - Non-root user execution
   - Specific Linux capabilities instead of privileged mode
   - Hardware access through group membership

4. **API Security**:
   - Rate limiting on public endpoints
   - Input validation and sanitization
   - CORS configuration based on environment

### Production Security Checklist

- [ ] Set `ENVIRONMENT=production` in .env
- [ ] Configure strong database passwords
- [ ] Set WebSocket authentication keys
- [ ] Enable JWT authentication
- [ ] Configure HTTPS for production
- [ ] Set up firewall rules for exposed ports
- [ ] Regular security updates and patches

## Performance Optimization

### Memory Management

The system implements intelligent memory management to handle 27 models efficiently:

```python
# Model Registry Configuration
self._max_loaded_models = 8  # Maximum models loaded concurrently
self.lazy_load_enabled = True  # Load models on-demand
self.quantization_mode = 'none'  # 'none', 'dynamic', or 'qat'
self.compile_enabled = False  # Enable torch.compile optimization
```

### Optimization Techniques

1. **Lazy Loading**: Models are loaded only when required by a task
2. **LRU Unloading**: Least Recently Used models are unloaded when memory limits are reached
3. **Core Model Protection**: Essential models are protected from unloading
4. **Model Quantization**: Optional quantization to reduce model size
5. **Torch Compile**: JIT compilation for faster model execution
6. **Task-Based Loading**: Only relevant models are loaded for each task

### Performance Monitoring

Real-time performance metrics are available through the API:

```bash
# Check system health
curl http://localhost:8000/api/health

# Get detailed metrics
curl http://localhost:8000/api/health/detailed

# Monitor model performance
curl http://localhost:8000/api/metrics/performance
```

## API Documentation

### Core Endpoints

| Endpoint | Method | Description | Authentication |
|----------|--------|-------------|----------------|
| `/api/health` | GET | System health check | None |
| `/api/health/detailed` | GET | Detailed health metrics | None |
| `/api/models` | GET | List available models | None |
| `/api/models/{model_id}` | GET | Get model details | None |
| `/api/tasks` | POST | Submit a task for processing | JWT |
| `/api/tasks/{task_id}` | GET | Get task status | JWT |
| `/api/training/start` | POST | Start model training | JWT |
| `/api/training/status` | GET | Training status | JWT |
| `/api/hardware/devices` | GET | List hardware devices | JWT |
| `/api/hardware/control` | POST | Control hardware device | JWT |

### WebSocket Endpoints

| Endpoint | Description | Authentication |
|----------|-------------|----------------|
| `/ws/streams/{stream_id}` | Real-time data streaming | API Key or JWT |
| `/ws/notifications` | System notifications | API Key or JWT |
| `/ws/hardware/{device_id}` | Hardware device control | JWT |

### Example API Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/api/health")
print(response.json())

# Submit a task
task_data = {
    "task": "Analyze this image and describe what you see",
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

## Port Configuration

### Default Port Assignment

| Port | Service | Description |
|------|---------|-------------|
| 8000 | Main API | Primary REST API service |
| 8766 | Realtime Stream Manager | WebSocket streaming service |
| 8001 | Manager Model | System coordination model |
| 8002 | Language Model | Natural language processing |
| 8003 | Knowledge Model | Knowledge management |
| 8004 | Vision Model | Computer vision |
| 8005 | Audio Model | Audio processing |
| 8006 | Autonomous Model | Autonomous decision making |
| 8007 | Programming Model | Code generation and analysis |
| 8008 | Planning Model | Task planning and scheduling |
| 8009 | Emotion Model | Emotional intelligence |
| 8010 | Spatial Model | Spatial reasoning |
| 8011 | Computer Vision Model | Advanced computer vision |
| 8012 | Sensor Model | Sensor data processing |
| 8013 | Motion Model | Motion planning and control |
| 8014 | Prediction Model | Predictive analytics |
| 8015 | Advanced Reasoning Model | Advanced logical reasoning |
| 8016 | Multi-Model Collaboration | Cross-model coordination |
| 8017 | Creative Problem Solving | Creative solution generation |
| 8018 | Meta Cognition | Self-awareness and reflection |
| 8019 | Value Alignment | Ethical alignment |
| 8020 | Vision Image | Image-specific processing |
| 8021 | Vision Video | Video processing |
| 8022 | Finance Model | Financial analysis |
| 8023 | Medical Model | Medical data analysis |
| 8024 | Collaboration Model | Human-AI collaboration |
| 8025 | Optimization Model | Optimization algorithms |
| 8026 | Computer Model | Computer system management |
| 8027 | Mathematics Model | Mathematical computation |
| 8028 | Data Fusion Model | Multi-source data fusion |
| 5175 | Frontend Dashboard | Vue.js user interface |

### Port Customization

To change port assignments, modify `config/model_services_config.json`:

```json
{
  "model_ports": {
    "manager": 9001,
    "language": 9002,
    // ... other models
  },
  "main_api": {
    "port": 9000,
    "host": "127.0.0.1"
  }
}
```

## Hardware Integration

### Supported Hardware

1. **Cameras**:
   - USB cameras
   - Ethernet cameras (IP cameras)
   - CSI cameras (Raspberry Pi)
   - Multiple camera simultaneous support

2. **Sensors**:
   - Temperature and humidity sensors
   - Motion sensors (accelerometer, gyroscope)
   - Environmental sensors (pressure, light, smoke)
   - Distance sensors (ultrasonic, infrared)

3. **Actuators**:
   - Robotic arms and manipulators
   - Motor controllers
   - LED controllers
   - Relay modules

4. **Communication Protocols**:
   - Serial (UART)
   - I2C
   - SPI
   - TCP/IP
   - UDP
   - WebSocket

### Hardware Configuration

Hardware configuration is defined in `config/model_services_config.json`:

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

### Docker Hardware Access

For hardware access in Docker containers:

```yaml
# docker-compose.yml
services:
  backend:
    devices:
      - /dev/video0:/dev/video0  # Camera access
      - /dev/ttyUSB0:/dev/ttyUSB0  # Serial device access
      - /dev/dri:/dev/dri  # GPU acceleration
    cap_add:
      - SYS_RAWIO  # Raw I/O access
      - SYS_ADMIN  # System administration
      - DAC_OVERRIDE  # File permission override
    group_add:
      - dialout  # Serial port access
      - video  # Video device access
      - gpio  # GPIO access (if available)
```

## Usage Guide

### Starting the System

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Using Python directly
python -m core.main

# Using development script
python scripts/start-dev.py
```

### Accessing the Dashboard

1. Open browser to `http://localhost:5175`
2. Login with default credentials (if authentication enabled)
3. Navigate through dashboard sections:
   - **System Overview**: Overall system status
   - **Model Management**: View and control AI models
   - **Training Interface**: Start and monitor training sessions
   - **Hardware Control**: Manage connected hardware devices
   - **API Documentation**: Interactive API documentation

### Basic Tasks

1. **Text Analysis**:
   ```bash
   curl -X POST http://localhost:8000/api/tasks \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"task": "Analyze this text", "modality": "text", "data": "Sample text to analyze"}'
   ```

2. **Image Processing**:
   ```bash
   # Convert image to base64 first
   base64_image=$(base64 -i image.jpg)
   
   curl -X POST http://localhost:8000/api/tasks \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d "{\"task\": \"Describe this image\", \"modality\": \"image\", \"data\": \"$base64_image\"}"
   ```

3. **Training Models**:
   ```bash
   curl -X POST http://localhost:8000/api/training/start \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"model_ids": ["language", "vision"], "epochs": 10, "batch_size": 32}'
   ```

### Monitoring and Maintenance

1. **Check System Health**:
   ```bash
   curl http://localhost:8000/api/health
   ```

2. **View Logs**:
   ```bash
   # Docker logs
   docker-compose logs -f backend
   
   # File logs
   tail -f logs/system.log
   ```

3. **Performance Monitoring**:
   ```bash
   curl http://localhost:8000/api/metrics/performance
   ```

## Troubleshooting

### Common Issues

#### Issue: "ImportError" or "NameError"
**Solution**: Run the import validation script:
```bash
python check_imports.py
python test_models_import.py
```

#### Issue: Docker permission errors
**Solution**: Ensure Docker has proper permissions:
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Restart Docker service
sudo systemctl restart docker
```

#### Issue: WebSocket connection failed
**Solution**: Check authentication configuration:
```bash
# Verify environment variables
echo $REALTIME_STREAM_API_KEY
echo $ENVIRONMENT

# Test WebSocket connection
python -c "import websocket; ws = websocket.WebSocket(); ws.connect('ws://localhost:8766/ws/streams/test?api_key=YOUR_KEY')"
```

#### Issue: Memory exhaustion (OOM)
**Solution**: Adjust memory management settings:
```python
# In model_registry.py or via environment
export MAX_LOADED_MODELS=5  # Reduce from 8 to 5
export LAZY_LOAD_ENABLED=true
```

### Performance Optimization Tips

1. **Reduce loaded models**: Set `MAX_LOADED_MODELS=5` for systems with limited RAM
2. **Enable quantization**: Set `QUANTIZATION_MODE=dynamic` to reduce model size
3. **Use GPU acceleration**: Ensure CUDA is available and configured
4. **Monitor resource usage**: Use the performance monitoring dashboard
5. **Optimize batch sizes**: Adjust based on available memory

### Debugging Techniques

1. **Enable debug logging**:
   ```bash
   export LOG_LEVEL=DEBUG
   docker-compose restart backend
   ```

2. **Check service status**:
   ```bash
   docker-compose ps
   docker-compose logs --tail=100 backend
   ```

3. **Test individual models**:
   ```bash
   curl http://localhost:8001/health  # Manager model
   curl http://localhost:8002/health  # Language model
   ```

## Contributing

We welcome contributions to Self Soul! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines

- Follow existing code style and conventions
- Add tests for new functionality
- Update documentation for changes
- Ensure backward compatibility
- Follow security best practices

### Testing

```bash
# Run import tests
python check_imports.py

# Run model import tests
python test_models_import.py

# Run security tests
python test_deployment_fixes.py

# Frontend type checking
cd app
npm run type-check
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Open Source Community**: For the incredible tools and libraries that make this project possible
- **AI Research Community**: For advancing the field of artificial intelligence
- **Contributors**: Everyone who has contributed code, issues, or ideas
- **Users**: For testing, feedback, and real-world deployment

### Special Thanks

1. **PyTorch Team**: For the excellent deep learning framework
2. **FastAPI Team**: For the high-performance web framework
3. **Vue.js Team**: For the progressive JavaScript framework
4. **Docker Team**: For containerization technology
5. **All External API Providers**: For making AI accessible

---

**Self Soul Team** - Building the future of Artificial General Intelligence

*For questions, issues, or contributions, please open an issue on GitHub or contact silencecrowtom@qq.com*