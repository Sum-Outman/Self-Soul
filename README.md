# Self Soul - Advanced AGI System

**Developed by**: Self Soul Team  
**Developer Email**: silencecrowtom@qq.com  
**Repository**: https://github.com/Sum-Outman/Self-Soul

## Project Overview

Self Soul is a sophisticated Artificial General Intelligence (AGI) platform that integrates 19 specialized AI models into a unified cognitive architecture. This open-source system provides comprehensive multi-modal intelligence capabilities including natural language processing, computer vision, audio analysis, emotional intelligence, autonomous learning, and advanced reasoning. The system is designed to support true training from scratch, multi-camera vision capabilities, external device integration, and seamless switching between local and external API models.

### Design Philosophy

Self Soul is built on the core principle that true AGI requires a cohesive, integrated architecture rather than isolated models. Our design philosophy emphasizes:

- **Unified Cognitive Architecture**: All 19 models work synergistically through a central coordination system, enabling emergent intelligence greater than the sum of individual components
- **From-Scratch Training**: By training all models from scratch without pre-trained foundations, we maintain full control over model development, ethical alignment, and AGI compliance
- **Human-Centered Intelligence**: Integrating emotional intelligence and value alignment to ensure AI behavior is responsible, ethical, and aligned with human values
- **Modular Extensibility**: A flexible architecture that allows easy integration of new AI capabilities while maintaining system coherence

### Technical Highlights

- **19 Specialized AI Models**: Comprehensive coverage of cognitive capabilities from basic perception to advanced reasoning
- **Advanced Multi-Modal Integration**: Seamless processing and fusion of text, image, audio, and video data
- **Adaptive Learning Engine**: Real-time optimization of learning strategies and training parameters based on performance metrics
- **Distributed Processing**: Each model runs on a dedicated port (8001-8019) enabling parallel processing and scalability
- **Modern UI/UX**: Intuitive Vue.js-based dashboard for system management and monitoring
- **Comprehensive API**: RESTful interface for integration with external systems and applications
- **Multimodal Dataset Support**: Expanded Multimodal Dataset v1 supports all 19 models for comprehensive training

Self Soul represents a significant advancement in AGI research and development, offering a fully functional platform for exploring the frontiers of artificial general intelligence.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Stars](https://img.shields.io/github/stars/Sum-Outman/Self-Soul)](https://github.com/Sum-Outman/Self-Soul/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/Sum-Outman/Self-Soul)](https://github.com/Sum-Outman/Self-Soul/issues)

### Core Features

#### Multi-Modal Integration
Seamlessly combines text, image, audio, and video processing with a unified cognitive architecture. This allows the system to understand and process information in multiple formats simultaneously, creating a more comprehensive understanding of the environment.

**Use Cases**: Media analysis, content creation, multi-sensory user interfaces, environmental perception systems

#### Autonomous Learning
Advanced self-directed learning system with continuous adaptation capabilities. The system can independently improve its performance without explicit human instruction.

- **Automatic model improvement** based on real-time performance metrics
- **Adaptive curriculum learning** that identifies and focuses on model weak areas
- **Real-time learning optimization** using meta-reinforcement learning
- **Multi-model coordination** for synchronized autonomous learning across all 19 AI capabilities
- **Built-in Adaptive Learning Engine** for strategy optimization

**Use Cases**: Self-improving customer service systems, adaptive educational platforms, autonomous research assistants

#### Autonomous Training
Comprehensive autonomous training capabilities accessible through the training interface at http://localhost:5175/#/training. Users can initiate fully autonomous training sessions with minimal configuration.

- **Fully autonomous training mode** accessible via the intuitive web interface
- **Self-optimizing training parameters** (learning rate, batch size, epochs) based on model performance
- **Intelligent dataset selection** based on model capabilities and training objectives
- **Multi-model coordination** for synchronized autonomous training across all 19 AI capabilities
- **Built-in Adaptive Learning Engine** for parameter adjustment
- **Continuous training loops** with minimal human supervision

#### Training Interface Features

The training interface provides a comprehensive set of tools for managing model training:

- **Training Mode Selection**: Choose between Individual or Joint training modes
  - **Individual Mode**: Train a single model at a time with focused parameters
  - **Joint Mode**: Train multiple models simultaneously with coordinated optimization

- **Model Selection System**:
  - **Recommended Combinations**: Pre-configured model combinations for common use cases
  - **Select All Models**: Quick selection of all 19 AI models for comprehensive training
  - **Model Grid**: Visual selection interface with tooltips and status indicators
  - **Dependency Visualization**: Shows model dependencies for joint training validation

- **Dataset Management**:
  - **Multimodal Dataset v1**: Supports all 19 models with comprehensive format support
  - **Specialized Datasets**: Language-only, vision-only, sensor-only, and other domain-specific datasets
  - **Upload Functionality**: Support for uploading custom datasets with format validation
  - **Supported Models Display**: Shows compatible models for each dataset

- **Parameter Configuration**:
  - **Basic Parameters**: Epochs, Batch Size, Learning Rate, Validation Split
  - **Advanced Parameters**: Dropout Rate, Weight Decay, Momentum, Optimizer Selection
  - **From-Scratch Option**: Toggle between training from scratch or using existing weights

- **Training Strategy Options**:
  - **Default Strategy**: Balanced approach suitable for most use cases
  - **Knowledge-Assisted Training**: Leverages existing knowledge base for faster learning
  - **Adaptive Curriculum Learning**: Dynamically adjusts training difficulty based on performance
  - **Transfer Learning**: Facilitates knowledge transfer between related models

- **Real-time Status Updates**:
  - **Progress Visualization**: Live training progress with metrics and charts
  - **Error Handling**: Clear error messages and troubleshooting suggestions
  - **Success Feedback**: Confirmation messages for completed operations

- **AGI Training State Monitoring**:
  - **Meta-learning Progress**: Tracks self-improvement capabilities
  - **Knowledge Integration Level**: Measures how well models integrate new knowledge
  - **Autonomous Learning Score**: Quantifies self-directed learning efficiency
  - **Adaptive Learning Efficiency**: Evaluates parameter optimization effectiveness

**Use Cases**: Rapid model development, continuous system improvement, large-scale training operations

#### Advanced Language Processing
Supports multiple languages with deep contextual understanding and reasoning capabilities. The language model features a custom transformer-based architecture for sophisticated text processing.

**Use Cases**: Cross-lingual communication, content generation, document analysis, conversational AI

#### Real-time Monitoring
Comprehensive system performance tracking and model metrics visualization. The monitoring system provides real-time insights into model behavior and system resource usage.

**Use Cases**: System optimization, performance debugging, resource allocation, training progress tracking

#### Interactive Dashboard
Modern Vue.js-based interface for intuitive system management. The dashboard provides a centralized location for controlling all aspects of the Self Soul system.

**Use Cases**: System administration, model configuration, training management, performance monitoring

#### Extensible Architecture
Modular design allowing easy integration of new AI capabilities. The system's component-based architecture enables seamless addition of new models and features.

**Use Cases**: Custom AI capability integration, third-party system connectivity, specialized application development

#### Knowledge Management
Advanced knowledge base with structured and unstructured data integration. The knowledge model organizes information in a way that enables efficient retrieval and reasoning.

**Use Cases**: Enterprise knowledge management, research assistance, information retrieval systems, decision support tools

#### Emotional Intelligence
Emotion recognition and response capabilities for human-like interactions. The emotion model can identify and appropriately respond to human emotional states.

**Use Cases**: Customer service automation, mental health support, personalized user experiences, social robotics

#### Advanced Reasoning
Complex logical reasoning and creative problem-solving abilities. The reasoning models enable the system to tackle complex problems and generate innovative solutions.

**Use Cases**: Scientific research assistance, engineering design, strategic planning, creative content generation

#### Value Alignment
Ethical guidelines integration to ensure responsible AI behavior. The value alignment model ensures the system's actions are consistent with human values and ethical principles.

**Use Cases**: Ethical AI applications, compliance monitoring, responsible decision-making systems, safe AGI development

## System Architecture

Self Soul employs a layered architecture that separates core AI capabilities from the user interface, enabling modular development and deployment. The architecture is designed to support the coordination of 19 specialized AI models while providing a unified user experience.

### Architecture Overview

The Self Soul architecture consists of three main layers:

1. **Presentation Layer**: Vue.js-based frontend application that provides user interfaces for system management, monitoring, and interaction
2. **API Layer**: RESTful API gateway that handles communication between frontend and backend services
3. **Core Layer**: Distributed AI models and management services that form the heart of the system

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

### Core Components

#### Frontend Components

- **Main Dashboard**: Central hub for system status and quick actions
- **Model Management**: Interface for configuring and controlling individual AI models
- **Training Interface**: Comprehensive tools for managing model training processes
- **Knowledge Base**: Interface for managing the system's knowledge repository
- **Settings Panel**: Configuration options for system behavior and preferences
- **Help Center**: Documentation and troubleshooting resources

#### Backend Services

- **Main API Gateway (Port 8000)**: Central entry point for all frontend requests
- **Model Service Manager**: Creates and manages instances of all 19 AI models
- **Model Registry**: Maintains information about all available models and their capabilities
- **Training Manager**: Coordinates training activities across all models
- **Autonomous Learning Manager**: Oversees self-learning processes and optimization
- **Joint Training Coordinator**: Manages multi-model training synchronization
- **Error Handling System**: Centralized logging and error management

### Component Interaction Flow

Self Soul components interact through a well-defined set of communication pathways:

#### User Request Flow

1. **User Action**: User interacts with the frontend interface (e.g., initiating model training)
2. **Frontend Processing**: Vue component processes the request and prepares API call
3. **API Request**: Frontend sends request to Main API Gateway (Port 8000)
4. **API Routing**: Gateway routes request to appropriate backend service
5. **Service Processing**: Target service processes the request (e.g., Training Manager)
6. **Model Interaction**: Service interacts with relevant AI models (e.g., initiating training on specific models)
7. **Response Flow**: Results are passed back through the same pathway to the user interface

#### Model Coordination Flow

1. **Manager Model Request**: Manager Model (Port 8001) identifies need for specialized model
2. **Service Discovery**: Model Service Manager locates appropriate model service
3. **Model Invocation**: Relevant model is invoked via its dedicated port (8002-8019)
4. **Result Processing**: Model processes request and returns results
5. **Coordinated Response**: Manager Model integrates results and returns final response

#### Training Flow

1. **Training Initiation**: User or Autonomous Learning Manager requests model training
2. **Dataset Selection**: Training Manager selects appropriate dataset based on model capabilities
3. **Training Configuration**: Parameters are optimized by Adaptive Learning Engine
4. **Model Training**: Individual models train on their dedicated ports
5. **Progress Monitoring**: Real-time metrics are collected and displayed
6. **Model Update**: Trained models are saved and reloaded into the system

### Data Flow

- **Input Data**: User inputs, uploaded files, and external data sources
- **Processing Pipeline**: Data flows through appropriate models based on type and context
- **Knowledge Integration**: Processed information is integrated into the knowledge base
- **Output Generation**: Final results are formatted for user presentation or system use

The modular architecture of Self Soul ensures that components can be developed, tested, and deployed independently while maintaining seamless integration with the rest of the system.

## From-Scratch Training Architecture

**Self Soul implements a complete from-scratch training architecture for all 19 models, without relying on any pre-trained models.** This ensures full control over model development, ethical alignment, and AGI compliance.

### Core Training Principles

Self Soul's training architecture is guided by four fundamental principles:

- **No Pre-trained Models**: All neural networks are built and trained from scratch using PyTorch, ensuring complete control over every parameter and layer
- **Custom Architectures**: Each model type has specialized neural network designs tailored to its specific cognitive function
- **AGI-Compliant Design**: Models follow AGI principles for unified cognitive architecture, enabling emergent intelligence through model coordination
- **Autonomous Improvement**: Built-in self-learning and meta-cognition capabilities allow models to continuously improve without human intervention

### From-Scratch Implementation Details

#### Training Pipeline Architecture

The from-scratch training pipeline consists of several key components:

1. **Data Preparation Layer**: Raw data processing, normalization, and augmentation
2. **Model Architecture Layer**: Custom neural network designs for each specialized model
3. **Training Execution Layer**: Distributed training across dedicated ports (8001-8019)
4. **Optimization Layer**: Adaptive Learning Engine for real-time parameter adjustment
5. **Evaluation Layer**: Comprehensive performance metrics and validation
6. **Persistence Layer**: Model saving, loading, and version management

#### Neural Network Design Approach

Each model employs a specialized neural network architecture designed specifically for its cognitive function:

- **Manager Model**: Hierarchical coordination networks with attention mechanisms for task allocation
- **Language Model**: Custom transformer-based architecture with variable-sized attention windows and dynamic layer configuration
- **Vision Models**: Hybrid CNN architectures with adaptive pooling and multi-scale feature extraction
- **Audio Model**: Recurrent-convolutional networks for temporal audio signal processing
- **Reasoning Models**: Graph neural networks with symbolic reasoning capabilities
- **Emotion Model**: Dense neural networks with attention mechanisms for emotional pattern recognition

#### Training Data Management

- **Multimodal Dataset v1**: Comprehensive dataset supporting all 19 models, with:
  - Text: Plain text, structured documents, code repositories
  - Images: Photographs, diagrams, charts across multiple domains
  - Audio: Speech, music, environmental sounds with metadata
  - Video: Multi-camera footage with synchronized audio
- **Data Augmentation**: Dynamic augmentation strategies tailored to each data type
- **Dataset Compatibility**: Frontend `supportedModels` property ensures dataset-model alignment
- **Backend Validation**: `dataset_manager.py` validates file formats against model requirements

### Model Architecture Overview

#### Foundational Models (From-Scratch Implementation)

- **Manager Model (Port 8001)**: CoordinationNeuralNetwork, TaskAllocationNetwork - Manages system resources and orchestrates other models with hierarchical attention mechanisms
- **Language Model (Port 8002)**: LanguageNeuralNetwork, FromScratchLanguageTrainer - Custom transformer architecture with 12 layers, 8 attention heads, and 768 hidden dimensions
- **Knowledge Model (Port 8003)**: KnowledgeGraphNetwork, SemanticEmbeddingLayer - Graph-based architecture for structured and unstructured knowledge organization
- **Vision Model (Port 8004)**: SimpleVisionCNN, VisionDataset - Custom CNN with 5 convolutional layers, adaptive pooling, and multi-scale feature fusion
- **Audio Model (Port 8005)**: AudioRecognitionNetwork, SpectrogramLayer - Recurrent-convolutional architecture for sound and speech understanding
- **Autonomous Model (Port 8006)**: AdvancedDecisionNetwork, ExperienceReplayBuffer - Reinforcement learning architecture with experience replay and value functions

#### Advanced Models (Complete From-Scratch Training)

- **Programming Model (Port 8007)**: ProgrammingNeuralNetwork, CodeEmbeddingLayer - Syntax-aware neural network for code generation and optimization
- **Planning Model (Port 8008)**: PlanningStrategyNetwork, StepPredictionNetwork - Hierarchical planning architecture with temporal reasoning
- **Emotion Model (Port 8009)**: EmotionRecognitionNetwork, AffectiveLayer - Dense network with attention mechanisms for emotional intelligence
- **Spatial Model (Port 8010)**: SpatialNeuralNetwork, GeometricEmbeddingLayer - 3D spatial reasoning architecture with coordinate transforms
- **Computer Vision Model (Port 8011)**: VisualUnderstandingNetwork, ObjectDetectionLayer - Advanced CNN for object detection and scene understanding
- **Sensor Model (Port 8012)**: SensorNeuralNetwork, SignalProcessingLayer - Multi-modal sensor data fusion architecture
- **Motion Model (Port 8013)**: TrajectoryPlanningNetwork, KinematicsLayer - Physics-aware motion prediction and planning
- **Prediction Model (Port 8014)**: PredictionNeuralNetwork (LSTM+Attention), TimeSeriesLayer - Sequential prediction architecture with attention mechanisms
- **Advanced Reasoning Model (Port 8015)**: ReasoningGraphNetwork, LogicLayer - Graph-based reasoning architecture with symbolic integration
- **Data Fusion Model (Port 8016)**: FusionNeuralNetwork, CrossModalAttentionLayer - Multi-source information integration with cross-modal attention
- **Creative Problem Solving Model (Port 8017)**: CreativeGenerationNetwork, AnalogicalReasoningLayer - Neural generators with combinatorial innovation capabilities
- **Meta Cognition Model (Port 8018)**: MetaLearningNetwork, SelfReflectionLayer - Experience-based learning and self-monitoring architecture
- **Value Alignment Model (Port 8019)**: EthicalReasoningNetwork, ValueSystemLayer - Custom tokenization and value alignment architecture

### Training System Features

#### Unified Training Interface

All models implement a standardized training interface:
- `enable_training()`: Initializes training mode and prepares resources
- `disable_training()`: Finalizes training and saves model state
- `train_step()`: Executes a single training iteration
- `evaluate()`: Runs validation and returns performance metrics

#### Autonomous Training Interface

The advanced training interface at http://localhost:5175/#/training provides:
- **One-click autonomous training mode** with minimal user configuration
- **Automatic parameter optimization** based on real-time model performance metrics
- **Adaptive training scheduling** and dynamic curriculum design that evolves with model capabilities
- **Real-time progress visualization** with actionable autonomous improvement suggestions
- **Intelligent dataset selection** based on training objectives and model capabilities
- **Multi-model coordination** for synchronized autonomous learning across all 19 AI capabilities
- **Seamless integration** with Multimodal Dataset v1, supporting all models for comprehensive training
- **Automatic performance monitoring** with autonomous training termination when optimal results are achieved

#### Adaptive Learning Engine

The core of Self Soul's autonomous training capabilities is the Adaptive Learning Engine:
- **Real-time Parameter Tuning**: Adjusts learning rate, batch size, and regularization on-the-fly
- **Performance-Based Optimization**: Uses validation metrics to optimize model hyperparameters
- **Curriculum Learning**: Dynamically adjusts training difficulty based on model performance
- **Transfer Learning Between Models**: Enables knowledge transfer across specialized models
- **Early Stopping Detection**: Automatically terminates training when performance plateaus

#### Model Persistence and Versioning

- **Standardized Save/Load**: All models implement `save_model()` and `load_model()` methods
- **Checkpoint System**: Regularly saves model state during training
- **Version Management**: Tracks model versions and training parameters
- **Compatibility Layer**: Ensures backward compatibility between model versions

#### Distributed Training Architecture

- **Dedicated Ports**: Each model runs on a dedicated port (8001-8019) for parallel processing
- **Centralized Management**: `FromScratchTrainingManager` coordinates all training activities
- **Synchronized Updates**: Ensures consistent model improvements across the system
- **Resource Optimization**: Dynamically allocates system resources based on training needs

#### Multimodal Dataset v1

The expanded Multimodal Dataset v1 provides comprehensive support for all models:
- **Format Support**: Multiple file formats per model type (e.g., JSON, CSV, TXT for language; JPG, PNG for vision)
- **Dataset-Model Mapping**: Explicit `supportedModels` property in frontend configuration
- **Backend Validation**: `dataset_manager.py` validates file formats against model requirements
- **Scalable Design**: Supports future model additions with minimal configuration changes

### Technical Challenges and Solutions

#### Challenge: Training 19 Models Simultaneously
**Solution**: Distributed architecture with dedicated ports and centralized coordination

#### Challenge: Maintaining Model Coherence
**Solution**: Unified cognitive architecture and knowledge transfer between models

#### Challenge: Optimization Across Diverse Model Types
**Solution**: Adaptive Learning Engine with model-specific optimization strategies

#### Challenge: Ensuring Ethical Alignment
**Solution**: Value Alignment Model (Port 8019) integrated into all decision-making processes

Self Soul's from-scratch training architecture represents a significant breakthrough in AGI development, enabling truly autonomous learning and adaptation across a comprehensive set of cognitive capabilities.

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

### Language Support

- English

### Detailed Configuration

#### Environment Variables

Create a `.env` file in the project root directory with the following configuration:

```
# API Configuration
API_KEY=your-api-key-here
API_BASE_URL=http://localhost:8000

# Model Configuration
MODEL_PATH=./models
DATA_PATH=./data
TRAINING_PATH=./training

# System Configuration
LOG_LEVEL=INFO
PORT=8000
WS_PORT=8001

# Database Configuration
DATABASE_URL=sqlite:///./self_soul.db

# External Service Configuration
EXTERNAL_API_URL=https://api.external-service.com
EXTERNAL_API_KEY=your-external-api-key-here
```

#### Model Configuration

Model configurations are stored in `config/model_services_config.json`: 

```json
{
  "model_services": [
    {"id": "manager", "name": "Manager Model", "port": 8001, "enabled": true},
    {"id": "language", "name": "Language Model", "port": 8002, "enabled": true},
    {"id": "knowledge", "name": "Knowledge Model", "port": 8003, "enabled": true},
    {"id": "vision", "name": "Vision Model", "port": 8004, "enabled": true},
    {"id": "audio", "name": "Audio Model", "port": 8005, "enabled": true},
    {"id": "autonomous", "name": "Autonomous Model", "port": 8006, "enabled": true},
    {"id": "programming", "name": "Programming Model", "port": 8007, "enabled": true},
    {"id": "planning", "name": "Planning Model", "port": 8008, "enabled": true},
    {"id": "emotion", "name": "Emotion Model", "port": 8009, "enabled": true},
    {"id": "spatial", "name": "Spatial Model", "port": 8010, "enabled": true},
    {"id": "computer_vision", "name": "Computer Vision Model", "port": 8011, "enabled": true},
    {"id": "sensor", "name": "Sensor Model", "port": 8012, "enabled": true},
    {"id": "motion", "name": "Motion Model", "port": 8013, "enabled": true},
    {"id": "prediction", "name": "Prediction Model", "port": 8014, "enabled": true},
    {"id": "advanced_reasoning", "name": "Advanced Reasoning Model", "port": 8015, "enabled": true},
    {"id": "data_fusion", "name": "Data Fusion Model", "port": 8016, "enabled": true},
    {"id": "creative_problem_solving", "name": "Creative Problem Solving Model", "port": 8017, "enabled": true},
    {"id": "meta_cognition", "name": "Meta Cognition Model", "port": 8018, "enabled": true},
    {"id": "value_alignment", "name": "Value Alignment Model", "port": 8019, "enabled": true}
  ]
}
```

### Usage Examples

#### 1. System Initialization

Start the backend service:

```bash
python core/main.py
```

Start the frontend application:

```bash
cd app && npm run dev
```

#### 2. Model Management

**Check Model Status**:
```bash
curl -X GET http://localhost:8000/api/models/status
```

**Switch Model to External API**:
```bash
curl -X POST http://localhost:8000/api/models/language/external \
  -H "Content-Type: application/json" \
  -d '{"api_url": "https://api.external-service.com/language", "api_key": "your-api-key"}'
```

**Switch Model to Local**:
```bash
curl -X POST http://localhost:8000/api/models/language/local
```

#### 3. Training

**Initiate Training via API**:
```bash
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{"model_id": "language", "dataset_id": "multimodal_v1", "parameters": {"epochs": 10, "batch_size": 32, "learning_rate": 0.001}}'
```

**Monitor Training Progress**:
```bash
# Connect to WebSocket endpoint for real-time updates
wscat -c ws://localhost:8000/ws/training/12345
```

#### 4. Knowledge Management

**Import Knowledge**:
```bash
curl -X POST http://localhost:8000/api/knowledge/import \
  -H "Content-Type: application/json" \
  -d '{"title": "Example Knowledge", "content": "This is an example of imported knowledge."}'
```

**Search Knowledge**:
```bash
curl -X GET "http://localhost:8000/api/knowledge/search?q=example"
```

#### 5. Chat Interface

**Chat with Language Model**:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?", "session_id": "test_session", "model_id": "language"}'
```

**Chat with Manager Model**:
```bash
curl -X POST http://localhost:8000/api/models/8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Coordinate with all models to analyze this text", "session_id": "test_session", "text": "This is a test text for analysis."}'
```

### Advanced Usage

#### Batch Model Operations

**Batch Switch Model Modes**:
```bash
curl -X POST http://localhost:8000/api/models/batch/switch \
  -H "Content-Type: application/json" \
  -d '{"model_ids": ["language", "vision", "audio"], "mode": "external", "api_config": {"api_url": "https://api.external-service.com"}}'
```

**Batch Update Model Parameters**:
```bash
curl -X POST http://localhost:8000/api/models/batch/update \
  -H "Content-Type: application/json" \
  -d '{"model_ids": ["language", "vision"], "parameters": {"max_tokens": 1000, "temperature": 0.7}}'
```

#### Autonomous Learning

**Start Autonomous Learning**:
```bash
curl -X POST http://localhost:8000/api/autonomous-learning/start \
  -H "Content-Type: application/json" \
  -d '{"model_ids": ["language", "vision"], "duration": 3600, "objective": "improve_comprehension"}'
```

**Stop Autonomous Learning**:
```bash
curl -X POST http://localhost:8000/api/autonomous-learning/stop
```

### Page-by-Page Functionality Guide

#### Homepage (http://localhost:5175/#/)

The homepage serves as the main dashboard for system monitoring and control, featuring:

- **Real-time System Monitoring**: Live status updates for all 19 AI models with visual indicators
- **Device Management**: Multi-camera control, sensor data visualization, and serial communication interfaces
- **WebSocket Communication**: Real-time device control and status updates
- **Stereo Vision Processing**: Advanced vision capabilities with camera calibration and depth perception
- **System Health Dashboard**: Performance metrics, resource usage, and error alerts
- **Quick Access Buttons**: One-click navigation to other main pages

#### Conversation Page (http://localhost:5175/#/conversation)

The conversation page enables multimodal interaction with the AGI system:

- **Multimodal Messaging**: Support for text, image, audio, and video inputs
- **Model Connection Status**: Real-time indication of management model (port 8001) connectivity
- **Emotion Analysis**: Display of emotion detection results and confidence scores
- **Message History**: Persistent conversation logs with timestamps
- **Error Handling**: Graceful degradation when model connectivity is lost
- **Responsive Design**: Adaptive layout for different screen sizes

#### Training Page (http://localhost:5175/#/training)

The training page provides comprehensive tools for model training and optimization:

- **Training Modes**: Individual (single model) or Joint (multiple models) training options
- **Model Selection**: Support for selecting individual models or recommended combinations
- **Dependency Management**: Visualization of model dependencies for joint training
- **Dataset Selection**: Support for Multimodal Dataset v1 and specialized datasets
- **Training Parameters**: Configurable epochs, batch size, learning rate, and regularization
- **Training Strategies**: Knowledge-assisted training and pretrained fine-tuning options
- **Progress Tracking**: Real-time monitoring via WebSocket with fallback polling
- **Evaluation Metrics**: Accuracy, loss, and confusion matrix visualization
- **Training History**: Logs and results from previous training sessions

#### Knowledge Page (http://localhost:5175/#/knowledge)

The knowledge page manages the system's knowledge base with advanced file handling:

- **Import Functionality**: Support for uploading PDF, DOCX, TXT, JSON, and CSV files
- **Domain Classification**: Automatic and manual categorization of knowledge files
- **File Management**: Preview, download, delete, and organize knowledge entries
- **Auto-Learning**: Scheduled knowledge processing with progress tracking
- **Statistics Dashboard**: Visual representation of knowledge distribution across domains
- **WebSocket Integration**: Real-time updates for auto-learning progress

#### Settings Page (http://localhost:5175/#/settings)

The settings page allows comprehensive configuration of system parameters:

- **Model Configuration**: Toggle between local and external API models
- **API Settings**: Configure API keys, URLs, and connection parameters
- **Hardware Settings**: Camera resolution, serial port baud rate, and sensor configurations
- **Batch Operations**: Start, stop, or restart all models simultaneously
- **System Management**: Restart system services and reset configurations
- **Connection Testing**: Verify connectivity to external API models

### Interface Usage Examples

#### Homepage Navigation

```bash
# Access the homepage
echo "Open http://localhost:5175/#/ in your browser"
```

#### Training Workflow

1. **Access Training Page**: Navigate to `http://localhost:5175/#/training`
2. **Select Training Mode**: Choose "Individual" or "Joint" mode
3. **Choose Models**: Select models from the grid or use recommended combinations
4. **Configure Parameters**: Set epochs, batch size, and learning rate
5. **Start Training**: Click "Start Training" and monitor progress

#### Knowledge Management

1. **Access Knowledge Page**: Navigate to `http://localhost:5175/#/knowledge`
2. **Upload Files**: Click "Import Knowledge" and select files to upload
3. **Classify Content**: Assign domains to uploaded files or use auto-classification
4. **View Statistics**: Check knowledge distribution across domains
5. **Start Auto-Learning**: Initiate knowledge processing for model improvement



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

Self Soul provides a comprehensive RESTful API and WebSocket endpoints for integration with other applications. The API is organized into several categories for easy navigation and usage.

### Base URL

```
http://localhost:8000
```

### API Categories

#### 1. System Status and Health

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/health` | GET | System health check | `{"status": "ok", "message": "Self Soul system is running normally"}` |
| `/api/models/status` | GET | Get status of all models | `{"status": "success", "data": {model_id: {status_info}}}` |
| `/api/models/language/status` | GET | Get status of language model | `{"status": "success", "data": {language_model_status}}` |
| `/api/models/management/status` | GET | Get status of management model | `{"status": "success", "data": {management_model_status}}` |
| `/api/models/from_scratch/status` | GET | Get status of from scratch models | `{"status": "success", "data": {from_scratch_status}}` |

#### 2. Model Management

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/api/models/getAll` | GET | Get information about all models | `{"status": "success", "data": [models]}` |
| `/api/models/available` | GET | Get available models for frontend selection | `{"status": "success", "models": [available_models]}` |
| `/api/models/config` | GET | Get configurations for all models | `{"status": "success", "data": {model_configs}}` |

#### 3. Data Processing

| Endpoint | Method | Description | Request Body | Response |
|----------|--------|-------------|--------------|----------|
| `/api/process/text` | POST | Process text input | `{"text": "text_to_process", "lang": "en"}` | `{"status": "success", "data": {processing_result}}` |
| `/api/chat` | POST | Chat with language model | `{"message": "user_message", "session_id": "session_123", "conversation_history": []}` | `{"status": "success", "data": {response, conversation_history}}` |
| `/api/models/8001/chat` | POST | Chat with manager model | `{"message": "user_message", "session_id": "session_123", "conversation_history": []}` | `{"status": "success", "data": {response, conversation_history}}` |

#### 4. Device and External Interfaces

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/api/devices/cameras` | GET | Get list of available cameras | `{"status": "success", "data": [cameras]}` |
| `/api/serial/ports` | GET | Get list of available serial ports | `{"status": "success", "data": [serial_ports]}` |
| `/api/devices/external` | GET | Get information about all external devices | `{"status": "success", "data": [devices]}` |
| `/api/serial/connect` | POST | Connect to a serial port device | `{"device_id": "dev_123", "port": "COM3", "baudrate": 9600}` | `{"status": "success", "data": {connection_result}}` |
| `/api/serial/disconnect` | POST | Disconnect from a serial port device | `{"device_id": "dev_123"}` | `{"status": "success", "data": {disconnection_result}}` |

#### 5. WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/ws/training/{job_id}` | Real-time training progress updates |
| `/ws/monitoring` | System monitoring data |
| `/ws/test-connection` | WebSocket connection test |
| `/ws/autonomous-learning/status` | Autonomous learning status updates |
| `/ws/audio-stream` | Real-time audio stream processing |
| `/ws/video-stream` | Real-time video stream processing |

### API Usage Examples

#### Health Check

```bash
curl -X GET http://localhost:8000/health
```

Response:
```json
{"status": "ok", "message": "Self Soul system is running normally"}
```

#### Chat with Language Model

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?", "session_id": "test_session"}'
```

Response:
```json
{
  "status": "success",
  "data": {
    "response": "I'm doing well, thank you for asking!",
    "conversation_history": [
      {"role": "user", "content": "Hello, how are you?"},
      {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
    ],
    "session_id": "test_session"
  }
}
```

#### Get Model Status

```bash
curl -X GET http://localhost:8000/api/models/status
```

Response:
```json
{
  "status": "success",
  "data": {
    "manager": {"id": "manager", "name": "Manager Model", "status": "active", ...},
    "language": {"id": "language", "name": "Language Model", "status": "active", ...},
    ...
  }
}
```

### API Documentation Access

For interactive API documentation with Swagger UI, visit:

http://localhost:8000/docs (when backend is running)

This interactive documentation provides detailed information about all API endpoints, request parameters, and response formats. You can also test API calls directly from the Swagger UI interface.

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
