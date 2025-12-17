# Self Soul Frontend Application

This is the frontend application for the Self Soul System, built with Vue 3 and Vite. This application provides a comprehensive interface for monitoring, controlling, and interacting with the 19 specialized AI models that form the Self Soul AGI-compliant system, all trained from scratch without any pre-trained models.

## From-Scratch Training Architecture Overview

The Self Soul system implements a complete from-scratch training architecture across all 19 model types:

### Core Training Principles
- **No Pre-trained Models**: All models are trained from scratch using custom neural network architectures
- **Unified Model Template**: All models inherit from `UnifiedModelTemplate` providing consistent training interfaces
- **End-to-End Training**: Complete training loops with custom datasets, optimizers, and loss functions
- **AGI-Compliant Design**: Architecture designed to support autonomous learning and self-improvement

### Model Training Capabilities
Each model in the system supports:
- `enable_training()` / `disable_training()` - Toggle training mode
- `train_step(data)` - Perform a single training step
- `save_model(path)` / `load_model(path)` - Model persistence
- Custom training datasets and data loaders
- Multiple optimizer options (Adam, SGD, etc.)
- Comprehensive loss functions for each domain

## Getting Started

### Prerequisites
- Node.js 14+ (recommended: 16+)
- npm 6+ (recommended: 8+)

### Installation

1. Navigate to the app directory:
```bash
cd app
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

### For Windows/PowerShell Users
Use the provided PowerShell script to start the development server:
```powershell
# From the root directory
.\start-app.ps1
```

### Manual Start
If you prefer to start manually, run these commands:
```bash
# Navigate to the app directory
cd app

# Start the development server
npm run dev
```

The application will be available at http://localhost:5175

## Project Structure
- `src/` - Source code directory
  - `components/` - Vue components
    - `Dashboard.vue` - Main dashboard for model monitoring
    - `TrainingControlPanel.vue` - Control training sessions
    - `MonitorDashboard.vue` - Real-time performance monitoring
    - `KnowledgeImport.vue` - Knowledge base management
    - `TerminalWindow.vue` - System interaction interface
  - `views/` - View components for routing
    - `HomeView.vue` - System overview
    - `TrainView.vue` - Training interface
    - `ChatFromScratch.vue` - Language model interaction
    - `KnowledgeView.vue` - Knowledge management
    - `SettingsView.vue` - System configuration
  - `utils/` - Utility functions and API services
    - `api.js` - Backend API communication
    - `modelTypes.js` - Model type definitions
    - `modelIdMapper.js` - Model ID mapping utilities
  - `router/` - Vue Router configuration
  - `App.vue` - Root component
  - `main.js` - Application entry point

## API Configuration
The application connects to the backend services using the following configuration:
- Main API Gateway: http://localhost:8000
- Real-time Stream Manager: http://localhost:8765
- Performance Monitoring: http://localhost:8081

## Model Services (From-Scratch Training)

The frontend interacts with 19 specialized models, each running on dedicated ports with complete from-scratch training capabilities:

| Port | Model Type | Training Components | Description |
|------|------------|---------------------|-------------|
| 8001 | Manager | CoordinationNeuralNetwork, TaskAllocationNetwork | System coordination and task management |
| 8002 | Language | LanguageNeuralNetwork, CustomTokenizer | Natural language processing and generation |
| 8003 | Knowledge | AGI-enhanced neural components | Knowledge representation and reasoning |
| 8004 | Vision | SimpleVisionCNN, VisionDataset | Computer vision and image processing |
| 8005 | Audio | Audio processing neural networks | Audio signal processing and analysis |
| 8006 | Autonomous | AdvancedDecisionNetwork, ExperienceReplayBuffer | Autonomous decision making and learning |
| 8007 | Programming | ProgrammingNeuralNetwork, ProgrammingDataset | Code generation and analysis |
| 8008 | Planning | PlanningStrategyNetwork, StepPredictionNetwork | Task planning and strategy development |
| 8009 | Emotion | EmotionRecognitionNetwork, EmotionDataset | Emotion recognition and analysis |
| 8010 | Spatial | SpatialNeuralNetwork, SpatialDataset | Spatial reasoning and navigation |
| 8011 | Computer | CommandPredictionNetwork, SystemOptimizationNetwork | System command prediction and optimization |
| 8012 | Sensor | SensorNeuralNetwork, SensorDataset | Sensor data processing and fusion |
| 8013 | Motion | TrajectoryPlanningNetwork, MotionControlNetwork | Motion planning and control |
| 8014 | Prediction | PredictionNeuralNetwork (LSTM+Attention) | Time series prediction and forecasting |
| 8015 | Optimization | OptimizationPolicyNetwork, ResourceAllocationNetwork | System optimization and resource management |
| 8016 | Collaboration | CollaborationNeuralNetwork, StrategyOptimizationNetwork | Multi-model collaboration and coordination |
| 8017 | Creative Problem Solving | Creative reasoning networks | Creative problem solving and innovation |
| 8018 | Meta-cognition | Meta-learning networks | Self-reflection and learning optimization |
| 8019 | Value Alignment | Ethical reasoning networks | Ethical decision making and value alignment |

## Available Scripts

### `npm run dev`
Starts the development server with hot-reload.

### `npm run build`
Builds the application for production to the `dist` folder.

### `npm run preview`
Previews the production build locally.

## Training Interface Features

The frontend provides comprehensive training controls:

1. **Model Selection**: Choose any of the 19 model types for training
2. **Training Parameters**: Configure learning rate, batch size, epochs
3. **Dataset Management**: Upload and manage custom training datasets
4. **Real-time Monitoring**: Visualize training loss, accuracy, and metrics
5. **Model Persistence**: Save and load trained model weights
6. **Performance Analytics**: Track model performance over time

## Notes
- Ensure the Python backend services are running before starting the frontend.
- All 19 model services must be running on their respective ports for full functionality.
- The frontend uses WebSockets for real-time communication with the training systems.
- Configuration for model services ports can be found in `../config/model_services_config.json`
- All models support from-scratch training - no pre-trained weights are used in the system.
