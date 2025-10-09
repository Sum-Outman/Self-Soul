# Installation Guide

## System Requirements
- Windows 10/11 or Linux/macOS
- Python 3.9+
- Node.js 16+
- 8GB+ RAM
- 10GB available disk space

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/agi-system.git
cd agi-system
```

### 2. Set Up Python Virtual Environment
```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment
- Windows:
```bash
.venv\Scripts\activate
```
- Linux/macOS:
```bash
source .venv/bin/activate
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 5. Install Frontend Dependencies
```bash
cd app
npm install
```

### 6. Configure Environment Variables
Create a `.env` file and add the following content:
```
API_KEY=your-api-key-here
MODEL_PATH=./models
DATA_PATH=./data
```

### 7. Start the System
```bash
# Start backend service
python core/main.py

# Start frontend application in a new terminal
cd app && npm run dev
```

### 8. Access the System
Open a browser and visit: http://localhost:5175

## System Features
1. **Multi-language Interface**: Supports 5 languages switching
2. **Model Management**: Configure and monitor 11 specialized models
3. **Real-time Dashboard**: Monitor system performance
4. **Interactive Help**: Provide complete documentation
5. **Multi-modal Interaction**: Support text, voice and visual input

## Troubleshooting

### Virtual Environment Issues
If you encounter issues with the virtual environment, try:
```bash
python -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
python -m venv .venv
```

### Dependency Installation Issues
If pip installation fails, try:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Frontend Startup Issues
If npm run dev fails, try:
```bash
cd app
rm -rf node_modules
npm install
npm run dev
```

## System Architecture
```
Self Soul /
├── core/                     # Core system
│   ├── models/               # 11 core model implementations
│   │   ├── audio/            # Audio processing model
│   │   ├── computer/         # Computer control model
│   │   ├── knowledge/        # Knowledge base expert model
│   │   ├── language/         # Large language model
│   │   ├── manager/          # Manager model
│   │   ├── medical/          # Medical specialized model
│   │   ├── motion/           # Motion control model
│   │   ├── planning/         # Planning model
│   │   ├── prediction/       # Prediction model
│   │   ├── programming/      # Programming model
│   │   ├── sensor/           # Sensor perception model
│   │   ├── spatial/          # Spatial positioning model
│   │   ├── video/            # Video processing model
│   │   └── vision/           # Visual processing model
│   ├── training/             # Training system
│   │   ├── audio_training.py
│   │   ├── computer_training.py
│   │   ├── joint_training.py
│   │   ├── knowledge_training.py
│   │   ├── language_training.py
│   │   ├── manager_training.py
│   │   ├── motion_training.py
│   │   ├── programming_training.py
│   │   ├── sensor_training.py
│   │   ├── spatial_training.py
│   │   ├── video_training.py
│   │   └── vision_training.py
│   ├── collaboration/        # Model collaboration
│   ├── data/                 # Data processing
│   ├── fusion/               # Multi-modal fusion
│   ├── knowledge/            # Knowledge enhancement
│   ├── optimization/         # Model optimization
│   ├── realtime/             # Real-time processing
│   ├── agi_coordinator.py    # AGI coordinator
│   ├── api_config_manager.py # API configuration management
│   ├── data_processor.py     # Data processor
│   ├── emotion_awareness.py  # Emotion awareness
│   ├── error_handling.py     # Error handling
│   ├── main.py               # Main entry point
│   ├── model_registry.py     # Model registry
│   ├── monitoring.py         # System monitoring
│   ├── self_learning.py      # Self-learning
│   └── training_manager.py   # Training manager
├── app/                      # Frontend Vue application
│   ├── src/
│   │   ├── components/       # Components
│   │   ├── locales/          # Multi-language files
│   │   ├── models/           # Frontend models
│   │   ├── plugins/          # Plugins
│   │   ├── router/           # Router
│   │   ├── views/            # View pages
│   │   ├── App.vue           # Main application
│   │   └── main.js           # Entry point
│   ├── public/               # Static resources
│   ├── package.json          # Dependency configuration
│   └── vite.config.js        # Vite configuration
├── config/                   # Configuration files
│   └── languages/            # Multi-language configuration
├── data/                     # Data files
│   └── knowledge/            # Knowledge base data
├── docs/                     # Documentation
├── logs/                     # System logs
├── tests/                    # Test suite
├── tools/                    # Tool scripts
└── requirements.txt          # Python dependencies
```

## Technical Support
For any questions, please contact: silencecrowtom@qq.com
