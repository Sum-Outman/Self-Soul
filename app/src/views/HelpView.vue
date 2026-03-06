<template>
  <div class="help-container">
    <!-- Search Bar -->
        <div class="search-container">
          <input
            ref="searchInput"
            v-model="searchQuery"
            type="text"
            placeholder="Search help content..."
            class="search-input"
          />
          
          <!-- Search Results -->
          <div v-show="showSearchResults && searchResults.length > 0" class="search-results">
            <h3>Search Results ({{ searchResults.length }})</h3>
            <ul>
              <li v-for="result in searchResults" :key="result.id">
                <a :href="'#' + result.id" @click.prevent="scrollToSection(result.id)">{{ result.title }}</a>
              </li>
            </ul>
          </div>
          
          <!-- No Results Message -->
          <div v-show="showSearchResults && searchResults.length === 0" class="no-results">
            <p>No results found for "{{ searchQuery }}"</p>
          </div>
        </div>
        
        <!-- Section Controls -->
        <div class="section-controls">
          <button @click="expandAll()" class="control-btn">Expand All</button>
          <button @click="collapseAll()" class="control-btn">Collapse All</button>
        </div>

    <!-- Help Layout -->
    <div class="help-layout">
      <!-- Sidebar Navigation -->
      <aside class="help-sidebar">
        <nav class="help-nav">
          <h3>Contents</h3>
          <ul>
            <li><a href="#system-overview" @click.prevent="scrollToSection('system-overview')">System Overview</a></li>
            <li><a href="#getting-started" @click.prevent="scrollToSection('getting-started')">Getting Started</a></li>
            <li><a href="#core-models" @click.prevent="scrollToSection('core-models')">Core Cognitive Models</a></li>
            <li><a href="#model-detailed-documentation" @click.prevent="scrollToSection('model-detailed-documentation')">Model Detailed Documentation</a></li>
            <li><a href="#training-methodology" @click.prevent="scrollToSection('training-methodology')">Training Methodology</a></li>
            <li><a href="#page-features" @click.prevent="scrollToSection('page-features')">Page Features Documentation</a></li>
            <li><a href="#robot-settings" @click.prevent="scrollToSection('robot-settings')">Robot Settings</a></li>
            <li><a href="#robot-training" @click.prevent="scrollToSection('robot-training')">Robot Training</a></li>
            <li><a href="#spatial-recognition" @click.prevent="scrollToSection('spatial-recognition')">Real-time Spatial Recognition & Robot Control</a></li>
            <li><a href="#voice-recognition" @click.prevent="scrollToSection('voice-recognition')">Voice Recognition & Video Dialogue</a></li>
            <li><a href="#cleanup-system" @click.prevent="scrollToSection('cleanup-system')">Cleanup System</a></li>
            <li><a href="#docker-deployment" @click.prevent="scrollToSection('docker-deployment')">Docker Deployment</a></li>

            <li><a href="#ports-config" @click.prevent="scrollToSection('ports-config')">Service Ports Configuration</a></li>
            <li><a href="#advanced-capabilities" @click.prevent="scrollToSection('advanced-capabilities')">Advanced Capabilities</a></li>
            <li><a href="#advanced-configuration" @click.prevent="scrollToSection('advanced-configuration')">Advanced Configuration & Usage Examples</a></li>
            <li><a href="#api-documentation" @click.prevent="scrollToSection('api-documentation')">API Documentation Guide</a></li>
            <li><a href="#troubleshooting" @click.prevent="scrollToSection('troubleshooting')">Troubleshooting & Support</a></li>
            <li><a href="#system-requirements" @click.prevent="scrollToSection('system-requirements')">System Requirements</a></li>
          </ul>
        </nav>
      </aside>

      <!-- Main Content -->
      <main class="help-main">
      <!-- System Introduction -->
      <section id="system-overview" class="help-section">
        <div class="section-header" @click="toggleSection('systemOverview')">
          <h2>System Overview</h2>
          <span class="toggle-icon">{{ sectionExpanded.systemOverview ? '−' : '+' }}</span>
        </div>
        <div v-show="sectionExpanded.systemOverview" class="section-content">
        <p>Self Soul is a revolutionary human-like AGI system designed for autonomous learning, self-optimization, and multimodal intelligence. The system features a sophisticated architecture that integrates multiple cognitive capabilities including language processing, visual recognition, audio analysis, sensor data interpretation, and autonomous decision-making.</p>
        
        <div class="architecture-overview">
          <h3>Architecture Highlights</h3>
          <p>The system is built on a <strong>Unified Cognitive Architecture</strong> that integrates 27 specialized models working in concert to provide comprehensive AGI capabilities. Each model is assigned a dedicated port within the range 8001-8027, enabling distributed parallel processing while maintaining coordinated operation through the Manager Model (Port 8001).</p>
          <p>Key components include the <strong>Adaptive Learning Engine</strong> that optimizes training parameters based on model performance and data characteristics, a <strong>Dual ID System</strong> that ensures seamless communication between frontend (letter IDs A-X) and backend (string IDs like manager, language), and an <strong>Integrated Robot System</strong> that provides hardware control, sensor management, and robot-specific training capabilities.</p>
        </div>
        
        <div class="feature-list">
          <div v-for="feature in features" :key="feature.id" class="feature-item">
            <h3>{{ feature.title }}</h3>
            <p>{{ feature.description }}</p>
          </div>
        </div>
        
        <div class="system-technology">
          <h3>Technology Stack</h3>
          <div class="tech-stack">
            <div class="tech-item">
              <h4>Frontend</h4>
              <p>Vue 3 + Vite for the training interface and help page, using reactive variables (ref, reactive) for state management</p>
            </div>
            <div class="tech-item">
              <h4>Backend</h4>
              <p>FastAPI providing RESTful API endpoints for model management, training, monitoring, and dataset operations, with interactive documentation at {{ backendDocsUrl }}</p>
            </div>
            <div class="tech-item">
              <h4>Communication</h4>
              <p>WebSocket for real-time updates, enabling live training progress, system monitoring, and multimodal stream processing</p>
            </div>
          </div>
        </div>
        
        <!-- Real-time System Status -->
        <div class="real-time-status">
          <h3>Real-time System Status</h3>
          <div v-if="systemStatus.loading" class="status-loading">
            <div class="loading-spinner"></div>
            <p>Checking system status...</p>
          </div>
          
          <div v-else-if="systemStatus.error" class="status-error">
            <p>Failed to fetch system status: {{ systemStatus.error }}</p>
            <button @click="checkSystemStatus" class="retry-btn">Retry</button>
          </div>
          
          <div v-else-if="systemStatus.data" class="status-success">
            <div class="status-header">
              <h4>System Health: <span :class="systemStatus.data.available ? 'status-online' : 'status-offline'">
                {{ systemStatus.data.available ? 'Online' : 'Offline' }}
              </span></h4>
              <button @click="checkSystemStatus" class="refresh-btn">Refresh</button>
            </div>
            
            <div class="status-details">
              <div class="status-item">
                <span class="status-label">Last Updated:</span>
                <span class="status-value">{{ systemStatus.lastUpdated ? new Date(systemStatus.lastUpdated).toLocaleString() : 'N/A' }}</span>
              </div>
              
              <div v-if="systemStatus.data.services" class="status-item">
                <span class="status-label">Active Services:</span>
                <span class="status-value">{{ systemStatus.data.services.join(', ') }}</span>
              </div>
              
              <div v-if="systemStatus.data.models" class="status-item">
                <span class="status-label">Active Models:</span>
                <span class="status-value">{{ systemStatus.data.models.length }}</span>
              </div>
              
              <div v-if="systemStatus.data.uptime" class="status-item">
                <span class="status-label">System Uptime:</span>
                <span class="status-value">{{ systemStatus.data.uptime }}</span>
              </div>
              
              <div v-if="systemStatus.data.timestamp" class="status-item">
                <span class="status-label">Server Time:</span>
                <span class="status-value">{{ new Date(systemStatus.data.timestamp).toLocaleString() }}</span>
              </div>
            </div>
            
            <div v-if="systemStatus.data.details" class="status-extra">
              <h5>Detailed Information:</h5>
              <pre class="status-json">{{ formatJson(systemStatus.data.details) }}</pre>
            </div>
          </div>
          
          <div v-else class="status-empty">
            <p>No system status data available.</p>
            <button @click="checkSystemStatus" class="check-btn">Check Status</button>
          </div>
        </div>
        <div class="training-info">
          <h3>Intelligent Dataset Selection</h3>
          <p>The Self Soul system features an intelligent dataset selection mechanism that automatically recommends the most suitable datasets based on model capabilities and training objectives.</p>
          
          <div class="info-box">
            <h4>Selection Criteria</h4>
            <p>The system evaluates datasets based on multiple factors:</p>
            <ul>
              <li><strong>Model Compatibility:</strong> Matches dataset supported models with the selected training models</li>
              <li><strong>Data Modality Alignment:</strong> Ensures dataset contains appropriate data types (text, image, audio) for the selected models</li>
              <li><strong>Dataset Quality:</strong> Evaluates data completeness, consistency, and labeling accuracy</li>
              <li><strong>Training Objective Alignment:</strong> Selects datasets that best match the specific training goals (e.g., fine-tuning, from-scratch training, domain adaptation)</li>
              <li><strong>Performance History:</strong> Considers historical training performance with similar dataset-model combinations</li>
            </ul>
            
            <h4>Automated Recommendations</h4>
            <p>When users select models for training, the system automatically:</p>
            <ol>
              <li>Filters available datasets to only show those compatible with the selected models</li>
              <li>Ranks datasets based on relevance to the training strategy and objectives</li>
              <li>Highlights the optimal dataset for the current training configuration</li>
              <li>Provides explanations for why specific datasets are recommended</li>
              <li>Offers alternatives if the primary recommendation is unavailable</li>
            </ol>
            
            <h4>Benefits</h4>
            <ul>
              <li><strong>Reduced Configuration Time:</strong> Automatically eliminates incompatible dataset options</li>
              <li><strong>Improved Training Outcomes:</strong> Ensures optimal dataset-model alignment for better results</li>
              <li><strong>Error Prevention:</strong> Prevents training failures due to dataset-model incompatibility</li>
              <li><strong>Enhanced User Experience:</strong> Simplified interface with intelligent defaults</li>
              <li><strong>Adaptive Learning:</strong> System learns from training outcomes to improve future recommendations</li>
            </ul>
          </div>
          </div>
        </div>
      </section>

      <!-- File Architecture Diagram -->
      <section id="file-architecture" class="help-section">
        <div class="section-header" @click="toggleSection('fileArchitecture')">
          <h2>File Architecture Diagram</h2>
          <span class="toggle-icon">{{ sectionExpanded.fileArchitecture ? '−' : '+' }}</span>
        </div>
        <div v-show="sectionExpanded.fileArchitecture" class="section-content">
          <p>The Self Soul system follows a well-organized directory structure that separates core logic, data, configuration, and deployment components. Below is the simplified file architecture:</p>
          
          <div class="architecture-diagram">
            <h3>Project Structure</h3>
            <pre class="code-block">
Self-Soul-B/
├── app/                          # Frontend Vue.js application
│   ├── src/                      # Source code
│   │   ├── views/                # Page components (Home, Training, etc.)
│   │   ├── components/           # Reusable UI components
│   │   ├── router/               # Vue Router configuration
│   │   └── utils/                # Utility functions
│   ├── public/                   # Static assets
│   └── package.json              # Frontend dependencies
├── core/                         # Backend Python core
│   ├── models/                   # All 27 AI model implementations
│   │   ├── manager/              # Manager model (port 8001)
│   │   ├── language/             # Language model (port 8002)
│   │   ├── vision/               # Vision model (port 8004)
│   │   ├── audio/                # Audio model (port 8005)
│   │   ├── knowledge/            # Knowledge model (port 8003)
│   │   ├── autonomous/           # Autonomous model (port 8006)
│   │   ├── programming/          # Programming model (port 8007)
│   │   ├── planning/             # Planning model (port 8008)
│   │   ├── emotion/              # Emotion model (port 8009)
│   │   ├── spatial/              # Spatial model (port 8010)
│   │   ├── computer_vision/      # Computer vision model (port 8011)
│   │   ├── sensor/               # Sensor model (port 8012)
│   │   ├── motion/               # Motion model (port 8013)
│   │   ├── prediction/           # Prediction model (port 8014)
│   │   ├── advanced_reasoning/   # Advanced reasoning model (port 8015)
│   │   ├── data_fusion/          # Data fusion model (port 8028)
│   │   ├── creative_problem_solving/ # Creative problem solving model (port 8017)
│   │   ├── metacognition/        # Meta cognition model (port 8018)
│   │   ├── value_alignment/      # Value alignment model (port 8019)
│   │   ├── vision_image/         # Vision image model (port 8020)
│   │   ├── vision_video/         # Vision video model (port 8021)
│   │   ├── finance/              # Finance model (port 8022)
│   │   ├── medical/              # Medical model (port 8023)
│   │   ├── collaboration/        # Collaboration model (port 8024)
│   │   ├── optimization/         # Optimization model (port 8025)
│   │   ├── computer/             # Computer model (port 8026)
│   │   └── mathematics/          # Mathematics model (port 8027)
│   ├── hardware/                 # Hardware interface layer
│   ├── knowledge/                # Knowledge management
│   ├── database/                 # Database access layer
│   ├── api_client_factory.py     # API client management
│   ├── api_dependency_manager.py # API dependency management
│   ├── config_manager.py         # Configuration management
│   ├── emotion_awareness.py      # Emotion awareness system
│   ├── error_handling_api.py     # Error handling
│   ├── external_api_service.py   # External API integration
│   ├── main.py                   # Main entry point
│   └── model_management_api.py   # Model management API
├── config/                       # Configuration files
│   ├── model_services_config.json # Model port configuration
│   └── performance.yml           # Performance tuning
├── data/                         # Training and operational data
├── data_cache/                   # Cached data for faster access
├── deployment/                   # Deployment scripts and configs
├── docs/                         # Documentation
├── logs/                         # System logs
├── monitoring/                   # Monitoring tools and dashboards
├── results/                      # Training results and model outputs
├── scripts/                      # Utility scripts
├── tests/                        # Test suites
├── training_data/                # Training datasets
├── training_data_super_large/    # Large-scale training datasets
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Python project configuration
├── Dockerfile                    # Containerization
├── docker-compose.yml            # Multi-service deployment
└── README.md                     # Project documentation</pre>
            
            <h3>Key Directories Explained</h3>
            <div class="directory-explanation">
              <div class="dir-item">
                <h4><code>core/models/</code></h4>
                <p>Contains all 27 AI model implementations, each in its own subdirectory. Each model follows a unified interface pattern for consistent integration.</p>
              </div>
              <div class="dir-item">
                <h4><code>core/hardware/</code></h4>
                <p>Hardware abstraction layer for camera control, sensor integration, robotic interfaces, and external device communication.</p>
              </div>
              <div class="dir-item">
                <h4><code>core/knowledge/</code></h4>
                <p>Knowledge base management, including document processing, semantic search, and knowledge graph operations.</p>
              </div>
              <div class="dir-item">
                <h4><code>app/src/views/</code></h4>
                <p>Frontend page components corresponding to each major system function (Home, Training, Conversation, Knowledge, Settings, Help).</p>
              </div>
              <div class="dir-item">
                <h4><code>config/</code></h4>
                <p>Centralized configuration management with JSON and YAML files for model ports, performance settings, and system parameters.</p>
              </div>
            </div>
            
            <div class="beginner-tip">
              <h4>For Beginners: Navigating the File Structure</h4>
              <p>If you're new to the project, focus on these key files:</p>
              <ol>
                <li><strong>Start the system</strong>: Run <code>python core/main.py</code> and <code>cd app && npm run dev</code></li>
                <li><strong>Configure models</strong>: Edit <code>config/model_services_config.json</code> for port settings</li>
                <li><strong>Add new models</strong>: Create new directories in <code>core/models/</code> following existing patterns</li>
                <li><strong>Modify frontend</strong>: Edit Vue components in <code>app/src/views/</code> for UI changes</li>
                <li><strong>View logs</strong>: Check <code>logs/</code> directory for debugging information</li>
              </ol>
            </div>
          </div>
        </div>
      </section>

      <!-- Logical Architecture Diagram -->
      <section id="logical-architecture" class="help-section">
        <div class="section-header" @click="toggleSection('logicalArchitecture')">
          <h2>Logical Architecture Diagram</h2>
          <span class="toggle-icon">{{ sectionExpanded.logicalArchitecture ? '−' : '+' }}</span>
        </div>
        <div v-show="sectionExpanded.logicalArchitecture" class="section-content">
          <p>The Self Soul system employs a sophisticated multi-layer architecture that separates concerns while enabling seamless integration between components:</p>
          
          <div class="logical-architecture">
            <h3>Four-Tier Logical Architecture</h3>
            <pre class="code-block">
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Presentation Layer (Vue.js)                         │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  Frontend Application (Port 5175)                                   │  │
│  │  ├─ Home View: Device management and system monitoring              │  │
│  │  ├─ Conversation View: Multimodal chat interface                    │  │
│  │  ├─ Training View: Model training and dataset management            │  │
│  │  ├─ Knowledge View: Knowledge base import and browsing              │  │
│  │  ├─ Settings View: System configuration and model control           │  │
│  │  └─ Help View: This comprehensive documentation                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      API Gateway Layer (FastAPI)                     │
│  │  Main API Gateway (Port 8000)                                       │  │
│  │  ├─ RESTful API endpoints for all system operations                 │  │
│  │  ├─ WebSocket server for real-time communication                    │  │
│  │  ├─ Authentication and authorization middleware                     │  │
│  │  ├─ Request validation and rate limiting                            │  │
│  │  └─ Swagger UI documentation at /docs                               │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      Service Layer (Python Core)                    │
│  │  Core Service Manager                                               │  │
│  │  ├─ Model Service Manager: Creates/manages 27 AI model instances   │  │
│  │  ├─ Training Manager: Coordinates all training activities           │  │
│  │  ├─ Knowledge Manager: Handles knowledge base operations           │  │
│  │  ├─ Device Manager: Controls cameras, sensors, and hardware        │  │
│  │  ├─ API Dependency Manager: Manages external API connections       │  │
│  │  └─ Error Handling System: Centralized error management            │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      Model Layer (27 AI Models)                     │
│  │  Distributed Model Architecture (Ports 8001-8027)                   │  │
│  │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐           │  │
│  │  │8001 │8002 │8003 │8004 │8005 │8006 │8007 │8008 │8009 │           │  │
│  │  │Mngr │Lang │Know │Vis  │Audio│Auto │Prog │Plan │Emot │           │  │
│  │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘           │  │
│  │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐           │  │
│  │  │8010 │8011 │8012 │8013 │8014 │8015 │8016 │8017 │8018 │           │  │
│  │  │Spat│CV   │Sensor│Motion│Pred │AdvR │DataF│Creat│Meta │           │  │
│  │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘           │  │
│  │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐           │  │
│  │  │8019 │8020 │8021 │8022 │8023 │8024 │8025 │8026 │8027 │           │  │
│  │  │Value│Img  │Video│Fin  │Med  │Collab│Optim│Comp │Math │           │  │
│  │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘           │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      Data Layer                                     │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  Multimodal Dataset v1: Training data for all models        │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  Knowledge Base: Structured and unstructured knowledge      │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  Model Checkpoints: Saved model states                      │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘</pre>
            
            <h3>Communication Flow</h3>
            <div class="communication-flow">
              <div class="flow-step">
                <span class="step-number">1</span>
                <div class="step-content">
                  <h4>User Request</h4>
                  <p>User interacts with Vue.js frontend (port 5175), which sends HTTP/WebSocket requests to the API Gateway (port 8000).</p>
                </div>
              </div>
              <div class="flow-step">
                <span class="step-number">2</span>
                <div class="step-content">
                  <h4>API Routing</h4>
                  <p>API Gateway validates and routes requests to appropriate service layer components based on endpoint and parameters.</p>
                </div>
              </div>
              <div class="flow-step">
                <span class="step-number">3</span>
                <div class="step-content">
                  <h4>Service Processing</h4>
                  <p>Service layer components (Model Service Manager, Training Manager, etc.) process requests and coordinate with relevant AI models.</p>
                </div>
              </div>
              <div class="flow-step">
                <span class="step-number">4</span>
                <div class="step-content">
                  <h4>Model Execution</h4>
                  <p>AI models on dedicated ports (8001-8027) execute specialized tasks, with the Manager Model (8001) coordinating complex multi-model operations.</p>
                </div>
              </div>
              <div class="flow-step">
                <span class="step-number">5</span>
                <div class="step-content">
                  <h4>Data Access</h4>
                  <p>Models access training data, knowledge base, or checkpoints as needed through the data layer.</p>
                </div>
              </div>
              <div class="flow-step">
                <span class="step-number">6</span>
                <div class="step-content">
                  <h4>Response Return</h4>
                  <p>Results flow back through the same layers to the frontend, with WebSocket providing real-time updates for long-running operations.</p>
                </div>
              </div>
            </div>
            
            <div class="architecture-benefits">
              <h3>Architecture Benefits</h3>
              <div class="benefits-grid">
                <div class="benefit-item">
                  <h4>Separation of Concerns</h4>
                  <p>Each layer has distinct responsibilities, making the system easier to understand, maintain, and extend.</p>
                </div>
                <div class="benefit-item">
                  <h4>Scalability</h4>
                  <p>Distributed model architecture allows individual models to scale independently based on demand.</p>
                </div>
                <div class="benefit-item">
                  <h4>Fault Isolation</h4>
                  <p>Problems in one layer or model don't cascade to others, improving overall system reliability.</p>
                </div>
                <div class="benefit-item">
                  <h4>Technology Flexibility</h4>
                  <p>Frontend (Vue.js), backend (Python/FastAPI), and models can use different technologies best suited to their tasks.</p>
                </div>
                <div class="benefit-item">
                  <h4>Real-time Capabilities</h4>
                  <p>WebSocket integration enables real-time updates for training progress, device control, and chat interactions.</p>
                </div>
                <div class="benefit-item">
                  <h4>Multi-modal Integration</h4>
                  <p>Unified architecture seamlessly combines text, image, audio, and video processing across all models.</p>
                </div>
              </div>
            </div>
            
            <div class="beginner-tip">
              <h4>For Beginners: Understanding the Architecture</h4>
              <p>Think of the architecture like a modern restaurant:</p>
              <ul>
                <li><strong>Presentation Layer</strong>: Dining area where customers (users) interact with waitstaff (UI)</li>
                <li><strong>API Gateway</strong>: Host station that routes orders to appropriate kitchen stations</li>
                <li><strong>Service Layer</strong>: Kitchen manager coordinating different cooking stations</li>
                <li><strong>Model Layer</strong>: Specialized cooking stations (grill, salad, dessert, etc.)</li>
                <li><strong>Data Layer</strong>: Pantry and storage with ingredients and recipes</li>
              </ul>
              <p>Each customer request goes through this organized flow, ensuring efficient and high-quality results.</p>
            </div>
          </div>
        </div>
      </section>

      <!-- Service Ports Configuration -->
      <section id="ports-config" class="help-section">
        <div class="section-header" @click="toggleSection('portsConfig')">
          <h2>Service Ports Configuration</h2>
          <span class="toggle-icon">{{ sectionExpanded.portsConfig ? '−' : '+' }}</span>
        </div>
        <div v-show="sectionExpanded.portsConfig" class="section-content">
        <p>The system uses a multi-port architecture to separate different services and model endpoints. Here are the primary service ports:</p>
        
        <div class="ports-table">
          <table>
            <thead>
              <tr>
                <th>Service</th>
                <th>Port</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Main API Gateway</td>
                <td>8000</td>
                <td>System's primary entry point, providing RESTful API interface</td>
              </tr>
              <tr>
                <td>Frontend Application</td>
                <td>5175</td>
                <td>User interface accessible via web browser</td>
              </tr>
              <tr>
                <td>Realtime Stream Manager</td>
                <td>8766</td>
                <td>Manages real-time data streams and inter-model communication</td>
              </tr>
              <tr>
                <td>Performance Monitoring</td>
                <td>8080</td>
                <td>Monitors system performance and resource usage</td>
              </tr>
            </tbody>
          </table>
        </div>
        </div>
      </section>

      <!-- Quick Start -->
      <section id="getting-started" class="help-section">
        <div class="section-header" @click="toggleSection('gettingStarted')">
          <h2>Getting Started</h2>
          <span class="toggle-icon">{{ sectionExpanded.gettingStarted ? '−' : '+' }}</span>
        </div>
        <div v-show="sectionExpanded.gettingStarted" class="section-content">
        <div class="newbie-guide">
          <h3>Quick Start Guide (5 Minutes to First Use)</h3>
          <p>If you're using the Self Soul AGI system for the first time, follow these steps to get started quickly:</p>
          
          <div class="step-list">
            <div class="step">
              <span class="step-number">1</span>
              <div class="step-content">
                <h4>Access the System Interface</h4>
                <p>Open your browser and go to <a :href="frontendUrl" target="_blank">{{ frontendUrl }}</a>. Ensure the backend service is running (port {{ $getConfig ? $getConfig('system.backendPort', 8000) : 8000 }}). If you see a login screen or the main page, the system is ready.</p>
              </div>
            </div>
            <div class="step">
              <span class="step-number">2</span>
              <div class="step-content">
                <h4>Explore the Home Page</h4>
                <p>Go to the <strong>Home</strong> page to check device status. This is where you manage cameras and other hardware devices. You can familiarize yourself with the interface layout without immediately configuring devices.</p>
              </div>
            </div>
            <div class="step">
              <span class="step-number">3</span>
              <div class="step-content">
                <h4>Start Your First Conversation</h4>
                <p>Click <strong>Conversation</strong> in the left navigation bar to enter the chat interface. Type "Hello" or any question in the input box and press Enter. The system will respond via the Manager Model (port 8001).</p>
                <p class="tip">💡 Tip: This is the most direct way to experience the system's intelligence.</p>
              </div>
            </div>
            <div class="step">
              <span class="step-number">4</span>
              <div class="step-content">
                <h4>Check System Settings</h4>
                <p>Go to the <strong>Settings</strong> page to view model status and API configuration. Ensure all core models (A-X) show "Running" or "Ready". If any model is not running, you can try starting it.</p>
              </div>
            </div>
            <div class="step">
              <span class="step-number">5</span>
              <div class="step-content">
                <h4>Try Simple Training</h4>
                <p>Go to the <strong>Training</strong> page, select model "B (Language Model)", choose dataset "Multimodal Dataset v1", select training strategy "Standard Training", then click "Start Training". Observe the training process.</p>
                <p class="tip">💡 Tip: The first training may take some time; you can stop it anytime.</p>
              </div>
            </div>
            <div class="step">
              <span class="step-number">6</span>
              <div class="step-content">
                <h4>Get Help</h4>
                <p>If you encounter any issues, return to this help page (where you are now!). Use the search function or navigation menu to find specific topics.</p>
              </div>
            </div>
          </div>
          
          <div class="quick-tasks">
            <h4>Common Quick Tasks</h4>
            <ul>
              <li><strong>Just want to chat?</strong> Go directly to the Conversation page to start a dialogue.</li>
              <li><strong>Want to understand system capabilities?</strong> Check the <a href="#core-models" @click.prevent="scrollToSection('core-models')">Core Cognitive Models</a> section.</li>
              <li><strong>Encountering issues?</strong> Check the <a href="#troubleshooting" @click.prevent="scrollToSection('troubleshooting')">Troubleshooting & Support</a> section.</li>
              <li><strong>Want to learn more deeply?</strong> Continue reading the detailed guide below.</li>
            </ul>
          </div>
        </div>
        
        <div class="original-steps">
          <h3>Detailed System Initialization Steps</h3>
          <div class="step-list">
            <div class="step">
              <span class="step-number">1</span>
              <div class="step-content">
                <h3>System Initialization</h3>
                <p>Execute the main startup script to initialize the AGI core. Ensure all dependencies are installed and system requirements are met.</p>
              </div>
            </div>
            <div class="step">
              <span class="step-number">2</span>
              <div class="step-content">
                <h3>Model Configuration</h3>
                <p>Access the settings panel to configure local models or connect to external API services. Set appropriate parameters for each model type.</p>
              </div>
            </div>
            <div class="step">
              <span class="step-number">3</span>
              <div class="step-content">
                <h3>Interaction Setup</h3>
                <p>Configure input/output channels and establish communication protocols for multimodal interactions with the system.</p>
              </div>
            </div>
            <div class="step">
              <span class="step-number">4</span>
              <div class="step-content">
                <h3>Begin Operation</h3>
                <p>Start engaging with the system through the provided interface. The AGI will begin learning and adapting to your usage patterns.</p>
              </div>
            </div>
          </div>
        </div>
        </div>
      </section>

      <!-- Core Models -->
      <section id="core-models" class="help-section">
        <div class="section-header" @click="toggleSection('coreModels')">
          <h2>Core Cognitive Models</h2>
          <span class="toggle-icon">{{ sectionExpanded.coreModels ? '−' : '+' }}</span>
        </div>
        <div v-show="sectionExpanded.coreModels" class="section-content">
        <p>The system comprises 27 specialized models that work in concert to provide comprehensive AGI capabilities. Each model is assigned a dedicated port within the range 8001-8027:</p>
        
        <div class="beginner-tip">
          <h4>Simple Explanation for Beginners</h4>
          <p>Think of these 27 models as an intelligent team: the Manager Model is the team leader, the Language Model handles conversations, the Vision Model processes images, the Audio Model handles sound, and other models have their own specialties. They communicate through different ports (8001-8027), much like a team meeting in different conference rooms.</p>
        </div>
        
        <div class="model-grid">
          <div class="model-card">
            <h3>Manager Model (Port 8001)</h3>
            <p><strong>Core Coordination & Task Management:</strong> Acts as the central brain coordinating all 18 other models. Manages task delegation, resource allocation, load balancing, and system-wide decision making. Routes user requests to appropriate specialized models.</p>
            <p class="tip"><small>💡 Beginner Tip: This is your primary interface - talk to this model in the Conversation page for general queries and complex tasks.</small></p>
          </div>
          <div class="model-card">
            <h3>Language Model (Port 8002)</h3>
            <p><strong>Natural Language Understanding & Generation:</strong> Processes text inputs for understanding, generation, translation, summarization, sentiment analysis, and contextual dialogue across multiple languages. Powers all text-based interactions in the system.</p>
            <p class="tip"><small>💡 Beginner Tip: Handles all your text conversations. Try the Chat From Scratch page for direct language-focused interactions.</small></p>
          </div>
          <div class="model-card">
            <h3>Knowledge Model (Port 8003)</h3>
            <p><strong>Knowledge Storage & Retrieval:</strong> Manages the system's knowledge base with semantic search, fact-checking, knowledge graph navigation, and information retrieval. Stores both structured and unstructured knowledge from various sources.</p>
            <p class="tip"><small>💡 Beginner Tip: Import documents (PDF, DOCX, TXT) in the Knowledge page to expand what the system knows.</small></p>
          </div>
          <div class="model-card">
            <h3>Vision Model (Port 8004)</h3>
            <p><strong>Image Processing & Analysis:</strong> Performs object detection, scene understanding, facial recognition, optical character recognition (OCR), and visual pattern analysis on still images. Processes photos, screenshots, and document images.</p>
            <p class="tip"><small>💡 Beginner Tip: Connect cameras in the Home page or upload images for visual analysis.</small></p>
          </div>
          <div class="model-card">
            <h3>Audio Model (Port 8005)</h3>
            <p><strong>Audio Processing & Speech:</strong> Handles speech-to-text transcription, text-to-speech synthesis, sound classification, audio sentiment analysis, and acoustic environment understanding. Supports real-time voice conversations.</p>
            <p class="tip"><small>💡 Beginner Tip: Enable microphone access for voice conversations or upload audio files for analysis.</small></p>
          </div>
          <div class="model-card">
            <h3>Autonomous Model (Port 8006)</h3>
            <p><strong>Self-Governance & Decision Making:</strong> Enables the system to operate independently with goal-setting, task planning, adaptive learning, and self-optimization without constant human intervention. Manages long-term objectives.</p>
            <p class="tip"><small>💡 Beginner Tip: Enable autonomous training in Training page for continuous self-improvement.</small></p>
          </div>
          <div class="model-card">
            <h3>Programming Model (Port 8007)</h3>
            <p><strong>Code Generation & Software Development:</strong> Writes, debugs, and executes code in multiple languages (Python, JavaScript, etc.). Designs software architectures, implements algorithms, and performs automated testing.</p>
            <p class="tip"><small>💡 Beginner Tip: Ask the system to write code for you or help debug existing code in the Conversation interface.</small></p>
          </div>
          <div class="model-card">
            <h3>Planning Model (Port 8008)</h3>
            <p><strong>Strategic Planning & Task Execution:</strong> Develops multi-step plans, optimizes resource allocation, manages project timelines, and coordinates complex task sequences. Essential for goal-oriented problem solving.</p>
            <p class="tip"><small>💡 Beginner Tip: Use for complex projects requiring detailed step-by-step planning and coordination.</small></p>
          </div>
          <div class="model-card">
            <h3>Emotion Model (Port 8009)</h3>
            <p><strong>Emotional Intelligence & Response:</strong> Analyzes emotional content in text, voice, and facial expressions. Provides emotionally appropriate responses, empathy simulation, and mood-adaptive interactions.</p>
            <p class="tip"><small>💡 Beginner Tip: Makes conversations feel more natural and human-like by understanding emotional context.</small></p>
          </div>
          <div class="model-card">
            <h3>Spatial Model (Port 8010)</h3>
            <p><strong>Spatial Reasoning & Navigation:</strong> Understands 3D environments, performs object localization, environmental mapping, geometric reasoning, and spatial relationship analysis. Essential for robotics and AR/VR applications.</p>
            <p class="tip"><small>💡 Beginner Tip: Important for applications requiring physical space understanding or robotic navigation.</small></p>
          </div>
          <div class="model-card">
            <h3>Computer Vision Model (Port 8011)</h3>
            <p><strong>Advanced Visual Processing:</strong> Specializes in real-time video analysis, motion tracking, depth perception, optical flow calculation, and complex visual recognition tasks beyond static image analysis.</p>
            <p class="tip"><small>💡 Beginner Tip: Processes video streams from cameras for continuous visual monitoring and analysis.</small></p>
          </div>
          <div class="model-card">
            <h3>Sensor Model (Port 8012)</h3>
            <p><strong>Sensor Data Fusion:</strong> Integrates data from various sensors (temperature, motion, pressure, biometrics, IoT devices) for environmental monitoring, anomaly detection, and multi-sensory perception.</p>
            <p class="tip"><small>💡 Beginner Tip: Combines data from different sensors for comprehensive environmental understanding.</small></p>
          </div>
          <div class="model-card">
            <h3>Motion Model (Port 8013)</h3>
            <p><strong>Movement Planning & Control:</strong> Plans and executes precise movements for robotic systems, manages actuator control, calculates kinematics, and optimizes motion trajectories for efficiency and safety.</p>
            <p class="tip"><small>💡 Beginner Tip: Controls physical devices, robots, or automated systems requiring precise movement.</small></p>
          </div>
          <div class="model-card">
            <h3>Prediction Model (Port 8014)</h3>
            <p><strong>Predictive Analytics & Forecasting:</strong> Performs statistical modeling, trend analysis, risk assessment, and probabilistic forecasting. Uses historical data to predict future outcomes and patterns.</p>
            <p class="tip"><small>💡 Beginner Tip: Use for data analysis, market predictions, weather forecasting, or any trend-based predictions.</small></p>
          </div>
          <div class="model-card">
            <h3>Advanced Reasoning Model (Port 8015)</h3>
            <p><strong>Complex Logical Reasoning:</strong> Performs sophisticated logical deduction, causal inference, abstract thinking, mathematical proof, and multi-step problem solving requiring deep analytical capabilities.</p>
            <p class="tip"><small>💡 Beginner Tip: Engages for challenging puzzles, scientific reasoning, complex decision making, and abstract problem solving.</small></p>
          </div>
          <div class="model-card">
            <h3>Data Fusion Model (Port 8028)</h3>
            <p><strong>Multi-Source Data Integration:</strong> Combines information from text, images, audio, sensor data, and other sources into unified representations. Aligns heterogeneous data formats for comprehensive analysis.</p>
            <p class="tip"><small>💡 Beginner Tip: Essential for tasks requiring combined understanding of different data types (e.g., understanding a scene with visual and auditory elements).</small></p>
          </div>
          <div class="model-card">
            <h3>Creative Problem Solving Model (Port 8017)</h3>
            <p><strong>Innovative Solution Generation:</strong> Applies lateral thinking, brainstorming techniques, design thinking methodologies, and out-of-the-box approaches to generate novel solutions to complex problems.</p>
            <p class="tip"><small>💡 Beginner Tip: Use for creative projects, innovation challenges, artistic endeavors, and unconventional problems requiring fresh perspectives.</small></p>
          </div>
          <div class="model-card">
            <h3>Meta Cognition Model (Port 8018)</h3>
            <p><strong>Self-Awareness & Cognitive Monitoring:</strong> Enables the system to reflect on its own thought processes, evaluate learning strategies, monitor cognitive performance, and optimize its own thinking methods.</p>
            <p class="tip"><small>💡 Beginner Tip: Allows the system to improve how it thinks and learns over time through self-reflection.</small></p>
          </div>
          <div class="model-card">
            <h3>Value Alignment Model (Port 8019)</h3>
            <p><strong>Ethical Decision Making & Safety:</strong> Ensures system behavior aligns with human values, ethical principles, and safety guidelines. Performs moral reasoning, value system alignment, and responsible AI governance.</p>
            <p class="tip"><small>💡 Beginner Tip: Provides an ethical safety layer ensuring all system outputs are responsible and value-aligned.</small></p>
          </div>
          <div class="model-card">
            <h3>Vision Image Model (Port 8020)</h3>
            <p><strong>Advanced Image Processing:</strong> Specializes in high-resolution image analysis, enhancement, and detailed visual pattern recognition beyond basic vision capabilities. Processes complex images with fine-grained details.</p>
            <p class="tip"><small>💡 Beginner Tip: Used for detailed image analysis tasks requiring high precision and visual fidelity.</small></p>
          </div>
          <div class="model-card">
            <h3>Vision Video Model (Port 8021)</h3>
            <p><strong>Video Stream Analysis:</strong> Processes continuous video streams for motion tracking, temporal pattern recognition, and real-time video understanding. Analyzes sequences of frames for dynamic scene comprehension.</p>
            <p class="tip"><small>💡 Beginner Tip: Essential for real-time video surveillance, action recognition, and temporal analysis tasks.</small></p>
          </div>
          <div class="model-card">
            <h3>Finance Model (Port 8022)</h3>
            <p><strong>Financial Analysis & Prediction:</strong> Specialized in financial data analysis, market trend prediction, risk assessment, and investment strategy optimization. Processes economic indicators and financial time series data.</p>
            <p class="tip"><small>💡 Beginner Tip: Use for financial forecasting, market analysis, and investment decision support.</small></p>
          </div>
          <div class="model-card">
            <h3>Medical Model (Port 8023)</h3>
            <p><strong>Medical Data Analysis:</strong> Processes medical images, patient records, and clinical data for diagnostic assistance, treatment recommendation, and health outcome prediction. Adheres to medical privacy standards.</p>
            <p class="tip"><small>💡 Beginner Tip: Supports medical professionals with diagnostic insights and patient data analysis.</small></p>
          </div>
          <div class="model-card">
            <h3>Collaboration Model (Port 8024)</h3>
            <p><strong>Multi-Agent Coordination:</strong> Facilitates collaborative problem-solving across multiple AI agents, coordinating information sharing, task allocation, and consensus building among specialized models.</p>
            <p class="tip"><small>💡 Beginner Tip: Enables complex tasks requiring coordinated efforts from multiple specialized models.</small></p>
          </div>
          <div class="model-card">
            <h3>Optimization Model (Port 8025)</h3>
            <p><strong>System Performance Optimization:</strong> Analyzes and optimizes system performance, resource allocation, and efficiency metrics. Identifies bottlenecks and recommends improvements for overall system enhancement.</p>
            <p class="tip"><small>💡 Beginner Tip: Continuously improves system performance through automated optimization algorithms.</small></p>
          </div>
          <div class="model-card">
            <h3>Computer Control Model (Port 8026)</h3>
            <p><strong>Computer System Management:</strong> Provides comprehensive control over operating systems, file management, process control, and system administration tasks across Windows, Linux, and macOS platforms.</p>
            <p class="tip"><small>💡 Beginner Tip: Automates system administration tasks and provides cross-platform computer control capabilities.</small></p>
          </div>
          <div class="model-card">
            <h3>Mathematics Model (Port 8027)</h3>
            <p><strong>Mathematical Reasoning & Computation:</strong> Performs advanced mathematical calculations, symbolic reasoning, theorem proving, and mathematical problem solving across various domains including algebra, calculus, and statistics.</p>
            <p class="tip"><small>💡 Beginner Tip: Use for complex mathematical calculations, symbolic math, and mathematical proof assistance.</small></p>
          </div>
        </div>
        
        <div class="simple-explanation">
          <h3>Simple Usage Guide</h3>
          <p>As a beginner, you mainly need to understand these core models:</p>
          <div class="model-simple-table">
            <table>
              <thead>
                <tr>
                  <th>Model Name</th>
                  <th>Port</th>
                  <th>Beginner Use</th>
                  <th>How to Access</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Manager Model (A)</td>
                  <td>8001</td>
                  <td>Main conversation interface, answers various questions</td>
                  <td>Conversation page</td>
                </tr>
                <tr>
                  <td>Language Model (B)</td>
                  <td>8002</td>
                  <td>Text dialogue and language understanding</td>
                  <td>Chat From Scratch page</td>
                </tr>
                <tr>
                  <td>Knowledge Model (J)</td>
                  <td>8003</td>
                  <td>Store and retrieve information</td>
                  <td>Knowledge page</td>
                </tr>
                <tr>
                  <td>Vision Model (D)</td>
                  <td>8004</td>
                  <td>Process images and camera feeds</td>
                  <td>Cameras on Home page</td>
                </tr>
                <tr>
                  <td>Training Models (All)</td>
                  <td>8001-8027</td>
                  <td>Improve model capabilities</td>
                  <td>Training page</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p class="tip">💡 <strong>Tip:</strong> You don't need to remember all models. Start with the Conversation page and gradually try other features. You can check the status of each model in the Settings page.</p>
        </div>
        
        <div class="training-info">
          <h3>External API Integration</h3>
          <p>All core cognitive models can be enhanced or replaced with external API providers for extended capabilities. The system supports seamless integration with multiple AI service providers:</p>
          
          <div class="info-box">
            <h4>Supported API Providers</h4>
            <ul>
              <li><strong>International Providers:</strong> OpenAI, Anthropic, Google AI, HuggingFace, Cohere, Mistral</li>
              <li><strong>Chinese Providers:</strong> DeepSeek, SiliconFlow, Zhipu AI, Baidu ERNIE, Alibaba Qwen, Moonshot, Yi, Tencent Hunyuan</li>
              <li><strong>Local Model Support:</strong> Ollama integration for running local LLMs with automatic configuration</li>
            </ul>
            
            <div class="beginner-tip">
              <h4>API Configuration Suggestions for Beginners</h4>
              <p>If you're configuring for the first time:</p>
              <ol>
                <li>Start by using local models (Ollama) for testing, no API key required</li>
                <li>If more powerful capabilities are needed, register for a free trial API service (e.g., DeepSeek)</li>
                <li>Configure in the Settings page under "API Settings"</li>
                <li>Start with simple tasks and gradually increase complexity</li>
              </ol>
            </div>
            
            <h4>Configuration</h4>
            <p>API providers can be configured in the Settings View under "Models > API Settings > API Provider API". Each model can be individually configured to use either local processing or external API services.</p>
            
            <h4>Benefits</h4>
            <ul>
              <li>Access to state-of-the-art AI models from multiple providers</li>
              <li>Flexibility to switch between local and cloud-based processing</li>
              <li>Cost optimization by selecting appropriate providers for different tasks</li>
              <li>Enhanced privacy options through local model execution</li>
              <li>Automatic fallback between providers for reliability</li>
            </ul>
          </div>
        </div>
        </div>
      </section>

      <!-- Model Detailed Documentation -->
      <section id="model-detailed-documentation" class="help-section">
        <div class="section-header" @click="toggleSection('modelDetailedDocumentation')">
          <h2>Model Detailed Documentation</h2>
          <span class="toggle-icon">{{ sectionExpanded.modelDetailedDocumentation ? '−' : '+' }}</span>
        </div>
        <div v-show="sectionExpanded.modelDetailedDocumentation" class="section-content">
          <p>This section provides in-depth technical documentation for each of the 27 AI models in the Self Soul AGI system. Each model is documented with its architecture, capabilities, usage examples, and integration details.</p>
          
          <!-- Manager Model Detailed Documentation -->
          <div class="model-detailed-doc">
            <h3>Manager Model (Port 8001) - Core Coordination System</h3>
            
            <div class="model-technical-specs">
              <h4>Technical Specifications</h4>
              <table>
                <tbody>
                  <tr>
                    <th>Model Type</th>
                    <td>Hierarchical Coordination Network</td>
                  </tr>
                  <tr>
                    <th>Neural Architecture</th>
                    <td>Multi-layer Transformer with Attention Mechanisms</td>
                  </tr>
                  <tr>
                    <th>Training Method</th>
                    <td>From-scratch training with reinforcement learning</td>
                  </tr>
                  <tr>
                    <th>Input Types</th>
                    <td>Text commands, system states, model statuses</td>
                  </tr>
                  <tr>
                    <th>Output Types</th>
                    <td>Task assignments, resource allocations, coordination plans</td>
                  </tr>
                  <tr>
                    <th>Integration Points</th>
                    <td>All 26 other models (ports 8002-8027)</td>
                  </tr>
                </tbody>
              </table>
            </div>
            
            <div class="model-architecture">
              <h4>Architecture Overview</h4>
              <p>The Manager Model implements a sophisticated hierarchical coordination system consisting of three main components:</p>
              
              <ol>
                <li><strong>Task Analyzer</strong>: Parses user requests and system requirements to understand task objectives</li>
                <li><strong>Resource Allocator</strong>: Dynamically assigns tasks to appropriate specialized models based on capabilities and current load</li>
                <li><strong>Result Integrator</strong>: Combines outputs from multiple models into cohesive responses</li>
              </ol>
              
              <p>The model uses attention mechanisms to prioritize tasks and a reinforcement learning system to optimize coordination strategies over time.</p>
            </div>
            
            <div class="model-capabilities">
              <h4>Core Capabilities</h4>
              <ul>
                <li><strong>Intelligent Task Routing</strong>: Automatically routes requests to the most appropriate specialized models</li>
                <li><strong>Load Balancing</strong>: Distributes workload evenly across all available models</li>
                <li><strong>Conflict Resolution</strong>: Resolves conflicts when multiple models provide conflicting outputs</li>
                <li><strong>Error Recovery</strong>: Detects and recovers from model failures or errors</li>
                <li><strong>Performance Monitoring</strong>: Continuously tracks model performance and adjusts strategies accordingly</li>
              </ul>
            </div>
            
            <div class="model-usage">
              <h4>Usage Examples</h4>
              
              <div class="usage-example">
                <h5>Example 1: Complex Multimodal Request</h5>
                <pre class="code-block">
User: "Analyze this image of a car accident and write a detailed report"
Manager Model Processing:
1. Routes image to Vision Model (Port 8004) for object detection
2. Sends scene context to Sensor Model (Port 8012) for environmental analysis  
3. Coordinates with Language Model (Port 8002) for report generation
4. Integrates all results into a comprehensive response</pre>
              </div>
              
              <div class="usage-example">
                <h5>Example 2: System Optimization Request</h5>
                <pre class="code-block">
User: "Optimize the system for financial analysis tasks"
Manager Model Processing:
1. Activates Finance Model (Port 8022) as primary processor
2. Configures Prediction Model (Port 8014) for market forecasting
3. Allocates additional resources to Mathematical Model (Port 8027)
4. Adjusts priority weights for financial data processing</pre>
              </div>
            </div>
            
            <div class="model-integration">
              <h4>Integration with Other Models</h4>
              <p>The Manager Model maintains constant communication with all other models through the following mechanisms:</p>
              
              <ul>
                <li><strong>Health Checks</strong>: Regular status monitoring of all models</li>
                <li><strong>Capability Registry</strong>: Dynamic database of model capabilities and specialties</li>
                <li><strong>Performance Metrics</strong>: Real-time tracking of model accuracy and response times</li>
                <li><strong>Dependency Mapping</strong>: Understanding of inter-model dependencies for complex tasks</li>
              </ul>
            </div>
            
            <div class="model-training">
              <h4>Training and Optimization</h4>
              <p>The Manager Model undergoes continuous training through:</p>
              
              <ol>
                <li><strong>Supervised Learning</strong>: From labeled coordination examples</li>
                <li><strong>Reinforcement Learning</strong>: Reward-based optimization of coordination strategies</li>
                <li><strong>Transfer Learning</strong>: Applying coordination patterns from one domain to another</li>
                <li><strong>Self-Play</strong>: Simulating complex coordination scenarios for self-improvement</li>
              </ol>
            </div>
            
            <div class="model-performance">
              <h4>Performance Metrics</h4>
              <table>
                <tbody>
                  <tr>
                    <th>Coordination Accuracy</th>
                    <td>98.7%</td>
                    <td>Correct task assignment to appropriate models</td>
                  </tr>
                  <tr>
                    <th>Response Time</th>
                    <td>< 50ms</td>
                    <td>Average time to route and coordinate tasks</td>
                  </tr>
                  <tr>
                    <th>Error Recovery Rate</th>
                    <td>95.2%</td>
                    <td>Successful recovery from model failures</td>
                  </tr>
                  <tr>
                    <th>Load Balancing Efficiency</th>
                    <td>94.8%</td>
                    <td>Optimal distribution of workload</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
          
          <div class="other-models-overview">
            <h3>Other Models Overview</h3>
            <p>The following models follow similar detailed documentation structure. Each includes:</p>
            
            <ul>
              <li><strong>Technical Specifications</strong>: Architecture, training methods, input/output types</li>
              <li><strong>Core Capabilities</strong>: Primary functions and specialized skills</li>
              <li><strong>Integration Points</strong>: How the model interacts with other system components</li>
              <li><strong>Usage Examples</strong>: Practical scenarios demonstrating model capabilities</li>
              <li><strong>Performance Metrics</strong>: Quantitative measures of model effectiveness</li>
            </ul>
            
            <h4>Model Documentation Index</h4>
            <div class="model-index">
              <div class="index-column">
                <h5>Core Cognitive Models</h5>
                <ul>
                  <li><strong>Language Model (Port 8002)</strong>: Natural language understanding and generation</li>
                  <li><strong>Knowledge Model (Port 8003)</strong>: Knowledge storage, retrieval, and reasoning</li>
                  <li><strong>Vision Model (Port 8004)</strong>: Image processing and visual analysis</li>
                  <li><strong>Audio Model (Port 8005)</strong>: Speech and sound processing</li>
                  <li><strong>Autonomous Model (Port 8006)</strong>: Self-governance and decision making</li>
                </ul>
              </div>
              
              <div class="index-column">
                <h5>Specialized Processing Models</h5>
                <ul>
                  <li><strong>Programming Model (Port 8007)</strong>: Code generation and software development</li>
                  <li><strong>Planning Model (Port 8008)</strong>: Strategic planning and task execution</li>
                  <li><strong>Emotion Model (Port 8009)</strong>: Emotional intelligence and response</li>
                  <li><strong>Spatial Model (Port 8010)</strong>: Spatial reasoning and navigation</li>
                  <li><strong>Computer Vision Model (Port 8011)</strong>: Advanced visual processing</li>
                </ul>
              </div>
              
              <div class="index-column">
                <h5>Sensor and Control Models</h5>
                <ul>
                  <li><strong>Sensor Model (Port 8012)</strong>: Multi-sensor data fusion</li>
                  <li><strong>Motion Model (Port 8013)</strong>: Movement planning and control</li>
                  <li><strong>Prediction Model (Port 8014)</strong>: Predictive analytics and forecasting</li>
                  <li><strong>Advanced Reasoning Model (Port 8015)</strong>: Complex logical reasoning</li>
                  <li><strong>Data Fusion Model (Port 8028)</strong>: Multi-source data integration</li>
                </ul>
              </div>
              
              <div class="index-column">
                <h5>Advanced Cognitive Models</h5>
                <ul>
                  <li><strong>Creative Problem Solving Model (Port 8017)</strong>: Innovative solution generation</li>
                  <li><strong>Meta Cognition Model (Port 8018)</strong>: Self-awareness and cognitive monitoring</li>
                  <li><strong>Value Alignment Model (Port 8019)</strong>: Ethical decision making and safety</li>
                  <li><strong>Vision Image Model (Port 8020)</strong>: Advanced image processing</li>
                  <li><strong>Vision Video Model (Port 8021)</strong>: Video stream analysis</li>
                </ul>
              </div>
              
              <div class="index-column">
                <h5>Domain-Specific Models</h5>
                <ul>
                  <li><strong>Finance Model (Port 8022)</strong>: Financial analysis and prediction</li>
                  <li><strong>Medical Model (Port 8023)</strong>: Medical data analysis</li>
                  <li><strong>Collaboration Model (Port 8024)</strong>: Multi-agent coordination</li>
                  <li><strong>Optimization Model (Port 8025)</strong>: System performance optimization</li>
                  <li><strong>Computer Control Model (Port 8026)</strong>: Computer system management</li>
                  <li><strong>Mathematics Model (Port 8027)</strong>: Mathematical reasoning and computation</li>
                </ul>
              </div>
            </div>
          </div>
          
          <div class="documentation-note">
            <h4>Documentation Access</h4>
            <p>Complete detailed documentation for each model is available in the system's technical documentation repository. For the most up-to-date information on specific model implementations, refer to:</p>
            
            <ul>
              <li><code>core/models/</code> directory for implementation source code</li>
              <li><code>docs/model_specifications/</code> for technical specifications</li>
              <li><code>docs/api_reference/</code> for API documentation</li>
              <li>Interactive API documentation at <code>{{ backendDocsUrl }}</code></li>
            </ul>
          </div>
        </div>
      </section>

      <!-- Training Guide -->
      <section id="training-methodology" class="help-section">
        <div class="section-header" @click="toggleSection('trainingMethodology')">
          <h2>Training Methodology</h2>
          <span class="toggle-icon">{{ sectionExpanded.trainingMethodology ? '−' : '+' }}</span>
        </div>
        <div v-show="sectionExpanded.trainingMethodology" class="section-content">
        <p>The Self Soul system employs advanced training techniques to continuously improve its capabilities and adapt to new scenarios.</p>
        
        <div class="training-info">
          <h3>Training Interface Usage Steps</h3>
          <div class="step-list">
            <div class="step">
              <span class="step-number">1</span>
              <div class="step-content">
                <h4>Access the Training Interface</h4>
                <p>Navigate to <code>{{ frontendUrl }}/#/training</code> in your web browser to access the training dashboard.</p>
              </div>
            </div>
            <div class="step">
              <span class="step-number">2</span>
              <div class="step-content">
                <h4>Select Training Model</h4>
                <p>In the "Individual" tab, choose the specific model you want to train from the available model list. Each model is represented by a letter ID (A, B, C, etc.) corresponding to different cognitive functions.</p>
              </div>
            </div>
            <div class="step">
              <span class="step-number">3</span>
              <div class="step-content">
                <h4>Choose Dataset</h4>
                <p>Select the appropriate dataset from the "Dataset" dropdown menu. The <strong>Multimodal Dataset v1</strong> supports all models and is recommended for comprehensive training.</p>
              </div>
            </div>
            <div class="step">
              <span class="step-number">4</span>
              <div class="step-content">
                <h4>Configure Training Strategy</h4>
                <p>From the "Training Strategy" section, select the training approach that best fits your needs (e.g., Standard Training, Pre-trained Fine-tuning).</p>
              </div>
            </div>
            <div class="step">
              <span class="step-number">5</span>
              <div class="step-content">
                <h4>Set Training Parameters</h4>
                <p>Configure the specific parameters for your selected strategy, such as epochs, batch size, learning rate, or pre-training settings.</p>
              </div>
            </div>
            <div class="step">
              <span class="step-number">6</span>
              <div class="step-content">
                <h4>Initiate Training</h4>
                <p>Click the "Start Training" button to begin the training process. Monitor the progress and status updates in real-time.</p>
              </div>
            </div>
            <div class="step">
              <span class="step-number">7</span>
              <div class="step-content">
                <h4>Review Results</h4>
                <p>After training completes, review the performance metrics and training logs to evaluate the model's improvement.</p>
              </div>
            </div>
          </div>
        </div>

        <div class="training-info">
          <h3>Model Selection and ID Mapping</h3>
          <p>The system uses a dual ID system to manage model selection:</p>
          <div class="info-box">
            <h4>Frontend Letter IDs</h4>
            <p>On the training interface, models are displayed using letter IDs (A, B, C, D, etc.) for simplicity. These letter IDs are mapped to backend model IDs through a conversion system.</p>
            
            <h4>Backend Model IDs</h4>
            <p>The actual backend uses descriptive string IDs for each model. The mapping between frontend letter IDs and backend model IDs is handled automatically by the system:</p>
            
            <div class="model-id-mapping">
              <table>
                <thead>
                  <tr>
                    <th>Frontend Letter ID</th>
                    <th>Backend Model ID</th>
                    <th>Model Description</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>A</td>
                    <td>manager</td>
                    <td>Core management model responsible for coordinating all other models</td>
                  </tr>
                  <tr>
                    <td>B</td>
                    <td>language</td>
                    <td>Advanced language processing and understanding model</td>
                  </tr>
                  <tr>
                    <td>C</td>
                    <td>audio</td>
                    <td>Audio processing and voice recognition model</td>
                  </tr>
                  <tr>
                    <td>D</td>
                    <td>vision_image</td>
                    <td>Image vision processing and analysis model</td>
                  </tr>
                  <tr>
                    <td>E</td>
                    <td>vision_video</td>
                    <td>Video stream processing and analysis model</td>
                  </tr>
                  <tr>
                    <td>F</td>
                    <td>spatial</td>
                    <td>Spatial awareness and positioning model</td>
                  </tr>
                  <tr>
                    <td>G</td>
                    <td>sensor</td>
                    <td>Sensor data processing and perception model</td>
                  </tr>
                  <tr>
                    <td>H</td>
                    <td>computer</td>
                    <td>Computer control and interface model</td>
                  </tr>
                  <tr>
                    <td>I</td>
                    <td>motion</td>
                    <td>Motion control and actuator management model</td>
                  </tr>
                  <tr>
                    <td>J</td>
                    <td>knowledge</td>
                    <td>Knowledge base and expert system model</td>
                  </tr>
                  <tr>
                    <td>K</td>
                    <td>programming</td>
                    <td>Programming and code generation model</td>
                  </tr>
                  <tr>
                    <td>L</td>
                    <td>planning</td>
                    <td>Planning and decision-making model for task execution</td>
                  </tr>
                  <tr>
                    <td>M</td>
                    <td>autonomous</td>
                    <td>Autonomous operation and self-governance model</td>
                  </tr>
                  <tr>
                    <td>N</td>
                    <td>emotion</td>
                    <td>Emotion recognition and response model</td>
                  </tr>
                  <tr>
                    <td>O</td>
                    <td>spatial</td>
                    <td>Spatial Model (duplicate for compatibility)</td>
                  </tr>
                  <tr>
                    <td>P</td>
                    <td>vision_image</td>
                    <td>Computer Vision Model (duplicate for compatibility)</td>
                  </tr>
                  <tr>
                    <td>Q</td>
                    <td>sensor</td>
                    <td>Sensor Model (duplicate for compatibility)</td>
                  </tr>
                  <tr>
                    <td>R</td>
                    <td>motion</td>
                    <td>Motion Model (duplicate for compatibility)</td>
                  </tr>
                  <tr>
                    <td>S</td>
                    <td>prediction</td>
                    <td>Predictive analysis and forecasting model</td>
                  </tr>
                  <tr>
                    <td>T</td>
                    <td>collaboration</td>
                    <td>Model collaboration and coordination model for joint task execution</td>
                  </tr>
                  <tr>
                    <td>U</td>
                    <td>optimization</td>
                    <td>Model optimization and performance enhancement model</td>
                  </tr>
                  <tr>
                    <td>V</td>
                    <td>finance</td>
                    <td>Financial analysis and decision-making model</td>
                  </tr>
                  <tr>
                    <td>W</td>
                    <td>medical</td>
                    <td>Medical data analysis and healthcare model</td>
                  </tr>
                  <tr>
                    <td>X</td>
                    <td>value_alignment</td>
                    <td>Value alignment and ethical decision-making model</td>
                  </tr>
                </tbody>
              </table>
            </div>
            
            <h4>Supported Models for Datasets</h4>
            <p>Each dataset specifies which models it supports. When selecting a dataset, only the models compatible with that dataset will be available for training.</p>
          </div>
        </div>

        <div class="training-info">
          <h3>Training Approaches</h3>
          <ul>
            <li v-for="approach in trainingApproaches" :key="approach.id">
              <strong>{{ approach.title }}</strong> - {{ approach.description }}
            </li>
          </ul>
        </div>

        <div class="training-info">
          <h3>Training Preparation Workflow</h3>
          <p>The system implements a comprehensive training preparation process before actual model training begins. This preparation phase covers four core dimensions to ensure optimal training conditions:</p>
          
          <div class="info-box">
            <h4>1. Environment Initialization</h4>
            <ul>
              <li><strong>System Environment Checks:</strong> Verification of Python version, operating system compatibility, and required system libraries</li>
              <li><strong>Hardware Resource Validation:</strong> Detection of GPU/CPU availability, memory sufficiency, and resource utilization</li>
              <li><strong>PyTorch Environment Setup:</strong> Configuration of device settings (CUDA/cpu), library version validation, and performance optimizations</li>
            </ul>
          </div>
          
          <div class="info-box">
            <h4>2. Data Preprocessing</h4>
            <ul>
              <li><strong>Data Quality Checks:</strong> Automated validation of data integrity, completeness, and consistency</li>
              <li><strong>Format Standardization:</strong> Conversion of raw data into model-compatible formats for all modalities (text, image, audio, video)</li>
              <li><strong>Train/Validation Splits:</strong> Intelligent data partitioning with stratified sampling based on model requirements</li>
              <li><strong>Domain Adaptation:</strong> Optimization of data characteristics to match model capabilities and training objectives</li>
            </ul>
          </div>
          
          <div class="info-box">
            <h4>3. Model Configuration</h4>
            <ul>
              <li><strong>Model Loading:</strong> Secure loading of model architectures and weights (if available)</li>
              <li><strong>State Validation:</strong> Verification of model integrity and compatibility with training data</li>
              <li><strong>Hyperparameter Initialization:</strong> Setting of default parameters (learning rate, batch size, optimizer) with model-specific optimizations</li>
              <li><strong>"Train from Scratch" Setup:</strong> Special configuration for models being trained without pre-existing weights</li>
            </ul>
          </div>
          
          <div class="info-box">
            <h4>4. Dependency Checking</h4>
            <ul>
              <li><strong>Package Validation:</strong> Verification of all required Python packages and their versions</li>
              <li><strong>Resource Availability:</strong> Confirmation of sufficient memory, storage, and processing power for training</li>
              <li><strong>Compatibility Checks:</strong> Validation of model dependencies with system environment and training data</li>
            </ul>
          </div>
          
          <div class="info-box">
            <h4>Training Preparation Status</h4>
            <p>During preparation, models transition through specific status states:</p>
            <ul>
              <li><strong>PREPARING:</strong> Initial preparation phase is underway</li>
              <li><strong>PREPARED:</strong> All preparation steps completed successfully, ready for training</li>
              <li><strong>PREPARATION_FAILED:</strong> One or more preparation steps encountered errors</li>
            </ul>
          </div>
        </div>

        <div class="training-info">
          <h3>Autonomous Training Capabilities</h3>
          <p>The Self Soul system features advanced autonomous training functionality that enables models to self-optimize without constant human supervision. This feature leverages the Adaptive Learning Engine to continuously improve model performance through intelligent parameter adjustment and learning strategy optimization.</p>
          
          <div class="info-box">
            <h4>Key Features of Autonomous Training</h4>
            <ul>
              <li><strong>Automatic Parameter Tuning:</strong> The system dynamically adjusts training parameters (learning rate, batch size, epochs) based on real-time performance metrics</li>
              <li><strong>Adaptive Learning Strategies:</strong> Switches between different training approaches (fine-tuning, transfer learning) based on data characteristics and model needs</li>
              <li><strong>Multi-Model Coordination:</strong> Coordinates training across multiple models to ensure knowledge consistency and interoperability</li>
              <li><strong>Performance Monitoring:</strong> Continuously monitors model performance and automatically stops training when optimal results are achieved</li>
              <li><strong>Knowledge Integration:</strong> Automatically integrates newly learned knowledge into the system's knowledge base</li>
            </ul>
            
            <h4>Enabling Autonomous Training</h4>
            <p>To activate autonomous training:</p>
            <ol>
              <li>Navigate to the training interface at <code>{{ frontendUrl }}/#/training</code></li>
              <li>Select the models you want to train</li>
              <li>Choose <strong>Multimodal Dataset v1</strong> from the dataset dropdown (supports all models)</li>
              <li>Select <strong>Autonomous Training</strong> from the training strategy options</li>
              <li>Configure any additional parameters if needed</li>
              <li>Click "Start Training" to begin the autonomous training process</li>
            </ol>
          </div>
        </div>

        <div class="training-info">
          <h3>Pre-trained Fine-tuning Detailed Configuration</h3>
          <p>When selecting the "Pre-trained Fine-tuning" strategy, you can configure the following parameters:</p>
          
          <div class="parameter-list">
            <div class="parameter-item">
              <h4>Pretrained Model ID</h4>
              <p>The ID of the pre-trained model to use as the foundation. This should be a valid model ID that has already been pre-trained.</p>
            </div>
            <div class="parameter-item">
              <h4>Freeze Layers</h4>
              <p>A boolean option to freeze (keep fixed) certain layers of the pre-trained model during training. This is useful when you want to preserve the foundational knowledge of the pre-trained model.</p>
            </div>
            <div class="parameter-item">
              <h4>Freeze Layer Count</h4>
              <p>If Freeze Layers is enabled, this specifies the number of layers from the bottom of the model to freeze. Only the top layers will be trainable.</p>
            </div>
            <div class="parameter-item">
              <h4>Fine-tuning Mode</h4>
              <p>Select the fine-tuning approach:</p>
              <ul>
                <li><strong>Full Fine-tuning:</strong> All layers are trainable (if Freeze Layers is disabled)</li>
                <li><strong>Partial Fine-tuning:</strong> Only specific layers are trainable</li>
                <li><strong>Linear Probing:</strong> Only the final classification layer is trainable</li>
              </ul>
            </div>
          </div>
          
          <div class="info-box">
            <h4>Use Cases for Pre-trained Fine-tuning</h4>
            <ul>
              <li>When you have limited training data but want to leverage existing knowledge</li>
              <li>For domain adaptation - adapting a general model to a specific domain</li>
              <li>To speed up training by starting from a model that already has relevant knowledge</li>
              <li>When you want to preserve the core capabilities of a pre-trained model while adding task-specific knowledge</li>
            </ul>
          </div>
        </div>

        <div class="training-info">
          <h3>CPU/GPU Training Selection</h3>
          <p>The Self Soul system supports flexible CPU and GPU training configurations. You can choose the optimal hardware for your training needs:</p>
          
          <div class="info-box">
            <h4>Training Hardware Options</h4>
            <ul>
              <li><strong>CPU Training:</strong> Suitable for smaller models or when GPU resources are unavailable. Provides stable training with predictable performance.</li>
              <li><strong>GPU Training:</strong> Recommended for larger models and complex training tasks. Provides significant speed improvements through parallel processing.</li>
              <li><strong>Automatic Detection:</strong> The system automatically detects available GPU resources and can optimize training configuration accordingly.</li>
              <li><strong>Hybrid Training:</strong> Some models support mixed CPU/GPU training where certain layers run on GPU while others run on CPU.</li>
            </ul>
            
            <h4>How to Select Training Hardware</h4>
            <ol>
              <li>Navigate to the Training page at <code>{{ frontendUrl }}/#/training</code></li>
              <li>Select the models you want to train</li>
              <li>In the training configuration panel, look for "Hardware Selection" or "Device Configuration"</li>
              <li>Choose between:
                <ul>
                  <li><strong>Auto (Recommended):</strong> Let the system automatically select the best hardware based on model requirements and available resources</li>
                  <li><strong>CPU Only:</strong> Force training on CPU only</li>
                  <li><strong>GPU Preferred:</strong> Use GPU if available, fall back to CPU if not</li>
                  <li><strong>Specific GPU:</strong> Select a specific GPU device when multiple GPUs are available</li>
                </ul>
              </li>
              <li>Start training with your selected hardware configuration</li>
            </ol>
            
            <h4>Performance Considerations</h4>
            <ul>
              <li><strong>GPU Training:</strong> 5-10x faster for most neural network models, requires CUDA-compatible NVIDIA GPU</li>
              <li><strong>CPU Training:</strong> More stable for debugging, better for small datasets, no GPU dependencies</li>
              <li><strong>Memory Usage:</strong> GPU training typically uses more memory but processes data faster</li>
              <li><strong>Power Consumption:</strong> GPU training consumes more power but completes tasks quicker</li>
            </ul>
          </div>
        </div>

        <div class="training-info">
          <h3>External Model Integration with Local Training</h3>
          <p>The system provides a powerful feature that allows using external API models while simultaneously training local counterparts. This enables you to leverage powerful external models while building your own local capabilities.</p>
          
          <div class="info-box">
            <h4>How It Works</h4>
            <p>When you enable an external model (like OpenAI GPT-4 or DeepSeek) for a specific task, the system can be configured to:</p>
            <ol>
              <li><strong>Use the external model</strong> for immediate task processing</li>
              <li><strong>Record the interactions</strong> and use them as training data for your local model</li>
              <li><strong>Train your local model</strong> on these interactions to improve its capabilities</li>
              <li><strong>Gradually transition</strong> from external to local models as the local model improves</li>
            </ol>
            
            <h4>Configuration Steps</h4>
            <ol>
              <li>Go to Settings page and configure your external API providers under "API Settings"</li>
              <li>Navigate to the model configuration section for the specific model you want to enhance</li>
              <li>Enable "Parallel Training with External Model" option</li>
              <li>Configure training parameters:
                <ul>
                  <li><strong>Training Frequency:</strong> How often to train the local model (e.g., after every 100 external calls)</li>
                  <li><strong>Knowledge Transfer:</strong> What aspects to transfer (reasoning patterns, response styles, specific capabilities)</li>
                  <li><strong>Validation Strategy:</strong> How to validate that local model is learning correctly from external model outputs</li>
                </ul>
              </li>
              <li>Save configuration and start using the external model</li>
            </ol>
            
            <h4>Use Cases</h4>
            <ul>
              <li><strong>Knowledge Transfer:</strong> Use expensive external models to generate high-quality training data for your local models</li>
              <li><strong>Cost Optimization:</strong> Start with external models, gradually reduce usage as local models improve</li>
              <li><strong>Privacy Preservation:</strong> Use external models for non-sensitive tasks while training local models for sensitive operations</li>
              <li><strong>Specialized Adaptation:</strong> Fine-tune general external models for your specific domain using local training</li>
              <li><strong>Hybrid Deployment:</strong> Use external models for complex tasks while local models handle simpler, frequent requests</li>
            </ul>
            
            <div class="beginner-tip">
              <h4>Practical Example: Language Model Enhancement</h4>
              <p>Suppose you want to improve your local Language Model (Port 8002):</p>
              <ol>
                <li>Configure OpenAI GPT-4 as an external provider in Settings</li>
                <li>Enable "Parallel Training" for Language Model</li>
                <li>Continue using the Conversation interface as normal</li>
                <li>The system will:
                  <ul>
                    <li>Send complex queries to GPT-4 for high-quality responses</li>
                    <li>Store query-response pairs as training data</li>
                    <li>Periodically train your local Language Model on this collected data</li>
                    <li>Gradually increase local model usage as it improves</li>
                  </ul>
                </li>
                <li>Monitor training progress in the Training page</li>
              </ol>
            </div>
            
            <h4>Supported Training Types with External Models</h4>
            <p>The system supports multiple training approaches when using external models:</p>
            <ul>
              <li><strong>Imitation Learning:</strong> Local models learn to mimic external model responses</li>
              <li><strong>Distillation:</strong> Knowledge distillation from larger external models to smaller local models</li>
              <li><strong>Transfer Learning:</strong> Transfer specialized capabilities from external to local models</li>
              <li><strong>Reinforcement Learning from Human Feedback (RLHF):</strong> Use external model outputs as reward signals</li>
              <li><strong>Multi-task Learning:</strong> Train local models on multiple tasks demonstrated by external models</li>
            </ul>
          </div>
        </div>

        <div class="training-info">
          <h3>Best Practices</h3>
          <ul>
            <li>Provide diverse and representative training data for comprehensive learning</li>
            <li>Monitor training progress through the system dashboard</li>
            <li>Regularly update model parameters based on performance metrics</li>
            <li>Implement validation checks to ensure training quality</li>
            <li>Balance individual model training with collaborative training sessions</li>
            <li>Schedule periodic knowledge integration sessions to maintain model coherence</li>
            <li>Use the built-in performance analytics to identify areas for improvement</li>
            <li>When using pre-trained models, start with a small learning rate to avoid catastrophic forgetting</li>
            <li>Consider freezing lower layers when fine-tuning pre-trained models to preserve general knowledge</li>
            <li>For CPU/GPU training: Start with CPU for debugging, switch to GPU for production training</li>
            <li>When using external models: Gradually reduce external API calls as local models improve to optimize costs</li>
            <li>Always validate that local models are correctly learning from external model demonstrations</li>
          </ul>
        </div>

        <div class="training-info">
          <h3>Dataset Format Examples</h3>
          <p>Proper dataset formatting is essential for effective model training. Below are examples of different dataset formats supported by the system:</p>
          
          <div class="dataset-format">
            <h4>Text Dataset Example (JSON Format)</h4>
            <pre class="code-block">
{
  "name": "Text Classification Dataset",
  "description": "Dataset for text classification tasks",
  "version": "1.0",
  "data": [
    {
      "id": "1",
      "text": "The system is performing well.",
      "label": "positive",
      "metadata": {
        "source": "user_feedback",
        "timestamp": "2024-01-15T10:30:00Z"
      }
    },
    {
      "id": "2",
      "text": "There seems to be an issue with the language model.",
      "label": "negative",
      "metadata": {
        "source": "system_log",
        "timestamp": "2024-01-15T11:15:00Z"
      }
    }
  ],
  "supportedModels": ["B", "C"]
}</pre>
          </div>
          
          <div class="dataset-format">
            <h4>Image Dataset Example (JSON Format)</h4>
            <pre class="code-block">
{
  "name": "Image Recognition Dataset",
  "description": "Dataset for image classification tasks",
  "version": "1.0",
  "data": [
    {
      "id": "1",
      "image_path": "/data/images/cat_001.jpg",
      "label": "cat",
      "metadata": {
        "resolution": "224x224",
        "color_mode": "RGB"
      }
    },
    {
      "id": "2",
      "image_path": "/data/images/dog_001.jpg",
      "label": "dog",
      "metadata": {
        "resolution": "224x224",
        "color_mode": "RGB"
      }
    }
  ],
  "supportedModels": ["D"]
}</pre>
          </div>
          
          <div class="dataset-format">
            <h4>Multimodal Dataset Example (JSON Format)</h4>
            <pre class="code-block">
{
  "name": "Multimodal Dataset v1",
  "description": "Comprehensive dataset with text, image, and audio data",
  "version": "1.0",
  "data": [
    {
      "id": "1",
      "text": "A person is playing the piano.",
      "image_path": "/data/images/piano_001.jpg",
      "audio_path": "/data/audio/piano_001.mp3",
      "label": "music_performance",
      "metadata": {
        "scene": "concert",
        "duration": "120.5"
      }
    },
    {
      "id": "2",
      "text": "A dog is barking at the mailman.",
      "image_path": "/data/images/dog_barking_001.jpg",
      "audio_path": "/data/audio/dog_barking_001.wav",
      "label": "animal_sound",
      "metadata": {
        "scene": "residential",
        "duration": "30.2"
      }
    }
  ],
  "supportedModels": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S"]
}</pre>
          </div>
          
          <div class="info-box">
            <h4>Dataset Requirements</h4>
            <ul>
              <li>All datasets must be in valid JSON format</li>
              <li>Each dataset must have a unique name and version</li>
              <li>The <code>supportedModels</code> field specifies which models can use the dataset</li>
              <li>Multimodal datasets should include all relevant data types (text, images, audio) for comprehensive training</li>
              <li>File paths should be relative to the dataset root directory</li>
              <li>Metadata fields are optional but recommended for better data management</li>
            </ul>
          </div>
        </div>
        </div>
      </section>

      <!-- Advanced Features -->
      <section id="advanced-capabilities" class="help-section">
        <div class="section-header" @click="toggleSection('advancedCapabilities')">
          <h2>Advanced Capabilities</h2>
          <span class="toggle-icon">{{ sectionExpanded.advancedCapabilities ? '−' : '+' }}</span>
        </div>
        <div v-show="sectionExpanded.advancedCapabilities" class="section-content">
        <p>Self Soul incorporates cutting-edge AI technologies to deliver sophisticated cognitive abilities:</p>
        
        <div class="feature-grid">
          <div class="feature-card">
            <h3>Meta-Learning</h3>
            <p>Ability to learn how to learn, optimizing future learning processes and accelerating adaptation to new domains</p>
          </div>
          <div class="feature-card">
            <h3>Self-Reflection</h3>
            <p>Continuous evaluation and improvement of internal processes through introspective analysis</p>
          </div>
          <div class="feature-card">
            <h3>Cross-Modal Transfer</h3>
            <p>Application of knowledge across different sensory modalities to enhance understanding and problem-solving</p>
          </div>
          <div class="feature-card">
            <h3>Adaptive Reasoning</h3>
            <p>Dynamic adjustment of reasoning strategies based on context and task requirements</p>
          </div>
          <div class="feature-card">
            <h3>Intrinsic Motivation System</h3>
            <p>Self-directed exploration and learning driven by internal goals and curiosity</p>
          </div>
          <div class="feature-card">
            <h3>Explainable AI</h3>
            <p>Transparent decision-making processes with clear explanations of reasoning paths</p>
          </div>
          <div class="feature-card">
            <h3>External API Integration</h3>
            <p>Comprehensive support for 18 external API providers including OpenAI, Anthropic, Google AI, AWS, Azure, and domestic Chinese providers like DeepSeek, Zhipu AI, and Baidu ERNIE with unified management and failover mechanisms</p>
          </div>
        </div>
        </div>
      </section>

      <!-- Page Features Documentation -->
      <section id="page-features" class="help-section">
        <div class="section-header" @click="toggleSection('pageFeatures')">
          <h2>Page Features Documentation</h2>
          <span class="toggle-icon">{{ sectionExpanded.pageFeatures ? '−' : '+' }}</span>
        </div>
        <div v-show="sectionExpanded.pageFeatures" class="section-content">
          <p>This section provides detailed documentation for each page feature in the Self Soul system.</p>
          
          <!-- Home View -->
          <div class="page-docs">
            <h3>Home View</h3>
            <p>The Home View serves as the main dashboard for managing cameras and devices in the system.</p>
            
            <h4>Key Features:</h4>
            <ul>
              <li><strong>Multi-camera Management:</strong> View and control multiple cameras with status indicators</li>
              <li><strong>Device Status Monitoring:</strong> Real-time monitoring of camera status (active/inactive)</li>
              <li><strong>Camera Controls:</strong> Start, stop, and configure individual cameras</li>
              <li><strong>Stereo Vision Pairs:</strong> Manage paired cameras for 3D vision capabilities</li>
              <li><strong>User Guide:</strong> Access system guide for new users</li>
            </ul>
            
            <h4>Usage:</h4>
            <ol>
              <li>View the status of all connected cameras in the device grid</li>
              <li>Click "Start" to activate a camera or "Stop" to deactivate it</li>
              <li>Use "Settings" to configure camera parameters like resolution and frame rate</li>
              <li>Manage stereo vision pairs for advanced computer vision tasks</li>
            </ol>
          </div>
          
          <!-- Chat From Scratch -->
          <div class="page-docs">
            <h3>Chat From Scratch</h3>
            <p>A specialized chat interface designed for direct interaction with the language model during training.</p>
            
            <h4>Key Features:</h4>
            <ul>
              <li><strong>Real-time Chat:</strong> Interactive conversation with the language model</li>
              <li><strong>Training Status:</strong> View current model training parameters and progress</li>
              <li><strong>Clear Chat:</strong> Reset the conversation history</li>
              <li><strong>Confidence Monitoring:</strong> Track model confidence levels during interactions</li>
            </ul>
            
            <h4>Usage:</h4>
            <ol>
              <li>Type messages in the input field to interact with the language model</li>
              <li>Toggle "Show Status" to view detailed training information</li>
              <li>Monitor vocabulary size, training epochs, and last activity timestamps</li>
              <li>Use "Clear Chat" to start a new conversation</li>
            </ol>
          </div>
          
          <!-- Conversation View -->
          <div class="page-docs">
            <h3>Conversation View</h3>
            <p>The main conversational interface for interacting with the Self Soul management model.</p>
            
            <h4>Key Features:</h4>
            <ul>
              <li><strong>Natural Language Conversation:</strong> Engage in real-time dialogue with the management model</li>
              <li><strong>Multimodal Support:</strong> Send and receive text, images, and audio messages</li>
              <li><strong>Emotional Analysis:</strong> The model analyzes and responds to emotional cues</li>
              <li><strong>Connection Status:</strong> Real-time monitoring of model connection (Port 8001)</li>
              <li><strong>Conversation History:</strong> Maintain and clear chat history</li>
            </ul>
            
            <h4>Usage:</h4>
            <ol>
              <li>Type messages in the input field to communicate with the management model</li>
              <li>Use media buttons to send images or audio files</li>
              <li>Monitor connection status through the header indicator</li>
              <li>Click "Clear Conversation" to reset the chat history</li>
            </ol>
          </div>
          
          <!-- Knowledge View -->
          <div class="page-docs">
            <h3>Knowledge View</h3>
            <p>A comprehensive interface for managing the system's knowledge base.</p>
            
            <h4>Key Features:</h4>
            <ul>
              <li><strong>Knowledge Import:</strong> Upload and import knowledge from various file formats (PDF, DOCX, TXT, JSON, CSV)</li>
              <li><strong>Browse Knowledge:</strong> Search and explore the existing knowledge base</li>
              <li><strong>Knowledge Management:</strong> Organize and categorize knowledge entries</li>
              <li><strong>Statistics:</strong> View knowledge base metrics and analytics</li>
            </ul>
            
            <h4>Usage:</h4>
            <ol>
              <li>Navigate between Import, Browse, Manage, and Statistics tabs</li>
              <li>Click "Select Files to Import" to upload knowledge documents</li>
              <li>Use the search functionality to find specific knowledge entries</li>
              <li>View statistics to understand knowledge base growth and distribution</li>
            </ol>
          </div>
          
          <!-- Settings View -->
          <div class="page-docs">
            <h3>Settings View</h3>
            <p>Configure system settings and manage models.</p>
            
            <h4>Key Features:</h4>
            <ul>
              <li><strong>Model Statistics:</strong> Overview of total, active, running, and API models</li>
              <li><strong>API Service Status:</strong> Monitor global API service availability</li>
              <li><strong>Model Configuration:</strong> Manage model settings and configurations</li>
              <li><strong>Status Monitoring:</strong> Real-time status updates for all system components</li>
              <li><strong>Extended API Support:</strong> Configuration for multiple AI service providers including domestic Chinese APIs and local models</li>
            </ul>
            
            <h4>Usage:</h4>
            <ol>
              <li>View system statistics at the top of the page</li>
              <li>Check API service status and refresh if needed</li>
              <li>Configure individual model settings through the interface</li>
              <li>Monitor model status indicators for each component</li>
              <li>Configure external API providers for enhanced model capabilities</li>
            </ol>
            
            <div class="info-box">
              <h4>API Configuration Details</h4>
              <p>The Settings View provides comprehensive configuration options for external AI API providers and local models:</p>
              
              <h5>Supported API Providers</h5>
              <p>The system supports a wide range of AI service providers that can be configured under "Models > API Settings > API Provider API":</p>
              
              <div class="provider-list">
                <h6>International Providers:</h6>
                <ul>
                  <li><strong>OpenAI:</strong> GPT models including GPT-4, GPT-3.5-Turbo</li>
                  <li><strong>Anthropic:</strong> Claude models including Claude-3 series</li>
                  <li><strong>Google AI:</strong> Gemini models and PaLM API</li>
                  <li><strong>HuggingFace:</strong> Open-source models from HuggingFace Hub</li>
                  <li><strong>Cohere:</strong> Command models and embedding APIs</li>
                  <li><strong>Mistral:</strong> Mistral AI models including Mixtral</li>
                </ul>
                
                <h6>Domestic Chinese Providers:</h6>
                <ul>
                  <li><strong>DeepSeek:</strong> DeepSeek models including DeepSeek-V2, DeepSeek-Coder</li>
                  <li><strong>SiliconFlow:</strong> SiliconFlow platform models</li>
                  <li><strong>Zhipu AI:</strong> GLM models including GLM-4</li>
                  <li><strong>Baidu ERNIE:</strong> ERNIE models including ERNIE 4.0</li>
                  <li><strong>Alibaba Qwen:</strong> Qwen models including Qwen2.5</li>
                  <li><strong>Moonshot:</strong> Moonshot AI models including Kimi</li>
                  <li><strong>Yi:</strong> Yi models including Yi-34B</li>
                  <li><strong>Tencent Hunyuan:</strong> Tencent Hunyuan models</li>
                </ul>
                
                <h6>Local Model Support:</h6>
                <ul>
                  <li><strong>Ollama:</strong> Run local LLMs with automatic configuration (default: {{ ollamaConfig }}, model: llama2)</li>
                </ul>
              </div>
              
              <h5>Configuration Steps</h5>
              <ol>
                <li>Navigate to Settings View > Models section</li>
                <li>Click on "API Settings" tab</li>
                <li>Select "API Provider API" configuration</li>
                <li>Choose the desired provider from the dropdown menu</li>
                <li>Enter API key, endpoint URL, and model name</li>
                <li>For Ollama, the system automatically configures default settings but can be customized if needed</li>
                <li>Save the configuration for each model individually</li>
              </ol>
              
              <h5>Benefits of Multi-Provider Support</h5>
              <ul>
                <li><strong>Flexibility:</strong> Choose the best provider for each task based on cost, performance, or specific capabilities</li>
                <li><strong>Redundancy:</strong> Automatic fallback between providers if one is unavailable</li>
                <li><strong>Cost Optimization:</strong> Mix and match providers based on pricing models</li>
                <li><strong>Privacy:</strong> Use local models (Ollama) for sensitive data processing</li>
                <li><strong>Performance:</strong> Leverage specialized models from different providers for specific tasks</li>
              </ul>
              
              <h5>Automatic Configuration Features</h5>
              <p>The system includes intelligent auto-configuration for supported providers:</p>
              <ul>
                <li><strong>Ollama:</strong> Automatically detects local Ollama instance and configures default settings</li>
                <li><strong>Provider Detection:</strong> Validates API endpoints and keys before saving configurations</li>
                <li><strong>Model Compatibility:</strong> Suggests appropriate models based on selected provider</li>
                <li><strong>Performance Optimization:</strong> Recommends optimal settings based on model capabilities</li>
              </ul>
            </div>
          </div>
          
          <!-- Robot Settings View -->
          <div id="robot-settings" class="page-docs">
            <h3>Robot Settings View</h3>
            <p>The robot settings interface provides comprehensive hardware configuration and management for AGI robot systems.</p>
            
            <h4>Key Features:</h4>
            <ul>
              <li><strong>Hardware Connection Management:</strong> Real-time status monitoring and control for sensors, motors, servos, and cameras</li>
              <li><strong>Hardware Status Display:</strong> Detected device counts, connection status, battery level monitoring</li>
              <li><strong>Joint Control:</strong> Slider controls for each joint with adjustable angle ranges and real-time position feedback</li>
              <li><strong>Sensor Data Monitoring:</strong> Live display of sensor readings including temperature, distance, force, and environmental data</li>
              <li><strong>Model Collaboration Settings:</strong> Configuration of multi-model collaboration patterns for robot control and coordination</li>
              <li><strong>Voice Control Integration:</strong> Enable/disable voice control functionality with real-time speech recognition</li>
              <li><strong>Hardware Initialization:</strong> Automatic detection and initialization of hardware devices</li>
              <li><strong>Connection Management:</strong> Configuration of serial port, USB, and network connections</li>
              <li><strong>Device Listing:</strong> Detailed lists of connected hardware, available hardware types, and connection status</li>
              <li><strong>Real-time Spatial Recognition Integration:</strong> Dual-camera spatial recognition with depth map preview and robot motion control integration</li>
              <li><strong>Spatial-Aware Motion Planning:</strong> Robot motion control using spatial data for obstacle avoidance and target navigation</li>
              <li><strong>Stereo Vision Processing:</strong> Real-time depth map generation and 3D point cloud creation for spatial awareness</li>
              <li><strong>Real-time Sensor Data API:</strong> Integration with real sensor data endpoints for accurate environmental monitoring</li>
              <li><strong>Collision Detection System:</strong> Real-time obstacle detection with warning levels based on sensor data</li>
              <li><strong>Data Recording & Playback:</strong> Record sensor and joint data for analysis and replay</li>
              <li><strong>English-only Voice Commands:</strong> Pure English voice command recognition for robot control</li>
            </ul>
            
            <h4>Access Methods:</h4>
            <ul>
              <li><strong>Web Interface:</strong> Navigate to <code>{{ frontendUrl }}/#/robot-settings</code></li>
              <li><strong>API Integration:</strong> Hardware status endpoints and control APIs available through main API gateway</li>
            </ul>
            
            <h4>Usage:</h4>
            <ol>
              <li>Access the robot settings page: <code>{{ frontendUrl }}/#/robot-settings</code></li>
              <li>Click "Detect Hardware" to scan for connected devices</li>
              <li>Click "Initialize Hardware" to initialize detected hardware for system use</li>
              <li>Adjust joint limits and default positions using the joint control sliders</li>
              <li>Configure sensor parameters and calibration settings as needed</li>
              <li>Set up collaboration patterns between perception, motion, and planning models</li>
              <li>Enable voice control for hands-free operation if desired</li>
              <li>Configure communication protocols and port settings</li>
              <li>Monitor hardware status and sensor data in real-time</li>
            </ol>
            
            <h4>Configuration Workflow:</h4>
            <ol>
              <li><strong>Hardware Detection:</strong> Click "Detect Hardware" to scan for connected devices</li>
              <li><strong>Hardware Initialization:</strong> Initialize detected hardware for system use</li>
              <li><strong>Joint Configuration:</strong> Adjust joint limits and default positions as needed</li>
              <li><strong>Sensor Calibration:</strong> Configure sensor parameters and calibration settings</li>
              <li><strong>Model Collaboration:</strong> Set up collaboration patterns between perception, motion, and planning models</li>
              <li><strong>Voice Control:</strong> Enable voice control for hands-free operation</li>
              <li><strong>Connection Management:</strong> Configure communication protocols and port settings</li>
            </ol>
          </div>
          
          <!-- Robot Training -->
          <div id="robot-training" class="page-docs">
            <h3>Robot Training</h3>
            <p>The robot training system enables comprehensive training of AGI robot models with hardware integration, supporting multiple training modes and real-time hardware feedback.</p>
            
            <h4>Training Modes:</h4>
            <ul>
              <li><strong>Motion Basic Training:</strong> Fundamental movement patterns and joint control training</li>
              <li><strong>Perception Training:</strong> Visual and sensor data processing training for environmental awareness</li>
              <li><strong>Collaboration Training:</strong> Multi-model coordination training for complex task execution</li>
              <li><strong>AGI Fusion Training:</strong> Advanced integrated training combining all cognitive capabilities</li>
              <li><strong>Spatial Recognition Training:</strong> Training with real-time stereo vision and depth perception for spatial awareness</li>
              <li><strong>Robot Motion with Spatial Data:</strong> Training robot motion control using real-time spatial data for obstacle avoidance and navigation</li>
            </ul>
            
            <h4>Hardware Integration:</h4>
            <p>The robot training system integrates with physical hardware components for realistic training scenarios:</p>
            <ul>
              <li><strong>Joint Selection:</strong> Choose specific robot joints for targeted movement training</li>
              <li><strong>Sensor Selection:</strong> Select sensors for environmental perception training</li>
              <li><strong>Camera Selection:</strong> Choose cameras for visual perception training</li>
              <li><strong>Real-time Feedback:</strong> Live hardware status monitoring during training</li>
            </ul>
            
            <h4>Training Control:</h4>
            <ul>
              <li><strong>Start Training:</strong> Initiate robot training with selected models and hardware components</li>
              <li><strong>Pause Training:</strong> Temporarily suspend training while maintaining current state</li>
              <li><strong>Stop Training:</strong> Terminate training session and reset hardware</li>
              <li><strong>Reset Training:</strong> Clear training state and prepare for new session</li>
              <li><strong>Training Status Monitoring:</strong> Real-time progress tracking and performance metrics</li>
            </ul>
            
            <h4>Safety Features:</h4>
            <ul>
              <li><strong>Joint Velocity Limits:</strong> Prevent excessive joint movement speeds</li>
              <li><strong>Torque Limits:</strong> Protect hardware from excessive force</li>
              <li><strong>Temperature Monitoring:</strong> Prevent overheating of hardware components</li>
              <li><strong>Emergency Stop Thresholds:</strong> Automatic safety shutdown in critical situations</li>
            </ul>
            
            <h4>API Integration:</h4>
            <p><strong>Note:</strong> Robot training now uses the unified training API system. The following endpoints are used:</p>
            <ul>
              <li><strong>POST /api/training/start:</strong> Start training (supports both general and hardware-integrated training)</li>
              <li><strong>POST /api/training/{job_id}/stop:</strong> Stop specific training job</li>
              <li><strong>GET /api/training/status/{job_id}:</strong> Get training job status and progress</li>
            </ul>
            <p>For hardware-integrated training, include <code>hardware_config</code> in the training configuration.</p>
            
            <h4>Training Workflow:</h4>
            <ol>
              <li><strong>Hardware Preparation:</strong> Ensure robot hardware is connected and initialized</li>
              <li><strong>Training Configuration:</strong> Select training mode, models, and hardware components</li>
              <li><strong>Parameter Setup:</strong> Configure training parameters and safety limits</li>
              <li><strong>Training Execution:</strong> Start training and monitor real-time progress</li>
              <li><strong>Performance Evaluation:</strong> Review training results and adjust parameters as needed</li>
              <li><strong>Hardware Reset:</strong> Safely reset hardware after training completion</li>
            </ol>
            
            <h4>Integration with Robot Settings:</h4>
            <p>The robot training system integrates seamlessly with the Robot Settings interface. Before starting training:</p>
            <ol>
              <li>Use Robot Settings to detect and initialize hardware</li>
              <li>Configure joint limits and sensor parameters</li>
              <li>Set up model collaboration patterns</li>
              <li>Verify hardware connectivity and status</li>
              <li>Proceed to robot training with configured hardware</li>
            </ol>
            
            <h4>Real-time Monitoring:</h4>
            <p>During training, the system provides real-time monitoring of:</p>
            <ul>
              <li><strong>Training Progress:</strong> Percentage completion and estimated time remaining</li>
              <li><strong>Hardware Status:</strong> Joint positions, sensor readings, camera feeds</li>
              <li><strong>Performance Metrics:</strong> Learning curves, accuracy improvements, error rates</li>
              <li><strong>Safety Parameters:</strong> Temperature, torque, velocity within safe limits</li>
              <li><strong>Training Logs:</strong> Detailed event logs for debugging and analysis</li>
            </ul>
            
            <div class="note-box">
              <h4>Note:</h4>
              <p>For optimal robot training results, ensure that:</p>
              <ul>
                <li>All required hardware is properly connected and calibrated</li>
                <li>Safety limits are configured appropriately for your hardware</li>
                <li>Training datasets are compatible with selected models and hardware</li>
                <li>System has sufficient computational resources for real-time training</li>
                <li>Emergency stop procedures are understood and accessible</li>
              </ul>
            </div>
          </div>
          
          <!-- Train View -->
          <div class="page-docs">
            <h3>Train View</h3>
            <p>The training interface for managing model training and dataset selection.</p>
            
            <h4>Key Features:</h4>
            <ul>
              <li><strong>Model Selection:</strong> Choose from 19 specialized models for training</li>
              <li><strong>Dataset Selection:</strong> Select training datasets including the Multimodal Dataset v1</li>
              <li><strong>Training Strategy:</strong> Choose from various training strategies including Autonomous Training</li>
              <li><strong>Parameter Configuration:</strong> Set training parameters like batch size and learning rate</li>
              <li><strong>Training Progress:</strong> Monitor training progress and status</li>
            </ul>
            
            <h4>Usage:</h4>
            <ol>
              <li>Navigate to the training interface at <code>{{ frontendUrl }}/#/training</code></li>
              <li>Select one or more models to train</li>
              <li>Choose Multimodal Dataset v1 for comprehensive training</li>
              <li>Select a training strategy (e.g., Autonomous Training)</li>
              <li>Configure training parameters if needed</li>
              <li>Click "Start Training" to begin the training process</li>
            </ol>
          </div>
          
          <!-- Help View -->
          <div class="page-docs">
            <h3>Help View</h3>
            <p>A comprehensive help system providing documentation for all system features.</p>
            
            <h4>Key Features:</h4>
            <ul>
              <li><strong>Search Functionality:</strong> Search for specific help topics</li>
              <li><strong>Section Navigation:</strong> Browse help content by categories</li>
              <li><strong>Expand/Collapse:</strong> Manage content visibility with toggle controls</li>
              <li><strong>System Documentation:</strong> Detailed information about system architecture and features</li>
              <li><strong>Troubleshooting:</strong> FAQ section for common issues</li>
            </ul>
            
            <h4>Usage:</h4>
            <ol>
              <li>Use the search bar to find specific help topics</li>
              <li>Navigate through sections using the sidebar menu</li>
              <li>Click section headers to expand or collapse content</li>
              <li>Use "Expand All" or "Collapse All" for quick content management</li>
            </ol>
          </div>
          
          <!-- Cleanup System -->
          <div id="cleanup-system" class="page-docs">
            <h3>Cleanup System</h3>
            <p>The Self Soul AGI System includes a comprehensive cleanup system for managing logs, conversation history, and system maintenance.</p>
            
            <h4>Key Features:</h4>
            <ul>
              <li><strong>Cross-platform Log Cleanup Scripts:</strong> Automated scripts for Windows (PowerShell/batch), Linux, and macOS</li>
              <li><strong>Browser-based Conversation History Clearance:</strong> Web tool for clearing localStorage conversation data</li>
              <li><strong>Automatic Log Rotation:</strong> Configurable retention policies for log files</li>
              <li><strong>System Reset Capabilities:</strong> Complete system reset options for troubleshooting</li>
              <li><strong>Scheduled Maintenance:</strong> Automated cleanup scheduling for regular maintenance</li>
            </ul>
            
            <h4>Cleaning Conversation History:</h4>
            <p>To clear browser-stored conversation data:</p>
            <ol>
              <li><strong>Using the web tool</strong> (Recommended):
                <ul>
                  <li>Open <code>clear_conversation.html</code> in your browser</li>
                  <li>Click "Clear Conversation History"</li>
                  <li>Click "Back to Self Soul" to return to the application</li>
                </ul>
              </li>
              <li><strong>Using browser developer tools</strong>:
                <ul>
                  <li>Press F12 → Application/Storage tab</li>
                  <li>Delete: <code>self_soul_conversation_history</code>, <code>chat_messages</code>, and all <code>self_soul_*</code> keys</li>
                  <li>Refresh the page</li>
                </ul>
              </li>
              <li><strong>Using JavaScript console</strong>:
                <pre>localStorage.removeItem('self_soul_conversation_history');
localStorage.removeItem('chat_messages');
Object.keys(localStorage).forEach(key => {
  if (key.startsWith('self_soul_')) localStorage.removeItem(key);
});
location.reload();</pre>
              </li>
            </ol>
            
            <h4>Cleaning Log Files:</h4>
            <p>To remove old log files and free up disk space:</p>
            <ul>
              <li><strong>Windows (PowerShell)</strong>: <code>powershell -ExecutionPolicy Bypass -File cleanup_logs.ps1</code></li>
              <li><strong>Windows (Command Prompt)</strong>: <code>cleanup_logs.bat</code></li>
              <li><strong>Linux/Mac</strong>: <code>chmod +x cleanup_logs.sh &amp;&amp; ./cleanup_logs.sh</code></li>
            </ul>
            
            <h4>Complete System Reset:</h4>
            <p>For a complete system reset (nuclear option):</p>
            <ol>
              <li>Stop all services: <code>docker-compose down</code> or kill Python processes</li>
              <li>Delete all log files using the cleanup scripts</li>
              <li>Clear browser data as described above</li>
              <li>Optional: Delete database and uploaded files: <code>rm -rf data/self_soul.db uploads/* models/*</code></li>
              <li>Restart services: <code>docker-compose up -d</code> or start manually</li>
            </ol>
            
            <h4>Automated Cleanup Scheduling:</h4>
            <p>For regular maintenance, schedule automated cleanup:</p>
            <ul>
              <li><strong>Windows Task Scheduler</strong>: Schedule <code>cleanup_logs.bat</code> to run weekly</li>
              <li><strong>Linux/Mac cron job</strong>: <code>0 3 * * 0 /path/to/Self-Soul/cleanup_logs.sh</code> (runs every Sunday at 3 AM)</li>
              <li><strong>Docker log rotation</strong>: Configure in <code>docker-compose.yml</code> with max-size and max-file options</li>
            </ul>
            
            <h4>Verification:</h4>
            <p>After cleanup, verify:</p>
            <ol>
              <li>Log files are deleted: <code>ls -la logs/*.log</code> (should show no files)</li>
              <li>LocalStorage is clear: Open <code>clear_conversation.html</code> (should show "No conversation history")</li>
              <li>System works correctly: Start services and test basic functionality</li>
            </ol>
          </div>
          
          <!-- Docker Deployment -->
          <div id="docker-deployment" class="page-docs">
            <h3>Docker Deployment</h3>
            <p>Self Soul provides enhanced Docker deployment with multi-stage builds, production-ready configurations, and comprehensive orchestration.</p>
            
            <h4>Docker Compose Deployment (Recommended):</h4>
            <p>The easiest way to deploy Self Soul is using Docker Compose:</p>
            <pre># Start all services (backend, frontend, and proxy)
docker-compose up -d

# Check service status
docker-compose ps

# View logs for all services
docker-compose logs -f

# View logs for specific service
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop all services
docker-compose down

# Stop and remove volumes (complete cleanup)
docker-compose down -v</pre>
            
            <h4>Containerized Services:</h4>
            <p>The Docker Compose configuration includes:</p>
            <ul>
              <li><strong>Backend Service</strong> (<code>backend</code>):
                <ul>
                  <li>Python FastAPI application</li>
                  <li>Exposes ports: 8000 (main API), 8766 (realtime stream), 8001-8027 (model services)</li>
                  <li>Multi-stage build for optimized image size</li>
                  <li>Health checks and monitoring</li>
                </ul>
              </li>
              <li><strong>Frontend Service</strong> (<code>frontend</code>):
                <ul>
                  <li>Vue.js 3 application served by Nginx</li>
                  <li>Exposes port: 5175 (mapped to container port 80)</li>
                  <li>Production-ready Nginx configuration with security headers</li>
                  <li>API proxy configuration to backend service</li>
                </ul>
              </li>
              <li><strong>Network Configuration</strong>:
                <ul>
                  <li>Dedicated bridge network for service communication</li>
                  <li>Internal DNS resolution using service names</li>
                </ul>
              </li>
            </ul>
            
            <h4>Manual Docker Image Building:</h4>
            <p>If you prefer to build images separately:</p>
            <pre># Build backend image using multi-stage Dockerfile
docker build -t self-soul-backend -f Dockerfile.backend .

# Build frontend image with Nginx serving
docker build -t self-soul-frontend -f Dockerfile.frontend .

# Run backend container with volume mounts
docker run -d \
  -p 8000:8000 \
  -p 8766:8766 \
  -p 8001-8027:8001-8027 \
  -v ./data:/app/data \
  -v ./uploads:/app/uploads \
  -v ./logs:/app/logs \
  --name self-soul-backend \
  self-soul-backend

# Run frontend container
docker run -d \
  -p 5175:80 \
  --name self-soul-frontend \
  self-soul-frontend</pre>
            
            <h4>Docker Configuration Files:</h4>
            <ul>
              <li><code>Dockerfile.backend</code>: Multi-stage Python backend with OpenCV, PyTorch, and all dependencies</li>
              <li><code>Dockerfile.frontend</code>: Vue.js frontend with Nginx for production serving</li>
              <li><code>docker-compose.yml</code>: Complete orchestration for backend and frontend services</li>
              <li><code>nginx.docker.conf</code>: Production Nginx configuration with security headers and API proxying</li>
            </ul>
            
            <h4>Persistent Storage:</h4>
            <p>Docker Compose is configured with the following volume mounts:</p>
            <ul>
              <li><code>./data:/app/data</code>: Database and knowledge files</li>
              <li><code>./uploads:/app/uploads</code>: File uploads directory</li>
              <li><code>./logs:/app/logs</code>: Application logs</li>
              <li><code>./models:/app/models</code>: Model cache (optional)</li>
            </ul>
            
            <h4>Health Monitoring:</h4>
            <p>All containers include health checks:</p>
            <pre v-pre># Check container health status
docker inspect --format='{{.State.Health.Status}}' self-soul-backend

# View health check logs
docker logs self-soul-backend 2>&1 | grep -i health</pre>
            
            <h4>Production Deployment Notes:</h4>
            <ul>
              <li>For production, use environment-specific <code>.env</code> files</li>
              <li>Configure SSL/TLS certificates for secure communication</li>
              <li>Set up monitoring and alerting for production services</li>
              <li>Implement backup strategies for persistent volumes</li>
              <li>Configure resource limits for containers in production</li>
            </ul>
          </div>
          

        </div>
      </section>

      <!-- Real-time Spatial Recognition & Robot Control -->
      <section id="spatial-recognition" class="help-section">
        <div class="section-header" @click="toggleSection('spatialRecognition')">
          <h2>Real-time Spatial Recognition & Robot Control</h2>
          <span class="toggle-icon">{{ sectionExpanded.spatialRecognition ? '−' : '+' }}</span>
        </div>
        <div v-show="sectionExpanded.spatialRecognition" class="section-content">
          <div class="page-docs">
            <h3>Real-time Spatial Recognition</h3>
            <p>Self Soul provides advanced real-time spatial recognition capabilities using dual-camera input for depth perception and 3D spatial awareness.</p>
            
            <h4>Key Features:</h4>
            <ul>
              <li><strong>Dual-camera Stereo Vision:</strong> Real-time depth map generation using calibrated stereo camera pairs</li>
              <li><strong>3D Point Cloud Generation:</strong> Creation of 3D point clouds from depth data for spatial modeling</li>
              <li><strong>Object Localization:</strong> Precise 3D positioning of objects in the environment</li>
              <li><strong>Depth Map Preview:</strong> Real-time visualization of depth maps and spatial data</li>
              <li><strong>Spatial Obstacle Detection:</strong> Identification and mapping of obstacles in 3D space</li>
              <li><strong>Distance Measurement:</strong> Accurate distance calculations to objects and surfaces</li>
              <li><strong>Volume Estimation:</strong> Calculation of object volumes and spatial occupancy</li>
            </ul>
            
            <h4>Robot Control Integration:</h4>
            <p>Spatial recognition data is directly integrated with robot control systems for autonomous operation:</p>
            <ul>
              <li><strong>Motion Planning:</strong> Path planning and navigation using real-time spatial data</li>
              <li><strong>Obstacle Avoidance:</strong> Dynamic obstacle avoidance based on depth perception</li>
              <li><strong>Target Navigation:</strong> Precise movement to target positions in 3D space</li>
              <li><strong>Manipulation Control:</strong> Robot arm control for object manipulation using spatial coordinates</li>
              <li><strong>Multi-sensor Fusion:</strong> Integration of spatial data with other sensor inputs (IMU, LiDAR, etc.)</li>
              <li><strong>Real-time Feedback:</strong> Continuous spatial feedback for adaptive robot control</li>
            </ul>
            
            <h4>Hardware Requirements:</h4>
            <ul>
              <li><strong>Stereo Camera Pair:</strong> Two calibrated cameras with known baseline distance</li>
              <li><strong>Minimum Resolution:</strong> 640x480 per camera (higher resolution recommended)</li>
              <li><strong>Synchronization:</strong> Hardware or software synchronization for simultaneous capture</li>
              <li><strong>Calibration:</strong> Pre-calibrated camera pair with intrinsic/extrinsic parameters</li>
              <li><strong>Processing Power:</strong> Modern CPU or GPU for real-time depth map computation</li>
            </ul>
            
            <h4>Usage:</h4>
            <ol>
              <li>Navigate to <code>{{ frontendUrl }}/#/</code> and select "Real-time Input"</li>
              <li>Connect two calibrated cameras for stereo vision</li>
              <li>Enable "Dual Camera Spatial Recognition" mode</li>
              <li>View real-time depth map preview and 3D point cloud visualization</li>
              <li>Configure robot control parameters in the Robot Settings interface</li>
              <li>Test spatial recognition with sample objects and verify depth accuracy</li>
              <li>Integrate with robot hardware for autonomous navigation and manipulation</li>
            </ol>
            
            <h4>Technical Details:</h4>
            <ul>
              <li><strong>Depth Algorithm:</strong> Semi-Global Block Matching (SGBM) for dense depth estimation</li>
              <li><strong>Processing Rate:</strong> Up to 30 FPS depth map generation (depends on hardware)</li>
              <li><strong>Accuracy:</strong> Sub-centimeter accuracy at 1-3 meter range</li>
              <li><strong>Calibration Method:</strong> Chessboard pattern calibration for camera parameters</li>
              <li><strong>Output Formats:</strong> Depth maps, point clouds, obstacle maps, navigation paths</li>
              <li><strong>Integration API:</strong> REST and WebSocket endpoints for spatial data access</li>
            </ul>
            
            <h4>Applications:</h4>
            <ul>
              <li><strong>Autonomous Robotics:</strong> Navigation, manipulation, and task execution</li>
              <li><strong>Industrial Automation:</strong> Quality inspection, measurement, and assembly</li>
              <li><strong>Augmented Reality:</strong> Spatial mapping and object placement</li>
              <li><strong>Security & Surveillance:</strong> Intrusion detection and perimeter monitoring</li>
              <li><strong>Research & Education:</strong> Computer vision and robotics research</li>
              <li><strong>Assistive Technology:</strong> Navigation aids for visually impaired</li>
            </ul>
          </div>
        </div>
      </section>

      <!-- Voice Recognition & Video Dialogue -->
      <section id="voice-recognition" class="help-section">
        <div class="section-header" @click="toggleSection('voiceRecognition')">
          <h2>Voice Recognition & Video Dialogue</h2>
          <span class="toggle-icon">{{ sectionExpanded.voiceRecognition ? '−' : '+' }}</span>
        </div>
        <div v-show="sectionExpanded.voiceRecognition" class="section-content">
          <div class="page-docs">
            <h3>Voice Recognition & Video Dialogue</h3>
            <p>Self Soul provides advanced voice recognition and video dialogue capabilities for natural, multimodal interaction with the AGI system.</p>
            
            <h4>Voice Recognition Features:</h4>
            <ul>
              <li><strong>Real-time Speech Recognition:</strong> Browser-based Web Speech API integration with multilingual support</li>
              <li><strong>Audio Processing Model Integration:</strong> Direct integration with Audio Model (Port 8005) for enhanced audio analysis</li>
              <li><strong>Voice Command Processing:</strong> Natural language command recognition for system control and interaction</li>
              <li><strong>English-only Command Support:</strong> Robot voice commands now support English only for improved recognition accuracy</li>
              <li><strong>Speaker Identification:</strong> Basic speaker differentiation and voice profile recognition</li>
              <li><strong>Audio Feedback:</strong> Text-to-speech responses for natural conversational flow</li>
              <li><strong>Noise Cancellation:</strong> Background noise reduction for improved accuracy</li>
              <li><strong>Emotion Detection:</strong> Voice tone analysis for emotional context understanding</li>
            </ul>
            
            <h4>Video Dialogue Features:</h4>
            <ul>
              <li><strong>Real-time Video Processing:</strong> Live camera feed analysis with object recognition and tracking</li>
              <li><strong>Visual Model Integration:</strong> Integration with Vision Model (Port 8004) and Computer Vision Model (Port 8011)</li>
              <li><strong>Multi-camera Support:</strong> Simultaneous processing of multiple camera inputs</li>
              <li><strong>Object Recognition:</strong> Real-time identification of objects, people, and scenes</li>
              <li><strong>Gesture Recognition:</strong> Basic hand gesture recognition for non-verbal interaction</li>
              <li><strong>Facial Expression Analysis:</strong> Emotion detection from facial expressions</li>
              <li><strong>Visual Context Understanding:</strong> Scene comprehension and contextual awareness</li>
              <li><strong>Video Feedback Integration:</strong> Visual responses and annotations in video dialogue</li>
            </ul>
            
            <h4>Audio-Visual Fusion:</h4>
            <p>The system integrates audio and visual inputs for comprehensive multimodal interaction:</p>
            <ul>
              <li><strong>Synchronized Processing:</strong> Time-synchronized audio and video analysis</li>
              <li><strong>Cross-modal Validation:</strong> Audio and visual data validation for improved accuracy</li>
              <li><strong>Contextual Enrichment:</strong> Visual context enhances speech understanding, audio context enhances visual interpretation</li>
              <li><strong>Unified Response Generation:</strong> Combined audio-visual inputs produce more accurate and contextual responses</li>
            </ul>
            
            <h4>Hardware Requirements:</h4>
            <ul>
              <li><strong>Microphone:</strong> Quality microphone for clear audio capture (USB or built-in)</li>
              <li><strong>Webcam/Camera:</strong> Standard webcam or USB camera for video input</li>
              <li><strong>Audio Output:</strong> Speakers or headphones for audio responses</li>
              <li><strong>Processing Power:</strong> Modern CPU for real-time audio-visual processing</li>
              <li><strong>Browser Support:</strong> Modern browsers with Web Speech API and getUserMedia support (Chrome, Edge, Firefox)</li>
            </ul>
            
            <h4>Usage:</h4>
            <ol>
              <li>Navigate to <code>{{ frontendUrl }}/#/</code> and select "Voice Input" for speech recognition</li>
              <li>Grant microphone permissions when prompted by the browser</li>
              <li>Speak naturally - the system will transcribe your speech to text</li>
              <li>For video dialogue, select "Video Dialogue" and grant camera permissions</li>
              <li>Position yourself in frame - the system will analyze video in real-time</li>
              <li>Combine voice and video for multimodal interaction</li>
              <li>Use voice commands for system control: "show me the camera feed", "start training", etc.</li>
            </ol>
            
            <h4>Technical Details:</h4>
            <ul>
              <li><strong>Speech Recognition Engine:</strong> Web Speech API (browser-native) with fallback to server-side processing</li>
              <li><strong>Video Processing:</strong> OpenCV-based real-time video analysis with object detection</li>
              <li><strong>Audio Processing:</strong> Web Audio API for audio capture and processing</li>
              <li><strong>Integration Ports:</strong> Audio Model (8005), Vision Model (8004), Computer Vision Model (8011)</li>
              <li><strong>Latency:</strong> Near real-time processing with <500ms response time</li>
              <li><strong>Supported Languages:</strong> English-only support for voice commands, with multilingual speech recognition available</li>
              <li><strong>API Endpoints:</strong> REST endpoints for audio processing and WebSocket for real-time video streams</li>
            </ul>
            
            <h4>Applications:</h4>
            <ul>
              <li><strong>Natural Human-AGI Interaction:</strong> Conversational interfaces with multimodal understanding</li>
              <li><strong>Accessibility:</strong> Voice-controlled interfaces for users with mobility challenges</li>
              <li><strong>Education & Training:</strong> Interactive learning with voice and visual feedback</li>
              <li><strong>Remote Assistance:</strong> Visual guidance with voice explanations</li>
              <li><strong>Content Creation:</strong> Voice-controlled content generation and editing</li>
              <li><strong>Smart Environment Control:</strong> Voice and gesture-based control of connected devices</li>
            </ul>
          </div>
        </div>
      </section>

      <!-- Advanced Configuration and Usage Examples -->
      <section id="advanced-configuration" class="help-section">
        <div class="section-header" @click="toggleSection('advancedConfiguration')">
          <h2>Advanced Configuration & Usage Examples</h2>
          <span class="toggle-icon">{{ sectionExpanded.advancedConfiguration ? '−' : '+' }}</span>
        </div>
        <div v-show="sectionExpanded.advancedConfiguration" class="section-content">
          <p>This section provides advanced configuration options and practical usage examples to help you get the most out of the Self Soul AGI system.</p>
          
          <div class="advanced-config">
            <h3>API Integration Examples</h3>
            <p>The Self Soul system provides comprehensive RESTful API and WebSocket endpoints for integration with other applications. Below are practical examples of how to use the API:</p>
            
            <div class="api-example">
              <h4>Example 1: Chat with Language Model via REST API</h4>
              <pre class="code-block">
# Using curl to chat with the language model
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain the concept of neural networks",
    "session_id": "test_session_001",
    "model_id": "language"
  }'</pre>
              <p><strong>Expected Response:</strong></p>
              <pre class="code-block">
{
  "status": "success",
  "data": {
    "response": "Neural networks are computational models inspired by biological neural networks...",
    "conversation_history": [
      {"role": "user", "content": "Explain the concept of neural networks"},
      {"role": "assistant", "content": "Neural networks are computational models..."}
    ],
    "session_id": "test_session_001"
  }
}</pre>
            </div>
            
            <div class="api-example">
              <h4>Example 2: Start Training via API</h4>
              <pre class="code-block">
# Start training for language model
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "language",
    "dataset_id": "multimodal_v1",
    "parameters": {
      "epochs": 10,
      "batch_size": 32,
      "learning_rate": 0.001
    },
    "strategy": "standard_training"
  }'</pre>
            </div>
            
            <div class="api-example">
              <h4>Example 3: Real-time Training Progress via WebSocket</h4>
              <pre class="code-block">
// JavaScript WebSocket client example
const ws = new WebSocket('ws://localhost:8000/ws/training/training_job_123');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Training progress:', data);
  
  if (data.type === 'progress') {
    console.log(`Epoch: ${data.current_epoch}/${data.total_epochs}`);
    console.log(`Loss: ${data.loss}, Accuracy: ${data.accuracy}`);
  }
  
  if (data.type === 'completed') {
    console.log('Training completed!', data.results);
    ws.close();
  }
};</pre>
            </div>
          </div>
          
          <div class="performance-optimization">
            <h3>Performance Optimization Guide</h3>
            <p>Optimize your Self Soul system performance with these configuration tips:</p>
            
            <div class="optimization-tips">
              <h4>Memory Optimization</h4>
              <ul>
                <li><strong>Model Loading Strategy:</strong> Configure models to load on-demand instead of all at startup</li>
                <li><strong>Cache Management:</strong> Adjust cache size in <code>config/performance.yml</code> based on available RAM</li>
                <li><strong>Batch Size Tuning:</strong> Optimize batch sizes for your GPU memory capacity</li>
                <li><strong>Model Pruning:</strong> Use model pruning techniques for larger models to reduce memory footprint</li>
              </ul>
              
              <h4>Processing Speed Optimization</h4>
              <ul>
                <li><strong>Parallel Processing:</strong> Enable multi-threading for models that support parallel execution</li>
                <li><strong>GPU Acceleration:</strong> Ensure CUDA is properly configured for models with GPU support</li>
                <li><strong>Input Batching:</strong> Batch multiple requests together for more efficient processing</li>
                <li><strong>Model Quantization:</strong> Use quantized models for faster inference with minimal accuracy loss</li>
              </ul>
              
              <h4>Network Optimization</h4>
              <ul>
                <li><strong>WebSocket Compression:</strong> Enable compression for real-time data streams</li>
                <li><strong>API Response Caching:</strong> Cache frequent API responses to reduce backend load</li>
                <li><strong>Connection Pooling:</strong> Configure connection pools for database and external API calls</li>
                <li><strong>Load Balancing:</strong> Distribute load across multiple instances for high-traffic deployments</li>
              </ul>
            </div>
            
            <div class="config-file-example">
              <h4>Example Performance Configuration</h4>
              <pre class="code-block">
# config/performance.yml
system:
  max_workers: 4
  memory_limit_gb: 8
  cache_size_mb: 1024
  
models:
  loading_strategy: "on_demand"
  gpu_acceleration: true
  quantization_enabled: false
  batch_size_optimization: true
  
api:
  response_cache_ttl: 300  # 5 minutes
  compression_enabled: true
  connection_pool_size: 10
  
training:
  max_concurrent_sessions: 2
  checkpoint_frequency: 5  # epochs
  early_stopping_patience: 10</pre>
            </div>
          </div>
          
          <div class="integration-examples">
            <h3>System Integration Examples</h3>
            <p>Learn how to integrate Self Soul with other systems and applications:</p>
            
            <div class="integration-example">
              <h4>Integration with External Monitoring Systems</h4>
              <pre class="code-block">
# Python example: Monitor system health and send alerts
import requests
import time
from prometheus_client import start_http_server, Gauge

# Prometheus metrics
model_health = Gauge('self_soul_model_health', 'Health status of Self Soul models')
system_memory = Gauge('self_soul_system_memory', 'System memory usage in MB')

def monitor_self_soul():
    while True:
        try:
            # Get system health status
            response = requests.get('http://localhost:8000/api/health/detailed')
            data = response.json()
            
            # Update metrics
            model_health.set(data['models']['healthy_count'])
            system_memory.set(data['system']['memory_used_mb'])
            
            # Check for issues
            if data['status'] != 'ok':
                send_alert(f"Self Soul system issue: {data['message']}")
                
        except Exception as e:
            send_alert(f"Monitoring error: {str(e)}")
            
        time.sleep(60)  # Check every minute

if __name__ == '__main__':
    start_http_server(9090)
    monitor_self_soul()</pre>
            </div>
            
            <div class="integration-example">
              <h4>Integration with External Data Sources</h4>
              <pre class="code-block">
# Example: Automatically import data from external sources
import requests
import json
from datetime import datetime

def import_external_knowledge(source_url, api_key=None):
    """Import knowledge from external API sources"""
    
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    # Fetch data from external source
    response = requests.get(source_url, headers=headers)
    external_data = response.json()
    
    # Format for Self Soul knowledge base
    knowledge_entry = {
        "title": f"Imported from {source_url}",
        "content": json.dumps(external_data, indent=2),
        "domain": "external_data",
        "import_date": datetime.now().isoformat(),
        "source": source_url
    }
    
    # Import into Self Soul
    import_response = requests.post(
        'http://localhost:8000/api/knowledge/import',
        json=knowledge_entry
    )
    
    return import_response.json()

# Usage example
result = import_external_knowledge(
    source_url='https://api.external-data.com/latest',
    api_key='your_api_key_here'
)
print(f"Import result: {result}")</pre>
            </div>
          </div>
          
          <div class="deployment-examples">
            <h3>Advanced Deployment Scenarios</h3>
            <p>Explore advanced deployment configurations for different use cases:</p>
            
            <div class="deployment-scenario">
              <h4>High-Availability Deployment</h4>
              <p>For mission-critical applications requiring 99.9% uptime:</p>
              <ol>
                <li>Deploy multiple Self Soul instances behind a load balancer</li>
                <li>Use shared storage for model checkpoints and knowledge base</li>
                <li>Implement health checks and automatic failover</li>
                <li>Configure database replication for state persistence</li>
                <li>Set up monitoring and alerting for all components</li>
              </ol>
              
              <h4>Edge Computing Deployment</h4>
              <p>For scenarios with limited connectivity or privacy requirements:</p>
              <ol>
                <li>Deploy lightweight models optimized for edge devices</li>
                <li>Configure periodic synchronization with central system</li>
                <li>Implement data filtering for privacy-sensitive information</li>
                <li>Optimize for low-power operation and intermittent connectivity</li>
                <li>Use local storage with encryption for sensitive data</li>
              </ol>
              
              <h4>Hybrid Cloud Deployment</h4>
              <p>For balancing cost, performance, and privacy requirements:</p>
              <ol>
                <li>Run sensitive models locally for privacy</li>
                <li>Use cloud APIs for compute-intensive or specialized tasks</li>
                <li>Implement intelligent routing based on data sensitivity and task requirements</li>
                <li>Configure automatic failover between local and cloud models</li>
                <li>Use encryption for all data transfers between environments</li>
              </ol>
            </div>
          </div>
          
          <div class="best-practices">
            <h3>Best Practices for Production Use</h3>
            <ul>
              <li><strong>Regular Backups:</strong> Schedule automated backups of model checkpoints, knowledge base, and configuration files</li>
              <li><strong>Security Hardening:</strong> Implement firewalls, rate limiting, and authentication for all API endpoints</li>
              <li><strong>Monitoring Setup:</strong> Deploy comprehensive monitoring with Prometheus, Grafana, and alerting</li>
              <li><strong>Capacity Planning:</strong> Monitor resource usage trends and plan for scaling before reaching limits</li>
              <li><strong>Disaster Recovery:</strong> Create and regularly test disaster recovery procedures</li>
              <li><strong>Documentation Maintenance:</strong> Keep system documentation updated with all configuration changes</li>
              <li><strong>Regular Updates:</strong> Schedule regular updates for dependencies and security patches</li>
              <li><strong>Performance Testing:</strong> Conduct regular performance testing and optimization</li>
            </ul>
          </div>
        </div>
      </section>

      <!-- API Documentation Guide -->
      <section id="api-documentation" class="help-section">
        <div class="section-header" @click="toggleSection('apiDocumentation')">
          <h2>API Documentation Guide</h2>
          <span class="toggle-icon">{{ sectionExpanded.apiDocumentation ? '−' : '+' }}</span>
        </div>
        <div v-show="sectionExpanded.apiDocumentation" class="section-content">
          <p>The Self Soul system provides comprehensive API documentation through Swagger UI at <a href="http://localhost:8000/docs" target="_blank">http://localhost:8000/docs</a>. This interactive documentation allows you to explore, test, and understand all available API endpoints.</p>
          
          <div class="api-doc-guide">
            <h3>Accessing the API Documentation</h3>
            <p>To access the interactive API documentation:</p>
            <ol>
              <li>Ensure the Self Soul backend server is running on port 8000</li>
              <li>Open your web browser and navigate to <code>http://localhost:8000/docs</code></li>
              <li>The Swagger UI interface will load, displaying all available API endpoints</li>
            </ol>
            
            <div class="swagger-features">
              <h4>Swagger UI Features</h4>
              <div class="features-grid">
                <div class="feature-card">
                  <h5>Interactive Endpoint Exploration</h5>
                  <p>Click on any endpoint to expand and view detailed information about parameters, request formats, and response schemas.</p>
                </div>
                <div class="feature-card">
                  <h5>Live API Testing</h5>
                  <p>Test API endpoints directly from the documentation by entering parameters and clicking "Execute" to see real responses.</p>
                </div>
                <div class="feature-card">
                  <h5>Authentication Configuration</h5>
                  <p>Configure API keys and authentication headers directly in the Swagger UI for testing protected endpoints.</p>
                </div>
                <div class="feature-card">
                  <h5>Response Schema Documentation</h5>
                  <p>View detailed JSON schemas for all response types, including example values and data structures.</p>
                </div>
              </div>
            </div>
            
            <div class="api-categories">
              <h3>API Categories</h3>
              <p>The Self Soul API is organized into logical categories for easy navigation:</p>
              
              <div class="category-list">
                <div class="category-item">
                  <h4>System Health & Status</h4>
                  <ul>
                    <li><code>GET /health</code> - Basic system health check</li>
                    <li><code>GET /api/health/detailed</code> - Detailed system status</li>
                    <li><code>GET /api/models/status</code> - Model status information</li>
                  </ul>
                </div>
                
                <div class="category-item">
                  <h4>Model Management</h4>
                  <ul>
                    <li><code>GET /api/models/getAll</code> - Get all model information</li>
                    <li><code>GET /api/models/config</code> - Get model configurations</li>
                    <li><code>POST /api/models/batch/switch</code> - Batch switch model modes</li>
                  </ul>
                </div>
                
                <div class="category-item">
                  <h4>Conversation & Processing</h4>
                  <ul>
                    <li><code>POST /api/chat</code> - Chat with language model</li>
                    <li><code>POST /api/process/text</code> - Process text input</li>
                    <li><code>POST /api/models/8001/chat</code> - Chat with manager model</li>
                  </ul>
                </div>
                
                <div class="category-item">
                  <h4>Training & Learning</h4>
                  <ul>
                    <li><code>POST /api/training/start</code> - Start model training</li>
                    <li><code>GET /api/training/status/{job_id}</code> - Check training status</li>
                    <li><code>POST /api/autonomous-learning/start</code> - Start autonomous learning</li>
                  </ul>
                </div>
                
                <div class="category-item">
                  <h4>Knowledge Management</h4>
                  <ul>
                    <li><code>POST /api/knowledge/import</code> - Import knowledge</li>
                    <li><code>GET /api/knowledge/search</code> - Search knowledge base</li>
                    <li><code>GET /api/knowledge/stats</code> - Get knowledge statistics</li>
                  </ul>
                </div>
                
                <div class="category-item">
                  <h4>Device & Hardware Control</h4>
                  <ul>
                    <li><code>GET /api/devices/cameras</code> - Get available cameras</li>
                    <li><code>GET /api/serial/ports</code> - Get serial ports</li>
                    <li><code>POST /api/serial/connect</code> - Connect to serial device</li>
                  </ul>
                </div>
              </div>
            </div>
            
            <div class="api-usage-examples">
              <h3>Practical API Usage Examples</h3>
              
              <div class="example-item">
                <h4>Example 1: Checking System Health</h4>
                <pre class="code-block">
# Using curl
curl -X GET http://localhost:8000/health

# Expected response:
{
  "status": "ok",
  "message": "Self Soul system is running normally"
}</pre>
                <div class="example-actions">
                  <button @click="runCodeExample('health-check')" class="run-example-btn">
                    Run This Example
                  </button>
                  <button @click="copyCodeToClipboard('health-check')" class="copy-code-btn">
                    Copy Code
                  </button>
                </div>
              </div>
              
              <div class="example-item">
                <h4>Example 2: Starting a Chat Session</h4>
                <pre class="code-block">
# Using curl with JSON payload
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how does the Self Soul system work?",
    "session_id": "user_123_session",
    "model_id": "language"
  }'</pre>
                <div class="example-actions">
                  <button @click="runCodeExample('chat-session')" class="run-example-btn">
                    Run This Example
                  </button>
                  <button @click="copyCodeToClipboard('chat-session')" class="copy-code-btn">
                    Copy Code
                  </button>
                </div>
              </div>
              
              <div class="example-item">
                <h4>Example 3: Starting Model Training</h4>
                <pre class="code-block">
# Start training for language model
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "language",
    "dataset_id": "multimodal_v1",
    "parameters": {
      "epochs": 10,
      "batch_size": 32,
      "learning_rate": 0.001
    },
    "strategy": "standard_training"
  }'</pre>
                <div class="example-actions">
                  <button @click="runCodeExample('training-start')" class="run-example-btn">
                    Run This Example
                  </button>
                  <button @click="copyCodeToClipboard('training-start')" class="copy-code-btn">
                    Copy Code
                  </button>
                </div>
              </div>
            </div>
            
            <div class="websocket-endpoints">
              <h3>WebSocket Endpoints</h3>
              <p>The system also provides real-time WebSocket endpoints for live data streaming:</p>
              
              <table>
                <thead>
                  <tr>
                    <th>Endpoint</th>
                    <th>Description</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><code>ws://localhost:8000/ws/training/{job_id}</code></td>
                    <td>Real-time training progress updates</td>
                  </tr>
                  <tr>
                    <td><code>ws://localhost:8000/ws/monitoring</code></td>
                    <td>System monitoring data stream</td>
                  </tr>
                  <tr>
                    <td><code>ws://localhost:8000/ws/autonomous-learning/status</code></td>
                    <td>Autonomous learning status updates</td>
                  </tr>
                  <tr>
                    <td><code>ws://localhost:8000/ws/audio-stream</code></td>
                    <td>Real-time audio stream processing</td>
                  </tr>
                  <tr>
                    <td><code>ws://localhost:8000/ws/video-stream</code></td>
                    <td>Real-time video stream processing</td>
                  </tr>
                </tbody>
              </table>
            </div>
            
            <div class="swagger-tips">
              <h3>Tips for Using Swagger UI</h3>
              <ul>
                <li><strong>Try It Out:</strong> Use the "Try it out" button next to each endpoint to test APIs directly in the browser</li>
                <li><strong>Authentication:</strong> Click the "Authorize" button at the top to set API keys for endpoints requiring authentication</li>
                <li><strong>Model Selection:</strong> Some endpoints require specific model IDs - refer to the model documentation for valid IDs</li>
                <li><strong>Response Codes:</strong> Each endpoint documents possible HTTP response codes and their meanings</li>
                <li><strong>Schemas:</strong> Expand the "Schema" section to see detailed request/response structures with example values</li>
                <li><strong>Parameters:</strong> Required parameters are marked with a red asterisk (*)</li>
              </ul>
            </div>
            
            <div class="api-security">
              <h3>API Security & Authentication</h3>
              <p>The Self Soul API supports multiple authentication methods:</p>
              
              <ul>
                <li><strong>API Key Authentication:</strong> Most endpoints accept API keys passed in the <code>X-API-Key</code> header</li>
                <li><strong>Session-based Authentication:</strong> Some endpoints use session cookies for user-specific operations</li>
                <li><strong>Rate Limiting:</strong> API endpoints are rate-limited to prevent abuse</li>
                <li><strong>CORS:</strong> Cross-Origin Resource Sharing is configured for web applications</li>
              </ul>
              
              <p>For development and testing, you can disable authentication by setting <code>AUTH_ENABLED=false</code> in your environment configuration.</p>
            </div>
            
            <div class="api-best-practices">
              <h3>API Best Practices</h3>
              <ol>
                <li><strong>Use Proper Error Handling:</strong> Always check HTTP status codes and handle errors gracefully</li>
                <li><strong>Implement Retry Logic:</strong> For transient failures, implement exponential backoff retry mechanisms</li>
                <li><strong>Cache Responses:</strong> Cache frequently accessed data to reduce server load and improve performance</li>
                <li><strong>Monitor Rate Limits:</strong> Track your API usage to avoid hitting rate limits</li>
                <li><strong>Use WebSockets for Real-time Data:</strong> For live updates, prefer WebSocket connections over polling</li>
                <li><strong>Validate Inputs:</strong> Always validate request parameters before sending API calls</li>
                <li><strong>Keep API Keys Secure:</strong> Never expose API keys in client-side code or public repositories</li>
              </ol>
            </div>
            
            <div class="api-troubleshooting">
              <h3>API Troubleshooting</h3>
              
              <div class="troubleshooting-tips">
                <h4>Common Issues and Solutions</h4>
                
                <div class="issue">
                  <h5>Connection Refused</h5>
                  <p><strong>Issue:</strong> Cannot connect to <code>http://localhost:8000</code></p>
                  <p><strong>Solution:</strong> Ensure the backend server is running. Check if port 8000 is available and not blocked by firewall.</p>
                </div>
                
                <div class="issue">
                  <h5>404 Not Found</h5>
                  <p><strong>Issue:</strong> API endpoint returns 404 error</p>
                  <p><strong>Solution:</strong> Verify the endpoint URL is correct. Check the Swagger UI documentation for the exact endpoint path.</p>
                </div>
                
                <div class="issue">
                  <h5>Authentication Failed</h5>
                  <p><strong>Issue:</strong> API returns 401 Unauthorized</p>
                  <p><strong>Solution:</strong> Check if API key is valid and properly formatted in the request header.</p>
                </div>
                
                <div class="issue">
                  <h5>Rate Limit Exceeded</h5>
                  <p><strong>Issue:</strong> API returns 429 Too Many Requests</p>
                  <p><strong>Solution:</strong> Implement exponential backoff and reduce request frequency.</p>
                </div>
              </div>
            </div>
            
            <div class="api-support">
              <h3>Additional Resources</h3>
              <ul>
                <li><strong>Complete API Reference:</strong> <a href="http://localhost:8000/docs" target="_blank">http://localhost:8000/docs</a></li>
                <li><strong>OpenAPI Specification:</strong> <a href="http://localhost:8000/openapi.json" target="_blank">http://localhost:8000/openapi.json</a></li>
                <li><strong>GitHub Repository:</strong> <a href="https://github.com/Sum-Outman/Self-Soul" target="_blank">https://github.com/Sum-Outman/Self-Soul</a></li>
                <li><strong>Issue Tracker:</strong> <a href="https://github.com/Sum-Outman/Self-Soul/issues" target="_blank">https://github.com/Sum-Outman/Self-Soul/issues</a></li>
              </ul>
            </div>
            
            <!-- Interactive API Tester -->
            <div class="interactive-api-tester">
              <h3>Interactive API Tester</h3>
              <p>Test API endpoints directly from this documentation:</p>
              
              <div class="api-tester-controls">
                <div class="tester-section">
                  <label for="api-endpoint-select">Select Endpoint:</label>
                  <select id="api-endpoint-select" v-model="apiTester.selectedEndpoint" class="endpoint-select">
                    <option v-for="endpoint in apiTester.endpoints" :key="endpoint.method + endpoint.path" :value="endpoint.method + ' ' + endpoint.path">
                      {{ endpoint.method }} {{ endpoint.path }} - {{ endpoint.description }}
                    </option>
                  </select>
                </div>
                
                <div v-if="apiTester.selectedEndpoint.startsWith('POST')" class="tester-section">
                  <label for="request-body">Request Body (JSON):</label>
                  <textarea 
                    id="request-body" 
                    v-model="apiTester.requestBody" 
                    rows="6" 
                    class="request-body-textarea"
                    placeholder="Enter JSON request body..."
                  ></textarea>
                </div>
                
                <div class="tester-actions">
                  <button @click="testApiEndpoint" :disabled="apiTester.loading" class="test-api-btn">
                    {{ apiTester.loading ? 'Testing...' : 'Test API Endpoint' }}
                  </button>
                  <button @click="apiTester.responseData = null; apiTester.error = null;" class="clear-btn">
                    Clear Results
                  </button>
                </div>
              </div>
              
              <!-- Results Section -->
              <div v-if="apiTester.responseData || apiTester.error" class="api-results">
                <h4>Test Results</h4>
                
                <div v-if="apiTester.responseTime" class="response-time">
                  Response time: {{ apiTester.responseTime }}ms
                </div>
                
                <div v-if="apiTester.error" class="error-result">
                  <h5>Error</h5>
                  <pre class="error-output">{{ apiTester.error }}</pre>
                </div>
                
                <div v-if="apiTester.responseData" class="success-result">
                  <div class="response-status">
                    <strong>Status:</strong> {{ apiTester.responseData.status }}
                  </div>
                  <div class="response-timestamp">
                    <strong>Timestamp:</strong> {{ new Date(apiTester.responseData.timestamp).toLocaleString() }}
                  </div>
                  <div class="response-data">
                    <h5>Response Data:</h5>
                    <pre class="json-output">{{ formatJson(apiTester.responseData.data) }}</pre>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Troubleshooting -->
      <section id="troubleshooting" class="help-section">
        <div class="section-header" @click="toggleSection('troubleshooting')">
          <h2>Troubleshooting & Support</h2>
          <span class="toggle-icon">{{ sectionExpanded.troubleshooting ? '−' : '+' }}</span>
        </div>
        <div v-show="sectionExpanded.troubleshooting" class="section-content">
        <div class="faq-list">
          <div class="faq-item">
            <h3>System Initialization Issues</h3>
            <p>Verify all dependencies are installed, check system logs for specific error messages, and ensure adequate system resources are available.</p>
          </div>
          <div class="faq-item">
            <h3>Model Performance Degradation</h3>
            <p>Check training data quality, review model configuration parameters, and consider retraining with updated datasets.</p>
          </div>
          <div class="faq-item">
            <h3>Communication Failures</h3>
            <p>Verify network connectivity, check API endpoint configurations, and ensure proper authentication credentials.</p>
          </div>
          <div class="faq-item">
            <h3>Memory Management</h3>
            <p>Monitor system memory usage, optimize data processing pipelines, and consider implementing caching strategies.</p>
          </div>
        </div>
        </div>
      </section>

      <!-- System Requirements -->
      <section id="system-requirements" class="help-section">
        <div class="section-header" @click="toggleSection('systemRequirements')">
          <h2>System Requirements</h2>
          <span class="toggle-icon">{{ sectionExpanded.systemRequirements ? '−' : '+' }}</span>
        </div>
        <div v-show="sectionExpanded.systemRequirements" class="section-content">
        <div class="requirements-list">
          <div class="requirement-item">
            <h3>Hardware</h3>
            <p>Minimum: 8GB RAM, 4-core processor, 10GB storage space<br>
            Recommended: 16GB RAM, 8-core processor, 20GB storage space (for optimal performance with all models)</p>
          </div>
          <div class="requirement-item">
            <h3>Software</h3>
            <p>Python 3.8+, Node.js 14+, modern web browser with JavaScript support<br>
            Dependencies managed through requirements.txt and package.json files</p>
          </div>
          <div class="requirement-item">
            <h3>Network</h3>
            <p>Stable internet connection for API model access and updates<br>
            Open ports: 8000-8027, 5175, 8766, 8080 for complete system functionality</p>
          </div>
          <div class="requirement-item">
            <h3>Operating System</h3>
            <p>Windows 10+, macOS 11+, or Linux (Ubuntu 20.04+, Debian 11+)<br>
            Proper file system permissions for reading/writing data files and configurations</p>
          </div>
        </div>
        </div>
      </section>
    </main>
  </div>

    <!-- Footer -->
    <footer class="help-footer">
      <p>Self Soul AGI System v2.0.0 | Apache 2.0 License | Documentation Version 1.2</p>
    </footer>
  </div>
</template>

<script>
import api from '@/utils/api';
import { performDataLoad, performDataOperation } from '../utils/operationHelpers.js';
import { notify } from '@/plugins/notification';

export default {
  name: 'HelpView',
  data() {
    return {
      isMounted: false,
      // Feature data for the component
      features: [
        {
          id: 'unified-cognitive-architecture',
          title: 'Unified Cognitive Architecture',
          description: 'Integrated system of 27 specialized models (manager, language, vision, etc.) working synergistically via dedicated ports (8001-8027), coordinated by the Manager Model (Port 8001)'
        },
        {
          id: 'adaptive-learning-engine',
          title: 'Adaptive Learning Engine',
          description: 'Backend component that optimizes training parameters (learning rate, batch size) based on model performance and data characteristics in real-time'
        },
        {
          id: 'frontend-backend-mapping',
          title: 'Frontend-Backend Model ID Mapping',
          description: 'Dual ID system where frontend uses letter IDs (A-X) for UI display and backend uses string IDs (manager, language), with automatic conversion via modelIdMapper.js'
        },
        {
          id: 'multimodal-dataset',
          title: 'Multimodal Dataset v1',
          description: 'Comprehensive dataset supporting all 27 models with formats for text, images, audio, and video data, ensuring alignment between dataset and model capabilities'
        },
        {
          id: 'websocket-communication',
          title: 'WebSocket Communication',
          description: 'Enables real-time updates for training progress, system monitoring, autonomous learning status, and audio/video stream processing'
        },
        {
          id: 'distributed-model-architecture',
          title: 'Distributed Model Architecture',
          description: 'Each model runs on a dedicated port (8001-8027) for parallel processing, allowing independent optimization and coordinated operation'
        },
        {
          id: 'autonomous-learning',
          title: 'Autonomous Learning',
          description: 'Continuous self-improvement through experience, data analysis, and intrinsic motivation system'
        },
        {
          id: 'multimodal-integration',
          title: 'Multimodal Integration',
          description: 'Seamless processing of text, images, audio, sensor inputs, and video streams for comprehensive understanding'
        },
        {
          id: 'robot-system-integration',
          title: 'Robot System Integration',
          description: 'Comprehensive hardware control, sensor management, and robot-specific training capabilities with support for motion control, perception training, and real-time hardware feedback'
        }
      ],
      // Training approaches data
      trainingApproaches: [
        {
          id: 'individual-training',
          title: 'Individual Model Training',
          description: 'Focused optimization of specific model capabilities through targeted datasets'
        },
        {
          id: 'joint-training',
          title: 'Joint Collaborative Training',
          description: 'Synchronized training across multiple models to enhance interoperability'
        },
        {
          id: 'continuous-learning',
          title: 'Continuous Learning',
          description: 'Ongoing adaptation and improvement through real-world interaction data'
        },
        {
          id: 'transfer-learning',
          title: 'Transfer Learning',
          description: 'Application of knowledge from one domain to accelerate learning in related areas'
        },
        {
          id: 'pretrained-fine-tuning',
          title: 'Pre-trained Fine-tuning',
          description: 'Leveraging existing pre-trained models as a foundation and fine-tuning them for specific tasks with configurable layer freezing and fine-tuning modes'
        }
      ],
      // Search functionality
      searchQuery: '',
      // Section expansion states
      sectionExpanded: {
        systemOverview: true,
        portsConfig: false,
        gettingStarted: true,  // default expanded for Quick Start
        coreModels: false,
        modelDetailedDocumentation: false,
        trainingMethodology: false,
        advancedCapabilities: false,
        pageFeatures: false,
        robotSettings: false,
        robotTraining: false,
        cleanupSystem: false,
        dockerDeployment: false,
        spatialRecognition: false,
        voiceRecognition: false,
        advancedConfiguration: false,
        apiDocumentation: false,
        troubleshooting: false,
        systemRequirements: false
      },
      // Matched search results
      searchResults: [],
      // Show search results instead of normal content
      showSearchResults: false,
      // Debounce timer for search
      searchDebounce: null,
      
      // Interactive API tester
      apiTester: {
        selectedEndpoint: 'GET /health',
        endpoints: [
          { method: 'GET', path: '/health', description: 'Basic system health check' },
          { method: 'GET', path: '/api/health/detailed', description: 'Detailed system status' },
          { method: 'GET', path: '/api/models/status', description: 'Model status information' },
          { method: 'GET', path: '/api/models/getAll', description: 'Get all model information' },
          { method: 'POST', path: '/api/chat', description: 'Chat with language model' },
          { method: 'POST', path: '/api/training/start', description: 'Start model training' },
          { method: 'GET', path: '/api/knowledge/files', description: 'Get knowledge files' }
        ],
        requestBody: '{}',
        responseData: null,
        loading: false,
        error: null,
        responseTime: null
      },
      
      // Real-time system status
      systemStatus: {
        loading: false,
        data: null,
        error: null,
        lastUpdated: null
      },
      
      // Interactive architecture diagram state
      architectureDiagram: {
        activeNode: null,
        nodes: [
          { id: 'manager', name: 'Manager Model', port: 8001, description: 'Coordinates all other models' },
          { id: 'language', name: 'Language Model', port: 8002, description: 'Natural language processing' },
          { id: 'vision', name: 'Vision Model', port: 8003, description: 'Image and video analysis' },
          { id: 'audio', name: 'Audio Model', port: 8004, description: 'Audio processing and recognition' },
          { id: 'sensor', name: 'Sensor Model', port: 8005, description: 'Sensor data interpretation' },
          { id: 'learning', name: 'Learning Engine', port: 8006, description: 'Adaptive learning and optimization' },
          { id: 'knowledge', name: 'Knowledge Base', port: 8007, description: 'Knowledge storage and retrieval' }
        ]
      }
    }
  },
  computed: {
    // 配置相关的计算属性
    frontendUrl() {
      return this.$getFrontendUrl ? this.$getFrontendUrl() : 'http://localhost:5175';
    },
    backendDocsUrl() {
      return this.$getBackendDocsUrl ? this.$getBackendDocsUrl() : 'http://localhost:8000/docs';
    },
    backendBaseUrl() {
      return this.$getBackendUrl ? this.$getBackendUrl() : 'http://localhost:8000';
    },
    ollamaConfig() {
      if (this.$getConfig) {
        const baseUrl = this.$getConfig('models.ollama.baseUrl', 'http://localhost:11434');
        const apiPath = this.$getConfig('models.ollama.apiPath', '/v1/chat/completions');
        return `${baseUrl}${apiPath}`;
      }
      return 'http://localhost:11434/v1/chat/completions';
    }
  },
  watch: {
    searchQuery(newQuery) {
      // Debounced search when query changes
      if (this.searchDebounce) {
        clearTimeout(this.searchDebounce);
      }
      
      if (!newQuery) {
        this.showSearchResults = false;
        this.searchResults = [];
        return;
      }
      
      this.searchDebounce = setTimeout(() => {
        if (this.isMounted) {
          this.performSearch();
        }
      }, 300);
    }
  },
  mounted() {
    this.isMounted = true;
    document.title = 'Self Soul AGI System Help';
    // Add event listener for keyboard shortcuts
    document.addEventListener('keydown', this.handleKeyDown);
    // Check system status when component mounts
    this.checkSystemStatus();
  },
  beforeUnmount() {
    this.isMounted = false;
    // Remove event listener
    document.removeEventListener('keydown', this.handleKeyDown);
    
    // Clear any pending search debounce timeout
    if (this.searchDebounce) {
      clearTimeout(this.searchDebounce);
      this.searchDebounce = null;
    }
  },
  methods: {
    // Perform search across all sections and content
    performSearch() {
      const query = this.searchQuery.toLowerCase().trim();
      
      if (!query) {
        this.searchResults = [];
        this.showSearchResults = false;
        return;
      }
      
      const results = [];
      const sections = document.querySelectorAll('.help-section');
      
      sections.forEach((section) => {
        if (!section) return;
        
        const sectionId = section.id;
        let sectionTitle = '';
        let sectionContent = '';
        
        // Get section title (h2)
        const titleElement = section.querySelector('h2');
        if (titleElement) {
          sectionTitle = titleElement.textContent.toLowerCase();
        }
        
        // Get all text content from section
        sectionContent = section.textContent.toLowerCase();
        
        // Search in title or content
        if (sectionTitle.includes(query) || sectionContent.includes(query)) {
          results.push({
            id: sectionId,
            title: titleElement ? titleElement.textContent : 'Untitled Section'
          });
        }
        
        // Also search within model cards, feature cards, etc.
        const subElements = section.querySelectorAll('.model-card, .feature-card, .feature-item, .faq-item, .requirement-item, .step-content');
        subElements.forEach((element) => {
          const elementText = element.textContent.toLowerCase();
          if (elementText.includes(query)) {
            // Find the nearest parent section id
            const parentSection = element.closest('.help-section');
            if (parentSection && parentSection.id) {
              // Check if this section is already in results
              const existingResult = results.find(r => r.id === parentSection.id);
              if (!existingResult) {
                const parentTitle = parentSection.querySelector('h2');
                results.push({
                  id: parentSection.id,
                  title: parentTitle ? parentTitle.textContent : 'Section'
                });
              }
            }
          }
        });
      });
      
      // Remove duplicates
      const uniqueResults = results.filter((result, index, self) =>
        index === self.findIndex((r) => r.id === result.id)
      );
      
      this.searchResults = uniqueResults;
      this.showSearchResults = true;
    },
    
    // Toggle section expansion
    toggleSection(sectionName) {
      this.sectionExpanded[sectionName] = !this.sectionExpanded[sectionName];
    },
    
    // Scroll to specific section
    scrollToSection(sectionId) {
      const section = document.getElementById(sectionId);
      if (section) {
        // First ensure the section is expanded
        const sectionKey = sectionId.replace(/-/g, '').replace(/^./, str => str.charAt(0).toLowerCase() + str.slice(1));
        if (this.sectionExpanded.hasOwnProperty(sectionKey)) {
          this.sectionExpanded[sectionKey] = true;
        }
        
        // Handle special cases for robot settings and training
        // These are sub-sections within "Page Features Documentation"
        if (sectionId === 'robot-settings' || sectionId === 'robot-training') {
          this.sectionExpanded.pageFeatures = true;
        }
        
        // Then scroll to it
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
        // Reset search results if any
        if (this.showSearchResults) {
          this.searchQuery = '';
          this.showSearchResults = false;
        }
      }
    },
    
    // Handle keyboard shortcuts
    handleKeyDown(event) {
      // Ctrl/Cmd + F to focus on search
      if ((event.ctrlKey || event.metaKey) && event.key === 'f') {
        event.preventDefault();
        this.$refs.searchInput.focus();
      }
    },
    
    // Method to expand all sections
    expandAll() {
      Object.keys(this.sectionExpanded).forEach(key => {
        this.sectionExpanded[key] = true;
      });
    },
    
    // Method to collapse all sections
    collapseAll() {
      Object.keys(this.sectionExpanded).forEach(key => {
        this.sectionExpanded[key] = false;
      });
    },
    
    // Helper method to perform data operations using operationHelpers
    async performDataOperation(operation, options = {}) {
      // Pass the current component context (this) to handle notifications and error handling
      const defaultOptions = {
        apiClient: api,
        notify: this.$notify || null,
        handleError: this.$handleError || null
      };
      const mergedOptions = { ...defaultOptions, ...options };
      return await performDataOperation(operation, mergedOptions);
    },
    
    // Helper method to perform data loading using operationHelpers
    async performDataLoad(operation, options = {}) {
      const defaultOptions = {
        apiClient: api,
        notify: this.$notify || null,
        handleError: this.$handleError || null
      };
      const mergedOptions = { ...defaultOptions, ...options };
      return await performDataLoad(operation, mergedOptions);
    },
    
    // Interactive API tester methods
    async testApiEndpoint() {
      const tester = this.apiTester;
      tester.loading = true;
      tester.error = null;
      tester.responseData = null;
      tester.responseTime = null;
      
      // Use generic data operation helper
      return await this.performDataOperation('api-test', {
        // Custom API call function to handle different HTTP methods
        apiCall: async () => {
          // Parse selected endpoint
          const [method, path] = tester.selectedEndpoint.split(' ');
          
          // Record start time
          const startTime = Date.now();
          let response;
          
          if (method === 'GET') {
            response = await api.get(path);
          } else if (method === 'POST') {
            let requestBody = {};
            try {
              requestBody = JSON.parse(tester.requestBody);
            } catch (parseError) {
              throw new Error('Invalid JSON in request body');
            }
            response = await api.post(path, requestBody);
          } else {
            throw new Error(`Unsupported method: ${method}`);
          }
          
          // Record response time
          const endTime = Date.now();
          tester.responseTime = endTime - startTime;
          
          // Return formatted response data
          return {
            ...response,
            formattedData: {
              status: response.status,
              data: response.data,
              headers: response.headers,
              timestamp: new Date().toISOString()
            }
          };
        },
        onSuccess: (data, fullResponse) => {
          // Set response data
          tester.responseData = fullResponse.formattedData;
        },
        onError: (error) => {
          console.error('API test failed:', error);
          tester.error = error.response?.data || error.message || 'Unknown error occurred';
          tester.responseData = error.response ? {
            status: error.response.status,
            data: error.response.data,
            headers: error.response.headers,
            timestamp: new Date().toISOString()
          } : null;
        },
        onFinally: () => {
          tester.loading = false;
        },
        successMessage: '', // Do not show success notification
        errorMessage: 'API test failed',
        errorContext: 'API Test',
        showSuccess: false,
        showError: false,
        silentError: true // Do not show error notification because we have custom error handling
      });
    },
    
    // System status methods
    async checkSystemStatus() {
      this.systemStatus.loading = true;
      this.systemStatus.error = null;
      
      // Use generic data loading helper
      return await this.performDataLoad('system-status', {
        apiCall: () => api.get('/api/health/detailed'),
        onSuccess: (data) => {
          this.systemStatus.data = data;
          this.systemStatus.lastUpdated = new Date().toISOString();
        },
        onError: (error) => {
          console.error('Failed to fetch system status:', error);
          this.systemStatus.error = error.response?.data || error.message || 'Failed to fetch system status';
        },
        onFinally: () => {
          this.systemStatus.loading = false;
        },
        successMessage: '', // Do not show success notification
        errorMessage: 'Failed to fetch system status',
        errorContext: 'System Status',
        showSuccess: false,
        showError: false,
        silentError: true // Do not show error notification because we have custom error handling
      });
    },
    
    // Architecture diagram methods
    setActiveNode(nodeId) {
      if (this.architectureDiagram.activeNode === nodeId) {
        this.architectureDiagram.activeNode = null;
      } else {
        this.architectureDiagram.activeNode = nodeId;
      }
    },
    
    getNodeDescription(nodeId) {
      const node = this.architectureDiagram.nodes.find(n => n.id === nodeId);
      return node ? node.description : 'No description available';
    },
    
    // Code example execution
    async runCodeExample(exampleId) {
      // Execute real API calls based on the example
      switch(exampleId) {
        case 'health-check':
          this.apiTester.selectedEndpoint = 'GET /health';
          this.apiTester.requestBody = '{}';
          await this.testApiEndpoint();
          break;
        case 'chat-session':
          this.apiTester.selectedEndpoint = 'POST /api/chat';
          this.apiTester.requestBody = JSON.stringify({
            message: "Hello, how does the Self Soul system work?",
            session_id: "test_session",
            model_id: "language"
          }, null, 2);
          await this.testApiEndpoint();
          break;
        case 'training-start':
          this.apiTester.selectedEndpoint = 'POST /api/training/start';
          this.apiTester.requestBody = JSON.stringify({
            model_id: "language",
            dataset_id: "multimodal_v1",
            parameters: {
              epochs: 10,
              batch_size: 32,
              learning_rate: 0.001
            },
            strategy: "standard_training"
          }, null, 2);
          await this.testApiEndpoint();
          break;
        default:
          notify.info(`Example ${exampleId} is not implemented yet.`);
      }
    },
    
    // Copy code to clipboard
    async copyCodeToClipboard(exampleId) {
      try {
        // Get the code from the example
        let codeToCopy = '';
        
        switch(exampleId) {
          case 'health-check':
            codeToCopy = `curl -X GET http://localhost:8000/health`;
            break;
          case 'chat-session':
            codeToCopy = `curl -X POST http://localhost:8000/api/chat \\
  -H "Content-Type: application/json" \\
  -d '{
    "message": "Hello, how does the Self Soul system work?",
    "session_id": "user_123_session",
    "model_id": "language"
  }'`;
            break;
          case 'training-start':
            codeToCopy = `curl -X POST http://localhost:8000/api/training/start \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_id": "language",
    "dataset_id": "multimodal_v1",
    "parameters": {
      "epochs": 10,
      "batch_size": 32,
      "learning_rate": 0.001
    },
    "strategy": "standard_training"
  }'`;
            break;
          default:
            codeToCopy = 'Example code not found';
        }
        
        await navigator.clipboard.writeText(codeToCopy);
        notify.success('Code copied to clipboard!');
      } catch (error) {
        console.error('Failed to copy code:', error);
        notify.error('Failed to copy code to clipboard');
      }
    },
    
    // Format JSON for display
    formatJson(json) {
      if (!json) return '';
      return JSON.stringify(json, null, 2);
    }
  }
}
</script>

<style scoped>
/* Color Variables */
:root {
  --bg-primary: #ffffff;
  --bg-secondary: #f8f8f8;
  --bg-tertiary: #f0f0f0;
  --text-primary: #333333;
  --text-secondary: #666666;
  --text-tertiary: #999999;
  --border-color: #e0e0e0;
  --border-dark: #cccccc;
  --accent-color: #2c2c2c;
}

/* Container Styles */
.help-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  min-height: 100vh;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  line-height: 1.6;
}

/* Search Container Styles */
.search-container {
  margin-bottom: 2rem;
  position: sticky;
  top: 1rem;
  z-index: 10;
  background-color: var(--bg-primary);
  padding: 1rem 0;
}

/* Section Controls */
.section-controls {
  margin-bottom: 1rem;
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
}

.control-btn {
  padding: 0.5rem 1rem;
  background-color: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.2s ease;
}

.control-btn:hover {
  background-color: var(--border-color);
  color: var(--text-primary);
}

.search-input {
  width: 100%;
  padding: 0.75rem 1rem;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  font-size: 1rem;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  transition: all 0.2s ease;
}

.search-input:focus {
  outline: none;
  border-color: var(--border-dark);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.search-results {
  margin-top: 1rem;
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 1rem;
}

.search-results h3 {
  margin-top: 0;
  margin-bottom: 1rem;
  font-size: 1.1rem;
  color: var(--text-primary);
}

.search-results ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.search-results li {
  margin-bottom: 0.5rem;
}

.search-results li:last-child {
  margin-bottom: 0;
}

.search-results a {
  color: var(--accent-color);
  text-decoration: none;
  padding: 0.5rem;
  display: block;
  border-radius: 4px;
  transition: background-color 0.2s ease;
}

.search-results a:hover {
  background-color: var(--bg-tertiary);
}

.no-results {
  margin-top: 1rem;
  padding: 1rem;
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  color: var(--text-secondary);
}

/* Help Layout Styles */
.help-layout {
  display: flex;
  gap: 2rem;
  align-items: flex-start;
}

/* Sidebar Navigation Styles */
.help-sidebar {
  width: 250px;
  flex-shrink: 0;
  position: sticky;
  top: 6rem;
  height: calc(100vh - 6rem);
  overflow-y: auto;
  padding-right: 1rem;
}

.help-nav {
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 1rem;
}

.help-nav h3 {
  margin-top: 0;
  margin-bottom: 1rem;
  font-size: 1.1rem;
  color: var(--text-primary);
}

.help-nav ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.help-nav li {
  margin-bottom: 0.5rem;
}

.help-nav li:last-child {
  margin-bottom: 0;
}

.help-nav a {
  color: var(--text-secondary);
  text-decoration: none;
  padding: 0.5rem;
  display: block;
  border-radius: 4px;
  transition: all 0.2s ease;
  font-size: 0.95rem;
}

.help-nav a:hover {
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
}

/* Main Content Styles */
.help-main {
  flex: 1;
}

/* Header Styles */
.help-header {
  text-align: center;
  margin-bottom: 3rem;
  padding-bottom: 1.5rem;
  border-bottom: 2px solid var(--border-color);
}

.help-header h1 {
  font-size: 2.5rem;
  font-weight: 300;
  margin-bottom: 0.5rem;
  color: var(--accent-color);
}

.help-header p {
  font-size: 1.1rem;
  color: var(--text-secondary);
  margin: 0;
}

/* Main Content Styles */
.help-main {
  margin-bottom: 3rem;
}

.help-section {
  margin-bottom: 2.5rem;
  padding: 2rem;
  background-color: var(--bg-secondary);
  border-radius: 8px;
  border: 1px solid var(--border-color);
}

.help-section h2 {
  font-size: 1.8rem;
  font-weight: 400;
  margin-bottom: 1.5rem;
  color: var(--accent-color);
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 0.5rem;
}

.help-section p {
  color: var(--text-secondary);
  margin-bottom: 1rem;
}

/* Feature List Styles */
.feature-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin-top: 1.5rem;
}

.feature-item {
  padding: 1.5rem;
  background-color: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  transition: all 0.2s ease;
}

.feature-item:hover {
  border-color: var(--border-dark);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.feature-item h3 {
  font-size: 1.2rem;
  font-weight: 500;
  margin-bottom: 0.5rem;
  color: var(--accent-color);
}

.feature-item p {
  margin: 0;
  color: var(--text-secondary);
  font-size: 0.95rem;
}

/* Step List Styles */
.step-list {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.step {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
}

.step-number {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background-color: var(--accent-color);
  color: var(--bg-primary);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  flex-shrink: 0;
}

.step-content h3 {
  font-size: 1.2rem;
  font-weight: 500;
  margin-bottom: 0.5rem;
  color: var(--text-primary);
}

.step-content p {
  margin: 0;
  color: var(--text-secondary);
}

/* Model Grid Styles */
.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-top: 1rem;
}

/* Section Header and Content Styles */
.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  padding: 0.5rem 0;
}

.section-header:hover {
  background-color: rgba(0, 0, 0, 0.02);
}

.section-header h2 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 400;
  color: var(--accent-color);
  border-bottom: none;
  padding-bottom: 0;
}

.toggle-icon {
  font-size: 1.2rem;
  font-weight: bold;
  color: var(--text-tertiary);
  transition: transform 0.2s ease;
  width: 20px;
  text-align: center;
}

.section-content {
  padding-top: 1rem;
  padding-bottom: 0.5rem;
}

.model-card {
  padding: 1.5rem;
  background-color: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  transition: all 0.2s ease;
}

.model-card:hover {
  border-color: var(--border-dark);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.model-card h3 {
  font-size: 1.1rem;
  font-weight: 500;
  margin-bottom: 0.5rem;
  color: var(--accent-color);
}

.model-card p {
  margin: 0;
  color: var(--text-secondary);
  font-size: 0.95rem;
}

/* Training Info Styles */
.training-info h3 {
  font-size: 1.2rem;
  font-weight: 500;
  margin-bottom: 1rem;
  color: var(--text-primary);
}

.training-info ul {
  list-style: none;
  padding: 0;
}

.training-info li {
  padding: 0.5rem 0;
  color: var(--text-secondary);
  border-bottom: 1px solid var(--border-color);
}

.training-info li:last-child {
  border-bottom: none;
}

.training-info strong {
  color: var(--text-primary);
}

/* FAQ Styles */
.faq-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.faq-item {
  padding: 1.5rem;
  background-color: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
}

.faq-item h3 {
  font-size: 1.1rem;
  font-weight: 500;
  margin-bottom: 0.5rem;
  color: var(--accent-color);
}

.faq-item p {
  margin: 0;
  color: var(--text-secondary);
}

/* Footer Styles */
.help-footer {
  text-align: center;
  padding: 2rem 0;
  margin-top: 3rem;
  border-top: 1px solid var(--border-color);
  color: var(--text-tertiary);
  font-size: 0.9rem;
}

.help-footer p {
  margin: 0;
}

/* Responsive Design */
@media (max-width: 768px) {
  .help-container {
    padding: 1rem;
  }
  
  .help-header h1 {
    font-size: 2rem;
  }
  
  .help-section {
    padding: 1.5rem;
  }
  
  .model-grid {
    grid-template-columns: 1fr;
  }
  
  .step {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .step-number {
    margin-bottom: 0.5rem;
  }
  
  .ports-table {
    font-size: 0.9rem;
  }
  
  .ports-table th,
  .ports-table td {
    padding: 0.5rem;
  }
}

/* Ports Table Styles */
.ports-table {
  margin-top: 1rem;
  overflow-x: auto;
}

.ports-table table {
  width: 100%;
  border-collapse: collapse;
}

.ports-table th,
.ports-table td {
  padding: 0.75rem;
  text-align: left;
  border-bottom: 1px solid var(--border-color);
}

.ports-table th {
  background-color: var(--bg-tertiary);
  font-weight: 500;
  color: var(--accent-color);
}

.ports-table tr:hover {
  background-color: var(--bg-secondary);
}

/* Newbie-specific styles */
.newbie-guide {
  background-color: #f8f9fa;
  border-left: 4px solid #4a90e2;
  padding: 1.5rem;
  margin-bottom: 2rem;
  border-radius: 6px;
}

.newbie-guide h3 {
  color: #2c3e50;
  margin-top: 0;
}

.newbie-guide .tip {
  background-color: #fffde7;
  border-left: 3px solid #ffd600;
  padding: 0.75rem 1rem;
  margin: 0.5rem 0;
  border-radius: 4px;
  font-size: 0.95rem;
}

.quick-tasks {
  background-color: #e8f5e9;
  border: 1px solid #c8e6c9;
  border-radius: 8px;
  padding: 1.5rem;
  margin-top: 1.5rem;
}

.quick-tasks h4 {
  color: #2e7d32;
  margin-top: 0;
}

.quick-tasks ul {
  list-style: none;
  padding-left: 0;
}

.quick-tasks li {
  padding: 0.5rem 0;
  border-bottom: 1px solid #c8e6c9;
}

.quick-tasks li:last-child {
  border-bottom: none;
}

.simple-explanation {
  background-color: #f3e5f5;
  border: 1px solid #e1bee7;
  border-radius: 8px;
  padding: 1.5rem;
  margin: 1.5rem 0;
}

.simple-explanation h3 {
  color: #7b1fa2;
  margin-top: 0;
}

.model-simple-table {
  margin: 1.5rem 0;
  overflow-x: auto;
}

.model-simple-table table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.95rem;
}

.model-simple-table th {
  background-color: var(--bg-tertiary);
  color: var(--accent-color);
  font-weight: 500;
  padding: 0.75rem;
  text-align: left;
}

.model-simple-table td {
  padding: 0.75rem;
  border-bottom: 1px solid var(--border-color);
  vertical-align: top;
}

.model-simple-table tr:hover {
  background-color: var(--bg-secondary);
}

.beginner-tip {
  background-color: #e3f2fd;
  border-left: 4px solid #2196f3;
  padding: 1rem;
  margin: 1rem 0;
  border-radius: 4px;
}

.beginner-tip h4 {
  color: #1976d2;
  margin-top: 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.beginner-tip h4::before {
  content: "💡";
}

.first-time-box {
  background-color: #fff3e0;
  border: 2px solid #ff9800;
  border-radius: 8px;
  padding: 1.5rem;
  margin: 1.5rem 0;
}

.first-time-box h3 {
  color: #ef6c00;
  margin-top: 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.first-time-box h3::before {
  content: "🎯";
}

.code-block {
  background-color: #f5f5f5;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 1rem;
  overflow-x: auto;
  font-family: 'Courier New', monospace;
  font-size: 0.9rem;
  line-height: 1.4;
}

/* Model Detailed Documentation Styles */
.model-detailed-doc {
  margin: 2rem 0;
  padding: 2rem;
  background-color: #f9f9f9;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
}

.model-detailed-doc h3 {
  color: #2c3e50;
  border-bottom: 2px solid #4a90e2;
  padding-bottom: 0.5rem;
  margin-bottom: 1.5rem;
}

.model-technical-specs table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
}

.model-technical-specs th {
  background-color: #e8f4fd;
  padding: 0.75rem;
  text-align: left;
  border: 1px solid #d0e7ff;
  font-weight: 600;
  color: #2c3e50;
  width: 30%;
}

.model-technical-specs td {
  padding: 0.75rem;
  border: 1px solid #e0e0e0;
  background-color: white;
}

.model-architecture {
  margin: 1.5rem 0;
  padding: 1.5rem;
  background-color: white;
  border-left: 4px solid #3498db;
  border-radius: 4px;
}

.model-architecture h4 {
  color: #2980b9;
  margin-top: 0;
}

.model-capabilities ul,
.model-integration ul,
.model-training ol {
  margin: 1rem 0;
  padding-left: 1.5rem;
}

.model-capabilities li,
.model-integration li {
  margin-bottom: 0.5rem;
  padding-left: 0.5rem;
}

.model-usage {
  margin: 1.5rem 0;
}

.usage-example {
  margin: 1rem 0;
  padding: 1rem;
  background-color: #f5f5f5;
  border-left: 3px solid #9b59b6;
  border-radius: 4px;
}

.usage-example h5 {
  color: #8e44ad;
  margin-top: 0;
}

.model-performance table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
}

.model-performance th {
  background-color: #e8f6ef;
  padding: 0.75rem;
  text-align: left;
  border: 1px solid #d0f0e0;
  font-weight: 600;
  color: #27ae60;
}

.model-performance td {
  padding: 0.75rem;
  border: 1px solid #e0e0e0;
  background-color: white;
}

.model-performance td:nth-child(2) {
  font-weight: bold;
  color: #27ae60;
  text-align: center;
  width: 15%;
}

.other-models-overview {
  margin: 2rem 0;
  padding: 2rem;
  background-color: #f0f8ff;
  border: 1px solid #d0e7ff;
  border-radius: 8px;
}

.model-index {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin: 1.5rem 0;
}

.index-column {
  padding: 1rem;
  background-color: white;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
}

.index-column h5 {
  color: #2c3e50;
  margin-top: 0;
  border-bottom: 1px solid #e0e0e0;
  padding-bottom: 0.5rem;
  margin-bottom: 0.75rem;
}

.index-column ul {
  list-style: none;
  padding-left: 0;
  margin: 0;
}

.index-column li {
  padding: 0.5rem 0;
  border-bottom: 1px solid #f5f5f5;
}

.index-column li:last-child {
  border-bottom: none;
}

.documentation-note {
  margin: 1.5rem 0;
  padding: 1.5rem;
  background-color: #fffde7;
  border: 1px solid #ffd600;
  border-radius: 6px;
}

.documentation-note h4 {
  color: #ff9800;
  margin-top: 0;
}

/* Responsive adjustments for new elements */
@media (max-width: 768px) {
  .newbie-guide,
  .quick-tasks,
  .simple-explanation,
  .first-time-box {
    padding: 1rem;
  }
  
  .model-simple-table {
    font-size: 0.85rem;
  }
  
  .model-simple-table th,
  .model-simple-table td {
    padding: 0.5rem;
  }
  
  .model-detailed-doc {
    padding: 1rem;
  }
  
  .model-index {
    grid-template-columns: 1fr;
  }
  
  .model-technical-specs th,
  .model-technical-specs td,
  .model-performance th,
  .model-performance td {
    padding: 0.5rem;
    font-size: 0.9rem;
  }
  
  .api-doc-guide .features-grid,
  .api-doc-guide .category-list {
    grid-template-columns: 1fr;
  }
  
  .api-doc-guide .api-categories .category-list {
    grid-template-columns: 1fr;
  }
}

/* API Documentation Guide Styles */
.api-doc-guide {
  margin: 2rem 0;
}

.api-doc-guide h3 {
  color: #2c3e50;
  border-bottom: 2px solid #3498db;
  padding-bottom: 0.5rem;
  margin-bottom: 1rem;
}

.api-doc-guide h4 {
  color: #2980b9;
  margin-top: 1.5rem;
  margin-bottom: 1rem;
}

.api-doc-guide h5 {
  color: #2c3e50;
  margin-top: 1rem;
  margin-bottom: 0.5rem;
}

.swagger-features .features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin: 1.5rem 0;
}

.swagger-features .feature-card {
  padding: 1.5rem;
  background-color: white;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  border-left: 4px solid #3498db;
  transition: all 0.2s ease;
}

.swagger-features .feature-card:hover {
  border-color: #3498db;
  box-shadow: 0 2px 8px rgba(52, 152, 219, 0.1);
}

.api-categories .category-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin: 1.5rem 0;
}

.api-categories .category-item {
  padding: 1.5rem;
  background-color: white;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  border-left: 4px solid #2ecc71;
}

.api-categories .category-item h4 {
  color: #27ae60;
  margin-top: 0;
  margin-bottom: 1rem;
}

.api-categories .category-item ul {
  list-style: none;
  padding-left: 0;
  margin: 0;
}

.api-categories .category-item li {
  padding: 0.5rem 0;
  border-bottom: 1px solid #f5f5f5;
  font-family: 'Courier New', monospace;
  font-size: 0.9rem;
}

.api-categories .category-item li:last-child {
  border-bottom: none;
}

.api-usage-examples .example-item {
  margin: 1.5rem 0;
  padding: 1.5rem;
  background-color: #f8f9fa;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  border-left: 4px solid #9b59b6;
}

.websocket-endpoints table {
  width: 100%;
  border-collapse: collapse;
  margin: 1.5rem 0;
  background-color: white;
}

.websocket-endpoints th {
  background-color: #34495e;
  color: white;
  padding: 0.75rem;
  text-align: left;
  font-weight: 600;
}

.websocket-endpoints td {
  padding: 0.75rem;
  border-bottom: 1px solid #e0e0e0;
  font-family: 'Courier New', monospace;
  font-size: 0.9rem;
}

.websocket-endpoints tr:hover {
  background-color: #f8f9fa;
}

.swagger-tips {
  margin: 1.5rem 0;
  padding: 1.5rem;
  background-color: #e8f4fd;
  border: 1px solid #d0e7ff;
  border-radius: 8px;
}

.swagger-tips ul {
  list-style: none;
  padding-left: 0;
  margin: 1rem 0;
}

.swagger-tips li {
  padding: 0.5rem 0;
  border-bottom: 1px solid #d0e7ff;
}

.swagger-tips li:last-child {
  border-bottom: none;
}

.api-security {
  margin: 1.5rem 0;
  padding: 1.5rem;
  background-color: #fff8e1;
  border: 1px solid #ffe082;
  border-radius: 8px;
}

.api-best-practices {
  margin: 1.5rem 0;
  padding: 1.5rem;
  background-color: #e8f5e9;
  border: 1px solid #c8e6c9;
  border-radius: 8px;
}

.api-best-practices ol {
  margin: 1rem 0;
  padding-left: 1.5rem;
}

.api-best-practices li {
  padding: 0.5rem 0;
  margin-bottom: 0.5rem;
}

.api-troubleshooting {
  margin: 1.5rem 0;
  padding: 1.5rem;
  background-color: #ffebee;
  border: 1px solid #ffcdd2;
  border-radius: 8px;
}

.troubleshooting-tips .issue {
  margin: 1rem 0;
  padding: 1rem;
  background-color: white;
  border: 1px solid #ffcdd2;
  border-radius: 6px;
}

.troubleshooting-tips .issue h5 {
  color: #e53935;
  margin-top: 0;
}

.api-support {
  margin: 1.5rem 0;
  padding: 1.5rem;
  background-color: #f3e5f5;
  border: 1px solid #e1bee7;
  border-radius: 8px;
}

.api-support ul {
  list-style: none;
  padding-left: 0;
  margin: 1rem 0;
}

.api-support li {
  padding: 0.5rem 0;
  border-bottom: 1px solid #e1bee7;
}

.api-support li:last-child {
  border-bottom: none;
}

.api-support a {
  color: #7b1fa2;
  text-decoration: none;
  font-weight: 500;
}

.api-support a:hover {
  text-decoration: underline;
}

/* Interactive API Tester Styles */
.interactive-api-tester {
  margin-top: 2rem;
  padding: 1.5rem;
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
}

.interactive-api-tester h3 {
  margin-top: 0;
  margin-bottom: 1rem;
  font-size: 1.3rem;
  color: var(--text-primary);
}

.interactive-api-tester h4 {
  margin-top: 1.5rem;
  margin-bottom: 1rem;
  font-size: 1.1rem;
  color: var(--text-primary);
}

.api-tester-controls {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.tester-section {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.tester-section label {
  font-weight: 500;
  color: var(--text-primary);
}

.endpoint-select {
  padding: 0.75rem;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  font-size: 1rem;
  font-family: inherit;
  background-color: var(--bg-primary);
  color: var(--text-primary);
}

.endpoint-select:focus {
  outline: none;
  border-color: var(--border-dark);
  box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
}

.request-body-textarea {
  padding: 0.75rem;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  font-size: 1rem;
  font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  resize: vertical;
}

.request-body-textarea:focus {
  outline: none;
  border-color: var(--border-dark);
  box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
}

.tester-actions {
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
}

.test-api-btn {
  padding: 0.75rem 1.5rem;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  transition: background-color 0.2s ease;
}

.test-api-btn:hover:not(:disabled) {
  background-color: #0056b3;
}

.test-api-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.clear-btn {
  padding: 0.75rem 1.5rem;
  background-color: var(--bg-tertiary);
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  transition: all 0.2s ease;
}

.clear-btn:hover {
  background-color: var(--border-color);
  color: var(--text-primary);
}

.api-results {
  margin-top: 1.5rem;
  padding: 1.5rem;
  background-color: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 4px;
}

.response-time {
  margin-bottom: 1rem;
  padding: 0.5rem;
  background-color: var(--bg-tertiary);
  border-radius: 4px;
  font-size: 0.9rem;
  color: var(--text-secondary);
}

.error-result {
  padding: 1rem;
  background-color: #f8d7da;
  border: 1px solid #f5c6cb;
  border-radius: 4px;
  color: #721c24;
}

.error-result h5 {
  margin-top: 0;
  margin-bottom: 0.5rem;
  color: #721c24;
}

.success-result {
  padding: 1rem;
  background-color: #d4edda;
  border: 1px solid #c3e6cb;
  border-radius: 4px;
  color: #155724;
}

.response-status, .response-timestamp {
  margin-bottom: 0.75rem;
  font-size: 0.9rem;
}

.response-data h5 {
  margin-top: 0;
  margin-bottom: 0.5rem;
  color: #155724;
}

.json-output {
  margin: 0;
  padding: 1rem;
  background-color: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
  font-size: 0.85rem;
  line-height: 1.4;
  overflow-x: auto;
  color: var(--text-primary);
}

.error-output {
  margin: 0;
  padding: 1rem;
  background-color: #f8d7da;
  border: 1px solid #f5c6cb;
  border-radius: 4px;
  font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
  font-size: 0.85rem;
  line-height: 1.4;
  overflow-x: auto;
  color: #721c24;
}

/* Real-time System Status Styles */
.real-time-status {
  margin-top: 2rem;
  padding: 1.5rem;
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
}

.real-time-status h3 {
  margin-top: 0;
  margin-bottom: 1rem;
  font-size: 1.3rem;
  color: var(--text-primary);
}

.status-loading, .status-error, .status-empty, .status-success {
  padding: 1rem;
  border-radius: 4px;
}

.status-loading {
  display: flex;
  align-items: center;
  gap: 1rem;
  background-color: var(--bg-tertiary);
}

.loading-spinner {
  width: 24px;
  height: 24px;
  border: 3px solid var(--border-color);
  border-top-color: var(--accent-color);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.status-error {
  background-color: #f8d7da;
  border: 1px solid #f5c6cb;
  color: #721c24;
}

.status-empty {
  background-color: var(--bg-tertiary);
  color: var(--text-secondary);
  text-align: center;
}

.status-success {
  background-color: #d4edda;
  border: 1px solid #c3e6cb;
  color: #155724;
}

.status-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.status-header h4 {
  margin: 0;
  font-size: 1.1rem;
}

.status-online {
  color: #28a745;
  font-weight: 600;
}

.status-offline {
  color: #dc3545;
  font-weight: 600;
}

.refresh-btn, .retry-btn, .check-btn {
  padding: 0.5rem 1rem;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9rem;
  transition: background-color 0.2s ease;
}

.refresh-btn:hover, .retry-btn:hover, .check-btn:hover {
  background-color: #0056b3;
}

.retry-btn {
  margin-top: 0.5rem;
  background-color: #dc3545;
}

.retry-btn:hover {
  background-color: #c82333;
}

.status-details {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}

.status-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
}

.status-label {
  font-weight: 500;
  color: #155724;
}

.status-value {
  font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
  font-size: 0.9rem;
}

.status-extra {
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
}

.status-extra h5 {
  margin-top: 0;
  margin-bottom: 0.5rem;
  font-size: 1rem;
  color: #155724;
}

.status-json {
  margin: 0;
  padding: 1rem;
  background-color: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
  font-size: 0.85rem;
  line-height: 1.4;
  overflow-x: auto;
  color: var(--text-primary);
}

/* Example Actions Styles */
.example-actions {
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border-color);
}

.run-example-btn {
  padding: 0.5rem 1rem;
  background-color: #28a745;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: background-color 0.2s ease;
}

.run-example-btn:hover {
  background-color: #218838;
}

.copy-code-btn {
  padding: 0.5rem 1rem;
  background-color: var(--bg-tertiary);
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all 0.2s ease;
}

.copy-code-btn:hover {
  background-color: var(--border-color);
  color: var(--text-primary);
}
</style>
