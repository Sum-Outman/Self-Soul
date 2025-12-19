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
          <div v-if="showSearchResults && searchResults.length > 0" class="search-results">
            <h3>Search Results ({{ searchResults.length }})</h3>
            <ul>
              <li v-for="result in searchResults" :key="result.id">
                <a :href="'#' + result.id" @click.prevent="scrollToSection(result.id)">{{ result.title }}</a>
              </li>
            </ul>
          </div>
          
          <!-- No Results Message -->
          <div v-else-if="showSearchResults" class="no-results">
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
            <li><a href="#ports-config" @click.prevent="scrollToSection('ports-config')">Service Ports Configuration</a></li>
            <li><a href="#getting-started" @click.prevent="scrollToSection('getting-started')">Getting Started</a></li>
            <li><a href="#core-models" @click.prevent="scrollToSection('core-models')">Core Cognitive Models</a></li>
            <li><a href="#training-methodology" @click.prevent="scrollToSection('training-methodology')">Training Methodology</a></li>
            <li><a href="#advanced-capabilities" @click.prevent="scrollToSection('advanced-capabilities')">Advanced Capabilities</a></li>
            <li><a href="#page-features" @click.prevent="scrollToSection('page-features')">Page Features Documentation</a></li>
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
        <div v-if="sectionExpanded.systemOverview" class="section-content">
        <p>Self Soul is a revolutionary human-like AGI system designed for autonomous learning, self-optimization, and multimodal intelligence. The system features a sophisticated architecture that integrates multiple cognitive capabilities including language processing, visual recognition, audio analysis, sensor data interpretation, and autonomous decision-making.</p>
        
        <div class="architecture-overview">
          <h3>Architecture Highlights</h3>
          <p>The system is built on a <strong>Unified Cognitive Architecture</strong> that integrates 19 specialized models working in concert to provide comprehensive AGI capabilities. Each model is assigned a dedicated port within the range 8001-8019, enabling distributed parallel processing while maintaining coordinated operation through the Manager Model (Port 8001).</p>
          <p>Key components include the <strong>Adaptive Learning Engine</strong> that optimizes training parameters based on model performance and data characteristics, and a <strong>Dual ID System</strong> that ensures seamless communication between frontend (letter IDs A-X) and backend (string IDs like manager, language).</p>
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
              <p>FastAPI providing RESTful API endpoints for model management, training, monitoring, and dataset operations, with interactive documentation at http://localhost:8000/docs</p>
            </div>
            <div class="tech-item">
              <h4>Communication</h4>
              <p>WebSocket for real-time updates, enabling live training progress, system monitoring, and multimodal stream processing</p>
            </div>
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
        <div v-if="sectionExpanded.portsConfig" class="section-content">
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
                <td>8765</td>
                <td>Manages real-time data streams and inter-model communication</td>
              </tr>
              <tr>
                <td>Performance Monitoring</td>
                <td>8081</td>
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
        <div v-if="sectionExpanded.gettingStarted" class="section-content">
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
      </section>

      <!-- Core Models -->
      <section id="core-models" class="help-section">
        <div class="section-header" @click="toggleSection('coreModels')">
          <h2>Core Cognitive Models</h2>
          <span class="toggle-icon">{{ sectionExpanded.coreModels ? '−' : '+' }}</span>
        </div>
        <div v-if="sectionExpanded.coreModels" class="section-content">
        <p>The system comprises 19 specialized models that work in concert to provide comprehensive AGI capabilities. Each model is assigned a dedicated port within the range 8001-8019:</p>
        <div class="model-grid">
          <div class="model-card">
            <h3>Manager Model (Port 8001)</h3>
            <p>System manager model for coordination</p>
          </div>
            <div class="model-card">
            <h3>Language Model (Port 8002)</h3>
            <p>Natural language processing model</p>
          </div>
          <div class="model-card">
            <h3>Knowledge Model (Port 8003)</h3>
            <p>Knowledge base and retrieval model</p>
          </div>
          <div class="model-card">
            <h3>Vision Model (Port 8004)</h3>
            <p>Computer vision and image processing model</p>
          </div>
          <div class="model-card">
            <h3>Audio Model (Port 8005)</h3>
            <p>Audio processing and speech recognition model</p>
          </div>
          <div class="model-card">
            <h3>Autonomous Model (Port 8006)</h3>
            <p>Self-governing and decision-making model</p>
          </div>
          <div class="model-card">
            <h3>Programming Model (Port 8007)</h3>
            <p>Code generation and software development model</p>
          </div>
          <div class="model-card">
            <h3>Planning Model (Port 8008)</h3>
            <p>Strategic planning and execution model</p>
          </div>
          <div class="model-card">
            <h3>Emotion Model (Port 8009)</h3>
            <p>Emotional analysis and response model</p>
          </div>
          <div class="model-card">
            <h3>Spatial Model (Port 8010)</h3>
            <p>Spatial reasoning and navigation model</p>
          </div>
          <div class="model-card">
            <h3>Computer Vision Model (Port 8011)</h3>
            <p>Advanced computer vision capabilities</p>
          </div>
          <div class="model-card">
            <h3>Sensor Model (Port 8012)</h3>
            <p>Sensor data processing and integration</p>
          </div>
          <div class="model-card">
            <h3>Motion Model (Port 8013)</h3>
            <p>Motion planning and control model</p>
          </div>
          <div class="model-card">
            <h3>Prediction Model (Port 8014)</h3>
            <p>Predictive analytics and forecasting model</p>
          </div>
          <div class="model-card">
            <h3>Advanced Reasoning Model (Port 8015)</h3>
            <p>Complex logical reasoning capabilities</p>
          </div>
          <div class="model-card">
            <h3>Data Fusion Model (Port 8016)</h3>
            <p>Multi-source data integration and fusion</p>
          </div>
          <div class="model-card">
            <h3>Creative Problem Solving Model (Port 8017)</h3>
            <p>Innovative problem-solving approaches</p>
          </div>
          <div class="model-card">
            <h3>Meta Cognition Model (Port 8018)</h3>
            <p>Self-awareness and cognitive monitoring</p>
          </div>
          <div class="model-card">
            <h3>Value Alignment Model (Port 8019)</h3>
            <p>Ethical decision making and value alignment</p>
          </div>
        </div>
        
        <div class="training-info">
          <h3>External API Integration</h3>
          <p>All core cognitive models can be enhanced or replaced with external API providers for extended capabilities. The system supports seamless integration with multiple AI service providers:</p>
          
          <div class="info-box">
            <h4>Supported API Providers</h4>
            <ul>
              <li><strong>International Providers:</strong> OpenAI, Anthropic, Google AI, HuggingFace, Cohere, Mistral</li>
              <li><strong>Domestic Chinese Providers:</strong> DeepSeek, SiliconFlow (硅基流动), Zhipu AI (智谱AI), Baidu ERNIE (百度文心一言), Alibaba Qwen (阿里通义千问), Moonshot (月之暗面), Yi (零一万物), Tencent Hunyuan (腾讯混元)</li>
              <li><strong>Local Model Support:</strong> Ollama integration for running local LLMs with automatic configuration</li>
            </ul>
            
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

      <!-- Training Guide -->
      <section id="training-methodology" class="help-section">
        <div class="section-header" @click="toggleSection('trainingMethodology')">
          <h2>Training Methodology</h2>
          <span class="toggle-icon">{{ sectionExpanded.trainingMethodology ? '−' : '+' }}</span>
        </div>
        <div v-if="sectionExpanded.trainingMethodology" class="section-content">
        <p>The Self Soul system employs advanced training techniques to continuously improve its capabilities and adapt to new scenarios.</p>
        
        <div class="training-info">
          <h3>Training Interface Usage Steps</h3>
          <div class="step-list">
            <div class="step">
              <span class="step-number">1</span>
              <div class="step-content">
                <h4>Access the Training Interface</h4>
                <p>Navigate to <code>http://localhost:5175/#/training</code> in your web browser to access the training dashboard.</p>
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
              <li>Navigate to the training interface at <code>http://localhost:5175/#/training</code></li>
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
        <div v-if="sectionExpanded.advancedCapabilities" class="section-content">
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
        </div>
        </div>
      </section>

      <!-- Page Features Documentation -->
      <section id="page-features" class="help-section">
        <div class="section-header" @click="toggleSection('pageFeatures')">
          <h2>Page Features Documentation</h2>
          <span class="toggle-icon">{{ sectionExpanded.pageFeatures ? '−' : '+' }}</span>
        </div>
        <div v-if="sectionExpanded.pageFeatures" class="section-content">
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
                  <li><strong>SiliconFlow (硅基流动):</strong> SiliconFlow platform models</li>
                  <li><strong>Zhipu AI (智谱AI):</strong> GLM models including GLM-4</li>
                  <li><strong>Baidu ERNIE (百度文心一言):</strong> ERNIE models including ERNIE 4.0</li>
                  <li><strong>Alibaba Qwen (阿里通义千问):</strong> Qwen models including Qwen2.5</li>
                  <li><strong>Moonshot (月之暗面):</strong> Moonshot AI models including Kimi</li>
                  <li><strong>Yi (零一万物):</strong> Yi models including Yi-34B</li>
                  <li><strong>Tencent Hunyuan (腾讯混元):</strong> Tencent Hunyuan models</li>
                </ul>
                
                <h6>Local Model Support:</h6>
                <ul>
                  <li><strong>Ollama:</strong> Run local LLMs with automatic configuration (default: http://localhost:11434/v1/chat/completions, model: llama2)</li>
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
              <li>Navigate to the training interface at <code>http://localhost:5175/#/training</code></li>
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
        </div>
      </section>

      <!-- Troubleshooting -->
      <section id="troubleshooting" class="help-section">
        <div class="section-header" @click="toggleSection('troubleshooting')">
          <h2>Troubleshooting & Support</h2>
          <span class="toggle-icon">{{ sectionExpanded.troubleshooting ? '−' : '+' }}</span>
        </div>
        <div v-if="sectionExpanded.troubleshooting" class="section-content">
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
        <div v-if="sectionExpanded.systemRequirements" class="section-content">
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
            Open ports: 8000-8020, 5175, 8765, 8081 for complete system functionality</p>
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
export default {
  name: 'HelpView',
  data() {
    return {
      // Feature data for the component
      features: [
        {
          id: 'unified-cognitive-architecture',
          title: 'Unified Cognitive Architecture',
          description: 'Integrated system of 19 specialized models (manager, language, vision, etc.) working synergistically via dedicated ports (8001-8019), coordinated by the Manager Model (Port 8001)'
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
          description: 'Comprehensive dataset supporting all 19 models with formats for text, images, audio, and video data, ensuring alignment between dataset and model capabilities'
        },
        {
          id: 'websocket-communication',
          title: 'WebSocket Communication',
          description: 'Enables real-time updates for training progress, system monitoring, autonomous learning status, and audio/video stream processing'
        },
        {
          id: 'distributed-model-architecture',
          title: 'Distributed Model Architecture',
          description: 'Each model runs on a dedicated port (8001-8019) for parallel processing, allowing independent optimization and coordinated operation'
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
        portsConfig: true,
        gettingStarted: true,
        coreModels: true,
        trainingMethodology: true,
        advancedCapabilities: true,
        pageFeatures: true,
        troubleshooting: true,
        systemRequirements: true
      },
      // Matched search results
      searchResults: [],
      // Show search results instead of normal content
      showSearchResults: false,
      // Debounce timer for search
      searchDebounce: null
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
        this.performSearch();
      }, 300);
    }
  },
  mounted() {
    document.title = 'Self Soul AGI System Help';
    // Add event listener for keyboard shortcuts
    document.addEventListener('keydown', this.handleKeyDown);
  },
  beforeUnmount() {
    // Remove event listener
    document.removeEventListener('keydown', this.handleKeyDown);
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
      this.showSearchResults = uniqueResults.length > 0;
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
</style>
