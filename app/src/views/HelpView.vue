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
                <a href="#{{ result.id }}" @click="scrollToSection(result.id)">{{ result.title }}</a>
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
            <li><a href="#system-overview" @click="scrollToSection('system-overview')">System Overview</a></li>
            <li><a href="#ports-config" @click="scrollToSection('ports-config')">Service Ports Configuration</a></li>
            <li><a href="#getting-started" @click="scrollToSection('getting-started')">Getting Started</a></li>
            <li><a href="#core-models" @click="scrollToSection('core-models')">Core Cognitive Models</a></li>
            <li><a href="#training-methodology" @click="scrollToSection('training-methodology')">Training Methodology</a></li>
            <li><a href="#advanced-capabilities" @click="scrollToSection('advanced-capabilities')">Advanced Capabilities</a></li>
            <li><a href="#troubleshooting" @click="scrollToSection('troubleshooting')">Troubleshooting & Support</a></li>
            <li><a href="#system-requirements" @click="scrollToSection('system-requirements')">System Requirements</a></li>
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
        <p>Self Soul is a revolutionary human-like AGI system designed for autonomous learning, self-optimization, and multimodal intelligence. The system features a sophisticated architecture that integrates multiple cognitive capabilities including language processing, visual recognition, audio analysis, and sensor data interpretation.</p>
        
        <div class="feature-list">
          <div v-for="feature in features" :key="feature.id" class="feature-item">
            <h3>{{ feature.title }}</h3>
            <p>{{ feature.description }}</p>
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
            <p>Orchestrates task distribution and coordinates model interactions</p>
          </div>
          <div class="model-card">
            <h3>Language Model (Port 8002)</h3>
            <p>Processes natural language inputs and generates coherent responses</p>
          </div>
          <div class="model-card">
            <h3>Knowledge Model (Port 8003)</h3>
            <p>Manages information storage, retrieval, and knowledge synthesis</p>
          </div>
          <div class="model-card">
            <h3>Vision Model (Port 8004)</h3>
            <p>Analyzes visual content including images, videos, and spatial data</p>
          </div>
          <div class="model-card">
            <h3>Audio Model (Port 8005)</h3>
            <p>Interprets auditory inputs including speech, music, and environmental sounds</p>
          </div>
          <div class="model-card">
            <h3>Autonomous Model (Port 8006)</h3>
            <p>Enables self-directed decision making and independent action execution</p>
          </div>
          <div class="model-card">
            <h3>Programming Model (Port 8007)</h3>
            <p>Generates, analyzes, and optimizes computer code across multiple languages</p>
          </div>
          <div class="model-card">
            <h3>Planning Model (Port 8008)</h3>
            <p>Develops strategic plans and optimizes task execution sequences</p>
          </div>
          <div class="model-card">
            <h3>Emotion Model (Port 8009)</h3>
            <p>Recognizes, interprets, and responds to emotional cues and contexts</p>
          </div>
          <div class="model-card">
            <h3>Spatial Model (Port 8010)</h3>
            <p>Understands and manipulates spatial relationships and geometries</p>
          </div>
          <div class="model-card">
            <h3>Computer Vision Model (Port 8011)</h3>
            <p>Advanced analysis of visual data for object recognition and scene understanding</p>
          </div>
          <div class="model-card">
            <h3>Sensor Model (Port 8012)</h3>
            <p>Processes data from various sensors and IoT devices</p>
          </div>
          <div class="model-card">
            <h3>Motion Model (Port 8013)</h3>
            <p>Analyzes and predicts movement patterns and physical interactions</p>
          </div>
          <div class="model-card">
            <h3>Prediction Model (Port 8014)</h3>
            <p>Performs statistical analysis and forecasts future outcomes</p>
          </div>
          <div class="model-card">
            <h3>Advanced Reasoning Model (Port 8015)</h3>
            <p>Performs complex logical reasoning and problem-solving tasks</p>
          </div>
          <div class="model-card">
            <h3>Data Fusion Model (Port 8016)</h3>
            <p>Integrates information from multiple sources for comprehensive understanding</p>
          </div>
          <div class="model-card">
            <h3>Creative Problem Solving Model (Port 8017)</h3>
            <p>Develops innovative solutions to complex challenges</p>
          </div>
          <div class="model-card">
            <h3>Meta Cognition Model (Port 8018)</h3>
            <p>Monitors and optimizes the system's own cognitive processes</p>
          </div>
          <div class="model-card">
            <h3>Value Alignment Model (Port 8019)</h3>
            <p>Ensures system behaviors align with defined ethical guidelines and values</p>
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
          <h3>Training Approaches</h3>
          <ul>
            <li v-for="approach in trainingApproaches" :key="approach.id">
              <strong>{{ approach.title }}</strong> - {{ approach.description }}
            </li>
          </ul>
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
          </ul>
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
          id: 'autonomous-learning',
          title: 'Autonomous Learning',
          description: 'Continuous self-improvement through experience and data analysis'
        },
        {
          id: 'multimodal-integration',
          title: 'Multimodal Integration',
          description: 'Seamless processing of text, images, audio, and sensor inputs'
        },
        {
          id: 'real-time-adaptation',
          title: 'Real-time Adaptation',
          description: 'Dynamic adjustment to changing environments and requirements'
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
        troubleshooting: true,
        systemRequirements: true
      },
      // Matched search results
      searchResults: [],
      // Show search results instead of normal content
      showSearchResults: false
    }
  },
  computed: {
    // Filter sections based on search query
    filteredSections() {
      if (!this.searchQuery) {
        this.showSearchResults = false;
        return null;
      }
      
      const query = this.searchQuery.toLowerCase();
      const results = [];
      const sections = document.querySelectorAll('.help-section');
      
      sections.forEach((section) => {
        const sectionId = section.id;
        const sectionTitle = section.querySelector('h2').textContent.toLowerCase();
        const sectionContent = section.textContent.toLowerCase();
        
        if (sectionTitle.includes(query) || sectionContent.includes(query)) {
          results.push({
            id: sectionId,
            title: section.querySelector('h2').textContent
          });
        }
      });
      
      this.searchResults = results;
      this.showSearchResults = results.length > 0;
      return results;
    }
  },
  watch: {
    searchQuery(newQuery) {
      // Reset section expansion when search query changes
      if (!newQuery) {
        this.showSearchResults = false;
      }
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
