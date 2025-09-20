<template>
  <div class="help-container">

    <!-- Main Content -->
    <main class="help-main">
      <!-- System Introduction -->
      <section class="help-section">
        <h2>System Introduction</h2>
        <p>Self Soul is a revolutionary human-like AGI system with the ability to think independently, learn autonomously, self-optimize and self-upgrade. The system adopts a multimodal architecture, integrating language, vision, audio, sensors and other perception capabilities.</p>
      </section>

      <!-- Quick Start -->
      <section class="help-section">
        <h2>Quick Start</h2>
        <div class="step-list">
          <div class="step">
            <span class="step-number">1</span>
            <div class="step-content">
              <h3>Start the System</h3>
              <p>Run the startup script to launch the AGI system</p>
            </div>
          </div>
          <div class="step">
            <span class="step-number">2</span>
            <div class="step-content">
              <h3>Configure Models</h3>
              <p>Configure local or external API models in the settings page</p>
            </div>
          </div>
          <div class="step">
            <span class="step-number">3</span>
            <div class="step-content">
              <h3>Start Using</h3>
              <p>Conduct multimodal conversations with the system through the interface</p>
            </div>
          </div>
        </div>
      </section>

      <!-- Core Models -->
      <section class="help-section">
        <h2>Core Models</h2>
        <div class="model-grid">
          <div class="model-card">
            <h3>Manager Model</h3>
            <p>Responsible for task allocation and model coordination</p>
          </div>
          <div class="model-card">
            <h3>Language Model</h3>
            <p>Processes text interactions and conversations</p>
          </div>
          <div class="model-card">
            <h3>Vision Model</h3>
            <p>Processes image and video content</p>
          </div>
          <div class="model-card">
            <h3>Audio Model</h3>
            <p>Processes voice and audio inputs</p>
          </div>
          <div class="model-card">
            <h3>Autonomous Model</h3>
            <p>Implements system autonomous decision-making and actions</p>
          </div>
          <div class="model-card">
            <h3>Knowledge Model</h3>
            <p>Manages and processes knowledge base content</p>
          </div>
          <div class="model-card">
            <h3>Planning Model</h3>
            <p>Develops task plans and execution strategies</p>
          </div>
          <div class="model-card">
            <h3>Prediction Model</h3>
            <p>Performs data analysis and trend prediction</p>
          </div>
          <div class="model-card">
            <h3>Programming Model</h3>
            <p>Generates and optimizes code</p>
          </div>
          <div class="model-card">
            <h3>Sensor Model</h3>
            <p>Processes various sensor input data</p>
          </div>
          <div class="model-card">
            <h3>Spatial Model</h3>
            <p>Understands and processes spatial relationships</p>
          </div>
          <div class="model-card">
            <h3>Emotion Model</h3>
            <p>Recognizes and processes emotional information</p>
          </div>
        </div>
      </section>

      <!-- Training Guide -->
      <section class="help-section">
        <h2>Training Guide</h2>
        <div class="training-info">
          <h3>Training Types</h3>
          <ul>
            <li><strong>Individual Training</strong> - Specialized training for a single model</li>
            <li><strong>Joint Training</strong> - Multiple models training collaboratively</li>
            <li><strong>Continuous Training</strong> - System continuous learning and optimization</li>
          </ul>
        </div>
      </section>

      <!-- Troubleshooting -->
      <section class="help-section">
        <h2>Troubleshooting</h2>
        <div class="faq-list">
          <div class="faq-item">
            <h3>System Startup Failure</h3>
            <p>Check system logs and dependency installation</p>
          </div>
          <div class="faq-item">
            <h3>Model Unresponsive</h3>
            <p>Check model status and network connection</p>
          </div>
          <div class="faq-item">
            <h3>Training Failure</h3>
            <p>Check dataset format and model dependencies</p>
          </div>
        </div>
      </section>
    </main>

    <!-- Footer -->
    <footer class="help-footer">
      <p>Self Soul System v2.0.0 | Apache 2.0 License</p>
    </footer>
  </div>
</template>

<script>
export default {
  name: 'HelpView',
  data() {
    return {
      // Minimal data for the component
    }
  },
  mounted() {
    document.title = 'Self Soul AGI System Help'
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
  max-width: 1000px;
  margin: 0 auto;
  padding: 2rem;
  min-height: 100vh;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  line-height: 1.6;
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
}
</style>
