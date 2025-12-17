<template>
  <div class="user-guide">
    <div class="guide-overlay" v-if="showGuide" @click="closeGuide"></div>
    <div class="guide-content" v-if="showGuide">
      <div class="guide-header">
        <h2>User Guide</h2>
        <button class="close-btn" @click="closeGuide">&times;</button>
      </div>
      <div class="guide-steps">
        <div class="step" v-for="(step, index) in steps" :key="index" v-show="currentStep === index + 1">
          <div class="step-number">{{ index + 1 }}</div>
          <div class="step-content">
            <h3>{{ step.title }}</h3>
            <p>{{ step.description }}</p>
          </div>
        </div>
      </div>
      <div class="guide-navigation">
        <button @click="prevStep" :disabled="currentStep === 1">Previous</button>
        <button @click="nextStep" :disabled="currentStep === steps.length">Next</button>
        <button @click="finishGuide" v-if="currentStep === steps.length">Finish</button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue';

export default {
  name: 'UserGuide',
  props: {
    showGuide: Boolean
  },
  emits: ['close'],
  setup(props, { emit }) {
    const currentStep = ref(1);
    const steps = [
      { title: 'Welcome to AGI System', description: 'This is an artificial intelligence system with self-awareness and autonomous learning capabilities. The system includes 11 core models working together to achieve human Soul-like functionality.' },
      { title: 'Model Management', description: 'The Manager Model (A) coordinates all sub-model tasks, supporting multiple interaction methods including voice, text, and images. Model configuration and monitoring can be done through the main interface.' },
      { title: 'Training and Control', description: 'The system supports both individual model training and joint training. The training control panel allows selecting models for training and setting training parameters.' },
      { title: 'Real-time Monitoring', description: 'The monitoring dashboard provides real-time system status, model performance, and resource usage. Supports real-time data stream viewing and model restart operations.' },
      { title: 'System Features', description: 'The system provides comprehensive AI capabilities including language processing, vision recognition, audio analysis, knowledge management, and autonomous learning.' }
    ];

    const closeGuide = () => {
      emit('close');
    };

    const nextStep = () => {
      if (currentStep.value < steps.length) {
        currentStep.value++;
      }
    };

    const prevStep = () => {
      if (currentStep.value > 1) {
        currentStep.value--;
      }
    };

    const finishGuide = () => {
      localStorage.setItem('guideCompleted', 'true');
      closeGuide();
    };

    return {
      currentStep,
      steps,
      closeGuide,
      nextStep,
      prevStep,
      finishGuide
    };
  }
};
</script>

<style scoped>
.user-guide {
  position: relative;
}

.guide-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.7);
  z-index: 999;
}

.guide-content {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 80%;
  max-width: 600px;
  background-color: white;
  border-radius: 8px;
  padding: 20px;
  z-index: 1000;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.guide-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.close-btn {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: #666;
}

.guide-steps {
  margin-bottom: 20px;
}

.step {
  display: flex;
  gap: 15px;
  margin-bottom: 20px;
}

.step-number {
  width: 30px;
  height: 30px;
  background-color: #666;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
}

.step-content h3 {
  margin: 0 0 10px 0;
  color: #333;
}

.step-content p {
  margin: 0;
  color: #666;
  line-height: 1.5;
}

.guide-navigation {
  display: flex;
  justify-content: space-between;
  gap: 10px;
}

.guide-navigation button {
  padding: 8px 15px;
  background-color: #666;
  color: white;
  border: 1px solid #ddd;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.guide-navigation button:hover {
  background-color: #444;
}

.guide-navigation button:disabled {
  background-color: #ccc;
  color: #999;
  cursor: not-allowed;
}
</style>