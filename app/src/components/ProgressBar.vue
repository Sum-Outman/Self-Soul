<template>
  <div class="progress-bar">
    <div class="progress-bar-track">
      <div 
        class="progress-bar-fill" 
        :style="{ width: computedValue + '%' }"
        :class="computedClass"
      ></div>
    </div>
    <span class="progress-bar-label" v-if="showLabel">
      {{ computedValue }}%
    </span>
  </div>
</template>

<script>
export default {
  name: 'ProgressBar',
  props: {
    value: {
      type: Number,
      default: 0,
      validator: (value) => value >= 0 && value <= 100
    },
    max: {
      type: Number,
      default: 100
    },
    showLabel: {
      type: Boolean,
      default: true
    },
    type: {
      type: String,
      default: 'default',
      validator: (value) => ['default', 'success', 'warning', 'error', 'info'].includes(value)
    }
  },
  computed: {
    computedValue() {
      return Math.min(100, (this.value / this.max) * 100);
    },
    computedClass() {
      return `progress-${this.type}`;
    }
  }
};
</script>

<style scoped>
.progress-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
}

.progress-bar-track {
  flex-grow: 1;
  height: 8px;
  background-color: #e0e0e0;
  border-radius: 4px;
  overflow: hidden;
}

.progress-bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s ease;
}

.progress-default {
  background: linear-gradient(90deg, #007bff, #0056b3);
}

.progress-success {
  background: var(--success-color);
}

.progress-warning {
  background: var(--warning-color);
}

.progress-error {
  background: var(--error-color);
}

.progress-info {
  background: var(--primary-color);
}

.progress-bar-label {
  font-size: 12px;
  color: #666;
  min-width: 40px;
  text-align: right;
}
</style>