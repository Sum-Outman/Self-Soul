<template>
  <div class="metric-card" :class="computedClass">
    <div class="metric-header">
      <i class="fas" :class="iconClass" v-if="icon"></i>
      <h4>{{ title }}</h4>
    </div>
    <div class="metric-value">
      <span class="primary-value">{{ formattedValue }}</span>
      <span class="unit" v-if="unit">{{ unit }}</span>
    </div>
    <div class="metric-secondary" v-if="secondaryValue !== undefined">
      <span>{{ formattedSecondaryValue }}</span>
    </div>
    <div class="metric-trend" v-if="trend">
      <i class="fas" :class="trendIcon"></i>
      <span>{{ trend }}</span>
    </div>
    <div class="progress-container" v-if="type === 'progress'">
      <div class="progress-bar">
        <div 
          class="progress-fill" 
          :style="{ width: progressPercentage + '%' }"
          :class="progressClass"
        ></div>
      </div>
      <span class="progress-label" v-if="showProgressLabel">
        {{ progressPercentage }}%
      </span>
    </div>
  </div>
</template>

<script>
export default {
  name: 'MetricCard',
  props: {
    title: {
      type: String,
      required: true
    },
    value: {
      type: [Number, String],
      required: true
    },
    secondaryValue: {
      type: [Number, String],
      default: undefined
    },
    unit: {
      type: String,
      default: ''
    },
    type: {
      type: String,
      default: 'value',
      validator: (value) => ['value', 'progress', 'dual'].includes(value)
    },
    max: {
      type: Number,
      default: 100
    },
    icon: {
      type: String,
      default: ''
    },
    trend: {
      type: String,
      default: ''
    },
    criticalThreshold: {
      type: Number,
      default: 90
    },
    showProgressLabel: {
      type: Boolean,
      default: true
    }
  },
  computed: {
    formattedValue() {
      if (typeof this.value === 'number') {
        return this.value.toFixed(1);
      }
      return this.value;
    },
    formattedSecondaryValue() {
      if (this.secondaryValue === undefined) return '';
      if (typeof this.secondaryValue === 'number') {
        return this.secondaryValue.toFixed(1);
      }
      return this.secondaryValue;
    },
    progressPercentage() {
      if (this.type !== 'progress') return 0;
      return Math.min(100, (this.value / this.max) * 100);
    },
    progressClass() {
      if (this.progressPercentage > this.criticalThreshold) {
        return 'progress-critical';
      } else if (this.progressPercentage > 70) {
        return 'progress-warning';
      }
      return 'progress-normal';
    },
    iconClass() {
      const icons = {
        cpu: 'fa-microchip',
        memory: 'fa-memory',
        network: 'fa-network-wired',
        disk: 'fa-hdd',
        tasks: 'fa-tasks',
        temperature: 'fa-thermometer-half',
        humidity: 'fa-tint',
        acceleration: 'fa-dashboard',
        light: 'fa-lightbulb'
      };
      return icons[this.icon] || 'fa-chart-line';
    },
    trendIcon() {
      const trends = {
        up: 'fa-arrow-up',
        down: 'fa-arrow-down',
        stable: 'fa-minus',
        fluctuating: 'fa-random'
      };
      return trends[this.trend] || 'fa-minus';
    },
    computedClass() {
      return {
        'metric-critical': this.progressPercentage > this.criticalThreshold,
        'metric-warning': this.progressPercentage > 70 && this.progressPercentage <= this.criticalThreshold
      };
    }
  }
};
</script>

<style scoped>
.metric-card {
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
}

.metric-card:hover {
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
  transform: translateY(-2px);
}

.metric-critical {
  border-color: #dc3545;
  background-color: #fff5f5;
}

.metric-warning {
  border-color: #ffc107;
  background-color: #fffdf5;
}

.metric-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.metric-header i {
  color: #666;
  font-size: 14px;
}

.metric-header h4 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: #333;
}

.metric-value {
  display: flex;
  align-items: baseline;
  gap: 4px;
  margin-bottom: 8px;
}

.primary-value {
  font-size: 24px;
  font-weight: bold;
  color: #2c3e50;
}

.unit {
  font-size: 14px;
  color: #666;
}

.metric-secondary {
  font-size: 14px;
  color: #666;
  margin-bottom: 8px;
}

.metric-trend {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: #666;
}

.metric-trend i {
  font-size: 10px;
}

.progress-container {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
}

.progress-bar {
  flex-grow: 1;
  height: 6px;
  background-color: #e9ecef;
  border-radius: 3px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.3s ease;
}

.progress-normal {
  background: linear-gradient(90deg, #28a745, #20c997);
}

.progress-warning {
  background: linear-gradient(90deg, #ffc107, #fd7e14);
}

.progress-critical {
  background: linear-gradient(90deg, #dc3545, #c82333);
}

.progress-label {
  font-size: 12px;
  color: #666;
  min-width: 40px;
  text-align: right;
}
</style>