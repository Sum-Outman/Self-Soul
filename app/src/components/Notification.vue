<template>
  <div 
    class="notification"
    :class="[typeClass, positionClass]"
    @click="handleClick"
    v-show="visible"
  >
    <div class="notification-icon">
      <i :class="iconClass"></i>
    </div>
    <div class="notification-content">
      <div class="notification-title">{{ title }}</div>
      <div class="notification-message">{{ message }}</div>
    </div>
    <div class="notification-actions">
      <button class="notification-close" @click.stop="close">
        <i class="fas fa-times"></i>
      </button>
    </div>
  </div>
</template>

<script>
import { 
  NOTIFICATION_TYPES, 
  NOTIFICATION_POSITIONS, 
  NOTIFICATION_STATUS 
} from '@/utils/notification/notificationTypes';

export default {
  name: 'Notification',
  props: {
    id: {
      type: String,
      required: true
    },
    title: {
      type: String,
      default: ''
    },
    message: {
      type: String,
      required: true
    },
    type: {
      type: String,
      default: NOTIFICATION_TYPES.INFO,
      validator: (value) => Object.values(NOTIFICATION_TYPES).includes(value)
    },
    position: {
      type: String,
      default: NOTIFICATION_POSITIONS.TOP_RIGHT,
      validator: (value) => Object.values(NOTIFICATION_POSITIONS).includes(value)
    },
    duration: {
      type: Number,
      default: null // null means persistent
    },
    autoClose: {
      type: Boolean,
      default: true
    },
    closeOnClick: {
      type: Boolean,
      default: true
    }
  },
  data() {
    return {
      visible: true,
      timeoutId: null
    };
  },
  computed: {
    typeClass() {
      return `notification-${this.type}`;
    },
    positionClass() {
      return `notification-${this.position.replace('-', '_')}`;
    },
    iconClass() {
      const icons = {
        [NOTIFICATION_TYPES.SUCCESS]: 'fas fa-check-circle',
        [NOTIFICATION_TYPES.ERROR]: 'fas fa-exclamation-circle',
        [NOTIFICATION_TYPES.WARNING]: 'fas fa-exclamation-triangle',
        [NOTIFICATION_TYPES.INFO]: 'fas fa-info-circle'
      };
      return icons[this.type] || icons[NOTIFICATION_TYPES.INFO];
    }
  },
  mounted() {
    if (this.autoClose && this.duration !== null && this.duration > 0) {
      this.timeoutId = setTimeout(() => {
        this.close();
      }, this.duration);
    }
  },
  beforeUnmount() {
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
    }
  },
  methods: {
    handleClick() {
      if (this.closeOnClick) {
        this.close();
      }
    },
    close() {
      this.visible = false;
      this.$emit('close', this.id);
    }
  }
};
</script>

<style scoped>
.notification {
  position: fixed;
  z-index: 9999;
  min-width: 300px;
  max-width: 400px;
  padding: 15px;
  margin: 10px;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  display: flex;
  align-items: flex-start;
  transition: all 0.3s ease;
  cursor: pointer;
}

.notification-icon {
  margin-right: 12px;
  font-size: 20px;
  flex-shrink: 0;
}

.notification-content {
  flex-grow: 1;
  min-width: 0;
}

.notification-title {
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 4px;
}

.notification-message {
  font-size: 13px;
  line-height: 1.4;
}

.notification-actions {
  margin-left: 12px;
  flex-shrink: 0;
}

.notification-close {
  background: none;
  border: none;
  color: inherit;
  cursor: pointer;
  opacity: 0.6;
  padding: 2px;
  border-radius: 4px;
  transition: opacity 0.2s;
}

.notification-close:hover {
  opacity: 1;
}

/* Type styles */
.notification-success {
  background-color: #d4edda;
  color: #155724;
  border-left: 4px solid #28a745;
}

.notification-error {
  background-color: #f8d7da;
  color: #721c24;
  border-left: 4px solid #dc3545;
}

.notification-warning {
  background-color: #fff3cd;
  color: #856404;
  border-left: 4px solid #ffc107;
}

.notification-info {
  background-color: #d1ecf1;
  color: #0c5460;
  border-left: 4px solid #17a2b8;
}

/* Position styles */
.notification-top_left {
  top: 20px;
  left: 20px;
}

.notification-top_center {
  top: 20px;
  left: 50%;
  transform: translateX(-50%);
}

.notification-top_right {
  top: 20px;
  right: 20px;
}

.notification-bottom_left {
  bottom: 20px;
  left: 20px;
}

.notification-bottom_center {
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
}

.notification-bottom_right {
  bottom: 20px;
  right: 20px;
}
</style>