<template>
  <transition name="fade">
    <div v-if="visible" class="notification" :class="type" :style="positionStyle">
      <div class="notification-content">
        <div v-if="title" class="notification-title">{{ title }}</div>
        <div class="notification-message">{{ message }}</div>
      </div>
      <button v-if="closable" class="notification-close" @click="close">×</button>
    </div>
  </transition>
</template>

<script>
import { ref, onMounted, onUnmounted, computed } from 'vue'

export default {
  name: 'Notification',
  props: {
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
      default: 'info',
      validator: (value) => {
        return ['info', 'success', 'warning', 'error'].includes(value)
      }
    },
    duration: {
      type: Number,
      default: 3000
    },
    closable: {
      type: Boolean,
      default: true
    },
    position: {
      type: String,
      default: 'top-right',
      validator: (value) => {
        return ['top-right', 'top-left', 'bottom-right', 'bottom-left', 'top-center', 'bottom-center'].includes(value)
      }
    },
    onClose: {
      type: Function,
      default: () => {}
    }
  },
  setup(props, { emit }) {
    const visible = ref(false)
    let timer = null

    const show = () => {
      visible.value = true
      if (props.duration > 0) {
        timer = setTimeout(() => {
          close()
        }, props.duration)
      }
    }

    const close = () => {
      visible.value = false
      props.onClose()
      if (timer) {
        clearTimeout(timer)
        timer = null
      }
      emit('close')
    }

    // 计算位置样式
    const positionStyle = computed(() => {
      const positionMap = {
        'top-right': { top: '20px', right: '20px' },
        'top-left': { top: '20px', left: '20px' },
        'bottom-right': { bottom: '20px', right: '20px' },
        'bottom-left': { bottom: '20px', left: '20px' },
        'top-center': { top: '20px', left: '50%', transform: 'translateX(-50%)' },
        'bottom-center': { bottom: '20px', left: '50%', transform: 'translateX(-50%)' }
      }
      return positionMap[props.position] || positionMap['top-right']
    })

    onMounted(() => {
      show()
    })

    onUnmounted(() => {
      if (timer) {
        clearTimeout(timer)
      }
    })

    return {
      visible,
      close,
      positionStyle
    }
  }
}
</script>

<style scoped>
.notification {
  position: fixed;
  padding: 15px 20px;
  border-radius: var(--border-radius);
  box-shadow: var(--shadow-md);
  background-color: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  z-index: 3000;
  display: flex;
  align-items: center;
  justify-content: space-between;
  max-width: 400px;
  min-width: 300px;
}

.notification-content {
  flex: 1;
}

.notification-title {
  font-weight: 600;
  margin-bottom: 5px;
  font-size: 14px;
}

.notification-message {
  font-size: 13px;
  line-height: 1.4;
}

.notification-close {
  background: none;
  border: none;
  color: var(--text-secondary);
  font-size: 20px;
  cursor: pointer;
  padding: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-left: 15px;
  border-radius: 50%;
  transition: var(--transition);
}

.notification-close:hover {
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
}

/* Type styles - Clean black and white theme */
.notification.info {
  background-color: var(--bg-secondary);
  border-color: var(--border-color);
  color: var(--text-primary);
}

.notification.success {
  background-color: var(--bg-secondary);
  border-color: var(--border-color);
  color: var(--text-primary);
}

.notification.warning {
  background-color: var(--bg-secondary);
  border-color: var(--border-color);
  color: var(--text-primary);
}

.notification.error {
  background-color: var(--bg-secondary);
  border-color: var(--border-color);
  color: var(--text-primary);
}

/* Animation */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateX(100%);
}

.fade-enter-to,
.fade-leave-from {
  opacity: 1;
  transform: translateX(0);
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .notification {
    max-width: 90%;
    min-width: auto;
    left: 50% !important;
    transform: translateX(-50%) !important;
    right: auto !important;
  }
}
</style>