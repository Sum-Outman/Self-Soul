<template>
  <div class="notification-tester">
    <h2>Notification System Test</h2>
    <div class="test-controls">
      <h3>Notification Types</h3>
      <button @click="showInfoNotification" class="test-btn">Show Info</button>
      <button @click="showSuccessNotification" class="test-btn">Show Success</button>
      <button @click="showWarningNotification" class="test-btn">Show Warning</button>
      <button @click="showErrorNotification" class="test-btn">Show Error</button>
    </div>
    <div class="test-controls">
      <h3>Positions</h3>
      <button @click="showTopRightNotification" class="test-btn">Top Right</button>
      <button @click="showTopLeftNotification" class="test-btn">Top Left</button>
      <button @click="showBottomRightNotification" class="test-btn">Bottom Right</button>
      <button @click="showBottomLeftNotification" class="test-btn">Bottom Left</button>
      <button @click="showTopCenterNotification" class="test-btn">Top Center</button>
      <button @click="showBottomCenterNotification" class="test-btn">Bottom Center</button>
    </div>
    <div class="test-controls">
      <h3>Duration</h3>
      <button @click="showShortNotification" class="test-btn">Short (1s)</button>
      <button @click="showMediumNotification" class="test-btn">Medium (3s)</button>
      <button @click="showLongNotification" class="test-btn">Long (5s)</button>
      <button @click="showPermanentNotification" class="test-btn">Permanent</button>
    </div>
    <div class="test-controls">
      <h3>Other Configurations</h3>
      <button @click="showWithTitleNotification" class="test-btn">With Title</button>
      <button @click="showWithoutCloseNotification" class="test-btn">No Close Button</button>
      <button @click="showMultipleNotification" class="test-btn">Multiple Notifications</button>
    </div>
  </div>
</template>

<script>
import { notify } from '../plugins/notification.js'

export default {
  name: 'NotificationTester',
  methods: {
    showInfoNotification() {
      notify.info('This is an information notification')
    },
    
    showSuccessNotification() {
      notify.success('This is a success notification')
    },
    
    showWarningNotification() {
      notify.warning('This is a warning notification')
    },
    
    showErrorNotification() {
      notify.error('This is an error notification')
    },
    
    showTopRightNotification() {
      notify.info('Top right notification', { position: 'top-right' })
    },
    
    showTopLeftNotification() {
      notify.info('Top left notification', { position: 'top-left' })
    },
    
    showBottomRightNotification() {
      notify.info('Bottom right notification', { position: 'bottom-right' })
    },
    
    showBottomLeftNotification() {
      notify.info('Bottom left notification', { position: 'bottom-left' })
    },
    
    showTopCenterNotification() {
      notify.info('Top center notification', { position: 'top-center' })
    },
    
    showBottomCenterNotification() {
      notify.info('Bottom center notification', { position: 'bottom-center' })
    },
    
    showShortNotification() {
      notify.info('Short duration notification', { duration: 1000 })
    },
    
    showMediumNotification() {
      notify.info('Medium duration notification', { duration: 3000 })
    },
    
    showLongNotification() {
      notify.info('Long duration notification', { duration: 5000 })
    },
    
    showPermanentNotification() {
      notify.info('Permanent notification - click to close', { duration: 0 })
    },
    
    showWithTitleNotification() {
      notify.info('This notification has a title and message', { 
        title: 'Notification Title'
      })
    },
    
    showWithoutCloseNotification() {
      notify.info('This notification cannot be closed manually', { 
        closable: false,
        duration: 3000
      })
    },
    
    showMultipleNotification() {
      notify.info('First notification')
      setTimeout(() => {
        notify.success('Second notification')
      }, 500)
      setTimeout(() => {
        notify.warning('Third notification')
      }, 1000)
      setTimeout(() => {
        notify.error('Fourth notification')
      }, 1500)
    }
  }
}
</script>

<style scoped>
.notification-tester {
  padding: 20px;
  max-width: 800px;
  margin: 0 auto;
}

.notification-tester h2 {
  font-size: 24px;
  margin-bottom: 20px;
  color: var(--text-primary);
}

.notification-tester h3 {
  font-size: 18px;
  margin: 20px 0 10px 0;
  color: var(--text-secondary);
}

.test-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 20px;
}

.test-btn {
  padding: 8px 16px;
  border: 1px solid var(--border-color);
  background-color: var(--bg-secondary);
  color: var(--text-primary);
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 14px;
  transition: var(--transition);
}

.test-btn:hover {
  background-color: var(--bg-tertiary);
  border-color: var(--primary-color);
}
</style>