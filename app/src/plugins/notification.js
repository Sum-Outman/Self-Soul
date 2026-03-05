// Notification plugin for Vue 3

// Notification types
export const NOTIFICATION_TYPES = {
  SUCCESS: 'success',
  ERROR: 'error',
  WARNING: 'warning',
  INFO: 'info'
};

// Notification manager class
class NotificationManager {
  constructor() {
    this.notifications = [];
    this.nextId = 1;
  }

  // Create a notification
  create(type, message, options = {}) {
    const id = this.nextId++;
    const notification = {
      id,
      type,
      message,
      title: options.title || this.getDefaultTitle(type),
      duration: options.duration || this.getDefaultDuration(type),
      persistent: options.persistent || false,
      timestamp: new Date().toISOString()
    };

    this.notifications.push(notification);

    // Auto remove after duration if not persistent
    if (!notification.persistent) {
      setTimeout(() => {
        this.remove(id);
      }, notification.duration);
    }

    return notification;
  }

  // Get default title based on type
  getDefaultTitle(type) {
    const titles = {
      [NOTIFICATION_TYPES.SUCCESS]: 'Success',
      [NOTIFICATION_TYPES.ERROR]: 'Error',
      [NOTIFICATION_TYPES.WARNING]: 'Warning',
      [NOTIFICATION_TYPES.INFO]: 'Information'
    };
    return titles[type] || 'Notification';
  }

  // Get default duration based on type
  getDefaultDuration(type) {
    const durations = {
      [NOTIFICATION_TYPES.SUCCESS]: 3000,
      [NOTIFICATION_TYPES.ERROR]: 5000,
      [NOTIFICATION_TYPES.WARNING]: 4000,
      [NOTIFICATION_TYPES.INFO]: 3000
    };
    return durations[type] || 3000;
  }

  // Remove a notification
  remove(id) {
    const index = this.notifications.findIndex(n => n.id === id);
    if (index !== -1) {
      this.notifications.splice(index, 1);
    }
  }

  // Clear all notifications
  clear() {
    this.notifications = [];
  }

  // Get all notifications
  getAll() {
    return this.notifications;
  }

  // Success notification
  success(message, options = {}) {
    return this.create(NOTIFICATION_TYPES.SUCCESS, message, options);
  }

  // Error notification
  error(message, options = {}) {
    return this.create(NOTIFICATION_TYPES.ERROR, message, options);
  }

  // Warning notification
  warning(message, options = {}) {
    return this.create(NOTIFICATION_TYPES.WARNING, message, options);
  }

  // Info notification
  info(message, options = {}) {
    return this.create(NOTIFICATION_TYPES.INFO, message, options);
  }
}

// Create singleton instance
const notificationManager = new NotificationManager();

// Vue plugin installation
export default {
  install(app) {
    // Add notification manager to app config
    app.config.globalProperties.$notify = notificationManager;
    
    // Provide notification manager
    app.provide('notify', notificationManager);
  }
};

// Export notification manager instance
export const notify = notificationManager;

export { NotificationManager };
