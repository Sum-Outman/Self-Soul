import { NOTIFICATION_TYPES, NOTIFICATION_POSITIONS, DEFAULT_NOTIFICATION_DURATION, NOTIFICATION_STATUS } from './notificationTypes.js';

class NotificationManager {
  constructor() {
    this.notifications = [];
    this.notificationCounter = 0;
    this.initializeNotificationContainer();
  }

  // Initialize the notification container in the DOM
  initializeNotificationContainer() {
    // Create container elements for all positions
    Object.values(NOTIFICATION_POSITIONS).forEach(position => {
      let container = document.getElementById(`notification-container-${position}`);
      if (!container) {
        container = document.createElement('div');
        container.id = `notification-container-${position}`;
        container.className = `notification-container notification-container-${position}`;
        document.body.appendChild(container);
      }
    });
  }

  // Create a new notification
  createNotification(options) {
    const notification = {
      id: `notification-${this.notificationCounter++}`,
      message: options.message || 'Notification message',
      type: options.type || NOTIFICATION_TYPES.INFO,
      duration: options.duration || DEFAULT_NOTIFICATION_DURATION.MEDIUM,
      position: options.position || NOTIFICATION_POSITIONS.TOP_RIGHT,
      status: NOTIFICATION_STATUS.PENDING,
      onClose: options.onClose || null,
      timestamp: Date.now()
    };

    this.notifications.push(notification);
    this.displayNotification(notification);
    return notification.id;
  }

  // Display a notification
  displayNotification(notification) {
    const container = document.getElementById(`notification-container-${notification.position}`);
    if (!container) {
      console.error(`Notification container not found for position: ${notification.position}`);
      return;
    }

    // Create notification element
    const notificationEl = document.createElement('div');
    notificationEl.id = notification.id;
    notificationEl.className = `notification notification-${notification.type} notification-${notification.position}`;
    notificationEl.innerHTML = `
      <div class="notification-content">
        <span class="notification-message">${notification.message}</span>
        <button class="notification-close-btn" aria-label="Close notification">×</button>
      </div>
    `;

    // Add close button event listener
    const closeBtn = notificationEl.querySelector('.notification-close-btn');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => this.closeNotification(notification.id));
    }

    // Add animation classes
    notificationEl.classList.add('notification-enter');

    // Append to container
    container.appendChild(notificationEl);

    // Update status
    notification.status = NOTIFICATION_STATUS.DISPLAYED;

    // Set timeout for auto-close
    if (notification.duration > 0) {
      notification.timeoutId = setTimeout(() => {
        this.closeNotification(notification.id, 'timed-out');
      }, notification.duration);
    }
  }

  // Close a notification
  closeNotification(id, reason = 'closed') {
    const notificationIndex = this.notifications.findIndex(n => n.id === id);
    if (notificationIndex === -1) return;

    const notification = this.notifications[notificationIndex];
    notification.status = reason === 'timed-out' ? NOTIFICATION_STATUS.TIMED_OUT : NOTIFICATION_STATUS.CLOSED;

    // Clear timeout if exists
    if (notification.timeoutId) {
      clearTimeout(notification.timeoutId);
    }

    // Remove from DOM with animation
    const notificationEl = document.getElementById(id);
    if (notificationEl) {
      notificationEl.classList.remove('notification-enter');
      notificationEl.classList.add('notification-exit');

      // Wait for animation to complete before removing
      setTimeout(() => {
        if (notificationEl.parentNode) {
          notificationEl.parentNode.removeChild(notificationEl);
        }
      }, 300);
    }

    // Call onClose callback if exists
    if (typeof notification.onClose === 'function') {
      notification.onClose(notification);
    }

    // Remove from array after animation
    setTimeout(() => {
      this.notifications.splice(notificationIndex, 1);
    }, 300);
  }

  // Close all notifications
  closeAllNotifications() {
    this.notifications.forEach(notification => {
      this.closeNotification(notification.id);
    });
  }

  // Get notifications by type
  getNotificationsByType(type) {
    return this.notifications.filter(notification => notification.type === type);
  }

  // Get notifications by status
  getNotificationsByStatus(status) {
    return this.notifications.filter(notification => notification.status === status);
  }

  // Get all notifications
  getAllNotifications() {
    return [...this.notifications];
  }

  // Success notification helper
  success(message, options = {}) {
    return this.createNotification({
      ...options,
      message,
      type: NOTIFICATION_TYPES.SUCCESS
    });
  }

  // Error notification helper
  error(message, options = {}) {
    return this.createNotification({
      ...options,
      message,
      type: NOTIFICATION_TYPES.ERROR
    });
  }

  // Warning notification helper
  warning(message, options = {}) {
    return this.createNotification({
      ...options,
      message,
      type: NOTIFICATION_TYPES.WARNING
    });
  }

  // Info notification helper
  info(message, options = {}) {
    return this.createNotification({
      ...options,
      message,
      type: NOTIFICATION_TYPES.INFO
    });
  }
}

// Export a singleton instance
export const notificationManager = new NotificationManager();