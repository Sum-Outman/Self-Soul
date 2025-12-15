// Notification system entry point
import { notificationManager } from './notificationManager.js';
import { NOTIFICATION_TYPES, NOTIFICATION_POSITIONS, DEFAULT_NOTIFICATION_DURATION } from './notificationTypes.js';

// Import styles
import './notificationStyles.css';

// Export notification manager instance and types
export {
  notificationManager,
  NOTIFICATION_TYPES,
  NOTIFICATION_POSITIONS,
  DEFAULT_NOTIFICATION_DURATION
};

// Convenience methods to export directly
export const { success, error, warning, info, closeAllNotifications } = notificationManager;