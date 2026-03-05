// Notification types
export const NOTIFICATION_TYPES = {
  SUCCESS: 'success',
  ERROR: 'error',
  WARNING: 'warning',
  INFO: 'info'
};

// Notification positions
export const NOTIFICATION_POSITIONS = {
  TOP_LEFT: 'top-left',
  TOP_CENTER: 'top-center',
  TOP_RIGHT: 'top-right',
  BOTTOM_LEFT: 'bottom-left',
  BOTTOM_CENTER: 'bottom-center',
  BOTTOM_RIGHT: 'bottom-right'
};

// Default notification duration in milliseconds
export const DEFAULT_NOTIFICATION_DURATION = {
  SHORT: 3000,
  MEDIUM: 5000,
  LONG: 8000
};

// Notification statuses
export const NOTIFICATION_STATUS = {
  PENDING: 'pending',
  DISPLAYED: 'displayed',
  CLOSED: 'closed',
  TIMED_OUT: 'timed-out'
};