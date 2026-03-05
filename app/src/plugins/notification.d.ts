import { App } from 'vue';

export const NOTIFICATION_TYPES: {
  SUCCESS: string;
  ERROR: string;
  WARNING: string;
  INFO: string;
};

export const notify: any;

declare const notificationPlugin: {
  install(app: App): void;
};

export default notificationPlugin;