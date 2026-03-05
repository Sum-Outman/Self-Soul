import { App } from 'vue';
import { configManager } from './configManager.js';

export const ConfigPlugin: {
  install(app: App, options?: any): void;
};

export { configManager };
export default ConfigPlugin;