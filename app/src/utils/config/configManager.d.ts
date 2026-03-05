// Type declarations for configManager.js

export interface ApiConfig {
  baseUrl: string
  timeout: number
  headers: Record<string, string>
}

export interface SystemConfig {
  frontendPort: number
  backendPort: number
  websocketPort: number
  environment: 'development' | 'staging' | 'production'
  debug: boolean
  logLevel: string
}

export interface ServiceConfig {
  realtimeStream: {
    host: string
    port: number
    path: string
  }
  valueAlignment: {
    host: string
    port: number
  }
}

export interface UiConfig {
  notifications: {
    defaultDuration: number
    maxVisible: number
    position: string
  }
  theme: {
    primaryColor: string
    secondaryColor: string
    darkMode: boolean
  }
}

export interface ModelsConfig {
  defaultLanguageModel: string
  ollama: {
    baseUrl: string
    apiPath: string
    defaultModel: string
  }
}

export interface AppConfig {
  api: ApiConfig
  system: SystemConfig
  services: ServiceConfig
  ui: UiConfig
  models: ModelsConfig
}

export class ConfigManager {
  constructor()
  get<T = any>(path: string, defaultValue?: T): T
  set(path: string, value: any, persist?: boolean): void
  getAll(): AppConfig
  reset(clearPersisted?: boolean): void
  getApiBaseUrl(): string
  getFrontendUrl(): string
  getBackendDocsUrl(): string
  getBackendUrl(endpoint?: string): string
  isDevelopment(): boolean
  isProduction(): boolean
}

declare const configManager: ConfigManager

export { configManager }
export default configManager