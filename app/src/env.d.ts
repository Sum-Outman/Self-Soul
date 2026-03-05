/// <reference types="vite/client" />

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  readonly VITE_APP_TITLE: string
  // add more env variables as needed
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

// Vue.js global properties
import { ComponentCustomProperties } from 'vue'
import { ConfigManager } from './utils/config/configManager.js'

declare module '@vue/runtime-core' {
  interface ComponentCustomProperties {
    $isEnglish: boolean
    $notify: any
    $config: any
    $getConfig: (path: string, defaultValue?: any) => any
    $getFrontendUrl: () => string
    $getBackendDocsUrl: () => string
    $getBackendUrl: (endpoint?: string) => string
  }
}