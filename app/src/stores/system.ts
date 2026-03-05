import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

// Types
export interface SystemStats {
  cpu_usage: number
  memory_usage: number
  disk_usage: number
  gpu_usage?: number
  gpu_memory?: number
  system_uptime: number
  process_count: number
  network_rx: number
  network_tx: number
  timestamp: string
}

export interface SystemHealth {
  status: 'healthy' | 'warning' | 'critical'
  checks: {
    database: boolean
    api: boolean
    websocket: boolean
    disk_space: boolean
    memory: boolean
  }
  warnings: string[]
  criticals: string[]
  last_checked: string
}

export interface ServiceStatus {
  name: string
  status: 'running' | 'stopped' | 'error'
  uptime: number
  cpu_usage: number
  memory_usage: number
  last_checked: string
}

export const useSystemStore = defineStore('system', () => {
  // State
  const systemStats = ref<SystemStats | null>(null)
  const systemHealth = ref<SystemHealth | null>(null)
  const serviceStatuses = ref<ServiceStatus[]>([])
  const isConnected = ref(true)
  const lastConnectionCheck = ref<Date | null>(null)
  const systemNotifications = ref<string[]>([])
  
  // System configuration
  const systemConfig = ref({
    autoRefresh: true,
    refreshInterval: 5000, // 5 seconds
    notificationEnabled: true,
    performanceMonitoring: true,
    logLevel: 'info' as 'debug' | 'info' | 'warn' | 'error'
  })

  // Getters
  const cpuUsage = computed(() => systemStats.value?.cpu_usage || 0)
  const memoryUsage = computed(() => systemStats.value?.memory_usage || 0)
  const diskUsage = computed(() => systemStats.value?.disk_usage || 0)
  const systemUptime = computed(() => systemStats.value?.system_uptime || 0)
  
  const overallHealth = computed(() => {
    if (!systemHealth.value) return 'unknown'
    return systemHealth.value.status
  })
  
  const hasWarnings = computed(() => {
    return (systemHealth.value?.warnings?.length || 0) > 0
  })
  
  const hasCriticals = computed(() => {
    return (systemHealth.value?.criticals?.length || 0) > 0
  })
  
  const runningServices = computed(() => {
    return serviceStatuses.value.filter(service => service.status === 'running')
  })
  
  const serviceCount = computed(() => serviceStatuses.value.length)
  
  const formattedUptime = computed(() => {
    const uptime = systemUptime.value
    const days = Math.floor(uptime / 86400)
    const hours = Math.floor((uptime % 86400) / 3600)
    const minutes = Math.floor((uptime % 3600) / 60)
    const seconds = Math.floor(uptime % 60)
    
    if (days > 0) return `${days}d ${hours}h ${minutes}m`
    if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`
    if (minutes > 0) return `${minutes}m ${seconds}s`
    return `${seconds}s`
  })

  // Actions
  const updateSystemStats = (stats: Partial<SystemStats>) => {
    const timestamp = new Date().toISOString()
    
    systemStats.value = {
      cpu_usage: stats.cpu_usage || 0,
      memory_usage: stats.memory_usage || 0,
      disk_usage: stats.disk_usage || 0,
      gpu_usage: stats.gpu_usage,
      gpu_memory: stats.gpu_memory,
      system_uptime: stats.system_uptime || 0,
      process_count: stats.process_count || 0,
      network_rx: stats.network_rx || 0,
      network_tx: stats.network_tx || 0,
      timestamp
    }
  }

  const updateSystemHealth = (health: Partial<SystemHealth>) => {
    systemHealth.value = {
      status: health.status || 'healthy',
      checks: {
        database: health.checks?.database || false,
        api: health.checks?.api || false,
        websocket: health.checks?.websocket || false,
        disk_space: health.checks?.disk_space || false,
        memory: health.checks?.memory || false,
      },
      warnings: health.warnings || [],
      criticals: health.criticals || [],
      last_checked: health.last_checked || new Date().toISOString()
    }
  }

  const updateServiceStatus = (service: ServiceStatus) => {
    const existingIndex = serviceStatuses.value.findIndex(s => s.name === service.name)
    
    if (existingIndex >= 0) {
      serviceStatuses.value[existingIndex] = service
    } else {
      serviceStatuses.value.push(service)
    }
  }

  const updateServiceStatuses = (services: ServiceStatus[]) => {
    serviceStatuses.value = services
  }

  const setConnectionStatus = (connected: boolean) => {
    isConnected.value = connected
    lastConnectionCheck.value = new Date()
    
    if (!connected) {
      addNotification('Lost connection to backend server')
    } else if (lastConnectionCheck.value) {
      const downtime = new Date().getTime() - lastConnectionCheck.value.getTime()
      if (downtime > 5000) {
        addNotification('Reconnected to backend server')
      }
    }
  }

  const addNotification = (message: string) => {
    if (systemConfig.value.notificationEnabled) {
      const timestamp = new Date().toLocaleTimeString()
      systemNotifications.value.push(`[${timestamp}] ${message}`)
      
      // Keep only last 50 notifications
      if (systemNotifications.value.length > 50) {
        systemNotifications.value = systemNotifications.value.slice(-50)
      }
    }
  }

  const clearNotifications = () => {
    systemNotifications.value = []
  }

  const updateConfig = (config: Partial<typeof systemConfig.value>) => {
    systemConfig.value = { ...systemConfig.value, ...config }
  }

  const resetSystemState = () => {
    systemStats.value = null
    systemHealth.value = null
    serviceStatuses.value = []
    isConnected.value = true
    lastConnectionCheck.value = null
    systemNotifications.value = []
    
    systemConfig.value = {
      autoRefresh: true,
      refreshInterval: 5000,
      notificationEnabled: true,
      performanceMonitoring: true,
      logLevel: 'info'
    }
  }

  // Health check helpers
  const checkDatabaseHealth = (): boolean => {
    return systemHealth.value?.checks.database || false
  }

  const checkApiHealth = (): boolean => {
    return systemHealth.value?.checks.api || false
  }

  const checkWebSocketHealth = (): boolean => {
    return systemHealth.value?.checks.websocket || false
  }

  const checkDiskSpaceHealth = (): boolean => {
    return systemHealth.value?.checks.disk_space || false
  }

  const checkMemoryHealth = (): boolean => {
    return systemHealth.value?.checks.memory || false
  }

  return {
    // State
    systemStats,
    systemHealth,
    serviceStatuses,
    isConnected,
    lastConnectionCheck,
    systemNotifications,
    systemConfig,
    
    // Getters
    cpuUsage,
    memoryUsage,
    diskUsage,
    systemUptime,
    overallHealth,
    hasWarnings,
    hasCriticals,
    runningServices,
    serviceCount,
    formattedUptime,
    
    // Actions
    updateSystemStats,
    updateSystemHealth,
    updateServiceStatus,
    updateServiceStatuses,
    setConnectionStatus,
    addNotification,
    clearNotifications,
    updateConfig,
    resetSystemState,
    
    // Health check helpers
    checkDatabaseHealth,
    checkApiHealth,
    checkWebSocketHealth,
    checkDiskSpaceHealth,
    checkMemoryHealth
  }
})