import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

// Types
export interface Model {
  id: string
  name: string
  type: 'local' | 'api'
  category: string
  endpoint?: string
  modelName?: string
  active: boolean
  status: 'connected' | 'disconnected' | 'running' | 'stopped' | 'testing' | 'failed'
  cpuUsage: number
  memoryUsage: number
  responseTime: number
  port?: string
  description?: string
  version?: string
  createdAt?: string
  updatedAt?: string
  dependencies?: string[]
  config?: Record<string, any>
}

export interface ModelMetric {
  modelId: string
  timestamp: string
  inferenceCount: number
  avgResponseTime: number
  errorRate: number
  successRate: number
  cpuUsage: number
  memoryUsage: number
}

export interface ModelDependency {
  modelId: string
  requiredBy: string[]
  dependencies: string[]
  status: 'met' | 'partial' | 'missing'
  missingDependencies: string[]
}

export const useModelsStore = defineStore('models', () => {
  // State
  const models = ref<Model[]>([])
  const modelMetrics = ref<ModelMetric[]>([])
  const modelDependencies = ref<ModelDependency[]>([])
  const availableModels = ref<string[]>([])
  const selectedModel = ref<string | null>(null)
  const isLoading = ref(false)
  const error = ref<string | null>(null)
  
  // Model filters
  const filters = ref({
    category: '' as string | '',
    status: '' as string | '',
    active: null as boolean | null,
    search: ''
  })

  // Getters
  const activeModels = computed(() => 
    models.value.filter(model => model.active)
  )
  
  const runningModels = computed(() => 
    models.value.filter(model => model.status === 'running')
  )
  
  const connectedModels = computed(() => 
    models.value.filter(model => model.status === 'connected')
  )
  
  const localModels = computed(() => 
    models.value.filter(model => model.type === 'local')
  )
  
  const apiModels = computed(() => 
    models.value.filter(model => model.type === 'api')
  )
  
  const filteredModels = computed(() => {
    let filtered = models.value
    
    if (filters.value.category) {
      filtered = filtered.filter(model => model.category === filters.value.category)
    }
    
    if (filters.value.status) {
      filtered = filtered.filter(model => model.status === filters.value.status)
    }
    
    if (filters.value.active !== null) {
      filtered = filtered.filter(model => model.active === filters.value.active)
    }
    
    if (filters.value.search) {
      const searchLower = filters.value.search.toLowerCase()
      filtered = filtered.filter(model => 
        model.name.toLowerCase().includes(searchLower) ||
        model.id.toLowerCase().includes(searchLower) ||
        model.description?.toLowerCase().includes(searchLower)
      )
    }
    
    return filtered
  })
  
  const modelCategories = computed(() => {
    const categories = new Set(models.value.map(model => model.category))
    return Array.from(categories).sort()
  })
  
  const totalCpuUsage = computed(() => 
    models.value.reduce((sum, model) => sum + model.cpuUsage, 0)
  )
  
  const totalMemoryUsage = computed(() => 
    models.value.reduce((sum, model) => sum + model.memoryUsage, 0)
  )
  
  const selectedModelDetails = computed(() => {
    if (!selectedModel.value) return null
    return models.value.find(model => model.id === selectedModel.value)
  })
  
  const modelStatusCounts = computed(() => {
    const counts = {
      connected: 0,
      disconnected: 0,
      running: 0,
      stopped: 0,
      testing: 0,
      failed: 0
    }
    
    models.value.forEach(model => {
      if (counts[model.status] !== undefined) {
        counts[model.status]++
      }
    })
    
    return counts
  })

  // Actions
  const setModels = (newModels: Model[]) => {
    models.value = newModels
  }

  const addModel = (model: Model) => {
    const existingIndex = models.value.findIndex(m => m.id === model.id)
    
    if (existingIndex >= 0) {
      models.value[existingIndex] = model
    } else {
      models.value.push(model)
    }
  }

  const updateModel = (modelId: string, updates: Partial<Model>) => {
    const modelIndex = models.value.findIndex(m => m.id === modelId)
    
    if (modelIndex >= 0) {
      models.value[modelIndex] = {
        ...models.value[modelIndex],
        ...updates,
        updatedAt: new Date().toISOString()
      }
    }
  }

  const removeModel = (modelId: string) => {
    models.value = models.value.filter(model => model.id !== modelId)
  }

  const setModelStatus = (modelId: string, status: Model['status']) => {
    updateModel(modelId, { status })
  }

  const setModelActive = (modelId: string, active: boolean) => {
    updateModel(modelId, { active })
  }

  const setSelectedModel = (modelId: string | null) => {
    selectedModel.value = modelId
  }

  const updateModelMetrics = (metrics: ModelMetric[]) => {
    modelMetrics.value = metrics
  }

  const addModelMetric = (metric: ModelMetric) => {
    modelMetrics.value.push(metric)
    
    // Keep only last 1000 metrics
    if (modelMetrics.value.length > 1000) {
      modelMetrics.value = modelMetrics.value.slice(-1000)
    }
  }

  const updateModelDependencies = (dependencies: ModelDependency[]) => {
    modelDependencies.value = dependencies
  }

  const setAvailableModels = (modelIds: string[]) => {
    availableModels.value = modelIds
  }

  const setFilters = (newFilters: Partial<typeof filters.value>) => {
    filters.value = { ...filters.value, ...newFilters }
  }

  const resetFilters = () => {
    filters.value = {
      category: '',
      status: '',
      active: null,
      search: ''
    }
  }

  const setLoading = (loading: boolean) => {
    isLoading.value = loading
  }

  const setError = (errorMessage: string | null) => {
    error.value = errorMessage
  }

  const clearError = () => {
    error.value = null
  }

  const resetModelsState = () => {
    models.value = []
    modelMetrics.value = []
    modelDependencies.value = []
    availableModels.value = []
    selectedModel.value = null
    isLoading.value = false
    error.value = null
    resetFilters()
  }

  // Helper actions
  const getModelById = (modelId: string): Model | undefined => {
    return models.value.find(model => model.id === modelId)
  }

  const getModelDependencies = (modelId: string): ModelDependency | undefined => {
    return modelDependencies.value.find(dep => dep.modelId === modelId)
  }

  const getModelMetrics = (modelId: string, limit = 100): ModelMetric[] => {
    return modelMetrics.value
      .filter(metric => metric.modelId === modelId)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, limit)
  }

  const checkModelDependencies = (modelId: string): 'met' | 'partial' | 'missing' => {
    const dependency = getModelDependencies(modelId)
    return dependency?.status || 'missing'
  }

  const getMissingDependencies = (modelId: string): string[] => {
    const dependency = getModelDependencies(modelId)
    return dependency?.missingDependencies || []
  }

  return {
    // State
    models,
    modelMetrics,
    modelDependencies,
    availableModels,
    selectedModel,
    isLoading,
    error,
    filters,
    
    // Getters
    activeModels,
    runningModels,
    connectedModels,
    localModels,
    apiModels,
    filteredModels,
    modelCategories,
    totalCpuUsage,
    totalMemoryUsage,
    selectedModelDetails,
    modelStatusCounts,
    
    // Actions
    setModels,
    addModel,
    updateModel,
    removeModel,
    setModelStatus,
    setModelActive,
    setSelectedModel,
    updateModelMetrics,
    addModelMetric,
    updateModelDependencies,
    setAvailableModels,
    setFilters,
    resetFilters,
    setLoading,
    setError,
    clearError,
    resetModelsState,
    
    // Helper actions
    getModelById,
    getModelDependencies,
    getModelMetrics,
    checkModelDependencies,
    getMissingDependencies
  }
})