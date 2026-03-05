import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

// Types
export interface TrainingConfig {
  modelIds: string[]
  datasetIds: string[]
  epochs: number
  batchSize: number
  learningRate: number
  device: string
  validationSplit: number
  saveFrequency: number
}

export interface TrainingJob {
  id: string
  modelIds: string[]
  datasetIds: string[]
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped'
  progress: number
  startTime: string
  endTime?: string
  device: string
  elapsedTime: string
  metrics?: {
    loss: number
    accuracy: number
    validationLoss: number
    validationAccuracy: number
  }
}

export interface ResourceUsage {
  cpu: number
  memory: number
  gpu?: number
  gpuMemory?: number
  timestamp: string
}

export interface DeviceInfo {
  id: string
  name: string
  type: 'cpu' | 'cuda' | 'mps'
  available: boolean
  recommended: boolean
  details: string
}

export const useTrainingStore = defineStore('training', () => {
  // State
  const activeJobs = ref<TrainingJob[]>([])
  const trainingHistory = ref<TrainingJob[]>([])
  const selectedModels = ref<string[]>([])
  const selectedDatasets = ref<string[]>([])
  const isTraining = ref(false)
  const currentJobId = ref<string | null>(null)
  const resourceUsage = ref<ResourceUsage[]>([])
  const availableDevices = ref<DeviceInfo[]>([])
  const selectedDevice = ref<string>('auto')
  
  // Current training configuration
  const trainingConfig = ref<TrainingConfig>({
    modelIds: [],
    datasetIds: [],
    epochs: 10,
    batchSize: 32,
    learningRate: 0.001,
    device: 'auto',
    validationSplit: 0.2,
    saveFrequency: 5
  })

  // Getters
  const activeJobCount = computed(() => activeJobs.value.length)
  const latestResourceUsage = computed(() => 
    resourceUsage.value.length > 0 ? resourceUsage.value[resourceUsage.value.length - 1] : null
  )
  const recommendedDevice = computed(() => 
    availableDevices.value.find(device => device.recommended) || null
  )
  const totalTrainingTime = computed(() => {
    return trainingHistory.value.reduce((total, job) => {
      const start = new Date(job.startTime)
      const end = job.endTime ? new Date(job.endTime) : new Date()
      return total + (end.getTime() - start.getTime())
    }, 0)
  })

  // Actions
  const startTraining = async (config: Partial<TrainingConfig>) => {
    isTraining.value = true
    
    // Merge with current config
    trainingConfig.value = {
      ...trainingConfig.value,
      ...config,
      modelIds: selectedModels.value,
      datasetIds: selectedDatasets.value,
      device: selectedDevice.value
    }
    
    // Create new job
    const job: TrainingJob = {
      id: `job_${Date.now()}`,
      modelIds: trainingConfig.value.modelIds,
      datasetIds: trainingConfig.value.datasetIds,
      status: 'running',
      progress: 0,
      startTime: new Date().toISOString(),
      device: trainingConfig.value.device,
      elapsedTime: '00:00:00'
    }
    
    currentJobId.value = job.id
    activeJobs.value.push(job)
    
    return job.id
  }

  const stopTraining = async (jobId?: string) => {
    const targetJobId = jobId || currentJobId.value
    if (!targetJobId) return
    
    const job = activeJobs.value.find(j => j.id === targetJobId)
    if (job) {
      job.status = 'stopped'
      job.endTime = new Date().toISOString()
      
      // Move to history
      trainingHistory.value.push(job)
      activeJobs.value = activeJobs.value.filter(j => j.id !== targetJobId)
    }
    
    if (targetJobId === currentJobId.value) {
      currentJobId.value = null
      isTraining.value = false
    }
  }

  const updateJobProgress = (jobId: string, progress: number, metrics?: TrainingJob['metrics']) => {
    const job = activeJobs.value.find(j => j.id === jobId)
    if (job) {
      job.progress = progress
      if (metrics) job.metrics = metrics
      
      // Update elapsed time
      const start = new Date(job.startTime)
      const now = new Date()
      const diff = now.getTime() - start.getTime()
      const hours = Math.floor(diff / 3600000)
      const minutes = Math.floor((diff % 3600000) / 60000)
      const seconds = Math.floor((diff % 60000) / 1000)
      job.elapsedTime = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
    }
  }

  const completeJob = (jobId: string, metrics?: TrainingJob['metrics']) => {
    const job = activeJobs.value.find(j => j.id === jobId)
    if (job) {
      job.status = 'completed'
      job.progress = 100
      job.endTime = new Date().toISOString()
      if (metrics) job.metrics = metrics
      
      // Move to history
      trainingHistory.value.push(job)
      activeJobs.value = activeJobs.value.filter(j => j.id !== jobId)
      
      if (jobId === currentJobId.value) {
        currentJobId.value = null
        isTraining.value = false
      }
    }
  }

  const failJob = (jobId: string, error?: string) => {
    const job = activeJobs.value.find(j => j.id === jobId)
    if (job) {
      job.status = 'failed'
      job.endTime = new Date().toISOString()
      
      // Move to history with error info
      trainingHistory.value.push(job)
      activeJobs.value = activeJobs.value.filter(j => j.id !== jobId)
      
      if (jobId === currentJobId.value) {
        currentJobId.value = null
        isTraining.value = false
      }
    }
  }

  const updateResourceUsage = (usage: Omit<ResourceUsage, 'timestamp'>) => {
    const newUsage: ResourceUsage = {
      ...usage,
      timestamp: new Date().toISOString()
    }
    
    resourceUsage.value.push(newUsage)
    
    // Keep only last 100 readings
    if (resourceUsage.value.length > 100) {
      resourceUsage.value = resourceUsage.value.slice(-100)
    }
  }

  const setAvailableDevices = (devices: DeviceInfo[]) => {
    availableDevices.value = devices
    
    // Auto-select recommended device if available
    const recommended = devices.find(d => d.recommended)
    if (recommended && !selectedDevice.value) {
      selectedDevice.value = recommended.id
    }
  }

  const setSelectedDevice = (deviceId: string) => {
    selectedDevice.value = deviceId
  }

  const clearTrainingHistory = () => {
    trainingHistory.value = []
  }

  const resetTrainingState = () => {
    activeJobs.value = []
    trainingHistory.value = []
    selectedModels.value = []
    selectedDatasets.value = []
    isTraining.value = false
    currentJobId.value = null
    resourceUsage.value = []
    selectedDevice.value = 'auto'
    
    trainingConfig.value = {
      modelIds: [],
      datasetIds: [],
      epochs: 10,
      batchSize: 32,
      learningRate: 0.001,
      device: 'auto',
      validationSplit: 0.2,
      saveFrequency: 5
    }
  }

  return {
    // State
    activeJobs,
    trainingHistory,
    selectedModels,
    selectedDatasets,
    isTraining,
    currentJobId,
    resourceUsage,
    availableDevices,
    selectedDevice,
    trainingConfig,
    
    // Getters
    activeJobCount,
    latestResourceUsage,
    recommendedDevice,
    totalTrainingTime,
    
    // Actions
    startTraining,
    stopTraining,
    updateJobProgress,
    completeJob,
    failJob,
    updateResourceUsage,
    setAvailableDevices,
    setSelectedDevice,
    clearTrainingHistory,
    resetTrainingState
  }
})