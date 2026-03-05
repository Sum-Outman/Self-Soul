import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

// Types
export interface RobotStatus {
  connected: boolean
  initialized: boolean
  operatingMode: 'manual' | 'autonomous' | 'semi-autonomous'
  batteryLevel: number
  batteryStatus: 'charging' | 'discharging' | 'full' | 'low' | 'unknown'
  cpuTemperature: number
  motorTemperature: number
  uptime: number
  lastUpdate: string
}

export interface HardwareComponent {
  id: string
  name: string
  type: 'sensor' | 'actuator' | 'motor' | 'camera' | 'lidar' | 'gpio' | 'serial'
  connected: boolean
  status: 'operational' | 'degraded' | 'faulty' | 'offline'
  port?: string
  address?: string
  driver?: string
  details?: Record<string, any>
}

export interface SensorData {
  sensorId: string
  timestamp: string
  values: Record<string, number | string>
  unit?: string
}

export interface GpioPin {
  pin: number
  mode: 'input' | 'output' | 'pwm'
  value: number
  pull?: 'up' | 'down' | 'none'
  frequency?: number
  dutyCycle?: number
}

export interface SerialPort {
  port: string
  baudRate: number
  dataBits: 5 | 6 | 7 | 8
  stopBits: 1 | 1.5 | 2
  parity: 'none' | 'odd' | 'even' | 'mark' | 'space'
  connected: boolean
  lastActivity?: string
}

export const useRobotStore = defineStore('robot', () => {
  // State
  const robotStatus = ref<RobotStatus | null>(null)
  const hardwareComponents = ref<HardwareComponent[]>([])
  const sensorData = ref<SensorData[]>([])
  const gpioPins = ref<GpioPin[]>([])
  const serialPorts = ref<SerialPort[]>([])
  const isHardwareInitializing = ref(false)
  const lastHardwareCheck = ref<Date | null>(null)
  const hardwareErrors = ref<string[]>([])
  
  // Robot configuration
  const robotConfig = ref({
    autoDetectHardware: true,
    hardwareCheckInterval: 5000,
    sensorPollingInterval: 1000,
    gpioPollingInterval: 100,
    serialBufferSize: 1024,
    motorSafeMode: true,
    emergencyStopEnabled: true,
    logHardwareEvents: true
  })

  // Getters
  const isConnected = computed(() => robotStatus.value?.connected || false)
  const isInitialized = computed(() => robotStatus.value?.initialized || false)
  const batteryLevel = computed(() => robotStatus.value?.batteryLevel || 0)
  const batteryStatus = computed(() => robotStatus.value?.batteryStatus || 'unknown')
  const operatingMode = computed(() => robotStatus.value?.operatingMode || 'manual')
  
  const connectedHardware = computed(() => 
    hardwareComponents.value.filter(component => component.connected)
  )
  
  const faultyHardware = computed(() => 
    hardwareComponents.value.filter(component => component.status === 'faulty')
  )
  
  const sensors = computed(() => 
    hardwareComponents.value.filter(component => component.type === 'sensor')
  )
  
  const actuators = computed(() => 
    hardwareComponents.value.filter(component => component.type === 'actuator')
  )
  
  const motors = computed(() => 
    hardwareComponents.value.filter(component => component.type === 'motor')
  )
  
  const latestSensorData = computed(() => {
    if (sensorData.value.length === 0) return {}
    
    const latest: Record<string, SensorData> = {}
    sensorData.value.forEach(data => {
      if (!latest[data.sensorId] || new Date(data.timestamp) > new Date(latest[data.sensorId].timestamp)) {
        latest[data.sensorId] = data
      }
    })
    
    return latest
  })
  
  const inputPins = computed(() => 
    gpioPins.value.filter(pin => pin.mode === 'input')
  )
  
  const outputPins = computed(() => 
    gpioPins.value.filter(pin => pin.mode === 'output')
  )
  
  const pwmPins = computed(() => 
    gpioPins.value.filter(pin => pin.mode === 'pwm')
  )
  
  const connectedSerialPorts = computed(() => 
    serialPorts.value.filter(port => port.connected)
  )

  const hasHardwareErrors = computed(() => hardwareErrors.value.length > 0)
  const hardwareErrorCount = computed(() => hardwareErrors.value.length)

  // Actions
  const updateRobotStatus = (status: Partial<RobotStatus>) => {
    const lastUpdate = new Date().toISOString()
    
    robotStatus.value = {
      connected: status.connected || false,
      initialized: status.initialized || false,
      operatingMode: status.operatingMode || 'manual',
      batteryLevel: status.batteryLevel || 0,
      batteryStatus: status.batteryStatus || 'unknown',
      cpuTemperature: status.cpuTemperature || 0,
      motorTemperature: status.motorTemperature || 0,
      uptime: status.uptime || 0,
      lastUpdate
    }
  }

  const updateHardwareComponents = (components: HardwareComponent[]) => {
    hardwareComponents.value = components
    lastHardwareCheck.value = new Date()
  }

  const addHardwareComponent = (component: HardwareComponent) => {
    const existingIndex = hardwareComponents.value.findIndex(c => c.id === component.id)
    
    if (existingIndex >= 0) {
      hardwareComponents.value[existingIndex] = component
    } else {
      hardwareComponents.value.push(component)
    }
  }

  const updateHardwareStatus = (componentId: string, status: HardwareComponent['status'], connected?: boolean) => {
    const component = hardwareComponents.value.find(c => c.id === componentId)
    
    if (component) {
      component.status = status
      if (connected !== undefined) {
        component.connected = connected
      }
    }
  }

  const addSensorData = (data: SensorData) => {
    sensorData.value.push(data)
    
    // Keep only last 1000 readings per sensor
    const sensorReadings = sensorData.value.filter(d => d.sensorId === data.sensorId)
    if (sensorReadings.length > 1000) {
      const toRemove = sensorReadings.length - 1000
      sensorData.value = sensorData.value.filter((d, index) => 
        d.sensorId !== data.sensorId || index >= toRemove
      )
    }
  }

  const updateGpioPins = (pins: GpioPin[]) => {
    gpioPins.value = pins
  }

  const updateGpioPin = (pinNumber: number, updates: Partial<GpioPin>) => {
    const pinIndex = gpioPins.value.findIndex(p => p.pin === pinNumber)
    
    if (pinIndex >= 0) {
      gpioPins.value[pinIndex] = { ...gpioPins.value[pinIndex], ...updates }
    } else {
      gpioPins.value.push({ pin: pinNumber, mode: 'input', value: 0, ...updates })
    }
  }

  const updateSerialPorts = (ports: SerialPort[]) => {
    serialPorts.value = ports
  }

  const updateSerialPort = (portName: string, updates: Partial<SerialPort>) => {
    const portIndex = serialPorts.value.findIndex(p => p.port === portName)
    
    if (portIndex >= 0) {
      serialPorts.value[portIndex] = { ...serialPorts.value[portIndex], ...updates }
    } else {
      serialPorts.value.push({ port: portName, baudRate: 9600, dataBits: 8, stopBits: 1, parity: 'none', connected: false, ...updates })
    }
  }

  const setHardwareInitializing = (initializing: boolean) => {
    isHardwareInitializing.value = initializing
  }

  const addHardwareError = (error: string) => {
    const timestamp = new Date().toLocaleTimeString()
    hardwareErrors.value.push(`[${timestamp}] ${error}`)
    
    // Keep only last 50 errors
    if (hardwareErrors.value.length > 50) {
      hardwareErrors.value = hardwareErrors.value.slice(-50)
    }
  }

  const clearHardwareErrors = () => {
    hardwareErrors.value = []
  }

  const updateRobotConfig = (config: Partial<typeof robotConfig.value>) => {
    robotConfig.value = { ...robotConfig.value, ...config }
  }

  const resetRobotState = () => {
    robotStatus.value = null
    hardwareComponents.value = []
    sensorData.value = []
    gpioPins.value = []
    serialPorts.value = []
    isHardwareInitializing.value = false
    lastHardwareCheck.value = null
    hardwareErrors.value = []
    
    robotConfig.value = {
      autoDetectHardware: true,
      hardwareCheckInterval: 5000,
      sensorPollingInterval: 1000,
      gpioPollingInterval: 100,
      serialBufferSize: 1024,
      motorSafeMode: true,
      emergencyStopEnabled: true,
      logHardwareEvents: true
    }
  }

  // Hardware control actions
  const setGpioPinValue = (pinNumber: number, value: number) => {
    updateGpioPin(pinNumber, { value })
  }

  const setGpioPinMode = (pinNumber: number, mode: GpioPin['mode']) => {
    updateGpioPin(pinNumber, { mode })
  }

  const setPwmFrequency = (pinNumber: number, frequency: number) => {
    updateGpioPin(pinNumber, { frequency })
  }

  const setPwmDutyCycle = (pinNumber: number, dutyCycle: number) => {
    updateGpioPin(pinNumber, { dutyCycle })
  }

  const connectSerialPort = (portName: string, baudRate: number = 9600) => {
    updateSerialPort(portName, { connected: true, baudRate })
  }

  const disconnectSerialPort = (portName: string) => {
    updateSerialPort(portName, { connected: false })
  }

  const setOperatingMode = (mode: RobotStatus['operatingMode']) => {
    if (robotStatus.value) {
      robotStatus.value.operatingMode = mode
      robotStatus.value.lastUpdate = new Date().toISOString()
    }
  }

  return {
    // State
    robotStatus,
    hardwareComponents,
    sensorData,
    gpioPins,
    serialPorts,
    isHardwareInitializing,
    lastHardwareCheck,
    hardwareErrors,
    robotConfig,
    
    // Getters
    isConnected,
    isInitialized,
    batteryLevel,
    batteryStatus,
    operatingMode,
    connectedHardware,
    faultyHardware,
    sensors,
    actuators,
    motors,
    latestSensorData,
    inputPins,
    outputPins,
    pwmPins,
    connectedSerialPorts,
    hasHardwareErrors,
    hardwareErrorCount,
    
    // Actions
    updateRobotStatus,
    updateHardwareComponents,
    addHardwareComponent,
    updateHardwareStatus,
    addSensorData,
    updateGpioPins,
    updateGpioPin,
    updateSerialPorts,
    updateSerialPort,
    setHardwareInitializing,
    addHardwareError,
    clearHardwareErrors,
    updateRobotConfig,
    resetRobotState,
    
    // Hardware control actions
    setGpioPinValue,
    setGpioPinMode,
    setPwmFrequency,
    setPwmDutyCycle,
    connectSerialPort,
    disconnectSerialPort,
    setOperatingMode
  }
})