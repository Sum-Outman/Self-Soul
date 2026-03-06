<template>
  <div class="robot-settings-view">
    <!-- Page title and status -->
    <div class="page-header">
      <h2>Robot Hardware Settings System</h2>
      <div class="robot-status">
        <span class="status-label">Status:</span>
        <span class="status-value" :class="robotState.status">{{ robotState.statusText }}</span>
        <span class="battery-indicator">🔋 {{ robotState.battery }}%</span>
        <span class="connection-status" :class="robotState.connected ? 'connected' : 'disconnected'">
          {{ robotState.connected ? 'Connected' : 'Not Connected' }}
        </span>
      </div>
    </div>

    <!-- Hardware connection settings -->
    <div class="hardware-control-section">
      <h3>Hardware Connection Management</h3>
      <div class="hardware-status">
        <p>Hardware Status: {{ hardwareState.initialized ? 'Initialized' : 'Not Initialized' }}</p>
        <p>Detected Hardware: {{ hardwareState.detectedDevices?.joints || 0 }} joints, {{ hardwareState.detectedDevices?.sensors || 0 }} sensors, {{ hardwareState.detectedDevices?.cameras || 0 }} cameras</p>
      </div>
      
      <!-- Hardware list display -->
      <div class="hardware-lists">
        <!-- Connected hardware list -->
        <div class="hardware-list-section">
          <div class="list-header" @click="toggleConnectedHardwareList">
            <h4>📊 Connected Hardware List ({{ totalConnectedHardwareCount }})</h4>
            <span class="toggle-icon">{{ showConnectedHardwareList ? '▼' : '▶' }}</span>
          </div>
          <div v-if="showConnectedHardwareList" class="list-content">
            <HardwareCategory
              v-for="category in connectedHardwareCategories"
              :key="category.title"
              :title="category.title"
              :items="category.items"
              :item-type="category.itemType"
              :status="category.status"
              :status-getter="category.statusGetter"
            />
          </div>
        </div>
        
        <!-- All available hardware list -->
        <div class="hardware-list-section">
          <div class="list-header" @click="toggleAvailableHardwareList">
            <h4>📋 All Available Hardware Types</h4>
            <span class="toggle-icon">{{ showAvailableHardwareList ? '▼' : '▶' }}</span>
          </div>
          <div v-if="showAvailableHardwareList" class="list-content">
            <HardwareCategory
              v-for="category in availableHardwareCategories"
              :key="category.title"
              :title="category.title"
              :items="category.items"
              :item-type="category.itemType"
              :status="category.status"
              :status-getter="category.statusGetter"
            />
          </div>
        </div>
      </div>
      
      <div class="hardware-buttons">
        <button @click="initializeHardware" :disabled="hardwareState.loading" class="btn btn-primary">
          {{ hardwareState.loading ? 'Initializing...' : 'Initialize Hardware' }}
        </button>
        <button @click="fetchDetectedHardware" class="btn btn-secondary">Detect Hardware</button>
        <button @click="disconnectHardware" class="btn btn-danger">Disconnect</button>
        <button @click="toggleVoiceControl" class="btn btn-info">
          {{ voiceControlState.enabled ? 'Stop Voice Control' : 'Start Voice Control' }}
        </button>
        <button @click="showVoiceCommandsHelp" class="btn btn-secondary" title="Show available voice commands">
          Voice Commands Help
        </button>
        <button @click="openSettingsDialog" class="btn btn-warning">Settings</button>
        <button @click="openChatDialog" class="btn btn-success">Chat with Robot</button>
      </div>
    </div>

    <!-- Joint Settings -->
    <div class="joint-control-section">
      <div class="joint-control-header">
        <h3>Joint Settings</h3>
        <button @click="openJointManagementDialog" class="btn btn-info">Manage Joints</button>
      </div>
      <div class="joint-controls">
        <div class="joint-control" v-for="joint in jointList" :key="joint.id">
          <div class="joint-control-row">
            <div class="joint-label">
              <label>{{ joint.name }}:</label>
              <span class="joint-value-display">{{ joint.value }}°</span>
            </div>
            
            <div class="joint-control-inputs">
              <!-- Fine-tuning buttons -->
              <div class="joint-fine-tune">
                <button @click="adjustJoint(joint.id, -joint.step)" class="btn btn-sm btn-secondary" :title="`Decrease by ${joint.step}°`">-</button>
                <button @click="adjustJoint(joint.id, joint.step)" class="btn btn-sm btn-secondary" :title="`Increase by ${joint.step}°`">+</button>
              </div>
              
              <!-- Precise numeric input -->
              <div class="joint-numeric-input">
                <input type="number" :min="joint.min" :max="joint.max" :step="joint.step" v-model.number="joint.value" 
                       @change="updateJoint(joint.id, joint.value)" 
                       @input="onJointInputChange(joint.id, $event)"
                       class="joint-number-input">
              </div>
              
              <!-- Speed control (if available) -->
              <div class="joint-speed-control" v-if="joint.speed !== undefined">
                <label class="speed-label">Speed:</label>
                <input type="range" min="0" max="100" v-model.number="joint.speed" @change="updateJointSpeed(joint.id, joint.speed)"
                       class="joint-speed-slider">
                <span class="speed-value">{{ joint.speed }}%</span>
              </div>
            </div>
            
            <!-- Slider control -->
            <div class="joint-slider-container">
              <input type="range" :min="joint.min" :max="joint.max" :step="joint.step" v-model.number="joint.value"
                     @input="onJointSliderInput(joint.id, $event)"
                     @change="updateJoint(joint.id, joint.value)"
                     class="joint-slider">
              <div class="joint-range-labels">
                <span>{{ joint.min }}°</span>
                <span>{{ joint.max }}°</span>
              </div>
            </div>
            
            <!-- Joint status (if available) -->
            <div class="joint-status" v-if="joint.status">
              <span class="joint-status-label">Status:</span>
              <span :class="`joint-status-value ${joint.status}`">{{ joint.status }}</span>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Joint control presets -->
      <div class="joint-presets" v-if="jointPresets.length > 0">
        <h4>Joint Presets</h4>
        <div class="preset-buttons">
          <button v-for="preset in jointPresets" :key="preset.name" 
                  @click="applyJointPreset(preset)" 
                  class="btn btn-sm btn-outline-primary">
            {{ preset.name }}
          </button>
        </div>
      </div>
    </div>

    <!-- Sensor Data -->
    <div class="sensor-data-section">
      <h3>Sensor Data</h3>
      <div class="sensor-grid">
        <div class="sensor-card" v-for="sensor in sensorData" :key="sensor.id">
          <div class="sensor-header">
            <span class="sensor-name">{{ sensor.name }}</span>
            <span class="sensor-status" :class="sensor.status"></span>
          </div>
          <div class="sensor-value">{{ sensor.value }} {{ sensor.unit }}</div>
          <div class="sensor-info">
            <small>Type: {{ sensor.type }}</small>
            <small>ID: {{ sensor.id }}</small>
          </div>
        </div>
        
        <!-- Sensor Calibration Controls -->
        <div class="sensor-calibration-section">
          <h5>Sensor Calibration</h5>
          <div style="margin-bottom: 15px; padding: 15px; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px;">
            <div style="margin-bottom: 10px;">
              <strong>Calibration Status:</strong>
              <span :style="{ 
                color: sensorCalibrationState.calibrationStatus === 'calibrated' ? '#28a745' : 
                       sensorCalibrationState.calibrationStatus === 'calibrating' ? '#ffc107' : 
                       sensorCalibrationState.calibrationStatus === 'error' ? '#dc3545' : '#6c757d' 
              }">
                {{ sensorCalibrationState.calibrationStatus.toUpperCase() }}
              </span>
              <span v-if="sensorCalibrationState.calibrationTimestamp" style="margin-left: 10px; font-size: 0.85em; color: #666;">
                (Last: {{ new Date(sensorCalibrationState.calibrationTimestamp).toLocaleTimeString() }})
              </span>
            </div>
            
            <div v-if="sensorCalibrationState.calibrating" style="margin-bottom: 10px;">
              <div style="display: flex; align-items: center; gap: 10px;">
                <div style="flex-grow: 1; height: 10px; background-color: #e9ecef; border-radius: 5px; overflow: hidden;">
                  <div :style="{ 
                    width: `${sensorCalibrationState.calibrationProgress}%`, 
                    height: '100%', 
                    backgroundColor: '#28a745',
                    transition: 'width 0.3s ease'
                  }"></div>
                </div>
                <span style="font-size: 0.9em;">{{ sensorCalibrationState.calibrationProgress }}%</span>
              </div>
            </div>
            
            <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
              <button @click="startSensorCalibration" :disabled="sensorCalibrationState.calibrating" class="btn btn-sm btn-primary">
                {{ sensorCalibrationState.calibrating ? 'Calibrating...' : 'Start Calibration' }}
              </button>
              <button @click="stopSensorCalibration" :disabled="!sensorCalibrationState.calibrating" class="btn btn-sm btn-warning">Stop</button>
              <button @click="resetSensorCalibration" class="btn btn-sm btn-secondary">Reset</button>
              <span style="margin-left: auto; font-size: 0.85em; color: #666;">
                Sensors: {{ sensorData.length }}
              </span>
            </div>
            
            <div v-if="sensorCalibrationState.calibrationError" style="margin-top: 10px; padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px; color: #721c24; font-size: 0.9em;">
              <strong>Calibration Error:</strong> {{ sensorCalibrationState.calibrationError }}
            </div>
            
            <div v-if="Object.keys(sensorCalibrationState.calibrationResults).length > 0" style="margin-top: 15px;">
              <strong>Calibration Results:</strong>
              <div style="margin-top: 5px; font-size: 0.85em;">
                <div v-for="(result, sensorType) in sensorCalibrationState.calibrationResults" :key="sensorType" style="padding: 5px 0; border-bottom: 1px solid #eee;">
                  <strong>{{ sensorType }}:</strong> Offset: {{ result.offset?.toFixed(3) || 'N/A' }}, 
                  Scale: {{ result.scale?.toFixed(3) || 'N/A' }}, 
                  Quality: {{ result.quality?.toFixed(3) || 'N/A' }}
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Movement Trajectory Visualization -->
        <div class="movement-trajectory">
          <h5>Movement Trajectory</h5>
          <div style="margin-bottom: 10px; display: flex; gap: 10px; align-items: center;">
            <span>Trajectory Points: {{ movementHistory.length }}</span>
            <button @click="clearTrajectory" class="btn btn-sm btn-secondary">Clear Trajectory</button>
          </div>
          <div class="trajectory-canvas-container" style="position: relative; width: 100%; height: 300px; background-color: #f5f5f5; border: 1px solid #ddd; border-radius: 4px; overflow: hidden;">
            <canvas ref="trajectoryCanvas" width="600" height="300" style="display: block;"></canvas>
            <div v-if="movementHistory.length === 0" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #999; text-align: center;">
              No trajectory data yet<br>
              <small>Move the robot to see trajectory visualization</small>
            </div>
          </div>
          <div style="margin-top: 10px; font-size: 0.85em; color: #666;">
            <div style="display: flex; gap: 15px; flex-wrap: wrap;">
              <div style="display: flex; align-items: center; gap: 5px;">
                <div style="width: 12px; height: 12px; background-color: #0066cc; border-radius: 50%;"></div>
                <span>Current Position</span>
              </div>
              <div style="display: flex; align-items: center; gap: 5px;">
                <div style="width: 12px; height: 12px; background-color: #00cc66; border-radius: 50%;"></div>
                <span>History Points</span>
              </div>
              <div style="display: flex; align-items: center; gap: 5px;">
                <div style="width: 12px; height: 2px; background-color: #ff6600;"></div>
                <span>Movement Path</span>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Sensor Data Visualization -->
        <div class="sensor-visualization" style="margin-top: 15px;">
          <h5 style="margin-bottom: 8px; font-size: 1em; color: #333;">Sensor Data Visualization</h5>
          <div style="margin-bottom: 10px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
            <span>Data Source:</span>
            <button @click="toggleSensorChartDataSource" class="btn btn-sm" :class="sensorChartDataSource === 'sensors' ? 'btn-primary' : 'btn-secondary'">
              {{ sensorChartDataSource === 'sensors' ? 'Real Sensors' : 'Simulation' }}
            </button>
            <span style="margin-left: auto; font-size: 0.85em; color: #666;">
              Points: {{ sensorChartData.length }}
            </span>
            <button @click="clearSensorChartData" class="btn btn-sm btn-outline-danger">Clear Data</button>
          </div>
          <div class="sensor-chart-container" style="position: relative; width: 100%; height: 250px; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 4px; padding: 10px;">
            <canvas ref="sensorChartCanvas" width="600" height="250"></canvas>
            <div v-if="sensorChartData.length === 0" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #999; text-align: center;">
              No sensor data yet<br>
              <small>Enable collision detection to start collecting sensor data</small>
            </div>
          </div>
          <div style="margin-top: 10px; font-size: 0.85em; color: #666; display: flex; justify-content: space-between;">
            <div>
              <span>Chart Type: </span>
              <select v-model="sensorChartType" @change="updateSensorChart" style="padding: 2px 5px; border: 1px solid #ccc; border-radius: 3px; background-color: white;">
                <option value="line">Line Chart</option>
                <option value="bar">Bar Chart</option>
                <option value="radar">Radar Chart</option>
              </select>
            </div>
            <div>
              <span>Time Window: </span>
              <select v-model="sensorChartTimeWindow" @change="updateSensorChart" style="padding: 2px 5px; border: 1px solid #ccc; border-radius: 3px; background-color: white;">
                <option value="30">30 seconds</option>
                <option value="60">1 minute</option>
                <option value="300">5 minutes</option>
                <option value="600">10 minutes</option>
                <option value="0">All Data</option>
              </select>
            </div>
          </div>
        </div>
        
        <!-- Trajectory Statistics -->
        <div class="trajectory-statistics" style="margin-top: 15px; padding: 10px; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 4px;">
          <h5 style="margin-bottom: 8px; font-size: 1em; color: #333;">Trajectory Statistics</h5>
          <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; font-size: 0.85em;">
            <div>
              <span style="color: #666;">Total Distance:</span>
              <span style="font-weight: bold; margin-left: 5px;">{{ trajectoryStats.totalDistance }} units</span>
            </div>
            <div>
              <span style="color: #666;">Average Speed:</span>
              <span style="font-weight: bold; margin-left: 5px;">{{ trajectoryStats.averageSpeed }} units/s</span>
            </div>
            <div>
              <span style="color: #666;">Movement Time:</span>
              <span style="font-weight: bold; margin-left: 5px;">{{ trajectoryStats.movementTime }} s</span>
            </div>
            <div>
              <span style="color: #666;">Points Recorded:</span>
              <span style="font-weight: bold; margin-left: 5px;">{{ trajectoryStats.pointCount }}</span>
            </div>
            <div v-if="trajectoryStats.startTime" style="grid-column: span 2;">
              <span style="color: #666;">Time Range:</span>
              <span style="font-weight: bold; margin-left: 5px;">
                {{ new Date(trajectoryStats.startTime).toLocaleTimeString() }} - 
                {{ trajectoryStats.endTime ? new Date(trajectoryStats.endTime).toLocaleTimeString() : 'Active' }}
              </span>
            </div>
          </div>
        </div>
        
        <!-- Data Recording and Playback -->
        <div class="data-recording-section">
          <h5>Data Recording and Playback</h5>
          <div style="margin-bottom: 15px; padding: 15px; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px;">
            <div style="margin-bottom: 10px;">
              <strong>Recording Status:</strong>
              <span :style="{ 
                color: dataRecordingState.recording ? '#dc3545' : 
                       dataRecordingState.playback ? '#ffc107' : 
                       '#28a745' 
              }">
                {{ dataRecordingState.recording ? 'RECORDING' : dataRecordingState.playback ? 'PLAYBACK' : 'IDLE' }}
              </span>
              <span v-if="dataRecordingState.recordStartTime" style="margin-left: 10px; font-size: 0.85em; color: #666;">
                (Duration: {{ Math.round(dataRecordingState.recordDuration / 1000) }}s)
              </span>
            </div>
            
            <div style="margin-bottom: 10px;">
              <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 0.9em;"><strong>Records:</strong> {{ dataRecordingState.records.length }}</span>
                <span style="font-size: 0.9em;"><strong>Max Records:</strong> {{ dataRecordingState.maxRecords }}</span>
                <span style="font-size: 0.9em;"><strong>Interval:</strong> {{ dataRecordingState.recordInterval }}ms</span>
              </div>
            </div>
            
            <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
              <button @click="startRecording" :disabled="dataRecordingState.recording || dataRecordingState.playback" class="btn btn-sm btn-danger">
                {{ dataRecordingState.recording ? 'Recording...' : 'Start Recording' }}
              </button>
              <button @click="stopRecording" :disabled="!dataRecordingState.recording" class="btn btn-sm btn-warning">Stop Recording</button>
              <button @click="startPlayback" :disabled="dataRecordingState.recording || dataRecordingState.playback || dataRecordingState.records.length === 0" class="btn btn-sm btn-primary">
                {{ dataRecordingState.playback ? 'Playing...' : 'Start Playback' }}
              </button>
              <button @click="stopPlayback" :disabled="!dataRecordingState.playback" class="btn btn-sm btn-secondary">Stop Playback</button>
              <button @click="clearRecordsWithConfirmation" class="btn btn-sm btn-outline-danger">Clear Records</button>
              
              <div style="margin-left: auto; display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 0.9em;">Playback Speed:</span>
                <select v-model="dataRecordingState.playbackSpeed" :disabled="dataRecordingState.playback" style="padding: 2px 5px; border: 1px solid #ccc; border-radius: 3px; background-color: white;">
                  <option value="0.25">0.25x</option>
                  <option value="0.5">0.5x</option>
                  <option value="1.0">1.0x</option>
                  <option value="1.5">1.5x</option>
                  <option value="2.0">2.0x</option>
                </select>
              </div>
            </div>
            
            <div v-if="dataRecordingState.records.length > 0" style="margin-top: 15px;">
              <strong>Record Preview (Last 5):</strong>
              <div style="margin-top: 5px; max-height: 150px; overflow-y: auto; font-size: 0.85em;">
                <div v-for="(record, index) in dataRecordingState.records.slice(-5).reverse()" :key="record.timestamp" 
                     style="padding: 4px 6px; border-bottom: 1px solid #eee; font-family: monospace;">
                  <div style="display: flex; justify-content: space-between;">
                    <span>{{ new Date(record.timestamp).toLocaleTimeString() }}</span>
                    <span style="color: #666;">{{ record.type || 'sensor' }}</span>
                  </div>
                  <div style="font-size: 0.8em; color: #666;">
                    <span v-if="record.sensorData">Sensors: {{ record.sensorData.length }}</span>
                    <span v-if="record.jointData" style="margin-left: 10px;">Joints: {{ Object.keys(record.jointData).length }}</span>
                    <span v-if="record.position" style="margin-left: 10px;">
                      Pos: X={{ record.position.x?.toFixed(2) || '0.00' }}, Y={{ record.position.y?.toFixed(2) || '0.00' }}
                    </span>
                  </div>
                </div>
              </div>
            </div>
            
            <div v-if="dataRecordingState.currentRecordIndex >= 0 && dataRecordingState.playback" style="margin-top: 10px; padding: 8px; background-color: #e8f4fd; border-radius: 4px; border: 1px solid #b3d7ff;">
              <div style="display: flex; align-items: center; gap: 10px;">
                <div style="flex-grow: 1; height: 6px; background-color: #e9ecef; border-radius: 3px; overflow: hidden;">
                  <div :style="{ 
                    width: `${((dataRecordingState.currentRecordIndex + 1) / dataRecordingState.records.length) * 100}%`, 
                    height: '100%', 
                    backgroundColor: '#007bff',
                    transition: 'width 0.3s ease'
                  }"></div>
                </div>
                <span style="font-size: 0.9em;">
                  {{ dataRecordingState.currentRecordIndex + 1 }} / {{ dataRecordingState.records.length }}
                  ({{ Math.round(((dataRecordingState.currentRecordIndex + 1) / dataRecordingState.records.length) * 100) }}%)
                </span>
              </div>
            </div>
          </div>
          
          <!-- Data Recording Usage Guide -->
          <div style="margin-top: 15px; padding: 10px; background-color: #e8f4fd; border-radius: 4px; border: 1px solid #b3d7ff; font-size: 0.85em;">
            <strong>Usage Guide:</strong>
            <ul style="margin: 5px 0 0 0; padding-left: 20px;">
              <li><strong>Start Recording:</strong> Click to begin recording sensor data, joint positions, and robot movement</li>
              <li><strong>Stop Recording:</strong> Click to stop the recording process</li>
              <li><strong>Start Playback:</strong> Click to replay recorded data (requires at least one record)</li>
              <li><strong>Clear Records:</strong> Click to delete all recorded data (requires confirmation)</li>
              <li><strong>Playback Speed:</strong> Adjust the playback speed from 0.25x to 2.0x</li>
              <li><strong>Max Records:</strong> System automatically keeps the latest 1000 records</li>
              <li><strong>Record Preview:</strong> Shows the last 5 recorded data points with timestamps and details</li>
            </ul>
          </div>
        </div>
        
        <!-- Collision Detection Status -->
        <div class="collision-detection-status">
          <h5>Collision Detection Status</h5>
          <div style="margin-bottom: 10px; display: flex; gap: 10px; align-items: center;">
            <div class="status-indicator" :style="{ 
              backgroundColor: collisionDetectionState.warningLevel === 'critical' ? '#dc3545' : 
                              collisionDetectionState.warningLevel === 'warning' ? '#ffc107' : 
                              collisionDetectionState.enabled ? '#28a745' : '#6c757d' 
            }" style="width: 12px; height: 12px; border-radius: 50%;"></div>
            <span><strong>Status:</strong> {{ collisionDetectionState.status }}</span>
            <span><strong>Enabled:</strong> {{ collisionDetectionState.enabled ? 'Yes' : 'No' }}</span>
            <button @click="toggleCollisionDetection" class="btn btn-sm" :class="collisionDetectionState.enabled ? 'btn-warning' : 'btn-success'">
              {{ collisionDetectionState.enabled ? 'Disable' : 'Enable' }}
            </button>
            <button @click="resetCollisionDetection" class="btn btn-sm btn-secondary">Reset</button>
          </div>
          
          <div v-if="collisionDetectionState.enabled" style="background-color: #f8f9fa; padding: 15px; border-radius: 4px; border: 1px solid #dee2e6;">
            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px;">
              <div class="status-item">
                <strong>Warning Level:</strong> 
                <span :style="{ 
                  color: collisionDetectionState.warningLevel === 'critical' ? '#dc3545' : 
                         collisionDetectionState.warningLevel === 'warning' ? '#ffc107' : 
                         '#28a745' 
                }">
                  {{ collisionDetectionState.warningLevel.toUpperCase() }}
                </span>
              </div>
              <div class="status-item">
                <strong>Obstacles Detected:</strong> {{ collisionDetectionState.obstacles.length }}
              </div>
              <div class="status-item" v-if="collisionDetectionState.nearestObstacle">
                <strong>Nearest Obstacle:</strong> {{ collisionDetectionState.nearestObstacle.distance.toFixed(2) }} m
              </div>
              <div class="status-item" v-if="collisionDetectionState.lastDetectionTime">
                <strong>Last Detection:</strong> {{ new Date(collisionDetectionState.lastDetectionTime).toLocaleTimeString() }}
              </div>
              <div class="status-item">
                <strong>Collision Threshold:</strong> {{ spatialConstraints.collisionThreshold.toFixed(2) }} m
              </div>
            </div>
            
            <!-- Obstacle List -->
            <div v-if="collisionDetectionState.obstacles.length > 0" style="margin-top: 15px;">
              <h6>Detected Obstacles</h6>
              <div style="max-height: 150px; overflow-y: auto; font-size: 0.85em;">
                <div v-for="obstacle in collisionDetectionState.obstacles" :key="obstacle.id" 
                     style="padding: 5px; border-bottom: 1px solid #eee; display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px;">
                  <span><strong>ID:</strong> {{ obstacle.id }}</span>
                  <span><strong>Distance:</strong> {{ obstacle.distance.toFixed(2) }} m</span>
                  <span><strong>Direction:</strong> {{ (obstacle.angle * 180 / Math.PI).toFixed(0) }}°</span>
                  <span><strong>Confidence:</strong> {{ (obstacle.confidence * 100).toFixed(0) }}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Movement History -->
        <div class="movement-history">
          <h5>Movement History</h5>
          
          <!-- Trajectory Visualization -->
          <div class="trajectory-visualization" style="margin-bottom: 20px;">
            <h6>Trajectory Visualization</h6>
            <div style="margin-bottom: 10px; display: flex; gap: 10px; align-items: center;">
              <span>Trajectory Points: {{ movementHistory.length }}</span>
              <button @click="clearTrajectory" class="btn btn-sm btn-secondary">Clear Trajectory</button>
            </div>
            <div class="trajectory-canvas-container" style="position: relative; width: 100%; height: 200px; background-color: #f5f5f5; border: 1px solid #ddd; border-radius: 4px; overflow: hidden;">
              <canvas ref="trajectoryCanvas" width="400" height="200" style="display: block;"></canvas>
              <div v-if="movementHistory.length === 0" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #999; text-align: center;">
                No trajectory data yet<br>
                <small>Move the robot to see trajectory visualization</small>
              </div>
            </div>
            <div style="margin-top: 10px; font-size: 0.8em; color: #666;">
              <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 5px;">
                  <div style="width: 10px; height: 10px; background-color: #0066cc; border-radius: 50%;"></div>
                  <span>Current Position</span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                  <div style="width: 10px; height: 10px; background-color: #00cc66; border-radius: 50%;"></div>
                  <span>History Points</span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                  <div style="width: 10px; height: 2px; background-color: #ff6600;"></div>
                  <span>Movement Path</span>
                </div>
              </div>
            </div>
          </div>
          
          <div class="movement-history-controls" style="margin-bottom: 10px; display: flex; gap: 10px; align-items: center;">
            <span>Recorded Movements: {{ movementHistory.length }}</span>
            <button @click="clearMovementHistory" class="btn btn-sm btn-secondary">Clear History</button>
          </div>
          <div class="history-list">
            <div v-if="movementHistory.length === 0" style="padding: 10px; text-align: center; color: #999;">
              No movement history yet
            </div>
            <div v-else>
              <div v-for="(record, index) in movementHistory.slice(0, 10)" :key="index" class="history-item">
                <div style="display: flex; justify-content: space-between;">
                  <span>{{ new Date(record.timestamp).toLocaleTimeString() }}</span>
                  <span v-if="record.keyboardState.forward">Forward</span>
                  <span v-if="record.keyboardState.backward">Backward</span>
                  <span v-if="record.keyboardState.left">Left</span>
                  <span v-if="record.keyboardState.right">Right</span>
                </div>
                <div style="font-size: 0.8em; color: #666;">
                  Position: X={{ record.position?.x?.toFixed(2) || '0.00' }}, 
                  Y={{ record.position?.y?.toFixed(2) || '0.00' }}, 
                  Z={{ record.position?.z?.toFixed(2) || '0.00' }}
                </div>
              </div>
              <div v-if="movementHistory.length > 10" style="padding: 5px; text-align: center; color: #999; font-size: 0.8em;">
                ... and {{ movementHistory.length - 10 }} more
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Model Collaboration Settings -->
    <div class="collaboration-control-section">
      <h3>🤝 Model Collaboration Settings</h3>
      
      <div class="collaboration-status" style="margin-bottom: 20px;">
        <p><strong>Status:</strong> {{ collaborationState.status }}</p>
        <p v-if="collaborationState.activePattern">
          <strong>Active Pattern:</strong> {{ collaborationState.activePattern }}
        </p>
        <p v-if="collaborationState.lastResult">
          <strong>Last Result:</strong> {{ collaborationState.lastResult.status }}
        </p>
      </div>

      <div class="collaboration-controls" style="display: flex; flex-direction: column; gap: 15px;">
        <div class="pattern-selection">
          <label><strong>Select Collaboration Pattern:</strong></label>
          <select v-model="selectedPattern" class="pattern-select" style="padding: 8px; width: 100%; max-width: 400px;">
            <option value="">-- Please select collaboration pattern --</option>
            <option v-for="pattern in collaborationPatterns" :key="pattern.name" :value="pattern.name">
              {{ pattern.name }} - {{ pattern.description }}
            </option>
          </select>
        </div>

        <div class="input-data" v-if="selectedPattern">
          <label><strong>Input Data (JSON):</strong></label>
          <textarea v-model="collaborationInput" class="input-textarea" 
                    placeholder="Enter JSON input for collaboration pattern (e.g., task configuration)" 
                    style="width: 100%; height: 80px; padding: 8px; font-family: monospace;">
          </textarea>
        </div>

        <!-- Collaboration Pattern Configuration -->
        <div class="collaboration-config" v-if="selectedPattern">
          <h4>Collaboration Pattern Configuration</h4>
          <div class="config-section">
            <label><strong>Input Type:</strong></label>
            <div class="input-type-selection">
              <label>
                <input type="radio" v-model="collaborationConfig.inputType" value="text"> Text
              </label>
              <label>
                <input type="radio" v-model="collaborationConfig.inputType" value="image"> Image
              </label>
              <label>
                <input type="radio" v-model="collaborationConfig.inputType" value="audio"> Audio
              </label>
              <label>
                <input type="radio" v-model="collaborationConfig.inputType" value="sensor"> Sensor Data
              </label>
              <label>
                <input type="radio" v-model="collaborationConfig.inputType" value="multimodal"> Multimodal
              </label>
            </div>
          </div>
          
          <div class="config-section">
            <label><strong>Output Type:</strong></label>
            <div class="output-type-selection">
              <label>
                <input type="checkbox" v-model="collaborationConfig.outputTypes" value="text"> Text
              </label>
              <label>
                <input type="checkbox" v-model="collaborationConfig.outputTypes" value="action"> Action Commands
              </label>
              <label>
                <input type="checkbox" v-model="collaborationConfig.outputTypes" value="decision"> Decision
              </label>
              <label>
                <input type="checkbox" v-model="collaborationConfig.outputTypes" value="visualization"> Visualization
              </label>
            </div>
          </div>
          
          <div class="config-section">
            <label><strong>Collaboration Priority:</strong></label>
            <select v-model="collaborationConfig.priority" class="config-select">
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="critical">Critical</option>
            </select>
          </div>
          
          <div class="config-section">
            <label><strong>Timeout (seconds):</strong></label>
            <input type="number" v-model.number="collaborationConfig.timeout" min="10" max="300" step="10" class="config-input">
          </div>
          
          <div class="config-section">
            <label><strong>Enable Real-time Monitoring:</strong></label>
            <input type="checkbox" v-model="collaborationConfig.realtimeMonitoring">
          </div>
        </div>

        <!-- Model Workflow Visualization -->
        <div class="workflow-visualization" v-if="selectedPattern && selectedPatternDetails">
          <h4>Model Workflow</h4>
          <div class="workflow-diagram">
            <div v-for="(step, index) in selectedPatternDetails.workflow" :key="index" class="workflow-step">
              <div class="step-header">
                <span class="step-number">{{ index + 1 }}</span>
                <span class="step-model">{{ step.model }}</span>
                <span class="step-task">{{ step.task }}</span>
              </div>
              <div class="step-dependencies" v-if="step.depends_on">
                <small>Depends on: {{ Array.isArray(step.depends_on) ? step.depends_on.join(', ') : step.depends_on }}</small>
              </div>
            </div>
          </div>
        </div>

        <div class="collaboration-buttons">
          <button @click="fetchCollaborationPatterns" class="btn btn-secondary">
            Refresh Collaboration Patterns
          </button>
          <button @click="startCollaboration" :disabled="!selectedPattern || collaborationState.loading" 
                  class="btn btn-primary">
            {{ collaborationState.loading ? 'Collaboration in progress...' : 'Start Collaboration' }}
          </button>
          <button @click="stopCollaboration" :disabled="!collaborationState.activePattern" 
                  class="btn btn-danger">
            Stop Collaboration
          </button>
        </div>

        <div class="collaboration-results" v-if="collaborationState.lastResult" style="margin-top: 20px;">
          <h4>Collaboration Results:</h4>
          <pre style="background: #f5f5f5; padding: 15px; border-radius: 4px; max-height: 300px; overflow-y: auto;">
{{ JSON.stringify(collaborationState.lastResult, null, 2) }}
          </pre>
        </div>

        <div class="available-patterns" style="margin-top: 20px;">
          <h4>Available Collaboration Patterns ({{ collaborationPatterns?.length || 0 }}):</h4>
          <ul style="list-style-type: none; padding: 0;">
            <li v-for="pattern in collaborationPatterns" :key="pattern.name" 
                style="padding: 8px; border-bottom: 1px solid #eee;">
              <strong>{{ pattern.name }}</strong>: {{ pattern.description }}
              <br>
              <small>Models: {{ pattern.models.join(', ') }} | Mode: {{ pattern.mode }}</small>
            </li>
          </ul>
        </div>
      </div>
    </div>

    <!-- Robot Training Module -->
    <div class="robot-training-section" style="border: 2px solid #20c997; padding: 20px; margin: 20px 0 30px 0; background-color: #f8f9fa; border-radius: 8px;">
      <h3>🤖 Robot Training Module</h3>
      
      <div class="training-status" style="margin-bottom: 20px;">
        <p><strong>Training Status:</strong> {{ trainingState.status }}</p>
        <p v-if="trainingState.activeTraining">
          <strong>Active Training:</strong> {{ trainingState.activeTraining.trainingId }} - {{ trainingState.activeTraining.mode }}
        </p>
        <p v-if="trainingState.progress > 0">
          <strong>Progress:</strong> {{ trainingState.progress }}%
        </p>
      </div>

      <!-- Hardware Status Display -->
      <div class="hardware-status" style="margin-bottom: 20px; padding: 15px; background-color: #e9ecef; border-radius: 5px; border-left: 4px solid #007bff;">
        <h5 style="margin-top: 0; color: #007bff;">🔧 Hardware Status</h5>
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
          <div class="hardware-status-item">
            <strong>Joints:</strong> {{ hardwareStatus?.joints || 0 }} connected
          </div>
          <div class="hardware-status-item">
            <strong>Sensors:</strong> {{ hardwareStatus?.sensors || 0 }} connected
          </div>
          <div class="hardware-status-item">
            <strong>Cameras:</strong> {{ hardwareStatus?.cameras || 0 }} connected
          </div>
          <div class="hardware-status-item">
            <strong>Battery:</strong> {{ hardwareStatus?.battery || 0 }}%
          </div>
          <div class="hardware-status-item">
            <strong>System:</strong> {{ hardwareStatus?.systemTemperature || 0 }}°C
          </div>
        </div>
        <div style="margin-top: 10px; display: flex; gap: 10px;">
          <button @click="refreshHardwareStatus" class="btn btn-sm btn-info" style="padding: 5px 10px;">
            🔄 Refresh
          </button>
          <button @click="testHardwareConnection" class="btn btn-sm btn-warning" style="padding: 5px 10px;">
            🔌 Test Connection
          </button>
          <button @click="initializeHardware" class="btn btn-sm btn-success" style="padding: 5px 10px;">
            ⚡ Initialize
          </button>
        </div>
      </div>

      <div class="training-controls" style="display: flex; flex-direction: column; gap: 15px;">
        <!-- Training Mode Selection -->
        <div class="mode-selection">
          <label><strong>Training Mode:</strong></label>
          <select v-model="selectedTrainingMode" class="mode-select" style="padding: 8px; width: 100%; max-width: 400px;">
            <option value="">-- Select training mode --</option>
            <option v-for="mode in trainingModes" :key="mode.id" :value="mode.id">
              {{ mode.name }} - {{ mode.description }}
            </option>
          </select>
        </div>

        <!-- Model Selection -->
        <div class="model-selection" v-if="selectedTrainingMode">
          <label><strong>Select Models for Training:</strong></label>
          <div class="model-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin-top: 10px;">
            <div v-for="model in availableTrainingModels" :key="model.id" 
                 class="model-option" 
                 :class="{ selected: isModelSelected(model.id) }"
                 @click.stop.prevent="toggleTrainingModel(model.id, $event)"
                 style="padding: 8px; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; text-align: center; position: relative; z-index: 1;"
                 :title="model.description">
              {{ model.name }}
              <br>
              <small>Port: {{ model.port }}</small>
            </div>
          </div>
          <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
            Selected: {{ selectedTrainingModels?.length || 0 }} models
          </div>
        </div>

        <!-- Dataset Selection -->
        <div class="dataset-selection" v-if="selectedTrainingMode && (selectedTrainingModels?.length || 0) > 0">
          <h4>Training Dataset</h4>
          <div class="dataset-controls" style="display: flex; flex-wrap: wrap; gap: 10px; align-items: center;">
            <div style="flex: 1; min-width: 200px;">
              <label><strong>Select Dataset:</strong></label>
              <select v-model="selectedDataset" class="dataset-select" style="padding: 8px; width: 100%; max-width: 400px;">
                <option value="">-- Select dataset --</option>
                <option v-for="dataset in datasets" :key="dataset.id" :value="dataset.id">
                  {{ dataset.name || dataset.id }}
                </option>
              </select>
            </div>
            <div style="display: flex; gap: 5px;">
              <button @click="loadDatasets" class="btn btn-secondary" style="padding: 8px 12px;">
                ↻ Refresh
              </button>
              <button @click="$refs.datasetInput?.click()" class="btn btn-secondary" style="padding: 8px 12px;">
                📁 Upload
              </button>
            </div>
          </div>
          <div v-if="datasetsError" style="color: #dc3545; font-size: 0.9em; margin-top: 5px;">
            Error: {{ datasetsError }}
          </div>
          <div v-if="datasetsLoading" style="color: #6c757d; font-size: 0.9em; margin-top: 5px;">
            Loading datasets...
          </div>
          <div v-else-if="datasets?.length === 0" style="color: #6c757d; font-size: 0.9em; margin-top: 5px;">
            No datasets available. Please upload a dataset.
          </div>
          <div v-else-if="selectedDataset" style="color: #28a745; font-size: 0.9em; margin-top: 5px;">
            Selected dataset: {{ datasets.find(d => d.id === selectedDataset)?.name || selectedDataset }}
          </div>
          <input 
            type="file" 
            ref="datasetInput" 
            style="display: none" 
            @change="handleDatasetUpload"
          >
        </div>

        <!-- Hardware Integration -->
        <div class="hardware-integration" v-if="selectedTrainingMode && (selectedTrainingModels?.length || 0) > 0">
          <h4>Hardware Integration</h4>
          <div class="hardware-selection" style="display: flex; flex-direction: column; gap: 10px;">
            <!-- Joint Selection -->
            <div class="joint-selection">
              <label><strong>Joints for Training:</strong></label>
              <div class="joint-checkboxes" style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 5px;">
                <label v-for="joint in jointList" :key="joint.id" style="display: flex; align-items: center; gap: 5px;">
                  <input type="checkbox" v-model="selectedJoints" :value="joint.id">
                  {{ joint.name }}
                </label>
              </div>
              <div style="font-size: 0.9em; color: #666; margin-top: 5px;">
                Selected: {{ selectedJoints?.length || 0 }} joints
              </div>
            </div>

            <!-- Sensor Selection -->
            <div class="sensor-selection">
              <label><strong>Sensors for Monitoring:</strong></label>
              <div class="sensor-checkboxes" style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 5px;">
                <label v-for="sensor in sensorInstances" :key="sensor.id" style="display: flex; align-items: center; gap: 5px;">
                  <input type="checkbox" v-model="selectedSensors" :value="sensor.id">
                  {{ sensor.description }}
                </label>
              </div>
              <div style="font-size: 0.9em; color: #666; margin-top: 5px;">
                Selected: {{ selectedSensors?.length || 0 }} sensors
              </div>
            </div>

            <!-- Camera Selection -->
            <div class="camera-selection">
              <label><strong>Cameras for Vision Training:</strong></label>
              <div class="camera-checkboxes" style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 5px;">
                <label v-for="camera in cameraInstances" :key="camera.id" style="display: flex; align-items: center; gap: 5px;">
                  <input type="checkbox" v-model="selectedCameras" :value="camera.id">
                  {{ camera.description }}
                </label>
              </div>
              <div style="font-size: 0.9em; color: #666; margin-top: 5px;">
                Selected: {{ selectedCameras?.length || 0 }} cameras
              </div>
            </div>
          </div>
        </div>

        <!-- Training Parameters -->
        <div class="training-parameters" v-if="selectedTrainingMode && (selectedTrainingModels?.length || 0) > 0">
          <h4>Training Parameters</h4>
          <div class="parameter-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px;">
            <div class="parameter">
              <label>Iterations:</label>
              <input type="number" v-model.number="trainingParams.iterations" min="100" max="10000" step="100">
            </div>
            <div class="parameter">
              <label>Learning Rate:</label>
              <input type="number" v-model.number="trainingParams.learningRate" min="0.00001" max="0.1" step="0.00001">
            </div>
            <div class="parameter">
              <label>Batch Size:</label>
              <input type="number" v-model.number="trainingParams.batchSize" min="1" max="128" step="1">
            </div>
            <div class="parameter">
              <label>Validation Split:</label>
              <input type="number" v-model.number="trainingParams.validationSplit" min="0.01" max="0.5" step="0.01">
            </div>
            <div class="parameter">
              <label>Training Device:</label>
              <select v-model="trainingParams.device">
                <option value="cpu">CPU</option>
                <option value="gpu">GPU</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Safety Limits -->
        <div class="safety-limits" v-if="selectedTrainingMode">
          <h4>Safety Limits</h4>
          <div class="safety-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px;">
            <div class="safety-param">
              <label>Max Joint Velocity:</label>
              <input type="number" v-model.number="safetyLimits.maxJointVelocity" min="0.1" max="10.0" step="0.1"> deg/s
            </div>
            <div class="safety-param">
              <label>Max Joint Torque:</label>
              <input type="number" v-model.number="safetyLimits.maxJointTorque" min="0.1" max="20.0" step="0.1"> Nm
            </div>
            <div class="safety-param">
              <label>Max Temperature:</label>
              <input type="number" v-model.number="safetyLimits.maxTemperature" min="30" max="100" step="1"> °C
            </div>
            <div class="safety-param">
              <label>Emergency Stop Threshold:</label>
              <input type="number" v-model.number="safetyLimits.emergencyStopThreshold" min="0.5" max="2.0" step="0.1"> × limit
            </div>
          </div>
        </div>

        <!-- Training Control Buttons -->
        <div class="training-buttons" style="display: flex; gap: 10px; flex-wrap: wrap;">
          <button @click="startRobotTraining" 
                  :disabled="!selectedTrainingMode || (selectedTrainingModels?.length || 0) === 0 || trainingState.status === 'training'"
                  class="btn btn-primary">
            {{ trainingState.status === 'training' ? 'Training in progress...' : 'Start Training' }}
          </button>
          <button @click="pauseRobotTraining" 
                  :disabled="trainingState.status !== 'training'"
                  class="btn btn-warning">
            Pause Training
          </button>
          <button @click="stopRobotTraining" 
                  :disabled="trainingState.status !== 'training' && trainingState.status !== 'paused'"
                  class="btn btn-danger">
            Stop Training
          </button>
          <button @click="resetTrainingConfig" class="btn btn-secondary">
            Reset Configuration
          </button>
        </div>

        <!-- Training Progress -->
        <div class="training-progress-display" v-if="trainingState.status === 'training' || trainingState.status === 'paused'">
          <h4>Training Progress</h4>
          <div class="progress-bar" style="width: 100%; height: 20px; background-color: #e9ecef; border-radius: 4px; overflow: hidden;">
            <div class="progress-fill" :style="{ width: trainingState.progress + '%', backgroundColor: '#28a745', height: '100%', transition: 'width 0.3s' }"></div>
          </div>
          <div style="display: flex; justify-content: space-between; margin-top: 5px;">
            <span>{{ trainingState.progress }}%</span>
            <span class="training-status-indicator" :class="trainingState.status">
              {{ trainingState.status.toUpperCase() }}
            </span>
          </div>
        </div>

        <!-- Training Log -->
        <div class="training-log" v-if="trainingLog?.length > 0" style="margin-top: 20px;">
          <h4>Training Log</h4>
          <div style="background: #f5f5f5; padding: 15px; border-radius: 4px; max-height: 200px; overflow-y: auto;">
            <div v-for="(log, index) in trainingLog.slice(-10)" :key="index" style="padding: 5px 0; border-bottom: 1px solid #ddd;">
              <small>[{{ log.timestamp }}] {{ log.message }}</small>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Stereo Vision Spatial Recognition Preview -->
    <div class="stereo-vision-section" style="border: 2px solid #3498db; padding: 20px; margin: 20px 0 30px 0; background-color: #f0f8ff; border-radius: 8px;">
      <h3>👀 Stereo Vision Spatial Recognition</h3>
      
      <!-- Camera Selection and Status -->
      <div class="stereo-camera-selection" style="margin-bottom: 20px;">
        <div class="camera-status">
          <p><strong>Stereo Camera Status:</strong> {{ stereoCameraStatus.connected ? 'Connected' : 'Disconnected' }}</p>
          <p v-if="stereoCameraStatus.connected">
            <strong>Left Camera:</strong> {{ stereoCameraStatus.leftCamera || 'Not selected' }} |
            <strong>Right Camera:</strong> {{ stereoCameraStatus.rightCamera || 'Not selected' }} |
            <strong>Calibration:</strong> {{ stereoCameraStatus.calibrated ? 'Calibrated' : 'Not calibrated' }}
          </p>
        </div>
        
        <div class="camera-selection-controls" style="display: flex; gap: 10px; margin-top: 10px;">
          <button @click="detectStereoCameras" class="btn btn-primary">Detect Stereo Cameras</button>
          <button @click="calibrateStereoCameras" :disabled="!stereoCameraStatus.connected" class="btn btn-warning">Calibrate</button>
          <button @click="enableStereoVision" :disabled="!stereoCameraStatus.connected" class="btn btn-success">Enable Stereo Vision</button>
          <button @click="disableStereoVision" :disabled="!stereoCameraStatus.enabled" class="btn btn-danger">Disable</button>
        </div>
      </div>
      
      <!-- Camera Preview and Depth Map -->
      <div class="stereo-preview-container" style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 20px;">
        <!-- Left Camera Preview -->
        <div class="camera-preview">
          <h5>Left Camera</h5>
          <div class="preview-placeholder" style="width: 100%; height: 200px; background-color: #2c3e50; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white;">
            <div v-if="stereoCameraStatus.leftCameraStream">
              <video ref="leftCameraVideo" autoplay playsinline style="width: 100%; height: 100%; object-fit: cover;"></video>
            </div>
            <div v-else>
              No Stream
            </div>
          </div>
        </div>
        
        <!-- Depth Map Preview -->
        <div class="depth-map-preview">
          <h5>Depth Map</h5>
          <div class="preview-placeholder" style="width: 100%; height: 200px; background-color: #34495e; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white;">
            <div v-if="depthMapData">
              <canvas ref="depthMapCanvas" style="width: 100%; height: 100%;"></canvas>
            </div>
            <div v-else>
              Depth Map Not Generated
            </div>
          </div>
        </div>
        
        <!-- Right Camera Preview -->
        <div class="camera-preview">
          <h5>Right Camera</h5>
          <div class="preview-placeholder" style="width: 100%; height: 200px; background-color: #2c3e50; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white;">
            <div v-if="stereoCameraStatus.rightCameraStream">
              <video ref="rightCameraVideo" autoplay playsinline style="width: 100%; height: 100%; object-fit: cover;"></video>
            </div>
            <div v-else>
              No Stream
            </div>
          </div>
        </div>
      </div>
      
      <!-- Spatial Recognition Controls -->
      <div class="spatial-recognition-controls">
        <h4>Spatial Recognition Controls</h4>
        <div class="control-buttons" style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 15px;">
          <button @click="generateDepthMap" :disabled="!stereoCameraStatus.connected" class="btn btn-primary">Generate Depth Map</button>
          <button @click="startSpatialMapping" :disabled="!stereoCameraStatus.connected" class="btn btn-info">Start Spatial Mapping</button>
          <button @click="exportSpatialData" :disabled="!depthMapData" class="btn btn-secondary">Export Spatial Data</button>
          <select v-model="selectedStereoMode" class="mode-select" style="padding: 8px; border-radius: 4px;">
            <option value="depth">Depth Perception</option>
            <option value="pointcloud">Point Cloud Generation</option>
            <option value="object_detection">Object Detection</option>
          </select>
        </div>
        
        <!-- Spatial Recognition Results -->
        <div class="spatial-results" v-if="spatialResults" style="background-color: #f8f9fa; padding: 15px; border-radius: 4px; border: 1px solid #dee2e6;">
          <h5>Spatial Recognition Results</h5>
          <div class="results-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px;">
            <div class="result-item">
              <strong>Min Distance:</strong> {{ spatialResults.minDistance?.toFixed(3) || 'N/A' }} m
            </div>
            <div class="result-item">
              <strong>Max Distance:</strong> {{ spatialResults.maxDistance?.toFixed(3) || 'N/A' }} m
            </div>
            <div class="result-item">
              <strong>Average Depth:</strong> {{ spatialResults.averageDepth?.toFixed(3) || 'N/A' }} m
            </div>
            <div class="result-item">
              <strong>Object Count:</strong> {{ spatialResults.objectCount || 0 }}
            </div>
            <div class="result-item">
              <strong>Processing Time:</strong> {{ spatialResults.processingTime || 'N/A' }} ms
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Robot Free Space Movement Control -->
    <div class="robot-movement-section">
      <h3>🚀 Robot Free Space Movement Control</h3>
      
      <!-- Movement Status -->
      <div class="movement-status" style="margin-bottom: 20px;">
        <p><strong>Movement Status:</strong> {{ movementState.status || 'Idle' }}</p>
        <p v-if="movementState.active">
          <strong>Active Movement:</strong> {{ movementState.mode || 'Manual' }} |
          <strong>Speed:</strong> {{ movementState.speed || 0 }}% |
          <strong>Position:</strong> X: {{ movementState.position?.x?.toFixed(2) || '0.00' }}, Y: {{ movementState.position?.y?.toFixed(2) || '0.00' }}, Z: {{ movementState.position?.z?.toFixed(2) || '0.00' }}
        </p>
      </div>
      
      <!-- Manual Movement Controls -->
      <div class="manual-movement-controls">
        <h4>Manual Movement Controls</h4>
        
        <!-- Directional Pad -->
        <div class="directional-pad">
          <div style="grid-column: 2; grid-row: 1;">
            <button @click="moveRobot('forward')" class="btn btn-primary" style="width: 100%; padding: 15px;">↑ Forward</button>
          </div>
          <div style="grid-column: 1; grid-row: 2;">
            <button @click="moveRobot('left')" class="btn btn-primary" style="width: 100%; padding: 15px;">← Left</button>
          </div>
          <div style="grid-column: 2; grid-row: 2;">
            <button @click="stopRobot" class="btn btn-danger" style="width: 100%; padding: 15px;">⏹ Stop</button>
          </div>
          <div style="grid-column: 3; grid-row: 2;">
            <button @click="moveRobot('right')" class="btn btn-primary" style="width: 100%; padding: 15px;">Right →</button>
          </div>
          <div style="grid-column: 2; grid-row: 3;">
            <button @click="moveRobot('backward')" class="btn btn-primary" style="width: 100%; padding: 15px;">↓ Backward</button>
          </div>
        </div>
        
        <!-- Speed Control -->
        <div class="speed-control">
          <label><strong>Movement Speed:</strong></label>
          <div style="display: flex; align-items: center; gap: 15px;">
            <input type="range" v-model.number="movementSpeed" min="0" max="100" step="5" style="flex: 1; max-width: 300px;">
            <span>{{ movementSpeed }}%</span>
            <button @click="setMovementSpeed" class="btn btn-secondary">Set Speed</button>
          </div>
        </div>
        
        <!-- Keyboard Control Indicator -->
        <div class="keyboard-control-indicator">
          <h5>Keyboard Controls Active</h5>
          <div class="keyboard-shortcuts">
            <div class="keyboard-shortcut">
              <span class="keyboard-key">W</span>
              <span>/</span>
              <span class="keyboard-key">↑</span>
              <span>Move Forward</span>
            </div>
            <div class="keyboard-shortcut">
              <span class="keyboard-key">S</span>
              <span>/</span>
              <span class="keyboard-key">↓</span>
              <span>Move Backward</span>
            </div>
            <div class="keyboard-shortcut">
              <span class="keyboard-key">A</span>
              <span>/</span>
              <span class="keyboard-key">←</span>
              <span>Move Left</span>
            </div>
            <div class="keyboard-shortcut">
              <span class="keyboard-key">D</span>
              <span>/</span>
              <span class="keyboard-key">→</span>
              <span>Move Right</span>
            </div>
            <div class="keyboard-shortcut">
              <span class="keyboard-key">Space</span>
              <span>Stop Movement</span>
            </div>
            <div class="keyboard-shortcut">
              <span class="keyboard-key">Q</span>
              <span>Rotate Left</span>
            </div>
            <div class="keyboard-shortcut">
              <span class="keyboard-key">E</span>
              <span>Rotate Right</span>
            </div>
            <div class="keyboard-shortcut">
              <span class="keyboard-key">R</span>
              <span>Reset Rotation</span>
            </div>
          </div>
          <p v-if="keyboardState.active" style="margin-top: 8px; margin-bottom: 0; color: #0066cc; font-weight: bold;">
            <span class="status-value" :class="{'connected': keyboardState.active}">Keyboard Active</span>
            <span v-if="keyboardState.forward" style="margin-left: 8px;">Forward</span>
            <span v-if="keyboardState.backward" style="margin-left: 8px;">Backward</span>
            <span v-if="keyboardState.left" style="margin-left: 8px;">Left</span>
            <span v-if="keyboardState.right" style="margin-left: 8px;">Right</span>
          </p>
          <p v-else style="margin-top: 8px; margin-bottom: 0; color: #666;">
            Press any movement key to activate
          </p>
        </div>
        
        <!-- Rotation Controls -->
        <div class="rotation-controls">
          <label><strong>Rotation:</strong></label>
          <div style="display: flex; gap: 10px;">
            <button @click="rotateRobot('left')" class="btn btn-warning">Rotate Left</button>
            <button @click="rotateRobot('right')" class="btn btn-warning">Rotate Right</button>
            <button @click="rotateRobot('reset')" class="btn btn-secondary">Reset Heading</button>
          </div>
        </div>
      </div>
      
      <!-- Vision-Based Autonomous Navigation -->
      <div class="autonomous-navigation">
        <h4>Vision-Based Autonomous Navigation</h4>
        
        <!-- Navigation Mode Selection -->
        <div class="navigation-mode" style="margin-bottom: 15px;">
          <label><strong>Navigation Mode:</strong></label>
          <select v-model="selectedNavigationMode" class="mode-select" style="padding: 8px; margin-left: 10px;">
            <option value="manual">Manual Control</option>
            <option value="waypoint">Waypoint Navigation</option>
            <option value="obstacle_avoidance">Obstacle Avoidance</option>
            <option value="target_following">Target Following</option>
            <option value="exploration">Autonomous Exploration</option>
          </select>
        </div>
        
        <!-- Waypoint Management -->
        <div class="waypoint-management" v-if="selectedNavigationMode === 'waypoint'">
          <h5>Waypoint Management</h5>
          <div style="display: flex; gap: 10px; align-items: center;">
            <input type="number" v-model.number="waypointX" placeholder="X" style="width: 80px; padding: 5px;">
            <input type="number" v-model.number="waypointY" placeholder="Y" style="width: 80px; padding: 5px;">
            <input type="number" v-model.number="waypointZ" placeholder="Z" style="width: 80px; padding: 5px;">
            <button @click="addWaypoint" class="btn btn-primary">Add Waypoint</button>
            <button @click="clearWaypoints" class="btn btn-secondary">Clear All</button>
          </div>
          <div v-if="waypoints.length > 0" style="margin-top: 10px;">
            <p><strong>Waypoints ({{ waypoints.length }}):</strong></p>
            <ul style="list-style-type: none; padding: 0;">
              <li v-for="(wp, index) in waypoints" :key="index" style="padding: 5px; border-bottom: 1px solid #eee;">
                #{{ index + 1 }}: ({{ wp.x.toFixed(2) }}, {{ wp.y.toFixed(2) }}, {{ wp.z.toFixed(2) }})
                <button @click="removeWaypoint(index)" class="btn btn-sm btn-danger" style="margin-left: 10px;">Remove</button>
              </li>
            </ul>
            
            <!-- Path Planning Controls -->
            <div class="path-planning-controls" style="margin-top: 15px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6;">
              <h6 style="margin-top: 0; margin-bottom: 10px;">Path Planning</h6>
              
              <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 10px;">
                <button @click="optimizeWaypointPath(waypoints, true)" class="btn btn-info btn-sm">
                  Optimize Path (With Obstacles)
                </button>
                <button @click="optimizeWaypointPath(waypoints, false)" class="btn btn-secondary btn-sm">
                  Optimize Path (No Obstacles)
                </button>
                <button @click="visualizeOptimizedPath" class="btn btn-warning btn-sm" 
                        :disabled="pathPlanningState.optimizedPath.length === 0">
                  Visualize Path
                </button>
                <button @click="clearOptimizedPath" class="btn btn-danger btn-sm"
                        :disabled="pathPlanningState.optimizedPath.length === 0">
                  Clear Path
                </button>
              </div>
              
              <div v-if="pathPlanningState.optimizedPath.length > 0" style="font-size: 0.85em;">
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 5px;">
                  <div><strong>Optimized Waypoints:</strong> {{ pathPlanningState.optimizedPath.length }}</div>
                  <div><strong>Total Distance:</strong> {{ pathPlanningState.totalDistance.toFixed(2) }} m</div>
                  <div><strong>Planning Time:</strong> {{ pathPlanningState.planningTime.toFixed(1) }} ms</div>
                  <div><strong>Obstacles Considered:</strong> {{ pathPlanningState.obstaclesConsidered ? 'Yes' : 'No' }}</div>
                  <div><strong>Algorithm:</strong> {{ pathPlanningState.planningAlgorithm.toUpperCase() }}</div>
                </div>
                
                <div v-if="pathPlanningState.optimizedPath.length > 0" style="margin-top: 10px; max-height: 100px; overflow-y: auto; font-size: 0.8em;">
                  <div v-for="(point, index) in pathPlanningState.optimizedPath.slice(0, 5)" :key="index" 
                       style="padding: 2px; border-bottom: 1px dashed #ddd;">
                    #{{ index + 1 }}: ({{ point.x.toFixed(2) }}, {{ point.y.toFixed(2) }}, {{ point.z.toFixed(2) }})
                  </div>
                  <div v-if="pathPlanningState.optimizedPath.length > 5" style="color: #666; font-style: italic;">
                    ... and {{ pathPlanningState.optimizedPath.length - 5 }} more waypoints
                  </div>
                </div>
              </div>
              
              <div v-else style="color: #666; font-style: italic; font-size: 0.9em;">
                No optimized path available. Click "Optimize Path" to generate an optimized route.
              </div>
            </div>
            
            <button @click="executeWaypointNavigation" class="btn btn-success" style="margin-top: 10px;">Execute Waypoint Navigation</button>
          </div>
        </div>
        
        <!-- Spatial Constraints -->
        <div class="spatial-constraints">
          <h5>Spatial Constraints</h5>
          <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px;">
            <div class="constraint">
              <label>Max Distance:</label>
              <input type="number" v-model.number="spatialConstraints.maxDistance" min="0.1" max="10.0" step="0.1"> m
            </div>
            <div class="constraint">
              <label>Min Safe Distance:</label>
              <input type="number" v-model.number="spatialConstraints.minSafeDistance" min="0.1" max="5.0" step="0.1"> m
            </div>
            <div class="constraint">
              <label>Max Speed:</label>
              <input type="number" v-model.number="spatialConstraints.maxSpeed" min="0.1" max="2.0" step="0.1"> m/s
            </div>
            <div class="constraint">
              <label>Collision Threshold:</label>
              <input type="number" v-model.number="spatialConstraints.collisionThreshold" min="0.1" max="1.0" step="0.01"> m
            </div>
          </div>
        </div>
        
        <!-- Autonomous Control Buttons -->
        <div class="autonomous-controls">
          <button @click="startAutonomousNavigation" :disabled="!stereoCameraStatus.connected" class="btn btn-primary">
            Start Autonomous Navigation
          </button>
          <button @click="pauseAutonomousNavigation" class="btn btn-warning">Pause</button>
          <button @click="stopAutonomousNavigation" class="btn btn-danger">Stop</button>
          <button @click="calibrateNavigation" :disabled="!stereoCameraStatus.connected" class="btn btn-secondary">Calibrate Navigation</button>
          <button @click="exportNavigationData" class="btn btn-info">Export Navigation Data</button>
        </div>
        
        <!-- Navigation Status -->
        <div class="navigation-status" v-if="navigationState.active">
          <h5>Navigation Status</h5>
          <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px;">
            <div class="status-item">
              <strong>Mode:</strong> {{ navigationState.mode }}
            </div>
            <div class="status-item">
              <strong>Progress:</strong> {{ navigationState.progress || 0 }}%
            </div>
            <div class="status-item">
              <strong>Waypoint:</strong> {{ navigationState.currentWaypoint || 'N/A' }}
            </div>
            <div class="status-item">
              <strong>Obstacles:</strong> {{ navigationState.obstacleCount || 0 }}
            </div>
            <div class="status-item">
              <strong>Distance Traveled:</strong> {{ navigationState.distanceTraveled?.toFixed(2) || '0.00' }} m
            </div>
            <div class="status-item">
              <strong>Battery Usage:</strong> {{ navigationState.batteryUsage?.toFixed(1) || '0.0' }}%
            </div>
          </div>
        </div>
        
        <!-- Movement History -->
        <div class="movement-history">
          <h5>Movement History</h5>
          <div class="movement-history-controls" style="margin-bottom: 10px; display: flex; gap: 10px; align-items: center;">
            <span>Recorded Movements: {{ movementHistory.length }}</span>
            <button @click="clearMovementHistory" class="btn btn-sm btn-secondary">Clear History</button>
          </div>
          <div class="history-list">
            <div v-if="movementHistory.length === 0" style="padding: 10px; text-align: center; color: #999;">
              No movement history yet
            </div>
            <div v-else>
              <div v-for="(record, index) in movementHistory.slice(0, 10)" :key="index" class="history-item">
                <div style="display: flex; justify-content: space-between;">
                  <span>{{ new Date(record.timestamp).toLocaleTimeString() }}</span>
                  <span v-if="record.keyboardState.forward">Forward</span>
                  <span v-if="record.keyboardState.backward">Backward</span>
                  <span v-if="record.keyboardState.left">Left</span>
                  <span v-if="record.keyboardState.right">Right</span>
                </div>
                <div style="font-size: 0.8em; color: #666;">
                  Position: X={{ record.position?.x?.toFixed(2) || '0.00' }}, 
                  Y={{ record.position?.y?.toFixed(2) || '0.00' }}, 
                  Z={{ record.position?.z?.toFixed(2) || '0.00' }}
                </div>
              </div>
              <div v-if="movementHistory.length > 10" style="padding: 5px; text-align: center; color: #999; font-size: 0.8em;">
                ... and {{ movementHistory.length - 10 }} more
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Settings Dialog -->
    <div v-if="showSettingsDialog" class="settings-modal-overlay" @click.self="closeSettingsDialog">
      <div class="settings-modal">
        <div class="settings-modal-header">
          <h3>⚙️ Hardware Settings</h3>
          <button @click="closeSettingsDialog" class="btn-close">×</button>
        </div>
        <div class="settings-modal-body">
          
          <!-- Sensor Selection and Quantity Settings -->
          <div class="settings-section">
            <h4>📡 Sensor Selection and Quantity Settings</h4>
            <div class="device-quantity-control">
              <div v-for="sensor in availableSensors" :key="sensor.id" class="device-quantity-item">
                <div class="device-info">
                  <h5>{{ sensor.name }} <small>({{ sensor.type }})</small></h5>
                  <p class="device-description">{{ sensor.description }}</p>
                  <div class="device-instance-list">
                    <div v-for="instance in getSensorInstances(sensor.id)" :key="instance.id" class="device-instance">
                      <span>{{ instance.id }}: {{ instance.description }}</span>
                      <span class="instance-config">Config: {{ instance.config }}</span>
                    </div>
                  </div>
                </div>
                <div class="device-quantity-controls">
                  <div class="quantity-display">
                    <span>Quantity: {{ sensor.count }} / {{ sensor.maxCount }}</span>
                  </div>
                  <div class="quantity-buttons">
                    <button @click="decreaseDeviceCount('sensor', sensor.id)" 
                            :disabled="sensor.count <= 0" 
                            class="btn btn-sm btn-danger">-</button>
                    <button @click="increaseDeviceCount('sensor', sensor.id)" 
                            :disabled="sensor.count >= sensor.maxCount" 
                            class="btn btn-sm btn-success">+</button>
                  </div>
                  <div class="device-config-control">
                    <label>Default Config:
                      <select v-model="sensorConfig[sensor.id]" :disabled="sensor.count === 0">
                        <option value="active">Active</option>
                        <option value="passive">Passive</option>
                        <option value="monitor">Monitor</option>
                      </select>
                    </label>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Camera Selection and Quantity Settings -->
          <div class="settings-section">
            <h4>📷 Camera Selection and Quantity Settings</h4>
            <div class="device-quantity-control">
              <div v-for="camera in availableCameras" :key="camera.id" class="device-quantity-item">
                <div class="device-info">
                  <h5>{{ camera.name }} <small>({{ camera.type }})</small></h5>
                  <p class="device-description">{{ camera.description }}</p>
                  <div class="device-instance-list">
                    <div v-for="instance in getCameraInstances(camera.id)" :key="instance.id" class="device-instance">
                      <span>{{ instance.id }}: {{ instance.description }}</span>
                      <span class="instance-config">Resolution: {{ instance.resolution }}, Frame Rate: {{ instance.fps }} FPS</span>
                    </div>
                  </div>
                </div>
                <div class="device-quantity-controls">
                  <div class="quantity-display">
                    <span>Quantity: {{ camera.count }} / {{ camera.maxCount }}</span>
                  </div>
                  <div class="quantity-buttons">
                    <button @click="decreaseDeviceCount('camera', camera.id)" 
                            :disabled="camera.count <= 0" 
                            class="btn btn-sm btn-danger">-</button>
                    <button @click="increaseDeviceCount('camera', camera.id)" 
                            :disabled="camera.count >= camera.maxCount" 
                            class="btn btn-sm btn-success">+</button>
                  </div>
                  <div class="device-config-control">
                    <label>Resolution:
                      <select v-model="cameraConfig[camera.id].resolution" :disabled="camera.count === 0">
                        <option value="640x480">640×480</option>
                        <option value="1280x720">1280×720</option>
                        <option value="1920x1080">1920×1080</option>
                      </select>
                    </label>
                    <label>Frame Rate:
                      <select v-model="cameraConfig[camera.id].fps" :disabled="camera.count === 0">
                        <option value="15">15 FPS</option>
                        <option value="30">30 FPS</option>
                        <option value="60">60 FPS</option>
                      </select>
                    </label>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Multi-Connection Access Management and Settings -->
          <div class="settings-section">
            <h4>🔌 Multi-Connection Access Management and Settings</h4>
            <div class="connection-management">
              <div class="connection-management-header">
                <div class="connection-stats">
                  <span>Total Connections: {{ connectionList?.length || 0 }}</span>
                  <span>Active Connections: {{ activeConnectionsCount }}</span>
                  <span>Total Connected Devices: {{ totalConnectedDevices }}</span>
                </div>
                <button @click="addNewConnection" class="btn btn-primary btn-sm">
                  + Add New Connection
                </button>
              </div>
              
              <div class="connection-list">
                <div v-for="(conn, index) in connectionList" :key="conn.id" class="connection-item-card">
                  <div class="connection-card-header">
                    <div class="connection-status">
                      <span class="status-indicator" :class="conn.enabled ? 'active' : 'inactive'"></span>
                      <strong>{{ conn.description }}</strong>
                      <span class="connection-type">{{ conn.type.toUpperCase() }}</span>
                      <span class="connection-port">{{ conn.port }}</span>
                    </div>
                    <div class="connection-actions">
                      <button @click="toggleConnectionEnabled(index)" class="btn btn-sm" :class="conn.enabled ? 'btn-warning' : 'btn-success'">
                        {{ conn.enabled ? 'Disable' : 'Enable' }}
                      </button>
                      <button @click="removeConnection(index)" class="btn btn-sm btn-danger">Delete</button>
                    </div>
                  </div>
                  
                  <div class="connection-card-body">
                    <div class="connection-config-grid">
                      <div class="config-group">
                        <label>Connection Type:</label>
                        <select v-model="conn.type" @change="updateConnectionType(index, $event)">
                          <option v-for="connType in availableConnectionTypes" :key="connType.id" :value="connType.id">
                            {{ connType.name }} (max {{ connType.maxPorts }} ports)
                          </option>
                        </select>
                      </div>
                      
                      <div class="config-group">
                        <label>Port:</label>
                        <input type="text" v-model="conn.port" placeholder="Enter port">
                      </div>
                      
                      <div class="config-group">
                        <label>Description:</label>
                        <input type="text" v-model="conn.description" placeholder="Enter description">
                      </div>
                      
                      <div class="config-group">
                        <label>Auto Reconnect:</label>
                        <input type="checkbox" v-model="conn.autoReconnect">
                      </div>
                      
                      <div class="config-group">
                        <label>Timeout (ms):</label>
                        <input type="number" v-model.number="conn.timeout" min="100" max="30000" step="100">
                      </div>
                      
                      <!-- USB Specific Configuration -->
                      <div v-if="conn.type === 'usb'" class="config-group usb-specific">
                        <label>USB Version:</label>
                        <select v-model="conn.usbVersion">
                          <option value="1.1">USB 1.1 (12 Mbps)</option>
                          <option value="2.0">USB 2.0 (480 Mbps)</option>
                          <option value="3.0">USB 3.0 (5 Gbps)</option>
                          <option value="3.1">USB 3.1 (10 Gbps)</option>
                          <option value="3.2">USB 3.2 (20 Gbps)</option>
                        </select>
                      </div>
                      
                      <!-- Serial Port Specific Configuration -->
                      <div v-if="conn.type === 'serial'" class="config-group serial-specific">
                        <label>Baud Rate:</label>
                        <input type="number" v-model.number="conn.baudRate" min="9600" max="1152000" step="9600">
                      </div>
                      
                      <!-- Ethernet Specific Configuration -->
                      <div v-if="conn.type === 'ethernet'" class="config-group ethernet-specific">
                        <label>IP Address:</label>
                        <input type="text" v-model="conn.ipAddress" placeholder="Enter IP address">
                      </div>
                    </div>
                    
                    <!-- Input Type Selection -->
                    <div class="input-types-section">
                      <h5>Supported Input Types:</h5>
                      <div class="input-types-selection">
                        <div v-for="inputType in availableInputTypes" :key="inputType.id" class="input-type-checkbox">
                          <label>
                            <input type="checkbox" v-model="conn.inputTypes" :value="inputType.id">
                            {{ inputType.name }}
                            <small>{{ inputType.description }}</small>
                          </label>
                        </div>
                      </div>
                    </div>
                    
                    <!-- Connected Devices List -->
                    <div class="connected-devices-section">
                      <h5>Connected Devices ({{ conn.connectedDevices?.length || 0 }}):</h5>
                      <div class="connected-devices-list">
                        <span v-for="deviceId in conn.connectedDevices" :key="deviceId" class="device-tag">
                          {{ deviceId }}
                        </span>
                        <span v-if="!conn.connectedDevices || conn.connectedDevices.length === 0" class="no-devices">No connected devices</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div class="connection-management-footer">
                <button @click="testAllConnections" class="btn btn-primary">Test All Connections</button>
                <button @click="enableAllConnections" class="btn btn-success">Enable All Connections</button>
                <button @click="disableAllConnections" class="btn btn-warning">Disable All Connections</button>
                <button @click="saveConnectionsConfig" class="btn btn-info">Save Connection Configuration</button>
              </div>
            </div>
          </div>

          <!-- External Device Selection and Quantity Settings -->
          <div class="settings-section">
            <h4>🔧 External Device Selection and Quantity Settings</h4>
            <div class="device-quantity-control">
              <div v-for="device in availableDeviceTypes" :key="device.id" class="device-quantity-item">
                <div class="device-info">
                  <h5>{{ device.name }}</h5>
                  <p class="device-description">{{ device.description }}</p>
                  <div class="device-instance-list">
                    <div v-for="instance in getDeviceInstances(device.id)" :key="instance.id" class="device-instance">
                      <span>{{ instance.id }}: {{ instance.description }}</span>
                      <span class="instance-config">Nature: {{ instance.nature }}, Protocol: {{ instance.protocol }}</span>
                    </div>
                  </div>
                </div>
                <div class="device-quantity-controls">
                  <div class="quantity-display">
                    <span>Quantity: {{ device.count }} / {{ device.maxCount }}</span>
                  </div>
                  <div class="quantity-buttons">
                    <button @click="decreaseDeviceCount('device', device.id)" 
                            :disabled="device.count <= 0" 
                            class="btn btn-sm btn-danger">-</button>
                    <button @click="increaseDeviceCount('device', device.id)" 
                            :disabled="device.count >= device.maxCount" 
                            class="btn btn-sm btn-success">+</button>
                  </div>
                  <div class="device-config-control">
                    <label>Default Nature:
                      <select v-model="deviceTypeConfig[device.id].nature" :disabled="device.count === 0">
                        <option value="input">Input Device</option>
                        <option value="output">Output Device</option>
                        <option value="bidirectional">Bidirectional Device</option>
                        <option value="sensor">Sensor</option>
                        <option value="actuator">Actuator</option>
                      </select>
                    </label>
                    <label>Default Protocol:
                      <select v-model="deviceTypeConfig[device.id].protocol" :disabled="device.count === 0">
                        <option value="i2c">I2C</option>
                        <option value="spi">SPI</option>
                        <option value="uart">UART</option>
                        <option value="modbus">Modbus</option>
                        <option value="can">CAN</option>
                        <option value="usb">USB</option>
                        <option value="bluetooth">Bluetooth</option>
                        <option value="ethernet">Ethernet</option>
                      </select>
                    </label>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Manual Connection Status Management -->
          <div class="settings-section">
            <h4>🔗 Manual Connection Status Management</h4>
            <div class="connection-control">
              <div class="connection-status-display">
                <p>Current Connection Status: <span :class="connectionStatusClass">{{ connectionStatusText }}</span></p>
                <p>Connected Devices Count: {{ connectedDevicesCount }}</p>
              </div>
              <div class="connection-buttons">
                <button @click="forceReconnect" class="btn btn-primary">Force Reconnect</button>
                <button @click="disconnectAll" class="btn btn-danger">Disconnect All</button>
                <button @click="testConnection" class="btn btn-secondary">Test Connection</button>
                <button @click="saveConnectionProfile" class="btn btn-info">Save Configuration</button>
              </div>
            </div>
          </div>

        </div>
        <div class="settings-modal-footer">
          <button @click="applySettings" class="btn btn-primary">Apply Settings</button>
          <button @click="closeSettingsDialog" class="btn btn-secondary">Cancel</button>
          <button @click="resetSettings" class="btn btn-warning">Reset</button>
        </div>
      </div>
    </div>

    <!-- Joint Management Dialog -->
    <div v-if="showJointManagementDialog" class="joint-management-modal-overlay" @click.self="closeJointManagementDialog">
      <div class="joint-management-modal">
        <div class="joint-management-modal-header">
          <h3>🦾 Joint Management</h3>
          <button @click="closeJointManagementDialog" class="btn-close">×</button>
        </div>
        <div class="joint-management-modal-body">
          
          <!-- Existing Joints List -->
          <div class="joint-management-section">
            <h4>Existing Joints List</h4>
            <div class="joint-list-management">
              <div v-for="joint in jointList" :key="joint.id" class="joint-management-item">
                <div class="joint-info">
                  <strong>{{ joint.name }}</strong>
                  <div>ID: {{ joint.id }}</div>
                  <div>Angle range: {{ joint.min }}° to {{ joint.max }}°</div>
                  <div>Current value: {{ joint.value }}°</div>
                </div>
                <div class="joint-actions">
                  <button @click="editJoint(joint.id)" class="btn btn-sm btn-primary">Edit</button>
                  <button @click="removeJoint(joint.id)" class="btn btn-sm btn-danger">Delete</button>
                </div>
              </div>
            </div>
          </div>

          <!-- Add New Joint -->
          <div class="joint-management-section">
            <h4>Add New Joint</h4>
            <div class="add-joint-form">
              <div class="form-group">
                <label>Joint Name:</label>
                <input type="text" v-model="newJoint.name" placeholder="Enter joint name">
              </div>
              <div class="form-group">
                <label>Joint ID:</label>
                <input type="text" v-model="newJoint.id" placeholder="Enter joint ID">
              </div>
              <div class="form-group">
                <label>Minimum Angle (°):</label>
                <input type="number" v-model.number="newJoint.min" min="-360" max="360">
              </div>
              <div class="form-group">
                <label>Maximum Angle (°):</label>
                <input type="number" v-model.number="newJoint.max" min="-360" max="360">
              </div>
              <div class="form-group">
                <label>Step Size:</label>
                <input type="number" v-model.number="newJoint.step" min="0.1" max="10" step="0.1">
              </div>
              <div class="form-group">
                <label>Initial Value (°):</label>
                <input type="number" v-model.number="newJoint.value" :min="newJoint.min" :max="newJoint.max">
              </div>
              <div class="form-buttons">
                <button @click="addNewJoint" class="btn btn-primary">
                  {{ editingJointId ? 'Update Joint' : 'Add Joint' }}
                </button>
                <button @click="resetNewJointForm" class="btn btn-secondary">Reset Form</button>
              </div>
            </div>
          </div>

          <!-- Batch Joint Settings -->
          <div class="joint-management-section">
            <h4>Batch Joint Settings</h4>
            <div class="batch-settings">
              <div class="form-group">
                <label>Set Minimum Angle for All Joints:</label>
                <input type="number" v-model.number="batchMin" min="-360" max="360">
                <button @click="applyBatchMin" class="btn btn-sm btn-warning">Apply</button>
              </div>
              <div class="form-group">
                <label>Set Maximum Angle for All Joints:</label>
                <input type="number" v-model.number="batchMax" min="-360" max="360">
                <button @click="applyBatchMax" class="btn btn-sm btn-warning">Apply</button>
              </div>
              <div class="form-group">
                <label>Set Step Size for All Joints:</label>
                <input type="number" v-model.number="batchStep" min="0.1" max="10" step="0.1">
                <button @click="applyBatchStep" class="btn btn-sm btn-warning">Apply</button>
              </div>
            </div>
          </div>

        </div>
        <div class="joint-management-modal-footer">
          <button @click="saveJointConfiguration" class="btn btn-primary">Save Configuration</button>
          <button @click="closeJointManagementDialog" class="btn btn-secondary">Cancel</button>
          <button @click="resetJointConfiguration" class="btn btn-danger">Reset to Default</button>
        </div>
      </div>
    </div>

    <!-- Chat with Robot Dialog -->
    <div v-if="showChatDialog" class="chat-modal-overlay" @click.self="closeChatDialog">
      <div class="chat-modal">
        <div class="chat-modal-header">
          <h3>🤖 Chat with Robot</h3>
          <button @click="closeChatDialog" class="btn-close">×</button>
        </div>
        <div class="chat-modal-body">
          <div class="chat-messages" ref="chatMessagesContainer">
            <div v-for="(message, index) in chatMessages" :key="index" class="chat-message" :class="message.type">
              <div class="message-content">{{ message.text }}</div>
              <div class="message-time">{{ message.time }}</div>
            </div>
          </div>
          <div class="chat-input-area">
            <textarea v-model="chatInputText" @keydown.enter.prevent="sendChatMessage" placeholder="Enter message" rows="3"></textarea>
            <div class="chat-buttons">
              <button @click="sendChatMessage" class="btn btn-primary">Send</button>
              <button @click="clearChat" class="btn btn-secondary">Clear</button>
              <button @click="toggleVoiceInput" class="btn btn-info" :class="voiceInputStatus.class">
                <span class="voice-status-indicator" :class="voiceInputStatus.class"></span>
                {{ voiceInputButtonText }}
              </button>
            </div>
          </div>
        </div>
        <div class="chat-modal-footer">
          <button @click="closeChatDialog" class="btn btn-secondary">Close</button>
          <button @click="exportChat" class="btn btn-info">Export Chat</button>
        </div>
      </div>
    </div>

    <!-- Debug Information -->
    <div class="debug-info">
      <h4>Debug Information</h4>
      <pre>{{ debugInfo }}</pre>
      <p>Click Count: {{ clickCount }}</p>
      <p>Current Time: {{ new Date().toLocaleTimeString() }}</p>
    </div>
    
    <!-- Confirmation Dialog -->
    <div v-if="uiConfirmState.show" class="confirmation-dialog-overlay" @click.self="cancelAction">
      <div class="confirmation-dialog">
        <div class="confirmation-dialog-header">
          <h4>{{ uiConfirmState.title }}</h4>
          <button @click="cancelAction" class="btn-close">&times;</button>
        </div>
        <div class="confirmation-dialog-body">
          <p>{{ uiConfirmState.message }}</p>
        </div>
        <div class="confirmation-dialog-footer">
          <button @click="cancelAction" class="btn btn-secondary">{{ uiConfirmState.cancelText }}</button>
          <button @click="confirmAction" class="btn btn-danger">{{ uiConfirmState.confirmText }}</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, reactive, onMounted, onUnmounted, watch, computed, onErrorCaptured } from 'vue'
import { useRoute } from 'vue-router'
import { apiClient } from '@/utils/api'
import api from '@/utils/api'
import { performDataLoad, performDataOperation } from '@/utils/operationHelpers'
import HardwareCategory from '@/components/HardwareCategory.vue'
import { Chart, registerables } from 'chart.js'



export default {
  name: 'RobotSettingsView',
  
  components: {
    HardwareCategory
  },
  
  setup() {
    // Register Chart.js components
    Chart.register(...registerables)
    
    if (import.meta.env.DEV) {
      console.log('RobotSettingsView setup() started executing')
    }

    // Error handling
    onErrorCaptured((error, instance, info) => {
      console.error('Error captured in RobotSettingsView:', error, info)
      // Prevent error propagation
      return false
    })
    
    // Component mount status tracking
    const isMounted = ref(true)
    
    // Route handling
    const route = useRoute()
    
    // Robot state
    const robotState = reactive({
      status: 'disconnected',
      statusText: 'Hardware not connected',
      battery: 0,
      connected: false,
      temperature: 0
    })

    // Hardware connection status
    const hardwareState = reactive({
      initialized: false,
      hardwareDetected: false,
      loading: false,
      detectedDevices: {
        joints: 0,
        sensors: 0,
        cameras: 0
      },
      detectedHardwareList: {
        joints: [],
        sensors: [],
        cameras: []
      }
      

    })

    // Hardware list display status
    const showConnectedHardwareList = ref(true)
    const showAvailableHardwareList = ref(false)
    
    // Configuration-driven hardware category definitions
    const connectedHardwareCategories = computed(() => {
      try {
        return [
          {
            title: 'Sensors',
            items: sensorInstances,
            itemType: 'sensor-instance',
            status: 'connected'
          },
          {
            title: 'Cameras',
            items: cameraInstances,
            itemType: 'camera-instance',
            status: 'connected'
          },
          {
            title: 'Devices',
            items: deviceInstances,
            itemType: 'device-instance',
            status: 'connected'
          },
          {
            title: 'Connections',
            items: connectionList,
            itemType: 'connection',
            statusGetter: (conn) => conn.enabled ? 'connected' : 'disconnected'
          }
        ]
      } catch (error) {
        console.error('Error in connectedHardwareCategories:', error)
        return []
      }
    })
    
    const availableHardwareCategories = computed(() => {
      try {
        return [
          {
            title: 'Sensor Types',
            items: availableSensors,
            itemType: 'sensor-type',
            status: 'available'
          },
          {
            title: 'Camera Types',
            items: availableCameras,
            itemType: 'camera-type',
            status: 'available'
          },
          {
            title: 'Device Types',
            items: availableDeviceTypes,
            itemType: 'device-type',
            status: 'available'
          },
          {
            title: 'Connection Types',
            items: availableConnectionTypes,
            itemType: 'connection-type',
            status: 'available'
          },
          {
            title: 'Input Types',
            items: availableInputTypes,
            itemType: 'input-type',
            status: 'available'
          }
        ]
      } catch (error) {
        console.error('Error in availableHardwareCategories:', error)
        return []
      }
    })
    
    // Unified toggle utility functions
    const createToggle = (options = {}) => {
      const {
        refValue,           // ref to toggle
        reactiveObj,        // reactive object to toggle property
        propPath,           // property path in reactive object (e.g., 'enabled' or 'state.enabled')
        arrayRef,           // ref to array for item toggling
        itemId,             // item ID for array toggling
        arrayPropRef,       // ref to array for property toggling
        arrayIndex,         // index for array property toggling
        propName,           // property name for array property toggling
        onToggle = null,    // callback function after toggle
        logMessage = null,  // debug log message
        successMessage = null // user feedback message
      } = options
      
      return () => {
        // Boolean toggle for ref
        if (refValue) {
          refValue.value = !refValue.value
          if (logMessage && import.meta.env.DEV) {
            console.log(`[Toggle] ${logMessage}: ${refValue.value}`)
          }
          if (successMessage) {
            debugInfo.value = refValue.value ? `${successMessage} enabled` : `${successMessage} disabled`
          }
          if (onToggle) {
            onToggle(refValue.value)
          }
          return refValue.value
        }
        
        // Boolean toggle for reactive object property
        if (reactiveObj && propPath) {
          // Support nested property paths
          const pathParts = propPath.split('.')
          let currentObj = reactiveObj
          for (let i = 0; i < pathParts.length - 1; i++) {
            currentObj = currentObj[pathParts[i]]
            if (!currentObj) break
          }
          
          if (currentObj && pathParts.length > 0) {
            const prop = pathParts[pathParts.length - 1]
            if (prop in currentObj) {
              currentObj[prop] = !currentObj[prop]
              if (logMessage && import.meta.env.DEV) {
                console.log(`[Toggle] ${logMessage}: ${currentObj[prop]}`)
              }
              if (successMessage) {
                debugInfo.value = currentObj[prop] ? `${successMessage} enabled` : `${successMessage} disabled`
              }
              if (onToggle) {
                onToggle(currentObj[prop])
              }
              return currentObj[prop]
            }
          }
        }
        
        // Array item toggle (add/remove)
        if (arrayRef && itemId !== undefined) {
          const itemIdStr = String(itemId)
          const index = arrayRef.value.indexOf(itemIdStr)
          if (index === -1) {
            arrayRef.value.push(itemIdStr)
            if (logMessage && import.meta.env.DEV) {
              console.log(`[Toggle] ${logMessage}: added ${itemIdStr}`)
            }
            if (successMessage) {
              debugInfo.value = `${successMessage} added: ${itemIdStr}`
            }
            if (onToggle) {
              onToggle(true, itemIdStr)
            }
            return true
          } else {
            arrayRef.value.splice(index, 1)
            if (logMessage && import.meta.env.DEV) {
              console.log(`[Toggle] ${logMessage}: removed ${itemIdStr}`)
            }
            if (successMessage) {
              debugInfo.value = `${successMessage} removed: ${itemIdStr}`
            }
            if (onToggle) {
              onToggle(false, itemIdStr)
            }
            return false
          }
        }
        
        // Array property toggle
        if (arrayPropRef && arrayIndex !== undefined && propName) {
          const item = arrayPropRef.value[arrayIndex]
          if (item && propName in item) {
            item[propName] = !item[propName]
            const status = item[propName] ? 'enabled' : 'disabled'
            if (logMessage && import.meta.env.DEV) {
              console.log(`[Toggle] ${logMessage}: ${status}`)
            }
            if (successMessage) {
              debugInfo.value = `${successMessage}: ${status}`
            }
            if (onToggle) {
              onToggle(item[propName], arrayIndex, item)
            }
            return item[propName]
          }
        }
        
        if (import.meta.env.DEV) {
          console.warn('[Toggle] Invalid toggle configuration')
        }
        return null
      }
    }
    
    // Toggle factory for common patterns
    const createBooleanToggle = (refValue, message) => {
      return createToggle({
        refValue,
        logMessage: message,
        successMessage: message
      })
    }
    
    const createReactiveToggle = (reactiveObj, propPath, message) => {
      return createToggle({
        reactiveObj,
        propPath,
        logMessage: message,
        successMessage: message
      })
    }
    
    const createArrayItemToggle = (arrayRef, itemId, message) => {
      return createToggle({
        arrayRef,
        itemId,
        logMessage: message,
        successMessage: message
      })
    }
    
    const createArrayPropertyToggle = (arrayRef, index, propName, message) => {
      return createToggle({
        arrayPropRef: arrayRef,
        arrayIndex: index,
        propName,
        logMessage: message,
        successMessage: message
      })
    }
    
    // Total connected hardware count computed property
    const totalConnectedHardwareCount = computed(() => {
      return (sensorInstances.value?.length || 0) + (cameraInstances.value?.length || 0) + (deviceInstances.value?.length || 0) + (connectionList.value?.length || 0)
    })
    
    // Toggle hardware list display - using unified toggle system
    const toggleConnectedHardwareList = createBooleanToggle(
      showConnectedHardwareList,
      'Connected hardware list'
    )
    
    const toggleAvailableHardwareList = createBooleanToggle(
      showAvailableHardwareList,
      'Available hardware type list'
    )

    // Voice settings status
    const voiceControlState = reactive({
      enabled: false,
      listening: false,
      supported: false
    })
    
    // Voice input button text computed property
    const voiceInputButtonText = computed(() => {
      if (!voiceInputActive.value) {
        return 'Voice Input'
      } else {
        if (isRecording.value) {
          return 'Recording...'
        } else if (useBackendRecognition.value) {
          return 'Using Backend API'
        } else {
          return 'Using Browser Recognition'
        }
      }
    })
    
    // Voice input status indicator
    const voiceInputStatus = computed(() => {
      if (!voiceInputActive.value) {
        return { text: 'Voice input ready', class: 'ready' }
      } else if (isRecording.value) {
        return { text: 'Recording audio...', class: 'recording' }
      } else if (useBackendRecognition.value) {
        return { text: 'Using backend speech recognition', class: 'backend' }
      } else {
        return { text: 'Using browser speech recognition', class: 'browser' }
      }
    })

    // Joint data
    const joints = reactive({
      arm: {
        left: { shoulder: 0, elbow: 0, wrist: 0 },
        right: { shoulder: 0, elbow: 0, wrist: 0 }
      },
      leg: {
        left: { hip: 0, knee: 0, ankle: 0 },
        right: { hip: 0, knee: 0, ankle: 0 }
      },
      head: { pan: 0, tilt: 0 },
      torso: { twist: 0, bend: 0 }
    })

    // Sensor data
    const sensorData = ref([])
    
    // Sensor calibration state
    const sensorCalibrationState = reactive({
      calibrating: false,
      calibrationProgress: 0,
      calibrationStatus: 'idle',
      calibrationResults: {},
      calibrationError: null,
      calibrationTimestamp: null,
      selectedSensorTypes: []
    })
    
    // Data recording and playback state
    const dataRecordingState = reactive({
      recording: false,
      playback: false,
      records: [],
      currentRecordIndex: -1,
      recordStartTime: null,
      recordDuration: 0,
      playbackSpeed: 1.0,
      maxRecords: 1000,
      recordInterval: 100, // milliseconds
      recordTimer: null,
      playbackTimer: null
    })
    
    // UI confirmation dialog state
    const uiConfirmState = reactive({
      show: false,
      title: '',
      message: '',
      confirmText: 'Confirm',
      cancelText: 'Cancel',
      onConfirm: null,
      onCancel: null
    })
    
    // Sensor data visualization
    const sensorChartCanvas = ref(null)
    const sensorChartData = ref([])
    const sensorChartDataSource = ref('simulation')
    const sensorChartType = ref('line')
    const sensorChartTimeWindow = ref(60) // 1 minute
    const sensorChartInstance = ref(null)

    // Joint list
    const jointList = ref([])
    
    // Joint presets
    const jointPresets = ref([
      {
        name: 'Home Position',
        description: 'All joints at zero position',
        positions: { default: 0 }
      },
      {
        name: 'Ready Position',
        description: 'Ready position for operation',
        positions: { default: 45 }
      },
      {
        name: 'Max Range',
        description: 'All joints at maximum range',
        positions: { default: 'max' }
      },
      {
        name: 'Min Range',
        description: 'All joints at minimum range',
        positions: { default: 'min' }
      }
    ])

    // Click count
    const clickCount = ref(0)
    
    // Debounce timer for joint input
    const jointInputDebounceTimer = ref(null)

    // Debug information
    const debugInfo = ref('')

    // Initialize joint list
    const initJointList = () => {
      // Initialize joint list based on detected hardware list
      if (hardwareState.detectedHardwareList?.joints && hardwareState.detectedHardwareList.joints.length > 0) {
        // Use real detected hardware data
        jointList.value = hardwareState.detectedHardwareList?.joints.map(joint => ({
          id: joint.id,
          name: joint.name,
          value: 0, // Initial position
          min: -180, // Default minimum value, should be adjusted according to actual hardware
          max: 180,  // Default maximum value, should be adjusted according to actual hardware
          step: 1,
          speed: 50, // Default speed (50%)
          status: 'ready', // Initial status
          detected: joint.detected,
          selected: joint.selected,
          type: joint.type
        }))
        debugInfo.value = `Initialized ${jointList.value.length} joints based on detected hardware`
      } else {
        // No hardware detected, joint list remains empty
        jointList.value = []
        debugInfo.value = 'No joint hardware detected, joint list is empty. Please connect real robot hardware.'
      }
    }

    // Collaboration settings status
    const collaborationState = reactive({
      status: 'idle',
      loading: false,
      activePattern: null,
      lastResult: null
    })

    // Collaboration patterns list
    const collaborationPatterns = ref([])

    // Selected collaboration pattern
    const selectedPattern = ref('')

    // Collaboration input data
    const collaborationInput = ref('{}')

    // Collaboration configuration
    const collaborationConfig = reactive({
      inputType: 'text',
      outputTypes: ['text', 'action'],
      priority: 'medium',
      timeout: 60,
      realtimeMonitoring: true
    })

    // Selected collaboration pattern details
    const selectedPatternDetails = ref(null)

    // ========== Robot Training Module Data Definitions ==========
    
    // Robot training state
    const trainingState = reactive({
      status: 'idle', // idle, training, paused, completed, error
      progress: 0,
      activeTraining: null,
      error: null
    })

    // Hardware status
    const hardwareStatus = reactive({
      joints: 0,
      sensors: 0,
      cameras: 0,
      battery: 0,
      systemTemperature: 0,
      initialized: false,
      lastUpdate: null
    })

    // Available training modes
    const trainingModes = ref([
      { id: 'motion_basic', name: 'Basic Motion Training', description: 'Fundamental robot motion control and coordination training' },
      { id: 'perception_training', name: 'Perception Training', description: 'Visual and sensor perception system training' },
      { id: 'collaboration_training', name: 'Collaboration Training', description: 'Multi-model collaboration for complex tasks' },
      { id: 'agi_fusion', name: 'AGI Fusion Training', description: 'Advanced AGI-level cognitive and motor integration' }
    ])
    
    // Debug: log training modes
    if (import.meta.env.DEV) {
      console.log('Training modes initialized:', trainingModes.value.map(mode => ({ id: mode.id, name: mode.name })))
    }

    // Available training models (27 models from user specification)
    const availableTrainingModels = ref([
      { id: 'manager', name: 'Manager Model', port: 8001, description: 'System coordination and task management' },
      { id: 'language', name: 'Language Model', port: 8002, description: 'Natural language processing and understanding' },
      { id: 'knowledge', name: 'Knowledge Model', port: 8003, description: 'Knowledge storage, retrieval and reasoning' },
      { id: 'vision', name: 'Vision Model', port: 8004, description: 'Image and visual data processing' },
      { id: 'audio', name: 'Audio Model', port: 8005, description: 'Audio processing and speech recognition' },
      { id: 'autonomous', name: 'Autonomous Model', port: 8006, description: 'Autonomous decision making and planning' },
      { id: 'programming', name: 'Programming Model', port: 8007, description: 'Code generation and system programming' },
      { id: 'planning', name: 'Planning Model', port: 8008, description: 'Task planning and execution sequencing' },
      { id: 'emotion', name: 'Emotion Model', port: 8009, description: 'Emotion recognition and response generation' },
      { id: 'spatial', name: 'Spatial Model', port: 8010, description: 'Spatial awareness and navigation' },
      { id: 'computer_vision', name: 'Computer Vision Model', port: 8011, description: 'Advanced computer vision algorithms' },
      { id: 'sensor', name: 'Sensor Model', port: 8012, description: 'Sensor data processing and fusion' },
      { id: 'motion', name: 'Motion Model', port: 8013, description: 'Motion planning and control' },
      { id: 'prediction', name: 'Prediction Model', port: 8014, description: 'Future state prediction and forecasting' },
      { id: 'advanced_reasoning', name: 'Advanced Reasoning Model', port: 8015, description: 'Complex logical reasoning and problem solving' },
      { id: 'data_fusion', name: 'Data Fusion Model', port: 8028, description: 'Multi-source data integration and fusion' },
      { id: 'creative_problem_solving', name: 'Creative Problem Solving Model', port: 8017, description: 'Creative thinking and innovative solutions' },
      { id: 'meta_cognition', name: 'Meta Cognition Model', port: 8018, description: 'Self-awareness and learning optimization' },
      { id: 'value_alignment', name: 'Value Alignment Model', port: 8019, description: 'Ethical decision making and value alignment' },
      { id: 'vision_image', name: 'Vision Image Model', port: 8020, description: 'Static image analysis and processing' },
      { id: 'vision_video', name: 'Vision Video Model', port: 8021, description: 'Video stream processing and analysis' },
      { id: 'finance', name: 'Finance Model', port: 8022, description: 'Financial analysis and prediction' },
      { id: 'medical', name: 'Medical Model', port: 8023, description: 'Medical data analysis and diagnosis' },
      { id: 'collaboration', name: 'Collaboration Model', port: 8024, description: 'Model collaboration and coordination' },
      { id: 'optimization', name: 'Optimization Model', port: 8025, description: 'System optimization and performance tuning' },
      { id: 'computer', name: 'Computer Model', port: 8026, description: 'Computer system management and control' },
      { id: 'mathematics', name: 'Mathematics Model', port: 8027, description: 'Mathematical computation and modeling' }
    ])

    // Selected training mode
    const selectedTrainingMode = ref('')

    // Selected training models
    const selectedTrainingModels = ref([])

    // Dataset selection
    const selectedDataset = ref('')
    const datasets = ref([])
    const datasetsLoading = ref(false)
    const datasetsError = ref(null)

    // Selected hardware components for training
    const selectedJoints = ref([])
    const selectedSensors = ref([])
    const selectedCameras = ref([])

    // Training parameters
    const trainingParams = reactive({
      iterations: 1000,
      learningRate: 0.001,
      batchSize: 32,
      validationSplit: 0.2,
      device: 'cpu'
    })

    // Safety limits
    const safetyLimits = reactive({
      maxJointVelocity: 1.0,
      maxJointTorque: 5.0,
      maxTemperature: 70.0,
      emergencyStopThreshold: 1.5
    })

    // Training log
    const trainingLog = ref([])

    // Add log entry
    const addTrainingLog = (message, type = 'info') => {
      const timestamp = new Date().toLocaleTimeString()
      trainingLog.value.push({
        timestamp,
        message,
        type
      })
      // Keep log size manageable
      if (trainingLog.value.length > 100) {
        trainingLog.value = trainingLog.value.slice(-50)
      }
    }

    // Toggle training model selection - using unified toggle system
    const toggleTrainingModel = (modelId, event) => {
      if (import.meta.env.DEV) {
        console.log('=== toggleTrainingModel DEBUG START ===')
        console.log('Event received:', event)
        console.log('Event type:', event?.type)
        console.log('Event target:', event?.target)
        console.log('Model ID parameter:', modelId, 'Type:', typeof modelId)
        
        const modelIdStr = String(modelId)
        console.log('toggleTrainingModel called with modelId:', modelId, '(converted to:', modelIdStr, ')')
        console.log('current selectedTrainingModels:', selectedTrainingModels.value)
        console.log('availableTrainingModels count:', availableTrainingModels.value.length)
        
        // Debug: log all available models
        console.log('Available models:', availableTrainingModels.value.map(m => ({id: m.id, name: m.name})))
      }
      
      const modelIdStr = String(modelId)
      
      // Use unified toggle system with custom onToggle callback
      const toggle = createToggle({
        arrayRef: selectedTrainingModels,
        itemId: modelIdStr,
        logMessage: 'Training model',
        successMessage: 'Training model',
        onToggle: (isAdded, itemId) => {
          if (isAdded) {
            addTrainingLog(`Model selected: ${itemId}`)
          } else {
            addTrainingLog(`Model deselected: ${itemId}`)
          }
        }
      })
      
      // Execute toggle
      const toggleResult = toggle()
      
      if (import.meta.env.DEV) {
        console.log('updated selectedTrainingModels:', selectedTrainingModels.value)
        console.log('=== toggleTrainingModel DEBUG END ===')
        
        // Force UI update
        setTimeout(() => {
          console.log('Post-update check - selectedTrainingModels:', selectedTrainingModels.value)
        }, 0)
      }
      
      return toggleResult
    }

    // Check if a model is selected
    const isModelSelected = (modelId) => {
      const modelIdStr = String(modelId)
      const isSelected = selectedTrainingModels.value.includes(modelIdStr)
      if (import.meta.env.DEV) {
        console.log('isModelSelected:', modelId, '->', isSelected, '(converted to:', modelIdStr, ')')
      }
      return isSelected
    }

    // Load datasets from API
    const loadDatasets = async () => {
      datasetsLoading.value = true
      datasetsError.value = null
      
      try {
        const response = await api.datasets.get()
        
        if (response.data && response.data.status === 'success' && response.data.data) {
          datasets.value = response.data.data
          addTrainingLog(`Loaded ${datasets.value.length} datasets from API`)
          
          // Set default dataset if available and not already selected
          if (!selectedDataset.value && datasets.value.length > 0) {
            selectedDataset.value = datasets.value[0].id
            addTrainingLog(`Default dataset selected: ${datasets.value[0].id}`)
          }
        } else {
          throw new Error('API returned unsuccessful status')
        }
      } catch (error) {
        const errorMsg = error.response?.data?.detail || error.message || 'Failed to load datasets'
        datasetsError.value = errorMsg
        addTrainingLog(`Error loading datasets: ${errorMsg}`, 'error')
      } finally {
        datasetsLoading.value = false
      }
    }

    // Upload dataset handler - Real API implementation
    const handleDatasetUpload = async (event) => {
      const files = event.target.files
      if (!files || files.length === 0) return
      
      const file = files[0]
      addTrainingLog(`Uploading dataset file: ${file.name}`)
      
      try {
        // Create FormData for file upload
        const formData = new FormData()
        formData.append('file', file)
        
        // Add metadata for robot training
        formData.append('dataset_type', 'robot_training')
        formData.append('description', `Robot training dataset: ${file.name}`)
        
        // Show upload progress
        addTrainingLog(`Uploading ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB) to backend...`)
        
        // Call real API
        const response = await api.datasets.upload(formData)
        
        if (response.data && response.data.status === 'success') {
          addTrainingLog(`Dataset uploaded successfully: ${response.data.data?.name || file.name}`, 'success')
          
          // Refresh dataset list
          await loadDatasets()
          
          // Select the newly uploaded dataset
          if (response.data.data && response.data.data.id) {
            selectedDataset.value = response.data.data.id
            addTrainingLog(`Selected dataset: ${response.data.data.name || response.data.data.id}`)
          }
        } else {
          addTrainingLog(`Dataset upload failed: ${response.data?.message || 'Unknown error'}`, 'error')
        }
      } catch (error) {
        const errorMsg = error.response?.data?.detail || error.response?.data?.message || error.message || 'Upload failed'
        addTrainingLog(`Error uploading dataset: ${errorMsg}`, 'error')
      } finally {
        // Reset file input
        event.target.value = ''
      }
    }

    // Start robot training - Unified API version
    const startRobotTraining = async () => {
      if (!selectedTrainingMode.value || selectedTrainingModels.value.length === 0) {
        addTrainingLog('Cannot start training: Please select training mode and at least one model', 'error')
        return
      }

      // Prepare unified training configuration
      const unifiedTrainingConfig = {
        mode: selectedTrainingMode.value,
        models: selectedTrainingModels.value,
        dataset: selectedDataset.value,
        parameters: trainingParams,
        hardware_config: {
          selected_joints: selectedJoints.value,
          selected_sensors: selectedSensors.value,
          selected_cameras: selectedCameras.value,
          safety_limits: safetyLimits
        }
      }

      addTrainingLog(`Sending unified training request: ${selectedTrainingMode.value} with ${selectedTrainingModels.value.length} models`)

      try {
        // Call unified training API
        const response = await api.training.start(unifiedTrainingConfig)
        
        // Unified API returns {status: "success", "job_id": "...", "training_type": "..."}
        if (response.data.status === 'success') {
          const trainingId = response.data.job_id
          
          // Update local training state with API response
          trainingState.status = 'training'
          trainingState.progress = 0
          trainingState.trainingId = trainingId
          trainingState.activeTraining = {
            trainingId,
            mode: selectedTrainingMode.value,
            models: [...selectedTrainingModels.value],
            startedAt: new Date().toISOString(),
            training_type: response.data.training_type || 'robot_hardware'
          }
          
          addTrainingLog(`Training started successfully: ${trainingId} (Type: ${response.data.training_type || 'robot_hardware'})`, 'success')
          
          // Start polling for training progress
          startTrainingProgressPolling(trainingId)
        } else {
          addTrainingLog(`Failed to start training: ${response.data.message || 'Unknown error'}`, 'error')
          trainingState.error = response.data.message || 'Unknown error'
        }
      } catch (error) {
        const errorMsg = error.response?.data?.detail || error.message || 'Unknown error'
        addTrainingLog(`Training API error: ${errorMsg}`, 'error')
        trainingState.error = errorMsg
        trainingState.status = 'error'
      }
    }

    // Start polling for training progress - Unified API version
    let progressPollInterval = null
    const startTrainingProgressPolling = (trainingId) => {
      // Clear any existing interval
      if (progressPollInterval) {
        clearInterval(progressPollInterval)
      }
      
      // Poll every 2 seconds
      progressPollInterval = setInterval(async () => {
        try {
          const response = await api.training.status(trainingId)
          // Unified API returns {status: "success", data: {status: "...", progress: ...}}
          if (response.data.status === 'success') {
            const statusData = response.data.data
            trainingState.progress = statusData.progress || 0
            
            // Map unified status to local status
            const unifiedStatus = statusData.status || 'unknown'
            trainingState.status = unifiedStatus
            
            // Update log with progress
            if (trainingState.progress > 0 && trainingState.progress < 100) {
              addTrainingLog(`Training progress: ${trainingState.progress}% (Status: ${unifiedStatus})`, 'info')
            }
            
            // Stop polling if training is completed or error
            if (unifiedStatus === 'completed' || unifiedStatus === 'error' || unifiedStatus === 'idle' || unifiedStatus === 'stopped') {
              clearInterval(progressPollInterval)
              progressPollInterval = null
              
              if (unifiedStatus === 'completed') {
                trainingState.status = 'completed'
                addTrainingLog('Training completed successfully', 'success')
              } else if (unifiedStatus === 'error') {
                trainingState.status = 'error'
                addTrainingLog('Training encountered an error', 'error')
              } else if (unifiedStatus === 'stopped') {
                trainingState.status = 'idle'
                addTrainingLog('Training stopped', 'info')
              }
            }
          }
        } catch (error) {
          addTrainingLog(`Progress polling error: ${error.message}`, 'error')
        }
      }, 2000)
    }

    // Pause robot training - Local state management (unified training system)
    const pauseRobotTraining = async () => {
      if (trainingState.status !== 'training') {
        addTrainingLog('Training is not active, cannot pause', 'warning')
        return
      }
      
      const trainingId = trainingState.trainingId
      if (!trainingId) {
        addTrainingLog('No active training session found', 'error')
        return
      }
      
      addTrainingLog(`Pausing training session: ${trainingId} (local state only)`)
      
      // Note: Unified training API does not support pause, so we only update local state
      // The training will continue in the backend but frontend shows it as paused
      trainingState.status = 'paused'
      addTrainingLog('Training marked as paused (local state)', 'info')
      
      // Note: To actually pause training, consider using the stop function
      addTrainingLog('Note: To fully stop training, use the Stop button', 'info')
    }

    // Stop robot training - Unified API version
    const stopRobotTraining = async () => {
      if (trainingState.status !== 'training' && trainingState.status !== 'paused') {
        addTrainingLog('Training is not active or paused, cannot stop', 'warning')
        return
      }
      
      const trainingId = trainingState.trainingId
      if (!trainingId) {
        addTrainingLog('No active training session found', 'error')
        return
      }
      
      addTrainingLog(`Sending stop request for training: ${trainingId}`)
      
      try {
        const response = await api.training.stop(trainingId)
        
        // Unified API returns {status: "success", stopped: true/false}
        if (response.data.status === 'success') {
          trainingState.status = 'idle'
          trainingState.progress = 0
          trainingState.activeTraining = null
          addTrainingLog('Training stopped successfully', 'success')
          
          // Clear progress polling interval
          if (progressPollInterval) {
            clearInterval(progressPollInterval)
            progressPollInterval = null
          }
        } else {
          addTrainingLog(`Failed to stop training: ${response.data.message || 'Unknown error'}`, 'error')
        }
      } catch (error) {
        const errorMsg = error.response?.data?.detail || error.message || 'Unknown error'
        addTrainingLog(`Stop API error: ${errorMsg}`, 'error')
      }
    }

    // Reset training configuration - Unified training system
    const resetTrainingConfig = async () => {
      const trainingId = trainingState.trainingId
      
      if (trainingId && trainingState.status === 'training') {
        addTrainingLog('Training is active, please stop training before resetting configuration', 'warning')
        return
      }
      
      // Note: Unified training API does not have reset endpoint
      // We only reset local configuration state
      if (trainingId) {
        addTrainingLog(`Resetting configuration for training: ${trainingId} (local state only)`)
      } else {
        addTrainingLog('Resetting training configuration', 'info')
      }
      
      // Reset local state
      selectedTrainingMode.value = ''
      selectedTrainingModels.value = []
      selectedDataset.value = ''
      selectedJoints.value = []
      selectedSensors.value = []
      selectedCameras.value = []
      
      trainingParams.iterations = 1000
      trainingParams.learningRate = 0.001
      trainingParams.batchSize = 32
      trainingParams.validationSplit = 0.2
      
      safetyLimits.maxJointVelocity = 1.0
      safetyLimits.maxJointTorque = 5.0
      safetyLimits.maxTemperature = 70.0
      safetyLimits.emergencyStopThreshold = 1.5
      
      trainingState.status = 'idle'
      trainingState.progress = 0
      trainingState.activeTraining = null
      trainingState.error = null
      trainingState.trainingId = null
      
      // Clear progress polling interval
      if (progressPollInterval) {
        clearInterval(progressPollInterval)
        progressPollInterval = null
      }
      
      addTrainingLog('Training configuration reset', 'info')
    }

    // ========== Hardware Management Functions ==========
    
    // Refresh hardware status
    const refreshHardwareStatus = async () => {
      try {
        addTrainingLog('Fetching hardware status...', 'info')
        const response = await apiClient.get('/api/robot/hardware/status')
        
        if (response.data.status === 'success') {
          const hardwareData = response.data.data
          
          // Update hardware status
          hardwareStatus.joints = hardwareData?.joints_connected || 0
          hardwareStatus.sensors = hardwareData?.sensors_connected || 0
          hardwareStatus.cameras = hardwareData?.cameras_connected || 0
          hardwareStatus.battery = hardwareData?.battery_level || 0
          hardwareStatus.systemTemperature = hardwareData?.system_temperature || 0
          hardwareStatus.initialized = hardwareData?.initialized || false
          hardwareStatus.lastUpdate = new Date().toISOString()
          
          addTrainingLog(`Hardware status updated: ${hardwareStatus?.joints || 0} joints, ${hardwareStatus?.sensors || 0} sensors, ${hardwareStatus?.cameras || 0} cameras`, 'success')
        } else {
          addTrainingLog(`Failed to fetch hardware status: ${response.data.message}`, 'error')
        }
      } catch (error) {
        const errorMsg = error.response?.data?.detail || error.message || 'Unknown error'
        addTrainingLog(`Hardware status error: ${errorMsg}`, 'error')
        
        // No hardware data available - keep status as zero/empty
        hardwareStatus.joints = 0
        hardwareStatus.sensors = 0
        hardwareStatus.cameras = 0
        hardwareStatus.battery = 0
        hardwareStatus.systemTemperature = 0
        hardwareStatus.initialized = false
        hardwareStatus.lastUpdate = new Date().toISOString()
      }
    }
    
    // Test hardware connection
    const testHardwareConnection = async () => {
      try {
        addTrainingLog('Testing hardware connection...', 'info')
        const response = await apiClient.post('/api/robot/hardware/test_connection')
        
        if (response.data.status === 'success') {
          addTrainingLog('Hardware connection test successful', 'success')
          return true
        } else {
          addTrainingLog(`Hardware connection test failed: ${response.data.message}`, 'error')
          return false
        }
      } catch (error) {
        const errorMsg = error.response?.data?.detail || error.message || 'Unknown error'
        addTrainingLog(`Hardware connection test error: ${errorMsg}`, 'error')
        return false
      }
    }
    
    // Initialize hardware for training module
    const initializeHardwareForTraining = async () => {
      try {
        addTrainingLog('Initializing hardware for training...', 'info')
        const response = await apiClient.post('/api/robot/hardware/initialize')
        
        if (response.data.status === 'success') {
          hardwareStatus.initialized = true
          hardwareStatus.lastUpdate = new Date().toISOString()
          addTrainingLog('Hardware initialized successfully for training', 'success')
          
          // Refresh hardware status after initialization
          await refreshHardwareStatus()
          return true
        } else {
          addTrainingLog(`Hardware initialization failed for training: ${response.data.message}`, 'error')
          return false
        }
      } catch (error) {
        const errorMsg = error.response?.data?.detail || error.message || 'Unknown error'
        addTrainingLog(`Hardware initialization error for training: ${errorMsg}`, 'error')
        return false
      }
    }

    // Initialize training module
    const initializeTrainingModule = () => {
      addTrainingLog('Robot training module initialized', 'info')
      
      // Initialize hardware status on module load
      refreshHardwareStatus()
    }

    // ========== End Robot Training Module ==========

    // ========== Stereo Vision Spatial Recognition Data Definitions ==========
    
    // Stereo camera status
    const stereoCameraStatus = reactive({
      connected: false,
      enabled: false,
      calibrated: false,
      leftCamera: null,
      rightCamera: null,
      leftCameraStream: null,
      rightCameraStream: null
    })
    
    // Depth map data
    const depthMapData = ref(null)
    
    // Selected stereo mode
    const selectedStereoMode = ref('depth')
    
    // Spatial recognition results
    const spatialResults = ref(null)
    
    // ========== Robot Free Space Movement Control Data Definitions ==========
    
    // Movement state
    const movementState = reactive({
      status: 'idle',
      active: false,
      mode: 'manual',
      speed: 0,
      position: { x: 0, y: 0, z: 0 }
    })
    
    // Movement speed (0-100%)
    const movementSpeed = ref(50)
    
    // Waypoints for navigation
    const waypoints = ref([])
    
    // Waypoint input values
    const waypointX = ref(0)
    const waypointY = ref(0)
    const waypointZ = ref(0)
    
    // Selected navigation mode
    const selectedNavigationMode = ref('manual')
    
    // Spatial constraints for autonomous navigation
    const spatialConstraints = reactive({
      maxDistance: 5.0,
      minSafeDistance: 0.5,
      maxSpeed: 1.0,
      collisionThreshold: 0.3
    })
    
    // Navigation state
    const navigationState = reactive({
      active: false,
      mode: 'manual',
      progress: 0,
      currentWaypoint: null,
      obstacleCount: 0,
      distanceTraveled: 0,
      batteryUsage: 0
    })
    
    // Collision detection state
    const collisionDetectionState = reactive({
      enabled: true,
      dataSource: 'sensors', // 'sensors' or 'simulation'
      status: 'clear',
      lastDetectionTime: null,
      obstacles: [],
      nearestObstacle: null,
      warningLevel: 'none', // none, caution, warning, critical
      sensorCount: 0,
      lastProcessingTime: 0,
      fusionEnabled: true,
      sensorTypes: [] // List of available sensor types
    })
    
    // Collision detection timer reference
    const collisionDetectionTimer = ref(null)
    
    // ========== Stereo Vision Methods ==========
    
    // Detect available stereo cameras
    const detectStereoCameras = async () => {
      return await performDataLoad('stereo-cameras', {
        apiClient: apiClient,
        apiMethod: 'get',
        apiEndpoint: '/api/robot/stereo/pairs',
        dataPath: 'stereo_pairs',
        onBeforeStart: () => {
          debugInfo.value = 'Detecting stereo cameras...'
        },
        onSuccess: (pairs, fullResponse) => {
          if (fullResponse.data && fullResponse.data.status === 'success' && pairs && pairs.length > 0) {
            const pair = pairs[0]
            stereoCameraStatus.connected = true
            stereoCameraStatus.leftCamera = pair.left_camera_id
            stereoCameraStatus.rightCamera = pair.right_camera_id
            stereoCameraStatus.calibrated = pair.is_calibrated
            debugInfo.value = `Detected stereo camera pair: ${pair.left_camera_id} & ${pair.right_camera_id}`
          } else {
            debugInfo.value = 'No stereo camera pairs found'
          }
        },
        onError: (error) => {
          console.error('Failed to detect stereo cameras:', error)
          debugInfo.value = `Failed to detect stereo cameras: ${error.message}`
        },
        successMessage: '',
        errorMessage: 'Failed to detect stereo cameras',
        errorContext: 'Detect Stereo Cameras',
        showSuccess: false,
        showError: false,
        notify: null,
        handleError: null,
        fallbackValue: []
      })
    }
    
    // Calibrate stereo cameras
    const calibrateStereoCameras = async () => {
      if (!stereoCameraStatus.connected) return
      
      return await performDataOperation('calibrate-stereo', {
        apiClient: apiClient,
        apiMethod: 'post',
        apiEndpoint: '/api/robot/stereo/calibrate',
        requestData: {
          left_camera_id: stereoCameraStatus.leftCamera,
          right_camera_id: stereoCameraStatus.rightCamera
        },
        onBeforeStart: () => {
          debugInfo.value = 'Calibrating stereo cameras...'
        },
        onSuccess: (result, fullResponse) => {
          if (fullResponse.data && fullResponse.data.status === 'success') {
            stereoCameraStatus.calibrated = true
            debugInfo.value = 'Stereo cameras calibrated successfully'
          } else {
            debugInfo.value = 'Stereo camera calibration failed'
          }
        },
        onError: (error) => {
          console.error('Failed to calibrate stereo cameras:', error)
          debugInfo.value = `Failed to calibrate stereo cameras: ${error.message}`
        },
        successMessage: '',
        errorMessage: 'Failed to calibrate stereo cameras',
        errorContext: 'Calibrate Stereo Cameras',
        showSuccess: false,
        showError: false,
        notify: null,
        handleError: null
      })
    }
    
    // Enable stereo vision
    const enableStereoVision = async () => {
      if (!stereoCameraStatus.connected) return
      
      return await performDataOperation('enable-stereo', {
        apiClient: apiClient,
        apiMethod: 'post',
        apiEndpoint: '/api/robot/stereo/enable',
        requestData: {},
        onBeforeStart: () => {
          debugInfo.value = 'Enabling stereo vision...'
        },
        onSuccess: (result, fullResponse) => {
          if (fullResponse.data && fullResponse.data.status === 'success') {
            stereoCameraStatus.enabled = true
            debugInfo.value = 'Stereo vision enabled successfully'
          } else {
            debugInfo.value = 'Failed to enable stereo vision'
          }
        },
        onError: (error) => {
          console.error('Failed to enable stereo vision:', error)
          debugInfo.value = `Failed to enable stereo vision: ${error.message}`
        },
        successMessage: '',
        errorMessage: 'Failed to enable stereo vision',
        errorContext: 'Enable Stereo Vision',
        showSuccess: false,
        showError: false,
        notify: null,
        handleError: null
      })
    }
    
    // Disable stereo vision
    const disableStereoVision = async () => {
      return await performDataOperation('disable-stereo', {
        apiClient: apiClient,
        apiMethod: 'post',
        apiEndpoint: '/api/robot/stereo/disable',
        requestData: {},
        onBeforeStart: () => {
          debugInfo.value = 'Disabling stereo vision...'
        },
        onSuccess: (result, fullResponse) => {
          if (fullResponse.data && fullResponse.data.status === 'success') {
            stereoCameraStatus.enabled = false
            debugInfo.value = 'Stereo vision disabled successfully'
          } else {
            debugInfo.value = 'Failed to disable stereo vision'
          }
        },
        onError: (error) => {
          console.error('Failed to disable stereo vision:', error)
          debugInfo.value = `Failed to disable stereo vision: ${error.message}`
        },
        successMessage: '',
        errorMessage: 'Failed to disable stereo vision',
        errorContext: 'Disable Stereo Vision',
        showSuccess: false,
        showError: false,
        notify: null,
        handleError: null
      })
    }
    
    // Generate depth map
    const generateDepthMap = async () => {
      if (!stereoCameraStatus.connected) return
      
      return await performDataOperation('generate-depth-map', {
        apiClient: apiClient,
        apiMethod: 'post',
        apiEndpoint: '/api/robot/spatial/depth',
        requestData: {
          cameras: [stereoCameraStatus.leftCamera, stereoCameraStatus.rightCamera],
          method: 'stereo',
          parameters: {
            min_disparity: 0,
            num_disparities: 64,
            block_size: 15
          }
        },
        onBeforeStart: () => {
          debugInfo.value = 'Generating depth map...'
        },
        onSuccess: (result, fullResponse) => {
          if (fullResponse.data && fullResponse.data.status === 'success') {
            depthMapData.value = fullResponse.data.result || fullResponse.data.data
            // Parse spatial results
            if (depthMapData.value && depthMapData.value.metrics) {
              spatialResults.value = depthMapData.value.metrics
            }
            debugInfo.value = 'Depth map generated successfully'
          } else {
            debugInfo.value = 'Depth map generation failed'
          }
        },
        onError: (error) => {
          console.error('Failed to generate depth map:', error)
          debugInfo.value = `Failed to generate depth map: ${error.message}`
        },
        successMessage: '',
        errorMessage: 'Failed to generate depth map',
        errorContext: 'Generate Depth Map',
        showSuccess: false,
        showError: false,
        notify: null,
        handleError: null
      })
    }
    
    // Start spatial mapping
    const startSpatialMapping = async () => {
      if (!stereoCameraStatus.connected) return
      
      return await performDataOperation('start-spatial-mapping', {
        apiClient: apiClient,
        apiMethod: 'post',
        apiEndpoint: '/api/robot/spatial/map',
        requestData: {
          operation: 'update',
          area: null
        },
        onBeforeStart: () => {
          debugInfo.value = 'Starting spatial mapping...'
        },
        onSuccess: (result, fullResponse) => {
          if (fullResponse.data && fullResponse.data.status === 'success') {
            debugInfo.value = 'Spatial mapping started successfully'
          } else {
            debugInfo.value = 'Spatial mapping failed to start'
          }
        },
        onError: (error) => {
          console.error('Failed to start spatial mapping:', error)
          debugInfo.value = `Failed to start spatial mapping: ${error.message}`
        },
        successMessage: '',
        errorMessage: 'Failed to start spatial mapping',
        errorContext: 'Start Spatial Mapping',
        showSuccess: false,
        showError: false,
        notify: null,
        handleError: null
      })
    }
    
    // Export spatial data
    const exportSpatialData = () => {
      if (!depthMapData.value) return
      
      try {
        const data = {
          depthMap: depthMapData.value,
          spatialResults: spatialResults.value,
          stereoCameraStatus: { ...stereoCameraStatus },
          timestamp: new Date().toISOString()
        }
        
        const dataStr = JSON.stringify(data, null, 2)
        const dataBlob = new Blob([dataStr], { type: 'application/json' })
        const url = URL.createObjectURL(dataBlob)
        const link = document.createElement('a')
        link.href = url
        link.download = `spatial-data-${new Date().getTime()}.json`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        URL.revokeObjectURL(url)
        
        debugInfo.value = 'Spatial data exported successfully'
      } catch (error) {
        console.error('Failed to export spatial data:', error)
        debugInfo.value = `Failed to export spatial data: ${error.message}`
      }
    }

    // ========== Robot Free Space Movement Control Methods ==========
    
    // Move robot in specified direction
    const moveRobot = async (direction) => {
      try {
        debugInfo.value = `Moving robot ${direction}...`
        movementState.active = true
        movementState.mode = 'manual'
        movementState.status = `moving_${direction}`
        
        const response = await apiClient.post('/api/robot/motion/execute', {
          command: 'move',
          direction: direction,
          speed: movementSpeed.value / 100, // Convert to 0-1 range
          duration: 1.0 // seconds
        })
        
        if (response.data && response.data.status === 'success') {
          debugInfo.value = `Robot moving ${direction} successfully`
          movementState.speed = movementSpeed.value
          // Update collision detection after movement
          await updateCollisionDetection()
        } else {
          debugInfo.value = `Failed to move robot: ${response.data?.message || 'Unknown error'}`
          movementState.active = false
          movementState.status = 'idle'
        }
      } catch (error) {
        console.error('Failed to move robot:', error)
        debugInfo.value = `Failed to move robot: ${error.message}`
        movementState.active = false
        movementState.status = 'idle'
      }
    }
    
    // Stop robot movement
    const stopRobot = async () => {
      try {
        debugInfo.value = 'Stopping robot...'
        const response = await apiClient.post('/api/robot/emergency/stop')
        
        if (response.data && response.data.status === 'success') {
          debugInfo.value = 'Robot stopped successfully'
          movementState.active = false
          movementState.status = 'idle'
          movementState.speed = 0
        } else {
          debugInfo.value = `Failed to stop robot: ${response.data?.message || 'Unknown error'}`
        }
      } catch (error) {
        console.error('Failed to stop robot:', error)
        debugInfo.value = `Failed to stop robot: ${error.message}`
      }
    }
    
    // Set movement speed
    const setMovementSpeed = () => {
      debugInfo.value = `Movement speed set to ${movementSpeed.value}%`
      movementState.speed = movementSpeed.value
    }
    
    // Rotate robot
    const rotateRobot = async (direction) => {
      try {
        debugInfo.value = `Rotating robot ${direction}...`
        movementState.active = true
        movementState.mode = 'manual'
        movementState.status = `rotating_${direction}`
        
        const response = await apiClient.post('/api/robot/motion/execute', {
          command: 'rotate',
          direction: direction,
          angle: direction === 'reset' ? 0 : 45, // 45 degrees for left/right, 0 for reset
          speed: movementSpeed.value / 100
        })
        
        if (response.data && response.data.status === 'success') {
          debugInfo.value = `Robot rotated ${direction} successfully`
          // Update collision detection after rotation
          await updateCollisionDetection()
        } else {
          debugInfo.value = `Failed to rotate robot: ${response.data?.message || 'Unknown error'}`
          movementState.active = false
          movementState.status = 'idle'
        }
      } catch (error) {
        console.error('Failed to rotate robot:', error)
        debugInfo.value = `Failed to rotate robot: ${error.message}`
        movementState.active = false
        movementState.status = 'idle'
      }
    }
    
    // Add waypoint
    const addWaypoint = () => {
      if (waypointX.value === 0 && waypointY.value === 0 && waypointZ.value === 0) {
        debugInfo.value = 'Please enter valid waypoint coordinates'
        return
      }
      
      const waypoint = {
        x: waypointX.value,
        y: waypointY.value,
        z: waypointZ.value
      }
      
      waypoints.value.push(waypoint)
      debugInfo.value = `Waypoint added: (${waypoint.x.toFixed(2)}, ${waypoint.y.toFixed(2)}, ${waypoint.z.toFixed(2)})`
      
      // Reset input values
      waypointX.value = 0
      waypointY.value = 0
      waypointZ.value = 0
    }
    
    // Remove waypoint
    const removeWaypoint = (index) => {
      if (index >= 0 && index < waypoints.value.length) {
        const removed = waypoints.value.splice(index, 1)[0]
        debugInfo.value = `Waypoint removed: (${removed.x.toFixed(2)}, ${removed.y.toFixed(2)}, ${removed.z.toFixed(2)})`
      }
    }
    
    // Clear all waypoints
    const clearWaypoints = () => {
      waypoints.value = []
      debugInfo.value = 'All waypoints cleared'
    }
    
    // ========== Path Planning Algorithms ==========
    
    // Path planning state
    const pathPlanningState = reactive({
      optimizedPath: [],
      planningAlgorithm: 'tsp', // tsp, nearest_neighbor, a_star, dijkstra
      totalDistance: 0,
      planningTime: 0,
      obstaclesConsidered: false
    })
    
    // Calculate Euclidean distance between two points
    const calculateDistance = (point1, point2) => {
      const dx = point2.x - point1.x
      const dy = point2.y - point1.y
      const dz = point2.z - point1.z
      return Math.sqrt(dx * dx + dy * dy + dz * dz)
    }
    
    // Nearest neighbor algorithm for path optimization (Traveling Salesman Problem)
    const optimizePathNearestNeighbor = (points, startPoint = null) => {
      if (points.length === 0) return []
      
      const unvisited = [...points]
      const optimized = []
      
      // Start from startPoint or first point
      let current = startPoint || unvisited.shift()
      optimized.push(current)
      
      while (unvisited.length > 0) {
        // Find nearest unvisited point
        let nearestIndex = 0
        let nearestDistance = calculateDistance(current, unvisited[0])
        
        for (let i = 1; i < unvisited.length; i++) {
          const distance = calculateDistance(current, unvisited[i])
          if (distance < nearestDistance) {
            nearestDistance = distance
            nearestIndex = i
          }
        }
        
        // Add nearest point to optimized path
        current = unvisited.splice(nearestIndex, 1)[0]
        optimized.push(current)
      }
      
      return optimized
    }
    
    // 2-opt optimization algorithm for TSP (improves existing path)
    const optimizePath2Opt = (path) => {
      if (path.length < 4) return [...path] // Too short to optimize
      
      let improved = true
      let bestPath = [...path]
      let bestDistance = calculateTotalDistance(bestPath)
      
      while (improved) {
        improved = false
        
        for (let i = 0; i < path.length - 3; i++) {
          for (let j = i + 2; j < path.length - 1; j++) {
            // Try reversing segment i+1 to j
            const newPath = [...bestPath]
            const segment = newPath.slice(i + 1, j + 1)
            segment.reverse()
            
            // Replace the segment with reversed version
            for (let k = i + 1; k <= j; k++) {
              newPath[k] = segment[k - i - 1]
            }
            
            const newDistance = calculateTotalDistance(newPath)
            
            if (newDistance < bestDistance) {
              bestPath = newPath
              bestDistance = newDistance
              improved = true
            }
          }
        }
      }
      
      return bestPath
    }
    
    // Calculate total distance of a path
    const calculateTotalDistance = (path) => {
      if (path.length < 2) return 0
      
      let total = 0
      for (let i = 1; i < path.length; i++) {
        total += calculateDistance(path[i - 1], path[i])
      }
      return total
    }
    
    // Check if a line segment intersects with any obstacle
    const checkLineSegmentObstacleCollision = (point1, point2, obstacles, safetyMargin = 0.5) => {
      for (const obstacle of obstacles) {
        // Simple circle obstacle collision check
        const obstaclePos = { x: obstacle.x || 0, y: obstacle.y || 0, z: obstacle.z || 0 }
        const obstacleRadius = obstacle.radius || 0.3
        
        // Check if line segment is too close to obstacle
        const distance = distancePointToLineSegment(point1, point2, obstaclePos)
        if (distance < obstacleRadius + safetyMargin) {
          return true // Collision detected
        }
      }
      return false // No collision
    }
    
    // Calculate distance from point to line segment
    const distancePointToLineSegment = (lineStart, lineEnd, point) => {
      const lineVec = { 
        x: lineEnd.x - lineStart.x, 
        y: lineEnd.y - lineStart.y, 
        z: lineEnd.z - lineStart.z 
      }
      const pointVec = { 
        x: point.x - lineStart.x, 
        y: point.y - lineStart.y, 
        z: point.z - lineStart.z 
      }
      
      const lineLengthSquared = lineVec.x * lineVec.x + lineVec.y * lineVec.y + lineVec.z * lineVec.z
      
      if (lineLengthSquared === 0) {
        // Line segment is actually a point
        return Math.sqrt(pointVec.x * pointVec.x + pointVec.y * pointVec.y + pointVec.z * pointVec.z)
      }
      
      // Calculate projection of pointVec onto lineVec
      const t = Math.max(0, Math.min(1, 
        (pointVec.x * lineVec.x + pointVec.y * lineVec.y + pointVec.z * lineVec.z) / lineLengthSquared
      ))
      
      // Calculate closest point on line segment
      const closestPoint = {
        x: lineStart.x + t * lineVec.x,
        y: lineStart.y + t * lineVec.y,
        z: lineStart.z + t * lineVec.z
      }
      
      // Calculate distance from point to closest point
      const dx = point.x - closestPoint.x
      const dy = point.y - closestPoint.y
      const dz = point.z - closestPoint.z
      
      return Math.sqrt(dx * dx + dy * dy + dz * dz)
    }
    
    // Generate obstacle avoidance waypoints
    const generateObstacleAvoidanceWaypoints = (startPoint, endPoint, obstacles) => {
      if (obstacles.length === 0) {
        return [startPoint, endPoint] // No obstacles, direct path
      }
      
      const pathPoints = []
      
      // Check if direct path is collision-free
      const directCollision = checkLineSegmentObstacleCollision(startPoint, endPoint, obstacles)
      
      if (!directCollision) {
        pathPoints.push(startPoint, endPoint)
        return pathPoints
      }
      
      // Find obstacles between start and end
      const blockingObstacles = obstacles.filter(obstacle => {
        const obstaclePos = { x: obstacle.x || 0, y: obstacle.y || 0, z: obstacle.z || 0 }
        return distancePointToLineSegment(startPoint, endPoint, obstaclePos) < (obstacle.radius || 0.3) + 0.5
      })
      
      if (blockingObstacles.length === 0) {
        pathPoints.push(startPoint, endPoint)
        return pathPoints
      }
      
      // Sort obstacles by distance from start point
      blockingObstacles.sort((a, b) => {
        const distA = calculateDistance(startPoint, { x: a.x || 0, y: a.y || 0, z: a.z || 0 })
        const distB = calculateDistance(startPoint, { x: b.x || 0, y: b.y || 0, z: b.z || 0 })
        return distA - distB
      })
      
      // Generate path that goes around obstacles
      let currentPoint = startPoint
      
      for (const obstacle of blockingObstacles) {
        const obstaclePos = { x: obstacle.x || 0, y: obstacle.y || 0, z: obstacle.z || 0 }
        
        // Calculate direction vector from start to end
        const direction = {
          x: endPoint.x - startPoint.x,
          y: endPoint.y - startPoint.y,
          z: endPoint.z - startPoint.z
        }
        
        // Calculate perpendicular direction (rotate 90 degrees in XY plane)
        const perpendicular = {
          x: -direction.y,
          y: direction.x,
          z: 0
        }
        
        // Normalize perpendicular vector
        const length = Math.sqrt(perpendicular.x * perpendicular.x + perpendicular.y * perpendicular.y)
        if (length > 0) {
          perpendicular.x /= length
          perpendicular.y /= length
        }
        
        // Create waypoint around obstacle (left side)
        const avoidDistance = (obstacle.radius || 0.3) + 1.0 // Safety margin
        const waypoint = {
          x: obstaclePos.x + perpendicular.x * avoidDistance,
          y: obstaclePos.y + perpendicular.y * avoidDistance,
          z: currentPoint.z // Keep same height
        }
        
        pathPoints.push(currentPoint, waypoint)
        currentPoint = waypoint
      }
      
      // Add final destination
      pathPoints.push(endPoint)
      
      return pathPoints
    }
    
    // Optimize waypoint path considering obstacles
    const optimizeWaypointPath = (waypoints, considerObstacles = true) => {
      const startTime = performance.now()
      
      if (waypoints.length === 0) {
        pathPlanningState.optimizedPath = []
        pathPlanningState.totalDistance = 0
        pathPlanningState.planningTime = 0
        pathPlanningState.obstaclesConsidered = false
        return []
      }
      
      let optimizedPath = []
      
      // Get obstacles for collision avoidance
      const obstacles = considerObstacles ? collisionDetectionState.obstacles : []
      
      if (waypoints.length === 1) {
        optimizedPath = [...waypoints]
      } else if (waypoints.length === 2) {
        // For 2 waypoints, just check for obstacles
        if (considerObstacles && obstacles.length > 0) {
          optimizedPath = generateObstacleAvoidanceWaypoints(waypoints[0], waypoints[1], obstacles)
        } else {
          optimizedPath = [...waypoints]
        }
      } else {
        // For multiple waypoints, use TSP optimization
        optimizedPath = optimizePathNearestNeighbor(waypoints)
        
        // Apply 2-opt optimization for better results
        optimizedPath = optimizePath2Opt(optimizedPath)
        
        // If obstacles need to be considered, insert avoidance waypoints
        if (considerObstacles && obstacles.length > 0) {
          const obstacleAwarePath = []
          
          for (let i = 0; i < optimizedPath.length - 1; i++) {
            const segmentPoints = generateObstacleAvoidanceWaypoints(
              optimizedPath[i], 
              optimizedPath[i + 1], 
              obstacles
            )
            
            // Add segment points (skip first point if it's the same as previous endpoint)
            if (i > 0 && segmentPoints.length > 1) {
              obstacleAwarePath.push(...segmentPoints.slice(1))
            } else {
              obstacleAwarePath.push(...segmentPoints)
            }
          }
          
          optimizedPath = obstacleAwarePath
        }
      }
      
      // Calculate statistics
      const endTime = performance.now()
      pathPlanningState.optimizedPath = optimizedPath
      pathPlanningState.totalDistance = calculateTotalDistance(optimizedPath)
      pathPlanningState.planningTime = endTime - startTime
      pathPlanningState.obstaclesConsidered = considerObstacles && obstacles.length > 0
      
      debugInfo.value = `Path optimized: ${optimizedPath.length} waypoints, ${pathPlanningState.totalDistance.toFixed(2)}m total distance, ${pathPlanningState.planningTime.toFixed(1)}ms planning time`
      
      return optimizedPath
    }
    
    // Visualize optimized path on trajectory canvas
    const visualizeOptimizedPath = () => {
      if (!trajectoryCanvas.value || pathPlanningState.optimizedPath.length === 0) {
        return
      }
      
      const ctx = trajectoryCanvas.value.getContext('2d')
      const width = trajectoryCanvas.value.width
      const height = trajectoryCanvas.value.height
      
      // Find min/max coordinates for scaling
      const allPoints = [...pathPlanningState.optimizedPath, ...movementHistory.value.map(h => h.position)]
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
      
      for (const point of allPoints) {
        minX = Math.min(minX, point.x)
        maxX = Math.max(maxX, point.x)
        minY = Math.min(minY, point.y)
        maxY = Math.max(maxY, point.y)
      }
      
      // Add padding
      const padding = 2.0
      minX -= padding
      maxX += padding
      minY -= padding
      maxY += padding
      
      // Calculate scale
      const scaleX = width / (maxX - minX)
      const scaleY = height / (maxY - minY)
      const scale = Math.min(scaleX, scaleY) * 0.9
      
      // Function to convert world coordinates to canvas coordinates
      const worldToCanvas = (x, y) => {
        const canvasX = (x - minX) * scale + (width - (maxX - minX) * scale) / 2
        const canvasY = height - ((y - minY) * scale + (height - (maxY - minY) * scale) / 2)
        return { x: canvasX, y: canvasY }
      }
      
      // Draw optimized path
      ctx.strokeStyle = '#00aa00'
      ctx.lineWidth = 3
      ctx.setLineDash([5, 5]) // Dashed line for planned path
      ctx.beginPath()
      
      for (let i = 0; i < pathPlanningState.optimizedPath.length; i++) {
        const point = pathPlanningState.optimizedPath[i]
        const canvasPos = worldToCanvas(point.x, point.y)
        
        if (i === 0) {
          ctx.moveTo(canvasPos.x, canvasPos.y)
        } else {
          ctx.lineTo(canvasPos.x, canvasPos.y)
        }
      }
      
      ctx.stroke()
      ctx.setLineDash([]) // Reset line dash
      
      // Draw waypoint markers
      for (let i = 0; i < pathPlanningState.optimizedPath.length; i++) {
        const point = pathPlanningState.optimizedPath[i]
        const canvasPos = worldToCanvas(point.x, point.y)
        
        // Draw waypoint circle
        ctx.fillStyle = i === 0 ? '#00ff00' : (i === pathPlanningState.optimizedPath.length - 1 ? '#ff0000' : '#0088ff')
        ctx.beginPath()
        ctx.arc(canvasPos.x, canvasPos.y, 6, 0, Math.PI * 2)
        ctx.fill()
        
        // Draw waypoint number
        ctx.fillStyle = '#ffffff'
        ctx.font = '10px Arial'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText((i + 1).toString(), canvasPos.x, canvasPos.y)
      }
      
      debugInfo.value = `Optimized path visualized on trajectory canvas (${pathPlanningState.optimizedPath.length} waypoints)`
    }
    
    // Clear optimized path visualization
    const clearOptimizedPath = () => {
      pathPlanningState.optimizedPath = []
      pathPlanningState.totalDistance = 0
      pathPlanningState.planningTime = 0
      pathPlanningState.obstaclesConsidered = false
      
      // Redraw trajectory canvas to remove path visualization
      if (trajectoryCanvas.value) {
        const ctx = trajectoryCanvas.value.getContext('2d')
        ctx.clearRect(0, 0, trajectoryCanvas.value.width, trajectoryCanvas.value.height)
        // Trigger redraw of trajectory
        if (movementHistory.value.length > 0) {
          drawTrajectory()
        }
      }
      
      debugInfo.value = 'Optimized path cleared'
    }
    
    // Execute waypoint navigation
    const executeWaypointNavigation = async () => {
      if (waypoints.value.length === 0) {
        debugInfo.value = 'No waypoints defined'
        return
      }
      
      try {
        debugInfo.value = 'Starting waypoint navigation...'
        navigationState.active = true
        navigationState.mode = 'waypoint'
        navigationState.progress = 0
        navigationState.currentWaypoint = 1
        movementState.active = true
        movementState.mode = 'autonomous'
        movementState.status = 'navigating'
        
        // Optimize path if enabled
        let navigationWaypoints = waypoints.value
        if (pathPlanningState.optimizedPath.length > 0) {
          debugInfo.value = `Using optimized path with ${pathPlanningState.optimizedPath.length} waypoints (total distance: ${pathPlanningState.totalDistance.toFixed(2)}m)`
          navigationWaypoints = pathPlanningState.optimizedPath
        } else {
          debugInfo.value = 'Using original waypoints (no optimization applied)'
        }
        
        const response = await apiClient.post('/api/robot/task/plan', {
          task_type: 'waypoint_navigation',
          waypoints: navigationWaypoints,
          constraints: spatialConstraints,
          path_optimization: pathPlanningState.optimizedPath.length > 0 ? {
            algorithm: pathPlanningState.planningAlgorithm,
            total_distance: pathPlanningState.totalDistance,
            planning_time: pathPlanningState.planningTime,
            obstacles_considered: pathPlanningState.obstaclesConsidered
          } : null
        })
        
        if (response.data && response.data.status === 'success') {
          debugInfo.value = 'Waypoint navigation started successfully'
          
          // Start task execution
          await apiClient.post('/api/robot/task/execute', {
            task_id: response.data.task_id
          })
          
          debugInfo.value = 'Waypoint navigation execution started'
        } else {
          debugInfo.value = `Failed to start waypoint navigation: ${response.data?.message || 'Unknown error'}`
          navigationState.active = false
          movementState.active = false
          movementState.status = 'idle'
        }
      } catch (error) {
        console.error('Failed to execute waypoint navigation:', error)
        debugInfo.value = `Failed to execute waypoint navigation: ${error.message}`
        navigationState.active = false
        movementState.active = false
        movementState.status = 'idle'
      }
    }
    
    // Start autonomous navigation
    const startAutonomousNavigation = async () => {
      try {
        debugInfo.value = 'Starting autonomous navigation...'
        navigationState.active = true
        navigationState.mode = selectedNavigationMode.value
        navigationState.progress = 0
        movementState.active = true
        movementState.mode = 'autonomous'
        movementState.status = 'navigating'
        
        const response = await apiClient.post('/api/robot/task/plan', {
          task_type: selectedNavigationMode.value,
          constraints: spatialConstraints
        })
        
        if (response.data && response.data.status === 'success') {
          debugInfo.value = 'Autonomous navigation started successfully'
          
          // Start task execution
          await apiClient.post('/api/robot/task/execute', {
            task_id: response.data.task_id
          })
          
          debugInfo.value = 'Autonomous navigation execution started'
          
          // Update collision detection for autonomous navigation
          await updateCollisionDetection()
        } else {
          debugInfo.value = `Failed to start autonomous navigation: ${response.data?.message || 'Unknown error'}`
          navigationState.active = false
          movementState.active = false
          movementState.status = 'idle'
        }
      } catch (error) {
        console.error('Failed to start autonomous navigation:', error)
        debugInfo.value = `Failed to start autonomous navigation: ${error.message}`
        navigationState.active = false
        movementState.active = false
        movementState.status = 'idle'
      }
    }
    
    // Pause autonomous navigation
    const pauseAutonomousNavigation = async () => {
      try {
        debugInfo.value = 'Pausing autonomous navigation...'
        const response = await apiClient.post('/api/robot/task/stop', {
          task_id: 'current' // This would need the actual task ID
        })
        
        if (response.data && response.data.status === 'success') {
          debugInfo.value = 'Autonomous navigation paused'
          navigationState.active = false
          movementState.active = false
          movementState.status = 'paused'
        } else {
          debugInfo.value = `Failed to pause navigation: ${response.data?.message || 'Unknown error'}`
        }
      } catch (error) {
        console.error('Failed to pause autonomous navigation:', error)
        debugInfo.value = `Failed to pause autonomous navigation: ${error.message}`
      }
    }
    
    // Stop autonomous navigation
    const stopAutonomousNavigation = async () => {
      try {
        debugInfo.value = 'Stopping autonomous navigation...'
        const response = await apiClient.post('/api/robot/task/stop', {
          task_id: 'current'
        })
        
        if (response.data && response.data.status === 'success') {
          debugInfo.value = 'Autonomous navigation stopped'
          navigationState.active = false
          movementState.active = false
          movementState.status = 'idle'
          navigationState.progress = 0
        } else {
          debugInfo.value = `Failed to stop navigation: ${response.data?.message || 'Unknown error'}`
        }
      } catch (error) {
        console.error('Failed to stop autonomous navigation:', error)
        debugInfo.value = `Failed to stop autonomous navigation: ${error.message}`
      }
    }
    
    // Calibrate navigation system
    const calibrateNavigation = async () => {
      try {
        debugInfo.value = 'Calibrating navigation system...'
        const response = await apiClient.post('/api/robot/system/calibrate')
        
        if (response.data && response.data.status === 'success') {
          debugInfo.value = 'Navigation system calibrated successfully'
        } else {
          debugInfo.value = `Failed to calibrate navigation: ${response.data?.message || 'Unknown error'}`
        }
      } catch (error) {
        console.error('Failed to calibrate navigation:', error)
        debugInfo.value = `Failed to calibrate navigation: ${error.message}`
      }
    }
    
    // ========== Collision Detection Methods ==========
    
    // Sensor data types for collision detection
    const sensorDataTypes = {
      LIDAR: 'lidar',
      ULTRASONIC: 'ultrasonic',
      INFRARED: 'infrared',
      DEPTH_CAMERA: 'depth_camera',
      BUMPER: 'bumper'
    }
    
    // ========== Sensor Calibration Methods ==========
    
    // Start sensor calibration process
    const startSensorCalibration = async () => {
      if (sensorCalibrationState.calibrating) {
        return
      }
      
      try {
        // Reset calibration state
        sensorCalibrationState.calibrating = true
        sensorCalibrationState.calibrationProgress = 0
        sensorCalibrationState.calibrationStatus = 'calibrating'
        sensorCalibrationState.calibrationError = null
        sensorCalibrationState.calibrationResults = {}
        
        debugInfo.value = 'Starting sensor calibration...'
        
        // Simulate calibration process (in real implementation, this would call backend API)
        // For now, we'll simulate a calibration process with progress updates
        const calibrationSteps = [
          'Initializing sensors...',
          'Collecting reference data...',
          'Calculating offsets...',
          'Applying calibration...',
          'Validating results...'
        ]
        
        for (let i = 0; i < calibrationSteps.length; i++) {
          if (!sensorCalibrationState.calibrating) {
            // Calibration was stopped
            debugInfo.value = 'Sensor calibration stopped'
            return
          }
          
          debugInfo.value = calibrationSteps[i]
          sensorCalibrationState.calibrationProgress = Math.round(((i + 1) / calibrationSteps.length) * 100)
          
          // Simulate processing time
          await new Promise(resolve => setTimeout(resolve, 1000))
        }
        
        // Simulate calibration results
        sensorCalibrationState.calibrationResults = {
          lidar: {
            offset: 0.012,
            scale: 1.001,
            quality: 0.95
          },
          ultrasonic: {
            offset: -0.005,
            scale: 0.998,
            quality: 0.88
          },
          infrared: {
            offset: 0.008,
            scale: 1.003,
            quality: 0.92
          }
        }
        
        sensorCalibrationState.calibrating = false
        sensorCalibrationState.calibrationProgress = 100
        sensorCalibrationState.calibrationStatus = 'calibrated'
        sensorCalibrationState.calibrationTimestamp = new Date().toISOString()
        
        debugInfo.value = 'Sensor calibration completed successfully'
        
      } catch (error) {
        console.error('Failed to calibrate sensors:', error)
        sensorCalibrationState.calibrating = false
        sensorCalibrationState.calibrationStatus = 'error'
        sensorCalibrationState.calibrationError = error.message
        debugInfo.value = `Sensor calibration failed: ${error.message}`
      }
    }
    
    // Stop sensor calibration
    const stopSensorCalibration = () => {
      if (!sensorCalibrationState.calibrating) {
        return
      }
      
      sensorCalibrationState.calibrating = false
      sensorCalibrationState.calibrationStatus = 'stopped'
      debugInfo.value = 'Sensor calibration stopped'
    }
    
    // Reset sensor calibration
    const resetSensorCalibration = () => {
      sensorCalibrationState.calibrating = false
      sensorCalibrationState.calibrationProgress = 0
      sensorCalibrationState.calibrationStatus = 'idle'
      sensorCalibrationState.calibrationResults = {}
      sensorCalibrationState.calibrationError = null
      sensorCalibrationState.calibrationTimestamp = null
      sensorCalibrationState.selectedSensorTypes = []
      
      debugInfo.value = 'Sensor calibration reset'
    }
    
    // Fetch sensor data from backend API
    const fetchSensorData = async () => {
      if (!collisionDetectionState.enabled) {
        return []
      }
      
      try {
        debugInfo.value = 'Fetching sensor data...'
        const response = await apiClient.get('/api/robot/sensors/data')
        
        if (response.data && response.data.status === 'success') {
          const sensorData = response.data.sensor_data || []
          debugInfo.value = `Received ${sensorData.length} sensor readings`
          return sensorData
        } else {
          debugInfo.value = 'Failed to fetch sensor data from API'
          return []
        }
      } catch (error) {
        console.error('Failed to fetch sensor data:', error)
        debugInfo.value = `Sensor data fetch error: ${error.message}`
        return []
      }
    }
    
    // Process raw sensor data into obstacle information
    const processSensorDataToObstacles = (sensorData) => {
      if (!sensorData || sensorData.length === 0) {
        return []
      }
      
      const obstacles = []
      const now = new Date()
      
      // Process each sensor reading
      for (const reading of sensorData) {
        const sensorType = reading.sensor_type || 'unknown'
        const value = reading.value || 0
        const unit = reading.unit || 'm'
        const confidence = reading.confidence || 0.5
        const sensorId = reading.sensor_id || 'unknown'
        const timestamp = reading.timestamp || now.toISOString()
        const direction = reading.direction || 0 // angle in radians
        
        // Skip invalid readings
        if (value <= 0 || !isFinite(value)) {
          continue
        }
        
        // Convert value to meters if needed
        let distanceMeters = value
        if (unit === 'cm') {
          distanceMeters = value / 100
        } else if (unit === 'mm') {
          distanceMeters = value / 1000
        }
        
        // Calculate obstacle position relative to robot
        // Assuming robot is at origin (0, 0, 0)
        const x = Math.cos(direction) * distanceMeters
        const y = Math.sin(direction) * distanceMeters
        const z = 0 // Assume ground plane for now
        
        // Adjust confidence based on sensor type
        let adjustedConfidence = confidence
        switch (sensorType) {
          case sensorDataTypes.LIDAR:
            adjustedConfidence = confidence * 0.95
            break
          case sensorDataTypes.DEPTH_CAMERA:
            adjustedConfidence = confidence * 0.90
            break
          case sensorDataTypes.ULTRASONIC:
            adjustedConfidence = confidence * 0.85
            break
          case sensorDataTypes.INFRARED:
            adjustedConfidence = confidence * 0.80
            break
          case sensorDataTypes.BUMPER:
            adjustedConfidence = 1.0 // Direct contact
            break
          default:
            adjustedConfidence = confidence * 0.70
        }
        
        // Create obstacle object
        obstacles.push({
          id: `obstacle_${sensorId}_${timestamp}`,
          distance: distanceMeters,
          angle: direction,
          x,
          y,
          z,
          confidence: adjustedConfidence,
          sensor_type: sensorType,
          sensor_id: sensorId,
          timestamp,
          raw_value: value,
          raw_unit: unit
        })
      }
      
      return obstacles
    }
    
    // Sensor data fusion - combine multiple sensor readings for the same obstacle
    const fuseSensorData = (obstacles) => {
      if (obstacles.length <= 1) {
        return obstacles
      }
      
      const fusedObstacles = []
      const fusionThreshold = 0.3 // meters - obstacles within this distance are considered the same
      
      // Group obstacles by proximity
      const groups = []
      
      for (const obstacle of obstacles) {
        let assignedToGroup = false
        
        for (const group of groups) {
          // Check if obstacle is close to any obstacle in the group
          for (const groupObstacle of group) {
            const dx = obstacle.x - groupObstacle.x
            const dy = obstacle.y - groupObstacle.y
            const distance = Math.sqrt(dx * dx + dy * dy)
            
            if (distance < fusionThreshold) {
              group.push(obstacle)
              assignedToGroup = true
              break
            }
          }
          
          if (assignedToGroup) {
            break
          }
        }
        
        if (!assignedToGroup) {
          groups.push([obstacle])
        }
      }
      
      // Fuse each group into a single obstacle
      for (const group of groups) {
        if (group.length === 1) {
          fusedObstacles.push(group[0])
          continue
        }
        
        // Calculate weighted average based on confidence
        let totalWeight = 0
        let weightedX = 0
        let weightedY = 0
        let weightedDistance = 0
        let weightedAngle = 0
        let maxConfidence = 0
        let sensorTypes = new Set()
        
        for (const obstacle of group) {
          const weight = obstacle.confidence
          totalWeight += weight
          weightedX += obstacle.x * weight
          weightedY += obstacle.y * weight
          weightedDistance += obstacle.distance * weight
          weightedAngle += obstacle.angle * weight
          maxConfidence = Math.max(maxConfidence, obstacle.confidence)
          sensorTypes.add(obstacle.sensor_type)
        }
        
        if (totalWeight > 0) {
          fusedObstacles.push({
            id: `fused_${group[0].id}`,
            distance: weightedDistance / totalWeight,
            angle: weightedAngle / totalWeight,
            x: weightedX / totalWeight,
            y: weightedY / totalWeight,
            z: 0,
            confidence: maxConfidence, // Use highest confidence
            sensor_type: Array.from(sensorTypes).join(','),
            sensor_id: 'fused',
            timestamp: new Date().toISOString(),
            fused_count: group.length,
            raw_value: null,
            raw_unit: null
          })
        }
      }
      
      return fusedObstacles
    }
    
    // Update collision detection with real sensor data
    const updateCollisionDetectionWithSensors = async () => {
      if (!collisionDetectionState.enabled) {
        collisionDetectionState.status = 'disabled'
        collisionDetectionState.warningLevel = 'none'
        collisionDetectionState.obstacles = []
        collisionDetectionState.lastDetectionTime = null
        collisionDetectionState.nearestObstacle = null
        debugInfo.value = 'Collision detection disabled'
        return
      }
      
      const startTime = performance.now()
      const now = new Date()
      
      try {
        // Fetch sensor data from backend
        const rawSensorData = await fetchSensorData()
        
        if (rawSensorData.length === 0) {
          // No sensor data available, fall back to simulation
          debugInfo.value = 'No sensor data available, using simulation'
          updateCollisionDetectionSimulation() // Call simulation function
          return
        }
        
        // Process sensor data into obstacles
        let obstacles = processSensorDataToObstacles(rawSensorData)
        
        // Fuse sensor data if multiple sensors available
        if (obstacles.length > 1) {
          obstacles = fuseSensorData(obstacles)
          debugInfo.value = `Sensor data fused: ${rawSensorData.length} readings → ${obstacles.length} obstacles`
        }
        
        // Update collision detection state
        collisionDetectionState.obstacles = obstacles
        collisionDetectionState.lastDetectionTime = now.toISOString()
        collisionDetectionState.dataSource = 'sensors'
        collisionDetectionState.sensorCount = rawSensorData.length
        
        // Find nearest obstacle
        if (obstacles.length > 0) {
          const nearest = obstacles.reduce((prev, current) => 
            prev.distance < current.distance ? prev : current
          )
          collisionDetectionState.nearestObstacle = nearest
          
          // Determine warning level based on distance and collision threshold
          const threshold = spatialConstraints.collisionThreshold
          if (nearest.distance < threshold * 0.5) {
            collisionDetectionState.status = 'collision_imminent'
            collisionDetectionState.warningLevel = 'critical'
            debugInfo.value = `⚠️ COLLISION WARNING: Obstacle at ${nearest.distance.toFixed(2)}m (sensor: ${nearest.sensor_type})`
          } else if (nearest.distance < threshold) {
            collisionDetectionState.status = 'warning'
            collisionDetectionState.warningLevel = 'warning'
            debugInfo.value = `⚠️ Warning: Obstacle detected at ${nearest.distance.toFixed(2)}m (sensor: ${nearest.sensor_type})`
          } else if (nearest.distance < threshold * 1.5) {
            collisionDetectionState.status = 'caution'
            collisionDetectionState.warningLevel = 'caution'
            debugInfo.value = `Obstacle in vicinity: ${nearest.distance.toFixed(2)}m`
          } else {
            collisionDetectionState.status = 'clear'
            collisionDetectionState.warningLevel = 'none'
            debugInfo.value = `Clear path, nearest obstacle at ${nearest.distance.toFixed(2)}m`
          }
        } else {
          collisionDetectionState.nearestObstacle = null
          collisionDetectionState.status = 'clear'
          collisionDetectionState.warningLevel = 'none'
          debugInfo.value = 'No obstacles detected by sensors'
        }
        
        const processingTime = performance.now() - startTime
        collisionDetectionState.lastProcessingTime = processingTime
        debugInfo.value += ` | Processing time: ${processingTime.toFixed(1)}ms`
        
        // Add data to sensor chart if using real sensors
        if (sensorChartDataSource.value === 'sensors') {
          addSensorChartDataPoint(rawSensorData)
        }
        
      } catch (error) {
        console.error('Failed to update collision detection with sensors:', error)
        debugInfo.value = `Sensor processing error: ${error.message}, falling back to simulation`
        
        // Fall back to simulation
        updateCollisionDetectionSimulation()
      }
    }
    
    // Update collision detection state (main entry point)
    const updateCollisionDetection = async () => {
      if (!collisionDetectionState.enabled) {
        collisionDetectionState.status = 'disabled'
        collisionDetectionState.warningLevel = 'none'
        collisionDetectionState.dataSource = 'none'
        collisionDetectionState.obstacles = []
        collisionDetectionState.lastDetectionTime = null
        collisionDetectionState.nearestObstacle = null
        collisionDetectionState.sensorCount = 0
        collisionDetectionState.lastProcessingTime = 0
        debugInfo.value = 'Collision detection disabled'
        return
      }
      
      // Use appropriate data source based on configuration
      if (collisionDetectionState.dataSource === 'sensors') {
        await updateCollisionDetectionWithSensors()
      } else {
        // Fall back to simulation
        updateCollisionDetectionSimulation()
      }
    }
    
    // Update collision detection with simulation (original implementation)
    const updateCollisionDetectionSimulation = () => {
      // Simulate collision detection based on navigation state and movement
      if (!collisionDetectionState.enabled) {
        collisionDetectionState.status = 'disabled'
        collisionDetectionState.warningLevel = 'none'
        return
      }
      
      // In a real implementation, this would query the robot's sensors
      // For simulation, we'll generate some mock data
      const now = new Date()
      const timeSinceLastDetection = collisionDetectionState.lastDetectionTime 
        ? now - new Date(collisionDetectionState.lastDetectionTime)
        : Infinity
      
      // Simulate obstacles based on navigation state
      const simulatedObstacles = []
      const obstacleCount = navigationState.obstacleCount || 0
      
      // Generate mock obstacles
      if (obstacleCount > 0) {
        for (let i = 0; i < Math.min(obstacleCount, 5); i++) {
          // Generate random obstacle positions relative to robot
          const distance = 0.2 + Math.random() * 2.0 // 0.2-2.2 meters
          const angle = Math.random() * Math.PI * 2 // random direction
          simulatedObstacles.push({
            id: `sim_obstacle_${i}`,
            distance,
            angle,
            x: Math.cos(angle) * distance,
            y: Math.sin(angle) * distance,
            z: 0,
            confidence: 0.7 + Math.random() * 0.3,
            sensor_type: 'simulation',
            sensor_id: 'sim',
            timestamp: now.toISOString(),
            raw_value: distance,
            raw_unit: 'm'
          })
        }
      }
      
      collisionDetectionState.obstacles = simulatedObstacles
      collisionDetectionState.lastDetectionTime = now.toISOString()
      collisionDetectionState.dataSource = 'simulation'
      collisionDetectionState.sensorCount = simulatedObstacles.length
      
      // Find nearest obstacle
      if (simulatedObstacles.length > 0) {
        const nearest = simulatedObstacles.reduce((prev, current) => 
          prev.distance < current.distance ? prev : current
        )
        collisionDetectionState.nearestObstacle = nearest
        
        // Determine warning level based on distance and collision threshold
        const threshold = spatialConstraints.collisionThreshold
        if (nearest.distance < threshold * 0.5) {
          collisionDetectionState.status = 'collision_imminent'
          collisionDetectionState.warningLevel = 'critical'
          debugInfo.value = `⚠️ COLLISION WARNING: Simulated obstacle at ${nearest.distance.toFixed(2)}m`
        } else if (nearest.distance < threshold) {
          collisionDetectionState.status = 'warning'
          collisionDetectionState.warningLevel = 'warning'
          debugInfo.value = `⚠️ Warning: Simulated obstacle detected at ${nearest.distance.toFixed(2)}m`
        } else if (nearest.distance < threshold * 1.5) {
          collisionDetectionState.status = 'caution'
          collisionDetectionState.warningLevel = 'caution'
          debugInfo.value = `Simulated obstacle in vicinity: ${nearest.distance.toFixed(2)}m`
        } else {
          collisionDetectionState.status = 'clear'
          collisionDetectionState.warningLevel = 'none'
          debugInfo.value = `Clear path, nearest simulated obstacle at ${nearest.distance.toFixed(2)}m`
        }
      } else {
        collisionDetectionState.nearestObstacle = null
        collisionDetectionState.status = 'clear'
        collisionDetectionState.warningLevel = 'none'
        debugInfo.value = 'No obstacles detected (simulation)'
      }
      
      // Add data to sensor chart if using simulation
      if (sensorChartDataSource.value === 'simulation') {
        addSensorChartDataPoint(simulatedObstacles)
      }
    }
    
    // Toggle collision detection
    const toggleCollisionDetection = async () => {
      collisionDetectionState.enabled = !collisionDetectionState.enabled
      debugInfo.value = `Collision detection ${collisionDetectionState.enabled ? 'enabled' : 'disabled'}`
      if (collisionDetectionState.enabled) {
        await updateCollisionDetection()
      } else {
        // Clear all state when disabling
        collisionDetectionState.obstacles = []
        collisionDetectionState.nearestObstacle = null
        collisionDetectionState.status = 'disabled'
        collisionDetectionState.warningLevel = 'none'
        collisionDetectionState.lastDetectionTime = null
        collisionDetectionState.sensorCount = 0
        collisionDetectionState.lastProcessingTime = 0
      }
    }
    
    // Toggle data source between sensors and simulation
    const toggleCollisionDataSource = async () => {
      collisionDetectionState.dataSource = collisionDetectionState.dataSource === 'sensors' ? 'simulation' : 'sensors'
      debugInfo.value = `Collision detection data source switched to: ${collisionDetectionState.dataSource}`
      if (collisionDetectionState.enabled) {
        await updateCollisionDetection()
      }
    }
    
    // Toggle sensor data fusion
    const toggleSensorFusion = async () => {
      collisionDetectionState.fusionEnabled = !collisionDetectionState.fusionEnabled
      debugInfo.value = `Sensor data fusion ${collisionDetectionState.fusionEnabled ? 'enabled' : 'disabled'}`
      if (collisionDetectionState.enabled) {
        await updateCollisionDetection()
      }
    }
    
    // Reset collision detection
    const resetCollisionDetection = () => {
      collisionDetectionState.obstacles = []
      collisionDetectionState.nearestObstacle = null
      collisionDetectionState.status = 'clear'
      collisionDetectionState.warningLevel = 'none'
      collisionDetectionState.lastDetectionTime = null
      collisionDetectionState.dataSource = 'sensors' // Reset to default
      collisionDetectionState.sensorCount = 0
      collisionDetectionState.lastProcessingTime = 0
      collisionDetectionState.fusionEnabled = true
      debugInfo.value = 'Collision detection reset to default settings'
    }
    
    // ========== Sensor Data Visualization Methods ==========
    
    // Toggle sensor chart data source
    const toggleSensorChartDataSource = () => {
      sensorChartDataSource.value = sensorChartDataSource.value === 'sensors' ? 'simulation' : 'sensors'
      debugInfo.value = `Sensor chart data source switched to: ${sensorChartDataSource.value}`
      updateSensorChart()
    }
    
    // Add data point to sensor chart
    const addSensorChartDataPoint = (sensorData) => {
      try {
        const now = new Date()
        const timestamp = now.getTime()
        
        // Filter sensor data based on type and valid readings
        const sensorReadings = sensorData.filter(reading => {
          if (!reading.sensor_type) {
            return false
          }
          // Check if reading has valid value (either value or distance field)
          const hasValue = (reading.value && reading.value > 0) || (reading.distance && reading.distance > 0)
          return hasValue
        })
        
        if (sensorReadings.length === 0) {
          return
        }
        
        // Calculate average distance from all sensors
        // Use distance field if available, otherwise use value field
        const totalDistance = sensorReadings.reduce((sum, reading) => {
          const distance = reading.distance || reading.value || 0
          return sum + distance
        }, 0)
        const averageDistance = totalDistance / sensorReadings.length
        
        // Calculate average confidence
        // Use confidence field if available, otherwise use default 0.5
        const totalConfidence = sensorReadings.reduce((sum, reading) => {
          const confidence = reading.confidence !== undefined ? reading.confidence : 0.5
          return sum + confidence
        }, 0)
        const averageConfidence = totalConfidence / sensorReadings.length
        
        // Count sensor types
        const sensorTypeCounts = {}
        sensorReadings.forEach(reading => {
          const type = reading.sensor_type
          sensorTypeCounts[type] = (sensorTypeCounts[type] || 0) + 1
        })
        
        // Create data point
        const dataPoint = {
          timestamp,
          timeLabel: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          averageDistance,
          averageConfidence,
          sensorCount: sensorReadings.length,
          sensorTypes: Object.keys(sensorTypeCounts),
          sensorTypeCounts,
          rawData: sensorReadings
        }
        
        // Add to chart data
        sensorChartData.value.push(dataPoint)
        
        // Limit data points based on time window
        if (sensorChartTimeWindow.value > 0) {
          const cutoffTime = timestamp - (sensorChartTimeWindow.value * 1000)
          sensorChartData.value = sensorChartData.value.filter(point => point.timestamp >= cutoffTime)
        }
        
        // Limit maximum data points to prevent performance issues
        if (sensorChartData.value.length > 1000) {
          sensorChartData.value = sensorChartData.value.slice(-500) // Keep last 500 points
        }
        
        // Update chart
        updateSensorChart()
        
      } catch (error) {
        console.error('Failed to add sensor chart data point:', error)
      }
    }
    
    // Update sensor chart visualization
    const updateSensorChart = () => {
      if (!sensorChartCanvas.value || sensorChartData.value.length === 0) {
        return
      }
      
      try {
        // Destroy existing chart if it exists
        if (sensorChartInstance.value) {
          sensorChartInstance.value.destroy()
          sensorChartInstance.value = null
        }
        
        // Prepare data for chart
        const labels = sensorChartData.value.map(point => point.timeLabel)
        const distances = sensorChartData.value.map(point => point.averageDistance)
        const confidences = sensorChartData.value.map(point => point.averageConfidence)
        const sensorCounts = sensorChartData.value.map(point => point.sensorCount)
        
        // Create chart data based on selected chart type
        let chartData
        let chartOptions
        
        if (sensorChartType.value === 'radar') {
          // Radar chart for sensor type distribution
          const latestPoint = sensorChartData.value[sensorChartData.value.length - 1]
          const sensorTypes = latestPoint?.sensorTypes || []
          const typeCounts = latestPoint?.sensorTypeCounts || {}
          
          chartData = {
            labels: sensorTypes,
            datasets: [{
              label: 'Sensor Count by Type',
              data: sensorTypes.map(type => typeCounts[type] || 0),
              backgroundColor: 'rgba(54, 162, 235, 0.2)',
              borderColor: 'rgba(54, 162, 235, 1)',
              borderWidth: 1
            }]
          }
          
          chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              r: {
                beginAtZero: true,
                ticks: {
                  stepSize: 1
                }
              }
            }
          }
        } else if (sensorChartType.value === 'bar') {
          // Bar chart for sensor counts over time
          chartData = {
            labels: labels,
            datasets: [{
              label: 'Sensor Count',
              data: sensorCounts,
              backgroundColor: 'rgba(75, 192, 192, 0.5)',
              borderColor: 'rgba(75, 192, 192, 1)',
              borderWidth: 1
            }]
          }
          
          chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: {
                beginAtZero: true,
                title: {
                  display: true,
                  text: 'Number of Sensors'
                }
              },
              x: {
                title: {
                  display: true,
                  text: 'Time'
                },
                ticks: {
                  maxTicksLimit: 10 // Limit number of labels for better readability
                }
              }
            }
          }
        } else {
          // Line chart (default) for distance and confidence
          chartData = {
            labels: labels,
            datasets: [
              {
                label: 'Average Distance (m)',
                data: distances,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.1,
                yAxisID: 'y'
              },
              {
                label: 'Confidence',
                data: confidences,
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                tension: 0.1,
                yAxisID: 'y1'
              }
            ]
          }
          
          chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
              mode: 'index',
              intersect: false
            },
            stacked: false,
            scales: {
              y: {
                type: 'linear',
                display: true,
                position: 'left',
                title: {
                  display: true,
                  text: 'Distance (m)'
                }
              },
              y1: {
                type: 'linear',
                display: true,
                position: 'right',
                title: {
                  display: true,
                  text: 'Confidence'
                },
                grid: {
                  drawOnChartArea: false
                },
                min: 0,
                max: 1
              }
            }
          }
        }
        
        // Create new chart instance
        const ctx = sensorChartCanvas.value.getContext('2d')
        sensorChartInstance.value = new Chart(ctx, {
          type: sensorChartType.value === 'radar' ? 'radar' : (sensorChartType.value === 'bar' ? 'bar' : 'line'),
          data: chartData,
          options: chartOptions
        })
        
      } catch (error) {
        console.error('Failed to update sensor chart:', error)
      }
    }
    
    // Clear sensor chart data
    const clearSensorChartData = () => {
      sensorChartData.value = []
      
      // Destroy chart if it exists
      if (sensorChartInstance.value) {
        sensorChartInstance.value.destroy()
        sensorChartInstance.value = null
      }
      
      debugInfo.value = 'Sensor chart data cleared'
    }
    
    // Initialize sensor chart
    const initializeSensorChart = () => {
      if (sensorChartCanvas.value) {
        // Chart will be created when data is available
        debugInfo.value = 'Sensor chart initialized'
      }
    }
    
    // Destroy sensor chart
    const destroySensorChart = () => {
      if (sensorChartInstance.value) {
        sensorChartInstance.value.destroy()
        sensorChartInstance.value = null
      }
    }
    
    // ========== Data Recording and Playback Methods ==========
    
    // Start recording sensor and joint data
    const startRecording = () => {
      if (dataRecordingState.recording || dataRecordingState.playback) {
        return
      }
      
      dataRecordingState.recording = true
      dataRecordingState.recordStartTime = Date.now()
      dataRecordingState.recordDuration = 0
      dataRecordingState.records = []
      dataRecordingState.currentRecordIndex = -1
      
      debugInfo.value = 'Started data recording'
      
      // Start recording timer
      dataRecordingState.recordTimer = setInterval(() => {
        if (!dataRecordingState.recording) {
          clearInterval(dataRecordingState.recordTimer)
          return
        }
        
        // Update recording duration
        dataRecordingState.recordDuration = Date.now() - dataRecordingState.recordStartTime
        
        // Create record
        const record = {
          timestamp: Date.now(),
          type: 'combined',
          sensorData: sensorData.value.slice(), // Copy current sensor data
          jointData: Object.fromEntries(jointList.value.map(joint => [joint.id, joint.value])),
          position: movementState.position ? { ...movementState.position } : null,
          collisionDetection: { ...collisionDetectionState },
          trajectoryStats: { ...trajectoryStats.value }
        }
        
        // Add to records
        dataRecordingState.records.push(record)
        
        // Limit records to maxRecords
        if (dataRecordingState.records.length > dataRecordingState.maxRecords) {
          dataRecordingState.records = dataRecordingState.records.slice(-dataRecordingState.maxRecords)
        }
        
      }, dataRecordingState.recordInterval)
    }
    
    // Stop recording
    const stopRecording = () => {
      if (!dataRecordingState.recording) {
        return
      }
      
      clearInterval(dataRecordingState.recordTimer)
      dataRecordingState.recording = false
      dataRecordingState.recordTimer = null
      
      debugInfo.value = `Stopped recording. Saved ${dataRecordingState.records.length} records.`
    }
    
    // Start playback of recorded data
    const startPlayback = () => {
      if (dataRecordingState.recording || dataRecordingState.playback || dataRecordingState.records.length === 0) {
        return
      }
      
      dataRecordingState.playback = true
      dataRecordingState.currentRecordIndex = -1
      
      debugInfo.value = `Starting playback of ${dataRecordingState.records.length} records`
      
      // Calculate playback interval based on speed
      const playbackInterval = dataRecordingState.recordInterval / dataRecordingState.playbackSpeed
      
      // Start playback timer
      dataRecordingState.playbackTimer = setInterval(() => {
        if (!dataRecordingState.playback) {
          clearInterval(dataRecordingState.playbackTimer)
          return
        }
        
        // Move to next record
        dataRecordingState.currentRecordIndex++
        
        // Check if playback is complete
        if (dataRecordingState.currentRecordIndex >= dataRecordingState.records.length) {
          stopPlayback()
          debugInfo.value = 'Playback completed'
          return
        }
        
        const record = dataRecordingState.records[dataRecordingState.currentRecordIndex]
        
        // Apply recorded data (simulate playback)
        // In a real implementation, this would update the robot state
        debugInfo.value = `Playback: ${dataRecordingState.currentRecordIndex + 1}/${dataRecordingState.records.length} (${new Date(record.timestamp).toLocaleTimeString()})`
        
      }, playbackInterval)
    }
    
    // Stop playback
    const stopPlayback = () => {
      if (!dataRecordingState.playback) {
        return
      }
      
      clearInterval(dataRecordingState.playbackTimer)
      dataRecordingState.playback = false
      dataRecordingState.playbackTimer = null
      dataRecordingState.currentRecordIndex = -1
      
      debugInfo.value = 'Playback stopped'
    }
    
    // Clear all records
    const clearRecords = () => {
      if (dataRecordingState.recording || dataRecordingState.playback) {
        debugInfo.value = 'Cannot clear records while recording or playback is active'
        return
      }
      
      dataRecordingState.records = []
      dataRecordingState.currentRecordIndex = -1
      dataRecordingState.recordStartTime = null
      dataRecordingState.recordDuration = 0
      
      debugInfo.value = 'All records cleared'
    }
    
    // Clear records with confirmation
    const clearRecordsWithConfirmation = () => {
      if (dataRecordingState.recording || dataRecordingState.playback) {
        debugInfo.value = 'Cannot clear records while recording or playback is active'
        return
      }
      
      if (dataRecordingState.records.length === 0) {
        debugInfo.value = 'No records to clear'
        return
      }
      
      showConfirmDialog(
        'Clear All Records',
        `Are you sure you want to clear all ${dataRecordingState.records.length} records? This action cannot be undone.`,
        'Clear All',
        'Cancel',
        () => {
          clearRecords()
        },
        () => {
          debugInfo.value = 'Record clearing cancelled'
        }
      )
    }
    
    // Show confirmation dialog
    const showConfirmDialog = (title, message, confirmText = 'Confirm', cancelText = 'Cancel', onConfirm, onCancel) => {
      uiConfirmState.show = true
      uiConfirmState.title = title
      uiConfirmState.message = message
      uiConfirmState.confirmText = confirmText
      uiConfirmState.cancelText = cancelText
      uiConfirmState.onConfirm = onConfirm
      uiConfirmState.onCancel = onCancel
    }
    
    // Hide confirmation dialog
    const hideConfirmDialog = () => {
      uiConfirmState.show = false
      uiConfirmState.title = ''
      uiConfirmState.message = ''
      uiConfirmState.confirmText = 'Confirm'
      uiConfirmState.cancelText = 'Cancel'
      uiConfirmState.onConfirm = null
      uiConfirmState.onCancel = null
    }
    
    // Confirm action
    const confirmAction = () => {
      if (uiConfirmState.onConfirm) {
        uiConfirmState.onConfirm()
      }
      hideConfirmDialog()
    }
    
    // Cancel action
    const cancelAction = () => {
      if (uiConfirmState.onCancel) {
        uiConfirmState.onCancel()
      }
      hideConfirmDialog()
    }
    
    // Export navigation data
    const exportNavigationData = () => {
      const data = {
        waypoints: waypoints.value,
        spatialConstraints: { ...spatialConstraints },
        navigationState: { ...navigationState },
        movementState: { ...movementState },
        timestamp: new Date().toISOString()
      }
      
      try {
        const dataStr = JSON.stringify(data, null, 2)
        const dataBlob = new Blob([dataStr], { type: 'application/json' })
        const url = URL.createObjectURL(dataBlob)
        const link = document.createElement('a')
        link.href = url
        link.download = `navigation-data-${new Date().getTime()}.json`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        URL.revokeObjectURL(url)
        
        debugInfo.value = 'Navigation data exported successfully'
      } catch (error) {
        console.error('Failed to export navigation data:', error)
        debugInfo.value = `Failed to export navigation data: ${error.message}`
      }
    }

    // ========== Keyboard Shortcut Control ==========
    
    // Track keyboard state for continuous movement
    const keyboardState = reactive({
      forward: false,
      backward: false,
      left: false,
      right: false,
      active: false
    })
    
    // Movement history for trajectory visualization
    const movementHistory = ref([])
    const maxHistoryLength = 100
    
    // Handle keyboard events for robot control
    const handleKeyDown = (event) => {
      // Prevent default behavior for movement keys to avoid scrolling
      const key = event.key.toLowerCase()
      
      switch (key) {
        case 'w':
        case 'arrowup':
          if (!keyboardState.forward) {
            keyboardState.forward = true
            keyboardState.active = true
            moveRobot('forward')
            debugInfo.value = 'Keyboard: Moving forward (W/↑)'
          }
          break
        case 's':
        case 'arrowdown':
          if (!keyboardState.backward) {
            keyboardState.backward = true
            keyboardState.active = true
            moveRobot('backward')
            debugInfo.value = 'Keyboard: Moving backward (S/↓)'
          }
          break
        case 'a':
        case 'arrowleft':
          if (!keyboardState.left) {
            keyboardState.left = true
            keyboardState.active = true
            moveRobot('left')
            debugInfo.value = 'Keyboard: Moving left (A/←)'
          }
          break
        case 'd':
        case 'arrowright':
          if (!keyboardState.right) {
            keyboardState.right = true
            keyboardState.active = true
            moveRobot('right')
            debugInfo.value = 'Keyboard: Moving right (D/→)'
          }
          break
        case ' ':
          // Space bar for stop
          event.preventDefault()
          stopRobot()
          debugInfo.value = 'Keyboard: Stop movement (Space)'
          break
        case 'q':
          // Q for rotate left
          rotateRobot('left')
          debugInfo.value = 'Keyboard: Rotate left (Q)'
          break
        case 'e':
          // E for rotate right
          rotateRobot('right')
          debugInfo.value = 'Keyboard: Rotate right (E)'
          break
        case 'r':
          // R for reset rotation
          rotateRobot('reset')
          debugInfo.value = 'Keyboard: Reset rotation (R)'
          break
      }
      
      // Record movement in history if any movement key was pressed
      if (['w', 's', 'a', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(key)) {
        recordMovementToHistory()
      }
    }
    
    // Handle key up events
    const handleKeyUp = (event) => {
      const key = event.key.toLowerCase()
      
      switch (key) {
        case 'w':
        case 'arrowup':
          keyboardState.forward = false
          break
        case 's':
        case 'arrowdown':
          keyboardState.backward = false
          break
        case 'a':
        case 'arrowleft':
          keyboardState.left = false
          break
        case 'd':
        case 'arrowright':
          keyboardState.right = false
          break
      }
      
      // Check if all movement keys are released
      if (!keyboardState.forward && !keyboardState.backward && 
          !keyboardState.left && !keyboardState.right) {
        keyboardState.active = false
      }
    }
    
    // Record movement to history for trajectory visualization
    const recordMovementToHistory = () => {
      const timestamp = new Date().toISOString()
      const position = { ...movementState.position }
      const movement = {
        timestamp,
        position,
        keyboardState: { ...keyboardState },
        movementState: { ...movementState }
      }
      
      movementHistory.value.unshift(movement)
      
      // Keep history within limit
      if (movementHistory.value.length > maxHistoryLength) {
        movementHistory.value = movementHistory.value.slice(0, maxHistoryLength)
      }
    }
    
    // Clear movement history
    const clearMovementHistory = () => {
      movementHistory.value = []
      debugInfo.value = 'Movement history cleared'
      clearTrajectory()
    }
    
    // Canvas reference for trajectory visualization
    const trajectoryCanvas = ref(null)
    
    // Mouse interaction for trajectory visualization
    const mousePosition = ref({ x: 0, y: 0 })
    const showMouseTooltip = ref(false)
    const mouseWorldCoords = ref({ x: 0, y: 0 })
    
    // Debounce function for performance optimization (returns cancelable function)
    const debounce = (func, wait) => {
      let timeout
      const debouncedFunc = function(...args) {
        const later = () => {
          clearTimeout(timeout)
          func(...args)
        }
        clearTimeout(timeout)
        timeout = setTimeout(later, wait)
      }
      
      // Add cancel method
      debouncedFunc.cancel = () => {
        if (timeout) {
          clearTimeout(timeout)
          timeout = null
        }
      }
      
      return debouncedFunc
    }
    
    // Debounced version of drawTrajectory (stored in ref for cleanup)
    const debouncedDrawTrajectoryRef = ref(null)
    onMounted(() => {
      debouncedDrawTrajectoryRef.value = debounce(() => {
        if (trajectoryCanvas.value && movementHistory.value.length > 0) {
          drawTrajectory()
        }
      }, 50) // 50ms debounce delay
    })
    
    // Handle mouse move on canvas
    const handleCanvasMouseMove = (event) => {
      if (!trajectoryCanvas.value) return
      
      const rect = trajectoryCanvas.value.getBoundingClientRect()
      mousePosition.value = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
      }
      
      // Convert canvas coordinates to world coordinates
      if (movementHistory.value.length > 0) {
        // Recalculate scaling factors (similar to drawTrajectory)
        let minX = Infinity, maxX = -Infinity
        let minY = Infinity, maxY = -Infinity
        
        movementHistory.value.forEach(record => {
          const x = record.position?.x || 0
          const y = record.position?.y || 0
          minX = Math.min(minX, x)
          maxX = Math.max(maxX, x)
          minY = Math.min(minY, y)
          maxY = Math.max(maxY, y)
        })
        
        const padding = 0.1
        const rangeX = Math.max(maxX - minX, 0.1)
        const rangeY = Math.max(maxY - minY, 0.1)
        minX -= rangeX * padding
        maxX += rangeX * padding
        minY -= rangeY * padding
        maxY += rangeY * padding
        
        const width = trajectoryCanvas.value.width
        const height = trajectoryCanvas.value.height
        const scaleX = width / (maxX - minX || 1)
        const scaleY = height / (maxY - minY || 1)
        const scale = Math.min(scaleX, scaleY) * 0.8
        
        // Convert from canvas to world coordinates
        const canvasX = mousePosition.value.x
        const canvasY = mousePosition.value.y
        
        // Reverse the transformation used in drawTrajectory
        const worldX = ((canvasX - (width - (maxX - minX) * scale) / 2) / scale) + minX
        const worldY = ((height - canvasY - (height - (maxY - minY) * scale) / 2) / scale) + minY
        
        mouseWorldCoords.value = {
          x: worldX,
          y: worldY
        }
      }
      
      showMouseTooltip.value = true
      
      // Redraw trajectory to show tooltip (debounced for performance)
      if (trajectoryCanvas.value && movementHistory.value.length > 0 && debouncedDrawTrajectoryRef.value) {
        debouncedDrawTrajectoryRef.value()
      }
    }
    
    const handleCanvasMouseLeave = () => {
      showMouseTooltip.value = false
      
      // Redraw trajectory to hide tooltip
      if (trajectoryCanvas.value && movementHistory.value.length > 0) {
        drawTrajectory()
      }
    }
    
    // Clear trajectory visualization
    const clearTrajectory = () => {
      if (trajectoryCanvas.value) {
        const ctx = trajectoryCanvas.value.getContext('2d')
        ctx.clearRect(0, 0, trajectoryCanvas.value.width, trajectoryCanvas.value.height)
      }
      debugInfo.value = 'Trajectory cleared'
    }
    
    // Cache for trajectory visualization
    const trajectoryCache = {
      gridCanvas: null,
      gridDrawn: false,
      lastHistoryLength: 0,
      lastMinMax: { minX: 0, maxX: 0, minY: 0, maxY: 0 },
      lastScale: 0
    }
    
    // Draw static grid and axes to cache
    const drawGridToCache = (width, height) => {
      if (!trajectoryCache.gridCanvas) {
        trajectoryCache.gridCanvas = document.createElement('canvas')
        trajectoryCache.gridCanvas.width = width
        trajectoryCache.gridCanvas.height = height
      } else if (trajectoryCache.gridCanvas.width !== width || trajectoryCache.gridCanvas.height !== height) {
        trajectoryCache.gridCanvas.width = width
        trajectoryCache.gridCanvas.height = height
        trajectoryCache.gridDrawn = false
      }
      
      if (trajectoryCache.gridDrawn) {
        return trajectoryCache.gridCanvas
      }
      
      const cacheCtx = trajectoryCache.gridCanvas.getContext('2d')
      
      // Clear cache canvas
      cacheCtx.clearRect(0, 0, width, height)
      
      // Draw grid background
      cacheCtx.strokeStyle = '#eee'
      cacheCtx.lineWidth = 0.5
      
      // Draw vertical grid lines
      const gridSize = 50
      for (let x = gridSize; x < width; x += gridSize) {
        cacheCtx.beginPath()
        cacheCtx.moveTo(x, 0)
        cacheCtx.lineTo(x, height)
        cacheCtx.stroke()
      }
      
      // Draw horizontal grid lines
      for (let y = gridSize; y < height; y += gridSize) {
        cacheCtx.beginPath()
        cacheCtx.moveTo(0, y)
        cacheCtx.lineTo(width, y)
        cacheCtx.stroke()
      }
      
      // Draw coordinate system
      cacheCtx.strokeStyle = '#999'
      cacheCtx.lineWidth = 1.5
      
      // X axis with arrow
      cacheCtx.beginPath()
      cacheCtx.moveTo(0, height / 2)
      cacheCtx.lineTo(width, height / 2)
      // Draw arrow at the end
      cacheCtx.lineTo(width - 10, height / 2 - 5)
      cacheCtx.moveTo(width, height / 2)
      cacheCtx.lineTo(width - 10, height / 2 + 5)
      cacheCtx.stroke()
      
      // Y axis with arrow
      cacheCtx.beginPath()
      cacheCtx.moveTo(width / 2, height)
      cacheCtx.lineTo(width / 2, 0)
      // Draw arrow at the end
      cacheCtx.lineTo(width / 2 - 5, 10)
      cacheCtx.moveTo(width / 2, 0)
      cacheCtx.lineTo(width / 2 + 5, 10)
      cacheCtx.stroke()
      
      // Add axis labels
      cacheCtx.fillStyle = '#666'
      cacheCtx.font = '12px Arial'
      cacheCtx.textAlign = 'right'
      cacheCtx.textBaseline = 'middle'
      // X axis label
      cacheCtx.fillText('X', width - 5, height / 2 - 15)
      // Y axis label
      cacheCtx.textAlign = 'center'
      cacheCtx.textBaseline = 'top'
      cacheCtx.fillText('Y', width / 2 + 15, 5)
      
      trajectoryCache.gridDrawn = true
      return trajectoryCache.gridCanvas
    }
    
    // Draw trajectory on canvas
    const drawTrajectory = () => {
      if (!trajectoryCanvas.value || movementHistory.value.length === 0) {
        return
      }
      
      const ctx = trajectoryCanvas.value.getContext('2d')
      const width = trajectoryCanvas.value.width
      const height = trajectoryCanvas.value.height
      
      // Clear canvas
      ctx.clearRect(0, 0, width, height)
      
      // Draw cached grid and axes
      const gridCanvas = drawGridToCache(width, height)
      ctx.drawImage(gridCanvas, 0, 0)
      
      // Find min/max coordinates for scaling
      let minX = Infinity, maxX = -Infinity
      let minY = Infinity, maxY = -Infinity
      
      movementHistory.value.forEach(record => {
        const x = record.position?.x || 0
        const y = record.position?.y || 0
        minX = Math.min(minX, x)
        maxX = Math.max(maxX, x)
        minY = Math.min(minY, y)
        maxY = Math.max(maxY, y)
      })
      
      // Add padding
      const padding = 0.1
      const rangeX = Math.max(maxX - minX, 0.1)
      const rangeY = Math.max(maxY - minY, 0.1)
      minX -= rangeX * padding
      maxX += rangeX * padding
      minY -= rangeY * padding
      maxY += rangeY * padding
      
      // Scale coordinates to canvas
      const scaleX = width / (maxX - minX || 1)
      const scaleY = height / (maxY - minY || 1)
      const scale = Math.min(scaleX, scaleY) * 0.8
      
      // Add coordinate values
      ctx.fillStyle = '#888'
      ctx.font = '10px Arial'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      
      // X axis tick marks and labels
      const xTickInterval = Math.max(1, Math.ceil((maxX - minX) / 5))
      for (let xVal = Math.ceil(minX / xTickInterval) * xTickInterval; xVal <= maxX; xVal += xTickInterval) {
        const canvasX = (xVal - minX) * scale + (width - (maxX - minX) * scale) / 2
        // Draw tick mark
        ctx.beginPath()
        ctx.moveTo(canvasX, height / 2 - 3)
        ctx.lineTo(canvasX, height / 2 + 3)
        ctx.stroke()
        // Draw label
        ctx.fillText(xVal.toFixed(1), canvasX, height / 2 + 8)
      }
      
      // Y axis tick marks and labels
      const yTickInterval = Math.max(1, Math.ceil((maxY - minY) / 5))
      for (let yVal = Math.ceil(minY / yTickInterval) * yTickInterval; yVal <= maxY; yVal += yTickInterval) {
        const canvasY = height - ((yVal - minY) * scale + (height - (maxY - minY) * scale) / 2)
        // Draw tick mark
        ctx.beginPath()
        ctx.moveTo(width / 2 - 3, canvasY)
        ctx.lineTo(width / 2 + 3, canvasY)
        ctx.stroke()
        // Draw label
        ctx.textAlign = 'right'
        ctx.textBaseline = 'middle'
        ctx.fillText(yVal.toFixed(1), width / 2 - 8, canvasY)
      }
      
      // Reset text alignment
      ctx.textAlign = 'left'
      ctx.textBaseline = 'alphabetic'
      
      // Draw movement path
      if (movementHistory.value.length >= 2) {
        ctx.strokeStyle = '#ff6600'
        ctx.lineWidth = 2
        ctx.beginPath()
        
        // Start from oldest to newest (reverse order since history is newest first)
        const reversedHistory = [...movementHistory.value].reverse()
        const pathPoints = []
        
        reversedHistory.forEach((record, index) => {
          const x = record.position?.x || 0
          const y = record.position?.y || 0
          
          // Convert to canvas coordinates
          const canvasX = (x - minX) * scale + (width - (maxX - minX) * scale) / 2
          const canvasY = height - ((y - minY) * scale + (height - (maxY - minY) * scale) / 2)
          
          pathPoints.push({ x: canvasX, y: canvasY })
          
          if (index === 0) {
            ctx.moveTo(canvasX, canvasY)
          } else {
            ctx.lineTo(canvasX, canvasY)
          }
        })
        
        ctx.stroke()
        
        // Draw arrows along the path to indicate direction
        if (pathPoints.length >= 2) {
          ctx.strokeStyle = '#ff3300'
          ctx.fillStyle = '#ff3300'
          ctx.lineWidth = 1.5
          
          // Function to draw an arrow at a given position with a given angle
          const drawArrow = (x, y, angle) => {
            const arrowLength = 10
            const arrowWidth = 6
            
            ctx.save()
            ctx.translate(x, y)
            ctx.rotate(angle)
            
            ctx.beginPath()
            ctx.moveTo(-arrowLength, -arrowWidth / 2)
            ctx.lineTo(0, 0)
            ctx.lineTo(-arrowLength, arrowWidth / 2)
            ctx.closePath()
            ctx.fill()
            
            ctx.restore()
          }
          
          // Draw arrows at regular intervals along the path
          const arrowSpacing = 50 // pixels between arrows
          let accumulatedDistance = 0
          
          for (let i = 1; i < pathPoints.length; i++) {
            const prev = pathPoints[i - 1]
            const curr = pathPoints[i]
            
            const dx = curr.x - prev.x
            const dy = curr.y - prev.y
            const segmentLength = Math.sqrt(dx * dx + dy * dy)
            
            if (segmentLength === 0) continue
            
            // Calculate direction angle (in radians)
            const angle = Math.atan2(dy, dx)
            
            // Draw arrows along this segment
            const numArrowsOnSegment = Math.floor(segmentLength / arrowSpacing)
            
            for (let j = 0; j < numArrowsOnSegment; j++) {
              const t = (j + 0.5) / numArrowsOnSegment // position along segment (0 to 1)
              const arrowX = prev.x + dx * t
              const arrowY = prev.y + dy * t
              
              drawArrow(arrowX, arrowY, angle)
            }
            
            accumulatedDistance += segmentLength
          }
          
          // Always draw an arrow at the end of the path (current direction)
          if (pathPoints.length >= 2) {
            const lastIndex = pathPoints.length - 1
            const prev = pathPoints[lastIndex - 1]
            const curr = pathPoints[lastIndex]
            const dx = curr.x - prev.x
            const dy = curr.y - prev.y
            const angle = Math.atan2(dy, dx)
            
            drawArrow(curr.x, curr.y, angle)
          }
        }
      }
      
      // Draw history points
      movementHistory.value.forEach(record => {
        const x = record.position?.x || 0
        const y = record.position?.y || 0
        
        // Convert to canvas coordinates
        const canvasX = (x - minX) * scale + (width - (maxX - minX) * scale) / 2
        const canvasY = height - ((y - minY) * scale + (height - (maxY - minY) * scale) / 2)
        
        // Draw point
        ctx.fillStyle = '#00cc66'
        ctx.beginPath()
        ctx.arc(canvasX, canvasY, 3, 0, Math.PI * 2)
        ctx.fill()
      })
      
      // Draw current position (most recent)
      if (movementHistory.value.length > 0) {
        const latest = movementHistory.value[0]
        const x = latest.position?.x || 0
        const y = latest.position?.y || 0
        
        // Convert to canvas coordinates
        const canvasX = (x - minX) * scale + (width - (maxX - minX) * scale) / 2
        const canvasY = height - ((y - minY) * scale + (height - (maxY - minY) * scale) / 2)
        
        // Draw current position
        ctx.fillStyle = '#0066cc'
        ctx.beginPath()
        ctx.arc(canvasX, canvasY, 6, 0, Math.PI * 2)
        ctx.fill()
        
        // Draw position indicator
        ctx.fillStyle = '#ffffff'
        ctx.beginPath()
        ctx.arc(canvasX, canvasY, 2, 0, Math.PI * 2)
        ctx.fill()
      }
      
      // Draw mouse tooltip if enabled
      if (showMouseTooltip.value && mousePosition.value.x > 0 && mousePosition.value.y > 0) {
        const mouseX = mousePosition.value.x
        const mouseY = mousePosition.value.y
        
        // Draw crosshair at mouse position
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(mouseX - 10, mouseY)
        ctx.lineTo(mouseX + 10, mouseY)
        ctx.stroke()
        ctx.beginPath()
        ctx.moveTo(mouseX, mouseY - 10)
        ctx.lineTo(mouseX, mouseY + 10)
        ctx.stroke()
        
        // Draw tooltip background
        const tooltipText = `(${mouseWorldCoords.value.x.toFixed(2)}, ${mouseWorldCoords.value.y.toFixed(2)})`
        ctx.font = '12px Arial'
        const textWidth = ctx.measureText(tooltipText).width
        const tooltipX = Math.min(mouseX + 15, width - textWidth - 10)
        const tooltipY = mouseY - 30
        
        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
        ctx.fillRect(tooltipX - 5, tooltipY - 5, textWidth + 10, 20)
        
        // Text
        ctx.fillStyle = 'white'
        ctx.textAlign = 'left'
        ctx.textBaseline = 'middle'
        ctx.fillText(tooltipText, tooltipX, tooltipY + 5)
      }
    }
    
    // Calculate trajectory statistics
    const calculateTrajectoryStats = () => {
      try {
        if (movementHistory.value.length < 2) {
          return {
            totalDistance: 0,
            averageSpeed: 0,
            movementTime: 0,
            startTime: null,
            endTime: null,
            pointCount: movementHistory.value.length
          }
        }
        
        // Sort history by timestamp (oldest to newest)
        const sortedHistory = [...movementHistory.value].sort((a, b) => 
          new Date(a.timestamp || 0) - new Date(b.timestamp || 0)
        )
        
        let totalDistance = 0
        let startTime = sortedHistory[0]?.timestamp ? new Date(sortedHistory[0].timestamp) : null
        let endTime = sortedHistory[sortedHistory.length - 1]?.timestamp ? 
          new Date(sortedHistory[sortedHistory.length - 1].timestamp) : null
        
        // Calculate total distance
        for (let i = 1; i < sortedHistory.length; i++) {
          const prev = sortedHistory[i - 1]
          const curr = sortedHistory[i]
          
          const prevX = prev.position?.x || 0
          const prevY = prev.position?.y || 0
          const currX = curr.position?.x || 0
          const currY = curr.position?.y || 0
          
          const dx = currX - prevX
          const dy = currY - prevY
          totalDistance += Math.sqrt(dx * dx + dy * dy)
        }
        
        // Calculate movement time in seconds
        let movementTime = 0
        if (startTime && endTime) {
          movementTime = (endTime - startTime) / 1000
        }
        
        // Calculate average speed (distance per second)
        const averageSpeed = movementTime > 0 ? totalDistance / movementTime : 0
        
        return {
          totalDistance: parseFloat(totalDistance.toFixed(3)),
          averageSpeed: parseFloat(averageSpeed.toFixed(3)),
          movementTime: parseFloat(movementTime.toFixed(1)),
          startTime,
          endTime,
          pointCount: sortedHistory.length
        }
      } catch (error) {
        console.error('Error calculating trajectory stats:', error)
        return {
          totalDistance: 0,
          averageSpeed: 0,
          movementTime: 0,
          startTime: null,
          endTime: null,
          pointCount: 0
        }
      }
    }
    
    // Computed property for trajectory statistics
    const trajectoryStats = computed(() => calculateTrajectoryStats())
    
    // Watch movement history and redraw trajectory
    watch(movementHistory, () => {
      if (movementHistory.value.length > 0) {
        // Use nextTick to ensure DOM is updated
        setTimeout(() => {
          drawTrajectory()
        }, 0)
      }
    }, { deep: true })
    
    // Initialize keyboard event listeners
    onMounted(async () => {
      window.addEventListener('keydown', handleKeyDown)
      window.addEventListener('keyup', handleKeyUp)
      debugInfo.value = 'Keyboard shortcuts enabled (WASD/Arrow keys for movement, Space to stop, Q/E/R for rotation)'
      
      // Add mouse event listeners for trajectory visualization
      setTimeout(() => {
        if (trajectoryCanvas.value) {
          trajectoryCanvas.value.addEventListener('mousemove', handleCanvasMouseMove)
          trajectoryCanvas.value.addEventListener('mouseleave', handleCanvasMouseLeave)
          debugInfo.value += ' | Canvas mouse interaction enabled'
        }
      }, 200)
      
      // Initialize trajectory visualization
      setTimeout(() => {
        if (movementHistory.value.length > 0) {
          drawTrajectory()
        }
      }, 100)
      
      // Initialize collision detection
      await updateCollisionDetection()
      
      // Initialize sensor chart visualization
      setTimeout(() => {
        initializeSensorChart()
      }, 300)
      
      // Start collision detection timer (simulate real-time updates)
      collisionDetectionTimer.value = setInterval(async () => {
        if (collisionDetectionState.enabled && navigationState.active) {
          await updateCollisionDetection()
        }
      }, 2000) // Update every 2 seconds
    })
    
    // Clean up keyboard event listeners
    onUnmounted(() => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
      
      // Clean up canvas event listeners
      if (trajectoryCanvas.value) {
        trajectoryCanvas.value.removeEventListener('mousemove', handleCanvasMouseMove)
        trajectoryCanvas.value.removeEventListener('mouseleave', handleCanvasMouseLeave)
      }
      
      // Clean up collision detection timer
      if (collisionDetectionTimer.value) {
        clearInterval(collisionDetectionTimer.value)
        collisionDetectionTimer.value = null
      }
      
      // Clean up debounced draw function
      if (debouncedDrawTrajectoryRef.value) {
        debouncedDrawTrajectoryRef.value.cancel()
        debouncedDrawTrajectoryRef.value = null
      }
      
      // Clean up sensor chart
      destroySensorChart()
    })

    // Fetch collaboration patterns list
    const fetchCollaborationPatterns = async () => {
      return await performDataLoad('collaboration-patterns', {
        apiClient: apiClient,
        apiMethod: 'get',
        apiEndpoint: '/api/robot/collaboration/patterns',
        dataPath: 'patterns',
        onBeforeStart: () => {
          debugInfo.value = 'Fetching collaboration patterns list...'
        },
        onSuccess: (patterns, fullResponse) => {
          if (fullResponse.data && fullResponse.data.status === 'success') {
            collaborationPatterns.value = patterns || []
            debugInfo.value = `Fetched ${collaborationPatterns.value.length} collaboration patterns`
          } else {
            debugInfo.value = 'Failed to fetch collaboration patterns list'
          }
        },
        onError: (error) => {
          console.error('Failed to fetch collaboration patterns list:', error)
          debugInfo.value = `Failed to fetch collaboration patterns list: ${error.message}`
        },
        successMessage: '', // We handle messages manually
        errorMessage: 'Failed to fetch collaboration patterns list',
        errorContext: 'Fetch Collaboration Patterns',
        showSuccess: false,
        showError: false,
        notify: null,
        handleError: null,
        fallbackValue: []
      })
    }

    // Watch selected collaboration pattern changes
    watch(selectedPattern, (newPattern) => {
      if (newPattern) {
        // Find details from collaboration patterns list
        const pattern = collaborationPatterns.value.find(p => p.name === newPattern)
        if (pattern) {
          selectedPatternDetails.value = pattern
          debugInfo.value = `Selected collaboration pattern: ${pattern.name}`
        } else {
          selectedPatternDetails.value = null
          debugInfo.value = `Collaboration pattern details not found: ${newPattern}`
        }
      } else {
        selectedPatternDetails.value = null
      }
    })

    // Start collaboration
    const startCollaboration = async () => {
      if (!selectedPattern.value) {
        debugInfo.value = 'Please select a collaboration pattern first'
        return
      }

      // Parse input data
      let inputData = {}
      try {
        inputData = JSON.parse(collaborationInput.value)
      } catch (e) {
        inputData = { raw_input: collaborationInput.value }
      }

      return await performDataOperation('start-collaboration', {
        apiClient: apiClient,
        apiMethod: 'post',
        apiEndpoint: '/api/robot/collaborate',
        requestData: {
          pattern: selectedPattern.value,
          input_data: inputData,
          custom_config: {}
        },
        onBeforeStart: () => {
          collaborationState.loading = true
          collaborationState.status = 'starting'
          debugInfo.value = `Starting collaboration pattern: ${selectedPattern.value}`
        },
        onSuccess: (responseData) => {
          if (responseData) {
            collaborationState.lastResult = responseData
            collaborationState.activePattern = selectedPattern.value
            collaborationState.status = responseData.status === 'success' ? 'success' : 'error'
            debugInfo.value = `Collaboration completed: ${responseData.status}`
          } else {
            collaborationState.status = 'error'
            debugInfo.value = 'Collaboration request failed'
          }
        },
        onError: (error) => {
          console.error('Failed to start collaboration:', error)
          collaborationState.status = 'error'
          debugInfo.value = `Failed to start collaboration: ${error.message}`
        },
        onFinally: () => {
          collaborationState.loading = false
        },
        successMessage: '', // We handle messages manually
        errorMessage: 'Failed to start collaboration',
        errorContext: 'Start Collaboration',
        showSuccess: false,
        showError: false,
        notify: null,
        handleError: null
      })
    }

    // Stop collaboration
    const stopCollaboration = async () => {
      try {
        debugInfo.value = 'Stopping collaboration...'
        // Here we can add API call to stop collaboration
        collaborationState.activePattern = null
        collaborationState.status = 'stopped'
        debugInfo.value = 'Collaboration stopped'
      } catch (error) {
        console.error('Failed to stop collaboration:', error)
        debugInfo.value = `Failed to stop collaboration: ${error.message}`
      }
    }

    // Update joint
    const updateJoint = async (jointId, value) => {
      clickCount.value++
      if (import.meta.env.DEV) {
        console.log(`Update joint ${jointId}: ${value}°`)
      }
      debugInfo.value = `Joint ${jointId} updated to ${value}°, total clicks: ${clickCount.value}`
      
      return await performDataOperation('update-joint', {
        apiClient: apiClient,
        apiMethod: 'post',
        apiEndpoint: '/api/robot/joint',
        requestData: {
          jointId: jointId,
          value: value
        },
        onBeforeStart: () => {
          debugInfo.value = `Sending joint control command: ${jointId} = ${value}°...`
        },
        onSuccess: (responseData) => {
          if (responseData && responseData.success) {
            debugInfo.value = `Joint control successful: ${jointId} = ${value}° - ${responseData.message}`
            if (import.meta.env.DEV) {
              console.log(`Joint control API response:`, responseData)
            }
          } else {
            debugInfo.value = `Joint control API returned failure: ${jointId} - ${responseData?.message || 'Unknown error'}`
            console.warn(`Joint control API returned failure:`, responseData)
            throw new Error(responseData?.message || 'Joint control API returned failure')
          }
        },
        onError: (error) => {
          console.error(`Joint control API call failed: ${jointId}`, error)
          debugInfo.value = `Joint control failed: ${jointId} (${error.message}), please check backend connection`
        },
        successMessage: '', // We handle messages manually
        errorMessage: 'Joint control failed',
        errorContext: 'Update Joint',
        showSuccess: false,
        showError: false,
        notify: null,
        handleError: null
      })
    }

    // Adjust joint position by delta
    const adjustJoint = async (jointId, delta) => {
      const joint = jointList.value.find(j => j.id === jointId)
      if (!joint) {
        debugInfo.value = `Joint ${jointId} not found`
        return
      }
      
      const newValue = joint.value + delta
      // Clamp to min/max range
      const clampedValue = Math.max(joint.min, Math.min(joint.max, newValue))
      
      if (clampedValue !== joint.value) {
        joint.value = clampedValue
        await updateJoint(jointId, clampedValue)
      }
    }

    // Handle joint numeric input change
    const onJointInputChange = (jointId, event) => {
      // Debounce input to avoid too many API calls
      clearTimeout(jointInputDebounceTimer.value)
      jointInputDebounceTimer.value = setTimeout(() => {
        const value = parseFloat(event.target.value)
        if (!isNaN(value)) {
          updateJoint(jointId, value)
        }
      }, 500)
    }

    // Handle joint slider input (real-time feedback)
    const onJointSliderInput = (jointId, event) => {
      const value = parseFloat(event.target.value)
      const joint = jointList.value.find(j => j.id === jointId)
      if (joint && !isNaN(value)) {
        // Update local value for real-time feedback
        joint.value = value
      }
    }

    // Update joint speed
    const updateJointSpeed = async (jointId, speed) => {
      const joint = jointList.value.find(j => j.id === jointId)
      if (!joint) {
        debugInfo.value = `Joint ${jointId} not found`
        return
      }
      
      // Update local speed value
      if (joint.speed === undefined) {
        joint.speed = 50 // Default speed if not set
      }
      joint.speed = speed
      
      debugInfo.value = `Joint ${jointId} speed set to ${speed}%`
      
      // Send speed command to backend if API supports it
      return await performDataOperation('update-joint-speed', {
        apiClient: apiClient,
        apiMethod: 'post',
        apiEndpoint: '/api/robot/joint/speed',
        requestData: {
          jointId: jointId,
          speed: speed
        },
        onBeforeStart: () => {
          debugInfo.value = `Setting joint ${jointId} speed to ${speed}%...`
        },
        onSuccess: (responseData) => {
          if (responseData && responseData.success) {
            debugInfo.value = `Joint speed updated: ${jointId} = ${speed}%`
          } else {
            debugInfo.value = `Joint speed update failed: ${jointId}`
          }
        },
        onError: (error) => {
          console.error(`Failed to update joint speed: ${jointId}`, error)
          debugInfo.value = `Joint speed update failed: ${jointId} (${error.message})`
        },
        successMessage: '',
        errorMessage: 'Joint speed update failed',
        errorContext: 'Update Joint Speed',
        showSuccess: false,
        showError: false,
        notify: null,
        handleError: null
      })
    }

    // Apply joint preset
    const applyJointPreset = async (preset) => {
      debugInfo.value = `Applying joint preset: ${preset.name}`
      
      // Apply preset to all joints
      for (const joint of jointList.value) {
        let targetValue
        if (preset.positions[joint.id] !== undefined) {
          targetValue = preset.positions[joint.id]
        } else if (preset.positions.default !== undefined) {
          if (preset.positions.default === 'max') {
            targetValue = joint.max
          } else if (preset.positions.default === 'min') {
            targetValue = joint.min
          } else {
            targetValue = preset.positions.default
          }
        } else {
          continue
        }
        
        // Clamp to joint range
        const clampedValue = Math.max(joint.min, Math.min(joint.max, targetValue))
        
        // Update joint value
        joint.value = clampedValue
        
        // Send update command with delay to avoid overwhelming the API
        setTimeout(() => {
          updateJoint(joint.id, clampedValue)
        }, 100 * jointList.value.indexOf(joint))
      }
      
      debugInfo.value = `Joint preset "${preset.name}" applied`
    }

    // Connection related computed properties
    const activeConnectionsCount = computed(() => {
      return connectionList.value?.filter(conn => conn.enabled).length || 0
    })
    
    const totalConnectedDevices = computed(() => {
      let total = 0
      connectionList.value?.forEach(conn => {
        total += conn.connectedDevices?.length || 0
      })
      return total
    })
    
    // Connection management methods
    const addNewConnection = () => {
      const newId = `conn_${Date.now()}`
      const defaultConnectionType = availableConnectionTypes.value[0]
      
      connectionList.value.push({
        id: newId,
        type: defaultConnectionType.id,
        port: `${defaultConnectionType.id.toUpperCase()}${connectionList.value.length + 1}`,
        description: `New ${defaultConnectionType.name} connection`,
        autoReconnect: true,
        timeout: 5000,
        enabled: true,
        inputTypes: ['sensor', 'generic'],
        usbVersion: '3.0',
        maxBandwidth: '5 Gbps',
        baudRate: 115200,
        ipAddress: '192.168.1.100',
        connectedDevices: []
      })
      
      debugInfo.value = `Added new connection: ${defaultConnectionType.name} connection`
    }
    
    const removeConnection = (index) => {
      if (connectionList.value.length <= 1) {
        debugInfo.value = 'At least one connection must be kept'
        return
      }
      
      const removedConn = connectionList.value.splice(index, 1)[0]
      debugInfo.value = `Deleted connection: ${removedConn.description}`
    }
    
    const toggleConnectionEnabled = (index) => {
      // Use unified toggle system for array property toggling
      const toggle = createArrayPropertyToggle(
        connectionList,
        index,
        'enabled',
        'Connection'
      )
      return toggle()
    }
    
    const updateConnectionType = (index, event) => {
      const newType = event.target.value
      const conn = connectionList.value[index]
      conn.type = newType
      
      // Set default values based on connection type
      if (newType === 'usb') {
        conn.usbVersion = '3.0'
        conn.maxBandwidth = '5 Gbps'
        conn.inputTypes = ['sensor', 'camera', 'microphone']
      } else if (newType === 'serial') {
        conn.baudRate = 115200
        conn.parity = 'none'
        conn.inputTypes = ['motor', 'servo']
      } else if (newType === 'ethernet') {
        conn.ipAddress = '192.168.1.100'
        conn.subnetMask = '255.255.255.0'
        conn.inputTypes = ['network', 'display']
      }
      
      debugInfo.value = `Connection type updated to: ${newType}`
    }
    
    const testAllConnections = async () => {
      debugInfo.value = 'Testing all connections...'
      
      // Test server connection
      try {
        const response = await apiClient.get('/health', { timeout: 5000 })
        if (response.data && response.data.status === 'ok') {
          debugInfo.value = 'Server connection normal, testing other connections...'
        } else {
          debugInfo.value = 'Server connection abnormal'
          return
        }
      } catch (error) {
        debugInfo.value = `Server connection test failed: ${error.message}`
        return
      }
      
      // Display test status for each enabled connection
      const enabledConnections = connectionList.value.filter(conn => conn.enabled)
      if (enabledConnections.length === 0) {
        debugInfo.value = 'No enabled connections need testing'
        return
      }
      
      debugInfo.value = `Testing ${enabledConnections.length} connections...`
      
      // Real implementation: call connection test API for each connection
      const testResults = []
      let successfulTests = 0
      let failedTests = 0
      
      for (const conn of enabledConnections) {
        try {
          debugInfo.value += `\nTesting connection ${conn.description} (${conn.type}:${conn.port})...`
          
          // Call real connection test API
          const response = await apiClient.post('/api/robot/hardware/test_connection', {
            connectionId: conn.id,
            connectionType: conn.type,
            port: conn.port,
            testType: 'connectivity'
          })
          
          if (response.data && response.data.status === 'success') {
            const testResult = response.data.testResults
            testResults.push({
              connection: conn.description,
              status: 'success',
              message: testResult.overallStatus === 'success' ? 'Connection test passed' : 'Connection test failed',
              details: testResult.tests
            })
            
            if (testResult.overallStatus === 'success') {
              successfulTests++
              debugInfo.value += ' ✅'
            } else {
              failedTests++
              debugInfo.value += ' ❌'
            }
          } else {
            testResults.push({
              connection: conn.description,
              status: 'error',
              message: 'API returned error status',
              details: response.data
            })
            failedTests++
            debugInfo.value += ' ❌'
          }
          
        } catch (error) {
          console.error(`Connection test ${conn.description} failed:`, error)
          testResults.push({
            connection: conn.description,
            status: 'error',
            message: error.message || 'Unknown error',
            details: null
          })
          failedTests++
          debugInfo.value += ' ❌'
          
          // Continue testing other connections
          continue
        }
      }
      
      // Display test result summary
      debugInfo.value += `\n\nConnection test completed: server connection normal`
      debugInfo.value += `\nSuccessful: ${successfulTests}, Failed: ${failedTests}`
      
      // If detailed results exist, display in console
      if (testResults.length > 0) {
        console.log('Connection test detailed results:', testResults)
        
        // If failed connections exist, display in debug info
        const failedConnections = testResults.filter(r => r.status !== 'success')
        if (failedConnections.length > 0) {
          debugInfo.value += `\n\nFailed connections:`
          failedConnections.forEach(failed => {
            debugInfo.value += `\n  - ${failed.connection}: ${failed.message}`
          })
        }
      }
      
      // Update hardware status
      if (successfulTests > 0) {
        hardwareState.initialized = true
        debugInfo.value += `\n\n${successfulTests} connection tests successful, hardware ready`
      }
    }
    
    const enableAllConnections = () => {
      connectionList.value.forEach(conn => {
        conn.enabled = true
      })
      debugInfo.value = 'All connections enabled'
    }
    
    const disableAllConnections = () => {
      connectionList.value.forEach(conn => {
        conn.enabled = false
      })
      debugInfo.value = 'All connections disabled'
    }
    
    const saveConnectionsConfig = () => {
      try {
        const config = {
          connections: connectionList.value,
          lastUpdated: new Date().toISOString()
        }
        localStorage.setItem('robot_connections_config', JSON.stringify(config))
        debugInfo.value = `Connection configuration saved: ${connectionList.value.length} connections`
      } catch (error) {
        console.error('Failed to save connection configuration:', error)
        debugInfo.value = `Failed to save connection configuration: ${error.message}`
      }
    }

    // Initialize hardware
    const initializeHardware = async () => {
      return await performDataOperation('initialize-hardware', {
        apiClient: apiClient,
        apiMethod: 'post',
        apiEndpoint: '/api/robot/hardware/initialize',
        onBeforeStart: () => {
          hardwareState.loading = true
          debugInfo.value = 'Initializing hardware...'
        },
        onSuccess: (responseData, fullResponse) => {
          if (responseData && fullResponse.data && fullResponse.data.status === 'success') {
            hardwareState.initialized = true
            robotState.connected = true
            robotState.status = 'idle'
            robotState.statusText = 'Hardware connected'
            debugInfo.value = 'Hardware initialization successful'
          } else {
            debugInfo.value = 'Hardware initialization failed'
          }
        },
        onError: (error) => {
          console.error('Hardware initialization failed:', error)
          debugInfo.value = `Hardware initialization failed: ${error.message}`
        },
        onFinally: () => {
          hardwareState.loading = false
        },
        successMessage: '', // We handle messages manually
        errorMessage: 'Hardware initialization failed',
        errorContext: 'Initialize Hardware',
        showSuccess: false,
        showError: false,
        notify: null,
        handleError: null
      })
    }

    // Detect hardware
    const fetchDetectedHardware = async () => {
      return await performDataLoad('detected-hardware', {
        apiClient: apiClient,
        apiMethod: 'get',
        apiEndpoint: '/api/robot/hardware/detect',
        onBeforeStart: () => {
          debugInfo.value = 'Detecting hardware...'
        },
        onSuccess: (responseData, fullResponse) => {
          // Regardless of API success or error status, try to update hardware status
          if (responseData) {
            // Even if status is error, may contain hardware information
            hardwareState.detectedDevices = responseData.detectedDevices || { joints: 0, sensors: 0, cameras: 0 }
            hardwareState.detectedHardwareList = responseData.detectedHardwareList || { joints: [], sensors: [], cameras: [] }
            
            // Check API status
            const apiStatus = fullResponse.data?.status || 'unknown'
            if (apiStatus === 'success') {
              hardwareState.hardwareDetected = true
              debugInfo.value = `Hardware detected: ${responseData.detectedDevices?.joints || 0} joints, ${responseData.detectedDevices?.sensors || 0} sensors, ${responseData.detectedDevices?.cameras || 0} cameras`
            } else {
              hardwareState.hardwareDetected = false
              debugInfo.value = `Hardware detection returned status: ${apiStatus}. Real hardware connection required.`
            }
            
            // Initialize joint list (based on detected hardware)
            initJointList()
          }
        },
        onError: (error) => {
          console.error('Hardware detection failed:', error)
          debugInfo.value = `Hardware detection failed: ${error.message}`
        },
        successMessage: '', // We handle messages manually
        errorMessage: 'Hardware detection failed',
        errorContext: 'Fetch Detected Hardware',
        showSuccess: false,
        showError: false,
        notify: null,
        handleError: null,
        fallbackValue: null
      })
    }

    // Disconnect hardware
    const disconnectHardware = async () => {
      return await performDataOperation('disconnect-hardware', {
        apiClient: apiClient,
        apiMethod: 'post',
        apiEndpoint: '/api/robot/hardware/disconnect',
        onBeforeStart: () => {
          debugInfo.value = 'Disconnecting hardware...'
        },
        onSuccess: () => {
          hardwareState.initialized = false
          robotState.connected = false
          robotState.status = 'disconnected'
          robotState.statusText = 'Hardware disconnected'
          debugInfo.value = 'Hardware connection disconnected'
        },
        onError: (error) => {
          console.error('Failed to disconnect hardware:', error)
          debugInfo.value = `Failed to disconnect hardware: ${error.message}`
        },
        successMessage: '', // We handle messages manually
        errorMessage: 'Failed to disconnect hardware',
        errorContext: 'Disconnect Hardware',
        showSuccess: false,
        showError: false,
        notify: null,
        handleError: null
      })
    }

    // Toggle voice control - using unified toggle system
    const toggleVoiceControl = createReactiveToggle(
      voiceControlState,
      'enabled',
      'Voice control'
    )

    // Load sensor data
    const loadSensorData = async () => {
      return await performDataLoad('sensor-data', {
        apiClient: apiClient,
        apiMethod: 'get',
        apiEndpoint: '/api/robot/sensors',
        dataPath: 'data.data',
        onSuccess: (data) => {
          sensorData.value = data || []
        },
        onError: (error) => {
          console.error('Failed to load sensor data:', error)
          sensorData.value = []
        },
        successMessage: '', // We handle messages manually
        errorMessage: 'Failed to load sensor data',
        errorContext: 'Load Sensor Data',
        showSuccess: false,
        showError: false,
        notify: null,
        handleError: null,
        fallbackValue: []
      })
    }

    // Execute when component mounts
    onMounted(() => {
      if (import.meta.env.DEV) {
        console.log('RobotSettingsView component mounted')
      }
      isMounted.value = true
      try {
        initJointList()
        loadSensorData()
        fetchDetectedHardware()
        fetchCollaborationPatterns()
        initializeTrainingModule()
        loadSettingsData()
        loadDatasets()
      } catch (error) {
        console.error('Error during RobotSettingsView component mount:', error)
      }
    })

    // Clean up when component unmounts
    onUnmounted(() => {
      if (import.meta.env.DEV) {
        console.log('RobotSettingsView component unmounted')
      }
      isMounted.value = false
      
      // Clean up speech recognition instance
      if (speechRecognition.value) {
        try {
          speechRecognition.value.stop()
        } catch (error) {
          // Ignore errors during stop
        }
        speechRecognition.value = null
      }
      
      // Clean up training progress polling interval
      if (progressPollInterval) {
        clearInterval(progressPollInterval)
        progressPollInterval = null
      }
      
      // Clean up audio recording resources
      if (audioStream.value) {
        audioStream.value.getTracks().forEach(track => track.stop())
        audioStream.value = null
      }
      
      if (mediaRecorder.value) {
        try {
          mediaRecorder.value.stop()
        } catch (error) {
          // Ignore errors during stop
        }
        mediaRecorder.value = null
      }
    })

    // Setup dialog state and methods
    // Setup dialog display state
    const showSettingsDialog = ref(false)
    
    // Available sensor list (obtained from detected hardware)
    const availableSensors = ref([])
    
    // Sensor instance list (actually connected sensors)
    const sensorInstances = ref([])
    
    // Sensor configuration template
    const sensorConfig = reactive({
      imu: 'active',
      battery: 'monitor',
      force: 'active',
      torque: 'active',
      proximity: 'passive',
      temperature: 'monitor',
      accelerometer: 'active',
      gyroscope: 'active'
    })
    
    // Available camera list
    const availableCameras = ref([])
    
    // Camera instance list (actually connected cameras)
    const cameraInstances = ref([])
    
    // Camera configuration template
    const cameraConfig = reactive({
      mono_camera: { resolution: '1280x720', fps: '30' },
      stereo_camera: { resolution: '640x480', fps: '30' },
      depth_camera: { resolution: '640x480', fps: '15' },
      ir_camera: { resolution: '320x240', fps: '30' }
    })
    
    // Connection settings list (supports multiple connections)
    const connectionList = ref([])
    
    // Available connection types
    const availableConnectionTypes = ref([
      { id: 'usb', name: 'USB', maxPorts: 8, description: 'Universal Serial Bus, supports multiple devices' },
      { id: 'serial', name: 'Serial Port', maxPorts: 6, description: 'Serial communication interface' },
      { id: 'ethernet', name: 'Ethernet', maxPorts: 4, description: 'Network communication interface' },
      { id: 'bluetooth', name: 'Bluetooth', maxPorts: 2, description: 'Wireless Bluetooth communication' },
      { id: 'i2c', name: 'I2C', maxPorts: 8, description: 'Integrated Circuit Bus' },
      { id: 'spi', name: 'SPI', maxPorts: 4, description: 'Serial Peripheral Interface' },
      { id: 'can', name: 'CAN', maxPorts: 2, description: 'Controller Area Network bus' }
    ])
    
    // Available input types
    const availableInputTypes = ref([
      { id: 'sensor', name: 'Sensor Input', description: 'Various sensor data input' },
      { id: 'camera', name: 'Camera Input', description: 'Vision and image data input' },
      { id: 'microphone', name: 'Microphone Input', description: 'Audio and voice input' },
      { id: 'motor', name: 'Motor Control', description: 'Motor control and feedback' },
      { id: 'servo', name: 'Servo Control', description: 'Servo motor control' },
      { id: 'network', name: 'Network Data', description: 'Network communication data' },
      { id: 'display', name: 'Display Output', description: 'Display device output' },
      { id: 'audio', name: 'Audio Output', description: 'Audio device output' },
      { id: 'generic', name: 'Generic Input', description: 'Generic data input' }
    ])
    
    // Backward compatibility: single connection settings (for existing code)
    const connectionSettings = reactive({
      type: 'usb',
      autoReconnect: true,
      timeout: 5000
    })
    
    // Available device types
    const availableDeviceTypes = ref([
      { id: 'motor', name: 'Motor Controller', count: 1, maxCount: 12, description: 'Controls motor speed and position' },
      { id: 'servo', name: 'Servo Driver', count: 1, maxCount: 20, description: 'High-precision position control servo' },
      { id: 'led', name: 'LED Controller', count: 0, maxCount: 8, description: 'Controls LED strips and indicator lights' },
      { id: 'display', name: 'Display Device', count: 0, maxCount: 3, description: 'LCD/OLED display screen' },
      { id: 'microphone', name: 'Microphone', count: 1, maxCount: 4, description: 'Audio input for speech recognition and recording' },
      { id: 'speaker', name: 'Speaker', count: 1, maxCount: 4, description: 'Audio output for voice feedback and alert sounds' },
      { id: 'network', name: 'Network Device', count: 1, maxCount: 2, description: 'Network communication and wireless connectivity' },
      { id: 'usb_hub', name: 'USB Hub', count: 1, maxCount: 4, description: 'Expands USB ports, connects multiple USB devices' },
      { id: 'serial_adapter', name: 'Serial Adapter', count: 0, maxCount: 6, description: 'Serial communication converter' },
      { id: 'bluetooth_adapter', name: 'Bluetooth Adapter', count: 0, maxCount: 2, description: 'Wireless Bluetooth communication' }
    ])
    
    // Device instance list (actually connected devices)
    const deviceInstances = ref([])
    
    // Device type configuration template
    const deviceTypeConfig = reactive({
      motor: { nature: 'actuator', protocol: 'uart' },
      servo: { nature: 'actuator', protocol: 'uart' },
      led: { nature: 'output', protocol: 'i2c' },
      display: { nature: 'output', protocol: 'spi' },
      microphone: { nature: 'input', protocol: 'usb' },
      speaker: { nature: 'output', protocol: 'usb' },
      network: { nature: 'bidirectional', protocol: 'ethernet' },
      usb_hub: { nature: 'bidirectional', protocol: 'usb' },
      serial_adapter: { nature: 'bidirectional', protocol: 'uart' },
      bluetooth_adapter: { nature: 'bidirectional', protocol: 'bluetooth' }
    })
    
    // Connection status
    const connectionStatusText = ref('Not connected')
    const connectionStatusClass = ref('disconnected')
    const connectedDevicesCount = ref(0)
    
    // Joint management dialog state
    const showJointManagementDialog = ref(false)
    
    // New joint form data
    const newJoint = reactive({
      name: '',
      id: '',
      min: -90,
      max: 90,
      step: 1,
      value: 0
    })
    
    // Edit mode state
    const editingJointId = ref('')
    
    // Batch setting values
    const batchMin = ref(-90)
    const batchMax = ref(90)
    const batchStep = ref(1)
    
    // Open settings dialog
    const openSettingsDialog = () => {
      showSettingsDialog.value = true
      debugInfo.value = 'Opened settings dialog'
    }
    
    // Close settings dialog
    const closeSettingsDialog = () => {
      showSettingsDialog.value = false
      debugInfo.value = 'Closed settings dialog'
    }
    
    // Apply settings
    const applySettings = () => {
      debugInfo.value = 'Applying settings...'
      
      // Save settings to local storage
      try {
        const settingsData = {
          selectedSensors: selectedSensors.value,
          selectedCameras: selectedCameras.value,
          selectedDeviceTypes: selectedDeviceTypes.value,
          sensorConfig: sensorConfig,
          cameraConfig: cameraConfig,
          connectionSettings: connectionSettings,
          deviceTypeConfig: deviceTypeConfig
        }
        
        localStorage.setItem('robot_hardware_settings', JSON.stringify(settingsData))
        debugInfo.value = `Settings saved: ${selectedSensors.value.length} sensors, ${selectedCameras.value.length} cameras, ${selectedDeviceTypes.value.length} device types`
        showSettingsDialog.value = false
      } catch (error) {
        console.error('Failed to save settings:', error)
        debugInfo.value = `Failed to save settings: ${error.message}`
      }
    }
    
    // Reset settings
    const resetSettings = () => {
      selectedSensors.value = ['imu', 'battery', 'force']
      selectedCameras.value = ['camera_0']
      selectedDeviceTypes.value = ['motor', 'servo']
      connectionSettings.type = 'usb'
      connectionSettings.autoReconnect = true
      connectionSettings.timeout = 5000
      // Clear saved hardware settings
      try {
        localStorage.removeItem('robot_hardware_settings')
        debugInfo.value = 'Settings reset to default values, saved configuration cleared'
      } catch (error) {
        console.error('Failed to clear hardware settings:', error)
        debugInfo.value = 'Settings reset to default values'
      }
    }
    
    // Force reconnect
    const forceReconnect = async () => {
      debugInfo.value = 'Force reconnecting...'
      connectionStatusText.value = 'Reconnecting'
      connectionStatusClass.value = 'connecting'
      
      try {
        // First disconnect
        await disconnectHardware()
        debugInfo.value = 'Disconnected, reinitializing...'
        
        // Wait for some time
        await new Promise(resolve => setTimeout(resolve, 500))
        
        // Reinitialize hardware
        await initializeHardware()
        
        // Update connection status
        connectionStatusText.value = 'Connected'
        connectionStatusClass.value = 'connected'
        
        // Calculate connected device count
        const sensorCount = selectedSensors.value.length
        const cameraCount = selectedCameras.value.length
        const deviceCount = selectedDeviceTypes.value.length
        connectedDevicesCount.value = sensorCount + cameraCount + deviceCount
        
        debugInfo.value = `Reconnect successful, connected devices: ${connectedDevicesCount.value}`
      } catch (error) {
        console.error('Force reconnect failed:', error)
        connectionStatusText.value = 'Reconnect failed'
        connectionStatusClass.value = 'disconnected'
        debugInfo.value = `Force reconnect failed: ${error.message}`
      }
    }
    
    // Disconnect all connections
    const disconnectAll = async () => {
      debugInfo.value = 'Disconnecting all connections...'
      connectionStatusText.value = 'Disconnected'
      connectionStatusClass.value = 'disconnected'
      connectedDevicesCount.value = 0
      
      // Call hardware disconnect API
      try {
        await disconnectHardware()
        debugInfo.value = 'All connections disconnected'
      } catch (error) {
        // Update local state even if API call fails
        console.error('Failed to disconnect all connections:', error)
        debugInfo.value = `Failed to disconnect all connections: ${error.message}`
      }
    }
    
    // Test connection
    const testConnection = async () => {
      return await performDataLoad('test-connection', {
        apiCall: () => apiClient.get('/health', { timeout: 5000 }),
        onBeforeStart: () => {
          debugInfo.value = 'Testing connection...'
        },
        onSuccess: (responseData, fullResponse) => {
          if (responseData && fullResponse.data && fullResponse.data.status === 'ok') {
            debugInfo.value = 'Connection test successful: server running normally'
          } else {
            debugInfo.value = 'Connection test failed: server returned abnormal status'
          }
        },
        onError: (error) => {
          console.error('Connection test failed:', error)
          debugInfo.value = `Connection test failed: ${error.message}`
        },
        successMessage: '', // We handle messages manually
        errorMessage: 'Connection test failed',
        errorContext: 'Test Connection',
        showSuccess: false,
        showError: false,
        notify: null,
        handleError: null,
        fallbackValue: null
      })
    }
    
    // Save connection configuration
    const saveConnectionProfile = () => {
      debugInfo.value = 'Saving connection configuration...'
      try {
        const config = {
          connections: connectionList.value,
          lastUpdated: new Date().toISOString(),
          profileName: 'Default connection configuration'
        }
        localStorage.setItem('robot_connection_profile', JSON.stringify(config))
        debugInfo.value = `Connection configuration saved: ${connectionList.value.length} connections`
      } catch (error) {
        console.error('Failed to save connection configuration:', error)
        debugInfo.value = `Failed to save connection configuration: ${error.message}`
      }
    }
    
    // Load settings data
    const loadSettingsData = () => {
      debugInfo.value = 'Loading settings data'
      
      // Load hardware settings from local storage
      try {
        const savedSettings = localStorage.getItem('robot_hardware_settings')
        if (savedSettings) {
          const settingsData = JSON.parse(savedSettings)
          
          // Apply saved settings
          if (settingsData.selectedSensors) {
            selectedSensors.value = settingsData.selectedSensors
          }
          if (settingsData.selectedCameras) {
            selectedCameras.value = settingsData.selectedCameras
          }
          if (settingsData.selectedDeviceTypes) {
            selectedDeviceTypes.value = settingsData.selectedDeviceTypes
          }
          // Note: reactive objects like sensorConfig require special handling
          
          debugInfo.value = 'Hardware settings loaded from local storage'
        } else {
          debugInfo.value = 'No saved hardware settings found, using default values'
        }
      } catch (error) {
        console.error('Failed to load settings data:', error)
        debugInfo.value = `Failed to load settings data: ${error.message}`
       }
       
       // Load joint configuration from local storage
       try {
         const savedJoints = localStorage.getItem('robot_joint_configuration')
         if (savedJoints) {
           const jointConfig = JSON.parse(savedJoints)
           if (jointConfig?.joints && Array.isArray(jointConfig.joints)) {
             jointList.value = jointConfig.joints
             debugInfo.value = `Joint configuration loaded: ${jointList.value.length} joints`
           }
         }
       } catch (error) {
         console.error('Failed to load joint configuration:', error)
         // Ignore error, use default joint configuration
       }
     }

    // Chat dialog related
    const showChatDialog = ref(false)
    const chatMessages = ref([])
    const chatInputText = ref('')
    const voiceInputActive = ref(false)
    const chatMessagesContainer = ref(null)

    // Open chat dialog
    const openChatDialog = () => {
      showChatDialog.value = true
      debugInfo.value = 'Chat dialog opened'
      // Scroll to bottom
      setTimeout(() => {
        if (isMounted.value && chatMessagesContainer.value) {
          chatMessagesContainer.value.scrollTop = chatMessagesContainer.value.scrollHeight
        }
      }, 100)
    }

    // Close chat dialog
    const closeChatDialog = () => {
      showChatDialog.value = false
      debugInfo.value = 'Chat dialog closed'
    }

    // Send chat message
    const sendChatMessage = () => {
      if (!chatInputText.value.trim()) return
      
      const userMessage = chatInputText.value.trim()
      const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      
      // Add user message
      chatMessages.value.push({
        text: userMessage,
        type: 'user',
        time: time
      })
      
      // Clear input field
      chatInputText.value = ''
      
      // Scroll to bottom
      setTimeout(() => {
        if (isMounted.value && chatMessagesContainer.value) {
          chatMessagesContainer.value.scrollTop = chatMessagesContainer.value.scrollHeight
        }
      }, 100)
      
      debugInfo.value = `Message sent: ${userMessage.substring(0, 30)}${userMessage.length > 30 ? '...' : ''}`
      
      // Call real backend API to get bot response
      const getBotResponse = async () => {
        try {
          debugInfo.value = 'Getting bot response...'
          const response = await apiClient.post('/api/chat', {
            message: userMessage,
            text: userMessage,
            conversation_history: chatMessages.value.map(msg => ({
              text: msg.text,
              type: msg.type
            }))
          })
          
          const botTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          let botResponseText = ''
          
          if (response.data && response.data.data && response.data.data.response) {
            botResponseText = response.data.data.response
          } else if (response.data && response.data.response) {
            botResponseText = response.data.response
          } else {
            botResponseText = `I received your message: "${userMessage}".`
          }
          
          chatMessages.value.push({
            text: botResponseText,
            type: 'bot',
            time: botTime
          })
          
          debugInfo.value = 'Received bot response'
          
        } catch (error) {
          console.error('Chat API call failed:', error)
          const botTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          
          // Real error handling - no mock response
          const errorMessage = `Sorry, an error occurred while connecting to the server: ${error.message || 'unknown error'}. Please check if the backend service is running on port 8000.`
          
          chatMessages.value.push({
            text: errorMessage,
            type: 'system',
            time: botTime
          })
          
          debugInfo.value = 'Chat API call failed'
        }
        
        // Scroll to bottom
        setTimeout(() => {
          if (isMounted.value && chatMessagesContainer.value) {
            chatMessagesContainer.value.scrollTop = chatMessagesContainer.value.scrollHeight
          }
        }, 100)
      }
      
      // Call API to get response
      getBotResponse()
    }

    // Clear chat history
    const clearChat = () => {
      chatMessages.value = [
        { text: 'Chat history cleared. I am a robot assistant, how can I help you?', type: 'bot', time: 'just now' }
      ]
      debugInfo.value = 'Chat history cleared'
    }

    // ========== Voice Command Recognition System ==========
    
    // Voice command vocabulary (support both English and Chinese)
    const voiceCommands = {
      // Movement commands
      'forward': ['forward', 'go forward', 'move forward'],
      'backward': ['backward', 'go backward', 'move backward'],
      'left': ['left', 'turn left', 'rotate left'],
      'right': ['right', 'turn right', 'rotate right'],
      'stop': ['stop', 'halt'],
      
      // Speed control
      'speed up': ['speed up', 'faster'],
      'slow down': ['slow down', 'slower'],
      
      // Navigation commands
      'go to waypoint': ['go to waypoint', 'navigate to waypoint'],
      'add waypoint': ['add waypoint', 'create waypoint'],
      'clear waypoints': ['clear waypoints', 'remove all waypoints'],
      
      // Robot control
      'enable collision detection': ['enable collision detection', 'turn on collision detection'],
      'disable collision detection': ['disable collision detection', 'turn off collision detection'],
      'reset position': ['reset position', 'go home'],
      
      // System commands
      'status': ['status', 'robot status'],
      'battery': ['battery', 'power level'],
      'help': ['help', 'what can you do']
    }
    
    // Parse voice command and execute corresponding action
    const parseAndExecuteVoiceCommand = (transcript) => {
      const normalizedTranscript = transcript.toLowerCase().trim()
      
      // Find matching command
      for (const [command, keywords] of Object.entries(voiceCommands)) {
        for (const keyword of keywords) {
          if (normalizedTranscript.includes(keyword.toLowerCase())) {
            executeVoiceCommand(command, transcript)
            return true // Command was recognized and executed
          }
        }
      }
      
      // No command recognized, treat as chat message
      return false
    }
    
    // Execute voice command
    const executeVoiceCommand = (command, originalTranscript) => {
      debugInfo.value = `Voice command recognized: "${command}" from "${originalTranscript}"`
      
      switch (command) {
        case 'forward':
          moveRobot('forward')
          break
        case 'backward':
          moveRobot('backward')
          break
        case 'left':
          rotateRobot('left')
          break
        case 'right':
          rotateRobot('right')
          break
        case 'stop':
          stopRobot()
          break
        case 'speed up':
          movementSpeed.value = Math.min(100, movementSpeed.value + 20)
          debugInfo.value = `Speed increased to ${movementSpeed.value}%`
          break
        case 'slow down':
          movementSpeed.value = Math.max(0, movementSpeed.value - 20)
          debugInfo.value = `Speed decreased to ${movementSpeed.value}%`
          break
        case 'go to waypoint':
          if (waypoints.value.length > 0) {
            executeWaypointNavigation()
          } else {
            debugInfo.value = 'No waypoints defined. Please add waypoints first.'
          }
          break
        case 'add waypoint':
          // Use current position or default values
          addWaypoint()
          break
        case 'clear waypoints':
          clearWaypoints()
          break
        case 'enable collision detection':
          collisionDetectionState.enabled = true
          debugInfo.value = 'Collision detection enabled'
          break
        case 'disable collision detection':
          collisionDetectionState.enabled = false
          debugInfo.value = 'Collision detection disabled'
          break
        case 'reset position':
          resetRobotPosition()
          break
        case 'status':
          debugInfo.value = `Robot status: ${robotState.statusText}, Battery: ${robotState.battery}%, Connected: ${robotState.connected}`
          break
        case 'battery':
          debugInfo.value = `Battery level: ${robotState.battery}%`
          break
        case 'help':
          const availableCommands = Object.keys(voiceCommands).join(', ')
          debugInfo.value = `Available voice commands: ${availableCommands}. Try saying "forward", "stop", or "status".`
          break
        default:
          debugInfo.value = `Command "${command}" recognized but not yet implemented`
      }
    }
    
    // Reset robot position (simulated)
    const resetRobotPosition = () => {
      debugInfo.value = 'Resetting robot position...'
      movementHistory.value = []
      if (trajectoryCanvas.value) {
        const ctx = trajectoryCanvas.value.getContext('2d')
        ctx.clearRect(0, 0, trajectoryCanvas.value.width, trajectoryCanvas.value.height)
      }
      // Add a reset position to history
      movementHistory.value.unshift({
        position: { x: 0, y: 0, z: 0 },
        timestamp: new Date().toISOString(),
        type: 'reset'
      })
      debugInfo.value = 'Robot position reset to origin'
    }
    
    // Show voice commands help
    const showVoiceCommandsHelp = () => {
      let helpMessage = 'Available Voice Commands:\n\n'
      
      // Group commands by category for better organization
      const commandCategories = {
        'Movement': ['forward', 'backward', 'left', 'right', 'stop'],
        'Speed Control': ['speed up', 'slow down'],
        'Navigation': ['go to waypoint', 'add waypoint', 'clear waypoints'],
        'Robot Control': ['enable collision detection', 'disable collision detection', 'reset position'],
        'System': ['status', 'battery', 'help']
      }
      
      for (const [category, commands] of Object.entries(commandCategories)) {
        helpMessage += `${category}:\n`
        for (const command of commands) {
          const keywords = voiceCommands[command]
          helpMessage += `  • ${command}: ${keywords.slice(0, 3).join(', ')}${keywords.length > 3 ? '...' : ''}\n`
        }
        helpMessage += '\n'
      }
      
      helpMessage += 'Examples:\n'
      helpMessage += '• "Robot, move forward"\n'
      helpMessage += '• "Turn left"\n'
      helpMessage += '• "What is the battery status?"\n'
      helpMessage += '• "Enable collision detection"\n'
      helpMessage += '\nSay "help" at any time to hear available commands.'
      
      alert(helpMessage)
      debugInfo.value = 'Voice commands help displayed'
    }

    // Enhanced speech recognition system
    const speechRecognition = ref(null)
    const mediaRecorder = ref(null)
    const audioChunks = ref([])
    const audioStream = ref(null)
    const isRecording = ref(false)
    const useBackendRecognition = ref(false)
    
    // Initialize enhanced speech recognition
    const initSpeechRecognition = () => {
      // First try: Browser built-in speech recognition
      if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
        speechRecognition.value = new SpeechRecognition()
        speechRecognition.value.continuous = false
        speechRecognition.value.interimResults = false
        speechRecognition.value.lang = 'en-US'
        
        speechRecognition.value.onresult = (event) => {
          const transcript = event.results[0][0].transcript
          
          // Try to parse as voice command first
          const isCommand = parseAndExecuteVoiceCommand(transcript)
          
          if (!isCommand) {
            // If not a command, treat as chat input
            chatInputText.value = transcript
            debugInfo.value = `Speech recognition completed: "${transcript.slice(0, 50)}${transcript.length > 50 ? '...' : ''}"`
          }
        }
        
        speechRecognition.value.onerror = (event) => {
          console.error('Speech recognition error:', event.error)
          debugInfo.value = `Speech recognition error: ${event.error}`
          voiceInputActive.value = false
          // If browser recognition fails, try backend API
          if (!useBackendRecognition.value) {
            debugInfo.value = 'Browser recognition failed, trying backend API...'
            useBackendRecognition.value = true
          }
        }
        
        speechRecognition.value.onend = () => {
          voiceInputActive.value = false
        }
        
        return true
      } else {
        // Browser doesn't support speech recognition, use backend API
        useBackendRecognition.value = true
        debugInfo.value = 'Browser speech recognition not available, using backend API'
        return true
      }
    }
    
    // Start recording audio for backend recognition
    const startAudioRecording = async () => {
      try {
        debugInfo.value = 'Requesting microphone access...'
        const stream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            channelCount: 1,
            sampleRate: 16000,
            echoCancellation: true,
            noiseSuppression: true
          } 
        })
        
        audioStream.value = stream
        mediaRecorder.value = new MediaRecorder(stream, {
          mimeType: 'audio/webm;codecs=opus'
        })
        
        audioChunks.value = []
        
        mediaRecorder.value.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunks.value.push(event.data)
          }
        }
        
        mediaRecorder.value.onstop = async () => {
          debugInfo.value = 'Processing audio...'
          isRecording.value = false
          
          // Combine audio chunks
          const audioBlob = new Blob(audioChunks.value, { type: 'audio/webm' })
          
          try {
            // Send audio to backend API for speech recognition
            const formData = new FormData()
            formData.append('audio', audioBlob, 'recording.webm')
            formData.append('language', 'en-US')
            formData.append('session_id', `robot_${Date.now()}`)
            
            const response = await api.process.audio({
              audio: await audioBlob.arrayBuffer(),
              language: 'en-US',
              session_id: `robot_${Date.now()}`,
              model_id: 'audio'
            })
            
            if (response && response.text) {
              const transcript = response.text
              // Try to parse as voice command first
              const isCommand = parseAndExecuteVoiceCommand(transcript)
              
              if (!isCommand) {
                // If not a command, treat as chat input
                chatInputText.value = transcript
                debugInfo.value = `Backend speech recognition: "${transcript.slice(0, 50)}${transcript.length > 50 ? '...' : ''}"`
              }
            } else if (response && response.result) {
              const transcript = response.result
              // Try to parse as voice command first
              const isCommand = parseAndExecuteVoiceCommand(transcript)
              
              if (!isCommand) {
                // If not a command, treat as chat input
                chatInputText.value = transcript
                debugInfo.value = `Backend speech recognition: "${transcript.slice(0, 50)}${transcript.length > 50 ? '...' : ''}"`
              }
            } else {
              debugInfo.value = 'Speech recognition completed but no text returned'
            }
          } catch (apiError) {
            console.error('Backend speech recognition API error:', apiError)
            debugInfo.value = 'Backend speech recognition failed, please use text input'
          } finally {
            // Clean up
            if (audioStream.value) {
              audioStream.value.getTracks().forEach(track => track.stop())
              audioStream.value = null
            }
          }
        }
        
        mediaRecorder.value.start(100) // Collect data every 100ms
        isRecording.value = true
        debugInfo.value = 'Recording audio... Speak now'
        return true
      } catch (error) {
        console.error('Failed to start audio recording:', error)
        debugInfo.value = 'Microphone access denied or failed. Please use text input.'
        voiceInputActive.value = false
        return false
      }
    }
    
    // Stop audio recording
    const stopAudioRecording = () => {
      if (mediaRecorder.value && isRecording.value) {
        mediaRecorder.value.stop()
        isRecording.value = false
      }
      
      if (audioStream.value) {
        audioStream.value.getTracks().forEach(track => track.stop())
        audioStream.value = null
      }
    }
    
    // Enhanced voice input toggle
    const toggleVoiceInput = () => {
      if (!voiceInputActive.value) {
        // Start voice input
        voiceInputActive.value = true
        
        // Initialize speech recognition if not already done
        if (!speechRecognition.value && !useBackendRecognition.value) {
          const supported = initSpeechRecognition()
          if (!supported) {
            debugInfo.value = 'Speech recognition initialization failed'
            voiceInputActive.value = false
            return
          }
        }
        
        // Use appropriate recognition method
        if (!useBackendRecognition.value && speechRecognition.value) {
          // Use browser speech recognition
          try {
            speechRecognition.value.start()
            debugInfo.value = 'Voice input activated (browser recognition), please start speaking...'
          } catch (error) {
            console.error('Failed to start browser speech recognition:', error)
            debugInfo.value = 'Browser recognition failed, switching to backend API...'
            useBackendRecognition.value = true
            // Retry with backend API
            startAudioRecording()
          }
        } else {
          // Use backend API with audio recording
          startAudioRecording()
        }
      } else {
        // Stop voice input
        if (speechRecognition.value && !useBackendRecognition.value) {
          try {
            speechRecognition.value.stop()
          } catch (error) {
            console.error('Failed to stop speech recognition:', error)
          }
        } else if (isRecording.value) {
          stopAudioRecording()
        }
        
        voiceInputActive.value = false
        debugInfo.value = 'Voice input stopped'
      }
    }

    // Export chat
    const exportChat = () => {
      const chatText = chatMessages.value.map(msg => `[${msg.time}] ${msg.type === 'user' ? 'You' : 'Robot'}: ${msg.text}`).join('\n')
      const blob = new Blob([chatText], { type: 'text/plain' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `robot_chat_${new Date().toISOString().slice(0, 10)}.txt`
      a.click()
      URL.revokeObjectURL(url)
      debugInfo.value = 'Chat exported as text file'
    }

    // Load settings when component mounts
    onMounted(() => {
      loadSettingsData()
    })

    // Joint management functions
    // Open joint management dialog
    const openJointManagementDialog = () => {
      showJointManagementDialog.value = true
      debugInfo.value = 'Opened joint management dialog'
    }
    
    // Close joint management dialog
    const closeJointManagementDialog = () => {
      showJointManagementDialog.value = false
      debugInfo.value = 'Closed joint management dialog'
      // Clear edit mode
      editingJointId.value = ''
      resetNewJointForm()
    }
    
    // Edit joint
    const editJoint = (jointId) => {
      const joint = jointList.value.find(j => j.id === jointId)
      if (joint) {
        // Copy joint data to edit form
        newJoint.name = joint.name
        newJoint.id = joint.id
        newJoint.min = joint.min
        newJoint.max = joint.max
        newJoint.step = joint.step
        newJoint.value = joint.value
        // Set edit mode
        editingJointId.value = jointId
        debugInfo.value = `Preparing to edit joint: ${joint.name}`
      }
    }
    
    // Delete joint
    const removeJoint = (jointId) => {
      const index = jointList.value.findIndex(j => j.id === jointId)
      if (index !== -1) {
        const jointName = jointList.value[index].name
        jointList.value.splice(index, 1)
        debugInfo.value = `Deleted joint: ${jointName}`
      }
    }
    
    // Helper function: Check if joint ID is unique
    const isJointIdUnique = (jointId, excludeId = '') => {
      if (excludeId) {
        return !jointList.value.some(j => j.id === jointId && j.id !== excludeId)
      }
      return !jointList.value.some(j => j.id === jointId)
    }
    
    // Add new joint
    const addNewJoint = () => {
      if (!newJoint.name.trim() || !newJoint.id.trim()) {
        debugInfo.value = 'Please fill in joint name and ID'
        return
      }
      
      if (newJoint.min >= newJoint.max) {
        debugInfo.value = 'Minimum angle must be less than maximum angle'
        return
      }
      
      if (newJoint.value < newJoint.min || newJoint.value > newJoint.max) {
        debugInfo.value = 'Initial value must be within angle range'
        return
      }
      
      // Check if in edit mode
      if (editingJointId.value) {
        // Edit mode: Update existing joint
        const jointIndex = jointList.value.findIndex(j => j.id === editingJointId.value)
        if (jointIndex !== -1) {
          // If ID is modified, check if new ID conflicts with other joints (except the currently edited joint)
          if (newJoint.id !== editingJointId.value && !isJointIdUnique(newJoint.id, editingJointId.value)) {
            debugInfo.value = `Joint ID "${newJoint.id}" already exists`
            return
          }
          
          // Update joint
          jointList.value[jointIndex] = {
            id: newJoint.id,
            name: newJoint.name,
            min: newJoint.min,
            max: newJoint.max,
            step: newJoint.step,
            value: newJoint.value
          }
          
          debugInfo.value = `Updated joint: ${newJoint.name} (${newJoint.min}° to ${newJoint.max}°)`
          editingJointId.value = ''
          resetNewJointForm()
        }
      } else {
        // Add mode: Check if ID already exists
        if (!isJointIdUnique(newJoint.id)) {
          debugInfo.value = `Joint ID "${newJoint.id}" already exists`
          return
        }
        
        // Add new joint
        jointList.value.push({
          id: newJoint.id,
          name: newJoint.name,
          min: newJoint.min,
          max: newJoint.max,
          step: newJoint.step,
          value: newJoint.value
        })
        
        debugInfo.value = `Added new joint: ${newJoint.name} (${newJoint.min}° to ${newJoint.max}°)`
        resetNewJointForm()
      }
    }
    
    // Reset new joint form
    const resetNewJointForm = () => {
      newJoint.name = ''
      newJoint.id = ''
      newJoint.min = -90
      newJoint.max = 90
      newJoint.step = 1
      newJoint.value = 0
      editingJointId.value = ''
    }
    
    // Apply batch minimum angle
    const applyBatchMin = () => {
      jointList.value.forEach(joint => {
        joint.min = batchMin.value
      })
      debugInfo.value = `Set minimum angle for all joints to ${batchMin.value}°`
    }
    
    // Apply batch maximum angle
    const applyBatchMax = () => {
      jointList.value.forEach(joint => {
        joint.max = batchMax.value
      })
      debugInfo.value = `Set maximum angle for all joints to ${batchMax.value}°`
    }
    
    // Apply batch step size
    const applyBatchStep = () => {
      jointList.value.forEach(joint => {
        joint.step = batchStep.value
      })
      debugInfo.value = `Set step size for all joints to ${batchStep.value}`
    }
    
    // Save joint configuration
    const saveJointConfiguration = () => {
      debugInfo.value = 'Saving joint configuration...'
      
      // Save joint configuration to local storage
      try {
        const jointConfiguration = {
          joints: jointList.value,
          lastUpdated: new Date().toISOString()
        }
        
        localStorage.setItem('robot_joint_configuration', JSON.stringify(jointConfiguration))
        debugInfo.value = `Joint configuration saved: ${jointList.value.length} joints`
        showJointManagementDialog.value = false
      } catch (error) {
        console.error('Failed to save joint configuration:', error)
        debugInfo.value = `Failed to save joint configuration: ${error.message}`
      }
    }
    
    // Device instance getter method
    const getSensorInstances = (sensorType) => {
      return sensorInstances.value.filter(instance => instance.type === sensorType)
    }
    
    const getCameraInstances = (cameraType) => {
      return cameraInstances.value.filter(instance => instance.type === cameraType)
    }
    
    const getDeviceInstances = (deviceType) => {
      return deviceInstances.value.filter(instance => instance.type === deviceType)
    }
    
    // Increase device count
    const increaseDeviceCount = (deviceCategory, deviceId) => {
      let deviceList, instanceList, configTemplate, categoryName
      
      if (deviceCategory === 'sensor') {
        deviceList = availableSensors.value
        instanceList = sensorInstances.value
        configTemplate = sensorConfig
        categoryName = 'Sensor'
      } else if (deviceCategory === 'camera') {
        deviceList = availableCameras.value
        instanceList = cameraInstances.value
        configTemplate = cameraConfig
        categoryName = 'Camera'
      } else if (deviceCategory === 'device') {
        deviceList = availableDeviceTypes.value
        instanceList = deviceInstances.value
        configTemplate = deviceTypeConfig
        categoryName = 'Device'
      } else {
        debugInfo.value = 'Unknown device category'
        return
      }
      
      const device = deviceList.find(d => d.id === deviceId)
      if (!device) {
        debugInfo.value = `Device not found: ${deviceId}`
        return
      }
      
      if (device.count >= device.maxCount) {
        debugInfo.value = `Maximum count reached: ${device.maxCount}`
        return
      }
      
      // Increase count
      device.count++
      
      // Create new instance
      const newInstanceId = `${deviceId}_${device.count}`
      let newInstance
      
      if (deviceCategory === 'sensor') {
        newInstance = {
          id: newInstanceId,
          type: deviceId,
          config: configTemplate[deviceId],
          description: `${device.name} #${device.count}`
        }
        instanceList.push(newInstance)
      } else if (deviceCategory === 'camera') {
        newInstance = {
          id: newInstanceId,
          type: deviceId,
          resolution: configTemplate[deviceId].resolution,
          fps: configTemplate[deviceId].fps,
          description: `${device.name} #${device.count}`
        }
        instanceList.push(newInstance)
      } else if (deviceCategory === 'device') {
        newInstance = {
          id: newInstanceId,
          type: deviceId,
          nature: configTemplate[deviceId].nature,
          protocol: configTemplate[deviceId].protocol,
          description: `${device.name} #${device.count}`
        }
        instanceList.push(newInstance)
      }
      
      debugInfo.value = `Added ${categoryName}: ${device.name} #${device.count}`
      updateConnectedDevicesCount()
    }
    
    // Decrease device count
    const decreaseDeviceCount = (deviceCategory, deviceId) => {
      let deviceList, instanceList, categoryName
      
      if (deviceCategory === 'sensor') {
        deviceList = availableSensors.value
        instanceList = sensorInstances.value
        categoryName = 'Sensor'
      } else if (deviceCategory === 'camera') {
        deviceList = availableCameras.value
        instanceList = cameraInstances.value
        categoryName = 'Camera'
      } else if (deviceCategory === 'device') {
        deviceList = availableDeviceTypes.value
        instanceList = deviceInstances.value
        categoryName = 'Device'
      } else {
        debugInfo.value = 'Unknown device category'
        return
      }
      
      const device = deviceList.find(d => d.id === deviceId)
      if (!device) {
        debugInfo.value = `Device not found: ${deviceId}`
        return
      }
      
      if (device.count <= 0) {
        debugInfo.value = `Count is already 0, cannot decrease`
        return
      }
      
      // Decrease count
      device.count--
      
      // Delete last instance
      const instanceIndex = instanceList.findIndex(inst => inst.type === deviceId)
      if (instanceIndex !== -1) {
        const removedInstance = instanceList.splice(instanceIndex, 1)[0]
        debugInfo.value = `Removed ${categoryName}: ${removedInstance.id}`
      }
      
      updateConnectedDevicesCount()
    }
    
    // Update connected device count
    const updateConnectedDevicesCount = () => {
      const totalDevices = sensorInstances.value.length + cameraInstances.value.length + deviceInstances.value.length
      connectedDevicesCount.value = totalDevices
      debugInfo.value = `Connected device count updated: ${totalDevices}`
    }
    
    // Reset joint configuration to default
    const resetJointConfiguration = () => {
      initJointList()
      // Clear saved joint configuration
      try {
        localStorage.removeItem('robot_joint_configuration')
        debugInfo.value = 'Joint configuration reset to default, saved configuration cleared'
      } catch (error) {
        console.error('Failed to clear joint configuration:', error)
        debugInfo.value = 'Joint configuration reset to default'
      }
    }
    
    // Scroll to training section if query parameter is set
    onMounted(() => {
      if (route.query.section === 'training') {
        // Scroll to training section after a short delay to ensure DOM is ready
        setTimeout(() => {
          const trainingSection = document.querySelector('.robot-training-section')
          if (trainingSection) {
            trainingSection.scrollIntoView({ behavior: 'smooth', block: 'start' })
          }
        }, 500)
      }
    })
    
    return {
      robotState,
      hardwareState,
      hardwareStatus,
      stereoCameraStatus,
      depthMapData,
      selectedStereoMode,
      spatialResults,
      voiceControlState,
      jointList,
      jointPresets,
      sensorData,
      clickCount,
      debugInfo,
      collaborationState,
      collaborationPatterns,
      selectedPattern,
      collaborationInput,
      collaborationConfig,
      selectedPatternDetails,
      initializeHardware,
      fetchDetectedHardware,
      disconnectHardware,
      toggleVoiceControl,
      updateJoint,
      adjustJoint,
      onJointInputChange,
      onJointSliderInput,
      updateJointSpeed,
      applyJointPreset,
      fetchCollaborationPatterns,
      startCollaboration,
      stopCollaboration,
      // Hardware list display related
      showConnectedHardwareList,
      showAvailableHardwareList,
      totalConnectedHardwareCount,
      connectedHardwareCategories,
      availableHardwareCategories,
      toggleConnectedHardwareList,
      toggleAvailableHardwareList,
      // Settings dialog related
      showSettingsDialog,
      availableSensors,
      sensorInstances,
      sensorConfig,
      availableCameras,
      cameraInstances,
      cameraConfig,
      connectionList,
      availableConnectionTypes,
      availableInputTypes,
      connectionSettings, // Backward compatibility
      availableDeviceTypes,
      deviceInstances,
      deviceTypeConfig,
      connectionStatusText,
      connectionStatusClass,
      connectedDevicesCount,
      activeConnectionsCount,
      totalConnectedDevices,
      openSettingsDialog,
      closeSettingsDialog,
      applySettings,
      resetSettings,
      forceReconnect,
      disconnectAll,
      testConnection,
      saveConnectionProfile,
      loadSettingsData,
      // Connection management related
      addNewConnection,
      removeConnection,
      toggleConnectionEnabled,
      updateConnectionType,
      testAllConnections,
      enableAllConnections,
      disableAllConnections,
      saveConnectionsConfig,
      // Device count control related
      getSensorInstances,
      getCameraInstances,
      getDeviceInstances,
      increaseDeviceCount,
      decreaseDeviceCount,
      updateConnectedDevicesCount,
      // Joint management related
      showJointManagementDialog,
      newJoint,
      editingJointId,
      batchMin,
      batchMax,
      batchStep,
      openJointManagementDialog,
      closeJointManagementDialog,
      editJoint,
      removeJoint,
      isJointIdUnique,
      addNewJoint,
      resetNewJointForm,
      applyBatchMin,
      applyBatchMax,
      applyBatchStep,
      saveJointConfiguration,
      resetJointConfiguration,
      // Chat dialog related
      showChatDialog,
      chatMessages,
      chatInputText,
      voiceInputActive,
      chatMessagesContainer,
      openChatDialog,
      closeChatDialog,
      sendChatMessage,
      clearChat,
      toggleVoiceInput,
      exportChat,
      // Robot training module related
      trainingState,
      trainingModes,
      availableTrainingModels,
      selectedTrainingMode,
      selectedTrainingModels,
      selectedJoints,
      selectedSensors,
      selectedCameras,
      trainingParams,
      safetyLimits,
      trainingLog,
      isModelSelected,
      toggleTrainingModel,
      startRobotTraining,
      pauseRobotTraining,
      stopRobotTraining,
      resetTrainingConfig,
      initializeTrainingModule,
      // Stereo vision methods
      detectStereoCameras,
      calibrateStereoCameras,
      enableStereoVision,
      disableStereoVision,
      generateDepthMap,
      startSpatialMapping,
      exportSpatialData,
      // Robot free space movement control related
      movementState,
      movementSpeed,
      waypoints,
      waypointX,
      waypointY,
      waypointZ,
      selectedNavigationMode,
      spatialConstraints,
      navigationState,
      moveRobot,
      stopRobot,
      setMovementSpeed,
      rotateRobot,
      addWaypoint,
      removeWaypoint,
      clearWaypoints,
      executeWaypointNavigation,
      startAutonomousNavigation,
      pauseAutonomousNavigation,
      stopAutonomousNavigation,
      calibrateNavigation,
      exportNavigationData,
      // Keyboard shortcut control
      keyboardState,
      movementHistory,
      clearMovementHistory,
      trajectoryCanvas,
      trajectoryStats,
      clearTrajectory,
      // Path planning
      pathPlanningState,
      calculateDistance,
      optimizePathNearestNeighbor,
      optimizePath2Opt,
      calculateTotalDistance,
      checkLineSegmentObstacleCollision,
      distancePointToLineSegment,
      generateObstacleAvoidanceWaypoints,
      optimizeWaypointPath,
      visualizeOptimizedPath,
      clearOptimizedPath,
      // Collision detection
      collisionDetectionState,
      toggleCollisionDetection,
      toggleCollisionDataSource,
      toggleSensorFusion,
      resetCollisionDetection,
      fetchSensorData,
      processSensorDataToObstacles,
      fuseSensorData,
      updateCollisionDetectionWithSensors,
      updateCollisionDetectionSimulation,
      // Sensor data visualization
      sensorChartCanvas,
      sensorChartData,
      sensorChartDataSource,
      sensorChartType,
      sensorChartTimeWindow,
      sensorChartInstance,
      toggleSensorChartDataSource,
      updateSensorChart,
      clearSensorChartData,
      initializeSensorChart,
      destroySensorChart,
      // Sensor calibration
      sensorCalibrationState,
      startSensorCalibration,
      stopSensorCalibration,
      resetSensorCalibration,
      // Data recording and playback
      dataRecordingState,
      startRecording,
      stopRecording,
      startPlayback,
      stopPlayback,
      clearRecords,
      clearRecordsWithConfirmation,
      // UI confirmation dialog
      uiConfirmState,
      showConfirmDialog,
      hideConfirmDialog,
      confirmAction,
      cancelAction
    }
  }
}
</script>

<style scoped>
.robot-settings-view {
  padding: 20px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
  padding-bottom: 15px;
  border-bottom: 1px solid #ddd;
}

.page-header h2 {
  margin: 0;
  color: #333;
}

.robot-status {
  display: flex;
  align-items: center;
  gap: 15px;
}

.status-label {
  font-weight: bold;
  color: #666;
}

.status-value {
  padding: 4px 8px;
  border-radius: 4px;
  font-weight: bold;
}

.status-value.connected {
  background-color: #d4edda;
  color: #155724;
}

.status-value.disconnected {
  background-color: #f8d7da;
  color: #721c24;
}

.status-value.idle {
  background-color: #fff3cd;
  color: #856404;
}

.battery-indicator {
  background-color: #f0f0f0;
  padding: 4px 8px;
  border-radius: 4px;
}

.connection-status {
  padding: 4px 8px;
  border-radius: 4px;
  font-weight: bold;
}

.connection-status.connected {
  background-color: #d4edda;
  color: #155724;
}

.connection-status.disconnected {
  background-color: #f8d7da;
  color: #721c24;
}

.connection-status.connecting {
  background-color: #fff3cd;
  color: #856404;
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.7; }
  100% { opacity: 1; }
}

.hardware-control-section {
  border: 2px solid #007bff;
  padding: 20px;
  margin: 20px 0 30px 0;
  background-color: #f8f9fa;
  border-radius: 8px;
}

.joint-control-section {
  border: 2px solid #28a745;
  padding: 20px;
  margin: 20px 0 30px 0;
  background-color: #f8f9fa;
  border-radius: 8px;
}

.sensor-data-section {
  border: 2px solid #fd7e14;
  padding: 20px;
  margin: 20px 0 30px 0;
  background-color: #f8f9fa;
  border-radius: 8px;
}

.collaboration-control-section {
  border: 2px solid #6f42c1;
  padding: 20px;
  margin: 20px 0 30px 0;
  background-color: #f8f9fa;
  border-radius: 8px;
}

.debug-info {
  border: 1px solid #ccc;
  padding: 15px;
  margin: 20px 0;
  background: #f9f9f9;
  border-radius: 4px;
}

.joint-control-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.hardware-buttons {
  display: flex;
  gap: 10px;
  margin-top: 15px;
  flex-wrap: wrap;
}

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: bold;
  transition: background-color 0.3s;
}

.btn-primary {
  background-color: #007bff;
  color: white;
}

.btn-primary:hover {
  background-color: #0056b3;
}

.btn-primary:disabled {
  background-color: #6c757d;
  cursor: not-allowed;
}

.btn-secondary {
  background-color: #6c757d;
  color: white;
}

.btn-secondary:hover {
  background-color: #545b62;
}

.btn-danger {
  background-color: #dc3545;
  color: white;
}

.btn-danger:hover {
  background-color: #bd2130;
}

.btn-info {
  background-color: #17a2b8;
  color: white;
}

.btn-info:hover {
  background-color: #117a8b;
}

.btn-warning {
  background-color: #ffc107;
  color: #212529;
}

.btn-warning:hover {
  background-color: #e0a800;
}

.joint-controls {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 20px;
  margin-top: 15px;
}

.joint-control {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.joint-control label {
  font-weight: bold;
  color: #333;
}

.joint-control input[type="range"] {
  width: 100%;
  height: 25px;
  -webkit-appearance: none;
  appearance: none;
  background: #ddd;
  outline: none;
  border-radius: 15px;
}

.joint-control input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 25px;
  height: 25px;
  border-radius: 50%;
  background: #007bff;
  cursor: pointer;
}

.sensor-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 15px;
  margin-top: 15px;
}

.sensor-card {
  padding: 15px;
  background: #f8f9fa;
  border: 1px solid #ddd;
  border-radius: 8px;
}

.sensor-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.sensor-name {
  font-weight: bold;
  color: #333;
}

.sensor-status {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  display: inline-block;
}

.sensor-status.active {
  background-color: #28a745;
}

.sensor-status.inactive {
  background-color: #dc3545;
}

.sensor-value {
  font-size: 1.5em;
  font-weight: bold;
  color: #007bff;
  margin: 10px 0;
}

.sensor-info {
  display: flex;
  justify-content: space-between;
  font-size: 0.85em;
  color: #666;
}

.debug-info pre {
  background: #f8f9fa;
  padding: 15px;
  border-radius: 4px;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 200px;
  overflow-y: auto;
}

/* Settings dialog styles */
.settings-modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.settings-modal {
  background: white;
  border-radius: 8px;
  width: 90%;
  max-width: 900px;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.settings-modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid #ddd;
}

.settings-modal-header h3 {
  margin: 0;
  color: #333;
}

.btn-close {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: #666;
  padding: 0;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.btn-close:hover {
  color: #333;
}

.settings-modal-body {
  padding: 20px;
}

.settings-section {
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid #eee;
}

.settings-section:last-child {
  border-bottom: none;
  margin-bottom: 0;
}

.settings-section h4 {
  margin-top: 0;
  margin-bottom: 15px;
  color: #444;
}

.sensor-selection, .camera-selection, .device-type-selection {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.sensor-item, .camera-item, .device-type-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px;
  background: #f8f9fa;
  border-radius: 4px;
}

.sensor-config, .camera-config, .device-type-config {
  display: flex;
  gap: 15px;
  align-items: center;
  margin-top: 8px;
  padding-left: 20px;
}

.camera-config, .device-type-config {
  flex-direction: column;
  align-items: flex-start;
  gap: 8px;
}

.camera-config label, .device-type-config label {
  display: flex;
  gap: 8px;
  align-items: center;
}

.connection-settings {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.connection-item {
  display: flex;
  gap: 15px;
  align-items: center;
}

.connection-item label {
  display: flex;
  gap: 8px;
  align-items: center;
}

.connection-control {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.connection-status-display {
  padding: 15px;
  background: #f8f9fa;
  border-radius: 4px;
}

.connection-buttons {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.settings-modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 20px;
  border-top: 1px solid #ddd;
}

select, input[type="number"] {
  padding: 6px 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
}

input[type="checkbox"] {
  margin-right: 8px;
}

/* Joint management dialog styles */
.joint-management-modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1001;
}

.joint-management-modal {
  background: white;
  border-radius: 8px;
  width: 90%;
  max-width: 800px;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.joint-management-modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid #ddd;
}

.joint-management-modal-header h3 {
  margin: 0;
  color: #333;
}

.joint-management-modal-body {
  padding: 20px;
}

.joint-management-section {
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid #eee;
}

.joint-management-section:last-child {
  border-bottom: none;
  margin-bottom: 0;
}

.joint-management-section h4 {
  margin-top: 0;
  margin-bottom: 15px;
  color: #444;
}

.joint-list-management {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.joint-management-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 4px;
  border-left: 4px solid #007bff;
}

.joint-info {
  flex: 1;
}

.joint-info strong {
  display: block;
  margin-bottom: 5px;
  color: #333;
}

.joint-info div {
  font-size: 0.9em;
  color: #666;
  margin: 2px 0;
}

.joint-actions {
  display: flex;
  gap: 8px;
}

.btn-sm {
  padding: 4px 8px;
  font-size: 0.85em;
}

.add-joint-form {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.form-group label {
  font-weight: bold;
  color: #555;
}

.form-group input {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
}

.form-buttons {
  display: flex;
  gap: 10px;
  margin-top: 15px;
}

.batch-settings {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.batch-settings .form-group {
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
}

.batch-settings input {
  width: 100px;
}

.joint-management-modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 20px;
  border-top: 1px solid #ddd;
}

/* Collaboration mode configuration styles */
.collaboration-config {
  margin-top: 15px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 4px;
  border: 1px solid #e9ecef;
}

.config-section {
  margin-bottom: 15px;
}

.config-section:last-child {
  margin-bottom: 0;
}

.config-section label {
  display: block;
  margin-bottom: 8px;
  font-weight: bold;
  color: #495057;
}

.input-type-selection, .output-type-selection {
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
  margin-top: 5px;
}

.input-type-selection label, .output-type-selection label {
  display: flex;
  align-items: center;
  gap: 5px;
  font-weight: normal;
  color: #6c757d;
}

.config-select, .config-input {
  padding: 6px 10px;
  border: 1px solid #ced4da;
  border-radius: 4px;
  font-size: 14px;
}

.config-select {
  width: 150px;
}

.config-input {
  width: 100px;
}

/* Workflow visualization styles */
.workflow-visualization {
  margin-top: 20px;
  padding: 15px;
  background: #fff;
  border-radius: 4px;
  border: 1px solid #dee2e6;
}

.workflow-diagram {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 10px;
}

.workflow-step {
  padding: 12px;
  background: #f1f3f4;
  border-radius: 4px;
  border-left: 4px solid #007bff;
  position: relative;
}

.step-header {
  display: flex;
  align-items: center;
  gap: 10px;
}

.step-number {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  background: #007bff;
  color: white;
  border-radius: 50%;
  font-size: 12px;
  font-weight: bold;
}

.step-model {
  font-weight: bold;
  color: #007bff;
  background: #e7f1ff;
  padding: 2px 8px;
  border-radius: 4px;
}

.step-task {
  color: #495057;
  flex: 1;
}

.step-dependencies {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed #dee2e6;
}

.step-dependencies small {
  color: #6c757d;
}

/* Chat dialog styles */
.chat-modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.chat-modal {
  background: white;
  border-radius: 8px;
  width: 90%;
  max-width: 800px;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.chat-modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid #ddd;
}

.chat-modal-header h3 {
  margin: 0;
  color: #333;
}

.chat-modal-body {
  display: flex;
  flex-direction: column;
  flex: 1;
  overflow: hidden;
}

.chat-messages {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  max-height: 400px;
}

.chat-message {
  margin-bottom: 15px;
  padding: 10px 15px;
  border-radius: 8px;
  max-width: 80%;
}

.chat-message.user {
  background: #e3f2fd;
  margin-left: auto;
  border-bottom-right-radius: 2px;
}

.chat-message.bot {
  background: #f5f5f5;
  margin-right: auto;
  border-bottom-left-radius: 2px;
}

.message-content {
  font-size: 14px;
  line-height: 1.5;
}

.message-time {
  font-size: 11px;
  color: #888;
  margin-top: 5px;
  text-align: right;
}

.chat-input-area {
  padding: 20px;
  border-top: 1px solid #eee;
}

.chat-input-area textarea {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  resize: vertical;
  margin-bottom: 10px;
}

.chat-buttons {
  display: flex;
  gap: 10px;
}

/* Device count control styles */
.device-quantity-control {
  display: flex;
  flex-direction: column;
  gap: 20px;
  margin-top: 15px;
}

.device-quantity-item {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 6px;
  border: 1px solid #e9ecef;
  transition: all 0.2s;
}

.device-quantity-item:hover {
  background: #f1f3f5;
  border-color: #dee2e6;
}

.device-info {
  flex: 1;
  min-width: 0;
}

.device-info h5 {
  margin: 0 0 5px 0;
  color: #212529;
  font-size: 16px;
}

.device-info h5 small {
  color: #6c757d;
  font-size: 12px;
  font-weight: normal;
}

.device-description {
  margin: 0 0 10px 0;
  color: #6c757d;
  font-size: 13px;
  line-height: 1.4;
}

.device-instance-list {
  margin-top: 10px;
}

.device-instance {
  padding: 5px 8px;
  background: white;
  border-radius: 3px;
  border: 1px solid #dee2e6;
  margin-bottom: 5px;
  font-size: 12px;
  color: #495057;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.instance-config {
  color: #6c757d;
  font-size: 11px;
}

.device-quantity-controls {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 10px;
  min-width: 250px;
}

.quantity-display {
  font-size: 14px;
  color: #495057;
  font-weight: 500;
}

.quantity-buttons {
  display: flex;
  gap: 10px;
}

.quantity-buttons button {
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  font-weight: bold;
  border-radius: 4px;
  border: none;
  cursor: pointer;
}

.quantity-buttons button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.device-config-control {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
}

.device-config-control label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 13px;
  color: #495057;
}

.device-config-control select {
  padding: 4px 8px;
  border: 1px solid #ced4da;
  border-radius: 4px;
  font-size: 13px;
  background: white;
}

.device-config-control select:disabled {
  background: #f8f9fa;
  color: #6c757d;
  cursor: not-allowed;
}

/* Connection management styles */
.connection-management {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.connection-management-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 6px;
  border: 1px solid #e9ecef;
}

.connection-stats {
  display: flex;
  gap: 20px;
  font-size: 14px;
  color: #495057;
}

.connection-stats span {
  padding: 4px 8px;
  background: white;
  border-radius: 4px;
  border: 1px solid #dee2e6;
}

.connection-list {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.connection-item-card {
  background: white;
  border-radius: 8px;
  border: 1px solid #dee2e6;
  overflow: hidden;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.connection-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 15px;
  background: #f8f9fa;
  border-bottom: 1px solid #dee2e6;
}

.connection-status {
  display: flex;
  align-items: center;
  gap: 10px;
}

.status-indicator {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.status-indicator.active {
  background: #28a745;
}

.status-indicator.inactive {
  background: #dc3545;
}

.connection-type {
  font-size: 12px;
  color: #6c757d;
  background: #e9ecef;
  padding: 2px 6px;
  border-radius: 3px;
}

.connection-port {
  font-size: 12px;
  color: #495057;
  background: #f1f3f5;
  padding: 2px 6px;
  border-radius: 3px;
}

.connection-actions {
  display: flex;
  gap: 8px;
}

.connection-card-body {
  padding: 15px;
}

.connection-config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 20px;
}

.config-group {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.config-group label {
  font-size: 13px;
  color: #495057;
  font-weight: 500;
}

.config-group select,
.config-group input[type="text"],
.config-group input[type="number"] {
  padding: 6px 10px;
  border: 1px solid #ced4da;
  border-radius: 4px;
  font-size: 13px;
}

.config-group input[type="checkbox"] {
  width: auto;
  align-self: flex-start;
}

.input-types-section {
  margin-top: 20px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 6px;
  border: 1px solid #e9ecef;
}

.input-types-section h5 {
  margin: 0 0 10px 0;
  color: #495057;
}

.input-types-selection {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 10px;
}

.input-type-checkbox label {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 13px;
  color: #495057;
  cursor: pointer;
}

.input-type-checkbox small {
  font-size: 11px;
  color: #6c757d;
}

.connected-devices-section {
  margin-top: 15px;
  padding: 15px;
  background: #f1f3f5;
  border-radius: 6px;
  border: 1px solid #dee2e6;
}

.connected-devices-section h5 {
  margin: 0 0 10px 0;
  color: #495057;
}

.connected-devices-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.device-tag {
  font-size: 12px;
  color: #495057;
  background: white;
  padding: 3px 8px;
  border-radius: 12px;
  border: 1px solid #ced4da;
}

.no-devices {
  font-size: 12px;
  color: #6c757d;
  font-style: italic;
}

.connection-management-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding-top: 15px;
  border-top: 1px solid #dee2e6;
}

/* Hardware list styles */
.hardware-lists {
  display: flex;
  flex-direction: column;
  gap: 20px;
  margin: 20px 0;
}

.hardware-list-section {
  background: white;
  border-radius: 8px;
  border: 1px solid #dee2e6;
  overflow: hidden;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  background: #f8f9fa;
  border-bottom: 1px solid #dee2e6;
  cursor: pointer;
  user-select: none;
}

.list-header:hover {
  background: #e9ecef;
}

.list-header h4 {
  margin: 0;
  font-size: 16px;
  color: #495057;
  font-weight: 600;
}

.toggle-icon {
  font-size: 14px;
  color: #6c757d;
}

.list-content {
  padding: 0;
}

.hardware-category {
  padding: 15px 20px;
  border-bottom: 1px solid #e9ecef;
}

.hardware-category:last-child {
  border-bottom: none;
}

.hardware-category h5 {
  margin: 0 0 10px 0;
  font-size: 14px;
  color: #495057;
  font-weight: 600;
  display: flex;
  justify-content: space-between;
}

.hardware-items {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.hardware-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 15px;
  border-radius: 6px;
  font-size: 13px;
}

.hardware-item.connected {
  background: #e7f4e4;
  border: 1px solid #c3e6cb;
}

.hardware-item.disconnected {
  background: #f8d7da;
  border: 1px solid #f5c6cb;
}

.hardware-item.available {
  background: #e7f1ff;
  border: 1px solid #b8daff;
}

.item-id, .item-name {
  font-weight: 600;
  color: #495057;
  min-width: 100px;
}

.item-type, .item-count {
  font-size: 12px;
  color: #6c757d;
  background: white;
  padding: 2px 6px;
  border-radius: 3px;
}

.item-description {
  flex: 1;
  color: #495057;
}

.item-config {
  font-size: 12px;
  color: #6c757d;
  background: white;
  padding: 3px 8px;
  border-radius: 4px;
  border: 1px solid #dee2e6;
}

.chat-modal-footer {
  display: flex;
  justify-content: space-between;
  padding: 15px 20px;
  border-top: 1px solid #ddd;
}

/* Robot Training Module styles */
.model-option {
  transition: all 0.2s ease;
  user-select: none;
}

.model-option:hover {
  background-color: #f0f7ff;
  border-color: #007bff !important;
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 123, 255, 0.1);
}

.model-option.selected {
  background-color: #e7f3ff;
  border-color: #007bff !important;
  border-width: 2px !important;
  color: #0056b3;
  font-weight: bold;
}

.model-option.selected::after {
  content: " ✓";
  color: #28a745;
  font-weight: bold;
}

.training-status-indicator {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.85em;
  font-weight: bold;
}

.training-status-indicator.training {
  background-color: #d4edda;
  color: #155724;
}

.training-status-indicator.paused {
  background-color: #fff3cd;
  color: #856404;
}

.training-status-indicator.idle {
  background-color: #e2e3e5;
  color: #383d41;
}

.training-status-indicator.completed {
  background-color: #d1ecf1;
  color: #0c5460;
}

.training-status-indicator.error {
  background-color: #f8d7da;
  color: #721c24;
}

.dataset-select, .mode-select {
  background-color: white;
  border: 1px solid #ced4da;
  border-radius: 4px;
  padding: 8px 12px;
  font-size: 14px;
  transition: border-color 0.15s ease-in-out;
}

.dataset-select:focus, .mode-select:focus {
  border-color: #80bdff;
  outline: 0;
  box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
}

/* Voice input status styles */
.btn-info.ready {
  background-color: #17a2b8;
  border-color: #17a2b8;
}

.btn-info.recording {
  background-color: #dc3545;
  border-color: #dc3545;
  animation: pulse 1.5s infinite;
}

.btn-info.backend {
  background-color: #28a745;
  border-color: #28a745;
}

.btn-info.browser {
  background-color: #007bff;
  border-color: #007bff;
}

@keyframes pulse {
  0% {
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
  100% {
    opacity: 1;
  }
}

/* Voice input status indicator */
.voice-status-indicator {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 6px;
}

.voice-status-indicator.ready {
  background-color: #17a2b8;
}

.voice-status-indicator.recording {
  background-color: #dc3545;
}

.voice-status-indicator.backend {
  background-color: #28a745;
}

.voice-status-indicator.browser {
  background-color: #007bff;
}

/* Stereo Vision Spatial Recognition Styles */
.stereo-vision-section {
  transition: all 0.3s ease;
}

.stereo-vision-section:hover {
  box-shadow: 0 4px 12px rgba(52, 152, 219, 0.2);
}

.stereo-camera-selection {
  background-color: #f8f9fa;
  padding: 15px;
  border-radius: 6px;
  border: 1px solid #dee2e6;
}

.camera-status p {
  margin: 5px 0;
}

.camera-selection-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}

.stereo-preview-container {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 15px;
}

.camera-preview, .depth-map-preview {
  background-color: #ffffff;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  padding: 15px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.camera-preview h5, .depth-map-preview h5 {
  margin-top: 0;
  margin-bottom: 10px;
  color: #2c3e50;
  text-align: center;
}

.preview-placeholder {
  width: 100%;
  height: 200px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.preview-placeholder video, .preview-placeholder canvas {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.spatial-recognition-controls {
  background-color: #ffffff;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  padding: 20px;
  margin-top: 20px;
}

.control-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}

.mode-select {
  padding: 8px 12px;
  border: 1px solid #ced4da;
  border-radius: 4px;
  background-color: #ffffff;
  color: #495057;
  min-width: 200px;
}

.spatial-results {
  background-color: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  padding: 15px;
  margin-top: 15px;
}

.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 15px;
}

.result-item {
  background-color: #ffffff;
  padding: 10px;
  border-radius: 4px;
  border: 1px solid #e9ecef;
}

.result-item strong {
  color: #2c3e50;
  display: block;
  margin-bottom: 5px;
}

.result-item span {
  color: #6c757d;
  font-size: 0.9em;
}

/* Robot free space movement control styles */
.robot-movement-section {
  border: 2px solid #9b59b6;
  padding: 20px;
  margin: 20px 0 30px 0;
  background-color: #f9f0ff;
  border-radius: 8px;
}

.manual-movement-controls {
  margin-bottom: 30px;
}

.directional-pad {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  max-width: 300px;
  margin: 20px auto;
}

.speed-control {
  margin-top: 20px;
}

.rotation-controls {
  margin-top: 20px;
}

.autonomous-navigation {
  border-top: 2px solid #3498db;
  padding-top: 20px;
}

.waypoint-management {
  margin-bottom: 15px;
}

.spatial-constraints {
  margin-bottom: 15px;
}

.autonomous-controls {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.navigation-status {
  margin-top: 15px;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 4px;
}

/* Keyboard control indicator */
.keyboard-control-indicator {
  margin-top: 15px;
  padding: 10px 15px;
  background-color: #e8f4fd;
  border: 1px solid #b3d7ff;
  border-radius: 6px;
  font-size: 0.9em;
}

.keyboard-control-indicator h5 {
  margin-top: 0;
  margin-bottom: 8px;
  color: #0066cc;
}

.keyboard-shortcuts {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 8px;
}

.keyboard-shortcut {
  display: flex;
  align-items: center;
  gap: 8px;
}

.keyboard-key {
  background-color: #f0f0f0;
  border: 1px solid #ccc;
  border-radius: 4px;
  padding: 2px 6px;
  font-family: monospace;
  font-weight: bold;
  min-width: 24px;
  text-align: center;
}

/* Movement history */
.movement-history {
  margin-top: 20px;
  padding: 15px;
  background-color: #f9f9f9;
  border: 1px solid #eee;
  border-radius: 6px;
}

.movement-history h5 {
  margin-top: 0;
  margin-bottom: 10px;
  color: #555;
}

.history-list {
  max-height: 200px;
  overflow-y: auto;
  font-size: 0.85em;
}

.history-item {
  padding: 4px 8px;
  border-bottom: 1px solid #eee;
  font-family: monospace;
}

.history-item:last-child {
  border-bottom: none;
}

/* Depth map visualization */
.depth-map-preview .preview-placeholder {
  background: linear-gradient(45deg, #1a237e, #4a148c, #880e4f, #b71c1c);
}

/* Camera preview visualization */
.camera-preview .preview-placeholder {
  background-color: #2c3e50;
}

/* Enhanced joint control styles */
.joint-control-section {
  border: 2px solid #007bff;
  padding: 20px;
  margin: 20px 0 30px 0;
  background-color: #e8f4ff;
  border-radius: 8px;
}

.joint-control-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.joint-control-header h3 {
  margin: 0;
  color: #0056b3;
}

.joint-controls {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.joint-control {
  background-color: #ffffff;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  padding: 15px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.joint-control-row {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.joint-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 5px;
}

.joint-label label {
  font-weight: bold;
  color: #333;
  font-size: 1em;
}

.joint-value-display {
  font-family: monospace;
  font-weight: bold;
  color: #007bff;
  background-color: #f8f9fa;
  padding: 2px 8px;
  border-radius: 4px;
  border: 1px solid #e9ecef;
}

.joint-control-inputs {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
}

.joint-fine-tune {
  display: flex;
  gap: 5px;
}

.joint-fine-tune button {
  width: 32px;
  height: 32px;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
}

.joint-numeric-input {
  flex-grow: 1;
  max-width: 120px;
}

.joint-number-input {
  width: 100%;
  padding: 6px 8px;
  border: 1px solid #ced4da;
  border-radius: 4px;
  font-family: monospace;
  text-align: center;
}

.joint-speed-control {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: auto;
}

.speed-label {
  font-size: 0.9em;
  color: #666;
}

.joint-speed-slider {
  width: 80px;
}

.speed-value {
  font-size: 0.9em;
  color: #007bff;
  min-width: 40px;
  text-align: right;
}

.joint-slider-container {
  margin-top: 10px;
}

.joint-slider {
  width: 100%;
  height: 6px;
  -webkit-appearance: none;
  appearance: none;
  background: #e9ecef;
  border-radius: 3px;
  outline: none;
}

.joint-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #007bff;
  cursor: pointer;
  border: 2px solid #ffffff;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}

.joint-slider::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #007bff;
  cursor: pointer;
  border: 2px solid #ffffff;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}

.joint-range-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 5px;
  font-size: 0.8em;
  color: #6c757d;
}

.joint-status {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid #f0f0f0;
}

.joint-status-label {
  font-size: 0.9em;
  color: #666;
}

.joint-status-value {
  font-size: 0.9em;
  font-weight: bold;
  padding: 2px 6px;
  border-radius: 3px;
}

.joint-status-value.ready {
  background-color: #d4edda;
  color: #155724;
}

.joint-status-value.moving {
  background-color: #fff3cd;
  color: #856404;
}

.joint-status-value.error {
  background-color: #f8d7da;
  color: #721c24;
}

.joint-presets {
  margin-top: 25px;
  padding-top: 20px;
  border-top: 1px solid #dee2e6;
}

.joint-presets h4 {
  margin-top: 0;
  margin-bottom: 15px;
  color: #495057;
  font-size: 1.1em;
}

.preset-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .stereo-preview-container {
    grid-template-columns: 1fr;
  }
  
  .control-buttons {
    flex-direction: column;
    align-items: stretch;
  }
  
  .mode-select {
    width: 100%;
  }
  
  /* Joint control responsive styles */
  .joint-control-section {
    padding: 15px;
    margin: 15px 0;
  }
  
  .joint-control-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 15px;
  }
  
  .joint-controls {
    gap: 15px;
  }
  
  .joint-control {
    padding: 12px;
  }
  
  .joint-control-inputs {
    flex-direction: column;
    align-items: stretch;
    gap: 8px;
  }
  
  .joint-fine-tune {
    justify-content: center;
    margin-bottom: 5px;
  }
  
  .joint-numeric-input {
    max-width: 100%;
  }
  
  .joint-speed-control {
    margin-left: 0;
    justify-content: space-between;
    width: 100%;
  }
  
  .joint-speed-slider {
    width: 60%;
  }
  
  .joint-slider-container {
    margin-top: 8px;
  }
  
  .joint-status {
    flex-direction: column;
    align-items: flex-start;
    gap: 5px;
  }
  
  .joint-presets {
    margin-top: 20px;
    padding-top: 15px;
  }
  
  .preset-buttons {
    flex-direction: column;
  }
  
  .preset-buttons button {
    width: 100%;
  }
  
  /* Sensor calibration styles */
  .sensor-calibration-section {
    margin-top: 15px;
    padding: 15px;
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4px;
  }
  
  .sensor-calibration-section h5 {
    margin-top: 0;
    margin-bottom: 15px;
    color: #333;
    font-size: 1.1em;
  }
  
  .calibration-progress-bar {
    height: 10px;
    background-color: #e9ecef;
    border-radius: 5px;
    overflow: hidden;
    margin: 10px 0;
  }
  
  .calibration-progress-fill {
    height: 100%;
    background-color: #28a745;
    transition: width 0.3s ease;
  }
  
  .calibration-error {
    padding: 10px;
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 4px;
    color: #721c24;
    font-size: 0.9em;
    margin-top: 10px;
  }
  
  .calibration-results {
    margin-top: 15px;
    font-size: 0.85em;
  }
  
  .calibration-result-item {
    padding: 5px 0;
    border-bottom: 1px solid #eee;
  }
  
  .calibration-result-item:last-child {
    border-bottom: none;
  }
  
  .calibration-result-item strong {
    color: #495057;
  }
  
  /* Data recording and playback styles */
  .data-recording-section {
    margin-top: 15px;
    padding: 15px;
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4px;
  }
  
  .data-recording-section h5 {
    margin-top: 0;
    margin-bottom: 15px;
    color: #333;
    font-size: 1.1em;
  }
  
  .recording-status {
    padding: 8px 12px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 0.9em;
  }
  
  .recording-status.recording {
    background-color: #f8d7da;
    color: #721c24;
    animation: pulse 1s infinite;
  }
  
  .recording-status.playback {
    background-color: #fff3cd;
    color: #856404;
    animation: pulse 1.5s infinite;
  }
  
  .recording-status.idle {
    background-color: #d4edda;
    color: #155724;
  }
  
  .record-preview {
    max-height: 150px;
    overflow-y: auto;
    font-size: 0.85em;
    margin-top: 10px;
    padding: 10px;
    background-color: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 4px;
  }
  
  .record-item {
    padding: 4px 6px;
    border-bottom: 1px solid #eee;
    font-family: monospace;
  }
  
  .record-item:last-child {
    border-bottom: none;
  }
  
  .playback-progress {
    height: 6px;
    background-color: #e9ecef;
    border-radius: 3px;
    overflow: hidden;
    margin: 10px 0;
  }
  
  .playback-progress-fill {
    height: 100%;
    background-color: #007bff;
    transition: width 0.3s ease;
  }
  
  /* Confirmation dialog styles */
  .confirmation-dialog-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }
  
  .confirmation-dialog {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    width: 90%;
    max-width: 400px;
    overflow: hidden;
  }
  
  .confirmation-dialog-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 20px;
    background-color: #f8f9fa;
    border-bottom: 1px solid #dee2e6;
  }
  
  .confirmation-dialog-header h4 {
    margin: 0;
    color: #333;
    font-size: 1.2em;
  }
  
  .confirmation-dialog-header .btn-close {
    background: none;
    border: none;
    font-size: 1.5em;
    color: #6c757d;
    cursor: pointer;
    padding: 0;
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .confirmation-dialog-header .btn-close:hover {
    color: #333;
  }
  
  .confirmation-dialog-body {
    padding: 20px;
  }
  
  .confirmation-dialog-body p {
    margin: 0;
    color: #555;
    line-height: 1.5;
  }
  
  .confirmation-dialog-footer {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    padding: 15px 20px;
    background-color: #f8f9fa;
    border-top: 1px solid #dee2e6;
  }
  
  .confirmation-dialog-footer button {
    min-width: 80px;
  }
}
</style>
