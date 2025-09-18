<template>
  <div class="real-time-input">
    <!-- 状态消息显示 -->
    <div class="status-messages">
      <div v-if="errorState.hasError" class="message error">
        {{ errorState.message }}
      </div>
      <div v-if="successState.hasSuccess" class="message success">
        {{ successState.message }}
      </div>
      <div v-if="warningState.hasWarning" class="message warning">
        {{ warningState.message }}
      </div>
      <div v-if="infoState.hasInfo" class="message info">
        {{ infoState.message }}
      </div>
    </div>
    
    <!-- 摄像头输入 -->
    <div class="camera-section">
      <h3>{{ $t('realtime.camera') }}</h3>
      <div class="camera-container">
        <video ref="videoElement" autoplay playsinline></video>
        <canvas ref="canvasElement" style="display: none;"></canvas>
      </div>
      <div class="camera-controls">
        <button @click="toggleCamera">{{ isCameraActive ? $t('realtime.stopCamera') : $t('realtime.startCamera') }}</button>
        <button @click="captureFrame" :disabled="!isCameraActive">{{ $t('realtime.capture') }}</button>
        <select v-model="selectedCamera" @change="changeCamera">
          <option v-for="camera in cameras" :key="camera.deviceId" :value="camera.deviceId">
            {{ camera.label || $t('realtime.camera') }} {{ $index + 1 }}
          </option>
        </select>
      </div>
    </div>

    <!-- 麦克风输入 -->
    <div class="microphone-section">
      <h3>{{ $t('realtime.microphone') }}</h3>
      <div class="audio-visualization">
        <canvas ref="audioCanvas"></canvas>
      </div>
      <div class="microphone-controls">
        <button @click="toggleMicrophone">{{ isMicrophoneActive ? $t('realtime.stopMicrophone') : $t('realtime.startMicrophone') }}</button>
        <select v-model="selectedMicrophone" @change="changeMicrophone">
          <option v-for="mic in microphones" :key="mic.deviceId" :value="mic.deviceId">
            {{ mic.label || $t('realtime.microphone') }} {{ $index + 1 }}
          </option>
        </select>
        <button @click="startSpeechRecognition" :disabled="!isMicrophoneActive">{{ $t('realtime.startRecognition') }}</button>
      </div>
      <div v-if="transcript" class="transcript">
        {{ $t('realtime.transcript') }}: {{ transcript }}
      </div>
    </div>

    <!-- 网络流输入 -->
    <div class="network-stream-section">
      <h3>{{ $t('realtime.networkStream') }}</h3>
      <div class="stream-controls">
        <div class="input-group">
          <label>{{ $t('realtime.videoStreamUrl') }}:</label>
          <input v-model="videoStreamUrl" placeholder="rtsp://或http://视频流地址" />
          <button @click="toggleVideoStream" :disabled="!videoStreamUrl">
            {{ isVideoStreamActive ? $t('realtime.stopStream') : $t('realtime.startStream') }}
          </button>
        </div>
        <div class="input-group">
          <label>{{ $t('realtime.audioStreamUrl') }}:</label>
          <input v-model="audioStreamUrl" placeholder="http://音频流地址" />
          <button @click="toggleAudioStream" :disabled="!audioStreamUrl">
            {{ isAudioStreamActive ? $t('realtime.stopStream') : $t('realtime.startStream') }}
          </button>
        </div>
      </div>
      <div v-if="isVideoStreamActive" class="stream-preview">
        <video ref="videoStreamElement" autoplay playsinline></video>
      </div>
    </div>

    <!-- 传感器数据输入 -->
    <div class="sensor-section">
      <h3>{{ $t('realtime.sensorData') }}</h3>
      <div class="sensor-controls">
        <button @click="toggleSensorData">{{ isSensorDataActive ? $t('realtime.stopSensors') : $t('realtime.startSensors') }}</button>
        <select v-model="selectedSensorInterface">
          <option value="serial">{{ $t('realtime.sensorSerial') }}</option>
          <option value="bluetooth">{{ $t('realtime.sensorBluetooth') }}</option>
          <option value="network">{{ $t('realtime.sensorNetwork') }}</option>
        </select>
      </div>
      <div v-if="isSensorDataActive" class="sensor-readings">
        <div class="sensor-reading" v-for="(value, sensor) in sensorData" :key="sensor">
          <span class="sensor-label">{{ $t(`realtime.${sensor}`) }}:</span>
          <span class="sensor-value">{{ value }}</span>
        </div>
      </div>
    </div>

    <!-- 多模态融合控制 -->
    <div class="fusion-controls">
      <h3>{{ $t('realtime.multimodalFusion') }}</h3>
      <div class="fusion-options">
        <label>
          <input type="checkbox" v-model="fuseAudioVisual"> 
          {{ $t('realtime.fuseAudioVisual') }}
        </label>
        <label>
          <input type="checkbox" v-model="fuseSensorCamera"> 
          {{ $t('realtime.fuseSensorCamera') }}
        </label>
        <label>
          <input type="checkbox" v-model="fuseAllModalities"> 
          {{ $t('realtime.fuseAll') }}
        </label>
      </div>
      <button @click="processFusedData" :disabled="!hasInputData">{{ $t('realtime.process') }}</button>
    </div>

    <!-- 实时对话控制 -->
    <div class="realtime-dialog-section">
      <h3>{{ $t('realtime.realTimeDialog') }}</h3>
      <div class="dialog-controls">
        <button @click="toggleRealTimeDialog" :class="{ active: isRealTimeDialogActive }">
          {{ isRealTimeDialogActive ? $t('realtime.stopDialog') : $t('realtime.startDialog') }}
        </button>
        <div class="dialog-status" :class="isRealTimeDialogActive ? 'active' : 'inactive'">
          {{ isRealTimeDialogActive ? $t('realtime.dialogActive') : $t('realtime.dialogInactive') }}
        </div>
      </div>
      <div v-if="isRealTimeDialogActive" class="dialog-output">
        <h4>{{ $t('realtime.realTimeResponses') }}</h4>
        <div class="response-messages">
          <div v-for="(response, index) in realTimeResponses" :key="index" class="response-message">
            {{ response }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import errorHandler from '@/utils/errorHandler';

export default {
  name: 'RealTimeInput',
  data() {
    return {
      // 摄像头相关状态
      isCameraActive: false,
      cameras: [],
      selectedCamera: '',
      stream: null,
      
      // 麦克风相关状态
      isMicrophoneActive: false,
      microphones: [],
      selectedMicrophone: '',
      audioContext: null,
      analyser: null,
      audioStream: null,
      transcript: '',
      recognition: null,
      
      // 网络流相关状态
      videoStreamUrl: '',
      audioStreamUrl: '',
      isVideoStreamActive: false,
      isAudioStreamActive: false,
      videoMediaSource: null,
      audioMediaSource: null,
      
      // 传感器相关状态
      isSensorDataActive: false,
      selectedSensorInterface: 'serial',
      sensorData: {
        temperature: '25.0°C',
        humidity: '45%',
        acceleration: '0.0g',
        light: '500lux',
        distance: '100cm'
      },
      
      // 多模态融合
      fuseAudioVisual: false,
      fuseSensorCamera: false,
      fuseAllModalities: false,
      hasInputData: false,
      
      // 实时对话状态
      isRealTimeDialogActive: false,
      realTimeWebSocket: null,
      audioProcessor: null,
      videoProcessor: null,
      realTimeResponses: [],
      webSocketUrl: 'ws://localhost:8000/ws', // 默认WebSocket地址
      audioCaptureInterval: null,
      videoCaptureInterval: null,
      isWebSocketConnected: false,
      
      // 状态消息系统
      errorState: {
        hasError: false,
        message: ''
      },
      successState: {
        hasSuccess: false,
        message: ''
      },
      warningState: {
        hasWarning: false,
        message: ''
      },
      infoState: {
        hasInfo: false,
        message: ''
      }
    }
  },
  mounted() {
    this.listDevices();
    this.setupSensorSimulation();
  },
  beforeDestroy() {
    this.stopCamera();
    this.stopMicrophone();
    this.stopSpeechRecognition();
    this.stopVideoStream();
    this.stopAudioStream();
    this.stopSensorData();
  },
  methods: {
    // 获取设备列表
    async listDevices() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        this.cameras = devices.filter(device => device.kind === 'videoinput');
        this.microphones = devices.filter(device => device.kind === 'audioinput');
        
        if (this.cameras.length > 0) {
          this.selectedCamera = this.cameras[0].deviceId;
        }
        
        if (this.microphones.length > 0) {
          this.selectedMicrophone = this.microphones[0].deviceId;
        }
      } catch (error) {
        console.error('获取设备列表失败:', error);
      }
    },
    
    // 切换摄像头
    async toggleCamera() {
      if (this.isCameraActive) {
        this.stopCamera();
      } else {
        await this.startCamera();
      }
    },
    
    // 启动摄像头
    async startCamera() {
      try {
        const constraints = {
          video: {
            deviceId: this.selectedCamera ? { exact: this.selectedCamera } : undefined,
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        };
        
        this.stream = await navigator.mediaDevices.getUserMedia(constraints);
        this.$refs.videoElement.srcObject = this.stream;
        this.isCameraActive = true;
        this.hasInputData = true;
        this.showSuccess(this.$t('realtime.cameraStarted'));
      } catch (error) {
        console.error('启动摄像头失败:', error);
        this.showError(this.$t('errors.cameraStartFailed'));
      }
    },
    
    // 停止摄像头
    stopCamera() {
      if (this.stream) {
        this.stream.getTracks().forEach(track => track.stop());
        this.stream = null;
      }
      this.isCameraActive = false;
      this.$refs.videoElement.srcObject = null;
    },
    
    // 切换摄像头设备
    async changeCamera() {
      if (this.isCameraActive) {
        this.stopCamera();
        await this.startCamera();
      }
    },
    
    // 捕获帧
    captureFrame() {
      if (!this.isCameraActive) return;
      
      const video = this.$refs.videoElement;
      const canvas = this.$refs.canvasElement;
      const context = canvas.getContext('2d');
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // 获取图像数据
      const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
      
      // 发送到处理模型
      this.processImageData(imageData);
    },
    
    // 处理图像数据
    processImageData(imageData) {
      console.log('处理图像数据:', imageData);
      // 实际实现中会发送到图像处理模型
      this.$emit('image-data', imageData);
    },
    
    // 切换麦克风
    async toggleMicrophone() {
      if (this.isMicrophoneActive) {
        this.stopMicrophone();
      } else {
        await this.startMicrophone();
      }
    },
    
    // 启动麦克风
    async startMicrophone() {
      try {
        const constraints = {
          audio: {
            deviceId: this.selectedMicrophone ? { exact: this.selectedMicrophone } : undefined
          }
        };
        
        this.audioStream = await navigator.mediaDevices.getUserMedia(constraints);
        this.isMicrophoneActive = true;
        this.hasInputData = true;
        
        // 设置音频分析器
        this.setupAudioAnalyser();
        this.showSuccess(this.$t('realtime.microphoneStarted'));
      } catch (error) {
        console.error('启动麦克风失败:', error);
        this.showError(this.$t('errors.microphoneStartFailed'));
      }
    },
    
    // 停止麦克风
    stopMicrophone() {
      if (this.audioStream) {
        this.audioStream.getTracks().forEach(track => track.stop());
        this.audioStream = null;
      }
      this.isMicrophoneActive = false;
      
      if (this.audioContext) {
        this.audioContext.close();
        this.audioContext = null;
      }
    },
    
    // 设置音频分析器
    setupAudioAnalyser() {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      this.analyser = this.audioContext.createAnalyser();
      const source = this.audioContext.createMediaStreamSource(this.audioStream);
      
      source.connect(this.analyser);
      this.analyser.fftSize = 256;
      
      // 开始可视化
      this.visualizeAudio();
    },
    
    // 音频可视化
    visualizeAudio() {
      if (!this.isMicrophoneActive || !this.analyser) return;
      
      const canvas = this.$refs.audioCanvas;
      const ctx = canvas.getContext('2d');
      const bufferLength = this.analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      
      const draw = () => {
        if (!this.isMicrophoneActive) return;
        
        requestAnimationFrame(draw);
        
        this.analyser.getByteFrequencyData(dataArray);
        
        ctx.fillStyle = 'rgb(0, 0, 0)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        const barWidth = (canvas.width / bufferLength) * 2.5;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
          const barHeight = dataArray[i] / 2;
          
          ctx.fillStyle = `rgb(${barHeight + 100}, 50, 50)`;
          ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
          
          x += barWidth + 1;
        }
      };
      
      draw();
    },
    
    // 切换麦克风设备
    async changeMicrophone() {
      if (this.isMicrophoneActive) {
        this.stopMicrophone();
        await this.startMicrophone();
      }
    },
    
    // 启动语音识别
    startSpeechRecognition() {
      if (!('webkitSpeechRecognition' in window)) {
        this.showError(this.$t('errors.speechRecognitionNotSupported'));
        return;
      }
      
      this.recognition = new window.webkitSpeechRecognition();
      this.recognition.continuous = true;
      this.recognition.interimResults = true;
      // 从全局获取语音识别语言设置
      const currentLanguage = localStorage.getItem('user-language') || 'zh';
      const langMap = {
        'zh': 'zh-CN',
        'en': 'en-US',
        'de': 'de-DE',
        'ja': 'ja-JP',
        'ru': 'ru-RU'
      };
      this.recognition.lang = langMap[currentLanguage] || 'en-US';
      
      this.recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }
        
        this.transcript = finalTranscript || interimTranscript;
      };
      
      this.recognition.onerror = (event) => {
        console.error('语音识别错误:', event.error);
        this.showError(this.$t('errors.speechRecognitionError'));
      };
      
      this.recognition.start();
      this.showInfo(this.$t('realtime.speechRecognitionStarted'));
    },
    
    // 停止语音识别
    stopSpeechRecognition() {
      if (this.recognition) {
        this.recognition.stop();
        this.recognition = null;
      }
    },
    
    // 切换视频流
    async toggleVideoStream() {
      if (this.isVideoStreamActive) {
        this.stopVideoStream();
      } else {
        await this.startVideoStream();
      }
    },
    
    // 启动视频流
    async startVideoStream() {
      try {
        if (this.videoStreamUrl) {
          this.$refs.videoStreamElement.src = this.videoStreamUrl;
          this.isVideoStreamActive = true;
          this.hasInputData = true;
          this.showSuccess(this.$t('realtime.videoStreamStarted'));
        }
      } catch (error) {
        console.error('启动视频流失败:', error);
        this.showError(this.$t('errors.videoStreamFailed'));
      }
    },
    
    // 停止视频流
    stopVideoStream() {
      if (this.$refs.videoStreamElement) {
        this.$refs.videoStreamElement.src = '';
      }
      this.isVideoStreamActive = false;
    },
    
    // 切换音频流
    async toggleAudioStream() {
      if (this.isAudioStreamActive) {
        this.stopAudioStream();
      } else {
        await this.startAudioStream();
      }
    },
    
    // 启动音频流
    async startAudioStream() {
      try {
        if (this.audioStreamUrl) {
          // 这里需要实现音频流处理
          console.log('启动音频流:', this.audioStreamUrl);
          this.isAudioStreamActive = true;
          this.hasInputData = true;
          this.showSuccess(this.$t('realtime.audioStreamStarted'));
        }
      } catch (error) {
        console.error('启动音频流失败:', error);
        this.showError(this.$t('errors.audioStreamFailed'));
      }
    },
    
    // 停止音频流
    stopAudioStream() {
      this.isAudioStreamActive = false;
    },
    
    // 切换传感器数据
    toggleSensorData() {
      if (this.isSensorDataActive) {
        this.stopSensorData();
      } else {
        this.startSensorData();
      }
    },
    
    // 启动传感器数据
    startSensorData() {
      this.isSensorDataActive = true;
      this.hasInputData = true;
      // 实际实现中会连接真实传感器
    },
    
    // 停止传感器数据
    stopSensorData() {
      this.isSensorDataActive = false;
    },
    
    // 设置传感器模拟
    setupSensorSimulation() {
      // 模拟传感器数据更新
      setInterval(() => {
        if (this.isSensorDataActive) {
          this.sensorData = {
            temperature: (20 + Math.random() * 10).toFixed(1) + '°C',
            humidity: (40 + Math.random() * 20).toFixed(0) + '%',
            acceleration: (Math.random() * 2).toFixed(1) + 'g',
            light: (100 + Math.random() * 900).toFixed(0) + 'lux',
            distance: (50 + Math.random() * 100).toFixed(0) + 'cm'
          };
        }
      }, 1000);
    },

    // 切换实时对话
    async toggleRealTimeDialog() {
      if (this.isRealTimeDialogActive) {
        await this.stopRealTimeDialog();
      } else {
        await this.startRealTimeDialog();
      }
    },

    // 启动实时对话
    async startRealTimeDialog() {
      try {
        // 连接WebSocket
        await this.connectWebSocket();
        
        // 启动音频捕获
        if (this.isMicrophoneActive) {
          this.startAudioCapture();
        }
        
        // 启动视频捕获
        if (this.isCameraActive) {
          this.startVideoCapture();
        }
        
        this.isRealTimeDialogActive = true;
        this.showSuccess(this.$t('realtime.dialogStarted'));
      } catch (error) {
        console.error('启动实时对话失败:', error);
        this.showError(this.$t('errors.dialogStartFailed'));
      }
    },

    // 停止实时对话
    async stopRealTimeDialog() {
      // 停止音频捕获
      if (this.audioCaptureInterval) {
        clearInterval(this.audioCaptureInterval);
        this.audioCaptureInterval = null;
      }
      
      // 停止视频捕获
      if (this.videoCaptureInterval) {
        clearInterval(this.videoCaptureInterval);
        this.videoCaptureInterval = null;
      }
      
      // 断开WebSocket连接
      await this.disconnectWebSocket();
      
      this.isRealTimeDialogActive = false;
      this.showInfo(this.$t('realtime.dialogStopped'));
    },

    // 连接WebSocket
    async connectWebSocket() {
      return new Promise((resolve, reject) => {
        try {
          this.realTimeWebSocket = new WebSocket(this.webSocketUrl);
          
          this.realTimeWebSocket.onopen = () => {
            this.isWebSocketConnected = true;
            this.showSuccess(this.$t('realtime.websocketConnected'));
            resolve();
          };
          
          this.realTimeWebSocket.onmessage = (event) => {
            this.handleWebSocketMessage(event.data);
          };
          
          this.realTimeWebSocket.onerror = (error) => {
            this.handleWebSocketError(error);
            reject(error);
          };
          
          this.realTimeWebSocket.onclose = () => {
            this.isWebSocketConnected = false;
            this.showWarning(this.$t('realtime.websocketDisconnected'));
          };
        } catch (error) {
          console.error('WebSocket连接失败:', error);
          this.showError(this.$t('errors.websocketConnectionFailed'));
          reject(error);
        }
      });
    },

    // 断开WebSocket连接
    async disconnectWebSocket() {
      if (this.realTimeWebSocket) {
        this.realTimeWebSocket.close();
        this.realTimeWebSocket = null;
      }
      this.isWebSocketConnected = false;
    },

    // 处理WebSocket消息
    handleWebSocketMessage(data) {
      try {
        const message = JSON.parse(data);
        
        if (message.type === 'response') {
          // 处理AI响应
          this.realTimeResponses.push(message.content);
          this.$emit('real-time-response', message.content);
          
          // 如果响应包含语音，可以在这里播放
          if (message.audio) {
            this.playAudioResponse(message.audio);
          }
        } else if (message.type === 'error') {
          this.showError(message.content);
        } else if (message.type === 'status') {
          this.showInfo(message.content);
        }
      } catch (error) {
        console.error('处理WebSocket消息失败:', error);
      }
    },

    // 处理WebSocket错误
    handleWebSocketError(error) {
      console.error('WebSocket错误:', error);
      this.showError(this.$t('errors.websocketError'));
    },

    // 启动音频捕获
    startAudioCapture() {
      if (!this.isMicrophoneActive || !this.audioStream) {
        this.showWarning(this.$t('realtime.microphoneNotActive'));
        return;
      }
      
      // 创建音频处理器
      this.audioProcessor = new (window.AudioContext || window.webkitAudioContext)();
      const source = this.audioProcessor.createMediaStreamSource(this.audioStream);
      const processor = this.audioProcessor.createScriptProcessor(4096, 1, 1);
      
      source.connect(processor);
      processor.connect(this.audioProcessor.destination);
      
      processor.onaudioprocess = (event) => {
        if (this.isWebSocketConnected && this.isRealTimeDialogActive) {
          const audioData = event.inputBuffer.getChannelData(0);
          this.sendAudioData(audioData);
        }
      };
      
      this.audioCaptureInterval = setInterval(() => {
        // 定期发送音频数据
        if (this.isWebSocketConnected && this.isRealTimeDialogActive) {
          this.captureAudioFrame();
        }
      }, 100); // 每100毫秒发送一次
    },

    // 捕获音频帧
    captureAudioFrame() {
      if (!this.analyser) return;
      
      const bufferLength = this.analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      this.analyser.getByteFrequencyData(dataArray);
      
      // 发送音频数据到服务器
      if (this.isWebSocketConnected) {
        this.realTimeWebSocket.send(JSON.stringify({
          type: 'audio',
          data: Array.from(dataArray),
          timestamp: Date.now()
        }));
      }
    },

    // 发送音频数据
    sendAudioData(audioData) {
      if (this.isWebSocketConnected && this.realTimeWebSocket) {
        // 将音频数据转换为适合传输的格式
        const compressedData = this.compressAudioData(audioData);
        
        this.realTimeWebSocket.send(JSON.stringify({
          type: 'audio_raw',
          data: compressedData,
          timestamp: Date.now()
        }));
      }
    },

    // 压缩音频数据
    compressAudioData(audioData) {
      // 简单的压缩：采样和量化
      const compressed = [];
      const sampleRate = 10; // 每10个样本取1个
      
      for (let i = 0; i < audioData.length; i += sampleRate) {
        compressed.push(Math.round(audioData[i] * 100) / 100); // 量化到2位小数
      }
      
      return compressed;
    },

    // 启动视频捕获
    startVideoCapture() {
      if (!this.isCameraActive || !this.stream) {
        this.showWarning(this.$t('realtime.cameraNotActive'));
        return;
      }
      
      this.videoCaptureInterval = setInterval(() => {
        if (this.isWebSocketConnected && this.isRealTimeDialogActive) {
          this.captureVideoFrame();
        }
      }, 100); // 每100毫秒捕获一帧
    },

    // 捕获视频帧
    captureVideoFrame() {
      const video = this.$refs.videoElement;
      const canvas = this.$refs.canvasElement;
      const context = canvas.getContext('2d');
      
      if (video.videoWidth === 0 || video.videoHeight === 0) return;
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // 获取图像数据并压缩
      const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
      const compressedImage = this.compressImageData(imageData);
      
      // 发送到服务器
      if (this.isWebSocketConnected) {
        this.realTimeWebSocket.send(JSON.stringify({
          type: 'video',
          data: compressedImage,
          width: canvas.width,
          height: canvas.height,
          timestamp: Date.now()
        }));
      }
    },

    // 压缩图像数据
    compressImageData(imageData) {
      // 简单的压缩：降低分辨率和质量
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      // 降低分辨率
      const scale = 0.5;
      canvas.width = imageData.width * scale;
      canvas.height = imageData.height * scale;
      
      // 绘制缩放后的图像
      const tempCanvas = document.createElement('canvas');
      const tempCtx = tempCanvas.getContext('2d');
      tempCanvas.width = imageData.width;
      tempCanvas.height = imageData.height;
      tempCtx.putImageData(imageData, 0, 0);
      
      ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
      
      // 获取压缩后的图像数据
      return canvas.toDataURL('image/jpeg', 0.7); // 70%质量
    },

    // 播放音频响应
    playAudioResponse(audioData) {
      if (!audioData) return;
      
      try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createBufferSource();
        
        // 这里需要根据音频数据格式进行解码和播放
        // 实际实现中可能需要base64解码或其他格式处理
        console.log('播放音频响应:', audioData);
        
        source.start();
      } catch (error) {
        console.error('播放音频响应失败:', error);
      }
    },
    
  // 状态消息辅助方法
  showError(message) {
    this.errorState = { hasError: true, message };
    setTimeout(() => {
      this.errorState = { hasError: false, message: '' };
    }, 5000);
  },
  
  showSuccess(message) {
    this.successState = { hasSuccess: true, message };
    setTimeout(() => {
      this.successState = { hasSuccess: false, message: '' };
    }, 3000);
  },
  
  showWarning(message) {
    this.warningState = { hasWarning: true, message };
    setTimeout(() => {
      this.warningState = { hasWarning: false, message: '' };
    }, 4000);
  },
  
  showInfo(message) {
    this.infoState = { hasInfo: true, message };
    setTimeout(() => {
      this.infoState = { hasInfo: false, message: '' };
    }, 3000);
  },
  
  // 处理融合数据
  processFusedData() {
    console.log('处理融合数据:', {
      audioVisual: this.fuseAudioVisual,
      sensorCamera: this.fuseSensorCamera,
      allModalities: this.fuseAllModalities
    });
    
    // 实际实现中会发送到主模型进行融合处理
    this.$emit('process-fused-data', {
      audioVisual: this.fuseAudioVisual,
      sensorCamera: this.fuseSensorCamera,
      allModalities: this.fuseAllModalities
    });
  }
  }
}
</script>

<style scoped>
.real-time-input {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 20px;
  padding: 20px;
  background: #f5f5f5;
  border-radius: 8px;
}

.camera-section, .microphone-section, .network-stream-section, .sensor-section, .fusion-controls {
  background: white;
  padding: 15px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.camera-container {
  position: relative;
  width: 100%;
  height: 300px;
  background: #000;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 10px;
}

video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.audio-visualization {
  width: 100%;
  height: 150px;
  background: #000;
  border-radius: 4px;
  margin-bottom: 10px;
}

.stream-preview {
  width: 100%;
  height: 200px;
  background: #000;
  border-radius: 4px;
  margin-top: 10px;
  overflow: hidden;
}

.camera-controls, .microphone-controls, .stream-controls, .sensor-controls, .fusion-options {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 10px;
}

.input-group {
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin-bottom: 10px;
}

.input-group label {
  font-weight: bold;
}

.input-group input {
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

button {
  padding: 8px 12px;
  background: #2196F3;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.3s;
}

button:disabled {
  background: #90CAF9;
  cursor: not-allowed;
}

button:hover:not(:disabled) {
  background: #0b7dda;
}

select {
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
}

.transcript {
  padding: 10px;
  background: #e3f2fd;
  border-radius: 4px;
  margin-top: 10px;
}

.sensor-readings {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 10px;
}

.sensor-reading {
  display: flex;
  justify-content: space-between;
  padding: 5px;
  background: #f5f5f5;
  border-radius: 4px;
}

.sensor-label {
  font-weight: bold;
}

.fusion-options {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 15px;
}

label {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-messages {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 1000;
  max-width: 400px;
}

.message {
  padding: 12px 16px;
  margin-bottom: 10px;
  border-radius: 6px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  animation: slideIn 0.3s ease-out;
  font-weight: 500;
}

.message.error {
  background: #fdecea;
  color: #d32f2f;
  border-left: 4px solid #d32f2f;
}

.message.success {
  background: #edf7ed;
  color: #2e7d32;
  border-left: 4px solid #2e7d32;
}

.message.warning {
  background: #fff4e5;
  color: #ed6c02;
  border-left: 4px solid #ed6c02;
}

.message.info {
  background: #e3f2fd;
  color: #1976d2;
  border-left: 4px solid #1976d2;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

@media (max-width: 768px) {
  .real-time-input {
    grid-template-columns: 1fr;
  }
  
  .sensor-readings {
    grid-template-columns: 1fr;
  }
  
  .status-messages {
    top: 10px;
    right: 10px;
    left: 10px;
    max-width: none;
  }
}
</style>
