<template>
  <div class="terminal-window">
    <div class="terminal-header">
      <div class="terminal-buttons">
        <span class="terminal-button close"></span>
        <span class="terminal-button minimize"></span>
        <span class="terminal-button maximize"></span>
      </div>
      <div class="terminal-title">{{ title }}</div>
    </div>
    <div class="terminal-body" ref="terminalBody">
      <div 
        v-for="(log, index) in logs" 
        :key="index" 
        class="log-entry"
        :class="getLogClass(log)"
      >
        <span class="timestamp" v-if="showTimestamps">{{ log.time }}</span>
        <span class="log-content">{{ log.message }}</span>
      </div>
      <div class="input-line" v-if="showInput">
        <span class="prompt">{{ displayPrompt }}</span>
        <input 
          type="text" 
          v-model="inputValue" 
          @keyup.enter="handleInput"
          @keyup.up="navigateHistory(-1)"
          @keyup.down="navigateHistory(1)"
          ref="inputField"
          class="terminal-input"
        />
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, watch, nextTick, onMounted } from 'vue'

export default {
  name: 'TerminalWindow',
  props: {
    logs: {
      type: Array,
      default: () => []
    },
    title: {
      type: String,
      default: 'Terminal'
    },
    showTimestamps: {
      type: Boolean,
      default: true
    },
    showInput: {
      type: Boolean,
      default: false
    },
    prompt: {
      type: String,
      default: '$'
    },
    autoScroll: {
      type: Boolean,
      default: true
    },
    maxLines: {
      type: Number,
      default: 1000
    }
  },
  setup(props, { emit }) {
    const terminalBody = ref(null)
    const inputValue = ref('')
    const inputField = ref(null)
    const commandHistory = ref([])
    const historyIndex = ref(-1)

    // Directly return the prompt property value or the default prompt
    const displayPrompt = computed(() => {
      // Prefer the passed prompt property value
      if (props.prompt && props.prompt !== '$') {
        return props.prompt
      }
      // Directly return the default value
      return '$'
    })

    const filteredLogs = computed(() => {
      if (props.maxLines > 0 && props.logs.length > props.maxLines) {
        return props.logs.slice(-props.maxLines)
      }
      return props.logs
    })

    const scrollToBottom = () => {
      if (terminalBody.value && props.autoScroll) {
        nextTick(() => {
          terminalBody.value.scrollTop = terminalBody.value.scrollHeight
        })
      }
    }

    const handleInput = () => {
      if (inputValue.value.trim()) {
        const command = inputValue.value.trim()
        commandHistory.value.push(command)
        historyIndex.value = commandHistory.value.length
        
        emit('command', command)
        inputValue.value = ''
        
        // Automatically scroll to the bottom
        scrollToBottom()
      }
    }

    const navigateHistory = (direction) => {
      if (commandHistory.value.length === 0) return
      
      historyIndex.value = Math.max(0, Math.min(commandHistory.value.length, historyIndex.value + direction))
      
      if (historyIndex.value >= 0 && historyIndex.value < commandHistory.value.length) {
        inputValue.value = commandHistory.value[historyIndex.value]
      } else if (historyIndex.value === commandHistory.value.length) {
        inputValue.value = ''
      }
    }

    const focusInput = () => {
      if (inputField.value) {
        inputField.value.focus()
      }
    }

    const getLogClass = (log) => {
      if (log.level) {
        return `log-${log.level}`
      }
      
      // Infer level based on message content
      const message = log.message.toLowerCase()
      if (message.includes('error') || message.includes('failed')) {
        return 'log-error'
      } else if (message.includes('warning')) {
        return 'log-warning'
      } else if (message.includes('info')) {
        return 'log-info'
      } else if (message.includes('success')) {
        return 'log-success'
      }
      return ''
    }

    const clearTerminal = () => {
      emit('clear')
    }

    // Monitor log changes and automatically scroll
    watch(() => props.logs, () => {
      scrollToBottom()
    }, { deep: true })

    onMounted(() => {
      scrollToBottom()
      if (props.showInput) {
        focusInput()
      }
    })

    return {
      terminalBody,
      inputValue,
      inputField,
      filteredLogs,
      handleInput,
      navigateHistory,
      focusInput,
      getLogClass,
      clearTerminal
    }
  }
}
</script>

<style scoped>
.terminal-window {
  background: #1e1e1e;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 13px;
  line-height: 1.4;
}

.terminal-header {
  background: #2d2d30;
  padding: 8px 12px;
  display: flex;
  align-items: center;
  border-bottom: 1px solid #3e3e42;
}

.terminal-controls {
  display: flex;
  gap: 6px;
  margin-right: 8px;
}

.control {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  cursor: pointer;
}

.control.close {
  background: #ff5f56;
}

.control.minimize {
  background: #ffbd2e;
}

.control.maximize {
  background: #27ca3f;
}

.terminal-title {
  color: #cccccc;
  font-size: 12px;
  font-weight: 500;
  flex: 1;
  text-align: center;
}

.terminal-body {
  height: 300px;
  overflow-y: auto;
  padding: 12px;
  background: #1e1e1e;
  color: #d4d4d4;
}

.terminal-body::-webkit-scrollbar {
  width: 8px;
}

.terminal-body::-webkit-scrollbar-track {
  background: #2d2d30;
}

.terminal-body::-webkit-scrollbar-thumb {
  background: #424245;
  border-radius: 4px;
}

.terminal-body::-webkit-scrollbar-thumb:hover {
  background: #4a4a4e;
}

.log-entry {
  margin-bottom: 4px;
  white-space: pre-wrap;
  word-break: break-all;
}

.timestamp {
  color: #6e7681;
  margin-right: 8px;
}

.log-content {
  color: #d4d4d4;
}

.log-error .log-content {
  color: #f44747;
}

.log-warning .log-content {
  color: #ff9c2a;
}

.log-info .log-content {
  color: #3794ff;
}

.log-success .log-content {
  color: #4ec9b0;
}

.input-line {
  display: flex;
  align-items: center;
  margin-top: 8px;
}

.prompt {
  color: #4ec9b0;
  margin-right: 8px;
  user-select: none;
}

.terminal-input {
  background: transparent;
  border: none;
  color: #d4d4d4;
  font-family: inherit;
  font-size: inherit;
  outline: none;
  flex: 1;
  caret-color: #d4d4d4;
}

.terminal-input::selection {
  background: #264f78;
}

/* Black and white theme variant */
.terminal-window.light {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
}

.terminal-window.light .terminal-header {
  background: var(--bg-tertiary);
  border-bottom: 1px solid var(--border-color);
}

.terminal-window.light .terminal-body {
  background: var(--bg-secondary);
  color: var(--text-primary);
}

.terminal-window.light .timestamp {
  color: var(--text-tertiary);
}

.terminal-window.light .log-content {
  color: var(--text-primary);
}

.terminal-window.light .log-error .log-content {
  color: var(--text-primary);
  font-weight: bold;
}

.terminal-window.light .log-warning .log-content {
  color: var(--text-primary);
  font-style: italic;
}

.terminal-window.light .log-info .log-content {
  color: var(--text-primary);
}

.terminal-window.light .log-success .log-content {
  color: var(--text-primary);
}

.terminal-window.light .terminal-body::-webkit-scrollbar-track {
  background: var(--bg-tertiary);
}

.terminal-window.light .terminal-body::-webkit-scrollbar-thumb {
  background: var(--border-color);
}

.terminal-window.light .terminal-body::-webkit-scrollbar-thumb:hover {
  background: var(--border-dark);
}

.terminal-window.light .prompt {
  color: var(--text-primary);
  font-weight: bold;
}

.terminal-window.light .terminal-input {
  color: var(--text-primary);
  caret-color: var(--text-primary);
}

.terminal-window.light .terminal-input::selection {
  background: var(--bg-tertiary);
}
</style>
