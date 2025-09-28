<template>
  <div class="knowledge-import">
    <h2>Import Knowledge</h2>
    
    <div class="import-section">
      <div class="file-upload">
        <input type="file" 
               ref="fileInput"
               @change="handleFileSelect"
               :accept="supportedFormats"
               multiple>
        <button @click="triggerFileInput" class="upload-button">
          Select Files
        </button>
        <span class="file-info">Supported Formats: JSON, Text, PDF, DOCX</span>
      </div>

      <div v-if="selectedFiles.length > 0" class="selected-files">
        <h3>Selected Files</h3>
        <div v-for="(file, index) in selectedFiles" :key="index" class="file-item">
          <span class="file-name">{{ file.name }}</span>
          <span class="file-size">({{ formatFileSize(file.size) }})</span>
          <button @click="removeFile(index)" class="remove-btn">Delete</button>
        </div>
      </div>

      <div class="import-options">
        <div class="option-group">
          <label>Domain:</label>
          <select v-model="selectedDomain">
            <option value="">Auto Detect</option>
            <option value="physics">Physics</option>
            <option value="mathematics">Mathematics</option>
            <option value="chemistry">Chemistry</option>
            <option value="biology">Biology</option>
            <option value="computer_science">Computer Science</option>
            <option value="medicine">Medicine</option>
            <option value="law">Law</option>
            <option value="economics">Economics</option>
            <option value="general">General Knowledge</option>
          </select>
        </div>

        <div class="option-group">
          <label>
            <input type="checkbox" v-model="overwriteExisting">
            Overwrite Existing Knowledge
          </label>
        </div>
      </div>

      <button @click="startImport" 
              :disabled="isImporting || selectedFiles.length === 0"
              class="import-button">
        {{ isImporting ? 'Importing...' : 'Start Import' }}
      </button>

      <div v-if="importResults.length > 0" class="import-results">
        <h3>Import Results</h3>
        <div v-for="(result, index) in importResults" :key="index" 
             :class="['result-item', result.success ? 'success' : 'error']">
          <div class="result-file">{{ result.fileName }}</div>
          <div class="result-status">
            {{ result.success ? 'Import Successful' : 'Import Failed' }}
          </div>
          <div v-if="result.message" class="result-message">{{ result.message }}</div>
          <div v-if="result.domain" class="result-domain">
            Domain: {{ result.domain }}
          </div>
        </div>
      </div>

      <div v-if="importStats" class="import-stats">
        <h3>Import Statistics</h3>
        <div class="stats-grid">
          <div class="stat-item">
            <span class="stat-label">Total Files:</span>
            <span class="stat-value">{{ importStats.totalFiles }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Successful:</span>
            <span class="stat-value">{{ importStats.successful }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Failed:</span>
            <span class="stat-value">{{ importStats.failed }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Total Size:</span>
            <span class="stat-value">{{ formatFileSize(importStats.totalSize) }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue';
import api from '@/utils/api.js';

export default {
  name: 'KnowledgeImport',
  setup() {
    const fileInput = ref(null);
    const selectedFiles = ref([]);
    const selectedDomain = ref('');
    const overwriteExisting = ref(false);
    const isImporting = ref(false);
    const importResults = ref([]);
    const importStats = ref(null);

    const supportedFormats = '.json,.txt,.pdf,.docx';
    const supportedFormatsText = 'JSON, Text, PDF, DOCX';

    const triggerFileInput = () => {
      fileInput.value?.click();
    };

    const handleFileSelect = (event) => {
      const files = Array.from(event.target.files);
      // Filter out unsupported file formats
      const validFiles = files.filter(file => {
        const ext = file.name.toLowerCase().split('.').pop();
        return ['json', 'txt', 'pdf', 'docx'].includes(ext);
      });
      
      selectedFiles.value = [...selectedFiles.value, ...validFiles];
      event.target.value = ''; // Reset input
    };

    const removeFile = (index) => {
      selectedFiles.value.splice(index, 1);
    };

    const formatFileSize = (bytes) => {
      if (bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    const startImport = async () => {
      if (selectedFiles.value.length === 0) return;

      isImporting.value = true;
      importResults.value = [];
      const stats = {
        totalFiles: selectedFiles.value.length,
        successful: 0,
        failed: 0,
        totalSize: 0
      };

      for (const file of selectedFiles.value) {
        stats.totalSize += file.size;
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('domain', selectedDomain.value);
        formData.append('overwrite', overwriteExisting.value);

        try {
          const response = await api.post('/api/knowledge/import', formData, {
            headers: {
              'Content-Type': 'multipart/form-data'
            }
          });

          if (response.data.success) {
            stats.successful++;
            importResults.value.push({
              fileName: file.name,
              success: true,
              domain: response.data.domain,
              message: `Successfully imported (${response.data.content_length} characters)`
            });
          } else {
            stats.failed++;
            importResults.value.push({
              fileName: file.name,
              success: false,
              message: response.data.error || 'Unknown error occurred'
            });
          }
        } catch (error) {
          stats.failed++;
          importResults.value.push({
            fileName: file.name,
            success: false,
            message: error.response?.data?.error || error.message || 'Upload error occurred'
          });
        }
      }

      importStats.value = stats;
      isImporting.value = false;
      
      // Clear selection if all files are processed
      if (stats.successful + stats.failed === stats.totalFiles) {
        selectedFiles.value = [];
      }
    };

    return {
      fileInput,
      selectedFiles,
      selectedDomain,
      overwriteExisting,
      isImporting,
      importResults,
      importStats,
      supportedFormats,

      supportedFormatsText,
      triggerFileInput,
      handleFileSelect,
      removeFile,
      formatFileSize,
      startImport
    };
  }
};
</script>

<style scoped>
.knowledge-import {
  padding: 20px;
  max-width: 800px;
  margin: 0 auto;
}

.import-section {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.file-upload {
  margin-bottom: 20px;
  text-align: center;
}

.upload-button {
  padding: 12px 24px;
  background: #2196F3;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
}

.upload-button:hover {
  background: #1976D2;
}

.file-info {
  display: block;
  margin-top: 8px;
  color: #666;
  font-size: 14px;
}

.selected-files {
  margin-bottom: 20px;
}

.file-item {
  display: flex;
  align-items: center;
  padding: 8px;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  margin-bottom: 8px;
  background: #f9f9f9;
}

.file-name {
  flex: 1;
  font-weight: 500;
}

.file-size {
  color: #666;
  margin: 0 12px;
}

.remove-btn {
  background: #f44336;
  color: white;
  border: none;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  cursor: pointer;
  font-size: 16px;
  line-height: 1;
}

.import-options {
  margin-bottom: 20px;
}

.option-group {
  margin-bottom: 12px;
}

.option-group label {
  display: block;
  margin-bottom: 4px;
  font-weight: 500;
}

.option-group select,
.option-group input[type="text"] {
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  width: 100%;
  max-width: 300px;
}

.import-button {
  padding: 12px 24px;
  background: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
}

.import-button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.import-button:hover:not(:disabled) {
  background: #45a049;
}

.import-results {
  margin-top: 20px;
}

.result-item {
  padding: 12px;
  border-radius: 4px;
  margin-bottom: 8px;
}

.result-item.success {
  background: #e8f5e9;
  border: 1px solid #4CAF50;
}

.result-item.error {
  background: #ffebee;
  border: 1px solid #f44336;
}

.result-file {
  font-weight: 500;
  margin-bottom: 4px;
}

.result-status {
  font-weight: 500;
  margin-bottom: 4px;
}

.result-message {
  color: #666;
  font-size: 14px;
}

.result-domain {
  color: #2196F3;
  font-size: 14px;
}

.import-stats {
  margin-top: 20px;
  padding: 16px;
  background: #f5f5f5;
  border-radius: 4px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.stat-label {
  font-size: 14px;
  color: #666;
  margin-bottom: 4px;
}

.stat-value {
  font-size: 18px;
  font-weight: bold;
  color: #333;
}

input[type="file"] {
  display: none;
}
</style>
