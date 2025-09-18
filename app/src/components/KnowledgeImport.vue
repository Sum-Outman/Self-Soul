<template>
  <div class="knowledge-import">
    <h2>{{ $t('knowledge.importTitle') }}</h2>
    
    <div class="import-section">
      <div class="file-upload">
        <input type="file" 
               ref="fileInput"
               @change="handleFileSelect"
               :accept="supportedFormats"
               multiple>
        <button @click="triggerFileInput" class="upload-button">
          {{ $t('knowledge.selectFiles') }}
        </button>
        <span class="file-info">{{ $t('knowledge.supportedFormats') }}: {{ supportedFormatsText }}</span>
      </div>

      <div v-if="selectedFiles.length > 0" class="selected-files">
        <h3>{{ $t('knowledge.selectedFiles') }}</h3>
        <div v-for="(file, index) in selectedFiles" :key="index" class="file-item">
          <span class="file-name">{{ file.name }}</span>
          <span class="file-size">({{ formatFileSize(file.size) }})</span>
          <button @click="removeFile(index)" class="remove-btn">{{ $t('common.delete') }}</button>
        </div>
      </div>

      <div class="import-options">
        <div class="option-group">
          <label>{{ $t('knowledge.domain') }}:</label>
          <select v-model="selectedDomain">
            <option value="">{{ $t('knowledge.autoDetect') }}</option>
            <option v-for="domain in domains" :key="domain" :value="domain">
              {{ $t(`knowledge.domains.${domain}`) }}
            </option>
          </select>
        </div>

        <div class="option-group">
          <label>
            <input type="checkbox" v-model="overwriteExisting">
            {{ $t('knowledge.overwriteExisting') }}
          </label>
        </div>
      </div>

      <button @click="startImport" 
              :disabled="isImporting || selectedFiles.length === 0"
              class="import-button">
        {{ isImporting ? $t('knowledge.importing') : $t('knowledge.startImport') }}
      </button>

      <div v-if="importResults.length > 0" class="import-results">
        <h3>{{ $t('knowledge.importResults') }}</h3>
        <div v-for="(result, index) in importResults" :key="index" 
             :class="['result-item', result.success ? 'success' : 'error']">
          <div class="result-file">{{ result.fileName }}</div>
          <div class="result-status">
            {{ result.success ? $t('knowledge.importSuccess') : $t('knowledge.importError') }}
          </div>
          <div v-if="result.message" class="result-message">{{ result.message }}</div>
          <div v-if="result.domain" class="result-domain">
            {{ $t('knowledge.domain') }}: {{ result.domain }}
          </div>
        </div>
      </div>

      <div v-if="importStats" class="import-stats">
        <h3>{{ $t('knowledge.importStatistics') }}</h3>
        <div class="stats-grid">
          <div class="stat-item">
            <span class="stat-label">{{ $t('knowledge.totalFiles') }}:</span>
            <span class="stat-value">{{ importStats.totalFiles }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">{{ $t('knowledge.successful') }}:</span>
            <span class="stat-value">{{ importStats.successful }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">{{ $t('knowledge.failed') }}:</span>
            <span class="stat-value">{{ importStats.failed }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">{{ $t('knowledge.totalSize') }}:</span>
            <span class="stat-value">{{ formatFileSize(importStats.totalSize) }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useI18n } from 'vue-i18n';
import axios from 'axios';

export default {
  name: 'KnowledgeImport',
  setup() {
    const { t } = useI18n();
    
    const fileInput = ref(null);
    const selectedFiles = ref([]);
    const selectedDomain = ref('');
    const overwriteExisting = ref(false);
    const isImporting = ref(false);
    const importResults = ref([]);
    const importStats = ref(null);

    const supportedFormats = '.json,.txt,.pdf,.docx';
    const domains = [
      'physics', 'mathematics', 'chemistry', 'biology',
      'computer_science', 'medicine', 'law', 'economics',
      'general'
    ];

    const supportedFormatsText = computed(() => {
      return t('knowledge.formatsJSON') + ', ' +
             t('knowledge.formatsTXT') + ', ' +
             t('knowledge.formatsPDF') + ', ' +
             t('knowledge.formatsDOCX');
    });

    const triggerFileInput = () => {
      fileInput.value?.click();
    };

    const handleFileSelect = (event) => {
      const files = Array.from(event.target.files);
      // 过滤不支持的文件格式
      const validFiles = files.filter(file => {
        const ext = file.name.toLowerCase().split('.').pop();
        return ['json', 'txt', 'pdf', 'docx'].includes(ext);
      });
      
      selectedFiles.value = [...selectedFiles.value, ...validFiles];
      event.target.value = ''; // 重置input
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
          const response = await axios.post('/api/knowledge/import', formData, {
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
              message: t('knowledge.importSuccessDetail', { length: response.data.content_length })
            });
          } else {
            stats.failed++;
            importResults.value.push({
              fileName: file.name,
              success: false,
              message: response.data.error || t('knowledge.unknownError')
            });
          }
        } catch (error) {
          stats.failed++;
          importResults.value.push({
            fileName: file.name,
            success: false,
            message: error.response?.data?.error || error.message || t('knowledge.uploadError')
          });
        }
      }

      importStats.value = stats;
      isImporting.value = false;
      
      // 如果所有文件都处理完成，清空选择
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
      domains,
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
