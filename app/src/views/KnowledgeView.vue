<template>
  <div class="knowledge-view">
      <div class="header">
        <div class="header-actions">
          <button :class="{ active: activeTab === 'import' }" @click="activeTab = 'import'">Import</button>
          <button :class="{ active: activeTab === 'browse' }" @click="activeTab = 'browse'">Browse</button>
          <button :class="{ active: activeTab === 'manage' }" @click="activeTab = 'manage'">Manage</button>
          <button :class="{ active: activeTab === 'stats' }" @click="activeTab = 'stats'">Statistics</button>
        </div>
      </div>

    <!-- Import Tab -->
    <div v-if="activeTab === 'import'" class="content">
        <h2>Import Knowledge</h2>
      <div class="import-section">
        <div class="upload-area">
          <input type="file" ref="fileInput" multiple @change="handleFileUpload" style="display: none;">
          <label @click="$refs.fileInput.click()" class="upload-label">
            <div class="upload-icon">📁</div>
            <p>Select Files to Import</p>
            <p class="small-text">Supports PDF, DOCX, TXT, JSON, CSV formats</p>
          </label>
        </div>
        
        <!-- Import status display -->
        <div v-if="uploading" class="upload-status">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: uploadProgress + '%' }"></div>
          </div>
          <p>Uploading: {{ currentUploadFile }}</p>
        </div>
        
        <!-- Domain selection for new files -->
        <div class="domain-selection">
          <label>Domain</label>
          <select v-model="selectedDomain">
            <option value="autoDetect">Auto Detect</option>
            <option v-for="domain in domains" :key="domain" :value="domain">
              {{ domain.charAt(0).toUpperCase() + domain.slice(1) }}
            </option>
          </select>
        </div>
      </div>
    </div>

    <!-- Browse Tab -->
    <div v-else-if="activeTab === 'browse'" class="content">
      <div class="browse-controls">
        <div class="search-box">
        <input type="text" v-model="searchQuery" placeholder="Search knowledge content...">
          <select v-model="searchDomain">
            <option value="">All Domains</option>
            <option v-for="domain in domains" :key="domain" :value="domain">
              {{ domain.charAt(0).toUpperCase() + domain.slice(1) }}
            </option>
          </select>
          <button @click="searchKnowledge">Search</button>
        </div>
      </div>

      <div v-if="searchResults.length > 0" class="search-results">
        <h3>Search Results ({{ searchResults.length }})</h3>
        <div v-for="(result, index) in searchResults" :key="index" class="result-item">
          <div class="result-domain">{{ result.domain.charAt(0).toUpperCase() + result.domain.slice(1) }}</div>
          <div class="result-content">{{ result.content }}</div>
          <div class="result-source">Source: {{ result.source }}</div>
        </div>
      </div>

      <div v-else-if="searchPerformed" class="no-results">
        No results found
      </div>
    </div>

    <!-- Manage Tab -->
    <div v-else-if="activeTab === 'manage'" class="content">
      <div class="manage-header">
        <h3>Manage Files</h3>
        <div class="header-controls">
          <button @click="loadFiles" class="refresh-btn">
            Refresh
          </button>
          <div class="auto-learning-toggle">
            <span>Auto Learning: </span>
            <label class="switch">
              <input type="checkbox" v-model="autoLearningEnabled" @change="toggleAutoLearning">
              <span class="slider round"></span>
            </label>
          </div>
          <button v-if="autoLearningEnabled" class="stop-learning-btn" @click="stopAutoLearning">
            Stop Learning
          </button>
          <button v-else class="start-learning-btn" @click="startAutoLearning">
            Start Learning
          </button>
        </div>
      </div>

      <div v-if="filesLoading" class="loading">
        Loading files...
      </div>

      <div v-else-if="files.length > 0" class="files-list">
        <div class="files-controls">
          <div class="filter-controls">
            <select v-model="filterDomain" @change="filterFiles">
              <option value="">All Domains</option>
              <option v-for="domain in domains" :key="domain" :value="domain">
                {{ domain.charAt(0).toUpperCase() + domain.slice(1) }}
              </option>
            </select>
            <input type="text" v-model="searchFileQuery" placeholder="Search files..." @input="filterFiles">
          </div>
          <div class="sort-controls">
              <select v-model="sortBy" @change="sortFiles">
                <option value="name">Sort by Name</option>
                <option value="size">Sort by Size</option>
                <option value="date">Sort by Date</option>
              </select>
              <select v-model="sortOrder" @change="sortFiles">
                <option value="asc">Ascending</option>
                <option value="desc">Descending</option>
              </select>
          </div>
        </div>

        <div class="files-grid">
          <div v-for="file in filteredFiles" :key="file.id" class="file-card">
            <div class="file-header">
              <span class="file-name">{{ file.name }}</span>
              <span class="file-domain">{{ file.domain.charAt(0).toUpperCase() + file.domain.slice(1) }}</span>
            </div>
            <div class="file-details">
              <span class="file-size">{{ formatFileSize(file.size) }}</span>
              <span class="file-date">{{ formatDate(file.upload_date) }}</span>
            </div>
            <div class="file-actions">
              <button @click="viewFile(file)" class="action-btn view">View</button>
              <button @click="downloadFile(file)" class="action-btn download">Download</button>
              <button @click="confirmDelete(file)" class="action-btn delete">Delete</button>
            </div>
          </div>
        </div>

        <div class="pagination" v-if="filteredFiles.length > 0">
          <button @click="prevPage" :disabled="currentPage === 1">Previous</button>
          <span>Page {{ currentPage }} of {{ totalPages }}</span>
          <button @click="nextPage" :disabled="currentPage === totalPages">Next</button>
        </div>
      </div>

        <div v-else class="no-files">
          No files available
        </div>
        
        <!-- Auto learning progress section -->
        <div v-if="autoLearningStatus !== 'idle'" class="learning-progress-card">
          <div class="card-header">
            <span>Learning Progress</span>
            <button size="small" @click="openLearningModal">View Details</button>
          </div>
          <div class="learning-progress">
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: autoLearningProgress + '%' }" :class="getProgressStatus()"></div>
            </div>
            <div class="progress-info">
              <span class="status-text">{{ getStatusText() }}</span>
              <span class="progress-percentage">{{ autoLearningProgress }}%</span>
            </div>
          </div>
        </div>
      </div>

    <!-- Stats Tab -->
    <div v-else-if="activeTab === 'stats'" class="content">
      <div class="stats-container">
        <h3>Knowledge Statistics</h3>
        <div v-if="statsLoading" class="loading">
          Loading...
        </div>
        <div v-else-if="knowledgeStats" class="stats-grid">
          <div class="stat-card">
            <div class="stat-value">{{ knowledgeStats.total_domains }}</div>
            <div class="stat-label">Total Domains</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{{ knowledgeStats.total_items }}</div>
            <div class="stat-label">Total Items</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{{ formatFileSize(knowledgeStats.total_size || 0) }}</div>
            <div class="stat-label">Total Size</div>
          </div>
        </div>

        <div v-if="knowledgeStats?.domains" class="domain-stats">
          <h4>By Domain</h4>
          <div v-for="(domainStats, domain) in knowledgeStats.domains" :key="domain" class="domain-item">
            <span class="domain-name">{{ domain.charAt(0).toUpperCase() + domain.slice(1) }}</span>
            <span class="domain-count">{{ domainStats.item_count }} Items</span>
            <span class="domain-updated" v-if="domainStats.last_updated">
              Last Updated: {{ formatDate(domainStats.last_updated) }}
            </span>
          </div>
        </div>
      </div>
    </div>

    <!-- Auto learning details modal -->
    <el-dialog title="Auto Learning Details" v-model="showLearningModal" width="60%" top="10%">
      <div class="learning-modal-content">
        <!-- Learning status -->
        <div class="learning-status">
          <h3>Current Status: <span class="status-badge" :class="getStatusClass()">{{ getStatusText() }}</span></h3>
          <el-progress :percentage="autoLearningProgress" :status="getProgressStatus()" class="mt-2"></el-progress>
        </div>
        
        <!-- Learning logs -->
        <div class="learning-logs mt-4">
          <h3>Activity Logs</h3>
          <div class="logs-container" ref="logsContainer">
            <div v-for="log in learningLogs" :key="log.timestamp" class="log-item">
              <span class="log-time">{{ formatTime(log.timestamp) }}</span>
              <span class="log-source">{{ log.source }}:</span>
              <span class="log-message">{{ log.message }}</span>
            </div>
            <div v-if="learningLogs.length === 0" class="no-logs">No logs available</div>
          </div>
        </div>
        
        <!-- Learning history -->
        <div class="learning-history mt-6">
          <h3>Learning History</h3>
          <div v-if="!Array.isArray(autoLearningHistory)" class="no-history">No history available</div>
          <div v-else-if="autoLearningHistory.length === 0" class="no-history">No history available</div>
          <el-table v-else :data="autoLearningHistory" style="width: 100%" size="small">
            <el-table-column prop="timestamp" label="Start Time" width="180">
              <template #default="{ row }">
                {{ row && row.timestamp ? formatDateTime(row.timestamp) : '-' }}
              </template>
            </el-table-column>
            <el-table-column prop="domains" label="Domains" width="150">
              <template #default="{ row }">
                <span v-if="row && row.domains">
                  <el-tag v-for="domain in row.domains" :key="domain" size="small" class="mr-1">
                    {{ domain }}
                  </el-tag>
                </span>
                <span v-else>-</span>
              </template>
            </el-table-column>
            <el-table-column prop="status" label="Status">
              <template #default="{ row }">
                <el-tag v-if="row && row.status" :type="getStatusTagType(row.status)">
                  {{ row.status }}
                </el-tag>
                <span v-else>-</span>
              </template>
            </el-table-column>
            <el-table-column prop="progress" label="Progress" width="100">
              <template #default="{ row }">
                {{ row && row.progress !== undefined ? row.progress : '-' }}%
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="closeLearningModal">Close</el-button>
          <el-button v-if="autoLearningStatus === 'running'" type="warning" @click="stopAutoLearning">
            Stop Learning
          </el-button>
          <el-button v-if="autoLearningStatus !== 'running'" type="primary" @click="startAutoLearning">
            Start New Session
          </el-button>
        </span>
      </template>
    </el-dialog>

    <!-- Delete confirmation modal -->
    <div v-if="showDeleteModal" class="modal-overlay">
      <div class="modal">
        <h3>Confirm Delete</h3>
        <p>Are you sure you want to delete the file "{{ fileToDelete.name }}"? This action cannot be undone.</p>
        <div class="modal-actions">
          <button @click="deleteFile" class="confirm-btn">Confirm</button>
          <button @click="cancelDelete" class="cancel-btn">Cancel</button>
        </div>
      </div>
    </div>

    <!-- File preview modal -->
    <div v-if="showPreviewModal" class="modal-overlay">
      <div class="modal preview-modal">
        <div class="preview-header">
          <h3>{{ currentPreviewFile?.name }}</h3>
          <button @click="closePreview" class="close-btn">&times;</button>
        </div>
        <div class="preview-content">
        <div v-if="previewLoading" class="loading-preview">
          Loading preview...
        </div>
        <div v-else-if="previewError" class="preview-error">
          Error loading preview
        </div>
          <div v-else-if="isTextFile(currentPreviewFile)" class="text-preview">
            <pre>{{ currentFileContent }}</pre>
          </div>
          <div v-else-if="isImageFile(currentPreviewFile)" class="image-preview">
            <img :src="currentFileContent" :alt="currentPreviewFile.name" />
          </div>
          <div v-else class="unsupported-preview">
            <p>Preview not available for this file type</p>
            <p class="file-info">
              File Type: {{ currentPreviewFile?.type || 'unknown' }}<br>
              File Size: {{ formatFileSize(currentPreviewFile?.size) }}<br>
              Upload Date: {{ formatDate(currentPreviewFile?.upload_date) }}
            </p>
          </div>
        </div>
        <div class="preview-actions">
          <button @click="downloadFile(currentPreviewFile)" class="action-btn download">
            Download
          </button>
          <button v-if="isTextFile(currentPreviewFile)" @click="copyText" class="action-btn copy">
            Copy Text
          </button>
          <button @click="closePreview" class="action-btn close">
            Close
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, computed } from 'vue';
import axios from 'axios';
import api from '@/utils/api';
import errorHandler from '@/utils/errorHandler';

export default {
  name: 'KnowledgeView',
  components: {
    // KnowledgeImport组件未被使用，已移除
  },
  setup() {
    
    const activeTab = ref('import');
    const searchQuery = ref('');
    const searchDomain = ref('');
    const searchResults = ref([]);
    const searchPerformed = ref(false);
    const knowledgeStats = ref(null);
    const statsLoading = ref(false);
    
    // Upload related refs
    const uploading = ref(false);
    const uploadProgress = ref(0);
    const currentUploadFile = ref('');
    const selectedDomain = ref('general');

    const domains = [
      'physics', 'mathematics', 'chemistry', 'biology',
      'computer_science', 'medicine', 'law', 'economics',
      'general'
    ];

    const searchKnowledge = async () => {
      if (!searchQuery.value.trim()) return;

      try {
        const response = await api.knowledge.search(searchQuery.value, searchDomain.value);

        if (response.data.success) {
          searchResults.value = response.data.results;
          isRealAPI.value = true;
        } else {
          searchResults.value = [];
        }
        searchPerformed.value = true;
      } catch (error) {
        errorHandler.handleError(error, 'Search knowledge failed');
        searchResults.value = [];
        searchPerformed.value = true;
        showSystemMessage('Failed to search knowledge. Please ensure the backend service is running.');
      }
    };

    const loadKnowledgeStats = async () => {
      statsLoading.value = true;
      try {
        const response = await api.knowledge.stats();
        if (response.data.success) {
          knowledgeStats.value = response.data;
        }
      } catch (error) {
        // Check if connection error
        if (error.code === 'ECONNREFUSED') {
          // Clearly inform the user that the backend service is not running
          errorHandler.handleError(error, 'Backend service not running. Please start the Python backend with "python core/main.py".');
        } else {
          errorHandler.handleError(error, 'Failed to load statistics');
        }
        knowledgeStats.value = {
          total_domains: 0,
          total_items: 0,
          total_size: 0,
          domains: {}
        };
        showSystemMessage('Failed to load statistics. Please ensure the backend service is running.');
      }
      statsLoading.value = false;
    };

    const formatFileSize = (bytes) => {
      if (bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    const formatDate = (dateString) => {
      return new Date(dateString).toLocaleDateString();
    };

    // File management related refs
    const files = ref([]);
    const filesLoading = ref(false);
    const filterDomain = ref('');
    const searchFileQuery = ref('');
    const sortBy = ref('name');
    const sortOrder = ref('asc');
    const currentPage = ref(1);
    const itemsPerPage = ref(10);
    const showDeleteModal = ref(false);
    const fileToDelete = ref(null);
    
    // Auto learning related refs
    const autoLearningEnabled = ref(false);
    const autoLearningStatus = ref('idle'); // idle, running, paused, completed
    const autoLearningProgress = ref(0);
    const autoLearningHistory = ref([]);
    const showLearningModal = ref(false);
    const learningLogs = ref([]);
    
    // File preview related refs
    const showPreviewModal = ref(false);
    const currentPreviewFile = ref(null);
    const previewLoading = ref(false);
    const previewError = ref(false);
    const currentFileContent = ref('');
    const isRealAPI = ref(false);
    
    // Computed properties for file filtering, sorting and pagination
    const filteredFiles = computed(() => {
      let result = files.value;
      
      // Filter by domain
      if (filterDomain.value) {
        result = result.filter(file => file.domain === filterDomain.value);
      }
      
      // Filter by search query
      if (searchFileQuery.value) {
        const query = searchFileQuery.value.toLowerCase();
        result = result.filter(file => 
          file.name.toLowerCase().includes(query) ||
          (file.domain && file.domain.toLowerCase().includes(query))
        );
      }
      
      // Sort files
      result = [...result].sort((a, b) => {
        let valueA, valueB;
        
        switch (sortBy.value) {
          case 'name':
            valueA = a.name.toLowerCase();
            valueB = b.name.toLowerCase();
            break;
          case 'size':
            valueA = a.size;
            valueB = b.size;
            break;
          case 'date':
            valueA = new Date(a.upload_date);
            valueB = new Date(b.upload_date);
            break;
          default:
            valueA = a.name.toLowerCase();
            valueB = b.name.toLowerCase();
        }
        
        if (sortOrder.value === 'asc') {
          return valueA < valueB ? -1 : valueA > valueB ? 1 : 0;
        } else {
          return valueA > valueB ? -1 : valueA < valueB ? 1 : 0;
        }
      });
      
      return result;
    });
    
    const paginatedFiles = computed(() => {
      const start = (currentPage.value - 1) * itemsPerPage.value;
      const end = start + itemsPerPage.value;
      return filteredFiles.value.slice(start, end);
    });
    
    const totalPages = computed(() => {
      return Math.ceil(filteredFiles.value.length / itemsPerPage.value);
    });
    
    const filterFiles = () => {
      currentPage.value = 1;
    };

    const sortFiles = () => {
      currentPage.value = 1;
    };

    const nextPage = () => {
      if (currentPage.value < totalPages.value) {
        currentPage.value++;
      }
    };

    const prevPage = () => {
      if (currentPage.value > 1) {
        currentPage.value--;
      }
    };

    // Load files from API with timeout and mock data fallback
    const loadFiles = async () => {
      filesLoading.value = true;
      try {
        const response = await api.knowledge.files();
        
        if (response.data && response.data.success) {
          files.value = response.data.files;
          filterFiles();
          showSystemMessage('File list loaded successfully');
          // Mark as real API connection
          isRealAPI.value = true;
        } else {
          files.value = [];
          showSystemMessage('File list is empty');
        }
      } catch (error) {
        // Check if connection error
        if (error.code === 'ECONNREFUSED') {
          // Clearly inform the user that the backend service is not running
          errorHandler.handleError(error, 'Backend service not running. Please start the Python backend with "python core/main.py".');
        } else {
          errorHandler.handleError(error, 'Failed to load files');
        }
        files.value = [];
        filterFiles();
        showSystemMessage('Failed to load files. Please ensure the backend service is running.');
      }
      filesLoading.value = false;
    };
    
    // Show system message
    const showSystemMessage = (message) => {
      if (typeof window !== 'undefined') {
        // In a real app, this would use a notification system
        console.log('[System]', message);
        // Simple alert as fallback
        setTimeout(() => alert(`[System] ${message}`), 100);
      }
    };


    // File operations with enhanced error handling
    const viewFile = async (file) => {
      try {
        // Try to open file from server
        const newTab = window.open(`/api/knowledge/files/${file.id}/view`, '_blank');
        
        // Check if popup was blocked
        if (!newTab || newTab.closed || typeof newTab.closed === 'undefined') {
          showSystemMessage('Please allow pop-ups to view file');
          // Fallback to preview modal
          openPreview(file);
        }
      } catch (error) {
        errorHandler.handleError(error, 'Failed to view file');
        showSystemMessage(`Failed to view file: ${file.name}. Please ensure the backend service is running.`);
        // Fallback to preview modal
        openPreview(file);
      }
    };

    // File type detection methods
    const isTextFile = (file) => {
      if (!file || !file.name) return false;
      const textExtensions = ['.txt', '.md', '.json', '.js', '.ts', '.html', '.css', '.xml', '.csv'];
      return textExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    };

    const isImageFile = (file) => {
      if (!file || !file.name) return false;
      const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'];
      return imageExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    };

    // Open file preview
    const openPreview = async (file) => {
      currentPreviewFile.value = file;
      showPreviewModal.value = true;
      previewLoading.value = true;
      previewError.value = false;
      currentFileContent.value = '';

      try {
        // Try to load actual file content from server
        const response = await api.knowledge.filePreview(file.id);

        if (response.data.success) {
          if (isTextFile(file)) {
            currentFileContent.value = response.data.content;
          } else if (isImageFile(file)) {
            currentFileContent.value = response.data.content || `data:image/jpeg;base64,${response.data.data}`;
          } else {
            // For unsupported types, just show file info
            currentFileContent.value = '';
          }
        } else {
          throw new Error('Failed to load file content');
        }
      } catch (error) {
          errorHandler.handleError(error, 'Failed to load preview');
          previewError.value = true;
          // When API fails, show file information
          if (isTextFile(file)) {
            currentFileContent.value = `Failed to load file content from server.\n\nFile Information:\nName: ${file.name}\nSize: ${formatFileSize(file.size)}\nUpload Date: ${formatDate(file.upload_date)}\nDomain: ${file.domain}\n\nPlease ensure the backend service is running.`;
          }
        }
      previewLoading.value = false;
    };

    // Close file preview
    const closePreview = () => {
      showPreviewModal.value = false;
      currentPreviewFile.value = null;
      previewLoading.value = false;
      previewError.value = false;
      currentFileContent.value = '';
    };

    // Copy text content to clipboard
    const copyText = async () => {
      try {
        await navigator.clipboard.writeText(currentFileContent.value);
          showSystemMessage('Text copied to clipboard');
        } catch (error) {
        errorHandler.handleError(error, 'Failed to copy text');
        showSystemMessage('Failed to copy text. Please try manually.');
      }
    };

    const downloadFile = async (file) => {
      try {
        // Try to download from server with timeout
        const controller = new AbortController();
        
        // Due to the need for blob response type, use axios directly instead of the encapsulated api instance
        const response = await axios.get(`/api/knowledge/files/${file.id}/download`, {
          responseType: 'blob',
          signal: controller.signal,
          timeout: 30000 // 30 seconds timeout
        });
        
        if (response.status !== 200) {
          throw new Error(`Download failed: ${response.status}`);
        }
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = file.name;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        showSystemMessage(`Download successful: ${file.name}`);
      } catch (error) {
        errorHandler.handleError(error, 'Download error');
        showSystemMessage(`Cannot download file: ${file.name}. Please ensure the backend service is running.`);
      }
    };

    const confirmDelete = (file) => {
      fileToDelete.value = file;
      showDeleteModal.value = true;
    };

    const cancelDelete = () => {
      showDeleteModal.value = false;
      fileToDelete.value = null;
    };

    const deleteFile = async () => {
      try {
        if (!fileToDelete.value) {
          showDeleteModal.value = false;
          return;
        }
        
        // Try to delete from server
        const response = await api.delete(`/api/knowledge/files/${fileToDelete.value.id}`, { timeout: 5000 });
        
        if (response.data.success) {
          // Remove file from list
          files.value = files.value.filter(f => f.id !== fileToDelete.value.id);
          filterFiles();
          showSystemMessage(`Delete successful: ${fileToDelete.value.name}`);
        } else {
          showSystemMessage(`Delete failed: ${fileToDelete.value.name}`);
        }
      } catch (error) {
        errorHandler.handleError(error, 'Delete error');
        showSystemMessage(`Failed to delete file: ${fileToDelete.value.name}. Please ensure the backend service is running.`);
      }
      showDeleteModal.value = false;
      fileToDelete.value = null;
    };
    
    // Auto learning control methods
    const toggleAutoLearning = async () => {
      if (autoLearningEnabled.value) {
        await startAutoLearning();
      } else {
        await stopAutoLearning();
      }
    };
    
    // WebSocket related variables
    let webSocketConnection = null;
    let connectionAttempts = 0;
    const maxReconnectAttempts = 3;
    let reconnectTimeout = null;
    let isUsingWebSocket = false;
    
    // Connect to WebSocket for auto learning updates - simplified to use polling directly
    const connectWebSocket = (sessionId) => {
      try {
        // Close any existing WebSocket connection if it exists
        if (reconnectTimeout) {
          clearTimeout(reconnectTimeout);
          reconnectTimeout = null;
        }
        
        if (webSocketConnection) {
          webSocketConnection.close();
          webSocketConnection = null;
          isUsingWebSocket = false;
        }
        
        addLearningLog('System', 'Skipping WebSocket connection, using polling instead for reliable updates');
        showSystemMessage('Using polling mode for progress updates');
        
        // Start polling immediately
        startProgressPolling();
      } catch (error) {
        errorHandler.handleError(error, 'Auto learning connection error');
        // Fall back to polling even if there's an error
        startProgressPolling();
      }
    };
    
    // Handle WebSocket errors
    const handleWebSocketError = (error) => {
      const errorMessage = error.message || 'Unknown error';
      errorHandler.handleError(error, 'WebSocket connection error');
      
      // If not using WebSocket yet, switch to polling
      if (autoLearningStatus.value === 'running' && !isUsingWebSocket) {
        addLearningLog('System', `WebSocket connection failed: ${errorMessage}. Using polling instead.`);
        startProgressPolling();
      }
    };
    
    // Close WebSocket connection
    const closeWebSocket = () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
      }
      
      if (webSocketConnection) {
        webSocketConnection.close();
        webSocketConnection = null;
        isUsingWebSocket = false;
      }
    };
    
    // Complete auto learning process
    const completeAutoLearning = () => {
      autoLearningProgress.value = 100;
      autoLearningStatus.value = 'completed';
      
      // Add completion log
      addLearningLog('System', 'Auto learning completed');
      
      // Update learning history
      if (autoLearningHistory.value.length > 0) {
        const lastItem = autoLearningHistory.value[autoLearningHistory.value.length - 1];
        lastItem.status = 'completed';
        lastItem.progress = 100;
        lastItem.end_time = new Date().toISOString();
      }
      
      // Close WebSocket
      closeWebSocket();
      
      // Refresh files to show any new knowledge
      setTimeout(() => {
        loadFiles();
        showSystemMessage('Knowledge base updated through auto learning');
      }, 1000);
    };
    
    const startAutoLearning = async () => {
      try {
        autoLearningStatus.value = 'running';
        autoLearningProgress.value = 0;
        learningLogs.value = [];
        isUsingWebSocket = false;
        connectionAttempts = 0;
        
        // Start auto learning on server using api.js method
        const response = await api.knowledge.autoLearning.start({
          domains: filterDomain.value ? [filterDomain.value] : [],
          priority: 'balanced'
        });
        
        if (response.data.status === 'success' || response.data.status === 'warning') {
          showSystemMessage('Auto learning started successfully');
          addLearningLog('System', 'Auto learning started');
          
          // Add learning history
          const sessionId = response.data.session_id || 'learn_' + Date.now();
          addLearningHistoryItem({
            id: sessionId,
            timestamp: new Date().toISOString(),
            status: 'running',
            domains: filterDomain.value ? [filterDomain.value] : ['general'],
            progress: 0
          });
          
          // Try to connect via WebSocket first
          connectWebSocket(sessionId);
        } else {
          throw new Error('Failed to start auto learning');
        }
      } catch (error) {
        errorHandler.handleError(error, 'Auto learning start failed');
        showSystemMessage('Failed to start auto learning: ' + error.message);
        autoLearningEnabled.value = false;
        autoLearningStatus.value = 'idle';
      }
    };
    
    const stopAutoLearning = async () => {
      try {
        autoLearningStatus.value = 'paused';
        
        // Close WebSocket connection
        closeWebSocket();
        
        // Try to stop auto learning on server using api.js method
        try {
          const response = await api.knowledge.autoLearning.stop();
          
          if (response.data.status === 'success' || response.data.status === 'warning') {
            showSystemMessage('Auto learning stopped successfully');
            addLearningLog('System', 'Auto learning stopped');
          }
        } catch (error) {
          // If API call fails, just show message
          showSystemMessage('Auto learning stopped locally');
        }
        
        autoLearningStatus.value = 'idle';
      } catch (error) {
        errorHandler.handleError(error, 'Stop learning error');
      }
    };
    
    // Progress polling function to get real updates from server (fallback when WebSocket fails)
    let progressPollingInterval = null;
    
    const startProgressPolling = () => {
      // Clear any existing interval
      if (progressPollingInterval) {
        clearInterval(progressPollingInterval);
      }
      
      // Don't start polling if WebSocket is already active
      if (isUsingWebSocket) {
        return;
      }
      
      // Start polling every 2 seconds
      let pollingInterval = 2000;
      let consecutiveFailures = 0;
      const maxFailures = 3;
      
      addLearningLog('System', `Starting polling mode (interval: ${pollingInterval/1000}s)`);
      
      progressPollingInterval = setInterval(async () => {
        try {
          if (autoLearningStatus.value !== 'running' || isUsingWebSocket) {
            clearInterval(progressPollingInterval);
            return;
          }
          
          const response = await api.knowledge.autoLearning.progress();
          
          // Reset failure counter
          consecutiveFailures = 0;
          
          if (response.data && response.data.progress !== undefined) {
            autoLearningProgress.value = response.data.progress;
            
            // Adaptive polling interval based on progress
            if (autoLearningProgress.value > 90) {
              pollingInterval = 1000; // Poll more frequently when nearing completion
            } else if (autoLearningProgress.value > 50) {
              pollingInterval = 1500;
            } else {
              pollingInterval = 2000;
            }
            
            // Add real logs if available
            if (response.data.logs && response.data.logs.length > 0) {
              response.data.logs.forEach(log => {
                addLearningLog('System', log);
              });
            }
            
            // Check if completed
            if (response.data.progress >= 100 || response.data.learning_status === 'completed') {
              autoLearningProgress.value = 100;
              autoLearningStatus.value = 'completed';
              
              // Add completion log
              addLearningLog('System', 'Auto learning completed');
              
              // Update learning history
              if (autoLearningHistory.value.length > 0) {
                const lastItem = autoLearningHistory.value[autoLearningHistory.value.length - 1];
                lastItem.status = 'completed';
                lastItem.progress = 100;
                lastItem.end_time = new Date().toISOString();
              }
              
              // Refresh files to show any new knowledge
              setTimeout(() => {
                loadFiles();
                showSystemMessage('Knowledge base updated through auto learning');
              }, 1000);
              
              clearInterval(progressPollingInterval);
            }
          }
        } catch (error) {
          consecutiveFailures++;
          
          // Log error but continue trying
          if (consecutiveFailures <= maxFailures) {
            addLearningLog('System', `Polling error (attempt ${consecutiveFailures}/${maxFailures}): ${error.message}`);
            
            // Increase polling interval on failure
            pollingInterval = Math.min(10000, pollingInterval * 1.5);
          } else {
            // Exceeded maximum failures, show error
            addLearningLog('System', `Maximum polling failures reached (${maxFailures})`);
            showSystemMessage('Failed to communicate with server');
            
            clearInterval(progressPollingInterval);
            
            // If training is still in progress, stop training
            if (autoLearningStatus.value === 'running') {
              showSystemMessage('Auto learning stopped due to connection issues');
              autoLearningEnabled.value = false;
              autoLearningStatus.value = 'idle';
            }
          }
        }
      }, pollingInterval);
    };
    
    const addLearningLog = (source, message) => {
      const log = {
        timestamp: new Date().toISOString(),
        source,
        message
      };
      learningLogs.value.push(log);
      
      // Keep only last 100 logs
      if (learningLogs.value.length > 100) {
        learningLogs.value.shift();
      }
    };
    
    const addLearningHistoryItem = (item) => {
      autoLearningHistory.value.push(item);
      
      // Keep only last 10 learning sessions
      if (autoLearningHistory.value.length > 10) {
        autoLearningHistory.value.shift();
      }
    };
    
    const openLearningModal = () => {
      showLearningModal.value = true;
    };
    
    const closeLearningModal = () => {
      showLearningModal.value = false;
    };
    
    // Helper methods for auto learning status and formatting
    const getStatusText = () => {
      const statusMap = {
        'idle': 'Idle',
        'running': 'Running',
        'paused': 'Paused',
        'completed': 'Completed'
      };
      return statusMap[autoLearningStatus.value] || 'Unknown';
    };
    
    const getProgressStatus = () => {
      if (autoLearningStatus.value === 'completed') {
        return 'success';
      }
      if (autoLearningStatus.value === 'running') {
        return 'primary';
      }
      if (autoLearningStatus.value === 'paused') {
        return 'warning';
      }
      return '';
    };
    
    const getStatusClass = () => {
      const statusClassMap = {
        'idle': 'status-idle',
        'running': 'status-running',
        'paused': 'status-paused',
        'completed': 'status-completed'
      };
      return statusClassMap[autoLearningStatus.value] || '';
    };
    
    const getStatusTagType = (status) => {
      const tagTypeMap = {
        'idle': 'default',
        'running': 'primary',
        'paused': 'warning',
        'completed': 'success'
      };
      return tagTypeMap[status] || 'default';
    };
    
    const formatTime = (timestamp) => {
      const date = new Date(timestamp);
      return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      });
    };
    
    const formatDateTime = (timestamp) => {
      const date = new Date(timestamp);
      return date.toLocaleString('en-US', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      });
    };

    // File upload handling
    const handleFileUpload = async (event) => {
      const files = event.target.files;
      if (!files.length) return;

      // File type and size validation
      const maxFileSize = 20 * 1024 * 1024; // 20MB
      const allowedTypes = ['application/pdf', 'application/msword', 
                           'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                           'application/vnd.ms-powerpoint', 
                           'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                           'text/plain', 'image/jpeg', 'image/png', 'application/json'];

      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        // Validate file type
        if (!allowedTypes.includes(file.type)) {
          showSystemMessage(`File type not supported: ${file.name}`);
          continue;
        }
        
        // Validate file size
          if (file.size > maxFileSize) {
            showSystemMessage(`File size exceeds limit: ${file.name}`);
            continue;
          }
        
        try {
          uploading.value = true;
          uploadProgress.value = 0;
          currentUploadFile.value = file.name;
          
          // 清除任何现有的进度模拟
          // 使用真实的上传进度
          
          // Create form data
          const formData = new FormData();
          formData.append('file', file);
          
          // Handle auto detection
          if (selectedDomain.value === 'autoDetect') {
            // In a real app, this would send to server for auto-detection
            formData.append('domain', 'general'); // Default to general
            showSystemMessage(`Automatically detecting domain for ${file.name}`);
          } else {
            formData.append('domain', selectedDomain.value);
          }
          
          // Try to upload to server
        try {
          const response = await api.post('/api/knowledge/upload', formData, {
              timeout: 30000, // 30 seconds timeout
              onUploadProgress: (event) => {
                if (event.total) {
                  uploadProgress.value = Math.round((event.loaded * 100) / event.total);
                }
              }
            });
            
            if (response.data.success) {
              showSystemMessage(`Upload successful: ${file.name}`);
              // Refresh file list
              loadFiles();
            } else {
              showSystemMessage(`Upload failed: ${file.name}`);
            }
          } catch (error) {
            errorHandler.handleError(error, 'Upload error');
            showSystemMessage(`Failed to upload file: ${file.name}. Please ensure the backend service is running.`);
          }
          
          clearInterval(progressInterval);
          uploadProgress.value = 100;
        } catch (error) {
            errorHandler.handleError(error, 'Error processing file');
            showSystemMessage(`Failed to process file: ${file.name}`);
          } finally {
          // Reset upload state after a short delay to show 100% progress
          setTimeout(() => {
            uploading.value = false;
            uploadProgress.value = 0;
            currentUploadFile.value = '';
          }, 500);
        }
      }
      
      // Clear file input
      event.target.value = '';
    };
    
    onMounted(() => {
      loadKnowledgeStats();
      loadFiles();
    });

    return {
      activeTab,
      searchQuery,
      searchDomain,
      searchResults,
      searchPerformed,
      knowledgeStats,
      statsLoading,
      domains,
      searchKnowledge,
      formatFileSize,
      formatDate,
      files,
      filesLoading,
      filteredFiles: paginatedFiles, // filteredFiles is used in the template, mapped to paginated files
      filterDomain,
      searchFileQuery,
      sortBy,
      sortOrder,
      currentPage,
      totalPages,
      showDeleteModal,
      fileToDelete,
      loadFiles,
      filterFiles,
      sortFiles, // 添加缺失的sortFiles函数
      nextPage,
      prevPage,
      viewFile,
      downloadFile,
      confirmDelete,
      deleteFile,
      cancelDelete,
      // Upload related refs
      uploading,
      uploadProgress,
      currentUploadFile,
      selectedDomain,
      handleFileUpload,
      // Auto learning related refs and methods
      autoLearningEnabled,
      autoLearningStatus,
      autoLearningProgress,
      autoLearningHistory,
      showLearningModal,
      learningLogs,
      toggleAutoLearning,
      startAutoLearning,
      stopAutoLearning,
      openLearningModal,
      closeLearningModal,
      getStatusText,
      getProgressStatus,
      getStatusClass,
      getStatusTagType,
      formatTime,
      formatDateTime
    };
  }
};
</script>

<style scoped>
:root {
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 12px;
  --spacing-lg: 16px;
  --spacing-xl: 24px;
  --border-radius: 6px;
  --border-radius-lg: 8px;
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow-md: 0 2px 4px rgba(0, 0, 0, 0.08);
  --shadow-lg: 0 4px 8px rgba(0, 0, 0, 0.12);
  --transition: all 0.15s ease;
  --bg-primary: #ffffff;
  --bg-secondary: #f7f7f7;
  --bg-tertiary: #f2f2f2;
  --text-primary: #111111;
  --text-secondary: #333333;
  --text-tertiary: #666666;
  --border-color: #e5e5e5;
  --border-light: #eeeeee;
  --accent-color: #444444;
  --accent-light: #cccccc;
}

.knowledge-view {
  padding: var(--spacing-lg);
  max-width: 1200px;
  margin: 70px auto 0;
  min-height: calc(100vh - 70px);
  font-family: var(--font-family);
  background: var(--bg-primary);
}

.header {
  display: flex;
  flex-direction: column;
  margin-bottom: var(--spacing-xl);
  padding-bottom: var(--spacing-lg);
  border-bottom: 1px solid var(--border-light);
}

.header h1 {
  margin: 0 0 var(--spacing-lg) 0;
  font-size: 32px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.025em;
}

.header-actions {
  display: flex;
  gap: var(--spacing-xs);
  flex-wrap: wrap;
  background: var(--bg-secondary);
  padding: var(--spacing-xs);
  border-radius: var(--border-radius-lg);
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--border-light);
}

.header-actions button {
  flex: 1;
  min-width: 120px;
  padding: var(--spacing-sm) var(--spacing-md);
  border: 1px solid transparent;
  background: transparent;
  cursor: pointer;
  color: var(--text-secondary);
  font-size: 14px;
  font-weight: 500;
  transition: var(--transition);
  text-align: center;
  border-radius: var(--border-radius);
  line-height: 1.5;
}

.header-actions button:hover {
  background: var(--bg-tertiary);
  color: var(--text-primary);
  transform: translateY(-1px);
}

.header-actions button.active {
  background: var(--bg-primary);
  color: var(--text-primary);
  font-weight: 600;
  border-color: var(--border-light);
  box-shadow: var(--shadow-sm);
}

.header-actions button:focus {
  outline: 2px solid var(--accent-light);
  outline-offset: 2px;
}

.content {
  background: var(--bg-primary);
  padding: var(--spacing-xl);
  border-radius: var(--border-radius-lg);
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--border-light);
  min-height: 500px;
  margin-top: var(--spacing-md);
  transition: var(--transition);
  width: 100%;
  max-width: 1200px;
}

/* Import Tab Styles */
.import-section {
  max-width: 1100px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xl);
  width: 100%;
}

.upload-area {
  text-align: center;
  padding: var(--spacing-2xl);
  border: 2px dashed var(--border-color);
  border-radius: var(--border-radius-lg);
  transition: var(--transition);
  background: var(--bg-secondary);
  cursor: pointer;
  max-width: 100%;
  width: 100%;
}

.upload-area:hover {
  border-color: var(--accent-color);
  background-color: var(--bg-tertiary);
  transform: translateY(-2px);
  box-shadow: var(--shadow-sm);
}

.upload-label {
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-lg);
  border-radius: var(--border-radius);
  transition: var(--transition);
}

.upload-icon {
  font-size: 64px;
  margin-bottom: var(--spacing-md);
  color: var(--text-secondary);
  opacity: 0.8;
  transition: var(--transition);
}

.upload-area:hover .upload-icon {
  opacity: 1;
  transform: scale(1.1);
}

.upload-label p {
  margin: 0;
  color: var(--text-secondary);
  font-size: 18px;
  font-weight: 600;
}

.upload-label .small-text {
  font-size: 14px;
  color: var(--text-tertiary);
  font-weight: normal;
  line-height: 1.4;
}

.upload-status {
  margin: var(--spacing-lg) 0;
  padding: var(--spacing-md);
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  border: 1px solid var(--border-light);
  box-shadow: var(--shadow-sm);
}

.progress-bar {
  width: 100%;
  height: 10px;
  background-color: var(--bg-tertiary);
  border-radius: 5px;
  overflow: hidden;
  margin-bottom: var(--spacing-sm);
}

.progress-fill {
  height: 100%;
  background-color: var(--text-secondary);
  transition: width 0.3s ease;
  border-radius: 5px;
}

.domain-selection {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  border: 1px solid var(--border-light);
  box-shadow: var(--shadow-sm);
}

.domain-selection label {
  font-weight: 500;
  color: var(--text-primary);
  font-size: 14px;
}

.domain-selection select {
  padding: var(--spacing-md);
  border: 1px solid var(--border-light);
  border-radius: var(--border-radius);
  background: var(--bg-primary);
  color: var(--text-primary);
  font-size: 14px;
  transition: var(--transition);
  outline: none;
}

.domain-selection select:focus {
  border-color: var(--accent-color);
  box-shadow: 0 0 0 2px rgba(100, 116, 139, 0.1);
}

.domain-selection select:focus {
  outline: none;
  border-color: var(--text-secondary);
  box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.05);
}

/* Browse Tab Styles */
.browse-controls {
  margin-bottom: var(--spacing-xl);
}

.search-box {
  display: flex;
  gap: var(--spacing-sm);
  align-items: center;
  flex-wrap: wrap;
  padding: var(--spacing-md);
  background: var(--bg-secondary);
  border-radius: var(--border-radius-lg);
  border: 1px solid var(--border-light);
  box-shadow: var(--shadow-sm);
}

.search-box input {
  flex: 1;
  min-width: 300px;
  padding: var(--spacing-md) calc(var(--spacing-md) + 24px);
  border: 1px solid var(--border-light);
  border-radius: var(--border-radius);
  background: var(--bg-primary);
  color: var(--text-primary);
  font-size: 14px;
  transition: var(--transition);
  position: relative;
}

.search-box input:focus {
  outline: none;
  border-color: var(--accent-color);
  box-shadow: 0 0 0 2px rgba(100, 116, 139, 0.1);
}

.search-box input::placeholder {
  color: var(--text-tertiary);
}

.search-box select {
  padding: var(--spacing-md);
  border: 1px solid var(--border-light);
  border-radius: var(--border-radius);
  min-width: 150px;
  background: var(--bg-primary);
  color: var(--text-primary);
  font-size: 14px;
  transition: var(--transition);
  cursor: pointer;
}

.search-box select:focus {
  outline: none;
  border-color: var(--accent-color);
  box-shadow: 0 0 0 2px rgba(100, 116, 139, 0.1);
}

.search-box button {
  padding: var(--spacing-md) var(--spacing-lg);
  background: var(--text-primary);
  color: white;
  border: none;
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: var(--transition);
  white-space: nowrap;
  min-width: 100px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-xs);
}

.search-box button:hover {
  background: var(--text-secondary);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}

.search-box button:active {
  transform: translateY(0);
}

.search-results h3 {
  margin-top: var(--spacing-xl);
  margin-bottom: var(--spacing-lg);
  color: var(--text-primary);
  font-size: 18px;
  padding: 0 var(--spacing-sm);
  font-weight: 600;
}

.result-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.result-item {
  border: 1px solid var(--border-light);
  border-radius: var(--border-radius);
  padding: var(--spacing-lg);
  background: var(--bg-primary);
  transition: var(--transition);
  box-shadow: var(--shadow-sm);
}

.result-item:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
  border-color: var(--accent-light);
}

.result-domain {
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: var(--spacing-xs);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  background: var(--bg-secondary);
  padding: 2px 8px;
  border-radius: 4px;
  display: inline-block;
}

.result-content {
  margin-bottom: var(--spacing-xs);
  line-height: 1.6;
  color: var(--text-secondary);
  font-size: 14px;
  padding: var(--spacing-sm) 0;
}

.result-source {
  font-size: 12px;
  color: var(--text-tertiary);
  padding-top: var(--spacing-xs);
  border-top: 1px solid var(--border-light);
}

.no-results {
  text-align: center;
  padding: var(--spacing-xl);
  color: var(--text-tertiary);
  font-style: italic;
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  margin-top: var(--spacing-lg);
  border: 1px solid var(--border-color);
}

/* Stats Tab Styles */
.stats-container {
  padding: 0;
}

.loading {
  text-align: center;
  padding: var(--spacing-xl);
  color: var(--text-tertiary);
  font-style: italic;
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  margin: var(--spacing-lg) 0;
  border: 1px solid var(--border-light);
  box-shadow: var(--shadow-sm);
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: var(--spacing-lg);
  margin-bottom: var(--spacing-xl);
  padding: var(--spacing-md);
  background: var(--bg-secondary);
  border-radius: var(--border-radius-lg);
  border: 1px solid var(--border-light);
  box-shadow: var(--shadow-sm);
}

.stat-card {
  background: var(--bg-primary);
  padding: var(--spacing-xl);
  border-radius: var(--border-radius);
  text-align: center;
  border: 1px solid var(--border-light);
  transition: var(--transition);
  min-height: 120px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  box-shadow: var(--shadow-sm);
  position: relative;
  overflow: hidden;
}

.stat-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 4px;
  height: 100%;
  background: var(--accent-color);
}

.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
  border-color: var(--accent-light);
}

.stat-value {
  font-size: 36px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: var(--spacing-xs);
  letter-spacing: -0.025em;
  line-height: 1.2;
}

.stat-label {
  color: var(--text-secondary);
  font-size: 14px;
  font-weight: 500;
  text-transform: capitalize;
}

.domain-stats {
  margin-top: var(--spacing-xl);
  padding: var(--spacing-md);
  background: var(--bg-secondary);
  border-radius: var(--border-radius-lg);
  border: 1px solid var(--border-light);
  box-shadow: var(--shadow-sm);
}

.domain-stats h4 {
  margin-bottom: var(--spacing-lg);
  color: var(--text-primary);
  font-size: 16px;
  padding: 0 var(--spacing-sm);
  font-weight: 600;
}

.domain-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.domain-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-lg);
  transition: var(--transition);
  background: var(--bg-primary);
  border-radius: var(--border-radius);
  border: 1px solid var(--border-light);
  box-shadow: var(--shadow-sm);
}

.domain-item:hover {
  background-color: var(--bg-tertiary);
  transform: translateX(2px);
  box-shadow: var(--shadow-md);
  border-color: var(--accent-light);
}

.domain-name {
  font-weight: 500;
  flex: 2;
  color: var(--text-primary);
  font-size: 14px;
}

.domain-count {
  color: var(--text-secondary);
  flex: 1;
  text-align: center;
  font-size: 14px;
  font-weight: 600;
}

.domain-updated {
  color: var(--text-tertiary);
  font-size: 12px;
  flex: 1;
  text-align: right;
}

/* Manage Tab Styles */
.manage-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-lg);
  padding-bottom: var(--spacing-md);
  border-bottom: 1px solid var(--border-light);
  flex-wrap: wrap;
  gap: var(--spacing-md);
  padding: 0 var(--spacing-sm) var(--spacing-md);
}

.manage-header h3 {
  margin: 0;
  color: var(--text-primary);
  font-size: 18px;
  font-weight: 600;
}

.refresh-btn {
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border-light);
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 14px;
  transition: var(--transition);
  min-width: 100px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-xs);
  box-shadow: var(--shadow-sm);
}

.refresh-btn:hover {
  background: var(--bg-tertiary);
  border-color: var(--accent-color);
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

.files-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-lg);
  gap: var(--spacing-md);
  flex-wrap: wrap;
  padding: var(--spacing-lg);
  background: var(--bg-secondary);
  border-radius: var(--border-radius-lg);
  border: 1px solid var(--border-light);
  box-shadow: var(--shadow-sm);
}

.filter-controls, .sort-controls {
  display: flex;
  gap: var(--spacing-sm);
  align-items: center;
  flex-wrap: wrap;
}

.filter-controls select,
.filter-controls input,
.sort-controls select {
  padding: var(--spacing-sm);
  border: 1px solid var(--border-light);
  border-radius: var(--border-radius);
  background: var(--bg-primary);
  color: var(--text-primary);
  font-size: 14px;
  transition: var(--transition);
  min-width: 180px;
}

.filter-controls select:focus,
.filter-controls input:focus,
.sort-controls select:focus {
  outline: none;
  border-color: var(--accent-color);
  box-shadow: 0 0 0 2px var(--accent-light);
}

.files-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: var(--spacing-lg);
  margin-bottom: var(--spacing-lg);
}

.file-card {
  background: var(--bg-primary);
  border: 1px solid var(--border-light);
  border-radius: var(--border-radius-lg);
  padding: var(--spacing-lg);
  transition: var(--transition);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}

.file-card:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
  border-color: var(--accent-light);
}

.file-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: var(--spacing-sm);
  flex-wrap: wrap;
  gap: var(--spacing-xs);
}

.file-name {
  font-weight: 600;
  color: var(--text-primary);
  flex: 1;
  font-size: 16px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-domain {
  font-size: 12px;
  color: var(--text-secondary);
  background: var(--bg-secondary);
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--border-radius-full);
  white-space: nowrap;
  font-weight: 500;
  border: 1px solid var(--border-light);
}

.file-details {
  display: flex;
  justify-content: space-between;
  margin-bottom: var(--spacing-md);
  font-size: 12px;
  color: var(--text-tertiary);
  gap: var(--spacing-sm);
  padding: var(--spacing-md) 0;
  border-top: 1px solid var(--border-light);
  border-bottom: 1px solid var(--border-light);
}

.file-actions {
  display: flex;
  gap: var(--spacing-sm);
  margin-top: auto;
  padding-top: var(--spacing-sm);
}

.action-btn {
  padding: var(--spacing-xs) var(--spacing-sm);
  border: 1px solid var(--border-light);
  background: transparent;
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 12px;
  transition: var(--transition);
  flex: 1;
  text-align: center;
  min-height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.action-btn:hover {
  background: var(--bg-secondary);
  border-color: var(--accent-color);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}

.action-btn.view {
  color: var(--accent-color);
  border-color: var(--accent-color);
}

.action-btn.view:hover {
  background: rgba(79, 70, 229, 0.1);
}

.action-btn.download {
  color: #10b981;
  border-color: #10b981;
}

.action-btn.download:hover {
  background: rgba(16, 185, 129, 0.1);
}

.action-btn.delete {
  color: #ef4444;
  border-color: #ef4444;
  background-color: transparent;
}

.action-btn.delete:hover {
  background-color: rgba(239, 68, 68, 0.1);
}

.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: var(--spacing-xs);
  margin-top: var(--spacing-xl);
  padding: var(--spacing-md);
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  border: 1px solid var(--border-light);
  box-shadow: var(--shadow-sm);
}

.pagination button {
  padding: var(--spacing-xs) var(--spacing-md);
  border: 1px solid var(--border-light);
  background: var(--bg-primary);
  border-radius: var(--border-radius);
  cursor: pointer;
  color: var(--text-secondary);
  font-size: 14px;
  transition: var(--transition);
  min-width: 40px;
  box-shadow: var(--shadow-sm);
}

.pagination button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.pagination button:not(:disabled):hover {
  background: var(--bg-tertiary);
  border-color: var(--accent-color);
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

.no-files {
  text-align: center;
  padding: var(--spacing-xl);
  color: var(--text-tertiary);
  font-style: italic;
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  margin-top: var(--spacing-lg);
  border: 1px solid var(--border-light);
  box-shadow: var(--shadow-sm);
}

/* Modal Styles */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
  padding: var(--spacing-lg);
  backdrop-filter: blur(2px);
}

.modal {
  background: var(--bg-primary);
  padding: var(--spacing-xl);
  border-radius: var(--border-radius-lg);
  box-shadow: var(--shadow-lg);
  border: 1px solid var(--border-light);
  max-width: 500px;
  width: 100%;
  max-height: 90vh;
  overflow-y: auto;
  transition: var(--transition);
  transform: scale(1);
}

.modal-enter {
  opacity: 0;
  transform: scale(0.95);
}

.modal-enter-active {
  opacity: 1;
  transform: scale(1);
  transition: opacity 300ms, transform 300ms;
}

.modal-exit {
  opacity: 1;
  transform: scale(1);
}

.modal-exit-active {
  opacity: 0;
  transform: scale(0.95);
  transition: opacity 300ms, transform 300ms;
}

.modal h3 {
  margin-bottom: var(--spacing-md);
  color: var(--text-primary);
  font-size: 18px;
  font-weight: 600;
}

.modal p {
  margin-bottom: var(--spacing-lg);
  color: var(--text-secondary);
  line-height: 1.6;
}

.modal-actions {
  display: flex;
  gap: var(--spacing-md);
  justify-content: flex-end;
  margin-top: var(--spacing-lg);
}

.confirm-btn {
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--text-primary);
  color: white;
  border: none;
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: var(--transition);
  box-shadow: var(--shadow-sm);
}

.confirm-btn:hover {
  background: var(--text-secondary);
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

.cancel-btn {
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--bg-primary);
  color: var(--text-secondary);
  border: 1px solid var(--border-light);
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 14px;
  transition: var(--transition);
}

.cancel-btn:hover {
  background: var(--bg-secondary);
  border-color: var(--border-color);
}

/* Preview Modal Styles */
.preview-modal {
  max-width: 800px;
  width: 100%;
}

.preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-md);
  padding-bottom: var(--spacing-md);
  border-bottom: 1px solid var(--border-light);
}

.preview-header h3 {
  margin: 0;
  font-size: 16px;
  color: var(--text-primary);
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 600;
}

.close-btn {
  background: var(--bg-secondary);
  border: 1px solid var(--border-light);
  font-size: 20px;
  cursor: pointer;
  color: var(--text-secondary);
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: var(--transition);
  box-shadow: var(--shadow-sm);
}

.close-btn:hover {
  background: var(--bg-tertiary);
  color: var(--text-primary);
  border-color: var(--border-color);
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

.preview-content {
  max-height: 500px;
  overflow-y: auto;
  margin-bottom: var(--spacing-lg);
  padding: var(--spacing-md);
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  border: 1px solid var(--border-light);
}

.loading-preview, .preview-error, .unsupported-preview {
  text-align: center;
  padding: var(--spacing-xl);
  color: var(--text-tertiary);
}

.text-preview {
  background: var(--bg-primary);
  padding: var(--spacing-lg);
  border-radius: var(--border-radius);
  border: 1px solid var(--border-light);
  max-height: 500px;
  overflow-y: auto;
  box-shadow: var(--shadow-sm);
}

.text-preview pre {
  margin: 0;
  font-family: 'Courier New', Courier, monospace;
  font-size: 13px;
  line-height: 1.5;
  color: var(--text-primary);
  white-space: pre-wrap;
  word-wrap: break-word;
}

.image-preview {
  text-align: center;
  max-height: 500px;
  overflow-y: auto;
}

.image-preview img {
  max-width: 100%;
  max-height: 400px;
  border-radius: var(--border-radius);
  border: 1px solid var(--border-light);
  box-shadow: var(--shadow-md);
  transition: var(--transition);
}

.image-preview img:hover {
  transform: scale(1.01);
  box-shadow: var(--shadow-lg);
}

.file-info {
  text-align: left;
  margin-top: var(--spacing-md);
  font-size: 13px;
  line-height: 1.6;
  color: var(--text-secondary);
  padding: var(--spacing-md);
  background: var(--bg-primary);
  border-radius: var(--border-radius);
  border: 1px solid var(--border-light);
}

.preview-actions {
  display: flex;
  gap: var(--spacing-sm);
  justify-content: flex-end;
  padding-top: var(--spacing-md);
  border-top: 1px solid var(--border-light);
}

/* Responsive Design */
@media (max-width: 768px) {
  .knowledge-view {
    padding: var(--spacing-md);
  }
  
  .header {
    margin-bottom: var(--spacing-md);
  }
  
  .header h1 {
    font-size: 24px;
  }
  
  .header-actions {
    justify-content: center;
  }
  
  .content {
    padding: var(--spacing-md);
  }
  
  .files-grid {
    grid-template-columns: 1fr;
  }
  
  .files-controls {
    flex-direction: column;
    align-items: stretch;
  }
  
  .filter-controls, .sort-controls {
    justify-content: center;
  }
  
  .search-box {
    flex-direction: column;
    align-items: stretch;
  }
  
  .search-box input, .search-box select {
    width: 100%;
  }
  
  .stats-grid {
    grid-template-columns: 1fr 1fr;
  }
  
  .domain-item {
    flex-direction: column;
    align-items: flex-start;
    gap: var(--spacing-xs);
    padding: var(--spacing-md);
  }
  
  .domain-name, .domain-count, .domain-updated {
    flex: none;
    text-align: left;
  }
}

@media (max-width: 480px) {
  .header-actions button {
    font-size: 12px;
    padding: var(--spacing-xs) var(--spacing-sm);
  }
  
  .stats-grid {
    grid-template-columns: 1fr;
  }
  
  .modal {
    padding: var(--spacing-md);
  }
  
  .modal-actions {
    flex-direction: column;
  }
  
  .modal-actions button {
    width: 100%;
  }
}

/* Auto Learning Styles */
.auto-learning-toggle {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  margin-left: var(--spacing-md);
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  border: 1px solid var(--border-light);
}

/* Switch Styles */
.switch {
  position: relative;
  display: inline-block;
  width: 44px;
  height: 24px;
}

.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: var(--border-light);
  transition: var(--transition);
  border-radius: 24px;
  border: 1px solid var(--border-color);
}

.slider:before {
  position: absolute;
  content: "";
  height: 18px;
  width: 18px;
  left: 2px;
  bottom: 2px;
  background-color: white;
  transition: var(--transition);
  border-radius: 50%;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

input:checked + .slider {
  background-color: var(--accent-color);
  border-color: var(--accent-color);
}

input:focus + .slider {
  box-shadow: 0 0 0 2px var(--accent-light);
}

input:checked + .slider:before {
  transform: translateX(20px);
}

/* Learning Progress Styles */
.learning-progress {
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  padding: var(--spacing-lg);
  margin-top: var(--spacing-lg);
  border: 1px solid var(--border-light);
  box-shadow: var(--shadow-sm);
}

.progress-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-xs);
  font-size: 14px;
}

.status-text {
  color: var(--text-primary);
  font-weight: 600;
}

.progress-percentage {
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 500;
}

.status-badge {
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 13px;
  font-weight: 500;
  text-transform: capitalize;
  border: 1px solid var(--border-light);
}

.status-idle {
  background: var(--bg-tertiary);
  color: var(--text-secondary);
}

.status-running {
  background: rgba(79, 70, 229, 0.1);
  color: var(--accent-color);
  border: 1px solid var(--accent-light);
}

.status-paused {
  background: rgba(251, 191, 36, 0.1);
  color: #d97706;
  border: 1px solid rgba(251, 191, 36, 0.3);
}

.status-completed {
  background: rgba(16, 185, 129, 0.1);
  color: #059669;
  border: 1px solid rgba(16, 185, 129, 0.3);
}

/* Learning Modal Styles */
.learning-modal-content {
  max-height: 60vh;
  overflow-y: auto;
}

.learning-status {
  margin-bottom: var(--spacing-lg);
  padding: var(--spacing-lg);
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  border: 1px solid var(--border-light);
}

.learning-status h3 {
  margin-bottom: var(--spacing-sm);
  color: var(--text-primary);
  font-size: 16px;
  font-weight: 600;
}

.learning-logs {
  margin-bottom: var(--spacing-xl);
}

.learning-logs h3 {
  margin-bottom: var(--spacing-md);
  color: var(--text-primary);
  font-size: 16px;
  font-weight: 600;
}

.logs-container {
  background: var(--bg-primary);
  border: 1px solid var(--border-light);
  border-radius: var(--border-radius);
  padding: var(--spacing-md);
  max-height: 200px;
  overflow-y: auto;
  font-size: 13px;
  line-height: 1.5;
  box-shadow: var(--shadow-sm);
}

.log-item {
  display: flex;
  align-items: flex-start;
  margin-bottom: var(--spacing-sm);
  padding-bottom: var(--spacing-sm);
  border-bottom: 1px solid var(--border-light);
  padding: var(--spacing-xs);
  transition: var(--transition);
  border-radius: var(--border-radius-xs);
}

.log-item:hover {
  background: var(--bg-secondary);
}

.log-item:last-child {
  border-bottom: none;
  margin-bottom: 0;
  padding-bottom: 0;
}

.log-time {
  color: var(--text-tertiary);
  margin-right: var(--spacing-md);
  min-width: 80px;
  font-size: 12px;
  font-weight: 500;
}

.log-source {
  font-weight: 600;
  margin-right: var(--spacing-sm);
  color: var(--accent-color);
  min-width: 80px;
  font-size: 12px;
}

.log-message {
  color: var(--text-primary);
  flex: 1;
}

.no-logs,
.no-history {
  text-align: center;
  color: var(--text-tertiary);
  padding: var(--spacing-lg);
  font-style: italic;
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  border: 1px solid var(--border-light);
}

.learning-history {
  margin-top: var(--spacing-lg);
  padding: var(--spacing-lg);
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  border: 1px solid var(--border-light);
}

.learning-history h3 {
  margin-bottom: var(--spacing-md);
  color: var(--text-primary);
  font-size: 16px;
  font-weight: 600;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: var(--spacing-md);
}

/* Responsive adjustments for auto learning */
@media (max-width: 768px) {
  .header-controls {
    flex-direction: column;
    align-items: stretch;
    gap: var(--spacing-sm);
  }
  
  .auto-learning-toggle {
    justify-content: center;
    margin-left: 0;
  }
  
  .log-item {
    flex-direction: column;
    gap: 2px;
  }
  
  .log-time,
  .log-source {
    min-width: auto;
  }
}
</style>
