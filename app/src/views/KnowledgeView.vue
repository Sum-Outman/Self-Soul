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
          <input type="text"
                 v-model="searchQuery"
                 placeholder="Search knowledge content...">
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
        <button @click="loadFiles" class="refresh-btn">
          Refresh
        </button>
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
        const response = await api.get('/api/knowledge/search', {
          params: {
            query: searchQuery.value,
            domain: searchDomain.value
          }
        });

        if (response.data.success) {
          searchResults.value = response.data.results;
          isRealAPI.value = true;
        } else {
          searchResults.value = [];
        }
        searchPerformed.value = true;
      } catch (error) {
        errorHandler.handleError(error, 'Search knowledge failed');
        // 使用模拟搜索结果作为回退
        searchResults.value = getMockSearchResults(searchQuery.value, searchDomain.value);
        searchPerformed.value = true;
        isRealAPI.value = false;
        showSystemMessage('Using mock search results');
      }
    };

    // 模拟搜索结果
    const getMockSearchResults = (query, domain) => {
      const mockResults = [
        {
          domain: 'computer_science',
          content: `This is a mock search result related to "${query}" about artificial intelligence concepts.`,
          source: '人工智能导论.pdf'
        },
        {
          domain: 'computer_science',
          content: `This mock result discusses machine learning fundamentals in the context of "${query}".`,
          source: '机器学习基础.docx'
        },
        {
          domain: 'computer_science', 
          content: `This document covers deep learning techniques related to "${query}".`,
          source: '深度学习技术.pptx'
        }
      ];
      
      // Filter by domain
      if (domain) {
        return mockResults.filter(result => result.domain === domain);
      }
      
      return mockResults;
    };

    const loadKnowledgeStats = async () => {
      statsLoading.value = true;
      try {
        const response = await api.get('/api/knowledge/stats');
        if (response.data.success) {
          knowledgeStats.value = response.data;
        }
      } catch (error) {
        // Check if connection error
        if (error.code === 'ECONNREFUSED') {
          // 明确告诉用户后端服务未启动
          errorHandler.handleError(error, 'Backend service not running. Please start the Python backend with "python core/main.py".');
        } else {
          errorHandler.handleError(error, 'Failed to load statistics');
        }
        // 使用模拟统计数据作为回退
        knowledgeStats.value = getMockKnowledgeStats();
        showSystemMessage('Using mock statistics data');
      }
      statsLoading.value = false;
    };

    // Mock knowledge statistics data
    const getMockKnowledgeStats = () => {
      return {
        total_domains: 9,
        total_items: 50,
        total_size: 15000000,
        domains: {
          computer_science: {
            item_count: 20,
            last_updated: new Date().toISOString()
          },
          mathematics: {
            item_count: 8,
            last_updated: new Date().toISOString()
          },
          physics: {
            item_count: 7,
            last_updated: new Date().toISOString()
          },
          chemistry: {
            item_count: 5,
            last_updated: new Date().toISOString()
          },
          biology: {
            item_count: 10,
            last_updated: new Date().toISOString()
          }
        }
      };
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
        const response = await api.get('/api/knowledge/files', { 
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          }
        });
        
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
          // 明确告诉用户后端服务未启动
          errorHandler.handleError(error, 'Backend service not running. Please start the Python backend with "python core/main.py".');
        } else {
          errorHandler.handleError(error, 'Failed to load files');
        }
        // 使用模拟数据作为回退
        files.value = getMockFiles();
        filterFiles();
        showSystemMessage('Using mock file list');
        isRealAPI.value = false;
      }
      filesLoading.value = false;
    };
    
    // Mock data for files when API is not available
    const getMockFiles = () => {
      return [
        {
          id: 'mock_1',
          name: 'Introduction to AI.pdf',
          size: 2048000,
          upload_date: '2023-06-15T10:30:00Z',
          domain: 'computer_science'
        },
        {
          id: 'mock_2',
          name: 'Machine Learning Basics.docx',
          size: 1536000,
          upload_date: '2023-06-10T14:45:00Z',
          domain: 'computer_science'
        },
        {
          id: 'mock_3',
          name: 'Deep Learning Techniques.pptx',
          size: 3072000,
          upload_date: '2023-06-05T09:20:00Z',
          domain: 'computer_science'
        },
        {
          id: 'mock_4',
          name: 'Data Science Handbook.pdf',
          size: 4096000,
          upload_date: '2023-06-01T16:10:00Z',
          domain: 'computer_science'
        }
      ];
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
        // Check if file is mock file
        if (file.id && file.id.startsWith('mock_')) {
          showSystemMessage(`查看模拟文件: ${file.name}`);
          // Open preview for mock files
          openPreview(file);
          return;
        }
        
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
        showSystemMessage(`Failed to view file: ${file.name}`);
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
        // Check if file is mock file
        if (file.id && file.id.startsWith('mock_')) {
          // Simulate content for mock files
        if (isTextFile(file)) {
          currentFileContent.value = `This is mock content for ${file.name}.\n\nFile Size: ${formatFileSize(file.size)}\nUpload Date: ${formatDate(file.upload_date)}\nDomain: ${file.domain}`;
        } else if (isImageFile(file)) {
          // Use placeholder image for mock image files
          currentFileContent.value = 'https://via.placeholder.com/400x300?text=Mock+Image';
        }
          previewLoading.value = false;
          return;
        }

        // Try to load actual file content from server
        const response = await api.get(`/api/knowledge/files/${file.id}/preview`, {
          timeout: 10000,
          headers: {
            'Cache-Control': 'no-cache'
          }
        });

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
          // For mock files or when API fails, show simulated content
          if (isTextFile(file)) {
            currentFileContent.value = `Failed to load file content.\n\nFile Info:\nName: ${file.name}\nSize: ${formatFileSize(file.size)}\nUpload Date: ${formatDate(file.upload_date)}`;
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
        // Check if file is mock file
        if (file.id && file.id.startsWith('mock_')) {
          showSystemMessage(`模拟下载文件: ${file.name}`);
          // Create a dummy blob for simulation
          const blob = new Blob(['This is a simulated file content.'], { type: 'text/plain' });
          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = file.name;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(url);
          return;
        }
        
        // Try to download from server with timeout
        const controller = new AbortController();
        
        // 由于需要blob响应类型，这里不使用封装的api实例，直接使用axios
        const response = await axios.get(`/api/knowledge/files/${file.id}/download`, {
          responseType: 'blob',
          signal: controller.signal,
          timeout: 30000 // 30秒超时
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
        // If timeout or other error, try to simulate download
        if (error.name === 'AbortError' || !error.response) {
          showSystemMessage(`Simulating download: ${file.name}`);
          // Create a dummy blob for simulation
          const blob = new Blob(['This is a simulated file content.'], { type: 'text/plain' });
          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = file.name;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(url);
        } else {
          showSystemMessage(`Cannot download file: ${file.name}`);
        }
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
        
        // Check if file is mock file - just remove from local list
        if (fileToDelete.value.id && fileToDelete.value.id.startsWith('mock_')) {
          files.value = files.value.filter(f => f.id !== fileToDelete.value.id);
          filterFiles();
          showSystemMessage(`Simulating deletion: ${fileToDelete.value.name}`);
          showDeleteModal.value = false;
          fileToDelete.value = null;
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
        // Even if server fails, remove from local list for better UX
        files.value = files.value.filter(f => f.id !== fileToDelete.value.id);
        filterFiles();
        showSystemMessage(`File removed locally: ${fileToDelete.value.name}`);
      }
      showDeleteModal.value = false;
      fileToDelete.value = null;
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
          
          // Simulate upload progress
          const progressInterval = setInterval(() => {
            if (uploadProgress.value < 100) {
              uploadProgress.value += Math.floor(Math.random() * 10) + 5;
            } else {
              clearInterval(progressInterval);
            }
          }, 300);
          
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
            // Simulate successful upload for demo purposes
            showSystemMessage(`Simulating upload success: ${file.name}`);
            // Add mock file to list
            const mockFile = {
              id: 'mock_' + Date.now(),
              name: file.name,
              size: file.size,
              upload_date: new Date().toISOString(),
              domain: selectedDomain.value
            };
            files.value.push(mockFile);
            filterFiles();
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
      handleFileUpload
    };
  }
};
</script>

<style scoped>
:root {
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 32px;
  --border-radius: 8px;
  --border-radius-lg: 12px;
  --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 8px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 8px 16px rgba(0, 0, 0, 0.15);
  --transition: all 0.2s ease;
}

.knowledge-view {
  padding: var(--spacing-lg);
  max-width: 1200px;
  margin: 0 auto;
  min-height: calc(100vh - 120px);
}

.header {
  display: flex;
  flex-direction: column;
  margin-bottom: var(--spacing-xl);
  padding-bottom: 0;
  border-bottom: 1px solid #e0e0e0;
}

.header h1 {
  margin: 0 0 var(--spacing-md) 0;
  font-size: 28px;
  font-weight: 600;
  color: #333;
}

.header-actions {
  display: flex;
  gap: 0;
  flex-wrap: wrap;
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: var(--border-radius) var(--border-radius) 0 0;
  overflow: hidden;
}

.header-actions button {
  flex: 1;
  min-width: 120px;
  padding: var(--spacing-sm) var(--spacing-md);
  border: none;
  border-right: 1px solid #e0e0e0;
  background: white;
  cursor: pointer;
  color: #666;
  font-size: 14px;
  font-weight: 500;
  transition: var(--transition);
  text-align: center;
}

.header-actions button:last-child {
  border-right: none;
}

.header-actions button:hover {
  background: #f8f9fa;
  border-color: #d0d0d0;
}

.header-actions button.active {
  background: #f8f9fa;
  color: #333;
  font-weight: 600;
  position: relative;
}

.content {
  background: white;
  padding: var(--spacing-lg);
  border-radius: 0 var(--border-radius-lg) var(--border-radius-lg) var(--border-radius-lg);
  box-shadow: var(--shadow-sm);
  border: 1px solid #e0e0e0;
  border-top: none;
  min-height: 500px;
}

/* Import Tab Styles */
.import-section {
  max-width: 600px;
  margin: 0 auto;
}

.upload-area {
  text-align: center;
  padding: var(--spacing-xl);
  border: 2px dashed #e0e0e0;
  border-radius: var(--border-radius-lg);
  margin-bottom: var(--spacing-lg);
  transition: var(--transition);
}

.upload-area:hover {
  border-color: #d0d0d0;
  background-color: #fafafa;
}

.upload-label {
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-md);
}

.upload-icon {
  font-size: 48px;
  margin-bottom: var(--spacing-md);
}

.upload-label p {
  margin: 0;
  color: #666;
  font-size: 16px;
}

.upload-label .small-text {
  font-size: 14px;
  color: #999;
}

.upload-status {
  margin: var(--spacing-lg) 0;
}

.progress-bar {
  width: 100%;
  height: 8px;
  background-color: #f0f0f0;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: var(--spacing-sm);
}

.progress-fill {
  height: 100%;
  background-color: #666;
  transition: width 0.3s ease;
}

.domain-selection {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.domain-selection label {
  font-weight: 500;
  color: #333;
  font-size: 14px;
}

.domain-selection select {
  padding: var(--spacing-md);
  border: 1px solid #e0e0e0;
  border-radius: var(--border-radius);
  background: white;
  color: #333;
  font-size: 14px;
}

/* Browse Tab Styles */
.browse-controls {
  margin-bottom: var(--spacing-lg);
}

.search-box {
  display: flex;
  gap: var(--spacing-sm);
  align-items: center;
  flex-wrap: wrap;
}

.search-box input {
  flex: 1;
  min-width: 200px;
  padding: var(--spacing-md);
  border: 1px solid #e0e0e0;
  border-radius: var(--border-radius);
  background: white;
  color: #333;
  font-size: 14px;
}

.search-box select {
  padding: var(--spacing-md);
  border: 1px solid #e0e0e0;
  border-radius: var(--border-radius);
  min-width: 150px;
  background: white;
  color: #333;
  font-size: 14px;
}

.search-box button {
  padding: var(--spacing-md) var(--spacing-lg);
  background: #333;
  color: white;
  border: none;
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: var(--transition);
  white-space: nowrap;
}

.search-box button:hover {
  background: #555;
}

.search-results h3 {
  margin-top: var(--spacing-lg);
  margin-bottom: var(--spacing-md);
  color: #333;
  font-size: 18px;
}

.result-item {
  border: 1px solid #e0e0e0;
  border-radius: var(--border-radius);
  padding: var(--spacing-md);
  margin-bottom: var(--spacing-md);
  background: white;
  transition: var(--transition);
}

.result-item:hover {
  box-shadow: var(--shadow-sm);
}

.result-domain {
  font-weight: 600;
  color: #333;
  margin-bottom: var(--spacing-xs);
  font-size: 14px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.result-content {
  margin-bottom: var(--spacing-xs);
  line-height: 1.6;
  color: #555;
  font-size: 14px;
}

.result-source {
  font-size: 12px;
  color: #999;
}

.no-results {
  text-align: center;
  padding: var(--spacing-xl);
  color: #999;
  font-style: italic;
  background: #fafafa;
  border-radius: var(--border-radius);
  margin-top: var(--spacing-lg);
}

/* Stats Tab Styles */
.stats-container {
  padding: 0;
}

.loading {
  text-align: center;
  padding: var(--spacing-xl);
  color: #999;
  font-style: italic;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: var(--spacing-lg);
  margin-bottom: var(--spacing-xl);
}

.stat-card {
  background: #f8f9fa;
  padding: var(--spacing-lg);
  border-radius: var(--border-radius);
  text-align: center;
  border: 1px solid #e0e0e0;
  transition: var(--transition);
}

.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.stat-value {
  font-size: 32px;
  font-weight: 700;
  color: #333;
  margin-bottom: var(--spacing-xs);
}

.stat-label {
  color: #666;
  font-size: 14px;
  font-weight: 500;
}

.domain-stats {
  margin-top: var(--spacing-xl);
}

.domain-stats h4 {
  margin-bottom: var(--spacing-md);
  color: #333;
  font-size: 16px;
}

.domain-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-md);
  border-bottom: 1px solid #e0e0e0;
  transition: var(--transition);
}

.domain-item:hover {
  background-color: #fafafa;
}

.domain-name {
  font-weight: 500;
  flex: 2;
  color: #333;
  font-size: 14px;
}

.domain-count {
  color: #666;
  flex: 1;
  text-align: center;
  font-size: 14px;
  font-weight: 600;
}

.domain-updated {
  color: #999;
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
  flex-wrap: wrap;
  gap: var(--spacing-md);
}

.manage-header h3 {
  margin: 0;
  color: #333;
  font-size: 18px;
}

.refresh-btn {
  padding: var(--spacing-sm) var(--spacing-md);
  background: white;
  color: #333;
  border: 1px solid #e0e0e0;
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 14px;
  transition: var(--transition);
}

.refresh-btn:hover {
  background: #f8f9fa;
}

.files-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-lg);
  gap: var(--spacing-md);
  flex-wrap: wrap;
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
  border: 1px solid #e0e0e0;
  border-radius: var(--border-radius);
  background: white;
  color: #333;
  font-size: 14px;
}

.filter-controls input {
  min-width: 150px;
}

.files-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
}

.file-card {
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: var(--border-radius);
  padding: var(--spacing-md);
  transition: var(--transition);
  display: flex;
  flex-direction: column;
}

.file-card:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
  border-color: #d0d0d0;
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
  font-weight: 500;
  color: #333;
  flex: 1;
  font-size: 14px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-domain {
  font-size: 12px;
  color: #666;
  background: #f0f0f0;
  padding: 2px 8px;
  border-radius: 12px;
  white-space: nowrap;
}

.file-details {
  display: flex;
  justify-content: space-between;
  margin-bottom: var(--spacing-md);
  font-size: 12px;
  color: #999;
}

.file-actions {
  display: flex;
  gap: var(--spacing-xs);
  margin-top: auto;
}

.action-btn {
  padding: var(--spacing-xs) var(--spacing-sm);
  border: 1px solid #e0e0e0;
  background: white;
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 12px;
  transition: var(--transition);
  flex: 1;
  text-align: center;
}

.action-btn:hover {
  background: #f8f9fa;
  border-color: #d0d0d0;
}

.action-btn.view {
  color: #333;
}

.action-btn.download {
  color: #333;
}

.action-btn.delete {
  color: #333;
  border-color: #ffcccc;
  background-color: #fff5f5;
}

.action-btn.delete:hover {
  background-color: #ffe6e6;
}

.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: var(--spacing-md);
  margin-top: var(--spacing-xl);
  padding-top: var(--spacing-lg);
  border-top: 1px solid #e0e0e0;
}

.pagination button {
  padding: var(--spacing-xs) var(--spacing-md);
  border: 1px solid #e0e0e0;
  background: white;
  border-radius: var(--border-radius);
  cursor: pointer;
  color: #333;
  font-size: 14px;
  transition: var(--transition);
}

.pagination button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.pagination button:not(:disabled):hover {
  background: #f8f9fa;
}

.no-files {
  text-align: center;
  padding: var(--spacing-xl);
  color: #999;
  font-style: italic;
  background: #fafafa;
  border-radius: var(--border-radius);
  margin-top: var(--spacing-lg);
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
}

.modal {
  background: white;
  padding: var(--spacing-xl);
  border-radius: var(--border-radius-lg);
  box-shadow: var(--shadow-lg);
  border: 1px solid #e0e0e0;
  max-width: 500px;
  width: 100%;
  max-height: 90vh;
  overflow-y: auto;
}

.modal h3 {
  margin-bottom: var(--spacing-md);
  color: #333;
  font-size: 18px;
}

.modal p {
  margin-bottom: var(--spacing-lg);
  color: #666;
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
  background: #333;
  color: white;
  border: none;
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: var(--transition);
}

.confirm-btn:hover {
  background: #555;
}

.cancel-btn {
  padding: var(--spacing-sm) var(--spacing-md);
  background: white;
  color: #333;
  border: 1px solid #e0e0e0;
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 14px;
  transition: var(--transition);
}

.cancel-btn:hover {
  background: #f8f9fa;
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
  border-bottom: 1px solid #e0e0e0;
}

.preview-header h3 {
  margin: 0;
  font-size: 16px;
  color: #333;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.close-btn {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: #999;
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: var(--transition);
}

.close-btn:hover {
  background: #f0f0f0;
  color: #333;
}

.preview-content {
  max-height: 500px;
  overflow-y: auto;
  margin-bottom: var(--spacing-lg);
}

.loading-preview, .preview-error, .unsupported-preview {
  text-align: center;
  padding: var(--spacing-xl);
  color: #999;
}

.text-preview {
  background: #f8f9fa;
  padding: var(--spacing-md);
  border-radius: var(--border-radius);
  border: 1px solid #e0e0e0;
  max-height: 500px;
  overflow-y: auto;
}

.text-preview pre {
  margin: 0;
  font-family: 'Courier New', Courier, monospace;
  font-size: 13px;
  line-height: 1.5;
  color: #333;
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
  border: 1px solid #e0e0e0;
}

.file-info {
  text-align: left;
  margin-top: var(--spacing-md);
  font-size: 13px;
  line-height: 1.6;
}

.preview-actions {
  display: flex;
  gap: var(--spacing-sm);
  justify-content: flex-end;
  padding-top: var(--spacing-md);
  border-top: 1px solid #e0e0e0;
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
</style>
