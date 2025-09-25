// 增强的模拟API实现 - 兼容axios响应格式
// 启用真实后端连接
const NO_BACKEND_MODE = false;

// 创建增强的API模拟对象
const mockApi = {
  // 健康检查API
  health: {
    get: () => Promise.resolve({
      data: {
        status: 'ok',
        message: 'Backend service is running normally',
        version: '1.0.0',
        timestamp: new Date().toISOString()
      },
      status: 200,
      statusText: 'OK'
    })
  },
  
  // 系统统计API
  system: {
    stats: () => Promise.resolve({
      data: {
        status: 'success',
        stats: {
          active_models: 3,
          total_models: 8,
          cpu_usage: 25.3,
          memory_usage: 42.7,
          disk_usage: 68.9,
          uptime: '02:45:18'
        }
      },
      status: 200,
      statusText: 'OK'
    }),
    restart: () => Promise.resolve({
      data: {
        status: 'success',
        message: 'System restart initiated'
      },
      status: 200,
      statusText: 'OK'
    })
  },
  
  // 模型API
  models: {
    get: () => Promise.resolve({
      data: {
        status: 'success',
        models: [
          {id: '8001', name: 'Management Model', type: 'manager', isActive: true, isPrimary: true, port: 8001, performance: 95, last_active: 'Just now'},
          {id: '8002', name: 'Language Model', type: 'language', isActive: true, isPrimary: false, port: 8002, performance: 92, last_active: '1 minute ago'},
          {id: '8003', name: 'Knowledge Model', type: 'knowledge', isActive: true, isPrimary: false, port: 8003, performance: 89, last_active: '2 minutes ago'},
          {id: '8004', name: 'Vision Model', type: 'vision', isActive: false, isPrimary: false, port: 8004, performance: 90, last_active: '5 minutes ago'},
          {id: '8005', name: 'Audio Model', type: 'audio', isActive: false, isPrimary: false, port: 8005, performance: 88, last_active: '10 minutes ago'},
          {id: '8006', name: 'Autonomous Model', type: 'autonomous', isActive: false, isPrimary: false, port: 8006, performance: 0, last_active: 'Never'},
          {id: '8007', name: 'Programming Model', type: 'programming', isActive: true, isPrimary: false, port: 8007, performance: 91, last_active: '15 minutes ago'},
          {id: '8008', name: 'Planning Model', type: 'planning', isActive: false, isPrimary: false, port: 8008, performance: 0, last_active: 'Never'},
          {id: '8009', name: 'Emotion Model', type: 'emotion', isActive: false, isPrimary: false, port: 8009, performance: 0, last_active: 'Never'},
          {id: '8010', name: 'Spatial Model', type: 'spatial', isActive: true, isPrimary: false, port: 8010, performance: 82, last_active: '20 minutes ago'},
          {id: '8011', name: 'Computer Vision Model', type: 'vision', isActive: true, isPrimary: false, port: 8011, performance: 93, last_active: '30 minutes ago'}
        ]
      },
      status: 200,
      statusText: 'OK'
    }),
    trainingStatus: () => Promise.resolve({
      data: {
        status: 'success',
        training_status: {
          active_jobs: 0,
          queued_jobs: 1,
          completed_jobs: 5
        }
      },
      status: 200,
      statusText: 'OK'
    }),
    fromScratchStatus: () => Promise.resolve({
      data: {
        training_mode: 'From Scratch',
        vocab_size: 10000,
        epochs: 50,
        last_activity: new Date().toLocaleString(),
        confidence: 75
      },
      status: 200,
      statusText: 'OK'
    })
  },
  
  // 数据集API
  datasets: {
    get: () => Promise.resolve({
      data: {
        status: 'success',
        datasets: [
          {id: '1', name: 'Text Classification Dataset', type: 'text', size: '100MB', samples: 10000},
          {id: '2', name: 'Image Recognition Dataset', type: 'image', size: '2.5GB', samples: 5000},
          {id: '3', name: 'Audio Transcription Dataset', type: 'audio', size: '1.2GB', samples: 2000}
        ]
      },
      status: 200,
      statusText: 'OK'
    })
  },
  
  // 训练API
  training: {
    start: () => Promise.resolve({
      data: {
        status: 'success',
        message: 'Training started successfully',
        job_id: 'train_' + Date.now()
      },
      status: 200,
      statusText: 'OK'
    }),
    status: (jobId) => Promise.resolve({
      data: {
        status: 'completed',
        job_id: jobId,
        progress: 100,
        metrics: {
          accuracy: 0.85,
          loss: 0.32
        }
      },
      status: 200,
      statusText: 'OK'
    }),
    stop: () => Promise.resolve({
      data: {
        status: 'success',
        message: 'Training stopped successfully'
      },
      status: 200,
      statusText: 'OK'
    }),
    history: () => Promise.resolve({
      data: {
        status: 'success',
        history: [
          {id: 'train_123', model_id: '2', status: 'completed', accuracy: 0.85, loss: 0.32, duration: '01:45:30'},
          {id: 'train_456', model_id: '3', status: 'completed', accuracy: 0.92, loss: 0.21, duration: '02:15:45'}
        ]
      },
      status: 200,
      statusText: 'OK'
    })
  },
  
  // 知识API
  knowledge: {
    files: () => Promise.resolve({
      data: {
        status: 'success',
        files: [
          {id: '1', name: 'system_architecture.pdf', type: 'pdf', size: '2.5 MB', last_modified: '2024-01-15T10:30:00'},
          {id: '2', name: 'model_documentation.md', type: 'md', size: '1.2 MB', last_modified: '2024-01-14T15:45:00'},
          {id: '3', name: 'training_dataset.csv', type: 'csv', size: '15.8 MB', last_modified: '2024-01-13T09:12:00'},
          {id: '4', name: 'knowledge_graph.json', type: 'json', size: '3.7 MB', last_modified: '2024-01-12T14:20:00'},
          {id: '5', name: 'user_manual.docx', type: 'docx', size: '4.1 MB', last_modified: '2024-01-11T11:05:00'}
        ]
      },
      status: 200,
      statusText: 'OK'
    }),
    filePreview: (fileId) => Promise.resolve({
      data: {
        status: 'success',
        file_id: fileId,
        content_preview: 'This is a preview of the file content...',
        content_type: 'text/plain'
      },
      status: 200,
      statusText: 'OK'
    }),
    search: (query, domain) => Promise.resolve({
      data: {
        status: 'success',
        results: [
          {id: '1', title: 'Machine Learning Basics', domain: domain || 'general', relevance: 0.92},
          {id: '2', title: 'Neural Network Architectures', domain: domain || 'general', relevance: 0.85}
        ]
      },
      status: 200,
      statusText: 'OK'
    }),
    stats: () => Promise.resolve({
      data: {
        status: 'success',
        stats: {
          total_domains: 5,
          total_items: 125,
          total_size: '15.8 MB'
        }
      },
      status: 200,
      statusText: 'OK'
    }),
    autoLearning: {
      start: () => Promise.resolve({
        data: {
          status: 'success',
          message: 'Auto-learning started successfully',
          session_id: 'auto_learn_' + Date.now()
        },
        status: 200,
        statusText: 'OK'
      }),
      stop: () => Promise.resolve({
        data: {
          status: 'success',
          message: 'Auto-learning stopped successfully'
        },
        status: 200,
        statusText: 'OK'
      }),
      progress: () => Promise.resolve({
        data: {
          status: 'in_progress',
          progress: 65,
          current_domain: 'machine_learning',
          estimated_time: '00:45:00'
        },
        status: 200,
        statusText: 'OK'
      })
    }
  },
  
  // 处理API
  process: {
    image: (data) => Promise.resolve({
      data: {
        status: 'success',
        message: 'Image processed successfully',
        result: {
          analysis: 'This is a sample image analysis result',
          confidence: 0.95
        }
      },
      status: 200,
      statusText: 'OK'
    }),
    video: (data) => Promise.resolve({
      data: {
        status: 'success',
        message: 'Video processed successfully',
        result: {
          analysis: 'This is a sample video analysis result',
          objects_detected: ['person', 'car', 'bicycle']
        }
      },
      status: 200,
      statusText: 'OK'
    }),
    audio: (data) => Promise.resolve({
      data: {
        status: 'success',
        message: 'Audio processed successfully',
        result: {
          transcription: 'This is a sample audio transcription',
          confidence: 0.92
        }
      },
      status: 200,
      statusText: 'OK'
    })
  },
  
  // 聊天API
  chat: (message) => {
    let responseText = 'This is a sample response to your message: ' + message.content;
    
    // 为常见查询提供自定义响应
    if (message.content.toLowerCase().includes('hello') || message.content.toLowerCase().includes('hi')) {
      responseText = 'Hello! I am your assistant. How can I help you today?';
    } else if (message.content.toLowerCase().includes('how are you')) {
      responseText = 'I am doing well, thank you for asking! I am ready to assist you with any questions or tasks you have.';
    } else if (message.content.toLowerCase().includes('what can you do')) {
      responseText = 'I can help you with a variety of tasks, including answering questions, providing information, assisting with programming, and much more. Feel free to ask me anything!';
    }
    
    return Promise.resolve({
      data: {
        status: 'success',
        response: responseText,
        model: 'language-model-1',
        confidence: 0.87
      },
      status: 200,
      statusText: 'OK'
    });
  },
  
  // 管理模型特定聊天API - 增强版
  managementChat: (message) => {
    let responseText = `Hello! I am the Management Model. You asked: "${message.content}". I manage all other AI models in the system.`;
    const lowerMessage = message.content.toLowerCase();
    
    // 增强的响应逻辑
    if (lowerMessage.includes('hello') || lowerMessage.includes('hi') || lowerMessage.includes('hey')) {
      responseText = 'Hello! I am your Management Model. I coordinate all AI models in the Self Soul system. How can I assist you today?';
    } else if (lowerMessage.includes('what models do you manage')) {
      responseText = 'I manage 11 different AI models in total: Management Model (myself), Language Model, Knowledge Model, Vision Model, Audio Model, Autonomous Model, Programming Model, Planning Model, Emotion Model, Spatial Model, and Computer Vision Model. I coordinate their activities to provide you with seamless assistance.';
    } else if (lowerMessage.includes('how are the models performing')) {
      responseText = 'Currently, 6 out of 11 models are active. The Management Model (me) is performing at 95%, Language Model at 92%, Knowledge Model at 89%, Vision Model at 90%, Audio Model at 88%, and Programming Model at 91%. The other 5 models are in standby mode but can be activated as needed.';
    } else if (lowerMessage.includes('can you start all models')) {
      responseText = 'Yes, I can start all models. Would you like me to initiate that process now? This will activate all 11 models simultaneously and may consume more system resources.';
    } else if (lowerMessage.includes('system status')) {
      responseText = 'The Self Soul system is currently running smoothly with 6 active models. CPU usage is at 25.3%, memory usage at 42.7%, and disk usage at 68.9%. The system has been running for 2 hours, 45 minutes, and 18 seconds.';
    } else if (lowerMessage.includes('what can you do')) {
      responseText = 'As the Management Model, I can: coordinate all other AI models, monitor system performance, start/stop models, provide information about the system, answer questions about AI capabilities, and help you with any task by leveraging the specialized models under my management.';
    } else if (lowerMessage.includes('training') || lowerMessage.includes('learn')) {
      responseText = 'The system supports various training modes including supervised training, unsupervised learning, and reinforcement learning. You can train individual models or coordinate joint training sessions. Currently, there is 1 queued training job and 5 completed jobs.';
    } else if (lowerMessage.includes('help') || lowerMessage.includes('guide')) {
      responseText = 'I can help you navigate and use the Self Soul system. You can ask me about specific models, system status, training processes, or how to use different features. What would you like to know more about?';
    } else if (lowerMessage.includes('thank') || lowerMessage.includes('thanks')) {
      responseText = 'You\'re welcome! I\'m here to help you make the most of the Self Soul system. Feel free to ask me anything at any time.';
    }
    
    return Promise.resolve({
      data: {
        status: 'success',
        response: responseText,
        confidence: 0.97,
        response_type: 'management',
        model_id: '8001',
        model_name: 'Management Model',
        timestamp: new Date().toISOString(),
        session_id: message.session_id || 'new_session_' + Date.now()
      },
      status: 200,
      statusText: 'OK'
    });
  },
  
  // 通用GET请求处理
  get: (url) => {
    console.log('Mock GET request to:', url);
    
    // 健康检查
    if (url === '/health') {
      return mockApi.health.get();
    }
    // 系统统计
    else if (url === '/api/system/stats') {
      return mockApi.system.stats();
    }
    // 模型列表
    else if (url === '/api/models') {
      return mockApi.models.get();
    }
    // 模型训练状态
    else if (url === '/api/models/training/status') {
      return mockApi.models.trainingStatus();
    }
    // 从头训练模型状态
    else if (url === '/api/models/from_scratch/status') {
      return mockApi.models.fromScratchStatus();
    }
    // 数据集列表
    else if (url === '/api/datasets') {
      return mockApi.datasets.get();
    }
    // 知识文件列表
    else if (url === '/api/knowledge/files') {
      return mockApi.knowledge.files();
    }
    // 知识文件预览
    else if (url.includes('/api/knowledge/files/') && url.includes('/preview')) {
      const fileId = url.split('/')[4];
      return mockApi.knowledge.filePreview(fileId);
    }
    // 训练历史
    else if (url === '/api/training/history') {
      return mockApi.training.history();
    }
    // 训练状态
    else if (url.includes('/api/training/status/')) {
      const jobId = url.split('/')[4];
      return mockApi.training.status(jobId);
    }
    // 自主学习进度
    else if (url === '/api/knowledge/auto-learning/progress') {
      return mockApi.knowledge.autoLearning.progress();
    }
    // 知识统计
    else if (url === '/api/knowledge/stats') {
      return mockApi.knowledge.stats();
    }
    // 知识搜索
    else if (url.includes('/api/knowledge/search')) {
      const urlParams = new URLSearchParams(url.split('?')[1]);
      const query = urlParams.get('query') || '';
      const domain = urlParams.get('domain');
      return mockApi.knowledge.search(query, domain);
    }
    // 默认响应
    else {
      return Promise.resolve({
        data: {
          status: 'success',
          message: 'Mock data response for: ' + url,
          data: null
        },
        status: 200,
        statusText: 'OK'
      });
    }
  },
  
  // 通用POST请求处理
  post: (url, data) => {
    console.log('Mock POST request to:', url, 'with data:', data);
    
    // 训练开始
    if (url === '/api/training/start') {
      return mockApi.training.start();
    }
    // 训练停止
    else if (url === '/api/training/stop') {
      return mockApi.training.stop();
    }
    // 管理模型聊天
    else if (url === '/api/models/8001/chat') {
      return mockApi.managementChat(data);
    }
    // 数据集上传
    else if (url === '/api/datasets/upload') {
      return Promise.resolve({
        data: {
          status: 'success',
          message: 'Dataset uploaded successfully',
          dataset_id: 'dataset_' + Date.now()
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 添加模型
    else if (url === '/api/models') {
      return Promise.resolve({
        data: {
          status: 'success',
          message: 'Model added successfully',
          model_id: 'model_' + Date.now()
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 模型启动
    else if (url.includes('/api/models/') && url.includes('/start')) {
      const modelId = url.split('/')[3];
      return Promise.resolve({
        data: {
          status: 'success',
          message: `Model ${modelId} started successfully`
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 模型停止
    else if (url.includes('/api/models/') && url.includes('/stop')) {
      const modelId = url.split('/')[3];
      return Promise.resolve({
        data: {
          status: 'success',
          message: `Model ${modelId} stopped successfully`
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 模型重启
    else if (url.includes('/api/models/') && url.includes('/restart')) {
      const modelId = url.split('/')[3];
      return Promise.resolve({
        data: {
          status: 'success',
          message: `Model ${modelId} restarted successfully`
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 所有模型启动
    else if (url === '/api/models/start-all') {
      return Promise.resolve({
        data: {
          status: 'success',
          message: 'All models started successfully'
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 所有模型停止
    else if (url === '/api/models/stop-all') {
      return Promise.resolve({
        data: {
          status: 'success',
          message: 'All models stopped successfully'
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 所有模型重启
    else if (url === '/api/models/restart-all') {
      return Promise.resolve({
        data: {
          status: 'success',
          message: 'All models restarted successfully'
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 系统重启
    else if (url === '/api/system/restart') {
      return mockApi.system.restart();
    }
    // 测试连接
    else if (url === '/api/models/test-connection') {
      return Promise.resolve({
        data: {
          status: 'success',
          connected: true,
          latency: 150,
          message: 'Connection test successful'
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 模型训练
    else if (url.includes('/api/models/') && url.includes('/train')) {
      const modelId = url.split('/')[3];
      return Promise.resolve({
        data: {
          status: 'success',
          message: `Training started for model ${modelId}`,
          job_id: 'train_' + modelId + '_' + Date.now()
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 停止模型训练
    else if (url.includes('/api/models/') && url.includes('/train/stop')) {
      const modelId = url.split('/')[3];
      return Promise.resolve({
        data: {
          status: 'success',
          message: `Training stopped for model ${modelId}`
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 获取模型训练状态
    else if (url.includes('/api/models/') && url.includes('/train/status')) {
      const modelId = url.split('/')[3];
      return Promise.resolve({
        data: {
          status: 'completed',
          model_id: modelId,
          progress: 100,
          metrics: {
            accuracy: 0.87,
            loss: 0.29
          }
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 自主学习开始
    else if (url === '/api/knowledge/auto-learning/start') {
      return mockApi.knowledge.autoLearning.start();
    }
    // 自主学习停止
    else if (url === '/api/knowledge/auto-learning/stop') {
      return mockApi.knowledge.autoLearning.stop();
    }
    // 知识上传
    else if (url === '/api/knowledge/upload') {
      return Promise.resolve({
        data: {
          status: 'success',
          message: 'Knowledge file uploaded successfully',
          file_id: 'knowledge_' + Date.now()
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 处理图像
    else if (url === '/api/process/image') {
      return mockApi.process.image(data);
    }
    // 处理视频
    else if (url === '/api/process/video') {
      return mockApi.process.video(data);
    }
    // 处理音频
    else if (url === '/api/process/audio') {
      return mockApi.process.audio(data);
    }
    // 聊天
    else if (url === '/api/chat') {
      return mockApi.chat(data);
    }
    // 默认响应
    else {
      return Promise.resolve({
        data: {
          status: 'success',
          message: 'Mock data created successfully for: ' + url,
          data: null
        },
        status: 200,
        statusText: 'OK'
      });
    }
  },
  
  // 通用PUT请求处理
  put: (url, data) => {
    console.log('Mock PUT request to:', url, 'with data:', data);
    
    // 模型激活状态
    if (url.includes('/api/models/') && url.includes('/activation')) {
      const modelId = url.split('/')[3];
      const newState = data.isActive ? 'activated' : 'deactivated';
      return Promise.resolve({
        data: {
          status: 'success',
          message: `Model ${modelId} ${newState} successfully`
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 设置主要模型
    else if (url.includes('/api/models/') && url.includes('/primary')) {
      const modelId = url.split('/')[3];
      return Promise.resolve({
        data: {
          status: 'success',
          message: `Model ${modelId} set as primary successfully`
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 更新模型
    else if (url === '/api/models') {
      return Promise.resolve({
        data: {
          status: 'success',
          message: 'Models updated successfully'
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 默认响应
    else {
      return Promise.resolve({
        data: {
          status: 'success',
          message: 'Mock data updated successfully for: ' + url,
          data: null
        },
        status: 200,
        statusText: 'OK'
      });
    }
  },
  
  // 通用DELETE请求处理
  delete: (url) => {
    console.log('Mock DELETE request to:', url);
    
    // 删除模型
    if (url.includes('/api/models/') && !url.includes('/activation') && !url.includes('/primary')) {
      const modelId = url.split('/')[3];
      return Promise.resolve({
        data: {
          status: 'success',
          message: `Model ${modelId} deleted successfully`
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 删除知识文件
    else if (url.includes('/api/knowledge/files/')) {
      const fileId = url.split('/')[4];
      return Promise.resolve({
        data: {
          status: 'success',
          message: `Knowledge file ${fileId} deleted successfully`
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 默认响应
    else {
      return Promise.resolve({
        data: {
          status: 'success',
          message: 'Mock data deleted successfully for: ' + url,
          data: null
        },
        status: 200,
        statusText: 'OK'
      });
    }
  },
  
  // 通用PATCH请求处理
  patch: (url, data) => {
    console.log('Mock PATCH request to:', url, 'with data:', data);
    
    // 更新模型部分信息
    if (url.includes('/api/models/')) {
      const modelId = url.split('/')[3];
      return Promise.resolve({
        data: {
          status: 'success',
          message: `Model ${modelId} updated successfully`
        },
        status: 200,
        statusText: 'OK'
      });
    }
    // 默认响应
    else {
      return Promise.resolve({
        data: {
          status: 'success',
          message: 'Mock data patched successfully for: ' + url,
          data: null
        },
        status: 200,
        statusText: 'OK'
      });
    }
  }
};

// 创建最终的API对象
const api = {
  get: (url) => mockApi.get(url),
  post: (url, data) => mockApi.post(url, data),
  put: (url, data) => mockApi.put(url, data),
  delete: (url) => mockApi.delete(url)
};

export default api;