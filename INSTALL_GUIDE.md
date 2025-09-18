# Self Soul 安装指南

## 系统要求
- Windows 10/11 或 Linux/macOS
- Python 3.9+
- Node.js 16+
- 8GB以上内存
- 10GB可用磁盘空间

## 安装步骤

### 1. 克隆代码库
```bash
git clone https://github.com/your-username/agi-system.git
cd agi-system
```

### 2. 设置Python虚拟环境
```bash
python -m venv .venv
```

### 3. 激活虚拟环境
- Windows:
```bash
.venv\Scripts\activate
```
- Linux/macOS:
```bash
source .venv/bin/activate
```

### 4. 安装Python依赖
```bash
pip install -r requirements.txt
```

### 5. 安装前端依赖
```bash
cd app
npm install
```

### 6. 配置环境变量
创建 `.env` 文件并添加以下内容：
```
API_KEY=your-api-key-here
MODEL_PATH=./models
DATA_PATH=./data
```

### 7. 启动系统
```bash
# 启动后端服务
python core/main.py

# 在新终端中启动前端应用
cd app && npm run dev
```

### 8. 访问系统
打开浏览器访问：http://localhost:5173

## 系统功能
1. **多语言界面**：支持5种语言切换
2. **模型管理**：配置和监控11个专业模型
3. **实时仪表盘**：监控系统性能
4. **交互式帮助**：提供完整的使用文档
5. **多模态交互**：支持文本、语音和视觉输入

## 常见问题解决

### 虚拟环境问题
如果遇到虚拟环境问题，请尝试：
```bash
python -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
python -m venv .venv
```

### 依赖安装问题
如果pip安装失败，请尝试：
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 前端启动问题
如果npm run dev失败，请尝试：
```bash
cd app
rm -rf node_modules
npm install
npm run dev
```

## 系统架构
```
Self Soul /
├── core/                     # 核心系统
│   ├── models/               # 11个核心模型实现
│   │   ├── audio/            # 音频处理模型
│   │   ├── computer/         # 计算机控制模型
│   │   ├── knowledge/        # 知识库专家模型
│   │   ├── language/         # 大语言模型
│   │   ├── manager/          # 管理模型
│   │   ├── medical/          # 医疗专业模型
│   │   ├── motion/           # 运动控制模型
│   │   ├── planning/         # 规划模型
│   │   ├── prediction/       # 预测模型
│   │   ├── programming/      # 编程模型
│   │   ├── sensor/           # 传感器感知模型
│   │   ├── spatial/          # 空间定位模型
│   │   ├── video/            # 视频处理模型
│   │   └── vision/           # 视觉处理模型
│   ├── training/             # 训练系统
│   │   ├── audio_training.py
│   │   ├── computer_training.py
│   │   ├── joint_training.py
│   │   ├── knowledge_training.py
│   │   ├── language_training.py
│   │   ├── manager_training.py
│   │   ├── motion_training.py
│   │   ├── programming_training.py
│   │   ├── sensor_training.py
│   │   ├── spatial_training.py
│   │   ├── video_training.py
│   │   └── vision_training.py
│   ├── collaboration/        # 模型协作
│   ├── data/                 # 数据处理
│   ├── fusion/               # 多模态融合
│   ├── knowledge/            # 知识增强
│   ├── optimization/         # 模型优化
│   ├── realtime/             # 实时处理
│   ├── agi_coordinator.py    # AGI协调器
│   ├── api_config_manager.py # API配置管理
│   ├── data_processor.py     # 数据处理器
│   ├── emotion_awareness.py  # 情感感知
│   ├── error_handling.py     # 错误处理
│   ├── main.py               # 主入口
│   ├── model_registry.py     # 模型注册表
│   ├── monitoring.py         # 系统监控
│   ├── self_learning.py      # 自主学习
│   └── training_manager.py   # 训练管理器
├── app/                      # 前端Vue应用
│   ├── src/
│   │   ├── components/       # 组件
│   │   ├── locales/          # 多语言文件
│   │   ├── models/           # 前端模型
│   │   ├── plugins/          # 插件
│   │   ├── router/           # 路由
│   │   ├── views/            # 视图页面
│   │   ├── App.vue           # 主应用
│   │   ├── i18n.js           # 国际化
│   │   └── main.js           # 入口
│   ├── public/               # 静态资源
│   ├── package.json          # 依赖配置
│   └── vite.config.js        # Vite配置
├── config/                   # 配置文件
│   └── languages/            # 多语言配置
├── data/                     # 数据文件
│   └── knowledge/            # 知识库数据
├── docs/                     # 文档
├── logs/                     # 系统日志
├── tests/                    # 测试套件
├── tools/                    # 工具脚本
└── requirements.txt          # Python依赖
```

## 技术支持
如有任何问题，请联系：silencecrowtom@qq.com