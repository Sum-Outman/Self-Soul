# 向量存储系统使用指南

## 📋 概述

向量存储系统是为Self-Soul-B项目设计的基础存储组件，支持多模态嵌入的存储、检索和管理。系统提供了统一的接口，支持多种后端存储，并集成了到现有数据处理器和记忆系统中。

### 核心特性
- **统一API**: 为所有向量存储操作提供一致的接口
- **多后端支持**: 支持ChromaDB和内存向量存储，可扩展更多后端
- **优雅降级**: 当主后端不可用时自动切换到备用方案
- **多模态支持**: 原生支持文本、图像、音频等模态的嵌入
- **语义检索**: 基于向量相似度的智能搜索能力
- **性能优化**: 内置缓存和批量操作优化

## 🏗️ 系统架构

```
向量存储系统架构:
┌─────────────────────────────────────────┐
│          应用层 (Application)            │
├─────────────────────────────────────────┤
│ 真实数据处理器     │      记忆系统       │
│ (TrueMultimodalDataProcessor) │ (MemorySystem) │
└───────────────┬─────────────────┬───────┘
                │                 │
                ▼                 ▼
┌─────────────────────────────────────────┐
│         向量存储管理器层 (Manager)       │
│           VectorStoreManager            │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         向量存储后端层 (Backends)        │
│  ┌─────────────┐  ┌─────────────┐      │
│  │ ChromaDB    │  │ Memory      │      │
│  │ VectorStore │  │ VectorStore │      │
│  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 基本使用

```python
# 导入向量存储管理器
from core.vector_store_manager import get_vector_store_manager

# 获取管理器实例
manager = get_vector_store_manager()

# 获取默认存储
store = manager.get_store()

# 添加嵌入向量
embedding = [0.1] * 768
metadata = {"modality": "text", "source": "test", "content": "测试文本"}
embedding_id = manager.add_embedding(
    embedding=embedding,
    metadata=metadata,
    document="这是测试文档内容",
    store_id="default"
)

# 搜索相似向量
query_embedding = [0.15] * 768
results = manager.search_similar(
    query_embedding=query_embedding,
    n_results=5,
    store_id="default"
)
```

### 2. 集成真实数据处理器

```python
from core.multimodal.true_data_processor import TrueMultimodalDataProcessor

# 创建启用向量存储的处理器
processor = TrueMultimodalDataProcessor(enable_vector_store=True)

# 处理并存储多模态输入
input_data = {
    "text": "这是一个测试文本",
    "image_data": b"\xff\xd8\xff\xe0\x00\x10JFIF..."  # 模拟图像数据
}

result = processor.process_and_store_multimodal_input(
    input_data=input_data,
    metadata={"source": "user_input", "timestamp": "2024-01-01T00:00:00"},
    store_id="default"
)

print(f"生成的嵌入: {list(result['embeddings'].keys())}")
print(f"存储结果: {result['storage_results']}")
```

### 3. 集成记忆系统

```python
from core.core_capabilities import MemorySystem

# 创建启用向量存储的记忆系统
memory_system = MemorySystem(max_items=100, enable_vector_store=True)

# 存储记忆
memory_id = memory_system.store(
    content="我喜欢学习人工智能",
    context={"topic": "AI", "feeling": "excited"},
    importance=0.8
)

# 传统关键词检索
keyword_memories = memory_system.retrieve("人工智能", limit=3)

# 向量相似度检索
similar_memories = memory_system.retrieve_similar("机器学习", limit=3)
```

## 🔧 配置选项

### 向量存储管理器配置

```python
from core.vector_store_manager import VectorStoreConfig

# 配置示例
config = VectorStoreConfig(
    backend_type="auto",  # 自动选择可用后端: "auto", "chroma", "memory"
    collection_name="multimodal_embeddings",
    persist_directory="./chroma_db",  # ChromaDB持久化目录
    embedding_dimension=768,
    
    # 性能配置
    max_collections=10,
    cache_size=1000,
    auto_persist=True,
    
    # 资源限制
    max_memory_mb=1024,
    max_disk_gb=10,
    max_embedding_count=1000000
)
```

### 配置文件示例

创建 `vector_store_config.json`:
```json
{
    "backend_type": "auto",
    "collection_name": "custom_embeddings",
    "embedding_dimension": 384,
    "max_collections": 5,
    "cache_size": 500,
    "chroma_config": {
        "anonymized_telemetry": false,
        "chroma_db_impl": "duckdb+parquet"
    }
}
```

使用配置文件初始化:
```python
manager = get_vector_store_manager(config_path="./vector_store_config.json")
```

## 📊 API参考

### VectorStoreManager 类

#### 主要方法

##### `get_store(store_id="default")`
获取向量存储实例。

**参数**:
- `store_id`: 存储ID，默认为"default"

**返回**: 向量存储实例

##### `add_embedding(embedding, metadata, document=None, store_id="default")`
添加单个嵌入向量。

**参数**:
- `embedding`: 嵌入向量列表
- `metadata`: 元数据字典
- `document`: 文档内容（可选）
- `store_id`: 存储ID

**返回**: 嵌入ID字符串

##### `add_embeddings(embeddings, metadatas, documents=None, ids=None, store_id="default")`
批量添加嵌入向量。

**参数**:
- `embeddings`: 嵌入向量列表的列表
- `metadatas`: 元数据字典列表
- `documents`: 文档内容列表（可选）
- `ids`: 自定义ID列表（可选）
- `store_id`: 存储ID

**返回**: ID列表

##### `search_similar(query_embedding, n_results=10, where=None, where_document=None, store_id="default")`
搜索相似的嵌入向量。

**参数**:
- `query_embedding`: 查询嵌入向量
- `n_results`: 返回结果数量
- `where`: 元数据过滤条件
- `where_document`: 文档内容过滤条件
- `store_id`: 存储ID

**返回**: 包含搜索结果（ids, distances, metadatas, documents, embeddings）的字典

##### `create_store(store_id, collection_name=None, persist_directory=None, embedding_dimension=None)`
创建新的向量存储。

**参数**:
- `store_id`: 新存储ID
- `collection_name`: 集合名称
- `persist_directory`: 持久化目录
- `embedding_dimension`: 嵌入维度

**返回**: 是否创建成功

##### `delete_store(store_id)`
删除向量存储。

**参数**:
- `store_id`: 要删除的存储ID

**返回**: 是否删除成功

##### `get_stats(store_id="default")`
获取存储统计信息。

**参数**:
- `store_id`: 存储ID

**返回**: 统计信息字典

### MemoryVectorStore 类

#### 性能特性
- **LRU缓存**: 自动缓存查询结果，缓存容量可配置
- **批量操作**: 支持批量添加和更新
- **内存优化**: 使用numpy数组存储嵌入向量，提高计算效率

#### 缓存配置
```python
# 创建带缓存的内存向量存储
store = MemoryVectorStore(
    collection_name="cached_collection",
    embedding_dimension=384
)

# 设置缓存容量（默认50）
store.query_cache.capacity = 100

# 获取缓存统计
cache_size = store.query_cache.size()
print(f"缓存大小: {cache_size}")
```

## 🔍 查询过滤条件

### 元数据过滤

```python
# 简单相等过滤
results = manager.search_similar(
    query_embedding=query_embedding,
    n_results=10,
    where={"modality": "text", "source": "user_input"}
)

# 复杂条件过滤
results = manager.search_similar(
    query_embedding=query_embedding,
    n_results=10,
    where={
        "modality": {"$in": ["text", "image"]},
        "timestamp": {"$gt": "2024-01-01"},
        "importance": {"$gte": 0.5}
    }
)
```

### 文档内容过滤

```python
# 文档包含特定内容
results = manager.search_similar(
    query_embedding=query_embedding,
    n_results=10,
    where_document={"$contains": "人工智能"}
)

# 不包含特定内容
results = manager.search_similar(
    query_embedding=query_embedding,
    n_results=10,
    where_document={"$not_contains": "测试"}
)
```

## 🧪 测试指南

### 单元测试

```python
# 运行所有向量存储测试
pytest tests/test_vector_store_integration.py -v

# 运行特定测试
pytest tests/test_vector_store_integration.py::test_chroma_vector_store -v
pytest tests/test_vector_store_integration.py::test_vector_store_manager_basic -v

# 运行集成测试
pytest tests/test_second_priority_integration.py -v
```

### 手动测试

```python
# 测试脚本示例
import sys
sys.path.insert(0, '.')
os.environ['ENVIRONMENT'] = 'development'

from core.vector_store_manager import get_vector_store_manager
from core.vector_store.memory_vector_store import test_memory_vector_store

# 测试内存向量存储
test_memory_vector_store()

# 测试向量存储管理器集成
manager = get_vector_store_manager()
store = manager.get_store()

# 执行基本操作测试
embedding = [0.1] * 768
metadata = {"test": "manual"}
embedding_id = manager.add_embedding(embedding, metadata)

results = manager.search_similar(embedding, n_results=1)
print(f"测试完成，找到{len(results['ids'])}个结果")
```

## 🚨 故障排除

### 常见问题

#### 1. ChromaDB导入失败
**症状**: `unable to infer type for attribute "chroma_server_nofile"`
**原因**: ChromaDB 1.5.2使用Pydantic v1，与Python 3.14+不兼容
**解决方案**: 系统自动降级到内存向量存储，功能保持完整

#### 2. 内存使用过高
**症状**: 内存消耗快速增长
**解决方案**:
- 配置内存限制: `VectorStoreConfig(max_memory_mb=1024)`
- 定期清理缓存: `store.query_cache.clear()`
- 分批处理大容量数据

#### 3. 查询性能慢
**解决方案**:
- 增加缓存容量: `store.query_cache.capacity = 100`
- 优化嵌入维度: 使用较小的维度（如384而不是768）
- 使用批量操作替代单个操作

### 调试日志

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 查看向量存储操作日志
logger = logging.getLogger("core.vector_store")
logger.setLevel(logging.DEBUG)
```

## 🔄 数据迁移

### 导出数据

```python
import json

# 导出所有嵌入向量
all_data = store.list_all(limit=None)

with open('embeddings_backup.json', 'w', encoding='utf-8') as f:
    json.dump({
        "ids": all_data["ids"],
        "embeddings": all_data["embeddings"],
        "metadatas": all_data["metadatas"],
        "documents": all_data["documents"]
    }, f, ensure_ascii=False, indent=2)
```

### 导入数据

```python
# 从备份恢复
with open('embeddings_backup.json', 'r', encoding='utf-8') as f:
    backup_data = json.load(f)

manager.add_embeddings(
    embeddings=backup_data["embeddings"],
    metadatas=backup_data["metadatas"],
    documents=backup_data["documents"],
    ids=backup_data["ids"]
)
```

## 📈 性能监控

### 统计信息

```python
# 获取系统统计
stats = manager.get_stats()

print(f"总嵌入数: {stats['total_embeddings']}")
print(f"查询次数: {stats['query_count']}")
print(f"插入次数: {stats['insert_count']}")
print(f"错误次数: {stats['error_count']}")
print(f"内存使用: {stats['memory_usage_mb']:.2f} MB")
print(f"磁盘使用: {stats['disk_usage_mb']:.2f} MB")
```

### 监控集成

```python
# 集成到系统监控
from core.system_monitor import SystemMonitor

monitor = SystemMonitor()

# 添加向量存储指标
@monitor.add_metric("vector_store")
def get_vector_store_metrics():
    stats = manager.get_stats()
    return {
        "embeddings_count": stats["total_embeddings"],
        "query_rate": stats["query_count"],
        "error_rate": stats["error_count"],
        "memory_usage_mb": stats["memory_usage_mb"]
    }
```

## 🔮 扩展开发

### 添加新后端

1. **创建后端类**:
```python
from typing import Dict, List, Any, Optional
import numpy as np

class CustomVectorStore:
    def __init__(self, collection_name: str, **kwargs):
        self.collection_name = collection_name
        # 初始化代码
    
    def add_embeddings(self, embeddings, metadatas, documents=None, ids=None):
        # 实现添加嵌入
        pass
    
    def search_similar(self, query_embedding, n_results=10, where=None, where_document=None):
        # 实现相似度搜索
        pass
    
    # 其他必需方法...
```

2. **集成到管理器**:
    - 修改 `core/vector_store/__init__.py` 导入新后端
    - 更新 `VectorStoreManager._initialize_default_store()` 和 `create_store()` 方法支持新后端类型

### 性能优化建议

1. **批量处理**: 尽可能使用 `add_embeddings()` 而非多次调用 `add_embedding()`
2. **缓存策略**: 根据访问模式调整缓存容量
3. **索引优化**: 对于大型数据集，考虑添加额外的索引结构
4. **异步操作**: 对耗时操作实现异步接口

## 📚 相关资源

- [多模态系统技术评估与路线图报告](./多模态系统技术评估与路线图报告.md)
- [测试框架文档](./tests/README.md)
- [部署指南](./DEPLOYMENT_GUIDE.md)
- [API集成指南](./API_INTEGRATION_GUIDE.md)

## 📝 更新记录

### 2026-03-06
- **向量存储系统完成集成**: 实现向量存储管理器、真实数据处理器和记忆系统的全面集成
- **解决ChromaDB兼容性问题**: 实现优雅降级机制，自动切换到内存向量存储
- **性能优化**: 为内存向量存储添加LRU缓存和批量操作优化
- **文档完善**: 创建完整的使用指南和API参考

### 2026-03-05  
- **向量存储管理器实现**: 创建统一的管理接口
- **系统集成**: 与真实数据处理器和记忆系统集成
- **测试框架**: 创建完整的集成测试

---

**维护者**: Self-Soul-B开发团队  
**版本**: 1.0.0  
**最后更新**: 2026-03-06