#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量存储管理器

提供统一的向量存储管理接口，支持多模态嵌入的存储、检索和管理。
基于现有的ChromaVectorStore和MultimodalVectorStore实现。

设计原则：
1. 统一接口：为所有向量存储操作提供一致的API
2. 多后端支持：支持ChromaDB等向量数据库
3. 配置驱动：通过配置文件管理向量存储设置
4. 资源感知：监控和管理向量存储资源使用
5. 集成友好：易于集成到现有系统中
"""

import logging
import threading
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import os
import json

logger = logging.getLogger(__name__)

# 尝试导入向量存储后端
ChromaVectorStore = None
MultimodalVectorStore = None
MemoryVectorStore = None

try:
    from core.vector_store.chroma_vector_store import ChromaVectorStore
    from core.vector_store.multimodal_vector_store import MultimodalVectorStore
    logger.debug("ChromaDB向量存储导入成功")
except Exception as e:
    logger.warning(f"ChromaDB向量存储导入失败: {e}")
    ChromaVectorStore = None
    MultimodalVectorStore = None

# 尝试导入内存向量存储
try:
    from core.vector_store.memory_vector_store import MemoryVectorStore
    logger.debug("内存向量存储导入成功")
except Exception as e:
    logger.warning(f"内存向量存储导入失败: {e}")
    MemoryVectorStore = None


@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    
    # 存储后端类型
    backend_type: str = "auto"  # auto, chroma, memory, faiss, pinecone, qdrant, etc.
    
    # 通用配置
    collection_name: str = "multimodal_embeddings"
    persist_directory: str = "./chroma_db"
    embedding_dimension: int = 768
    
    # 性能配置
    max_collections: int = 10
    cache_size: int = 1000
    auto_persist: bool = True
    persist_interval_seconds: int = 300  # 5分钟
    
    # 后端特定配置
    chroma_config: Dict[str, Any] = field(default_factory=lambda: {
        "anonymized_telemetry": False,
        "chroma_db_impl": "duckdb+parquet"
    })
    
    # 多模态配置
    multimodal_config: Dict[str, Any] = field(default_factory=lambda: {
        "enable_text_embeddings": True,
        "enable_image_embeddings": True,
        "enable_audio_embeddings": True,
        "enable_video_embeddings": False,
        "text_model": "all-MiniLM-L6-v2",  # 384维
        "image_model": "resnet50"
    })
    
    # 资源限制
    max_memory_mb: int = 1024  # 最大内存使用（MB）
    max_disk_gb: int = 10  # 最大磁盘使用（GB）
    max_embedding_count: int = 1000000  # 最大嵌入数量


@dataclass
class VectorStoreStats:
    """向量存储统计信息"""
    
    # 基本信息
    store_id: str
    collection_name: str
    backend_type: str
    
    # 存储统计
    total_embeddings: int = 0
    total_collections: int = 1
    storage_size_bytes: int = 0
    
    # 性能统计
    query_count: int = 0
    insert_count: int = 0
    update_count: int = 0
    delete_count: int = 0
    
    # 错误统计
    error_count: int = 0
    last_error: Optional[str] = None
    
    # 时间统计
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_query_at: Optional[str] = None
    last_insert_at: Optional[str] = None
    
    # 资源使用
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0


class VectorStoreManager:
    """向量存储管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config_path: Optional[str] = None):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化向量存储管理器"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._lock = threading.RLock()
        self._initialized = True
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 存储实例
        self.stores: Dict[str, Any] = {}
        self.store_stats: Dict[str, VectorStoreStats] = {}
        
        # 默认存储
        self.default_store_id = "default"
        
        # 初始化默认存储
        self._initialize_default_store()
        
        logger.info(f"向量存储管理器初始化完成，后端类型: {self.config.backend_type}")
    
    def _load_config(self, config_path: Optional[str]) -> VectorStoreConfig:
        """加载配置"""
        # 默认配置
        config = VectorStoreConfig()
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    
                # 更新配置
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                        
                logger.info(f"从 {config_path} 加载配置")
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}, 使用默认配置")
        else:
            logger.info("使用默认向量存储配置")
        
        return config
    
    def _initialize_default_store(self):
        """初始化默认向量存储"""
        with self._lock:
            try:
                # 确定要使用的后端类型
                backend_to_use = self.config.backend_type
                
                if backend_to_use == "auto":
                    # 自动选择可用的后端
                    if ChromaVectorStore is not None and MultimodalVectorStore is not None:
                        backend_to_use = "chroma"
                        logger.info("自动选择ChromaDB作为向量存储后端")
                    elif MemoryVectorStore is not None:
                        backend_to_use = "memory"
                        logger.info("自动选择内存作为向量存储后端")
                    else:
                        raise RuntimeError("没有可用的向量存储后端")
                
                # 根据后端类型创建存储实例
                if backend_to_use == "chroma" and MultimodalVectorStore is not None:
                    # 使用多模态向量存储
                    store = MultimodalVectorStore(
                        collection_name=self.config.collection_name,
                        persist_directory=self.config.persist_directory,
                        embedding_dimension=self.config.embedding_dimension
                    )
                elif backend_to_use == "memory" and MemoryVectorStore is not None:
                    # 使用内存向量存储
                    store = MemoryVectorStore(
                        collection_name=self.config.collection_name,
                        embedding_dimension=self.config.embedding_dimension
                    )
                else:
                    raise ValueError(f"不支持的向量存储后端或后端不可用: {backend_to_use}")
                
                # 存储实例
                self.stores[self.default_store_id] = store
                
                # 初始化统计
                self.store_stats[self.default_store_id] = VectorStoreStats(
                    store_id=self.default_store_id,
                    collection_name=self.config.collection_name,
                    backend_type=backend_to_use
                )
                
                logger.info(f"初始化默认向量存储: {self.default_store_id}, 后端: {backend_to_use}")
                
            except Exception as e:
                logger.error(f"初始化默认向量存储失败: {e}")
                raise
    
    def get_store(self, store_id: str = "default") -> Any:
        """
        获取向量存储实例
        
        Args:
            store_id: 存储ID
            
        Returns:
            向量存储实例
        """
        with self._lock:
            if store_id not in self.stores:
                raise ValueError(f"向量存储不存在: {store_id}")
            
            return self.stores[store_id]
    
    def create_store(self, 
                     store_id: str,
                     collection_name: Optional[str] = None,
                     persist_directory: Optional[str] = None,
                     embedding_dimension: Optional[int] = None) -> bool:
        """
        创建新的向量存储
        
        Args:
            store_id: 存储ID
            collection_name: 集合名称
            persist_directory: 持久化目录
            embedding_dimension: 嵌入维度
            
        Returns:
            是否创建成功
        """
        with self._lock:
            if store_id in self.stores:
                logger.warning(f"向量存储已存在: {store_id}")
                return False
            
            try:
                # 使用配置值或传入的参数
                coll_name = collection_name or f"{self.config.collection_name}_{store_id}"
                persist_dir = persist_directory or os.path.join(
                    self.config.persist_directory, store_id
                )
                embed_dim = embedding_dimension or self.config.embedding_dimension
                
                # 确定要使用的后端类型
                backend_to_use = self.config.backend_type
                
                if backend_to_use == "auto":
                    # 自动选择可用的后端
                    if ChromaVectorStore is not None and MultimodalVectorStore is not None:
                        backend_to_use = "chroma"
                    elif MemoryVectorStore is not None:
                        backend_to_use = "memory"
                    else:
                        raise RuntimeError("没有可用的向量存储后端")
                
                # 根据后端类型创建存储实例
                if backend_to_use == "chroma" and MultimodalVectorStore is not None:
                    store = MultimodalVectorStore(
                        collection_name=coll_name,
                        persist_directory=persist_dir,
                        embedding_dimension=embed_dim
                    )
                elif backend_to_use == "memory" and MemoryVectorStore is not None:
                    store = MemoryVectorStore(
                        collection_name=coll_name,
                        embedding_dimension=embed_dim
                    )
                else:
                    raise ValueError(f"不支持的向量存储后端或后端不可用: {backend_to_use}")
                
                # 存储实例
                self.stores[store_id] = store
                
                # 初始化统计
                self.store_stats[store_id] = VectorStoreStats(
                    store_id=store_id,
                    collection_name=coll_name,
                    backend_type=backend_to_use
                )
                
                logger.info(f"创建向量存储成功: {store_id}, 后端: {backend_to_use}")
                return True
                
            except Exception as e:
                logger.error(f"创建向量存储失败: {e}")
                return False
    
    def delete_store(self, store_id: str) -> bool:
        """
        删除向量存储
        
        Args:
            store_id: 存储ID
            
        Returns:
            是否删除成功
        """
        with self._lock:
            if store_id not in self.stores:
                logger.warning(f"向量存储不存在: {store_id}")
                return False
            
            if store_id == self.default_store_id:
                logger.error(f"不能删除默认向量存储: {store_id}")
                return False
            
            try:
                # 从字典中移除
                del self.stores[store_id]
                del self.store_stats[store_id]
                
                logger.info(f"删除向量存储成功: {store_id}")
                return True
                
            except Exception as e:
                logger.error(f"删除向量存储失败: {e}")
                return False
    
    def add_embedding(self,
                     embedding: List[float],
                     metadata: Dict[str, Any],
                     document: Optional[str] = None,
                     store_id: str = "default",
                     **kwargs) -> Optional[str]:
        """
        添加单个嵌入向量
        
        Args:
            embedding: 嵌入向量
            metadata: 元数据
            document: 文档内容（可选）
            store_id: 存储ID
            **kwargs: 其他参数
            
        Returns:
            嵌入ID，如果失败则返回None
        """
        try:
            store = self.get_store(store_id)
            
            # 添加嵌入
            ids = store.add_embeddings(
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[document] if document else None,
                **kwargs
            )
            
            # 更新统计
            if store_id in self.store_stats:
                stats = self.store_stats[store_id]
                stats.insert_count += 1
                stats.total_embeddings += 1
                stats.last_insert_at = datetime.now().isoformat()
            
            return ids[0] if ids else None
            
        except Exception as e:
            logger.error(f"添加嵌入失败: {e}")
            
            # 更新错误统计
            if store_id in self.store_stats:
                stats = self.store_stats[store_id]
                stats.error_count += 1
                stats.last_error = str(e)
            
            return None
    
    def add_multimodal_item(self,
                           modality: str,
                           data: Any,
                           metadata: Dict[str, Any],
                           store_id: str = "default",
                           **kwargs) -> Optional[str]:
        """
        添加多模态项目
        
        Args:
            modality: 模态类型（text, image, audio, video）
            data: 数据
            metadata: 元数据
            store_id: 存储ID
            **kwargs: 其他参数
            
        Returns:
            项目ID，如果失败则返回None
        """
        try:
            store = self.get_store(store_id)
            
            if hasattr(store, 'add_multimodal_item'):
                item_id = store.add_multimodal_item(
                    modality=modality,
                    data=data,
                    metadata=metadata,
                    **kwargs
                )
                
                # 更新统计
                if store_id in self.store_stats:
                    stats = self.store_stats[store_id]
                    stats.insert_count += 1
                    stats.total_embeddings += 1
                    stats.last_insert_at = datetime.now().isoformat()
                
                return item_id
            else:
                logger.warning(f"向量存储不支持add_multimodal_item方法: {store_id}")
                return None
                
        except Exception as e:
            logger.error(f"添加多模态项目失败: {e}")
            
            # 更新错误统计
            if store_id in self.store_stats:
                stats = self.store_stats[store_id]
                stats.error_count += 1
                stats.last_error = str(e)
            
            return None
    
    def search_similar(self,
                      query_embedding: List[float],
                      n_results: int = 10,
                      store_id: str = "default",
                      **kwargs) -> Dict[str, Any]:
        """
        搜索相似的嵌入向量
        
        Args:
            query_embedding: 查询嵌入向量
            n_results: 返回结果数量
            store_id: 存储ID
            **kwargs: 其他参数
            
        Returns:
            搜索结果
        """
        try:
            store = self.get_store(store_id)
            
            # 搜索
            results = store.search_similar(
                query_embedding=query_embedding,
                n_results=n_results,
                **kwargs
            )
            
            # 更新统计
            if store_id in self.store_stats:
                stats = self.store_stats[store_id]
                stats.query_count += 1
                stats.last_query_at = datetime.now().isoformat()
            
            return results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            
            # 更新错误统计
            if store_id in self.store_stats:
                stats = self.store_stats[store_id]
                stats.error_count += 1
                stats.last_error = str(e)
            
            return {"ids": [], "distances": [], "metadatas": [], "documents": [], "embeddings": []}
    
    def search_multimodal(self,
                         modality: str,
                         query: Any,
                         n_results: int = 10,
                         store_id: str = "default",
                         **kwargs) -> Dict[str, Any]:
        """
        搜索多模态项目
        
        Args:
            modality: 模态类型（text, image, audio, video）
            query: 查询数据
            n_results: 返回结果数量
            store_id: 存储ID
            **kwargs: 其他参数
            
        Returns:
            搜索结果
        """
        try:
            store = self.get_store(store_id)
            
            if hasattr(store, 'search_multimodal'):
                results = store.search_multimodal(
                    modality=modality,
                    query=query,
                    n_results=n_results,
                    **kwargs
                )
                
                # 更新统计
                if store_id in self.store_stats:
                    stats = self.store_stats[store_id]
                    stats.query_count += 1
                    stats.last_query_at = datetime.now().isoformat()
                
                return results
            else:
                logger.warning(f"向量存储不支持search_multimodal方法: {store_id}")
                return {"ids": [], "distances": [], "metadatas": [], "documents": [], "embeddings": []}
                
        except Exception as e:
            logger.error(f"多模态搜索失败: {e}")
            
            # 更新错误统计
            if store_id in self.store_stats:
                stats = self.store_stats[store_id]
                stats.error_count += 1
                stats.last_error = str(e)
            
            return {"ids": [], "distances": [], "metadatas": [], "documents": [], "embeddings": []}
    
    def get_stats(self, store_id: str = "default") -> Dict[str, Any]:
        """
        获取向量存储统计信息
        
        Args:
            store_id: 存储ID
            
        Returns:
            统计信息字典
        """
        with self._lock:
            if store_id not in self.store_stats:
                return {}
            
            stats = self.store_stats[store_id]
            
            # 尝试获取实际存储信息
            try:
                store = self.get_store(store_id)
                if hasattr(store, 'get_collection_info'):
                    info = store.get_collection_info()
                    stats.total_embeddings = info.get('total_embeddings', stats.total_embeddings)
            except Exception as e:
                logger.debug(f"获取存储信息失败: {e}")
            
            return {
                "store_id": stats.store_id,
                "collection_name": stats.collection_name,
                "backend_type": stats.backend_type,
                "total_embeddings": stats.total_embeddings,
                "total_collections": stats.total_collections,
                "query_count": stats.query_count,
                "insert_count": stats.insert_count,
                "update_count": stats.update_count,
                "delete_count": stats.delete_count,
                "error_count": stats.error_count,
                "last_error": stats.last_error,
                "created_at": stats.created_at,
                "last_query_at": stats.last_query_at,
                "last_insert_at": stats.last_insert_at,
                "storage_size_bytes": stats.storage_size_bytes,
                "memory_usage_mb": stats.memory_usage_mb,
                "disk_usage_mb": stats.disk_usage_mb
            }
    
    def list_stores(self) -> List[Dict[str, Any]]:
        """
        列出所有向量存储
        
        Returns:
            存储列表
        """
        with self._lock:
            stores = []
            for store_id in self.stores:
                stats = self.get_stats(store_id)
                stores.append(stats)
            
            return stores
    
    def clear_store(self, store_id: str = "default") -> bool:
        """
        清空向量存储
        
        Args:
            store_id: 存储ID
            
        Returns:
            是否清空成功
        """
        try:
            store = self.get_store(store_id)
            
            if hasattr(store, 'clear_collection'):
                result = store.clear_collection()
                
                # 重置统计
                if store_id in self.store_stats:
                    stats = self.store_stats[store_id]
                    stats.total_embeddings = 0
                    stats.insert_count = 0
                    stats.update_count = 0
                    stats.delete_count = 0
                
                return result
            else:
                logger.warning(f"向量存储不支持clear_collection方法: {store_id}")
                return False
                
        except Exception as e:
            logger.error(f"清空向量存储失败: {e}")
            return False


# 全局实例获取函数
_vector_store_manager_instance = None
_vector_store_manager_lock = threading.Lock()

def get_vector_store_manager(config_path: Optional[str] = None) -> VectorStoreManager:
    """获取向量存储管理器实例"""
    global _vector_store_manager_instance
    
    with _vector_store_manager_lock:
        if _vector_store_manager_instance is None:
            _vector_store_manager_instance = VectorStoreManager(config_path)
        
        return _vector_store_manager_instance


def test_vector_store_manager():
    """测试向量存储管理器"""
    import tempfile
    import shutil
    
    print("=== 测试向量存储管理器 ===")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 获取管理器实例
        manager = get_vector_store_manager()
        
        print("1. 向量存储管理器初始化成功")
        
        # 测试获取默认存储
        store = manager.get_store()
        print(f"2. 获取默认存储成功: {store}")
        
        # 测试添加嵌入
        test_embedding = [0.1] * 768
        test_metadata = {"modality": "text", "source": "test", "content": "测试文本"}
        
        embedding_id = manager.add_embedding(
            embedding=test_embedding,
            metadata=test_metadata,
            document="这是测试文档"
        )
        
        print(f"3. 添加嵌入成功，ID: {embedding_id}")
        
        # 测试搜索
        results = manager.search_similar(
            query_embedding=test_embedding,
            n_results=5
        )
        
        print(f"4. 相似度搜索成功，返回 {len(results['ids'])} 个结果")
        
        # 测试获取统计
        stats = manager.get_stats()
        print(f"5. 获取统计成功，总嵌入数: {stats.get('total_embeddings', 0)}")
        
        # 测试列出存储
        stores = manager.list_stores()
        print(f"6. 列出存储成功，存储数量: {len(stores)}")
        
        # 测试创建新存储
        new_store_id = "test_store"
        created = manager.create_store(
            store_id=new_store_id,
            persist_directory=os.path.join(temp_dir, new_store_id)
        )
        
        if created:
            print(f"7. 创建新存储成功: {new_store_id}")
            
            # 测试在新存储中添加嵌入
            new_embedding_id = manager.add_embedding(
                embedding=test_embedding,
                metadata=test_metadata,
                store_id=new_store_id
            )
            print(f"8. 在新存储中添加嵌入成功，ID: {new_embedding_id}")
            
            # 测试删除存储
            deleted = manager.delete_store(new_store_id)
            print(f"9. 删除存储成功: {deleted}")
        
        # 测试清空存储
        cleared = manager.clear_store()
        print(f"10. 清空存储成功: {cleared}")
        
        print("✅ 所有测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("✅ 临时目录清理完成")


if __name__ == "__main__":
    test_vector_store_manager()