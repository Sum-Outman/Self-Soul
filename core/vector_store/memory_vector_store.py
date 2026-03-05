#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存向量存储

简单的内存向量存储实现，作为ChromaDB的替代方案。
当ChromaDB不可用时提供基本功能。
"""

import numpy as np
import logging
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from collections import defaultdict, OrderedDict
import math
import json

logger = logging.getLogger(__name__)


class LRUCache:
    """简单的LRU缓存"""
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: str):
        """获取缓存值，如果存在则移动到最新"""
        if key not in self.cache:
            return None
        
        # 移动到最新（表示最近使用）
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any):
        """放入缓存，如果达到容量则移除最旧的"""
        if key in self.cache:
            # 更新现有键的值
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # 添加新键
            self.cache[key] = value
            
            # 如果超过容量，移除最旧的
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)


class MemoryVectorStore:
    """内存向量存储"""
    
    def __init__(self, 
                 collection_name: str = "memory_embeddings",
                 embedding_dimension: int = 768):
        """
        初始化内存向量存储
        
        Args:
            collection_name: 集合名称
            embedding_dimension: 嵌入维度
        """
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        
        # 内存存储
        self.embeddings: Dict[str, List[float]] = {}
        self.metadatas: Dict[str, Dict[str, Any]] = {}
        self.documents: Dict[str, str] = {}
        
        # 索引：用于快速搜索
        self.embedding_arrays: Dict[str, np.ndarray] = {}
        
        # 查询缓存（LRU缓存）
        self.query_cache = LRUCache(capacity=50)
        
        logger.info(f"内存向量存储初始化完成，嵌入维度: {embedding_dimension}")
    
    def add_embeddings(self,
                      embeddings: List[List[float]],
                      metadatas: List[Dict[str, Any]],
                      documents: Optional[List[str]] = None,
                      ids: Optional[List[str]] = None) -> List[str]:
        """
        添加嵌入向量到存储
        
        Args:
            embeddings: 嵌入向量列表
            metadatas: 元数据列表
            documents: 文档内容列表（可选）
            ids: 自定义ID列表（可选）
            
        Returns:
            生成的ID列表
        """
        if not embeddings:
            raise ValueError("嵌入向量列表不能为空")
        
        if not metadatas:
            raise ValueError("元数据列表不能为空")
        
        if len(embeddings) != len(metadatas):
            raise ValueError(f"嵌入向量数量({len(embeddings)})与元数据数量({len(metadatas)})不匹配")
        
        if documents and len(embeddings) != len(documents):
            raise ValueError(f"嵌入向量数量({len(embeddings)})与文档数量({len(documents)})不匹配")
        
        # 生成ID（如果未提供）
        if not ids:
            ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
        
        # 验证嵌入维度
        for i, emb in enumerate(embeddings):
            if len(emb) != self.embedding_dimension:
                raise ValueError(f"嵌入向量{i}的维度({len(emb)})与期望维度({self.embedding_dimension})不匹配")
        
        # 添加嵌入到内存存储
        for i, embedding_id in enumerate(ids):
            self.embeddings[embedding_id] = embeddings[i]
            self.metadatas[embedding_id] = metadatas[i]
            if documents:
                self.documents[embedding_id] = documents[i]
            else:
                self.documents[embedding_id] = ""
            
            # 存储为numpy数组用于计算
            self.embedding_arrays[embedding_id] = np.array(embeddings[i], dtype=np.float32)
        
        logger.info(f"成功添加{len(embeddings)}个嵌入向量到内存存储")
        # 清空查询缓存（数据已更改）
        self.query_cache.clear()
        return ids
    
    def search_similar(self,
                      query_embedding: List[float],
                      n_results: int = 10,
                      where: Optional[Dict[str, Any]] = None,
                      where_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        搜索相似的嵌入向量
        
        Args:
            query_embedding: 查询嵌入向量
            n_results: 返回结果数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            
        Returns:
            搜索结果
        """
        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(f"查询嵌入维度({len(query_embedding)})与期望维度({self.embedding_dimension})不匹配")
        
        # 生成缓存键
        cache_key = self._generate_cache_key(query_embedding, n_results, where, where_document)
        
        # 检查缓存
        cached_result = self.query_cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"查询缓存命中，键: {cache_key[:50]}...")
            return cached_result
        
        # 过滤嵌入向量
        filtered_ids = self._filter_embeddings(where, where_document)
        
        if not filtered_ids:
            result = {
                "ids": [],
                "distances": [],
                "metadatas": [],
                "documents": [],
                "embeddings": []
            }
            # 缓存空结果
            self.query_cache.put(cache_key, result)
            return result
        
        # 将查询向量转换为numpy数组
        query_array = np.array(query_embedding, dtype=np.float32)
        
        # 计算相似度（余弦相似度）
        similarities = []
        for embedding_id in filtered_ids:
            emb_array = self.embedding_arrays[embedding_id]
            
            # 计算余弦相似度
            similarity = self._cosine_similarity(query_array, emb_array)
            similarities.append((similarity, embedding_id))
        
        # 按相似度排序（从高到低）
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # 获取前n_results个结果
        top_n = similarities[:n_results]
        
        # 构建结果
        ids = []
        distances = []
        metadatas = []
        documents = []
        embeddings = []
        
        for similarity, embedding_id in top_n:
            ids.append(embedding_id)
            distances.append(1.0 - similarity)  # 将相似度转换为距离
            metadatas.append(self.metadatas[embedding_id])
            documents.append(self.documents[embedding_id])
            embeddings.append(self.embeddings[embedding_id])
        
        logger.debug(f"内存相似度搜索完成，返回{len(ids)}个结果")
        
        # 构建结果字典
        result = {
            "ids": ids,
            "distances": distances,
            "metadatas": metadatas,
            "documents": documents,
            "embeddings": embeddings
        }
        
        # 缓存结果
        self.query_cache.put(cache_key, result)
        
        return result
    
    def _filter_embeddings(self,
                          where: Optional[Dict[str, Any]] = None,
                          where_document: Optional[Dict[str, Any]] = None) -> List[str]:
        """根据条件过滤嵌入向量"""
        filtered_ids = list(self.embeddings.keys())
        
        # 元数据过滤
        if where:
            filtered_ids = [
                embedding_id for embedding_id in filtered_ids
                if self._matches_where(self.metadatas[embedding_id], where)
            ]
        
        # 文档内容过滤
        if where_document:
            filtered_ids = [
                embedding_id for embedding_id in filtered_ids
                if self._matches_where_document(self.documents[embedding_id], where_document)
            ]
        
        return filtered_ids
    
    def _matches_where(self, metadata: Dict[str, Any], where: Dict[str, Any]) -> bool:
        """检查元数据是否匹配where条件"""
        for key, condition in where.items():
            if key not in metadata:
                return False
            
            metadata_value = metadata[key]
            
            # 简单相等检查（可以扩展为支持更多操作符）
            if isinstance(condition, dict):
                # 支持操作符：$eq, $ne, $in, $gt, $lt等
                for op, value in condition.items():
                    if op == "$eq":
                        if metadata_value != value:
                            return False
                    elif op == "$ne":
                        if metadata_value == value:
                            return False
                    elif op == "$in":
                        if metadata_value not in value:
                            return False
                    elif op == "$gt":
                        if not (metadata_value > value):
                            return False
                    elif op == "$lt":
                        if not (metadata_value < value):
                            return False
                    else:
                        # 默认使用相等
                        if metadata_value != value:
                            return False
            else:
                # 直接相等比较
                if metadata_value != condition:
                    return False
        
        return True
    
    def _matches_where_document(self, document: str, where_document: Dict[str, Any]) -> bool:
        """检查文档是否匹配where_document条件"""
        for key, condition in where_document.items():
            if key == "$contains":
                if condition not in document:
                    return False
            elif key == "$not_contains":
                if condition in document:
                    return False
            else:
                # 默认使用包含检查
                if str(condition) not in document:
                    return False
        
        return True
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def get_by_id(self, id: str) -> Dict[str, Any]:
        """根据ID获取嵌入向量"""
        if id not in self.embeddings:
            return {
                "ids": [],
                "distances": [],
                "metadatas": [],
                "documents": [],
                "embeddings": []
            }
        
        return {
            "ids": [id],
            "distances": [0.0],
            "metadatas": [self.metadatas[id]],
            "documents": [self.documents[id]],
            "embeddings": [self.embeddings[id]]
        }
    
    def update_embedding(self,
                        id: str,
                        embedding: Optional[List[float]] = None,
                        metadata: Optional[Dict[str, Any]] = None,
                        document: Optional[str] = None) -> bool:
        """
        更新嵌入向量
        
        Args:
            id: 要更新的ID
            embedding: 新的嵌入向量（可选）
            metadata: 新的元数据（可选）
            document: 新的文档内容（可选）
            
        Returns:
            是否成功
        """
        if id not in self.embeddings:
            return False
        
        # 更新嵌入向量
        if embedding is not None:
            if len(embedding) != self.embedding_dimension:
                raise ValueError(f"嵌入维度({len(embedding)})与期望维度({self.embedding_dimension})不匹配")
            
            self.embeddings[id] = embedding
            self.embedding_arrays[id] = np.array(embedding, dtype=np.float32)
        
        # 更新元数据
        if metadata is not None:
            current_metadata = self.metadatas[id]
            current_metadata.update(metadata)
            self.metadatas[id] = current_metadata
        
        # 更新文档
        if document is not None:
            self.documents[id] = document
        
        logger.info(f"成功更新嵌入向量: {id}")
        # 清空查询缓存（数据已更改）
        self.query_cache.clear()
        return True
    
    def delete_embeddings(self, ids: Union[str, List[str]]) -> bool:
        """
        删除嵌入向量
        
        Args:
            ids: 要删除的ID或ID列表
            
        Returns:
            是否成功
        """
        if isinstance(ids, str):
            ids = [ids]
        
        success = True
        for embedding_id in ids:
            if embedding_id in self.embeddings:
                del self.embeddings[embedding_id]
                del self.metadatas[embedding_id]
                del self.documents[embedding_id]
                if embedding_id in self.embedding_arrays:
                    del self.embedding_arrays[embedding_id]
                logger.info(f"删除嵌入向量: {embedding_id}")
            else:
                logger.warning(f"嵌入向量不存在: {embedding_id}")
                success = False
        
        # 清空查询缓存（数据已更改）
        if success:
            self.query_cache.clear()
        
        return success
    
    def count(self) -> int:
        """获取嵌入向量数量"""
        return len(self.embeddings)
    
    def clear_collection(self) -> bool:
        """清空集合"""
        self.embeddings.clear()
        self.metadatas.clear()
        self.documents.clear()
        self.embedding_arrays.clear()
        
        logger.info("清空内存向量存储集合")
        # 清空查询缓存（数据已更改）
        self.query_cache.clear()
        return True
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        return {
            "collection_name": self.collection_name,
            "embedding_dimension": self.embedding_dimension,
            "total_embeddings": self.count(),
            "storage_type": "memory",
            "created_at": datetime.now().isoformat()
        }
    
    def list_all(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        列出所有嵌入向量
        
        Args:
            limit: 限制返回数量
            
        Returns:
            所有嵌入向量信息
        """
        all_ids = list(self.embeddings.keys())
        
        if limit is not None and limit > 0:
            all_ids = all_ids[:limit]
        
        return {
            "ids": all_ids,
            "distances": [0.0] * len(all_ids),
            "metadatas": [self.metadatas[id] for id in all_ids],
            "documents": [self.documents[id] for id in all_ids],
            "embeddings": [self.embeddings[id] for id in all_ids]
        }
    
    def _generate_cache_key(self, 
                           query_embedding: List[float], 
                           n_results: int,
                           where: Optional[Dict[str, Any]] = None,
                           where_document: Optional[Dict[str, Any]] = None) -> str:
        """
        生成缓存键
        
        Args:
            query_embedding: 查询嵌入向量
            n_results: 返回结果数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            
        Returns:
            缓存键字符串
        """
        import hashlib
        import json
        
        # 将查询嵌入向量转换为字符串表示
        query_str = json.dumps(query_embedding, sort_keys=True)
        
        # 将过滤条件转换为字符串表示
        where_str = json.dumps(where, sort_keys=True) if where else "null"
        where_doc_str = json.dumps(where_document, sort_keys=True) if where_document else "null"
        
        # 组合所有参数
        key_parts = [
            query_str,
            str(n_results),
            where_str,
            where_doc_str
        ]
        
        key_string = "|".join(key_parts)
        
        # 生成哈希（缩短键长度）
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()


def test_memory_vector_store():
    """测试内存向量存储"""
    import tempfile
    
    print("=== 测试内存向量存储 ===")
    
    try:
        # 创建向量存储
        vector_store = MemoryVectorStore(
            collection_name="test_memory_collection",
            embedding_dimension=384
        )
        
        print("1. 内存向量存储初始化成功")
        
        # 生成测试嵌入向量
        test_embeddings = [
            [0.1 * i for i in range(384)],
            [0.2 * i for i in range(384)],
            [0.3 * i for i in range(384)]
        ]
        
        test_metadatas = [
            {"modality": "text", "source": "test1", "timestamp": "2024-01-01"},
            {"modality": "image", "source": "test2", "timestamp": "2024-01-02"},
            {"modality": "audio", "source": "test3", "timestamp": "2024-01-03"}
        ]
        
        test_documents = [
            "这是第一个测试文档",
            "这是第二个测试文档",
            "这是第三个测试文档"
        ]
        
        # 添加嵌入向量
        ids = vector_store.add_embeddings(
            embeddings=test_embeddings,
            metadatas=test_metadatas,
            documents=test_documents
        )
        
        print(f"2. 添加{len(ids)}个嵌入向量成功，ID: {ids}")
        
        # 搜索相似向量
        query_embedding = [0.15 * i for i in range(384)]
        results = vector_store.search_similar(query_embedding, n_results=2)
        
        print(f"3. 相似度搜索成功，返回{len(results['ids'])}个结果")
        print(f"   最相似ID: {results['ids'][0] if results['ids'] else '无'}")
        
        # 根据ID获取
        retrieved = vector_store.get_by_id(ids[0])
        print(f"4. 根据ID获取成功: {retrieved['ids'][0]}")
        
        # 更新嵌入向量
        updated = vector_store.update_embedding(
            id=ids[0],
            metadata={"modality": "text", "source": "updated", "timestamp": "2024-01-04"}
        )
        print(f"5. 更新嵌入向量成功: {updated}")
        
        # 获取数量
        count = vector_store.count()
        print(f"6. 集合数量: {count}")
        
        # 列出所有
        all_items = vector_store.list_all(limit=2)
        print(f"7. 列出所有成功，获取{len(all_items['ids'])}个项目")
        
        # 获取集合信息
        info = vector_store.get_collection_info()
        print(f"8. 集合信息: {info}")
        
        # 删除嵌入向量
        deleted = vector_store.delete_embeddings(ids[2])
        print(f"9. 删除嵌入向量成功: {deleted}")
        
        # 清空集合
        cleared = vector_store.clear_collection()
        print(f"10. 清空集合成功: {cleared}")
        
        print("✅ 所有测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_memory_vector_store()