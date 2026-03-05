"""
多模态向量存储

支持文本、图像、音频、视频等多种模态的嵌入生成、存储和检索。
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import torch

from .chroma_vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class MultimodalVectorStore:
    """多模态向量存储"""
    
    def __init__(self,
                 collection_name: str = "multimodal_embeddings",
                 persist_directory: str = "./chroma_db",
                 embedding_dimension: int = 768):
        """
        初始化多模态向量存储
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
            embedding_dimension: 嵌入维度
        """
        self.embedding_dimension = embedding_dimension
        
        # 初始化底层向量存储
        self.vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_dimension=embedding_dimension
        )
        
        # 初始化嵌入生成器
        self.embedding_generators = {}
        self._initialize_embedding_generators()
        
        logger.info(f"多模态向量存储初始化完成，嵌入维度: {embedding_dimension}")
    
    def _initialize_embedding_generators(self):
        """初始化嵌入生成器"""
        # 延迟加载嵌入生成器
        logger.info("嵌入生成器将在需要时延迟加载")
    
    def _get_text_embedding_generator(self):
        """获取文本嵌入生成器"""
        try:
            if "text" not in self.embedding_generators:
                from sentence_transformers import SentenceTransformer
                
                # 使用轻量级模型
                model_name = "all-MiniLM-L6-v2"  # 384维
                model = SentenceTransformer(model_name)
                self.embedding_generators["text"] = model
                logger.info(f"文本嵌入生成器加载完成: {model_name}")
            
            return self.embedding_generators["text"]
        except ImportError:
            logger.warning("sentence-transformers不可用，使用简单文本嵌入")
            return None
    
    def _get_image_embedding_generator(self):
        """获取图像嵌入生成器"""
        try:
            if "image" not in self.embedding_generators:
                import torchvision.models as models
                import torchvision.transforms as transforms
                from PIL import Image
                
                # 使用ResNet预训练模型
                model = models.resnet50(pretrained=True)
                # 移除最后的全连接层
                model = torch.nn.Sequential(*list(model.children())[:-1])
                model.eval()
                
                # 图像预处理
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                
                self.embedding_generators["image"] = {
                    "model": model,
                    "transform": transform
                }
                logger.info("图像嵌入生成器加载完成: ResNet50")
            
            return self.embedding_generators["image"]
        except ImportError:
            logger.warning("torchvision不可用，无法进行图像嵌入")
            return None
    
    def generate_text_embedding(self, text: str) -> List[float]:
        """
        生成文本嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            文本嵌入向量
        """
        generator = self._get_text_embedding_generator()
        
        if generator is not None:
            try:
                # 使用sentence-transformers
                embedding = generator.encode(text)
                # 确保维度匹配
                if len(embedding) != self.embedding_dimension:
                    # 如果维度不匹配，进行投影或填充
                    if len(embedding) < self.embedding_dimension:
                        # 填充
                        padding = [0.0] * (self.embedding_dimension - len(embedding))
                        embedding = np.concatenate([embedding, padding])
                    else:
                        # 截断
                        embedding = embedding[:self.embedding_dimension]
                
                return embedding.tolist()
            except Exception as e:
                logger.error(f"文本嵌入生成失败: {e}")
        
        # 降级方案：使用简单文本哈希嵌入
        logger.warning("使用简单文本嵌入（降级方案）")
        return self._generate_simple_text_embedding(text)
    
    def _generate_simple_text_embedding(self, text: str) -> List[float]:
        """生成简单文本嵌入（降级方案）"""
        import hashlib
        import numpy as np
        
        # 使用确定性哈希生成嵌入
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash, 16) % 10000
        
        # 使用确定性随机数生成嵌入
        np.random.seed(seed)
        embedding = np.random.randn(self.embedding_dimension).astype(np.float32)
        
        # 归一化
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()
    
    def generate_image_embedding(self, image_data: Union[bytes, np.ndarray, torch.Tensor]) -> List[float]:
        """
        生成图像嵌入
        
        Args:
            image_data: 图像数据（字节、numpy数组或张量）
            
        Returns:
            图像嵌入向量
        """
        generator = self._get_image_embedding_generator()
        
        if generator is not None:
            try:
                model = generator["model"]
                transform = generator["transform"]
                
                # 转换图像数据
                if isinstance(image_data, bytes):
                    from PIL import Image
                    import io
                    image = Image.open(io.BytesIO(image_data))
                    image = image.convert("RGB")
                elif isinstance(image_data, np.ndarray):
                    from PIL import Image
                    image = Image.fromarray(image_data.astype(np.uint8))
                elif isinstance(image_data, torch.Tensor):
                    # 假设已经是处理过的张量
                    image_tensor = image_data
                else:
                    raise ValueError(f"不支持的图像数据类型: {type(image_data)}")
                
                # 应用变换（如果不是张量）
                if not isinstance(image_data, torch.Tensor):
                    image_tensor = transform(image)
                    # 添加批次维度
                    image_tensor = image_tensor.unsqueeze(0)
                
                # 生成嵌入
                with torch.no_grad():
                    embedding = model(image_tensor)
                    embedding = embedding.squeeze().numpy()
                
                # 归一化
                embedding = embedding / np.linalg.norm(embedding)
                
                # 确保维度匹配
                if len(embedding) != self.embedding_dimension:
                    if len(embedding) < self.embedding_dimension:
                        padding = np.zeros(self.embedding_dimension - len(embedding))
                        embedding = np.concatenate([embedding, padding])
                    else:
                        embedding = embedding[:self.embedding_dimension]
                
                return embedding.tolist()
                
            except Exception as e:
                logger.error(f"图像嵌入生成失败: {e}")
        
        # 降级方案：使用随机嵌入
        logger.warning("使用随机图像嵌入（降级方案）")
        return self._generate_random_embedding()
    
    def _generate_random_embedding(self) -> List[float]:
        """生成随机嵌入（降级方案）"""
        import numpy as np
        embedding = np.random.randn(self.embedding_dimension).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    def add_multimodal_item(self,
                           modality: str,
                           data: Any,
                           metadata: Dict[str, Any],
                           document: Optional[str] = None,
                           custom_id: Optional[str] = None) -> str:
        """
        添加多模态项目
        
        Args:
            modality: 模态类型（"text", "image", "audio", "video"）
            data: 模态数据
            metadata: 元数据
            document: 关联文档（可选）
            custom_id: 自定义ID（可选）
            
        Returns:
            项目ID
        """
        # 生成嵌入
        if modality == "text":
            embedding = self.generate_text_embedding(data)
        elif modality == "image":
            embedding = self.generate_image_embedding(data)
        elif modality == "audio":
            # 音频嵌入（待实现）
            embedding = self._generate_random_embedding()
            logger.warning(f"音频嵌入使用随机嵌入（待实现）")
        elif modality == "video":
            # 视频嵌入（待实现）
            embedding = self._generate_random_embedding()
            logger.warning(f"视频嵌入使用随机嵌入（待实现）")
        else:
            raise ValueError(f"不支持的模态类型: {modality}")
        
        # 添加模态信息到元数据
        metadata_with_modality = metadata.copy()
        metadata_with_modality["modality"] = modality
        metadata_with_modality["added_at"] = datetime.now().isoformat()
        
        # 添加到向量存储
        ids = self.vector_store.add_embeddings(
            embeddings=[embedding],
            metadatas=[metadata_with_modality],
            documents=[document] if document else [""],
            ids=[custom_id] if custom_id else None
        )
        
        logger.info(f"添加{modality}模态项目成功，ID: {ids[0]}")
        return ids[0]
    
    def search_similar(self,
                      query_embedding: List[float],
                      n_results: int = 10,
                      modality_filter: Optional[str] = None,
                      metadata_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        搜索相似项目
        
        Args:
            query_embedding: 查询嵌入向量
            n_results: 返回结果数量
            modality_filter: 模态过滤条件
            metadata_filter: 元数据过滤条件
            
        Returns:
            搜索结果
        """
        # 构建查询条件
        where = metadata_filter.copy() if metadata_filter else {}
        if modality_filter:
            where["modality"] = modality_filter
        
        return self.vector_store.search_similar(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where if where else None
        )
    
    def search_by_text(self,
                      query_text: str,
                      n_results: int = 10,
                      modality_filter: Optional[str] = None,
                      metadata_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        通过文本搜索相似项目
        
        Args:
            query_text: 查询文本
            n_results: 返回结果数量
            modality_filter: 模态过滤条件
            metadata_filter: 元数据过滤条件
            
        Returns:
            搜索结果
        """
        # 生成文本嵌入
        query_embedding = self.generate_text_embedding(query_text)
        
        return self.search_similar(
            query_embedding=query_embedding,
            n_results=n_results,
            modality_filter=modality_filter,
            metadata_filter=metadata_filter
        )
    
    def search_by_image(self,
                       query_image: Union[bytes, np.ndarray, torch.Tensor],
                       n_results: int = 10,
                       modality_filter: Optional[str] = None,
                       metadata_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        通过图像搜索相似项目
        
        Args:
            query_image: 查询图像
            n_results: 返回结果数量
            modality_filter: 模态过滤条件
            metadata_filter: 元数据过滤条件
            
        Returns:
            搜索结果
        """
        # 生成图像嵌入
        query_embedding = self.generate_image_embedding(query_image)
        
        return self.search_similar(
            query_embedding=query_embedding,
            n_results=n_results,
            modality_filter=modality_filter,
            metadata_filter=metadata_filter
        )
    
    def get_item(self, item_id: str) -> Dict[str, Any]:
        """
        获取项目信息
        
        Args:
            item_id: 项目ID
            
        Returns:
            项目信息
        """
        return self.vector_store.get_by_id(item_id)
    
    def update_item(self,
                   item_id: str,
                   metadata: Optional[Dict[str, Any]] = None,
                   document: Optional[str] = None) -> bool:
        """
        更新项目
        
        Args:
            item_id: 项目ID
            metadata: 新的元数据
            document: 新的文档
            
        Returns:
            是否成功
        """
        return self.vector_store.update_embedding(
            id=item_id,
            metadata=metadata,
            document=document
        )
    
    def delete_item(self, item_id: str) -> bool:
        """
        删除项目
        
        Args:
            item_id: 项目ID
            
        Returns:
            是否成功
        """
        return self.vector_store.delete_embeddings(item_id)
    
    def count_items(self, modality: Optional[str] = None) -> int:
        """
        统计项目数量
        
        Args:
            modality: 模态过滤条件
            
        Returns:
            项目数量
        """
        if modality:
            # 需要获取所有项目并过滤
            all_items = self.vector_store.list_all(limit=10000)
            count = 0
            for metadata in all_items.get("metadatas", []):
                if metadata and metadata.get("modality") == modality:
                    count += 1
            return count
        else:
            return self.vector_store.count()
    
    def list_items(self,
                  limit: int = 100,
                  offset: int = 0,
                  modality: Optional[str] = None) -> Dict[str, Any]:
        """
        列出项目
        
        Args:
            limit: 返回数量限制
            offset: 偏移量
            modality: 模态过滤条件
            
        Returns:
            项目列表
        """
        all_items = self.vector_store.list_all(limit=limit + offset, offset=0)
        
        if modality:
            # 过滤模态
            filtered_ids = []
            filtered_metadatas = []
            filtered_documents = []
            filtered_embeddings = []
            
            for i, metadata in enumerate(all_items.get("metadatas", [])):
                if metadata and metadata.get("modality") == modality:
                    filtered_ids.append(all_items["ids"][i])
                    filtered_metadatas.append(metadata)
                    filtered_documents.append(all_items["documents"][i])
                    if all_items.get("embeddings"):
                        filtered_embeddings.append(all_items["embeddings"][i])
            
            # 应用分页
            start_idx = offset
            end_idx = min(offset + limit, len(filtered_ids))
            
            return {
                "ids": filtered_ids[start_idx:end_idx],
                "metadatas": filtered_metadatas[start_idx:end_idx],
                "documents": filtered_documents[start_idx:end_idx],
                "embeddings": filtered_embeddings[start_idx:end_idx] if filtered_embeddings else []
            }
        else:
            # 直接应用分页
            start_idx = offset
            end_idx = min(offset + limit, len(all_items["ids"]))
            
            return {
                "ids": all_items["ids"][start_idx:end_idx],
                "metadatas": all_items["metadatas"][start_idx:end_idx],
                "documents": all_items["documents"][start_idx:end_idx],
                "embeddings": all_items["embeddings"][start_idx:end_idx] if all_items.get("embeddings") else []
            }
    
    def clear_all(self) -> bool:
        """清空所有项目"""
        return self.vector_store.clear_collection()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_count = self.vector_store.count()
        
        # 统计各模态数量
        modality_counts = {}
        all_items = self.vector_store.list_all(limit=total_count)
        
        for metadata in all_items.get("metadatas", []):
            if metadata:
                modality = metadata.get("modality", "unknown")
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
        
        return {
            "total_items": total_count,
            "modality_counts": modality_counts,
            "embedding_dimension": self.embedding_dimension,
            "collection_info": self.vector_store.get_collection_info()
        }


def test_multimodal_vector_store():
    """测试多模态向量存储"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== 测试多模态向量存储 ===")
    
    try:
        # 创建多模态向量存储
        vector_store = MultimodalVectorStore(
            collection_name="test_multimodal",
            persist_directory="./test_multimodal_db",
            embedding_dimension=384
        )
        
        print("1. 多模态向量存储初始化成功")
        
        # 添加文本项目
        text_id = vector_store.add_multimodal_item(
            modality="text",
            data="这是一个测试文本",
            metadata={"source": "test", "category": "example"},
            document="完整的测试文档内容"
        )
        print(f"2. 添加文本项目成功，ID: {text_id}")
        
        # 添加图像项目（使用模拟数据）
        import numpy as np
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image_id = vector_store.add_multimodal_item(
            modality="image",
            data=test_image,
            metadata={"source": "test", "category": "example"},
            document="测试图像描述"
        )
        print(f"3. 添加图像项目成功，ID: {image_id}")
        
        # 搜索相似文本
        text_results = vector_store.search_by_text(
            query_text="测试文本",
            n_results=5
        )
        print(f"4. 文本搜索成功，返回{len(text_results['ids'])}个结果")
        
        # 搜索相似图像
        image_results = vector_store.search_by_image(
            query_image=test_image,
            n_results=5
        )
        print(f"5. 图像搜索成功，返回{len(image_results['ids'])}个结果")
        
        # 获取项目
        item = vector_store.get_item(text_id)
        print(f"6. 获取项目成功: {item['ids'][0]}")
        
        # 更新项目
        updated = vector_store.update_item(
            item_id=text_id,
            metadata={"source": "updated", "category": "example", "updated": True}
        )
        print(f"7. 更新项目成功: {updated}")
        
        # 统计信息
        stats = vector_store.get_statistics()
        print(f"8. 统计信息: {stats}")
        
        # 列出项目
        items = vector_store.list_items(limit=2)
        print(f"9. 列出项目成功，获取{len(items['ids'])}个项目")
        
        # 删除项目
        deleted = vector_store.delete_item(image_id)
        print(f"10. 删除项目成功: {deleted}")
        
        # 清空所有
        cleared = vector_store.clear_all()
        print(f"11. 清空所有成功: {cleared}")
        
        print("✅ 所有测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_multimodal_vector_store()