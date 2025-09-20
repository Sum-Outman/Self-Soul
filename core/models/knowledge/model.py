"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

"""
Knowledge Expert Model - Multidisciplinary Knowledge System
"""

import json
import logging
import os
import time
import datetime
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from ..base_model import BaseModel
from core.api_model_connector import api_model_connector
from core.system_settings_manager import system_settings_manager

# Try to import PDF and DOCX processing libraries
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("PyPDF2 not installed, PDF import functionality will be unavailable")

try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    logging.warning("python-docx not installed, DOCX import functionality will be unavailable")


"""
KnowledgeModel类 - 中文类描述
KnowledgeModel Class - English class description
"""
class KnowledgeModel(BaseModel):
    """核心知识库专家模型
    Core Knowledge Expert Model
    
    功能：多学科知识存储与检索，辅助其他模型，教学辅导
    Function: Multidisciplinary knowledge storage/retrieval,
              assisting other models, teaching/tutoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化函数 - 中文函数描述
        __init__ Function - English function description
        
        Args:
            params: 参数描述 (Parameter description)
            
        Returns:
            返回值描述 (Return value description)
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.model_id = "knowledge"
        self.knowledge_graph = {}
        self.knowledge_embeddings = {}  # 知识向量存储 | Knowledge vector storage
        self.domain_weights = {
            "physics": 0.9, "mathematics": 0.95, "chemistry": 0.85,
            "medicine": 0.9, "law": 0.8, "history": 0.75,
            "sociology": 0.8, "humanities": 0.85, "psychology": 0.9,
            "economics": 0.85, "management": 0.9, "mechanical_engineering": 0.9,
            "electrical_engineering": 0.9, "food_engineering": 0.8,
            "chemical_engineering": 0.85, "computer_science": 0.95
        }  # 各学科知识权重 | Knowledge weights for each domain
        
        # 检查是否使用外部API模型 | Check if using external API model
        self.use_external_api = False
        if config and config.get('use_external_api'):
            self.use_external_api = True
            self.external_model_name = config.get('external_model_name', '')
            self.logger.info(f"知识库模型配置为使用外部API: {self.external_model_name} | Knowledge model configured to use external API: {self.external_model_name}")
        
        # 初始化语义嵌入模型 | Initialize semantic embedding model
        self._init_embedding_model()
        
        # 加载知识库数据 | Load knowledge base
        self.load_knowledge_base()
        self.logger.info("知识库模型初始化完成 | Knowledge model initialized")
    
    def initialize(self) -> Dict[str, Any]:
        """初始化模型资源 | Initialize model resources
        返回:
            初始化结果 | Initialization result
        """
        try:
            # 知识库模型已经在__init__中完成了大部分初始化
            # Knowledge model already completed most initialization in __init__
            self.is_initialized = True
            self.logger.info("知识库模型资源初始化完成 | Knowledge model resources initialized")
            return {"success": True, "message": "知识库模型初始化成功 | Knowledge model initialized successfully"}
        except Exception as e:
            self.logger.error(f"知识库模型初始化失败: {str(e)} | Knowledge model initialization failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据 | Process input data
        参数:
            input_data: 输入数据，包含操作类型和参数 | Input data containing operation type and parameters
        返回:
            处理结果 | Processing result
        """
        try:
            operation = input_data.get("operation", "query")
            
            if operation == "query":
                # 知识查询操作 | Knowledge query operation
                domain = input_data.get("domain")
                query = input_data.get("query", "")
                top_k = input_data.get("top_k", 5)
                
                if self.embedding_model:
                    # 使用语义搜索 | Use semantic search
                    results = self.semantic_search(query, domain, top_k)
                    return {
                        "success": True,
                        "operation": "semantic_search",
                        "results": results,
                        "count": len(results)
                    }
                else:
                    # 使用关键词搜索 | Use keyword search
                    if domain:
                        result = self.query_knowledge(domain, query)
                        return {
                            "success": True,
                            "operation": "keyword_search",
                            "results": result.get("results", []),
                            "count": len(result.get("results", []))
                        }
                    else:
                        # 如果没有指定领域，搜索所有领域 | If no domain specified, search all domains
                        all_results = []
                        for domain_name in self.knowledge_graph.keys():
                            result = self.query_knowledge(domain_name, query)
                            all_results.extend(result.get("results", []))
                        return {
                            "success": True,
                            "operation": "keyword_search_all",
                            "results": all_results[:top_k],
                            "count": len(all_results)
                        }
            
            elif operation == "assist":
                # 辅助其他模型 | Assist other models
                model_id = input_data.get("model_id")
                task_context = input_data.get("task_context", {})
                result = self.assist_model(model_id, task_context)
                return {
                    "success": True,
                    "operation": "assist",
                    "result": result
                }
            
            elif operation == "explain":
                # 解释知识概念 | Explain knowledge concept
                concept = input_data.get("concept")
                if concept:
                    result = self.explain_knowledge(concept)
                    return {
                        "success": True,
                        "operation": "explain",
                        "result": result
                    }
                else:
                    return {"success": False, "error": "缺少概念参数 | Missing concept parameter"}
            
            elif operation == "summary":
                # 获取知识库摘要 | Get knowledge base summary
                domain = input_data.get("domain")
                result = self.get_knowledge_summary(domain)
                return {
                    "success": True,
                    "operation": "summary",
                    "result": result
                }
            
            elif operation == "add":
                # 添加知识 | Add knowledge
                concept = input_data.get("concept")
                attributes = input_data.get("attributes", {})
                relationships = input_data.get("relationships", [])
                domain = input_data.get("domain", "general")
                
                if concept:
                    result = self.add_knowledge(concept, attributes, relationships, domain)
                    return {
                        "success": result.get("status") == "success",
                        "operation": "add",
                        "result": result
                    }
                else:
                    return {"success": False, "error": "缺少概念参数 | Missing concept parameter"}
            
            else:
                return {"success": False, "error": f"不支持的操作类型: {operation} | Unsupported operation type: {operation}"}
                
        except Exception as e:
            self.logger.error(f"知识库处理失败: {str(e)} | Knowledge processing failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _init_embedding_model(self):
        """初始化语义嵌入模型 | Initialize semantic embedding model"""
        try:
            # 尝试导入sentence-transformers用于语义搜索 | Try to import sentence-transformers for semantic search
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("语义嵌入模型加载成功 | Semantic embedding model loaded successfully")
        except ImportError:
            self.logger.warning("sentence-transformers未安装，将使用关键词搜索 | sentence-transformers not installed, will use keyword search")
            self.embedding_model = None

    def load_knowledge_base(self):
        """加载多学科知识库 | Load multidisciplinary knowledge base"""
        # 使用绝对路径确保在不同操作系统上都能正确找到知识库目录
        # Use absolute path to ensure knowledge base directory can be found correctly on different operating systems
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        knowledge_path = os.path.join(base_dir, "data", "knowledge")
        required_domains = [
            "physics", "mathematics", "chemistry", "medicine", "law", "history",
            "sociology", "humanities", "psychology", "economics", "management",
            "mechanical_engineering", "electrical_engineering", "food_engineering",
            "chemical_engineering", "computer_science"
        ]
        
        try:
            # 确保加载所有必需领域 | Ensure all required domains are loaded
            for domain in required_domains:
                file_path = os.path.join(knowledge_path, f"{domain}.json")
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        self.knowledge_graph[domain] = json.load(f)
                        self.logger.info(f"已加载{domain}知识领域 | Loaded {domain} knowledge domain")
                else:
                    self.logger.warning(f"缺失知识领域文件: {domain}.json | Missing knowledge domain file: {domain}.json")
                    self.knowledge_graph[domain] = {}  # 创建空知识库 | Create empty knowledge base
            
            # 加载其他可选领域 | Load additional optional domains
            for domain_file in os.listdir(knowledge_path):
                if domain_file.endswith(".json"):
                    domain = domain_file.split(".")[0]
                    if domain not in required_domains:  # 避免重复加载 | Avoid duplicate loading
                        with open(os.path.join(knowledge_path, domain_file), "r", encoding="utf-8") as f:
                            self.knowledge_graph[domain] = json.load(f)
                            self.logger.info(f"已加载额外知识领域: {domain} | Loaded additional knowledge domain: {domain}")
            
            # 构建知识嵌入 | Build knowledge embeddings
            self.build_knowledge_embeddings()
            
        except Exception as e:
            self.logger.error(f"知识库加载失败: {str(e)} | Knowledge base loading failed: {str(e)}")

    def build_knowledge_embeddings(self):
        """构建知识库的语义嵌入 | Build semantic embeddings for knowledge base"""
        if self.embedding_model is None:
            self.logger.warning("无嵌入模型可用，跳过嵌入构建 | No embedding model available, skipping embedding construction")
            return
        
        try:
            self.knowledge_embeddings = {}
            for domain, concepts in self.knowledge_graph.items():
                domain_embeddings = {}
                for concept, details in concepts.items():
                    # 为每个概念创建嵌入 | Create embedding for each concept
                    # 从attributes中获取definition，或者使用默认描述
                    description_text = ""
                    if isinstance(details, dict) and 'attributes' in details:
                        if 'definition' in details['attributes']:
                            description_text = details['attributes']['definition']
                        elif 'description' in details['attributes']:
                            description_text = details['attributes']['description']
                    
                    text_to_embed = f"{concept} {description_text}"
                    embedding = self.embedding_model.encode(text_to_embed)
                    domain_embeddings[concept] = {
                        "embedding": embedding,
                        "details": details
                    }
                self.knowledge_embeddings[domain] = domain_embeddings
                self.logger.info(f"已构建{domain}领域知识嵌入 | Built embeddings for {domain} domain")
        except Exception as e:
            self.logger.error(f"知识嵌入构建失败: {str(e)} | Knowledge embedding construction failed: {str(e)}")

    def semantic_search(self, query: str, domain: str = None, top_k: int = 5) -> List[Dict]:
        """语义搜索知识库 | Semantic search knowledge base
        参数:
            query: 查询文本 | Query text
            domain: 特定领域 (可选) | Specific domain (optional)
            top_k: 返回结果数量 | Number of results to return
        返回:
            相关知识点列表 | List of relevant knowledge points
        """
        if self.embedding_model is None:
            self.logger.warning("无嵌入模型可用，退回关键词搜索 | No embedding model available, falling back to keyword search")
            return self.query_knowledge(domain or "", query).get("results", [])[:top_k]
        
        try:
            query_embedding = self.embedding_model.encode(query)
            results = []
            
            search_domains = [domain] if domain else self.knowledge_embeddings.keys()
            
            for search_domain in search_domains:
                if search_domain in self.knowledge_embeddings:
                    for concept, embedding_data in self.knowledge_embeddings[search_domain].items():
                        similarity = np.dot(query_embedding, embedding_data["embedding"]) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(embedding_data["embedding"])
                        )
                        
                        if similarity > 0.3:  # 相似度阈值 | Similarity threshold
                            results.append({
                                "domain": search_domain,
                                "concept": concept,
                                "similarity": float(similarity),
                                "details": embedding_data["details"]
                            })
            
            # 按相似度排序 | Sort by similarity
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"语义搜索失败: {str(e)} | Semantic search failed: {str(e)}")
            return []

    def query_knowledge(self, domain: str, query: str) -> Dict[str, Any]:
        """查询特定领域知识 | Query knowledge in specific domain
        参数:
            domain: 知识领域 (物理/数学/化学等)
            query: 查询关键词
        返回:
            相关知识点列表 | List of relevant knowledge points
        """
        # 如果使用外部API，优先调用外部模型
        if self.use_external_api and self.external_model_name:
            return self._query_external_api(domain, query)
        
        try:
            results = []
            if domain in self.knowledge_graph:
                # 简单关键词匹配 | Simple keyword matching
                for concept, details in self.knowledge_graph[domain].items():
                    # 获取描述文本，从attributes中获取definition或description
                    description_texts = []
                    if isinstance(details, dict) and 'attributes' in details:
                        if 'definition' in details['attributes']:
                            description_texts.append(details['attributes']['definition'])
                        if 'description' in details['attributes']:
                            if isinstance(details['attributes']['description'], list):
                                description_texts.extend(details['attributes']['description'])
                            else:
                                description_texts.append(details['attributes']['description'])
                    
                    # 检查查询是否匹配概念名称或描述
                    query_matches = (
                        query.lower() in concept.lower() or 
                        any(query.lower() in desc.lower() for desc in description_texts)
                    )
                    
                    if query_matches:
                        results.append({
                            "concept": concept,
                            "description": description_texts,
                            "related": details.get("related", [])
                        })
            return {"domain": domain, "results": results}
        except Exception as e:
            self.logger.error(f"知识查询失败: {str(e)} | Knowledge query failed: {str(e)}")
            return {"error": str(e)}

    def search_knowledge(self, domain: str, query: str) -> Dict[str, Any]:
        """搜索知识库 (query_knowledge的别名方法)
        Search knowledge base (alias method for query_knowledge)
        
        参数:
            domain: 知识领域 | Knowledge domain
            query: 搜索查询 | Search query
            
        返回:
            搜索结果 | Search results
        """
        return self.query_knowledge(domain, query)

    def assist_model(self, model_id: str, task_context: Dict) -> Dict[str, Any]:
        """辅助其他模型完成任务 | Assist other models in completing tasks
        参数:
            model_id: 需要辅助的模型ID
            task_context: 任务上下文信息
        返回:
            辅助建议和知识支持 | Assistance suggestions and knowledge support
        """
        try:
            # 模型类型到知识领域的映射 | Model type to knowledge domain mapping
            model_domain_map = {
                "manager": ("management", ["任务分解策略", "资源优化分配"]),
                "language": ("linguistics", ["情感分析框架", "多语言处理技术"]),
                "audio": ("acoustics", ["声纹识别技术", "音频降噪算法"]),
                "vision": ("computer_vision", ["图像增强技术", "目标检测算法"]),
                "video": ("video_processing", ["关键帧提取", "运动估计技术"]),
                "spatial": ("spatial_reasoning", ["3D重建技术", "SLAM算法"]),
                "sensor": ("sensor_fusion", ["多传感器融合", "卡尔曼滤波"]),
                "computer": ("computer_science", ["分布式计算", "容错机制"]),
                "motion": ("robotics", ["运动规划算法", "动力学模型"]),
                "knowledge": ("knowledge_engineering", ["知识图谱推理", "本体论建模"]),
                "programming": ("software_engineering", ["模块化设计", "自动化测试"])
            }
            
            assistance = {"suggestions": [], "knowledge": {}}
            if model_id in model_domain_map:
                domain, suggestions = model_domain_map[model_id]
                assistance["suggestions"] = suggestions
                # 查询该领域相关知识 | Query related knowledge in this domain
                assistance["knowledge"] = self.query_knowledge(domain, "基础原理")
            else:
                self.logger.warning(f"未知模型ID: {model_id} | Unknown model ID: {model_id}")
                assistance["suggestions"] = ["通用优化策略", "错误分析技术"]
                assistance["knowledge"] = self.query_knowledge("general", "问题解决方法")
            
            # 添加任务特定建议 | Add task-specific suggestions
            if "task_type" in task_context:
                task_specific = self.query_knowledge("task_optimization", task_context["task_type"])
                if task_specific.get("results"):
                    assistance["suggestions"].extend([item["concept"] for item in task_specific["results"][:2]])
            
            return {
                "target_model": model_id,
                "suggestions": assistance["suggestions"],
                "knowledge_support": assistance["knowledge"],
                "confidence": 0.85
            }
        except Exception as e:
            self.logger.error(f"模型辅助失败: {str(e)} | Model assistance failed: {str(e)}")
            return {"error": str(e)}

    def _query_external_api(self, domain: str, query: str) -> Dict[str, Any]:
        """查询外部API模型 | Query external API model
        参数:
            domain: 知识领域 | Knowledge domain
            query: 查询关键词 | Query keyword
        返回:
            API响应结果 | API response result
        """
        try:
            # 使用API模型连接器调用外部模型
            # Use API model connector to call external model
            result = api_model_connector.call_model(
                self.external_model_name,
                {
                    "domain": domain,
                    "query": query,
                    "max_results": 5
                }
            )
            
            # 处理API响应 | Process API response
            if result.get("status") == "success":
                # 标准化API结果格式 | Standardize API result format
                standardized_results = {
                    "domain": domain,
                    "results": []
                }
                
                for item in result.get("data", []):
                    standardized_results["results"].append({
                        "concept": item.get("concept", "未知概念"),
                        "description": item.get("description", ["无描述"]),
                        "related": item.get("related", [])
                    })
                
                return standardized_results
            else:
                self.logger.warning(f"外部API调用失败: {result.get('error', '未知错误')} | External API call failed: {result.get('error', 'Unknown error')}")
                # 回退到本地知识库查询 | Fall back to local knowledge base query
                return self.query_knowledge(domain, query)
        except Exception as e:
            self.logger.error(f"外部API调用异常: {str(e)} | External API call exception: {str(e)}")
            # 回退到本地知识库查询 | Fall back to local knowledge base query
            return self.query_knowledge(domain, query)

    def add_knowledge(self, concept: str, attributes: Dict[str, Any], relationships: List[Dict], domain: str) -> Dict[str, Any]:
        """添加新的知识概念 | Add new knowledge concept
        参数:
            concept: 概念名称 | Concept name
            attributes: 概念属性 | Concept attributes
            relationships: 概念关系 | Concept relationships
            domain: 知识领域 | Knowledge domain
        返回:
            操作结果 | Operation result
        """
        try:
            # 确保领域存在 | Ensure domain exists
            if domain not in self.knowledge_graph:
                self.knowledge_graph[domain] = {}
            
            # 添加或更新概念 | Add or update concept
            self.knowledge_graph[domain][concept] = {
                "description": attributes.get("description", []),
                "related": relationships,
                "source": attributes.get("source", "system"),
                "confidence": attributes.get("confidence", 0.8),
                "timestamp": attributes.get("timestamp", time.time())
            }
            
            # 更新知识嵌入 | Update knowledge embeddings
            if self.embedding_model:
                self.build_knowledge_embeddings()
            
            return {"status": "success", "message": f"知识概念 '{concept}' 添加成功"}
        except Exception as e:
            self.logger.error(f"添加知识失败: {str(e)} | Adding knowledge failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    def update_knowledge(self, concept: str, updates: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """更新现有知识概念 | Update existing knowledge concept
        参数:
            concept: 概念名称 | Concept name
            updates: 更新内容 | Update content
            domain: 知识领域 | Knowledge domain
        返回:
            操作结果 | Operation result
        """
        try:
            # 检查概念是否存在 | Check if concept exists
            if domain not in self.knowledge_graph or concept not in self.knowledge_graph[domain]:
                return {"status": "error", "message": f"概念 '{concept}' 在领域 '{domain}' 中不存在"}
            
            # 更新概念 | Update concept
            for key, value in updates.items():
                if key in self.knowledge_graph[domain][concept]:
                    self.knowledge_graph[domain][concept][key] = value
            
            # 更新时间戳 | Update timestamp
            self.knowledge_graph[domain][concept]["timestamp"] = time.time()
            
            # 更新知识嵌入 | Update knowledge embeddings
            if self.embedding_model:
                self.build_knowledge_embeddings()
            
            return {"status": "success", "message": f"知识概念 '{concept}' 更新成功"}
        except Exception as e:
            self.logger.error(f"更新知识失败: {str(e)} | Updating knowledge failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    def remove_knowledge(self, concept: str, domain: str) -> Dict[str, Any]:
        """删除知识概念 | Remove knowledge concept
        参数:
            concept: 概念名称 | Concept name
            domain: 知识领域 | Knowledge domain
        返回:
            操作结果 | Operation result
        """
        try:
            # 检查概念是否存在 | Check if concept exists
            if domain not in self.knowledge_graph or concept not in self.knowledge_graph[domain]:
                return {"status": "error", "message": f"概念 '{concept}' 在领域 '{domain}' 中不存在"}
            
            # 删除概念 | Delete concept
            del self.knowledge_graph[domain][concept]
            
            # 更新知识嵌入 | Update knowledge embeddings
            if self.embedding_model:
                self.build_knowledge_embeddings()
            
            return {"status": "success", "message": f"知识概念 '{concept}' 删除成功"}
        except Exception as e:
            self.logger.error(f"删除知识失败: {str(e)} | Removing knowledge failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    def import_knowledge(self, file_path: str, domain: str) -> Dict[str, Any]:
        """从文件导入知识 | Import knowledge from file
        参数:
            file_path: 文件路径 | File path
            domain: 目标知识领域 | Target knowledge domain
        返回:
            操作结果 | Operation result
        """
        try:
            # 检查文件是否存在 | Check if file exists
            if not os.path.exists(file_path):
                return {"status": "error", "message": f"文件 '{file_path}' 不存在"}
            
            # 根据文件扩展名选择解析方法 | Choose parsing method based on file extension
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            if ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    knowledge_data = json.load(f)
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    # 简单文本解析，假设每行一个概念 | Simple text parsing, assuming one concept per line
                    lines = f.readlines()
                    knowledge_data = {}
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # 简单分割概念和描述 | Simple split of concept and description
                            if ':' in line:
                                concept, desc = line.split(':', 1)
                                knowledge_data[concept.strip()] = {
                                    "description": [desc.strip()],
                                    "related": [],
                                    "source": file_path
                                }
                            else:
                                knowledge_data[line] = {
                                    "description": ["无描述"],
                                    "related": [],
                                    "source": file_path
                                }
            elif ext == '.pdf' and PDF_SUPPORT:
                # 解析PDF文件 | Parse PDF file
                content = self._parse_pdf_file(file_path)
                knowledge_data = self._parse_text_content_to_knowledge(content, file_path)
            elif ext == '.docx' and DOCX_SUPPORT:
                # 解析DOCX文件 | Parse DOCX file
                content = self._parse_docx_file(file_path)
                knowledge_data = self._parse_text_content_to_knowledge(content, file_path)
            else:
                if ext in ['.pdf', '.docx'] and not PDF_SUPPORT and not DOCX_SUPPORT:
                    return {"status": "error", "message": f"PDF/DOCX支持库未安装，请安装PyPDF2和python-docx | PDF/DOCX support libraries not installed, please install PyPDF2 and python-docx"}
                return {"status": "error", "message": f"不支持的文件格式: {ext}"}
            
            # 导入知识 | Import knowledge
            if domain not in self.knowledge_graph:
                self.knowledge_graph[domain] = {}
            
            self.knowledge_graph[domain].update(knowledge_data)
            
            # 更新知识嵌入 | Update knowledge embeddings
            if self.embedding_model:
                self.build_knowledge_embeddings()
            
            return {"success": True, "message": f"成功从文件 '{file_path}' 导入知识到领域 '{domain}'", "domain": domain, "content_length": len(knowledge_data)}
        except Exception as e:
            self.logger.error(f"导入知识失败: {str(e)} | Importing knowledge failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    def export_knowledge(self, domain: str, format: str = "json") -> Dict[str, Any]:
        """导出知识库数据 | Export knowledge base data
        参数:
            domain: 知识领域 | Knowledge domain
            format: 导出格式 (json/csv/xml) | Export format (json/csv/xml)
        返回:
            格式化知识数据 | Formatted knowledge data
        """
        try:
            if domain not in self.knowledge_graph:
                return {"error": f"未知知识领域: {domain} | Unknown knowledge domain: {domain}"}
            
            if format == "json":
                return {
                    "domain": domain,
                    "concepts": self.knowledge_graph[domain],
                    "export_format": "json",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                }
            elif format == "csv":
                # 简化的CSV导出实现 | Simplified CSV export implementation
                csv_data = "concept,description,related,source,confidence\n"
                for concept, details in self.knowledge_graph[domain].items():
                    desc = ";".join(details.get("description", []))
                    rel = ";".join([f"{r['target']}({r['type']})" for r in details.get("related", [])])
                    source = details.get("source", "unknown")
                    confidence = str(details.get("confidence", 0.0))
                    csv_data += f'"{concept}","{desc}","{rel}","{source}",{confidence}\n'
                return {
                    "domain": domain,
                    "data": csv_data,
                    "export_format": "csv",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                }
            else:
                return {"error": f"不支持的导出格式: {format} | Unsupported export format: {format}"}
        except Exception as e:
            self.logger.error(f"导出知识失败: {str(e)} | Exporting knowledge failed: {str(e)}")
            return {"error": str(e)}

    def integrate_knowledge(self, knowledge_update: Any) -> bool:
        """整合新知识到知识库 | Integrate new knowledge into knowledge base
        参数:
            knowledge_update: 知识更新数据 | Knowledge update data
        返回:
            整合是否成功 | Whether integration was successful
        """
        try:
            # 这里应该实现具体的知识整合逻辑
            # Concrete knowledge integration logic should be implemented here
            # 示例实现：从更新中提取概念和关系并添加到知识库
            # Example implementation: Extract concepts and relationships from update and add to knowledge base
            if hasattr(knowledge_update, 'source_model') and hasattr(knowledge_update, 'content'):
                source_model = knowledge_update.source_model
                content = knowledge_update.content
                
                # 这里应该有更复杂的知识整合逻辑
                # More complex knowledge integration logic should be here
                # 简单示例：如果更新包含概念列表，则添加它们
                # Simple example: If update contains concept list, add them
                if isinstance(content, dict) and 'concepts' in content:
                    domain = content.get('domain', 'general')
                    for concept, details in content['concepts'].items():
                        self.add_knowledge(
                            concept,
                            {
                                'description': details.get('description', []),
                                'source': source_model,
                                'confidence': details.get('confidence', 0.5)
                            },
                            details.get('relationships', []),
                            domain
                        )
            
            return True
        except Exception as e:
            self.logger.error(f"知识整合失败: {str(e)} | Knowledge integration failed: {str(e)}")
            return False

    def get_knowledge_summary(self, domain: str = None) -> Dict[str, Any]:
        """获取知识库摘要 | Get knowledge base summary
        参数:
            domain: 特定领域 (可选) | Specific domain (optional)
        返回:
            知识库统计信息 | Knowledge base statistics
        """
        summary = {
            "total_domains": 0,
            "total_concepts": 0,
            "domains": {}
        }
        
        # 如果指定了领域，只返回该领域的摘要 | If domain is specified, only return summary for that domain
        if domain:
            if domain in self.knowledge_graph:
                summary["total_domains"] = 1
                summary["total_concepts"] = len(self.knowledge_graph[domain])
                summary["domains"][domain] = {
                    "concept_count": len(self.knowledge_graph[domain]),
                    "embedding_available": domain in self.knowledge_embeddings if self.embedding_model else False
                }
        else:
            # 返回所有领域的摘要 | Return summary for all domains
            summary["total_domains"] = len(self.knowledge_graph)
            for domain_name, concepts in self.knowledge_graph.items():
                concept_count = len(concepts)
                summary["total_concepts"] += concept_count
                summary["domains"][domain_name] = {
                    "concept_count": concept_count,
                    "embedding_available": domain_name in self.knowledge_embeddings if self.embedding_model else False
                }
        
        summary["embedding_model"] = "sentence-transformers" if self.embedding_model else "none"
        summary["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        return summary

    def evaluate_knowledge_confidence(self, domain: str = None) -> Dict[str, Any]:
        """评估知识库置信度 | Evaluate knowledge base confidence
        参数:
            domain: 特定领域 (可选) | Specific domain (optional)
        返回:
            置信度评估结果 | Confidence evaluation results
        """
        try:
            evaluation = {
                "total_confidence": 0.0,
                "domain_confidences": {},
                "low_confidence_concepts": []
            }
            
            # 如果指定了领域，只评估该领域 | If domain is specified, only evaluate that domain
            domains_to_evaluate = [domain] if domain else self.knowledge_graph.keys()
            
            total_concepts = 0
            total_confidence = 0.0
            
            for domain_name in domains_to_evaluate:
                if domain_name not in self.knowledge_graph:
                    continue
                
                domain_concepts = 0
                domain_confidence = 0.0
                low_confidence = []
                
                for concept, details in self.knowledge_graph[domain_name].items():
                    # 确保details是字典类型 | Ensure details is a dictionary
                    if not isinstance(details, dict):
                        continue
                        
                    confidence = details.get("confidence", 0.0)
                    domain_confidence += confidence
                    domain_concepts += 1
                    total_confidence += confidence
                    total_concepts += 1
                    
                    # 记录低置信度概念 | Record low confidence concepts
                    if confidence < 0.5:
                        low_confidence.append({
                            "concept": concept,
                            "confidence": confidence
                        })
                
                # 计算领域平均置信度 | Calculate domain average confidence
                if domain_concepts > 0:
                    domain_avg_confidence = domain_confidence / domain_concepts
                    evaluation["domain_confidences"][domain_name] = {
                        "average_confidence": domain_avg_confidence,
                        "concept_count": domain_concepts
                    }
                    
                    # 添加该领域的低置信度概念 | Add low confidence concepts from this domain
                    evaluation["low_confidence_concepts"].extend(low_confidence)
            
            # 计算总体平均置信度 | Calculate overall average confidence
            if total_concepts > 0:
                evaluation["total_confidence"] = total_confidence / total_concepts
            
            return evaluation
        except Exception as e:
            self.logger.error(f"知识置信度评估失败: {str(e)} | Knowledge confidence evaluation failed: {str(e)}")
            return {"error": str(e)}

    def optimize_knowledge_structure(self) -> Dict[str, Any]:
        """优化知识库结构 | Optimize knowledge base structure
        返回:
            优化结果 | Optimization results
        """
        try:
            optimization_results = {
                "duplicates_removed": 0,
                "relationships_optimized": 0,
                "clusters_created": 0
            }
            
            # 1. 检测并移除重复概念 | 1. Detect and remove duplicate concepts
            duplicates_removed = self._remove_duplicate_concepts()
            optimization_results["duplicates_removed"] = duplicates_removed
            
            # 2. 优化概念关系 | 2. Optimize concept relationships
            relationships_optimized = self._optimize_relationships()
            optimization_results["relationships_optimized"] = relationships_optimized
            
            # 3. 对概念进行聚类 | 3. Cluster concepts
            clusters_created = self._cluster_concepts()
            optimization_results["clusters_created"] = clusters_created
            
            # 更新知识嵌入 | Update knowledge embeddings
            if self.embedding_model:
                self.build_knowledge_embeddings()
            
            return {
                "status": "success",
                "optimization_results": optimization_results,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
        except Exception as e:
            self.logger.error(f"知识库结构优化失败: {str(e)} | Knowledge base structure optimization failed: {str(e)}")
            return {"error": str(e)}

    def _remove_duplicate_concepts(self) -> int:
        """移除重复概念 | Remove duplicate concepts
        返回:
            移除的重复概念数量 | Number of duplicate concepts removed
        """
        duplicates_removed = 0
        
        # 这里应该实现具体的重复概念检测和移除逻辑
        # Concrete duplicate concept detection and removal logic should be implemented here
        # 简单示例：检测相同名称的概念并保留置信度高的 | Simple example: Detect concepts with same name and keep the one with higher confidence
        concept_names = {}
        
        for domain, concepts in self.knowledge_graph.items():
            concepts_to_remove = []
            
            for concept_name, details in concepts.items():
                # 使用概念名称的标准化形式进行比较 | Use standardized form of concept name for comparison
                normalized_name = concept_name.lower().strip()
                
                if normalized_name in concept_names:
                    # 找到重复概念 | Found duplicate concept
                    existing_domain, existing_details = concept_names[normalized_name]
                    
                    # 比较置信度，移除置信度低的 | Compare confidence, remove the one with lower confidence
                    if details.get("confidence", 0.0) > existing_details.get("confidence", 0.0):
                        # 移除已存在的概念，保留当前概念 | Remove existing concept, keep current concept
                        if existing_domain in self.knowledge_graph and existing_details in self.knowledge_graph[existing_domain]:
                            self.knowledge_graph[existing_domain].pop(concept_name)
                            duplicates_removed += 1
                        # 更新概念记录为当前概念 | Update concept record to current concept
                        concept_names[normalized_name] = (domain, details)
                    else:
                        # 移除当前概念，保留已存在的概念 | Remove current concept, keep existing concept
                        concepts_to_remove.append(concept_name)
                        duplicates_removed += 1
                else:
                    # 记录新概念 | Record new concept
                    concept_names[normalized_name] = (domain, details)
            
            # 移除当前领域中的重复概念 | Remove duplicate concepts in current domain
            for concept_name in concepts_to_remove:
                if concept_name in self.knowledge_graph[domain]:
                    self.knowledge_graph[domain].pop(concept_name)
        
        return duplicates_removed

    def _optimize_relationships(self) -> int:
        """优化概念关系 | Optimize concept relationships
        返回:
            优化的关系数量 | Number of relationships optimized
        """
        relationships_optimized = 0
        
        # 这里应该实现具体的关系优化逻辑
        # Concrete relationship optimization logic should be implemented here
        # 简单示例：检测并修复无效关系 | Simple example: Detect and fix invalid relationships
        for domain, concepts in self.knowledge_graph.items():
            for concept_name, details in concepts.items():
                if "related" in details:
                    original_count = len(details["related"])
                    # 过滤掉目标不存在的关系 | Filter out relationships where target doesn't exist
                    valid_relationships = []
                    for rel in details["related"]:
                        target = rel.get("target")
                        if target:
                            # 检查目标概念是否存在于任何领域 | Check if target concept exists in any domain
                            target_exists = False
                            for check_domain, check_concepts in self.knowledge_graph.items():
                                if target in check_concepts:
                                    target_exists = True
                                    break
                            if target_exists:
                                valid_relationships.append(rel)
                    # 更新关系列表 | Update relationship list
                    details["related"] = valid_relationships
                    relationships_optimized += original_count - len(valid_relationships)
        
        return relationships_optimized

    def _cluster_concepts(self) -> int:
        """对概念进行聚类 | Cluster concepts
        返回:
            创建的聚类数量 | Number of clusters created
        """
        clusters_created = 0
        
        # 这里应该实现具体的概念聚类逻辑
        # Concrete concept clustering logic should be implemented here
        # 简单示例：基于领域进行基本聚类 | Simple example: Basic clustering based on domains
        # 注意：这是一个简化实现，实际聚类应该基于语义相似性 | Note: This is a simplified implementation, actual clustering should be based on semantic similarity
        
        # 初始化聚类映射 | Initialize cluster mapping
        self.concept_clusters = {}
        
        for domain, concepts in self.knowledge_graph.items():
            # 使用领域作为基本聚类 | Use domain as basic cluster
            for concept_name in concepts.keys():
                self.concept_clusters[concept_name] = domain
            
            if len(concepts) > 0:
                clusters_created += 1
        
        # 如果有嵌入模型，可以实现更高级的语义聚类 | If embedding model is available, more advanced semantic clustering can be implemented
        if self.embedding_model:
            # 这里应该实现基于嵌入的语义聚类 | Semantic clustering based on embeddings should be implemented here
            # 为简化实现，这里不展开 | For simplicity, not expanded here
            pass
        
        return clusters_created

    def get_concept_connections(self, concept: str, depth: int = 1) -> Dict[str, Any]:
        """获取概念的关联网络 | Get concept connection network
        参数:
            concept: 中心概念 | Central concept
            depth: 搜索深度 | Search depth
        返回:
            概念关联网络 | Concept connection network
        """
        try:
            connections = {
                "central_concept": concept,
                "depth": depth,
                "nodes": [concept],
                "edges": [],
                "found": False
            }
            
            # 查找概念所在的领域 | Find domain where concept is located
            concept_domain = None
            for domain, concepts in self.knowledge_graph.items():
                if concept in concepts:
                    concept_domain = domain
                    connections["found"] = True
                    break
            
            if not connections["found"]:
                return connections
            
            # 根据深度获取关联概念 | Get connected concepts based on depth
            visited = {concept}
            to_visit = [(concept, 0)]
            
            while to_visit:
                current_concept, current_depth = to_visit.pop(0)
                
                # 如果达到最大深度，停止搜索 | Stop searching if maximum depth is reached
                if current_depth >= depth:
                    continue
                
                # 查找当前概念在所有领域中的出现 | Find occurrences of current concept in all domains
                for domain, concepts in self.knowledge_graph.items():
                    if current_concept in concepts:
                        # 获取当前概念的关系 | Get relationships of current concept
                        relationships = concepts[current_concept].get("related", [])
                        
                        for rel in relationships:
                            target_concept = rel.get("target")
                            rel_type = rel.get("type", "related")
                            
                            if target_concept and target_concept not in visited:
                                visited.add(target_concept)
                                connections["nodes"].append(target_concept)
                                connections["edges"].append({
                                    "source": current_concept,
                                    "target": target_concept,
                                    "type": rel_type
                                })
                                
                                # 将目标概念加入待访问列表 | Add target concept to visit list
                                to_visit.append((target_concept, current_depth + 1))
            
            return connections
        except Exception as e:
            self.logger.error(f"获取概念关联失败: {str(e)} | Getting concept connections failed: {str(e)}")
            return {"error": str(e)}

    def generate_knowledge_visualization(self, domain: str = None) -> Dict[str, Any]:
        """生成知识库可视化数据 | Generate knowledge base visualization data
        参数:
            domain: 特定领域 (可选) | Specific domain (optional)
        返回:
            可视化数据 | Visualization data
        """
        try:
            visualization = {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "domain": domain or "all"
                }
            }
            
            # 确定要可视化的领域 | Determine domains to visualize
            domains_to_visualize = [domain] if domain else self.knowledge_graph.keys()
            
            # 为每个领域创建颜色映射 | Create color mapping for each domain
            colors = [
                "#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#FF33F3", "#33FFF3",
                "#FF8C33", "#8C33FF", "#33FF8C", "#FF3333", "#33FF33", "#3333FF"
            ]
            domain_color_map = {}
            color_index = 0
            
            for domain_name in domains_to_visualize:
                if domain_name not in self.knowledge_graph:
                    continue
                
                # 为领域分配颜色 | Assign color to domain
                if domain_name not in domain_color_map:
                    domain_color_map[domain_name] = colors[color_index % len(colors)]
                    color_index += 1
                
                domain_color = domain_color_map[domain_name]
                
                # 添加节点 | Add nodes
                for concept_name, details in self.knowledge_graph[domain_name].items():
                    # 计算节点大小（基于概念的关系数量和置信度）| Calculate node size (based on concept's relationship count and confidence)
                    relationships_count = len(details.get("related", []))
                    confidence = details.get("confidence", 0.5)
                    node_size = 10 + (relationships_count * 5) * confidence
                    
                    # 添加节点到可视化数据 | Add node to visualization data
                    visualization["nodes"].append({
                        "id": concept_name,
                        "label": concept_name,
                        "size": node_size,
                        "color": domain_color,
                        "domain": domain_name,
                        "confidence": confidence
                    })
                    
                    # 添加边 | Add edges
                    for rel in details.get("related", []):
                        target_concept = rel.get("target")
                        rel_type = rel.get("type", "related")
                        rel_strength = rel.get("strength", 0.5)
                        
                        if target_concept:
                            visualization["edges"].append({
                                "source": concept_name,
                                "target": target_concept,
                                "type": rel_type,
                                "strength": rel_strength,
                                "color": domain_color
                            })
            
            return visualization
        except Exception as e:
            self.logger.error(f"生成知识库可视化数据失败: {str(e)} | Generating knowledge base visualization data failed: {str(e)}")
            return {"error": str(e)}

    def get_learning_progress(self) -> Dict[str, Any]:
        """获取学习进度信息 | Get learning progress information
        返回:
            学习进度数据 | Learning progress data
        """
        try:
            # 这里应该实现具体的学习进度计算逻辑
            # Concrete learning progress calculation logic should be implemented here
            # 简单示例：基于知识库大小和最近更新 | Simple example: Based on knowledge base size and recent updates
            progress = {
                "total_concepts": 0,
                "new_concepts_week": 0,
                "updated_concepts_week": 0,
                "domains_covered": len(self.knowledge_graph),
                "confidence_score": 0.0
            }
            
            # 计算一周前的时间戳 | Calculate timestamp for one week ago
            week_ago = time.time() - (7 * 24 * 60 * 60)
            
            total_confidence = 0.0
            
            for domain, concepts in self.knowledge_graph.items():
                progress["total_concepts"] += len(concepts)
                
                for concept_name, details in concepts.items():
                    # 确保details是字典类型 | Ensure details is a dictionary
                    if not isinstance(details, dict):
                        continue
                        
                    # 计算置信度 | Calculate confidence
                    confidence = details.get("confidence", 0.0)
                    total_confidence += confidence
                    
                    # 检查是否是本周新增或更新的概念 | Check if concept was added or updated this week
                    timestamp = details.get("timestamp", 0)
                    # 确保timestamp是数字类型 | Ensure timestamp is numeric type
                    if isinstance(timestamp, (int, float)) and timestamp > week_ago:
                        if "source" in details and details["source"] != "initial":
                            # 假设source不是initial的概念是新增的 | Assume concepts with source not 'initial' are new
                            progress["new_concepts_week"] += 1
                        else:
                            # 否则认为是更新的概念 | Otherwise consider as updated concept
                            progress["updated_concepts_week"] += 1
            
            # 计算平均置信度 | Calculate average confidence
            if progress["total_concepts"] > 0:
                progress["confidence_score"] = total_confidence / progress["total_concepts"]
            
            # 添加学习趋势指标 | Add learning trend indicators
            progress["learning_rate"] = progress["new_concepts_week"] / 7  # 每天新增概念数 | New concepts per day
            progress["update_rate"] = progress["updated_concepts_week"] / 7  # 每天更新概念数 | Updated concepts per day
            progress["last_calculated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            return progress
        except Exception as e:
            self.logger.error(f"获取学习进度失败: {str(e)} | Getting learning progress failed: {str(e)}")
            return {"error": str(e)}

    def adapt_to_new_knowledge(self, new_information: Dict[str, Any]) -> Dict[str, Any]:
        """适应新知识 | Adapt to new knowledge
        参数:
            new_information: 新信息 | New information
        返回:
            适应结果 | Adaptation results
        """
        try:
            # 这里应该实现具体的知识适应逻辑
            # Concrete knowledge adaptation logic should be implemented here
            # 简单示例：根据新信息更新知识库 | Simple example: Update knowledge base based on new information
            adaptation_results = {
                "concepts_added": 0,
                "concepts_updated": 0,
                "confidence_adjustments": 0,
                "domain_updates": []
            }
            
            # 处理新信息 | Process new information
            if isinstance(new_information, dict):
                # 检查是否包含领域信息 | Check if domain information is included
                domain = new_information.get("domain", "general")
                
                # 检查是否包含概念列表 | Check if concept list is included
                if "concepts" in new_information:
                    for concept_name, concept_data in new_information["concepts"].items():
                        # 检查概念是否已存在 | Check if concept exists
                        concept_exists = False
                        for check_domain, check_concepts in self.knowledge_graph.items():
                            if concept_name in check_concepts:
                                concept_exists = True
                                # 更新现有概念 | Update existing concept
                                update_result = self.update_knowledge(concept_name, concept_data, check_domain)
                                if update_result.get("status") == "success":
                                    adaptation_results["concepts_updated"] += 1
                                    if check_domain not in adaptation_results["domain_updates"]:
                                        adaptation_results["domain_updates"].append(check_domain)
                                break
                        
                        if not concept_exists:
                            # 添加新概念 | Add new concept
                            add_result = self.add_knowledge(
                                concept_name,
                                concept_data,
                                concept_data.get("relationships", []),
                                domain
                            )
                            if add_result.get("status") == "success":
                                adaptation_results["concepts_added"] += 1
                                if domain not in adaptation_results["domain_updates"]:
                                    adaptation_results["domain_updates"].append(domain)
                
                # 处理置信度调整 | Process confidence adjustments
                if "confidence_adjustments" in new_information:
                    for concept_name, new_confidence in new_information["confidence_adjustments"].items():
                        # 查找概念并调整置信度 | Find concept and adjust confidence
                        for check_domain, check_concepts in self.knowledge_graph.items():
                            if concept_name in check_concepts:
                                check_concepts[concept_name]["confidence"] = new_confidence
                                adaptation_results["confidence_adjustments"] += 1
                                if check_domain not in adaptation_results["domain_updates"]:
                                    adaptation_results["domain_updates"].append(check_domain)
                                break
            
            # 更新知识嵌入 | Update knowledge embeddings
            if self.embedding_model:
                self.build_knowledge_embeddings()
            
            return adaptation_results
        except Exception as e:
            self.logger.error(f"适应新知识失败: {str(e)} | Adapting to new knowledge failed: {str(e)}")
            return {"error": str(e)}

    def assist_training(self, model_id: str, training_data_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """辅助模型训练 - 提供相关知识支持
        Assist model training - Provide relevant knowledge support
        
        参数:
            model_id: 需要辅助的模型ID | Model ID that needs assistance
            training_data_metadata: 训练数据元数据 | Training data metadata
            
        返回:
            知识上下文和支持信息 | Knowledge context and support information
        """
        try:
            # 模型类型到知识领域的映射 | Model type to knowledge domain mapping
            model_domain_map = {
                "manager": ["management", "task_optimization", "resource_allocation"],
                "language": ["linguistics", "natural_language_processing", "sentiment_analysis"],
                "audio": ["acoustics", "signal_processing", "audio_analysis"],
                "vision": ["computer_vision", "image_processing", "pattern_recognition"],
                "video": ["video_processing", "motion_analysis", "temporal_modeling"],
                "spatial": ["spatial_reasoning", "3d_modeling", "geometry"],
                "sensor": ["sensor_fusion", "data_processing", "signal_analysis"],
                "computer": ["computer_science", "distributed_systems", "operating_systems"],
                "motion": ["robotics", "kinematics", "control_systems"],
                "knowledge": ["knowledge_engineering", "ontology", "semantic_web"],
                "programming": ["software_engineering", "algorithms", "data_structures"]
            }
            
            # 获取模型对应的知识领域 | Get knowledge domains for the model
            domains = model_domain_map.get(model_id, ["general", "machine_learning"])
            
            # 基于训练数据元数据获取更具体的知识 | Get more specific knowledge based on training data metadata
            specific_knowledge = {}
            if training_data_metadata:
                # 从元数据中提取关键词进行知识搜索 | Extract keywords from metadata for knowledge search
                keywords = []
                if "task_type" in training_data_metadata:
                    keywords.append(training_data_metadata["task_type"])
                if "data_type" in training_data_metadata:
                    keywords.append(training_data_metadata["data_type"])
                if "domain" in training_data_metadata:
                    keywords.append(training_data_metadata["domain"])
                
                # 搜索相关知识 | Search for relevant knowledge
                for keyword in keywords:
                    for domain in domains:
                        search_results = self.query_knowledge(domain, keyword)
                        if search_results.get("results"):
                            if domain not in specific_knowledge:
                                specific_knowledge[domain] = []
                            specific_knowledge[domain].extend(search_results["results"])
            
            # 获取通用训练知识 | Get general training knowledge
            general_knowledge = {}
            for domain in domains:
                general_results = self.query_knowledge(domain, "training optimization")
                if general_results.get("results"):
                    general_knowledge[domain] = general_results["results"]
            
            # 构建训练建议 | Build training suggestions
            training_suggestions = self._generate_training_suggestions(model_id, training_data_metadata)
            
            return {
                "model_id": model_id,
                "specific_knowledge": specific_knowledge,
                "general_knowledge": general_knowledge,
                "training_suggestions": training_suggestions,
                "confidence": 0.85,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"辅助训练失败: {str(e)} | Training assistance failed: {str(e)}")
            return {"error": str(e), "model_id": model_id}

    def _generate_training_suggestions(self, model_id: str, metadata: Dict[str, Any] = None) -> List[str]:
        """生成训练建议 | Generate training suggestions"""
        suggestions = []
        
        # 基于模型类型的通用建议 | General suggestions based on model type
        model_suggestions = {
            "manager": [
                "优化任务分配策略",
                "改进资源调度算法",
                "增强模型协同机制"
            ],
            "language": [
                "增加多语言训练数据",
                "优化情感分析模型",
                "改进上下文理解能力"
            ],
            "audio": [
                "增强噪声抑制能力",
                "改进音频特征提取",
                "优化语音识别精度"
            ],
            "vision": [
                "增加图像增强技术",
                "改进目标检测算法",
                "优化图像分类模型"
            ],
            "video": [
                "增强时序建模能力",
                "改进运动估计技术",
                "优化视频压缩算法"
            ],
            "spatial": [
                "提高3D重建精度",
                "改进SLAM算法",
                "优化空间感知能力"
            ],
            "sensor": [
                "增强多传感器融合",
                "改进数据滤波算法",
                "优化信号处理流程"
            ],
            "computer": [
                "优化系统资源管理",
                "改进任务调度策略",
                "增强容错机制"
            ],
            "motion": [
                "改进运动规划算法",
                "优化动力学模型",
                "增强实时控制能力"
            ],
            "knowledge": [
                "扩展知识覆盖范围",
                "改进知识推理能力",
                "优化知识检索效率"
            ],
            "programming": [
                "增强代码生成能力",
                "改进算法优化技术",
                "优化软件架构设计"
            ]
        }
        
        # 添加通用建议 | Add general suggestions
        suggestions.extend(model_suggestions.get(model_id, [
            "调整学习率参数",
            "增加训练轮次",
            "优化批次大小",
            "使用数据增强技术"
        ]))
        
        # 基于元数据的特定建议 | Specific suggestions based on metadata
        if metadata:
            if metadata.get("data_size", 0) < 1000:
                suggestions.append("增加训练数据量以提高泛化能力")
            if metadata.get("complexity", "low") == "high":
                suggestions.append("使用更复杂的模型架构")
            if metadata.get("real_time", False):
                suggestions.append("优化推理速度以满足实时要求")
        
        return suggestions

    def explain_knowledge(self, concept: str) -> Dict[str, Any]:
        """解释知识概念 | Explain knowledge concept
        参数:
            concept: 要解释的概念 | Concept to explain
        返回:
            概念解释 | Concept explanation
        """
        try:
            explanation = {
                "concept": concept,
                "found": False,
                "explanation": "",
                "sources": [],
                "related_concepts": []
            }
            
            # 查找概念 | Find concept
            for domain, concepts in self.knowledge_graph.items():
                if concept in concepts:
                    concept_data = concepts[concept]
                    explanation["found"] = True
                    explanation["domain"] = domain
                    
                    # 构建解释文本 | Build explanation text
                    description = concept_data.get("description", [])
                    if description:
                        explanation["explanation"] = " ".join(description)
                    
                    # 添加来源 | Add sources
                    source = concept_data.get("source", "unknown")
                    if source not in explanation["sources"]:
                        explanation["sources"].append(source)
                    
                    # 添加相关概念 | Add related concepts
                    for rel in concept_data.get("related", []):
                        target = rel.get("target")
                        if target:
                            explanation["related_concepts"].append({
                                "name": target,
                                "relationship": rel.get("type", "related")
                            })
                    
                    break
            
            # 如果找不到概念，尝试语义搜索 | If concept not found, try semantic search
            if not explanation["found"]:
                search_results = self.semantic_search(concept, top_k=3)
                if search_results:
                    explanation["similar_concepts"] = search_results
            
            return explanation
        except Exception as e:
            self.logger.error(f"解释知识概念失败: {str(e)} | Explaining knowledge concept failed: {str(e)}")
            return {"error": str(e)}

    def get_knowledge_deficiency(self, domain: str = None) -> Dict[str, Any]:
        """获取知识库缺陷 | Get knowledge base deficiencies
        参数:
            domain: 特定领域 (可选) | Specific domain (optional)
        返回:
            知识库缺陷分析 | Knowledge base deficiency analysis
        """
        try:
            deficiencies = {
                "low_confidence_concepts": [],
                "missing_relationships": [],
                "underrepresented_domains": [],
                "recommendations": []
            }
            
            # 确定要分析的领域 | Determine domains to analyze
            domains_to_analyze = [domain] if domain else self.knowledge_graph.keys()
            
            # 分析低置信度概念 | Analyze low confidence concepts
            for domain_name in domains_to_analyze:
                if domain_name not in self.knowledge_graph:
                    continue
                
                for concept_name, details in self.knowledge_graph[domain_name].items():
                    confidence = details.get("confidence", 0.0)
                    if confidence < 0.5:
                        deficiencies["low_confidence_concepts"].append({
                            "concept": concept_name,
                            "domain": domain_name,
                            "confidence": confidence
                        })
                    
                    # 分析缺失的关系 | Analyze missing relationships
                    relationships = details.get("related", [])
                    if len(relationships) == 0:
                        deficiencies["missing_relationships"].append({
                            "concept": concept_name,
                            "domain": domain_name
                        })
            
            # 分析代表性不足的领域 | Analyze underrepresented domains
            for domain_name in domains_to_analyze:
                if domain_name not in self.knowledge_graph:
                    continue
                
                concept_count = len(self.knowledge_graph[domain_name])
                if concept_count < 10:  # 假设少于10个概念的领域是代表性不足的 | Assume domains with fewer than 10 concepts are underrepresented
                    deficiencies["underrepresented_domains"].append({
                        "domain": domain_name,
                        "concept_count": concept_count,
                        "target_count": 50  # 目标概念数量 | Target concept count
                    })
            
            # 生成建议 | Generate recommendations
            if deficiencies["low_confidence_concepts"]:
                deficiencies["recommendations"].append("需要验证和更新低置信度概念")
            
            if deficiencies["missing_relationships"]:
                deficiencies["recommendations"].append("需要添加概念间的关系连接")
            
            if deficiencies["underrepresented_domains"]:
                deficiencies["recommendations"].append("需要扩展代表性不足的领域")
            
            # 如果有嵌入模型，基于语义分析生成更具体的建议 | If embedding model is available, generate more specific recommendations based on semantic analysis
            if self.embedding_model:
                deficiencies["recommendations"].append("可以使用语义嵌入来改进知识组织和检索")
            
            return deficiencies
        except Exception as e:
            self.logger.error(f"获取知识库缺陷失败: {str(e)} | Getting knowledge base deficiencies failed: {str(e)}")
            return {"error": str(e)}

    def train_knowledge_model(self, training_data: List[Dict[str, Any]], epochs: int = 5) -> Dict[str, Any]:
        """训练知识库模型 | Train knowledge base model
        参数:
            training_data: 训练数据 | Training data
            epochs: 训练轮次 | Training epochs
        返回:
            训练结果 | Training results
        """
        try:
            training_results = {
                "epochs": epochs,
                "concepts_trained": 0,
                "improvement_metrics": {},
                "status": "success"
            }
            
            # 这里应该实现具体的模型训练逻辑
            # Concrete model training logic should be implemented here
            # 简单示例：使用训练数据更新知识库 | Simple example: Update knowledge base using training data
            concepts_trained = 0
            
            for _ in range(epochs):
                for data_item in training_data:
                    concept = data_item.get("concept")
                    domain = data_item.get("domain", "general")
                    attributes = data_item.get("attributes", {})
                    relationships = data_item.get("relationships", [])
                    
                    if concept:
                        # 检查概念是否已存在 | Check if concept exists
                        concept_exists = False
                        for check_domain, check_concepts in self.knowledge_graph.items():
                            if concept in check_concepts:
                                concept_exists = True
                                # 更新现有概念 | Update existing concept
                                self.update_knowledge(concept, attributes, check_domain)
                                break
                        
                        if not concept_exists:
                            # 添加新概念 | Add new concept
                            self.add_knowledge(concept, attributes, relationships, domain)
                        
                        concepts_trained += 1
            
            # 更新知识嵌入 | Update knowledge embeddings
            if self.embedding_model:
                self.build_knowledge_embeddings()
            
            training_results["concepts_trained"] = concepts_trained
            training_results["improvement_metrics"] = {
                "confidence_improvement": 0.1,  # 示例值 | Example value
                "coverage_improvement": 0.05   # 示例值 | Example value
            }
            
            return training_results
        except Exception as e:
            self.logger.error(f"训练知识库模型失败: {str(e)} | Training knowledge base model failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    def validate_knowledge(self, validation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """验证知识库 | Validate knowledge base
        参数:
            validation_data: 验证数据 | Validation data
        返回:
            验证结果 | Validation results
        """
        try:
            validation_results = {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "accuracy": 0.0,
                "error_categories": {}
            }
            
            # 这里应该实现具体的知识库验证逻辑
            # Concrete knowledge base validation logic should be implemented here
            # 简单示例：基于验证数据检查知识库的准确性 | Simple example: Check knowledge base accuracy based on validation data
            total_tests = 0
            passed_tests = 0
            error_categories = defaultdict(int)
            
            for test_item in validation_data:
                total_tests += 1
                concept = test_item.get("concept")
                domain = test_item.get("domain")
                expected_attributes = test_item.get("attributes", {})
                
                if concept and domain:
                    # 在指定领域中查找概念 | Find concept in specified domain
                    if domain in self.knowledge_graph and concept in self.knowledge_graph[domain]:
                        concept_data = self.knowledge_graph[domain][concept]
                        test_passed = True
                        
                        # 检查属性 | Check attributes
                        for attr_name, expected_value in expected_attributes.items():
                            if attr_name in concept_data:
                                # 根据属性类型进行比较 | Compare based on attribute type
                                if isinstance(expected_value, list):
                                    # 对于列表类型，检查是否包含所有期望的值 | For list type, check if all expected values are included
                                    if not all(item in concept_data[attr_name] for item in expected_value):
                                        test_passed = False
                                        error_categories["incorrect_attribute_value"] += 1
                                        break
                                else:
                                    # 对于其他类型，直接比较 | For other types, compare directly
                                    if concept_data[attr_name] != expected_value:
                                        test_passed = False
                                        error_categories["incorrect_attribute_value"] += 1
                                        break
                            else:
                                test_passed = False
                                error_categories["missing_attribute"] += 1
                                break
                        
                        if test_passed:
                            passed_tests += 1
                    else:
                        # 概念不存在 | Concept does not exist
                        test_passed = False
                        error_categories["concept_not_found"] += 1
                else:
                    # 测试项缺少必要信息 | Test item missing necessary information
                    test_passed = False
                    error_categories["invalid_test_item"] += 1
            
            # 计算准确性 | Calculate accuracy
            accuracy = passed_tests / total_tests if total_tests > 0 else 0.0
            
            validation_results["total_tests"] = total_tests
            validation_results["passed_tests"] = passed_tests
            validation_results["failed_tests"] = total_tests - passed_tests
            validation_results["accuracy"] = accuracy
            validation_results["error_categories"] = dict(error_categories)
            
            return validation_results
        except Exception as e:
            self.logger.error(f"验证知识库失败: {str(e)} | Validating knowledge base failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    def get_model_capabilities(self) -> Dict[str, Any]:
        """获取模型能力描述 | Get model capabilities description
        返回:
            模型能力信息 | Model capabilities information
        """
        return {
            "model_id": "knowledge",
            "model_name": "知识库专家模型",
            "model_version": "1.0.0",
            "capabilities": [
                "多学科知识存储与检索",
                "语义搜索与相似性匹配",
                "辅助其他模型完成任务",
                "知识可视化",
                "知识整合与适应",
                "知识置信度评估",
                "知识库结构优化",
                "教学辅导功能"
            ],
            "supported_domains": list(self.domain_weights.keys()),
            "external_api_support": self.use_external_api,
            "embedding_model_available": self.embedding_model is not None
        }

    def get_domain_statistics(self) -> Dict[str, Any]:
        """获取领域统计信息 | Get domain statistics
        返回:
            各领域概念数量和统计信息 | Concept counts and statistics for each domain
        """
        try:
            statistics = {
                "total_domains": len(self.knowledge_graph),
                "total_concepts": 0,
                "domains": {},
                "embedding_available": self.embedding_model is not None
            }
            
            # 统计每个领域的详细信息 | Collect detailed information for each domain
            for domain_name, concepts in self.knowledge_graph.items():
                # 过滤掉非字典的概念（如列表）| Filter out non-dict concepts (like lists)
                valid_concepts = {k: v for k, v in concepts.items() if isinstance(v, dict)}
                concept_count = len(valid_concepts)
                statistics["total_concepts"] += concept_count
                
                # 计算领域平均置信度 | Calculate domain average confidence
                total_confidence = 0.0
                for concept_details in valid_concepts.values():
                    total_confidence += concept_details.get("confidence", 0.0)
                
                avg_confidence = total_confidence / concept_count if concept_count > 0 else 0.0
                
                statistics["domains"][domain_name] = {
                    "concept_count": concept_count,
                    "average_confidence": round(avg_confidence, 3),
                    "has_embeddings": domain_name in self.knowledge_embeddings if self.embedding_model else False,
                    "non_dict_concepts": len(concepts) - concept_count  # 记录非字典概念数量 | Record non-dict concept count
                }
            
            return statistics
        except Exception as e:
            self.logger.error(f"获取领域统计信息失败: {str(e)} | Getting domain statistics failed: {str(e)}")
            return {"error": str(e)}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取模型性能指标 | Get model performance metrics
        返回:
            性能指标数据 | Performance metrics data
        """
        try:
            metrics = {
                "query_response_time": 0.0,
                "search_accuracy": 0.0,
                "knowledge_coverage": 0.0,
                "confidence_score": 0.0,
                "update_frequency": 0.0
            }
            
            # 这里应该实现具体的性能指标计算逻辑
            # Concrete performance metrics calculation logic should be implemented here
            # 简单示例：基于模型状态计算指标 | Simple example: Calculate metrics based on model state
            
            # 计算知识覆盖度 | Calculate knowledge coverage
            total_domains = len(self.domain_weights)
            loaded_domains = 0
            total_concepts_possible = 10000  # 假设的最大概念数 | Assumed maximum number of concepts
            total_concepts_actual = 0
            
            for domain in self.domain_weights.keys():
                if domain in self.knowledge_graph:
                    loaded_domains += 1
                    total_concepts_actual += len(self.knowledge_graph[domain])
            
            # 计算指标 | Calculate metrics
            metrics["knowledge_coverage"] = (loaded_domains / total_domains) * 0.5 + (min(total_concepts_actual / total_concepts_possible, 1.0)) * 0.5
            
            # 获取置信度评估 | Get confidence evaluation
            confidence_evaluation = self.evaluate_knowledge_confidence()
            if "total_confidence" in confidence_evaluation:
                metrics["confidence_score"] = confidence_evaluation["total_confidence"]
            
            # 添加其他示例指标 | Add other example metrics
            metrics["query_response_time"] = 0.5  # 示例值 (秒) | Example value (seconds)
            metrics["search_accuracy"] = 0.85  # 示例值 | Example value
            metrics["update_frequency"] = 0.1  # 示例值 (每天更新) | Example value (updates per day)
            
            return metrics
        except Exception as e:
            self.logger.error(f"获取模型性能指标失败: {str(e)} | Getting model performance metrics failed: {str(e)}")
            return {"error": str(e)}

    def on_model_loaded(self):
        """模型加载时的回调方法 | Callback method when model is loaded"""
        self.logger.info("知识库模型已加载 | Knowledge model loaded")
        
        # 确保所有必要的领域都已加载 | Ensure all necessary domains are loaded
        self.load_knowledge_base()
        
        # 如果启用了自主学习，设置学习参数 | If autonomous learning is enabled, set learning parameters
        if hasattr(self, 'learning_enabled') and self.learning_enabled:
            self.logger.info("知识库模型自主学习已启用 | Knowledge model autonomous learning enabled")

    def on_model_unloaded(self):
        """模型卸载时的回调方法 | Callback method when model is unloaded"""
        self.logger.info("知识库模型已卸载 | Knowledge model unloaded")
        
        # 清理资源 | Clean up resources
        self.embedding_model = None

    def _parse_pdf_file(self, file_path: str) -> str:
        """解析PDF文件 | Parse PDF file
        参数:
            file_path: PDF文件路径 | PDF file path
        返回:
            提取的文本内容 | Extracted text content
        """
        content = ""
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
            return content
        except Exception as e:
            self.logger.error(f"PDF文件解析失败: {str(e)} | PDF file parsing failed: {str(e)}")
            raise

    def _parse_docx_file(self, file_path: str) -> str:
        """解析DOCX文件 | Parse DOCX file
        参数:
            file_path: DOCX文件路径 | DOCX file path
        返回:
            提取的文本内容 | Extracted text content
        """
        content = ""
        try:
            doc = docx.Document(file_path)
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            return content
        except Exception as e:
            self.logger.error(f"DOCX文件解析失败: {str(e)} | DOCX file parsing failed: {str(e)}")
            raise

    def _parse_text_content_to_knowledge(self, content: str, source: str) -> Dict[str, Any]:
        """将文本内容解析为知识库格式 | Parse text content into knowledge base format
        参数:
            content: 文本内容 | Text content
            source: 来源信息 | Source information
        返回:
            知识库格式的数据 | Knowledge base formatted data
        """
        knowledge_data = {}
        try:
            # 简单的文本解析逻辑：按行分割，每行作为一个概念
            # Simple text parsing logic: split by lines, each line as a concept
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 尝试按冒号分割概念和描述 | Try to split concept and description by colon
                    if ':' in line:
                        concept, desc = line.split(':', 1)
                        concept = concept.strip()
                        desc = desc.strip()
                        if concept:
                            knowledge_data[concept] = {
                                "description": [desc] if desc else ["无描述"],
                                "related": [],
                                "source": source,
                                "confidence": 0.7
                            }
                    else:
                        # 如果没有冒号，整行作为概念
                        # If no colon, use the whole line as concept
                        if line:
                            knowledge_data[line] = {
                                "description": ["无描述"],
                                "related": [],
                                "source": source,
                                "confidence": 0.6
                            }
            return knowledge_data
        except Exception as e:
            self.logger.error(f"文本内容解析失败: {str(e)} | Text content parsing failed: {str(e)}")
            return {}

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息 | Get knowledge base statistics
        返回:
            知识库统计信息，包括概念数量、领域覆盖、置信度等
            Knowledge base statistics including concept count, domain coverage, confidence, etc.
        """
        try:
            # 获取领域统计信息 | Get domain statistics
            domain_stats = self.get_domain_statistics()
            
            # 获取置信度评估 | Get confidence evaluation
            confidence_eval = self.evaluate_knowledge_confidence()
            
            # 获取学习进度 | Get learning progress
            learning_progress = self.get_learning_progress()
            
            # 获取性能指标 | Get performance metrics
            performance_metrics = self.get_performance_metrics()
            
            # 构建完整的统计信息 | Build complete statistics
            stats = {
                "status": "success",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "domain_statistics": domain_stats,
                "confidence_evaluation": confidence_eval,
                "learning_progress": learning_progress,
                "performance_metrics": performance_metrics,
                "model_capabilities": self.get_model_capabilities(),
                "external_api_enabled": self.use_external_api,
                "embedding_model_available": self.embedding_model is not None,
                "total_domains": len(self.knowledge_graph),
                "total_concepts": sum(len(concepts) for concepts in self.knowledge_graph.values()),
                "supported_file_formats": {
                    "json": True,
                    "txt": True,
                    "pdf": PDF_SUPPORT,
                    "docx": DOCX_SUPPORT
                }
            }
            
            # 添加知识库健康状态评估 | Add knowledge base health assessment
            stats["health_status"] = self._assess_health_status(stats)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"获取知识库统计信息失败: {str(e)} | Getting knowledge base statistics failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _assess_health_status(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """评估知识库健康状态 | Assess knowledge base health status
        参数:
            stats: 统计信息 | Statistics
        返回:
            健康状态评估 | Health status assessment
        """
        health_status = {
            "overall_health": "good",
            "issues": [],
            "recommendations": []
        }
        
        # 检查概念数量 | Check concept count
        total_concepts = stats.get("total_concepts", 0)
        if total_concepts < 100:
            health_status["overall_health"] = "needs_improvement"
            health_status["issues"].append("概念数量不足")
            health_status["recommendations"].append("需要导入更多知识文件")
        
        # 检查置信度 | Check confidence
        confidence_eval = stats.get("confidence_evaluation", {})
        total_confidence = confidence_eval.get("total_confidence", 0.0)
        if total_confidence < 0.6:
            health_status["overall_health"] = "needs_improvement"
            health_status["issues"].append("平均置信度较低")
            health_status["recommendations"].append("需要验证和更新低置信度概念")
        
        # 检查领域覆盖 | Check domain coverage
        domain_stats = stats.get("domain_statistics", {})
        total_domains = domain_stats.get("total_domains", 0)
        if total_domains < 5:
            health_status["overall_health"] = "needs_improvement"
            health_status["issues"].append("领域覆盖不足")
            health_status["recommendations"].append("需要扩展知识领域")
        
        # 检查嵌入模型 | Check embedding model
        if not stats.get("embedding_model_available", False):
            health_status["issues"].append("语义嵌入模型不可用")
            health_status["recommendations"].append("安装sentence-transformers库以启用语义搜索")
        
        return health_status

    def _detect_domain(self, content: Any) -> str:
        """自动检测知识领域 | Automatically detect knowledge domain
        参数:
            content: 内容数据 | Content data
        返回:
            检测到的领域 | Detected domain
        """
        # 简单的关键词检测逻辑 | Simple keyword detection logic
        domains_keywords = {
            "physics": ["物理", "physics", "力学", "电磁", "量子", "力学", "热力学"],
            "mathematics": ["数学", "math", "公式", "计算", "几何", "代数", "微积分"],
            "chemistry": ["化学", "chemistry", "元素", "反应", "分子", "原子", "化合物"],
            "biology": ["生物", "biology", "细胞", "基因", "进化", "遗传", "生态"],
            "computer_science": ["计算机", "编程", "算法", "代码", "software", "程序", "开发"],
            "medicine": ["医学", "医疗", "疾病", "治疗", "medicine", "健康", "诊断"],
            "law": ["法律", "法规", "律师", "法院", "law", "法规", "司法"],
            "economics": ["经济", "金融", "市场", "货币", "economics", "投资", "贸易"],
            "engineering": ["工程", "技术", "设计", "制造", "机械", "电子", "电气"],
            "psychology": ["心理", "心理学", "行为", "认知", "情绪", "人格", "治疗"]
        }
        
        # 处理不同类型的内容 | Handle different types of content
        if isinstance(content, dict):
            # 如果是字典，将其转换为字符串进行关键词检测
            # If it's a dictionary, convert to string for keyword detection
            content_str = json.dumps(content, ensure_ascii=False)
            content_lower = content_str.lower()
        elif isinstance(content, str):
            content_lower = content.lower()
        else:
            # 其他类型转换为字符串 | Convert other types to string
            content_lower = str(content).lower()
        
        # 检测领域 | Detect domain
        for domain, keywords in domains_keywords.items():
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    return domain
        
        return "general"
