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
import json
import os
import time
import numpy as np
from datetime import datetime
from collections import defaultdict
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from core.error_handling import error_handler

class KnowledgeGraph:
    """知识图谱 - 结构化知识表示"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_counter = 0
        self.relation_counter = 0
        self._initialize_core_schema()
    
    def _initialize_core_schema(self):
        """初始化核心图谱模式"""
        # 添加基本实体类型
        core_entities = [
            ('concept', '概念'),
            ('event', '事件'),
            ('object', '对象'),
            ('person', '人物'),
            ('location', '地点'),
            ('time', '时间')
        ]
        
        # 添加基本关系类型
        core_relations = [
            ('is_a', '是一种'),
            ('part_of', '的一部分'),
            ('related_to', '相关于'),
            ('causes', '导致'),
            ('located_in', '位于'),
            ('occurred_at', '发生于')
        ]
        
        for entity_type, label in core_entities:
            self.graph.add_node(entity_type, type='entity_type', label=label)
        
        for relation_type, label in core_relations:
            self.graph.add_node(relation_type, type='relation_type', label=label)
    
    def add_entity(self, entity_id, entity_type, properties=None):
        """添加实体到知识图谱"""
        if entity_id not in self.graph:
            self.graph.add_node(entity_id, type='entity', entity_type=entity_type)
            self.entity_counter += 1
        
        if properties:
            for key, value in properties.items():
                self.graph.nodes[entity_id][key] = value
        
        return entity_id
    
    def add_relation(self, source_id, relation_type, target_id, properties=None):
        """添加关系到知识图谱"""
        relation_id = f"rel_{self.relation_counter}"
        self.graph.add_edge(source_id, target_id, key=relation_id, 
                           relation_type=relation_type, weight=1.0)
        self.relation_counter += 1
        
        if properties:
            for key, value in properties.items():
                self.graph[source_id][target_id][key] = value
        
        return relation_id
    
    def query(self, query_pattern, max_results=10):
        """查询知识图谱"""
        results = []
        
        # 简化的查询实现
        if isinstance(query_pattern, dict):
            # 实体查询
            if 'entity_type' in query_pattern:
                for node, data in self.graph.nodes(data=True):
                    if data.get('type') == 'entity' and data.get('entity_type') == query_pattern['entity_type']:
                        results.append({'node': node, 'data': data})
        
        # 关系查询
        elif isinstance(query_pattern, tuple) and len(query_pattern) == 3:
            source, relation, target = query_pattern
            for u, v, data in self.graph.edges(data=True):
                if (source is None or u == source) and \
                   (relation is None or data.get('relation_type') == relation) and \
                   (target is None or v == target):
                    results.append({'source': u, 'target': v, 'relation': data})
        
        return results[:max_results]
    
    def get_statistics(self):
        """获取图谱统计信息"""
        return {
            'total_entities': self.entity_counter,
            'total_relations': self.relation_counter,
            'entity_types': len([n for n, d in self.graph.nodes(data=True) 
                               if d.get('type') == 'entity_type']),
            'relation_types': len([n for n, d in self.graph.nodes(data=True) 
                                 if d.get('type') == 'relation_type']),
            'graph_density': nx.density(self.graph)
        }

class ExperienceMemory:
    """经验记忆 - 存储和检索具体经验"""
    
    def __init__(self, max_size=10000):
        self.memories = []
        self.max_size = max_size
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.memory_vectors = None
        self._fit_vectorizer()
    
    def _fit_vectorizer(self):
        """训练向量化器"""
        # 使用一些初始文本进行拟合
        sample_texts = [
            "学习新知识", "解决问题", "处理输入", "训练模型",
            "协调任务", "自我改进", "知识积累", "经验总结"
        ]
        self.vectorizer.fit(sample_texts)
    
    def store(self, experience):
        """存储经验"""
        if len(self.memories) >= self.max_size:
            # 移除最旧的记忆
            self.memories.pop(0)
            if self.memory_vectors is not None:
                self.memory_vectors = self.memory_vectors[1:]
        
        memory_entry = {
            'id': f"memory_{len(self.memories)}_{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'experience': experience,
            'embedding': self._embed_experience(experience)
        }
        
        self.memories.append(memory_entry)
        
        # 更新向量矩阵
        if self.memory_vectors is None:
            self.memory_vectors = np.array([memory_entry['embedding']])
        else:
            self.memory_vectors = np.vstack([self.memory_vectors, memory_entry['embedding']])
        
        return memory_entry['id']
    
    def _embed_experience(self, experience):
        """将经验转换为向量表示"""
        # 简化的嵌入实现
        text_representation = str(experience)
        try:
            embedding = self.vectorizer.transform([text_representation]).toarray()[0]
        except:
            embedding = np.zeros(1000)  # 默认向量
        
        return embedding
    
    def retrieve(self, query, max_results=5):
        """检索相关经验"""
        if not self.memories:
            return []
        
        # 将查询转换为向量
        query_embedding = self._embed_experience({'query': query})
        
        # 计算相似度
        similarities = cosine_similarity([query_embedding], self.memory_vectors)[0]
        
        # 获取最相关的记忆
        indices = np.argsort(similarities)[::-1][:max_results]
        
        results = []
        for idx in indices:
            if similarities[idx] > 0.1:  # 相似度阈值
                results.append({
                    'memory': self.memories[idx],
                    'similarity': float(similarities[idx])
                })
        
        return results
    
    def get_statistics(self):
        """获取记忆统计信息"""
        return {
            'total_memories': len(self.memories),
            'memory_capacity': self.max_size,
            'utilization_percentage': (len(self.memories) / self.max_size) * 100
        }

class SemanticIndex:
    """语义索引 - 快速语义搜索"""
    
    def __init__(self):
        self.index = defaultdict(list)
        self.entity_index = defaultdict(set)
        self.concept_index = defaultdict(set)
    
    def index(self, content, content_id):
        """索引内容"""
        # 提取关键概念和实体
        concepts = self._extract_concepts(content)
        entities = self._extract_entities(content)
        
        # 更新索引
        for concept in concepts:
            self.concept_index[concept].add(content_id)
        
        for entity in entities:
            self.entity_index[entity].add(content_id)
        
        # 添加到主索引
        self.index[content_id] = {
            'concepts': concepts,
            'entities': entities,
            'timestamp': time.time()
        }
    
    def _extract_concepts(self, content):
        """提取概念"""
        # 简化的概念提取
        text = str(content).lower()
        concepts = []
        
        # 常见概念关键词
        concept_keywords = [
            'learn', 'know', 'understand', 'think', 'reason',
            'solve', 'create', 'generate', 'optimize', 'improve'
        ]
        
        for keyword in concept_keywords:
            if keyword in text:
                concepts.append(keyword)
        
        return list(set(concepts))
    
    def _extract_entities(self, content):
        """提取实体"""
        # 简化的实体提取
        text = str(content)
        entities = []
        
        # 识别可能的大写实体（简化实现）
        words = text.split()
        for word in words:
            if len(word) > 2 and word[0].isupper():
                entities.append(word)
        
        return list(set(entities))
    
    def search(self, query, max_results=10):
        """语义搜索"""
        query_concepts = self._extract_concepts({'query': query})
        query_entities = self._extract_entities({'query': query})
        
        results = set()
        
        # 概念匹配
        for concept in query_concepts:
            if concept in self.concept_index:
                results.update(self.concept_index[concept])
        
        # 实体匹配
        for entity in query_entities:
            if entity in self.entity_index:
                results.update(self.entity_index[entity])
        
        # 计算相关性分数
        scored_results = []
        for result_id in results:
            score = self._calculate_relevance(result_id, query_concepts, query_entities)
            scored_results.append({'id': result_id, 'score': score})
        
        # 按分数排序
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        return scored_results[:max_results]
    
    def _calculate_relevance(self, content_id, query_concepts, query_entities):
        """计算相关性分数"""
        content_data = self.index.get(content_id, {})
        content_concepts = set(content_data.get('concepts', []))
        content_entities = set(content_data.get('entities', []))
        
        # 概念匹配分数
        concept_match = len(content_concepts.intersection(query_concepts))
        
        # 实体匹配分数
        entity_match = len(content_entities.intersection(query_entities))
        
        # 时间衰减（新内容更有价值）
        time_decay = 1.0 / (1.0 + (time.time() - content_data.get('timestamp', 0)) / 3600)
        
        return concept_match * 0.6 + entity_match * 0.4 + time_decay * 0.2

class StructuredKnowledgeBase:
    """结构化知识库系统"""
    
    def __init__(self, from_scratch=False):
        self.knowledge_graph = KnowledgeGraph()
        self.experience_memory = ExperienceMemory()
        self.semantic_index = SemanticIndex()
        
        # 加载现有知识，除非是从零开始训练
        if not from_scratch:
            self._load_existing_knowledge()
        
        error_handler.log_info("结构化知识库初始化完成", "StructuredKnowledgeBase")
    
    def _load_existing_knowledge(self):
        """加载现有知识"""
        # 这里可以加载预定义的知识或从文件加载
        try:
            # 添加一些核心概念
            ai_concept = self.knowledge_graph.add_entity('artificial_intelligence', 'concept', 
                                                       {'description': '人工智能', 'importance': 0.9})
            ml_concept = self.knowledge_graph.add_entity('machine_learning', 'concept',
                                                       {'description': '机器学习', 'importance': 0.8})
            
            # 添加关系
            self.knowledge_graph.add_relation(ml_concept, 'is_a', ai_concept,
                                            {'certainty': 0.95})
            
        except Exception as e:
            error_handler.handle_error(e, "StructuredKnowledgeBase", "加载现有知识失败")
    
    def add_experience(self, experience):
        """添加结构化的经验"""
        try:
            # 提取关键信息
            entities = self._extract_entities(experience)
            relationships = self._extract_relationships(experience)
            concepts = self._extract_concepts(experience)
            
            # 更新知识图谱
            for entity_id, entity_data in entities.items():
                self.knowledge_graph.add_entity(entity_id, entity_data['type'], entity_data['properties'])
            
            for rel_data in relationships:
                self.knowledge_graph.add_relation(rel_data['source'], rel_data['relation_type'], 
                                                rel_data['target'], rel_data['properties'])
            
            # 存储经验记忆
            memory_id = self.experience_memory.store(experience)
            
            # 索引经验
            self.semantic_index.index(experience, memory_id)
            
            return {
                'success': True,
                'entities_added': len(entities),
                'relations_added': len(relationships),
                'memory_id': memory_id
            }
            
        except Exception as e:
            error_handler.handle_error(e, "StructuredKnowledgeBase", "添加经验失败")
            return {"error": str(e)}
    
    def _extract_entities(self, experience):
        """从经验中提取实体"""
        # 简化的实体提取
        entities = {}
        text = str(experience).lower()
        
        # 识别可能的关键实体
        if 'error' in text:
            entities['error_occurrence'] = {
                'type': 'event',
                'properties': {'severity': 'medium', 'context': text[:100]}
            }
        
        if 'success' in text:
            entities['success_event'] = {
                'type': 'event', 
                'properties': {'context': text[:100]}
            }
        
        return entities
    
    def _extract_relationships(self, experience):
        """从经验中提取关系"""
        # 简化的关系提取
        relationships = []
        text = str(experience).lower()
        
        if 'error' in text and 'process' in text:
            relationships.append({
                'source': 'error_occurrence',
                'relation_type': 'related_to',
                'target': 'processing_task',
                'properties': {'context': 'error during processing'}
            })
        
        return relationships
    
    def _extract_concepts(self, experience):
        """从经验中提取概念"""
        # 简化的概念提取
        concepts = []
        text = str(experience).lower()
        
        if 'learn' in text:
            concepts.append('learning')
        if 'process' in text:
            concepts.append('processing')
        if 'error' in text:
            concepts.append('error_handling')
        
        return concepts
    
    def retrieve_relevant_knowledge(self, query):
        """检索相关知识"""
        try:
            # 语义搜索
            semantic_results = self.semantic_index.search(query)
            
            # 图查询
            graph_results = self.knowledge_graph.query(query)
            
            # 经验检索
            experience_results = self.experience_memory.retrieve(query)
            
            # 整合结果
            integrated_results = {
                'semantic_matches': semantic_results,
                'graph_matches': graph_results,
                'experience_matches': experience_results,
                'total_relevance': len(semantic_results) + len(graph_results) + len(experience_results)
            }
            
            return integrated_results
            
        except Exception as e:
            error_handler.handle_error(e, "StructuredKnowledgeBase", "知识检索失败")
            return {"error": str(e)}
    
    def get_knowledge_stats(self):
        """获取知识库统计信息"""
        return {
            'knowledge_graph': self.knowledge_graph.get_statistics(),
            'experience_memory': self.experience_memory.get_statistics(),
            'total_indexed_items': len(self.semantic_index.index)
        }
    
    def export_knowledge(self, file_path):
        """导出知识到文件"""
        try:
            export_data = {
                'knowledge_graph': nx.node_link_data(self.knowledge_graph.graph),
                'experience_memory_count': len(self.experience_memory.memories),
                'semantic_index_stats': {
                    'concept_count': len(self.semantic_index.concept_index),
                    'entity_count': len(self.semantic_index.entity_index)
                },
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return {"success": True, "export_path": file_path}
            
        except Exception as e:
            error_handler.handle_error(e, "StructuredKnowledgeBase", "知识导出失败")
            return {"error": str(e)}
    
    def import_knowledge(self, file_path):
        """从文件导入知识"""
        try:
            if not os.path.exists(file_path):
                return {"error": "文件不存在"}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 这里可以实现具体的导入逻辑
            # 目前返回成功状态
            
            return {"success": True, "imported_items": "待实现"}
            
        except Exception as e:
            error_handler.handle_error(e, "StructuredKnowledgeBase", "知识导入失败")
            return {"error": str(e)}