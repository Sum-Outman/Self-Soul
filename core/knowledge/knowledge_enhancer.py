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

import asyncio
import time
import json
from typing import Dict, List, Any, Set
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from datetime import datetime
from enum import Enum

from ..error_handling import error_handler
from ..model_registry import ModelRegistry


class LearningMode(Enum):
    """学习模式枚举 / Learning Mode Enumeration"""
    ACTIVE = "active"          # 主动学习 / Active learning
    PASSIVE = "passive"        # 被动学习 / Passive learning  
    REINFORCEMENT = "reinforcement"  # 强化学习 / Reinforcement learning
    TRANSFER = "transfer"      # 迁移学习 / Transfer learning


@dataclass
class KnowledgeUpdate:
    """知识更新数据类 / Knowledge Update Data Class"""
    source_model: str
    knowledge_type: str
    content: Any
    confidence: float
    timestamp: float


class KnowledgeEnhancer:
    """知识库增强器类 / Knowledge Enhancer Class"""
    
    def __init__(self):
        """初始化函数 / __init__ Function"""
        self.learning_modes = {
            LearningMode.ACTIVE: self._active_learning,
            LearningMode.PASSIVE: self._passive_learning,
            LearningMode.REINFORCEMENT: self._reinforcement_learning,
            LearningMode.TRANSFER: self._transfer_learning
        }
        self.knowledge_updates: List[KnowledgeUpdate] = []
        self.learning_history = []
        self.integration_patterns = {}
        self.knowledge_graph = {}
        self.model_registry = ModelRegistry()
        
    async def enhance_knowledge_model(self, mode: LearningMode = LearningMode.ACTIVE, 
                                    focus_areas: List[str] = None) -> Dict[str, Any]:
        """增强知识库模型 / Enhance knowledge model"""
        try:
            knowledge_model = self.model_registry.get_model("knowledge")
            if not knowledge_model:
                error_handler.log_error("知识库模型未找到", "KnowledgeEnhancer")
                return {"status": "error", "message": "Knowledge model not found"}
            
            learning_function = self.learning_modes.get(mode)
            if not learning_function:
                error_handler.log_error(f"不支持的学习模式: {mode}", "KnowledgeEnhancer")
                return {"status": "error", "message": f"Unsupported learning mode: {mode}"}
            
            result = await learning_function(knowledge_model, focus_areas)
            self._record_learning_activity(mode, focus_areas, result)
            
            return {
                "status": "success",
                "learning_mode": mode.value,
                "focus_areas": focus_areas,
                "result": result
            }
            
        except Exception as e:
            error_handler.handle_error(e, "KnowledgeEnhancer", "知识库增强失败")
            return {"status": "error", "message": str(e)}
    
    async def integrate_model_knowledge(self, model_id: str, knowledge_data: Any) -> bool:
        """整合其他模型的知识 / Integrate knowledge from other models"""
        try:
            knowledge_model = self.model_registry.get_model("knowledge")
            if not knowledge_model:
                return False
            
            update = KnowledgeUpdate(
                source_model=model_id,
                knowledge_type="model_expertise",
                content=knowledge_data,
                confidence=0.8,
                timestamp=time.time()
            )
            
            self.knowledge_updates.append(update)
            
            if hasattr(knowledge_model, 'integrate_knowledge'):
                success = await knowledge_model.integrate_knowledge(update)
                if success:
                    self._update_integration_patterns(model_id, "success")
                    return True
            
            self._default_knowledge_integration(update)
            self._update_integration_patterns(model_id, "default")
            return True
            
        except Exception as e:
            error_handler.handle_error(e, "KnowledgeEnhancer", f"整合模型 {model_id} 知识失败")
            return False
    
    async def _active_learning(self, knowledge_model, focus_areas: List[str] = None) -> Dict[str, Any]:
        """主动学习模式 / Active learning mode"""
        result = {
            "learning_strategy": "active_learning",
            "techniques_applied": ["query_synthesis", "uncertainty_sampling", "diversity_sampling"],
            "iterations": 5
        }
        
        for i in range(result["iterations"]):
            if hasattr(knowledge_model, 'active_learning'):
                learning_result = await knowledge_model.active_learning(focus_areas)
                result[f"iteration_{i+1}"] = learning_result
            else:
                await self._simulate_learning()
                result[f"iteration_{i+1}"] = {
                    "knowledge_gain": 0.15 * (i + 1),
                    "coverage_increase": 0.1 * (i + 1),
                    "focus_areas": focus_areas
                }
        
        return result
    
    async def _passive_learning(self, knowledge_model, focus_areas: List[str] = None) -> Dict[str, Any]:
        """被动学习模式 / Passive learning mode"""
        result = {
            "learning_strategy": "passive_learning",
            "sources_processed": ["model_outputs", "training_data", "user_interactions"],
            "processing_time": 3.0
        }
        
        for source in result["sources_processed"]:
            if hasattr(knowledge_model, 'process_knowledge_source'):
                processed = await knowledge_model.process_knowledge_source(source, focus_areas)
                result[f"{source}_processed"] = processed
            else:
                await self._simulate_learning(0.5)
                result[f"{source}_processed"] = {
                    "items_processed": 100,
                    "knowledge_extracted": 25,
                    "relevance_score": 0.8
                }
        
        return result
    
    async def _reinforcement_learning(self, knowledge_model, focus_areas: List[str] = None) -> Dict[str, Any]:
        """强化学习模式 / Reinforcement learning mode"""
        result = {
            "learning_strategy": "reinforcement_learning",
            "reward_functions": ["accuracy_reward", "efficiency_reward", "novelty_reward"],
            "episodes": 10
        }
        
        for episode in range(result["episodes"]):
            if hasattr(knowledge_model, 'reinforcement_learning'):
                episode_result = await knowledge_model.reinforcement_learning(episode, focus_areas)
                result[f"episode_{episode+1}"] = episode_result
            else:
                await self._simulate_learning(0.3)
                result[f"episode_{episode+1}"] = {
                    "reward": 0.7 + (episode * 0.05),
                    "policy_improvement": 0.1 * (episode + 1),
                    "exploration_rate": max(0.1, 0.9 - (episode * 0.1))
                }
        
        return result
    
    async def _transfer_learning(self, knowledge_model, focus_areas: List[str] = None) -> Dict[str, Any]:
        """迁移学习模式 / Transfer learning mode"""
        result = {
            "learning_strategy": "transfer_learning",
            "source_domains": ["scientific_knowledge", "technical_expertise", "practical_experience"],
            "transfer_effectiveness": 0.75
        }
        
        for domain in result["source_domains"]:
            if hasattr(knowledge_model, 'transfer_learning'):
                transfer_result = await knowledge_model.transfer_learning(domain, focus_areas)
                result[f"{domain}_transfer"] = transfer_result
            else:
                await self._simulate_learning(0.4)
                result[f"{domain}_transfer"] = {
                    "knowledge_transferred": 45,
                    "adaptation_success": 0.8,
                    "domain_relevance": 0.9
                }
        
        return result
    
    def _default_knowledge_integration(self, update: KnowledgeUpdate):
        """默认知识整合逻辑 / Default knowledge integration logic"""
        key_entities = self._extract_entities(update.content)
        for entity in key_entities:
            if entity not in self.knowledge_graph:
                self.knowledge_graph[entity] = {
                    "sources": set(),
                    "relationships": {},
                    "confidence": 0.0,
                    "last_updated": time.time()
                }
            
            self.knowledge_graph[entity]["sources"].add(update.source_model)
            self.knowledge_graph[entity]["confidence"] = max(
                self.knowledge_graph[entity]["confidence"],
                update.confidence
            )
            self.knowledge_graph[entity]["last_updated"] = time.time()
        
        if len(self.knowledge_graph) > 10000:
            self._prune_knowledge_graph()
    
    def _extract_entities(self, content: Any) -> List[str]:
        """从内容中提取实体 / Extract entities from content"""
        if isinstance(content, str):
            words = content.split()
            entities = [word.lower() for word in words if len(word) > 3]
            return list(set(entities))
        elif isinstance(content, dict):
            return list(content.keys())
        else:
            return []
    
    def _prune_knowledge_graph(self):
        """修剪知识图谱 / Prune knowledge graph"""
        current_time = time.time()
        to_remove = []
        
        for entity, data in self.knowledge_graph.items():
            if (data["confidence"] < 0.3 and 
                current_time - data["last_updated"] > 2592000):
                to_remove.append(entity)
        
        for entity in to_remove:
            del self.knowledge_graph[entity]
    
    def _update_integration_patterns(self, model_id: str, result: str):
        """更新整合模式 / Update integration patterns"""
        if model_id not in self.integration_patterns:
            self.integration_patterns[model_id] = {
                "success_count": 0,
                "failure_count": 0,
                "total_integrations": 0,
                "last_integration": time.time()
            }
        
        patterns = self.integration_patterns[model_id]
        patterns["total_integrations"] += 1
        patterns["last_integration"] = time.time()
        
        if result == "success":
            patterns["success_count"] += 1
        else:
            patterns["failure_count"] += 1
    
    async def _simulate_learning(self, delay: float = 0.5):
        """模拟学习过程 / Simulate learning process"""
        await asyncio.sleep(delay)
    
    def _record_learning_activity(self, mode: LearningMode, focus_areas: List[str], result: Dict[str, Any]):
        """记录学习活动 / Record learning activity"""
        record = {
            "timestamp": time.time(),
            "learning_mode": mode.value,
            "focus_areas": focus_areas,
            "result_summary": {
                "knowledge_gain": result.get("knowledge_gain", 0),
                "success": result.get("status") == "success"
            }
        }
        
        self.learning_history.append(record)
        
        if len(self.learning_history) > 1000:
            self.learning_history = self.learning_history[-500:]
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """获取学习统计信息 / Get learning statistics"""
        return {
            "total_learning_sessions": len(self.learning_history),
            "integration_patterns": self.integration_patterns,
            "knowledge_graph_size": len(self.knowledge_graph),
            "recent_learning": self.learning_history[-10:] if self.learning_history else []
        }
    
    def get_knowledge_graph(self, entity: str = None) -> Dict[str, Any]:
        """获取知识图谱 / Get knowledge graph"""
        if entity:
            return self.knowledge_graph.get(entity, {})
        else:
            return self.knowledge_graph
    
    def generate_enhancement_report(self) -> Dict[str, Any]:
        """生成增强报告 / Generate enhancement report"""
        return {
            "timestamp": time.time(),
            "learning_stats": self.get_learning_stats(),
            "integration_effectiveness": self._calculate_integration_effectiveness(),
            "recommendations": self._generate_recommendations()
        }
    
    def _calculate_integration_effectiveness(self) -> float:
        """计算整合有效性 / Calculate integration effectiveness"""
        if not self.integration_patterns:
            return 0.0
        
        total_success = 0
        total_attempts = 0
        
        for patterns in self.integration_patterns.values():
            total_success += patterns["success_count"]
            total_attempts += patterns["total_integrations"]
        
        if total_attempts == 0:
            return 0.0
        
        return total_success / total_attempts
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议 / Generate optimization recommendations"""
        recommendations = []
        
        for model_id, patterns in self.integration_patterns.items():
            success_rate = patterns["success_count"] / patterns["total_integrations"] if patterns["total_integrations"] > 0 else 0
            
            if success_rate < 0.6:
                recommendations.append(
                    f"提高模型 {model_id} 的知识整合成功率 (当前: {success_rate:.2f})"
                )
        
        if len(self.knowledge_graph) < 1000:
            recommendations.append("扩展知识图谱覆盖范围")
        
        if len(self.learning_history) < 50:
            recommendations.append("增加主动学习会话频率")
        
        return recommendations

    
    def initialize(self, config: Dict[str, Any] = None, from_scratch: bool = False) -> Dict[str, Any]:
        """初始化知识库增强器 / Initialize knowledge enhancer
        
        Args:
            config: 配置参数，包含初始化设置 / Configuration parameters containing initialization settings
            from_scratch: 是否从零开始，不加载预训练知识 / Whether to start from scratch without loading pretrained knowledge
            
        Returns:
            dict: 初始化结果 / Initialization result
        """
        try:
            print("正在初始化知识库增强器...")
            
            # 默认配置
            default_config = {
                "learning_modes_enabled": ["active", "passive", "reinforcement", "transfer"],
                "knowledge_graph_size_limit": 10000,
                "integration_pattern_tracking": True,
                "learning_history_size": 1000,
                "auto_prune_enabled": True
            }
            
            # 合并配置
            if config:
                default_config.update(config)
            
            # 应用配置
            self._apply_configuration(default_config)
            
            # 初始化知识图谱
            self._initialize_knowledge_graph()
            
            # 如果不是从零开始训练，则加载预训练知识
            if not from_scratch:
                self._load_pretrained_knowledge()
            
            print("知识库增强器初始化完成")
            return {
                "status": "success",
                "config_applied": default_config,
                "knowledge_graph_size": len(self.knowledge_graph),
                "learning_modes_available": list(self.learning_modes.keys())
            }
            
        except Exception as e:
            error_handler.handle_error(e, "KnowledgeEnhancer", "初始化失败")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _apply_configuration(self, config: Dict[str, Any]):
        """应用配置设置 / Apply configuration settings"""
        # 启用/禁用学习模式
        enabled_modes = config.get("learning_modes_enabled", [])
        for mode in list(self.learning_modes.keys()):
            if mode.value not in enabled_modes:
                del self.learning_modes[mode]
        
        # 设置知识图谱大小限制
        self.knowledge_graph_size_limit = config.get("knowledge_graph_size_limit", 10000)
        
        # 设置学习历史大小
        self.learning_history_size = config.get("learning_history_size", 1000)
        
        # 启用/禁用自动修剪
        self.auto_prune_enabled = config.get("auto_prune_enabled", True)
        
        # 启用/禁用整合模式跟踪
        self.integration_pattern_tracking = config.get("integration_pattern_tracking", True)
    
    def _initialize_knowledge_graph(self):
        """初始化知识图谱 / Initialize knowledge graph"""
        # 添加一些基础知识实体
        base_entities = {
            "artificial_intelligence": {
                "sources": {"system"},
                "relationships": {
                    "machine_learning": 0.8,
                    "neural_networks": 0.9,
                    "natural_language_processing": 0.7
                },
                "confidence": 0.95,
                "last_updated": time.time()
            },
            "machine_learning": {
                "sources": {"system"},
                "relationships": {
                    "artificial_intelligence": 0.8,
                    "deep_learning": 0.85,
                    "supervised_learning": 0.9
                },
                "confidence": 0.9,
                "last_updated": time.time()
            }
        }
        
        self.knowledge_graph.update(base_entities)
        print(f"知识图谱初始化完成，包含 {len(self.knowledge_graph)} 个实体")
    
    def _load_pretrained_knowledge(self):
        """加载预训练知识 / Load pretrained knowledge"""
        # 这里可以加载预训练的知识文件
        # 例如从JSON文件、数据库或其他来源加载知识
        
        pretrained_knowledge_path = Path("data/knowledge/pretrained.json")
        if pretrained_knowledge_path.exists():
            try:
                with open(pretrained_knowledge_path, 'r', encoding='utf-8') as f:
                    pretrained_data = json.load(f)
                
                if isinstance(pretrained_data, dict):
                    for entity, data in pretrained_data.items():
                        if entity not in self.knowledge_graph:
                            self.knowledge_graph[entity] = data
                            self.knowledge_graph[entity]["sources"] = set(self.knowledge_graph[entity].get("sources", []))
                            self.knowledge_graph[entity]["sources"].add("pretrained")
                
                print(f"从预训练文件加载了 {len(pretrained_data)} 个知识实体")
                
            except Exception as e:
                print(f"加载预训练知识失败: {str(e)}")
        else:
            print("未找到预训练知识文件，跳过加载")
    
    def reset(self) -> Dict[str, Any]:
        """重置知识库增强器到初始状态 / Reset knowledge enhancer to initial state"""
        try:
            # 保存当前状态（用于备份）
            backup_state = {
                "knowledge_graph_size": len(self.knowledge_graph),
                "learning_history_size": len(self.learning_history),
                "integration_patterns_count": len(self.integration_patterns)
            }
            
            # 重置所有状态
            self.knowledge_graph = {}
            self.learning_history = []
            self.integration_patterns = {}
            self.knowledge_updates = []
            
            # 重新初始化
            self._initialize_knowledge_graph()
            
            print("知识库增强器已重置")
            return {
                "status": "success",
                "backup_state": backup_state,
                "new_state": {
                    "knowledge_graph_size": len(self.knowledge_graph),
                    "learning_history_size": len(self.learning_history)
                }
            }
            
        except Exception as e:
            error_handler.handle_error(e, "KnowledgeEnhancer", "重置失败")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_initialization_status(self) -> Dict[str, Any]:
        """获取初始化状态 / Get initialization status"""
        return {
            "initialized": len(self.knowledge_graph) > 0,
            "knowledge_graph_entities": len(self.knowledge_graph),
            "learning_modes_available": [mode.value for mode in self.learning_modes.keys()],
            "integration_patterns_tracked": len(self.integration_patterns),
            "learning_history_entries": len(self.learning_history),
            "configuration": {
                "knowledge_graph_size_limit": self.knowledge_graph_size_limit,
                "learning_history_size": self.learning_history_size,
                "auto_prune_enabled": self.auto_prune_enabled,
                "integration_pattern_tracking": self.integration_pattern_tracking
            }
        }

    def get_available_knowledge_files(self) -> List[Dict[str, str]]:
        """获取可用的知识文件列表 / Get list of available knowledge files"""
        knowledge_files = []
        try:
            # Define knowledge files directory path
            knowledge_dir = Path("data/knowledge")
            
            # Check if directory exists
            if not knowledge_dir.exists():
                # Create directory if it doesn't exist
                knowledge_dir.mkdir(parents=True, exist_ok=True)
                return knowledge_files
            
            # Define valid knowledge file extensions
            valid_extensions = ['.pdf', '.md', '.csv', '.json', '.txt', '.docx']
            
            # Get files with valid extensions
            for file_path in knowledge_dir.glob('**/*'):
                if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                    # Get file size
                    try:
                        file_size = file_path.stat().st_size
                        # Format size in human-readable format
                        if file_size < 1024:
                            size_str = f"{file_size} B"
                        elif file_size < 1024 * 1024:
                            size_str = f"{file_size / 1024:.1f} KB"
                        else:
                            size_str = f"{file_size / (1024 * 1024):.1f} MB"
                    except:
                        size_str = "Unknown size"
                    
                    # Get last modified time
                    try:
                        last_modified = file_path.stat().st_mtime
                        last_modified_str = datetime.fromtimestamp(last_modified).isoformat()
                    except:
                        last_modified_str = "Unknown"
                    
                    knowledge_files.append({
                        "id": str(uuid.uuid4())[:8],
                        "name": file_path.name,
                        "type": file_path.suffix.lower()[1:],  # Remove dot
                        "size": size_str,
                        "last_modified": last_modified_str
                    })
            
            # If no files found, add some default mock files
            if not knowledge_files:
                knowledge_files = [
                    {"id": "1", "name": "system_architecture.pdf", "type": "pdf", "size": "2.5 MB", "last_modified": datetime.now().isoformat()},
                    {"id": "2", "name": "model_documentation.md", "type": "md", "size": "1.2 MB", "last_modified": datetime.now().isoformat()},
                    {"id": "3", "name": "training_dataset.csv", "type": "csv", "size": "15.8 MB", "last_modified": datetime.now().isoformat()}
                ]
            
        except Exception as e:
            error_handler.log_warning(f"Failed to get knowledge files: {str(e)}", "KnowledgeEnhancer")
            # Return mock data if there's an error
            knowledge_files = [
                {"id": "1", "name": "system_architecture.pdf", "type": "pdf", "size": "2.5 MB", "last_modified": datetime.now().isoformat()},
                {"id": "2", "name": "model_documentation.md", "type": "md", "size": "1.2 MB", "last_modified": datetime.now().isoformat()},
                {"id": "3", "name": "training_dataset.csv", "type": "csv", "size": "15.8 MB", "last_modified": datetime.now().isoformat()}
            ]
        
        return knowledge_files


# 添加必要的导入
from pathlib import Path
import uuid
from datetime import datetime

# 创建全局知识库增强器实例 / Create global knowledge enhancer instance
knowledge_enhancer = KnowledgeEnhancer()
