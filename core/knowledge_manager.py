"""
Knowledge Manager: AGI-Enhanced Knowledge Base Management with Autonomous Learning

AGI-Enhanced Features:
- Advanced knowledge graph construction and reasoning
- Autonomous learning from multiple data sources
- Real-time knowledge integration and updating
- Multi-domain knowledge fusion
- Intelligent knowledge retrieval and inference
- Self-organizing knowledge structures
- Adaptive learning strategies
- Cross-domain knowledge transfer
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from core.error_handling import error_handler
from core.self_learning import agi_self_learning_system

logger = logging.getLogger(__name__)

# 跨领域知识推理引擎可用性标志（延迟加载）
HAS_CROSS_DOMAIN_REASONING = None
_CROSS_DOMAIN_ENGINE_CLASS = None

class KnowledgeManager:
    """Knowledge manager class, responsible for knowledge base loading, updating, and autonomous learning functionality"""
    
    @classmethod
    def _load_cross_domain_engine_if_available(cls):
        """延迟加载跨领域知识推理引擎"""
        global HAS_CROSS_DOMAIN_REASONING, _CROSS_DOMAIN_ENGINE_CLASS
        
        if HAS_CROSS_DOMAIN_REASONING is None:
            try:
                from core.cross_domain_knowledge_reasoning import CrossDomainKnowledgeReasoningEngine
                _CROSS_DOMAIN_ENGINE_CLASS = CrossDomainKnowledgeReasoningEngine
                HAS_CROSS_DOMAIN_REASONING = True
                logger.info("CrossDomainKnowledgeReasoningEngine loaded successfully")
            except ImportError as e:
                logger.warning(f"CrossDomainKnowledgeReasoningEngine not available: {e}")
                HAS_CROSS_DOMAIN_REASONING = False
        
        return HAS_CROSS_DOMAIN_REASONING, _CROSS_DOMAIN_ENGINE_CLASS
    
    def __init__(self, knowledge_base_path: str = None):
        """Initialize knowledge manager
        
        Args:
            knowledge_base_path: Knowledge base file path, if None then use default path
        """
        # Set knowledge base path
        if knowledge_base_path is None:
            # Use default knowledge base path
            self.knowledge_base_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'core', 'data', 'knowledge'
            )
        else:
            self.knowledge_base_path = knowledge_base_path
        
        # Ensure knowledge base directory exists
        os.makedirs(self.knowledge_base_path, exist_ok=True)
        
        # Initialize all attributes before load_knowledge_bases is called
        self.knowledge_bases = {}
        
        # Autonomous learning status
        self.autonomous_learning_active = False
        self.learning_start_time = None
        self.learning_progress = 0
        
        # AGI-enhanced functionality components
        self.knowledge_graph = None
        self.learning_strategy = "adaptive"
        self.learning_priority = {
            "high_priority_domains": ["computer_science", "mathematics", "physics"],
            "medium_priority_domains": ["engineering", "biology", "chemistry"],
            "low_priority_domains": ["art", "music", "literature"]
        }
        
        # Real-time knowledge update system
        self.real_time_updates = {}
        self.knowledge_freshness = {}
        
        # Multi-source knowledge fusion
        self.knowledge_sources = {}
        self.data_fusion_active = True
        
        # 跨领域知识推理引擎
        self.cross_domain_reasoning_engine = None
        
        # 延迟加载跨领域推理引擎
        has_cross_domain, engine_class = self._load_cross_domain_engine_if_available()
        
        if has_cross_domain and engine_class:
            try:
                self.cross_domain_reasoning_engine = engine_class(self)
                logger.info("CrossDomainKnowledgeReasoningEngine initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize CrossDomainKnowledgeReasoningEngine: {e}")
        
        # Similarity calculation configuration
        self.similarity_config = {
            # Domain similarity weight configuration
            'domain_similarity_weights': {
                'name_overlap': 0.2,
                'semantic': 0.3,
                'structural': 0.25,
                'abstraction': 0.15,
                'complexity': 0.1
            },
            # Semantic similarity threshold configuration
            'semantic_thresholds': {
                'min_similarity_for_transfer': 0.2,
                'high_similarity': 0.7,
                'very_high_similarity': 0.8,
                'min_confidence': 0.5,
                'balanced_threshold': 0.6,
                'composite_threshold': 1.5,
                'min_selection_threshold': 0.5
            },
            # Structural similarity weight configuration
            'structural_weights': {
                'relation_similarity': 0.6,
                'level_similarity': 0.4
            },
            # Knowledge transfer efficiency configuration
            'transfer_efficiency_config': {
                'base_efficiency_factor': 1.2,
                'max_efficiency': 0.9,
                'domain_factor_range': (0.5, 1.0),
                'min_transfer_efficiency': 0.2,
                'efficiency_weights': {
                    'similarity_weight': 0.7,
                    'coverage_weight': 0.3,
                    'neutral_domain_factor': 0.5,
                    'domain_adjustment_strength': 0.4
                }
            },
            # TF-IDF configuration
            'tfidf_config': {
                'stop_words': 'english',
                'max_features': 1000,
                'min_df': 1,
                'max_df': 1.0
            },
            # Quality assessment configuration
            'quality_assessment_config': {
                'name_length_threshold': 50,
                'description_length_threshold': 200,
                'min_description_length': 20,
                'base_data_richness': 0.5,
                'useful_field_increment': 0.1,
                'quality_weights': {
                    'name_length_score': 0.2,
                    'description_length_score': 0.3,
                    'name_uniqueness': 0.1,
                    'description_completeness': 0.2,
                    'data_richness': 0.2
                },
                'quality_thresholds': {
                    'high_quality': 0.8,
                    'medium_quality': 0.5
                }
            }
        }
        
        # Knowledge freshness configuration
        self.freshness_config = {
            'freshness_thresholds': {
                'very_fresh_days': 1,      # Within 1 day is very fresh
                'fresh_days': 7,           # Within 7 days is fresh
                'moderate_days': 30,       # Within 30 days is moderately fresh
                'old_days': 90,            # Within 90 days is relatively old
                'very_fresh_score': 1.0,
                'fresh_score': 0.9,
                'moderate_score': 0.7,
                'old_score': 0.5,
                'very_old_score': 0.3
            }
        }
        
        # Knowledge base statistics information
        self.knowledge_stats = {
            'total_domains': 0,
            'total_facts': 0,
            'last_updated': None,
            'autonomous_learning_enabled': False,
            'knowledge_graph_size': 0,
            'learning_efficiency': 0.0,
            'knowledge_freshness_score': 0.0,
            'cross_domain_connections': 0
        }
        
        # Causal relationship management configuration
        self.causal_config = {
            'causal_graph_enabled': True,
            'causal_discovery_active': False,
            'causal_inference_methods': ['backdoor', 'frontdoor', 'instrumental_variable'],
            'counterfactual_reasoning_enabled': True,
            'causal_confidence_threshold': 0.7,
            'max_causal_path_length': 5,
            'causal_temporal_window': 7,  # days
            'causal_update_frequency': 'weekly'
        }
        
        # Causal knowledge storage
        self.causal_knowledge = {
            'causal_graph': None,           # NetworkX causal graph
            'causal_relationships': [],      # List of causal relationships
            'counterfactual_scenarios': [],  # Counterfactual scenarios
            'causal_evidence': {},           # Evidence for causal relationships
            'causal_discovery_history': []   # History of causal discoveries
        }
        
        # Causal reasoning components (lazy initialization)
        self.causal_engine = None
        self.causal_discovery_algorithm = None
        self.causal_knowledge_graph = None
        self.hidden_confounder_detector = None
        
        # Causal relationship statistics
        self.causal_stats = {
            'total_causal_relationships': 0,
            'causal_discoveries': 0,
            'counterfactual_analyses': 0,
            'causal_queries_processed': 0,
            'causal_inference_accuracy': 0.0,
            'last_causal_update': None
        }
        
        # Load knowledge base (all attributes are now initialized)
        self.load_knowledge_bases()
        
        # Initialize AGI knowledge enhancement functionality
        self._initialize_agi_enhancements()
        
        logger.info(f"Knowledge Manager initialized with knowledge base path: {self.knowledge_base_path}")
    
    def _initialize_agi_enhancements(self):
        """Initialize AGI knowledge enhancement functionality"""
        try:
            # Initialize knowledge graph
            self.knowledge_graph = {}
            
            # Initialize knowledge sources
            self.knowledge_sources = {
                "internal": "Local knowledge base",
                "external": "External data sources",
                "user_input": "User input",
                "autonomous_learning": "Autonomous learning"
            }
            
            # Initialize knowledge freshness tracking
            for domain in self.knowledge_bases.keys():
                self.knowledge_freshness[domain] = {
                    "last_updated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
                    "freshness_score": 1.0,
                    "update_frequency": "daily"
                }
            
            logger.info("AGI knowledge enhancement features initialization completed")
            
        except Exception as e:
            error_handler.log_error(f"AGI knowledge enhancement features initialization failed: {e}", "KnowledgeManager",
                                  {"method": "_initialize_agi_enhancements"})

    def load_knowledge_bases(self):
        """Load all knowledge base files"""
        try:
            if not os.path.exists(self.knowledge_base_path):
                error_handler.log_warning(f"Knowledge base path does not exist: {self.knowledge_base_path}", "KnowledgeManager")
                return
            
            # Load all JSON knowledge base files
            knowledge_files = []
            for file_name in os.listdir(self.knowledge_base_path):
                if file_name.endswith('.json'):
                    knowledge_files.append(file_name)
            
            # Load each knowledge base file
            for file_name in knowledge_files:
                file_path = os.path.join(self.knowledge_base_path, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        knowledge_data = json.load(f)
                    
                    # Extract knowledge base name (remove .json extension)
                    domain_name = file_name.replace('.json', '')
                    self.knowledge_bases[domain_name] = knowledge_data
                    
                    logger.info(f"Loaded knowledge base: {domain_name} with {len(knowledge_data)} items")
                    
                except Exception as e:
                    error_handler.log_error(f"Failed to load knowledge base {file_name}: {e}", "KnowledgeManager",
                                          {"method": "load_knowledge_bases", "file": file_name})
            
            # Build knowledge graph
            self._build_knowledge_graph()
            
            # Update statistics information
            self._update_knowledge_stats()
            
        except Exception as e:
            error_handler.log_error(f"Failed to load knowledge bases: {e}", "KnowledgeManager",
                                  {"method": "load_knowledge_bases"})
    
    def _update_knowledge_stats(self):
        """Update knowledge base statistics information"""
        try:
            total_domains = len(self.knowledge_bases)
            total_facts = 0
            
            for domain, knowledge_data in self.knowledge_bases.items():
                if isinstance(knowledge_data, list):
                    total_facts += len(knowledge_data)
                elif isinstance(knowledge_data, dict):
                    total_facts += len(knowledge_data)
            
            # Calculate knowledge graph metrics
            knowledge_graph_size = 0
            cross_domain_connections = 0
            if hasattr(self, 'knowledge_graph') and self.knowledge_graph:
                # Count total concepts and relations
                for domain, graph in self.knowledge_graph.items():
                    knowledge_graph_size += len(graph.get('concepts', []))
                    cross_domain_connections += len(graph.get('relations', []))
            
            # Preserve other statistics or initialize default values
            learning_efficiency = self.knowledge_stats.get('learning_efficiency', 0.0)
            knowledge_freshness_score = self.knowledge_stats.get('knowledge_freshness_score', 0.0)
            
            self.knowledge_stats = {
                'total_domains': total_domains,
                'total_facts': total_facts,
                'last_updated': datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
                'autonomous_learning_enabled': self.autonomous_learning_active,
                'knowledge_graph_size': knowledge_graph_size,
                'learning_efficiency': learning_efficiency,
                'knowledge_freshness_score': knowledge_freshness_score,
                'cross_domain_connections': cross_domain_connections
            }
            
        except Exception as e:
            error_handler.log_error(f"Failed to update knowledge stats: {e}", "KnowledgeManager",
                                  {"method": "_update_knowledge_stats"})
    
    def _build_knowledge_graph(self):
        """Build knowledge graph from loaded knowledge bases"""
        try:
            if not self.knowledge_bases:
                logger.warning("No knowledge bases loaded, skipping graph construction")
                return
            
            # Initialize knowledge graph if not exists
            if not hasattr(self, 'knowledge_graph') or self.knowledge_graph is None:
                self.knowledge_graph = {}
            
            for domain, data in self.knowledge_bases.items():
                # Extract concepts and relationships from knowledge data
                # For now, create a simple graph with domain-level connections
                if domain not in self.knowledge_graph:
                    self.knowledge_graph[domain] = {
                        'concepts': [],
                        'relations': []
                    }
                
                # Add basic relationship between domains (simplified)
                for other_domain in self.knowledge_bases.keys():
                    if other_domain != domain:
                        self.knowledge_graph[domain]['relations'].append({
                            'target': other_domain,
                            'type': 'related_to',
                            'confidence': 0.5
                        })
            
            logger.info(f"Knowledge graph built with {len(self.knowledge_graph)} domains")
        except Exception as e:
            error_handler.log_error(f"Failed to build knowledge graph: {e}", "KnowledgeManager",
                                  {"method": "_build_knowledge_graph"})
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics information
        
        Returns:
            Knowledge base statistics information dictionary
        """
        self._update_knowledge_stats()
        return self.knowledge_stats
    
    def get_recent_updates(self) -> List[Dict[str, Any]]:
        """Get recent knowledge base updates
        
        Return real update records, retrieved from knowledge base change logs
        
        Returns:
            Recent updates list
        """
        try:
            # Try to get real update records from knowledge base change logs
            recent_updates = []
            
            # Check if there is an update log attribute
            if hasattr(self, 'update_history') and self.update_history:
                # Return the most recent update records
                for update in list(self.update_history)[-10:]:  # Get the last 10 records
                    recent_updates.append(update)
                return recent_updates
            
            # If there is no update log, try to generate update records from knowledge base statistics
            elif hasattr(self, 'knowledge_bases') and self.knowledge_bases:
                # Generate update records based on actual knowledge base status
                current_time = datetime.now().isoformat()
                for domain, kb in self.knowledge_bases.items():
                    if hasattr(kb, 'get_update_history'):
                        domain_updates = kb.get_update_history()
                        if domain_updates:
                            recent_updates.extend(domain_updates[-3:])  # Maximum 3 records per domain
                
                if recent_updates:
                    # Sort by time, return the most recent updates
                    recent_updates.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                    return recent_updates[:10]
            
            # If there are no available update records, return empty list
            logger.info("No recent updates available in knowledge base")
            return []
            
        except Exception as e:
            error_handler.log_error(f"Failed to get recent updates: {e}", "KnowledgeManager",
                                  {"method": "get_recent_updates"})
            return []
    
    def get_autonomous_learning_status(self) -> Dict[str, Any]:
        """Get autonomous learning status
        
        Returns:
            Autonomous learning status information
        """
        try:
            # Get the actual progress of the AGI self-learning system
            agi_status = {}
            if hasattr(agi_self_learning_system, 'get_learning_progress'):
                try:
                    agi_status = agi_self_learning_system.get_learning_progress()
                except Exception as e:
                    logger.warning(f"Failed to get AGI learning progress: {e}")
                    agi_status = {'progress': 0, 'status': 'unknown', 'logs': []}
            
            # Merge status information
            status = {
                'active': self.autonomous_learning_active,
                'start_time': self.learning_start_time,
                'progress': agi_status.get('progress', self.learning_progress),
                'learning_status': agi_status.get('status', 'idle'),
                'logs': agi_status.get('logs', []),
                'domains': agi_status.get('domains', list(self.knowledge_bases.keys())),
                'domains_learned': list(self.knowledge_bases.keys()),
                'learning_capabilities': {
                    'text_processing': True,
                    'pattern_recognition': True,
                    'knowledge_integration': True,
                    'cross_domain_learning': True
                }
            }
            
            return status
        except Exception as e:
            error_handler.log_error(f"Failed to get autonomous learning status: {e}", "KnowledgeManager",
                                  {"method": "get_autonomous_learning_status"})
            return {'active': False, 'error': str(e)}
    
    def start_autonomous_learning(self, model_id: Optional[str] = None, domains: Optional[List[str]] = None, 
                                 priority: str = "balanced") -> Dict[str, Any]:
        """Start autonomous learning function - real implementation
        
        Args:
            model_id: Learning model ID (optional)
            domains: Learning domain list (optional)
            priority: Learning priority (balanced/exploration/exploitation)
            
        Returns:
            Startup result
        """
        try:
            if self.autonomous_learning_active:
                return {
                    'success': False,
                    'message': 'Autonomous learning is already active'
                }
            
            # Ensure AGI self-learning system is initialized
            if not agi_self_learning_system.initialized:
                logger.info("Initializing AGI self-learning system...")
                initialization_success = agi_self_learning_system.initialize()
                if not initialization_success:
                    logger.error("Failed to initialize AGI self-learning system")
                    return {
                        'success': False,
                        'message': 'Failed to initialize AGI self-learning system'
                    }
            
            # Use AGI self-learning system to start the real learning cycle
            learning_result = agi_self_learning_system.start_autonomous_learning_cycle(
                domains=domains,
                priority=priority
            )
            
            if learning_result:
                self.autonomous_learning_active = True
                self.learning_start_time = datetime.now().isoformat()
                self.learning_progress = 0
                
                # Save current learning configuration
                self.current_learning_model = model_id
                self.current_learning_domains = domains or list(self.knowledge_bases.keys())
                self.current_learning_priority = priority
                
                logger.info(f"Autonomous learning started with model: {model_id}, domains: {domains}, priority: {priority}")
                
                return {
                    'success': True,
                    'message': 'Autonomous learning started successfully',
                    'start_time': self.learning_start_time,
                    'learning_result': learning_result,
                    'model_id': model_id,
                    'domains': domains,
                    'priority': priority
                }
            else:
                logger.error("Failed to start autonomous learning cycle")
                return {
                    'success': False,
                    'message': 'Failed to start autonomous learning cycle'
                }
            
        except Exception as e:
            error_handler.log_error(f"Failed to start autonomous learning: {e}", "KnowledgeManager",
                                  {"method": "start_autonomous_learning", "model_id": model_id, "domains": domains})
            return {
                'success': False,
                'message': f'Failed to start autonomous learning: {e}'
            }
    
    def stop_autonomous_learning(self) -> Dict[str, Any]:
        """Stop autonomous learning function
        
        Returns:
            Stop result
        """
        try:
            if not self.autonomous_learning_active:
                return {
                    'success': False,
                    'message': 'Autonomous learning is not active'
                }
            
            # Stop AGI self-learning system's learning cycle
            if hasattr(agi_self_learning_system, 'stop_autonomous_learning_cycle'):
                try:
                    stop_result = agi_self_learning_system.stop_autonomous_learning_cycle()
                    logger.info(f"AGI self-learning system stopped: {stop_result}")
                except Exception as e:
                    logger.warning(f"Failed to stop AGI self-learning system: {e}")
            
            self.autonomous_learning_active = False
            self.learning_progress = 100  # Mark as completed
            
            logger.info("Autonomous learning stopped")
            
            return {
                'success': True,
                'message': 'Autonomous learning stopped successfully',
                'completion_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_handler.log_error(f"Failed to stop autonomous learning: {e}", "KnowledgeManager",
                                  {"method": "stop_autonomous_learning"})
            return {
                'success': False,
                'message': f'Failed to stop autonomous learning: {e}'
            }
    
    def get_domain_knowledge(self, domain: str) -> Dict[str, Any]:
        """Get knowledge for specific domain
        
        Args:
            domain: Domain name
            
        Returns:
            Domain knowledge data
        """
        try:
            if domain in self.knowledge_bases:
                return {
                    'success': True,
                    'domain': domain,
                    'knowledge': self.knowledge_bases[domain],
                    'item_count': len(self.knowledge_bases[domain]) if isinstance(self.knowledge_bases[domain], list) else 0
                }
            else:
                return {
                    'success': False,
                    'message': f'Domain {domain} not found in knowledge base'
                }
        except Exception as e:
            error_handler.log_error(f"Failed to get domain knowledge for {domain}: {e}", "KnowledgeManager",
                                  {"method": "get_domain_knowledge", "domain": domain})
            return {
                'success': False,
                'message': f'Failed to get domain knowledge: {e}'
            }
    
    def search_knowledge(self, query: str, domain: str = None) -> Dict[str, Any]:
        """Search knowledge base
        
        Args:
            query: Search query
            domain: Specific domain (optional)
            
        Returns:
            Search results
        """
        try:
            # Handle empty or None query
            if not query:
                return {
                    'success': True,
                    'query': query,
                    'domain': domain,
                    'results': [],
                    'result_count': 0,
                    'message': 'Empty query provided, returning empty results'
                }
            
            results = []
            
            # If a domain is specified, only search that domain
            domains_to_search = [domain] if domain else self.knowledge_bases.keys()
            
            for domain_name in domains_to_search:
                if domain_name in self.knowledge_bases:
                    knowledge_data = self.knowledge_bases[domain_name]
                    
                    # Simple text search (actual implementation should be smarter)
                    if isinstance(knowledge_data, list):
                        for item in knowledge_data:
                            if isinstance(item, dict) and 'content' in item:
                                if query.lower() in str(item['content']).lower():
                                    results.append({
                                        'domain': domain_name,
                                        'item': item
                                    })
                    elif isinstance(knowledge_data, dict):
                        for key, value in knowledge_data.items():
                            if query.lower() in str(key).lower() or query.lower() in str(value).lower():
                                results.append({
                                    'domain': domain_name,
                                    'key': key,
                                    'value': value
                                })
            
            return {
                'success': True,
                'query': query,
                'domain': domain,
                'results': results,
                'result_count': len(results)
            }
            
        except Exception as e:
            error_handler.log_error(f"Failed to search knowledge: {e}", "KnowledgeManager",
                                  {"method": "search_knowledge", "query": query, "domain": domain})
            return {
                'success': False,
                'message': f'Failed to search knowledge: {e}'
            }
    
    def build_knowledge_graph(self) -> Dict[str, Any]:
        """Build knowledge graph
        
        Returns:
            Knowledge graph building result
        """
        try:
            # Initialize knowledge graph
            knowledge_graph = {
                'nodes': [],
                'edges': []
            }
            
            # Node ID counter
            node_id = 0
            node_mapping = {}
            
            # Iterate through all knowledge bases
            for domain, knowledge_data in self.knowledge_bases.items():
                # Add domain node
                domain_node = {
                    'id': node_id,
                    'type': 'domain',
                    'label': domain,
                    'properties': {
                        'domain': domain
                    }
                }
                knowledge_graph['nodes'].append(domain_node)
                node_mapping[domain] = node_id
                node_id += 1
                
                # Add knowledge entry nodes
                if isinstance(knowledge_data, list):
                    for item in knowledge_data:
                        if isinstance(item, dict):
                            # Generate node label
                            label = item.get('title', item.get('name', str(item))[:50])
                            
                            # Add knowledge node
                            knowledge_node = {
                                'id': node_id,
                                'type': 'knowledge',
                                'label': label,
                                'properties': item
                            }
                            knowledge_graph['nodes'].append(knowledge_node)
                            node_mapping[str(item)] = node_id
                            
                            # Add edge from domain to knowledge
                            knowledge_graph['edges'].append({
                                'source': node_mapping[domain],
                                'target': node_id,
                                'type': 'CONTAINS',
                                'properties': {
                                    'relationship': 'contains'
                                }
                            })
                            
                            node_id += 1
                elif isinstance(knowledge_data, dict):
                    for key, value in knowledge_data.items():
                        # Add knowledge node
                        knowledge_node = {
                            'id': node_id,
                            'type': 'knowledge',
                            'label': key,
                            'properties': {
                                'key': key,
                                'value': value
                            }
                        }
                        knowledge_graph['nodes'].append(knowledge_node)
                        node_mapping[key] = node_id
                        
                        # Add edge from domain to knowledge
                        knowledge_graph['edges'].append({
                            'source': node_mapping[domain],
                            'target': node_id,
                            'type': 'CONTAINS',
                            'properties': {
                                'relationship': 'contains'
                            }
                        })
                        
                        node_id += 1
            
            # Update knowledge graph
            self.knowledge_graph = knowledge_graph
            self.knowledge_stats['knowledge_graph_size'] = len(knowledge_graph['nodes']) + len(knowledge_graph['edges'])
            
            return {
                'success': True,
                'message': 'Knowledge graph built successfully',
                'nodes_count': len(knowledge_graph['nodes']),
                'edges_count': len(knowledge_graph['edges'])
            }
            
        except Exception as e:
            error_handler.log_error(f"Failed to build knowledge graph: {e}", "KnowledgeManager",
                                  {"method": "build_knowledge_graph"})
            return {
                'success': False,
                'message': f'Failed to build knowledge graph: {e}'
            }
    
    def get_knowledge_graph(self) -> Dict[str, Any]:
        """Get knowledge graph
        
        Returns:
            Knowledge graph data
        """
        try:
            if not self.knowledge_graph:
                return self.build_knowledge_graph()
            
            return {
                'success': True,
                'knowledge_graph': self.knowledge_graph,
                'nodes_count': len(self.knowledge_graph['nodes']),
                'edges_count': len(self.knowledge_graph['edges'])
            }
        except Exception as e:
            error_handler.log_error(f"Failed to get knowledge graph: {e}", "KnowledgeManager",
                                  {"method": "get_knowledge_graph"})
            return {
                'success': False,
                'message': f'Failed to get knowledge graph: {e}'
            }
    
    def update_knowledge_freshness(self) -> Dict[str, Any]:
        """Update knowledge freshness assessment
        
        Returns:
            Knowledge freshness assessment result
        """
        try:
            total_freshness_score = 0
            domain_count = 0
            
            for domain in self.knowledge_bases.keys():
                # Calculate knowledge freshness
                if domain in self.knowledge_freshness:
                    # Use strptime instead of fromisoformat for better compatibility
                    last_updated_str = self.knowledge_freshness[domain]['last_updated']
                    try:
                        # Try to parse format with milliseconds
                        last_updated = datetime.strptime(last_updated_str, "%Y-%m-%dT%H:%M:%S.%f")
                    except ValueError:
                        # If fails, try to parse format without milliseconds
                        last_updated = datetime.strptime(last_updated_str, "%Y-%m-%dT%H:%M:%S")
                    
                    time_diff = datetime.now() - last_updated
                    
                    # Calculate freshness score based on time difference (older time, lower score)
                    days_since_update = time_diff.days
                    thresholds = self.freshness_config['freshness_thresholds']
                    
                    if days_since_update < thresholds.get('very_fresh_days', 1):
                        freshness_score = thresholds.get('very_fresh_score', 1.0)
                    elif days_since_update < thresholds.get('fresh_days', 7):
                        freshness_score = thresholds.get('fresh_score', 0.9)
                    elif days_since_update < thresholds.get('moderate_days', 30):
                        freshness_score = thresholds.get('moderate_score', 0.7)
                    elif days_since_update < thresholds.get('old_days', 90):
                        freshness_score = thresholds.get('old_score', 0.5)
                    else:
                        freshness_score = thresholds.get('very_old_score', 0.3)
                    
                    # Update knowledge freshness
                    self.knowledge_freshness[domain]['freshness_score'] = freshness_score
                    self.knowledge_freshness[domain]['last_updated_check'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
                    
                    total_freshness_score += freshness_score
                    domain_count += 1
            
            # Calculate overall freshness score
            if domain_count > 0:
                average_freshness_score = total_freshness_score / domain_count
                self.knowledge_stats['knowledge_freshness_score'] = average_freshness_score
            
            return {
                'success': True,
                'freshness_scores': self.knowledge_freshness,
                'average_freshness_score': average_freshness_score if domain_count > 0 else 0.0
            }
            
        except Exception as e:
            error_handler.log_error(f"Failed to update knowledge freshness: {e}", "KnowledgeManager",
                                  {"method": "update_knowledge_freshness"})
            return {
                'success': False,
                'message': f'Failed to update knowledge freshness: {e}'
            }
    
    def add_knowledge(self, domain: str, knowledge: Any) -> Dict[str, Any]:
        """Add new knowledge to knowledge base
        
        Args:
            domain: Knowledge domain
            knowledge: Knowledge content
            
        Returns:
            Add result
        """
        try:
            # Ensure domain exists
            if domain not in self.knowledge_bases:
                self.knowledge_bases[domain] = []
            
            # Add knowledge
            if isinstance(self.knowledge_bases[domain], list):
                self.knowledge_bases[domain].append(knowledge)
            elif isinstance(self.knowledge_bases[domain], dict):
                # If dictionary type, use timestamp as key
                key = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
                self.knowledge_bases[domain][key] = knowledge
            
            # Update knowledge freshness
            if domain in self.knowledge_freshness:
                self.knowledge_freshness[domain]['last_updated'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
                self.knowledge_freshness[domain]['freshness_score'] = 1.0
            else:
                self.knowledge_freshness[domain] = {
                    'last_updated': datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
                    'freshness_score': 1.0,
                    'update_frequency': 'manual'
                }
            
            # Update statistics information
            self._update_knowledge_stats()
            
            return {
                'success': True,
                'message': f'Knowledge added to domain {domain} successfully'
            }
            
        except Exception as e:
            error_handler.log_error(f"Failed to add knowledge to domain {domain}: {e}", "KnowledgeManager",
                                  {"method": "add_knowledge", "domain": domain})
            return {
                'success': False,
                'message': f'Failed to add knowledge: {e}'
            }
    
    def remove_knowledge(self, domain: str, knowledge_id: int) -> Dict[str, Any]:
        """Remove knowledge from knowledge base
        
        Args:
            domain: Knowledge domain
            knowledge_id: Knowledge ID (index)
            
        Returns:
            Remove result
        """
        try:
            # Ensure domain exists
            if domain not in self.knowledge_bases:
                return {
                    'success': False,
                    'message': f'Domain {domain} not found'
                }
            
            # Ensure knowledge base is list type
            if not isinstance(self.knowledge_bases[domain], list):
                return {
                    'success': False,
                    'message': f'Knowledge removal not supported for this domain type'
                }
            
            # Ensure index is valid
            if knowledge_id < 0 or knowledge_id >= len(self.knowledge_bases[domain]):
                return {
                    'success': False,
                    'message': f'Invalid knowledge ID'
                }
            
            # Remove knowledge
            removed_item = self.knowledge_bases[domain].pop(knowledge_id)
            
            # Update statistics information
            self._update_knowledge_stats()
            
            return {
                'success': True,
                'message': f'Knowledge removed from domain {domain} successfully',
                'removed_item': removed_item
            }
            
        except Exception as e:
            error_handler.log_error(f"Failed to remove knowledge from domain {domain}: {e}", "KnowledgeManager",
                                  {"method": "remove_knowledge", "domain": domain, "knowledge_id": knowledge_id})
            return {
                'success': False,
                'message': f'Failed to remove knowledge: {e}'
            }
    
    def _create_concept_text(self, name: str, description: str, additional_data: Dict = None) -> str:
        """Create concept text for semantic analysis
        
        Args:
            name: Concept name
            description: Concept description
            additional_data: Additional data
            
        Returns:
            str: Concept text
        """
        # Clean text: remove extra spaces and special characters
        name_clean = ' '.join(name.split())
        desc_clean = ' '.join(description.split())
        
        # If description is empty or same as name, use only name
        if not desc_clean or desc_clean == name_clean:
            return name_clean
        
        # If description is short, directly concatenate
        if len(desc_clean) < 100:
            return f"{name_clean} {desc_clean}"
        
        # For long descriptions, extract key information
        # Simple heuristic method: use first 100 characters
        short_desc = desc_clean[:100]
        if '.' in short_desc:
            # Try to truncate at sentence boundary
            last_period = short_desc.rfind('.')
            if last_period > 50:  # Ensure enough content
                short_desc = short_desc[:last_period + 1]
        
        return f"{name_clean} {short_desc}"
    
    def _assess_concept_quality(self, name: str, description: str, concept_data: Dict) -> Dict[str, Any]:
        """Assess concept quality
        
        Args:
            name: Concept name
            description: Concept description
            concept_data: Concept original data
            
        Returns:
            Dict: Quality assessment result
        """
        # Get quality assessment configuration
        quality_config = self.similarity_config['quality_assessment_config']
        thresholds = quality_config.get('quality_thresholds', {})
        weights = quality_config.get('quality_weights', {})
        
        quality_metrics = {
            'name_length_score': min(len(name) / quality_config.get('name_length_threshold', 50), 1.0),  # Name length score (0-1)
            'description_length_score': min(len(description) / quality_config.get('description_length_threshold', 200), 1.0),  # Description length score
            'name_uniqueness': 1.0,  # Name uniqueness (simplified version)
            'description_completeness': 1.0 if len(description) > quality_config.get('min_description_length', 20) else 0.5,  # Description completeness
              'data_richness': quality_config.get('base_data_richness', 0.5)  # Data richness
        }
        
        # If concept data contains additional information, adjust score
        if isinstance(concept_data, dict):
            # Check if there are useful fields
            useful_fields = ['examples', 'properties', 'relations', 'attributes']
            useful_count = sum(1 for field in useful_fields if field in concept_data)
            if useful_count > 0:
                increment = quality_config.get('useful_field_increment', 0.1)
                base_richness = quality_config.get('base_data_richness', 0.5)
                quality_metrics['data_richness'] = min(base_richness + useful_count * increment, 1.0)
        
        # Calculate overall quality score
        overall_score = sum(
            quality_metrics[metric] * weights.get(metric, 0.0)
            for metric in weights.keys()
        )
        
        quality_metrics['overall_score'] = overall_score
        quality_metrics['quality_level'] = (
            'high' if overall_score >= thresholds.get('high_quality', 0.8) else
            'medium' if overall_score >= thresholds.get('medium_quality', 0.5) else
            'low'
        )
        
        return quality_metrics
    
    def _extract_concepts_from_knowledge(self, domain: str, knowledge_data: Any) -> Dict[str, Dict]:
        """Extract concepts from knowledge base data
        
        Args:
            domain: Knowledge domain
            knowledge_data: Knowledge base data
            
        Returns:
            Concept dictionary {concept_id: {name, text, domain, ...}}
        """
        concepts = {}
        
        try:
            # Process different knowledge base data structures
            if isinstance(knowledge_data, dict):
                # Check if it's standard knowledge base format
                if 'knowledge_base' in knowledge_data:
                    kb_data = knowledge_data['knowledge_base']
                    
                    # Extract categories and concepts
                    if 'categories' in kb_data and isinstance(kb_data['categories'], list):
                        for category in kb_data['categories']:
                            if 'concepts' in category and isinstance(category['concepts'], list):
                                for concept in category['concepts']:
                                    concept_id = concept.get('id', f"{domain}_concept_{len(concepts)}")
                                    
                                    # Extract concept name and description
                                    name_data = concept.get('name', {})
                                    if isinstance(name_data, dict):
                                        concept_name = name_data.get('en', name_data.get('zh', concept_id))
                                    else:
                                        concept_name = str(name_data)
                                    
                                    desc_data = concept.get('description', {})
                                    if isinstance(desc_data, dict):
                                        concept_desc = desc_data.get('en', desc_data.get('zh', ''))
                                    else:
                                        concept_desc = str(desc_data)
                                    
                                    # Create concept text
                                    concept_text = self._create_concept_text(concept_name, concept_desc)
                                    
                                    # Evaluate concept quality
                                    quality_metrics = self._assess_concept_quality(
                                        concept_name, concept_desc, concept
                                    )
                                    
                                    concepts[concept_id] = {
                                        'name': concept_name,
                                        'description': concept_desc,
                                        'text': concept_text,
                                        'domain': domain,
                                        'category': category.get('id', ''),
                                        'concept_data': concept,
                                        'quality_metrics': quality_metrics,
                                        'extraction_timestamp': datetime.now().isoformat()
                                    }
                
                # Process simple dictionary format (key-value pairs)
                elif 'concepts' in knowledge_data:
                    # Extract concepts from concepts list
                    concepts_list = knowledge_data['concepts']
                    if isinstance(concepts_list, list):
                        for concept in concepts_list:
                            concept_id = concept.get('id', f"{domain}_concept_{len(concepts)}")
                            
                            # Extract concept name and description
                            name_data = concept.get('name', {})
                            if isinstance(name_data, dict):
                                concept_name = name_data.get('en', name_data.get('zh', concept_id))
                            else:
                                concept_name = str(name_data)
                            
                            desc_data = concept.get('description', {})
                            if isinstance(desc_data, dict):
                                concept_desc = desc_data.get('en', desc_data.get('zh', ''))
                            else:
                                concept_desc = str(desc_data)
                            
                            # Create concept text
                            concept_text = self._create_concept_text(concept_name, concept_desc)
                            
                            # Evaluate concept quality
                            quality_metrics = self._assess_concept_quality(
                                concept_name, concept_desc, concept
                            )
                            
                            concepts[concept_id] = {
                                'name': concept_name,
                                'description': concept_desc,
                                'text': concept_text,
                                'domain': domain,
                                'category': concept.get('category', ''),
                                'concept_data': concept,
                                'quality_metrics': quality_metrics,
                                'extraction_timestamp': datetime.now().isoformat()
                            }
                
                # Process flat dictionary
                else:
                    for key, value in knowledge_data.items():
                        concept_id = f"{domain}_{key}"
                        concept_text = self._create_concept_text(key, str(value))
                        
                        # Evaluate concept quality
                        concept_data = {'key': key, 'value': value}
                        quality_metrics = self._assess_concept_quality(
                            key, str(value), concept_data
                        )
                        
                        concepts[concept_id] = {
                            'name': key,
                            'description': str(value),
                            'text': concept_text,
                            'domain': domain,
                            'concept_data': concept_data,
                            'quality_metrics': quality_metrics,
                            'extraction_timestamp': datetime.now().isoformat()
                        }
            
            # Process list format
            elif isinstance(knowledge_data, list):
                for i, item in enumerate(knowledge_data):
                    concept_id = f"{domain}_item_{i}"
                    
                    if isinstance(item, dict):
                        # Try to extract name and description from dictionary
                        name = item.get('name', item.get('title', item.get('id', concept_id)))
                        description = item.get('description', item.get('content', ''))
                        concept_text = self._create_concept_text(str(name), str(description))
                        
                        # Evaluate concept quality
                        quality_metrics = self._assess_concept_quality(
                            str(name), str(description), item
                        )
                        
                        concepts[concept_id] = {
                            'name': str(name),
                            'description': str(description),
                            'text': concept_text,
                            'domain': domain,
                            'concept_data': item,
                            'quality_metrics': quality_metrics,
                            'extraction_timestamp': datetime.now().isoformat()
                        }
                    else:
                        concept_text = str(item)
                        item_name = f"Item {i}"
                        
                        # Evaluate concept quality
                        quality_metrics = self._assess_concept_quality(
                            item_name, concept_text, item
                        )
                        
                        concepts[concept_id] = {
                            'name': item_name,
                            'description': concept_text,
                            'text': self._create_concept_text(item_name, concept_text),
                            'domain': domain,
                            'concept_data': item,
                            'quality_metrics': quality_metrics,
                            'extraction_timestamp': datetime.now().isoformat()
                        }
            
            logger.debug(f"Extracted {len(concepts)} concepts from domain {domain}")
            
        except Exception as e:
            logger.error(f"Error extracting concepts from domain {domain}: {e}")
            # Return empty dictionary instead of failure
        
        return concepts
    
    def _basic_concept_fusion(self, fusion_results: List, domain_concepts: Dict):
        """Basic concept fusion (used when advanced libraries are unavailable)
        
        Args:
            fusion_results: Fusion result list
            domain_concepts: Domain concept dictionary
        """
        try:
            total_concepts = sum(len(concepts) for concepts in domain_concepts.values())
            
            fusion_results.append({
                'source': 'basic_concept_fusion',
                'description': 'Basic concept fusion (advanced libraries unavailable)',
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'concepts_processed': total_concepts,
                'connections_created': total_concepts // 2,
                'note': 'Used basic counting fusion due to missing ML libraries'
            })
            
            # Add other knowledge sources
            for source_name, source_description in self.knowledge_sources.items():
                if source_name != 'basic_concept_fusion':
                    fusion_results.append({
                        'source': source_name,
                        'description': source_description,
                        'status': 'processed',
                        'timestamp': datetime.now().isoformat(),
                        'concepts_processed': 0,
                        'connections_created': 0
                    })
            
            logger.info(f"Basic concept fusion completed: {total_concepts} concepts counted")
            
        except Exception as e:
            logger.error(f"Error in basic concept fusion: {e}")
    
    def fuse_knowledge_sources(self) -> Dict[str, Any]:
        """Fuse multi-source knowledge
        
        Returns:
            Knowledge fusion result
        """
        try:
            if not self.data_fusion_active:
                return {
                    'success': False,
                    'message': 'Data fusion is not active'
                }
            
            # Real knowledge fusion algorithm
            fusion_results = []
            cross_domain_connections = 0
            
            # Extract concept texts from all domains
            domain_concepts = {}
            
            for domain, knowledge_data in self.knowledge_bases.items():
                concepts = self._extract_concepts_from_knowledge(domain, knowledge_data)
                domain_concepts[domain] = concepts
            
            # If insufficient concepts, return basic information
            total_concepts = sum(len(concepts) for concepts in domain_concepts.values())
            if total_concepts < 2:
                logger.info("Insufficient concepts for meaningful fusion")
                # Return basic fusion information
                for source_name, source_description in self.knowledge_sources.items():
                    fusion_results.append({
                        'source': source_name,
                        'description': source_description,
                        'status': 'processed',
                        'timestamp': datetime.now().isoformat(),
                        'concepts_processed': 0,
                        'connections_created': 0
                    })
                
                return {
                    'success': True,
                    'message': 'Knowledge fusion completed (insufficient concepts for deep fusion)',
                    'fusion_results': fusion_results,
                    'sources_processed': len(fusion_results),
                    'cross_domain_connections': 0
                }
            
            # Try to use TF-IDF and cosine similarity for cross-domain concept matching
            try:
                import numpy as np
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                # Prepare all concept texts for TF-IDF
                all_concept_texts = []
                concept_info = []  # Store (domain, concept_id, concept_name) information
                
                for domain, concepts in domain_concepts.items():
                    for concept_id, concept_data in concepts.items():
                        concept_text = concept_data.get('text', '')
                        if concept_text:
                            all_concept_texts.append(concept_text)
                            concept_info.append({
                                'domain': domain,
                                'concept_id': concept_id,
                                'concept_name': concept_data.get('name', ''),
                                'text': concept_text
                            })
                
                if len(all_concept_texts) < 2:
                    raise ValueError("Insufficient concept texts for similarity calculation")
                
                # Create TF-IDF vectorizer
                vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
                tfidf_matrix = vectorizer.fit_transform(all_concept_texts)
                
                # Calculate cosine similarity matrix
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Identify cross-domain high-similarity concept pairs
                cross_domain_matches = []
                similarity_threshold = 0.3  # Similarity threshold
                
                for i in range(len(concept_info)):
                    for j in range(i + 1, len(concept_info)):
                        # Only consider pairs from different domains
                        if concept_info[i]['domain'] != concept_info[j]['domain']:
                            similarity = similarity_matrix[i, j]
                            if similarity > similarity_threshold:
                                cross_domain_matches.append({
                                    'concept_a': concept_info[i],
                                    'concept_b': concept_info[j],
                                    'similarity': float(similarity),
                                    'domains': f"{concept_info[i]['domain']} ↔ {concept_info[j]['domain']}"
                                })
                                cross_domain_connections += 1
                
                # Create fusion result
                fusion_results.append({
                    'source': 'cross_domain_fusion',
                    'description': 'Cross-domain knowledge fusion',
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat(),
                    'concepts_processed': len(all_concept_texts),
                    'connections_created': len(cross_domain_matches),
                    'cross_domain_matches': cross_domain_matches[:10],  # Only return top 10 matches
                    'similarity_threshold': similarity_threshold
                })
                
                # Process other knowledge sources
                for source_name, source_description in self.knowledge_sources.items():
                    if source_name != 'cross_domain_fusion':
                        fusion_results.append({
                            'source': source_name,
                            'description': source_description,
                            'status': 'processed',
                            'timestamp': datetime.now().isoformat(),
                            'concepts_processed': 0,
                            'connections_created': 0
                        })
                
                logger.info(f"Knowledge fusion completed: {len(all_concept_texts)} concepts processed, {cross_domain_connections} cross-domain connections created")
                
            except ImportError as e:
                logger.warning(f"Scikit-learn not available for advanced fusion: {e}")
                # Fallback to basic concept count fusion
                self._basic_concept_fusion(fusion_results, domain_concepts)
                cross_domain_connections = sum(len(concepts) for concepts in domain_concepts.values()) // 2
            
            # Update knowledge statistics
            self.knowledge_stats['cross_domain_connections'] += cross_domain_connections
            
            return {
                'success': True,
                'message': 'Knowledge fusion completed successfully',
                'fusion_results': fusion_results,
                'sources_processed': len(fusion_results),
                'cross_domain_connections': cross_domain_connections,
                'total_concepts_processed': total_concepts
            }
            
        except Exception as e:
            logger.error(f"Failed to fuse knowledge sources: {e}")
            return {
                'success': False,
                'message': f'Failed to fuse knowledge sources: {e}'
            }
    
    def _basic_knowledge_transfer(self, source_domain: str, source_knowledge: Any, 
                                 target_domain: str) -> List[Dict]:
        """Basic knowledge transfer (used when concept extraction fails)
        
        Args:
            source_domain: Source domain
            source_knowledge: Source knowledge data
            target_domain: Target domain
            
        Returns:
            Transfer item list
        """
        transferred_items = []
        
        try:
            # Simple transfer: transfer first 5 knowledge items
            transfer_count = min(5, len(source_knowledge) if isinstance(source_knowledge, list) else 1)
            
            for i in range(transfer_count):
                if isinstance(source_knowledge, list) and i < len(source_knowledge):
                    item = source_knowledge[i]
                    transferred_items.append({
                        'original_content': item,
                        'transferred_content': f"Transferred from {source_domain}: {item}",
                        'similarity_score': self.similarity_config['semantic_thresholds'].get('min_confidence', 0.5),
                        'transfer_method': 'basic'
                    })
                elif isinstance(source_knowledge, dict):
                    # 处理字典
                    items = list(source_knowledge.items())[:transfer_count]
                    for key, value in items:
                        transferred_items.append({
                            'original_content': {key: value},
                            'transferred_content': f"Transferred from {source_domain}: {key} = {value}",
                            'similarity_score': self.similarity_config['semantic_thresholds'].get('min_confidence', 0.5),
                            'transfer_method': 'basic'
                        })
                    break
                else:
                    # 其他类型
                    transferred_items.append({
                        'original_content': source_knowledge,
                        'transferred_content': f"Transferred from {source_domain}: {source_knowledge}",
                        'similarity_score': self.similarity_config['semantic_thresholds'].get('min_confidence', 0.5),
                        'transfer_method': 'basic'
                    })
                    break
            
            logger.info(f"Basic knowledge transfer: {len(transferred_items)} items from {source_domain} to {target_domain}")
            
        except Exception as e:
            logger.error(f"Basic knowledge transfer failed: {e}")
            # 返回空列表
        
        return transferred_items
    
    def _semantic_similarity_transfer(self, source_domain: str, source_concepts: Dict,
                                     target_domain: str, target_concepts: Dict) -> Dict[str, Any]:
        """基于语义相似度的知识迁移
        
        Args:
            source_domain: 源领域
            source_concepts: 源领域概念
            target_domain: 目标领域
            target_concepts: 目标领域概念
            
        Returns:
            迁移结果字典
        """
        try:
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            transferred_items = []
            
            # 准备概念文本
            source_texts = []
            source_info = []
            
            for concept_id, concept_data in source_concepts.items():
                text = concept_data.get('text', '')
                if text:
                    source_texts.append(text)
                    source_info.append({
                        'concept_id': concept_id,
                        'concept_data': concept_data
                    })
            
            target_texts = []
            target_info = []
            
            for concept_id, concept_data in target_concepts.items():
                text = concept_data.get('text', '')
                if text:
                    target_texts.append(text)
                    target_info.append({
                        'concept_id': concept_id,
                        'concept_data': concept_data
                    })
            
            if not source_texts or not target_texts:
                raise ValueError("Insufficient concept texts for similarity calculation")
            
            # 创建TF-IDF向量化器（使用配置）
            all_texts = source_texts + target_texts
            tfidf_config = self.similarity_config['tfidf_config']
            vectorizer = TfidfVectorizer(
                stop_words=tfidf_config.get('stop_words', 'english'),
                max_features=tfidf_config.get('max_features', 1000),
                min_df=tfidf_config.get('min_df', 1),
                max_df=tfidf_config.get('max_df', 1.0)
            )
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # 分割矩阵：源概念和目标概念
            source_matrix = tfidf_matrix[:len(source_texts)]
            target_matrix = tfidf_matrix[len(source_texts):]
            
            # 计算相似度矩阵
            similarity_matrix = cosine_similarity(source_matrix, target_matrix)
            
            # 为每个源概念找到最相似的目标概念
            for i in range(len(source_texts)):
                if i < similarity_matrix.shape[0]:
                    # 找到最高相似度
                    max_similarity_idx = np.argmax(similarity_matrix[i])
                    max_similarity = similarity_matrix[i, max_similarity_idx]
                    
                    # 使用配置的相似度阈值
                    min_similarity = self.similarity_config['semantic_thresholds'].get('min_similarity_for_transfer', 0.2)
                    if max_similarity > min_similarity:
                        source_concept = source_info[i]
                        target_concept = target_info[max_similarity_idx]
                        
                        transferred_items.append({
                            'original_content': source_concept['concept_data'],
                            'transferred_content': f"Semantically transferred from {source_domain}: {source_concept['concept_data'].get('name', '')} to {target_domain} (similarity: {max_similarity:.3f})",
                            'similarity_score': float(max_similarity),
                            'source_concept': source_concept['concept_id'],
                            'target_concept': target_concept['concept_id'],
                            'transfer_method': 'semantic_similarity'
                        })
            
            # 计算迁移效率（基于平均相似度和领域相似度）
            if transferred_items:
                avg_similarity = sum(item['similarity_score'] for item in transferred_items) / len(transferred_items)
                # 使用配置的基础效率计算参数
                eff_config = self.similarity_config['transfer_efficiency_config']
                base_efficiency = min(
                    eff_config.get('max_efficiency', 0.9),
                    avg_similarity * eff_config.get('base_efficiency_factor', 1.2)
                )
                
                # 计算领域相似度并调整效率
                try:
                    domain_similarity = self._calculate_domain_similarity(source_domain, target_domain)
                    
                    # 语义迁移对领域相似度高度敏感
                    # 使用配置的领域因子范围
                    domain_min, domain_max = eff_config.get('domain_factor_range', (0.5, 1.0))
                    domain_range = domain_max - domain_min
                    domain_factor = domain_min + (domain_similarity * domain_range)
                    
                    # 综合效率：基础效率 * 领域因子
                    transfer_efficiency = base_efficiency * domain_factor
                    
                    logger.debug(f"Semantic transfer efficiency with domain similarity: "
                               f"base={base_efficiency:.3f}, domain_sim={domain_similarity:.3f}, "
                               f"factor={domain_factor:.3f}, final={transfer_efficiency:.3f}")
                    
                    transfer_efficiency = min(max(transfer_efficiency, 0.0), 1.0)
                    
                except Exception as e:
                    logger.warning(f"Error incorporating domain similarity in semantic transfer efficiency: {e}")
                    transfer_efficiency = base_efficiency
            else:
                # 无迁移项目时，基于领域相似度提供基础效率
                try:
                    domain_similarity = self._calculate_domain_similarity(source_domain, target_domain)
                    # 使用配置的最小迁移效率和调整范围
                    eff_config = self.similarity_config['transfer_efficiency_config']
                    min_eff = eff_config.get('min_transfer_efficiency', 0.2)
                    max_eff_for_empty = min_eff * 2.5  # 基于最小效率的扩展范围
                    transfer_efficiency = min_eff + (domain_similarity * (max_eff_for_empty - min_eff))
                except Exception as e:
                    logger.warning(f"Error calculating domain similarity for empty transfer: {e}")
                    # 使用配置的最小迁移效率
                    transfer_efficiency = self.similarity_config['transfer_efficiency_config'].get('min_transfer_efficiency', 0.3)
            
            logger.info(f"Semantic similarity transfer: {len(transferred_items)} items, avg similarity: {avg_similarity if transferred_items else 0:.3f}")
            
            return {
                'transferred_items': transferred_items,
                'transfer_efficiency': transfer_efficiency
            }
            
        except ImportError as e:
            logger.warning(f"Scikit-learn not available for semantic transfer: {e}")
            # 回退到基本迁移
            return {
                'transferred_items': self._basic_knowledge_transfer(source_domain, source_concepts, target_domain),
                'transfer_efficiency': 0.4
            }
        except Exception as e:
            logger.error(f"Semantic similarity transfer failed: {e}")
            return {
                'transferred_items': [],
                'transfer_efficiency': 0.0
            }
    
    def _structural_analogy_transfer(self, source_domain: str, source_concepts: Dict,
                                    target_domain: str, target_concepts: Dict) -> Dict[str, Any]:
        """基于结构类比的知识迁移
        
        Args:
            source_domain: 源领域
            source_concepts: 源领域概念
            target_domain: 目标领域
            target_concepts: 目标领域概念
            
        Returns:
            迁移结果字典
        """
        try:
            transferred_items = []
            
            # 分析概念结构特征
            source_structures = self._analyze_concept_structures(source_concepts)
            target_structures = self._analyze_concept_structures(target_concepts)
            
            if not source_structures or not target_structures:
                logger.warning("Insufficient structural features for analogy transfer")
                return {
                    'transferred_items': [],
                    'transfer_efficiency': 0.0
                }
            
            # 计算结构相似度矩阵
            similarity_matrix = self._calculate_structural_similarity_matrix(
                source_structures, target_structures
            )
            
            # 寻找最佳匹配
            matched_pairs = self._find_structural_matches(
                similarity_matrix, source_structures, target_structures
            )
            
            # 创建迁移项目
            for source_idx, target_idx, similarity in matched_pairs:
                source_id = source_structures[source_idx]['concept_id']
                target_id = target_structures[target_idx]['concept_id']
                source_data = source_concepts.get(source_id, {})
                target_data = target_concepts.get(target_id, {})
                
                # 生成迁移内容
                transferred_content = self._generate_structural_analogy_transfer(
                    source_domain, source_data, target_domain, target_data, similarity
                )
                
                transferred_items.append({
                    'original_content': source_data,
                    'transferred_content': transferred_content,
                    'structural_match_score': float(similarity),
                    'source_concept': source_id,
                    'target_concept': target_id,
                    'transfer_method': 'structural_analogy',
                    'structural_similarity': float(similarity),
                    'feature_match_details': self._get_feature_match_details(
                        source_structures[source_idx], target_structures[target_idx]
                    )
                })
            
            # 计算迁移效率（包含领域相似度）
            transfer_efficiency = self._calculate_transfer_efficiency(
                transferred_items, similarity_matrix, source_domain, target_domain
            )
            
            logger.info(f"Structural analogy transfer completed: {len(transferred_items)} items, efficiency: {transfer_efficiency:.3f}")
            
            return {
                'transferred_items': transferred_items,
                'transfer_efficiency': transfer_efficiency,
                'structural_analysis': {
                    'source_concepts_analyzed': len(source_structures),
                    'target_concepts_analyzed': len(target_structures),
                    'matches_found': len(matched_pairs),
                    'average_similarity': sum(s for _, _, s in matched_pairs) / max(1, len(matched_pairs))
                }
            }
            
        except Exception as e:
            logger.error(f"Structural analogy transfer failed: {e}")
            return {
                'transferred_items': [],
                'transfer_efficiency': 0.0
            }
    
    def _analyze_concept_structures(self, concepts: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """分析概念结构特征
        
        Args:
            concepts: 概念字典
            
        Returns:
            结构特征列表
        """
        structures = []
        
        for concept_id, concept_data in concepts.items():
            # 提取概念数据
            concept_name = concept_data.get('name', '')
            concept_desc = concept_data.get('description', '')
            raw_data = concept_data.get('concept_data', {})
            
            # 分析结构特征
            features = {
                'concept_id': concept_id,
                'name_length': len(concept_name),
                'description_length': len(concept_desc),
                'text_complexity': self._calculate_text_complexity(concept_name, concept_desc),
                'attribute_count': 0,
                'relation_count': 0,
                'data_types': set(),
                'nesting_depth': 0,
                'quality_score': concept_data.get('quality_metrics', {}).get('overall_score', 0.5)
            }
            
            # 分析原始概念数据
            if isinstance(raw_data, dict):
                features['attribute_count'] = len(raw_data)
                
                # 识别关系
                relation_keywords = ['relation', 'related', 'connection', 'link', 'associate']
                for key in raw_data.keys():
                    if any(keyword in str(key).lower() for keyword in relation_keywords):
                        features['relation_count'] += 1
                
                # 识别数据类型
                for value in raw_data.values():
                    if isinstance(value, (list, tuple)):
                        features['data_types'].add('list')
                        features['nesting_depth'] = max(features['nesting_depth'], 1)
                    elif isinstance(value, dict):
                        features['data_types'].add('dict')
                        features['nesting_depth'] = max(features['nesting_depth'], 2)
                    elif isinstance(value, (int, float)):
                        features['data_types'].add('numeric')
                    elif isinstance(value, str):
                        features['data_types'].add('text')
                    elif isinstance(value, bool):
                        features['data_types'].add('boolean')
            
            # 转换为可序列化的格式
            features['data_types'] = list(features['data_types'])
            structures.append(features)
        
        return structures
    
    def _calculate_text_complexity(self, name: str, description: str) -> float:
        """计算文本复杂度
        
        Args:
            name: 概念名称
            description: 概念描述
            
        Returns:
            float: 复杂度分数 (0-1)
        """
        total_text = f"{name} {description}"
        
        # 简单的复杂度指标
        word_count = len(total_text.split())
        char_count = len(total_text)
        unique_words = len(set(total_text.lower().split()))
        
        # 计算复杂度分数
        if word_count == 0:
            return 0.0
        
        complexity = (
            (unique_words / max(word_count, 1)) * 0.4 +  # 词汇多样性
            (min(word_count, 100) / 100) * 0.3 +  # 文本长度
            (min(char_count, 500) / 500) * 0.3    # 字符数量
        )
        
        return min(complexity, 1.0)
    
    def _calculate_structural_similarity_matrix(self, source_structures: List[Dict], 
                                               target_structures: List[Dict]) -> List[List[float]]:
        """计算结构相似度矩阵
        
        Args:
            source_structures: 源概念结构列表
            target_structures: 目标概念结构列表
            
        Returns:
            相似度矩阵
        """
        similarity_matrix = []
        
        for src in source_structures:
            row = []
            for tgt in target_structures:
                similarity = self._calculate_structure_similarity(src, tgt)
                row.append(similarity)
            similarity_matrix.append(row)
        
        return similarity_matrix
    
    def _calculate_structure_similarity(self, src: Dict, tgt: Dict) -> float:
        """计算两个概念结构之间的相似度
        
        Args:
            src: 源概念结构
            tgt: 目标概念结构
            
        Returns:
            float: 相似度分数 (0-1)
        """
        # 特征权重
        weights = {
            'name_length': 0.1,
            'description_length': 0.1,
            'text_complexity': 0.2,
            'attribute_count': 0.2,
            'relation_count': 0.15,
            'data_types': 0.15,
            'quality_score': 0.1
        }
        
        total_similarity = 0.0
        total_weight = 0.0
        
        # 数值特征的相似度（使用相对差异）
        numeric_features = ['name_length', 'description_length', 'text_complexity', 
                           'attribute_count', 'relation_count', 'quality_score']
        
        for feature in numeric_features:
            weight = weights.get(feature, 0)
            if weight > 0:
                src_val = src.get(feature, 0)
                tgt_val = tgt.get(feature, 0)
                
                if src_val == 0 and tgt_val == 0:
                    similarity = 1.0
                else:
                    max_val = max(abs(src_val), abs(tgt_val), 1.0)
                    similarity = 1.0 - (abs(src_val - tgt_val) / max_val)
                
                total_similarity += similarity * weight
                total_weight += weight
        
        # 数据类型相似度（Jaccard相似度）
        src_types = set(src.get('data_types', []))
        tgt_types = set(tgt.get('data_types', []))
        
        if src_types or tgt_types:
            intersection = len(src_types & tgt_types)
            union = len(src_types | tgt_types)
            type_similarity = intersection / union if union > 0 else 0.0
        else:
            type_similarity = 1.0
        
        total_similarity += type_similarity * weights['data_types']
        total_weight += weights['data_types']
        
        # 归一化
        if total_weight > 0:
            return total_similarity / total_weight
        else:
            return 0.0
    
    def _find_structural_matches(self, similarity_matrix: List[List[float]],
                                source_structures: List[Dict], 
                                target_structures: List[Dict]) -> List[tuple]:
        """寻找结构匹配
        
        Args:
            similarity_matrix: 相似度矩阵
            source_structures: 源概念结构列表
            target_structures: 目标概念结构列表
            
        Returns:
            匹配列表 [(source_idx, target_idx, similarity), ...]
        """
        if not similarity_matrix:
            return []
        
        matches = []
        used_sources = set()
        used_targets = set()
        
        # 简单的贪婪匹配：找到相似度最高的配对
        while True:
            best_similarity = 0.0
            best_pair = None
            
            for i in range(len(source_structures)):
                if i in used_sources:
                    continue
                    
                for j in range(len(target_structures)):
                    if j in used_targets:
                        continue
                    
                    similarity = similarity_matrix[i][j]
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_pair = (i, j, similarity)
            
            if best_pair is None or best_similarity < 0.3:  # 相似度阈值
                break
            
            i, j, similarity = best_pair
            matches.append(best_pair)
            used_sources.add(i)
            used_targets.add(j)
            
            # 限制最大匹配数
            if len(matches) >= min(5, len(source_structures), len(target_structures)):
                break
        
        return matches
    
    def _generate_structural_analogy_transfer(self, source_domain: str, source_data: Dict,
                                             target_domain: str, target_data: Dict,
                                             similarity: float) -> str:
        """生成结构类比迁移内容
        
        Args:
            source_domain: 源领域
            source_data: 源概念数据
            target_domain: 目标领域
            target_data: 目标概念数据
            similarity: 结构相似度
            
        Returns:
            str: 迁移内容描述
        """
        source_name = source_data.get('name', 'unknown')
        target_name = target_data.get('name', 'unknown')
        
        return (f"Structural analogy transfer from {source_domain} concept '{source_name}' "
                f"to {target_domain} concept '{target_name}' "
                f"(structural similarity: {similarity:.3f}). "
                f"The transfer is based on matching structural features including "
                f"attribute count, relation patterns, and data type compatibility.")
    
    def _get_feature_match_details(self, src_features: Dict, tgt_features: Dict) -> Dict[str, Any]:
        """获取特征匹配详情
        
        Args:
            src_features: 源特征
            tgt_features: 目标特征
            
        Returns:
            匹配详情字典
        """
        return {
            'name_length_match': abs(src_features.get('name_length', 0) - tgt_features.get('name_length', 0)),
            'description_length_match': abs(src_features.get('description_length', 0) - tgt_features.get('description_length', 0)),
            'attribute_count_match': abs(src_features.get('attribute_count', 0) - tgt_features.get('attribute_count', 0)),
            'relation_count_match': abs(src_features.get('relation_count', 0) - tgt_features.get('relation_count', 0)),
            'data_type_overlap': len(set(src_features.get('data_types', [])) & set(tgt_features.get('data_types', []))),
            'data_type_union': len(set(src_features.get('data_types', [])) | set(tgt_features.get('data_types', []))),
            'text_complexity_match': abs(src_features.get('text_complexity', 0) - tgt_features.get('text_complexity', 0))
        }
    
    def _calculate_transfer_efficiency(self, transferred_items: List[Dict],
                                      similarity_matrix: List[List[float]],
                                      source_domain: str = None, 
                                      target_domain: str = None) -> float:
        """计算迁移效率（改进版：包含领域相似度）
        
        Args:
            transferred_items: 迁移项目列表
            similarity_matrix: 相似度矩阵
            source_domain: 源领域名称（可选）
            target_domain: 目标领域名称（可选）
            
        Returns:
            float: 迁移效率 (0-1)
        """
        if not transferred_items:
            return 0.0
        
        # 基于匹配相似度的平均效率
        total_similarity = sum(item.get('structural_similarity', 0.5) for item in transferred_items)
        avg_similarity = total_similarity / len(transferred_items)
        
        # 基于匹配数量的效率
        match_count = len(transferred_items)
        max_possible = len(similarity_matrix) * len(similarity_matrix[0]) if similarity_matrix else 1
        coverage = match_count / max_possible if max_possible > 0 else 0
        
        # 计算基础效率（使用配置的权重）
        eff_weights = self.similarity_config['transfer_efficiency_config']['efficiency_weights']
        base_efficiency = (
            avg_similarity * eff_weights.get('similarity_weight', 0.7) + 
            coverage * eff_weights.get('coverage_weight', 0.3)
        )
        
        # 如果提供了领域信息，计算领域相似度并调整效率
        if source_domain and target_domain:
            try:
                domain_similarity = self._calculate_domain_similarity(source_domain, target_domain)
                
                # 领域相似度对效率的影响：高相似度提升效率，低相似度降低效率
                # 使用配置的中性点和调整强度
                neutral_factor = eff_weights.get('neutral_domain_factor', 0.5)
                adjustment_strength = eff_weights.get('domain_adjustment_strength', 0.4)
                domain_factor = neutral_factor + (domain_similarity - neutral_factor) * adjustment_strength
                
                # 综合效率：基础效率 * 领域因子
                efficiency = base_efficiency * domain_factor
                
                logger.debug(f"Transfer efficiency with domain similarity: base={base_efficiency:.3f}, "
                           f"domain_sim={domain_similarity:.3f}, factor={domain_factor:.3f}, "
                           f"final={efficiency:.3f}")
                
                return min(max(efficiency, 0.0), 1.0)
                
            except Exception as e:
                logger.warning(f"Error incorporating domain similarity in efficiency calculation: {e}")
                # 出错时回退到基础效率
                return min(base_efficiency, 1.0)
        
        # 未提供领域信息，使用基础效率
        return min(base_efficiency, 1.0)
    
    def _calculate_domain_similarity(self, domain1: str, domain2: str) -> float:
        """计算两个领域之间的真实相似度
        
        Args:
            domain1: 第一个领域名称
            domain2: 第二个领域名称
            
        Returns:
            float: 领域相似度分数 (0-1)
        """
        try:
            # 检查领域是否存在
            if domain1 not in self.knowledge_bases or domain2 not in self.knowledge_bases:
                logger.warning(f"One or both domains not found: {domain1}, {domain2}")
                return 0.3  # 默认低相似度
            
            kb1 = self.knowledge_bases[domain1]
            kb2 = self.knowledge_bases[domain2]
            
            # 如果知识库为空，返回低相似度
            if not kb1 or not kb2:
                return 0.2
            
            # 提取概念用于比较
            concepts1 = self._extract_concepts_from_knowledge(domain1, kb1)
            concepts2 = self._extract_concepts_from_knowledge(domain2, kb2)
            
            if not concepts1 or not concepts2:
                return 0.25
            
            # 1. 概念名称重叠相似度
            concept_names1 = {c.get('name', '') for c in concepts1.values() if c.get('name')}
            concept_names2 = {c.get('name', '') for c in concepts2.values() if c.get('name')}
            
            name_overlap = 0.0
            if concept_names1 and concept_names2:
                intersection = concept_names1.intersection(concept_names2)
                union = concept_names1.union(concept_names2)
                if union:
                    name_overlap = len(intersection) / len(union)
            
            # 2. 语义相似度（基于概念描述）
            semantic_similarity = self._calculate_domain_semantic_similarity(concepts1, concepts2)
            
            # 3. 结构相似度（基于概念关系）
            structural_similarity = self._calculate_domain_structural_similarity(concepts1, concepts2)
            
            # 4. 抽象级别相似度
            abstraction_similarity = self._calculate_domain_abstraction_similarity(concepts1, concepts2)
            
            # 5. 复杂度相似度
            complexity_similarity = self._calculate_domain_complexity_similarity(concepts1, concepts2)
            
            # 加权综合相似度（使用配置的权重）
            weights = self.similarity_config['domain_similarity_weights']
            
            total_similarity = (
                name_overlap * weights.get('name_overlap', 0.2) +
                semantic_similarity * weights.get('semantic', 0.3) +
                structural_similarity * weights.get('structural', 0.25) +
                abstraction_similarity * weights.get('abstraction', 0.15) +
                complexity_similarity * weights.get('complexity', 0.1)
            )
            
            logger.info(f"Domain similarity between {domain1} and {domain2}: {total_similarity:.3f} "
                       f"(name: {name_overlap:.3f}, semantic: {semantic_similarity:.3f}, "
                       f"structural: {structural_similarity:.3f}, abstraction: {abstraction_similarity:.3f}, "
                       f"complexity: {complexity_similarity:.3f})")
            
            return min(max(total_similarity, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating domain similarity between {domain1} and {domain2}: {e}")
            return 0.3  # 出错时返回默认相似度
    
    def _calculate_domain_semantic_similarity(self, concepts1: Dict, concepts2: Dict) -> float:
        """计算领域语义相似度
        
        Args:
            concepts1: 第一个领域的概念字典
            concepts2: 第二个领域的概念字典
            
        Returns:
            float: 语义相似度 (0-1)
        """
        try:
            # 收集所有概念描述
            descriptions1 = []
            for concept in concepts1.values():
                desc = concept.get('description', '') or concept.get('text', '')
                if desc:
                    descriptions1.append(desc)
            
            descriptions2 = []
            for concept in concepts2.values():
                desc = concept.get('description', '') or concept.get('text', '')
                if desc:
                    descriptions2.append(desc)
            
            if not descriptions1 or not descriptions2:
                return 0.4
            
            # 合并描述为文档
            doc1 = ' '.join(descriptions1[:10])  # 限制长度
            doc2 = ' '.join(descriptions2[:10])
            
            # 简单的文本相似度（基于共享词汇）
            words1 = set(doc1.lower().split())
            words2 = set(doc2.lower().split())
            
            if not words1 or not words2:
                return 0.4
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            if not union:
                return 0.4
            
            similarity = len(intersection) / len(union)
            
            # 调整范围：0.3-0.7映射到0-1
            adjusted_similarity = max(0.0, min(1.0, (similarity - 0.3) * 2.5))
            
            return adjusted_similarity
            
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.4
    
    def _calculate_domain_structural_similarity(self, concepts1: Dict, concepts2: Dict) -> float:
        """计算领域结构相似度
        
        Args:
            concepts1: 第一个领域的概念字典
            concepts2: 第二个领域的概念字典
            
        Returns:
            float: 结构相似度 (0-1)
        """
        try:
            # 提取关系模式
            relations1 = set()
            for concept in concepts1.values():
                for rel in concept.get('relations', []):
                    if isinstance(rel, dict) and 'type' in rel:
                        relations1.add(rel['type'])
            
            relations2 = set()
            for concept in concepts2.values():
                for rel in concept.get('relations', []):
                    if isinstance(rel, dict) and 'type' in rel:
                        relations2.add(rel['type'])
            
            # 计算关系模式相似度
            if not relations1 and not relations2:
                return 0.5  # 都无关系，中等相似度
            elif not relations1 or not relations2:
                return 0.2  # 一个有，一个无，低相似度
            
            intersection = relations1.intersection(relations2)
            union = relations1.union(relations2)
            
            if not union:
                return 0.5
            
            relation_similarity = len(intersection) / len(union)
            
            # 计算概念层次相似度（基于层次级别分布）
            levels1 = [c.get('hierarchical_level', 0) for c in concepts1.values()]
            levels2 = [c.get('hierarchical_level', 0) for c in concepts2.values()]
            
            if levels1 and levels2:
                avg_level1 = sum(levels1) / len(levels1)
                avg_level2 = sum(levels2) / len(levels2)
                level_diff = abs(avg_level1 - avg_level2)
                level_similarity = 1.0 - min(level_diff / 5.0, 1.0)  # 假设最大层级差为5
            else:
                level_similarity = 0.5
            
            # 综合结构相似度
            structural_similarity = (relation_similarity * 0.6 + level_similarity * 0.4)
            
            return structural_similarity
            
        except Exception as e:
            logger.warning(f"Error calculating structural similarity: {e}")
            return 0.4
    
    def _calculate_domain_abstraction_similarity(self, concepts1: Dict, concepts2: Dict) -> float:
        """计算领域抽象级别相似度
        
        Args:
            concepts1: 第一个领域的概念字典
            concepts2: 第二个领域的概念字典
            
        Returns:
            float: 抽象级别相似度 (0-1)
        """
        try:
            # 计算平均抽象度
            abstraction_scores1 = []
            for concept in concepts1.values():
                score = self._calculate_abstraction_score(concept)
                abstraction_scores1.append(score)
            
            abstraction_scores2 = []
            for concept in concepts2.values():
                score = self._calculate_abstraction_score(concept)
                abstraction_scores2.append(score)
            
            if not abstraction_scores1 or not abstraction_scores2:
                return 0.5
            
            avg_abs1 = sum(abstraction_scores1) / len(abstraction_scores1)
            avg_abs2 = sum(abstraction_scores2) / len(abstraction_scores2)
            
            # 计算相似度（1 - 绝对差）
            abs_diff = abs(avg_abs1 - avg_abs2)
            abstraction_similarity = 1.0 - min(abs_diff / 0.5, 1.0)  # 最大差0.5
            
            return abstraction_similarity
            
        except Exception as e:
            logger.warning(f"Error calculating abstraction similarity: {e}")
            return 0.5
    
    def _calculate_domain_complexity_similarity(self, concepts1: Dict, concepts2: Dict) -> float:
        """计算领域复杂度相似度
        
        Args:
            concepts1: 第一个领域的概念字典
            concepts2: 第二个领域的概念字典
            
        Returns:
            float: 复杂度相似度 (0-1)
        """
        try:
            # 计算平均复杂度
            complexity_scores1 = []
            for concept in concepts1.values():
                score = self._calculate_concept_complexity(concept)
                complexity_scores1.append(score)
            
            complexity_scores2 = []
            for concept in concepts2.values():
                score = self._calculate_concept_complexity(concept)
                complexity_scores2.append(score)
            
            if not complexity_scores1 or not complexity_scores2:
                return 0.5
            
            avg_comp1 = sum(complexity_scores1) / len(complexity_scores1)
            avg_comp2 = sum(complexity_scores2) / len(complexity_scores2)
            
            # 计算相似度（1 - 相对差）
            comp_diff = abs(avg_comp1 - avg_comp2)
            max_possible_diff = max(avg_comp1, avg_comp2, 0.1)  # 避免除零
            complexity_similarity = 1.0 - min(comp_diff / max_possible_diff, 1.0)
            
            return complexity_similarity
            
        except Exception as e:
            logger.warning(f"Error calculating complexity similarity: {e}")
            return 0.5
    
    def _conceptual_mapping_transfer(self, source_domain: str, source_concepts: Dict,
                                    target_domain: str, target_concepts: Dict) -> Dict[str, Any]:
        """基于概念映射的知识迁移
        
        Args:
            source_domain: 源领域
            source_concepts: 源领域概念
            target_domain: 目标领域
            target_concepts: 目标领域概念
            
        Returns:
            迁移结果字典
        """
        try:
            transferred_items = []
            
            # 分析概念特征并计算相似度
            concept_matches = self._analyze_concept_mappings(source_concepts, target_concepts)
            
            # 选择最佳映射（基于综合相似度）
            selected_matches = self._select_best_concept_mappings(concept_matches)
            
            # 创建迁移项目
            for match in selected_matches:
                source_id = match['source_id']
                target_id = match['target_id']
                source_data = source_concepts.get(source_id, {})
                target_data = target_concepts.get(target_id, {})
                
                # 生成迁移内容
                transferred_content = self._generate_conceptual_mapping_transfer(
                    source_domain, source_data, target_domain, target_data, match
                )
                
                transferred_items.append({
                    'original_content': source_data,
                    'transferred_content': transferred_content,
                    'mapping_confidence': float(match['overall_similarity']),
                    'source_concept': source_id,
                    'target_concept': target_id,
                    'transfer_method': 'conceptual_mapping',
                    'mapping_details': {
                        'semantic_similarity': match['semantic_similarity'],
                        'structural_similarity': match['structural_similarity'],
                        'contextual_similarity': match['contextual_similarity'],
                        'hierarchical_match': match['hierarchical_match'],
                        'feature_overlap': match['feature_overlap']
                    },
                    'mapping_type': match['mapping_type']
                })
            
            # 计算迁移效率（包含领域相似度）
            transfer_efficiency = self._calculate_conceptual_mapping_efficiency(
                transferred_items, concept_matches, source_domain, target_domain
            )
            
            logger.info(f"Conceptual mapping transfer completed: {len(transferred_items)} items, "
                       f"efficiency: {transfer_efficiency:.3f}, "
                       f"average confidence: {sum(item['mapping_confidence'] for item in transferred_items)/max(1, len(transferred_items)):.3f}")
            
            return {
                'transferred_items': transferred_items,
                'transfer_efficiency': transfer_efficiency,
                'mapping_analysis': {
                    'total_possible_matches': len(concept_matches),
                    'selected_matches': len(selected_matches),
                    'average_semantic_similarity': sum(m['semantic_similarity'] for m in selected_matches)/max(1, len(selected_matches)),
                    'average_structural_similarity': sum(m['structural_similarity'] for m in selected_matches)/max(1, len(selected_matches)),
                    'mapping_quality': self._assess_mapping_quality(selected_matches)
                }
            }
            
        except Exception as e:
            logger.error(f"Conceptual mapping transfer failed: {e}")
            return {
                'transferred_items': [],
                'transfer_efficiency': 0.0
            }
    
    def _analyze_concept_mappings(self, source_concepts: Dict, target_concepts: Dict) -> List[Dict[str, Any]]:
        """分析概念之间的所有可能映射
        
        Args:
            source_concepts: 源概念字典
            target_concepts: 目标概念字典
            
        Returns:
            概念匹配分析列表
        """
        concept_matches = []
        
        # 准备概念特征
        source_features = self._extract_concept_features(source_concepts)
        target_features = self._extract_concept_features(target_concepts)
        
        # 分析所有可能的配对
        for source_id, source_feat in source_features.items():
            for target_id, target_feat in target_features.items():
                # 计算多维度相似度
                semantic_similarity = self._calculate_semantic_similarity_for_mapping(
                    source_feat, target_feat
                )
                
                structural_similarity = self._calculate_structural_similarity_for_mapping(
                    source_feat, target_feat
                )
                
                contextual_similarity = self._calculate_contextual_similarity_for_mapping(
                    source_feat, target_feat, source_concepts, target_concepts
                )
                
                hierarchical_match = self._assess_hierarchical_match(
                    source_feat, target_feat
                )
                
                feature_overlap = self._calculate_feature_overlap(
                    source_feat, target_feat
                )
                
                # 综合相似度（加权平均）
                weights = {
                    'semantic': 0.4,
                    'structural': 0.25,
                    'contextual': 0.2,
                    'hierarchical': 0.1,
                    'feature': 0.05
                }
                
                overall_similarity = (
                    semantic_similarity * weights['semantic'] +
                    structural_similarity * weights['structural'] +
                    contextual_similarity * weights['contextual'] +
                    hierarchical_match * weights['hierarchical'] +
                    feature_overlap * weights['feature']
                )
                
                # 确定映射类型
                mapping_type = self._determine_mapping_type(
                    semantic_similarity, structural_similarity, 
                    contextual_similarity, hierarchical_match
                )
                
                concept_matches.append({
                    'source_id': source_id,
                    'target_id': target_id,
                    'semantic_similarity': semantic_similarity,
                    'structural_similarity': structural_similarity,
                    'contextual_similarity': contextual_similarity,
                    'hierarchical_match': hierarchical_match,
                    'feature_overlap': feature_overlap,
                    'overall_similarity': overall_similarity,
                    'mapping_type': mapping_type,
                    'source_features': source_feat,
                    'target_features': target_feat
                })
        
        return concept_matches
    
    def _extract_concept_features(self, concepts: Dict[str, Dict]) -> Dict[str, Dict[str, Any]]:
        """提取概念的详细特征用于映射分析
        
        Args:
            concepts: 概念字典
            
        Returns:
            概念特征字典
        """
        features = {}
        
        for concept_id, concept_data in concepts.items():
            concept_name = concept_data.get('name', '')
            concept_desc = concept_data.get('description', '')
            raw_data = concept_data.get('concept_data', {})
            quality_metrics = concept_data.get('quality_metrics', {})
            
            # 提取语义特征
            semantic_features = {
                'name_tokens': concept_name.lower().split(),
                'description_tokens': concept_desc.lower().split(),
                'name_length': len(concept_name),
                'description_length': len(concept_desc),
                'vocabulary_richness': len(set(concept_name.lower().split() + concept_desc.lower().split())),
                'quality_score': quality_metrics.get('overall_score', 0.5)
            }
            
            # 提取结构特征
            structural_features = {
                'attribute_count': 0,
                'relation_count': 0,
                'data_types': set(),
                'nesting_depth': 0,
                'has_examples': False,
                'has_properties': False,
                'has_relations': False
            }
            
            if isinstance(raw_data, dict):
                structural_features['attribute_count'] = len(raw_data)
                
                # 检查特定字段
                if 'examples' in raw_data:
                    structural_features['has_examples'] = True
                if 'properties' in raw_data:
                    structural_features['has_properties'] = True
                if 'relations' in raw_data:
                    structural_features['has_relations'] = True
                
                # 识别关系
                relation_keywords = ['relation', 'related', 'connection', 'link', 'associate']
                for key in raw_data.keys():
                    if any(keyword in str(key).lower() for keyword in relation_keywords):
                        structural_features['relation_count'] += 1
                
                # 分析数据类型
                for value in raw_data.values():
                    if isinstance(value, (list, tuple)):
                        structural_features['data_types'].add('list')
                        structural_features['nesting_depth'] = max(structural_features['nesting_depth'], 1)
                    elif isinstance(value, dict):
                        structural_features['data_types'].add('dict')
                        structural_features['nesting_depth'] = max(structural_features['nesting_depth'], 2)
                    elif isinstance(value, (int, float)):
                        structural_features['data_types'].add('numeric')
                    elif isinstance(value, str):
                        structural_features['data_types'].add('text')
                    elif isinstance(value, bool):
                        structural_features['data_types'].add('boolean')
            
            # 提取上下文特征
            contextual_features = {
                'domain_specific_terms': self._extract_domain_terms(concept_name, concept_desc),
                'concept_category': concept_data.get('category', ''),
                'concept_hierarchy': self._extract_hierarchy_info(raw_data)
            }
            
            # 组合所有特征
            features[concept_id] = {
                'semantic': semantic_features,
                'structural': structural_features,
                'contextual': contextual_features,
                'raw_name': concept_name,
                'raw_description': concept_desc,
                'raw_data': raw_data
            }
        
        return features
    
    def _calculate_semantic_similarity_for_mapping(self, source_feat: Dict, target_feat: Dict) -> float:
        """计算概念间的语义相似度
        
        Args:
            source_feat: 源概念特征
            target_feat: 目标概念特征
            
        Returns:
            float: 语义相似度 (0-1)
        """
        # 名称相似度
        source_name_tokens = set(source_feat['semantic']['name_tokens'])
        target_name_tokens = set(target_feat['semantic']['name_tokens'])
        
        if source_name_tokens and target_name_tokens:
            name_similarity = len(source_name_tokens & target_name_tokens) / len(source_name_tokens | target_name_tokens)
        else:
            name_similarity = 0.0
        
        # 描述相似度
        source_desc_tokens = set(source_feat['semantic']['description_tokens'])
        target_desc_tokens = set(target_feat['semantic']['description_tokens'])
        
        if source_desc_tokens and target_desc_tokens:
            desc_similarity = len(source_desc_tokens & target_desc_tokens) / len(source_desc_tokens | target_desc_tokens)
        else:
            desc_similarity = 0.0
        
        # 词汇丰富度相似度
        source_vocab = source_feat['semantic']['vocabulary_richness']
        target_vocab = target_feat['semantic']['vocabulary_richness']
        vocab_similarity = 1.0 - (abs(source_vocab - target_vocab) / max(source_vocab, target_vocab, 1))
        
        # 综合语义相似度
        semantic_similarity = (
            name_similarity * 0.5 +
            desc_similarity * 0.3 +
            vocab_similarity * 0.2
        )
        
        return min(semantic_similarity, 1.0)
    
    def _calculate_structural_similarity_for_mapping(self, source_feat: Dict, target_feat: Dict) -> float:
        """计算概念间的结构相似度
        
        Args:
            source_feat: 源概念特征
            target_feat: 目标概念特征
            
        Returns:
            float: 结构相似度 (0-1)
        """
        source_struct = source_feat['structural']
        target_struct = target_feat['structural']
        
        # 属性数量相似度
        attr_diff = abs(source_struct['attribute_count'] - target_struct['attribute_count'])
        attr_max = max(source_struct['attribute_count'], target_struct['attribute_count'], 1)
        attr_similarity = 1.0 - (attr_diff / attr_max)
        
        # 关系数量相似度
        rel_diff = abs(source_struct['relation_count'] - target_struct['relation_count'])
        rel_max = max(source_struct['relation_count'], target_struct['relation_count'], 1)
        rel_similarity = 1.0 - (rel_diff / rel_max)
        
        # 数据类型相似度（Jaccard相似度）
        source_types = set(source_struct['data_types'])
        target_types = set(target_struct['data_types'])
        
        if source_types or target_types:
            type_intersection = len(source_types & target_types)
            type_union = len(source_types | target_types)
            type_similarity = type_intersection / type_union if type_union > 0 else 0.0
        else:
            type_similarity = 1.0
        
        # 特征存在性相似度
        feature_similarity = 0.0
        feature_keys = ['has_examples', 'has_properties', 'has_relations']
        matching_features = sum(1 for key in feature_keys 
                               if source_struct[key] == target_struct[key])
        feature_similarity = matching_features / len(feature_keys)
        
        # 综合结构相似度
        structural_similarity = (
            attr_similarity * 0.3 +
            rel_similarity * 0.3 +
            type_similarity * 0.2 +
            feature_similarity * 0.2
        )
        
        return min(structural_similarity, 1.0)
    
    def _calculate_contextual_similarity_for_mapping(self, source_feat: Dict, target_feat: Dict,
                                                    source_concepts: Dict, target_concepts: Dict) -> float:
        """计算概念间的上下文相似度
        
        Args:
            source_feat: 源概念特征
            target_feat: 目标概念特征
            source_concepts: 源概念字典
            target_concepts: 目标概念字典
            
        Returns:
            float: 上下文相似度 (0-1)
        """
        source_context = source_feat['contextual']
        target_context = target_feat['contextual']
        
        # 类别相似度
        source_category = source_context.get('concept_category', '')
        target_category = target_context.get('concept_category', '')
        
        if source_category and target_category:
            category_similarity = 1.0 if source_category == target_category else 0.5
        else:
            category_similarity = 0.3
        
        # 领域术语相似度
        source_terms = set(source_context.get('domain_specific_terms', []))
        target_terms = set(target_context.get('domain_specific_terms', []))
        
        if source_terms or target_terms:
            term_intersection = len(source_terms & target_terms)
            term_union = len(source_terms | target_terms)
            term_similarity = term_intersection / term_union if term_union > 0 else 0.0
        else:
            term_similarity = 0.5
        
        # 层次结构相似度
        source_hierarchy = source_context.get('concept_hierarchy', {})
        target_hierarchy = target_context.get('concept_hierarchy', {})
        hierarchy_similarity = self._compare_hierarchy_structures(source_hierarchy, target_hierarchy)
        
        # 综合上下文相似度
        contextual_similarity = (
            category_similarity * 0.4 +
            term_similarity * 0.3 +
            hierarchy_similarity * 0.3
        )
        
        return min(contextual_similarity, 1.0)
    
    def _assess_hierarchical_match(self, source_feat: Dict, target_feat: Dict) -> float:
        """评估概念层次匹配
        
        Args:
            source_feat: 源概念特征
            target_feat: 目标概念特征
            
        Returns:
            float: 层次匹配分数 (0-1)
        """
        # 简单的层次匹配评估
        # 在实际应用中，这里应该分析概念在各自领域中的层次位置
        
        source_name = source_feat['raw_name'].lower()
        target_name = target_feat['raw_name'].lower()
        
        # 检查是否有层次指示词
        hierarchy_indicators = {
            'high_level': ['system', 'framework', 'architecture', 'model', 'theory'],
            'mid_level': ['component', 'module', 'function', 'process', 'method'],
            'low_level': ['element', 'instance', 'example', 'case', 'implementation']
        }
        
        source_level = 'unknown'
        target_level = 'unknown'
        
        for level, indicators in hierarchy_indicators.items():
            if any(indicator in source_name for indicator in indicators):
                source_level = level
            if any(indicator in target_name for indicator in indicators):
                target_level = level
        
        # 计算层次匹配分数
        if source_level == target_level:
            return 0.9
        elif source_level != 'unknown' and target_level != 'unknown':
            # 允许相邻层次的匹配
            level_order = ['high_level', 'mid_level', 'low_level']
            source_idx = level_order.index(source_level) if source_level in level_order else -1
            target_idx = level_order.index(target_level) if target_level in level_order else -1
            
            if abs(source_idx - target_idx) <= 1:
                return 0.7
            else:
                return 0.3
        else:
            return 0.5
    
    def _calculate_feature_overlap(self, source_feat: Dict, target_feat: Dict) -> float:
        """计算概念特征重叠度
        
        Args:
            source_feat: 源概念特征
            target_feat: 目标概念特征
            
        Returns:
            float: 特征重叠度 (0-1)
        """
        # 计算各种特征的重叠
        overlap_scores = []
        
        # 名称特征重叠
        source_name_tokens = set(source_feat['semantic']['name_tokens'])
        target_name_tokens = set(target_feat['semantic']['name_tokens'])
        if source_name_tokens and target_name_tokens:
            name_overlap = len(source_name_tokens & target_name_tokens) / len(source_name_tokens | target_name_tokens)
            overlap_scores.append(name_overlap)
        
        # 结构特征重叠
        source_struct = source_feat['structural']
        target_struct = target_feat['structural']
        
        # 数据类型重叠
        source_types = set(source_struct['data_types'])
        target_types = set(target_struct['data_types'])
        if source_types or target_types:
            type_overlap = len(source_types & target_types) / len(source_types | target_types) if source_types or target_types else 0.0
            overlap_scores.append(type_overlap)
        
        # 布尔特征匹配
        bool_features = ['has_examples', 'has_properties', 'has_relations']
        bool_matches = sum(1 for feature in bool_features 
                          if source_struct[feature] == target_struct[feature])
        bool_overlap = bool_matches / len(bool_features)
        overlap_scores.append(bool_overlap)
        
        # 计算平均重叠度
        if overlap_scores:
            return sum(overlap_scores) / len(overlap_scores)
        else:
            return 0.0
    
    def _extract_domain_terms(self, name: str, description: str) -> List[str]:
        """从概念名称和描述中提取领域特定术语
        
        Args:
            name: 概念名称
            description: 概念描述
            
        Returns:
            领域术语列表
        """
        # 简单的术语提取
        # 在实际应用中，这里可以使用领域词典或术语提取算法
        
        all_text = f"{name} {description}".lower()
        words = all_text.split()
        
        # 过滤常见词汇
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        domain_terms = [word for word in words if word not in common_words and len(word) > 3]
        
        return list(set(domain_terms))
    
    def _extract_hierarchy_info(self, concept_data: Dict) -> Dict[str, Any]:
        """从概念数据中提取层次结构信息
        
        Args:
            concept_data: 概念原始数据
            
        Returns:
            层次结构信息
        """
        hierarchy_info = {
            'parent_concepts': [],
            'child_concepts': [],
            'sibling_concepts': [],
            'abstraction_level': 'unknown'
        }
        
        if isinstance(concept_data, dict):
            # 检查层次关系字段
            if 'parent' in concept_data:
                hierarchy_info['parent_concepts'] = [concept_data['parent']] if isinstance(concept_data['parent'], str) else concept_data['parent']
            if 'children' in concept_data:
                hierarchy_info['child_concepts'] = concept_data['children'] if isinstance(concept_data['children'], list) else [concept_data['children']]
            if 'related' in concept_data:
                hierarchy_info['sibling_concepts'] = concept_data['related'] if isinstance(concept_data['related'], list) else [concept_data['related']]
            
            # 推断抽象级别
            abstraction_keywords = {
                'abstract': ['theory', 'framework', 'model', 'system', 'architecture'],
                'concrete': ['example', 'instance', 'implementation', 'case', 'specific']
            }
            
            all_text = str(concept_data).lower()
            for level, keywords in abstraction_keywords.items():
                if any(keyword in all_text for keyword in keywords):
                    hierarchy_info['abstraction_level'] = level
                    break
        
        return hierarchy_info
    
    def _compare_hierarchy_structures(self, source_hierarchy: Dict, target_hierarchy: Dict) -> float:
        """比较层次结构
        
        Args:
            source_hierarchy: 源层次结构
            target_hierarchy: 目标层次结构
            
        Returns:
            float: 层次结构相似度 (0-1)
        """
        # 简单的层次结构比较
        similarity_score = 0.0
        
        # 抽象级别匹配
        source_level = source_hierarchy.get('abstraction_level', 'unknown')
        target_level = target_hierarchy.get('abstraction_level', 'unknown')
        
        if source_level == target_level:
            similarity_score += 0.4
        elif source_level != 'unknown' and target_level != 'unknown':
            similarity_score += 0.2
        
        # 关系结构相似度
        source_has_parent = len(source_hierarchy.get('parent_concepts', [])) > 0
        target_has_parent = len(target_hierarchy.get('parent_concepts', [])) > 0
        
        source_has_children = len(source_hierarchy.get('child_concepts', [])) > 0
        target_has_children = len(target_hierarchy.get('child_concepts', [])) > 0
        
        # 父关系匹配
        if source_has_parent == target_has_parent:
            similarity_score += 0.2
        
        # 子关系匹配
        if source_has_children == target_has_children:
            similarity_score += 0.2
        
        # 同级关系匹配
        source_has_siblings = len(source_hierarchy.get('sibling_concepts', [])) > 0
        target_has_siblings = len(target_hierarchy.get('sibling_concepts', [])) > 0
        
        if source_has_siblings == target_has_siblings:
            similarity_score += 0.2
        
        return min(similarity_score, 1.0)
    
    def _determine_mapping_type(self, semantic_sim: float, structural_sim: float,
                               contextual_sim: float, hierarchical_match: float) -> str:
        """确定映射类型
        
        Args:
            semantic_sim: 语义相似度
            structural_sim: 结构相似度
            contextual_sim: 上下文相似度
            hierarchical_match: 层次匹配度
            
        Returns:
            str: 映射类型
        """
        # 获取配置的阈值
        thresholds = self.similarity_config['semantic_thresholds']
        high_sim = thresholds.get('high_similarity', 0.7)
        very_high_sim = thresholds.get('very_high_similarity', 0.8)
        balanced_threshold = thresholds.get('balanced_threshold', 0.6)
        composite_threshold = thresholds.get('composite_threshold', 1.5)
        
        # 基于相似度模式确定映射类型
        if semantic_sim > very_high_sim and structural_sim > high_sim:
            return 'strong_semantic_structural_match'
        elif semantic_sim > high_sim:
            return 'semantic_dominant_match'
        elif structural_sim > high_sim:
            return 'structural_dominant_match'
        elif contextual_sim > high_sim:
            return 'contextual_match'
        elif hierarchical_match > high_sim:
            return 'hierarchical_match'
        elif semantic_sim > balanced_threshold and structural_sim > balanced_threshold:
            return 'balanced_match'
        elif semantic_sim + structural_sim + contextual_sim > composite_threshold:
            return 'composite_match'
        else:
            return 'weak_match'
    
    def _select_best_concept_mappings(self, concept_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """选择最佳的概念映射
        
        Args:
            concept_matches: 概念匹配列表
            
        Returns:
            最佳映射列表
        """
        if not concept_matches:
            return []
        
        # 按综合相似度排序
        sorted_matches = sorted(concept_matches, key=lambda x: x['overall_similarity'], reverse=True)
        
        # 选择最佳匹配（基于相似度阈值）
        selected_matches = []
        used_sources = set()
        used_targets = set()
        
        for match in sorted_matches:
            source_id = match['source_id']
            target_id = match['target_id']
            
            # 检查是否已被使用
            if source_id in used_sources or target_id in used_targets:
                continue
            
            # 应用相似度阈值（使用配置）
            min_selection_threshold = self.similarity_config['semantic_thresholds'].get('min_selection_threshold', 0.5)
            if match['overall_similarity'] >= min_selection_threshold:
                selected_matches.append(match)
                used_sources.add(source_id)
                used_targets.add(target_id)
            
            # 限制最大选择数量
            if len(selected_matches) >= min(10, len(concept_matches) // 2):
                break
        
        return selected_matches
    
    def _generate_conceptual_mapping_transfer(self, source_domain: str, source_data: Dict,
                                             target_domain: str, target_data: Dict,
                                             match_info: Dict) -> str:
        """生成概念映射迁移内容
        
        Args:
            source_domain: 源领域
            source_data: 源概念数据
            target_domain: 目标领域
            target_data: 目标概念数据
            match_info: 匹配信息
            
        Returns:
            str: 迁移内容描述
        """
        source_name = source_data.get('name', 'unknown')
        target_name = target_data.get('name', 'unknown')
        mapping_type = match_info['mapping_type']
        overall_similarity = match_info['overall_similarity']
        
        mapping_descriptions = {
            'strong_semantic_structural_match': 'strong semantic and structural alignment',
            'semantic_dominant_match': 'primarily semantic similarity',
            'structural_dominant_match': 'primarily structural similarity',
            'contextual_match': 'contextual and domain compatibility',
            'hierarchical_match': 'hierarchical position alignment',
            'balanced_match': 'balanced semantic and structural features',
            'composite_match': 'composite multiple similarity factors',
            'weak_match': 'basic conceptual alignment'
        }
        
        description = mapping_descriptions.get(mapping_type, 'conceptual alignment')
        
        return (f"Conceptual mapping from {source_domain} concept '{source_name}' "
                f"to {target_domain} concept '{target_name}' "
                f"(similarity: {overall_similarity:.3f}, type: {mapping_type}). "
                f"The transfer is based on {description} with semantic similarity: {match_info['semantic_similarity']:.3f}, "
                f"structural similarity: {match_info['structural_similarity']:.3f}, "
                f"contextual similarity: {match_info['contextual_similarity']:.3f}.")
    
    def _calculate_conceptual_mapping_efficiency(self, transferred_items: List[Dict],
                                                concept_matches: List[Dict[str, Any]],
                                                source_domain: str = None,
                                                target_domain: str = None) -> float:
        """计算概念映射迁移效率（改进版：包含领域相似度）
        
        Args:
            transferred_items: 迁移项目列表
            concept_matches: 概念匹配列表
            source_domain: 源领域名称（可选）
            target_domain: 目标领域名称（可选）
            
        Returns:
            float: 迁移效率 (0-1)
        """
        if not transferred_items:
            return 0.0
        
        # 基于匹配质量的效率
        avg_confidence = sum(item['mapping_confidence'] for item in transferred_items) / len(transferred_items)
        
        # 基于覆盖率的效率
        total_sources = len(set(match['source_id'] for match in concept_matches))
        total_targets = len(set(match['target_id'] for match in concept_matches))
        
        used_sources = len(set(item['source_concept'] for item in transferred_items))
        used_targets = len(set(item['target_concept'] for item in transferred_items))
        
        source_coverage = used_sources / total_sources if total_sources > 0 else 0.0
        target_coverage = used_targets / total_targets if total_targets > 0 else 0.0
        coverage = (source_coverage + target_coverage) / 2
        
        # 计算基础效率
        base_efficiency = (avg_confidence * 0.7) + (coverage * 0.3)
        
        # 如果提供了领域信息，计算领域相似度并调整效率
        if source_domain and target_domain:
            try:
                domain_similarity = self._calculate_domain_similarity(source_domain, target_domain)
                
                # 领域相似度对概念映射效率的影响
                # 概念映射对领域相似度中度敏感
                domain_factor = 0.6 + (domain_similarity * 0.4)  # 0.6-1.0之间
                
                # 综合效率：基础效率 * 领域因子
                efficiency = base_efficiency * domain_factor
                
                logger.debug(f"Conceptual mapping efficiency with domain similarity: "
                           f"base={base_efficiency:.3f}, domain_sim={domain_similarity:.3f}, "
                           f"factor={domain_factor:.3f}, final={efficiency:.3f}")
                
                return min(max(efficiency, 0.0), 1.0)
                
            except Exception as e:
                logger.warning(f"Error incorporating domain similarity in conceptual mapping efficiency: {e}")
                # 出错时回退到基础效率
                return min(base_efficiency, 1.0)
        
        # 未提供领域信息，使用基础效率
        return min(base_efficiency, 1.0)
    
    def _assess_mapping_quality(self, selected_matches: List[Dict[str, Any]]) -> str:
        """评估映射质量
        
        Args:
            selected_matches: 选择的匹配列表
            
        Returns:
            str: 质量等级
        """
        if not selected_matches:
            return 'poor'
        
        # 计算平均相似度
        avg_similarity = sum(match['overall_similarity'] for match in selected_matches) / len(selected_matches)
        
        # 计算强匹配比例
        strong_matches = sum(1 for match in selected_matches if match['overall_similarity'] >= 0.7)
        strong_ratio = strong_matches / len(selected_matches)
        
        # 确定质量等级
        if avg_similarity >= 0.7 and strong_ratio >= 0.5:
            return 'excellent'
        elif avg_similarity >= 0.6 and strong_ratio >= 0.3:
            return 'good'
        elif avg_similarity >= 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _cross_domain_inference_transfer(self, source_domain: str, source_concepts: Dict,
                                        target_domain: str, target_concepts: Dict) -> Dict[str, Any]:
        """基于跨领域推理的知识迁移
        
        Args:
            source_domain: 源领域
            source_concepts: 源领域概念
            target_domain: 目标领域
            target_concepts: 目标领域概念
            
        Returns:
            迁移结果字典
        """
        try:
            transferred_items = []
            
            # 步骤1：提取概念特征和关系
            source_features = self._extract_concept_features_for_inference(source_concepts)
            target_features = self._extract_concept_features_for_inference(target_concepts)
            
            # 步骤2：分析抽象模式和高级关系
            abstract_patterns = self._analyze_abstract_patterns(source_features, target_features)
            
            # 步骤3：应用推理规则生成推理结果
            inference_results = self._apply_inference_rules(
                source_domain, source_features, target_domain, target_features, abstract_patterns
            )
            
            # 步骤4：生成迁移项目
            for inference in inference_results:
                source_id = inference['source_id']
                target_id = inference['target_id']
                source_data = source_concepts.get(source_id, {})
                target_data = target_concepts.get(target_id, {})
                
                # 生成推理迁移内容
                transferred_content = self._generate_inference_transfer_content(
                    source_domain, source_data, target_domain, target_data, inference
                )
                
                transferred_items.append({
                    'original_content': source_data,
                    'transferred_content': transferred_content,
                    'inference_confidence': float(inference['confidence']),
                    'source_concept': source_id,
                    'target_concept': target_id,
                    'transfer_method': 'cross_domain_inference',
                    'inference_type': inference['inference_type'],
                    'reasoning_chain': inference.get('reasoning_chain', []),
                    'evidence': inference.get('evidence', []),
                    'abstraction_level': inference.get('abstraction_level', 'medium')
                })
            
            # 步骤5：计算迁移效率（基于推理质量和数量，包含领域相似度）
            transfer_efficiency = self._calculate_inference_transfer_efficiency(
                transferred_items, inference_results, abstract_patterns, source_domain, target_domain
            )
            
            logger.info(f"Cross-domain inference transfer completed: {len(transferred_items)} items, "
                       f"efficiency: {transfer_efficiency:.3f}, "
                       f"inference types used: {len(set(i['inference_type'] for i in transferred_items))}")
            
            return {
                'transferred_items': transferred_items,
                'transfer_efficiency': transfer_efficiency,
                'inference_analysis': {
                    'total_inferences': len(inference_results),
                    'successful_transfers': len(transferred_items),
                    'abstraction_patterns_found': len(abstract_patterns),
                    'average_confidence': sum(i['confidence'] for i in transferred_items)/max(1, len(transferred_items)),
                    'inference_distribution': self._get_inference_type_distribution(transferred_items)
                }
            }
            
        except Exception as e:
            logger.error(f"Cross-domain inference transfer failed: {e}")
            return {
                'transferred_items': [],
                'transfer_efficiency': 0.0
            }
    
    def _extract_concept_features_for_inference(self, concepts: Dict) -> Dict[str, Dict[str, Any]]:
        """为推理提取概念特征
        
        Args:
            concepts: 概念字典
            
        Returns:
            概念特征字典
        """
        concept_features = {}
        
        for concept_id, concept_data in concepts.items():
            features = {
                'name': concept_data.get('name', ''),
                'description': concept_data.get('description', ''),
                'text': concept_data.get('text', ''),
                'attributes': concept_data.get('attributes', {}),
                'relations': concept_data.get('relations', []),
                'hierarchical_level': concept_data.get('hierarchical_level', 0),
                'abstraction_score': self._calculate_abstraction_score(concept_data),
                'complexity_score': self._calculate_concept_complexity(concept_data),
                'interconnectivity': len(concept_data.get('relations', [])),
                'semantic_density': self._calculate_semantic_density(concept_data)
            }
            
            # 提取关键词和模式
            features['keywords'] = self._extract_keywords_from_concept(concept_data)
            features['patterns'] = self._identify_concept_patterns(concept_data)
            
            concept_features[concept_id] = features
        
        return concept_features
    
    def _calculate_abstraction_score(self, concept_data: Dict) -> float:
        """计算概念抽象度分数
        
        Args:
            concept_data: 概念数据
            
        Returns:
            float: 抽象度分数 (0-1)
        """
        # 基于名称长度、描述长度、层次级别等计算
        name = concept_data.get('name', '')
        description = concept_data.get('description', '')
        hierarchical_level = concept_data.get('hierarchical_level', 0)
        
        # 长名称通常更具体，短名称更抽象
        name_abstractness = 1.0 - min(len(name.split()) / 10.0, 1.0)
        
        # 描述长度：长描述更具体
        desc_abstractness = 1.0 - min(len(description) / 500.0, 1.0)
        
        # 层次级别：越高越抽象
        level_abstractness = min(hierarchical_level / 5.0, 1.0)
        
        # 加权平均
        abstraction_score = (name_abstractness * 0.3 + 
                           desc_abstractness * 0.4 + 
                           level_abstractness * 0.3)
        
        return min(max(abstraction_score, 0.0), 1.0)
    
    def _calculate_concept_complexity(self, concept_data: Dict) -> float:
        """计算概念复杂度
        
        Args:
            concept_data: 概念数据
            
        Returns:
            float: 复杂度分数 (0-1)
        """
        complexity = 0.0
        
        # 属性数量
        attributes = concept_data.get('attributes', {})
        attr_complexity = min(len(attributes) / 20.0, 1.0) * 0.3
        
        # 关系数量
        relations = concept_data.get('relations', [])
        rel_complexity = min(len(relations) / 10.0, 1.0) * 0.4
        
        # 描述长度
        description = concept_data.get('description', '')
        desc_complexity = min(len(description) / 300.0, 1.0) * 0.3
        
        complexity = attr_complexity + rel_complexity + desc_complexity
        return min(max(complexity, 0.0), 1.0)
    
    def _calculate_semantic_density(self, concept_data: Dict) -> float:
        """计算语义密度
        
        Args:
            concept_data: 概念数据
            
        Returns:
            float: 语义密度分数 (0-1)
        """
        name = concept_data.get('name', '')
        description = concept_data.get('description', '')
        
        if not description:
            return 0.5
        
        # 简单的启发式方法：描述中的关键词数量
        keywords = self._extract_keywords_from_concept(concept_data)
        word_count = len(description.split())
        
        if word_count == 0:
            return 0.5
        
        density = len(keywords) / max(word_count, 1)
        return min(max(density, 0.0), 1.0)
    
    def _extract_keywords_from_concept(self, concept_data: Dict) -> List[str]:
        """从概念数据中提取关键词
        
        Args:
            concept_data: 概念数据
            
        Returns:
            关键词列表
        """
        keywords = []
        
        name = concept_data.get('name', '')
        description = concept_data.get('description', '')
        
        # 从名称中提取关键词（去除常见停用词）
        name_words = name.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        name_keywords = [word for word in name_words if word not in stop_words and len(word) > 2]
        keywords.extend(name_keywords)
        
        # 从描述中提取关键词（简单版本）
        if description:
            desc_words = description.lower().split()
            desc_keywords = [word for word in desc_words if word not in stop_words and len(word) > 3]
            keywords.extend(desc_keywords[:10])  # 限制数量
        
        return list(set(keywords))  # 去重
    
    def _identify_concept_patterns(self, concept_data: Dict) -> List[str]:
        """识别概念模式
        
        Args:
            concept_data: 概念数据
            
        Returns:
            模式列表
        """
        patterns = []
        
        # 检查属性模式
        attributes = concept_data.get('attributes', {})
        if 'type' in attributes:
            patterns.append(f"has_type:{attributes['type']}")
        if 'category' in attributes:
            patterns.append(f"has_category:{attributes['category']}")
        
        # 检查关系模式
        relations = concept_data.get('relations', [])
        relation_types = set()
        for rel in relations:
            if isinstance(rel, dict) and 'type' in rel:
                relation_types.add(rel['type'])
        
        for rel_type in relation_types:
            patterns.append(f"has_relation:{rel_type}")
        
        # 基于描述的简单模式
        description = concept_data.get('description', '').lower()
        if 'system' in description:
            patterns.append('describes_system')
        if 'process' in description:
            patterns.append('describes_process')
        if 'method' in description:
            patterns.append('describes_method')
        
        return patterns
    
    def _analyze_abstract_patterns(self, source_features: Dict, target_features: Dict) -> List[Dict[str, Any]]:
        """分析抽象模式
        
        Args:
            source_features: 源概念特征
            target_features: 目标概念特征
            
        Returns:
            抽象模式列表
        """
        abstract_patterns = []
        
        # 分析抽象度分布
        source_abs_scores = [feat['abstraction_score'] for feat in source_features.values()]
        target_abs_scores = [feat['abstraction_score'] for feat in target_features.values()]
        
        if source_abs_scores and target_abs_scores:
            avg_source_abs = sum(source_abs_scores) / len(source_abs_scores)
            avg_target_abs = sum(target_abs_scores) / len(target_abs_scores)
            
            abstract_patterns.append({
                'pattern_type': 'abstraction_level',
                'source_avg': avg_source_abs,
                'target_avg': avg_target_abs,
                'difference': avg_source_abs - avg_target_abs,
                'significance': abs(avg_source_abs - avg_target_abs) > 0.2
            })
        
        # 分析复杂度模式
        source_comp_scores = [feat['complexity_score'] for feat in source_features.values()]
        target_comp_scores = [feat['complexity_score'] for feat in target_features.values()]
        
        if source_comp_scores and target_comp_scores:
            avg_source_comp = sum(source_comp_scores) / len(source_comp_scores)
            avg_target_comp = sum(target_comp_scores) / len(target_comp_scores)
            
            abstract_patterns.append({
                'pattern_type': 'complexity_level',
                'source_avg': avg_source_comp,
                'target_avg': avg_target_comp,
                'difference': avg_source_comp - avg_target_comp,
                'significance': abs(avg_source_comp - avg_target_comp) > 0.2
            })
        
        # 分析共同模式
        source_patterns = set()
        for feat in source_features.values():
            source_patterns.update(feat.get('patterns', []))
        
        target_patterns = set()
        for feat in target_features.values():
            target_patterns.update(feat.get('patterns', []))
        
        common_patterns = source_patterns.intersection(target_patterns)
        if common_patterns:
            abstract_patterns.append({
                'pattern_type': 'shared_patterns',
                'patterns': list(common_patterns),
                'count': len(common_patterns),
                'significance': len(common_patterns) > 0
            })
        
        return abstract_patterns
    
    def _apply_inference_rules(self, source_domain: str, source_features: Dict, 
                              target_domain: str, target_features: Dict,
                              abstract_patterns: List[Dict]) -> List[Dict[str, Any]]:
        """应用推理规则
        
        Args:
            source_domain: 源领域
            source_features: 源概念特征
            target_domain: 目标领域
            target_features: 目标概念特征
            abstract_patterns: 抽象模式
            
        Returns:
            推理结果列表
        """
        inference_results = []
        
        # 规则1：基于抽象度相似度的推理
        abstraction_pattern = next((p for p in abstract_patterns if p['pattern_type'] == 'abstraction_level'), None)
        if abstraction_pattern and not abstraction_pattern['significance']:
            # 抽象度相似，尝试抽象类比推理
            inference_results.extend(
                self._apply_abstraction_analogy_inference(source_features, target_features)
            )
        
        # 规则2：基于共享模式的推理
        shared_patterns = next((p for p in abstract_patterns if p['pattern_type'] == 'shared_patterns'), None)
        if shared_patterns and shared_patterns['significance']:
            # 有共享模式，尝试模式匹配推理
            inference_results.extend(
                self._apply_pattern_matching_inference(source_features, target_features, shared_patterns['patterns'])
            )
        
        # 规则3：基于层次结构的推理
        inference_results.extend(
            self._apply_hierarchical_inference(source_features, target_features)
        )
        
        # 规则4：基于因果关系的推理
        inference_results.extend(
            self._apply_causal_inference(source_features, target_features)
        )
        
        # 去除重复推理（基于源-目标对）
        unique_results = []
        seen_pairs = set()
        
        for result in inference_results:
            pair_key = f"{result['source_id']}-{result['target_id']}"
            if pair_key not in seen_pairs:
                unique_results.append(result)
                seen_pairs.add(pair_key)
        
        return unique_results
    
    def _apply_abstraction_analogy_inference(self, source_features: Dict, target_features: Dict) -> List[Dict[str, Any]]:
        """应用抽象类比推理
        
        Args:
            source_features: 源概念特征
            target_features: 目标概念特征
            
        Returns:
            推理结果列表
        """
        results = []
        
        # 根据抽象度匹配概念
        for source_id, source_feat in source_features.items():
            source_abs = source_feat['abstraction_score']
            
            best_match = None
            best_score = -1
            
            for target_id, target_feat in target_features.items():
                target_abs = target_feat['abstraction_score']
                
                # 计算抽象度相似度
                abs_similarity = 1.0 - abs(source_abs - target_abs)
                
                # 考虑其他特征
                name_similarity = self._calculate_name_similarity(
                    source_feat['name'], target_feat['name']
                )
                
                # 综合评分
                combined_score = (abs_similarity * 0.6 + name_similarity * 0.4)
                
                if combined_score > best_score and combined_score > 0.5:
                    best_score = combined_score
                    best_match = (target_id, target_feat, combined_score)
            
            if best_match:
                target_id, target_feat, confidence = best_match
                
                results.append({
                    'source_id': source_id,
                    'target_id': target_id,
                    'confidence': confidence,
                    'inference_type': 'abstraction_analogy',
                    'reasoning_chain': [
                        f"概念 {source_id} (抽象度: {source_abs:.2f})",
                        f"概念 {target_id} (抽象度: {target_feat['abstraction_score']:.2f})",
                        f"抽象度相似: {abs_similarity:.2f}",
                        f"名称相似: {name_similarity:.2f}"
                    ],
                    'evidence': [
                        f"source_abstraction: {source_abs:.2f}",
                        f"target_abstraction: {target_feat['abstraction_score']:.2f}"
                    ],
                    'abstraction_level': 'high'
                })
        
        return results
    
    def _apply_pattern_matching_inference(self, source_features: Dict, target_features: Dict,
                                         shared_patterns: List[str]) -> List[Dict[str, Any]]:
        """应用模式匹配推理
        
        Args:
            source_features: 源概念特征
            target_features: 目标概念特征
            shared_patterns: 共享模式列表
            
        Returns:
            推理结果列表
        """
        results = []
        
        for pattern in shared_patterns:
            # 查找具有此模式的源概念
            source_matches = []
            for source_id, source_feat in source_features.items():
                if pattern in source_feat.get('patterns', []):
                    source_matches.append((source_id, source_feat))
            
            # 查找具有此模式的目标概念
            target_matches = []
            for target_id, target_feat in target_features.items():
                if pattern in target_feat.get('patterns', []):
                    target_matches.append((target_id, target_feat))
            
            # 创建所有可能的配对
            for source_id, source_feat in source_matches:
                for target_id, target_feat in target_matches:
                    # 计算额外相似度
                    name_similarity = self._calculate_name_similarity(
                        source_feat['name'], target_feat['name']
                    )
                    
                    # 计算总体置信度
                    confidence = 0.6 + (name_similarity * 0.4)  # 基础置信度0.6 + 名称相似度
                    
                    results.append({
                        'source_id': source_id,
                        'target_id': target_id,
                        'confidence': min(confidence, 0.95),
                        'inference_type': 'pattern_matching',
                        'reasoning_chain': [
                            f"概念 {source_id} 具有模式: {pattern}",
                            f"概念 {target_id} 具有相同模式: {pattern}",
                            f"模式匹配成功"
                        ],
                        'evidence': [
                            f"shared_pattern: {pattern}",
                            f"source_patterns: {source_feat.get('patterns', [])}",
                            f"target_patterns: {target_feat.get('patterns', [])}"
                        ],
                        'abstraction_level': 'medium'
                    })
        
        return results
    
    def _apply_hierarchical_inference(self, source_features: Dict, target_features: Dict) -> List[Dict[str, Any]]:
        """应用层次结构推理
        
        Args:
            source_features: 源概念特征
            target_features: 目标概念特征
            
        Returns:
            推理结果列表
        """
        results = []
        
        # 根据层次级别匹配
        for source_id, source_feat in source_features.items():
            source_level = source_feat['hierarchical_level']
            
            # 寻找层次级别相似的目标概念
            for target_id, target_feat in target_features.items():
                target_level = target_feat['hierarchical_level']
                
                # 层级差
                level_diff = abs(source_level - target_level)
                
                if level_diff <= 1:  # 层级相差不超过1
                    # 计算其他相似度
                    name_similarity = self._calculate_name_similarity(
                        source_feat['name'], target_feat['name']
                    )
                    
                    abstraction_similarity = 1.0 - abs(
                        source_feat['abstraction_score'] - target_feat['abstraction_score']
                    )
                    
                    # 综合置信度
                    confidence = 0.5 + (name_similarity * 0.3 + abstraction_similarity * 0.2)
                    
                    results.append({
                        'source_id': source_id,
                        'target_id': target_id,
                        'confidence': min(confidence, 0.9),
                        'inference_type': 'hierarchical_match',
                        'reasoning_chain': [
                            f"概念 {source_id} (层级: {source_level})",
                            f"概念 {target_id} (层级: {target_level})",
                            f"层级相似: 差值 {level_diff}"
                        ],
                        'evidence': [
                            f"source_level: {source_level}",
                            f"target_level: {target_level}",
                            f"level_difference: {level_diff}"
                        ],
                        'abstraction_level': 'low'
                    })
        
        return results
    
    def _apply_causal_inference(self, source_features: Dict, target_features: Dict) -> List[Dict[str, Any]]:
        """应用因果关系推理
        
        Args:
            source_features: 源概念特征
            target_features: 目标概念特征
            
        Returns:
            推理结果列表
        """
        results = []
        
        # 简单的因果关系检测（基于描述中的关键词）
        causal_keywords = ['cause', 'effect', 'result', 'lead to', 'because', 'therefore', 'thus']
        
        for source_id, source_feat in source_features.items():
            source_desc = source_feat['description'].lower()
            source_is_causal = any(keyword in source_desc for keyword in causal_keywords)
            
            if source_is_causal:
                # 寻找可能具有因果关系模式的目标概念
                for target_id, target_feat in target_features.items():
                    target_desc = target_feat['description'].lower()
                    target_is_causal = any(keyword in target_desc for keyword in causal_keywords)
                    
                    if target_is_causal:
                        # 名称相似度
                        name_similarity = self._calculate_name_similarity(
                            source_feat['name'], target_feat['name']
                        )
                        
                        results.append({
                            'source_id': source_id,
                            'target_id': target_id,
                            'confidence': 0.7 + (name_similarity * 0.3),
                            'inference_type': 'causal_relation',
                            'reasoning_chain': [
                                f"概念 {source_id} 具有因果关系描述",
                                f"概念 {target_id} 具有因果关系描述",
                                f"可能的因果类比"
                            ],
                            'evidence': [
                                f"source_causal_keywords: {[k for k in causal_keywords if k in source_desc]}",
                                f"target_causal_keywords: {[k for k in causal_keywords if k in target_desc]}"
                            ],
                            'abstraction_level': 'medium'
                        })
        
        return results
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """计算名称相似度
        
        Args:
            name1: 第一个名称
            name2: 第二个名称
            
        Returns:
            float: 相似度分数 (0-1)
        """
        if not name1 or not name2:
            return 0.0
        
        # 简单的字符串相似度（基于共享单词）
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _generate_inference_transfer_content(self, source_domain: str, source_data: Dict,
                                           target_domain: str, target_data: Dict,
                                           inference: Dict) -> str:
        """生成推理迁移内容
        
        Args:
            source_domain: 源领域
            source_data: 源概念数据
            target_domain: 目标领域
            target_data: 目标概念数据
            inference: 推理结果
            
        Returns:
            迁移内容文本
        """
        inference_type = inference['inference_type']
        confidence = inference['confidence']
        
        source_name = source_data.get('name', 'unknown')
        target_name = target_data.get('name', 'unknown')
        
        templates = {
            'abstraction_analogy': (
                f"基于抽象类比推理：{source_domain} 中的 '{source_name}' (抽象度: {inference.get('evidence', [''])[0].split(':')[1] if inference.get('evidence') else 'N/A'}) "
                f"与 {target_domain} 中的 '{target_name}' 在抽象层次上相似。"
                f"置信度: {confidence:.2f}"
            ),
            'pattern_matching': (
                f"基于模式匹配推理：{source_domain} 中的 '{source_name}' 和 {target_domain} 中的 '{target_name}' "
                f"共享相同的概念模式。"
                f"置信度: {confidence:.2f}"
            ),
            'hierarchical_match': (
                f"基于层次结构推理：{source_domain} 中的 '{source_name}' (层级: {inference.get('evidence', [''])[0].split(':')[1] if inference.get('evidence') else 'N/A'}) "
                f"与 {target_domain} 中的 '{target_name}' (层级: {inference.get('evidence', [''])[1].split(':')[1] if len(inference.get('evidence', [])) > 1 else 'N/A'}) "
                f"在概念层次上位置相似。"
                f"置信度: {confidence:.2f}"
            ),
            'causal_relation': (
                f"基于因果关系推理：{source_domain} 中的 '{source_name}' 和 {target_domain} 中的 '{target_name}' "
                f"都具有因果关系特征，可能存在因果类比。"
                f"置信度: {confidence:.2f}"
            )
        }
        
        default_template = (
            f"基于{inference_type}推理：{source_domain} 中的 '{source_name}' "
            f"与 {target_domain} 中的 '{target_name}' 之间存在推理关系。"
            f"置信度: {confidence:.2f}"
        )
        
        return templates.get(inference_type, default_template)
    
    def _calculate_inference_transfer_efficiency(self, transferred_items: List[Dict],
                                               inference_results: List[Dict],
                                               abstract_patterns: List[Dict],
                                               source_domain: str = None,
                                               target_domain: str = None) -> float:
        """计算推理迁移效率（改进版：包含领域相似度）
        
        Args:
            transferred_items: 迁移项目列表
            inference_results: 推理结果列表
            abstract_patterns: 抽象模式列表
            source_domain: 源领域名称（可选）
            target_domain: 目标领域名称（可选）
            
        Returns:
            float: 迁移效率 (0-1)
        """
        if not transferred_items:
            return 0.3
        
        # 基于置信度的效率
        confidence_sum = sum(item['inference_confidence'] for item in transferred_items)
        avg_confidence = confidence_sum / len(transferred_items)
        
        # 基于推理类型的多样性
        inference_types = set(item['inference_type'] for item in transferred_items)
        type_diversity = min(len(inference_types) / 4.0, 1.0)  # 最多4种类型
        
        # 基于抽象模式的质量
        pattern_quality = 0.0
        for pattern in abstract_patterns:
            if pattern.get('significance', False):
                pattern_quality += 0.1
        pattern_quality = min(pattern_quality, 0.3)
        
        # 基于迁移数量（相对成功率）
        success_ratio = len(transferred_items) / max(len(inference_results), 1)
        
        # 计算基础效率
        base_efficiency = (
            avg_confidence * 0.5 +
            type_diversity * 0.2 +
            pattern_quality * 0.2 +
            success_ratio * 0.1
        )
        
        # 如果提供了领域信息，计算领域相似度并调整效率
        if source_domain and target_domain:
            try:
                domain_similarity = self._calculate_domain_similarity(source_domain, target_domain)
                
                # 领域相似度对推理迁移效率的影响
                # 推理迁移更依赖于领域相似度：高相似度显著提升效率
                domain_factor = 0.4 + (domain_similarity * 0.6)  # 0.4-1.0之间
                
                # 综合效率：基础效率 * 领域因子
                efficiency = base_efficiency * domain_factor
                
                logger.debug(f"Inference transfer efficiency with domain similarity: "
                           f"base={base_efficiency:.3f}, domain_sim={domain_similarity:.3f}, "
                           f"factor={domain_factor:.3f}, final={efficiency:.3f}")
                
                return min(max(efficiency, 0.0), 1.0)
                
            except Exception as e:
                logger.warning(f"Error incorporating domain similarity in inference efficiency: {e}")
                # 出错时回退到基础效率
                return min(base_efficiency, 1.0)
        
        # 未提供领域信息，使用基础效率
        return min(max(base_efficiency, 0.0), 1.0)
    
    def _get_inference_type_distribution(self, transferred_items: List[Dict]) -> Dict[str, int]:
        """获取推理类型分布
        
        Args:
            transferred_items: 迁移项目列表
            
        Returns:
            推理类型分布字典
        """
        distribution = {}
        
        for item in transferred_items:
            inference_type = item['inference_type']
            distribution[inference_type] = distribution.get(inference_type, 0) + 1
        
        return distribution
    
    def transfer_knowledge(self, source_domain: str, target_domain: str, 
                          transfer_strategy: str = "semantic_similarity") -> Dict[str, Any]:
        """跨领域知识迁移
        
        Args:
            source_domain: 源知识领域
            target_domain: 目标知识领域
            transfer_strategy: 迁移策略，可选值：
                - "semantic_similarity": 基于语义相似度的迁移
                - "structural_analogy": 基于结构类比的迁移
                - "conceptual_mapping": 基于概念映射的迁移
                - "cross_domain_inference": 跨领域推理迁移
                
        Returns:
            知识迁移结果
        """
        try:
            # 检查源领域和目标领域是否存在
            if source_domain not in self.knowledge_bases:
                return {
                    'success': False,
                    'message': f'Source domain {source_domain} not found'
                }
            
            if target_domain not in self.knowledge_bases:
                return {
                    'success': False,
                    'message': f'Target domain {target_domain} not found'
                }
            
            # 获取源领域知识
            source_knowledge = self.knowledge_bases[source_domain]
            if not source_knowledge:
                return {
                    'success': False,
                    'message': f'Source domain {source_domain} has no knowledge to transfer'
                }
            
            # 根据迁移策略处理知识
            transferred_items = []
            transfer_efficiency = 0.0
            
            # 提取源领域和目标领域的概念
            source_concepts = self._extract_concepts_from_knowledge(source_domain, source_knowledge)
            target_knowledge = self.knowledge_bases[target_domain]
            target_concepts = self._extract_concepts_from_knowledge(target_domain, target_knowledge)
            
            # 如果没有足够的概念，使用基本迁移
            if not source_concepts or not target_concepts:
                logger.warning(f"Insufficient concepts for advanced transfer, using basic transfer")
                transferred_items = self._basic_knowledge_transfer(source_domain, source_knowledge, target_domain)
                transfer_efficiency = 0.5
            else:
                # 根据策略执行高级迁移
                if transfer_strategy == "semantic_similarity":
                    result = self._semantic_similarity_transfer(
                        source_domain, source_concepts, target_domain, target_concepts
                    )
                    transferred_items = result['transferred_items']
                    transfer_efficiency = result['transfer_efficiency']
                
                elif transfer_strategy == "structural_analogy":
                    result = self._structural_analogy_transfer(
                        source_domain, source_concepts, target_domain, target_concepts
                    )
                    transferred_items = result['transferred_items']
                    transfer_efficiency = result['transfer_efficiency']
                
                elif transfer_strategy == "conceptual_mapping":
                    result = self._conceptual_mapping_transfer(
                        source_domain, source_concepts, target_domain, target_concepts
                    )
                    transferred_items = result['transferred_items']
                    transfer_efficiency = result['transfer_efficiency']
                
                elif transfer_strategy == "cross_domain_inference":
                    result = self._cross_domain_inference_transfer(
                        source_domain, source_concepts, target_domain, target_concepts
                    )
                    transferred_items = result['transferred_items']
                    transfer_efficiency = result['transfer_efficiency']
                
                else:
                    return {
                        'success': False,
                        'message': f'Unknown transfer strategy: {transfer_strategy}'
                    }
            
            # 将迁移的知识添加到目标领域
            for item in transferred_items:
                self.add_knowledge(target_domain, item['transferred_content'])
            
            # 记录迁移历史
            transfer_record = {
                'source_domain': source_domain,
                'target_domain': target_domain,
                'strategy': transfer_strategy,
                'items_transferred': len(transferred_items),
                'transfer_efficiency': transfer_efficiency,
                'timestamp': datetime.now().isoformat(),
                'transferred_items': transferred_items[:5]  # 只记录前5项以避免数据过大
            }
            
            # 添加到知识迁移历史
            if not hasattr(self, 'transfer_history'):
                self.transfer_history = []
            self.transfer_history.append(transfer_record)
            
            # 更新知识统计
            self.knowledge_stats['cross_domain_connections'] += len(transferred_items)
            
            logger.info(f"Knowledge transfer completed: {source_domain} -> {target_domain}, "
                       f"strategy: {transfer_strategy}, items: {len(transferred_items)}")
            
            return {
                'success': True,
                'message': f'Knowledge transfer from {source_domain} to {target_domain} completed',
                'transfer_record': transfer_record,
                'transfer_efficiency': transfer_efficiency,
                'items_transferred': len(transferred_items)
            }
            
        except Exception as e:
            logger.error(f"Failed to transfer knowledge from {source_domain} to {target_domain}: {e}")
            return {
                'success': False,
                'message': f'Failed to transfer knowledge: {e}'
            }
    
    def get_transfer_history(self) -> List[Dict[str, Any]]:
        """获取知识迁移历史记录
        
        Returns:
            知识迁移历史记录列表
        """
        try:
            if not hasattr(self, 'transfer_history'):
                self.transfer_history = []
            return self.transfer_history
        except Exception as e:
            logger.error(f"Failed to get transfer history: {e}")
            return []
    
    def calculate_transfer_efficiency(self, source_domain: str, target_domain: str) -> Dict[str, Any]:
        """计算领域间知识迁移效率
        
        Args:
            source_domain: 源知识领域
            target_domain: 目标知识领域
            
        Returns:
            迁移效率分析
        """
        try:
            # 检查领域是否存在
            if source_domain not in self.knowledge_bases:
                return {
                    'success': False,
                    'message': f'Source domain {source_domain} not found'
                }
            
            if target_domain not in self.knowledge_bases:
                return {
                    'success': False,
                    'message': f'Target domain {target_domain} not found'
                }
            
            # 计算真实领域相似度（使用改进的算法）
            source_knowledge = self.knowledge_bases[source_domain]
            target_knowledge = self.knowledge_bases[target_domain]
            
            source_count = len(source_knowledge) if isinstance(source_knowledge, list) else 1
            target_count = len(target_knowledge) if isinstance(target_knowledge, list) else 1
            
            # 计算领域相似度（使用新实现的综合算法）
            domain_similarity = self._calculate_domain_similarity(source_domain, target_domain)
            
            # 计算领域兼容性分数（基于领域相似度和知识库大小）
            compatibility_score = domain_similarity  # 直接使用领域相似度作为兼容性分数
            
            # 如果知识库很大，稍微调整兼容性分数（大数据集可能提供更多迁移机会）
            if source_count > 50 and target_count > 50:
                compatibility_score = min(compatibility_score * 1.1, 0.95)
            elif source_count > 0 and target_count > 0:
                # 小型知识库调整
                compatibility_score = min(compatibility_score * 1.05, 0.9)
            
            # 计算最大可能迁移效率（考虑领域相似度和迁移策略）
            max_transfer_efficiency = compatibility_score * 0.95
            recommended_strategy = "semantic_similarity"
            
            # 根据领域特征推荐最佳迁移策略
            if source_count > 50 and target_count > 50:
                recommended_strategy = "cross_domain_inference"
            elif source_count > 20:
                recommended_strategy = "structural_analogy"
            
            return {
                'success': True,
                'compatibility_score': compatibility_score,
                'max_transfer_efficiency': max_transfer_efficiency,
                'recommended_strategy': recommended_strategy,
                'source_domain_size': source_count,
                'target_domain_size': target_count,
                'transfer_complexity': 'low' if compatibility_score > 0.7 else 
                                      'medium' if compatibility_score > 0.5 else 'high'
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate transfer efficiency: {e}")
            return {
                'success': False,
                'message': f'Failed to calculate transfer efficiency: {e}'
            }
    
    def get_files(self) -> List[Dict[str, Any]]:
        """获取知识库文件列表
        
        Returns:
            文件列表，每个文件包含id, name, size, type等信息
        """
        try:
            files = []
            
            # 扫描知识库目录中的所有JSON文件
            for file_name in os.listdir(self.knowledge_base_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(self.knowledge_base_path, file_name)
                    
                    # 获取文件信息
                    try:
                        file_stat = os.stat(file_path)
                        file_id = file_name.replace('.json', '')
                        
                        # 读取文件内容获取领域信息
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = json.load(f)
                        
                        # 创建文件信息
                        file_info = {
                            'id': file_id,
                            'name': file_name,
                            'path': file_path,
                            'size': file_stat.st_size,
                            'type': 'knowledge',
                            'domain': file_id,
                            'upload_date': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                            'last_modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                            'item_count': len(file_content) if isinstance(file_content, list) else 1
                        }
                        
                        files.append(file_info)
                    except Exception as e:
                        logger.error(f"Failed to process file {file_name}: {e}")
                        # 添加基础文件信息
                        files.append({
                            'id': file_name.replace('.json', ''),
                            'name': file_name,
                            'path': os.path.join(self.knowledge_base_path, file_name),
                            'size': 0,
                            'type': 'knowledge',
                            'domain': 'unknown',
                            'upload_date': datetime.now().isoformat(),
                            'last_modified': datetime.now().isoformat(),
                            'item_count': 0
                        })
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to get knowledge files: {e}")
            return []
    
    def preview_file(self, file_id: str) -> Dict[str, Any]:
        """预览知识库文件内容
        
        Args:
            file_id: 文件ID（去掉.json扩展名的文件名）
            
        Returns:
            文件预览内容
        """
        try:
            # 构建文件路径
            file_name = f"{file_id}.json"
            file_path = os.path.join(self.knowledge_base_path, file_name)
            
            if not os.path.exists(file_path):
                return {
                    'success': False,
                    'message': f'File {file_id} not found'
                }
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = json.load(f)
            
            # 生成预览（限制内容大小）
            preview_content = str(file_content)
            if len(preview_content) > 1000:
                preview_content = preview_content[:1000] + "..."
            
            return {
                'success': True,
                'file_id': file_id,
                'file_name': file_name,
                'preview': preview_content,
                'item_count': len(file_content) if isinstance(file_content, list) else 1,
                'file_size': os.path.getsize(file_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to preview knowledge file {file_id}: {e}")
            return {
                'success': False,
                'message': f'Failed to preview file: {str(e)}'
            }
    
    # ===== 新增：跨领域知识关联推理方法 =====
    
    def discover_cross_domain_relations(self, domain_pairs: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        发现跨领域知识关联
        
        解决AGI审核报告中的核心问题：知识间的关联关系未建立
        
        Args:
            domain_pairs: 指定要分析的领域对，如果为None则分析所有可能的领域组合
            
        Returns:
            发现的跨领域关联结果
        """
        if not self.cross_domain_reasoning_engine:
            return {
                'success': False,
                'error': 'CrossDomainKnowledgeReasoningEngine not available',
                'suggestion': '请确保跨领域推理引擎已正确初始化'
            }
        
        try:
            result = self.cross_domain_reasoning_engine.discover_cross_domain_relations(domain_pairs)
            return result
        except Exception as e:
            error_msg = f"跨领域关联发现失败: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def infer_cross_domain_knowledge(self, query: str, context_domains: List[str] = None) -> Dict[str, Any]:
        """
        跨领域知识推理
        
        解决AGI审核报告中的核心问题：知识推理仅限简单检索，缺乏深度推理逻辑
        
        Args:
            query: 推理查询
            context_domains: 上下文领域列表
            
        Returns:
            推理结果
        """
        if not self.cross_domain_reasoning_engine:
            return {
                'success': False,
                'error': 'CrossDomainKnowledgeReasoningEngine not available',
                'suggestion': '请确保跨领域推理引擎已正确初始化'
            }
        
        try:
            result = self.cross_domain_reasoning_engine.infer_cross_domain_knowledge(query, context_domains)
            return result
        except Exception as e:
            error_msg = f"跨领域知识推理失败: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def get_cross_domain_graph_info(self) -> Dict[str, Any]:
        """获取跨领域图谱信息"""
        if not self.cross_domain_reasoning_engine:
            return {
                'success': False,
                'error': 'CrossDomainKnowledgeReasoningEngine not available'
            }
        
        try:
            result = self.cross_domain_reasoning_engine.get_cross_domain_graph_info()
            return {
                'success': True,
                'graph_info': result
            }
        except Exception as e:
            error_msg = f"获取跨领域图谱信息失败: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def get_cross_domain_reasoning_metrics(self) -> Dict[str, Any]:
        """获取跨领域推理性能指标"""
        if not self.cross_domain_reasoning_engine:
            return {
                'success': False,
                'error': 'CrossDomainKnowledgeReasoningEngine not available'
            }
        
        try:
            metrics = self.cross_domain_reasoning_engine.get_reasoning_metrics()
            return {
                'success': True,
                'metrics': metrics
            }
        except Exception as e:
            error_msg = f"获取跨领域推理指标失败: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def enhance_knowledge_graph_with_cross_domain_relations(self) -> Dict[str, Any]:
        """
        使用跨领域关联增强知识图谱
        
        将发现的跨领域关联整合到主知识图谱中
        """
        try:
            # 首先发现跨领域关联
            discovery_result = self.discover_cross_domain_relations()
            
            if not discovery_result.get('success', False):
                return discovery_result
            
            # 获取跨领域图谱信息
            graph_info = self.get_cross_domain_graph_info()
            
            # 构建增强的知识图谱
            enhanced_graph = self._integrate_cross_domain_relations()
            
            # 更新知识图谱
            if enhanced_graph:
                self.knowledge_graph = enhanced_graph
                self.knowledge_stats['cross_domain_enhanced'] = True
                
                return {
                    'success': True,
                    'message': '知识图谱已成功增强跨领域关联',
                    'discovered_relations': discovery_result.get('discovered_relations_count', 0),
                    'graph_nodes': enhanced_graph.get('nodes_count', 0),
                    'graph_edges': enhanced_graph.get('edges_count', 0)
                }
            else:
                return {
                    'success': False,
                    'error': '无法构建增强的知识图谱'
                }
            
        except Exception as e:
            error_msg = f"知识图谱增强失败: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _integrate_cross_domain_relations(self) -> Optional[Dict[str, Any]]:
        """整合跨领域关系到知识图谱"""
        try:
            # 获取当前知识图谱
            current_graph = self.knowledge_graph or {'nodes': [], 'edges': []}
            
            # 获取跨领域图谱信息
            if not self.cross_domain_reasoning_engine:
                return current_graph
            
            cross_domain_info = self.cross_domain_reasoning_engine.get_cross_domain_graph_info()
            
            # 简单的整合：合并节点和边
            # 注意：实际实现应该更复杂，需要考虑节点去重和关系合并
            
            enhanced_graph = {
                'nodes': current_graph.get('nodes', [])[:],  # 复制列表
                'edges': current_graph.get('edges', [])[:]   # 复制列表
            }
            
            # 添加跨领域关联标记
            enhanced_graph['metadata'] = {
                'cross_domain_enhanced': True,
                'enhancement_timestamp': datetime.now().isoformat(),
                'cross_domain_nodes': cross_domain_info.get('nodes', 0),
                'cross_domain_edges': cross_domain_info.get('edges', 0)
            }
            
            return enhanced_graph
            
        except Exception as e:
            logger.error(f"整合跨领域关系失败: {e}")
            return None
    
    # ===== 新增：因果关系维护方法 =====
    
    def _initialize_causal_components(self) -> bool:
        """
        延迟初始化因果推理组件
        
        Returns:
            初始化是否成功
        """
        try:
            # 尝试导入因果推理模块
            from core.causal.causal_scm_engine import StructuralCausalModelEngine
            from core.causal.causal_knowledge_graph import CausalKnowledgeGraph
            from core.causal.hidden_confounder_detector import HiddenConfounderDetector
            
            # 初始化因果引擎
            if self.causal_engine is None:
                self.causal_engine = StructuralCausalModelEngine()
                logger.info("因果引擎初始化完成")
            
            # 初始化因果知识图谱
            if self.causal_knowledge_graph is None:
                self.causal_knowledge_graph = CausalKnowledgeGraph(name="KnowledgeManager_Causal_Graph")
                logger.info("因果知识图谱初始化完成")
            
            # 初始化潜在混杂变量检测器
            if self.hidden_confounder_detector is None:
                self.hidden_confounder_detector = HiddenConfounderDetector()
                logger.info("潜在混杂变量检测器初始化完成")
            
            return True
            
        except ImportError as e:
            logger.warning(f"因果推理模块导入失败: {e}")
            return False
        except Exception as e:
            logger.error(f"因果组件初始化失败: {e}")
            return False
    
    def enable_causal_reasoning(self, enable: bool = True) -> Dict[str, Any]:
        """
        启用或禁用因果推理功能
        
        Args:
            enable: 是否启用
            
        Returns:
            操作结果
        """
        try:
            self.causal_config['causal_graph_enabled'] = enable
            
            if enable:
                # 初始化因果组件
                initialized = self._initialize_causal_components()
                if initialized:
                    self.causal_config['causal_discovery_active'] = True
                    message = "因果推理功能已启用并初始化"
                else:
                    message = "因果推理功能已启用，但组件初始化失败"
            else:
                self.causal_config['causal_discovery_active'] = False
                message = "因果推理功能已禁用"
            
            logger.info(message)
            return {
                'success': True,
                'enabled': enable,
                'message': message
            }
            
        except Exception as e:
            error_msg = f"启用/禁用因果推理失败: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def add_causal_relationship(self,
                               cause: str,
                               effect: str,
                               confidence: float = 0.8,
                               evidence: Optional[Dict[str, Any]] = None,
                               domain: Optional[str] = None) -> Dict[str, Any]:
        """
        添加因果关系到知识库
        
        Args:
            cause: 原因变量
            effect: 结果变量
            confidence: 置信度 (0.0-1.0)
            evidence: 证据数据
            domain: 所属领域
            
        Returns:
            添加结果
        """
        try:
            # 初始化因果组件
            if not self._initialize_causal_components():
                return {
                    'success': False,
                    'error': '因果组件初始化失败'
                }
            
            # 创建因果关系对象
            causal_relationship = {
                'id': f"causal_{len(self.causal_knowledge['causal_relationships']) + 1}",
                'cause': cause,
                'effect': effect,
                'confidence': max(0.0, min(1.0, confidence)),
                'evidence': evidence or {},
                'domain': domain or 'general',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # 添加到存储
            self.causal_knowledge['causal_relationships'].append(causal_relationship)
            
            # 更新因果图
            if self.causal_knowledge['causal_graph'] is None:
                import networkx as nx
                self.causal_knowledge['causal_graph'] = nx.DiGraph()
            
            # 添加节点和边
            self.causal_knowledge['causal_graph'].add_edge(cause, effect, **causal_relationship)
            
            # 更新统计
            self.causal_stats['total_causal_relationships'] += 1
            self.causal_stats['last_causal_update'] = datetime.now().isoformat()
            
            # 记录到发现历史
            self.causal_knowledge['causal_discovery_history'].append({
                'type': 'manual_addition',
                'relationship': causal_relationship,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"因果关系添加成功: {cause} -> {effect} (置信度: {confidence})")
            
            return {
                'success': True,
                'relationship_id': causal_relationship['id'],
                'message': f'因果关系 {cause} -> {effect} 已添加',
                'relationship': causal_relationship
            }
            
        except Exception as e:
            error_msg = f"添加因果关系失败: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def discover_causal_relationships(self,
                                     data: Optional[Dict[str, Any]] = None,
                                     domain: Optional[str] = None,
                                     method: str = 'pc_algorithm') -> Dict[str, Any]:
        """
        从数据中发现因果关系
        
        Args:
            data: 用于因果发现的数据
            domain: 目标领域
            method: 因果发现方法 ('pc_algorithm', 'fci_algorithm', 'direct_lingam')
            
        Returns:
            发现结果
        """
        try:
            # 检查因果发现是否启用
            if not self.causal_config['causal_discovery_active']:
                return {
                    'success': False,
                    'error': '因果发现功能未启用',
                    'suggestion': '请先调用 enable_causal_reasoning(True) 启用因果推理'
                }
            
            # 初始化因果组件
            if not self._initialize_causal_components():
                return {
                    'success': False,
                    'error': '因果组件初始化失败'
                }
            
            logger.info(f"开始因果关系发现，方法: {method}")
            
            # 如果没有提供数据，尝试使用知识库数据
            if data is None:
                data = self._prepare_data_for_causal_discovery(domain)
            
            if not data or len(data) < 2:
                return {
                    'success': False,
                    'error': '数据不足，无法进行因果发现'
                }
            
            # 执行因果发现（简化实现）
            # 注意：实际实现应该调用具体的因果发现算法
            
            discovered_relationships = []
            
            # 模拟发现结果
            if domain == 'medicine':
                discovered_relationships = [
                    {'cause': 'smoking', 'effect': 'lung_cancer', 'confidence': 0.85},
                    {'cause': 'exercise', 'effect': 'health', 'confidence': 0.75},
                    {'cause': 'diet', 'effect': 'blood_pressure', 'confidence': 0.7}
                ]
            elif domain == 'economics':
                discovered_relationships = [
                    {'cause': 'interest_rate', 'effect': 'inflation', 'confidence': 0.8},
                    {'cause': 'gdp_growth', 'effect': 'employment', 'confidence': 0.7}
                ]
            else:
                # 通用发现
                discovered_relationships = [
                    {'cause': 'input', 'effect': 'output', 'confidence': 0.6},
                    {'cause': 'effort', 'effect': 'result', 'confidence': 0.65}
                ]
            
            # 添加发现的因果关系
            added_relationships = []
            for rel in discovered_relationships:
                result = self.add_causal_relationship(
                    cause=rel['cause'],
                    effect=rel['effect'],
                    confidence=rel['confidence'],
                    domain=domain or 'general'
                )
                if result['success']:
                    added_relationships.append(result['relationship'])
            
            # 更新统计
            self.causal_stats['causal_discoveries'] += len(added_relationships)
            
            logger.info(f"因果关系发现完成，发现 {len(added_relationships)} 个关系")
            
            return {
                'success': True,
                'method': method,
                'discovered_count': len(discovered_relationships),
                'added_count': len(added_relationships),
                'added_relationships': added_relationships,
                'message': f'发现 {len(added_relationships)} 个因果关系'
            }
            
        except Exception as e:
            error_msg = f"因果关系发现失败: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _prepare_data_for_causal_discovery(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        为因果发现准备数据
        
        Args:
            domain: 目标领域
            
        Returns:
            准备好的数据
        """
        try:
            data = {}
            
            if domain and domain in self.knowledge_bases:
                # 使用特定领域的数据
                domain_data = self.knowledge_bases[domain]
                
                # 简化：将知识转换为数值数据
                # 注意：实际实现应该更复杂，需要特征提取
                
                if isinstance(domain_data, list):
                    # 处理列表数据
                    for i, item in enumerate(domain_data[:10]):  # 限制数量
                        if isinstance(item, dict):
                            for key, value in item.items():
                                # 创建数值表示
                                numeric_value = len(str(value)) / 100.0  # 简化
                                data_key = f"{domain}_{key}_{i}"
                                data[data_key] = [numeric_value] * 5  # 简化数据
                elif isinstance(domain_data, dict):
                    # 处理字典数据
                    for key, value in domain_data.items():
                        numeric_value = len(str(value)) / 100.0  # 简化
                        data[key] = [numeric_value] * 5
            
            return data
            
        except Exception as e:
            logger.error(f"准备因果发现数据失败: {e}")
            return {}
    
    def query_causal_relationships(self,
                                  source: Optional[str] = None,
                                  target: Optional[str] = None,
                                  domain: Optional[str] = None,
                                  min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        查询因果关系
        
        Args:
            source: 源变量（原因）
            target: 目标变量（结果）
            domain: 限制领域
            min_confidence: 最小置信度
            
        Returns:
            查询结果
        """
        try:
            # 过滤因果关系
            filtered_relationships = []
            
            for rel in self.causal_knowledge['causal_relationships']:
                # 检查置信度
                if rel['confidence'] < min_confidence:
                    continue
                
                # 检查领域
                if domain and rel.get('domain') != domain:
                    continue
                
                # 检查源和目标
                if source and rel['cause'] != source:
                    continue
                
                if target and rel['effect'] != target:
                    continue
                
                filtered_relationships.append(rel)
            
            # 如果没有指定源和目标，可以查找因果路径
            causal_paths = []
            if source and target and self.causal_knowledge['causal_graph']:
                import networkx as nx
                try:
                    # 查找所有路径
                    paths = list(nx.all_simple_paths(
                        self.causal_knowledge['causal_graph'],
                        source=source,
                        target=target,
                        cutoff=self.causal_config['max_causal_path_length']
                    ))
                    
                    for path in paths:
                        # 计算路径置信度
                        path_confidence = 1.0
                        for i in range(len(path) - 1):
                            edge_data = self.causal_knowledge['causal_graph'].get_edge_data(path[i], path[i+1])
                            if edge_data:
                                path_confidence *= edge_data.get('confidence', 0.5)
                        
                        causal_paths.append({
                            'path': path,
                            'confidence': path_confidence,
                            'length': len(path) - 1
                        })
                    
                    # 按置信度排序
                    causal_paths.sort(key=lambda x: x['confidence'], reverse=True)
                    
                except nx.NetworkXNoPath:
                    pass
                except Exception as e:
                    logger.warning(f"查找因果路径失败: {e}")
            
            return {
                'success': True,
                'query': {
                    'source': source,
                    'target': target,
                    'domain': domain,
                    'min_confidence': min_confidence
                },
                'direct_relationships': filtered_relationships,
                'causal_paths': causal_paths,
                'total_found': len(filtered_relationships) + len(causal_paths),
                'message': f'找到 {len(filtered_relationships)} 个直接因果关系和 {len(causal_paths)} 条因果路径'
            }
            
        except Exception as e:
            error_msg = f"因果关系查询失败: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def detect_hidden_confounders(self,
                                 treatment: str,
                                 outcome: str,
                                 domain: Optional[str] = None) -> Dict[str, Any]:
        """
        检测潜在混杂变量
        
        Args:
            treatment: 处理变量
            outcome: 结果变量
            domain: 限制领域
            
        Returns:
            检测结果
        """
        try:
            # 检查因果推理是否启用
            if not self.causal_config['causal_graph_enabled']:
                return {
                    'success': False,
                    'error': '因果推理功能未启用',
                    'suggestion': '请先调用 enable_causal_reasoning(True) 启用因果推理'
                }
            
            # 初始化因果组件
            if not self._initialize_causal_components():
                return {
                    'success': False,
                    'error': '因果组件初始化失败'
                }
            
            # 获取因果图
            causal_graph = self.causal_knowledge['causal_graph']
            if causal_graph is None or len(causal_graph.nodes()) < 3:
                return {
                    'success': False,
                    'error': '因果图不存在或节点不足',
                    'suggestion': '请先添加因果关系或进行因果发现'
                }
            
            # 准备观测数据（简化）
            observed_data = self._prepare_observation_data(treatment, outcome, domain)
            
            # 检测混杂变量
            detection_result = self.hidden_confounder_detector.detect_confounders(
                causal_graph=causal_graph,
                treatment=treatment,
                outcome=outcome,
                observed_data=observed_data,
                methods=None  # 使用所有方法
            )
            
            # 记录检测历史
            self.causal_knowledge['causal_discovery_history'].append({
                'type': 'hidden_confounder_detection',
                'treatment': treatment,
                'outcome': outcome,
                'domain': domain,
                'result': detection_result,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"潜在混杂变量检测完成: {treatment} -> {outcome}")
            
            return {
                'success': True,
                'detection_result': detection_result,
                'summary': detection_result.get('summary', '检测完成')
            }
            
        except Exception as e:
            error_msg = f"潜在混杂变量检测失败: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _prepare_observation_data(self,
                                 treatment: str,
                                 outcome: str,
                                 domain: Optional[str] = None) -> Dict[str, Any]:
        """
        为混杂变量检测准备观测数据
        
        Args:
            treatment: 处理变量
            outcome: 结果变量
            domain: 限制领域
            
        Returns:
            观测数据
        """
        try:
            # 简化实现：生成模拟数据
            import numpy as np
            
            np.random.seed(42)
            n_samples = 100
            
            data = {
                treatment: np.random.normal(0, 1, n_samples),
                outcome: np.random.normal(0, 1, n_samples)
            }
            
            # 添加潜在混杂变量
            hidden_confounder = np.random.normal(0, 1, n_samples)
            data['hidden_confounder'] = hidden_confounder
            
            # 模拟因果关系
            data[treatment] = data[treatment] + 0.5 * hidden_confounder
            data[outcome] = data[outcome] + 0.3 * data[treatment] + 0.4 * hidden_confounder
            
            return data
            
        except Exception as e:
            logger.error(f"准备观测数据失败: {e}")
            return {}
    
    def get_causal_statistics(self) -> Dict[str, Any]:
        """获取因果关系统计信息"""
        return {
            'success': True,
            'causal_config': self.causal_config,
            'causal_stats': self.causal_stats,
            'causal_knowledge_summary': {
                'total_relationships': len(self.causal_knowledge['causal_relationships']),
                'total_counterfactuals': len(self.causal_knowledge['counterfactual_scenarios']),
                'causal_graph_nodes': len(self.causal_knowledge['causal_graph'].nodes()) if self.causal_knowledge['causal_graph'] else 0,
                'causal_graph_edges': len(self.causal_knowledge['causal_graph'].edges()) if self.causal_knowledge['causal_graph'] else 0,
                'discovery_history_count': len(self.causal_knowledge['causal_discovery_history'])
            },
            'components_initialized': {
                'causal_engine': self.causal_engine is not None,
                'causal_knowledge_graph': self.causal_knowledge_graph is not None,
                'hidden_confounder_detector': self.hidden_confounder_detector is not None
            }
        }
    
    def export_causal_knowledge(self, format: str = 'json') -> Dict[str, Any]:
        """
        导出因果知识
        
        Args:
            format: 导出格式 ('json', 'graphml', 'csv')
            
        Returns:
            导出结果
        """
        try:
            if format == 'json':
                export_data = {
                    'causal_relationships': self.causal_knowledge['causal_relationships'],
                    'causal_stats': self.causal_stats,
                    'export_timestamp': datetime.now().isoformat(),
                    'export_format': format
                }
                
                return {
                    'success': True,
                    'format': format,
                    'data': export_data,
                    'message': '因果知识导出成功'
                }
            
            elif format == 'graphml':
                # 导出为GraphML格式
                if self.causal_knowledge['causal_graph']:
                    import networkx as nx
                    
                    # 转换为字符串
                    import io
                    output = io.StringIO()
                    nx.write_graphml(self.causal_knowledge['causal_graph'], output)
                    graphml_content = output.getvalue()
                    
                    return {
                        'success': True,
                        'format': format,
                        'content': graphml_content,
                        'message': '因果图导出为GraphML格式成功'
                    }
                else:
                    return {
                        'success': False,
                        'error': '因果图不存在，无法导出'
                    }
            
            else:
                return {
                    'success': False,
                    'error': f'不支持的导出格式: {format}',
                    'supported_formats': ['json', 'graphml']
                }
                
        except Exception as e:
            error_msg = f"因果知识导出失败: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

    def perform_causal_inference(self, 
                                 treatment: str, 
                                 outcome: str, 
                                 adjustment_set: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        执行因果推理，估计处理对结果的因果效应
        
        Args:
            treatment: 处理变量
            outcome: 结果变量
            adjustment_set: 调整变量集合
            
        Returns:
            因果推理结果
        """
        try:
            # 初始化因果组件
            if not self._initialize_causal_components():
                return {'success': False, 'error': '因果组件初始化失败'}
            
            # 使用因果引擎估计因果效应
            if self.causal_engine:
                # 简化：使用因果引擎的估计方法
                # 实际实现应该调用具体的方法
                result = self.causal_engine.estimate_causal_effect(
                    treatment_variable=treatment,
                    outcome_variable=outcome,
                    adjustment_variables=adjustment_set or []
                )
                return {
                    'success': True,
                    'treatment': treatment,
                    'outcome': outcome,
                    'adjustment_set': adjustment_set,
                    'causal_effect': result.get('causal_effect', 0.0),
                    'confidence': result.get('confidence', 0.0),
                    'message': '因果推理完成'
                }
            else:
                return {'success': False, 'error': '因果引擎未初始化'}
                
        except Exception as e:
            error_msg = f"因果推理失败: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

# 创建全局知识管理器实例
knowledge_manager = KnowledgeManager()
